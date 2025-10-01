"""
Stage 02: Render Robinson-projected JPEGs for aggregated global rasters (gross emissions only),
plus derived composites and 4-panel maps.

This version:
  - Assumes layers are GROSS EMISSIONS (positive-only signal; <=0 masked).
  - Removes duplicate exports (no "for_pres" variant).
  - Automatically renders THREE profiles per layer: asinh, linear, and steps (no log/sqrt).
  - Uses land-only percentile statistics by default for more stable scaling on sparse globals.
  - Adds products:
      (a) drained emissions
      (b) burned emissions
      (c) drained + burned emissions (sum, masked to >0)
      (d) binary drained/undrained peat extent (via reclass of a drained-state raster)
  - Adds 4-panel map composer (global + three AOIs) for drained and burned emissions.

Reclassify drained-state raster as:
    0          -> nodata
    20,000,000 -> nodata
    16,000,000 -> UNDRAINED peat (0)
    all other values -> DRAINED peat (1)

Example usage:

# Read mosaics directly from S3 and upload rendered assets back to S3:
python -m src.scripts.postprocessing.visualization.create_global_displays \
  --date_tag 20250923 --read_from_s3 --run_name ogh_sensitivity_1km \
  --model_version 0_8_0

# Render locally only (no S3 download/upload) into DISPLAY_OUT_ROOT:
python -m src.scripts.postprocessing.visualization.create_global_displays \
  --date_tag 20250923 --run_name ogh_sensitivity_1km --model_version 0_8_0 \
  --local_display_only

# All rendering options
python -m src.scripts.postprocessing.visualization.create_global_displays \
  --date_tag 20250923 --read_from_s3 --run_name ogh_sensitivity_1km --model_version 0_8_0 \
  --ds_drained_emis drained_emissions \
  --ds_burned_emis burned_emissions \
  --ds_drained_state drained_state \
  --profiles asinh steps linear \
  --clip_lo 5 --clip_hi 99.97 \
  --cmap inferno --interpolation bilinear

# 4 panel drained or burned
python -m src.scripts.postprocessing.visualization.create_global_displays \
  --date_tag 20250923 --read_from_s3 --run_name ogh_sensitivity_1km --model_version 0_8_0 \
  --ds_drained_emis drained_emissions \
  --make_four_panel_drained \
  --four_panel_profile asinh \
  --aoi "A:-81.6,27.1,-80.2,28.5" "B:2.0,-2.0,7.0,3.0" "C:71.0,66.5,78.0,69.5"

"""

from __future__ import annotations

import argparse
import posixpath
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform as rio_transform,
)
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import (
    LinearSegmentedColormap,
    Normalize,
    BoundaryNorm,
    FuncNorm,
    ListedColormap,
)
try:
    # mpl >= 3.6
    from matplotlib.colors import AsinhNorm  # type: ignore
except Exception:  # pragma: no cover
    AsinhNorm = None
from matplotlib.ticker import ScalarFormatter
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import mapping

# Optional smoothing/cleanup (graceful fallback if SciPy missing)
try:
    from scipy.ndimage import gaussian_filter, label as cc_label
except Exception:  # pragma: no cover
    gaussian_filter, cc_label = None, None

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import log_utilities as lu
from src.scripts.utilities import universal_utilities as uu

from src.scripts.postprocessing.visualization.create_global_map_common import (
    DEFAULT_DATE_TAG,
    DEFAULT_MODEL_VERSION,
    DEFAULT_TARGET_DEG,
    DISPLAY_OUT_ROOT,
    OUTPUT_ROOT,
    assert_grid_divides_world,
    build_download_upload_dict,
    deg_to_label,
    ensure_dir,
    gdalize_s3_url,
    resolve_versioned_paths,
    to_local_mirror,
)

# ------------------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------------------

def _split_s3_url(s3_url: str) -> Tuple[str, str]:
    if not s3_url.startswith("s3://"):
        raise ValueError(f"Expected s3:// URL, received: {s3_url}")
    bucket_key = s3_url[len("s3://") :]
    parts = bucket_key.split("/", 1)
    bucket = parts[0]
    key_prefix = parts[1] if len(parts) > 1 else ""
    return bucket, key_prefix


def _upload_display_outputs(
    created_files: List[Path],
    out_dir_local: Path,
    dest_s3_dir: str,
    logger,
) -> None:
    if not created_files:
        return

    if not dest_s3_dir.startswith("s3://"):
        logger.warning(
            "Display output directory is not an S3 path; skipping upload: %s", dest_s3_dir
        )
        return

    bucket, prefix = _split_s3_url(dest_s3_dir.rstrip("/") + "/")
    prefix = prefix.rstrip("/")

    for local_path in created_files:
        try:
            rel_path = local_path.relative_to(out_dir_local)
        except ValueError:
            logger.warning(
                "Skipping upload for %s because it is outside %s", local_path, out_dir_local
            )
            continue

        rel_key = "/".join(rel_path.parts)
        key = f"{prefix}/{rel_key}" if prefix else rel_key

        uu.upload_file_to_s3(str(local_path), bucket, key)
        logger.info("Uploaded %s to s3://%s/%s", local_path, bucket, key)


def _rgb_to_mpl(rgb):
    return tuple(v / 255 for v in rgb)


def _rgb_palette_to_mpl(rgb_palette):
    return [_rgb_to_mpl(rgb) for rgb in rgb_palette]


def _reproject_if_needed(out_tif: str, in_tif: str, target_crs: str):
    """Reproject to ``target_crs`` if ``out_tif`` does not exist; set nodata=0."""
    if Path(out_tif).exists():
        return out_tif
    with rasterio.open(in_tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        compression = profile.get("compress", "deflate")
        profile.update(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            nodata=0,
            compress=compression,
            tiled=True,
            blockxsize=min(512, width),
            blockysize=min(512, height),
        )
        ensure_dir(Path(out_tif).parent)
        with rasterio.open(out_tif, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest,
                src_nodata=0,
                dst_nodata=0,
                init_dest_nodata=True,
            )
    return out_tif


def _plot_country_layers(ax, shp_gdf):
    if shp_gdf is None or shp_gdf.empty:
        return
    try:
        for geom in shp_gdf.geometry:
            try:
                if hasattr(geom, "exterior"):
                    x, y = geom.exterior.xy
                    ax.fill(x, y, color=_rgb_to_mpl(cn.land_bkgrnd), zorder=1)
                else:
                    for part in geom.geoms:
                        px, py = part.exterior.xy
                        ax.fill(px, py, color=_rgb_to_mpl(cn.land_bkgrnd), zorder=1)
            except Exception:
                continue
        shp_gdf.boundary.plot(
            ax=ax,
            edgecolor=_rgb_to_mpl(cn.boundary_color),
            linewidth=cn.boundary_width,
            zorder=3,
        )
    except Exception:
        pass


def _legend(fig_or_ax, img, title_text: str, steps_levels: Optional[np.ndarray] = None):
    """Norm-aware legend that works for Linear/Boundary norms (and AsinhNorm/FuncNorm)."""
    fig = fig_or_ax.figure if hasattr(fig_or_ax, "figure") else fig_or_ax
    cax = fig.add_axes(cn.colorbar_dimensions)

    if steps_levels is not None and steps_levels.size >= 2:
        cb = plt.colorbar(img, cax=cax, orientation="vertical",
                          boundaries=steps_levels, ticks=steps_levels)
    else:
        cb = plt.colorbar(img, cax=cax, orientation="vertical")

    cb.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    cb.ax.tick_params(labelsize=cn.legend_fontsize)

    cax.text(
        0,
        1.1,
        title_text,
        fontsize=cn.legend_fontsize,
        ha="left",
        va="bottom",
        transform=cax.transAxes,
    )


def _draw_frame(ax, extent, masked, cmap, norm, ocean_rgb, interpolation: str):
    ax.set_facecolor(_rgb_to_mpl(ocean_rgb))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    img = ax.imshow(
        masked,
        cmap=cmap,
        norm=norm,
        extent=extent,
        origin="upper",
        interpolation=interpolation,
        zorder=2,
    )
    return img


def _save_single(ax, base_out: Path, year_label: str | int, profile_suffix: str) -> Path:
    # Interval appears exactly once here.
    out_path = Path(f"{base_out}__{year_label}__{profile_suffix}.jpeg")
    ax.text(
        0.5,
        0.07,
        str(year_label),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=18,
        weight="bold",
        color="black",
    )
    plt.savefig(out_path, dpi=cn.dpi_jpeg, bbox_inches="tight", pad_inches=0)
    plt.close()
    return out_path


def _load_or_project_raster(base_tif_noext: str, target_crs: str, work_dir: Path) -> str:
    tif_unproj = base_tif_noext + ".tif"
    tif_proj = work_dir / (Path(base_tif_noext).name + "_reproj.tif")
    return _reproject_if_needed(str(tif_proj), tif_unproj, target_crs)


def _maybe_load_world_boundaries(logger) -> Optional[gpd.GeoDataFrame]:
    try:
        rp = Path(getattr(cn, "reprojected_shapefile_path", ""))
        op = Path(getattr(cn, "original_shapefile_path", ""))
        if rp.exists():
            return gpd.read_file(rp).to_crs(cn.Robinson_crs)
        if op.exists():
            return gpd.read_file(op).to_crs(cn.Robinson_crs)
        logger.warning(f"World boundaries not found: {rp} or {op}")
        return None
    except Exception as e:
        logger.warning(f"World boundaries load failed: {e}")
        return None


def _try_open_first(paths_noext: List[str], target_crs: str, work_dir: Path) -> Tuple[str, str]:
    last_error = None
    for base in paths_noext:
        try:
            kind = "s3" if base.startswith("/vsis3/") or base.startswith("s3://") else "local"
            tif_proj = _load_or_project_raster(base, target_crs, work_dir)
            with rasterio.open(tif_proj):
                pass
            return tif_proj, kind
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(
        f"Could not open any candidate raster: {paths_noext}. Last error: {last_error}"
    )

# ------------------------------------------------------------------------------
# Emissions viz helpers (profiles: asinh / linear / steps)
# ------------------------------------------------------------------------------

def _profile_suffix(name: str, clip_lo: float, clip_hi: float, steps: Optional[List[float]] = None):
    def fmt_p(p):
        # 99.5 -> 99p5 for filenames
        s = str(p).replace(".", "p")
        return s
    if name == "steps":
        if not steps:
            return "prof-steps"
        parts = [("q" + fmt_p(v)) for v in steps]
        return "prof-steps__" + "-".join(parts)
    else:
        return f"prof-{name}__p{fmt_p(clip_lo)}-{fmt_p(clip_hi)}"


def _select_profiles(
    profiles_arg: Optional[List[str]],
    override_clip_lo: Optional[float],
    override_clip_hi: Optional[float],
    override_floor: Optional[float],
    override_steps: Optional[List[float]],
) -> List[Dict]:
    """
    Only 'asinh', 'linear', and 'steps' are supported.
    """
    default_profiles = [
        {"name": "asinh",  "stretch": "asinh",  "clip_lo": 5.0, "clip_hi": 99.97, "floor": None, "steps": None},
        {"name": "linear", "stretch": "linear", "clip_lo": 5.0, "clip_hi": 95.0,  "floor": None, "steps": None},
        {"name": "steps",  "stretch": "steps",  "clip_lo": 5.0, "clip_hi": 99.97, "floor": None,
         "steps": [60, 80, 90, 95, 98, 99, 99.5, 99.9]},
    ]
    available = {p["name"]: p for p in default_profiles}

    if not profiles_arg or "all" in [p.lower() for p in profiles_arg]:
        selected = [available[k] for k in ("asinh", "linear", "steps")]
    else:
        selected = []
        for name in profiles_arg:
            key = name.lower()
            if key in available:
                selected.append(available[key])

    # Apply overrides
    out = []
    for p in selected:
        pc = dict(p)  # copy
        if override_clip_lo is not None:
            pc["clip_lo"] = float(override_clip_lo)
        if override_clip_hi is not None:
            pc["clip_hi"] = float(override_clip_hi)
        if override_floor is not None:
            pc["floor"] = float(override_floor)
        if override_steps is not None and pc["name"] == "steps":
            pc["steps"] = list(override_steps)
        out.append(pc)
    return out


def _land_stats_mask_from_shp(shp_gdf: Optional[gpd.GeoDataFrame], out_shape: Tuple[int, int], transform):
    """Rasterize land polygons to a boolean mask aligned to the raster grid (True on land)."""
    if shp_gdf is None or shp_gdf.empty:
        return None
    shapes = [(mapping(geom), 1) for geom in shp_gdf.geometry if geom is not None]
    if not shapes:
        return None
    mask = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        all_touched=False,
        fill=0,
        dtype="uint8",
    )
    return mask.astype(bool)


def _asinh_norm(vmin: float, vmax: float):
    if AsinhNorm is not None:  # mpl >= 3.6
        return AsinhNorm(vmin=max(vmin, 0.0), vmax=vmax)
    # Fallback: emulate asinh with FuncNorm
    def fwd(x): return np.arcsinh(x)
    def inv(y): return np.sinh(y)
    return FuncNorm((fwd, inv), vmin=max(vmin, 0.0), vmax=vmax)


def _norm_for(stretch: str, vmin: float, vmax: float, steps_levels: Optional[np.ndarray] = None):
    if stretch == "asinh":
        return _asinh_norm(vmin, vmax)
    if stretch == "steps":
        if steps_levels is None or steps_levels.size < 2:
            return Normalize(vmin=vmin, vmax=vmax)
        return BoundaryNorm(steps_levels, ncolors=256, clip=True)
    # linear
    return Normalize(vmin=vmin, vmax=vmax)

# ------------------------------------------------------------------------------
# Array loading & composition
# ------------------------------------------------------------------------------

def _get_items_for_dataset_interval(d: Dict, dataset_name: str, interval: str):
    for v in d.values():
        if v.get("dataset") == dataset_name and v.get("interval") == interval:
            return v
    return None


def _candidates_for_dataset_interval(
    d: Dict,
    dataset_name: str,
    interval: str,
    read_from_s3: bool,
    resolved_base_url: str,
    run_name: str,
    pixel_resolution: str,
    date_tag: str,
    res_label: str,
) -> List[str]:
    """Find candidate no-ext paths for a dataset/interval using job dict; fallback to versioned path."""
    items = _get_items_for_dataset_interval(d, dataset_name, interval)
    candidates: List[str] = []
    if items:
        canonical_noext = posixpath.join(items["global_dir"], items["global_pattern"][:-4])
        versioned_noext = posixpath.join(
            resolved_base_url.rstrip("/"),
            dataset_name,
            run_name,
            "five_year_intervals",
            interval,
            pixel_resolution,
            date_tag,
            f"{res_label}_global__{dataset_name}__{interval}",
        )
        candidates = [canonical_noext, versioned_noext]
    else:
        versioned_noext = posixpath.join(
            resolved_base_url.rstrip("/"),
            dataset_name,
            run_name,
            "five_year_intervals",
            interval,
            pixel_resolution,
            date_tag,
            f"{res_label}_global__{dataset_name}__{interval}",
        )
        candidates = [versioned_noext]
    if read_from_s3:
        candidates = [gdalize_s3_url(p) for p in candidates]
    return candidates


def _open_array_from_candidates(
    candidates: List[str],
    target_crs: str,
    work_dir: Path,
) -> Tuple[np.ndarray, rasterio.coords.BoundingBox, any]:
    tif_proj, _ = _try_open_first(candidates, target_crs, work_dir)
    with rasterio.open(tif_proj) as src:
        arr = src.read(1)
        bounds = src.bounds
        transform = src.transform
    arr = np.nan_to_num(arr, nan=0.0)
    return arr, bounds, transform


def _compose_sum_positive(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Return (a+b) with <=0 masked (as 0)."""
    s = np.where(np.isfinite(a), a, 0.0) + np.where(np.isfinite(b), b, 0.0)
    s = np.nan_to_num(s, nan=0.0)
    s[s <= 0] = 0.0
    return s


def _reclass_drained_state_to_binary(
    state_array: np.ndarray,
) -> np.ma.MaskedArray:
    """
    0 -> nodata
    20,000,000 -> nodata
    16,000,000 -> UNDRAINED peat (0)
    all other values -> DRAINED peat (1)
    """
    nodata_mask = (state_array == 0) | (state_array == 20000000)
    undrained_mask = (state_array == 16000000)
    drained_mask = (~nodata_mask) & (~undrained_mask)

    out = np.zeros_like(state_array, dtype=np.uint8)
    out[drained_mask] = 1
    out[undrained_mask] = 0
    return np.ma.masked_array(out, mask=nodata_mask)

# ------------------------------------------------------------------------------
# Rendering: emissions profiles (as before) + binary extent
# ------------------------------------------------------------------------------

def _compute_domain_and_norm(
    masked_positive: np.ma.MaskedArray,
    stretch: str,
    clip_lo: float,
    clip_hi: float,
    floor: Optional[float],
    land_stats_mask: Optional[np.ndarray],
    stats_scope: str,
    steps: Optional[List[float]] = None,
):
    """Compute vmin/vmax (+ steps levels if needed) and build a Norm."""
    if land_stats_mask is not None and stats_scope == "land":
        stats_values = masked_positive[land_stats_mask].compressed()
    else:
        stats_values = masked_positive.compressed()

    if stats_values.size == 0:
        vmin, vmax = 1.0, 1.0
    else:
        lo = float(np.percentile(stats_values, clip_lo))
        hi = float(np.percentile(stats_values, clip_hi))
        hi = max(hi, lo * 1.01)
        if stretch in ("asinh",):  # needs positive floor
            floor_eff = (0.01 * hi) if floor is None else float(floor)
            vmin = max(lo, floor_eff, 1e-12)
        else:
            vmin = max(lo, 0.0)
        vmax = max(hi, vmin * 1.01)

    steps_levels = None
    if stretch == "steps" and stats_values.size > 0:
        pp = np.unique(np.clip(np.array(steps or [], dtype=float), 0, 100))
        if pp.size > 0:
            ld = np.log10(stats_values)
            q = np.percentile(ld, pp)
            steps_levels = np.unique(np.concatenate(([np.log10(vmin)], q, [np.log10(vmax)])))
            steps_levels = np.power(10.0, steps_levels)
            steps_levels = steps_levels[(steps_levels >= vmin) & (steps_levels <= vmax)]
            if steps_levels.size < 2:
                steps_levels = None

    norm = _norm_for(stretch, vmin, vmax, steps_levels)
    return vmin, vmax, steps_levels, norm


def _render_emissions_single_profile(
    *,
    arr: np.ndarray,
    bounds,
    legend_title: str,
    out_dir_local: str,
    out_name_base: str,
    label_value: str,
    stretch: str,
    clip_lo: float,
    clip_hi: float,
    floor: Optional[float],
    steps: Optional[List[float]],
    shapefile_gdf: Optional[gpd.GeoDataFrame],
    land_stats_mask: Optional[np.ndarray],
    stats_scope: str,
    cmap,
    interpolation: str,
    ocean_color,
    logger,
):
    masked_positive = np.ma.masked_where(arr <= 0, arr)

    vmin, vmax, steps_levels, norm = _compute_domain_and_norm(
        masked_positive, stretch, clip_lo, clip_hi, floor, land_stats_mask, stats_scope, steps
    )

    fig, ax = plt.subplots(figsize=cn.panel_dims)
    _plot_country_layers(ax, shapefile_gdf)
    img = _draw_frame(
        ax,
        [bounds.left, bounds.right, bounds.bottom, bounds.top],
        masked_positive,
        cmap,
        norm,
        ocean_color,
        interpolation=interpolation,
    )
    _legend(fig, img, legend_title, steps_levels=steps_levels)

    base_out = Path(out_dir_local) / out_name_base  # base has NO interval now
    suffix = _profile_suffix(stretch, clip_lo, clip_hi, steps)
    out_img = _save_single(ax, base_out, label_value, suffix)
    logger.info(
        "Rendered %s %s | %s | vmin=%.3g vmax=%.3g", out_name_base, label_value, suffix, vmin, vmax
    )
    return out_img


def _render_binary_extent(
    *,
    binary_mask: np.ma.MaskedArray,
    bounds,
    out_dir_local: str,
    out_name_base: str,
    label_value: str,
    shapefile_gdf: Optional[gpd.GeoDataFrame],
    interpolation: str,
    ocean_color,
    drained_color=(204/255, 76/255, 2/255, 1.0),      # orange-brown
    undrained_color=(102/255, 194/255, 164/255, 1.0), # teal
    logger,
):
    # 1 -> drained, 0 -> undrained; mask = outside peat (transparent)
    fig, ax = plt.subplots(figsize=cn.panel_dims)
    _plot_country_layers(ax, shapefile_gdf)

    # Build categorical colormap
    cmap = ListedColormap([undrained_color, drained_color])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    img = _draw_frame(
        ax,
        [bounds.left, bounds.right, bounds.bottom, bounds.top],
        binary_mask,
        cmap,
        norm,
        ocean_color,
        interpolation=interpolation,
    )

    # Legend-like patches
    from matplotlib.patches import Patch
    handles = [Patch(color=undrained_color, label="Undrained peat"),
               Patch(color=drained_color, label="Drained peat")]
    ax.legend(handles=handles, loc="lower right", fontsize=cn.legend_fontsize)

    base_out = Path(out_dir_local) / out_name_base  # base has NO interval now
    suffix = "binary_drained_extent"
    out_img = _save_single(ax, base_out, label_value, suffix)
    logger.info("Rendered %s %s | %s", out_name_base, label_value, suffix)
    return out_img

# ------------------------------------------------------------------------------
# 4-panel composer (global + A/B/C zooms)
# ------------------------------------------------------------------------------

def _to_crs_obj(val):
    return val if isinstance(val, CRS) else CRS.from_string(val)

def _lonlat_box_to_robinson(bbox_ll: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Convert lon/lat bbox -> Robinson (meters) bbox."""
    minlon, minlat, maxlon, maxlat = bbox_ll
    src_crs = CRS.from_epsg(4326)
    dst_crs = _to_crs_obj(cn.Robinson_crs)
    xs, ys = rio_transform(src_crs, dst_crs,
                           [minlon, maxlon, maxlon, minlon],
                           [minlat, minlat, maxlat, maxlat])
    return min(xs), min(ys), max(xs), max(ys)


def _add_box(ax, bbox_xy: Tuple[float, float, float, float], label_txt: str, color="black"):
    from matplotlib.patches import Rectangle
    x0, y0, x1, y1 = bbox_xy
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.5, zorder=4)
    ax.add_patch(rect)
    ax.text(x0, y1, f" {label_txt}", ha="left", va="bottom", fontsize=10, color=color, weight="bold", zorder=5)


def _add_scale_bar(ax, bbox_xy: Tuple[float, float, float, float]):
    """Simple scalebar ~1/4 of AOI width, labeled in km (projection meters)."""
    x0, y0, x1, y1 = bbox_xy
    width = x1 - x0
    height = y1 - y0
    candidates = np.array([10e3, 25e3, 50e3, 100e3, 200e3, 500e3, 1e6])  # meters
    target = width / 4.0
    length = candidates[candidates <= target]
    length = length[-1] if length.size else candidates[0]
    # position near bottom-left
    pad_x = 0.08 * width
    pad_y = 0.08 * height
    x_left = x0 + pad_x
    x_right = x_left + length
    y = y0 + pad_y
    ax.plot([x_left, x_right], [y, y], color="black", linewidth=2.5, zorder=6)
    ax.text((x_left + x_right) / 2, y + 0.02 * height, f"{int(length/1000)} km",
            ha="center", va="bottom", fontsize=9, color="black", zorder=6)


def _four_panel(
    *,
    arr: np.ndarray,
    bounds,
    shapefile_gdf: Optional[gpd.GeoDataFrame],
    land_stats_mask: Optional[np.ndarray],
    stats_scope: str,
    profile_cfg: Dict,
    cmap,
    interpolation: str,
    ocean_color,
    aoi_boxes_ll: List[Tuple[str, Tuple[float, float, float, float]]],
    legend_title: str,
    out_dir_local: str,
    out_name_base: str,
    label_value: str,
    logger,
):
    """Compose 4 panels: Global + 3 AOIs."""
    masked_positive = np.ma.masked_where(arr <= 0, arr)
    vmin, vmax, steps_levels, norm = _compute_domain_and_norm(
        masked_positive, profile_cfg["stretch"], profile_cfg["clip_lo"], profile_cfg["clip_hi"],
        profile_cfg.get("floor", None), land_stats_mask, stats_scope, profile_cfg.get("steps")
    )

    # AOIs to projection
    aoi_proj = []
    for name, bb in aoi_boxes_ll:
        aoi_proj.append((name, _lonlat_box_to_robinson(bb)))

    # Figure + axes
    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.15, wspace=0.05)
    ax_global = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # Global panel
    _plot_country_layers(ax_global, shapefile_gdf)
    img = _draw_frame(
        ax_global,
        [bounds.left, bounds.right, bounds.bottom, bounds.top],
        masked_positive,
        cmap, norm, ocean_color, interpolation
    )
    # Boxes
    for (name, bb) in aoi_proj:
        _add_box(ax_global, bb, name, color="black")
    ax_global.set_title("Global overview", fontsize=11)

    # Zoom panels A/B/C
    for ax, (name, bb) in zip([ax_a, ax_b, ax_c], aoi_proj):
        _plot_country_layers(ax, shapefile_gdf)
        _draw_frame(
            ax,
            [bounds.left, bounds.right, bounds.bottom, bounds.top],
            masked_positive,
            cmap, norm, ocean_color, interpolation
        )
        ax.set_xlim(bb[0], bb[2])
        ax.set_ylim(bb[1], bb[3])
        _add_scale_bar(ax, bb)
        ax.set_title(f"Site {name} [{bb[0]:.0f}, {bb[1]:.0f}, {bb[2]:.0f}, {bb[3]:.0f}]", fontsize=10)

    # Shared colorbar
    cax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
    if steps_levels is not None and steps_levels.size >= 2:
        cb = plt.colorbar(img, cax=cax, orientation="vertical",
                          boundaries=steps_levels, ticks=steps_levels)
    else:
        cb = plt.colorbar(img, cax=cax, orientation="vertical")
    cb.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    cb.ax.tick_params(labelsize=10)
    cax.text(0, 1.02, legend_title, fontsize=10, ha="left", va="bottom", transform=cax.transAxes)

    base_out = Path(out_dir_local) / out_name_base  # base has NO interval now
    suffix = f"fourpanel_{profile_cfg['name']}"
    out_img = Path(f"{base_out}__{label_value}__{suffix}.jpeg")
    plt.savefig(out_img, dpi=cn.dpi_jpeg, bbox_inches="tight", pad_inches=0)
    plt.close()
    logger.info("Rendered 4-panel %s -> %s", out_name_base, out_img)
    return out_img

# ------------------------------------------------------------------------------
# Core rendering per dataset/interval (profiles + binary + four-panels)
# ------------------------------------------------------------------------------

def make_displays_for_dataset(
    *,
    dataset_name: str,
    interval: str,
    tif_path_noext_candidates: List[str],
    out_dir_local: str,
    palette_rgb: List[Tuple[int, int, int]],
    shapefile_gdf: Optional[gpd.GeoDataFrame],
    out_name_base: str,
    years: Optional[List[int]] = None,
    target_crs: str = cn.Robinson_crs,
    profiles: List[Dict],
    interpolation: str,
    stats_scope: str,
    min_cluster_pixels: int,
    smooth_sigma: float,
    cmap_choice: str,
    logger=None,
):
    """
    Render multiple profiles (asinh, linear, steps) for a single dataset/interval.
    Assumes gross emissions (mask <= 0).
    """
    logger = logger or lu.setup_logging()
    t0 = time.time()

    ensure_dir(out_dir_local)
    work_dir = Path(out_dir_local) / "_reproj_cache"
    ensure_dir(work_dir)

    # Colormap
    if cmap_choice.lower() == "custom":
        colors_mpl = _rgb_palette_to_mpl(palette_rgb)
        cmap = LinearSegmentedColormap.from_list("custom", colors_mpl)
    else:
        try:
            cmap = cm.get_cmap(cmap_choice)
        except Exception:
            cmap = cm.get_cmap("inferno")

    # Sparse-friendly: transparent masked; subtle under/over
    try:
        cmap = cmap.with_extremes(
            under=_rgb_to_mpl(cn.ocean_color) + (0.15,),
            over=_rgb_to_mpl(cn.ocean_color) + (0.15,),
        )
    except Exception:
        cmap.set_under(_rgb_to_mpl(cn.ocean_color))
        cmap.set_over(_rgb_to_mpl(cn.ocean_color))
    cmap.set_bad((0, 0, 0, 0))

    created_files: List[Path] = []
    labels = years or [interval]

    legend_title = (
        "Gross greenhouse gas emissions\nkt CO$_2$e yr$^{-1}$"
        if "co2" not in dataset_name.lower()
        else "Gross CO$_2$ emissions\nkt CO$_2$ yr$^{-1}$"
    )

    tif_proj, _ = _try_open_first(tif_path_noext_candidates, target_crs, work_dir)
    with rasterio.open(tif_proj) as src:
        data_full = src.read(1)
        bounds = src.bounds
        transform = src.transform

    data_full = np.nan_to_num(data_full, nan=0.0)
    masked_positive = np.ma.masked_where(data_full <= 0, data_full)
    land_stats_mask = _land_stats_mask_from_shp(shapefile_gdf, masked_positive.shape, transform)

    for frame_label in labels:
        # Optional cleanup/smoothing for plotting
        plot_array = masked_positive
        if min_cluster_pixels and min_cluster_pixels > 1 and cc_label is not None:
            pos_mask = (~plot_array.mask).astype(np.uint8)
            lab, nlab = cc_label(pos_mask)  # type: ignore
            if nlab and nlab > 0:
                counts = np.bincount(lab.ravel())
                small_ids = np.where(counts < min_cluster_pixels)[0]
                if small_ids.size > 0:
                    small_mask = np.isin(lab, small_ids)
                    plot_array = np.ma.masked_where(small_mask | (data_full <= 0), data_full)

        if smooth_sigma and gaussian_filter is not None and smooth_sigma > 0:
            data = plot_array.filled(0.0)
            w = (~plot_array.mask).astype(float)
            num = gaussian_filter(data, smooth_sigma)  # type: ignore
            den = gaussian_filter(w,    smooth_sigma)  # type: ignore
            sm = num / np.maximum(den, 1e-9)
            plot_array = np.ma.array(sm, mask=(w == 0))

        for p in profiles:
            vmin, vmax, steps_levels, norm = _compute_domain_and_norm(
                plot_array, p["stretch"], p["clip_lo"], p["clip_hi"], p.get("floor", None),
                land_stats_mask, stats_scope, p.get("steps")
            )

            fig, ax = plt.subplots(figsize=cn.panel_dims)
            _plot_country_layers(ax, shapefile_gdf)
            img = _draw_frame(
                ax,
                [bounds.left, bounds.right, bounds.bottom, bounds.top],
                plot_array,
                cmap,
                norm,
                cn.ocean_color,
                interpolation=interpolation,
            )
            _legend(fig, img, legend_title, steps_levels=steps_levels)

            # NOTE: base path has NO interval; _save_single adds it once
            base_out = Path(out_dir_local) / out_name_base
            suffix = _profile_suffix(p["name"], p["clip_lo"], p["clip_hi"], p.get("steps"))
            out_img = _save_single(ax, base_out, frame_label, suffix)
            created_files.append(out_img)

            pos_count = int(plot_array.count())
            hi_clipped = int((plot_array > vmax).sum())
            logger.info(
                "Profile=%s | vmin=%.3g vmax=%.3g | positives=%d | clipped_hi=%.2f%%",
                p["name"], vmin, vmax, pos_count, 100.0 * hi_clipped / max(1, pos_count)
            )

    logger.info(
        "Rendered %s %s (%d profiles) in %s s → %s",
        dataset_name,
        interval,
        len(profiles),
        round(time.time() - t0),
        posixpath.join(out_dir_local, out_name_base),
    )
    return created_files

# ------------------------------------------------------------------------------
# Orchestration: build tasks including derived composites & four-panels
# ------------------------------------------------------------------------------

def display_main(
    *,
    date_tag: str,
    read_from_s3: bool,
    run_name: str,
    pixel_resolution: str,
    target_deg: float = DEFAULT_TARGET_DEG,
    model_version: str = DEFAULT_MODEL_VERSION,
    outputs_root: str = OUTPUT_ROOT,
    base_url: Optional[str] = None,
    outputs_base: Optional[str] = None,
    upload_to_s3: bool = True,
    # profile controls
    profiles_arg: Optional[List[str]] = None,
    clip_lo: Optional[float] = None,
    clip_hi: Optional[float] = None,
    floor: Optional[float] = None,
    steps: Optional[List[float]] = None,
    interpolation: str = "nearest",
    stats_scope: str = "land",
    min_cluster_pixels: int = 0,
    smooth_sigma: float = 0.0,
    cmap_choice: str = "custom",
    # dataset names for drained/burned/state
    ds_drained_emis: Optional[str] = None,
    ds_burned_emis: Optional[str] = None,
    ds_drained_state: Optional[str] = None,
    # four-panel controls
    make_four_panel_drained: bool = False,
    make_four_panel_burned: bool = False,
    four_panel_profile: str = "asinh",
    aoi_boxes: Optional[List[str]] = None,  # ["A:minlon,minlat,maxlon,maxlat", ...]
    only_extent: bool = False,
):
    assert_grid_divides_world(target_deg)
    res_label = deg_to_label(target_deg)

    logger = lu.setup_logging_main()
    resolved_base_url, resolved_outputs_base = resolve_versioned_paths(
        model_version=model_version,
        outputs_root=outputs_root,
        base_url=base_url,
        outputs_base=outputs_base,
    )

    # Build job list
    d = build_download_upload_dict(
        pixel_resolution=pixel_resolution,
        run_name=run_name,
        target_deg=target_deg,
        base_url=resolved_base_url,
        output_date=date_tag,
        outputs_base=resolved_outputs_base,
    )

    shp = _maybe_load_world_boundaries(logger)

    # Default palettes
    net_palette = [
        (0, 60, 48),
        (1, 102, 94),
        (53, 151, 143),
        (128, 205, 193),
        (199, 234, 229),
        (246, 232, 195),
        (223, 194, 125),
        (191, 129, 45),
        (140, 81, 10),
        (84, 48, 5),
    ]
    emissions_palette = net_palette[5:]  # warm half

    # Select which profiles to render (asinh/linear/steps)
    profile_cfgs = _select_profiles(
        profiles_arg=profiles_arg,
        override_clip_lo=clip_lo,
        override_clip_hi=clip_hi,
        override_floor=floor,
        override_steps=steps,
    )

    # Single profile for the four-panel map (defaults to 'asinh')
    four_panel_cfg = [p for p in profile_cfgs if p["name"] == four_panel_profile]
    if not four_panel_cfg:
        four_panel_cfg = [{"name": "asinh", "stretch": "asinh", "clip_lo": 5.0, "clip_hi": 99.97, "floor": None, "steps": None}]
    else:
        four_panel_cfg = four_panel_cfg[0]

    # Try to infer dataset names if not provided
    if ds_drained_emis is None:
        for v in d.values():
            nm = v.get("dataset", "").lower()
            if "drain" in nm and "emiss" in nm:
                ds_drained_emis = v["dataset"]; break
    if ds_burned_emis is None:
        for v in d.values():
            nm = v.get("dataset", "").lower()
            if "burn" in nm and "emiss" in nm:
                ds_burned_emis = v["dataset"]; break
    if ds_drained_state is None:
        for v in d.values():
            nm = v.get("dataset", "").lower()
            if "drain" in nm and "state" in nm:
                ds_drained_state = v["dataset"]; break

    if only_extent and ds_drained_state is None:
        logger.error("only_extent requested but --ds_drained_state not provided and could not be inferred.")
        return

    # AOIs
    aoi_list = []
    if aoi_boxes:
        for s in aoi_boxes:
            # "A:minlon,minlat,maxlon,maxlat"
            name, coords = s.split(":")
            parts = [float(x) for x in coords.split(",")]
            if len(parts) != 4:
                continue
            aoi_list.append((name.strip(), (parts[0], parts[1], parts[2], parts[3])))

    # Iterate intervals found in job dict
    intervals = sorted({v["interval"] for v in d.values()})
    for interval in intervals:

        # -------------------------
        # 1) Binary drained/undrained extent (from drained-state raster)
        # -------------------------
        if ds_drained_state:
            cand_state = _candidates_for_dataset_interval(
                d, ds_drained_state, interval, read_from_s3, resolved_base_url,
                run_name, pixel_resolution, date_tag, res_label
            )
            items_state = _get_items_for_dataset_interval(d, ds_drained_state, interval)
            if items_state:
                out_dir_base_state = posixpath.join(items_state["global_dir"], "display", interval, "drained_extent_binary")
            else:
                out_dir_base_state = posixpath.join(OUTPUT_ROOT, "display", interval, "drained_extent_binary")
            out_dir_local_state = to_local_mirror(out_dir_base_state, DISPLAY_OUT_ROOT)
            ensure_dir(out_dir_local_state)

            logger.info("[Stage 02] Reclassifying drained-state dataset=%s interval=%s | candidates=%s",
                        ds_drained_state, interval, cand_state)

            work_dir = Path(out_dir_local_state) / "_reproj_cache"; ensure_dir(work_dir)
            state_arr, state_bounds, _ = _open_array_from_candidates(cand_state, cn.Robinson_crs, work_dir)
            binary = _reclass_drained_state_to_binary(state_arr)

            out_base_state = "drained_extent"  # base has NO interval
            out_img = _render_binary_extent(
                binary_mask=binary, bounds=state_bounds, out_dir_local=out_dir_local_state,
                out_name_base=out_base_state, label_value=interval, shapefile_gdf=shp,
                interpolation=interpolation, ocean_color=cn.ocean_color, logger=logger
            )
            if upload_to_s3:
                _upload_display_outputs([out_img], Path(out_dir_local_state), out_dir_base_state, logger)

            if only_extent:
                continue  # skip other products when only extent requested

        # -------------------------
        # 2) Drained emissions (global profiles)
        # -------------------------
        if ds_drained_emis:
            cand_drained = _candidates_for_dataset_interval(
                d, ds_drained_emis, interval, read_from_s3, resolved_base_url,
                run_name, pixel_resolution, date_tag, res_label
            )
            items_drained = _get_items_for_dataset_interval(d, ds_drained_emis, interval)
            if items_drained:
                out_dir_base_drained = posixpath.join(items_drained["global_dir"], "display", interval, ds_drained_emis)
            else:
                out_dir_base_drained = posixpath.join(OUTPUT_ROOT, "display", interval, ds_drained_emis)
            out_dir_local_drained = to_local_mirror(out_dir_base_drained, DISPLAY_OUT_ROOT)
            ensure_dir(out_dir_local_drained)

            logger.info("[Stage 02] Rendering drained emissions dataset=%s interval=%s | candidates=%s",
                        ds_drained_emis, interval, cand_drained)

            created = make_displays_for_dataset(
                dataset_name=ds_drained_emis, interval=interval,
                tif_path_noext_candidates=cand_drained, out_dir_local=out_dir_local_drained,
                palette_rgb=emissions_palette, shapefile_gdf=shp, out_name_base=ds_drained_emis,  # base has NO interval
                years=None, target_crs=cn.Robinson_crs, profiles=profile_cfgs,
                interpolation=interpolation, stats_scope=stats_scope, min_cluster_pixels=min_cluster_pixels,
                smooth_sigma=smooth_sigma, cmap_choice=cmap_choice, logger=logger,
            )
            if upload_to_s3:
                _upload_display_outputs(created, Path(out_dir_local_drained), out_dir_base_drained, logger)

        # -------------------------
        # 3) Burned emissions (global profiles)
        # -------------------------
        if ds_burned_emis:
            cand_burned = _candidates_for_dataset_interval(
                d, ds_burned_emis, interval, read_from_s3, resolved_base_url,
                run_name, pixel_resolution, date_tag, res_label
            )
            items_burned = _get_items_for_dataset_interval(d, ds_burned_emis, interval)
            if items_burned:
                out_dir_base_burned = posixpath.join(items_burned["global_dir"], "display", interval, ds_burned_emis)
            else:
                out_dir_base_burned = posixpath.join(OUTPUT_ROOT, "display", interval, ds_burned_emis)
            out_dir_local_burned = to_local_mirror(out_dir_base_burned, DISPLAY_OUT_ROOT)
            ensure_dir(out_dir_local_burned)

            logger.info("[Stage 02] Rendering burned emissions dataset=%s interval=%s | candidates=%s",
                        ds_burned_emis, interval, cand_burned)

            created = make_displays_for_dataset(
                dataset_name=ds_burned_emis, interval=interval,
                tif_path_noext_candidates=cand_burned, out_dir_local=out_dir_local_burned,
                palette_rgb=emissions_palette, shapefile_gdf=shp, out_name_base=ds_burned_emis,  # base has NO interval
                years=None, target_crs=cn.Robinson_crs, profiles=profile_cfgs,
                interpolation=interpolation, stats_scope=stats_scope, min_cluster_pixels=min_cluster_pixels,
                smooth_sigma=smooth_sigma, cmap_choice=cmap_choice, logger=logger,
            )
            if upload_to_s3:
                _upload_display_outputs(created, Path(out_dir_local_burned), out_dir_base_burned, logger)

        # -------------------------
        # 4) Drained + Burned emissions (sum; global profiles)
        # -------------------------
        if ds_drained_emis and ds_burned_emis:
            work_dir_sum = Path(DISPLAY_OUT_ROOT) / "derived_sum" / "_reproj_cache"
            ensure_dir(work_dir_sum)

            drained_arr, bounds_d, transform_d = _open_array_from_candidates(
                _candidates_for_dataset_interval(d, ds_drained_emis, interval, read_from_s3,
                                                 resolved_base_url, run_name, pixel_resolution, date_tag, res_label),
                cn.Robinson_crs, work_dir_sum
            )
            burned_arr, bounds_b, transform_b = _open_array_from_candidates(
                _candidates_for_dataset_interval(d, ds_burned_emis, interval, read_from_s3,
                                                 resolved_base_url, run_name, pixel_resolution, date_tag, res_label),
                cn.Robinson_crs, work_dir_sum
            )
            combined = _compose_sum_positive(drained_arr, burned_arr)

            items_drained = _get_items_for_dataset_interval(d, ds_drained_emis, interval)
            if items_drained:
                out_dir_base_sum = posixpath.join(items_drained["global_dir"], "display", interval, "drained_plus_burned")
            else:
                out_dir_base_sum = posixpath.join(OUTPUT_ROOT, "display", interval, "drained_plus_burned")
            out_dir_local_sum = to_local_mirror(out_dir_base_sum, DISPLAY_OUT_ROOT)
            ensure_dir(out_dir_local_sum)

            logger.info("[Stage 02] Rendering drained+burned (sum) interval=%s", interval)

            created = []
            masked_positive = np.ma.masked_where(combined <= 0, combined)
            land_stats_mask = None
            if shp is not None:
                land_stats_mask = _land_stats_mask_from_shp(shp, masked_positive.shape, transform_d)

            # Colormap
            if cmap_choice.lower() == "custom":
                colors_mpl = _rgb_palette_to_mpl(emissions_palette)
                cmap = LinearSegmentedColormap.from_list("custom", colors_mpl)
            else:
                try:
                    cmap = cm.get_cmap(cmap_choice)
                except Exception:
                    cmap = cm.get_cmap("inferno")
            try:
                cmap = cmap.with_extremes(
                    under=_rgb_to_mpl(cn.ocean_color) + (0.15,),
                    over=_rgb_to_mpl(cn.ocean_color) + (0.15,),
                )
            except Exception:
                cmap.set_under(_rgb_to_mpl(cn.ocean_color))
                cmap.set_over(_rgb_to_mpl(cn.ocean_color))
            cmap.set_bad((0, 0, 0, 0))

            legend_title = "Gross greenhouse gas emissions (drained + burned)\nkt CO$_2$e yr$^{-1}$"
            for p in profile_cfgs:
                vmin, vmax, steps_levels, norm = _compute_domain_and_norm(
                    masked_positive, p["stretch"], p["clip_lo"], p["clip_hi"], p.get("floor", None),
                    land_stats_mask, stats_scope, p.get("steps")
                )
                fig, ax = plt.subplots(figsize=cn.panel_dims)
                _plot_country_layers(ax, shp)
                img = _draw_frame(ax, [bounds_d.left, bounds_d.right, bounds_d.bottom, bounds_d.top],
                                  masked_positive, cmap, norm, cn.ocean_color, interpolation)
                _legend(fig, img, legend_title, steps_levels=steps_levels)

                base_out = Path(out_dir_local_sum) / "drained_plus_burned"  # base has NO interval
                suffix = _profile_suffix(p["name"], p["clip_lo"], p["clip_hi"], p.get("steps"))
                out_img = _save_single(ax, base_out, interval, suffix)
                created.append(out_img)

            if upload_to_s3:
                _upload_display_outputs(created, Path(out_dir_local_sum), out_dir_base_sum, logger)

        # -------------------------
        # 5) 4-panel maps (drained and/or burned), if requested
        # -------------------------
        if aoi_list and (make_four_panel_drained or make_four_panel_burned):
            # Shared cmap for four-panel
            if cmap_choice.lower() == "custom":
                colors_mpl = _rgb_palette_to_mpl(emissions_palette)
                cmap4 = LinearSegmentedColormap.from_list("custom", colors_mpl)
            else:
                try:
                    cmap4 = cm.get_cmap(cmap_choice)
                except Exception:
                    cmap4 = cm.get_cmap("inferno")
            try:
                cmap4 = cmap4.with_extremes(
                    under=_rgb_to_mpl(cn.ocean_color) + (0.15,),
                    over=_rgb_to_mpl(cn.ocean_color) + (0.15,),
                )
            except Exception:
                cmap4.set_under(_rgb_to_mpl(cn.ocean_color))
                cmap4.set_over(_rgb_to_mpl(cn.ocean_color))
            cmap4.set_bad((0, 0, 0, 0))

            # Four-panel for drained
            if make_four_panel_drained and ds_drained_emis:
                cand_drained = _candidates_for_dataset_interval(
                    d, ds_drained_emis, interval, read_from_s3, resolved_base_url,
                    run_name, pixel_resolution, date_tag, res_label
                )
                work_dir = Path(DISPLAY_OUT_ROOT) / "four_panel" / "_reproj_cache"; ensure_dir(work_dir)
                arr_d, bounds_d, transform_d = _open_array_from_candidates(cand_drained, cn.Robinson_crs, work_dir)
                masked_d = np.ma.masked_where(arr_d <= 0, arr_d)
                land_stats_mask = _land_stats_mask_from_shp(shp, masked_d.shape, transform_d)

                items_drained = _get_items_for_dataset_interval(d, ds_drained_emis, interval)
                if items_drained:
                    out_dir_base_fp = posixpath.join(items_drained["global_dir"], "display", interval, f"{ds_drained_emis}_fourpanel")
                else:
                    out_dir_base_fp = posixpath.join(OUTPUT_ROOT, "display", interval, f"{ds_drained_emis}_fourpanel")
                out_dir_local_fp = to_local_mirror(out_dir_base_fp, DISPLAY_OUT_ROOT)
                ensure_dir(out_dir_local_fp)

                out_img = _four_panel(
                    arr=arr_d, bounds=bounds_d, shapefile_gdf=shp, land_stats_mask=land_stats_mask,
                    stats_scope=stats_scope, profile_cfg=four_panel_cfg, cmap=cmap4, interpolation=interpolation,
                    ocean_color=cn.ocean_color, aoi_boxes_ll=aoi_list, legend_title="Drained emissions\nkt CO$_2$e yr$^{-1}$",
                    out_dir_local=out_dir_local_fp, out_name_base=ds_drained_emis,  # base has NO interval
                    label_value=interval, logger=logger
                )
                if upload_to_s3:
                    _upload_display_outputs([out_img], Path(out_dir_local_fp), out_dir_base_fp, logger)

            # Four-panel for burned
            if make_four_panel_burned and ds_burned_emis:
                cand_burned = _candidates_for_dataset_interval(
                    d, ds_burned_emis, interval, read_from_s3, resolved_base_url,
                    run_name, pixel_resolution, date_tag, res_label
                )
                work_dir = Path(DISPLAY_OUT_ROOT) / "four_panel" / "_reproj_cache"; ensure_dir(work_dir)
                arr_b, bounds_b, transform_b = _open_array_from_candidates(cand_burned, cn.Robinson_crs, work_dir)
                masked_b = np.ma.masked_where(arr_b <= 0, arr_b)
                land_stats_mask = _land_stats_mask_from_shp(shp, masked_b.shape, transform_b)

                items_burned = _get_items_for_dataset_interval(d, ds_burned_emis, interval)
                if items_burned:
                    out_dir_base_fp = posixpath.join(items_burned["global_dir"], "display", interval, f"{ds_burned_emis}_fourpanel")
                else:
                    out_dir_base_fp = posixpath.join(OUTPUT_ROOT, "display", interval, f"{ds_burned_emis}_fourpanel")
                out_dir_local_fp = to_local_mirror(out_dir_base_fp, DISPLAY_OUT_ROOT)
                ensure_dir(out_dir_local_fp)

                out_img = _four_panel(
                    arr=arr_b, bounds=bounds_b, shapefile_gdf=shp, land_stats_mask=land_stats_mask,
                    stats_scope=stats_scope, profile_cfg=four_panel_cfg, cmap=cmap4, interpolation=interpolation,
                    ocean_color=cn.ocean_color, aoi_boxes_ll=aoi_list, legend_title="Burned emissions\nkt CO$_2$e yr$^{-1}$",
                    out_dir_local=out_dir_local_fp, out_name_base=ds_burned_emis,  # base has NO interval
                    label_value=interval, logger=logger
                )
                if upload_to_s3:
                    _upload_display_outputs([out_img], Path(out_dir_local_fp), out_dir_base_fp, logger)

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render global display assets (gross emissions + composites + 4-panels)."
    )
    parser.add_argument("--date_tag", required=True)
    parser.add_argument("--run_name", default="ogh_sensitivity_1km")
    parser.add_argument("-p", "--pixel_resolution", default="40000_pixels")
    parser.add_argument("--target_deg", type=float, default=DEFAULT_TARGET_DEG)
    parser.add_argument("--read_from_s3", action="store_true")
    parser.add_argument(
        "--model_version",
        default=DEFAULT_MODEL_VERSION,
        help="Model version string (underscore separated) used to build S3 paths.",
    )
    parser.add_argument(
        "--outputs_root",
        default=OUTPUT_ROOT,
        help="Root S3 directory for model outputs.",
    )
    parser.add_argument(
        "--base_url",
        default=None,
        help="Optional override for the versioned base URL containing per-tile rasters.",
    )
    parser.add_argument(
        "--outputs_base",
        default=None,
        help="Optional override for the destination of aggregated rasters.",
    )
    parser.add_argument(
        "--local_display_only",
        action="store_true",
        help="Skip uploading rendered displays to S3 (local mirror only).",
    )

    # Profiles (asinh, linear, steps)
    parser.add_argument("--profiles", nargs="+", default=["all"],
                        help="Profiles to render for emissions layers: any of {asinh, linear, steps} or 'all'.")
    parser.add_argument("--clip_lo", type=float, default=None,
                        help="Lower percentile for ALL profiles (0–100).")
    parser.add_argument("--clip_hi", type=float, default=None,
                        help="Upper percentile for ALL profiles (0–100).")
    parser.add_argument("--floor", type=float, default=None,
                        help="Absolute minimum vmin for 'asinh' (data units). If omitted, uses 1%% of vmax.")
    parser.add_argument("--steps", type=float, nargs="+", default=None,
                        help="Breakpoints for 'steps' profile (e.g., 60 80 90 95 98 99 99.5 99.9).")
    parser.add_argument("--interpolation", choices=["nearest", "bilinear", "none"], default="nearest",
                        help="imshow interpolation.")
    parser.add_argument("--stats_scope", choices=["land", "all"], default="land",
                        help="Use only land pixels for percentile statistics (recommended).")
    parser.add_argument("--min_cluster_pixels", type=int, default=0,
                        help="Remove connected components < N pixels (0=off).")
    parser.add_argument("--smooth_sigma", type=float, default=0.0,
                        help="Gaussian sigma in pixels for gentle smoothing (0=off).")
    parser.add_argument("--cmap", choices=["custom", "inferno", "magma", "plasma", "viridis", "cividis"],
                        default="custom", help="Colormap for emissions layers.")

    # Dataset names
    parser.add_argument("--ds_drained_emis", type=str, default=None,
                        help="Dataset name for drained emissions.")
    parser.add_argument("--ds_burned_emis", type=str, default=None,
                        help="Dataset name for burned emissions.")
    parser.add_argument("--ds_drained_state", type=str, default=None,
                        help="Dataset name for drained-state raster used to derive binary extent.")

    # Four-panel controls
    parser.add_argument("--make_four_panel_drained", action="store_true",
                        help="Create 4-panel map for drained emissions.")
    parser.add_argument("--make_four_panel_burned", action="store_true",
                        help="Create 4-panel map for burned emissions.")
    parser.add_argument("--four_panel_profile", choices=["asinh", "linear", "steps"], default="asinh",
                        help="Profile to use for 4-panel maps.")
    parser.add_argument("--aoi", dest="aoi_boxes", nargs="+", default=None,
                        help="AOIs as 'A:minlon,minlat,maxlon,maxlat' (repeat 3 times for A/B/C).")

    # Extent-only shortcut
    parser.add_argument("--only_extent", action="store_true",
                        help="Only render the drained/undrained binary extent map.")

    args = parser.parse_args()
    upload_displays = not args.local_display_only

    display_main(
        date_tag=args.date_tag,
        read_from_s3=args.read_from_s3,
        run_name=args.run_name,
        pixel_resolution=args.pixel_resolution,
        target_deg=args.target_deg,
        model_version=args.model_version,
        outputs_root=args.outputs_root,
        base_url=args.base_url,
        outputs_base=args.outputs_base,
        upload_to_s3=upload_displays,
        profiles_arg=args.profiles,
        clip_lo=args.clip_lo,
        clip_hi=args.clip_hi,
        floor=args.floor,
        steps=args.steps,
        interpolation=args.interpolation,
        stats_scope=args.stats_scope,
        min_cluster_pixels=args.min_cluster_pixels,
        smooth_sigma=args.smooth_sigma,
        cmap_choice=args.cmap,
        ds_drained_emis=args.ds_drained_emis,
        ds_burned_emis=args.ds_burned_emis,
        ds_drained_state=args.ds_drained_state,
        make_four_panel_drained=args.make_four_panel_drained,
        make_four_panel_burned=args.make_four_panel_burned,
        four_panel_profile=args.four_panel_profile,
        aoi_boxes=args.aoi_boxes,
        only_extent=args.only_extent,
    )


if __name__ == "__main__":
    main()
