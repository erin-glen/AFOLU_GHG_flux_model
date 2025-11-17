"""
Stage 02: Render Robinson-projected JPEGs for aggregated global rasters.

This version:
  - Preserves source nodata during reprojection (important for UInt8 255 nodata).
  - Treats aggregated `drained_state` as binary (UInt8): 1=peat_drained, 0=peat_undrained, 255=nodata.
    If a non-binary `drained_state` is detected (e.g., un-aggregated tiles), it auto-falls back
    to categorical rendering using zonal constants.
  - Uses land-only percentile statistics by default (stable on sparse signals).
  - Renders three profiles for gross layers: asinh, linear, and stepped percentiles.
  - Optional speckle cleanup (min connected pixels) and light Gaussian smoothing.
  - Adds a derived gross layer: drained + burned totals.
  - Legend units for gross layers are in **Mg yr⁻¹** (matches inputs; no rescaling here).

Examples
--------
# Read mosaics directly from S3 and upload rendered assets back to S3:
python -m src.scripts.postprocessing.visualization.create_global_displays \
  --date_tag 20250923 --read_from_s3 --run_name ogh_sensitivity_1km \
  --model_version 0_8_0

# Render locally only (no S3 upload) into DISPLAY_OUT_ROOT:
python -m src.scripts.postprocessing.visualization.create_global_displays \
  --date_tag 20250923 --run_name ogh_sensitivity_1km --model_version 0_8_0 \
  --local_display_only
"""

from __future__ import annotations

import argparse
import posixpath
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import (
    LinearSegmentedColormap,
    Normalize,
    BoundaryNorm,
    FuncNorm,
    ListedColormap,
)
try:  # mpl >= 3.6
    from matplotlib.colors import AsinhNorm  # type: ignore
except Exception:  # pragma: no cover
    AsinhNorm = None
from matplotlib.ticker import ScalarFormatter
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import mapping

# Optional smoothing/cleanup (graceful fallback if SciPy missing)
try:
    from scipy.ndimage import gaussian_filter as ndi_gaussian_filter, label as ndi_label
except Exception:  # pragma: no cover
    ndi_gaussian_filter, ndi_label = None, None

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

# ----------------------------
# Configuration
# ----------------------------

SUM_DATASET_NAME = "drained_plus_burned_total_Mg_CO2e_pixel_yr"
DRAINED_DATASET = "drained_total_Mg_CO2e_pixel_yr"
BURNED_DATASET  = "burned_total_Mg_CO2e_pixel_yr"

# ----------------------------
# Utility helpers
# ----------------------------

def _split_s3_url(s3_url: str) -> Tuple[str, str]:
    if not s3_url.startswith("s3://"):
        raise ValueError(f"Expected s3:// URL, received: {s3_url}")
    bucket_key = s3_url[len("s3://"):]
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
        logger.warning("Display output directory is not an S3 path; skipping upload: %s", dest_s3_dir)
        return

    bucket, prefix = _split_s3_url(dest_s3_dir.rstrip("/") + "/")
    prefix = prefix.rstrip("/")

    for local_path in created_files:
        try:
            rel_path = local_path.relative_to(out_dir_local)
        except ValueError:
            logger.warning("Skipping upload for %s because it is outside %s", local_path, out_dir_local)
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
    """Reproject to ``target_crs`` if ``out_tif`` does not exist; **preserve** nodata."""
    if Path(out_tif).exists():
        return out_tif
    with rasterio.open(in_tif) as src:
        src_nodata = src.nodata
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
            nodata=src_nodata,            # preserve nodata
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
                src_nodata=src_nodata,      # preserve nodata in reprojection
                dst_nodata=src_nodata,
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

    cax.text(0, 1.1, title_text, fontsize=cn.legend_fontsize,
             ha="left", va="bottom", transform=cax.transAxes)


def _format_category_label(label: str) -> str:
    pretty = label.replace("__", " / ").replace("_", " ")
    return pretty.title()


def _build_categorical_config() -> Dict[str, Dict[str, object]]:
    """
    Categorical configuration for display. For aggregated global rasters:
      - drained_state is binary (UInt8): 1=peat_drained, 0=peat_undrained, nodata=255.
      - burned_state remains a multi-code categorical (use zc if available).
    If later we detect drained_state is actually non-binary, we will rebuild its mapping on the fly.
    """
    base: Dict[str, Dict[str, object]] = {
        "drained_state": {
            "legend_title": "Drainage State (Peat Only)",
            "code_to_label": {1: "peat_drained", 0: "peat_undrained"},
            "order": ["peat_drained", "peat_undrained"],
            "cmap": "tab20",
            "nodata": 255,  # UInt8 nodata used by Stage 01 aggregation
        },
        "burned_state": {
            "legend_title": "Burned State",
            "code_to_label": {},
            "order": [],
            "cmap": "tab20",
            "nodata": 0,
        },
    }

    # Keep burned_state labels from zonal_constants if available.
    try:
        from src.scripts.zonal_statistics import zonal_constants as zc  # type: ignore
        burned_lookup: Dict[int, str] = {}
        for code, meaning in zc.BURNED_STATE_NODE_MEANINGS.items():
            try:
                burned_lookup[int(code)] = meaning
            except ValueError:
                continue
        base["burned_state"]["code_to_label"] = burned_lookup
    except Exception:
        pass

    return base

CATEGORICAL_DATASET_CONFIG = _build_categorical_config()

# ----------------------------
# Land mask for stats
# ----------------------------

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

# ----------------------------
# Norms, profiles, palettes
# ----------------------------

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


def _profile_suffix(name: str, clip_lo: float, clip_hi: float, steps: Optional[List[float]] = None):
    def fmt_p(p):
        # 99.5 -> 99p5 for filenames
        return str(p).replace(".", "p")
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
    Build the list of profile configs. If overrides are provided, they apply to ALL selected profiles.
    (Only 'asinh', 'linear', and 'steps' are supported.)
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

# ----------------------------
# Core rendering (gross / categorical)
# ----------------------------

def _draw_frame(ax, extent, masked, cmap, norm, ocean_rgb, interpolation: str):
    ax.set_facecolor(_rgb_to_mpl(ocean_rgb))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xticklabels([]); ax.set_yticklabels([])
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
    out_path = Path(f"{base_out}__{year_label}__{profile_suffix}.jpeg")
    ax.text(0.5, 0.07, str(year_label), transform=ax.transAxes,
            ha="center", va="top", fontsize=18, weight="bold", color="black")
    plt.savefig(out_path, dpi=cn.dpi_jpeg, bbox_inches="tight", pad_inches=0)
    plt.close()
    return out_path


def _categorical_legend(fig_or_ax, img, labels: List[str], title_text: str):
    fig = fig_or_ax.figure if hasattr(fig_or_ax, "figure") else fig_or_ax
    cax = fig.add_axes(cn.colorbar_dimensions)
    ticks = np.arange(len(labels)) if labels else np.array([0])
    cb = plt.colorbar(img, cax=cax, orientation="vertical", ticks=ticks)
    if labels:
        cb.ax.set_yticklabels(labels)
    cb.ax.tick_params(labelsize=cn.legend_fontsize - 1)
    cax.text(0, 1.1, title_text, fontsize=cn.legend_fontsize,
             ha="left", va="bottom", transform=cax.transAxes)


def _map_categorical_values(
    data: np.ndarray,
    config: Dict[str, object],
    nodata: Optional[int | float],
) -> Tuple[np.ma.MaskedArray, List[str]]:
    mapping = config.get("code_to_label", {})
    nodata_value = config.get("nodata", nodata)

    valid_mask = np.ones(data.shape, dtype=bool)
    if nodata_value is not None:
        valid_mask &= data != nodata_value

    if not np.any(valid_mask):
        return np.ma.masked_all(data.shape), []

    unique_codes = np.unique(data[valid_mask])
    labels_for_code: Dict[int, str] = {}
    for code in unique_codes:
        code_int = int(code)
        label = None
        if isinstance(mapping, dict):
            label = mapping.get(code_int) or mapping.get(str(code_int))  # type: ignore[index]
        if label is None:
            label = str(code_int)
        labels_for_code[code_int] = label

    if not labels_for_code:
        return np.ma.masked_all(data.shape), []

    order = config.get("order")
    ordered_labels: List[str] = []
    if isinstance(order, list):
        for label in order:
            if label in labels_for_code.values() and label not in ordered_labels:
                ordered_labels.append(label)

    for label in sorted(set(labels_for_code.values()) - set(ordered_labels)):
        ordered_labels.append(label)

    label_to_index = {label: idx for idx, label in enumerate(ordered_labels)}
    mapped = np.full(data.shape, -1, dtype=np.int16)
    for code_int, label in labels_for_code.items():
        idx = label_to_index[label]
        mapped[data == code_int] = idx

    masked = np.ma.masked_equal(mapped, -1)
    return masked, ordered_labels


def _is_binary_drained_state(data: np.ndarray, nodata: Optional[int | float]) -> bool:
    """Return True if non-nodata set ⊆ {0,1}."""
    if data.size == 0:
        return True
    mask = np.ones(data.shape, dtype=bool)
    if nodata is not None:
        mask &= data != nodata
    vals = np.unique(data[mask])
    return set(map(int, vals.tolist())) <= {0, 1}


def make_categorical_displays_for_dataset(
    *,
    dataset_name: str,
    interval: str,
    tif_path_noext_candidates: List[str],
    out_dir_local: str,
    shapefile_gdf: Optional[gpd.GeoDataFrame],
    out_name_base: str,
    target_crs: str,
    config: Dict[str, object],
    logger=None,
) -> List[Path]:
    """Render a single categorical map for ``dataset_name`` using discrete colors."""
    logger = logger or lu.setup_logging()
    ensure_dir(out_dir_local)
    work_dir = Path(out_dir_local) / "_reproj_cache"
    ensure_dir(work_dir)

    created_files: List[Path] = []

    try:
        tif_proj, kind = _try_open_first(tif_path_noext_candidates, target_crs, work_dir)
        logger.info("Opened categorical raster (%s) for dataset=%s interval=%s", kind, dataset_name, interval)
    except Exception:
        logger.exception("Failed to open categorical raster for dataset=%s interval=%s", dataset_name, interval)
        return created_files

    with rasterio.open(tif_proj) as src:
        data = src.read(1)
        bounds = src.bounds
        nodata_val = src.nodata

    # If this is drained_state but not binary, rebuild mapping from zonal constants (fallback)
    cfg = dict(config)
    if dataset_name == "drained_state" and not _is_binary_drained_state(data, nodata_val):
        try:
            from src.scripts.zonal_statistics import zonal_constants as zc  # type: ignore
            drained_lookup: Dict[int, str] = {}
            for code, meaning in zc.DRAINED_STATE_NODE_MEANINGS.items():
                try:
                    # Use root label only (before any '__' suffix)
                    drained_lookup[int(code)] = meaning.split("__")[0]
                except ValueError:
                    continue
            cfg["code_to_label"] = drained_lookup
            # Prefer the source nodata if present (tiles may not use 255)
            if nodata_val is not None:
                cfg["nodata"] = int(nodata_val)
            logger.info("Drained_state detected as non-binary; using zonal-constants mapping (%d classes).",
                        len(drained_lookup))
        except Exception:
            logger.warning("Failed to import zonal_constants for non-binary drained_state; showing raw codes.")

    mapped, raw_labels = _map_categorical_values(data, cfg, nodata_val)
    if not raw_labels:
        logger.warning("No valid categories found for dataset=%s interval=%s; skipping", dataset_name, interval)
        return created_files

    formatter = cfg.get("label_formatter")
    if not callable(formatter):
        formatter = _format_category_label
    display_labels = [formatter(label) for label in raw_labels]

    cmap_name = str(cfg.get("cmap", "tab20"))
    base_cmap = cm.get_cmap(cmap_name, max(len(display_labels), 1))
    colors = base_cmap(np.linspace(0, 1, base_cmap.N))
    cmap = ListedColormap(colors)
    boundaries = np.arange(-0.5, len(display_labels) + 0.5, 1)
    norm = BoundaryNorm(boundaries, cmap.N)

    fig, ax = plt.subplots(figsize=cn.panel_dims)
    _plot_country_layers(ax, shapefile_gdf)
    img = _draw_frame(
        ax,
        [bounds.left, bounds.right, bounds.bottom, bounds.top],
        mapped,
        cmap,
        norm,
        cn.ocean_color,
        interpolation="nearest",
    )

    legend_title = str(cfg.get("legend_title", dataset_name.replace("_", " ").title()))
    _categorical_legend(fig, img, display_labels, legend_title)

    base_out = Path(out_dir_local) / out_name_base
    out_img = _save_single(ax, base_out, interval, "prof-categorical")
    created_files.append(out_img)

    logger.info("Rendered categorical dataset=%s interval=%s (categories=%d)",
                dataset_name, interval, len(display_labels))
    return created_files


def _try_open_first(paths_noext: List[str], target_crs: str, work_dir: Path) -> Tuple[str, str]:
    """Attempt to reproject the first available candidate. Returns (reprojected_tif_path, input_kind)."""
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
    raise RuntimeError(f"Could not open any candidate raster: {paths_noext}. Last error: {last_error}")


def _load_or_project_raster(base_tif_noext: str, target_crs: str, work_dir: Path) -> str:
    tif_unproj = base_tif_noext + ".tif"
    tif_proj = work_dir / (Path(base_tif_noext).name + "_reproj.tif")
    return _reproject_if_needed(str(tif_proj), tif_unproj, target_crs)


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
) -> List[Path]:
    """
    Render **multiple profiles** (asinh, linear, steps) for a single dataset/interval.
    Assumes **gross emissions**: masks data <= 0 (transparent).
    """
    logger = logger or lu.setup_logging()
    t0 = time.time()

    ensure_dir(out_dir_local)
    work_dir = Path(out_dir_local) / "_reproj_cache"
    ensure_dir(work_dir)

    # Colormap: custom warm ramp, or a perceptual sequential ramp
    if cmap_choice.lower() == "custom":
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
        emissions_palette = net_palette[5:]
        colors_mpl = _rgb_palette_to_mpl(emissions_palette)
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
        # Older Matplotlib
        cmap.set_under(_rgb_to_mpl(cn.ocean_color))
        cmap.set_over(_rgb_to_mpl(cn.ocean_color))
    cmap.set_bad((0, 0, 0, 0))

    created_files: List[Path] = []
    labels = years or [interval]

    # Legend units reflect data units (Mg yr^-1)
    legend_title = (
        "Gross greenhouse gas emissions\nMg CO$_2$e yr$^{-1}$"
        if "co2" not in dataset_name.lower()
        else "Gross CO$_2$ emissions\nMg CO$_2$ yr$^{-1}$"
    )

    # Load / reproject once
    tif_proj, _ = _try_open_first(tif_path_noext_candidates, target_crs, work_dir)
    with rasterio.open(tif_proj) as src:
        data_full = src.read(1)
        bounds = src.bounds
        transform = src.transform

    # Prepare data once
    data_full = np.nan_to_num(data_full, nan=0.0)
    masked_positive = np.ma.masked_where(data_full <= 0, data_full)

    # Land mask for STATS (not for rendering)
    land_stats_mask = _land_stats_mask_from_shp(shapefile_gdf, masked_positive.shape, transform)

    for label in labels:
        for p in profiles:
            name = p["name"]
            stretch = p["stretch"]
            clip_lo = float(p["clip_lo"])
            clip_hi = float(p["clip_hi"])
            floor = p.get("floor", None)
            steps = p.get("steps", None)

            # Values used for percentile statistics
            if land_stats_mask is not None and stats_scope == "land":
                stats_values = masked_positive[land_stats_mask].compressed()
            else:
                stats_values = masked_positive.compressed()

            # Compute domain (vmin/vmax) from stats_values
            if stats_values.size == 0:
                vmin, vmax = 1.0, 1.0
            else:
                lo = float(np.percentile(stats_values, clip_lo))
                hi = float(np.percentile(stats_values, clip_hi))
                hi = max(hi, lo * 1.01)  # ensure span
                if stretch in ("asinh",):  # needs positive floor
                    floor_eff = (0.01 * hi) if floor is None else float(floor)
                    vmin = max(lo, floor_eff, 1e-12)
                else:  # linear/steps
                    vmin = max(lo, 0.0)
                vmax = max(hi, vmin * 1.01)

            # Steps: compute boundaries from percentiles in LOG domain (spreads top tail)
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

            # Build plotting array with optional cleanup/smoothing
            plot_array = masked_positive

            # (a) Remove tiny isolated blobs on the positive mask
            if min_cluster_pixels and min_cluster_pixels > 1 and ndi_label is not None:
                pos_mask = (~plot_array.mask).astype(np.uint8)
                labs, nlab = ndi_label(pos_mask)
                if nlab and nlab > 0:
                    counts = np.bincount(labs.ravel())
                    small_ids = np.where(counts < min_cluster_pixels)[0]
                    if small_ids.size > 0:
                        small_mask = np.isin(labs, small_ids)
                        plot_array = np.ma.masked_where(small_mask | (data_full <= 0), data_full)

            # (b) Gentle Gaussian that respects mask (normalized)
            if smooth_sigma and smooth_sigma > 0 and ndi_gaussian_filter is not None:
                data = plot_array.filled(0.0)
                w = (~plot_array.mask).astype(float)
                num = ndi_gaussian_filter(data, smooth_sigma)
                den = ndi_gaussian_filter(w,    smooth_sigma)
                sm = num / np.maximum(den, 1e-9)
                plot_array = np.ma.array(sm, mask=(w == 0))

            # Draw
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

            base_out = Path(out_dir_local) / out_name_base
            suffix = _profile_suffix(name, clip_lo, clip_hi, steps)
            out_img = _save_single(ax, base_out, label, suffix)
            created_files.append(out_img)

            # Tiny diagnostic in logs
            pos_count = int(masked_positive.count())
            hi_clipped = int((masked_positive > vmax).sum())
            logger.info(
                "Profile=%s | vmin=%.3g vmax=%.3g | positives=%d | clipped_hi=%.2f%%",
                name, vmin, vmax, pos_count, 100.0 * hi_clipped / max(1, pos_count)
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

# ----------------------------
# SUM (drained + burned) helper
# ----------------------------

def _render_sum_drained_burned(
    *,
    interval: str,
    drained_candidates: List[str],
    burned_candidates: List[str],
    out_jpeg_dir_base: str,
    read_from_s3: bool,
    shapefile_gdf: Optional[gpd.GeoDataFrame],
    profiles: List[Dict],
    interpolation: str,
    stats_scope: str,
    min_cluster_pixels: int,
    smooth_sigma: float,
    cmap_choice: str,
    resolved_outputs_base: str,
    res_label: str,
    upload_to_s3: bool,
    logger,
) -> None:
    """Create a temporary reprojected sum GeoTIFF and render with gross profiles."""
    out_jpeg_dir_local = to_local_mirror(out_jpeg_dir_base, DISPLAY_OUT_ROOT)
    ensure_dir(out_jpeg_dir_local)
    work_dir = Path(out_jpeg_dir_local) / "_reproj_cache"
    ensure_dir(work_dir)

    # Prepare candidate lists (already noext); respect /vsis3/ if requested
    dr_cand = [gdalize_s3_url(p) for p in drained_candidates] if read_from_s3 else drained_candidates
    bu_cand = [gdalize_s3_url(p) for p in burned_candidates] if read_from_s3 else burned_candidates

    # Open both in the same Robinson reprojection (to local cache)
    tif_dr, _ = _try_open_first(dr_cand, cn.Robinson_crs, work_dir)
    tif_bu, _ = _try_open_first(bu_cand, cn.Robinson_crs, work_dir)

    with rasterio.open(tif_dr) as s1, rasterio.open(tif_bu) as s2:
        a1 = s1.read(1)
        if (s2.width, s2.height) != (s1.width, s1.height) or s2.transform != s1.transform or s2.crs != s1.crs:
            # Reproject burned to drained grid if needed (should rarely happen)
            a2 = np.zeros_like(a1, dtype=np.float32)
            reproject(
                source=rasterio.band(s2, 1),
                destination=a2,
                src_transform=s2.transform,
                src_crs=s2.crs,
                dst_transform=s1.transform,
                dst_crs=s1.crs,
                resampling=Resampling.nearest,
                src_nodata=0,
                dst_nodata=0,
            )
        else:
            a2 = s2.read(1)

        sum_arr = np.nan_to_num(a1, nan=0.0, copy=False) + np.nan_to_num(a2, nan=0.0, copy=False)

        # Write a temp GeoTIFF for the sum (so the standard renderer can reuse its flow)
        profile = s1.profile.copy()
        profile.update(nodata=0, dtype="float32", compress=profile.get("compress", "deflate"))
        sum_tif = work_dir / f"{res_label}_global__{SUM_DATASET_NAME}__{interval}_reproj.tif"
        with rasterio.open(sum_tif, "w", **profile) as dst:
            dst.write(sum_arr.astype(np.float32), 1)

    # Render
    created = make_displays_for_dataset(
        dataset_name=SUM_DATASET_NAME,
        interval=interval,
        tif_path_noext_candidates=[str(sum_tif)[:-4]],  # base without .tif
        out_dir_local=out_jpeg_dir_local,
        palette_rgb=[],  # ignored when cmap_choice != 'custom'
        shapefile_gdf=shapefile_gdf,
        out_name_base=f"{SUM_DATASET_NAME}__{interval}",
        years=None,
        target_crs=cn.Robinson_crs,
        profiles=profiles,
        interpolation=interpolation,
        stats_scope=stats_scope,
        min_cluster_pixels=min_cluster_pixels,
        smooth_sigma=smooth_sigma,
        cmap_choice="custom",  # use the warm ramp by default
        logger=logger,
    )

    if upload_to_s3:
        _upload_display_outputs(created, Path(out_jpeg_dir_local), out_jpeg_dir_base, logger)

# ----------------------------
# Orchestration
# ----------------------------

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
    # visualization controls
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
    include_sum_layer: bool = True,
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

    # Build job list from aggregated outputs
    d = build_download_upload_dict(
        pixel_resolution=pixel_resolution,
        run_name=run_name,
        target_deg=target_deg,
        base_url=resolved_base_url,
        output_date=date_tag,
        outputs_base=resolved_outputs_base,
    )

    shp = None
    try:
        # best-effort: boundaries for land stats & drawing
        rp = Path(getattr(cn, "reprojected_shapefile_path", ""))
        op = Path(getattr(cn, "original_shapefile_path", ""))
        if rp.exists():
            shp = gpd.read_file(rp).to_crs(cn.Robinson_crs)
        elif op.exists():
            shp = gpd.read_file(op).to_crs(cn.Robinson_crs)
    except Exception as e:
        logger.warning(f"World boundaries load failed: {e}")

    # Profiles
    profiles = _select_profiles(
        profiles_arg=profiles_arg,
        override_clip_lo=clip_lo,
        override_clip_hi=clip_hi,
        override_floor=floor,
        override_steps=steps,
    )

    # To support SUM: collect drained & burned candidates by interval
    sum_plan: Dict[str, Dict[str, List[str]]] = {}

    for key, items in d.items():
        dataset = items["dataset"]
        interval = items["interval"]

        canonical_noext = posixpath.join(items["global_dir"], items["global_pattern"][:-4])
        versioned_noext = posixpath.join(
            resolved_base_url.rstrip("/"),
            dataset,
            run_name,
            "five_year_intervals",
            interval,
            pixel_resolution,
            date_tag,
            f"{res_label}_global__{dataset}__{interval}",
        )
        candidates: List[str] = [canonical_noext, versioned_noext]
        if read_from_s3:
            candidates = [gdalize_s3_url(p) for p in candidates]

        out_jpeg_dir_base = posixpath.join(items["global_dir"], "display", interval, dataset)
        out_jpeg_dir_local = to_local_mirror(out_jpeg_dir_base, DISPLAY_OUT_ROOT)
        ensure_dir(out_jpeg_dir_local)

        # Categorical datasets
        categorical_config = CATEGORICAL_DATASET_CONFIG.get(dataset)
        if categorical_config:
            logger.info("[Stage 02] Categorical dataset=%s interval=%s | candidates=%s", dataset, interval, candidates)
            created_files = make_categorical_displays_for_dataset(
                dataset_name=dataset,
                interval=interval,
                tif_path_noext_candidates=candidates,
                out_dir_local=out_jpeg_dir_local,
                shapefile_gdf=shp,
                out_name_base=f"{dataset}__{interval}",
                target_crs=cn.Robinson_crs,
                config=categorical_config,
                logger=logger,
            )
            if upload_to_s3:
                _upload_display_outputs(created_files, Path(out_jpeg_dir_local), out_jpeg_dir_base, logger)
            # Record candidates for SUM layer if applicable
            if include_sum_layer:
                plan = sum_plan.setdefault(interval, {"drained": [], "burned": []})
                if dataset == DRAINED_DATASET:
                    plan["drained"] = [canonical_noext, versioned_noext]
                if dataset == BURNED_DATASET:
                    plan["burned"] = [canonical_noext, versioned_noext]
            continue

        # Gross emissions datasets (positive-only signal)
        if ("net" in dataset.lower()) or ("removal" in dataset.lower()) or ("sink" in dataset.lower()):
            logger.info("Skipping non-emissions dataset: %s", dataset)
            continue

        logger.info("[Stage 02] Rendering dataset=%s interval=%s | candidates=%s | profiles=%s",
                    dataset, interval, candidates, ",".join([p["name"] for p in profiles]))

        created_files = make_displays_for_dataset(
            dataset_name=dataset,
            interval=interval,
            tif_path_noext_candidates=candidates,
            out_dir_local=out_jpeg_dir_local,
            palette_rgb=[],  # ignored when cmap_choice != 'custom'
            shapefile_gdf=shp,
            out_name_base=f"{dataset}__{interval}",
            years=None,
            target_crs=cn.Robinson_crs,
            profiles=profiles,
            interpolation=interpolation,
            stats_scope=stats_scope,
            min_cluster_pixels=min_cluster_pixels,
            smooth_sigma=smooth_sigma,
            cmap_choice=cmap_choice,
            logger=logger,
        )

        if upload_to_s3:
            _upload_display_outputs(created_files, Path(out_jpeg_dir_local), out_jpeg_dir_base, logger)

        # Record candidates for SUM layer
        if include_sum_layer:
            plan = sum_plan.setdefault(interval, {"drained": [], "burned": []})
            if DRAINED_DATASET in dataset:
                plan["drained"] = [canonical_noext, versioned_noext]
            if BURNED_DATASET in dataset:
                plan["burned"] = [canonical_noext, versioned_noext]

    # Render SUM layers where both components exist
    if include_sum_layer:
        for interval, comp in sum_plan.items():
            if comp.get("drained") and comp.get("burned"):
                # Build SUM output dir shape consistent with aggregated tree
                sum_global_dir = posixpath.join(
                    resolved_outputs_base,
                    f"{res_label}_output_aggregation",
                    SUM_DATASET_NAME,
                    run_name,
                    interval,
                )
                out_jpeg_dir_base = posixpath.join(sum_global_dir, "display", interval, SUM_DATASET_NAME)
                logger.info("[Stage 02] Rendering SUM (%s) for interval=%s", SUM_DATASET_NAME, interval)

                _render_sum_drained_burned(
                    interval=interval,
                    drained_candidates=comp["drained"],
                    burned_candidates=comp["burned"],
                    out_jpeg_dir_base=out_jpeg_dir_base,
                    read_from_s3=read_from_s3,
                    shapefile_gdf=shp,
                    profiles=profiles,
                    interpolation=interpolation,
                    stats_scope=stats_scope,
                    min_cluster_pixels=min_cluster_pixels,
                    smooth_sigma=smooth_sigma,
                    cmap_choice=cmap_choice,
                    resolved_outputs_base=resolved_outputs_base,
                    res_label=res_label,
                    upload_to_s3=upload_to_s3,
                    logger=logger,
                )

# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render global display assets (gross emissions + categorical) from aggregated rasters."
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

    # Profile selection & overrides (only asinh, linear, steps)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["all"],
        help="Which profiles to render: any of {asinh, linear, steps} or 'all'. Default: all.",
    )
    parser.add_argument("--clip_lo", type=float, default=None,
                        help="Override lower percentile for ALL profiles (0–100).")
    parser.add_argument("--clip_hi", type=float, default=None,
                        help="Override upper percentile for ALL profiles (0–100).")
    parser.add_argument("--floor", type=float, default=None,
                        help="Absolute minimum vmin for 'asinh' (data units). If omitted, uses 1%% of vmax.")
    parser.add_argument("--steps", type=float, nargs="+", default=None,
                        help="Percentile breakpoints for the 'steps' profile (e.g., 60 80 90 95 98 99 99.5 99.9).")
    parser.add_argument("--interpolation", choices=["nearest", "bilinear", "none"], default="nearest",
                        help="imshow interpolation (visual smoothing only).")

    # Quality controls
    parser.add_argument("--stats_scope", choices=["land", "all"], default="land",
                        help="Use only land pixels for percentile statistics (recommended).")
    parser.add_argument("--min_cluster_pixels", type=int, default=0,
                        help="Remove connected components smaller than this many pixels (0=off).")
    parser.add_argument("--smooth_sigma", type=float, default=0.0,
                        help="Gaussian sigma in pixels for gentle smoothing (0=off).")
    parser.add_argument("--cmap", choices=["custom", "inferno", "magma", "plasma", "viridis", "cividis"],
                        default="custom", help="Colormap choice. 'custom' uses a warm ramp; others are perceptual ramps.")
    parser.add_argument("--no_sum_layer", action="store_true",
                        help="Disable rendering of the derived drained+burned sum layer.")

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
        include_sum_layer=(not args.no_sum_layer),
    )


if __name__ == "__main__":
    main()
