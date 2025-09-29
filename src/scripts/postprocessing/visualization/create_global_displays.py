"""Stage 02: render Robinson-projected JPEGs and GIFs from aggregated global rasters.


# Read mosaics directly from S3 and upload rendered assets back to S3:
python -m src.scripts.postprocessing.visualization.create_global_displays \
  --date_tag 20250923 --read_from_s3 --run_name ogh_sensitivity_1km \
  --model_version 0_8_0

# Render locally only (no S3 download/upload) into DISPLAY_OUT_ROOT:
python -m src.scripts.postprocessing.visualization.create_global_displays \
  --date_tag 20250923 --run_name ogh_sensitivity_1km --model_version 0_8_0 \
  --local_display_only

"""


from __future__ import annotations

import argparse
import math
import posixpath
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from PIL import Image
import geopandas as gpd

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
            )
    return out_tif


def _compute_percentile_breaks(data, percentiles, ignore_zero=True):
    d = data[data != 0] if ignore_zero else data
    if d.size == 0:
        return np.array([0.0 for _ in percentiles], dtype=float)
    return np.percentile(d, percentiles)


def _mask_and_norm(data: np.ndarray, mode: str, breaks: np.ndarray):
    """Return masked array, normaliser, and legend tuple for ``mode``."""

    if mode == "diverging":
        vmin, vcenter, vmax = breaks[0], 0.0, breaks[-1]
        masked = np.ma.masked_where(data == 0, data)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        return masked, norm, (vmin, vcenter, vmax)
    if mode == "emissions":
        vmin, vmax = breaks[0], breaks[-1]
        masked = np.ma.masked_where(data <= 0, data)
        norm = Normalize(vmin=vmin, vmax=vmax)
        return masked, norm, (vmin, vmax)
    if mode == "removals":
        vmin, vmax = breaks[0], breaks[-1]
        masked = np.ma.masked_where(data >= 0, data)
        norm = Normalize(vmin=vmin, vmax=vmax)
        return masked, norm, (vmin, vmax)

    vmin, vmax = breaks[0], breaks[-1]
    masked = np.ma.masked_where(data == 0, data)
    norm = Normalize(vmin=vmin, vmax=vmax)
    return masked, norm, (vmin, vmax)


def _choose_recipe(dataset_name: str):
    lower = dataset_name.lower()
    if "net" in lower:
        return "diverging", "Net greenhouse gas flux\nkt CO$_2$e yr$^{-1}$"
    if "removal" in lower or "sink" in lower:
        return "removals", "Gross CO$_2$ removals\nkt CO$_2$ yr$^{-1}$"
    return "emissions", "Gross greenhouse gas emissions\nkt CO$_2$e yr$^{-1}$"


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


def _legend(ax_or_fig, img, mode, vtuple, title_text, data_min, data_max):
    fig = ax_or_fig.figure if hasattr(ax_or_fig, "figure") else ax_or_fig
    cax = fig.add_axes(cn.colorbar_dimensions)
    cb = plt.colorbar(img, cax=cax, orientation="vertical")

    if mode == "diverging":
        vmin, vcenter, vmax = vtuple
        tick_labels = [
            f"< {math.ceil((data_min/1e3)*100)/100:.0f}  (sink)",
            "0           (neutral)",
            f"> {math.floor((data_max/1e3)*100)/100:.0f}  (source)",
        ]
        cb.set_ticks([vmin, vcenter, vmax])
        cb.set_ticklabels(tick_labels, fontsize=cn.legend_fontsize)
    else:
        vmin, vmax = vtuple
        cb.set_ticks([vmin, vmax])
        cb.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"], fontsize=cn.legend_fontsize)

    cax.text(
        0,
        1.1,
        title_text,
        fontsize=cn.legend_fontsize,
        ha="left",
        va="bottom",
        transform=cax.transAxes,
    )


def _draw_frame(ax, extent, masked, cmap, norm, ocean_rgb):
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
        zorder=2,
    )
    return img


def _save_two_versions(ax, base_out: Path, year_label: str | int) -> Tuple[Path, Path]:
    base_no_note = Path(f"{base_out}__{year_label}.jpeg")
    base_with_note = Path(f"{base_out}__{year_label}__for_pres.jpeg")

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

    plt.savefig(base_no_note, dpi=cn.dpi_jpeg, bbox_inches="tight", pad_inches=0)

    ax.text(
        0.98,
        0.04,
        cn.pres_text,
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="top",
        color="black",
    )
    plt.savefig(base_with_note, dpi=cn.dpi_jpeg, bbox_inches="tight", pad_inches=0)
    plt.close()
    return base_with_note, base_no_note


def _gif_from_frames(base_out: Path, first_label, last_label, frames: List[str]) -> List[Path]:
    if not frames:
        return []

    imgs = [Image.open(p) for p in frames]
    out_fast = Path(f"{base_out}__{first_label}_{last_label}__fast.gif")
    out_slow = Path(f"{base_out}__{first_label}_{last_label}__slow.gif")

    try:
        imgs[0].save(
            str(out_fast),
            save_all=True,
            append_images=imgs[1:],
            duration=1000,
            loop=0,
        )
        imgs[0].save(
            str(out_slow),
            save_all=True,
            append_images=imgs[1:],
            duration=2500,
            loop=0,
        )
    finally:
        for img in imgs:
            img.close()

    return [out_fast, out_slow]


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


def make_displays_for_dataset(
    dataset_name: str,
    interval: str,
    tif_path_noext_candidates: List[str],
    out_dir_local: str,
    palette_rgb: List[Tuple[int, int, int]],
    percentiles: List[int],
    shapefile_gdf: Optional[gpd.GeoDataFrame],
    out_name_base: str,
    years: Optional[List[int]] = None,
    diverging_if_zero_center: bool = False,
    target_crs: str = cn.Robinson_crs,
    logger=None,
):
    logger = logger or lu.setup_logging()
    t0 = time.time()

    ensure_dir(out_dir_local)
    work_dir = Path(out_dir_local) / "_reproj_cache"
    ensure_dir(work_dir)

    colors_mpl = _rgb_palette_to_mpl(palette_rgb)
    cmap = LinearSegmentedColormap.from_list("custom", colors_mpl)

    recipe_mode, legend_title = _choose_recipe(dataset_name)
    mode = "diverging" if diverging_if_zero_center else recipe_mode

    frames_for_gif: List[str] = []
    created_files: List[Path] = []
    labels = years or [interval]

    for label in labels:
        tif_proj, _ = _try_open_first(
            tif_path_noext_candidates, target_crs, work_dir
        )

        with rasterio.open(tif_proj) as src:
            data = src.read(1)
            bounds = src.bounds

        data = np.nan_to_num(data, nan=0.0)

        breaks = _compute_percentile_breaks(data, percentiles, ignore_zero=True)
        masked, norm, vtuple = _mask_and_norm(data, mode, breaks)

        fig, ax = plt.subplots(figsize=cn.panel_dims)
        _plot_country_layers(ax, shapefile_gdf)
        img = _draw_frame(
            ax,
            [bounds.left, bounds.right, bounds.bottom, bounds.top],
            masked,
            cmap,
            norm,
            cn.ocean_color,
        )

        comp = masked.compressed()
        if comp.size == 0:
            dmin, dmax = 0.0, 0.0
        else:
            dmin, dmax = float(comp.min()), float(comp.max())
        _legend(fig, img, mode, vtuple, legend_title, dmin, dmax)

        base_out = Path(out_dir_local) / out_name_base
        frame_with_note, frame_no_note = _save_two_versions(ax, base_out, label)
        frames_for_gif.append(str(frame_with_note))
        created_files.extend([frame_with_note, frame_no_note])

    try:
        if len(frames_for_gif) > 1 and isinstance(labels[0], int):
            gif_paths = _gif_from_frames(
                Path(out_dir_local) / out_name_base,
                labels[0],
                labels[-1],
                frames_for_gif,
            )
            created_files.extend(gif_paths)
    except Exception as e:
        logger.warning(f"GIF creation failed: {e}")

    logger.info(
        "Rendered %s %s in %s s → %s",
        dataset_name,
        interval,
        round(time.time() - t0),
        posixpath.join(out_dir_local, out_name_base),
    )

    return created_files


def display_main(
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

    d = build_download_upload_dict(
        pixel_resolution=pixel_resolution,
        run_name=run_name,
        target_deg=target_deg,
        base_url=resolved_base_url,
        output_date=date_tag,
        outputs_base=resolved_outputs_base,
    )

    shp = _maybe_load_world_boundaries(logger)

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
    removals_palette = net_palette[0:5]
    emissions_palette = net_palette[5:]

    net_percentiles = [5, 25, 50, 75, 89, 91, 92, 93, 94, 99]
    gross_percentiles = [5, 25, 50, 75, 99]

    for key, items in d.items():
        dataset = items["dataset"]
        interval = items["interval"]

        canonical_noext = posixpath.join(
            items["global_dir"], items["global_pattern"][:-4]
        )
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

        out_jpeg_dir_base = posixpath.join(
            items["global_dir"], "display", interval, dataset
        )
        out_jpeg_dir_local = to_local_mirror(out_jpeg_dir_base, DISPLAY_OUT_ROOT)
        ensure_dir(out_jpeg_dir_local)

        logger.info(
            "[Stage 02] Rendering dataset=%s interval=%s | candidates=%s",
            dataset,
            interval,
            candidates,
        )

        mode, _ = _choose_recipe(dataset)
        palette = (
            net_palette
            if mode == "diverging"
            else (emissions_palette if mode == "emissions" else removals_palette)
        )
        ptiles = net_percentiles if mode == "diverging" else gross_percentiles

        out_base = f"{dataset}__{interval}"

        created_files = make_displays_for_dataset(
            dataset_name=dataset,
            interval=interval,
            tif_path_noext_candidates=candidates,
            out_dir_local=out_jpeg_dir_local,
            palette_rgb=palette,
            percentiles=list(ptiles),
            shapefile_gdf=shp,
            out_name_base=out_base,
            years=None,
            diverging_if_zero_center=("net" in dataset.lower()),
            target_crs=cn.Robinson_crs,
            logger=logger,
        )

        if upload_to_s3:
            _upload_display_outputs(
                created_files,
                Path(out_jpeg_dir_local),
                out_jpeg_dir_base,
                logger,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render global display assets from aggregated rasters."
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
    )


if __name__ == "__main__":
    main()