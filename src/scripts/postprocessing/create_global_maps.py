"""
Create global rasters at a chosen target angular resolution (e.g., 0.04°) and
publish display maps & GIFs.

Two-stage pipeline:
  Stage A: Aggregate 10×10° tiles from native resolution to --target_deg
           (with optional per-hectare → per-pixel conversion via pixel area).
  Stage B: Render Robinson-projected JPEGs (and GIFs for time series).

Option A (this script):
  • Stage B never writes to /vsis3/. It reads from S3 via /vsis3/, reprojects to a
    local cache, and writes all JPEG/GIF outputs to a local mirror path.
  • To publish displays to S3, sync the local mirror (see examples).
  • Stage A uploads canonical TIFFs when not run in local mode.

Environment variables
---------------------
DISPLAY_OUT_ROOT
    Root for local display outputs (mirrors S3 path below this root).
    Default: "/tmp/create_global_maps/display"

Examples
--------
# Aggregate only (0.04° by default):
python -m src.scripts.postprocessing.create_global_maps \
  aggregate -cn create_maps --run_name ogh_standard_model \
  --base_url s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_7_5

# Aggregate at 0.01°:
python -m src.scripts.postprocessing.create_global_maps \
  aggregate -cn create_maps --run_name ogh_standard_model --target_deg 0.01 \
  --base_url s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_7_5

# Display only (read global rasters directly in S3 via GDAL /vsis3/):
python -m src.scripts.postprocessing.create_global_maps \
  display --date_tag 20250914 --read_from_s3 --run_name ogh_standard_model \
  --base_url s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_7_5

# End-to-end (aggregate then display) at 0.04°:
python -m src.scripts.postprocessing.create_global_maps \
  all -cn create_maps --date_tag 20250914 --run_name ogh_standard_model \
  --read_from_s3 --base_url s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_7_5

# After display-only or end-to-end runs, publish displays to S3 (optional):
aws s3 sync /tmp/create_global_maps/display \
  s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/ \
  --exclude "*" --include "*.jpeg" --include "*.gif"
"""

from __future__ import annotations

import argparse
import math
import os
import posixpath
import time
from pathlib import Path
from typing import Optional, Tuple, List

import dask
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from PIL import Image
import geopandas as gpd

# Project utilities
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu


# ---------------------------------------------------------------------------
# Defaults (override via CLI)
# ---------------------------------------------------------------------------
DEFAULT_NATIVE_DEG = 0.00025
DEFAULT_TARGET_DEG = 0.04

DATA_TYPES = [
    # Typical datasets (toggle as needed)
    "burned_total_Mg_CO2e_pixel_yr",
    "drained_total_Mg_CO2e_pixel_yr",
]

INTEGER_DATASETS: set[str] = set()  # modal aggregation for these dataset names

INVENTORY_PERIODS = ["2021_2024"]

# Keep default aligned to latest runs in your logs; can be overridden via CLI
BASE_URL = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_7_5"
OUTPUTS_BASE = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"

DEFAULT_DATE_TAG = "20250914"  # used within input dataset paths unless overridden

# Local root for display outputs (mirrors the S3 tree below this root)
DISPLAY_OUT_ROOT = os.environ.get("DISPLAY_OUT_ROOT", "/tmp/create_global_maps/display")


# ---------------------------------------------------------------------------
# Helpers (common)
# ---------------------------------------------------------------------------
def deg_to_label(deg: float) -> str:
    """
    Convert 0.04 -> '0_04deg'; 0.01 -> '0_01deg'; 0.005 -> '0_005deg'
    """
    s = f"{deg:.5f}".rstrip("0").rstrip(".")
    return f"{s.replace('.', '_')}deg"


def assert_grid_divides_world(target_deg: float):
    """
    Ensure the target_deg divides 180 and 360 cleanly (within rounding tolerance).
    """
    rows = round(180 / target_deg)
    cols = round(360 / target_deg)
    if not (np.isclose(rows * target_deg, 180.0) and np.isclose(cols * target_deg, 360.0)):
        raise ValueError(
            f"--target_deg={target_deg} must divide 180 and 360 degrees evenly for a global grid."
        )


def _ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def _gdalize_s3_url(s3_url: str) -> str:
    """s3://bucket/key.tif -> /vsis3/bucket/key.tif"""
    if s3_url.startswith("s3://"):
        return "/vsis3/" + s3_url[len("s3://"):]
    return s3_url


def _to_local_mirror(path_like: str) -> str:
    """
    Map an S3 path (s3://...) or /vsis3/... to a local mirror path under DISPLAY_OUT_ROOT.
    This preserves the bucket/key layout beneath the root for easy syncing.
    """
    if path_like.startswith("/vsis3/"):
        rel = path_like[len("/vsis3/") :].lstrip("/")
    elif path_like.startswith("s3://"):
        rel = path_like[len("s3://") :].lstrip("/")
    else:
        # Treat it as an absolute/relative local path; mirror it under DISPLAY_OUT_ROOT
        rel = path_like.lstrip("/")

    local = posixpath.join(DISPLAY_OUT_ROOT.rstrip("/"), rel)
    return local


# ---------------------------------------------------------------------------
# Stage A: Aggregation
# ---------------------------------------------------------------------------
def get_input_datasets(
    pixel_resolution: str,
    data_types: Optional[List[str]] = None,
    inventory_periods: Optional[List[str]] = None,
    run_name: str = "ogh_standard_model",
    base_url: str = BASE_URL,
    output_date: str = DEFAULT_DATE_TAG,
) -> List[str]:
    """
    Return list of S3 folders for input rasters.
    """
    data_types = data_types or DATA_TYPES
    inventory_periods = inventory_periods or INVENTORY_PERIODS

    paths: List[str] = []
    for period in inventory_periods:
        for dtype in data_types:
            path = (
                f"{base_url}/{dtype}/{run_name}/"
                f"five_year_intervals/{period}/{pixel_resolution}/{output_date}"
            )
            paths.append(path if path.endswith("/") else path + "/")
    return paths


def agg_tile_to_target(
    tile_id: str,
    bounds: Tuple[float, float, float, float],
    chunk_length_pixels: int,
    pixel_area_tile: Optional[str],
    mg_ha_yr_tile: str,
    per_pixel_output_tile: Optional[str],
    per_pixel_output_path: Optional[str],
    use_pixel_area: bool,
    native_deg: float,
    target_deg: float,
    is_final: bool,  # <— pass-through to uu.save_and_upload_single_raster
):
    """
    Aggregate one 10×10° tile from native_deg to target_deg.
    If use_pixel_area=True (and dataset is continuous), convert per-ha to per-pixel using pixel area first.
    """
    logger = lu.setup_logging()

    logger.info(
        f"Getting rasters for {tile_id}\n{pixel_area_tile}\n{mg_ha_yr_tile}"
    )

    mg_ha_yr_tile_chunk = uu.get_tile_dataset_rio(
        mg_ha_yr_tile, "Float32", bounds, chunk_length_pixels, is_final, logger
    )[0]

    dataset_name = posixpath.basename(mg_ha_yr_tile).split("__")[1]
    is_integer = dataset_name in INTEGER_DATASETS
    if is_integer:
        mg_ha_yr_tile_chunk = mg_ha_yr_tile_chunk.astype(np.int32)

    if use_pixel_area and not is_integer:
        pixel_area_tile_chunk = uu.get_tile_dataset_rio(
            pixel_area_tile, "Float32", bounds, chunk_length_pixels, is_final, logger
        )[0]

        mg_per_pixel_tile_chunk = (
            mg_ha_yr_tile_chunk * pixel_area_tile_chunk * cn.m2_to_ha
        )

        if per_pixel_output_tile and per_pixel_output_path:
            data_type = mg_per_pixel_tile_chunk.dtype.name
            uu.save_and_upload_single_raster(
                bounds,
                chunk_length_pixels,
                tile_id,
                mg_per_pixel_tile_chunk,
                data_type,
                per_pixel_output_tile,
                per_pixel_output_path,
                is_final,  # <-- canonical names when final
                logger,
            )

        return uu.reaggregate_resolution(
            mg_per_pixel_tile_chunk, native_deg, target_deg
        )

    if is_integer:
        return uu.reaggregate_mode(
            mg_ha_yr_tile_chunk, native_deg, target_deg
        )

    # average per-ha values within each target cell
    summed = uu.reaggregate_resolution(
        mg_ha_yr_tile_chunk, native_deg, target_deg
    )
    factor = target_deg / native_deg
    if not np.isclose(round(factor), factor):
        raise ValueError(
            f"target_deg/{native_deg} must be an integer. Got {target_deg}/{native_deg}."
        )
    factor = int(round(factor))
    return summed / float(factor * factor)


def combine_global_raster(
    tiles: List[np.ndarray],
    bounds_list: List[Tuple[float, float, float, float]],
    res_label: str,
    global_outfile: str,
    global_output_path: str,
    target_deg: float,
    is_final: bool,  # <— critical: use the real final-ness for canonical upload
):
    """
    Paste aggregated tiles into a single global array at target_deg and save/upload.
    """
    logger = lu.setup_logging()

    rows = int(round(180 / target_deg))
    cols = int(round(360 / target_deg))
    global_raster = np.full((rows, cols), np.nan, dtype=np.float32)

    for tile, bounds in zip(tiles, bounds_list):
        min_x, min_y, max_x, max_y = bounds
        x_start = int(round((min_x + 180) / target_deg))
        x_end   = int(round((max_x + 180) / target_deg))
        y_start = int(round((90 - max_y) / target_deg))
        y_end   = int(round((90 - min_y) / target_deg))

        th, tw = tile.shape
        assert (y_end - y_start) == th
        assert (x_end - x_start) == tw

        np.copyto(
            global_raster[y_start:y_end, x_start:x_end],
            tile,
            where=~np.isnan(tile),
        )

    global_bounds = (-180, -90, 180, 90)

    uu.save_and_upload_single_raster(
        global_bounds,
        global_raster.shape[1],
        f"{res_label}_global",
        global_raster,
        np.float32,
        global_outfile,
        global_output_path,
        is_final,  # <-- canonical if final
        logger,
    )
    return "Success"


def build_download_upload_dict(
    pixel_resolution: str,
    run_name: str,
    target_deg: float,
    base_url: str = BASE_URL,
    output_date: str = DEFAULT_DATE_TAG,
    outputs_base: str = OUTPUTS_BASE,
    data_types: Optional[List[str]] = None,
    inventory_periods: Optional[List[str]] = None,
) -> dict:
    """
    Build a dictionary of input/output paths keyed by "{dataset}__{interval}".
    Output locations and filenames include the target resolution label.
    """
    res_label = deg_to_label(target_deg)
    data_types = data_types or DATA_TYPES
    inventory_periods = inventory_periods or INVENTORY_PERIODS

    dictionary = {}
    for path in get_input_datasets(
        pixel_resolution,
        data_types,
        inventory_periods,
        run_name,
        base_url,
        output_date,
    ):
        parts = path.rstrip("/").split("/")
        # path template:
        # .../outputs/version_x_y_z/{dataset}/{run_name}/five_year_intervals/{interval}/{pixel_resolution}/{date}/
        dataset = parts[8]
        interval = parts[11]
        key = f"{dataset}__{interval}"

        mg_ha_yr_dir = path
        mg_ha_yr_pattern = f"__{dataset}__{interval}.tif"

        # If dataset is per-ha, we may want a per-pixel companion name;
        # otherwise, keep the same dataset name (avoid double '_pixel').
        if dataset.endswith("_ha") or dataset.endswith("_ha_yr"):
            dataset_pixel = dataset.replace("_ha_yr", "_pixel_yr").replace("_ha", "_pixel")
        else:
            dataset_pixel = dataset

        mg_per_pixel_dir = mg_ha_yr_dir.replace(dataset, dataset_pixel)
        mg_per_pixel_pattern = f"__{dataset_pixel}__{interval}.tif"

        out_dir = (
            f"{outputs_base}/{res_label}_output_aggregation/"
            f"{dataset}/{interval}/"
        )

        dictionary[key] = {
            "dataset": dataset,
            "interval": interval,
            "mg_ha_yr_dir": mg_ha_yr_dir,
            "mg_ha_yr_pattern": mg_ha_yr_pattern,
            "mg_per_pixel_dir": mg_per_pixel_dir,
            "mg_per_pixel_pattern": mg_per_pixel_pattern,
            "global_dir": out_dir,
            "global_pattern": f"{res_label}_global__{dataset}_{interval}.tif",
        }
    return dictionary


def aggregate_main(
    cluster_name: str,
    pixel_resolution: str,
    run_name: str = "ogh_standard_model",
    run_local: bool = False,
    use_pixel_area: bool = True,
    native_deg: float = DEFAULT_NATIVE_DEG,
    target_deg: float = DEFAULT_TARGET_DEG,
    base_url: str = BASE_URL,
    output_date: str = DEFAULT_DATE_TAG,
    outputs_base: str = OUTPUTS_BASE,
):
    """
    Drive Stage A: aggregation to target_deg and global mosaic.
    """
    assert_grid_divides_world(target_deg)

    logger = lu.setup_logging_main()
    is_final = not run_local

    cluster, client, run_local = uu.connect_to_cluster(
        cluster_name, run_local=run_local
    )
    # Recompute is_final if connect_to_cluster forces local
    is_final = not run_local

    download_upload_dictionary = build_download_upload_dict(
        pixel_resolution=pixel_resolution,
        run_name=run_name,
        target_deg=target_deg,
        base_url=base_url,
        output_date=output_date,
        outputs_base=outputs_base,
    )

    res_label = deg_to_label(target_deg)

    for key, items in download_upload_dictionary.items():
        bounds_list: List[Tuple[float, float, float, float]] = []
        delayed_results: List = []

        dataset_name = items["dataset"]
        is_integer = dataset_name in INTEGER_DATASETS

        for tile_id in cn.tile_id_list:
            stage = f"aggregate tiles to {res_label} for {key}"
            start_time = uu.timestr()
            lu.print_and_log(
                f"Stage {stage} started at: {start_time}", is_final, logger
            )

            mg_ha_yr_tile = f"{items['mg_ha_yr_dir']}{tile_id}{items['mg_ha_yr_pattern']}"
            pixel_area_tile = (
                f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tile_id}.tif"
                if use_pixel_area and not is_integer
                else None
            )

            per_pixel_tile_outfile = None
            per_pixel_output_path = None
            if use_pixel_area and not is_integer:
                per_pixel_tile_outfile = f"{tile_id}{items['mg_per_pixel_pattern']}"
                per_pixel_output_path = items["mg_per_pixel_dir"]

            bounds = uu.get_10x10_tile_bounds(tile_id)
            bounds_list.append(bounds)
            chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)

            delayed_results.append(
                dask.delayed(agg_tile_to_target)(
                    tile_id,
                    bounds,
                    chunk_length_pixels,
                    pixel_area_tile,
                    mg_ha_yr_tile,
                    per_pixel_tile_outfile,
                    per_pixel_output_path,
                    use_pixel_area,
                    native_deg,
                    target_deg,
                    is_final,  # <-- critical pass-through
                )
            )

        stage = f"build {res_label} global mosaic for {key}"
        start_time = uu.timestr()
        lu.print_and_log(
            f"Stage {stage} started at: {start_time}", is_final, logger
        )

        tiles = dask.compute(*delayed_results)

        # Decide global filename
        if use_pixel_area and not is_integer and items["dataset"].endswith(("_ha", "_ha_yr")):
            global_outfile = f"{res_label}_global{items['mg_per_pixel_pattern']}"
        else:
            global_outfile = items["global_pattern"]

        global_output_path = items["global_dir"]

        _ = combine_global_raster(
            tiles=list(tiles),
            bounds_list=bounds_list,
            res_label=res_label,
            global_outfile=global_outfile,
            global_output_path=global_output_path,
            target_deg=target_deg,
            is_final=is_final,  # <-- canonical names when final
        )
        lu.print_and_log(
            f"Global raster saved to {global_output_path}{global_outfile}",
            is_final,
            logger,
        )

    client.close()


# ---------------------------------------------------------------------------
# Stage B: Display
# ---------------------------------------------------------------------------
def _rgb_to_mpl(rgb):
    return tuple(v / 255 for v in rgb)


def _rgb_palette_to_mpl(rgb_palette):
    return [_rgb_to_mpl(rgb) for rgb in rgb_palette]


def _reproject_if_needed(out_tif: str, in_tif: str, target_crs: str):
    """
    Reproject to target_crs if out_tif does not exist; set nodata=0 in output.
    out_tif MUST be a local filesystem path (we never write to /vsis3/).
    """
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
        _ensure_dir(Path(out_tif).parent)
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
    """
    mode: 'diverging' | 'emissions' | 'removals' | 'default'
    returns: masked_data, norm, (vmin, vcenter, vmax) or (vmin, vmax)
    """
    if mode == "diverging":
        vmin, vcenter, vmax = breaks[0], 0.0, breaks[-1]
        masked = np.ma.masked_where(data == 0, data)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        return masked, norm, (vmin, vcenter, vmax)
    elif mode == "emissions":
        vmin, vmax = breaks[0], breaks[-1]
        masked = np.ma.masked_where(data <= 0, data)
        norm = Normalize(vmin=vmin, vmax=vmax)
        return masked, norm, (vmin, vmax)
    elif mode == "removals":
        vmin, vmax = breaks[0], breaks[-1]
        masked = np.ma.masked_where(data >= 0, data)
        norm = Normalize(vmin=vmin, vmax=vmax)
        return masked, norm, (vmin, vmax)
    else:
        vmin, vmax = breaks[0], breaks[-1]
        masked = np.ma.masked_where(data == 0, data)
        norm = Normalize(vmin=vmin, vmax=vmax)
        return masked, norm, (vmin, vmax)


def _choose_recipe(dataset_name: str):
    """
    Simple mapping from dataset name to rendering mode and legend text.
    """
    lower = dataset_name.lower()
    if "net" in lower:
        return "diverging", "Net greenhouse gas flux\nkt CO$_2$e yr$^{-1}$"
    if "removal" in lower or "sink" in lower:
        return "removals", "Gross CO$_2$ removals\nkt CO$_2$ yr$^{-1}$"
    return "emissions", "Gross greenhouse gas emissions\nkt CO$_2$e yr$^{-1}$"


def _plot_country_layers(ax, shp_gdf):
    if shp_gdf is None or shp_gdf.empty:
        return
    # land fill
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
        # boundaries
        shp_gdf.boundary.plot(ax=ax, edgecolor=_rgb_to_mpl(cn.boundary_color),
                              linewidth=cn.boundary_width, zorder=3)
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

    cax.text(0, 1.1, title_text, fontsize=cn.legend_fontsize, ha="left", va="bottom",
             transform=cax.transAxes)


def _draw_frame(ax, extent, masked, cmap, norm, ocean_rgb):
    ax.set_facecolor(_rgb_to_mpl(ocean_rgb))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xticklabels([]); ax.set_yticklabels([])
    img = ax.imshow(masked, cmap=cmap, norm=norm, extent=extent, origin="upper", zorder=2)
    return img


def _save_two_versions(ax, base_out: Path, year_label: str | int):
    base_no_note = f"{base_out}__{year_label}.jpeg"
    base_with_note = f"{base_out}__{year_label}__for_pres.jpeg"

    ax.text(0.5, 0.07, str(year_label), transform=ax.transAxes, ha="center", va="top",
            fontsize=18, weight="bold", color="black")

    plt.savefig(base_no_note, dpi=cn.dpi_jpeg, bbox_inches="tight", pad_inches=0)

    ax.text(0.98, 0.04, cn.pres_text, transform=ax.transAxes, fontsize=7,
            ha="right", va="top", color="black")
    plt.savefig(base_with_note, dpi=cn.dpi_jpeg, bbox_inches="tight", pad_inches=0)
    plt.close()
    return base_with_note


def _gif_from_frames(base_out: Path, first_label, last_label, frames: List[str]):
    if not frames:
        return
    imgs = [Image.open(p) for p in frames]
    out_fast = f"{base_out}__{first_label}_{last_label}__fast.gif"
    out_slow = f"{base_out}__{first_label}_{last_label}__slow.gif"
    imgs[0].save(out_fast, save_all=True, append_images=imgs[1:], duration=1000, loop=0)
    imgs[0].save(out_slow, save_all=True, append_images=imgs[1:], duration=2500, loop=0)


def _load_or_project_raster(base_tif_noext: str, target_crs: str, work_dir: Path) -> str:
    """
    base_tif_noext is the base path without ".tif". It may be an /vsis3/ path or local path.
    We always write the reprojected file to 'work_dir' (local).
    """
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
        logger.warning(
            f"World boundaries not found: {rp} or {op}"
        )
        return None
    except Exception as e:
        logger.warning(f"World boundaries load failed: {e}")
        return None


def _try_open_first(paths_noext: List[str], target_crs: str, work_dir: Path) -> Tuple[str, str]:
    """
    Attempt to reproject the first available candidate. Returns (reprojected_tif_path, input_kind).
    input_kind is 's3' if input was /vsis3/|s3://, else 'local'.
    """
    last_error = None
    for base in paths_noext:
        try:
            kind = "s3" if base.startswith("/vsis3/") or base.startswith("s3://") else "local"
            tif_proj = _load_or_project_raster(base, target_crs, work_dir)
            # Confirm it opens
            with rasterio.open(tif_proj):
                pass
            return tif_proj, kind
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Could not open any candidate raster: {paths_noext}. Last error: {last_error}")


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
    """
    Render one dataset (and optionally a series) to JPEGs and a GIF.
    """
    logger = logger or lu.setup_logging()
    t0 = time.time()

    _ensure_dir(out_dir_local)
    work_dir = Path(out_dir_local) / "_reproj_cache"
    _ensure_dir(work_dir)

    colors_mpl = _rgb_palette_to_mpl(palette_rgb)
    cmap = LinearSegmentedColormap.from_list("custom", colors_mpl)

    recipe_mode, legend_title = _choose_recipe(dataset_name)
    mode = "diverging" if diverging_if_zero_center else recipe_mode

    frames_for_gif: List[str] = []
    labels = years or [interval]

    # Reproject once per (dataset/interval or year)
    for label in labels:
        # Choose the first available candidate, reproject → local cache
        tif_proj, input_kind = _try_open_first(tif_path_noext_candidates, target_crs, work_dir)

        with rasterio.open(tif_proj) as src:
            data = src.read(1)
            bounds = src.bounds

        # Treat zeros as background for masking; convert NaNs to zero.
        data = np.nan_to_num(data, nan=0.0)

        breaks = _compute_percentile_breaks(data, percentiles, ignore_zero=True)
        masked, norm, vtuple = _mask_and_norm(data, mode, breaks)

        fig, ax = plt.subplots(figsize=cn.panel_dims)
        _plot_country_layers(ax, shapefile_gdf)
        img = _draw_frame(ax, [bounds.left, bounds.right, bounds.bottom, bounds.top],
                          masked, cmap, norm, cn.ocean_color)

        # Use compressed array to avoid masked->nan warning for min/max
        comp = masked.compressed()
        if comp.size == 0:
            dmin, dmax = 0.0, 0.0
        else:
            dmin, dmax = float(comp.min()), float(comp.max())
        _legend(fig, img, mode, vtuple, legend_title, dmin, dmax)

        base_out = Path(out_dir_local) / out_name_base
        frame = _save_two_versions(ax, base_out, label)
        frames_for_gif.append(frame)

    try:
        if len(frames_for_gif) > 1 and isinstance(labels[0], int):
            _gif_from_frames(Path(out_dir_local) / out_name_base, labels[0], labels[-1], frames_for_gif)
    except Exception as e:
        logger.warning(f"GIF creation failed: {e}")

    logger.info(
        f"Rendered {dataset_name} {interval} in {round(time.time()-t0)} s → {posixpath.join(out_dir_local, out_name_base)}"
    )


def display_main(
    date_tag: str,
    read_from_s3: bool,
    run_name: str,
    pixel_resolution: str,
    target_deg: float = DEFAULT_TARGET_DEG,
    base_url: str = BASE_URL,
):
    """
    Drive Stage B: create display images for the specified target resolution.
    """
    assert_grid_divides_world(target_deg)
    res_label = deg_to_label(target_deg)

    logger = lu.setup_logging_main()
    d = build_download_upload_dict(
        pixel_resolution=pixel_resolution,
        run_name=run_name,
        target_deg=target_deg,
        base_url=base_url,
        output_date=date_tag,
        outputs_base=OUTPUTS_BASE,
    )

    shp = _maybe_load_world_boundaries(logger)

    # Diverging palette (BrBG 10); gross palettes are subsets
    net_palette = [
        (0, 60, 48), (1, 102, 94), (53, 151, 143), (128, 205, 193), (199, 234, 229),
        (246, 232, 195), (223, 194, 125), (191, 129, 45), (140, 81, 10), (84, 48, 5)
    ]
    removals_palette = net_palette[0:5]
    emissions_palette = net_palette[5:]

    net_percentiles = [5, 25, 50, 75, 89, 91, 92, 93, 94, 99]
    gross_percentiles = [5, 25, 50, 75, 99]

    for key, items in d.items():
        dataset = items["dataset"]
        interval = items["interval"]

        # Build the preferred and fallback global raster candidates (without .tif)
        # 1) Preferred: canonical aggregated output
        canonical_noext = posixpath.join(items["global_dir"], items["global_pattern"][:-4])

        # 2) Fallback: a versioned directory “global” (if present in versioned structure)
        versioned_noext = posixpath.join(
            base_url.rstrip("/"),
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
            candidates = [_gdalize_s3_url(p) for p in candidates]

        out_jpeg_dir_base = posixpath.join(items["global_dir"], "display", interval, dataset)
        out_jpeg_dir_local = _to_local_mirror(out_jpeg_dir_base)
        _ensure_dir(out_jpeg_dir_local)

        logger.info(
            f"[Stage B] Rendering dataset={dataset} interval={interval} | mode=auto | candidates={candidates}"
        )

        mode, _ = _choose_recipe(dataset)
        palette = net_palette if mode == "diverging" else (emissions_palette if mode == "emissions" else removals_palette)
        ptiles = net_percentiles if mode == "diverging" else gross_percentiles

        out_base = f"{dataset}__{interval}"

        make_displays_for_dataset(
            dataset_name=dataset,
            interval=interval,
            tif_path_noext_candidates=candidates,
            out_dir_local=out_jpeg_dir_local,
            palette_rgb=palette,
            percentiles=list(ptiles),
            shapefile_gdf=shp,
            out_name_base=out_base,
            years=None,  # Provide a list if you also maintain annual global stacks.
            diverging_if_zero_center=("net" in dataset.lower()),
            target_crs=cn.Robinson_crs,
            logger=logger,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Aggregate to target_deg and/or create display maps.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_agg = sub.add_parser("aggregate", help="Run Stage A (aggregation) only.")
    p_agg.add_argument("-cn", "--cluster_name", required=True)
    p_agg.add_argument("-p", "--pixel_resolution", default="40000_pixels")
    p_agg.add_argument("--run_name", default="ogh_standard_model")
    p_agg.add_argument("--run_local", action="store_true")
    p_agg.add_argument("--skip_pixel_area", action="store_true")
    p_agg.add_argument("--native_deg", type=float, default=DEFAULT_NATIVE_DEG)
    p_agg.add_argument("--target_deg", type=float, default=DEFAULT_TARGET_DEG)
    p_agg.add_argument("--base_url", default=BASE_URL)
    p_agg.add_argument("--date_tag", default=DEFAULT_DATE_TAG)
    p_agg.add_argument("--outputs_base", default=OUTPUTS_BASE)

    p_disp = sub.add_parser("display", help="Run Stage B (display) only.")
    p_disp.add_argument("--date_tag", required=True, help="Date tag used in input paths (YYYYMMDD).")
    p_disp.add_argument("--read_from_s3", action="store_true", help="Use GDAL /vsis3/ to read rasters in place.")
    p_disp.add_argument("--run_name", default="ogh_standard_model")
    p_disp.add_argument("-p", "--pixel_resolution", default="40000_pixels")
    p_disp.add_argument("--target_deg", type=float, default=DEFAULT_TARGET_DEG)
    p_disp.add_argument("--base_url", default=BASE_URL)

    p_all = sub.add_parser("all", help="Run Stage A then Stage B.")
    p_all.add_argument("-cn", "--cluster_name", required=True)
    p_all.add_argument("--date_tag", required=True)
    p_all.add_argument("-p", "--pixel_resolution", default="40000_pixels")
    p_all.add_argument("--run_name", default="ogh_standard_model")
    p_all.add_argument("--run_local", action="store_true")
    p_all.add_argument("--skip_pixel_area", action="store_true")
    p_all.add_argument("--native_deg", type=float, default=DEFAULT_NATIVE_DEG)
    p_all.add_argument("--target_deg", type=float, default=DEFAULT_TARGET_DEG)
    p_all.add_argument("--base_url", default=BASE_URL)
    p_all.add_argument("--outputs_base", default=OUTPUTS_BASE)
    p_all.add_argument("--read_from_s3", action="store_true")

    args = parser.parse_args()

    # Use a local variable, don't reassign the module-level constant (avoids SyntaxError).
    selected_base_url = getattr(args, "base_url", None) or BASE_URL

    if args.cmd == "aggregate":
        aggregate_main(
            cluster_name=args.cluster_name,
            pixel_resolution=args.pixel_resolution,
            run_name=args.run_name,
            run_local=args.run_local,
            use_pixel_area=not args.skip_pixel_area,
            native_deg=args.native_deg,
            target_deg=args.target_deg,
            base_url=selected_base_url,
            output_date=args.date_tag,
            outputs_base=args.outputs_base,
        )
    elif args.cmd == "display":
        display_main(
            date_tag=args.date_tag,
            read_from_s3=args.read_from_s3,
            run_name=args.run_name,
            pixel_resolution=args.pixel_resolution,
            target_deg=args.target_deg,
            base_url=selected_base_url,
        )
    else:  # all
        aggregate_main(
            cluster_name=args.cluster_name,
            pixel_resolution=args.pixel_resolution,
            run_name=args.run_name,
            run_local=args.run_local,
            use_pixel_area=not args.skip_pixel_area,
            native_deg=args.native_deg,
            target_deg=args.target_deg,
            base_url=selected_base_url,
            output_date=args.date_tag,
            outputs_base=args.outputs_base,
        )
        display_main(
            date_tag=args.date_tag,
            read_from_s3=args.read_from_s3,
            run_name=args.run_name,
            pixel_resolution=args.pixel_resolution,
            target_deg=args.target_deg,
            base_url=selected_base_url,
        )


if __name__ == "__main__":
    main()
