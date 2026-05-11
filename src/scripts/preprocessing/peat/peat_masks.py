#!/usr/bin/env python
# peat_masks.py ─ robust peat-mask tiler for GPD | PEATML | PEATMAP
# (Integrated: 2025-04-30)

import os
import argparse
import logging
import tempfile
import posixpath as pp
import re
import math
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import geopandas as gpd
import rasterio
import botocore
import dask
from dask import delayed
from dask.distributed import LocalCluster, Client
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import box

# Adjust imports according to your folder structure
import src.scripts.preprocessing.preprocessing_constants as cn
from src.scripts.utilities import universal_utilities as uutil
import src.scripts.preprocessing.utilities as uu
from src.scripts.preprocessing.hansenize.hansenize_coiled import (
    build_vrt_gdal_coiled,
    warp_to_hansen_coiled,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("peat-tiler")

BUCKET = cn.s3_bucket_name
RESOLUTION = cn.resolution
PEAT_CACHE = Path(tempfile.gettempdir()) / "peatmap_cache"
DATE_TOKEN_RE = re.compile(r"^\d{8}$")
WINDOWS_ABS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")

################################################################################
# Helper Functions
################################################################################

def bounds_for_tile(tid):
    return uutil.get_10x10_tile_bounds(tid)

def strip_date_suffix(path):
    clean = str(path).replace("\\", "/").rstrip("/")
    if DATE_TOKEN_RE.fullmatch(pp.basename(clean)):
        return pp.dirname(clean)
    return clean


def dated_raw_path(path, raw_date):
    if not raw_date:
        return path

    parts = str(path).replace("\\", "/").rstrip("/").split("/")
    if not parts:
        return path

    basename = parts[-1]
    if "." in basename:
        if len(parts) >= 2 and DATE_TOKEN_RE.fullmatch(parts[-2]):
            parts[-2] = raw_date
        else:
            parts.insert(-1, raw_date)
    elif DATE_TOKEN_RE.fullmatch(basename):
        parts[-1] = raw_date
    else:
        parts.append(raw_date)
    return "/".join(parts)


def _has_uri_scheme(path):
    return bool(urlparse(str(path)).scheme)


def _is_http_url(path):
    return urlparse(str(path)).scheme in {"http", "https"}


def _is_s3_uri(path):
    return str(path).startswith("s3://")


def _is_vsi_path(path):
    return str(path).startswith("/vsi")


def _is_absolute_path(path):
    raw = str(path)
    return os.path.isabs(raw) or bool(WINDOWS_ABS_PATH_RE.match(raw))


def _split_s3_uri(uri):
    bucket, key = str(uri)[len("s3://") :].split("/", 1)
    return bucket, key


def _is_single_tif_path(path):
    parsed = urlparse(str(path))
    path_part = parsed.path if parsed.scheme else str(path)
    return path_part.lower().endswith((".tif", ".tiff"))


def _source_file_exists(path):
    if _is_s3_uri(path):
        bucket, key = _split_s3_uri(path)
        return uutil.s3_file_exists(bucket, key)
    if _is_http_url(path) or _is_vsi_path(path):
        # Let GDAL do the detailed open/error handling for HTTP/VSI sources.
        # This avoids issuing one extra HEAD request per output tile.
        return True
    return Path(path).exists()


def _gdal_input_path(path):
    raw = str(path)
    if raw.startswith("/vsi"):
        return raw
    if _is_s3_uri(raw):
        return raw.replace("s3://", "/vsis3/", 1)
    if _is_http_url(raw):
        return f"/vsicurl/{raw}"
    return raw


def _tile_intersects_bounds(tile_id, source_bounds):
    tile_w, tile_s, tile_e, tile_n = bounds_for_tile(tile_id)
    src_w, src_s, src_e, src_n = source_bounds
    return (
        max(tile_w, src_w) < min(tile_e, src_e)
        and max(tile_s, src_s) < min(tile_n, src_n)
    )


def _source_raster_bounds(path):
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
    ):
        with rasterio.open(_gdal_input_path(path)) as src:
            bounds = src.bounds
            if src.crs and src.crs.to_epsg() != 4326:
                left, bottom, right, top = transform_bounds(
                    src.crs, "EPSG:4326", *bounds, densify_pts=21
                )
            else:
                left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
            return (float(left), float(bottom), float(right), float(top))


def _filter_tiles_to_source_bounds(tids, source_path):
    if not source_path or not _is_single_tif_path(source_path):
        return list(tids)

    try:
        source_bounds = _source_raster_bounds(source_path)
    except Exception as exc:
        log.warning("Could not read source bounds for %s; processing all tiles. Error: %s", source_path, exc)
        return list(tids)

    filtered = [tid for tid in tids if _tile_intersects_bounds(tid, source_bounds)]
    skipped = len(tids) - len(filtered)
    log.info(
        "Source bounds %s overlap %d/%d tile(s); skipping %d out-of-bounds tile(s).",
        source_bounds,
        len(filtered),
        len(tids),
        skipped,
    )
    return filtered


def _raster_has_positive_data(path):
    with rasterio.open(path) as src:
        for _, window in src.block_windows(1):
            data = src.read(1, window=window, masked=True)
            if bool((data > 0).any()):
                return True
    return False


def _round_and_clip_window(window, width, height):
    col_off = max(0, int(math.floor(window.col_off)))
    row_off = max(0, int(math.floor(window.row_off)))
    col_end = min(width, int(math.ceil(window.col_off + window.width)))
    row_end = min(height, int(math.ceil(window.row_off + window.height)))
    if col_end <= col_off or row_end <= row_off:
        return None
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def _window_intersection(a, b):
    col_off = max(a.col_off, b.col_off)
    row_off = max(a.row_off, b.row_off)
    col_end = min(a.col_off + a.width, b.col_off + b.width)
    row_end = min(a.row_off + a.height, b.row_off + b.height)
    if col_end <= col_off or row_end <= row_off:
        return None
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def _source_window_has_positive_data(source_path, tile_bounds):
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
    ):
        with rasterio.open(_gdal_input_path(source_path)) as src:
            if src.crs and src.crs.to_epsg() != 4326:
                return True
            raw_window = from_bounds(*tile_bounds, transform=src.transform)
            tile_window = _round_and_clip_window(raw_window, src.width, src.height)
            if tile_window is None:
                return False

            for _, block_window in src.block_windows(1):
                read_window = _window_intersection(tile_window, block_window)
                if read_window is None:
                    continue
                data = src.read(1, window=read_window, masked=True)
                if bool((data > 0).any()):
                    return True
    return False


def _apply_threshold_in_blocks(path, threshold_value):
    with rasterio.open(path, "r+") as dst:
        for _, window in dst.block_windows(1):
            arr = dst.read(1, window=window)
            dst.write((arr > threshold_value).astype("uint8"), 1, window=window)


def resolve_raw_path(ds, raw_date=None, raw_path=None):
    raw = raw_path or ds["s3_raw"]
    if raw_date and raw_path is None:
        raw = dated_raw_path(raw, raw_date)
    if (
        not _has_uri_scheme(raw)
        and not _is_vsi_path(raw)
        and not _is_absolute_path(raw)
    ):
        raw = f"s3://{BUCKET}/{str(raw).lstrip('/')}"
    return raw


def output_paths(ds_key, tid, date_str=None):
    ds = cn.datasets["peat"][ds_key]
    date_str = date_str or cn.today_date
    fname = f"{tid}_{ds_key}_mask.tif"

    local_dir = Path(strip_date_suffix(ds["local_processed"])) / date_str
    local_dir.mkdir(parents=True, exist_ok=True)
    local = local_dir / fname

    s3_base = strip_date_suffix(ds["s3_processed"])
    s3_key = f"{s3_base.rstrip('/')}/{date_str}/{fname}"
    return local, s3_key

def cache_shapefile(prefix, dest_dir):
    """
    Downloads the shapefile components (.shp, .dbf, .shx, .prj) for a given prefix.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    extensions = [".shp", ".shx", ".dbf", ".prj"]
    for ext in extensions:
        key = f"{prefix}{ext}"
        local = dest_dir / Path(key).name
        if not local.exists():
            uutil.download_file_from_s3(key, str(local), BUCKET)

################################################################################
# Vector Processor (PEATMAP)
################################################################################
def vector_job(tid, mode="default", date_str=None):
    """
    Processes the 'peatmap' dataset (vector) for a given tile. Clips to tile
    bounds, then rasterizes at the final resolution (EPSG:4326, 0.00025).
    """
    import fiona

    ds_key = "peatmap"
    local_out, s3_out = output_paths(ds_key, tid, date_str=date_str)

    if mode != "test" and uutil.s3_file_exists(BUCKET, s3_out):
        log.info(f"[{ds_key}|{tid}] already on S3. Skipping.")
        return

    tile_box = gpd.GeoSeries([box(*bounds_for_tile(tid))], crs="EPSG:4326")
    shp_folder = cn.datasets["peat"]["peatmap"]["s3_raw"]  # relative path from constants

    # --- 1) If needed, prepend s3://{BUCKET}/
    if not shp_folder.startswith("s3://"):
        shp_folder = f"s3://{BUCKET}/{shp_folder.lstrip('/')}"

    # Attempt to list .shp files in the folder/prefix
    try:
        all_files = uutil.list_s3_files(BUCKET, shp_folder.replace(f"s3://{BUCKET}/", "", 1))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "AccessDenied":
            log.warning(f"[peatmap|{tid}] AccessDenied listing {shp_folder}. Skipping tile.")
            return
        else:
            raise

    # Filter for shapefile endings
    shp_keys = [k for k in all_files if k.lower().endswith(".shp")]

    pieces = []
    for key in shp_keys:
        prefix = key[:-4]  # remove ".shp"
        cache_dir = PEAT_CACHE / tid
        cache_shapefile(prefix, cache_dir)
        local_shp = cache_dir / Path(key).name
        try:
            with fiona.open(local_shp) as src:
                layer_crs = src.crs_wkt or src.crs
                layer_gdf = gpd.read_file(local_shp)

                # If no CRS, set EPSG:4326 or fallback to the shapefile's WKT
                if layer_gdf.crs is None:
                    layer_gdf.set_crs(layer_crs or "EPSG:4326", inplace=True)

                # Reproject to EPSG:4326 if needed
                layer_gdf = layer_gdf.to_crs("EPSG:4326")

                # Clip to tile bounding box
                clipped = layer_gdf[layer_gdf.intersects(tile_box.unary_union)]
                if not clipped.empty:
                    pieces.append(clipped[["geometry"]])

        except Exception as e:
            log.warning(f"Reading failed for {local_shp.name}: {e}")

    if not pieces:
        log.info(f"[{ds_key}|{tid}] no data found, skipping.")
        return

    combined_gdf = gpd.GeoDataFrame(pd.concat(pieces), crs="EPSG:4326")
    combined_gdf = combined_gdf.explode(index_parts=False).reset_index(drop=True)

    # Rasterize with desired resolution (Hansen style)
    uu.rasterize_shapefile_no_ref(
        combined_gdf,
        str(local_out),
        bounds_for_tile(tid),
        cn.resolution,  # e.g., 0.00025
        fill_value=0,
        burn_value=1,
        dtype="uint8"
    )

    if mode != "test":
        uutil.upload_file_to_s3(str(local_out), BUCKET, s3_out)
        local_out.unlink()

    log.info(f"[{ds_key}|{tid}] completed.")

################################################################################
# Raster Processor for PeatML / GPD
################################################################################
def mosaic_and_warp_raster(
    ds_key,
    tid,
    mode="default",
    date_str=None,
    raw_date=None,
    raw_path=None,
    skip_empty_tiles=True,
):
    """
    If ds['s3_raw'] is a single .tif, skip listing and warp that file directly.
    Otherwise, it's a folder containing multiple .tif => build a VRT if needed.
    """
    ds = cn.datasets["peat"][ds_key]
    local_out, s3_out = output_paths(ds_key, tid, date_str=date_str)

    if mode != "test" and uutil.s3_file_exists(BUCKET, s3_out):
        log.info(f"[{ds_key}|{tid}] already on S3. Skipping.")
        return

    raw_path = resolve_raw_path(ds, raw_date=raw_date, raw_path=raw_path)

    # Single .tif approach
    if _is_single_tif_path(raw_path):
        source_for_warp = raw_path
        log.info(f"[{ds_key}|{tid}] Using single-file approach: {source_for_warp}")

    else:
        if not _is_s3_uri(raw_path):
            raise ValueError(
                f"[{ds_key}|{tid}] folder-style raw inputs must be S3 prefixes; "
                f"got {raw_path!r}. Pass a single .tif/.tiff URL or file with --raw_path."
            )
        # Possibly multiple .tif => do the mosaic approach
        # Remove s3://bucket/ to get the prefix for listing
        listing_prefix = raw_path
        raw_pattern = ds.get('raw_pattern', '*.tif')

        try:
            all_rasters = uutil.list_s3_files_with_pattern(listing_prefix, raw_pattern)
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "AccessDenied":
                log.warning(f"[{ds_key}|{tid}] AccessDenied listing {listing_prefix} with pattern '{raw_pattern}'. Skipping tile.")
                return
            else:
                raise

        if not all_rasters:
            log.warning(f"No matching rasters found for dataset {ds_key}. Skipping tile {tid}.")
            return

        if len(all_rasters) > 1:
            # Build a mosaic VRT
            vrt_name = f"mosaic_{ds_key}.vrt"
            vrt_s3_path = f"{raw_path.rstrip('/')}/{vrt_name}"
            local_vrt = os.path.join(tempfile.gettempdir(), vrt_name)

            build_vrt_gdal_coiled(all_rasters, vrt_s3_path, local_vrt=local_vrt)
            source_for_warp = vrt_s3_path
        else:
            source_for_warp = all_rasters[0]

    # Verify the raster exists before warping
    if not _source_file_exists(source_for_warp):
        log.error(f"[{ds_key}|{tid}] raw raster not found: {raw_path}")
        return

    # Warp to 10×10 deg output
    xmin, ymin, xmax, ymax = bounds_for_tile(tid)
    tile_bounds = (xmin, ymin, xmax, ymax)
    if skip_empty_tiles and ds_key in {"ogh", "ogh_unthresholded"} and _is_single_tif_path(source_for_warp):
        if not _source_window_has_positive_data(source_for_warp, tile_bounds):
            log.info(f"[{ds_key}|{tid}] source window contains no positive values. Skipping warp/upload.")
            return

    warp_to_hansen_coiled(
        source_vrt_path=source_for_warp,
        filename=str(local_out),
        output_raster_s3_path_and_name=None,  # or a real path if desired
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        dt=uutil.string_to_gdal_dtype_mapping["Byte"],
        no_data=0,
        tiled=True,
        x_pixel_window=400,
        y_pixel_window=400
    )

    # Apply threshold if needed
    threshold_value = ds.get("threshold", None)
    if threshold_value is not None:
        _apply_threshold_in_blocks(local_out, threshold_value)

    if skip_empty_tiles and ds_key in {"ogh", "ogh_unthresholded"}:
        if not _raster_has_positive_data(local_out):
            log.info(f"[{ds_key}|{tid}] output contains no positive values. Skipping upload.")
            if local_out.exists():
                local_out.unlink()
            return

    if mode != "test":
        uutil.upload_file_to_s3(str(local_out), BUCKET, s3_out)
        local_out.unlink()

    log.info(f"[{ds_key}|{tid}] completed.")

################################################################################
# Build and submit tasks
################################################################################
def build_tasks(tids, ds_keys, mode, date_str=None, raw_date=None, raw_path=None, skip_empty_tiles=True):
    tasks = []
    for tid in tids:
        for k in ds_keys:
            if k == "peatmap":
                tasks.append(delayed(vector_job)(tid, mode, date_str))
            else:
                tasks.append(
                    delayed(mosaic_and_warp_raster)(
                        k,
                        tid,
                        mode,
                        date_str,
                        raw_date,
                        raw_path,
                        skip_empty_tiles,
                    )
                )
    return tasks

################################################################################
# Main orchestrator
################################################################################
def main(
    tile_id=None,
    dataset=None,
    client="coiled",
    run_mode="default",
    date=None,
    raw_date=None,
    raw_path=None,
    cluster_name="peat_masks",
    n_workers=20,
    worker_memory="32GiB",
    skip_empty_tiles=True,
):
    cluster = None
    client_obj = None
    date_str = date or cn.today_date

    if client == "local":
        cluster = LocalCluster(processes=False, dashboard_address=None)
        client_obj = Client(cluster)
        log.info("Running locally.")
    else:
        cluster, client_obj, run_local = uutil.connect_to_cluster(
            cluster_name=cluster_name,
            n_workers=n_workers,
            region="us-east-1",
            worker_memory=worker_memory,
        )

        if run_local:
            log.info("Coiled cluster unavailable. Falling back to a local Dask cluster.")
            cluster = LocalCluster(processes=False, dashboard_address=None)
            client_obj = Client(cluster)
        else:
            log.info(f"Running on Coiled: {cluster.name}")

    ds_keys = [dataset] if dataset else ["peatml", "gpd", "peatmap", "ogh", "ogh_unthresholded"]
    if raw_path and len(ds_keys) != 1:
        raise ValueError("--raw_path requires --dataset so it is applied to exactly one source")
    tids = [tile_id] if tile_id else cn.tile_id_list
    if len(ds_keys) == 1 and ds_keys[0] != "peatmap":
        raw_for_bounds = resolve_raw_path(
            cn.datasets["peat"][ds_keys[0]],
            raw_date=raw_date,
            raw_path=raw_path,
        )
        tids = _filter_tiles_to_source_bounds(tids, raw_for_bounds)

    log.info(
        f"Datasets: {ds_keys}, Tiles: {len(tids)}, output_date={date_str}, "
        f"raw_date={raw_date}, cluster_name={cluster_name}, n_workers={n_workers}, "
        f"skip_empty_tiles={skip_empty_tiles}"
    )
    if not tids:
        log.info("No tiles remain after source-bounds filtering. Nothing to do.")
    tasks = build_tasks(tids, ds_keys, run_mode, date_str, raw_date, raw_path, skip_empty_tiles)

    dask.compute(*tasks)

    # Close out
    if client_obj is not None:
        client_obj.close()
    if cluster is not None:
        cluster.close()
    log.info("All tasks completed.")

################################################################################
# CLI
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust peat-mask tiler")
    parser.add_argument("--tile_id", help="Single tile ID (optional)")
    parser.add_argument("--dataset", choices=["peatml", "gpd", "peatmap", "ogh", "ogh_unthresholded"], help="Dataset (optional)")
    parser.add_argument("--client", default="coiled", choices=["local", "coiled"], help="Run mode (default: coiled)")
    parser.add_argument("--run_mode", default="default", choices=["default", "test"], help="Run mode")
    parser.add_argument("--cluster_name", default="peat_masks", help="Coiled cluster name to attach to.")
    parser.add_argument("--n_workers", type=int, default=20, help="Expected Coiled worker count for logging/cluster helper.")
    parser.add_argument("--worker_memory", default="32GiB", help="Expected Coiled worker memory for logging/cluster helper.")
    parser.add_argument(
        "--upload_empty_tiles",
        action="store_true",
        help="Upload all-zero OGH output tiles instead of skipping them.",
    )
    parser.add_argument("--date", default=None, help="Output date tag (YYYYMMDD). Defaults to today's UTC date.")
    parser.add_argument(
        "--raw_date",
        default=None,
        help=(
            "Date folder for dated raw raster inputs. For OGH single-file inputs, "
            "this replaces or inserts the YYYYMMDD folder before the configured "
            "GeoTIFF name. Ignored when --raw_path is supplied."
        ),
    )
    parser.add_argument(
        "--raw_path",
        default=None,
        help=(
            "Fully specified raw raster path/key for the selected dataset. "
            "Supports s3://, https://, /vsi*, and local paths. Requires --dataset."
        ),
    )
    args = parser.parse_args()
    main(
        args.tile_id,
        args.dataset,
        args.client,
        args.run_mode,
        args.date,
        args.raw_date,
        args.raw_path,
        args.cluster_name,
        args.n_workers,
        args.worker_memory,
        not args.upload_empty_tiles,
    )
