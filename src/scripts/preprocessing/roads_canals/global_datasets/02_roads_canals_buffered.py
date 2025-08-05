#!/usr/bin/env python
"""Ring-buffered distance from roads and canals using the 30m union peat mask.

This script reads raw road/canal shapefiles (EPSG:4326) at tile level,
clips them to sub-chunks of the union mask grid (expanded by
``MAX_DISTANCE`` pixels to avoid edge effects), rasterises presence at 30 m,
computes pixel-distance rings (1 for features, 2 for one pixel away … up to
30), and warps the result to the Hansen grid. Use this script for the 30 m
workflow. The 1 km density workflow remains in
``02_roads_canals_coiled.py``.
"""

import os
import logging
import gc
import boto3
import numpy as np
import dask
import dask_geopandas as dgpd
import posixpath
import xarray as xr
import rioxarray as rxr
import rasterio
from rasterio.merge import merge as merge_rasters
from pyogrio.errors import FeatureError
from rasterio.features import rasterize
from shapely.geometry import box
from rasterio.warp import transform_bounds
import tempfile

from src.scripts.preprocessing.hansenize.hansenize_coiled import (
    warp_to_hansen_coiled,
)
from src.scripts.utilities import universal_utilities as uutil
import src.scripts.preprocessing.preprocessing_constants as cn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PEAT_30M_PATTERN = "_union_mask.tif"
MAX_DISTANCE = 30


def build_chunk_bounds(tile_bounds, chunk_size=2):
    """Subdivide tile bounds into ``chunk_size`` degree boxes."""
    (min_x, min_y, max_x, max_y) = tile_bounds
    x, y = (min_x, min_y)
    chunks = []
    while y < max_y:
        while x < max_x:
            chunks.append([x, y, x + chunk_size, y + chunk_size])
            x += chunk_size
        x = min_x
        y += chunk_size
    return chunks


def dask_gdf_is_empty(dgdf, data_path=None):
    """Return ``True`` if the Dask GeoDataFrame is empty."""
    try:
        lengths = dgdf.map_partitions(len).compute()
        length = sum(lengths)
    except FeatureError as exc:
        msg = f"FeatureError while reading {data_path}: {exc}"
        logging.error(msg)
        return True
    return length == 0


def dilate(arr):
    """Simple 8-connected binary dilation implemented with NumPy."""
    padded = np.pad(arr, 1, mode="constant", constant_values=0)
    out = np.zeros_like(arr, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            out |= padded[1 + dy : 1 + dy + arr.shape[0], 1 + dx : 1 + dx + arr.shape[1]]
    return out


def compute_buffered(binary, mask, max_dist=MAX_DISTANCE):
    """Return ring-buffered distances in pixel units."""
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:  # pragma: no cover - SciPy not installed in some envs
        distance_transform_edt = None

    if distance_transform_edt is not None:
        dist = distance_transform_edt(~binary)
        buffered = np.floor(dist).astype(np.uint8) + 1
        buffered[buffered > max_dist] = 0
        buffered[~mask] = 0
        return buffered

    # Fallback without SciPy: iterative dilations
    current = binary.copy()
    visited = current.copy()
    buffered = np.zeros(binary.shape, dtype=np.uint8)
    buffered[current] = 1
    for d in range(2, max_dist + 1):
        current = dilate(current)
        current &= mask
        ring = current & (~visited)
        if not ring.any():
            break
        buffered[ring] = d
        visited |= ring
    return buffered


@dask.delayed
def process_chunk_buffered(bounds, tile_id, feature_type):
    """Rasterise lines to distance rings using the 30 m union mask."""
    chunk_str = "_".join(map(str, bounds))
    group, sub = feature_type.split('_', 1)
    local_dir = cn.datasets[group][sub]['local_processed']
    chunk_px = uutil.calc_chunk_length_pixels(bounds)
    s3_dir = posixpath.join(
        cn.datasets[group][sub]['s3_processed_base'],
        f"{chunk_px}_pixels",
        cn.today_date,
    )

    chunk_name = f"{tile_id}__{chunk_str}__{feature_type}_buffered.tif"
    local_out = os.path.join(local_dir, chunk_name)
    s3_out_key = f"{s3_dir}/{chunk_name}"

    logging.info(f"[{tile_id}|{chunk_str}] Starting buffered chunk processing")

    prefix_30m = cn.datasets["peat"]["union_mask"]["30m"]
    raster_path = f"/vsis3/{cn.s3_bucket_name}/{prefix_30m}{tile_id}{PEAT_30M_PATTERN}"

    try:
        ds_main = rasterio.open(raster_path)
    except Exception as e:
        logging.error(f"[{tile_id}|{chunk_str}] Could not open union mask: {e}")
        return

    bounds_wgs84 = bounds
    minx_crs, miny_crs, maxx_crs, maxy_crs = transform_bounds(
        "EPSG:4326", ds_main.crs, *bounds, densify_pts=21
    )
    xres, yres = ds_main.res
    pad_x = abs(xres) * MAX_DISTANCE
    pad_y = abs(yres) * MAX_DISTANCE
    expanded_bounds_crs = (
        minx_crs - pad_x,
        miny_crs - pad_y,
        maxx_crs + pad_x,
        maxy_crs + pad_y,
    )
    expanded_bounds_wgs84 = transform_bounds(
        ds_main.crs,
        "EPSG:4326",
        *expanded_bounds_crs,
        densify_pts=21,
    )

    tile_ids = {
        uutil.xy_to_tile_id(x, y)
        for x in (expanded_bounds_wgs84[0], expanded_bounds_wgs84[2])
        for y in (expanded_bounds_wgs84[1], expanded_bounds_wgs84[3])
    }
    if tile_id not in tile_ids:
        tile_ids.add(tile_id)

    datasets = [ds_main]
    for tid in tile_ids:
        if tid == tile_id:
            continue
        path = f"/vsis3/{cn.s3_bucket_name}/{prefix_30m}{tid}{PEAT_30M_PATTERN}"
        try:
            datasets.append(rasterio.open(path))
        except Exception as e:
            logging.error(f"[{tid}|{chunk_str}] Could not open union mask: {e}")

    if not datasets:
        logging.info(f"[{tile_id}|{chunk_str}] No union mask data => skip")
        return

    mosaic_arr, mosaic_transform = merge_rasters(datasets)
    crs = datasets[0].crs
    for ds in datasets:
        ds.close()

    da = xr.DataArray(mosaic_arr[0], dims=("y", "x"))
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform(mosaic_transform)

    try:
        expanded_da = da.rio.clip_box(
            minx=expanded_bounds_crs[0],
            miny=expanded_bounds_crs[1],
            maxx=expanded_bounds_crs[2],
            maxy=expanded_bounds_crs[3],
        )
        if expanded_da.isnull().all():
            logging.info(
                f"[{tile_id}|{chunk_str}] No data in expanded raster. Skipping."
            )
            return
        orig_da = expanded_da.rio.clip_box(
            minx=minx_crs, miny=miny_crs, maxx=maxx_crs, maxy=maxy_crs
        )
        if orig_da.isnull().all():
            logging.info(
                f"[{tile_id}|{chunk_str}] No data in original bounds. Skipping."
            )
            return
    except Exception as e:
        logging.error(f"[{tile_id}|{chunk_str}] clip_box error: {e}")
        return

    mask_data = expanded_da.data == 1
    if np.all(mask_data == 0):
        logging.info(f"[{tile_id}|{chunk_str}] expanded mask is all zeros => skip")
        return

    s3_raw_prefix = cn.datasets[group][sub]['s3_raw']
    base_name = "roads" if "roads" in feature_type else "canals"
    line_dgdfs = []
    for tid in tile_ids:
        shp_name = f"{base_name}_{tid}.shp"
        s3_path = os.path.join(s3_raw_prefix, shp_name).replace("\\", "/")
        vsis3_path = f"/vsis3/{cn.s3_bucket_name}/{s3_path}"
        logging.info(f"[{tid}|{chunk_str}] Reading lines from {vsis3_path}")
        try:
            dgdf = dgpd.read_file(vsis3_path, npartitions=8)
        except Exception as e:
            logging.error(f"Could not read lines: {e}")
            continue
        if dgdf.crs is None:
            dgdf = dgdf.set_crs("EPSG:4326")
        if dask_gdf_is_empty(dgdf, data_path=vsis3_path):
            continue
        line_dgdfs.append(dgdf)

    if not line_dgdfs:
        logging.info(f"[{tile_id}|{chunk_str}] lines are empty => skip")
        return

    lines_dgdf = dgpd.concat(line_dgdfs)
    chunk_poly = box(*expanded_bounds_crs)
    lines_clip = dgpd.clip(lines_dgdf.to_crs(da.rio.crs), chunk_poly)
    if dask_gdf_is_empty(lines_clip):
        logging.info(f"[{tile_id}|{chunk_str}] lines do not intersect chunk => skip")
        return

    try:
        lines_gdf = lines_clip.compute()
    except FeatureError as exc:
        logging.error(f"FeatureError while reading lines: {exc}")
        return

    transform_exp = expanded_da.rio.transform()
    out_shape_exp = expanded_da.shape
    shapes = ((geom, 1) for geom in lines_gdf.geometry)
    burned = rasterize(
        shapes,
        out_shape=out_shape_exp,
        transform=transform_exp,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )

    binary = (burned > 0) & mask_data
    buffered_full = compute_buffered(binary, mask_data, max_dist=MAX_DISTANCE)

    pad_px = MAX_DISTANCE
    buffered = buffered_full[
        pad_px : pad_px + orig_da.shape[0], pad_px : pad_px + orig_da.shape[1]
    ]

    if np.all(buffered == 0):
        logging.info(f"[{tile_id}|{chunk_str}] buffered raster all zeros => skip")
        return

    xr_ras = xr.DataArray(
        buffered, dims=("y", "x"), coords={"y": orig_da.y, "x": orig_da.x}
    )
    xr_ras = xr_ras.rio.write_crs(da.rio.crs, inplace=True)
    xr_ras = xr_ras.rio.write_transform(orig_da.rio.transform(), inplace=True)
    local_dir_path = os.path.dirname(local_out)
    os.makedirs(local_dir_path, exist_ok=True)

    xr_ras.rio.to_raster(local_out, compress="lzw")

    hansen_filename = os.path.basename(local_out)
    warp_to_hansen_coiled(
        source_vrt_path=local_out,
        filename=hansen_filename,
        output_raster_s3_path_and_name=None,
        xmin=bounds_wgs84[0],
        ymin=bounds_wgs84[1],
        xmax=bounds_wgs84[2],
        ymax=bounds_wgs84[3],
        dt=uutil.string_to_gdal_dtype_mapping["Byte"],
        no_data=0,
        tiled=True,
        x_pixel_window=400,
        y_pixel_window=400,
    )
    hansen_local = os.path.join(tempfile.gettempdir(), hansen_filename)

    s3_client = boto3.client("s3")
    s3_client.upload_file(hansen_local, cn.s3_bucket_name, s3_out_key)
    logging.info(
        f"[{tile_id}|{chunk_str}] uploaded => s3://{cn.s3_bucket_name}/{s3_out_key}"
    )
    os.remove(local_out)
    if os.path.exists(hansen_local):
        os.remove(hansen_local)

    del expanded_da, lines_dgdf
    gc.collect()

    return f"[{tile_id}|{chunk_str}] done"


def process_tile(tile_id, feature_type, chunk_size=2, chunk_bounds=None):
    """Build delayed tasks for each chunk of ``tile_id``."""
    tile_bb = uutil.get_10x10_tile_bounds(tile_id)
    if chunk_bounds:
        chunks = [chunk_bounds]
    else:
        chunks = build_chunk_bounds(tile_bb, chunk_size=chunk_size)

    tasks = []
    for b in chunks:
        tasks.append(process_chunk_buffered(b, tile_id, feature_type))
    return tasks


def process_all_tiles(feature_type, chunk_size=2):
    """Iterate over the 30 m union mask prefix and build tasks for each tile."""
    prefix = cn.datasets["peat"]["union_mask"]["30m"]
    pattern = PEAT_30M_PATTERN
    import boto3
    s3 = boto3.client("s3")

    tasks = []
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=cn.s3_bucket_name, Prefix=prefix)
    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith(pattern):
                    base_name = os.path.basename(key)
                    tile_id = base_name.replace(pattern, "")
                    tasks.extend(process_tile(tile_id, feature_type, chunk_size=chunk_size))

    return tasks


def main(
    tile_id=None,
    feature_type="osm_roads",
    chunk_bounds=None,
    chunk_size=2,
    client="local",
):
    run_local = client == "local"
    cluster, dclient, run_local = uutil.connect_to_cluster(
        cluster_name="roads_canals",
        n_workers=20,
        region="us-east-1",
        run_local=run_local,
    )
    if run_local:
        logging.info("Running locally without Dask/Coiled.")
    else:
        logging.info(f"Using coiled cluster: {cluster.name}")

    try:
        if tile_id:
            tasks = process_tile(
                tile_id, feature_type, chunk_size=chunk_size, chunk_bounds=chunk_bounds
            )
        else:
            tasks = process_all_tiles(feature_type, chunk_size=chunk_size)

        if not tasks:
            logging.warning("No tasks generated; nothing to compute.")
            return

        dask.compute(*tasks)
    finally:
        if not run_local:
            dclient.close()
            cluster.close()
        logging.info("Completed processing")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Buffered distance for roads/canals using the 30m peat union mask"
    )
    parser.add_argument("--tile_id", help="Specific tile ID e.g. 00N_110E")
    parser.add_argument(
        "--feature_type",
        default="osm_roads",
        choices=["osm_roads", "osm_canals", "grip_roads"],
        help="Which lines to use",
    )
    parser.add_argument(
        "--chunk_bounds",
        default=None,
        help="Optional single chunk: 'minx,miny,maxx,maxy'",
    )
    parser.add_argument("--chunk_size", type=float, default=2, help="Chunk size in degrees")
    parser.add_argument("--client", default="local", choices=["local", "coiled"])
    args = parser.parse_args()

    cb = None
    if args.chunk_bounds:
        cb = [float(x) for x in args.chunk_bounds.split(",")]

    main(
        tile_id=args.tile_id,
        feature_type=args.feature_type,
        chunk_bounds=cb,
        chunk_size=args.chunk_size,
        client=args.client,
    )

"""
python -m src.scripts.preprocessing.roads_canals.global_datasets.02_roads_canals_buffered --tile_id 00N_110E --feature_type osm_roads --client coiled
"""