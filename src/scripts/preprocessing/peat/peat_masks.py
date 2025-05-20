#!/usr/bin/env python
# peat_masks.py ─ robust peat-mask tiler for GPD | PEATML | PEATMAP
# (Integrated: 2025-04-30)

import os
import argparse
import logging
import tempfile
from pathlib import Path

import pandas as pd
import geopandas as gpd
import rasterio
import fiona
import botocore
import dask
from dask import delayed
from dask.distributed import LocalCluster, Client
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

################################################################################
# Helper Functions
################################################################################

def bounds_for_tile(tid):
    return uutil.get_10x10_tile_bounds(tid)

def output_paths(ds_key, tid):
    ds = cn.datasets["peat"][ds_key]
    fname = f"{tid}_{ds_key}_mask.tif"
    local = Path(ds["local_processed"]) / fname
    s3_key = f"{ds['s3_processed'].rstrip('/')}/{fname}"
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
def vector_job(tid, mode="default"):
    """
    Processes the 'peatmap' dataset (vector) for a given tile. Clips to tile
    bounds, then rasterizes at the final resolution (EPSG:4326, 0.00025).
    """
    ds_key = "peatmap"
    local_out, s3_out = output_paths(ds_key, tid)

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
def mosaic_and_warp_raster(ds_key, tid, mode="default"):
    """
    If ds['s3_raw'] is a single .tif, skip listing and warp that file directly.
    Otherwise, it's a folder containing multiple .tif => build a VRT if needed.
    """
    ds = cn.datasets["peat"][ds_key]
    local_out, s3_out = output_paths(ds_key, tid)

    if mode != "test" and uutil.s3_file_exists(BUCKET, s3_out):
        log.info(f"[{ds_key}|{tid}] already on S3. Skipping.")
        return

    raw_path = ds['s3_raw']
    # If it doesn't start with s3://, add s3://bucket/ prefix
    if not raw_path.startswith("s3://"):
        raw_path = f"s3://{BUCKET}/{raw_path.lstrip('/')}"

    # Single .tif approach
    if raw_path.lower().endswith('.tif'):
        source_for_warp = raw_path
        log.info(f"[{ds_key}|{tid}] Using single-file approach: {source_for_warp}")

    else:
        # Possibly multiple .tif => do the mosaic approach
        # Remove s3://bucket/ to get the prefix for listing
        listing_prefix = raw_path.replace(f"s3://{BUCKET}/", "", 1)
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

    # Warp to 10×10 deg output
    xmin, ymin, xmax, ymax = bounds_for_tile(tid)
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
        with rasterio.open(local_out, "r+") as dst:
            arr = dst.read(1)
            arr = (arr > threshold_value).astype("uint8")
            dst.write(arr, 1)

    if mode != "test":
        uutil.upload_file_to_s3(str(local_out), BUCKET, s3_out)
        local_out.unlink()

    log.info(f"[{ds_key}|{tid}] completed.")

################################################################################
# Build and submit tasks
################################################################################
def build_tasks(tids, ds_keys, mode):
    tasks = []
    for tid in tids:
        for k in ds_keys:
            if k == "peatmap":
                tasks.append(delayed(vector_job)(tid, mode))
            else:
                tasks.append(delayed(mosaic_and_warp_raster)(k, tid, mode))
    return tasks

################################################################################
# Main orchestrator
################################################################################
def main(tile_id=None, dataset=None, client="coiled", run_mode="default"):
    if client == "local":
        cluster = LocalCluster(processes=False, dashboard_address=None)
        client_obj = Client(cluster)
        log.info("Running locally.")
    else:
        cluster, client = uutil.connect_to_cluster(
            cluster_name="roads_canals",
            n_workers=20,
            region="us-east-1",
        )
        log.info(f"Running on Coiled: {cluster.name}")

    ds_keys = [dataset] if dataset else ["peatml", "gpd", "peatmap"]
    tids = [tile_id] if tile_id else cn.tile_id_list

    log.info(f"Datasets: {ds_keys}, Tiles: {len(tids)}")
    tasks = build_tasks(tids, ds_keys, run_mode)

    dask.compute(*tasks)

    # Close out
    client_obj.close()
    if client == "coiled":
        cluster.close()
    log.info("All tasks completed.")

################################################################################
# CLI
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust peat-mask tiler")
    parser.add_argument("--tile_id", help="Single tile ID (optional)")
    parser.add_argument("--dataset", choices=["peatml", "gpd", "peatmap"], help="Dataset (optional)")
    parser.add_argument("--client", default="coiled", choices=["local", "coiled"], help="Run mode (default: coiled)")
    parser.add_argument("--run_mode", default="default", choices=["default", "test"], help="Run mode")
    args = parser.parse_args()
    main(args.tile_id, args.dataset, args.client, args.run_mode)