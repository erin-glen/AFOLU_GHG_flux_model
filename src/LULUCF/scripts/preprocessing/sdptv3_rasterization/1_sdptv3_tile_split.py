"""

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test:
python -m src.LULUCF.scripts.preprocessing.sdptv3_rasterization.1_sdptv3_tile_split
python -m src.LULUCF.scripts.preprocessing.sdptv3_rasterization.1_sdptv3_tile_split --run_local

python -m src.utilities.create_cluster -n 1 -t 1 -m 16 -cn 10x10_tiles
python -m src.LULUCF.scripts.preprocessing.sdptv3_rasterization.1_rasterize_SDPTv3 -cn 10x10_tiles

"""
import os
import fiona
import boto3
import argparse
import shutil
import zipfile
import tempfile
import geopandas as gpd
import pandas as pd
from packaging.version import Version, InvalidVersion
import geopandas.io.file

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import universal_utilities as uu
from src.utilities import log_utilities as lu

# --- Robust GDAL Version Patch for ArcGIS Pro Environment ---
# This patch resolves the "Invalid version: '3.8.1e'" error that arises due to ArcGIS Pro's non-standard GDAL versioning.
original_to_file_fiona = geopandas.io.file._to_file_fiona

def patched_to_file_fiona(df, filename, driver, schema, crs=None, mode="w", **kwargs):
    try:
        gdal_version = fiona.env.get_gdal_release_name()
        Version(gdal_version)
    except InvalidVersion:
        gdal_version = "3.8.1"
        kwargs['GEOPANDAS_FIX_GDAL_VERSION'] = gdal_version

    original_version_class = geopandas.io.file.Version
    geopandas.io.file.Version = lambda _: Version("3.8.1")

    try:
        return original_to_file_fiona(df, filename, driver, schema, crs, mode, **kwargs)
    finally:
        geopandas.io.file.Version = original_version_class

geopandas.io.file._to_file_fiona = patched_to_file_fiona
# --- End GDAL Version Patch ---

# Quickly obtain bounding box from layer metadata.
def get_layer_bounds(gdb_path, layer_name):

    logger_worker = lu.setup_logging_worker()

    try:
        with fiona.open(gdb_path, layer=layer_name) as src:
            return src.bounds if src.bounds else None
    except Exception as e:
        lu.print_and_log(f"Error reading layer '{layer_name}': {e}", False, logger_worker)
        return None

# Check if two bounding boxes intersect
def bounding_boxes_intersect(b1, b2):
    return not (b2[0] > b1[2] or b2[2] < b1[0] or b2[1] > b1[3] or b2[3] < b1[1])

# Keep only polygon and multipolygon geometries for shapefile compatibility.
def filter_to_polygons(gdf):
    return gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

def list_and_print_files_in_dir(local_dir):
    logger_worker = lu.setup_logging_worker()

    for root, dirs, files in os.walk(local_dir):
        for d in dirs:
            lu.print_and_log(f"[DIR ] {os.path.relpath(os.path.join(root, d), local_dir)}", False, logger_worker)
        for f in files:
            lu.print_and_log(f"[FILE] {os.path.relpath(os.path.join(root, f), local_dir)}", False, logger_worker)
#TODO: Move to uu

def clean_tmp_dir(tmp_dir):

    logger_worker = lu.setup_logging_worker()
    
    try:
        shutil.rmtree(tmp_dir)
        if not os.path.exists(tmp_dir):
            lu.print_and_log(f"Deleted local tmp dir: {tmp_dir}", True, logger_worker)
    except Exception as e:
        lu.print_and_log(f"Error deleting local tmp dir: {tmp_dir} — {e}", False, logger_worker)
#TODO: Move to uu

#Download entire s3 dir to local tmp path
def download_s3_dir(s3_uri_dir, local_dir):
    s3 = boto3.client("s3")
    bucket, prefix = uu.split_s3_path(s3_uri_dir)

    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    found_any = False
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            found_any = True
            key = obj["Key"]
            rel = key[len(prefix):]  # relative path under .gdb
            if not rel:                 # skip the directory “placeholder”
                continue
            dst = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            s3.download_file(bucket, key, dst)
    if not found_any:
        raise FileNotFoundError(f"No objects found under {s3_uri_dir}")
#TODO: Move to uu

# Processes SDPT polygons by tiles from GDB layers
def split_sdpt_by_tile(country_bb_df, countries_gdb, tile_row, tile_id_field, out_s3_dir):

    logger_worker = lu.setup_logging_worker()

    # Get tile id and bounding coordinates for this tile
    tile_geom = tile_row.geometry
    tile_id = tile_row[tile_id_field]
    tile_bounds = tile_geom.bounds

    # TODO: Remove if statement and un-indent the rest of the code
    if tile_id == '40N_120E':

        # Step 1: Organize tile shapefile tmp dir and output paths
        tmp_dir = tempfile.mkdtemp(prefix="tile_clip_")
        out_dir = os.path.join(tmp_dir, tile_id)
        os.makedirs(out_dir, exist_ok=True)

        tile_name = f"{tile_id}_sdptv3"
        tile_out_path_local = os.path.join(out_dir, f"{tile_name}.shp")
        tile_out_path_s3 = f"{out_s3_dir}{tile_name}.shp"

        if uu.exists_in_s3(tile_out_path_s3):
            return f"Skipped tile {tile_id}. Already exists in s3."

        # Step 2: Filter to only countries with polygons in the tile bounding box
        relevant_layers = country_bb_df[
            country_bb_df["bounds"].apply(lambda b: bounding_boxes_intersect(b, tile_bounds))
        ]

        if relevant_layers.empty:
            return f"Skipped tile {tile_id}. No intersecting layers."

        # Step 3: Clip polygons to tile bounds for all relevant countries
        all_clips = []

        for _, layer_row in relevant_layers.iterrows():
            layer_name = layer_row["layer_name"]
            lu.print_and_log(f"     Processing tile {tile_id} with layer '{layer_name}'", False, logger_worker)

            try:
                ctry_gdf = gpd.read_file(countries_gdb, layer=layer_name, bbox=tile_bounds)
                if ctry_gdf.empty:
                    lu.print_and_log(f"     Layer '{layer_name}' has no features in bbox. Skipping.", False, logger_worker)
                    continue

                clipped = gpd.clip(ctry_gdf, tile_geom)

                if not clipped.empty:
                    clipped["source_layer"] = layer_name
                    all_clips.append(clipped)

            except Exception as e:
                lu.print_and_log(f"     Error processing layer '{layer_name}': {e}", False, logger_worker)

        if all_clips:
            merged_gdf = gpd.GeoDataFrame(pd.concat(all_clips, ignore_index=True), crs=all_clips[0].crs)
            merged_gdf = filter_to_polygons(merged_gdf)       # Only polygons allowed for shapefile format

            if merged_gdf.empty:
                return f"Skipped tile {tile_id}: no polygon features after filtering"

            lu.print_and_log(f"     Writing {len(merged_gdf)} polygon features to {tile_out_path_local}", False, logger_worker)
            merged_gdf.to_file(tile_out_path_local)
        else:
            lu.print_and_log(f"     Tile {tile_id}: no features after clipping.", False, logger_worker)

        # Step 4: Upload all files to s3 and check that they have been successfully uploaded
        uploaded = []
        exts = [".shp", ".shx", ".dbf", ".prj", ".cpg"]
        for ext in exts:
            p = os.path.join(out_dir, f"{tile_name}{ext}")
            if os.path.exists(p):
                s3_key = out_s3_dir + os.path.basename(p)
                uu.upload_s3_file(s3_key, p)
                if uu.exists_in_s3(s3_key):
                    uploaded.append(s3_key)

        # Step 5: Cleanup local tile files and return message
        clean_tmp_dir(tmp_dir)
        if len(uploaded) == 5:
            return f"Success for {tile_id}. ({len(uploaded)} files uploaded): {uu.timestr('time')}"
        else:
            return f"Error for {tile_id}. Only ({len(uploaded)} files uploaded): {uu.timestr('time')}"


def main(cluster_name, run_local):

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)
    client

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, f"Creating sdptv3 vector tiles",
                                                                   run_local, 'standard',f'Creating sdptv3 vector tiles')

    # Data sources
    tile_grid_path = cn.fishnet_10x10deg_uri.replace("s3://", "/vsis3/")    #can alternatively use 1x1 deg chunks
    zip_file = "sdpt_v3_final.gdb.zip"
    gdb_name = "sdpt_v3_final.gdb"
    countries_gdb_s3_dir = f"s3://gfw2-data/plantations/sdpt_v3/{zip_file}"
    out_s3_dir = "s3://gfw2-data/plantations/sdpt_v3/sdpt_v3_vector_tiles/tiles_10x10/"
    tile_id_field = "tile_id"
    #TODO: Update paths in constants and names and use .replace("s3://", "/vsis3/")

    # Step 1: Read in tile grid (using vsis3) and check that tile_id_field exists
    main_logger.info(f"Reading tile grid: {tile_grid_path}")
    tiles_gdf = gpd.read_file(tile_grid_path)
    if tile_id_field not in tiles_gdf.columns:
        raise ValueError(f"Tile grid missing field '{tile_id_field}'.")

    # Step 2: Read in zipped gdb and extract (using local disk space)
    tmp_dir = tempfile.mkdtemp(prefix="sdpt_gdb_")

    #Prepare local zipped dir and exrtaction dir
    zip_dir = os.path.join(tmp_dir, 'zipped_gdb')
    extract_dir = os.path.join(tmp_dir, 'unzipped_gdb')
    os.makedirs(zip_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    local_zip = os.path.join(zip_dir, zip_file)

    main_logger.info(f"Downloading {countries_gdb_s3_dir} -> {local_zip}: {uu.timestr('time')}")
    uu.download_s3_file(countries_gdb_s3_dir, local_zip)
    #list_and_print_files_in_dir(zip_dir)
    #Note: This took about 9 minutes locally to download zipped gdb

    main_logger.info(f"Unzipping to {extract_dir}: {uu.timestr('time')}")
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(extract_dir)
    #list_and_print_files_in_dir(extract_dir)
    # Note: This took about 9 minutes locally

    countries_gdb = os.path.join(extract_dir, gdb_name)
    #os.makedirs(countries_gdb, exist_ok=True)
    #TODO: read in .zip (faster) and extract

    #download_s3_dir(countries_gdb_s3_dir, countries_gdb)
    #TODO: make sure it downloads to the correct folder structure

    # Count the number of country layers in sdptv3 gdb and create df with the spatial extent of each country's polygons
    layer_names = fiona.listlayers(countries_gdb)
    main_logger.info(f"Found {len(layer_names)} layers in '{countries_gdb}'.")

    country_bb_df = pd.DataFrame([
        {"layer_name": layer_name, "bounds": get_layer_bounds(countries_gdb, layer_name)}
        for layer_name in layer_names
        if get_layer_bounds(countries_gdb, layer_name) is not None
    ])

    # Step 3: Create sdpt shapefile for tiles
    if run_local:
        for idx, tile_row in tiles_gdf.iterrows():
            tile_result = split_sdpt_by_tile(country_bb_df, countries_gdb, tile_row, tile_id_field, out_s3_dir)
            tile_result
    else:
        futures = []
        for idx, tile_row in tiles_gdf.iterrows():
            future = client.submit(split_sdpt_by_tile, tiles_gdf, country_bb_df, countries_gdb, tile_row, tile_id_field, out_s3_dir)
            futures.append(future)
        tile_results = client.gather(futures)
        tile_results

    # Clean tmp disk space
    clean_tmp_dir(tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create sdptv3 vector tiles")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    run_local = args.run_local

    main(cluster_name, run_local)