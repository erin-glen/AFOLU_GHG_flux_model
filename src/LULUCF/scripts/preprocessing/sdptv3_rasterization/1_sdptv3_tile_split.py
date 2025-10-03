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
import argparse
import shutil
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

def clean_tmp_dir(tmp_dir):

    logger_worker = lu.setup_logging_worker()
    
    try:
        shutil.rmtree(tmp_dir)
        if not os.path.exists(tmp_dir):
            lu.print_and_log(f"Deleted local tmp dir: {tmp_dir}", True, logger_worker)
    except Exception as e:
        lu.print_and_log(f"Error deleting local tmp dir: {tmp_dir} — {e}", False, logger_worker)
#TODO: Move to uu

# Processes SDPT polygons by tiles from GDB layers
def split_sdpt_by_tiles(tile_grid_path, countries_gdb, out_s3_dir, tile_id_field="tile_id"):

    logger_worker = lu.setup_logging_worker()

    # Step 1: Read in tile grid and check that tile_id_field exists
    lu.print_and_log(f"Reading tile grid: {tile_grid_path}", False, logger_worker)
    tiles_gdf = gpd.read_file(tile_grid_path)
    if tile_id_field not in tiles_gdf.columns:
        raise ValueError(f"Tile grid missing field '{tile_id_field}'.")

    # Step 2: Check the number of country layers in sdptv3 gdb and create df with the spatial extent of each country's polygons
    #TODO: Check that this works using vsis3 or read into python object
    layer_names = fiona.listlayers(countries_gdb)
    lu.print_and_log(f"Found {len(layer_names)} layers in '{countries_gdb}'.", False, logger_worker)

    country_bb_df = pd.DataFrame([
        {"layer_name": layer_name, "bounds": get_layer_bounds(countries_gdb, layer_name)}
        for layer_name in layer_names
        if get_layer_bounds(countries_gdb, layer_name) is not None
    ])

    # TODO: Remove if statement and un-indent
    # TODO: Parallelize instead of for loop
    for idx, tile_row in tiles_gdf.iterrows():
        tile_geom = tile_row.geometry
        tile_id = tile_row[tile_id_field]
        tile_bounds = tile_geom.bounds

        if tile_id == '40N_120E':

            tmp_dir = tempfile.mkdtemp(prefix="tile_clip_")
            out_dir = os.path.join(tmp_dir, tile_id)
            os.makedirs(out_dir, exist_ok=True)

            tile_name = f"{tile_id}_sdptv3"
            tile_out_path_local = os.path.join(out_dir, f"{tile_name}.shp")
            tile_out_path_s3 = f"{out_s3_dir}{tile_name}.shp"

            if uu.exists_in_s3(tile_out_path_s3):
                lu.print_and_log(f"Tile {tile_id}: already exists in s3. Skipping.", False, logger_worker)
                continue

            relevant_layers = country_bb_df[
                country_bb_df["bounds"].apply(lambda b: bounding_boxes_intersect(b, tile_bounds))
            ]

            if relevant_layers.empty:
                lu.print_and_log(f"Tile {tile_id}: no intersecting layers. Skipping.", False, logger_worker)
                continue

            all_clips = []

            for _, layer_row in relevant_layers.iterrows():
                layer_name = layer_row["layer_name"]
                lu.print_and_log(f" - Processing tile {tile_id} with layer '{layer_name}'", False, logger_worker)

                try:
                    ctry_gdf = gpd.read_file(countries_gdb, layer=layer_name, bbox=tile_bounds)
                    if ctry_gdf.empty:
                        lu.print_and_log(f"   Layer '{layer_name}' has no features in bbox. Skipping.", False, logger_worker)
                        continue

                    clipped = gpd.clip(ctry_gdf, tile_geom)

                    if not clipped.empty:
                        clipped["source_layer"] = layer_name
                        all_clips.append(clipped)

                except Exception as e:
                    lu.print_and_log(f"   Error processing layer '{layer_name}': {e}", False, logger_worker)

            if all_clips:
                merged_gdf = gpd.GeoDataFrame(pd.concat(all_clips, ignore_index=True), crs=all_clips[0].crs)

                # --- NEW FIX: Only polygons allowed for shapefile format ---
                merged_gdf = filter_to_polygons(merged_gdf)

                if merged_gdf.empty:
                    lu.print_and_log(f"Tile {tile_id}: no polygon features after filtering. Skipping write.", False, logger_worker)
                    continue

                lu.print_and_log(f"Writing {len(merged_gdf)} polygon features to {tile_out_path_local}", False, logger_worker)
                merged_gdf.to_file(tile_out_path_local)
            else:
                lu.print_and_log(f"Tile {tile_id}: no features after clipping.", False, logger_worker)

            # Step 5: Upload all related files to s3 and check that they have been successfully uploaded
            uploaded = []
            exts = [".shp", ".shx", ".dbf", ".prj", ".cpg"]
            for ext in exts:
                p = os.path.join(out_dir, f"{tile_name}{ext}")
                if os.path.exists(p):
                    s3_key = out_s3_dir + os.path.basename(p)
                    uu.upload_s3_file(s3_key, p)
                    if uu.exists_in_s3(s3_key):
                        uploaded.append(s3_key)
            #TODO: Skip this step if running locally


def main(cluster_name, run_local):

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)
    client

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, f"Creating sdptv3 vector tiles",
                                                                   run_local, 'standard',f'Creating sdptv3 vector tiles')

    # Data sources
    tile_grid_path = cn.fishnet_10x10deg_uri.replace("s3://", "/vsis3/")
    countries_gdb = "/vsis3/gfw2-data/plantations/sdpt_v3/sdpt_v3_final.gdb/sdpt_v3_final.gdb"
    out_s3_dir = "s3://gfw2-data/plantations/sdpt_v3/sdpt_v3_vector_tiles/tiles_10x10/"
    tile_id_field = "tile_id"
    #TODO: Update paths in constants and names and use .replace("s3://", "/vsis3/") to make them all /vsis3/

    #TODO: Change so that this runs locally or in coiled
    split_sdpt_by_tiles(tile_grid_path, countries_gdb, out_s3_dir, tile_id_field)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create sdptv3 vector tiles")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    run_local = args.run_local

    main(cluster_name, run_local)