import os
import argparse
import sys
import subprocess
import logging
import posixpath

import pandas as pd
import geopandas as gpd
from dask.distributed import Client

# 1) Import your constants and utilities
import src.scripts.utilities.constants_and_names as cn
import src.preprocessing.pp_utilities as uu

# ---------------------------------------------------------------------------
# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ---------------------------------------------------------------------------
# Final integer mapping for categories
final_mapping = {
    'oil_palm': 1,
    'unknown_tc': 2,
    'short_rotation': 3,
    'long_rotation': 4,
    'unknown_rotation': 5
}

# Rasterization settings
RASTER_RES = 0.00025
RASTER_NODATA = 0
RASTER_DTYPE = 'Byte'

# CSV in S3
SDPT_RECLASS_S3_CSV = "climate/AFOLU_flux_model/organic_soils/inputs/raw/plantations/sdpt/updated_classified_planted_forest_species.csv"

# ---------------------------------------------------------------------------
def load_species_reclassification(local_csv_path=None, s3_csv_key=None):
    """
    Load reclassification CSV into a dict from 'vernacName' -> 'rotation_category'.
    If s3_csv_key is provided, download from S3 into cn.local_temp_dir first.
    """
    try:
        if s3_csv_key:
            # Download CSV into the local_temp_dir
            local_csv_path = os.path.join(cn.local_temp_dir, os.path.basename(s3_csv_key))
            uu.download_file_from_s3(s3_csv_key, local_csv_path, cn.s3_bucket_name)
            logging.info(f"Downloaded CSV from s3://{cn.s3_bucket_name}/{s3_csv_key} to {local_csv_path}")

        if not local_csv_path or not os.path.exists(local_csv_path):
            logging.warning(f"Reclassification CSV not found at {local_csv_path}. Using 'unknown_rotation'.")
            return {}

        df = pd.read_csv(local_csv_path)
        mapping = dict(zip(
            df["vernacName"].str.strip(),
            df["rotation_category"].str.strip()
        ))
        logging.info(f"Loaded {len(mapping)} species from {local_csv_path}")
        return mapping

    except Exception as e:
        logging.error(f"Error loading reclassification CSV: {e}")
        return {}

# ---------------------------------------------------------------------------
def classify_plantation(row, species_to_rotation):
    """
    For each row, assign a classification string:
      - If simpleType == "tree crops":
          if 'oil palm' in simpleName => "oil_palm"
          else => "unknown_tc"
      - If simpleType == "planted forest":
          use species_to_rotation
      - Otherwise, None
    """
    simple_name = str(row.get("simpleName", "")).strip().lower()
    vernac_name = str(row.get("vernacName", "")).strip()
    simple_type = str(row.get("simpleType", "")).strip().lower()

    if simple_type == "tree crops":
        if "oil palm" in simple_name:
            return "oil_palm"
        else:
            return "unknown_tc"
    elif simple_type == "planted forest":
        return species_to_rotation.get(vernac_name, "unknown_rotation")
    else:
        return None

# ---------------------------------------------------------------------------
def custom_download_shapefile_remove_prefix(
    s3_prefix: str,
    local_dir: str,
    local_shp_basename: str,
    s3_bucket_name: str
):
    """
    Download a shapefile from S3 to a local directory, removing the 'tile_' prefix
    when creating the local file name.

    Example:
      s3_prefix = "myfolder/tile_00N_110E"
      local_shp_basename = "00N_110E"
      => We'll download tile_00N_110E.shp -> 00N_110E.shp, etc.
    """
    uu.create_directory_if_not_exists(local_dir)
    extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']

    for ext in extensions:
        s3_path = s3_prefix + ext  # e.g. "myfolder/tile_00N_110E.shp"
        local_path = os.path.join(local_dir, local_shp_basename + ext)  # "C:/tmp/sdpt/00N_110E.shp"
        logging.info(f"Attempting to download: s3://{s3_bucket_name}/{s3_path} to {local_path}")
        try:
            uu.download_file_from_s3(s3_path, local_path, s3_bucket_name)
            if not os.path.exists(local_path):
                logging.error(f"Failed to download {s3_path} to {local_path}")
        except Exception as e:
            logging.error(f"Error downloading file from S3: {e}")

# ---------------------------------------------------------------------------
def rasterize_tile(tile_id, species_to_rotation, run_mode='default'):
    """
    Rasterize a single tile by:
      1. Downloading the shapefile from S3, removing "tile_" prefix locally
      2. Classifying rows
      3. Using gdal_rasterize to create a GeoTIFF
      4. Uploading to S3 if run_mode='default'
    """
    try:
        logging.info(f"Rasterizing tile {tile_id}")

        # 1) Build local shapefile folder: C:/tmp/sdpt
        local_shp_folder = os.path.join(cn.local_temp_dir, "sdpt")
        uu.create_directory_if_not_exists(local_shp_folder)

        # We'll store shapefile locally as "00N_110E.shp" (no tile_ prefix).
        local_shp_basename = tile_id

        # 2) Build the S3 prefix for the tile shapefile
        #    e.g. "climate/AFOLU_flux_model/organic_soils/inputs/raw/plantations/sdpt/tile_00N_110E"
        s3_shapefile_prefix = posixpath.join(
            cn.datasets['sdpt']['s3_raw'],  # .../sdpt
            f"tile_{tile_id}"               # tile_00N_110E
        )
        # => "climate/AFOLU_flux_model/organic_soils/inputs/raw/plantations/sdpt/tile_00N_110E"

        # 3) Download shapefile from S3
        custom_download_shapefile_remove_prefix(
            s3_prefix=s3_shapefile_prefix,
            local_dir=local_shp_folder,
            local_shp_basename=local_shp_basename,
            s3_bucket_name=cn.s3_bucket_name
        )

        # 4) Build local path to the .shp => "C:/tmp/sdpt/00N_110E.shp"
        local_shp_path = os.path.join(local_shp_folder, f"{tile_id}.shp")
        if not os.path.exists(local_shp_path):
            logging.warning(f"Shapefile {local_shp_path} not found. Skipping tile {tile_id}.")
            return

        # 5) Read into GeoDataFrame
        gdf = gpd.read_file(local_shp_path)
        if gdf.empty:
            logging.info(f"No features found in {local_shp_path}. Skipping.")
            return

        # 6) Classify each row
        gdf["plantation_type"] = gdf.apply(lambda row: classify_plantation(row, species_to_rotation), axis=1)
        gdf.dropna(subset=["plantation_type"], inplace=True)
        if gdf.empty:
            logging.info(f"No plantation rows in tile {tile_id} after classification.")
            return

        gdf["raster_val"] = gdf["plantation_type"].map(final_mapping)
        if gdf["raster_val"].isnull().all():
            logging.info(f"All plantation rows mapped to null for tile {tile_id}.")
            return

        # 7) Save shapefile for gdal_rasterize
        temp_shp_base = f"{tile_id}_temp"
        temp_shp_path = os.path.join(local_shp_folder, f"{temp_shp_base}.shp")
        gdf.to_file(temp_shp_path)

        # 8) Construct and run gdal_rasterize
        minx, miny, maxx, maxy = gdf.total_bounds
        out_raster_path = os.path.join(local_shp_folder, f"{tile_id}_plantations.tif")

        gdal_cmd = [
            'gdal_rasterize',
            '-a', 'raster_val',
            '-te', str(minx), str(miny), str(maxx), str(maxy),
            '-tr', str(RASTER_RES), str(RASTER_RES),
            '-a_nodata', str(RASTER_NODATA),
            '-init', str(RASTER_NODATA),
            '-ot', RASTER_DTYPE,
            '-co', 'COMPRESS=DEFLATE',
            '-co', 'TILED=YES',
            temp_shp_path,
            out_raster_path
        ]
        logging.info(f"Running GDAL command: {' '.join(gdal_cmd)}")
        subprocess.run(gdal_cmd, check=True)

        # 9) Remove temp shapefile
        for ext in ['shp', 'shx', 'dbf', 'prj', 'cpg']:
            p = os.path.join(local_shp_folder, f"{temp_shp_base}.{ext}")
            if os.path.exists(p):
                os.remove(p)

        if not os.path.exists(out_raster_path):
            logging.error(f"Rasterize failed to produce {out_raster_path}.")
            return

        # 10) If run_mode=='default', upload to S3
        if run_mode == 'default':
            s3_raster_key = posixpath.join(
                cn.datasets['sdpt']['s3_processed'],
                f"{tile_id}_plantations.tif"
            )
            logging.info(f"Uploading {out_raster_path} to s3://{cn.s3_bucket_name}/{s3_raster_key}")
            uu.upload_file_to_s3(out_raster_path, cn.s3_bucket_name, s3_raster_key)

            # Remove local after uploading
            os.remove(out_raster_path)
            logging.info(f"Removed local raster {out_raster_path}")
        else:
            logging.info(f"Test mode: local output retained at {out_raster_path}")

    except subprocess.CalledProcessError as e:
        logging.error(f"GDAL error for tile {tile_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error for tile {tile_id}: {e}")

# ---------------------------------------------------------------------------
def main(tile_id=None, run_mode='default', client_type='local'):
    """
    1. Optionally connect to Coiled
    2. Gather tile IDs from S3 (assuming tile_ prefix)
    3. Rasterize in parallel
    """
    # 1) Setup Dask
    if client_type == 'coiled':
        client, cluster = uu.setup_coiled_cluster()
        logging.info(f"Coiled cluster: {cluster.name}")
    else:
        client = Client()
        logging.info("Local Dask client started.")

    try:
        # 2) Load CSV from S3
        species_to_rotation = load_species_reclassification(
            s3_csv_key=SDPT_RECLASS_S3_CSV
        )

        # 3) Decide which tiles to process
        if tile_id:
            tile_ids = [tile_id]
        else:
            # e.g. "climate/AFOLU_flux_model/organic_soils/inputs/raw/plantations/sdpt"
            s3_folder = cn.datasets['sdpt']['s3_raw']  # no 'sdpt_by_tiles'
            existing_files = uu.list_s3_files(cn.s3_bucket_name, s3_folder)
            tile_ids = []
            for key in existing_files:
                # e.g. "tile_00N_110E.shp"
                base = os.path.basename(key)
                if base.startswith("tile_") and base.endswith(".shp"):
                    # tile_00N_110E.shp => 00N_110E
                    tid = base.replace("tile_", "").replace(".shp", "")
                    tile_ids.append(tid)

        logging.info(f"Tiles to process: {tile_ids}")

        # 4) Submit tasks
        futures = [client.submit(rasterize_tile, tid, species_to_rotation, run_mode) for tid in tile_ids]
        client.gather(futures)

    finally:
        # 5) Cleanup
        client.close()
        logging.info("Dask client closed.")
        if client_type == 'coiled':
            cluster.close()
            logging.info("Coiled cluster closed.")

    logging.info("All tiles processed successfully.")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rasterize plantation shapefiles with reclassification.')
    parser.add_argument('--tile_id', type=str, help='Tile ID (e.g., 00N_110E, no "tile_" prefix)')
    parser.add_argument('--run_mode', type=str, choices=['default', 'test'], default='default',
                        help='Run mode (default uploads to S3, test leaves local file)')
    parser.add_argument('--client', type=str, choices=['local', 'coiled'], default='local',
                        help='Dask client type (local or coiled)')
    args = parser.parse_args()

    # If no command-line args, run a sample
    if not any(sys.argv[1:]):
        logging.info("Running in test mode with sample tile_id 00N_110E...")
        main(tile_id='00N_110E', run_mode='default', client_type='local')
    else:
        main(tile_id=args.tile_id, run_mode=args.run_mode, client_type=args.client)
