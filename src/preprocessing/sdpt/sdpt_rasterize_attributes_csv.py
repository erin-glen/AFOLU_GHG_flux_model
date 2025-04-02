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
            # Download CSV into cn.local_temp_dir
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
      - Otherwise => None
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
def rasterize_tile(tile_id, species_to_rotation, run_mode='default'):
    """
    Rasterize a single tile by:
      1. If run_mode == 'default', check if final TIF in S3 => skip if found
      2. Build /vsis3/ path for tile_{tile_id}.shp
      3. Read shapefile in memory
      4. Classify each record
      5. Use gdal_rasterize to create GeoTIFF locally
      6. Upload to S3 if run_mode='default'
    """
    try:
        logging.info(f"Rasterizing tile {tile_id}")

        # If run_mode=default => check if final TIF is already in S3
        if run_mode == 'default':
            s3_raster_key = posixpath.join(
                cn.datasets['sdpt']['s3_processed'],
                f"{tile_id}_plantations.tif"
            )
            if uu.s3_file_exists(cn.s3_bucket_name, s3_raster_key):
                logging.info(f"Tile {tile_id}: final TIF already in s3://{cn.s3_bucket_name}/{s3_raster_key}, skipping.")
                return

        # 1) Build the /vsis3/ path for the tile shapefile
        # e.g. "/vsis3/gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/raw/plantations/sdpt/tile_00N_110E.shp"
        s3_shp_uri = f"/vsis3/{cn.s3_bucket_name}/{cn.datasets['sdpt']['s3_raw']}/tile_{tile_id}.shp"
        logging.info(f"Reading shapefile for tile {tile_id} from: {s3_shp_uri}")

        # 2) Read shapefile directly from S3
        gdf = gpd.read_file(s3_shp_uri)
        if gdf.empty:
            logging.info(f"No features found in tile_{tile_id}.shp. Skipping.")
            return

        # 3) Classify each row
        gdf["plantation_type"] = gdf.apply(lambda row: classify_plantation(row, species_to_rotation), axis=1)
        gdf.dropna(subset=["plantation_type"], inplace=True)
        if gdf.empty:
            logging.info(f"No plantation rows in tile {tile_id} after classification.")
            return

        gdf["raster_val"] = gdf["plantation_type"].map(final_mapping)
        if gdf["raster_val"].isnull().all():
            logging.info(f"All plantation rows mapped to null for tile {tile_id}.")
            return

        # 4) Write a temporary shapefile locally for gdal_rasterize
        out_folder = os.path.join(cn.local_temp_dir, "sdpt_no_chunks")
        uu.create_directory_if_not_exists(out_folder)
        temp_shp_base = f"{tile_id}_temp"
        temp_shp_path = os.path.join(out_folder, f"{temp_shp_base}.shp")
        gdf.to_file(temp_shp_path)

        minx, miny, maxx, maxy = gdf.total_bounds
        final_raster_path = os.path.join(out_folder, f"{tile_id}_plantations.tif")

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
            final_raster_path
        ]
        logging.info(f"Running GDAL command: {' '.join(gdal_cmd)}")
        subprocess.run(gdal_cmd, check=True)

        # Remove the temp shapefile pieces
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            p = os.path.join(out_folder, f"{temp_shp_base}{ext}")
            if os.path.exists(p):
                os.remove(p)

        if not os.path.exists(final_raster_path):
            logging.error(f"gdal_rasterize failed to produce {final_raster_path}.")
            return

        # 5) If run_mode=='default', upload to S3 and remove local
        if run_mode == 'default':
            s3_raster_key = posixpath.join(
                cn.datasets['sdpt']['s3_processed'],
                f"{tile_id}_plantations.tif"
            )
            logging.info(f"Uploading {final_raster_path} to s3://{cn.s3_bucket_name}/{s3_raster_key}")
            uu.upload_file_to_s3(final_raster_path, cn.s3_bucket_name, s3_raster_key)
            os.remove(final_raster_path)
            logging.info(f"Removed local raster {final_raster_path}")
        else:
            logging.info(f"Test mode => local raster retained => {final_raster_path}")

    except subprocess.CalledProcessError as e:
        logging.error(f"GDAL error for tile {tile_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error for tile {tile_id}: {e}")

# ---------------------------------------------------------------------------
def main(tile_id=None, run_mode='default', client_type='local'):
    """
    1. Optionally connect to Coiled
    2. Gather tile IDs from S3 (assuming tile_{tile_id}.shp)
    3. Rasterize in parallel, skipping if final TIF in S3 (run_mode=default)
    """
    # 1) Setup Dask
    if client_type == 'coiled':
        client, cluster = uu.setup_coiled_cluster()
        logging.info(f"Coiled cluster: {cluster.name}")
    else:
        client = Client()
        logging.info("Local Dask client started.")

    try:
        # 2) Load the classification CSV from S3 (downloaded locally)
        species_to_rotation = load_species_reclassification(
            s3_csv_key=SDPT_RECLASS_S3_CSV
        )

        # 3) Determine tile IDs
        if tile_id:
            tile_ids = [tile_id]
        else:
            s3_folder = cn.datasets['sdpt']['s3_raw']
            existing_files = uu.list_s3_files(cn.s3_bucket_name, s3_folder)
            tile_ids = []
            for key in existing_files:
                base = os.path.basename(key)
                if base.startswith("tile_") and base.endswith(".shp"):
                    tid = base.replace("tile_", "").replace(".shp", "")
                    tile_ids.append(tid)

        logging.info(f"Tiles to process: {tile_ids}")

        # 4) Submit tasks to Dask
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
    parser = argparse.ArgumentParser(
        description='Non-chunked SDPT script reading shapefiles from /vsis3/, skipping tiles if final TIF in S3.'
    )
    parser.add_argument('--tile_id', type=str, help='Tile ID (e.g., 00N_110E, no "tile_" prefix).')
    parser.add_argument('--run_mode', type=str, choices=['default', 'test'], default='default',
                        help='Run mode => default => upload final TIF to S3, test => keep TIF locally.')
    parser.add_argument('--client', type=str, choices=['local','coiled'], default='local',
                        help='Dask client type => local or coiled.')
    args = parser.parse_args()

    # If no command-line args, run a sample
    if not any(sys.argv[1:]):
        logging.info("No CLI args => sample run tile_id=00N_110E, run_mode=default, local client.")
        # main(tile_id='00N_110E', run_mode='default', client_type='local')
        main(run_mode='default', client_type='local')
    else:
        main(tile_id=args.tile_id, run_mode=args.run_mode, client_type=args.client)
