import os
import argparse
import sys
import subprocess
import logging
import posixpath
import gc

import pandas as pd
import geopandas as gpd
from dask.distributed import Client

# Import constants and utilities
import src.scripts.utilities.constants_and_names as cn
import src.scripts.preprocessing.pp_utilities as uu

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Constants
final_mapping = {
    'oil_palm': 1,
    'unknown_tc': 2,
    'short_rotation': 3,
    'long_rotation': 4,
    'unknown_rotation': 5
}

RASTER_RES = 0.00025
RASTER_NODATA = 0
RASTER_DTYPE = 'Byte'

SDPT_RECLASS_S3_CSV = "climate/AFOLU_flux_model/organic_soils/inputs/raw/plantations/sdpt/updated_classified_planted_forest_species.csv"


def load_species_reclassification(local_csv_path=None, s3_csv_key=None):
    try:
        if s3_csv_key:
            local_csv_path = os.path.join(cn.local_temp_dir, os.path.basename(s3_csv_key))
            uu.download_file_from_s3(s3_csv_key, local_csv_path, cn.s3_bucket_name)

        df = pd.read_csv(local_csv_path)
        mapping = dict(zip(
            df["vernacName"].str.strip(),
            df["rotation_category"].str.strip()
        ))
        return mapping
    except Exception as e:
        logging.error(f"Error loading reclassification CSV: {e}")
        return {}


def classify_plantation(row, species_to_rotation):
    simple_name = str(row.get("simpleName", "")).strip().lower()
    vernac_name = str(row.get("vernacName", "")).strip()
    simple_type = str(row.get("simpleType", "")).strip().lower()

    if simple_type == "tree crops":
        return "oil_palm" if "oil palm" in simple_name else "unknown_tc"
    elif simple_type == "planted forest":
        return species_to_rotation.get(vernac_name, "unknown_rotation")
    return None


def rasterize_tile(tile_id, species_to_rotation, run_mode='default'):
    try:
        logging.info(f"Rasterizing tile {tile_id}")

        s3_raster_key = posixpath.join(cn.datasets['sdpt']['s3_processed'], f"{tile_id}_plantations.tif")
        if run_mode == 'default' and uu.s3_file_exists(cn.s3_bucket_name, s3_raster_key):
            logging.info(f"Tile {tile_id} already processed. Skipping.")
            return

        s3_shp_uri = f"/vsis3/{cn.s3_bucket_name}/{cn.datasets['sdpt']['s3_raw']}/tile_{tile_id}.shp"
        gdf = gpd.read_file(s3_shp_uri)
        if gdf.empty:
            logging.info(f"No features in {tile_id}. Skipping.")
            return

        # Shorten the field name explicitly to avoid ESRI warnings
        gdf["ptype"] = gdf.apply(lambda row: classify_plantation(row, species_to_rotation), axis=1)
        gdf.dropna(subset=["ptype"], inplace=True)
        if gdf.empty:
            logging.info(f"No plantations after classification in {tile_id}.")
            return

        gdf["rval"] = gdf["ptype"].map(final_mapping)
        if gdf["rval"].isnull().all():
            logging.info(f"All null classifications in {tile_id}.")
            return

        out_folder = os.path.join(cn.local_temp_dir, "sdpt_no_chunks")
        uu.create_directory_if_not_exists(out_folder)
        temp_shp_base = f"{tile_id}_temp"
        temp_shp_path = os.path.join(out_folder, f"{temp_shp_base}.shp")

        # Save with shortened field names explicitly
        gdf[['geometry', 'rval']].to_file(temp_shp_path)

        minx, miny, maxx, maxy = gdf.total_bounds
        final_raster_path = os.path.join(out_folder, f"{tile_id}_plantations.tif")

        gdal_cmd = [
            'gdal_rasterize', '-a', 'rval',
            '-te', str(minx), str(miny), str(maxx), str(maxy),
            '-tr', str(RASTER_RES), str(RASTER_RES),
            '-a_nodata', str(RASTER_NODATA), '-init', str(RASTER_NODATA),
            '-ot', RASTER_DTYPE, '-co', 'COMPRESS=DEFLATE', '-co', 'TILED=YES',
            temp_shp_path, final_raster_path
        ]
        subprocess.run(gdal_cmd, check=True)

        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            p = os.path.join(out_folder, f"{temp_shp_base}{ext}")
            if os.path.exists(p):
                os.remove(p)

        if run_mode == 'default':
            uu.upload_file_to_s3(final_raster_path, cn.s3_bucket_name, s3_raster_key)
            os.remove(final_raster_path)

        del gdf
        gc.collect()
        logging.info(f"Memory cleared after {tile_id}")

    except subprocess.CalledProcessError as e:
        logging.error(f"GDAL error for {tile_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error for {tile_id}: {e}")
        del gdf
        gc.collect()


def main(tile_id=None, run_mode='default', client_type='local'):
    client = Client(n_workers=4, threads_per_worker=1, memory_limit='4GB')

    try:
        species_to_rotation = load_species_reclassification(s3_csv_key=SDPT_RECLASS_S3_CSV)

        if tile_id:
            tile_ids = [tile_id]
        else:
            s3_folder = cn.datasets['sdpt']['s3_raw']
            existing_files = uu.list_s3_files(cn.s3_bucket_name, s3_folder)
            tile_ids = [
                base.replace("tile_", "").replace(".shp", "")
                for key in existing_files
                if (base := os.path.basename(key)).startswith("tile_") and base.endswith(".shp")
            ]

        logging.info(f"{len(tile_ids)} tiles to process.")

        futures = [client.submit(rasterize_tile, tid, species_to_rotation, run_mode) for tid in tile_ids]
        client.gather(futures)

    finally:
        client.close()
        logging.info("Dask client closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile_id', type=str)
    parser.add_argument('--run_mode', choices=['default', 'test'], default='default')
    parser.add_argument('--client', choices=['local', 'coiled'], default='local')
    args = parser.parse_args()

    if not any(sys.argv[1:]):
        main(run_mode='default', client_type='local')
    else:
        main(tile_id=args.tile_id, run_mode=args.run_mode, client_type=args.client)
