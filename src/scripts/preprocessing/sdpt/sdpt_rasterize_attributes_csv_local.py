import os
import argparse
import subprocess
import logging
import gc
import time
import tempfile
import posixpath
import pandas as pd
import geopandas as gpd
from dask.distributed import Client
from osgeo import gdal

# Import constants and utilities
import src.scripts.utilities.constants_and_names as cn
import src.scripts.preprocessing.utilities as uu
from src.scripts.utilities import universal_utilities as uutil

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

# Hardcoded local shapefile directory
LOCAL_SHP_DIR = r"C:\GIS\Data\Global\Plantation\sdpt_by_tiles"


def load_species_reclassification(s3_csv_key):
    local_csv_path = os.path.join(tempfile.gettempdir(), os.path.basename(s3_csv_key))
    uutil.download_file_from_s3(s3_csv_key, local_csv_path, cn.s3_bucket_name)
    df = pd.read_csv(local_csv_path)
    mapping = dict(zip(df["vernacName"].str.strip(), df["rotation_category"].str.strip()))
    return mapping


def classify_plantation(row, species_to_rotation):
    simple_name = str(row.get("simpleName", "")).strip().lower()
    vernac_name = str(row.get("vernacName", "")).strip()
    simple_type = str(row.get("simpleType", "")).strip().lower()

    if simple_type == "tree crops":
        return "oil_palm" if "oil palm" in simple_name else "unknown_tc"
    elif simple_type == "planted forest":
        return species_to_rotation.get(vernac_name, "unknown_rotation")
    return None


def rasterize_to_tif(shp_path, tif_path, minx, miny, maxx, maxy):
    gdal_cmd = [
        'gdal_rasterize', '-a', 'rval',
        '-te', str(minx), str(miny), str(maxx), str(maxy),
        '-tr', str(RASTER_RES), str(RASTER_RES),
        '-a_nodata', str(RASTER_NODATA), '-init', str(RASTER_NODATA),
        '-ot', RASTER_DTYPE, '-co', 'COMPRESS=DEFLATE', '-co', 'TILED=YES',
        shp_path, tif_path
    ]
    subprocess.run(gdal_cmd, check=True)


def rasterize_tile(tile_id, species_to_rotation, run_mode='default'):
    gdal.UseExceptions()
    try:
        logging.info(f"Rasterizing tile {tile_id}")

        s3_raster_key = posixpath.join(cn.datasets['sdpt']['s3_processed'], f"{tile_id}_plantations.tif")
        if run_mode == 'default' and uutil.s3_file_exists(cn.s3_bucket_name, s3_raster_key):
            logging.info(f"Tile {tile_id} already processed. Skipping.")
            return

        local_shp_path = os.path.join(LOCAL_SHP_DIR, f"tile_{tile_id}.shp")
        if not os.path.exists(local_shp_path):
            logging.warning(f"Local shapefile for tile {tile_id} not found.")
            return

        gdf = gpd.read_file(local_shp_path)
        if gdf.empty:
            logging.info(f"No features in {tile_id}. Skipping.")
            return

        gdf["ptype"] = gdf.apply(lambda row: classify_plantation(row, species_to_rotation), axis=1)
        gdf.dropna(subset=["ptype"], inplace=True)
        if gdf.empty:
            logging.info(f"No plantations after classification in {tile_id}.")
            return

        gdf["rval"] = gdf["ptype"].map(final_mapping)
        if gdf["rval"].isnull().all():
            logging.info(f"All null classifications in {tile_id}.")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_shp_path = os.path.join(temp_dir, f"{tile_id}_temp.shp")
            final_raster_path = os.path.join(temp_dir, f"{tile_id}_plantations.tif")

            gdf[['geometry', 'rval']].to_file(temp_shp_path)

            minx, miny, maxx, maxy = gdf.total_bounds
            rasterize_to_tif(temp_shp_path, final_raster_path, minx, miny, maxx, maxy)

            if run_mode == 'default':
                uutil.upload_file_to_s3(final_raster_path, cn.s3_bucket_name, s3_raster_key)

        del gdf

    except subprocess.CalledProcessError as e:
        logging.error(f"GDAL rasterization error for {tile_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error for {tile_id}: {e}")
    finally:
        gc.collect()
        time.sleep(0.5)
        gc.collect()


def main(tile_id=None, run_mode='default'):
    client = Client(n_workers=2, threads_per_worker=1, memory_limit='4GB', processes=True,
                    lifetime="20 minutes", lifetime_stagger="5 minutes")

    try:
        species_to_rotation = load_species_reclassification(SDPT_RECLASS_S3_CSV)

        if tile_id:
            tile_ids = [tile_id]
        else:
            tile_ids = [os.path.splitext(f)[0].replace("tile_", "") for f in os.listdir(LOCAL_SHP_DIR) if f.endswith(".shp")]

        logging.info(f"{len(tile_ids)} tiles to process.")

        futures = [client.submit(rasterize_tile, tid, species_to_rotation, run_mode, pure=False) for tid in tile_ids]

        client.gather(futures)

    finally:
        client.restart()
        client.close()
        logging.info("Dask client closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile_id', type=str)
    parser.add_argument('--run_mode', choices=['default', 'test'], default='default')
    args = parser.parse_args()

    main(tile_id=args.tile_id, run_mode=args.run_mode)