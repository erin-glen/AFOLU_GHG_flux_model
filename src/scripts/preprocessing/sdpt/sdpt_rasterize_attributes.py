import os
import argparse
import sys
from dask.distributed import Client
import geopandas as gpd
import subprocess
import logging
import tempfile

from src.scripts.utilities import universal_utilities as uu

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Plantation reclassification dictionaries
plantation_reclass = {
    "Oil palm": "oil_palm",
    "Oil palm mix": "oil_palm",
    "Wood fiber or timber": "short_rotation",
    "Wood fiber or timber mix": "short_rotation",
    "Rubber": "long_rotation",
    "Rubber mix": "long_rotation",
    "Fruit": "long_rotation",
    "Fruit mix": "long_rotation",
    "Other": "short_rotation",
    "Other mix": "short_rotation",
    "Unknown": "short_rotation",
    "Unknown mix": "short_rotation",
}

plantation_mapping = {'oil_palm': 1, 'short_rotation': 2, 'long_rotation': 3}

RASTER_RES = 0.00025
RASTER_NODATA = 0
RASTER_DTYPE = 'Byte'

# Hardcoded paths for testing
path_dict = {
    'shapefile_dir': r"C:\GIS\Data\Global\Plantation\sdpt_by_tiles",
    'output_dir': r"C:\GIS\Data\Global\Plantation\sdpt_rasters",
    's3_output_prefix': 's3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/sdpt',
}

def rasterize_tile(tile_id, paths_dict, run_mode='default'):
    try:
        logging.info(f"Rasterizing tile {tile_id}")

        shp_path = os.path.join(paths_dict['shapefile_dir'], f"tile_{tile_id}.shp")
        output_raster = os.path.join(paths_dict['output_dir'], f"{tile_id}_plantations.tif")

        # Check if raster already exists locally
        if os.path.exists(output_raster):
            logging.info(f"Raster already exists locally: {output_raster}")
        else:
            if not os.path.exists(shp_path):
                logging.warning(f"Shapefile {shp_path} missing. Skipping.")
                return

            # Read and preprocess shapefile
            gdf = gpd.read_file(shp_path)
            gdf['plantation_type'] = gdf['simpleName'].map(plantation_reclass)
            gdf = gdf.dropna(subset=['plantation_type'])

            if gdf.empty:
                logging.info(f"No plantations in tile {tile_id}.")
                return

            gdf['raster_val'] = gdf['plantation_type'].map(plantation_mapping)

            # Save temp shapefile
            temp_shp = os.path.join(tempfile.gettempdir(), f"{tile_id}_temp.shp")
            gdf.to_file(temp_shp)

            minx, miny, maxx, maxy = gdf.total_bounds

            # GDAL rasterize command
            gdal_cmd = [
                'gdal_rasterize',
                '-a', 'raster_val',
                '-te', str(minx), str(miny), str(maxx), str(maxy),
                '-tr', str(RASTER_RES), str(RASTER_RES),
                '-a_nodata', '0',  # Set NoData explicitly to 0
                '-init', '0',      # Initialize raster with 0 values
                '-ot', RASTER_DTYPE,
                '-co', 'COMPRESS=DEFLATE',
                '-co', 'TILED=YES',
                temp_shp,
                output_raster
            ]

            logging.info(f"Running: {' '.join(gdal_cmd)}")
            subprocess.run(gdal_cmd, check=True)

            # Clean up temporary shapefile components
            for ext in ['shp', 'shx', 'dbf', 'prj', 'cpg']:
                os.remove(os.path.join(tempfile.gettempdir(), f"{tile_id}_temp.{ext}"))

        # Conditional upload based on run_mode
        if run_mode == 'default':
            s3_out = f"{paths_dict['s3_output_prefix']}/{tile_id}_plantations.tif"
            logging.info(f"Uploading {output_raster} to {s3_out}")
            uu.upload_s3_file(s3_out, output_raster)

            # Remove local file after successful upload
            os.remove(output_raster)
            logging.info(f"Local raster file {output_raster} removed after upload.")
        else:
            logging.info(f"Test mode enabled, raster {output_raster} saved locally. No upload performed.")

    except subprocess.CalledProcessError as e:
        logging.error(f"GDAL error for tile {tile_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error for tile {tile_id}: {e}")




def main(tile_id=None, run_mode='default', client_type='local'):
    if client_type == 'coiled':
        cluster, client = uu.connect_to_Coiled_cluster('plantations_cluster', run_local=False)
    else:
        client = Client()

    os.makedirs(path_dict['output_dir'], exist_ok=True)

    tile_ids = [tile_id] if tile_id else [
        fname.split("_")[1].replace(".shp", "")
        for fname in os.listdir(path_dict['shapefile_dir']) if fname.endswith(".shp")
    ]

    logging.info(f"Starting rasterization for tiles: {tile_ids}")

    futures = [
        client.submit(rasterize_tile, tid, path_dict)
        for tid in tile_ids
    ]

    client.gather(futures)

    client.close()
    if client_type == 'coiled':
        cluster.close()

    logging.info("Rasterization completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rasterize plantation shapefiles.')
    parser.add_argument('--tile_id', type=str, help='Tile ID to process')
    parser.add_argument('--run_mode', type=str, choices=['default', 'test'], default='default',
                        help='Run mode (default or test)')
    parser.add_argument('--client', type=str, choices=['local', 'coiled'], default='local',
                        help='Dask client type to use (local or coiled)')
    args = parser.parse_args()

    if not any(sys.argv[1:]):
        # Defaults for IDE (e.g., PyCharm) testing
        logging.info("Running with default IDE values.")
        main(tile_id='10N_110E', run_mode='test', client_type='local')
    else:
        main(tile_id=args.tile_id, run_mode=args.run_mode, client_type=args.client)

"""
Example usage:

Local IDE test run:
(No arguments, automatically uses defaults)

Command line example for specific tile with local Dask:
python -m src.preprocessing.sdpt.sdpt_rasterize_attributes --tile_id 10N_110E --run_mode default --client local

Command line example for Coiled cluster:
python -m src.preprocessing.sdpt.sdpt_rasterize_attributes --tile_id 10N_110E --run_mode default --client coiled
"""
