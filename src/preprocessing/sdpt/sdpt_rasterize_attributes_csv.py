import os
import argparse
import sys
import subprocess
import logging
import tempfile

import pandas as pd
import geopandas as gpd
from dask.distributed import Client

# Import your universal utilities if needed (for S3 upload)
from src.scripts.utilities import universal_utilities as uu

# ---------------------------------------------------------------------------
# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ---------------------------------------------------------------------------
# Reclassification mapping: final integer values
final_mapping = {
    'oil_palm': 1,
    'unknown_tc': 2,
    'short_rotation': 3,
    'long_rotation': 4,
    'unknown_rotation': 5
}

# Hardcoded paths for testing
path_dict = {
    'shapefile_dir': r"C:\GIS\Data\Global\Plantation\sdpt_by_tiles",
    'output_dir': r"C:\GIS\Data\Global\Plantation\sdpt_rasters",
    's3_output_prefix': 's3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/sdpt',
}

# Rasterization settings
RASTER_RES = 0.00025
RASTER_NODATA = 0
RASTER_DTYPE = 'Byte'

# Path to the reclassification CSV for planted forest species.
# This CSV should have at least two columns:
#   - simpleName: the species (or commodity) name
#   - rotation: one of "short_rotation", "long_rotation", or "unknown_rotation"
RECLASS_CSV_PATH = r"C:\GIS\Data\Global\Plantation\updated_classified_planted_forest_species.csv"

# ---------------------------------------------------------------------------
def load_species_reclassification(reclass_csv_path):
    """
    Load the reclassification CSV and return a dictionary mapping simpleName to rotation.
    If the file does not exist, return an empty dict.
    """
    if os.path.exists(reclass_csv_path):
        reclass_df = pd.read_csv(reclass_csv_path)
        # Ensure the keys and values are stripped and lowercased if needed.
        # (Assuming keys in reclassification CSV match simpleName exactly.)
        mapping = dict(zip(reclass_df["vernacName"].str.strip(), reclass_df["rotation_category"].str.strip()))
        logging.info(f"Loaded reclassification mapping for {len(mapping)} species from {reclass_csv_path}")
        return mapping
    else:
        logging.warning(f"Reclassification CSV not found: {reclass_csv_path}. "
                        "Planted forest rows will be classified as 'unknown_rotation'.")
        return {}

# ---------------------------------------------------------------------------
def classify_plantation(row, species_to_rotation):
    """
    For each row, assign a classification string based on:
      - if simpleType equals "Tree crops" (case-insensitive):
            if simpleName contains "oil palm" (case-insensitive) then "oil_palm"
            else "unkown_tc"
      - if simpleType equals "Planted forest" (case-insensitive):
            use the species_to_rotation mapping from the CSV; if not found, return "unknown_rotation"
      - Otherwise, return None.
    """
    simple_name = str(row["simpleName"]).strip()
    vernac_name = str(row["vernacName"]).strip()
    simple_type = str(row["simpleType"]).strip().lower()

    if simple_type == "tree crops":
        if "oil palm" in simple_name.lower():
            return "oil_palm"
        else:
            return "unkown_tc"
    elif simple_type == "planted forest":
        # Look up the species name in the reclassification mapping.
        # If not found, default to "unknown_rotation"
        return species_to_rotation.get(vernac_name, "unknown_rotation")
    else:
        return None

# ---------------------------------------------------------------------------
def rasterize_tile(tile_id, paths_dict, species_to_rotation, run_mode='default'):
    try:
        logging.info(f"Rasterizing tile {tile_id}")

        shp_path = os.path.join(paths_dict['shapefile_dir'], f"tile_{tile_id}.shp")
        output_raster = os.path.join(paths_dict['output_dir'], f"{tile_id}_plantations.tif")

        # Check if raster already exists locally
        if os.path.exists(output_raster):
            logging.info(f"Raster already exists locally: {output_raster}")
        else:
            if not os.path.exists(shp_path):
                logging.warning(f"Shapefile {shp_path} missing. Skipping tile {tile_id}.")
                return

            # Read shapefile
            gdf = gpd.read_file(shp_path)

            # Classify each record using our new logic.
            # This will assign a string classification to a new column 'plantation_type'.
            gdf["plantation_type"] = gdf.apply(lambda row: classify_plantation(row, species_to_rotation), axis=1)
            # Drop rows that did not meet either "Tree crops" or "Planted forest"
            gdf = gdf.dropna(subset=["plantation_type"])

            if gdf.empty:
                logging.info(f"No plantation records found in tile {tile_id}.")
                return

            # Map classification strings to final integer values
            gdf["raster_val"] = gdf["plantation_type"].map(final_mapping)
            if gdf["raster_val"].isnull().all():
                logging.info(f"After mapping, no valid values for tile {tile_id}.")
                return

            # Save temporary shapefile for GDAL rasterize
            temp_shp = os.path.join(tempfile.gettempdir(), f"{tile_id}_temp.shp")
            gdf.to_file(temp_shp)

            minx, miny, maxx, maxy = gdf.total_bounds

            # Construct GDAL rasterize command
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
                temp_shp,
                output_raster
            ]

            logging.info(f"Running GDAL command: {' '.join(gdal_cmd)}")
            subprocess.run(gdal_cmd, check=True)

            # Clean up temporary shapefile files
            for ext in ['shp', 'shx', 'dbf', 'prj', 'cpg']:
                temp_file = os.path.join(tempfile.gettempdir(), f"{tile_id}_temp.{ext}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        # Upload to S3 if run_mode is 'default'
        if run_mode == 'default':
            s3_out = f"{paths_dict['s3_output_prefix']}/{tile_id}_plantations.tif"
            logging.info(f"Uploading {output_raster} to {s3_out}")
            uu.upload_s3_file(s3_out, output_raster)
            os.remove(output_raster)
            logging.info(f"Local raster file {output_raster} removed after upload.")
        else:
            logging.info(f"Test mode enabled, raster {output_raster} saved locally. No upload performed.")

    except subprocess.CalledProcessError as e:
        logging.error(f"GDAL error for tile {tile_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error for tile {tile_id}: {e}")

# ---------------------------------------------------------------------------
def main(tile_id=None, run_mode='default', client_type='local'):
    # Set up Dask client
    if client_type == 'coiled':
        cluster, client = uu.connect_to_Coiled_cluster('plantations_cluster', run_local=False)
    else:
        client = Client()

    # Ensure output directory exists
    os.makedirs(path_dict['output_dir'], exist_ok=True)

    # Load the reclassification CSV for planted forest species
    species_to_rotation = load_species_reclassification(RECLASS_CSV_PATH)

    # Determine list of tile_ids to process
    if tile_id:
        tile_ids = [tile_id]
    else:
        tile_ids = [
            fname.split("_")[1].replace(".shp", "")
            for fname in os.listdir(path_dict['shapefile_dir']) if fname.endswith(".shp")
        ]

    logging.info(f"Starting rasterization for tiles: {tile_ids}")

    futures = [client.submit(rasterize_tile, tid, path_dict, species_to_rotation, run_mode)
               for tid in tile_ids]

    client.gather(futures)
    client.close()
    if client_type == 'coiled':
        cluster.close()

    logging.info("Rasterization completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rasterize plantation shapefiles with reclassification.')
    parser.add_argument('--tile_id', type=str, help='Tile ID to process')
    parser.add_argument('--run_mode', type=str, choices=['default', 'test'], default='default',
                        help='Run mode (default: uploads to S3, test: local only)')
    parser.add_argument('--client', type=str, choices=['local', 'coiled'], default='local',
                        help='Dask client type to use (local or coiled)')
    args = parser.parse_args()

    if not any(sys.argv[1:]):
        logging.info("Running with default IDE values.")
        main(tile_id='00N_110E', run_mode='test', client_type='local')
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