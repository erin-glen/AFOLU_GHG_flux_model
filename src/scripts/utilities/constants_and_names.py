# constants_and_names.py

import posixpath
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import numpy as np

# ---------------------------------------------------
# 1. General Configuration
# ---------------------------------------------------

# S3 Configuration
s3_bucket_name = 'gfw2-data'
s3_region_name = 'us-east-1'

short_bucket_prefix = 'gfw2-data'
full_bucket_prefix = "s3://" + s3_bucket_name

# Project Directories
project_dir = 'climate/AFOLU_flux_model/organic_soils'
raw_dir = 'inputs/raw'
processed_dir = 'inputs/processed'

# Define s3_out_dir for outputs
s3_out_dir = 's3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/drainage_model'

local_log_path = '/tmp'
s3_log_path = "climate/AFOLU_flux_model/organic_soils/model_logs/"

tile_id_pattern = r"[0-9]{2}[A-Z][_][0-9]{3}[A-Z]"

# Local Directories
local_root = 'C:/GIS/Data/Global'  # Adjust as needed for your local environment
local_temp_dir = 'C:/tmp'  # Adjust based on your environment

# Date Configuration
today_date = datetime.today().strftime('%Y%m%d')

# File Patterns
peat_pattern = '_peat_mask_processed.tif'
peat_tiles_prefix = 'climate/AFOLU_flux_model/organic_soils/inputs/raw/soils/GFW_Global_Peatlands'
peat_tiles_prefix_1km = 'climate/AFOLU_flux_model/organic_soils/inputs/processed/peat_mask/1km/'

# Sample Tile ID Placeholder
sample_tile_id = '{tile_id}'

# model version for log
model_version = 0.2

# Local path for chunk stats
chunk_stats_path = posixpath.join(local_temp_dir, 'chunk_stats/')

# ---------------------------------------------------
# 2. Dataset Configurations
# ---------------------------------------------------

datasets = {
    'osm': {
        'roads': {
            's3_raw': posixpath.join(project_dir, raw_dir, 'roads', 'osm_roads', 'roads_by_tile'),
            's3_processed_base': posixpath.join(project_dir, processed_dir, 'osm_roads_density'),
            's3_processed_small': posixpath.join(project_dir, processed_dir, 'osm_roads_density', '4000_pixels',
                                                 today_date),
            's3_processed': posixpath.join(project_dir, processed_dir, 'osm_roads_density', today_date),
            'local_processed': posixpath.join(local_temp_dir, 'osm_roads_density', today_date),
            'working_version': posixpath.join(project_dir, processed_dir, 'osm_roads_density', '40000_pixels',
                                              '20240925')
        },
        'canals': {
            's3_raw': posixpath.join(project_dir, raw_dir, 'roads', 'osm_roads', 'canals_by_tile'),
            's3_processed_base': posixpath.join(project_dir, processed_dir, 'osm_canals_density'),
            's3_processed_small': posixpath.join(project_dir, processed_dir, 'osm_canals_density', '4000_pixels',
                                                 today_date),
            's3_processed': posixpath.join(project_dir, processed_dir, 'osm_canals_density', today_date),
            'local_processed': posixpath.join(local_temp_dir, 'osm_canals_density', today_date),
            'working_version': posixpath.join(project_dir, processed_dir, 'osm_canals_density', '40000_pixels',
                                              '20240822')
        }
    },
    'grip': {
        'roads': {
            's3_raw': posixpath.join(project_dir, raw_dir, 'roads', 'grip_roads', 'roads_by_tile'),
            's3_processed_base': posixpath.join(project_dir, processed_dir, 'grip_density'),
            's3_processed_small': posixpath.join(project_dir, processed_dir, 'grip_density', '4000_pixels', today_date),
            's3_processed': posixpath.join(project_dir, processed_dir, 'grip_density', today_date),
            'local_processed': posixpath.join(local_temp_dir, 'grip_density', today_date),
            'working_version': posixpath.join(project_dir, processed_dir, 'grip_density', '40000_pixels', '20240925')
        }
    },
    'engert': {
        's3_raw': posixpath.join(project_dir, raw_dir, 'roads', 'engert_roads',
                                 'engert_asiapac_ghrdens_1km_resample_30m.tif'),
        's3_processed_base': posixpath.join(project_dir, processed_dir, 'engert_density', '30m'),
        's3_processed': posixpath.join(project_dir, processed_dir, 'engert_density', '30m', today_date),
        'local_processed': posixpath.join(local_temp_dir, 'engert_density', today_date),
        'working_version': posixpath.join(project_dir, processed_dir, 'engert_density', '30m', '20240925')
    },
    'dadap': {
        's3_raw': posixpath.join(project_dir, raw_dir, 'canals', 'Dadap_SEA_Drainage', 'canal_length_data',
                                 'canal_length_1km_resample_30m.tif'),
        's3_processed_base': posixpath.join(project_dir, processed_dir, 'dadap_density', '30m'),
        's3_processed': posixpath.join(project_dir, processed_dir, 'dadap_density', '30m', today_date),
        'local_processed': posixpath.join(local_temp_dir, 'dadap_density', today_date),
        'working_version': posixpath.join(project_dir, processed_dir, 'dadap_density', '30m', '20240925')
    },
    'planted_forest_type': {
        's3_processed_base': posixpath.join('climate', 'carbon_model', 'other_emissions_inputs', 'plantation_type',
                                            'SDPTv2', '20230911'),
        'working_version': posixpath.join('climate', 'carbon_model', 'other_emissions_inputs', 'plantation_type',
                                          'SDPTv2', '20230911')
    },
    'descals_oil_palm': {
        'plant_year': {
            's3_processed_base': posixpath.join(project_dir, processed_dir, 'descals_plantation', 'year'),
            's3_processed': posixpath.join(project_dir, processed_dir, 'descals_plantation', 'year', today_date),
            'local_processed': posixpath.join(local_temp_dir, 'descals_plantation', 'year', today_date),
            'working_version': posixpath.join(project_dir, processed_dir, 'descals_plantation', 'year', '20241105')
        },
        'plant_type': {
            's3_processed_base': posixpath.join(project_dir, processed_dir, 'descals_plantation', 'extent'),
            's3_processed': posixpath.join(project_dir, processed_dir, 'descals_plantation', 'extent', today_date),
            'local_processed': posixpath.join(local_temp_dir, 'descals_plantation', 'extent', today_date),
            'working_version': posixpath.join(project_dir, processed_dir, 'descals_plantation', 'extent', '20241105')
        }
    },
    'extraction': {
        'finland': {
            's3_raw': posixpath.join(project_dir, raw_dir, 'extraction', 'Finland', 'Finland_turvetuotantoalueet',
                                     'turvetuotantoalueet_jalkikaytto'),
        },
        'ireland': {
            's3_raw': posixpath.join(project_dir, raw_dir, 'extraction', 'Ireland', 'Ireland_Habibetal',
                                     'RF_S2_LU_5_11_23.tif'),
        },
        'russia': {
            's3_raw': [
                posixpath.join(project_dir, raw_dir, 'extraction', 'Russia', 'allocated_without_licenses',
                               'allocated_mineral_reserve'),
                posixpath.join(project_dir, raw_dir, 'extraction', 'Russia', 'allocated_with_licenses',
                               'peat_extraction_dates')
            ],
        },
        's3_processed_base': posixpath.join(project_dir, processed_dir, 'extraction'),
        's3_processed': posixpath.join(project_dir, processed_dir, 'extraction', today_date),
        'local_processed': posixpath.join(local_temp_dir, 'extraction', today_date),
        'working_version': posixpath.join(project_dir, processed_dir, 'extraction', '20241021')
    },
    'continent_ecozone': "climate/carbon_model/fao_ecozones/ecozone_continent/20190116/processed/"

    # Add other datasets as needed
}

# ---------------------------------------------------
# 3. General Paths and Constants
# ---------------------------------------------------


# IPCC Codes
ipcc_codes = {
    'forest': 1,
    'cropland': 2,
    'settlement': 3,
    'wetland': 4,
    'grassland': 5,
    'otherland': 6
}

# Ecozone Codes
ecozone_codes = {
    'unknown': 0,
    'boreal': 1,
    'temperate': 2,
    'tropical': 3
}

# Nutrient Status Codes
nutrient_status_codes = {
    'unknown': 0,
    'poor': 1,
    'rich': 2
}


"""
Some info on SDPT data from David:
They're for SDPTv2, from 20240911, I think. For planted forest type, 
1-oil palm, 2-woodfiber, 3-other. For planted_forest_tree_crop, I believe 1-planted forest, 2-tree crop.
"""

#TODO this needs to be updated!!!
# Plantation Types Codes
plantation_type_codes = {
    'unknown': 0,
    'long_rotation': 2,
    'short_rotation': 2,
    'oil_palm': 1,
    'sago_palm': 3
}

# File Name Patterns
file_patterns = {
    'land_cover': "IPCC_basic_classes_2020",
    'vegetation_height': "vegetation_height",
    'planted_forest_type_layer': "planted_forest_type",
    'planted_forest_tree_crop_layer': "planted_forest_tree_crop",
    'peat': "peat",
    'dadap': "dadap",
    'engert': "engert",
    'grip': "grip",
    'osm_roads': "osm_roads",
    'osm_canals': "osm_canals",
    'extraction': "extraction",
    'descals_type': "descals_type",
    'descals_year': "descals_year",
    'continent_ecozone': "continent_ecozone"
}

# ---------------------------------------------------
# 4. Download Dictionary
# ---------------------------------------------------
# Prepare download dictionary using 'working_version' paths

lc_uri = 'climate/AFOLU_flux_model/LULUCF/outputs/IPCC_basic_classes/2020/40000_pixels/20240205'

download_dict = {
    file_patterns[
        'land_cover']: f's3://{s3_bucket_name}/{posixpath.join(lc_uri, f"{sample_tile_id}__IPCC_classes_2020.tif")}',
    file_patterns['peat']: f's3://{s3_bucket_name}/{posixpath.join(peat_tiles_prefix, f"{sample_tile_id}.tif")}',
    file_patterns[
        'dadap']: f's3://{s3_bucket_name}/{posixpath.join(datasets["dadap"]["working_version"], f"dadap_{sample_tile_id}.tif")}',
    file_patterns[
        'engert']: f's3://{s3_bucket_name}/{posixpath.join(datasets["engert"]["working_version"], f"engert_{sample_tile_id}.tif")}',
    file_patterns[
        'grip']: f's3://{s3_bucket_name}/{posixpath.join(datasets["grip"]["roads"]["working_version"], f"{sample_tile_id}_grip_density.tif")}',
    file_patterns[
        'osm_roads']: f's3://{s3_bucket_name}/{posixpath.join(datasets["osm"]["roads"]["working_version"], f"{sample_tile_id}_osm_roads_density.tif")}',
    file_patterns[
        'osm_canals']: f's3://{s3_bucket_name}/{posixpath.join(datasets["osm"]["canals"]["working_version"], f"{sample_tile_id}_osm_canals_density.tif")}',
    file_patterns[
        'planted_forest_type_layer']: f's3://{s3_bucket_name}/{posixpath.join(datasets["planted_forest_type"]["working_version"], f"{sample_tile_id}_plantation_type_oilpalm_woodfiber_other.tif")}',
    file_patterns[
        'extraction']: f's3://{s3_bucket_name}/{posixpath.join(datasets["extraction"]["working_version"], f"{sample_tile_id}_extraction.tif")}',
    file_patterns[
        'continent_ecozone']: f's3://{s3_bucket_name}/{posixpath.join(datasets["continent_ecozone"], f"{sample_tile_id}_fao_ecozones_continents_processed.tif")}',
    file_patterns[
        'descals_type']: f's3://{s3_bucket_name}/{posixpath.join(datasets["descals_oil_palm"]["plant_type"]["working_version"], f"descals_extent_{sample_tile_id}.tif")}',

}


### Miscellaneous

full_raster_dims = 40000  # Size of a 10x10 deg raster in pixels

# Threshold for height loss to be counted as tree loss (meters)
sig_height_loss_threshold = 5

# Height minimum for trees (meters)
tree_threshold = 5

# Converts tonnes to megatonnes
t_to_Mt = 10 ** -3

combined_log = "combined_log"

# Constants for GWPs
gwp_ch4 = np.float32(28.0)  # For example
gwp_n2o = np.float32(265.0)  # For example

# constants_and_names.py
c_to_co2 = np.float32(3.67)      # Conversion factor from C to CO₂
n2o_n_to_n2o = np.float32(1.571) # Conversion factor from N₂O-N to N₂O
