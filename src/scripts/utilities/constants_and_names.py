import posixpath
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import numpy as np
import os

# ---------------------------------------------------
# 1. General Configuration
# ---------------------------------------------------

s3_bucket_name = 'gfw2-data'
s3_region_name = 'us-east-1'

short_bucket_prefix = 'gfw2-data'
full_bucket_prefix = "s3://" + s3_bucket_name

project_dir = 'climate/AFOLU_flux_model/organic_soils'
raw_dir = 'inputs/raw'
processed_dir = 'inputs/processed'

s3_out_dir = 's3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/drainage_model'

local_log_path = "./logs/"
os.makedirs("./logs/", exist_ok=True)
s3_log_path = "climate/AFOLU_flux_model/organic_soils/model_logs/"

tile_id_pattern = r"[0-9]{2}[A-Z][_][0-9]{3}[A-Z]"
small_chunk_pattern = r"__-?\d+_-?\d+_-?\d+_-?\d+__"

local_root = 'C:/GIS/Data/Global'
local_temp_dir = '/tmp'

today_date = datetime.today().strftime('%Y%m%d')

peat_pattern = '_peat_mask_processed.tif'
peat_tiles_prefix = 'climate/AFOLU_flux_model/organic_soils/inputs/raw/soils/GFW_Global_Peatlands'
peat_tiles_prefix_1km = 'climate/AFOLU_flux_model/organic_soils/inputs/processed/peat_mask/1km/'

sample_tile_id = '{tile_id}'
model_version = 0.2

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
    'climate_domain': "climate/carbon_model/inputs_for_carbon_pools/processed/fao_ecozones_bor_tem_tro/20190418/",
    'sdpt': {
        's3_raw': posixpath.join(project_dir, raw_dir, 'plantations', 'sdpt'),
        's3_processed': posixpath.join(project_dir, processed_dir, 'sdpt', today_date),
        's3_processed_small': posixpath.join(project_dir, processed_dir, 'sdpt', '4000_pixels', today_date),
        'local_processed': posixpath.join(local_temp_dir, 'sdpt', today_date),
        'local_processed_small': posixpath.join(local_temp_dir, 'sdpt_chunks_4000', today_date)
    },
}

# ---------------------------------------------------
# 3. General Paths and Constants
# ---------------------------------------------------

ipcc_codes = {
    'forest': 1,
    'cropland': 2,
    'settlement': 3,
    'wetland': 4,
    'grassland': 5,
    'otherland': 6
}

ecozone_codes = {
    'unknown': 0,
    'tropical': 1,
    'boreal': 2,
    'temperate': 3,
}

nutrient_status_codes = {
    'unknown': 0,
    'poor': 1,
    'rich': 2
}

plantation_type_codes = {
    'unknown': 0,
    'long_rotation': 2,
    'short_rotation': 2,
    'oil_palm': 1,
    'sago_palm': 3
}

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
    'climate_domain': "climate_domain"
}

lc_uri = 'climate/AFOLU_flux_model/LULUCF/outputs/IPCC_basic_classes/2020/40000_pixels/20240205'

download_dict = {
    file_patterns['land_cover']:
        f's3://{s3_bucket_name}/{posixpath.join(lc_uri, f"{sample_tile_id}__IPCC_classes_2020.tif")}',

    file_patterns['peat']:
        f's3://{s3_bucket_name}/{posixpath.join(peat_tiles_prefix, f"{sample_tile_id}.tif")}',

    file_patterns['dadap']:
        f's3://{s3_bucket_name}/{posixpath.join(datasets["dadap"]["working_version"], f"dadap_{sample_tile_id}.tif")}',

    file_patterns['engert']:
        f's3://{s3_bucket_name}/{posixpath.join(datasets["engert"]["working_version"], f"engert_{sample_tile_id}.tif")}',

    file_patterns['grip']:
        f's3://{s3_bucket_name}/{posixpath.join(datasets["grip"]["roads"]["working_version"], f"{sample_tile_id}_grip_density.tif")}',

    file_patterns['osm_roads']:
        f's3://{s3_bucket_name}/{posixpath.join(datasets["osm"]["roads"]["working_version"], f"{sample_tile_id}_osm_roads_density.tif")}',

    file_patterns['osm_canals']:
        f's3://{s3_bucket_name}/{posixpath.join(datasets["osm"]["canals"]["working_version"], f"{sample_tile_id}_osm_canals_density.tif")}',

    file_patterns['planted_forest_type_layer']:
        f's3://{s3_bucket_name}/{posixpath.join(datasets["planted_forest_type"]["working_version"], f"{sample_tile_id}_plantation_type_oilpalm_woodfiber_other.tif")}',

    file_patterns['extraction']:
        f's3://{s3_bucket_name}/{posixpath.join(datasets["extraction"]["working_version"], f"{sample_tile_id}_extraction.tif")}',

    file_patterns['climate_domain']:
        f's3://{s3_bucket_name}/{posixpath.join(datasets["climate_domain"], f"{sample_tile_id}_fao_ecozones_bor_tem_tro_processed.tif")}',

    file_patterns['descals_type']:
        f's3://{s3_bucket_name}/{posixpath.join(datasets["descals_oil_palm"]["plant_type"]["working_version"], f"descals_extent_{sample_tile_id}.tif")}',
}

full_raster_dims = 40000

sig_height_loss_threshold = 5
tree_threshold = 5
t_to_Mt = 1e-3
combined_log = "combined_log"

gwp_ch4 = np.float32(28.0)
gwp_n2o = np.float32(265.0)
gwp_co = np.float32(1.9) #need to check this!
c_to_co2 = np.float32(3.67)
n2o_n_to_n2o = np.float32(1.571)
# ---------------------------------------------------
# 4. Additional Constants for Interval Handling
# ---------------------------------------------------

intervals_annual = "annual"
intervals_five_years = "five_years"
intervals_hybrid = "hybrid"

burned_area_final_dir = f"{full_bucket_prefix}/fires/MODIS_burned_area/MCD64A1.061/2_final_outputs__Hansenized/"
burned_area_final_pattern = "burned_area_final"

land_cover_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/outputs/IPCC_basic_classes/"
land_cover_pattern = "land_cover"

# ---------------------------------------------------
# 5. Additional references for drainage script usage
# (Because the code references: cn.<X>_dir and cn.<X>_pattern)
# ---------------------------------------------------

planted_forest_type_dir = f"{full_bucket_prefix}/{project_dir}/{processed_dir}/planted_forest_type/SDPTv2/20230911/"
planted_forest_type_pattern = "plantation_type_oilpalm_woodfiber_other"

extraction_dir = f"{full_bucket_prefix}/{project_dir}/{processed_dir}/extraction/20241021/"
extraction_pattern = "extraction"

nutrient_status_dir = f"{full_bucket_prefix}/{project_dir}/{processed_dir}/extraction/20241021/"
nutrient_status_pattern = "nutrient_status"

descals_dir = f"{full_bucket_prefix}/{project_dir}/{processed_dir}/descals_plantation/extent/20241105/"
descals_pattern = "descals_extent"

dadap_dir = f"{full_bucket_prefix}/{project_dir}/{processed_dir}/dadap_density/30m/20240925/"
dadap_pattern = "dadap"

osm_roads_dir = f"{full_bucket_prefix}/{project_dir}/{processed_dir}/osm_roads_density/40000_pixels/20240925/"
osm_roads_pattern = "osm_roads_density"

osm_canals_dir = f"{full_bucket_prefix}/{project_dir}/{processed_dir}/osm_canals_density/40000_pixels/20240822/"
osm_canals_pattern = "osm_canals_density"

engert_dir = f"{full_bucket_prefix}/{project_dir}/{processed_dir}/engert_density/30m/20240925/"
engert_pattern = "engert"

grip_dir = f"{full_bucket_prefix}/{project_dir}/{processed_dir}/grip_density/40000_pixels/20240925/"
grip_pattern = "grip_density"

climate_domain_dir = f"{full_bucket_prefix}/climate/carbon_model/inputs_for_carbon_pools/processed/fao_ecozones_bor_tem_tro/20190418/"
climate_domain_pattern = "fao_ecozones_bor_tem_tro_processed"
