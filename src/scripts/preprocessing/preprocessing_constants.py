import posixpath as pp
from pathlib import Path

from src.scripts.utilities.constants_and_names import (
    s3_bucket_name,
    s3_region_name,
    full_bucket_prefix,
    project_dir,
    raw_dir,
    processed_dir,
    local_root,
    local_temp_dir,
    today_date,
    pixel_area_dir,
    pixel_area_pattern,
    m2_to_ha,
    ipcc_codes,
    t_to_Mt,
    sample_tile_id,
    full_raster_dims,
)

# Peat mask related paths
peat_pattern = "_peat_mask_processed.tif"
peat_tiles_prefix_1km = pp.join(processed_dir, "peat_mask", "GFW", "1km") + "/"
peat_tiles_prefix_1km_3395 = pp.join(processed_dir, "peat_mask", "1km_3395") + "/"
peat_tiles_prefix = "climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/"

# Global tile index (no extension)
index_shapefile_prefix = pp.join(raw_dir, "index/Global_Peatlands")

# 30 m resolution in degrees (approx. 1 km)
resolution = 0.000025

# ---------------------------------------------------------------------
# 2. Dataset configurations
# ---------------------------------------------------------------------
datasets = {
    'osm': {
        'roads': {
            's3_raw': pp.join(raw_dir, 'roads', 'osm_roads', 'roads_by_tile'),
            's3_processed_base': pp.join(processed_dir, 'osm_roads_density'),
            's3_processed_small': pp.join(processed_dir, 'osm_roads_density',
                                          '4000_pixels', today_date),
            's3_processed': pp.join(processed_dir, 'osm_roads_density', today_date),
            'local_processed': pp.join(local_temp_dir, 'osm_roads_density', today_date),
            's3_projected': pp.join(raw_dir, 'roads', 'osm_roads', 'roads_by_tile_3395')
        },
        'canals': {
            's3_raw': pp.join(raw_dir, 'roads', 'osm_roads', 'canals_by_tile'),
            's3_processed_base': pp.join(processed_dir, 'osm_canals_density'),
            's3_processed_small': pp.join(processed_dir, 'osm_canals_density',
                                          '4000_pixels', today_date),
            's3_processed': pp.join(processed_dir, 'osm_canals_density', today_date),
            'local_processed': pp.join(local_temp_dir, 'osm_canals_density', today_date),
            's3_projected': pp.join(raw_dir, 'roads', 'osm_roads', 'canals_by_tile_3395')
        }
    },
    "glclu_composite": {
        "s3_raw": pp.join(
            "climate",
            "AFOLU_flux_model",
            "LULUCF",
            "landcover",
            "composite",
            "{interval}",
            "v2",
            "raw",
            "{year}",
            "{tile_id}.tif",
        )
    },
    'grip': {
        'roads': {
            's3_raw': pp.join(raw_dir, 'roads', 'grip_roads', 'roads_by_tile'),
            's3_processed_base': pp.join(processed_dir, 'grip_density'),
            's3_processed_small': pp.join(processed_dir, 'grip_density','4000_pixels', today_date),
            's3_processed': pp.join(processed_dir, 'grip_density', today_date),
            'local_processed': pp.join(local_temp_dir, 'grip_density', today_date),
            's3_projected': pp.join(raw_dir, 'roads', 'grip_roads', 'roads_by_tile_3395')
        }
    },
    'engert': {
        's3_raw': pp.join(raw_dir, 'roads', 'engert_roads',
                          'engert_asiapac_ghrdens_1km_resample_30m.tif'),
        's3_processed_base': pp.join(processed_dir, 'engert_density', '30m'),
        's3_processed': pp.join(processed_dir, 'engert_density', '30m', today_date),
        'local_processed': pp.join(local_temp_dir, 'engert_density', today_date),
        'working_version': pp.join(processed_dir, 'engert_density', '30m', '20240925')
    },
    'dadap': {
        's3_raw': pp.join(raw_dir, 'canals', 'Dadap_SEA_Drainage',
                          'canal_length_data', 'canal_length_1km_resample_30m.tif'),
        's3_processed_base': pp.join(processed_dir, 'dadap_density', '30m'),
        's3_processed': pp.join(processed_dir, 'dadap_density', '30m', today_date),
        'local_processed': pp.join(local_temp_dir, 'dadap_density', today_date),
        'working_version': pp.join(processed_dir, 'dadap_density', '30m', '20240925')

    },
    'planted_forest_type': {
        's3_processed_base': pp.join('climate', 'carbon_model', 'other_emissions_inputs',
                                     'plantation_type', 'SDPTv2', '20230911')
    },
    'sdpt': {
        's3_raw': pp.join('plantations', 'sdpt_v3','sdpt_v3_vector_tiles','tiles_10d'),
        's3_processed_base': pp.join(processed_dir, 'sdpt'),
        's3_processed_small': pp.join(processed_dir, 'sdpt', 'chunks', today_date),
        's3_processed': pp.join(processed_dir, 'sdpt', today_date),
        'local_processed': pp.join(local_temp_dir, 'sdpt', today_date)
    },
    'extraction': {
        'finland': {
            's3_raw': f'{raw_dir}/extracion/Finland/Finland_turvetuotantoalueet/'
                      f'turvetuotantoalueet_jalkikaytto',
            's3_processed_base': f'{processed_dir}/extraction/',
            's3_processed': f'{processed_dir}/extraction/{today_date}/',
            'local_processed': f'{local_temp_dir}/extraction/finland/{today_date}/'
        },
        'ireland': {
            's3_raw': f'{raw_dir}/extraction/Ireland/Ireland_Habibetal/RF_S2_LU_5_11_23.tif',
            's3_processed_base': f'{processed_dir}/extraction/',
            's3_processed': f'{processed_dir}/extraction/{today_date}/',
            'local_processed': f'{local_temp_dir}/extraction/ireland/{today_date}/'
        },
        'russia': {
            's3_raw': [
                f'{raw_dir}/extraction/Russia/allocated_without_licenses/'
                f'allocated_mineral_reserve',
                f'{raw_dir}/extraction/Russia/allocated_with_licenses/'
                f'peat_extraction_dates'
            ],
            's3_processed_base': f'{processed_dir}/extraction/',
            's3_processed': f'{processed_dir}/extraction/{today_date}/',
            'local_processed': f'{local_temp_dir}/extraction/russia/{today_date}/'
        }
    },
    'descals_oil_palm': {
        'plantation_year': {
            's3_raw': pp.join(raw_dir, 'plantations', 'plantation_year'),
            's3_processed_base': pp.join(processed_dir, 'descals_plantation', 'year'),
            's3_processed': pp.join(processed_dir, 'descals_plantation',
                                    'year', today_date),
            'local_processed': pp.join(local_temp_dir, 'descals_plantation', 'year', today_date),
            'working_version': pp.join(processed_dir, 'descals_plantation',
                                       'year', '20240823')
        },
        'plantation_type': {
            's3_raw': pp.join(raw_dir, 'plantations', 'plantation_extent'),
            's3_processed_base': pp.join(processed_dir, 'descals_plantation', 'extent'),
            's3_processed': pp.join(processed_dir, 'descals_plantation',
                                    'extent', today_date),
            'local_processed': pp.join(local_temp_dir, 'descals_plantation', 'extent', today_date),
            'working_version': pp.join(processed_dir, 'descals_plantation',
                                       'extent', '20240823')
        }
    },
    'land_cover_ipcc': {
        's3_processed_base': pp.join(processed_dir, 'land_cover_ipcc'),
        's3_processed': pp.join(processed_dir, 'land_cover_ipcc', today_date),
        'local_processed': pp.join(local_temp_dir, 'land_cover_ipcc', today_date),
    },
    'peat': {
        'gfw': {
            's3_processed': 'climate/AFOLU_flux_model/organic_soils/inputs/raw/soils/GFW_Global_Peatlands/'
        },
        'gpd': {
            'input_type': 'raster',
            's3_raw': pp.join(raw_dir, 'soils', 'GPD', 'peatGPA22WGS_2cl.tif'),
            's3_processed': pp.join(processed_dir, 'peat_mask', 'GPD', 'tiles') + '/',
            'local_processed': pp.join(local_temp_dir, 'peat', 'gpd', 'tiles') + '/'
        },
        'peatmap': {
            'input_type': 'vector',
            's3_raw': pp.join(raw_dir, 'soils', 'PEATMAP'),
            'file_pattern': '*.shp',
            's3_processed': pp.join(processed_dir, 'peat_mask',
                                    'PEATMAP', 'tiles') + '/',
            'local_processed': pp.join(local_temp_dir, 'peat', 'peatmap', 'tiles') + '/'
        },
        'peatml': {
            'input_type': 'raster',
            'threshold': 50,
            's3_raw': pp.join(raw_dir, 'soils',
                              'PEATML', 'Peat-ML_global_peatland_extent.tif'),
            's3_processed': pp.join(processed_dir, 'peat_mask',
                                    'PEATML', 'tiles') + '/',
            'local_processed': pp.join(local_temp_dir, 'peat', 'peatml', 'tiles') + '/'
        },
        'ogh': {
            'input_type': 'raster',
            's3_raw': pp.join(raw_dir, 'soils', 'OGH', 'organic_soils_extent.tif'),
            's3_processed': pp.join(processed_dir, 'peat_mask', 'OGH', 'tiles') + '/',
            'local_processed': pp.join(local_temp_dir, 'peat', 'ogh', 'tiles') + '/',
            'threshold': 23
        },
        'ogh_unthresholded': {
            'input_type': 'raster',
            's3_raw': pp.join(raw_dir, 'soils', 'OGH', 'organic_soils_extent.tif'),
            's3_processed': pp.join(processed_dir, 'peat_mask', 'OGH', 'tiles_unthresholded') + '/',
            'local_processed': pp.join(local_temp_dir, 'peat', 'ogh_unthresholded', 'tiles') + '/'
        },
        'union_mask': {
            '30m': pp.join(
                processed_dir, 'peat_mask', 'union', '30m', 'tiles'
            )
            + '/',
            '1km': pp.join(
                processed_dir, 'peat_mask', 'union', '1km', 'tiles'
            )
            + '/',
            '1km_3395': pp.join(processed_dir, 'peat_mask','union','1km_3395',
            )
            + '/',
        }
    }
}
# ---------------------------------------------------------------------
# 3. General paths / constants
# ---------------------------------------------------------------------
lc_uri = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/inputs/LC"

file_patterns = {
    'land_cover': "IPCC_basic_classes",
    'vegetation_height': "vegetation_height",
    'planted_forest_type_layer': "planted_forest_type",
    'planted_forest_tree_crop_layer': "planted_forest_tree_crop",
    'peat': "peat",
    'peat_gpd': "peat_gpd",
    'peat_peatmap': "peat_peatmap",
    'peat_peatml': "peat_peatml",
    'peat_ogh': "peat_ogh",
    'peat_ogh_unthresholded': "peat_ogh_unthresholded",
    'dadap': "dadap",
    'engert': "engert",
    'grip': "grip",
    'osm_roads': "osm_roads",
    'osm_canals': "osm_canals",
}

download_dict = {}

sig_height_loss_threshold = 5
tree_threshold = 5

# ---------------------------------------------------------------------
# 4. Helper function
# ---------------------------------------------------------------------

def check_s3_path_exists(s3_client, bucket, path):
    try:
        s3_client.head_object(Bucket=bucket, Key=path)
        return True
    except Exception as e:
        print(f"S3 path check failed: {e}")
        return False

# -----------------------------------------------------------------
# 5. Make sure local_processed folders exist
# -----------------------------------------------------------------

def _walk(d):
    if isinstance(d, dict):
        yield d
        for v in d.values():
            yield from _walk(v)

for subdict in _walk(datasets):
    loc = subdict.get("local_processed")
    if loc:
        Path(loc).mkdir(parents=True, exist_ok=True)

# List of available tile IDs

tile_id_list = [
    '00N_000E', '00N_010E', '00N_020E', '00N_030E', '00N_040E', '00N_040W', '00N_050W', '00N_060W', '00N_070E',
    '00N_070W', '00N_080W', '00N_090E', '00N_090W', '00N_100E', '00N_100W', '00N_110E', '00N_120E', '00N_130E',
    '00N_140E', '00N_150E', '00N_160E', '10N_000E', '10N_010E', '10N_010W', '10N_020E', '10N_020W', '10N_030E',
    '10N_040E', '10N_050W', '10N_060W', '10N_070E', '10N_070W', '10N_080E', '10N_080W', '10N_090E', '10N_090W',
    '10N_100E', '10N_100W', '10N_110E', '10N_120E', '10N_130E', '10N_150E', '10N_160E', '10S_010E', '10S_020E',
    '10S_030E', '10S_040E', '10S_040W', '10S_050E', '10S_050W', '10S_060W', '10S_070W', '10S_080W', '10S_110E',
    '10S_120E', '10S_130E', '10S_140E', '10S_150E', '10S_160E', '10S_170E', '10S_180W', '20N_000E', '20N_010E',
    '20N_010W', '20N_020E', '20N_020W', '20N_030E', '20N_040E', '20N_050E', '20N_060W', '20N_070E', '20N_070W',
    '20N_080E', '20N_080W', '20N_090E', '20N_090W', '20N_100E', '20N_100W', '20N_110E', '20N_110W', '20N_120E',
    '20N_120W', '20N_160W', '20S_010E', '20S_020E', '20S_030E', '20S_040E', '20S_050E', '20S_050W', '20S_060W',
    '20S_070W', '20S_080W', '20S_110E', '20S_120E', '20S_130E', '20S_140E', '20S_150E', '20S_160E', '20S_180W',
    '30N_000E', '30N_010E', '30N_010W', '30N_020E', '30N_020W', '30N_030E', '30N_040E', '30N_050E', '30N_060E',
    '30N_070E', '30N_080E', '30N_080W', '30N_090E', '30N_090W', '30N_100E', '30N_100W', '30N_110E', '30N_110W',
    '30N_120E', '30N_120W', '30N_160W', '30N_170W', '30S_010E', '30S_020E', '30S_030E', '30S_060W', '30S_070W',
    '30S_080W', '30S_110E', '30S_120E', '30S_130E', '30S_140E', '30S_150E', '30S_170E', '40N_000E', '40N_010E',
    '40N_010W', '40N_020E', '40N_020W', '40N_030E', '40N_040E', '40N_050E', '40N_060E', '40N_070E', '40N_070W',
    '40N_080E', '40N_080W', '40N_090E', '40N_090W', '40N_100E', '40N_100W', '40N_110E', '40N_110W', '40N_120E',
    '40N_120W', '40N_130E', '40N_130W', '40N_140E', '40S_070W', '40S_080W', '40S_140E', '40S_160E', '40S_170E',
    '50N_000E', '50N_010E', '50N_010W', '50N_020E', '50N_030E', '50N_040E', '50N_050E', '50N_060E', '50N_060W',
    '50N_070E', '50N_070W', '50N_080E', '50N_080W', '50N_090E', '50N_090W', '50N_100E', '50N_100W', '50N_110E',
    '50N_110W', '50N_120E', '50N_120W', '50N_130E', '50N_130W', '50N_140E', '50N_150E', '50S_060W', '50S_070W',
    '50S_080W', '60N_000E', '60N_010E', '60N_010W', '60N_020E', '60N_020W', '60N_030E', '60N_040E', '60N_050E',
    '60N_060E', '60N_060W', '60N_070E', '60N_070W', '60N_080E', '60N_080W', '60N_090E', '60N_090W', '60N_100E',
    '60N_100W', '60N_110E', '60N_110W', '60N_120E', '60N_120W', '60N_130E', '60N_130W', '60N_140E', '60N_140W',
    '60N_150E', '60N_150W', '60N_160E', '60N_160W', '60N_170E', '60N_170W', '60N_180W', '70N_000E', '70N_010E',
    '70N_020E', '70N_030E', '70N_040E', '70N_050E', '70N_060E', '70N_070E', '70N_070W', '70N_080E', '70N_080W',
    '70N_090E', '70N_090W', '70N_100E', '70N_100W', '70N_110E', '70N_110W', '70N_120E', '70N_120W', '70N_130E',
    '70N_130W', '70N_140E', '70N_140W', '70N_150E', '70N_150W', '70N_160E', '70N_160W', '70N_170E', '70N_170W',
    '70N_180W', '80N_010E', '80N_020E', '80N_030E', '80N_070E', '80N_080E', '80N_090E', '80N_100E', '80N_110E',
    '80N_120E', '80N_130E', '80N_130W', '80N_140E', '80N_140W', '80N_150E', '80N_150W', '80N_160E', '80N_160W',
    '80N_170E', '80N_170W'
]