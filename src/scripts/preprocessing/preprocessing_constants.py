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
    tile_index_shapefile_prefix,
    tile_index_shapefile_name,
    tile_id_list as master_tile_id_list,
    peat_mask_dirs,
    dirs as model_input_dirs,
)


def _relative_peat_path(path: str) -> str:
    """Remove the bucket prefix and normalise a peat mask S3 directory."""

    prefix = f"s3://{s3_bucket_name}/"
    if path.startswith(prefix):
        path = path[len(prefix) :]
    return path.rstrip("/") + "/"

# Peat mask related paths
peat_pattern = "_peat_mask_processed.tif"
union_30m_prefix = _relative_peat_path(peat_mask_dirs["union_mask"])
peat_tiles_prefix = _relative_peat_path(peat_mask_dirs["gpd"])
peat_tiles_prefix_1km = union_30m_prefix.replace("/30m/", "/1km/")
peat_tiles_prefix_1km_3395 = union_30m_prefix.replace("/30m/", "/1km_3395/")

# Peat/organic-soil extraction source datasets and final model-facing union.
extraction_source_datasets = ("finland", "ireland", "russia")
extraction_final_s3_processed = _relative_peat_path(model_input_dirs["extraction"]).rstrip("/")
extraction_final_version = pp.basename(extraction_final_s3_processed)
extraction_final_local_processed = pp.join(local_temp_dir, 'extraction', 'final', extraction_final_version)

# Global tile index (no extension)
index_shapefile_prefix = tile_index_shapefile_prefix
index_shapefile_name = tile_index_shapefile_name

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
        # Original Dadap et al. (2021) 5 m binary canal mask (Stanford SDR
        # DOI 10.25740/yj761xk5815, CC-BY-ND), staged as a VRT/mosaic of the
        # 1664 zoom-15 tiles. Source for the distance-to-canal product below.
        's3_raw_canals_5m': pp.join(raw_dir, 'canals', 'Dadap_SEA_Drainage',
                                    'canals_5m', 'dadap_canals_5m.vrt'),
        's3_processed_base': pp.join(processed_dir, 'dadap_density', '30m'),
        's3_processed': pp.join(processed_dir, 'dadap_density', '30m', today_date),
        'local_processed': pp.join(local_temp_dir, 'dadap_density', today_date),
        'working_version': pp.join(processed_dir, 'dadap_density', '30m', '20240925'),
        # Distance-to-canal product (metres, model-facing). Built by
        # dadap_canal_distance.py; consumed by the drainage model like osm_canals.
        's3_distance_base': pp.join(processed_dir, 'dadap_density', 'distance'),
    },
    'planted_forest_type': {
        's3_processed_base': pp.join('climate', 'carbon_model', 'other_emissions_inputs',
                                     'plantation_type', 'SDPTv2', '20230911')
    },
    'sdpt': {
        's3_raw': pp.join('plantations', 'sdpt_v3','oct2025_updates','sdpt_by_tiles'),
        's3_processed_base': pp.join(processed_dir, 'sdpt'),
        's3_processed_small': pp.join(processed_dir, 'sdpt', 'chunks', today_date),
        's3_processed': pp.join(processed_dir, 'sdpt', today_date),
        'local_processed': pp.join(local_temp_dir, 'sdpt', today_date)
    },
    'extraction': {
        'finland': {
            's3_raw': f'{raw_dir}/extraction/Finland/Finland_turvetuotantoalueet/'
                      f'turvetuotantoalueet_jalkikaytto',
            's3_processed_base': f'{processed_dir}/extraction/',
            's3_processed': f'{processed_dir}/extraction/source/finland/{today_date}/',
            'local_processed': f'{local_temp_dir}/extraction/finland/{today_date}/'
        },
        'ireland': {
            's3_raw': f'{raw_dir}/extraction/Ireland/Ireland_Habibetal/RF_S2_LU_5_11_23.tif',
            's3_processed_base': f'{processed_dir}/extraction/',
            's3_processed': f'{processed_dir}/extraction/source/ireland/{today_date}/',
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
            's3_processed': f'{processed_dir}/extraction/source/russia/{today_date}/',
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
    'mangrove_extent': {
        's3_raw_root': pp.join('global-mangrove-extent', 'version3', 'smoothed', 'raster'),
        's3_processed_base': pp.join(processed_dir, 'mangrove_extent', 'hansen'),
        's3_processed': pp.join(processed_dir, 'mangrove_extent', 'hansen', today_date),
        'local_processed': pp.join(local_temp_dir, 'mangrove_extent', 'hansen', today_date)
    },
    'tidal_marshes': {
        's3_raw': pp.join(
            'climate',
            'AFOLU_flux_model',
            'organic_soils',
            'inputs',
            'raw',
            'coastal',
            'tidal_marsh',
        ),
        's3_processed_base': pp.join(processed_dir, 'tidal_marshes', 'hansen'),
        's3_processed': pp.join(processed_dir, 'tidal_marshes', 'hansen', today_date),
        'local_processed': pp.join(local_temp_dir, 'tidal_marshes', 'hansen', today_date),
    },
    'land_cover_ipcc': {
        's3_processed_base': pp.join(processed_dir, 'land_cover_ipcc'),
        's3_processed': pp.join(processed_dir, 'land_cover_ipcc', today_date),
        'local_processed': pp.join(local_temp_dir, 'land_cover_ipcc', today_date),
    },
    'peat': {
        'gfw': {
            's3_processed': _relative_peat_path(peat_mask_dirs['gfw'])
        },
        'gpd': {
            'input_type': 'raster',
            's3_raw': pp.join(raw_dir, 'soils', 'GPD', 'peatGPA22WGS_2cl.tif'),
            's3_processed': _relative_peat_path(peat_mask_dirs['gpd']),
            'local_processed': pp.join(local_temp_dir, 'peat', 'gpd', 'tiles', '20251110') + '/'
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
            's3_raw': 'https://s3.opengeohub.org/global/soil.types/organic.soils_ensemble.organic_p_30m_s_20000101_20241231_go_epsg.4326_v20260513.tif',
            's3_processed': pp.join(processed_dir, 'peat_mask', 'OGH', 'tiles') + '/',
            'local_processed': pp.join(local_temp_dir, 'peat', 'ogh', 'tiles') + '/',
            'threshold': 9
        },
        'ogh_unthresholded': {
            'input_type': 'raster',
            's3_raw': 'https://s3.opengeohub.org/global/soil.types/organic.soils_ensemble.organic_p_30m_s_20000101_20241231_go_epsg.4326_v20260513.tif',
            's3_processed': _relative_peat_path(peat_mask_dirs['ogh_unthresholded']),
            'local_processed': pp.join(local_temp_dir, 'peat', 'ogh_unthresholded', 'tiles', '20260513') + '/'
        },
        'union_mask': {
            '30m': union_30m_prefix,
            '1km': union_30m_prefix.replace('/30m/', '/1km/'),
            '1km_3395': union_30m_prefix.replace('/30m/', '/1km_3395/'),
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

tile_id_list = list(master_tile_id_list)
