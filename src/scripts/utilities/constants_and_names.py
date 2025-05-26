import posixpath
from datetime import datetime
import numpy as np
import os

# ---------------------------------------------------
# 1. General Configuration
# ---------------------------------------------------

# ── version helpers ──────────────────────────────────────────────
model_version            = "0.3.0"              # dotted string
model_version_underscore = model_version.replace(".", "_")   # "0_3_0"

s3_bucket_name = 'gfw2-data'
full_bucket_prefix = f"s3://{s3_bucket_name}"
s3_region_name = 'us-east-1'

project_dir = 'climate/AFOLU_flux_model/organic_soils'
raw_dir = posixpath.join(project_dir, 'inputs/raw')
processed_dir = posixpath.join(project_dir, 'inputs/processed')

# organic‑soils constants file
outputs_path = posixpath.join(
    full_bucket_prefix,                      # "s3://gfw2-data"
    project_dir,                             # "climate/AFOLU_flux_model/organic_soils"
    "outputs",
    f"version_{model_version_underscore}"    # "version_0_3_0"
)   # → s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_0

s3_log_path = posixpath.join(project_dir, 'model_logs')

local_log_path = "./logs/"
os.makedirs(local_log_path, exist_ok=True)

local_root = 'C:/GIS/Data/Global'
local_temp_dir = '/tmp'

today_date = datetime.today().strftime('%Y%m%d')

# ---------------------------------------------------
# 2. Tile and Chunk Patterns
# ---------------------------------------------------

tile_id_pattern = r"[0-9]{2}[A-Z][_][0-9]{3}[A-Z]"
small_chunk_pattern = r"__-?\d+_-?\d+_-?\d+_-?\d+__"

sample_tile_id = '{tile_id}'

full_raster_dims = 40000
chunk_stats_path = posixpath.join(local_temp_dir, 'chunk_stats')

# directory containing per-pixel area rasters (hectares)
pixel_area_ha_dir = posixpath.join(
    full_bucket_prefix,
    processed_dir,
    f'pixel_area_ha/{full_raster_dims}_pixels/20240101'
)

# directory containing per-pixel area rasters (square meters)
pixel_area_dir = posixpath.join(
    full_bucket_prefix,
    processed_dir,
    f'pixel_area_m2/{full_raster_dims}_pixels/20240101'
)

# file pattern for pixel area rasters
pixel_area_pattern = 'pixel_area'

# conversion factor from square meters to hectares
m2_to_ha = 1e-4


# ---------------------------------------------------
# 3. Dataset File Patterns
# ---------------------------------------------------

patterns = {
    'land_cover': "{tile_id}.tif",
    'peat': "{tile_id}.tif",
    'dadap': "dadap_{tile_id}.tif",
    'engert': "engert_{tile_id}.tif",
    'grip': "{tile_id}_grip_density.tif",
    'osm_roads': "{tile_id}_osm_roads_density.tif",
    'osm_canals': "{tile_id}_osm_canals_density.tif",
    'planted_forest_type': "{tile_id}_sdpt.tif",
    'extraction': "{tile_id}_extraction.tif",
    'climate_domain': "{tile_id}_fao_ecozones_bor_tem_tro_processed.tif",
    'descals_type': "plantation_type_{tile_id}.tif",
    'ogh': "{tile_id}.tif",
    'burned_area_final': "{tile_id}_burned_area_final_{year}.tif"
}

# ---------------------------------------------------
# 4. Dataset Directories
# ---------------------------------------------------

dirs = {
    'land_cover': posixpath.join(full_bucket_prefix, 'climate/AFOLU_flux_model/LULUCF/landcover/composite/{interval_type}/v1/raw'),
    'peat': posixpath.join(full_bucket_prefix, raw_dir, 'soils/GFW_Global_Peatlands'),
    'dadap': posixpath.join(full_bucket_prefix, processed_dir, 'dadap_density/30m/20240925'),
    'engert': posixpath.join(full_bucket_prefix, processed_dir, 'engert_density/30m/20240925'),
    'grip': posixpath.join(full_bucket_prefix, processed_dir, f'grip_density/{full_raster_dims}_pixels/20240925'),
    'osm_roads': posixpath.join(full_bucket_prefix, processed_dir, f'osm_roads_density/{full_raster_dims}_pixels/20240925'),
    'osm_canals': posixpath.join(full_bucket_prefix, processed_dir, f'osm_canals_density/{full_raster_dims}_pixels/20240822'),
    'planted_forest_type': posixpath.join(full_bucket_prefix, processed_dir, f'sdpt/{full_raster_dims}_pixels/20240925'),
    'extraction': posixpath.join(full_bucket_prefix, processed_dir, 'extraction/20241021'),
    'climate_domain': posixpath.join(full_bucket_prefix, 'climate/carbon_model/inputs_for_carbon_pools/processed/fao_ecozones_bor_tem_tro/20190418'),
    'descals_type': posixpath.join(full_bucket_prefix, processed_dir, 'descals_plantation/extent/20241105'),
    'ogh': posixpath.join(full_bucket_prefix, raw_dir, 'soils/OGH'),
    'burned_area_final': posixpath.join(full_bucket_prefix, 'fires/MODIS_burned_area/MCD64A1.061/2_final_outputs__Hansenized/{year}')
}

# ---------------------------------------------------
# 5. Classification and Conversion Constants
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
    'temperate': 3
}

nutrient_status_codes = {
    'unknown': 0,
    'poor': 1,
    'rich': 2
}

# TODO these need to be updated when SDPT finalized
plantation_type_codes = {
    'unknown': 0,
    'long_rotation': 2,
    'short_rotation': 2,
    'oil_palm': 1,
    'sago_palm': 3
}

sig_height_loss_threshold = 5
tree_threshold = 5

t_to_Mt = 1e-3
combined_log = "combined_log"

combustion_factor = np.float32(0.75)

# Global warming potentials (GWP) and emission conversions
gwp_ch4 = np.float32(28.0)
gwp_n2o = np.float32(265.0)
gwp_co = np.float32(1.9)  # Verify this value as needed
c_to_co2 = np.float32(3.67)
n2o_n_to_n2o = np.float32(1.571)

# ---------------------------------------------------
# 6. Time Interval Handling Constants
# ---------------------------------------------------

intervals_annual = "annual"
intervals_five_years = "five_years"
intervals_hybrid = "hybrid"

burned_area_final_pattern = "burned_area_final"
land_cover_pattern = "land_cover"

# ---------------------------------------------------
# 7. Dynamic Download Dictionary Function
# ---------------------------------------------------

def get_dynamic_download_dict(tile_id, interval_start_year, interval_end_year=None):
    if interval_end_year is None:
        interval_end_year = interval_start_year

    lc_year = interval_end_year
    interval_type = (
        intervals_five_years
        if lc_year in [2000, 2005, 2010, 2015, 2020]
        else intervals_annual
    )

    # LULUCF land cover dir update
    lulucf_land_cover_dir = posixpath.join(
        full_bucket_prefix,
        'climate/AFOLU_flux_model/LULUCF/landcover/composite',
        interval_type,
        'v1',
        'raw',
        str(lc_year)
    )


    dynamic_dict = {
        'land_cover': posixpath.join(lulucf_land_cover_dir, patterns['land_cover'].format(tile_id=tile_id)),
        'peat': posixpath.join(dirs['peat'], patterns['peat'].format(tile_id=tile_id)),
        'dadap': posixpath.join(dirs['dadap'], patterns['dadap'].format(tile_id=tile_id)),
        'engert': posixpath.join(dirs['engert'], patterns['engert'].format(tile_id=tile_id)),
        'grip': posixpath.join(dirs['grip'], patterns['grip'].format(tile_id=tile_id)),
        'osm_roads': posixpath.join(dirs['osm_roads'], patterns['osm_roads'].format(tile_id=tile_id)),
        'osm_canals': posixpath.join(dirs['osm_canals'], patterns['osm_canals'].format(tile_id=tile_id)),
        'planted_forest_type': posixpath.join(dirs['planted_forest_type'], patterns['planted_forest_type'].format(tile_id=tile_id)),
        'extraction': posixpath.join(dirs['extraction'], patterns['extraction'].format(tile_id=tile_id)),
        'climate_domain': posixpath.join(dirs['climate_domain'], patterns['climate_domain'].format(tile_id=tile_id)),
        'descals_type': posixpath.join(dirs['descals_type'], patterns['descals_type'].format(tile_id=tile_id)),
        'ogh': posixpath.join(dirs['ogh'], patterns['ogh'].format(tile_id=tile_id)),
    }

    # Add burned area layers for each year in the interval
    for yr in range(interval_start_year, interval_end_year + 1):
        burned_key = f"burned_area_final_{yr}"
        dynamic_dict[burned_key] = posixpath.join(
            dirs['burned_area_final'].format(year=yr),
            patterns['burned_area_final'].format(tile_id=tile_id, year=yr)
        )

    return dynamic_dict


# ---------------------------------------------------
# 8. Test Path Construction
# ---------------------------------------------------

if __name__ == "__main__":
    test_dict = get_dynamic_download_dict('00N_110E', 2015)
    for k, v in test_dict.items():
        print(f"{k}: {v}")