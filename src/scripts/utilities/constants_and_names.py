import logging
import posixpath
from datetime import datetime
import numpy as np
import os

from src.scripts.utilities.tile_index import load_tile_ids_from_s3
from src.scripts.utilities.lulucf_constants_and_names import (
    climate_domain_dir as lulucf_climate_domain_dir,
    climate_domain_pattern as lulucf_climate_domain_pattern,
)
from src.scripts.utilities import local_output_paths as lop

# ---------------------------------------------------
# 1. General Configuration
# ---------------------------------------------------

# ── version helpers ──────────────────────────────────────────────
model_version = "0.1.4"              # dotted string
model_version_underscore = model_version.replace(".", "_")   # "0_3_0"

s3_bucket_name = 'gfw2-data'

logger = logging.getLogger(__name__)
full_bucket_prefix = f"s3://{s3_bucket_name}"
s3_region_name = 'us-east-1'

short_bucket_prefix = "gfw2-data"
full_bucket_prefix_length = len(full_bucket_prefix)+1

project_dir = 'climate/AFOLU_flux_model/organic_soils'
raw_dir = posixpath.join(project_dir, 'inputs/raw')
processed_dir = posixpath.join(project_dir, 'inputs/processed')

date_date_range_pattern = r'_\d{4}(_\d{4})?'

# organic‑soils constants file
outputs_path = posixpath.join(
    full_bucket_prefix,                      # "s3://gfw2-data"
    project_dir,                             # "climate/AFOLU_flux_model/organic_soils"
    "outputs",
    f"version_{model_version_underscore}"    # "version_0_3_0"
)   # → s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_0

drainage_outputs_path_mega_zarr = posixpath.join(outputs_path, "mega_zarr")

drainage_outputs_to_zarr = [
    "drained_soil",
    "combined_state",
    "burned_years_count",
    "drained_co2_Mg_CO2_ha_yr",
    "drained_n2o_Mg_CO2e_ha_yr",
    "drained_ch4_land_Mg_CO2e_ha_yr",
    "drained_ch4_ditch_Mg_CO2e_ha_yr",
    "drained_co2_offsite_Mg_CO2_ha_yr",
    "drained_total_Mg_CO2e_ha_yr",
    "burned_co2_Mg_CO2_ha_yr",
    "burned_co_Mg_CO_ha_yr",
    "burned_ch4_Mg_CO2e_ha_yr",
    "burned_total_Mg_CO2e_ha_yr",
]

# Optional legacy categorical state outputs.
# These are computed by the model for combined-state packing/diagnostics but
# are not written by default in standard production runs.
drainage_optional_state_outputs = [
    "drained_state",
    "burned_state",
]

drainage_output_dtypes = {
    "drained_soil": "uint32",
    "drained_state": "uint32",
    "burned_state": "uint32",
    "combined_state": "uint32",
    "burned_years_count": "uint32",
    "drained_co2_Mg_CO2_ha_yr": "float32",
    "drained_n2o_Mg_CO2e_ha_yr": "float32",
    "drained_ch4_land_Mg_CO2e_ha_yr": "float32",
    "drained_ch4_ditch_Mg_CO2e_ha_yr": "float32",
    "drained_co2_offsite_Mg_CO2_ha_yr": "float32",
    "drained_total_Mg_CO2e_ha_yr": "float32",
    "burned_co2_Mg_CO2_ha_yr": "float32",
    "burned_co_Mg_CO_ha_yr": "float32",
    "burned_ch4_Mg_CO2e_ha_yr": "float32",
    "burned_total_Mg_CO2e_ha_yr": "float32",
}

s3_log_path = posixpath.join(project_dir, 'model_logs')

# Shapefile of global 1x1 degree chunks with GADM ISO codes
fishnet_1x1deg_uri = posixpath.join(
    full_bucket_prefix,
    'climate/AFOLU_flux_model/fishnet_1x1deg/20250429',
    'fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp'
)


# Global 10x10 degree tile index shapefile used to drive preprocessing and
# extent selection in organic-soils workflows.
tile_index_shapefile_prefix = posixpath.join(
    'climate/AFOLU_flux_model/fishnet_10x10deg/20250925',
    'fishnet_10x10deg_20251107_intersect'
)
tile_index_shapefile_uri = posixpath.join(
    full_bucket_prefix,
    f"{tile_index_shapefile_prefix}.shp"
)
tile_index_shapefile_name = f"{posixpath.basename(tile_index_shapefile_prefix)}.shp"


local_log_path = "./logs/"
os.makedirs(local_log_path, exist_ok=True)

local_root = 'C:/GIS/Data/Global'
local_temp_dir = '/tmp'

today_date = datetime.utcnow().strftime('%Y%m%d')


def get_today_date_utc() -> str:
    """Return today's UTC date as YYYYMMDD, evaluated at call time."""
    return datetime.utcnow().strftime('%Y%m%d')

# ---------------------------------------------------
# 2. Tile and Chunk Patterns
# ---------------------------------------------------

tile_id_pattern = r"[0-9]{2}[A-Z][_][0-9]{3}[A-Z]"
small_chunk_pattern = r"__-?\d+_-?\d+_-?\d+_-?\d+__"
# Legacy single-underscore chunk notation, e.g., _8_-1_9_0_
old_small_chunk_pattern = r"_-?\d+_-?\d+_-?\d+_-?\d+_"

sample_tile_id = '{tile_id}'

full_raster_dims = 40000
local_chunk_stats_path = lop.chunk_stats_root()
s3_chunk_stats_path = "climate/AFOLU_flux_model/organic_soils/chunk_stats/"

pixel_area_dir = f"{full_bucket_prefix}/analyses/area_28m/"
pixel_area_pattern = "hanson_2013_area"

# Canonical global contextual zarrs (shared with vegetation/zonal workflows).
contextual_layer_global_zarr_root = (
    f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr"
)

WDPA_pattern = "wdpa"
WDPA_zarr_path = (
    f"{contextual_layer_global_zarr_root}/wdpa/20260224/global_wdpa_20260224.zarr"
)
# WDPA classes are currently binary: 0=outside, 1=inside.
WDPA_codes = np.array([0, 1], dtype=np.uint16)

KBA_pattern = "kba"
KBA_zarr_path = (
    f"{contextual_layer_global_zarr_root}/kba/20260224/global_kba_20260224.zarr"
)
# KBA classes are currently binary: 0=outside, 1=inside.
KBA_codes = np.array([0, 1], dtype=np.uint16)

progress_tracking_path = "climate/AFOLU_flux_model/organic_soils/task_progress_txts/"

possible_task_statuses = [
    "pending_",
    "loading_",
    "preprocessing_",
    "calculating_",
    "uploading_",
    "error_",
]


# conversion factor from square meters to hectares
m2_to_ha = 1e-4


# ---------------------------------------------------
# 3. Dataset File Patterns
# ---------------------------------------------------

# TODO add all peat patterns (and look at thresholds)
patterns = {
    'land_cover': "{tile_id}__lc_ipcc.tif",
    'peat': "{tile_id}.tif",
    'dadap': "dadap_{tile_id}.tif",
    'engert': "engert_{tile_id}.tif",
    'grip': "{tile_id}__grip_roads_distance.tif",
    'osm_roads': "{tile_id}__osm_roads_distance.tif",
    'osm_canals': "{tile_id}__osm_canals_distance.tif",
    'planted_forest_type': "{tile_id}__sdpt.tif",
    'extraction': "{tile_id}_extraction.tif",
    'climate_domain': f"{{tile_id}}_{lulucf_climate_domain_pattern}.tif",
    'descals_type': "plantation_type_{tile_id}.tif",
    'mangrove_extent': "{tile_id}__gmw_mangrove_any_year.tif",
    'tidal_marsh': "{tile_id}.tif",
    'ogh': "{tile_id}_ogh_mask.tif",
    'burned_area_final': "{tile_id}_burned_area_final_{year}.tif"
}

# ---------------------------------------------------
# 4. Dataset Directories
# ---------------------------------------------------

dirs = {
    'land_cover': posixpath.join(
        full_bucket_prefix,
        'climate/AFOLU_flux_model/LULUCF/landcover/composite/{interval_type}',
        '{land_cover_version}',
        'raw'
    ),
    'peat': posixpath.join(full_bucket_prefix, raw_dir, 'soils/GFW_Global_Peatlands'),
    'dadap': posixpath.join(full_bucket_prefix, processed_dir, 'dadap_density/30m/20240925'),
    'engert': posixpath.join(full_bucket_prefix, processed_dir, 'engert_density/30m/20240925'),
    'grip': posixpath.join(full_bucket_prefix, processed_dir, f'grip_density/distance/{full_raster_dims}_pixels/20251115'),
    'osm_roads': posixpath.join(full_bucket_prefix, processed_dir, f'osm_roads_density/distance/{full_raster_dims}_pixels/20251114'),
    'osm_canals': posixpath.join(full_bucket_prefix, processed_dir, f'osm_canals_density/distance/{full_raster_dims}_pixels/20251115'),
    'planted_forest_type': posixpath.join(full_bucket_prefix, processed_dir, f'sdpt/{full_raster_dims}_pixels/20250531'),
    'extraction': posixpath.join(full_bucket_prefix, processed_dir, 'extraction/20241021'),
    'climate_domain': lulucf_climate_domain_dir,
    'descals_type': posixpath.join(full_bucket_prefix, processed_dir, 'descals_plantation/extent/20241105'),
    'mangrove_extent': posixpath.join(full_bucket_prefix, processed_dir, 'mangrove_extent/hansen/20251112'),
    'tidal_marsh': posixpath.join(full_bucket_prefix, processed_dir, 'tidal_marshes/hansen/20251112'),
    'burned_area_final': posixpath.join(full_bucket_prefix, processed_dir, 'fires/{year}')
}

# directories for 30 m peat mask datasets
peat_mask_dirs = {
    # TODO add full gfw path here, change all references
    'gfw': dirs['peat'],
    'gpd': posixpath.join(
        full_bucket_prefix,
        processed_dir,
        'peat_mask/GPD/tiles/20251110',
    ),
    'peatmap': posixpath.join(full_bucket_prefix, processed_dir, 'peat_mask/PEATMAP/tiles'),
    'peatml': posixpath.join(full_bucket_prefix, processed_dir, 'peat_mask/PEATML/tiles'),
    'ogh': posixpath.join(full_bucket_prefix, processed_dir, 'peat_mask/OGH/tiles'),
    'ogh_unthresholded': posixpath.join(
        full_bucket_prefix,
        processed_dir,
        'peat_mask/OGH/tiles_unthresholded/20251110',
    ),
    'union_mask': posixpath.join(
        full_bucket_prefix,
        processed_dir,
        'peat_mask/union/30m/tiles/20251110',
    ),
}

peat_dataset_choices = tuple(peat_mask_dirs.keys())

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

# Mapping from FAO ecozone classes to simplified climate domain codes
climate_domain_remap = {
    0: ecozone_codes['unknown'],
    1: ecozone_codes['tropical'],
    2: ecozone_codes['boreal'],
    3: ecozone_codes['temperate'],
}

nutrient_status_codes = {
    'unknown': 0,
    'poor': 1,
    'rich': 2
}

plantation_type_codes = {
    'oil_palm': 1,
    'short_rotation': 2,
    'long_rotation': 3
}

sig_height_loss_threshold = 5
tree_threshold = 5

t_to_Mt = 1e-3
combined_log = "combined_log"

# Do not apply this to organic-soil fire factors. Those factor tables store
# IPCC Table 2.6 fuel consumption values that already represent MB * Cf.
combustion_factor = np.float32(0.75)

# Global warming potentials (GWP) and emission conversions
gwp_ch4 = np.float32(27.0)
gwp_n2o = np.float32(273.0)
c_to_co2 = np.float32(3.67)
n2o_n_to_n2o = np.float32(1.571)

# ---------------------------------------------------
# 6. Time Interval Handling Constants
# ---------------------------------------------------

intervals_annual = "annual"
intervals_five_year = "five_year"
intervals_hybrid = "hybrid"

annual_land_cover_start_year = 2015

# Version mapping for land cover composites
land_cover_version_map = {
    intervals_annual: "v2",
    intervals_five_year: "v1",
}

# Date sub-folders for IPCC land cover composites by interval type
land_cover_ipcc_dates = {
    intervals_annual: "20250715",
    intervals_five_year: "20250710",
}

# End years that correspond to available five year land cover composites
five_year_land_cover_years = [2005, 2010, 2015, 2020, 2024]

# Convenience list of five year inventory periods. The first period now uses the 2005 land cover composite (2000–2004). The final period uses the 2024 land cover composite for the 2024 inventory year.
five_year_inventory_periods = [
    (2001, 2005),
    (2006, 2010),
    (2011, 2015),
    (2016, 2020),
    (2021, 2024),
]

# -------------------------------------------------------------------
# Convenience list of all 10x10 degree tile IDs used by the model
# -------------------------------------------------------------------
# Duplicated from ``preprocessing_constants.tile_id_list`` so that core
# model code can access the list without importing preprocessing modules.
_FALLBACK_TILE_ID_LIST = [
    '00N_000E', '00N_010E', '00N_020E', '00N_020W', '00N_030E', '00N_040E',
    '00N_040W', '00N_050E', '00N_050W', '00N_060W', '00N_070E', '00N_070W',
    '00N_080W', '00N_090E', '00N_090W', '00N_100E', '00N_100W', '00N_110E',
    '00N_120E', '00N_130E', '00N_140E', '00N_140W', '00N_150E', '00N_150W',
    '00N_160E', '00N_160W', '00N_170E', '00N_170W', '00N_180W', '10N_000E',
    '10N_010E', '10N_010W', '10N_020E', '10N_020W', '10N_030E', '10N_040E',
    '10N_050E', '10N_050W', '10N_060W', '10N_070E', '10N_070W', '10N_080E',
    '10N_080W', '10N_090E', '10N_090W', '10N_100E', '10N_100W', '10N_110E',
    '10N_110W', '10N_120E', '10N_120W', '10N_130E', '10N_140E', '10N_150E',
    '10N_160E', '10N_160W', '10N_170E', '10N_170W', '10N_180W', '10S_010E',
    '10S_010W', '10S_020E', '10S_030E', '10S_030W', '10S_040E', '10S_040W',
    '10S_050E', '10S_050W', '10S_060E', '10S_060W', '10S_070W', '10S_080W',
    '10S_090E', '10S_100E', '10S_110E', '10S_120E', '10S_130E', '10S_140E',
    '10S_140W', '10S_150E', '10S_150W', '10S_160E', '10S_160W', '10S_170E',
    '10S_170W', '10S_180W', '20N_000E', '20N_010E', '20N_010W', '20N_020E',
    '20N_020W', '20N_030E', '20N_030W', '20N_040E', '20N_050E', '20N_060E',
    '20N_060W', '20N_070E', '20N_070W', '20N_080E', '20N_080W', '20N_090E',
    '20N_090W', '20N_100E', '20N_100W', '20N_110E', '20N_110W', '20N_120E',
    '20N_120W', '20N_130E', '20N_140E', '20N_150E', '20N_160E', '20N_160W',
    '20N_170E', '20N_170W', '20N_180W', '20S_010E', '20S_020E', '20S_030E',
    '20S_030W', '20S_040E', '20S_040W', '20S_050E', '20S_050W', '20S_060E',
    '20S_060W', '20S_070W', '20S_080W', '20S_090W', '20S_110E', '20S_110W',
    '20S_120E', '20S_120W', '20S_130E', '20S_130W', '20S_140E', '20S_140W',
    '20S_150E', '20S_150W', '20S_160E', '20S_160W', '20S_170E', '20S_170W',
    '20S_180W', '30N_000E', '30N_010E', '30N_010W', '30N_020E', '30N_020W',
    '30N_030E', '30N_040E', '30N_050E', '30N_060E', '30N_070E', '30N_070W',
    '30N_080E', '30N_080W', '30N_090E', '30N_090W', '30N_100E', '30N_100W',
    '30N_110E', '30N_110W', '30N_120E', '30N_120W', '30N_130E', '30N_140E',
    '30N_150E', '30N_160E', '30N_160W', '30N_170W', '30N_180W', '30S_010E',
    '30S_010W', '30S_020E', '30S_020W', '30S_030E', '30S_050W', '30S_060W',
    '30S_070E', '30S_070W', '30S_080W', '30S_090W', '30S_110E', '30S_120E',
    '30S_130E', '30S_140E', '30S_150E', '30S_160E', '30S_170E', '30S_180W',
    '40N_000E', '40N_010E', '40N_010W', '40N_020E', '40N_020W', '40N_030E',
    '40N_030W', '40N_040E', '40N_040W', '40N_050E', '40N_060E', '40N_070E',
    '40N_070W', '40N_080E', '40N_080W', '40N_090E', '40N_090W', '40N_100E',
    '40N_100W', '40N_110E', '40N_110W', '40N_120E', '40N_120W', '40N_130E',
    '40N_130W', '40N_140E', '40S_010W', '40S_020W', '40S_040E', '40S_050E',
    '40S_060E', '40S_070E', '40S_070W', '40S_080W', '40S_140E', '40S_160E',
    '40S_170E', '40S_180W', '50N_000E', '50N_010E', '50N_010W', '50N_020E',
    '50N_020W', '50N_030E', '50N_030W', '50N_040E', '50N_040W', '50N_050E',
    '50N_060E', '50N_060W', '50N_070E', '50N_070W', '50N_080E', '50N_080W',
    '50N_090E', '50N_090W', '50N_100E', '50N_100W', '50N_110E', '50N_110W',
    '50N_120E', '50N_120W', '50N_130E', '50N_130W', '50N_140E', '50N_140W',
    '50N_150E', '50S_000E', '50S_030W', '50S_040W', '50S_050W', '50S_060E',
    '50S_060W', '50S_070E', '50S_070W', '50S_080W', '50S_150E', '50S_160E',
    '50S_170E', '60N_000E', '60N_010E', '60N_010W', '60N_020E', '60N_020W',
    '60N_030E', '60N_040E', '60N_050E', '60N_050W', '60N_060E', '60N_060W',
    '60N_070E', '60N_070W', '60N_080E', '60N_080W', '60N_090E', '60N_090W',
    '60N_100E', '60N_100W', '60N_110E', '60N_110W', '60N_120E', '60N_120W',
    '60N_130E', '60N_130W', '60N_140E', '60N_140W', '60N_150E', '60N_150W',
    '60N_160E', '60N_160W', '60N_170E', '60N_170W', '60N_180W', '70N_000E',
    '70N_010E', '70N_010W', '70N_020E', '70N_020W', '70N_030E', '70N_030W',
    '70N_040E', '70N_040W', '70N_050E', '70N_050W', '70N_060E', '70N_060W',
    '70N_070E', '70N_070W', '70N_080E', '70N_080W', '70N_090E', '70N_090W',
    '70N_100E', '70N_100W', '70N_110E', '70N_110W', '70N_120E', '70N_120W',
    '70N_130E', '70N_130W', '70N_140E', '70N_140W', '70N_150E', '70N_150W',
    '70N_160E', '70N_160W', '70N_170E', '70N_170W', '70N_180W', '80N_000E',
    '80N_010E', '80N_010W', '80N_020E', '80N_020W', '80N_030E', '80N_030W',
    '80N_040E', '80N_040W', '80N_050E', '80N_050W', '80N_060E', '80N_060W',
    '80N_070E', '80N_070W', '80N_080E', '80N_080W', '80N_090E', '80N_090W',
    '80N_100E', '80N_100W', '80N_110E', '80N_110W', '80N_120E', '80N_120W',
    '80N_130E', '80N_130W', '80N_140E', '80N_140W', '80N_150E', '80N_150W',
    '80N_160E', '80N_160W', '80N_170E', '80N_170W', '80N_180W'
]

_loaded_tile_ids = load_tile_ids_from_s3(
    s3_bucket_name=s3_bucket_name,
    s3_prefix=tile_index_shapefile_prefix,
    local_dir=local_temp_dir,
)

if _loaded_tile_ids:
    tile_id_list = _loaded_tile_ids
    tile_id_list_source = tile_index_shapefile_uri
else:
    logger.warning(
        "Falling back to hard-coded tile_id_list because the 10x10 tile index "
        "shapefile could not be loaded."
    )
    tile_id_list = list(_FALLBACK_TILE_ID_LIST)
    tile_id_list_source = "fallback_tile_roster"

burned_area_final_pattern = "burned_area_final"
land_cover_pattern = "land_cover"

# --- Display / cartography constants (for global JPEGs) ---

_SHAPEFILE_DIR = os.environ.get(
    "AFOLU_WORLD_BOUNDARIES_DIR", lop.world_boundaries_dir()
)
original_shapefile_path = posixpath.join(_SHAPEFILE_DIR, "admin0_simp_dis.shp")
reprojected_shapefile_path = posixpath.join(_SHAPEFILE_DIR, "admin0_simp_dis.shp")

# Map projection
Robinson_crs = "ESRI:54030"

# Styling
land_bkgrnd        = (245, 245, 245)   # land fill under no-data
ocean_color        = (235, 235, 235)   # panel background
boundary_color     = (150, 150, 150)
boundary_width     = 0.2
panel_dims         = (12, 6)           # inches
dpi_jpeg           = 300
legend_fontsize    = 9
colorbar_dimensions = [0.14, 0.17, 0.02, 0.13]  # [left, bottom, width, height]

# Optional note on the panel (bottom-right)
pres_text = f"Preliminary organic soils fluxes (model v{model_version})"

# ---------------------------------------------------
# 7. Dynamic Download Dictionary Function
# ---------------------------------------------------

def get_dynamic_download_dict(tile_id, interval_start_year, interval_end_year=None, peat_dataset='ogh'):
    if interval_end_year is None:
        interval_end_year = interval_start_year

    lc_year = interval_end_year
    if lc_year in {2015, 2020, 2024}:
        # These inventory years are backed by the annual land cover composites
        interval_type = intervals_annual
    elif lc_year in five_year_land_cover_years:
        interval_type = intervals_five_year
    else:
        interval_type = intervals_annual

    # Updated IPCC land cover directory
    pixel_resolution = "40000_pixels"
    land_cover_ipcc_dir = posixpath.join(
        full_bucket_prefix,
        processed_dir,
        'land_cover_ipcc',
        land_cover_ipcc_dates[interval_type],
        interval_type,
        str(lc_year),
        pixel_resolution,
    )


    peat_dataset = peat_dataset.lower()
    if peat_dataset not in peat_mask_dirs:
        raise ValueError(f"Unknown peat dataset: {peat_dataset}")

    peat_path_dataset = peat_dataset
    if peat_dataset == 'ogh':
        # Use the unthresholded probability surface for OGH and apply
        # thresholds downstream in the core model.
        peat_path_dataset = 'ogh_unthresholded'

    if peat_path_dataset == 'gfw':
        peat_path = posixpath.join(
            peat_mask_dirs['gfw'], patterns['peat'].format(tile_id=tile_id)
        )
    elif peat_path_dataset == 'union_mask':
        peat_path = posixpath.join(
            peat_mask_dirs['union_mask'], f"{tile_id}_union_mask.tif"
        )
    else:
        peat_path = posixpath.join(
            peat_mask_dirs[peat_path_dataset],
            f"{tile_id}_{peat_path_dataset}_mask.tif"
        )

    dynamic_dict = {
        'land_cover': posixpath.join(
            land_cover_ipcc_dir,
            patterns['land_cover'].format(tile_id=tile_id)
        ),
        'peat': peat_path,
        'dadap': posixpath.join(dirs['dadap'], patterns['dadap'].format(tile_id=tile_id)),
        'engert': posixpath.join(dirs['engert'], patterns['engert'].format(tile_id=tile_id)),
        'grip': posixpath.join(dirs['grip'], patterns['grip'].format(tile_id=tile_id)),
        'osm_roads': posixpath.join(dirs['osm_roads'], patterns['osm_roads'].format(tile_id=tile_id)),
        'osm_canals': posixpath.join(dirs['osm_canals'], patterns['osm_canals'].format(tile_id=tile_id)),
        'planted_forest_type': posixpath.join(dirs['planted_forest_type'], patterns['planted_forest_type'].format(tile_id=tile_id)),
        'extraction': posixpath.join(dirs['extraction'], patterns['extraction'].format(tile_id=tile_id)),
        'climate_domain': posixpath.join(
            dirs['climate_domain'],
            patterns['climate_domain'].format(tile_id=tile_id),
        ),
        'descals_type': posixpath.join(dirs['descals_type'], patterns['descals_type'].format(tile_id=tile_id)),
        'mangrove_extent': posixpath.join(dirs['mangrove_extent'], patterns['mangrove_extent'].format(tile_id=tile_id)),
        'tidal_marsh': posixpath.join(dirs['tidal_marsh'], patterns['tidal_marsh'].format(tile_id=tile_id)),
        'ogh': posixpath.join(
            peat_mask_dirs['ogh_unthresholded'],
            f"{tile_id}_ogh_unthresholded_mask.tif",
        ),
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
