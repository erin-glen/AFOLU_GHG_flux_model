"""
Script to check contents of mega-zarr in a 1x1 deg chunk (to confirm it has data).
Adapted from zu.zarr_1x1_deg_stats

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

python -m src.LULUCF.test_code.check_mega_zarr_contents
"""

import fsspec
import time
import numpy as np
import pandas as pd
from dask.distributed import print
import zarr

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import numba_utilities as nu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu

# Settings-- modify these
bounds = [124, -30, 125, -29]
zarr_path = 's3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_3_AUS_only/mega_zarr/standard_model/annual_intervals/10000_pixels/20251209'
var_name = 'carbon_density__AGC__MgC'


bounds_str = uu.boundstr(bounds)  # String form of chunk bounds, from e.g., [8, -1, 9, 0] to 8_-1_9_0
tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W

zarr_stats_raw_all_years = []

print(f"Getting stats for {var_name} for {bounds_str}: {uu.timestr()}")

# Bounding box to get stats for, reformatted for zarr extraction
target_box = {
    "lat_min": bounds[1],
    "lat_max": bounds[3],
    "lon_min": bounds[0],
    "lon_max": bounds[2]
}

# print(f"Getting indices for {bounds_str}")
lat0, lon0 = zu.latlon_to_global_zarr_indices(target_box["lat_max"], target_box["lon_min"], cn.resolution)
lat1, lon1 = zu.latlon_to_global_zarr_indices(target_box["lat_min"], target_box["lon_max"], cn.resolution)

fs = fsspec.filesystem("s3", anon=False)

# Calculates chunk stats on the chunk of the zarr.
# Rather than encoding rows as input or output layer, they are encoded by whether they are raw or rechunked zarr
# since all of these are outputs.
# Chunk stats are dictionaries.
# print(f"Getting mapper for {bounds_str}")
zarr_mapper = fs.get_mapper(zarr_path)
# print(f"Opening zarr for {bounds_str}")
zarr_group = zarr.open(zarr_mapper, mode="r")
# print(f"Getting array for {bounds_str}")
zarr_chunk_array = zarr_group[var_name][:, lat0:lat1, lon0:lon1]

# ### For 9x4000x4000 zarr
# for year_idx, year in enumerate(cn.interval_end_years_annual):
#     zarr_chunk_array_year = zarr_chunk_array[year_idx]
#
#     # The dataset pattern being analyzed, with year and units added
#     pattern_with_units = zu.add_units_year_to_pattern(var_name, year)
#
#     # print(f"Calculating stats for {bounds_str}")
#     zarr_stats_raw_year = uu.calculate_stats(zarr_chunk_array_year, pattern_with_units, bounds_str, tile_id,'zarr_stats')
#     # print(zarr_stats_raw_year)
#
#     zarr_stats_raw_all_years.append(zarr_stats_raw_year)
#
# # Returns the chunk stats from the zarr as a list of dictionaries, with each element being one chunk
# print(zarr_stats_raw_all_years)


### For 1x10000x10000 zarr
for year in cn.interval_end_years_annual:
    # The dataset pattern being analyzed, with year and units added
    pattern_with_units = zu.add_units_year_to_pattern(var_name, year)

    # print(f"Calculating stats for {bounds_str}")
    zarr_stats_raw_year = uu.calculate_stats(zarr_chunk_array, pattern_with_units, bounds_str, tile_id,'zarr_stats')
    # print(zarr_stats_raw_year)

    zarr_stats_raw_all_years.append(zarr_stats_raw_year)

# Returns the chunk stats from the zarr as a list of dictionaries, with each element being one chunk
print(zarr_stats_raw_all_years)