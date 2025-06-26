"""
Run from src/LULUCF

Local test (Dask part does not work):
python -m scripts.core_model.0_calculate_LULUCF_fluxes -bb 10 49.75 10.25 50 -cs 0.25 --no_upload -yr 2000 2023 --run_date YYYYMMDD

Coiled small tests:
python -m scripts.utilities.create_cluster -n 1 -t 1 -m 16 -cn LULUCF_model
python -m scripts.core_model.0_calculate_LULUCF_fluxes -cn LULUCF_model -bb 10 49.75 10.25 50 -cs 0.25 -yr 2000 2023 --run_date YYYYMMDD
python -m scripts.core_model.0_calculate_LULUCF_fluxes -cn LULUCF_model -bb 115.25 -3.75 115.5 -3.5 -cs 0.25 --no_upload -yr 2000 2023 --run_date YYYYMMDD
python -m scripts.core_model.0_calculate_LULUCF_fluxes -cn LULUCF_model -bb 10 49 11 50 -cs 1 --no_upload -yr 2000 2023 --run_date YYYYMMDD

Coiled large shapefile test:
python -m scripts.utilities.create_cluster -n 100 -t 1 -m 32 -cn LULUCF_model
python -m scripts.core_model.0_calculate_LULUCF_fluxes -cn LULUCF_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp -yr 2000 2023 --run_date YYYYMMDD

Full run:
python -m scripts.utilities.create_cluster -n 200 -t 1 -m 32 -cn LULUCF_model
python -m scripts.core_model.0_calculate_LULUCF_fluxes -cn LULUCF_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -yr 2000 2023 --run_date YYYYMMDD  --log_note "This is a full run."

To download all outputs locally:
python scripts/utilities/download_outputs_local.py v1 23_-4_24_-3

Using more than 1 thread/worker slows down processing a lot when there are more tasks than workers for the core LULUCF model,
which is the situation for large analyses, obviously.
https://app.asana.com/1/25496124013636/task/1206230383901961/comment/1210641504248464?focus=true
"""

import argparse
import concurrent.futures
import gc
import os
import psutil
import time
import sys
import numpy as np

from concurrent.futures import ThreadPoolExecutor

from dask.distributed import print
from numba import jit

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu
from ..utilities import log_utilities as lu
from ..utilities import numba_utilities as nu
from ..utilities import resize_cluster


# Function to calculate LULUCF fluxes and carbon densities
# Operates pixel by pixel, so uses numba (Python compiled to C++).
@jit(nopython=True)
def LULUCF_fluxes(in_dict_uint8, in_dict_int16, in_dict_int32, in_dict_float32,
                  primary_forest_RF_array, partial_disturbance_EF_array, mangrove_C_ratio_array,
                  model_start_year, end_year, interval_type, interval_year_diff_list, interval_length_list, interval_end_years, is_final):

    # Separate dictionaries for output numpy arrays of each datatype, named by output data type).
    # This is because a dictionary in a Numba function cannot have arrays with multiple data types, so each dictionary has to store only one data type,
    # just like inputs to the function.
    out_dict_uint8 = {}
    out_dict_uint16 = {}
    out_dict_uint32 = {}
    out_dict_float32 = {}

    # The maximum possible number of digits of the state_out in the current decision tree.
    # This needs to be changed if the decision tree is deepened.
    max_digits_state_out = 8

    # Carbon density arrays determined by the starting year of the model (Mg C/ha),
    # but the starting C densities have the same key in the dictionary regardless of the starting year
    agc_dens_block = in_dict_float32[cn.agc_raw_dens_pattern].astype('float32')
    bgc_dens_block = in_dict_float32[cn.bgc_raw_dens_pattern].astype('float32')
    deadwood_c_dens_block = in_dict_float32[cn.deadwood_c_raw_dens_pattern].astype('float32')
    litter_c_dens_block = in_dict_float32[cn.litter_c_raw_dens_pattern].astype('float32')

    # print(agc_dens_block.max())
    # print(bgc_dens_block.max())
    # print(deadwood_c_dens_block.max())
    # print(litter_c_dens_block.max())

    # Root:shoot (unitless)
    r_s_ratio_non_mang_block = in_dict_float32[cn.r_s_ratio_non_mang_pattern].astype('float32')

    # Natural forest regrowth curves (Mg C/ha/yr)
    natrl_forest_curve_0_5_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__0_5_years"].astype('float32')
    natrl_forest_curve_6_10_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__6_10_years"].astype('float32')
    natrl_forest_curve_11_15_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__11_15_years"].astype('float32')
    natrl_forest_curve_16_20_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__16_20_years"].astype('float32')
    natrl_forest_curve_21_40_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__21_40_years"].astype('float32')
    natrl_forest_curve_41_60_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__41_60_years"].astype('float32')
    natrl_forest_curve_61_80_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__61_80_years"].astype('float32')
    natrl_forest_curve_81_100_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__81_100_years"].astype('float32')
    natrl_forest_curve_21_100_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__21_100_years"].astype('float32')

    # Removal factor (Mg C/ha/yr)
    # Because this is used to store the RF from the previous interval,
    # it persists from one interval to the next. Therefore, it must be defined before the first iteration.
    # That way, removal factors can be over-written by those used in the most recent interval.
    agc_rf_pre_dist_out_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('float32')

    planted_forest_type_block = in_dict_uint8[cn.planted_forest_type_pattern]
    planted_forest_tree_crop_block = in_dict_uint8[cn.planted_forest_tree_crop_pattern]
    planted_forest_AGC_RF_block = in_dict_float32[cn.planted_forest_AGC_removal_factor_pattern]
    planted_forest_AGC_BGC_RF_block = in_dict_float32[cn.planted_forest_AGC_BGC_removal_factor_pattern]
    oil_palm_2000_extent_block = in_dict_uint8[cn.oil_palm_2000_extent_pattern]
    oil_palm_first_year_block = in_dict_int16[cn.oil_palm_first_year_pattern]

    ifl_primary_block = in_dict_uint8[cn.ifl_primary_pattern]
    drivers_block = in_dict_uint8[cn.drivers_pattern]
    continent_ecozone_block = in_dict_int16[cn.continent_ecozone_pattern]
    climate_zone_block = in_dict_uint8[cn.climate_zone_pattern]
    elevation_block = in_dict_int16[cn.elevation_pattern]
    climate_domain_block = in_dict_int16[cn.climate_domain_pattern]
    precipitation_block = in_dict_int32[cn.precipitation_pattern]

    forest_age_start_year_block = in_dict_uint8[cn.forest_age_start_year_pattern]

    # Sets a fallback value for continent_ecozone for the chunk in case any pixels fall outside the continent-ecozone boundary.
    # Also, excludes water codes (1022, 2022, 4022, 7022) from the fallback value.
    # Note that these continent-ecozone values need to explicitly be ignored further down when
    # the continent-ecozone pixel is applied.
    # This only excludes certain continent-ecozone values from use in the creation of the fallback value.
    # fallback_value is only used if the chunk doesn't have any continent_ecozone pixels in it at all.
    continent_ecozone_fallback = nu.fallback_conteco_climzone_value(continent_ecozone_block, 2020)


    # Sets a fallback value for climate zone for the chunk in case any pixels fall outside the climate zone boundary.
    # fallback_value is only used if the chunk doesn't have any climate_zone pixels in it at all.
    climate_zone_fallback = nu.fallback_conteco_climzone_value(climate_zone_block, 5)


    ## Test/intermediate outputs blocks

    # Stores the forest disturbance blocks for the entire model duration (added to progressively during each interval)
    forest_dist_blocks_all_intervals_so_far = []

    # Stores the last year that each pixel did not have tall vegetation composite land cover.
    # 0=Always tall vegetation so far. Other values represent the last year of non-tall vegetation.
    # This is assessed at the pixel level because numba wouldn't allow the needed logical operations on numpy arrays (chunks).
    # Tall vegetation is basd on the composite land cover maps, not the canopy height maps.
    most_recent_year_not_tall_veg_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('uint16')

    # Forest age for each output year of the model
    forest_age_annual_block = forest_age_start_year_block

    # Year in which forest loss occurs/is assigned during an interval (0 if no loss)
    ### TODO Never actually used. What did I intend to do with this?
    year_of_forest_loss_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('uint16')

    # Maximum height of vegetation since the last interval in which there was not forest
    max_height_since_last_time_not_tall_veg_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('uint8')

    # Tracks whether the height has already decreased more than the signif. height loss threshold compared to
    # maximum vegetation height since the last time the pixel was non-tall vegetation land cover.
    # This prevents a pixel from repeatedly (multiple intervals) being counted as having a height loss disturbance compared to
    # the maximum height; this can only be triggered once since the maximum height is attained.
    # This is used to determine if a forest->forest disturbance based on height loss relative to the max height should be reported.
    # 0=no significant height loss relative to the maximum vegetation height since last non-tall vegetation.
    # 1=height loss relative to the maximum vegetation height occurred in this interval.
    # 2=height loss relative to the maximum vegetation height occurred in a previous interval.
    first_time_sig_loss_from_max_height_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('uint8')

    # print("interval_end_years:", interval_end_years)

    # All years covered by the model, including the start year of the model
    years_so_far = [model_start_year]

    # Iterates through model intervals
    for i, interval_end_year in enumerate(interval_end_years):

        # print(f"Now at interval ending in {interval_end_year}:")

        # Length of the interval and difference between the start and end years (years)
        interval_length = interval_length_list[i]
        interval_year_diff = interval_year_diff_list[i]
        interval_start_year = interval_end_year - interval_length

        # Model years so far, including the model start year.
        # Eventually used to determine whether current height has decreased significantly from maximum height
        # since last non-tall veg year over multiple intervals (gradual height loss).
        years_so_far = years_so_far + [interval_end_year]
        # print(years_so_far)

        # Pre-fetches vegetation height data for this chunk and stores in a dictionary or list.
        # Eventually used to determine whether current height has decreased significantly from maximum height since last non-tall veg year over multiple intervals (gradual height loss).
        # Suggested by https://chatgpt.com/share/e/6724d803-aca4-800a-928c-11d76d38c0ec to work well with numba
        # and speed the code up. I was trying to get vegetation height so far in a variety of ways and it kept being slow.
        # This approach, in conjunction with some pixel-level operations below, seems to not slow down the code.
        vegetation_heights_so_far_block = [
            in_dict_uint8[f"{cn.vegetation_height_pattern}_{year}"]
            for year in years_so_far
        ]
        # print(vegetation_heights_so_far_block)

        # Land cover and vegetation height at the start and end of the interval,
        # e.g., 2005/2010 or 2016/2017
        LC_prev_block = in_dict_uint8[f"{cn.land_cover_pattern}_{interval_end_year - interval_length}"]
        LC_curr_block = in_dict_uint8[f"{cn.land_cover_pattern}_{interval_end_year}"]
        veg_h_prev_block = in_dict_uint8[f"{cn.vegetation_height_pattern}_{interval_end_year - interval_length}"]
        veg_h_curr_block = in_dict_uint8[f"{cn.vegetation_height_pattern}_{interval_end_year}"]

        # print(f"{cn.land_cover_pattern}_{interval_end_year - interval_length}:", LC_prev_block)
        # print(f"{cn.land_cover_pattern}_{interval_end_year}:", LC_curr_block)
        # print(f"{cn.vegetation_height_pattern}_{interval_end_year - interval_length}:", veg_h_prev_block)
        # print(f"{cn.vegetation_height_pattern}_{interval_end_year}:", veg_h_curr_block)

        # Stores the burned area blocks for the current interval (recreated/overwritten for each interval)
        burned_area_curr_interval_block_list = []

        # Creates a list of all the burned area arrays for the current interval.
        # It lists all burned area chunks in the interval
        # For example, for a 5-year interval 2001-2005, it will get burned area for 2001, 2002, 2003, 2004, and 2005.
        # For annual interval 2015-2016, it will get burned area for 2015 and 2016.
        # For 2016-2017, it will get burned area for 2016-2017.
        for year in range(interval_end_year - interval_length, interval_end_year+1):
            burned_area_for_year_in_interval = f"{cn.burned_area_final_pattern}_{year}"
            burned_area_curr_interval_block_list.append(in_dict_uint8[burned_area_for_year_in_interval])
            # print(year)
            # print(burned_area_for_year_in_interval)
            # print(burned_area_curr_interval_block_list)

        # print("burned_area_curr_interval_block_list")
        # print(burned_area_curr_interval_block_list)


        # Creates a list of all the annual Potapov forest disturbance rasters from 2001 to the end of the interval.
        # The values in the list are the disturbance year starting from 1, e.g., 2001=1, 2008=8, 2017=17.
        # It works by getting the annual disturbance chunks for the current interval and appending them to a list of
        # chunks from previous intervals.
        # Only does it for model run using 5-year intervals, as annual disturbance isn't needed for annual interval models
        if interval_length == 5:
            for year in range(interval_end_year-interval_year_diff, interval_end_year+1):

                # The name of the disturbance layer in the input dictionary
                annual_disturbance_for_year_in_interval = f"{cn.forest_disturbance_layer_name}_{year}"
                # print(annual_disturbance_for_year_in_interval)

                # Replaces the binary annual disturbance array with the year of disturbance (1, 2, 3...23)
                # print(in_dict_uint8[annual_disturbance_for_year_in_interval])
                year_disturb_array = in_dict_uint8[annual_disturbance_for_year_in_interval] * (year - model_start_year)
                # print(year_disturb_array)

                # Makes a list of disturbance arrays with the disturbance year.
                # uint8 is okay because the highest value should be 23 (not 2020).
                forest_dist_blocks_all_intervals_so_far.append(year_disturb_array.astype('uint8'))

            # print("forest_dist_blocks_all_intervals_so_far")
            # print(forest_dist_blocks_all_intervals_so_far)

        # Tracks whether a partial disturbance occurred in the last interval due to any cause (not including just fire,
        # which does not count as a partial disturbance in this model if height does not decrease significantly with it).
        # Rewritten for every interval. Hence, it is defined inside the interval loop.
        part_or_full_dist_in_prev_interval_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('uint8')

        # Tracks whether there was fire in the last interval
        burned_in_curr_interval_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('uint8')

        # Numpy arrays for outputs that don't depend on previous interval's values
        state_out_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('uint32')  # Land cover state at end of interval

        # Number of years of canopy growth.
        # First digit is pre-disturbance years of growth.
        # Second digit (if it exists) is post-disturbance years of growth
        gain_year_count_out_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('uint8')

        agc_gross_emis_out_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('float32')
        bgc_gross_emis_out_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('float32')
        deadwood_c_gross_emis_out_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('float32')
        litter_c_gross_emis_out_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('float32')

        ch4_gross_emis_out_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('float32')
        n2o_gross_emis_out_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('float32')

        agc_gross_removals_out_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('float32')
        bgc_gross_removals_out_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('float32')
        deadwood_c_gross_removals_out_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('float32')
        litter_c_gross_removals_out_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('float32')

        # Aboveground carbon emission factors
        agc_ef_out_block = np.zeros(in_dict_float32[cn.agc_raw_dens_pattern].shape).astype('float32')


        # Iterates through all pixels in the chunk
        for row in range(LC_curr_block.shape[0]):
            for col in range(LC_curr_block.shape[1]):

                ### Defines pixel/cell values

                LC_prev = LC_prev_block[row, col]
                LC_curr = LC_curr_block[row, col]
                veg_h_prev = veg_h_prev_block[row, col]
                veg_h_curr = veg_h_curr_block[row, col]

                # Secondary forest removal factors (Mg AGC/ha/yr)
                natrl_forest_curve_0_5_AGC_RF_cell = natrl_forest_curve_0_5_AGC_RF_block[row, col]
                natrl_forest_curve_6_10_AGC_RF_cell = natrl_forest_curve_6_10_AGC_RF_block[row, col]
                natrl_forest_curve_11_15_AGC_RF_cell = natrl_forest_curve_11_15_AGC_RF_block[row, col]
                natrl_forest_curve_16_20_AGC_RF_cell = natrl_forest_curve_16_20_AGC_RF_block[row, col]
                natrl_forest_curve_21_40_AGC_RF_cell = natrl_forest_curve_21_40_AGC_RF_block[row, col]
                natrl_forest_curve_41_60_AGC_RF_cell = natrl_forest_curve_41_60_AGC_RF_block[row, col]
                natrl_forest_curve_61_80_AGC_RF_cell = natrl_forest_curve_61_80_AGC_RF_block[row, col]
                natrl_forest_curve_81_100_AGC_RF_cell = natrl_forest_curve_81_100_AGC_RF_block[row, col]
                natrl_forest_curve_21_100_AGC_RF_cell = natrl_forest_curve_21_100_AGC_RF_block[row, col]

                planted_forest_type_cell = planted_forest_type_block[row, col]
                planted_forest_tree_crop_cell = planted_forest_tree_crop_block[row, col]
                planted_forest_AGC_RF_cell = planted_forest_AGC_RF_block[row, col]
                planted_forest_AGC_BGC_RF_cell = planted_forest_AGC_BGC_RF_block[row, col]
                planted_forest_BGC_RF_cell = planted_forest_AGC_BGC_RF_cell - planted_forest_AGC_RF_cell

                oil_palm_2000_extent_cell = oil_palm_2000_extent_block[row, col]
                oil_palm_first_year_cell = oil_palm_first_year_block[row, col]

                ifl_primary_cell = ifl_primary_block[row, col]
                drivers_cell = drivers_block[row, col]
                continent_ecozone_cell = continent_ecozone_block[row, col]
                climate_zone_cell = climate_zone_block[row, col]

                forest_age_annual_cell = forest_age_annual_block[row, col]

                # Applies the continent_ecozone fallback value when there isn't a value for the pixel (0)
                # or the value represents water (the XX22s).
                # We don't want pixels that are in a water ecozone to not be included in land
                # (and therefore not have a removal factor assigned); ecozone shouldn't delineate land.
                # These water codes are also excluded from the creation of continent_ecozone_fallback.
                if continent_ecozone_cell in [0, 1022, 2022, 4022, 7022]:
                    continent_ecozone_cell = continent_ecozone_fallback

                # Applies the climate_zone fallback value when there isn't a value for the pixel
                if climate_zone_cell == 0:
                    climate_zone_cell = climate_zone_fallback

                # Determines the removal factor for primary forests/IFL based on the continent-ecozone combination (Mg AGC/ha/yr)
                primary_forest_AGC_RF = nu.calc_primary_forest_RF(continent_ecozone_cell, primary_forest_RF_array)

                # Determines the emission factor for the cell's driver for partially disturbed forest based on the continent-ecozone combination (unit: fraction AGC lost)
                partial_disturbance_EF_for_driver = nu.calc_partial_disturbance_EFs(drivers_cell, continent_ecozone_cell, partial_disturbance_EF_array)

                elevation_cell = elevation_block[row, col]
                climate_domain_cell = climate_domain_block[row, col]
                precipitation_cell = precipitation_block[row, col]

                # Ratios of deadwood C:AGC and litter C:AGC (for deadwood C and litter C removal factors) for non-mangrove forests
                deadwood_c_ratio_non_mang, litter_c_ratio_non_mang = nu.calc_deadwood_litter_ratios(elevation_cell, climate_domain_cell, precipitation_cell)

                # BGC:AGC for non-mangrove forest
                r_s_ratio_non_mang = r_s_ratio_non_mang_block[row, col]

                # Replaces pixel without R:S (0) with the global non-mangrove R:S default
                if r_s_ratio_non_mang == 0:
                    r_s_ratio_non_mang = cn.default_r_s_non_mang

                #TODO @Mel Apply whatever rules you think are right to determine if mangrove C ratios should be calculated.
                # The ingestion of the ratios should be correct, but do check it.
                #Separate mangrove carbon pool ratios-- doesn't overwrite the non-mangrove ratios
                # if {MANGROVE_PRESENT in pixel for this interval}:
                #     r_s_ratio_mang = mangrove_C_ratio_array[np.where(mangrove_C_ratio_array[:, 0] == continent_ecozone_cell)][0, 1]
                #     deadwood_c_ratio_mang = mangrove_C_ratio_array[np.where(mangrove_C_ratio_array[:, 0] == continent_ecozone_cell)][0, 2]
                #     litter_c_ratio_mang = mangrove_C_ratio_array[np.where(mangrove_C_ratio_array[:, 0] == continent_ecozone_cell)][0, 3]

                # 5-year intervals: Burned area and Potapov annual disturbance raster stacks during the interval
                if interval_length == 5:
                    # Note: Stacking the burned area rasters using ndstack, stack, or flatten outside the pixel iteration did not work with numba.
                    # So just reading each raster from the list of rasters separately.
                    burned_area_t_4 = burned_area_curr_interval_block_list[-5][row, col]
                    burned_area_t_3 = burned_area_curr_interval_block_list[-4][row, col]
                    burned_area_t_2 = burned_area_curr_interval_block_list[-3][row, col]
                    burned_area_t_1 = burned_area_curr_interval_block_list[-2][row, col]
                    burned_area_t = burned_area_curr_interval_block_list[-1][row, col]
                    # Most recent year with burned area during the interval
                    most_recent_year_burned_during_interval = max([burned_area_t_4, burned_area_t_3, burned_area_t_2, burned_area_t_1, burned_area_t])
                    burned_in_curr_interval = (most_recent_year_burned_during_interval > 0)

                    # Note: Stacking the forest disturbance rasters using ndstack, stack, or flatten outside the pixel iteration did not work with numba.
                    # So just reading each raster from the list of rasters separately.
                    forest_dist_t_4 = forest_dist_blocks_all_intervals_so_far[-5][row, col]
                    forest_dist_t_3 = forest_dist_blocks_all_intervals_so_far[-4][row, col]
                    forest_dist_t_2 = forest_dist_blocks_all_intervals_so_far[-3][row, col]
                    forest_dist_t_1 = forest_dist_blocks_all_intervals_so_far[-2][row, col]
                    forest_dist_t = forest_dist_blocks_all_intervals_so_far[-1][row, col]
                    # Most recent year with forest disturbance during the interval
                    forest_dist_last = max([forest_dist_t_4, forest_dist_t_3, forest_dist_t_2, forest_dist_t_1, forest_dist_t])

                    # print(burned_in_curr_interval)
                    # print(forest_dist_last)

                # Annual intervals: Burned area for end year of interval only (year t).
                # Use of burned area for end year of interval as opposed to start year is by analogy with
                # Tree Cover Loss due to Fires (TCLF).
                # Annual Potapov disturbance rasters not relevant.
                elif interval_length == 1:
                    # burned_area_t_1 = burned_area_curr_interval_block_list[0][row, col]
                    burned_area_t = burned_area_curr_interval_block_list[-1][row, col]
                    # Most recent year with burned area during the interval (but for annual interval, the only year)
                    most_recent_year_burned_during_interval = max([burned_area_t])
                    burned_in_curr_interval = (most_recent_year_burned_during_interval > 0)
                    forest_dist_last = 0   # Annual Potapov forest disturbance raster not used for annual model

                    # print(burned_in_curr_interval)
                    # print(forest_dist_last)

                else:
                    raise ValueError("interval_length not valid: must be 1 or 5")

                # if forest_dist_last > 0:
                #     print(forest_dist_last)


                # Records the first year of forest disturbance in the pixel, to indicate whether disturbance was reported at all
                # in the record
                first_forest_dist_in_record = 0

                # Loops over annual forest disturbance rasters since 2001 to see if there was a disturbance.
                # Stops once a disturbance is detected because all that matters here is that there was a disturbance at some point (not the specific year).
                #TODO generalize to include disturbances of >= 5 m height reduction (would apply to annual model as well)?
                for forest_dist_year in forest_dist_blocks_all_intervals_so_far:
                    # Update the maximum value for this pixel
                    if forest_dist_year[row, col] > 0:
                        first_forest_dist_in_record = forest_dist_year[row, col]
                        break

                # if first_forest_dist_in_record > 0:
                #     print("disturbance", row, col, first_forest_dist_in_record)

                # Input carbon densities for the pools using the end of the previous interval (Mg C/ha)
                agc_dens_in = agc_dens_block[row, col]
                bgc_dens_in = bgc_dens_block[row, col]
                deadwood_c_dens_in = deadwood_c_dens_block[row, col]
                litter_c_dens_in = litter_c_dens_block[row, col]

                # Makes a list of carbon densities to save space in the decision tree below.
                # This list is input to flux calculation functions as one argument, rather than a separate argument
                # for each pool (Mg C/ha)
                c_dens_in = [agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in]

                # Forces starting AGC and BGC pools to 0 when there is tree cover gain.
                # This assumes no residual AGC and BGC when tree cover gain occurs.
                # It also assumes that there can be some deadwood and litter C left over.
                # Need to force the AGC and BGC to float32.
                c_dens_in_NT_T = [np.float32(0), np.float32(0), deadwood_c_dens_in, litter_c_dens_in]

                # One-time removal factor for gain of short vegetation (Mg C/ha) based on climate zone
                short_veg_AGC_RF, short_veg_BGC_RF = nu.calc_short_veg_removals(climate_zone_cell)


                ### Defines specific land cover classes

                # Tree presence at start and end of interval based on canopy heights (not LC composites)
                tree_prev = (veg_h_prev >= cn.tree_threshold)
                tree_curr = (veg_h_curr >= cn.tree_threshold)

                # Tree gain and loss based on canopy heights at start and end of interval (not based on LC composites)
                tree_gain = (not tree_prev and tree_curr)
                tree_loss = (tree_prev and not tree_curr)

                # Booleans of vegetation height classes for start (prev) and end (curr) of current interval based on LC composites
                short_veg_LC_prev, tall_veg_LC_prev = nu.classify_veg_height(LC_prev)
                short_veg_LC_curr, tall_veg_LC_curr = nu.classify_veg_height(LC_curr)

                SDPT_planted_trees = (planted_forest_type_cell > 0)  # All SDPT planted trees
                SDPT_oil_palm = (planted_forest_type_cell == cn.SDPT_oil_palm_code)  # Oil palm in SDPT planted trees
                oil_palm_pre_2000 = (oil_palm_2000_extent_cell == 1) # Oil palm that existed in the year 2000, according to that specific map/input

                # Establishes if the interval ends after Descals oil palm planting year. Rules are different for annual and 5-year intervals.
                # Second condition for each used to exclude NoData (0s) from first year of oil palm.
                # If the interval end year is after planting year, the interval is after Descals planting
                if interval_length == 1:
                    oil_palm_year_of_Descals_or_later = (interval_end_year >= oil_palm_first_year_cell) and (oil_palm_first_year_cell != 0)
                # If the interval start year is after planting year, the interval is after Descals planting.
                # This "all or nothing" approach simplifies things compared to saying that an interval can be partially before and partially after Descals planting year.
                elif interval_length == 5:
                    oil_palm_year_of_Descals_or_later = (interval_start_year >= oil_palm_first_year_cell) and (oil_palm_first_year_cell != 0)
                else:
                    raise ValueError("interval_length not valid: must be 1 or 5")



                # All planted trees in the given interval
                all_planted_trees = (SDPT_planted_trees or oil_palm_pre_2000 or oil_palm_year_of_Descals_or_later)

                # All oil palm in the given interval
                all_oil_palm = (SDPT_oil_palm or oil_palm_pre_2000 or oil_palm_year_of_Descals_or_later)

                # Flag for whether the Descals year of planting is:
                # Annual intervals: planting year one year after the the end of the interval
                # 5-year intervals: planting year during the 5-year interval
                # (to determine if forest loss should occur in the year before oil palm is detected)
                if interval_length == 1:  # For annual intervals
                    interval_before_converted_to_oil_palm = (interval_end_year == oil_palm_first_year_cell - interval_length)
                elif interval_length == 5:  # For 5-year intervals
                    interval_before_converted_to_oil_palm = ((interval_start_year < oil_palm_first_year_cell) and
                                                             (interval_end_year > oil_palm_first_year_cell))
                else:
                    raise ValueError("interval_length not valid: must be 1 or 5")

                ## Various pixel metrics that are used in the decision tree

                ## The most recent year of non-tall vegetation composite land cover in the cell before this interval
                most_recent_year_not_tall_veg = most_recent_year_not_tall_veg_block[row, col]

                # Checks whether to update whether the most recent year of non-tall vegetation land cover.
                # Returns the last year that was non-tall vegetation land cover.
                most_recent_year_not_tall_veg = nu.check_most_recent_year_not_tall_veg(LC_curr, LC_prev, most_recent_year_not_tall_veg, interval_end_year)

                # Calculates the maximum canopy height since the last time a pixel was classified as non-tall vegetation land cover.
                # This is eventually used to determine whether current height has decreased significantly from this maximum height over multiple intervals (gradual height loss).
                # Initializes a fixed-length array for vegetation_height_so_far_cell.
                # Suggested by https://chatgpt.com/share/e/6724d803-aca4-800a-928c-11d76d38c0ec to work well with numba.
                vegetation_height_so_far_cell = np.empty(len(years_so_far), dtype=np.uint8)

                # Populates the fixed-length array by accessing vegetation_heights_all_years
                # Used to determine whether current height has decreased significantly from this maximum height over multiple intervals (gradual height loss).
                # Suggested by https://chatgpt.com/share/e/6724d803-aca4-800a-928c-11d76d38c0ec to work well with numba
                # and speed the code up. I was trying to get vegetation height so far in a variety of ways and it kept being slow.
                # This approach, in conjunction with some the chunk-level preparation above, seems to not slow down the code.
                for i, year_data in enumerate(vegetation_heights_so_far_block):
                    vegetation_height_so_far_cell[i] = year_data[row, col]

                # Returns the maximum vegetation height since the last year of non-tall vegetation land cover
                max_height_since_last_time_not_tall_veg = nu.calc_max_height_since_last_time_not_tall_veg(most_recent_year_not_tall_veg, vegetation_height_so_far_cell, years_so_far)

                # Height change from maximum height since last time not tall veg land cover.
                # Need to recast to signed int8 from uint8 so that negative values (height gain) stay negative.
                height_change_max_curr = np.int8(max_height_since_last_time_not_tall_veg - veg_h_curr)

                # Is height loss significant in absolute change (m) compared to the maximum height since the last time
                # not tall vegetation land cover?
                sig_height_loss_max_current_abs = (height_change_max_curr >= cn.sig_height_loss_threshold_abs)

                # Tracks whether the height has already decreased more than the signif. height loss threshold compared to
                # maximum vegetation height since the last time the pixel was non-tall vegetation land cover.
                # This prevents a pixel from repeatedly (multiple intervals) being counted as having a disturbance compared to
                # the maximum height; this can only be triggered once since the maximum height is attained.
                # This is used to determine if a forest->forest disturbance based on height loss relative to the max height should be reported.
                # 0=no significant height loss relative to the maximum vegetation height since last non-tall vegetation.
                # 1=height loss relative to the maximum vegetation height occurred in this interval.
                # 2=height loss relative to the maximum vegetation height occurred in a previous interval.
                first_time_sig_loss_from_max_height = first_time_sig_loss_from_max_height_block[row,col]

                # Updates the tracker of whether this is the first time that significant height loss relative to
                # the height maximum has occurred to determine if a forest->forest disturbance should be reported.
                # If this is the first interval in which there has been a significant height decrease relative to the maximum
                # height since the last non-tall vegetation interval, the flag is changed to 1 to show it is being reported in this interval.
                # If this height decrease already occurred in a previous interval, the flag is changed to 2 to show it has
                # already been reported.
                if first_time_sig_loss_from_max_height == 1:
                    first_time_sig_loss_from_max_height = 2

                if sig_height_loss_max_current_abs:
                    if first_time_sig_loss_from_max_height == 0:
                        first_time_sig_loss_from_max_height = 1

                ## Height change during the interval. Need to recast to signed int8 from uint8 so that negative values (height gain) stay negative.
                height_change_prev_curr = np.int8(veg_h_prev - veg_h_curr)

                # Is height loss during the interval significant in absolute change (m)?
                sig_height_loss_prev_curr_abs = (height_change_prev_curr >= cn.sig_height_loss_threshold_abs)

                # Is height gain during the interval significant in absolute change (m)?
                # Significant height gain should not occur using annual intervals.
                # However, I'm not forcing sig_height_gain_prev_curr_abs = 0 when using annual intervals in order to try to catch strange cases.
                sig_height_gain_prev_curr_abs = (height_change_prev_curr <= cn.sig_height_gain_threshold_abs)

                # Whether tall vegetation was partially disturbed in the last interval
                part_or_full_dist_in_prev_interval = (forest_dist_last > 0) or (sig_height_loss_prev_curr_abs) or (first_time_sig_loss_from_max_height == 1)

                # Resets age to 0 if there was a partial disturbance in the last interval
                if part_or_full_dist_in_prev_interval:
                    forest_age_annual_cell = 0

                # Assigns pixel to "primary forest proxy".
                # Definition depends on model starting year.
                if model_start_year == 2000:  # For model starting in 2000, primary forest proxy uses primary forest/IFL map and age (>=100 years)
                    if (forest_age_annual_cell >= 100) or (ifl_primary_cell == 1):
                        primary_forest_proxy = 1
                    else:
                        primary_forest_proxy = 0
                elif model_start_year == 2015:  # For model starting in 2015, primary forest proxy is just age (>= 100 years)
                    if forest_age_annual_cell >= 100:
                        primary_forest_proxy = 1
                    else:
                        primary_forest_proxy = 0
                else:
                    raise ValueError("invalid start year: must be 2000 or 2015")

                # Aboveground removal factors based on stand age at the start of the interval (as opposed to the end) (Mg C/ha/yr)
                if 0 <= forest_age_annual_cell <= 5:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_0_5_AGC_RF_cell
                elif 6 <= forest_age_annual_cell <= 10:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_6_10_AGC_RF_cell
                elif 11 <= forest_age_annual_cell <= 15:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_11_15_AGC_RF_cell
                elif 16 <= forest_age_annual_cell <= 20:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_16_20_AGC_RF_cell
                elif 21 <= forest_age_annual_cell <= 40:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_21_40_AGC_RF_cell
                elif 41 <= forest_age_annual_cell <= 60:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_41_60_AGC_RF_cell
                elif 61 <= forest_age_annual_cell <= 80:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_61_80_AGC_RF_cell
                elif 81 <= forest_age_annual_cell < 100:  # < 100 because 100 is the value assigned to all forests 100 and above
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_81_100_AGC_RF_cell
                else:  # Use the primary forest/IFL RF for forest >100 years old
                    natrl_forest_age_dependent_agc_rf = primary_forest_AGC_RF


                # Gef for fire emissions for different gases for forests specifically (g respective gas/kg dry matter)
                Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest = nu.calc_Gef_forest(climate_domain_cell)

                # Cf for fire emissions for all gases for forests specifically (unitless)
                # Based on driver of loss, not the interval-end land cover
                Cf_forest = nu.calc_Cf_forest(climate_domain_cell, drivers_cell, primary_forest_proxy)


                ### Starting output pixel values

                # Starting decision tree node value
                # All C pool fluxes and densities are kept in Mg C (as opposed to Mg CO2) until they are saved to
                # the output dictionaries. This simplifies arithmetic.
                RF_AGC_final = 0  # The output AGC RF (pre-disturbance, if applicable) (Mg AGC/ha/yr)
                # Need to force arrays into float32 because numba is so particular about datatypes.
                # Initializes dummy output C gross emissions (Mg C/ha/interval->later converted to Mg CO2/ha/yr): AGC, BGC, deadwood C, litter C.
                c_gross_emis_out = np.array([0, 0, 0, 0]).astype('float32')
                # Initializes dummy output C gross removals (Mg C/ha/interval->later converted to Mg CO2/ha/yr): AGC, BGC, deadwood C, litter C.
                c_gross_removals_out = np.array([0, 0, 0, 0]).astype('float32')
                # Initializes dummy output non-CO2 fluxes (Mg CO2e/ha/interval->later converted to Mg CO2e/ha/yr): CH4, N2O
                non_co2_flux_out = np.array([0, 0]).astype('float32')

                node = 0
                gain_year_count = 0
                agc_ef_out_cell = 0  # AGC emission factor is reported as 0 unless specified otherwise


                #TODO @Mel include mangroves as the first if and change if tree_gain: to elif tree_gain:
                # That way, the mangrove decision tree gets priority.
                # Mangroves will have top-level node=1

                ### Tree gain
                if tree_gain:  # Non-tree converted to tree (2)    ##TODO: @Mel If mangrove branch at top, no exception needed here?
                    node = nu.accrete_node(node, 2)
                    if all_planted_trees:  # New planted trees (21)
                        node = nu.accrete_node(node, 1)
                        if all_oil_palm:  # New oil palm (incl. SDPT) (211)
                            state_out = nu.accrete_node(node, 1)
                            RF_AGC_final = cn.oil_palm_agc_rf
                            RF_BGC_final = cn.oil_palm_bgc_rf
                            (c_gross_emis_out, c_gross_removals_out, c_dens_out, gain_year_count, forest_age_annual_cell) = (
                                nu.calc_NT_T(interval_length, RF_AGC_final, RF_BGC_final, c_dens_in_NT_T, deadwood_c_ratio=0, litter_c_ratio=0))
                        else: # New non-oil palm planted trees (212)
                            state_out = nu.accrete_node(node, 2)
                            RF_AGC_final = planted_forest_AGC_RF_cell
                            RF_BGC_final = planted_forest_BGC_RF_cell
                            (c_gross_emis_out, c_gross_removals_out, c_dens_out, gain_year_count, forest_age_annual_cell) = (
                                nu.calc_NT_T(interval_length, RF_AGC_final, RF_BGC_final, c_dens_in_NT_T, deadwood_c_ratio=0, litter_c_ratio=0))
                    else:  # New non-planted trees (22)
                        node = nu.accrete_node(node, 2)
                        if tall_veg_LC_curr:  # New terrestrial natural forest (221)
                            state_out = nu.accrete_node(node, 1)
                            RF_AGC_final = natrl_forest_curve_0_5_AGC_RF_cell   # Forces new forest to use the first interval of the age curve
                            RF_BGC_final = RF_AGC_final * r_s_ratio_non_mang
                            (c_gross_emis_out, c_gross_removals_out, c_dens_out, gain_year_count, forest_age_annual_cell) = (
                                nu.calc_NT_T(interval_length, RF_AGC_final, RF_BGC_final, c_dens_in_NT_T, deadwood_c_ratio=deadwood_c_ratio_non_mang, litter_c_ratio=litter_c_ratio_non_mang))
                        else:  # New trees outside forests (222)
                            state_out = nu.accrete_node(node, 2)
                            RF_AGC_final = cn.trees_outside_forests_agc_rf_max
                            RF_BGC_final = RF_AGC_final * r_s_ratio_non_mang
                            (c_gross_emis_out, c_gross_removals_out, c_dens_out, gain_year_count, forest_age_annual_cell) = (
                                nu.calc_NT_T(interval_length, RF_AGC_final, RF_BGC_final, c_dens_in_NT_T, deadwood_c_ratio=0, litter_c_ratio=0))

                ### Tree loss
                elif tree_loss:  # Tree converted to non-tree (3)    ##TODO: @Mel If mangrove branch at top, no exception needed here?
                    node = nu.accrete_node(node, 3)
                    if all_planted_trees:  # Full loss of planted trees (31)
                        node = nu.accrete_node(node, 1)
                        if all_oil_palm:  # Full loss of oil palm (incl. SDPT) (311->3111/3112)
                            node = nu.accrete_node(node, 1)
                            agc_rf_in = cn.oil_palm_agc_rf  # 5-year intervals only
                            bgc_rf_in = cn.oil_palm_bgc_rf  # 5-year intervals only
                            c_pools_EF_fire_CO2 = cn.agc_emissions_only
                            c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                            c_pools_EF_no_fire = cn.all_non_soil_pools
                            rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                            (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                             RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_NT(
                                node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                deadwood_c_ratio=0, litter_c_ratio=0)
                        else:  # Full loss of non-oil palm planted trees (312)
                            node = nu.accrete_node(node, 2)
                            if LC_curr == cn.cropland:  # Plantation harvested as cropland (3121->31211/31212)
                                node = nu.accrete_node(node, 1)
                                agc_rf_in = planted_forest_AGC_RF_cell  # 5-year intervals only
                                bgc_rf_in = planted_forest_BGC_RF_cell  # 5-year intervals only
                                c_pools_EF_fire_CO2 = cn.biomass_emissions_only
                                c_pools_EF_fire_non_CO2 = cn.all_non_soil_pools
                                c_pools_EF_no_fire = cn.all_non_soil_pools
                                rf_post_dist = np.array([cn.cropland_rf, 0, 0, 0]).astype('float32')
                                (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                 RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_NT(
                                    node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                    c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio=0, litter_c_ratio=0)
                            elif short_veg_LC_curr:  # Plantation harvested as short vegetation (3122->31221/31222)
                                node = nu.accrete_node(node, 2)
                                agc_rf_in = planted_forest_AGC_RF_cell  # 5-year intervals only
                                bgc_rf_in = planted_forest_BGC_RF_cell  # 5-year intervals only
                                c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                c_pools_EF_fire_non_CO2 = cn.all_but_bgc_emissions
                                c_pools_EF_no_fire = cn.biomass_emissions_only
                                rf_post_dist = np.array([short_veg_AGC_RF, short_veg_BGC_RF, 0, 0]).astype('float32')
                                (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                 RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_NT(
                                    node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                    c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio=0, litter_c_ratio=0)
                            elif LC_curr == cn.builtup:  # Plantation converted to settlement (3123->31231/31232)
                                node = nu.accrete_node(node, 4)
                                agc_rf_in = planted_forest_AGC_RF_cell  # 5-year intervals only
                                bgc_rf_in = planted_forest_BGC_RF_cell  # 5-year intervals only
                                c_pools_EF_fire_CO2 = cn.biomass_emissions_only
                                c_pools_EF_fire_non_CO2 = cn.all_non_soil_pools
                                c_pools_EF_no_fire = cn.all_non_soil_pools
                                rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                 RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_NT(
                                    node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                    c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio=0, litter_c_ratio=0)
                            else:  # Plantation converted to anything else (3124->31241/31242)
                                node = nu.accrete_node(node, 5)
                                agc_rf_in = planted_forest_AGC_RF_cell  # 5-year intervals only
                                bgc_rf_in = planted_forest_BGC_RF_cell  # 5-year intervals only
                                c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                c_pools_EF_fire_non_CO2 = cn.all_but_bgc_emissions
                                c_pools_EF_no_fire = cn.biomass_emissions_only
                                rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                 RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_NT(
                                    node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                    c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio=0, litter_c_ratio=0)
                    else:  # Full loss of non-planted trees (32)
                        node = nu.accrete_node(node, 2)
                        if tall_veg_LC_prev:  # Full loss of natural forest (321)
                            node = nu.accrete_node(node, 1)
                            if LC_curr == cn.cropland:  # Natural forest converted to cropland (3211->32111/32112)
                                node = nu.accrete_node(node, 1)
                                agc_rf_in = natrl_forest_age_dependent_agc_rf  # 5-year intervals only
                                bgc_rf_in = agc_rf_in * r_s_ratio_non_mang     # 5-year intervals only
                                c_pools_EF_fire_CO2 = cn.all_non_soil_pools
                                c_pools_EF_fire_non_CO2 = cn.all_non_soil_pools
                                c_pools_EF_no_fire = cn.all_non_soil_pools
                                rf_post_dist = np.array([cn.cropland_rf, 0, 0, 0]).astype('float32')
                                (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                 RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_NT(
                                    node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                    c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio_non_mang, litter_c_ratio_non_mang)
                            elif short_veg_LC_curr:  # Natural forest converted to short vegetation (3212)
                                node = nu.accrete_node(node, 2)
                                if drivers_cell in cn.drivers_non_soil_C: # Natural forest converted to short vegetation with disturbance that emits all non-soil C pools (32121->321211/321212)
                                    node = nu.accrete_node(node, 1)
                                    agc_rf_in = natrl_forest_age_dependent_agc_rf  # 5-year intervals only
                                    bgc_rf_in = agc_rf_in * r_s_ratio_non_mang     # 5-year intervals only
                                    c_pools_EF_fire_CO2 = cn.all_non_soil_pools
                                    c_pools_EF_fire_non_CO2 = cn.all_non_soil_pools
                                    c_pools_EF_no_fire = cn.all_non_soil_pools
                                    rf_post_dist = np.array([short_veg_AGC_RF, short_veg_BGC_RF, 0, 0]).astype('float32')
                                    (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                     RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_NT(
                                        node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                        c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio_non_mang, litter_c_ratio_non_mang)
                                else:  # Natural forest converted to short vegetation with disturbance that emits biomass C pools only (32122->321221/321222)
                                    node = nu.accrete_node(node, 2)
                                    agc_rf_in = natrl_forest_age_dependent_agc_rf  # 5-year intervals only
                                    bgc_rf_in = agc_rf_in * r_s_ratio_non_mang     # 5-year intervals only
                                    c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                    c_pools_EF_fire_non_CO2 = cn.all_but_bgc_emissions
                                    c_pools_EF_no_fire = cn.biomass_emissions_only
                                    rf_post_dist = np.array([short_veg_AGC_RF, short_veg_BGC_RF, 0, 0]).astype('float32')
                                    (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                     RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_NT(
                                        node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                        c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio_non_mang, litter_c_ratio_non_mang)
                            elif LC_curr == cn.builtup:  # Natural forest converted to settlement (3213->32131/32132)
                                node = nu.accrete_node(node, 3)
                                agc_rf_in = natrl_forest_age_dependent_agc_rf  # 5-year intervals only
                                bgc_rf_in = agc_rf_in * r_s_ratio_non_mang     # 5-year intervals only
                                c_pools_EF_fire_CO2 = cn.all_non_soil_pools
                                c_pools_EF_fire_non_CO2 = cn.all_non_soil_pools
                                c_pools_EF_no_fire = cn.all_non_soil_pools
                                rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                 RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_NT(
                                    node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                    c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio_non_mang, litter_c_ratio_non_mang)
                            else:  # Natural forest converted to anything else (wetland/open water/ice, etc.) (3214->32141/32142)
                                node = nu.accrete_node(node, 4)
                                agc_rf_in = natrl_forest_age_dependent_agc_rf  # 5-year intervals only
                                bgc_rf_in = agc_rf_in * r_s_ratio_non_mang     # 5-year intervals only
                                c_pools_EF_fire_CO2 = np.array([0, 0, 0, 0]).astype('float32')   # This particular node can't have fire emissions
                                c_pools_EF_fire_non_CO2 = np.array([0, 0, 0, 0]).astype('float32') # This particular node can't have fire emissions
                                c_pools_EF_no_fire = cn.biomass_emissions_only
                                burned_in_curr_interval = 0  # This particular node can't have fire emissions, so this is forced to 0
                                rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                 RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_NT(
                                    node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                    c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio_non_mang, litter_c_ratio_non_mang)
                        else:  # Full loss of trees outside forests (322)  (slightly compressed variable assignments compared to elsewhere)
                            node = nu.accrete_node(node, 2)
                            if LC_curr == cn.cropland:  # Full loss of trees outside forests converted to cropland (3221->32211/32212)
                                node = nu.accrete_node(node, 1)
                                agc_rf_in = cn.trees_outside_forests_agc_rf_max  # 5-year intervals only
                                bgc_rf_in = agc_rf_in * r_s_ratio_non_mang  # 5-year intervals only
                                c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                c_pools_EF_no_fire = cn.biomass_emissions_only
                                rf_post_dist = np.array([cn.cropland_rf, 0, 0, 0]).astype('float32')
                                (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                 RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_NT(
                                    node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                    c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio=0, litter_c_ratio=0)
                            elif LC_curr == short_veg_LC_curr:  # Full loss of trees outside forests converted to short vegetation (3222->32221/32222)
                                node = nu.accrete_node(node, 2)
                                agc_rf_in = cn.trees_outside_forests_agc_rf_max  # 5-year intervals only
                                bgc_rf_in = agc_rf_in * r_s_ratio_non_mang       # 5-year intervals only
                                c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                c_pools_EF_no_fire = cn.biomass_emissions_only
                                rf_post_dist = np.array([short_veg_AGC_RF, short_veg_BGC_RF, 0, 0]).astype('float32')
                                (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                 RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_NT(
                                    node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                    c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio=0, litter_c_ratio=0)
                            elif LC_curr == cn.builtup:  # Full loss of trees outside forests converted to settlement (3223->32231/32232)
                                node = nu.accrete_node(node, 3)
                                agc_rf_in = cn.trees_outside_forests_agc_rf_max  # 5-year intervals only
                                bgc_rf_in = agc_rf_in * r_s_ratio_non_mang       # 5-year intervals only
                                c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                c_pools_EF_no_fire = cn.biomass_emissions_only
                                rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                 RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_NT(
                                    node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                    c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio=0, litter_c_ratio=0)
                            else:  # Full loss of trees outside forests converted to anything else (3224->32241/32242)
                                node = nu.accrete_node(node, 4)
                                agc_rf_in = cn.trees_outside_forests_agc_rf_max  # 5-year intervals only
                                bgc_rf_in = agc_rf_in * r_s_ratio_non_mang       # 5-year intervals only
                                c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                c_pools_EF_no_fire = cn.biomass_emissions_only
                                rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                 RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_NT(
                                    node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                    c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio=0, litter_c_ratio=0)

                ### Trees remaining trees
                elif (tree_prev) and (tree_curr):  # Trees remaining trees (4)    ##TODO: @Mel If mangrove branch at top, no exception needed here?
                    node = nu.accrete_node(node, 4)
                    if (not all_planted_trees) and interval_before_converted_to_oil_palm: # Non-planted trees with oil palm planted in the next interval (41->411/412)
                        node = nu.accrete_node(node, 1)
                        agc_rf_in = natrl_forest_age_dependent_agc_rf
                        bgc_rf_in = agc_rf_in * r_s_ratio_non_mang
                        c_pools_EF_fire_CO2 = cn.all_non_soil_pools
                        c_pools_EF_fire_non_CO2 = cn.all_non_soil_pools
                        c_pools_EF_no_fire = cn.all_non_soil_pools
                        # If annual interval, no oil palm removals in the interval of loss
                        if interval_length == 1:
                            rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                        # If 5-year interval, there is one year of oil palm removals in the interval of loss, regardless of the year.
                        # This is for simplicity: the post-conversion oil palm RF can be applied one time, like cropland or short veg RFs.
                        elif interval_length == 5:
                            rf_post_dist = np.array([cn.oil_palm_agc_rf, cn.oil_palm_agc_rf, 0, 0]).astype('float32')
                        else:
                            raise ValueError("interval_length not valid: must be 1 or 5")
                        (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                         RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_NT(
                            node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in,
                            c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                            c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                            rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest,
                            Gef_n2o_forest, deadwood_c_ratio_non_mang, litter_c_ratio_non_mang)
                    else:  # Trees remaining trees- no conversion to oil palm (42)
                        node = nu.accrete_node(node, 2)
                        if part_or_full_dist_in_prev_interval:  # Trees partially disturbed in the last interval (421)
                            node = nu.accrete_node(node, 1)
                            if all_planted_trees:   # Planted trees partially disturbed in the last interval (4211)
                                node = nu.accrete_node(node, 1)
                                if sig_height_gain_prev_curr_abs:  # Oil palm/planted trees partially disturbed in the last interval with signif. height increase after (42111)
                                    # NOTE: This should only occur with 5-year interval data, not annual data.  #TODO Confirm with 5-year data
                                    node = nu.accrete_node(node, 1)
                                    # Calculation function only uses the RFs for 5-year intervals but assigning them regardless of interval type for consistency
                                    if all_oil_palm:  # Oil palm partially disturbed in the last interval with signif. height increase after (421111->4211111/4211112)
                                        node = nu.accrete_node(node, 1)
                                        agc_rf_in = cn.oil_palm_agc_rf
                                        bgc_rf_in = cn.oil_palm_bgc_rf
                                        c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                        c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                        c_pools_EF_no_fire = np.array([partial_disturbance_EF_for_driver, partial_disturbance_EF_for_driver, 0, 0]).astype('float32')
                                        rf_post_dist = np.array([agc_rf_in, bgc_rf_in, 0, 0]).astype('float32')
                                        (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                         RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_T_non_stand_disturbs(
                                            node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in,
                                            c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                            c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                            rf_post_dist, most_recent_year_not_tall_veg,
                                            Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest, deadwood_c_ratio=0, litter_c_ratio=0)
                                    else: # Planted trees partially disturbed in the last interval with signif. height increase after (421112->4211121/4211122)
                                        node = nu.accrete_node(node, 2)
                                        agc_rf_in = planted_forest_AGC_RF_cell
                                        bgc_rf_in = planted_forest_BGC_RF_cell
                                        c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                        c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                        c_pools_EF_no_fire = np.array([partial_disturbance_EF_for_driver, partial_disturbance_EF_for_driver, 0, 0]).astype('float32')
                                        rf_post_dist = np.array([agc_rf_in, bgc_rf_in, 0, 0]).astype('float32')
                                        (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                         RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_T_non_stand_disturbs(
                                            node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in,
                                            c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                            c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                            rf_post_dist, most_recent_year_not_tall_veg,
                                            Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest, deadwood_c_ratio=0, litter_c_ratio=0)
                                else:  # Oil palm/planted trees partially disturbed in the last interval without signif. height increase after (42112)
                                    # NOTE: All annual interval data is expected to use this branch.
                                    node = nu.accrete_node(node, 2)
                                    # Calculation function only uses the RFs for 5-year intervals but assigning them regardless of interval type for consistency
                                    if all_oil_palm: # Oil palm partially disturbed in the last interval without signif. height increase after (421121->4211211/4211212)
                                        node = nu.accrete_node(node, 1)
                                        agc_rf_in = cn.oil_palm_agc_rf
                                        bgc_rf_in = cn.oil_palm_bgc_rf
                                        c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                        c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                        c_pools_EF_no_fire = np.array([partial_disturbance_EF_for_driver, partial_disturbance_EF_for_driver, 0, 0]).astype('float32')
                                        rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                        (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                         RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_T_non_stand_disturbs(
                                            node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in,
                                            c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                            c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                            rf_post_dist, most_recent_year_not_tall_veg,
                                            Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest, deadwood_c_ratio=0, litter_c_ratio=0)
                                    else: # Planted trees partially disturbed in the last interval without signif. height increase after (421122->4211221/4211222)
                                        node = nu.accrete_node(node, 2)
                                        agc_rf_in = planted_forest_AGC_RF_cell
                                        bgc_rf_in = planted_forest_BGC_RF_cell
                                        c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                        c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                        c_pools_EF_no_fire = np.array([partial_disturbance_EF_for_driver, partial_disturbance_EF_for_driver, 0, 0]).astype('float32')
                                        rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                        (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                         RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_T_non_stand_disturbs(
                                            node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in,
                                            c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                            c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                            rf_post_dist, most_recent_year_not_tall_veg,
                                            Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest, deadwood_c_ratio=0, litter_c_ratio=0)
                            else:  # Non-planted trees partially disturbed in the last interval (4212)
                                node = nu.accrete_node(node, 2)
                                if tall_veg_LC_curr:  # Forest partially disturbed in the last interval (42121)
                                    node = nu.accrete_node(node, 1)
                                    if sig_height_gain_prev_curr_abs:  # Forest partially disturbed in the last interval with signif. height increase after (421211->4212111/4212112)
                                        # NOTE: This should only occur with 5-year interval data, not annual data.
                                        node = nu.accrete_node(node, 1)
                                        # Calculation function only uses the RFs for 5-year intervals but assigning them regardless of interval type for consistency
                                        agc_rf_in = natrl_forest_age_dependent_agc_rf
                                        bgc_rf_in = agc_rf_in * r_s_ratio_non_mang
                                        c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                        c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                        c_pools_EF_no_fire = np.array([partial_disturbance_EF_for_driver, partial_disturbance_EF_for_driver, 0, 0]).astype('float32')
                                        agc_rf_post = natrl_forest_curve_0_5_AGC_RF_cell # Post-dist RF is 0-5 year secondary forest
                                        bgc_rf_post = agc_rf_post * r_s_ratio_non_mang
                                        rf_post_dist = np.array([agc_rf_post, bgc_rf_post, 0, 0]).astype('float32')
                                        (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                         RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_T_non_stand_disturbs(
                                            node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                            c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                            rf_post_dist, most_recent_year_not_tall_veg,
                                            Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                            deadwood_c_ratio=deadwood_c_ratio_non_mang, litter_c_ratio=litter_c_ratio_non_mang)
                                    else:  # Forest partially disturbed in the last interval without signif. height increase after (421212->4212121/4212122)
                                        # NOTE: All annual interval data is expected to use this branch.
                                        node = nu.accrete_node(node, 2)
                                        # Calculation function only uses the RFs for 5-year intervals but assigning them regardless of interval type for consistency
                                        agc_rf_in = natrl_forest_age_dependent_agc_rf
                                        bgc_rf_in = agc_rf_in * r_s_ratio_non_mang
                                        rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')  # No post-disturbance RFs or removals
                                        c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                        c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                        c_pools_EF_no_fire = np.array([partial_disturbance_EF_for_driver, partial_disturbance_EF_for_driver, 0, 0]).astype('float32')
                                        (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                         RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_T_non_stand_disturbs(
                                            node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                            c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                            rf_post_dist, most_recent_year_not_tall_veg,
                                            Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                            deadwood_c_ratio=deadwood_c_ratio_non_mang, litter_c_ratio=litter_c_ratio_non_mang)
                                else:  # Trees outside forests partially disturbed in the last interval (42122)
                                    node = nu.accrete_node(node, 2)
                                    if sig_height_gain_prev_curr_abs:  # Trees outside forests partially disturbed in the last interval with signif. height increase after (421221->4212211/4212212)
                                        # NOTE: This should only occur with 5-year interval data, not annual data.
                                        node = nu.accrete_node(node, 1)
                                        # Calculation function only uses the RFs for 5-year intervals but assigning them regardless of interval type for consistency
                                        agc_rf_in = cn.trees_outside_forests_agc_rf_max
                                        bgc_rf_in = agc_rf_in * r_s_ratio_non_mang
                                        c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                        c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                        c_pools_EF_no_fire = np.array([partial_disturbance_EF_for_driver, partial_disturbance_EF_for_driver, 0, 0]).astype('float32')
                                        agc_rf_post = cn.trees_outside_forests_agc_rf_max
                                        bgc_rf_post = agc_rf_post * r_s_ratio_non_mang
                                        rf_post_dist = np.array([agc_rf_post, bgc_rf_post, 0, 0]).astype('float32')
                                        (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                         RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_T_non_stand_disturbs(
                                            node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                            c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                            rf_post_dist, most_recent_year_not_tall_veg,
                                            Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                            deadwood_c_ratio=0, litter_c_ratio=0)
                                    else:  # Trees outside forests partially disturbed in the last interval without signif. height increase after (421222->4212221/4212222)
                                        # NOTE: All annual interval data is expected to use this branch.
                                        node = nu.accrete_node(node, 2)
                                        # Calculation function only uses the RFs for 5-year intervals but assigning them regardless of interval type for consistency
                                        agc_rf_in = cn.trees_outside_forests_agc_rf_max
                                        bgc_rf_in = agc_rf_in * r_s_ratio_non_mang
                                        rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')  # No post-disturbance RFs or removals
                                        c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                        c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                        c_pools_EF_no_fire = np.array([partial_disturbance_EF_for_driver, partial_disturbance_EF_for_driver, 0, 0]).astype('float32')
                                        (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                         RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_annual_cell) = nu.calc_T_T_non_stand_disturbs(
                                            node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                            c_pools_EF_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                            rf_post_dist, most_recent_year_not_tall_veg,
                                            Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                            deadwood_c_ratio=0, litter_c_ratio=0)
                        else:  # Trees not disturbed in the last interval (422)
                            node = nu.accrete_node(node, 2)
                            if all_planted_trees:  # Oil palm/planted trees not disturbed in the last interval (4221)
                                node = nu.accrete_node(node, 1)
                                # Calculation function only uses the RFs for 5-year intervals but assigning them regardless of interval type for consistency
                                if all_oil_palm:  # Oil palm not disturbed in the last interval (42211->422111/422112)
                                    node = nu.accrete_node(node, 1)
                                    RF_AGC_final = cn.oil_palm_agc_rf
                                    RF_BGC_final = cn.oil_palm_bgc_rf
                                    c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                    c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                    (state_out, c_gross_emis_out, c_gross_removals_out, agc_ef_out_cell,
                                     non_co2_flux_out, c_dens_out, gain_year_count, forest_age_annual_cell) = nu.calc_T_T_no_disturbs(
                                        node, interval_length, forest_age_annual_cell,
                                        most_recent_year_burned_during_interval,
                                        RF_AGC_final, RF_BGC_final, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                        interval_end_year, c_dens_in, most_recent_year_not_tall_veg,
                                        Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest, deadwood_c_ratio=0, litter_c_ratio=0)
                                else: # Planted trees not disturbed in the last interval (42212->422121/422122)
                                    node = nu.accrete_node(node, 2)
                                    RF_AGC_final = planted_forest_AGC_RF_cell
                                    RF_BGC_final = planted_forest_BGC_RF_cell
                                    c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                    c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                    (state_out, c_gross_emis_out, c_gross_removals_out, agc_ef_out_cell,
                                     non_co2_flux_out, c_dens_out, gain_year_count, forest_age_annual_cell) = nu.calc_T_T_no_disturbs(
                                        node, interval_length, forest_age_annual_cell,
                                        most_recent_year_burned_during_interval,
                                        RF_AGC_final, RF_BGC_final, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                        interval_end_year, c_dens_in, most_recent_year_not_tall_veg,
                                        Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest, deadwood_c_ratio=0, litter_c_ratio=0)
                            else:  # Non-planted trees not disturbed in last interval (4222)
                                node = nu.accrete_node(node, 2)
                                if tall_veg_LC_curr:  # Natural forest not disturbed in last interval (42221)
                                    node = nu.accrete_node(node, 1)
                                    if (most_recent_year_not_tall_veg > 0) or (first_forest_dist_in_record > 0):  # Young secondary natural forest (422211->4222111/4222112)
                                        node = nu.accrete_node(node, 1)
                                        RF_AGC_final = natrl_forest_age_dependent_agc_rf
                                        RF_BGC_final = RF_AGC_final * r_s_ratio_non_mang
                                        c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                        c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                        (state_out, c_gross_emis_out, c_gross_removals_out, agc_ef_out_cell,
                                         non_co2_flux_out, c_dens_out, gain_year_count, forest_age_annual_cell) = nu.calc_T_T_no_disturbs(
                                            node, interval_length, forest_age_annual_cell, most_recent_year_burned_during_interval,
                                            RF_AGC_final, RF_BGC_final, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                            interval_end_year, c_dens_in, most_recent_year_not_tall_veg,
                                            Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                            deadwood_c_ratio=deadwood_c_ratio_non_mang, litter_c_ratio=litter_c_ratio_non_mang)
                                    else:  # Natural forest undisturbed since model start (422212)
                                        node = nu.accrete_node(node, 2)
                                        if primary_forest_proxy:  # Primary forest (4222121->42221211/42221212)
                                            node = nu.accrete_node(node, 1)
                                            RF_AGC_final = primary_forest_AGC_RF
                                            RF_BGC_final = RF_AGC_final * r_s_ratio_non_mang
                                            c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                            c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                            (state_out, c_gross_emis_out, c_gross_removals_out, agc_ef_out_cell,
                                             non_co2_flux_out, c_dens_out, gain_year_count, forest_age_annual_cell) = nu.calc_T_T_no_disturbs(
                                                node, interval_length, forest_age_annual_cell, most_recent_year_burned_during_interval,
                                                RF_AGC_final, RF_BGC_final, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                                interval_end_year, c_dens_in, most_recent_year_not_tall_veg,
                                                Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                                deadwood_c_ratio=deadwood_c_ratio_non_mang, litter_c_ratio=litter_c_ratio_non_mang)
                                        else: # Old secondary forest (4222122->42221221/42221222)
                                            node = nu.accrete_node(node, 2)
                                            RF_AGC_final = natrl_forest_age_dependent_agc_rf
                                            RF_BGC_final = RF_AGC_final * r_s_ratio_non_mang
                                            c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                            c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                            (state_out, c_gross_emis_out, c_gross_removals_out, agc_ef_out_cell,
                                             non_co2_flux_out, c_dens_out, gain_year_count, forest_age_annual_cell) = nu.calc_T_T_no_disturbs(
                                                node, interval_length, forest_age_annual_cell, most_recent_year_burned_during_interval,
                                                RF_AGC_final, RF_BGC_final, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                                interval_end_year, c_dens_in, most_recent_year_not_tall_veg,
                                                Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                                deadwood_c_ratio=deadwood_c_ratio_non_mang, litter_c_ratio=litter_c_ratio_non_mang)
                                else:  # Trees outside forests not disturbed in the last interval (42222->422221/422222)
                                    node = nu.accrete_node(node, 2)
                                    RF_AGC_final = cn.trees_outside_forests_agc_rf_max
                                    RF_BGC_final = RF_AGC_final * r_s_ratio_non_mang
                                    c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                    c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                    (state_out, c_gross_emis_out, c_gross_removals_out, agc_ef_out_cell,
                                     non_co2_flux_out, c_dens_out, gain_year_count, forest_age_annual_cell) = nu.calc_T_T_no_disturbs(
                                        node, interval_length, forest_age_annual_cell, most_recent_year_burned_during_interval,
                                        RF_AGC_final, RF_BGC_final, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                        interval_end_year, c_dens_in, most_recent_year_not_tall_veg,
                                        Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest, deadwood_c_ratio=0, litter_c_ratio=0)

                ### Non-cropland/non-tree to cropland (without tall vegetation)
                elif (LC_prev != cn.cropland) and (LC_curr == cn.cropland): ##TODO: @Mel If mangrove branch at top, no exception needed here?
                    node = nu.accrete_node(node, cn.cropland_node)  # General cropland node code (5)
                    state_out = nu.accrete_node(node, 1)  # Cropland gain (51)
                    c_pools_EF_no_fire = cn.all_non_soil_pools  # Fire not considered in years of cropland gain
                    RF_AGC_final = cn.cropland_rf
                    agc_ef_out_cell = c_pools_EF_no_fire[0]  # Emission factor used for output geotif
                    rf_array = np.array([RF_AGC_final, 0, 0, 0]).astype('float32')
                    forest_age_annual_cell = 0  # Sets forest age to 0 because there's no forest
                    c_gross_emis_out, c_gross_removals_out, c_dens_out = nu.calc_NT_cropland_gain(c_pools_EF_no_fire, c_dens_in, rf_array)

                ### Cropland converted to non-cropland (without tall vegetation)
                elif (LC_prev == cn.cropland) and (LC_curr != cn.cropland): ##TODO: @Mel If mangrove branch at top, no exception needed here?
                    node = nu.accrete_node(node, cn.cropland_node)  # General cropland node code (5)
                    node = nu.accrete_node(node, 2)  # Cropland loss (52->521/522)
                    c_pools_EF_no_fire = cn.agc_emissions_only  # There should only be AGC in cropland anyway
                    c_dens_in = [cn.cropland_agc_dens, 0, 0 ,0]  # Forces input AGC to cropland default; other pools forced to 0, regardless of existing value.
                    agc_ef_out_cell = c_pools_EF_no_fire[0]  # Emission factor used for output geotif
                    forest_age_annual_cell = 0  # Sets forest age to 0 because there's no forest
                    (state_out, c_gross_emis_out, c_gross_removals_out,
                     c_dens_out, non_co2_flux_out) = nu.calc_cropland_non_cropland(node, c_dens_in, c_pools_EF_no_fire, most_recent_year_burned_during_interval)

                ### Cropland remaining cropland (without tall vegetation)
                # TODO Allow cropland to have multiple fire emissions instances during a 5-year interval
                elif (LC_prev == cn.cropland) and (LC_curr == cn.cropland): ##TODO: @Mel If mangrove branch at top, no exception needed here?
                    node = nu.accrete_node(node, cn.cropland_node)  # General cropland node code (5)
                    node = nu.accrete_node(node, 3)  # Cropland->cropland (53->531/532)
                    c_dens_in = [cn.cropland_agc_dens, 0, 0 ,0]  # Forces input AGC to cropland default; other pools forced to 0, regardless of existing value. Same values output.
                    forest_age_annual_cell = 0  # Sets forest age to 0 because there's no forest
                    (state_out, c_gross_emis_out, c_gross_removals_out,
                     c_dens_out, non_co2_flux_out) = nu.calc_cropland_cropland(node, c_dens_in, most_recent_year_burned_during_interval)

                ### Non-tree/cropland/short vegetation converted to short vegetation
                # TODO revisit constants used here. Never really resolved issues about starting carbon or what to do with residual carbon.
                # TODO Confirm that short veg default values are used correctly, i.e. starting C densities -> short veg
                # TODO include bare ground getting set to 0 AGC
                elif (not short_veg_LC_prev) and (short_veg_LC_curr): ##TODO: @Mel If mangrove branch at top, no exception needed here?
                    node = nu.accrete_node(node, cn.grassland_node)  # General short veg node code (6)
                    state_out = nu.accrete_node(node, 1)  # Short vegetation gain (61)
                    RF_AGC_final = short_veg_AGC_RF
                    RF_BGC_final = short_veg_BGC_RF
                    rf_array = np.array([RF_AGC_final, RF_BGC_final, 0, 0]).astype('float32')
                    forest_age_annual_cell = 0  # Sets forest age to 0 because there's no forest
                    c_gross_emis_out, c_gross_removals_out, c_dens_out = nu.calc_short_veg_gain(rf_array)

                ### Short vegetation converted to non-short vegetation, non-forest or non-cropland
                # TODO revisit this. Never really resolved issues about starting carbon or what to do with residual carbon.
                # TODO confirm that fire emissions are correct. Didn't check that.
                elif (short_veg_LC_prev) and (not short_veg_LC_curr): ##TODO: @Mel If mangrove branch at top, no exception needed here?
                    node = nu.accrete_node(node, cn.grassland_node)  # General short veg node code (6)
                    node = nu.accrete_node(node, 2)  # Short vegetation loss (62->621/622)
                    c_dens_in = [short_veg_AGC_RF, short_veg_BGC_RF, 0 ,0]  # Forces input carbon densities to short veg default; other pools forced to 0, regardless of existing value.
                    c_pools_EF_no_fire = cn.biomass_emissions_only
                    agc_ef_out_cell = c_pools_EF_no_fire[0]  # Emission factor used for output geotif
                    forest_age_annual_cell = 0  # Sets forest age to 0 because there's no forest
                    (state_out, c_gross_emis_out, c_gross_removals_out,
                     c_dens_out, non_co2_flux_out) = nu.calc_short_veg_loss(node, c_dens_in, c_pools_EF_no_fire, most_recent_year_burned_during_interval)

                ### Short vegetation remaining short vegetation
                # TODO revisit this. Never really resolved issues about starting carbon or what to do with residual carbon.
                # TODO confirm that fire emissions are correct. Didn't check that.
                # TODO Allow short veg to have multiple fire emissions instances during a 5-year interval
                elif short_veg_LC_prev and short_veg_LC_curr: ##TODO: @Mel If mangrove branch at top, no exception needed here?
                    node = nu.accrete_node(node, cn.grassland_node)  # General short veg node code (6)
                    node = nu.accrete_node(node, 3)  # Short vegetation->Short vegetation (63->631/632)
                    c_dens_in = [short_veg_AGC_RF, short_veg_BGC_RF, 0 ,0]  # Forces input carbon densities to short veg default; other pools forced to 0, regardless of existing value.
                    forest_age_annual_cell = 0  # Sets forest age to 0 because there's no forest
                    (state_out, c_gross_emis_out, c_gross_removals_out,
                     c_dens_out, non_co2_flux_out) = nu.calc_short_veg_short_veg(node, c_dens_in, most_recent_year_burned_during_interval)

                #TODO Do I need to add a non-veg state node (with carbon density of 0)?

                # When decision trees above do not apply
                else:

                    state_out = 2000000

                    # If no decision tree branches apply, forest age set to 0 years
                    forest_age_annual_cell = 0

                    # If no decision tree branches apply, carbon densities set to 0 Mg C/ha
                    c_dens_out = np.array([0, 0, 0, 0]).astype('float32')

                ### Populates the output arrays with the calculated fluxes and densities

                # Stops model if state_out is more digits than the expected maximum (currently 6).
                # This means that the tree is deeper than expected and max_digits_state_out needs to be increased.
                if state_out > (10 ** max_digits_state_out)-1:
                    raise ValueError("Maximum state_out is greater than the expected number of digits")

                # Converts the state to 6 digits (trailing 0s) for consistency across all nodes
                state_out = nu.pad_to_6_digits(state_out, max_digits_state_out)

                state_out_block[row, col] = state_out

                # Sets the RF from this interval as the RF from the previous interval for the next interval (Mg AGC/ha/yr)
                agc_rf_pre_dist_out_block[row, col] = RF_AGC_final

                # Gross emissions for each C pool for the interval before conversion to CO2 (Mg C/ha/interval)
                agc_gross_emis_out_block[row, col] = c_gross_emis_out[0]
                bgc_gross_emis_out_block[row, col] = c_gross_emis_out[1]
                deadwood_c_gross_emis_out_block[row, col] = c_gross_emis_out[2]
                litter_c_gross_emis_out_block[row, col] = c_gross_emis_out[3]

                # Non-CO2 emissions for the interval (Mg CO2e/ha/interval)
                ch4_gross_emis_out_block[row, col] = non_co2_flux_out[0]
                n2o_gross_emis_out_block[row, col] = non_co2_flux_out[1]

                # Gross removals for each C pool for the interval before conversion to CO2 (Mg C/ha/interval)
                agc_gross_removals_out_block[row, col] = c_gross_removals_out[0]
                bgc_gross_removals_out_block[row, col] = c_gross_removals_out[1]
                deadwood_c_gross_removals_out_block[row, col] = c_gross_removals_out[2]
                litter_c_gross_removals_out_block[row, col] = c_gross_removals_out[3]

                # C density for each C pool at the end of the interval (Mg C/ha)
                agc_dens_block[row, col] = c_dens_out[0]
                bgc_dens_block[row, col] = c_dens_out[1]
                deadwood_c_dens_block[row, col] = c_dens_out[2]
                litter_c_dens_block[row, col] = c_dens_out[3]

                # Test/intermediate outputs
                forest_age_annual_block[row, col] = forest_age_annual_cell
                gain_year_count_out_block[row, col] = gain_year_count
                most_recent_year_not_tall_veg_block[row, col] = most_recent_year_not_tall_veg
                max_height_since_last_time_not_tall_veg_block[row, col] = max_height_since_last_time_not_tall_veg
                first_time_sig_loss_from_max_height_block[row, col] = first_time_sig_loss_from_max_height
                part_or_full_dist_in_prev_interval_block[row, col] = part_or_full_dist_in_prev_interval
                burned_in_curr_interval_block[row, col] = burned_in_curr_interval
                agc_ef_out_block[row, col] = agc_ef_out_cell

        # os.quit()   # For testing the first interval

        ### End of one iteration calculations and outputs

        # Adds the output arrays to the dictionary with the appropriate data type
        # Outputs need .copy() so that previous intervals' arrays in dictionary aren't overwritten because arrays in dictionaries are mutable (courtesy of ChatGPT).
        # This applies even for the outputs that aren't reused in the next interval;
        # they will still get overwritten with the final interval's values, I believe.
        year_range = f"{interval_end_year - interval_year_diff}_{interval_end_year}"

        out_dict_uint32[f"{cn.land_state_pattern}_{year_range}"] = state_out_block.copy()

        out_dict_float32[f"{cn.agc_rf_pre_dist_pattern}{cn.flux_density_pixel_meaning}_{year_range}"] = agc_rf_pre_dist_out_block.copy()

        # Converts carbon pool fluxes from Mg C/ha/interval to Mg CO2/ha/yr.
        # Gross emissions are positive. Gross removals are negative.
        out_dict_float32[f"{cn.agc_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{year_range}"] = (agc_gross_emis_out_block * cn.C_to_CO2_numba / interval_length).copy()
        out_dict_float32[f"{cn.bgc_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{year_range}"] = (bgc_gross_emis_out_block * cn.C_to_CO2_numba / interval_length).copy()
        out_dict_float32[f"{cn.deadwood_c_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{year_range}"] = (deadwood_c_gross_emis_out_block * cn.C_to_CO2_numba / interval_length).copy()
        out_dict_float32[f"{cn.litter_c_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{year_range}"] = (litter_c_gross_emis_out_block * cn.C_to_CO2_numba / interval_length).copy()

        out_dict_float32[f"{cn.agc_gross_removals_pattern}{cn.flux_density_pixel_meaning}_{year_range}"] = (agc_gross_removals_out_block * cn.C_to_CO2_numba / interval_length).copy()
        out_dict_float32[f"{cn.bgc_gross_removals_pattern}{cn.flux_density_pixel_meaning}_{year_range}"] = (bgc_gross_removals_out_block * cn.C_to_CO2_numba / interval_length).copy()
        out_dict_float32[f"{cn.deadwood_c_gross_removals_pattern}{cn.flux_density_pixel_meaning}_{year_range}"] = (deadwood_c_gross_removals_out_block * cn.C_to_CO2_numba / interval_length).copy()
        out_dict_float32[f"{cn.litter_c_gross_removals_pattern}{cn.flux_density_pixel_meaning}_{year_range}"] = (litter_c_gross_removals_out_block * cn.C_to_CO2_numba / interval_length).copy()

        # Converts non-CO2 emissions from Mg CO2e/ha/interval to Mg CO2e/ha/yr. No conversion of Mg C/ha to Mg CO2 because these are already in Mg CO2e/ha.
        out_dict_float32[f"{cn.ch4_flux_pattern}{cn.flux_density_pixel_meaning}_{year_range}"] = (ch4_gross_emis_out_block / interval_length).copy()
        out_dict_float32[f"{cn.n2o_flux_pattern}{cn.flux_density_pixel_meaning}_{year_range}"] = (n2o_gross_emis_out_block / interval_length).copy()

        # Still Mg C/ha at the interval end year
        out_dict_float32[f"{cn.agc_raw_dens_pattern}{cn.C_density_pixel_meaning}_{interval_end_year}"] = agc_dens_block.copy()
        out_dict_float32[f"{cn.bgc_raw_dens_pattern}{cn.C_density_pixel_meaning}_{interval_end_year}"] = bgc_dens_block.copy()
        out_dict_float32[f"{cn.deadwood_c_raw_dens_pattern}{cn.C_density_pixel_meaning}_{interval_end_year}"] = deadwood_c_dens_block.copy()
        out_dict_float32[f"{cn.litter_c_raw_dens_pattern}{cn.C_density_pixel_meaning}_{interval_end_year}"] = litter_c_dens_block.copy()

        # Test/intermediate outputs only saved if not a large run
        if not is_final:
            out_dict_uint8[f"{cn.forest_age_output_pattern}_{interval_end_year}"] = forest_age_annual_block.copy()
            out_dict_uint8[f"{cn.gain_year_count_pattern}_{year_range}"] = gain_year_count_out_block.copy()
            out_dict_uint16[f"{cn.most_recent_year_not_tall_veg}_{model_start_year}_{interval_end_year}"] = most_recent_year_not_tall_veg_block.copy()    # Years represent from model start to current interval end
            out_dict_uint16[f"{cn.year_of_forest_loss}_{year_range}"] = year_of_forest_loss_block.copy()
            out_dict_uint8[f"{cn.max_height_since_last_time_not_tall_veg}_{year_range}"] = max_height_since_last_time_not_tall_veg_block.copy()
            out_dict_uint8[f"{cn.first_time_sig_loss_from_max_height}_{year_range}"] = first_time_sig_loss_from_max_height_block.copy()
            out_dict_uint8[f"{cn.part_or_full_dist_in_prev_interval}_{year_range}"] = part_or_full_dist_in_prev_interval_block.copy()
            out_dict_uint8[f"{cn.burned_in_curr_interval}_{year_range}"] = burned_in_curr_interval_block.copy()
            out_dict_float32[f"{cn.agc_emission_factor}_{year_range}"] = agc_ef_out_block.copy()

    return out_dict_uint8, out_dict_uint16, out_dict_uint32, out_dict_float32


# Downloads inputs, prepares data, calculates LULUCF stocks and fluxes, and uploads outputs to s3
def calculate_and_upload_LULUCF_fluxes(bounds, primary_forest_RF_array, partial_disturbance_EF_array, mangrove_C_ratio_array,
                                       download_dict_with_data_types, start_year, end_year, interval_type, interval_year_diff_list,
                                       interval_length_list, interval_end_years, is_final, no_upload, output_folders, stage):

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []

    process = psutil.Process(os.getpid())

    logger_worker = lu.setup_logging_worker()

    chunk_start_time = time.time()

    uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_final, logger_worker)

    bounds_str = uu.boundstr(bounds)  # String form of chunk bounds, from e.g., [8, -1, 9, 0] to 8_-1_9_0
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)

    ### Part 1: Downloads all inputs for chunk.
    ### No checks about whether the chunk has data because the way the chunk_list is constructed,
    ### every chunk is relevant and should be processed, so they don't need to be checked.

    # Replaces the placeholder tile_id in the download data dictionary from main with the tile_id for this chunk
    updated_download_dict = uu.replace_tile_id_in_dict(download_dict_with_data_types, tile_id)

    # If a particular tile doesn't exist for an input, an array of 0s of the correct size and datatype is returned instead.
    # Thus, this returns a complete set of inputs (missing chunks filled).
    # Note: If running in a local Dask cluster, prints to console may be duplicated. Doesn't happen with a Coiled cluster of the same size (1 worker).
    # Seems to be a problem with local Dask getting overwhelmed by so many futures being created and downloaded from s3.
    futures = uu.prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_final, logger_worker)
    # print(futures)

    lu.print_and_log(f"Waiting for requests for data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)

    # Dictionary that stores the dataset name (key) and downloaded data and their statuses (values)
    layers = {}

    # Ensures futures stores Future objects
    # Revised with https://chatgpt.com/share/e/67bde66c-d9a0-800a-a524-a9ef88c641a2 to return status messages for chunks
    for future in concurrent.futures.as_completed(futures):
        layer = futures[future]  # Gets the corresponding key
        data, status = future.result()  # Unpacks the tuple result
        if 'success' not in status: # Prints and logs any inputs that couldn't be accessed and are downloaded as all 0s
            lu.print_and_log(f"{status}: {uu.timestr()}", is_final, logger_worker)
        layers[layer] = data

    # Test prints
    # print(layers)
    # print(layers['burned_area_2015'].max())
    # print(layers[cn.climate_zone_pattern].max())
    # print(layers[cn.planted_forest_AGC_BGC_removal_factor_pattern])
    # print(layers[cn.planted_forest_AGC_BGC_removal_factor_pattern].max())
    # print(layers[cn.forest_age_start_year_pattern].dtype)
    # print(layers[cn.climate_zone_pattern].dtype)


    ### Part 2: Calculates min, mean, and max for each input chunk.
    ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.

    # Calculates stats for the input layers
    for key, array in layers.items():
        chunk_stats.append(uu.calculate_stats(array, key, bounds_str, tile_id, 'input_layer'))
    # print(chunk_stats)


    ### Part 3: Creates a separate dictionary for each chunk datatype so that they can be passed to Numba as separate arguments.
    ### Numba functions can accept (and return) dictionaries of arrays as long as each dictionary only has arrays of one data type (e.g., uint8, float32).
    ### Note: need to add new code if inputs with other data types are added

    lu.print_and_log(f"Creating typed dictionaries for chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

    # Creates the typed dictionaries for all input layers (including those that originally had no data)
    typed_dict_uint8, typed_dict_uint16, typed_dict_int16, typed_dict_int32, typed_dict_float32 = nu.create_typed_dicts(layers)

    # print("uint8_typed_list:", typed_dict_uint8)
    # print("uint16_typed_list:", typed_dict_uint16)
    # print("int16_typed_list:", typed_dict_int16)
    # print("int32_typed_list:", typed_dict_int32)
    # print("float32_typed_list:", typed_dict_float32)

    # Frees up a little memory (~15 MB) before moving on
    del updated_download_dict
    del futures
    gc.collect()


    ### Part 4: Calculates LULUCF fluxes and densities

    lu.print_and_log(f"Calculating LULUCF fluxes and carbon densities in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)
    uu.rename_s3_task_file(stage, bounds, "calculating_", is_final, logger_worker)
    numba_start = time.time()

    out_dict_uint8, out_dict_uint16, out_dict_uint32, out_dict_float32 = LULUCF_fluxes(
        typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32,
        primary_forest_RF_array, partial_disturbance_EF_array, mangrove_C_ratio_array,
        start_year, end_year, interval_type, interval_year_diff_list, interval_length_list, interval_end_years, is_final)

    numba_end = time.time()
    lu.print_and_log(f"Done calculating LULUCF fluxes and carbon densities in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)
    lu.print_and_log(f"Memory usage after numba calculations completed for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB", is_final, logger_worker)
    lu.print_and_log(f"Calculated LULUCF fluxes and carbon densities in {bounds_str} in {tile_id} in {round(numba_end-numba_start)} seconds: {uu.timestr()}", False, logger_worker)

    # print(out_dict_uint32)
    # print(out_dict_float32)
    # print(f"Average of {list(out_dict_uint32.keys())[0]} is: {list(out_dict_uint32.values())[0].mean()}")

    # Fresh non-Numba-constrained dictionary that stores all numpy arrays.
    # The dictionaries by datatype that are returned from the numba function have limitations on them,
    # e.g., they can't be combined with other datatypes. This prevents the addition of attributes needed for uploading to s3.
    # So the trick here is to copy the numba-exported arrays into normal Python arrays to which we can do anything in Python.
    # Everything in out_dict also needs to be in cn.LULUCF_core_output_dirs
    # because that has the list of basic output directories which are customized for this run
    out_dict_all_dtypes = {}

    # Transfers the dictionaries of numpy arrays for each data type to a new, Pythonic array
    out_dicts = [out_dict_uint8, out_dict_uint16, out_dict_uint32, out_dict_float32]

    # Loop through each dictionary and update out_dict_all_dtypes
    for out_dict in out_dicts:
        for key, value in out_dict.items():
            out_dict_all_dtypes[key] = value

        # Clear memory of unneeded arrays
        del out_dict


    # Deletes all unnecessary input dictionaries before moving on
    # Suggested by ChatGPT: https://chatgpt.com/share/e/672bbf2e-ebbc-800a-aae3-3d92f5a1d663
    in_dicts = [layers, typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32]
    [in_dict.clear() for in_dict in in_dicts]


    ### Part 5: Calculates per ha min, per ha mean, per ha max, and per pixel sum for each output chunk.
    ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.
    ### Also useful for a quick sum of outputs without doing zonal stats

    lu.print_and_log(f"Populating chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)

    # The relevant pixel area (m^2) file in s3
    pixel_area_uri = f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tile_id}.tif"

    # Gets numpy arrays of the model output being analyzed and the area (m^2) per pixel
    pixel_area_chunk = uu.get_tile_dataset_rio(pixel_area_uri, bounds, chunk_length_pixels, 'Float32')
    pixel_area_chunk = pixel_area_chunk[0]  # Converts downloaded tuple (array, status) to just the array

    # Calculates stats for the output layers from create_starting_C_densities as a dictionary with chunk attributes
    for key, array_per_ha in out_dict_all_dtypes.items():

        # Converts per hectare values to per pixel values for the output numpy array
        output_per_pixel = array_per_ha * pixel_area_chunk * cn.m2_to_ha

        chunk_stats.append(uu.calculate_stats(array_per_ha, key, bounds_str, tile_id, 'output_layer', output_per_pixel))

    lu.print_and_log(f"Populated chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)


    ### Part 6: Saves numpy arrays as rasters and uploads to s3

    uu.rename_s3_task_file(stage, bounds, "uploading_", is_final, logger_worker)

    # Only saves arrays to geotifs and uploads them to s3 if enabled
    if no_upload == False:

        out_no_data_val = 0  # NoData value for output raster (optional)

        # Adds metadata used for uploading outputs to s3 to the dictionary
        for key, value in out_dict_all_dtypes.items():
            data_type = value.dtype.name
            # print("key", key)
            # print(data_type)

            # Retrieves the file name pattern and date(s) covered for the output file for use in s3 folder construction
            out_pattern, year_range = uu.strip_and_extract_years(key)
            # print("out_pattern:", out_pattern)
            # print("year_range:", year_range)

            # Gets the core filename pattern and pixel meaning
            out_pattern_without_pixel_meaning, pixel_meaning = uu.strip_pixel_meaning(out_pattern)
            # print("out_pattern_without_pixel_meaning:", out_pattern_without_pixel_meaning)

            # Retrieves the relevant output s3 path for this specific output  (list of one element)
            # First, finds the output folders for all intervals with the relevant patterns
            matched_output_s3_folders = [item for item in output_folders if out_pattern_without_pixel_meaning in item]
            # print("matched_output_s3_folders:", matched_output_s3_folders)

            # Second, finds the output folder with the right interval for that pattern
            matched_output_s3_folder_list = [item for item in matched_output_s3_folders if year_range in item]
            # print("matched_output_s3_folder_list:", matched_output_s3_folder_list)

            # Output paths without bucket (s3://gfw2-data).
            # Needs [0] because matched_output_s3_folder_list is a list of all intervals.
            s3_path_without_bucket = f"{matched_output_s3_folder_list[0][cn.full_bucket_prefix_length:]}"
            # print("s3_path_without_bucket:", s3_path_without_bucket)

            # Dictionary with metadata for each array
            out_dict_all_dtypes[key] = [value, data_type, out_pattern, year_range, s3_path_without_bucket]


        # Converts output numpy arrays to local rasters and puts them in a list of files to upload in parallel
        upload_tasks = uu.save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str,
                                                        out_dict_all_dtypes, is_final, logger_worker, out_no_data_val)

        lu.print_and_log(f"Upload tasks created for {bounds_str} in {tile_id}. Uploading now: {uu.timestr()}", False, logger_worker)

        # Execute uploads in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks)

        lu.print_and_log(f"Uploads completed for {bounds_str} in {tile_id} using {cn.outputs_path}: {uu.timestr()}", is_final, logger_worker)

    chunk_end_time = time.time()
    lu.print_and_log(f"{bounds_str} took {round(chunk_end_time - chunk_start_time)} seconds: {uu.timestr()}", False, logger_worker)

    return_message = f"Success for {bounds_str}: {uu.timestr()}"

    # Removes task tracking file from S3 once task is successful
    uu.delete_s3_task_file(stage, bounds, is_final, logger_worker)

    return return_message, chunk_stats  # Return both the success message and the statistics


def main(cluster_name, run_date, year_range, run_local=False, no_stats=False, no_log=False, no_upload=False,
         chunk_shapefile_uri=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = 'LULUCF_fluxes'
    model_type = 'standard_model'

    # Runs chunks in batches of specified size.
    # Each batch slows down processing because chunks inevitably lag and that happens more the more batches there are.
    batch_size = 2000
    # batch_size = 5  # For testing batch processing

    # Determines if arguments for start and end year are valid
    if year_range not in [[cn.first_model_year_5_years, cn.last_model_year_5_years],  # 2000-2020
                          [cn.first_model_year_5_years, cn.last_model_year_annual],  # 2000-2023
                          [cn.first_model_year_annual, cn.last_model_year_annual]]:  # 2015-2023
        print("Year range selection not valid")
        sys.exit()
    else:
        start_year = year_range[0]
        end_year = year_range[1]
        # print(f"Start year: {start_year}")
        # print(f"End year: {end_year}")

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    import dask
    # Followed https://docs.dask.org/en/latest/configuration.html#specify-configuration to set TTL timeout to 800 seconds
    print("TTL from dask config:", dask.config.get("distributed.scheduler.worker-ttl"))

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    start_time = uu.timestr() # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Start year: {start_year}; end year: {end_year}")
    main_logger.info(f"Run date: {run_date}")
    main_logger.info(f"Batch size: {batch_size} chunks")
    main_logger.info(f"no_upload: {no_upload}")

    # Calculates the interval type, difference between start and end years of intervals, and the model output years
    # for the model run
    interval_type, interval_year_diff_list, interval_length_list, interval_end_years = uu.get_interval_info(end_year, main_logger, start_year)

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size, first_chunks, fishnet_iso_df, main_logger)

    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_final = False
    # is_final = True  # For simulating a large run
    if len(chunk_list) > 20:
        is_final = True
        main_logger.info("Running as final model.")

    # This is just a placeholder tile_id that is used to obtain the datatype of each input tile set.
    # It is overwritten when chunks are assigned and analyzed.
    # Using this placeholder allows the full path and tile name to be specified up front, which simplifies things.
    # Otherwise, we'd have just the path but not the file name now and would have to add in the file name later
    # (probably at the chunk level).
    # It shouldn't really matter what the sample_tile_id is.
    sample_tile_id = "00N_000E"

    # Dictionary of data to download (inputs to model).
    # These inputs don't depend on the starting year of the model.
    download_dict = {
        cn.r_s_ratio_non_mang_pattern: f"{cn.r_s_ratio_non_mang_dir}{sample_tile_id}_{cn.r_s_ratio_non_mang_pattern}.tif",

        cn.drivers_pattern: f"{cn.drivers_path}{sample_tile_id}_{cn.drivers_pattern}.tif",

        cn.planted_forest_type_pattern: f"{cn.planted_forest_type_dir}{sample_tile_id}_{cn.planted_forest_type_pattern}.tif",
        cn.planted_forest_AGC_removal_factor_pattern: f"{cn.planted_forest_AGC_removal_factor_dir}{sample_tile_id}_{cn.planted_forest_AGC_removal_factor_pattern}.tif",
        cn.planted_forest_AGC_BGC_removal_factor_pattern: f"{cn.planted_forest_AGC_BGC_removal_factor_dir}{sample_tile_id}_{cn.planted_forest_AGC_BGC_removal_factor_pattern}.tif",
        cn.oil_palm_2000_extent_pattern: f"{cn.oil_palm_2000_extent_dir}{sample_tile_id}_{cn.oil_palm_2000_extent_pattern}.tif",
        cn.oil_palm_first_year_pattern: f"{cn.oil_palm_first_year_dir}{cn.oil_palm_first_year_pattern}_{sample_tile_id}.tif",   # Pattern is before tile_id for this input
        # Originally from gfw-data-lake, so it's in 400x400 windows
        cn.planted_forest_tree_crop_pattern: f"{cn.planted_forest_tree_crop_dir}{sample_tile_id}.tif",

        # Originally from gfw-data-lake, so it's in 400x400 windows
        cn.elevation_pattern: f"{cn.elevation_dir}{sample_tile_id}_{cn.elevation_pattern}.tif",
        cn.climate_domain_pattern: f"{cn.climate_domain_dir}{sample_tile_id}_{cn.climate_domain_pattern}.tif",
        cn.climate_zone_pattern: f"{cn.climate_zone_processed_dir}{sample_tile_id}_{cn.climate_zone_pattern}.tif",
        cn.precipitation_pattern: f"{cn.precipitation_dir}{sample_tile_id}_{cn.precipitation_pattern}.tif",
        # "ecozone": f"s3://gfw2-data/fao_ecozones/v2000/raster/epsg-4326/10/40000/class/gdal-geotiff/{sample_tile_id}.tif",   # Originally from gfw-data-lake, so it's in 400x400 windows
        cn.ifl_primary_pattern: f"{cn.ifl_primary_dir}{sample_tile_id}_{cn.ifl_primary_pattern}.tif",
        cn.continent_ecozone_pattern: f"{cn.continent_ecozone_dir}{sample_tile_id}_{cn.continent_ecozone_pattern}.tif",
        cn.pixel_area_pattern: f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{sample_tile_id}.tif"
    }

    # Starting carbon pools depend on the starting year
    # Uses the carbon maps for 2000 for 5-year and hybrid-interval models
    if interval_type in [cn.intervals_five_years, cn.intervals_hybrid]:
        download_dict[cn.agc_raw_dens_pattern] = f"{cn.agc_2000_dir}{sample_tile_id}__{cn.agc_2000_pattern}.tif"
        download_dict[cn.bgc_raw_dens_pattern] = f"{cn.bgc_2000_dir}{sample_tile_id}__{cn.bgc_2000_pattern}.tif"
        download_dict[cn.deadwood_c_raw_dens_pattern] = f"{cn.deadwood_c_2000_dir}{sample_tile_id}__{cn.deadwood_c_2000_pattern}.tif"
        download_dict[cn.litter_c_raw_dens_pattern] = f"{cn.litter_c_2000_dir}{sample_tile_id}__{cn.litter_c_2000_pattern}.tif"
    ##TODO: 2015 carbon maps are still using the 2000 mangrove carbon map!!
    elif interval_type == cn.intervals_annual:
        download_dict[cn.agc_raw_dens_pattern] = f"{cn.agc_2015_dir}{sample_tile_id}__{cn.agc_2015_pattern}.tif"
        download_dict[cn.bgc_raw_dens_pattern] = f"{cn.bgc_2015_dir}{sample_tile_id}__{cn.bgc_2015_pattern}.tif"
        download_dict[cn.deadwood_c_raw_dens_pattern] = f"{cn.deadwood_c_2015_dir}{sample_tile_id}__{cn.deadwood_c_2015_pattern}.tif"
        download_dict[cn.litter_c_raw_dens_pattern] = f"{cn.litter_c_2015_dir}{sample_tile_id}__{cn.litter_c_2015_pattern}.tif"
    else:
        sys.exit('interval_type not found')

    # Land cover and vegetation height timeseries depend on interval_type
    if interval_type == cn.intervals_five_years:
        # Land cover and vegetation height rasters (5-year intervals)
        for year in range(cn.first_model_year_5_years, cn.last_model_year_5_years + 1, cn.five_year_interval_duration):
            download_dict[f"{cn.land_cover_pattern}_{year}"] = f"{cn.land_cover_5_year_path}{year}/{sample_tile_id}.tif"
            download_dict[f"{cn.vegetation_height_pattern}_{year}"] = f"{cn.vegetation_height_5_year_path}{year}/{sample_tile_id}_{cn.vegetation_height_5_year_pattern}_{year}.tif"
    elif interval_type == cn.intervals_annual:
        # Land cover and vegetation height rasters (annual intervals)
        for year in range(cn.first_model_year_annual, cn.last_model_year_annual + 1):
            download_dict[f"{cn.land_cover_pattern}_{year}"] = f"{cn.land_cover_annual_path}{year}/{sample_tile_id}.tif"
            download_dict[f"{cn.vegetation_height_pattern}_{year}"] = f"{cn.vegetation_height_annual_path}{year}/{sample_tile_id}.tif"
    elif interval_type == cn.intervals_hybrid:
        # Land cover and vegetation height rasters (5-year intervals for 2000, 2005, and 2010 only)
        for year in range(cn.first_model_year_5_years, 2010+1, cn.five_year_interval_duration):
            download_dict[f"{cn.land_cover_pattern}_{year}"] = f"{cn.land_cover_5_year_path}{year}/{sample_tile_id}.tif"
            download_dict[f"{cn.vegetation_height_pattern}_{year}"] = f"{cn.vegetation_height_5_year_path}{year}/{sample_tile_id}_{cn.vegetation_height_5_year_pattern}_{year}.tif"
        # Land cover and vegetation height rasters (annual intervals for 2015 onwards)
        for year in range(cn.first_model_year_annual, cn.last_model_year_annual + 1):
            download_dict[f"{cn.land_cover_pattern}_{year}"] = f"{cn.land_cover_annual_path}{year}/{sample_tile_id}.tif"
            download_dict[f"{cn.vegetation_height_pattern}_{year}"] = f"{cn.vegetation_height_annual_path}{year}/{sample_tile_id}.tif"
    else:
        sys.exit('interval_type not found')


    # Burned area rasters (every year)-- same code for annual, 5-year model, or hybrid.
    # Each burned area year needs to be in its own folder.
    for year in range(start_year, end_year + 1):  # Annual burned area maps start in 2000
        download_dict[f"{cn.burned_area_final_pattern}_{year}"] = f"{cn.full_bucket_prefix}/{cn.burned_area_final_dir}{year}/{sample_tile_id}_{cn.burned_area_final_pattern}_{year}.tif"

    # Forest disturbance rasters (every year)-- only for 5-year intervals
    # All years need to be in their own folder
    if interval_type in cn.intervals_five_years:
        for year in range(cn.first_model_year_5_years + 1, cn.last_model_year_5_years + 1):  # Annual forest disturbance maps start in 2001 and ends in 2020
            download_dict[f"{cn.forest_disturbance_layer_name}_{year}"] = f"{cn.forest_disturbance_annual_dir}{year}/{year}_{sample_tile_id}.tif"
    elif interval_type in cn.intervals_hybrid:
        for year in range(cn.first_model_year_5_years + 1, 2016):  # Hybrid model uses annual disturbance data through 2015. Annual data used in 2015-2016 onwards.
            download_dict[f"{cn.forest_disturbance_layer_name}_{year}"] = f"{cn.forest_disturbance_annual_dir}{year}/{year}_{sample_tile_id}.tif"
    else:  # Annual model does not use annual disturbance data
        pass


    # Young natural forest rasters (several age intervals)
    # Each growth interval's rate is in its own folder
    for growth_interval in cn.natural_forest_growth_curve_intervals:
        download_dict[f"{cn.natural_forest_growth_curve_pattern}__{growth_interval}_years"] = f"{cn.natural_forest_growth_curve_dir}rate_{growth_interval}/{sample_tile_id}_{cn.natural_forest_growth_curve_pattern}__{growth_interval}_years.tif"

    # Starting forest age
    if interval_type == cn.intervals_annual:
        download_dict[f"{cn.forest_age_start_year_pattern}"] = f"{cn.forest_age_2015_interpolated_dir}{sample_tile_id}__{cn.forest_age_2015_interpolated_pattern}.tif"
    # TODO: Need to make starting forest age for 2000
    else:
        download_dict[f"{cn.forest_age_start_year_pattern}"] = f"{cn.forest_age_2015_interpolated_dir}{sample_tile_id}__{cn.forest_age_2015_interpolated_pattern}.tif"

    # Replaces the placeholder parts of the input paths with relevant values
    download_dict = {
        key: value.replace("CHUNK_SIZE", '40000')
        for key, value in download_dict.items()
    }
    download_dict = {
        key: value.replace("PER_HA_OR_PIXEL", cn.C_density_pixel_meaning)
        for key, value in download_dict.items()
    }

    # print(download_dict)

    # Returns the first tile in each input so that the datatype can be determined.
    # This is done up front, once per tile set, rather than on each chunk, since
    # all tiles have the same datatype for each input-- it only needs to be done once at the very beginning of the stage.
    main_logger.info(f"Getting tile_id of first tile in each tile set: {uu.timestr()}")
    first_tiles = uu.first_file_name_in_s3_folder(download_dict)

    # Creates a download dictionary with the datatype of each input in the values.
    # This is supplied to each chunk that is being analyzed.
    # This also serves as a check of whether all inputs are being found (s3 paths correct)
    main_logger.info(f"Getting datatype of first tile in each tile set: {uu.timestr()}")
    download_dict_with_data_types = uu.add_file_type_to_dict(first_tiles)
    # print(download_dict_with_data_types)


    # Creates a list of output directories (core and intermediates) for all outputs and intervals based on specifics of the model run
    output_dir_list_core_intermediate = cn.LULUCF_core_output_dirs + cn.LULUCF_intermediate_output_dirs
    output_dir_list = uu.create_output_dir_name_list(output_dir_list_core_intermediate, interval_type, start_year,
                                                     chunk_size_pixels, model_type, interval_end_years,
                                                     interval_year_diff_list, run_date, "per_ha")
    output_dir_list.sort()  # Alphabetically order the outputs (modifies output_dir_list)
    # print(output_dir_list)

    # Creates numpy array of IPCC Tier 1 primary forest removal factors by continent-ecozone combination.
    # Needs to by a numpy array for the numba function to use it.
    # Inputs are Mg AGB/ha/yr. Outputs are Mg AGB/ha/yr. Conversion to Mg AGC/ha/yr is done below.
    primary_forest_RF_array = uu.convert_lookup_table_to_array(cn.RF_C_ratio_spreadsheet_full_path,
                                                               cn.IPCC_removal_factor_table_tab,
                                                               ['gainEcoCon', 'growth_primary'])

    # Converts primary forest AGB RFs to AGC RFs (Mg AGB/ha/yr -> Mg AGC/ha/yr)
    primary_forest_RF_array[:, 1] = primary_forest_RF_array[:, 1] * cn.biomass_to_carbon_non_mangrove


    # Creates numpy array of ratios of BGC, deadwood C, and litter C relative to AGC. Relevant columns must be specified.
    mangrove_C_ratio_array = uu.convert_lookup_table_to_array(cn.RF_C_ratio_spreadsheet_full_path, cn.mangrove_rate_ratio_tab,
                                                              ['gainEcoCon', 'BGC_AGC', 'deadwood_AGC', 'litter_AGC'])

    # Creates numpy array of emission factors for partially disturbed forest by driver and continent-ecozone combination
    partial_disturbance_EF_array = uu.convert_lookup_table_to_array(cn.partial_disturbance_emission_factor_table_full_path,
                                                                    cn.partial_disturbance_emission_factor_table_tab,
                                                                    ['gainEcoCon', '1_perm_ag_EF', '2_hard_comm_EF',
                                                                     '3_shift_cult_EF',	'4_logging_EF',	'5_wildfire_EF',
                                                                     '6_sett_infrastr_EF', '7_natrl_dist_EF'])

    # Makes a txt for each task in the list. These are deleted as tasks are completed.
    main_logger.info("Creating task txts in s3...")
    uu.create_s3_task_files(stage, chunk_list)


    ### Step 2: Create 1x1 degree outputs

    # Creates list of tasks to run (1 task = 1 chunk)
    main_logger.info(f"Creating tasks and starting processing: {uu.timestr()}")
    main_logger.info("Workers' logs to be appended after main function log"+ "\n")

    chunk_batches = [chunk_list[i:i + batch_size] for i in range(0, len(chunk_list), batch_size)]
    main_logger.info(f"There are {len(chunk_batches)} batches to process: {uu.timestr()}")

    # Accumulates all output messages and statistics across batches
    # From https://chatgpt.com/share/e/5599b6b0-1aaa-4d54-98d3-c720a436dd9a
    all_flux_results = []
    all_1x1_stats = []
    success_count = 0  # Count of successful chunks

    # Iterates through the batches
    for i, chunk_batch in enumerate(chunk_batches):
        main_logger.info(f"Processing batch {i + 1}/{len(chunk_batches)} ({len(chunk_batch)} chunks): {uu.timestr()}")
        main_logger.info("Creating batch task txts in s3...")
        uu.create_s3_task_files(stage, chunk_batch)

        # This approach handles large task lists (graphs) better than [dask.delayed(calculate_and_upload_LULUCF_fluxes ... )]
        futures = []
        for chunk in chunk_batch:
            future = client.submit(calculate_and_upload_LULUCF_fluxes,
                                   chunk, primary_forest_RF_array, partial_disturbance_EF_array, mangrove_C_ratio_array,
                                   download_dict_with_data_types, start_year, end_year, interval_type, interval_year_diff_list,
                                   interval_length_list, interval_end_years, is_final, no_upload, output_dir_list, stage)
            futures.append(future)

        batch_flux_results = client.gather(futures)

        all_flux_results.extend(batch_flux_results)

        success_count, batch_stats = uu.count_successful_chunks(chunk_batch, is_final, main_logger, batch_flux_results)
        all_1x1_stats.extend(batch_stats)

        del futures
        del batch_flux_results
        client.run(gc.collect)

        uu.stage_duration(start_time, uu.timestr(), f"{stage}, batch {i}", main_logger)


    ### Step 3: Counts files in output folders, chunk stats for 1x1 degree outputs, aggregates logs

    # Resizes cluster down to 1 worker for chunk stats and log aggregation since that only needs a minimal remainder of the
    # cluster, not all the workers.
    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster(cluster_name, 1)

    # Iterates through output folders and counts the number of output rasters (only if uploads enabled)
    if not no_upload:
        for output_folder in output_dir_list:

            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
            main_logger.info(f"Output rasters in {output_folder}: {file_count}")
            # print(geotiff_files)

    # Prepares chunk stats spreadsheet: min, mean, max, and sum for all input and output chunks,
    # and min and max values across all chunks for all inputs and outputs
    # only if not suppressed by the --no_stats flag and at least one chunk was successful (wasn't skipped).
    if (not no_stats) and (success_count > 0):
        uu.compile_1x1_chunk_stats(all_1x1_stats, chunk_shapefile_uri, stage, no_upload, main_logger)

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats", main_logger)

    # Worker logs are not aggregated if doing a local run (since there are no workers)
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats and worker log compilation", main_logger)

        # Adds the workers' logs to the main log and uploads to s3
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)

    # Closes the Dask client if not running locally
    if not run_local:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate LULUCF fluxes.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--run_date', help='Date of run, in YYYYMMDD')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('-yr', '--year_range', nargs=2, type=int, required=True, help='Starting and ending years for model. Start options: 2000, 2015. End options: 2020, 2023.')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    run_date = args.run_date
    bounding_box = args.bounding_box
    chunk_size = args.chunk_size
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_chunks = args.first_chunks
    year_range = args.year_range
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, run_date, year_range, run_local, no_stats, no_log, no_upload, chunk_shapefile_uri,
         bounding_box=bounding_box, chunk_size=chunk_size, first_chunks=first_chunks, log_note=log_note)