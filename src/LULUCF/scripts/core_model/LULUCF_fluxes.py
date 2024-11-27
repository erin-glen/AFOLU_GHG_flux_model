"""
Run from src/LULUCF

Test:
python -m scripts.utilities.create_cluster -n 1
python -m scripts.core_model.LULUCF_fluxes -cn AFOLU_flux_model_scripts -bb 10 49.75 10.25 50 -cs 0.25
python -m scripts.core_model.LULUCF_fluxes -cn AFOLU_flux_model_scripts -bb 115.25 -3.75 115.5 -3.5 -cs 0.25 --no_upload
python -m scripts.core_model.LULUCF_fluxes -cn AFOLU_flux_model_scripts -cl s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20241125/ -f 1

Full run:
python -m scripts.utilities.create_cluster -n 200
python -m scripts.core_model.LULUCF_fluxes -cn AFOLU_flux_model_scripts -cl s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20241125/
"""

import argparse
import concurrent.futures
import sys
import re
import numpy as np

from dask.distributed import print
from numba import jit

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu
from ..utilities import log_utilities as lu
from ..utilities import numba_utilities as nu


# Function to calculate LULUCF fluxes and carbon densities
# Operates pixel by pixel, so uses numba (Python compiled to C++).
@jit(nopython=True)
def LULUCF_fluxes(in_dict_uint8, in_dict_int16, in_dict_int32, in_dict_float32, primary_forest_RFs, is_final):

    # Separate dictionaries for output numpy arrays of each datatype, named by output data type).
    # This is because a dictionary in a Numba function cannot have arrays with multiple data types, so each dictionary has to store only one data type,
    # just like inputs to the function.
    out_dict_uint8 = {}
    out_dict_uint16 = {}
    out_dict_uint32 = {}
    out_dict_float32 = {}

    # Numpy arrays for outputs that do depend on previous interval's values
    agc_dens_block = in_dict_float32[cn.agc_2000_pattern].astype('float32')
    bgc_dens_block = in_dict_float32[cn.bgc_2000_pattern].astype('float32')
    deadwood_c_dens_block = in_dict_float32[cn.deadwood_c_2000_pattern].astype('float32')
    litter_c_dens_block = in_dict_float32[cn.litter_c_2000_pattern].astype('float32')
    soil_c_dens_block = in_dict_int16[cn.soil_c_2000_pattern].astype('float32')

    r_s_ratio_block = in_dict_float32[cn.r_s_ratio_pattern].astype('float32')

    natrl_forest_curve_0_5_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__0_5_years"].astype('float32')
    natrl_forest_curve_6_10_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__6_10_years"].astype('float32')
    natrl_forest_curve_11_15_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__11_15_years"].astype('float32')
    natrl_forest_curve_16_20_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__16_20_years"].astype('float32')
    natrl_forest_curve_21_100_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__21_100_years"].astype('float32')

    # Removal factor (Mg C/ha/yr)
    # Because this is used to store the RF from the previous interval,
    # it persists from one interval to the next. Therefore, it must be defined before the first iteration.
    # That way, removal factors can be over-written by those used in the latest interval.
    agc_rf_pre_dist_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')

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


    ## Test/intermediate outputs blocks

    # Stores the burned area blocks for the entire model duration (added to progressively during each interval)
    burned_area_blocks_all_intervals_so_far = []

    # Stores the forest disturbance blocks for the entire model duration (added to progressively during each interval)
    forest_dist_blocks_all_intervals_so_far = []

    # Stores the last year that each pixel did not have tall vegetation composite land cover.
    # 0=Always tall vegetation so far. Other values represent the last year of non-tall vegetation.
    # This is assessed at the pixel level because numba wouldn't allow the needed logical operations on numpy arrays (chunks).
    # Tall vegetation is basd on the composite land cover maps, not the canopy height maps.
    most_recent_year_not_tall_veg_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('uint16')

    # Number of years of regrowth for new forest
    years_of_forest_regrowth_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('uint8')

    # Year in which forest loss occurs/is assigned during an interval (0 if no loss)
    year_of_forest_loss_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('uint16')

    # Maximum height of vegetation since the last interval in which there was not forest
    max_height_since_last_time_not_tall_veg_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('uint8')

    # Tracks whether the height has already decreased more than the signif. height loss threshold compared to
    # maximum vegetation height since the last time the pixel was non-tall vegetation land cover.
    # This prevents a pixel from repeatedly (multiple intervals) being counted as having a height loss disturbance compared to
    # the maximum height; this can only be triggered once since the maximum height is attained.
    # This is used to determine if a forest->forest disturbance based on height loss relative to the max height should be reported.
    # 0=no significant height loss relative to the maximum vegetation height since last non-tall vegetation.
    # 1=height loss relative to the maximum vegetation height occurred in this interval.
    # 2=height loss relative to the maximum vegetation height occurred in a previous interval.
    first_time_sig_loss_from_max_height_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('uint8')


    # Iterates through model intervals
    for interval_end_year in list(range(cn.first_model_year, cn.last_model_year + 1, cn.interval_years))[1:]:

        # print(f"Now at {interval_end_year}:")

        # Model intervals so far, including the model start year.
        # Eventually used to determine whether current height has decreased significantly from maximum height since last non-tall veg year over multiple intervals (gradual height loss).
        years_so_far = list(range(cn.first_model_year, interval_end_year + 1, cn.interval_years))

        # Pre-fetches vegetation height data for this chunk and stores in a dictionary or list.
        # Eventually used to determine whether current height has decreased significantly from maximum height since last non-tall veg year over multiple intervals (gradual height loss).
        # Suggested by https://chatgpt.com/share/e/6724d803-aca4-800a-928c-11d76d38c0ec to work well with numba
        # and speed the code up. I was trying to get vegetation height so far in a variety of ways and it kept being slow.
        # This approach, in conjunction with some pixel-level operations below, seems to not slow down the code.
        vegetation_heights_so_far_block = [
            in_dict_uint8[f"{cn.vegetation_height_pattern}_{year}"]
            for year in years_so_far
        ]

        # Writes the dictionary entries to a chunk for use in the decision tree
        LC_prev_block = in_dict_uint8[f"{cn.land_cover_pattern}_{interval_end_year - cn.interval_years}"]
        LC_curr_block = in_dict_uint8[f"{cn.land_cover_pattern}_{interval_end_year}"]
        veg_h_prev_block = in_dict_uint8[f"{cn.vegetation_height_pattern}_{interval_end_year - cn.interval_years}"]
        veg_h_curr_block = in_dict_uint8[f"{cn.vegetation_height_pattern}_{interval_end_year}"]

        # Creates a list of all the burned area arrays from 2001 to the end of the interval.
        # It works by getting the burned area chunks for the current interval and appending them to a list of chunks
        # from previous intervals.
        for year_offset in range(interval_end_year-4, interval_end_year+1):
            year_key = f"{cn.burned_area_pattern}_{year_offset}"
            burned_area_blocks_all_intervals_so_far.append(in_dict_uint8[year_key])


        # Creates a list of all the forest disturbance arrays from 2001 to the end of the interval.
        # The values in the list are the disturbance year starting from 1, e.g., 2001=1, 2008=8, 2017=17.
        # It works by getting the annual disturbance chunks for the current interval and appending them to a list of
        # chunks from previous intervals.
        for year_offset in range(interval_end_year-4, interval_end_year+1):

            # The name of the disturbance layer in the input dictionary
            year_key = f"{cn.forest_disturbance_layer_name}_{year_offset}"

            # Replaces the binary annual disturbance array with the year of disturbance (1, 2, 3...20)
            year_disturb_array = in_dict_uint8[year_key] * (year_offset - cn.first_model_year)

            # Makes a list of disturbance arrays with the disturbance year.
            # uint8 is okay because the highest value should be 20 (not 2020).
            forest_dist_blocks_all_intervals_so_far.append(year_disturb_array.astype('uint8'))


        # Numpy arrays for outputs that don't depend on previous interval's values
        state_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('uint32')  # Land cover state at end of interval

        # Number of years of canopy growth.
        # First digit is pre-disturbance years of growth.
        # Second digit (if it exists) is post-disturbance years of growth
        gain_year_count_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('uint8')

        agc_gross_emis_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')
        bgc_gross_emis_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')
        deadwood_c_gross_emis_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')
        litter_c_gross_emis_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')

        ch4_gross_emis_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')
        n2o_gross_emis_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')

        agc_gross_removals_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')
        bgc_gross_removals_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')
        deadwood_c_gross_removals_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')
        litter_c_gross_removals_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')


        # Iterates through all pixels in the chunk
        for row in range(LC_curr_block.shape[0]):
            for col in range(LC_curr_block.shape[1]):

                ### Defines pixel values

                LC_prev = LC_prev_block[row, col]
                LC_curr = LC_curr_block[row, col]
                veg_h_prev = veg_h_prev_block[row, col]
                veg_h_curr = veg_h_curr_block[row, col]

                r_s_ratio_cell = r_s_ratio_block[row, col]

                # Replaces pixel without R:S (0) with the global non-mangrove R:S default #TODO This is the non-mangrove default. Need to adjust if mangrove pixel?
                if r_s_ratio_cell == 0:
                    r_s_ratio_cell = cn.default_r_s_non_mang

                # Secondary forest removal factors (Mg AGC/ha/yr)
                natrl_forest_curve_0_5_AGC_RF_cell = natrl_forest_curve_0_5_AGC_RF_block[row, col]
                natrl_forest_curve_6_10_AGC_RF_cell = natrl_forest_curve_6_10_AGC_RF_block[row, col]
                natrl_forest_curve_11_15_AGC_RF_cell = natrl_forest_curve_11_15_AGC_RF_block[row, col]
                natrl_forest_curve_16_20_AGC_RF_cell = natrl_forest_curve_16_20_AGC_RF_block[row, col]
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

                # Determines the removal factor for primary forests/IFL based on the ecozone (Mg AGC/ha/yr)
                primary_forest_AGC_RF = nu.calc_primary_forest_RF(continent_ecozone_cell, primary_forest_RFs)

                # Assigns the previous interval's pre-disturbance removal factor to this interval.
                # This is only used for natural forests (non-SDPT/oil palm), where the RF can change from interval to interval.
                # If it's the first interval (i.e. no previous interval),
                # primary forest/IFL RF is used if in that, otherwise old secondary forest RF is used.
                # This effectively assumes that any disturbed natural forest has
                # primary forest or old secondary forest RF before disturbance (not young secondary).
                # Obviously not realistic but a fine starting simplification, I think.
                #TODO: Include better way to determine what the starting natural forest RF should be than just assuming
                # primary or old secondary forest
                if interval_end_year == 2005:   # If first interval...
                    if ifl_primary_cell:   # And if in IFL/primary forest...
                        agc_rf_pre_dist_prev = primary_forest_AGC_RF   # Use the primary forest/IFL RF
                    else:  # If not in IFL/primary forest...
                        agc_rf_pre_dist_prev = natrl_forest_curve_21_100_AGC_RF_cell   # Use old secondary forest RF
                else:  # If not first interval...
                    agc_rf_pre_dist_prev = agc_rf_pre_dist_out_block[row, col]  # Use removal factor from the previous interval


                # Note: Stacking the burned area rasters using ndstack, stack, or flatten outside the pixel iteration did not work with numba.
                # So just reading each raster from the list of rasters separately.
                burned_area_t_4 = burned_area_blocks_all_intervals_so_far[-5][row, col]
                burned_area_t_3 = burned_area_blocks_all_intervals_so_far[-4][row, col]
                burned_area_t_2 = burned_area_blocks_all_intervals_so_far[-3][row, col]
                burned_area_t_1 = burned_area_blocks_all_intervals_so_far[-2][row, col]
                burned_area_t = burned_area_blocks_all_intervals_so_far[-1][row, col]
                # Most recent year with burned area during the interval
                most_recent_year_burned = max([burned_area_t_4, burned_area_t_3, burned_area_t_2, burned_area_t_1, burned_area_t])
                burned_in_last_interval = (most_recent_year_burned > 0)

                # Note: Stacking the forest disturbance rasters using ndstack, stack, or flatten outside the pixel iteration did not work with numba.
                # So just reading each raster from the list of rasters separately.
                forest_dist_t_4 = forest_dist_blocks_all_intervals_so_far[-5][row, col]
                forest_dist_t_3 = forest_dist_blocks_all_intervals_so_far[-4][row, col]
                forest_dist_t_2 = forest_dist_blocks_all_intervals_so_far[-3][row, col]
                forest_dist_t_1 = forest_dist_blocks_all_intervals_so_far[-2][row, col]
                forest_dist_t = forest_dist_blocks_all_intervals_so_far[-1][row, col]
                # Most recent year with forest disturbance during the interval
                forest_dist_last = max([forest_dist_t_4, forest_dist_t_3, forest_dist_t_2, forest_dist_t_1, forest_dist_t])

                # if forest_dist_last > 0:
                #     print(forest_dist_last)

                # Records the first year of burned area in the pixel, to indicate whether fire was reported at all
                # in the record
                first_burn_in_record = 0

                # Records the first year of forest disturbance in the pixel, to indicate whether disturbance was reported at all
                # in the record
                first_forest_dist_in_record = 0

                # Loops over burned area pixels since 2001 to see if there was a fire.
                # Stops once a fire is detected because all that matters here is that there was a fire at some point.
                for burned_area_year in burned_area_blocks_all_intervals_so_far:
                    # Update the maximum value for this pixel
                    if burned_area_year[row, col] > 0:
                        first_burn_in_record = burned_area_year[row, col]
                        break

                # if first_burn_in_record > 0:
                #     print("fire", row, col, first_burn_in_record)

                # Loops over forest disturbance pixels since 2001 to see if there was a disturbance.
                # Stops once a disturbance is detected because all that matters here is that there was a disturbance at some point (not the specific year).
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
                soil_c_dens = soil_c_dens_block[row, col]

                # Makes a list of carbon densities to save space in the decision tree below.
                # This list is input to flux calculation functions as one argument, rather than a separate argument
                # for each pool (Mg C/ha)
                c_dens_in = [agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in]

                elevation_cell = elevation_block[row, col]
                climate_domain_cell = climate_domain_block[row, col]
                precipitation_cell = precipitation_block[row, col]

                # Ratios of deadwood C:AGC and litter C:AGC (for deadwood C and litter C removal factors)
                deadwood_c_ratio, litter_c_ratio = nu.calc_deadwood_litter_ratios(elevation_cell,
                                                        climate_domain_cell, precipitation_cell)

                # One-time removal factor for gain of medium-height vegetation (Mg C/ha)
                #TODO Correct and complete this function. Currently using just climate domain as a stand in for IPCC climate zone.
                medium_height_veg_AGC_RF, medium_height_veg_BGC_RF = nu.calc_medium_height_veg_removals(climate_domain_cell)

                # Gef for fire emissions for different gases for forests specifically (g respective gas/kg dry matter)
                Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest = nu.calc_Gef_forest(climate_domain_cell)

                # Cf for fire emissions for all gases for forests specifically (unitless)
                #TODO Eventually decide if we want to revise this from using drivers to using ending LC
                Cf = nu.calc_Cf_forest(climate_domain_cell, drivers_cell, ifl_primary_cell)


                ### Defines specific land cover classes

                # Based on individual canopy height raster
                tree_prev = (veg_h_prev >= cn.tree_threshold)
                tree_curr = (veg_h_curr >= cn.tree_threshold)

                tall_veg_gain = (not tree_prev and tree_curr)
                tall_veg_loss = (tree_prev and not tree_curr)

                # Returns vegetation height classes for start and end of current interval
                short_veg_prev, short_veg_curr, med_veg_prev, med_veg_curr, tall_veg_prev, tall_veg_curr = nu.classify_veg_height(LC_curr, LC_prev)

                SDPT_planted_trees = (planted_forest_type_cell > 0)  # All SDPT planted trees
                SDPT_oil_palm = (planted_forest_type_cell == cn.SDPT_oil_palm_code)  # Oil palm in SDPT planted trees
                oil_palm_after_Descals = (interval_end_year > oil_palm_first_year_cell) and (oil_palm_first_year_cell != 0) # Second condition to exclude NoData (0s) from first year of oil palm
                oil_palm_pre_2000 = (oil_palm_2000_extent_cell == 1)

                all_planted_trees = (SDPT_planted_trees or oil_palm_pre_2000 or oil_palm_after_Descals)
                all_oil_palm = (SDPT_oil_palm or oil_palm_pre_2000 or oil_palm_after_Descals)

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
                sig_height_gain_prev_curr_abs = (height_change_prev_curr <= cn.sig_height_gain_threshold_abs)

                # Whether tall vegetation was partially disturbed in the last interval
                partially_disturbed_in_last_interval = (forest_dist_last > 0) or (sig_height_loss_prev_curr_abs) or (first_time_sig_loss_from_max_height == 1)

                ## Number of years of regrowth for new forest since last time not forest
                years_of_forest_regrowth = years_of_forest_regrowth_block[row, col]

                # Calculates the number of years of forest regrowth since the last year of not-tall vegetation
                # or partial disturbance.
                # Can override the pre-existing value.
                #TODO: This does not seem to work correctly after partial disturbances, at least in primary forest. It doesn't increment the years since disturbance.
                # Look at ArcMap bookmark "Primary forest->partial disturbance->stable forest->stable forest"
                years_of_forest_regrowth = nu.calculate_years_of_forest_regrowth(interval_end_year, most_recent_year_not_tall_veg, tall_veg_curr, partially_disturbed_in_last_interval, years_of_forest_regrowth)

                # Assigns an AGC RF for natural forest based on years since last time not tall vegetation (years_of_forest_regrowth) (Mg AGC/ha/yr).
                # If there are no years of forest regrowth (i.e. no record of non-tall veg land cover), the previous interval's RF is used.
                if years_of_forest_regrowth > 0 and years_of_forest_regrowth <= 5:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_0_5_AGC_RF_cell
                elif years_of_forest_regrowth > 5 and years_of_forest_regrowth <= 10:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_6_10_AGC_RF_cell
                elif years_of_forest_regrowth > 10 and years_of_forest_regrowth <= 15:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_11_15_AGC_RF_cell
                elif years_of_forest_regrowth > 15 and years_of_forest_regrowth <= 20:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_16_20_AGC_RF_cell
                else:
                    natrl_forest_age_dependent_agc_rf = agc_rf_pre_dist_prev


                ### Starting output pixel values

                # Starting decision tree node value
                # All C pool fluxes and densities are kept in Mg C (as opposed to Mg CO2) until they are saved to
                # the output dictionaries. This simplifies arithmetic.
                agc_rf = 0  # (Mg AGC/ha/yr)
                # Need to force arrays into float32 because numba is so particular about datatypes.
                # Initializes dummy output C gross emissions (Mg C/ha/interval->later converted to Mg CO2/ha/yr): AGC, BGC, deadwood C, litter C.
                c_gross_emis_out = np.array([0, 0, 0, 0]).astype('float32')
                # Initializes dummy output C gross removals (Mg C/ha/interval->later converted to Mg CO2/ha/yr): AGC, BGC, deadwood C, litter C.
                c_gross_removals_out = np.array([0, 0, 0, 0]).astype('float32')
                # Initializes dummy output non-CO2 fluxes (Mg CO2e/ha/interval->later converted to Mg CO2e/ha/yr): CH4, N2O
                non_co2_flux_out = np.array([0, 0]).astype('float32')

                node = 0
                gain_year_count = 0

                ### Tree gain
                if tall_veg_gain:  # Non-tree converted to tree (1)    #TODO: Include mangrove exception.
                    node = nu.accrete_node(node, 1)
                    if all_planted_trees:  # New planted trees (11)
                        node = nu.accrete_node(node, 1)
                        if all_oil_palm:  # New oil palm (incl. SDPT) (111)
                            state_out = nu.accrete_node(node, 1)
                            agc_rf = cn.oil_palm_agc_rf
                            bgc_rf = cn.oil_palm_bgc_rf
                            c_gross_emis_out, c_gross_removals_out, c_dens_out, gain_year_count = nu.calc_NT_T(agc_rf, bgc_rf, c_dens_in,
                                                                                                               deadwood_c_ratio=0, litter_c_ratio=0)
                        else: # New non-oil palm planted trees (112)
                            state_out = nu.accrete_node(node, 2)
                            agc_rf = planted_forest_AGC_RF_cell
                            bgc_rf = planted_forest_BGC_RF_cell
                            c_gross_emis_out, c_gross_removals_out, c_dens_out, gain_year_count = nu.calc_NT_T(agc_rf, bgc_rf, c_dens_in,
                                                                                                               deadwood_c_ratio=0, litter_c_ratio=0)
                    else:  # New non-planted trees (12)
                        node = nu.accrete_node(node, 2)
                        if tall_veg_curr:  # New terrestrial natural forest (121)
                            state_out = nu.accrete_node(node, 1)
                            agc_rf = natrl_forest_curve_0_5_AGC_RF_cell
                            bgc_rf = agc_rf * r_s_ratio_cell
                            c_gross_emis_out, c_gross_removals_out, c_dens_out, gain_year_count = nu.calc_NT_T(agc_rf, bgc_rf, c_dens_in,
                                                                            deadwood_c_ratio=deadwood_c_ratio, litter_c_ratio=litter_c_ratio)
                        else:  # New trees outside forests (122)
                            state_out = nu.accrete_node(node, 2)
                            agc_rf = cn.trees_outside_forests_agc_rf_max
                            bgc_rf = agc_rf * r_s_ratio_cell
                            c_gross_emis_out, c_gross_removals_out, c_dens_out, gain_year_count = nu.calc_NT_T(agc_rf, bgc_rf, c_dens_in,
                                                                                                               deadwood_c_ratio=0, litter_c_ratio=0)

                ### Tree loss
                elif tall_veg_loss:  # Tree converted to non-tree (2)    #TODO: Include mangrove exception.
                    node = nu.accrete_node(node, 2)
                    if all_planted_trees:  # Full loss of planted trees (21)
                        node = nu.accrete_node(node, 1)
                        if all_oil_palm:  # Full loss of oil palm (incl. SDPT) (211->2111/2112)
                            node = nu.accrete_node(node, 1)
                            agc_rf = cn.oil_palm_agc_rf
                            bgc_rf = cn.oil_palm_bgc_rf
                            rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                            c_pools_fire_CO2 = cn.agc_emissions_only
                            c_pools_fire_non_CO2 = cn.agc_emissions_only
                            c_pools_no_fire = cn.biomass_emissions_only
                            state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(
                                node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                rf_post_dist, most_recent_year_not_tall_veg, Cf, Gef_ch4_forest, Gef_n2o_forest,
                                deadwood_c_ratio=0, litter_c_ratio=0)
                        else:  # Full loss of non-oil palm planted trees (212)
                            node = nu.accrete_node(node, 2)
                            if LC_curr == cn.cropland:  # Plantation harvested as cropland (2121->21211/21212)
                                node = nu.accrete_node(node, 1)
                                agc_rf = planted_forest_AGC_RF_cell
                                bgc_rf = planted_forest_BGC_RF_cell
                                rf_post_dist = np.array([cn.cropland_rf, 0, 0, 0]).astype('float32')
                                c_pools_fire_CO2 = cn.agc_emissions_only
                                c_pools_fire_non_CO2 = cn.all_but_bgc_emissions
                                c_pools_no_fire = cn.biomass_emissions_only
                                state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(
                                    node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                    c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio=0, litter_c_ratio=0)
                            elif short_veg_curr:  # Plantation harvested as short vegetation (2122->21221/21222)
                                node = nu.accrete_node(node, 2)
                                agc_rf = planted_forest_AGC_RF_cell
                                bgc_rf = planted_forest_BGC_RF_cell
                                rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                c_pools_fire_CO2 = cn.agc_emissions_only
                                c_pools_fire_non_CO2 = cn.all_but_bgc_emissions
                                c_pools_no_fire = cn.biomass_emissions_only
                                state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(
                                    node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                    c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio=0, litter_c_ratio=0)
                            elif med_veg_curr:  # Plantation harvested as medium vegetation (2123->21231/21232)
                                node = nu.accrete_node(node, 3)
                                agc_rf = planted_forest_AGC_RF_cell
                                bgc_rf = planted_forest_BGC_RF_cell
                                rf_post_dist = np.array([medium_height_veg_AGC_RF, medium_height_veg_BGC_RF, 0, 0]).astype('float32')
                                c_pools_fire_CO2 = cn.agc_emissions_only
                                c_pools_fire_non_CO2 = cn.all_but_bgc_emissions
                                c_pools_no_fire = cn.biomass_emissions_only
                                state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(
                                    node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                    c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio=0, litter_c_ratio=0)
                            elif LC_curr == cn.builtup:  # Plantation converted to settlement (2124->21241/21242)
                                node = nu.accrete_node(node, 4)
                                agc_rf = planted_forest_AGC_RF_cell
                                bgc_rf = planted_forest_BGC_RF_cell
                                rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                c_pools_fire_CO2 = cn.agc_emissions_only
                                c_pools_fire_non_CO2 = cn.all_but_bgc_emissions
                                c_pools_no_fire = cn.biomass_emissions_only
                                state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(
                                    node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                    c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio=0, litter_c_ratio=0)
                            else:  # Plantation converted to anything else (2125->21251/21252)
                                node = nu.accrete_node(node, 5)
                                agc_rf = planted_forest_AGC_RF_cell
                                bgc_rf = planted_forest_BGC_RF_cell
                                rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                c_pools_fire_CO2 = cn.agc_emissions_only
                                c_pools_fire_non_CO2 = cn.all_but_bgc_emissions
                                c_pools_no_fire = cn.biomass_emissions_only
                                state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(
                                    node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                    c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio=0, litter_c_ratio=0)
                    else:  # Full loss of non-planted trees (22)
                        node = nu.accrete_node(node, 2)
                        if tall_veg_prev:  # Full loss of natural forest (221)
                            node = nu.accrete_node(node, 1)
                            if LC_curr == cn.cropland:  # Natural forest converted to cropland (2211->22111/22112)
                                node = nu.accrete_node(node, 1)
                                agc_rf = natrl_forest_age_dependent_agc_rf
                                bgc_rf = agc_rf * r_s_ratio_cell
                                rf_post_dist = np.array([cn.cropland_rf, 0, 0, 0]).astype('float32')
                                c_pools_fire_CO2 = cn.all_non_soil_pools
                                c_pools_fire_non_CO2 = cn.all_non_soil_pools
                                c_pools_no_fire = cn.all_non_soil_pools
                                state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(
                                    node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                    c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio, litter_c_ratio)
                            elif short_veg_curr:  # Natural forest converted to short vegetation (2212)
                                node = nu.accrete_node(node, 2)
                                if drivers_cell in cn.drivers_non_soil_C: # Natural forest converted to short vegetation with disturbance that emits all non-soil C pools (22121->221211/221212)
                                    node = nu.accrete_node(node, 1)
                                    agc_rf = natrl_forest_age_dependent_agc_rf
                                    bgc_rf = agc_rf * r_s_ratio_cell
                                    rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                    c_pools_fire_CO2 = cn.all_non_soil_pools
                                    c_pools_fire_non_CO2 = cn.all_non_soil_pools
                                    c_pools_no_fire = cn.all_non_soil_pools
                                    state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(
                                        node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                        c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg, Cf, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio, litter_c_ratio)
                                else:  # Natural forest converted to short vegetation with disturbance that emits biomass C pools only (22122->221221/221222)
                                    node = nu.accrete_node(node, 2)
                                    agc_rf = natrl_forest_age_dependent_agc_rf
                                    bgc_rf = agc_rf * r_s_ratio_cell
                                    rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                    c_pools_fire_CO2 = cn.agc_emissions_only
                                    c_pools_fire_non_CO2 = cn.all_but_bgc_emissions
                                    c_pools_no_fire = cn.biomass_emissions_only
                                    state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(
                                        node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                        c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg, Cf, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio, litter_c_ratio)
                            elif med_veg_curr:  # Natural forest converted to medium vegetation (2213)
                                node = nu.accrete_node(node, 3)
                                if drivers_cell in cn.drivers_non_soil_C: # Natural forest converted to medium vegetation with disturbance that emits all non-soil C pools (22131->221311/221312)
                                    node = nu.accrete_node(node, 1)
                                    agc_rf = natrl_forest_age_dependent_agc_rf
                                    bgc_rf = agc_rf * r_s_ratio_cell
                                    rf_post_dist = np.array([medium_height_veg_AGC_RF, medium_height_veg_BGC_RF, 0, 0]).astype('float32')
                                    c_pools_fire_CO2 = cn.all_non_soil_pools
                                    c_pools_fire_non_CO2 = cn.all_non_soil_pools
                                    c_pools_no_fire = cn.all_non_soil_pools
                                    state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(
                                        node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                        c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg, Cf, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio, litter_c_ratio)
                                else:  # Natural forest converted to medium vegetation with disturbance that emits biomass C pools only (22132->221321/221322)
                                    node = nu.accrete_node(node, 2)
                                    agc_rf = natrl_forest_age_dependent_agc_rf
                                    bgc_rf = agc_rf * r_s_ratio_cell
                                    rf_post_dist = np.array([medium_height_veg_AGC_RF, medium_height_veg_BGC_RF, 0, 0]).astype('float32')
                                    c_pools_fire_CO2 = cn.agc_emissions_only
                                    c_pools_fire_non_CO2 = cn.all_but_bgc_emissions
                                    c_pools_no_fire = cn.biomass_emissions_only
                                    state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(
                                        node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                        c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg, Cf, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio, litter_c_ratio)
                            elif LC_curr == cn.builtup:  # Natural forest converted to settlement (2214->22141/22142)
                                node = nu.accrete_node(node, 4)
                                agc_rf = natrl_forest_age_dependent_agc_rf
                                bgc_rf = agc_rf * r_s_ratio_cell
                                rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                c_pools_fire_CO2 = cn.all_non_soil_pools
                                c_pools_fire_non_CO2 = cn.all_non_soil_pools
                                c_pools_no_fire = cn.all_non_soil_pools
                                state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(
                                    node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                    c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio, litter_c_ratio)
                            else:  # Natural forest converted to anything else (wetland/open water/ice, etc.) (2215->22151/22152)
                                node = nu.accrete_node(node, 5)
                                agc_rf = natrl_forest_age_dependent_agc_rf
                                bgc_rf = agc_rf * r_s_ratio_cell
                                rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                c_pools_fire_CO2 = np.array([0, 0, 0, 0]).astype('float32')   # This particular combination doesn't ever have fire emissions
                                c_pools_fire_non_CO2 = np.array([0, 0, 0, 0]).astype('float32') # This particular combination doesn't ever have fire emissions
                                c_pools_no_fire = cn.biomass_emissions_only
                                burned_in_last_interval = 0  # This particular combination doesn't ever have fire emissions, so this is forced to 0
                                state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(
                                    node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                    c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio, litter_c_ratio)
                        else:  # Full loss of trees outside forests (222->2221/2222)
                            node = nu.accrete_node(node, 2)
                            agc_rf = cn.trees_outside_forests_agc_rf_max
                            bgc_rf = agc_rf * r_s_ratio_cell
                            rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                            c_pools_fire_CO2 = cn.agc_emissions_only
                            c_pools_fire_non_CO2 = cn.agc_emissions_only
                            c_pools_no_fire = cn.biomass_emissions_only
                            state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(
                                node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                rf_post_dist, most_recent_year_not_tall_veg, Cf, Gef_ch4_forest, Gef_n2o_forest,
                                deadwood_c_ratio=0, litter_c_ratio=0)

                ### Trees remaining trees
                elif (tree_prev) and (tree_curr):  # Trees remaining trees (3)    ##TODO: Include mangrove exception.
                    node = nu.accrete_node(node, 3)
                    if partially_disturbed_in_last_interval:  # Trees partially disturbed in the last interval (31)
                        node = nu.accrete_node(node, 1)
                        if all_planted_trees:   # Planted trees partially disturbed in the last interval (311)
                            node = nu.accrete_node(node, 1)
                            if sig_height_gain_prev_curr_abs:  # Planted trees partially disturbed in the last interval with signif. height increase after (3111->31111/31112)
                                node = nu.accrete_node(node, 1)
                                if all_oil_palm:  # Because this planted tree node includes SDPT + non-SDPT oil palm, we need to assign the right RFs
                                    agc_rf = cn.oil_palm_agc_rf
                                    bgc_rf = cn.oil_palm_bgc_rf
                                else:
                                    agc_rf = planted_forest_AGC_RF_cell
                                    bgc_rf = planted_forest_BGC_RF_cell
                                rf_post_dist = np.array([agc_rf, bgc_rf, 0, 0]).astype('float32')
                                c_pools_fire_CO2 = cn.agc_emissions_only
                                c_pools_fire_non_CO2 = cn.agc_emissions_only
                                c_pools_no_fire = np.array([0.5, 0.5, 0, 0]).astype('float32')  ## TODO: Use actual values!
                                state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_T_non_stand_disturbs(
                                    node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                    c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg,
                                    Cf, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest, deadwood_c_ratio=0, litter_c_ratio=0)
                            else:  # Planted trees partially disturbed in the last interval without signif. height increase after (3112->31121/31122)
                                node = nu.accrete_node(node, 2)
                                if all_oil_palm:  # Because this planted tree node includes SDPT + non-SDPT oil palm, we need to assign the right RFs
                                    agc_rf = cn.oil_palm_agc_rf
                                    bgc_rf = cn.oil_palm_bgc_rf
                                else:
                                    agc_rf = planted_forest_AGC_RF_cell
                                    bgc_rf = planted_forest_BGC_RF_cell
                                rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                c_pools_fire_CO2 = cn.agc_emissions_only
                                c_pools_fire_non_CO2 = cn.agc_emissions_only
                                c_pools_no_fire = np.array([0.5, 0.5, 0, 0]).astype('float32')  ## TODO: Use actual values!
                                state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_T_non_stand_disturbs(
                                    node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                    c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg,
                                    Cf, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest, deadwood_c_ratio=0, litter_c_ratio=0)
                        else:  # Non-planted trees partially disturbed in the last interval (312)
                            node = nu.accrete_node(node, 2)
                            if tall_veg_curr:  # Forest partially disturbed in the last interval (3121)
                                node = nu.accrete_node(node, 1)
                                if sig_height_gain_prev_curr_abs:  # Forest partially disturbed in the last interval with signif. height increase after (31211->312111/312112)
                                    node = nu.accrete_node(node, 1)
                                    agc_rf = natrl_forest_age_dependent_agc_rf
                                    bgc_rf = agc_rf * r_s_ratio_cell
                                    agc_rf_post = natrl_forest_curve_0_5_AGC_RF_cell # Post-dist RF is 0-5 year secondary forest
                                    bgc_rf_post = agc_rf_post * r_s_ratio_cell
                                    rf_post_dist = np.array([agc_rf_post, bgc_rf_post, 0, 0]).astype('float32')
                                    c_pools_fire_CO2 = cn.agc_emissions_only
                                    c_pools_fire_non_CO2 = cn.agc_emissions_only
                                    c_pools_no_fire = np.array([0.5, 0.5, 0, 0]).astype('float32')  ## TODO: Use actual values!
                                    state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_T_non_stand_disturbs(
                                        node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                        c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg,
                                        Cf, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio=deadwood_c_ratio, litter_c_ratio=litter_c_ratio)
                                else:  # Forest partially disturbed in the last interval without signif. height increase after (31212->312121/312122)
                                    node = nu.accrete_node(node, 2)
                                    agc_rf = natrl_forest_age_dependent_agc_rf
                                    bgc_rf = agc_rf * r_s_ratio_cell
                                    rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')  # No post-disturbance RFs or removals
                                    c_pools_fire_CO2 = cn.agc_emissions_only
                                    c_pools_fire_non_CO2 = cn.agc_emissions_only
                                    c_pools_no_fire = np.array([0.5, 0.5, 0, 0]).astype('float32')  ## TODO: Use actual values!
                                    state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_T_non_stand_disturbs(
                                        node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                        c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg,
                                        Cf, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio=deadwood_c_ratio, litter_c_ratio=litter_c_ratio)
                            else:  # Trees outside forests partially disturbed in the last interval (3122)
                                node = nu.accrete_node(node, 2)
                                if sig_height_gain_prev_curr_abs:  # Trees outside forests partially disturbed in the last interval with signif. height increase after (31221->312211/312212)
                                    node = nu.accrete_node(node, 1)
                                    agc_rf = cn.trees_outside_forests_agc_rf_max
                                    bgc_rf = agc_rf * r_s_ratio_cell
                                    agc_rf_post = cn.trees_outside_forests_agc_rf_max
                                    bgc_rf_post = agc_rf_post * r_s_ratio_cell
                                    rf_post_dist = np.array([agc_rf_post, bgc_rf_post, 0, 0]).astype('float32')
                                    c_pools_fire_CO2 = cn.agc_emissions_only
                                    c_pools_fire_non_CO2 = cn.agc_emissions_only
                                    c_pools_no_fire = np.array([0.5, 0.5, 0, 0]).astype('float32')  ## TODO: Use actual values!
                                    state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_T_non_stand_disturbs(
                                        node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                        c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg,
                                        Cf, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio=0, litter_c_ratio=0)
                                else:  # Trees outside forests partially disturbed in the last interval with signif. height increase after (31222->312221/312222)
                                    node = nu.accrete_node(node, 2)
                                    agc_rf = cn.trees_outside_forests_agc_rf_max
                                    bgc_rf = agc_rf * r_s_ratio_cell
                                    rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')  # No post-disturbance RFs or removals
                                    c_pools_fire_CO2 = cn.agc_emissions_only
                                    c_pools_fire_non_CO2 = cn.agc_emissions_only
                                    c_pools_no_fire = np.array([0.5, 0.5, 0, 0]).astype('float32')  ## TODO: Use actual values!
                                    state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_T_non_stand_disturbs(
                                        node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                        c_pools_no_fire, forest_dist_last, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg,
                                        Cf, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio=0, litter_c_ratio=0)
                    else:  # Trees not disturbed in the last interval (32)
                        node = nu.accrete_node(node, 2)
                        if all_planted_trees:  # Planted trees not disturbed in the last interval (321->3211/3212)
                            node = nu.accrete_node(node, 1)
                            if all_oil_palm:  # Because this planted tree node includes SDPT + non-SDPT oil palm, we need to assign the right RFs
                                agc_rf = cn.oil_palm_agc_rf
                                bgc_rf = cn.oil_palm_bgc_rf
                            else:
                                agc_rf = planted_forest_AGC_RF_cell
                                bgc_rf = planted_forest_BGC_RF_cell
                            c_pools_fire_CO2 = cn.agc_emissions_only
                            c_pools_fire_non_CO2 = cn.agc_emissions_only
                            state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_T_no_disturbs(
                                node, most_recent_year_burned, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                interval_end_year, c_dens_in, most_recent_year_not_tall_veg,
                                Cf, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest, deadwood_c_ratio=0, litter_c_ratio=0)
                        else:  # Non-planted trees not disturbed in last interval (322)
                            node = nu.accrete_node(node, 2)
                            if tall_veg_curr:  # Forest not disturbed in last interval (3221)
                                node = nu.accrete_node(node, 1)
                                if first_forest_dist_in_record or (first_time_sig_loss_from_max_height > 0) or (most_recent_year_not_tall_veg > 0): # Young secondary natural forest (32211->322111/322112)
                                    node = nu.accrete_node(node, 1)
                                    # Because the pixel had a stand-replacing or non-stand-replacing disturbance at some point
                                    # it shouldn't use primary forest or old secondary forest RF anymore.
                                    # This replaces those RFs with a young secondary forest RF.
                                    # +/- 0.01 the primary forest and old secondary RF are to provide some tolerance around
                                    # those RFs in case numba is rounding them and they aren't exact matches.
                                    if (natrl_forest_age_dependent_agc_rf > primary_forest_AGC_RF - 0.01) and (natrl_forest_age_dependent_agc_rf < primary_forest_AGC_RF + 0.01):
                                        agc_rf = natrl_forest_curve_0_5_AGC_RF_cell
                                    elif (natrl_forest_age_dependent_agc_rf > natrl_forest_curve_21_100_AGC_RF_cell - 0.01) and (natrl_forest_age_dependent_agc_rf < natrl_forest_curve_21_100_AGC_RF_cell + 0.01):
                                        agc_rf = natrl_forest_curve_0_5_AGC_RF_cell
                                    else:  # If not using primary or old secondary RF, it can use whatever the relevant young secondary RF is
                                        agc_rf = natrl_forest_age_dependent_agc_rf
                                    bgc_rf = agc_rf * r_s_ratio_cell
                                    c_pools_fire_CO2 = cn.agc_emissions_only
                                    c_pools_fire_non_CO2 = cn.agc_emissions_only
                                    state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_T_no_disturbs(
                                        node, most_recent_year_burned, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                        interval_end_year, c_dens_in, most_recent_year_not_tall_veg,
                                        Cf, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio=deadwood_c_ratio, litter_c_ratio=litter_c_ratio)
                                else:  # Natural forest undisturbed since 2000 (32212)
                                    node = nu.accrete_node(node, 2)
                                    if ifl_primary_cell:  # Primary forest (322121->3221211/3221212)
                                        node = nu.accrete_node(node, 1)
                                        agc_rf = primary_forest_AGC_RF
                                        bgc_rf = agc_rf * r_s_ratio_cell
                                        c_pools_fire_CO2 = cn.agc_emissions_only
                                        c_pools_fire_non_CO2 = cn.agc_emissions_only
                                        state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_T_no_disturbs(
                                            node, most_recent_year_burned, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                            interval_end_year, c_dens_in, most_recent_year_not_tall_veg,
                                            Cf, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                            deadwood_c_ratio=deadwood_c_ratio, litter_c_ratio=litter_c_ratio)
                                    else:  # Old secondary forest (322122->3221221/3221222)
                                        node = nu.accrete_node(node, 2)
                                        agc_rf = natrl_forest_curve_21_100_AGC_RF_cell
                                        bgc_rf = agc_rf * r_s_ratio_cell
                                        c_pools_fire_CO2 = cn.agc_emissions_only
                                        c_pools_fire_non_CO2 = cn.agc_emissions_only
                                        state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_T_no_disturbs(
                                            node, most_recent_year_burned, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                            interval_end_year, c_dens_in, most_recent_year_not_tall_veg,
                                            Cf, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                            deadwood_c_ratio=deadwood_c_ratio, litter_c_ratio=litter_c_ratio)
                            else:  # Trees outside forests not disturbed in the last interval (3222->32221/32222)
                                node = nu.accrete_node(node, 2)
                                agc_rf = cn.trees_outside_forests_agc_rf_max
                                bgc_rf = agc_rf * r_s_ratio_cell
                                c_pools_fire_CO2 = cn.agc_emissions_only
                                c_pools_fire_non_CO2 = cn.agc_emissions_only
                                state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_T_no_disturbs(
                                    node, most_recent_year_burned, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                                    interval_end_year, c_dens_in, most_recent_year_not_tall_veg,
                                    Cf, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest, deadwood_c_ratio=0, litter_c_ratio=0)

                # When decision trees above do not apply
                else:
                    # Need to know when a state isn't being assigned. It should always be assigned.
                    state_out = 4000000000

                    # If no C fluxes calculated in decision tree, densities out should be densities in (Mg C/ha).
                    # Otherwise, they get reset to 0.
                    c_dens_out = np.array(c_dens_in).astype('float32')


                ### Populates the output arrays with the calculated fluxes and densities
                state_out_block[row, col] = state_out

                # Sets the RF from this interval as the RF from the previous interval for the next interval (Mg AGC/ha/yr)
                agc_rf_pre_dist_out_block[row, col] = agc_rf

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
                gain_year_count_out_block[row, col] = gain_year_count
                most_recent_year_not_tall_veg_block[row, col] = most_recent_year_not_tall_veg
                years_of_forest_regrowth_block[row, col] = years_of_forest_regrowth
                max_height_since_last_time_not_tall_veg_block[row, col] = max_height_since_last_time_not_tall_veg
                first_time_sig_loss_from_max_height_block[row, col] = first_time_sig_loss_from_max_height

        # os.quit()   # For testing the first interval

        ### End of iteration calculations and outputs

        # Adds the output arrays to the dictionary with the appropriate data type
        # Outputs need .copy() so that previous intervals' arrays in dictionary aren't overwritten because arrays in dictionaries are mutable (courtesy of ChatGPT).
        # This applies even for the outputs that aren't reused in the next interval;
        # they will still get overwritten with the final interval's values, I believe.
        year_range = f"{interval_end_year - cn.interval_years}_{interval_end_year}"

        out_dict_uint32[f"{cn.land_state_pattern}_{year_range}"] = state_out_block.copy()

        out_dict_float32[f"{cn.agc_rf_pre_dist_pattern}_{year_range}"] = agc_rf_pre_dist_out_block.copy()

        # Converts carbon pool fluxes from Mg C/ha/interval to Mg CO2/ha/yr.
        # Gross emissions are positive. Gross removals are negative.
        out_dict_float32[f"{cn.agc_gross_emis_pattern}_{year_range}"] = (agc_gross_emis_out_block*cn.C_to_CO2_numba/cn.interval_years).copy()
        out_dict_float32[f"{cn.bgc_gross_emis_pattern}_{year_range}"] = (bgc_gross_emis_out_block*cn.C_to_CO2_numba/cn.interval_years).copy()
        out_dict_float32[f"{cn.deadwood_c_gross_emis_pattern}_{year_range}"] = (deadwood_c_gross_emis_out_block*cn.C_to_CO2_numba/cn.interval_years).copy()
        out_dict_float32[f"{cn.litter_c_gross_emis_pattern}_{year_range}"] = (litter_c_gross_emis_out_block*cn.C_to_CO2_numba/cn.interval_years).copy()

        out_dict_float32[f"{cn.agc_gross_removals_pattern}_{year_range}"] = (agc_gross_removals_out_block*cn.C_to_CO2_numba/cn.interval_years).copy()
        out_dict_float32[f"{cn.bgc_gross_removals_pattern}_{year_range}"] = (bgc_gross_removals_out_block*cn.C_to_CO2_numba/cn.interval_years).copy()
        out_dict_float32[f"{cn.deadwood_c_gross_removals_pattern}_{year_range}"] = (deadwood_c_gross_removals_out_block*cn.C_to_CO2_numba/cn.interval_years).copy()
        out_dict_float32[f"{cn.litter_c_gross_removals_pattern}_{year_range}"] = (litter_c_gross_removals_out_block*cn.C_to_CO2_numba/cn.interval_years).copy()

        # Converts non-CO2 emissions from Mg CO2e/ha/interval to Mg CO2e/ha/yr
        out_dict_float32[f"{cn.ch4_flux_pattern}_{year_range}"] = (ch4_gross_emis_out_block/cn.interval_years).copy()
        out_dict_float32[f"{cn.n2o_flux_pattern}_{year_range}"] = (n2o_gross_emis_out_block/cn.interval_years).copy()

        # Still Mg C/ha
        out_dict_float32[f"{cn.agc_dens_pattern}_{interval_end_year}"] = agc_dens_block.copy()
        out_dict_float32[f"{cn.bgc_dens_pattern}_{interval_end_year}"] = bgc_dens_block.copy()
        out_dict_float32[f"{cn.deadwood_c_dens_pattern}_{interval_end_year}"] = deadwood_c_dens_block.copy()
        out_dict_float32[f"{cn.litter_c_dens_pattern}_{interval_end_year}"] = litter_c_dens_block.copy()

        # Test/intermediate outputs only saved if not a large run
        if not is_final:
            out_dict_uint8[f"{cn.gain_year_count_pattern}_{year_range}"] = gain_year_count_out_block.copy()
            out_dict_uint16[f"{cn.most_recent_year_not_tall_veg}_{cn.first_model_year}_{interval_end_year}"] = most_recent_year_not_tall_veg_block.copy()    # Years represent from model start to current interval end
            out_dict_uint8[f"{cn.years_of_forest_regrowth}_{interval_end_year}"] = years_of_forest_regrowth_block.copy()
            out_dict_uint16[f"{cn.year_of_forest_loss}_{year_range}"] = year_of_forest_loss_block.copy()
            out_dict_uint8[f"{cn.max_height_since_last_time_not_tall_veg}_{year_range}"] = max_height_since_last_time_not_tall_veg_block.copy()
            out_dict_uint8[f"{cn.first_time_sig_loss_from_max_height_block}_{year_range}"] = first_time_sig_loss_from_max_height_block.copy()

    return out_dict_uint8, out_dict_uint16, out_dict_uint32, out_dict_float32


# Downloads inputs, prepares data, calculates LULUCF stocks and fluxes, and uploads outputs to s3
def calculate_and_upload_LULUCF_fluxes(bounds, primary_forest_RFs, download_dict_with_data_types, fishnet_iso_df, is_final, no_upload):

    logger = lu.setup_logging()

    bounds_str = uu.boundstr(bounds)  # String form of chunk bounds, from e.g., [8, -1, 9, 0] to 8_-1_9_0
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []


    ### Part 1: Checks if tile exists at all, downloads data in chunk if it does exist, and checks if chunk actually has relevant data.
    ### I haven't figured out a good way to check if the chunk has relevant data before downloading,
    ### so inputs are downloaded and then checked.

    # Replaces the placeholder tile_id in the download data dictionary from main with the tile_id for this chunk
    updated_download_dict = uu.replace_tile_id_in_dict(download_dict_with_data_types, tile_id)

    # If a particular tile doesn't exist for an input, an array of 0s of the correct size and datatype is returned instead.
    # Thus, this returns a complete set of inputs (missing chunks filled).
    # Note: If running in a local Dask cluster, prints to console may be duplicated. Doesn't happen with a Coiled cluster of the same size (1 worker).
    # Seems to be a problem with local Dask getting overwhelmed by so many futures being created and downloaded from s3.
    futures = uu.prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_final, logger)
    # print(futures)

    # Only prints if not a final run
    if not is_final:
        lu.print_and_log(f"Waiting for requests for data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger)

    # Dictionary that stores the downloaded data
    layers = {}

    # Waits for requests to come back with data from S3
    for future in concurrent.futures.as_completed(futures):
        layer = futures[future]
        layers[layer] = future.result()

    # Test prints
    # print(layers)
    # print(layers['burned_area_2002'].max())
    # print(layers[cn.planted_forest_AGC_BGC_removal_factor_pattern])
    # print(layers[cn.planted_forest_AGC_BGC_removal_factor_pattern].max())
    # print(layers[soil_c_2000_pattern].dtype)


    ### Part 2: Calculates min, mean, and max for each input chunk.
    ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.

    # Calculates stats for the input layers
    for key, array in layers.items():
        chunk_stats.append(uu.calculate_stats(array, key, bounds_str, tile_id, 'input_layer', fishnet_iso_df))
    # print(stats)


    ### Part 3: Creates a separate dictionary for each chunk datatype so that they can be passed to Numba as separate arguments.
    ### Numba functions can accept (and return) dictionaries of arrays as long as each dictionary only has arrays of one data type (e.g., uint8, float32).
    ### Note: need to add new code if inputs with other data types are added

    # Only prints if not a final run
    if not is_final:
        lu.print_and_log(f"Creating typed dictionaries for chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger)

    # Creates the typed dictionaries for all input layers (including those that originally had no data)
    typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32 = nu.create_typed_dicts(layers)

    # print("uint8_typed_list:", typed_dict_uint8)
    # print("int16_typed_list:", typed_dict_int16)
    # print("int32_typed_list:", typed_dict_int32)
    # print("float32_typed_list:", out_dict_float32)


    ### Part 4: Calculates LULUCF fluxes and densities

    lu.print_and_log(f"Calculating LULUCF fluxes and carbon densities in {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger)

    out_dict_uint8, out_dict_uint16, out_dict_uint32, out_dict_float32 = LULUCF_fluxes(typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32, primary_forest_RFs, is_final)

    lu.print_and_log(f"Done calculating LULUCF fluxes and carbon densities in {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger)

    # print(out_dict_uint32)
    # print(out_dict_float32)
    # print(f"Average of {list(out_dict_uint32.keys())[0]} is: {list(out_dict_uint32.values())[0].mean()}")

    # Fresh non-Numba-constrained dictionary that stores all numpy arrays.
    # The dictionaries by datatype that are returned from the numba function have limitations on them,
    # e.g., they can't be combined with other datatypes. This prevents the addition of attributes needed for uploading to s3.
    # So the trick here is to copy the numba-exported arrays into normal Python arrays to which we can do anything in Python.
    out_dict_all_dtypes = {}

    # Transfers the dictionaries of numpy arrays for each data type to a new, Pythonic array
    out_dicts = [out_dict_uint8, out_dict_uint16, out_dict_uint32, out_dict_float32]

    # Loop through each dictionary and update out_dict_all_dtypes
    for out_dict in out_dicts:
        for key, value in out_dict.items():
            out_dict_all_dtypes[key] = value

        # Clear memory of unneeded arrays
        del out_dict


    ### Part 6: Calculates combined gross fluxes and net fluxes.
    ### Doing this outside numba function to minimize pixel-level calculations and chunks being returned by numba function.

    # Deletes all unnecessary input dictionaries before the memory-intensive derived output calculations
    # Suggested by ChatGPT: https://chatgpt.com/share/e/672bbf2e-ebbc-800a-aae3-3d92f5a1d663
    in_dicts = [layers, typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32]
    [in_dict.clear() for in_dict in in_dicts]

    for interval_end_year in cn.interval_end_years:

        year_range = f"{interval_end_year-5}_{interval_end_year}"

        # Gross emissions across all carbon pools
        out_dict_all_dtypes[f"{cn.gross_emis_all_C_pools_CO2_only_pattern}_{year_range}"] = (
                out_dict_all_dtypes[f"{cn.agc_gross_emis_pattern}_{year_range}"]
                + out_dict_all_dtypes[f"{cn.bgc_gross_emis_pattern}_{year_range}"]
                + out_dict_all_dtypes[f"{cn.deadwood_c_gross_emis_pattern}_{year_range}"]
                + out_dict_all_dtypes[f"{cn.litter_c_gross_emis_pattern}_{year_range}"])

        # Gross emissions for non-CO2 emissions
        out_dict_all_dtypes[f"{cn.gross_emis_non_CO2_only_pattern}_{year_range}"] = (
                out_dict_all_dtypes[f"{cn.ch4_flux_pattern}_{year_range}"]
                + out_dict_all_dtypes[f"{cn.n2o_flux_pattern}_{year_range}"])

        # Gross emissions for all carbon pools and all gases
        out_dict_all_dtypes[f"{cn.gross_emis_all_C_pools_all_gases_pattern}_{year_range}"] = (
            out_dict_all_dtypes[f"{cn.gross_emis_all_C_pools_CO2_only_pattern}_{year_range}"]
            + out_dict_all_dtypes[f"{cn.gross_emis_non_CO2_only_pattern}_{year_range}"]
        )

        # Gross removals across all carbon pools
        out_dict_all_dtypes[f"{cn.gross_removals_all_C_pools_pattern}_{year_range}"] = (
                out_dict_all_dtypes[f"{cn.agc_gross_removals_pattern}_{year_range}"]
                + out_dict_all_dtypes[f"{cn.bgc_gross_removals_pattern}_{year_range}"]
                + out_dict_all_dtypes[f"{cn.deadwood_c_gross_removals_pattern}_{year_range}"]
                + out_dict_all_dtypes[f"{cn.litter_c_gross_removals_pattern}_{year_range}"])

        # Net flux for each carbon pool
        out_dict_all_dtypes[f"{cn.agc_net_flux_pattern}_{year_range}"] = out_dict_all_dtypes[f"{cn.agc_gross_emis_pattern}_{year_range}"] + out_dict_all_dtypes[f"{cn.agc_gross_removals_pattern}_{year_range}"]
        out_dict_all_dtypes[f"{cn.bgc_net_flux_pattern}_{year_range}"] = out_dict_all_dtypes[f"{cn.bgc_gross_emis_pattern}_{year_range}"] + out_dict_all_dtypes[f"{cn.bgc_gross_removals_pattern}_{year_range}"]
        out_dict_all_dtypes[f"{cn.deadwood_c_net_flux_pattern}_{year_range}"] = out_dict_all_dtypes[f"{cn.deadwood_c_gross_emis_pattern}_{year_range}"] + out_dict_all_dtypes[f"{cn.deadwood_c_gross_removals_pattern}_{year_range}"]
        out_dict_all_dtypes[f"{cn.litter_c_net_flux_pattern}_{year_range}"] = out_dict_all_dtypes[f"{cn.litter_c_gross_emis_pattern}_{year_range}"] + out_dict_all_dtypes[f"{cn.litter_c_gross_removals_pattern}_{year_range}"]

        # Net flux across all carbon pools but for CO2 only
        out_dict_all_dtypes[f"{cn.net_flux_all_C_pools_CO2_only_pattern}_{year_range}"] = (
                out_dict_all_dtypes[f"{cn.agc_net_flux_pattern}_{year_range}"]
                + out_dict_all_dtypes[f"{cn.bgc_net_flux_pattern}_{year_range}"]
                + out_dict_all_dtypes[f"{cn.deadwood_c_net_flux_pattern}_{year_range}"]
                + out_dict_all_dtypes[f"{cn.litter_c_net_flux_pattern}_{year_range}"])

        # Net flux across all carbon pools, plus non-pool non-CO2 emissions
        out_dict_all_dtypes[f"{cn.net_flux_all_C_pools_all_gases_pattern}_{year_range}"] = (
                out_dict_all_dtypes[f"{cn.net_flux_all_C_pools_CO2_only_pattern}_{year_range}"]
                + out_dict_all_dtypes[f"{cn.gross_emis_non_CO2_only_pattern}_{year_range}"])


    ### Part 7: Calculates per ha min, per ha mean, per ha max, and per pixel sum for each output chunk.
    ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.
    ### Also useful for a quick sum of outputs without doing zonal stats

    # The relevant pixel area (m^2) file in s3
    pixel_area_uri = f"{cn.pixel_area_path}{cn.pixel_area_pattern}_{tile_id}.tif"

    # Gets numpy arrays of the model output being analyzed and the area (m^2) per pixel
    pixel_area_chunk = uu.get_tile_dataset_rio(pixel_area_uri, 'Float32', bounds, chunk_length_pixels, is_final, logger)

    # Calculates stats for the output layers from create_starting_C_densities as a dictionary with chunk attributes
    for key, array_per_ha in out_dict_all_dtypes.items():

        # Converts per hectare values to per pixel values for the output numpy array
        output_per_pixel = array_per_ha * pixel_area_chunk * cn.m2_to_ha

        chunk_stats.append(uu.calculate_stats(array_per_ha, key, bounds_str, tile_id, 'output_layer', output_per_pixel))


    ### Part 8: Saves numpy arrays as rasters and uploads to s3

    # Only saves arrays to geotifs and uploads them to s3 if enabled
    if not no_upload:

        out_no_data_val = 0  # NoData value for output raster (optional)

        # Adds metadata used for uploading outputs to s3 to the dictionary
        for key, value in out_dict_all_dtypes.items():
            data_type = value.dtype.name

            # Retrieves the file name pattern and date(s) covered for the output file for use in s3 folder construction
            out_pattern, year_range = uu.strip_and_extract_years(key)

            # Dictionary with metadata for each array
            out_dict_all_dtypes[key] = [value, data_type, out_pattern, year_range]

        uu.save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str, out_dict_all_dtypes,
                                            is_final, logger, out_no_data_val)

    # Clears memory of unneeded arrays
    del out_dict_all_dtypes

    success_message = f"Success for {bounds_str}: {uu.timestr()}"
    return success_message, chunk_stats  # Return both the success message and the statistics


def main(cluster_name, run_local=False, no_stats=False, no_log=False, no_upload=False,
         bounding_box=None, chunk_size=None, chunk_list=None, first_chunks=None):

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Model stage being running
    stage = 'LULUCF_fluxes'

    # Starting time for stage
    start_time = uu.timestr()
    print(f"Stage {stage} started at: {start_time}")  #TODO in all main() functions, add print statements to log

    # Returns a dataframe of chunk_id and ISO for the GADM3.6 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso()

    # Makes list of chunks to analyze from the bounding box and chunk size (deg)
    # Outut list form is [[115.25, -3.75, 115.5, -3.5], [...], [...], ...]
    if bounding_box and chunk_size:

        print("Using bounding box and chunk size to determine chunks")
        chunks = uu.get_chunk_bounds_from_bounding_box(bounding_box, chunk_size)

    # Makes list of chunks to analyze from a shapefile attribute table.
    # Attribute table column must be formatted as W_S_E_N.
    # Output list form is [[115.25, -3.75, 115.5, -3.5], [...], [...], ...]
    elif chunk_list:

        print("Using chunk list shapefile (and optional number of test chunks) to determine 1x1 deg chunks")

        # gdf = gpd.read_file(cn.fishnet_s3_uri)  # Reads shapefile attribute table
        fishnet_1x1_chunk_id_df = fishnet_iso_df[['chunk_id']]  # Creates dataframe

        # If argument for number of chunks in shapefile is supplied, limit to that
        if first_chunks:
            fishnet_1x1_chunk_id_df = fishnet_1x1_chunk_id_df[:first_chunks]

        # Converts dataframe column of chunk bounds to nested list
        # Per https://chatgpt.com/share/e/674747ee-d588-800a-995c-1f897a8ace31
        chunks = fishnet_1x1_chunk_id_df['chunk_id'].apply(uu.process_chunk_id).tolist()

    else:
        print("Chunk list cannot be determined")
        sys.exit()

    print(f"Processing {len(chunks)} chunks")

    # Determines if the output file names for final versions of outputs should be used
    is_final = False
    if len(chunks) > 20:
        is_final = True
        print("Running as final model.")

    # Accumulates all statistics and output messages from chunk analysis
    # From https://chatgpt.com/share/e/5599b6b0-1aaa-4d54-98d3-c720a436dd9a
    all_stats = []
    return_messages = []

    # This is just a placeholder tile_id that is used to obtain the datatype of each tile set.
    # It is overwritten when chunks are assigned and analyzed.
    # Using this placeholder allows the full path and tile name to be specified up front, which simplifies things.
    # Otherwise, we'd have just the path but not the file name now and would have to add in the file name later
    # (probably at the chunk level).
    sample_tile_id = "00N_000E"

    # Dictionary of data to download (inputs to model)
    download_dict = {

        cn.agc_2000_pattern: f"{cn.agc_2000_path}{sample_tile_id}__{cn.agc_2000_pattern}.tif",
        cn.bgc_2000_pattern: f"{cn.bgc_2000_path}{sample_tile_id}__{cn.bgc_2000_pattern}.tif",
        cn.deadwood_c_2000_pattern: f"{cn.deadwood_c_2000_path}{sample_tile_id}__{cn.deadwood_c_2000_pattern}.tif",
        cn.litter_c_2000_pattern: f"{cn.litter_c_2000_path}{sample_tile_id}__{cn.litter_c_2000_pattern}.tif",
        cn.soil_c_2000_pattern: f"{cn.soil_c_2000_path}{sample_tile_id}_{cn.soil_c_2000_pattern}.tif",

        cn.r_s_ratio_pattern: f"{cn.r_s_ratio_path}{sample_tile_id}_{cn.r_s_ratio_pattern}.tif",

        cn.drivers_pattern: f"{cn.drivers_path}{sample_tile_id}_{cn.drivers_pattern}.tif",  #TODO update to latest version

        cn.planted_forest_type_pattern: f"{cn.planted_forest_type_path}{sample_tile_id}_{cn.planted_forest_type_pattern}.tif",
        cn.planted_forest_AGC_removal_factor_pattern: f"{cn.planted_forest_AGC_removal_factor_path}{sample_tile_id}_{cn.planted_forest_AGC_removal_factor_pattern}.tif",
        cn.planted_forest_AGC_BGC_removal_factor_pattern: f"{cn.planted_forest_AGC_BGC_removal_factor_path}{sample_tile_id}_{cn.planted_forest_AGC_BGC_removal_factor_pattern}.tif",
        cn.oil_palm_2000_extent_pattern: f"{cn.oil_palm_2000_extent_path}{sample_tile_id}_{cn.oil_palm_2000_extent_pattern}.tif",
        cn.oil_palm_first_year_pattern: f"{cn.oil_palm_first_year_path}{cn.oil_palm_first_year_pattern}_{sample_tile_id}.tif",   # Pattern is before tile_id for this input
        # Originally from gfw-data-lake, so it's in 400x400 windows
        cn.planted_forest_tree_crop_pattern: f"{cn.planted_forest_tree_crop_path}{sample_tile_id}.tif",

        # Originally from gfw-data-lake, so it's in 400x400 windows
        cn.organic_soil_extent_pattern: f"{cn.organic_soil_extent_path}{sample_tile_id}_{cn.organic_soil_extent_pattern}.tif",
        cn.elevation_pattern: f"{cn.elevation_path}{sample_tile_id}_{cn.elevation_pattern}.tif",
        cn.climate_domain_pattern: f"{cn.climate_domain_path}{sample_tile_id}_{cn.climate_domain_pattern}.tif",
        cn.climate_zone_pattern: f"{cn.climate_zone_path}{sample_tile_id}_{cn.climate_zone_pattern}.tif",
        cn.precipitation_pattern: f"{cn.precipitation_path}{sample_tile_id}_{cn.precipitation_pattern}.tif",
        # "ecozone": f"s3://gfw2-data/fao_ecozones/v2000/raster/epsg-4326/10/40000/class/gdal-geotiff/{sample_tile_id}.tif",   # Originally from gfw-data-lake, so it's in 400x400 windows
        # "iso": f"s3://gfw2-data/gadm_administrative_boundaries/v3.6/raster/epsg-4326/10/40000/adm0/gdal-geotiff/{sample_tile_id}.tif",  # Originally from gfw-data-lake, so it's in 400x400 windows
        cn.ifl_primary_pattern: f"{cn.ifl_primary_path}{sample_tile_id}_{cn.ifl_primary_pattern}.tif",
        cn.continent_ecozone_pattern: f"{cn.continent_ecozone_path}{sample_tile_id}_{cn.continent_ecozone_pattern}.tif",
        cn.pixel_area_pattern: f"{cn.pixel_area_path}{cn.pixel_area_pattern}_{sample_tile_id}.tif"
    }

    # Land cover and vegetation height rasters (5-year intervals)
    for year in range(cn.first_model_year, cn.last_model_year + 1, cn.interval_years):
        download_dict[f"{cn.land_cover_pattern}_{year}"] = f"{cn.land_cover_path}{year}/raw/{sample_tile_id}.tif"
        download_dict[f"{cn.vegetation_height_pattern}_{year}"] = f"{cn.vegetation_height_path}{year}/{sample_tile_id}_{cn.vegetation_height_pattern}_{year}.tif"

    # Burned area rasters (annual)
    # All years need to be in their own folder
    for year in range(cn.first_model_year, cn.last_model_year + 1):  # Annual burned area maps start in 2000
        download_dict[f"{cn.burned_area_pattern}_{year}"] = f"{cn.burned_area_path}{year}/{cn.burned_area_pattern}_{year}_{sample_tile_id}.tif"

    # Forest disturbance rasters (annual)
    # All years need to be in their own folder
    for year in range(cn.first_model_year + 1, cn.last_model_year + 1):  # Annual forest disturbance maps start in 2001 and ends in 2020
        download_dict[f"{cn.forest_disturbance_layer_name}_{year}"] = f"{cn.annual_forest_disturbance_path}{year}/{year}_{sample_tile_id}.tif"

    # Young natural forest rasters (several age intervals)
    # Each growth interval's rate is in its own folder
    for growth_interval in cn.natural_forest_growth_curve_intervals:
        download_dict[f"{cn.natural_forest_growth_curve_pattern}__{growth_interval}_years"] = f"{cn.natural_forest_growth_curve_path}rate_{growth_interval}/{sample_tile_id}_{cn.natural_forest_growth_curve_pattern}__{growth_interval}_years.tif"

    # Returns the first tile in each input so that the datatype can be determined.
    # This is done up front, once per tile set, rather than on each chunk, since
    # all tiles have the same datatype for each input-- it only needs to be done once at the very beginning of the stage.
    print(f"Getting tile_id of first tile in each tile set: {uu.timestr()}")
    first_tiles = uu.first_file_name_in_s3_folder(download_dict)

    # Creates a download dictionary with the datatype of each input in the values.
    # This is supplied to each chunk that is being analyzed.
    # This also serves as a check of whether all inputs are being found (s3 paths correct)
    print(f"Getting datatype of first tile in each tile set: {uu.timestr()}")
    download_dict_with_data_types = uu.add_file_type_to_dict(first_tiles)

    # Creates numpy array of IPCC Tier 1 primary forest removal factors by continent-ecozone combination.
    # Needs to by a numpy array for the numba function to use it.
    # Inputs are Mg AGB/ha/yr. Outputs are Mg AGB/ha/yr. Conversion to Mg AGC/ha/yr is done below.
    primary_forest_RFs = uu.convert_lookup_table_to_array(cn.IPCC_removal_factor_table_full_path,
                                                          cn.IPCC_removal_factor_table_tab,
                                                          ['gainEcoCon', 'growth_primary'])

    # Converts primary forest AGB RFs to AGC RFs (Mg AGB/ha/yr -> Mg AGC/ha/yr)
    primary_forest_RFs[:, 1] = primary_forest_RFs[:, 1] * cn.biomass_to_carbon_non_mangrove

    # Creates list of tasks to run (1 task = 1 chunk)
    print(f"Creating tasks and starting processing: {uu.timestr()}")

    # This approach handles large task lists (graphs) better than [dask.delayed(calculate_and_upload_LULUCF_fluxes ... )]
    futures = []
    for chunk in chunks:
        future = client.submit(calculate_and_upload_LULUCF_fluxes, chunk, primary_forest_RFs,
                               download_dict_with_data_types, fishnet_iso_df, is_final, no_upload)
        futures.append(future)

    # Collect the results once they are finished
    results = client.gather(futures)

    # Initializes counters for different types of return messages
    success_count = 0
    skipping_chunk_count = 0

    #TODO Can I resize the cluster down to just 1 worker at this point and still do the tile stats and logs?
    # I shouldn't need all the workers for at least the tile stats spreadsheet creation.

    # Processes the chunk stats and returned messages
    # Results are the messages from the chunks and chunk stats
    for result in results:
        return_message, chunk_stats = result

        if "Success" in return_message:
            success_count += 1

        if "Skipped chunk" in return_message:
            skipping_chunk_count += 1

        if return_message:
            return_messages.append(return_message)

        if chunk_stats is not None:
            all_stats.extend(chunk_stats)

    #TODO Test if including the success_message returns or printing them slows down large runs.
    # Don't return the success messages or print them if they slow down large runs.

    # Prints the returned messages
    for message in return_messages:
        print(message)

    # Print the counts
    print(f"Number of 'Success' chunks: {success_count}")
    print(f"Number of 'Skipped' chunks: {skipping_chunk_count}")
    print(f"Difference between submitted chunks and processed chunks: {len(chunks) - (success_count + skipping_chunk_count)}")

    # Iterates through output folders and counts the number of output rasters.
    # Only useful when doing a global run.
    for LULUCF_output_folder in cn.LULUCF_output_folders:

        LULUCF_output_folder = re.sub('RES_pixels', '4000_pixels', LULUCF_output_folder)
        LULUCF_output_folder = re.sub('DATE', uu.timestr()[:8], LULUCF_output_folder)

        geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(LULUCF_output_folder)
        print(f"Output rasters in {LULUCF_output_folder}: {file_count}")
        # print(geotiff_files)

    end_time_1 = uu.timestr()
    print(f"Stage {stage} ended at: {end_time_1}")
    uu.stage_duration(start_time, end_time_1, stage)

    # Prepares chunk stats spreadsheet: min, mean, max for all input and output chunks,
    # and min and max values across all chunks for all inputs and outputs
    # only if not suppressed by the --no_stats flag and at least one chunk was successfully (wasn't skipped).
    if (not no_stats) and (success_count > 0):
        uu.aggregate_chunk_stats(all_stats, stage, no_upload)

    # Ending time for stage
    end_time_2 = uu.timestr()
    print(f"Stage {stage} tile stats ended at: {end_time_2}")
    uu.stage_duration(start_time, end_time_2, stage)

    # Creates combined log from all workers if not deactivated
    log_note = f"{stage} run"
    lu.compile_and_upload_log(no_log, client, cluster, stage, len(chunks), chunk_size, start_time, end_time_1, end_time_2,
                              success_count, skipping_chunk_count, bounding_box, log_note)

    if not run_local:
        # Closes the Dask client if not running locally
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate LULUCF fluxes.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cl', '--chunk_list', help='Shapefile of chunks')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    bounding_box = args.bounding_box
    chunk_size = args.chunk_size
    chunk_list = args.chunk_list
    first_chunks = args.first_chunks
    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, run_local, no_stats, no_log, no_upload,
         bounding_box=bounding_box, chunk_size=chunk_size,
         chunk_list=chunk_list, first_chunks=first_chunks)