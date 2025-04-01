import math
import numpy as np
from numba import jit
from numba.typed import Dict
from numba.core import types

# Project imports
from . import constants_and_names as cn


# Adds latest decision tree branch to the state node
@jit(nopython=True)
def accrete_node(combo, new):
    combo = combo*10 + new
    return combo


# Calculates a backup continent-ecozone value in case pixels don't have one.
# There are many ways that are more efficient or at least succinct to calculate the mode of an array in Python,
# but they don't work with Numba. So, I'm going with this.
# https://chatgpt.com/share/e/67bf7958-351c-800a-bd00-259213586471
@jit(nopython=True)
def backup_continent_ecozone(continent_ecozone_block):

    # Flattens 2D array to 1D for counting
    continent_ecozone_block_flat = continent_ecozone_block.ravel()

    # Removes 0s so that the mode of the remaining pixels can be determined
    non_zero_values = continent_ecozone_block_flat[continent_ecozone_block_flat > 0]
    counts = np.bincount(non_zero_values)  # Counts the number of pixels with that value
    # print("Counts:", counts)
    if len(counts) == 0:   # If the only values in the chunk are 0 -> there are no counts of non-zero pixels
        continent_ecozone_fallback = 2020
    else:   # Otherwise, there are non-zero values in the chunk -> uses the most common non-zero value
        continent_ecozone_fallback = np.argmax(counts)

    # print("Fallback 1 continent_ecozone for chunk:", continent_ecozone_fallback)
    return continent_ecozone_fallback


# Creates a separate dictionary for each chunk datatype so that they can be passed to Numba as separate arguments.
# Numba functions can accept (and return) dictionaries of arrays as long as each dictionary only has arrays of one data type (e.g., uint8, float32)
# Note: need to add new code if inputs with other data types are added
def create_typed_dicts(layers):

    # Initializes empty dictionaries for each type
    uint8_dict_layers = {}
    uint16_dict_layers = {}
    int16_dict_layers = {}
    int32_dict_layers = {}
    float32_dict_layers = {}

    # Iterates through the downloaded chunk dictionary and distributes arrays to a separate dictionary for each data type
    for key, array in layers.items():

        # Skips the dictionary entry if it has no data (generally because the chunk doesn't exist for that input)
        if array is None:
            continue

        # print(key, print(array.dtype))

        # Suggested by https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/672bad5a-cda0-800a-8889-09657ed7e888
        # to optimize memory allocation for numba. Not sure it helps but it doesn't seemt to hurt, so leaving it in.
        contig_array = np.ascontiguousarray(array)

        # If there is data, it puts the data in the corresponding dictionary for that datatype
        if array.dtype == np.uint8:
            uint8_dict_layers[key] = contig_array
        elif array.dtype == np.uint16:
            uint16_dict_layers[key] = contig_array
        elif array.dtype == np.int16:
            int16_dict_layers[key] = contig_array
        elif array.dtype == np.int32:
            int32_dict_layers[key] = contig_array
        elif array.dtype == np.float32:
            float32_dict_layers[key] = contig_array
        else:
            raise TypeError(f"{key} dtype not in list")


    # print(f"uint8 datasets: {uint8_dict_layers.keys()}")
    # print(f"uint16 datasets: {uint16_dict_layers.keys()}")
    # print(f"int16 datasets: {int16_dict_layers.keys()}")
    # print(f"int32 datasets: {int32_dict_layers.keys()}")
    # print(f"float32 datasets: {float32_dict_layers.keys()}")

    # Creates numba-compliant typed dict for each type of array
    typed_dict_uint8 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.uint8, 2, 'C')  # Assuming 2D arrays of uint8
    )

    typed_dict_uint16 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.uint16, 2, 'C')  # Assuming 2D arrays of int16
    )

    typed_dict_int16 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.int16, 2, 'C')  # Assuming 2D arrays of int16
    )

    typed_dict_int32 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.int32, 2, 'C')  # Assuming 2D arrays of int32
    )

    typed_dict_float32 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.float32, 2, 'C')  # Assuming 2D arrays of float32
    )

    # Populates the numba-compliant typed dicts
    for key, array in uint8_dict_layers.items():
        typed_dict_uint8[key] = array

    for key, array in uint16_dict_layers.items():
        typed_dict_uint16[key] = array

    for key, array in int16_dict_layers.items():
        typed_dict_int16[key] = array

    for key, array in int32_dict_layers.items():
        typed_dict_int32[key] = array

    for key, array in float32_dict_layers.items():
        typed_dict_float32[key] = array

    return typed_dict_uint8, typed_dict_uint16, typed_dict_int16, typed_dict_int32, typed_dict_float32

# Classifies pixels into age bins for application of Robinson carbon growth curves
@jit(nopython=True)
def classify_forest_age(age):

    # Iterates through age bins for rates
    for age_bin in cn.natural_forest_growth_curve_intervals:
        start_str, end_str = age_bin.split('_')
        start, end = int(start_str), int(end_str)

        if start <= age <= end:
            return age_bin

    return 111

# Classifies vegetation height classes for start and end of current interval
@jit(nopython=True)
def classify_veg_height(LC_curr, LC_prev):
    tall_veg_prev = (((LC_prev >= cn.tree_dry_min_height_code) and (LC_prev <= cn.tree_dry_max_height_code)) or
                     ((LC_prev >= cn.tree_wet_min_height_code) and (LC_prev <= cn.tree_wet_max_height_code)))
    tall_veg_curr = (((LC_curr >= cn.tree_dry_min_height_code) and (LC_curr <= cn.tree_dry_max_height_code)) or
                     ((LC_curr >= cn.tree_wet_min_height_code) and (LC_curr <= cn.tree_wet_max_height_code)))
    med_veg_prev = (((LC_prev >= 25) and (LC_prev <= 26)) or ((LC_prev >= 125) and (LC_prev <= 126)))
    med_veg_curr = (((LC_curr >= 25) and (LC_curr <= 26)) or ((LC_curr >= 125) and (LC_curr <= 126)))
    short_veg_prev = (((LC_prev >= 2) and (LC_prev <= 24)) or ((LC_prev >= 102) and (LC_prev <= 124)))
    short_veg_curr = (((LC_curr >= 2) and (LC_curr <= 24)) or ((LC_curr >= 102) and (LC_curr <= 124)))

    return short_veg_prev, short_veg_curr, med_veg_prev, med_veg_curr, tall_veg_prev, tall_veg_curr


# Checks if pixel does not have tall vegetation. If so, updates the value to the most recent year without tall vegetation.
# Tall vegetation/non-tall vegetation is based on the composite land cover maps, not the canopy height maps.
# 0=Always tall vegetation so far. Other values represent the last year of non-tall vegetation.
# Theoretically, checking whether pixels are ever not forest could be done at the chunk level rather
# than at the pixel level (as done here). However, numba doesn't allow the conditional operation
# that would have to be applied to numpy arrays, so I'm doing it at the pixel level instead.
# I have no idea if checking if pixels have ever not been forest is faster or slower at the pixel level
# than at the numpy array level, but the array-level operation isn't even an option.
@jit(nopython=True)
def check_most_recent_year_not_tall_veg(LC_curr, LC_prev, most_recent_year_not_forest, interval_end_year):

    # For the first interval, the land cover in 2000 has to be checked for tall vegetation as well
    if interval_end_year == (cn.first_model_year_5_years + cn.interval_duration):

        # Criteria for excluding tall vegetation land cover
        not_tall_veg_condition = (
                (LC_prev < cn.tree_dry_min_height_code) |
                ((LC_prev > cn.tree_dry_max_height_code) & (LC_prev < cn.tree_wet_min_height_code))
                | (LC_prev > cn.tree_wet_max_height_code)
        )

        # Sets cell to the model start year wherever land cover is not tall vegetation
        if not_tall_veg_condition == 1:
            most_recent_year_not_forest = cn.first_model_year_5_years


    # Checks the current end of interval land cover
    # Criteria for excluding tall vegetation land cover
    not_tall_veg_condition = (
            (LC_curr < cn.tree_dry_min_height_code) |
            ((LC_curr > cn.tree_dry_max_height_code) & (LC_curr < cn.tree_wet_min_height_code))
            | (LC_curr > cn.tree_wet_max_height_code)
    )

    # Sets cell to interval end year wherever land cover is not tall vegetation
    if not_tall_veg_condition == 1:
        most_recent_year_not_forest = interval_end_year

    return most_recent_year_not_forest


# Calculates the number of years of forest regrowth since the last year of not-tall vegetation
@jit(nopython=True)
def calculate_years_of_forest_regrowth(interval_end_year, most_recent_year_not_forest, tall_veg_curr,
                                       partially_disturbed_in_last_interval, years_of_forest_regrowth):

    # Determines if the number of years of regrowth should be calculated, based on last stand-replacing disturbance
    # or partial disturbance.
    # Resets the number of years since disturbance to 0 if partial or complete disturbances occur.

    # Partial disturbance: if a partial disturbance occurred in the last interval, years of regrowth is set to
    # part of the interval.
    # Years of forest growth doesn't start until the end of the interval, regardless of the year in which the
    # disturbance occurs, if known.
    # For example, if a partial disturbance occurs in 2001 (as identified by the annual disturbance raster),
    # regrowth is assumed not to begin until the start of the next interval (2005).
    if partially_disturbed_in_last_interval:

        years_of_forest_regrowth = 0
        return years_of_forest_regrowth

    # Resets the growth year counter in cases where there was tall vegetation and then there wasn't in the next interval.
    # Otherwise, the years counter would continue accruing even if tall veg was lost.
    if not tall_veg_curr:
        years_of_forest_regrowth = 0
        return years_of_forest_regrowth

    # Increases the years of forest regrowth if certain conditions are met:
    # Condition 1: The end of the interval must be after the last year that was not tall vegetation,
    # i.e. there was not tall vegetation previously but there is at the end of this interval (indicating regrowth).
    # Condition 2: There must have been some year that was not forest,
    # i.e. the years of regrowth is only relevant when there was not forest some year.
    else:
        if (interval_end_year > most_recent_year_not_forest) & (most_recent_year_not_forest > 0):

            years_of_forest_regrowth = years_of_forest_regrowth + cn.interval_duration

        else:  # No change
            years_of_forest_regrowth = years_of_forest_regrowth

    return years_of_forest_regrowth


# Calculates the maximum canopy height since the last time a pixel was classified as not tall vegetation land cover.
# This is used to determine whether current height has decreased significantly from this maximum height.
@jit(nopython=True)
def calc_max_height_since_last_time_not_tall_veg(most_recent_year_not_tall_veg, vegetation_height_so_far_cell, years_so_far_cell):

    # Determines the maximum height so far if the pixel has been tall vegetation land cover since the beginning of the model
    if most_recent_year_not_tall_veg == 0:

        # The maximum vegetation height through all intervals so far
        max_height_since_last_time_not_tall_veg = max(vegetation_height_so_far_cell)

    # Determines the maximum height so far if the pixel hasn't had tall vegetation land cover at least one year since the beginning of the model
    else:

        heights_since_last_time_not_tall_veg = []

        # Loops over the years and corresponding heights to only get heights that are after the most recent
        # non-tall vegetation year.
        # This could be done more elegantly with conditional numpy arrays but that approach
        # isn't supported in the numba function, unfortunately.
        # https://chatgpt.com/share/e/6718fb20-48d8-800a-9eb2-d751bd6b1a8f
        for i in range(len(years_so_far_cell)):
            if years_so_far_cell[i] > most_recent_year_not_tall_veg:
                heights_since_last_time_not_tall_veg.append(vegetation_height_so_far_cell[i])

        # In case the pixel is currently non-tall vegetation land cover, so there are no intervals since then
        # and therefore no heights
        if len(heights_since_last_time_not_tall_veg) == 0:

            # Uses the current vegetation height (which would exist when the land cover is not tall vegetation
            # but there is still tall vegetation in the individual tree height layer)
            max_height_since_last_time_not_tall_veg = vegetation_height_so_far_cell[-1]
            # max_height_since_last_time_not_tall_veg = 0

        # When the pixel was previously non-tall vegetation but is now tall vegetation,
        # so there are intervals since then.
        else:

            # The maximum height in the years since the last non-tall vegetation land cover interval
            max_height_since_last_time_not_tall_veg = max(heights_since_last_time_not_tall_veg)

    return max_height_since_last_time_not_tall_veg


# Calculates the ratio of deadwood C:AGC and litter C:AGC based on climate domain, elevation, and precip
# for natural, terrestrial forests.
# Deadwood and litter carbon as fractions of AGC are from
# https://cdm.unfccc.int/methodologies/ARmethodologies/tools/ar-am-tool-12-v3.0.pdf
# "Clean Development Mechanism A/R Methodological Tool:
# Estimation of carbon stocks and change in carbon stocks in dead wood and litter in A/R CDM project activities version 03.0"
# Tables on pages 18 (deadwood) and 19 (litter).
# They depend on the climate domain, elevation, and precipitation.
@jit(nopython=True)
def calc_deadwood_litter_ratios(elevation, climate_domain, precipitation):

    if climate_domain == 1:  # Tropical/subtropical
        if elevation <= 2000:  # Low elevation
            if precipitation <= 1000:  # Low precipitation or no precip raster
                deadwood_c_ratio = cn.tropical_low_elev_low_precip_deadwood_c_ratio
                litter_c_ratio = cn.tropical_low_elev_low_precip_litter_c_ratio
            elif ((precipitation > 1000) and (precipitation <= 1600)):  # Medium precipitation
                deadwood_c_ratio = cn.tropical_low_elev_med_precip_deadwood_c_ratio
                litter_c_ratio = cn.tropical_low_elev_med_precip_litter_c_ratio
            else:  # High precipitation
                deadwood_c_ratio = cn.tropical_low_elev_high_precip_deadwood_c_ratio
                litter_c_ratio = cn.tropical_low_elev_high_precip_litter_c_ratio
        else:  # High elevation
            deadwood_c_ratio = cn.tropical_high_elev_deadwood_c_ratio
            litter_c_ratio = cn.tropical_high_elev_litter_c_ratio
    else:  # Temperate/boreal
        deadwood_c_ratio = cn.non_tropical_deadwood_c_ratio
        litter_c_ratio = cn.non_tropical_litter_c_ratio

    return float(deadwood_c_ratio), float(litter_c_ratio)


# Returns AGC and BGC one-time removal factors for the gain of medium-height vegetation (Mg C/ha)
#TODO Correct and complete this function. Currently using just climate domain as a stand in for IPCC climate zone.
@jit(nopython=True)
def calc_medium_height_veg_removals(climate_domain):

    if climate_domain == 1:  # Tropical/subtropical
        medium_height_veg_AGB_RF = 4.3   # Average of the two tropical classes for now
        medium_height_veg_BGB_RF = (12.4-medium_height_veg_AGB_RF)  # Average of the two tropical classes for now
    elif climate_domain == 2: # Temperate
        medium_height_veg_AGB_RF = 2.1
        medium_height_veg_BGB_RF = 1.0
    elif climate_domain == 3: # Boreal
        medium_height_veg_AGB_RF = 1.7  # Average of the four temperate classes for now
        medium_height_veg_BGB_RF = (9.9-medium_height_veg_AGB_RF)  # Average of the four temperate classes for now
    else: # Outside ecozone bounds
        medium_height_veg_AGB_RF = 1.7
        medium_height_veg_BGB_RF = (8.5-medium_height_veg_AGB_RF)

    medium_height_veg_AGC_RF = medium_height_veg_AGB_RF * cn.biomass_to_carbon_non_mangrove
    medium_height_veg_BGC_RF = medium_height_veg_BGB_RF * cn.biomass_to_carbon_non_mangrove

    return medium_height_veg_AGC_RF, medium_height_veg_BGC_RF


# Returns the starting carbon density for each carbon pool
@jit(nopython=True)
def unpack_starting_carbon_densities(c_dens_in):

    agc_dens_in = np.float32(c_dens_in[0])
    bgc_dens_in = np.float32(c_dens_in[1])
    deadwood_c_dens_in = np.float32(c_dens_in[2])
    litter_c_dens_in = np.float32(c_dens_in[3])

    return agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in


# Returns the emission factor for each carbon pool
@jit(nopython=True)
def unpack_emission_factors(ef):

    agc_ef = np.float32(ef[0])
    bgc_ef = np.float32(ef[1])
    deadwood_c_ef = np.float32(ef[2])
    litter_c_ef = np.float32(ef[3])

    return agc_ef, bgc_ef, deadwood_c_ef, litter_c_ef


# Returns the removal factor for primary forest/IFL based on the continent-ecozone combination (Mg AGC/ha/yr)
# From https://chatgpt.com/share/e/67340a6f-b8cc-800a-84e3-f98e600001e5
@jit(nopython=True)
def calc_primary_forest_RF(continent_ecozone_cell, primary_forest_RFs):

    primary_forest_RF_indices = np.where(primary_forest_RFs[:, 0] == continent_ecozone_cell)

    # Checks if there are matching indices and extracts corresponding primary forest RF
    if primary_forest_RF_indices[0].size > 0:  # If matching continent-ecozone combination...
        primary_forest_RF = primary_forest_RFs[primary_forest_RF_indices[0][0], 1]  # Uses matching RF
    else:  # If no matching continent-ecozone combination...
        primary_forest_RF = np.mean(primary_forest_RFs[:, 1]) # Uses average of all primary forest RFs
    return primary_forest_RF


# Calculates Cf for fire emissions from forests (as opposed to savanna/grassland or biofuel burning).
# From IPCC 2019 Table 2.6 (unitless)
@jit(nopython=True)
def calc_Cf_forest(climate_domain_cell, drivers_cell, ifl_primary_cell):

    # Groups of drivers with different Cfs
    driver_group_1 = [cn.permanent_agriculture, cn.shifting_cultivation, cn.hard_commodities, cn.wildfire, cn.settlements_and_infrastruct]
    driver_group_2 = [cn.forest_management]
    driver_group_3 = [cn.other_natural_disturbances]

    if climate_domain_cell == 1:  # Tropical/subtropical
        if ifl_primary_cell:  # Tropical/subtropical, primary forest
            Cf = 0.36
        else:  # Tropical/subtropical, not primary forest
            Cf = 0.55
    elif climate_domain_cell == 2:   # Temperate
        if drivers_cell in driver_group_1:  # Temperate, driver group 1
            Cf = 0.51
        elif drivers_cell in driver_group_2:  # Temperate, driver group 2
            Cf = 0.62
        elif drivers_cell in driver_group_3:  # Temperate, driver group 3
            Cf = 0.45
        else:  # Temperate, no driver assigned
            Cf = 0.45
    elif climate_domain_cell == 3:
        if drivers_cell in driver_group_1:  # Boreal, driver group 1
            Cf = 0.59
        elif drivers_cell in driver_group_2:  # Boreal, driver group 2
            Cf = 0.33
        elif drivers_cell in driver_group_3:  # Boreal, driver group 3
            Cf = 0.34
        else:  # Boreal, no driver assigned
            Cf = 0.34
    else:  # Outside ecozone bounds
        if drivers_cell in driver_group_1:  # Outside ecozone bounds, driver group 1
            Cf = 0.59
        elif drivers_cell in driver_group_2:  # Outside ecozone bounds, driver group 2
            Cf = 0.33
        elif drivers_cell in driver_group_3:  # Outside ecozone bounds, driver group 2
            Cf = 0.34
        else:  # Outside ecozone bounds, no driver assigned
            Cf = 0.34

    return Cf


# Calculates Gef for fire emissions from forests (as opposed to savanna/grassland or biofuel burning).
# From IPCC 2019 Table 2.5 (g respective gas/kg dry matter)
@jit(nopython=True)
def calc_Gef_forest(climate_domain_cell):

    if climate_domain_cell == 1:  # Tropical/subtropical
        Gef_CO2_forest = 1580.0
        Gef_CH4_forest = 6.8
        Gef_N2O_forest = 0.2
    elif climate_domain_cell == 2 or climate_domain_cell == 3:   # Temperate/boreal
        Gef_CO2_forest = 1569.0
        Gef_CH4_forest = 4.7
        Gef_N2O_forest = 0.26
    else:  # Outside ecozone bounds
        Gef_CO2_forest = 1569.0
        Gef_CH4_forest = 4.7
        Gef_N2O_forest = 0.26

    return Gef_CO2_forest, Gef_CH4_forest, Gef_N2O_forest


# Calculates non-CO2 emissions (CH4 and N2O) separately.
# Cf is the combustion factor
# Gef_ch4 and Gef_n2o are the emission factors for their respective gases.
# biomass_to_carbon can be hard-coded as non-mangrove because we assume that mangroves don't have fires.
# From IPCC 2019 Eqn. 2.27
@jit(nopython=True)
def non_CO2_fire_equations(carbon_in, Cf, Gef_ch4, Gef_n2o):

    # print(f"Carbon in: {carbon_in}; Cf: {Cf}; Gef_ch4: {Gef_ch4}; GWP CH4: {cn.gwp_ch4}")

    ch4_flux_out = (carbon_in/cn.biomass_to_carbon_non_mangrove) * Cf * Gef_ch4 * cn.g_to_kg * cn.gwp_ch4
    n2o_flux_out = (carbon_in/cn.biomass_to_carbon_non_mangrove) * Cf * Gef_n2o * cn.g_to_kg * cn.gwp_n2o

    # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
    # os.quit()

    return ch4_flux_out, n2o_flux_out


# Gross and net fluxes and ending carbon stocks for non-tree converted to tree.
# Carbon pool fluxes and densities are input and output as Mg C/ha(/interval) rather than Mg CO2 for arithmetic simplicity.
@jit(nopython=True)
def calc_NT_T_5_yrs(agc_rf, bgc_rf, c_dens_in, deadwood_c_ratio, litter_c_ratio):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    # Step 1: Calculates the number of years of carbon gain (years)
    gain_year_count = cn.NT_T_gain_year_count_default

    # Step 2: Calculates gross removals by carbon pools (Mg C/ha/interval). Gross removals are negative.
    agc_gross_removals_out = float((agc_rf * gain_year_count) * -1)  #float() necessary for Numba typing
    bgc_gross_removals_out = float((bgc_rf * gain_year_count) * -1)  #float() necessary for Numba typing
    deadwood_c_gross_removals_out = agc_gross_removals_out * deadwood_c_ratio
    litter_c_gross_removals_out = agc_gross_removals_out * litter_c_ratio

    # Step 3: Calculates gross emissions by carbon pools (Mg C/ha/interval). Gross emissions are positive.
    # There are no gross emissions in NT->T pixels, so C pool emissions set to 0.
    agc_gross_emis_out = 0
    bgc_gross_emis_out = 0
    deadwood_c_gross_emis_out = 0
    litter_c_gross_emis_out = 0

    # Step 4: Calculates ending carbon densities by carbon pool (Mg C/ha)
    agc_dens_out = agc_dens_in - agc_gross_removals_out
    bgc_dens_out = bgc_dens_in - bgc_gross_removals_out
    deadwood_c_dens_out = deadwood_c_dens_in - deadwood_c_gross_removals_out
    litter_c_dens_out = litter_c_dens_in - litter_c_gross_removals_out

    # Step 5: Prepares outputs
    # Consolidates all gross fluxes from all carbon pools into arrays to reduce the number of arguments returned to the decision tree
    # Must specify float32 because numba is quite particular about datatypes
    c_gross_removals_out = np.array([agc_gross_removals_out, bgc_gross_removals_out, deadwood_c_gross_removals_out, litter_c_gross_removals_out]).astype('float32')  # Mg C/ha/interval
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')  # Mg C/ha/interval
    c_dens_out = np.array([agc_dens_out, bgc_dens_out, deadwood_c_dens_out, litter_c_dens_out]).astype('float32')  # Mg C/ha

    return c_gross_emissions_out, c_gross_removals_out, c_dens_out, gain_year_count


# Gross fluxes and ending carbon stocks for trees converted to non-trees with and without fire.
# Non-CO2 gas emissions are only calculated if fire was detected during the interval.
# CO2 emissions are calculated differently depending on if fire was detected during the interval and if a Gef_CO2 is supplied.
# Carbon pool fluxes and densities are input and output as Mg C/ha(/interval) rather than Mg CO2 for arithmetic simplicity.
@jit(nopython=True)
def calc_T_NT_5_yrs(node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2, c_pools_no_fire,
                    forest_dist_last, interval_end_year, c_dens_in,
                    post_dist_regrowth, most_recent_year_not_tall_veg, Cf, Gef_ch4, Gef_n2o,
                    deadwood_c_ratio, litter_c_ratio):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    # Establishes which carbon pools are emitted depending on whether fire was detected during the interval.
    # For T->NT, emission factors are binary (1 means full emissions, 0 means no emissions).
    # Carbon pools that are emitted as CO2 if fire was detected.
    if burned_in_last_interval:
        agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_fire_CO2)
    else:
        # Carbon pools that are emitted as CO2 if fire was not detected.
        agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_no_fire)


    ## Step 1: Calculates the number of years of carbon gain before loss occurred (years)
    if forest_dist_last > 0:
        # If a forest disturbance was detected, the gain_year_count are the number of years until detection of the last disturbance.
        # There is no growth in the year of disturbance or the years after.
        # The - 1 at the excludes the disturbance year from the gain_year_count since we decided there are no removals in the disturbance year.
        # For example, if the time interval is 2010-2015 and the disturbance is detected in 2013 (t-2),
        # there should be 2 years of growth (years t-4 and t-3, 2011 and 2012).
        # This table illustrates each case for the example interval of 2010-2015.
        # 0 years         11               - ((2015              - 2000)                - 5) - 1   (year t-4)
        # 1 years         12               - ((2015              - 2000)                - 5) - 1   (year t-3)
        # 2 years         13               - ((2015              - 2000)                - 5) - 1   (year t-2)
        # 3 years         14               - ((2015              - 2000)                - 5) - 1   (year t-1)
        # 4 years         15               - ((2015              - 2000)                - 5) - 1   (year t)
        gain_year_count = forest_dist_last - ((interval_end_year - cn.first_model_year_5_years) - cn.interval_duration) - 1
    else:
        # If a forest disturbance was not detected, the disturbance is assumed to occur in the middle of the interval
        # (year t-2), with removals until then (years t-4 and t-3). There are no removals in the year of assumed
        # disturbance or the years after.
        gain_year_count = math.floor(cn.interval_duration / 2)


    # Step 2: Assigns deadwood C and litter C ratios for removal factors, if relevant (unitless)
    # Deadwood and litter C removals only occur in pixels that were not tall vegetation at some point (natural forest only).
    # Thus, we need to check whether the pixel was non-tall vegetation at some point during the model before the end of this interval.
    # If conditions aren't met, the deadwood and litter ratios are set to 0 (no removals).
    # For simplicity, there are no deadwood or litter removals in loss intervals.
    if most_recent_year_not_tall_veg == 0 or most_recent_year_not_tall_veg == interval_end_year:
        deadwood_c_ratio = 0.0
        litter_c_ratio = 0.0


    # Step 3: Calculates gross removals by carbon pools (Mg C/ha/interval). Gross removals are negative.
    agc_gross_removals_out = float((agc_rf * gain_year_count) * -1)   # float() necessary for Numba typing
    bgc_gross_removals_out = float((bgc_rf * gain_year_count) * -1)   # float() necessary for Numba typing
    deadwood_c_gross_removals_out= agc_gross_removals_out * deadwood_c_ratio
    litter_c_gross_removals_out= agc_gross_removals_out * litter_c_ratio

    # Consolidates outputs into array to reduce the number of arguments returned to the decision tree.
    # Must specify float32 because numba is quite particular about datatypes.
    c_gross_removals_out = np.array([agc_gross_removals_out, bgc_gross_removals_out, deadwood_c_gross_removals_out, litter_c_gross_removals_out]).astype('float32')


    # Step 4: Calculates carbon densities at the year of loss by carbon pool (Mg C/ha).
    # This is not output from the model.
    agc_pre_disturb = agc_dens_in - agc_gross_removals_out
    bgc_pre_disturb = bgc_dens_in - bgc_gross_removals_out
    deadwood_c_pre_disturb = deadwood_c_dens_in - deadwood_c_gross_removals_out
    litter_c_pre_disturb = litter_c_dens_in - litter_c_gross_removals_out

    # Pre-disturbance carbon densities as an array, used as input for non-CO2 fire emissions and post-disturbance removals (if applicable)
    c_pre_disturb = np.array([agc_pre_disturb, bgc_pre_disturb, deadwood_c_pre_disturb, litter_c_pre_disturb])


    # Step 5: Calculates CO2 gross emissions by carbon pools (Mg C/ha/interval). Gross emissions are positive.
    # Which ones are emitted depends on whether fire was detected.
    agc_gross_emis_out = agc_pre_disturb * agc_ef_CO2
    bgc_gross_emis_out = bgc_pre_disturb * bgc_ef_CO2
    deadwood_c_gross_emis_out = deadwood_c_pre_disturb * deadwood_c_ef_CO2
    litter_c_gross_emis_out = litter_c_pre_disturb * litter_c_ef_CO2

    # Consolidates outputs into array to reduce the number of arguments returned to the decision tree.
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')


    # Step 6: Updates gross removals to include one-time post-disturbance regrowth,
    # if applicable (medium height veg and cropland) (Mg C/ha/interval).
    # Regrowth of medium height veg and cropland is a one-time value, not annual, so no multiplication by gain year count.
    c_gross_removals_out = c_gross_removals_out - post_dist_regrowth


    # Step 7: Calculates ending carbon densities by carbon pool (Mg C/ha).
    # Starts with carbon density in (list converted to np array), adds gross removals (subtracts negative value), subtracts emissions (positive value).
    # Ending carbon pools are not affected by non-CO2 emissions in the next step.
    c_dens_out = np.array(c_dens_in).astype('float32') - c_gross_removals_out - c_gross_emissions_out


    # Step 8: Calculates non-CO2 emissions (if relevant) (Mg CO2e/ha/interval)
    # Default non-CO2 emissions values
    ch4_flux_out = 0
    n2o_flux_out = 0

    # Only assigns fire node code and calculates CH4 and N2O emissions if the pixel burned in the last interval
    if burned_in_last_interval:

        state_out = accrete_node(node, 1)

        # Selects just the carbon pools that have non-CO2 emissions from fire
        c_pools_for_fire_non_CO2 = np.where(c_pools_fire_non_CO2 == 1, c_pre_disturb, 0)

        # Sums the C pools that have non-CO2 fire emissions. We don't track which C pools the CH4 and N2O emissions come from,
        # so the pools are combined.
        c_pools_for_fire_total = np.sum(c_pools_for_fire_non_CO2)

        # Calculates non-CO2 fire emissions using the selected C pools in the year before disturbance
        ch4_flux_out, n2o_flux_out = non_CO2_fire_equations(c_pools_for_fire_total, Cf, Gef_ch4, Gef_n2o)

        # # For testing non-CO2 emissions
        # print("c_dens_in:", c_dens_in)
        # print("c_pre_disturb:", c_pre_disturb)
        # print(f"Cf: {Cf}; Gef_ch4: {Gef_ch4}; GWP CH4: {cn.gwp_ch4}")
        # print(f"Cf: {Cf}; Gef_n2o: {Gef_n2o}; GWP N2O: {cn.gwp_n2o}")
        # print("c_pools_for_fire_non_CO2:", c_pools_for_fire_non_CO2)
        # print("c_pools_for_fire_total:", c_pools_for_fire_total)
        # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
        # os.quit()

    # Node code if no fire in the last interval. No CH4 and N2O emissions calculated.
    else:

        state_out = accrete_node(node, 2)

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')

    # # For testing
    # if burned_in_last_interval:
    #
    #     print("agc_dens_in:", agc_dens_in)
    #     print("agc_gross_removals_out:", agc_gross_removals_out)
    #     print("agc_pre_disturb:", agc_pre_disturb)
    #     print("biomass_to_carbon_non_mangrove:", cn.biomass_to_carbon_non_mangrove)
    #     print("Cf:", Cf)
    #     print("Gef_ch4:", Gef_ch4)
    #     print("Gef_n2o:", Gef_n2o)
    #     print("agc_gross_emis_out:", agc_gross_emis_out)
    #     print("c_dens_out:", c_dens_out)
    #     os.quit()

    return state_out, c_gross_emissions_out, c_gross_removals_out, non_co2_fluxes_out, c_dens_out, gain_year_count


# Gross and net fluxes and ending carbon stocks for trees remaining trees with non-stand-replacing disturbances.
# Carbon pool fluxes and densities are input and output as Mg C/ha(/interval) rather than Mg CO2 for arithmetic simplicity.
@jit(nopython=True)
def calc_T_T_non_stand_disturbs_5_yrs(node, burned_in_last_interval, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2, c_pools_no_fire,
                                      forest_dist_last, interval_end_year, c_dens_in,
                                      post_dist_RF, most_recent_year_not_tall_veg, Cf, Gef_co2, Gef_ch4, Gef_n2o,
                                      deadwood_c_ratio, litter_c_ratio):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    # Establishes which carbon pools are emitted depending on whether fire was detected during the interval.
    # For T->T, emission factors can range between 0 and 1, with 0 meaning no emissions and 1 meaning full emissions.
    # Carbon pools that are emitted as CO2 if fire was detected.
    if burned_in_last_interval:
        agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_fire_CO2)
    else:
        # Carbon pools that are emitted as CO2 if fire was not detected.
        agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_no_fire)


    ## Step 1: Calculates the number of years of carbon gain before the non-stand-replacing disturbance occurred (years)
    if forest_dist_last > 0:
        # If a forest disturbance was detected, the gain_year_count are the number of years until detection of the last disturbance.
        # There is no growth in the year of disturbance or the years after.
        # The - 1 at the excludes the disturbance year from the gain_year_count since we decided there are no removals in the disturbance year.
        # For example, if the time interval is 2010-2015 and the disturbance is detected in 2013 (t-2),
        # there should be 2 years of growth (years t-4 and t-3, 2011 and 2012).
        # This table illustrates each case for the example interval of 2010-2015.
        # 0 years         11               - ((2015              - 2000)                - 5) - 1   (year t-4)
        # 1 years         12               - ((2015              - 2000)                - 5) - 1   (year t-3)
        # 2 years         13               - ((2015              - 2000)                - 5) - 1   (year t-2)
        # 3 years         14               - ((2015              - 2000)                - 5) - 1   (year t-1)
        # 4 years         15               - ((2015              - 2000)                - 5) - 1   (year t)
        gain_year_count = forest_dist_last - ((interval_end_year - cn.first_model_year_5_years) - cn.interval_duration) - 1
    else:
        # If a forest disturbance was not detected, the disturbance is assumed to occur in the middle of the interval
        # (year t-2), with removals until then (years t-4 and t-3). There are no removals in the year of assumed
        # disturbance or the years after.
        gain_year_count = math.floor(cn.interval_duration / 2)


    # Step 2: Assigns deadwood C and litter C ratios for removal factors, if relevant (unitless).
    # Deadwood and litter C removals only occur in pixels that were not tall vegetation at some point (natural forest only).
    # Thus, we need to check whether the pixel was non-tall vegetation at some point during the model before the end of this interval.
    # If conditions aren't met, the deadwood and litter ratios are set to 0 (no removals).
    if most_recent_year_not_tall_veg == 0 or most_recent_year_not_tall_veg == interval_end_year:
        deadwood_c_ratio = 0.0
        litter_c_ratio = 0.0


    # Step 3: Calculates gross removals by carbon pools before disturbance (Mg CO2/ha/interval). Gross removals are negative.
    agc_gross_removals_out = float((agc_rf * gain_year_count) * -1) #float() necessary for Numba typing
    bgc_gross_removals_out = float((bgc_rf * gain_year_count) * -1) #float() necessary for Numba typing
    deadwood_c_gross_removals_out= agc_gross_removals_out * deadwood_c_ratio
    litter_c_gross_removals_out= agc_gross_removals_out * litter_c_ratio

    # Consolidates outputs into array to reduce the number of arguments returned to the decision tree.
    # Must specify float32 because numba is quite particular about datatypes.
    c_gross_removals_out = np.array([agc_gross_removals_out, bgc_gross_removals_out, deadwood_c_gross_removals_out, litter_c_gross_removals_out]).astype('float32')


    # Step 4: Calculates carbon densities at the year of disturbance by carbon pool (Mg C/ha).
    # This is not output from the model.
    agc_pre_disturb = agc_dens_in - agc_gross_removals_out
    bgc_pre_disturb = bgc_dens_in - bgc_gross_removals_out
    deadwood_c_pre_disturb = deadwood_c_dens_in - deadwood_c_gross_removals_out
    litter_c_pre_disturb = litter_c_dens_in - litter_c_gross_removals_out

    # Pre-disturbance carbon densities as an array, used as input for non-CO2 fire emissions and post-disturbance removals (if applicable)
    c_pre_disturb = np.array([agc_pre_disturb, bgc_pre_disturb, deadwood_c_pre_disturb, litter_c_pre_disturb])


    # Step 5: Calculates CO2 gross emissions by carbon pools (Mg C/ha/interval). Gross emissions are positive.
    # Which ones are emitted depends on whether fire was detected.

    # Calculates CO2 emissions from fire for each C pool using fire emission factors
    # if a Gef for CO2 is supplied AND if there was fire during the interval.
    # This is used for non-stand replacing forest disturbances (as opposed to entire C pools being combusted).
    # From IPCC 2019 Eqn. 2.27
    if burned_in_last_interval:

        agc_gross_emis_out = ((agc_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * agc_ef_CO2
        bgc_gross_emis_out = ((bgc_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * bgc_ef_CO2
        deadwood_c_gross_emis_out = ((deadwood_c_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * deadwood_c_ef_CO2
        litter_c_gross_emis_out = ((litter_c_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * litter_c_ef_CO2

        # # For testing CO2 fire emissions
        # print("c_dens_in:", c_dens_in)
        # print("agc_rf:", agc_rf)
        # print("gain_year_count:", gain_year_count)
        # print("agc_pre_disturb:", agc_pre_disturb)
        # print(f"Cf: {Cf}; Gef_co2: {Gef_co2}")
        # print("AGC emission factor for fire:", agc_ef_CO2)
        # print("agc_gross_emis_out:", agc_gross_emis_out)
        # os.quit()

    # Calculates CO2 emissions from forest loss for each C pool when no fire is detected
    else:

        agc_gross_emis_out = agc_pre_disturb * agc_ef_CO2
        bgc_gross_emis_out = bgc_pre_disturb * bgc_ef_CO2
        deadwood_c_gross_emis_out = deadwood_c_pre_disturb * deadwood_c_ef_CO2
        litter_c_gross_emis_out = litter_c_pre_disturb * litter_c_ef_CO2

    # Gross emissions as an array
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')


    # Step 6: Updates gross removals to include post-disturbance gross removals,
    # if applicable (>=5 m height gain) (Mg C/ha/interval).
    # post_dist_gain_year_count here is the number of years between the disturbance and the end of the interval.
    post_dist_gain_year_count = cn.interval_duration - gain_year_count - 1
    post_dist_gross_removals = post_dist_gain_year_count * post_dist_RF

    c_gross_removals_out = c_gross_removals_out - post_dist_gross_removals


    # Step 7: Calculates ending carbon densities by carbon pool (Mg C/ha).
    # Starts with carbon density in (list converted to np array), adds gross removals (subtracts negative value), subtracts emissions (positive value).
    # Ending carbon pools are not affected by non-CO2 emissions in the next step.
    c_dens_out = np.array(c_dens_in).astype('float32') - c_gross_removals_out - c_gross_emissions_out


    # Step 8: Calculates non-CO2 emissions (if relevant) (Mg CO2e/ha/interval)
    # Default non-CO2 emissions values
    ch4_flux_out = 0
    n2o_flux_out = 0

    # Only assigns fire node code and calculates CH4 and N2O emissions if the pixel burned in the last interval
    if burned_in_last_interval:

        state_out = accrete_node(node, 1)

        # Selects just the carbon pools that have non-CO2 emissions from fire
        c_pools_for_fire_non_CO2 = np.where(c_pools_fire_non_CO2 == 1, c_pre_disturb, 0)

        # Sums the C pools that have non-CO2 fire emissions. We don't track which C pools the CH4 and N2O emissions come from,
        # so the pools are combined.
        c_pools_for_fire_total = np.sum(c_pools_for_fire_non_CO2)

        # Calculates non-CO2 fire emissions using the selected C pools in the year before disturbance
        ch4_flux_out, n2o_flux_out = non_CO2_fire_equations(c_pools_for_fire_total, Cf, Gef_ch4, Gef_n2o)

        # # For testing non-CO2 emissions
        # print("c_dens_in:", c_dens_in)
        # print("c_pre_disturb:", c_pre_disturb)
        # print(f"Cf: {Cf}; Gef_ch4: {Gef_ch4}; GWP CH4: {cn.gwp_ch4}")
        # print(f"Cf: {Cf}; Gef_n2o: {Gef_n2o}; GWP N2O: {cn.gwp_n2o}")
        # print("c_pools_for_fire_non_CO2:", c_pools_for_fire_non_CO2)
        # print("c_pools_for_fire_total:", c_pools_for_fire_total)
        # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
        # os.quit()

    # Node code if no fire in the last interval. No CH4 and N2O emissions calculated.
    else:

        state_out = accrete_node(node, 2)

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')

    return state_out, c_gross_emissions_out, c_gross_removals_out, non_co2_fluxes_out, c_dens_out, gain_year_count


# Gross and net fluxes and ending carbon stocks for trees remaining trees with non-stand-replacing disturbances.
# Carbon pool fluxes and densities are input and output as Mg C/ha(/interval) rather than Mg CO2 for arithmetic simplicity.
@jit(nopython=True)
def calc_T_T_no_disturbs_5_yrs(node, most_recent_year_burned, agc_rf, bgc_rf, c_pools_fire_CO2, c_pools_fire_non_CO2,
                               interval_end_year, c_dens_in,
                               most_recent_year_not_tall_veg, Cf, Gef_co2, Gef_ch4, Gef_n2o,
                               deadwood_c_ratio, litter_c_ratio):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    # Establishes which carbon pools are emitted if a fire was detected during the interval.
    agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_fire_CO2)


    # Step 1: Calculates the number of years of carbon gain before a fire occurred (years).
    # Note that removals continue after the year of fire, too. This gain_year_count is used to determine removals
    # until fire (i.e. carbon densities at the year of fire).
    if most_recent_year_burned > 0:
        # If a forest disturbance was detected, the gain_year_count are the number of years until detection of the last disturbance.
        # There is no growth in the year of disturbance or the years after.
        # The - 1 at the excludes the disturbance year from the gain_year_count since we decided there are no removals in the disturbance year.
        # For example, if the time interval is 2010-2015 and the disturbance is detected in 2013 (t-2),
        # there should be 2 years of growth (years t-4 and t-3, 2011 and 2012).
        # This table illustrates each case for the example interval of 2010-2015.
        # 0 years         11               - ((2015              - 2000)                - 5) - 1   (year t-4)
        # 1 years         12               - ((2015              - 2000)                - 5) - 1   (year t-3)
        # 2 years         13               - ((2015              - 2000)                - 5) - 1   (year t-2)
        # 3 years         14               - ((2015              - 2000)                - 5) - 1   (year t-1)
        # 4 years         15               - ((2015              - 2000)                - 5) - 1   (year t)
        gain_year_count = most_recent_year_burned - ((interval_end_year - cn.first_model_year_5_years) - cn.interval_duration) - 1
    else:
        # If no fire was detected, removals occurred every year
        gain_year_count = cn.interval_duration


    # Step 2: Assigns deadwood C and litter C ratios for removal factors, if relevant (unitless).
    # Deadwood and litter C removals only occur in pixels that were not tall vegetation at some point (natural forest only).
    # Thus, we need to check whether the pixel was non-tall vegetation at some point during the model before the end of this interval.
    # If conditions aren't met, the deadwood and litter ratios are set to 0 (no removals).
    #TODO Refactor this rule into its own function and use that in T->T disturbed, and T->T undisturbed.
    # T->NT has a slightly different formulation of the rule for when to set these to 0.
    if most_recent_year_not_tall_veg == 0 or most_recent_year_not_tall_veg == interval_end_year:
        deadwood_c_ratio = 0.0
        litter_c_ratio = 0.0


    # Step 3: Calculates gross removals by carbon pools before disturbance (Mg C/ha/interval). Gross removals are negative.
    agc_gross_removals_out = float((agc_rf * gain_year_count) * -1) #float() necessary for Numba typing
    bgc_gross_removals_out = float((bgc_rf * gain_year_count) * -1) #float() necessary for Numba typing
    deadwood_c_gross_removals_out= agc_gross_removals_out * deadwood_c_ratio
    litter_c_gross_removals_out= agc_gross_removals_out * litter_c_ratio

    # Consolidates outputs into array to reduce the number of arguments returned to the decision tree.
    # Must specify float32 because numba is quite particular about datatypes.
    c_gross_removals_out = np.array([agc_gross_removals_out, bgc_gross_removals_out, deadwood_c_gross_removals_out, litter_c_gross_removals_out]).astype('float32')


    # Step 4: Calculates carbon densities at the year of loss by carbon pool (Mg C/ha).
    # This is not output from the model.
    agc_pre_disturb = agc_dens_in - agc_gross_removals_out
    bgc_pre_disturb = bgc_dens_in - bgc_gross_removals_out
    deadwood_c_pre_disturb = deadwood_c_dens_in - deadwood_c_gross_removals_out
    litter_c_pre_disturb = litter_c_dens_in - litter_c_gross_removals_out

    # Pre-disturbance carbon densities as an array, used as input for non-CO2 fire emissions and post-disturbance removals (if applicable)
    c_pre_disturb = np.array([agc_pre_disturb, bgc_pre_disturb, deadwood_c_pre_disturb, litter_c_pre_disturb])


    # Step 5: Calculates CO2 gross emissions by carbon pools (Mg C/ha/interval).  Which ones are emitted depends on whether fire was detected.

    # Calculates CO2 emissions from fire for each C pool using fire emission factors
    # if a Gef for CO2 is supplied AND if there was fire during the interval.
    # This is used for non-stand replacing forest disturbances (as opposed to entire C pools being combusted).
    # From IPCC 2019 Eqn. 2.27
    if most_recent_year_burned > 0:

        agc_gross_emis_out = ((agc_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * agc_ef_CO2
        bgc_gross_emis_out = ((bgc_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * bgc_ef_CO2
        deadwood_c_gross_emis_out = ((deadwood_c_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * deadwood_c_ef_CO2
        litter_c_gross_emis_out = ((litter_c_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * litter_c_ef_CO2

        # # For testing CO2 fire emissions
        # print("c_dens_in:", c_dens_in)
        # print("agc_rf:", agc_rf)
        # print("gain_year_count:", gain_year_count)
        # print("agc_pre_disturb:", agc_pre_disturb)
        # print(f"Cf: {Cf}; Gef_co2: {Gef_co2}")
        # print("AGC emission factor for fire:", agc_ef_CO2)
        # print("agc_gross_emis_out:", agc_gross_emis_out)
        # os.quit()

    # No emissions if no fire detected
    else:

        agc_gross_emis_out = 0
        bgc_gross_emis_out = 0
        deadwood_c_gross_emis_out = 0
        litter_c_gross_emis_out = 0

    # Gross emissions as an array
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')


    # Step 6: Updates gross removals to include post-fire gross removals, if applicable (Mg C/ha/interval).
    # gain year count is the number of years between the disturbance and the end of the interval.
    # This uses the same RFs before and after the fire.
    if most_recent_year_burned > 0:

        post_dist_RF = np.array([agc_rf, bgc_rf, agc_rf*deadwood_c_ratio, agc_rf*litter_c_ratio]).astype('float32')
        post_dist_gain_year_count = cn.interval_duration - gain_year_count - 1
        post_dist_gross_removals = post_dist_gain_year_count * post_dist_RF

        c_gross_removals_out = c_gross_removals_out - post_dist_gross_removals

        # print("post_dist_RF:", post_dist_RF)
        # print("gain_year_count:", gain_year_count)
        # print("post_dist_gain_year_count:", post_dist_gain_year_count)
        # print("post_dist_gross_removals:", post_dist_gross_removals)
        # print("c_gross_removals_out_after_dist:", c_gross_removals_out)
        # os.quit()


    # Step 7: Calculates ending carbon densities by carbon pool.
    # Starts with carbon density in (list converted to np array), adds gross removals (subtracts negative value), subtracts emissions.
    # Ending carbon pools are not affected by non-CO2 emissions in the next step.
    c_dens_out = np.array(c_dens_in).astype('float32') - c_gross_removals_out - c_gross_emissions_out


    # Step 7: Calculates non-CO2 emissions (if relevant) (Mg CO2e/ha/interval)
    # Default non-CO2 emissions values
    ch4_flux_out = 0
    n2o_flux_out = 0

    # Only assigns fire node code and calculates CH4 and N2O emissions if the pixel burned in the last interval
    if most_recent_year_burned > 0:

        state_out = accrete_node(node, 1)

        # Selects just the carbon pools that have non-CO2 emissions from fire
        c_pools_for_fire_non_CO2 = np.where(c_pools_fire_non_CO2 == 1, c_pre_disturb, 0)

        # Sums the C pools that have non-CO2 fire emissions. We don't track which C pools the CH4 and N2O emissions come from,
        # so the pools are combined.
        c_pools_for_fire_total = np.sum(c_pools_for_fire_non_CO2)

        # Calculates non-CO2 fire emissions using the selected C pools in the year before disturbance
        ch4_flux_out, n2o_flux_out = non_CO2_fire_equations(c_pools_for_fire_total, Cf, Gef_ch4, Gef_n2o)

        # # For testing non-CO2 emissions
        # print("c_dens_in:", c_dens_in)
        # print("c_pre_disturb:", c_pre_disturb)
        # print(f"Cf: {Cf}; Gef_ch4: {Gef_ch4}; GWP CH4: {cn.gwp_ch4}")
        # print(f"Cf: {Cf}; Gef_n2o: {Gef_n2o}; GWP N2O: {cn.gwp_n2o}")
        # print("c_pools_for_fire_non_CO2:", c_pools_for_fire_non_CO2)
        # print("c_pools_for_fire_total:", c_pools_for_fire_total)
        # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
        # os.quit()

    # Node code if no fire in the last interval. No CH4 and N2O emissions calculated.
    else:

        state_out = accrete_node(node, 2)

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')

    return state_out, c_gross_emissions_out, c_gross_removals_out, non_co2_fluxes_out, c_dens_out, gain_year_count

