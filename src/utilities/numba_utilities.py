import math
import numpy as np
from numba import jit
from numba.typed import Dict
from numba.core import types
import sys

# Project imports
from src.utilities import constants_and_names as cn


# Adds latest decision tree branch to the state node
@jit(nopython=True)
def accrete_node(combined, new_digit):
    combined = combined * 10 + new_digit
    return combined

# Makes all output states the same number of digits (currently 7) by padding 0s to the right
@jit(nopython=True)
def pad_to_6_digits(state_out, max_digits_state_out):

    if state_out < 10 ** (max_digits_state_out-1):
        digits = int(np.log10(state_out)) + 1 if state_out > 0 else 1
        pad_zeros = max_digits_state_out - digits
        state_out = state_out * (10 ** pad_zeros)

    return np.uint32(state_out)

# Calculates a backup continent-ecozone value or climate zone value in case pixels don't have one.
# There are many ways that are more efficient or at least succinct to calculate the mode of an array in Python,
# but they don't work with Numba. So, I'm going with this.
# https://chatgpt.com/share/e/67bf7958-351c-800a-bd00-259213586471
# Excludes continent-ecozone codes for water.
@jit(nopython=True)
def fallback_conteco_climzone_value(conteco_or_climzone_block, fallback_value):

    # Flattens the array
    flat = conteco_or_climzone_block.ravel()

    # Manually masks out values to ignore in calculating dominant continent-ecozone: 0, 1022, 2022, 4022, 7022 (water codes)
    valid = []
    for i in range(flat.size):
        v = flat[i]
        if v != 0 and v != 1022 and v != 2022 and v != 4022 and v != 7022:  # Last 4 are water codes
            valid.append(v)

    # If no valid pixels after excluding the codes to ignore, the fallback value is used
    if len(valid) == 0:
        return fallback_value

    # Converts list to NumPy array
    valid_arr = np.array(valid)

    # Gets the most common (mode) continent-ecozone
    counts = np.bincount(valid_arr)
    return np.argmax(counts)


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


# Classifies GLCLU as short (<5 m) or tall (>= 5 m) vegetation
# No medium-height vegetation used as it's not available from annual GLCLU data.
@jit(nopython=True)
def classify_veg_height(LC):
    short_veg = (((LC >= cn.short_veg_dry_min_code) and (LC <= cn.short_veg_dry_max_code)) or
                      ((LC >= cn.short_veg_wet_min_code) and (LC <= cn.short_veg_wet_max_code)))
    tall_veg = (((LC >= cn.tall_veg_dry_min_code) and (LC <= cn.tall_veg_dry_max_code)) or
                ((LC >= cn.tall_veg_wet_min_code) and (LC <= cn.tall_veg_wet_max_code)))

    return short_veg, tall_veg


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
    if interval_end_year == (cn.first_model_year_5_years + cn.five_year_interval_duration):

        # Criteria for excluding tall vegetation land cover
        not_tall_veg_condition = (
                (LC_prev < cn.tall_veg_dry_min_code) |
                ((LC_prev > cn.tall_veg_dry_max_code) & (LC_prev < cn.tall_veg_wet_min_code))
                | (LC_prev > cn.tall_veg_wet_max_code)
        )

        # Sets cell to the model start year wherever land cover is not tall vegetation
        if not_tall_veg_condition == 1:
            most_recent_year_not_forest = cn.first_model_year_5_years


    # Checks the current end of interval land cover
    # Criteria for excluding tall vegetation land cover
    not_tall_veg_condition = (
            (LC_curr < cn.tall_veg_dry_min_code) |
            ((LC_curr > cn.tall_veg_dry_max_code) & (LC_curr < cn.tall_veg_wet_min_code))
            | (LC_curr > cn.tall_veg_wet_max_code)
    )

    # Sets cell to interval end year wherever land cover is not tall vegetation
    if not_tall_veg_condition == 1:
        most_recent_year_not_forest = interval_end_year

    return most_recent_year_not_forest


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


# Returns AGC and BGC one-time removal factors for the gain of short-height vegetation (Mg C/ha)
# RF values are from IPCC 2006, V4, Ch. 6, Table 6.4- DEFAULT BIOMASS STOCKS PRESENT ON GRASSLAND, AFTER CONVERSION FROM OTHER LAND USE (no 2019 update).
# short_veg_AGB_RF is from the "peak above-ground biomass" columns.
# short_veg_BGB_RF is the difference between short_veg_AGB_RF and the "total (above-ground and below-ground) non-woody biomass" column.
# Climate zone is from IPCC 2019 Corrigenda map. See constants_and_names.py for more information.
@jit(nopython=True)
def calc_short_veg_removals(climate_zone):

    if climate_zone >= 9:  # Boreal- dry and wet (and polar)
        short_veg_AGB_RF = 1.7
        short_veg_BGB_RF = (8.5 - short_veg_AGB_RF)
    elif climate_zone == 8:  # Cold temperate- dry
        short_veg_AGB_RF = 1.7
        short_veg_BGB_RF = (6.5 - short_veg_AGB_RF)
    elif climate_zone == 7:  # Cold temperate- wet
        short_veg_AGB_RF = 2.4
        short_veg_BGB_RF = (13.6 - short_veg_AGB_RF)
    elif climate_zone == 6:  # Warm temperate- dry
        short_veg_AGB_RF = 1.6
        short_veg_BGB_RF = (6.1 - short_veg_AGB_RF)
    elif climate_zone == 5:  # Warm temperate- wet
        short_veg_AGB_RF = 2.7
        short_veg_BGB_RF = (13.5 - short_veg_AGB_RF)
    elif climate_zone == 4:  # Tropical- dry
        short_veg_AGB_RF = 2.3
        short_veg_BGB_RF = (8.7 - short_veg_AGB_RF)
    elif climate_zone == 2 or climate_zone == 3:  # Tropical- wet/moist
        short_veg_AGB_RF = 6.2
        short_veg_BGB_RF = (16.1 - short_veg_AGB_RF)
    elif climate_zone == 1:  # Tropical- montane (average of tropical dry and tropical wet/moist values)
        short_veg_AGB_RF = (2.3 + 6.2)/2
        short_veg_BGB_RF = (((8.7 + 16.1)/2) - short_veg_AGB_RF)
    else: # Outside ecozone bounds-- apply boreal values
        short_veg_AGB_RF = 1.7
        short_veg_BGB_RF = (8.5-short_veg_AGB_RF)

    short_veg_AGC_RF = short_veg_AGB_RF * cn.biomass_to_carbon_non_mangrove
    short_veg_BGC_RF = short_veg_BGB_RF * cn.biomass_to_carbon_non_mangrove

    return short_veg_AGC_RF, short_veg_BGC_RF


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
def calc_primary_forest_RF(continent_ecozone_cell, primary_forest_RF_array):

    primary_forest_RF_indices = np.where(primary_forest_RF_array[:, 0] == continent_ecozone_cell)

    # Checks if there are matching indices and extracts corresponding primary forest RF
    if primary_forest_RF_indices[0].size > 0:  # If matching continent-ecozone combination...
        primary_forest_RF = primary_forest_RF_array[primary_forest_RF_indices[0][0], 1]  # Uses matching RF
    else:  # If no matching continent-ecozone combination...
        primary_forest_RF = np.mean(primary_forest_RF_array[:, 1]) # Uses average of all primary forest RFs

    return primary_forest_RF


# Returns the emission factors for partially disturbed forest by driver based on the continent-ecozone combination (unit: fraction AGC lost)
# From https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67feb6f2-c124-800a-9279-61f0e3a67faf
@jit(nopython=True)
def calc_partial_disturbance_EFs(drivers_cell, continent_ecozone_cell, partial_disturbance_EF_array):

    # # For testing-- force driver or continent_ecozone values
    # drivers_cell = 3
    # continent_ecozone_cell = 9999

    # Selects correct driver column. Sets fallback to driver 4 (logging) if driver value isn't legitimate.
    if 1 <= drivers_cell <= 7:
        col_index = drivers_cell  # EF columns are at indexes 1–7 (drivers 1-7)
    else:
        col_index = 4  # Defaults to 5th column (index 4: logging driver)

    partial_disturbance_EF_indices = np.where(partial_disturbance_EF_array[:, 0] == continent_ecozone_cell)

    if partial_disturbance_EF_indices[0].size > 0:
        row_index = partial_disturbance_EF_indices[0][0]
        partial_disturbance_EF = partial_disturbance_EF_array[row_index, col_index]
    else:
        # Manual mean of the specified column (col_index) because numba has all kinds of restrictions!
        total = 0.0
        n_rows = partial_disturbance_EF_array.shape[0]
        for i in range(n_rows):
            total += partial_disturbance_EF_array[i, col_index]
        partial_disturbance_EF = total / n_rows

    return partial_disturbance_EF

# Calculates Cf for fire emissions from forests (as opposed to savanna/grassland or crop residue burning).
# From IPCC 2019 Table 2.6 (unitless)
@jit(nopython=True)
def calc_Cf_forest(climate_domain_cell, drivers_cell, ifl_primary_cell):

    # Groups of drivers with different Cfs
    driver_group_1 = [cn.permanent_agriculture, cn.shifting_cultivation, cn.hard_commodities, cn.wildfire, cn.settlements_and_infrastruct]
    driver_group_2 = [cn.forest_management]
    driver_group_3 = [cn.other_natural_disturbances]

    if climate_domain_cell == 1:  # Tropical/subtropical
        if ifl_primary_cell:  # Tropical/subtropical, primary forest
            Cf_forest = 0.36  # Row "All primary tropical forest"
        else:  # Tropical/subtropical, not primary forest
            Cf_forest = 0.55  # Row "All secondary tropical forest"
    elif climate_domain_cell == 2:   # Temperate
        if drivers_cell in driver_group_1:  # Temperate, driver group 1
            Cf_forest = 0.51     # Row "Felled and burned (land-clearing fire)" temperate forest
        elif drivers_cell in driver_group_2:  # Temperate, driver group 2
            Cf_forest = 0.62     # Row "Post logging slash burn" temperate forest
        elif drivers_cell in driver_group_3:  # Temperate, driver group 3
            Cf_forest = 0.45     # Row "all other temperate forest"
        else:  # Temperate, no driver assigned
            Cf_forest = 0.45
    elif climate_domain_cell == 3:  # Boreal
        if drivers_cell in driver_group_1:  # Boreal, driver group 1
            Cf_forest = 0.59     # Row "Land clearing fire" boreal forest
        elif drivers_cell in driver_group_2:  # Boreal, driver group 2
            Cf_forest = 0.33     # Row "Post logging slash burn" boreal forest
        elif drivers_cell in driver_group_3:  # Boreal, driver group 3
            Cf_forest = 0.34     # Row "All boreal forest"
        else:  # Boreal, no driver assigned
            Cf_forest = 0.34     # Row "All boreal forest"
    else:  # Outside ecozone bounds
        if drivers_cell in driver_group_1:  # Outside ecozone bounds, driver group 1
            Cf_forest = 0.59     # Row "Land clearing fire" boreal forest
        elif drivers_cell in driver_group_2:  # Outside ecozone bounds, driver group 2
            Cf_forest = 0.33     # Row "Post logging slash burn" boreal forest
        elif drivers_cell in driver_group_3:  # Outside ecozone bounds, driver group 2
            Cf_forest = 0.34     # Row "All boreal forest"
        else:  # Outside ecozone bounds, no driver assigned
            Cf_forest = 0.34     # Row "All boreal forest"

    return Cf_forest


# Calculates Gef for fire emissions from forests (as opposed to savanna/grassland or biofuel burning).
# From IPCC 2019 Table 2.5 (g respective gas/kg dry matter)
@jit(nopython=True)
def calc_Gef_forest(climate_domain_cell):

    if climate_domain_cell == 1:  # Tropical/subtropical
        Gef_CO2_forest = 1580.0   # Row "tropical forest"
        Gef_CH4_forest = 6.8      # Row "tropical forest"
        Gef_N2O_forest = 0.2      # Row "tropical forest"
    elif climate_domain_cell == 2 or climate_domain_cell == 3:   # Temperate/boreal
        Gef_CO2_forest = 1569.0   # Row "extra-tropical forest"
        Gef_CH4_forest = 4.7      # Row "extra-tropical forest"
        Gef_N2O_forest = 0.26     # Row "extra-tropical forest"
    else:  # Outside ecozone bounds
        Gef_CO2_forest = 1569.0   # Row "extra-tropical forest"
        Gef_CH4_forest = 4.7      # Row "extra-tropical forest"
        Gef_N2O_forest = 0.26     # Row "extra-tropical forest"

    return Gef_CO2_forest, Gef_CH4_forest, Gef_N2O_forest


# Calculates non-CO2 emissions (CH4 and N2O) separately.
# Cf_forest is the combustion factor
# Gef_ch4 and Gef_n2o are the emission factors for their respective gases.
# biomass_to_carbon can be hard-coded as non-mangrove because we assume that mangroves don't have fires.
# From IPCC 2019 Eqn. 2.27
@jit(nopython=True)
def non_CO2_fire_equations(carbon_in, Cf, Gef_ch4, Gef_n2o):

    # print(f"Carbon in: {carbon_in}; Cf_forest: {Cf_forest}; Gef_ch4: {Gef_ch4}; GWP CH4: {cn.gwp_ch4}")

    ch4_flux_out = (carbon_in/cn.biomass_to_carbon_non_mangrove) * Cf * Gef_ch4 * cn.g_to_kg * cn.gwp_ch4
    n2o_flux_out = (carbon_in/cn.biomass_to_carbon_non_mangrove) * Cf * Gef_n2o * cn.g_to_kg * cn.gwp_n2o

    # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
    # os.quit()

    return ch4_flux_out, n2o_flux_out


# Gross fluxes and ending carbon stocks for non-tree converted to tree.
# Carbon pool fluxes and densities are input and output as Mg C/ha(/interval) rather than Mg CO2 for arithmetic simplicity.
# Applies to 5-year intervals and annual intervals. Only difference is the gain_year_count.
@jit(nopython=True)
def calc_NT_T(interval_length, agc_rf, bgc_rf, c_dens_in, deadwood_c_ratio, litter_c_ratio):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    # Step 1: Calculates the number of years of carbon gain (years)
    if interval_length == 5:
        gain_year_count = cn.NT_T_gain_year_count_default
    elif interval_length == 1:
        gain_year_count = 1
    else:
        raise ValueError("interval_length not valid: must be 1 or 5")

    # Step 2: Calculates gross removals by carbon pools (Mg C/ha/interval). Gross removals are negative.
    agc_gross_removals_out = float((agc_rf * gain_year_count) * -1)  #float() necessary for Numba typing
    bgc_gross_removals_out = float((bgc_rf * gain_year_count) * -1)  #float() necessary for Numba typing
    deadwood_c_gross_removals_out = agc_gross_removals_out * deadwood_c_ratio
    litter_c_gross_removals_out = agc_gross_removals_out * litter_c_ratio

    # Step 3: Calculates gross emissions by carbon pools (Mg C/ha/interval). Gross emissions are positive.
    # There are no gross emissions in NT->T pixels, so C pool emissions set to 0.
    # Included here just for completeness.
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

    # Step 6: Increments the forest age (years).
    # Forest age starts at 0 years by definition for NT->T.
    forest_age = 0 + gain_year_count

    return c_gross_emissions_out, c_gross_removals_out, c_dens_out, gain_year_count, forest_age


# Gross fluxes and ending carbon stocks for trees converted to non-trees with and without fire.
# Non-CO2 gas emissions are only calculated if fire was detected during the interval.
# CO2 emissions are calculated differently depending on if fire was detected during the interval and if a Gef_CO2 is supplied.
# Carbon pool fluxes and densities are input and output as Mg C/ha(/interval) rather than Mg CO2 for arithmetic simplicity.
# Applies to 5-year intervals and annual intervals. Main difference is that the calculation of gain before loss
# only applies to the former.
@jit(nopython=True)
def calc_T_NT(node, interval_length, burned_in_curr_interval, RF_AGC_in, RF_BGC_in, c_pools_fire_CO2, c_pools_fire_non_CO2, c_pools_no_fire,
              forest_dist_last, interval_end_year, c_dens_in,
              post_dist_regrowth, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4, Gef_n2o,
              deadwood_c_ratio, litter_c_ratio):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    # Establishes which carbon pools are emitted depending on whether fire was detected during the interval.
    # For T->NT, emission factors are binary (1 means full emissions, 0 means no emissions).
    # Carbon pools that are emitted as CO2 if fire was detected.
    if burned_in_curr_interval:
        agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_fire_CO2)
    else:
        # Carbon pools that are emitted as CO2 if fire was not detected.
        agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_no_fire)


    # Step 1: Calculates the number of years of carbon gain before loss occurred (years).
    # Annual model has no gain before loss, so gain_year_count = 0.
    if interval_length == 5:
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
            gain_year_count = forest_dist_last - ((interval_end_year - cn.first_model_year_5_years) - cn.five_year_interval_duration) - 1
        else:
            # If a forest disturbance was not detected, the disturbance is assumed to occur in the middle of the interval
            # (year t-2), with removals until then (years t-4 and t-3). There are no removals in the year of assumed
            # disturbance or the years after.
            gain_year_count = math.floor(cn.five_year_interval_duration / 2)

    elif interval_length == 1:
        gain_year_count = 0

    else:
        raise ValueError("interval_length not valid: must be 1 or 5")


    # Step 2: Assigns deadwood C and litter C ratios for removal factors, if relevant (unitless).
    # Deadwood and litter C removals only occur in pixels that were not tall vegetation at some point (natural forest only).
    # Thus, we need to check whether the pixel was ever not tall vegetation at some point during the model before the end of this interval.
    # If conditions aren't met, the deadwood and litter ratios are set to 0 (no removals).
    # For simplicity, there are no deadwood or litter removals in loss intervals.
    # These reatios aren't used for annual intervals (just used to calculate gain before loss) but not limiting it to just 5-year intervals
    # because it's not much computation.
    if (most_recent_year_not_tall_veg == 0) or (most_recent_year_not_tall_veg == interval_end_year):
        deadwood_c_ratio = 0.0
        litter_c_ratio = 0.0


    # Step 3: Calculates pre-disturbance gross removals by carbon pools (Mg C/ha/interval). Gross removals are negative.
    # This should only have a non-0 value for 5-year intervals; it should be 0 for annual intervals.
    if interval_length == 5:

        # Assigns the pre-disturbance RFs to the output RFs for the interval.
        # This way, the pre-disturbance RFs are reported for this 5-year interval.
        RF_AGC_out = RF_AGC_in
        RF_BGC_out = RF_BGC_in

        agc_gross_removals_out = float((RF_AGC_in * gain_year_count) * -1)   # float() necessary for Numba typing
        bgc_gross_removals_out = float((RF_BGC_in * gain_year_count) * -1)   # float() necessary for Numba typing
        deadwood_c_gross_removals_out= agc_gross_removals_out * deadwood_c_ratio
        litter_c_gross_removals_out= agc_gross_removals_out * litter_c_ratio

        # Consolidates outputs into array to reduce the number of arguments returned to the decision tree.
        # Must specify float32 because numba is quite particular about datatypes.
        c_gross_removals_out = np.array([agc_gross_removals_out, bgc_gross_removals_out, deadwood_c_gross_removals_out, litter_c_gross_removals_out]).astype('float32')

    # Assigns pre-disturbance RFs and gross removals 0 for consistency between 5-year and annual intervals
    elif interval_length == 1:

        # Because there are no removals during a 1-year interval in which there is tree loss, RFs are reassigned to 0.
        # This way, no RFs are reported for this 1-year interval.
        RF_AGC_out = 0
        RF_BGC_out = 0

        # Assigns 0 to gross removals for annual intervals because no removals in the year of loss.
        agc_gross_removals_out = 0
        bgc_gross_removals_out = 0
        deadwood_c_gross_removals_out = 0
        litter_c_gross_removals_out = 0
        c_gross_removals_out = np.array([agc_gross_removals_out, bgc_gross_removals_out, deadwood_c_gross_removals_out, litter_c_gross_removals_out]).astype('float32')

    else:
        raise ValueError("interval_length not valid: must be 1 or 5")


    # Step 4: Calculates carbon densities at the year of loss by carbon pool (Mg C/ha). This is not output from the model.
    # For 5-year intervals, C pools pre-disturbance can differ from input carbon pools because there can be gain before loss.
    # For annual intervals, C pools pre-disturbance are the same as input carbon pools because there is no gain before loss.
    if interval_length == 5:
        agc_pre_disturb = agc_dens_in - agc_gross_removals_out  # Gross removals is negative, so this adds carbon
        bgc_pre_disturb = bgc_dens_in - bgc_gross_removals_out
        deadwood_c_pre_disturb = deadwood_c_dens_in - deadwood_c_gross_removals_out
        litter_c_pre_disturb = litter_c_dens_in - litter_c_gross_removals_out

        # Pre-disturbance carbon densities as an array, used as input for non-CO2 fire emissions and post-disturbance removals (if applicable)
        c_pre_disturb = np.array([agc_pre_disturb, bgc_pre_disturb, deadwood_c_pre_disturb, litter_c_pre_disturb]).astype('float32')

    elif interval_length == 1:
        agc_pre_disturb = agc_dens_in
        bgc_pre_disturb = bgc_dens_in
        deadwood_c_pre_disturb = deadwood_c_dens_in
        litter_c_pre_disturb = litter_c_dens_in
        c_pre_disturb = np.array(c_dens_in).astype('float32')

    else:
        raise ValueError("interval_length not valid: must be 1 or 5")


    # Step 5: Calculates CO2 gross emissions by carbon pools (Mg C/ha/interval). Gross emissions are positive.
    # Which pools are emitted is controlled by the ef_CO2 flags.
    agc_gross_emis_out = agc_pre_disturb * agc_ef_CO2
    bgc_gross_emis_out = bgc_pre_disturb * bgc_ef_CO2
    deadwood_c_gross_emis_out = deadwood_c_pre_disturb * deadwood_c_ef_CO2
    litter_c_gross_emis_out = litter_c_pre_disturb * litter_c_ef_CO2

    # Consolidates outputs into array to reduce the number of arguments returned to the decision tree.
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')


    # Step 6: Updates gross removals to include one-time post-disturbance regrowth,
    # if applicable (medium height veg and cropland) (Mg C/ha/interval).
    # Regrowth of medium height veg and cropland is a one-time value, not annual, so no multiplication by gain year count.
    # Post-disturbance regrowth can occur for 5-year and annual intervals because either can have an ending land cover
    # with aboveground carbon.
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
    if burned_in_curr_interval:

        state_out = accrete_node(node, cn.land_state_node_fire_value)

        # Selects just the carbon pools that have non-CO2 emissions from fire
        c_pools_for_fire_non_CO2 = np.where(c_pools_fire_non_CO2 == 1, c_pre_disturb, 0)

        # Sums the C pools that have non-CO2 fire emissions. We don't track which C pools the CH4 and N2O emissions come from,
        # so the pools are combined.
        c_pools_for_fire_total = np.sum(c_pools_for_fire_non_CO2)

        # Calculates non-CO2 fire emissions using the selected C pools in the year before disturbance
        ch4_flux_out, n2o_flux_out = non_CO2_fire_equations(c_pools_for_fire_total, Cf_forest, Gef_ch4, Gef_n2o)

        # # For testing non-CO2 emissions
        # print("c_dens_in:", c_dens_in)
        # print("c_pre_disturb:", c_pre_disturb)
        # print(f"Cf_forest: {Cf_forest}; Gef_ch4: {Gef_ch4}; GWP CH4: {cn.gwp_ch4}")
        # print(f"Cf_forest: {Cf_forest}; Gef_n2o: {Gef_n2o}; GWP N2O: {cn.gwp_n2o}")
        # print("c_pools_for_fire_non_CO2:", c_pools_for_fire_non_CO2)
        # print("c_pools_for_fire_total:", c_pools_for_fire_total)
        # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
        # os.quit()

    # Node code if no fire in the last interval. No CH4 and N2O emissions calculated.
    else:

        state_out = accrete_node(node, 2)

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')

    # # For testing
    # if burned_in_prev_interval:
    #
    #     print("agc_dens_in:", agc_dens_in)
    #     print("agc_gross_removals_out:", agc_gross_removals_out)
    #     print("agc_pre_disturb:", agc_pre_disturb)
    #     print("biomass_to_carbon_non_mangrove:", cn.biomass_to_carbon_non_mangrove)
    #     print("Cf_forest:", Cf_forest)
    #     print("Gef_ch4:", Gef_ch4)
    #     print("Gef_n2o:", Gef_n2o)
    #     print("agc_gross_emis_out:", agc_gross_emis_out)
    #     print("c_dens_out:", c_dens_out)
    #     os.quit()

    # Step 9: Resets the forest age to 0 because there was a stand-replacing disturbance
    forest_age_interval_end = 0

    return (state_out, c_gross_emissions_out, c_gross_removals_out, non_co2_fluxes_out, c_dens_out,
            RF_AGC_out, RF_BGC_out, agc_ef_CO2, gain_year_count, forest_age_interval_end)


# Gross fluxes and ending carbon stocks for trees remaining trees with non-stand-replacing disturbances.
# Carbon pool fluxes and densities are input and output as Mg C/ha(/interval) rather than Mg CO2 for arithmetic simplicity.
# Applies to 5-year intervals and annual intervals.
@jit(nopython=True)
def calc_T_T_non_stand_disturbs(node, interval_length, burned_in_curr_interval, RF_AGC_pre_dist_in, RF_BGC_pre_dist_in,
                                c_pools_fire_CO2, c_pools_fire_non_CO2, c_pools_no_fire,
                                forest_dist_last, interval_end_year, c_dens_in,
                                RF_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_co2, Gef_ch4, Gef_n2o,
                                deadwood_c_ratio, litter_c_ratio):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    # Establishes which carbon pools are emitted depending on whether fire was detected during the interval.
    # For T->T, emission factors can range between 0 and 1, with 0 meaning no emissions and 1 meaning full emissions.
    if burned_in_curr_interval:      # Carbon pools that are emitted as CO2 if fire was detected.
        agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_fire_CO2)
    else:          # Carbon pools that are emitted as CO2 if fire was not detected.
        agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_no_fire)


    # Step 1: Calculates the number of years of carbon gain before the non-stand-replacing disturbance occurred (years).
    # Annual model has no gain before disturbance (no gain in disturbance year),
    # so in this function gain_year_count_pre_dist = 0 always.
    if interval_length == 5:
        if forest_dist_last > 0:
            # If a forest disturbance was detected, the gain_year_count_pre_dist are the number of years until detection of the last disturbance.
            # There is no growth in the year of disturbance or the years after.
            # The - 1 at the excludes the disturbance year from the gain_year_count_pre_dist since we decided there are no removals in the disturbance year.
            # For example, if the time interval is 2010-2015 and the disturbance is detected in 2013 (t-2),
            # there should be 2 years of growth (years t-4 and t-3, 2011 and 2012).
            # This table illustrates each case for the example interval of 2010-2015.
            # 0 years         11               - ((2015              - 2000)                - 5) - 1   (year t-4)
            # 1 years         12               - ((2015              - 2000)                - 5) - 1   (year t-3)
            # 2 years         13               - ((2015              - 2000)                - 5) - 1   (year t-2)
            # 3 years         14               - ((2015              - 2000)                - 5) - 1   (year t-1)
            # 4 years         15               - ((2015              - 2000)                - 5) - 1   (year t)
            gain_year_count_pre_dist = forest_dist_last - ((interval_end_year - cn.first_model_year_5_years) - cn.five_year_interval_duration) - 1
        else:
            # If a forest disturbance was not detected, the disturbance is assumed to occur in the middle of the interval
            # (year t-2), with removals until then (years t-4 and t-3). There are no removals in the year of assumed
            # disturbance or the years after.
            gain_year_count_pre_dist = math.floor(cn.five_year_interval_duration / 2)

    elif interval_length == 1:
        gain_year_count_pre_dist = 0

    else:
        raise ValueError("interval_length not valid: must be 1 or 5")


    # Step 2: Assigns deadwood C and litter C ratios for removal factors, if relevant (unitless).
    # Deadwood and litter C removals only occur in pixels that were not tall vegetation at some point (natural forest only).
    # Thus, we need to check whether the pixel was non-tall vegetation at some point during the model before the end of this interval.
    # If conditions aren't met, the deadwood and litter ratios are set to 0 (no removals).
    # This isn't used for annual intervals (just used to calculate gain before loss) but not limiting it to just 5-year intervals
    # because it's not much computation.
    if most_recent_year_not_tall_veg == 0 or most_recent_year_not_tall_veg == interval_end_year:
        deadwood_c_ratio = 0.0
        litter_c_ratio = 0.0


    # Step 3: Calculates pre-disturbance gross removals by carbon pools (Mg C/ha/interval) for 5-year intervals. Gross removals are negative.
    # This should only have a non-0 value for 5-year intervals; it should be 0 for annual intervals.
    if interval_length == 5:

        # Assigns the pre-disturbance RFs to the output RFs for the interval.
        # This way, the pre-disturbance RFs are reported for this 5-year interval.
        RF_AGC_pre_dist_out = RF_AGC_pre_dist_in
        RF_BGC_pre_dist_out = RF_BGC_pre_dist_in

        agc_gross_removals_out = float((RF_AGC_pre_dist_in * gain_year_count_pre_dist) * -1) #float() necessary for Numba typing
        bgc_gross_removals_out = float((RF_BGC_pre_dist_in * gain_year_count_pre_dist) * -1) #float() necessary for Numba typing
        deadwood_c_gross_removals_out= agc_gross_removals_out * deadwood_c_ratio
        litter_c_gross_removals_out= agc_gross_removals_out * litter_c_ratio

        # Consolidates outputs into array to reduce the number of arguments returned to the decision tree.
        # Must specify float32 because numba is quite particular about datatypes.
        c_gross_removals_out = np.array([agc_gross_removals_out, bgc_gross_removals_out, deadwood_c_gross_removals_out, litter_c_gross_removals_out]).astype('float32')

    # Assigns pre-disturbance RFs and gross removals 0 for consistency between 5-year and annual intervals.
    # There are no removals in the year of disturbance, so we know removals in a partially disturbed forest with annual intervals
    # is always 0 and can skip the calculations in the 5-year interval branch to save some time.
    elif interval_length == 1:

        # Because there are no removals during a 1-year interval in which there is tree loss, RFs are reassigned to 0.
        # This way, not RFs are reported for this 1-year interval.
        RF_AGC_pre_dist_out = 0
        RF_BGC_pre_dist_out = 0

        # Assigns gross removals to 0 for annual intervals because no removals in the year of loss.
        agc_gross_removals_out = 0
        bgc_gross_removals_out = 0
        deadwood_c_gross_removals_out = 0
        litter_c_gross_removals_out = 0
        c_gross_removals_out = np.array([agc_gross_removals_out, bgc_gross_removals_out, deadwood_c_gross_removals_out, litter_c_gross_removals_out]).astype('float32')

    else:
        raise ValueError("interval_length not valid: must be 1 or 5")


    # Step 4: Calculates carbon densities at the year of disturbance by carbon pool (Mg C/ha). This is not output from the model.
    # For 5-year intervals, C pools pre-disturbance differ from input carbon pools.
    # For annual intervals, C pools pre-disturbance are the same as input carbon pools because there is no gain before loss.
    if interval_length == 5:
        agc_pre_disturb = agc_dens_in - agc_gross_removals_out
        bgc_pre_disturb = bgc_dens_in - bgc_gross_removals_out
        deadwood_c_pre_disturb = deadwood_c_dens_in - deadwood_c_gross_removals_out
        litter_c_pre_disturb = litter_c_dens_in - litter_c_gross_removals_out

        # Pre-disturbance carbon densities as an array, used as input for non-CO2 fire emissions and post-disturbance removals (if applicable)
        c_pre_disturb = np.array([agc_pre_disturb, bgc_pre_disturb, deadwood_c_pre_disturb, litter_c_pre_disturb]).astype('float32')

    elif interval_length == 1:
        agc_pre_disturb = agc_dens_in
        bgc_pre_disturb = bgc_dens_in
        deadwood_c_pre_disturb = deadwood_c_dens_in
        litter_c_pre_disturb = litter_c_dens_in
        c_pre_disturb = np.array(c_dens_in).astype('float32')

    else:
        raise ValueError("interval_length not valid: must be 1 or 5")


    # Step 5: Calculates CO2 gross emissions by carbon pools (Mg C/ha/interval). Gross emissions are positive.

    # Calculates CO2 emissions from fire for each C pool using fire emission factors
    # if a Gef for CO2 is supplied AND if there was fire during the interval.
    # This is used for non-stand replacing forest disturbances (as opposed to entire C pools being combusted).
    # From IPCC 2019 Eqn. 2.27
    if burned_in_curr_interval:

        # Equations divide by C_to_CO2 to put the emissions back in Mg C/ha. They are later converted back to Mg CO2/ha,
        # but we need CO2 fire emissions in Mg C/ha here for consistency with all other outputs.
        agc_gross_emis_out = ((agc_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf_forest * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * agc_ef_CO2
        bgc_gross_emis_out = ((bgc_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf_forest * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * bgc_ef_CO2
        deadwood_c_gross_emis_out = ((deadwood_c_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf_forest * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * deadwood_c_ef_CO2
        litter_c_gross_emis_out = ((litter_c_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf_forest * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * litter_c_ef_CO2

        # Emission factor for burned forest is the combustion factor for forrest
        agc_ef_CO2 = Cf_forest

        # # For testing CO2 fire emissions
        # print("c_dens_in:", c_dens_in)
        # print("agc_rf:", RF_AGC_pre_dist)
        # print("gain_year_count_pre_dist:", gain_year_count_pre_dist)
        # print("agc_pre_disturb:", agc_pre_disturb)
        # print(f"Cf_forest: {Cf_forest}; Gef_co2: {Gef_co2}")
        # print("AGC emission factor for fire:", agc_ef_CO2)
        # print("agc_gross_emis_out:", agc_gross_emis_out)
        # sys.exit()

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
    # gain_year_count_post_dist here is the number of years between the disturbance and the end of the interval.
    # This applies only to 5-year interval data. There is no gross removals adjustment to annual data.
    if interval_length == 5:
        gain_year_count_post_dist = cn.five_year_interval_duration - gain_year_count_pre_dist - 1
        post_dist_gross_removals = gain_year_count_post_dist * RF_post_dist

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
    if burned_in_curr_interval:

        state_out = accrete_node(node, cn.land_state_node_fire_value)

        # Selects just the carbon pools that have non-CO2 emissions from fire
        c_pools_for_fire_non_CO2 = np.where(c_pools_fire_non_CO2 == 1, c_pre_disturb, 0)

        # Sums the C pools that have non-CO2 fire emissions. We don't track which C pools the CH4 and N2O emissions come from,
        # so the pools are combined.
        c_pools_for_fire_total = np.sum(c_pools_for_fire_non_CO2)

        # Calculates non-CO2 fire emissions using the selected C pools in the year before disturbance
        ch4_flux_out, n2o_flux_out = non_CO2_fire_equations(c_pools_for_fire_total, Cf_forest, Gef_ch4, Gef_n2o)

        # # For testing non-CO2 emissions
        # print("c_dens_in:", c_dens_in)
        # print("c_pre_disturb:", c_pre_disturb)
        # print(f"Cf_forest: {Cf_forest}; Gef_ch4: {Gef_ch4}; GWP CH4: {cn.gwp_ch4}")
        # print(f"Cf_forest: {Cf_forest}; Gef_n2o: {Gef_n2o}; GWP N2O: {cn.gwp_n2o}")
        # print("c_pools_for_fire_non_CO2:", c_pools_for_fire_non_CO2)
        # print("c_pools_for_fire_total:", c_pools_for_fire_total)
        # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
        # os.quit()

    # Node code if no fire in the last interval. No CH4 and N2O emissions calculated.
    else:

        state_out = accrete_node(node, 2)

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')

    # Step 9: Resets the forest age to 0 because this function always has a partial disturbance
    # either with or without fire-- it doesn't matter. Age is reset either way.
    forest_age_interval_end = 0

    return (state_out, c_gross_emissions_out, c_gross_removals_out, non_co2_fluxes_out, c_dens_out,
            RF_AGC_pre_dist_out, RF_BGC_pre_dist_out, agc_ef_CO2, gain_year_count_pre_dist, forest_age_interval_end)


# Gross fluxes and ending carbon stocks for trees remaining trees with non-stand-replacing disturbances (fires are allowed).
# Carbon pool fluxes and densities are input and output as Mg C/ha(/interval) rather than Mg CO2 for arithmetic simplicity.
# Applies to 5-year intervals and annual intervals.
@jit(nopython=True)
def calc_T_T_no_disturbs(node, interval_length, forest_age_interval_start, first_year_burned_during_interval, RF_AGC, RF_BGC,
                         c_pools_fire_CO2, c_pools_fire_non_CO2, interval_end_year, c_dens_in,
                         most_recent_year_not_tall_veg, Cf_forest, Gef_co2, Gef_ch4, Gef_n2o,
                         deadwood_c_ratio, litter_c_ratio):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    # Establishes which carbon pools are emitted if a fire was detected during the interval.
    agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_fire_CO2)


    # Step 1: Calculates the number of years of carbon gain before a fire occurred (years).
    # Note that removals continue after the year of fire, too, for 5-year intervals. This gain_year_count_pre_dist is used to determine removals
    # until the fire (i.e. carbon densities at the year of fire).
    # Annual model has no gain in the year of disturbance (including fire),
    # so gain_year_count_pre_dist = 0 when there is fire and = 1 when there is no fire.
    # If there were multiple years of fires during the interval, the first one is used.
    if interval_length == 5:
        if first_year_burned_during_interval > 0:
            # If a forest disturbance was detected, the gain_year_count_pre_dist are the number of years until detection of the last disturbance.
            # There is no growth in the year of disturbance or the years after.
            # The - 1 at the excludes the disturbance year from the gain_year_count_pre_dist since we decided there are no removals in the disturbance year.
            # For example, if the time interval is 2010-2015 and the disturbance is detected in 2013 (t-2),
            # there should be 2 years of growth (years t-4 and t-3, 2011 and 2012).
            # This table illustrates each case for the example interval of 2010-2015.
            # 0 years         11               - ((2015              - 2000)                - 5) - 1   (year t-4)
            # 1 years         12               - ((2015              - 2000)                - 5) - 1   (year t-3)
            # 2 years         13               - ((2015              - 2000)                - 5) - 1   (year t-2)
            # 3 years         14               - ((2015              - 2000)                - 5) - 1   (year t-1)
            # 4 years         15               - ((2015              - 2000)                - 5) - 1   (year t)
            gain_year_count_pre_dist = first_year_burned_during_interval - ((interval_end_year - cn.first_model_year_5_years) - cn.five_year_interval_duration) - 1
        else:
            # If no fire was detected, removals occurred every year
            gain_year_count_pre_dist = cn.five_year_interval_duration

    elif interval_length == 1:
        if first_year_burned_during_interval > 0:
            gain_year_count_pre_dist = 0  # No removals in a disturbance/fire year, so no removals during annual interval with fire
        else:
            gain_year_count_pre_dist = 1  # One year of gain when there is no fire

    else:
        raise ValueError("interval_length not valid: must be 1 or 5")


    # Step 2: Assigns deadwood C and litter C ratios for removal factors, if relevant (unitless).
    # Deadwood and litter C removals only occur in pixels that were not tall vegetation at some point (natural forest only).
    # Thus, we need to check whether the pixel was non-tall vegetation at some point during the model before the end of this interval.
    # If conditions aren't met, the deadwood and litter ratios are set to 0 (no removals).
    # For simplicity, there are no deadwood or litter removals in loss intervals.
    # This isn't used for annual intervals (just used to calculate gain before loss) but not limiting it to just 5-year intervals
    # because it's not much computation.
    if most_recent_year_not_tall_veg == 0 or most_recent_year_not_tall_veg == interval_end_year:
        deadwood_c_ratio = 0.0
        litter_c_ratio = 0.0


    # Step 3: Calculates pre-disturbance gross removals by carbon pools (Mg C/ha/interval) for 5-year and annual intervals. Gross removals are negative.
    # Works for 5-year and annual intervals alike.
    agc_gross_removals_out = float((RF_AGC * gain_year_count_pre_dist) * -1) #float() necessary for Numba typing
    bgc_gross_removals_out = float((RF_BGC * gain_year_count_pre_dist) * -1) #float() necessary for Numba typing
    deadwood_c_gross_removals_out= agc_gross_removals_out * deadwood_c_ratio
    litter_c_gross_removals_out= agc_gross_removals_out * litter_c_ratio

    # Consolidates outputs into array to reduce the number of arguments returned to the decision tree.
    # Must specify float32 because numba is quite particular about datatypes.
    c_gross_removals_out = np.array([agc_gross_removals_out, bgc_gross_removals_out, deadwood_c_gross_removals_out, litter_c_gross_removals_out]).astype('float32')


    # Step 4: Calculates carbon densities at the year of fire by carbon pool (Mg C/ha). This is not output from the model.
    # For 5-year intervals, C pools pre-disturbance differ from input carbon pools.
    # For annual intervals, C pools pre-disturbance are the same as input carbon pools because there is no gain before disturbance/fire.
    if interval_length == 5:
        agc_pre_disturb = agc_dens_in - agc_gross_removals_out
        bgc_pre_disturb = bgc_dens_in - bgc_gross_removals_out
        deadwood_c_pre_disturb = deadwood_c_dens_in - deadwood_c_gross_removals_out
        litter_c_pre_disturb = litter_c_dens_in - litter_c_gross_removals_out

        # Pre-disturbance carbon densities as an array, used as input for non-CO2 fire emissions and post-disturbance removals (if applicable)
        c_pre_disturb = np.array([agc_pre_disturb, bgc_pre_disturb, deadwood_c_pre_disturb, litter_c_pre_disturb]).astype('float32')

    # Assigning interval start C pools to pre-disturbance C pools rather than calculating them like in the 5-year interval
    # branch reduces the number of calculations and is more explicit
    elif interval_length == 1:
        agc_pre_disturb = agc_dens_in
        bgc_pre_disturb = bgc_dens_in
        deadwood_c_pre_disturb = deadwood_c_dens_in
        litter_c_pre_disturb = litter_c_dens_in
        c_pre_disturb = np.array(c_dens_in).astype('float32')

    else:
        raise ValueError("interval_length not valid: must be 1 or 5")


    # Step 5: Calculates CO2 gross emissions from fire by carbon pools (Mg C/ha/interval).  Which ones are emitted depends on whether fire was detected.

    # Calculates CO2 emissions from fire for each C pool using fire emission factors
    # if a Gef for CO2 is supplied AND if there was fire during the interval.
    # This is used for non-stand replacing forest disturbances (as opposed to entire C pools being combusted).
    # From IPCC 2019 Eqn. 2.27
    if first_year_burned_during_interval > 0:

        # Equations divide by C_to_CO2 to put the emissions back in Mg C/ha. They are later converted back to Mg CO2/ha,
        # but we need CO2 fire emissions in Mg C/ha here for consistency with all other outputs.
        agc_gross_emis_out = ((agc_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf_forest * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * agc_ef_CO2
        bgc_gross_emis_out = ((bgc_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf_forest * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * bgc_ef_CO2
        deadwood_c_gross_emis_out = ((deadwood_c_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf_forest * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * deadwood_c_ef_CO2
        litter_c_gross_emis_out = ((litter_c_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf_forest * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * litter_c_ef_CO2

        # Emission factor for burned forest is the combustion factor for forrest
        agc_ef_CO2 = Cf_forest

        # # For testing CO2 fire emissions
        # print("c_dens_in:", c_dens_in)
        # print("agc_rf:", agc_rf)
        # print("gain_year_count_pre_dist:", gain_year_count_pre_dist)
        # print("agc_pre_disturb:", agc_pre_disturb)
        # print(f"Cf_forest: {Cf_forest}; Gef_co2: {Gef_co2}")
        # print("AGC emission factor for fire:", agc_ef_CO2)
        # print("agc_gross_emis_out:", agc_gross_emis_out)
        # os.quit()

    # No emissions or emission factor if no fire detected
    else:

        agc_gross_emis_out = 0
        bgc_gross_emis_out = 0
        deadwood_c_gross_emis_out = 0
        litter_c_gross_emis_out = 0
        agc_ef_CO2 = 0

    # Gross emissions as an array
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')


    # Step 6: Updates gross removals to include post-fire gross removals, if applicable (Mg C/ha/interval).
    # gain year count is the number of years between the disturbance and the end of the interval.
    # This uses the same RFs before and after the fire.
    # Only applies to 5-year interval data.
    if (first_year_burned_during_interval > 0) and (interval_length == 5):

        post_dist_RF = np.array([RF_AGC, RF_BGC, RF_AGC * deadwood_c_ratio, RF_AGC * litter_c_ratio]).astype('float32')
        gain_year_count_post_dist = cn.five_year_interval_duration - gain_year_count_pre_dist - 1
        post_dist_gross_removals = gain_year_count_post_dist * post_dist_RF

        c_gross_removals_out = c_gross_removals_out - post_dist_gross_removals

        # print("post_dist_RF:", post_dist_RF)
        # print("gain_year_count_pre_dist:", gain_year_count_pre_dist)
        # print("gain_year_count_post_dist:", gain_year_count_post_dist)
        # print("post_dist_gross_removals:", post_dist_gross_removals)
        # print("c_gross_removals_out_after_dist:", c_gross_removals_out)
        # os.quit()


    # Step 7: Calculates ending carbon densities by carbon pool.
    # Starts with carbon density in (list converted to np array), adds gross removals (subtracts negative value), subtracts emissions.
    # Ending carbon pools are not affected by non-CO2 emissions in the next step.
    c_dens_out = np.array(c_dens_in).astype('float32') - c_gross_removals_out - c_gross_emissions_out


    # Step 8: Calculates non-CO2 emissions (if relevant) (Mg CO2e/ha/interval)
    # Default non-CO2 emissions values
    ch4_flux_out = 0
    n2o_flux_out = 0

    # Only assigns fire node code and calculates CH4 and N2O emissions if the pixel burned in the last interval
    if first_year_burned_during_interval > 0:

        state_out = accrete_node(node, cn.land_state_node_fire_value)

        # Selects just the carbon pools that have non-CO2 emissions from fire
        c_pools_for_fire_non_CO2 = np.where(c_pools_fire_non_CO2 == 1, c_pre_disturb, 0)

        # Sums the C pools that have non-CO2 fire emissions. We don't track which C pools the CH4 and N2O emissions come from,
        # so the pools are combined.
        c_pools_for_fire_total = np.sum(c_pools_for_fire_non_CO2)

        # Calculates non-CO2 fire emissions using the selected C pools in the year before disturbance
        ch4_flux_out, n2o_flux_out = non_CO2_fire_equations(c_pools_for_fire_total, Cf_forest, Gef_ch4, Gef_n2o)

        # # For testing non-CO2 emissions
        # print("c_dens_in:", c_dens_in)
        # print("c_pre_disturb:", c_pre_disturb)
        # print(f"Cf_forest: {Cf_forest}; Gef_ch4: {Gef_ch4}; GWP CH4: {cn.gwp_ch4}")
        # print(f"Cf_forest: {Cf_forest}; Gef_n2o: {Gef_n2o}; GWP N2O: {cn.gwp_n2o}")
        # print("c_pools_for_fire_non_CO2:", c_pools_for_fire_non_CO2)
        # print("c_pools_for_fire_total:", c_pools_for_fire_total)
        # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
        # os.quit()

    # Node code if no fire in the last interval. No CH4 and N2O emissions calculated.
    else:

        state_out = accrete_node(node, 2)

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')


    # Step 9: Updates the forest age. Increments by the number of years in the interval.
    # Age is not affected by fire, so age always increases in this function.
    if interval_length == 5:
        forest_age_interval_end = forest_age_interval_start + cn.five_year_interval_duration
    elif interval_length == 1:
        forest_age_interval_end = forest_age_interval_start + 1
    else:
        raise ValueError("interval_length not valid: must be 1 or 5")

    return state_out, c_gross_emissions_out, c_gross_removals_out, agc_ef_CO2, non_co2_fluxes_out, c_dens_out, gain_year_count_pre_dist, forest_age_interval_end


# Gross fluxes and ending carbon stocks for non-cropland (without tall vegetation) converted to cropland (without tall vegetation).
# Applies to 5-year intervals and annual intervals.
@jit(nopython=True)
def calc_NT_cropland_gain(c_pools_no_fire, c_dens_in, RF_array):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    # Carbon pools that are emitted as CO2 if fire was not detected.
    agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_no_fire)

    # Step 1: Calculates CO2 gross emissions by carbon pools (Mg C/ha/interval). Gross emissions are positive.
    # Which pools are emitted is controlled by the ef_CO2 flags.
    agc_gross_emis_out = agc_dens_in * agc_ef_CO2
    bgc_gross_emis_out = bgc_dens_in * bgc_ef_CO2
    deadwood_c_gross_emis_out = deadwood_c_dens_in * deadwood_c_ef_CO2
    litter_c_gross_emis_out = litter_c_dens_in * litter_c_ef_CO2

    # Consolidates outputs into array to reduce the number of arguments returned to the decision tree.
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')

    # Step 2: Calculates gross removals.
    # Gross removals is the annual crop AGC removals after residual carbon is lost (Mg C/ha/interval).
    # Gross removals is negative.
    c_gross_removals_out = -1 * RF_array

    # Step 3: Calculates ending carbon densities by carbon pool (Mg C/ha).
    # Starts with carbon density in (list converted to np array), subtracts emissions (positive value), subtracts cropland removals (negative value).
    # Ending carbon pools are not affected by non-CO2 emissions in the next step.
    c_dens_out = np.array(c_dens_in).astype('float32') - c_gross_emissions_out - c_gross_removals_out

    return c_gross_emissions_out, c_gross_removals_out, c_dens_out


# Gross fluxes and ending carbon stocks for cropland converted to non-cropland (without tall vegetation).
# Removals only if converted to short vegetation. Non-CO2 emissions only if fire.
@jit(nopython=True)
def calc_cropland_non_cropland(node, c_dens_in, c_pools_no_fire, times_burned_in_interval, RF_post_dist):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_no_fire)

    # Step 1: Calculates carbon gross emissions (AGC only) (Mg C/ha/interval)
    agc_gross_emis_out = agc_dens_in * agc_ef_CO2
    bgc_gross_emis_out = bgc_dens_in * bgc_ef_CO2
    deadwood_c_gross_emis_out = deadwood_c_dens_in * deadwood_c_ef_CO2
    litter_c_gross_emis_out = litter_c_dens_in * litter_c_ef_CO2
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')

    # Step 2: Calculates carbon gross removals (Mg C/ha/interval) (only if converted to short vegetation). Gross removals are negative.
    c_gross_removals_out = RF_post_dist * -1  # Would be short vegetation removals (only for AGC and BGC)

    # Step 3: Calculates ending carbon densities (Mg C/ha)
    c_dens_out = np.array(c_dens_in).astype('float32') - c_gross_removals_out - c_gross_emissions_out

    # Step 4: Calculates non-CO2 emissions (if relevant) (Mg CO2e/ha/interval)
    # Default non-CO2 emissions values
    ch4_flux_out = 0
    n2o_flux_out = 0

    # Only assigns fire node code and calculates CH4 and N2O emissions if the pixel burned in the last interval
    if times_burned_in_interval > 0:

        state_out = accrete_node(node, cn.land_state_node_fire_value)

        # Residue carbon in cropland is aboveground carbon only. Shouldn't be carbon in any other pools.
        residue_carbon = c_dens_in[0] * cn.cropland_residue_harvest_ratio

        # Calculates non-CO2 fire emissions using aboveground carbon (only cropland pool) for a single year of burning
        ch4_flux_out, n2o_flux_out = non_CO2_fire_equations(residue_carbon, cn.Cf_crop_residue,
                                                                     cn.Gef_CH4_crop_residue, cn.Gef_N2O_crop_residue)

        # Multiplies the per-burn emissions by the number of times burned to get total emissions during the interval
        ch4_flux_out = ch4_flux_out * times_burned_in_interval
        n2o_flux_out = n2o_flux_out * times_burned_in_interval

        # # For testing non-CO2 emissions
        # print("c_dens_in:", c_dens_in)
        # print("residue_carbon:", residue_carbon)
        # print("times_burned_in_interval:", np.float32(times_burned_in_interval))
        # print(f"Cf: {cn.Cf_crop_residue}; Gef_ch4: {cn.Gef_CH4_crop_residue}; GWP CH4: {cn.gwp_ch4}")
        # print(f"Cf: {cn.Cf_crop_residue}; Gef_n2o: {cn.Gef_N2O_crop_residue}; GWP N2O: {cn.gwp_n2o}")
        # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
        # sys.quit()

    # Node code if no fire in the last interval. No CH4 and N2O emissions calculated.
    else:

        state_out = accrete_node(node, 2)

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')

    return state_out, c_gross_emissions_out, c_gross_removals_out, c_dens_out, non_co2_fluxes_out


# Gross fluxes and ending carbon stocks for cropland remaining cropland (without tall vegetation).
# Carbon densities don't change.
# No CO2 emissions or removals but there are non-CO2 emissions if there is fire (crop residue burning).
@jit(nopython=True)
def calc_cropland_cropland(node, c_dens_in, times_burned_in_interval):

    # Step 1: Calculates carbon densities, carbon gross emissions and carbon gross removals (no changes to any)
    c_dens_out = np.array(c_dens_in).astype('float32')
    c_gross_emissions_out = np.array([0, 0, 0, 0]).astype('float32')  # Specified for completeness
    c_gross_removals_out = np.array([0, 0, 0, 0]).astype('float32')  # Specified for completeness


    # Step 2: Calculates non-CO2 emissions (if relevant) (Mg CO2e/ha/interval)
    # Default non-CO2 emissions values
    ch4_flux_out = 0
    n2o_flux_out = 0

    # Only assigns fire node code and calculates CH4 and N2O emissions if the pixel burned in the last interval
    if times_burned_in_interval > 0:

        state_out = accrete_node(node, cn.land_state_node_fire_value)

        # Residue carbon in cropland is aboveground carbon only. Shouldn't be carbon in any other pools.
        residue_carbon = c_dens_in[0] * cn.cropland_residue_harvest_ratio

        # Calculates non-CO2 fire emissions using aboveground carbon (only cropland pool) for a single year of burning
        ch4_flux_out, n2o_flux_out = non_CO2_fire_equations(residue_carbon, cn.Cf_crop_residue,
                                                                     cn.Gef_CH4_crop_residue, cn.Gef_N2O_crop_residue)

        # Multiplies the per-burn emissions by the number of times burned to get total emissions during the interval
        ch4_flux_out = ch4_flux_out * times_burned_in_interval
        n2o_flux_out = n2o_flux_out * times_burned_in_interval

        # # For testing non-CO2 emissions
        # if times_burned_in_interval > 1:
        #     print("c_dens_in:", c_dens_in)
        #     print("times_burned_in_interval:", np.float32(times_burned_in_interval))
        #     print(f"Cf: {cn.Cf_crop_residue}; Gef_ch4: {cn.Gef_CH4_crop_residue}; GWP CH4: {cn.gwp_ch4}")
        #     print(f"Cf: {cn.Cf_crop_residue}; Gef_n2o: {cn.Gef_N2O_crop_residue}; GWP N2O: {cn.gwp_n2o}")
        #     print(f"ch4_flux_out_single: {ch4_flux_out_single}; n2o_flux_out_single: {n2o_flux_out_single};")
        #     print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
        #     # sys.quit()


    # Node code if no fire in the last interval. No CH4 and N2O emissions calculated.
    else:

        state_out = accrete_node(node, 2)

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')

    return state_out, c_gross_emissions_out, c_gross_removals_out, c_dens_out, non_co2_fluxes_out


# Gross fluxes and ending carbon stocks for non-short vegetation/non-forest/non-cropland converted to short vegetation.
# Applies to 5-year intervals and annual intervals.
@jit(nopython=True)
def calc_short_veg_gain(rf):

    # C densities should already be 0 because the starting LC should have been forced to 0s, but this is safer.
    c_dens_in = [0, 0, 0, 0]

    # Step 1: Calculates carbon densities, carbon gross emissions and carbon gross removals (no changes to any)
    c_gross_emissions_out = np.array([0, 0, 0, 0]).astype('float32')  # Specified for completeness

    # Step 2: Calculates gross removals.
    # Gross removals is the annual crop AGC removals after residual carbon is lost (Mg C/ha/interval).
    # Gross removals is negative.
    c_gross_removals_out = -1 * rf

    # Step 3: Calculates ending carbon densities by carbon pool (Mg C/ha).
    # Starts with carbon density in (list converted to np array), subtracts emissions (positive value), subtracts cropland removals (negative value).
    # Ending carbon pools are not affected by non-CO2 emissions in the next step.
    c_dens_out = np.array(c_dens_in).astype('float32') - c_gross_removals_out

    return c_gross_emissions_out, c_gross_removals_out, c_dens_out


# Gross fluxes and ending carbon stocks for short vegetation converted to non-short vegetation, non-forest or non-cropland.
# No CO2 removals. CO2 emissions occur.
# There are non-CO2 emissions where there is fire (biomass burning).
@jit(nopython=True)
def calc_short_veg_loss(node, c_dens_in, c_pools_no_fire, times_burned_in_interval):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_no_fire)


    # Step 1: Calculates carbon densities (all pools 0 Mg C/ha), carbon gross emissions (AGC only) and carbon gross removals (none)
    agc_gross_emis_out = agc_dens_in * agc_ef_CO2
    bgc_gross_emis_out = bgc_dens_in * bgc_ef_CO2
    deadwood_c_gross_emis_out = deadwood_c_dens_in * deadwood_c_ef_CO2
    litter_c_gross_emis_out = litter_c_dens_in * litter_c_ef_CO2

    agc_dens_out = agc_dens_in - agc_gross_emis_out
    bgc_dens_out = bgc_dens_in - bgc_gross_emis_out
    deadwood_c_dens_out = deadwood_c_dens_in - deadwood_c_gross_emis_out
    litter_c_dens_out = litter_c_dens_in - litter_c_gross_emis_out

    c_dens_out = np.array([agc_dens_out, bgc_dens_out, deadwood_c_dens_out, litter_c_dens_out]).astype('float32')
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')
    c_gross_removals_out = np.array([0, 0, 0, 0]).astype('float32')  # Specified for completeness


    # Step 2: Calculates non-CO2 emissions (if relevant) (Mg CO2e/ha/interval)
    # Default non-CO2 emissions values
    ch4_flux_out = 0
    n2o_flux_out = 0

    # Only assigns fire node code and calculates CH4 and N2O emissions if the pixel burned in the last interval
    if times_burned_in_interval > 0:

        state_out = accrete_node(node, cn.land_state_node_fire_value)

        # Calculates non-CO2 fire emissions using aboveground carbon only for a single year of burning
        ch4_flux_out, n2o_flux_out = non_CO2_fire_equations(c_dens_in[0],
                                                            cn.Cf_grassland, cn.Gef_CH4_grassland, cn.Gef_N2O_grassland)

        # Multiplies the per-burn emissions by the number of times burned to get total emissions during the interval
        ch4_flux_out = ch4_flux_out * times_burned_in_interval
        n2o_flux_out = n2o_flux_out * times_burned_in_interval

        # # For testing non-CO2 emissions
        # print("c_dens_in:", c_dens_in)
        # print("times_burned_in_interval:", times_burned_in_interval)
        # print(f"Cf: {cn.Cf_crop_residue}; Gef_ch4: {cn.Gef_CH4_crop_residue}; GWP CH4: {cn.gwp_ch4}")
        # print(f"Cf: {cn.Cf_crop_residue}; Gef_n2o: {cn.Gef_N2O_crop_residue}; GWP N2O: {cn.gwp_n2o}")
        # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
        # sys.quit()

    # Node code if no fire in the last interval. No CH4 and N2O emissions calculated.
    else:

        state_out = accrete_node(node, 2)

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')

    return state_out, c_gross_emissions_out, c_gross_removals_out, c_dens_out, non_co2_fluxes_out


# Gross fluxes and ending carbon stocks for short vegetation remaining short vegetation.
# Carbon densities don't change.
# No CO2 emissions or removals but there are non-CO2 emissions if there is fire (biomass burning).
@jit(nopython=True)
def calc_short_veg_short_veg(node, c_dens_in, times_burned_in_interval):

    # Step 1: Calculates carbon densities, carbon gross emissions and carbon gross removals (no changes to any)
    c_dens_out = np.array(c_dens_in).astype('float32')
    c_gross_emissions_out = np.array([0, 0, 0, 0]).astype('float32')  # Specified for completeness
    c_gross_removals_out = np.array([0, 0, 0, 0]).astype('float32')  # Specified for completeness


    # Step 2: Calculates non-CO2 emissions (if relevant) (Mg CO2e/ha/interval)
    # Default non-CO2 emissions values
    ch4_flux_out = 0
    n2o_flux_out = 0

    # Only assigns fire node code and calculates CH4 and N2O emissions if the pixel burned in the last interval
    if times_burned_in_interval > 0:

        state_out = accrete_node(node, cn.land_state_node_fire_value)

        # Calculates non-CO2 fire emissions using aboveground carbon only for a single year of burning
        ch4_flux_out, n2o_flux_out = non_CO2_fire_equations(c_dens_in[0],
                                                            cn.Cf_grassland, cn.Gef_CH4_grassland, cn.Gef_N2O_grassland)

        # Multiplies the per-burn emissions by the number of times burned to get total emissions during the interval
        ch4_flux_out = ch4_flux_out * times_burned_in_interval
        n2o_flux_out = n2o_flux_out * times_burned_in_interval

        # # For testing non-CO2 emissions
        # print("c_dens_in:", c_dens_in)
        # print("times_burned_in_interval:", times_burned_in_interval)
        # print(f"Cf: {cn.Cf_crop_residue}; Gef_ch4: {cn.Gef_CH4_crop_residue}; GWP CH4: {cn.gwp_ch4}")
        # print(f"Cf: {cn.Cf_crop_residue}; Gef_n2o: {cn.Gef_N2O_crop_residue}; GWP N2O: {cn.gwp_n2o}")
        # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
        # sys.quit()

    # Node code if no fire in the last interval. No CH4 and N2O emissions calculated.
    else:

        state_out = accrete_node(node, 2)

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')

    return state_out, c_gross_emissions_out, c_gross_removals_out, c_dens_out, non_co2_fluxes_out