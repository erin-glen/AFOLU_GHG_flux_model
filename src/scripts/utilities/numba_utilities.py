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


# Creates a separate dictionary for each chunk datatype so that they can be passed to Numba as separate arguments.
# Numba functions can accept (and return) dictionaries of arrays as long as each dictionary only has arrays of one data type (e.g., uint8, float32)
# Note: need to add new code if inputs with other data types are added
def create_typed_dicts(layers):
    # Initializes empty dictionaries for each type
    uint8_dict_layers = {}
    int16_dict_layers = {}
    int32_dict_layers = {}
    float32_dict_layers = {}

    # Iterates through the downloaded chunk dictionary and distributes arrays to a separate dictionary for each data type
    for key, array in layers.items():

        # Skips the dictionary entry if it has no data (generally because the chunk doesn't exist for that input)
        if array is None:
            continue

        # If there is data, it puts the data in the corresponding dictionary for that datatype
        if array.dtype == np.uint8:
            uint8_dict_layers[key] = array
        elif array.dtype == np.int16:
            int16_dict_layers[key] = array
        elif array.dtype == np.int32:
            int32_dict_layers[key] = array
        elif array.dtype == np.float32:
            float32_dict_layers[key] = array
        else:
            pass
            # raise TypeError(f"{key} dtype not in list")

    print(f"uint8 datasets: {uint8_dict_layers.keys()}")
    print(f"int16 datasets: {int16_dict_layers.keys()}")
    print(f"int32 datasets: {int32_dict_layers.keys()}")
    print(f"float32 datasets: {float32_dict_layers.keys()}")

    # Creates numba-compliant typed dict for each type of array
    typed_dict_uint8 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.uint8, 2, 'C')  # Assuming 2D arrays of uint8
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

    for key, array in int16_dict_layers.items():
        typed_dict_int16[key] = array

    for key, array in int32_dict_layers.items():
        typed_dict_int32[key] = array

    for key, array in float32_dict_layers.items():
        typed_dict_float32[key] = array

    return typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32

@jit(nopython=True)
def calculate_emissions_co2e(
    ef_co2, ef_n2o, ef_ch4_land, ef_ch4_ditch, ef_co2_offsite, frac_ditch,
    c_to_co2, n2o_n_to_n2o, gwp_n2o, gwp_ch4
):
    """
    Calculates emissions in CO₂ equivalents per hectare per year.

    Args:
        ef_co2 (float32): Emission factor for CO₂ (tonnes C/ha/year).
        ef_n2o_n (float32): Emission factor for N₂O-N (kg N₂O-N/ha/year).
        ef_ch4_land (float32): Emission factor for CH₄ from land (kg CH₄/ha/year).
        ef_ch4_ditch (float32): Emission factor for CH₄ from ditches (kg CH₄/ha/year).
        ef_co2_offsite (float32): Emission factor for offsite CO₂ emissions (tonnes C/ha/year).
        frac_ditch (float32): Fraction of the area that is covered by ditches.
        c_to_co2 (float32): Conversion factor from C to CO₂ (44/12 ≈ 3.67).
        n2o_n_to_n2o (float32): Conversion factor from N₂O-N to N₂O (44/28 ≈ 1.571).
        gwp_n2o (float32): Global Warming Potential for N₂O (e.g., 265).
        gwp_ch4 (float32): Global Warming Potential for CH₄ (e.g., 27).

    Returns:
        co2_emissions (float32): CO₂ emissions (tonnes CO₂/ha/year).
        n2o_emissions_co2e (float32): N₂O emissions (tonnes CO₂e/ha/year).
        ch4_land_emissions_co2e (float32): CH₄ emissions from land (tonnes CO₂e/ha/year).
        ch4_ditch_emissions_co2e (float32): CH₄ emissions from ditches (tonnes CO₂e/ha/year).
        co2_offsite_emissions (float32): Offsite CO₂ emissions (tonnes CO₂/ha/year).
    """
    # CO₂ emissions (convert from t C to t CO₂)
    co2_emissions = ef_co2 * c_to_co2

    # Offsite CO₂ emissions (convert from t C to t CO₂)
    co2_offsite_emissions = ef_co2_offsite * c_to_co2

    # N₂O emissions in tonnes CO₂e/ha/year
    n2o_emissions_co2e = (ef_n2o * n2o_n_to_n2o * gwp_n2o) / 1000.0  # Convert kg to tonnes

    # CH₄ emissions from land in tonnes CO₂e/ha/year
    ch4_land_emissions_co2e = (ef_ch4_land / 1000.0) * gwp_ch4  # Convert kg to tonnes, then multiply by GWP

    # CH₄ emissions from ditches in tonnes CO₂e/ha/year (adjusted by frac_ditch)
    ch4_ditch_emissions_co2e = (ef_ch4_ditch / 1000.0) * gwp_ch4 * frac_ditch  # Multiply by frac_ditch

    return (
        co2_emissions,
        n2o_emissions_co2e,
        ch4_land_emissions_co2e,
        ch4_ditch_emissions_co2e,
        co2_offsite_emissions
    )


