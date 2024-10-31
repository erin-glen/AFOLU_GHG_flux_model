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
def calculate_emissions_co2e(ef_co2, ef_n2o, ef_ch4_land, ef_ch4_ditch, ef_co2_offsite, area, gwp_ch4, gwp_n2o):
    """
    Calculates emissions in CO₂ equivalents based on emission factors and area.

    Args:
        ef_co2 (float32): Emission factor for CO₂ (tonnes CO₂/ha/year).
        ef_n2o (float32): Emission factor for N₂O (kg N₂O/ha/year).
        ef_ch4_land (float32): Emission factor for CH₄ from land (kg CH₄/ha/year).
        ef_ch4_ditch (float32): Emission factor for CH₄ from ditches (kg CH₄/ha/year).
        ef_co2_offsite (float32): Emission factor for offsite CO₂ emissions (tonnes CO₂/ha/year).
        area (float32): Area in hectares.
        gwp_ch4 (float32): Global Warming Potential for CH₄.
        gwp_n2o (float32): Global Warming Potential for N₂O.

    Returns:
        co2_emissions (float32): CO₂ emissions (tonnes CO₂/year).
        n2o_emissions_co2e (float32): N₂O emissions in tonnes CO₂e/year.
        ch4_land_emissions_co2e (float32): CH₄ emissions from land in tonnes CO₂e/year.
        ch4_ditch_emissions_co2e (float32): CH₄ emissions from ditches in tonnes CO₂e/year.
        co2_offsite_emissions (float32): Offsite CO₂ emissions (tonnes CO₂/year).
    """
    # CO₂ emissions
    co2_emissions = ef_co2 * area

    # Offsite CO₂ emissions
    co2_offsite_emissions = ef_co2_offsite * area

    # N₂O emissions in tonnes CO₂e
    n2o_emissions_co2e = (ef_n2o * area * gwp_n2o) / 1000.0  # Convert kg to tonnes

    # CH₄ emissions from land in tonnes CO₂e
    ch4_land_emissions_co2e = (ef_ch4_land * area * gwp_ch4) / 1000.0  # Convert kg to tonnes

    # CH₄ emissions from ditches in tonnes CO₂e
    ch4_ditch_emissions_co2e = (ef_ch4_ditch * area * gwp_ch4) / 1000.0  # Convert kg to tonnes

    return (co2_emissions, n2o_emissions_co2e,
            ch4_land_emissions_co2e, ch4_ditch_emissions_co2e,
            co2_offsite_emissions)
