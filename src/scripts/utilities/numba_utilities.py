import numpy as np
from numba import jit
from numba.typed import Dict
from numba.core import types

# Project imports
from . import constants_and_names as cn

@jit(nopython=True)
def accrete_node(combo, new):
    combo = combo*10 + new
    return combo


def create_typed_dicts(layers):
    """
    Existing code unchanged – distributing arrays into typed dictionaries.
    """
    uint8_dict_layers = {}
    int16_dict_layers = {}
    int32_dict_layers = {}
    float32_dict_layers = {}

    for key, array in layers.items():
        if array is None:
            continue

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

    typed_dict_uint8 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.uint8, 2, 'C')
    )
    typed_dict_int16 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.int16, 2, 'C')
    )
    typed_dict_int32 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.int32, 2, 'C')
    )
    typed_dict_float32 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.float32, 2, 'C')
    )

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
    Existing drainage-based logic, unchanged.
    Returns drainage emissions in CO2e.
    """
    co2_emissions = ef_co2 * c_to_co2
    co2_offsite_emissions = ef_co2_offsite * c_to_co2

    n2o_emissions_co2e = (ef_n2o * n2o_n_to_n2o * gwp_n2o) / 1000.0
    ch4_land_emissions_co2e = (ef_ch4_land / 1000.0) * gwp_ch4
    ch4_ditch_emissions_co2e = (ef_ch4_ditch / 1000.0) * gwp_ch4 * frac_ditch

    return (
        co2_emissions,
        n2o_emissions_co2e,
        ch4_land_emissions_co2e,
        ch4_ditch_emissions_co2e,
        co2_offsite_emissions
    )

@jit(nopython=True)
def calculate_fire_emissions(mass_burnt, combustion_factor, gef_co2, gef_co, gef_ch4):
    """
    Calculates burned-area emissions in tonnes/pixel,
    assuming each pixel is effectively 1 "unit area."

    L_fire = M_B * C_f * G_ef * 10^-3
      where M_B is in t DM,
            G_ef is in g gas / kg DM,
            10^-3 converts g->kg or g->tonnes
               depending on how you set M_B.

    For now, we treat each pixel as 1 ha
    and skip an explicit area factor.
    """
    burn_co2 = mass_burnt * combustion_factor * gef_co2 * 1e-3
    burn_co  = mass_burnt * combustion_factor * gef_co  * 1e-3
    burn_ch4 = mass_burnt * combustion_factor * gef_ch4 * 1e-3
    return burn_co2, burn_co, burn_ch4

