import numpy as np
from numba import jit
from numba.typed import Dict
from numba.core import types

# Project imports
from . import constants_and_names as cn

@jit(nopython=True)
def accrete_node(combo, new):
    """
    Combine two integer codes, preserving digit order.
    Example: node=1, new=3 => 13
    """
    combo = combo*10 + new
    return combo


@jit(nopython=True)
def join_codes(base, extra):
    """Append all digits of ``extra`` to ``base``."""
    factor = 1
    temp = extra
    while temp > 0:
        factor *= 10
        temp //= 10
    return base * factor + extra


def create_typed_dicts(layers):
    """
    Distribute arrays into typed dictionaries by dtype, so that we can pass them
    to Numba in dictionary form.
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
            # or raise TypeError(f"{key} dtype not recognized")

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
def calculate_drainage_emissions_co2e(
        ef_co2, ef_n2o, ef_ch4_land, ef_ch4_ditch,
        ef_co2_offsite, frac_ditch,
        c_to_co2, n2o_n_to_n2o, gwp_n2o, gwp_ch4
):
    """
    Return drainage emission factors per hectare.

    Args:
      ef_co2 (float32): baseline CO2 in tonne C/ha/yr
      ef_n2o (float32): N2O in kg N/ha/yr
      ef_ch4_land (float32): land CH4 in kg CH4/ha/yr
      ef_ch4_ditch (float32): ditch CH4 in kg CH4/ha/yr
      ef_co2_offsite (float32): offsite CO2 in tonne C/ha/yr
      frac_ditch (float32): fraction for ditch area

    Returns drainage partial EFs plus total, all in **tonnes CO2e per ha**.
    """
    # 1) Convert from tonne C to tonne CO2
    co2_emissions = ef_co2 * c_to_co2
    co2_offsite_emissions = ef_co2_offsite * c_to_co2

    # 2) N2O in kg => tonne => multiply GWP
    n2o_emissions_co2e = (ef_n2o * n2o_n_to_n2o * gwp_n2o) / 1000.0

    # 3) CH4 in kg => tonne => multiply GWP
    ch4_land_emissions_co2e = (ef_ch4_land / 1000.0) * gwp_ch4
    ch4_ditch_emissions_co2e = (ef_ch4_ditch / 1000.0) * gwp_ch4 * frac_ditch

    # 4) Sum into a drainage total (still per ha at this point)
    drainage_total_co2e = (
            co2_emissions
            + n2o_emissions_co2e
            + ch4_land_emissions_co2e
            + ch4_ditch_emissions_co2e
            + co2_offsite_emissions
    )

    return (
        co2_emissions,
        n2o_emissions_co2e,
        ch4_land_emissions_co2e,
        ch4_ditch_emissions_co2e,
        co2_offsite_emissions,
        drainage_total_co2e,
    )


@jit(nopython=True)
def calculate_burned_area_emissions(
        mass_burnt,
        combustion_factor,
        gef_co2,
        gef_co,
        gef_ch4,
        gwp_co,
        gwp_ch4
):
    """
    Return burned emissions per hectare.
    """
    burn_co2 = mass_burnt * combustion_factor * gef_co2 * 1e-3
    burn_co = mass_burnt * combustion_factor * gef_co * 1e-3 * gwp_co
    burn_ch4 = mass_burnt * combustion_factor * gef_ch4 * 1e-3 * gwp_ch4

    total_burned_emissions_co2e = burn_co2 + burn_co + burn_ch4
    return burn_co2, burn_co, burn_ch4, total_burned_emissions_co2e

