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
def calculate_drainage_emissions_co2e(
    ef_co2, ef_n2o, ef_ch4_land, ef_ch4_ditch, ef_co2_offsite, frac_ditch,
    c_to_co2, n2o_n_to_n2o, gwp_n2o, gwp_ch4
):
    """
    Calculates drainage EF partials (CO2, N2O, CH4, offsite CO2) and
    returns a final drainage_total_co2e in tonnes CO2-equivalent.

    Args:
        ef_co2 (float32): Baseline CO2 (tonnes C/ha/yr) ...
        ef_n2o (float32): N2O (kg N/ha/yr) ...
        ef_ch4_land (float32): CH4 from land (kg CH4/ha/yr) ...
        ef_ch4_ditch (float32): CH4 from ditch (kg CH4/ha/yr) ...
        ef_co2_offsite (float32): Offsite CO2 (tonnes C/ha/yr)
        frac_ditch (float32): Fraction of area that is ditch ...
        c_to_co2, n2o_n_to_n2o, gwp_n2o, gwp_ch4: conversion constants

    Returns:
        co2_emissions (float32): drainage CO2 in tonnes CO2/ha/yr
        n2o_emissions_co2e (float32): drainage N2O in tonnes CO2e/ha/yr
        ch4_land_emissions_co2e (float32): drainage CH4 (land) in tonnes CO2e/ha/yr
        ch4_ditch_emissions_co2e (float32): drainage CH4 (ditch) in tonnes CO2e/ha/yr
        co2_offsite_emissions (float32): drainage offsite CO2 in tonnes CO2/ha/yr
        drainage_total_co2e (float32): sum of all the above, in tonnes CO2e/ha/yr
    """
    # Convert baseline CO2 from tonne C -> tonne CO2
    co2_emissions = ef_co2 * c_to_co2  # e.g. 1 tonne C => 3.67 tonne CO2
    co2_offsite_emissions = ef_co2_offsite * c_to_co2

    # N2O in kg N/ha => convert to kg N2O => convert to tonne => multiply GWP
    n2o_emissions_co2e = (ef_n2o * n2o_n_to_n2o * gwp_n2o) / 1000.0  # e.g. 265 GWP

    # CH4 from land or ditch in kg CH4 => convert to tonne => multiply GWP (28)
    ch4_land_emissions_co2e = (ef_ch4_land / 1000.0) * gwp_ch4
    ch4_ditch_emissions_co2e = (ef_ch4_ditch / 1000.0) * gwp_ch4 * frac_ditch

    # Combine partials into a final drainage total
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
        drainage_total_co2e
    )

@jit(nopython=True)
def calculate_burned_area_emissions(pixel_area_ha,
                                    mass_burnt,
                                    combustion_factor,
                                    gef_co2,
                                    gef_co,
                                    gef_ch4,
                                    gwp_co,
                                    gwp_ch4):
    """
    Calculates burned-area emissions (CO₂, CO, CH₄) in tonnes per pixel and total CO₂e.

    Formula:
        L_fire = pixel_area_ha * mass_burnt * combustion_factor * G_ef * 1e-3

    Args:
        pixel_area_ha (float32): Area of the pixel (ha), typically 1 ha.
        mass_burnt (float32): Mass of fuel burnt per hectare (t DM/ha).
        combustion_factor (float32): Fraction of biomass combusted (unitless).
        gef_co2 (float32): Emission factor for CO₂ (g/kg DM).
        gef_co (float32): Emission factor for CO (g/kg DM).
        gef_ch4 (float32): Emission factor for CH₄ (g/kg DM).
        gwp_co (float32): Global Warming Potential for CO.
        gwp_ch4 (float32): Global Warming Potential for CH₄.

    Returns:
        burn_co2 (float32): CO₂ emissions (tonnes).
        burn_co (float32): CO emissions (tonnes).
        burn_ch4 (float32): CH₄ emissions (tonnes).
        total_burned_emissions_co2e (float32): Total burned-area emissions in CO₂e.
    """
    burn_co2 = pixel_area_ha * mass_burnt * combustion_factor * gef_co2 * 1e-3
    burn_co  = pixel_area_ha * mass_burnt * combustion_factor * gef_co  * 1e-3 * gwp_co
    burn_ch4 = pixel_area_ha * mass_burnt * combustion_factor * gef_ch4 * 1e-3 * gwp_ch4

    total_burned_emissions_co2e = burn_co2 + (burn_co * gwp_co) + (burn_ch4 * gwp_ch4)

    return burn_co2, burn_co, burn_ch4, total_burned_emissions_co2e

