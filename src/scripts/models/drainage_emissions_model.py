# drainage_emissions_model.py

import argparse
import concurrent.futures
import numpy as np
import gc
import os
import sys
from datetime import datetime

from dask.distributed import print
from numba import jit, types
from numba.typed import Dict

# Project-specific imports (ensure these modules are available)
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu
from src.scripts.utilities import numba_utilities as nu

# Import constants for gas conversions
c_to_co2 = np.float32(cn.c_to_co2)
n2o_n_to_n2o = np.float32(cn.n2o_n_to_n2o)
gwp_ch4 = np.float32(cn.gwp_ch4)
gwp_n2o = np.float32(cn.gwp_n2o)
gwp_co = np.float32(cn.gwp_co)


# Define constants for land cover codes
forest_code = cn.ipcc_codes['forest']
cropland_code = cn.ipcc_codes['cropland']
settlement_code = cn.ipcc_codes['settlement']
wetland_code = cn.ipcc_codes['wetland']
grassland_code = cn.ipcc_codes['grassland']
otherland_code = cn.ipcc_codes['otherland']

# Define constants for ecozone codes
boreal_code = cn.ecozone_codes['boreal']
temperate_code = cn.ecozone_codes['temperate']
tropical_code = cn.ecozone_codes['tropical']
unknown_ecozone_code = cn.ecozone_codes['unknown']

# Define constants for nutrient status codes
poor_nutrient_code = cn.nutrient_status_codes['poor']
rich_nutrient_code = cn.nutrient_status_codes['rich']
unknown_nutrient_code = cn.nutrient_status_codes['unknown']

# Define constants for plantation type codes
long_rotation_code = cn.plantation_type_codes['long_rotation']
short_rotation_code = cn.plantation_type_codes['short_rotation']
oil_palm_code = cn.plantation_type_codes['oil_palm']
sago_palm_code = cn.plantation_type_codes['sago_palm']
unknown_plantation_code = cn.plantation_type_codes['unknown']


@jit(nopython=True)
def calculate_drainage_and_emissions(in_dict_uint8, in_dict_int16, in_dict_float32):
    """
    Calculates drainage status and emissions (CO₂, N₂O, CH₄) for each pixel, plus
    additional burned-area emissions (CO₂, CO, CH₄) if a burned-area layer is provided.

    Args:
        in_dict_uint8 (Dict): Dictionary of uint8 input arrays, possibly including burned-area blocks.
        in_dict_int16 (Dict): Dictionary of int16 input arrays.
        in_dict_float32 (Dict): Dictionary of float32 input arrays.

    Returns:
        out_dict_uint32 (Dict): Dictionary of uint32 output arrays.
        out_dict_float32 (Dict): Dictionary of float32 output arrays.
    """
    # Initialize output typed dictionaries
    out_dict_uint32 = Dict.empty(key_type=types.unicode_type, value_type=types.uint32[:, :])
    out_dict_float32 = Dict.empty(key_type=types.unicode_type, value_type=types.float32[:, :])

    # Extract required input arrays
    peat_block = in_dict_uint8['peat']
    land_cover_block = in_dict_uint8['land_cover']
    planted_forest_type_block = in_dict_uint8['planted_forest_type']
    dadap_block = in_dict_float32['dadap']
    osm_roads_block = in_dict_float32['osm_roads']
    osm_canals_block = in_dict_float32['osm_canals']
    engert_block = in_dict_float32['engert']
    grip_block = in_dict_float32['grip']
    extraction_block = in_dict_uint8['extraction']
    ecozone_block = in_dict_int16['climate_domain']
    nutrient_block = in_dict_uint8['nutrient_status']
    descals_type_block = in_dict_int16['descals_type']

    # OPTIONAL: See if a burned-area layer is present (e.g. "burned_area_final_YYYY").
    # For simplicity, we assume only one burned block is used (single-year).
    burned_block = None
    for k in in_dict_uint8.keys():
        if "burned_area_final_" in k:
            burned_block = in_dict_uint8[k]
            break

    # We treat each pixel as 1 ha (the same as drainage)
    pixel_area_ha = np.float32(1.0)

    # Prepare output arrays
    rows, cols = peat_block.shape

    # 1) Drainage status & state
    soil_block = np.zeros((rows, cols), dtype=np.uint32)
    state_out_block = np.zeros((rows, cols), dtype=np.uint32)

    # 2) Drainage emission arrays
    co2_emissions_out = np.zeros((rows, cols), dtype=np.float32)
    n2o_emissions_out = np.zeros((rows, cols), dtype=np.float32)
    ch4_land_emissions_out = np.zeros((rows, cols), dtype=np.float32)
    ch4_ditch_emissions_out = np.zeros((rows, cols), dtype=np.float32)
    co2_offsite_emissions_out = np.zeros((rows, cols), dtype=np.float32)

    # 3) Burned-area emission arrays (CO₂, CO, CH₄, CO₂e) in tonnes/pixel
    burned_co2_out = np.zeros((rows, cols), dtype=np.float32)
    burned_co_out = np.zeros((rows, cols), dtype=np.float32)
    burned_ch4_out = np.zeros((rows, cols), dtype=np.float32)
    burned_total_emissions_co2e_out = np.zeros((rows, cols), dtype=np.float32)

    # Loop over pixels
    for row in range(rows):
        for col in range(cols):
            peat = peat_block[row, col]
            land_cover = land_cover_block[row, col]
            planted_forest_type = planted_forest_type_block[row, col]
            dadap = dadap_block[row, col]
            osm_roads = osm_roads_block[row, col]
            osm_canals = osm_canals_block[row, col]
            engert = engert_block[row, col]
            grip = grip_block[row, col]
            extraction = extraction_block[row, col]
            ecozone = ecozone_block[row, col]
            nutrient = nutrient_block[row, col]
            descals_type = descals_type_block[row, col]

            # We'll store computed EFs (emission factors) for drainage
            ef_co2 = np.float32(0.0)
            ef_n2o = np.float32(0.0)
            ef_ch4_land = np.float32(0.0)
            ef_ch4_ditch = np.float32(0.0)
            ef_co2_offsite = np.float32(0.0)
            frac_ditch = np.float32(0.0)

            node = 0  # Node for logic tracing
            drained = False

            # Decide if peat is drained
            if peat == 1:
                node = nu.accrete_node(node, 1)
                if dadap > 0 or osm_canals > 0:
                    node = nu.accrete_node(node, 1)
                    drained = True
                elif engert > 0 or grip > 0 or osm_roads > 0:
                    node = nu.accrete_node(node, 2)
                    drained = True
                elif land_cover in (cropland_code, settlement_code):
                    node = nu.accrete_node(node, 3)
                    drained = True
                elif planted_forest_type > 0 or descals_type_block[row, col] > 0:
                    node = nu.accrete_node(node, 4)
                    drained = True
                elif extraction > 0:
                    node = nu.accrete_node(node, 5)
                    drained = True
                else:
                    node = nu.accrete_node(node, 6)

                if drained:
                    soil_block[row, col] = 2  # drained peat
                else:
                    soil_block[row, col] = 1  # undrained peat
            else:
                node = nu.accrete_node(node, 2)
                soil_block[row, col] = 0  # not peat

            state_out_block[row, col] = node

            # If drained peat, fill drainage EFs
            if soil_block[row, col] == 2:
                node = nu.accrete_node(node, 1)

                # Offsite defaults for each ecozone
                if ecozone == boreal_code:
                    node = nu.accrete_node(node, 1)
                    ef_co2_offsite = 0.12
                    if land_cover == forest_code:
                        node = nu.accrete_node(node, 1)
                        if nutrient == poor_nutrient_code:
                            node = nu.accrete_node(node, 1)
                            ef_co2 = 0.25
                            ef_n2o = 0.22
                            ef_ch4_land = 7.0
                            ef_ch4_ditch = 217.0
                            frac_ditch = 0.025
                        elif nutrient == rich_nutrient_code:
                            node = nu.accrete_node(node, 2)
                            ef_co2 = 0.95
                            ef_n2o = 3.2
                            ef_ch4_land = 2.0
                            ef_ch4_ditch = 217.0
                            frac_ditch = 0.025
                        else:
                            node = nu.accrete_node(node, 3)
                            ef_co2 = 0.0
                            ef_n2o = 0.0
                            ef_ch4_land = 0.0
                            ef_ch4_ditch = 0.0
                            frac_ditch = 0
                    elif land_cover == grassland_code:
                        node = nu.accrete_node(node, 2)
                        ef_co2 = 5.7
                        ef_n2o = 9.5
                        ef_ch4_land = 1.4
                        ef_ch4_ditch = 1165.0
                        frac_ditch = 0.05
                    elif land_cover == cropland_code:
                        node = nu.accrete_node(node, 3)
                        ef_co2 = 7.9
                        ef_n2o = 13.0
                        ef_ch4_land = 0.0
                        ef_ch4_ditch = 1165.0
                        frac_ditch = 0.05
                    elif extraction > 0:
                        node = nu.accrete_node(node, 4)
                        ef_co2 = 2.8
                        ef_n2o = 0.30
                        ef_ch4_land = 6.1
                        ef_ch4_ditch = 542.0
                        frac_ditch = 0.05
                    else:
                        node = nu.accrete_node(node, 5)
                        ef_co2 = 0.0
                        ef_n2o = 0.0
                        ef_ch4_land = 0.0
                        ef_ch4_ditch = 0.0
                        frac_ditch = 0

                elif ecozone == temperate_code:
                    node = nu.accrete_node(node, 2)
                    ef_co2_offsite = 0.31
                    if land_cover == forest_code:
                        node = nu.accrete_node(node, 1)
                        ef_co2 = 2.6
                        ef_n2o = 2.8
                        ef_ch4_land = 2.5
                        ef_ch4_ditch = 217.0
                        frac_ditch = 0.05
                    elif land_cover == grassland_code:
                        node = nu.accrete_node(node, 2)
                        ef_ch4_ditch = 1165.0
                        if nutrient == poor_nutrient_code:
                            node = nu.accrete_node(node, 1)
                            ef_co2 = 5.3
                            ef_n2o = 4.3
                            ef_ch4_land = 1.8
                            frac_ditch = 0.05
                        elif nutrient == rich_nutrient_code:
                            node = nu.accrete_node(node, 2)
                            ef_co2 = 6.1
                            ef_n2o = 8.2
                            ef_ch4_land = 16.0
                            frac_ditch = 0.05
                        else:
                            node = nu.accrete_node(node, 3)
                            ef_co2 = 0.0
                            ef_n2o = 0.0
                            ef_ch4_land = 0.0
                            frac_ditch = 0.05
                    elif land_cover == cropland_code:
                        node = nu.accrete_node(node, 3)
                        ef_co2 = 10.5
                        ef_n2o = 13.0
                        ef_ch4_land = 0.0
                        ef_ch4_ditch = 1165.0
                        frac_ditch = 0.05
                    elif extraction > 0:
                        node = nu.accrete_node(node, 4)
                        ef_co2 = 3.0
                        ef_n2o = 0.3
                        ef_ch4_land = 6.1
                        ef_ch4_ditch = 542.0
                        frac_ditch = 0.05
                    else:
                        node = nu.accrete_node(node, 5)
                        ef_co2 = 0.0
                        ef_n2o = 0.0
                        ef_ch4_land = 0.0
                        ef_ch4_ditch = 0.0
                        frac_ditch = 0

                elif ecozone == tropical_code:
                    node = nu.accrete_node(node, 3)
                    ef_co2_offsite = 0.82
                    ef_ch4_ditch = 2259.0
                    frac_ditch = 0.02
                    if planted_forest_type > 0:
                        node = nu.accrete_node(node, 1)
                        if planted_forest_type == long_rotation_code:
                            node = nu.accrete_node(node, 1)
                            ef_co2 = 15.0
                            ef_n2o = 2.4
                            ef_ch4_land = 2.7
                        elif planted_forest_type == short_rotation_code:
                            node = nu.accrete_node(node, 2)
                            ef_co2 = 20.0
                            ef_n2o = 2.4
                            ef_ch4_land = 2.7
                        elif planted_forest_type == oil_palm_code:
                            node = nu.accrete_node(node, 3)
                            ef_co2 = 11.0
                            ef_n2o = 1.2
                            ef_ch4_land = 0.0
                        elif planted_forest_type == sago_palm_code:
                            node = nu.accrete_node(node, 4)
                            ef_co2 = 1.5
                            ef_n2o = 3.3
                            ef_ch4_land = 26.2
                        else:
                            node = nu.accrete_node(node, 5)
                            ef_co2 = 0.0
                            ef_n2o = 0.0
                            ef_ch4_land = 0.0
                            ef_ch4_ditch = 0.0
                    elif land_cover == forest_code:
                        node = nu.accrete_node(node, 2)
                        ef_co2 = 5.3
                        ef_n2o = 2.4
                        ef_ch4_land = 4.9
                    elif land_cover == grassland_code:
                        node = nu.accrete_node(node, 3)
                        ef_co2 = 9.6
                        ef_n2o = 5.0
                        ef_ch4_land = 7.0
                    elif land_cover == cropland_code:
                        node = nu.accrete_node(node, 4)
                        ef_co2 = 14.0
                        ef_n2o = 5.0
                        ef_ch4_land = 7.0
                    elif extraction > 0:
                        node = nu.accrete_node(node, 5)
                        ef_co2 = 2.0
                        ef_n2o = 0.0
                        ef_ch4_land = 0.0
                    else:
                        node = nu.accrete_node(node, 6)
                        ef_co2 = 0.0
                        ef_n2o = 0.0
                        ef_ch4_land = 0.0
                        ef_ch4_ditch = 0.0

                else:
                    node = nu.accrete_node(node, 4)
                    ef_co2 = 0.0
                    ef_n2o = 0.0
                    ef_ch4_land = 0.0
                    ef_ch4_ditch = 0.0
                    ef_co2_offsite = 0.0

                # Summation to CO2e
                (co2_emissions,
                 n2o_emissions_co2e,
                 ch4_land_emissions_co2e,
                 ch4_ditch_emissions_co2e,
                 co2_offsite_emissions
                 ) = nu.calculate_emissions_co2e(
                    ef_co2,
                    ef_n2o,
                    ef_ch4_land,
                    ef_ch4_ditch,
                    ef_co2_offsite,
                    frac_ditch,
                    c_to_co2,
                    n2o_n_to_n2o,
                    gwp_n2o,
                    gwp_ch4
                )

                co2_emissions_out[row, col] = co2_emissions
                n2o_emissions_out[row, col] = n2o_emissions_co2e
                ch4_land_emissions_out[row, col] = ch4_land_emissions_co2e
                ch4_ditch_emissions_out[row, col] = ch4_ditch_emissions_co2e
                co2_offsite_emissions_out[row, col] = co2_offsite_emissions

            else:
                # Not peat or not drained => no drainage emissions
                node = nu.accrete_node(node, 2)
                state_out_block[row, col] = node
                co2_emissions_out[row, col] = 0.0
                n2o_emissions_out[row, col] = 0.0
                ch4_land_emissions_out[row, col] = 0.0
                ch4_ditch_emissions_out[row, col] = 0.0
                co2_offsite_emissions_out[row, col] = 0.0

            # -----------------------------------------------------
            # Burned-area logic (IPCC Equation 2.8) for any pixel,
            # drained or undrained (just illustrate in "else" block).
            # If you want to handle drains as well, put this logic
            # outside the else so it applies to all.
            # -----------------------------------------------------
                # Fully disaggregated burned-area logic
                if burned_block is not None:
                    burned_val = burned_block[row, col]
                    if burned_val > 0 and soil_block[row, col] in (1, 2):
                        combustion_factor = np.float32(0.75)
                        mass_burnt = np.float32(50.0)

                        ecozone = ecozone_block[row, col]

                        if ecozone == boreal_code:
                            burned_node = nu.accrete_node(burned_node, 1)
                            if soil_block[row, col] == 2:  # Drained peat
                                burned_node = nu.accrete_node(burned_node, 1)
                                gef_co2, gef_co, gef_ch4 = 1650.0, 110.0, 12.0
                            elif soil_block[row, col] == 1:  # Undrained peat
                                burned_node = nu.accrete_node(burned_node, 2)
                                gef_co2, gef_co, gef_ch4 = 1450.0, 90.0, 10.0

                        elif ecozone == temperate_code:
                            burned_node = nu.accrete_node(burned_node, 2)
                            if soil_block[row, col] == 2:  # Drained peat
                                burned_node = nu.accrete_node(burned_node, 1)
                                gef_co2, gef_co, gef_ch4 = 1650.0, 110.0, 12.0
                            elif soil_block[row, col] == 1:  # Undrained peat
                                burned_node = nu.accrete_node(burned_node, 2)
                                gef_co2, gef_co, gef_ch4 = 1450.0, 90.0, 10.0

                        elif ecozone == tropical_code:
                            burned_node = nu.accrete_node(burned_node, 3)
                            if soil_block[row, col] == 2:  # Drained peat
                                burned_node = nu.accrete_node(burned_node, 1)
                                if land_cover == cropland_code:
                                    burned_node = nu.accrete_node(burned_node, 3)
                                    gef_co2, gef_co, gef_ch4 = 1700.0, 200.0, 15.0
                                elif planted_forest_type > 0:
                                    burned_node = nu.accrete_node(burned_node, 4)
                                    gef_co2, gef_co, gef_ch4 = 1700.0, 200.0, 15.0
                                else:
                                    burned_node = nu.accrete_node(burned_node, 5)
                                    gef_co2, gef_co, gef_ch4 = 1600.0, 180.0, 14.0
                            elif soil_block[row, col] == 1:  # Undrained tropical peat
                                # No burned-area emissions are calculated for undrained tropical peat, following specific guidelines.
                                gef_co2, gef_co, gef_ch4 = 0.0, 0.0, 0.0

                        else:
                            burned_node = nu.accrete_node(burned_node, 4)
                            gef_co2, gef_co, gef_ch4 = 0.0, 0.0, 0.0

                        burn_co2, burn_co, burn_ch4, burn_total_co2e = nu.calculate_burned_area_emissions(
                            pixel_area_ha,
                            mass_burnt,
                            combustion_factor,
                            gef_co2,
                            gef_co,
                            gef_ch4,
                            gwp_co,
                            gwp_ch4
                        )

                        burned_co2_out[row, col] = burn_co2
                        burned_co_out[row, col] = burn_co
                        burned_ch4_out[row, col] = burn_ch4
                        burned_total_emissions_co2e_out[row, col] = burn_total_co2e
                        state_out_block[row, col] = nu.accrete_node(state_out_block[row, col], burned_node)
                    else:
                        burned_co2_out[row, col] = 0.0
                        burned_co_out[row, col] = 0.0
                        burned_ch4_out[row, col] = 0.0
                        burned_total_emissions_co2e_out[row, col] = 0.0

    # Add drainage outputs to typed dict
    out_dict_uint32["soil"] = soil_block
    out_dict_uint32["state"] = state_out_block
    out_dict_float32["co2_emissions"] = co2_emissions_out
    out_dict_float32["n2o_emissions_co2e"] = n2o_emissions_out
    out_dict_float32["ch4_land_emissions_co2e"] = ch4_land_emissions_out
    out_dict_float32["ch4_ditch_emissions_co2e"] = ch4_ditch_emissions_out
    out_dict_float32["co2_offsite_emissions"] = co2_offsite_emissions_out

    # Total drainage-based emissions (CO₂e)
    total_emissions_out = (
            co2_emissions_out
            + co2_offsite_emissions_out
            + n2o_emissions_out
            + ch4_land_emissions_out
            + ch4_ditch_emissions_out
    )
    out_dict_float32["total_emissions"] = total_emissions_out

    # Add burned emissions (including CO₂e) to typed dict
    out_dict_float32["burned_co2"] = burned_co2_out
    out_dict_float32["burned_co"] = burned_co_out
    out_dict_float32["burned_ch4"] = burned_ch4_out
    out_dict_float32["burned_total_emissions_co2e"] = burned_total_emissions_co2e_out

    return out_dict_uint32, out_dict_float32

def calculate_and_upload_drainage(bounds,
                                  download_dict_with_data_types,
                                  is_final,
                                  no_upload,
                                  interval_start_year=None,
                                  interval_end_year=None):
    """
    Processes one chunk of drainage emissions, plus optional burned-area logic.
    """
    logger = lu.setup_logging_worker()
    bounds_str = uu.boundstr(bounds)
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)

    chunk_stats = []

    # Replace placeholder {tile_id} in S3 references
    updated_download_dict = uu.replace_tile_id_in_dict(download_dict_with_data_types, tile_id)

    # Quick check if tile exists
    tile_exists = uu.check_for_tile(updated_download_dict, is_final, logger)
    if not tile_exists:
        return (f"Skipped chunk {bounds_str} because {tile_id} does not exist for any inputs: {uu.timestr()}",
                chunk_stats)

    # If interval-based, add references to burned layers for each year
    if interval_start_year and interval_end_year:
        for yr in range(interval_start_year, interval_end_year + 1):
            burned_key = f"{cn.burned_area_final_pattern}_{yr}"
            if burned_key not in updated_download_dict:
                updated_download_dict[burned_key] = None

    # Download chunk layers
    futures = uu.prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_final, logger)
    layers = {}
    for fut in concurrent.futures.as_completed(futures):
        key = futures[fut]
        try:
            arr = fut.result()  # array only
        except Exception as e:
            logger.error(f"Error downloading layer {key} for chunk {bounds_str}: {e}")
            return f"Failed to download layer {key} for chunk {bounds_str}: {e}", chunk_stats
        layers[key] = arr

    # Fill missing inputs
    uint8_list = [
        'land_cover',
        'peat',
        'planted_forest_type',
        'extraction',
        'nutrient_status'
    ]
    int16_list = ['climate_domain', 'descals_type']
    int32_list = []
    float32_list = ['dadap', 'osm_roads', 'osm_canals', 'engert', 'grip']

    # If multi-year, add the burned keys to fill list
    if interval_start_year and interval_end_year:
        for yr in range(interval_start_year, interval_end_year + 1):
            uint8_list.append(f"{cn.burned_area_final_pattern}_{yr}")

    layers = uu.fill_missing_input_layers_with_no_data(
        layers,
        uint8_list,
        int16_list,
        int32_list,
        float32_list,
        bounds_str,
        tile_id,
        is_final,
        logger
    )

    # Stats for input arrays
    for k, arr in layers.items():
        stats = uu.calculate_stats(arr, k, bounds_str, tile_id, 'input_layer')
        chunk_stats.append(stats)

    # Create typed dictionaries for Numba
    typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32 = nu.create_typed_dicts(layers)

    lu.print_and_log(f"Calculating drainage in chunk {bounds_str} for {tile_id}: {uu.timestr()}",
                     is_final, logger)
    try:
        out_dict_uint32, out_dict_float32 = calculate_drainage_and_emissions(
            typed_dict_uint8,
            typed_dict_int16,
            typed_dict_float32
        )
    except Exception as e:
        logger.error(f"Error in drainage/emissions function: {e}")
        return f"Failed in Numba function for {bounds_str}: {e}", chunk_stats

    # Combine
    out_dict_all_dtypes = {**out_dict_uint32, **out_dict_float32}

    # Stats for output arrays
    for k, arr in out_dict_all_dtypes.items():
        stats = uu.calculate_stats(arr, k, bounds_str, tile_id, 'output_layer')
        chunk_stats.append(stats)

    # Optionally upload
    if not no_upload:
        out_no_data_val = 0
        # Single-year vs multi-year naming
        if interval_start_year and interval_end_year and interval_start_year != interval_end_year:
            year = f"{interval_start_year}_{interval_end_year}"
        else:
            year = "2020"  # fallback

        for key, arr in out_dict_all_dtypes.items():
            data_type = arr.dtype.name
            out_pattern = key
            out_dict_all_dtypes[key] = [arr, data_type, out_pattern, year]

        uu.save_and_upload_small_raster_set(
            bounds, chunk_length_pixels, tile_id, bounds_str,
            out_dict_all_dtypes, is_final, logger, out_no_data_val
        )

    return f"Success for {bounds_str}: {uu.timestr()}", chunk_stats


def run_drainage_model(cluster_name=None,
                       bounding_box=None,
                       chunk_size=None,
                       run_local=False,
                       no_stats=False,
                       no_log=False,
                       no_upload=False,
                       start_year=None,
                       end_year=None):
    """
    Main function adapted to explicitly align burned areas and land cover intervals with LULUCF.
    """
    stage = "drainage_model"
    start_time = uu.timestr()

    # Connect to Coiled or local Dask cluster
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Setup logging
    main_logger, main_log_local_path = lu.populate_main_log_header(
        bounding_box=bounding_box,
        use_shapefile=False,
        client=client,
        cluster=cluster,
        log_note="Drainage model run aligned with LULUCF annual/5-year intervals",
        run_local=run_local,
        model_type="organic_soils",
        stage=stage
    )
    main_logger.info(f"Stage {stage} started at: {start_time}")

    # Default bounding box/chunk size if not set
    bounding_box = bounding_box or [110, -10, 120, 0]
    chunk_size = chunk_size or 2

    chunk_list = uu.get_chunk_bounds(bounding_box, chunk_size)
    is_final = len(chunk_list) > 20

    if is_final:
        main_logger.info("Running as final model due to >20 chunks")

    futures = []
    if start_year and end_year and (start_year <= end_year):
        # Interval-based (annual or multi-year)
        main_logger.info(f"Running intervals: {start_year}-{end_year}")
        for yr in range(start_year, end_year + 1):
            for bds in chunk_list:
                tile_id = uu.xy_to_tile_id(bds[0], bds[3])
                download_dict = cn.get_dynamic_download_dict(tile_id, yr)

                # Determine data types from download dict for the first tile
                download_dict_with_data_types = uu.add_file_type_to_dict(download_dict)

                fut = client.submit(
                    calculate_and_upload_drainage,
                    bds,
                    download_dict_with_data_types,
                    is_final,
                    no_upload,
                    interval_start_year=yr,
                    interval_end_year=yr
                )
                futures.append(fut)
    else:
        # Single-interval fallback (default to 2020 if no year provided)
        year = start_year or 2020
        main_logger.info(f"Running single interval year: {year}")
        for bds in chunk_list:
            tile_id = uu.xy_to_tile_id(bds[0], bds[3])
            download_dict = cn.get_dynamic_download_dict(tile_id, year)

            download_dict_with_data_types = uu.add_file_type_to_dict(download_dict)

            fut = client.submit(
                calculate_and_upload_drainage,
                bds,
                download_dict_with_data_types,
                is_final,
                no_upload,
                interval_start_year=year,
                interval_end_year=year
            )
            futures.append(fut)

    # Gather results
    results = client.gather(futures)
    success_count = sum("Success" in msg for msg, _ in results)
    skipping_count = sum("Skipped" in msg for msg, _ in results)
    all_stats = [stat for _, stats in results for stat in stats]

    main_logger.info(f"Number of 'Success' chunks: {success_count}")
    main_logger.info(f"Number of 'Skipped' chunks: {skipping_count}")

    # Optional chunk-level stats
    if not no_stats:
        try:
            uu.calculate_chunk_stats(all_stats, stage)
        except AttributeError as e:
            main_logger.info(f"Cannot print chunk stats: {e}")

    end_time = uu.timestr()
    main_logger.info(f"Stage {stage} ended at: {end_time}")
    uu.stage_duration(start_time, end_time, stage)

    if not run_local:
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)
        client.close()
        cluster.close()
    else:
        main_logger.info("Local run completed. Worker logs not compiled.")


def main(argv=None):
    """
    Command-line entry point for drainage model with optional multi-year burned areas.
    If no CLI args, we run a small local test.
    """
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Drainage model (with burned-area logic).")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name', default=None)
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')
    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Skip chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Skip worker log merging & uploading')
    parser.add_argument('--no_upload', action='store_true', help='Skip uploading outputs to S3')

    # Optional multi-interval approach
    parser.add_argument('--start_year', type=int, help='Start year for multi-year approach')
    parser.add_argument('--end_year', type=int, help='End year for multi-year approach')

    args = parser.parse_args(argv)

    # If no arguments, do local test run
    if not argv:
        print("No CLI args provided. Using defaults for test run.")
        run_drainage_model(
            cluster_name=None,
            bounding_box=[112, -2, 113, -1],
            chunk_size=1,
            run_local=True,
            no_stats=False,
            no_log=False,
            no_upload=False,
            start_year=2015,
            end_year=2020
        )
    else:
        run_drainage_model(
            cluster_name=args.cluster_name,
            bounding_box=args.bounding_box,
            chunk_size=args.chunk_size,
            run_local=args.run_local,
            no_stats=args.no_stats,
            no_log=args.no_log,
            no_upload=args.no_upload,
            start_year=args.start_year,
            end_year=args.end_year
        )


if __name__ == "__main__":
    main()
