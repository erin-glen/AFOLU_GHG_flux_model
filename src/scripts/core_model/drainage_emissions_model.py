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

# Project-specific imports
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu
from src.scripts.utilities import numba_utilities as nu

# ----------------------
#  Constants and Codes
# ----------------------
c_to_co2 = np.float32(cn.c_to_co2)
n2o_n_to_n2o = np.float32(cn.n2o_n_to_n2o)
gwp_ch4 = np.float32(cn.gwp_ch4)
gwp_n2o = np.float32(cn.gwp_n2o)
gwp_co = np.float32(cn.gwp_co)
combustion_factor = np.float32(cn.combustion_factor)

forest_code = cn.ipcc_codes['forest']
cropland_code = cn.ipcc_codes['cropland']
settlement_code = cn.ipcc_codes['settlement']
wetland_code = cn.ipcc_codes['wetland']
grassland_code = cn.ipcc_codes['grassland']
otherland_code = cn.ipcc_codes['otherland']

boreal_code = cn.ecozone_codes['boreal']
temperate_code = cn.ecozone_codes['temperate']
tropical_code = cn.ecozone_codes['tropical']
unknown_ecozone_code = cn.ecozone_codes['unknown']

poor_nutrient_code = cn.nutrient_status_codes['poor']
rich_nutrient_code = cn.nutrient_status_codes['rich']
unknown_nutrient_code = cn.nutrient_status_codes['unknown']

long_rotation_code = cn.plantation_type_codes['long_rotation']
short_rotation_code = cn.plantation_type_codes['short_rotation']
oil_palm_code = cn.plantation_type_codes['oil_palm']
sago_palm_code = cn.plantation_type_codes['sago_palm']
unknown_plantation_code = cn.plantation_type_codes['unknown']


@jit(nopython=True)
def calculate_drainage_and_emissions(in_dict_uint8, in_dict_int16, in_dict_float32):
    """
    Calculates:
      1) Drainage status (soil_block) and state (state_out_block) for each pixel,
      2) Drainage-based emissions (CO₂, N₂O, CH₄, offsite CO₂, plus a total CO₂e sum),
      3) Additional burned-area emissions (CO₂, CO, CH₄, and total CO₂e)
         if a burned-area layer is provided.
    """
    # Prepare typed output dictionaries
    out_dict_uint32 = Dict.empty(key_type=types.unicode_type, value_type=types.uint32[:, :])
    out_dict_float32 = Dict.empty(key_type=types.unicode_type, value_type=types.float32[:, :])

    # Required input arrays
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

    # Optional combined burned-area key
    burned_block = None
    for k in in_dict_uint8.keys():
        if k.startswith("burned_area_combined_"):
            burned_block = in_dict_uint8[k]
            break

    # Optional pixel_area_ha block (fallback to ones if missing)
    rows_, cols_ = peat_block.shape
    if 'pixel_area_ha' in in_dict_float32:
        pixel_area_block = in_dict_float32['pixel_area_ha']
    else:
        pixel_area_block = np.ones((rows_, cols_), dtype=np.float32)

    rows, cols = peat_block.shape

    # Prepare output arrays
    soil_block = np.zeros((rows, cols), dtype=np.uint32)
    state_out_block = np.zeros((rows, cols), dtype=np.uint32)

    # Drainage partial + total
    drained_co2_out = np.zeros((rows, cols), dtype=np.float32)
    drained_n2o_out = np.zeros((rows, cols), dtype=np.float32)
    drained_ch4_land_out = np.zeros((rows, cols), dtype=np.float32)
    drained_ch4_ditch_out = np.zeros((rows, cols), dtype=np.float32)
    drained_co2_offsite_out = np.zeros((rows, cols), dtype=np.float32)
    drained_total_co2e_out = np.zeros((rows, cols), dtype=np.float32)

    # Burned partial + total
    burned_state_out = np.zeros((rows, cols), dtype=np.uint32)
    burned_co2_out = np.zeros((rows, cols), dtype=np.float32)
    burned_co_out = np.zeros((rows, cols), dtype=np.float32)
    burned_ch4_out = np.zeros((rows, cols), dtype=np.float32)
    burned_total_co2e_out = np.zeros((rows, cols), dtype=np.float32)

    # Main loop
    for row in range(rows):
        for col in range(cols):
            pixel_area_ha = pixel_area_block[row, col]
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

            ef_co2 = np.float32(0.0)
            ef_n2o = np.float32(0.0)
            ef_ch4_land = np.float32(0.0)
            ef_ch4_ditch = np.float32(0.0)
            ef_co2_offsite = np.float32(0.0)
            frac_ditch = np.float32(0.0)

            node = 0
            drained = False

            # A) Drainage Classification
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
                elif planted_forest_type > 0 or descals_type > 0:
                    node = nu.accrete_node(node, 4)
                    drained = True
                elif extraction > 0:
                    node = nu.accrete_node(node, 5)
                    drained = True
                else:
                    node = nu.accrete_node(node, 6)

                if drained:
                    soil_block[row, col] = 2  # drained
                else:
                    soil_block[row, col] = 1  # undrained
            else:
                node = nu.accrete_node(node, 2)
                soil_block[row, col] = 0  # not peat

            state_out_block[row, col] = node

            # B) Drainage EF / Emissions
            if soil_block[row, col] == 2:
                node = nu.accrete_node(node, 1)

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

                elif ecozone == tropical_code:
                    node = nu.accrete_node(node, 3)
                    ef_co2_offsite = 0.82
                    ef_ch4_ditch = 2259.0
                    frac_ditch = 0.02
                    if planted_forest_type > 0:
                        node = nu.accrete_node(node, 1)
                        if planted_forest_type == long_rotation_code:
                            ef_co2 = 15.0
                            ef_n2o = 2.4
                            ef_ch4_land = 2.7
                        elif planted_forest_type == short_rotation_code:
                            ef_co2 = 20.0
                            ef_n2o = 2.4
                            ef_ch4_land = 2.7
                        elif planted_forest_type == oil_palm_code:
                            ef_co2 = 11.0
                            ef_n2o = 1.2
                            ef_ch4_land = 0.0
                        elif planted_forest_type == sago_palm_code:
                            ef_co2 = 1.5
                            ef_n2o = 3.3
                            ef_ch4_land = 26.2
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

                # Calculate drainage EFs (per ha)
                (co2_emissions,
                 n2o_emissions_co2e,
                 ch4_land_emissions_co2e,
                 ch4_ditch_emissions_co2e,
                 co2_offsite_emissions,
                 drainage_total_co2e) = nu.calculate_drainage_emissions_co2e(
                     ef_co2,
                     ef_n2o,
                     ef_ch4_land,
                     ef_ch4_ditch,
                     ef_co2_offsite,
                     frac_ditch,
                     c_to_co2,
                     n2o_n_to_n2o,
                     gwp_n2o,
                     gwp_ch4,
                     pixel_area_ha
                )

                # Store partial + total
                drained_co2_out[row, col] = co2_emissions
                drained_n2o_out[row, col] = n2o_emissions_co2e
                drained_ch4_land_out[row, col] = ch4_land_emissions_co2e
                drained_ch4_ditch_out[row, col] = ch4_ditch_emissions_co2e
                drained_co2_offsite_out[row, col] = co2_offsite_emissions
                drained_total_co2e_out[row, col] = drainage_total_co2e

            else:
                # not drained or not peat => zero out partial & total
                drained_co2_out[row, col] = 0.0
                drained_n2o_out[row, col] = 0.0
                drained_ch4_land_out[row, col] = 0.0
                drained_ch4_ditch_out[row, col] = 0.0
                drained_co2_offsite_out[row, col] = 0.0
                drained_total_co2e_out[row, col] = 0.0

            # (C) Burned-Area Emissions
            burned_node = 0
            if burned_block is not None:
                burned_val = burned_block[row, col]
                if burned_val > 0 and soil_block[row, col] in (1, 2):
                    if ecozone == boreal_code:
                        burned_node = nu.accrete_node(burned_node, 1)
                        if soil_block[row, col] == 2:
                            gef_co2, gef_co, gef_ch4 = 1650.0, 110.0, 12.0
                            mass_burnt = 250.0
                        else:
                            gef_co2, gef_co, gef_ch4 = 1450.0, 90.0, 10.0
                            mass_burnt = 75.0
                    elif ecozone == temperate_code:
                        burned_node = nu.accrete_node(burned_node, 2)
                        if soil_block[row, col] == 2:
                            gef_co2, gef_co, gef_ch4 = 1650.0, 110.0, 12.0
                            mass_burnt = 200.0
                        else:
                            gef_co2, gef_co, gef_ch4 = 1450.0, 90.0, 10.0
                            mass_burnt = 50.0
                    elif ecozone == tropical_code:
                        burned_node = nu.accrete_node(burned_node, 3)
                        if soil_block[row, col] == 2:
                            if land_cover in (cropland_code,) or planted_forest_type > 0:
                                gef_co2, gef_co, gef_ch4 = 1700.0, 200.0, 15.0
                                mass_burnt = 150.0
                            else:
                                gef_co2, gef_co, gef_ch4 = 1600.0, 180.0, 14.0
                                mass_burnt = 300.0
                        else:
                            gef_co2, gef_co, gef_ch4 = 0.0, 0.0, 0.0
                            mass_burnt = 0.0
                    else:
                        burned_node = nu.accrete_node(burned_node, 4)
                        gef_co2, gef_co, gef_ch4 = 0.0, 0.0, 0.0
                        mass_burnt = 0.0

                    (burn_co2,
                     burn_co,
                     burn_ch4,
                     burn_total_co2e) = nu.calculate_burned_area_emissions(
                         np.float32(pixel_area_ha),  # pass pixel area as float32
                         np.float32(mass_burnt),
                         combustion_factor,
                         np.float32(gef_co2),
                         np.float32(gef_co),
                         np.float32(gef_ch4),
                         gwp_co,
                         gwp_ch4
                    )

                    burned_co2_out[row, col] = burn_co2
                    burned_co_out[row, col] = burn_co
                    burned_ch4_out[row, col] = burn_ch4
                    burned_total_co2e_out[row, col] = burn_total_co2e
                else:
                    burned_co2_out[row, col] = 0.0
                    burned_co_out[row, col] = 0.0
                    burned_ch4_out[row, col] = 0.0
                    burned_total_co2e_out[row, col] = 0.0
            else:
                burned_co2_out[row, col] = 0.0
                burned_co_out[row, col] = 0.0
                burned_ch4_out[row, col] = 0.0
                burned_total_co2e_out[row, col] = 0.0

            burned_state_out[row, col] = burned_node

    # Populate out_dict for integer outputs
    out_dict_uint32["soil"] = soil_block
    out_dict_uint32["state"] = state_out_block
    out_dict_uint32["burned_state"] = burned_state_out

    # Drainage partial + total
    out_dict_float32["drained_co2"] = drained_co2_out
    out_dict_float32["drained_n2o_co2e"] = drained_n2o_out
    out_dict_float32["drained_ch4_land_co2e"] = drained_ch4_land_out
    out_dict_float32["drained_ch4_ditch_co2e"] = drained_ch4_ditch_out
    out_dict_float32["drained_co2_offsite"] = drained_co2_offsite_out
    out_dict_float32["drained_total_co2e"] = drained_total_co2e_out

    # Burned partial + total
    out_dict_float32["burned_co2"] = burned_co2_out
    out_dict_float32["burned_co_co2e"] = burned_co_out
    out_dict_float32["burned_ch4_co2e"] = burned_ch4_out
    out_dict_float32["burned_total_co2e"] = burned_total_co2e_out

    return out_dict_uint32, out_dict_float32


def combine_burned_area(layers, interval_start, interval_end):
    """
    Combine all burned-area layers from interval_start..interval_end
    into a single mask (1=burned in any year, 0=unburned).
    """
    any_burned_key = False
    shape = None
    combined = None

    for yr in range(interval_start, interval_end + 1):
        ba_key = f"{cn.burned_area_final_pattern}_{yr}"
        if ba_key in layers:
            arr = layers[ba_key]
            if shape is None:
                shape = arr.shape
                combined = np.zeros(shape, dtype=np.uint8)
            combined = combined + (arr > 0).astype(np.uint8)
            any_burned_key = True

    if any_burned_key and combined is not None:
        combined[combined > 0] = 1
        out_key = f"burned_area_combined_{interval_start}_{interval_end}"
        layers[out_key] = combined


def get_intervals(start_year, end_year, interval_type):
    intervals = []
    if interval_type == "annual":
        for y in range(start_year, end_year + 1):
            intervals.append((y, y))
    elif interval_type == "five_year":
        current = start_year
        while current <= end_year:
            interval_end = current + 4
            if interval_end > end_year:
                interval_end = end_year
            intervals.append((current, interval_end))
            current += 5
    else:
        intervals.append((start_year, end_year))
    return intervals


def calculate_and_upload_drainage(bounds,
                                  download_dict_with_data_types,
                                  is_final,
                                  no_upload,
                                  interval_start,
                                  interval_end,
                                  use_actual_pixel_area=False):
    """
    Chunk-level function that merges any burned layers, optionally loads 'pixel_area_ha',
    calculates drainage + fire EFs, and optionally uploads results.
    """
    logger = lu.setup_logging_worker()
    bounds_str = uu.boundstr(bounds)
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)
    chunk_stats = []

    updated_download_dict = uu.replace_tile_id_in_dict(download_dict_with_data_types, tile_id)

    tile_exists = uu.check_for_tile(updated_download_dict, is_final, logger)
    if not tile_exists:
        return (f"Skipped chunk {bounds_str} because {tile_id} does not exist for any inputs: {uu.timestr()}",
                chunk_stats)

    # Add burned keys in case they're missing
    for yr in range(interval_start, interval_end + 1):
        burned_key = f"{cn.burned_area_final_pattern}_{yr}"
        if burned_key not in updated_download_dict:
            updated_download_dict[burned_key] = None

    # Download
    futures = uu.prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_final, logger)
    layers = {}
    for fut in concurrent.futures.as_completed(futures):
        key = futures[fut]
        try:
            arr = fut.result()
        except Exception as e:
            logger.error(f"Error downloading layer {key} for chunk {bounds_str}: {e}")
            return f"Failed to download layer {key} for chunk {bounds_str}: {e}", chunk_stats
        layers[key] = arr

    # fill_missing
    uint8_list = ['peat', 'planted_forest_type', 'extraction', 'nutrient_status']
    int16_list = ['climate_domain', 'descals_type']
    int32_list = []
    # Always add 'pixel_area_ha' to float32 to have it recognized (can fallback to ones)
    float32_list = ['dadap', 'osm_roads', 'osm_canals', 'engert', 'grip', 'pixel_area_ha']

    # Single land cover
    uint8_list.append('land_cover')

    # Add burned
    for yr in range(interval_start, interval_end + 1):
        uint8_list.append(f"{cn.burned_area_final_pattern}_{yr}")

    layers = uu.fill_missing_input_layers_with_no_data(
        layers, uint8_list, int16_list, int32_list, float32_list,
        bounds_str, tile_id, is_final, logger
    )

    # Stats
    for k, arr in layers.items():
        stats = uu.calculate_stats(arr, k, bounds_str, tile_id, 'input_layer')
        chunk_stats.append(stats)

    # Combine burned
    combine_burned_area(layers, interval_start, interval_end)

    # If you had time-coded land cover, rename it here if desired

    # Create typed dicts
    typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32 = nu.create_typed_dicts(layers)

    lu.print_and_log(
        f"Calculating drainage for {tile_id}, {bounds_str}, years {interval_start}-{interval_end}: {uu.timestr()}",
        is_final, logger)
    try:
        out_dict_uint32, out_dict_float32 = calculate_drainage_and_emissions(
            typed_dict_uint8,
            typed_dict_int16,
            typed_dict_float32
        )
    except Exception as e:
        logger.error(f"Error in drainage/emissions: {e}")
        return f"Failed in Numba function for {bounds_str}: {e}", chunk_stats

    out_dict_all_dtypes = {**out_dict_uint32, **out_dict_float32}

    # Stats for outputs
    for k, arr in out_dict_all_dtypes.items():
        stats = uu.calculate_stats(arr, k, bounds_str, tile_id, 'output_layer')
        chunk_stats.append(stats)

    # Upload if desired
    if not no_upload:
        out_no_data_val = 0
        if interval_start != interval_end:
            year_str = f"{interval_start}_{interval_end}"
        else:
            year_str = f"{interval_start}"

        for key, arr in out_dict_all_dtypes.items():
            data_type = arr.dtype.name
            out_pattern = key
            out_dict_all_dtypes[key] = [arr, data_type, out_pattern, year_str]

        model_version_tag = cn.model_version_tag
        uu.save_and_upload_small_raster_set(
            bounds, chunk_length_pixels, tile_id, bounds_str,
            out_dict_all_dtypes, is_final, logger, model_version_tag, out_no_data_val
        )

    return f"Success for {bounds_str}, block {interval_start}-{interval_end}: {uu.timestr()}", chunk_stats


def run_drainage_model(cluster_name=None,
                       bounding_box=None,
                       chunk_size=None,
                       run_local=False,
                       no_stats=False,
                       no_log=False,
                       no_upload=False,
                       start_year=None,
                       end_year=None,
                       interval_type="annual",
                       use_actual_pixel_area=False):
    """
    Main function that can handle either 'annual' or 'five_year'
    blocks. Land cover + burned-area are time-based. Everything else is static.

    If use_actual_pixel_area==True, we try to load a 'pixel_area_ha'
    dataset for each tile (or fill it with 1.0 if missing).
    """
    stage = "drainage_model"
    start_time = uu.timestr()

    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    main_logger, main_log_local_path = lu.populate_main_log_header(
        bounding_box=bounding_box,
        use_shapefile=False,
        client=client,
        cluster=cluster,
        log_note="Drainage model multi-interval run",
        run_local=run_local,
        model_type="organic_soils",
        stage=stage
    )
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"use_actual_pixel_area={use_actual_pixel_area}")

    bounding_box = bounding_box or [110, -10, 120, 0]
    chunk_size   = chunk_size or 2
    chunk_list   = uu.get_chunk_bounds(bounding_box, chunk_size)
    is_final     = (len(chunk_list) > 20)
    if is_final:
        main_logger.info("Running as final model due to >20 chunks")

    if not (start_year and end_year):
        start_year = start_year or 2020
        end_year   = end_year or start_year
        interval_type = "annual"

    intervals = get_intervals(start_year, end_year, interval_type)

    if not chunk_list:
        main_logger.info("No chunks to process. Exiting.")
        return

    # figure out data types from first chunk
    first_chunk    = chunk_list[0]
    sample_tile_id = uu.xy_to_tile_id(first_chunk[0], first_chunk[3])
    sample_dict    = cn.get_dynamic_download_dict(sample_tile_id, start_year, end_year)

    # If user wants actual area => we add 'pixel_area_ha' for the sample tile
    if use_actual_pixel_area:
        # Must define cn.pixel_area_ha_dir in your constants if not existing
        sample_dict['pixel_area_ha'] = os.path.join(
            cn.pixel_area_ha_dir,  # e.g. "s3://bucket/pixel_area_ha/"
            f"{sample_tile_id}_pixel_area_ha.tif"
        )

    dwn_dict_types = uu.add_file_type_to_dict(sample_dict)

    futures = []
    for (iv_start, iv_end) in intervals:
        main_logger.info(f"Queueing tasks for {iv_start}-{iv_end}")
        for bds in chunk_list:
            fut = client.submit(
                calculate_and_upload_drainage,
                bds,
                dwn_dict_types,
                is_final,
                no_upload,
                iv_start,
                iv_end,
                use_actual_pixel_area
            )
            futures.append(fut)

    results = client.gather(futures)
    success_count = sum("Success" in msg for msg, _ in results)
    skip_count    = sum("Skipped" in msg for msg, _ in results)
    all_stats = [stat for _, stats in results for stat in stats]

    main_logger.info(f"Number of 'Success' chunks: {success_count}")
    main_logger.info(f"Number of 'Skipped' chunks: {skip_count}")

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
    Command-line entry point.

    Usage Example:
      python drainage_emissions_model.py \
        --start_year 2015 --end_year 2020 \
        --interval_type five_year \
        --use_actual_pixel_area
    """
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Drainage model with multi-interval approach.")
    parser.add_argument('-cn', '--cluster_name', default=None)
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')
    parser.add_argument('--run_local', action='store_true', help='Run locally without Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Skip stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Skip worker log merging')
    parser.add_argument('--no_upload', action='store_true', help='Skip uploading outputs to S3')

    parser.add_argument('--start_year', type=int, help='Starting year')
    parser.add_argument('--end_year', type=int, help='Ending year')
    parser.add_argument('--interval_type', choices=['annual', 'five_year'], default='annual',
                        help='annual vs five_year intervals')

    parser.add_argument('--use_actual_pixel_area', action='store_true',
                        help='If set, the model attempts to load a pixel_area_ha dataset and multiplies partial+total EFs by it')

    args = parser.parse_args(argv)

    if not argv:
        print("No CLI args provided. Running local test with default 1ha/pixel approach.")
        run_drainage_model(
            cluster_name=None,
            bounding_box=[112, -2, 113, -1],
            chunk_size=1,
            run_local=True,
            no_stats=False,
            no_log=False,
            no_upload=False,
            start_year=2015,
            end_year=2019,
            interval_type="five_year",
            use_actual_pixel_area=False
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
            end_year=args.end_year,
            interval_type=args.interval_type,
            use_actual_pixel_area=args.use_actual_pixel_area
        )


if __name__ == "__main__":
    main()
