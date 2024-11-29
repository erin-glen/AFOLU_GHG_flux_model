# drainage_model.py

import argparse
import concurrent.futures
import numpy as np
import gc
import os

from dask.distributed import Client
from numba import jit, types
from numba.typed import Dict
from datetime import datetime

# Function to calculate drainage and emissions using Numba
from numba import jit, types
from numba.typed import Dict
import numpy as np

# Project-specific imports (ensure these modules are available)
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu
from src.scripts.utilities import numba_utilities as nu

# Import constants
c_to_co2 = np.float32(cn.c_to_co2)
n2o_n_to_n2o = np.float32(cn.n2o_n_to_n2o)
gwp_ch4 = np.float32(cn.gwp_ch4)
gwp_n2o = np.float32(cn.gwp_n2o)

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


# Main function
@jit(nopython=True)
def calculate_drainage_and_emissions(in_dict_uint8, in_dict_int16, in_dict_float32):
    """
    Calculates drainage status and emissions (CO₂, N₂O, CH₄) for each pixel.

    Args:
        in_dict_uint8 (Dict): Dictionary of uint8 input arrays.
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
    land_cover_block = in_dict_uint8['IPCC_basic_classes_2020']
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

    # Initialize output arrays
    rows, cols = peat_block.shape
    soil_block = np.zeros((rows, cols), dtype=np.uint32)
    state_out_block = np.zeros((rows, cols), dtype=np.uint32)
    co2_emissions_out = np.zeros((rows, cols), dtype=np.float32)
    n2o_emissions_out = np.zeros((rows, cols), dtype=np.float32)
    ch4_land_emissions_out = np.zeros((rows, cols), dtype=np.float32)
    ch4_ditch_emissions_out = np.zeros((rows, cols), dtype=np.float32)
    co2_offsite_emissions_out = np.zeros((rows, cols), dtype=np.float32)

    # Iterate over each pixel
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
            ecozone = ecozone_block[row, col]  # Numeric code for ecozone
            nutrient = nutrient_block[row, col]  # Numeric code for nutrient status
            descals_type = descals_type_block[row, col]

            ef_co2 = np.float32(0.0)
            ef_n2o = np.float32(0.0)
            ef_ch4_land = np.float32(0.0)
            ef_ch4_ditch = np.float32(0.0)
            ef_co2_offsite = np.float32(0.0)
            frac_ditch = np.float32(0.0)

            node = 0

            if peat == 1:
                node = nu.accrete_node(node, 1)
                if dadap > 0 or osm_canals > 0:
                    node = nu.accrete_node(node, 1)
                    soil_block[row, col] = 1  # 'drained'
                elif engert > 0 or grip > 0 or osm_roads > 0:
                    node = nu.accrete_node(node, 2)
                    soil_block[row, col] = 1  # 'drained'
                elif land_cover == cropland_code or land_cover == settlement_code:
                    node = nu.accrete_node(node, 3)
                    soil_block[row, col] = 1  # 'drained'
                elif planted_forest_type or descals_type > 0:
                    node = nu.accrete_node(node, 4)
                    soil_block[row, col] = 1  # 'drained'
                elif extraction > 0:
                    node = nu.accrete_node(node, 5)
                    soil_block[row, col] = 1  # 'drained'
                else:
                    node = nu.accrete_node(node, 6)
                    soil_block[row, col] = 0  # 'undrained peat'
            else:
                node = nu.accrete_node(node, 2)
                soil_block[row, col] = 0  # 'not peat'

            # Update state_out with the node value
            state_out_block[row, col] = node

            # New decision tree for emission factors where soil_block == 1
            if soil_block[row, col] == 1:
                node = nu.accrete_node(node, 1)
                # Start of emission factor decision tree
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
                            ef_co2 = 0.0  # Handle unknown nutrient status
                            ef_n2o = 0.0
                            ef_ch4_land = 0.0
                            ef_ch4_ditch = 0.0
                            frac_ditch = 0
                    elif land_cover == grassland_code:
                        node = nu.accrete_node(node, 2)
                        ef_co2 = 5.7
                        ef_n2o = 9.5
                        ef_ch4_land = 1.4
                        ef_ch4_ditch = 1165.0  # using deep default
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
                        ef_co2 = 0.0  # No emissions or default value
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
                        ef_ch4_ditch = 1165.0  # using deep default
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
                            ef_ch4_land = 16.0  # using deep default
                            frac_ditch = 0.05
                        else:
                            node = nu.accrete_node(node, 3)
                            ef_co2 = 0.0  # Handle unknown nutrient status
                            ef_n2o = 0.0
                            ef_ch4_land = 0.0
                            ef_ch4_ditch = 0.0
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
                        ef_co2 = 0.0  # No emissions or default value
                        ef_n2o = 0.0
                        ef_ch4_land = 0.0
                        ef_ch4_ditch = 0.0
                        frac_ditch = 0
                elif ecozone == tropical_code:
                    node = nu.accrete_node(node, 3)
                    ef_co2_offsite = 0.82
                    ef_ch4_ditch = 2259.0  # Assigned before conditions
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
                            ef_co2 = 0.0  # Handle unknown plantation type
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
                        ef_co2 = 0.0  # No emissions or default value
                        ef_n2o = 0.0
                        ef_ch4_land = 0.0
                        ef_ch4_ditch = 0.0
                else:
                    node = nu.accrete_node(node, 4)
                    # Handle unknown ecozone by setting emission factors to zero
                    ef_co2 = 0.0
                    ef_n2o = 0.0
                    ef_ch4_land = 0.0
                    ef_ch4_ditch = 0.0
                    ef_co2_offsite = 0.0

                # Update state_out with the node value after emission factor decisions
                state_out_block[row, col] = node

                # Use the helper function to calculate emissions in CO₂e
                (co2_emissions,
                 n2o_emissions_co2e,
                 ch4_land_emissions_co2e,
                 ch4_ditch_emissions_co2e,
                 co2_offsite_emissions
                 ) = nu.calculate_emissions_co2e(
                    ef_co2, ef_n2o, ef_ch4_land, ef_ch4_ditch, ef_co2_offsite, frac_ditch,
                    c_to_co2, n2o_n_to_n2o, gwp_n2o, gwp_ch4
                )
                # Assign emissions to output arrays
                co2_emissions_out[row, col] = co2_emissions
                n2o_emissions_out[row, col] = n2o_emissions_co2e
                ch4_land_emissions_out[row, col] = ch4_land_emissions_co2e
                ch4_ditch_emissions_out[row, col] = ch4_ditch_emissions_co2e
                co2_offsite_emissions_out[row, col] = co2_offsite_emissions

            else:
                node = nu.accrete_node(node, 2)
                # Update state_out with the node value
                state_out_block[row, col] = node
                # No emissions for undrained peat or non-peat areas
                co2_emissions_out[row, col] = 0.0
                n2o_emissions_out[row, col] = 0.0
                ch4_land_emissions_out[row, col] = 0.0
                ch4_ditch_emissions_out[row, col] = 0.0
                co2_offsite_emissions_out[row, col] = 0.0

    # Add outputs to dictionaries
    out_dict_uint32["soil"] = soil_block
    out_dict_uint32["state"] = state_out_block
    out_dict_float32["co2_emissions"] = co2_emissions_out
    out_dict_float32["n2o_emissions_co2e"] = n2o_emissions_out
    out_dict_float32["ch4_land_emissions_co2e"] = ch4_land_emissions_out
    out_dict_float32["ch4_ditch_emissions_co2e"] = ch4_ditch_emissions_out
    out_dict_float32["co2_offsite_emissions"] = co2_offsite_emissions_out

    # Optionally, calculate total emissions
    total_emissions_out = (co2_emissions_out + co2_offsite_emissions_out +
                           n2o_emissions_out + ch4_land_emissions_out +
                           ch4_ditch_emissions_out)
    out_dict_float32["total_emissions"] = total_emissions_out

    return out_dict_uint32, out_dict_float32


def calculate_and_upload_drainage(bounds, download_dict_with_data_types, is_final, no_upload):
    logger = lu.setup_logging()

    bounds_str = uu.boundstr(bounds)
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)

    chunk_stats = []

    # Replace placeholder {tile_id} in download_dict with the actual tile ID
    updated_download_dict = uu.replace_tile_id_in_dict(download_dict_with_data_types, tile_id)

    # Check if the necessary tiles exist
    tile_exists = uu.check_for_tile(updated_download_dict, is_final, logger)
    if not tile_exists:
        return f"Skipped chunk {bounds_str} because {tile_id} does not exist for any inputs: {uu.timestr()}", chunk_stats

    # Prepare to download the chunk
    futures = uu.prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_final, logger)

    # Wait for downloads to complete and collect layers
    layers = {}
    for future in concurrent.futures.as_completed(futures):
        layer = futures[future]
        try:
            layers[layer] = future.result()
        except Exception as e:
            logger.error(f"Error downloading layer {layer} for chunk {bounds_str}: {e}")
            return f"Failed to download layer {layer} for chunk {bounds_str}: {e}", chunk_stats

    # Define expected data type lists for layers
    # Define expected data type lists for layers
    uint8_list = ['IPCC_basic_classes_2020', 'peat', 'planted_forest_type', 'extraction', 'nutrient_status']
    int16_list = ['climate_domain', 'descals_type']
    int32_list = []
    float32_list = ['dadap', 'osm_roads', 'osm_canals', 'engert', 'grip']

    # Fill missing layers with NoData if necessary
    layers = uu.fill_missing_input_layers_with_no_data(
        layers, uint8_list, int16_list, int32_list, float32_list, bounds_str, tile_id, is_final, logger
    )

    # Troubleshooting Step 1: Log available layers after filling
    logger.info(f"Available layers after filling: {list(layers.keys())}")

    # Troubleshooting Step 2: Verify that all required layers are present
    expected_layers = uint8_list + int16_list + int32_list + float32_list
    missing_layers = [layer for layer in expected_layers if layer not in layers]
    if missing_layers:
        logger.error(f"Missing layers after filling: {missing_layers}")
        return f"Failed due to missing layers: {missing_layers}", chunk_stats

    # Troubleshooting Step 3: Log data types of each layer
    for key in layers:
        logger.info(f"Layer '{key}' data type: {layers[key].dtype}")

    # Create typed dictionaries for Numba functions
    typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32 = nu.create_typed_dicts(layers)

    # Troubleshooting Step 4: Log keys in typed dictionaries
    logger.info(f"Keys in typed_dict_uint8: {list(typed_dict_uint8.keys())}")
    logger.info(f"Keys in typed_dict_int16: {list(typed_dict_int16.keys())}")
    logger.info(f"Keys in typed_dict_int32: {list(typed_dict_int32.keys())}")
    logger.info(f"Keys in typed_dict_float32: {list(typed_dict_float32.keys())}")

    # Verify typed dictionaries have all required keys
    missing_uint8_keys = [key for key in uint8_list if key not in typed_dict_uint8]
    missing_int16_keys = [key for key in int16_list if key not in typed_dict_int16]
    missing_int32_keys = [key for key in int32_list if key not in typed_dict_int32]
    missing_float32_keys = [key for key in float32_list if key not in typed_dict_float32]
    if missing_uint8_keys or missing_int16_keys or missing_int32_keys or missing_float32_keys:
        logger.error(
            f"Typed dictionaries missing keys. uint8: {missing_uint8_keys}, int16: {missing_int16_keys}, int32: {missing_int32_keys}, float32: {missing_float32_keys}")
        return f"Failed due to missing keys in typed dictionaries", chunk_stats

    # Calculate statistics for input layers
    for key, array in layers.items():
        stats = uu.calculate_stats(array, key, bounds_str, tile_id, 'input_layer')
        chunk_stats.append(stats)

    # Run the drainage and emissions calculation
    lu.print_and_log(f"Calculating drainage and emissions in {bounds_str} in {tile_id}: {uu.timestr()}", is_final,
                     logger)
    try:
        out_dict_uint32, out_dict_float32 = calculate_drainage_and_emissions(
            typed_dict_uint8, typed_dict_int16, typed_dict_float32
        )
    except KeyError as e:
        logger.error(f"KeyError during Numba function execution: {e}")
        logger.error("Possible missing key in typed dictionaries passed to Numba function.")
        return f"Failed due to KeyError in Numba function: {e}", chunk_stats
    except Exception as e:
        logger.error(f"Exception during Numba function execution: {e}")
        return f"Failed due to exception in Numba function: {e}", chunk_stats

    # Combine outputs into a single dictionary
    out_dict_all_dtypes = {**out_dict_uint32, **out_dict_float32}

    # Calculate statistics for output layers
    for key, array in out_dict_all_dtypes.items():
        stats = uu.calculate_stats(array, key, bounds_str, tile_id, 'output_layer')
        chunk_stats.append(stats)

    # Save and upload outputs if required
    if not no_upload:
        out_no_data_val = 0  # Define NoData value if needed

        # Prepare output dictionary for saving
        for key, value in out_dict_all_dtypes.items():
            data_type = value.dtype.name
            out_pattern = key
            year = 2020  # Adjust if necessary
            out_dict_all_dtypes[key] = [value, data_type, out_pattern, f'{year}']

        # Save and upload raster outputs
        uu.save_and_upload_small_raster_set(
            bounds, chunk_length_pixels, tile_id, bounds_str,
            out_dict_all_dtypes, is_final, logger, out_no_data_val
        )

    # Clear memory
    del out_dict_all_dtypes
    del layers
    gc.collect()

    success_message = f"Success for {bounds_str}: {uu.timestr()}"
    return success_message, chunk_stats


def run_drainage_model(cluster_name=None, bounding_box=None, chunk_size=None,
                       run_local=False, no_stats=False, no_log=False, no_upload=False):
    """
    Main function to run the drainage model.

    Args:
        cluster_name (str, optional): Name of the Coiled cluster.
        bounding_box (list, optional): List of coordinates [W, S, E, N] in degrees.
        chunk_size (float, optional): Size of each chunk in degrees.
        run_local (bool, optional): Run locally without Dask/Coiled.
        no_stats (bool, optional): Do not create the chunk stats spreadsheet.
        no_log (bool, optional): Do not create the combined log.
        no_upload (bool, optional): Do not save and upload outputs to S3.

    Returns:
        None
    """
    # Set default values if None
    if cluster_name is None:
        cluster_name = 'default_cluster'
    if bounding_box is None:
        bounding_box = [110, -10, 120, 0]  # Default bounding box
    if chunk_size is None:
        chunk_size = 2  # Default chunk size

    # Connect to cluster
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Stage information
    stage = 'drainage_model'
    start_time = uu.timestr()
    print(f"Stage {stage} started at: {start_time}")

    # Prepare chunks
    chunks = uu.get_chunk_bounds(bounding_box, chunk_size)
    print(f"Processing {len(chunks)} chunks")

    # Determine if the run is final
    is_final = False
    if len(chunks) > 20:
        is_final = True
        print("Running as final model.")

    # Accumulate stats and messages
    all_stats = []
    return_messages = []

    # Prepare the download dictionary
    # This dictionary should include paths to all required input datasets
    download_dict = cn.download_dict

    # Get first tile names and data types
    print(f"Getting tile_id of first tile in each tile set: {uu.timestr()}")
    first_tiles = uu.first_file_name_in_s3_folder(download_dict)

    print(f"Getting datatype of first tile in each tile set: {uu.timestr()}")
    download_dict_with_data_types = uu.add_file_type_to_dict(first_tiles)

    # Create tasks and start processing
    print(f"Creating tasks and starting processing: {uu.timestr()}")
    futures = []
    for chunk in chunks:
        future = client.submit(
            calculate_and_upload_drainage,
            chunk, download_dict_with_data_types, is_final, no_upload
        )
        futures.append(future)

    # Collect the results once they are finished
    results = client.gather(futures)

    # Process results
    success_count = 0
    skipping_chunk_count = 0

    for result in results:
        return_message, chunk_stats = result

        print(return_message)

        if "Success" in return_message:
            success_count += 1

        if "skipped chunk" in return_message.lower():
            skipping_chunk_count += 1

        if return_message:
            return_messages.append(return_message)

        if chunk_stats is not None:
            all_stats.extend(chunk_stats)

    # Print counts
    print(f"Number of 'Success' chunks: {success_count}")
    print(f"Number of 'skipped chunk' chunks: {skipping_chunk_count}")

    # Calculate stats if not suppressed
    if not no_stats:
        try:
            uu.calculate_chunk_stats(all_stats, stage)
        except AttributeError:
            print(
                "Can't print chunk stats: module 'src.scripts.utilities.constants_and_names' has no attribute 'chunk_stats_path'")

    # End time
    end_time = uu.timestr()
    print(f"Stage {stage} ended at: {end_time}")
    uu.stage_duration(start_time, end_time, stage)

    # Compile and upload logs
    log_note = "Drainage model run"
    try:
        lu.compile_and_upload_log(
            no_log,
            client,
            cluster,
            stage,
            len(chunks),  # Total number of chunks
            chunk_size,
            start_time,
            end_time,
            success_count,  # New argument: Count of successfully processed chunks
            skipping_chunk_count,  # New argument: Count of skipped chunks
            log_note  # Log note
        )

    except AttributeError as e:
        print(f"Error during log compilation and upload: {e}")

    # Close the client and cluster if not running locally
    if not run_local:
        client.close()
        cluster.close()


def main(argv=None):
    """
    Main function to run the drainage model from the command line.

    This script calculates the drainage model using specified parameters.
    It can be run with default settings or customized via command-line arguments.
    """
    import sys
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Calculate drainage model with emissions.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')
    args = parser.parse_args(argv)

    # If no arguments are provided, use default values
    if not argv:
        print("No command-line arguments provided. Using default values for testing.")
        run_drainage_model(
            cluster_name='drainage',

            # Define your test bounding box here
            # bounding_box=[110, -10, 120, 0],  # Example bounding box
            # bounding_box=[112, -4, 114, -2],  # one 2-degree chunk with data in Borneo 00N_110E
            # bounding_box=[110, -10, 120, 0], # 10x10 degree tile Borneo
            bounding_box=[-74, -4, -72, -2],  # one 2-degree chunk with data Peru 00N_080W
            # bounding_box=[-80.0, -10.0, -70.0, 0.0],  # 10x10 degree tile peru 00N_080W
            # bounding_box=[16.0, 6.0, 18.0, 8.0],  # 2-degree chunk Congo 10N_010E
            # bounding_box=[-10.0, 0.0, 0.0, 10],  # 10x10 degree tile Congo 10N_010E
            # bounding_box=[-8, 52, -6, 54, 2],  # 2-degree chunk Ireland 60N_010W
            # bounding_box=[-110.0, 50.0, -100.0, 60.0],  # 10x10 degree tile Ireland 60N_010W
            chunk_size=2,
            run_local=True,
            no_stats=False,
            no_log=False,
            no_upload=False
        )
    else:
        run_drainage_model(
            cluster_name=args.cluster_name,
            bounding_box=args.bounding_box,
            chunk_size=args.chunk_size,
            run_local=args.run_local,
            no_stats=args.no_stats,
            no_log=args.no_log,
            no_upload=args.no_upload
        )


if __name__ == "__main__":
    main()
