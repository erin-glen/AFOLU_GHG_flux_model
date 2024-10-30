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

# Project-specific imports (ensure these modules are available)
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu
from src.scripts.utilities import numba_utilities as nu

# Define constants for land cover codes
forest_code = cn.ipcc_codes['forest']
cropland_code = cn.ipcc_codes['cropland']
settlement_code = cn.ipcc_codes['settlement']
wetland_code = cn.ipcc_codes['wetland']
grassland_code = cn.ipcc_codes['grassland']
otherland_code = cn.ipcc_codes['otherland']

# Function to calculate drainage and emissions using Numba
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
    ecozone_block = in_dict_uint8['ecozone']           # Ecozone codes: 1=boreal, 2=temperate, 3=tropical
    nutrient_block = in_dict_uint8['nutrient_status']  # Nutrient status codes: 1=poor, 2=rich

    # Initialize output arrays
    rows, cols = peat_block.shape
    soil_block = np.zeros((rows, cols), dtype=np.uint32)
    state_out = np.zeros((rows, cols), dtype=np.uint32)
    CO2_emissions_out = np.zeros((rows, cols), dtype=np.float32)
    N2O_emissions_out = np.zeros((rows, cols), dtype=np.float32)
    # Initialize arrays for CH₄ emissions split into land and ditch components
    CH4_land_emissions_out = np.zeros((rows, cols), dtype=np.float32)
    CH4_ditch_emissions_out = np.zeros((rows, cols), dtype=np.float32)
    # Initialize array for offsite CO₂ emissions
    CO2_offsite_emissions_out = np.zeros((rows, cols), dtype=np.float32)
    # Initialize placeholders for additional emissions if needed in the future
    # e.g., other_gas_emissions_out = np.zeros((rows, cols), dtype=np.float32)

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
            ecozone_code = ecozone_block[row, col]        # Numeric code for ecozone
            nutrient_status_code = nutrient_block[row, col]  # Numeric code for nutrient status

            node = 0
            ef_co2 = 0.0  # Initialize emission factor for CO₂
            ef_n2o = 0.0  # Initialize emission factor for N₂O
            ef_ch4_land = 0.0  # Initialize emission factor for CH₄ from land
            ef_ch4_ditch = 0.0  # Initialize emission factor for CH₄ from ditches
            ef_co2_offsite = 0.0  # Initialize emission factor for offsite CO₂

            if peat == 1:
                node = nu.accrete_node(node, 1)
                node = nu.accrete_node(node, 1)
                if dadap > 0 or osm_canals > 0:
                    node = nu.accrete_node(node, 1)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node
                elif engert > 0 or grip > 0 or osm_roads > 0:
                    node = nu.accrete_node(node, 2)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node
                elif land_cover == cropland_code or land_cover == settlement_code:
                    node = nu.accrete_node(node, 3)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node
                elif planted_forest_type > 0:
                    node = nu.accrete_node(node, 4)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node
                elif extraction > 0:
                    node = nu.accrete_node(node, 5)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node
                else:
                    node = nu.accrete_node(node, 6)
                    soil_block[row, col] = 0  # 'undrained peat'
                    state_out[row, col] = node
            else:
                soil_block[row, col] = 0  # 'not peat'
                node = nu.accrete_node(node, 2)
                state_out[row, col] = node

            # todo define data for ecozone, nutrient status, group plantation types, and pixel area
            # todo build function for calculating actual emissions

            # Map ecozone_code to ecozone string
            # Ecozone codes: 1=boreal, 2=temperate, 3=tropical
            if ecozone_code == 1:
                ecozone = 'boreal'
            elif ecozone_code == 2:
                ecozone = 'temperate'
            elif ecozone_code == 3:
                ecozone = 'tropical'
            else:
                ecozone = 'unknown'  # Handle unknown ecozone

            # Map nutrient_status_code to nutrient status
            # Nutrient status codes: 1=poor, 2=rich
            if nutrient_status_code == 1:
                nutrient = 'poor'
            elif nutrient_status_code == 2:
                nutrient = 'rich'
            else:
                nutrient = 'unknown'  # Handle unknown nutrient status

            # Determine plantation_type based on planted_forest_type
            # Plantation types: 1=long_rotation, 2=short_rotation, 3=oil_palm, 4=sago_palm
            if planted_forest_type == 1:
                plantation_type = 'long_rotation'
            elif planted_forest_type == 2:
                plantation_type = 'short_rotation'
            elif planted_forest_type == 3:
                plantation_type = 'oil_palm'
            elif planted_forest_type == 4:
                plantation_type = 'sago_palm'
            else:
                plantation_type = 'unknown'

            # New decision tree for emission factors where soil_block == 1
            if soil_block[row, col] == 1:
                # Start of emission factor decision tree
                if ecozone == 'boreal':
                    ef_co2_offsite = 0.12
                    if land_cover == forest_code:
                        if nutrient == 'poor':
                            ef_co2 = 0.25
                            ef_n2o = 0.22
                            ef_ch4_land = 7.0
                            ef_ch4_ditch = 217
                        elif nutrient == 'rich':
                            ef_co2 = 0.95
                            ef_n2o = 3.2
                            ef_ch4_land = 2.0
                            ef_ch4_ditch = 217
                        else:
                            ef_co2 = 0.0  # Handle unknown nutrient status
                            ef_n2o = 0.0
                            ef_ch4_land = 0.0
                            ef_ch4_ditch = 0.0
                    elif land_cover == grassland_code:
                        ef_co2 = 5.7
                        ef_n2o = 9.5
                        ef_ch4_land = 1.4
                        ef_ch4_ditch = 1165 # using deep default
                    elif land_cover == cropland_code:
                        ef_co2 = 7.9
                        ef_n2o = 13
                        ef_ch4_land = 0
                        ef_ch4_ditch = 1165
                    elif extraction > 0:
                        ef_co2 = 2.8
                        ef_n2o = 0.30
                        ef_ch4_land = 6.1
                        ef_ch4_ditch = 542
                    else:
                        ef_co2 = 0.0  # No emissions or default value
                        ef_n2o = 0.0
                        ef_ch4_land = 0.0
                        ef_ch4_ditch = 0.0
                elif ecozone == 'temperate':
                    ef_co2_offsite = 0.31
                    if land_cover == forest_code:
                        ef_co2 = 2.6
                        ef_n2o = 2.8
                        ef_ch4_land = 2.5
                        ef_ch4_ditch = 217
                    elif land_cover == grassland_code:
                        ef_ch4_ditch = 1165 # using deep default
                        if nutrient == 'poor':
                            ef_co2 = 5.3
                            ef_n2o = 4.3
                            ef_ch4_land = 1.8
                        elif nutrient == 'rich':
                            ef_co2 = 6.1
                            ef_n2o = 8.2
                            ef_ch4_land = 16 # using deep default
                        else:
                            ef_co2 = 0.0  # Handle unknown nutrient status
                            ef_n2o = 0.0
                            ef_ch4_land = 0.0
                    elif land_cover == cropland_code:
                        ef_co2 = 10.5
                        ef_n2o = 13
                        ef_ch4_land = 0
                        ef_ch4_ditch = 1165
                    elif extraction > 0:
                        ef_co2 = 3.0
                        ef_n2o = 0.3
                        ef_ch4_land = 6.1
                        ef_ch4_ditch = 542
                    else:
                        ef_co2 = 0.0  # No emissions or default value
                        ef_n2o = 0.0
                        ef_ch4_land = 0.0
                        ef_ch4_ditch = 0.0
                elif ecozone == 'tropical':
                    ef_ch4_ditch = 2259
                    ef_co2_offsite = 0.82
                    if planted_forest_type > 0:
                        if plantation_type == 'long_rotation':
                            ef_co2 = 15.0
                            ef_n2o = 2.4
                            ef_ch4_land = 2.7
                        elif plantation_type == 'short_rotation':
                            ef_co2 = 20.0
                            ef_n2o = 2.4
                            ef_ch4_land = 2.7
                        elif plantation_type == 'oil_palm':
                            ef_co2 = 11.0
                            ef_n2o = 1.2
                            ef_ch4_land = 0
                        elif plantation_type == 'sago_palm':
                            ef_co2 = 1.5
                            ef_n2o = 3.3
                            ef_ch4_land = 26.2
                        else:
                            ef_co2 = 0.0  # Handle unknown plantation type
                            ef_n2o = 0.0
                            ef_ch4_land = 0.0
                    elif land_cover == forest_code:
                        ef_co2 = 5.3
                        ef_n2o = 2.4
                        ef_ch4_land = 4.9
                    elif land_cover == grassland_code:
                        ef_co2 = 9.6
                        ef_n2o = 5.0
                        ef_ch4_land = 7.0
                    elif land_cover == cropland_code:
                        ef_co2 = 14.0
                        ef_n2o = 5.0
                        ef_ch4_land = 7.0
                    elif extraction > 0:
                        ef_co2 = 2.0
                        ef_n2o = 0
                        ef_ch4_land = 0
                    else:
                        ef_co2 = 0.0  # No emissions or default value
                        ef_n2o = 0.0
                        ef_ch4_land = 0.0
                else:
                    # todo insert message about uknown ecozone


                # Calculate emissions
                # Assuming each pixel represents 1 hectare
                area = 1.0  # Adjust if pixel area is different
                CO2_emissions_out[row, col] = ef_co2 * area
                N2O_emissions_out[row, col] = ef_n2o * area
                CH4_land_emissions_out[row, col] = ef_ch4_land * area
                CH4_ditch_emissions_out[row, col] = ef_ch4_ditch * area
                CO2_offsite_emissions_out[row, col] = ef_co2_offsite * area
            else:
                # No emissions for undrained peat or non-peat areas
                CO2_emissions_out[row, col] = 0.0
                N2O_emissions_out[row, col] = 0.0
                CH4_land_emissions_out[row, col] = 0.0
                CH4_ditch_emissions_out[row, col] = 0.0
                CO2_offsite_emissions_out[row, col] = 0.0

        # Add outputs to dictionaries
        out_dict_uint32["soil"] = soil_block
        out_dict_uint32["state"] = state_out
        out_dict_float32["CO2_emissions"] = CO2_emissions_out
        out_dict_float32["N2O_emissions"] = N2O_emissions_out
        out_dict_float32["CH4_land_emissions"] = CH4_land_emissions_out
        out_dict_float32["CH4_ditch_emissions"] = CH4_ditch_emissions_out
        out_dict_float32["CO2_offsite_emissions"] = CO2_offsite_emissions_out
        # Add additional emissions to the dictionaries if needed
        # e.g., out_dict_float32["other_gas_emissions"] = other_gas_emissions_out

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
    uint8_list = ['IPCC_basic_classes_2020', 'peat', 'planted_forest_type', 'extraction']
    int16_list = []  # Add layer names as needed
    int32_list = []  # Add layer names as needed
    float32_list = ['dadap', 'osm_roads', 'osm_canals', 'engert', 'grip']

    # Fill missing layers with NoData if necessary
    layers = uu.fill_missing_input_layers_with_no_data(
        layers, uint8_list, int16_list, int32_list, float32_list, bounds_str, tile_id, is_final, logger
    )

    # Verify that all required layers are present
    expected_layers = uint8_list + int16_list + int32_list + float32_list
    missing_layers = [layer for layer in expected_layers if layer not in layers]
    if missing_layers:
        logger.error(f"Missing layers after filling: {missing_layers}")
        return f"Failed due to missing layers: {missing_layers}", chunk_stats

    # Create typed dictionaries for Numba functions
    typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32 = nu.create_typed_dicts(layers)

    # Verify typed dictionaries have all required keys
    missing_uint8_keys = [key for key in uint8_list if key not in typed_dict_uint8]
    missing_float32_keys = [key for key in float32_list if key not in typed_dict_float32]
    if missing_uint8_keys or missing_float32_keys:
        logger.error(f"Typed dictionaries missing keys. uint8: {missing_uint8_keys}, float32: {missing_float32_keys}")
        return f"Failed due to missing keys in typed dictionaries", chunk_stats

    # Calculate statistics for input layers
    for key, array in layers.items():
        stats = uu.calculate_stats(array, key, bounds_str, tile_id, 'input_layer')
        chunk_stats.append(stats)

    # Run the drainage and emissions calculation
    lu.print_and_log(f"Calculating drainage and emissions in {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger)
    out_dict_uint32, out_dict_float32 = calculate_drainage_and_emissions(
        typed_dict_uint8, typed_dict_int16, typed_dict_float32
    )

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
    download_dict = {
        'IPCC_basic_classes_2020': f"{cn.land_cover_path}2020/raw/{{tile_id}}.tif",
        'peat': f"{cn.peat_extent_path}{{tile_id}}.tif",
        'planted_forest_type': f"{cn.planted_forest_type_path}{{tile_id}}.tif",
        'dadap': f"{cn.dadap_path}{{tile_id}}.tif",
        'osm_roads': f"{cn.osm_roads_path}{{tile_id}}.tif",
        'osm_canals': f"{cn.osm_canals_path}{{tile_id}}.tif",
        'engert': f"{cn.engert_path}{{tile_id}}.tif",
        'grip': f"{cn.grip_path}{{tile_id}}.tif",
        'extraction': f"{cn.extraction_path}{{tile_id}}.tif",
    }

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
            print("Can't print chunk stats: module 'src.scripts.utilities.constants_and_names' has no attribute 'chunk_stats_path'")

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
            bounding_box=[110, -10, 120, 0],  # Example bounding box

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
