# drainage_model.py

import argparse
import concurrent.futures
import dask
import numpy as np
import gc
import os

from dask.distributed import Client
from numba import jit, types
from numba.typed import Dict
from datetime import datetime

# Project imports
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu
from src.scripts.utilities import numba_utilities as nu

# Define constants for land cover codes
CROPLAND_CODE = cn.ipcc_codes['cropland']
SETTLEMENT_CODE = cn.ipcc_codes['settlement']

# Function to calculate drainage using Numba
@jit(nopython=True)
def calculate_drainage(in_dict_uint8, in_dict_int16, in_dict_float32):
    # Initialize output typed dictionaries
    out_dict_uint32 = Dict.empty(key_type=types.unicode_type, value_type=types.uint32[:,:])
    # Since we have no float32 outputs, we can omit out_dict_float32
    # out_dict_float32 = Dict.empty(key_type=types.unicode_type, value_type=types.float32[:,:])  # Only if needed

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
    descals_type_block = in_dict_int16['descals_type']

    # Initialize output arrays
    rows, cols = peat_block.shape
    soil_block = np.zeros((rows, cols), dtype=np.uint32)
    state_out = np.zeros((rows, cols), dtype=np.uint32)

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
            descals_type = descals_type_block[row, col]

            node = 0

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
                elif land_cover == CROPLAND_CODE or land_cover == SETTLEMENT_CODE:
                    node = nu.accrete_node(node, 3)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node
                elif planted_forest_type or descals_type > 0:
                    node = nu.accrete_node(node, 4)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node
                elif extraction > 0:
                    node = nu.accrete_node(node, 5)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node
                else:
                    node = nu.accrete_node(node, 6)
                    soil_block[row, col] = 0  # 'undrained'
                    state_out[row, col] = node
            else:
                soil_block[row, col] = 0  # 'undrained'
                node = nu.accrete_node(node, 2)
                state_out[row, col] = node

    # Add outputs to dictionaries
    out_dict_uint32["soil"] = soil_block
    out_dict_uint32["state"] = state_out

    return out_dict_uint32  # No float32 outputs in this case

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
    uint8_list = ['IPCC_basic_classes_2020', 'peat', 'planted_forest_type','extraction']
    int16_list = ['descals_type']  # Add layer names as needed
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

    # Run the drainage calculation
    lu.print_and_log(f"Calculating drainage in {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger)
    out_dict_uint32 = calculate_drainage(
        typed_dict_uint8, typed_dict_int16, typed_dict_float32
    )

    # Combine outputs into a single dictionary
    out_dict_all_dtypes = {**out_dict_uint32}

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

    [Usage examples and documentation omitted for brevity]
    """
    import sys
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Calculate drainage model.")
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

            # bounding_box=[112, -4, 114, -2], # one 2-degree chunk with data in Borneo 00N_110E
            # bounding_box=[110, -10, 120, 0], # 10x10 degree tile Borneo
            # bounding_box=[-74, -4, -72, -2],  # one 2-degree chunk with data Peru 00N_080W
            # bounding_box=[-80.0, -10.0, -70.0, 0.0],  # 10x10 degree tile peru 00N_080W
            # bounding_box=[16.0, 6.0, 18.0, 8.0],  # 2-degree chunk Congo 10N_010E
            # bounding_box=[-10.0, 0.0, 0.0, 10],  # 10x10 degree tile Congo 10N_010E
            bounding_box=[-8, 52, -6, 54, 2],  # 2-degree chunk Ireland 60N_010W
            # bounding_box=[-110.0, 50.0, -100.0, 60.0],  # 10x10 degree tile Ireland 60N_010W

            chunk_size=2,
            run_local=True,
            no_stats=False, #testing running this as false
            no_log=False, #testing running this as false
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


    """
    Main function to run the drainage model from the command line.

    This script calculates the drainage model using specified parameters.
    It can be run with default settings or customized via command-line arguments.

    **Usage Examples:**

    1. **Run with Default Settings (for Testing Purposes):**

        ```bash
        python drainage_model.py
        ```

        - **Description:** Executes the script using default parameters. This is useful for testing and ensuring that the script runs without any custom configurations.

    2. **Specify a Coiled Cluster and a Test AOI:**

        ```bash
        python -m src.scripts.models.drainage_model -cn drainage_cluster -bb 112 -4 114 -2 -cs 2
        ```

        - **Description:** Runs the drainage model for a specific Area of Interest (AOI) using a Coiled cluster.
        - **Parameters:**
            - `-cn drainage_cluster` or `--cluster_name drainage_cluster`: Specifies the name of the Coiled cluster to use for distributed computing. Replace `drainage_cluster` with your actual Coiled cluster name.
            - `-bb 112 -4 114 -2` or `--bounding_box 112 -4 114 -2`: Defines the bounding box for the AOI with west longitude `112°`, south latitude `-4°`, east longitude `114°`, and north latitude `-2°`.
            - `-cs 2` or `--chunk_size 2`: Sets the chunk size to `2` degrees. Adjust based on your computational resources and data size.

    3. **Run Locally Without Dask/Coiled:**

        ```bash
        python drainage_model.py --run_local
        ```

        - **Description:** Executes the script locally without leveraging Dask or Coiled for distributed computing. Useful for debugging or running on machines without access to a Dask cluster.

    4. **Exclude Statistics Spreadsheet:**

        ```bash
        python drainage_model.py --no_stats
        ```

        - **Description:** Runs the script without generating the chunk statistics spreadsheet. Use this option if you do not require statistical summaries of each data chunk.

    5. **Skip Log Creation:**

        ```bash
        python drainage_model.py --no_log
        ```

        - **Description:** Prevents the creation of the combined log file. Useful when logging is handled separately or is not needed.

    6. **Disable Uploading Outputs to S3:**

        ```bash
        python drainage_model.py --no_upload
        ```

        - **Description:** Runs the script without saving and uploading outputs to Amazon S3. Use this option if you prefer to handle output storage manually or store outputs locally.

    7. **Combine Multiple Options:**

        ```bash
        python -m src.scripts.models.drainage_model -cn drainage_cluster -bb 112 -4 114 -2 -cs 2 --no_stats --no_log --no_upload
        ```

        - **Description:** Executes the script with a combination of specified options:
            - Connects to the `drainage_cluster` Coiled cluster.
            - Processes data within the bounding box [112°W, 4°S, 114°E, 2°N].
            - Sets the chunk size to 2 degrees.
            - Skips generating the statistics spreadsheet.
            - Prevents log creation.
            - Does not upload outputs to S3.

    8. **Run for a Larger AOI with a Different Chunk Size:**

        ```bash
        python -m src.scripts.models.drainage_model -cn large_cluster -bb 110 -10 120 0 -cs 5
        ```

        - **Description:** Runs the script for a larger Area of Interest with a chunk size of 5 degrees.
        - **Parameters:**
            - `-cn large_cluster`: Uses the `large_cluster` Coiled cluster.
            - `-bb 110 -10 120 0`: Defines the bounding box from west longitude `110°`, south latitude `-10°`, east longitude `120°`, and north latitude `0°`.
            - `-cs 5`: Sets the chunk size to `5` degrees.

    **Arguments:**

    - `-cn`, `--cluster_name`:
        - **Type:** `str`
        - **Description:** Name of the Coiled cluster to use for distributed computing.

    - `-bb`, `--bounding_box`:
        - **Type:** `float` (expects four values)
        - **Description:** Defines the geographical bounding box for data processing as four float numbers representing [West, South, East, North] in degrees.

    - `-cs`, `--chunk_size`:
        - **Type:** `float`
        - **Description:** Specifies the size of each data chunk in degrees.

    - `--run_local`:
        - **Action:** `store_true`
        - **Description:** Runs the script locally without utilizing Dask or Coiled clusters.

    - `--no_stats`:
        - **Action:** `store_true`
        - **Description:** Prevents the creation of the chunk statistics spreadsheet.

    - `--no_log`:
        - **Action:** `store_true`
        - **Description:** Skips the creation of the combined log file.

    - `--no_upload`:
        - **Action:** `store_true`
        - **Description:** Does not save and upload output files to Amazon S3.

    **Default Behavior:**

    - If no command-line arguments are provided, the script uses default values for all parameters, which are suitable for testing purposes. For example, it processes a predefined bounding box and chunk size, runs locally, and creates all logs and statistics unless specified otherwise.

    **Notes:**

    - **Module Execution:** To successfully run the script and avoid `ModuleNotFoundError`, it's recommended to execute the script as a module using the `-m` flag from the project's root directory. This ensures that Python recognizes the `src` directory as a top-level package.

    - **Setting `PYTHONPATH`:** Alternatively, you can set the `PYTHONPATH` environment variable to include the project's root directory. This allows you to run the script directly without the `-m` flag.

        ```bash
        export PYTHONPATH=$(pwd)
        python src/scripts/models/drainage_model.py -cn drainage_cluster -bb 112 -4 114 -2 -cs 2
        ```

    - **Ensure `__init__.py` Files:** Make sure that each directory in the `src` path contains an `__init__.py` file. This file can be empty but is necessary for Python to recognize the directories as packages.

        ```bash
        touch src/__init__.py
        touch src/scripts/__init__.py
        touch src/scripts/models/__init__.py
        touch src/scripts/utilities/__init__.py
        ```

    - **Use Virtual Environments:** It's good practice to use virtual environments to manage dependencies.

        ```bash
        # Create a virtual environment
        python -m venv venv

        # Activate the virtual environment
        source venv/bin/activate

        # Install required packages
        pip install -r requirements.txt
        ```

    - **Check Dependencies:** Ensure all required packages (e.g., `dask`, `numba`, `numpy`, `pandas`) are installed in your environment.

        ```bash
        pip install dask numba numpy pandas
        ```

    **Example Scenario: Running for a Test AOI with a Coiled Cluster**

    Suppose you want to process a small test AOI within the bounding box [112°W, 4°S, 114°E, 2°N], using a chunk size of 2 degrees, and leverage a Coiled cluster named `drainage_cluster`. You also want to generate statistics and logs, and upload the outputs to S3. You would execute:

    ```bash
    python -m src.scripts.models.drainage_model -cn drainage_cluster -bb 112 -4 114 -2 -cs 2
    ```

    **Explanation:**

    - `-m src.scripts.models.drainage_model`: Runs the `drainage_model.py` script as a module within the `src.scripts.models` package.
    - `-cn drainage_cluster`: Connects to the `drainage_cluster` Coiled cluster for distributed computing.
    - `-bb 112 -4 114 -2`: Sets the bounding box for the AOI to west longitude `112°`, south latitude `-4°`, east longitude `114°`, and north latitude `-2°`.
    - `-cs 2`: Divides the AOI into chunks of 2 degrees each.
    - **Flags Not Included:** By omitting `--no_stats`, `--no_log`, and `--no_upload`, the script will generate the statistics spreadsheet, create logs, and upload outputs to S3 as per default behavior.

    **Another Example: Running a Test AOI Without Uploading Outputs**

    If you prefer to run the same AOI and cluster but want to skip uploading the outputs to S3, you would add the `--no_upload` flag:

    ```bash
    python -m src.scripts.models.drainage_model -cn drainage_cluster -bb 112 -4 114 -2 -cs 2 --no_upload
    ```

    **Explanation:**

    - All parameters are the same as the previous example.
    - `--no_upload`: Prevents the script from uploading the output data to Amazon S3. Useful for local testing or when manual upload is preferred.

    **Running Multiple Options Together:**

    For a comprehensive run that connects to a Coiled cluster, processes a specific AOI with a defined chunk size, and disables both statistics generation and log creation, you can combine multiple flags:

    ```bash
    python -m src.scripts.models.drainage_model -cn drainage_cluster -bb 112 -4 114 -2 -cs 2 --no_stats --no_log
    ```

    **Explanation:**

    - `-cn drainage_cluster`: Uses the `drainage_cluster` Coiled cluster.
    - `-bb 112 -4 114 -2`: Sets the AOI bounding box.
    - `-cs 2`: Sets the chunk size.
    - `--no_stats`: Skips generating the statistics spreadsheet.
    - `--no_log`: Skips creating the combined log file.
    - **Output Upload:** Since `--no_upload` is not specified, outputs will still be uploaded to S3.

    **Tips for Running the Script:**

    - **Verify Cluster Availability:**
        - Ensure that the specified Coiled cluster (`drainage_cluster` in the example) is active and accessible.
        - You can manage and monitor your Coiled clusters through the [Coiled Dashboard](https://app.coiled.io/) or via the Coiled CLI.

    - **Check Bounding Box Coordinates:**
        - Double-check the AOI coordinates to ensure they cover the intended geographical area.
        - Incorrect bounding box specifications can lead to processing unintended regions or empty datasets.

    - **Adjust Chunk Size Appropriately:**
        - Depending on the size of the AOI and available computational resources, choose a chunk size that balances performance and resource utilization.

    - **Monitor Logs and Outputs:**
        - If not disabling logs and statistics, regularly check these outputs to monitor the progress and performance of the script.
        - If you encounter errors, refer to the log files for detailed information.

    - **Handle Large AOIs with Care:**
        - Processing very large AOIs can be resource-intensive. Ensure that your Coiled cluster has sufficient resources (e.g., memory, CPU) to handle the workload.
        - Consider breaking down extremely large AOIs into smaller, manageable chunks if necessary.

    - **Customize Output Paths if Needed:**
        - By default, the script saves outputs to predefined directories. You can modify the script to change output paths based on your project structure or preferences.
        
        Some Testing tiles and bounds:
            bounding_box=[112, -4, 114, -2], # one 2-degree chunk with data in Borneo 00N_110E
            bounding_box=[-74, -4, -72, -2],  # one 2-degree chunk with data Peru 00N_080W
            bounding_box=[-80.0, -10.0, -70.0, 0.0],  # 10x10 degree tile peru 00N_080W
            bounding_box=[-10.0, 0.0, 0.0, 10],  # 10x10 degree tile Congo 10N_010E
            bounding_box=[16.0, 6.0, 18.0, 8.0],  # 2-degree chunk Congo 10N_010E
            bounding_box=[-8, 52, -6, 54, 2],  #  2-degree chunk Ireland 60N_010W
            bounding_box=[-110.0, 50.0, -100.0, 60.0],  # 10x10 degree tile Ireland 60N_010W
"""