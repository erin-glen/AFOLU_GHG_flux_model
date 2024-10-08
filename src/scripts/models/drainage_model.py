# drainage_model.py

import argparse
import concurrent.futures
import dask
import numpy as np
import os
import sys
import logging
import boto3
import gc
from osgeo import gdal
from numba import jit

# Project imports
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu
from src.scripts.utilities import numba_utilities as nu

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
# Configure logging
logger = logging.getLogger(__name__)

def setup_aws_credentials():
    """
    Sets up AWS credentials for GDAL by retrieving them from a Boto3 session.
    Returns a dictionary of AWS credentials if successful, None otherwise.
    """
    try:
        # Create a Boto3 session
        session = boto3.Session()
        credentials = session.get_credentials()

        if credentials is None:
            logger.error("No AWS credentials found.")
            return None

        access_key = credentials.access_key
        secret_key = credentials.secret_key
        session_token = credentials.token  # Needed if using temporary credentials

        if not access_key or not secret_key:
            logger.error("Incomplete AWS credentials. Access Key or Secret Key is missing.")
            return None

        # Set GDAL configuration options
        gdal.SetConfigOption('AWS_ACCESS_KEY_ID', access_key)
        gdal.SetConfigOption('AWS_SECRET_ACCESS_KEY', secret_key)
        if session_token:
            gdal.SetConfigOption('AWS_SESSION_TOKEN', session_token)
            logger.info("Using temporary AWS credentials with session token.")

        logger.info("AWS credentials have been set for GDAL.")

        # Return credentials as a dictionary
        aws_credentials = {
            'AWS_ACCESS_KEY_ID': access_key,
            'AWS_SECRET_ACCESS_KEY': secret_key,
            'AWS_SESSION_TOKEN': session_token
        }
        return aws_credentials

    except Exception as e:
        logger.exception(f"Error setting up AWS credentials: {e}")
        return None

def check_s3_file_exists(s3_client, bucket, key):
    """
    Check if a specific file exists in an S3 bucket.

    Args:
        s3_client (boto3.client): The boto3 S3 client.
        bucket (str): The name of the S3 bucket.
        key (str): The S3 object key/path.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            logger.error(f"Error checking {bucket}/{key}: {e}")
            return False

def test_boto3_s3_access(bucket_name):
    """
    Tests access to AWS S3 using Boto3 to confirm permissions.
    Returns True if access is successful, False otherwise.
    """
    try:
        s3_client = boto3.client('s3')
        # Attempt to list objects in the specified bucket (or root)
        s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
        logger.info(f"Successfully accessed S3 bucket '{bucket_name}' using Boto3.")
        return True
    except NoCredentialsError:
        logger.error("No AWS credentials found by Boto3.")
        return False
    except ClientError as e:
        logger.error(f"Boto3 S3 access error: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error during Boto3 S3 access: {e}")
        return False

def open_dataset_with_gdal(s3_path):
    """
    Attempts to open the dataset at the given S3 path using GDAL.
    Returns the dataset if successful, None otherwise.
    """
    try:
        logger.info(f"Attempting to open dataset with GDAL: {s3_path}")
        dataset = gdal.Open(s3_path, gdal.GA_ReadOnly)

        if dataset is None:
            logger.error("Failed to open dataset with GDAL.")
            return None

        logger.info("Dataset opened successfully with GDAL.")
        return dataset

    except RuntimeError as e:
        logger.error(f"GDAL RuntimeError: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error during GDAL dataset open: {e}")
        return None

@jit(nopython=True)
def calculate_drainage(in_dict_uint8, in_dict_int16, in_dict_float32):
    # Initialize output dictionaries
    out_dict_uint32 = {}
    out_dict_float32 = {}

    # Extract required input arrays
    peat_block = in_dict_uint8[cn.file_patterns['peat']]
    land_cover_block = in_dict_uint8[f"{cn.file_patterns['land_cover']}_2020"]
    planted_forest_type_block = in_dict_uint8[cn.file_patterns['planted_forest_type_layer']]
    dadap_block = in_dict_float32[cn.file_patterns['dadap']]
    osm_roads_block = in_dict_float32[cn.file_patterns['osm_roads']]
    osm_canals_block = in_dict_float32[cn.file_patterns['osm_canals']]
    engert_block = in_dict_float32[cn.file_patterns['engert']]
    grip_block = in_dict_float32[cn.file_patterns['grip']]

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

            node = 0

            if peat == 1:
                node = nu.accrete_node(node, 1)
                if dadap > 0 or osm_canals > 0:
                    node = nu.accrete_node(node, 1)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node
                elif engert > 0 or grip > 0 or osm_roads > 0:
                    node = nu.accrete_node(node, 2)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node
                elif land_cover == cn.ipcc_codes['cropland'] or land_cover == cn.ipcc_codes['settlement']:
                    node = nu.accrete_node(node, 3)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node
                elif planted_forest_type > 0:
                    node = nu.accrete_node(node, 4)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node
                else:
                    node = nu.accrete_node(node, 5)
                    soil_block[row, col] = 0  # 'undrained'
                    state_out[row, col] = node
            else:
                soil_block[row, col] = 0  # 'undrained'
                node = nu.accrete_node(node, 2)
                state_out[row, col] = node

    # Add outputs to dictionaries
    out_dict_uint32["soil"] = soil_block
    out_dict_uint32["state"] = state_out

    return out_dict_uint32, out_dict_float32  # No float32 outputs in this case

def calculate_and_upload_drainage(bounds, download_dict_with_data_types, is_final, no_upload, aws_credentials):
    """
    Calculate drainage status and upload the results.

    Args:
        bounds (list): The geographic bounds of the chunk [W, S, E, N].
        download_dict_with_data_types (dict): Dictionary containing data paths with data types.
        is_final (bool): Flag indicating if this is the final run.
        no_upload (bool): Flag to prevent uploading outputs to S3.
        aws_credentials (dict): AWS credentials for GDAL.

    Returns:
        tuple: A message indicating success or failure, and a list of chunk statistics.
    """
    # Set up logging
    logger = lu.setup_logging()

    # Set GDAL configuration options in the worker process
    try:
        gdal.SetConfigOption('AWS_ACCESS_KEY_ID', aws_credentials['AWS_ACCESS_KEY_ID'])
        gdal.SetConfigOption('AWS_SECRET_ACCESS_KEY', aws_credentials['AWS_SECRET_ACCESS_KEY'])
        if aws_credentials['AWS_SESSION_TOKEN']:
            gdal.SetConfigOption('AWS_SESSION_TOKEN', aws_credentials['AWS_SESSION_TOKEN'])
            logger.info("Worker using temporary AWS credentials with session token.")
        logger.info("AWS credentials have been set for GDAL in worker process.")

        # Enable GDAL debug output (optional)
        # Uncomment the following lines if you need detailed GDAL logs
        # gdal.SetConfigOption('CPL_DEBUG', 'ON')
        # gdal.SetConfigOption('CPL_VSIL_CURL_VERBOSE', 'YES')
        # gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
        # gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif')
        # gdal.SetConfigOption('CPL_VSIL_CURL_NON_CACHED', 'YES')

    except Exception as e:
        logger.exception(f"Error setting AWS credentials in worker: {e}")
        raise

    # Get tile ID and chunk information
    bounds_str = uu.boundstr(bounds)
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # Using W and N coordinates for tile ID
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

    # Check for data presence in required layers
    required_layers = {
        f"{cn.file_patterns['land_cover']}_2020": layers.get(f"{cn.file_patterns['land_cover']}_2020"),
        cn.file_patterns['peat']: layers.get(cn.file_patterns['peat'])
    }
    data_in_chunk = uu.check_chunk_for_data(required_layers, bounds_str, tile_id, "all", is_final, logger)
    if not data_in_chunk:
        return f"Skipped chunk {bounds_str} due to lack of data: {uu.timestr()}", chunk_stats

    # Calculate statistics for input layers
    for key, array in layers.items():
        stats = uu.calculate_stats(array, key, bounds_str, tile_id, 'input_layer')
        chunk_stats.append(stats)

    # Create typed dictionaries for Numba functions
    typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32 = nu.create_typed_dicts(layers)

    # Run the drainage calculation
    lu.print_and_log(f"Calculating drainage in {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger)
    out_dict_uint32, out_dict_float32 = calculate_drainage(
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
    # Set up AWS credentials
    aws_credentials = setup_aws_credentials()
    if aws_credentials is None:
        logger.error("Failed to set up AWS credentials for GDAL. Exiting.")
        sys.exit(1)

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

    # Use download_dict from constants_and_names.py
    download_dict = cn.download_dict

    # Get first tile names and data types
    print(f"Getting tile_id of first tile in each tile set: {uu.timestr()}")
    first_tiles = uu.first_file_name_in_s3_folder(download_dict)

    print(f"Getting datatype of first tile in each tile set: {uu.timestr()}")
    download_dict_with_data_types = uu.add_file_type_to_dict(first_tiles)

    # Create delayed tasks for each chunk, passing aws_credentials
    print(f"Creating tasks and starting processing: {uu.timestr()}")
    delayed_results = [
        dask.delayed(calculate_and_upload_drainage)(
            chunk, download_dict_with_data_types, is_final, no_upload, aws_credentials
        ) for chunk in chunks
    ]

    # Compute tasks
    results = dask.compute(*delayed_results)

    # Process results
    success_count = 0
    skipping_chunk_count = 0

    for result in results:
        return_message, chunk_stats = result

        print(return_message)

        if "Success" in return_message:
            success_count += 1

        if "skipping chunk" in return_message.lower():
            skipping_chunk_count += 1

        if return_message:
            return_messages.append(return_message)

        if chunk_stats is not None:
            all_stats.extend(chunk_stats)

    # Print counts
    print(f"Number of 'Success' chunks: {success_count}")
    print(f"Number of 'skipping chunk' chunks: {skipping_chunk_count}")

    # Calculate stats if not suppressed
    if not no_stats:
        uu.calculate_chunk_stats(all_stats, stage)

    # End time
    end_time = uu.timestr()
    print(f"Stage {stage} ended at: {end_time}")
    uu.stage_duration(start_time, end_time, stage)

    # Compile and upload logs
    log_note = "Drainage model run"
    lu.compile_and_upload_log(
        no_log, client, cluster, stage, len(chunks), chunk_size,
        start_time, end_time, log_note
    )

    # Close the client and cluster if not running locally
    if not run_local:
        client.close()
        cluster.close()

def main(argv=None):
    """
    Main function to run the drainage model from the command line.

    This script calculates the drainage model using specified parameters.
    It can be run with default settings or customized via command-line arguments.

    Command-line Arguments:
        -cn, --cluster_name CLUSTER_NAME
            (str) Name of the Coiled cluster to use.
            Default: 'default_cluster'
            Example: -cn my_cluster_name

        -bb, --bounding_box W S E N
            (float, float, float, float) Define the geographic area to process.
            Coordinates are specified as four floating-point numbers representing
            West, South, East, and North in degrees.
            Default: [110, -10, 120, 0]
            Example: -bb 100 -5 105 5

        -cs, --chunk_size CHUNK_SIZE
            (float) Specify the size of each chunk in degrees.
            Default: 2
            Example: -cs 1

        --run_local
            (flag) Include this flag to run the script locally without using Dask/Coiled.
            Default: False
            Example: --run_local

        --no_stats
            (flag) Include this flag to prevent the creation of the chunk stats spreadsheet.
            Default: False
            Example: --no_stats

        --no_log
            (flag) Include this flag to prevent the creation of the combined log.
            Default: False
            Example: --no_log

        --no_upload
            (flag) Include this flag to prevent saving and uploading outputs to S3.
            Default: False
            Example: --no_upload

    Usage Examples:

    1. **Run with default parameters:**

       ```
       python drainage_model.py
       ```

       This uses all default values and runs locally.

    2. **Run with a specific cluster name:**

       ```
       python drainage_model.py -cn my_cluster_name
       ```

       Sets the cluster name to 'my_cluster_name' while using default values for other parameters.

    3. **Define a custom bounding box:**

       ```
       python drainage_model.py -bb 100 -5 105 5
       ```

       Sets the bounding box to:
       - West: 100 degrees
       - South: -5 degrees
       - East: 105 degrees
       - North: 5 degrees

    4. **Set a custom chunk size:**

       ```
       python drainage_model.py -cs 1
       ```

       Sets the chunk size to 1 degree.

    5. **Run locally without Dask/Coiled:**

       ```
       python drainage_model.py --run_local
       ```

       Tells the script to run locally without connecting to a Dask/Coiled cluster.

    6. **Prevent stats and log creation:**

       ```
       python drainage_model.py --no_stats --no_log
       ```

       Prevents the script from creating the chunk stats spreadsheet and the combined log.

    7. **Run without uploading outputs to S3:**

       ```
       python drainage_model.py --no_upload
       ```

       Prevents the script from saving and uploading outputs to S3.

    8. **Combine multiple options:**

       ```
       python drainage_model.py -cn my_cluster_name -bb 100 -5 105 5 -cs 1 --run_local --no_stats --no_log --no_upload
       ```

       This command:
       - Sets the cluster name to 'my_cluster_name'.
       - Uses a bounding box from 100°W to 105°E and -5°S to 5°N.
       - Sets the chunk size to 1 degree.
       - Runs the script locally without Dask/Coiled.
       - Prevents creation of stats and logs.
       - Prevents uploading outputs to S3.

    9. **View help message:**

       ```
       python drainage_model.py -h
       ```

       Displays the help message with information about all available options.

    Notes:
    - **Order of Arguments:** The order of arguments doesn't matter, but values must follow their respective flags.
    - **Default Values:** If you omit an argument, the script uses its default value as specified.
    - **Ensure Dependencies are Installed:** Make sure all required Python packages are installed in your environment.
    - **Check AWS Credentials:** If the script interacts with AWS services (like S3), ensure your AWS credentials are correctly configured.

    Troubleshooting:
    - **Argument Errors:** If you receive errors about arguments, double-check that you've provided the correct number and type of arguments.
    - **Script Errors:** Review any error messages provided by the script to identify issues.
    - **Help and Documentation:** Use the `-h` or `--help` flag to get information about the script's usage.
    """
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
            cluster_name='default_cluster',
            bounding_box=[110, -10, 120, 0],
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
