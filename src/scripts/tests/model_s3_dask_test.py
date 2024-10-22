# model_s3_dask_test.py

import sys
import os
import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from osgeo import gdal
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
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


def open_dataset_with_gdal(s3_path, retries=3, delay=5):
    """
    Attempts to open the dataset at the given S3 path using GDAL with retry logic.
    Returns True if successful, False otherwise.

    Args:
        s3_path (str): The /vsis3/ path to the raster file on S3.
        retries (int): Number of retry attempts.
        delay (int): Delay in seconds between retries.

    Returns:
        bool: True if GDAL can open the dataset, False otherwise.
    """
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Attempt {attempt}: Opening dataset with GDAL: {s3_path}")
            dataset = gdal.Open(s3_path, gdal.GA_ReadOnly)

            if dataset is None:
                logger.error(f"Attempt {attempt}: Failed to open dataset with GDAL.")
            else:
                logger.info(f"Attempt {attempt}: Dataset opened successfully with GDAL.")

                # Retrieve and log basic information
                driver = dataset.GetDriver().ShortName + "/" + dataset.GetDriver().LongName
                size = f"{dataset.RasterXSize} x {dataset.RasterYSize} x {dataset.RasterCount}"
                projection = dataset.GetProjectionRef()
                geotransform = dataset.GetGeoTransform()
                origin = f"({geotransform[0]}, {geotransform[3]})" if geotransform else "N/A"
                pixel_size = f"({geotransform[1]}, {geotransform[5]})" if geotransform else "N/A"

                logger.info(f"Driver: {driver}")
                logger.info(f"Size: {size}")
                logger.info(f"Projection: {projection}")
                logger.info(f"Origin: {origin}")
                logger.info(f"Pixel Size: {pixel_size}")

                # Optionally, read a small data block
                band = dataset.GetRasterBand(1)
                data = band.ReadAsArray(0, 0, 10, 10)  # Read a 10x10 block
                if data is not None:
                    logger.info(f"Read data block shape: {data.shape}")
                else:
                    logger.warning("No data read from the dataset.")

                return True

        except RuntimeError as e:
            logger.error(f"Attempt {attempt}: GDAL RuntimeError: {e}")
        except Exception as e:
            logger.exception(f"Attempt {attempt}: Unexpected error during GDAL dataset open: {e}")

        if attempt < retries:
            logger.info(f"Attempt {attempt}: Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            logger.error(f"All {retries} attempts failed to open dataset: {s3_path}")

    return False


def load_download_dict(project_dir):
    """
    Loads the download_dict and s3_bucket_name from constants_and_names.py.

    Args:
        project_dir (str): The absolute path to the project directory containing constants_and_names.py.

    Returns:
        tuple: (download_dict, s3_bucket_name) if successful, exits otherwise.
    """
    try:
        sys.path.append(project_dir)
        import constants_and_names as cn
    except ImportError as e:
        logger.exception(f"Error importing constants_and_names.py: {e}")
        sys.exit(1)

    download_dict = getattr(cn, 'download_dict', None)
    s3_bucket_name = getattr(cn, 's3_bucket_name', None)

    if not download_dict:
        logger.error("download_dict not found in constants_and_names.py.")
        sys.exit(1)
    if not s3_bucket_name:
        logger.error("s3_bucket_name not found in constants_and_names.py.")
        sys.exit(1)

    return download_dict, s3_bucket_name


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


def test_download_dict_for_tile_id(tile_id, download_dict, s3_bucket_name, max_workers=4):
    """
    Test the download_dict for a specific tile_id by checking existence and GDAL access.

    Args:
        tile_id (str): The tile ID to test.
        download_dict (dict): Dictionary mapping layer names to S3 paths with {tile_id} placeholders.
        s3_bucket_name (str): The name of the S3 bucket.
        max_workers (int): Maximum number of worker threads for concurrency.
    """
    s3_client = boto3.client('s3')
    logger.info(f"Starting tests for tile_id: {tile_id}")

    # Prepare the download_dict with the given tile_id
    download_dict_with_tile = {}
    for key, s3_path in download_dict.items():
        s3_path_with_tile = s3_path.replace('{tile_id}', tile_id)
        download_dict_with_tile[key] = s3_path_with_tile

    # Function to process each file
    def process_file(name, s3_path):
        result = {'name': name, 'exists': False, 'gdal_access': False}
        logger.info(f"Processing {name}: {s3_path}")

        # Validate S3 path format
        if not s3_path.startswith("s3://"):
            logger.warning(f"Invalid S3 path format for {name}: {s3_path}")
            return result

        # Extract bucket and key
        s3_path_no_prefix = s3_path[5:]
        bucket_key = s3_path_no_prefix.split('/', 1)
        if len(bucket_key) != 2:
            logger.warning(f"Invalid S3 path format for {name}: {s3_path}")
            return result

        bucket, key = bucket_key

        # Check if the file exists
        file_exists = check_s3_file_exists(s3_client, bucket, key)
        result['exists'] = file_exists
        if file_exists:
            logger.info(f"{name}: File exists at s3://{bucket}/{key}")
            # Attempt to open with GDAL
            vsis3_path = f"/vsis3/{bucket}/{key}"
            gdal_success = open_dataset_with_gdal(vsis3_path)
            result['gdal_access'] = gdal_success
            if gdal_success:
                logger.info(f"{name}: GDAL successfully accessed the file.")
            else:
                logger.error(f"{name}: GDAL failed to access the file.")
        else:
            logger.error(f"{name}: File NOT found at s3://{bucket}/{key}")

        return result

    # Use ThreadPoolExecutor for concurrency
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_name = {
            executor.submit(process_file, name, s3_path): name
            for name, s3_path in download_dict_with_tile.items()
        }

        # Process completed futures
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                data = future.result()
                if not data['exists']:
                    logger.warning(f"{name}: File does not exist.")
                elif not data['gdal_access']:
                    logger.warning(f"{name}: GDAL could not access the file.")
                else:
                    logger.info(f"{name}: Passed all tests.")
            except Exception as e:
                logger.exception(f"Exception occurred while processing {name}: {e}")

    logger.info(f"Completed tests for tile_id: {tile_id}")


def main():
    """
    Main function to execute the test script.
    """
    parser = argparse.ArgumentParser(description="Test S3 file existence and GDAL access for a given tile_id.")
    parser.add_argument(
        '-td', '--tile_id',
        type=str,
        default='00N_110E',
        help='Tile ID to test (default: 00N_110E)'
    )
    parser.add_argument(
        '-pd', '--project_dir',
        type=str,
        default=r"C:\GIS\git\AFOLU_GHG_flux_model",
        help='Absolute path to the project directory containing constants_and_names.py (default: C:\\GIS\\git\\AFOLU_GHG_flux_model)'
    )
    parser.add_argument(
        '-mw', '--max_workers',
        type=int,
        default=4,
        help='Maximum number of worker threads for concurrency (default: 4)'
    )
    args = parser.parse_args()

    tile_id = args.tile_id
    project_dir = args.project_dir
    max_workers = args.max_workers

    # Load download_dict and s3_bucket_name
    download_dict, s3_bucket_name = load_download_dict(project_dir)

    # Test Boto3 S3 access to the bucket
    logger.info(f"Testing Boto3 access to bucket: {s3_bucket_name}")
    if not test_boto3_s3_access(s3_bucket_name):
        logger.error("Boto3 S3 access test failed. Exiting.")
        sys.exit(1)

    # Set up AWS credentials for GDAL
    aws_credentials = setup_aws_credentials()
    if aws_credentials is None:
        logger.error("Failed to set up AWS credentials for GDAL. Exiting.")
        sys.exit(1)

    # Run the download and GDAL access tests
    test_download_dict_for_tile_id(tile_id, download_dict, s3_bucket_name, max_workers=max_workers)


if __name__ == "__main__":
    main()
