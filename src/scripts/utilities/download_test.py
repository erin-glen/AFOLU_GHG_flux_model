"""Manual S3 download checker for organic-soils input paths.

This module is a command-line utility, not a pytest suite.
"""

__test__ = False

import boto3
from botocore.exceptions import ClientError
import rasterio
import sys
import os
import tempfile

from src.scripts.utilities import constants_and_names as cn


def check_s3_file_exists(s3_client, bucket, key):
    """
    Check if a specific file exists in an S3 bucket and retrieve its Content-Type.

    Args:
        s3_client (boto3.client): The boto3 S3 client.
        bucket (str): The name of the S3 bucket.
        key (str): The S3 object key/path.

    Returns:
        tuple: (bool, str or None) True and Content-Type if the file exists, False and None otherwise.
    """
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        content_type = response.get('ContentType', 'Unknown')
        return True, content_type
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False, None
        else:
            print(f"Error checking {bucket}/{key}: {e}")
            return False, None


def get_tiff_data_type(s3_client, bucket, key, download_dir):
    """
    Download a TIFF file from S3 and retrieve its data type.

    Args:
        s3_client (boto3.client): The boto3 S3 client.
        bucket (str): The name of the S3 bucket.
        key (str): The S3 object key/path.
        download_dir (str): The directory to download the file to.

    Returns:
        str: The data type of the raster data (e.g., 'uint8', 'int16', 'float32'), or 'Unknown' if unable to determine.
    """
    try:
        # Define the local path for the temporary file
        local_temp_path = os.path.join(download_dir, os.path.basename(key))

        # Download the file
        s3_client.download_file(bucket, key, local_temp_path)

        # Open the TIFF file with rasterio to get the data type
        with rasterio.open(local_temp_path) as dataset:
            dtype = dataset.dtypes[0]  # Assuming single-band raster; adjust if multi-band
            return dtype
    except Exception as e:
        print(f"Error retrieving data type for s3://{bucket}/{key}: {e}")
        return "Unknown"
    finally:
        # Clean up the temporary file if it exists
        if os.path.exists(local_temp_path):
            try:
                os.remove(local_temp_path)
            except Exception as e:
                print(f"Error deleting temporary file {local_temp_path}: {e}")


def test_download_dict_for_tile_id(tile_id):
    """
    Test the download_dict for a specific tile_id and print the existence and data type of each file.

    Args:
        tile_id (str): The tile ID to test.
    """
    s3_client = boto3.client('s3')

    # Use the s3_bucket_name from constants_and_names.py
    s3_bucket_name = cn.s3_bucket_name

    # Prepare the download_dict with the given tile_id
    # Replace '{tile_id}' with the actual tile_id in each path
    download_dict = {}
    for key, s3_path in cn.download_dict.items():
        s3_path_with_tile = s3_path.replace('{tile_id}', tile_id)
        download_dict[key] = s3_path_with_tile

    # Create a temporary directory to store downloaded files
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Remove 's3://' from the beginning to get the bucket and key
        for name, s3_path in download_dict.items():
            if s3_path.startswith("s3://"):
                s3_path_no_prefix = s3_path[5:]
            else:
                print(f"Invalid S3 path for {name}: {s3_path}")
                continue

            # Split into bucket and key
            bucket_and_key = s3_path_no_prefix.split('/', 1)
            if len(bucket_and_key) != 2:
                print(f"Invalid S3 path for {name}: {s3_path}")
                continue

            bucket, key = bucket_and_key
            file_exists, content_type = check_s3_file_exists(s3_client, bucket, key)
            if file_exists:
                # Retrieve data type
                data_type = get_tiff_data_type(s3_client, bucket, key, tmpdirname)
                print(f"{name}: File exists at s3://{bucket}/{key} with Data Type: {data_type}")
            else:
                print(f"{name}: File NOT found at s3://{bucket}/{key}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        tile_id = sys.argv[1]
    else:
        tile_id = '50N_130E'  # Default tile_id
        print(f"No tile_id provided. Using default tile_id: {tile_id}")
    test_download_dict_for_tile_id(tile_id)
