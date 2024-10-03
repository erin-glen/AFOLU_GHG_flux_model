# test_download_dict.py

import boto3
from botocore.exceptions import ClientError

# Import the download_dict from constants_and_names.py
import sys
import os

# Add the path to the directory containing constants_and_names.py
# Adjust the path as necessary if it's in a different directory
sys.path.append(os.path.abspath('path_to_your_project_directory'))

import constants_and_names as cn

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
            print(f"Error checking {bucket}/{key}: {e}")
            return False

def test_download_dict_for_tile_id(tile_id):
    """
    Test the download_dict for a specific tile_id.

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
        file_exists = check_s3_file_exists(s3_client, bucket, key)
        if file_exists:
            print(f"{name}: File exists at s3://{bucket}/{key}")
        else:
            print(f"{name}: File NOT found at s3://{bucket}/{key}")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        tile_id = sys.argv[1]
    else:
        tile_id = '00N_110E'  # Default tile_id
        print(f"No tile_id provided. Using default tile_id: {tile_id}")
    test_download_dict_for_tile_id(tile_id)