# Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6794049f-2ba0-800a-b6ef-9445cfdd94a8

import os
import dask
import dask.bag as db
import boto3
import requests
from coiled import Cluster

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu
from ..utilities import log_utilities as lu
from ..utilities import numba_utilities as nu
from ..utilities import resize_cluster

# AWS S3 Configuration
S3_BUCKET = "gfw2-data"

# Base URL and year folders
BASE_URL = "https://glad.geog.umd.edu/Potapov/Global_TCH_2015-23"
# YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
YEARS = [2015]

def download_and_upload_file(file_url, s3_key):
    """
    Download a file from the given URL and upload it to the specified S3 location.
    """

    s3_client = boto3.client("s3")

    try:
        # Download the file
        response = requests.get(file_url, stream=True)
        response.raise_for_status()  # Raise an error for HTTP issues

        # Upload to S3
        s3_client.upload_fileobj(response.raw, S3_BUCKET, s3_key)
        print(f"Uploaded {file_url} to s3://{S3_BUCKET}/{s3_key}")
        return True
    except Exception as e:
        print(f"Error processing {file_url}: {e}")
        return False


def process_year(year):
    """
    Process all files for a specific year by downloading and uploading them to S3.
    """
    year_url = f"{BASE_URL}/TCH_filter_{year}/"
    response = requests.get(year_url)

    if response.status_code != 200:
        raise ValueError(f"Unable to access {year_url}")

    # Extract file names from the HTML page (assuming file links end with .tif)
    file_names = [line.split('"')[1] for line in response.text.splitlines() if '.tif' in line]

    # Create full file URLs and corresponding S3 keys
    tasks = []
    for file_name in file_names:
        file_url = f"{year_url}{file_name}"
        s3_key = f"landcover/vegetation_height/annual/20250114/{year}/{file_name}"  # Maintain year-based folder structure
        tasks.append((file_url, s3_key))

    return tasks


if __name__ == "__main__":

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster("AFOLU_flux_model_scripts", False)


    # Create tasks for all years
    all_tasks = []
    for year in YEARS:
        all_tasks.extend(process_year(year))

    # Create a Dask Bag for parallel processing
    bag = db.from_sequence(all_tasks, partition_size=10)
    results = bag.map(lambda x: download_and_upload_file(x[0], x[1])).compute()

    print("Transfer complete. Success:", sum(results), "Failures:", len(results) - sum(results))
