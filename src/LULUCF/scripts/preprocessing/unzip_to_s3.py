"""
This function downloads a zip file from s3, extracts the files, then uploads those files to s3.
File is set up to do this for the Bukowski planted forest C accumulation maps but can be updated to do this for any other process

Local test:
python -m src.LULUCF.scripts.preprocessing.unzip_to_s3 -p process --run_local

Coiled:
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn gpfc_C_accumulation
python -m src.LULUCF.scripts.preprocessing.unzip_to_s3 -cn gpfc_C_accumulation -p gpfc_C_accumulation
"""

import argparse
import os
from pathlib import Path
import sys
import shutil
import zipfile

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import universal_utilities as uu
from src.utilities import log_utilities as lu

def list_and_print_files_in_dir(local_dir):
    logger_worker = lu.setup_logging_worker()

    for root, dirs, files in os.walk(local_dir):
        for d in dirs:
            lu.print_and_log(f"[DIR ] {os.path.relpath(os.path.join(root, d), local_dir)}", False, logger_worker)
        for f in files:
            lu.print_and_log(f"[FILE] {os.path.relpath(os.path.join(root, f), local_dir)}", False, logger_worker)
#TODO: Move to uu

def main(cluster_name, zip_file, zip_s3_path, upload_s3_path, run_local=False):

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)
    client

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, f"Unzipping Bukowski et al C accumlation rates",
                                                                   run_local, 'standard', f'Bukowski C accumlation rates')

    # Prepare local paths
    local_zip_dir = "/tmp/zip/"
    local_extract_dir = "/tmp/unzipped/"
    os.makedirs(local_zip_dir, exist_ok=True)
    os.makedirs(local_extract_dir, exist_ok=True)
    local_zip = os.path.join(local_zip_dir, zip_file)

    main_logger.info(f"STEP 1: Downloading {zip_s3_path} -> {local_zip_dir}{zip_file}: {uu.timestr('time')}\n")
    uu.download_s3_file(zip_s3_path, local_zip_dir)
    list_and_print_files_in_dir(local_zip_dir)

    main_logger.info(f"STEP 2: Unzipping to {local_extract_dir}: - {uu.timestr('time')}\n")
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(local_extract_dir)
    list_and_print_files_in_dir(local_extract_dir)

    main_logger.info(f"STEP 3: Uploading extracted files to {upload_s3_path}: {uu.timestr('time')}\n")
    for root, dirs, files in os.walk(local_extract_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path   = os.path.relpath(local_path, local_extract_dir).replace("\\", "/")
            dest_s3    = f"{upload_s3_path}{rel_path}"

            # Upload to s3 and delete local files
            uu.upload_s3_file(dest_s3, local_path)
            if uu.check_s3_file_created(dest_s3, main_logger):
                try:
                    os.remove(local_path)
                    if not os.path.exists(local_path):
                        main_logger.info(f"Deleted {local_path}")
                except Exception as e:
                    main_logger.info(f"Error deleting {local_path} — {e}")

            main_logger.info(f"Successfully uploaded to {dest_s3} - {uu.timestr('time')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unzipping Bukowski et al C accumluation rates and upload to s3")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-p', '--process', choices=['gpfc_C_accumulation'])
    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    args = parser.parse_args()

    cluster_name = args.cluster_name
    run_local = args.run_local
    process = args.process
    
    if process == 'gpfc_C_accumulation':
        zip_file = "05_C_accumulation.zip"
        zip_s3_path = f"s3://gfw2-data/plantations/gpfc_C_accumulation/{zip_file}"
        upload_s3_path = "s3://gfw2-data/plantations/gpfc_C_accumulation/v1/"

    main(cluster_name, zip_file, zip_s3_path, upload_s3_path, run_local)
