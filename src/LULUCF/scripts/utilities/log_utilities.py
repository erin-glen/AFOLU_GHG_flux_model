import boto3
import logging
import os
import sys
import time

from dask.distributed import print
from datetime import datetime

from . import constants_and_names as cn
from . import universal_utilities as uu

# Log for main function
# per https://chatgpt.com/share/e/67ae4ae8-ff64-800a-8b75-484d388e6a43
def setup_logging_main(log_filename=None):
    """Set up logging to log both to console and a file."""

    logger = logging.getLogger("flm_logger")
    logger.setLevel(logging.INFO)

    # Ensure no duplicate handlers
    if not logger.hasHandlers():
        formatter = logging.Formatter('flm: %(message)s')

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler if filename is provided
        if log_filename:
            file_handler = logging.FileHandler(log_filename)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


# Populates the log for the main function with various header and run information
def populate_main_log_header(client, cluster, log_note, run_local, model_type, stage):

    main_log_name = f"{cn.combined_log}_main_{stage}_{time.strftime('%Y%m%d_%H_%M_%S')}.log"
    main_log_local_path = f"{cn.local_log_path}{main_log_name}"
    os.makedirs("logs", exist_ok=True)  # Ensures logs directory exists
    main_logger = setup_logging_main(main_log_local_path)  # Sets up logging for the main function

    if run_local:
        worker_memory = "N/A- local run"
        n_workers = "N/A- local run"
        nthreads = "N/A- local run"
    else:
        worker_memory, n_workers, nthreads = uu.get_cluster_info(client, cluster)

    main_logger.info(f"Model type: {model_type}")
    main_logger.info(f"Stage: {stage}")
    main_logger.info(f"Model version: {cn.model_version}")
    main_logger.info(f"Number of workers: {n_workers}")
    main_logger.info(f"Memory per worker: {worker_memory}")
    main_logger.info(f"Threads per worker: {nthreads}")
    main_logger.info(f"Log note: {log_note}")

    return main_logger, main_log_local_path

# Configure logging for the distributed workers
# https://chatgpt.com/share/e/6f80ccde-6a85-4837-94a0-4fcf09b96e43
def setup_logging_worker():
    logger = logging.getLogger('distributed.worker')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Apply formatter to existing handlers (Dask may have already added handlers)
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    # Ensure there's at least one handler
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Determines whether statement should be printed to the console as well as logged
def print_and_log(text, is_final, logger):
    logger.info(f"flm: {text}")
    if not is_final:
        print(text, flush=True)


# Log compilation and uploading
# From https://chatgpt.com/share/e/4fe1e9c8-05a0-4e9d-8eee-64168891b5e2
# Gets the logs for all workers
def compile_worker_logs(no_log, cluster, stage, start_time_str, logger):

    # Only consolidates the worker logs and uploads to s3 if not deactivated
    if no_log:
        return

    worker_log_name = f"{cn.combined_log}_workers_{stage}_{time.strftime('%Y%m%d_%H_%M_%S')}.log"
    worker_log_local_path = f"{cn.local_log_path}{worker_log_name}"

    logger.info(f"Preparing consolidated log {worker_log_name}")

    # Recovers legs from Coiled
    logs = cluster.get_logs()

    # Filters lines containing 'flm',
    filtered_logs = []
    for worker_id, log in logs.items():
        for line in log.split('\n'):
            if 'flm' in line:
                filtered_logs.append(line)


    combined_filtered_logs = (
            "\n".join(filtered_logs) + "\n"
    )

    # Saves the filtered logs to a text file
    with open(worker_log_local_path, "w") as file:
        file.write(combined_filtered_logs)

    return worker_log_local_path


# Merges the log from main() with all the worker logs after all processing and uploads to s3
def merge_main_and_worker_upload_logs(no_log, main_log, worker_log, stage):

    combined_log_name = f"{cn.combined_log}_combined_{stage}_{time.strftime('%Y%m%d_%H_%M_%S')}.log"
    combined_local_log = f"{cn.local_log_path}{combined_log_name}"

    # Open the output file in write mode
    with open(combined_local_log, "w") as outfile:
        with open(main_log, "r") as infile1:
            outfile.write(infile1.read())
            outfile.write("\n")  # Adds blank line between files

            # Only adds worker logs if no_log not selected
            if not no_log:

                with open(worker_log, "r") as infile2:
                    outfile.write(infile2.read())

        print(f"Combined log saved as {combined_local_log}")  # Does not go in the log because it's closed

    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call
    s3_client.upload_file(combined_local_log, "gfw2-data", Key=f"{cn.s3_log_path}{combined_log_name}")

    os.remove(main_log)

    if not no_log:
        os.remove(worker_log)