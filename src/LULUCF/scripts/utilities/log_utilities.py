import boto3
import logging
import sys
import time

from dask.distributed import print
from datetime import datetime

from . import constants_and_names as cn
from . import universal_utilities as uu


# Log compilation and uploading
# From https://chatgpt.com/share/e/4fe1e9c8-05a0-4e9d-8eee-64168891b5e2
# Gets the logs for all workers
#TODO Wait to run this until all entries have been added to the Coiled log--
# running this right after the model finishes means that final log entries haven't made it into Coiled yet.
def compile_and_upload_log(no_log, cluster, stage, start_time_str, logger):

    # Only consolidates the worker logs and uploads to s3 if not deactivated
    if no_log:
        return

    #TODO Create log folder if it doesn't exist already
    log_name = f"{cn.combined_log}_{stage}_{time.strftime('%Y%m%d_%H_%M_%S')}.txt"
    local_log = f"{cn.local_log_path}{log_name}"

    logger.info(f"Preparing consolidated log {log_name}")

    # Recovers legs from Coiled
    logs = cluster.get_logs()

    # Converts the start and end times of the stage run from string to datetime.
    # Uses start_time to filter log entries to only those after the start_time
    start_time = datetime.strptime(start_time_str, "%Y%m%d_%H_%M_%S")

    # Filter lines containing both 'distributed.worker' and 'flm',
    # and where the datetime is greater than start_time
    filtered_logs = []
    for worker_id, log in logs.items():
        for line in log.split('\n'):
            if 'distributed.worker' in line and 'flm' in line:
                # Extract the datetime from the end of the log line
                log_time_str = line.split()[-1]
                try:
                    log_time = datetime.strptime(log_time_str, "%Y%m%d_%H_%M_%S")
                    # Include the line only if log_time is greater than start_time
                    if log_time > start_time:
                        filtered_logs.append(line)
                except ValueError:
                    # If the datetime format is incorrect, skip this line
                    continue

    combined_filtered_logs = (
            "\n".join(filtered_logs) + "\n"
    )

    # Save the filtered logs to a text file
    with open(local_log, "w") as file:
        file.write(combined_filtered_logs)

    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call
    s3_client.upload_file(local_log, "gfw2-data", Key=f"{cn.s3_log_path}{log_name}")

    logger.info(f"Log uploaded to {cn.s3_log_path}{log_name}")


# Determines whether statement should be printed to the console as well as logged
def print_and_log(text, is_final, logger):

    logger.info(f"flm: {text}")
    if not is_final:
        print(f"flm: {text}")


# Configure logging for the distributed workers
# https://chatgpt.com/share/e/6f80ccde-6a85-4837-94a0-4fcf09b96e43
def setup_logging_worker():
    logger = logging.getLogger('distributed.worker')
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def setup_logging_main(log_filename=None):
    """Setup logging to log both to console and a file."""

    logger = logging.getLogger("flm_logger")  # Unified logger for both workers and main
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