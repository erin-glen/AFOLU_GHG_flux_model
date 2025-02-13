import boto3
import logging
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
def compile_and_upload_log(no_log, client, cluster, stage, chunk_count, chunk_size_deg,
                           start_time_str, end_time_1_str, end_time_2_str,
                           success_count, skipping_chunk_count, bounding_box, log_note):

    # Only consolidates the worker logs and uploads to s3 if not deactivated
    if no_log:
        return

    #TODO Create log folder if it doesn't exist already
    log_name = f"{cn.combined_log}_{stage}_{time.strftime('%Y%m%d_%H_%M_%S')}.txt"
    local_log = f"{cn.local_log_path}{log_name}"

    print(f"Preparing consolidated log {log_name}")

    # Recovers legs from Coiled
    logs = cluster.get_logs()

    # Converts the start and end times of the stage run from string to datetime.
    # Uses start_time to filter log entries to only those after the start_time
    start_time = datetime.strptime(start_time_str, "%Y%m%d_%H_%M_%S")
    end_time_1 = datetime.strptime(end_time_1_str, "%Y%m%d_%H_%M_%S")
    end_time_2 = datetime.strptime(end_time_2_str, "%Y%m%d_%H_%M_%S")

    # Retrieves properties of the workers
    workers = client.scheduler_info()["workers"]

    # Retrieves the number of workers
    n_workers = len(workers)

    # Retrieves the number of threads per worker
    # https://chatgpt.com/share/e/672503f1-eef8-800a-9218-281624acf27e
    first_worker_address = next(iter(workers.keys()))
    nthreads = workers[first_worker_address]["nthreads"]

    # Retrieves scheduler info for other cluster properties
    scheduler_info = cluster.scheduler_info  # Access scheduler info directly as a dictionary

    # Gets memory per worker.
    # Can't get it to report the worker instance type
    try:
        worker_memory_bytes = scheduler_info['workers'][next(iter(scheduler_info['workers']))]['memory_limit']
        worker_memory_gb = worker_memory_bytes / (1024 ** 3)  # Convert bytes to GB
        worker_memory = f"{worker_memory_gb:.2f} GB"  # Format to 2 decimal places
        # worker_type = coiled_cluster.config.get('worker_options', {}).get('instance_type', "Unknown")
    except KeyError:
        worker_memory = "Unknown"
        # worker_type = "Unknown"

    # Create header lines
    header_lines = [
        f"Stage: {stage}",
        f"Model version: {cn.model_version}",
        f"Number of workers: {n_workers}",
        f"Memory per worker: {worker_memory}",
        f"Threads per worker: {nthreads}",
        f"Bounding box: {bounding_box}",
        f"Number of chunks: {chunk_count}",
        f"Chunk size (degrees): {chunk_size_deg}",
        # f"Worker Type: {worker_type}",
        f"Log note: {log_note}",
        f"Starting time: {start_time_str}",
        "",
        "Filtered logs:",
        ""
    ]

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
            "\n".join(header_lines) + "\n" +
            "\n".join(filtered_logs) + "\n" +
            f"Stage {stage} ended at: {end_time_1_str}\n"
            f"Elapsed time for {stage}: {end_time_1 - start_time}\n"
            f"Number of 'Success' chunks: {success_count}\n"
            f"Number of 'Skipped' chunks: {skipping_chunk_count}\n"
            f"Difference between submitted chunks and processed chunks: {chunk_count - (success_count + skipping_chunk_count)}\n"
            f"Stage {stage} tile stats ended at: {end_time_2_str}\n"
            f"Elapsed time for {stage} including tile stats: {end_time_2 - start_time}"
    )

    # Save the filtered logs to a text file
    with open(local_log, "w") as file:
        file.write(combined_filtered_logs)

    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call
    s3_client.upload_file(local_log, "gfw2-data", Key=f"{cn.s3_log_path}{log_name}")

    print(f"Log uploaded to {cn.s3_log_path}{log_name}")


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

import sys

def setup_logging_main(log_filename=None):
    """Setup logging to log both to console and a file."""

    logger = logging.getLogger("flm_logger")  # Unified logger for both workers and main
    logger.setLevel(logging.INFO)

    # Ensure no duplicate handlers
    if not logger.hasHandlers():
        formatter = logging.Formatter('%(asctime)s - %(message)s')

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