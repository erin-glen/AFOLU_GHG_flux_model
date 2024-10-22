import boto3
import logging
import time
import os

from dask.distributed import print
from datetime import datetime


from . import constants_and_names as cn
from . import universal_utilities as uu


# Log compilation and uploading
# From https://chatgpt.com/share/e/4fe1e9c8-05a0-4e9d-8eee-64168891b5e2
# Gets the logs for all workers
#TODO Wait to run this until all entries have been added to the Coiled log--
# running this right after the model finishes means that final log entries haven't made it into Coiled yet.
# Log compilation and uploading
# Log compilation and uploading
def compile_and_upload_log(no_log, client, cluster, stage, chunk_count, chunk_size_deg,
                           start_time_str, end_time_str, success_count, skipping_chunk_count, log_note):

    # Only consolidate and upload logs if logging is enabled
    if no_log:
        return

    log_name = f"{cn.combined_log}_{stage}_{time.strftime('%Y%m%d_%H_%M_%S')}.txt"
    local_log = os.path.join(cn.local_log_path, log_name)

    print(f"Preparing consolidated log {log_name}")

    # Retrieve logs from cluster or client
    if cluster is not None:
        try:
            logs = cluster.get_logs()
            print("Retrieved logs from cluster.")
        except Exception as e:
            print(f"Error retrieving logs from cluster: {e}")
            logs = {}
    elif client is not None:
        try:
            logs = client.get_worker_logs()
            print("Retrieved logs from client.")
        except Exception as e:
            print(f"Error retrieving logs from client: {e}")
            logs = {}
    else:
        print("No cluster or client provided. Skipping log retrieval.")
        logs = {}

    # Convert the start and end times of the stage run from string to datetime.
    try:
        start_time = datetime.strptime(start_time_str, "%Y%m%d_%H_%M_%S")
        end_time = datetime.strptime(end_time_str, "%Y%m%d_%H_%M_%S")
    except ValueError as ve:
        print(f"Error parsing start or end time: {ve}")
        return

    # Retrieve the number of workers
    try:
        if cluster is not None:
            scheduler_info = cluster.scheduler_info
        elif client is not None:
            scheduler_info = client.scheduler_info()
        else:
            scheduler_info = {}
        n_workers = len(scheduler_info.get('workers', {}))
    except Exception as e:
        print(f"Error retrieving scheduler info: {e}")
        n_workers = "Unknown"

    # Get memory per worker.
    try:
        if scheduler_info and 'workers' in scheduler_info:
            worker_memory_bytes = next(iter(scheduler_info['workers'].values())).get('memory_limit', None)
            if worker_memory_bytes:
                worker_memory_gb = worker_memory_bytes / (1024 ** 3)  # Convert bytes to GB
                worker_memory = f"{worker_memory_gb:.2f} GB"
            else:
                worker_memory = "Unknown"
        else:
            worker_memory = "Unknown"
    except Exception:
        worker_memory = "Unknown"

    # Create header lines
    header_lines = [
        f"Stage: {stage}",
        f"Model version: {cn.model_version}",
        f"Number of workers: {n_workers}",
        f"Memory per worker: {worker_memory}",
        f"Number of chunks: {chunk_count}",
        f"Chunk size (degrees): {chunk_size_deg}",
        f"Log note: {log_note}",
        f"Starting time: {start_time_str}",
        f"Ending time: {end_time_str}",
        "",
        "Filtered logs:",
        ""
    ]

    # Filter logs containing both 'distributed.worker' and 'flm',
    # and where the datetime is greater than start_time
    filtered_logs = []
    for worker_id, log_entries in logs.items():
        for log_entry in log_entries:
            # Determine if log_entry is a tuple or a string
            if isinstance(log_entry, tuple):
                # Assuming the tuple structure is (timestamp, message)
                # Adjust the index if your tuple structure is different
                if len(log_entry) >= 2:
                    message = log_entry[1]
                else:
                    # If tuple does not have enough elements, skip
                    continue
            elif isinstance(log_entry, str):
                message = log_entry
            else:
                # If log_entry is neither tuple nor string, skip
                continue

            # Split the message into lines
            for line in message.split('\n'):
                if 'distributed.worker' in line and 'flm' in line:
                    # Extract the datetime from the end of the log line
                    try:
                        # Assuming the datetime is the last element in the line
                        log_time_str = line.strip().split()[-1]
                        log_time = datetime.strptime(log_time_str, "%Y%m%d_%H_%M_%S")
                        # Include the line only if log_time is greater than start_time
                        if log_time > start_time:
                            filtered_logs.append(line)
                    except (ValueError, IndexError) as ve:
                        # If the datetime format is incorrect or not found, skip this line
                        continue

    # Create summary messages
    end_time_message = f"Stage ended at: {end_time_str}"
    stage_duration = f"Elapsed time for {stage}: {end_time - start_time}"
    success_chunk_message = f"Number of 'Success' chunks: {success_count}"
    skip_chunk_message = f"Number of 'Skipped' chunks: {skipping_chunk_count}"
    difference_message = f"Difference between submitted chunks and processed chunks: {chunk_count - (success_count + skipping_chunk_count)}"

    # Combine all parts into the final log content
    combined_filtered_logs = (
        "\n".join(header_lines) +
        "\n".join(filtered_logs) +
        "\n" + end_time_message +
        "\n" + stage_duration +
        "\n" + success_chunk_message +
        "\n" + skip_chunk_message +
        "\n" + difference_message
    )

    # Save the consolidated log to a local file
    try:
        with open(local_log, "w") as file:
            file.write(combined_filtered_logs)
        print(f"Consolidated log saved to {local_log}")
    except Exception as e:
        print(f"Error saving consolidated log: {e}")
        return

    # Upload the log file to S3
    try:
        s3_client = boto3.client("s3")
        s3_client.upload_file(local_log, "gfw2-data", f"{cn.s3_log_path}{log_name}")
        print(f"Log uploaded to {cn.s3_log_path}{log_name}")
    except Exception as e:
        print(f"Error uploading log to S3: {e}")

# Determines whether statement should be printed to the console as well as logged
def print_and_log(text, is_final, logger):

    logger.info(f"flm: {text}")
    if not is_final:
        print(f"flm: {text}")


# Configure logging for the distributed workers
# https://chatgpt.com/share/e/6f80ccde-6a85-4837-94a0-4fcf09b96e43
def setup_logging():
    logger = logging.getLogger('distributed.worker')
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger