"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Can only run on 1x1 degree chunks that do not have the run timestamp in the file name.
The way this builds the input file names, it can't handle filenames with the run timestamp.
It also can't handle chunks smaller than 1x1 degree.
Aggregates in batches because there are so many tasks for this step that it's just more likely to fail part way through
and so having batches allows easier restarting. The tasks are in a pre-determined alphabetical order,
so each batch should have the same contents for any given set of inputs.

Local test:
python -m src.LULUCF.scripts.core_model.3_aggregate_LULUCF_outputs_xr_version -yr 2000 2024 --first_folders_to_process 2 --first_10x10s_to_process 2 --input_date YYYYMMDD --run_local

Coiled small test:
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.core_model.3_aggregate_LULUCF_outputs_xr_version -cn LULUCF_postprocessing -yr 2000 2024 --first_folders_to_process 2 --first_10x10s_to_process 2 --input_date YYYYMMDD

Coiled large shapefile test (create a cluster with 1 worker, then resize it to 100 workers after local processing is done):
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.core_model.3_aggregate_LULUCF_outputs_xr_version -cn LULUCF_postprocessing -yr 2000 2024 --input_date YYYYMMDD -nw 100 -ln "This is the aggregation of the 1884-chunk run."

Full Coiled run (create a cluster with 1 worker, then resize it to 200 workers after local processing is done):
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.core_model.3_aggregate_LULUCF_outputs_xr_version -cn LULUCF_postprocessing -yr 2000 2024 --input_date YYYYMMDD -nw 200 -ln "This is intended to be the definitive global run."

Notes on optimizing threads/worker: https://app.asana.com/1/25496124013636/task/1206230383901961/comment/1210803828525318?focus=true
Tests of this aggregation and other aggregations show that 1 thread/worker with 4GB workers is low in Coiled credit usage
and runs quickly compared to other configurations.

14256 tiles in 273 folders for 1884-chunk output

Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68af555f-ee54-8320-956e-217eed17e61a?model=gpt-4o
"""

import argparse
import fsspec
import re
from collections import defaultdict
import os
import s3fs
import sys
import tempfile
import time
import xarray as xr
from dask import delayed, compute
from dask.distributed import Client, print

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import resize_cluster


@delayed
def merge_and_write(group_key, file_list, output_dir, folder_uri=None):

    logger_worker = lu.setup_logging_worker()

    task_start_time = time.time()

    # Obtains output file name components from sample input geotif
    sample_tif = file_list[0]
    sample_input_filename = sample_tif.split('/')[-1]
    tile_id = sample_input_filename[:8]

    output_layer_pattern = re.compile(r"version_\d+_\d+_\d+/([^/]+)/")
    match = output_layer_pattern.search(sample_tif)
    output_layer = match.group(1)

    date_pattern = re.compile(r"intervals/([^/]+)")
    match = date_pattern.search(sample_tif)
    layer_date = match.group(1)

    unit_pattern = re.compile(r"([^/]+)/4000")
    match = unit_pattern.search(sample_tif)
    layer_unit = match.group(1)

    out_file = f"{tile_id}__{output_layer}{layer_unit}_{layer_date}.tif"
    s3_output_path = f"{output_dir.rstrip('/')}/{out_file}"

    # if fs.exists(s3_output_path):
    #     print(f"Skipping {s3_output_path} (already exists)")
    #     return s3_output_path

    lu.print_and_log(f"Aggregating {folder_uri}, with output of {s3_output_path}: {uu.timestr()}", False, logger_worker)

    try:
        ds = xr.open_mfdataset(
            file_list,
            combine='by_coords',
            parallel=True,
            chunks={'x': 4000, 'y': 4000}
        ).squeeze()

        ds = ds.rio.write_crs("EPSG:4326")

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, out_file)
            ds.rio.to_raster(local_path, compress="LZW")

            with open(local_path, 'rb') as src_file:
                with fsspec.open(s3_output_path, 'wb') as dst_file:
                    dst_file.write(src_file.read())

        lu.print_and_log(f"Uploaded {s3_output_path}", False, logger_worker)

        task_end_time = time.time()
        lu.print_and_log(f"{out_file} took {round(task_end_time - task_start_time)} seconds: {uu.timestr()}", False, logger_worker)

        return s3_output_path

    except Exception as e:
        lu.print_and_log(f"Error processing tile {group_key} in {folder_uri}: {e}", False, logger_worker)
        return None


@delayed
def merge_folder(first_10x10s_to_process, folder_uri):

    logger_worker = lu.setup_logging_worker()

    # Lists all tifs in folder
    fs = s3fs.S3FileSystem(anon=False)
    files = fs.ls(folder_uri)
    tif_files = [f"s3://{f}" for f in files if f.endswith(".tif")]

    chunk_id_pattern = re.compile(r"__(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)__")

    tile_groups = defaultdict(list)

    # Groups all 1x1 deg tiles into 10x10 deg groupings
    for f in tif_files:
        match = chunk_id_pattern.search(f)
        if match:
            lat = int(match.group(1))
            lon = int(match.group(2))
            key = ((lat // 10) * 10, (lon // 10) * 10)
            tile_groups[key].append(f)

    # Applies how many 10x10s to process (for testing)
    if first_10x10s_to_process:
        tile_items = list(tile_groups.items())[:first_10x10s_to_process]
    else:
        tile_items = tile_groups.items()

    # Changes the output path to the appropriate number of pixels
    output_root = folder_uri.replace("4000_pixels", "40000_pixels")

    # For testing. Redirects outputs to a different version folder.
    output_root = output_root.replace(cn.model_version_underscore, f"{cn.model_version_underscore}_xr_aggreg")

    results = []
    for group_key, files in tile_items:
        lu.print_and_log(f"Listing {first_10x10s_to_process} tiles in {folder_uri}: {uu.timestr()}", False, logger_worker)
        result = merge_and_write(group_key, files, output_root, folder_uri=folder_uri)
        results.append(result)

    return results


def main(cluster_name, year_range, input_date, number_of_workers, run_local=False, no_stats=False, no_log=False, no_upload=False,
         first_folders_to_process=None, first_10x10s_to_process=None, log_note=None):


    ### Step 1: Preparation

    # Model stage being run
    stage = 'LULUCF_aggregation_to_10x10_deg'
    model_type = 'standard_model'

    # Runs chunks in batches of specified size.
    # Each batch slows down processing because chunks inevitably lag and that happens more the more batches there are.
    batch_size = 3000
    # batch_size = 15  # For testing batch processing
    # batch_size = 1  # For testing batch processing

    # Determines if arguments for start and end year are valid
    if year_range not in [[cn.first_model_year_5_years, cn.last_model_year_5_years],  # 2000-2020
                          [cn.first_model_year_5_years, cn.last_model_year_annual],  # 2000-2024
                          [cn.first_model_year_annual, cn.last_model_year_annual]]:  # 2015-2024
        print("Year range selection not valid")
        sys.exit()
    else:
        start_year = year_range[0]
        end_year = year_range[1]
        # print(f"Start year: {start_year}")
        # print(f"End year: {end_year}")

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    # Starting time for stage
    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Start year: {start_year}; end year: {end_year}")
    main_logger.info(f"Run date: {input_date}")
    main_logger.info(f"Batch size: {batch_size} chunks")
    main_logger.info(f"no_upload: {no_upload}")

    input_folders = [
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_2/AGC_emission_factor_CO2_only__fraction/standard_model/hybrid_intervals/2001_2005/4000_pixels/20250806/",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_2/AGC_emission_factor_CO2_only__fraction/standard_model/hybrid_intervals/2006_2010/4000_pixels/20250806/",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_2/gross_emissions__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2001_2005/_ha_yr/4000_pixels/20250806/",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_2/gross_emissions__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2006_2010/_ha_yr/4000_pixels/20250806/"
    ]


    # Limits folders to process (for testing)
    if first_folders_to_process:
        input_folders = input_folders[:first_folders_to_process]

    main_logger.info(f"input folders: {input_folders}")
    main_logger.info(f"Processing {len(input_folders)} folders: {uu.timestr()}")

    futures = [merge_folder(first_10x10s_to_process, folder)
               for folder in input_folders]

    folder_results = compute(*futures)

    # Flattens separate lists of 10x10 aggregations for each s3 folder into a single list that can be fully parallelized
    tile_tasks = [task for sublist in folder_results for task in sublist if task is not None]

    main_logger.info(f"Scheduling {len(tile_tasks)} aggregation tasks across {len(input_folders)}: {uu.timestr()}")
    tile_results = compute(*tile_tasks)

    uu.stage_duration(start_time, uu.timestr(), {stage}, main_logger)

    main_logger.info(f"Aggregation complete. {len(tile_results)} tiles written across {len(input_folders)} folders: {uu.timestr()}")

    # Sets it so that no worker logs are created if doing a local run
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats and worker log compilation", main_logger)

        # Adds the workers' logs to the main log and uploads to s3
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)

    # Closes the Dask client if not running locally
    if not run_local:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate 1x1 degree outputs from LULUCF model to 10x10 degree geotifs.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--input_date', help='Date of core model run, in YYYYMMDD')
    parser.add_argument('-yr', '--year_range', nargs=2, type=int, required=True, help='Starting and ending years for model. Start options: 2000, 2015. End options: 2020, 2024.')
    parser.add_argument('-ffol', '--first_folders_to_process', type=int, help='Number of folders to process from input list')
    parser.add_argument('-ften', '--first_10x10s_to_process', type=int, help='Number of chunks to process from input list')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')
    parser.add_argument('-nw', '--number_of_workers', type=int, help='Number of workers to rescale to after local input list processing is done. Optonal')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    input_date = args.input_date
    year_range = args.year_range
    first_folders_to_process = args.first_folders_to_process
    first_10x10s_to_process = args.first_10x10s_to_process
    log_note = args.log_note
    number_of_workers = args.number_of_workers

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    main(cluster_name, year_range, input_date, number_of_workers, run_local, no_stats, no_log, no_upload,
         first_folders_to_process=first_folders_to_process,
         first_10x10s_to_process=first_10x10s_to_process, log_note=log_note)