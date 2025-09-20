"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Can only run on 1x1 degree chunks that do not have the run timestamp in the file name.
The way this builds the input file names, it can't handle filenames with the run timestamp.
It also can't handle chunks smaller than 1x1 degree.
Aggregates in batches because there are so many tasks for this step that it's just more likely to fail part way through
and so having batches allows easier restarting. The tasks are in a pre-determined alphabetical order,
so each batch should have the same contents for any given set of inputs.

Local test:
python -m src.LULUCF.scripts.vegetation_model.3_aggregate_LULUCF_outputs_xr_version -yr 2000 2024 --first_folders_to_process 2 --first_10x10s_to_process 2 --input_date YYYYMMDD --run_local

Coiled small test:
python -m src.utilities.create_cluster -n 2 -m 32 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.vegetation_model.3_aggregate_LULUCF_outputs_xr_version -cn LULUCF_postprocessing -yr 2000 2024 --first_folders_to_process 2 --first_10x10s_to_process 2 --input_date YYYYMMDD

Coiled large shapefile test (create a cluster with 1 worker, then resize it to 100 workers after local processing is done):
python -m src.utilities.create_cluster -n 2 -m 32 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.vegetation_model.3_aggregate_LULUCF_outputs_xr_version -cn LULUCF_postprocessing -yr 2000 2024 --input_date YYYYMMDD -nw 100 -ln "This is the aggregation of the 1884-chunk run."

Full Coiled run (create a cluster with 1 worker, then resize it to 200 workers after local processing is done):
python -m src.utilities.create_cluster -n 2 -m 32 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.vegetation_model.3_aggregate_LULUCF_outputs_xr_version -cn LULUCF_postprocessing -yr 2000 2024 --input_date YYYYMMDD -nw 200 -ln "This is intended to be the definitive global run."

Notes on optimizing memory/worker and threads/worker: https://app.asana.com/1/25496124013636/task/1206230383901961/comment/1211174775511287?focus=true
Tests showed that 16GB/worker is simply not large enough and that not specifying the number of threads/worker is faster.
Hence, no -t argument.

Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68af555f-ee54-8320-956e-217eed17e61a?model=gpt-4o
"""

import argparse
import fsspec
import re
from collections import defaultdict
import numpy as np
import os
import s3fs
import sys
import tempfile
import time
import xarray as xr
from dask import delayed, compute
from dask.distributed import Client, print
from osgeo import gdal

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

    layer_pattern = re.search(r"version_\d+_\d+_\d+/([^/]+)/", sample_tif).group(1)
    # print(layer_pattern)

    layer_date = re.search(r"intervals/([^/]+)", sample_tif).group(1)
    # print(layer_date)

    layer_unit = re.search(r"([^/]+)/4000", sample_tif).group(1)
    # print(layer_unit)

    # For emission factor and some other outputs, there is no specific layer_unit part of the output path;
    # it is reported as the layer_date.
    # So, if layer_unit = layer_date, we don't use the layer_unit (since it winds up being the same as the date).
    if layer_unit == layer_date:
        out_file = f"{tile_id}__{layer_pattern}_{layer_date}.tif"
    else:
        out_file = f"{tile_id}__{layer_pattern}{layer_unit}_{layer_date}.tif"
    s3_output_path = f"{output_dir.rstrip('/')}/{out_file}"

    # if fs.exists(s3_output_path):
    #     print(f"Skipping {s3_output_path} (already exists)")
    #     return s3_output_path

    lu.print_and_log(f"Aggregating {folder_uri}, with output of {s3_output_path}: {uu.timestr()}", False, logger_worker)

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

        lu.print_and_log(f"Uploaded {s3_output_path}: {uu.timestr()}", False, logger_worker)

        ### Counts valid pixels in the output raster
        lu.print_and_log(f"Counting pixels in {tile_id} for {out_file}: {uu.timestr()}", False, logger_worker)

        try:
            ds = gdal.Open(local_path)
            if ds is not None:
                band = ds.GetRasterBand(1)
                valid_pixel_count = 0

                # Get raster dimensions
                x_size = band.XSize
                y_size = band.YSize

                # Read in chunks
                block_size_x, block_size_y = band.GetBlockSize()

                for y in range(0, y_size, block_size_y):
                    rows_to_read = min(block_size_y, y_size - y)
                    for x in range(0, x_size, block_size_x):
                        cols_to_read = min(block_size_x, x_size - x)

                        block = band.ReadAsArray(x, y, cols_to_read, rows_to_read)

                        if block is not None:
                            valid_pixel_count += np.count_nonzero(block != 0)

                ds = None
            else:
                valid_pixel_count = -1
        except Exception as e:
            lu.print_and_log(f"Error counting pixels for {local_path}: {e}", False, logger_worker)
            print(f"Error counting pixels for {local_path}: {e}")
            return f"failure counting pixels for {out_file}"

    task_end_time = time.time()
    lu.print_and_log(f"{out_file} took {round(task_end_time - task_start_time)} seconds: {uu.timestr()}", False, logger_worker)


    # Most stats for the 10x10 aren't calculated.
    # Only the pixel count is because it is compared to the pixel counts in all the relevant 1x1s.
    # Dictionary is in a list because it's necessary for chunk stats processing later.
    chunk_stats = [{
        'chunk_id': 'N/A',
        'tile_id': tile_id,
        'layer_name': out_file,
        'tile_name': out_file,
        'in_out': 'output_layer',
        'pattern': layer_pattern,
        'years': layer_date,
        'min_value': 'no data',
        'mean_value': 'no data',
        'max_value': 'no data',
        'count_value': valid_pixel_count,
        'sum_value': 'no data',
        'data_type': 'no data'
    }]

    return f"Success merging {out_file}", chunk_stats


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
    if first_10x10s_to_process is not None:
        tile_items = list(tile_groups.items())[:first_10x10s_to_process]
    else:
        tile_items = tile_groups.items()
        first_10x10s_to_process = "all"

    # Changes the output path to the appropriate number of pixels
    output_root = folder_uri.replace("4000_pixels", "40000_pixels")

    #TODO For testing. Redirects outputs to a different version folder.
    output_root = output_root.replace(cn.model_version_underscore, f"{cn.model_version_underscore}_xr_aggreg")

    results = []
    for group_key, files in tile_items:
        # lu.print_and_log(f"Listing {first_10x10s_to_process} tiles in {folder_uri}: {uu.timestr()}", False, logger_worker)
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
    batch_size = 100
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

    # Calculates the interval type, difference between start and end years of intervals,
    # and the model output years for the model run
    interval_type, interval_year_diff, interval_length, interval_end_years = uu.get_interval_info(end_year, main_logger, start_year)

    # # Testing list with a variety of inputs: no unit_type, date_range, date
    # output_dir_list = [
    #     "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_2/AGC_emission_factor_CO2_only__fraction/standard_model/hybrid_intervals/2001_2005/4000_pixels/20250806/",
    #     "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_2/AGC_emission_factor_CO2_only__fraction/standard_model/hybrid_intervals/2006_2010/4000_pixels/20250806/",
    #     "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_2/gross_emissions__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2001_2005/_ha_yr/4000_pixels/20250806/",
    #     "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_2/gross_emissions__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2006_2010/_ha_yr/4000_pixels/20250806/",
    #     "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_2/carbon_density__non_soil__MgC/standard_model/hybrid_intervals/2005/_ha/4000_pixels/20250806/",
    #     "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_2/carbon_density__non_soil__MgC/standard_model/hybrid_intervals/2010/_ha/4000_pixels/20250806/"
    # ]

    # Unlike numba-based scripts, this one doesn't construct the download dictionary in the main function.
    # Instead, it creates a list of input folders, from which a download dictionary is created for each chunk (in the chunk-level function).
    # It's a little simpler this way. Since the datatypes of the inputs don't need to be specified in advance for this script
    # (since it's not using numba), there's no need to centrally create a download dictionary with each input's datatype
    # just once on the scheduler, as is more efficient for scripts that use numba.
    # Creates a list of input directories used in summative output creation based on specifics of the model run

    # Only make 10x10s of the summative outputs. It keeps the workload smaller and these are the only ones that
    # have per-pixel outputs, which also need to be aggregated into 10x10s.
    output_dir_list_per_ha = uu.create_output_dir_name_list(cn.LULUCF_summative_output_dirs, interval_type, start_year,'4000',
                                                     model_type, interval_end_years, interval_year_diff, input_date, "per_ha")
    output_dir_list_per_pixel = uu.create_output_dir_name_list(cn.LULUCF_summative_output_dirs, interval_type, start_year,'4000',
                                                     model_type, interval_end_years, interval_year_diff, input_date, "per_pixel")

    # Also need to aggregate the land state nodes
    land_state_node_list = [s for s in cn.veg_core_output_dirs if cn.land_state_pattern in s]
    land_state_node_list = uu.create_output_dir_name_list(land_state_node_list, interval_type, start_year,'4000',
                                                     model_type, interval_end_years, interval_year_diff, input_date)

    # Full list of folders to aggregate, in alphabetical order
    output_dir_list = output_dir_list_per_ha + output_dir_list_per_pixel + land_state_node_list
    output_dir_list.sort()  # Sorts in-place

    # Limits folders to process (for testing)
    if first_folders_to_process:
        output_dir_list = output_dir_list[:first_folders_to_process]

    main_logger.info(f"Directories to aggregate: {output_dir_list}")
    main_logger.info(f"There are {len(output_dir_list)} folders to aggregate to 10x10s")

    # Creates lists of 10x10s to aggregate but doesn't aggregate them
    futures = [merge_folder(first_10x10s_to_process, folder)
               for folder in output_dir_list]

    folder_results = compute(*futures)

    # Flattens separate lists of 10x10 aggregations for each s3 folder into a single list that can be fully parallelized
    tile_tasks = [task for sublist in folder_results for task in sublist if task is not None]

    tile_results = []

    main_logger.info(f"Executing {len(tile_tasks)} aggregation tasks in batches of {batch_size}: {uu.timestr()}")

    # Batching code per ChatGPT: https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68af555f-ee54-8320-956e-217eed17e61a
    for i in range(0, len(tile_tasks), batch_size):
        batch = tile_tasks[i:i + batch_size]
        main_logger.info(f"Processing batch {i // batch_size + 1} of {len(batch)}: tasks {i} to {i + len(batch) - 1}")
        batch_results = compute(*batch)
        tile_results.extend(batch_results)

    # # Actually executes the aggregation on the flat list of tasks
    # main_logger.info(f"Executing {len(tile_tasks)} aggregation tasks across {len(output_dir_list)}: {uu.timestr()}")
    # tile_results = compute(*tile_tasks)

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)

    main_logger.info(f"Aggregation complete. {len(tile_results)} tiles written across {len(output_dir_list)} folders: {uu.timestr()}")

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