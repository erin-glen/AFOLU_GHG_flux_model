"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Can only run on 1x1 degree chunks that do not have the run timestamp in the file name.
The way this builds the input file names, it can't handle filenames with the run timestamp.
It also can't handle chunks smaller than 1x1 degree.

Local test:
python -m src.LULUCF.scripts.vegetation_model.3_per_pixel_veg_outputs -bb 10 49 11 50 -cs 1 --no_upload -yr 2000 2024 --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.3_per_pixel_veg_outputs -bb 20 -1 21 0 -cs 1 --no_upload -yr 2000 2024 --input_date YYYYMMDD

Coiled small tests (1x1 deg chunk needs a 32GB worker):
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.vegetation_model.3_per_pixel_veg_outputs -cn LULUCF_postprocessing -bb 10 49 11 50 -cs 1 --no_upload -yr 2000 2024 --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.3_per_pixel_veg_outputs -cn LULUCF_postprocessing -bb 20 -1 21 0 -cs 1 -yr 2000 2024 --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.3_per_pixel_veg_outputs -cn LULUCF_postprocessing -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -f 1 -yr 2000 2024 --input_date YYYYMMDD

Coiled large shapefile test (1x1 deg chunk needs a 32GB worker):
python -m src.utilities.create_cluster -n 100 -t 1 -m 32 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.vegetation_model.3_per_pixel_veg_outputs -cn LULUCF_postprocessing -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp -yr 2000 2024 --input_date YYYYMMDD -ln "Per-pixel outputs for 1884-feature shapefile for model v0.4.0."

Full run (1x1 deg chunk needs a 32GB worker):
python -m src.utilities.create_cluster -n 100 -t 1 -m 32 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.vegetation_model.3_per_pixel_veg_outputs -cn LULUCF_postprocessing -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -yr 2000 2024 --input_date YYYYMMDD -ln "This is intended to be the definitive global run."

Optimization notes: https://app.asana.com/1/25496124013636/task/1206230383901961/comment/1210788116876880?focus=true
"""

import argparse
import dask
import concurrent.futures
import os
import psutil
import sys
import time

from concurrent.futures import ThreadPoolExecutor
from dask.distributed import print

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import resize_cluster

# Speeds up accessing the input geotifs from s3 when they are in a folder with lots of files.
# The more files in an s3 folder, the longer it takes to access them without this environment variable.
# It takes about 9 minutes to access the inputs for a 1x1 deg summative output without this and <1 minute with it.
# Per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68bb4948-c75c-8331-bdf7-1d892029dc0f
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "TRUE"


def create_per_pixel_LULUCF_outputs(bounds, start_year, end_year, interval_type, interval_year_diff, interval_length,
                                    interval_end_years, is_final, no_upload,
                                    summative_inputs_by_interval_dir_list, summative_outputs_by_interval_dir_list,
                                    stage):

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []

    process = psutil.Process(os.getpid())

    logger_worker = lu.setup_logging_worker()

    chunk_start_time = time.time()

    uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_final, logger_worker)

    bounds_str = uu.boundstr(bounds)  # String form of chunk bounds, from e.g., [8, -1, 9, 0] to 8_-1_9_0
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)

    ### Part 1: Downloads all inputs for chunk.
    ### No checks about whether the chunk has data because the way the chunk_list is constructed,
    ### every chunk is relevant and should be processed, so they don't need to be checked.

    # Dictionary of data to download (inputs to model)
    download_dict = {}

    # Iterates through inputs and creates the dictionary of patterns and download paths
    for summative_input_by_interval in summative_inputs_by_interval_dir_list:

        # All the components of the input path
        parts = summative_input_by_interval.strip('/').split('/')

        # Gets the segment for the input pattern
        pattern_idx = parts.index(f"version_{cn.model_version_underscore}")
        pattern_segment = parts[pattern_idx + 1]

        # Gets the segment for the input interval
        interval_idx = parts.index(f"{interval_type}_intervals")
        interval_segment = parts[interval_idx + 1]

        # Gets the segment for the pixel meaning. Different possibilities for carbon pools, fluxes, and everything else.
        if "_ha_yr" in parts:
            pix_meaning_idx = parts.index("_ha_yr")
            pix_meaning_segment = parts[pix_meaning_idx]
        elif "_ha" in parts:
            pix_meaning_idx = parts.index("_ha")
            pix_meaning_segment = parts[pix_meaning_idx]
        else:
            pix_meaning_segment = ''

        # Constructs the dictionary entry.
        # Value has to be a list because prepare_to_download_chunk expects the download dictionary keys to be lists.
        download_dict[f"{pattern_segment}_{interval_segment}{pix_meaning_segment}"] = [f"{summative_input_by_interval}{tile_id}__{bounds_str}__{pattern_segment}{pix_meaning_segment}_{interval_segment}.tif"]

    # If a particular tile doesn't exist for an input, an array of 0s of the correct size and datatype is returned instead.
    # Thus, this returns a complete set of inputs (missing chunks filled).
    # Note: If running in a local Dask cluster, prints to console may be duplicated. Doesn't happen with a Coiled cluster of the same size (1 worker).
    # Seems to be a problem with local Dask getting overwhelmed by so many futures being created and downloaded from s3.
    futures = uu.prepare_to_download_chunk(bounds, download_dict, chunk_length_pixels, is_final, logger_worker, True)

    lu.print_and_log(f"Waiting for requests for data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)

    # Dictionary that stores the dataset name (key) and downloaded data and their statuses (values)
    layers = {}

    # Ensures futures stores Future objects
    # Revised with https://chatgpt.com/share/e/67bde66c-d9a0-800a-a524-a9ef88c641a2 to return status messages for chunks
    for future in concurrent.futures.as_completed(futures):
        layer = futures[future]  # Gets the corresponding key
        data, status = future.result()  # Unpacks the tuple result
        if 'success' not in status: # Prints and logs any inputs that couldn't be accessed (downloaded as all 0s) or had to be padded
            lu.print_and_log(f"{status}: {uu.timestr()}", False, logger_worker)
        layers[layer] = data


    ### Part 2: Creates per-pixel outputs

    lu.print_and_log(f"Calculating per-pixel versions of outputs in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)
    uu.rename_s3_task_file(stage, bounds, "calculating_", is_final, logger_worker)

    # Dictionary for per-pixel outputs
    out_dict = {}

    # The relevant pixel area (m^2) file in s3
    pixel_area_uri = f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tile_id}.tif"

    # Gets numpy arrays of the model output being analyzed and the area (m^2) per pixel
    pixel_area_chunk = uu.get_tile_dataset_rio(pixel_area_uri, bounds, chunk_length_pixels, 'Float32')
    pixel_area_chunk = pixel_area_chunk[0]  # Converts downloaded tuple (array, status) to just the array

    for key, array_per_ha in layers.items():

        # Converts per hectare values to per pixel values for the output numpy array
        output_per_pixel = array_per_ha * pixel_area_chunk * cn.m2_to_ha

        # Gets the pattern and interval years from the filename
        out_pattern, interval_year_range = uu.strip_and_extract_years(key)

        # Gets the core filename pattern and pixel meaning
        out_pattern_without_pixel_meaning, pixel_meaning = uu.strip_pixel_meaning(out_pattern)

        # Replaces the per-ha pixel meaning with the appropriate per-pixel pixel meaning
        if pixel_meaning == cn.C_density_pixel_meaning:
            pixel_meaning = cn.C_per_pixel_pixel_meaning
        elif pixel_meaning == cn.flux_density_pixel_meaning:
            pixel_meaning = cn.flux_per_pixel_pixel_meaning
        else:
            sys.exit(f"Pixel meaning not found for {key}")

        out_dict[f"{out_pattern_without_pixel_meaning}{pixel_meaning}_{interval_year_range}"] = output_per_pixel

        chunk_stats.append(uu.calculate_stats(array_per_ha, key, bounds_str, tile_id, 'output_layer', output_per_pixel))

    lu.print_and_log(f"Populated chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)
    lu.print_and_log(f"After creating per-pixel dict for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB", False, logger_worker)


    ### Part 4: Saves numpy arrays as rasters and uploads to s3

    uu.rename_s3_task_file(stage, bounds, "uploading_", is_final, logger_worker)

    # Only saves arrays to geotifs and uploads them to s3 if enabled
    if not no_upload:

        out_no_data_val = 0  # NoData value for output raster (optional)

        # Adds metadata used for uploading outputs to s3 to the dictionary
        for key, value in out_dict.items():
            data_type = value.dtype.name
            # print("key:", key)

            # Retrieves the file name pattern and date(s) covered for the output file for use in s3 folder construction
            out_pattern, interval_year_range = uu.strip_and_extract_years(key)
            # print("out_pattern:", out_pattern)
            # print("year_range:", year_range)

            # Gets the core filename pattern and pixel meaning
            out_pattern_without_pixel_meaning, pixel_meaning = uu.strip_pixel_meaning(out_pattern)
            # print("out_pattern_without_pixel_meaning:", out_pattern_without_pixel_meaning)

            # Retrieves the relevant output s3 path for this specific output (list of one element).
            # First, finds the output folders for all intervals with the relevant patterns
            matched_output_s3_folders = [item for item in summative_outputs_by_interval_dir_list if out_pattern_without_pixel_meaning in item]
            # print("matched_output_s3_folders:", matched_output_s3_folders)

            # Second, finds the output folder with the right interval for that pattern
            matched_output_s3_folder_list = [item for item in matched_output_s3_folders if interval_year_range in item]
            # print("matched_output_s3_folder_list:", matched_output_s3_folder_list)

            # Output paths without bucket (s3://gfw2-data).
            # Needs [0] because matched_output_s3_folder_list is a list of all intervals.
            s3_path_without_bucket = f"{matched_output_s3_folder_list[0][cn.full_bucket_prefix_length:]}"
            # print("s3_path_without_bucket:", s3_path_without_bucket)

            # Dictionary with metadata for each array
            out_dict[key] = [value, data_type, out_pattern, interval_year_range, s3_path_without_bucket]

        # Converts output numpy arrays to local rasters and puts them in a list of files to upload in parallel
        upload_tasks = uu.save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str,
                                                           out_dict, is_final, logger_worker, out_no_data_val)

        lu.print_and_log(f"Upload tasks created for {bounds_str} in {tile_id}. Uploading now: {uu.timestr()}", False, logger_worker)

        # Execute uploads in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks)

        lu.print_and_log(f"Uploads completed for {bounds_str} in {tile_id} using {cn.outputs_path}: {uu.timestr()}",
                         is_final, logger_worker)

    chunk_end_time = time.time()
    lu.print_and_log(f"{bounds_str} took {round(chunk_end_time - chunk_start_time)} seconds: {uu.timestr()}", False, logger_worker)

    return_message = f"Success for {bounds_str}: {uu.timestr()}"

    # Removes task tracking file from S3 once task is successful
    uu.delete_s3_task_file(stage, bounds, is_final, logger_worker)

    return return_message, chunk_stats  # Return both the success message and the statistics


def main(cluster_name, input_date, year_range, run_local=False, no_stats=False, no_log=False, no_upload=False,
         chunk_shapefile_uri=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):


    ### Step 1: Preparation

    # Model stage being run
    stage = 'LULUCF_per_pixel_outputs'
    model_type = 'standard_model'

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

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    # Starting time for stage
    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Start year: {start_year}; end year: {end_year}")
    main_logger.info(f"Run date: {input_date}")
    main_logger.info(f"no_upload: {no_upload}")

    # Calculates the interval type, difference between start and end years of intervals,
    # and the model output years for the model run
    interval_type, interval_year_diff, interval_length, interval_end_years = uu.get_interval_info(end_year, main_logger, start_year)

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size, first_chunks, fishnet_iso_df, main_logger)

    # Can only run on 1x1 degree chunks
    if chunk_size_pixels != 4000:
        sys.exit("This stage can only be run on 1x1 degree (4000 pixel) chunks.")

    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_final = False
    # is_final = True  # For simulating a large run
    if len(chunk_list) > 20:
        is_final = True
        main_logger.info("Running as final model.")


    # Unlike numba-based scripts, this one doesn't construct the download dictionary in the main function.
    # Instead, it creates a list of input folders, from which a download dictionary is created for each chunk (in the chunk-level function).
    # It's a little simpler this way. Since the datatypes of the inputs don't need to be specified in advance for this script
    # (since it's not using numba), there's no need to centrally create a download dictionary with each input's datatype
    # just once on the scheduler, as is more efficient for scripts that use numba.
    # Creates a list of input directories used in summative output creation based on specifics of the model run
    summative_inputs_by_interval_dir_list = uu.create_output_dir_name_list(cn.LULUCF_summative_output_dirs, interval_type, start_year,
                                                                           chunk_size_pixels, model_type, interval_end_years,
                                                                           interval_year_diff, input_date, "per_ha")
    # print(summative_inputs_by_interval_dir_list)
    if is_final:
        main_logger.info(f"summative_inputs_by_interval_dir_list")
        for item in summative_inputs_by_interval_dir_list:
            main_logger.info(f"  {item}")

    summative_outputs_by_interval_dir_list = uu.create_output_dir_name_list(cn.LULUCF_summative_output_dirs, interval_type, start_year,
                                                                           chunk_size_pixels, model_type, interval_end_years,
                                                                           interval_year_diff, input_date, "per_pixel")
    # print(summative_outputs_by_interval_dir_list)
    if is_final:
        main_logger.info(f"summative_outputs_by_interval_dir_list")
        for item in summative_outputs_by_interval_dir_list:
            main_logger.info(f"  {item}")

    # Makes a txt for each task in the list. These are deleted as tasks are completed.
    main_logger.info("Creating task txts in s3...")
    uu.create_s3_task_files(stage, chunk_list)


    ### Step 2: Create 1x1 degree outputs

    summative_output_delayed_results = [dask.delayed(create_per_pixel_LULUCF_outputs)
                       (chunk, start_year, end_year, interval_type, interval_year_diff, interval_length, interval_end_years,
                        is_final, no_upload,
                        summative_inputs_by_interval_dir_list, summative_outputs_by_interval_dir_list, stage)
                       for chunk in chunk_list]

    # Runs analysis and gathers results
    summative_output_results = dask.compute(*summative_output_delayed_results)

    success_count, all_stats = uu.count_successful_chunks(chunk_list, is_final, main_logger, summative_output_results)

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


    ### Step 3: Counts files in output folders, chunk stats for 1x1 degree outputs, aggregates logs

    # Resizes cluster down to 1 worker for chunk stats and log aggregation since that only needs a minimal remainder of the
    # cluster, not all the workers.
    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster(cluster_name, 1)

    # Iterates through output folders and counts the number of output rasters (only if uploads enabled)
    if not no_upload and is_final:
        for output_folder in summative_outputs_by_interval_dir_list:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
            main_logger.info(f"Output rasters in {output_folder}: {file_count}")
            # print(geotiff_files)

    # Prepares 1x1 deg chunk stats spreadsheet: min, mean, max, and sum for all input and output chunks,
    # and min and max values across all chunks for all inputs and outputs
    # only if not suppressed by the --no_stats flag and at least one chunk was successfully (wasn't skipped).
    if (not no_stats) and (success_count > 0):
        uu.compile_1x1_chunk_stats(all_stats, chunk_shapefile_uri, stage, no_upload, main_logger)

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats", main_logger)

    # Sets it so that no worker logs are created if doing a local run
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats, and worker log compilation", main_logger)

        # Adds the workers' logs to the main log and uploads to s3
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)

    # Closes the Dask client if not running locally
    if not run_local:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate per-pixel outputs from LULUCF model.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--input_date', help='Date of run, in YYYYMMDD')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('-yr', '--year_range', nargs=2, type=int, required=True, help='Starting and ending years for model. Start options: 2000, 2015. End options: 2020, 2024.')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    input_date = args.input_date
    bounding_box = args.bounding_box
    chunk_size = args.chunk_size
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_chunks = args.first_chunks
    year_range = args.year_range
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, input_date, year_range, run_local, no_stats, no_log, no_upload, chunk_shapefile_uri,
         bounding_box=bounding_box, chunk_size=chunk_size,
         first_chunks=first_chunks, log_note=log_note)
