"""
Run from src/LULUCF

Can only run on 1x1 degree chunks that do not have the run timestamp in the file name.
The way this builds the input file names, it can't handle filenames with the run timestamp.
It also can't handle chunks smaller than 1x1 degree.

Test:
python -m scripts.utilities.create_cluster -n 1 -cn LULUCF_model
python -m scripts.core_model.1_summative_LULUCF_outputs -cn LULUCF_model -bb 10 49 11 50 -cs 1 --no_upload -yr 2015 2023
python -m scripts.core_model.1_summative_LULUCF_outputs -cn LULUCF_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20241125/ -f 1 -yr 2015 2023

Full run:
python -m scripts.utilities.create_cluster -n 50
python -m scripts.core_model.1_summative_LULUCF_outputs -cn LULUCF_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20241125/
"""

import argparse
import dask
import concurrent.futures
import os
import psutil
import sys

from concurrent.futures import ThreadPoolExecutor
from dask.distributed import print

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu
from ..utilities import log_utilities as lu
from ..utilities import resize_cluster


def create_summative_LULUCF_outputs(bounds, start_year, end_year, interval_type, interval_year_diff, interval_length,
                                    interval_end_years, is_final, no_upload,
                                    summative_inputs_by_interval_dir_list, summative_outputs_by_interval_dir_list,
                                    stage):

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []

    process = psutil.Process(os.getpid())

    logger_worker = lu.setup_logging_worker()

    try:

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

            # Constructs the dictionary entry.
            # Value has to be a list because prepare_to_download_chunk expects the download dictionary keys to be lists.
            download_dict[f"{pattern_segment}_{interval_segment}"] = [f"{summative_input_by_interval}{tile_id}__{bounds_str}__{pattern_segment}_{interval_segment}.tif"]

        # If a particular tile doesn't exist for an input, an array of 0s of the correct size and datatype is returned instead.
        # Thus, this returns a complete set of inputs (missing chunks filled).
        # Note: If running in a local Dask cluster, prints to console may be duplicated. Doesn't happen with a Coiled cluster of the same size (1 worker).
        # Seems to be a problem with local Dask getting overwhelmed by so many futures being created and downloaded from s3.
        futures = uu.prepare_to_download_chunk(bounds, download_dict, chunk_length_pixels, is_final, logger_worker)

        lu.print_and_log(f"Waiting for requests for data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)

        # Dictionary that stores the dataset name (key) and downloaded data and their statuses (values)
        layers = {}

        # Ensures futures stores Future objects
        # Revised with https://chatgpt.com/share/e/67bde66c-d9a0-800a-a524-a9ef88c641a2 to return status messages for chunks
        for future in concurrent.futures.as_completed(futures):
            layer = futures[future]  # Gets the corresponding key
            data, status = future.result()  # Unpacks the tuple result
            if 'success' not in status: # Prints and logs any inputs that couldn't be accessed and are downloaded as all 0s
                lu.print_and_log(f"{status}: {uu.timestr()}", is_final, logger_worker)
            layers[layer] = data


        ### Part 2: Creates summative outputs

        lu.print_and_log(f"Summing derivative outputs in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)

        # Everything in out_dict also needs to be in cn.LULUCF_summative_output_dirs
        # because that has the list of basic output directories which are customized for this run
        out_dict = {}

        # Summative outputs for outputs with year ranges in their name (i.e. fluxes)
        for interval_end_year in interval_end_years:

            year_range = f"{interval_end_year - interval_year_diff}_{interval_end_year}"

            # Gross emissions across all carbon pools
            out_dict[f"{cn.gross_emis_all_C_pools_CO2_only_pattern}_{year_range}"] = (
                    layers[f"{cn.agc_gross_emis_pattern}_{year_range}"] + layers[f"{cn.bgc_gross_emis_pattern}_{year_range}"]
                    + layers[f"{cn.deadwood_c_gross_emis_pattern}_{year_range}"] + layers[f"{cn.litter_c_gross_emis_pattern}_{year_range}"])

            # Gross emissions for non-CO2 emissions
            out_dict[f"{cn.gross_emis_all_C_pools_non_CO2_only_pattern}_{year_range}"] = (
                    layers[f"{cn.ch4_flux_pattern}_{year_range}"]
                    + layers[f"{cn.n2o_flux_pattern}_{year_range}"])

            # Gross emissions for all carbon pools and all gases
            out_dict[f"{cn.gross_emis_all_C_pools_all_gases_pattern}_{year_range}"] = (
                out_dict[f"{cn.gross_emis_all_C_pools_CO2_only_pattern}_{year_range}"]
                + out_dict[f"{cn.gross_emis_all_C_pools_non_CO2_only_pattern}_{year_range}"]
            )

            # Gross removals across all carbon pools
            out_dict[f"{cn.gross_removals_all_C_pools_pattern}_{year_range}"] = (
                    layers[f"{cn.agc_gross_removals_pattern}_{year_range}"]
                    + layers[f"{cn.bgc_gross_removals_pattern}_{year_range}"]
                    + layers[f"{cn.deadwood_c_gross_removals_pattern}_{year_range}"]
                    + layers[f"{cn.litter_c_gross_removals_pattern}_{year_range}"])

            # Net flux for each carbon pool
            out_dict[f"{cn.agc_net_flux_pattern}_{year_range}"] = layers[f"{cn.agc_gross_emis_pattern}_{year_range}"] + layers[f"{cn.agc_gross_removals_pattern}_{year_range}"]
            out_dict[f"{cn.bgc_net_flux_pattern}_{year_range}"] = layers[f"{cn.bgc_gross_emis_pattern}_{year_range}"] + layers[f"{cn.bgc_gross_removals_pattern}_{year_range}"]
            out_dict[f"{cn.deadwood_c_net_flux_pattern}_{year_range}"] = layers[f"{cn.deadwood_c_gross_emis_pattern}_{year_range}"] + layers[f"{cn.deadwood_c_gross_removals_pattern}_{year_range}"]
            out_dict[f"{cn.litter_c_net_flux_pattern}_{year_range}"] = layers[f"{cn.litter_c_gross_emis_pattern}_{year_range}"] + layers[f"{cn.litter_c_gross_removals_pattern}_{year_range}"]

            # Net flux across all carbon pools but for CO2 only
            out_dict[f"{cn.net_flux_all_C_pools_CO2_only_pattern}_{year_range}"] = (
                    out_dict[f"{cn.agc_net_flux_pattern}_{year_range}"]
                    + out_dict[f"{cn.bgc_net_flux_pattern}_{year_range}"]
                    + out_dict[f"{cn.deadwood_c_net_flux_pattern}_{year_range}"]
                    + out_dict[f"{cn.litter_c_net_flux_pattern}_{year_range}"])

            # Net flux across all carbon pools, plus non-pool non-CO2 emissions
            out_dict[f"{cn.net_flux_all_C_pools_all_gases_pattern}_{year_range}"] = (
                    out_dict[f"{cn.net_flux_all_C_pools_CO2_only_pattern}_{year_range}"]
                    + out_dict[f"{cn.gross_emis_all_C_pools_non_CO2_only_pattern}_{year_range}"])

        # Summative outputs for outputs with specific years in their name (i.e. carbon densities)
        for interval_end_year in interval_end_years:

            out_dict[f"{cn.non_soil_c_dens_pattern}_{interval_end_year}"] = (
                    layers[f"{cn.agc_dens_pattern}_{interval_end_year}"]
                    + layers[f"{cn.bgc_dens_pattern}_{interval_end_year}"]
                    + layers[f"{cn.deadwood_c_dens_pattern}_{interval_end_year}"]
                    + layers[f"{cn.litter_c_dens_pattern}_{interval_end_year}"])

        lu.print_and_log(f"Done summing derivative outputs in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)
        lu.print_and_log(f"After creating summative outputs for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB", False, logger_worker)


        ### Part 3: Calculates per ha min, per ha mean, per ha max, and per pixel sum for each output chunk.
        ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.
        ### Also useful for a quick sum of outputs without doing zonal stats

        lu.print_and_log(f"Populating chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)

        # The relevant pixel area (m^2) file in s3
        pixel_area_uri = f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tile_id}.tif"

        # Gets numpy arrays of the model output being analyzed and the area (m^2) per pixel
        pixel_area_chunk = uu.get_tile_dataset_rio(pixel_area_uri, bounds, chunk_length_pixels, 'Float32')
        pixel_area_chunk = pixel_area_chunk[0]  # Converts downloaded tuple (array, status) to just the array

        # Calculates stats for the output layers from create_starting_C_densities as a dictionary with chunk attributes
        for key, array_per_ha in out_dict.items():
            # Converts per hectare values to per pixel values for the output numpy array
            output_per_pixel = array_per_ha * pixel_area_chunk * cn.m2_to_ha

            chunk_stats.append(uu.calculate_stats(array_per_ha, key, bounds_str, tile_id, 'output_layer', output_per_pixel))

        lu.print_and_log(f"Populated chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)


        ### Part 4: Saves numpy arrays as rasters and uploads to s3

        uu.rename_s3_task_file(stage, bounds, "uploading_", is_final, logger_worker)

        # Only saves arrays to geotifs and uploads them to s3 if enabled
        if not no_upload:

            out_no_data_val = 0  # NoData value for output raster (optional)

            # Adds metadata used for uploading outputs to s3 to the dictionary
            for key, value in out_dict.items():
                data_type = value.dtype.name

                # Retrieves the file name pattern and date(s) covered for the output file for use in s3 folder construction
                out_pattern, year_range = uu.strip_and_extract_years(key)
                # print(out_pattern)
                # print(year_range)

                # Retrieves the relevant output s3 path for this specific output (list of one element).
                # First, finds the output folders for all intervals with the relevant patterns
                matched_output_s3_folders = [item for item in summative_outputs_by_interval_dir_list if out_pattern in item]
                # print(matched_output_s3_folders)

                # Second, finds the output folder with the right interval for that pattern
                matched_output_s3_folder_list = [item for item in matched_output_s3_folders if year_range in item]
                # print("matched_output_s3_folder_list:", matched_output_s3_folder_list)

                # Output paths without bucket (s3://gfw2-data).
                # Needs [0] because matched_output_s3_folder_list is a list of all intervals.
                s3_path_without_bucket = f"{matched_output_s3_folder_list[0][cn.full_bucket_prefix_length:]}"

                # Dictionary with metadata for each array
                out_dict[key] = [value, data_type, out_pattern, year_range, s3_path_without_bucket]

            # Converts output numpy arrays to local rasters and puts them in a list of files to upload in parallel
            upload_tasks = uu.save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str,
                                                               out_dict, is_final, logger_worker, out_no_data_val)

            lu.print_and_log(f"Upload tasks created for {bounds_str} in {tile_id}. Uploading now: {uu.timestr()}", False, logger_worker)

            # Execute uploads in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                executor.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks)

            lu.print_and_log(f"Uploads completed for {bounds_str} in {tile_id} using {cn.outputs_path}: {uu.timestr()}",
                             is_final, logger_worker)

        return_message = f"Success for {bounds_str}: {uu.timestr()}"

        # Removes task tracking file from S3 once task is successful
        uu.delete_s3_task_file(stage, bounds, is_final, logger_worker)

    except Exception as e:

        return_message = f"Error processing chunk {bounds}: {e}: {uu.timestr()}"

        lu.print_and_log(return_message, is_final, logger_worker)
        print(return_message)
        uu.rename_s3_task_file(stage, bounds, "error_", is_final, logger_worker)

    return return_message, chunk_stats  # Return both the success message and the statistics


def main(cluster_name, year_range, run_local=False, no_stats=False, no_log=False, no_upload=False,
         use_shapefile=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):


    ### Step 1: Preparation

    # Model stage being run
    stage = 'LULUCF_summative_output_calculation'
    model_type = 'standard_model'
    run_date = '20250415'

    # Determines if arguments for start and end year are valid
    if year_range not in [[cn.first_model_year_5_years, cn.last_model_year_5_years],  # 2000-2020
                          [cn.first_model_year_5_years, cn.last_model_year_annual],  # 2000-2023
                          [cn.first_model_year_annual, cn.last_model_year_annual]]:  # 2015-2023
        print("Year range selection not valid")
        sys.exit()
    else:
        start_year = year_range[0]
        end_year = year_range[1]
        print(f"Start year: {start_year}")
        print(f"End year: {end_year}")

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(bounding_box, use_shapefile, client, cluster, log_note, run_local, model_type, stage)

    # Starting time for stage
    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Start year: {start_year}; end year: {end_year}")

    # Calculates the interval type, difference between start and end years of intervals,
    # and the model output years for the model run
    interval_type, interval_year_diff, interval_length, interval_end_years = uu.get_interval_info(end_year, main_logger, start_year)

    # Returns a dataframe of chunk_id and ISO for the GADM3.6 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso()

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, use_shapefile, chunk_size, first_chunks, fishnet_iso_df, main_logger)

    # Can only run on 1x1 degree chunks
    if chunk_size_pixels != 4000:
        sys.exit("This stage can only be run on 1x1 degree (4000 pixel) chunks.")

    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_final = False
    if len(chunk_list) > 20:
        is_final = True
        main_logger.info("Running as final model.")


    # Unlike other scripts, this one doesn't construct the download dictionary in the main function.
    # Instead, it creates a list of input folders, from which a download dictionary is created for each chunk (in the chunk-level function).
    # It's a little simpler this way. Since the datatypes of the inputs don't need to be specified in advance for this script
    # (since it's not using numba), there's no need to centrally create a download dictionary with each input's datatype
    # just once on the scheduler, as is more efficient for scripts that use numba.
    # Creates a list of input directories used in summative output creation based on specifics of the model run
    summative_inputs_by_interval_dir_list = uu.create_output_dir_name_list(cn.LULUCF_core_output_dirs, interval_type, start_year,
                                                     chunk_size_pixels, model_type, interval_end_years, interval_year_diff, run_date)

    # Creates a list of output directories for all outputs and intervals based on specifics of the model run
    summative_outputs_by_interval_dir_list = uu.create_output_dir_name_list(cn.LULUCF_summative_output_dirs, interval_type, start_year,
                                                     chunk_size_pixels, model_type, interval_end_years, interval_year_diff, run_date)


    # Makes a txt for each task in the list. These are deleted as tasks are completed.
    main_logger.info("Creating task txts in s3...")
    uu.create_s3_task_files(stage, chunk_list)


    ### Step 2: Create 1x1 degree outputs

    summative_output_delayed_results = [dask.delayed(create_summative_LULUCF_outputs)
                       (chunk, start_year, end_year, interval_type, interval_year_diff, interval_length, interval_end_years,
                        is_final, no_upload,
                        summative_inputs_by_interval_dir_list, summative_outputs_by_interval_dir_list, stage)
                       for chunk in chunk_list]

    # Runs analysis and gathers results
    summative_output_results = dask.compute(*summative_output_delayed_results)

    success_count, all_stats = uu.count_successful_chunks(chunk_list, is_final, main_logger, summative_output_results)


    # Iterates through output folders and counts the number of output rasters (only if uploads enabled)
    if not no_upload:
        for output_folder in summative_outputs_by_interval_dir_list:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
            main_logger.info(f"Output rasters in {output_folder}: {file_count}")
            # print(geotiff_files)

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


    ### Step 3: Chunk stats for 1x1 degree outputs, aggregates logs

    # Resizes cluster down to 1 worker for chunk stats and log aggregation since that only needs a minimal remainder of the
    # cluster, not all the workers.
    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        # TODO Or maybe just have it terminate the cluster altogether, rather than resize it. Need to make sure that chunk stats and log still work, though.
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster("AFOLU_flux_model_scripts", 1)

    # Prepares 1x1 deg chunk stats spreadsheet: min, mean, max, and sum for all input and output chunks,
    # and min and max values across all chunks for all inputs and outputs
    # only if not suppressed by the --no_stats flag and at least one chunk was successfully (wasn't skipped).
    if (not no_stats) and (success_count > 0):
        uu.compile_1x1_chunk_stats(all_stats, stage, no_upload, main_logger)

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
    parser = argparse.ArgumentParser(description="Calculate summative outputs of core LULUCF model.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cshp', '--use_shapefile', help='Shapefile of chunks')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('-yr', '--year_range', nargs=2, type=int, required=True, help='Starting and ending years for model. Start options: 2000, 2015. End options: 2020, 2023.')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    bounding_box = args.bounding_box
    chunk_size = args.chunk_size
    use_shapefile = args.use_shapefile
    first_chunks = args.first_chunks
    year_range = args.year_range
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, year_range, run_local, no_stats, no_log, no_upload, use_shapefile,
         bounding_box=bounding_box, chunk_size=chunk_size,
         first_chunks=first_chunks, log_note=log_note)
