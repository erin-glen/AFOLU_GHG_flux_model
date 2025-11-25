"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test:
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.0_create_stock_and_stock_change -bb 110 -1 111 0 -cs 10

Coiled small test (1x1 deg):
python -m src.utilities.create_cluster -n 1 -t 1 -m 4 -cn LULUCF_mineral_soil
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.0_create_stock_and_stock_change -cn LULUCF_mineral_soil -bb 110 -1 111 0 -cs 10

Coiled 10x10 deg test (not using):
python -m src.utilities.create_cluster -n 1 -t 1 -m 128 -cn LULUCF_mineral_soil
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.0_create_stock_and_stock_change -cn LULUCF_mineral_soil -bb 110 -10 120 0 -cs 10

Coiled large shapefile test:
python -m src.utilities.create_cluster -n 50 -t 1 -m 4 -cn LULUCF_mineral_soil
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.0_create_stock_and_stock_change -cn LULUCF_mineral_soil -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp -ln "SOC timeseries for 1884-feature shapefile."

Full run:
python -m src.utilities.create_cluster -n 100 -t 1 -m 4 -cn LULUCF_mineral_soil
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.0_create_stock_and_stock_change -cn LULUCF_mineral_soil -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -ln "This is intended to be the definitive SOC timeseries creation."

Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6877a34b-02cc-800a-88cc-a123cdc9ed1b
Selection of 100 workers for full run based on tests in https://app.asana.com/1/25496124013636/task/1206230383901961/comment/1211069739892865?focus=true

Cluster: https://cloud.coiled.io/clusters/1102243/account/wri-forest-research/information?organization=wri
Peak memory: ~1600 MB/worker
Average processing time per chunk (from log, with only 9000 tasks in it): average of 83 seconds (range: 21-176 seconds, stdev = 24)
Time until chunk stats: 5:13:59
Time with chunk stats: 5:18:34
Coiled tasks: 18832 (expected number)
Coiled credits: 522 (101/hr)
AWS cost: $11.8 ($2.28/hr)
"""

import argparse
import gc
import os
import concurrent.futures
import numpy as np
import pandas as pd
import psutil
import rasterio
import time
from dask.distributed import print
from concurrent.futures import ThreadPoolExecutor

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import resize_cluster

def create_soil_C_density_and_change_tiles(bounds, is_final, stage, no_upload, nodata_val, outputs_by_interval_dir_list):

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []

    process = psutil.Process(os.getpid())

    logger_worker = lu.setup_logging_worker()

    chunk_start_time = time.time()

    # uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_final, logger_worker)

    bounds_str = uu.boundstr(bounds)  # String form of chunk bounds, from e.g., [8, -1, 9, 0] to 8_-1_9_0
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)

    # Download dictionary is the SOC global COGs
    download_dict = cn.SOC_COGS

    # Converts the raw COG's kg C/m^3 (top 30 cm) that is rescaled by 10 -> Mg C/ha without the rescaling.
    # OGH rescaled the global COGs by 10 to make them ints instead of floats to save storage.
    SOC_CONVERSION_FACTOR = 3.0 / 10.0  # = 0.3


    ### Part 1: Downloads all inputs for chunk

    # If a particular tile doesn't exist for an input, an array of 0s of the correct size and datatype is returned instead.
    # Thus, this returns a complete set of inputs (missing chunks filled).
    # Note: If running in a local Dask cluster, prints to console may be duplicated. Doesn't happen with a Coiled cluster of the same size (1 worker).
    # Seems to be a problem with local Dask getting overwhelmed by so many futures being created and downloaded from s3.
    futures = uu.prepare_to_download_chunk(bounds, download_dict, chunk_length_pixels, is_final, logger_worker, False)

    lu.print_and_log(f"Waiting for requests for data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)

    # Dictionary that stores the dataset name (key) and downloaded data and their statuses (values)
    layers = {}

    # Output dictionaries for full extent and mineral soil extents.
    # Combined into single dict at end for uploading but separate now because calculating deltas is easier with them separate.
    out_dict_full_extent = {}
    out_dict_min_soil_extent = {}

    # Ensures futures stores Future objects
    # Revised with https://chatgpt.com/share/e/67bde66c-d9a0-800a-a524-a9ef88c641a2 to return status messages for chunks
    for future in concurrent.futures.as_completed(futures):
        layer = futures[future]  # Gets the corresponding key
        data, status = future.result()  # Unpacks the tuple result
        if 'success' not in status: # Prints and logs any inputs that couldn't be accessed (downloaded as all 0s) or had to be padded
            lu.print_and_log(f"{status}: {uu.timestr()}", False, logger_worker)
        layers[layer] = data

    organic_soil_mask_uri = f"{cn.organic_soil_extent_dir}{tile_id}_{cn.organic_soil_extent_pattern}.tif"

    organic_soil_mask = uu.get_tile_dataset_rio(organic_soil_mask_uri, bounds, chunk_length_pixels, 'uint8')
    organic_soil_mask = organic_soil_mask[0]  # Converts downloaded tuple (array, status) to just the array


    ### Part 2: Calculate carbon densities at each interval (Mg C/ha for 0-30 cm)

    for year_range in list(layers.keys()):
        interval_array_full_extent = layers[year_range]

        # Replace COG int16 NoData with 0
        interval_array_full_extent = np.where(interval_array_full_extent == nodata_val, 0, interval_array_full_extent)

        # Convert units from kg/m³ * 10 -> Mg/ha
        converted_array_full_extent = (interval_array_full_extent * SOC_CONVERSION_FACTOR).astype(np.float32)

        # print(f"\n--- Chunk {bounds_str} ---")
        # print(f"SOC density array shape for {bounds_str} for {year_range}: {converted_array_full_extent.shape}")
        # print(f"Organic soil mask shape for {bounds_str} for {year_range}: {organic_soil_mask.shape}")

        # Masks extent to just mineral soil (excludes pixels with high chance of being organic soil, per OpenGeoHub analysis)
        converted_array_min_soil_extent = np.where(organic_soil_mask == 0, converted_array_full_extent, 0)

        # Save back to output dicts with the converted unit arrays
        out_dict_full_extent[f"{cn.SOC_density_full_extent_pattern}{cn.C_density_pixel_meaning}__{year_range}"] = converted_array_full_extent
        out_dict_min_soil_extent[f"{cn.SOC_density_min_soil_extent_pattern}{cn.C_density_pixel_meaning}__{year_range}"] = converted_array_min_soil_extent

        lu.print_and_log(f"After calculating densities for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB",False, logger_worker)

    # Need to put the SOC layers in chronological order so they can be differenced later for full extent and mineral soil extent
    out_dict_full_extent_ordered = dict(sorted(out_dict_full_extent.items()))
    out_dict_min_soil_extent_ordered = dict(sorted(out_dict_min_soil_extent.items()))


    ### Part 3: Calculate density changes between adjacent intervals (Mg C/ha/yr for 0-30 cm)

    # Computes and save deltas. Iterates through both full extent and mineral soil extent
    year_ranges = list(out_dict_full_extent_ordered.keys())    # Need to read from the chronologically ordered dictionary, not the unordered out_dict_full_extent
    # print(year_ranges)
    lu.print_and_log(f"Calculating consecutive SOC changes for {bounds_str}: {uu.timestr()}", False, logger_worker)
    for i in range(len(year_ranges) - 1):
        start_interval = year_ranges[i][-9:]
        end_interval = year_ranges[i + 1][-9:]
        year_diff = int(end_interval[:4])-int(start_interval[:4])

        lu.print_and_log(f"Calculating SOC change for {start_interval} to {end_interval} for {bounds_str}: {uu.timestr()}", False, logger_worker)

        delta_full_extent = (out_dict_full_extent_ordered[f"{cn.SOC_density_full_extent_pattern}{cn.C_density_pixel_meaning}__{end_interval}"] -
                             out_dict_full_extent_ordered[f"{cn.SOC_density_full_extent_pattern}{cn.C_density_pixel_meaning}__{start_interval}"]) / year_diff  # Interval arrays must be unsigned so difference can be negative
        delta_min_soil = (out_dict_min_soil_extent_ordered[f"{cn.SOC_density_min_soil_extent_pattern}{cn.C_density_pixel_meaning}__{end_interval}"] -
                          out_dict_min_soil_extent_ordered[f"{cn.SOC_density_min_soil_extent_pattern}{cn.C_density_pixel_meaning}__{start_interval}"]) / year_diff  # Interval arrays must be unsigned so difference can be negative

        # Saves back to output dicts with the converted unit arrays
        out_dict_full_extent_ordered[f"{cn.SOC_change_full_extent_pattern}{cn.flux_density_pixel_meaning}__{start_interval}_{end_interval}"] = delta_full_extent
        out_dict_min_soil_extent_ordered[f"{cn.SOC_change_min_soil_extent_pattern}{cn.flux_density_pixel_meaning}__{start_interval}_{end_interval}"] = delta_min_soil

        lu.print_and_log(f"After calculating deltas for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB",False, logger_worker)

    # print(out_dict_full_extent_ordered)
    # print(out_dict_min_soil_extent_ordered)


    ### Part 4: Calculate SOC density change between start and end intervals (Mg C/ha/yr for 0-30 cm)

    lu.print_and_log(f"Calculating full-period SOC change for {bounds_str}: {uu.timestr()}", False, logger_worker)

    start_interval = year_ranges[0][-9:]
    end_interval = year_ranges[-1][-9:]
    year_diff = int(end_interval[:4]) - int(start_interval[:4])
    delta_full_period_full_extent = (out_dict_full_extent_ordered[f"{cn.SOC_density_full_extent_pattern}{cn.C_density_pixel_meaning}__{end_interval}"] -
                                     out_dict_full_extent_ordered[f"{cn.SOC_density_full_extent_pattern}{cn.C_density_pixel_meaning}__{start_interval}"]) / year_diff
    delta_full_period_min_soil_extent = (out_dict_min_soil_extent_ordered[f"{cn.SOC_density_min_soil_extent_pattern}{cn.C_density_pixel_meaning}__{end_interval}"] -
                                     out_dict_min_soil_extent_ordered[f"{cn.SOC_density_min_soil_extent_pattern}{cn.C_density_pixel_meaning}__{start_interval}"]) / year_diff

    # Saves back to output dicts with the converted unit arrays
    out_dict_full_extent_ordered[f"{cn.SOC_change_full_extent_pattern}{cn.flux_density_pixel_meaning}__{start_interval}_{end_interval}"] = delta_full_period_full_extent
    out_dict_min_soil_extent_ordered[f"{cn.SOC_change_min_soil_extent_pattern}{cn.flux_density_pixel_meaning}__{start_interval}_{end_interval}"] = delta_full_period_min_soil_extent

    lu.print_and_log(f"After calculating full-period deltas for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB", False, logger_worker)


    ### Part 5: Calculates per ha min, per ha mean, per ha max, and per pixel sum for each output chunk.
    ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.
    ### Also useful for a quick sum of outputs without doing zonal stats

    # Combines the two output dictionaries into a single dictionary
    out_dict = out_dict_full_extent_ordered | out_dict_min_soil_extent_ordered

    lu.print_and_log(f"Populating chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

    # The relevant pixel area (m^2) file in s3
    pixel_area_uri = f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tile_id}.tif"

    # Gets numpy arrays of the model output being analyzed and the area (m^2) per pixel
    pixel_area_chunk = uu.get_tile_dataset_rio(pixel_area_uri, bounds, chunk_length_pixels, 'Float32')
    pixel_area_chunk = pixel_area_chunk[0]  # Converts downloaded tuple (array, status) to just the array

    # Calculates stats for the output layers as a dictionary with chunk attributes.
    # NOTE: The full-interval chunk sums don't exactly match the sums of the individual intervals' chunk sums
    # because of float32 rounding errors. However the output full-model rasters are definitely close enough
    # at the pixel level, so I'm fine with this slight difference.
    # Worked on it in https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/681244d9-83dc-800a-b397-0706e79391c0
    # but never implemented the fix because the very slight rounding results in <0.01% difference.

    for key, array_per_ha in out_dict.items():

        # Converts per hectare values to per pixel values for the output numpy array
        output_per_pixel = array_per_ha * pixel_area_chunk * cn.m2_to_ha

        chunk_stats.append(uu.calculate_stats(array_per_ha, key, bounds_str, tile_id, 'output_layer', output_per_pixel))

    lu.print_and_log(f"Populated chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)


    ### Part 4: Saves numpy arrays as rasters and uploads to s3

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
            # print("interval_year_range:", interval_year_range)

            # Gets the core filename pattern and pixel meaning
            out_pattern_without_pixel_meaning, pixel_meaning = uu.strip_pixel_meaning(out_pattern)
            # print("out_pattern_without_pixel_meaning:", out_pattern_without_pixel_meaning)

            # Retrieves the relevant output s3 path for this specific output (list of one element).
            # First, finds the output folders for all intervals with the relevant patterns
            matched_output_s3_folders = [item for item in outputs_by_interval_dir_list if
                                         out_pattern_without_pixel_meaning in item]
            # print("matched_output_s3_folders:", matched_output_s3_folders)

            # Second, finds the output folder with the right interval for that pattern
            matched_output_s3_folder_list = [item for item in matched_output_s3_folders if interval_year_range in item]
            # print("matched_output_s3_folder_list:", matched_output_s3_folder_list)

            # Output paths without bucket (s3://gfw2-data).
            # Needs [0] because matched_output_s3_folder_list is a list of all intervals.
            s3_path_without_bucket = f"{matched_output_s3_folder_list[0][cn.full_bucket_prefix_length:]}"
            # print("s3_path_without_bucket:", s3_path_without_bucket)
            # print("")

            # Dictionary with metadata for each array
            out_dict[key] = [value, data_type, out_pattern, interval_year_range, s3_path_without_bucket]

        # Converts output numpy arrays to local rasters and puts them in a list of files to upload in parallel
        upload_tasks = uu.save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str,
                                                           out_dict, is_final, logger_worker, out_no_data_val)

        lu.print_and_log(f"Upload tasks created for {bounds_str} in {tile_id}. Uploading now: {uu.timestr()}", False,logger_worker)

        # Execute uploads in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks)

        lu.print_and_log(f"Uploads completed for {bounds_str} in {tile_id} using {cn.outputs_path}: {uu.timestr()}", is_final, logger_worker)

    chunk_end_time = time.time()
    lu.print_and_log(f"  {bounds_str} downloads and density calcs took {round(chunk_end_time - chunk_start_time)} seconds: {uu.timestr()}",False, logger_worker)

    return_message = f"Success for {bounds_str}: {uu.timestr()}"

    # Removes task tracking file from S3 once task is successful
    uu.delete_s3_task_file(stage, bounds, is_final, logger_worker)

    # return return_message  # Return both the success message and the statistics
    return return_message, chunk_stats  # Return both the success message and the statistics


def main(cluster_name, run_date, run_local=False, no_stats=False, no_log=False, no_upload=False,
         chunk_shapefile_uri=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = 'soil_carbon_densities_and_changes'
    model_type = 'standard_model'

    # Runs chunks in batches of specified size.
    # Each batch slows down processing because chunks inevitably lag and that happens more the more batches there are.
    batch_size = 3000
    # batch_size = 3  # For testing batch processing

    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    start_time = uu.timestr() # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Run date: {run_date}")
    main_logger.info(f"no_upload: {no_upload}")

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size, None, fishnet_iso_df, main_logger)

    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_final = False
    # is_final = True  # For simulating a large run
    if len(chunk_list) > 20:
        is_final = True
        main_logger.info("Running as final model.")

    # List of output paths in s3 before each interval is added, with placeholders replaced
    outputs_dir_list = [cn.SOC_density_full_extent_dir, cn.SOC_change_full_extent_dir,
                        cn.SOC_density_min_soil_extent_dir, cn.SOC_change_min_soil_extent_dir]
    outputs_dir_list = [path.replace("PER_HA_OR_PIXEL", "per_ha") for path in outputs_dir_list]
    outputs_dir_list = [path.replace("CHUNK_SIZE_pixels", f"{chunk_size_pixels}_pixels") for path in outputs_dir_list]
    outputs_dir_list = [path.replace("RUN_DATE", cn.SOC_timeseries_run_date) for path in outputs_dir_list]
    # print(outputs_dir_list)

    # List of output paths by interval in s3
    outputs_by_interval_dir_list = []

    # Replaces the placeholder intervals with the actual intervals
    for output_dir in outputs_dir_list:
        if "density" in output_dir:
            for SOC_density_interval in cn.SOC_density_intervals:
                output_dir_interval = output_dir.replace("START_END", SOC_density_interval)
                outputs_by_interval_dir_list = outputs_by_interval_dir_list + [output_dir_interval]

        if "change" in output_dir:
            for SOC_change_interval in cn.SOC_change_intervals:
                output_dir_interval = output_dir.replace("START_END", SOC_change_interval)
                outputs_by_interval_dir_list = outputs_by_interval_dir_list + [output_dir_interval]

    # print(outputs_by_interval_dir_list)
    if is_final:
        main_logger.info(f"outputs_dir_list ({len(outputs_dir_list)} folders):")
        for item in outputs_by_interval_dir_list:
            main_logger.info(f"  {item}")

    # Gets the first COG's URL and gets the NoData value
    first_url = list(cn.SOC_COGS.values())[0][0]
    with rasterio.Env(AWS_NO_SIGN_REQUEST='YES'):
        with rasterio.open(first_url) as src:
            nodata_val = src.nodata

    # Makes a txt for each task in the list. These are deleted as tasks are completed.
    main_logger.info("Creating task txts in s3...")
    uu.create_s3_task_files(stage, chunk_list)


    ### Step 2: Create outputs

    # Creates list of tasks to run (1 task = 1 chunk)
    main_logger.info(f"Creating tasks and starting processing: {uu.timestr()}")
    main_logger.info("Workers' logs to be appended after main function log"+ "\n")

    chunk_batches = [chunk_list[i:i + batch_size] for i in range(0, len(chunk_list), batch_size)]
    main_logger.info(f"There are {len(chunk_batches)} batches to process: {uu.timestr()}")

    # Accumulates all output messages and statistics across batches
    # From https://chatgpt.com/share/e/5599b6b0-1aaa-4d54-98d3-c720a436dd9a
    all_results = []
    all_stats = []
    success_count = 0  # Count of successful chunks

    # Iterates through the batches
    for i, chunk_batch in enumerate(chunk_batches):
        main_logger.info(f"Processing batch {i + 1}/{len(chunk_batches)} ({len(chunk_batch)} chunks): {uu.timestr()}")
        main_logger.info("Creating batch task txts in s3...")
        uu.create_s3_task_files(stage, chunk_batch)

        # This approach handles large task lists (graphs) better than [dask.delayed(calculate_and_upload_LULUCF_fluxes ... )]
        futures = []
        for chunk in chunk_batch:

            future = client.submit(create_soil_C_density_and_change_tiles, chunk,
                                   is_final, stage, no_upload, nodata_val, outputs_by_interval_dir_list)
            futures.append(future)

        batch_results = client.gather(futures)

        all_results.extend(batch_results)

        success_count, batch_stats = uu.count_successful_chunks(chunk_batch, is_final, main_logger, batch_results)
        all_stats.extend(batch_stats)

        # Saves stats from batch in Excel locally in case the run fails, but only if there are multiple batches.
        # That way there are some basic chunk stats (not sorted or anything) to fall back on.
        if len(chunk_batches) > 1:
            main_logger.info(f"Writing batch stats to spreadsheet: {uu.timestr()}")
            df_batch_stats = pd.DataFrame(batch_stats)
            out_spreadsheet = f'TEMP_BATCH_{i}__{stage}__{uu.timestr()}.xlsx'
            local_spreadsheet = f"{cn.local_chunk_stats_path}{out_spreadsheet}"
            with pd.ExcelWriter(local_spreadsheet) as writer:
                df_batch_stats.to_excel(writer, sheet_name=f'stats__batch_{i}', index=False)

        del futures
        del batch_results
        client.run(gc.collect)

        uu.stage_duration(start_time, uu.timestr(), f"{stage}, batch {i}", main_logger)


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
        for output_folder in outputs_by_interval_dir_list:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
            main_logger.info(f"Output rasters in {output_folder}: {file_count}")
            # print(geotiff_files)

    # Prepares 1x1 deg chunk stats spreadsheet: min, mean, max, and sum for all input and output chunks,
    # and min and max values across all chunks for all inputs and outputs
    # only if not suppressed by the --no_stats flag and at least one chunk was successfully (wasn't skipped).
    if (not no_stats) and (success_count > 0):
        chunk_stats_path = uu.compile_1x1_chunk_stats(all_stats, chunk_shapefile_uri, stage, no_upload, main_logger)

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
    parser = argparse.ArgumentParser(description="Extract and compute SOC stock + change in 10x10 deg tiles.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--run_date', help='Date of run, in YYYYMMDD')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    run_date = args.run_date
    bounding_box = args.bounding_box
    chunk_size = args.chunk_size
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_chunks = args.first_chunks
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, run_date, run_local, no_stats, no_log, no_upload, chunk_shapefile_uri,
         bounding_box=bounding_box, chunk_size=chunk_size, first_chunks=first_chunks, log_note=log_note)
