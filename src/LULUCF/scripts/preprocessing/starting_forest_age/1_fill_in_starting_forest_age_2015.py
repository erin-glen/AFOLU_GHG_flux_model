"""
Run from src/LULUCF

Local:
python -m scripts.preprocessing.starting_forest_age.1_fill_in_starting_forest_age_2015 -cn AFOLU_flux_model_scripts -bb 10 49 11 50 -cs 1 --run_local --no_upload

Coiled tiny test:
python -m scripts.utilities.create_cluster -cn AFOLU_flux_model_scripts -n 1
python -m scripts.preprocessing.starting_forest_age.1_fill_in_starting_forest_age_2015 -cn AFOLU_flux_model_scripts -bb 10 49 11 50 -cs 1

Full run:
python -m scripts.utilities.create_cluster -n 16 -t 9 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing.starting_forest_age.1_fill_in_starting_forest_age_2015 -cn AFOLU_flux_model_scripts -cshp -ln "This is intended to be the definitive forest age 2010/2015 run."


"""

import argparse
import sys

import numpy as np
import rasterio
import dask
import xarray as xr
import fsspec
import os

from scipy.ndimage import distance_transform_edt

# Project imports
from ...utilities import constants_and_names as cn
from ...utilities import log_utilities as lu
from ...utilities import universal_utilities as uu
from ...utilities import resize_cluster


def fill_in_starting_forest_age(bounds, fishnet_iso_df, is_final, no_upload, output_dir_list, stage):
    chunk_stats = []
    logger_worker = lu.setup_logging_worker()

    # try:
    # uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_final, logger_worker)
    bounds_str = uu.boundstr(bounds)
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)


    # Input path: S3 GeoTIFF
    input_path = f"{cn.forest_age_2015_dir}{tile_id}__{bounds_str}__{cn.forest_age_2015_pattern}.tif"
    output_path = f"{output_dir_list[0]}{tile_id}__{bounds_str}__{cn.forest_age_2015_filled_in_pattern}.tif"

    input_path = input_path.replace("CHUNK_SIZE", str(chunk_length_pixels))

    # Read raster from S3
    fs = fsspec.filesystem("s3")
    with fs.open(input_path, "rb") as f:
        with rasterio.open(f) as src:
            profile = src.profile
            age_data = src.read(1)
            nodata_val = 0  # Treat 0 as NoData
            profile.update(nodata=nodata_val, compress='lzw')

    # Create mask where 0 is invalid/missing
    mask = age_data != nodata_val
    if not np.any(mask):
        raise ValueError(f"No valid (non-zero) data found in chunk {bounds_str}")

    # Fill 0s using nearest non-zero values
    distance, indices = distance_transform_edt(~mask, return_indices=True)
    filled_data = age_data[tuple(indices)]

    print(filled_data)

    # Write to output GeoTIFF
    # output_temp_path = f"/tmp/filled_{tile_id}.tif"
    output_temp_path = f"/mnt/c/GIS/AFOLU_flux_model/forest_age/filled_in/{tile_id}__{bounds_str}__{cn.forest_age_2015_filled_in_pattern}.tif"
    with rasterio.open(output_temp_path, "w", **profile) as dst:
        dst.write(filled_data, 1)

    # # Upload to S3 if applicable
    # if not no_upload:
    #     with fs.open(output_path, "wb") as f_out:
    #         with open(output_temp_path, "rb") as f_in:
    #             f_out.write(f_in.read())
    #
    # # Clean up temp file
    # if os.path.exists(output_temp_path):
    #     os.remove(output_temp_path)

    return_message = f"Success creating filled forest age for {bounds_str}: {uu.timestr()}"
    # uu.delete_s3_task_file(stage, bounds, is_final, logger_worker)

    # Optionally track stats
    chunk_stats.append({
        "chunk_id": tile_id,
        "min": int(filled_data.min()),
        "mean": float(filled_data[mask].mean()),
        "max": int(filled_data.max())
    })

    print(chunk_stats)

    # except Exception as e:
    #     return_message = f"Error processing chunk {bounds}: {e}: {uu.timestr()}"
    #     lu.print_and_log(return_message, False, logger_worker)
    #     # uu.rename_s3_task_file(stage, bounds, "error_", is_final, logger_worker)

    return return_message, chunk_stats




def main(cluster_name, run_local=False, no_stats=False, no_log=False, no_upload=False,
         use_shapefile=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    age_years = [2015]
    stage = f'fill_in_forest_age_{age_years[0]}__1x1_deg'
    model_type = 'standard'

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(bounding_box, "N/A", client, cluster, log_note, run_local, model_type, stage)

    # Starting time for stage
    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Years for age maps: {age_years}")

    # Returns a dataframe of chunk_id and ISO for the GADM3.6 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso()

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, use_shapefile, chunk_size, first_chunks, fishnet_iso_df, main_logger)

    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_final = False
    if len(chunk_list) > 20:
        is_final = True
        main_logger.info("Running as final model.")

    # Creates list of output directories specific to the run
    output_dir_list = [cn.forest_age_2015_filled_in_dir]
    output_dir_list = [path.replace("CHUNK_SIZE", str(chunk_size_pixels)) for path in output_dir_list]


    ### Step 2: Create 1x1 degree outputs

    # Creates list of tasks to run (1 task = 1 chunk)
    main_logger.info(f"Creating tasks and starting processing: {uu.timestr()}")
    main_logger.info("Workers' logs to be appended after main function log"+ "\n")

    # # Makes a txt for each task in the list. These are deleted as tasks are completed.
    # main_logger.info("Creating task txts in s3...")
    # uu.create_s3_task_files(stage, chunk_list)

    delayed_results_1x1_deg = [dask.delayed(fill_in_starting_forest_age)
                       (chunk, fishnet_iso_df, is_final, no_upload, output_dir_list, stage)
                       for chunk in chunk_list]

    # Runs analysis and gathers results
    results_1x1_deg = dask.compute(*delayed_results_1x1_deg)

    success_count_1x1, all_1x1_stats = uu.count_successful_chunks(chunk_list, is_final, main_logger, results_1x1_deg)

    # Iterates through output folders and counts the number of output rasters (only if uploads enabled)
    if not no_upload:
        for output_folder in output_dir_list:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
            main_logger.info(f"Output rasters in {output_folder}: {file_count}")

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)

    sys.quit()


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
    if (not no_stats) and (success_count_1x1 > 0):
        uu.aggregate_1x1_chunk_stats(all_1x1_stats, stage, no_upload, main_logger)

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats", main_logger)

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
    parser = argparse.ArgumentParser(description="Create carbon pools in 2000.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cshp', '--use_shapefile', action='store_true', help='Use shapefile to determine chunks')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
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
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    main(cluster_name, run_local, no_stats, no_log, no_upload, use_shapefile,
         bounding_box=bounding_box, chunk_size=chunk_size,
         first_chunks=first_chunks, log_note=log_note)
