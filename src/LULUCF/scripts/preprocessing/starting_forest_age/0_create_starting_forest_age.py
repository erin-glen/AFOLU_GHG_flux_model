"""
Run from src/LULUCF

This preprocessing step doesn't scale quite like others, as far as I can tell.
It starts by reading in the relevant ZARR pieces for the chunks being processed, but I don't know how that scales.
Reading the ZARR chunks involves dozens of tasks and takes much longer than all the subsequent processing.
That's why I am trying the full/global run with -n 6 -t 8; I don't know how helpful it is to have lots of workers for this.
When I was developing this script, I was able to quickly get something that worked on a single 1x1 chunk but then
slowed down in proportion to the number of chunks, even when there was an ample number of workers.
Obviously, that's not how it should go with Dask.
It took a lot longer to work out how to have the script not slow down as I scaled.
I tried several things (in conjunction with ChatGPT) but ultimately settled on having the script read the
ZARR pieces into an array at original resolution, then process the arrays real-time (not lazily).
The original approach was to have everything occur lazily and only compute at the very end, at the time of upload--
but that simply did not scale.
Note that I also tried processing 10x10 degree chunks but the problem there was that the processing of the chunks
once downloaded took too much memory and would've required really large workers.


Local:
python -m scripts.preprocessing.starting_forest_age.0_create_starting_forest_age -cn AFOLU_flux_model_scripts -bb 10 49 11 50 -cs 1 --run_local --no_upload

Coiled tiny test:
python -m scripts.utilities.create_cluster -cn AFOLU_flux_model_scripts -n 1
python -m scripts.preprocessing.starting_forest_age.0_create_starting_forest_age -cn AFOLU_flux_model_scripts -bb 10 49 11 50 -cs 1

Coiled larger test (because this doesn't always scale beyond 1 chunk well):
python -m scripts.utilities.create_cluster -cn AFOLU_flux_model_scripts -n 4 -t 4
python -m scripts.preprocessing.starting_forest_age.0_create_starting_forest_age -cn AFOLU_flux_model_scripts -bb 10 47 13 50 -cs 1

Full run:
python -m scripts.utilities.create_cluster -n 7 -t 9 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing.starting_forest_age.0_create_starting_forest_age -cn AFOLU_flux_model_scripts -cshp -ln "This is intended to be the definitive forest age 2010/2015 run."


https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=8f5974e7-3ece-11ef-967a-4ffbfe06208e
https://datapub.gfz-potsdam.de/download/10.5880.GFZ.1.4.2023.006-VEnuo/
Metadata: https://datapub.gfz-potsdam.de/download/10.5880.GFZ.1.4.2023.006-VEnuo/2023-006_Besnard-et-al_Data-Description-v2.1.pdf
Also needs zarr package

Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67dcb99b-edb8-800a-abd8-f718de76043c
https://chatgpt.com/share/e/67e1b7f6-c0d4-800a-a945-3133de9bf3a0

"""

import argparse
import sys

import dask
import numpy as np
import rasterio
import os
import boto3
import uuid
import shutil
import xarray as xr
from affine import Affine
import fsspec
from fsspec.implementations.cached import CachingFileSystem

# Project imports
from ...utilities import constants_and_names as cn
from ...utilities import log_utilities as lu
from ...utilities import universal_utilities as uu
from ...utilities import resize_cluster


def try_open_zarr(url, cache_path, consolidated=True):
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    os.makedirs(cache_path, exist_ok=True)

    fs = CachingFileSystem(
        fs=fsspec.filesystem("s3", anon=True, endpoint_url="https://s3.gfz-potsdam.de"),
        cache_storage=cache_path,
        block_size=0,
        check_files=False
    )
    store = fs.get_mapper(url)
    return xr.open_zarr(store, consolidated=consolidated)

def calculate_forest_age(bounds, is_final, no_upload, output_dir_list, stage):

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []

    logger_worker = lu.setup_logging_worker()
    s3 = boto3.client("s3")

    bounds_str = uu.boundstr(bounds)
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)

    zarr_url = "s3://dog.atlaseo-glm.eo-gridded-data/collections/GAMI/GAMI_v2.1.zarr"

    base_cache_dir = os.path.expanduser("~/zarr_cache")
    os.makedirs(base_cache_dir, exist_ok=True)

    tile_uuid = uuid.uuid4().hex[:6]
    cache_dir = os.path.join(base_cache_dir, f"{tile_id}_{tile_uuid}")

    try:
        uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_final, logger_worker)
        lu.print_and_log(f"Processing chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

        try:
            ds = try_open_zarr(zarr_url, cache_dir, consolidated=True)
        except ValueError as e:
            if "mmap length is greater than file size" in str(e):
                lu.print_and_log(f"Zarr mmap error in {tile_id}, retrying with consolidated=False...", False, logger_worker)
                ds = try_open_zarr(zarr_url, cache_dir, consolidated=False)
            else:
                raise
        # Gets the forest age dimension of the ZARR
        forest_age = ds["forest_age"]

        # Bounding box for the chunk
        lon_min, lon_max = bounds[0], bounds[2]
        lat_min, lat_max = bounds[1], bounds[3]

        # The chunk needs to be buffered by +/- double the pixel resolution so that the resampled output extends all the way
        # to the edge of the bounding box. Without cn.resolution*2, there was a one pixel-wide border around the edge with
        # no values in it. I believe that's because the resampling was only for pixels with their centers in the targeted
        # slice, so the slice has to be extended a bit so that the outermost resampled pixels have their center within
        # the (expanded) bounding box. The output raster is still the exact right size.
        buffer = cn.resolution * 2

        uu.rename_s3_task_file(stage, bounds, "loading_", is_final, logger_worker)

        # Loads only selected chunk. Loads into memory so that subsequent steps are "eager", not "lazy"
        lu.print_and_log(f"Loading data into memory {bounds_str}: {uu.timestr()}", False, logger_worker)  # Prints even during full run
        da_chunk = forest_age.sel(
            time="2010-01-01",
            latitude=slice(lat_max + buffer, lat_min - buffer),
            longitude=slice(lon_min - buffer, lon_max + buffer)
        ).load()

        # Deletes cache as soon as the data are loaded into memory
        shutil.rmtree(cache_dir, ignore_errors=True)

        uu.rename_s3_task_file(stage, bounds, "calculating_", is_final, logger_worker)

        lu.print_and_log(f"Cleaning {bounds_str}: {uu.timestr()}", False, logger_worker) # Prints even during full run
        da_cleaned = da_chunk.where(da_chunk != -9999, 0)
        da_median = da_cleaned.median(dim="members")  # Median of the 20 ESA AGB estimates upon which age is based

        # Target high-res lat/lon grid
        new_lat = np.arange(lat_max - cn.resolution / 2, lat_min, -cn.resolution)
        new_lon = np.arange(lon_min + cn.resolution / 2, lon_max, cn.resolution)

        # Essentially, resamples from original to final resolution
        lu.print_and_log(f"Interpolating {bounds_str}: {uu.timestr()}", is_final, logger_worker)
        da_resampled = da_median.interp(latitude=new_lat, longitude=new_lon, method="nearest")

        # Rounds from float to int, sets min as 0 and max as 100, and makes NoData = 0
        da_2010 = da_resampled.round().clip(0, 100).fillna(0).astype("int8")
        arr_2010 = da_2010.values

        # Creates the 2015 age map by adding 5 to the 2010 age map where it does not equal 0, then setting to max to 100
        arr_2015 = np.where(arr_2010 != 0, arr_2010 + 5, 0).clip(0, 100).astype("int8")

        transform = Affine.translation(lon_min, lat_max) * Affine.scale(cn.resolution, -cn.resolution)
        crs = "EPSG:4326"

        # Output paths
        file_2010 = f"/tmp/{tile_id}__{bounds_str}__forest_age_2010.tif"
        file_2015 = f"/tmp/{tile_id}__{bounds_str}__forest_age_2015.tif"

        # Writes 2010 age to raster
        lu.print_and_log(f"Saving 2010 raster {file_2010}: {uu.timestr()}", is_final, logger_worker)
        with rasterio.open(file_2010, 'w', driver='GTiff',
            height=arr_2010.shape[0], width=arr_2010.shape[1],
            count=1, dtype="int8", crs=crs, transform=transform,
            compress="LZW", tiled=True, blockxsize=400, blockysize=400
        ) as dst:
            dst.write(arr_2010, 1)

        # Writes 2015 age to raster
        lu.print_and_log(f"Saving 2015 raster {file_2015}: {uu.timestr()}", is_final, logger_worker)
        with rasterio.open(file_2015, 'w', driver='GTiff',
            height=arr_2015.shape[0], width=arr_2015.shape[1],
            count=1, dtype="int8", crs=crs, transform=transform,
            compress="LZW", tiled=True, blockxsize=400, blockysize=400
        ) as dst:
            dst.write(arr_2015, 1)

        chunk_stats.append(uu.calculate_stats(arr_2010, f"forest_age_2010", bounds_str, tile_id, 'output_layer'))
        chunk_stats.append(uu.calculate_stats(arr_2015, f"forest_age_2015", bounds_str, tile_id, 'output_layer'))

        uu.rename_s3_task_file(stage, bounds, "uploading_", is_final, logger_worker)

        if not no_upload:

            # Uploads to S3
            s3_key_2010 = f"{output_dir_list[0][cn.full_bucket_prefix_length:]}{os.path.basename(file_2010)}"
            s3_key_2015 = f"{output_dir_list[1][cn.full_bucket_prefix_length:]}{os.path.basename(file_2015)}"

            lu.print_and_log(f"Uploading to S3: {s3_key_2010}, {s3_key_2015}", is_final, logger_worker)
            s3.upload_file(file_2010, cn.short_bucket_prefix, s3_key_2010)
            s3.upload_file(file_2015, cn.short_bucket_prefix, s3_key_2015)

        lu.print_and_log(f"Finished chunk {bounds_str}: {uu.timestr()}", is_final, logger_worker)

        return_message = f"Success creating 2010/2015 age maps for {bounds_str}: {uu.timestr()}"

        # Cleans up the worker
        os.remove(file_2010)
        os.remove(file_2015)

        # Removes task tracking file from S3 once task is successful
        uu.delete_s3_task_file(stage, bounds, is_final, logger_worker)

    except Exception as e:

        return_message = f"Error creating 2010/2015 age maps for chunk {bounds}: {e}: {uu.timestr()}"

        lu.print_and_log(return_message, False, logger_worker)
        uu.rename_s3_task_file(stage, bounds, "error_", is_final, logger_worker)

        shutil.rmtree(cache_dir, ignore_errors=True)

    return return_message, chunk_stats  # Returns both the success message and the chunk statistics



def main(cluster_name, run_local=False, no_stats=False, no_log=False, no_upload=False,
         use_shapefile=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = f'create_forest_age_2010_2015__1x1_deg'
    model_type = 'standard'
    age_years = [2010, 2015]

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

    output_dir_list = [cn.forest_age_2010_dir, cn.forest_age_2015_dir]

    output_dir_list = [path.replace("DATE", uu.timestr()[:8]) for path in output_dir_list]
    output_dir_list = [path.replace("CHUNK_SIZE", str(chunk_size_pixels)) for path in output_dir_list]


    ### Step 2: Create 1x1 degree outputs

    # Creates list of tasks to run (1 task = 1 chunk)
    main_logger.info(f"Creating tasks and starting processing: {uu.timestr()}")
    main_logger.info("Workers' logs to be appended after main function log"+ "\n")

    # Makes a txt for each task in the list. These are deleted as tasks are completed.
    main_logger.info("Creating task txts in s3...")
    # uu.create_s3_task_files(stage, chunk_list)

    # futures = []
    #
    # for chunk in chunk_list:
    #     future = client.submit(calculate_forest_age, chunk, is_final, no_upload, output_dir_list, stage)
    #     futures.append(future)
    #
    # forest_age_results = client.gather(futures)
    #
    # success_count_1x1, all_1x1_stats = uu.count_successful_chunks(chunk_list, is_final, main_logger, forest_age_results)
    #
    # # Iterates through output folders and counts the number of output rasters (only if uploads enabled)
    # if not no_upload:
    #     for output_folder in output_dir_list:
    #         geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
    #         main_logger.info(f"Output rasters in {output_folder}: {file_count}")
    #         # print(geotiff_files)
    #
    # uu.stage_duration(start_time, uu.timestr(), stage, main_logger)



    batch_size = 500
    # batch_size = 5  # For testing
    chunk_batches = [chunk_list[i:i + batch_size] for i in range(0, len(chunk_list), batch_size)]
    all_forest_age_results = []
    all_1x1_stats = []

    for i, chunk_batch in enumerate(chunk_batches):
        main_logger.info(f"Processing batch {i + 1}/{len(chunk_batches)} ({len(chunk_batch)} chunks)")
        main_logger.info("Creating task txts in s3...")
        uu.create_s3_task_files(stage, chunk_batch)

        futures = [client.submit(calculate_forest_age, chunk, is_final, no_upload, output_dir_list, stage)
                   for chunk in chunk_batch]

        try:
            forest_age_results = client.gather(futures)
        except Exception as e:
            main_logger.error(f"Batch {i + 1} failed: {e}")
            sys.exit()

        all_forest_age_results.extend(forest_age_results)

        success_count_1x1, batch_stats = uu.count_successful_chunks(chunk_batch, is_final, main_logger, forest_age_results)
        all_1x1_stats.extend(batch_stats)

        if not no_upload:
            for output_folder in output_dir_list:
                geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
                main_logger.info(f"Output rasters in {output_folder}: {file_count}")

        main_logger.info(f"Batch {i + 1}/{len(chunk_batches)} complete: {success_count_1x1} succeeded")

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
