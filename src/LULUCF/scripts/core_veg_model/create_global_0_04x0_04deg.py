"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test:
python -m src.LULUCF.scripts.core_veg_model.create_chunks_0_04x0_04deg -bb 10 49 11 50 -cs 1 --no_upload -yr 2000 2024 --input_date YYYYMMDD

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 4 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.core_veg_model.create_chunks_0_04x0_04deg -cn LULUCF_postprocessing -bb 10 49 11 50 -cs 1 --no_upload -yr 2000 2024 --input_date YYYYMMDD

Coiled large shapefile test:
python -m src.utilities.create_cluster -n 50 -t 1 -m 4 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.core_veg_model.create_chunks_0_04x0_04deg -cn LULUCF_postprocessing -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp -yr 2000 2024 --input_date YYYYMMDD -ln "This is intended to be the definitive 1884-chunk 0.04x0.04 deg output run."

Full run:
python -m src.utilities.create_cluster -n 100 -t 1 -m 4 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.core_veg_model.create_chunks_0_04x0_04deg -cn LULUCF_postprocessing -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -yr 2000 2024 --input_date YYYYMMDD -ln "This is intended to be the definitive 0.04x0.04 deg output run."

"""

import argparse
import sys
import time
import psutil
import rasterio
from rasterio.merge import merge
import fsspec
from dask import delayed, compute
import os
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

@delayed
def mosaic_tiles_to_global(s3_folder, global_out_path):

    process = psutil.Process(os.getpid())

    logger_worker = lu.setup_logging_worker()

    start_time = time.time()

    fs = fsspec.filesystem("s3", anon=False)

    # 1. Collects all S3 tile files
    tile_files = fs.glob(f"{s3_folder}/*.tif")
    if not tile_files:
        return f"  No tiles found in {s3_folder}"

    # Prefix with s3:// for rasterio
    tile_files = [f"s3://{fp}" for fp in tile_files]

    # 2. Opens with rasterio
    lu.print_and_log(f"Opening files in {s3_folder}: {uu.timestr()}", False, logger_worker)
    src_files_to_mosaic = [rasterio.open(fp) for fp in tile_files]

    lu.print_and_log(f"After merging 0.04 deg outputs globally: {process.memory_info().rss / 1024 ** 2:.2f} MB", False, logger_worker)

    # 3. Merges into one array
    print(f"Merging files in {s3_folder}: {uu.timestr()}")
    mosaic_arr, mosaic_transform = merge(src_files_to_mosaic)

    # Uses metadata of first tile as base
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        # "driver": "COG",
        "height": mosaic_arr.shape[1],
        "width": mosaic_arr.shape[2],
        "transform": mosaic_transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512
    })

    for src in src_files_to_mosaic:
        src.close()

    # 4. Writes to local temp file
    lu.print_and_log(f"Saving files in {s3_folder}: {uu.timestr()}", False, logger_worker)
    local_out = os.path.basename(global_out_path)
    with rasterio.open(local_out, "w", **out_meta) as dest:
        dest.write(mosaic_arr)

    # 5. Uploads to S3
    lu.print_and_log(f"Uploading files in {s3_folder}: {uu.timestr()}", False, logger_worker)
    fs.put(local_out, global_out_path)

    end_time = time.time()
    lu.print_and_log(f"{global_out_path} took {round(end_time - start_time)} seconds: {uu.timestr()}", False, logger_worker)

    return f"Global mosaic written to {global_out_path}"


def main(cluster_name, input_date, year_range, run_local=False, no_stats=False, no_log=False, no_upload=False,
         chunk_shapefile_uri=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = 'LULUCF_0_04deg_output_global'
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

    start_time = uu.timestr() # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Start year: {start_year}; end year: {end_year}")
    main_logger.info(f"Run date: {input_date}")
    main_logger.info(f"no_upload: {no_upload}")

    # Calculates the interval type, difference between start and end years of intervals,
    # and the model output years for the model run
    interval_type, interval_year_diff_list, interval_length_list, interval_end_years_list = uu.get_interval_info(end_year, main_logger, start_year)

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


    # # Unlike numba-based scripts, this one doesn't construct the download dictionary in the main function.
    # # Instead, it creates a list of input folders, from which a download dictionary is created for each chunk (in the chunk-level function).
    # # It's a little simpler this way. Since the datatypes of the inputs don't need to be specified in advance for this script
    # # (since it's not using numba), there's no need to centrally create a download dictionary with each input's datatype
    # # just once on the scheduler, as is more efficient for scripts that use numba.
    # # Creates a list of input directories used in output creation based on specifics of the model run
    # inputs_by_interval_dir_list = uu.create_output_dir_name_list(cn.LULUCF_summative_output_dirs, interval_type, start_year,
    #                                                                        chunk_size_pixels, model_type, interval_end_years_list,
    #                                                                        interval_year_diff_list, input_date, "per_ha")

    inputs_by_interval_dir_list = [
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2015_2016/_0_04deg_yr/160_pixels/20250904/",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2016_2017/_0_04deg_yr/160_pixels/20250904/",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2017_2018/_0_04deg_yr/160_pixels/20250904/",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2018_2019/_0_04deg_yr/160_pixels/20250904/",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2019_2020/_0_04deg_yr/160_pixels/20250904/",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2020_2021/_0_04deg_yr/160_pixels/20250904/",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2021_2022/_0_04deg_yr/160_pixels/20250904/",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2022_2023/_0_04deg_yr/160_pixels/20250904/",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2023_2024/_0_04deg_yr/160_pixels/20250904/"
    ]

    # print(inputs_by_interval_dir_list)
    if is_final:
        main_logger.info(f"inputs_by_interval_dir_list:")
        for item in inputs_by_interval_dir_list:
            main_logger.info(f"  {item}")

    # # Creates a list of output directories for all outputs and intervals based on specifics of the model run
    # outputs_by_interval_dir_list = uu.create_output_dir_name_list(cn.LULUCF_summative_output_dirs, interval_type, start_year,
    #                                                                         chunk_size_pixels, model_type, interval_end_years_list,
    #                                                                         interval_year_diff_list, input_date, "per_ha")

    outputs_by_interval_dir_list = [
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2015_2016/_0_04deg_yr/global/20250904/net_flux__all_C_pools__all_gases__MgCO2e_2015_2016_0_04deg_yr.tif",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2016_2017/_0_04deg_yr/global/20250904/net_flux__all_C_pools__all_gases__MgCO2e_2016_2017_0_04deg_yr.tif",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2017_2018/_0_04deg_yr/global/20250904/net_flux__all_C_pools__all_gases__MgCO2e_2017_2018_0_04deg_yr.tif",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2018_2019/_0_04deg_yr/global/20250904/net_flux__all_C_pools__all_gases__MgCO2e_2018_2019_0_04deg_yr.tif",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2019_2020/_0_04deg_yr/global/20250904/net_flux__all_C_pools__all_gases__MgCO2e_2019_2020_0_04deg_yr.tif",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2020_2021/_0_04deg_yr/global/20250904/net_flux__all_C_pools__all_gases__MgCO2e_2020_2021_0_04deg_yr.tif",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2021_2022/_0_04deg_yr/global/20250904/net_flux__all_C_pools__all_gases__MgCO2e_2021_2022_0_04deg_yr.tif",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2022_2023/_0_04deg_yr/global/20250904/net_flux__all_C_pools__all_gases__MgCO2e_2022_2023_0_04deg_yr.tif",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2023_2024/_0_04deg_yr/global/20250904/net_flux__all_C_pools__all_gases__MgCO2e_2023_2024_0_04deg_yr.tif"
    ]

    if is_final:
        main_logger.info(f"outputs_dir_list:")
        for item in outputs_by_interval_dir_list:
            main_logger.info(f"  {item}")

    # Makes a txt for each task in the list. These are deleted as tasks are completed.
    main_logger.info("Creating task txts in s3...")
    uu.create_s3_task_files(stage, chunk_list)


    ### Step 2: Creates outputs

    # Create one delayed task per mosaic
    tasks = [mosaic_tiles_to_global(f, o) for f, o in zip(folders, outputs)]

    # Run them in parallel
    results = compute(*tasks)
    print(results)

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


    ### Step 3: Counts files in output folders, chunk stats for outputs, aggregates logs

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

    # # Prepares 1x1 deg chunk stats spreadsheet: min, mean, max, and sum for all input and output chunks,
    # # and min and max values across all chunks for all inputs and outputs
    # # only if not suppressed by the --no_stats flag and at least one chunk was successfully (wasn't skipped).
    # if (not no_stats) and (success_count > 0):
    #     uu.compile_1x1_chunk_stats(all_stats, chunk_shapefile_uri, stage, no_upload, main_logger)
    #
    # uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats", main_logger)

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
    parser = argparse.ArgumentParser(description="Creat global 0.04x0.04 deg output maps.")
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

