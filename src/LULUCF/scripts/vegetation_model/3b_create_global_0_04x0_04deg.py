"""
Creates global outputs at 0.04x0.04 deg resolution (approximately 4x4 km at the equator) for specified inputs.
Units are Mg CO2(e)/0.04x0.04 deg pixel/year for interval-level outputs and
Mg CO2(e)/0.04x0.04 deg pixel/full model period for the aggregations over the entire model period (currently 2016-ENDYEAR).
These are for presentations and other static displays.
They are not to be used for calculations or statistics.

Can only run on 1x1 degree chunks that do not have the run timestamp in the file name.
The way this builds the input file names, it can't handle filenames with the run timestamp.
It also can't handle chunks smaller than 1x1 degree.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test:
python -m src.LULUCF.scripts.vegetation_model.3b_create_global_0_04x0_04deg -bb 10 49 11 50 -cs 1 --no_upload --input_date YYYYMMDD

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 4 -cn vegetation_postprocessing
python -m src.LULUCF.scripts.3b_vegetation_model.create_global_0_04x0_04deg -cn vegetation_postprocessing --no_upload --input_date YYYYMMDD

Coiled large shapefile test:
python -m src.utilities.create_cluster -n 10 -t 1 -m 4 -cn vegetation_postprocessing
python -m src.LULUCF.scripts.3b_vegetation_model.create_global_0_04x0_04deg -cn vegetation_postprocessing -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD -ln "This is intended to be the definitive 1884-chunk 0.04x0.04 deg output run."

Full run:
python -m src.utilities.create_cluster -n 10 -t 1 -m 4 -cn vegetation_postprocessing  (because running 6 outputs with 10 years each (including full model total)=60 maps, plus a few workers for safety)
python -m src.LULUCF.scripts.3b_core_veg_model.create_global_0_04x0_04deg -cn vegetation_postprocessing -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --input_date YYYYMMDD -ln "This is intended to be the definitive global 0.04x0.04 deg output run for model v1.0.0 (2016-2024)."

THIS TAKES IMPOSSIBLY LONG! >12 HOURS FOR 60 INPUT FOLDERS!!!!!!!!

# Per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant
"""

import argparse
import sys
import time
import psutil
import rasterio
from osgeo import gdal
import fsspec
from dask import delayed, compute
import os
import re
import tempfile
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

def gdal_vrt_progress(pct, message, data):
    """
    GDAL progress callback.
    pct: 0.0–1.0
    message: current operation
    """
    pct_int = int(pct * 100)
    print(f" GDAL VRT build progress for {data}: {pct_int}%: {uu.timestr()}", flush=True)
    return 1  # return 0 would cancel

def gdal_translate_progress(pct, message, data):
    """
    GDAL progress callback.
    pct: 0.0–1.0
    message: current operation
    """
    pct_int = int(pct * 100)
    print(f" GDAL.translate progress for {data}: {pct_int}%: {uu.timestr()}", flush=True)
    return 1  # return 0 would cancel

@delayed
def mosaic_tiles_to_global(input_folder, output_folder):
    """
    Build a global mosaic using a VRT from a text file of input tiles.
    """
    process = psutil.Process(os.getpid())
    logger_worker = lu.setup_logging_worker()
    start_time = time.time()

    fs = fsspec.filesystem("s3", anon=False)

    # 1. Collect all S3 tile files
    tile_files = fs.glob(f"{input_folder}*.tif")
    if not tile_files:
        return f"No tiles found in {input_folder}"
    lu.print_and_log(f"{len(tile_files)} tiles found in {input_folder}: {uu.timestr()}", False, logger_worker)

    # tile_files = tile_files[0:50]  # For testing
    tile_files = [f"/vsis3/{fp}" for fp in tile_files]  # Faster than vsis3_streaming, by experiment

    # All the components of the input path
    parts = input_folder.strip('/').split('/')

    # Gets the segment for the input pattern
    pattern_idx = parts.index(f"version_{cn.model_version_underscore}")
    pattern_segment = parts[pattern_idx + 1]

    # Gets the segment for the input interval
    interval_idx = parts.index(f"annual_intervals")
    interval_segment = parts[interval_idx + 1]

    output_name = f"{pattern_segment}{cn.flux_aggreg_pixel_meaning}_{interval_segment}_global.tif"


    # 2. Create a temporary working directory for this worker
    tmpdir = tempfile.mkdtemp(prefix="mosaic_")
    safe_name = re.sub(r'[^0-9a-zA-Z]+', '_', input_folder.strip('/'))

    list_path = os.path.join(tmpdir, f"tile_list_{safe_name}.txt")
    vrt_path = os.path.join(tmpdir, f"mosaic_{safe_name}.vrt")

    with open(list_path, "w") as f:
        f.write("\n".join(tile_files))

    # 3. Build VRT
    lu.print_and_log(f"Building VRT for {input_folder} into {vrt_path}: {uu.timestr()}", False, logger_worker)
    # Build VRT directly from list of files
    vrt = gdal.BuildVRT(vrt_path,
                        tile_files,
                        callback=gdal_vrt_progress,
                        callback_data=os.path.basename(output_name))
    if vrt is None:
        raise RuntimeError(f"gdal.BuildVRT failed for {input_folder}")

    vrt = None  # flush to disk

    # 3b. Validate VRT
    try:
        info = gdal.Info(vrt_path, format="json")
        size = info.get("size", [])
        vrt_end_time = time.time()
        lu.print_and_log(f"{vrt_path} created successfully with size {size}, took {round(vrt_end_time - start_time)} seconds: {uu.timestr()}",False, logger_worker)
    except Exception as e:
        lu.print_and_log(f"VRT validation failed: {e}", False, logger_worker)
        raise RuntimeError(f"VRT validation failed for {vrt_path}")

    # 4. Translate VRT → GeoTIFF

    local_out = os.path.join(tmpdir, output_name)

    gtiff_options = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=[
            "COMPRESS=DEFLATE",
            "TILED=YES",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512"
        ],
        callback=gdal_translate_progress,
        callback_data=os.path.basename(output_name)
    )
    lu.print_and_log(f"Writing vrt to geotif for {output_folder}: {uu.timestr()}", False, logger_worker)

    writing_start_time = time.time()
    gdal.Translate(local_out, vrt_path, options=gtiff_options)
    writing_end_time = time.time()
    lu.print_and_log(f"Wrote vrt to geotif for {output_folder}, took {round(writing_end_time - writing_start_time)} seconds: {uu.timestr()}", False, logger_worker)


    lu.print_and_log(f"Uploading geotif for {output_folder}: {uu.timestr()}", False, logger_worker)
    fs.put(local_out, output_folder)
    lu.print_and_log(f"Uploaded mosaic to {output_folder}: {uu.timestr()}", False, logger_worker)

    end_time = time.time()
    lu.print_and_log(f"{output_folder} took {round(end_time - start_time)} seconds: {uu.timestr()}", False, logger_worker)

    return f"Global mosaic written to {output_folder}"


def main(cluster_name, input_date, year_range, run_local=False, no_stats=False, no_log=False, no_upload=False,
         chunk_shapefile_uri=False, first_folders_to_process=None,
         bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):

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
    if len(chunk_list) > 8:
        is_final = True
        main_logger.info("Running as final model.")


    # Unlike numba-based scripts, this one doesn't construct the download dictionary in the main function.
    # Instead, it creates a list of input folders, from which a download dictionary is created for each chunk (in the chunk-level function).
    # It's a little simpler this way. Since the datatypes of the inputs don't need to be specified in advance for this script
    # (since it's not using numba), there's no need to centrally create a download dictionary with each input's datatype
    # just once on the scheduler, as is more efficient for scripts that use numba.
    # Creates a list of input directories used in output creation based on specifics of the model run

    basic_dirs_to_expand = [
        # f"{cn.outputs_path}{cn.gross_emis_all_C_pools_CO2_only_pattern}/{cn.model_type_placholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        # f"{cn.outputs_path}{cn.gross_emis_all_C_pools_non_CO2_only_pattern}/{cn.model_type_placholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        # f"{cn.outputs_path}{cn.gross_emis_all_C_pools_all_gases_pattern}/{cn.model_type_placholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        # f"{cn.outputs_path}{cn.gross_removals_all_C_pools_pattern}/{cn.model_type_placholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        # f"{cn.outputs_path}{cn.net_flux_all_C_pools_CO2_only_pattern}/{cn.model_type_placholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        f"{cn.outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/{cn.model_type_placholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"
    ]

    inputs_by_interval_dir_list = uu.create_output_dir_name_list(basic_dirs_to_expand, interval_type, start_year,
                                                                           25, model_type, interval_end_years_list,
                                                                           # interval_year_diff_list, input_date, True, cn.flux_aggreg_pixel_meaning)
                                                                           interval_year_diff_list, input_date, False, cn.flux_aggreg_pixel_meaning)
    # Limits folders to process (for testing)
    if first_folders_to_process:
        inputs_by_interval_dir_list = inputs_by_interval_dir_list[:first_folders_to_process]

    # print(inputs_by_interval_dir_list)
    if is_final:
        main_logger.info(f"inputs_by_interval_dir_list:")
        for item in inputs_by_interval_dir_list:
            main_logger.info(f"  {item}")

    # Creates a list of output directories for all outputs and intervals based on specifics of the model run
    outputs_by_interval_dir_list = uu.create_output_dir_name_list(basic_dirs_to_expand, interval_type, start_year,
                                                                            "global", model_type, interval_end_years_list,
                                                                            # interval_year_diff_list, input_date, True, cn.flux_aggreg_pixel_meaning)
                                                                            interval_year_diff_list, input_date, False, cn.flux_aggreg_pixel_meaning)
    # Limits folders to process (for testing)
    if first_folders_to_process:
        outputs_by_interval_dir_list = outputs_by_interval_dir_list[:first_folders_to_process]

    if is_final:
        main_logger.info(f"outputs_dir_list:")
        for item in outputs_by_interval_dir_list:
            main_logger.info(f"  {item}")


    ### Step 2: Creates outputs

    # Create one delayed task per mosaic
    tasks = [mosaic_tiles_to_global(f, o) for f, o in zip(inputs_by_interval_dir_list, outputs_by_interval_dir_list)]

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

        # Reduces number of workers in the cluster down to 1 if there is more than 8
        if n_workers > 8:
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
    parser.add_argument('-ffol', '--first_folders_to_process', type=int, help='Number of folders to process from input list')
    parser.add_argument('-yr', '--year_range', nargs=2, type=int, default=[cn.first_model_year_annual, cn.last_model_year_annual],
                        help='Starting and ending years for model. Start options: 2000, 2015. End options: 2020, 2024.')
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
    first_folders_to_process = args.first_folders_to_process
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, input_date, year_range, run_local, no_stats, no_log, no_upload,
         chunk_shapefile_uri, first_folders_to_process=first_folders_to_process,
         bounding_box=bounding_box, chunk_size=chunk_size,
         first_chunks=first_chunks, log_note=log_note)

