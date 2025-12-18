"""
Creates global outputs at 0.04x0.04 deg resolution (approximately 4x4 km at the equator) for specified inputs.
Units are Mg CO2(e)/0.04x0.04 deg pixel/year for interval-level outputs and
Mg CO2(e)/0.04x0.04 deg pixel/full model period for the aggregations over the entire model period (currently 2016-ENDYEAR).
These are for presentations and other static displays.
They are not to be used for calculations or statistics.

Can only run on 10x10 degree tiles already in 0.04x0.04 deg resolution.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test:
python -m src.LULUCF.scripts.vegetation_model.4_create_global_0_04x0_04deg -fy 1 -fv 1 -ft 1 --run_local --no_upload --input_date YYYYMMDD

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 4 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4_create_global_0_04x0_04deg -cn vegetation_model -fy 1 -fv 1 -ft 1 --no_upload --input_date YYYYMMDD

Coiled large shapefile test:
python -m src.utilities.create_cluster -n 10 -t 1 -m 4 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4_create_global_0_04x0_04deg -cn vegetation_model -fy 2 -fv 2 -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD -ln "This is intended to be the definitive 1884-chunk 0.04x0.04 deg output run."

Full run:
python -m src.utilities.create_cluster -n 10 -t 1 -m 4 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4_create_global_0_04x0_04deg -cn vegetation_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --input_date YYYYMMDD -ln "This is intended to be the definitive global 0.04x0.04 deg output run for model v1.0.0 (2016-2024)."

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


def mosaic_tiles_to_global(var_name, year_idx, tile_ids_to_process, base_path, no_upload):

    process = psutil.Process(os.getpid())

    logger_worker = lu.setup_logging_worker()

    start_time = time.time()

    year = cn.interval_end_years_annual[year_idx]

    # Establishes year/year range and units for dataset
    if "density" in var_name:
        units = cn.C_density_aggreg_pixel_meaning
    elif "emis" in var_name:
        units = cn.flux_aggreg_pixel_meaning
    elif "removals" in var_name:
        units = cn.flux_aggreg_pixel_meaning
    elif "net" in var_name:
        units = cn.flux_aggreg_pixel_meaning
    elif cn.land_state_pattern in var_name:
        units = ""
    else:
        units = ""

    # Input s3 folder for dataset and year
    base_path = base_path.replace("PATTERN", var_name)
    base_path = base_path.replace("START_END", str(year))
    base_path = base_path.replace("PER_HA_OR_PIXEL", units)

    # Hacky way to fix land_state and other unitless outputs that otherwise have in the pay YYYY//40000_pixels.
    # This removes the extra / .
    base_path = base_path.replace("//CHUNK_SIZE_pixels", "/CHUNK_SIZE_pixels")

    input_path = base_path.replace("CHUNK_SIZE_pixels", f"{cn.global_aggregation_factor}_pixels")

    # Output s3 folder for dataset and year
    output_path = base_path.replace("CHUNK_SIZE_pixels", "global")

    fs = fsspec.filesystem("s3", anon=False)

    # 1. Collect all S3 tile files
    tile_files = fs.glob(f"{input_path}*.tif")
    if not tile_files:
        return f"No tiles found in {input_path}"
    lu.print_and_log(f"{len(tile_files)} tiles found in {input_path}: {uu.timestr()}", False, logger_worker)

    tile_files = [f"/vsis3/{fp}" for fp in tile_files]  # Faster than vsis3_streaming, by experiment
    # print(tile_files)

    output_name = f"{var_name}{units}_{year}_global.tif"
    # print(output_name)


    # 2. Create a temporary working directory for this worker
    tmpdir = tempfile.mkdtemp(prefix="mosaic_")
    safe_name = re.sub(r'[^0-9a-zA-Z]+', '_', input_path.strip('/'))

    list_path = os.path.join(tmpdir, f"tile_list_{safe_name}.txt")
    vrt_path = os.path.join(tmpdir, f"mosaic_{safe_name}.vrt")

    with open(list_path, "w") as f:
        f.write("\n".join(tile_files))

    # 3. Build VRT
    lu.print_and_log(f"Building VRT for {input_path} into {vrt_path}: {uu.timestr()}", False, logger_worker)
    # Build VRT directly from list of files
    vrt = gdal.BuildVRT(vrt_path,
                        tile_files,
                        callback=gdal_vrt_progress,
                        callback_data=os.path.basename(output_name))
    if vrt is None:
        raise RuntimeError(f"gdal.BuildVRT failed for {input_path}")

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
    lu.print_and_log(f"Writing vrt to geotif for {output_path}: {uu.timestr()}", False, logger_worker)

    writing_start_time = time.time()
    gdal.Translate(local_out, vrt_path, options=gtiff_options)
    writing_end_time = time.time()
    lu.print_and_log(f"Wrote vrt to geotif for {output_path}, took {round(writing_end_time - writing_start_time)} seconds: {uu.timestr()}", False, logger_worker)


    lu.print_and_log(f"Uploading geotif for {output_path}: {uu.timestr()}", False, logger_worker)
    fs.put(local_out, output_path)
    lu.print_and_log(f"Uploaded mosaic to {output_path}: {uu.timestr()}", False, logger_worker)

    end_time = time.time()
    lu.print_and_log(f"{output_path} took {round(end_time - start_time)} seconds: {uu.timestr()}", False, logger_worker)

    return f"Global mosaic written to {output_path}"


def main(cluster_name, input_date, run_local, no_log, no_upload, chunk_shapefile_uri=False, bounding_box=None,
         first_variables_to_process=None, first_years_to_process=None,
         first_tiles_to_process=None, log_note=None):


    ### Step 1: Preparation

    # Model stage being run
    stage = 'vegetation_0_04deg_output_global'
    model_type = 'standard_model'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    start_time = uu.timestr() # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Start year: {cn.first_model_year_annual}; end year: {cn.last_model_year_annual}")
    main_logger.info(f"Input date: {input_date}")
    main_logger.info(f"no_upload: {no_upload}")

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_size_deg = 1   # Chunk size for geotifs is set at 1x1 deg
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size_deg, None, fishnet_iso_df, main_logger)

    # Gets a list of unique tile_ids from the chunk list
    tile_ids = []
    for chunk in chunk_list:
        tile_id = uu.xy_to_tile_id(chunk[0], chunk[3])  # tile_id in YYN/S_XXXE/W
        tile_ids.append(tile_id)

    unique_tile_ids = sorted(list(set(tile_ids)))

    # Outputs to turn into 10x10 tile
    # full_list_of_vars = cn.full_outputs_to_zarr   # If all variables are to be made into 10x10s (but very expensive)
    full_list_of_vars = cn.veg_summative_output_patterns + [cn.land_state_pattern] # Summative outputs + land state nodes

    # Limits the processed variables to the supplied number (for testing)
    if first_variables_to_process:
        vars_to_process = full_list_of_vars[0:first_variables_to_process]
    else:
        vars_to_process = full_list_of_vars
    main_logger.info(f"Variables to create 10x10 deg tiles for: {vars_to_process} ({len(vars_to_process)} out of {len(full_list_of_vars)})")

    # Limits the processed years to the supplied number (for testing)
    if first_years_to_process:
        years_to_process = first_years_to_process
    else:
        years_to_process = len(cn.interval_end_years_annual)
    main_logger.info(f"Years to aggregate to 10x10 deg and compare chunk stats for: {years_to_process} out of {len(cn.interval_end_years_annual)}")

    if first_tiles_to_process:
        tile_ids_to_process = unique_tile_ids[0:first_tiles_to_process]
    else:
        tile_ids_to_process = unique_tile_ids
    main_logger.info(f"tile_ids to aggregate to 10x10 deg and compare chunk stats for: {tile_ids_to_process} ({len(tile_ids_to_process)} out of {len(unique_tile_ids)})")

    tile_ids_to_process = ['20S_120E']  #TODO testing

    # Determines if the output file names for final versions of outputs should be used
    is_large_run = False
    # is_large_run = True  # For simulating a large run
    if len(tile_ids) > 20:
        is_large_run = True
        main_logger.info(f"Running as large-scale run model: {is_large_run}")

    base_path = f"{cn.veg_outputs_path}PATTERN/{model_type}/annual_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{input_date}/"
    main_logger.info(f"Core output path for aggregation: {base_path}")


    ### Step 2: Creates outputs

    futures = []

    main_logger.info(f"Starting processing: {uu.timestr()}")

    for var_name in vars_to_process:

        for year_idx in range(years_to_process):

            # future = client.submit(uu.create_10x10_deg_geotif_from_zarr,
            future = client.submit(mosaic_tiles_to_global,
                                   var_name, year_idx, tile_ids_to_process, base_path, no_upload)
            futures.append(future)

    results = client.gather(futures)
    print(results)

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)



    # ### Step 3: Counts files in output folders, chunk stats for outputs, aggregates logs
    #
    # # Resizes cluster down to 1 worker for chunk stats and log aggregation since that only needs a minimal remainder of the
    # # cluster, not all the workers.
    # if not run_local:
    #     workers = client.scheduler_info()["workers"]
    #     n_workers = len(workers)
    #
    #     # Reduces number of workers in the cluster down to 1 if there is more than 8
    #     if n_workers > 8:
    #         main_logger.info("Resizing cluster to 1 worker")
    #
    #         resize_cluster.resize_coiled_cluster(cluster_name, 1)
    #
    # # Iterates through output folders and counts the number of output rasters (only if uploads enabled)
    # if not no_upload and is_final:
    #     for output_folder in outputs_by_interval_dir_list:
    #         geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
    #         main_logger.info(f"Output rasters in {output_folder}: {file_count}")
    #         # print(geotiff_files)
    #
    # # # Prepares 1x1 deg chunk stats spreadsheet: min, mean, max, and sum for all input and output chunks,
    # # # and min and max values across all chunks for all inputs and outputs
    # # # only if not suppressed by the --no_stats flag and at least one chunk was successfully (wasn't skipped).
    # # if (not no_stats) and (success_count > 0):
    # #     uu.compile_1x1_chunk_stats(all_stats, chunk_shapefile_uri, stage, no_upload, main_logger)
    # #
    # # uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats", main_logger)
    #
    # # Sets it so that no worker logs are created if doing a local run
    # if not run_local:
    #
    #     # Creates combined log from all workers if not deactivated
    #     worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
    #     uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats, and worker log compilation", main_logger)
    #
    #     # Adds the workers' logs to the main log and uploads to s3
    #     lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)
    #
    # # Closes the Dask client if not running locally
    # if not run_local:
    #     client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creat global 0.04x0.04 deg output maps.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--input_date', help='Date of run, in YYYYMMDD')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-fv', '--first_variables_to_process', type=int, help='Number of variables to process from raw mega-zarr (for testing)')
    parser.add_argument('-ft', '--first_tiles_to_process', type=int, help='Number of tiles to process (for testing)')
    parser.add_argument('-fy', '--first_years_to_process', type=int, help='Number of years to process from raw mega-zarr (for testing)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    input_date = args.input_date
    bounding_box = args.bounding_box
    first_tiles_to_process = args.first_tiles_to_process
    first_variables_to_process = args.first_variables_to_process
    first_years_to_process = args.first_years_to_process
    chunk_shapefile_uri = args.chunk_shapefile_uri
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, input_date, run_local, no_log, no_upload, chunk_shapefile_uri, bounding_box=bounding_box,
         first_variables_to_process=first_variables_to_process, first_years_to_process=first_years_to_process,
         first_tiles_to_process=first_tiles_to_process, log_note=log_note)
