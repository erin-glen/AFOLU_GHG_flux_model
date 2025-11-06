"""
Creates 10x10 deg geotifs from global rechunked zarr.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test (Dask part does not work because of client.submit()):
python -m src.LULUCF.scripts.vegetation_model.4a_aggregate_outputs_to_10x10deg -bb 10 49 11 50 --run_local --no_upload -fy 1 -fv 1 -ft 1 --input_date YYYYMMDD

Coiled small tests (needs 64 GB because of per-ha and per-pixel outputs):
python -m src.utilities.create_cluster -n 1 -t 1 -m 64 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4a_aggregate_outputs_to_10x10deg -cn vegetation_model -bb 10 49 11 50 fy 2 -fv 2 -ft 2 --input_date YYYYMMDD

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 64 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4a_aggregate_outputs_to_10x10deg -cn vegetation_model -bb -64 -22 -63 -21 fy 3 -fv 3 -ft 3 --input_date YYYYMMDD

Coiled Cerrado test (174 features):
python -m src.utilities.create_cluster -n 20 -t 1 -m 64 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4a_aggregate_outputs_to_10x10deg -cn vegetation_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__Cerrado_center_in.shp --input_date YYYYMMDD

Coiled large shapefile test (1884 features):
python -m src.utilities.create_cluster -n 100 -t 1 -m 64 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4a_aggregate_outputs_to_10x10deg -cn vegetation_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD

Full run:
python -m src.utilities.create_cluster -n 200 -t 1 -m 64 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4a_aggregate_outputs_to_10x10deg -cn LULUCF_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --input_date YYYYMMDD --log_note "This is a global run for model v1.0.0 (2016-2024)."

Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/690a21cd-2ea0-8333-9c7f-7091f8016fb3
"""

import argparse
import numpy as np
import rasterio
import tempfile
import fsspec
import time
import psutil
import os
from dask.distributed import print
from rasterio.transform import from_origin
import zarr

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import resize_cluster


# Write single GeoTIFF to S3 using in-memory buffer
def write_single_geotiff_to_s3(var, year_idx, tile_id, data, transform, s3_path, logger_worker):

    fs = fsspec.filesystem("s3", anon=False)

    year = cn.interval_end_years_annual[year_idx]

    lu.print_and_log(f"  Writing {var} for year {year} for {tile_id} to {s3_path}: {uu.timestr()}", False, logger_worker)
    upload_start_time = time.time()

    height, width = data.shape

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": data.dtype,
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "LZW",
        "nodata": 0,
        "tiled": True,
        "blockxsize": 400,
        "blockysize": 400,
    }

    # Counts non-zero pixels for comparison with 1x1 dego geotifs
    valid_pixel_count = int(np.count_nonzero(data != 0))
    # print("pixel count:", valid_pixel_count)

    # Writes to temporary file on disk
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=True) as tmpfile:
        with rasterio.open(tmpfile.name, "w", **profile) as dst:
            dst.write(data, 1)

        # Uploads efficiently using multipart S3 upload
        fs.put_file(tmpfile.name, s3_path)

    upload_end_time = time.time()
    lu.print_and_log(f"  Wrote {var} for year {year} for {tile_id} to {s3_path} in {round(upload_end_time-upload_start_time)} seconds: {uu.timestr()}", False, logger_worker)

    return valid_pixel_count


# Extracts a 10x10° tile from a Zarr store and writes to GeoTIFF on S3
def extract_10x10(var, year_idx, tile_id, raw_path, output_base, no_upload):

    process = psutil.Process(os.getpid())

    logger_worker = lu.setup_logging_worker()

    # Convert tile_id to bounding box (W, S, E, N)
    min_x, min_y, max_x, max_y = uu.get_10x10_tile_bounds(tile_id)

    # Open Zarr group using fsspec mapper
    fs = fsspec.filesystem("s3", anon=False)
    model_zarr_store = zarr.open_group(fs.get_mapper(raw_path), mode="r")

    # Determine pixel indices (applies to model outputs and pixel area)
    lat_array = model_zarr_store["y"][:]
    lon_array = model_zarr_store["x"][:]

    # Get index ranges (applies to model outputs and pixel area)
    y0 = np.searchsorted(lat_array[::-1], max_y, side='right')
    y1 = np.searchsorted(lat_array[::-1], min_y, side='left')
    x0 = np.searchsorted(lon_array, min_x, side='left')
    x1 = np.searchsorted(lon_array, max_x, side='right')

    # Flips y indices since lat is descending
    y0, y1 = len(lat_array) - y1, len(lat_array) - y0
    if y0 > y1:
        y0, y1 = y1, y0

    year = cn.interval_end_years_annual[year_idx]

    lu.print_and_log(f"Extracting {var} for {year} for {tile_id}: {uu.timestr()}", False, logger_worker)
    extract_start_time = time.time()

    # Loads model output data block
    data_per_ha = model_zarr_store[var][year_idx, y0:y1, x0:x1]

    # Calculates per-pixel output (for numeric outputs only)
    pixel_area_zarr_store = zarr.open_group(fs.get_mapper(cn.pixel_area_global_zarr), mode="r")
    pixel_area = pixel_area_zarr_store['pixel_area'][y0:y1, x0:x1]

    # Converts per-ha to per-pixel
    data_per_pixel = data_per_ha * pixel_area / 10000

    # GeoTransform (top-left corner)
    transform = from_origin(min_x, max_y, cn.resolution, cn.resolution)

    extract_end_time = time.time()
    lu.print_and_log(f"  Extracted {var} for year {year} for {tile_id} in {round(extract_end_time - extract_start_time)} seconds: {uu.timestr()}", False, logger_worker)
    lu.print_and_log(f"  Memory usage after 10x10 extraction for {var} for year {year} for {tile_id}: {process.memory_info().rss / 1024 ** 2:.2f} MB", False, logger_worker)

    # Establishes year/year range and units for dataset
    if "density" in var:
        year_or_range = f"{year}"
        per_ha_units = "_ha"
        per_pixel_units = "_pixel"
    elif "emis" in var:
        year_or_range = f"{year - 1}_{year}"
        per_ha_units = "_ha_yr"
        per_pixel_units = "_pixel_yr"
    elif "removals" in var:
        year_or_range = f"{year - 1}_{year}"
        per_ha_units = "_ha_yr"
        per_pixel_units = "_pixel_yr"
    elif "net" in var:
        year_or_range = f"{year - 1}_{year}"
        per_ha_units = "_ha_yr"
        per_pixel_units = "_pixel_yr"
    elif cn.land_state_pattern in var:
        year_or_range = f"{year - 1}_{year}"
        per_ha_units = ""
        per_pixel_units = ""
    else:
        year_or_range = f"{year}"
        per_ha_units = ""
        per_pixel_units = ""

    # Output names and paths for per-ha and per-pixel outputs
    output_path = output_base.replace("PATTERN", var)
    output_path = output_path.replace("START_END", year_or_range)
    output_path_per_ha = output_path.replace("PER_HA_OR_PIXEL", per_ha_units)
    output_name_per_ha = f"{tile_id}__{var}{per_ha_units}_{year_or_range}.tif"
    s3_filename_per_ha = f"{output_path_per_ha}{output_name_per_ha}"

    output_path_per_pixel = output_path.replace("PER_HA_OR_PIXEL", per_pixel_units)
    output_name_per_pixel = f"{tile_id}__{var}{per_pixel_units}_{year_or_range}.tif"
    s3_filename_per_pixel = f"{output_path_per_pixel}{output_name_per_pixel}"

    # Uploads to s3 if requested
    if no_upload == False:

        # Writes geotif to S3
        valid_pixel_count_per_ha = write_single_geotiff_to_s3(var, year_idx, tile_id, data_per_ha, transform, s3_filename_per_ha, logger_worker)

        # Conditionally writes per-pixel output (only if dataset is float32, i.e. numeric output from model).
        # Pixel count from per-pixel outputs is not used.
        if model_zarr_store[var].dtype == np.float32:
            valid_pixel_count_per_pixel = write_single_geotiff_to_s3(
                var, year_idx, tile_id, data_per_pixel, transform, s3_filename_per_pixel, logger_worker
            )
        else:
            valid_pixel_count_per_pixel = None

        # Most stats for the 10x10 deg outputs aren't calculated.
        # Only the pixel count is because it is compared to the pixel counts in all the relevant 1x1s.
        # Dictionary is in a list because it's necessary for chunk stats processing later.
        chunk_stats = [{
            'chunk_id': 'N/A',
            'tile_id': tile_id,
            'layer_name': output_name_per_ha,
            'tile_name': output_name_per_ha,
            'in_out': 'output_layer',
            'pattern': var,
            'years': year_or_range,
            'min_value': 'no data',
            'mean_value': 'no data',
            'max_value': 'no data',
            'count_value': valid_pixel_count_per_ha,
            'sum_value': 'no data',
            'data_type': 'no data'
        }]

    else:

        # Most stats for the 10x10 aren't calculated.
        # Only the pixel count is because it is compared to the pixel counts in all the relevant 1x1s.
        # Dictionary is in a list because it's necessary for chunk stats processing later.
        chunk_stats = [{
            'chunk_id': 'N/A',
            'tile_id': tile_id,
            'layer_name': output_name_per_ha,
            'tile_name': output_name_per_ha,
            'in_out': 'output_layer',
            'pattern': var,
            'years': year_or_range,
            'min_value': 'no data',
            'mean_value': 'no data',
            'max_value': 'no data',
            'count_value': 'not calculated',
            'sum_value': 'no data',
            'data_type': 'no data'
        }]

    return chunk_stats



def main(cluster_name, input_date, run_local, no_log, no_upload, model_chunk_stats_table_name, chunk_shapefile_uri=False, bounding_box=None,
         first_variables_to_process=None, first_years_to_process=None,
         first_tiles_to_process=None, log_note=None):


    ### Step 1: Preparation

    # Model stage being run
    stage = 'vegetation_aggregation_to_10x10_deg'
    model_type = 'standard_model'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    start_time = uu.timestr() # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Input date: {input_date}")

    # Creates s3 paths for the raw and rechunked mega-zarrs
    raw_mega_zarr_path = zu.create_mega_zarr_paths(cn.chunk_dims, 'annual', model_type, input_date)
    rechunked_mega_zarr_path = zu.create_mega_zarr_paths(cn.zarr_pixel_chunks, 'annual', model_type, input_date)

    main_logger.info(f"Raw mega-zarr path: {raw_mega_zarr_path}")
    main_logger.info(f"Rechunked mega-zarr path: {rechunked_mega_zarr_path}")
    main_logger.info(f"Number of years to rechunk: {first_years_to_process}")

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

    unique_tile_ids = list(set(tile_ids))

    # Limits the processed variables to the supplied number (for testing)
    if first_variables_to_process:
        # vars_to_process = cn.full_outputs_to_zarr[0:first_variables_to_process]
        vars_to_process = cn.veg_summative_output_patterns[0:first_variables_to_process]   #TODO for testing
    else:
        # vars_to_process = cn.full_outputs_to_zarr
        vars_to_process = cn.veg_summative_output_patterns   #TODO for testing
    main_logger.info(f"Variables to rechunk and compare chunk stats for: {vars_to_process} ({len(vars_to_process)} out of {len(cn.full_outputs_to_zarr)})")

    if first_years_to_process:
        years_to_process = first_years_to_process
    else:
        years_to_process = len(cn.interval_end_years_annual)
    main_logger.info(f"Years to rechunk and compare chunk stats for: {years_to_process} out of {len(cn.interval_end_years_annual)}")

    if first_tiles_to_process:
        tile_ids_to_process = unique_tile_ids[0:first_tiles_to_process]
    else:
        tile_ids_to_process = unique_tile_ids
    main_logger.info(f"tile_ids to rechunk and compare chunk stats for: {tile_ids_to_process} ({len(tile_ids_to_process)} out of {len(unique_tile_ids)})")

    # Determines if the output file names for final versions of outputs should be used
    is_large_run = False
    # is_large_run = True  # For simulating a large run
    if len(tile_ids) > 20:
        is_large_run = True
        main_logger.info(f"Running as large-scale run model: {is_large_run}")

    # The zarr path that's being used
    zarr_path = zu.create_mega_zarr_paths(cn.zarr_pixel_chunks, 'annual', model_type, input_date)
    main_logger.info(f"Aggregating from: {zarr_path}")

    output_base = f"{cn.outputs_path}PATTERN/{model_type}/annual_intervals/START_END/PER_HA_OR_PIXEL/{cn.full_raster_dims}_pixels/{input_date}/"
    main_logger.info(f"Core output path for aggregation: {output_base}")


    ### Step 2: Create 10x10 deg outputs

    futures = []

    main_logger.info(f"Starting processing: {uu.timestr()}")

    for var in vars_to_process:

        for year_idx in range(years_to_process):

            for tile_id in tile_ids_to_process:

                future = client.submit(extract_10x10,
                                       var, year_idx, tile_id, zarr_path, output_base, no_upload)
                futures.append(future)

    # Results is a list of single-element lists, each with a dictionary of chunk stats for a single variable-year-tile.
    results = client.gather(futures)

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


    ### Step 3: Chunk stats (i.e. pixel counts) for 10x10 degree outputs, aggregates logs

    # Resizes cluster down to 1 worker for chunk stats and log aggregation since that only needs a minimal remainder of the
    # cluster, not all the workers.
    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster(cluster_name, 1)

    # Flattens the tile-level lists of chunk stat dictionaries into a single list.
    all_10x10_stats = [item for sublist in results for item in sublist]

    # Prepares 10x10 deg chunk stats spreadsheet: pixel count for outputs
    uu.aggregate_10x10_chunk_stats(all_10x10_stats, stage, no_upload, main_logger)

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

    parser = argparse.ArgumentParser(description="Create a global rechunked mega-zarr.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--input_date', help='Date of run, in YYYYMMDD')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-fv', '--first_variables_to_process', type=int, help='Number of variables to process from raw mega-zarr (for testing)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-ft', '--first_tiles_to_process', type=int, help='Number of tiles to process (for testing)')
    parser.add_argument('-fy', '--first_years_to_process', type=int, help='Number of years to process from raw mega-zarr (for testing)')
    parser.add_argument('-mcstn', '--model_chunk_stats_table_name', help='s3 path for model chunk stats table that will be compared with zarr chunk stats')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    input_date = args.input_date
    bounding_box = args.bounding_box
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_tiles_to_process = args.first_tiles_to_process
    first_variables_to_process = args.first_variables_to_process
    first_years_to_process = args.first_years_to_process
    model_chunk_stats_table_name = args.model_chunk_stats_table_name
    log_note = args.log_note

    run_local = args.run_local
    no_log = args.no_log
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, input_date, run_local, no_log, no_upload, model_chunk_stats_table_name, chunk_shapefile_uri, bounding_box=bounding_box,
         first_variables_to_process=first_variables_to_process, first_years_to_process=first_years_to_process,
         first_tiles_to_process=first_tiles_to_process, log_note=log_note)
