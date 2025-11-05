"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test (Dask part does not work because of client.submit()):
python -m src.LULUCF.scripts.vegetation_model.4a_aggregate_outputs_to_10x10deg -bb 10 49.75 10.25 50 --run_local --no_upload --input_date YYYYMMDD

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.1_calculate_veg_fluxes -cn vegetation_model -bb 116.25 -2.25 116.5 -2 --input_date YYYYMMDD

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.1_calculate_veg_fluxes -cn vegetation_model -bb -64 -22 -63 -21 --input_date YYYYMMDD --create_zarr

Coiled Cerrado test (174 features):
python -m src.utilities.create_cluster -n 20 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.1_calculate_veg_fluxes -cn vegetation_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__Cerrado_center_in.shp --input_date YYYYMMDD

Coiled large shapefile test (1884 features):
python -m src.utilities.create_cluster -n 100 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.1_calculate_veg_fluxes -cn vegetation_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD

Full run:
python -m src.utilities.create_cluster -n 200 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.1_calculate_veg_fluxes -cn LULUCF_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --input_date 20250921 --log_note "This is a global run for model v1.0.0 (2016-2024)."

Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/690a21cd-2ea0-8333-9c7f-7091f8016fb3
"""

import dask
import xarray as xr
import argparse
import numpy as np
import rasterio
import tempfile
import fsspec
import time
from dask.distributed import print
from rasterio.transform import from_origin
import s3fs
import zarr

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import numba_utilities as nu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import resize_cluster
from src.utilities.constants_and_names import full_outputs_to_zarr

# STEP 1: Start Dask cluster via Coiled

cluster, client, run_local = uu.connect_to_Coiled_cluster("aggregation_test", False)

# STEP 2: Parameters
zarr_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_1_0_3_1884_chunks/mega_zarr/standard_model/annual_intervals/10000_pixels/20251027/"
# tile_ids = ["20S_020E", "50N_000E"]  # example list – you can supply many more
# tile_ids = ["00N_020E"]  # example list – you can supply many more
# tile_ids = ["30N_020W"]  # example list – you can supply many more
tile_size_deg = 10
resolution = 0.00025
samples_per_tile = int(tile_size_deg / resolution)  # 40000
output_base = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_1_0_3_1884_chunks/10x10_test"

# S3 Filesystem
fs = s3fs.S3FileSystem(anon=False)


# STEP 5: Write single GeoTIFF to S3 using in-memory buffer
def write_geotiff_to_s3(var, year_idx, data, transform, s3_path):
    """Write 2D array to S3 as a GeoTIFF using Rasterio's in-memory buffer."""
    print(f"Writing {var} for year {year_idx} to {s3_path}: {uu.timestr()}")
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

    # Write to temporary file on disk
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=True) as tmpfile:
        with rasterio.open(tmpfile.name, "w", **profile) as dst:
            dst.write(data, 1)

        # Upload efficiently using multipart S3 upload
        fs.put_file(tmpfile.name, s3_path)

    upload_end_time = time.time()
    print(f"  Wrote {var} for year {year_idx} to {s3_path} in {round(upload_end_time-upload_start_time)} seconds: {uu.timestr()}")


# STEP 6: Main export function
def extract_10x10(var, year_idx, tile_id, raw_path, output_base):
    """Extracts a 10x10° tile from a Zarr store and writes to GeoTIFF on S3."""

    # Convert tile_id to bounding box (W, S, E, N)
    min_x, min_y, max_x, max_y = uu.get_10x10_tile_bounds(tile_id)

    # Open Zarr group using fsspec mapper
    fs = fsspec.filesystem("s3", anon=False)
    zarr_store = zarr.open_group(fs.get_mapper(raw_path), mode="r")

    # Determine pixel indices
    lat_array = zarr_store["y"][:]
    lon_array = zarr_store["x"][:]

    # Get index ranges
    y0 = np.searchsorted(lat_array[::-1], max_y, side='right')
    y1 = np.searchsorted(lat_array[::-1], min_y, side='left')
    x0 = np.searchsorted(lon_array, min_x, side='left')
    x1 = np.searchsorted(lon_array, max_x, side='right')

    # Flip y indices since lat is descending
    y0, y1 = len(lat_array) - y1, len(lat_array) - y0

    if y0 > y1:
        y0, y1 = y1, y0

    year = cn.interval_end_years_annual[year_idx]

    print(f"Extracting {var} for {year} at {tile_id} (x: {x0}-{x1}, y: {y0}-{y1}): {uu.timestr()}")
    extract_start_time = time.time()

    # Load data block (Zarr lazy indexing)
    block = zarr_store[var][year_idx, y0:y1, x0:x1]
    data = block.astype(np.float32)

    # GeoTransform (top-left corner)
    transform = from_origin(min_x, max_y, resolution, resolution)

    extract_end_time = time.time()
    print(f"  Extracted {var} for year {year_idx} in {round(extract_end_time - extract_start_time)} seconds: {uu.timestr()}")

    # Output path
    s3_filename = f"{output_base}/{var}/{year}/{tile_id}.tif"

    # Write to S3
    write_geotiff_to_s3(var, year_idx, data, transform, s3_filename)

    return s3_filename



def main(cluster_name, input_date, run_local, no_log, no_upload, model_chunk_stats_table_name, chunk_shapefile_uri=False, bounding_box=None,
         test_print_stats_chunk=None, first_variables_to_process=None, first_years_to_process=None,
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
    main_logger.info(f"Test chunk (to print stats): {test_print_stats_chunk}")
    main_logger.info(f"Tolerance for comparison between model and zarr chunk stat metrics: {cn.zarr_difference_tolerance}")

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_size_deg = 1   # Chunk size for geotifs is set at 1x1 deg
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size_deg, None, fishnet_iso_df, main_logger)
    main_logger.info(f"Chunks to get stats for from rechunked mega-zarr: {len(chunk_list)}")

    tile_ids = []
    for chunk in chunk_list:
        tile_id = uu.xy_to_tile_id(chunk[0], chunk[3])  # tile_id in YYN/S_XXXE/W
        tile_ids.append(tile_id)

    unique_tile_ids = list(set(tile_ids))

    # Limits the processed variables to the supplied number (for testing)
    if first_variables_to_process:
        vars_to_process = cn.full_outputs_to_zarr[0:first_variables_to_process]
    else:
        vars_to_process = cn.full_outputs_to_zarr
    main_logger.info(f"Variables to rechunk and compare chunk stats for: {vars_to_process} ({len(vars_to_process)} out of {len(cn.full_outputs_to_zarr)}: {uu.timestr()}")

    if first_years_to_process:
        years_to_process = first_years_to_process
    else:
        years_to_process = len(cn.interval_end_years_annual)
    main_logger.info(f"Years to rechunk and compare chunk stats for: {years_to_process} out of {len(cn.interval_end_years_annual)}: {uu.timestr()}")

    if first_tiles_to_process:
        tile_ids_to_process = unique_tile_ids[0:first_tiles_to_process]
    else:
        tile_ids_to_process = unique_tile_ids
    main_logger.info(f"tile_ids to rechunk and compare chunk stats for: {tile_ids_to_process} ({len(tile_ids_to_process)} out of {len(unique_tile_ids)}): {uu.timestr()}")

    # Determines if the output file names for final versions of outputs should be used
    is_large_run = False
    # is_large_run = True  # For simulating a large run
    if len(tile_ids) > 20:
        is_large_run = True
        main_logger.info(f"Running as large-scale run model: {is_large_run}")

    sys.quit()


    futures = []

    main_logger.info(f"Starting processing: {uu.timestr()}")

    for var in vars_to_process:

        for year_idx in range(years_to_process):

            for tile_id in tile_ids_to_process:

                future = client.submit(extract_10x10,
                                       var, year_idx, tile_id, zarr_path, output_base)
                futures.append(future)

    results = client.gather(futures)


    main_logger.info(f"All 10x10 aggregation completed: {uu.timestr()}")
    for r in results[0:5]:
        print(f"  → {r}")


    ### Step 6: Combine and upload logs

    # Worker logs are not aggregated if doing a local run (since there are no workers)
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)

        # Adds the workers' logs to the main log and uploads to s3
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)

        uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats and worker log compilation", main_logger)

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
    parser.add_argument('-tpsc', '--test_print_stats_chunk', nargs=4, type=float, help='Bounding box to print rechunked zarr stats from: W, S, E, N (degrees)')
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
    test_print_stats_chunk = args.test_print_stats_chunk
    model_chunk_stats_table_name = args.model_chunk_stats_table_name
    log_note = args.log_note

    run_local = args.run_local
    no_log = args.no_log
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, input_date, run_local, no_log, no_upload, model_chunk_stats_table_name, chunk_shapefile_uri, bounding_box=bounding_box,
         test_print_stats_chunk=test_print_stats_chunk, first_variables_to_process=first_variables_to_process, first_years_to_process=first_years_to_process,
         first_tiles_to_process=first_tiles_to_process, log_note=log_note)
