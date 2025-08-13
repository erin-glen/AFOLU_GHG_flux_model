"""
To run:

python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn LULUCF_mineral_soil
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.create_stock_and_stock_change_10x10s \
  -cn LULUCF_mineral_soil -bb 110 -10 120 0 -cs 10 -yr 2000 2024 --input_date YYYYMMDD
"""

import argparse
import os
import dask
import numpy as np
import psutil
import rasterio
import tempfile
import s3fs
import time
from rasterio.windows import Window
from rasterio.transform import Affine

from dask.distributed import print
import coiled

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import resize_cluster

TILE_DEGREES = 10
fs = s3fs.S3FileSystem(anon=False)

SOC_COGS = {
    # "2000_2005": "https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics/oc_iso.10694.1995.mg.cm3_m_30m_b0cm..30cm_20000101_20051231_g_epsg.4326_v20250204.tif",
    # "2005_2010": "https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics/oc_iso.10694.1995.mg.cm3_m_30m_b0cm..30cm_20050101_20101231_g_epsg.4326_v20250204.tif",
    "2010_2015": "https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics/oc_iso.10694.1995.mg.cm3_m_30m_b0cm..30cm_20100101_20151231_g_epsg.4326_v20250204.tif",
    "2015_2020": "https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics/oc_iso.10694.1995.mg.cm3_m_30m_b0cm..30cm_20150101_20201231_g_epsg.4326_v20250204.tif"
}


def extract_tile_from_global_cog(cog_url, bounds, chunk_length_pixels):
    with rasterio.Env(AWS_NO_SIGN_REQUEST='YES'):
        with rasterio.open(cog_url) as src:
            col_start = round((bounds[0] - src.bounds.left) / cn.resolution)
            row_start = round((src.bounds.top - bounds[3]) / cn.resolution)

            window = Window(col_start, row_start, chunk_length_pixels, chunk_length_pixels)
            data = src.read(1, window=window)

            transform = Affine.translation(bounds[0], bounds[3]) * Affine.scale(cn.resolution, -cn.resolution)

            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": chunk_length_pixels,
                "width": chunk_length_pixels,
                "transform": transform,
                "compress": "DEFLATE"
            })

            return data, out_meta


def create_soil_C_density_and_change_tiles(bounds, is_final, stage):

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []

    process = psutil.Process(os.getpid())

    logger_worker = lu.setup_logging_worker()

    density_start_time = time.time()

    uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_final, logger_worker)

    bounds_str = uu.boundstr(bounds)  # String form of chunk bounds, from e.g., [8, -1, 9, 0] to 8_-1_9_0
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)

    soc_data = {}

    ### Part 1: Calculate densities at each interval

    for year_range, url in SOC_COGS.items():
        lu.print_and_log(f"Downloading SOC density for {year_range} for {tile_id}: {uu.timestr()}", False, logger_worker)
        data, profile = extract_tile_from_global_cog(url, bounds, chunk_length_pixels)
        soc_data[year_range] = data

        lu.print_and_log(f"  Saving {year_range} for {tile_id}: {uu.timestr()}", False, logger_worker)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            with rasterio.open(tmp.name, "w", **profile) as dst:
                dst.write(data, 1)

            # Sets up s3 destination folder. File name depends on whether output is 40000x40000 pixels or smaller.
            if chunk_length_pixels == cn.full_raster_dims:
                s3_key = f"{cn.min_soil_density_dir}{tile_id}_{cn.min_soil_density_pattern}__{year_range}.tif"
            else:
                s3_key = f"{cn.min_soil_density_dir}{tile_id}_{bounds_str}_{cn.min_soil_density_pattern}__{year_range}__{uu.timestr()}.tif"
            s3_key = s3_key.replace("s3://", "")  # For s3fs access
            s3_key = s3_key.replace("START_END", year_range)
            s3_key = s3_key.replace("PER_HA_OR_PIXEL", "per_ha")
            s3_key = s3_key.replace("CHUNK_SIZE_pixels", f"{chunk_length_pixels}_pixels")

            lu.print_and_log(f"  Uploading {s3_key}: {uu.timestr()}", False, logger_worker)
            fs.put(tmp.name, s3_key)
            lu.print_and_log(f"  Uploaded {s3_key}: {uu.timestr()}", False, logger_worker)

    density_end_time = time.time()
    lu.print_and_log(f"  {bounds_str} took {round(density_end_time - density_start_time)} seconds: {uu.timestr()}", False, logger_worker)


    ### Part 2: Calculate density changes between adjacent intervals

    # Compute and save deltas
    year_range = list(SOC_COGS.keys())
    lu.print_and_log(f"Calculating SOC deltas for {tile_id}: {uu.timestr()}", False, logger_worker)
    for i in range(len(year_range) - 1):
        y1 = year_range[i]
        y2 = year_range[i + 1]
        year_diff = int(y2[:4])-int(y1[:4])
        delta = (soc_data[y2].astype(np.int16) - soc_data[y1].astype(np.int16))/year_diff  # Interval arrays must be unsigned so difference can be negative

        # Density change has to be float32 because it's not an integer
        delta_int = delta.astype(np.float32)

        delta_profile = profile.copy()
        delta_profile.update({
            "dtype": "float32"
        })

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            with rasterio.open(tmp.name, "w", **delta_profile) as dst:
                dst.write(delta_int, 1)

            # Sets up s3 destination folder. File name depends on whether output is 40000x40000 pixels or smaller.
            if chunk_length_pixels == cn.full_raster_dims:
                delta_key = f"{cn.min_soil_density_dir}{tile_id}_{cn.min_soil_density_pattern}_{y1}_{y2}.tif"
            else:
                delta_key = f"{cn.min_soil_change_dir}{tile_id}_{bounds_str}_{cn.min_soil_change_pattern}_{year_range}__{uu.timestr()}.tif"
            delta_key = delta_key.replace("s3://", "")  # For s3fs access
            delta_key = delta_key.replace("START_END", f"{y1}_{y2}")
            delta_key = delta_key.replace("PER_HA_OR_PIXEL", "per_ha")
            delta_key = delta_key.replace("CHUNK_SIZE_pixels", f"{chunk_length_pixels}_pixels")

            fs.put(tmp.name, delta_key)
            lu.print_and_log(f"  Uploaded {delta_key}: {uu.timestr()}", False, logger_worker)


    ### Part 3: Calculate density change between start and end intervals

    return_message = f"Success for {bounds_str}: {uu.timestr()}"

    # Removes task tracking file from S3 once task is successful
    uu.delete_s3_task_file(stage, bounds, is_final, logger_worker)

    return return_message  # Return both the success message and the statistics
    # return return_message, chunk_stats  # Return both the success message and the statistics


def get_tile_grid():
    grid = []
    for west in range(-180, 180, TILE_DEGREES):
        for north in range(80, -60, -TILE_DEGREES):  # Note: descending latitude (north to south)
    # for west in range(110, 120, TILE_DEGREES):
    #     for north in range(0, -10, -TILE_DEGREES):  # Note: descending latitude (north to south)
            grid.append((west, north))
    return grid


def main(cluster_name, run_date, year_range, run_local=False, no_stats=False, no_log=False, no_upload=False,
         chunk_shapefile_uri=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = 'soil_carbon_densities_and_changes'
    model_type = 'standard_model'

    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    start_time = uu.timestr() # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Run date: {run_date}")
    main_logger.info(f"no_upload: {no_upload}")

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size, None, None, main_logger)

    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_final = False
    # is_final = True  # For simulating a large run
    if len(chunk_list) > 20:
        is_final = True
        main_logger.info("Running as final model.")

    # Makes a txt for each task in the list. These are deleted as tasks are completed.
    main_logger.info("Creating task txts in s3...")
    uu.create_s3_task_files(stage, chunk_list)


    ### Step 2: Create outputs

    output_delayed_tasks = [dask.delayed(create_soil_C_density_and_change_tiles)(chunk, is_final, stage) for chunk in chunk_list]

    results = dask.compute(*output_delayed_tasks)


    print("All done.")
    for res in results:
        print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and compute SOC stock + change in 10x10 deg tiles.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--run_date', help='Date of run, in YYYYMMDD')
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
    run_date = args.run_date
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
    main(cluster_name, run_date, year_range, run_local, no_stats, no_log, no_upload, chunk_shapefile_uri,
         bounding_box=bounding_box, chunk_size=chunk_size, first_chunks=first_chunks, log_note=log_note)
