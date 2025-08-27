"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Can only run on 1x1 degree chunks that do not have the run timestamp in the file name.
The way this builds the input file names, it can't handle filenames with the run timestamp.
It also can't handle chunks smaller than 1x1 degree.
Aggregates in batches because there are so many tasks for this step that it's just more likely to fail part way through
and so having batches allows easier restarting. The tasks are in a pre-determined alphabetical order,
so each batch should have the same contents for any given set of inputs.

Local test:
python -m src.LULUCF.scripts.core_model.3_aggregate_LULUCF_outputs -yr 2000 2024 --first_10x10s_to_process 2 --input_date YYYYMMDD --run_local

Coiled small test:
python -m src.utilities.create_cluster -n 1 -t 1 -m 4 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.core_model.3_aggregate_LULUCF_outputs -cn LULUCF_postprocessing -yr 2000 2024 --first_10x10s_to_process 2 --input_date YYYYMMDD

Coiled large shapefile test (create a cluster with 1 worker, then resize it to 100 workers after local processing is done):
python -m src.utilities.create_cluster -n 1 -t 1 -m 4 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.core_model.3_aggregate_LULUCF_outputs -cn LULUCF_postprocessing -yr 2000 2024 --input_date YYYYMMDD -nw 100 -ln "This is the aggregation of the 1884-chunk run."

Full Coiled run (create a cluster with 1 worker, then resize it to 200 workers after local processing is done):
python -m src.utilities.create_cluster -n 1 -t 1 -m 4 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.core_model.3_aggregate_LULUCF_outputs -cn LULUCF_postprocessing -yr 2000 2024 --input_date YYYYMMDD -nw 200 -ln "This is intended to be the definitive global run."

Notes on optimizing threads/worker: https://app.asana.com/1/25496124013636/task/1206230383901961/comment/1210803828525318?focus=true
Tests of this aggregation and other aggregations show that 1 thread/worker with 4GB workers is low in Coiled credit usage
and runs quickly compared to other configurations.

14256 tiles in 273 folders for 1884-chunk output
"""

import argparse
import fsspec
import sys
import re
from collections import defaultdict
import os
import pandas as pd
import s3fs
import sys
import tempfile
import xarray as xr
import rioxarray
from dask import delayed, compute
from dask.distributed import Client, print

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import resize_cluster

def group_files_into_10x10_tiles(tif_files):
    # Updated regex to extract lat/lon bounds from filename
    pattern = re.compile(r"__(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)__")

    tile_groups = defaultdict(list)

    for f in tif_files:
        match = pattern.search(f)
        if not match:
            continue

        lat_min = int(match.group(1))
        lon_min = int(match.group(2))
        # lat_max = int(match.group(3))  # unused
        # lon_max = int(match.group(4))  # unused

        # Group by 10x10 tiles using lat_min/lon_min
        lat10 = (lat_min // 10) * 10
        lon10 = (lon_min // 10) * 10
        key = (lat10, lon10)

        tile_groups[key].append(f)

    return tile_groups



@delayed
def merge_and_write(group_key, file_list, output_dir, folder_uri=None):

    fs = s3fs.S3FileSystem(anon=False)

    lat10, lon10 = group_key
    folder_id = os.path.basename(folder_uri.strip("/")) if folder_uri else "unknown_folder"
    output_filename = f"{folder_id}__tile_lat_{lat10}_lon_{lon10}.tif"
    s3_output_path = f"{output_dir.rstrip('/')}/{output_filename}"

    # if fs.exists(s3_output_path):
    #     print(f"Skipping {s3_output_path} (already exists)")
    #     return s3_output_path

    print(f"Process: {group_key}")
    # print(file_list)
    # print(output_dir)
    # print(folder_uri)


    try:
        ds = xr.open_mfdataset(
            file_list,
            combine='by_coords',
            parallel=True,
            chunks={'x': 4000, 'y': 4000}
        ).squeeze()

        ds = ds.rio.write_crs("EPSG:4326")

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, output_filename)
            ds.rio.to_raster(local_path, compress="LZW")

            with open(local_path, 'rb') as src_file:
                with fsspec.open(s3_output_path, 'wb') as dst_file:
                    dst_file.write(src_file.read())

        print(f"✅ Uploaded {s3_output_path}")
        return s3_output_path

    except Exception as e:
        print(f"❌ Error processing tile {group_key} in {folder_uri}: {e}")
        return None


@delayed
def merge_folder(first_10x10s_to_process, folder_uri):

    fs = s3fs.S3FileSystem(anon=False)
    files = fs.ls(folder_uri)
    # print(files)
    tif_files = [f"s3://{f}" for f in files if f.endswith(".tif")]
    # print(tif_files)

    # Regex to extract bounding box
    pattern = re.compile(r"__(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)__")
    tile_groups = defaultdict(list)

    for f in tif_files:
        match = pattern.search(f)
        if match:
            lat = int(match.group(1))
            lon = int(match.group(2))
            key = ((lat // 10) * 10, (lon // 10) * 10)
            tile_groups[key].append(f)

    # print(tile_groups)

    output_root = folder_uri.replace("4000_pixels", "40000_pixels")

    # Submit per-tile merge jobs
    results = []
    for group_key, files in tile_groups.items():
        print(f"Merge: {group_key}")
        # print(files)
        result = merge_and_write(group_key, files, output_root, folder_uri=folder_uri)
        results.append(result)

    return results



def main(cluster_name, year_range, input_date, number_of_workers, run_local=False, no_stats=False, no_log=False, no_upload=False,
         first_folders_to_process=None, first_10x10s_to_process=None, log_note=None):


    ### Step 1: Preparation

    # Model stage being run
    stage = 'LULUCF_aggregation_to_10x10_deg'
    model_type = 'standard_model'

    # Runs chunks in batches of specified size.
    # Each batch slows down processing because chunks inevitably lag and that happens more the more batches there are.
    batch_size = 3000
    # batch_size = 15  # For testing batch processing
    # batch_size = 1  # For testing batch processing

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

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    # Starting time for stage
    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Start year: {start_year}; end year: {end_year}")
    main_logger.info(f"Run date: {input_date}")
    main_logger.info(f"Batch size: {batch_size} chunks")
    main_logger.info(f"no_upload: {no_upload}")

    input_folders = ["s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_2/AGC_emission_factor_CO2_only__fraction/standard_model/hybrid_intervals/2001_2005/4000_pixels/20250806/",
                     "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_2/AGC_emission_factor_CO2_only__fraction/standard_model/hybrid_intervals/2006_2010/4000_pixels/20250806/"]


    # Limit if requested
    if first_folders_to_process:
        input_folders = input_folders[:first_folders_to_process]

    print(f"📂 Found {len(input_folders)} folders to process")
    print(input_folders)

    futures = [merge_folder(first_10x10s_to_process, folder)
               for folder in input_folders]

    folder_results = compute(*futures)
    # print(folder_results)
    tile_tasks = [task for sublist in folder_results for task in sublist if task is not None]

    print(f"🧮 Scheduling {len(tile_tasks)} tile tasks")
    tile_results = compute(*tile_tasks)

    main_logger.info(f"✅ Aggregation complete. {len(tile_results)} tiles written.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate 1x1 degree outputs from LULUCF model to 10x10 degree geotifs.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--input_date', help='Date of core model run, in YYYYMMDD')
    parser.add_argument('-yr', '--year_range', nargs=2, type=int, required=True, help='Starting and ending years for model. Start options: 2000, 2015. End options: 2020, 2024.')
    parser.add_argument('-ffol', '--first_folders_to_process', type=int, help='Number of folders to process from input list')
    parser.add_argument('-ften', '--first_10x10s_to_process', type=int, help='Number of chunks to process from input list')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')
    parser.add_argument('-nw', '--number_of_workers', type=int, help='Number of workers to rescale to after local input list processing is done. Optonal')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    input_date = args.input_date
    year_range = args.year_range
    first_folders_to_process = args.first_folders_to_process
    first_10x10s_to_process = args.first_10x10s_to_process
    log_note = args.log_note
    number_of_workers = args.number_of_workers

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    main(cluster_name, year_range, input_date, number_of_workers, run_local, no_stats, no_log, no_upload,
         first_folders_to_process=first_folders_to_process,
         first_10x10s_to_process=first_10x10s_to_process, log_note=log_note)