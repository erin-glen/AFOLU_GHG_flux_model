"""
Rechunks the global mega-zarr (zarr group with all variables/datasets and years) from 4000x4000 pixels into
10000x10000 pixels by copying data from the populated raw zarr to the metadata-only rechunked zarr chunk by chunk
for each dataset-year combination.

I went through many iterations for how to rechunk the zarr and wound up with this one, which is kind of cumbersome and
manual and doesn't use native Dask/xarray abilities.
First, I tried using the rechunker library but that requires using zarr v2 file format and I need zarr v3 file format
for zonal stats with flox. So, rechunker was out.
Then, I tried a variety of systems using Dask and xarray. Those mostly ran into problems with having graph sizes that
were too large because they were trying to arrange the rechunking of an entire dataset-year at once.
Then, I tried rechunking by dataset-year-latitude band to keep the Dask graph sizes lower. I think that may have worked
if I had set it up differently but I configured it in a way that had multiple workers writing 4000x4000 chunks to the
same 10000x1000 destination chunk at the same time, which zarr can't handle. (Can't write to the same chunk concurrently,
I learned.) If I had tried transferring in chunks of 10000x10000 instead of 4000x4000, I think this would have
been similar to what I am ultimately doing, except with the added complication of iterating by latitude band.
BTW, this situation didn't show any errors; the failure was silent. It was only noticeable because I was calculating
chunk stats (min, mean max) for test chunks in the raw and rechunked zarr and noticed that they were different compared
to the numpy-based chunk stats spreadsheet directly from the model. It turned out that whenever I tried concurrently
writing to a single chunk with multiple workers, it would just drop data (presumably because of conflict over write access
to the chunk), so the destination chunk would be missing some or all of its data.
Then, I tried just creating a chunk=10000x10000 Zarr up front and having the vegetation model write to that.
That didn't work for the same reason; I was writing multiple 4000x4000 chunks to a single 10000x10000 destination
chunk concurrently. It had the same missing data issue as above (in comparison with the model output chunk stats
spreadsheet).
Finally, I settled on this approach, which grabs 10000x10000 pixel chunks from a raw zarr dataset-year
and writes them into the rechunked one. The 10000x10000 chunks for a given dataset-year are transferred in parallel
but each chunk is written by only one worker. It then iterates through the datasets and years in a nested for loop
(which is at all Dask or xarray).

After transferring a single dataset-year to the rechunked zarr, it then gets 1x1 deg chunk stats for the provided
chunk list from the rechunked zarr. The chunk stats are in the same format as the chunk stats from
numpy arrays from the main model. It compares the min, mean, max and pixel count for each zarr chunk against the
original model chunk stats and prints any rows that have a difference in any metric above a specified tolerance.
After the chunk stats for each dataset-year combination are compared, a spreadsheet or parquet file is
written with the cumulative comparison between model and rechunked zarr stats. This chunk stats comparison
allows confirmation that data wasn't modified or lost during the transfer from geotifs to rechunked zarr.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test (Dask part does not work because of client.submit()):
python -m src.LULUCF.scripts.vegetation_model.3_create_rechunked_mega_zarr --run_local -fv 1 -fy 1 --test_print_stats_chunk 0 41 1 42 -mcstn vegetation_fluxes_1x1_chunk_statistics_20251027_16_16_26__v1_0_2_1884_chunk_run__KEEP.xlsx --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.3_create_rechunked_mega_zarr --run_local -fv 1 -fy 1 --test_print_stats_chunk 0 41 1 42 -mcstn parquet_20250921_17_33_57__XYX/LULUCF_fluxes_20250921_17_33_45_XYZ --input_date YYYYMMDD

Small test run:
python -m src.utilities.create_cluster -n 50 -t 1 -m 4 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.3_create_rechunked_mega_zarr -cn vegetation_model -fv 2 -fy 2 --test_print_stats_chunk 0 41 1 42 -bb 0 41 1 42 -mcstn vegetation_fluxes_1x1_chunk_statistics_20251027_16_16_26__v1_0_2_1884_chunk_run__KEEP.xlsx --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.3_create_rechunked_mega_zarr -cn vegetation_model -fv 2 -fy 2 --test_print_stats_chunk 0 41 1 42 -bb 0 41 1 42 -mcstn parquet_20250921_17_33_57__XYX/LULUCF_fluxes_20250921_17_33_45_XYZ --input_date YYYYMMDD

Coiled large shapefile test (1884 features):
python -m src.utilities.create_cluster -n 50 -t 1 -m 4 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.3_create_rechunked_mega_zarr -cn vegetation_model --test_print_stats_chunk 0 41 1 42 -mcstn vegetation_fluxes_1x1_chunk_statistics_20251027_16_16_26__v1_0_2_1884_chunk_run__KEEP.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.3_create_rechunked_mega_zarr -cn vegetation_model --test_print_stats_chunk 0 41 1 42 -mcstn parquet_20250921_17_33_57__XYX/LULUCF_fluxes_20250921_17_33_45_XYZ -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD

Full run:
python -m src.utilities.create_cluster -n 50 -t 1 -m 4 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.3_create_rechunked_mega_zarr -cn vegetation_model --test_print_stats_chunk 0 41 1 42 -mcstn parquet_20250921_17_33_57__XYX/LULUCF_fluxes_20250921_17_33_45_XYZ -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --input_date YYYYMMDD -ln "This is the definitive rechunking run."

Most recent ChatGPT convo about rechunking approach: https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6900ce1b-e728-832a-9b87-4702f646da42
"""

import argparse
import numpy as np
import xarray as xr
import zarr
import fsspec
import pandas as pd
import boto3
import os
import sys
import time
from dask.distributed import Client, print

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import resize_cluster

pd.set_option('display.float_format', '{:.6e}'.format)


# Copies the 10000x10000 chunk
def copy_block(var, year_idx, y0, y1, x0, x1, raw_path, dest_path):

    # print(f"Transferring {var} for {year_idx} for {y0}:{y1}, {x0}:{x1}: {uu.timestr()}")
    # start_time = time.time()

    fs = fsspec.filesystem("s3", anon=False)
    raw_store = zarr.open_group(fs.get_mapper(raw_path), mode="r")
    rechunked_store = zarr.open_group(fs.get_mapper(dest_path), mode="r+")

    year = cn.interval_end_years_annual[year_idx]

    block = raw_store[var][year_idx, y0:y1, x0:x1]
    rechunked_store[var][year_idx, y0:y1, x0:x1] = block

    # end_time = time.time()
    # print(f"  Transferred {var} for {year_idx} for y={y0}:{y1}, x={x0}:{x1} in {round(end_time - start_time)} seconds: {uu.timestr()}")

    return f"Copied {var} year {year} region y={y0}:{y1}, x={x0}:{x1}"


# Parallelizes copy across 10000x10000 chunks for a given dataset-year combination
def run_parallel_copy(
    client: Client,
    var: str,
    year_idx: int,
    ny: int,
    nx: int,
    block_size: int,
    raw_path: str,
    dest_path: str,
):
    futures = []

    for y0 in range(0, ny, block_size):
        y1 = min(y0 + block_size, ny)
        for x0 in range(0, nx, block_size):
            x1 = min(x0 + block_size, nx)

            future = client.submit(copy_block,
                                   var, year_idx, y0, y1, x0, x1, raw_path, dest_path, retries=2)
            futures.append(future)

    results = client.gather(futures)

    # for r in results[:5]:
    #     print(r)
    # print(f"  All blocks copied for {var} for year {year_idx}: {uu.timestr()}")


def main(cluster_name, input_date, run_local, no_log, model_chunk_stats_table_name, chunk_shapefile_uri=False, bounding_box=None,
         test_print_stats_chunk=None, first_variables_to_process=None, first_years_to_process=None,
         first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = 'rechunk_global_mega_zarr'
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
    raw_mega_zarr_path = uu.create_mega_zarr_paths(cn.chunk_dims, 'annual', model_type, input_date)
    rechunked_mega_zarr_path = uu.create_mega_zarr_paths(cn.zarr_pixel_chunks, 'annual', model_type, input_date)

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
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size_deg, first_chunks, fishnet_iso_df, main_logger)
    main_logger.info(f"Chunks to get stats for from rechunked mega-zarr: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_large_run = False
    # is_large_run = True  # For simulating a large run
    if len(chunk_list) > 20:
        is_large_run = True
        main_logger.info(f"Running as large-scale run model: {is_large_run}")

    # Limits the processed variables to the supplied number (for testing)
    if first_variables_to_process:
        vars_to_process = cn.outputs_to_zarr[0:first_variables_to_process]
    else:
        vars_to_process = cn.outputs_to_zarr
    main_logger.info(f"Variables to rechunk and compare chunk stats for: {vars_to_process}")

    if first_years_to_process:
        years_to_process = first_years_to_process
    else:
        years_to_process = len(cn.interval_end_years_annual)
    main_logger.info(f"Years to rechunk and compare chunk stats for: {years_to_process}")

    lat_size = int(180 / cn.resolution)  # 720000 rows
    lon_size = int(360 / cn.resolution)  # 1440000 columns


    ### Step 2: Create metadata-only chunk=10000x10000 mega-zarr

    start_time = uu.timestr()

    # Creates a metadata-only rechunked zarr that will be populated with rechunked data copied in
    uu.initialize_global_mega_zarr(rechunked_mega_zarr_path, vars_to_process, years_to_process,
                                   (1, cn.zarr_pixel_chunks, cn.zarr_pixel_chunks), main_logger)

    fs = fsspec.filesystem("s3", anon=False)
    source_mapper = fs.get_mapper(rechunked_mega_zarr_path)
    ds = xr.open_zarr(source_mapper, consolidated=False)
    print(ds)


    ### Step 3: Prepare model chunk stats for comparison with zarr chunk stats

    main_logger.info(f"Reading local model chunk stats tables: {uu.timestr()}")
    model_chunk_stats_path = os.path.join(cn.local_chunk_stats_path, model_chunk_stats_table_name)

    # Text added to output chunk stats tables (Excel or Parquet)
    comparison_insert = "_rechunk_zarr_comparison"

    # Separate logic for naming chunk stat comparison outputs if using Excel or Parquet (very large runs)
    if "xlsx" in model_chunk_stats_table_name:
        main_logger.info(f"Reading model chunk stats from local file: {model_chunk_stats_path}")
        chunk_stats_model_gross = pd.read_excel(model_chunk_stats_path, sheet_name='gross_outputs_1x1')
        chunk_stats_model_other = pd.read_excel(model_chunk_stats_path, sheet_name='other_outputs_1x1')
        chunk_stats_model_net = pd.read_excel(model_chunk_stats_path, sheet_name='net_outputs_1x1')

        # Name of output Excel spreadsheet with chunk stats comparisons
        name, ext = os.path.splitext(model_chunk_stats_path)
        zarr_comparison_stats_path = f"{name}{comparison_insert}_{uu.timestr()}{ext}"
        zarr_comparison_stats_name = os.path.basename(zarr_comparison_stats_path)
        # print(zarr_comparison_stats_path)
        # print(zarr_comparison_stats_name)

    elif "parquet" in model_chunk_stats_table_name:
        main_logger.info(f"Reading parquet tables from local files: {model_chunk_stats_path}")
        chunk_stats_model_gross = pd.read_parquet(f"{model_chunk_stats_path}__gross_outputs_1x1.parquet")
        chunk_stats_model_other = pd.read_parquet(f"{model_chunk_stats_path}__other_outputs_1x1.parquet")
        chunk_stats_model_net = pd.read_parquet(f"{model_chunk_stats_path}__net_outputs_1x1.parquet")

        # Names of output parquet tables with chunk stats comparisons
        zarr_comparison_stats_gross_name = f"{model_chunk_stats_path}__gross_outputs_1x1_{comparison_insert}_{uu.timestr()}.parquet"
        zarr_comparison_stats_other_name = f"{model_chunk_stats_path}__other_outputs_1x1_{comparison_insert}_{uu.timestr()}.parquet"
        zarr_comparison_stats_net_name = f"{model_chunk_stats_path}__net_outputs_1x1_{comparison_insert}_{uu.timestr()}.parquet"
        zarr_comparison_stats_path = [zarr_comparison_stats_gross_name, zarr_comparison_stats_other_name, zarr_comparison_stats_net_name]
        zarr_comparison_stats_name = [os.path.basename(stats_path) for stats_path in zarr_comparison_stats_path]
        # print(zarr_comparison_stats_path)
        # print(zarr_comparison_stats_name)

    else:
        sys.exit("Table type not found")

    # The model chunk stat tables
    tables_to_compare_dict = {"model_gross": chunk_stats_model_gross,
                              "model_other": chunk_stats_model_other,
                              "model_net": chunk_stats_model_net}


    ### Step 4: Copy from chunk=4000x4000 zarr to chunk=10000x10000 zarr and obtain chunk stats
    ### for the rechunked zarr

    main_logger.info(f"Starting rechunk transfers and rechunk stats: {uu.timestr()}")

    # List of dataframes with original and zarr chunk stats and their difference for each dataset-year combination
    all_merged_tables = []

    # Number of chunks with differences between original and zarr exceeding tolerance
    chunks_count_exceeding_total = 0

    # Iterates through variables/datasets. Each chunk=10000x10000 is transferred by just one task/worker
    # so that multiple workers aren't touching the same zarr chunk at the same time.
    for var_name in vars_to_process:

        main_logger.info(f"Starting {var_name}: {uu.timestr()}")
        var_start_time = time.time()

        # Iterates through years
        for year_idx in range(years_to_process):

            year = cn.interval_end_years_annual[year_idx]

            main_logger.info(f"  Starting transfer of {var_name} for year {year}: {uu.timestr()}")
            year_start_time = time.time()

            # Transfers to rechunked zarr for a given dataset-year in parallel
            run_parallel_copy(
                client=client,
                var=var_name,
                year_idx=year_idx,
                ny=lat_size,
                nx=lon_size,
                block_size=cn.zarr_pixel_chunks,
                raw_path=raw_mega_zarr_path,
                dest_path=rechunked_mega_zarr_path,
            )
            year_end_time = time.time()
            main_logger.info(f"    Transferred {var_name} for year {year} in {round(year_end_time - year_start_time)} seconds: {uu.timestr()}")

            # Only prints test chunk stats if selected
            if test_print_stats_chunk:

                # Converts chunk bounds to the form needed for getting chunk stats
                target_box = {
                    "lat_min": test_print_stats_chunk[1],
                    "lat_max": test_print_stats_chunk[3],
                    "lon_min": test_print_stats_chunk[0],
                    "lon_max": test_print_stats_chunk[2]
                }

                main_logger.info(f"    Original (4000x4000) zarr:")
                uu.check_region_stats(raw_mega_zarr_path, var_name, year_idx, target_box, main_logger)
                main_logger.info(f"    Rechunked (10000x10000) zarr:")
                uu.check_region_stats(rechunked_mega_zarr_path, var_name, year_idx, target_box, main_logger)

            # Gets stats for selected 1x1 deg chunks in raw and rechunked zarrs
            main_logger.info(f"  Starting zarr stats for {var_name} for year {year}: {uu.timestr()}")
            year_start_time = time.time()

            # Runs chunk stats for a dataset-year in the zarr in parallel
            chunk_stats_variable_year_rechunked_zarr = uu.run_parallel_stats(
                client=client,
                chunk_list=chunk_list,
                var=var_name,
                year_idx=year_idx,
                zarr_path=rechunked_mega_zarr_path,
            )
            year_end_time = time.time()
            main_logger.info(f"    Got zarr stats for {var_name} for year {year} in {round(year_end_time - year_start_time)} seconds: {uu.timestr()}")

            all_merged_tables, chunks_count_exceeding = uu.compare_dataset_year_chunk_stats(all_merged_tables,
                                                                 chunk_stats_variable_year_rechunked_zarr,
                                                                 main_logger, tables_to_compare_dict, var_name, year,
                                                                 zarr_comparison_stats_path)

            chunks_count_exceeding_total += chunks_count_exceeding

        var_end_time = time.time()
        main_logger.info(f"  Processed {var_name}  in {round(var_end_time - var_start_time)} seconds: {uu.timestr()}")


    ### Step 5: All iterations done. Tallies chunks that had differences exceeding the tolerance and uploads chunk stats comparisons.

    if chunks_count_exceeding_total > 0:
        main_logger.warning(f"WARNING: {chunks_count_exceeding_total} chunks exceeded difference tolerance! Check log!")
    else:
        main_logger.info(f"{chunks_count_exceeding_total} chunks exceeded the difference tolerance).")

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)

    s3_client = boto3.client("s3")

    if "xlsx" in model_chunk_stats_table_name:
        main_logger.info(f"Uploading chunk stats comparison Excel spreadsheet to s3: {uu.timestr()}")
        try:
            s3_client.upload_file(zarr_comparison_stats_path, cn.short_bucket_prefix, Key=f"{cn.s3_chunk_stats_path}{zarr_comparison_stats_name}")
            main_logger.info(f"Chunk stats spreadsheet uploaded to {cn.full_bucket_prefix}/{cn.s3_chunk_stats_path}{zarr_comparison_stats_name}: {uu.timestr()}")
        except Exception as e:
            main_logger.warning(f"Chunk stats upload to s3 failed: {e}. Continuing without halting.")

    elif "parquet" in model_chunk_stats_table_name:
        main_logger.info(f"Uploading chunk stats comparison parquet tables to s3: {uu.timestr()}")

        for parquet_name, parquet_path in zip(zarr_comparison_stats_name, zarr_comparison_stats_path):
            s3_key = f"{cn.s3_chunk_stats_path}{parquet_name}"
            main_logger.info(f"Uploading {parquet_path} to {s3_key}: {uu.timestr()}")
            s3_client.upload_file(parquet_path, cn.short_bucket_prefix, Key=s3_key)

    else:
        sys.exit("Table type not found")

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats", main_logger)


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
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('-fy', '--first_years_to_process', type=int, help='Number of years to process from raw mega-zarr (for testing)')
    parser.add_argument('-tpsc', '--test_print_stats_chunk', nargs=4, type=float, help='Bounding box to print rechunked zarr stats from: W, S, E, N (degrees)')
    parser.add_argument('-mcstn', '--model_chunk_stats_table_name', help='s3 path for model chunk stats table that will be compared with zarr chunk stats')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    input_date = args.input_date
    bounding_box = args.bounding_box
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_chunks = args.first_chunks
    first_variables_to_process = args.first_variables_to_process
    first_years_to_process = args.first_years_to_process
    test_print_stats_chunk = args.test_print_stats_chunk
    model_chunk_stats_table_name = args.model_chunk_stats_table_name
    log_note = args.log_note

    run_local = args.run_local
    no_log = args.no_log

    # Create the cluster with command line arguments
    main(cluster_name, input_date, run_local, no_log, model_chunk_stats_table_name, chunk_shapefile_uri, bounding_box=bounding_box,
         test_print_stats_chunk=test_print_stats_chunk, first_variables_to_process=first_variables_to_process, first_years_to_process=first_years_to_process,
         first_chunks=first_chunks, log_note=log_note)