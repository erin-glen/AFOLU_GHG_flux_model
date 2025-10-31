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
chunk list from the raw and rechunked zarrs. The chunk stats are in the same format as the chunk stats from
numpy arrays from the main model. The chunk stats from the raw and rechunked zarrs can then be compared against the
chunk stats from the numpy arrays to make sure nothing has changed during the transfer from 1x1 geotifs to
raw zarr to rechunked zarr.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test (Dask part does not work because of client.submit()):
python -m src.LULUCF.scripts.vegetation_model.3_create_rechunked_mega_zarr --run_local -fv 1 -fy 1 --test_print_stats_chunk 0 41 1 42 --input_date YYYYMMDD

Small test run:
python -m src.utilities.create_cluster -n 50 -t 1 -m 4 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.3_create_rechunked_mega_zarr -cn vegetation_model -fv 2 -fy 2 --test_print_stats_chunk 0 41 1 42 -bb 0 41 1 42 --input_date YYYYMMDD

Coiled large shapefile test (1884 features):
python -m src.utilities.create_cluster -n 50 -t 1 -m 4 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.3_create_rechunked_mega_zarr -cn vegetation_model --test_print_stats_chunk 0 41 1 42 -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD

Full run:
python -m src.utilities.create_cluster -n 50 -t 1 -m 4 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.3_create_rechunked_mega_zarr -cn vegetation_model --test_print_stats_chunk 0 41 1 42 --input_date YYYYMMDD

Most recent ChatGPT convo: https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6900ce1b-e728-832a-9b87-4702f646da42
"""

import argparse
import numpy as np
import zarr
import fsspec
import xarray as xr
import pandas as pd
import sys
import time
from dask.distributed import Client, print

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import resize_cluster


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


# Checks the stats for a bounding box in a zarr for a given dataset and year
def check_region_stats(store_url, dataset_key, year_idx, target_box):

    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(store_url)
    z = zarr.open(mapper, mode="r")

    lat0, lon0 = uu.latlon_to_global_zarr_indices(target_box["lat_max"], target_box["lon_min"], cn.resolution)
    lat1, lon1 = uu.latlon_to_global_zarr_indices(target_box["lat_min"], target_box["lon_max"], cn.resolution)

    region_array = z[dataset_key][year_idx, lat0:lat1, lon0:lon1]

    # Non-zero pixels in the array
    non_zero_count = np.count_nonzero(region_array)

    # Statement to print
    print_statement = f"      {dataset_key} year {year_idx}: min={region_array.min()}, mean={region_array.mean()}, max={region_array.max()}, non-zero cells={non_zero_count}"
    print(print_statement)

    return_statement = {'min_value': region_array.min(), 'mean_value': region_array.mean(), 'max_value': region_array.max()}
    return(return_statement)

# Calculates stats in 1x1 deg chunk in raw and rechunked zarrs
def zarr_1x1_deg_stats(bounds, var_name, year_idx, raw_path, rechunk_path):

    # lu.print_and_log(f"Getting stats for {var_name} for year {year_idx} for {bounds}: {uu.timestr()}", False, logger_worker)
    # start_time = time.time()

    bounds_str = uu.boundstr(bounds)  # String form of chunk bounds, from e.g., [8, -1, 9, 0] to 8_-1_9_0
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
    year = cn.interval_end_years_annual[year_idx]

    # Bounding box to get stats for, reformatted for zarr extraction
    target_box = {
        "lat_min": bounds[1],
        "lat_max": bounds[3],
        "lon_min": bounds[0],
        "lon_max": bounds[2]
    }

    # The dataset pattern being analyzed, with year and units added
    pattern_with_units = uu.add_units_year_to_pattern(var_name, year)

    lat0, lon0 = uu.latlon_to_global_zarr_indices(target_box["lat_max"], target_box["lon_min"], cn.resolution)
    lat1, lon1 = uu.latlon_to_global_zarr_indices(target_box["lat_min"], target_box["lon_max"], cn.resolution)

    fs = fsspec.filesystem("s3", anon=False)

    # Calculates chunk stats on the chunk of the zarr.
    # Rather than encoding rows as input or output layer, they are encoded by whether they are raw or rechunked zarr
    # since all of these are outputs.
    # Chunk stats are dictionaries.
    raw_mapper = fs.get_mapper(raw_path)
    raw_zarr = zarr.open(raw_mapper, mode="r")
    raw_array = raw_zarr[var_name][year_idx, lat0:lat1, lon0:lon1]
    chunk_stats_raw = uu.calculate_stats(raw_array, pattern_with_units, bounds_str, tile_id, 'raw_zarr')

    rechunk_mapper = fs.get_mapper(rechunk_path)
    rechunk_zarr = zarr.open(rechunk_mapper, mode="r")
    rechunk_array = rechunk_zarr[var_name][year_idx, lat0:lat1, lon0:lon1]
    chunk_stats_rechunk = uu.calculate_stats(rechunk_array, pattern_with_units, bounds_str, tile_id, 'rechunked_zarr')

    # end_time = time.time()
    # lu.print_and_log(f"  Calculated stats for {pattern_with_units} for {year} for {bounds} in {round(end_time - start_time)} seconds: {uu.timestr()}", False, logger_worker)

    # print("chunk_stats_raw:", chunk_stats_raw)
    # print("chunk_stats_rechunk:", chunk_stats_rechunk)

    # Returns the chunk stats from the raw and rechunked zarrs as separate dictionaries
    return chunk_stats_raw, chunk_stats_rechunk


# Parallelizes stats calculating in 1x1 deg chunks in raw and rechunked zarrs for a given dataset-year
def run_parallel_stats(client, chunk_list, var, year_idx, raw_path, dest_path):

    futures = []

    # Iterates through all chunks in the list for a given dataset-year
    for chunk in chunk_list:
        future = client.submit(zarr_1x1_deg_stats,
                               chunk, var, year_idx, raw_path, dest_path, retries=2)
        futures.append(future)

    # results is a list of chunks containing tuples of raw and rechunked stats, each of which is a dictionary
    results = client.gather(futures)

    # Initializes empty lists for raw_zarr and rechunked_zarr
    raw_zarr_stats_list = []
    rechunked_zarr_stats_list = []

    # Iterates through the tuples
    # Per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6903c69f-8104-8321-964b-0a4f561cd8e2
    for raw_dict, rechunked_dict in results:
        if raw_dict.get("in_out") == "raw_zarr":
            raw_zarr_stats_list.append(raw_dict)
        if rechunked_dict.get("in_out") == "rechunked_zarr":
            rechunked_zarr_stats_list.append(rechunked_dict)

    # print("results_raw_zarr_stats:", raw_zarr_stats_list)
    # print("results_rechunk_zarr_stats:", rechunked_zarr_stats_list)

    return raw_zarr_stats_list, rechunked_zarr_stats_list



def main(cluster_name, input_date, run_local, no_stats, no_log, chunk_shapefile_uri=False, bounding_box=None,
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

    # Creates s3 paths for the raw mega-zarr
    raw_mega_zarr_path = uu.create_mega_zarr_paths(cn.chunk_dims, 'annual', model_type, input_date)
    rechunked_mega_zarr_path = uu.create_mega_zarr_paths(cn.zarr_pixel_chunks, 'annual', model_type, input_date)

    main_logger.info(f"Raw mega-zarr path: {raw_mega_zarr_path}")
    main_logger.info(f"Rechunked mega-zarr path: {rechunked_mega_zarr_path}")
    main_logger.info(f"Number of years to rechunk: {first_years_to_process}")
    main_logger.info(f"Test chunk (to print stats): {test_print_stats_chunk}")

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
    main_logger.info(f"Variables to rechunk and get chunk stats for: {vars_to_process}")

    if first_years_to_process:
        years_to_process = first_years_to_process
    else:
        years_to_process = len(cn.interval_end_years_annual)
    main_logger.info(f"Years to rechunk and get chunk stats for: {years_to_process}")

    lat_size = int(180 / cn.resolution)  # 720000 rows
    lon_size = int(360 / cn.resolution)  # 1440000 columns


    ### Step 2: Create metadata-only chunk=10000x10000 mega-zarr

    start_time = uu.timestr()

    # Creates a metadata-only rechunked zarr that will be populated with rechunked data copied in
    uu.initialize_global_mega_zarr(rechunked_mega_zarr_path, vars_to_process, years_to_process,
                                   (1, cn.zarr_pixel_chunks, cn.zarr_pixel_chunks), main_logger)

    # fs = fsspec.filesystem("s3", anon=False)
    # source_mapper = fs.get_mapper(rechunked_mega_zarr_path)
    # ds = xr.open_zarr(source_mapper, consolidated=False)
    # print(ds)


    ### Step 3: Copy from chunk=4000x4000 zarr to chunk=10000x10000 zarr and obtain chunk stats
    ### for the raw and rechunked zarrs

    main_logger.info(f"Starting rechunk transfers and rechunk stats: {uu.timestr()}")

    # Separate lists of chunk stats from raw and rechunked zarrs
    chunk_stats_raw_zarr = []
    chunk_stats_rechunked_zarr = []

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
            main_logger.info(f"  Starting stats of raw and rechunked {var_name} for year {year}: {uu.timestr()}")
            year_start_time = time.time()

            chunk_stats_variable_year_raw_zarr, chunk_stats_variable_year_rechunked_zarr = run_parallel_stats(
                client=client,
                chunk_list=chunk_list,
                var=var_name,
                year_idx=year_idx,
                raw_path=raw_mega_zarr_path,
                dest_path=rechunked_mega_zarr_path,
            )
            year_end_time = time.time()
            main_logger.info(f"    Got stats for {var_name} for year {year} in {round(year_end_time - year_start_time)} seconds: {uu.timestr()}")

            chunk_stats_raw_zarr.append(chunk_stats_variable_year_raw_zarr)
            chunk_stats_rechunked_zarr.append(chunk_stats_variable_year_rechunked_zarr)

        var_end_time = time.time()
        main_logger.info(f"  Processed {var_name} in {round(var_end_time - var_start_time)} seconds: {uu.timestr()}")

    # Chunk stats for all datasets-years is a nested list (each dataset-year is a list in the combined list).
    # This flattens all dataset-year chunk stats into flat lists.
    chunk_stats_raw_zarr = uu.flatten_list(chunk_stats_raw_zarr)
    chunk_stats_rechunked_zarr = uu.flatten_list(chunk_stats_rechunked_zarr)

    # print("chunk_stats_raw_zarr:", chunk_stats_raw_zarr)
    # print("chunk_stats_rechunked_zarr:", chunk_stats_rechunked_zarr)


    ### Step 4: Compare original model and zarr chunk stats and process logs

    # Resizes cluster down to 1 worker for chunk stats and log aggregation since that only needs a minimal remainder of the
    # cluster, not all the workers.
    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster(cluster_name, 1)

    if (not no_stats):
        raw_stage = stage.replace("rechunk", "raw")
        raw_zarr_chunk_stats_path = uu.compile_1x1_chunk_stats(chunk_stats_raw_zarr, chunk_shapefile_uri, raw_stage, False, main_logger)
        rechunked_zarr_chunk_stats_path = uu.compile_1x1_chunk_stats(chunk_stats_rechunked_zarr, chunk_shapefile_uri, stage, False, main_logger)

        print(raw_zarr_chunk_stats_path)
        print(rechunked_zarr_chunk_stats_path)

        model_stats_path = "chunk_stats/vegetation_fluxes_1x1_chunk_statistics_20251027_16_16_26__v1_0_2_1884_chunk_run__KEEP.xlsx"
        zarr_comparison_stats_path = "chunk_stats/vegetation_fluxes_1x1_chunk_statistics_20251027_16_16_26__v1_0_2_1884_chunk_run__KEEP__zarr_comparison.xlsx"

        rechunked_zarr_chunk_stats_path = "chunk_stats/rechunk_global_mega_zarr_1x1_chunk_statistics_20251030_16_48_20.xlsx"

        tables_to_compare = ['gross_outputs_1x1', 'other_outputs_1x1', 'net_outputs_1x1']

        # From https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6903d1dd-555c-8321-8547-0aa4772c9878
        with pd.ExcelWriter(zarr_comparison_stats_path, engine='openpyxl', mode='w') as writer:
            for table in tables_to_compare:

                main_logger.info(f"Processing table {table}: {uu.timestr()}")

                main_logger.info(f"Reading model and zarr tables: {uu.timestr()}")
                model_table = pd.read_excel(model_stats_path, sheet_name=table)
                rechunked_zarr_table = pd.read_excel(rechunked_zarr_chunk_stats_path, sheet_name=table)

                # Step 1: Selects only the needed columns from rechunked_zarr_table
                main_logger.info(f"Subsetting zarr table to numeric columns: {uu.timestr()}")
                zarr_subset_table = rechunked_zarr_table[['chunk_name', 'min_value', 'mean_value', 'max_value', 'count_value']].copy()

                # Step 2: Renames columns in raw_subset to distinguish them after merge
                main_logger.info(f"Renaming zarr columns: {uu.timestr()}")
                zarr_subset_table = zarr_subset_table.rename(columns={
                    'min_value': 'min_value_zarr',
                    'mean_value': 'mean_value_zarr',
                    'max_value': 'max_value_zarr',
                    'count_value': 'count_value_zarr'
                })

                # Step 3: Converts all zarr value columns to numeric, coercing errors to NaN
                main_logger.info(f"Converting zarr columns to numeric: {uu.timestr()}")
                for col in ['min_value_zarr', 'mean_value_zarr', 'max_value_zarr', 'count_value_zarr']:
                    zarr_subset_table[col] = pd.to_numeric(zarr_subset_table[col], errors='coerce')

                # Step 4: Merges with model_gross on 'chunk_name', left join
                main_logger.info(f"Merging zarr data to original model data: {uu.timestr()}")
                model_table = model_table.merge(zarr_subset_table, on='chunk_name', how='left')

                # Step 5: Calculates differences and store in new columns
                main_logger.info(f"Calculating differences: {uu.timestr()}")
                model_table['min_value_diff'] = model_table['min_value'] - model_table['min_value_zarr']
                model_table['mean_value_diff'] = model_table['mean_value'] - model_table['mean_value_zarr']
                model_table['max_value_diff'] = model_table['max_value'] - model_table['max_value_zarr']
                model_table['count_value_diff'] = model_table['count_value'] - model_table['count_value_zarr']
                print(model_table.head())

                # Step 6: Writes outputs
                model_table.to_excel(writer, sheet_name=table, index=False)

                # Optional: log success
                main_logger.info(f"Saved {table} sheet to {zarr_comparison_stats_path}: {uu.timestr()}")

        uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats", main_logger)


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
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
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
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log

    # Create the cluster with command line arguments
    main(cluster_name, input_date, run_local, no_stats, no_log, chunk_shapefile_uri, bounding_box=bounding_box,
         test_print_stats_chunk=test_print_stats_chunk, first_variables_to_process=first_variables_to_process, first_years_to_process=first_years_to_process,
         first_chunks=first_chunks, log_note=log_note)