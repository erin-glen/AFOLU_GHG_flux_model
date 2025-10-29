"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test (Dask part does not work because of client.submit()):
python -m src.LULUCF.scripts.vegetation_model.3_create_rechunked_mega_zarr --run_local -fv 1 -fy 1 --test_chunk 0 41 1 42 --input_date YYYYMMDD

Test run:
python -m src.utilities.create_cluster -n 50 -t 1 -m 4 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.3_create_rechunked_mega_zarr -cn vegetation_model -fv 2 -fy 2 --test_chunk 0 41 1 42 --input_date YYYYMMDD

Full run:
python -m src.utilities.create_cluster -n 50 -t 1 -m 4 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.3_create_rechunked_mega_zarr -cn vegetation_model --test_chunk 0 41 1 42 --input_date YYYYMMDD

Most recent ChatGPT convo: https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6900ce1b-e728-832a-9b87-4702f646da42
"""

import argparse
import numpy as np
import xarray as xr
import zarr
import fsspec
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

            fut = client.submit(
                copy_block,
                var,
                year_idx,
                y0, y1,
                x0, x1,
                raw_path,
                dest_path,
                retries=2
            )
            futures.append(fut)

    results = client.gather(futures)
    # for r in results[:5]:
    #     print(r)
    # print(f"  All blocks copied for {var} for year {year_idx}: {uu.timestr()}")


def main(cluster_name, input_date, run_local, no_stats, no_log, test_chunk=None,
         first_variables_to_process=None, first_years_to_process=None,
         log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = 'rechunk_global_mega_zarr'
    model_type = 'standard_model'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    start_time = uu.timestr() # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Run date: {input_date}")

    # Creates s3 paths for the raw mega-zarr
    raw_mega_zarr_path = uu.create_mega_zarr_paths(cn.chunk_dims, 'annual', model_type, input_date)
    rechunked_mega_zarr_path = uu.create_mega_zarr_paths(cn.zarr_pixel_chunks, 'annual', model_type, input_date)

    main_logger.info(f"Raw mega-zarr path: {raw_mega_zarr_path}")
    main_logger.info(f"Rechunked mega-zarr path: {rechunked_mega_zarr_path}")
    main_logger.info(f"Number of years to rechunk: {first_years_to_process}")
    main_logger.info(f"Test chunk (to print stats): {test_chunk}")

    test_dataset_keys = [
        "carbon_density__deadwood_C__MgC",
        "carbon_density__litter_C__MgC",
        "gross_emissions__BGC__MgCO2",
        "gross_removals__BGC__MgCO2",
    ]

    if first_variables_to_process:
        vars_to_process = test_dataset_keys[0:first_variables_to_process]  #TODO testing
    else:
        vars_to_process = test_dataset_keys                                #TODO testing
    main_logger.info(f"Variables to rechunk: {vars_to_process}")

    if first_years_to_process:
        years_to_process = first_years_to_process
    else:
        years_to_process = len(cn.interval_end_years_annual)
    main_logger.info(f"Years to rechunk: {years_to_process}")

    lat_size = int(180 / cn.resolution)  # 720000 rows
    lon_size = int(360 / cn.resolution)  # 1440000 columns

    start_time = uu.timestr()


    ### Step 2: Create metadata-only chunk=10000x10000 mega-zarr

    # Creates a metadata-only rechunked zarr that will be populated with rechunked data copied in
    uu.initialize_global_mega_zarr(rechunked_mega_zarr_path, vars_to_process, years_to_process,
                                   (1, cn.zarr_pixel_chunks, cn.zarr_pixel_chunks), main_logger)

    fs = fsspec.filesystem("s3", anon=False)
    source_mapper = fs.get_mapper(rechunked_mega_zarr_path)
    ds = xr.open_zarr(source_mapper, consolidated=False)
    main_logger.info(ds)


    ### Step 3: Copy from chunk=4000x4000 zarr to chunk=10000x10000 zarr

    main_logger.info(f"Starting rechunk transfers: {uu.timestr()}")

    for var_name in vars_to_process:

        main_logger.info(f"Starting {var_name}: {uu.timestr()}")
        var_start_time = time.time()

        for year_idx in range(years_to_process):

            year = cn.interval_end_years_annual[year_idx]

            main_logger.info(f"  Starting {var_name} for year {year}: {uu.timestr()}")
            year_start_time = time.time()

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
            if test_chunk:

                target_box = {
                    "lat_min": test_chunk[1],
                    "lat_max": test_chunk[3],
                    "lon_min": test_chunk[0],
                    "lon_max": test_chunk[2]
                }

                main_logger.info(f"    Original (4000x4000) zarr:")
                uu.check_region_stats(raw_mega_zarr_path, var_name, year_idx, target_box, main_logger)
                main_logger.info(f"    Rechunked (10000x10000) zarr:")
                uu.check_region_stats(rechunked_mega_zarr_path, var_name, year_idx, target_box, main_logger)

        var_end_time = time.time()
        main_logger.info(f"  Transferred {var_name} in {round(var_end_time - var_start_time)} seconds: {uu.timestr()}")


    ### Step 4: Process logs

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)

    # Worker logs are not aggregated if doing a local run (since there are no workers)
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with worker log compilation", main_logger)

        # Adds the workers' logs to the main log and uploads to s3
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)

    # Closes the Dask client if not running locally
    if not run_local:
        client.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a global rechunked mega-zarr.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--input_date', help='Date of run, in YYYYMMDD')
    parser.add_argument('-tc', '--test_chunk', nargs=4, type=float, help='Bounding box to print rechunked zarr stats from: W, S, E, N (degrees)')
    parser.add_argument('-fv', '--first_variables_to_process', type=int, help='Number of variables to process from raw mega-zarr (for testing)')
    parser.add_argument('-fy', '--first_years_to_process', type=int, help='Number of years to process from raw mega-zarr (for testing)')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    input_date = args.input_date
    test_chunk = args.test_chunk
    first_variables_to_process = args.first_variables_to_process
    first_years_to_process = args.first_years_to_process
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log

    # Create the cluster with command line arguments
    main(cluster_name, input_date, run_local, no_stats, no_log, test_chunk=test_chunk,
         first_variables_to_process=first_variables_to_process, first_years_to_process=first_years_to_process,
         log_note=log_note)