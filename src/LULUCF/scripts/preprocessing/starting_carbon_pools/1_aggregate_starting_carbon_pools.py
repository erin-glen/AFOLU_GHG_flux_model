"""
Run from src/LULUCF/
python -m scripts.utilities.create_cluster -n 1 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing.starting_carbon_pools.1_aggregate_starting_carbon_pools -cn AFOLU_flux_model_scripts --year 2015 --first_chunks 2 --run_local --input_date YYYYMMDD

python -m scripts.utilities.create_cluster -n 40 -t 5 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing.starting_carbon_pools.1_aggregate_starting_carbon_pools -cn AFOLU_flux_model_scripts --year 2015 --input_date YYYYMMDD
Time: 16:32 through calculation; 16:48 through tile stats; Credits: 59; Cost: $1.90
Using more than -t 5 seemed to cause some tile_ids to randomly fail, even though memory usage was not high.
So, best to stay with -t 5 even though the Dask dashboard indicates low memory usage compared to what's available (e.g., 5 out of 32 GB being used).
"""

import argparse
import dask
import re
import sys

# Project imports
from ...utilities import constants_and_names as cn
from ...utilities import universal_utilities as uu
from ...utilities import log_utilities as lu
from ...utilities import resize_cluster



def main(cluster_name, year, input_date, run_local=False, no_stats=False, no_log=False, no_upload= False,
         first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = f'starting_carbon_pools_{year}_10x10_deg_aggreg'
    model_type = 'standard'

    # Directories to process
    if year == 2000:
        output_dir_list = [cn.agc_2000_dir, cn.bgc_2000_dir, cn.deadwood_c_2000_dir, cn.litter_c_2000_dir]
    elif year == 2015:
        output_dir_list = [cn.agc_2015_dir, cn.bgc_2015_dir, cn.deadwood_c_2015_dir, cn.litter_c_2015_dir]
        # output_dir_list = [cn.deadwood_c_2015_dir]  # To test a specific carbon pool
    else:
        print(f"Year input {year} not valid. Terminating.")
        sys.exit()

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header('N/A', 'N/A', client, cluster, log_note, run_local, model_type, stage)

    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Year for initial carbon pools: {year}")
    main_logger.info(f"Date for 1x1 deg rasters being aggregated: {input_date}")

    # Creates list of output directories specific to the run
    output_dir_list = [path.replace("DATE", input_date) for path in output_dir_list]
    output_dir_list = [path.replace("CHUNK_SIZE", str(4000)) for path in output_dir_list]
    main_logger.info(f"Directories to aggregate: {output_dir_list}")


    ### Step 2: Aggregates 1x1 degree outputs to 10x10 degree outputs

    # Creates the list of aggregated 10x10 rasters that will be created (list of dictionaries of input s3 folder and output aggregated raster name.
    # These are the basis for the aggregation tasks.
    list_of_s3_name_dicts_total = uu.create_list_for_aggregation(output_dir_list, main_logger)

    # For testing. Limits the number of output rasters to that given in the command line
    if first_chunks:
        list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[0:first_chunks]

    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[338:339]  # To limit it to a specific tile

    # Extracts and lists unique tile_ids, the target for aggregation
    tile_ids = set()
    for entry in list_of_s3_name_dicts_total:
        for key, filenames in entry.items():
            for filename in filenames:
                match = re.search(cn.tile_id_pattern, filename)
                if match:
                    tile_ids.add(match.group())

    # Converts set of tile_ids to sorted list of tile_ids
    chunk_list = sorted(tile_ids)
    main_logger.info(f"tile_ids to process: {chunk_list}")
    main_logger.info(f"Number of tile_ids to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_final = False
    if len(chunk_list) > 20:
        is_final = True
        main_logger.info("Running as final model.")

    main_logger.info(f"Aggregating 1x1 deg outputs to 10x10 deg outputs: {uu.timestr()}")

    # Each task is a single 10x10 deg aggregated geotif
    delayed_results_10x10_deg = [dask.delayed(uu.merge_small_tiles_gdal)(s3_name_dict, is_final, no_upload)
                                        for s3_name_dict in list_of_s3_name_dicts_total]

    results_10x10_deg = dask.compute(*delayed_results_10x10_deg)

    success_count_10x10, all_10x10_stats = uu.count_successful_chunks(chunk_list, is_final, main_logger, results_10x10_deg)

    uu.stage_duration(start_time, uu.timestr(), f"{stage}", main_logger)


    ### Step 3: Chunk stats (i.e. pixel counts) for 10x10 degree outputs, aggregates logs

    # Resizes cluster down to 1 worker for chunk stats and log aggregation since that only needs a minimal remainder of the
    # cluster, not all the workers.
    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster("AFOLU_flux_model_scripts", 1)

    # Prepares 10x10 deg chunk stats spreadsheet: pixel count for outputs
    if (not no_stats) and (success_count_10x10 > 0):
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
    parser = argparse.ArgumentParser(description="Create carbon pools in 2000.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('--year', type=int, required=True, help='Year for carbon pools')
    parser.add_argument('--input_date', required=True, help='Date YYYYMMDD of carbon pool 1x1s to process')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    first_chunks = args.first_chunks
    year = args.year
    input_date = args.input_date
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    main(cluster_name, year, input_date, run_local, no_stats, no_log, no_upload, first_chunks=first_chunks, log_note=log_note)

