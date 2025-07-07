"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model/

Currently only aggregates per-ha outputs to 10x10 deg, not per-pixel outputs.

Local test:
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.3_aggregate_starting_carbon_pools --year 2015 --first_10x10s_to_process 2

Coiled test:
python -m src.utilities.create_cluster -n 1 -t 1 -m 2 -cn LULUCF_preprocessing
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.3_aggregate_starting_carbon_pools -cn LULUCF_preprocessing --year 2000 --first_10x10s_to_process 2

Full run 2000:
python -m src.utilities.create_cluster -n 200 -t 1 -m 2 -cn LULUCF_preprocessing
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.3_aggregate_starting_carbon_pools -cn LULUCF_preprocessing --year 2000
Peak memory per worker: ~350-400 MB
Time for total processing for each task: average of 303 seconds, min of 87 seconds and max of 1327 seconds (based on extraction from log)
Time until chunk stats: 1:43:55
Time after chunk stats: 1:44:22
Coiled credits: 737.4 (402/hr for 200 t3.small workers, according to dashboard)
AWS cost: $3.86 ($2.10/hr for 200 t3.small workers, according to dashboard)

Full run 2015:
python -m src.utilities.create_cluster -n 200 -t 1 -m 2 -cn LULUCF_preprocessing
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.3_aggregate_starting_carbon_pools -cn LULUCF_preprocessing --year 2015
Peak memory per worker: ~350-400 MB
Time for total processing for each task: average of 231 seconds, min of 86 seconds and max of 690 seconds (based on extraction from log)
Time until chunk stats: 1:17:04
Time after chunk stats: 1:17:23
Coiled credits: 557 (402/hr for 200 t3.small workers, according to dashboard)
AWS cost: $2.93 ($2.10/hr for 200 t3.small workers, according to dashboard)
I don't know why the 2015 was 30 minutes faster and used almost 200 fewer Coiled credits than the 2000 one. Maybe just luck of the cluster?

There are 3560 10x10s across 10 folders (356/folder), so 200 workers seems appropriate.
"""

import argparse
import dask
import re
import sys

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import resize_cluster


def main(cluster_name, year, run_local=False, no_stats=False, no_log=False, no_upload= False,
         first_10x10s_to_process=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = f'starting_carbon_pools_{year}_10x10_deg_aggreg'
    model_type = 'standard'

    # Directories to process
    if year == 2000:
        output_dir_list = [cn.agc_2000_raw_dir, cn.bgc_2000_raw_dir, cn.deadwood_c_2000_raw_dir, cn.litter_c_2000_raw_dir, cn.non_soil_c_2000_raw_dir,
                           cn.agc_2000_LC_masked_dir, cn.bgc_2000_LC_masked_dir, cn.deadwood_c_2000_LC_masked_dir, cn.litter_c_2000_LC_masked_dir, cn.non_soil_c_2000_LC_masked_dir]
        # output_dir_list = [cn.agc_2000_raw_dir]  # To test a specific carbon pool
    elif year == 2015:
        output_dir_list = [cn.agc_2015_raw_dir, cn.bgc_2015_raw_dir, cn.deadwood_c_2015_raw_dir, cn.litter_c_2015_raw_dir, cn.non_soil_c_2015_raw_dir,
                           cn.agc_2015_LC_masked_dir, cn.bgc_2015_LC_masked_dir, cn.deadwood_c_2015_LC_masked_dir, cn.litter_c_2015_LC_masked_dir, cn.non_soil_c_2015_LC_masked_dir]
    else:
        print(f"Year input {year} not valid. Terminating.")
        sys.exit()

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Year for initial carbon pools: {year}")
    input_date = re.findall(r"\d{8}", output_dir_list[0])[0]
    main_logger.info(f"Date for 1x1 deg rasters being aggregated: {input_date}")


    # Creates list of output directories specific to the run
    output_dir_list = [path.replace("CHUNK_SIZE", str(4000)) for path in output_dir_list]
    output_dir_list = [path.replace("PER_HA_OR_PIXEL", cn.C_density_pixel_meaning) for path in output_dir_list]
    main_logger.info(f"Directories to aggregate: {output_dir_list}")


    ### Step 2: Aggregates 1x1 degree outputs to 10x10 degree outputs

    # Creates the list of aggregated 10x10 rasters that will be created (list of dictionaries of input s3 folder and output aggregated raster name.
    # These are the basis for the aggregation tasks.
    list_of_s3_name_dicts_total = uu.create_list_for_aggregation(output_dir_list, main_logger)

    # For testing. Limits the number of output rasters to that given in the command line
    if first_10x10s_to_process:
        list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[0:first_10x10s_to_process]

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

            resize_cluster.resize_coiled_cluster(cluster_name, 1)

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
    parser = argparse.ArgumentParser(description="Aggregate 1x1 degree starting carbon densities to 10x10 degree COGs.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-f', '--first_10x10s_to_process', type=int, help='Number of chunks to process from input list')
    parser.add_argument('--year', type=int, required=True, help='Year for carbon pools')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    first_10x10s_to_process = args.first_10x10s_to_process
    year = args.year
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    main(cluster_name, year, run_local, no_stats, no_log, no_upload, first_10x10s_to_process=first_10x10s_to_process, log_note=log_note)

