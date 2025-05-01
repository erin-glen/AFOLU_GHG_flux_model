"""
Run from src/LULUCF

Can only run on 1x1 degree chunks that do not have the run timestamp in the file name.
The way this builds the input file names, it can't handle filenames with the run timestamp.
It also can't handle chunks smaller than 1x1 degree.

Local test:
python -m scripts.core_model.3_aggregate_LULUCF_outputs -yr 2015 2023 --first_10x10s_to_process 2 --run_date YYYYMMDD

Coiled small test:
python -m scripts.utilities.create_cluster -n 1 -cn LULUCF_postprocessing
python -m scripts.core_model.3_aggregate_LULUCF_outputs -cn LULUCF_postprocessing -yr 2015 2023 --first_10x10s_to_process 2 --run_date YYYYMMDD

Coiled large shapefile test:
python -m scripts.core_model.3_aggregate_LULUCF_outputs -cn LULUCF_postprocessing -yr 2015 2023 --run_date YYYYMMDD

Full Coiled run:
python -m scripts.utilities.create_cluster -n 50 -cn LULUCF_postprocessing
python -m scripts.core_model.3_aggregate_LULUCF_outputs -cn LULUCF_postprocessing -yr 2015 2023 --run_date YYYYMMDD

From before:
Took about 30 minutes to do the aggregated gross and net flux outputs. A few 10x10 tiles from many of the folders
weren't output, and I got various GDAL errors throughout. Not investigating further now.
Log to explore is https://cloud.coiled.io/clusters/676603/account/wri-forest-research/information?workspace=WRI-forest-research&tab=Logs&filterPattern=&showLifecycle=0
It has some potentially useful errors.
"""

import argparse
import dask
import re
import sys

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import log_utilities as lu
from ..utilities import universal_utilities as uu
from ..utilities import resize_cluster


def main(cluster_name, year_range, run_date, run_local=False, no_stats=False, no_log=False, no_upload=False,
         first_10x10s_to_process=None, log_note=None):


    ### Step 1: Preparation

    # Model stage being run
    stage = 'LULUCF_aggregation_to_10x10_deg'
    model_type = 'standard_model'

    # Determines if arguments for start and end year are valid
    if year_range not in [[cn.first_model_year_5_years, cn.last_model_year_5_years],  # 2000-2020
                          [cn.first_model_year_5_years, cn.last_model_year_annual],  # 2000-2023
                          [cn.first_model_year_annual, cn.last_model_year_annual]]:  # 2015-2023
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
    main_logger.info(f"Run date: {run_date}")

    # Calculates the interval type, difference between start and end years of intervals,
    # and the model output years for the model run
    interval_type, interval_year_diff, interval_length, interval_end_years = uu.get_interval_info(end_year, main_logger, start_year)

    # Unlike numba-based scripts, this one doesn't construct the download dictionary in the main function.
    # Instead, it creates a list of input folders, from which a download dictionary is created for each chunk (in the chunk-level function).
    # It's a little simpler this way. Since the datatypes of the inputs don't need to be specified in advance for this script
    # (since it's not using numba), there's no need to centrally create a download dictionary with each input's datatype
    # just once on the scheduler, as is more efficient for scripts that use numba.
    # Creates a list of input directories used in summative output creation based on specifics of the model run

    # Combine the core LULUCF outputs and the summative ones into a single list that will be used for 1x1 deg aggregation
    LULUCF_aggreg_dirs = cn.LULUCF_core_output_dirs + cn.LULUCF_summative_output_dirs

    output_dir_list = uu.create_output_dir_name_list(LULUCF_aggreg_dirs, interval_type, start_year,'4000',
                                                     model_type, interval_end_years, interval_year_diff, run_date)

    # # For testing- first folder only, so contents of all folders don't need to be listed
    # output_dir_list = output_dir_list[0:1]
    # print(output_dir_list)

    main_logger.info(f"There are {len(output_dir_list)} folders to aggregate to 10x10s")


    ### Step 2: Aggregates 1x1 degree outputs to 10x10 degree outputs

    # Creates the list of aggregated 10x10 rasters that will be created (list of dictionaries of input s3 folder and output aggregated raster name.
    # These are the basis for the aggregation tasks.
    list_of_s3_name_dicts_total = uu.create_list_for_aggregation(output_dir_list, main_logger)

    # For testing. Limits the number of output 10x10 deg rasters to that given in the command line
    if first_10x10s_to_process:
        list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[0:first_10x10s_to_process]

    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[338:339]  # To limit it to a specific tile
    # print(list_of_s3_name_dicts_total)

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
    parser = argparse.ArgumentParser(description="Aggregate 1x1 degree outputs from LULUCF model to 10x10 degree geotifs.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--run_date', help='Date of run, in YYYYMMDD')
    parser.add_argument('-yr', '--year_range', nargs=2, type=int, required=True, help='Starting and ending years for model. Start options: 2000, 2015. End options: 2020, 2023.')
    parser.add_argument('-f', '--first_10x10s_to_process', type=int, help='Number of chunks to process from input list')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    run_date = args.run_date
    year_range = args.year_range
    first_10x10s_to_process = args.first_10x10s_to_process
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    main(cluster_name, year_range, run_date, run_local, no_stats, no_log, no_upload, first_10x10s_to_process=first_10x10s_to_process, log_note=log_note)