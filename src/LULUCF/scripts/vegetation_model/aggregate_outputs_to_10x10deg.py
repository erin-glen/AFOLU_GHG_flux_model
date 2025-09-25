"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Can only run on 1x1 degree chunks that do not have the run timestamp in the file name.
The way this builds the input file names, it can't handle filenames with the run timestamp.
It also can't handle chunks smaller than 1x1 degree.
Aggregates in batches because there are so many tasks for this step that it's just more likely to fail part way through
and so having batches allows easier restarting. The tasks are in a pre-determined alphabetical order,
so each batch should have the same contents for any given set of inputs.

Local test:
python -m src.LULUCF.scripts.vegetation_model.aggregate_outputs_to_10x10deg -yr 2000 2024 --first_10x10s_to_process 2 --input_date YYYYMMDD --run_local

Coiled small test:
python -m src.utilities.create_cluster -n 1 -t 1 -m 4 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.vegetation_model.aggregate_outputs_to_10x10deg -cn LULUCF_postprocessing -yr 2000 2024 --first_10x10s_to_process 2 --input_date YYYYMMDD

Coiled large shapefile test (create a cluster with 1 worker, then resize it to 100 workers after local processing is done):
python -m src.utilities.create_cluster -n 1 -t 1 -m 4 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.vegetation_model.aggregate_outputs_to_10x10deg -cn LULUCF_postprocessing -yr 2000 2024 --input_date YYYYMMDD -nw 100 -ln "This is the aggregation of the 1884-chunk run."

Full Coiled run (create a cluster with 1 worker, then resize it to 200 workers after local processing is done):
python -m src.utilities.create_cluster -n 1 -t 1 -m 4 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.vegetation_model.aggregate_outputs_to_10x10deg -cn LULUCF_postprocessing -yr 2000 2024 --input_date YYYYMMDD -nw 200 -ln "This is intended to be the definitive global run."

Notes on optimizing threads/worker: https://app.asana.com/1/25496124013636/task/1206230383901961/comment/1210803828525318?focus=true
Tests of this aggregation and other aggregations show that 1 thread/worker with 4GB workers is low in Coiled credit usage
and runs quickly compared to other configurations.

14256 tiles in 273 folders for 1884-chunk output
"""

import argparse
import dask
import re
import sys
import pandas as pd

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import resize_cluster


def main(cluster_name, year_range, input_date, number_of_workers, run_local=False, no_stats=False, no_log=False, no_upload=False,
         first_10x10s_to_process=None, log_note=None):


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

    # Calculates the interval type, difference between start and end years of intervals,
    # and the model output years for the model run
    interval_type, interval_year_diff, interval_length, interval_end_years = uu.get_interval_info(end_year, main_logger, start_year)

    # Unlike numba-based scripts, this one doesn't construct the download dictionary in the main function.
    # Instead, it creates a list of input folders, from which a download dictionary is created for each chunk (in the chunk-level function).
    # It's a little simpler this way. Since the datatypes of the inputs don't need to be specified in advance for this script
    # (since it's not using numba), there's no need to centrally create a download dictionary with each input's datatype
    # just once on the scheduler, as is more efficient for scripts that use numba.
    # Creates a list of input directories used in summative output creation based on specifics of the model run

    # Only make 10x10s of the summative outputs. It keeps the workload smaller and these are the only ones that
    # have per-pixel outputs, which also need to be aggregated into 10x10s.
    output_dir_list_per_ha = uu.create_output_dir_name_list(cn.veg_summative_output_dirs, interval_type, start_year, '4000',
                                                            model_type, interval_end_years, interval_year_diff, input_date, "per_ha")
    output_dir_list_per_pixel = uu.create_output_dir_name_list(cn.veg_summative_output_dirs, interval_type, start_year, '4000',
                                                               model_type, interval_end_years, interval_year_diff, input_date, "per_pixel")

    # Also need to aggregate the land state nodes
    land_state_node_list = [s for s in cn.veg_core_output_dirs if cn.land_state_pattern in s]
    land_state_node_list = uu.create_output_dir_name_list(land_state_node_list, interval_type, start_year,'4000',
                                                     model_type, interval_end_years, interval_year_diff, input_date)

    # Full list of folders to aggregate, in alphabetical order
    output_dir_list = output_dir_list_per_ha + output_dir_list_per_pixel + land_state_node_list
    output_dir_list.sort()  # Sorts in-place
    main_logger.info(f"Directories to aggregate: {output_dir_list}")

    # # For testing- first folder only, so contents of all folders don't need to be listed
    # output_dir_list = output_dir_list[0:1]
    output_dir_list = output_dir_list[0:10]
    main_logger.info(f"output_dir_list: {output_dir_list}")

    main_logger.info(f"There are {len(output_dir_list)} folders to aggregate to 10x10s")


    ### Step 2: Aggregates 1x1 degree outputs to 10x10 degree outputs

    # Creates the list of aggregated 10x10 rasters that will be created (list of dictionaries of input s3 folder and output aggregated raster name.
    # These are the basis for the aggregation tasks.
    list_of_s3_name_dicts_total = uu.create_list_for_aggregation(output_dir_list, main_logger)

    # For testing. Limits the number of output 10x10 deg rasters to that given in the command line
    if first_10x10s_to_process:
        list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[0:first_10x10s_to_process]

    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[338:339]  # To limit it to a specific tile
    print("list_of_s3_name_dicts_total:", list_of_s3_name_dicts_total)

    # Extracts and lists unique tile_ids, the target for aggregation
    tile_ids = set()
    for entry in list_of_s3_name_dicts_total:
        for key, filenames in entry.items():
            for filename in filenames:
                match = re.search(cn.tile_id_pattern, filename)
                if match:
                    tile_ids.add(match.group())

    # Converts set of tile_ids to sorted list of tile_ids
    tile_id_list = sorted(tile_ids)
    main_logger.info(f"tile_ids to aggregate within: {tile_id_list} ({len(tile_id_list)}) tile_ids")

    # Determines if the output file names for final versions of outputs should be used
    is_final = False
    if len(list_of_s3_name_dicts_total) > 30:
        is_final = True
        main_logger.info("Running as final model.")

    main_logger.info(f"Aggregating 1x1 deg outputs to 10x10 deg outputs: {uu.timestr()}")

    chunk_batches = [list_of_s3_name_dicts_total[i:i + batch_size] for i in range(0, len(list_of_s3_name_dicts_total), batch_size)]
    main_logger.info(f"There are {len(chunk_batches)} batches to process: {uu.timestr()}")

    # Resizes cluster up to more workers now that all the local enumerating is done.
    # This way, it doesn't run a large cluster while all the local preprocessing is done.
    if not run_local:

        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Adds more workers if less than 9 were originally specified. 11 is an arbitrary number above which I'm not likely to be doing testing.
        # Otherwise, just keeps the number of workers already there (if number not specified for this script).
        if (n_workers < 11) and (number_of_workers):
            main_logger.info(f"Resizing cluster to specified number of workers: {number_of_workers}")
            resize_cluster.resize_coiled_cluster(cluster_name, number_of_workers)
        elif is_final:
            main_logger.info(f"Resizing cluster to large run number of workers: 100")
            resize_cluster.resize_coiled_cluster(cluster_name, 100)
        else:
            main_logger.info("Not resizing cluster")

    # Accumulates all output messages and statistics across batches
    # From https://chatgpt.com/share/e/5599b6b0-1aaa-4d54-98d3-c720a436dd9a
    all_10x10_results = []
    all_10x10_stats = []
    success_count = 0  # Count of successful chunks

    # Iterates through the batches
    for i, chunk_batch in enumerate(chunk_batches):
        main_logger.info(f"Processing batch {i + 1}/{len(chunk_batches)} ({len(chunk_batch)} chunks): {uu.timestr()}")

        # Each task is a single 10x10 deg aggregated geotif (10x10 deg for a given output)
        delayed_results_10x10_deg = [dask.delayed(uu.merge_small_tiles_gdal)(s3_name_dict, is_final, no_upload)
                                            for s3_name_dict in chunk_batch]

        results_batch_10x10_deg = dask.compute(*delayed_results_10x10_deg)

        all_10x10_results.extend(results_batch_10x10_deg)

        success_count, batch_stats = uu.count_successful_chunks(chunk_batch, is_final, main_logger, results_batch_10x10_deg)
        all_10x10_stats.extend(batch_stats)

        # Saves stats from batch in Excel locally in case the run fails, but only if there are multiple batches.
        # That way there are some basic chunk stats (not sorted or anything) to fall back on.
        if len(chunk_batches) > 1:
            main_logger.info(f"Writing batch stats to spreadsheet: {uu.timestr()}")
            df_all_10x10_stats = pd.DataFrame(all_10x10_stats)
            out_spreadsheet = f'TEMP_BATCH_{stage}_10x10_chunk_statistics__thru_batch_{i}_{uu.timestr()}.xlsx'
            local_spreadsheet = f"{cn.local_chunk_stats_path}{out_spreadsheet}"
            with pd.ExcelWriter(local_spreadsheet) as writer:
                df_all_10x10_stats.to_excel(writer, sheet_name=f'pix_counts__thru_batch_{i}', index=False)

        uu.stage_duration(start_time, uu.timestr(), f"{stage}, batch {i}", main_logger)


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
    if (not no_stats) and (success_count > 0):
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
    parser.add_argument('-rd', '--input_date', help='Date of core model run, in YYYYMMDD')
    parser.add_argument('-yr', '--year_range', nargs=2, type=int, required=True, help='Starting and ending years for model. Start options: 2000, 2015. End options: 2020, 2024.')
    parser.add_argument('-f', '--first_10x10s_to_process', type=int, help='Number of chunks to process from input list')
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
    first_10x10s_to_process = args.first_10x10s_to_process
    log_note = args.log_note
    number_of_workers = args.number_of_workers

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    main(cluster_name, year_range, input_date, number_of_workers, run_local, no_stats, no_log, no_upload,  first_10x10s_to_process=first_10x10s_to_process, log_note=log_note)