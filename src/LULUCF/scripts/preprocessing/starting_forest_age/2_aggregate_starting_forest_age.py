"""
Run from src/LULUCF/
python -m scripts.utilities.create_cluster -n 1 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing.starting_forest_age.2_aggregate_starting_forest_age -cn AFOLU_flux_model_scripts --first_chunks 2 --run_local

python -m scripts.utilities.create_cluster -n 33 -t 7 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing.starting_forest_age.2_aggregate_starting_forest_age -cn AFOLU_flux_model_scripts

Time: 2:02 through aggregation; 2:19 through tile stats; Credits: 9.3; Cost: $0.34
-n 33 -t 7 worked fine.
I noted that each worker started by processing 11 tasks instead of the t+1 tasks I expected. Don't know why.
So, maybe the number of threads doesn't matter. Given that each worker starts on 11 tasks at the beginning and there are
only 356 tasks, 33 workers ought to be sufficient to cover all the tasks in one pass (though that's not necessary).
"""

import argparse
import dask
import re

# Project imports
from ...utilities import constants_and_names as cn
from ...utilities import universal_utilities as uu
from ...utilities import log_utilities as lu
from ...utilities import resize_cluster


def main(cluster_name, run_local=False, no_stats=False, no_log=False, no_upload= False,
         first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    age_years = [2015]  # Could be expanded to use age in 2000 as well
    stage = f'starting_forest_age_interpolated__{age_years[0]}__10x10_deg_aggreg'
    model_type = 'standard'

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header('N/A', 'N/A', client, cluster, log_note, run_local, model_type, stage)

    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Year for forest age map: {age_years}")

    # Creates list of output directories specific to the run
    output_dir_list = [cn.forest_age_2015_interpolated_dir]
    output_dir_list = [path.replace("CHUNK_SIZE", str(4000)) for path in output_dir_list]
    main_logger.info(f"Directories to aggregate: {output_dir_list}")


    ### Step 2: Aggregates 1x1 degree outputs to 10x10 degree outputs

    # Creates the list of aggregated 10x10 rasters that will be created (list of dictionaries of input s3 folder and output aggregated raster name.
    # These are the basis for the aggregation tasks.
    list_of_s3_name_dicts_total = uu.create_list_for_aggregation(output_dir_list, main_logger)

    # For testing. Limits the number of output rasters to that given in the command line
    if first_chunks:
        list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[0:first_chunks]

    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[338:339]  # To limit it to a specific 10x10 deg tile

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
    parser = argparse.ArgumentParser(description="Aggregate starting forest age to 10x10 deg geotifs.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    first_chunks = args.first_chunks
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    main(cluster_name, run_local, no_stats, no_log, no_upload, first_chunks=first_chunks, log_note=log_note)

