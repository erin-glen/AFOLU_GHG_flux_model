"""
Run from src/LULUCF/
python -m scripts.utilities.create_cluster -n 1 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing.create_starting_carbon_pools -cn AFOLU_flux_model_scripts -bb 116 -3 116.25 -2.75 -cs 0.25 --no_stats --year YYYY
python -m scripts.preprocessing.create_starting_carbon_pools -cn AFOLU_flux_model_scripts -cshp -f 1 --year YYYY

python -m scripts.utilities.create_cluster -n 60 -t 10 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing.create_starting_carbon_pools -cn AFOLU_flux_model_scripts -cshp --year 2000 -ln "This is intended to be the definitive global run for carbon pool 2000 creation."
Max memory usage: ~18 GB/worker
Time: 23:17 through calculation; 39:26 through aggregation; 40:25 through tile stats; Credits: 170; Cost: $6.00

python -m scripts.utilities.create_cluster -n 50 -t 12 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing.create_starting_carbon_pools -cn AFOLU_flux_model_scripts -cshp --year 2015 -ln "This is intended to be the definitive global run for carbon pool 2015 creation."
Max memory usage: ~20 GB/worker
Time:  through calculation;  through aggregation;  through tile stats; Credits: ; Cost: $

NOTE: Maybe there's some way to configure this to output 10x10 deg tiles but I can't figure it out.
Instead, it creates 1x1 deg tiles and then merges them to 10x10 deg tiles.

To create a vrt of the 10x10 deg outputs, do:
aws s3 ls s3://gfw2-data/climate/ESA_CCI_biomass/v5_01/2015/year_2015_derived_carbon_pools/litter_C_density_MgC_ha/40000_pixels/ --recursive | grep .tif$ | awk '{print "/vsis3/gfw2-data/"$4}' > litter_C_2015_file_list.txt
gdalbuildvrt -input_file_list litter_C_2015_file_list.txt deadwood_C2015_mosaic.vrt
"""

import argparse
import concurrent.futures
import dask
import numpy as np
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from dask.distributed import print
from numba import jit

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu
from ..utilities import log_utilities as lu
from ..utilities import numba_utilities as nu
from ..utilities import resize_cluster



def main(cluster_name, year, input_date, run_local=False, no_stats=False, no_log=False, no_aggregate=False, no_upload= False,
         use_shapefile=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being running
    stage = f'starting_carbon_pools_{year}_10x10_deg_aggreg'
    model_type = 'standard'

    # Determines if argument for year is valid
    if year in [2000, 2015]:
        print("Year selection valid")
    else:
        print("Year selection not valid")
        sys.exit()

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(bounding_box, use_shapefile, client, cluster, log_note, run_local, model_type, stage)

    # Starting time for stage
    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Year for carbon pools: {year}")
    main_logger.info(f"Input data for 1x1 deg carbon pools to aggregate: {input_date}")

    # Directories to process
    if year == 2000:
        output_dir_list = [cn.agc_2000_dir, cn.bgc_2000_dir, cn.deadwood_c_2000_dir, cn.litter_c_2000_dir]
    elif year == 2015:
        output_dir_list = [cn.agc_2015_dir, cn.bgc_2015_dir, cn.deadwood_c_2015_dir, cn.litter_c_2015_dir]
    else:
        print(f"Year input {year} not valid. Terminating.")
        sys.exit()

    # Creates list of output directories specific to the run
    output_dir_list = [path.replace("DATE", uu.timestr()[:8]) for path in output_dir_list]
    output_dir_list = [path.replace("CHUNK_SIZE", str(4000)) for path in output_dir_list]
    main_logger.info(f"Directories to aggregate: {output_dir_list}")


    ### Step 2: Aggregates 1x1 degree outputs to 10x10 degree outputs (if not disabled)

    # Creates the list of aggregated 10x10 rasters that will be created (list of dictionaries of input s3 folder and output aggregated raster name.
    # These are the basis for the aggregation tasks.
    list_of_s3_name_dicts_total = uu.create_list_for_aggregation(output_dir_list, main_logger)

    # For testing. Limits the number of output rasters
    list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[0:1]  # First 1 tile
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[0:3]  # First 2 tiles
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[40:41] # 10N_130E; Internal chunks missing and padding needed on right; FID40
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[0:8]
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[16:17] # 00N_110E
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[41:42]  # 10S_010E; No padding needed; FID41
    # print(list_of_s3_name_dicts_total)

    # Extracts and lists unique tile_ids
    tile_ids = set()
    for entry in list_of_s3_name_dicts_total:
        for key, filenames in entry.items():
            for filename in filenames:
                match = re.search(cn.tile_id_pattern, filename)
                if match:
                    tile_ids.add(match.group())

    # Converts set to sorted list
    chunk_list = sorted(tile_ids)

    # Determines if the output file names for final versions of outputs should be used
    is_final = False
    if len(chunk_list) > 20:
        is_final = True
        main_logger.info("Running as final model.")

    main_logger.info(f"Aggregating 1x1 deg outputs to 10x10 deg outputs: {uu.timestr()}")

    # Each task is a single 10x10 deg aggregated geotif
    C_pool_10x10_deg_delayed_results = [dask.delayed(uu.merge_small_tiles_gdal)(s3_name_dict, is_final, no_upload)
                                        for s3_name_dict in list_of_s3_name_dicts_total]

    C_pool_10x10_deg_results = dask.compute(*C_pool_10x10_deg_delayed_results)

    success_count_10x10, all_10x10_stats = uu.count_successful_chunks(chunk_list, is_final, main_logger, C_pool_10x10_deg_results)

    uu.stage_duration(start_time, uu.timestr(), f"{stage}", main_logger)


    ### Step 3: Chunk stats for 10x10 degree outputs, aggregates logs

    # Resizes cluster down to 1 worker for chunk stats and log aggregation since that only needs a minimal remainder of the
    # cluster, not all the workers.
    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        # TODO Or maybe just have it terminate the cluster altogether, rather than resize it. Need to make sure that chunk stats and log still work, though.
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster("AFOLU_flux_model_scripts", 1)

    # Prepares 1x1 deg chunk stats spreadsheet: min, mean, max, and sum for all input and output chunks,
    # and min and max values across all chunks for all inputs and outputs
    # only if not suppressed by the --no_stats flag and at least one chunk was successfully (wasn't skipped).
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
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cshp', '--use_shapefile', action='store_true', help='Use shapefile to determine chunks')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('--year', type=int, required=True, help='Year for carbon pools')
    parser.add_argument('--input_date', required=True, help='Date YYYYMMDD of carbon pool 1x1s to process')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_aggregate', action='store_true', help='Do not aggregate 1x1 degrees outputs to 10x10 degree outputs')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    bounding_box = args.bounding_box
    chunk_size = args.chunk_size
    use_shapefile = args.use_shapefile
    first_chunks = args.first_chunks
    year = args.year
    input_date = args.input_date
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_aggregate = args.no_aggregate
    no_upload = args.no_upload

    main(cluster_name, year, input_date, run_local, no_stats, no_log, no_aggregate, no_upload, use_shapefile,
         bounding_box=bounding_box, chunk_size=chunk_size,
         first_chunks=first_chunks, log_note=log_note)

