"""
Run from src/LULUCF

Test:
python -m scripts.utilities.create_cluster -n 1 -cn LULUCF_model
python -m scripts.core_model.1_summative_LULUCF_outputs -cn LULUCF_model -bb 10 49.75 10.25 50 -cs 0.25 -yr 2015 2023
python -m scripts.core_model.1_summative_LULUCF_outputs -cn LULUCF_model -bb 115.25 -3.75 115.5 -3.5 -cs 0.25 --no_upload -yr 2015 2023
python -m scripts.core_model.1_summative_LULUCF_outputs -cn LULUCF_model -bb 10 49 11 50 -cs 1 --no_upload -yr 2015 2023
python -m scripts.core_model.1_summative_LULUCF_outputs -cn LULUCF_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20241125/ -f 1 -yr 2015 2023

Full run:
python -m scripts.utilities.create_cluster -n 200
python -m scripts.core_model.1_summative_LULUCF_outputs -cn LULUCF_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20241125/
"""

import argparse
import concurrent.futures
import gc
import os
import psutil
import time
import sys
import numpy as np

from concurrent.futures import ThreadPoolExecutor

from dask.distributed import print

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu
from ..utilities import log_utilities as lu
from ..utilities import numba_utilities as nu
from ..utilities import resize_cluster


# ## Part 5: Calculates combined gross fluxes and net fluxes.
# ## Useful for QC-- to see if there are any egregiously incorrect or unexpected values.
# ## Doing this outside numba function to minimize pixel-level calculations and chunks being returned by numba function.
#
# lu.print_and_log(f"Summing derivative outputs in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)
#
# for interval_end_year in interval_end_years:
#
#     year_range = f"{interval_end_year - interval_year_diff}_{interval_end_year}"
#
#     # Gross emissions across all carbon pools
#     out_dict_all_dtypes[f"{cn.gross_emis_all_C_pools_CO2_only_pattern}_{year_range}"] = (
#             out_dict_all_dtypes[f"{cn.agc_gross_emis_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.bgc_gross_emis_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.deadwood_c_gross_emis_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.litter_c_gross_emis_pattern}_{year_range}"])
#
#     # Gross emissions for non-CO2 emissions
#     out_dict_all_dtypes[f"{cn.gross_emis_all_C_pools_non_CO2_only_pattern}_{year_range}"] = (
#             out_dict_all_dtypes[f"{cn.ch4_flux_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.n2o_flux_pattern}_{year_range}"])
#
#     # Gross emissions for all carbon pools and all gases
#     out_dict_all_dtypes[f"{cn.gross_emis_all_C_pools_all_gases_pattern}_{year_range}"] = (
#         out_dict_all_dtypes[f"{cn.gross_emis_all_C_pools_CO2_only_pattern}_{year_range}"]
#         + out_dict_all_dtypes[f"{cn.gross_emis_all_C_pools_non_CO2_only_pattern}_{year_range}"]
#     )
#
#     # Gross removals across all carbon pools
#     out_dict_all_dtypes[f"{cn.gross_removals_all_C_pools_pattern}_{year_range}"] = (
#             out_dict_all_dtypes[f"{cn.agc_gross_removals_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.bgc_gross_removals_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.deadwood_c_gross_removals_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.litter_c_gross_removals_pattern}_{year_range}"])
#
#     # Net flux for each carbon pool
#     out_dict_all_dtypes[f"{cn.agc_net_flux_pattern}_{year_range}"] = out_dict_all_dtypes[f"{cn.agc_gross_emis_pattern}_{year_range}"] + out_dict_all_dtypes[f"{cn.agc_gross_removals_pattern}_{year_range}"]
#     out_dict_all_dtypes[f"{cn.bgc_net_flux_pattern}_{year_range}"] = out_dict_all_dtypes[f"{cn.bgc_gross_emis_pattern}_{year_range}"] + out_dict_all_dtypes[f"{cn.bgc_gross_removals_pattern}_{year_range}"]
#     out_dict_all_dtypes[f"{cn.deadwood_c_net_flux_pattern}_{year_range}"] = out_dict_all_dtypes[f"{cn.deadwood_c_gross_emis_pattern}_{year_range}"] + out_dict_all_dtypes[f"{cn.deadwood_c_gross_removals_pattern}_{year_range}"]
#     out_dict_all_dtypes[f"{cn.litter_c_net_flux_pattern}_{year_range}"] = out_dict_all_dtypes[f"{cn.litter_c_gross_emis_pattern}_{year_range}"] + out_dict_all_dtypes[f"{cn.litter_c_gross_removals_pattern}_{year_range}"]
#
#     # Net flux across all carbon pools but for CO2 only
#     out_dict_all_dtypes[f"{cn.net_flux_all_C_pools_CO2_only_pattern}_{year_range}"] = (
#             out_dict_all_dtypes[f"{cn.agc_net_flux_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.bgc_net_flux_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.deadwood_c_net_flux_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.litter_c_net_flux_pattern}_{year_range}"])
#
#     # Net flux across all carbon pools, plus non-pool non-CO2 emissions
#     out_dict_all_dtypes[f"{cn.net_flux_all_C_pools_all_gases_pattern}_{year_range}"] = (
#             out_dict_all_dtypes[f"{cn.net_flux_all_C_pools_CO2_only_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.gross_emis_all_C_pools_non_CO2_only_pattern}_{year_range}"])
#
# lu.print_and_log(f"Done summing derivative outputs in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)
# print(f"After creating summative outputs for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB")

def create_summative_LULUCF_outputs(bounds, start_year, end_year, interval_type, interval_year_diff, interval_length,
                                    interval_end_years, fishnet_iso_df, is_final, no_upload, output_dir_list, stage):

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []

    process = psutil.Process(os.getpid())

    logger_worker = lu.setup_logging_worker()

    uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_final, logger_worker)

    bounds_str = uu.boundstr(bounds)  # String form of chunk bounds, from e.g., [8, -1, 9, 0] to 8_-1_9_0
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)





def main(cluster_name, year_range, run_local=False, no_stats=False, no_log=False, no_upload=False,
         use_shapefile=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):


    ### Step 1: Preparation

    # Model stage being run
    stage = 'LULUCF_summative_output_calculation'
    model_type = 'standard_model'
    run_date = '20250415'

    # Determines if arguments for start and end year are valid
    if year_range not in [[cn.first_model_year_5_years, cn.last_model_year_5_years],  # 2000-2020
                          [cn.first_model_year_5_years, cn.last_model_year_annual],  # 2000-2023
                          [cn.first_model_year_annual, cn.last_model_year_annual]]:  # 2015-2023
        print("Year range selection not valid")
        sys.exit()
    else:
        start_year = year_range[0]
        end_year = year_range[1]
        print(f"Start year: {start_year}")
        print(f"End year: {end_year}")

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(bounding_box, use_shapefile, client, cluster, log_note, run_local, model_type, stage)

    # Starting time for stage
    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Start year: {start_year}; end year: {end_year}")

    # Calculates the interval type, difference between start and end years of intervals,
    # and the model output years for the model run
    interval_type, interval_year_diff, interval_length, interval_end_years = uu.get_interval_info(end_year, main_logger, start_year)

    # Returns a dataframe of chunk_id and ISO for the GADM3.6 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso()

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, use_shapefile, chunk_size, first_chunks, fishnet_iso_df, main_logger)

    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_final = False
    if len(chunk_list) > 20:
        is_final = True
        main_logger.info("Running as final model.")


    # This is just a placeholder tile_id that is used to obtain the datatype of each tile set.
    # It is overwritten when chunks are assigned and analyzed.
    # Using this placeholder allows the full path and tile name to be specified up front, which simplifies things.
    # Otherwise, we'd have just the path but not the file name now and would have to add in the file name later
    # (probably at the chunk level).
    sample_tile_id = "00N_000E"

    download_dict = {}

    output_dir_list = uu.create_output_dir_name_list(cn.LULUCF_core_output_dirs, interval_type, start_year,
                                                     chunk_size_pixels, model_type, interval_end_years, interval_year_diff, run_date)
    print(output_dir_list)

    # Makes a txt for each task in the list. These are deleted as tasks are completed.
    main_logger.info("Creating task txts in s3...")
    uu.create_s3_task_files(stage, chunk_list)

    # This approach handles large task lists (graphs) better than [dask.delayed(calculate_and_upload_LULUCF_fluxes ... )]
    futures = []
    for chunk in chunk_list:
        future = client.submit(create_summative_LULUCF_outputs,
                               chunk, start_year, end_year, interval_type, interval_year_diff, interval_length, interval_end_years,
                               fishnet_iso_df, is_final, no_upload, output_dir_list, stage)
        futures.append(future)

    # Collect the results once they are finished
    flux_1x1_results = client.gather(futures)

    success_count_1x1, all_1x1_stats = uu.count_successful_chunks(chunk_list, is_final, main_logger, flux_1x1_results)


    # for LULUCF_core_output_dir in cn.LULUCF_core_output_dirs:
    #     print(LULUCF_core_output_dir)
    #
    #     # Remove the prefix
    #     relative_path = LULUCF_core_output_dir[len(cn.outputs_path):]
    #
    #     # Split by '/' and get the first folder
    #     pattern = relative_path.split('/')[0]
    #
    #     LULUCF_core_output_dir = LULUCF_core_output_dir.replace("MODEL_TYPE", model_type)
    #     LULUCF_core_output_dir = LULUCF_core_output_dir.replace("MODEL_INTERVAL_TYPE", interval_type)
    #     LULUCF_core_output_dir = LULUCF_core_output_dir.replace("CHUNK_SIZE", str(chunk_size_pixels))
    #     LULUCF_core_output_dir = LULUCF_core_output_dir.replace("DATE", '20250415'[:8])
    #
    #     for year in range(start_year + 1, end_year + 1, interval_length):
    #         download_dict[f"{pattern}_{year}"] = f"{LULUCF_core_output_dir}{sample_tile_id}_{pattern}_{year}.tif"
    #
    #     print(download_dict)
    #
    #     sys.quit()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate summative outputs of core LULUCF model.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cshp', '--use_shapefile', help='Shapefile of chunks')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('-yr', '--year_range', nargs=2, type=int, required=True, help='Starting and ending years for model. Start options: 2000, 2015. End options: 2020, 2023.')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    bounding_box = args.bounding_box
    chunk_size = args.chunk_size
    use_shapefile = args.use_shapefile
    first_chunks = args.first_chunks
    year_range = args.year_range
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, year_range, run_local, no_stats, no_log, no_upload, use_shapefile,
         bounding_box=bounding_box, chunk_size=chunk_size,
         first_chunks=first_chunks, log_note=log_note)
