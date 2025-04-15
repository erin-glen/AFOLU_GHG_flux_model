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

    # # Create the cluster with command line arguments
    # main(cluster_name, year_range, run_local, no_stats, no_log, no_upload, use_shapefile,
    #      bounding_box=bounding_box, chunk_size=chunk_size,
    #      first_chunks=first_chunks, log_note=log_note)
