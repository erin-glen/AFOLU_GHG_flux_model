"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model



Local test:
python -m src.LULUCF.scripts.core_model.3_create_individual_zarrs -yr 2000 2024 --first_folders_to_process 2 --first_10x10s_to_process 2 --input_date YYYYMMDD --run_local

Coiled small test:
python -m src.utilities.create_cluster -n 2 -m 32 -cn LULUCF_postprocessing

Coiled large area test:
python -m src.utilities.create_cluster -n 2 -m 32 -cn LULUCF_postprocessing

Full Coiled run:
python -m src.utilities.create_cluster -n 2 -m 32 -cn LULUCF_postprocessing

Based on discussion with Justin Terry and https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68bf3334-a09c-8320-a556-153f43ef9cd0
"""

import argparse
import fsspec
import rasterio
import re
from collections import defaultdict
import numpy as np
import os
import sys
import time
import warnings
import xarray as xr
from dask.distributed import Client, print

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu

# To hide the warning "UserWarning: Consolidated metadata is currently not part in the Zarr format 3 specification. It may not be supported by other zarr implementations and may change in the future."
warnings.filterwarnings(
    "ignore",
    message="Consolidated metadata is currently not part in the Zarr format 3 specification.*",
    category=UserWarning,
    module="zarr.api.asynchronous"
)

def build_mega_zarr(output_dir_list, main_logger):

    main_logger.info(f"Combining zarrs in {len(output_dir_list)} folders: {uu.timestr()}")

    data_vars_by_name = defaultdict(list)

    for folder in output_dir_list:
        main_logger.info(f"Scanning {folder}: {uu.timestr()}")

        fs = fsspec.filesystem('s3') if folder.startswith("s3://") else fsspec.filesystem("file")
        zarr_paths = [f"s3://{path}" for path in fs.glob(os.path.join(folder, '*.zarr'))]

        if not zarr_paths:
            main_logger.warning(f"No zarr datasets found in: {folder}")
            continue

        for zarr_path in zarr_paths:
            main_logger.info(f"Opening zarr {zarr_path}: {uu.timestr()}")
            ds = xr.open_zarr(zarr_path, consolidated=True)

            # Extracts interval from path
            interval = re.search(r"hybrid_intervals/([^/]+)", zarr_path).group(1)

            for var_name in ds.data_vars:
                base_var_name = re.sub(r"_\d{4}\.zarr$", "", var_name)
                da = ds[var_name].expand_dims({'interval': [int(interval[-4:])]})  # interval variable is just the end year
                data_vars_by_name[base_var_name].append(da)

    if not data_vars_by_name:
        main_logger.error(f"No variables found to combine: {uu.timestr()}")
        return

    main_logger.info(f"Concatenating variables across intervals: {uu.timestr()}")

    final_vars = {}

    for var_name, da_list in data_vars_by_name.items():
        main_logger.info(f"Combining variable {var_name}: {uu.timestr()}")
        combined = xr.concat(da_list, dim='interval')
        final_vars[var_name] = combined.chunk({'x': cn.zarr_pixel_chunks, 'y': cn.zarr_pixel_chunks, 'interval': 1})

    mega_ds = xr.Dataset(final_vars)

    main_logger.info("MEGAZARR!!!!!! properties:")
    main_logger.info(mega_ds)
    main_logger.info(f"megazarr interval values: {mega_ds.coords["interval"].values}")

    output_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3_zarr_testing_small/mega_zarr/20250904/all_outputs.zarr"

    main_logger.info(f"Writing megazarr to {output_path}: {uu.timestr()}")
    mega_ds.to_zarr(output_path, mode='w', consolidated=True)

    main_logger.info(f"Megazarr writing complete: {uu.timestr()}")


def main(cluster_name, year_range, input_date, run_local=False, no_stats=False, no_log=False, no_upload=False,
         first_folders_to_process=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = 'LULUCF_output_mega_zarr_creation'
    model_type = 'standard_model'

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
    main_logger.info(f"no_upload: {no_upload}")

    # Calculates the interval type, difference between start and end years of intervals,
    # and the model output years for the model run
    interval_type, interval_year_diff, interval_length, interval_end_years = uu.get_interval_info(end_year, main_logger, start_year)

    # Testing list with a variety of inputs: no unit_type, date_range, date
    output_dir_list = [
        # "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/AGC_emission_factor_CO2_only__fraction/standard_model/hybrid_intervals/2001_2005/4000_pixels/20250904/",
        # "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/AGC_emission_factor_CO2_only__fraction/standard_model/hybrid_intervals/2006_2010/4000_pixels/20250904/",
        # "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/gross_emissions__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2001_2005/_ha_yr/4000_pixels/20250904/",
        # "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/gross_emissions__all_C_pools__all_gases__MgCO2e/standard_model/hybrid_intervals/2006_2010/_ha_yr/4000_pixels/20250904/",
        # "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/carbon_density__AGC__MgC/standard_model/hybrid_intervals/2005/_ha/4000_pixels/20250904/",
        # "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3/carbon_density__AGC__MgC/standard_model/hybrid_intervals/2010/_ha/4000_pixels/20250904/"
        f"s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3_zarr_testing_small/carbon_density__AGC__MgC/standard_model/hybrid_intervals/2005/_ha/{cn.zarr_output_pattern}/20250904/",
        f"s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3_zarr_testing_small/carbon_density__AGC__MgC/standard_model/hybrid_intervals/2010/_ha/{cn.zarr_output_pattern}/20250904/",
        f"s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3_zarr_testing_small/carbon_density__BGC__MgC/standard_model/hybrid_intervals/2005/_ha/{cn.zarr_output_pattern}/20250904/",
        f"s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3_zarr_testing_small/carbon_density__BGC__MgC/standard_model/hybrid_intervals/2010/_ha/{cn.zarr_output_pattern}/20250904/"

    ]

    # Unlike numba-based scripts, this one doesn't construct the download dictionary in the main function.
    # Instead, it creates a list of input folders, from which a download dictionary is created for each chunk (in the chunk-level function).
    # It's a little simpler this way. Since the datatypes of the inputs don't need to be specified in advance for this script
    # (since it's not using numba), there's no need to centrally create a download dictionary with each input's datatype
    # just once on the scheduler, as is more efficient for scripts that use numba.
    # Creates a list of input directories used in summative output creation based on specifics of the model run

    # # Only make 10x10s of the summative outputs. It keeps the workload smaller and these are the only ones that
    # # have per-pixel outputs, which also need to be aggregated into 10x10s.
    # output_dir_list_per_ha = uu.create_output_dir_name_list(cn.LULUCF_summative_output_dirs, interval_type, start_year,'4000',
    #                                                  model_type, interval_end_years, interval_year_diff, input_date, "per_ha")
    # output_dir_list_per_pixel = uu.create_output_dir_name_list(cn.LULUCF_summative_output_dirs, interval_type, start_year,'4000',
    #                                                  model_type, interval_end_years, interval_year_diff, input_date, "per_pixel")
    #
    # # Also need to aggregate the land state nodes
    # land_state_node_list = [s for s in cn.LULUCF_core_output_dirs if cn.land_state_pattern in s]
    # land_state_node_list = uu.create_output_dir_name_list(land_state_node_list, interval_type, start_year,'4000',
    #                                                  model_type, interval_end_years, interval_year_diff, input_date)
    #
    # # Full list of folders to aggregate, in alphabetical order
    # output_dir_list = output_dir_list_per_ha + output_dir_list_per_pixel + land_state_node_list
    output_dir_list.sort()  # Sorts in-place

    # Limits folders to process (for testing)
    if first_folders_to_process:
        output_dir_list = output_dir_list[:first_folders_to_process]

    main_logger.info(f"Directories to zarr: {output_dir_list}")
    main_logger.info(f"There are {len(output_dir_list)} folders to convert into global zarrs")

    build_mega_zarr(output_dir_list, main_logger)

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)

    main_logger.info(f"Zarring complete: {uu.timestr()}")

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
    parser.add_argument('-ffol', '--first_folders_to_process', type=int, help='Number of folders to process from input list')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    input_date = args.input_date
    year_range = args.year_range
    first_folders_to_process = args.first_folders_to_process
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    main(cluster_name, year_range, input_date, run_local, no_stats, no_log, no_upload,
         first_folders_to_process=first_folders_to_process, log_note=log_note)