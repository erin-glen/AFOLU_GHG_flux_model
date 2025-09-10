"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Creates a separate zarr for each model output folder

Local test:
python -m src.LULUCF.scripts.core_model.3_create_individual_zarrs -yr 2000 2024 --first_folders_to_process 2 --first_1x1s_to_process 2 --input_date YYYYMMDD --run_local

Coiled small test:
python -m src.utilities.create_cluster -n 2 -m 16 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.core_model.3_create_individual_zarrs -cn LULUCF_postprocessing -yr 2000 2024 --first_folders_to_process 2 --first_1x1s_to_process 2 --input_date YYYYMMDD

Coiled large area test:
python -m src.utilities.create_cluster -n 2 -m 32 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.core_model.3_create_individual_zarrs -cn LULUCF_postprocessing -yr 2000 2024 --first_folders_to_process 2 --input_date YYYYMMDD


Full Coiled run:
python -m src.utilities.create_cluster -n 100 -m 32 -cn LULUCF_postprocessing
python -m src.LULUCF.scripts.core_model.3_create_individual_zarrs -cn LULUCF_postprocessing -yr 2000 2024 --input_date YYYYMMDD

Based on discussion with Justin Terry and https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68bf3334-a09c-8320-a556-153f43ef9cd0
"""

import argparse
import fsspec
import rasterio
import re
import numpy as np
import os
import sys
import time
import rioxarray
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

def build_global_zarr(output_dir_list, first_1x1s_to_process, main_logger):

    main_logger.info(f"Starting Zarr creation for {len(output_dir_list)} folders: {uu.timestr()}")

    for folder in output_dir_list:
        main_logger.info(f"Scanning geotiffs in {folder}: {uu.timestr()}")
        folder_start_time = time.time()

        fs = fsspec.filesystem('s3', use_listings_cache=False, anon=False) if folder.startswith('s3://') else fsspec.filesystem('file')

        if folder.startswith('s3://'):
            keys = fs.glob(os.path.join(folder, '*.tif'))
            chunk_list = [f if f.startswith("s3://") else f"s3://{f}" for f in keys]
        else:
            chunk_list = fs.glob(os.path.join(folder, '*.tif'))

        # Limits chunks to process (for testing)
        if first_1x1s_to_process:
            chunk_list = chunk_list[:first_1x1s_to_process]

        # Sample tif for getting various path and metadata from
        sample_tif = chunk_list[0]

        with rasterio.open(sample_tif) as src:
            data_type = src.dtypes[0]

        layer_pattern = re.search(r"version_\d+_\d+_\d+(?:_[^/]*)?/([^/]+)/", sample_tif).group(1)
        # print(layer_pattern)

        layer_date = re.search(r"intervals/([^/]+)", sample_tif).group(1)
        # print(layer_date)

        layer_unit = re.search(r"([^/]+)/4000", sample_tif).group(1)
        # print(layer_unit)

        # Constructs the output path
        if layer_unit == layer_date:
            out_file = f"{layer_pattern}_{layer_date}.zarr"
        else:
            out_file = f"{layer_pattern}{layer_unit}_{layer_date}.zarr"

        out_path = folder.replace("4000_pixels", cn.zarr_output_pattern)
        out_path_final = out_path + out_file
        # print(out_path_final)

        main_logger.info(f"Opening files in {folder}: {uu.timestr()}")
        da = xr.open_mfdataset(
            chunk_list,
            parallel=True,
            chunks={'x': cn.zarr_pixel_chunks, 'y': cn.zarr_pixel_chunks}  # This doesn't actually rechunk to 10000x10000. It takes it up to 4000x4000.
        ).astype(data_type)

        # print(da)  # To check chunks

        # Explicitly rechunks to 10000x10000
        da = da.chunk({'x': cn.zarr_pixel_chunks, 'y': cn.zarr_pixel_chunks})

        # print(da)  # To check chunks

        main_logger.info(f"Saving {folder} as zarr: {uu.timestr()}")
        original_var_name = list(da.data_vars.keys())[0]
        da = da.rename({original_var_name: out_file})
        da.to_zarr(out_path_final, mode='w')
        # print(da)

        folder_end_time = time.time()
        main_logger.info(f"Zarring {folder} took {round(folder_end_time - folder_start_time)} seconds: {uu.timestr()}")


def main(cluster_name, year_range, input_date, run_local=False, no_stats=False, no_log=False, no_upload=False,
         first_folders_to_process=None, first_1x1s_to_process=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = 'LULUCF_dataset_zarr_creation'
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
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3_zarr_testing_small/carbon_density__AGC__MgC/standard_model/hybrid_intervals/2005/_ha/4000_pixels/20250904/",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3_zarr_testing_small/carbon_density__AGC__MgC/standard_model/hybrid_intervals/2010/_ha/4000_pixels/20250904/",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3_zarr_testing_small/carbon_density__BGC__MgC/standard_model/hybrid_intervals/2005/_ha/4000_pixels/20250904/",
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_0_4_3_zarr_testing_small/carbon_density__BGC__MgC/standard_model/hybrid_intervals/2010/_ha/4000_pixels/20250904/"
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
    main_logger.info(f"There are {len(output_dir_list)} folders to conert into global zarrs")

    build_global_zarr(output_dir_list, first_1x1s_to_process, main_logger)

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
    parser.add_argument('-ften', '--first_1x1s_to_process', type=int, help='Number of chunks to process from input list')
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
    first_1x1s_to_process = args.first_1x1s_to_process
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    main(cluster_name, year_range, input_date, run_local, no_stats, no_log, no_upload,
         first_folders_to_process=first_folders_to_process,
         first_1x1s_to_process=first_1x1s_to_process, log_note=log_note)