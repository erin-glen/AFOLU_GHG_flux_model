"""
Creates 10x10 deg geotifs from global rechunked zarr.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test (Dask part does not work because of client.submit()):
python -m src.LULUCF.scripts.vegetation_model.4a_aggregate_outputs_to_10x10deg -bb 10 49 11 50 --run_local --no_upload -fy 1 -fv 1 -ft 1 --input_date YYYYMMDD

Coiled small tests (needs 64 GB because of per-ha and per-pixel outputs):
python -m src.utilities.create_cluster -n 1 -t 1 -m 64 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4a_aggregate_outputs_to_10x10deg -cn vegetation_model -mcstn vegetation_fluxes_1x1_chunk_statistics_20251027_16_16_26__v1_0_2_1884_chunk_run__KEEP.xlsx -bb 10 49 11 50 fy 2 -fv 2 -ft 2 --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.4a_aggregate_outputs_to_10x10deg -cn vegetation_model -mcstn parquet_20250921_17_33_57__XYX/LULUCF_fluxes_20250921_17_33_45_XYZ -bb 10 49 11 50 fy 2 -fv 2 -ft 2 --input_date YYYYMMDD

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 64 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4a_aggregate_outputs_to_10x10deg -cn vegetation_model -mcstn vegetation_fluxes_1x1_chunk_statistics_20251027_16_16_26__v1_0_2_1884_chunk_run__KEEP.xlsx -bb -64 -22 -63 -21 fy 3 -fv 3 -ft 3 --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.4a_aggregate_outputs_to_10x10deg -cn vegetation_model -mcstn parquet_20250921_17_33_57__XYX/LULUCF_fluxes_20250921_17_33_45_XYZ -bb -64 -22 -63 -21 fy 3 -fv 3 -ft 3 --input_date YYYYMMDD

Coiled Cerrado test (174 features):
python -m src.utilities.create_cluster -n 20 -t 1 -m 64 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4a_aggregate_outputs_to_10x10deg -cn vegetation_model -mcstn vegetation_fluxes_1x1_chunk_statistics_20251027_16_16_26__v1_0_2_1884_chunk_run__KEEP.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__Cerrado_center_in.shp --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.4a_aggregate_outputs_to_10x10deg -cn vegetation_model -mcstn parquet_20250921_17_33_57__XYX/LULUCF_fluxes_20250921_17_33_45_XYZ -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__Cerrado_center_in.shp --input_date YYYYMMDD

Coiled large shapefile test (1884 features):
python -m src.utilities.create_cluster -n 100 -t 1 -m 64 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4a_aggregate_outputs_to_10x10deg -cn vegetation_model -mcstn vegetation_fluxes_1x1_chunk_statistics_20251027_16_16_26__v1_0_2_1884_chunk_run__KEEP.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.4a_aggregate_outputs_to_10x10deg -cn vegetation_model -mcstn parquet_20250921_17_33_57__XYX/LULUCF_fluxes_20250921_17_33_45_XYZ -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD

Full run:
python -m src.utilities.create_cluster -n 200 -t 1 -m 64 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4a_aggregate_outputs_to_10x10deg -cn vegetation_model -mcstn vegetation_fluxes_1x1_chunk_statistics_20251027_16_16_26__v1_0_2_1884_chunk_run__KEEP.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --input_date YYYYMMDD --log_note "This is a global run for model v1.0.0 (2016-2024)."
python -m src.LULUCF.scripts.vegetation_model.4a_aggregate_outputs_to_10x10deg -cn vegetation_model -mcstn parquet_20250921_17_33_57__XYX/LULUCF_fluxes_20250921_17_33_45_XYZ -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --input_date YYYYMMDD --log_note "This is a global run for model v1.0.0 (2016-2024)."

Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/690a21cd-2ea0-8333-9c7f-7091f8016fb3
"""

import argparse
import numpy as np
import pandas as pd
import rasterio
import tempfile
import fsspec
import time
import psutil
import os
from dask.distributed import print
import zarr

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import resize_cluster


def main(cluster_name, input_date, run_local, no_log, no_upload, model_chunk_stats_table_name, chunk_shapefile_uri=False, bounding_box=None,
         first_variables_to_process=None, first_years_to_process=None,
         first_tiles_to_process=None, log_note=None):


    ### Step 1: Preparation

    # Model stage being run
    stage = 'vegetation_aggregation_to_10x10_deg'
    model_type = 'standard_model'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    start_time = uu.timestr() # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Input date: {input_date}")

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_size_deg = 1   # Chunk size for geotifs is set at 1x1 deg
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size_deg, None, fishnet_iso_df, main_logger)

    # Gets a list of unique tile_ids from the chunk list
    tile_ids = []
    for chunk in chunk_list:
        tile_id = uu.xy_to_tile_id(chunk[0], chunk[3])  # tile_id in YYN/S_XXXE/W
        tile_ids.append(tile_id)

    unique_tile_ids = list(set(tile_ids))

    # Limits the processed variables to the supplied number (for testing)
    if first_variables_to_process:
        vars_to_process = cn.full_outputs_to_zarr[0:first_variables_to_process]
    else:
        vars_to_process = cn.full_outputs_to_zarr
    main_logger.info(f"Variables to aggregate to 10x10 deg and compare chunk stats for: {vars_to_process} ({len(vars_to_process)} out of {len(cn.full_outputs_to_zarr)})")

    if first_years_to_process:
        years_to_process = first_years_to_process
    else:
        years_to_process = len(cn.interval_end_years_annual)
    main_logger.info(f"Years to aggregate to 10x10 deg and compare chunk stats for: {years_to_process} out of {len(cn.interval_end_years_annual)}")

    if first_tiles_to_process:
        tile_ids_to_process = unique_tile_ids[0:first_tiles_to_process]
    else:
        tile_ids_to_process = unique_tile_ids
    main_logger.info(f"tile_ids to aggregate to 10x10 deg and compare chunk stats for: {tile_ids_to_process} ({len(tile_ids_to_process)} out of {len(unique_tile_ids)})")

    # Determines if the output file names for final versions of outputs should be used
    is_large_run = False
    # is_large_run = True  # For simulating a large run
    if len(tile_ids) > 20:
        is_large_run = True
        main_logger.info(f"Running as large-scale run model: {is_large_run}")

    # The zarr path that's being used (rechunked zarr)
    zarr_path = zu.create_mega_zarr_paths(cn.zarr_pixel_chunks, 'annual', model_type, input_date)
    main_logger.info(f"Aggregating from rechunked zarr (10000 pixel chunks): {zarr_path}")

    output_base = f"{cn.outputs_path}PATTERN/{model_type}/annual_intervals/START_END/PER_HA_OR_PIXEL/{cn.full_raster_dims}_pixels/{input_date}/"
    main_logger.info(f"Core output path for aggregation: {output_base}")


    ### Step 2: Prepare model chunk stats for comparison with zarr chunk stats

    main_logger.info(f"Reading local model chunk stats tables: {uu.timestr()}")
    model_chunk_stats_path = os.path.join(cn.local_chunk_stats_path, model_chunk_stats_table_name)

    # Text added to output chunk stats table name(s) (Excel or Parquet)
    comparison_insert = "_10x10_deg_aggregation_comparison"

    tables_to_compare_dict, zarr_comparison_stats_name, zarr_comparison_stats_path = zu.get_table_names_for_zarr_stats_comparison(
        comparison_insert, main_logger, model_chunk_stats_path)

    model_10x10_counts_df = tables_to_compare_dict[cn.counts_1x1_in_10x10]


    ### Step 3: Create 10x10 deg outputs

    futures = []

    main_logger.info(f"Starting processing: {uu.timestr()}")

    for var in vars_to_process:

        for year_idx in range(years_to_process):

            for tile_id in tile_ids_to_process:

                future = client.submit(uu.extract_10x10,
                                       var, year_idx, tile_id, zarr_path, output_base, no_upload)
                futures.append(future)

    # Results is a list of single-element lists, each with a dictionary of chunk stats for a single variable-year-tile.
    results = client.gather(futures)

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


    ### Step 3: Compares pixel counts in original 1x1 deg geotifs to pixel counts in 10x10 deg geotifs

    # Resizes cluster down to 1 worker for chunk stats and log aggregation since that only needs a minimal remainder of the
    # cluster, not all the workers.
    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster(cluster_name, 1)

    # Flattens the tile-level lists of chunk stat dictionaries into a single list.
    counts_10x10_stats_list = [item for sublist in results for item in sublist]

    # Converts the pixel counts in the 10x10s into a dataframe
    counts_10x10_df = pd.DataFrame(counts_10x10_stats_list)

    # Merges the pixel counts for the 10x10 tiles against the pixel counts for the 1x1s
    merged_10x10_counts_df = model_10x10_counts_df.merge(counts_10x10_df, on='tile_name', how='left')

    # Gets the difference between pixel counts in 10x10s and 1x1s for each tile
    merged_10x10_counts_df['pixel_count_diff'] = merged_10x10_counts_df['total_count'] - merged_10x10_counts_df['count_value']
    max_pixel_count_diff = merged_10x10_counts_df['pixel_count_diff'].abs().max()
    if max_pixel_count_diff > 0:
        main_logger.warning(f"WARNING: at least one tile has a difference in pixel counts between 1x1s and 10x10s! Max difference is {max_pixel_count_diff}: {uu.timestr()}")
    else:
        main_logger.info(f"No tiles have a difference in pixel counts between 1x1s and 10x10s.")

    # Number of rows from model output without matching 10x10 aggregation pixel counts
    main_logger.info(f"Rows without pixel count comparison: {merged_10x10_counts_df['pixel_count_diff'].isna().sum()}")

    # Prepares 10x10 deg chunk stats spreadsheet: pixel count for outputs
    uu.aggregate_10x10_chunk_stats(merged_10x10_counts_df, stage, no_upload, main_logger)


    ### Step 4: Aggregates logs

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

    parser = argparse.ArgumentParser(description="Create a global rechunked mega-zarr.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--input_date', help='Date of run, in YYYYMMDD')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-fv', '--first_variables_to_process', type=int, help='Number of variables to process from raw mega-zarr (for testing)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-ft', '--first_tiles_to_process', type=int, help='Number of tiles to process (for testing)')
    parser.add_argument('-fy', '--first_years_to_process', type=int, help='Number of years to process from raw mega-zarr (for testing)')
    parser.add_argument('-mcstn', '--model_chunk_stats_table_name', help='s3 path for model chunk stats table that will be compared with zarr chunk stats')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    input_date = args.input_date
    bounding_box = args.bounding_box
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_tiles_to_process = args.first_tiles_to_process
    first_variables_to_process = args.first_variables_to_process
    first_years_to_process = args.first_years_to_process
    model_chunk_stats_table_name = args.model_chunk_stats_table_name
    log_note = args.log_note

    run_local = args.run_local
    no_log = args.no_log
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, input_date, run_local, no_log, no_upload, model_chunk_stats_table_name, chunk_shapefile_uri, bounding_box=bounding_box,
         first_variables_to_process=first_variables_to_process, first_years_to_process=first_years_to_process,
         first_tiles_to_process=first_tiles_to_process, log_note=log_note)
