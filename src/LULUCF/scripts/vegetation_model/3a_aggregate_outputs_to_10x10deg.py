"""
Creates 10x10 deg per-hectare and per-pixel geotifs from global zarr for numeric model outputs.
Coded to run for summative outputs + land state nodes but can change code to run for more or less variables.
Limited to just summative + land state nodes for now because it is quite expensive for just these variables.
It creates a task list for all datasets, years, and 10x10 deg tiles for the variables, years, and area of interest,
then runs that giant task list in parallel.

Providing a bounding box with -bb or a chunk shapefile limits the 10x10 deg creation
to the 10x10 deg tiles that contain the bounding box or shapefile.
The entire 10x10 deg tile that contains the selected chunks will be processed (not just the parts with the selected chunks).

The chunk stats table argument (xlsx or Parquet) allows the pixel counts in the 10x10 deg tiles to be compared to
the pixel counts in the constituent 1x1 deg tiles to make sure that pixels aren't being lost during 10x10 deg tile
creation.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test (Dask part does not work because of client.submit()):
python -m src.LULUCF.scripts.vegetation_model.3a_aggregate_outputs_to_10x10deg -bb 10 49 11 50 --run_local --no_upload -fy 1 -fv 1 -ft 1 --input_date YYYYMMDD

Coiled small tests (needs 32 GB because of per-ha and per-pixel outputs):
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.3a_aggregate_outputs_to_10x10deg -cn vegetation_model -bb 10 49 11 50 fy 2 -fv 2 -ft 2 -mcstn vegetation_fluxes_1x1_chunk_statistics_XYZ.xlsx  --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.3a_aggregate_outputs_to_10x10deg -cn vegetation_model -bb 10 49 11 50 fy 2 -fv 2 -ft 2 -mcstn parquet_20250921_17_33_57__XYX/LULUCF_fluxes_20250921_17_33_45_XYZ --input_date YYYYMMDD

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.3a_aggregate_outputs_to_10x10deg -cn vegetation_model -bb -64 -22 -63 -21 fy 3 -fv 3 -ft 3 -mcstn vegetation_fluxes_1x1_chunk_statistics_XYZ.xlsx --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.3a_aggregate_outputs_to_10x10deg -cn vegetation_model -bb -64 -22 -63 -21 fy 3 -fv 3 -ft 3 -mcstn parquet_20250921_17_33_57__XYX/LULUCF_fluxes_20250921_17_33_45_XYZ --input_date YYYYMMDD

Coiled Cerrado test (174 features):
python -m src.utilities.create_cluster -n 20 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.3a_aggregate_outputs_to_10x10deg -cn vegetation_model -mcstn vegetation_fluxes_1x1_chunk_statistics_XYZ.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__Cerrado_center_in.shp --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.3a_aggregate_outputs_to_10x10deg -cn vegetation_model -mcstn parquet_20250921_17_33_57__XYX/LULUCF_fluxes_20250921_17_33_45_XYZ -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__Cerrado_center_in.shp --input_date YYYYMMDD

Coiled large shapefile test (1884 features):
python -m src.utilities.create_cluster -n 100 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.3a_aggregate_outputs_to_10x10deg -cn vegetation_model -mcstn vegetation_fluxes_1x1_chunk_statistics_XYZ.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.3a_aggregate_outputs_to_10x10deg -cn vegetation_model -mcstn parquet_20250921_17_33_57__XYX/LULUCF_fluxes_20250921_17_33_45_XYZ -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD

Full run:
python -m src.utilities.create_cluster -n 200 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.3a_aggregate_outputs_to_10x10deg -cn vegetation_model -mcstn vegetation_fluxes_1x1_chunk_statistics_XYZ.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --input_date YYYYMMDD --log_note "This is a global run for model v1.0.0 (2016-2024)."
python -m src.LULUCF.scripts.vegetation_model.3a_aggregate_outputs_to_10x10deg -cn vegetation_model -mcstn parquet_20250921_17_33_57__XYX/LULUCF_fluxes_20250921_17_33_45_XYZ -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --input_date YYYYMMDD --log_note "This is a global run for model v1.0.0 (2016-2024)."

Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/690a21cd-2ea0-8333-9c7f-7091f8016fb3
"""

import argparse
import pandas as pd
import os
from dask.distributed import print

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
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

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

    unique_tile_ids = sorted(list(set(tile_ids)))

    # Outputs to turn into 10x10 tile
    # full_list_of_vars = cn.full_outputs_to_zarr   # If all variables are to be made into 10x10s (but very expensive)
    full_list_of_vars = cn.veg_summative_output_patterns + [cn.land_state_pattern] # Summative outputs + land state nodes

    # Limits the processed variables to the supplied number (for testing)
    if first_variables_to_process:
        vars_to_process = full_list_of_vars[0:first_variables_to_process]
    else:
        vars_to_process = full_list_of_vars
    main_logger.info(f"Variables to create 10x10 deg tiles for: {vars_to_process} ({len(vars_to_process)} out of {len(full_list_of_vars)})")

    # Limits the processed years to the supplied number (for testing)
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

    # lat-long chunk size for source zarr
    # source_zarr_chunk_size = cn.full_raster_dims  #10000x10000
    source_zarr_chunk_size = cn.chunk_dims  #4000x4000

    # The zarr path that's being used
    source_mega_zarr_path = zu.create_mega_zarr_path(cn.veg_outputs_path_mega_zarr, source_zarr_chunk_size, 'annual', model_type, input_date, main_logger)
    main_logger.info(f"Aggregating from zarr ({source_zarr_chunk_size} pixel chunks): {source_mega_zarr_path}")

    output_base = f"{cn.veg_outputs_path}PATTERN/{model_type}/annual_intervals/START_END/PER_HA_OR_PIXEL/{cn.full_raster_dims}_pixels/{input_date}/"
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

    for var_name in vars_to_process:

        for year_idx in range(years_to_process):

            for tile_id in tile_ids_to_process:

                # future = client.submit(uu.create_10x10_deg_geotif_from_zarr,
                future = client.submit(uu.create_10x10_deg_geotif_from_zarr,
                                       var_name, year_idx, tile_id, source_mega_zarr_path, output_base, no_upload)
                futures.append(future)

    # Results is a list of tuples, where each tuple is the per-ha and per-pixel chunk stats, each of which is a dictionary
    # for a single variable-year-tile
    # e.g., ([{'chunk_id': 'N/A', 'tile_id': '00N_050W', 'layer_name': '00N_050W__gross_emissions__all_C_pools__CO2_only__MgCO2_ha_yr_2016.tif',
    # 'tile_name': '00N_050W__gross_emissions__all_C_pools__CO2_only__MgCO2_ha_yr_2016.tif', 'in_out': 'output_layer',
    # 'pattern': 'gross_emissions__all_C_pools__CO2_only__MgCO2', 'years': 2016, 'min_value': 'no data', 'mean_value': 'no data', 'max_value': 'no data',
    # 'count_value': 7912448, 'sum_value': 'no data', 'data_type': 'no data'}],
    # [{'chunk_id': 'N/A', 'tile_id': '00N_050W', 'layer_name': '00N_050W__gross_emissions__all_C_pools__CO2_only__MgCO2_pixel_yr_2016.tif',
    # 'tile_name': '00N_050W__gross_emissions__all_C_pools__CO2_only__MgCO2_pixel_yr_2016.tif', 'in_out': 'output_layer',
    # 'pattern': 'gross_emissions__all_C_pools__CO2_only__MgCO2', 'years': 2016, 'min_value': 'no data', 'mean_value': 'no data',
    # 'max_value': 'no data', 'count_value': 7912448, 'sum_value': 'no data', 'data_type': 'no data'}]),
    # ([{'chunk_id': 'N/A', ... 'data_type': 'no data'}])]
    results = client.gather(futures)
    # print(results)

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

    # Extracts the per-ha and per-pixel dictionaries from the returned tile stats so they are separate flat lists
    # https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/693c28bf-ade0-8325-b28b-de531cad2408
    counts_per_ha_10x10_stats_list = [ha_dict for (ha_group, pixel_group) in results for ha_dict in ha_group]
    counts_per_pixel_10x10_stats_list = [pixel_dict for (ha_group, pixel_group) in results for pixel_dict in pixel_group]
    # print(counts_per_ha_10x10_stats_list)
    # print(counts_per_pixel_10x10_stats_list)

    # Converts the pixel counts from per-ha and per-pixel in the 10x10s into dataframes
    counts_per_ha_10x10_df = pd.DataFrame(counts_per_ha_10x10_stats_list)
    counts_per_pixel_10x10_df = pd.DataFrame(counts_per_pixel_10x10_stats_list)

    # Merges the per-ha pixel counts for the 10x10 tiles against the pixel counts for the 1x1s
    merged_10x10_counts_per_ha_df = model_10x10_counts_df.merge(counts_per_ha_10x10_df, on='tile_name', how='left')

    # Renames the counts in the 1x1 df from ha to pixel so that their tile names match the per-pixel output
    # and they can be joined. Otherwise, the per-pixel tile names won't match the pixel counts from the 1x1s (since they say ha).
    model_10x10_counts_df['tile_name'] = (
        model_10x10_counts_df['tile_name'].str.replace('ha', 'pixel', regex=False)
    )

    merged_10x10_counts_per_pixel_df = model_10x10_counts_df.merge(counts_per_pixel_10x10_df, on='tile_name', how='left')

    # Gets the difference between pixel counts in 10x10s and 1x1s for each tile
    merged_10x10_counts_per_ha_df['pixel_count_diff'] = merged_10x10_counts_per_ha_df['total_count'] - merged_10x10_counts_per_ha_df['count_value']
    max_pixel_count_diff_per_ha = merged_10x10_counts_per_ha_df['pixel_count_diff'].abs().max()

    merged_10x10_counts_per_pixel_df['pixel_count_diff'] = merged_10x10_counts_per_pixel_df['total_count'] - merged_10x10_counts_per_pixel_df['count_value']
    max_pixel_count_diff_per_pixel = merged_10x10_counts_per_pixel_df['pixel_count_diff'].abs().max()

    if max_pixel_count_diff_per_ha > 0:
        main_logger.warning(f"WARNING: at least one per-hectare tile has a difference in pixel counts between 1x1s and 10x10s! Max difference is {max_pixel_count_diff_per_ha}: {uu.timestr()}")
    else:
        main_logger.info(f"No per-hectare tiles have a difference in pixel counts between 1x1s and 10x10s.")

    if max_pixel_count_diff_per_pixel > 0:
        main_logger.warning(f"WARNING: at least one per-pixel tile has a difference in pixel counts between 1x1s and 10x10s! Max difference is {max_pixel_count_diff_per_pixel}: {uu.timestr()}")
    else:
        main_logger.info(f"No per-pixel tiles have a difference in pixel counts between 1x1s and 10x10s.")

    # Number of rows from model output without matching 10x10 aggregation pixel counts
    main_logger.info(f"Rows without pixel count comparison for per-ha output: {merged_10x10_counts_per_ha_df['pixel_count_diff'].isna().sum()}")
    main_logger.info(f"Rows without pixel count comparison for per-pixel output: {merged_10x10_counts_per_pixel_df['pixel_count_diff'].isna().sum()}")

    # Prepares 10x10 deg chunk stats spreadsheet: pixel count for outputs
    uu.aggregate_10x10_chunk_stats(merged_10x10_counts_per_ha_df, f"{stage}_per_ha", no_upload, main_logger)
    uu.aggregate_10x10_chunk_stats(merged_10x10_counts_per_pixel_df, f"{stage}_per_pixel", no_upload, main_logger)


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

    parser = argparse.ArgumentParser(description="Create 10x10 deg per-ha and per-pixel output geotifs")
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
