"""
Creates 10x10 deg per-hectare and per-pixel geotifs from global zarr for numeric model outputs.
It creates a task list for all datasets, years, and 10x10 deg tiles for the variables, years, and area of interest,
then runs that giant task list in parallel in batches (as a safeguard against failure during a large task list).

Providing a bounding box with -bb or a chunk shapefile limits the 10x10 deg creation
to the 10x10 deg tiles that contain the bounding box or shapefile.
The entire 10x10 deg tile that contains the selected chunks will be processed (not just the parts with the selected chunks).

The chunk stats table argument (xlsx or Parquet) allows the pixel counts in the 10x10 deg tiles to be compared to
the pixel counts in the constituent 1x1 deg tiles to make sure that pixels aren't being lost during 10x10 deg tile
creation.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test (Dask part does not work because of client.submit()):
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.2_SOC_outputs_to_10x10deg -bb 10 49 11 50 --run_local --no_upload -mt standard -mpd global -fy 1 -fv 1 -ft 1 --input_date YYYYMMDD

Coiled small tests (needs 32 GB because of per-ha and per-pixel outputs):
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn mineral_soil
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.2_SOC_outputs_to_10x10deg -cn mineral_soil -bb 10 49 11 50 -fy 2 -fv 2 -ft 2 -mt standard -mpd global -mcstn soil_carbon_densities_and_changes_1x1_chunk_statistics_20251224_20_16_36__KEEP.xlsx  --input_date YYYYMMDD

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn mineral_soil
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.2_SOC_outputs_to_10x10deg -cn mineral_soil -bb -64 -22 -63 -21 -fy 3 -fv 3 -ft 3 -mt standard -mpd global -mcstn soil_carbon_densities_and_changes_1x1_chunk_statistics_20251224_20_16_36__KEEP.xlsx --input_date YYYYMMDD

Coiled Cerrado test (174 features):
python -m src.utilities.create_cluster -n 20 -t 1 -m 32 -cn mineral_soil
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.2_SOC_outputs_to_10x10deg -cn mineral_soil -mt standard -mpd global -mcstn soil_carbon_densities_and_changes_1x1_chunk_statistics_20251224_20_16_36__KEEP.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__Cerrado_center_in.shp --input_date YYYYMMDD

Coiled large shapefile test (1884 features):
python -m src.utilities.create_cluster -n 100 -t 1 -m 32 -cn mineral_soil
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.2_SOC_outputs_to_10x10deg -cn mineral_soil -mt standard -mpd global -mcstn soil_carbon_densities_and_changes_1x1_chunk_statistics_20251224_20_16_36__KEEP.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD

Full run:
python -m src.utilities.create_cluster -n 200 -t 1 -m 32 -cn mineral_soil
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.2_SOC_outputs_to_10x10deg -cn mineral_soil -mt standard -mpd global -mcstn soil_carbon_densities_and_changes_1x1_chunk_statistics_20251224_20_16_36__KEEP.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --input_date YYYYMMDD -log_note "10x10 deg tile creation for vegetation model v1.0.5 (2016-2024)."

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

import psutil
import numpy as np
import fsspec
import zarr
import time
import resource
from rasterio.transform import from_origin
# Extracts a 10x10° tile from a Zarr store and writes to GeoTIFF on S3
def create_10x10_deg_geotif_from_zarr(var, year_idx, tile_id, raw_path, output_base,
                                      model_version, model_type, model_path_description, no_upload, use_start_year):

    process = psutil.Process(os.getpid())

    logger_worker = lu.setup_logging_worker()

    # Convert tile_id to bounding box (W, S, E, N)
    min_x, min_y, max_x, max_y = uu.get_10x10_tile_bounds(tile_id)

    # Establishes year/year range and units for dataset
    if ("density" in var) and (not cn.starting_C_pools_LC_masked_source_flag_pattern in var):
        per_ha_units = cn.C_density_pixel_meaning
        per_pixel_units = cn.C_per_pixel_pixel_meaning
        coarse_units = cn.C_density_aggreg_pixel_meaning
        # var_per_ha = f"{var}{per_ha_units}"
        var_per_ha = var  #TODO using this for SOC 10x10 creation because zarr layers don't have units at end (added later). Delete for next OGH soil version.
    elif "emis" in var:
        per_ha_units = cn.flux_density_pixel_meaning
        per_pixel_units = cn.flux_per_pixel_pixel_meaning
        coarse_units = cn.flux_aggreg_pixel_meaning
        var_per_ha = f"{var}{per_ha_units}"
    elif "removals" in var:
        per_ha_units = cn.flux_density_pixel_meaning
        per_pixel_units = cn.flux_per_pixel_pixel_meaning
        coarse_units = cn.flux_aggreg_pixel_meaning
        var_per_ha = f"{var}{per_ha_units}"
    elif "net" in var:
        per_ha_units = cn.flux_density_pixel_meaning
        per_pixel_units = cn.flux_per_pixel_pixel_meaning
        coarse_units = cn.flux_aggreg_pixel_meaning
        var_per_ha = f"{var}{per_ha_units}"
    elif cn.land_state_pattern in var:
        per_ha_units = ""
        per_pixel_units = ""
        coarse_units = ""
        var_per_ha = var
    elif "change" in var:  # For SOC change
        per_ha_units = cn.flux_density_pixel_meaning
        per_pixel_units = cn.flux_per_pixel_pixel_meaning
        coarse_units = cn.flux_aggreg_pixel_meaning
        # var_per_ha = f"{var}{per_ha_units}" #TODO
        var_per_ha = var  # using this for SOC 10x10 creation because zarr layers don't have units at end (added later). Delete for next OGH soil version.
    else:
        per_ha_units = ""
        per_pixel_units = ""
        coarse_units = ""
        var_per_ha = var

    # If creating outputs from the model start year, it just uses that year.
    # Renames variable to use units and year.
    if use_start_year == True:
        year = cn.first_model_year_annual
        var_with_unit = var_per_ha
    else:      # For timeseries data, uses specified output years (e.g., vegetation, SOC density, SOC change)
        if "SOC_density" in var:
            year = cn.SOC_density_intervals[year_idx]
        elif "SOC_change" in var:
            year = cn.SOC_change_intervals[year_idx]
        else:  # Vegetation timeseries
            year = cn.interval_end_years_annual[year_idx]
        var_with_unit = var_per_ha  # Doesn't add year to variable/unit name

    # Open Zarr group using fsspec mapper
    fs = fsspec.filesystem("s3", anon=False)
    model_zarr_store = zarr.open_group(fs.get_mapper(raw_path), mode="r")

    # Determine pixel indices (applies to model outputs and pixel area)
    lat_array_model = model_zarr_store["y"][:]
    lon_array_model = model_zarr_store["x"][:]

    # Get index ranges (applies to model outputs and pixel area)
    y0_model = np.searchsorted(lat_array_model[::-1], max_y, side='right')
    y1_model = np.searchsorted(lat_array_model[::-1], min_y, side='left')
    x0_model = np.searchsorted(lon_array_model, min_x, side='left')
    x1_model = np.searchsorted(lon_array_model, max_x, side='right')

    # Flips y indices since lat is descending
    y0_model, y1_model = len(lat_array_model) - y1_model, len(lat_array_model) - y0_model
    if y0_model > y1_model:
        y0_model, y1_model = y1_model, y0_model

    lu.print_and_log(f"  Extracting {var_with_unit} for {year} for {tile_id}: {uu.timestr()}", False, logger_worker)
    extract_start_time = time.time()

    # Loads model output data block
    if "SOC_change" in var:
        # SOC change has no data in the zarr for the first time slice because it has one fewer year than SOC density,
        # so change intervals are actually shifted back by 1 year compared to SOC density
        data_per_ha = model_zarr_store[var_with_unit][year_idx+1, y0_model:y1_model, x0_model:x1_model]
    else:
        data_per_ha = model_zarr_store[var_with_unit][year_idx, y0_model:y1_model, x0_model:x1_model]

    # Calculates per-pixel output (for numeric outputs only)
    pixel_area_zarr_store = uu.get_pixel_area_store()

    # Determine pixel indices (applies to model outputs and pixel area)
    lat_array_pixel_area = pixel_area_zarr_store["y"][:]
    lon_array_pixel_area = pixel_area_zarr_store["x"][:]

    # Get index ranges (applies to model outputs and pixel area)
    y0_pixel_area = np.searchsorted(lat_array_pixel_area[::-1], max_y, side='right')
    y1_pixel_area = np.searchsorted(lat_array_pixel_area[::-1], min_y, side='left')
    x0_pixel_area = np.searchsorted(lon_array_pixel_area, min_x, side='left')
    x1_pixel_area = np.searchsorted(lon_array_pixel_area, max_x, side='right')

    # Flips y indices since lat is descending
    y0_pixel_area, y1_pixel_area = len(lat_array_pixel_area) - y1_pixel_area, len(lat_array_pixel_area) - y0_pixel_area
    if y0_pixel_area > y1_pixel_area:
        y0_pixel_area, y1_pixel_area = y1_pixel_area, y0_pixel_area

    pixel_area = pixel_area_zarr_store['band_data'][y0_pixel_area:y1_pixel_area, x0_pixel_area:x1_pixel_area]
    # print("y0:", y0_pixel_area)
    # print("y1:", y1_pixel_area)
    # print("x0:", x0_pixel_area)
    # print("x1:", x1_pixel_area)
    # print(pixel_area)
    # sys.quit()

    # Converts per-ha to per-pixel
    data_per_pixel = data_per_ha * pixel_area * cn.m2_to_ha

    # Cleanup. Without this, memory exceeds 24GB/worker and eventually tasks get repeated because of too much memory spillage or something
    del pixel_area

    # Creates 0.04x0.04 deg geotif in Mg CO2(e)/0.04x0.04deg pixel/yr
    # per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6941b3f0-30c8-8332-ab19-9b154c0a2b43

    # Trims fine grid so it splits evenly into coarse blocks
    ny, nx = data_per_pixel.shape
    ny_trim = ny - (ny % cn.global_aggregation_factor)
    nx_trim = nx - (nx % cn.global_aggregation_factor)
    data_fine_trim = data_per_pixel[:ny_trim, :nx_trim]

    # Reshapes and sums over each block
    # Deals with NaNs-- otherwise, individual aggregated pixels with NaN might get dropped.
    # per https://chatgpt.com/c/6941b3f0-30c8-8332-ab19-9b154c0a2b43
    coarse_agg = np.nansum(
        data_fine_trim.reshape(
            ny_trim // cn.global_aggregation_factor, cn.global_aggregation_factor,
            nx_trim // cn.global_aggregation_factor, cn.global_aggregation_factor
        ),
        axis=(1, 3)
    )

    # Warning if there are no valid aggregated pixels
    if not np.isfinite(coarse_agg).any():
        logger_worker.warning(f"All-NaN coarse aggregation for {tile_id}, {var}, {year}")

    extract_end_time = time.time()
    lu.print_and_log(f"  Calculated {var_with_unit} for year {year} for {tile_id} in {round(extract_end_time - extract_start_time)} seconds: {uu.timestr()}", False, logger_worker)
    lu.print_and_log(f"  Memory usage after 10x10 extraction for {var_with_unit} for year {year} for {tile_id}: {process.memory_info().rss / 1024 ** 2:.2f} MB", False, logger_worker)

    # Name and s3 folder for per-hectare output
    output_path = output_base.replace("PATTERN", var)
    output_path = output_path.replace("START_END", str(year))
    output_path = output_path.replace(cn.model_version_type_description_placeholder, f"version_{model_version}__{model_type}__{model_path_description}")
    output_path_per_ha = output_path.replace("CHUNK_SIZE_pixels", f"{cn.full_raster_dims}_pixels")
    output_path_per_ha = output_path_per_ha.replace("PER_HA_OR_PIXEL", per_ha_units)
    output_name_per_ha = f"{tile_id}__{var}{per_ha_units}_{str(year)}.tif"
    s3_filename_per_ha = f"{output_path_per_ha}{output_name_per_ha}"

    # Hacky way to fix land_state and other unitless outputs that otherwise have in the pay YYYY//40000_pixels.
    # This removes the extra / .
    s3_filename_per_ha = s3_filename_per_ha.replace("//40000", "/40000")

    # Name and s3 folder for per-pixel output
    output_path_per_pixel = output_path.replace("CHUNK_SIZE_pixels", f"{cn.full_raster_dims}_pixels")
    output_path_per_pixel = output_path_per_pixel.replace("PER_HA_OR_PIXEL", per_pixel_units)
    output_name_per_pixel = f"{tile_id}__{var}{per_pixel_units}_{str(year)}.tif"
    s3_filename_per_pixel = f"{output_path_per_pixel}{output_name_per_pixel}"

    # Name and s3 folder for 0.04x0.04 deg output
    output_path_coarse = output_path.replace("CHUNK_SIZE_pixels", f"{cn.global_aggregation_factor}_pixels")
    output_path_coarse = output_path_coarse.replace("PER_HA_OR_PIXEL", coarse_units)
    output_name_coarse = f"{tile_id}__{var}{coarse_units}_{str(year)}.tif"
    s3_filename_coarse = f"{output_path_coarse}{output_name_coarse}"

    # Uploads to s3 if requested
    if no_upload == False:

        # GeoTransform for 0.00025 deg resolution grid (top-left corner)
        transform = from_origin(min_x, max_y, cn.resolution, cn.resolution)

        # GeoTransform for coarse (0.04 deg) resolution grid
        coarse_transform = from_origin(min_x, max_y, cn.global_geotif_resolution, cn.global_geotif_resolution)

        # Writes per-ha geotif to S3
        valid_pixel_count_per_ha = uu.write_single_geotiff_to_s3(var, year, tile_id, data_per_ha, transform, s3_filename_per_ha, logger_worker)

        # Conditionally writes per-pixel output and 0.04x0.04 res output (only if dataset is float32, i.e. numeric output from model).
        if model_zarr_store[var_with_unit].dtype == np.float32:
            valid_pixel_count_per_pixel = uu.write_single_geotiff_to_s3(
                var, year, tile_id, data_per_pixel, transform, s3_filename_per_pixel, logger_worker
            )

            # valid_pixel_count_coarse not used. Not doing anything with stats from the aggregated output
            valid_pixel_count_coarse = uu.write_single_geotiff_to_s3(
                var, year, tile_id, coarse_agg, coarse_transform, s3_filename_coarse, logger_worker
            )
        else:
            valid_pixel_count_per_pixel = None
            valid_pixel_count_coarse = None

        # # More cleanup. This doesn't actually seem to reduce memory. Leaving it in commented just for reference.
        # del data_per_ha
        # del data_per_pixel

        # Most stats for the 10x10 deg outputs aren't calculated.
        # Only the pixel count is because it is compared to the pixel counts in all the relevant 1x1s.
        # Dictionary is in a list because it's necessary for chunk stats processing later.
        chunk_stats_per_ha = [{
            'chunk_id': 'N/A',
            'tile_id': tile_id,
            'layer_name': output_name_per_ha,
            'tile_name': output_name_per_ha,
            'in_out': 'output_layer',
            'pattern': var,
            'years': year,
            'min_value': 'no data',
            'mean_value': 'no data',
            'max_value': 'no data',
            'count_value': valid_pixel_count_per_ha,
            'sum_value': 'no data',
            'data_type': 'no data'
        }]

        chunk_stats_per_pixel = [{
            'chunk_id': 'N/A',
            'tile_id': tile_id,
            'layer_name': output_name_per_pixel,
            'tile_name': output_name_per_pixel,
            'in_out': 'output_layer',
            'pattern': var,
            'years': year,
            'min_value': 'no data',
            'mean_value': 'no data',
            'max_value': 'no data',
            'count_value': valid_pixel_count_per_pixel,
            'sum_value': 'no data',
            'data_type': 'no data'
        }]

    else:

        # Most stats for the 10x10 aren't calculated.
        # Only the pixel count is because it is compared to the pixel counts in all the relevant 1x1s.
        # Dictionary is in a list because it's necessary for chunk stats processing later.
        chunk_stats_per_ha = [{
            'chunk_id': 'N/A',
            'tile_id': tile_id,
            'layer_name': output_name_per_ha,
            'tile_name': output_name_per_ha,
            'in_out': 'output_layer',
            'pattern': var,
            'years': year,
            'min_value': 'no data',
            'mean_value': 'no data',
            'max_value': 'no data',
            'count_value': 'not calculated',
            'sum_value': 'no data',
            'data_type': 'no data'
        }]

        chunk_stats_per_pixel = [{
            'chunk_id': 'N/A',
            'tile_id': tile_id,
            'layer_name': output_name_per_pixel,
            'tile_name': output_name_per_pixel,
            'in_out': 'output_layer',
            'pattern': var,
            'years': year,
            'min_value': 'no data',
            'mean_value': 'no data',
            'max_value': 'no data',
            'count_value': 'not calculated',
            'sum_value': 'no data',
            'data_type': 'no data'
        }]

    tile_end_time = time.time()
    lu.print_and_log(f"  Total chunk processing for tile {var} for year {year} in {round(tile_end_time - extract_start_time)} seconds: {uu.timestr()}", False, logger_worker)

    # To track peak memory usage
    # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6949a74e-1388-832d-8f8e-5e9bf084ecb8
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak_kb / 1024 ** 2
    lu.print_and_log(f"Peak memory for {tile_id}: {peak_gb:.2f} GB", False, logger_worker)

    return chunk_stats_per_ha, chunk_stats_per_pixel



def main(cluster_name, input_date, model_type, run_local, no_log, no_upload, model_chunk_stats_table_name, chunk_shapefile_uri=False, bounding_box=None,
         first_variables_to_process=None, first_years_to_process=None,
         first_tiles_to_process=None, model_path_description=None, log_note=None):


    ### Step 1: Preparation

    # Model stage being run
    stage = 'soil_carbon_aggregation_to_10x10_deg'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    batch_size = 3000  # X batches to cover all tasks
    # batch_size = 4  # large-scale testing

    start_time = uu.timestr() # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Model version: {cn.SOC_soil_model_version}")
    main_logger.info(f"Model path descriptor: {model_path_description}")
    main_logger.info(f"Input date: {input_date}")
    main_logger.info(f"no_upload: {no_upload}")
    main_logger.info(f"Batch size: {batch_size} tasks")

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

    # # Outputs to turn into 10x10 tiles. #TODO
    # # Separate lists for density and change because they have different numbers of years, so they need to be handled separately
    # full_list_of_vars_density = [
    #     cn.SOC_density_full_extent_pattern,
    #     cn.SOC_density_min_soil_extent_pattern
    # ]

    full_list_of_vars_change = [
        cn.SOC_change_full_extent_pattern,
        cn.SOC_change_min_soil_extent_pattern
    ]

    # Limits the processed variables to the supplied number (for testing)
    if first_variables_to_process:
        # vars_to_process_density = full_list_of_vars_density[0:first_variables_to_process] #TODO
        vars_to_process_change = full_list_of_vars_change[0:first_variables_to_process]
    else:
        # vars_to_process_density = full_list_of_vars_density #TODO
        vars_to_process_change = full_list_of_vars_change
    # main_logger.info(f"Variables to create 10x10 deg density tiles for: {vars_to_process_density} ({len(vars_to_process_density)} out of {len(full_list_of_vars_density)})") #TODO
    main_logger.info(f"Variables to create 10x10 deg change tiles for: {vars_to_process_change} ({len(vars_to_process_change)} out of {len(full_list_of_vars_change)})")

    # Limits the processed years to the supplied number (for testing)
    if first_years_to_process:
        years_to_process_density = first_years_to_process
        years_to_process_change = first_years_to_process
    else:
        years_to_process_density = len(cn.SOC_density_intervals)
        years_to_process_change = len(cn.SOC_change_intervals)
    main_logger.info(f"Years to aggregate to 10x10 deg density and compare chunk stats for: {years_to_process_density} out of {len(cn.SOC_density_intervals)}")
    main_logger.info(f"Years to aggregate to 10x10 deg change and compare chunk stats for: {years_to_process_change} out of {len(cn.SOC_change_intervals)}")

    if first_tiles_to_process:
        tile_ids_to_process = unique_tile_ids[0:first_tiles_to_process]
    else:
        tile_ids_to_process = unique_tile_ids
    main_logger.info(f"tile_ids to aggregate to 10x10 deg and compare chunk stats for: {tile_ids_to_process} ({len(tile_ids_to_process)} out of {len(unique_tile_ids)})")

    # lat-long chunk size for source zarr
    source_zarr_chunk_size = cn.chunk_dims  #4000x4000

    # The zarr path that's being used
    mega_zarr_path = zu.create_zarr_path(cn.SOC_path_mega_zarr, source_zarr_chunk_size, 'N/A',
                                         model_type, cn.SOC_soil_model_version_underscore, model_path_description,
                                         input_date, main_logger)
    main_logger.info(f"Aggregating from zarr ({source_zarr_chunk_size} pixel chunks): {mega_zarr_path}")

    output_base = f"{cn.SOC_outputs_path}PATTERN/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{input_date}/"
    main_logger.info(f"Core output path for aggregation: {output_base}")


    ### Step 2: Prepare model chunk stats for comparison with zarr chunk stats

    main_logger.info(f"Reading local model chunk stats tables: {uu.timestr()}")
    model_chunk_stats_path = os.path.join(cn.local_chunk_stats_path, model_chunk_stats_table_name)

    # Text added to output chunk stats table name(s) (Excel or Parquet)
    comparison_insert = "_10x10_deg_aggregation_comparison"

    tables_to_compare_dict, zarr_comparison_stats_name, zarr_comparison_stats_path = zu.get_table_names_for_zarr_stats_comparison(
        comparison_insert, main_logger, model_chunk_stats_path)

    model_10x10_counts_df = tables_to_compare_dict[cn.counts_1x1_in_10x10]

    # Limits the pixel counts in the model output df to just the model outputs that are being aggregated in this step.
    # That way, pixel count differences between the core model and the aggregation aren't being reported at the end for
    # model outputs that aren't being run here.
    # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69437ae5-a94c-8326-b6d4-ad0dab6bb903
    # pattern = "|".join(vars_to_process_density).join(vars_to_process_change) #TODO
    pattern = "|".join(vars_to_process_change)
    print("pattern:", pattern)
    model_10x10_counts_df = model_10x10_counts_df[model_10x10_counts_df["layer_name"].str.contains(pattern, regex=True, na=False)]


    ### Step 3: Create 10x10 deg outputs

    # # Creates full list of tasks to run (var_name, year_idx, tile_id) #TODO
    # tasks_density = [
    #     (var_name, year_idx, tile_id)
    #     for var_name in vars_to_process_density
    #     for year_idx in range(years_to_process_density)
    #     for tile_id in tile_ids_to_process
    # ]
    tasks_change = [
        (var_name, year_idx, tile_id)
        for var_name in vars_to_process_change
        for year_idx in range(years_to_process_change)
        for tile_id in tile_ids_to_process
    ]
    all_tasks = tasks_change
    # all_tasks = tasks_density+tasks_change #TODO

    # Determines if the output file names for final versions of outputs should be used
    is_large_run = False
    # is_large_run = True  # For simulating a large run
    if len(all_tasks) > 20:
        is_large_run = True
        main_logger.info(f"Running as large-scale run model: {is_large_run}")

    task_batches = [all_tasks[i:i + batch_size] for i in range(0, len(all_tasks), batch_size)]

    # main_logger.info(f"There are {len(all_tasks)} tasks to process ({len(vars_to_process_density) + len(vars_to_process_change)} variables x " #TODO
    main_logger.info(f"There are {len(all_tasks)} tasks to process ({len(vars_to_process_change)} variables x "
                     f"average of {(years_to_process_density + years_to_process_change)/2} years x {len(tile_ids_to_process)} tiles)")

    main_logger.info(f"Tasks divided into {len(task_batches)} batches of up to {batch_size} tasks: {uu.timestr()}")

    all_results = []

    # Iterates through the batches
    # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6980e192-5198-8332-8e69-b994315e91c0
    for i, task_batch in enumerate(task_batches):
        main_logger.info(f"Processing batch {i + 1}/{len(task_batches)} ({len(task_batch)} tasks): {uu.timestr()}")

        futures = []

        for var_name, year_idx, tile_id in task_batch:  # Must be the same order as the tuple in all_tasks

            # future = client.submit(zu.create_10x10_deg_geotif_from_zarr,  #TODO
            future = client.submit(create_10x10_deg_geotif_from_zarr,
                                   var_name, year_idx, tile_id, mega_zarr_path, output_base,
                                   cn.SOC_soil_model_version_underscore, model_type, model_path_description, no_upload, False, retries=3)
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
        batch_results = client.gather(futures)
        # print(batch_results)

        all_results.extend(batch_results)

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


    ### Step 3: Gather worker logs

    # Collects worker logs before moving to processing that doesn't need the cluster
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} worker log compilation", main_logger)


    ### Step 4: Resize cluster down to 1 worker for chunk stats and log aggregation since that only needs a minimal remainder of the
    ### cluster, not all the workers.

    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster(cluster_name, 1)


    ### Step 5: Compare pixel counts in original 1x1 deg geotifs to pixel counts in 10x10 deg geotifs
    ### NOTE: This is comparing the full extent density pixel counts only (not either change extent or mineral soil density).

    # Extracts the per-ha and per-pixel dictionaries from the returned tile stats so they are separate flat lists
    # https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/693c28bf-ade0-8325-b28b-de531cad2408
    counts_per_ha_10x10_stats_list = [ha_dict for (ha_group, pixel_group) in all_results for ha_dict in ha_group]
    counts_per_pixel_10x10_stats_list = [pixel_dict for (ha_group, pixel_group) in all_results for pixel_dict in pixel_group]
    # print(counts_per_ha_10x10_stats_list)
    # print(counts_per_pixel_10x10_stats_list)

    # Converts the pixel counts from per-ha and per-pixel in the 10x10s into dataframes
    counts_per_ha_10x10_df = pd.DataFrame(counts_per_ha_10x10_stats_list)
    counts_per_pixel_10x10_df = pd.DataFrame(counts_per_pixel_10x10_stats_list)
    print("counts_per_pixel_10x10_df:", counts_per_pixel_10x10_df)

    # Merges the per-ha pixel counts for the 10x10 tiles against the pixel counts for the 1x1s
    merged_10x10_counts_per_ha_df = model_10x10_counts_df.merge(counts_per_ha_10x10_df, on='tile_name', how='left')

    # Renames the counts in the 1x1 df from ha to pixel so that their tile names match the per-pixel output
    # and they can be joined. Otherwise, the per-pixel tile names won't match the pixel counts from the 1x1s (since they say ha).
    # per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6945fa31-cd14-8333-bf8a-f71cc6918ae6
    model_10x10_counts_df.loc[:, 'layer_name'] = (
        model_10x10_counts_df['layer_name'].str.replace('_ha_', '_pixel_', regex=False)
    )
    model_10x10_counts_df.loc[:, 'tile_name'] = (
        model_10x10_counts_df['tile_name'].str.replace('_ha_', '_pixel_', regex=False)
    )
    print("model_10x10_counts_df:", model_10x10_counts_df)

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

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with chunk stat comparison", main_logger)


    # ### Step 6: Count output geotifs in s3
    #
    # # Counts per-hectare outputs
    # output_dir_list_per_ha = uu.create_output_dir_name_list([cn.SOC_change_full_extent_dir, cn.SOC_density_full_extent_dir],
    #                                                         '', cn.first_model_year_annual,
    #                                                  cn.full_raster_dims, model_type, cn.veg_model_version_underscore, model_path_description, cn.SOC_change_intervals,
    #                                                  interval_year_diff_list, input_date, False, "per_ha")
    # output_dir_list_per_ha.sort()  # Alphabetically order the outputs (modifies output_dir_list_per_ha)
    # if is_large_run:
    #     main_logger.info(f"output_dir_list_per_ha for {stage}:")
    #     for item in output_dir_list_per_ha:
    #         main_logger.info(f"  {item}")
    # # print(output_dir_list_per_ha)

    # # Iterates through output folders and counts the number of output rasters (only if uploads enabled and a large run (to save console space))
    # if not no_upload and is_large_run:
    #     for output_folder in output_dir_list_per_ha:
    #         geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
    #         main_logger.info(f"Output per-ha rasters in {output_folder}: {file_count}")
    #         # print(geotiff_files)
    #
    # # Counts per-pixel outputs
    # output_dir_list_per_pixel = uu.create_output_dir_name_list(cn.veg_summative_output_dirs, 'annual', cn.first_model_year_annual,
    #                                                  cn.full_raster_dims, model_type, cn.veg_model_version_underscore, model_path_description, interval_end_years,
    #                                                  interval_year_diff_list, input_date, False, "per_pixel")
    # output_dir_list_per_pixel.sort()
    # if is_large_run:
    #     main_logger.info(f"output_dir_list_per_pixel for {stage}:")
    #     for item in output_dir_list_per_pixel:
    #         main_logger.info(f"  {item}")
    # # print(output_dir_list_per_pixel)
    #
    # if not no_upload and is_large_run:
    #     for output_folder in output_dir_list_per_pixel:
    #         geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
    #         main_logger.info(f"Output per-pixel rasters in {output_folder}: {file_count}")
    #         # print(geotiff_files)
    #
    # # Counts 0.04x0.04 deg outputs
    # output_dir_list_aggreg = uu.create_output_dir_name_list(cn.veg_summative_output_dirs, 'annual', cn.first_model_year_annual,
    #                                                  cn.global_aggregation_factor, model_type, cn.veg_model_version_underscore, model_path_description, interval_end_years,
    #                                                  interval_year_diff_list, input_date, False, "_0_04deg_yr")
    # output_dir_list_aggreg.sort()
    # if is_large_run:
    #     main_logger.info(f"output_dir_list_aggreg for {stage}:")
    #     for item in output_dir_list_aggreg:
    #         main_logger.info(f"  {item}")
    # # print(output_dir_list_aggreg)
    #
    # if not no_upload and is_large_run:
    #     for output_folder in output_dir_list_aggreg:
    #         geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
    #         main_logger.info(f"Output aggregated rasters in {output_folder}: {file_count}")
    #         # print(geotiff_files)
    #
    # uu.stage_duration(start_time, uu.timestr(), f"{stage} with output counts", main_logger)


    ### Step 7: Merge worker and local logs
    if not run_local:

        # Adds the workers' logs to the main log and uploads to s3
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)

    # Closes the Dask client if not running locally
    if not run_local:
        client.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create 10x10 deg per-ha and per-pixel output geotifs")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-id', '--input_date', required=True, help='Date of run, in YYYYMMDD')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-fv', '--first_variables_to_process', type=int, help='Number of variables to process from raw zarr (for testing). '
                                                'Actually, doubles this because it includes the specified number of density variables and change variables')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-ft', '--first_tiles_to_process', type=int, help='Number of tiles to process (for testing)')
    parser.add_argument('-fy', '--first_years_to_process', type=int, help='Number of years to process from raw mega-zarr (for testing)')
    parser.add_argument('-mcstn', '--model_chunk_stats_table_name', required=True, help='s3 path for model chunk stats table that will be compared with zarr chunk stats')
    parser.add_argument('-mt', '--model_type', default='standard', help='Type of model run (e.g., standard).')
    parser.add_argument('-mpd', '--model_path_description', help='Description of model run (e.g., global, test, X_area).')
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
    model_type = args.model_type
    model_path_description = args.model_path_description
    log_note = args.log_note

    run_local = args.run_local
    no_log = args.no_log
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, input_date, model_type, run_local, no_log, no_upload, model_chunk_stats_table_name, chunk_shapefile_uri, bounding_box=bounding_box,
         first_variables_to_process=first_variables_to_process, first_years_to_process=first_years_to_process,
         first_tiles_to_process=first_tiles_to_process, model_path_description=model_path_description, log_note=log_note)
