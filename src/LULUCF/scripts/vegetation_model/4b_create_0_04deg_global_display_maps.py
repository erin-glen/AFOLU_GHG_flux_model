"""
Creates 10x10 deg geotifs from global rechunked zarr.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test (Dask part does not work because of client.submit()):
python -m src.LULUCF.scripts.vegetation_model.4b_create_0_04deg_global_display_maps -bb 10 49 11 50 --run_local --no_upload -fy 1 -fv 1 -ft 1 --input_date YYYYMMDD

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 8 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4b_create_0_04deg_global_display_maps -cn vegetation_model -bb 10 49 11 50 fy 2 -fv 2 -ft 2 --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.4b_create_0_04deg_global_display_maps -cn vegetation_model -bb 10 49 11 50 fy 2 -fv 2 -ft 2 --input_date YYYYMMDD

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 8 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4b_create_0_04deg_global_display_maps -cn vegetation_model -bb -64 -22 -63 -21 fy 3 -fv 3 -ft 3 --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.4b_create_0_04deg_global_display_maps -cn vegetation_model -bb -64 -22 -63 -21 fy 3 -fv 3 -ft 3 --input_date YYYYMMDD

Coiled Cerrado test (174 features):
python -m src.utilities.create_cluster -n 50 -t 1 -m 8 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4b_create_0_04deg_global_display_maps -cn vegetation_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__Cerrado_center_in.shp --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.4b_create_0_04deg_global_display_maps -cn vegetation_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__Cerrado_center_in.shp --input_date YYYYMMDD

Coiled large shapefile test (1884 features):
python -m src.utilities.create_cluster -n 15 -t 1 -m 8 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4b_create_0_04deg_global_display_maps -cn vegetation_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.4b_create_0_04deg_global_display_maps -cn vegetation_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD

Full run:
python -m src.utilities.create_cluster -n 15 -t 1 -m 8 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4b_create_0_04deg_global_display_maps -cn vegetation_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --input_date YYYYMMDD --log_note "This is a global run for model v1.0.0 (2016-2024)."
python -m src.LULUCF.scripts.vegetation_model.4b_create_0_04deg_global_display_maps -cn vegetation_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --input_date YYYYMMDD --log_note "This is a global run for model v1.0.0 (2016-2024)."

Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6912af84-deb4-832d-81f0-da2b22b0737d
"""

import argparse
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import rasterio
import tempfile
import fsspec
import time
import psutil
import os
from dask.distributed import print
from rasterio.transform import from_origin
import zarr

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import resize_cluster

# Pixel aggregation parameters
AGG_FACTOR = int(0.04 / 0.00025)  # 160 native pixels per 0.04 deg

def write_global_geotiff_to_s3(data, transform, s3_path, logger, year, var):
    fs = fsspec.filesystem("s3", anon=False)

    height, width = data.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "LZW",
        "nodata": 0,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    print(f"  Writing {var} for {year} to {s3_path}")
    start = time.time()

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=True) as tmpfile:
        with rasterio.open(tmpfile.name, "w", **profile) as dst:
            dst.write(data.astype("float32"), 1)
        fs.put_file(tmpfile.name, s3_path)

    print(f"  Finished writing in {round(time.time() - start)} seconds")


def process_variable_year(var, year_idx, input_date, zarr_path, pixel_area_path, output_base, no_upload):
    """
    Processes a single variable-year pair from the global vegetation model Zarr.
    Converts per-hectare data to per-pixel using pixel area, aggregates to 0.04°,
    and uploads the resulting GeoTIFF to S3.
    """

    logger = lu.setup_logging_worker()
    year = cn.interval_end_years_annual[year_idx]
    year_label = f"{year - 1}_{year}" if "flux" in var or "net" in var else f"{year}"

    print(f"Processing: {var} {year}")

    # --- Open Zarr groups ---
    fs = fsspec.filesystem("s3", anon=False)
    model_zarr = zarr.open_group(fs.get_mapper(zarr_path), mode="r")
    pixel_area_zarr = zarr.open_group(fs.get_mapper(pixel_area_path), mode="r")

    # --- Coordinate arrays ---
    lat_model = model_zarr["y"][:]
    lon_model = model_zarr["x"][:]
    lat_pixel = pixel_area_zarr["y"][:]
    lon_pixel = pixel_area_zarr["x"][:]

    # --- Determine overlap ---
    lat_min = max(lat_model.min(), lat_pixel.min())
    lat_max = min(lat_model.max(), lat_pixel.max())
    print(f"Latitude overlap: {lat_min:.6f} to {lat_max:.6f}")

    # --- Model slice indices (lat_model is descending) ---
    y0_model = np.searchsorted(lat_model[::-1], lat_max, side="right")
    y1_model = np.searchsorted(lat_model[::-1], lat_min, side="left")
    y0_model = len(lat_model) - y1_model
    y1_model = len(lat_model) - y0_model
    if y0_model > y1_model:
        y0_model, y1_model = y1_model, y0_model

    lat_model_slice = lat_model[y0_model:y1_model]
    print(f"Model slice indices for {var} for year {year}: {y0_model} to {y1_model} -> {y1_model - y0_model} rows")

    # --- Pixel area slice indices (matched by coordinate values) ---
    # Round coordinates to avoid floating-point mismatches
    lat_model_vals = np.round(lat_model_slice, 6)
    lat_pixel_vals = np.round(lat_pixel, 6)

    # Find matching latitudes in pixel area Zarr
    lat_indices = np.nonzero(np.isin(lat_pixel_vals, lat_model_vals))[0]

    assert len(lat_indices) > 0, "No overlap found between model and pixel area latitudes!"
    assert len(lat_indices) == len(lat_model_slice), (
        f"Mismatch: model slice has {len(lat_model_slice)} rows, "
        f"but pixel area overlap found {len(lat_indices)}"
    )

    y0_pixel = lat_indices.min()
    y1_pixel = lat_indices.max() + 1  # inclusive slice
    print(f"Pixel area slice indices: {y0_pixel} to {y1_pixel} -> {y1_pixel - y0_pixel} rows")

    # --- Slice arrays from Zarr ---
    data_ha = da.from_zarr(model_zarr[var])[year_idx, y0_model:y1_model, :]
    pixel_area = da.from_zarr(pixel_area_zarr["band_data"])[y0_pixel:y1_pixel, :]

    # --- Shape sanity check ---
    print("SHAPES:")
    print("data_ha.shape:", data_ha.shape)
    print("pixel_area.shape:", pixel_area.shape)
    assert data_ha.shape == pixel_area.shape, f"Shape mismatch: {data_ha.shape} vs {pixel_area.shape}"

    # --- Convert to per-pixel values ---
    print(f"Calculating per-pixel values for {var} for year {year}")
    data_pixel = data_ha * pixel_area / 10000
    print(data_pixel)

    # --- Create xarray DataArray for aggregation ---
    lat_subset = lat_model[y0_model:y1_model]
    data_pixel_xr = xr.DataArray(
        data_pixel,
        dims=("lat", "lon"),
        coords={"lat": lat_subset, "lon": lon_model},
    ).sortby("lat", ascending=False)
    print(data_pixel_xr)

    # --- Aggregate to 0.04° resolution ---
    print(f"Aggregating per-pixel values for {var} for year {year}")
    coarsened = data_pixel_xr.coarsen(lat=AGG_FACTOR, lon=AGG_FACTOR, boundary="trim").sum()
    print(coarsened)
    print(f"Computing per-pixel values for {var} for year {year}")
    coarsened_data = coarsened.compute()
    print(coarsened_data)

    # --- Upload to S3 (if enabled) ---
    print(f"Uploading global map for {var} for year {year}")
    if not no_upload:
        output_name = f"{var}__pixel_yr_summed_0.04deg__{year_label}.tif"
        output_path = f"{output_base}/{output_name}"
        transform = from_origin(-180, lat_subset.max(), 0.04, 0.04)
        write_global_geotiff_to_s3(coarsened_data.values, transform, output_path, logger, year, var)

    return {"var": var, "year": year, "status": "done"}




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

    zarr_path = zu.create_mega_zarr_paths(cn.zarr_pixel_chunks, "annual", model_type, input_date)
    pixel_area_path = cn.pixel_area_global_zarr
    output_base = f"{cn.outputs_path}global_0p04deg_aggregates/{model_type}/annual_intervals/{input_date}"

    main_logger.info(f"Using Zarr: {zarr_path}")
    main_logger.info(f"Pixel area Zarr: {pixel_area_path}")
    main_logger.info(f"Output path: {output_base}")

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
        # vars_to_process = cn.full_outputs_to_zarr[0:first_variables_to_process]
        vars_to_process = cn.veg_summative_output_patterns[0:first_variables_to_process]   #TODO for testing
    else:
        # vars_to_process = cn.full_outputs_to_zarr
        vars_to_process = cn.veg_summative_output_patterns   #TODO for testing
    main_logger.info(f"Variables to rechunk and compare chunk stats for: {vars_to_process} ({len(vars_to_process)} out of {len(cn.full_outputs_to_zarr)})")

    if first_years_to_process:
        years_to_process = first_years_to_process
    else:
        years_to_process = len(cn.interval_end_years_annual)
    main_logger.info(f"Years to rechunk and compare chunk stats for: {years_to_process} out of {len(cn.interval_end_years_annual)}")

    if first_tiles_to_process:
        tile_ids_to_process = unique_tile_ids[0:first_tiles_to_process]
    else:
        tile_ids_to_process = unique_tile_ids
    main_logger.info(f"tile_ids to rechunk and compare chunk stats for: {tile_ids_to_process} ({len(tile_ids_to_process)} out of {len(unique_tile_ids)})")

    # Determines if the output file names for final versions of outputs should be used
    is_large_run = False
    # is_large_run = True  # For simulating a large run
    if len(tile_ids) > 20:
        is_large_run = True
        main_logger.info(f"Running as large-scale run model: {is_large_run}")


    ### Step 3: Create global 0.04x0.04 deg geotifs

    for var in vars_to_process:
        for year_idx in range(years_to_process):
                process_variable_year(var, year_idx, input_date, zarr_path, pixel_area_path, output_base, no_upload)

                main_logger.info(f"Processing complete for {var} for year {year_idx} variable-year combinations: {uu.timestr()}")

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


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
