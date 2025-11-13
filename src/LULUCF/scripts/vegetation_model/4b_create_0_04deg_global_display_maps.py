"""
Creates global 0.04x0.04 deg geotifs from numeric model outputs (units: Mg CO2(e)/aggregated pixel/yr).
It iterates through each dataset-year combination, processing each one using Dask/xarray before moving on to the next one.
Thus, parallelization is at the level of the dataset-year global map, not across them.
No chunk stats are created, nor checks for completeness/missing data.

Note that this can only be run globally; there's no way to run it on just a part of the world.
Thus, making global geotifs from even small test areas (e.g., Cerrado) needs a full set of workers to finish in a timely manner.
It takes the same amount of time/workers to create a geotif for all land chunks as it does for any subset because
it's running the entire planet regardless.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test (Dask part does not work because of client.submit()):
python -m src.LULUCF.scripts.vegetation_model.4b_create_0_04deg_global_display_maps --run_local --no_upload -fy 1 -fv 1 --input_date YYYYMMDD

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 8 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4b_create_0_04deg_global_display_maps -cn vegetation_model -fy 2 -fv 2 --input_date YYYYMMDD

Coiled large run:
python -m src.utilities.create_cluster -n 80 -t 1 -m 8 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4b_create_0_04deg_global_display_maps -cn vegetation_model --input_date YYYYMMDD

Full run:
python -m src.utilities.create_cluster -n 80 -t 1 -m 8 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.4b_create_0_04deg_global_display_maps -cn vegetation_model --input_date YYYYMMDD --log_note "This is a global run for model v1.0.0 (2016-2024)."

Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6912af84-deb4-832d-81f0-da2b22b0737d
"""

import argparse
import numpy as np
import xarray as xr
import dask.array as da
import fsspec
import time
from dask.distributed import print
from rasterio.transform import from_origin
import zarr

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu

# Processes a single variable-year pair from the global rechunked mega-zarr.
# Converts per-hectare data to per-pixel using pixel area, aggregates to 0.04°,
# and uploads the resulting GeoTIFF to S3.
# From https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6912af84-deb4-832d-81f0-da2b22b0737d
def global_map_for_variable_year(var, year_idx, zarr_path, pixel_area_overlap, output_base, no_upload, main_logger):

    start_time = time.time()

    year = cn.interval_end_years_annual[year_idx]

    main_logger.info(f"Processing {var} for year {year}: {uu.timestr()}")

    # Opens Zarr groups
    fs = fsspec.filesystem("s3", anon=False)

    # TODO Try this with revised mega-zarr creation. Try running with both original (4000) chunks and rechunked versions.
    #This didn't work with the existing mega-zarrs because of that float32/fill value issue, but maybe it will with a pending fix in the zarr creation function.
    #Can adopt this if it works with the new mega-zarrs. Otherwise, revert to the below.

    # Load Dask-backed xarray datasets
    model_ds = xr.open_zarr(fs.get_mapper(zarr_path), chunks={"lat": 10000, "lon": 10000}, consolidated=False)

    #TODO suggested by ChatGPT to reduce warm-up time for first iterations (https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6912af84-deb4-832d-81f0-da2b22b0737d). Haven't tested.
    #=== Warm-up Dask and load metadata (OPTION A) ===
    _ = model_ds.isel(year=0, y=slice(0, 10), x=slice(0, 10)).to_array().compute()

    # Select one year of model data
    data_ha = model_ds[var].isel(time=year_idx)

    # Ensure lat/lon alignment
    lat_min = max(data_ha["lat"].min().item(), pixel_area_overlap["lat"].min().item())
    lat_max = min(data_ha["lat"].max().item(), pixel_area_overlap["lat"].max().item())
    lat_min = np.round(lat_min, 6)
    lat_max = np.round(lat_max, 6)

    # Slice both by coordinates
    data_ha = data_ha.sel(lat=slice(lat_max, lat_min))  # descending

    # Multiply and coarsen
    data_pixel = data_ha * pixel_area_overlap * cn.m2_to_ha
    coarsened = data_pixel.coarsen(lat=cn.global_aggregation_factor, lon=cn.global_aggregation_factor, boundary="trim").sum()
    main_logger.info(f"Computing per-pixel values for {var} for year {year}: {uu.timestr()}")
    coarsened_data = coarsened.compute()


    #TODO The below commented chunk worked using the existing megazarrs (1884 chunk ones) and their fillvalue.
    # Can revert to this if the above doesn't work.

    # model_zarr = zarr.open_group(fs.get_mapper(zarr_path), mode="r")
    # pixel_area_zarr = zarr.open_group(fs.get_mapper(pixel_area_path), mode="r")
    #
    #TODO suggested by ChatGPT to reduce warm-up time for first iterations (https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6912af84-deb4-832d-81f0-da2b22b0737d). Haven't tested.
    #=== Warm-up Dask and load metadata (OPTION A) ===
    # _ = model_ds.isel(year=0, y=slice(0, 10), x=slice(0, 10)).to_array().compute()
    # _ = pixel_area_da.isel(y=slice(0, 10), x=slice(0, 10)).compute()
    #
    #
    # # Code below basically gets the pixel area and model mega-zarr to use the same latitude extent.
    # # It was somewhat convoluted with ChatGPT; the model extent kept not being sliced/clipped correctly.
    #
    # # Coordinate arrays
    # lat_model = model_zarr["y"][:]
    # lon_model = model_zarr["x"][:]
    # lat_pixel = pixel_area_zarr["y"][:]
    # lon_pixel = pixel_area_zarr["x"][:]
    #
    # # Determines overlap in latitude between pixel area (80N-60S) and mega-zarr (90N-90S)
    # lat_min = max(lat_model.min(), lat_pixel.min())
    # lat_max = min(lat_model.max(), lat_pixel.max())
    #
    # # Model slice indices (lat_model is descending)
    # y0_model = np.searchsorted(lat_model[::-1], lat_max, side="right")
    # y1_model = np.searchsorted(lat_model[::-1], lat_min, side="left")
    # y0_model = len(lat_model) - y1_model
    # y1_model = len(lat_model) - y0_model
    # if y0_model > y1_model:
    #     y0_model, y1_model = y1_model, y0_model
    #
    # # Gets the latitude slice of the mega-zarr from the model
    # lat_model_slice = lat_model[y0_model:y1_model]
    # # print(f"Model slice indices for {var} for year {year}: {y0_model} to {y1_model} -> {y1_model - y0_model} rows")
    #
    # # Pixel area slice indices (matched by coordinate values).
    # # Rounds coordinates to avoid floating-point mismatches (this was a problem before).
    # lat_model_vals = np.round(lat_model_slice, 6)
    # lat_pixel_vals = np.round(lat_pixel, 6)
    #
    # # Finds matching latitudes in pixel area Zarr
    # lat_indices = np.nonzero(np.isin(lat_pixel_vals, lat_model_vals))[0]
    #
    # y0_pixel = lat_indices.min()
    # y1_pixel = lat_indices.max() + 1  # inclusive slice
    # # print(f"Pixel area slice indices: {y0_pixel} to {y1_pixel} -> {y1_pixel - y0_pixel} rows")
    #
    # # Slices arrays from Zarrs (overlapping latitude)
    # data_ha = da.from_zarr(model_zarr[var])[year_idx, y0_model:y1_model, :]
    # pixel_area = da.from_zarr(pixel_area_zarr["band_data"])[y0_pixel:y1_pixel, :]
    #
    # # Converts to per-pixel values
    # data_pixel = data_ha * pixel_area * cn.m2_to_ha
    # # print(data_pixel)
    #
    # # Create xarray DataArray for aggregation to 0.04x0.04 deg
    # data_pixel_xr = xr.DataArray(
    #     data_pixel,
    #     dims=("lat", "lon"),
    #     coords={"lat": lat_model_slice, "lon": lon_model},
    # ).sortby("lat", ascending=False)
    # # print(data_pixel_xr)
    #
    # # Aggregates to 0.04x0.04 deg (factor of 160)
    # coarsened = data_pixel_xr.coarsen(lat=cn.global_aggregation_factor, lon=cn.global_aggregation_factor, boundary="trim").sum()
    # main_logger.info(f"Computing per-pixel values for {var} for year {year}: {uu.timestr()}")
    # coarsened_data = coarsened.compute()
    # # print(coarsened_data)

    # Uploads to S3 (if enabled)
    main_logger.info(f"Uploading global map for {var} for year {year}: {uu.timestr()}")

    if not no_upload:

        # Establishes year/year range and units for dataset
        if "density" in var:
            year_or_range = f"{year}"
            global_map_units = cn.C_density_aggreg_pixel_meaning
        elif "emis" in var:
            year_or_range = f"{year - 1}_{year}"
            global_map_units = cn.flux_aggreg_pixel_meaning
        elif "removals" in var:
            year_or_range = f"{year - 1}_{year}"
            global_map_units = cn.flux_aggreg_pixel_meaning
        elif "net" in var:
            year_or_range = f"{year - 1}_{year}"
            global_map_units = cn.flux_aggreg_pixel_meaning
        else:
            year_or_range = f"{year - 1}_{year}"
            global_map_units = cn.flux_aggreg_pixel_meaning

        output_path = output_base.replace("PATTERN", var)
        output_path = output_path.replace("START_END", year_or_range)
        output_path = output_path.replace("PER_HA_OR_PIXEL", global_map_units)
        output_name = f"{var}{global_map_units}_{year_or_range}__global.tif"
        s3_filename = f"{output_path}{output_name}"

        transform = from_origin(-180, lat_model_slice.max(), cn.global_geotif_resolution, cn.global_geotif_resolution)

        data_vals = coarsened_data.values

        valid_pixel_count_per_ha = uu.write_single_geotiff_to_s3(var, year, "global", data_vals, transform, s3_filename, main_logger)
        print(f"valid_pixel_count_per_ha: {valid_pixel_count_per_ha}")

    end_time = time.time()
    main_logger.info(f"  Created global geotif for {var} for {year} in {round(end_time - start_time)} seconds: {uu.timestr()}")

    # Compute summary statistics
    nonzero_vals = data_vals[data_vals != 0]
    data_min = float(np.min(nonzero_vals))
    data_mean = float(np.mean(nonzero_vals))
    data_max = float(np.max(nonzero_vals))

    return {"var": var, "year": year, "status": "done", "min": data_min, "mean": data_mean, "max": data_max}



def main(cluster_name, input_date, run_local, no_log, no_upload,
         first_variables_to_process=None, first_years_to_process=None, log_note=None):


    ### Step 1: Preparation

    # Model stage being run
    stage = 'create_global_0_04x0_04deg_maps'
    model_type = 'standard_model'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    start_time = uu.timestr() # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Input date: {input_date}")

    # rechunked_mega_zarr_path = zu.create_mega_zarr_paths(cn.zarr_pixel_chunks, "annual", model_type, input_date)
    rechunked_mega_zarr_path = zu.create_mega_zarr_paths(4000, "annual", model_type, input_date)
    pixel_area_zarr_path = cn.pixel_area_global_zarr
    output_base = f"{cn.outputs_path}PATTERN/{model_type}/annual_intervals/START_END/PER_HA_OR_PIXEL/{input_date}/"

    main_logger.info(f"Using rechunked mega-zarr: {rechunked_mega_zarr_path}")
    main_logger.info(f"Pixel area zarr: {pixel_area_zarr_path}")
    main_logger.info(f"Core output path for global maps: {output_base}")

    # Outputs to turn into 10x10 tile
    # full_list_of_vars = cn.full_outputs_to_zarr    # If all variables are to be made into 10x10s (but very expensive)
    full_list_of_vars = cn.veg_summative_output_patterns # Summative outputs only

    # Limits the processed variables to the supplied number (for testing)
    if first_variables_to_process:
        vars_to_process = full_list_of_vars[0:first_variables_to_process]
    else:
        vars_to_process = full_list_of_vars
    main_logger.info(f"Variables to create global maps for: {vars_to_process} ({len(vars_to_process)} out of {len(full_list_of_vars)})")

    # Limits the processed years to the supplied number (for testing)
    if first_years_to_process:
        years_to_process = first_years_to_process
    else:
        years_to_process = len(cn.interval_end_years_annual)
    main_logger.info(f"Years to create global maps for: {years_to_process} out of {len(cn.interval_end_years_annual)}")

    ### Step 2: Open pixel area once upfront

    # --- Open pixel area Zarr once ---
    fs = fsspec.filesystem("s3", anon=False)
    pixel_area_da = xr.open_zarr(fs.get_mapper(pixel_area_zarr_path), chunks={"y": 10000, "x": 10000})["band_data"]

    # --- Get lat bounds for overlap with model ---
    model_lat_min = -90  # or read from model Zarr if needed
    model_lat_max = 90
    pixel_lat_min = pixel_area_da["y"].min().item()
    pixel_lat_max = pixel_area_da["y"].max().item()

    lat_min = max(model_lat_min, pixel_lat_min)
    lat_max = min(model_lat_max, pixel_lat_max)

    # Round to avoid floating point mismatch
    lat_min = np.round(lat_min, 6)
    lat_max = np.round(lat_max, 6)

    # --- Slice & persist ---
    pixel_area_overlap = pixel_area_da.sel(y=slice(lat_max, lat_min))
    pixel_area_overlap = pixel_area_overlap.persist()

    ### Step 2: Create global 0.04x0.04 deg geotifs (one for each dataset-year).
    ### Each dataset-year is processed sequentially but using Dask to parallelize each one

    for var_name in vars_to_process:

        main_logger.info(f"Starting {var_name}: {uu.timestr()}")
        var_start_time = time.time()

        for year_idx in range(years_to_process):

                map_stats = global_map_for_variable_year(var_name, year_idx, rechunked_mega_zarr_path, pixel_area_overlap, output_base, no_upload, main_logger)
                main_logger.info(map_stats)

        var_end_time = time.time()
        main_logger.info(f"  Processed {var_name} in {round(var_end_time - var_start_time)} seconds: {uu.timestr()}")

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


    ### Step 3: Aggregates logs

    # Sets it so that no worker logs are created if doing a local run
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with worker log compilation", main_logger)

        # Adds the workers' logs to the main log and uploads to s3
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)

    # Closes the Dask client if not running locally
    if not run_local:
        client.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a global rechunked mega-zarr.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--input_date', help='Date of run, in YYYYMMDD')
    parser.add_argument('-fv', '--first_variables_to_process', type=int, help='Number of variables to process from raw mega-zarr (for testing)')
    parser.add_argument('-fy', '--first_years_to_process', type=int, help='Number of years to process from raw mega-zarr (for testing)')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    input_date = args.input_date
    first_variables_to_process = args.first_variables_to_process
    first_years_to_process = args.first_years_to_process
    log_note = args.log_note

    run_local = args.run_local
    no_log = args.no_log
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, input_date, run_local, no_log, no_upload,
         first_variables_to_process=first_variables_to_process, first_years_to_process=first_years_to_process, log_note=log_note)
