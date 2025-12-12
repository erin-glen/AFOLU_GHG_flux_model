import os
import boto3
import fsspec
import time
import numpy as np
import pandas as pd
import sys
from dask.distributed import print
import dask.array as da
import xarray as xr
import zarr
# from zarr.storage import FSStore

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu

# Creates the s3 paths for the raw and rechunked mega-zarrs
def create_mega_zarr_path(zarr_basic_path, chunk_size_pixels, interval_type, model_type, run_date, main_logger):

    # Sets the output zarr location based on the model run
    mega_zarr_path = zarr_basic_path.replace(cn.model_type_placeholder, model_type)
    mega_zarr_path = mega_zarr_path.replace("MODEL_INTERVAL_TYPE", interval_type)
    mega_zarr_path = mega_zarr_path.replace("RUN_DATE", run_date)
    mega_zarr_path = mega_zarr_path.replace("CHUNK_SIZE", str(chunk_size_pixels))

    main_logger.info(f"Zarr path created: {mega_zarr_path}")

    return mega_zarr_path


# Gets the row and column indexes in a global grid for a given lat and long using a given resolution
def latlon_to_global_zarr_indices(lat, lon, resolution):
    lat_max = 90.0
    lon_min = -180.0

    lat_idx = int(round((lat_max - lat) / resolution))
    lon_idx = int(round((lon - lon_min) / resolution))

    return lat_idx, lon_idx


# Creates a Zarr group with individual datasets on S3 with coordinate arrays (x/y/year),
# spatial_ref metadata, and dataset definitions WITHOUT allocating global arrays.
# That is, it doesn't computer anything upfront or locally. It just creates the zarr group
# with datasets inside.
# In addition to x and y dimensions, there is also a time dimension (intervals), which uses an index (not the actual year).
# This zarr-related code from https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68f984c6-9aa0-8327-a910-5ad9a8d170fc
# and maybe some later chats, too.
def initialize_global_mega_zarr(store_url, dataset_keys, n_years, chunk_size, main_logger, fill_value= np.nan):

    # Computes dimensions
    lat_size = int(180 / cn.resolution)
    lon_size = int(360 / cn.resolution)

    # Creates coordinate arrays globally and for all years
    lats = np.arange(90.0 - cn.resolution / 2, -90, -cn.resolution)[:lat_size]
    lons = np.arange(-180.0 + cn.resolution / 2, 180, cn.resolution)[:lon_size]
    year_index = np.arange(n_years)  # Can't assign the year dimension the true years (2016...2024). Needs to be the year index

    # Spatial reference (CRS metadata)
    spatial_attrs = {
        "grid_mapping_name": "latitude_longitude",
        "epsg_code": 4326,
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
    }

    compressor = {
        "name": "zstd",
        "configuration": {"level": 3}
    }
    print(f"Using zstd compression (level=3): {uu.timestr()}")

    start_time = time.time()

    # For each dataset, uses Dask arrays filled lazily (no memory blowup)
    data_vars = {}
    encoding = {}

    for key in dataset_keys:
        main_logger.info(f"Creating {key} in global mega-zarr: {uu.timestr()}")

        # Rather than pre-creating an output datatype dictionary, I'm taking the hard-coded route
        # and just assigning the output datatype here for each dataset that goes in the zarr
        if "density" in key:
            dtype = 'float32'
        elif "emis" in key:
            dtype = 'float32'
        elif "removals" in key:
            dtype = 'float32'
        elif "net" in key:
            dtype = 'float32'
        elif cn.land_state_pattern in key:
            dtype = 'uint32'
        elif cn.composite_primary_forest in key:
            dtype = 'uint8'
        elif cn.forest_age_output_pattern in key:
            dtype = 'uint16'
        else:
            sys.exit(f"Dataset {key} not assigned a data type for addition to global zarr")

        dask_data = da.full(
            (n_years, lat_size, lon_size),
            fill_value,
            dtype=dtype,
            chunks=chunk_size
        )

        data_vars[key] = xr.DataArray(
            dask_data,
            dims=("year", "y", "x"),
            coords={"year": year_index, "y": lats, "x": lons},
            name=key,
            attrs={"grid_mapping": "spatial_ref"},
        )

        # Define encoding (compression, dtype, and chunks)
        encoding[key] = {
            "compressors": compressor,
         }

    # Constructs dataset
    main_logger.info(f"Constructing megazarr dataset with metadata only: {uu.timestr()}")
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "x": lons,
            "y": lats,
            "year": year_index
        },
    )

    ds["spatial_ref"] = xr.DataArray(
        np.array(0, dtype="int32"),
        attrs=spatial_attrs,
    )

    main_logger.info(f"dataset info: {ds}: {uu.timestr()}")

    # Writes only metadata to s3 (lazy), not values
    main_logger.info(f"Writing metadata for mega-zarr: {uu.timestr()}")
    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(store_url)
    ds.to_zarr(
        store=mapper,
        mode="w",
        compute=False,
        encoding=encoding,
        zarr_format=3
    )

    main_logger.info(f"Created metadata for mega-zarr: {uu.timestr()}")

    z = zarr.open_group(mapper, mode="r")
    main_logger.info(f"Mega-zarr group info: {z.info}: {uu.timestr()}")

    # Clean _FillValue in populated zarr
    # Need to remove _FillValue attribute in zarr because it's being encoded in some way that is incompatible with xarray while using zarr v3,
    # per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68f984c6-9aa0-8327-a910-5ad9a8d170fc.
    # There doesn't seem to be a way to create the zarr with a correctly encoded _FillValue in the first place,
    # hence this fix after the fact.
    main_logger.info(f"Cleaning zarr _FillValue from each dataset: {uu.timestr()}")

    # Open Zarr group in read/write mode
    z = zarr.open_group(store=mapper, mode="r+")

    # Loop through all arrays
    for key in z.array_keys():
        arr = z[key]
        if "_FillValue" in arr.attrs:
            main_logger.info(f"Removing _FillValue from {key}: {uu.timestr()}")
            del arr.attrs["_FillValue"]

    main_logger.info(f"Cleaned _FillValue from Zarr metadata: {uu.timestr()}")

    end_time = time.time()
    main_logger.info(f"Initialized spatial mega-zarr metadata at {store_url} in {round(end_time-start_time)} seconds: {uu.timestr()}")


# Populates pre-existing global mega-zarr with select output numpy arrays (out_dict_all_dtypes)
def populate_zarr(bounds, bounds_str, create_zarr, interval_end_years, is_large_run, logger_worker, mega_zarr_path,
                  out_dict_all_dtypes, outputs_to_zarr, process, stage, tile_id):

    if create_zarr:

        lu.print_and_log(f"Writing select outputs to global zarr for {bounds_str} in {tile_id}: {uu.timestr()}", is_large_run, logger_worker)
        uu.rename_s3_task_file(stage, bounds, "zarr_population_", is_large_run, logger_worker)
        zarr_start = time.time()

        # Opens pre-created global mega-zarr
        fs = fsspec.filesystem("s3", anon=False)
        mapper = fs.get_mapper(mega_zarr_path)
        z = zarr.open(mapper, mode="r+")

        lu.print_and_log(f"Available datasets in global mega-zarr: {list(z.array_keys())}: {uu.timestr()}", False, logger_worker)

        # Iterates through each output that we want to include in the zarr and each interval to add it
        for output_to_zarr_pattern in outputs_to_zarr:
            for i, year in enumerate(interval_end_years):

                # lu.print_and_log(f"Writing {output_to_zarr_pattern} for {year} for {bounds_str} to zarr: {uu.timestr()}",
                #     is_large_run, logger_worker)

                # Converts bounding box corners to row and column indices
                lat_start, lon_start = latlon_to_global_zarr_indices(bounds[3], bounds[0], cn.resolution)  # north, west
                lat_end, lon_end = latlon_to_global_zarr_indices(bounds[1], bounds[2], cn.resolution)  # south, east

                pattern_with_units = add_units_year_to_pattern(output_to_zarr_pattern, year)

                # Selects the relevant output numpy array for insertion into zarr.
                # Only inserts into zarr if that data is in the output dictionary.
                # That way, it won't try to insert summative outputs in the global zarr when the summative outputs aren't in the output dictionary.
                if pattern_with_units in out_dict_all_dtypes:
                    data = out_dict_all_dtypes[pattern_with_units]

                    # Writes numpy array to global zarr
                    z[output_to_zarr_pattern][
                        i,                  # year index (not the actual year)
                        lat_start:lat_end,  # rows (Y)
                        lon_start:lon_end,  # columns (X)
                    ] = data

                else:
                    lu.print_and_log(f"Skipping missing key {pattern_with_units} for inclusion in zarr: {uu.timestr()}", is_large_run, logger_worker)


        # # Checks min, mean and max values for chunk in the zarr for comparison with chunk stats spreadsheet
        # # that directly uses original numpy arrays.
        # # For QC only.
        # for output_to_zarr_pattern in outputs_to_zarr:
        #     for year_idx, year in enumerate(interval_end_years):
        #
        #         target_box = {
        #             "lat_min": bounds[1],
        #             "lat_max": bounds[3],
        #             "lon_min": bounds[0],
        #             "lon_max": bounds[2]
        #         }
        #
        #         if "density__AGC" in output_to_zarr_pattern:  # Just calculates and prints for AGC density for QC purposes
        #             check_region_stats(mega_zarr_path, output_to_zarr_pattern, year_idx, target_box, logger_worker)
        #
        zarr_end = time.time()
        # lu.print_and_log(f"Memory usage after writing to zarr completed for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB",False, logger_worker)
        lu.print_and_log(f"Wrote outputs to global zarrs for {bounds_str} in {tile_id} in {round(zarr_end - zarr_start)} seconds: {uu.timestr()}",False, logger_worker)

    else:
        lu.print_and_log(f"Not writing outputs for {bounds_str} in {tile_id} to global zarrs: {uu.timestr()}",False, logger_worker)


# Checks the stats for a bounding box in a zarr for a given dataset and year
def check_region_stats(store_url, dataset_key, year_idx, target_box, logger_worker=None, main_logger=None):

    """Check min/max of the region written."""
    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(store_url)
    z = zarr.open(mapper, mode="r")

    lat0, lon0 = latlon_to_global_zarr_indices(target_box["lat_max"], target_box["lon_min"], cn.resolution)
    lat1, lon1 = latlon_to_global_zarr_indices(target_box["lat_min"], target_box["lon_max"], cn.resolution)

    region_array = z[dataset_key][year_idx, lat0:lat1, lon0:lon1]

    # Non-zero pixels in the array
    non_zero_count = np.count_nonzero(region_array)

    statement = f"      {dataset_key} year {year_idx}: min={region_array.min()}, mean={region_array.mean()}, max={region_array.max()}, non-zero cells={non_zero_count}"

    # This can be accessed either by the main function or a worker, so it is designed to print to the log from either
    if logger_worker:
        lu.print_and_log(statement, False, logger_worker)
    if main_logger:
        main_logger.info(statement)


# Calculates regular chunk stats in 1x1 deg chunk of dataset-year slice of zarr.
# Chunk stats are calculated using the same function as used on numpy array outputs from models.
def zarr_1x1_deg_stats(bounds, var_name, zarr_path):

    bounds_str = uu.boundstr(bounds)  # String form of chunk bounds, from e.g., [8, -1, 9, 0] to 8_-1_9_0
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W

    zarr_stats_raw_all_years = []

    # print(f"Getting stats for {var_name} for year {year_idx} for {bounds_str}: {uu.timestr()}")
    start_time = time.time()

    # Bounding box to get stats for, reformatted for zarr extraction
    target_box = {
        "lat_min": bounds[1],
        "lat_max": bounds[3],
        "lon_min": bounds[0],
        "lon_max": bounds[2]
    }

    # print(f"Getting indices for {bounds_str}")
    lat0, lon0 = latlon_to_global_zarr_indices(target_box["lat_max"], target_box["lon_min"], cn.resolution)
    lat1, lon1 = latlon_to_global_zarr_indices(target_box["lat_min"], target_box["lon_max"], cn.resolution)

    fs = fsspec.filesystem("s3", anon=False)

    # Calculates chunk stats on the chunk of the zarr.
    # Rather than encoding rows as input or output layer, they are encoded by whether they are raw or rechunked zarr
    # since all of these are outputs.
    # Chunk stats are dictionaries.
    # print(f"Getting mapper for {bounds_str}")
    zarr_mapper = fs.get_mapper(zarr_path)
    # print(f"Opening zarr for {bounds_str}")
    zarr_group = zarr.open(zarr_mapper, mode="r")
    # print(f"Getting array for {bounds_str}")
    zarr_chunk_array = zarr_group[var_name][:, lat0:lat1, lon0:lon1]

    for year_idx, year in enumerate(cn.interval_end_years_annual):

        zarr_chunk_array_year = zarr_chunk_array[year_idx]

        # The dataset pattern being analyzed, with year and units added
        pattern_with_units = add_units_year_to_pattern(var_name, year)

        # print(f"Calculating stats for {bounds_str}")
        zarr_stats_raw_year = uu.calculate_stats(zarr_chunk_array_year, pattern_with_units, bounds_str, tile_id, 'zarr_stats')
        # print(zarr_stats_raw_year)

        zarr_stats_raw_all_years.append(zarr_stats_raw_year)

    # end_time = time.time()
    # print(f"  Calculated stats for {pattern_with_units} for {year} for {bounds} in {round(end_time - start_time)} seconds: {uu.timestr()}")

    # print(f"zarr_stats_raw for {bounds_str}: {zarr_stats_raw}")

    # Returns the chunk stats from the zarr as a list of dictionaries, with each element being one chunk
    return zarr_stats_raw_all_years


# Parallelizes stats calculation in 1x1 deg chunks in raw and rechunked zarrs for a given dataset-year
def run_parallel_stats(client, chunk_list, var, zarr_path):

    futures = []

    # Iterates through all chunks in the list for a given dataset-year
    for chunk in chunk_list:
        future = client.submit(zarr_1x1_deg_stats,
                               chunk, var, zarr_path, retries=2)
        futures.append(future)

    # List of dictionaries, where each dictionary is stats for a single chunk
    results = client.gather(futures)

    return results


# Compares chunk stats from model and from zarr for a dataset-year combination
# Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6903d1dd-555c-8321-8547-0aa4772c9878
def compare_dataset_year_chunk_stats(all_merged_tables, chunk_stats_variable_zarr, main_logger,
                                     tables_to_compare_dict, var_name, zarr_comparison_stats_path):

    # Selects relevant model output table
    # The formatting of the year depends on the variable.
    if "gross" in var_name:
        model_table = tables_to_compare_dict[cn.gross_outputs_1x1]
        # year = year
    elif "net" in var_name:
        model_table = tables_to_compare_dict[cn.net_outputs_1x1]
        # year = year
    else:
        model_table = tables_to_compare_dict[cn.other_outputs_1x1]
        # For reasons I can't really trace back, the year datatype for C densities is object, not int.
        # So, it needs to be recast to a str or int to match the chunk_stats table.
        # year = str(year)

    # Converts zarr chunk stats from list of dictionaries to dataframe.
    # Need to flatten the list because each chunk for each dataset is a list of dictionaries, where each element is a year.
    # So, flattening the list makes all years for all variables and chunks flat, rather than years being nested in each chunk-dataset.
    chunk_stats_variable_zarr_flat = uu.flatten_list(chunk_stats_variable_zarr)
    # print("chunk_stats_variable_zarr_flat:", chunk_stats_variable_zarr_flat)
    zarr_df = pd.DataFrame(chunk_stats_variable_zarr_flat)
    # print("zarr_df:", zarr_df)

    # Subsets model chunk stats to relevant pattern
    subset_model_table = model_table[(model_table['pattern'].str.contains(var_name, na=False))]
    # print("subset_model_table", subset_model_table)

    # Selects only the needed columns from rechunked_zarr_table
    # main_logger.info(f"    Subsetting zarr table to numeric columns for {var_name}: {uu.timestr()}")
    zarr_subset_table = zarr_df[['chunk_name', 'min_value', 'mean_value', 'max_value', 'count_value']].copy()
    # print("zarr_subset_table", zarr_subset_table)

    # Renames columns in raw_subset to distinguish them after merge
    # main_logger.info(f"    Renaming zarr columns for {var_name}: {uu.timestr()}")
    zarr_subset_table = zarr_subset_table.rename(columns={
        'min_value': 'min_value_zarr',
        'mean_value': 'mean_value_zarr',
        'max_value': 'max_value_zarr',
        'count_value': 'count_value_zarr'
    })
    # print("zarr_subset_table", zarr_subset_table)

    # Converts all zarr value columns to numeric, coercing errors to NaN
    # main_logger.info(f"    Converting zarr columns to numeric for {var_name}: {uu.timestr()}")
    for col in ['min_value_zarr', 'mean_value_zarr', 'max_value_zarr', 'count_value_zarr']:
        zarr_subset_table[col] = pd.to_numeric(zarr_subset_table[col], errors='coerce')

    # Merges with subset_model_table on 'chunk_name', left join (keeps all model output rows)
    # main_logger.info(f"    Merging zarr data to original model data for {var_name}: {uu.timestr()}")
    merged_table = subset_model_table.merge(zarr_subset_table, on='chunk_name', how='left')

    # Calculates differences for four metrics and stores in new columns
    main_logger.info(f"    Calculating differences for {var_name} ({merged_table['count_value_zarr'].sum().item()} pixels in zarr): {uu.timestr()}")
    merged_table['min_value_diff'] = merged_table['min_value'] - merged_table['min_value_zarr']
    merged_table['mean_value_diff'] = merged_table['mean_value'] - merged_table['mean_value_zarr']
    merged_table['max_value_diff'] = merged_table['max_value'] - merged_table['max_value_zarr']
    merged_table['count_value_diff'] = merged_table['count_value'] - merged_table['count_value_zarr']
    # print(merged_table.head())

    # Calculates max absolute difference across the four metrics' difference columns
    merged_table['maximum_diff_value'] = merged_table[
        ['min_value_diff', 'mean_value_diff', 'max_value_diff', 'count_value_diff']
    ].abs().max(axis=1)

    # Identifies rows (chunks) which have stats that differ between model and zarr
    mask = merged_table['maximum_diff_value'] > cn.zarr_difference_tolerance

    # Number of rows from model output without matching zarr pixel counts

    # Excludes rows where model 'count_value' is non-numeric (like 'no data') (no model outputs in those chunks).
    # Coerces to numeric and check for valid values.
    valid_count_mask = pd.to_numeric(merged_table['count_value'], errors='coerce').notna()

    # From those valid rows, counts how many have no zarr stats
    chunks_without_zarr_stats = merged_table[valid_count_mask]['count_value_diff'].isna().sum().item()
    main_logger.info(f"    Rows with data without pixel count comparison: {chunks_without_zarr_stats}")

    # Applies the mask to filter those rows
    differences_exceeding_tolerance = merged_table[mask]

    # Prints rows that exceed the tolerance for difference between original and zarr chunk stats
    if len(differences_exceeding_tolerance) > 0:
        main_logger.warning(f"    WARNING: There are {len(differences_exceeding_tolerance)} rows in {var_name} that have differences exceeding the tolerance!")

        # Selects chunk_id and all difference to print in the console for easy viewing
        cols_to_print = [
            'chunk_id',
            'min_value_diff',
            'mean_value_diff',
            'max_value_diff',
            'count_value_diff',
            'maximum_diff_value'
        ]

        main_logger.warning(differences_exceeding_tolerance[cols_to_print])

    else:
        main_logger.info(f"    No rows in {var_name} have metrics with differences exceeding the tolerance.")

    # Adds df for this dataset-year combination to the list of all the dataset-year dfs
    all_merged_tables.append(merged_table)

    # Writes cumulative results (all dataset-year combinations) to Excel file or parquet tables

    # Concatenates all merged dataset-year tables into a single DataFrame
    final_merged_table = pd.concat(all_merged_tables, ignore_index=True)

    # Splits output rows based on 'layer_name' containing 'flux', 'gross', or 'net'
    gross_flux_1x1_outputs = final_merged_table[final_merged_table['layer_name'].str.contains('gross', case=False, na=False)]
    net_flux_1x1_outputs = final_merged_table[final_merged_table['layer_name'].str.contains('net|flux', case=False, na=False)]

    # Puts output rows that don't contain 'flux|gross|net' in a separate table
    other_1x1_outputs = final_merged_table[~final_merged_table['layer_name'].str.contains('flux|gross|net', case=False, na=False)]

    # Saves output to three tabs in Excel
    if "xlsx" in zarr_comparison_stats_path:
        # Writes to Excel after each iteration of dataset-year to check results more easily (not have to wait until end)
        with pd.ExcelWriter(zarr_comparison_stats_path, engine='openpyxl', mode='w') as writer:

            gross_flux_1x1_outputs.to_excel(writer, sheet_name=cn.gross_outputs_1x1, index=False)
            net_flux_1x1_outputs.to_excel(writer, sheet_name=cn.net_outputs_1x1, index=False)
            other_1x1_outputs.to_excel(writer, sheet_name=cn.other_outputs_1x1, index=False)

    # Saves output to three parquet tables.
    # These must be written in the same order as the file names are created in zu.get_table_names_for_zarr_stats_comparison()
    elif "parquet" in zarr_comparison_stats_path[0]:
        gross_flux_1x1_outputs.to_parquet(zarr_comparison_stats_path[0], index=False)
        net_flux_1x1_outputs.to_parquet(zarr_comparison_stats_path[1], index=False)
        other_1x1_outputs.to_parquet(zarr_comparison_stats_path[2], index=False)

    else:
        sys.exit("Table type not found")


    # Need to return the combined table so that it can be added to in the next iteration
    return all_merged_tables, len(differences_exceeding_tolerance), chunks_without_zarr_stats


# Gets the names of the gross, other, and net chunk stats tables that should be compared against
# the zarr, as well as the name of the output comparison tables.
# Works for both Excel and Parquet model chunk stats.
# If the model being compared against output Parquet chunk stats, this will return
# Parquet table names for the comparison chunk stats. 
def get_table_names_for_zarr_stats_comparison(comparison_insert, main_logger, model_chunk_stats_path):

    # Separate logic for naming chunk stat comparison outputs if using Excel or Parquet (very large runs)
    if "xlsx" in model_chunk_stats_path:
        main_logger.info(f"Reading model chunk stats from local file: {model_chunk_stats_path}")
        chunk_stats_model_gross = pd.read_excel(model_chunk_stats_path, sheet_name=cn.gross_outputs_1x1)
        chunk_stats_model_other = pd.read_excel(model_chunk_stats_path, sheet_name=cn.other_outputs_1x1)
        chunk_stats_model_net = pd.read_excel(model_chunk_stats_path, sheet_name=cn.net_outputs_1x1)
        chunk_stats_model_1x1_in_10x10 = pd.read_excel(model_chunk_stats_path, sheet_name=cn.counts_1x1_in_10x10)

        # Name of output Excel spreadsheet with chunk stats comparisons
        name, ext = os.path.splitext(model_chunk_stats_path)
        zarr_comparison_stats_path = f"{name}{comparison_insert}_{uu.timestr()}{ext}"
        zarr_comparison_stats_name = os.path.basename(zarr_comparison_stats_path)
        # print(zarr_comparison_stats_path)
        # print(zarr_comparison_stats_name)

    elif "parquet" in model_chunk_stats_path:
        main_logger.info(f"Reading parquet tables from local files: {model_chunk_stats_path}")
        chunk_stats_model_gross = pd.read_parquet(f"{model_chunk_stats_path}__{cn.gross_outputs_1x1}.parquet")
        chunk_stats_model_other = pd.read_parquet(f"{model_chunk_stats_path}__{cn.other_outputs_1x1}.parquet")
        chunk_stats_model_net = pd.read_parquet(f"{model_chunk_stats_path}__{cn.net_outputs_1x1}.parquet")
        chunk_stats_model_1x1_in_10x10 = pd.read_parquet(f"{model_chunk_stats_path}__{cn.counts_1x1_in_10x10}.parquet")

        # Names of output parquet tables with chunk stats comparisons
        zarr_comparison_stats_gross_name = f"{model_chunk_stats_path}__{cn.gross_outputs_1x1}_{comparison_insert}_{uu.timestr()}.parquet"
        zarr_comparison_stats_other_name = f"{model_chunk_stats_path}__{cn.other_outputs_1x1}_{comparison_insert}_{uu.timestr()}.parquet"
        zarr_comparison_stats_net_name = f"{model_chunk_stats_path}__{cn.net_outputs_1x1}_{comparison_insert}_{uu.timestr()}.parquet"
        zarr_comparison_stats_1x1_in_10x10_name = f"{model_chunk_stats_path}__{cn.counts_1x1_in_10x10}_{comparison_insert}_{uu.timestr()}.parquet"
        zarr_comparison_stats_path = [zarr_comparison_stats_gross_name, zarr_comparison_stats_other_name,
                                      zarr_comparison_stats_net_name, zarr_comparison_stats_1x1_in_10x10_name]
        zarr_comparison_stats_name = [os.path.basename(stats_path) for stats_path in zarr_comparison_stats_path]
        # print(zarr_comparison_stats_path)
        # print(zarr_comparison_stats_name)

    else:
        sys.exit("Table type not found")
    # The model chunk stat tables
    tables_to_compare_dict = {cn.gross_outputs_1x1: chunk_stats_model_gross,
                              cn.net_outputs_1x1: chunk_stats_model_net,
                              cn.other_outputs_1x1: chunk_stats_model_other,
                              cn.counts_1x1_in_10x10: chunk_stats_model_1x1_in_10x10}
    return tables_to_compare_dict, zarr_comparison_stats_name, zarr_comparison_stats_path


# Adds units and year specifications to core pattern
def add_units_year_to_pattern(core_pattern, year):
    if "density" in core_pattern:
        pattern_with_units = f"{core_pattern}_ha_{year}"
    elif "emis" in core_pattern:
        pattern_with_units = f"{core_pattern}_ha_yr_{year}"
    elif "removals" in core_pattern:
        pattern_with_units = f"{core_pattern}_ha_yr_{year}"
    elif "net" in core_pattern:
        pattern_with_units = f"{core_pattern}_ha_yr_{year}"
    elif cn.land_state_pattern in core_pattern:
        pattern_with_units = f"{core_pattern}_{year}"
    elif cn.composite_primary_forest in core_pattern:
        pattern_with_units = f"{core_pattern}_{year}"
    elif cn.forest_age_output_pattern in core_pattern:
        pattern_with_units = f"{core_pattern}_{year}"
    else:
        sys.exit(f"Dataset {core_pattern} not assigned a pattern with units for addition to global zarr")

    return pattern_with_units


def upload_zarr_chunk_stat_comparisons(chunks_count_exceeding_total, chunks_without_zarr_stats_total,
                                       main_logger, model_chunk_stats_table_name, stage,
                                       start_time, zarr_comparison_stats_name, zarr_comparison_stats_path):

    if chunks_count_exceeding_total > 0:
        main_logger.warning(f"WARNING: {chunks_count_exceeding_total} chunks exceeded difference tolerance! Check log!")
    else:
        main_logger.info(f"{chunks_count_exceeding_total} chunks exceeded the difference tolerance for one or more chunk stat metrics.")
    uu.stage_duration(start_time, uu.timestr(), f"{stage} with zarr chunk stat comparison", main_logger)

    if chunks_without_zarr_stats_total > 0:
        main_logger.warning(f"WARNING: {chunks_without_zarr_stats_total} chunks are missing corresponding zarr chunk stats! Check log!")
    else:
        main_logger.info(f"{chunks_without_zarr_stats_total} chunks were missing corresponding zarr chunk stats.")
    uu.stage_duration(start_time, uu.timestr(), f"{stage} with zarr chunk stat comparison", main_logger)

    s3_client = boto3.client("s3")

    if "xlsx" in zarr_comparison_stats_path:
        main_logger.info(f"Uploading chunk stats comparison Excel spreadsheet to s3: {uu.timestr()}")
        try:
            s3_client.upload_file(zarr_comparison_stats_path, cn.short_bucket_prefix,
                                  Key=f"{cn.s3_chunk_stats_path}{zarr_comparison_stats_name}")
            main_logger.info(
                f"Chunk stats spreadsheet uploaded to {cn.full_bucket_prefix}/{cn.s3_chunk_stats_path}{zarr_comparison_stats_name}: {uu.timestr()}")
        except Exception as e:
            main_logger.warning(f"Chunk stats upload to s3 failed: {e}. Continuing without halting.")

    elif "parquet" in zarr_comparison_stats_path[0]:  # Because this is a list, so just use the first one to get the pattern
        main_logger.info(f"Uploading chunk stats comparison parquet tables to s3: {uu.timestr()}")

        for parquet_name, parquet_path in zip(zarr_comparison_stats_name, zarr_comparison_stats_path):
            parquet_folder = parquet_path.split('/')[1]   # parquet_YYYYMMDD_HH_MM_SS
            s3_key = f"{cn.s3_chunk_stats_path}{parquet_folder}/{parquet_name}"
            # print(cn.s3_chunk_stats_path)
            # print(parquet_folder)
            # print(parquet_name)
            # print(s3_key)
            main_logger.info(f"Uploading {parquet_path} to {s3_key}: {uu.timestr()}")
            s3_client.upload_file(parquet_path, cn.short_bucket_prefix, Key=s3_key)

    else:
        sys.exit("Table type not found")

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats", main_logger)