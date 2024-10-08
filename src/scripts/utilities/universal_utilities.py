import os
import coiled
import dask
import boto3
import time
import math
import numpy as np
import pandas as pd
import pytz
import rasterio
import rasterio.transform
import rasterio.windows
import subprocess
import re
import requests
import concurrent.futures
from botocore.config import Config
from dask.distributed import print
from dask.distributed import Client
from datetime import datetime
from io import BytesIO
from osgeo import gdal

# Project imports
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import log_utilities as lu

# Time in Eastern US timezone as a string
def timestr():
    # Define the Eastern Time timezone
    eastern = pytz.timezone('US/Eastern')
    # Get the current time in UTC and convert to Eastern Time
    eastern_time = datetime.now(eastern)
    # Format the time as a string
    return eastern_time.strftime("%Y%m%d_%H_%M_%S")


# Connects to a Coiled cluster of a specified name if the local flag isn't on
def connect_to_Coiled_cluster(cluster_name, run_local):
    if run_local:
        print("Running locally with a local Dask client.")
        cluster = None
        client = Client()
        return cluster, client
    else:
        cluster = coiled.Cluster(
            name=cluster_name,
            account='wri-forest-research',  # Specify the workspace here
            # Include other cluster configurations as needed
        )
        client = cluster.get_client()
        return cluster, client

# Chunk bounds as a string
def boundstr(bounds):
    bounds_str = "_".join([str(round(x)) for x in bounds])
    return bounds_str


# Chunk length in pixels
def calc_chunk_length_pixels(bounds):
    chunk_length_pixels = int((bounds[3] - bounds[1]) * (40000 / 10))
    return chunk_length_pixels


# Maps GDAL data type to the appropriate string value
gdal_dtype_mapping = {
    gdal.GDT_Byte: 'Byte',
    gdal.GDT_UInt16: 'UInt16',
    gdal.GDT_Int16: 'Int16',
    gdal.GDT_UInt32: 'UInt32',
    gdal.GDT_Int32: 'Int32',
    gdal.GDT_Float32: 'Float32',
    gdal.GDT_Float64: 'Float64'
}

# Maps GDAL datatypes to numpy datatypes
def map_to_numpy_dtype(data_type):
    dtype_map = {
        'Float32': 'float32',
        'Float64': 'float64',
        'Byte': 'uint8',
        'UInt16': 'uint16',
        'Int16': 'int16',
        'UInt32': 'uint32',
        'Int32': 'int32',
        # Add more mappings as needed
    }
    return dtype_map.get(data_type, 'float32')  # Defaults to 'float32' if argument not found


# Gets the W, S, E, N bounds of a 10x10 degree tile
def get_10x10_tile_bounds(tile_id):
    if "S" in tile_id:
        max_y = -1 * (int(tile_id[:2]))
        min_y = -1 * (int(tile_id[:2]) + 10)
    else:
        max_y = (int(tile_id[:2]))
        min_y = (int(tile_id[:2]) - 10)

    if "W" in tile_id:
        max_x = -1 * (int(tile_id[4:7]) - 10)
        min_x = -1 * (int(tile_id[4:7]))
    else:
        max_x = (int(tile_id[4:7]) + 10)
        min_x = (int(tile_id[4:7]))

    return min_x, min_y, max_x, max_y  # W, S, E, N


# Returns list of all chunk boundaries within a bounding box for chunks of a given size
def get_chunk_bounds(bounding_box, chunk_size):
    min_x = bounding_box[0]
    min_y = bounding_box[1]
    max_x = bounding_box[2]
    max_y = bounding_box[3]

    x, y = (min_x, min_y)
    chunks = []

    # Polygon Size
    while y < max_y:
        while x < max_x:
            bounds = [
                x,
                y,
                x + chunk_size,
                y + chunk_size,
            ]
            chunks.append(bounds)
            x += chunk_size
        x = min_x
        y += chunk_size

    return chunks


# Returns the encompassing tile_id string in the form YYN/S_XXXE/W based on a coordinate
def xy_to_tile_id(top_left_x, top_left_y):
    lat_ceil = math.ceil(top_left_y / 10.0) * 10
    lng_floor = math.floor(top_left_x / 10.0) * 10

    lng = f"{str(abs(lng_floor)).zfill(3)}{'E' if lng_floor >= 0 else 'W'}"
    lat = f"{str(abs(lat_ceil)).zfill(2)}{'N' if lat_ceil >= 0 else 'S'}"

    return f"{lat}_{lng}"


# Calculates the elapsed time for a stage
def stage_duration(start_time_str, end_time_str, stage):

    start_time = datetime.strptime(start_time_str, "%Y%m%d_%H_%M_%S")
    end_time = datetime.strptime(end_time_str, "%Y%m%d_%H_%M_%S")

    print(f"Elapsed time for {stage}: {end_time - start_time}")


# Lazily opens tile within provided bounds (i.e., one chunk) and returns as a numpy array.
def get_tile_dataset_rio(uri, data_type, bounds, chunk_length_pixels, is_final, logger):

    try:
        with rasterio.open(uri) as ds:
            window = rasterio.windows.from_bounds(*bounds, ds.transform)
            data = ds.read(1, window=window)
    except Exception as e:
        numpy_dtype = map_to_numpy_dtype(data_type)
        data = np.full((chunk_length_pixels, chunk_length_pixels), 0).astype(numpy_dtype)
        lu.print_and_log(f"flm: Error accessing the dataset. Returning array of all 0s: {e}", is_final, logger)

    return data


# Prepares list of chunks to download.
def prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_final, logger):

    futures = {}

    bounds_str = boundstr(bounds)
    tile_id = xy_to_tile_id(bounds[0], bounds[3])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        lu.print_and_log(f"Requesting data in chunk {bounds_str} in {tile_id}: {timestr()}", is_final, logger)

        for key, value in updated_download_dict.items():
            futures[executor.submit(get_tile_dataset_rio, value[0], value[1], bounds, chunk_length_pixels, is_final, logger)] = key

    return futures


# Checks if tiles exist at all
def check_for_tile(download_dict, is_final, logger):

    # Configures S3 client with increased retries; retries can max out for global analyses
    s3_config = Config(
        retries={
            'max_attempts': 10,  # Increases the number of retry attempts
            'mode': 'standard'
        }
    )
    s3_client = boto3.client("s3", config=s3_config)

    for value in download_dict.values():
        # Extract the S3 key
        s3_key = value[0][len("s3://gfw2-data/"):]  # Adjust if the bucket name differs

        # Extract the tile_id using the tile_id_pattern
        tile_id_matches = re.findall(cn.tile_id_pattern, value[0])
        if not tile_id_matches:
            logger.warning(f"No tile_id found in the file path: {value[0]}")
            continue  # Skip if no tile_id is found

        tile_id = tile_id_matches[0]

        # Check if the object exists in S3
        try:
            s3_client.head_object(Bucket='gfw2-data', Key=s3_key)
            lu.print_and_log(f"Tile id {tile_id} exists for some inputs. Proceeding: {timestr()}", is_final, logger)
            return True  # If at least one tile exists, return True
        except Exception as e:
            pass  # Continue checking other tiles

    lu.print_and_log(f"Tile id {tile_id} does not exist. Skipped chunk: {timestr()}", is_final, logger)
    return False


# Checks whether a chunk has data in it.
def check_chunk_for_data(required_layers, bounds_str, tile_id, any_or_all, is_final, logger):
    if any_or_all == "any":

        for array in required_layers.values():
            min_val = np.min(array)

            if min_val != None:
                logger.info(f"flm: Data in chunk {bounds_str}. Proceeding: {timestr()}")
                print(f"flm: Data in chunk {bounds_str}. Proceeding: {timestr()}")
                return True

        logger.info(f"flm: No data in chunk {bounds_str} for assessed inputs: {timestr()}")
        print(f"flm: No data in chunk {bounds_str} for assessed inputs: {timestr()}")
        return False

    elif any_or_all == "all":

        for key, array in required_layers.items():
            min_val = np.min(array)
            max_val = np.max(array)

            if min_val == max_val:
                logger.info(f"flm: Chunk {bounds_str} does not exist for {key}. Skipped chunk: {timestr()}")
                print(f"flm: Chunk {bounds_str} does not exist for {key}. Skipped chunk: {timestr()}")
                return False

        logger.info(f"flm: Chunk {bounds_str} has data for all assessed inputs: {timestr()}")
        print(f"flm: Chunk {bounds_str} has data for all assessed inputs: {timestr()}")
        return True

    else:
        raise Exception("any_or_all argument not valid")


# Saves array as a raster locally, then uploads it to s3. NoData value for outputs is optional
def save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id,
                                     bounds_str, output_dict, is_final, logger, no_data_val=None):

    # Configures S3 client with increased retries; retries can max out for global analyses
    s3_config = Config(
        retries={
            'max_attempts': 10,  # Increases the number of retry attempts
            'mode': 'standard'
        }
    )
    s3_client = boto3.client("s3", config=s3_config)  # Uses the configured client with more retries

    transform = rasterio.transform.from_bounds(*bounds, width=chunk_length_pixels, height=chunk_length_pixels)

    file_info = f'{tile_id}__{bounds_str}'

    if is_final:
        lu.print_and_log(f"Saving and uploading outputs for {bounds_str} in {tile_id}: {timestr()}", is_final, logger)

    # For every output file, saves from array to local raster, then to s3.
    # Can't save directly to s3, unfortunately, so need to save locally first.
    for key, value in output_dict.items():

        data_array = value[0]
        data_type = value[1]
        data_meaning = value[2]
        year_out = value[3]

        if is_final:
            file_name = f"{file_info}__{key}.tif"
        else:
            file_name = f"{file_info}__{key}__{timestr()}.tif"

        # Only prints if not a final run
        if not is_final:
            lu.print_and_log(f"Saving {bounds_str} in {tile_id} for {year_out}: {timestr()}", is_final, logger)

        # Includes NoData value in output raster
        if no_data_val is not None:
            with rasterio.open(f"/tmp/{file_name}", 'w', driver='GTiff', width=chunk_length_pixels,
                               height=chunk_length_pixels, count=1,
                               dtype=data_type, crs='EPSG:4326', transform=transform, compress='lzw', blockxsize=400,
                               blockysize=400, nodata=no_data_val) as dst:
                dst.write(data_array, 1)

        # No NoData value in output raster
        else:
            with rasterio.open(f"/tmp/{file_name}", 'w', driver='GTiff', width=chunk_length_pixels,
                               height=chunk_length_pixels, count=1,
                               dtype=data_type, crs='EPSG:4326', transform=transform, compress='lzw', blockxsize=400,
                               blockysize=400) as dst:
                dst.write(data_array, 1)

        s3_path = f"{cn.s3_out_dir}/{data_meaning}/{year_out}/{chunk_length_pixels}_pixels/{time.strftime('%Y%m%d')}"

        # Only prints if not a final run
        if not is_final:
            lu.print_and_log(f"Uploading {bounds_str} in {tile_id} for {year_out} to {s3_path}: {timestr()}", is_final, logger)

        s3_client.upload_file(f"/tmp/{file_name}", "gfw2-data", Key=f"{s3_path}/{file_name}")

        # Deletes the local raster
        os.remove(f"/tmp/{file_name}")


# Lists rasters in an s3 folder and returns their names as a list
def list_rasters_in_folder(full_in_folder):

    cmd = ['aws', 's3', 'ls', full_in_folder]
    s3_contents_bytes = subprocess.check_output(cmd)

    # Converts subprocess results to useful string
    s3_contents_str = s3_contents_bytes.decode('utf-8')
    s3_contents_list = s3_contents_str.splitlines()
    rasters = [line.split()[-1] for line in s3_contents_list]
    rasters = [i for i in rasters if "tif" in i]

    return rasters


# Uploads a shapefile to s3
def upload_shp(in_folder, shp):

    print(f"flm: Uploading to {in_folder}{shp}: {timestr()}")

    shp_pattern = shp[:-4]

    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call
    s3_client.upload_file(f"/tmp/{shp}", "gfw2-data", Key=f"{in_folder[15:]}{shp}")
    s3_client.upload_file(f"/tmp/{shp_pattern}.dbf", "gfw2-data", Key=f"{in_folder[15:]}{shp_pattern}.dbf")
    s3_client.upload_file(f"/tmp/{shp_pattern}.prj", "gfw2-data", Key=f"{in_folder[15:]}{shp_pattern}.prj")
    s3_client.upload_file(f"/tmp/{shp_pattern}.shx", "gfw2-data", Key=f"{in_folder[15:]}{shp_pattern}.shx")

    os.remove(f"/tmp/{shp}")
    os.remove(f"/tmp/{shp_pattern}.dbf")
    os.remove(f"/tmp/{shp_pattern}.prj")
    os.remove(f"/tmp/{shp_pattern}.shx")

    print(f"flm: Uploaded to {in_folder}{shp}: {timestr()}")


# Makes a shapefile of the footprints of rasters in a folder, for checking geographical completeness of rasters
def make_tile_footprint_shp(input_dict):

    in_folder = list(input_dict.keys())[0]
    pattern = list(input_dict.values())[0]

    # Task properties
    print(f"flm: Making tile index shapefile for: {in_folder}: {timestr()}")

    # Folder including s3 key
    s3_in_folder = in_folder
    vsis3_in_folder = f'/vsis3/{in_folder[5:]}'  # [5] drops the s3:// at the front

    # List of all the filenames in the folder
    filenames = list_rasters_in_folder(s3_in_folder)

    # List of the tile paths in the folder
    tile_paths = [vsis3_in_folder + filename for filename in filenames]

    file_paths_txt = f's3_paths_{pattern}.txt'

    with open(f"/tmp/{file_paths_txt}", 'w') as file:
        for item in tile_paths:
            file.write(item + '\n')

    # Output shapefile name
    shp = f"raster_footprints_{pattern}.shp"

    cmd = ["gdaltindex", "-t_srs", "EPSG:4326", f"/tmp/{shp}", "--optfile", f"/tmp/{file_paths_txt}"]
    subprocess.check_call(cmd)

    # Uploads shapefile to s3
    upload_shp(s3_in_folder, shp)

    os.remove(f"/tmp/{file_paths_txt}")

    return f"Completed: {timestr()}"


# Saves an xarray data array locally as a raster and then uploads it to s3
def save_and_upload_raster_10x10(**kwargs):

    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call

    data_array = kwargs['data']  # The data being saved
    out_file_name = kwargs['out_file_name']  # The output file name
    out_folder = kwargs['out_folder']  # The output folder

    print(f"flm: Saving {out_file_name} locally")

    profile_kwargs = {'compress': 'lzw'}  # Adds attribute to compress the output raster
    data_array.rio.to_raster(f"/tmp/{out_file_name}", **profile_kwargs)

    print(f"flm: Saving {out_file_name} to {out_folder[10:]}{out_file_name}")

    s3_client.upload_file(f"/tmp/{out_file_name}", "gfw2-data", Key=f"{out_folder[10:]}{out_file_name}")

    # Deletes the local raster
    os.remove(f"/tmp/{out_file_name}")


# Flattens a nested list
def flatten_list(nested_list):
    return [x for xs in nested_list for x in xs]


# Calculates stats for a chunk (numpy array)
def calculate_stats(array, name, bounds_str, tile_id, in_out):
    if array is None or not np.any(array):  # Check if the array is None or empty
        return {
            'chunk_id': bounds_str,
            'tile_id': tile_id,
            'layer_name': name,
            'in_out': in_out,
            'min_value': 'no data',
            'mean_value': 'no data',
            'max_value': 'no data',
            'data_type': 'no data'
        }
    else:    # Only calculates stats if there is data in the array
        return {
            'chunk_id': bounds_str,
            'tile_id': tile_id,
            'layer_name': name,
            'in_out': in_out,
            'min_value': np.min(array),
            'mean_value': np.mean(array),
            'max_value': np.max(array),
            'data_type': array.dtype.name
        }


# Calculates chunk-level stats for all inputs and outputs and saves to Excel spreadsheet
def calculate_chunk_stats(all_stats, stage):

    if not all_stats:
        print("No statistics to calculate; all_stats is empty.")
        return

    print("Calculating tile stats...")

    # Convert accumulated statistics to a DataFrame
    df_all_stats = pd.DataFrame(all_stats)

    # Convert problematic non-numeric values to NaN
    df_all_stats['min_value'] = pd.to_numeric(df_all_stats['min_value'], errors='coerce')
    df_all_stats['max_value'] = pd.to_numeric(df_all_stats['max_value'], errors='coerce')

    # Sort the DataFrame by 'in_out' and 'layer_name'
    sorted_stats = df_all_stats.sort_values(by=['in_out', 'layer_name']).reset_index(drop=True)

    # Calculate the min and max values for each layer_name
    min_max_stats = df_all_stats.groupby('layer_name').agg(
        min_value=('min_value', 'min'),
        max_value=('max_value', 'max')
    ).reset_index()

    # Creates a dictionary to store separate DataFrames for each 'in_out' value
    in_out_tables = {in_out_value: sorted_stats[sorted_stats['in_out'] == in_out_value]
                     for in_out_value in sorted_stats['in_out'].unique()}

    # Write the combined statistics to a single Excel file
    try:
        with pd.ExcelWriter(f'{cn.chunk_stats_path}{stage}_chunk_statistics_{timestr()}.xlsx') as writer:

            # Writes each 'in_out' DataFrame to its own sheet
            for in_out_value, table in in_out_tables.items():
                sheet_name = f"chunk_stats_{str(in_out_value)}"
                table.to_excel(writer, sheet_name=sheet_name, index=False)

            # Write the min and max statistics to the second sheet
            min_max_stats.to_excel(writer, sheet_name='min_max_for_layers', index=False)

        print(sorted_stats.head())  # Show first few rows of the stats DataFrame for inspection

    except Exception as e:
        print(f"Can't print chunk stats: {e}")


# Gets the name of the first file in a dictionary of dataset names and folders in s3.
def first_file_name_in_s3_folder(download_dict):
    s3_client = boto3.client("s3")
    first_tiles = {}
    sample_tile_id = '00N_110E'  # Use a valid sample tile ID

    for key, folder_path in download_dict.items():
        # Replace {tile_id} with sample_tile_id
        folder_path = folder_path.replace('{tile_id}', sample_tile_id)
        # Splits the path to get the directory part
        dir_path = os.path.dirname(folder_path)
        # Drops the s3:// prefix
        dir_path = dir_path[len('s3://'):]
        # Extract bucket and prefix
        bucket, prefix = dir_path.split('/', 1)
        prefix += '/'  # Ensure prefix ends with '/'
        # List objects in the specified bucket and prefix
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' in response and len(response['Contents']) > 0:
            first_file_key = response['Contents'][0]['Key']
            first_tiles[key] = f"s3://{bucket}/{first_file_key}"
        else:
            first_tiles[key] = None

    return first_tiles


def get_dtype_from_s3(file_path):
    if file_path is None:
        print("File path is None. Cannot determine data type.")
        return None
    vsis3_path = f'/vsis3/{file_path[len("s3://"):]}'
    dataset = gdal.Open(vsis3_path)
    if dataset:
        band = dataset.GetRasterBand(1)
        data_type = gdal.GetDataTypeName(band.DataType)
        return data_type
    else:
        raise ValueError(f"Could not open file {vsis3_path}")


def add_file_type_to_dict(first_tiles):
    download_dict_with_data_types = {}
    for key, file_path in first_tiles.items():
        if file_path is None:
            print(f"No file found for {key}. Skipping...")
            continue
        try:
            dtype = get_dtype_from_s3(file_path)
            if dtype is None:
                print(f"Could not determine data type for {key}. Skipping...")
                continue
            download_dict_with_data_types[key] = [file_path, dtype]
        except Exception as e:
            print(f"Error getting data type for {key}: {e}")
    return download_dict_with_data_types


# Replaces a tile_id in s3 paths in a dictionary with another tile_id
def replace_tile_id_in_dict(data_dict, new_tile_id):
    tile_id_pattern = cn.tile_id_pattern if hasattr(cn, 'tile_id_pattern') else r'\d{2}[NS]_\d{3}[EW]'

    for key, value in data_dict.items():
        file_path = value[0]
        updated_file_path = re.sub(tile_id_pattern, new_tile_id, file_path)
        data_dict[key][0] = updated_file_path

    return data_dict


# Fills any missing chunks (layers) with NoData (0s) of the correct datatype.
def fill_missing_input_layers_with_no_data(layers, uint8_list, int16_list, int32_list, float32_list,
                                           bounds_str, tile_id, is_final, logger):

   # Fills missing layers with arrays of the appropriate data type and size
    for key, array in layers.items():
        if array is None:

            # Determines the appropriate dtype based on the categorized lists
            if key in uint8_list:
                dtype = np.uint8
            elif key in int16_list:
                dtype = np.int16
            elif key in int32_list:
                dtype = np.int32
            elif key in float32_list:
                dtype = np.float32
            else:
                raise ValueError(f"Key {key} for chunk {bounds_str} in {tile_id} not found in any data type lists: {timestr()}")

            # Finds an existing array to use as a template for size
            existing_array = next((arr for arr in layers.values() if arr is not None), None)
            if existing_array is not None:
                # Creates an array of zeros with the same shape and the determined dtype
                layers[key] = np.zeros(existing_array.shape, dtype=dtype)
                # Log the creation of the missing layer
                lu.print_and_log(f"Created {key} for chunk {bounds_str} in {tile_id}: {timestr()}", is_final, logger)
            else:
                # Handles the case where no data exists at all
                raise ValueError(f"No data available to determine the size for the missing layer {key} for chunk {bounds_str} in {tile_id}: {timestr()}")

    return layers


# Creates numpy array of rates or ratios from a tab in an Excel spreadsheet, e.g., removal factors or carbon pool ratios
def convert_lookup_table_to_array(spreadsheet, sheet_name, fields_to_keep):
    # Fetches the file content.
    response = requests.get(spreadsheet)
    response.raise_for_status()  # Ensure we notice bad responses

    # Converts to Excel.
    excel_df = pd.read_excel(BytesIO(response.content), sheet_name=sheet_name)

    # Retains only the relevant columns
    filtered_data = excel_df[fields_to_keep]

    # Converts from dataframe to Numpy array
    filtered_array = filtered_data.to_numpy().astype(
        float)  # Need to convert Pandas dataframe to numpy array because Numba jit-decorated function can't use dataframes.

    return filtered_array


# Creates arrays of 0s for any missing inputs and puts them in the corresponding typed dictionary
def complete_inputs(existing_input_list, typed_dict, datatype, chunk_length_pixels, bounds_str, tile_id, is_final, logger):
    for dataset_name in existing_input_list:
        if dataset_name not in typed_dict.keys():
            typed_dict[dataset_name] = np.full((chunk_length_pixels, chunk_length_pixels), 0, dtype=datatype)
            lu.print_and_log(f"Created {dataset_name} for chunk {bounds_str} in {tile_id}: {timestr()}", is_final, logger)
    return typed_dict
