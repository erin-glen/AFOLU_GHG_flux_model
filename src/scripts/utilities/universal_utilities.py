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
import tempfile
from botocore.config import Config
from dask.distributed import print
from dask.distributed import Client
from datetime import datetime
from io import BytesIO
from osgeo import gdal
import sys

# Local project imports
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import log_utilities as lu

################################################################################
# Time / Logging / Cluster
################################################################################

def timestr():
    """
    Returns the current US/Eastern time as a string in YYYYMMDD_H_M_S format.
    """
    eastern = pytz.timezone('US/Eastern')
    now_eastern = datetime.now(eastern)
    return now_eastern.strftime("%Y%m%d_%H_%M_%S")


def connect_to_Coiled_cluster(cluster_name, run_local):
    """
    Connects to a Coiled cluster of a specified name, or uses a local Dask Client if run_local=True.
    """
    if run_local:
        print("Running locally with a local Dask client.")
        cluster = None
        client = Client()
        return cluster, client
    else:
        # Minimal Coiled cluster creation for organic_soils approach
        cluster = coiled.Cluster(
            name=cluster_name,
            account='wri-forest-research',
            # Include other cluster configurations if needed
        )
        client = cluster.get_client()
        return cluster, client

################################################################################
# Chunk / Tiling Logic
################################################################################

def boundstr(bounds):
    """
    Converts bounding box [W, S, E, N] into a string with integer rounding: 'W_S_E_N'
    """
    return "_".join([str(round(x)) for x in bounds])


def calc_chunk_length_pixels(bounds):
    """
    Returns the pixel dimension along one side of the bounding box, assuming 0.00025 deg (40000 px in 10 deg).
    """
    return int((bounds[3] - bounds[1]) * (40000 / 10))


def get_10x10_tile_bounds(tile_id):
    """
    For a tile_id like '02N_010E' or '10S_050W', returns (W, S, E, N).
    """
    if "S" in tile_id:
        max_y = -1 * (int(tile_id[:2]))
        min_y = -1 * (int(tile_id[:2]) + 10)
    else:
        max_y = int(tile_id[:2])
        min_y = int(tile_id[:2]) - 10

    if "W" in tile_id:
        max_x = -1 * (int(tile_id[4:7]) - 10)
        min_x = -1 * (int(tile_id[4:7]))
    else:
        max_x = int(tile_id[4:7]) + 10
        min_x = int(tile_id[4:7])

    return min_x, min_y, max_x, max_y  # W, S, E, N


def get_chunk_bounds(bounding_box, chunk_size):
    """
    Returns list of [W, S, E, N] sub-chunks subdividing 'bounding_box',
    each chunk_size degrees wide/tall.
    """
    min_x, min_y, max_x, max_y = bounding_box
    x = min_x
    y = min_y

    chunks = []
    while y < max_y:
        while x < max_x:
            bounds = [x, y, x + chunk_size, y + chunk_size]
            chunks.append(bounds)
            x += chunk_size
        x = min_x
        y += chunk_size
    return chunks


def xy_to_tile_id(top_left_x, top_left_y):
    """
    From top-left XY, produce the standard tile_id string: 'YYN_XXXE' or 'YYS_XXXW'.
    """
    lat_ceil = math.ceil(top_left_y / 10.0) * 10
    lng_floor = math.floor(top_left_x / 10.0) * 10

    lng = f"{str(abs(lng_floor)).zfill(3)}{'E' if lng_floor >= 0 else 'W'}"
    lat = f"{str(abs(lat_ceil)).zfill(2)}{'N' if lat_ceil >= 0 else 'S'}"
    return f"{lat}_{lng}"


def stage_duration(start_time_str, end_time_str, stage):
    """
    Logs elapsed time for a named stage, from start_time_str to end_time_str.
    Both are in timestr() format: YYYYMMDD_H_M_S.
    """
    start_time = datetime.strptime(start_time_str, "%Y%m%d_%H_%M_%S")
    end_time = datetime.strptime(end_time_str, "%Y%m%d_%H_%M_%S")
    print(f"Elapsed time for {stage}: {end_time - start_time}")

################################################################################
# GDAL / Data Type Mappings
################################################################################

# Old "gdal_dtype_mapping" was partial. We'll keep a more complete dictionary:
gdal_dtype_mapping = {
    gdal.GDT_Byte: 'Byte',
    gdal.GDT_UInt16: 'UInt16',
    gdal.GDT_Int16: 'Int16',
    gdal.GDT_UInt32: 'UInt32',
    gdal.GDT_Int32: 'Int32',
    gdal.GDT_Float32: 'Float32',
    gdal.GDT_Float64: 'Float64'
}

def map_to_numpy_dtype(data_type):
    """
    Converts a string like 'Float32'/'Byte' into a Numpy dtype
    """
    dtype_map = {
        'Float32': 'float32',
        'Float64': 'float64',
        'Byte': 'uint8',
        'UInt16': 'uint16',
        'Int16': 'int16',
        'UInt32': 'uint32',
        'Int32': 'int32'
        # Add more as needed
    }
    return dtype_map.get(data_type, 'float32')

################################################################################
# S3 Path Splitting / File Transfer
################################################################################

def split_s3_path(s3_path):
    """
    Splits 's3://bucket/...key...' into (bucket, key).
    """
    s3_path_clean = s3_path.replace("s3://", "")
    bucket, key = s3_path_clean.split("/", 1)
    return bucket, key


def download_s3_file(s3_path, local_path):
    """
    Downloads a file from s3_path to local_path.
    """
    s3 = boto3.client('s3')
    bucket, key = split_s3_path(s3_path)
    s3.download_file(bucket, key, local_path)


def upload_s3_file(s3_path, local_path):
    """
    Uploads a local file to s3_path.
    """
    s3 = boto3.client('s3')
    bucket, key = split_s3_path(s3_path)
    s3.upload_file(local_path, bucket, key)

################################################################################
# Tile Reading and Checking
################################################################################

def get_tile_dataset_rio(uri, data_type, bounds, chunk_length_pixels, is_final, logger):
    """
    Lazily opens tile within provided bounds ([W, S, E, N]) using rasterio
    and returns it as a numpy array. If the tile or window is unavailable,
    returns an array of zeros with the same shape & correct data type.

    Note: This old-branch-friendly version returns JUST 'data' (no status).
    """
    try:
        with rasterio.open(uri) as ds:
            window = rasterio.windows.from_bounds(*bounds, ds.transform)
            data = ds.read(1, window=window)
    except Exception as e:
        numpy_dtype = map_to_numpy_dtype(data_type)
        data = np.full((chunk_length_pixels, chunk_length_pixels), 0).astype(numpy_dtype)
        lu.print_and_log(f"flm: Error accessing dataset at {uri}. Returning all 0s: {e}",
                         is_final, logger)

    return data


def prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_final, logger):
    """
    Submits concurrent tasks to read each input layer chunk with get_tile_dataset_rio.
    Returns a dict of {Future: layer_key}.
    """
    futures = {}
    bounds_str = boundstr(bounds)
    tile_id = xy_to_tile_id(bounds[0], bounds[3])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        lu.print_and_log(f"Requesting data in chunk {bounds_str} in {tile_id}: {timestr()}",
                         is_final, logger)
        for key, value in updated_download_dict.items():
            future = executor.submit(get_tile_dataset_rio,
                                     value[0],  # s3_path
                                     value[1],  # data_type
                                     bounds,
                                     chunk_length_pixels,
                                     is_final,
                                     logger)
            futures[future] = key
    return futures


def check_for_tile(download_dict, is_final, logger):
    """
    Checks if at least one tile in download_dict actually exists in S3
    by using head_object. If none exist, returns False.
    """
    s3_config = Config(retries={'max_attempts': 10, 'mode': 'standard'})
    s3_client = boto3.client("s3", config=s3_config)

    tile_id = None
    for value in download_dict.values():
        s3_path = value[0]
        # strip the initial "s3://gfw2-data/"
        s3_key = s3_path.replace("s3://gfw2-data/", "", 1)

        # Extract the tile_id for logging
        matches = re.findall(cn.tile_id_pattern, s3_path)
        if matches:
            tile_id = matches[0]
        else:
            logger.warning(f"No tile_id found in file path: {s3_path}")

        try:
            # If head_object succeeds on any tile, we return True
            s3_client.head_object(Bucket='gfw2-data', Key=s3_key)
            lu.print_and_log(f"Tile id {tile_id} exists. Proceeding: {timestr()}", is_final, logger)
            return True
        except Exception:
            pass

    if tile_id:
        lu.print_and_log(f"Tile id {tile_id} does not exist. Skipped chunk: {timestr()}",
                         is_final, logger)
    else:
        lu.print_and_log(f"No tile_id found at all. Skipped chunk: {timestr()}",
                         is_final, logger)

    return False


def check_chunk_for_data(required_layers, bounds_str, tile_id, any_or_all, is_final, logger):
    """
    Checks if the chunk has actual data. If any_or_all == 'any', returns True
    as soon as it finds a layer that isn't all zeros. If 'all', it checks min vs. max.
    """
    if any_or_all == "any":
        for array in required_layers.values():
            min_val = np.min(array)
            # If min_val is not None (which it always is) but we do this check anyway:
            if min_val != 0:
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
                logger.info(f"flm: Chunk {bounds_str} does not exist for {key}. "
                            f"Skipped chunk: {timestr()}")
                print(f"flm: Chunk {bounds_str} does not exist for {key}. Skipped chunk: {timestr()}")
                return False

        logger.info(f"flm: Chunk {bounds_str} has data for all assessed inputs: {timestr()}")
        print(f"flm: Chunk {bounds_str} has data for all assessed inputs: {timestr()}")
        return True

    else:
        raise ValueError("any_or_all argument must be 'any' or 'all'.")

################################################################################
# Save & Upload Rasters
################################################################################

def save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id,
                                     bounds_str, output_dict, is_final, logger,
                                     no_data_val=None):
    """
    Saves output arrays locally as GeoTIFFs, then uploads them to S3, then removes local files.
    Uses cross-platform 'tempfile.gettempdir()' for the local directory.

    Args:
        bounds: [W, S, E, N]
        chunk_length_pixels: e.g. 8000
        tile_id: e.g. "00N_110E"
        bounds_str: e.g. "112_-4_114_-2"
        output_dict: {
            'soil': [data_array, 'float32', 'soil', '2020'],
            'state': [data_array, 'uint8', 'state', '2020']
        }
        is_final: bool controlling verbosity
        logger: logging.Logger instance
        no_data_val: optional nodata for the raster

    Returns:
        None
    """
    s3_config = Config(retries={'max_attempts': 10, 'mode': 'standard'})
    s3_client = boto3.client("s3", config=s3_config)

    from rasterio.transform import from_bounds
    try:
        transform = from_bounds(*bounds, width=chunk_length_pixels, height=chunk_length_pixels)
    except Exception as e:
        logger.error(f"Failed to create transform from bounds {bounds}: {e}")
        raise

    file_info = f"{tile_id}__{bounds_str}"

    if is_final:
        lu.print_and_log(f"Saving and uploading outputs for {bounds_str} in {tile_id}: {timestr()}",
                         is_final, logger)

    # Cross-platform temporary directory
    temp_dir = tempfile.gettempdir()
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)

    for key, value in output_dict.items():
        data_array = value[0]
        data_type = value[1]
        data_meaning = value[2]
        year_out = value[3]

        if is_final:
            file_name = f"{file_info}__{key}.tif"
        else:
            file_name = f"{file_info}__{key}__{timestr()}.tif"

        local_file_path = os.path.join(temp_dir, file_name)

        # Optionally log this step
        if not is_final:
            lu.print_and_log(f"Saving chunk {bounds_str} in {tile_id} for {year_out}: {timestr()}",
                             is_final, logger)

        rasterio_kwargs = {
            'driver': 'GTiff',
            'width': chunk_length_pixels,
            'height': chunk_length_pixels,
            'count': 1,
            'dtype': data_type,
            'crs': 'EPSG:4326',
            'transform': transform,
            'compress': 'lzw',
            'blockxsize': 400,
            'blockysize': 400
        }
        if no_data_val is not None:
            rasterio_kwargs['nodata'] = no_data_val

        # Save locally
        try:
            with rasterio.open(local_file_path, 'w', **rasterio_kwargs) as dst:
                dst.write(data_array, 1)
        except Exception as e:
            lu.print_and_log(f"Failed to save raster {file_name} locally: {e}",
                             is_final, logger)
            continue

        # Construct an S3 path for this example. If your code uses a certain layout, do so here:
        # This is just one example that might match older references:
        # "s3://gfw2-data/climate/organic_soils/{data_meaning}/{year_out}/.../CHUNK"
        # If needed, adjust to your pipeline’s structure. We’ll keep it direct:
        s3_folder = f"s3://gfw2-data/climate/organic_soils/{data_meaning}/{year_out}/{chunk_length_pixels}_pixels"
        # Clean up path
        s3_folder_no_prefix = s3_folder.replace("s3://gfw2-data/", "")
        s3_key = f"{s3_folder_no_prefix}/{file_name}"

        # Upload
        try:
            s3_client.upload_file(local_file_path, "gfw2-data", s3_key)
            lu.print_and_log(f"Uploaded {local_file_path} to s3://gfw2-data/{s3_key}",
                             is_final, logger)
        except Exception as e:
            lu.print_and_log(f"Failed to upload {local_file_path} to s3://gfw2-data/{s3_key}: {e}",
                             is_final, logger)
            continue

        # Verify
        try:
            s3_client.head_object(Bucket="gfw2-data", Key=s3_key)
            lu.print_and_log(f"Confirmed upload of {file_name} to s3://gfw2-data/{s3_key}",
                             is_final, logger)
        except Exception as e:
            lu.print_and_log(f"Upload verification failed for s3://gfw2-data/{s3_key}: {e}",
                             is_final, logger)
            continue

        # Remove local file
        try:
            os.remove(local_file_path)
        except OSError as e:
            lu.print_and_log(f"Failed to delete local raster {file_name}: {e}",
                             is_final, logger)

    if is_final:
        lu.print_and_log(f"All rasters for {bounds_str} in {tile_id} have been processed and uploaded.",
                         is_final, logger)


################################################################################
# Listing / Uploading Shapefiles and Rasters
################################################################################

def list_rasters_in_folder(full_in_folder):
    """
    Shells out 'aws s3 ls' to get all .tif files in an S3 folder. Returns list of filenames only.
    """
    cmd = ['aws', 's3', 'ls', full_in_folder]
    s3_contents_bytes = subprocess.check_output(cmd)
    s3_contents_str = s3_contents_bytes.decode('utf-8')
    s3_contents_list = s3_contents_str.splitlines()
    rasters = [line.split()[-1] for line in s3_contents_list if "tif" in line]
    return rasters


def upload_shp(in_folder, shp):
    """
    Uploads a shapefile (and sidecar files) from /tmp to s3 in the given folder.
    in_folder should be an s3 path like 's3://gfw2-data/xyz/'
    """
    print(f"flm: Uploading to {in_folder}{shp}: {timestr()}")
    shp_pattern = shp[:-4]
    s3_client = boto3.client("s3")

    # For old references: in_folder[15:] was used to remove 's3://gfw2-data/' from the front.
    # If your prefix differs, adjust accordingly:
    s3_prefix = in_folder.replace("s3://gfw2-data/", "")

    s3_client.upload_file(f"/tmp/{shp}",        "gfw2-data", f"{s3_prefix}{shp}")
    s3_client.upload_file(f"/tmp/{shp_pattern}.dbf", "gfw2-data", f"{s3_prefix}{shp_pattern}.dbf")
    s3_client.upload_file(f"/tmp/{shp_pattern}.prj", "gfw2-data", f"{s3_prefix}{shp_pattern}.prj")
    s3_client.upload_file(f"/tmp/{shp_pattern}.shx", "gfw2-data", f"{s3_prefix}{shp_pattern}.shx")

    os.remove(f"/tmp/{shp}")
    os.remove(f"/tmp/{shp_pattern}.dbf")
    os.remove(f"/tmp/{shp_pattern}.prj")
    os.remove(f"/tmp/{shp_pattern}.shx")

    print(f"flm: Uploaded to {in_folder}{shp}: {timestr()}")


def make_tile_footprint_shp(input_dict):
    """
    Creates shapefile footprints of rasters in the given S3 folder, using 'gdaltindex'.
    input_dict is { folder_s3_path: pattern_string }.
    """
    in_folder = list(input_dict.keys())[0]
    pattern = list(input_dict.values())[0]

    print(f"flm: Making tile index shapefile for: {in_folder}: {timestr()}")

    s3_in_folder = in_folder
    vsis3_in_folder = f"/vsis3/{in_folder[5:]}"  # remove "s3://"

    filenames = list_rasters_in_folder(s3_in_folder)
    tile_paths = [vsis3_in_folder + fn for fn in filenames]

    file_paths_txt = f"s3_paths_{pattern}.txt"
    with open(f"/tmp/{file_paths_txt}", 'w') as f:
        for item in tile_paths:
            f.write(item + '\n')

    shp = f"raster_footprints_{pattern}.shp"

    cmd = [
        "gdaltindex",
        "-t_srs", "EPSG:4326",
        f"/tmp/{shp}",
        "--optfile", f"/tmp/{file_paths_txt}"
    ]
    subprocess.check_call(cmd)

    # Upload shapefile
    upload_shp(s3_in_folder, shp)
    os.remove(f"/tmp/{file_paths_txt}")

    return f"Completed: {timestr()}"


def save_and_upload_raster_10x10(**kwargs):
    """
    Saves a single xarray data array as a raster locally (/tmp), then uploads to S3.
    Typically used by older code.
    Required kwargs keys: 'data', 'out_file_name', 'out_folder'.
    """
    data_array = kwargs['data']
    out_file_name = kwargs['out_file_name']
    out_folder = kwargs['out_folder']

    s3_client = boto3.client("s3")
    print(f"flm: Saving {out_file_name} locally")

    profile_kwargs = {'compress': 'lzw'}
    data_array.rio.to_raster(f"/tmp/{out_file_name}", **profile_kwargs)

    # out_folder might be like "s3://gfw2-data/...subfolder..."
    print(f"flm: Saving {out_file_name} to {out_folder[10:]}{out_file_name}")
    s3_client.upload_file(f"/tmp/{out_file_name}", "gfw2-data", Key=f"{out_folder[10:]}{out_file_name}")

    os.remove(f"/tmp/{out_file_name}")

################################################################################
# Stats
################################################################################

def calculate_stats(array, name, bounds_str, tile_id, in_out):
    """
    Computes min, mean, max for a given chunk array. If array is empty or None, returns 'no data' stats.
    """
    if array is None or not np.any(array):
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
    else:
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


def calculate_chunk_stats(all_stats, stage):
    """
    Takes a list of per-chunk stats dictionaries and writes them to an Excel file
    in cn.chunk_stats_path. Minimally updated from older code.
    """
    if not all_stats:
        print("No statistics to calculate; all_stats is empty.")
        return

    print("Calculating tile stats...")

    df_all_stats = pd.DataFrame(all_stats)
    df_all_stats['min_value'] = pd.to_numeric(df_all_stats['min_value'], errors='coerce')
    df_all_stats['max_value'] = pd.to_numeric(df_all_stats['max_value'], errors='coerce')

    sorted_stats = df_all_stats.sort_values(by=['in_out', 'layer_name']).reset_index(drop=True)

    min_max_stats = df_all_stats.groupby('layer_name').agg(
        min_value=('min_value', 'min'),
        max_value=('max_value', 'max')
    ).reset_index()

    in_out_tables = {
        val: sorted_stats[sorted_stats['in_out'] == val]
        for val in sorted_stats['in_out'].unique()
    }

    stats_dir = cn.chunk_stats_path
    if not os.path.exists(stats_dir):
        try:
            os.makedirs(stats_dir, exist_ok=True)
            print(f"Created directory for chunk stats at: {stats_dir}")
        except Exception as e:
            print(f"Error creating directory {stats_dir}: {e}")
            return

    filename = f"{stage}_chunk_statistics_{timestr()}.xlsx"
    filepath = os.path.join(stats_dir, filename)

    try:
        with pd.ExcelWriter(filepath) as writer:
            for in_out_value, table in in_out_tables.items():
                sheet_name = f"chunk_stats_{str(in_out_value)}"
                table.to_excel(writer, sheet_name=sheet_name, index=False)
            min_max_stats.to_excel(writer, sheet_name='min_max_for_layers', index=False)
        print(f"Chunk statistics successfully saved to {filepath}")
        print(sorted_stats.head())
    except Exception as e:
        print(f"Can't print chunk stats: {e}")

################################################################################
# Data Type Checking
################################################################################

def first_file_name_in_s3_folder(download_dict):
    """
    Retrieves first file found under each path in download_dict, substituting '{tile_id}' with '00N_110E' (example).
    Returns {key: 's3://...firstfile...'} or None if none found.
    """
    s3_client = boto3.client("s3")
    first_tiles = {}
    sample_tile_id = '00N_110E'  # Hard-coded example

    for key, folder_path in download_dict.items():
        replaced_path = folder_path.replace('{tile_id}', sample_tile_id)
        dir_path = os.path.dirname(replaced_path)
        # remove 's3://'
        dir_path_no_prefix = dir_path[len('s3://'):]
        if '/' in dir_path_no_prefix:
            bucket, prefix = dir_path_no_prefix.split('/', 1)
        else:
            bucket = dir_path_no_prefix
            prefix = ''
        prefix += '/'

        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' in response and response['Contents']:
            first_file_key = response['Contents'][0]['Key']
            first_tiles[key] = f"s3://{bucket}/{first_file_key}"
        else:
            first_tiles[key] = None

    return first_tiles


def get_dtype_from_s3(file_path):
    """
    Opens a raster in S3 via /vsis3/ to retrieve its GDAL data type name.
    """
    if file_path is None:
        print("File path is None. Cannot determine data type.")
        return None
    vsis3_path = f"/vsis3/{file_path[len('s3://'):]}"
    dataset = gdal.Open(vsis3_path)
    if dataset:
        band = dataset.GetRasterBand(1)
        data_type = gdal.GetDataTypeName(band.DataType)
        return data_type
    else:
        raise ValueError(f"Could not open file {vsis3_path}")


def add_file_type_to_dict(first_tiles):
    """
    For each dataset key in 'first_tiles', gets the data type from the first tile found,
    then returns a dict {key: [file_path, data_type]}.
    """
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


def replace_tile_id_in_dict(data_dict, new_tile_id):
    """
    Replaces a tile_id pattern in the S3 paths of data_dict with 'new_tile_id'.
    """
    tile_id_pattern = cn.tile_id_pattern if hasattr(cn, 'tile_id_pattern') else r'\d{2}[NS]_\d{3}[EW]'
    for key, value in data_dict.items():
        file_path = value[0]
        updated_file_path = re.sub(tile_id_pattern, new_tile_id, file_path)
        data_dict[key][0] = updated_file_path
    return data_dict


################################################################################
# Handling Missing Layers
################################################################################

def fill_missing_input_layers_with_no_data(layers,
                                           uint8_list,
                                           int16_list,
                                           int32_list,
                                           float32_list,
                                           bounds_str,
                                           tile_id,
                                           is_final,
                                           logger):
    """
    Fills any missing layers with 0 arrays of the appropriate dtype, guided by the lists provided.
    """
    existing_array = next(iter(layers.values()), None)
    if existing_array is not None:
        array_shape = existing_array.shape
    else:
        raise ValueError(f"No data in any layer to determine shape for missing layers, chunk {bounds_str} in {tile_id}: {timestr()}")

    data_type_lists = {
        np.uint8: uint8_list,
        np.int16: int16_list,
        np.int32: int32_list,
        np.float32: float32_list
    }
    for dtype, keys_list in data_type_lists.items():
        for k in keys_list:
            if k not in layers:
                layers[k] = np.zeros(array_shape, dtype=dtype)
                lu.print_and_log(f"Filled missing layer '{k}' with NoData for chunk {bounds_str} in {tile_id}: {timestr()}",
                                 is_final, logger)
    return layers


def convert_lookup_table_to_array(spreadsheet, sheet_name, fields_to_keep):
    """
    Downloads an Excel file from 'spreadsheet', reads 'sheet_name', extracts columns in fields_to_keep,
    returns them as a float32 numpy array. Typically used for emission factor tables or so.
    """
    response = requests.get(spreadsheet)
    response.raise_for_status()
    excel_df = pd.read_excel(BytesIO(response.content), sheet_name=sheet_name)
    filtered_data = excel_df[fields_to_keep]
    filtered_array = filtered_data.to_numpy().astype(float)
    return filtered_array


def complete_inputs(existing_input_list, typed_dict, datatype, chunk_length_pixels,
                    bounds_str, tile_id, is_final, logger):
    """
    For each dataset_name in existing_input_list, if it's not in typed_dict,
    create a zero array of shape (chunk_length_pixels, chunk_length_pixels) in the given 'datatype'.
    """
    for dataset_name in existing_input_list:
        if dataset_name not in typed_dict.keys():
            typed_dict[dataset_name] = np.full((chunk_length_pixels, chunk_length_pixels), 0,
                                               dtype=datatype)
            lu.print_and_log(f"Created {dataset_name} for chunk {bounds_str} in {tile_id}: {timestr()}",
                             is_final, logger)
    return typed_dict

################################################################################
# Aggregation / Merging
################################################################################

def flatten_list(nested_list):
    """
    Flattens a nested list [[a,b],[c,d]] -> [a,b,c,d].
    """
    return [x for xs in nested_list for x in xs]


def create_list_for_aggregation(s3_in_folders):
    """
    For each folder in s3_in_folders, lists all .tif, transforms the chunked
    filenames into final 10x10 names, and returns a combined list of dictionary items:
       [{folder_path: [output_filename]}, ...]
    """
    list_of_s3_names_total = []
    for s3_in_folder in s3_in_folders:
        print(f"Listing files in {s3_in_folder}")
        simple_output_file_names = []

        filenames = list_raster_names_in_s3_folder(s3_in_folder)
        for filename in filenames:
            result = re.sub(cn.small_chunk_pattern, '__', filename)
            simple_output_file_names.append(result)

        simple_output_file_names = np.unique(simple_output_file_names).tolist()
        simple_output_file_names = [[item] for item in simple_output_file_names]

        list_of_s3_name_dicts = [{key: value} for value in simple_output_file_names for key in [s3_in_folder]]
        list_of_s3_names_total.append(list_of_s3_name_dicts)

    list_of_s3_names_total = flatten_list(list_of_s3_names_total)
    print(f"There are {len(list_of_s3_names_total)} 10x10 deg rasters to create across {len(s3_in_folders)} input folders.")
    return list_of_s3_names_total


def list_raster_names_in_s3_folder(s3_in_folder):
    """
    Lists all objects ending in .tif from the given s3_in_folder (like 's3://bucket/prefix').
    Returns list of filenames (no path).
    """
    print(f"Listing files in {s3_in_folder}")
    s3_in_folder = s3_in_folder.replace('s3://', '')
    bucket_name, *prefix_parts = s3_in_folder.split('/')
    prefix = '/'.join(prefix_parts)

    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    filenames = []
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('.tif') or key.endswith('.tiff'):
                    filename = key.split('/')[-1]
                    filenames.append(filename)
    return filenames


def merge_small_tiles_gdal(s3_name_dict, is_final, no_upload, no_log):
    """
    Merges multiple 1x1 deg rasters into a single 10x10 deg raster using gdal_merge.py,
    then optionally uploads to S3.
    """
    logger = lu.setup_logging()
    in_folder = list(s3_name_dict.keys())[0]
    out_file_name = list(s3_name_dict.values())[0][0]

    s3_in_folder = in_folder
    vsis3_in_folder = f"/vsis3/{in_folder[5:]}"  # remove 's3://'

    # gather all rasters in the folder
    filenames = list_raster_names_in_s3_folder(s3_in_folder)
    tile_id = out_file_name[:8]
    filenames_in_focus_area = [i for i in filenames if tile_id in i]
    tile_paths = [vsis3_in_folder + fn for fn in filenames_in_focus_area]

    lu.print_and_log(f"Merging small rasters in {tile_id} in {vsis3_in_folder}",
                     is_final, logger)

    # updates the output folder from e.g. 8000_pixels to 40000_pixels
    out_folder = re.sub(r'\d+_pixels', f'{cn.full_raster_dims}_pixels', in_folder)

    min_x, min_y, max_x, max_y = get_10x10_tile_bounds(tile_id)

    first_raster_path = tile_paths[0]
    ds = gdal.Open(first_raster_path)
    raster_datatype = ds.GetRasterBand(1).DataType
    raster_nodata_value = ds.GetRasterBand(1).GetNoDataValue()
    if raster_nodata_value is None:
        raster_nodata_value = 0
    ds = None

    dtype_str = gdal_dtype_mapping.get(raster_datatype, 'Float32')

    merged_file = f"/tmp/merged_{out_file_name}"
    merge_command = [
        'gdal_merge.py',
        '-o', merged_file,
        '-of', 'GTiff',
        '-co', 'COMPRESS=DEFLATE',
        '-co', 'TILED=YES',
        '-co', 'BLOCKXSIZE=400',
        '-co', 'BLOCKYSIZE=400',
        '-ul_lr', str(min_x), str(max_y), str(max_x), str(min_y),
        '-ot', dtype_str,
        '-a_nodata', str(raster_nodata_value)
    ]
    merge_command.extend(tile_paths)

    try:
        subprocess.check_call(merge_command)
        lu.print_and_log(f"Successfully merged rasters into {merged_file}",
                         is_final, logger)
    except subprocess.CalledProcessError as e:
        lu.print_and_log(f"Error merging rasters: {e}", is_final, logger)
        return f"failure for {s3_name_dict}"

    if not no_upload:
        s3_client = boto3.client("s3")
        out_key = f"{out_folder[15:]}{out_file_name}"  # removing 's3://gfw2-data/'
        lu.print_and_log(f"Saving {out_file_name} to s3: {out_folder}{out_file_name}",
                         is_final, logger)
        try:
            s3_client.upload_file(merged_file, "gfw2-data", out_key)
            lu.print_and_log(f"Successfully uploaded {out_file_name} to s3", is_final, logger)
        except boto3.exceptions.S3UploadFailedError as e:
            lu.print_and_log(f"Error uploading file to s3: {e}", is_final, logger)
            return f"failure for {s3_name_dict}"

    os.remove(merged_file)
    return f"success for {s3_name_dict}"


def list_raster_full_paths_in_s3_folder_and_count(s3_path):
    """
    Lists all .tif/.tiff files (full S3 URIs) within a prefix, returning (list, count).
    """
    s3_client = boto3.client('s3')
    geotiff_files = []
    try:
        if s3_path.startswith("s3://"):
            path_parts = s3_path[5:].split("/", 1)
            bucket_name = path_parts[0]
            prefix = path_parts[1] if len(path_parts) > 1 else ""
        else:
            raise ValueError(f"Invalid S3 path: {s3_path}")

        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                files = [obj['Key'] for obj in page['Contents']]
                geotiffs = [f"s3://{bucket_name}/{f}" for f in files if f.endswith(('.tif', '.tiff'))]
                geotiff_files.extend(geotiffs)
    except Exception as e:
        print(f"Error accessing {s3_path}: {e}")

    return geotiff_files, len(geotiff_files)

def get_interval_info(start_year, end_year, main_logger):
    """
    Determines whether we're doing annual, 5-year, or hybrid intervals
    based on the given start_year and end_year. Returns:
      - interval_type   (cn.intervals_annual, cn.intervals_five_years, or cn.intervals_hybrid)
      - interval_year_diff  (the effective 'interval_duration - 1' for five_year or 1 for annual, or a list if hybrid)
      - interval_length     (5 or 1, or a list if hybrid)
      - output_years        (the list of 'interval_end_years')

    Currently supports:
      - (2000, 2020) => 5-year intervals
      - (2015, 2023) => annual intervals
      - (2015, 2020) => annual intervals
      - (2000, 2023) => hybrid
    """
    # We can do partial logic for 2015..2020 as annual intervals:
    if start_year == 2000 and end_year == 2020:
        interval_type = cn.intervals_five_years
        interval_year_diff = cn.interval_duration - 1  # 4
        interval_length = cn.interval_duration         # 5
        output_years = cn.interval_end_years_5_years   # [2005, 2010, 2015, 2020]

    elif start_year == 2015 and end_year == 2023:
        interval_type = cn.intervals_annual
        interval_year_diff = 1
        interval_length = 1
        output_years = cn.interval_end_years_annual    # [2016, 2017, ..., 2023]

    elif start_year == 2015 and end_year == 2020:
        # We'll treat 2015..2020 as an annual run. We just define the end years as [2016, 2017, ..., 2020].
        interval_type = cn.intervals_annual
        interval_year_diff = 1
        interval_length = 1
        # So we need interval_end_years for 2015..2020:
        output_years = list(range(start_year + 1, end_year + 1))  # [2016, 2017, 2018, 2019, 2020]

    elif start_year == 2000 and end_year == 2023:
        interval_type = cn.intervals_hybrid
        # first part: the 2000–2020 5-year intervals
        # second part: the 2020–2023 annual intervals
        # Typically you'd do something like:
        interval_year_diff = (
            [cn.interval_duration - 1] * len(cn.interval_end_years_5_years[:-1]) +
            [1] * len(cn.interval_end_years_annual)
        )
        interval_length = (
            [cn.interval_duration] * len(cn.interval_end_years_5_years[:-1]) +
            [1] * len(cn.interval_end_years_annual)
        )
        output_years = cn.interval_end_years_5_years[:-1] + cn.interval_end_years_annual

    else:
        # Anything else => "interval_type not valid"
        main_logger.error("interval_type not valid")
        sys.exit(1)

    main_logger.info(f"Interval type: {interval_type}")
    main_logger.info(f"Interval duration: {interval_length} years")
    main_logger.info(f"Interval end years/Output years: {output_years}")

    return interval_type, interval_year_diff, interval_length, output_years
