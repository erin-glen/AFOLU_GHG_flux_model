"""
universal_utilities.py
Shared helpers for AFOLU models (LULUCF & Organic‑soils).
All comments and log strings use plain ASCII.
"""

from __future__ import annotations
import concurrent.futures
import math
import os
import posixpath
import re
from datetime import datetime
from typing import Dict, List, Tuple

import boto3
import coiled
import numpy as np
import pandas as pd
import rasterio
from botocore.config import Config
from dask.distributed import Client
from rasterio import open as rio_open
from rasterio.windows import from_bounds

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import log_utilities as lu


# ----------------------------------------------------------------------
# basic helpers
# ----------------------------------------------------------------------
def timestr() -> str:
    import pytz

    return datetime.now(pytz.timezone("US/Eastern")).strftime("%Y%m%d_%H_%M_%S")


def stage_duration(start: str, end: str, stage: str) -> None:
    fmt = "%Y%m%d_%H_%M_%S"
    print(
        f"Elapsed for {stage}: {datetime.strptime(end, fmt) - datetime.strptime(start, fmt)}"
    )


# ----------------------------------------------------------------------
# tiling helpers
# ----------------------------------------------------------------------
def get_chunk_bounds(bounding_box: List[float], chunk_size: float) -> List[List[float]]:
    """
    Subdivide a bounding box [W, S, E, N] into square chunks that are
    chunk_size degrees on a side.  Returns a list of [W,S,E,N] lists.
    """
    min_x, min_y, max_x, max_y = bounding_box
    chunks = []
    y = min_y
    while y < max_y:
        x = min_x
        while x < max_x:
            chunks.append([x, y, x + chunk_size, y + chunk_size])
            x += chunk_size
        y += chunk_size
    return chunks


def get_10x10_tile_bounds(tile_id: str) -> Tuple[float, float, float, float]:
    """
    Convert a tile_id like '02N_010E' or '10S_050W' to (W,S,E,N) bounds.
    """
    lat_val = int(tile_id[:2])
    lon_val = int(tile_id[4:7])
    south = -lat_val - 10 if "S" in tile_id else lat_val - 10
    north = -lat_val if "S" in tile_id else lat_val
    west = -lon_val if "W" in tile_id else lon_val
    east = west + 10
    return west, south, east, north


def boundstr(bounds: List[float]) -> str:
    return "_".join(str(round(x)) for x in bounds)


def calc_chunk_length_pixels(bounds: List[float]) -> int:
    return int((bounds[3] - bounds[1]) * (40000 / 10))


def xy_to_tile_id(x: float, y: float) -> str:
    lat = math.ceil(y / 10.0) * 10
    lon = math.floor(x / 10.0) * 10
    lat_part = f"{abs(lat):02d}{'N' if lat >= 0 else 'S'}"
    lon_part = f"{abs(lon):03d}{'E' if lon >= 0 else 'W'}"
    return f"{lat_part}_{lon_part}"


# ----------------------------------------------------------------------
# Coiled / Dask
# ----------------------------------------------------------------------
# Connects to a Coiled cluster of a specified name if the local flag isn't on
def connect_to_cluster(cluster_name, run_local):
    """
    Connect to an existing Coiled cluster with the specified name.
    If no existing cluster is found, create a new one.
    If run_local is True, skip Coiled and run locally.
    """
    if run_local:
        print("Running locally without Dask/Coiled.")
        return None, None
    else:
        try:
            # Attempt to connect to an existing cluster by name.
            cluster = coiled.Cluster.from_name(cluster_name)
            print(f"Connected to existing cluster: {cluster_name}")
        except Exception as e:
            # If no such cluster exists, create a new one.
            print(f"No existing cluster with name '{cluster_name}' found. Creating a new cluster.")
            cluster = coiled.Cluster(name=cluster_name, shutdown_on_close=False)
        client = Client(cluster)
        return cluster, client



# ----------------------------------------------------------------------
# S3 / raster read helpers
# ----------------------------------------------------------------------

_DTYPE_MAP = {
    "Byte": np.uint8,
    "UInt16": np.uint16,
    "Int16": np.int16,
    "UInt32": np.uint32,
    "Int32": np.int32,
    "Float32": np.float32,
    "Float64": np.float64,
}


def _dtype(gdal_dtype: str) -> np.dtype:
    return _DTYPE_MAP[gdal_dtype]


def open_window_as_array(
    s3_uri: str,
    gdal_dtype: str,
    bounds: List[float],
    chunk_px: int,
    logger,
    is_final: bool = False,
) -> np.ndarray:
    """
    Read a window from an S3 GeoTIFF using GDAL's /vsis3/ driver.

    * If s3_uri is None (placeholder for a missing layer) we return an
      all‑zero array of the requested dtype.
    * Credentials are obtained automatically from environment variables
      or ~/.aws/credentials.  No s3fs / AWSSession wrapper is used.
    """
    if s3_uri is None:
        return np.zeros((chunk_px, chunk_px), dtype=_dtype(gdal_dtype))

    vsipath = f"/vsis3/{s3_uri[5:]}"  # strip the 's3://'
    try:
        with rasterio.Env():  # GDAL handles auth internally
            with rio_open(vsipath) as ds:
                return ds.read(1, window=from_bounds(*bounds, ds.transform))
    except Exception as exc:
        logger.warning(f"WARNING: {s3_uri} failed ({exc}) -> zeros")
        return np.zeros((chunk_px, chunk_px), dtype=_dtype(gdal_dtype))


def queue_chunk_downloads(bounds, typed_dict, chunk_px, logger, max_threads=16, is_final=False):
    futs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as ex:
        for k, (uri, dt) in typed_dict.items():
            futs[
                ex.submit(
                    open_window_as_array, uri, dt, bounds, chunk_px, logger, is_final
                )
            ] = k
    return futs


# ----------------------------------------------------------------------
# tile existence quick‑check
# ----------------------------------------------------------------------
def check_for_tile(typed_dict, is_final, logger) -> bool:
    s3c = boto3.client("s3", config=Config(retries={"max_attempts": 10, "mode": "standard"}))
    for uri, _ in typed_dict.values():
        if uri is None:
            continue
        key = uri.removeprefix("s3://gfw2-data/")
        tid_match = re.findall(cn.tile_id_pattern, uri)
        tid = tid_match[0] if tid_match else "unknown"
        try:
            s3c.head_object(Bucket="gfw2-data", Key=key)
            lu.print_and_log(f"Tile {tid} exists, continue", is_final, logger)
            return True
        except Exception:
            pass
    lu.print_and_log("Tile not found, skip chunk", is_final, logger)
    return False


# ----------------------------------------------------------------------
# stats helpers
# ----------------------------------------------------------------------
def calculate_stats(arr: np.ndarray | None, name, bstr, tid, in_out):
    if arr is None or arr.size == 0 or not np.any(arr):
        return dict(
            chunk_id=bstr,
            tile_id=tid,
            layer_name=name,
            in_out=in_out,
            min_value="no data",
            mean_value="no data",
            max_value="no data",
            data_type="no data",
        )
    return dict(
        chunk_id=bstr,
        tile_id=tid,
        layer_name=name,
        in_out=in_out,
        min_value=float(arr.min()),
        mean_value=float(arr.mean()),
        max_value=float(arr.max()),
        data_type=str(arr.dtype),
    )


def calculate_chunk_stats(all_stats, stage):
    if not all_stats:
        return
    df = pd.DataFrame(all_stats)
    df["min_value"] = pd.to_numeric(df["min_value"], errors="coerce")
    df["max_value"] = pd.to_numeric(df["max_value"], errors="coerce")
    out_dir = cn.chunk_stats_path
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stage}_chunk_stats_{timestr()}.xlsx")
    with pd.ExcelWriter(path) as xls:
        for val in df["in_out"].unique():
            df[df["in_out"] == val].to_excel(xls, sheet_name=f"chunk_{val}", index=False)
        df.groupby("layer_name").agg(
            min_value=("min_value", "min"), max_value=("max_value", "max")
        ).to_excel(xls, sheet_name="min_max", index=True)
    print(f"Stats written to {path}")


# ----------------------------------------------------------------------
# output path builder + uploader
# ----------------------------------------------------------------------
def build_output_s3_folder(
    data_meaning: str,
    year_out: str,
    chunk_px: int,
    interval_type: str,
    model_type: str = "standard_model",
    date_str: str | None = None,
) -> str:
    date_str = date_str or cn.today_date
    return posixpath.join(
        cn.outputs_path,
        data_meaning,
        model_type,
        f"{interval_type}_intervals",
        year_out,
        f"{chunk_px}_pixels",
        date_str,
    )


def save_and_upload_small_raster_set(
    bounds,
    chunk_px,
    tile_id,
    bstr,
    out_dict,
    is_final,
    logger,
    interval_type,
    model_type="standard_model",
    no_data_val=None,
):
    import tempfile
    from rasterio.transform import from_bounds

    s3c = boto3.client("s3", config=Config(retries={"max_attempts": 10, "mode": "standard"}))
    transform = from_bounds(*bounds, width=chunk_px, height=chunk_px)
    temp_dir = tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)

    for key, (arr, dtype, data_meaning, year_out) in out_dict.items():
        fname = (
            f"{tile_id}__{bstr}__{key}.tif"
            if is_final
            else f"{tile_id}__{bstr}__{key}__{timestr()}.tif"
        )
        lpath = os.path.join(temp_dir, fname)
        profile = dict(
            driver="GTiff",
            width=chunk_px,
            height=chunk_px,
            count=1,
            dtype=dtype,
            crs="EPSG:4326",
            transform=transform,
            compress="lzw",
            blockxsize=400,
            blockysize=400,
        )
        if no_data_val is not None:
            profile["nodata"] = no_data_val
        with rasterio.open(lpath, "w", **profile) as dst:
            dst.write(arr, 1)
        s3_folder = build_output_s3_folder(
            data_meaning, year_out, chunk_px, interval_type, model_type
        )
        s3_key = posixpath.join(s3_folder.removeprefix("s3://gfw2-data/"), fname)
        s3c.upload_file(lpath, "gfw2-data", s3_key)
        os.remove(lpath)
        lu.print_and_log(f"uploaded {fname} to {s3_folder}", is_final, logger)


# ----------------------------------------------------------------------
# helpers required by drainage model
# ----------------------------------------------------------------------
def fill_missing_input_layers_with_no_data(
    layers: Dict[str, np.ndarray],
    uint8_list,
    int16_list,
    int32_list,
    float32_list,
    bstr,
    tid,
    is_final,
    logger,
):
    template = next(iter(layers.values()))
    shp = template.shape
    for k in uint8_list:
        if k not in layers:
            layers[k] = np.zeros(shp, dtype=np.uint8)
    for k in int16_list:
        if k not in layers:
            layers[k] = np.zeros(shp, dtype=np.int16)
    for k in int32_list:
        if k not in layers:
            layers[k] = np.zeros(shp, dtype=np.int32)
    for k in float32_list:
        if k not in layers:
            layers[k] = np.zeros(shp, dtype=np.float32)
    return layers


# dtype sniffing helpers -----------------------------------------------------
def first_file_name_in_s3_folder(download_dict):
    s3 = boto3.client("s3")
    first_tiles = {}
    sample_tid = "00N_110E"
    for k, p in download_dict.items():
        dir_path = os.path.dirname(p.replace("{tile_id}", sample_tid))
        bucket, prefix = dir_path[5:].split("/", 1)
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/")
        if "Contents" in resp and resp["Contents"]:
            first_tiles[k] = f"s3://{bucket}/{resp['Contents'][0]['Key']}"
        else:
            first_tiles[k] = None
    return first_tiles


def get_dtype_from_s3(file_path):
    from osgeo import gdal

    vsipath = f"/vsis3/{file_path[5:]}"
    ds = gdal.Open(vsipath)
    if not ds:
        return None
    return gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)


def add_file_type_to_dict(first_tiles):
    out = {}
    for k, fp in first_tiles.items():
        if fp is None:
            continue
        dt = get_dtype_from_s3(fp)
        if dt:
            out[k] = [fp, dt]
    return out


def replace_tile_id_in_dict(d: Dict[str, List], new_tid: str):
    for k, v in d.items():
        v[0] = re.sub(cn.tile_id_pattern, new_tid, v[0])
    return d

def get_cluster_info(client, cluster):

    # Retrieves properties of the workers
    workers = client.scheduler_info()["workers"]

    # Retrieves the number of workers
    n_workers = len(workers)

    # Retrieves the number of threads per worker
    # https://chatgpt.com/share/e/672503f1-eef8-800a-9218-281624acf27e
    first_worker_address = next(iter(workers.keys()))
    nthreads = workers[first_worker_address]["nthreads"]

    # Retrieves scheduler info for other cluster properties
    scheduler_info = cluster.scheduler_info  # Access scheduler info directly as a dictionary

    # Gets memory per worker.
    # Can't get it to report the worker instance type
    try:
        worker_memory_bytes = scheduler_info['workers'][next(iter(scheduler_info['workers']))]['memory_limit']
        worker_memory_gb = worker_memory_bytes / (1024 ** 3)  # Convert bytes to GB
        worker_memory = f"{worker_memory_gb:.2f} GB"  # Format to 2 decimal places
        # worker_type = coiled_cluster.config.get('worker_options', {}).get('instance_type', "Unknown")
    except KeyError:
        worker_memory = "Unknown"
        # worker_type = "Unknown"

    return worker_memory, n_workers, nthreads
