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
from typing import Dict, List, Tuple, Union

import subprocess
import boto3
import coiled
import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
from botocore.config import Config
from dask.distributed import Client, LocalCluster
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
def get_chunk_bounds(
    bounding_box: List[float],
    chunk_size: float,
    *,
    as_polygons: bool = False,
) -> List[Union[List[float], "Polygon"]]:
    """Subdivide *bounding_box* into square chunks.

    Parameters
    ----------
    bounding_box : list[float]
        Bounds in ``[W, S, E, N]`` order.
    chunk_size : float
        Size of each chunk in degrees.
    as_polygons : bool, optional
        If ``True`` return :class:`shapely.geometry.Polygon` objects,
        otherwise return numeric bounds.  Defaults to ``False``.

    Returns
    -------
    list
        Sequence of chunk bounds or polygons.
    """
    min_x, min_y, max_x, max_y = bounding_box
    chunks = []
    y = min_y
    while y < max_y:
        x = min_x
        while x < max_x:
            if as_polygons:
                try:
                    from shapely.geometry import box
                except Exception as exc:  # pragma: no cover - import guard
                    raise ImportError(
                        "shapely is required when as_polygons is True"
                    ) from exc
                chunks.append(box(x, y, x + chunk_size, y + chunk_size))
            else:
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
# Connects to or creates a Coiled cluster unless running locally
def connect_to_cluster(
    cluster_name="afolu_cluster",
    n_workers: int = 20,
    region: str = "us-east-1",
    run_local: bool = False,
    worker_memory: str = "32GiB",
):
    """Connect to an existing Coiled cluster or create one.

    Parameters
    ----------
    cluster_name : str
        Name of the cluster to connect to or create.
    n_workers : int
        Number of workers when creating a new cluster.
    region : str
        AWS region for the cluster.
    run_local : bool
        If ``True``, create a local cluster instead of using Coiled.
        worker_memory : str
        Memory limit for each worker when creating a new cluster.
    """

    if run_local:
        cluster = LocalCluster()
        client = Client(cluster)
        return cluster, client

    try:
        cluster = coiled.Cluster.from_name(cluster_name)
        print(f"Connected to existing cluster: {cluster_name}")
    except Exception:
        print(
            f"No existing cluster with name '{cluster_name}' found. Creating a new cluster."
        )
        cluster = coiled.Cluster(
            name=cluster_name,
            n_workers=n_workers,
            use_best_zone=True,
            compute_purchase_option="spot_with_fallback",
            idle_timeout="15 minutes",
            region=region,
            account="wri-forest-research",
            worker_memory=worker_memory,
            shutdown_on_close=False,
        )
    client = cluster.get_client()
    return cluster, client


# ----------------------------------------------------------------------
# General S3 helpers
# ----------------------------------------------------------------------


def s3_file_exists(bucket: str, key: str) -> bool:
    """Return True if the given S3 object exists."""
    s3c = boto3.client("s3")
    try:
        s3c.head_object(Bucket=bucket, Key=key)
        return True
    except boto3.exceptions.Boto3Error:
        return False
    except Exception:
        return False


def list_s3_files(bucket: str, prefix: str) -> list:
    """List all keys under *prefix* in *bucket*."""
    keys = []
    s3c = boto3.client("s3")
    paginator = s3c.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def upload_file_to_s3(
    local_file_path: str, bucket_name: str, s3_file_path: str
) -> None:
    """Upload a local file to S3."""
    boto3.client("s3").upload_file(local_file_path, bucket_name, s3_file_path)


def upload_fileobj_to_s3(file_obj, bucket_name: str, s3_file_path: str) -> None:
    """Upload a file-like object to S3."""
    s3c = boto3.client("s3")
    file_obj.seek(0)
    s3c.upload_fileobj(file_obj, bucket_name, s3_file_path)


def download_file_from_s3(
    s3_file_path: str, local_file_path: str, bucket_name: str
) -> None:
    """Download an S3 key to a local path."""
    boto3.client("s3").download_file(bucket_name, s3_file_path, local_file_path)


def download_shapefile_from_s3(
    s3_prefix: str, local_dir: str, s3_bucket_name: str
) -> None:
    """Download a shapefile (and sidecars) from S3 to ``local_dir``."""
    s3c = boto3.client("s3")
    os.makedirs(local_dir, exist_ok=True)
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        s3_path = f"{s3_prefix}{ext}"
        local_path = os.path.join(local_dir, os.path.basename(s3_prefix) + ext)
        s3c.download_file(s3_bucket_name, s3_path, local_path)


def read_shapefile_from_s3(
    s3_prefix: str, local_dir: str, s3_bucket_name: str
) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame loaded from a shapefile stored on S3."""
    download_shapefile_from_s3(s3_prefix, local_dir, s3_bucket_name)
    shp = os.path.join(local_dir, os.path.basename(s3_prefix) + ".shp")
    return gpd.read_file(shp)


def get_existing_s3_files(s3_bucket: str, s3_prefix: str) -> set:
    """Return set of keys under ``s3_prefix``."""
    existing = set()
    s3c = boto3.client("s3")
    paginator = s3c.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)
    for page in pages:
        for obj in page.get("Contents", []):
            existing.add(obj["Key"])
    return existing


def list_s3_files_with_pattern(s3_path: str, pattern: str) -> list:
    """Return a list of S3 paths under ``s3_path`` containing ``pattern``."""
    bucket_name, prefix = split_s3_path(s3_path)
    matching_files = []
    s3c = boto3.client("s3")
    continuation_token = None
    while True:
        if continuation_token:
            resp = s3c.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                ContinuationToken=continuation_token,
            )
        else:
            resp = s3c.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if pattern in key:
                matching_files.append(f"s3://{bucket_name}/{key}")
        if resp.get("IsTruncated"):
            continuation_token = resp["NextContinuationToken"]
        else:
            break
    return matching_files


def list_raster_names_in_s3_folder(s3_in_folder: str) -> list:
    """Return GeoTIFF filenames from ``s3_in_folder``.

    Parameters
    ----------
    s3_in_folder : str
        Folder path like ``s3://bucket/prefix``. The value must begin with
        ``"s3://"``.

    Returns
    -------
    list
        Filenames (no paths) for ``.tif`` or ``.tiff`` objects.
    """
    if not s3_in_folder.startswith("s3://"):
        raise ValueError("S3 path must start with 's3://'")

    s3_in_folder = s3_in_folder.replace("s3://", "")
    bucket_name, *prefix_parts = s3_in_folder.split("/")
    prefix = "/".join(prefix_parts)
    s3c = boto3.client("s3")
    paginator = s3c.get_paginator("list_objects_v2")
    filenames = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".tif") or key.endswith(".tiff"):
                filenames.append(key.split("/")[-1])
    return filenames


def list_raster_full_paths_in_s3_folder_and_count(s3_path: str) -> tuple:
    """Return list of GeoTIFF paths under ``s3_path`` and a count."""
    s3c = boto3.client("s3")
    if s3_path.startswith("s3://"):
        parts = s3_path[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
    else:
        raise ValueError(f"Invalid S3 path: {s3_path}")
    paginator = s3c.get_paginator("list_objects_v2")
    geotiff_files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".tif") or key.endswith(".tiff"):
                geotiff_files.append(f"s3://{bucket}/{key}")
    return geotiff_files, len(geotiff_files)


def flatten_list(nested_list: list) -> list:
    """Flatten one level of nesting."""
    return [x for xs in nested_list for x in xs]


def create_list_for_aggregation(s3_in_folders: list, logger=None) -> list:
    """Return [{folder: [output_name]}, ...] for aggregating 1° tiles."""
    list_of_s3_names_total = []
    for folder in s3_in_folders:
        if logger:
            logger.info(f"Listing files in {folder}")
        simple_names = []
        for fname in list_raster_names_in_s3_folder(folder):
            simple = re.sub(cn.small_chunk_pattern, "__", fname)
            simple_names.append(simple)
        simple_names = np.unique(simple_names).tolist()
        list_of_s3_name_dicts = [{folder: [name]} for name in simple_names]
        list_of_s3_names_total.append(list_of_s3_name_dicts)
    list_of_s3_names_total = flatten_list(list_of_s3_names_total)
    if logger:
        logger.info(
            f"There are {len(list_of_s3_names_total)} 10x10 deg rasters to create across {len(s3_in_folders)} input folders."
        )
    else:
        print(
            f"There are {len(list_of_s3_names_total)} 10x10 deg rasters to create across {len(s3_in_folders)} input folders."
        )
    return list_of_s3_names_total


def merge_small_tiles_gdal(s3_name_dict, is_final, no_upload, no_log):
    logger = lu.setup_logging_worker()
    in_folder = list(s3_name_dict.keys())[0]
    out_file_name = list(s3_name_dict.values())[0][0]

    vsis3_in_folder = f"/vsis3/{in_folder[5:]}"  # Correct handling of vsis3 prefix

    filenames = list_raster_names_in_s3_folder(in_folder)
    tile_id = out_file_name[:8]
    filenames_in_focus_area = [i for i in filenames if tile_id in i]
    tile_paths = [f"{vsis3_in_folder}/{fn}" for fn in filenames_in_focus_area]

    lu.print_and_log(
        f"Merging small rasters in {tile_id} in {vsis3_in_folder}",
        is_final,
        logger,
    )

    if not tile_paths:
        lu.print_and_log(
            f"No tiles found for merging for {tile_id} in {vsis3_in_folder}",
            is_final,
            logger,
        )
        return f"no tiles found for {s3_name_dict}"

    out_folder = re.sub(r"\d+_pixels", f"{cn.full_raster_dims}_pixels", in_folder)

    min_x, min_y, max_x, max_y = get_10x10_tile_bounds(tile_id)

    first_raster_path = tile_paths[0]
    ds = gdal.Open(first_raster_path)
    raster_datatype = ds.GetRasterBand(1).DataType
    raster_nodata_value = ds.GetRasterBand(1).GetNoDataValue() or 0
    ds = None

    dtype_str = gdal_to_string_dtype_mapping.get(raster_datatype, "Float32")

    merged_file = f"/tmp/merged_{out_file_name}"
    merge_command = [
        "gdal_merge.py",
        "-o",
        merged_file,
        "-of",
        "GTiff",
        "-co", "COMPRESS=DEFLATE",
        "-co", "TILED=YES",
        "-co", "BLOCKXSIZE=400",
        "-co", "BLOCKYSIZE=400",
        "-ul_lr",
        str(min_x), str(max_y), str(max_x), str(min_y),
        "-ot", dtype_str,
        "-a_nodata", str(raster_nodata_value),
    ] + tile_paths

    try:
        subprocess.check_output(merge_command, stderr=subprocess.STDOUT)
        lu.print_and_log(
            f"Successfully merged into {merged_file}", is_final, logger
        )

        if not no_upload:
            s3_client = boto3.client("s3")
            out_key = f"{out_folder[15:]}/{out_file_name}"
            try:
                s3_client.upload_file(merged_file, "gfw2-data", out_key)
                lu.print_and_log(
                    f"Uploaded {out_file_name} to s3", is_final, logger
                )
            except boto3.exceptions.S3UploadFailedError as exc:
                lu.print_and_log(f"Error uploading to s3: {exc}", is_final, logger)
                return f"upload failure for {s3_name_dict}"
    except subprocess.CalledProcessError as exc:
        lu.print_and_log(f"GDAL merge error: {exc.output.decode()}", is_final, logger)
        return f"failure for {s3_name_dict}"
    finally:
        if os.path.exists(merged_file):
            os.remove(merged_file)

    return f"success for {s3_name_dict}"



def split_s3_path(s3_path: str) -> tuple:
    """Split ``s3://bucket/key`` into (bucket, key)."""
    s3_path = s3_path.replace("s3://", "")
    return tuple(s3_path.split("/", 1))


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

gdal_to_string_dtype_mapping = {
    gdal.GDT_Byte: "Byte",
    gdal.GDT_UInt16: "UInt16",
    gdal.GDT_Int16: "Int16",
    gdal.GDT_UInt32: "UInt32",
    gdal.GDT_Int32: "Int32",
    gdal.GDT_Float32: "Float32",
    gdal.GDT_Float64: "Float64",
    "Int8": "Int8",
    14: "Int8",
}

string_to_gdal_dtype_mapping = {
    "Byte": gdal.GDT_Byte,
    "UInt16": gdal.GDT_UInt16,
    "Int16": gdal.GDT_Int16,
    "UInt32": gdal.GDT_UInt32,
    "Int32": gdal.GDT_Int32,
    "Float32": gdal.GDT_Float32,
    "Float64": gdal.GDT_Float64,
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


def queue_chunk_downloads(
    bounds, typed_dict, chunk_px, logger, max_threads=16, is_final=False
):
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
    s3c = boto3.client(
        "s3", config=Config(retries={"max_attempts": 10, "mode": "standard"})
    )
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
            df[df["in_out"] == val].to_excel(
                xls, sheet_name=f"chunk_{val}", index=False
            )
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

    s3c = boto3.client(
        "s3", config=Config(retries={"max_attempts": 10, "mode": "standard"})
    )
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
    first_worker_address = next(iter(workers.keys()))
    nthreads = workers[first_worker_address]["nthreads"]

    # Retrieves scheduler info for other cluster properties
    scheduler_info = (
        cluster.scheduler_info
    )  # Access scheduler info directly as a dictionary

    # Gets memory per worker.
    # Can't get it to report the worker instance type
    try:
        worker_memory_bytes = scheduler_info["workers"][
            next(iter(scheduler_info["workers"]))
        ]["memory_limit"]
        worker_memory_gb = worker_memory_bytes / (1024**3)  # Convert bytes to GB
        worker_memory = f"{worker_memory_gb:.2f} GB"  # Format to 2 decimal places
        # worker_type = coiled_cluster.config.get('worker_options', {}).get('instance_type', "Unknown")
    except KeyError:
        worker_memory = "Unknown"
        # worker_type = "Unknown"

    return worker_memory, n_workers, nthreads