"""Shared helpers for roads/canals preprocessing stages.

These helpers keep directory layout, filename conventions, and common
operations (mask loading, path construction) consistent across the 30 m
presence and distance scripts.
"""

from __future__ import annotations

import os
import posixpath
from typing import Iterable, List, Tuple

import boto3
import numpy as np
import rioxarray as rxr
from rasterio.warp import transform_bounds

from src.scripts.utilities import universal_utilities as uutil
import src.scripts.preprocessing.preprocessing_constants as cn


PEAT_30M_PATTERN = "_union_mask.tif"


def _split_feature_type(feature_type: str) -> Tuple[str, str]:
    """Return (group, sub) keys from feature_type such as ``osm_roads``."""

    try:
        return feature_type.split("_", 1)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"feature_type must contain '_' (got {feature_type!r})") from exc


def fmt_deg(value: float, precision: int = 1) -> str:
    """Format coordinates for chunk identifiers without unnecessary decimals."""

    formatted = f"{float(value):.{precision}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    if formatted == "-0":
        formatted = "0"
    return formatted


def chunk_bounds_to_str(bounds: Iterable[float], precision: int = 1) -> str:
    """Join bounds into the canonical ``minx_miny_maxx_maxy`` string."""

    return "_".join(fmt_deg(v, precision=precision) for v in bounds)


def product_prefix(feature_type: str, product: str, chunk_px: int, date_str: str) -> str:
    """Prefix under ``s3_processed_base`` for a given product/date/chunk size."""

    group, sub = _split_feature_type(feature_type)
    base = cn.datasets[group][sub]['s3_processed_base']
    return posixpath.join(base, product, f"{chunk_px}_pixels", date_str)


def build_s3_uri(feature_type: str, product: str, chunk_px: int, date_str: str, filename: str) -> str:
    prefix = product_prefix(feature_type, product, chunk_px, date_str)
    key = posixpath.join(prefix, filename)
    return f"s3://{cn.s3_bucket_name}/{key}", key


def local_product_dir(feature_type: str, product: str) -> str:
    """Return the local output directory for a product (ensure existence)."""

    group, sub = _split_feature_type(feature_type)
    base = cn.datasets[group][sub]['local_processed']
    directory = os.path.join(base, product)
    os.makedirs(directory, exist_ok=True)
    return directory


def presence_raster_name(tile_id: str, bounds: Iterable[float], feature_type: str) -> str:
    chunk_str = chunk_bounds_to_str(bounds)
    return f"{tile_id}__{chunk_str}__{feature_type}_presence.tif"


def distance_raster_name(tile_id: str, bounds: Iterable[float], feature_type: str) -> str:
    chunk_str = chunk_bounds_to_str(bounds)
    return f"{tile_id}__{chunk_str}__{feature_type}_distance.tif"


def peat_mask_path(tile_id: str) -> str:
    prefix = cn.datasets["peat"]["union_mask"]["30m"]
    key = f"{prefix}{tile_id}{PEAT_30M_PATTERN}"
    return f"/vsis3/{cn.s3_bucket_name}/{key}"


def load_mask_tile(tile_id: str):
    """Open an entire 10x10 union mask tile as an xarray.DataArray."""

    return rxr.open_rasterio(peat_mask_path(tile_id), masked=True)


def load_mask_chunk(tile_id: str, bounds_wgs84: List[float]):
    """Return (dataarray_clip, mask_bool) for a 1° chunk within ``tile_id``."""

    da_tile = load_mask_tile(tile_id)
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", da_tile.rio.crs, *bounds_wgs84, densify_pts=21)
    da_chunk = da_tile.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    da_chunk = da_chunk.load()
    if da_chunk.isnull().all():
        da_tile.close()
        return da_chunk, None

    mask_bool = (da_chunk[0].data == 1)
    da_tile.close()
    if np.all(mask_bool == 0):
        return da_chunk, None
    return da_chunk, mask_bool


def ensure_s3_client():
    if hasattr(uutil, "get_s3_client"):
        client = uutil.get_s3_client()
        if client is not None:
            return client
    return boto3.client("s3")


def upload_file(local_path: str, s3_key: str) -> str:
    client = ensure_s3_client()
    client.upload_file(local_path, cn.s3_bucket_name, s3_key)
    return f"s3://{cn.s3_bucket_name}/{s3_key}"


def tile_id_from_cell(lon_w: int, lat_s: int) -> str:
    """Return 10x10 tile id that contains the 1° cell defined by west/south edges."""

    import math

    lat_n = lat_s + 1
    tile_north = int(math.ceil(lat_n / 10.0) * 10)
    tile_west = int(math.floor(lon_w / 10.0) * 10)

    lat_code = f"{abs(tile_north):02d}{'N' if tile_north >= 0 else 'S'}"
    ew = 'E' if tile_west >= 0 else 'W'
    lon_code = f"{abs(tile_west):03d}{ew}"
    return f"{lat_code}_{lon_code}"