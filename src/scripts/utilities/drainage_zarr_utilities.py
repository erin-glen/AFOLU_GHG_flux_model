from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import fsspec
import numpy as np
import zarr

from src.scripts.utilities import constants_and_names as cn


GLOBAL_WEST = -180.0
GLOBAL_NORTH = 90.0


def pixel_resolution() -> float:
    return 10.0 / cn.full_raster_dims


def global_grid_shape() -> tuple[int, int]:
    res = pixel_resolution()
    width = int(round(360.0 / res))
    height = int(round(180.0 / res))
    return height, width


@lru_cache(maxsize=1)
def global_coords() -> tuple[np.ndarray, np.ndarray, float]:
    res = pixel_resolution()
    height, width = global_grid_shape()
    x = GLOBAL_WEST + res * (0.5 + np.arange(width, dtype=np.float32))
    y = GLOBAL_NORTH - res * (0.5 + np.arange(height, dtype=np.float32))
    return x, y, res


def bounds_to_indices(bounds: Sequence[float]) -> tuple[int, int, int, int]:
    west, south, east, north = bounds
    _, _, res = global_coords()
    x0 = int(round((west - GLOBAL_WEST) / res))
    x1 = int(round((east - GLOBAL_WEST) / res))
    y0 = int(round((GLOBAL_NORTH - north) / res))
    y1 = int(round((GLOBAL_NORTH - south) / res))
    return y0, y1, x0, x1


def create_mega_zarr_path(
    base_path: str,
    chunk_size_pixels: int,
    interval_type: str,
    model_type: str,
    run_date: str,
    logger,
) -> str:
    path = "/".join(
        part.strip("/")
        for part in (
            base_path,
            model_type,
            interval_type,
            f"{chunk_size_pixels}_pixels",
            run_date,
            "mega.zarr",
        )
        if part
    )
    logger.info("mega-zarr path: %s", path)
    return path


def initialize_global_mega_zarr(
    zarr_path: str,
    outputs_to_zarr: Iterable[str],
    years: Sequence[int],
    chunk_size_pixels: int,
    interval_type: str,
    logger,
) -> None:
    logger.info("Initializing drainage mega-zarr: %s", zarr_path)
    store = fsspec.get_mapper(zarr_path)
    group = zarr.open_group(store=store, mode="w")

    outputs = list(outputs_to_zarr)
    x, y, _ = global_coords()
    group.create_dataset(
        "x",
        data=x,
        shape=x.shape,
        dtype=x.dtype,
        chunks=x.shape,
        overwrite=True,
    )
    group["x"].attrs["_ARRAY_DIMENSIONS"] = ["x"]
    group.create_dataset(
        "y",
        data=y,
        shape=y.shape,
        dtype=y.dtype,
        chunks=y.shape,
        overwrite=True,
    )
    group["y"].attrs["_ARRAY_DIMENSIONS"] = ["y"]
    year_data = np.asarray(years, dtype=np.int32)
    group.create_dataset(
        "year",
        data=year_data,
        shape=year_data.shape,
        dtype=year_data.dtype,
        chunks=year_data.shape,
        overwrite=True,
    )
    group["year"].attrs["_ARRAY_DIMENSIONS"] = ["year"]

    height, width = global_grid_shape()
    for name in outputs:
        dtype = cn.drainage_output_dtypes.get(name, "float32")
        arr = group.create_dataset(
            name,
            shape=(len(years), height, width),
            chunks=(1, chunk_size_pixels, chunk_size_pixels),
            dtype=dtype,
            fill_value=0,
            overwrite=True,
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = ["year", "y", "x"]

    group.attrs.update(
        {
            "model": "organic_soils_drainage",
            "interval_type": interval_type,
            "chunk_size_pixels": chunk_size_pixels,
        }
    )
    logger.info("Initialized mega-zarr with %d datasets", len(outputs))


def populate_mega_zarr(
    zarr_path: str,
    outputs: dict[str, np.ndarray],
    outputs_to_zarr: Iterable[str],
    bounds: Sequence[float],
    year_index: int,
) -> None:
    store = fsspec.get_mapper(zarr_path)
    group = zarr.open_group(store=store, mode="r+")
    y0, y1, x0, x1 = bounds_to_indices(bounds)
    for name in outputs_to_zarr:
        if name not in outputs:
            continue
        group[name][year_index, y0:y1, x0:x1] = outputs[name]


def full_model_year_index(interval_type: str) -> list[int]:
    if interval_type == cn.intervals_annual:
        last_year = cn.five_year_inventory_periods[-1][1]
        return list(range(cn.annual_land_cover_start_year, last_year + 1))
    return [end for _, end in cn.five_year_inventory_periods]


def open_zarr_window(
    zarr_path: str,
    dataset: str,
    bounds: Sequence[float],
    year_index: int,
) -> np.ndarray:
    store = fsspec.get_mapper(zarr_path)
    group = zarr.open_group(store=store, mode="r")
    y0, y1, x0, x1 = bounds_to_indices(bounds)
    return group[dataset][year_index, y0:y1, x0:x1]