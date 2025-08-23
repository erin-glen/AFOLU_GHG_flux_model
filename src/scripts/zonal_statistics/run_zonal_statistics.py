# -*- coding: utf-8 -*-
"""Run organic‑soils zonal statistics.

This script converts the workflow developed in
``LULUCF_output_zonal_stats_20250613__basic_working_zonal_stats.ipynb``
into a command line utility.  It can run locally or connect to a Coiled
Dask cluster for distributed execution.  The paths and flux layers have
been updated for the organic soils model.

Key behaviors:
- Always stage Parquet locally, then upload the folder to S3 (no direct S3 writes).
- Do not compute flux densities (per‑ha). Keep flux sums and area only (area is m²→ha).
- Build Zarr caches only if missing/invalid; handle zarr v2/v3 properly.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List
from functools import lru_cache

import dask.array as da
import dask
import fsspec
import numpy as np
import pandas as pd
import flox
import boto3
import posixpath
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.fs as pafs
import shutil

from packaging.version import Version
from flox import ReindexArrayType, ReindexStrategy
from flox.xarray import xarray_reduce

# Node lookups & code sets
from src.scripts.zonal_statistics.zonal_constants import (
    DRAINED_STATE_NODE_MEANINGS,
    BURNED_STATE_NODE_MEANINGS,
    ALL_DRAINED_STATE_CODES,
    ALL_BURNED_STATE_CODES,
)
import s3fs
import xarray as xr
import zarr

# Absolute import so the script can run as `python run_zonal_statistics.py`
import src.scripts.zonal_statistics.zonal_constants as zc

# Runtime utilities
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities.universal_utilities import timestr
from src.scripts.utilities import log_utilities as lu

SPARSE_DEFAULT = True

ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
OUTPUT_BASE = "{root}/version_{model_version}"

# Dataset manifest
# - State node rasters are categorical.
# - Flux totals support *either* per-pixel or per-ha inputs; we auto-detect which exists.
DATASETS: Dict[str, Dict[str, Any]] = {
    "drained_state_nodes": {
        "folder": "drained_state",
        "zarr": "drained_state_node_{interval}.zarr",
        "var": "drained_state_nodes",
    },
    "burned_state_nodes": {
        "folder": "burned_state",
        "zarr": "burned_state_node_{interval}.zarr",
        "var": "burned_state_nodes",
    },
    "drained_total": {
        "folder_candidates": [
            "drained_total_Mg_CO2e_pixel_yr",
            "drained_total_Mg_CO2e_ha_yr",
        ],
        "zarr_by_unit": {
            "pixel": "drained_total_Mg_CO2e_pixel_yr_{interval}.zarr",
            "ha": "drained_total_Mg_CO2e_ha_yr_{interval}.zarr",
        },
        "var": "drained_total",
    },
    "burned_total": {
        "folder_candidates": [
            "burned_total_Mg_CO2e_pixel_yr",
            "burned_total_Mg_CO2e_ha_yr",
        ],
        "zarr_by_unit": {
            "pixel": "burned_total_Mg_CO2e_pixel_yr_{interval}.zarr",
            "ha": "burned_total_Mg_CO2e_ha_yr_{interval}.zarr",
        },
        "var": "burned_total",
    },
}

# Zarr cache paths (written once if missing)
ZARR_CACHE_PREFIX = OUTPUT_BASE + "/zarr/{run_date}/{run_name}/tile_{tile_pixels}/{interval}/"
COMBINED_ZARR_TEMPLATE = OUTPUT_BASE + "/zarr/{run_date}/{run_name}/tile_{tile_pixels}/{interval}/combined_interval.zarr"
FOLDER_TEMPLATE = (
        OUTPUT_BASE
        + "/{folder}/{run_name}/five_year_intervals/{interval}/"
          "{tile_pixels}_pixels/{run_date}/"
)

# Contextual layers (static)
ADM0_GTIFF_FOLDER = "s3://gfw2-data/gadm_administrative_boundaries/v4.1/v4.1.64__from_gfw-data-lake/raster/epsg-4326/10/40000/adm0/gdal-geotiff/"
ADM0_ZARR = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/GADM4_1_adm0_global/20250604/global_GADM41_adm0_20250604.zarr"
PIXEL_AREA_GTIFF_FOLDER = "s3://gfw2-data/analyses/umd_area_2013__from_gfw-data-lake/v1.10/raster/epsg-4326/10/40000/area_m/gdal-geotiff/"
PIXEL_AREA_ZARR = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/pixel_area/20250730/global_pixel_area_20250730.zarr"


# -------------------- helpers --------------------
def flox_sparse_reindex_kwargs(use_sparse: bool) -> dict:
    """Return kwargs for flox.xarray_reduce enabling sparse‑COO if available."""
    if not use_sparse:
        return {}
    if ReindexStrategy is None or ReindexArrayType is None:
        logging.warning(
            "Sparse re-index helpers missing – falling back to dense aggregation."
        )
        return {}
    return {
        "reindex": ReindexStrategy(blockwise=False, array_type=ReindexArrayType.SPARSE_COO),
        "fill_value": 0,
    }


def build_output_parquet(model_version: str, years: list[int]) -> str:
    """Return default S3 output location for staged Parquet."""
    base = posixpath.join(ROOT, f"version_{model_version}", "zonal_stats")
    year_part = "_".join(str(y) for y in years)
    return posixpath.join(base, f"zonal_stats_{year_part}/")


def list_folder_uris(base_uri: str) -> pd.Series:
    """Return GeoTIFF URIs within ``base_uri`` recursively."""
    fs = s3fs.S3FileSystem(anon=False)
    pattern = base_uri.rstrip("/") + "/**/*.tif"
    tif_files = [
        (f if f.startswith("s3://") else f"s3://{f}") for f in fs.glob(pattern)
    ]
    if not tif_files:
        raise FileNotFoundError(f"No GeoTIFFs found in {base_uri}")
    return pd.Series(tif_files, dtype="string")


def parse_pattern_from_uri(uri_series: pd.Series) -> str:
    """Extract dataset name from filenames ending with _<start>_<end>.tif.

    Supports both …__<name>_<yyyy>_<yyyy>.tif and …__<name>_(pixel_yr|ha_yr)_<yyyy>_<yyyy>.tif.
    """
    if uri_series.empty:
        raise ValueError("No GeoTIFFs found – cannot derive flux‑type pattern")

    patterns = [
        r"__([A-Za-z0-9_]+)_\d{4}_\d{4}\.tif$",
        r"__([A-Za-z0-9_]+)_(?:pixel_yr|ha_yr)_\d{4}_\d{4}\.tif$",
    ]
    matches = {
        re.search(pat, u).group(1)
        for u in uri_series
        for pat in patterns
        if re.search(pat, u)
    }
    if len(matches) != 1:
        raise ValueError(f"Mixed or unparseable flux‑type patterns: {matches}")
    return matches.pop()


def make_xarray_chunks(tile_uris: pd.Series, chunk_size: int) -> xr.Dataset:
    """Open multiple GeoTIFFs into a chunked Xarray dataset."""
    return xr.open_mfdataset(
        tile_uris.values.tolist(),
        parallel=True,
        chunks={"x": chunk_size, "y": chunk_size},
    ).squeeze()


def _first_xy_var(ds: xr.Dataset | xr.DataArray) -> xr.DataArray:
    """Return the first data variable with x/y dims."""
    if isinstance(ds, xr.DataArray):
        da = ds
    else:
        vars_xy = [v for v in ds.data_vars.values() if {"x", "y"}.issubset(v.dims)]
        da = vars_xy[0] if vars_xy else next(iter(ds.data_vars.values()))
    if "band" in da.dims:
        da = da.isel(band=0, drop=True)
    return da


def safe_crop(ds, ref):
    """Crops one input to the other input's extent (x/y)."""
    return (
        ds.sel(x=ref.x, y=ref.y, method="nearest")
        .assign_coords(x=ref.x, y=ref.y)
    )


def open_zarr_region(
        path: str,
        bbox: Optional[List[float]],
        chunk_size: int,
) -> xr.DataArray:
    """Open a 2‑D array from zarr; drop leading band; crop to bbox; rechunk."""
    s3_opts = {"anon": False}
    with dask.annotate(label=f"open:{Path(path).stem}"):
        ds = xr.open_zarr(
            path,
            consolidated=None,  # v2 auto-detect; v3 tolerated
            storage_options=s3_opts,
        )

    data_arr = _first_xy_var(ds)

    if bbox is not None and {"x", "y"}.issubset(data_arr.dims):
        west, south, east, north = bbox
        x_ascending = data_arr.x[0] < data_arr.x[-1]
        y_ascending = data_arr.y[0] < data_arr.y[-1]
        x_slice = slice(min(west, east), max(west, east)) if x_ascending else slice(max(east, west), min(east, west))
        y_slice = slice(min(south, north), max(south, north)) if y_ascending else slice(max(north, south),
                                                                                        min(north, south))
        data_arr = data_arr.sel(x=x_slice, y=y_slice)
    elif bbox is not None:
        logging.warning("Skipping spatial crop for %s – x/y dims not present.", path)

    chunk_dict = {d: chunk_size for d in ("x", "y") if d in data_arr.dims}
    if chunk_dict:
        data_arr = data_arr.chunk(chunk_dict)

    return data_arr


def crop_and_chunk(data_arr: xr.DataArray, bbox: Optional[List[float]], chunk_size: int) -> xr.DataArray:
    """Crop a DataArray to bbox (if provided) and rechunk on x/y."""
    if bbox is not None and {"x", "y"}.issubset(data_arr.dims):
        west, south, east, north = bbox
        x_ascending = data_arr.x[0] < data_arr.x[-1]
        y_ascending = data_arr.y[0] < data_arr.y[-1]
        x_slice = slice(min(west, east), max(west, east)) if x_ascending else slice(max(east, west), min(east, west))
        y_slice = slice(min(south, north), max(south, north)) if y_ascending else slice(max(north, south),
                                                                                        min(north, south))
        data_arr = data_arr.sel(x=x_slice, y=y_slice)
    chunk_dict = {d: chunk_size for d in ("x", "y") if d in data_arr.dims}
    if chunk_dict:
        data_arr = data_arr.chunk(chunk_dict)
    return data_arr


def convert_to_coord_dict(flux_results: xr.DataArray, interval: str) -> dict:
    """Return flox results as a coordinate dictionary (sparse or dense)."""
    logging.info("   Post-processing %s : %s", interval, timestr())
    arr = flux_results.data
    if isinstance(arr, da.Array):
        arr = arr.compute()
    dim_names = flux_results.dims

    if hasattr(arr, "coords") and hasattr(arr, "data"):  # sparse.COO
        indices = arr.coords
        values = arr.data
    else:  # dense np.ndarray
        grid = np.indices(arr.shape)
        indices = grid.reshape(len(arr.shape), -1)
        values = arr.ravel()

    coord_dict = {
        dim: flux_results.coords[dim].values[indices[i]]
        for i, dim in enumerate(dim_names)
    }
    coord_dict["value"] = values
    return coord_dict


def build_interval_pairs(end_years: list[int]) -> list[tuple[int, int]]:
    """Translate interval end years to (start, end) tuples."""
    mapping = {end: (start, end) for start, end in cn.five_year_inventory_periods}
    pairs = []
    for year in end_years:
        if year not in mapping:
            raise ValueError(
                f"Interval end year {year} not supported. "
                f"Valid options: {sorted(mapping)}"
            )
        pairs.append(mapping[year])
    return pairs


def create_interval_df(
        coord_dict: dict,
        flux_type_dict: dict,
        interval_start_year: int,
        interval_end_year: int,
) -> pd.DataFrame:
    """Convert flox output to a processed dataframe (no per‑ha densities)."""
    df = pd.DataFrame(coord_dict)
    df["flux_type"] = df["flux_type"].replace(flux_type_dict)

    # Map node codes to human-readable meanings if present
    if "drained_state_nodes" in df.columns:
        df["drained_state_meaning"] = (
            df["drained_state_nodes"].astype("string").str.zfill(8).map(
                DRAINED_STATE_NODE_MEANINGS
            )
        )
    if "burned_state_nodes" in df.columns:
        df["burned_state_meaning"] = (
            df["burned_state_nodes"].astype("string").str.zfill(8).map(
                BURNED_STATE_NODE_MEANINGS
            )
        )

    # Tag interval start/end and convert area m² → ha
    df["interval_start"] = interval_start_year
    df["interval_end"] = interval_end_year
    df["value_unit"] = np.where(df["flux_type"].eq("area__ha"), "ha", "Mg CO2e")
    df.loc[df["flux_type"].eq("area__ha"), "value"] = df["value"] / 10000
    return df


def ensure_zarr_exists(uri_list: pd.Series, zarr_path: str, chunk_size: int) -> None:
    """Ensure a Zarr store with valid x/y coords exists; build only if missing/invalid.

    - If group & consolidated metadata & x/y exist → return.
    - Else build from GeoTIFFs once.
    - Consolidate metadata only for zarr v2 (v3 has no consolidate step).
    """
    logging.debug("Ensuring Zarr store %s", zarr_path)
    fs, path = fsspec.core.url_to_fs(zarr_path)
    group_exists = fs.exists(f"{path}/.zgroup")
    metadata_exists = fs.exists(f"{path}/.zmetadata")
    has_xy = False

    if group_exists:
        logging.debug("Opening existing Zarr store %s", zarr_path)
        ds = xr.open_zarr(
            zarr_path,
            consolidated=None,
            storage_options=getattr(fs, "storage_options", {"anon": False}),
        )
        has_xy = {"x", "y"}.issubset(ds.dims)

    if group_exists and metadata_exists and has_xy:
        return

    if uri_list.empty:
        raise FileNotFoundError(f"No GeoTIFFs found for {zarr_path}")

    if not group_exists or not has_xy:
        if group_exists and not has_xy:
            logging.debug("Rebuilding Zarr store %s due to missing x/y", zarr_path)
        else:
            logging.debug("Creating new Zarr store %s", zarr_path)

        ds = make_xarray_chunks(uri_list, chunk_size)

        # Snap coords to 1e-12 degrees to align exactly with contextual layers
        for axis in ("x", "y"):
            if axis in ds.coords:
                ds = ds.assign_coords({axis: np.round(ds[axis].astype(float), 12)})

        ds = ds.chunk({"x": chunk_size, "y": chunk_size})
        ds.to_zarr(zarr_path, mode="w")

    if not metadata_exists or not has_xy:
        if Version(zarr.__version__).major < 3:
            logging.debug("Consolidating metadata for %s (zarr v2)", zarr_path)
            zarr.convenience.consolidate_metadata(fs.get_mapper(path))
        else:
            logging.debug("Skipping consolidate_metadata for zarr v3 store %s", zarr_path)


def log_array_summary(
        logger: logging.Logger,
        name: str,
        arr: xr.DataArray,
        *,
        categories: bool = False,
        sample_size: int = 5,
) -> None:
    logger.debug("%s → shape=%s chunks=%s dtype=%s", name, arr.shape, arr.chunks, arr.dtype)
    min_val, max_val, mean_val = dask.compute(arr.min(), arr.max(), arr.mean())
    sample = arr.data.ravel()[:sample_size].compute().tolist()
    logger.debug("%s stats: min=%s max=%s mean=%s sample=%s", name, min_val, max_val, mean_val, sample)
    if categories:
        unique_vals = np.array(da.unique(arr.data).compute())
        logger.debug("%s unique categories (%d): %s", name, unique_vals.size, unique_vals.tolist())


@lru_cache(maxsize=None)
def s3_exists(prefix: str) -> bool:
    """Return True if any object exists under the prefix."""
    fs = s3fs.S3FileSystem(anon=False)
    try:
        return any(fs.glob(prefix.rstrip("/") + "/**"))
    except FileNotFoundError:
        return False


def _resolve_flux_folder_and_unit(folder_candidates: list[str], interval: str, **fmt_kw) -> Tuple[str, str]:
    """
    Given candidate flux folder names (pixel and ha), return (existing_folder_uri, unit).
    Prefers per-ha if both exist (cheaper in practice); otherwise falls back to pixel.
    """
    ranked = []
    for name in folder_candidates:
        uri = FOLDER_TEMPLATE.format(folder=name, interval=interval, **fmt_kw)
        unit = "ha" if "_ha_" in name else "pixel"
        ranked.append((uri, unit))
    ranked.sort(key=lambda x: 0 if x[1] == "ha" else 1)  # prefer per-ha first
    for uri, unit in ranked:
        if s3_exists(uri):
            return uri, unit
    raise FileNotFoundError(f"No matching flux folders for interval {interval}: {folder_candidates}")


def build_paths(interval: str, *, tile_pixels: int, **kw) -> Dict[str, Dict[str, Any]]:
    """Return folder/zarr paths (and unit for flux) for all datasets in *interval*."""
    zarr_base = ZARR_CACHE_PREFIX.format(interval=interval, tile_pixels=tile_pixels, **kw)
    paths: Dict[str, Dict[str, Any]] = {}
    kw2 = dict(kw)
    kw2["tile_pixels"] = tile_pixels

    for name, spec in DATASETS.items():
        if "folder_candidates" in spec:
            folder_uri, unit = _resolve_flux_folder_and_unit(spec["folder_candidates"], interval, **kw2)
            zarr_name = spec["zarr_by_unit"][unit].format(interval=interval)
            paths[name] = {"folder": folder_uri, "zarr": zarr_base + zarr_name, "unit": unit, "var": spec["var"]}
        else:
            folder_uri = FOLDER_TEMPLATE.format(folder=spec["folder"], interval=interval, **kw2)
            paths[name] = {"folder": folder_uri, "zarr": zarr_base + spec["zarr"].format(interval=interval),
                           "unit": None, "var": spec["var"]}
    return paths


def ensure_combined_interval_zarr(paths: dict, chunk_size: int, combined_path: str) -> None:
    """
    Build a single Zarr per interval with all required variables if missing/invalid.
    Reduces open/metadata overhead when many variables are read together.
    """
    fs, inner = fsspec.core.url_to_fs(combined_path)
    group_exists = fs.exists(f"{inner}/.zgroup")
    metadata_exists = fs.exists(f"{inner}/.zmetadata")
    if group_exists and metadata_exists:
        try:
            xr.open_zarr(combined_path, consolidated=None,
                         storage_options=getattr(fs, "storage_options", {"anon": False}))
            return
        except Exception:
            logging.debug("Rebuilding combined Zarr due to open error: %s", combined_path)

    ds_vars = {}
    for key in ("drained_state_nodes", "burned_state_nodes", "drained_total", "burned_total"):
        uris = list_folder_uris(paths[key]["folder"])
        ds_in = make_xarray_chunks(uris, chunk_size)
        da_in = _first_xy_var(ds_in)
        # Snap coords to 1e-12 deg to align with contextual layers
        for axis in ("x", "y"):
            if axis in da_in.coords:
                da_in = da_in.assign_coords({axis: np.round(da_in[axis].astype(float), 12)})
        ds_vars[paths[key]["var"]] = da_in

    ds = xr.Dataset(ds_vars).chunk({"x": chunk_size, "y": chunk_size})
    ds.to_zarr(combined_path, mode="w")
    if Version(zarr.__version__).major < 3:
        zarr.convenience.consolidate_metadata(fs.get_mapper(inner))


# -------------------- main driver --------------------
def run(args: argparse.Namespace) -> None:
    stage = "zonal_statistics"
    start_ts = uu.timestr()
    cluster, client, run_local = uu.connect_to_cluster(
        cluster_name=args.cluster_name,
        run_local=args.run_local,
    )

    # ROI as bbox or union of tiles
    bbox = None
    if args.bounding_box:
        bbox = [float(x) for x in args.bounding_box]
    elif args.tile_ids:
        tiles: list[str] = []
        for item in args.tile_ids:
            tiles.extend(t.strip() for t in item.split(",") if t.strip())
        if tiles:
            bounds = [uu.get_10x10_tile_bounds(t) for t in tiles]
            west = min(b[0] for b in bounds)
            south = min(b[1] for b in bounds)
            east = max(b[2] for b in bounds)
            north = max(b[3] for b in bounds)
            bbox = [west, south, east, north]

    logger, _ = lu.populate_main_log_header(
        bounding_box=bbox,
        use_shapefile=False,
        client=client,
        cluster=cluster,
        log_note="Organic soils zonal statistics",
        run_local=run_local,
        model_type="organic_soils",
        stage=stage,
    )

    # Output path handling (we still *remove* existing S3 prefix to avoid mixups)
    args.output_parquet = build_output_parquet(args.model_version, args.interval_end_years)
    bucket, prefix = uu.split_s3_path(args.output_parquet)
    logger.info("Output parquet path resolved to s3://%s/%s", bucket, prefix)
    existing = uu.get_existing_s3_files(bucket, prefix)
    if existing:
        logger.warning("Output path %s exists – removing before write", args.output_parquet)
        boto3.client("s3").delete_objects(
            Bucket=bucket, Delete={"Objects": [{"Key": k} for k in existing]}
        )

    if args.debug:
        logger.setLevel(logging.DEBUG)
    logger.debug("Starting run with args: %s", args)
    logger.debug("Detected flox version %s", flox.__version__)

    logger.info("Connected to cluster %s", cluster.name if cluster else "local-threaded")
    if client:
        logger.debug("Dask client info: %s", client)
    if bbox:
        logger.debug("Using bounding box: %s", bbox)

    # Resolve manifest placeholders
    OUTPUT_KW = dict(root=ROOT, model_version=args.model_version, run_date=args.run_date, run_name=args.run_name)

    adm0_folder, adm0_zarr_name = ADM0_GTIFF_FOLDER, ADM0_ZARR
    pixel_area_folder, pixel_area_zarr_name = PIXEL_AREA_GTIFF_FOLDER, PIXEL_AREA_ZARR

    # Ensure contextual zarrs exist
    logger.debug("Checking contextual layer adm0")
    ensure_zarr_exists(list_folder_uris(adm0_folder), adm0_zarr_name, args.chunk_size)
    logger.debug("Checking contextual layer pixel_area")
    ensure_zarr_exists(list_folder_uris(pixel_area_folder), pixel_area_zarr_name, args.chunk_size)

    logger.debug("Opening contextual layers")
    adm0 = open_zarr_region(adm0_zarr_name, bbox, args.chunk_size).astype("uint32")
    pixel_area = open_zarr_region(pixel_area_zarr_name, bbox, args.chunk_size).persist()

    # Expected groups
    gadm_adm0_ids = zc.GADM_ADM0_IDS
    drained_codes_arr = np.array(sorted({0, *map(int, ALL_DRAINED_STATE_CODES)}), dtype=np.uint32)
    burned_codes_arr = np.array(sorted({0, *map(int, ALL_BURNED_STATE_CODES)}), dtype=np.uint32)

    # Local staging dirs
    local_arrow = pafs.LocalFileSystem()
    base_dir_root = Path(args.local_output).expanduser().resolve()
    base_dir_root.mkdir(parents=True, exist_ok=True)
    base_dir_drained = base_dir_root / "drained"
    base_dir_burned = base_dir_root / "burned"

    logger.debug("Writing Parquet to local staging directory %s", base_dir_root)

    interval_pairs = build_interval_pairs(args.interval_end_years)
    for interval_start_year, interval_end_year in interval_pairs:
        interval = f"{interval_start_year}_{interval_end_year}"
        logger.info("Processing interval %s : %s", interval, timestr())
        logger.debug("Opening flux zarrs for interval %s", interval)

        # Build dataset folders & cache paths
        paths = build_paths(interval, tile_pixels=args.tile_pixels, **OUTPUT_KW)

        # Cache S3 listings and ensure zarrs once per dataset
        cached_uri_lists = {key: list_folder_uris(spec["folder"]) for key, spec in paths.items()}
        if args.combine_zarr == "interval":
            combined_path = COMBINED_ZARR_TEMPLATE.format(interval=interval, tile_pixels=args.tile_pixels, **OUTPUT_KW)
            ensure_combined_interval_zarr(paths, args.chunk_size, combined_path)
            s3_opts = {"anon": False}
            ds_combined = xr.open_zarr(combined_path, consolidated=None, storage_options=s3_opts)
            drained_total = crop_and_chunk(ds_combined[paths["drained_total"]["var"]], bbox, args.chunk_size)
            burned_total = crop_and_chunk(ds_combined[paths["burned_total"]["var"]], bbox, args.chunk_size)
            drained_state_nodes = crop_and_chunk(ds_combined[paths["drained_state_nodes"]["var"]].astype("uint32"),
                                                 bbox, args.chunk_size)
            burned_state_nodes = crop_and_chunk(ds_combined[paths["burned_state_nodes"]["var"]].astype("uint32"), bbox,
                                                args.chunk_size)
        else:
            for key, spec in paths.items():
                ensure_zarr_exists(cached_uri_lists[key], spec["zarr"], args.chunk_size)
            # Open flux/state layers
            drained_total = open_zarr_region(paths["drained_total"]["zarr"], bbox, args.chunk_size)
            burned_total = open_zarr_region(paths["burned_total"]["zarr"], bbox, args.chunk_size)
            drained_state_nodes = open_zarr_region(paths["drained_state_nodes"]["zarr"], bbox, args.chunk_size).astype(
                "uint32")
            burned_state_nodes = open_zarr_region(paths["burned_state_nodes"]["zarr"], bbox, args.chunk_size).astype(
                "uint32")
        logger.debug("Flux layers opened for interval %s", interval)

        # Align everything to drained_state_nodes grid
        reference = drained_state_nodes
        try:
            adm0_aligned = safe_crop(adm0, reference)
            pixel_area_aligned = safe_crop(pixel_area, reference)
            drained_total_aligned = safe_crop(drained_total, reference)
            burned_total_aligned = safe_crop(burned_total, reference)
            burned_state_nodes_aligned = safe_crop(burned_state_nodes, reference)
        except ValueError as exc:
            logger.error("%s", exc)
            raise
        logger.debug("Datasets aligned for interval %s", interval)

        adm0_aligned.name = "gadm_adm0"
        drained_state_nodes.name = "drained_state_nodes"
        burned_state_nodes_aligned.name = "burned_state_nodes"

        # Convert per-ha flux inputs to per-pixel totals lazily
        if paths["drained_total"]["unit"] == "ha":
            drained_total_aligned = drained_total_aligned * (pixel_area_aligned / 10000.0)
        if paths["burned_total"]["unit"] == "ha":
            burned_total_aligned = burned_total_aligned * (pixel_area_aligned / 10000.0)

        if args.debug:
            for n, arr in {
                "drained_total": drained_total_aligned,
                "burned_total": burned_total_aligned,
                "pixel_area": pixel_area_aligned,
                "adm0": adm0_aligned,
                "drained_state_nodes": drained_state_nodes,
                "burned_state_nodes": burned_state_nodes_aligned,
            }.items():
                log_array_summary(
                    logger, n, arr, categories=n in {"adm0", "drained_state_nodes", "burned_state_nodes"}
                )

        # -------- Drained aggregation (sum) --------
        with dask.annotate(label=f"reduce:drained:{interval}"):
            cube_d = xr.concat([drained_total_aligned, pixel_area_aligned], dim="flux_type").assign_coords(
                flux_type=("flux_type", [0, 2])
            )
            res_d = xarray_reduce(
                cube_d,
                adm0_aligned,
                drained_state_nodes,
                func="sum",
                expected_groups=(gadm_adm0_ids, drained_codes_arr),
                **flox_sparse_reindex_kwargs(not args.no_sparse),
            ).compute()
        dict_d = convert_to_coord_dict(res_d, interval)
        ft_dict_d = {
            0: parse_pattern_from_uri(cached_uri_lists["drained_total"]),
            2: "area__ha",
        }
        df_d = create_interval_df(dict_d, ft_dict_d, interval_start_year, interval_end_year)

        ds.write_dataset(
            pa.Table.from_pandas(df_d, preserve_index=False),
            base_dir=str(base_dir_drained),
            filesystem=local_arrow,  # ← LOCAL ONLY
            partitioning=["interval_end"],
            format="parquet",
            existing_data_behavior="delete_matching",
        )
        logger.info("Wrote %s rows (drained) for %s", len(df_d), interval)

        # -------- Burned aggregation (sum) --------
        with dask.annotate(label=f"reduce:burned:{interval}"):
            cube_b = xr.concat([burned_total_aligned, pixel_area_aligned], dim="flux_type").assign_coords(
                flux_type=("flux_type", [1, 2])
            )
            res_b = xarray_reduce(
                cube_b,
                adm0_aligned,
                burned_state_nodes_aligned,
                func="sum",
                expected_groups=(gadm_adm0_ids, burned_codes_arr),
                **flox_sparse_reindex_kwargs(not args.no_sparse),
            ).compute()
        dict_b = convert_to_coord_dict(res_b, interval)
        ft_dict_b = {
            1: parse_pattern_from_uri(cached_uri_lists["burned_total"]),
            2: "area__ha",
        }
        df_b = create_interval_df(dict_b, ft_dict_b, interval_start_year, interval_end_year)

        ds.write_dataset(
            pa.Table.from_pandas(df_b, preserve_index=False),
            base_dir=str(base_dir_burned),
            filesystem=local_arrow,  # ← LOCAL ONLY
            partitioning=["interval_end"],
            format="parquet",
            existing_data_behavior="delete_matching",
        )
        logger.info("Wrote %s rows (burned)  for %s", len(df_b), interval)

    # ---------- Upload the staged local directory to S3 ----------
    # Upload subfolders directly to avoid an extra '/zonal_stats/' level in S3.
    logger.debug("Uploading staged data to %s", args.output_parquet)
    fs_s3 = s3fs.S3FileSystem()
    for sub in ("drained", "burned"):
        local_sub = str(base_dir_root / sub)
        dest_sub = posixpath.join(args.output_parquet.rstrip("/"), sub)
        fs_s3.put(local_sub, dest_sub, recursive=True)

    # Optional cleanup
    if not args.keep_local:
        logger.debug("Removing local staging directory %s", base_dir_root)
        shutil.rmtree(base_dir_root, ignore_errors=True)

    if args.debug:
        fs = s3fs.S3FileSystem(anon=False)
        list_target = args.output_parquet.rstrip("/") + "/"
        logger.debug("Listing S3 contents: %s", list_target)
        try:
            logger.debug(fs.ls(list_target, detail=True))
        except FileNotFoundError:
            logger.debug("Destination folder does not exist (yet).")

    if client:
        logger.debug("Closing Dask client")
        client.close()
    if cluster:
        logger.debug("Closing cluster")
        cluster.close()
    uu.stage_duration(start_ts, uu.timestr(), stage)


def main(argv=None):
    """Parse CLI args and dispatch to :func:`run`.

    If no args are provided, runs a 1×1‑degree local smoke test.
    """
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("No CLI args: running 1×1‑degree local smoke test")
        argv = [
            "--model_version", "test",
            "--run_date", "20250101",
            "--interval_end_years", "2020",
            "--run_local",
            "--bounding_box", "112", "-2", "113", "-1",
            "--tile_pixels", "4000",
            "--chunk_size", "10000",
            "--combine_zarr", "interval",
        ]

    parser = argparse.ArgumentParser(description="Run organic‑soils zonal statistics")
    parser.add_argument("--model_version", required=True, help="Model version string")
    parser.add_argument("--run_date", required=True, help="Model run date")
    parser.add_argument("--interval_end_years", nargs="+", type=int, required=True, help="Interval end years")
    parser.add_argument("--chunk_size", type=int, default=4000, help="Tile chunk in pixels")
    parser.add_argument("--tile_pixels", type=int, default=40000,
                        help="Input tile size in pixels: 40000 for 10×10°, 4000 for 1×1°")
    parser.add_argument("--local_output", default="/tmp/zonal_stats", help="Local staging directory for Parquet output")
    parser.add_argument("--keep-local", action="store_true", help="Keep staged files after upload")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    parser.add_argument("--no_sparse", action="store_true", default=not SPARSE_DEFAULT,
                        help="Disable sparse-COO output (dense fallback).")
    parser.add_argument("--run_name", default="ogh_standard_model", help="Model run name")
    parser.add_argument("--combine_zarr", choices=["none", "interval"], default="none",
                        help="Build and use a single Zarr per interval to reduce overhead.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run_local", action="store_true", help="Run locally without Dask/Coiled")
    mode.add_argument("--cluster_name", default="zonal_stats", help="Name of the Coiled cluster to attach to")
    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W S E N")
    parser.add_argument("--tile_ids", action="append", help="Comma separated tile IDs")

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()

"""
Example:
python -m src.scripts.zonal_statistics.run_zonal_statistics \
  --interval_end_years 2005 2010 2015 2020 2024 \
  --cluster_name zonal_stats \
  --run_date 20250820 \
  --model_version 0_6_5 \
  --tile_pixels 4000 \
  --chunk_size 10000 \
  --combine_zarr none

python -m src.scripts.zonal_statistics.run_zonal_statistics \
  --interval_end_years 2024 \
  --cluster_name zonal_stats \
  --run_date 20250820 \
  --model_version 0_6_5 \
  --tile_pixels 4000 \
  --chunk_size 10000 \
  --combine_zarr none

python -m src.scripts.zonal_statistics.run_zonal_statistics \
  --interval_end_years 2024 \
  --cluster_name zonal_stats \
  --run_date 20250820 \
  --model_version 0_6_5 \
  --tile_pixels 4000 \
  --chunk_size 10000 \
  --combine_zarr none \
  --tile_id 00N_110E

"""