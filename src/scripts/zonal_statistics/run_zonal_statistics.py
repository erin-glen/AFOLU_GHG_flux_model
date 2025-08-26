# -*- coding: utf-8 -*-
"""Run organic-soils zonal statistics (robust; per-interval upload; run_name in paths).

Converts the notebook workflow into a CLI utility that can run locally or
attach to a Coiled Dask cluster. Designed for robustness and reproducibility.

Key guarantees in this version:
- Skips unreadable/corrupt TIFFs when building Zarrs (does not fail the run).
- Never parses flux names from TIFF filenames; uses explicit labels.
- Zarr caches under .../zarr/{run_name}/{run_date}/{interval}/ (no tile-size suffix).
- Only build Zarrs if missing/invalid; supports both Zarr v2 and v3.
- Stage Parquet locally per full interval (no Hive partitioning) and upload the
  finished folder for each interval to S3 immediately under .../{interval}/(drained|burned)/.
- No per-ha densities; sums only; area is m² → ha in post-process.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List
from functools import lru_cache

import dask
import dask.array as da
import fsspec
import numpy as np
import pandas as pd
import posixpath
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import s3fs
import shutil
import xarray as xr
import zarr
import rasterio

import flox
from flox import ReindexArrayType, ReindexStrategy
from flox.xarray import xarray_reduce
from packaging.version import Version

# Absolute imports so this can run via `python -m ...run_zonal_statistics`
import src.scripts.zonal_statistics.zonal_constants as zc
from src.scripts.zonal_statistics.zonal_constants import (
    DRAINED_STATE_NODE_MEANINGS,
    BURNED_STATE_NODE_MEANINGS,
    ALL_DRAINED_STATE_CODES,
    ALL_BURNED_STATE_CODES,
)
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities.universal_utilities import timestr
from src.scripts.utilities import log_utilities as lu

# --------------------------------- config ---------------------------------
SPARSE_DEFAULT = True

ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
OUTPUT_BASE = "{root}/version_{model_version}"

# Dataset manifest
# - State node rasters are categorical.
# - Flux totals may exist as per-pixel or per-ha inputs (we auto-detect).
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

# Zarr caches (NO tile-size suffix in path) — include run_name
ZARR_CACHE_PREFIX = OUTPUT_BASE + "/zarr/{run_name}/{run_date}/{interval}/"
COMBINED_ZARR_TEMPLATE = ZARR_CACHE_PREFIX + "combined_interval.zarr"

# Input folders (still depend on tile size to locate inputs)
FOLDER_TEMPLATE = (
    OUTPUT_BASE
    + "/{folder}/{run_name}/five_year_intervals/{interval}/{tile_pixels}_pixels/{run_date}/"
)

# Contextual layers (static)
ADM0_GTIFF_FOLDER = (
    "s3://gfw2-data/gadm_administrative_boundaries/v4.1/"
    "v4.1.64__from_gfw-data-lake/raster/epsg-4326/10/40000/adm0/gdal-geotiff/"
)
ADM0_ZARR = (
    "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/"
    "GADM4_1_adm0_global/20250604/global_GADM41_adm0_20250604.zarr"
)
PIXEL_AREA_GTIFF_FOLDER = (
    "s3://gfw2-data/analyses/umd_area_2013__from_gfw-data-lake/"
    "v1.10/raster/epsg-4326/10/40000/area_m/gdal-geotiff/"
)
PIXEL_AREA_ZARR = (
    "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/"
    "pixel_area/20250730/global_pixel_area_20250730.zarr"
)

# ------------------------------ small utils ------------------------------
def flox_sparse_reindex_kwargs(use_sparse: bool) -> dict:
    """Return kwargs for flox.xarray_reduce enabling sparse-COO if available."""
    if not use_sparse:
        return {}
    if ReindexStrategy is None or ReindexArrayType is None:
        logging.warning("Sparse re-index helpers missing – using dense aggregation.")
        return {}
    return {
        "reindex": ReindexStrategy(blockwise=False, array_type=ReindexArrayType.SPARSE_COO),
        "fill_value": 0,
    }


def build_output_parquet(model_version: str, run_name: str, run_date: str, interval: str) -> str:
    """Return the S3 prefix for zonal stats, aligned to Zarr layout.

    Zarr caches live under:
        .../zarr/{run_name}/{run_date}/{interval}/

    We mirror that for Parquet:
        .../zonal_stats/{run_name}/{run_date}/{interval}/
    """
    base = posixpath.join(
        ROOT,
        f"version_{model_version}",
        "zonal_stats",
        run_name,
        run_date,
        interval,
    )
    return base.rstrip("/") + "/"


def list_folder_uris(base_uri: str) -> pd.Series:
    """Return GeoTIFF URIs within base_uri recursively; error if none."""
    fs = s3fs.S3FileSystem(anon=False)
    pattern = base_uri.rstrip("/") + "/**/*.tif"
    tif_files = [(f if f.startswith("s3://") else f"s3://{f}") for f in fs.glob(pattern)]
    if not tif_files:
        raise FileNotFoundError(f"No GeoTIFFs found in {base_uri}")
    return pd.Series(tif_files, dtype="string")


def make_xarray_chunks(tile_uris: pd.Series, chunk_size: int) -> xr.Dataset:
    """Open multiple GeoTIFFs into a chunked Xarray dataset."""
    return xr.open_mfdataset(
        tile_uris.values.tolist(),
        parallel=True,
        chunks={"x": chunk_size, "y": chunk_size},
    ).squeeze()


def _filter_valid_tiffs(uri_series: pd.Series) -> pd.Series:
    """
    Return only TIFFs that rasterio can open (width/height > 0, >=1 band).
    Skips zero-byte, truncated, or otherwise unreadable tiles.
    """
    if uri_series.empty:
        return uri_series
    valid: List[str] = []
    bad = 0
    for u in uri_series.tolist():
        try:
            with rasterio.Env():
                with rasterio.open(u) as src:
                    if src.count >= 1 and src.width > 0 and src.height > 0:
                        valid.append(u)
                    else:
                        bad += 1
        except Exception as e:
            logging.warning("Skipping unreadable TIFF: %s (%s)", u, e)
            bad += 1
    if bad:
        logging.warning("Filtered out %d invalid TIFF(s); proceeding with %d valid.", bad, len(valid))
    return pd.Series(valid, dtype="string")


def _first_xy_var(ds_or_da: xr.Dataset | xr.DataArray) -> xr.DataArray:
    if isinstance(ds_or_da, xr.DataArray):
        da_ = ds_or_da
    else:
        vars_xy = [v for v in ds_or_da.data_vars.values() if {"x", "y"}.issubset(v.dims)]
        da_ = vars_xy[0] if vars_xy else next(iter(ds_or_da.data_vars.values()))
    if "band" in da_.dims:
        da_ = da_.isel(band=0, drop=True)
    return da_


def safe_crop(ds, ref):
    """Crop ds to ref's x/y extent (nearest), then force coords equal to ref."""
    return ds.sel(x=ref.x, y=ref.y, method="nearest").assign_coords(x=ref.x, y=ref.y)


def open_zarr_region(path: str, bbox: Optional[List[float]], chunk_size: int) -> xr.DataArray:
    """Open 2-D array from Zarr; drop band; crop bbox; rechunk."""
    s3_opts = {"anon": False}
    with dask.annotate(label=f"open:{Path(path).stem}"):
        dsx = xr.open_zarr(path, consolidated=None, storage_options=s3_opts)

    data_arr = _first_xy_var(dsx)

    if bbox is not None and {"x", "y"}.issubset(data_arr.dims):
        west, south, east, north = bbox
        x_asc = bool(data_arr.x[0] < data_arr.x[-1])
        y_asc = bool(data_arr.y[0] < data_arr.y[-1])
        x_slice = slice(min(west, east), max(west, east)) if x_asc else slice(max(east, west), min(east, west))
        y_slice = slice(min(south, north), max(south, north)) if y_asc else slice(max(north, south), min(north, south))
        data_arr = data_arr.sel(x=x_slice, y=y_slice)

    chunk_dict = {d: chunk_size for d in ("x", "y") if d in data_arr.dims}
    if chunk_dict:
        data_arr = data_arr.chunk(chunk_dict)

    return data_arr


def crop_and_chunk(data_arr: xr.DataArray, bbox: Optional[List[float]], chunk_size: int) -> xr.DataArray:
    if bbox is not None and {"x", "y"}.issubset(data_arr.dims):
        west, south, east, north = bbox
        x_asc = bool(data_arr.x[0] < data_arr.x[-1])
        y_asc = bool(data_arr.y[0] < data_arr.y[-1])
        x_slice = slice(min(west, east), max(west, east)) if x_asc else slice(max(east, west), min(east, west))
        y_slice = slice(min(south, north), max(south, north)) if y_asc else slice(max(north, south), min(north, south))
        data_arr = data_arr.sel(x=x_slice, y=y_slice)
    chunk_dict = {d: chunk_size for d in ("x", "y") if d in data_arr.dims}
    if chunk_dict:
        data_arr = data_arr.chunk(chunk_dict)
    return data_arr


def convert_to_coord_dict(flox_result: xr.DataArray, interval: str) -> dict:
    logging.info("   Post-processing %s : %s", interval, timestr())
    arr = flox_result.data
    if isinstance(arr, da.Array):
        arr = arr.compute()
    dim_names = flox_result.dims

    # sparse.COO -> coords/data; dense -> indices.ravel
    if hasattr(arr, "coords") and hasattr(arr, "data"):
        indices = arr.coords
        values = arr.data
    else:
        grid = np.indices(arr.shape)
        indices = grid.reshape(len(arr.shape), -1)
        values = arr.ravel()

    coord_dict = {dim: flox_result.coords[dim].values[indices[i]] for i, dim in enumerate(dim_names)}
    coord_dict["value"] = values
    return coord_dict


def build_interval_pairs(end_years: List[int]) -> List[Tuple[int, int]]:
    mapping = {end: (start, end) for start, end in cn.five_year_inventory_periods}
    pairs = []
    for year in end_years:
        if year not in mapping:
            raise ValueError(
                f"Interval end year {year} not supported. Valid options: {sorted(mapping)}"
            )
        pairs.append(mapping[year])
    return pairs


def create_interval_df(coord_dict: dict, flux_type_dict: dict, interval_end_year: int) -> pd.DataFrame:
    """Convert flox output to processed dataframe (no per-ha densities)."""
    df = pd.DataFrame(coord_dict)
    df["flux_type"] = df["flux_type"].replace(flux_type_dict)

    # Optional human-readable node meanings
    if "drained_state_nodes" in df.columns:
        df["drained_state_meaning"] = (
            df["drained_state_nodes"].astype("string").str.zfill(8).map(DRAINED_STATE_NODE_MEANINGS)
        )
    if "burned_state_nodes" in df.columns:
        df["burned_state_meaning"] = (
            df["burned_state_nodes"].astype("string").str.zfill(8).map(BURNED_STATE_NODE_MEANINGS)
        )

    # Tag interval end; convert area m² → ha
    df["interval_end"] = interval_end_year  # kept for downstream analysis
    df.loc[df["flux_type"].eq("area__ha"), "value"] = df["value"] / 10000.0
    return df


def _zarr_store_exists(fs, path: str) -> Tuple[bool, bool, bool]:
    """Return (has_zgroup, has_zmetadata, has_zarr_json)."""
    return fs.exists(f"{path}/.zgroup"), fs.exists(f"{path}/.zmetadata"), fs.exists(f"{path}/zarr.json")


def ensure_zarr_exists(uri_list: pd.Series, zarr_path: str, chunk_size: int) -> None:
    """Ensure a Zarr store with x/y coords exists; build only if missing/invalid."""
    logging.debug("Ensuring Zarr store %s", zarr_path)
    fs, inner = fsspec.core.url_to_fs(zarr_path)
    has_zgroup, has_zmeta, has_v3json = _zarr_store_exists(fs, inner)
    has_xy = False

    if has_zgroup or has_v3json:
        logging.debug("Opening existing Zarr store %s", zarr_path)
        dsx = xr.open_zarr(zarr_path, consolidated=None, storage_options=getattr(fs, "storage_options", {"anon": False}))
        has_xy = {"x", "y"}.issubset(dsx.dims)

    if (has_v3json or (has_zgroup and has_zmeta)) and has_xy:
        return

    if uri_list.empty:
        raise FileNotFoundError(f"No GeoTIFFs found for {zarr_path}")

    # Build (or rebuild) from GeoTIFFs; if open_mfdataset fails, filter unreadable TIFFs and retry
    try:
        dsx = make_xarray_chunks(uri_list, chunk_size)
    except Exception as e:
        logging.warning(
            "Initial Zarr build failed opening %d TIFF(s) for %s: %s. "
            "Filtering invalid tiles and retrying.",
            len(uri_list), zarr_path, e
        )
        filtered = _filter_valid_tiffs(uri_list)
        if filtered.empty:
            raise FileNotFoundError(f"No valid GeoTIFFs remain after filtering for {zarr_path}")
        dsx = make_xarray_chunks(filtered, chunk_size)

    for axis in ("x", "y"):
        if axis in dsx.coords:
            dsx = dsx.assign_coords({axis: np.round(dsx[axis].astype(float), 12)})

    dsx = dsx.chunk({"x": chunk_size, "y": chunk_size})
    dsx.to_zarr(zarr_path, mode="w")

    # Consolidate metadata for zarr v2 only
    if Version(zarr.__version__).major < 3:
        has_zgroup, has_zmeta, _ = _zarr_store_exists(fs, inner)
        if has_zgroup and not has_zmeta:
            logging.debug("Consolidating metadata for %s (zarr v2)", zarr_path)
            zarr.convenience.consolidate_metadata(fs.get_mapper(inner))


def ensure_combined_interval_zarr(paths: dict, chunk_size: int, combined_path: str) -> None:
    """Build a single Zarr per interval with all variables if missing/invalid."""
    fs, inner = fsspec.core.url_to_fs(combined_path)
    has_zgroup, has_zmeta, has_v3json = _zarr_store_exists(fs, inner)
    if (has_v3json or (has_zgroup and has_zmeta)):
        try:
            xr.open_zarr(combined_path, consolidated=None,
                         storage_options=getattr(fs, "storage_options", {"anon": False}))
            return
        except Exception:
            logging.debug("Rebuilding combined Zarr due to open error: %s", combined_path)

    ds_vars = {}
    for key in ("drained_state_nodes", "burned_state_nodes", "drained_total", "burned_total"):
        uris = list_folder_uris(paths[key]["folder"])
        try:
            ds_in = make_xarray_chunks(uris, chunk_size)
        except Exception as e:
            logging.warning(
                "Combined Zarr: failed opening %d TIFF(s) for %s: %s. Filtering invalid tiles.",
                len(uris), paths[key]["folder"], e
            )
            filtered = _filter_valid_tiffs(uris)
            if filtered.empty:
                logging.error("Combined Zarr: no valid TIFFs remain for %s; skipping.", paths[key]["folder"])
                continue
            ds_in = make_xarray_chunks(filtered, chunk_size)
        da_in = _first_xy_var(ds_in)
        for axis in ("x", "y"):
            if axis in da_in.coords:
                da_in = da_in.assign_coords({axis: np.round(da_in[axis].astype(float), 12)})
        ds_vars[paths[key]["var"]] = da_in

    if not ds_vars:
        raise FileNotFoundError(f"Combined Zarr: no variables could be built for {combined_path}")

    dsx = xr.Dataset(ds_vars).chunk({"x": chunk_size, "y": chunk_size})
    dsx.to_zarr(combined_path, mode="w")
    if Version(zarr.__version__).major < 3:
        zarr.convenience.consolidate_metadata(fs.get_mapper(inner))


def log_array_summary(logger: logging.Logger, name: str, arr: xr.DataArray, *, categories: bool = False) -> None:
    logger.debug("%s → shape=%s chunks=%s dtype=%s", name, arr.shape, arr.chunks, arr.dtype)
    min_val, max_val, mean_val = dask.compute(arr.min(), arr.max(), arr.mean())
    logger.debug("%s stats: min=%s max=%s mean=%s", name, min_val, max_val, mean_val)
    if categories:
        unique_vals = np.array(da.unique(arr.data).compute())
        logger.debug("%s unique categories (%d): %s", name, unique_vals.size, unique_vals.tolist())


@lru_cache(maxsize=None)
def s3_exists(prefix: str) -> bool:
    fs = s3fs.S3FileSystem(anon=False)
    try:
        return any(fs.glob(prefix.rstrip("/") + "/**"))
    except FileNotFoundError:
        return False


def _resolve_flux_folder_and_unit(folder_candidates: List[str], interval: str, **fmt_kw) -> Tuple[str, str]:
    """Return (existing_folder_uri, unit) for flux; prefer per-ha, else per-pixel."""
    ranked: List[Tuple[str, str]] = []
    for name in folder_candidates:
        uri = FOLDER_TEMPLATE.format(folder=name, interval=interval, **fmt_kw)
        unit = "ha" if "_ha_" in name else "pixel"
        ranked.append((uri, unit))
    ranked.sort(key=lambda x: 0 if x[1] == "ha" else 1)
    for uri, unit in ranked:
        if s3_exists(uri):
            return uri, unit
    raise FileNotFoundError(f"No matching flux folders for interval {interval}: {folder_candidates}")


def build_paths(interval: str, *, tile_pixels: int, **kw) -> Dict[str, Dict[str, Any]]:
    """Return folder/zarr paths (and unit for flux) for all datasets in *interval*."""
    zarr_base = ZARR_CACHE_PREFIX.format(interval=interval, **kw)
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


# ------------------------------ upload utils ------------------------------
def _upload_partition_dir(fs_s3: s3fs.S3FileSystem, local_dir: Path, dest_prefix: str, logger: logging.Logger) -> int:
    """Upload all files from local_dir into dest_prefix (S3), preserving names. Returns uploaded file count."""
    if not local_dir.exists():
        logger.warning("Local subfolder missing, skipping upload: %s", local_dir)
        return 0

    # Clear any previous content to avoid mixing runs
    try:
        fs_s3.rm(dest_prefix.rstrip("/") + "/", recursive=True)
    except Exception:
        pass

    uploaded = 0
    for p in local_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir)
            remote_path = posixpath.join(dest_prefix.rstrip("/"), *rel.parts)
            fs_s3.put(str(p), remote_path)
            uploaded += 1
    return uploaded


# --------------------------------- driver ---------------------------------
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
        tiles: List[str] = []
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

    if args.debug:
        logger.setLevel(logging.DEBUG)
    logger.debug("Starting run with args: %s", args)
    logger.debug("Detected flox version %s", flox.__version__)
    logger.info("Connected to cluster %s", cluster.name if cluster else "local-threaded")
    if client:
        logger.debug("Dask client info: %s", client)
    if bbox:
        logger.debug("Using bounding box: %s", bbox)

    # Placeholders for path formatting
    OUTPUT_KW = dict(root=ROOT, model_version=args.model_version, run_date=args.run_date, run_name=args.run_name)

    # Ensure contextual Zarrs exist
    logger.debug("Checking contextual layer adm0")
    ensure_zarr_exists(list_folder_uris(ADM0_GTIFF_FOLDER), ADM0_ZARR, args.chunk_size)
    logger.debug("Checking contextual layer pixel_area")
    ensure_zarr_exists(list_folder_uris(PIXEL_AREA_GTIFF_FOLDER), PIXEL_AREA_ZARR, args.chunk_size)

    logger.debug("Opening contextual layers")
    adm0 = open_zarr_region(ADM0_ZARR, bbox, args.chunk_size).astype("uint32")
    pixel_area = open_zarr_region(PIXEL_AREA_ZARR, bbox, args.chunk_size).persist()

    # Expected groups
    gadm_adm0_ids = zc.GADM_ADM0_IDS
    drained_codes_arr = np.array(sorted({0, *map(int, ALL_DRAINED_STATE_CODES)}), dtype=np.uint32)
    burned_codes_arr = np.array(sorted({0, *map(int, ALL_BURNED_STATE_CODES)}), dtype=np.uint32)

    # Local staging (always write locally first), no Hive partitioning
    local_arrow = pafs.LocalFileSystem()
    base_dir_root = Path(args.local_output).expanduser().resolve()
    base_dir_root.mkdir(parents=True, exist_ok=True)
    base_dir_drained = base_dir_root / "drained"
    base_dir_burned = base_dir_root / "burned"

    logger.debug("Writing Parquet to local staging directory %s", base_dir_root)

    fs_s3 = s3fs.S3FileSystem(anon=False)

    # Process each interval
    interval_pairs = build_interval_pairs(args.interval_end_years)
    for interval_start_year, interval_end_year in interval_pairs:
        interval = f"{interval_start_year}_{interval_end_year}"
        logger.info("Processing interval %s : %s", interval, timestr())

        # Build dataset folders & cache paths with the requested tile size
        paths = build_paths(interval, tile_pixels=args.tile_pixels, **OUTPUT_KW)

        logger.info(
            "Interval %s inputs: drained_total=%s (unit=%s), burned_total=%s (unit=%s), drained_state=%s, burned_state=%s",
            interval,
            paths["drained_total"]["folder"], paths["drained_total"]["unit"],
            paths["burned_total"]["folder"], paths["burned_total"]["unit"],
            paths["drained_state_nodes"]["folder"], paths["burned_state_nodes"]["folder"],
        )

        if args.combine_zarr == "interval":
            combined_path = COMBINED_ZARR_TEMPLATE.format(interval=interval, **OUTPUT_KW)
            ensure_combined_interval_zarr(paths, args.chunk_size, combined_path)
            ds_combined = xr.open_zarr(combined_path, consolidated=None, storage_options={"anon": False})
            drained_total = crop_and_chunk(ds_combined[paths["drained_total"]["var"]], bbox, args.chunk_size)
            burned_total = crop_and_chunk(ds_combined[paths["burned_total"]["var"]], bbox, args.chunk_size)
            drained_state_nodes = crop_and_chunk(ds_combined[paths["drained_state_nodes"]["var"]].astype("uint32"),
                                                 bbox, args.chunk_size)
            burned_state_nodes = crop_and_chunk(ds_combined[paths["burned_state_nodes"]["var"]].astype("uint32"),
                                                bbox, args.chunk_size)
        else:
            cached_uri_lists = {key: list_folder_uris(spec["folder"]) for key, spec in paths.items()}
            for key, spec in paths.items():
                ensure_zarr_exists(cached_uri_lists[key], spec["zarr"], args.chunk_size)
            drained_total = open_zarr_region(paths["drained_total"]["zarr"], bbox, args.chunk_size)
            burned_total = open_zarr_region(paths["burned_total"]["zarr"], bbox, args.chunk_size)
            drained_state_nodes = open_zarr_region(paths["drained_state_nodes"]["zarr"], bbox, args.chunk_size).astype("uint32")
            burned_state_nodes = open_zarr_region(paths["burned_state_nodes"]["zarr"], bbox, args.chunk_size).astype("uint32")

        # Align everything to drained_state_nodes grid
        reference = drained_state_nodes
        adm0_aligned = safe_crop(adm0, reference)
        pixel_area_aligned = safe_crop(pixel_area, reference)
        drained_total_aligned = safe_crop(drained_total, reference)
        burned_total_aligned = safe_crop(burned_total, reference)
        burned_state_nodes_aligned = safe_crop(burned_state_nodes, reference)

        adm0_aligned.name = "gadm_adm0"
        drained_state_nodes.name = "drained_state_nodes"
        burned_state_nodes_aligned.name = "burned_state_nodes"

        # Convert per-ha flux inputs to per-pixel totals lazily (only when needed)
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
                log_array_summary(logger, n, arr, categories=n in {"adm0", "drained_state_nodes", "burned_state_nodes"})

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
        ft_dict_d = {0: "drained_total_Mg_CO2e", 2: "area__ha"}
        df_d = create_interval_df(dict_d, ft_dict_d, interval_end_year)

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
        ft_dict_b = {1: "burned_total_Mg_CO2e", 2: "area__ha"}
        df_b = create_interval_df(dict_b, ft_dict_b, interval_end_year)

        # -------- Local per-interval staging (no Hive) --------
        local_d = (base_dir_drained / interval)
        local_b = (base_dir_burned / interval)
        for pth in (local_d, local_b):
            if pth.exists():
                shutil.rmtree(pth, ignore_errors=True)
            pth.mkdir(parents=True, exist_ok=True)

        ds.write_dataset(
            pa.Table.from_pandas(df_d, preserve_index=False),
            base_dir=str(local_d),
            filesystem=local_arrow,
            format="parquet",
            existing_data_behavior="overwrite_or_ignore",
        )
        logger.info("Wrote %s rows (drained) for %s → %s", len(df_d), interval, local_d)

        ds.write_dataset(
            pa.Table.from_pandas(df_b, preserve_index=False),
            base_dir=str(local_b),
            filesystem=local_arrow,
            format="parquet",
            existing_data_behavior="overwrite_or_ignore",
        )
        logger.info("Wrote %s rows (burned) for %s → %s", len(df_b), interval, local_b)

        # -------- Upload to S3 under full-interval directory --------
        dest_root = build_output_parquet(args.model_version, args.run_name, args.run_date, interval)
        dest_d = posixpath.join(dest_root.rstrip("/"), "drained")
        dest_b = posixpath.join(dest_root.rstrip("/"), "burned")
        logger.info("Uploading interval %s → %s{drained,burned}", interval, dest_root)

        n_d = _upload_partition_dir(fs_s3, local_d, dest_d, logger)
        n_b = _upload_partition_dir(fs_s3, local_b, dest_b, logger)

        # Verify uploads
        uploaded_d = fs_s3.glob(posixpath.join(dest_d, "*.parquet"))
        uploaded_b = fs_s3.glob(posixpath.join(dest_b, "*.parquet"))
        logger.info("Drained upload: %d file(s) at %s", len(uploaded_d), posixpath.join(dest_d, "*.parquet"))
        logger.info("Burned  upload: %d file(s) at %s", len(uploaded_b), posixpath.join(dest_b, "*.parquet"))

        if not args.keep_local:
            shutil.rmtree(local_d, ignore_errors=True)
            shutil.rmtree(local_b, ignore_errors=True)

    # Optional cleanup after all intervals
    if not args.keep_local:
        logger.debug("Removing local staging directory %s", base_dir_root)
        shutil.rmtree(base_dir_root, ignore_errors=True)

    # Teardown
    if client:
        logger.debug("Closing Dask client")
        client.close()
    if cluster:
        logger.debug("Closing cluster")
        cluster.close()
    uu.stage_duration(start_ts, uu.timestr(), stage)


def main(argv=None):
    """Parse CLI args and dispatch to :func:`run`.

    If no args are provided, runs a 1×1-degree local smoke test.
    """
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("No CLI args: running 1×1-degree local smoke test")
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

    parser = argparse.ArgumentParser(description="Run organic-soils zonal statistics")
    parser.add_argument("--model_version", required=True, help="Model version string")
    parser.add_argument("--run_date", required=True, help="Run date (YYYYMMDD)")
    parser.add_argument("--interval_end_years", nargs="+", type=int, required=True, help="Interval end years")
    parser.add_argument("--chunk_size", type=int, default=4000, help="Dask chunk size in pixels")
    parser.add_argument(
        "--tile_pixels",
        type=int,
        default=4000,  # ← default to 4000 since inputs are always 1×1°
        help="Input tile size in pixels (4000 for 1×1°, 40000 for 10×10°).",
    )
    parser.add_argument("--local_output", default="/tmp/zonal_stats", help="Local staging directory for Parquet output")
    parser.add_argument("--keep-local", action="store_true", help="Keep staged files after upload")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    parser.add_argument("--no_sparse", action="store_true", default=not SPARSE_DEFAULT,
                        help="Disable sparse-COO output (dense fallback).")
    parser.add_argument("--run_name", default="ogh_standard_model", help="Model run name")
    parser.add_argument("--combine_zarr", choices=["none", "interval"], default="none",
                        help="Build and use a single Zarr per interval to reduce open/metadata overhead.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run_local", action="store_true", help="Run locally without Dask/Coiled")
    mode.add_argument("--cluster_name", default="zonal_stats", help="Name of the Coiled cluster to attach to")
    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W S E N")
    parser.add_argument("--tile_ids", action="append", help="Comma separated 10×10 tile IDs (e.g., 00N_110E)")

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()

"""
Examples:

# Single interval (robust; skips corrupt tiles if any). Uses combined Zarr per interval.
python -m src.scripts.zonal_statistics.run_zonal_statistics \
  --interval_end_years 2005 \
  --cluster_name zonal_stats \
  --run_date 20250825 \
  --model_version 0_7_0 \
  --run_name ogh_standard_model \
  --tile_pixels 4000 \
  --chunk_size 10000 \
  --combine_zarr interval

# Multiple intervals, default (separate Zarrs, immediate per-interval uploads)
python -m src.scripts.zonal_statistics.run_zonal_statistics \
  --interval_end_years 2005 2010 2015 2020 2024 \
  --cluster_name zonal_stats \
  --run_date 20250825 \
  --model_version 0_7_0 \
  --run_name ogh_standard_model \
  --tile_pixels 4000 \
  --chunk_size 10000 \
  --combine_zarr none

# Multiple intervals filtered by tile IDs
python -m src.scripts.zonal_statistics.run_zonal_statistics \
  --interval_end_years 2005 2010 \
  --cluster_name zonal_stats \
  --run_date 20250825 \
  --model_version 0_7_0 \
  --run_name ogh_standard_model \
  --tile_pixels 4000 \
  --chunk_size 10000 \
  --combine_zarr none \
  --tile_ids 00N_110E

# Bounding box run (local smoke test)
python -m src.scripts.zonal_statistics.run_zonal_statistics \
  --interval_end_years 2020 \
  --run_local \
  --bounding_box 112 -2 113 -1 \
  --run_date 20250101 \
  --model_version test \
  --run_name smoke \
  --tile_pixels 4000 \
  --chunk_size 10000 \
  --combine_zarr interval
"""
