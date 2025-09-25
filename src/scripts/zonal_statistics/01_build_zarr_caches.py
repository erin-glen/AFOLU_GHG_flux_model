# -*- coding: utf-8 -*-
"""
Build per-variable Zarr caches for organic-soils zonal statistics (per-pixel only).

What this does
--------------
- For each requested dataset and interval:
  * Lists source GeoTIFFs on S3 via boto3 (no s3fs globbing)
  * Opens TIFFs with xarray+rasterio (parallel=True by default; disable with --no-open-parallel)
  * Writes a single Zarr per variable under .../zarr/{run_name}/{run_date}/{interval}/
- Per-pixel only for flux totals (no per-ha support in this builder).
- **Symmetric chunking** by default: {"y": chunk_size, "x": chunk_size}.
- Skips rebuild if a valid Zarr already exists with x/y dims.

This script does NOT run any reductions or write Parquet.

Examples
--------
# Build all datasets for one interval (cluster)
python -m src.scripts.zonal_statistics.01_build_zarr_caches \
  --interval_end_years 2024 \
  --cluster_name zonal_stats \
  --run_date 20250923 \
  --model_version 0_8_0 \
  --run_name ogh_sensitivity_1km \
  --tile_pixels 4000 \
  --chunk_size 10000

# Build a subset for multiple intervals (cluster), explicitly disabling parallel open
python -m src.scripts.zonal_statistics.01_build_zarr_caches \
  --interval_end_years 2015 2020 2024 \
  --datasets drained_total burned_total \
  --no-open-parallel \
  --cluster_name zonal_stats \
  --run_date 20250923 \
  --model_version 0_8_0 \
  --run_name ogh_sensitivity_1km \
  --tile_pixels 4000 \
  --chunk_size 10000

# Local smoke build (global write but on local scheduler)
python -m src.scripts.zonal_statistics.01_build_zarr_caches \
  --interval_end_years 2020 \
  --run_local \
  --run_date 20250101 \
  --model_version test \
  --run_name smoke \
  --tile_pixels 4000 \
  --chunk_size 10000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from functools import lru_cache

import fsspec
import numpy as np
import pandas as pd
import posixpath
import s3fs  # required for fsspec S3 mapping when writing Zarr
import xarray as xr
import zarr
import rasterio
from packaging.version import Version

from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu

# ------------------------------- config --------------------------------
ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
OUTPUT_BASE = "{root}/version_{model_version}"

# Per-pixel only dataset manifest
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
        "folder": "drained_total_Mg_CO2e_pixel_yr",
        "zarr": "drained_total_Mg_CO2e_pixel_yr_{interval}.zarr",
        "var": "drained_total",
    },
    "burned_total": {
        "folder": "burned_total_Mg_CO2e_pixel_yr",
        "zarr": "burned_total_Mg_CO2e_pixel_yr_{interval}.zarr",
        "var": "burned_total",
    },
}

ZARR_CACHE_PREFIX = OUTPUT_BASE + "/zarr/{run_name}/{run_date}/{interval}/"
FOLDER_TEMPLATE = (
    OUTPUT_BASE
    + "/{folder}/{run_name}/five_year_intervals/{interval}/{tile_pixels}_pixels/{run_date}/"
)

# ------------------------------ utils ----------------------------------
@lru_cache(maxsize=None)
def s3_exists(prefix: str) -> bool:
    """Fast existence check using uu.list_raster_full_paths_in_s3_folder_and_count (boto3)."""
    try:
        files, n = uu.list_raster_full_paths_in_s3_folder_and_count(prefix)
        return n > 0
    except Exception:
        return False


def list_folder_uris(base_uri: str) -> pd.Series:
    """List TIFFs with boto3 (through uu.* helper). No pre-validation reads."""
    files, n = uu.list_raster_full_paths_in_s3_folder_and_count(base_uri)
    if n == 0:
        raise FileNotFoundError(f"No GeoTIFFs found in {base_uri}")
    return pd.Series(files, dtype="string")


def make_xarray_chunks(
    tile_uris: pd.Series,
    chunk_size: int,
    *,
    open_parallel: bool,
) -> xr.Dataset:
    """Open multiple GeoTIFFs into a chunked Xarray dataset."""
    return xr.open_mfdataset(
        tile_uris.values.tolist(),
        engine="rasterio",
        combine="by_coords",
        parallel=bool(open_parallel),
        chunks={"x": chunk_size, "y": chunk_size},
    ).squeeze()


def _filter_valid_tiffs(uri_series: pd.Series) -> pd.Series:
    """
    Recovery-only filter: try to open each TIFF once and keep ones that succeed.
    (Used *only if* open_mfdataset fails; this is not a pre-open step.)
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
        except Exception:
            bad += 1
    if bad:
        logging.warning("Filtered %d TIFF(s) after open_mfdataset failure; proceeding with %d valid.", bad, len(valid))
    return pd.Series(valid, dtype="string")


def _zarr_store_exists(fs, path: str) -> Tuple[bool, bool, bool]:
    return fs.exists(f"{path}/.zgroup"), fs.exists(f"{path}/.zmetadata"), fs.exists(f"{path}/zarr.json")


def _first_xy_var(ds_or_da: xr.Dataset | xr.DataArray) -> xr.DataArray:
    if isinstance(ds_or_da, xr.DataArray):
        da_ = ds_or_da
    else:
        vars_xy = [v for v in ds_or_da.data_vars.values() if {"x", "y"}.issubset(v.dims)]
        da_ = vars_xy[0] if vars_xy else next(iter(ds_or_da.data_vars.values()))
    if "band" in da_.dims:
        da_ = da_.isel(band=0, drop=True)
    return da_


def _clear_store_if_exists(zarr_path: str, logger: logging.Logger) -> None:
    """Remove an existing Zarr store recursively (for invalid/partial stores)."""
    fs, inner = fsspec.core.url_to_fs(zarr_path)
    if fs.exists(inner):
        logger.warning("Removing existing Zarr store before rebuild: %s", zarr_path)
        try:
            fs.rm(inner.rstrip("/") + "/", recursive=True)
        except Exception as e:
            logger.warning("Failed to remove %s; proceeding with overwrite may fail (%s).", zarr_path, e)


def ensure_zarr_exists(
    uri_list: pd.Series,
    zarr_path: str,
    chunk_size: int,
    logger: logging.Logger,
    *,
    open_parallel: bool = True,
) -> None:
    """Ensure a Zarr with x/y dims exists at zarr_path; build if missing/invalid."""
    logger.debug("Ensuring Zarr store %s", zarr_path)
    fs, inner = fsspec.core.url_to_fs(zarr_path)
    has_zgroup, has_zmeta, has_v3json = _zarr_store_exists(fs, inner)

    if (has_v3json or (has_zgroup and has_zmeta)):
        try:
            dsx = xr.open_zarr(zarr_path, consolidated=None, storage_options=getattr(fs, "storage_options", {"anon": False}))
            if {"x", "y"}.issubset(dsx.dims):
                logger.info("Zarr exists and is valid: %s", zarr_path)
                return
            else:
                logger.warning("Existing Zarr missing x/y dims; rebuilding: %s", zarr_path)
                _clear_store_if_exists(zarr_path, logger)
        except Exception as e:
            logger.warning("Existing Zarr failed to open (%s); rebuilding: %s", e, zarr_path)
            _clear_store_if_exists(zarr_path, logger)

    if uri_list.empty:
        raise FileNotFoundError(f"No GeoTIFFs found for {zarr_path}")

    # Fast path: trust listing; open all TIFFs
    try:
        dsx = make_xarray_chunks(uri_list, chunk_size, open_parallel=open_parallel)
    except Exception as e:
        logger.warning("open_mfdataset failed (%s). Filtering unreadable tiles and retrying.", e)
        filtered = _filter_valid_tiffs(uri_list)
        if filtered.empty:
            raise FileNotFoundError(f"No valid GeoTIFFs remain after filtering for {zarr_path}")
        dsx = make_xarray_chunks(filtered, chunk_size, open_parallel=open_parallel)

    # Use first x/y variable; round coords to avoid float stitching mismatches
    da_in = _first_xy_var(dsx)
    for axis in ("x", "y"):
        if axis in da_in.coords:
            da_in = da_in.assign_coords({axis: np.round(da_in[axis].astype(float), 12)})

    # Write a single Zarr in one shot, like the older version
    ds_out = da_in.to_dataset(name=da_in.name or "variable").chunk({"x": chunk_size, "y": chunk_size})
    ds_out.to_zarr(zarr_path, mode="w", consolidated=None)

    # Consolidate metadata for zarr v2 only
    if Version(zarr.__version__).major < 3:
        has_zgroup, has_zmeta, _ = _zarr_store_exists(fs, inner)
        if has_zgroup and not has_zmeta:
            zarr.convenience.consolidate_metadata(fs.get_mapper(inner))

    logger.info("Built Zarr: %s", zarr_path)


def build_paths(interval: str, *, tile_pixels: int, **kw) -> Dict[str, Dict[str, Any]]:
    zarr_base = ZARR_CACHE_PREFIX.format(interval=interval, **kw)
    paths: Dict[str, Dict[str, Any]] = {}
    kw2 = dict(kw)
    kw2["tile_pixels"] = tile_pixels
    for name, spec in DATASETS.items():
        folder_uri = FOLDER_TEMPLATE.format(folder=spec["folder"], interval=interval, **kw2)
        paths[name] = {
            "folder": folder_uri,
            "zarr": zarr_base + spec["zarr"].format(interval=interval),
            "var": spec["var"],
        }
    return paths

# -------------------------------- driver --------------------------------
def run(args: argparse.Namespace) -> None:
    stage = "zarr_build"
    start_ts = uu.timestr()
    cluster, client, run_local = uu.connect_to_cluster(
        cluster_name=args.cluster_name, run_local=args.run_local,
    )

    logger, _ = lu.populate_main_log_header(
        bounding_box=None, use_shapefile=False, client=client, cluster=cluster,
        log_note="Organic soils Zarr build", run_local=run_local,
        model_type="organic_soils", stage=stage,
    )

    if args.debug:
        logger.setLevel(logging.DEBUG)
    logger.debug("Starting Zarr build with args: %s", args)

    OUTPUT_KW = dict(root=ROOT, model_version=args.model_version, run_date=args.run_date, run_name=args.run_name)
    intervals = [f"{s}_{e}" for (s, e) in args.interval_pairs]

    for interval in intervals:
        logger.info("Interval %s", interval)
        paths = build_paths(interval, tile_pixels=args.tile_pixels, **OUTPUT_KW)

        for ds_name in args.datasets:
            if ds_name not in DATASETS:
                logger.warning("Unknown dataset '%s' requested; skipping.", ds_name)
                continue
            folder = paths[ds_name]["folder"]
            zpath = paths[ds_name]["zarr"]
            logger.info("Building %s → %s", folder, zpath)
            uris = list_folder_uris(folder)
            logger.info("  • %d TIFF(s) found", len(uris))
            ensure_zarr_exists(
                uris, zpath, args.chunk_size, logger,
                open_parallel=args.open_parallel
            )

    if client:
        client.close()
    if cluster:
        cluster.close()
    uu.stage_duration(start_ts, uu.timestr(), stage)


def _parse_interval_pairs(end_years: List[int]) -> List[Tuple[int, int]]:
    from src.scripts.utilities import constants_and_names as cn
    mapping = {end: (start, end) for start, end in cn.five_year_inventory_periods}
    pairs = []
    for y in end_years:
        if y not in mapping:
            raise ValueError(f"Interval end year {y} not supported. Valid: {sorted(mapping)}")
        pairs.append(mapping[y])
    return pairs


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Build per-variable Zarr caches (per-pixel only)")
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--run_date", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--interval_end_years", nargs="+", type=int, required=True)
    parser.add_argument("--tile_pixels", type=int, default=4000,
                        help="Input tile size in pixels (4000 for 1×1°, 40000 for 10×10°).")
    parser.add_argument("--chunk_size", type=int, default=10000,
                        help="Chunk size for x/y; default symmetric chunking.")
    parser.add_argument("--datasets", nargs="+",
                        default=["drained_total", "burned_total", "drained_state_nodes", "burned_state_nodes"],
                        choices=list(DATASETS.keys()))
    # Default to parallel; allow disabling with --no-open-parallel
    parser.add_argument("--open-parallel", dest="open_parallel", action="store_true",
                        help="Use open_mfdataset(parallel=True). Default: True.")
    parser.add_argument("--no-open-parallel", dest="open_parallel", action="store_false",
                        help="Disable parallel opening (open_mfdataset parallel=False).")
    parser.set_defaults(open_parallel=True)
    parser.add_argument("--debug", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run_local", action="store_true")
    mode.add_argument("--cluster_name", default="zarr_build")
    args = parser.parse_args(argv)

    args.interval_pairs = _parse_interval_pairs(args.interval_end_years)
    run(args)


if __name__ == "__main__":
    main()
