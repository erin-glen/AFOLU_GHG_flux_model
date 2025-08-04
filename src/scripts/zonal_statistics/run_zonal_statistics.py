# -*- coding: utf-8 -*-
"""Run organic‑soils zonal statistics.

This script converts the workflow developed in
``LULUCF_output_zonal_stats_20250613__basic_working_zonal_stats.ipynb``
into a command line utility.  It can run locally or connect to a Coiled
Dask cluster for distributed execution.  The paths and flux layers have
been updated for the organic soils model.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import dask.array as da
import dask
import fsspec
import numpy as np
import pandas as pd
import flox
import boto3
import tempfile
import posixpath
import shutil

# ── node-meaning look-ups ──────────────────────────────────────────────
from src.scripts.zonal_statistics.zonal_constants import (
    DRAINED_STATE_NODE_MEANINGS,
    BURNED_STATE_NODE_MEANINGS,
)
import s3fs
import xarray as xr
import zarr

from flox import ReindexArrayType, ReindexStrategy
from flox.xarray import xarray_reduce

# ──────────────────────────────────────────────────────────────────────────


def flox_sparse_reindex_kwargs() -> dict:
    """Return kwargs enabling sparse reindexing when supported.

    The :mod:`flox` ``ReindexStrategy`` API was introduced in version 0.10.
    When unavailable we fall back to dense aggregation while emitting a
    warning so the caller is aware that memory usage may increase.
    """

    if ReindexStrategy is None or ReindexArrayType is None:
        logging.warning(
            "Sparse re-index helpers missing – using dense aggregation; "
            "memory use will be higher. Consider installing flox >= 0.10."
        )
        return {}

    return {
        "reindex": ReindexStrategy(
            blockwise=True, array_type=ReindexArrayType.SPARSE_COO
        )
    }

# Absolute import so the script can run as `python run_zonal_statistics.py`
import src.scripts.zonal_statistics.zonal_constants as zc

# Runtime utilities
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities.universal_utilities import timestr
from src.scripts.utilities import log_utilities as lu

ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
# Base output folder (no trailing slash to simplify joining)
OUTPUT_BASE = "{root}/version_{model_version}"

DATASETS = {
    "drained_state_nodes": {
        "folder": "drained_state",
        "zarr": "drained_state_node_{interval}.zarr",
    },
    "burned_state_nodes": {
        "folder": "burned_state",
        "zarr": "burned_state_node_{interval}.zarr",
    },
    "drained_total_Mg_CO2e_pixel": {
        "folder": "drained_total_Mg_CO2e_pixel_yr",
        "zarr": "drained_total_Mg_CO2e_pixel_yr_{interval}.zarr",
    },
    "burned_total_Mg_CO2e_pixel": {
        "folder": "burned_total_Mg_CO2e_pixel",
        "zarr": "burned_total_Mg_CO2e_pixel_{interval}.zarr",
    },
}

# Template pieces used to build full paths on demand
ZARR_CACHE_PREFIX = OUTPUT_BASE + "/zarr/{run_date}/{interval}/"
FOLDER_TEMPLATE = (
    OUTPUT_BASE
    + "/{folder}/ogh_standard_model/five_year_intervals/{interval}/"
    "40000_pixels/{run_date}/"
)


def build_paths(interval: str, **kw) -> dict[str, dict[str, str]]:
    """Return folder/zarr paths for all datasets in *interval*.

    Parameters are passed via ``kw`` (typically ``root``, ``model_version`` and
    ``run_date``).  Each entry in the returned dictionary has the keys
    ``"folder"`` and ``"zarr"``.
    """

    zarr_base = ZARR_CACHE_PREFIX.format(interval=interval, **kw)
    paths: dict[str, dict[str, str]] = {}
    for name, spec in DATASETS.items():
        paths[name] = {
            "folder": FOLDER_TEMPLATE.format(
                folder=spec["folder"], interval=interval, **kw
            ),
            "zarr": zarr_base + spec["zarr"].format(interval=interval),
        }
    return paths

# Contextual layers (static)
ADM0_GTIFF_FOLDER = "s3://gfw2-data/gadm_administrative_boundaries/v4.1/v4.1.64__from_gfw-data-lake/raster/epsg-4326/10/40000/adm0/gdal-geotiff/"
ADM0_ZARR = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/GADM4_1_adm0_global/20250604/global_GADM41_adm0_20250604.zarr"
PIXEL_AREA_GTIFF_FOLDER = "s3://gfw2-data/analyses/umd_area_2013__from_gfw-data-lake/v1.10/raster/epsg-4326/10/40000/area_m/gdal-geotiff/"
PIXEL_AREA_ZARR = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/pixel_area/20250730/global_pixel_area_20250730.zarr"

# ===  END PATH MANIFEST  ======================================================

def build_output_parquet(model_version: str, years: list[int]) -> str:
    """Return default Parquet output location.

    Paths mirror those produced by the core model. ``pandas.DataFrame`` writes
    partitioned datasets to a *directory*, so the ``.parquet`` extension is
    intentionally omitted.
    """

    base = posixpath.join(ROOT, f"version_{model_version}", "zonal_stats")
    year_part = "_".join(str(y) for y in years)
    return posixpath.join(base, f"zonal_stats_{year_part}/")


def create_state_node_df(
    state_node_lookup_table_local: str, state_node_lookup_table_s3: str, sheet_name: str
) -> pd.DataFrame:
    """"Load the state node lookup table from S3 falling back to a local file.""
    pass  # Deprecated – retained for reference
    """


def list_folder_uris(base_uri: str) -> pd.Series:
    """Return GeoTIFF URIs within ``base_uri`` (recursively)."""
    fs = s3fs.S3FileSystem(anon=False)
    pattern = base_uri.rstrip("/") + "/**/*.tif"
    tif_files = [
        (f if f.startswith("s3://") else f"s3://{f}") for f in fs.glob(pattern)
    ]
    if not tif_files:
        raise FileNotFoundError(f"No GeoTIFFs found in {base_uri}")
    return pd.Series(tif_files, dtype="string")


def parse_pattern_from_uri(uri_series: pd.Series) -> str:
    """Extract the dataset name from the first URI in ``uri_series``.

    The filenames are expected to end with ``_<start>_<end>.tif`` where
    ``start`` and ``end`` are four digit years.  Examples include organic soils
    outputs such as ``drained_total_Mg_CO2e_ha`` and ``burned_total_Mg_CO2e_ha``.
    """
    if uri_series.empty:
        raise ValueError("No GeoTIFFs found – cannot derive flux‑type pattern")

    patterns = [
        r"__([A-Za-z0-9_]+)_\d{4}_\d{4}\.tif$",
        r"__([A-Za-z0-9_]+)_pixel_yr_\d{4}_\d{4}\.tif$",
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


# Crops one input to the other input's extent.
# ref is the reference dataset that is being cropped to.
# From long chat in https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/684749fe-7b30-800a-ba8b-c502377f2c3a
def safe_crop(ds, ref):
    return ds.sel(x=ref.x, y=ref.y, method="nearest")


def crop_to_bbox(ds: xr.DataArray, bbox: list[float]) -> xr.DataArray:
    """Slice ``ds`` to ``bbox`` handling coordinate order."""
    west, south, east, north = bbox
    x_slice = slice(west, east) if ds.x[0] < ds.x[-1] else slice(east, west)
    y_slice = slice(south, north) if ds.y[0] < ds.y[-1] else slice(north, south)
    return ds.sel(x=x_slice, y=y_slice)


# ───────────────────────────────────────────────────────────────────────────
# Helper – open a Zarr store, drop “band”, optional crop, apply chunking
# ───────────────────────────────────────────────────────────────────────────
def open_zarr_region(
        path: str,
        bbox: list[float] | None,
        chunk_size: int,
) -> xr.DataArray:
    """
    Return a 2‑D DataArray from *path*.

    • Works whether the Zarr was written as (y,x) **or** (band,y,x):
      the first `band` slice is kept and the dimension is dropped.

    • If *bbox* is supplied and the array has `x` and `y` coords,
      the data are cropped with `.sel(x=…, y=…)`.

    • The result is rechunked *only* along the spatial dims that exist
      (typically `"x"` and `"y"`).
    """
    mapper = fsspec.get_mapper(path, anon=False, check=False)

    # Use consolidated metadata to avoid repeated per‑chunk metadata fetches.
    with dask.annotate(label=f"open:{Path(path).stem}"):
        ds = xr.open_zarr(mapper, consolidated=True)

    # ── select a variable ────────────────────────────────────────────────
    if isinstance(ds, xr.DataArray):
        data_arr = ds
    else:  # pick first var with x & y
        vars_xy = [v for v in ds.data_vars.values() if {"x", "y"}.issubset(v.dims)]
        data_arr = vars_xy[0] if vars_xy else next(iter(ds.data_vars.values()))

    # ── drop leading “band” dimension if present ─────────────────────────
    if "band" in data_arr.dims:
        data_arr = data_arr.isel(band=0, drop=True)

    # ── corrected spatial crop ────────────────────────────────────────────
    if bbox is not None and {"x", "y"}.issubset(data_arr.dims):
        west, south, east, north = bbox
        x_ascending = data_arr.x[0] < data_arr.x[-1]
        y_ascending = data_arr.y[0] < data_arr.y[-1]

        x_slice = slice(min(west, east), max(west, east)) if x_ascending else slice(max(east, west), min(east, west))
        y_slice = slice(min(south, north), max(south, north)) if y_ascending else slice(max(north, south), min(north, south))

        data_arr = data_arr.sel(x=x_slice, y=y_slice)
    elif bbox is not None:
        logging.warning("Skipping spatial crop for %s – x/y dims not present.", path)

    # ── rechunk only existing spatial dims ───────────────────────────────
    chunk_dict = {d: chunk_size for d in ("x", "y") if d in data_arr.dims}
    if chunk_dict:  # avoid .chunk({}) for 1‑D vars
        data_arr = data_arr.chunk(chunk_dict)

    if isinstance(data_arr.data, da.core.Array):
        data_arr.data = data_arr.data.map_blocks(lambda x: x, name=f"{Path(path).stem}")

    return data_arr



def convert_to_coord_dict(flux_results: xr.DataArray, interval: str) -> dict:
    """Return flox results as a coordinate dictionary.

    ``flox`` >= 0.10 returns a :class:`sparse.COO` array which exposes
    ``coords`` and ``data`` attributes.  Older versions yield a dense
    :class:`numpy.ndarray`.  This helper supports both representations.
    """

    logging.info("   Post-processing %s : %s", interval, timestr())

    arr = flux_results.data
    dim_names = flux_results.dims

    if hasattr(arr, "coords") and hasattr(arr, "data"):
        # sparse.COO array
        indices = arr.coords
        values = arr.data
    else:
        # Dense numpy array – expand to full coordinate grid
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
                f"Interval end year {year} not supported."
                f" Valid options: {sorted(mapping)}"
            )
        pairs.append(mapping[year])
    return pairs


def create_interval_df(
        coord_dict: dict,
        flux_type_dict: dict,
        interval_end_year: int,
) -> pd.DataFrame:
    """Convert flox output to a processed dataframe."""
    df = pd.DataFrame(coord_dict)
    df["flux_type"] = df["flux_type"].replace(flux_type_dict)
    # Map node codes to human-readable meanings
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
    df["interval_end"] = interval_end_year
    df.loc[df["flux_type"].eq("area__ha"), "value"] = df["value"] / 10000
    return df


def calculate_interval_flux_densities(
        df: pd.DataFrame, contextual_layer_names: list[str]
) -> pd.DataFrame:
    """Calculate per-hectare flux densities."""
    area_df = df[df["flux_type"] == "area__ha"].copy()
    flux_df = df[df["flux_type"] != "area__ha"].copy()
    merged = pd.merge(
        flux_df,
        area_df[contextual_layer_names + ["interval_end", "value"]],
        on=contextual_layer_names,
        how="left",
        suffixes=("", "_area"),
    )
    # Guard against divide-by-zero for degenerate zones
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["value_per_ha"] = merged["value"] / merged["value_area"]
    merged.loc[merged["value_area"] == 0, "value_per_ha"] = np.nan
    new_rows = merged.copy()
    new_rows["flux_type"] = new_rows["flux_type"].astype(str) + "__CO2_per_ha"
    new_rows["value"] = new_rows["value_per_ha"]
    new_rows = new_rows.drop(
        columns=["value_area", "interval_end_area", "value_per_ha"]
    )
    return pd.concat([df, new_rows], ignore_index=True)


def ensure_zarr_exists(uri_list: pd.Series, zarr_path: str, chunk_size: int) -> None:
    """Ensure a Zarr store with consolidated metadata and valid coordinates.

    The store is created from the provided URIs when missing, and ``.zmetadata``
    is generated if absent.  Existing stores are opened to verify that ``x`` and
    ``y`` coordinates are present; if not, the store is rebuilt.
    """
    logging.debug("Ensuring Zarr store %s", zarr_path)
    fs, path = fsspec.core.url_to_fs(zarr_path)
    group_exists = fs.exists(f"{path}/.zgroup")
    metadata_exists = fs.exists(f"{path}/.zmetadata")

    has_xy = False
    if group_exists:
        logging.debug("Opening existing Zarr store %s", zarr_path)
        ds = xr.open_zarr(fs.get_mapper(path), consolidated=metadata_exists)
        has_xy = {"x", "y"}.issubset(ds.dims)

    if group_exists and metadata_exists and has_xy:
        return

    if uri_list.empty:
        raise FileNotFoundError(f"No GeoTIFFs found for {zarr_path}")

    if not group_exists or not has_xy:
        if group_exists and not has_xy:
            logging.debug(
                "Rebuilding Zarr store %s due to missing x/y coordinates", zarr_path
            )
        else:
            logging.debug("Creating new Zarr store %s", zarr_path)
        ds = make_xarray_chunks(uri_list, chunk_size)

        # ── Fix 1 ────────────────────────────────────────────────────────
        # Snap every coordinate to the nearest 1×10‑12° so that the
        # concatenated Zarr uses *exactly* the same values as the contextual
        # layers (pre‑empting sub‑nanometre rounding noise).
        for axis in ("x", "y"):
            if axis in ds.coords:
                ds = ds.assign_coords({axis: np.round(ds[axis].astype(float), 12)})
        # ----------------------------------------------------------------

        ds = ds.chunk({"x": chunk_size, "y": chunk_size})
        ds.to_zarr(zarr_path, mode="w")

    if not metadata_exists or not has_xy:
        logging.debug("Consolidating metadata for %s", zarr_path)
        zarr.convenience.consolidate_metadata(fs.get_mapper(path))


def run(args: argparse.Namespace) -> None:
    stage = "zonal_statistics"
    start_ts = uu.timestr()
    cluster, client, run_local = uu.connect_to_cluster(
        cluster_name=args.cluster_name,
        run_local=args.run_local,
    )

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

    bucket, prefix = uu.split_s3_path(args.output_parquet)
    logger.info("Output parquet path resolved to s3://%s/%s", bucket, prefix)
    existing = uu.get_existing_s3_files(bucket, prefix)
    if existing:
        logger.warning(
            "Output path %s exists – removing before write", args.output_parquet
        )
        boto3.client("s3").delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in existing]},
        )

    if args.debug:
        logger.setLevel(logging.DEBUG)
    logger.debug("Starting run with args: %s", args)
    logger.debug("Detected flox version %s", flox.__version__)

    logger.info(
        "Connected to cluster %s", cluster.name if cluster else "local-threaded"
    )
    if client:
        logger.debug("Dask client info: %s", client)
    if bbox:
        logger.debug("Using bounding box: %s", bbox)

    # Resolve manifest placeholders ------------------------------------------------
    OUTPUT_KW = dict(
        root=ROOT,
        model_version=args.model_version,
        run_date=args.run_date,
    )

    adm0_folder, adm0_zarr_name = ADM0_GTIFF_FOLDER, ADM0_ZARR
    pixel_area_folder, pixel_area_zarr_name = PIXEL_AREA_GTIFF_FOLDER, PIXEL_AREA_ZARR

    # Ensure contextual zarrs exist
    logger.debug("Checking contextual layer adm0")
    ensure_zarr_exists(list_folder_uris(adm0_folder), adm0_zarr_name, args.chunk_size)
    logger.debug("Checking contextual layer pixel_area")
    ensure_zarr_exists(
        list_folder_uris(pixel_area_folder), pixel_area_zarr_name, args.chunk_size
    )

    logger.debug("Opening contextual layers")
    # Persist static contextual layers once so repeated interval crops do not
    # rebuild Dask graphs or re-read from disk
    adm0 = open_zarr_region(adm0_zarr_name, bbox, args.chunk_size).persist()
    pixel_area = (
        open_zarr_region(pixel_area_zarr_name, bbox, args.chunk_size).persist()
    )

    adm0 = open_zarr_region(adm0_zarr_name, bbox, args.chunk_size)
    pixel_area = open_zarr_region(pixel_area_zarr_name, bbox, args.chunk_size)

    contextual_layer_names = ["drained_state_nodes", "burned_state_nodes", "gadm_adm0"]

    node_codes = zc.NODE_CODES
    gadm_adm0_ids = zc.GADM_ADM0_IDS

    interval_pairs = build_interval_pairs(args.interval_end_years)

    # Collect per‑interval results so we can write once at the end
    dfs: list[pd.DataFrame] = []
    for interval_start_year, interval_end_year in interval_pairs:
        interval = f"{interval_start_year}_{interval_end_year}"
        logger.info("Processing interval %s : %s", interval, timestr())
        logger.debug("Opening flux zarrs for interval %s", interval)

        # Build dataset folders and cache paths for this interval
        paths = build_paths(interval, **OUTPUT_KW)

        # Cache S3 listings to avoid repeated folder scans
        cached_uri_lists = {
            key: list_folder_uris(spec["folder"])
            for key, spec in paths.items()
        }

        # --- build flux‑layer caches if missing ----------------------------
        for key, spec in paths.items():
            ensure_zarr_exists(
                cached_uri_lists[key], spec["zarr"], args.chunk_size
            )
        # -------------------------------------------------------------------

        drained_total = open_zarr_region(
            paths["drained_total_Mg_CO2e_pixel"]["zarr"], bbox, args.chunk_size
        )
        burned_total = open_zarr_region(
            paths["burned_total_Mg_CO2e_pixel"]["zarr"], bbox, args.chunk_size
        )
        drained_state_nodes = open_zarr_region(
            paths["drained_state_nodes"]["zarr"], bbox, args.chunk_size
        )
        burned_state_nodes = open_zarr_region(
            paths["burned_state_nodes"]["zarr"], bbox, args.chunk_size
        )
        logger.debug("Flux layers opened for interval %s", interval)

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

        # Grab one URI from each folder to label flux_type
        flux_type_dict = {
            0: parse_pattern_from_uri(cached_uri_lists["drained_total_Mg_CO2e_pixel"]),
            1: parse_pattern_from_uri(cached_uri_lists["burned_total_Mg_CO2e_pixel"]),
            2: "area__ha",
        }

        flux_cube = xr.DataArray(
            da.stack(
                [
                    drained_total_aligned,
                    burned_total_aligned,
                    pixel_area_aligned,
                ]
            ),
            dims=("flux_type", "y", "x"),
        )
        logger.debug("Flux cube stacked for interval %s", interval)

        flux_cube, adm0_aligned, drained_state_nodes, burned_state_nodes_aligned = xr.align(
            flux_cube,
            adm0_aligned,
            drained_state_nodes,
            burned_state_nodes_aligned,
            join="override",
        )
        logger.debug("Arrays aligned for reduction for interval %s", interval)

        adm0_aligned.name = "gadm_adm0"
        drained_state_nodes.name = "drained_state_nodes"
        burned_state_nodes_aligned.name = "burned_state_nodes"

        # ─── DEBUG: Check input arrays before reduction ────────────────────
        logger.debug("drained_total_aligned mean: %s", drained_total_aligned.mean().values)
        logger.debug("burned_total_aligned mean: %s", burned_total_aligned.mean().values)
        logger.debug("pixel_area_aligned mean: %s", pixel_area_aligned.mean().values)

        logger.debug("adm0_aligned unique values: %s",
                     np.unique(adm0_aligned.values[~np.isnan(adm0_aligned.values)]))

        logger.debug("drained_state_nodes unique values: %s",
                     np.unique(drained_state_nodes.values[~np.isnan(drained_state_nodes.values)]))

        logger.debug("burned_state_nodes_aligned unique values: %s",
                     np.unique(burned_state_nodes_aligned.values[~np.isnan(burned_state_nodes_aligned.values)]))
        # ───────────────────────────────────────────────────────────────────

        logger.debug("Running flox reduce for interval %s", interval)
        with dask.annotate(label=f"reduce:{interval}"):
            flux_results = xarray_reduce(
                flux_cube,
                *(adm0_aligned, drained_state_nodes, burned_state_nodes_aligned),
                func="sum",
                expected_groups=(
                    gadm_adm0_ids,
                    node_codes,
                    node_codes,
                ),
                fill_value=np.nan,
                **flox_sparse_reindex_kwargs(),
            ).compute()
        logger.debug("Flox reduce complete for interval %s", interval)

        coord_dict = convert_to_coord_dict(flux_results, interval)
        df = create_interval_df(coord_dict, flux_type_dict, interval_end_year)
        df = calculate_interval_flux_densities(df, contextual_layer_names)
        dfs.append(df)
        logger.info("Processed interval %s", interval)

    # ╭──────────────────────────────────────────────────────────────────╮
    # │  Write all intervals locally then upload to S3 (core‑model style)│
    # ╰──────────────────────────────────────────────────────────────────╯

    full_df = pd.concat(dfs, ignore_index=True)

    # ── 1.  Guarantee Arrow‑friendly dtype for the partition column ─────────
    if "interval_end" in full_df.columns:
        full_df["interval_end"] = full_df["interval_end"].astype("int64", copy=False)

    # ── 2.  Quick sanity check before writing ───────────────────────────────
    logger.debug("Rows in concatenated DataFrame   : %s", f"{len(full_df):,}")

    local_dir = Path(tempfile.mkdtemp(prefix="zonal_stats_"))
    full_df.to_parquet(
        local_dir,
        partition_cols=["interval_end"],
        index=False,
        compression="zstd",
        engine="pyarrow",
    )

    logger.debug(
        "Files written locally            : %s",
        [
            p.relative_to(local_dir).as_posix()
            for p in local_dir.rglob('*')
            if p.is_file()
        ],
    )

    # ── 3.  Upload every file we just wrote ─────────────────────────────────
    uploaded = 0
    for f in local_dir.rglob("*"):
        if f.is_file():
            rel_key = posixpath.join(prefix, f.relative_to(local_dir).as_posix())
            assert not rel_key.startswith(
                "/"
            ), "S3 object key must never begin with '/'"
            full_s3 = f"s3://{bucket}/{rel_key}"
            logger.debug("Uploading %s to %s", f, full_s3)
            uu.upload_file_to_s3(str(f), bucket, rel_key)
            uploaded += 1

    logger.debug(
        "Uploaded %s object%s to s3://%s/%s",
        uploaded,
        "" if uploaded == 1 else "s",
        bucket,
        prefix,
    )

    shutil.rmtree(local_dir)
    logger.info(
        "Parquet write complete – partitions uploaded to %s", args.output_parquet
    )

    # ── 4.  Confirm that the dataset is now present on S3 ───────────────────
    fs = s3fs.S3FileSystem(anon=False)
    list_target = args.output_parquet.rstrip("/") + "/"
    print(f"Listing contents of: {list_target}")
    print(fs.ls(list_target, detail=True))

    if client:
        logger.debug("Closing Dask client")
        client.close()
    if cluster:
        logger.debug("Closing cluster")
        cluster.close()
    uu.stage_duration(start_ts, uu.timestr(), stage)



def main(argv=None):
    """Parse CLI args and dispatch to :func:`run`.

    When ``argv`` is empty the function runs a small local smoke test
    over a 1×1‑degree tile using default S3 paths.
    """

    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("No CLI args: running 1×1‑degree local smoke test")
        argv = [
            "--model_version",
            "test",
            "--run_date",
            "20250101",
            "--interval_end_years",
            "2020",
            "--run_local",
            "--bounding_box",
            "112",
            "-2",
            "113",
            "-1",
        ]

    parser = argparse.ArgumentParser(description="Run organic‑soils zonal statistics")
    parser.add_argument("--model_version", required=True, help="Model version string")
    parser.add_argument("--run_date", required=True, help="Model run date")
    parser.add_argument(
        "--interval_end_years",
        nargs="+",
        type=int,
        required=True,
        help="Interval end years",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4000,
        help="Tile chunk in pixels (lower -> less per-task memory)",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run_local",
        action="store_true",
        help="Run locally without Dask/Coiled",
    )
    mode.add_argument(
        "--cluster_name",
        default="zonal_stats",
        help="Name of the Coiled cluster to attach to",
    )
    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W S E N")
    parser.add_argument("--tile_ids", action="append", help="Comma separated tile IDs")

    args = parser.parse_args(argv)
    args.output_parquet = build_output_parquet(
        args.model_version, args.interval_end_years
    )

    run(args)


if __name__ == "__main__":
    main()

"""
python -m src.scripts.zonal_statistics.run_zonal_statistics \
       --interval_end_years 2024 \
       --cluster_name zonal_stats \
       --run_date 20250724 \
       --tile_ids 00N_110E \
       --model_version 0_5_0 \
       --debug

python -m src.scripts.zonal_statistics.run_zonal_statistics \
       --interval_end_years 2024 \
       --cluster_name zonal_stats \
       --run_date 20250724 \
       --model_version 0_5_0

python -m src.scripts.zonal_statistics.run_zonal_statistics \
       --interval_end_years 2005 2010 2015 2020 2024 \
       --cluster_name zonal_stats \
       --run_date 20250724 \
       --tile_ids 00N_110E \
       --model_version 0_5_0
"""

# TODO test improved safe crop (eventually)
# TODO post processing for output parquet