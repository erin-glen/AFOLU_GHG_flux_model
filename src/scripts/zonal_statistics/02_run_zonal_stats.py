# -*- coding: utf-8 -*-
"""Run organic-soils zonal statistics (per-pixel only; robust alignment; per-interval upload).

What this does
--------------
- Assumes per-variable Zarr caches already exist under:
  .../zarr/{run_name}/{run_date}/{interval}/
- Opens Zarrs for adm0, pixel_area, drained_total, burned_total, and state nodes
- Aligns all variables to the pixel_area grid using nearest-with-tolerance
- Runs flox-based grouped sums (adm0 × state) with a mask (adm0 > 0)
- Writes local Parquet per interval and uploads to:
  .../zonal_stats/{run_name}/{run_date}/{interval}/{drained|burned}/

Important
---------
- **Per-pixel only** for flux totals; no per-ha fallback or conversion exists here.
- If a required Zarr is missing, the script fails with a clear message.

Examples
--------
# Single interval (cluster)
python -m src.scripts.zonal_statistics.02_run_zonal_stats \
  --interval_end_years 2024 \
  --cluster_name zonal_stats \
  --run_date 20250923 \
  --model_version 0_8_0 \
  --run_name ogh_sensitivity_1km \
  --chunk_size 10000

# Multiple intervals (cluster)
python -m src.scripts.zonal_statistics.02_run_zonal_stats \
  --interval_end_years 2010 2015 2020 2024 \
  --cluster_name zonal_stats \
  --run_date 20250825 \
  --model_version 0_7_0 \
  --run_name ogh_standard_model \
  --chunk_size 10000

# Local smoke test (1×1° ROI)
python -m src.scripts.zonal_statistics.02_run_zonal_stats \
  --interval_end_years 2020 \
  --run_local \
  --bounding_box 112 -2 113 -1 \
  --run_date 20250101 \
  --model_version test \
  --run_name smoke \
  --chunk_size 10000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List

import boto3
import dask
import dask.array as da
import numpy as np
import pandas as pd
import posixpath
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import s3fs
import xarray as xr

import flox
from flox import ReindexArrayType, ReindexStrategy
from flox.xarray import xarray_reduce

from src.scripts.zonal_statistics import zonal_constants as zc
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities.universal_utilities import timestr
from src.scripts.utilities import log_utilities as lu

# ------------------------------- config --------------------------------
SPARSE_DEFAULT = True
ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
OUTPUT_BASE = "{root}/version_{model_version}"

DATASETS: Dict[str, Dict[str, Any]] = {
    "drained_state_nodes": {
        "zarr": "drained_state_node_{interval}.zarr",
        "var": "drained_state_nodes",
    },
    "burned_state_nodes": {
        "zarr": "burned_state_node_{interval}.zarr",
        "var": "burned_state_nodes",
    },
    "drained_total": {
        "zarr": "drained_total_Mg_CO2e_pixel_yr_{interval}.zarr",
        "var": "drained_total",
    },
    "burned_total": {
        "zarr": "burned_total_Mg_CO2e_pixel_yr_{interval}.zarr",
        "var": "burned_total",
    },
}

# Zarr caches (NO tile-size suffix in path)
ZARR_CACHE_PREFIX = OUTPUT_BASE + "/zarr/{run_name}/{run_date}/{interval}/"

# Contextual layers (Zarrs built separately)
ADM0_ZARR = (
    "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/"
    "GADM4_1_adm0_global/20250604/global_GADM41_adm0_20250604.zarr"
)
PIXEL_AREA_ZARR = (
    "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/"
    "pixel_area/20250730/global_pixel_area_20250730.zarr"
)

# ------------------------------ utils ----------------------------------
def flox_sparse_reindex_kwargs(use_sparse: bool) -> dict:
    if not use_sparse or ReindexStrategy is None or ReindexArrayType is None:
        return {}
    return {
        "reindex": ReindexStrategy(blockwise=False, array_type=ReindexArrayType.SPARSE_COO),
        "fill_value": 0,
    }

def build_output_parquet(model_version: str, run_name: str, run_date: str, interval: str) -> str:
    base = posixpath.join(ROOT, f"version_{model_version}", "zonal_stats", run_name, run_date, interval)
    return base.rstrip("/") + "/"

def _split_s3(path: str) -> tuple[str, str]:
    if not path.startswith("s3://"):
        raise ValueError(f"Expected s3:// path, got {path}")
    bucket, key = path[5:].split("/", 1)
    return bucket, key

def zarr_exists(zarr_path: str) -> bool:
    """True if a Zarr v2 (.zgroup) or v3 (zarr.json) root exists."""
    b, k = _split_s3(zarr_path.rstrip("/"))
    s3 = boto3.client("s3")
    for probe in ("zarr.json", ".zgroup"):
        try:
            s3.head_object(Bucket=b, Key=f"{k}/{probe}")
            return True
        except Exception:
            continue
    return False

def _first_xy_var(ds_or_da: xr.Dataset | xr.DataArray) -> xr.DataArray:
    if isinstance(ds_or_da, xr.DataArray):
        da_ = ds_or_da
    else:
        vars_xy = [v for v in ds_or_da.data_vars.values() if {"x", "y"}.issubset(v.dims)]
        da_ = vars_xy[0] if vars_xy else next(iter(ds_or_da.data_vars.values()))
    if "band" in da_.dims:
        da_ = da_.isel(band=0, drop=True)
    return da_

def open_zarr_region(path: str, bbox: Optional[List[float]], chunk_size: int) -> xr.DataArray:
    dsx = xr.open_zarr(path, consolidated=None, storage_options={"anon": False})
    data_arr = _first_xy_var(dsx)
    if bbox is not None and {"x", "y"}.issubset(data_arr.dims):
        west, south, east, north = bbox
        x0, x1 = float(data_arr.x.values[0]), float(data_arr.x.values[-1])
        y0, y1 = float(data_arr.y.values[0]), float(data_arr.y.values[-1])
        x_slice = slice(min(west, east), max(west, east)) if x0 < x1 else slice(max(east, west), min(east, west))
        y_slice = slice(min(south, north), max(south, north)) if y0 < y1 else slice(max(north, south), min(north, south))
        data_arr = data_arr.sel(x=x_slice, y=y_slice)
    chunk_dict = {d: chunk_size for d in ("x", "y") if d in data_arr.dims}
    if chunk_dict:
        data_arr = data_arr.chunk(chunk_dict)
    return data_arr

def pixel_step(arr: xr.DataArray) -> float:
    """Return approximate pixel size in degrees along x (absolute value)."""
    xvals = arr.x.values
    if xvals.size >= 2:
        return float(abs(xvals[1] - xvals[0]))
    # fallback for very small test windows
    return 1.0 / 4000.0  # 0.00025°

def align_like_nearest_tol(arr: xr.DataArray, ref: xr.DataArray, tol: float) -> xr.DataArray:
    """
    Align arr to ref's x/y using nearest-neighbor with a tolerance in degrees.
    If the nearest coordinate is farther than tol, NaN is produced (no far snapping).
    """
    # reindex_like preserves ref shape and coords
    return arr.reindex_like(ref, method="nearest", tolerance=tol)

def _upload_dir(fs_s3: s3fs.S3FileSystem, local_dir: Path, dest_prefix: str) -> int:
    uploaded = 0
    for p in local_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir)
            remote_path = posixpath.join(dest_prefix.rstrip("/"), *rel.parts)
            fs_s3.put(str(p), remote_path)
            uploaded += 1
    return uploaded

def convert_to_coord_dict(flox_result: xr.DataArray) -> dict:
    arr = flox_result.data
    if isinstance(arr, da.Array):
        arr = arr.compute()
    dims = flox_result.dims
    if hasattr(arr, "coords") and hasattr(arr, "data"):   # sparse.COO
        indices, values = arr.coords, arr.data
    else:
        grid = np.indices(arr.shape)
        indices = grid.reshape(len(arr.shape), -1)
        values = arr.ravel()
    return {dim: flox_result.coords[dim].values[indices[i]] for i, dim in enumerate(dims)} | {"value": values}

def _df_from_result(res: xr.DataArray, flux_map: Dict[int, str], interval_end: int) -> pd.DataFrame:
    coord_dict = convert_to_coord_dict(res)
    df = pd.DataFrame(coord_dict)
    df["flux_type"] = df["flux_type"].replace(flux_map)

    if "drained_state_nodes" in df.columns:
        df["drained_state_meaning"] = (
            df["drained_state_nodes"].astype("string").str.zfill(8).map(zc.DRAINED_STATE_NODE_MEANINGS)
        )
    if "burned_state_nodes" in df.columns:
        df["burned_state_meaning"] = (
            df["burned_state_nodes"].astype("string").str.zfill(8).map(zc.BURNED_STATE_NODE_MEANINGS)
        )

    df["interval_end"] = interval_end
    # Convert area m² → ha
    df.loc[df["flux_type"].eq("area__ha"), "value"] = df["value"] / 10000.0
    return df

def build_zarr_paths(interval: str, **fmt_kw) -> Dict[str, Dict[str, Any]]:
    zarr_base = ZARR_CACHE_PREFIX.format(interval=interval, **fmt_kw)
    out: Dict[str, Dict[str, Any]] = {}
    for name, spec in DATASETS.items():
        out[name] = {"zarr": zarr_base + spec["zarr"].format(interval=interval),
                     "var": spec["var"]}
    return out

# -------------------------------- driver --------------------------------
def run(args: argparse.Namespace) -> None:
    stage = "zonal_statistics"
    start_ts = uu.timestr()
    cluster, client, run_local = uu.connect_to_cluster(
        cluster_name=args.cluster_name, run_local=args.run_local,
    )

    # ROI from bbox or union of tile_ids
    bbox: Optional[List[float]] = None
    if args.bounding_box:
        bbox = [float(x) for x in args.bounding_box]
    elif args.tile_ids:
        tiles: List[str] = []
        for item in args.tile_ids:
            tiles.extend(t.strip() for t in item.split(",") if t.strip())
        if tiles:
            bounds = [uu.get_10x10_tile_bounds(t) for t in tiles]
            west = min(b[0] for b in bounds); south = min(b[1] for b in bounds)
            east = max(b[2] for b in bounds); north = max(b[3] for b in bounds)
            bbox = [west, south, east, north]

    logger, _ = lu.populate_main_log_header(
        bounding_box=bbox, use_shapefile=False, client=client, cluster=cluster,
        log_note="Organic soils zonal statistics (per-pixel; robust alignment)", run_local=run_local,
        model_type="organic_soils", stage=stage,
    )
    if args.debug:
        logger.setLevel(logging.DEBUG)
    logger.debug("Starting run with args: %s", args)

    OUTPUT_KW = dict(root=ROOT, model_version=args.model_version, run_date=args.run_date, run_name=args.run_name)

    # Open contextual layers (canonical grid comes from pixel_area)
    adm0 = open_zarr_region(ADM0_ZARR, bbox, args.chunk_size).astype("uint32")
    pixel_area = open_zarr_region(PIXEL_AREA_ZARR, bbox, args.chunk_size).persist()

    # Expected groups (exclude 0 to avoid ocean bookkeeping)
    gadm_adm0_ids = np.array([i for i in zc.GADM_ADM0_IDS if i > 0], dtype=np.uint32)
    drained_codes_arr = np.array(sorted({0, *map(int, zc.ALL_DRAINED_STATE_CODES)}), dtype=np.uint32)
    burned_codes_arr  = np.array(sorted({0, *map(int, zc.ALL_BURNED_STATE_CODES)}),  dtype=np.uint32)

    # Local staging
    local_arrow = pafs.LocalFileSystem()
    base_dir_root = Path(args.local_output).expanduser().resolve()
    base_dir_root.mkdir(parents=True, exist_ok=True)
    base_dir_drained = base_dir_root / "drained"
    base_dir_burned = base_dir_root / "burned"
    fs_s3 = s3fs.S3FileSystem(anon=False)

    # Canonical reference grid = pixel_area
    ref = pixel_area
    dx = pixel_step(ref)
    tol = float(args.align_tolerance_fraction) * dx  # e.g., 0.49 * pixel width

    # Intervals
    mapping = {end: (start, end) for start, end in cn.five_year_inventory_periods}
    interval_pairs = [mapping[y] for y in args.interval_end_years if y in mapping]

    for interval_start_year, interval_end_year in interval_pairs:
        interval = f"{interval_start_year}_{interval_end_year}"
        logger.info("Processing interval %s : %s", interval, timestr())

        # Build Zarr paths and assert existence
        paths = build_zarr_paths(interval, **OUTPUT_KW)
        for key in ("drained_total", "burned_total", "drained_state_nodes", "burned_state_nodes"):
            zpath = paths[key]["zarr"]
            if not zarr_exists(zpath):
                raise FileNotFoundError(
                    f"Missing Zarr for {key}: {zpath}\n"
                    f"Build with: python -m src.scripts.zonal_statistics.01_build_zarr_caches "
                    f"--interval_end_years {interval_end_year} --run_date {args.run_date} "
                    f"--model_version {args.model_version} --run_name {args.run_name} "
                    f"--chunk_size {args.chunk_size}"
                )

        # Open variables (raw)
        drained_total_raw = open_zarr_region(paths["drained_total"]["zarr"], bbox, args.chunk_size)
        burned_total_raw  = open_zarr_region(paths["burned_total"]["zarr"],  bbox, args.chunk_size)
        drained_nodes_raw = open_zarr_region(paths["drained_state_nodes"]["zarr"], bbox, args.chunk_size).astype("uint32")
        burned_nodes_raw  = open_zarr_region(paths["burned_state_nodes"]["zarr"],  bbox, args.chunk_size).astype("uint32")

        # Align all to canonical ref (pixel_area) with nearest + tolerance
        adm0_aligned         = align_like_nearest_tol(adm0,         ref, tol)
        drained_total_aligned= align_like_nearest_tol(drained_total_raw, ref, tol)
        burned_total_aligned = align_like_nearest_tol(burned_total_raw,  ref, tol)
        drained_nodes_aligned= align_like_nearest_tol(drained_nodes_raw, ref, tol)
        burned_nodes_aligned = align_like_nearest_tol(burned_nodes_raw,  ref, tol)

        # Diagnostics: how much flux lies where adm0 == 0 (before masking)?
        try:
            flux_mask = ((drained_total_aligned > 0) | (burned_total_aligned > 0))
            ocean_mask = (adm0_aligned == 0)
            leak = (flux_mask & ocean_mask).sum().compute()
            denom = flux_mask.sum().compute()
            leak_ratio = float(leak) / float(denom) if denom else 0.0
            if leak_ratio > args.leak_warn_threshold:
                logger.warning(
                    "Flux-over-ocean (adm0==0) ratio is %.4f (> %.4f). "
                    "This typically indicates grid misalignment or missing tiles.",
                    leak_ratio, args.leak_warn_threshold
                )
            else:
                logger.info("Flux-over-ocean (adm0==0) ratio: %.4f", leak_ratio)
        except Exception:
            # Non-fatal; continue
            pass

        # Labels & mask for flox
        where_mask = (adm0_aligned > 0)
        adm0_labels = adm0_aligned
        drained_nodes_aligned = drained_nodes_aligned
        burned_nodes_aligned  = burned_nodes_aligned

        # ------ Drained aggregation (sum) ------
        with dask.annotate(label=f"reduce:drained:{interval}"):
            cube_d = xr.concat([drained_total_aligned, ref], dim="flux_type").assign_coords(
                flux_type=("flux_type", [0, 2])
            )
            res_d = xarray_reduce(
                cube_d,
                adm0_labels,
                drained_nodes_aligned,
                func="sum",
                expected_groups=(gadm_adm0_ids, drained_codes_arr),
                where=where_mask,
                **flox_sparse_reindex_kwargs(not args.no_sparse),
            ).compute()
        df_d = _df_from_result(res_d, {0: "drained_total_Mg_CO2e", 2: "area__ha"}, interval_end_year)

        # ------ Burned aggregation (sum) ------
        with dask.annotate(label=f"reduce:burned:{interval}"):
            cube_b = xr.concat([burned_total_aligned, ref], dim="flux_type").assign_coords(
                flux_type=("flux_type", [1, 2])
            )
            res_b = xarray_reduce(
                cube_b,
                adm0_labels,
                burned_nodes_aligned,
                func="sum",
                expected_groups=(gadm_adm0_ids, burned_codes_arr),
                where=where_mask,
                **flox_sparse_reindex_kwargs(not args.no_sparse),
            ).compute()
        df_b = _df_from_result(res_b, {1: "burned_total_Mg_CO2e", 2: "area__ha"}, interval_end_year)

        # ------ Write local Parquet (per interval) ------
        import shutil
        local_d = base_dir_drained / interval
        local_b = base_dir_burned / interval
        for pth in (local_d, local_b):
            if pth.exists():
                shutil.rmtree(pth, ignore_errors=True)
            pth.mkdir(parents=True, exist_ok=True)

        ds.write_dataset(pa.Table.from_pandas(df_d, preserve_index=False), base_dir=str(local_d),
                         filesystem=local_arrow, format="parquet", existing_data_behavior="overwrite_or_ignore")
        ds.write_dataset(pa.Table.from_pandas(df_b, preserve_index=False), base_dir=str(local_b),
                         filesystem=local_arrow, format="parquet", existing_data_behavior="overwrite_or_ignore")
        logger.info("Wrote %s rows (drained) and %s rows (burned) for %s", len(df_d), len(df_b), interval)

        # ------ Upload to S3 ------
        dest_root = build_output_parquet(args.model_version, args.run_name, args.run_date, interval)
        dest_d = posixpath.join(dest_root.rstrip("/"), "drained")
        dest_b = posixpath.join(dest_root.rstrip("/"), "burned")
        _upload_dir(fs_s3, local_d, dest_d)
        _upload_dir(fs_s3, local_b, dest_b)
        logger.info("Uploaded interval %s → %s{drained,burned}", interval, dest_root)

        if not args.keep_local:
            shutil.rmtree(local_d, ignore_errors=True)
            shutil.rmtree(local_b, ignore_errors=True)

    if not args.keep_local:
        import shutil
        shutil.rmtree(base_dir_root, ignore_errors=True)

    if client: client.close()
    if cluster: cluster.close()
    uu.stage_duration(start_ts, uu.timestr(), stage)

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Run organic-soils zonal statistics (per-pixel; robust alignment)")
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--run_date", required=True)
    parser.add_argument("--interval_end_years", nargs="+", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--local_output", default="/tmp/zonal_stats")
    parser.add_argument("--keep_local", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_sparse", action="store_true", default=not SPARSE_DEFAULT)
    parser.add_argument("--run_name", default="ogh_standard_model")
    # Alignment controls
    parser.add_argument("--align_tolerance_fraction", type=float, default=0.49,
                        help="Fraction of one pixel for nearest reindex tolerance (default 0.49).")
    parser.add_argument("--leak_warn_threshold", type=float, default=0.002,
                        help="Warn if fraction of flux where adm0==0 exceeds this (default 0.002 = 0.2%).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run_local", action="store_true")
    mode.add_argument("--cluster_name", default="zonal_stats")
    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W S E N")
    parser.add_argument("--tile_ids", action="append", help="Comma separated 10×10 tile IDs (e.g., 00N_110E)")
    args = parser.parse_args(argv)
    run(args)

if __name__ == "__main__":
    main()
