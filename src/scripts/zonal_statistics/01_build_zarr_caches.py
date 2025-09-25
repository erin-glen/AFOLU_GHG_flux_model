# -*- coding: utf-8 -*-
"""Build aligned Zarr caches for organic-soils zonal statistics (canonical grid = pixel_area).

What this does
--------------
- Lists input GeoTIFFs per dataset/interval under:
  .../version_{model_version}/{folder}/{run_name}/five_year_intervals/{interval}/{tile_pixels}_pixels/{run_date}/
- Opens them with xarray/dask, coerces coords to float, rounds, and
  **reindexes to the pixel_area Zarr grid using nearest-with-tolerance**.
- Writes per-variable Zarrs under:
  .../version_{model_version}/zarr/{run_name}/{run_date}/{interval}/
- Validates that each Zarr matches the pixel_area grid (shape/chunks),
  prints quick stats (object count and total size), and samples a few blocks.

Key design choices
------------------
- Alignment is done at build time (Step 1), so Step 2 aggregation doesn't
  need to correct for misalignment (which led to adm0==0 leakage).
- Tolerance defaults to 0.49 * pixel_width (in degrees) to prevent far "nearest"
  snapping. If your grid has minor floating-point jitter, this captures
  the intended neighbors but won't leap across larger gaps.

Examples
--------
# Global 10x10° inputs (40000 px/10°), overwrite existing, moderate chunks
python -m src.scripts.zonal_statistics.01_build_zarr_caches \
  --interval_end_years 2024 \
  --cluster_name zonal_stats \
  --run_date 20250923 \
  --model_version 0_8_0 \
  --run_name ogh_sensitivity_1km \
  --tile_pixels 40000 \
  --chunk_size 8000 \
  --write_mode w

# Skip if exists (validate only)
python -m src.scripts.zonal_statistics.01_build_zarr_caches \
  --interval_end_years 2024 \
  --cluster_name zonal_stats \
  --run_date 20250923 \
  --model_version 0_8_0 \
  --run_name ogh_sensitivity_1km \
  --tile_pixels 40000 \
  --chunk_size 8000 \
  --write_mode w-
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import dask
import dask.array as da
import fsspec
import numpy as np
import pandas as pd
import posixpath
import s3fs
import xarray as xr
import zarr

from packaging.version import Version

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu

# ------------------------------ constants ------------------------------
ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
OUTPUT_BASE = "{root}/version_{model_version}"

DATASETS: Dict[str, Dict[str, Any]] = {
    "drained_total": {
        "folder": "drained_total_Mg_CO2e_pixel_yr",
        "zarr": "drained_total_Mg_CO2e_pixel_yr_{interval}.zarr",
        "var": "drained_total",
        "dtype": "float32",
    },
    "burned_total": {
        "folder": "burned_total_Mg_CO2e_pixel_yr",
        "zarr": "burned_total_Mg_CO2e_pixel_yr_{interval}.zarr",
        "var": "burned_total",
        "dtype": "float32",
    },
    "drained_state_nodes": {
        "folder": "drained_state",
        "zarr": "drained_state_node_{interval}.zarr",
        "var": "drained_state_nodes",
        "dtype": "uint32",
    },
    "burned_state_nodes": {
        "folder": "burned_state",
        "zarr": "burned_state_node_{interval}.zarr",
        "var": "burned_state_nodes",
        "dtype": "uint32",
    },
}

ZARR_CACHE_PREFIX = OUTPUT_BASE + "/zarr/{run_name}/{run_date}/{interval}/"
FOLDER_TEMPLATE = (
    OUTPUT_BASE
    + "/{folder}/{run_name}/five_year_intervals/{interval}/{tile_pixels}_pixels/{run_date}/"
)

# Canonical reference grid (pixel_area)
CONTEXTUAL_ZARR_ROOT = (
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/global_contextual_zarrs"
)
PIXEL_AREA_DATASET = "pixel_area"
PIXEL_AREA_DATE = "20250730"
PIXEL_AREA_ZARR = posixpath.join(
    CONTEXTUAL_ZARR_ROOT,
    PIXEL_AREA_DATASET,
    PIXEL_AREA_DATE,
    f"global_pixel_area_{PIXEL_AREA_DATE}.zarr",
)

# ------------------------------ helpers --------------------------------
def _split_s3(path: str) -> tuple[str, str]:
    if not path.startswith("s3://"):
        raise ValueError(f"Expected s3:// path, got {path}")
    bucket, key = path[5:].split("/", 1)
    return bucket, key

def zarr_exists(zarr_path: str) -> bool:
    """True if a Zarr v2 (.zgroup) or v3 (zarr.json) root exists (checked via boto3)."""
    b, k = _split_s3(zarr_path.rstrip("/"))
    s3 = boto3.client("s3")
    for probe in ("zarr.json", ".zgroup"):
        try:
            s3.head_object(Bucket=b, Key=f"{k}/{probe}")
            return True
        except Exception:
            continue
    return False

def remove_store_recursively(prefix: str) -> None:
    fs = s3fs.S3FileSystem(anon=False)
    fs.rm(prefix.rstrip("/") + "/", recursive=True)

def list_folder_uris(base_uri: str) -> List[str]:
    fs = s3fs.S3FileSystem(anon=False)
    pattern = base_uri.rstrip("/") + "/**/*.tif"
    files = fs.glob(pattern)
    return [(f if f.startswith("s3://") else f"s3://{f}") for f in files]

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
    arr = _first_xy_var(dsx)
    if bbox is not None and {"x", "y"}.issubset(arr.dims):
        west, south, east, north = bbox
        x0, x1 = float(arr.x.values[0]), float(arr.x.values[-1])
        y0, y1 = float(arr.y.values[0]), float(arr.y.values[-1])
        x_slice = slice(min(west, east), max(west, east)) if x0 < x1 else slice(max(east, west), min(east, west))
        y_slice = slice(min(south, north), max(south, north)) if y0 < y1 else slice(max(north, south), min(north, south))
        arr = arr.sel(x=x_slice, y=y_slice)
    if {"x", "y"}.issubset(arr.dims):
        arr = arr.chunk({d: chunk_size for d in ("x", "y")})
    return arr

def pixel_step(arr: xr.DataArray) -> float:
    xvals = arr.x.values
    return float(abs(xvals[1] - xvals[0])) if xvals.size >= 2 else 1.0 / 4000.0

def round_xy(arr: xr.DataArray, ndig: int = 12) -> xr.DataArray:
    out = arr
    if "x" in arr.coords:
        out = out.assign_coords(x=np.round(out.x.astype(float), ndig))
    if "y" in arr.coords:
        out = out.assign_coords(y=np.round(out.y.astype(float), ndig))
    return out

def align_like_nearest_tol(arr: xr.DataArray, ref: xr.DataArray, tol_deg: float) -> xr.DataArray:
    """Reindex to ref coords with nearest+bounded tolerance; yields NaN where too far."""
    # ensure numeric, rounded coords (reduces FP jitter issues)
    arr2 = round_xy(arr)
    ref2 = round_xy(ref)
    return arr2.reindex_like(ref2, method="nearest", tolerance=tol_deg)

def make_xarray_chunks_from_tiffs(tiffs: List[str], chunk_size: int) -> xr.DataArray:
    """Open TIFFs → Dataset → first xy var → chunk."""
    if not tiffs:
        raise FileNotFoundError("No GeoTIFFs found for dataset.")
    with dask.annotate(label="open_mfdataset"):
        ds = xr.open_mfdataset(
            tiffs,
            parallel=True,
            chunks={"x": chunk_size, "y": chunk_size},
        ).squeeze()
    da_ = _first_xy_var(ds)
    return da_

def _dtype_cast(arr: xr.DataArray, dtype_str: str) -> xr.DataArray:
    if dtype_str == "uint32":
        return arr.fillna(0).astype("uint32")
    if dtype_str == "float32":
        return arr.fillna(0).astype("float32")
    return arr

def ensure_consolidated_v2(store_path: str) -> None:
    """Consolidate metadata only for Zarr v2 stores."""
    # Try to detect v2 by presence of .zgroup and absence of zarr.json
    fs, inner = fsspec.core.url_to_fs(store_path)
    has_zgroup = fs.exists(posixpath.join(inner, ".zgroup"))
    has_v3json = fs.exists(posixpath.join(inner, "zarr.json"))
    if has_zgroup and not has_v3json:
        try:
            zarr.convenience.consolidate_metadata(fs.get_mapper(inner))
        except Exception:
            pass

def validate_zarr(zarr_path: str, ref: xr.DataArray, *, logger: logging.Logger) -> None:
    """Open and report: shape, chunking, object count/size, sample a few blocks; assert shape==ref."""
    arr = _first_xy_var(xr.open_zarr(zarr_path, consolidated=None, storage_options={"anon": False}))
    # Shape check
    if arr.sizes.get("x") != ref.sizes.get("x") or arr.sizes.get("y") != ref.sizes.get("y"):
        raise ValueError(
            f"Zarr grid mismatch for {zarr_path} "
            f"(got (y:{arr.sizes.get('y')}, x:{arr.sizes.get('x')}), "
            f"expected (y:{ref.sizes.get('y')}, x:{ref.sizes.get('x')}))"
        )
    # Chunk info
    try:
        cx = arr.data.chunks[1] if isinstance(arr.data, da.Array) else ()
        cy = arr.data.chunks[0] if isinstance(arr.data, da.Array) else ()
    except Exception:
        cx, cy = (), ()

    # S3 object stats
    fs = s3fs.S3FileSystem(anon=False)
    b, k = _split_s3(zarr_path.rstrip("/"))
    objs = fs.find(f"{b}/{k}")
    nobj = len(objs)
    size_bytes = 0
    for o in objs:
        try:
            size_bytes += fs.size(o)
        except Exception:
            pass
    size_gb = size_bytes / (1024 ** 3)

    # Sample a few small slices to ensure there is real data
    samples: List[Tuple[Tuple[int, int], float]] = []
    try:
        for i in (0, max(0, arr.sizes["y"] // 2), max(0, arr.sizes["y"] - 128)):
            for j in (0, max(0, arr.sizes["x"] // 2), max(0, arr.sizes["x"] - 128)):
                sli = arr.isel(y=slice(i, min(i + 128, arr.sizes["y"])),
                               x=slice(j, min(j + 128, arr.sizes["x"])))
                data = sli.data
                if isinstance(data, da.Array):
                    data = data.compute()
                samples.append((sli.shape, float(np.nanmax(data)) if data.size else float("nan")))
    except Exception as e:
        logger.warning("Sampling warning for %s: %s", zarr_path, e)

    logger.info(
        "flm: Validated Zarr ✅ %s | shape=(y:%d, x:%d) chunk_x=%s chunk_y=%s | objects=%d size=%.2f GB | samples=%s",
        zarr_path, arr.sizes["y"], arr.sizes["x"], cx, cy, nobj, size_gb, samples
    )

# ------------------------------ pipeline --------------------------------
def build_paths(interval: str, *, tile_pixels: int, **kw) -> Dict[str, Dict[str, Any]]:
    """Return input folder and output zarr path for each dataset."""
    zarr_base = ZARR_CACHE_PREFIX.format(interval=interval, **kw)
    kw2 = dict(kw); kw2["tile_pixels"] = tile_pixels
    out: Dict[str, Dict[str, Any]] = {}
    for name, spec in DATASETS.items():
        folder_uri = FOLDER_TEMPLATE.format(folder=spec["folder"], interval=interval, **kw2)
        out[name] = {
            "folder": folder_uri,
            "zarr":   zarr_base + spec["zarr"].format(interval=interval),
            "var":    spec["var"],
            "dtype":  spec["dtype"],
        }
    return out

def run(args: argparse.Namespace) -> None:
    stage = "zarr_build"
    start_ts = uu.timestr()
    cluster, client, run_local = uu.connect_to_cluster(
        cluster_name=args.cluster_name, run_local=args.run_local
    )

    logger, _ = lu.populate_main_log_header(
        bounding_box=None,
        use_shapefile=False,
        client=client,
        cluster=cluster,
        log_note="Organic soils Zarr build",
        run_local=run_local,
        model_type="organic_soils",
        stage=stage,
    )
    if args.debug:
        logger.setLevel(logging.DEBUG)

    OUTPUT_KW = dict(root=ROOT, model_version=args.model_version,
                     run_date=args.run_date, run_name=args.run_name)

    # Canonical reference grid (pixel_area), optionally cropped for smoke tests
    ref = open_zarr_region(PIXEL_AREA_ZARR, None, args.chunk_size)  # always open full for shape
    if args.bounding_box:
        west, south, east, north = [float(x) for x in args.bounding_box]
        ref = ref.sel(x=slice(west, east), y=slice(south, north))

    # Tolerance in degrees
    dx = pixel_step(ref)
    tol = float(args.align_tolerance_fraction) * dx

    logger.info("flm: Model version: %s", args.model_version)
    if client:
        try:
            mem, n_workers, nth = uu.get_cluster_info(client, cluster)  # coiled convenience
            logger.info("flm: Number of workers: %s", n_workers)
            logger.info("flm: Memory per worker: %s", mem)
            logger.info("flm: Threads per worker: %s", nth)
        except Exception:
            pass

    # Map end years → (start, end) intervals
    mapping = {end: (start, end) for start, end in cn.five_year_inventory_periods}
    interval_pairs = [mapping[y] for y in args.interval_end_years if y in mapping]

    for iv_start, iv_end in interval_pairs:
        interval = f"{iv_start}_{iv_end}"
        logger.info("flm: Interval %s", interval)

        paths = build_paths(interval, tile_pixels=args.tile_pixels, **OUTPUT_KW)

        for key in ("drained_total", "burned_total", "drained_state_nodes", "burned_state_nodes"):
            spec = paths[key]
            folder = spec["folder"]
            zpath  = spec["zarr"]
            var    = spec["var"]
            dtype  = spec["dtype"]

            logger.info(
                "flm: Building %s → %s",
                folder, zpath
            )

            # Handle write mode
            exists = zarr_exists(zpath)
            if exists and args.write_mode == "w-":
                logger.info("flm: Zarr exists and write_mode is 'w-': %s", zpath)
                # Validate existing store
                validate_zarr(zpath, ref, logger=logger)
                continue
            if exists and args.write_mode == "w":
                logger.info("flm: Removing existing store (write_mode='w'): %s", zpath)
                remove_store_recursively(zpath)

            # List input TIFFs
            tiffs = list_folder_uris(folder)
            if not tiffs:
                raise FileNotFoundError(f"No GeoTIFFs found under {folder}")
            logger.info("flm:   • %d TIFF(s) found", len(tiffs))

            # Build → align → cast
            try:
                da_in = make_xarray_chunks_from_tiffs(tiffs, args.chunk_size)
            except Exception as e:
                # Fallback: try to filter unreadable TIFFs (rare)
                logger.warning("open_mfdataset failed: %s. Attempting to filter unreadable tiles.", e)
                valid: List[str] = []
                for u in tiffs:
                    try:
                        with xr.open_dataset(u) as _:
                            valid.append(u)
                    except Exception:
                        logger.warning("Skipping unreadable TIFF: %s", u)
                if not valid:
                    raise RuntimeError(f"All tiles failed for {folder}")
                da_in = make_xarray_chunks_from_tiffs(valid, args.chunk_size)

            # Align to the canonical reference grid
            da_aligned = align_like_nearest_tol(da_in, ref, tol)

            # Type and chunk normalization
            da_aligned = _dtype_cast(da_aligned, dtype)
            da_aligned = da_aligned.chunk({d: args.chunk_size for d in ("x", "y") if d in da_aligned.dims})

            # Write Zarr
            ds_out = xr.Dataset({var: da_aligned})
            with dask.annotate(label=f"to_zarr:{key}:{interval}"):
                ds_out.to_zarr(zpath, mode="w")

            # Consolidate for zarr v2 only
            if Version(zarr.__version__).major < 3:
                ensure_consolidated_v2(zpath)

            # Validate
            validate_zarr(zpath, ref, logger=logger)

    uu.stage_duration(start_ts, uu.timestr(), stage)
    if client:
        client.close()
    if cluster:
        cluster.close()

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Build aligned Zarr caches for organic soils")
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--run_date", required=True)
    parser.add_argument("--interval_end_years", nargs="+", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, default=8000)
    parser.add_argument("--tile_pixels", type=int, default=40000,
                        help="Input tile size in pixels in the source folder path (4000 or 40000).")
    parser.add_argument("--run_name", default="ogh_standard_model")
    parser.add_argument("--write_mode", choices=["w", "w-"], default="w-",
                        help="'w' overwrite, 'w-' skip if exists (validate only).")
    parser.add_argument("--align_tolerance_fraction", type=float, default=0.49,
                        help="Fraction of one pixel as nearest reindex tolerance (default 0.49).")
    parser.add_argument("--debug", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run_local", action="store_true")
    mode.add_argument("--cluster_name", default="zarr_build")
    # Optional: crop pixel_area during dev smoke tests
    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W S E N (optional, dev only)")
    args = parser.parse_args(argv)
    run(args)

if __name__ == "__main__":
    main()