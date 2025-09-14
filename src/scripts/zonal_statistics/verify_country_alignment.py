#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_country_alignment.py

Purpose
-------
Diagnose whether large zonal totals in GADM adm0 == 0 (ocean/void) are caused
by grid misalignment (1-pixel "snap") in the alignment step.

Method
------
For a chosen interval/ROI, the script:
  1) Loads drained & burned totals, drained state-nodes, adm0, and pixel_area.
     - Prefers GeoTIFF tiles; falls back to combined-interval Zarr; then per-var Zarr.
     - Detects whether flux totals are per-ha or per-pixel and converts to per-pixel if needed.
  2) Aligns layers to the drained_state_nodes grid using:
       A) "Naive" snap (mirrors your current `safe_crop`: nearest, no tolerance).
       B) "Tolerant" reindex (nearest WITH tolerance = half a pixel). Fails if larger snap is needed.
  3) Computes human-friendly metrics:
       - Share of flux in adm0==0 (ocean) for drained & burned.
       - Share of drained flux in adm0==0 where nodes are valid (nodes>0) — strong misalignment signal.
  4) Runs a ±1 pixel shift test on adm0 (E/W/N/S) under the naive alignment and reports the best reduction.
  5) Prints a clear VERDICT with suggested next actions.

CLI Examples
------------
# Minimal: one interval, 1x1 tiles
python -m src.scripts.zonal_statistics.verify_country_alignment \
  --model_version 0_7_0 --run_date 20250825 --run_name ogh_standard_model \
  --interval_end_years 2020 \
  --tile_pixels 4000 --chunk_size 10000 \
  --bounding_box 112 -2 113 -1 \
  --out_dir /mnt/c/tmp/os_verify --debug

# If your tiles are 10x10 (40000 pixels):
python -m src.scripts.zonal_statistics.verify_country_alignment \
  --model_version 0_7_0 --run_date 20250825 --run_name ogh_standard_model \
  --interval_end_years 2020 \
  --tile_pixels 40000 --chunk_size 20000 \
  --out_dir /mnt/c/tmp/os_verify

Outputs
-------
- A concise console report with percentages & verdict.
- CSV summary per interval in --out_dir:
    * verify_summary_{interval}.csv
    * verify_shift_{interval}.csv

Notes
-----
- We do NOT mask out adm0==0; we only measure its contribution.
- “Valid nodes” means drained_state_nodes > 0 (node code 0 is treated as nodata).
- If tolerant alignment collapses ocean totals and/or a ±1px shift kills the signal,
  you have a classic 1-pixel misalignment issue.

Author: Organic Soils Model Development
"""

from __future__ import annotations
import argparse, logging, posixpath, shutil, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import dask.array as da
import xarray as xr
import s3fs
import zarr

# ------------------------------- CONSTANTS -------------------------------

ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
OUTPUT_BASE = "{root}/version_{model_version}"

# Tile GeoTIFF locations
FOLDER_TEMPLATE = (
    OUTPUT_BASE
    + "/{folder}/{run_name}/five_year_intervals/{interval}/{tile_pixels}_pixels/{run_date}/"
)

# Zarr caches (if present)
ZARR_CACHE_PREFIX = OUTPUT_BASE + "/zarr/{run_name}/{run_date}/{interval}/"
COMBINED_ZARR = ZARR_CACHE_PREFIX + "combined_interval.zarr"

# Dataset map
DATASETS = {
    "drained_state_nodes": {"folder": "drained_state", "zarr": "drained_state_node_{interval}.zarr", "var": "drained_state_nodes"},
    "burned_state_nodes":  {"folder": "burned_state",  "zarr": "burned_state_node_{interval}.zarr",  "var": "burned_state_nodes"},
    # Flux candidates
    "drained_total_pixel": {"folder": "drained_total_Mg_CO2e_pixel_yr", "zarr": "drained_total_Mg_CO2e_pixel_yr_{interval}.zarr", "var": "drained_total"},
    "drained_total_ha":    {"folder": "drained_total_Mg_CO2e_ha_yr",    "zarr": "drained_total_Mg_CO2e_ha_yr_{interval}.zarr",    "var": "drained_total"},
    "burned_total_pixel":  {"folder": "burned_total_Mg_CO2e_pixel_yr",  "zarr": "burned_total_Mg_CO2e_pixel_yr_{interval}.zarr",  "var": "burned_total"},
    "burned_total_ha":     {"folder": "burned_total_Mg_CO2e_ha_yr",     "zarr": "burned_total_Mg_CO2e_ha_yr_{interval}.zarr",     "var": "burned_total"},
}

# Contextual Zarrs
ADM0_ZARR = (
    "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/"
    "GADM4_1_adm0_global/20250604/global_GADM41_adm0_20250604.zarr"
)
PIXEL_AREA_ZARR = (
    "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/"
    "pixel_area/20250730/global_pixel_area_20250730.zarr"
)

# ------------------------------ UTILITIES -------------------------------

def set_log_levels(human_only: bool):
    """Suppress very chatty third-party logs at INFO unless --debug is set."""
    if not human_only:
        return
    noisy = ["botocore", "aiobotocore", "s3fs", "fsspec", "rasterio", "urllib3", "asyncio", "numexpr"]
    for n in noisy:
        logging.getLogger(n).setLevel(logging.WARNING)

def s3fs_client():
    return s3fs.S3FileSystem(anon=False)

def s3_glob(pattern: str) -> List[str]:
    fs = s3fs_client()
    return [("s3://" + p) if not p.startswith("s3://") else p for p in fs.glob(pattern)]

def s3_exists(prefix: str) -> bool:
    fs = s3fs_client()
    try:
        return any(fs.glob(prefix.rstrip("/") + "/**"))
    except FileNotFoundError:
        return False

def list_folder_tifs(prefix: str) -> List[str]:
    return s3_glob(prefix.rstrip("/") + "/**/*.tif")

def open_mf(uris: List[str], chunk: int) -> xr.Dataset:
    if not uris:
        raise OSError("no files to open")
    return xr.open_mfdataset(uris, parallel=True, chunks={"x": chunk, "y": chunk}).squeeze()

def first_xy_var(ds: xr.Dataset | xr.DataArray) -> xr.DataArray:
    da_ = ds if isinstance(ds, xr.DataArray) else next(iter([v for v in ds.data_vars.values() if {"x","y"}.issubset(v.dims)]), None)
    if da_ is None:
        da_ = next(iter(ds.data_vars.values()))
    if "band" in da_.dims:
        da_ = da_.isel(band=0, drop=True)
    return da_

def crop_bbox(arr: xr.DataArray, bbox: Optional[List[float]]) -> xr.DataArray:
    if bbox is None or not {"x","y"}.issubset(arr.dims):
        return arr
    west,south,east,north = [float(x) for x in bbox]
    x_asc = bool(arr.x[0] < arr.x[-1]); y_asc = bool(arr.y[0] < arr.y[-1])
    xs = slice(min(west,east), max(west,east)) if x_asc else slice(max(east,west), min(east,west))
    ys = slice(min(south,north),max(south,north)) if y_asc else slice(max(north,south),min(north,south))
    return arr.sel(x=xs, y=ys)

def reindex_like_tolerant(arr: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Nearest neighbor reindex with half-pixel tolerance (no silent 1px snap)."""
    arr2 = crop_bbox(arr, [float(ref.x.min()), float(ref.y.min()), float(ref.x.max()), float(ref.y.max())])
    px = float(abs(ref.x[1] - ref.x[0])) if ref.sizes["x"] > 1 else 0.00025
    py = float(abs(ref.y[1] - ref.y[0])) if ref.sizes["y"] > 1 else 0.00025
    tolx, toly = px/2.1, py/2.1
    return arr2.reindex(x=ref.x, y=ref.y, method="nearest", tolerance={"x": tolx, "y": toly})

def naive_snap(arr: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Mirror of your current safe_crop behavior (nearest + overwrite coords)."""
    out = arr.sel(x=ref.x, y=ref.y, method="nearest")
    return out.assign_coords(x=ref.x, y=ref.y)

def shift_mask(mask: xr.DataArray, dx: int, dy: int) -> xr.DataArray:
    return mask.shift(x=dx, y=dy)

def _sum(arr: xr.DataArray, mask: Optional[xr.DataArray]=None) -> float:
    a = arr.where(mask) if mask is not None else arr
    val = float(da.nansum(a.data).compute())
    return val

def format_pct(x: float) -> str:
    return f"{(100.0 * x):.4f}%"

def interval_label(end_year: int) -> str:
    # For 5-yr inventory periods, start = end - 4 (matches e.g., 2016_2020)
    return f"{end_year-4}_{end_year}"

# ------------------------- DATA DISCOVERY & OPEN -------------------------

def discover_flux_unit(interval: str, tile_pixels: int, OUTPUT_KW: dict) -> Tuple[str, str]:
    """
    Return (unit, source) where unit in {'ha','pixel'} and source is the S3 folder used.
    Preference: per-ha if present.
    """
    candidates = [
        (DATASETS["drained_total_ha"]["folder"], "ha"),
        (DATASETS["drained_total_pixel"]["folder"], "pixel"),
    ]
    for folder, unit in candidates:
        pref = FOLDER_TEMPLATE.format(folder=folder, interval=interval, tile_pixels=tile_pixels, **OUTPUT_KW)
        if s3_exists(pref):
            return unit, pref
    # If neither tile folder exists, still try to infer unit for Zarr-based runs:
    # If the HA zarr exists, pick 'ha'; else 'pixel' if that zarr exists.
    zbase = ZARR_CACHE_PREFIX.format(interval=interval, **OUTPUT_KW)
    ha_z = zbase + DATASETS["drained_total_ha"]["zarr"].format(interval=interval)
    px_z = zbase + DATASETS["drained_total_pixel"]["zarr"].format(interval=interval)
    if zarr_exists(ha_z): return "ha", ha_z
    if zarr_exists(px_z): return "pixel", px_z
    # Unknown; default to 'pixel' (safer for relative shares)
    return "pixel", "(unknown)"

def zarr_exists(path: str) -> bool:
    fs = s3fs_client()
    parts = path.replace("s3://","").split("/",1)
    bucket, key = parts[0], parts[1]
    markers = [".zgroup", ".zmetadata", "zarr.json"]
    return any(fs.exists(posixpath.join(bucket, key, m)) for m in markers)

def try_open_from_tif(interval: str, args, OUTPUT_KW) -> Optional[Tuple[xr.DataArray,...,str]]:
    """Load from GeoTIFF tiles (preferred). Returns (..., source_name) if successful."""
    # State nodes
    dn_pref = FOLDER_TEMPLATE.format(folder=DATASETS["drained_state_nodes"]["folder"], interval=interval, tile_pixels=args.tile_pixels, **OUTPUT_KW)
    bn_pref = FOLDER_TEMPLATE.format(folder=DATASETS["burned_state_nodes"]["folder"],  interval=interval, tile_pixels=args.tile_pixels, **OUTPUT_KW)
    dn_files = list_folder_tifs(dn_pref)
    bn_files = list_folder_tifs(bn_pref)
    if not dn_files or not bn_files:
        return None

    # Flux totals (prefer per-ha)
    unit_d, d_src = discover_flux_unit(interval, args.tile_pixels, OUTPUT_KW)
    unit_b = None; b_files = []
    for folder, unit in [(DATASETS["burned_total_ha"]["folder"], "ha"), (DATASETS["burned_total_pixel"]["folder"], "pixel")]:
        pref = FOLDER_TEMPLATE.format(folder=folder, interval=interval, tile_pixels=args.tile_pixels, **OUTPUT_KW)
        b_files = list_folder_tifs(pref)
        if b_files:
            unit_b = unit; break
    if unit_b is None:
        return None

    dn = first_xy_var(open_mf(dn_files, args.chunk_size))
    bn = first_xy_var(open_mf(bn_files, args.chunk_size))
    dt = first_xy_var(open_mf(list_folder_tifs(FOLDER_TEMPLATE.format(folder=(DATASETS["drained_total_ha"]["folder"] if unit_d=="ha" else DATASETS["drained_total_pixel"]["folder"]), interval=interval, tile_pixels=args.tile_pixels, **OUTPUT_KW)), args.chunk_size))
    bt = first_xy_var(open_mf(b_files, args.chunk_size))

    # Contextual
    adm0 = first_xy_var(xr.open_zarr(ADM0_ZARR, consolidated=None, storage_options={"anon": False}))
    area = first_xy_var(xr.open_zarr(PIXEL_AREA_ZARR, consolidated=None, storage_options={"anon": False}))

    # Optional crop
    dn = crop_bbox(dn, args.bounding_box)
    bn = crop_bbox(bn, args.bounding_box)
    dt = crop_bbox(dt, args.bounding_box)
    bt = crop_bbox(bt, args.bounding_box)
    adm0 = crop_bbox(adm0, args.bounding_box)
    area = crop_bbox(area, args.bounding_box)

    return dn, bn, dt, bt, area, adm0, unit_d, unit_b, "GeoTIFF"

def try_open_from_combined_zarr(interval: str, args, OUTPUT_KW) -> Optional[Tuple[xr.DataArray,...,str]]:
    zpath = COMBINED_ZARR.format(interval=interval, **OUTPUT_KW)
    if not zarr_exists(zpath):
        return None
    ds = xr.open_zarr(zpath, consolidated=None, storage_options={"anon": False})
    dn = crop_bbox(ds[DATASETS["drained_state_nodes"]["var"]], args.bounding_box)
    bn = crop_bbox(ds[DATASETS["burned_state_nodes"]["var"]],  args.bounding_box)
    dt = crop_bbox(ds[DATASETS["drained_total_pixel"]["var"]], args.bounding_box)  # var name is 'drained_total'
    bt = crop_bbox(ds[DATASETS["burned_total_pixel"]["var"]],  args.bounding_box)  # 'burned_total'
    adm0 = crop_bbox(first_xy_var(xr.open_zarr(ADM0_ZARR, consolidated=None, storage_options={"anon": False})), args.bounding_box)
    area = crop_bbox(first_xy_var(xr.open_zarr(PIXEL_AREA_ZARR, consolidated=None, storage_options={"anon": False})), args.bounding_box)

    # Infer unit for dt/bt from S3 layout (prefer HA if present)
    unit_d, _ = discover_flux_unit(interval, args.tile_pixels, OUTPUT_KW)
    unit_b = unit_d  # reasonable default; burned often mirrors drained
    return dn, bn, dt, bt, area, adm0, unit_d, unit_b, "Combined-Zarr"

def try_open_from_pervar_zarr(interval: str, args, OUTPUT_KW) -> Optional[Tuple[xr.DataArray,...,str]]:
    zbase = ZARR_CACHE_PREFIX.format(interval=interval, **OUTPUT_KW)

    def _open_z(name: str) -> Optional[xr.DataArray]:
        zname = DATASETS[name]["zarr"].format(interval=interval)
        zpath = zbase + zname
        if not zarr_exists(zpath):
            return None
        ds = xr.open_zarr(zpath, consolidated=None, storage_options={"anon": False})
        return first_xy_var(ds)

    dn = _open_z("drained_state_nodes")
    bn = _open_z("burned_state_nodes")
    dt = _open_z("drained_total_ha") or _open_z("drained_total_pixel")
    bt = _open_z("burned_total_ha")  or _open_z("burned_total_pixel")
    if any(x is None for x in (dn, bn, dt, bt)):
        return None

    adm0 = first_xy_var(xr.open_zarr(ADM0_ZARR, consolidated=None, storage_options={"anon": False}))
    area = first_xy_var(xr.open_zarr(PIXEL_AREA_ZARR, consolidated=None, storage_options={"anon": False}))

    dn = crop_bbox(dn, args.bounding_box)
    bn = crop_bbox(bn, args.bounding_box)
    dt = crop_bbox(dt, args.bounding_box)
    bt = crop_bbox(bt, args.bounding_box)
    adm0 = crop_bbox(adm0, args.bounding_box)
    area = crop_bbox(area, args.bounding_box)

    # Unit inference
    unit_d, _ = discover_flux_unit(interval, args.tile_pixels, OUTPUT_KW)
    unit_b = unit_d
    return dn, bn, dt, bt, area, adm0, unit_d, unit_b, "Per-var-Zarr"

def open_interval(interval: str, args, OUTPUT_KW) -> Tuple[xr.DataArray,...,str]:
    got = try_open_from_tif(interval, args, OUTPUT_KW)
    if got: return got
    got = try_open_from_combined_zarr(interval, args, OUTPUT_KW)
    if got: return got
    got = try_open_from_pervar_zarr(interval, args, OUTPUT_KW)
    if got: return got
    raise FileNotFoundError(
        f"No inputs found for interval {interval}. Tried GeoTIFFs (tile size {args.tile_pixels}), "
        f"Combined-Zarr, and per-var Zarrs.\nCheck --model_version/--run_name/--run_date and tile size."
    )

# ------------------------------- DRIVER ----------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify whether adm0==0 totals are caused by grid misalignment")
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--run_date", required=True)
    parser.add_argument("--run_name", default="ogh_standard_model")
    parser.add_argument("--interval_end_years", nargs="+", type=int, required=True)
    parser.add_argument("--tile_pixels", type=int, default=4000)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--bounding_box", nargs=4, type=float, help="W S E N")
    parser.add_argument("--out_dir", default="/tmp/verify_country_alignment")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # Human-friendly logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(message)s"
    )
    set_log_levels(human_only=not args.debug)
    log = logging.getLogger("verify")

    OUTPUT_KW = dict(root=ROOT, model_version=args.model_version, run_date=args.run_date, run_name=args.run_name)

    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("▶ VERIFY COUNTRY ASSIGNMENT ISSUE (adm0==0) — Human-friendly report")
    log.info("  • model_version=%s  run_date=%s  run_name=%s", args.model_version, args.run_date, args.run_name)
    if args.bounding_box:
        log.info("  • ROI bbox = [W=%g, S=%g, E=%g, N=%g]", *args.bounding_box)
    log.info("  • tile_pixels=%d  chunk_size=%d", args.tile_pixels, args.chunk_size)
    log.info("")

    for end in args.interval_end_years:
        interval = interval_label(end)
        log.info("━" * 78)
        log.info("INTERVAL: %s", interval)
        log.info("STEP 1/4 — Discover and open inputs")

        # Open inputs with auto-discovery
        dn, bn, dt, bt, area, adm0, unit_d, unit_b, source = open_interval(interval, args, OUTPUT_KW)
        log.info("  • Source: %s", source)
        log.info("  • Flux units (drained/burned): %s / %s", unit_d, unit_b)

        # Alignments to test
        log.info("STEP 2/4 — Align to drained_state_nodes grid (two methods)")
        # NOTE: keep nodes as float so NaNs survive; treat 0 as nodata below.
        ref = dn

        # Naive snap (mirrors current pipeline behavior)
        adm0_naive   = naive_snap(adm0, ref)
        area_naive   = naive_snap(area, ref)
        dt_naive     = naive_snap(dt,   ref)
        bt_naive     = naive_snap(bt,   ref)
        bn_naive     = naive_snap(bn,   ref)

        # Tolerant reindex (half-pixel tolerance)
        adm0_tol = reindex_like_tolerant(adm0, ref)
        area_tol = reindex_like_tolerant(area, ref)
        dt_tol   = reindex_like_tolerant(dt,   ref)
        bt_tol   = reindex_like_tolerant(bt,   ref)
        bn_tol   = reindex_like_tolerant(bn,   ref)

        # Convert per-ha to per-pixel totals if necessary
        if unit_d == "ha":
            dt_naive = dt_naive * (area_naive / 10000.0)
            dt_tol   = dt_tol   * (area_tol   / 10000.0)
        if unit_b == "ha":
            bt_naive = bt_naive * (area_naive / 10000.0)
            bt_tol   = bt_tol   * (area_tol   / 10000.0)

        # Masks
        nodes_valid_naive = (ref.notnull()) & (ref != 0)
        nodes_valid_tol   = nodes_valid_naive  # same grid

        ocean_naive = (adm0_naive <= 0) | xr.ufuncs.isnan(adm0_naive)
        ocean_tol   = (adm0_tol   <= 0) | xr.ufuncs.isnan(adm0_tol)
        land_naive  = ~ocean_naive
        land_tol    = ~ocean_tol

        # Totals (drained focus + burned for context)
        total_d_naive = _sum(dt_naive)
        ocean_d_naive = _sum(dt_naive, ocean_naive)
        land_d_naive  = _sum(dt_naive, land_naive)
        nodes_ocean_d_naive = _sum(dt_naive, ocean_naive & nodes_valid_naive)

        total_d_tol = _sum(dt_tol)
        ocean_d_tol = _sum(dt_tol, ocean_tol)
        land_d_tol  = _sum(dt_tol, land_tol)
        nodes_ocean_d_tol = _sum(dt_tol, ocean_tol & nodes_valid_tol)

        total_b_naive = _sum(bt_naive)
        ocean_b_naive = _sum(bt_naive, ocean_naive)
        total_b_tol   = _sum(bt_tol)
        ocean_b_tol   = _sum(bt_tol, ocean_tol)

        # Shares
        share_ocean_d_naive = (ocean_d_naive / total_d_naive) if total_d_naive else 0.0
        share_ocean_d_tol   = (ocean_d_tol   / total_d_tol  ) if total_d_tol   else 0.0
        share_nodes_ocean_d_naive = (nodes_ocean_d_naive / total_d_naive) if total_d_naive else 0.0
        share_nodes_ocean_d_tol   = (nodes_ocean_d_tol   / total_d_tol  ) if total_d_tol   else 0.0

        # Shift test (naive alignment)
        log.info("STEP 3/4 — ±1 pixel shift test on adm0 (naive alignment)")
        shifts = [("none",0,0), ("E",1,0), ("W",-1,0), ("N",0,-1), ("S",0,1)]
        shift_rows = []
        for label, dx, dy in shifts:
            om = shift_mask(ocean_naive, dx, dy)
            f_d = _sum(dt_naive, om)
            f_b = _sum(bt_naive, om)
            share_d = (f_d / total_d_naive) if total_d_naive else 0.0
            shift_rows.append({"shift": label, "dx": dx, "dy": dy, "ocean_flux_drained": f_d, "ocean_share_drained": share_d, "ocean_flux_burned": f_b})
        df_shift = pd.DataFrame(shift_rows)
        df_shift.sort_values("ocean_share_drained", inplace=True)
        (out_dir / f"verify_shift_{interval}.csv").write_text(df_shift.to_csv(index=False))

        best_shift = df_shift.iloc[0]
        best_reduction = share_ocean_d_naive - best_shift["ocean_share_drained"]

        # Summary (CSV)
        summary = pd.DataFrame([{
            "interval": interval,
            "source": source,
            "unit_drained": unit_d,
            "unit_burned": unit_b,
            "total_flux_drained": total_d_naive,
            "ocean_share_drained__naive": share_ocean_d_naive,
            "ocean_share_drained__tolerant": share_ocean_d_tol,
            "nodes_valid_ocean_share_drained__naive": share_nodes_ocean_d_naive,
            "nodes_valid_ocean_share_drained__tolerant": share_nodes_ocean_d_tol,
            "total_flux_burned": total_b_naive,
            "ocean_share_burned__naive": (ocean_b_naive / total_b_naive) if total_b_naive else 0.0,
            "ocean_share_burned__tolerant": (ocean_b_tol / total_b_tol) if total_b_tol else 0.0,
            "best_shift_label": best_shift["shift"],
            "best_shift_dx": int(best_shift["dx"]),
            "best_shift_dy": int(best_shift["dy"]),
            "best_shift_ocean_share_drained": float(best_shift["ocean_share_drained"]),
            "best_shift_reduction_vs_naive": float(best_reduction),
        }])
        (out_dir / f"verify_summary_{interval}.csv").write_text(summary.to_csv(index=False))

        # -------------------- HUMAN-FRIENDLY REPORT --------------------
        log.info("STEP 4/4 — Report")
        log.info("  • Drained flux in adm0==0 (naive):      %s", format_pct(share_ocean_d_naive))
        log.info("  • Drained flux in adm0==0 (tolerant):   %s", format_pct(share_ocean_d_tol))
        log.info("  • With valid nodes>0 only (naive):      %s", format_pct(share_nodes_ocean_d_naive))
        log.info("  • With valid nodes>0 only (tolerant):   %s", format_pct(share_nodes_ocean_d_tol))
        log.info("  • Best 1-pixel shift: %s (dx=%+d, dy=%+d) → adm0==0 share = %s (Δ vs naive = %s)",
                 best_shift["shift"], int(best_shift["dx"]), int(best_shift["dy"]),
                 format_pct(best_shift["ocean_share_drained"]), format_pct(best_reduction))

        # ----------------------- DIAGNOSIS LOGIC -----------------------
        verdict_lines = []
        MISALIGN_THRESHOLD = 0.001  # 0.1% of drained flux
        big_ocean = share_ocean_d_naive > MISALIGN_THRESHOLD
        big_nodes_ocean = share_nodes_ocean_d_naive > MISALIGN_THRESHOLD
        collapsed_by_tolerant = (share_ocean_d_tol < share_ocean_d_naive * 0.2)  # 80% reduction
        collapsed_by_shift = (best_shift["ocean_share_drained"] < share_ocean_d_naive * 0.2)

        if big_ocean and (collapsed_by_tolerant or collapsed_by_shift) and big_nodes_ocean:
            verdict = "LIKELY GRID MISALIGNMENT (1-pixel snap)."
            verdict_lines.append("• Strong evidence of misalignment:")
            verdict_lines.append(f"  - Naive adm0==0 share is {format_pct(share_ocean_d_naive)}, with {format_pct(share_nodes_ocean_d_naive)} falling on valid nodes.")
            if collapsed_by_tolerant:
                verdict_lines.append(f"  - Tolerant reindex drops adm0==0 share to {format_pct(share_ocean_d_tol)}.")
            if collapsed_by_shift:
                verdict_lines.append(f"  - Shifting adm0 by {best_shift['shift']} collapses share to {format_pct(best_shift['ocean_share_drained'])}.")
            verdict_lines.append("• Next actions: switch to tolerance-based reindex; add preflight grid checks; optionally relabel coastal slivers to nearest country.")
        elif big_ocean and not (collapsed_by_tolerant or collapsed_by_shift):
            verdict = "ADM0 ZERO IS REAL (NOT ALIGNMENT) — flux likely outside land mask."
            verdict_lines.append("• Ocean totals do not shrink under tolerant alignment or ±1px shift.")
            verdict_lines.append("• Inspect source flux rasters near coasts/sea (possible leakage); consider applying a strict land mask.")
        else:
            verdict = "NO MATERIAL ISSUE DETECTED."
            verdict_lines.append("• Ocean share is negligible or already minimal under naive alignment.")
            verdict_lines.append("• Proceed; keep preflight checks to prevent future regressions.")

        log.info("")
        log.info("VERDICT: %s", verdict)
        for line in verdict_lines:
            log.info("  %s", line)
        log.info("")
        log.info("Files written: %s , %s", out_dir / f"verify_summary_{interval}.csv", out_dir / f"verify_shift_{interval}.csv")

    log.info("Done.")

if __name__ == "__main__":
    main()
