#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alignment QA/QC (GTIFF-first + contextual-Zarr fallback) with scan mode.

Modes
-----
1) Single ROI (exact 1×1° bbox):
   - Reads the single drained_total + drained_state 1×1° GTIFF tile for the bbox.
   - Reads adm0 + pixel_area (10×10° GTIFF if available, else Zarr cropped to bbox).
   - Reports ocean-share metrics and a ±1px shift test; prints a verdict.

2) Scan mode (--scan-s3):
   - Discovers all 1×1° drained_total tiles on S3 for each interval.
   - Runs the same QA per tile concurrently (local threads).
   - Writes a CSV (if --out_csv) and prints a compact summary.

Verdicts
--------
OK
LIKELY MISALIGNMENT               (ocean share collapses with ±1px shift; nodes>0 ocean share non-trivial)
NOT ALIGNMENT (OCEAN FLUX)
ZERO FLUX
MISSING INPUTS / ERROR

Exit code
---------
- 1 if any interval/tile is LIKELY MISALIGNMENT (unless --no-fail)
- 0 otherwise
"""

from __future__ import annotations
import argparse
import math
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import s3fs
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds as tf_from_bounds
from rasterio.crs import CRS

EPSG4326 = CRS.from_epsg(4326)

# ----------------------------- Constants / Paths -----------------------------
ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
OUTPUT_BASE = "{root}/version_{model_version}"

# Organic-soils 1×1° inputs
TILES_1x1 = (
    OUTPUT_BASE
    + "/{folder}/{run_name}/five_year_intervals/{interval}/"
      "{tile_pixels}_pixels/{run_date}/"
)

# Contextual GeoTIFFs (10×10°); filenames may not include __W_S_E_N__ tokens
ADM0_GTIFF_FOLDER = (
    "s3://gfw2-data/gadm_administrative_boundaries/v4.1/"
    "v4.1.64__from_gfw-data-lake/raster/epsg-4326/10/40000/adm0/gdal-geotiff/"
)
PIXEL_AREA_GTIFF_FOLDER = (
    "s3://gfw2-data/analyses/umd_area_2013__from_gfw-data-lake/"
    "v1.10/raster/epsg-4326/10/40000/area_m/gdal-geotiff/"
)

# Contextual Zarr caches (from run script)
ADM0_ZARR = (
    "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/"
    "GADM4_1_adm0_global/20250604/global_GADM41_adm0_20250604.zarr"
)
PIXEL_AREA_ZARR = (
    "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/"
    "pixel_area/20250730/global_pixel_area_20250730.zarr"
)

DATASETS = {
    "drained_state_nodes": {"folder": "drained_state"},
    "drained_total": {
        "folder_candidates": [
            "drained_total_Mg_CO2e_ha_yr",
            "drained_total_Mg_CO2e_pixel_yr",
        ],
    },
}

# ----------------------------- Small utils ----------------------------------
def quiet_third_party(debug: bool):
    if debug:
        return
    import logging
    for name in ("botocore","aiobotocore","s3fs","fsspec","rasterio","urllib3","asyncio","xarray","zarr"):
        try:
            logging.getLogger(name).setLevel(logging.WARNING)
        except Exception:
            pass

def try_import_interval_pairs() -> Optional[List[Tuple[int, int]]]:
    try:
        from src.scripts.utilities import constants_and_names as cn  # type: ignore
        return list(cn.five_year_inventory_periods)
    except Exception:
        return None

def build_interval_pairs(end_years: List[int]) -> List[Tuple[int, int]]:
    mapping_src = try_import_interval_pairs()
    if mapping_src:
        m = {end: (start, end) for start, end in mapping_src}
        out = []
        for y in end_years:
            if y not in m:
                raise ValueError(f"Interval end year {y} not in model periods. Valid: {sorted(m)}")
            out.append(m[y])
        return out
    return [(y - 4, y) for y in end_years]

def s3fs_client():
    return s3fs.S3FileSystem(anon=False)

def s3_glob(fs: s3fs.S3FileSystem, pattern: str) -> List[str]:
    hits = fs.glob(pattern)
    return [("s3://"+h) if not h.startswith("s3://") else h for h in hits]

def s3_glob_one(fs: s3fs.S3FileSystem, pattern: str) -> Optional[str]:
    hits = s3_glob(fs, pattern)
    return hits[0] if hits else None

def bbox_to_ints(bbox: List[float]) -> Tuple[int,int,int,int]:
    W,S,E,N = bbox
    return (int(round(W)), int(round(S)), int(round(E)), int(round(N)))

def tile10_bounds_for(bbox: List[float]) -> Tuple[int,int,int,int]:
    W,S,E,N = bbox
    W10 = int(math.floor(W/10.0))*10
    S10 = int(math.floor(S/10.0))*10
    return (W10, S10, W10+10, S10+10)

def open_tiff_window(path: str, bbox: List[float], out_dtype=None, out_nodata=None):
    with rasterio.Env():
        with rasterio.open(path) as src:
            win = from_bounds(*bbox, transform=src.transform)
            data = src.read(1, window=win, boundless=True, masked=False)
            transform = src.window_transform(win)
            if out_dtype is not None and data.dtype != out_dtype:
                data = data.astype(out_dtype, copy=False)
            nodata = src.nodata if src.nodata is not None else out_nodata
            crs = src.crs or EPSG4326
            return data, transform, crs, nodata

def reproject_to_grid(src_arr, src_transform, src_crs, dst_shape, dst_transform,
                      dst_dtype, src_nodata=None, dst_nodata=None, resampling=Resampling.nearest):
    out = np.full(dst_shape, dst_nodata, dtype=dst_dtype) if dst_nodata is not None else np.zeros(dst_shape, dtype=dst_dtype)
    reproject(
        source=src_arr,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=src_crs,  # everything is EPSG:4326 here
        dst_nodata=dst_nodata,
        resampling=resampling,
        num_threads=2,
    )
    return out

def resolve_1x1_tile(fs, base_kwargs: Dict, bbox: List[float], folder: str) -> Optional[str]:
    W,S,E,N = bbox_to_ints(bbox)
    prefix = TILES_1x1.format(folder=folder, **base_kwargs).rstrip("/")
    pat = f"{prefix}/**/*__{W}_{S}_{E}_{N}__*.tif"
    return s3_glob_one(fs, pat)

def resolve_flux_1x1(fs, base_kwargs: Dict, bbox: List[float]) -> Tuple[Optional[str], Optional[str]]:
    for folder in DATASETS["drained_total"]["folder_candidates"]:
        p = resolve_1x1_tile(fs, base_kwargs, bbox, folder)
        if p:
            unit = "ha" if folder.endswith("_ha_yr") else "pixel"
            return p, unit
    return None, None

def resolve_10x10_tile(fs, base_folder: str, bbox: List[float]) -> Optional[str]:
    """Try explicit numeric-token pattern first; if missing, return None (we'll use Zarr)."""
    W10,S10,E10,N10 = tile10_bounds_for(bbox)
    base = base_folder.rstrip("/")
    pat = f"{base}/**/*__{W10}_{S10}_{E10}_{N10}__*.tif"
    return s3_glob_one(fs, pat)

def nansum(a: np.ndarray, mask: Optional[np.ndarray]=None) -> float:
    if mask is not None:
        a = np.where(mask, a, np.nan)
    return float(np.nansum(a, dtype=np.float64))

def shift_mask(m: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.full_like(m, False)
    xs = slice(max(0, dx), m.shape[1]+min(0, dx))
    ys = slice(max(0, dy), m.shape[0]+min(0, dy))
    xs2 = slice(max(0, -dx), m.shape[1]+min(0, -dx))
    ys2 = slice(max(0, -dy), m.shape[0]+min(0, -dy))
    out[ys2, xs2] = m[ys, xs]
    return out

# ---------------------- Contextual Zarr fallback helpers ---------------------
def first_xy_var(ds) -> "xarray.DataArray":
    import xarray as xr
    if isinstance(ds, xr.DataArray):
        da_ = ds
    else:
        vars_xy = [v for v in ds.data_vars.values() if {"x", "y"}.issubset(v.dims)]
        da_ = vars_xy[0] if vars_xy else next(iter(ds.data_vars.values()))
    if "band" in da_.dims:
        da_ = da_.isel(band=0, drop=True)
    return da_

def open_contextual_from_zarr(zarr_path: str, bbox: List[float], as_dtype, dst_nodata):
    """
    Open a small window from a contextual Zarr by x/y slice over bbox.
    Returns (arr_np, src_transform). CRS is EPSG:4326 implicit.
    """
    import xarray as xr
    with rasterio.Env():
        ds = xr.open_zarr(zarr_path, consolidated=None, storage_options={"anon": False})
    da = first_xy_var(ds)

    west, south, east, north = bbox
    x = da.coords["x"]; y = da.coords["y"]
    x_asc = bool(x[0] < x[-1]); y_asc = bool(y[0] < y[-1])

    x_slice = slice(min(west, east), max(west, east)) if x_asc else slice(max(east, west), min(east, west))
    y_slice = slice(min(south, north), max(south, north)) if y_asc else slice(max(north, south), min(north, south))

    tile = da.sel(x=x_slice, y=y_slice)
    arr = tile.astype(as_dtype).data
    if hasattr(arr, "compute"):
        arr = arr.compute()
    arr = np.asarray(arr)

    xc = tile.coords["x"].values
    yc = tile.coords["y"].values
    nx, ny = xc.size, yc.size
    if nx == 0 or ny == 0:
        return arr, None  # empty

    dx = float(abs(xc[-1] - xc[0]) / (nx - 1)) if nx > 1 else 0.00025
    dy = float(abs(yc[-1] - yc[0]) / (ny - 1)) if ny > 1 else 0.00025

    west_edge = float(xc.min() - dx/2.0)
    east_edge = float(xc.max() + dx/2.0)
    if yc[0] > yc[-1]:  # descending (north->south)
        north_edge = float(yc[0] + dy/2.0)
        south_edge = float(yc[-1] - dy/2.0)
    else:               # ascending
        north_edge = float(yc[-1] + dy/2.0)
        south_edge = float(yc[0] - dy/2.0)

    transform = tf_from_bounds(west_edge, south_edge, east_edge, north_edge, nx, ny)
    return arr, transform

# ----------------------------- Core QA --------------------------------------
def analyze_bbox(interval: str, bbox: List[float], args) -> Dict:
    """Run QA for a single 1×1° bbox. Returns one row dict."""
    end_year = int(interval.split("_")[1])
    base_kwargs = dict(root=ROOT, model_version=args.model_version,
                       run_name=args.run_name, interval=interval,
                       tile_pixels=args.tile_pixels, run_date=args.run_date)
    fs = s3fs_client()

    dn_path = resolve_1x1_tile(fs, base_kwargs, bbox, DATASETS["drained_state_nodes"]["folder"])
    dt_path, dt_unit = resolve_flux_1x1(fs, base_kwargs, bbox)
    adm0_path = resolve_10x10_tile(fs, ADM0_GTIFF_FOLDER, bbox)
    area_path = resolve_10x10_tile(fs, PIXEL_AREA_GTIFF_FOLDER, bbox)

    row = {
        "interval_end": end_year,
        "interval_span": interval,
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "source": "GTIFF+CTX",
        "unit_d": dt_unit or "",
        "ocean_share_drained__naive": np.nan,
        "nodes_ocean_share_drained__naive": np.nan,
        "best_shift_label": "",
        "best_shift_share": np.nan,
        "best_shift_reduction": np.nan,
        "verdict": "MISSING INPUTS",
        "notes": "",
    }

    if not dn_path or not dt_path:
        row["notes"] = "Missing drained_state or drained_total 1×1° tile."
        return row

    # Read drained_total (target grid)
    dt_data, dt_transform, dt_crs, _ = open_tiff_window(dt_path, bbox, out_dtype=np.float32, out_nodata=np.nan)
    H, W = dt_data.shape
    if H == 0 or W == 0:
        row["notes"] = "Empty drained_total window."
        return row

    # Read state nodes; align (nearest)
    dn_data_src, dn_transform, dn_crs, dn_nodata = open_tiff_window(dn_path, bbox, out_dtype=np.int32, out_nodata=0)
    if (dn_transform != dt_transform) or (dn_crs != dt_crs) or (dn_data_src.shape != dt_data.shape):
        dn_data = reproject_to_grid(dn_data_src, dn_transform, dn_crs,
                                    dt_data.shape, dt_transform,
                                    dst_dtype=np.int32, src_nodata=dn_nodata, dst_nodata=0,
                                    resampling=Resampling.nearest)
    else:
        dn_data = dn_data_src

    # adm0 (GTIFF or Zarr)
    if adm0_path:
        adm0_src, adm0_transform, adm0_crs, adm0_nodata = open_tiff_window(adm0_path, bbox, out_dtype=np.int32, out_nodata=0)
        adm0 = reproject_to_grid(adm0_src, adm0_transform, adm0_crs,
                                 dt_data.shape, dt_transform,
                                 dst_dtype=np.int32, src_nodata=adm0_nodata, dst_nodata=0,
                                 resampling=Resampling.nearest)
    else:
        adm0_src, adm0_transform = open_contextual_from_zarr(ADM0_ZARR, bbox, as_dtype=np.int32, dst_nodata=0)
        if adm0_transform is None or adm0_src.size == 0:
            row["notes"] = "adm0 GTIFF missing and Zarr empty."
            return row
        adm0 = reproject_to_grid(adm0_src, adm0_transform, EPSG4326,
                                 dt_data.shape, dt_transform,
                                 dst_dtype=np.int32, src_nodata=0, dst_nodata=0,
                                 resampling=Resampling.nearest)

    # Convert per-ha → per-pixel totals if needed
    if dt_unit == "ha":
        if area_path:
            area_src, area_transform, area_crs, area_nodata = open_tiff_window(area_path, bbox, out_dtype=np.float32, out_nodata=np.nan)
            area_crs = area_crs or EPSG4326
        else:
            area_src, area_transform = open_contextual_from_zarr(PIXEL_AREA_ZARR, bbox, as_dtype=np.float32, dst_nodata=np.nan)
            area_nodata = np.nan
            area_crs = EPSG4326
            if area_transform is None or area_src.size == 0:
                row["notes"] = "pixel_area GTIFF missing and Zarr empty."
                return row

        area = reproject_to_grid(area_src, area_transform, area_crs,
                                 dt_data.shape, dt_transform,
                                 dst_dtype=np.float32, src_nodata=area_nodata, dst_nodata=np.nan,
                                 resampling=Resampling.bilinear)
        dt_data = dt_data * (area / 10000.0)  # m² → ha
        dt_data = np.where(np.isfinite(area) & (area > 0), dt_data, np.nan)

    # Metrics
    total_d = nansum(dt_data)
    if total_d == 0.0:
        row.update({
            "ocean_share_drained__naive": 0.0,
            "nodes_ocean_share_drained__naive": 0.0,
            "verdict": "ZERO FLUX",
            "notes": "Drained flux total is zero in ROI/interval.",
        })
        return row

    ocean_mask = (adm0 <= 0) | ~np.isfinite(adm0)
    nodes_valid = (dn_data != 0)

    ocean_total = nansum(dt_data, ocean_mask)
    nodes_ocean_total = nansum(dt_data, ocean_mask & nodes_valid)

    share_ocean = ocean_total / total_d
    share_nodes_ocean = nodes_ocean_total / total_d

    # ±1px shift test (naive mask)
    shifts = [("none", 0, 0), ("E", 1, 0), ("W", -1, 0), ("N", 0, -1), ("S", 0, 1)]
    best_label, best_share = "none", share_ocean
    for label, dx, dy in shifts:
        s = nansum(dt_data, shift_mask(ocean_mask, dx, dy)) / total_d
        if s < best_share:
            best_share, best_label = s, label

    row.update({
        "ocean_share_drained__naive": share_ocean,
        "nodes_ocean_share_drained__naive": share_nodes_ocean,
        "best_shift_label": best_label,
        "best_shift_share": best_share,
        "best_shift_reduction": share_ocean - best_share,
    })

    MISALIGN_THRESHOLD = 0.001  # 0.1%
    big_ocean = share_ocean > MISALIGN_THRESHOLD
    big_nodes = share_nodes_ocean > MISALIGN_THRESHOLD
    collapsed_by_shift = best_share < share_ocean * 0.2

    if big_ocean and big_nodes and collapsed_by_shift:
        row["verdict"] = "LIKELY MISALIGNMENT"
        row["notes"] = "Ocean share collapses with ±1px shift; nodes>0 ocean share non-trivial."
    elif big_ocean and not collapsed_by_shift:
        row["verdict"] = "NOT ALIGNMENT (OCEAN FLUX)"
        row["notes"] = "Ocean share persists under shift. Inspect flux/land mask."
    else:
        row["verdict"] = "OK"
        row["notes"] = "Ocean share negligible or already minimal."

    return row

# ----------------------------- Scan helpers ---------------------------------
TOKEN_RE = re.compile(r"__(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)__")

def discover_bboxes_for_interval(fs, interval: str, args) -> List[List[float]]:
    """Find all 1×1° bboxes that have drained_total tiles for this interval."""
    base_kw = dict(root=ROOT, model_version=args.model_version,
                   run_name=args.run_name, interval=interval,
                   tile_pixels=args.tile_pixels, run_date=args.run_date)
    boxes: set[Tuple[int,int,int,int]] = set()
    # Prefer HA, otherwise pixel
    for folder in DATASETS["drained_total"]["folder_candidates"]:
        prefix = TILES_1x1.format(folder=folder, **base_kw).rstrip("/")
        # Try non-recursive first (common layout); fall back to '**/*.tif' if needed
        try:
            keys = fs.ls(prefix)
        except Exception:
            keys = []
        if not keys:
            keys = fs.glob(prefix + "/*.tif")
        if not keys:
            continue
        for k in keys:
            m = TOKEN_RE.search(k)
            if not m:
                continue
            W,S,E,N = map(int, m.groups())
            boxes.add((W,S,E,N))
        if boxes:
            break
    # Convert to float bbox lists
    return [[float(W), float(S), float(E), float(N)] for (W,S,E,N) in sorted(boxes)]

# ----------------------------- CLI / Runner ---------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alignment QA/QC for adm0==0 totals (organic-soils) — single bbox or scan mode")
    p.add_argument("--model_version", required=True, help="e.g., 0_7_0")
    p.add_argument("--run_date", required=True, help="YYYYMMDD used in S3 folder structure")
    p.add_argument("--interval_end_years", nargs="+", type=int, required=True, help="e.g., 2005 2010 2015 2020 2024")
    p.add_argument("--run_name", default="ogh_standard_model", help="Run name used in S3 paths")
    p.add_argument("--tile_pixels", type=int, default=4000, help="4000 (1×1°) or 40000 (10×10°)")
    p.add_argument("--chunk_size", type=int, default=10000, help="(unused; kept for compat)")
    p.add_argument("--bounding_box", nargs=4, type=float, help="W S E N (single-ROI mode)")
    p.add_argument("--scan-s3", action="store_true", help="Scan S3 for all 1×1° drained_total tiles and run QA on each")
    p.add_argument("--max_workers", type=int, default=8, help="Concurrency for scan mode (threads)")
    p.add_argument("--sample_pct", type=float, default=1.0, help="Fraction of tiles to sample in scan mode (0<..<=1)")
    p.add_argument("--limit", type=int, help="Hard cap on number of tiles to process (scan mode)")
    p.add_argument("--out_csv", help="Write results to this CSV (scan mode)")
    p.add_argument("--verbose", action="store_true", help="Print discovered keys in single-ROI mode")
    p.add_argument("--no-fail", action="store_true", help="Always exit 0 (useful in dashboards)")
    p.add_argument("--debug", action="store_true", help="Show 3rd-party debug logs")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    quiet_third_party(args.debug)

    pairs = build_interval_pairs(args.interval_end_years)
    intervals = [f"{s}_{e}" for (s, e) in pairs]

    if not args.scan_s3:
        # ---- single-ROI mode ----
        if not args.bounding_box:
            print("ERROR: single-ROI mode requires --bounding_box W S E N", file=sys.stderr)
            sys.exit(2)
        print("\n=== Alignment QA/QC (single ROI) ===")
        print(f"  model_version={args.model_version}  run_date={args.run_date}  run_name={args.run_name}")
        print(f"  ROI bbox = [W={args.bounding_box[0]}, S={args.bounding_box[1]}, E={args.bounding_box[2]}, N={args.bounding_box[3]}]")
        print(f"  tile_pixels={args.tile_pixels}\n")

        rows = []
        for itv in intervals:
            try:
                rows.append(analyze_bbox(itv, [float(x) for x in args.bounding_box], args))
            except Exception as e:
                rows.append({
                    "interval_end": int(itv.split("_")[1]),
                    "interval_span": itv,
                    "bbox": ",".join(map(str, args.bounding_box)),
                    "source": "GTIFF+CTX",
                    "unit_d": "",
                    "ocean_share_drained__naive": np.nan,
                    "nodes_ocean_share_drained__naive": np.nan,
                    "best_shift_label": "",
                    "best_shift_share": np.nan,
                    "best_shift_reduction": np.nan,
                    "verdict": "ERROR",
                    "notes": f"{type(e).__name__}: {e}",
                })

        df = pd.DataFrame(rows).sort_values(["interval_end","bbox"])

        def pct(x): return "" if pd.isna(x) else f"{100*x:.4f}%"
        show = df.copy()
        for c in ("ocean_share_drained__naive","nodes_ocean_share_drained__naive","best_shift_share","best_shift_reduction"):
            show[c] = show[c].map(pct)
        cols = ["interval_end","interval_span","bbox","source","unit_d",
                "ocean_share_drained__naive","nodes_ocean_share_drained__naive",
                "best_shift_label","best_shift_share","best_shift_reduction","verdict","notes"]
        print("=== Summary ===")
        print(show[cols].to_string(index=False))

        if not args.no_fail and (df["verdict"] == "LIKELY MISALIGNMENT").any():
            print("\nQAQC: FAIL — alignment likely off (see 'verdict').")
            sys.exit(1)
        print("\nQAQC: PASS")
        sys.exit(0)

    # ---- scan mode ----
    print("\n=== Alignment QA/QC (scan mode) ===")
    print(f"  model_version={args.model_version}  run_date={args.run_date}  run_name={args.run_name}")
    print(f"  tile_pixels={args.tile_pixels}  max_workers={args.max_workers}  sample_pct={args.sample_pct}  limit={args.limit}\n")

    fs = s3fs_client()
    all_rows = []

    for itv in intervals:
        print(f"[{itv}] Discovering tiles ...")
        try:
            bboxes = discover_bboxes_for_interval(fs, itv, args)
        except Exception as e:
            print(f"  ! Discovery error: {e}")
            bboxes = []

        if not bboxes:
            print("  (no drained_total tiles found)")
            continue

        # Optional sampling/limit
        bboxes = bboxes if args.sample_pct >= 1.0 else bboxes[::max(1, int(1/args.sample_pct))]
        if args.limit:
            bboxes = bboxes[:args.limit]

        print(f"  -> {len(bboxes)} 1×1° tiles to check")

        rows = []
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
            futs = {ex.submit(analyze_bbox, itv, bb, args): tuple(bb) for bb in bboxes}
            for fut in as_completed(futs):
                try:
                    rows.append(fut.result())
                except Exception as e:
                    bb = futs[fut]
                    rows.append({
                        "interval_end": int(itv.split("_")[1]),
                        "interval_span": itv,
                        "bbox": f"{bb[0]},{bb[1]},{bb[2]},{bb[3]}",
                        "source": "GTIFF+CTX",
                        "unit_d": "",
                        "ocean_share_drained__naive": np.nan,
                        "nodes_ocean_share_drained__naive": np.nan,
                        "best_shift_label": "",
                        "best_shift_share": np.nan,
                        "best_shift_reduction": np.nan,
                        "verdict": "ERROR",
                        "notes": f"{type(e).__name__}: {e}",
                    })

        df_itv = pd.DataFrame(rows).sort_values(["interval_end","bbox"])
        all_rows.extend(rows)

        # Per-interval quick tally
        tally = df_itv["verdict"].value_counts().to_dict()
        print(f"  Tally: {tally}")

    if not all_rows:
        print("No results.")
        sys.exit(0 if args.no_fail else 1)

    df_all = pd.DataFrame(all_rows).sort_values(["interval_end","bbox"])

    if args.out_csv:
        df_all.to_csv(args.out_csv, index=False)
        print(f"\nWrote: {args.out_csv}")

    # Compact console view
    def pct(x): return "" if pd.isna(x) else f"{100*x:.4f}%"
    show = df_all.copy()
    for c in ("ocean_share_drained__naive","nodes_ocean_share_drained__naive","best_shift_share","best_shift_reduction"):
        show[c] = show[c].map(pct)
    cols = ["interval_end","interval_span","bbox","verdict","ocean_share_drained__naive","nodes_ocean_share_drained__naive","best_shift_label","best_shift_share","notes"]
    print("\n=== Summary (first 40 rows) ===")
    print(show[cols].head(40).to_string(index=False))

    if not args.no_fail and (df_all["verdict"] == "LIKELY MISALIGNMENT").any():
        print("\nQAQC: FAIL — one or more tiles show likely alignment issues.")
        sys.exit(1)

    print("\nQAQC: PASS")
    sys.exit(0)

if __name__ == "__main__":
    main()
