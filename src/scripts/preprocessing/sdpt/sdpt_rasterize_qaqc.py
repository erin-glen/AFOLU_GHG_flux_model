#!/usr/bin/env python3
"""
QAQC for SDPT partial rasters.

It searches:
  s3://<bucket>/<base_prefix>/*_pixels/<date>/<tile>__*__sdpt.tif

Checks per file:
  - dtype == uint8
  - CRS == EPSG:4326
  - nodata == 0
  - pixel size == 0.00025 (both x/y)
  - grid alignment to 0.00025 from (-180, -90)
  - value histogram (0,1,2)

Usage:
  python qaqc_sdpt_partials.py \
    --bucket gfw2-data \
    --base-prefix climate/AFOLU_flux_model/organic_soils/inputs/processed/sdpt \
    --date 20250808 \
    --tile 00N_110E
"""

import argparse
import sys
import math
from pathlib import PurePosixPath

import rasterio as rio
import numpy as np
import s3fs


RES = 0.00025
ORIGIN_X = -180.0
ORIGIN_Y = -90.0

def on_grid(v, o, r=RES, tol=1e-9):
    q = (v - o) / r
    return abs(q - round(q)) < 1e-6

def check_file(s3_url):
    ok = True
    msgs = []

    try:
        with rio.open(s3_url) as ds:
            # Basic metadata
            crs_ok = (ds.crs and ds.crs.to_string().upper() in ("EPSG:4326", "GEOGCS[\"WGS 84\"]"))
            dtype_ok = (ds.dtypes[0] == "uint8")
            nodata_ok = (ds.nodata == 0)
            # pixel size
            px = abs(ds.transform.a)
            py = abs(ds.transform.e)
            px_ok = abs(px - RES) < 1e-12
            py_ok = abs(py - RES) < 1e-12
            # alignment
            minx = ds.bounds.left
            miny = ds.bounds.bottom
            maxx = ds.bounds.right
            maxy = ds.bounds.top
            align_ok = all([
                on_grid(minx, ORIGIN_X),
                on_grid(maxx, ORIGIN_X),
                on_grid(miny, ORIGIN_Y),
                on_grid(maxy, ORIGIN_Y),
            ])

            # quick value histogram
            a = ds.read(1, masked=False)
            vals, counts = np.unique(a, return_counts=True)
            hist = dict(zip(vals.tolist(), counts.tolist()))

            if not crs_ok:   ok=False; msgs.append(f"CRS not EPSG:4326: {ds.crs}")
            if not dtype_ok: ok=False; msgs.append(f"dtype not uint8: {ds.dtypes[0]}")
            if not nodata_ok: ok=False; msgs.append(f"nodata not 0: {ds.nodata}")
            if not (px_ok and py_ok):
                ok=False; msgs.append(f"pixel size not {RES}: ({px}, {py})")
            if not align_ok:
                ok=False; msgs.append(f"grid alignment FAIL: bounds={ds.bounds}")

            # Build a concise report line
            report = {
                "crs_ok": crs_ok,
                "dtype_ok": dtype_ok,
                "nodata_ok": nodata_ok,
                "px_ok": px_ok,
                "py_ok": py_ok,
                "align_ok": align_ok,
                "hist": hist,
                "shape": (ds.height, ds.width),
                "transform": (ds.transform.a, ds.transform.b, ds.transform.c,
                              ds.transform.d, ds.transform.e, ds.transform.f),
            }
            return ok, report, msgs
    except Exception as e:
        return False, None, [f"open error: {e}"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, help="S3 bucket name")
    ap.add_argument("--base-prefix", required=True, help="Base prefix under the bucket")
    ap.add_argument("--date", required=True, help="YYYYMMDD date folder")
    ap.add_argument("--tile", required=True, help="Tile ID, e.g., 00N_110E")
    ap.add_argument("--dry-run", action="store_true", help="Only list matching keys")
    args = ap.parse_args()

    fs = s3fs.S3FileSystem(anon=False)

    # Search all *pixels folders (e.g., 8000_pixels) for this date+tile
    # Pattern: s3://bucket/base/*_pixels/date/<tile>__*__sdpt.tif
    base = PurePosixPath(args.base_prefix)
    glob_pat = f"{args.bucket}/{base}/*_pixels/{args.date}/{args.tile}__*__sdpt.tif"
    keys = fs.glob(glob_pat)

    if not keys:
        print("No matching partials found.")
        print("Looked under:", f"s3://{args.bucket}/{args.base_prefix}/*_pixels/{args.date}/")
        sys.exit(2)

    print(f"Found {len(keys)} partial(s):")
    for k in keys:
        print("  s3://"+k)

    if args.dry_run:
        return

    failures = 0
    for k in keys:
        s3_url = "s3://" + k
        ok, report, msgs = check_file(s3_url)
        print("\n---", s3_url)
        if ok:
            print("OK")
        else:
            print("FAIL")
            failures += 1
        if report:
            print("  crs_ok:", report["crs_ok"])
            print("  dtype_ok:", report["dtype_ok"])
            print("  nodata_ok:", report["nodata_ok"])
            print("  px_ok / py_ok:", report["px_ok"], report["py_ok"])
            print("  align_ok:", report["align_ok"])
            print("  shape (h,w):", report["shape"])
            a,b,c,d,e,f = report["transform"]
            print(f"  transform: a={a}, e={e}, c={c}, f={f}")
            print("  hist:", report["hist"])
        if msgs:
            for m in msgs:
                print("  note:", m)

    if failures:
        print(f"\nCompleted with {failures} failure(s).")
        sys.exit(1)
    else:
        print("\nAll checks passed.")
        sys.exit(0)

if __name__ == "__main__":
    main()
