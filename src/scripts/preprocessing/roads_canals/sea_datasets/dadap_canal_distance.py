#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build distance-to-canal rasters from the Dadap 5 m canal mask.

The drainage model historically consumed the Dadap canal product as a 1 km
*density* surface and flagged any pixel with density > 0 as drained, independent
of the ``--drainage_distance_threshold_m`` sweep. That left insular SE Asia -- the
dominant global peat source -- unresponsive to the drainage-distance sensitivity.

This script converts the original Dadap 5 m binary canal mask into a
*distance-to-nearest-canal* surface in the SAME format as the OSM/GRIP distance
products (float32 metres, nodata 0, canal cells = 1.0, optionally clipped at 1 km),
so the model can treat Dadap exactly like ``osm_canals``/``grip`` and the distance
sweep applies in SE Asia too. Output matches the OSM/GRIP convention:

    s3://gfw2-data/.../dadap_density/distance/40000_pixels/<date>/
        {tile_id}__dadap_canals_distance.tif

Method mirrors ``1_2_distance_from_presence_mosaic.py``:
- Process each 10x10 tile in 1-degree chunks with a halo (default 1000 m) so the
  Euclidean distance is seamless across chunk and tile edges.
- Warp the 5 m binary canal mask onto the Hansen 0.00025 deg grid with
  resampleAlg="max" so ANY 5 m canal in a 30 m cell marks canal presence.
  IMPORTANT: do NOT pass dstNodata=0 on a 0/1 mask -- GDAL then treats output 0 as
  nodata during max-resampling and marks every cell as canal. Use srcNodata=255.
- Euclidean distance in METRES with latitude-aware pixel sizes, clipped at maxdist.
- Canal cells set to distance 1.0 (0 is reserved for nodata), per the OSM convention.

Source 5 m canals (Dadap et al. 2021, AGU Advances; Stanford Digital Repository
DOI 10.25740/yj761xk5815, CC-BY-ND): a single VRT/mosaic of the 1664 zoom-15
binary tiles (EPSG:3857). Pass it via --canal_source (local VRT or s3/vsis3 path).

CLI examples
------------
# One 1-degree chunk (fast smoke test), write a standalone GeoTIFF, no upload:
python -m src.scripts.preprocessing.roads_canals.sea_datasets.dadap_canal_distance \
  --tile_id 00N_110E --chunk_bounds "113,-3,114,-2" \
  --canal_source C:/tmp/afolu/dadap_5m/dadap_canals_5m.vrt --no_upload --out_dir C:/tmp/afolu/dadap_5m/out

# All Dadap-overlapping tiles, assemble full 40000 px tiles, upload to S3:
python -m src.scripts.preprocessing.roads_canals.sea_datasets.dadap_canal_distance \
  --canal_source s3://gfw2-data/.../dadap_canals_5m.vrt --date 20260602
"""
from __future__ import annotations

import argparse
import math
import os
import posixpath
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds as window_from_bounds
from osgeo import gdal
from scipy.ndimage import distance_transform_edt

import src.scripts.preprocessing.preprocessing_constants as pc
from src.scripts.utilities import universal_utilities as uutil

gdal.UseExceptions()

RES = 0.00025                       # Hansen grid degrees (~30 m)
TILE_PX = 40000                     # pixels per 10 deg tile edge
M_PER_DEG_LAT = 111132.0
M_PER_DEG_LON = 111320.0
CANAL_DISTANCE_VALUE = 1.0          # canal cells store 1.0 (0 reserved for nodata)
DEFAULT_DATE = "20260602"

# 10x10 tiles overlapping the Dadap extent (lon 95-119E, lat -5..6N).
DEFAULT_TILES = [
    "10N_090E", "10N_100E", "10N_110E",
    "00N_090E", "00N_100E", "00N_110E",
]


def meters_per_pixel(lat_center_deg: float) -> Tuple[float, float]:
    lat_rad = math.radians(lat_center_deg)
    pix_h_m = RES * M_PER_DEG_LAT
    pix_w_m = max(RES * M_PER_DEG_LON * math.cos(lat_rad), 1e-6)
    return pix_h_m, pix_w_m


def chunk_iter(tile_bounds, chunk_size=1.0):
    minx, miny, maxx, maxy = tile_bounds
    eps = 1e-9
    y = miny
    while y < maxy - eps:
        x = minx
        while x < maxx - eps:
            yield [x, y, min(x + chunk_size, maxx), min(y + chunk_size, maxy)]
            x += chunk_size
        y += chunk_size


def warp_canal_presence(canal_source: str, bounds) -> Tuple[np.ndarray, tuple]:
    """Warp the 0/1 canal mask onto the Hansen grid over `bounds` (max-resampled).

    Returns (presence_uint8, geotransform). Areas outside the canal source are 0.
    """
    W, S, E, N = bounds
    ds = gdal.Warp(
        "", canal_source, format="MEM", dstSRS="EPSG:4326",
        outputBounds=[W, S, E, N], xRes=RES, yRes=RES,
        targetAlignedPixels=True, resampleAlg="max",
        srcNodata=255, outputType=gdal.GDT_Byte,  # no dstNodata: keep 0 as valid "no canal"
    )
    arr = ds.ReadAsArray()
    gt = ds.GetGeoTransform()
    return arr, gt


def distance_for_chunk(canal_source, chunk_bounds, halo_m, maxdist_m):
    """Compute distance-to-canal (m) for one 1-deg chunk, with halo for seamless edges.

    Returns (dist_interior_float32, interior_bounds) or (None, None) if no canals
    anywhere in the haloed window.
    """
    cminx, cminy, cmaxx, cmaxy = chunk_bounds
    lat_center = 0.5 * (cminy + cmaxy)
    pix_h_m, pix_w_m = meters_per_pixel(lat_center)

    lat_pad = math.ceil(halo_m / pix_h_m) * RES
    lon_pad = math.ceil(halo_m / pix_w_m) * RES
    exp_bounds = [cminx - lon_pad, cminy - lat_pad, cmaxx + lon_pad, cmaxy + lat_pad]

    presence, gt = warp_canal_presence(canal_source, exp_bounds)
    if presence is None or presence.max() == 0:
        return None, None

    inv = (presence == 0)
    dist = distance_transform_edt(inv, sampling=(pix_h_m, pix_w_m)).astype(np.float32)
    if maxdist_m and maxdist_m > 0:
        dist = np.minimum(dist, np.float32(maxdist_m))

    # crop to the interior chunk using the warped (expanded) transform
    exp_transform = rasterio.transform.from_origin(gt[0], gt[3], RES, RES)
    win = window_from_bounds(cminx, cminy, cmaxx, cmaxy, transform=exp_transform)
    r0, c0 = int(round(win.row_off)), int(round(win.col_off))
    h = int(round(win.height)); w = int(round(win.width))
    dist_i = dist[r0:r0 + h, c0:c0 + w]
    pres_i = presence[r0:r0 + h, c0:c0 + w]

    # canal cells -> distance 1.0 (0 reserved for nodata, matching OSM/GRIP)
    dist_i = np.where(pres_i > 0, CANAL_DISTANCE_VALUE, dist_i).astype(np.float32)
    return dist_i, [cminx, cminy, cmaxx, cmaxy]


def process_tile(canal_source, tile_id, out_dir, halo_m, maxdist_m, upload, date_str):
    W, S, E, N = uutil.get_10x10_tile_bounds(tile_id)
    out_name = f"{tile_id}__dadap_canals_distance.tif"
    local_out = os.path.join(out_dir, out_name)
    transform = from_origin(W, N, RES, RES)
    profile = dict(driver="GTiff", height=TILE_PX, width=TILE_PX, count=1,
                   dtype="float32", crs="EPSG:4326", transform=transform,
                   nodata=0.0, compress="deflate", tiled=True,
                   blockxsize=400, blockysize=400, BIGTIFF="IF_SAFER")

    n_written = 0
    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(local_out, "w", **profile) as dst:
        for cb in chunk_iter([W, S, E, N], 1.0):
            dist_i, ib = distance_for_chunk(canal_source, cb, halo_m, maxdist_m)
            if dist_i is None:
                continue
            win = window_from_bounds(ib[0], ib[1], ib[2], ib[3], transform=transform)
            r0, c0 = int(round(win.row_off)), int(round(win.col_off))
            dst.write(dist_i, 1, window=Window(c0, r0, dist_i.shape[1], dist_i.shape[0]))
            n_written += 1

    if n_written == 0:
        os.remove(local_out)
        print(f"[{tile_id}] no canals in tile; nothing written.")
        return None

    print(f"[{tile_id}] wrote {n_written} chunk(s) -> {local_out}")
    if upload:
        s3_dir = posixpath.join(pc.datasets["dadap"]["s3_distance_base"],
                                "40000_pixels", date_str)
        s3_key = posixpath.join(s3_dir, out_name)
        uutil.upload_file_to_s3(local_out, pc.s3_bucket_name, s3_key)
        print(f"[{tile_id}] uploaded -> s3://{pc.s3_bucket_name}/{s3_key}")
    return local_out


def main():
    p = argparse.ArgumentParser(description="Dadap 5 m canals -> distance-to-canal (model distance format).")
    p.add_argument("--canal_source", required=True,
                   help="VRT/mosaic of the Dadap 5 m binary canal tiles (local path or /vsis3/ or s3://).")
    p.add_argument("--tile_id", help="Single 10x10 tile (default: all Dadap-overlapping tiles).")
    p.add_argument("--chunk_bounds", help="Optional single 1-deg chunk 'minx,miny,maxx,maxy' (smoke test).")
    p.add_argument("--date", default=DEFAULT_DATE, help=f"Output date folder. Default {DEFAULT_DATE}.")
    p.add_argument("--halo_m", type=float, default=1000.0, help="Halo radius (m) for seamless edges.")
    p.add_argument("--maxdist", type=float, default=1000.0, help="Clip distances at this many metres (<=0 disables).")
    p.add_argument("--out_dir", default=tempfile.gettempdir(), help="Local output directory.")
    p.add_argument("--no_upload", action="store_true", help="Do not upload to S3 (local only).")
    args = p.parse_args()

    canal = args.canal_source
    if canal.startswith("s3://"):
        canal = "/vsis3/" + canal[len("s3://"):]
    upload = not args.no_upload
    maxdist = args.maxdist if args.maxdist and args.maxdist > 0 else None

    if args.chunk_bounds:
        if not args.tile_id:
            raise SystemExit("--chunk_bounds requires --tile_id")
        cb = [float(x) for x in args.chunk_bounds.split(",")]
        dist_i, ib = distance_for_chunk(canal, cb, args.halo_m, maxdist)
        os.makedirs(args.out_dir, exist_ok=True)
        out = os.path.join(args.out_dir, f"{args.tile_id}__dadap_canals_distance__chunk.tif")
        if dist_i is None:
            print("No canals in chunk window; nothing written."); return
        tr = from_origin(ib[0], ib[3], RES, RES)
        prof = dict(driver="GTiff", height=dist_i.shape[0], width=dist_i.shape[1], count=1,
                    dtype="float32", crs="EPSG:4326", transform=tr, nodata=0.0,
                    compress="deflate", tiled=True)
        with rasterio.open(out, "w", **prof) as d:
            d.write(dist_i, 1)
        print(f"[{args.tile_id}] chunk distance -> {out}  "
              f"(canal cells={int((dist_i==CANAL_DISTANCE_VALUE).sum())}, "
              f"max={float(dist_i.max()):.0f} m)")
        return

    tiles = [args.tile_id] if args.tile_id else DEFAULT_TILES
    for t in tiles:
        process_tile(canal, t, args.out_dir, args.halo_m, maxdist, upload, args.date)


if __name__ == "__main__":
    main()
