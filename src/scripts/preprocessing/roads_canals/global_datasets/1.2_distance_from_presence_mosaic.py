#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1.2_distance_from_presence_mosaic.py
Stage-2 distance computation from 30 m binary presence rasters (roads/canals).

Key behavior
------------
- Reads *presence* GeoTIFF chunks (1° x 1°) previously produced on the Hansen grid.
- For each 1° chunk, builds a presence mosaic with a configurable HALO that
  can extend into neighboring *tiles* so distance is seamless across 10° tile edges.
- Computes Euclidean distance in METERS using latitude-aware pixel sizes.
- Crops back to the interior 1° chunk and masks to peat (union 30 m).
- Writes GeoTIFF locally, then uses the same hansenize/upload pattern as your pipeline.

Assumptions matched to your logs
--------------------------------
- Presence rasters are written under:
    s3://{cn.s3_bucket_name}/<s3_processed_base>/<chunk_px>_pixels/<date>/
  with filenames:
    {tile_id}__{minx}_{miny}_{maxx}_{maxy}__{feature_type}_presence.tif
  e.g.
    00N_110E__116.0_-4.0_117.0_-3.0__osm_roads_presence.tif
- Vectors used to generate presence were shapefiles (EPSG:3395 or EPSG:4326); this
  script never touches vectors. It only consumes the presence rasters.
- Union 30 m peat mask tiles live at:
    /vsis3/{cn.s3_bucket_name}/{cn.datasets["peat"]["union_mask"]["30m"]}{tile_id}_union_mask.tif
- Hansen grid is geographic (~0.00025 deg), so we compute meters per pixel from latitude.

CLI examples
------------
# Compute distance for a single 1° chunk (with 1 km halo) within a tile:
python -m src.scripts.preprocessing.roads_canals.global_datasets.distance_from_presence_mosaic \
  --tile_id 00N_110E --feature_type osm_roads --client coiled \
  --chunk_bounds "116,-4,117,-3" --halo_m 1000 --date 20251113 --loglevel INFO

# Compute distance for *all* 1° chunks of a tile (default halo 1000 m):
python -m src.scripts.preprocessing.roads_canals.global_datasets.distance_from_presence_mosaic \
  --tile_id 00N_110E --feature_type osm_roads --client coiled \
  --date 20251113 --batch_size 20 --loglevel INFO
"""

import os
import math
import gc
import logging
import posixpath
import tempfile
from typing import List, Optional, Tuple

import boto3
import numpy as np
import dask
from dask import delayed

import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.transform import Affine
import rioxarray as rxr
import xarray as xr

try:
    from scipy.ndimage import distance_transform_edt
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

from src.scripts.utilities import universal_utilities as uutil
import src.scripts.preprocessing.preprocessing_constants as cn
from src.scripts.preprocessing.hansenize.hansenize_coiled import warp_to_hansen_coiled

LOG = logging.getLogger("rc_distance")

PEAT_30M_PATTERN = "_union_mask.tif"


# ------------------------------ Utilities ------------------------------

def _fmt_deg(x: float) -> str:
    # filenames in presence use one decimal (e.g., 116.0_-4.0_117.0_-3.0)
    return f"{float(x):.1f}"

def _chunk_bounds_iter(tile_bounds: Tuple[float, float, float, float], chunk_size: float = 1.0):
    minx, miny, maxx, maxy = tile_bounds
    y = miny
    eps = 1e-12
    while y < maxy - eps:
        x = minx
        while x < maxx - eps:
            yield [x, y, x + chunk_size, y + chunk_size]
            x += chunk_size
        y += chunk_size

def _tile_id_from_cell(lon_w: int, lat_s: int) -> str:
    """
    Infer 10x10 tile_id using the conventions evident in your logs:
    - tile bounds example: 00N_110E => [110, -10, 120, 0]
      => north edge = 0, west edge = 110.
    Given a 1° cell [lon_w, lat_s, lon_w+1, lat_s+1], the 10° tile north edge is ceil(lat_s+1 to next 10),
    the west edge is floor(lon_w to previous 10). But in practice, presence filenames use the tile_id
    that *contains* the 1° cell. A simpler rule that matches your 00N_110E case:
      - tile north edge is (lat_s + 1) rounded up to the next multiple of 10.
      - tile west edge is lon_w rounded down to the previous multiple of 10.
    """
    lat_n = lat_s + 1  # north edge of the 1° cell
    # snap to 10° grid
    tile_north = int(math.ceil(lat_n / 10.0) * 10)
    tile_west = int(math.floor(lon_w / 10.0) * 10)

    # encode strings
    lat_code = f"{abs(tile_north):02d}{'N' if tile_north >= 0 else 'S'}"
    ew = 'E' if tile_west >= 0 else 'W'
    lon_code = f"{abs(tile_west):03d}{ew}"
    return f"{lat_code}_{lon_code}"

def _presence_prefix(group: str, sub: str, chunk_px: int, date_str: str) -> str:
    """
    S3 prefix that holds presence and distance rasters for this dataset/date.
    Mirrors your current uploader convention.
    """
    base = cn.datasets[group][sub]['s3_processed_base']    # e.g. "climate/.../osm_roads_density"
    return posixpath.join(base, f"{chunk_px}_pixels", date_str)

def _presence_key_for_cell(tile_id: str, lon_w: int, lat_s: int, feature_type: str) -> str:
    chunk_str = f"{_fmt_deg(lon_w)}_{_fmt_deg(lat_s)}_{_fmt_deg(lon_w+1)}_{_fmt_deg(lat_s+1)}"
    return f"{tile_id}__{chunk_str}__{feature_type}_presence.tif"

def _distance_key_for_cell(tile_id: str, lon_w: int, lat_s: int, feature_type: str) -> str:
    chunk_str = f"{_fmt_deg(lon_w)}_{_fmt_deg(lat_s)}_{_fmt_deg(lon_w+1)}_{_fmt_deg(lat_s+1)}"
    return f"{tile_id}__{chunk_str}__{feature_type}_distance.tif"

def _s3_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def _meters_per_pixel(transform: Affine, lat_center_deg: float) -> Tuple[float, float]:
    """
    Compute (pixel_height_m, pixel_width_m) for a geographic grid at given latitude.
    """
    # Degrees per pixel
    xres_deg = abs(transform.a)
    yres_deg = abs(transform.e)
    # Meters per degree
    lat_rad = math.radians(lat_center_deg)
    m_per_deg_lat = 111132.0  # close average
    m_per_deg_lon = 111320.0 * math.cos(lat_rad)
    pix_h_m = yres_deg * m_per_deg_lat
    pix_w_m = xres_deg * m_per_deg_lon
    # Guard against extreme high lat where cos ~ 0:
    pix_w_m = max(pix_w_m, 1e-6)
    return pix_h_m, pix_w_m

def _expanded_bounds(bounds_wgs84: List[float], transform: Affine, halo_m: float) -> List[float]:
    """
    Expand WGS84 bounds by halo_m (meters) independently in X/Y, using local meters-per-pixel.
    """
    minx, miny, maxx, maxy = bounds_wgs84
    lat_center = 0.5 * (miny + maxy)
    pix_h_m, pix_w_m = _meters_per_pixel(transform, lat_center)
    # How many pixels of halo in each direction
    haloy_pix = int(math.ceil(halo_m / pix_h_m))
    halox_pix = int(math.ceil(halo_m / pix_w_m))
    # Convert to degrees via pixel size (in degrees)
    lat_pad = haloy_pix * abs(transform.e)
    lon_pad = halox_pix * abs(transform.a)
    return [minx - lon_pad, miny - lat_pad, maxx + lon_pad, maxy + lat_pad], haloy_pix, halox_pix


# -------------------------- Presence mosaic ---------------------------

def _collect_presence_sources_for_roi(
    s3_client,
    bucket: str,
    prefix: str,
    feature_type: str,
    roi_bounds: List[float],
) -> List[str]:
    """
    Given ROI bounds (WGS84), build the list of VSIS3 presence rasters (1° cells)
    that intersect the ROI — across current or neighbor tiles.

    We select cells by enumerating integer degree cells overlapped by ROI; for each
    cell we derive its owning 10° tile_id and compose the expected filename.
    """
    minx, miny, maxx, maxy = roi_bounds

    lon_start = int(math.floor(minx))
    lon_end   = int(math.ceil(maxx))
    lat_start = int(math.floor(miny))
    lat_end   = int(math.ceil(maxy))

    paths = []
    for lon_w in range(lon_start, lon_end):
        for lat_s in range(lat_start, lat_end):
            tile_id = _tile_id_from_cell(lon_w, lat_s)
            fname = _presence_key_for_cell(tile_id, lon_w, lat_s, feature_type)
            key = posixpath.join(prefix, fname)
            if _s3_exists(s3_client, bucket, key):
                paths.append(f"/vsis3/{bucket}/{key}")
            else:
                LOG.debug("[presence] missing: s3://%s/%s", bucket, key)
    return paths

def _mosaic_presence(bounds_wgs84: List[float],
                     feature_type: str,
                     chunk_px: int,
                     date_str: str,
                     halo_m: float) -> Tuple[np.ndarray, Affine]:
    """
    Create a presence mosaic covering bounds expanded by halo_m (meters).
    Returns (presence_uint8, mosaic_transform).
    """
    # 1) We need a reference transform in geographic degrees to compute an expanded ROI.
    #    Use any union mask tile that overlaps bounds; easiest is to use the tile_id provided
    #    by the caller to open the union mask for that tile. But bounds may cross tiles.
    #    We'll open the union mask of the tile that contains the *center* of the bounds.
    cx = 0.5 * (bounds_wgs84[0] + bounds_wgs84[2])
    cy = 0.5 * (bounds_wgs84[1] + bounds_wgs84[3])

    # Derive tile_id from center:
    lon_w = int(math.floor(cx))
    lat_s = int(math.floor(cy))
    tile_id_ref = _tile_id_from_cell(lon_w, lat_s)

    # Open union mask to obtain resolution/transform (Hansen grid footprint)
    mask_prefix = cn.datasets["peat"]["union_mask"]["30m"]
    mask_path = f"/vsis3/{cn.s3_bucket_name}/{mask_prefix}{tile_id_ref}{PEAT_30M_PATTERN}"
    da_ref = rxr.open_rasterio(mask_path, masked=True)
    transform_ref = da_ref.rio.transform()  # Affine
    da_ref.close()

    # 2) Expand bounds by halo in meters -> degrees (via transform at center latitude)
    expanded_bounds, _, _ = _expanded_bounds(bounds_wgs84, transform_ref, halo_m)

    # 3) Collect presence sources intersecting expanded ROI
    group, sub = feature_type.split("_", 1)
    prefix = _presence_prefix(group, sub, chunk_px, date_str)
    s3_client = boto3.client("s3")
    srcs = _collect_presence_sources_for_roi(
        s3_client, cn.s3_bucket_name, prefix, feature_type, expanded_bounds
    )
    if not srcs:
        return None, None

    # 4) Mosaic with rasterio.merge (read only needed windows)
    #    Presence is 0/1; use method='max' to combine.
    #    Beware of band axis: merge returns (count, H, W).
    with rasterio.Env():
        datasets = [rasterio.open(p) for p in srcs]
        try:
            mosaic, out_transform = rio_merge(
                datasets, bounds=expanded_bounds, nodata=0, method="max"
            )
        finally:
            for ds in datasets:
                ds.close()

    presence = mosaic[0].astype(np.uint8)
    return presence, out_transform


# ------------------------ Distance per 1° chunk ------------------------

@delayed
def _process_chunk_distance(
    tile_id: str,
    chunk_bounds: List[float],
    feature_type: str,
    date_str: str,
    halo_m: float,
    maxdist_m: Optional[float] = None,
) -> dict:
    """
    Build presence mosaic with halo, compute distance in meters, crop to interior,
    mask to peat, and upload.
    """
    chunk_str = "_".join([_fmt_deg(v) for v in chunk_bounds])
    LOG.info("[tile %s | %s] distance-from-presence start", tile_id, chunk_str)

    if not _HAVE_SCIPY:
        msg = "SciPy is required for distance_transform_edt; not available."
        LOG.error(msg)
        return dict(tile=tile_id, bounds=chunk_bounds, status="error", s3=[], msgs=msg)

    # 0) Where to write
    group, sub = feature_type.split("_", 1)
    chunk_px = uutil.calc_chunk_length_pixels(chunk_bounds)
    s3_dir = posixpath.join(
        cn.datasets[group][sub]['s3_processed_base'],
        f"{chunk_px}_pixels",
        date_str,
    )
    local_dir = cn.datasets[group][sub]['local_processed']
    os.makedirs(local_dir, exist_ok=True)

    # 1) Presence mosaic with halo
    presence, mosaic_transform = _mosaic_presence(
        bounds_wgs84=chunk_bounds,
        feature_type=feature_type,
        chunk_px=chunk_px,
        date_str=date_str,
        halo_m=halo_m,
    )
    if presence is None:
        msg = "presence mosaic empty (no presence rasters found in ROI+halo)"
        LOG.info("[tile %s | %s] %s", tile_id, chunk_str, msg)
        return dict(tile=tile_id, bounds=chunk_bounds, status="skip", s3=[], msgs=msg)

    # 2) Compute distance (meters) on the expanded mosaic
    #    distance_transform_edt computes distance to zeros if input is nonzero;
    #    we want distance-to-ONES, so invert presence (ones->lines).
    inv = (presence == 0).astype(np.uint8)

    # meters-per-pixel at center latitude of the *mosaic*
    minx, miny, maxx, maxy = chunk_bounds
    lat_center = 0.5 * (miny + maxy)
    pix_h_m, pix_w_m = _meters_per_pixel(mosaic_transform, lat_center)

    dist_m = distance_transform_edt(inv, sampling=(pix_h_m, pix_w_m)).astype(np.float32)
    if maxdist_m is not None and maxdist_m > 0:
        dist_m = np.minimum(dist_m, float(maxdist_m)).astype(np.float32)

    # 3) Crop back to interior 1° chunk
    win_interior = window_from_bounds(
        minx, miny, maxx, maxy, transform=mosaic_transform
    ).round_offsets().round_lengths()
    row_off = int(win_interior.row_off)
    col_off = int(win_interior.col_off)
    height = int(win_interior.height)
    width = int(win_interior.width)

    dist_crop = dist_m[row_off:row_off+height, col_off:col_off+width]
    # Compute transform for the crop
    crop_transform = mosaic_transform * Affine.translation(col_off, row_off)

    # 4) Mask to peat (union 30 m) for *this tile's* interior chunk
    #    We only need the interior chunk mask; no need to mosaic masks since
    #    output is a single 1° cell for this tile.
    mask_prefix = cn.datasets["peat"]["union_mask"]["30m"]
    mask_path = f"/vsis3/{cn.s3_bucket_name}/{mask_prefix}{tile_id}{PEAT_30M_PATTERN}"
    da_mask_tile = rxr.open_rasterio(mask_path, masked=True)
    da_mask_interior = da_mask_tile.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    da_mask_tile.close()

    # da_mask_interior is aligned to Hansen grid; select band 0 and create boolean
    peat_bool = (da_mask_interior[0].data == 1)
    # Align dimensions just in case (should already match)
    if peat_bool.shape != dist_crop.shape:
        # Re-open as xarray with coords to reindex; but shapes should match when all aligned.
        LOG.warning("[tile %s | %s] mask shape %s != dist shape %s; attempting safe clip",
                    tile_id, chunk_str, peat_bool.shape, dist_crop.shape)
        h = min(peat_bool.shape[0], dist_crop.shape[0])
        w = min(peat_bool.shape[1], dist_crop.shape[1])
        peat_bool = peat_bool[:h, :w]
        dist_crop = dist_crop[:h, :w]

    # Zero-out outside peat (retain 0 as no-data consistent with prior pipeline)
    dist_crop_masked = np.where(peat_bool, dist_crop, 0.0).astype(np.float32)

    # 5) Save, hansenize, upload
    y_coords = da_mask_interior.y.values
    x_coords = da_mask_interior.x.values
    out_da = xr.DataArray(
        dist_crop_masked,
        dims=("y", "x"),
        coords={"y": y_coords, "x": x_coords}
    )
    out_da = out_da.rio.write_crs(da_mask_interior.rio.crs, inplace=True)
    out_da = out_da.rio.write_transform(crop_transform, inplace=True)

    out_name = _distance_key_for_cell(
        tile_id=tile_id,
        lon_w=int(math.floor(minx)),
        lat_s=int(math.floor(miny)),
        feature_type=feature_type
    )
    local_out = os.path.join(local_dir, out_name)
    out_da.rio.to_raster(local_out, compress="lzw")

    # Hansenize/retile and upload (keep same pattern as your stage-1 script)
    # NOTE: We pass output_raster_s3_path_and_name=None and then upload via boto3, as in your script;
    # if your warp function is configured to upload directly, that's fine too.
    warp_to_hansen_coiled(
        source_vrt_path=local_out,
        filename=out_name,
        output_raster_s3_path_and_name=None,
        xmin=minx, ymin=miny, xmax=maxx, ymax=maxy,
        dt=uutil.string_to_gdal_dtype_mapping["Float32"],
        no_data=0,
        tiled=True,
        x_pixel_window=400,
        y_pixel_window=400,
    )

    # Upload to the same processed path used for presence
    s3_client = boto3.client("s3")
    s3_key = posixpath.join(s3_dir, out_name)
    # hansenize writes to /tmp (or user tmp); try both locations
    hansen_local = os.path.join(tempfile.gettempdir(), out_name)
    if not os.path.exists(hansen_local):
        hansen_local = os.path.join(os.path.expanduser("~"), "tmp", out_name)

    if os.path.exists(hansen_local):
        s3_client.upload_file(hansen_local, cn.s3_bucket_name, s3_key)
        s3_uri = f"s3://{cn.s3_bucket_name}/{s3_key}"
        LOG.info("[tile %s | %s] uploaded => %s", tile_id, chunk_str, s3_uri)
    else:
        # Fall back to uploading the non-hansenized local copy (should rarely happen)
        s3_client.upload_file(local_out, cn.s3_bucket_name, s3_key)
        s3_uri = f"s3://{cn.s3_bucket_name}/{s3_key}"
        LOG.warning("[tile %s | %s] hansenized file not found; uploaded non-hansenized %s",
                    tile_id, chunk_str, s3_uri)

    # Clean up locals
    try:
        if os.path.exists(local_out):
            os.remove(local_out)
    except Exception:
        pass
    try:
        if os.path.exists(hansen_local):
            os.remove(hansen_local)
    except Exception:
        pass

    del presence, dist_m, dist_crop, dist_crop_masked, da_mask_interior
    gc.collect()

    return dict(tile=tile_id, bounds=chunk_bounds, status="ok", s3=[s3_uri], msgs="ok")


# ------------------------------ Orchestrator ------------------------------

def _process_tile(
    tile_id: str,
    feature_type: str,
    date_str: str,
    chunk_size: float = 1.0,
    chunk_bounds: Optional[List[float]] = None,
    halo_m: float = 1000.0,
    maxdist_m: Optional[float] = 1000.0,
) -> List[delayed]:
    """
    Build delayed tasks per 1° chunk within a 10x10 tile (or a single chunk if provided).
    """
    tile_bb = uutil.get_10x10_tile_bounds(tile_id)
    if chunk_bounds:
        chunks = [chunk_bounds]
    else:
        chunks = list(_chunk_bounds_iter(tile_bb, chunk_size=float(chunk_size)))

    tasks = []
    for b in chunks:
        tasks.append(_process_chunk_distance(
            tile_id=tile_id,
            chunk_bounds=b,
            feature_type=feature_type,
            date_str=date_str,
            halo_m=float(halo_m),
            maxdist_m=float(maxdist_m) if maxdist_m is not None else None,
        ))
    return tasks


def main(
    tile_id: str,
    feature_type: str = "osm_roads",
    date: Optional[str] = None,
    chunk_bounds: Optional[str] = None,
    chunk_size: float = 1.0,
    halo_m: float = 1000.0,
    maxdist: Optional[float] = 1000.0,
    client: str = "local",
    batch_size: int = 20,
    loglevel: str = "INFO",
):
    logging.basicConfig(
        level=getattr(logging, str(loglevel).upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    LOG.info("Log level set to %s", str(loglevel).upper())

    if not tile_id:
        raise SystemExit("--tile_id is required for this stage-2 script.")

    if date is None or str(date).strip().lower() in ("", "today", "now"):
        # Default to cn.today_date to match stage-1 outputs
        date_str = cn.today_date
    else:
        date_str = str(date)

    # Connect to Dask/Coiled
    run_local = (client == "local")
    cluster, dclient, run_local = uutil.connect_to_cluster(
        cluster_name="roads_canals",
        n_workers=20,
        region="us-east-1",
        run_local=run_local,
    )
    if run_local:
        LOG.info("Running locally without Dask/Coiled.")
    else:
        LOG.info("Using coiled cluster: %s", cluster.name)

    try:
        cb = None
        if chunk_bounds:
            cb = [float(x) for x in str(chunk_bounds).split(",")]

        tasks = _process_tile(
            tile_id=tile_id,
            feature_type=feature_type,
            date_str=date_str,
            chunk_size=chunk_size,
            chunk_bounds=cb,
            halo_m=halo_m,
            maxdist_m=maxdist,
        )

        if not tasks:
            LOG.warning("No tasks generated; nothing to compute.")
            return

        LOG.info("Submitting %d task(s) in batches of %d", len(tasks), int(batch_size))
        completed = 0
        for i in range(0, len(tasks), int(batch_size)):
            batch = tasks[i:i+int(batch_size)]
            LOG.info("Submitting batch of %d tasks (completed=%d)", len(batch), completed)
            _ = dask.compute(*batch)
            completed += len(batch)

        LOG.info("Completed processing")
    finally:
        if not run_local:
            dclient.close()
            cluster.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("Seamless 30 m distance from presence mosaics with halo (cross-tile aware).")
    p.add_argument("--tile_id", required=True, help="10x10 tile ID, e.g. 00N_110E")
    p.add_argument("--feature_type", default="osm_roads",
                   choices=["osm_roads", "osm_canals", "grip_roads"])
    p.add_argument("--date", default=None, help="Date folder of presence products (e.g., 20251113). Default: cn.today_date")
    p.add_argument("--chunk_bounds", default=None, help="Optional single chunk 'minx,miny,maxx,maxy' (WGS84)")
    p.add_argument("--chunk_size", type=float, default=1.0, help="Chunk size in degrees (defaults to 1.0 for 1° cells)")
    p.add_argument("--halo_m", type=float, default=1000.0, help="Halo radius in meters to include presence across edges")
    p.add_argument("--maxdist", type=float, default=1000.0, help="Cap distances at this value in meters (set <=0 to disable)")
    p.add_argument("--client", default="local", choices=["local", "coiled"])
    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--loglevel", default="INFO")
    args = p.parse_args()

    main(
        tile_id=args.tile_id,
        feature_type=args.feature_type,
        date=args.date,
        chunk_bounds=args.chunk_bounds,
        chunk_size=args.chunk_size,
        halo_m=args.halo_m,
        maxdist=args.maxdist,
        client=args.client,
        batch_size=args.batch_size,
        loglevel=args.loglevel,
    )
