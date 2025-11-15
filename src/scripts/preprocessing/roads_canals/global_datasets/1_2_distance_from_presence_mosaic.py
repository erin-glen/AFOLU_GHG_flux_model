#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1_2_distance_from_presence_mosaic.py
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
    s3://{cn.s3_bucket_name}/<s3_processed_base>/presence/<chunk_px>_pixels/<date>/
  with filenames:
    {tile_id}__{minx}_{miny}_{maxx}_{maxy}__{feature_type}_presence.tif
  e.g.
    00N_110E__116_-4_117_-3__osm_roads_presence.tif
- Vectors used to generate presence were shapefiles (EPSG:3395 or EPSG:4326); this
  script never touches vectors. It only consumes the presence rasters.
- Union 30 m peat mask tiles live at:
    /vsis3/{cn.s3_bucket_name}/{cn.datasets["peat"]["union_mask"]["30m"]}{tile_id}_union_mask.tif
- Hansen grid is geographic (~0.00025 deg), so we compute meters per pixel from latitude.

CLI examples
------------
# Compute distance for a single 1° chunk (with 1 km halo) within a tile:
python -m src.scripts.preprocessing.roads_canals.global_datasets.1_2_distance_from_presence_mosaic \
  --tile_id 00N_110E --feature_type osm_roads --client coiled \
  --chunk_bounds "116,-4,117,-3" --halo_m 1000 --date 20251113 --loglevel INFO

# Compute distance for *all* 1° chunks of a tile (default halo 1000 m):
python -m src.scripts.preprocessing.roads_canals.global_datasets.1_2_distance_from_presence_mosaic \
  --tile_id 00N_110E --feature_type osm_roads --client coiled \
  --date 20251113 --batch_size 20 --loglevel INFO

Outputs are written to:
    s3://{cn.s3_bucket_name}/<s3_processed_base>/distance/<chunk_px>_pixels/<date>/
e.g. climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_roads_density/distance/4000_pixels/20251113/
matching the presence hierarchy.
"""

import os
import math
import gc
import logging
import posixpath
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import dask
from dask import delayed

import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.transform import Affine
import xarray as xr

try:
    from scipy.ndimage import distance_transform_edt
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

from src.scripts.preprocessing.hansenize.hansenize_coiled import warp_to_hansen_coiled
from src.scripts.preprocessing.roads_canals.global_datasets import roads_io
from src.scripts.utilities import universal_utilities as uutil
import src.scripts.preprocessing.preprocessing_constants as cn

LOG = logging.getLogger("rc_distance")


# ------------------------------ Utilities ------------------------------

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
            tile_id = roads_io.tile_id_from_cell(lon_w, lat_s)
            chunk_bounds = [lon_w, lat_s, lon_w + 1, lat_s + 1]
            fname = roads_io.presence_raster_name(tile_id, chunk_bounds, feature_type)
            key = posixpath.join(prefix, fname)
            if _s3_exists(s3_client, bucket, key):
                paths.append(f"/vsis3/{bucket}/{key}")
                continue

            legacy_fname = roads_io.legacy_presence_raster_name(tile_id, chunk_bounds, feature_type)
            legacy_key = posixpath.join(prefix, legacy_fname)
            if legacy_fname != fname and _s3_exists(s3_client, bucket, legacy_key):
                LOG.debug(
                    "[presence] using legacy-named raster for %s: s3://%s/%s",
                    chunk_bounds,
                    bucket,
                    legacy_key,
                )
                paths.append(f"/vsis3/{bucket}/{legacy_key}")
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
    tile_id_ref = roads_io.tile_id_from_cell(lon_w, lat_s)

    da_ref = roads_io.load_mask_tile(tile_id_ref)
    transform_ref = da_ref.rio.transform()
    da_ref.close()

    # 2) Expand bounds by halo in meters -> degrees (via transform at center latitude)
    expanded_bounds, _, _ = _expanded_bounds(bounds_wgs84, transform_ref, halo_m)

    # 3) Collect presence sources intersecting expanded ROI
    prefix = roads_io.product_prefix(feature_type, "presence", chunk_px, date_str)
    s3_client = roads_io.ensure_s3_client()
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
    chunk_str = roads_io.chunk_bounds_to_str(chunk_bounds)
    LOG.info("[tile %s | %s] distance-from-presence start", tile_id, chunk_str)

    if not _HAVE_SCIPY:
        msg = "SciPy is required for distance_transform_edt; not available."
        LOG.error(msg)
        return dict(tile=tile_id, bounds=chunk_bounds, status="error", s3=[], msgs=msg)

    # 0) Where to write
    chunk_px = uutil.calc_chunk_length_pixels(chunk_bounds)
    local_dir = roads_io.local_product_dir(feature_type, "distance")

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
    presence_crop = presence[row_off:row_off+height, col_off:col_off+width]
    # Compute transform for the crop
    crop_transform = mosaic_transform * Affine.translation(col_off, row_off)

    # 4) Mask to peat (union 30 m) for *this tile's* interior chunk
    #    We only need the interior chunk mask; no need to mosaic masks since
    #    output is a single 1° cell for this tile.
    da_mask_interior, peat_bool = roads_io.load_mask_chunk(tile_id, chunk_bounds)
    if peat_bool is None:
        msg = "peat mask empty for chunk"
        LOG.info("[tile %s | %s] %s", tile_id, chunk_str, msg)
        return dict(tile=tile_id, bounds=chunk_bounds, status="skip", s3=[], msgs=msg)

    # Align dimensions just in case (should already match)
    if peat_bool.shape != dist_crop.shape:
        # Re-open as xarray with coords to reindex; but shapes should match when all aligned.
        LOG.warning("[tile %s | %s] mask shape %s != dist shape %s; attempting safe clip",
                    tile_id, chunk_str, peat_bool.shape, dist_crop.shape)
        h = min(peat_bool.shape[0], dist_crop.shape[0])
        w = min(peat_bool.shape[1], dist_crop.shape[1])
        peat_bool = peat_bool[:h, :w]
        dist_crop = dist_crop[:h, :w]
        presence_crop = presence_crop[:h, :w]

    # Ensure cells overlapping the road/canal (presence==1) are stored as distance 1
    # rather than 0. The downstream consumer expects strictly positive distances
    # inside the raster, reserving 0 for nodata outside peat.
    dist_crop[presence_crop > 0] = 1.0

    # Zero-out outside peat (retain 0 as no-data consistent with prior pipeline)
    dist_crop_masked = np.where(peat_bool, dist_crop, 0.0).astype(np.float32)

    # 5) Save, hansenize, upload
    out_name = roads_io.distance_raster_name(
        tile_id=tile_id,
        bounds=chunk_bounds,
        feature_type=feature_type,
    )
    local_out = os.path.join(local_dir, out_name)

    # Use rasterio to avoid dimension mismatches from inherited coordinates.
    mask_crs = da_mask_interior.rio.crs
    raster_meta = {
        "driver": "GTiff",
        "height": dist_crop_masked.shape[0],
        "width": dist_crop_masked.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": mask_crs,
        "transform": crop_transform,
        "nodata": 0.0,
        "compress": "lzw",
    }
    with rasterio.Env():
        with rasterio.open(local_out, "w", **raster_meta) as dst:
            dst.write(dist_crop_masked, 1)
    if hasattr(da_mask_interior, "close"):
        da_mask_interior.close()

    # Hansenize/retile and upload (keep same pattern as your stage-1 script)
    s3_uri, s3_key = roads_io.build_s3_uri(
        feature_type, "distance", chunk_px, date_str, out_name
    )
    hansen_local = os.path.join(tempfile.gettempdir(), out_name)
    try:
        warp_to_hansen_coiled(
            source_vrt_path=local_out,
            filename=out_name,
            output_raster_s3_path_and_name=s3_uri,
            xmin=minx, ymin=miny, xmax=maxx, ymax=maxy,
            dt=uutil.string_to_gdal_dtype_mapping["Float32"],
            no_data=0,
            tiled=True,
            x_pixel_window=400,
            y_pixel_window=400,
        )
        roads_io.upload_file(hansen_local, s3_key)
    except Exception as exc:
        LOG.error("[tile %s | %s] distance upload failed: %s", tile_id, chunk_str, exc)
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
        return dict(tile=tile_id, bounds=chunk_bounds, status="error", s3=[], msgs=f"upload_failed:{exc}")
    else:
        s3_result = [s3_uri]
    finally:
        for path in (local_out, hansen_local):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    del presence, dist_m, dist_crop, dist_crop_masked, da_mask_interior
    gc.collect()

    return dict(tile=tile_id, bounds=chunk_bounds, status="ok", s3=s3_result, msgs="ok")


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


def _process_all_tiles(
    feature_type: str,
    date_str: str,
    chunk_size: float = 1.0,
    halo_m: float = 1000.0,
    maxdist_m: Optional[float] = 1000.0,
) -> List[delayed]:
    s3 = roads_io.ensure_s3_client()
    prefix = cn.datasets["peat"]["union_mask"]["30m"]

    tasks: List[delayed] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=cn.s3_bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(roads_io.PEAT_30M_PATTERN):
                continue

            tile_id = os.path.basename(key).replace(roads_io.PEAT_30M_PATTERN, "")
            tasks.extend(
                _process_tile(
                    tile_id=tile_id,
                    feature_type=feature_type,
                    date_str=date_str,
                    chunk_size=chunk_size,
                    chunk_bounds=None,
                    halo_m=halo_m,
                    maxdist_m=maxdist_m,
                )
            )

    return tasks


def main(
    tile_id: Optional[str] = None,
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
        if tile_id:
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
        else:
            if chunk_bounds:
                LOG.warning("Ignoring --chunk_bounds because no --tile_id was provided.")

            tasks = _process_all_tiles(
                feature_type=feature_type,
                date_str=date_str,
                chunk_size=chunk_size,
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
    p.add_argument("--tile_id", help="10x10 tile ID, e.g. 00N_110E")
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