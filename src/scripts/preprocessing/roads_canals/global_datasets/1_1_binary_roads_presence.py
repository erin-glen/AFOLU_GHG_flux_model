#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1_1_binary_roads_presence.py — 30 m presence (plus optional diagnostic distance)

Assumptions per ops:
- Vectors exist ONLY as shapefiles and ONLY in one projection: EPSG:3395
  (e.g., .../roads_by_tile_3395/roads_{TILE}.shp or canals_{TILE}.shp)
- Do not derive/flip tile_id from bounds; use CLI --tile_id literally.
- Work in the mask CRS (usually geographic) for rasterization.

Fixes:
- Pass a real S3 URI to warp_to_hansen_coiled(output_raster_s3_path_and_name)
  to eliminate "uploaded to None" and do the upload there.
- Log clean [result] lines for each processed chunk.

Products:
- presence: 0/1 raster within peat mask (==1) written to presence/<chunk_px>_pixels/<date>/
- distance: optional Euclidean distance written to distance/<chunk_px>_pixels/<date>/ for diagnostics
- density: placeholder at 30 m; warn to use the 1 km workflow

Example path layout on S3/local scratch (osm_roads, 30 m, 4000 px chunk):
    climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_roads_density/presence/4000_pixels/20251113/
    climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_roads_density/distance/4000_pixels/20251113/

"""

import os
import logging
import gc
import tempfile
import numpy as np
import dask
import dask_geopandas as dgpd
import geopandas as gpd
import xarray as xr

from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from pyogrio.errors import FeatureError

# optional distance
try:
    from scipy.ndimage import distance_transform_edt
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

from src.scripts.preprocessing.hansenize.hansenize_coiled import warp_to_hansen_coiled
from src.scripts.preprocessing.roads_canals.global_datasets import roads_io
from src.scripts.utilities import universal_utilities as uutil
import src.scripts.preprocessing.preprocessing_constants as cn

LOG = logging.getLogger("roads_canals_30m")

VECTORS_EPSG = "EPSG:3395"  # single known projection
DISTANCE_PRODUCT = "distance"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _build_chunk_bounds(tile_bounds, chunk_size=2.0):
    (min_x, min_y, max_x, max_y) = tile_bounds
    x, y = (min_x, min_y)
    chunks = []
    eps = 1e-12
    while y < max_y - eps:
        while x < max_x - eps:
            chunks.append([x, y, x + chunk_size, y + chunk_size])
            x += chunk_size
        x = min_x
        y += chunk_size
    return chunks


def _dask_gdf_is_empty(dgdf, data_path=None):
    try:
        lengths = dgdf.map_partitions(len).compute()
        total = int(sum(lengths))
    except FeatureError as exc:
        if data_path:
            LOG.error("FeatureError while reading %s: %s", data_path, exc)
        return True
    return total == 0


def _vector_path_for_tile(tile_id, feature_type):
    """
    Build the VSIS3 path for EPSG:3395 shapefile for this tile_id.
    Uses cn.datasets[group][sub]['s3_projected'] only (roads_by_tile_3395).
    """
    group, sub = feature_type.split("_", 1)
    base = "roads" if "roads" in feature_type else "canals"
    proj_prefix = cn.datasets[group][sub].get("s3_projected")
    if not proj_prefix:
        raise RuntimeError(f"s3_projected not configured for dataset '{group}.{sub}'")
    key = os.path.join(proj_prefix, f"{base}_{tile_id}.shp").replace("\\", "/")
    return f"/vsis3/{cn.s3_bucket_name}/{key}"


def _read_lines_3395_to_mask_crs(tile_id, feature_type, dst_crs):
    """Read EPSG:3395 shapefile and reproject to dst_crs."""
    vsis3_path = _vector_path_for_tile(tile_id, feature_type)
    LOG.info("[vec] reading shapefile (EPSG:3395): %s", vsis3_path)
    try:
        dgdf = dgpd.read_file(vsis3_path, npartitions=8)
    except Exception as e:
        LOG.error("Failed to read vectors: %s", e)
        # empty in dst_crs to keep pipeline predictable
        empty = gpd.GeoDataFrame({"geometry": []}, crs=dst_crs)
        return dgpd.from_geopandas(empty, npartitions=1), vsis3_path

    # Ensure CRS is set to EPSG:3395 (override only if missing)
    if dgdf.crs is None:
        LOG.warning("[vec] dataset has no CRS; setting to %s for %s", VECTORS_EPSG, vsis3_path)
        dgdf = dgdf.set_crs(VECTORS_EPSG)
    # Reproject to mask CRS as needed
    if str(dgdf.crs) != str(dst_crs):
        dgdf = dgdf.to_crs(dst_crs)
    return dgdf, vsis3_path


def _open_clip_mask(tile_id, bounds_wgs84):
    """Open 30 m union mask for tile_id, clip to bounds_wgs84 (WGS84)."""
    return roads_io.load_mask_chunk(tile_id, bounds_wgs84)


def _rasterize_presence(da_chunk, mask_bool, lines_gdf):
    transform = da_chunk.rio.transform()
    out_shape = da_chunk.shape[1:]
    shapes = ((geom, 1) for geom in lines_gdf.geometry)
    burned = rasterize(
        shapes, out_shape=out_shape, transform=transform,
        fill=0, dtype=np.uint8, all_touched=True
    )
    return ((burned > 0) & mask_bool).astype(np.uint8)


def _rasterize_distance(da_chunk, mask_bool, lines_gdf, maxdist=None):
    """Distance in pixel units if CRS is geographic; capped if maxdist provided."""
    if not _HAVE_SCIPY:
        LOG.warning("[distance] SciPy not available; skipping.")
        return None
    pres = _rasterize_presence(da_chunk, mask_bool, lines_gdf)
    inv = (pres == 0).astype(np.uint8)
    pix = float(abs(da_chunk.rio.resolution()[0]))  # degrees or meters
    dist_pix = distance_transform_edt(inv)
    if maxdist is not None and pix > 0:
        dist_pix = np.minimum(dist_pix, maxdist / pix)  # crude cap if pix is degrees
    return dist_pix.astype(np.float32)

# ---------------------------------------------------------------------
# Chunk worker
# ---------------------------------------------------------------------

@dask.delayed
def _process_chunk(bounds_wgs84, tile_id, feature_type, products, maxdist):
    """
    For one sub-bounds:
    - open/clip mask
    - read 3395 vectors, reproject to mask CRS
    - rasterize presence (+ optional distance)
    - warp to Hansen grid, upload via hansenizer
    """
    chunk_str = "_".join(map(str, bounds_wgs84))
    LOG.info("[tile %s | %s] start", tile_id, chunk_str)

    try:
        da_chunk, mask_bool = _open_clip_mask(tile_id, bounds_wgs84)
    except Exception as e:
        return dict(tile=tile_id, bounds=bounds_wgs84, status="skip", s3=[], msgs=f"mask_open_error: {e}")

    if mask_bool is None:
        return dict(tile=tile_id, bounds=bounds_wgs84, status="skip", s3=[], msgs="No mask in expanded chunk")

    dst_crs = da_chunk.rio.crs
    lines_dgdf, used_path = _read_lines_3395_to_mask_crs(tile_id, feature_type, dst_crs)
    if _dask_gdf_is_empty(lines_dgdf, data_path=used_path):
        return dict(tile=tile_id, bounds=bounds_wgs84, status="skip", s3=[], msgs="lines_empty_after_bbox")

    # Clip lines to chunk in dst_crs
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", dst_crs, *bounds_wgs84, densify_pts=21)
    chunk_poly = box(minx, miny, maxx, maxy)
    lines_clip = dgpd.clip(lines_dgdf, chunk_poly)
    if _dask_gdf_is_empty(lines_clip, data_path=used_path):
        return dict(tile=tile_id, bounds=bounds_wgs84, status="skip", s3=[], msgs="lines_do_not_intersect_chunk")

    try:
        lines_gdf = lines_clip.compute()
    except FeatureError as exc:
        return dict(tile=tile_id, bounds=bounds_wgs84, status="skip", s3=[], msgs=f"FeatureError: {exc}")

    chunk_px = uutil.calc_chunk_length_pixels(bounds_wgs84)
    presence_local_dir = roads_io.local_product_dir(feature_type, "presence")

    transform = da_chunk.rio.transform()
    y, x = da_chunk.y, da_chunk.x
    s3_uploaded = []
    msgs = []

    # PRESENCE
    if "presence" in products:
        arr = _rasterize_presence(da_chunk, mask_bool, lines_gdf)
        if np.any(arr > 0):
            fn = roads_io.presence_raster_name(tile_id, bounds_wgs84, feature_type)
            local_out = os.path.join(presence_local_dir, fn)
            xr.DataArray(arr, dims=("y","x"), coords={"y": y, "x": x})\
              .rio.write_crs(dst_crs, inplace=True)\
              .rio.write_transform(transform, inplace=True)\
              .rio.to_raster(local_out, compress="lzw")

            s3_uri, s3_key = roads_io.build_s3_uri(
                feature_type, "presence", chunk_px, cn.today_date, fn
            )
            hansen_local = os.path.join(tempfile.gettempdir(), fn)
            try:
                warp_to_hansen_coiled(
                    source_vrt_path=local_out,
                    filename=fn,
                    output_raster_s3_path_and_name=s3_uri,
                    xmin=bounds_wgs84[0], ymin=bounds_wgs84[1],
                    xmax=bounds_wgs84[2], ymax=bounds_wgs84[3],
                    dt=uutil.string_to_gdal_dtype_mapping["Byte"],
                    no_data=0,
                    tiled=True, x_pixel_window=400, y_pixel_window=400,
                )
                roads_io.upload_file(hansen_local, s3_key)
            except Exception as exc:
                LOG.error("[tile %s | %s] presence upload failed: %s", tile_id, chunk_str, exc)
                msgs.append(f"presence_upload_failed:{exc}")
            else:
                s3_uploaded.append(s3_uri)
            finally:
                for path in (local_out, hansen_local):
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception:
                        pass
        else:
            msgs.append("presence_all_zero")

    # DISTANCE (optional)
    if "distance" in products:
        dist = _rasterize_distance(da_chunk, mask_bool, lines_gdf, maxdist=maxdist)
        if dist is not None and np.any(dist > 0):
            quicklook_local_dir = roads_io.local_product_dir(feature_type, DISTANCE_PRODUCT)
            fn = roads_io.distance_raster_name(tile_id, bounds_wgs84, feature_type)
            local_out = os.path.join(quicklook_local_dir, fn)
            xr.DataArray(dist, dims=("y","x"), coords={"y": y, "x": x})\
              .rio.write_crs(dst_crs, inplace=True)\
              .rio.write_transform(transform, inplace=True)\
              .rio.to_raster(local_out, compress="lzw")

            s3_uri, s3_key = roads_io.build_s3_uri(
                feature_type, DISTANCE_PRODUCT, chunk_px, cn.today_date, fn
            )
            hansen_local = os.path.join(tempfile.gettempdir(), fn)
            try:
                warp_to_hansen_coiled(
                    source_vrt_path=local_out,
                    filename=fn,
                    output_raster_s3_path_and_name=s3_uri,
                    xmin=bounds_wgs84[0], ymin=bounds_wgs84[1],
                    xmax=bounds_wgs84[2], ymax=bounds_wgs84[3],
                    dt=uutil.string_to_gdal_dtype_mapping["Float32"],
                    no_data=0,
                    tiled=True, x_pixel_window=400, y_pixel_window=400,
                )
                roads_io.upload_file(hansen_local, s3_key)
            except Exception as exc:
                LOG.error("[tile %s | %s] distance upload failed: %s", tile_id, chunk_str, exc)
                msgs.append(f"distance_upload_failed:{exc}")
            else:
                s3_uploaded.append(s3_uri)
            finally:
                for path in (local_out, hansen_local):
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception:
                        pass
        else:
            msgs.append("distance_all_zero_or_missing_scipy")

    # DENSITY at 30 m is a stub
    if "density" in products:
        msgs.append("density_requested_at_30m_use_1km_workflow")

    del da_chunk, lines_dgdf
    gc.collect()

    status = "ok" if s3_uploaded else "skip"
    msg = "; ".join(msgs) if msgs else "ok"
    return dict(tile=tile_id, bounds=bounds_wgs84, status=status, s3=s3_uploaded, msgs=msg)


def _submit_in_batches(tasks, batch_size):
    completed = 0
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        LOG.info("Submitting batch of %d tasks (completed=%d)", len(batch), completed)
        results = dask.compute(*batch)
        for r in results:
            LOG.info("[result] tile=%s bounds=%s status=%s s3=%s msgs=%s",
                     r.get("tile"), r.get("bounds"), r.get("status"),
                     r.get("s3"), r.get("msgs"))
        completed += len(batch)

# ---------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------

def _process_tile(tile_id, feature_type, chunk_size=2.0, chunk_bounds=None,
                  products=("presence",), maxdist=1000):
    # tile bounds from the encoded tile_id (no flips)
    tile_bb = uutil.get_10x10_tile_bounds(tile_id)
    chunks = [chunk_bounds] if chunk_bounds else _build_chunk_bounds(tile_bb, chunk_size=float(chunk_size))
    tasks = []
    for b in chunks:
        tasks.append(_process_chunk(b, tile_id, feature_type, products, maxdist))
    return tasks


def _process_all_tiles(feature_type, chunk_size=2.0, products=("presence",), maxdist=1000):
    prefix = cn.datasets["peat"]["union_mask"]["30m"]
    s3 = uutil.get_s3_client() if hasattr(uutil, "get_s3_client") else None
    if s3 is None:
        import boto3 as _b3  # local import
        s3 = _b3.client("s3")

    tasks = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=cn.s3_bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(roads_io.PEAT_30M_PATTERN):
                tile_id = os.path.basename(key).replace(roads_io.PEAT_30M_PATTERN, "")
                tasks.extend(_process_tile(tile_id, feature_type, chunk_size=chunk_size,
                                           products=products, maxdist=maxdist))
    return tasks

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(
    tile_id=None,
    feature_type="osm_roads",
    chunk_bounds=None,
    chunk_size=2.0,
    client="local",
    resolution="30m",           # fixed for this script
    product="presence",
    maxdist=1000,
    batch_size=20,
    loglevel="INFO",
):
    logging.basicConfig(
        level=getattr(logging, str(loglevel).upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    LOG.info("Log level set to %s", str(loglevel).upper())

    products = tuple([p.strip().lower() for p in str(product).split(",") if p.strip()])

    # Connect Dask/Coiled (or run local)
    run_local = (client == "local")
    cluster, dclient, run_local = uutil.connect_to_cluster(
        cluster_name="roads_canals",
        n_workers=20,
        region="us-east-1",
        run_local=run_local,
    )
    LOG.info("Running locally without Dask/Coiled.") if run_local else LOG.info("Using coiled cluster: %s", cluster.name)

    try:
        if str(resolution).lower() != "30m":
            LOG.warning("This script is the 30m workflow. Use the 1km pipeline for density.")

        if tile_id:
            cb = [float(x) for x in chunk_bounds.split(",")] if chunk_bounds else None
            tasks = _process_tile(tile_id, feature_type, chunk_size=chunk_size,
                                  chunk_bounds=cb, products=products, maxdist=int(maxdist))
        else:
            tasks = _process_all_tiles(feature_type, chunk_size=chunk_size,
                                       products=products, maxdist=int(maxdist))

        if not tasks:
            LOG.warning("No tasks generated; nothing to compute.")
            return

        LOG.info("Submitting %d task(s) to Dask in batches of %d", len(tasks), int(batch_size))
        _submit_in_batches(tasks, int(batch_size))

    finally:
        if not run_local:
            dclient.close()
            cluster.close()
        LOG.info("Completed processing")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("30m roads/canals presence/distance (shapefile-only EPSG:3395, tile-id pinned).")
    parser.add_argument("--tile_id", help="Tile ID, e.g. 00N_110E")
    parser.add_argument("--feature_type", default="osm_roads",
                        choices=["osm_roads", "osm_canals", "grip_roads"])
    parser.add_argument("--chunk_bounds", default=None, help="Optional single chunk: 'minx,miny,maxx,maxy' (WGS84)")
    parser.add_argument("--chunk_size", type=float, default=2.0, help="Chunk size in degrees")
    parser.add_argument("--client", default="local", choices=["local","coiled"])
    parser.add_argument("--resolution", default="30m", help="Fixed at 30m for this script")
    parser.add_argument("--product", default="presence",
                        help="Comma list: presence,distance[,density] (distance uploads under distance/)")
    parser.add_argument("--maxdist", default=1000, help="Cap distance (meters) for distance (pixel-units if geographic)")
    parser.add_argument("--batch_size", default=20, help="Dask submission batch size")
    parser.add_argument("--loglevel", default="INFO", help="DEBUG, INFO, ...")
    args = parser.parse_args()

    main(tile_id=args.tile_id,
         feature_type=args.feature_type,
         chunk_bounds=args.chunk_bounds,
         chunk_size=args.chunk_size,
         client=args.client,
         resolution=args.resolution,
         product=args.product,
         maxdist=args.maxdist,
         batch_size=args.batch_size,
         loglevel=args.loglevel)

"""
Example runs
============

Single 1° chunk (presence + diagnostic distance preview):

python -m src.scripts.preprocessing.roads_canals.global_datasets.1_1_binary_roads_presence \
  --tile_id 00N_110E \
  --feature_type osm_roads \
  --client coiled \
  --resolution 30m \
  --product presence,distance \
  --maxdist 1000 \
  --chunk_bounds "116,-4,117,-3" \
  --chunk_size 1.0 \
  --batch_size 1 \
  --loglevel INFO

Full tile (presence only):

python -m src.scripts.preprocessing.roads_canals.global_datasets.1_1_binary_roads_presence \
  --tile_id 00N_110E \
  --feature_type osm_roads \
  --client coiled \
  --resolution 30m \
  --product presence \
  --maxdist 1000 \
  --chunk_size 1.0 \
  --batch_size 20 \
  --loglevel INFO

"""