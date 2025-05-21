#!/usr/bin/env python
"""
pp_roads_canals_chunks_fishnet.py

Restored workflow using Dask GeoDataFrames to:
  1) Read 1 km union peat mask from S3 (EPSG:3395).
  2) Create fishnet cells from the mask (where data == 1).
  3) Read reprojected roads/canals from S3 as a Dask GeoDataFrame (EPSG:3395).
  4) Intersect and assign partial line length to fishnet cells.
  5) Convert fishnet to a raster, write locally, and upload to S3.
  6) Optionally mosaic the chunk outputs and warp to final Hansen-style tiles
     using ``--postprocess``.

Chunk-based approach:
  - We might chunk each 10×10 tile into sub-bounds (2° × 2°, etc.),
    or just process the entire tile if memory allows.

Usage example:
  python pp_roads_canals_chunks_fishnet.py --tile_id 00N_110E --feature_type osm_roads
"""

import os
import logging
import gc
import boto3
import numpy as np
import dask
import dask_geopandas as dgpd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from rasterio.features import rasterize
from shapely.geometry import box
from dask.distributed import Client, LocalCluster
from rasterio.warp import transform_bounds
import posixpath
import tempfile

from src.scripts.utilities import universal_utilities as uutil
import src.scripts.preprocessing.preprocessing_constants as cn
import src.scripts.preprocessing.utilities as uu
from src.scripts.preprocessing.hansenize.hansenize_coiled import (
    build_vrt_gdal_coiled,
    warp_to_hansen_coiled,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# If you store the 1 km union mask in "peat.union_mask.1km_3395",
# define a pattern:
PEAT_1KM_PATTERN = "_union_mask_1km.tif"

def build_chunk_bounds(tile_bounds, chunk_size=2):
    """
    Subdivide the tile bounds into smaller bounding boxes.
    e.g. chunk_size=2 => 5 sub-chunks horizontally in a 10×10 tile.
    """
    (min_x, min_y, max_x, max_y) = tile_bounds
    x, y = (min_x, min_y)
    chunks = []
    while y < max_y:
        while x < max_x:
            chunk = [x, y, x+chunk_size, y+chunk_size]
            chunks.append(chunk)
            x += chunk_size
        x = min_x
        y += chunk_size
    return chunks

def mask_peatraster(data):
    """
    Convert raster data == 1 => 1, else 0.
    """
    return (data == 1).astype(np.uint8)

def create_fishnet_from_masked(masked_data, transform):
    """
    Create a fishnet (cells) where ``masked_data`` equals 1.

    This function used to iterate over every cell in the raster which was
    expensive for large arrays.  We now locate only the non-zero pixels using
    ``numpy.nonzero`` and build polygons for those locations.  The result is
    wrapped in a ``dask_geopandas`` GeoDataFrame for parallel operations.
    """

    rows, cols = np.nonzero(masked_data)
    polygons = []
    for r, c in zip(rows, cols):
        x1, y1 = transform * (c, r)
        x2, y2 = transform * (c + 1, r + 1)
        polygons.append(box(x1, y1, x2, y2))

    if not polygons:
        return dgpd.from_geopandas(gpd.GeoDataFrame({'geometry': []}, crs="EPSG:3395"), npartitions=1)

    gdf = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:3395")
    return dgpd.from_geopandas(gdf, npartitions=10)

def dask_gdf_is_empty(dgdf):
    """Return True if a Dask GeoDataFrame has no rows."""
    return dgdf.map_partitions(len).compute().sum() == 0

def read_reprojected_lines_dask(tile_id, feature_type):
    """
    Read the reprojected lines as a Dask GeoDataFrame in EPSG:3395.
    Suppose we have them in
      cn.datasets[group][sub]['s3_projected'] => "s3://..."
    or we open them with vsis3.

    If the shapefiles are chunked by tile => e.g. "roads_00N_110E.shp".
    We'll build that name and read it with dask_geopandas.
    """
    group, sub = feature_type.split('_', 1)
    s3_proj_prefix = cn.datasets[group][sub]['s3_projected']  # e.g. "climate/.../osm/roads_3395"
    # Suppose the file is "roads_00N_110E.shp" or "canals_00N_110E.shp"
    if "roads" in feature_type:
        base_name = "roads"
    elif "canals" in feature_type:
        base_name = "canals"
    else:
        base_name = "roads"  # fallback

    shp_name = f"{base_name}_{tile_id}.shp"
    s3_path = os.path.join(s3_proj_prefix, shp_name).replace("\\", "/")
    vsis3_path = f"/vsis3/{cn.s3_bucket_name}/{s3_path}"
    logging.info(f"Reading lines from {vsis3_path}")
    try:
        lines_dgdf = dgpd.read_file(vsis3_path, npartitions=8)
        return lines_dgdf
    except Exception as e:
        logging.error(f"Could not read reprojected lines: {e}")
        # Return empty
        return dgpd.from_geopandas(gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:3395"), npartitions=1)


def _chunk_paths(tile_id, feature_type):
    """Return S3 paths for chunk rasters belonging to a tile."""
    group, sub = feature_type.split("_", 1)
    prefix = cn.datasets[group][sub]["s3_processed_small"].rstrip("/") + "/"
    keys = uutil.list_s3_files(cn.s3_bucket_name, prefix)
    out = []
    for k in keys:
        if tile_id in os.path.basename(k) and k.endswith(".tif"):
            out.append(f"s3://{cn.s3_bucket_name}/{k}")
    return out


@dask.delayed
def postprocess_tile(tile_id, feature_type, _deps=None):
    """Mosaic chunk rasters and warp to Hansen-style GeoTIFF."""
    group, sub = feature_type.split("_", 1)
    out_prefix = cn.datasets[group][sub]["s3_processed"].rstrip("/")
    local_dir = cn.datasets[group][sub]["local_processed"]
    os.makedirs(local_dir, exist_ok=True)

    chunk_paths = _chunk_paths(tile_id, feature_type)
    if not chunk_paths:
        logging.warning(f"[{feature_type}|{tile_id}] no chunk rasters found for postprocess")
        return None

    vrt_name = f"{tile_id}_{feature_type}_mosaic.vrt"
    vrt_s3 = posixpath.join(out_prefix, vrt_name)
    local_vrt = os.path.join(tempfile.gettempdir(), vrt_name)
    build_vrt_gdal_coiled(chunk_paths, vrt_s3, local_vrt=local_vrt)

    xmin, ymin, xmax, ymax = uutil.get_10x10_tile_bounds(tile_id)

    local_out = os.path.join(local_dir, f"{tile_id}_{feature_type}_density.tif")
    s3_out = posixpath.join(out_prefix, f"{tile_id}_{feature_type}_density.tif")

    warp_to_hansen_coiled(
        source_vrt_path=vrt_s3,
        filename=os.path.basename(local_out),
        output_raster_s3_path_and_name=None,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        dt=uutil.string_to_gdal_dtype_mapping["Float32"],
        no_data=0,
        tiled=True,
        x_pixel_window=400,
        y_pixel_window=400,
    )

    uutil.upload_file_to_s3(local_out, cn.s3_bucket_name, s3_out)
    os.remove(local_out)
    logging.info(f"[{feature_type}|{tile_id}] hansenized")
    return s3_out

@dask.delayed
def process_chunk(bounds, tile_id, feature_type):
    """
    1) Open the 1 km union mask in EPSG:3395
    2) Clip to chunk
    3) Mask => create fishnet
    4) Intersect lines => partial line length
    5) Convert fishnet to raster => local, then upload to S3
    """
    chunk_str = "_".join(map(str, bounds))
    group, sub = feature_type.split('_', 1)
    local_dir = cn.datasets[group][sub]['local_processed']
    s3_dir = cn.datasets[group][sub]['s3_processed_small']

    local_out = os.path.join(local_dir, f"{tile_id}_{chunk_str}_{feature_type}_density_fishnet.tif")
    s3_out_key = f"{s3_dir}/{tile_id}_{chunk_str}_{feature_type}_density_fishnet.tif"

    logging.info(f"[{tile_id}|{chunk_str}] Starting chunk processing")

    # 1) open union mask
    prefix_1km_3395 = cn.datasets["peat"]["union_mask"]["1km_3395"]
    raster_path = f"/vsis3/{cn.s3_bucket_name}/{prefix_1km_3395}{tile_id}{PEAT_1KM_PATTERN}"
    logging.debug(f"Opening {raster_path}")

    try:
        da = rxr.open_rasterio(raster_path, masked=True)
        logging.debug(f"Union mask CRS: {da.rio.crs}")
        logging.debug(f"Union mask bounds: {da.rio.bounds()}")
    except Exception as e:
        logging.error(f"[{tile_id}|{chunk_str}] Could not open union mask: {e}")
        return

    # 2) clip by bounds
    minx, miny, maxx, maxy = bounds
    try:
        minx, miny, maxx, maxy = transform_bounds(
            "EPSG:4326", da.rio.crs, minx, miny, maxx, maxy, densify_pts=21
        )
        chunked_da = da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        if chunked_da.isnull().all():
            logging.info(f"[{tile_id}|{chunk_str}] No data in this chunk's raster. Skipping.")
            return
    except Exception as e:
        logging.error(f"[{tile_id}|{chunk_str}] clip_box error: {e}")
        return

    # 3) mask => create fishnet
    masked = (chunked_da[0].data == 1).astype(np.uint8)
    if np.all(masked == 0):
        logging.info(f"[{tile_id}|{chunk_str}] chunk is all zeros => skipping fishnet.")
        return

    # create fishnet
    transform = chunked_da.rio.transform()
    fishnet_dgdf = create_fishnet_from_masked(masked, transform)
    if dask_gdf_is_empty(fishnet_dgdf):
        logging.info(f"[{tile_id}|{chunk_str}] fishnet is empty => skipping.")
        return

    # 4) read lines => partial line length
    lines_dgdf = read_reprojected_lines_dask(tile_id, feature_type)
    if dask_gdf_is_empty(lines_dgdf):
        logging.info(f"[{tile_id}|{chunk_str}] lines are empty => skip.")
        return

    # We can do an intersection in memory
    # We must do: fishnet_dgdf & lines_dgdf => partial line length
    # Because both are in EPSG:3395, we do approximate partial length.

    # Keep operations on the Dask GeoDataFrames as long as possible
    # to avoid materializing large intermediate pandas objects.

    # Filter lines that intersect the chunk bounds
    chunk_poly = box(minx, miny, maxx, maxy)
    lines_clip = dgpd.clip(lines_dgdf, chunk_poly)
    if dask_gdf_is_empty(lines_clip):
        logging.info(f"[{tile_id}|{chunk_str}] lines do not intersect chunk => skip.")
        return

    # Bring Dask GeoDataFrames to pandas for geometric ops
    lines_gdf = lines_clip.compute()
    fishnet_gdf = fishnet_dgdf.compute()

    # Clip road segments to the fishnet grid (union of cells)
    clipped = gpd.clip(lines_gdf, fishnet_gdf)
    if clipped.empty:
        logging.info(f"[{tile_id}|{chunk_str}] no lines within fishnet => skip.")
        return

    # Assign each clipped line to a cell and sum lengths per cell
    fishnet_gdf = fishnet_gdf.reset_index(drop=True)
    fishnet_gdf["cell_id"] = fishnet_gdf.index
    joined = gpd.sjoin(
        clipped,
        fishnet_gdf[["geometry", "cell_id"]],
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        logging.info(f"[{tile_id}|{chunk_str}] join produced no segments => skip.")
        return

    joined["partial_len"] = joined.apply(
        lambda row: row.geometry.intersection(
            fishnet_gdf.loc[row["index_right"], "geometry"]
        ).length,
        axis=1,
    )
    length_by_cell = (
        joined.groupby("cell_id")["partial_len"].sum().reset_index()
    )

    fishnet_gdf = fishnet_gdf.merge(length_by_cell, on="cell_id", how="left")
    fishnet_gdf["partial_len"].fillna(0, inplace=True)

    # 5) rasterize fishnet => local_out
    # We'll rasterize partial_len.
    # Transform => same as chunked_da transform, shape => chunked_da shape
    shapes = [
        (geom, val) for geom, val
        in zip(fishnet_gdf.geometry, fishnet_gdf["partial_len"])
        if val > 0
    ]
    if not shapes:
        logging.info(f"[{tile_id}|{chunk_str}] no shapes => skip.")
        return

    # shape is (height,width)
    out_shape = chunked_da.shape[1:]
    burned = rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0.0,
        dtype=np.float32,
        all_touched=True
    )
    # optional: convert to km
    burned /= 1000.0

    if np.allclose(burned, 0, equal_nan=True):
        logging.info(f"[{tile_id}|{chunk_str}] all 0 => skip writing.")
        return

    # write out
    xr_ras = xr.DataArray(burned, dims=("y","x"), coords={"y": chunked_da.y, "x": chunked_da.x})
    xr_ras = xr_ras.rio.write_crs("EPSG:3395", inplace=True)
    xr_ras = xr_ras.rio.write_transform(transform, inplace=True)
    local_dir_path = os.path.dirname(local_out)
    os.makedirs(local_dir_path, exist_ok=True)

    xr_ras.rio.to_raster(local_out, compress="lzw")
    logging.info(f"[{tile_id}|{chunk_str}] saved local => {local_out}")

    # upload
    s3_client = boto3.client("s3")
    s3_client.upload_file(local_out, cn.s3_bucket_name, s3_out_key)
    logging.info(f"[{tile_id}|{chunk_str}] uploaded => s3://{cn.s3_bucket_name}/{s3_out_key}")
    os.remove(local_out)

    # cleanup
    del chunked_da, lines_dgdf, fishnet_dgdf
    gc.collect()

    return f"[{tile_id}|{chunk_str}] done"

def process_tile(tile_id, feature_type, chunk_size=2, chunk_bounds=None, postprocess=False):
    """
    Build tasks to process each sub-chunk of a tile in EPSG:3395.

    - tile_id e.g. 00N_110E
    - feature_type e.g. osm_roads, osm_canals, grip_roads
    - chunk_size e.g. 2 => 2-degree sub-chunks
    - chunk_bounds => optional single sub-chunk
    """
    tile_bb = uutil.get_10x10_tile_bounds(tile_id)
    if chunk_bounds:
        chunks = [chunk_bounds]
    else:
        chunks = build_chunk_bounds(tile_bb, chunk_size=chunk_size)

    tasks = [process_chunk(b, tile_id, feature_type) for b in chunks]
    if postprocess:
        post_task = postprocess_tile(tile_id, feature_type, tasks)
        tasks.append(post_task)
    return tasks

def process_all_tiles(feature_type, chunk_size=2, postprocess=False):
    """
    Iterate over the S3 prefix with 1km union mask in EPSG:3395, build tasks for each tile.
    """
    prefix_1km_3395 = cn.datasets["peat"]["union_mask"]["1km_3395"]
    # We'll list objects that end with PEAT_1KM_PATTERN
    import boto3
    s3 = boto3.client("s3")

    tasks = []
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=cn.s3_bucket_name, Prefix=prefix_1km_3395)
    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith(PEAT_1KM_PATTERN):
                    # e.g. "00N_110E_union_mask_1km.tif"
                    base_name = os.path.basename(key)
                    tile_id = base_name.replace(PEAT_1KM_PATTERN, "")
                    tasks.extend(
                        process_tile(
                            tile_id,
                            feature_type,
                            chunk_size=chunk_size,
                            postprocess=postprocess,
                        )
                    )

    return tasks

def main(tile_id=None, feature_type="osm_roads", chunk_bounds=None, chunk_size=2, client="local", postprocess=False):
    if client == "coiled":
        cluster, dclient = uutil.connect_to_cluster(
            cluster_name="roads_canals",
            n_workers=20,
            region="us-east-1",
        )
        logging.info(f"Using coiled cluster: {cluster.name}")
    else:
        cluster = LocalCluster()
        dclient = Client(cluster)
        logging.info("Using local cluster")

    try:
        if tile_id:
            tasks = process_tile(
                tile_id,
                feature_type,
                chunk_size=chunk_size,
                chunk_bounds=chunk_bounds,
                postprocess=postprocess,
            )
        else:
            tasks = process_all_tiles(
                feature_type,
                chunk_size=chunk_size,
                postprocess=postprocess,
            )

        if not tasks:
            logging.warning("No tasks generated; nothing to compute.")
            return

        dask.compute(*tasks)
    finally:
        # Cleanup resources after processing tasks or early exit
        dclient.close()
        if client == "coiled":
            cluster.close()
        logging.info("Completed processing")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Fishnet-based line density for roads/canals with Dask + dask_geopandas.")
    parser.add_argument("--tile_id", help="Specific tile ID e.g. 00N_110E")
    parser.add_argument("--feature_type", default="osm_roads", choices=["osm_roads","osm_canals","grip_roads"],
                        help="Which lines to use")
    parser.add_argument("--chunk_bounds", default=None,
                        help="Optional single chunk: 'minx,miny,maxx,maxy'")
    parser.add_argument("--chunk_size", type=float, default=2, help="Chunk size in degrees e.g. 2 => 2x2 sub-chunks")
    parser.add_argument("--client", default="local", choices=["local","coiled"])
    parser.add_argument("--postprocess", action="store_true", help="Warp chunk outputs to final Hansen tiles")
    args = parser.parse_args()

    cb = None
    if args.chunk_bounds:
        cb = [float(x) for x in args.chunk_bounds.split(",")]

    main(
        tile_id=args.tile_id,
        feature_type=args.feature_type,
        chunk_bounds=cb,
        chunk_size=args.chunk_size,
        client=args.client,
        postprocess=args.postprocess,
    )