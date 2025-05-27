"""
A memory-optimized variant of ``sdpt_rasterize_attributes``.

This script demonstrates loading SDPT shapefiles in small chunks by using
GDAL/OGR to clip geometries directly before constructing ``GeoDataFrames``.
The overall rasterisation workflow mirrors the original implementation but
avoids loading full tiles into memory.
"""

import os
import sys
import logging
import argparse
import warnings
import posixpath
import gc

import dask
from dask.distributed import Client, LocalCluster
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from osgeo import ogr

# Our universal constants & utilities
import src.scripts.preprocessing.preprocessing_constants as cn
import src.scripts.preprocessing.utilities as uu
from src.scripts.utilities import universal_utilities as uutil

warnings.filterwarnings("ignore", "Geometry is in a geographic CRS.", UserWarning)

# ---------------------------------------------------------------------------
# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Reclassification CSV in S3
ADVANCED_REMAP_S3 = (
    "climate/AFOLU_flux_model/organic_soils/inputs/raw/plantations/sdpt/remapping_tables/advanced_remapping.csv"
)

# Rasterisation settings
RASTER_RES = 0.00025
RASTER_NODATA = 0
RASTER_DTYPE = np.uint8


# ---------------------------------------------------------------------------
# Utility helpers


def list_sdpt_shapefiles():
    """Return the list of SDPT shapefiles stored on S3."""
    prefix = cn.datasets["sdpt"]["s3_raw"]
    return [
        k
        for k in uutil.list_s3_files(cn.s3_bucket_name, prefix)
        if k.lower().endswith(".shp")
    ]


def load_species_reclassification():
    """Return a mapping of vernacular names to numeric rotation codes."""
    import pandas as pd

    local_csv = os.path.join(cn.local_temp_dir, os.path.basename(ADVANCED_REMAP_S3))

    if not os.path.exists(local_csv):
        try:
            uutil.download_file_from_s3(ADVANCED_REMAP_S3, local_csv, cn.s3_bucket_name)
            logging.info(f"Downloaded CSV => {local_csv}")
        except Exception as exc:  # pragma: no cover - network errors
            logging.warning(f"Failed to download CSV from S3: {exc}")
    else:
        logging.info(f"Local CSV already exists => {local_csv}, skipping download.")

    if not os.path.exists(local_csv):
        logging.warning("Advanced remapping CSV not found; falling back to classification logic.")
        return {}

    df = pd.read_csv(local_csv)
    try:
        mapping = dict(zip(df["vernacName"].str.strip(), df["rotation_code"]))
    except KeyError:
        logging.warning("advanced_remapping.csv missing expected columns")
        mapping = {}

    logging.info(f"Loaded {len(mapping)} species from advanced remapping CSV.")
    return mapping


from .create_remapping import classify as fallback_classify
from .create_remapping import ROTATION_CLASS_CODES


# ---------------------------------------------------------------------------
# GDAL/OGR loading utilities


import json

def load_and_clip_shapefile(vsis3_shp_path, bbox):
    """Load and clip a shapefile using GDAL/OGR before creating a GeoDataFrame."""
    minx, miny, maxx, maxy = bbox
    bbox_geom = ogr.CreateGeometryFromWkt(
        f"POLYGON(({minx} {miny}, {maxx} {miny}, {maxx} {maxy}, {minx} {maxy}, {minx} {miny}))"
    )

    shp_ds = ogr.Open(vsis3_shp_path)
    if shp_ds is None:
        raise RuntimeError(f"Failed to open {vsis3_shp_path}")
    layer = shp_ds.GetLayer()
    layer.SetSpatialFilter(bbox_geom)

    clipped_features = []
    for feature in layer:
        json_feature = feature.ExportToJson()
        clipped_features.append(json.loads(json_feature))  # Parse JSON string to dict

    del shp_ds, layer

    if not clipped_features:
        return gpd.GeoDataFrame(columns=["geometry"])
    return gpd.GeoDataFrame.from_features(clipped_features)

def load_and_clip_shapefile(vsis3_shp_path, bbox, simplify_tolerance=0.0001):
    """Load, clip, validate, and simplify geometries using GDAL/OGR and GeoPandas."""

    minx, miny, maxx, maxy = bbox
    bbox_geom = ogr.CreateGeometryFromWkt(
        f"POLYGON(({minx} {miny}, {maxx} {miny}, {maxx} {maxy}, {minx} {maxy}, {minx} {miny}))"
    )

    shp_ds = ogr.Open(vsis3_shp_path)
    if shp_ds is None:
        raise RuntimeError(f"Failed to open {vsis3_shp_path}")

    layer = shp_ds.GetLayer()
    layer.SetSpatialFilter(bbox_geom)

    clipped_features = []
    for feature in layer:
        json_feature = json.loads(feature.ExportToJson())
        geom = shape(json_feature["geometry"])

        # Validate geometry; fix invalid geometries using buffer(0)
        if not geom.is_valid:
            geom = geom.buffer(0)

        # Simplify geometry to reduce memory footprint
        geom = geom.simplify(simplify_tolerance, preserve_topology=True)

        # Skip empty geometries after simplification
        if geom.is_empty:
            continue

        properties = json_feature.get("properties", {})
        clipped_features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": properties
        })

    # Clean up GDAL dataset
    del layer, shp_ds
    gc.collect()

    if not clipped_features:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    # Create GeoDataFrame with explicitly set CRS
    gdf = gpd.GeoDataFrame.from_features(clipped_features, crs="EPSG:4326")

    del clipped_features
    gc.collect()

    return gdf



@dask.delayed
def load_tile_bbox(tile_id, bbox):
    """Load and clip ``tile_id`` shapefile to ``bbox`` using ``GDAL/OGR``."""
    vsis3_tile_shp = f"/vsis3/{cn.s3_bucket_name}/{cn.datasets['sdpt']['s3_raw']}/tile_{tile_id}.shp"
    logging.info(f"Reading {vsis3_tile_shp} for bbox {bbox}")
    try:
        gdf = load_and_clip_shapefile(vsis3_tile_shp, bbox)
    except Exception as exc:
        logging.error(f"Error reading tile_{tile_id}.shp => {exc}")
        return gpd.GeoDataFrame(columns=["geometry"])
    return gdf


# ---------------------------------------------------------------------------
# Classification and rasterisation helpers


def classify_plantation(row, species_map):
    """Return the numeric rotation code for ``row``."""
    simple_type = str(row.get("simpleType", "")).strip().lower()
    simple_name = str(row.get("simpleName", "")).strip().lower()
    vernac_name = str(row.get("vernacName", "")).strip()

    code = species_map.get(vernac_name)
    if code is not None:
        return code

    if simple_type == "tree crops":
        return ROTATION_CLASS_CODES.get("oil_palm") if "oil palm" in simple_name else None

    if simple_type == "planted forest":
        cls = fallback_classify(row)
        return ROTATION_CLASS_CODES.get(cls)

    return None


@dask.delayed
def classify_features(sub_gdf, species_map):
    if hasattr(species_map, "result"):
        species_map = species_map.result()
    sub_gdf["raster_val"] = sub_gdf.apply(lambda r: classify_plantation(r, species_map), axis=1)
    sub_gdf.dropna(subset=["raster_val"], inplace=True)
    return sub_gdf[["geometry", "raster_val"]]


def rasterize_chunk_df(subset_gdf, bbox, tile_id, run_mode):
    """Rasterise a GeoDataFrame subset directly in memory."""
    chunk_str = uutil.boundstr(bbox)
    chunk_px = uutil.calc_chunk_length_pixels(bbox)
    chunk_name = f"{tile_id}__{chunk_str}__sdpt.tif"
    local_dir = cn.datasets["sdpt"]["local_processed"]
    uu.create_directory_if_not_exists(local_dir)
    out_tif = os.path.join(local_dir, chunk_name)

    s3_chunk = posixpath.join(
        cn.datasets["sdpt"]["s3_processed_base"],
        f"{chunk_px}_pixels",
        cn.today_date,
        chunk_name,
    )

    if run_mode == "default":
        if uutil.s3_file_exists(cn.s3_bucket_name, s3_chunk):
            logging.info(f"Partial TIF => s3://{cn.s3_bucket_name}/{s3_chunk} exists => skipping.")
            return
    else:
        if os.path.exists(out_tif):
            logging.info(f"Partial TIF => {out_tif} exists locally => skipping.")
            return

    shapes = [(geom, val) for geom, val in zip(subset_gdf.geometry, subset_gdf["raster_val"])]
    if not shapes:
        logging.info(f"No shapes to rasterize in {bbox}, skipping.")
        return

    minx, miny, maxx, maxy = bbox
    width = int(round((maxx - minx) / RASTER_RES))
    height = int(round((maxy - miny) / RASTER_RES))
    transform = from_origin(minx, maxy, RASTER_RES, RASTER_RES)

    burned = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=RASTER_NODATA,
        dtype=RASTER_DTYPE,
        all_touched=True,
    )

    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": RASTER_DTYPE,
        "crs": "EPSG:4326",
        "transform": transform,
        "tiled": True,
        "compress": "DEFLATE",
        "nodata": RASTER_NODATA,
    }
    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(burned, 1)

    if run_mode == "default":
        logging.info(f"Uploading partial TIF => s3://{cn.s3_bucket_name}/{s3_chunk}")
        uutil.upload_file_to_s3(out_tif, cn.s3_bucket_name, s3_chunk)
        os.remove(out_tif)
    else:
        logging.info(f"Test mode => partial TIF => {out_tif} retained locally.")

    del burned
    gc.collect()


# ---------------------------------------------------------------------------
# Tile processing


def process_tile(tile_id, species_map, chunk_size=2.0, run_mode="default"):
    logging.info(f"Processing entire tile => {tile_id} in ~{chunk_size} deg sub-chunks")

    minx, miny, maxx, maxy = uutil.get_10x10_tile_bounds(tile_id)
    chunk_bboxes = uutil.get_chunk_bounds([minx, miny, maxx, maxy], chunk_size)

    tasks = []
    for bbox in chunk_bboxes:
        # Construct output paths FIRST
        chunk_str = uutil.boundstr(bbox)
        chunk_px = uutil.calc_chunk_length_pixels(bbox)
        chunk_name = f"{tile_id}__{chunk_str}__sdpt.tif"
        local_dir = cn.datasets["sdpt"]["local_processed"]
        uu.create_directory_if_not_exists(local_dir)
        out_tif = os.path.join(local_dir, chunk_name)

        s3_chunk = posixpath.join(
            cn.datasets["sdpt"]["s3_processed_base"],
            f"{chunk_px}_pixels",
            cn.today_date,
            chunk_name,
        )

        # Early check for existing outputs BEFORE expensive clipping/classifying
        if run_mode == "default":
            if uutil.s3_file_exists(cn.s3_bucket_name, s3_chunk):
                logging.info(f"(Early Skip) Partial TIF => s3://{cn.s3_bucket_name}/{s3_chunk} exists, skipping bbox {bbox}.")
                continue
        else:
            if os.path.exists(out_tif):
                logging.info(f"(Early Skip) Partial TIF => {out_tif} exists locally, skipping bbox {bbox}.")
                continue

        # Proceed only if chunk needs processing
        clipped = load_tile_bbox(tile_id, bbox)
        classified = classify_features(clipped, species_map)
        tasks.append(dask.delayed(rasterize_chunk_df)(classified, bbox, tile_id, run_mode))

    return tasks



def process_tile_with_bounds(tile_id, chunk_bounds, species_map, run_mode="default"):
    """Process a single bounding box for ``tile_id``."""
    logging.info(f"Processing single bounding box => {chunk_bounds} for tile => {tile_id}")

    if isinstance(chunk_bounds, str):
        minx, miny, maxx, maxy = map(float, chunk_bounds.split(","))
        chunk_bounds = (minx, miny, maxx, maxy)

    clipped = load_tile_bbox(tile_id, chunk_bounds)
    classified = classify_features(clipped, species_map)

    return [dask.delayed(rasterize_chunk_df)(classified, chunk_bounds, tile_id, run_mode)]


def process_all_tiles(species_map, chunk_size=2.0, run_mode="default"):
    """Process every SDPT tile sequentially."""
    for shp_key in list_sdpt_shapefiles():
        tile_id = os.path.basename(shp_key)[len("tile_") : -4]
        tasks = process_tile(tile_id, species_map, chunk_size, run_mode)
        if tasks:
            logging.info(f"Computing {len(tasks)} chunk tasks for tile => {tile_id} ...")
            dask.compute(*tasks)


# ---------------------------------------------------------------------------
# CLI entry point


def main(tile_id=None, chunk_size=2.0, chunk_bounds=None, run_mode="default", client="local"):
    """Entry point for the testing script."""
    logging.info(
        f"SDPT testing script => base S3 path {cn.datasets['sdpt']['s3_processed_base']}"
    )

    if client == "coiled":
        cluster, client = uutil.connect_to_cluster(
            cluster_name="sdpt_rasterization", n_workers=40, region="us-east-1", worker_memory="64GiB"
        )
        logging.info(f"Coiled cluster => {cluster.name}")
    else:
        cluster = LocalCluster()
        client = Client(cluster)
        logging.info("Local Dask client started.")

    mapping = load_species_reclassification()
    species_map = client.scatter(mapping, broadcast=True)

    tasks = []

    try:
        if tile_id:
            if chunk_bounds:
                tasks = process_tile_with_bounds(tile_id, chunk_bounds, species_map, run_mode)
            else:
                tasks = process_tile(tile_id, species_map, chunk_size, run_mode)
            logging.info(f"Computing {len(tasks)} chunk tasks ...")
            dask.compute(*tasks)
        else:
            logging.info("No tile_id provided => processing all tiles.")
            process_all_tiles(species_map, chunk_size, run_mode)
    finally:
        client.close()
        logging.info("Dask client closed.")
        if client == "coiled":
            cluster.close()
            logging.info("Coiled cluster closed.")

    logging.info("All chunk tasks completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SDPT testing script => partial TIFs stored under s3_processed_base/<px>/YYYYMMDD."
    )
    parser.add_argument("--tile_id", type=str, help="Tile ID (e.g. 00N_110E). Omit to process all tiles.")
    parser.add_argument("--chunk_size", type=float, default=2.0, help="Chunk size (deg).")
    parser.add_argument(
        "--chunk_bounds",
        type=str,
        help='Optional single bounding box "min_x,min_y,max_x,max_y" for quick testing.',
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        choices=["default", "test"],
        default="default",
        help="default => partial TIF => S3, test => local partial TIFs.",
    )
    parser.add_argument(
        "--client", type=str, choices=["local", "coiled"], default="local", help="Dask client type"
    )

    args = parser.parse_args()

    if not any(sys.argv[1:]):
        logging.info("No CLI => processing all tiles locally in test mode for demonstration.")
        main(tile_id=None, chunk_size=2.0, chunk_bounds=None, run_mode="test", client="local")
    else:
        main(
            tile_id=args.tile_id,
            chunk_size=args.chunk_size,
            chunk_bounds=args.chunk_bounds,
            run_mode=args.run_mode,
            client=args.client,
        )