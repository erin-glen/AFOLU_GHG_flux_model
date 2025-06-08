#!/usr/bin/env python
"""Reclassify land cover rasters to the 6 IPCC classes for multiple years.

This script processes land cover tiles in chunks, writing GeoTIFFs directly to
S3. It automatically iterates over the provided years for both five-year and
annual intervals, organizing outputs accordingly. Tiles that are missing on S3
are skipped gracefully.
"""

import os
import argparse
import logging
import posixpath
import tempfile
import gc

import dask
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from dask.distributed import Client, LocalCluster

from src.scripts.preprocessing import preprocessing_constants as pcn
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uutil
from src.scripts.utilities.constants_and_names import ipcc_codes

NODATA = 0
DTYPE = np.uint8

FIVE_YEAR_YEARS = [2000, 2005, 2010, 2015, 2020]
ANNUAL_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

GLCLU_MAPPING = {}
GLCLU_MAPPING.update({i: ipcc_codes["otherland"] for i in range(0, 2)})
GLCLU_MAPPING.update({i: ipcc_codes["grassland"] for i in range(2, 27)})
GLCLU_MAPPING.update({i: ipcc_codes["forest"] for i in range(27, 49)})
GLCLU_MAPPING.update({i: ipcc_codes["wetland"] for i in range(100, 102)})
GLCLU_MAPPING.update({i: ipcc_codes["grassland"] for i in range(102, 127)})
GLCLU_MAPPING.update({i: ipcc_codes["forest"] for i in range(127, 149)})
GLCLU_MAPPING.update({i: ipcc_codes["wetland"] for i in range(200, 205)})
GLCLU_MAPPING.update({i: ipcc_codes["otherland"] for i in range(205, 208)})
GLCLU_MAPPING[241] = ipcc_codes["otherland"]
GLCLU_MAPPING[244] = ipcc_codes["cropland"]
GLCLU_MAPPING[250] = ipcc_codes["settlement"]
GLCLU_MAPPING[254] = ipcc_codes["otherland"]


def reclassify_array(arr, mapping):
    out = np.full(arr.shape, NODATA, dtype=DTYPE)

    unique_vals = np.unique(arr)
    logging.info(f"Raster input values: {unique_vals}")
    logging.info(f"Mapping keys: {list(mapping.keys())}")

    remap_array = np.vectorize(lambda x: mapping.get(int(x), NODATA))
    out = remap_array(arr).astype(DTYPE)

    unique_reclass_vals = np.unique(out)
    logging.info(f"Reclassified output values: {unique_reclass_vals}")

    return out


def reclassify_chunk(lc_path, bbox, mapping, tile_id, interval, year, pixel_resolution, run_mode):
    chunk_str = uutil.boundstr(bbox)
    fname = f"{tile_id}__{chunk_str}__lc_ipcc.tif"
    s3_key = posixpath.join(
        pcn.datasets["land_cover_ipcc"]["s3_processed"],
        interval,
        str(year),
        pixel_resolution,
        fname
    )

    if run_mode == "default" and uutil.s3_file_exists(cn.s3_bucket_name, s3_key):
        logging.info(f"Chunk {s3_key} exists on S3, skipping.")
        return

    try:
        with rasterio.open(lc_path) as src:
            window = from_bounds(*bbox, src.transform)
            arr = src.read(1, window=window)
            transform = rasterio.windows.transform(window, src.transform)
    except rasterio.errors.RasterioIOError as e:
        logging.error(
            f"Tile {tile_id}, Year {year}, Interval {interval}: {e}. Skipping tile."
        )
        return

    arr = reclassify_array(arr, mapping)

    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": DTYPE,
        "crs": "EPSG:4326",
        "transform": transform,
        "tiled": True,
        "compress": "DEFLATE",
        "nodata": NODATA,
    }

    with tempfile.NamedTemporaryFile(suffix=".tif") as tmpfile:
        with rasterio.open(tmpfile.name, "w", **profile) as dst:
            dst.write(arr, 1)
        uutil.upload_file_to_s3(tmpfile.name, cn.s3_bucket_name, s3_key)

    # cleanup
    del arr
    gc.collect()

    logging.info(f"Uploaded {s3_key}")
    return f"{tile_id}|{interval}|{year}|{chunk_str} done"


def process_tile(
    tile_id,
    interval,
    year,
    mapping,
    pixel_resolution,
    chunk_size=2.0,
    run_mode="default",
):
    """Create delayed tasks for one tile.

    Returns a list of :func:`dask.delayed` tasks which reclassify chunks of the
    tile. If the source raster does not exist on S3, an empty list is returned
    so that callers can safely extend their task lists without additional
    checks.
    """

    lc_dict = cn.get_dynamic_download_dict(tile_id, year)
    lc_s3_path = lc_dict["land_cover"]

    # Check if source tile exists on S3
    s3_prefix = f"s3://{cn.s3_bucket_name}/"
    s3_key = (
        lc_s3_path[len(s3_prefix) :] if lc_s3_path.startswith(s3_prefix) else lc_s3_path.replace("s3://", "")
    )
    if not uutil.s3_file_exists(cn.s3_bucket_name, s3_key):
        logging.warning(
            f"Tile {tile_id}, Year {year}, Interval {interval}: Source raster {lc_s3_path} not found. Skipping tile."
        )
        return []

    # Correctly set lc_path for rasterio
    lc_path = lc_s3_path.replace("s3://", "/vsis3/")

    minx, miny, maxx, maxy = uutil.get_10x10_tile_bounds(tile_id)
    chunks = uutil.get_chunk_bounds([minx, miny, maxx, maxy], chunk_size)

    tasks = [
        dask.delayed(reclassify_chunk)(
            lc_path,
            b,
            mapping,
            tile_id,
            interval,
            year,
            pixel_resolution,
            run_mode,
        )
        for b in chunks
    ]

    return tasks



def main(
    tile_id=None,
    chunk_size=2.0,
    pixel_resolution="8000_pixels",
    client="local",
    run_mode="default",
    interval_choice="both",
):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if interval_choice == "five_year":
        intervals = [("five_year", FIVE_YEAR_YEARS)]
    elif interval_choice == "annual":
        intervals = [("annual", ANNUAL_YEARS)]
    else:
        intervals = [("five_year", FIVE_YEAR_YEARS), ("annual", ANNUAL_YEARS)]

    if client == "coiled":
        cluster, dclient = uutil.connect_to_cluster(
            cluster_name="reclassify_ipcc",
            n_workers=20,
            region="us-east-1",
        )
        logging.info(f"Using coiled cluster: {cluster.name}")
    else:
        cluster = LocalCluster()
        dclient = Client(cluster)
        logging.info("Using local cluster")

    mapping = GLCLU_MAPPING
    if client != "local":
        dclient.scatter(mapping, broadcast=True)

    try:
        tiles = [tile_id] if tile_id else pcn.tile_id_list
        for interval, years in intervals:
            for year in years:
                year_tasks = []
                for tid in tiles:
                    year_tasks.extend(
                        process_tile(
                            tid,
                            interval,
                            year,
                            mapping,
                            pixel_resolution,
                            chunk_size,
                            run_mode,
                        )
                    )

                if year_tasks:
                    logging.info(
                        f"Computing {len(year_tasks)} tasks for {interval} {year}"
                    )
                    dask.compute(*year_tasks)
    finally:
        dclient.close()
        cluster.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Land cover reclassification to IPCC classes for multiple years")
    parser.add_argument("--tile_id", help="Tile ID to process")
    parser.add_argument("--chunk_size", type=float, default=2.0)
    parser.add_argument("--pixel_resolution", default="8000_pixels")
    parser.add_argument("--client", default="local", choices=["local", "coiled"])
    parser.add_argument("--run_mode", default="default", choices=["default", "test"])
    parser.add_argument(
        "--interval",
        default="both",
        choices=["five_year", "annual", "both"],
        help="Select time interval to process",
    )
    args = parser.parse_args()

    main(
        args.tile_id,
        args.chunk_size,
        args.pixel_resolution,
        args.client,
        args.run_mode,
        args.interval,
    )