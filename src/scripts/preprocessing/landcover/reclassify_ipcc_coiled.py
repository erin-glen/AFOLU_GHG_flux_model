#!/usr/bin/env python
"""Reclassify land cover rasters to the 6 IPCC classes.

This script processes land cover tiles in chunks, writing GeoTIFFs directly to S3.
It can run locally or on a Coiled Dask cluster.
"""

import os
import argparse
import logging
import posixpath
import tempfile

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

# Mapping of GLCLU land cover codes to IPCC classes
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


def load_mapping():
    return GLCLU_MAPPING


def reclassify_array(arr, mapping):
    out = np.full(arr.shape, NODATA, dtype=DTYPE)
    for val, code in mapping.items():
        out[arr == val] = np.uint8(code)
    return out


def reclassify_chunk(lc_path, bbox, mapping, tile_id, run_mode):
    chunk_str = uutil.boundstr(bbox)
    fname = f"{tile_id}__{chunk_str}__lc_ipcc.tif"
    s3_key = posixpath.join(pcn.datasets["land_cover_ipcc"]["s3_processed"], fname)

    if run_mode == "default" and uutil.s3_file_exists(cn.s3_bucket_name, s3_key):
        logging.info(f"Chunk {s3_key} exists on S3, skipping.")
        return

    with rasterio.open(lc_path) as src:
        window = from_bounds(*bbox, src.transform)
        arr = src.read(1, window=window)
        transform = rasterio.windows.transform(window, src.transform)

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

    logging.info(f"Uploaded {s3_key}")
    return f"{tile_id}|{chunk_str} done"


def process_tile(tile_id, lc_year, mapping, chunk_size=2.0, run_mode="default"):
    lc_dict = cn.get_dynamic_download_dict(tile_id, lc_year)
    lc_path = lc_dict['land_cover'].replace('s3://', '/vsis3/').replace('five_years', 'five_year')
    minx, miny, maxx, maxy = uutil.get_10x10_tile_bounds(tile_id)
    chunks = uutil.get_chunk_bounds([minx, miny, maxx, maxy], chunk_size)

    tasks = [
        dask.delayed(reclassify_chunk)(lc_path, b, mapping, tile_id, run_mode)
        for b in chunks
    ]
    dask.compute(*tasks)




def main(tile_id=None, year=2015, chunk_size=2.0, client="local", run_mode="default"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    mapping = load_mapping()

    if client == "coiled":
        cluster, dclient = uutil.connect_to_cluster(
            cluster_name="land_cover_ipcc", n_workers=20, region="us-east-1"
        )
    else:
        cluster = LocalCluster()
        dclient = Client(cluster)

    try:
        mapping_future = dclient.scatter(mapping, broadcast=True)
        tiles = [tile_id] if tile_id else pcn.tile_id_list
        for tid in tiles:
            process_tile(tid, year, mapping_future, chunk_size, run_mode)
    finally:
        dclient.close()
        if client == "coiled":
            cluster.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Land cover reclassification to IPCC classes")
    parser.add_argument("--tile_id", help="Tile ID to process")
    parser.add_argument("--year", type=int, default=2015, help="Land cover year")
    parser.add_argument("--chunk_size", type=float, default=2.0)
    parser.add_argument("--client", default="local", choices=["local", "coiled"])
    parser.add_argument("--run_mode", default="default", choices=["default", "test"])
    args = parser.parse_args()

    main(args.tile_id, args.year, args.chunk_size, args.client, args.run_mode)
