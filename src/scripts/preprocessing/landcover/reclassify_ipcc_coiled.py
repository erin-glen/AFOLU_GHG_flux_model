#!/usr/bin/env python
"""Reclassify land cover rasters to the 6 IPCC classes.

This script processes land cover tiles in chunks and writes partial
GeoTIFFs.  It can run locally or on a Coiled Dask cluster.
"""

import os
import argparse
import logging
import posixpath

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


def build_years(interval_type, start_year=None, end_year=None):
    """Return a sorted list of years based on the interval type.

    Parameters
    ----------
    interval_type : str
        One of ``cn.intervals_annual``, ``cn.intervals_five_years`` or
        ``cn.intervals_hybrid``.
    start_year : int, optional
        Starting year of the range. Defaults to
        ``cn.annual_land_cover_start_year``.
    end_year : int, optional
        Ending year of the range. Defaults to the final inventory year from
        ``cn.five_year_inventory_periods``.

    Returns
    -------
    list[int]
        Sorted list of unique years for which land cover data should be
        processed.
    """

    start_year = start_year or cn.annual_land_cover_start_year
    end_year = end_year or cn.five_year_inventory_periods[-1][1]

    years = []

    if interval_type in (cn.intervals_annual, cn.intervals_hybrid):
        annual_start = max(start_year, cn.annual_land_cover_start_year)
        annual_end = min(end_year, cn.five_year_inventory_periods[-1][1])
        years.extend(range(annual_start, annual_end + 1))

    if interval_type in (cn.intervals_five_years, cn.intervals_hybrid):
        five_years = [
            y
            for y in cn.five_year_land_cover_years
            if start_year <= y <= end_year
        ]
        years.extend(five_years)

    return sorted(set(years))


def load_mapping():
    """Return the internal GLCLU → IPCC mapping."""
    return GLCLU_MAPPING


def reclassify_array(arr, mapping):
    out = np.full(arr.shape, NODATA, dtype=DTYPE)
    for val, code in mapping.items():
        out[arr == val] = np.uint8(code)
    return out


def reclassify_glclu(arr):
    """Reclassify GLCLU land cover codes to IPCC classes."""
    return reclassify_array(arr, GLCLU_MAPPING)


def reclassify_chunk(lc_path, bbox, mapping, tile_id, run_mode):
    chunk_str = uutil.boundstr(bbox)
    out_dir = pcn.datasets["land_cover_ipcc"]["local_processed"]
    uutil.create_directory_if_not_exists(out_dir)
    fname = f"{tile_id}__{chunk_str}__lc_ipcc.tif"
    out_path = os.path.join(out_dir, fname)
    s3_key = posixpath.join(
        pcn.datasets["land_cover_ipcc"]["s3_processed"], fname
    )

    # ------------------------------------------------------------------
    # Early check to avoid redundant work
    if run_mode == "default":
        if uutil.s3_file_exists(cn.s3_bucket_name, s3_key):
            logging.info(
                f"Chunk TIF => s3://{cn.s3_bucket_name}/{s3_key} exists => skipping."
            )
            return
    else:
        if os.path.exists(out_path):
            logging.info(f"Chunk TIF => {out_path} exists locally => skipping.")
            return

    with rasterio.open(lc_path) as src:
        window = from_bounds(*bbox, src.transform)
        arr = src.read(1, window=window)
        transform = rasterio.windows.transform(window, src.transform)

    if mapping is None:
        arr = reclassify_glclu(arr)
    else:
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

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)

    if run_mode == "default":
        uutil.upload_file_to_s3(out_path, cn.s3_bucket_name, s3_key)
        os.remove(out_path)
    else:
        logging.info(f"Test mode => {out_path} retained locally")

    return f"{tile_id}|{chunk_str} done"


def process_tile(tile_id, lc_year, mapping, chunk_size=2.0, run_mode="default"):
    lc_dict = cn.get_dynamic_download_dict(tile_id, lc_year)
    lc_path = f"/vsis3/{lc_dict['land_cover'][5:]}"
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
            cluster_name="land_cover_ipcc",
            n_workers=20,
            region="us-east-1",
        )
        logging.info(f"Using coiled cluster: {cluster.name}")
    else:
        cluster = LocalCluster()
        dclient = Client(cluster)
        logging.info("Using local cluster")

    try:
        mapping_future = dclient.scatter(mapping, broadcast=True)
        if tile_id:
            process_tile(tile_id, year, mapping_future, chunk_size, run_mode)
        else:
            for tid in pcn.tile_id_list:
                process_tile(tid, year, mapping_future, chunk_size, run_mode)
    finally:
        dclient.close()
        if client == "coiled":
            cluster.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Land cover reclassification to IPCC classes")
    parser.add_argument("--tile_id", help="Single tile ID to process")
    parser.add_argument("--year", type=int, default=2015, help="Land cover year")
    parser.add_argument("--chunk_size", type=float, default=2.0, help="Chunk size in degrees")
    parser.add_argument("--client", default="local", choices=["local", "coiled"])
    parser.add_argument("--run_mode", default="default", choices=["default", "test"])
    args = parser.parse_args()

    main(
        tile_id=args.tile_id,
        year=args.year,
        chunk_size=args.chunk_size,
        client=args.client,
        run_mode=args.run_mode,
    )