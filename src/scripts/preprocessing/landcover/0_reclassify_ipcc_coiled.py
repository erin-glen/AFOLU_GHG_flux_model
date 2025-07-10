#!/usr/bin/env python
"""Reclassify land‑cover rasters to the 6 IPCC classes (Tier‑1).

Key improvements
----------------
* Uses a constant look‑up table (LUT) instead of ``np.vectorize``.
* Verifies the ipcc_codes dictionary on import.
* Asserts GLCLU codes ≤ 255 before casting to ``uint8``.
* Logs unique values in and out for quick sanity‑checks.

The command‑line interface is identical to the previous version.
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

# ---------------------------------------------------------------------------
# ─── CONSTANTS & EARLY VALIDATION ───────────────────────────────────────────
# ---------------------------------------------------------------------------

NODATA: int = 0
DTYPE = np.uint8

# Canonical Tier‑1 codes
EXPECTED_IPCC = {
    'forest': 1,
    'cropland': 2,
    'settlement': 3,
    'wetland': 4,
    'grassland': 5,
    'otherland': 6
}

if ipcc_codes != EXPECTED_IPCC:
    raise ValueError(
        f"ipcc_codes in constants_and_names is\n{ipcc_codes}\n"
        f"but expected\n{EXPECTED_IPCC}\n"
        "Fix the dictionary (or update this script) before proceeding."
    )

# Time slices
FIVE_YEAR_YEARS = [2000, 2005, 2010, 2015, 2020]
ANNUAL_YEARS    = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

# ---------------------------------------------------------------------------
# ─── GLCLU → IPCC LOOK‑UP TABLE ------------------------------------------------
# ---------------------------------------------------------------------------

GLCLU_MAPPING: dict[int, int] = {}
GLCLU_MAPPING.update({i: ipcc_codes["otherland"] for i in range(0,   5)})   # 0‑4
GLCLU_MAPPING.update({i: ipcc_codes["grassland"] for i in range(5,  27)})   # 5‑26
GLCLU_MAPPING.update({i: ipcc_codes["forest"]    for i in range(27, 49)})   # 27‑48

GLCLU_MAPPING.update({i: ipcc_codes["otherland"] for i in range(100, 105)}) # 100‑104
GLCLU_MAPPING.update({i: ipcc_codes["grassland"] for i in range(105, 127)}) # 105‑126
GLCLU_MAPPING.update({i: ipcc_codes["forest"]    for i in range(127, 149)}) # 127‑148

GLCLU_MAPPING.update({i: ipcc_codes["wetland"]   for i in range(200, 205)}) # 200‑204
GLCLU_MAPPING.update({i: ipcc_codes["otherland"] for i in range(205, 208)}) # 205‑207

GLCLU_MAPPING[241] = ipcc_codes["otherland"]
GLCLU_MAPPING[244] = ipcc_codes["cropland"]
GLCLU_MAPPING[250] = ipcc_codes["settlement"]
GLCLU_MAPPING[254] = ipcc_codes["otherland"]

# ---------------------------------------------------------------------------
# Build a 256‑element LUT once, broadcast to workers if using Dask
# ---------------------------------------------------------------------------
MAX_CODE = 255
LUT = np.full(MAX_CODE + 1, NODATA, dtype=DTYPE)
for src_val, dst_val in GLCLU_MAPPING.items():
    LUT[src_val] = dst_val

# ---------------------------------------------------------------------------
# ─── HELPER FUNCTIONS ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def reclassify_array(arr: np.ndarray, lut: np.ndarray = LUT) -> np.ndarray:
    """Map GLCLU codes to IPCC classes using a constant NumPy LUT."""
    # Defensive checks
    max_in_arr = arr.max()
    assert max_in_arr <= MAX_CODE, (
        f"Found GLCLU code {max_in_arr} > {MAX_CODE}. "
        "Increase LUT size or cast to a wider integer."
    )

    logging.info(f"Raster input unique values: {np.unique(arr)}")
    out = lut[arr]               # pure NumPy fancy‑indexing, O(1)
    logging.info(f"Reclassified unique values: {np.unique(out)}")

    return out.astype(DTYPE, copy=False)


def reclassify_chunk(
    lc_path: str,
    bbox: tuple[float, float, float, float],
    tile_id: str,
    interval: str,
    year: int,
    pixel_resolution: str,
    run_mode: str,
) -> str | None:
    """Read one bounding box from the source raster, reclassify, upload."""
    chunk_str = uutil.boundstr(bbox)
    fname = f"{tile_id}__{chunk_str}__lc_ipcc.tif"
    s3_key = posixpath.join(
        pcn.datasets["land_cover_ipcc"]["s3_processed"],
        interval,
        str(year),
        pixel_resolution,
        fname,
    )

    if run_mode == "default" and uutil.s3_file_exists(cn.s3_bucket_name, s3_key):
        logging.info(f"Chunk {s3_key} exists on S3 → skipping.")
        return

    try:
        with rasterio.open(lc_path) as src:
            window = from_bounds(*bbox, src.transform)
            arr = src.read(1, window=window)
            transform = rasterio.windows.transform(window, src.transform)
    except rasterio.errors.RasterioIOError as e:
        logging.error(f"{tile_id}|{year}|{interval}: {e} → skipping.")
        return

    arr = reclassify_array(arr)

    profile = {
        "driver":    "GTiff",
        "height":    arr.shape[0],
        "width":     arr.shape[1],
        "count":     1,
        "dtype":     DTYPE,
        "crs":       "EPSG:4326",
        "transform": transform,
        "tiled":     True,
        "compress":  "DEFLATE",
        "nodata":    NODATA,
    }

    with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
        with rasterio.open(tmp.name, "w", **profile) as dst:
            dst.write(arr, 1)
        uutil.upload_file_to_s3(tmp.name, cn.s3_bucket_name, s3_key)

    del arr
    gc.collect()

    logging.info(f"Uploaded {s3_key}")
    return f"{tile_id}|{interval}|{year}|{chunk_str} done"


def process_tile(
    tile_id: str,
    interval: str,
    year: int,
    pixel_resolution: str,
    chunk_size: float,
    run_mode: str,
) -> list:
    """Return a list of Dask‑delayed tasks for one tile."""
    lc_dict = cn.get_dynamic_download_dict(tile_id, year)
    lc_s3_path = lc_dict["land_cover"]

    s3_key = lc_s3_path.replace("s3://", "")
    if not uutil.s3_file_exists(cn.s3_bucket_name, s3_key):
        logging.warning(f"Source raster {lc_s3_path} not found → skipping tile.")
        return []

    lc_path = lc_s3_path.replace("s3://", "/vsis3/")

    minx, miny, maxx, maxy = uutil.get_10x10_tile_bounds(tile_id)
    chunks = uutil.get_chunk_bounds([minx, miny, maxx, maxy], chunk_size)

    tasks = [
        dask.delayed(reclassify_chunk)(
            lc_path,
            b,
            tile_id,
            interval,
            year,
            pixel_resolution,
            run_mode,
        )
        for b in chunks
    ]
    return tasks


# ---------------------------------------------------------------------------
# ─── MAIN DRIVER ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main(
    tile_id: str | None = None,
    chunk_size: float = 2.0,
    pixel_resolution: str = "8000_pixels",
    cluster_name: str = "reclassify_ipcc",
    run_local: bool = False,
    run_mode: str = "default",
    interval_choice: str = "both",
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if interval_choice == "five_year":
        intervals = [("five_year", FIVE_YEAR_YEARS)]
    elif interval_choice == "annual":
        intervals = [("annual", ANNUAL_YEARS)]
    else:
        intervals = [
            ("five_year", FIVE_YEAR_YEARS),
            ("annual", ANNUAL_YEARS),
        ]

    cluster, dclient, run_local_flag = uutil.connect_to_cluster(
        cluster_name=cluster_name,
        n_workers=20,
        region="us-east-1",
        run_local=run_local,
    )

    if run_local_flag:
        cluster = LocalCluster()
        dclient = Client(cluster)
        logging.info("Running on a local Dask cluster.")
    else:
        logging.info(f"Using Coiled cluster: {cluster.name}")

    # Broadcast LUT to workers so it is not serialized with every task
    if not run_local_flag:
        dclient.scatter(LUT, broadcast=True)

    try:
        tiles = [tile_id] if tile_id else pcn.tile_id_list
        for interval, years in intervals:
            for year in years:
                tasks = []
                for tid in tiles:
                    tasks += process_tile(
                        tid,
                        interval,
                        year,
                        pixel_resolution,
                        chunk_size,
                        run_mode,
                    )

                if tasks:
                    logging.info(f"Computing {len(tasks)} tasks for {interval} {year}")
                    dask.compute(*tasks)
    finally:
        if dclient:
            dclient.close()
        if cluster:
            cluster.close()


# ---------------------------------------------------------------------------
# ─── CLI ENTRY POINT ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Land‑cover reclassification to IPCC classes (Tier‑1)."
    )
    parser.add_argument("--tile_id", help="Specific tile ID to process")
    parser.add_argument("--chunk_size", type=float, default=2.0)
    parser.add_argument("--pixel_resolution", default="8000_pixels")
    parser.add_argument("--cluster_name", default="reclassify_ipcc")
    parser.add_argument("--run_local", action="store_true")
    parser.add_argument(
        "--run_mode", default="default", choices=["default", "test"],
        help="'default' skips chunks already on S3; 'test' overwrites"
    )
    parser.add_argument(
        "--interval",
        default="both",
        choices=["five_year", "annual", "both"],
        help="Select which temporal interval(s) to process",
    )
    args = parser.parse_args()

    main(
        tile_id=args.tile_id,
        chunk_size=args.chunk_size,
        pixel_resolution=args.pixel_resolution,
        cluster_name=args.cluster_name,
        run_local=args.run_local,
        run_mode=args.run_mode,
        interval_choice=args.interval,
    )

"""
python -m src.scripts.preprocessing.landcover.0_reclassify_ipcc_coiled --interval five_year --tile_id 00N_110E
"""