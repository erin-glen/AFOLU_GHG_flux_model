# -*- coding: utf-8 -*-
"""
Stage A: Aggregate 10×10° tiles to a global raster at a target angular resolution
(0.04° by default), with proper per-ha/per-pixel semantics, and upload the global
mosaics to S3.

This is functionally equivalent to the previous "aggregate" stage and safe to
run against your current outputs. It uses constants from cn when present but
has safe defaults.

Examples
--------
python -m src.scripts.postprocessing.aggregate_global \
  -cn create_maps \
  --run_name ogh_standard_model \
  --date_tag 20250825 \
  --target_deg 0.04 \
  -p 40000_pixels
"""
from __future__ import annotations

import argparse
import math
import posixpath
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import boto3
import dask
import numpy as np

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu


# ---------- safe access to constants ----------
def _cn(name: str, default):
    return getattr(cn, name, default)

# ---------- defaults ----------
DEFAULT_NATIVE_DEG = 0.00025
DEFAULT_TARGET_DEG = 0.04

DATA_TYPES = [
    "burned_total_Mg_CO2e_pixel_yr",
    "drained_total_Mg_CO2e_pixel_yr",
]

INTEGER_DATASETS: set[str] = set()
INVENTORY_PERIODS = ["2021_2024"]

_BASE_URL_FALLBACK = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_7_0"
BASE_URL = _cn("outputs_path", _BASE_URL_FALLBACK)
OUTPUTS_BASE = posixpath.dirname(BASE_URL)
DEFAULT_DATE_TAG = "20250825"

_S3_CLIENT = _cn("s3_client", boto3.client("s3"))

# ---------- helpers ----------
def deg_to_label(deg: float) -> str:
    s = f"{deg:.5f}".rstrip("0").rstrip(".")
    return f"{s.replace('.', '_')}deg"

def assert_grid_divides_world(target_deg: float):
    rows = round(180 / target_deg); cols = round(360 / target_deg)
    if not (np.isclose(rows*target_deg, 180.0) and np.isclose(cols*target_deg, 360.0)):
        raise ValueError("--target_deg must divide 180 and 360 evenly.")

def _is_density_name(dataset: str) -> bool:
    return dataset.endswith("_ha") or dataset.endswith("_ha_yr")

def _is_pixel_name(dataset: str) -> bool:
    return dataset.endswith("_pixel") or dataset.endswith("_pixel_yr")

def _density_to_pixel_name(dataset: str) -> str:
    return dataset.replace("_ha_yr", "_pixel_yr").replace("_ha", "_pixel")

def _split_s3(url: str) -> Tuple[str, str]:
    assert url.startswith("s3://")
    rest = url[len("s3://"):]
    b, _, k = rest.partition("/")
    return b, k

def _join_s3(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"

def _list_s3_keys(prefix_s3: str) -> Iterable[str]:
    bucket, prefix = _split_s3(prefix_s3)
    token = None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix)
        if token: kw["ContinuationToken"] = token
        resp = _S3_CLIENT.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            yield obj["Key"]
        if not resp.get("IsTruncated"): break
        token = resp.get("NextContinuationToken")

def _discover_tiles_in_dir(tile_dir: str, tile_regex: str) -> List[str]:
    pat = re.compile(tile_regex)
    tiles = set()
    for key in _list_s3_keys(tile_dir):
        name = key.rsplit("/", 1)[-1]
        m = pat.search(name)
        if m: tiles.add(m.group(0))
    return sorted(tiles)

# ---------- I/O dictionaries ----------
def build_download_upload_dict(
    pixel_resolution: str,
    run_name: str,
    base_url: str,
    output_date: str,
    outputs_base: str,
    data_types: list[str] | None = None,
    inventory_periods: list[str] | None = None,
) -> dict:
    data_types = data_types or DATA_TYPES
    inventory_periods = inventory_periods or INVENTORY_PERIODS
    d = {}
    for period in inventory_periods:
        for dataset in data_types:
            src_dir = (
                f"{base_url}/{dataset}/{run_name}/"
                f"five_year_intervals/{period}/{pixel_resolution}/{output_date}/"
            )
            out_dir = f"{outputs_base}/{{res_label}}_output_aggregation/{dataset}/{period}/"
            d[f"{dataset}__{period}"] = {
                "src_dir": src_dir,
                "src_pattern": f"__{dataset}__{period}.tif",
                "global_dir": out_dir,
            }
    return d

# ---------- aggregation kernels ----------
def agg_tile_to_target(
    tile_id: str,
    bounds: tuple[float, float, float, float],
    chunk_length_pixels: int,
    pixel_area_tile: str | None,
    src_tile_path: str,
    per_pixel_output_tile: str | None,
    per_pixel_output_path: str | None,
    use_pixel_area: bool,
    native_deg: float,
    target_deg: float,
    dataset_name: str,
):
    is_final = False
    logger = lu.setup_logging()

    arr = uu.get_tile_dataset_rio(src_tile_path, "Float32", bounds, chunk_length_pixels, is_final, logger)[0]

    is_integer = dataset_name in INTEGER_DATASETS
    if is_integer:
        arr = arr.astype(np.int32)

    is_density = _is_density_name(dataset_name)
    is_pixel = _is_pixel_name(dataset_name)

    if is_integer:
        return uu.reaggregate_mode(arr, native_deg, target_deg)

    if is_density:
        if use_pixel_area:
            if pixel_area_tile is None:
                raise ValueError("Pixel-area conversion requested but pixel_area_tile was not provided.")
            pa = uu.get_tile_dataset_rio(pixel_area_tile, "Float32", bounds, chunk_length_pixels, is_final, logger)[0]
            per_pixel = arr * pa * _cn("m2_to_ha", 1.0/10000.0)
            if per_pixel_output_tile and per_pixel_output_path:
                uu.save_and_upload_single_raster(bounds, chunk_length_pixels, tile_id,
                                                 per_pixel, per_pixel.dtype.name,
                                                 per_pixel_output_tile, per_pixel_output_path,
                                                 is_final, logger)
            return uu.reaggregate_resolution(per_pixel, native_deg, target_deg)
        else:
            summed = uu.reaggregate_resolution(arr, native_deg, target_deg)
            factor = target_deg / native_deg
            if not math.isclose(round(factor), factor):
                raise ValueError(f"target_deg/native_deg must be integer; got {target_deg}/{native_deg}.")
            f = int(round(factor))
            return summed / float(f * f)

    if is_pixel:
        return uu.reaggregate_resolution(arr, native_deg, target_deg)

    return uu.reaggregate_resolution(arr, native_deg, target_deg)

def combine_global_raster(
    tiles: list[np.ndarray],
    bounds_list: list[tuple[float, float, float, float]],
    res_label: str,
    global_outfile: str,
    global_output_path: str,
    target_deg: float,
):
    is_final = False
    logger = lu.setup_logging()

    rows = int(round(180 / target_deg))
    cols = int(round(360 / target_deg))
    global_raster = np.full((rows, cols), np.nan, dtype=np.float32)

    for tile, bounds in zip(tiles, bounds_list):
        min_x, min_y, max_x, max_y = bounds
        x0 = int(round((min_x + 180) / target_deg))
        x1 = int(round((max_x + 180) / target_deg))
        y0 = int(round((90 - max_y) / target_deg))
        y1 = int(round((90 - min_y) / target_deg))
        th, tw = tile.shape
        assert (y1 - y0) == th and (x1 - x0) == tw
        np.copyto(global_raster[y0:y1, x0:x1], tile, where=~np.isnan(tile))

    global_bounds = (-180, -90, 180, 90)
    uu.save_and_upload_single_raster(global_bounds, global_raster.shape[1],
                                     f"{res_label}_global", global_raster, np.float32,
                                     global_outfile, global_output_path,
                                     is_final, logger)
    return "Success"

# ---------- main ----------
def aggregate_main(
    cluster_name: str,
    pixel_resolution: str,
    run_name: str = "ogh_standard_model",
    run_local: bool = False,
    use_pixel_area: bool = True,
    native_deg: float = DEFAULT_NATIVE_DEG,
    target_deg: float = DEFAULT_TARGET_DEG,
    base_url: str = BASE_URL,
    output_date: str = DEFAULT_DATE_TAG,
    outputs_base: str = OUTPUTS_BASE,
):
    assert_grid_divides_world(target_deg)
    logger = lu.setup_logging_main()
    is_final = not run_local

    cluster, client, run_local = uu.connect_to_cluster(cluster_name, run_local=run_local)

    d = build_download_upload_dict(pixel_resolution, run_name, base_url, output_date, outputs_base,
                                   data_types=DATA_TYPES, inventory_periods=INVENTORY_PERIODS)
    res_label = deg_to_label(target_deg)

    for key, items in d.items():
        dataset, interval = key.split("__")
        tiles = _discover_tiles_in_dir(items["src_dir"], _cn("tile_id_pattern", r"[0-9]{2}[A-Z][_][0-9]{3}[A-Z]"))
        if not tiles:
            raise FileNotFoundError(f"No tiles under {items['src_dir']} using pattern {_cn('tile_id_pattern','N/A')}")

        bounds_list = []
        delayed = []

        is_density = _is_density_name(dataset)
        is_integer = dataset in INTEGER_DATASETS
        dataset_out = _density_to_pixel_name(dataset) if (is_density and use_pixel_area and not is_integer) else dataset

        for tile_id in tiles:
            lu.print_and_log(f"aggregate {res_label} {key} @ {uu.timestr()}", is_final, logger)
            src_tile = f"{items['src_dir']}{tile_id}{items['src_pattern']}"
            pixel_area_tile = (
                f"{_cn('pixel_area_dir','s3://gfw2-data/analyses/area_28m/')}{_cn('pixel_area_pattern','hanson_2013_area')}_{tile_id}.tif"
                if (is_density and use_pixel_area and not is_integer) else None
            )
            per_pixel_tile_outfile = None
            per_pixel_output_path = None
            if is_density and use_pixel_area and not is_integer:
                per_pixel_tile_outfile = f"{tile_id}__{dataset_out}__{interval}.tif"
                per_pixel_output_path = f"{base_url}/{dataset_out}/{run_name}/five_year_intervals/{interval}/{pixel_resolution}/{output_date}/"

            bounds = uu.get_10x10_tile_bounds(tile_id)
            bounds_list.append(bounds)
            chunk_len = uu.calc_chunk_length_pixels(bounds)

            delayed.append(
                dask.delayed(agg_tile_to_target)(
                    tile_id, bounds, chunk_len,
                    pixel_area_tile, src_tile,
                    per_pixel_tile_outfile, per_pixel_output_path,
                    use_pixel_area, native_deg, target_deg, dataset
                )
            )

        lu.print_and_log(f"build global {res_label} for {key} @ {uu.timestr()}", is_final, logger)
        tiles_arrays = dask.compute(*delayed)

        global_outfile = f"{res_label}_global__{dataset_out}_{interval}.tif"
        global_output_path = items["global_dir"].format(res_label=res_label)

        _ = combine_global_raster(list(tiles_arrays), bounds_list, res_label,
                                  global_outfile, global_output_path, target_deg)
        lu.print_and_log(f"Saved global raster under {global_output_path} (base {global_outfile})",
                         is_final, logger)

    client.close()

def main():
    p = argparse.ArgumentParser(description="Aggregate tiles to a global raster at target resolution.")
    p.add_argument("-cn", "--cluster_name", required=True)
    p.add_argument("-p", "--pixel_resolution", default="40000_pixels")
    p.add_argument("--run_name", default="ogh_standard_model")
    p.add_argument("--run_local", action="store_true")
    p.add_argument("--skip_pixel_area", action="store_true")
    p.add_argument("--native_deg", type=float, default=DEFAULT_NATIVE_DEG)
    p.add_argument("--target_deg", type=float, default=DEFAULT_TARGET_DEG)
    p.add_argument("--base_url", default=BASE_URL)
    p.add_argument("--date_tag", default=DEFAULT_DATE_TAG)
    p.add_argument("--outputs_base", default=OUTPUTS_BASE)
    args = p.parse_args()

    aggregate_main(
        cluster_name=args.cluster_name,
        pixel_resolution=args.pixel_resolution,
        run_name=args.run_name,
        run_local=args.run_local,
        use_pixel_area=not args.skip_pixel_area,
        native_deg=args.native_deg,
        target_deg=args.target_deg,
        base_url=args.base_url,
        output_date=args.date_tag,
        outputs_base=args.outputs_base,
    )

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Stage A: Aggregate 10×10° tiles to global rasters at a chosen target angular resolution.

Key behaviors
-------------
- Unit-aware aggregation:
  * *_pixel or *_pixel_yr inputs → summed into target cells.
  * *_ha or *_ha_yr inputs:
      - with pixel-area ON (default): convert to per-pixel totals, then sum.
      - with pixel-area OFF: average densities within each target cell.
  * Integer/categorical datasets (listed in INTEGER_DATASETS) → modal aggregation.
- Tile discovery: finds tile IDs automatically by regex.
- Outputs: one global GeoTIFF per dataset/interval under
  .../{res_label}_output_aggregation/{dataset}/{interval}/.

Examples
--------

# --- Cluster setup ---
# Create a Coiled cluster for aggregation (50 workers, 32 GB each):
python -m src.scripts.utilities.create_cluster -n 50 -m 32 -cn create_maps

# --- Aggregation only ---
# Aggregate at default 0.04° (≈4 km) resolution:
python -m src.scripts.postprocessing.aggregate_global \
  -cn create_maps \
  --run_name ogh_standard_model \
  --date_tag 20250825

# Aggregate at finer 0.01° (≈1 km):
python -m src.scripts.postprocessing.aggregate_global \
  -cn create_maps \
  --run_name ogh_standard_model \
  --date_tag 20250825 \
  --target_deg 0.01

# Aggregate at 0.04° but skip pixel-area conversion (averages per-ha instead of converting):
python -m src.scripts.postprocessing.aggregate_global \
  -cn create_maps \
  --run_name ogh_standard_model \
  --date_tag 20250825 \
  --skip_pixel_area

# Run locally (no Coiled); still pass a placeholder cluster name:
python -m src.scripts.postprocessing.aggregate_global \
  --run_local \
  -cn local_dev \
  --run_name test_run \
  --date_tag 20250101 \
  --target_deg 0.04

# Advanced: use alternate input/output roots (e.g., for a newer model version):
python -m src.scripts.postprocessing.aggregate_global \
  -cn create_maps \
  --run_name ogh_standard_model \
  --date_tag 20250825 \
  --base_url s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_7_1 \
  --outputs_base s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs
"""
