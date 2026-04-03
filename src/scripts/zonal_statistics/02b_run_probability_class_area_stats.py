# -*- coding: utf-8 -*-
"""Run organic-probability class-area zonal statistics by adm0.

This script is intentionally separate from the emissions zonal stats pipeline.
It aggregates area (ha) by probability class (1..100) and adm0 from contextual
layers only:
- global adm0 contextual zarr
- global pixel-area contextual zarr (m2)
- global OGH unthresholded probability contextual zarr (uint8 0..100)

The resulting class-area table can be post-processed into a threshold curve
(area where probability >= threshold) without rerunning the raster reduction.

Example:
python -m src.scripts.zonal_statistics.02b_run_probability_class_area_stats \
  --model_version 0_9_7 \
  --run_date 20260403 \
  --contextual_date 20250925 \
  --probability_date 20251105
"""

from __future__ import annotations

import argparse
import json
import logging
import posixpath
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import dask
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import s3fs
import xarray as xr
from flox import ReindexArrayType, ReindexStrategy
from flox.xarray import xarray_reduce

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import log_utilities as lu
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities.universal_utilities import timestr
from src.scripts.zonal_statistics import zonal_constants as zc

ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
UNCERTAINTY_ROOT = posixpath.join(ROOT, "uncertainty")
CONTEXTUAL_ZARR_ROOT = (
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/global_contextual_zarrs"
)

ADM0_DATASET = "GADM4_1_adm0_global"
ADM0_FILENAME_TEMPLATE = "global_GADM41_adm0_{date}.zarr"

PIXEL_AREA_DATASET = "pixel_area"
PIXEL_AREA_FILENAME_TEMPLATE = "global_pixel_area_{date}.zarr"

ORGANIC_PROBABILITY_DATASET = "ogh_unthresholded_probability"
ORGANIC_PROBABILITY_FILENAME_TEMPLATE = "global_ogh_unthresholded_probability_{date}.zarr"
ORGANIC_PROBABILITY_VAR_NAME = "organic_probability"

AREA_SCALE = np.float32(cn.m2_to_ha)


def adm0_zarr_path(date: str) -> str:
    return posixpath.join(CONTEXTUAL_ZARR_ROOT, ADM0_DATASET, date, ADM0_FILENAME_TEMPLATE.format(date=date))


def pixel_area_zarr_path(date: str) -> str:
    return posixpath.join(
        CONTEXTUAL_ZARR_ROOT,
        PIXEL_AREA_DATASET,
        date,
        PIXEL_AREA_FILENAME_TEMPLATE.format(date=date),
    )


def organic_probability_zarr_path(date: str) -> str:
    return posixpath.join(
        CONTEXTUAL_ZARR_ROOT,
        ORGANIC_PROBABILITY_DATASET,
        date,
        ORGANIC_PROBABILITY_FILENAME_TEMPLATE.format(date=date),
    )


def flox_sparse_reindex_kwargs(use_sparse: bool) -> dict:
    if not use_sparse or ReindexStrategy is None or ReindexArrayType is None:
        return {}
    return {"reindex": ReindexStrategy(blockwise=False, array_type=ReindexArrayType.SPARSE_COO), "fill_value": 0}


def _first_xy_var(ds_or_da: xr.Dataset | xr.DataArray) -> xr.DataArray:
    if isinstance(ds_or_da, xr.DataArray):
        da_ = ds_or_da
    else:
        vars_xy = [v for v in ds_or_da.data_vars.values() if {"x", "y"}.issubset(v.dims)]
        da_ = vars_xy[0] if vars_xy else next(iter(ds_or_da.data_vars.values()))
    if "band" in da_.dims:
        da_ = da_.isel(band=0, drop=True)
    return da_


def open_zarr_region(path: str, bbox: Optional[List[float]], chunk_size: int) -> xr.DataArray:
    dsx = xr.open_zarr(path, consolidated=None, storage_options={"anon": False})
    arr = _first_xy_var(dsx)
    if bbox is not None and {"x", "y"}.issubset(arr.dims):
        west, south, east, north = bbox
        x0, x1 = float(arr.x.values[0]), float(arr.x.values[-1])
        y0, y1 = float(arr.y.values[0]), float(arr.y.values[-1])
        x_slice = slice(min(west, east), max(west, east)) if x0 < x1 else slice(max(east, west), min(east, west))
        y_slice = slice(min(south, north), max(south, north)) if y0 < y1 else slice(max(north, south), min(north, south))
        arr = arr.sel(x=x_slice, y=y_slice)
    if {"x", "y"}.issubset(arr.dims):
        arr = arr.chunk({d: chunk_size for d in ("x", "y")})
    return arr


def align_auto(arr: xr.DataArray, ref: xr.DataArray, tol: float, force: bool = False) -> xr.DataArray:
    if force:
        return arr.reindex_like(ref, method="nearest", tolerance=tol)
    same_x = bool(arr.sizes.get("x") == ref.sizes.get("x")) and np.array_equal(arr.x.values, ref.x.values)
    same_y = bool(arr.sizes.get("y") == ref.sizes.get("y")) and np.array_equal(arr.y.values, ref.y.values)
    if same_x and same_y:
        return arr
    return arr.reindex_like(ref, method="nearest", tolerance=tol)


def pixel_step(arr: xr.DataArray) -> float:
    xvals = arr.x.values
    return float(abs(xvals[1] - xvals[0])) if xvals.size >= 2 else 1.0 / 4000.0


def _normalize_bbox(bbox: List[float]) -> List[float]:
    west, south, east, north = [float(v) for v in bbox]
    if east < west:
        west, east = east, west
    if north < south:
        south, north = north, south
    return [west, south, east, north]


def parse_tile_ids(tile_args: Optional[List[str]]) -> List[str]:
    if not tile_args:
        return []
    out: List[str] = []
    for entry in tile_args:
        if not entry:
            continue
        out.extend([t.strip() for t in entry.split(",") if t.strip()])
    return sorted(set(out))


def build_exact_tile_mask(ref: xr.DataArray, tiles: List[str]) -> xr.DataArray:
    if not tiles:
        return xr.full_like(ref, True, dtype=bool)

    mask = xr.zeros_like(ref, dtype=bool)
    x0, x1 = float(ref.x.values[0]), float(ref.x.values[-1])
    y0, y1 = float(ref.y.values[0]), float(ref.y.values[-1])
    for tile in tiles:
        west, south, east, north = uu.get_10x10_tile_bounds(tile)
        x_slice = slice(min(west, east), max(west, east)) if x0 < x1 else slice(max(east, west), min(east, west))
        y_slice = slice(min(south, north), max(south, north)) if y0 < y1 else slice(max(north, south), min(north, south))
        tile_sel = ref.sel(x=x_slice, y=y_slice)
        tile_mask = xr.full_like(tile_sel, True, dtype=bool)
        mask.loc[dict(x=tile_mask.x, y=tile_mask.y)] = tile_mask
    return mask


def _df_from_result(res: xr.DataArray) -> pd.DataFrame:
    arr = res.values
    idx = np.nonzero(arr)
    if len(idx[0]) == 0:
        return pd.DataFrame(columns=["adm0_id", "probability_class", "area_ha"])

    adm0_ids = res.coords[res.dims[0]].values[idx[0]].astype(np.uint32)
    p_classes = res.coords[res.dims[1]].values[idx[1]].astype(np.uint8)
    values_ha = arr[idx].astype(np.float64)

    out = pd.DataFrame(
        {
            "adm0_id": adm0_ids,
            "probability_class": p_classes,
            "area_ha": values_ha,
        }
    )
    return out.sort_values(["adm0_id", "probability_class"]).reset_index(drop=True)


def output_prefix(contextual_date: str, probability_date: str) -> str:
    return posixpath.join(
        UNCERTAINTY_ROOT,
        "probability_area_stats",
        contextual_date,
        probability_date,
        "by_adm0_probability_class",
    ).rstrip("/") + "/"


def run(args: argparse.Namespace) -> None:
    stage = "probability_area_zonal"
    start_ts = uu.timestr()

    tiles = parse_tile_ids(args.tile_ids)
    bbox = None
    if args.bounding_box:
        bbox = _normalize_bbox([float(x) for x in args.bounding_box])
    elif tiles:
        bounds = [uu.get_10x10_tile_bounds(tile) for tile in tiles]
        bbox = _normalize_bbox([
            min(b[0] for b in bounds),
            min(b[1] for b in bounds),
            max(b[2] for b in bounds),
            max(b[3] for b in bounds),
        ])

    cluster = client = None
    run_local = bool(args.run_local)
    try:
        cluster, client, run_local = uu.connect_to_cluster(cluster_name=args.cluster_name, run_local=args.run_local)
        logger, _ = lu.populate_main_log_header(
            bounding_box=bbox,
            use_shapefile=False,
            client=client,
            cluster=cluster,
            log_note="Organic probability class-area zonal statistics",
            run_local=run_local,
            model_type="organic_soils",
            stage=stage,
        )
        if args.debug:
            logger.setLevel(logging.DEBUG)

        adm0_path = adm0_zarr_path(args.contextual_date)
        pixel_area_path = pixel_area_zarr_path(args.contextual_date)
        prob_path = organic_probability_zarr_path(args.probability_date)

        logger.info("Opening contextual layers: adm0=%s pixel_area=%s probability=%s", adm0_path, pixel_area_path, prob_path)
        adm0 = open_zarr_region(adm0_path, bbox, args.chunk_size).astype("uint32")
        pixel_area = open_zarr_region(pixel_area_path, bbox, args.chunk_size).astype("float32")
        probability = open_zarr_region(prob_path, bbox, args.chunk_size).astype("uint8")

        ref = pixel_area
        tol = float(args.align_tolerance_fraction) * pixel_step(ref)
        adm0_aligned = align_auto(adm0, ref, tol, args.force_align).astype("uint32")
        prob_aligned = align_auto(probability, ref, tol, args.force_align).astype("uint8")

        where_mask = (adm0_aligned > 0) & (prob_aligned > 0)
        if tiles:
            exact_tile_mask = build_exact_tile_mask(ref, tiles)
            where_mask = where_mask & exact_tile_mask

        area_ha = (ref * AREA_SCALE).astype("float32")

        expected_adm0 = np.array([i for i in zc.GADM_ADM0_IDS if i > 0], dtype=np.uint32)
        expected_prob_classes = np.arange(1, 101, dtype=np.uint8)

        logger.info("Reduction start: %s", timestr())
        with dask.annotate(label="reduce:adm0_probability_class_area"):
            res = xarray_reduce(
                area_ha,
                adm0_aligned,
                prob_aligned,
                func="sum",
                expected_groups=(expected_adm0, expected_prob_classes),
                where=where_mask,
                **flox_sparse_reindex_kwargs(not args.no_sparse),
            ).compute()
        logger.info("Reduction end: %s", timestr())

        df = _df_from_result(res)

        out_meta = {
            "model_version": args.model_version,
            "run_name": args.run_name,
            "run_date": args.run_date,
            "contextual_date": args.contextual_date,
            "probability_date": args.probability_date,
            "probability_range_included": [1, 100],
            "excluded_probability_values": [0],
            "area_units": "ha",
            "roi_bbox": bbox,
            "tile_ids": tiles,
            "row_count": int(len(df)),
        }

        local_root = Path(args.local_output).expanduser().resolve()
        local_root.mkdir(parents=True, exist_ok=True)
        local_dir = local_root / "by_adm0_probability_class"
        if local_dir.exists():
            shutil.rmtree(local_dir, ignore_errors=True)
        local_dir.mkdir(parents=True, exist_ok=True)

        local_arrow = pafs.LocalFileSystem()
        ds.write_dataset(
            pa.Table.from_pandas(df, preserve_index=False),
            base_dir=str(local_dir),
            filesystem=local_arrow,
            format="parquet",
            existing_data_behavior="overwrite_or_ignore",
        )
        (local_dir / "manifest.json").write_text(json.dumps(out_meta, indent=2) + "\n", encoding="utf-8")

        remote_prefix = output_prefix(args.contextual_date, args.probability_date)
        fs_s3 = s3fs.S3FileSystem(anon=False)
        remote_exists = fs_s3.exists(remote_prefix.rstrip("/"))
        if remote_exists and not args.overwrite_existing:
            raise FileExistsError(
                f"Remote output already exists at {remote_prefix}. "
                "Pass --overwrite_existing to replace it."
            )
        if remote_exists and args.overwrite_existing:
            fs_s3.rm(remote_prefix.rstrip("/") + "/", recursive=True)

        for fp in local_dir.rglob("*"):
            if fp.is_file():
                rel = fp.relative_to(local_dir).as_posix()
                fs_s3.put(str(fp), posixpath.join(remote_prefix, rel))

        logger.info("Uploaded probability class-area outputs to %s", remote_prefix)

        if not args.keep_local:
            shutil.rmtree(local_root, ignore_errors=True)

    finally:
        if client:
            client.close()
        if cluster:
            cluster.close()
        uu.stage_duration(start_ts, uu.timestr(), stage)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Run organic probability class-area zonal statistics by adm0")
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--run_date", required=True)
    parser.add_argument("--run_name", default="ogh_probability_contextual")
    parser.add_argument("--contextual_date", default="20250925", help="Date tag for adm0/pixel_area contextual zarrs.")
    parser.add_argument("--probability_date", default="20251105", help="Date tag for ogh_unthresholded_probability contextual zarr.")
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--local_output", default="/tmp/probability_area_stats")
    parser.add_argument("--keep_local", action="store_true")
    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_sparse", action="store_true", default=False)
    parser.add_argument("--align_tolerance_fraction", type=float, default=0.49)
    parser.add_argument("--force_align", action="store_true")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run_local", action="store_true")
    mode.add_argument("--cluster_name", default="probability_area_stats")

    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W S E N")
    parser.add_argument("--tile_ids", action="append", help="Comma-separated 10x10 tile IDs")

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
