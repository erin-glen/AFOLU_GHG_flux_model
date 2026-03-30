# -*- coding: utf-8 -*-
"""Run organic-soils zonal statistics (per-pixel only; robust alignment; per-interval upload).

Production-lean defaults:
- Diagnostics OFF by default (skip flux-over-ocean full scan).
- Smart alignment: skip reindex_like if coords already equal to pixel_area.

python -m src.scripts.zonal_statistics.02_run_zonal_stats \
  --interval_end_years 2024 \
  --cluster_name drainage_cluster \
  --run_date 20251118 \
  --model_version 0_9_7 \
  --run_name ogh_sensitivity_500m_10 \
  --chunk_size 10000 \
  --diagnostics off \
  --datasets drained_co2 drained_n2o

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List

import boto3
import dask
import dask.array as da
import numpy as np
import pandas as pd
import posixpath
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import s3fs
import xarray as xr

import flox
from flox import ReindexArrayType, ReindexStrategy
from flox.xarray import xarray_reduce

from src.scripts.zonal_statistics import zonal_constants as zc
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities.universal_utilities import timestr
from src.scripts.utilities import log_utilities as lu

# ------------------------------- config --------------------------------
SPARSE_DEFAULT = True
ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
OUTPUT_BASE = "{root}/version_{model_version}"

DATASETS: Dict[str, Dict[str, Any]] = {
    "emissions_state_nodes": {"zarr": "emissions_state_node_{interval}.zarr", "var": "emissions_state_nodes"},
    "drained_state_nodes": {"zarr": "drained_state_node_{interval}.zarr", "var": "drained_state_nodes"},
    "burned_state_nodes":  {"zarr": "burned_state_node_{interval}.zarr",  "var": "burned_state_nodes"},
    "drained_total":       {"zarr": "drained_total_Mg_CO2e_pixel_yr_{interval}.zarr", "var": "drained_total"},
    "drained_co2":         {"zarr": "drained_co2_Mg_CO2_pixel_yr_{interval}.zarr", "var": "drained_co2"},
    "drained_n2o":         {"zarr": "drained_n2o_Mg_CO2e_pixel_yr_{interval}.zarr", "var": "drained_n2o"},
    "drained_total_co2":   {"zarr": "drained_total_co2_Mg_CO2_pixel_yr_{interval}.zarr", "var": "drained_total_co2"},
    "drained_total_ch4":   {"zarr": "drained_total_ch4_Mg_CO2e_pixel_yr_{interval}.zarr", "var": "drained_total_ch4"},
    "burned_total":        {"zarr": "burned_total_Mg_CO2e_pixel_yr_{interval}.zarr",  "var": "burned_total"},
    "burned_total_co2":    {"zarr": "burned_total_co2_Mg_CO2_pixel_yr_{interval}.zarr",  "var": "burned_total_co2"},
    "burned_total_ch4":    {"zarr": "burned_total_ch4_Mg_CO2e_pixel_yr_{interval}.zarr",  "var": "burned_total_ch4"},
}

FLUX_SPECS = {
    "drained_total": {"code": 0, "label": "drained_total_Mg_CO2e", "group": "drained"},
    "drained_co2": {"code": 3, "label": "drained_co2_Mg_CO2", "group": "drained"},
    "drained_n2o": {"code": 4, "label": "drained_n2o_Mg_CO2e", "group": "drained"},
    "drained_total_co2": {"code": 5, "label": "drained_total_co2_Mg_CO2", "group": "drained"},
    "drained_total_ch4": {"code": 6, "label": "drained_total_ch4_Mg_CO2e", "group": "drained"},
    "burned_total": {"code": 1, "label": "burned_total_Mg_CO2e", "group": "burned"},
    "burned_total_co2": {"code": 7, "label": "burned_total_co2_Mg_CO2", "group": "burned"},
    "burned_total_ch4": {"code": 8, "label": "burned_total_ch4_Mg_CO2e", "group": "burned"},
}

ZARR_CACHE_PREFIX = OUTPUT_BASE + "/zarr/{run_name}/{run_date}/{interval}/"


def ordered_dataset_keys(selected: Optional[List[str]]) -> List[str]:
    if not selected:
        return list(DATASETS.keys())
    return [k for k in DATASETS if k in set(selected)]

# ---- Contextual Zarrs (must match Step 1) ----
CONTEXTUAL_ZARR_ROOT = (
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/global_contextual_zarrs"
)

ADM0_DATASET = "GADM4_1_adm0_global"
ADM0_DATE = "20250925"
ADM0_FILENAME_TEMPLATE = "global_GADM41_adm0_{date}.zarr"

def adm0_zarr_path(date: str = ADM0_DATE) -> str:
    return posixpath.join(
        CONTEXTUAL_ZARR_ROOT, ADM0_DATASET, date, ADM0_FILENAME_TEMPLATE.format(date=date),
    )

PIXEL_AREA_DATASET = "pixel_area"
PIXEL_AREA_DATE = "20250925"
PIXEL_AREA_ZARR = posixpath.join(
    CONTEXTUAL_ZARR_ROOT, PIXEL_AREA_DATASET, PIXEL_AREA_DATE, f"global_pixel_area_{PIXEL_AREA_DATE}.zarr",
)

# ------------------------------ utils ----------------------------------
def flox_sparse_reindex_kwargs(use_sparse: bool) -> dict:
    if not use_sparse or ReindexStrategy is None or ReindexArrayType is None:
        return {}
    return {"reindex": ReindexStrategy(blockwise=False, array_type=ReindexArrayType.SPARSE_COO), "fill_value": 0}

def build_output_parquet(model_version: str, run_name: str, run_date: str, interval: str) -> str:
    return posixpath.join(ROOT, f"version_{model_version}", "zonal_stats", run_name, run_date, interval).rstrip("/") + "/"

def _split_s3(path: str) -> tuple[str, str]:
    if not path.startswith("s3://"):
        raise ValueError(f"Expected s3:// path, got {path}")
    bucket, key = path[5:].split("/", 1)
    return bucket, key

def zarr_exists(zarr_path: str) -> bool:
    b, k = _split_s3(zarr_path.rstrip("/"))
    s3 = boto3.client("s3")
    for probe in ("zarr.json", ".zgroup"):
        try:
            s3.head_object(Bucket=b, Key=f"{k}/{probe}")
            return True
        except Exception:
            continue
    return False

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
    data_arr = _first_xy_var(dsx)
    if bbox is not None and {"x", "y"}.issubset(data_arr.dims):
        west, south, east, north = bbox
        x0, x1 = float(data_arr.x.values[0]), float(data_arr.x.values[-1])
        y0, y1 = float(data_arr.y.values[0]), float(data_arr.y.values[-1])
        x_slice = slice(min(west, east), max(west, east)) if x0 < x1 else slice(max(east, west), min(east, west))
        y_slice = slice(min(south, north), max(south, north)) if y0 < y1 else slice(max(north, south), min(north, south))
        data_arr = data_arr.sel(x=x_slice, y=y_slice)
    chunk_dict = {d: chunk_size for d in ("x", "y") if d in data_arr.dims}
    if chunk_dict:
        data_arr = data_arr.chunk(chunk_dict)
    return data_arr

def pixel_step(arr: xr.DataArray) -> float:
    xvals = arr.x.values
    return float(abs(xvals[1] - xvals[0])) if xvals.size >= 2 else 1.0 / 4000.0

def align_like_nearest_tol(arr: xr.DataArray, ref: xr.DataArray, tol: float) -> xr.DataArray:
    return arr.reindex_like(ref, method="nearest", tolerance=tol)

def coords_match(a: xr.DataArray, b: xr.DataArray) -> bool:
    """True if x/y sizes AND coordinate values match exactly."""
    if not {"x", "y"}.issubset(a.dims) or not {"x", "y"}.issubset(b.dims):
        return False
    try:
        return (
            a.sizes.get("x") == b.sizes.get("x")
            and a.sizes.get("y") == b.sizes.get("y")
            and np.array_equal(a["x"].values, b["x"].values)
            and np.array_equal(a["y"].values, b["y"].values)
        )
    except Exception:
        return False

def align_auto(arr: xr.DataArray, ref: xr.DataArray, tol: float, force_align: bool) -> xr.DataArray:
    """Skip reindex if coords already match; otherwise reindex with tolerance."""
    if not force_align and coords_match(arr, ref):
        return arr
    return align_like_nearest_tol(arr, ref, tol)

def _upload_dir(fs_s3: s3fs.S3FileSystem, local_dir: Path, dest_prefix: str) -> int:
    uploaded = 0
    for p in local_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir)
            remote_path = posixpath.join(dest_prefix.rstrip("/"), *rel.parts)
            fs_s3.put(str(p), remote_path)
            uploaded += 1
    return uploaded

def convert_to_coord_dict(flox_result: xr.DataArray) -> dict:
    arr = flox_result.data
    if isinstance(arr, da.Array):
        arr = arr.compute()
    dims = flox_result.dims
    if hasattr(arr, "coords") and hasattr(arr, "data"):
        indices, values = arr.coords, arr.data
    else:
        grid = np.indices(arr.shape)
        indices = grid.reshape(len(arr.shape), -1)
        values = arr.ravel()
    return {dim: flox_result.coords[dim].values[indices[i]] for i, dim in enumerate(dims)} | {"value": values}

def _df_from_result(res: xr.DataArray, flux_map: Dict[int, str], interval_end: int) -> pd.DataFrame:
    coord_dict = convert_to_coord_dict(res)
    df = pd.DataFrame(coord_dict)
    df["flux_type"] = df["flux_type"].replace(flux_map)
    if "drained_state_nodes" in df.columns:
        df["drained_state_meaning"] = (
            df["drained_state_nodes"].astype("string").str.zfill(8).map(zc.DRAINED_STATE_NODE_MEANINGS)
        )
    if "burned_state_nodes" in df.columns:
        df["burned_state_meaning"] = (
            df["burned_state_nodes"].astype("string").str.zfill(8).map(zc.BURNED_STATE_NODE_MEANINGS)
        )
    df["interval_end"] = interval_end
    df.loc[df["flux_type"].eq("area__ha"), "value"] = df["value"] / 10000.0  # m² -> ha
    return df

def build_zarr_paths(interval: str, dataset_names: Optional[List[str]] = None, **fmt_kw) -> Dict[str, Dict[str, Any]]:
    zarr_base = ZARR_CACHE_PREFIX.format(interval=interval, **fmt_kw)
    return {
        name: {"zarr": zarr_base + spec["zarr"].format(interval=interval), "var": spec["var"]}
        for name, spec in DATASETS.items()
        if name in ordered_dataset_keys(dataset_names)
    }

def leak_ratio_check(drained: xr.DataArray, burned: xr.DataArray, adm0: xr.DataArray,
                     mode: str, threshold: float, logger: logging.Logger) -> None:
    """Compute flux-over-ocean leak ratio with selectable cost."""
    if mode == "off":
        return

    dt, bt, am = drained, burned, adm0
    if mode == "basic":
        # Strided sampling to approximate ratio cheaply (~0.5–1% of pixels)
        # Target ~5k samples along each axis.
        nx = dt.sizes.get("x", 0); ny = dt.sizes.get("y", 0)
        sx = max(1, int(round(nx / 5000))) if nx else 1
        sy = max(1, int(round(ny / 5000))) if ny else 1
        dt = dt.isel(x=slice(0, None, sx), y=slice(0, None, sy))
        bt = bt.isel(x=slice(0, None, sx), y=slice(0, None, sy))
        am = am.isel(x=slice(0, None, sx), y=slice(0, None, sy))

    # mode == "full" computes at full resolution
    flux_mask = ((dt > 0) | (bt > 0))
    ocean_mask = (am == 0)
    leak = (flux_mask & ocean_mask).sum().compute()
    denom = flux_mask.sum().compute()
    ratio = float(leak) / float(denom) if denom else 0.0

    if ratio > threshold:
        logger.warning(
            "Flux-over-ocean (adm0==0) ratio is %.4f (> %.4f). "
            "This typically indicates grid misalignment or missing tiles.",
            ratio, threshold
        )
    else:
        logger.info("Flux-over-ocean (adm0==0) ratio: %.4f", ratio)

# -------------------------------- driver --------------------------------
def run(args: argparse.Namespace) -> None:
    stage = "zonal_statistics"
    start_ts = uu.timestr()
    cluster, client, run_local = uu.connect_to_cluster(cluster_name=args.cluster_name, run_local=args.run_local)

    # ROI from bbox or union of tiles
    bbox: Optional[List[float]] = None
    if args.bounding_box:
        bbox = [float(x) for x in args.bounding_box]
    elif args.tile_ids:
        tiles: List[str] = []
        for item in args.tile_ids:
            tiles.extend(t.strip() for t in item.split(",") if t.strip())
        if tiles:
            bounds = [uu.get_10x10_tile_bounds(t) for t in tiles]
            west = min(b[0] for b in bounds); south = min(b[1] for b in bounds)
            east = max(b[2] for b in bounds); north = max(b[3] for b in bounds)
            bbox = [west, south, east, north]

    logger, _ = lu.populate_main_log_header(
        bounding_box=bbox, use_shapefile=False, client=client, cluster=cluster,
        log_note="Organic soils zonal statistics (per-pixel; robust alignment)", run_local=run_local,
        model_type="organic_soils", stage=stage,
    )
    if args.debug:
        logger.setLevel(logging.DEBUG)
    logger.debug("Starting run with args: %s", args)
    logger.info("Diagnostics mode: %s | Force align: %s", args.diagnostics, args.force_align)

    OUTPUT_KW = dict(root=ROOT, model_version=args.model_version, run_date=args.run_date, run_name=args.run_name)
    selected_names = ordered_dataset_keys(args.datasets)
    drained_fluxes = [k for k in selected_names if FLUX_SPECS.get(k, {}).get("group") == "drained"]
    burned_fluxes = [k for k in selected_names if FLUX_SPECS.get(k, {}).get("group") == "burned"]
    drained_only_gases = set(drained_fluxes) == {"drained_co2", "drained_n2o"}
    required_names = set(selected_names)
    if drained_fluxes or burned_fluxes:
        required_names.add("emissions_state_nodes")
    if drained_fluxes:
        required_names.add("drained_state_nodes")
    if burned_fluxes:
        required_names.add("burned_state_nodes")
    dataset_names = ordered_dataset_keys(list(required_names))

    # Open contextual layers (built in Step 1)
    adm0_zarr = adm0_zarr_path()
    adm0 = open_zarr_region(adm0_zarr, bbox, args.chunk_size).astype("uint32")
    pixel_area = open_zarr_region(PIXEL_AREA_ZARR, bbox, args.chunk_size).persist()

    # Expected groups (exclude 0 → ocean)
    gadm_adm0_ids = np.array([i for i in zc.GADM_ADM0_IDS if i > 0], dtype=np.uint32)
    drained_codes_arr = np.array(sorted({0, *map(int, zc.ALL_DRAINED_STATE_CODES)}), dtype=np.uint32)
    burned_codes_arr  = np.array(sorted({0, *map(int, zc.ALL_BURNED_STATE_CODES)}),  dtype=np.uint32)

    # Local staging
    local_arrow = pafs.LocalFileSystem()
    base_dir_root = Path(args.local_output).expanduser().resolve()
    base_dir_root.mkdir(parents=True, exist_ok=True)
    base_dir_drained = base_dir_root / "drained"
    base_dir_burned = base_dir_root / "burned"
    fs_s3 = s3fs.S3FileSystem(anon=False)

    # Canonical reference grid = pixel_area
    ref = pixel_area
    dx = pixel_step(ref)
    tol = float(args.align_tolerance_fraction) * dx

    # Intervals
    mapping = {end: (start, end) for start, end in cn.five_year_inventory_periods}
    interval_pairs = [mapping[y] for y in args.interval_end_years if y in mapping]

    for interval_start_year, interval_end_year in interval_pairs:
        interval = f"{interval_start_year}_{interval_end_year}"
        logger.info("Processing interval %s : %s", interval, timestr())

        paths = build_zarr_paths(interval, dataset_names=dataset_names, **OUTPUT_KW)
        for key in dataset_names:
            zpath = paths[key]["zarr"]
            if not zarr_exists(zpath):
                if key == "emissions_state_nodes":
                    logger.warning(
                        "Optional dataset %s missing for %s; will fall back to legacy state nodes if available.",
                        key,
                        interval,
                    )
                    continue
                raise FileNotFoundError(
                    f"Missing Zarr for {key}: {zpath}\n"
                    f"Build with: python -m src.scripts.zonal_statistics.01_build_zarr_caches "
                    f"--interval_end_years {interval_end_year} --run_date {args.run_date} "
                    f"--model_version {args.model_version} --run_name {args.run_name} "
                    f"--chunk_size {args.chunk_size}"
                )

        zarr_data: Dict[str, xr.DataArray] = {}
        for key in drained_fluxes + burned_fluxes:
            zarr_data[key] = open_zarr_region(paths[key]["zarr"], bbox, args.chunk_size)

        drained_nodes_aligned = burned_nodes_aligned = None
        emissions_nodes_aligned = None
        if "emissions_state_nodes" in paths:
            try:
                emissions_nodes_raw = open_zarr_region(
                    paths["emissions_state_nodes"]["zarr"], bbox, args.chunk_size
                ).astype("uint32")
                emissions_nodes_aligned = align_auto(emissions_nodes_raw, ref, tol, args.force_align)
            except Exception as exc:
                logger.warning(
                    "emissions_state_nodes unavailable for %s (%s); attempting legacy node inputs.",
                    interval,
                    exc,
                )

        if emissions_nodes_aligned is not None:
            dec_drained, dec_burned = zc.unpack_emissions_state_to_legacy(
                emissions_nodes_aligned.values.astype(np.uint32, copy=False)
            )
            drained_nodes_aligned = xr.DataArray(
                data=dec_drained,
                dims=emissions_nodes_aligned.dims,
                coords=emissions_nodes_aligned.coords,
            )
            burned_nodes_aligned = xr.DataArray(
                data=dec_burned,
                dims=emissions_nodes_aligned.dims,
                coords=emissions_nodes_aligned.coords,
            )
        else:
            if "drained_state_nodes" in paths:
                drained_nodes_raw = open_zarr_region(
                    paths["drained_state_nodes"]["zarr"], bbox, args.chunk_size
                ).astype("uint32")
                drained_nodes_aligned = align_auto(drained_nodes_raw, ref, tol, args.force_align)
            if "burned_state_nodes" in paths:
                burned_nodes_raw = open_zarr_region(
                    paths["burned_state_nodes"]["zarr"], bbox, args.chunk_size
                ).astype("uint32")
                burned_nodes_aligned = align_auto(burned_nodes_raw, ref, tol, args.force_align)

        # Smart alignment (skip if coords already match)
        adm0_aligned = align_auto(adm0, ref, tol, args.force_align)
        aligned_flux = {k: align_auto(v, ref, tol, args.force_align) for k, v in zarr_data.items()}

        # Optional flux-over-ocean diagnostic (off by default)
        if {"drained_total", "burned_total"}.issubset(aligned_flux):
            leak_ratio_check(aligned_flux["drained_total"], aligned_flux["burned_total"], adm0_aligned,
                             args.diagnostics, args.leak_warn_threshold, logger)
        else:
            logger.info(
                "Skipping flux-over-ocean diagnostic because drained_total and burned_total were not both selected."
            )

        where_mask = (adm0_aligned > 0)

        if drained_fluxes:
            if drained_nodes_aligned is None:
                raise RuntimeError("drained_state_nodes dataset is required when processing drained fluxes")
            with dask.annotate(label=f"reduce:drained:{interval}"):
                flux_codes = [FLUX_SPECS[k]["code"] for k in drained_fluxes]
                cube_d = xr.concat([aligned_flux[k] for k in drained_fluxes] + [ref], dim="flux_type").assign_coords(
                    flux_type=("flux_type", flux_codes + [2])
                )
                res_d = xarray_reduce(
                    cube_d, adm0_aligned, drained_nodes_aligned, func="sum",
                    expected_groups=(gadm_adm0_ids, drained_codes_arr),
                    where=where_mask, **flox_sparse_reindex_kwargs(not args.no_sparse),
                ).compute()
            flux_map = {FLUX_SPECS[k]["code"]: FLUX_SPECS[k]["label"] for k in drained_fluxes}
            flux_map[2] = "area__ha"
            df_d = _df_from_result(res_d, flux_map, interval_end_year)
        else:
            df_d = None

        if burned_fluxes:
            if burned_nodes_aligned is None:
                raise RuntimeError("burned_state_nodes dataset is required when processing burned fluxes")
            with dask.annotate(label=f"reduce:burned:{interval}"):
                flux_codes_b = [FLUX_SPECS[k]["code"] for k in burned_fluxes]
                cube_b = xr.concat([aligned_flux[k] for k in burned_fluxes] + [ref], dim="flux_type").assign_coords(
                    flux_type=("flux_type", flux_codes_b + [2])
                )
                res_b = xarray_reduce(
                    cube_b, adm0_aligned, burned_nodes_aligned, func="sum",
                    expected_groups=(gadm_adm0_ids, burned_codes_arr),
                    where=where_mask, **flox_sparse_reindex_kwargs(not args.no_sparse),
                ).compute()
            flux_map_b = {FLUX_SPECS[k]["code"]: FLUX_SPECS[k]["label"] for k in burned_fluxes}
            flux_map_b[2] = "area__ha"
            df_b = _df_from_result(res_b, flux_map_b, interval_end_year)
        else:
            df_b = None

        # Write local Parquet (per interval)
        import shutil
        dest_root = build_output_parquet(args.model_version, args.run_name, args.run_date, interval)
        drained_subdir = "drained_co2_n2o" if drained_only_gases else "drained"
        if df_d is not None:
            local_d = base_dir_drained / interval
            if local_d.exists():
                shutil.rmtree(local_d, ignore_errors=True)
            local_d.mkdir(parents=True, exist_ok=True)
            ds.write_dataset(pa.Table.from_pandas(df_d, preserve_index=False), base_dir=str(local_d),
                             filesystem=local_arrow, format="parquet", existing_data_behavior="overwrite_or_ignore")
            dest_d = posixpath.join(dest_root.rstrip("/"), drained_subdir)
            _upload_dir(fs_s3, local_d, dest_d)
            logger.info("Uploaded drained interval %s → %s", interval, dest_d)
            if not args.keep_local:
                shutil.rmtree(local_d, ignore_errors=True)

        if df_b is not None:
            local_b = base_dir_burned / interval
            if local_b.exists():
                shutil.rmtree(local_b, ignore_errors=True)
            local_b.mkdir(parents=True, exist_ok=True)
            ds.write_dataset(pa.Table.from_pandas(df_b, preserve_index=False), base_dir=str(local_b),
                             filesystem=local_arrow, format="parquet", existing_data_behavior="overwrite_or_ignore")
            dest_b = posixpath.join(dest_root.rstrip("/"), "burned")
            _upload_dir(fs_s3, local_b, dest_b)
            logger.info("Uploaded burned interval %s → %s", interval, dest_b)
            if not args.keep_local:
                shutil.rmtree(local_b, ignore_errors=True)

    if not args.keep_local:
        import shutil
        shutil.rmtree(base_dir_root, ignore_errors=True)

    if client: client.close()
    if cluster: cluster.close()
    uu.stage_duration(start_ts, uu.timestr(), stage)

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Run organic-soils zonal statistics (per-pixel; robust alignment)")
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--run_date", required=True)
    parser.add_argument("--interval_end_years", nargs="+", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--local_output", default="/tmp/zonal_stats")
    parser.add_argument("--keep_local", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_sparse", action="store_true", default=not SPARSE_DEFAULT)
    parser.add_argument("--run_name", default="ogh_standard_model")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS.keys()),
                        help="Datasets to process (default: all)")
    parser.add_argument("--align_tolerance_fraction", type=float, default=0.49,
                        help="Fraction of one pixel for nearest reindex tolerance (default 0.49).")
    parser.add_argument("--leak_warn_threshold", type=float, default=0.002,
                        help="Warn if fraction of flux where adm0==0 exceeds this (default 0.002 = 0.2%).")
    # New: diagnostics & alignment behavior
    parser.add_argument("--diagnostics", choices=["off", "basic", "full"], default="off",
                        help="Flux-over-ocean QA: 'off' (fast, default), 'basic' (sampled), 'full' (slow).")
    parser.add_argument("--force_align", action="store_true",
                        help="Always reindex to pixel_area even if coords already match.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run_local", action="store_true")
    mode.add_argument("--cluster_name", default="zonal_stats")
    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W S E N")
    parser.add_argument("--tile_ids", action="append", help="Comma separated 10×10 tile IDs (e.g., 00N_110E)")
    args = parser.parse_args(argv)
    run(args)

if __name__ == "__main__":
    main()
