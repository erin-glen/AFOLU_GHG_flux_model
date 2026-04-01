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
import difflib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

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
from src.scripts.utilities import drainage_zarr_utilities as dzu

# ------------------------------- config --------------------------------
SPARSE_DEFAULT = True
ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
AREA_SCALE = np.float32(cn.m2_to_ha)

DATASETS: Dict[str, Dict[str, Any]] = {
    "emissions_state_nodes": {"source_var": "emissions_state", "kind": "state", "state_alias": "emissions_state_nodes"},
    "drained_state_nodes": {"source_var": "drained_state", "kind": "state", "state_alias": "drained_state_nodes"},
    "burned_state_nodes":  {"source_var": "burned_state",  "kind": "state", "state_alias": "burned_state_nodes"},
    "drained_total":       {"source_var": "drained_total_Mg_CO2e_ha_yr", "kind": "flux_per_ha_yr"},
    "drained_co2":         {"source_var": "drained_co2_Mg_CO2_ha_yr", "kind": "flux_per_ha_yr"},
    "drained_n2o":         {"source_var": "drained_n2o_Mg_CO2e_ha_yr", "kind": "flux_per_ha_yr"},
    "drained_total_co2":   {"source_var": "drained_co2_Mg_CO2_ha_yr", "kind": "flux_per_ha_yr"},
    "drained_total_ch4":   {"source_var": ["drained_ch4_land_Mg_CO2e_ha_yr", "drained_ch4_ditch_Mg_CO2e_ha_yr"], "kind": "flux_per_ha_yr_sum"},
    "burned_total":        {"source_var": "burned_total_Mg_CO2e_ha_yr",  "kind": "flux_per_ha_yr"},
    "burned_total_co2":    {"source_var": "burned_co2_Mg_CO2_ha_yr",  "kind": "flux_per_ha_yr"},
    "burned_total_ch4":    {"source_var": "burned_ch4_Mg_CO2e_ha_yr",  "kind": "flux_per_ha_yr"},
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

def resolve_mega_zarr_path(model_version: str, run_name: str, run_date: str, interval_type: str,
                           zarr_chunk_size_pixels: Optional[int], logger: logging.Logger) -> str:
    base_path = posixpath.join(ROOT, f"version_{model_version}", "mega_zarr")
    model_type = run_name
    if zarr_chunk_size_pixels is not None:
        return dzu.create_mega_zarr_path(
            base_path,
            zarr_chunk_size_pixels,
            interval_type,
            model_type,
            run_date,
            logger,
        )

    fs = s3fs.S3FileSystem(anon=False)
    prefix = posixpath.join(base_path, model_type, interval_type, "*_pixels", run_date, "mega.zarr")
    matches = sorted(fs.glob(prefix))
    if not matches:
        logger.error(
            "Mega-zarr discovery failed: model_version=%s run_name=%s run_date=%s interval_type=%s searched_glob=s3://%s",
            model_version, run_name, run_date, interval_type, prefix,
        )
        try:
            sibling_prefix = posixpath.join(base_path, model_type, interval_type, "*_pixels")
            sibling_matches = sorted(fs.glob(sibling_prefix))
            if sibling_matches:
                logger.error(
                    "Mega-zarr sibling chunk-size directories sample (%d total): %s",
                    len(sibling_matches),
                    [f"s3://{m.lstrip('/')}" for m in sibling_matches[:5]],
                )
            run_date_prefix = posixpath.join(base_path, model_type, interval_type, "*_pixels", "*", "mega.zarr")
            run_date_matches = sorted(fs.glob(run_date_prefix))
            if run_date_matches:
                logger.error(
                    "Mega-zarr sibling run-date stores sample (%d total): %s",
                    len(run_date_matches),
                    [f"s3://{m.lstrip('/')}" for m in run_date_matches[:5]],
                )
        except Exception as exc:
            logger.warning("Unable to list sibling mega-zarr candidates for diagnostics: %s", exc)
        raise FileNotFoundError(f"No mega-zarr found at pattern: s3://{prefix}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple mega-zarr candidates found ({len(matches)}). Pass --zarr_chunk_size_pixels.")
    return f"s3://{matches[0].lstrip('/')}"


def open_mega_zarr_region(path: str, year: int, bbox: Optional[List[float]], chunk_size: int) -> xr.Dataset:
    dsx = xr.open_zarr(path, consolidated=None, storage_options={"anon": False})
    if "year" not in dsx.coords:
        raise ValueError(f"Mega-zarr missing year coordinate: {path}")
    dsy = dsx.sel(year=year, drop=True)
    if "year" in dsy.coords and "year" not in dsy.dims:
        dsy = dsy.reset_coords("year", drop=True)
    if bbox is not None:
        west, south, east, north = bbox
        x0, x1 = float(dsy.x.values[0]), float(dsy.x.values[-1])
        y0, y1 = float(dsy.y.values[0]), float(dsy.y.values[-1])
        x_slice = slice(min(west, east), max(west, east)) if x0 < x1 else slice(max(east, west), min(east, west))
        y_slice = slice(min(south, north), max(south, north)) if y0 < y1 else slice(max(north, south), min(north, south))
        dsy = dsy.sel(x=x_slice, y=y_slice)
    chunk_dict = {d: chunk_size for d in ("x", "y") if d in dsy.dims}
    if chunk_dict:
        dsy = dsy.chunk(chunk_dict)
    return dsy


def dataset_from_mega(
    spec: Dict[str, Any],
    ds: xr.Dataset,
    *,
    dataset_key: str = "unknown",
    mega_zarr_path: Optional[str] = None,
) -> xr.DataArray:
    """Extract a raw dataset array from the mega-zarr (no alignment/scaling)."""
    source_var = spec["source_var"]

    if isinstance(source_var, list):
        arr = None
        for name in source_var:
            if name not in ds:
                raise KeyError(
                    f"Missing mega-zarr source variable for dataset_key='{dataset_key}': "
                    f"source_var='{name}' path='{mega_zarr_path or 'unknown'}'"
                )
            arr = ds[name] if arr is None else (arr + ds[name])
    else:
        if source_var not in ds:
            raise KeyError(
                f"Missing mega-zarr source variable for dataset_key='{dataset_key}': "
                f"source_var='{source_var}' path='{mega_zarr_path or 'unknown'}'"
            )
        arr = ds[source_var]

    if spec.get("state_alias"):
        arr = arr.rename(spec["state_alias"])
    return arr

def prepare_analysis_array(
    spec: Dict[str, Any],
    dataset_key: str,
    ds: xr.Dataset,
    ref: xr.DataArray,
    tol: float,
    force_align: bool,
    mega_zarr_path: Optional[str] = None,
) -> xr.DataArray:
    """Extract, align to canonical grid, and apply per-ha -> per-pixel scaling when needed."""
    arr = dataset_from_mega(spec, ds, dataset_key=dataset_key, mega_zarr_path=mega_zarr_path)
    arr = align_auto(arr, ref, tol, force_align)
    if spec.get("kind") in {"flux_per_ha_yr", "flux_per_ha_yr_sum"}:
        arr = (arr * ref * AREA_SCALE).astype("float32")
    return drop_scalar_year_coord(arr)


def drop_scalar_year_coord(arr: xr.DataArray) -> xr.DataArray:
    if "year" in arr.coords and "year" not in arr.dims:
        return arr.reset_coords("year", drop=True)
    return arr


def _chunk_structure(arr: xr.DataArray) -> Optional[Tuple[Tuple[int, ...], ...]]:
    chunks = getattr(arr.data, "chunks", None)
    return chunks


def _num_chunks(arr: xr.DataArray) -> Optional[int]:
    nblocks = getattr(arr.data, "numblocks", None)
    if nblocks is None:
        chunks = _chunk_structure(arr)
        if chunks is None:
            return None
        count = 1
        for axis in chunks:
            count *= len(axis)
        return int(count)
    count = 1
    for axis in nblocks:
        count *= int(axis)
    return int(count)


def maybe_persist_reference(
    ref: xr.DataArray,
    *,
    client,
    logger: logging.Logger,
    allow_persist: bool,
    max_chunks: int = 512,
) -> xr.DataArray:
    chunk_structure = _chunk_structure(ref)
    total_chunks = _num_chunks(ref)
    logger.info(
        "Reference grid diagnostics: shape=%s chunks=%s total_chunks=%s allow_persist=%s",
        tuple(ref.shape), chunk_structure, total_chunks, allow_persist,
    )
    if not allow_persist:
        logger.info("Reference grid persist decision: left lazy (persist disabled).")
        return ref
    if total_chunks is None:
        logger.info("Reference grid persist decision: left lazy (non-dask reference or unknown chunks).")
        return ref
    if total_chunks > max_chunks:
        logger.info(
            "Reference grid persist decision: left lazy (chunk count %s exceeds max_chunks=%s).",
            total_chunks, max_chunks,
        )
        return ref
    logger.info("Reference grid persist decision: persisting (chunk count %s <= max_chunks=%s).", total_chunks, max_chunks)
    _ = client
    return ref.persist()


def validate_selected_sources(
    mega_ds: xr.Dataset,
    selected_dataset_keys: List[str],
    *,
    mega_zarr_path: str,
    interval: str,
    logger: logging.Logger,
) -> None:
    required_vars: set[str] = set()
    dataset_to_required: Dict[str, List[str]] = {}
    for key in selected_dataset_keys:
        spec = DATASETS[key]
        src = spec["source_var"]
        src_vars = list(src) if isinstance(src, list) else [src]
        dataset_to_required[key] = src_vars
        required_vars.update(src_vars)

    available_vars = sorted(list(mega_ds.data_vars))
    missing = sorted(var for var in required_vars if var not in mega_ds)
    logger.info(
        "Validating selected mega-zarr sources: interval=%s selected_keys=%s required_source_vars=%s available_var_count=%d",
        interval, selected_dataset_keys, sorted(required_vars), len(available_vars),
    )
    if not missing:
        return

    failed_dataset_keys = sorted(
        key for key, vars_needed in dataset_to_required.items()
        if any(v in missing for v in vars_needed)
    )
    suggestions = {
        m: difflib.get_close_matches(m, available_vars, n=3, cutoff=0.6)
        for m in missing
    }
    raise ValueError(
        "Missing required mega-zarr source variables for zonal stats. "
        f"interval='{interval}' mega_zarr_path='{mega_zarr_path}' "
        f"dataset_keys={failed_dataset_keys} missing_source_vars={missing} "
        f"available_var_sample={available_vars[:20]} suggestions={suggestions}"
    )


def decode_emissions_state_to_legacy(
    emissions_nodes_aligned: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    def _decode(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        drained, burned = zc.unpack_emissions_state_to_legacy(x.astype(np.uint32, copy=False))
        return drained.astype(np.uint32, copy=False), burned.astype(np.uint32, copy=False)

    drained, burned = xr.apply_ufunc(
        _decode,
        emissions_nodes_aligned,
        dask="parallelized",
        output_core_dims=[[], []],
        output_dtypes=[np.uint32, np.uint32],
    )
    drained = drained.rename("drained_state_nodes")
    burned = burned.rename("burned_state_nodes")
    drained = drained.assign_coords(emissions_nodes_aligned.coords).transpose(*emissions_nodes_aligned.dims)
    burned = burned.assign_coords(emissions_nodes_aligned.coords).transpose(*emissions_nodes_aligned.dims)
    return drained, burned


def decode_emissions_state_branch(
    emissions_nodes_aligned: xr.DataArray,
    *,
    branch: str,
) -> xr.DataArray:
    if branch not in {"drained", "burned"}:
        raise ValueError(f"Unsupported branch for decode_emissions_state_branch: {branch}")

    def _decode_one(x: np.ndarray) -> np.ndarray:
        drained, burned = zc.unpack_emissions_state_to_legacy(x.astype(np.uint32, copy=False))
        out = drained if branch == "drained" else burned
        return out.astype(np.uint32, copy=False)

    decoded = xr.apply_ufunc(
        _decode_one,
        emissions_nodes_aligned,
        dask="parallelized",
        output_dtypes=[np.uint32],
    )
    decoded_name = "drained_state_nodes" if branch == "drained" else "burned_state_nodes"
    return decoded.rename(decoded_name).assign_coords(emissions_nodes_aligned.coords).transpose(*emissions_nodes_aligned.dims)

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

def remote_prefix_has_parquet(fs_s3: s3fs.S3FileSystem, prefix: str) -> bool:
    """Return True when any parquet object exists under prefix (recursive)."""
    try:
        for path in fs_s3.find(prefix.rstrip("/")):
            if path.endswith(".parquet"):
                return True
    except FileNotFoundError:
        return False
    return False

def delete_remote_prefix(fs_s3: s3fs.S3FileSystem, prefix: str) -> None:
    """Delete all objects under a specific prefix subtree."""
    target = prefix.rstrip("/")
    try:
        entries = fs_s3.find(target)
    except FileNotFoundError:
        return
    if not entries:
        return
    fs_s3.rm(entries, recursive=False)

def build_exact_tile_mask(ref: xr.DataArray, tile_ids: List[str]) -> xr.DataArray:
    """Build exact union mask for the requested 10x10 tiles using selection slices."""
    tile_masks: List[xr.DataArray] = []
    for tile_id in tile_ids:
        try:
            west, south, east, north = uu.get_10x10_tile_bounds(tile_id)
        except Exception as exc:
            raise ValueError(f"Could not resolve tile_id '{tile_id}': {exc}") from exc
        x0, x1 = float(ref.x.values[0]), float(ref.x.values[-1])
        y0, y1 = float(ref.y.values[0]), float(ref.y.values[-1])
        x_slice = slice(min(west, east), max(west, east)) if x0 < x1 else slice(max(east, west), min(east, west))
        y_slice = slice(min(south, north), max(south, north)) if y0 < y1 else slice(max(north, south), min(north, south))
        selected = ref.sel(x=x_slice, y=y_slice)
        if selected.sizes.get("x", 0) == 0 or selected.sizes.get("y", 0) == 0:
            raise ValueError(f"tile_id '{tile_id}' selects no pixels on the reference grid.")
        x_membership = xr.DataArray(
            np.isin(ref.x.values, selected.x.values),
            dims=("x",),
            coords={"x": ref.x},
        )
        y_membership = xr.DataArray(
            np.isin(ref.y.values, selected.y.values),
            dims=("y",),
            coords={"y": ref.y},
        )
        tile_mask = (x_membership & y_membership).transpose(*ref.dims)
        tile_masks.append(tile_mask)
    if not tile_masks:
        raise ValueError("No pixels selected by --tile_ids on the reference grid.")
    return xr.concat(tile_masks, dim="tile").any(dim="tile")

def _normalize_tile_ids(raw_tile_args: Optional[List[str]]) -> List[str]:
    tiles: List[str] = []
    for item in raw_tile_args or []:
        tiles.extend(t.strip() for t in item.split(",") if t.strip())
    return sorted(set(tiles))

def _normalize_bbox(raw_bbox: Optional[List[float]]) -> Optional[List[float]]:
    if raw_bbox is None:
        return None
    west, south, east, north = [float(v) for v in raw_bbox]
    return [min(west, east), min(south, north), max(west, east), max(south, north)]

def normalized_roi_metadata(
    tiles: List[str],
    bbox: Optional[List[float]],
) -> Dict[str, Any]:
    if tiles:
        return {
            "roi_mode": "tile_ids",
            "bounding_box": _normalize_bbox(bbox),
            "tile_ids": tiles,
        }
    if bbox is not None:
        return {
            "roi_mode": "bounding_box",
            "bounding_box": _normalize_bbox(bbox),
            "tile_ids": None,
        }
    return {
        "roi_mode": "global",
        "bounding_box": None,
        "tile_ids": None,
    }

def resolve_interval_pairs(interval_type: str, requested_years: List[int]) -> List[Tuple[int, int]]:
    if interval_type == cn.intervals_five_year:
        pairs = list(cn.five_year_inventory_periods)
    elif interval_type == cn.intervals_annual:
        annual_ends = dzu.full_model_year_index(cn.intervals_annual)
        pairs = [(year, year) for year in annual_ends]
    else:
        raise ValueError(f"Unsupported interval_type: {interval_type}")
    mapping = {end: (start, end) for start, end in pairs}
    allowed = [end for _, end in pairs]
    requested_deduped = list(dict.fromkeys(requested_years))
    invalid = [y for y in requested_deduped if y not in mapping]
    if invalid:
        raise ValueError(
            f"Invalid interval_end_years for interval_type '{interval_type}': {invalid}. "
            f"Allowed interval-end years: {allowed}"
        )
    return [mapping[y] for y in requested_deduped]

def _validate_flux_selection(selected_names: List[str]) -> Tuple[List[str], List[str]]:
    drained_fluxes = [k for k in selected_names if FLUX_SPECS.get(k, {}).get("group") == "drained"]
    burned_fluxes = [k for k in selected_names if FLUX_SPECS.get(k, {}).get("group") == "burned"]
    if not drained_fluxes and not burned_fluxes:
        raise ValueError(
            "At least one drained or burned flux dataset is required; "
            "state-node datasets alone are not direct zonal-stats outputs."
        )
    return drained_fluxes, burned_fluxes

def build_branch_manifest(
    args: argparse.Namespace,
    interval: str,
    branch: str,
    selected_fluxes: List[str],
    roi_meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "model_version": args.model_version,
        "run_name": args.run_name,
        "run_date": args.run_date,
        "interval": interval,
        "interval_type": args.interval_type,
        "branch": branch,
        "selected_fluxes": selected_fluxes,
        "align_tolerance_fraction": args.align_tolerance_fraction,
        "force_align": bool(args.force_align),
        "roi_mode": roi_meta["roi_mode"],
        "bounding_box": roi_meta["bounding_box"],
        "tile_ids": roi_meta["tile_ids"],
        "adm0_zarr_path": adm0_zarr_path(),
        "pixel_area_zarr_path": PIXEL_AREA_ZARR,
        # Informational-only metadata (not used for skip equivalence checks):
        "diagnostics": args.diagnostics,
    }

def read_remote_manifest(fs_s3: s3fs.S3FileSystem, prefix: str) -> Optional[Dict[str, Any]]:
    manifest_path = posixpath.join(prefix.rstrip("/"), "_zonal_stats_manifest.json")
    try:
        if not fs_s3.exists(manifest_path):
            return None
        with fs_s3.open(manifest_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def write_local_manifest(local_dir: Path, manifest: Dict[str, Any]) -> Path:
    manifest_path = local_dir / "_zonal_stats_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path

def completion_marker_path(prefix: str) -> str:
    return posixpath.join(prefix.rstrip("/"), "_COMPLETE.json")

def read_remote_completion_marker(fs_s3: s3fs.S3FileSystem, prefix: str) -> Optional[Dict[str, Any]]:
    marker_path = completion_marker_path(prefix)
    try:
        if not fs_s3.exists(marker_path):
            return None
        with fs_s3.open(marker_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def write_local_completion_marker(local_dir: Path, marker: Dict[str, Any]) -> Path:
    marker_path = local_dir / "_COMPLETE.json"
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return marker_path

def completion_marker_matches(
    marker: Dict[str, Any],
    *,
    branch: str,
    interval: str,
    run_name: str,
    run_date: str,
    model_version: str,
) -> bool:
    uploaded_file_count = marker.get("uploaded_file_count")
    return (
        marker.get("success") is True
        and marker.get("branch") == branch
        and marker.get("interval") == interval
        and marker.get("run_name") == run_name
        and marker.get("run_date") == run_date
        and marker.get("model_version") == model_version
        and isinstance(uploaded_file_count, int)
        and uploaded_file_count >= 1
    )

def manifests_match(existing: Dict[str, Any], current: Dict[str, Any]) -> bool:
    match_keys = [
        "model_version",
        "run_name",
        "run_date",
        "interval",
        "interval_type",
        "branch",
        "selected_fluxes",
        "align_tolerance_fraction",
        "force_align",
        "roi_mode",
        "bounding_box",
        "tile_ids",
        "adm0_zarr_path",
        "pixel_area_zarr_path",
    ]
    return all(existing.get(key) == current.get(key) for key in match_keys)

def convert_to_coord_dict(flox_result: xr.DataArray) -> dict:
    arr = flox_result.data
    if isinstance(arr, da.Array):
        arr = arr.compute()
    dims = flox_result.dims
    if hasattr(arr, "coords") and hasattr(arr, "data"):
        indices, values = arr.coords, arr.data
    else:
        arr_np = np.asarray(arr)
        nz = np.nonzero(arr_np)
        if len(nz) == 0 or nz[0].size == 0:
            return {
                dim: np.array([], dtype=flox_result.coords[dim].values.dtype)
                for dim in dims
            } | {"value": np.array([], dtype=arr_np.dtype)}
        indices = nz
        values = arr_np[nz]
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
    if "emissions_state_nodes" in df.columns:
        df["combined_state_nodes"] = df["emissions_state_nodes"].astype("uint32", copy=False)
        packed = df["emissions_state_nodes"].fillna(0).astype("uint32").to_numpy(copy=False)
        drained_nodes, burned_nodes = zc.unpack_emissions_state_to_legacy(packed)
        df["drained_state_nodes"] = drained_nodes.astype("uint32", copy=False)
        df["burned_state_nodes"] = burned_nodes.astype("uint32", copy=False)
        df["drained_state_meaning"] = (
            df["drained_state_nodes"].astype("string").str.zfill(8).map(zc.DRAINED_STATE_NODE_MEANINGS)
        )
        df["burned_state_meaning"] = (
            df["burned_state_nodes"].astype("string").str.zfill(8).map(zc.BURNED_STATE_NODE_MEANINGS)
        )
    df["interval_end"] = interval_end
    df.loc[df["flux_type"].eq("area__ha"), "value"] = df["value"] / 10000.0  # m² -> ha
    return df

def leak_ratio_check(drained: xr.DataArray, burned: xr.DataArray, adm0: xr.DataArray,
                     mode: str, threshold: float, logger: logging.Logger,
                     analysis_mask: Optional[xr.DataArray] = None) -> None:
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
    if analysis_mask is not None:
        analysis_mask_local = analysis_mask
        if mode == "basic":
            analysis_mask_local = analysis_mask_local.isel(x=slice(0, None, sx), y=slice(0, None, sy))
        flux_mask = flux_mask & analysis_mask_local
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
    if args.bounding_box and args.tile_ids:
        raise ValueError("--bounding_box and --tile_ids are mutually exclusive.")

    interval_pairs = resolve_interval_pairs(args.interval_type, args.interval_end_years)
    selected_names = ordered_dataset_keys(args.datasets)
    drained_fluxes, burned_fluxes = _validate_flux_selection(selected_names)
    drained_only_gases = set(drained_fluxes) == {"drained_co2", "drained_n2o"}

    tiles = _normalize_tile_ids(args.tile_ids)

    # ROI from bbox or union envelope of requested tiles (mask applied later for exact tile semantics).
    bbox: Optional[List[float]] = None
    if args.bounding_box:
        bbox = _normalize_bbox([float(x) for x in args.bounding_box])
    elif tiles:
        bounds = []
        for tile in tiles:
            try:
                bounds.append(uu.get_10x10_tile_bounds(tile))
            except Exception as exc:
                raise ValueError(f"Could not resolve tile_id '{tile}': {exc}") from exc
        west = min(b[0] for b in bounds); south = min(b[1] for b in bounds)
        east = max(b[2] for b in bounds); north = max(b[3] for b in bounds)
        bbox = _normalize_bbox([west, south, east, north])
    roi_meta = normalized_roi_metadata(tiles, bbox)

    cluster = client = None
    run_local = bool(args.run_local)
    try:
        cluster, client, run_local = uu.connect_to_cluster(cluster_name=args.cluster_name, run_local=args.run_local)
        logger, _ = lu.populate_main_log_header(
            bounding_box=bbox, use_shapefile=False, client=client, cluster=cluster,
            log_note="Organic soils zonal statistics (per-pixel; robust alignment)", run_local=run_local,
            model_type="organic_soils", stage=stage,
        )
        if args.debug:
            logger.setLevel(logging.DEBUG)
        logger.debug("Starting run with args: %s", args)
        logger.info("Diagnostics mode: %s | Force align: %s", args.diagnostics, args.force_align)

        # Open contextual layers
        logger.info("Contextual layer open start: adm0=%s pixel_area=%s", adm0_zarr_path(), PIXEL_AREA_ZARR)
        adm0_zarr = adm0_zarr_path()
        adm0 = open_zarr_region(adm0_zarr, bbox, args.chunk_size).astype("uint32")
        pixel_area = open_zarr_region(PIXEL_AREA_ZARR, bbox, args.chunk_size).astype("float32")
        logger.info("Contextual layer open end: adm0 shape=%s pixel_area shape=%s", tuple(adm0.shape), tuple(pixel_area.shape))

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
        base_dir_combined = base_dir_root / "combined_state"
        fs_s3 = s3fs.S3FileSystem(anon=False)

        # Canonical reference grid = pixel_area
        ref = maybe_persist_reference(
            pixel_area,
            client=client,
            logger=logger,
            allow_persist=bool(args.persist_reference),
            max_chunks=int(args.persist_reference_max_chunks),
        )
        ref_chunks = _num_chunks(ref)
        if (not bbox) and (not tiles) and ref_chunks is not None and ref_chunks >= int(args.full_domain_chunk_warn_threshold):
            logger.warning(
                "High-risk full-domain mode detected: no --bounding_box and no --tile_ids with reference chunk count=%s. "
                "One-shot full-domain interval reduction can be high-memory/high-graph-risk; bounded ROI is safer for debugging.",
                ref_chunks,
            )
        dx = pixel_step(ref)
        tol = float(args.align_tolerance_fraction) * dx
        exact_tile_mask = build_exact_tile_mask(ref, tiles) if tiles else None

        mega_zarr_path = resolve_mega_zarr_path(
            args.model_version,
            args.run_name,
            args.run_date,
            args.interval_type,
            args.zarr_chunk_size_pixels,
            logger,
        )
        logger.info("Using mega-zarr source: %s", mega_zarr_path)

        for interval_start_year, interval_end_year in interval_pairs:
            interval = f"{interval_start_year}_{interval_end_year}"
            dest_root = build_output_parquet(args.model_version, args.run_name, args.run_date, interval)
            drained_subdir = "drained_co2_n2o" if drained_only_gases else "drained"
            dest_d = posixpath.join(dest_root.rstrip("/"), drained_subdir)
            dest_b = posixpath.join(dest_root.rstrip("/"), "burned")
            dest_e = posixpath.join(dest_root.rstrip("/"), "combined_state")
            manifest_d = build_branch_manifest(args, interval, "drained", drained_fluxes, roi_meta)
            manifest_b = build_branch_manifest(args, interval, "burned", burned_fluxes, roi_meta)
            manifest_e = build_branch_manifest(args, interval, "combined_state", drained_fluxes + burned_fluxes, roi_meta)

            need_drained = bool(drained_fluxes)
            need_burned = bool(burned_fluxes)
            need_emissions = bool(drained_fluxes or burned_fluxes)
            drained_exists = need_drained and remote_prefix_has_parquet(fs_s3, dest_d)
            burned_exists = need_burned and remote_prefix_has_parquet(fs_s3, dest_b)
            emissions_exists = need_emissions and remote_prefix_has_parquet(fs_s3, dest_e)
            if not args.overwrite_existing:
                if need_drained and drained_exists:
                    existing_manifest_d = read_remote_manifest(fs_s3, dest_d)
                    if existing_manifest_d is None or not manifests_match(existing_manifest_d, manifest_d):
                        raise ValueError(
                            f"Existing drained output at {dest_d} does not match current selection/config "
                            f"for interval {interval}. Use --overwrite_existing or a different destination."
                        )
                    completion_d = read_remote_completion_marker(fs_s3, dest_d)
                    if completion_d is None or not completion_marker_matches(
                        completion_d,
                        branch="drained",
                        interval=interval,
                        run_name=args.run_name,
                        run_date=args.run_date,
                        model_version=args.model_version,
                    ):
                        raise ValueError(
                            f"Existing drained output at {dest_d} appears incomplete for interval {interval} "
                            f"(missing/invalid completion marker). Use --overwrite_existing or clean the prefix."
                        )
                    logger.info("Skipping drained branch for interval %s; remote parquet+manifest match at %s", interval, dest_d)
                    need_drained = False
                if need_burned and burned_exists:
                    existing_manifest_b = read_remote_manifest(fs_s3, dest_b)
                    if existing_manifest_b is None or not manifests_match(existing_manifest_b, manifest_b):
                        raise ValueError(
                            f"Existing burned output at {dest_b} does not match current selection/config "
                            f"for interval {interval}. Use --overwrite_existing or a different destination."
                        )
                    completion_b = read_remote_completion_marker(fs_s3, dest_b)
                    if completion_b is None or not completion_marker_matches(
                        completion_b,
                        branch="burned",
                        interval=interval,
                        run_name=args.run_name,
                        run_date=args.run_date,
                        model_version=args.model_version,
                    ):
                        raise ValueError(
                            f"Existing burned output at {dest_b} appears incomplete for interval {interval} "
                            f"(missing/invalid completion marker). Use --overwrite_existing or clean the prefix."
                        )
                    logger.info("Skipping burned branch for interval %s; remote parquet+manifest match at %s", interval, dest_b)
                    need_burned = False
                if need_emissions and emissions_exists:
                    existing_manifest_e = read_remote_manifest(fs_s3, dest_e)
                    if existing_manifest_e is None or not manifests_match(existing_manifest_e, manifest_e):
                        raise ValueError(
                            f"Existing combined_state output at {dest_e} does not match current selection/config "
                            f"for interval {interval}. Use --overwrite_existing or a different destination."
                        )
                    completion_e = read_remote_completion_marker(fs_s3, dest_e)
                    if completion_e is None or not completion_marker_matches(
                        completion_e,
                        branch="combined_state",
                        interval=interval,
                        run_name=args.run_name,
                        run_date=args.run_date,
                        model_version=args.model_version,
                    ):
                        raise ValueError(
                            f"Existing combined_state output at {dest_e} appears incomplete for interval {interval} "
                            f"(missing/invalid completion marker). Use --overwrite_existing or clean the prefix."
                        )
                    logger.info("Skipping combined_state branch for interval %s; remote parquet+manifest match at %s", interval, dest_e)
                    need_emissions = False
                if not need_drained and not need_burned and not need_emissions:
                    logger.info("Skipping interval %s entirely; all requested outputs already exist remotely.", interval)
                    continue
            else:
                if need_drained and drained_exists:
                    logger.info("Overwriting drained branch for interval %s by deleting remote prefix %s", interval, dest_d)
                    delete_remote_prefix(fs_s3, dest_d)
                if need_burned and burned_exists:
                    logger.info("Overwriting burned branch for interval %s by deleting remote prefix %s", interval, dest_b)
                    delete_remote_prefix(fs_s3, dest_b)
                if need_emissions and emissions_exists:
                    logger.info("Overwriting combined_state branch for interval %s by deleting remote prefix %s", interval, dest_e)
                    delete_remote_prefix(fs_s3, dest_e)

            logger.info("Processing interval %s : %s", interval, timestr())

            logger.info("Mega-zarr open start: interval=%s year=%s path=%s", interval, interval_end_year, mega_zarr_path)
            mega_ds = open_mega_zarr_region(mega_zarr_path, interval_end_year, bbox, args.chunk_size)
            logger.info("Mega-zarr open end: interval=%s vars=%d dims=%s", interval, len(mega_ds.data_vars), dict(mega_ds.sizes))
            validation_keys = []
            if need_drained:
                validation_keys.extend(drained_fluxes)
                if "drained_state" in mega_ds:
                    validation_keys.append("drained_state_nodes")
            if need_burned:
                validation_keys.extend(burned_fluxes)
                if "burned_state" in mega_ds:
                    validation_keys.append("burned_state_nodes")
            if need_emissions:
                validation_keys.extend(drained_fluxes + burned_fluxes)
            if (need_drained and "drained_state" not in mega_ds) or (need_burned and "burned_state" not in mega_ds):
                validation_keys.append("emissions_state_nodes")
            if need_emissions and "emissions_state" not in mega_ds:
                if "drained_state" in mega_ds and "burned_state" in mega_ds:
                    validation_keys.extend(["drained_state_nodes", "burned_state_nodes"])
                else:
                    validation_keys.append("emissions_state_nodes")
            validation_keys = list(dict.fromkeys(validation_keys))
            validate_selected_sources(
                mega_ds,
                validation_keys,
                mega_zarr_path=mega_zarr_path,
                interval=interval,
                logger=logger,
            )

            zarr_data: Dict[str, xr.DataArray] = {}
            drained_fluxes_for_prepare = drained_fluxes if (need_drained or need_emissions) else []
            burned_fluxes_for_prepare = burned_fluxes if (need_burned or need_emissions) else []
            logger.info(
                "Branch flux preparation start: interval=%s drained_flux_keys=%s burned_flux_keys=%s combined_state=%s",
                interval,
                drained_fluxes_for_prepare,
                burned_fluxes_for_prepare,
                need_emissions,
            )
            for key in list(dict.fromkeys(drained_fluxes_for_prepare + burned_fluxes_for_prepare)):
                zarr_data[key] = prepare_analysis_array(
                    DATASETS[key], key, mega_ds, ref, tol, args.force_align, mega_zarr_path=mega_zarr_path
                ).astype("float32")
            logger.info("Branch flux preparation end: interval=%s prepared_flux_layers=%s", interval, sorted(zarr_data.keys()))

            drained_nodes_aligned = burned_nodes_aligned = None
            emissions_nodes_aligned = emissions_decoded = None
            logger.info("State raster preparation start: interval=%s need_drained=%s need_burned=%s", interval, need_drained, need_burned)
            need_drained_from_emissions = bool(need_drained and ("drained_state" not in mega_ds))
            need_burned_from_emissions = bool(need_burned and ("burned_state" not in mega_ds))
            if need_drained:
                if "drained_state" in mega_ds:
                    drained_nodes_aligned = prepare_analysis_array(
                        DATASETS["drained_state_nodes"], "drained_state_nodes", mega_ds, ref, tol, args.force_align,
                        mega_zarr_path=mega_zarr_path,
                    ).astype("uint32")
                    logger.info("State raster source selected: interval=%s branch=drained source=direct:drained_state", interval)
                elif "emissions_state" in mega_ds:
                    emissions_nodes_aligned = prepare_analysis_array(
                        DATASETS["emissions_state_nodes"], "emissions_state_nodes", mega_ds, ref, tol, args.force_align,
                        mega_zarr_path=mega_zarr_path,
                    ).astype("uint32")
                    if need_burned_from_emissions:
                        logger.info(
                            "Lazy emissions-state decode start: interval=%s target_branch=both source=emissions_state chunks=%s",
                            interval, _chunk_structure(emissions_nodes_aligned),
                        )
                        emissions_decoded = decode_emissions_state_to_legacy(emissions_nodes_aligned)
                        logger.info("Lazy emissions-state decode end: interval=%s target_branch=both", interval)
                        drained_nodes_aligned = emissions_decoded[0]
                    else:
                        logger.info(
                            "Lazy emissions-state decode start: interval=%s target_branch=drained source=emissions_state chunks=%s",
                            interval, _chunk_structure(emissions_nodes_aligned),
                        )
                        drained_nodes_aligned = decode_emissions_state_branch(emissions_nodes_aligned, branch="drained")
                        logger.info("Lazy emissions-state decode end: interval=%s target_branch=drained", interval)
                    logger.info("State raster source selected: interval=%s branch=drained source=derived:emissions_state", interval)
                else:
                    raise ValueError(
                        "Unable to resolve required drained state raster. "
                        f"interval='{interval}' branch='drained' mega_zarr_path='{mega_zarr_path}' "
                        "missing_direct_state_var='drained_state' emissions_state_available=False"
                    )
            if need_burned:
                if "burned_state" in mega_ds:
                    burned_nodes_aligned = prepare_analysis_array(
                        DATASETS["burned_state_nodes"], "burned_state_nodes", mega_ds, ref, tol, args.force_align,
                        mega_zarr_path=mega_zarr_path,
                    ).astype("uint32")
                    logger.info("State raster source selected: interval=%s branch=burned source=direct:burned_state", interval)
                elif "emissions_state" in mega_ds:
                    if emissions_decoded is None:
                        if emissions_nodes_aligned is None:
                            emissions_nodes_aligned = prepare_analysis_array(
                                DATASETS["emissions_state_nodes"], "emissions_state_nodes", mega_ds, ref, tol, args.force_align,
                                mega_zarr_path=mega_zarr_path,
                            ).astype("uint32")
                        if need_drained_from_emissions:
                            logger.info(
                                "Lazy emissions-state decode start: interval=%s target_branch=both source=emissions_state chunks=%s",
                                interval, _chunk_structure(emissions_nodes_aligned),
                            )
                            emissions_decoded = decode_emissions_state_to_legacy(emissions_nodes_aligned)
                            logger.info("Lazy emissions-state decode end: interval=%s target_branch=both", interval)
                        else:
                            logger.info(
                                "Lazy emissions-state decode start: interval=%s target_branch=burned source=emissions_state chunks=%s",
                                interval, _chunk_structure(emissions_nodes_aligned),
                            )
                            burned_nodes_aligned = decode_emissions_state_branch(emissions_nodes_aligned, branch="burned")
                            logger.info("Lazy emissions-state decode end: interval=%s target_branch=burned", interval)
                    if burned_nodes_aligned is None:
                        burned_nodes_aligned = emissions_decoded[1]
                    logger.info("State raster source selected: interval=%s branch=burned source=derived:emissions_state", interval)
                else:
                    raise ValueError(
                        "Unable to resolve required burned state raster. "
                        f"interval='{interval}' branch='burned' mega_zarr_path='{mega_zarr_path}' "
                        "missing_direct_state_var='burned_state' emissions_state_available=False"
                    )
            logger.info("State raster preparation end: interval=%s", interval)

            emissions_nodes_for_reduce = emissions_nodes_aligned
            if need_emissions and emissions_nodes_for_reduce is None:
                if "emissions_state" in mega_ds:
                    emissions_nodes_for_reduce = prepare_analysis_array(
                        DATASETS["emissions_state_nodes"], "emissions_state_nodes", mega_ds, ref, tol, args.force_align,
                        mega_zarr_path=mega_zarr_path,
                    ).astype("uint32")
                    logger.info("State raster source selected: interval=%s branch=combined_state source=direct:emissions_state", interval)
                elif ("drained_state" in mega_ds) and ("burned_state" in mega_ds):
                    drained_for_pack = (
                        drained_nodes_aligned
                        if drained_nodes_aligned is not None
                        else prepare_analysis_array(
                            DATASETS["drained_state_nodes"], "drained_state_nodes", mega_ds, ref, tol, args.force_align,
                            mega_zarr_path=mega_zarr_path,
                        ).astype("uint32")
                    )
                    burned_for_pack = (
                        burned_nodes_aligned
                        if burned_nodes_aligned is not None
                        else prepare_analysis_array(
                            DATASETS["burned_state_nodes"], "burned_state_nodes", mega_ds, ref, tol, args.force_align,
                            mega_zarr_path=mega_zarr_path,
                        ).astype("uint32")
                    )
                    emissions_nodes_for_reduce = xr.apply_ufunc(
                        zc.pack_emissions_state,
                        drained_for_pack,
                        burned_for_pack,
                        dask="parallelized",
                        output_dtypes=[np.uint32],
                    ).rename("emissions_state_nodes")
                    logger.info("State raster source selected: interval=%s branch=combined_state source=derived:pack(drained_state,burned_state)", interval)
                else:
                    raise ValueError(
                        "Unable to resolve required emissions state raster. "
                        f"interval='{interval}' branch='combined_state' mega_zarr_path='{mega_zarr_path}' "
                        "requires emissions_state or (drained_state and burned_state)"
                    )

            # Smart alignment (skip if coords already match)
            adm0_aligned = align_auto(adm0, ref, tol, args.force_align)

            # Optional flux-over-ocean diagnostic (off by default)
            if {"drained_total", "burned_total"}.issubset(zarr_data):
                leak_ratio_check(
                    zarr_data["drained_total"], zarr_data["burned_total"], adm0_aligned,
                    args.diagnostics, args.leak_warn_threshold, logger, analysis_mask=exact_tile_mask
                )
            else:
                logger.info(
                    "Skipping flux-over-ocean diagnostic because drained_total and burned_total were not both selected."
                )

            where_mask = (adm0_aligned > 0) if exact_tile_mask is None else ((adm0_aligned > 0) & exact_tile_mask)

            if need_drained:
                if drained_nodes_aligned is None:
                    raise RuntimeError("drained_state_nodes dataset is required when processing drained fluxes")
                with dask.annotate(label=f"reduce:drained:{interval}"):
                    flux_codes = [FLUX_SPECS[k]["code"] for k in drained_fluxes]
                    area_layer_d = ref if ref.dtype == np.float32 else ref.astype("float32")
                    area_layer_d = drop_scalar_year_coord(area_layer_d)
                    logger.info(
                        "Drained concat inputs: interval=%s branch=drained first_flux_dims=%s first_flux_coords=%s first_flux_dtype=%s area_dims=%s area_coords=%s area_dtype=%s",
                        interval,
                        zarr_data[drained_fluxes[0]].dims,
                        list(zarr_data[drained_fluxes[0]].coords),
                        zarr_data[drained_fluxes[0]].dtype,
                        area_layer_d.dims,
                        list(area_layer_d.coords),
                        area_layer_d.dtype,
                    )
                    cube_d = xr.concat(
                        [zarr_data[k] for k in drained_fluxes] + [area_layer_d],
                        dim="flux_type",
                        coords="minimal",
                    ).assign_coords(
                        flux_type=("flux_type", flux_codes + [2])
                    )
                    logger.info(
                        "Drained reduction dtypes: interval=%s branch=drained first_flux_dtype=%s area_dtype=%s cube_dtype=%s",
                        interval, zarr_data[drained_fluxes[0]].dtype, area_layer_d.dtype, cube_d.dtype,
                    )
                    logger.info(
                        "Drained reduction start: interval=%s branch=drained flux_layers=%d cube_dims=%s cube_chunks=%s cube_numblocks=%s",
                        interval, len(drained_fluxes), dict(cube_d.sizes), _chunk_structure(cube_d), getattr(cube_d.data, "numblocks", None),
                    )
                    res_d = xarray_reduce(
                        cube_d, adm0_aligned, drained_nodes_aligned, func="sum",
                        expected_groups=(gadm_adm0_ids, drained_codes_arr),
                        where=where_mask, **flox_sparse_reindex_kwargs(not args.no_sparse),
                    ).compute()
                    logger.info("Drained reduction end: interval=%s", interval)
                flux_map = {FLUX_SPECS[k]["code"]: FLUX_SPECS[k]["label"] for k in drained_fluxes}
                flux_map[2] = "area__ha"
                df_d = _df_from_result(res_d, flux_map, interval_end_year)
            else:
                df_d = None

            if need_burned:
                if burned_nodes_aligned is None:
                    raise RuntimeError("burned_state_nodes dataset is required when processing burned fluxes")
                with dask.annotate(label=f"reduce:burned:{interval}"):
                    flux_codes_b = [FLUX_SPECS[k]["code"] for k in burned_fluxes]
                    area_layer_b = ref if ref.dtype == np.float32 else ref.astype("float32")
                    area_layer_b = drop_scalar_year_coord(area_layer_b)
                    logger.info(
                        "Burned concat inputs: interval=%s branch=burned first_flux_dims=%s first_flux_coords=%s first_flux_dtype=%s area_dims=%s area_coords=%s area_dtype=%s",
                        interval,
                        zarr_data[burned_fluxes[0]].dims,
                        list(zarr_data[burned_fluxes[0]].coords),
                        zarr_data[burned_fluxes[0]].dtype,
                        area_layer_b.dims,
                        list(area_layer_b.coords),
                        area_layer_b.dtype,
                    )
                    cube_b = xr.concat(
                        [zarr_data[k] for k in burned_fluxes] + [area_layer_b],
                        dim="flux_type",
                        coords="minimal",
                    ).assign_coords(
                        flux_type=("flux_type", flux_codes_b + [2])
                    )
                    logger.info(
                        "Burned reduction dtypes: interval=%s branch=burned first_flux_dtype=%s area_dtype=%s cube_dtype=%s",
                        interval, zarr_data[burned_fluxes[0]].dtype, area_layer_b.dtype, cube_b.dtype,
                    )
                    logger.info(
                        "Burned reduction start: interval=%s branch=burned flux_layers=%d cube_dims=%s cube_chunks=%s cube_numblocks=%s",
                        interval, len(burned_fluxes), dict(cube_b.sizes), _chunk_structure(cube_b), getattr(cube_b.data, "numblocks", None),
                    )
                    res_b = xarray_reduce(
                        cube_b, adm0_aligned, burned_nodes_aligned, func="sum",
                        expected_groups=(gadm_adm0_ids, burned_codes_arr),
                        where=where_mask, **flox_sparse_reindex_kwargs(not args.no_sparse),
                    ).compute()
                    logger.info("Burned reduction end: interval=%s", interval)
                flux_map_b = {FLUX_SPECS[k]["code"]: FLUX_SPECS[k]["label"] for k in burned_fluxes}
                flux_map_b[2] = "area__ha"
                df_b = _df_from_result(res_b, flux_map_b, interval_end_year)
            else:
                df_b = None

            if need_emissions:
                if emissions_nodes_for_reduce is None:
                    raise RuntimeError("emissions_state_nodes dataset is required when processing combined_state branch")
                with dask.annotate(label=f"reduce:combined_state:{interval}"):
                    emissions_flux_keys = drained_fluxes + burned_fluxes
                    flux_codes_e = [FLUX_SPECS[k]["code"] for k in emissions_flux_keys]
                    area_layer_e = ref if ref.dtype == np.float32 else ref.astype("float32")
                    area_layer_e = drop_scalar_year_coord(area_layer_e)
                    cube_e = xr.concat(
                        [zarr_data[k] for k in emissions_flux_keys] + [area_layer_e],
                        dim="flux_type",
                        coords="minimal",
                    ).assign_coords(
                        flux_type=("flux_type", flux_codes_e + [2])
                    )
                    logger.info(
                        "Emissions-state reduction start: interval=%s flux_layers=%d cube_dims=%s cube_chunks=%s cube_numblocks=%s",
                        interval, len(emissions_flux_keys), dict(cube_e.sizes), _chunk_structure(cube_e), getattr(cube_e.data, "numblocks", None),
                    )
                    res_e = xarray_reduce(
                        cube_e, adm0_aligned, emissions_nodes_for_reduce, func="sum",
                        where=where_mask, **flox_sparse_reindex_kwargs(not args.no_sparse),
                    ).compute()
                    logger.info("Emissions-state reduction end: interval=%s", interval)
                flux_map_e = {FLUX_SPECS[k]["code"]: FLUX_SPECS[k]["label"] for k in emissions_flux_keys}
                flux_map_e[2] = "area__ha"
                df_e = _df_from_result(res_e, flux_map_e, interval_end_year)
            else:
                df_e = None

            # Write local Parquet (per interval)
            import shutil
            if df_d is not None:
                local_d = base_dir_drained / interval
                if local_d.exists():
                    shutil.rmtree(local_d, ignore_errors=True)
                local_d.mkdir(parents=True, exist_ok=True)
                ds.write_dataset(pa.Table.from_pandas(df_d, preserve_index=False), base_dir=str(local_d),
                                 filesystem=local_arrow, format="parquet", existing_data_behavior="overwrite_or_ignore")
                write_local_manifest(local_d, manifest_d)
                uploaded_count = _upload_dir(fs_s3, local_d, dest_d)
                marker_d = {
                    "success": True,
                    "branch": "drained",
                    "interval": interval,
                    "run_name": args.run_name,
                    "run_date": args.run_date,
                    "model_version": args.model_version,
                    "uploaded_file_count": uploaded_count,
                    "manifest_match_keys_version": 1,
                }
                local_marker_d = write_local_completion_marker(local_d, marker_d)
                fs_s3.put(str(local_marker_d), completion_marker_path(dest_d))
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
                write_local_manifest(local_b, manifest_b)
                uploaded_count = _upload_dir(fs_s3, local_b, dest_b)
                marker_b = {
                    "success": True,
                    "branch": "burned",
                    "interval": interval,
                    "run_name": args.run_name,
                    "run_date": args.run_date,
                    "model_version": args.model_version,
                    "uploaded_file_count": uploaded_count,
                    "manifest_match_keys_version": 1,
                }
                local_marker_b = write_local_completion_marker(local_b, marker_b)
                fs_s3.put(str(local_marker_b), completion_marker_path(dest_b))
                logger.info("Uploaded burned interval %s → %s", interval, dest_b)
                if not args.keep_local:
                    shutil.rmtree(local_b, ignore_errors=True)

            if df_e is not None:
                local_e = base_dir_combined / interval
                if local_e.exists():
                    shutil.rmtree(local_e, ignore_errors=True)
                local_e.mkdir(parents=True, exist_ok=True)
                ds.write_dataset(pa.Table.from_pandas(df_e, preserve_index=False), base_dir=str(local_e),
                                 filesystem=local_arrow, format="parquet", existing_data_behavior="overwrite_or_ignore")
                write_local_manifest(local_e, manifest_e)
                uploaded_count = _upload_dir(fs_s3, local_e, dest_e)
                marker_e = {
                    "success": True,
                    "branch": "combined_state",
                    "interval": interval,
                    "run_name": args.run_name,
                    "run_date": args.run_date,
                    "model_version": args.model_version,
                    "uploaded_file_count": uploaded_count,
                    "manifest_match_keys_version": 1,
                }
                local_marker_e = write_local_completion_marker(local_e, marker_e)
                fs_s3.put(str(local_marker_e), completion_marker_path(dest_e))
                logger.info("Uploaded combined_state interval %s → %s", interval, dest_e)
                if not args.keep_local:
                    shutil.rmtree(local_e, ignore_errors=True)

        if not args.keep_local:
            import shutil
            shutil.rmtree(base_dir_root, ignore_errors=True)
    finally:
        if client:
            client.close()
        if cluster:
            cluster.close()
        uu.stage_duration(start_ts, uu.timestr(), stage)

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Run organic-soils zonal statistics (per-pixel; robust alignment)")
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--run_date", required=True)
    parser.add_argument("--interval_type", default="five_year", choices=[cn.intervals_annual, cn.intervals_five_year])
    parser.add_argument("--zarr_chunk_size_pixels", type=int, default=None,
                        help="Chunk-size segment in mega-zarr path (e.g., 1). If omitted, auto-discovers unique *_pixels store.")
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
    parser.add_argument("--persist_reference", action="store_true",
                        help="Optionally persist reference pixel_area when chunk count is modest.")
    parser.add_argument("--persist_reference_max_chunks", type=int, default=512,
                        help="Maximum reference chunk count eligible for --persist_reference (default: 512).")
    parser.add_argument("--full_domain_chunk_warn_threshold", type=int, default=1024,
                        help="Warn in global mode when reference chunk count exceeds this threshold (default: 1024).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run_local", action="store_true")
    mode.add_argument("--cluster_name", default="zonal_stats")
    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W S E N")
    parser.add_argument("--tile_ids", action="append", help="Comma separated 10×10 tile IDs (e.g., 00N_110E)")
    parser.add_argument("--overwrite_existing", action="store_true",
                        help="Delete and replace existing remote interval subtree(s) before upload.")
    args = parser.parse_args(argv)
    run(args)

if __name__ == "__main__":
    main()
