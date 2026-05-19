# -*- coding: utf-8 -*-
"""Run organic-soils zonal statistics (per-pixel only; robust alignment; per-interval upload).

Production-lean defaults:
- Diagnostics OFF by default (skip flux-over-ocean full scan).
- Smart alignment: skip reindex_like if coords already equal to pixel_area.

python -m src.scripts.zonal_statistics.02_run_zonal_stats \
  --interval_end_years 2024 \
  --cluster_name drainage_cluster \
  --run_date 20260417 \
  --model_version 0_1_4 \
  --run_name ogh_biome_thresholds \
  --chunk_size 10000

python -m src.scripts.zonal_statistics.02_run_zonal_stats \
  --interval_end_years 2024 \
  --cluster_name drainage_cluster \
  --run_date 20260403 \
  --model_version 0_1_2 \
  --run_name zarr_test_full \
  --chunk_size 10000 \
  --tile_ids 00N_110E,10N_120E \
  --datasets drained_total burned_total drained_co2_onsite drained_n2o

# Explicit tiled execution (recommended for global/disjoint/large runs)
python -m src.scripts.zonal_statistics.02_run_zonal_stats \
  --interval_end_years 2024 \
  --cluster_name drainage_cluster \
  --run_date 20251118 \
  --model_version 0_9_7 \
  --run_name ogh_sensitivity_500m_10 \
  --execution_mode tile \
  --tile_ids 00N_110E,10N_120E \
  --chunk_size 10000 \
  --datasets drained_total burned_total \
  --keep_tile_stage

# Auto execution mode with threshold control (auto switches ROI/tile)
python -m src.scripts.zonal_statistics.02_run_zonal_stats \
  --interval_end_years 2024 \
  --cluster_name drainage_cluster \
  --run_date 20251118 \
  --model_version 0_9_7 \
  --run_name ogh_sensitivity_500m_10 \
  --execution_mode auto \
  --auto_tile_threshold_tiles 8 \
  --bounding_box 110 -10 120 0 \
  --datasets drained_total burned_total

"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
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
from src.scripts.utilities import local_output_paths as lop

# ------------------------------- config --------------------------------
SPARSE_DEFAULT = True
ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
AREA_SCALE = np.float32(cn.m2_to_ha)

STATE_DATASETS: Dict[str, Dict[str, Any]] = {
    "combined_state_nodes": {"source_var": "combined_state", "kind": "state", "state_alias": "combined_state_nodes"},
    "drained_state_nodes": {"source_var": "drained_state", "kind": "state", "state_alias": "drained_state_nodes"},
    "burned_state_nodes": {"source_var": "burned_state", "kind": "state", "state_alias": "burned_state_nodes"},
}

FLUX_DATASETS: Dict[str, Dict[str, Any]] = {
    "drained_total": {"source_var": "drained_total_Mg_CO2e_ha_yr", "kind": "flux_per_ha_yr"}, #standard drained output
    "drained_co2_onsite": {"source_var": "drained_co2_Mg_CO2_ha_yr", "kind": "flux_per_ha_yr"}, #subset for FAO/NGHGI
    "drained_co2_offsite": {"source_var": "drained_co2_offsite_Mg_CO2_ha_yr", "kind": "flux_per_ha_yr"}, #off-site DOC CO2
    "drained_n2o": {"source_var": "drained_n2o_Mg_CO2e_ha_yr", "kind": "flux_per_ha_yr"}, #subset for FAO/NGHGI
    "drained_total_co2": {"source_var": ["drained_co2_Mg_CO2_ha_yr", "drained_co2_offsite_Mg_CO2_ha_yr"], "kind": "flux_per_ha_yr_sum"}, #for LULUCF
    "drained_total_ch4": {"source_var": ["drained_ch4_land_Mg_CO2e_ha_yr", "drained_ch4_ditch_Mg_CO2e_ha_yr"], "kind": "flux_per_ha_yr_sum"}, #for LULUCF
    "burned_total": {"source_var": "burned_total_Mg_CO2e_ha_yr", "kind": "flux_per_ha_yr"}, #standard burned output
    "burned_total_co2": {"source_var": "burned_co2_Mg_CO2_ha_yr", "kind": "flux_per_ha_yr"}, #for LULUCF
    "burned_total_ch4": {"source_var": "burned_ch4_Mg_CO2e_ha_yr", "kind": "flux_per_ha_yr"}, #for LULUCF
}

FLUX_SPECS = {
    "drained_total": {"code": 0, "label": "drained_total_Mg_CO2e", "group": "drained"},
    "drained_co2_onsite": {"code": 3, "label": "drained_co2_onsite_Mg_CO2", "group": "drained"},
    "drained_co2_offsite": {"code": 9, "label": "drained_co2_offsite_Mg_CO2", "group": "drained"},
    "drained_n2o": {"code": 4, "label": "drained_n2o_Mg_CO2e", "group": "drained"},
    "drained_total_co2": {"code": 5, "label": "drained_total_co2_Mg_CO2", "group": "drained"},
    "drained_total_ch4": {"code": 6, "label": "drained_total_ch4_Mg_CO2e", "group": "drained"},
    "burned_total": {"code": 1, "label": "burned_total_Mg_CO2e", "group": "burned"},
    "burned_total_co2": {"code": 7, "label": "burned_total_co2_Mg_CO2", "group": "burned"},
    "burned_total_ch4": {"code": 8, "label": "burned_total_ch4_Mg_CO2e", "group": "burned"},
}
FLUX_DATASET_ALIASES: Dict[str, str] = {
    # One-release compatibility for older CLI args/manifests. New outputs use
    # the explicit on-site name so off-site CO2 cannot be mistaken for absent.
    "drained_co2": "drained_co2_onsite",
}
ALL_DATASETS: Dict[str, Dict[str, Any]] = {**STATE_DATASETS, **FLUX_DATASETS}

CANONICAL_CONTEXTUAL_GROUPER_ORDER = ("wdpa", "kba", "drivers_of_loss")


def default_local_output(model_version: str, run_name: str, run_date: str) -> str:
    """Return the default local staging directory for this zonal-stats run."""

    return lop.zonal_stats_staging_dir(model_version, run_name, run_date)


def _expected_groups_with_zero(codes: Any, *, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(codes, dtype=dtype)
    if arr.size == 0:
        return np.array([0], dtype=dtype)
    if not np.any(arr == 0):
        arr = np.concatenate([np.array([0], dtype=dtype), arr.astype(dtype, copy=False)])
    return np.unique(arr.astype(dtype, copy=False))


OPTIONAL_CONTEXTUAL_GROUPERS: Dict[str, Dict[str, Any]] = {
    "wdpa": {
        "name": cn.WDPA_pattern,
        "zarr_path": cn.WDPA_zarr_path,
        "expected_groups": _expected_groups_with_zero(cn.WDPA_codes, dtype=np.uint16),
        "dtype": np.uint16,
        "source_label": "WDPA",
    },
    "kba": {
        "name": cn.KBA_pattern,
        "zarr_path": cn.KBA_zarr_path,
        "expected_groups": _expected_groups_with_zero(cn.KBA_codes, dtype=np.uint16),
        "dtype": np.uint16,
        "source_label": "KBA",
    },
    "drivers_of_loss": {
        "name": cn.drivers_of_loss_pattern,
        "zarr_path": cn.drivers_of_loss_zarr_path,
        "expected_groups": _expected_groups_with_zero(cn.drivers_codes, dtype=np.uint8),
        "dtype": np.uint8,
        "source_label": "drivers_of_TCL_1_km",
    },
}


def ordered_dataset_keys(selected: Optional[List[str]]) -> List[str]:
    if not selected:
        return list(FLUX_DATASETS.keys())
    selected_canonical = {FLUX_DATASET_ALIASES.get(k, k) for k in selected}
    return [k for k in FLUX_DATASETS if k in selected_canonical]

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

def build_aggregated_tile_prefix(
    *,
    model_version: str,
    dataset: str,
    run_name: str,
    interval_type: str,
    interval: str,
    pixel_resolution: str,
    run_date: str,
) -> str:
    return (
        posixpath.join(
            ROOT,
            f"version_{model_version}",
            dataset,
            run_name,
            f"{interval_type}_intervals",
            interval,
            pixel_resolution,
            run_date,
        ).rstrip("/")
        + "/"
    )

def extract_tile_id_from_path(path: str) -> Optional[str]:
    match = re.search(cn.tile_id_pattern, posixpath.basename(str(path)))
    return match.group(0) if match else None

def discover_aggregated_data_tile_ids(fs_s3: s3fs.S3FileSystem, prefix: str) -> List[str]:
    matches = sorted(fs_s3.glob(posixpath.join(prefix.rstrip("/"), "*.tif")))
    tile_ids = sorted(
        {
            tile_id
            for tile_id in (extract_tile_id_from_path(path) for path in matches)
            if tile_id is not None
        }
    )
    if not tile_ids:
        raise FileNotFoundError(
            f"No aggregated tile rasters found for zonal stats data-tile filter at {prefix}"
        )
    return tile_ids

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


def patch_zarr_asyncarray_config_on_workers(client, logger: logging.Logger) -> None:
    """Patch older worker-side zarr for AsyncArray objects serialized by newer clients."""

    def _patch() -> dict:
        try:
            import zarr
            from zarr.core.array import AsyncArray, parse_array_config

            if not hasattr(AsyncArray, "_config"):
                AsyncArray._config = parse_array_config(None)
            return {
                "ok": True,
                "zarr": getattr(zarr, "__version__", "unknown"),
                "patched": hasattr(AsyncArray, "_config"),
            }
        except Exception as exc:  # pragma: no cover - defensive worker compatibility hook
            return {"ok": False, "error": repr(exc)}

    try:
        results = client.run(_patch)
    except Exception as exc:
        logger.warning("Unable to apply worker zarr AsyncArray compatibility patch: %s", exc)
        return

    failed = {worker: result for worker, result in results.items() if not result.get("ok")}
    if failed:
        logger.warning("Worker zarr AsyncArray compatibility patch failures: %s", failed)
    else:
        versions = sorted({result.get("zarr", "unknown") for result in results.values()})
        logger.info(
            "Worker zarr AsyncArray compatibility patch applied on %d workers; zarr_versions=%s",
            len(results),
            versions,
        )


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
        spec = ALL_DATASETS[key]
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


def decode_combined_state_to_legacy(
    combined_nodes_aligned: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    def _decode(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        drained, burned = zc.unpack_combined_state_to_legacy(x.astype(np.uint32, copy=False))
        return drained.astype(np.uint32, copy=False), burned.astype(np.uint32, copy=False)

    drained, burned = xr.apply_ufunc(
        _decode,
        combined_nodes_aligned,
        dask="parallelized",
        output_core_dims=[[], []],
        output_dtypes=[np.uint32, np.uint32],
    )
    drained = drained.rename("drained_state_nodes")
    burned = burned.rename("burned_state_nodes")
    drained = drained.assign_coords(combined_nodes_aligned.coords).transpose(*combined_nodes_aligned.dims)
    burned = burned.assign_coords(combined_nodes_aligned.coords).transpose(*combined_nodes_aligned.dims)
    return drained, burned


def decode_combined_state_branch(
    combined_nodes_aligned: xr.DataArray,
    *,
    branch: str,
) -> xr.DataArray:
    if branch not in {"drained", "burned"}:
        raise ValueError(f"Unsupported branch for decode_combined_state_branch: {branch}")

    def _decode_one(x: np.ndarray) -> np.ndarray:
        drained, burned = zc.unpack_combined_state_to_legacy(x.astype(np.uint32, copy=False))
        out = drained if branch == "drained" else burned
        return out.astype(np.uint32, copy=False)

    decoded = xr.apply_ufunc(
        _decode_one,
        combined_nodes_aligned,
        dask="parallelized",
        output_dtypes=[np.uint32],
    )
    decoded_name = "drained_state_nodes" if branch == "drained" else "burned_state_nodes"
    return decoded.rename(decoded_name).assign_coords(combined_nodes_aligned.coords).transpose(*combined_nodes_aligned.dims)

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


def build_bbox_mask(ref: xr.DataArray, bbox: List[float]) -> xr.DataArray:
    """Build exact mask for pixels intersecting a requested bbox."""
    west, south, east, north = _normalize_bbox(bbox)
    x0, x1 = float(ref.x.values[0]), float(ref.x.values[-1])
    y0, y1 = float(ref.y.values[0]), float(ref.y.values[-1])
    x_slice = slice(min(west, east), max(west, east)) if x0 < x1 else slice(max(east, west), min(east, west))
    y_slice = slice(min(south, north), max(south, north)) if y0 < y1 else slice(max(north, south), min(north, south))
    selected = ref.sel(x=x_slice, y=y_slice)
    if selected.sizes.get("x", 0) == 0 or selected.sizes.get("y", 0) == 0:
        raise ValueError("Requested --bounding_box selects no pixels on the reference grid.")
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
    return (x_membership & y_membership).transpose(*ref.dims)

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

def resolve_flux_selection(selected_names: List[str], logger: Optional[logging.Logger] = None) -> Dict[str, List[str]]:
    ignored_state_keys = [k for k in selected_names if k in STATE_DATASETS]
    if ignored_state_keys and logger is not None:
        logger.warning("Ignoring state dataset keys passed to --datasets (flux-only selection now): %s", ignored_state_keys)
    canonical_names = list(dict.fromkeys(FLUX_DATASET_ALIASES.get(k, k) for k in selected_names))
    filtered = [k for k in canonical_names if k in FLUX_DATASETS]
    drained_fluxes = [k for k in filtered if FLUX_SPECS.get(k, {}).get("group") == "drained"]
    burned_fluxes = [k for k in filtered if FLUX_SPECS.get(k, {}).get("group") == "burned"]
    if not filtered:
        raise ValueError(
            "At least one drained or burned flux dataset is required; "
            "state-node datasets alone are not direct zonal-stats outputs."
        )
    return {
        "selected_fluxes_ordered": filtered,
        "drained_fluxes": drained_fluxes,
        "burned_fluxes": burned_fluxes,
    }


def resolve_requested_contextual_groupers(raw_values: Optional[List[str]]) -> List[str]:
    requested = [str(v).strip().lower() for v in (raw_values or []) if str(v).strip()]
    requested_set = set(requested)
    return [k for k in CANONICAL_CONTEXTUAL_GROUPER_ORDER if k in requested_set]


def open_optional_contextual_grouper(
    spec: Dict[str, Any],
    bbox: Optional[List[float]],
    chunk_size: int,
    logger: logging.Logger,
) -> xr.DataArray:
    logger.info("Contextual grouper open start: name=%s source=%s path=%s", spec["name"], spec["source_label"], spec["zarr_path"])
    arr = open_zarr_region(spec["zarr_path"], bbox, chunk_size)
    logger.info("Contextual grouper open end: name=%s dims=%s chunks=%s", spec["name"], dict(arr.sizes), _chunk_structure(arr))
    return arr


def prepare_contextual_grouper(
    arr: xr.DataArray,
    ref: xr.DataArray,
    tol: float,
    force_align: bool,
    name: str,
    dtype: Any,
) -> xr.DataArray:
    out = align_auto(arr, ref, tol, force_align)
    out = drop_scalar_year_coord(out).astype(dtype).rename(name)
    return out


def zero_like_contextual(ref: xr.DataArray, name: str, dtype: Any) -> xr.DataArray:
    return xr.zeros_like(ref, dtype=dtype).rename(name)


def resolve_contextual_groupers_for_extent(
    *,
    requested_keys: List[str],
    bbox: Optional[List[float]],
    chunk_size: int,
    ref: xr.DataArray,
    tol: float,
    force_align: bool,
    logger: logging.Logger,
) -> tuple[List[xr.DataArray], List[np.ndarray]]:
    resolved_arrays: List[xr.DataArray] = []
    resolved_expected_groups: List[np.ndarray] = []
    for key in requested_keys:
        spec = OPTIONAL_CONTEXTUAL_GROUPERS[key]
        arr = open_optional_contextual_grouper(spec, bbox, chunk_size, logger)
        if arr.sizes.get("x", 0) == 0 or arr.sizes.get("y", 0) == 0:
            logger.warning(
                "Contextual grouper empty for extent; substituting zero-like fallback: key=%s name=%s bbox=%s",
                key, spec["name"], bbox,
            )
            prepared = zero_like_contextual(ref, spec["name"], spec["dtype"])
        else:
            prepared = prepare_contextual_grouper(arr, ref, tol, force_align, spec["name"], spec["dtype"])
        logger.info(
            "Contextual grouper prepared: key=%s name=%s dims=%s chunks=%s dtype=%s",
            key, spec["name"], dict(prepared.sizes), _chunk_structure(prepared), prepared.dtype,
        )
        resolved_arrays.append(prepared)
        resolved_expected_groups.append(spec["expected_groups"])
    return resolved_arrays, resolved_expected_groups


def tile_ids_for_global() -> List[str]:
    return sorted(cn.tile_id_list)


def bbox_intersects_tile(bbox: List[float], tile_bounds: List[float]) -> bool:
    west, south, east, north = bbox
    tw, ts, te, tn = tile_bounds
    return not (east <= tw or te <= west or north <= ts or tn <= south)


def tile_ids_for_bbox(bbox: List[float]) -> List[str]:
    return [
        tile_id
        for tile_id in tile_ids_for_global()
        if bbox_intersects_tile(bbox, list(uu.get_10x10_tile_bounds(tile_id)))
    ]


def resolve_execution_plan(args: argparse.Namespace) -> Dict[str, Any]:
    tiles = _normalize_tile_ids(args.tile_ids)
    bbox = _normalize_bbox(args.bounding_box)
    tile_source = "none"
    tile_ids_to_process: List[str] = []
    if tiles:
        tile_ids_to_process = tiles
        tile_source = "explicit_ids"
    elif bbox is not None:
        tile_ids_to_process = tile_ids_for_bbox(bbox)
        tile_source = "bbox_intersection"
    else:
        tile_ids_to_process = tile_ids_for_global()
        tile_source = "canonical_global_roster"

    if args.execution_mode == "roi":
        resolved_mode = "roi"
    elif args.execution_mode == "tile":
        resolved_mode = "tile"
    else:
        if tiles:
            resolved_mode = "tile"
        elif bbox is None:
            resolved_mode = "tile"
        else:
            resolved_mode = "tile" if len(tile_ids_to_process) >= int(args.auto_tile_threshold_tiles) else "roi"

    roi_bbox: Optional[List[float]] = bbox
    exact_tile_mask_required = False
    if resolved_mode == "roi" and tiles:
        bounds = [uu.get_10x10_tile_bounds(tile_id) for tile_id in tiles]
        roi_bbox = _normalize_bbox(
            [
                min(b[0] for b in bounds),
                min(b[1] for b in bounds),
                max(b[2] for b in bounds),
                max(b[3] for b in bounds),
            ]
        )
        exact_tile_mask_required = True

    return {
        "execution_mode_resolved": resolved_mode,
        "bbox": roi_bbox,
        "tile_ids_to_process": tile_ids_to_process,
        "tile_count": len(tile_ids_to_process),
        "tile_source": tile_source,
        "explicit_tile_ids": tiles,
        "exact_tile_mask_required": exact_tile_mask_required,
        "roi_mode": "tile_ids" if (resolved_mode == "roi" and tiles) else ("bounding_box" if bbox is not None else "global"),
        "is_global_request": (bbox is None and not tiles),
        "base_tile_count": len(tile_ids_to_process),
        "data_tile_filter": "not_resolved",
        "data_tile_filter_dataset": None,
        "data_tile_filter_prefix": None,
        "data_tile_filter_available_count": None,
        "data_tile_filter_dropped_count": 0,
    }

def resolve_interval_execution_plan(
    *,
    base_plan: Dict[str, Any],
    args: argparse.Namespace,
    interval: str,
    fs_s3: s3fs.S3FileSystem,
    logger: logging.Logger,
) -> Dict[str, Any]:
    plan = dict(base_plan)
    plan["tile_ids_to_process"] = list(base_plan["tile_ids_to_process"])
    plan["base_tile_count"] = len(plan["tile_ids_to_process"])
    plan["data_tile_filter"] = args.data_tile_filter
    plan["data_tile_filter_dataset"] = args.data_tile_filter_dataset
    plan["data_tile_filter_prefix"] = None
    plan["data_tile_filter_available_count"] = None
    plan["data_tile_filter_dropped_count"] = 0

    if args.data_tile_filter == "off" or plan["execution_mode_resolved"] != "tile":
        if args.data_tile_filter != "off" and plan["execution_mode_resolved"] != "tile":
            logger.info(
                "Data-tile filtering is only applied in tile execution mode; leaving interval %s in %s mode.",
                interval,
                plan["execution_mode_resolved"],
            )
        return plan

    prefix = build_aggregated_tile_prefix(
        model_version=args.model_version,
        dataset=args.data_tile_filter_dataset,
        run_name=args.run_name,
        interval_type=args.interval_type,
        interval=interval,
        pixel_resolution=args.data_tile_filter_pixel_resolution,
        run_date=args.run_date,
    )
    data_tile_ids = discover_aggregated_data_tile_ids(fs_s3, prefix)
    data_tile_set = set(data_tile_ids)
    filtered_tile_ids = [
        tile_id for tile_id in plan["tile_ids_to_process"] if tile_id in data_tile_set
    ]
    dropped_count = len(plan["tile_ids_to_process"]) - len(filtered_tile_ids)
    if not filtered_tile_ids:
        raise ValueError(
            "Data-tile filter removed every candidate tile for "
            f"interval={interval}, dataset={args.data_tile_filter_dataset}, prefix={prefix}. "
            "Check that aggregation completed for this run/date/interval, or rerun with "
            "--data_tile_filter off."
        )

    plan["tile_ids_to_process"] = filtered_tile_ids
    plan["tile_count"] = len(filtered_tile_ids)
    plan["tile_source"] = (
        f"{base_plan['tile_source']}+aggregated_{args.data_tile_filter_dataset}"
    )
    plan["data_tile_filter_prefix"] = prefix
    plan["data_tile_filter_available_count"] = len(data_tile_ids)
    plan["data_tile_filter_dropped_count"] = dropped_count
    logger.info(
        "Data-tile filter applied: interval=%s dataset=%s prefix=%s "
        "candidate_tiles=%d available_data_tiles=%d selected_tiles=%d dropped_tiles=%d",
        interval,
        args.data_tile_filter_dataset,
        prefix,
        plan["base_tile_count"],
        len(data_tile_ids),
        len(filtered_tile_ids),
        dropped_count,
    )
    return plan

def build_branch_manifest(
    args: argparse.Namespace,
    interval: str,
    branch: str,
    selected_fluxes: List[str],
    roi_meta: Dict[str, Any],
    selected_contextual_groupers: List[str],
) -> Dict[str, Any]:
    selected_flux_type_labels = [
        zc.ZONAL_FLUX_LABELS_BY_KEY[k]
        for k in selected_fluxes
        if k in zc.ZONAL_FLUX_LABELS_BY_KEY
    ]
    return {
        "model_version": args.model_version,
        "run_name": args.run_name,
        "run_date": args.run_date,
        "interval": interval,
        "interval_type": args.interval_type,
        "branch": branch,
        "selected_fluxes": selected_fluxes,
        "selected_flux_type_labels": selected_flux_type_labels,
        "selected_contextual_groupers": selected_contextual_groupers,
        "contextual_grouper_paths": {
            key: OPTIONAL_CONTEXTUAL_GROUPERS[key]["zarr_path"]
            for key in selected_contextual_groupers
        },
        "align_tolerance_fraction": args.align_tolerance_fraction,
        "force_align": bool(args.force_align),
        "roi_mode": roi_meta["roi_mode"],
        "bounding_box": roi_meta["bounding_box"],
        "tile_ids": roi_meta["tile_ids"],
        "processed_tile_ids": None,
        "data_tile_filter": None,
        "data_tile_filter_dataset": None,
        "data_tile_filter_prefix": None,
        "data_tile_filter_available_count": None,
        "data_tile_filter_dropped_count": None,
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
        "selected_contextual_groupers",
        "contextual_grouper_paths",
        "align_tolerance_fraction",
        "force_align",
        "roi_mode",
        "bounding_box",
        "tile_ids",
        "processed_tile_ids",
        "execution_mode",
        "tile_source",
        "tile_count",
        "data_tile_filter",
        "data_tile_filter_dataset",
        "data_tile_filter_prefix",
        "data_tile_filter_available_count",
        "data_tile_filter_dropped_count",
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
    if "combined_state_nodes" in df.columns:
        df["combined_state_nodes"] = df["combined_state_nodes"].astype("uint32", copy=False)
        packed = df["combined_state_nodes"].fillna(0).astype("uint32").to_numpy(copy=False)
        drained_nodes, burned_nodes = zc.unpack_combined_state_to_legacy(packed)
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


def _canonicalize_output_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize output dtypes so parquet shards share a stable schema."""
    canonical_strings = {
        "flux_type",
        "tile_id",
        "drained_state_meaning",
        "burned_state_meaning",
    }
    for col in df.columns:
        if col in canonical_strings or col.endswith("_meaning"):
            df[col] = df[col].astype("string")
    return df


def _table_from_canonical_frame(df: pd.DataFrame) -> pa.Table:
    """Build an Arrow table with explicit string fields to avoid null-typed shards."""
    frame = _canonicalize_output_frame(df.copy())
    fields = []
    for col in frame.columns:
        if pd.api.types.is_string_dtype(frame[col]):
            fields.append(pa.field(col, pa.string()))
        else:
            fields.append(pa.field(col, pa.array(frame[col], from_pandas=True).type))
    return pa.Table.from_pandas(frame, preserve_index=False, schema=pa.schema(fields))


def resolve_combined_state_nodes(
    mega_ds: xr.Dataset,
    ref: xr.DataArray,
    tol: float,
    force_align: bool,
    mega_zarr_path: str,
    interval: str,
    logger: logging.Logger,
) -> xr.DataArray:
    if "combined_state" in mega_ds:
        logger.info("State raster source selected: interval=%s branch=combined_state source=direct:combined_state", interval)
        return prepare_analysis_array(
            {**STATE_DATASETS["combined_state_nodes"], "source_var": "combined_state"},
            "combined_state_nodes",
            mega_ds,
            ref,
            tol,
            force_align,
            mega_zarr_path=mega_zarr_path,
        ).astype("uint32")
    if "emissions_state" in mega_ds:
        logger.info("State raster source selected: interval=%s branch=combined_state source=direct:emissions_state", interval)
        return prepare_analysis_array(
            {**STATE_DATASETS["combined_state_nodes"], "source_var": "emissions_state"},
            "combined_state_nodes",
            mega_ds,
            ref,
            tol,
            force_align,
            mega_zarr_path=mega_zarr_path,
        ).astype("uint32")
    if ("drained_state" in mega_ds) and ("burned_state" in mega_ds):
        logger.info("State raster source selected: interval=%s branch=combined_state source=derived:pack(drained_state,burned_state)", interval)
        drained_for_pack = prepare_analysis_array(
            STATE_DATASETS["drained_state_nodes"],
            "drained_state_nodes",
            mega_ds,
            ref,
            tol,
            force_align,
            mega_zarr_path=mega_zarr_path,
        ).astype("uint32")
        burned_for_pack = prepare_analysis_array(
            STATE_DATASETS["burned_state_nodes"],
            "burned_state_nodes",
            mega_ds,
            ref,
            tol,
            force_align,
            mega_zarr_path=mega_zarr_path,
        ).astype("uint32")
        return xr.apply_ufunc(
            zc.pack_combined_state,
            drained_for_pack,
            burned_for_pack,
            dask="parallelized",
            output_dtypes=[np.uint32],
        ).rename("combined_state_nodes")
    raise ValueError(
        "Unable to resolve combined_state raster. Requires combined_state or emissions_state or drained_state+burned_state."
    )


def run_combined_state_reduce(
    *,
    selected_flux_arrays: List[xr.DataArray],
    selected_flux_keys: List[str],
    combined_nodes_for_reduce: xr.DataArray,
    adm0_aligned: xr.DataArray,
    ref: xr.DataArray,
    expected_groups: List[np.ndarray],
    extra_groupers: tuple[xr.DataArray, ...] = (),
    extra_expected_groups: tuple[np.ndarray, ...] = (),
    where_mask: Optional[xr.DataArray],
    interval_end_year: int,
    no_sparse: bool,
    logger: logging.Logger,
    reduce_label: str,
) -> pd.DataFrame:
    flux_codes = [FLUX_SPECS[k]["code"] for k in selected_flux_keys]
    area_layer = ref if ref.dtype == np.float32 else ref.astype("float32")
    area_layer = drop_scalar_year_coord(area_layer)
    cube_e = xr.concat(selected_flux_arrays + [area_layer], dim="flux_type", coords="minimal").assign_coords(
        flux_type=("flux_type", flux_codes + [2])
    )
    logger.info(
        "Reduction start: label=%s flux_layers=%d cube_dims=%s cube_chunks=%s cube_numblocks=%s",
        reduce_label, len(selected_flux_keys), dict(cube_e.sizes), _chunk_structure(cube_e), getattr(cube_e.data, "numblocks", None),
    )
    all_groupers = [adm0_aligned, combined_nodes_for_reduce, *list(extra_groupers)]
    all_expected_groups = [*expected_groups, *list(extra_expected_groups)]
    logger.info(
        "Reduction groupers: label=%s groupers=%s expected_group_lengths=%s where_mask=%s",
        reduce_label,
        [g.name for g in all_groupers],
        [int(len(gv)) for gv in all_expected_groups],
        where_mask is not None,
    )
    with dask.annotate(label=reduce_label):
        result = xarray_reduce(
            cube_e,
            *all_groupers,
            func="sum",
            expected_groups=tuple(all_expected_groups),
            where=where_mask,
            **flox_sparse_reindex_kwargs(not no_sparse),
        ).compute()
    logger.info("Reduction end: label=%s", reduce_label)
    flux_map = {FLUX_SPECS[k]["code"]: FLUX_SPECS[k]["label"] for k in selected_flux_keys}
    flux_map[2] = "area__ha"
    return _df_from_result(result, flux_map, interval_end_year)


def finalize_interval_tile_outputs(tile_stage_dir: Path) -> pd.DataFrame:
    try:
        table = ds.dataset(str(tile_stage_dir), format="parquet").to_table()
        frame = table.to_pandas()
    except pa.ArrowNotImplementedError as exc:
        if "cast_null" not in str(exc):
            raise
        parquet_paths = sorted(tile_stage_dir.glob("*.parquet"))
        if not parquet_paths:
            return pd.DataFrame()
        frame = pd.concat((pd.read_parquet(path) for path in parquet_paths), ignore_index=True)
    if frame.empty:
        return frame
    drop_cols = {
        "value",
        "tile_id",
        "drained_state_nodes",
        "burned_state_nodes",
        "drained_state_meaning",
        "burned_state_meaning",
    }
    group_cols = [c for c in frame.columns if c not in drop_cols and not c.endswith("_meaning")]
    out = frame.groupby(group_cols, as_index=False, dropna=False)["value"].sum()
    packed = out["combined_state_nodes"].fillna(0).astype("uint32").to_numpy(copy=False)
    drained_nodes, burned_nodes = zc.unpack_combined_state_to_legacy(packed)
    out["drained_state_nodes"] = drained_nodes.astype("uint32", copy=False)
    out["burned_state_nodes"] = burned_nodes.astype("uint32", copy=False)
    out["drained_state_meaning"] = out["drained_state_nodes"].astype("string").str.zfill(8).map(zc.DRAINED_STATE_NODE_MEANINGS)
    out["burned_state_meaning"] = out["burned_state_nodes"].astype("string").str.zfill(8).map(zc.BURNED_STATE_NODE_MEANINGS)
    return out

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
    interval_pairs = resolve_interval_pairs(args.interval_type, args.interval_end_years)
    selected_names = ordered_dataset_keys(args.datasets)

    cluster = client = None
    run_local = bool(args.run_local)
    try:
        cluster, client, run_local = uu.connect_to_cluster(cluster_name=args.cluster_name, run_local=args.run_local)
        logger, _ = lu.populate_main_log_header(
            bounding_box=_normalize_bbox(args.bounding_box), use_shapefile=False, client=client, cluster=cluster,
            log_note="Organic soils zonal statistics (per-pixel; robust alignment)", run_local=run_local,
            model_type="organic_soils", stage=stage,
        )
        if client is not None and not run_local:
            patch_zarr_asyncarray_config_on_workers(client, logger)
        if args.debug:
            logger.setLevel(logging.DEBUG)
        logger.debug("Starting run with args: %s", args)
        logger.info("Diagnostics mode: %s | Force align: %s", args.diagnostics, args.force_align)
        flux_selection = resolve_flux_selection(selected_names, logger=logger)
        selected_fluxes_ordered = flux_selection["selected_fluxes_ordered"]
        selected_contextual_groupers = resolve_requested_contextual_groupers(args.contextual_groupers)
        logger.info(
            "Requested contextual groupers: raw=%s resolved=%s",
            args.contextual_groupers,
            selected_contextual_groupers,
        )
        logger.info(
            "Contextual grouper source paths: %s",
            {
                key: OPTIONAL_CONTEXTUAL_GROUPERS[key]["zarr_path"]
                for key in selected_contextual_groupers
            },
        )
        execution_plan = resolve_execution_plan(args)
        logger.info(
            "Resolved execution mode: requested=%s resolved=%s roi_mode=%s tile_count=%s tile_source=%s bbox=%s",
            args.execution_mode,
            execution_plan["execution_mode_resolved"],
            execution_plan["roi_mode"],
            execution_plan["tile_count"],
            execution_plan["tile_source"],
            execution_plan["bbox"],
        )
        if execution_plan["execution_mode_resolved"] == "roi" and execution_plan["is_global_request"]:
            logger.warning("High-risk full-domain ROI mode detected (global one-shot reduction).")

        gadm_adm0_ids = np.array([i for i in zc.GADM_ADM0_IDS if i > 0], dtype=np.uint32)
        combined_state_codes_arr = zc.COMBINED_STATE_GROUP_VALUES.astype(np.uint32, copy=False)
        local_arrow = pafs.LocalFileSystem()
        base_dir_root = Path(args.local_output).expanduser().resolve()
        base_dir_root.mkdir(parents=True, exist_ok=True)
        base_dir_combined = base_dir_root / "combined_state"
        tile_stage_root = base_dir_root / "_tile_stage"
        fs_s3 = s3fs.S3FileSystem(anon=False)

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
            interval_plan = resolve_interval_execution_plan(
                base_plan=execution_plan,
                args=args,
                interval=interval,
                fs_s3=fs_s3,
                logger=logger,
            )
            roi_meta = normalized_roi_metadata(interval_plan["explicit_tile_ids"], interval_plan["bbox"])
            dest_root = build_output_parquet(args.model_version, args.run_name, args.run_date, interval)
            dest_e = posixpath.join(dest_root.rstrip("/"), "combined_state")
            manifest_e = build_branch_manifest(
                args,
                interval,
                "combined_state",
                selected_fluxes_ordered,
                roi_meta,
                selected_contextual_groupers,
            )
            manifest_e["execution_mode"] = interval_plan["execution_mode_resolved"]
            manifest_e["tile_source"] = interval_plan["tile_source"]
            manifest_e["tile_count"] = interval_plan["tile_count"]
            manifest_e["processed_tile_ids"] = (
                interval_plan["tile_ids_to_process"]
                if interval_plan["execution_mode_resolved"] == "tile"
                else None
            )
            manifest_e["data_tile_filter"] = interval_plan["data_tile_filter"]
            manifest_e["data_tile_filter_dataset"] = interval_plan["data_tile_filter_dataset"]
            manifest_e["data_tile_filter_prefix"] = interval_plan["data_tile_filter_prefix"]
            manifest_e["data_tile_filter_available_count"] = interval_plan["data_tile_filter_available_count"]
            manifest_e["data_tile_filter_dropped_count"] = interval_plan["data_tile_filter_dropped_count"]
            logger.info(
                "Manifest contextual metadata: selected_contextual_groupers=%s contextual_grouper_paths=%s",
                manifest_e.get("selected_contextual_groupers"),
                manifest_e.get("contextual_grouper_paths"),
            )

            emissions_exists = remote_prefix_has_parquet(fs_s3, dest_e)
            if not args.overwrite_existing:
                if emissions_exists:
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
                    continue
            elif emissions_exists:
                logger.info("Overwriting combined_state branch for interval %s by deleting remote prefix %s", interval, dest_e)
                delete_remote_prefix(fs_s3, dest_e)

            logger.info("Processing interval %s : %s", interval, timestr())
            if interval_plan["execution_mode_resolved"] == "roi":
                bbox = interval_plan["bbox"]
                logger.info("Mega-zarr open start: interval=%s year=%s path=%s", interval, interval_end_year, mega_zarr_path)
                mega_ds = open_mega_zarr_region(mega_zarr_path, interval_end_year, bbox, args.chunk_size)
                adm0 = open_zarr_region(adm0_zarr_path(), bbox, args.chunk_size).astype("uint32")
                pixel_area = open_zarr_region(PIXEL_AREA_ZARR, bbox, args.chunk_size).astype("float32")
                logger.info("Pre-flight grid diagnostic: ref_shape=%s mega_xy=%s", tuple(pixel_area.shape), (mega_ds.sizes.get("y"), mega_ds.sizes.get("x")))
                if abs(pixel_area.sizes.get("x", 0) - mega_ds.sizes.get("x", 0)) > 3 or abs(pixel_area.sizes.get("y", 0) - mega_ds.sizes.get("y", 0)) > 3:
                    logger.warning("HIGH-VISIBILITY WARNING: substantial ROI grid mismatch before reduction.")
                ref = maybe_persist_reference(pixel_area, client=client, logger=logger, allow_persist=bool(args.persist_reference), max_chunks=int(args.persist_reference_max_chunks))
                tol = float(args.align_tolerance_fraction) * pixel_step(ref)
                validate_selected_sources(mega_ds, selected_fluxes_ordered, mega_zarr_path=mega_zarr_path, interval=interval, logger=logger)
                flux_arrays = [prepare_analysis_array(FLUX_DATASETS[k], k, mega_ds, ref, tol, args.force_align, mega_zarr_path=mega_zarr_path).astype("float32") for k in selected_fluxes_ordered]
                combined_nodes = resolve_combined_state_nodes(mega_ds, ref, tol, args.force_align, mega_zarr_path, interval, logger)
                adm0_aligned = align_auto(adm0, ref, tol, args.force_align)
                contextual_groupers, contextual_expected_groups = resolve_contextual_groupers_for_extent(
                    requested_keys=selected_contextual_groupers,
                    bbox=bbox,
                    chunk_size=args.chunk_size,
                    ref=ref,
                    tol=tol,
                    force_align=args.force_align,
                    logger=logger,
                )
                where_mask = (adm0_aligned > 0)
                if interval_plan["exact_tile_mask_required"]:
                    where_mask = where_mask & build_exact_tile_mask(ref, interval_plan["explicit_tile_ids"])
                df_e = run_combined_state_reduce(
                    selected_flux_arrays=flux_arrays,
                    selected_flux_keys=selected_fluxes_ordered,
                    combined_nodes_for_reduce=combined_nodes,
                    adm0_aligned=adm0_aligned,
                    ref=ref,
                    expected_groups=[gadm_adm0_ids, combined_state_codes_arr],
                    extra_groupers=tuple(contextual_groupers),
                    extra_expected_groups=tuple(contextual_expected_groups),
                    where_mask=where_mask,
                    interval_end_year=interval_end_year,
                    no_sparse=args.no_sparse,
                    logger=logger,
                    reduce_label=f"reduce:combined_state:{interval}",
                )
            else:
                stage_dir = tile_stage_root / interval
                import shutil
                if stage_dir.exists():
                    shutil.rmtree(stage_dir, ignore_errors=True)
                stage_dir.mkdir(parents=True, exist_ok=True)
                for tile_id in interval_plan["tile_ids_to_process"]:
                    logger.info("Tile start: interval=%s tile_id=%s", interval, tile_id)
                    tile_bbox = list(uu.get_10x10_tile_bounds(tile_id))
                    mega_ds = open_mega_zarr_region(mega_zarr_path, interval_end_year, tile_bbox, args.chunk_size)
                    adm0 = open_zarr_region(adm0_zarr_path(), tile_bbox, args.chunk_size).astype("uint32")
                    pixel_area = open_zarr_region(PIXEL_AREA_ZARR, tile_bbox, args.chunk_size).astype("float32")
                    ref = pixel_area
                    tol = float(args.align_tolerance_fraction) * pixel_step(ref)
                    validate_selected_sources(mega_ds, selected_fluxes_ordered, mega_zarr_path=mega_zarr_path, interval=interval, logger=logger)
                    flux_arrays = [prepare_analysis_array(FLUX_DATASETS[k], k, mega_ds, ref, tol, args.force_align, mega_zarr_path=mega_zarr_path).astype("float32") for k in selected_fluxes_ordered]
                    combined_nodes = resolve_combined_state_nodes(mega_ds, ref, tol, args.force_align, mega_zarr_path, interval, logger)
                    adm0_aligned = align_auto(adm0, ref, tol, args.force_align)
                    contextual_groupers, contextual_expected_groups = resolve_contextual_groupers_for_extent(
                        requested_keys=selected_contextual_groupers,
                        bbox=tile_bbox,
                        chunk_size=args.chunk_size,
                        ref=ref,
                        tol=tol,
                        force_align=args.force_align,
                        logger=logger,
                    )
                    tile_where_mask = (adm0_aligned > 0)
                    if interval_plan["bbox"] is not None:
                        logger.info("Applying bbox clip mask in tile mode: interval=%s tile_id=%s bbox=%s", interval, tile_id, interval_plan["bbox"])
                        tile_where_mask = tile_where_mask & build_bbox_mask(ref, interval_plan["bbox"])
                    df_tile = run_combined_state_reduce(
                        selected_flux_arrays=flux_arrays,
                        selected_flux_keys=selected_fluxes_ordered,
                        combined_nodes_for_reduce=combined_nodes,
                        adm0_aligned=adm0_aligned,
                        ref=ref,
                        expected_groups=[gadm_adm0_ids, combined_state_codes_arr],
                        extra_groupers=tuple(contextual_groupers),
                        extra_expected_groups=tuple(contextual_expected_groups),
                        where_mask=tile_where_mask,
                        interval_end_year=interval_end_year,
                        no_sparse=args.no_sparse,
                        logger=logger,
                        reduce_label=f"reduce:combined_state:{interval}:{tile_id}",
                    )
                    df_tile["tile_id"] = tile_id
                    ds.write_dataset(_table_from_canonical_frame(df_tile), base_dir=str(stage_dir / tile_id), filesystem=local_arrow, format="parquet", existing_data_behavior="overwrite_or_ignore")
                    logger.info("Tile end: interval=%s tile_id=%s", interval, tile_id)
                logger.info("Tile re-aggregation start: interval=%s tile_stage_dir=%s", interval, stage_dir)
                df_e = finalize_interval_tile_outputs(stage_dir)
                logger.info("Tile re-aggregation end: interval=%s", interval)

            import shutil
            local_e = base_dir_combined / interval
            if local_e.exists():
                shutil.rmtree(local_e, ignore_errors=True)
            local_e.mkdir(parents=True, exist_ok=True)
            ds.write_dataset(_table_from_canonical_frame(df_e), base_dir=str(local_e),
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
                if interval_plan["execution_mode_resolved"] == "tile" and not args.keep_tile_stage:
                    shutil.rmtree(tile_stage_root / interval, ignore_errors=True)

        if not args.keep_local and not args.keep_tile_stage:
            import shutil
            shutil.rmtree(base_dir_root, ignore_errors=True)
        elif not args.keep_local and args.keep_tile_stage:
            logger.info("Preserving local tile-stage artifacts because --keep_tile_stage was enabled: %s", tile_stage_root)
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
    parser.add_argument(
        "--local_output",
        default=None,
        help=(
            "Local staging directory for Parquet output. Defaults to "
            "AFOLU_LOCAL_OUTPUT_ROOT/staging/zonal_stats/<model_version>/<run_name>/<run_date>."
        ),
    )
    parser.add_argument("--keep_local", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_sparse", action="store_true", default=not SPARSE_DEFAULT)
    parser.add_argument("--run_name", default="ogh_standard_model")
    parser.add_argument("--datasets", nargs="+", choices=sorted(list(FLUX_DATASETS.keys()) + list(FLUX_DATASET_ALIASES.keys()) + list(STATE_DATASETS.keys())),
                        help="Flux datasets to process (default: all flux datasets). State keys are ignored for compatibility.")
    parser.add_argument(
        "--contextual_groupers",
        nargs="+",
        choices=list(CANONICAL_CONTEXTUAL_GROUPER_ORDER),
        default=[],
        help="Optional contextual grouping axes (default: none). Choices: wdpa kba drivers_of_loss",
    )
    parser.add_argument("--align_tolerance_fraction", type=float, default=0.49,
                        help="Fraction of one pixel for nearest reindex tolerance (default 0.49).")
    parser.add_argument("--leak_warn_threshold", type=float, default=0.002,
                        help="Warn if fraction of flux where adm0==0 exceeds this (default 0.002 = 0.2%%).")
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
    parser.add_argument("--execution_mode", choices=["auto", "roi", "tile"], default="auto")
    parser.add_argument("--auto_tile_threshold_tiles", type=int, default=8)
    parser.add_argument(
        "--data_tile_filter",
        choices=["auto", "off"],
        default="auto",
        help=(
            "In tile execution mode, auto-discover per-interval 10x10 tiles that "
            "have aggregated data and run zonal stats only for those tiles. Use "
            "'off' to process the full candidate tile set."
        ),
    )
    parser.add_argument(
        "--data_tile_filter_dataset",
        default="combined_state",
        help="Aggregated output dataset used to discover tiles with data (default: combined_state).",
    )
    parser.add_argument(
        "--data_tile_filter_pixel_resolution",
        default=f"{cn.full_raster_dims}_pixels",
        help="Aggregated output pixel-resolution folder used by --data_tile_filter (default: 40000_pixels).",
    )
    parser.add_argument("--keep_tile_stage", action="store_true")
    args = parser.parse_args(argv)
    if args.local_output is None:
        args.local_output = default_local_output(args.model_version, args.run_name, args.run_date)
    run(args)

if __name__ == "__main__":
    main()
