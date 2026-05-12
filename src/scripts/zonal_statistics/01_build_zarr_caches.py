# -*- coding: utf-8 -*-
"""Prepare contextual Zarr inputs for organic-soils zonal statistics.

This script still contains legacy model-output cache builders, but those are
disabled by default because zonal statistics now reads flux/state layers
directly from the drainage mega-zarr.

python -m src.scripts.zonal_statistics.01_build_zarr_caches \
  --cluster_name drainage_cluster \
  --chunk_size 8000

python -m src.scripts.zonal_statistics.01_build_zarr_caches \
  --cluster_name drainage_cluster \
  --include_legacy_model_output_caches \
  --model_version 0_9_7 \
  --run_date 20251118 \
  --interval_end_years 2024 \
  --run_name ogh_sensitivity_500m_10 \
  --datasets drained_total burned_total \
  --tile_pixels 40000 \
  --chunk_size 8000

"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import boto3
import dask
import dask.array as da
import fsspec
import numpy as np
import posixpath
import rasterio
import s3fs
import xarray as xr
import zarr
from packaging.version import Version

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu

# ------------------------------ constants ------------------------------
ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
OUTPUT_BASE = "{root}/version_{model_version}"

DATASETS: Dict[str, Dict[str, Any]] = {
    "drained_total": {
        "folder": "drained_total_Mg_CO2e_pixel_yr",
        "zarr": "drained_total_Mg_CO2e_pixel_yr_{interval}.zarr",
        "var": "drained_total",
        "dtype": "float32",
    },
    "drained_co2_onsite": {
        "folder": "drained_co2_Mg_CO2_pixel_yr",
        "zarr": "drained_co2_onsite_Mg_CO2_pixel_yr_{interval}.zarr",
        "var": "drained_co2_onsite",
        "dtype": "float32",
    },
    "drained_co2_offsite": {
        "folder": "drained_co2_offsite_Mg_CO2_pixel_yr",
        "zarr": "drained_co2_offsite_Mg_CO2_pixel_yr_{interval}.zarr",
        "var": "drained_co2_offsite",
        "dtype": "float32",
    },
    "drained_n2o": {
        "folder": "drained_n2o_Mg_CO2e_pixel_yr",
        "zarr": "drained_n2o_Mg_CO2e_pixel_yr_{interval}.zarr",
        "var": "drained_n2o",
        "dtype": "float32",
    },
    "drained_total_co2": {
        "folders": [
            "drained_co2_Mg_CO2_pixel_yr",
            "drained_co2_offsite_Mg_CO2_pixel_yr",
        ],
        "zarr": "drained_total_co2_Mg_CO2_pixel_yr_{interval}.zarr",
        "var": "drained_total_co2",
        "dtype": "float32",
    },
    "drained_total_ch4": {
        "folder": "drained_total_ch4_Mg_CO2e_pixel_yr",
        "zarr": "drained_total_ch4_Mg_CO2e_pixel_yr_{interval}.zarr",
        "var": "drained_total_ch4",
        "dtype": "float32",
    },
    "burned_total": {
        "folder": "burned_total_Mg_CO2e_pixel_yr",
        "zarr": "burned_total_Mg_CO2e_pixel_yr_{interval}.zarr",
        "var": "burned_total",
        "dtype": "float32",
    },
    "burned_total_co2": {
        "folder": "burned_total_co2_Mg_CO2_pixel_yr",
        "zarr": "burned_total_co2_Mg_CO2_pixel_yr_{interval}.zarr",
        "var": "burned_total_co2",
        "dtype": "float32",
    },
    "burned_total_ch4": {
        "folder": "burned_total_ch4_Mg_CO2e_pixel_yr",
        "zarr": "burned_total_ch4_Mg_CO2e_pixel_yr_{interval}.zarr",
        "var": "burned_total_ch4",
        "dtype": "float32",
    },
    "drained_state_nodes": {
        "folder": "drained_state",
        "zarr": "drained_state_node_{interval}.zarr",
        "var": "drained_state_nodes",
        "dtype": "uint32",
    },
    "burned_state_nodes": {
        "folder": "burned_state",
        "zarr": "burned_state_node_{interval}.zarr",
        "var": "burned_state_nodes",
        "dtype": "uint32",
    },
    "combined_state_nodes": {
        "folder": "combined_state",
        "zarr": "combined_state_node_{interval}.zarr",
        "var": "combined_state_nodes",
        "dtype": "uint32",
    },
}
FLUX_DATASET_KEYS: Tuple[str, ...] = (
    "drained_total",
    "drained_co2_onsite",
    "drained_co2_offsite",
    "drained_n2o",
    "drained_total_co2",
    "drained_total_ch4",
    "burned_total",
    "burned_total_co2",
    "burned_total_ch4",
)
DATASET_ALIASES: Dict[str, str] = {
    "drained_co2": "drained_co2_onsite",
}

ZARR_CACHE_PREFIX = OUTPUT_BASE + "/zarr/{run_name}/{run_date}/{interval}/"
FOLDER_TEMPLATE = (
    OUTPUT_BASE
    + "/{folder}/{run_name}/five_year_intervals/{interval}/{tile_pixels}_pixels/{run_date}/"
)


def ordered_dataset_keys(selected: Optional[List[str]]) -> List[str]:
    if not selected:
        return list(DATASETS.keys())
    selected_canonical = {DATASET_ALIASES.get(k, k) for k in selected}
    return [k for k in DATASETS if k in selected_canonical]

# ---- Contextual Zarrs (canonical reference grid = pixel_area) ----
CONTEXTUAL_ZARR_ROOT = (
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/global_contextual_zarrs"
)

# Update when refreshed contextual layers are published.
PIXEL_AREA_DATASET = "pixel_area"
PIXEL_AREA_DATE = "20250925"
PIXEL_AREA_ZARR = posixpath.join(
    CONTEXTUAL_ZARR_ROOT,
    PIXEL_AREA_DATASET,
    PIXEL_AREA_DATE,
    f"global_pixel_area_{PIXEL_AREA_DATE}.zarr",
)

# Authoritative pixel-area GeoTIFF tiles (10×10°, 40000 px per 10°)
PIXEL_AREA_GTIF_FOLDER = (
    "s3://gfw2-data/analyses/umd_area_2013__from_gfw-data-lake/"
    "v1.10/raster/epsg-4326/10/40000/area_m/gdal-geotiff/"
)

ORGANIC_PROBABILITY_DATASET = "ogh_unthresholded_probability"
ORGANIC_PROBABILITY_DATE = "20260508"
ORGANIC_PROBABILITY_VAR_NAME = "organic_probability"
ORGANIC_PROBABILITY_FILENAME_TEMPLATE = "global_ogh_unthresholded_probability_{date}.zarr"
ORGANIC_PROBABILITY_GTIF_FOLDER_TEMPLATE = (
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/"
    "peat_mask/OGH/tiles_unthresholded/{date}/"
)
DEFAULT_MIN_ORGANIC_PROBABILITY_CHUNKS = 1000
DEFAULT_MIN_ORGANIC_PROBABILITY_SIZE_GB = 1.0
DEFAULT_MIN_ORGANIC_PROBABILITY_SOURCE_TIFS = 100
DEFAULT_ORGANIC_PROBABILITY_PROBE_TILES = 3

CLIMATE_DOMAIN_DATASET = "climate_domain"
CLIMATE_DOMAIN_DATE = "20190418"
CLIMATE_DOMAIN_VAR_NAME = "climate_domain"
CLIMATE_DOMAIN_FILENAME_TEMPLATE = "global_climate_domain_{date}.zarr"

ADM0_DATASET = "GADM4_1_adm0_global"
ADM0_DATE = "20250925"  # keep in sync with Step 2
ADM0_FILENAME_TEMPLATE = "global_GADM41_adm0_{date}.zarr"
ADM0_VAR_NAME = "gadm_adm0"
ADM0_GTIF_FOLDER = (
    "s3://gfw2-data/gadm_administrative_boundaries/v4.1/"
    "v4.1.64__from_gfw-data-lake/raster/epsg-4326/10/40000/adm0/gdal-geotiff/"
)

def adm0_zarr_path(date: str = ADM0_DATE) -> str:
    return posixpath.join(
        CONTEXTUAL_ZARR_ROOT,
        ADM0_DATASET,
        date,
        ADM0_FILENAME_TEMPLATE.format(date=date),
    )

def organic_probability_zarr_path(date: str = ORGANIC_PROBABILITY_DATE) -> str:
    return posixpath.join(
        CONTEXTUAL_ZARR_ROOT,
        ORGANIC_PROBABILITY_DATASET,
        date,
        ORGANIC_PROBABILITY_FILENAME_TEMPLATE.format(date=date),
    )

def organic_probability_gtif_folder(date: str = ORGANIC_PROBABILITY_DATE) -> str:
    return ORGANIC_PROBABILITY_GTIF_FOLDER_TEMPLATE.format(date=date)

def climate_domain_zarr_path(date: str = CLIMATE_DOMAIN_DATE) -> str:
    return posixpath.join(
        CONTEXTUAL_ZARR_ROOT,
        CLIMATE_DOMAIN_DATASET,
        date,
        CLIMATE_DOMAIN_FILENAME_TEMPLATE.format(date=date),
    )

# ------------------------------ helpers --------------------------------
def _split_s3(path: str) -> tuple[str, str]:
    if not path.startswith("s3://"):
        raise ValueError(f"Expected s3:// path, got {path}")
    bucket, key = path[5:].split("/", 1)
    return bucket, key

def zarr_exists(zarr_path: str) -> bool:
    """True if a Zarr v2 (.zgroup) or v3 (zarr.json) root exists (checked via boto3)."""
    b, k = _split_s3(zarr_path.rstrip("/"))
    s3 = boto3.client("s3")
    for probe in ("zarr.json", ".zgroup"):
        try:
            s3.head_object(Bucket=b, Key=f"{k}/{probe}")
            return True
        except Exception:
            continue
    return False

def remove_store_recursively(prefix: str) -> None:
    fs = s3fs.S3FileSystem(anon=False)
    fs.rm(prefix.rstrip("/") + "/", recursive=True)

def list_folder_uris(base_uri: str) -> List[str]:
    fs = s3fs.S3FileSystem(anon=False)
    pattern = base_uri.rstrip("/") + "/**/*.tif"
    files = fs.glob(pattern)
    return [(f if f.startswith("s3://") else f"s3://{f}") for f in files]

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

def pixel_step(arr: xr.DataArray) -> float:
    xvals = arr.x.values
    return float(abs(xvals[1] - xvals[0])) if xvals.size >= 2 else 1.0 / 4000.0

def round_xy(arr: xr.DataArray, ndig: int = 12) -> xr.DataArray:
    out = arr
    if "x" in arr.coords:
        out = out.assign_coords(x=np.round(out.x.astype(float), ndig))
    if "y" in arr.coords:
        out = out.assign_coords(y=np.round(out.y.astype(float), ndig))
    return out

def align_like_nearest_tol(arr: xr.DataArray, ref: xr.DataArray, tol_deg: float) -> xr.DataArray:
    arr2 = round_xy(arr)
    ref2 = round_xy(ref)
    return arr2.reindex_like(ref2, method="nearest", tolerance=tol_deg)

def make_xarray_chunks_from_tiffs(
    tiffs: List[str],
    chunk_size: int,
    *,
    mask_and_scale: bool = True,
) -> xr.DataArray:
    if not tiffs:
        raise FileNotFoundError("No GeoTIFFs found for dataset.")
    with dask.annotate(label="open_mfdataset"):
        ds = xr.open_mfdataset(
            tiffs,
            parallel=True,
            chunks={"x": chunk_size, "y": chunk_size},
            mask_and_scale=mask_and_scale,
        ).squeeze()
    da_ = _first_xy_var(ds)
    return da_

def strip_cf_scale_metadata(arr: xr.DataArray) -> xr.DataArray:
    """Remove decode metadata after intentionally reading native raster values."""
    out = arr.copy(deep=False)
    for key in ("_FillValue", "missing_value", "scale_factor", "add_offset"):
        out.attrs.pop(key, None)
        out.encoding.pop(key, None)
    return out

def _dtype_cast(arr: xr.DataArray, dtype_str: str) -> xr.DataArray:
    if dtype_str == "uint32":
        return arr.fillna(0).astype("uint32")
    if dtype_str == "float32":
        return arr.fillna(0).astype("float32")
    return arr

def ensure_consolidated_v2(store_path: str) -> None:
    fs, inner = fsspec.core.url_to_fs(store_path)
    has_zgroup = fs.exists(posixpath.join(inner, ".zgroup"))
    has_v3json = fs.exists(posixpath.join(inner, "zarr.json"))
    if has_zgroup and not has_v3json:
        try:
            zarr.convenience.consolidate_metadata(fs.get_mapper(inner))
        except Exception:
            pass

def _s3_object_stats(zarr_path: str) -> Tuple[int, float]:
    fs = s3fs.S3FileSystem(anon=False)
    b, k = _split_s3(zarr_path.rstrip("/"))
    objs = fs.find(f"{b}/{k}")
    nobj = len(objs)
    size_bytes = 0
    for o in objs:
        try:
            size_bytes += fs.size(o)
        except Exception:
            pass
    size_gb = size_bytes / (1024 ** 3)
    return nobj, size_gb

def _s3_uri_to_vsis3(uri: str) -> str:
    return uri.replace("s3://", "/vsis3/", 1)

def _s3_size_gb(uris: List[str]) -> float:
    fs = s3fs.S3FileSystem(anon=False)
    size_bytes = 0
    for uri in uris:
        key = uri.replace("s3://", "", 1)
        try:
            size_bytes += fs.size(key)
        except Exception:
            pass
    return size_bytes / (1024 ** 3)

def _zarr_data_chunk_stats(zarr_path: str, variable_name: str) -> Tuple[int, float]:
    fs = s3fs.S3FileSystem(anon=False)
    bucket, key = _split_s3(zarr_path.rstrip("/"))
    root = f"{bucket}/{key}"
    objects = fs.find(root)
    v3_prefix = f"/{variable_name}/c/"
    v2_prefix = f"/{variable_name}/"
    chunks = [
        obj for obj in objects
        if (
            v3_prefix in obj
            or (
                v2_prefix in obj
                and not obj.endswith(".zarray")
                and not obj.endswith(".zattrs")
                and not obj.endswith("zarr.json")
            )
        )
    ]
    size_bytes = 0
    for obj in chunks:
        try:
            size_bytes += fs.size(obj)
        except Exception:
            pass
    return len(chunks), size_bytes / (1024 ** 3)

@dataclass
class RasterProbe:
    tile_name: str
    x: float
    y: float
    source_value: int

def _find_nonzero_source_probe(tif_uri: str, grid_steps: int = 6, window_size: int = 256) -> RasterProbe | None:
    """Find one nonzero pixel in a source GeoTIFF without reading the whole tile."""
    tile_name = posixpath.basename(tif_uri)
    with rasterio.open(_s3_uri_to_vsis3(tif_uri)) as src:
        row_starts = np.linspace(0, max(0, src.height - window_size), grid_steps, dtype=int)
        col_starts = np.linspace(0, max(0, src.width - window_size), grid_steps, dtype=int)
        for row in row_starts:
            for col in col_starts:
                win = rasterio.windows.Window(
                    int(col),
                    int(row),
                    min(window_size, src.width - int(col)),
                    min(window_size, src.height - int(row)),
                )
                data = src.read(1, window=win, masked=True)
                filled = np.ma.filled(data, 0)
                nz = np.argwhere(filled > 0)
                if nz.size == 0:
                    continue
                rr, cc = nz[0]
                value = int(filled[rr, cc])
                x, y = src.xy(int(row) + int(rr), int(col) + int(cc))
                return RasterProbe(tile_name=tile_name, x=float(x), y=float(y), source_value=value)
    return None

def _select_source_probe_tifs(tiffs: List[str], n_tiles: int) -> List[str]:
    fs = s3fs.S3FileSystem(anon=False)
    with_sizes = []
    for uri in tiffs:
        try:
            with_sizes.append((fs.size(uri.replace("s3://", "", 1)), uri))
        except Exception:
            pass
    with_sizes.sort(reverse=True)
    return [uri for _, uri in with_sizes[:max(0, n_tiles)]]

def validate_organic_probability_content(
    zarr_path: str,
    *,
    date: str,
    logger: logging.Logger,
    min_data_chunks: int,
    min_size_gb: float,
    min_source_tifs: int,
    probe_tiles: int,
) -> None:
    """Fail fast if the OGH probability zarr looks sparse or diverges from source tiles."""
    source_tifs = list_folder_uris(organic_probability_gtif_folder(date))
    source_size_gb = _s3_size_gb(source_tifs)
    data_chunks, chunk_size_gb = _zarr_data_chunk_stats(zarr_path, ORGANIC_PROBABILITY_VAR_NAME)
    logger.info(
        "flm: OGH probability content check | source_tifs=%d source_size=%.2f GB "
        "| zarr_data_chunks=%d zarr_data_size=%.2f GB",
        len(source_tifs), source_size_gb, data_chunks, chunk_size_gb,
    )

    failures: List[str] = []
    if len(source_tifs) < min_source_tifs:
        failures.append(f"source TIFF count {len(source_tifs)} < minimum {min_source_tifs}")
    if data_chunks < min_data_chunks:
        failures.append(f"zarr data chunk count {data_chunks} < minimum {min_data_chunks}")
    if chunk_size_gb < min_size_gb:
        failures.append(f"zarr data chunk size {chunk_size_gb:.2f} GB < minimum {min_size_gb:.2f} GB")

    if probe_tiles > 0:
        probes: List[RasterProbe] = []
        for tif_uri in _select_source_probe_tifs(source_tifs, probe_tiles * 4):
            probe = _find_nonzero_source_probe(tif_uri)
            if probe is not None:
                probes.append(probe)
            if len(probes) >= probe_tiles:
                break

        if len(probes) < probe_tiles:
            failures.append(f"found only {len(probes)} nonzero source probe tile(s), expected {probe_tiles}")
        else:
            arr = _first_xy_var(xr.open_zarr(zarr_path, consolidated=None, storage_options={"anon": False}))
            tol = pixel_step(arr) * 0.51
            probe_rows = []
            for probe in probes:
                value = arr.sel(x=probe.x, y=probe.y, method="nearest", tolerance=tol)
                if isinstance(value.data, da.Array):
                    value = value.compute()
                zarr_value = int(np.asarray(value).item())
                probe_rows.append((probe.tile_name, probe.source_value, zarr_value))
                if zarr_value <= 0:
                    failures.append(
                        f"zarr value is zero at source nonzero probe "
                        f"{probe.tile_name} ({probe.x:.6f}, {probe.y:.6f}); source={probe.source_value}"
                    )
            logger.info("flm: OGH probability source/zarr probes: %s", probe_rows)

    if failures:
        raise ValueError(
            "OGH probability zarr content validation failed: " + "; ".join(failures)
        )

def validate_zarr(zarr_path: str, ref: xr.DataArray, *, logger: logging.Logger) -> None:
    arr = _first_xy_var(xr.open_zarr(zarr_path, consolidated=None, storage_options={"anon": False}))
    if arr.sizes.get("x") != ref.sizes.get("x") or arr.sizes.get("y") != ref.sizes.get("y"):
        raise ValueError(
            f"Zarr grid mismatch for {zarr_path} "
            f"(got (y:{arr.sizes.get('y')}, x:{arr.sizes.get('x')}), "
            f"expected (y:{ref.sizes.get('y')}, x:{ref.sizes.get('x')}))"
        )
    try:
        cx = arr.data.chunks[1] if isinstance(arr.data, da.Array) else ()
        cy = arr.data.chunks[0] if isinstance(arr.data, da.Array) else ()
    except Exception:
        cx, cy = (), ()
    nobj, size_gb = _s3_object_stats(zarr_path)
    # sample a few blocks
    samples: List[Tuple[Tuple[int, int], float]] = []
    try:
        for i in (0, max(0, arr.sizes["y"] // 2), max(0, arr.sizes["y"] - 128)):
            for j in (0, max(0, arr.sizes["x"] // 2), max(0, arr.sizes["x"] - 128)):
                sli = arr.isel(y=slice(i, min(i + 128, arr.sizes["y"])),
                               x=slice(j, min(j + 128, arr.sizes["x"])))
                data = sli.data
                if isinstance(data, da.Array):
                    data = data.compute()
                samples.append((sli.shape, float(np.nanmax(data)) if data.size else float("nan")))
    except Exception as e:
        logger.warning("Sampling warning for %s: %s", zarr_path, e)
    logger.info(
        "flm: Validated Zarr ✅ %s | shape=(y:%d, x:%d) chunk_x=%s chunk_y=%s | objects=%d size=%.2f GB | samples=%s",
        zarr_path, arr.sizes["y"], arr.sizes["x"], cx, cy, nobj, size_gb, samples
    )

def validate_zarr_basic(zarr_path: str, *, logger: logging.Logger) -> xr.DataArray:
    """Open and log basic info (no reference grid check). Returns the opened array."""
    arr = _first_xy_var(xr.open_zarr(zarr_path, consolidated=None, storage_options={"anon": False}))
    try:
        cx = arr.data.chunks[1] if isinstance(arr.data, da.Array) else ()
        cy = arr.data.chunks[0] if isinstance(arr.data, da.Array) else ()
    except Exception:
        cx, cy = (), ()
    nobj, size_gb = _s3_object_stats(zarr_path)
    logger.info(
        "flm: Contextual Zarr OK %s | shape=(y:%d, x:%d) chunk_x=%s chunk_y=%s | objects=%d size=%.2f GB",
        zarr_path, arr.sizes["y"], arr.sizes["x"], cx, cy, nobj, size_gb
    )
    return arr

# -------------------------- contextual builders --------------------------
def ensure_pixel_area_contextual_zarr(*, chunk_size: int, logger: logging.Logger) -> str:
    """Ensure the global pixel-area contextual Zarr exists (build from GeoTIFFs if missing)."""
    zarr_path = PIXEL_AREA_ZARR
    if zarr_exists(zarr_path):
        try:
            validate_zarr_basic(zarr_path, logger=logger)
            logger.info("flm: Using existing pixel_area contextual Zarr: %s", zarr_path)
            return zarr_path
        except Exception as exc:
            logger.warning("flm: pixel_area Zarr validation failed (%s); rebuilding.", exc)
            try:
                remove_store_recursively(zarr_path)
            except Exception:
                pass

    logger.info("flm: Building pixel_area contextual Zarr from GeoTIFF tiles.")
    tiffs = list_folder_uris(PIXEL_AREA_GTIF_FOLDER)
    if not tiffs:
        raise FileNotFoundError(f"No GeoTIFFs found under {PIXEL_AREA_GTIF_FOLDER} for pixel_area")

    da_in = make_xarray_chunks_from_tiffs(tiffs, chunk_size)
    da_out = _dtype_cast(da_in, "float32").chunk({d: chunk_size for d in ("x", "y") if d in da_in.dims})
    ds_out = xr.Dataset({"pixel_area": da_out})

    try:
        remove_store_recursively(zarr_path)
    except Exception:
        pass

    with dask.annotate(label="to_zarr:pixel_area"):
        ds_out.to_zarr(zarr_path, mode="w")
    if Version(zarr.__version__).major < 3:
        ensure_consolidated_v2(zarr_path)

    validate_zarr_basic(zarr_path, logger=logger)
    logger.info("flm: Built pixel_area contextual Zarr ✅ %s", zarr_path)
    return zarr_path

def ensure_adm0_contextual_zarr(
    *, ref: xr.DataArray, tol: float, chunk_size: int, logger: logging.Logger, date: str = ADM0_DATE
) -> str:
    zarr_path = adm0_zarr_path(date)
    if zarr_exists(zarr_path):
        try:
            validate_zarr(zarr_path, ref, logger=logger)
            logger.info("flm: Using existing adm0 contextual Zarr: %s", zarr_path)
            return zarr_path
        except Exception as exc:
            logger.warning("flm: adm0 Zarr validation failed (%s); rebuilding.", exc)
            try:
                remove_store_recursively(zarr_path)
            except Exception:
                pass

    logger.info("flm: Building adm0 contextual Zarr for date %s from GeoTIFF tiles.", date)
    tiffs = list_folder_uris(ADM0_GTIF_FOLDER)
    if not tiffs:
        raise FileNotFoundError(f"No GeoTIFFs found under {ADM0_GTIF_FOLDER} to build adm0")

    da_in = make_xarray_chunks_from_tiffs(tiffs, chunk_size)
    da_aligned = align_like_nearest_tol(da_in, ref, tol)
    da_uint = da_aligned.fillna(0).astype("uint32").chunk({d: chunk_size for d in ("x", "y") if d in da_aligned.dims})
    ds_out = xr.Dataset({ADM0_VAR_NAME: da_uint})

    try:
        remove_store_recursively(zarr_path)
    except Exception:
        pass

    with dask.annotate(label="to_zarr:adm0"):
        ds_out.to_zarr(zarr_path, mode="w")
    if Version(zarr.__version__).major < 3:
        ensure_consolidated_v2(zarr_path)

    validate_zarr(zarr_path, ref, logger=logger)
    logger.info("flm: Built adm0 contextual Zarr ✅ %s", zarr_path)
    return zarr_path

def ensure_organic_probability_contextual_zarr(
    *,
    ref: xr.DataArray,
    tol: float,
    chunk_size: int,
    logger: logging.Logger,
    date: str = ORGANIC_PROBABILITY_DATE,
    force_rebuild: bool = False,
    min_data_chunks: int = DEFAULT_MIN_ORGANIC_PROBABILITY_CHUNKS,
    min_size_gb: float = DEFAULT_MIN_ORGANIC_PROBABILITY_SIZE_GB,
    min_source_tifs: int = DEFAULT_MIN_ORGANIC_PROBABILITY_SOURCE_TIFS,
    probe_tiles: int = DEFAULT_ORGANIC_PROBABILITY_PROBE_TILES,
) -> str:
    zarr_path = organic_probability_zarr_path(date)
    if force_rebuild and zarr_exists(zarr_path):
        logger.info(
            "flm: Removing existing ogh_unthresholded_probability Zarr because "
            "--rebuild_organic_probability was passed: %s",
            zarr_path,
        )
        remove_store_recursively(zarr_path)

    if zarr_exists(zarr_path):
        try:
            validate_zarr(zarr_path, ref, logger=logger)
            validate_organic_probability_content(
                zarr_path,
                date=date,
                logger=logger,
                min_data_chunks=min_data_chunks,
                min_size_gb=min_size_gb,
                min_source_tifs=min_source_tifs,
                probe_tiles=probe_tiles,
            )
            logger.info("flm: Using existing ogh_unthresholded_probability contextual Zarr: %s", zarr_path)
            return zarr_path
        except Exception as exc:
            logger.warning("flm: ogh_unthresholded_probability Zarr validation failed (%s); rebuilding.", exc)
            try:
                remove_store_recursively(zarr_path)
            except Exception:
                pass

    logger.info("flm: Building ogh_unthresholded_probability contextual Zarr for date %s from GeoTIFF tiles.", date)
    source_folder = organic_probability_gtif_folder(date)
    tiffs = list_folder_uris(source_folder)
    if not tiffs:
        raise FileNotFoundError(
            f"No GeoTIFFs found under {source_folder} to build ogh_unthresholded_probability"
        )

    da_in = strip_cf_scale_metadata(
        make_xarray_chunks_from_tiffs(tiffs, chunk_size, mask_and_scale=False)
    )
    da_aligned = align_like_nearest_tol(da_in, ref, tol)
    da_uint = strip_cf_scale_metadata(da_aligned.fillna(0).astype("uint8")).chunk(
        {d: chunk_size for d in ("x", "y") if d in da_aligned.dims}
    )
    ds_out = xr.Dataset({ORGANIC_PROBABILITY_VAR_NAME: da_uint})

    try:
        remove_store_recursively(zarr_path)
    except Exception:
        pass

    with dask.annotate(label="to_zarr:ogh_unthresholded_probability"):
        ds_out.to_zarr(zarr_path, mode="w")
    if Version(zarr.__version__).major < 3:
        ensure_consolidated_v2(zarr_path)

    validate_zarr(zarr_path, ref, logger=logger)
    validate_organic_probability_content(
        zarr_path,
        date=date,
        logger=logger,
        min_data_chunks=min_data_chunks,
        min_size_gb=min_size_gb,
        min_source_tifs=min_source_tifs,
        probe_tiles=probe_tiles,
    )
    logger.info("flm: Built ogh_unthresholded_probability contextual Zarr ✅ %s", zarr_path)
    return zarr_path


def ensure_climate_domain_contextual_zarr(
    *,
    ref: xr.DataArray,
    tol: float,
    chunk_size: int,
    logger: logging.Logger,
    date: str = CLIMATE_DOMAIN_DATE,
) -> str:
    zarr_path = climate_domain_zarr_path(date)
    if zarr_exists(zarr_path):
        try:
            validate_zarr(zarr_path, ref, logger=logger)
            logger.info("flm: Using existing climate_domain contextual Zarr: %s", zarr_path)
            return zarr_path
        except Exception as exc:
            logger.warning("flm: climate_domain Zarr validation failed (%s); rebuilding.", exc)
            try:
                remove_store_recursively(zarr_path)
            except Exception:
                pass

    logger.info("flm: Building climate_domain contextual Zarr from GeoTIFF tiles.")
    source_folder = cn.dirs["climate_domain"]
    tiffs = list_folder_uris(source_folder)
    if not tiffs:
        raise FileNotFoundError(
            f"No GeoTIFFs found under {source_folder} to build climate_domain"
        )

    da_in = make_xarray_chunks_from_tiffs(tiffs, chunk_size)
    da_aligned = align_like_nearest_tol(da_in, ref, tol)
    da_cast = da_aligned.fillna(0).astype("int16").chunk(
        {d: chunk_size for d in ("x", "y") if d in da_aligned.dims}
    )
    ds_out = xr.Dataset({CLIMATE_DOMAIN_VAR_NAME: da_cast})

    try:
        remove_store_recursively(zarr_path)
    except Exception:
        pass

    with dask.annotate(label="to_zarr:climate_domain"):
        ds_out.to_zarr(zarr_path, mode="w")
    if Version(zarr.__version__).major < 3:
        ensure_consolidated_v2(zarr_path)

    validate_zarr(zarr_path, ref, logger=logger)
    logger.info("flm: Built climate_domain contextual Zarr ✅ %s", zarr_path)
    return zarr_path


# ------------------------------ pipeline --------------------------------
def build_paths(interval: str, *, tile_pixels: int, dataset_names: Optional[List[str]] = None, **kw) -> Dict[str, Dict[str, Any]]:
    zarr_base = ZARR_CACHE_PREFIX.format(interval=interval, **kw)
    kw2 = dict(kw); kw2["tile_pixels"] = tile_pixels
    out: Dict[str, Dict[str, Any]] = {}
    for name in ordered_dataset_keys(dataset_names):
        spec = DATASETS[name]
        path_spec = {
            "zarr": zarr_base + spec["zarr"].format(interval=interval),
            "var": spec["var"],
            "dtype": spec["dtype"],
        }
        if "folders" in spec:
            path_spec["folders"] = [
                FOLDER_TEMPLATE.format(folder=folder, interval=interval, **kw2)
                for folder in spec["folders"]
            ]
        else:
            path_spec["folder"] = FOLDER_TEMPLATE.format(folder=spec["folder"], interval=interval, **kw2)
        out[name] = path_spec
    return out

def run(args: argparse.Namespace) -> None:
    stage = "zarr_build"
    start_ts = uu.timestr()
    cluster, client, run_local = uu.connect_to_cluster(cluster_name=args.cluster_name, run_local=args.run_local)

    logger, _ = lu.populate_main_log_header(
        bounding_box=None, use_shapefile=False, client=client, cluster=cluster,
        log_note="Organic soils Zarr build", run_local=run_local, model_type="organic_soils", stage=stage,
    )
    if args.debug:
        logger.setLevel(logging.DEBUG)

    requested_datasets = ordered_dataset_keys(args.datasets)

    # 0) Ensure canonical reference grid exists, then open it
    ensure_pixel_area_contextual_zarr(chunk_size=args.chunk_size, logger=logger)
    pixel_area_full = open_zarr_region(PIXEL_AREA_ZARR, None, args.chunk_size)
    ref = pixel_area_full
    if args.bounding_box:
        west, south, east, north = [float(x) for x in args.bounding_box]
        ref = ref.sel(x=slice(west, east), y=slice(south, north))

    # 1) Alignment tolerance in degrees (from reference dx)
    dx = pixel_step(pixel_area_full)
    tol = float(args.align_tolerance_fraction) * dx

    # 2) Ensure adm0 contextual grid exists and matches ref
    ensure_adm0_contextual_zarr(ref=pixel_area_full, tol=tol, chunk_size=args.chunk_size, logger=logger)
    ensure_organic_probability_contextual_zarr(
        ref=pixel_area_full,
        tol=tol,
        chunk_size=args.chunk_size,
        logger=logger,
        date=args.organic_probability_date,
        force_rebuild=args.rebuild_organic_probability,
        min_data_chunks=args.min_organic_probability_chunks,
        min_size_gb=args.min_organic_probability_size_gb,
        min_source_tifs=args.min_organic_probability_source_tifs,
        probe_tiles=args.organic_probability_probe_tiles,
    )
    ensure_climate_domain_contextual_zarr(ref=pixel_area_full, tol=tol, chunk_size=args.chunk_size, logger=logger)

    logger.info("flm: Model version: %s", args.model_version)
    if client:
        try:
            mem, n_workers, nth = uu.get_cluster_info(client, cluster)
            logger.info("flm: Number of workers: %s", n_workers)
            logger.info("flm: Memory per worker: %s", mem)
            logger.info("flm: Threads per worker: %s", nth)
        except Exception:
            pass

    if not args.include_legacy_model_output_caches:
        logger.info(
            "flm: Skipping legacy model-output cache materialization. "
            "Contextual zarrs are ready and zonal stats should read mega-zarr directly."
        )
        uu.stage_duration(start_ts, uu.timestr(), stage)
        if client:
            client.close()
        if cluster:
            cluster.close()
        return

    if not requested_datasets:
        logger.info(
            "flm: Legacy cache mode requested, but no datasets were specified. "
            "Skipping model-output cache build. Pass --datasets with one or more flux datasets to build them."
        )
        uu.stage_duration(start_ts, uu.timestr(), stage)
        if client:
            client.close()
        if cluster:
            cluster.close()
        return

    dataset_names = [k for k in requested_datasets if k in FLUX_DATASET_KEYS]
    ignored_non_flux = [k for k in requested_datasets if k not in FLUX_DATASET_KEYS]
    if ignored_non_flux:
        logger.warning(
            "flm: Ignoring non-flux dataset selections in legacy mode: %s. "
            "Only flux zarr caches are built by this script.",
            ignored_non_flux,
        )
    if not dataset_names:
        logger.info(
            "flm: No flux datasets requested; skipping legacy model-output cache build. "
            "Contextual zarrs are ready."
        )
        uu.stage_duration(start_ts, uu.timestr(), stage)
        if client:
            client.close()
        if cluster:
            cluster.close()
        return

    missing_legacy_args: List[str] = []
    if not args.model_version:
        missing_legacy_args.append("--model_version")
    if not args.run_date:
        missing_legacy_args.append("--run_date")
    if not args.interval_end_years:
        missing_legacy_args.append("--interval_end_years")
    if missing_legacy_args:
        raise ValueError(
            "Legacy model-output cache mode requires: "
            + ", ".join(missing_legacy_args)
        )

    OUTPUT_KW = dict(root=ROOT, model_version=args.model_version, run_date=args.run_date, run_name=args.run_name)

    # 3) Build per-variable aligned Zarrs (legacy compatibility mode)
    mapping = {end: (start, end) for start, end in cn.five_year_inventory_periods}
    interval_pairs = [mapping[y] for y in args.interval_end_years if y in mapping]

    for iv_start, iv_end in interval_pairs:
        interval = f"{iv_start}_{iv_end}"
        logger.info("flm: Interval %s", interval)
        paths = build_paths(interval, tile_pixels=args.tile_pixels, dataset_names=dataset_names, **OUTPUT_KW)

        for key in dataset_names:
            spec = paths[key]
            folders = spec.get("folders") or [spec["folder"]]
            folder_label = " + ".join(folders)
            zpath = spec["zarr"]; var = spec["var"]; dtype = spec["dtype"]

            logger.info("flm: Building %s → %s", folder_label, zpath)

            exists = zarr_exists(zpath)
            if exists and args.write_mode == "w-":
                logger.info("flm: Zarr exists and write_mode is 'w-': %s (validate only)", zpath)
                # validate against full reference grid
                validate_zarr(zpath, pixel_area_full, logger=logger)
                continue
            if exists and args.write_mode == "w":
                logger.info("flm: Removing existing store (write_mode='w'): %s", zpath)
                remove_store_recursively(zpath)

            # Open inputs
            da_parts: List[xr.DataArray] = []
            for folder in folders:
                try:
                    tiffs = list_folder_uris(folder)
                except FileNotFoundError:
                    if key == "combined_state_nodes":
                        legacy_folder = folder.replace("/combined_state/", "/emissions_state/")
                        logger.warning(
                            "flm: canonical combined_state cache input missing; using legacy archived folder %s",
                            legacy_folder,
                        )
                        tiffs = list_folder_uris(legacy_folder)
                    else:
                        raise
                if not tiffs:
                    raise FileNotFoundError(f"No GeoTIFFs found under {folder}")
                logger.info("flm:   • %d TIFF(s) found in %s", len(tiffs), folder)

                try:
                    da_part = make_xarray_chunks_from_tiffs(tiffs, args.chunk_size)
                except Exception as e:
                    logger.warning("open_mfdataset failed: %s. Attempting to filter unreadable tiles.", e)
                    valid: List[str] = []
                    for u in tiffs:
                        try:
                            with xr.open_dataset(u):
                                valid.append(u)
                        except Exception:
                            logger.warning("Skipping unreadable TIFF: %s", u)
                    if not valid:
                        raise RuntimeError(f"All tiles failed for {folder}")
                    da_part = make_xarray_chunks_from_tiffs(valid, args.chunk_size)
                da_parts.append(da_part)

            da_in = da_parts[0]
            for da_part in da_parts[1:]:
                da_in = da_in + da_part

            # Align to canonical grid
            da_aligned = align_like_nearest_tol(da_in, ref, tol)
            da_aligned = _dtype_cast(da_aligned, dtype).chunk({d: args.chunk_size for d in ("x", "y") if d in da_aligned.dims})
            ds_out = xr.Dataset({var: da_aligned})

            with dask.annotate(label=f"to_zarr:{key}:{interval}"):
                ds_out.to_zarr(zpath, mode="w")
            if Version(zarr.__version__).major < 3:
                ensure_consolidated_v2(zpath)

            # Validate against full reference
            validate_zarr(zpath, pixel_area_full, logger=logger)

    uu.stage_duration(start_ts, uu.timestr(), stage)
    if client: client.close()
    if cluster: cluster.close()

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Build aligned Zarr caches for organic soils")
    parser.add_argument("--model_version",
                        help="Legacy mode only: model version (required with --include_legacy_model_output_caches).")
    parser.add_argument("--run_date",
                        help="Legacy mode only: run date (required with --include_legacy_model_output_caches).")
    parser.add_argument("--interval_end_years", nargs="+", type=int,
                        help="Legacy mode only: interval end years (required with --include_legacy_model_output_caches).")
    parser.add_argument("--chunk_size", type=int, default=8000)
    parser.add_argument(
        "--organic_probability_date",
        "--probability_date",
        dest="organic_probability_date",
        default=ORGANIC_PROBABILITY_DATE,
        help=(
            "Date tag for OGH unthresholded probability GeoTIFF tiles and "
            "the contextual Zarr output. Default: %(default)s"
        ),
    )
    parser.add_argument("--tile_pixels", type=int, default=40000,
                        help="Input tile size in pixels in the source folder path (4000 or 40000).")
    parser.add_argument("--run_name", default="ogh_standard_model")
    parser.add_argument("--datasets", nargs="+", choices=sorted(list(DATASETS.keys()) + list(DATASET_ALIASES.keys())),
                        help="Legacy mode only: explicit flux datasets to process.")
    parser.add_argument("--write_mode", choices=["w", "w-"], default="w-",
                        help="'w' overwrite, 'w-' skip if exists (validate only).")
    parser.add_argument("--align_tolerance_fraction", type=float, default=0.49,
                        help="Fraction of one pixel as nearest reindex tolerance (default 0.49).")
    parser.add_argument(
        "--rebuild_organic_probability",
        action="store_true",
        help=(
            "Remove and rebuild the OGH probability contextual Zarr even if a "
            "store already exists. Use this when a previous build may have "
            "partially written the store."
        ),
    )
    parser.add_argument(
        "--min_organic_probability_chunks",
        type=int,
        default=DEFAULT_MIN_ORGANIC_PROBABILITY_CHUNKS,
        help=(
            "Minimum OGH probability data chunks required after build/validation. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--min_organic_probability_size_gb",
        type=float,
        default=DEFAULT_MIN_ORGANIC_PROBABILITY_SIZE_GB,
        help=(
            "Minimum OGH probability data-chunk size in GB required after "
            "build/validation. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--min_organic_probability_source_tifs",
        type=int,
        default=DEFAULT_MIN_ORGANIC_PROBABILITY_SOURCE_TIFS,
        help=(
            "Minimum processed source GeoTIFF count expected for OGH probability. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--organic_probability_probe_tiles",
        type=int,
        default=DEFAULT_ORGANIC_PROBABILITY_PROBE_TILES,
        help=(
            "Number of large source GeoTIFFs to probe and compare against the "
            "built zarr. Set to 0 to disable source/zarr probe comparison. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--include_legacy_model_output_caches",
        action="store_true",
        help="Also build per-interval model-output zarr caches from GeoTIFF folders (legacy mode).",
    )
    parser.add_argument("--debug", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run_local", action="store_true")
    mode.add_argument("--cluster_name", default="zarr_build")
    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W S E N (optional, dev only)")
    args = parser.parse_args(argv)
    if args.include_legacy_model_output_caches and args.datasets:
        requested_fluxes = [k for k in args.datasets if k in FLUX_DATASET_KEYS]
        missing = []
        if requested_fluxes and not args.model_version:
            missing.append("--model_version")
        if requested_fluxes and not args.run_date:
            missing.append("--run_date")
        if requested_fluxes and not args.interval_end_years:
            missing.append("--interval_end_years")
        if missing:
            parser.error(
                "Legacy flux cache mode requires: " + ", ".join(missing)
            )
    run(args)

if __name__ == "__main__":
    main()
