# -*- coding: utf-8 -*-
"""
Run organic-soils zonal statistics directly from GeoTIFF/COG (VRT + windowed streaming).

This script replaces Zarr reads with a COG-native workflow that:
- Builds in-memory GDAL VRTs for each layer (adm0, pixel_area, drained/burned totals & state nodes).
- Warps all inputs to a single explicit canonical grid (CRS=EPSG:4326) via rasterio.WarpedVRT.
- Streams over fixed windows and aggregates with fast 2D bincount histograms.
- Writes per-interval Parquet locally, then uploads immediately to S3.

Key properties:
- **Exact alignment**: no floating coordinate matching or "0.5-pixel tolerance".
- **No Zarr**: avoids metadata/consolidation/version fragility.
- **Streaming**: bounded memory; robust on large ROIs.

Outputs:
- S3 layout mirrors your prior `zonal_stats/{run_name}/{run_date}/{interval}/(drained|burned)/`.
- Columns: for drained → [gadm_adm0, drained_state_nodes, drained_state_meaning, flux_type, value, interval_end]
           for burned →  [gadm_adm0, burned_state_nodes,  burned_state_meaning,  flux_type, value, interval_end]

Notes:
- The script reuses your folder structure and auto-detects per-ha vs per-pixel flux inputs.
- Environment variables such as `GDAL_HTTP_MULTIRANGE=YES` and `CPL_VSIL_CURL_CHUNK_SIZE=16777216` help COG I/O.

"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import rasterio
import s3fs
from affine import Affine
from osgeo import gdal
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

# Project utilities (absolute imports to allow `python -m ...`)
import src.scripts.zonal_statistics.zonal_constants as zc
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities.universal_utilities import timestr
from src.scripts.utilities import log_utilities as lu


# ------------------------------- config --------------------------------

ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
OUTPUT_BASE = "{root}/version_{model_version}"

# Input folders (depend on tile size to locate inputs)
FOLDER_TEMPLATE = (
    OUTPUT_BASE
    + "/{folder}/{run_name}/five_year_intervals/{interval}/{tile_pixels}_pixels/{run_date}/"
)

# Contextual layers (static GeoTIFF folders; we build VRTs at runtime)
ADM0_GTIFF_FOLDER = (
    "s3://gfw2-data/gadm_administrative_boundaries/v4.1/"
    "v4.1.64__from_gfw-data-lake/raster/epsg-4326/10/40000/adm0/gdal-geotiff/"
)
PIXEL_AREA_GTIFF_FOLDER = (
    "s3://gfw2-data/analyses/umd_area_2013__from_gfw-data-lake/"
    "v1.10/raster/epsg-4326/10/40000/area_m/gdal-geotiff/"
)

# Dataset manifest (same labels you use downstream)
DATASETS: Dict[str, Dict[str, Any]] = {
    "drained_state_nodes": {
        "folder": "drained_state",
        "var": "drained_state_nodes",
    },
    "burned_state_nodes": {
        "folder": "burned_state",
        "var": "burned_state_nodes",
    },
    "drained_total": {
        "folder_candidates": [
            "drained_total_Mg_CO2e_pixel_yr",
            "drained_total_Mg_CO2e_ha_yr",
        ],
        "var": "drained_total",
    },
    "burned_total": {
        "folder_candidates": [
            "burned_total_Mg_CO2e_pixel_yr",
            "burned_total_Mg_CO2e_ha_yr",
        ],
        "var": "burned_total",
    },
}


# ------------------------------- helpers --------------------------------

def build_output_parquet(model_version: str, run_name: str, run_date: str, interval: str) -> str:
    """Return the S3 prefix for zonal stats output (mirrors prior conventions)."""
    base = "/".join(
        [ROOT.rstrip("/"), f"version_{model_version}", "zonal_stats", run_name, run_date, interval]
    )
    return base.rstrip("/") + "/"


def s3_exists(prefix: str) -> bool:
    fs = s3fs.S3FileSystem(anon=False)
    try:
        return any(fs.glob(prefix.rstrip("/") + "/**"))
    except FileNotFoundError:
        return False


def list_folder_uris(base_uri: str) -> List[str]:
    """Return list of GeoTIFF URIs within base_uri recursively; error if none."""
    fs = s3fs.S3FileSystem(anon=False)
    pattern = base_uri.rstrip("/") + "/**/*.tif"
    tif_files = [(f if f.startswith("s3://") else f"s3://{f}") for f in fs.glob(pattern)]
    if not tif_files:
        raise FileNotFoundError(f"No GeoTIFFs found in {base_uri}")
    return sorted(tif_files)


def _filter_valid_tiffs(uris: List[str], label: str, logger: logging.Logger) -> List[str]:
    """Keep only TIFFs that rasterio can open; log and skip unreadable tiles."""
    valid: List[str] = []
    bad = 0
    for u in uris:
        try:
            with rasterio.Env():
                with rasterio.open(u) as src:
                    if src.count >= 1 and src.width > 0 and src.height > 0:
                        valid.append(u)
                    else:
                        bad += 1
        except Exception as e:
            logger.warning("Skipping unreadable TIFF for %s: %s (%s)", label, u, e)
            bad += 1
    if bad:
        logger.warning("Filtered out %d invalid TIFF(s) for %s; using %d valid.", bad, label, len(valid))
    if not valid:
        raise FileNotFoundError(f"No valid GeoTIFFs remain for {label}")
    return valid


def _resolve_flux_folder_and_unit(folder_candidates: List[str], interval: str, **fmt_kw) -> Tuple[str, str]:
    """Return (existing_folder_uri, unit) for flux; prefer per-ha, else per-pixel."""
    ranked: List[Tuple[str, str]] = []
    for name in folder_candidates:
        uri = FOLDER_TEMPLATE.format(folder=name, interval=interval, **fmt_kw)
        unit = "ha" if "_ha_" in name else "pixel"
        ranked.append((uri, unit))
    # Prefer per-ha
    ranked.sort(key=lambda x: 0 if x[1] == "ha" else 1)
    for uri, unit in ranked:
        if s3_exists(uri):
            return uri, unit
    raise FileNotFoundError(f"No matching flux folders for interval {interval}: {folder_candidates}")


def build_paths(interval: str, *, tile_pixels: int, **kw) -> Dict[str, Dict[str, Any]]:
    """Return input folder paths (and unit for flux) for all datasets in *interval*."""
    paths: Dict[str, Dict[str, Any]] = {}
    kw2 = dict(kw)
    kw2["tile_pixels"] = tile_pixels

    for name, spec in DATASETS.items():
        if "folder_candidates" in spec:
            folder_uri, unit = _resolve_flux_folder_and_unit(spec["folder_candidates"], interval, **kw2)
            paths[name] = {"folder": folder_uri, "unit": unit, "var": spec["var"]}
        else:
            folder_uri = FOLDER_TEMPLATE.format(folder=spec["folder"], interval=interval, **kw2)
            paths[name] = {"folder": folder_uri, "unit": None, "var": spec["var"]}
    return paths


@dataclass
class GridSpec:
    crs: str
    transform: Affine
    width: int
    height: int
    bounds: Tuple[float, float, float, float]  # (W, S, E, N)


def _compute_pixel_size_deg(tile_pixels: int) -> float:
    """Return pixel size in degrees, consistent with 1×1° @ 4000px and 10×10° @ 40000px."""
    deg_per_tile = 10.0 if tile_pixels >= 40000 else 1.0
    return deg_per_tile / float(tile_pixels)


def make_grid(bbox: List[float], pixel_size_deg: float) -> GridSpec:
    """Create a canonical north-up grid for bbox, snapped to pixel boundaries deterministically."""
    west, south, east, north = bbox
    px = float(pixel_size_deg)
    w = math.floor(west / px) * px
    n = math.ceil(north / px) * px
    width = int(round((east - w) / px))
    height = int(round((n - south) / px))
    transform = Affine.translation(w, n) * Affine.scale(px, -px)  # from_origin(w, n, px, px)
    return GridSpec("EPSG:4326", transform, width, height, (w, south, east, n))


def build_vrt_from_uris(uris: List[str], vrt_name: str, logger: logging.Logger) -> str:
    """Build an in-memory GDAL VRT from a list of (COG) URIs; return the /vsimem/ path."""
    vrt_path = f"/vsimem/{vrt_name}.vrt"
    try:
        gdal.Unlink(vrt_path)
    except Exception:
        pass
    # Options: use highest resolution, no alpha; nodata taken from sources
    opts = gdal.BuildVRTOptions(resolution="highest", addAlpha=False)
    ds = gdal.BuildVRT(vrt_path, uris, options=opts)
    if ds is None:
        raise RuntimeError(f"gdal.BuildVRT failed for {vrt_name}")
    ds = None  # close
    logger.debug("Built VRT: %s (%d sources)", vrt_path, len(uris))
    return vrt_path


def open_aligned_reader(vrt_path: str, grid: GridSpec, resampling: Resampling) -> Tuple[WarpedVRT, rasterio.io.DatasetReader]:
    """Open a VRT and return a WarpedVRT aligned exactly to *grid* (plus base dataset for cleanup)."""
    base = rasterio.open(vrt_path)
    vrt = WarpedVRT(
        base,
        crs=grid.crs,
        transform=grid.transform,
        width=grid.width,
        height=grid.height,
        resampling=resampling,
        src_nodata=base.nodata,
    )
    return vrt, base


def iter_windows(width: int, height: int, block: int = 1024) -> List[Window]:
    """Yield deterministic windows over the canonical grid."""
    for row_off in range(0, height, block):
        h = min(block, height - row_off)
        for col_off in range(0, width, block):
            w = min(block, width - col_off)
            yield Window(col_off, row_off, w, h)


# -------------------------- aggregation kernel --------------------------

def _build_luts() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare LUTs for category → index mapping."""
    gadm_ids = zc.GADM_ADM0_IDS.astype("uint16")  # NG
    drained_codes = np.array(sorted({0, *map(int, zc.ALL_DRAINED_STATE_CODES)}), dtype=np.uint32)  # ND
    burned_codes = np.array(sorted({0, *map(int, zc.ALL_BURNED_STATE_CODES)}), dtype=np.uint32)   # NB

    # GADM lookup: direct array index (O(1))
    g_max = int(gadm_ids.max()) if gadm_ids.size else 0
    g_lut = np.full(g_max + 1, -1, dtype=np.int32)
    g_lut[gadm_ids.astype(np.int64)] = np.arange(gadm_ids.size, dtype=np.int32)
    return gadm_ids, drained_codes, burned_codes, g_lut


def _map_to_index(values_uint32: np.ndarray, code_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map raw codes → sorted index space; returns (idx, ok_mask)."""
    idx = np.searchsorted(code_vec, values_uint32)
    ok = (idx < code_vec.size) & (code_vec[idx] == values_uint32)
    out = np.full(values_uint32.size, -1, dtype=np.int64)
    out[ok] = idx[ok]
    return out, ok


def _bincount_2d_sum(g_idx: np.ndarray, s_idx: np.ndarray, weight: np.ndarray, nG: int, nS: int) -> np.ndarray:
    """Compute sum(weight) by (g_idx, s_idx) in flattened O(N) time."""
    valid = (g_idx >= 0) & (s_idx >= 0) & np.isfinite(weight)
    if not valid.any():
        return np.zeros((nG, nS), dtype=np.float64)
    key = (g_idx[valid].astype(np.int64) * nS) + s_idx[valid].astype(np.int64)
    flat = np.bincount(key, weights=weight[valid].astype(np.float64), minlength=nG * nS)
    return flat.reshape(nG, nS)


def _share_to_gadm0(mat: np.ndarray, g_lut: np.ndarray, logger: logging.Logger, label: str) -> None:
    """QA log: percent of sum going to GADM=0."""
    total = float(mat.sum())
    ocean = float(mat[g_lut[0], :].sum()) if g_lut.size > 0 and g_lut[0] >= 0 else 0.0
    pct = (ocean / total * 100.0) if total else 0.0
    logger.info("QA: share to GADM=0 for %s = %.4f%%", label, pct)
    if pct > 0.5:
        logger.warning("QA: High share to GADM=0 for %s (%.2f%%). Check alignment.", label, pct)


def aggregate_interval(
    *,
    interval: str,
    interval_end_year: int,
    grid: GridSpec,
    paths: Dict[str, Dict[str, Any]],
    logger: logging.Logger,
    window_block: int = 1024,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run drained & burned zonal statistics for *interval* on the canonical *grid*."""

    # --- 1) Build VRTs for each layer ---
    vrt_paths: Dict[str, str] = {}
    temp_vrts: List[str] = []

    # Contextual
    adm0_uris = _filter_valid_tiffs(list_folder_uris(ADM0_GTIFF_FOLDER), "adm0", logger)
    pix_uris = _filter_valid_tiffs(list_folder_uris(PIXEL_AREA_GTIFF_FOLDER), "pixel_area", logger)
    vrt_paths["adm0"] = build_vrt_from_uris(adm0_uris, f"adm0_{interval}", logger); temp_vrts.append(vrt_paths["adm0"])
    vrt_paths["pixel_area"] = build_vrt_from_uris(pix_uris, f"pixel_area_{interval}", logger); temp_vrts.append(vrt_paths["pixel_area"])

    # Interval inputs
    # State nodes
    for key in ("drained_state_nodes", "burned_state_nodes"):
        uris = _filter_valid_tiffs(list_folder_uris(paths[key]["folder"]), key, logger)
        vrt_paths[key] = build_vrt_from_uris(uris, f"{key}_{interval}", logger); temp_vrts.append(vrt_paths[key])

    # Totals
    for key in ("drained_total", "burned_total"):
        uris = _filter_valid_tiffs(list_folder_uris(paths[key]["folder"]), key, logger)
        vrt_paths[key] = build_vrt_from_uris(uris, f"{key}_{interval}", logger); temp_vrts.append(vrt_paths[key])

    # --- 2) Open aligned readers (exact same grid) ---
    adm0_r, adm0_base = open_aligned_reader(vrt_paths["adm0"], grid, Resampling.nearest)
    pix_r,  pix_base  = open_aligned_reader(vrt_paths["pixel_area"], grid, Resampling.nearest)
    d_tot_r, d_tot_base = open_aligned_reader(vrt_paths["drained_total"], grid, Resampling.nearest)
    b_tot_r, b_tot_base = open_aligned_reader(vrt_paths["burned_total"],  grid, Resampling.nearest)
    d_st_r,  d_st_base  = open_aligned_reader(vrt_paths["drained_state_nodes"], grid, Resampling.nearest)
    b_st_r,  b_st_base  = open_aligned_reader(vrt_paths["burned_state_nodes"],  grid, Resampling.nearest)

    # Units: convert per-ha flux inputs to per-pixel totals using pixel_area
    per_ha = {
        "drained_total": (paths["drained_total"]["unit"] == "ha"),
        "burned_total":  (paths["burned_total"]["unit"] == "ha"),
    }
    logger.info(
        "Interval %s units: drained_total=%s, burned_total=%s",
        interval, "ha" if per_ha["drained_total"] else "pixel", "ha" if per_ha["burned_total"] else "pixel"
    )

    # --- 3) Prepare LUTs and accumulators ---
    gadm_ids, drained_codes, burned_codes, g_lut = _build_luts()
    NG, ND, NB = gadm_ids.size, drained_codes.size, burned_codes.size

    agg_d_total   = np.zeros((NG, ND), dtype=np.float64)
    agg_d_area_ha = np.zeros((NG, ND), dtype=np.float64)
    agg_b_total   = np.zeros((NG, NB), dtype=np.float64)
    agg_b_area_ha = np.zeros((NG, NB), dtype=np.float64)

    # --- 4) Stream windows across the canonical grid ---
    logger.info("Aggregating %s across %dx%d grid in %dx%d windows", interval, grid.width, grid.height, window_block, window_block)
    for win in iter_windows(grid.width, grid.height, block=window_block):
        # Read as masked arrays; fill with 0 for counting/summing in bins
        gadm = adm0_r.read(1, window=win, masked=True).filled(0).astype(np.uint16)
        px_m2 = pix_r.read(1, window=win, masked=True).filled(0).astype(np.float64)

        d_tot = d_tot_r.read(1, window=win, masked=True).filled(0).astype(np.float64)
        b_tot = b_tot_r.read(1, window=win, masked=True).filled(0).astype(np.float64)

        d_st  = d_st_r.read(1, window=win, masked=True).filled(0).astype(np.uint32)
        b_st  = b_st_r.read(1, window=win, masked=True).filled(0).astype(np.uint32)

        if per_ha["drained_total"]:
            d_tot *= (px_m2 / 10000.0)
        if per_ha["burned_total"]:
            b_tot *= (px_m2 / 10000.0)

        # Flatten
        gadm_flat = gadm.ravel()
        d_tot_flat = d_tot.ravel()
        b_tot_flat = b_tot.ravel()
        d_st_flat = d_st.ravel()
        b_st_flat = b_st.ravel()
        px_ha_flat = (px_m2.ravel() / 10000.0)

        # Map to category indices
        g_idx = g_lut[gadm_flat.astype(np.int64)].astype(np.int64)
        d_idx, d_ok = _map_to_index(d_st_flat, drained_codes)
        b_idx, b_ok = _map_to_index(b_st_flat, burned_codes)

        # Totals
        agg_d_total += _bincount_2d_sum(g_idx, d_idx, d_tot_flat, NG, ND)
        agg_b_total += _bincount_2d_sum(g_idx, b_idx, b_tot_flat, NG, NB)

        # Areas only for valid class pixels
        area_d = np.zeros_like(px_ha_flat)
        area_b = np.zeros_like(px_ha_flat)
        if d_ok.any():
            area_d[d_ok] = px_ha_flat[d_ok]
        if b_ok.any():
            area_b[b_ok] = px_ha_flat[b_ok]
        agg_d_area_ha += _bincount_2d_sum(g_idx, d_idx, area_d, NG, ND)
        agg_b_area_ha += _bincount_2d_sum(g_idx, b_idx, area_b, NG, NB)

    # --- 5) QA: share to GADM=0 ---
    _share_to_gadm0(agg_d_total, g_lut, logger, f"drained_total {interval}")
    _share_to_gadm0(agg_b_total, g_lut, logger, f"burned_total {interval}")

    # --- 6) Build DataFrames matching your schema ---
    rows_d: List[Tuple[float, int, str, float]] = []
    for gi, gval in enumerate(gadm_ids.tolist()):
        for si, sval in enumerate(drained_codes.tolist()):
            rows_d.append((float(gval), int(sval), "drained_total_Mg_CO2e", float(agg_d_total[gi, si])))
            rows_d.append((float(gval), int(sval), "area__ha",              float(agg_d_area_ha[gi, si])))
    df_d = pd.DataFrame(rows_d, columns=["gadm_adm0", "drained_state_nodes", "flux_type", "value"])
    df_d["drained_state_meaning"] = (
        df_d["drained_state_nodes"].astype("string").str.zfill(8).map(zc.DRAINED_STATE_NODE_MEANINGS)
    )
    df_d["interval_end"] = interval_end_year

    rows_b: List[Tuple[float, int, str, float]] = []
    for gi, gval in enumerate(gadm_ids.tolist()):
        for si, sval in enumerate(burned_codes.tolist()):
            rows_b.append((float(gval), int(sval), "burned_total_Mg_CO2e", float(agg_b_total[gi, si])))
            rows_b.append((float(gval), int(sval), "area__ha",             float(agg_b_area_ha[gi, si])))
    df_b = pd.DataFrame(rows_b, columns=["gadm_adm0", "burned_state_nodes", "flux_type", "value"])
    df_b["burned_state_meaning"] = (
        df_b["burned_state_nodes"].astype("string").str.zfill(8).map(zc.BURNED_STATE_NODE_MEANINGS)
    )
    df_b["interval_end"] = interval_end_year

    # --- 7) Cleanup VRTs/readers ---
    for ds in (adm0_r, pix_r, d_tot_r, b_tot_r, d_st_r, b_st_r):
        try:
            ds.close()
        except Exception:
            pass
    for ds in (adm0_base, pix_base, d_tot_base, b_tot_base, d_st_base, b_st_base):
        try:
            ds.close()
        except Exception:
            pass
    for p in temp_vrts:
        try:
            gdal.Unlink(p)
        except Exception:
            pass

    return df_d, df_b


# ------------------------------- driver ---------------------------------

def build_interval_pairs(end_years: List[int]) -> List[Tuple[int, int]]:
    mapping = {end: (start, end) for start, end in cn.five_year_inventory_periods}
    pairs = []
    for year in end_years:
        if year not in mapping:
            raise ValueError(
                f"Interval end year {year} not supported. Valid options: {sorted(mapping)}"
            )
        pairs.append(mapping[year])
    return pairs


def run(args: argparse.Namespace) -> None:
    stage = "zonal_statistics_tiff"
    start_ts = uu.timestr()
    cluster, client, run_local = uu.connect_to_cluster(
        cluster_name=args.cluster_name,
        run_local=args.run_local,
    )

    # ROI as bbox or union of tiles
    bbox = None
    if args.bounding_box:
        bbox = [float(x) for x in args.bounding_box]
    elif args.tile_ids:
        tiles: List[str] = []
        for item in args.tile_ids:
            tiles.extend(t.strip() for t in item.split(",") if t.strip())
        if tiles:
            bounds = [uu.get_10x10_tile_bounds(t) for t in tiles]
            west = min(b[0] for b in bounds)
            south = min(b[1] for b in bounds)
            east = max(b[2] for b in bounds)
            north = max(b[3] for b in bounds)
            bbox = [west, south, east, north]

    logger, _ = lu.populate_main_log_header(
        bounding_box=bbox,
        use_shapefile=False,
        client=client,
        cluster=cluster,
        log_note="Organic soils zonal statistics (GeoTIFF/COG)",
        run_local=run_local,
        model_type="organic_soils",
        stage=stage,
    )
    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.debug("Starting run with args: %s", args)
    logger.info("Connected to cluster %s", cluster.name if cluster else "local-threaded")

    if not bbox:
        raise ValueError("A bounding box or tile_ids is required.")

    # Canonical grid for this ROI
    px_deg = _compute_pixel_size_deg(args.tile_pixels)
    grid = make_grid(bbox, px_deg)
    logger.info(
        "Canonical grid: CRS=%s, px=%.10f°, size=%dx%d, bounds=(%.6f, %.6f, %.6f, %.6f)",
        grid.crs, px_deg, grid.width, grid.height, *grid.bounds
    )

    # Placeholders for path formatting
    OUTPUT_KW = dict(root=ROOT, model_version=args.model_version, run_date=args.run_date, run_name=args.run_name)

    # Local staging (always write locally first), no Hive partitioning
    local_arrow = pafs.LocalFileSystem()
    base_dir_root = Path(args.local_output).expanduser().resolve()
    base_dir_root.mkdir(parents=True, exist_ok=True)
    base_dir_drained = base_dir_root / "drained"
    base_dir_burned = base_dir_root / "burned"

    fs_s3 = s3fs.S3FileSystem(anon=False)

    # Process each interval
    interval_pairs = build_interval_pairs(args.interval_end_years)
    for interval_start_year, interval_end_year in interval_pairs:
        interval = f"{interval_start_year}_{interval_end_year}"
        logger.info("Processing interval %s : %s", interval, timestr())

        # Build dataset folders & units with the requested tile size
        paths = build_paths(interval, tile_pixels=args.tile_pixels, **OUTPUT_KW)
        logger.info(
            "Interval %s inputs: drained_total=%s (unit=%s), burned_total=%s (unit=%s), drained_state=%s, burned_state=%s",
            interval,
            paths["drained_total"]["folder"], paths["drained_total"]["unit"],
            paths["burned_total"]["folder"], paths["burned_total"]["unit"],
            paths["drained_state_nodes"]["folder"], paths["burned_state_nodes"]["folder"],
        )

        # Aggregate (windowed on VRT)
        df_d, df_b = aggregate_interval(
            interval=interval,
            interval_end_year=interval_end_year,
            grid=grid,
            paths=paths,
            logger=logger,
            window_block=args.chunk_size,  # reuse CLI name; acts as read block/window
        )

        # -------- Local per-interval staging (no Hive) --------
        local_d = (base_dir_drained / interval)
        local_b = (base_dir_burned / interval)
        for pth in (local_d, local_b):
            if pth.exists():
                import shutil
                shutil.rmtree(pth, ignore_errors=True)
            pth.mkdir(parents=True, exist_ok=True)

        ds.write_dataset(
            pa.Table.from_pandas(df_d, preserve_index=False),
            base_dir=str(local_d),
            filesystem=local_arrow,
            format="parquet",
            existing_data_behavior="overwrite_or_ignore",
        )
        logger.info("Wrote %s rows (drained) for %s → %s", len(df_d), interval, local_d)

        ds.write_dataset(
            pa.Table.from_pandas(df_b, preserve_index=False),
            base_dir=str(local_b),
            filesystem=local_arrow,
            format="parquet",
            existing_data_behavior="overwrite_or_ignore",
        )
        logger.info("Wrote %s rows (burned) for %s → %s", len(df_b), interval, local_b)

        # -------- Upload to S3 under full-interval directory --------
        dest_root = build_output_parquet(args.model_version, args.run_name, args.run_date, interval)
        dest_d = "/".join([dest_root.rstrip("/"), "drained"])
        dest_b = "/".join([dest_root.rstrip("/"), "burned"])
        logger.info("Uploading interval %s → %s{drained,burned}", interval, dest_root)

        # Clear any previous content to avoid mixing runs
        try:
            fs_s3.rm(dest_d.rstrip("/") + "/", recursive=True)
        except Exception:
            pass
        try:
            fs_s3.rm(dest_b.rstrip("/") + "/", recursive=True)
        except Exception:
            pass

        # Upload
        uploaded = 0
        for p in local_d.rglob("*.parquet"):
            rel = p.relative_to(local_d)
            remote_path = "/".join([dest_d.rstrip("/"), *rel.parts])
            fs_s3.put(str(p), remote_path); uploaded += 1
        logger.info("Drained upload: %d file(s) at %s/*.parquet", uploaded, dest_d)

        uploaded = 0
        for p in local_b.rglob("*.parquet"):
            rel = p.relative_to(local_b)
            remote_path = "/".join([dest_b.rstrip("/"), *rel.parts])
            fs_s3.put(str(p), remote_path); uploaded += 1
        logger.info("Burned  upload: %d file(s) at %s/*.parquet", uploaded, dest_b)

        if not args.keep_local:
            import shutil
            shutil.rmtree(local_d, ignore_errors=True)
            shutil.rmtree(local_b, ignore_errors=True)

    # Optional cleanup after all intervals
    if not args.keep_local:
        import shutil
        logger.debug("Removing local staging directory %s", base_dir_root)
        shutil.rmtree(base_dir_root, ignore_errors=True)

    # Teardown
    if client:
        logger.debug("Closing Dask client")
        client.close()
    if cluster:
        logger.debug("Closing cluster")
        cluster.close()
    uu.stage_duration(start_ts, uu.timestr(), stage)


def main(argv=None):
    """Parse CLI args and dispatch to :func:`run`.

    If no args are provided, runs a 1×1-degree local smoke test (Borneo snippet).
    """
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("No CLI args: running 1×1-degree local smoke test")
        argv = [
            "--model_version", "test",
            "--run_date", "20250101",
            "--interval_end_years", "2020",
            "--run_local",
            "--bounding_box", "112", "-2", "113", "-1",
            "--tile_pixels", "4000",
            "--chunk_size", "1024",
            "--run_name", "smoke",
        ]

    parser = argparse.ArgumentParser(description="Run organic-soils zonal statistics directly on GeoTIFF/COG (VRT streaming)")
    parser.add_argument("--model_version", required=True, help="Model version string")
    parser.add_argument("--run_date", required=True, help="Run date (YYYYMMDD)")
    parser.add_argument("--interval_end_years", nargs="+", type=int, required=True, help="Interval end years (e.g., 2010 2015 2020 2024)")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Window block size in pixels for streaming reads (e.g., 1024)")
    parser.add_argument(
        "--tile_pixels",
        type=int,
        default=4000,  # 4000 for 1×1°, 40000 for 10×10° (both give 1/4000° resolution)
        help="Input tile size in pixels used by the source outputs (4000 for 1×1°, 40000 for 10×10°).",
    )
    parser.add_argument("--local_output", default="/tmp/zonal_stats", help="Local staging directory for Parquet output")
    parser.add_argument("--keep_local", action="store_true", help="Keep staged files after upload")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    parser.add_argument("--run_name", default="ogh_standard_model", help="Model run name")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run_local", action="store_true", help="Run locally without Dask/Coiled")
    mode.add_argument("--cluster_name", default="zonal_stats", help="Name of the Coiled cluster to attach to")
    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W S E N")
    parser.add_argument("--tile_ids", action="append", help="Comma separated 10×10 tile IDs (e.g., 00N_110E)")

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()

"""
Examples:

# Single interval (GeoTIFF/COG; COG-native; per-interval upload)
python -m src.scripts.zonal_statistics.run_zonal_statistics_tiff \
  --interval_end_years 2005 \
  --cluster_name zonal_stats \
  --run_date 20250825 \
  --model_version 0_7_0 \
  --run_name ogh_standard_model \
  --tile_pixels 4000 \
  --chunk_size 4000 \
  --tile_ids 00N_110E

# Multiple intervals (no Zarr; immediate per-interval uploads)
python -m src.scripts.zonal_statistics.run_zonal_statistics_tiff \
  --interval_end_years 2010 2015 2020 2024 \
  --cluster_name zonal_stats \
  --run_date 20250825 \
  --model_version 0_7_0 \
  --run_name ogh_standard_model \
  --tile_pixels 4000 \
  --chunk_size 1024

# Bounding box run (local smoke test)
python -m src.scripts.zonal_statistics.run_zonal_statistics_tiff \
  --interval_end_years 2020 \
  --run_local \
  --bounding_box 112 -2 113 -1 \
  --run_date 20250101 \
  --model_version test \
  --run_name smoke \
  --tile_pixels 4000 \
  --chunk_size 1024
"""
