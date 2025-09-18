# -*- coding: utf-8 -*-
"""Run organic-soils zonal statistics (GeoTIFF-only; no fill; no halo; warn on GADM=0).

- Reads all inputs directly from GeoTIFFs (no Zarr creation).
- Aligns contextual layers (ADM0, pixel_area) to the *state-node* raster grid
  with a strict "nearest within 0.5 pixel" rule.
- Sums flux and area by (gadm_adm0, state_nodes, flux_type). Area is on-land only.
- Emits warnings (not failures) if some flux lands in gadm_adm0=0 (unassigned/ocean)
  or if the state-node layer appears to be all zeros in the AOI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dask
import dask.array as da
import numpy as np
import pandas as pd
import posixpath
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import s3fs
import shutil
import xarray as xr

import flox
from flox import ReindexArrayType, ReindexStrategy
from flox.xarray import xarray_reduce

# Project imports
import src.scripts.zonal_statistics.zonal_constants as zc
from src.scripts.zonal_statistics.zonal_constants import (
    DRAINED_STATE_NODE_MEANINGS,
    BURNED_STATE_NODE_MEANINGS,
    ALL_DRAINED_STATE_CODES,
    ALL_BURNED_STATE_CODES,
)
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities.universal_utilities import timestr
from src.scripts.utilities import log_utilities as lu

# ------------------------------ configuration ------------------------------
SPARSE_DEFAULT = True

ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
OUTPUT_BASE = "{root}/version_{model_version}"

# Inputs (same layout you’ve been using)
FOLDER_TEMPLATE = (
    OUTPUT_BASE
    + "/{folder}/{run_name}/five_year_intervals/{interval}/{tile_pixels}_pixels/{run_date}/"
)

DATASETS = {
    "drained_state_nodes": {"folder": "drained_state", "var": "drained_state_nodes"},
    "burned_state_nodes":  {"folder": "burned_state",  "var": "burned_state_nodes"},
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

# Contextual (10° / 40000 px @ 0.00025°)
ADM0_GTIFF_FOLDER = (
    "s3://gfw2-data/gadm_administrative_boundaries/v4.1/"
    "v4.1.64__from_gfw-data-lake/raster/epsg-4326/10/40000/adm0/gdal-geotiff/"
)
PIXEL_AREA_GTIFF_FOLDER = (
    "s3://gfw2-data/analyses/umd_area_2013__from_gfw-data-lake/"
    "v1.10/raster/epsg-4326/10/40000/area_m/gdal-geotiff/"
)

# ------------------------------ helpers -----------------------------------
def flox_sparse_kwargs(use_sparse: bool) -> dict:
    if not use_sparse:
        return {}
    try:
        return {
            "reindex": ReindexStrategy(blockwise=False, array_type=ReindexArrayType.SPARSE_COO),
            "fill_value": 0,
        }
    except Exception:
        return {}

def build_output_parquet(model_version: str, run_name: str, run_date: str, interval: str) -> str:
    return posixpath.join(
        ROOT, f"version_{model_version}", "zonal_stats", run_name, run_date, interval, ""
    )

def list_geotiffs(base_uri: str) -> List[str]:
    fs = s3fs.S3FileSystem(anon=False)
    return [f if f.startswith("s3://") else f"s3://{f}" for f in fs.glob(base_uri.rstrip("/") + "/**/*.tif")]

def uris_for_dataset(
    interval: str,
    ds_key: str,
    *,
    root: str,
    tile_pixels: int,
    run_name: str,
    run_date: str,
    model_version: str,
) -> Tuple[List[str], Optional[str]]:
    """Return (uris, unit) for dataset in an interval; unit is 'ha' or 'pixel' for flux."""
    fmt = dict(
        root=root,
        model_version=model_version,
        run_name=run_name,
        run_date=run_date,
        tile_pixels=tile_pixels,
        interval=interval,
    )
    spec = DATASETS[ds_key]
    if "folder_candidates" in spec:
        ranked = []
        for name in spec["folder_candidates"]:
            uri = FOLDER_TEMPLATE.format(folder=name, **fmt)
            unit = "ha" if "_ha_" in name else "pixel"
            ranked.append((uri, unit))
        ranked.sort(key=lambda x: 0 if x[1] == "ha" else 1)
        for base, unit in ranked:
            got = list_geotiffs(base)
            if got:
                return got, unit
        raise FileNotFoundError(f"No GeoTIFFs found for {ds_key} in {ranked}")
    else:
        base = FOLDER_TEMPLATE.format(folder=spec["folder"], **fmt)
        got = list_geotiffs(base)
        if not got:
            raise FileNotFoundError(f"No GeoTIFFs found for {ds_key} in {base}")
        return got, None

def filter_uris_by_bbox(uris: List[str], bbox: Optional[List[float]]) -> List[str]:
    if not bbox:
        return uris
    w, s, e, n = bbox
    out = []
    import rasterio
    for u in uris:
        try:
            with rasterio.Env(), rasterio.open(u) as src:
                b = src.bounds
                if (b.right <= w) or (b.left >= e) or (b.top <= s) or (b.bottom >= n):
                    continue
                out.append(u)
        except Exception:
            continue
    return out

def open_mosaic(uris: List[str], chunk_px: int, bbox: Optional[List[float]]) -> xr.DataArray:
    """Open many single-band GeoTIFFs as one 2-D mosaic (dask-backed)."""
    ds = xr.open_mfdataset(
        uris,
        engine="rasterio",
        chunks={"x": chunk_px, "y": chunk_px},
        combine="by_coords",
        parallel=True,
        mask_and_scale=False,
    )
    da = next(iter(ds.data_vars.values()))
    if "band" in da.dims:
        da = da.isel(band=0, drop=True)
    da = da.rename({k: {"lon": "x", "latitude": "y", "lat": "y"}.get(k, k) for k in da.dims})
    if "x" in da.coords:
        da = da.assign_coords(x=np.round(da.x.astype(float), 12))
    if "y" in da.coords:
        da = da.assign_coords(y=np.round(da.y.astype(float), 12))
    if bbox and {"x","y"}.issubset(da.dims):
        w, s, e, n = bbox
        xasc = bool(da.x[0] < da.x[-1])
        yasc = bool(da.y[0] < da.y[-1])
        xs = slice(min(w,e), max(w,e)) if xasc else slice(max(e,w), min(e,w))
        ys = slice(min(s,n), max(s,n)) if yasc else slice(max(n,s), min(n,s))
        da = da.sel(x=xs, y=ys)
    return da

def pixel_step(a: xr.DataArray, dim: str) -> float:
    v = a.coords[dim].values
    return float(abs(v[1] - v[0])) if v.size > 1 else np.nan

def align_like(ds: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Nearest neighbor with 0.5-pixel tolerance per axis; warn if NaNs introduced."""
    if not {"x","y"}.issubset(ds.dims):
        return ds
    for axis in ("x","y"):
        if axis in ds.coords:
            ds = ds.assign_coords({axis: np.round(ds[axis].astype(float), 12)})
    rx, ry = pixel_step(ref,"x"), pixel_step(ref,"y")
    kwx = {"method": "nearest"}
    if rx == rx:
        kwx["tolerance"] = rx/2.0
    kwy = {"method": "nearest"}
    if ry == ry:
        kwy["tolerance"] = ry/2.0
    out = ds.sel(x=ref.x, **kwx).sel(y=ref.y, **kwy).assign_coords(x=ref.x, y=ref.y)
    if bool(out.isnull().any().compute().item()):
        logging.warning("Alignment introduced NaNs; check grids (tolerance=0.5 px).")
    return out

def log_array_summary(logger: logging.Logger, name: str, arr: xr.DataArray, *, categories=False) -> None:
    mn, mx, mean = dask.compute(arr.min(), arr.max(), arr.mean())
    logger.debug("%s stats → min=%s max=%s mean=%s", name, mn, mx, mean)
    if categories:
        samp = arr.isel(y=slice(0,None,512), x=slice(0,None,512))
        uniq = np.array(da.unique(samp.data).compute())
        logger.debug("%s unique sample (%d): %s", name, uniq.size, uniq.tolist()[:20])

def build_interval_pairs(end_years: List[int]) -> List[Tuple[int,int]]:
    mapping = {end: (start, end) for start, end in cn.five_year_inventory_periods}
    out = []
    for y in end_years:
        if y not in mapping:
            raise ValueError(f"Interval end year {y} not supported. Valid: {sorted(mapping)}")
        out.append(mapping[y])
    return out

def make_df_for_reduce(res: xr.DataArray, interval_end: int, flux_type_map: Dict[int,str], state_var: str, adm_field: str) -> pd.DataFrame:
    arr = res.data
    if isinstance(arr, da.Array):
        arr = arr.compute()
    if hasattr(arr, "coords") and hasattr(arr, "data"):  # sparse.COO
        indices, values = arr.coords, arr.data
        coord_vals = {dim: res.coords[dim].values[indices[i]] for i, dim in enumerate(res.dims)}
        coord_vals["value"] = values
    else:
        grid = np.indices(arr.shape)
        coord_vals = {dim: res.coords[dim].values[grid[i].ravel()] for i, dim in enumerate(res.dims)}
        coord_vals["value"] = arr.ravel()
    df = pd.DataFrame(coord_vals)
    df["flux_type"] = df["flux_type"].replace(flux_type_map)
    df["interval_end"] = interval_end
    if adm_field in df.columns:
        df[adm_field] = df[adm_field].round().astype("uint32")
    if state_var == "drained_state_nodes" and "drained_state_nodes" in df.columns:
        df["drained_state_meaning"] = df["drained_state_nodes"].astype("string").str.zfill(8).map(DRAINED_STATE_NODE_MEANINGS)
    if state_var == "burned_state_nodes" and "burned_state_nodes" in df.columns:
        df["burned_state_meaning"] = df["burned_state_nodes"].astype("string").str.zfill(8).map(BURNED_STATE_NODE_MEANINGS)
    df.loc[df["flux_type"].eq("area__ha"), "value"] = df["value"] / 10000.0
    return df

def warn_share_to_gadm0(res: xr.DataArray, flux_value: int, label: str, logger: logging.Logger) -> None:
    try:
        arr = res.sel(flux_type=flux_value)
    except Exception:
        return
    if 0 not in arr.coords["gadm_adm0"].values:
        return
    total = float(arr.sum().compute())
    ocean = float(arr.sel(gadm_adm0=0).sum().compute())
    pct = (ocean / total * 100.0) if total else 0.0
    if pct > 0:
        logger.warning("Unassigned (gadm_adm0=0) share for %s: %.4f%% of sum.", label, pct)

# ------------------------------ main runner --------------------------------
def run(args: argparse.Namespace) -> None:
    stage = "zonal_statistics_gtiff_simple"
    start_ts = uu.timestr()

    cluster, client, run_local = uu.connect_to_cluster(
        cluster_name=args.cluster_name, run_local=args.run_local
    )
    logger, _ = lu.populate_main_log_header(
        bounding_box=args.bounding_box,
        use_shapefile=False,
        client=client,
        cluster=cluster,
        log_note="Organic soils zonal statistics (GeoTIFF-only, no fill, no halo)",
        run_local=run_local,
        model_type="organic_soils",
        stage=stage,
    )
    if args.debug:
        logger.setLevel(logging.DEBUG)

    bbox = [float(x) for x in args.bounding_box] if args.bounding_box else None
    logger.debug("Starting GeoTIFF-only (simple) with args: %s", args)

    # Output + local staging
    OUTPUT_KW = dict(root=ROOT, model_version=args.model_version, run_date=args.run_date, run_name=args.run_name)
    fs_s3 = s3fs.S3FileSystem(anon=False)
    local_fs = pafs.LocalFileSystem()
    base_dir_root = Path(args.local_output).expanduser().resolve()
    base_dir_root.mkdir(parents=True, exist_ok=True)
    local_d = base_dir_root / "drained"
    local_b = base_dir_root / "burned"
    for p in (local_d, local_b):
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True, exist_ok=True)

    # Contextual mosaics (filter to AOI tiles)
    adm0_uris = filter_uris_by_bbox(list_geotiffs(ADM0_GTIFF_FOLDER), bbox)
    area_uris = filter_uris_by_bbox(list_geotiffs(PIXEL_AREA_GTIFF_FOLDER), bbox)
    logger.info("Contextual tiles selected: ADM0=%d, pixel_area=%d", len(adm0_uris), len(area_uris))

    gadm_adm0_ids = np.array(zc.GADM_ADM0_IDS, dtype=np.float32)
    drained_codes_arr = np.array(sorted({0, *map(int, ALL_DRAINED_STATE_CODES)}), dtype=np.uint32)
    burned_codes_arr  = np.array(sorted({0, *map(int, ALL_BURNED_STATE_CODES)}), dtype=np.uint32)

    for start_y, end_y in build_interval_pairs(args.interval_end_years):
        interval = f"{start_y}_{end_y}"
        logger.info("Processing interval %s : %s", interval, timestr())

        # Flux + state URIs
        drained_total_uris, drained_unit = uris_for_dataset(interval, "drained_total",
                                                            tile_pixels=args.tile_pixels, **OUTPUT_KW)
        burned_total_uris, burned_unit   = uris_for_dataset(interval, "burned_total",
                                                            tile_pixels=args.tile_pixels, **OUTPUT_KW)
        drained_state_uris, _ = uris_for_dataset(interval, "drained_state_nodes",
                                                 tile_pixels=args.tile_pixels, **OUTPUT_KW)
        burned_state_uris,  _ = uris_for_dataset(interval, "burned_state_nodes",
                                                 tile_pixels=args.tile_pixels, **OUTPUT_KW)

        # Filter to AOI and log counts
        drained_total_uris = filter_uris_by_bbox(drained_total_uris, bbox)
        burned_total_uris  = filter_uris_by_bbox(burned_total_uris,  bbox)
        drained_state_uris = filter_uris_by_bbox(drained_state_uris, bbox)
        burned_state_uris  = filter_uris_by_bbox(burned_state_uris,  bbox)
        logger.info("URIs in AOI — drained_total=%d, burned_total=%d, drained_state=%d, burned_state=%d",
                    len(drained_total_uris), len(burned_total_uris), len(drained_state_uris), len(burned_state_uris))
        if args.debug:
            for lbl, lst in [("drained_state (sample)", drained_state_uris),
                             ("drained_total (sample)", drained_total_uris),
                             ("burned_state (sample)", burned_state_uris),
                             ("burned_total (sample)", burned_total_uris)]:
                if lst:
                    logging.debug("%s: %s", lbl, lst[:3])

        # Hard guard: we need state rasters present in AOI
        if not drained_state_uris:
            raise FileNotFoundError(f"No drained_state_nodes GeoTIFFs overlap AOI for {interval}.")
        if not burned_state_uris:
            logging.warning("No burned_state_nodes GeoTIFFs overlap AOI for %s — burned totals may be empty.", interval)

        # Open mosaics (dask-backed)
        drained_state = open_mosaic(drained_state_uris, args.chunk_size, bbox).astype("uint32")
        burned_state  = open_mosaic(burned_state_uris,  args.chunk_size, bbox).astype("uint32") if burned_state_uris else xr.zeros_like(drained_state, dtype="uint32")

        reference = drained_state  # define canonical grid

        drained_total = open_mosaic(drained_total_uris, args.chunk_size, bbox)
        burned_total  = open_mosaic(burned_total_uris,  args.chunk_size, bbox)

        adm0         = open_mosaic(adm0_uris,        args.chunk_size, bbox)
        pixel_area   = open_mosaic(area_uris,        args.chunk_size, bbox)

        # Align contextual + flux to the state grid
        adm0_aligned       = align_like(adm0, reference).fillna(0.0)
        pixel_area_aligned = align_like(pixel_area, reference).fillna(0.0)
        drained_total      = align_like(drained_total, reference).fillna(0.0)
        burned_total       = align_like(burned_total, reference).fillna(0.0)
        burned_state       = align_like(burned_state, reference).fillna(0).astype("uint32")

        # Convert per-ha → per-pixel totals if needed
        if drained_unit == "ha":
            drained_total = drained_total * (pixel_area_aligned / 10000.0)
        if burned_unit == "ha":
            burned_total = burned_total * (pixel_area_aligned / 10000.0)

        # Land-only area (but do NOT zero-out flux; we only zero area on ocean)
        pixel_area_land = xr.where(adm0_aligned > 0, pixel_area_aligned, 0.0)

        # QC: prove we actually have signal + state codes > 0 in the AOI
        if args.debug:
            log_array_summary(logger, "drained_total", drained_total)
            log_array_summary(logger, "burned_total",  burned_total)
            log_array_summary(logger, "pixel_area_land", pixel_area_land)
            log_array_summary(logger, "adm0_aligned", adm0_aligned, categories=True)
            log_array_summary(logger, "drained_state", drained_state, categories=True)
            log_array_summary(logger, "burned_state",  burned_state,  categories=True)

        try:
            samp = drained_state.isel(y=slice(0,None,512), x=slice(0,None,512))
            has_nonzero_state = bool((samp.data > 0).any().compute().item())
        except Exception:
            has_nonzero_state = True
        if not has_nonzero_state:
            logger.warning(
                "AOI %s: drained_state_nodes look all zeros after alignment. "
                "This will push everything into state=0 bins. Check inputs and alignment.",
                interval
            )

        # --- Reduce: drained ---
        with dask.annotate(label=f"reduce:drained:{interval}"):
            cube_d = xr.concat([drained_total, pixel_area_land], dim="flux_type").assign_coords(
                flux_type=("flux_type", [0, 2])
            )
            res_d = xarray_reduce(
                cube_d,
                adm0_aligned.astype("float32"),
                drained_state,
                func="sum",
                expected_groups=(gadm_adm0_ids, drained_codes_arr),
                **flox_sparse_kwargs(not args.no_sparse),
            ).compute()

        warn_share_to_gadm0(res_d, flux_value=0, label=f"drained_total {interval}", logger=logger)
        df_d = make_df_for_reduce(res_d, end_y, {0: "drained_total_Mg_CO2e", 2: "area__ha"},
                                  "drained_state_nodes", "gadm_adm0")

        # --- Reduce: burned ---
        with dask.annotate(label=f"reduce:burned:{interval}"):
            cube_b = xr.concat([burned_total, pixel_area_land], dim="flux_type").assign_coords(
                flux_type=("flux_type", [1, 2])
            )
            res_b = xarray_reduce(
                cube_b,
                adm0_aligned.astype("float32"),
                burned_state,
                func="sum",
                expected_groups=(gadm_adm0_ids, burned_codes_arr),
                **flox_sparse_kwargs(not args.no_sparse),
            ).compute()

        warn_share_to_gadm0(res_b, flux_value=1, label=f"burned_total {interval}", logger=logger)
        df_b = make_df_for_reduce(res_b, end_y, {1: "burned_total_Mg_CO2e", 2: "area__ha"},
                                  "burned_state_nodes", "gadm_adm0")

        # --- Write locally (per-interval, single file each) ---
        out_d = (local_d / f"{interval}.parquet")
        out_b = (local_b / f"{interval}.parquet")
        ds.write_dataset(pa.Table.from_pandas(df_d, preserve_index=False),
                         base_dir=str(out_d), filesystem=local_fs, format="parquet",
                         existing_data_behavior="overwrite_or_ignore")
        ds.write_dataset(pa.Table.from_pandas(df_b, preserve_index=False),
                         base_dir=str(out_b), filesystem=local_fs, format="parquet",
                         existing_data_behavior="overwrite_or_ignore")

        # --- Upload (replace interval folders) ---
        dest_root = build_output_parquet(args.model_version, args.run_name, args.run_date, interval)
        dest_d = posixpath.join(dest_root, "drained")
        dest_b = posixpath.join(dest_root, "burned")
        try: fs_s3.rm(dest_d + "/", recursive=True)
        except Exception: pass
        try: fs_s3.rm(dest_b + "/", recursive=True)
        except Exception: pass
        for pth, dest in [(out_d, dest_d), (out_b, dest_b)]:
            for f in Path(pth).rglob("*.parquet"):
                fs_s3.put(str(f), posixpath.join(dest, f.name))
        logger.info("Uploaded drained → %s", dest_d)
        logger.info("Uploaded burned  → %s", dest_b)

    # Cleanup
    if not args.keep_local:
        shutil.rmtree(base_dir_root, ignore_errors=True)

    if client: client.close()
    if cluster: cluster.close()
    uu.stage_duration(start_ts, uu.timestr(), stage)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Run organic-soils zonal statistics (GeoTIFF-only; no fill/halo)")
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--run_date", required=True)
    parser.add_argument("--interval_end_years", nargs="+", type=int, required=True)
    parser.add_argument("--tile_pixels", type=int, default=4000)
    parser.add_argument("--chunk_size", type=int, default=10000, help="Dask chunk size in pixels")
    parser.add_argument("--local_output", default="/tmp/zonal_stats_gtiff")
    parser.add_argument("--keep_local", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_sparse", action="store_true", default=not SPARSE_DEFAULT)
    parser.add_argument("--run_name", default="ogh_standard_model")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run_local", action="store_true")
    mode.add_argument("--cluster_name", default="zonal_stats")
    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W S E N")
    args = parser.parse_args(argv)
    run(args)

if __name__ == "__main__":
    main()

"""
python -m src.scripts.zonal_statistics.run_zonal_statistics_gtiff \
  --interval_end_years 2005 \
  --cluster_name zonal_stats \
  --run_date 20250825 \
  --model_version 0_7_0 \
  --run_name ogh_standard_model \
  --tile_pixels 4000 \
  --chunk_size 10000 \
  --bounding_box 95 -9 120 4 \
  --debug

python -m src.scripts.zonal_statistics.run_zonal_statistics_gtiff \
  --interval_end_years 2005 \
  --cluster_name zonal_stats \
  --run_date 20250825 \
  --model_version 0_7_0 \
  --run_name ogh_standard_model \
  --tile_pixels 4000 \
  --chunk_size 10000 \
  --tile_id 00N_110E \
  --debug

"""