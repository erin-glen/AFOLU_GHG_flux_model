# -*- coding: utf-8 -*-
"""
QA/QC for drained_state:
- Compares raw GeoTIFF mosaic vs. Zarr cache for drained_state_nodes
- Uses the same path logic as run_zonal_statistics (real S3 folders)
- Summarizes unique codes, selected code counts, and raw↔zarr mismatches

Example:
python -m src.scripts.zonal_statistics.qaqc_drained_state \
    --model_version 0_6_0 \
    --run_date 20250807 \
    --interval_end_years 2020 2024 \
    --tile_ids 00N_110E \
    --cluster_name zonal_stats \
    --debug
"""

import argparse
import logging
import posixpath
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pandas as pd
import rasterio as rio
import s3fs
import xarray as xr

# ── Repo utilities / constants ──────────────────────────────────────────
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities.universal_utilities import timestr
from src.scripts.utilities import log_utilities as lu
from src.scripts.utilities import constants_and_names as cn

# ── Keep these in sync with run_zonal_statistics ────────────────────────
ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
OUTPUT_BASE = "{root}/version_{model_version}"

DATASETS = {
    "drained_state_nodes": {
        "folder": "drained_state",
        "zarr": "drained_state_node_{interval}.zarr",
    },
}

ZARR_CACHE_PREFIX = OUTPUT_BASE + "/zarr/{run_date}/{interval}/"
FOLDER_TEMPLATE = (
    OUTPUT_BASE
    + "/{folder}/ogh_standard_model/five_year_intervals/{interval}/"
      "40000_pixels/{run_date}/"
)

NONPEAT = np.uint32(20000000)
PEAT_UNDRAINED = np.uint32(16000000)

# ── Pathing helpers (mirrors run_zonal_statistics) ──────────────────────
def build_paths(interval: str, **kw) -> dict[str, dict[str, str]]:
    zarr_base = ZARR_CACHE_PREFIX.format(interval=interval, **kw)
    paths: dict[str, dict[str, str]] = {}
    for name, spec in DATASETS.items():
        paths[name] = {
            "folder": FOLDER_TEMPLATE.format(folder=spec["folder"], interval=interval, **kw),
            "zarr": posixpath.join(zarr_base, spec["zarr"].format(interval=interval)),
        }
    return paths

def list_folder_uris(base_uri: str) -> pd.Series:
    fs = s3fs.S3FileSystem(anon=False)
    pattern = base_uri.rstrip("/") + "/**/*.tif"
    tif_files = [(f if f.startswith("s3://") else f"s3://"+f) for f in fs.glob(pattern)]
    return pd.Series(tif_files, dtype="string")

# ── Data opening helpers ────────────────────────────────────────────────
def make_xarray_chunks(tile_uris: pd.Series, chunk_size: int) -> xr.Dataset:
    """Mosaic by coords with rasterio; squeeze band; stable ordering."""
    uris = sorted(tile_uris.tolist())
    return xr.open_mfdataset(
        uris,
        engine="rasterio",
        combine="by_coords",
        preprocess=lambda ds: ds.squeeze(drop=True),
        parallel=True,
        chunks={"x": chunk_size, "y": chunk_size},
        mask_and_scale=False,  # keep integer codes as-is
    )

def open_zarr_region(path: str, bbox, chunk_size: int) -> xr.DataArray:
    """Open a Zarr (v2/v3) and crop to bbox if present."""
    s3_opts = {"anon": False}
    with dask.annotate(label=f"open:{Path(path).stem}"):
        ds = xr.open_zarr(path, consolidated=None, storage_options=s3_opts)

    if isinstance(ds, xr.DataArray):
        da_ = ds
    else:
        vars_xy = [v for v in ds.data_vars.values() if {"x","y"}.issubset(v.dims)]
        da_ = vars_xy[0] if vars_xy else next(iter(ds.data_vars.values()))

    if "band" in da_.dims:
        da_ = da_.isel(band=0, drop=True)

    if bbox is not None and {"x","y"}.issubset(da_.dims):
        west, south, east, north = bbox
        x_asc = bool(da_.x[0] < da_.x[-1])
        y_asc = bool(da_.y[0] < da_.y[-1])
        x_slice = slice(min(west, east), max(west, east)) if x_asc else slice(max(east, west), min(east, west))
        y_slice = slice(min(south, north), max(south, north)) if y_asc else slice(max(north, south), min(north, south))
        da_ = da_.sel(x=x_slice, y=y_slice)

    chunk_dict = {d: chunk_size for d in ("x","y") if d in da_.dims}
    if chunk_dict:
        da_ = da_.chunk(chunk_dict)
    return da_

def safe_crop(ds: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    return ds.sel(x=ref.x, y=ref.y, method="nearest").assign_coords(x=ref.x, y=ref.y)

# ── Simple metrics ──────────────────────────────────────────────────────
def dask_unique(arr: xr.DataArray, sample: int | None = None) -> np.ndarray:
    """Unique values; if sample set, sample N rows/cols (fast mode)."""
    data = arr.data
    if sample:
        # Take a central window for a quick scan
        ys = slice(max(0, arr.sizes["y"]//2 - sample//2), min(arr.sizes["y"], arr.sizes["y"]//2 + sample//2))
        xs = slice(max(0, arr.sizes["x"]//2 - sample//2), min(arr.sizes["x"], arr.sizes["x"]//2 + sample//2))
        data = arr.isel(y=ys, x=xs).data
    return np.array(da.unique(data).compute())

def count_code(arr: xr.DataArray, code: np.uint32) -> int:
    return int((arr.data == code).sum().compute())

def mismatch_summary(raw: xr.DataArray, z: xr.DataArray, pairs=( (NONPEAT, np.uint32(0)), (np.uint32(0), NONPEAT) )) -> dict:
    out = {}
    neq = (raw.data != z.data).sum().compute()
    out["mismatch_total_pixels"] = int(neq)
    for (a, b) in pairs:
        cnt = ((raw.data == a) & (z.data == b)).sum().compute()
        out[f"raw_{int(a)}__z_{int(b)}"] = int(cnt)
    return out

def report_geotiff_meta(tile_uris: list[str], logger: logging.Logger, max_files: int = 20):
    """Print dtype/nodata for a sample of input GeoTIFFs and flag inconsistencies."""
    dtypes, nodatas = [], []
    has_zero, has_nonpeat, has_undrained = False, False, False
    for i, uri in enumerate(sorted(tile_uris)[:max_files]):
        with rio.open(uri) as ds:
            dtypes.append(ds.dtypes[0])
            nodatas.append(ds.nodata)
            # quick small window read
            wnd = ds.read(1, window=((0, min(1024, ds.height)), (0, min(1024, ds.width))), masked=False)
            u = np.unique(wnd)
            has_zero |= (0 in u)
            has_nonpeat |= (20000000 in u)
            has_undrained |= (16000000 in u)
            logger.debug("GeoTIFF %s: dtype=%s nodata=%s sample_min=%s sample_max=%s", uri, ds.dtypes[0], ds.nodata, u[0], u[-1])
    logger.info("GeoTIFF dtypes (sample): %s", sorted(set(dtypes)))
    logger.info("GeoTIFF nodata  (sample): %s", sorted(set(nodatas)))
    logger.info("GeoTIFF sample presence → 0:%s  16000000:%s  20000000:%s", has_zero, has_undrained, has_nonpeat)

# ── Interval mapping (same logic as run_zonal) ─────────────────────────
def build_interval_pairs(end_years: list[int]) -> list[tuple[int, int]]:
    mapping = {end: (start, end) for start, end in cn.five_year_inventory_periods}
    pairs = []
    for y in end_years:
        if y not in mapping:
            raise ValueError(f"Interval end year {y} not supported. Valid: {sorted(mapping)}")
        pairs.append(mapping[y])
    return pairs

# ── Main QA routine ────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> None:
    # optional cluster attach
    cluster, client, run_local = uu.connect_to_cluster(
        cluster_name=args.cluster_name,
        run_local=args.run_local,
    )
    logger, _ = lu.populate_main_log_header(
        bounding_box=None,
        use_shapefile=False,
        client=client,
        cluster=cluster,
        log_note="QAQC drained_state vs Zarr",
        run_local=run_local,
        model_type="organic_soils",
        stage="qaqc_drained_state",
    )
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # bbox from tile ids (union)
    bbox = None
    if args.bounding_box:
        bbox = [float(x) for x in args.bounding_box]
    elif args.tile_ids:
        tiles = []
        for item in args.tile_ids:
            tiles.extend(t.strip() for t in item.split(",") if t.strip())
        if tiles:
            bounds = [uu.get_10x10_tile_bounds(t) for t in tiles]
            west = min(b[0] for b in bounds); south = min(b[1] for b in bounds)
            east = max(b[2] for b in bounds); north = max(b[3] for b in bounds)
            bbox = [west, south, east, north]
    if bbox is None:
        raise SystemExit("Provide --tile_ids or --bounding_box for a focused QA window.")

    # Resolve paths
    OUTPUT_KW = dict(root=ROOT, model_version=args.model_version, run_date=args.run_date)
    interval_pairs = build_interval_pairs(args.interval_end_years)

    for start_year, end_year in interval_pairs:
        interval = f"{start_year}_{end_year}"
        logger.info("==== Interval %s ====", interval)
        paths = build_paths(interval, **OUTPUT_KW)
        folder = paths["drained_state_nodes"]["folder"]
        zarr_path = paths["drained_state_nodes"]["zarr"]
        logger.info("GeoTIFF folder: %s", folder)
        logger.info("Zarr cache    : %s", zarr_path)

        # Gather GeoTIFFs; restrict to tiles if provided
        uris = list_folder_uris(folder)
        if args.tile_ids:
            tiles_filter = []
            for item in args.tile_ids:
                tiles_filter.extend(t.strip() for t in item.split(",") if t.strip())
            sel = pd.Series(False, index=uris.index)
            for t in tiles_filter:
                sel = sel | uris.str.contains(t)
            tile_uris = uris[sel].tolist()
        else:
            tile_uris = uris.tolist()

        if not tile_uris:
            logger.error("No GeoTIFFs found for %s (interval %s)", folder, interval)
            continue

        # Report GeoTIFF meta (dtype, nodata, quick value presence)
        report_geotiff_meta(tile_uris, logger, max_files=20)

        # Open raw mosaic constrained to bbox
        logger.info("Opening raw mosaic (GeoTIFF) ...")
        ds_raw = make_xarray_chunks(pd.Series(tile_uris, dtype="string"), args.chunk_size)
        raw = ds_raw
        if isinstance(raw, xr.Dataset):
            # pick first var with x,y
            v = next((v for v in raw.data_vars.values() if {"x","y"}.issubset(v.dims)), None)
            if v is None:
                raise RuntimeError("Could not find (x,y) var in raw mosaic.")
            raw = v
        if "band" in raw.dims:
            raw = raw.isel(band=0, drop=True)
        # crop to bbox
        w,s,e,n = bbox
        x_asc = bool(raw.x[0] < raw.x[-1])
        y_asc = bool(raw.y[0] < raw.y[-1])
        x_slice = slice(min(w,e), max(w,e)) if x_asc else slice(max(e,w), min(e,w))
        y_slice = slice(min(s,n), max(s,n)) if y_asc else slice(max(n,s), min(n,s))
        raw = raw.sel(x=x_slice, y=y_slice).astype("uint32").chunk({"x": args.chunk_size, "y": args.chunk_size})

        # Open Zarr and align
        logger.info("Opening Zarr region ...")
        z_da = open_zarr_region(zarr_path, bbox, args.chunk_size).astype("uint32")
        # align by coords
        try:
            z_da_aligned = safe_crop(z_da, raw)
        except Exception:
            raw = safe_crop(raw, z_da)  # whichever works; we just need identical grids
            z_da_aligned = z_da
        # Final names for clarity
        raw.name = "drained_state_nodes_raw"
        z_da_aligned.name = "drained_state_nodes_zarr"

        # Uniques (optionally sampled)
        logger.info("Computing unique code sets (sample=%s)", args.unique_sample)
        u_raw = sorted(map(int, dask_unique(raw, sample=args.unique_sample)))
        u_z   = sorted(map(int, dask_unique(z_da_aligned, sample=args.unique_sample)))
        logger.info("Unique (raw): %s", u_raw[:50])
        logger.info("Unique (zarr): %s", u_z[:50])

        # Key code counts
        logger.info("Counting key codes over bbox ...")
        stats = {
            "raw_count_0": count_code(raw, np.uint32(0)),
            "raw_count_16000000": count_code(raw, PEAT_UNDRAINED),
            "raw_count_20000000": count_code(raw, NONPEAT),
            "zarr_count_0": count_code(z_da_aligned, np.uint32(0)),
            "zarr_count_16000000": count_code(z_da_aligned, PEAT_UNDRAINED),
            "zarr_count_20000000": count_code(z_da_aligned, NONPEAT),
        }
        logger.info("Counts → raw: {raw_count_0} (0), {raw_count_16000000} (16M), {raw_count_20000000} (20M)".format(**stats))
        logger.info("Counts → zarr: {zarr_count_0} (0), {zarr_count_16000000} (16M), {zarr_count_20000000} (20M)".format(**stats))

        # Mismatch summary
        logger.info("Computing mismatch summary (this may take time for full bbox) ...")
        mm = mismatch_summary(raw, z_da_aligned, pairs=((NONPEAT, np.uint32(0)), (np.uint32(0), NONPEAT)))
        logger.info("Total mismatched pixels: %s", mm["mismatch_total_pixels"])
        logger.info("raw=20000000 & zarr=0 : %s", mm["raw_20000000__z_0"])
        logger.info("raw=0 & zarr=20000000 : %s", mm["raw_0__z_20000000"])

    # close cluster
    if client:
        client.close()
    if cluster:
        cluster.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="QA/QC drained_state vs Zarr")
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--run_date", required=True)
    parser.add_argument("--interval_end_years", nargs="+", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, default=4000)
    parser.add_argument("--unique_sample", type=int, default=None,
                        help="Pixels per side for a central sample window (fast); omit for full.")
    parser.add_argument("--debug", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run_local", action="store_true")
    mode.add_argument("--cluster_name", default="zonal_stats")
    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W S E N")
    parser.add_argument("--tile_ids", action="append", help="Comma separated tile IDs (e.g., 00N_110E)")
    args = parser.parse_args(argv)

    run(args)


if __name__ == "__main__":
    main()

"""
python -m src.scripts.zonal_statistics.qaqc_drained_state \
  --model_version 0_6_0 \
  --run_date 20250807 \
  --interval_end_years 2020 2024 \
  --tile_ids 00N_110E \
  --cluster_name zonal_stats \
  --debug
"""