# -*- coding: utf-8 -*-
"""
Simplified SDPT rasterisation script (file-based; TypeCode).

What it does
------------
- Reads SDPT vector tiles from an S3 folder (default):
    s3://gfw2-data/plantations/sdpt_v3/sdpt_v3_vector_tiles/tiles_5d
- Expects attribute "TypeCode" with classes:
    Planted forest -> 1
    Tree crops     -> 2
  (Robust to either numeric {1,2} or string values; case-insensitive.)
- Rasterises to a 0.00025° grid (EPSG:4326), writing one partial GeoTIFF per chunk.

Outputs
-------
- Partial GeoTIFFs named: <src_basename>__<minx_miny_maxx_maxy>__sdpt.tif
  * default upload path: cn.datasets["sdpt"]["s3_processed_base"]/<px>_pixels/<YYYYMMDD>/
  * with --run_mode test: written locally to cn.datasets["sdpt"]["local_processed"]

Notes
-----
- Chunks default to 2°; change with --chunk_size.
- Will reproject layers to EPSG:4326 if needed.
- Uses module-level S3 bucket/prefix for inputs; override via CLI.
"""

from __future__ import annotations

import os
import sys
import re
import math
import struct
import logging
import argparse
import warnings
import posixpath
import gc
import tempfile
import subprocess
import shutil
import traceback
from typing import Optional, Tuple, Dict, List

import dask
import dask_geopandas as dgpd
from dask.distributed import Client, LocalCluster
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import shapely
from pyogrio import read_dataframe as _read_df
from pyogrio.errors import DataSourceError

# Our universal constants & utilities
import src.scripts.preprocessing.preprocessing_constants as cn
import src.scripts.preprocessing.utilities as uu
from src.scripts.utilities import universal_utilities as uutil

warnings.filterwarnings("ignore", "Geometry is in a geographic CRS.", UserWarning)

# ---------------------------------------------------------------------------
# Env tuning (safe defaults; can be overridden by shell env)
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", "shp,shx,dbf,prj,cpg")
os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")  # allow opening/creating .shx on the fly
os.environ.setdefault("DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL", "None")

# ---------------------------------------------------------------------------
# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Rasterization settings
RASTER_RES = 0.00025
RASTER_NODATA = 0
RASTER_DTYPE = np.uint8

# Grid alignment guard
SNAP_ORIGIN_X = -180.0
SNAP_ORIGIN_Y = -90.0


def _assert_bbox_on_grid(bbox, res=RASTER_RES, ox=SNAP_ORIGIN_X, oy=SNAP_ORIGIN_Y):
    minx, miny, maxx, maxy = map(float, bbox)
    def _ok(v, o):
        q = (v - o) / res
        return abs(q - round(q)) < 1e-6
    if not all((_ok(minx, ox), _ok(maxx, ox), _ok(miny, oy), _ok(maxy, oy))):
        raise ValueError(f"BBox {bbox} not aligned to {res} grid from origin ({ox},{oy}).")


# ────────────────────────────────────────────────────────────────
# Defaults for SDPT inputs (can override via CLI)
S3_BUCKET_RAW_DEFAULT = "gfw2-data"
S3_PREFIX_RAW_DEFAULT = "plantations/sdpt_v3/sdpt_v3_vector_tiles/tiles_5d"
ATTR_FIELD_DEFAULT = "TypeCode"

# Effective (mutable) globals filled in main()
S3_BUCKET_RAW = S3_BUCKET_RAW_DEFAULT
S3_PREFIX_RAW = S3_PREFIX_RAW_DEFAULT
ATTR_FIELD = ATTR_FIELD_DEFAULT

# File index: src_basename -> exact S3 key
BASENAME_TO_KEY: Dict[str, str] = {}


def _basename(key: str) -> str:
    return os.path.splitext(os.path.basename(key))[0]


def _vsis3_path_for_key(key: str) -> str:
    return f"/vsis3/{S3_BUCKET_RAW}/{key}"


def _s3_url_from_vsis3(vsis3_path: str) -> str:
    if vsis3_path.startswith("/vsis3/"):
        parts = vsis3_path.split("/", 3)
        return "s3://" + parts[2] + "/" + parts[3]
    return vsis3_path


def _shp_bbox_from_s3(bucket: str, key: str) -> Tuple[float, float, float, float]:
    """
    Read the shapefile header directly from S3 and return (minx, miny, maxx, maxy).
    Uses the ESRI SHP main header (bytes 36..68, little-endian doubles).
    """
    try:
        import s3fs
    except Exception as e:
        raise RuntimeError("s3fs is required to read shapefile header from S3") from e

    fs = s3fs.S3FileSystem(anon=False)
    with fs.open(f"{bucket}/{key}", "rb") as fh:
        header = fh.read(100)
    if len(header) < 100:
        raise DataSourceError("Shapefile header is truncated (<100 bytes)")
    # xmin, ymin, xmax, ymax
    minx, miny, maxx, maxy = struct.unpack("<4d", header[36:68])
    return (minx, miny, maxx, maxy)


def _snap_val_min(v: float, res: float, origin: float) -> float:
    return math.floor((v - origin) / res + 1e-12) * res + origin


def _snap_val_max(v: float, res: float, origin: float) -> float:
    return math.ceil((v - origin) / res - 1e-12) * res + origin


def _snap_bbox_to_grid(bbox: Tuple[float, float, float, float],
                       res: float = RASTER_RES,
                       ox: float = SNAP_ORIGIN_X,
                       oy: float = SNAP_ORIGIN_Y) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bbox
    minx_s = _snap_val_min(minx, res, ox)
    miny_s = _snap_val_min(miny, res, oy)
    maxx_s = _snap_val_max(maxx, res, ox)
    maxy_s = _snap_val_max(maxy, res, oy)
    return (minx_s, miny_s, maxx_s, maxy_s)


def _download_shapefile_sidecars_to_tmp(s3_url: str, tmpdir: str) -> str:
    """Download SHP+sidecars (if present) to tmpdir, return local .shp path."""
    try:
        import s3fs
    except Exception as e:
        raise RuntimeError("s3fs is required for on-the-fly shapefile repair") from e

    assert s3_url.startswith("s3://")
    _, rest = s3_url.split("s3://", 1)
    bucket, key = rest.split("/", 1)
    base_no_ext = key[:-4] if key.lower().endswith(".shp") else key

    fs = s3fs.S3FileSystem(anon=False)
    local_base = os.path.join(tmpdir, "tile")
    found_any = False
    for ext in [".shp", ".dbf", ".shx", ".prj", ".cpg"]:
        src = f"{bucket}/{base_no_ext}{ext}"
        dst = f"{local_base}{ext}"
        try:
            if fs.exists(src):
                fs.get(src, dst)
                found_any = True
        except Exception:
            pass

    shp_path = f"{local_base}.shp"
    if not (found_any and os.path.exists(shp_path)):
        raise DataSourceError(f"Missing .shp: {s3_url}")
    return shp_path


def _try_repair_and_read(vsis3_path: str, bbox: tuple):
    """
    Fallback path for corrupt/missing .shx or mismatched drivers:
    1) Try s3:// path directly.
    2) Download locally; rely on SHAPE_RESTORE_SHX=YES to regenerate .shx.
    3) If still failing and ogr2ogr is available, rewrite then read.
    Reads *all* columns (bbox-filtered) so we can pick ATTR_FIELD case-insensitively.
    """
    s3_url = _s3_url_from_vsis3(vsis3_path)

    # 1) Straight s3://
    try:
        return _read_df(s3_url, columns=None, bbox=bbox)
    except Exception as e1:
        logging.warning(f"Retry via s3:// failed for {s3_url}: {e1}")

    local_shp = None
    # 2) Materialize locally and let GDAL rebuild .shx
    with tempfile.TemporaryDirectory() as td:
        try:
            local_shp = _download_shapefile_sidecars_to_tmp(s3_url, td)
            return _read_df(local_shp, columns=None, bbox=bbox)
        except Exception as e2:
            logging.warning(f"Local open with SHAPE_RESTORE_SHX failed: {e2}")

        # 3) ogr2ogr rewrite only if we successfully downloaded something
        if local_shp and shutil.which("ogr2ogr"):
            repaired = os.path.join(td, "repaired.shp")
            try:
                subprocess.run(
                    ["ogr2ogr", "-skipfailures", "-f", "ESRI Shapefile", repaired, local_shp],
                    check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                return _read_df(repaired, columns=None, bbox=bbox)
            except Exception as e3:
                logging.error(f"ogr2ogr repair failed: {e3}")
                raise
        else:
            raise DataSourceError("ogr2ogr not applicable (no local .shp) or not found; cannot repair corrupt shapefile")


def _read_key_bbox(vsis3_path: str, bbox):
    """Stream features intersecting *bbox* with all columns (for robust field pick)."""
    bbox = tuple(bbox)
    try:
        return _read_df(vsis3_path, columns=None, bbox=bbox)
    except DataSourceError as e:
        logging.warning(f"pyogrio open failed for {vsis3_path}: {e} — attempting repair.")
        return _try_repair_and_read(vsis3_path, bbox)


def _normalize_column_lookup(gdf: gpd.GeoDataFrame, names: List[str]):
    """Return the actual column name in *gdf* matching one of *names* (case/space-insensitive), or None."""
    norm_map = {str(c).strip().lower(): c for c in gdf.columns}
    for n in names:
        key = str(n).strip().lower()
        if key in norm_map:
            return norm_map[key]
    return None


def _coerce_typecode_to_uint8(gdf: gpd.GeoDataFrame, attr_fallbacks=("TypeCode", "simpleType")):
    """
    Add a uint8 'raster_val' column according to:
      Planted forest -> 1
      Tree crops     -> 2
    Accepts either numeric {1,2} or string labels (case-insensitive).
    Drops rows that do not map to {1,2}.
    """
    col = _normalize_column_lookup(gdf, [ATTR_FIELD] + list(attr_fallbacks))
    if col is None:
        return gdf.assign(raster_val=pd.Series(dtype="uint8")).iloc[0:0]

    s = gdf[col]

    # Try numeric first
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any():
        s_num = s_num.where(s_num.isin([1, 2]))
        out = s_num.astype("UInt8")
    else:
        # String mapping
        s_str = s.astype("string").str.strip().str.lower()
        mapping = {
            "planted forest": 1,
            "planted_forest": 1,
            "planted forests": 1,
            "tree crops": 2,
            "tree_crops": 2,
            "tree crop": 2,
        }
        out = s_str.map(mapping).astype("UInt8")

    # Drop unmapped
    gdf = gdf.assign(raster_val=out)
    gdf = gdf.dropna(subset=["raster_val"])
    gdf["raster_val"] = gdf["raster_val"].astype("uint8", copy=False)
    return gdf


def list_input_shapefiles() -> List[str]:
    """Return all .shp keys under the raw S3 prefix."""
    prefix = S3_PREFIX_RAW
    return [
        k for k in uutil.list_s3_files(S3_BUCKET_RAW, prefix)
        if k.lower().endswith(".shp")
    ]


def initialize_file_index() -> None:
    """Populate BASENAME_TO_KEY by listing all .shp under the raw prefix."""
    global BASENAME_TO_KEY
    keys = list_input_shapefiles()
    if not keys:
        logging.warning(f"No shapefiles found under s3://{S3_BUCKET_RAW}/{S3_PREFIX_RAW}")
    BASENAME_TO_KEY = { _basename(k): k for k in keys }
    logging.info(f"Discovered {len(BASENAME_TO_KEY)} shapefiles under s3://{S3_BUCKET_RAW}/{S3_PREFIX_RAW}")


def rasterize_chunk_df(subset_gdf, bbox, out_base, run_mode):
    """Rasterize a GeoDataFrame subset directly in memory → partial GeoTIFF."""
    chunk_str = uutil.boundstr(bbox)
    chunk_px = uutil.calc_chunk_length_pixels(bbox)
    chunk_name = f"{out_base}__{chunk_str}__sdpt.tif"
    local_dir = cn.datasets["sdpt"]["local_processed"]
    uu.create_directory_if_not_exists(local_dir)
    out_tif = os.path.join(local_dir, chunk_name)

    s3_chunk = posixpath.join(
        cn.datasets["sdpt"]["s3_processed_base"],
        f"{chunk_px}_pixels",
        cn.today_date,
        chunk_name,
    )

    # Skip if this exact partial exists (prevents duplication)
    if run_mode == "default":
        if uutil.s3_file_exists(cn.s3_bucket_name, s3_chunk):
            logging.info(f"Partial TIF exists, skipping: s3://{cn.s3_bucket_name}/{s3_chunk}")
            return
    else:
        if os.path.exists(out_tif):
            logging.info(f"Local partial exists, skipping: {out_tif}")
            return

    # ── geometry hygiene ─────────────────────────────────────────
    raw_n = len(subset_gdf)
    subset_gdf = subset_gdf.dropna(subset=["geometry"]).copy()
    if hasattr(subset_gdf, "is_empty"):
        subset_gdf = subset_gdf[~subset_gdf.geometry.is_empty]
    if hasattr(subset_gdf, "is_valid"):
        inv_mask = ~subset_gdf.geometry.is_valid
        if inv_mask.any():
            try:
                subset_gdf.loc[inv_mask, "geometry"] = shapely.make_valid(
                    subset_gdf.loc[inv_mask, "geometry"].values
                )
            except Exception:
                subset_gdf.loc[inv_mask, "geometry"] = subset_gdf.loc[inv_mask, "geometry"].buffer(0)
    subset_gdf = subset_gdf.explode(index_parts=False, ignore_index=True)
    subset_gdf = subset_gdf[subset_gdf.geometry.notna()]
    if hasattr(subset_gdf, "is_empty"):
        subset_gdf = subset_gdf[~subset_gdf.geometry.is_empty]
    clean_n = len(subset_gdf)
    if clean_n == 0:
        logging.info(f"No valid shapes after cleaning in {bbox} (raw={raw_n}). Skipping.")
        return
    logging.info(f"QA: cleaned geoms in {bbox} raw={raw_n} -> clean={clean_n}")

    shapes = list(zip(subset_gdf.geometry.values, subset_gdf["raster_val"].values))
    if not shapes:
        logging.info(f"No shapes to rasterize in {bbox}, skipping.")
        return

    # Grid alignment guard (cheap assert)
    _assert_bbox_on_grid(bbox)

    minx, miny, maxx, maxy = bbox
    width = int(round((maxx - minx) / RASTER_RES))
    height = int(round((maxy - miny) / RASTER_RES))
    transform = from_origin(minx, maxy, RASTER_RES, RASTER_RES)

    burned = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=RASTER_NODATA,
        dtype=RASTER_DTYPE,
        all_touched=False,
    )

    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": RASTER_DTYPE,
        "crs": "EPSG:4326",
        "transform": transform,
        "tiled": True,
        "compress": "DEFLATE",
        "nodata": RASTER_NODATA,
    }
    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(burned, 1)

    if run_mode == "default":
        logging.info(f"Uploading partial TIF => s3://{cn.s3_bucket_name}/{s3_chunk}")
        uutil.upload_file_to_s3(out_tif, cn.s3_bucket_name, s3_chunk)
        os.remove(out_tif)
    else:
        logging.info(f"Test mode => partial TIF => {out_tif} retained locally.")

    del burned
    gc.collect()


@dask.delayed
def rasterize_bbox_task(shp_key: str, bbox: tuple, run_mode: str):
    """Robust per-bbox task; never raises to scheduler (logs and returns)."""
    vsis3_path = _vsis3_path_for_key(shp_key)
    out_base = _basename(shp_key)

    try:
        gdf = _read_key_bbox(vsis3_path, bbox)
    except Exception as e:
        logging.error(f"[{out_base} {bbox}] read failed: {e}\n{traceback.format_exc()}")
        return

    if gdf is None or gdf.empty:
        logging.info(f"No geometries in {bbox} for {out_base} – skipped.")
        return

    # Reproject to EPSG:4326 if needed
    try:
        if getattr(gdf, "crs", None) and str(gdf.crs).lower() not in ("epsg:4326",):
            gdf = gdf.to_crs("EPSG:4326")
    except Exception as e:
        logging.warning(f"CRS conversion failed; assuming EPSG:4326. Error: {e}")

    # Map attribute → raster_val
    gdf = _coerce_typecode_to_uint8(gdf)
    if gdf.empty:
        logging.info(f"All geometries mapped to None in {bbox} – skipped.")
        return

    try:
        rasterize_chunk_df(gdf, bbox, out_base, run_mode)
    except Exception as e:
        logging.error(f"[{out_base} {bbox}] rasterize failed: {e}\n{traceback.format_exc()}")
        return


def process_key(shp_key: str, chunk_size=1.0, run_mode="default"):
    """Process one shapefile key in chunks."""
    out_base = _basename(shp_key)
    # Get bbox from header, snap to grid
    tb = _shp_bbox_from_s3(S3_BUCKET_RAW, shp_key)
    snapped = _snap_bbox_to_grid(tb, RASTER_RES, SNAP_ORIGIN_X, SNAP_ORIGIN_Y)

    logging.info(f"Processing {out_base} in ~{chunk_size}° chunks; bbox={snapped}")
    chunk_bboxes = uutil.get_chunk_bounds(list(snapped), chunk_size)

    # NEW: safe-resume preflight — schedule only missing outputs
    to_schedule = []
    skipped = 0
    for bb in chunk_bboxes:
        chunk_str = uutil.boundstr(bb)
        chunk_px = uutil.calc_chunk_length_pixels(bb)
        chunk_name = f"{out_base}__{chunk_str}__sdpt.tif"
        local_dir = cn.datasets["sdpt"]["local_processed"]
        out_tif = os.path.join(local_dir, chunk_name)
        s3_chunk = posixpath.join(
            cn.datasets["sdpt"]["s3_processed_base"],
            f"{chunk_px}_pixels",
            cn.today_date,
            chunk_name,
        )
        exists = (uutil.s3_file_exists(cn.s3_bucket_name, s3_chunk)
                  if run_mode == "default" else os.path.exists(out_tif))
        if exists:
            skipped += 1
            continue
        to_schedule.append(rasterize_bbox_task(shp_key, tuple(bb), run_mode))

    total = len(chunk_bboxes)
    logging.info(f"{out_base}: total={total}, skipped_existing={skipped}, scheduled={len(to_schedule)}")
    return to_schedule


def process_all_files(chunk_size=1.0, run_mode="default", src_basename: Optional[str] = None):
    """Process every SDPT shapefile sequentially (or a specific one by basename)."""
    if src_basename:
        if src_basename not in BASENAME_TO_KEY:
            logging.error(f"basename '{src_basename}' not found under inputs s3://{S3_BUCKET_RAW}/{S3_PREFIX_RAW}")
            return
        keys = [BASENAME_TO_KEY[src_basename]]
    else:
        keys = [BASENAME_TO_KEY[b] for b in sorted(BASENAME_TO_KEY.keys())]

    for shp_key in keys:
        tasks = process_key(shp_key, chunk_size, run_mode)
        if tasks:
            logging.info(f"Computing {len(tasks)} chunk tasks for => {os.path.basename(shp_key)} ...")
            try:
                dask.compute(*tasks)
            except Exception as e:
                logging.error(f"File {shp_key} encountered an error: {e}\n{traceback.format_exc()}")
        else:
            logging.info(f"All chunks already exist for {os.path.basename(shp_key)} — nothing to do.")


def main(
    src_basename: Optional[str] = None,
    chunk_size: float = 1.0,
    run_mode: str = "default",
    client: str = "local",
    smoke_test: bool = False,
    s3_bucket_raw: str = S3_BUCKET_RAW_DEFAULT,
    s3_prefix_raw: str = S3_PREFIX_RAW_DEFAULT,
    attr_field: str = ATTR_FIELD_DEFAULT,
    show_keys: int = 0,
):
    """
    Processes all shapefiles under the S3 prefix (or a single file via --src_basename).
    """
    global S3_BUCKET_RAW, S3_PREFIX_RAW, ATTR_FIELD
    S3_BUCKET_RAW = s3_bucket_raw
    S3_PREFIX_RAW = s3_prefix_raw.rstrip("/")
    ATTR_FIELD = attr_field

    logging.info(
        "SDPT rasterizer config: "
        f"inputs s3://{S3_BUCKET_RAW}/{S3_PREFIX_RAW} | attr_field='{ATTR_FIELD}' | "
        f"outputs base s3://{cn.s3_bucket_name}/{cn.datasets['sdpt']['s3_processed_base']}"
    )

    # Build file index (basename -> key)
    initialize_file_index()

    if show_keys and BASENAME_TO_KEY:
        head = sorted(BASENAME_TO_KEY.keys())[: int(show_keys)]
        logging.info("First %d discovered basenames:", len(head))
        for bn in head:
            print(bn, "->", BASENAME_TO_KEY[bn])
        return

    # Smoke test BEFORE connecting to any cluster
    if smoke_test:
        if not BASENAME_TO_KEY:
            logging.error("No SDPT shapefiles found on S3.")
            return
        test_key = next(iter(BASENAME_TO_KEY.values()))
        tb = _shp_bbox_from_s3(S3_BUCKET_RAW, test_key)
        bbx = _snap_bbox_to_grid(tb)
        # small 0.5° probe on SW corner
        minx, miny, maxx, maxy = bbx
        bbx_probe = (minx, miny, min(minx + 0.5, maxx), min(miny + 0.5, maxy))
        _ = _read_key_bbox(_vsis3_path_for_key(test_key), bbx_probe)
        logging.info(f"pyogrio smoke test OK for {os.path.basename(test_key)} bbox {bbx_probe}")
        return

    # Connect to cluster only when computing
    run_local_flag = client == "local"
    n_workers = int(os.getenv("SDPT_WORKERS", "12"))
    worker_mem = os.getenv("SDPT_WORKER_MEMORY", "16GiB")
    cluster, dask_client, run_local_flag = uutil.connect_to_cluster(
        cluster_name="sdpt_rasterize",
        n_workers=n_workers,
        region="us-east-1",
        run_local=run_local_flag,
        worker_memory=worker_mem,
    )
    if run_local_flag:
        cluster = LocalCluster()
        dask_client = Client(cluster)
        logging.info("Running on a local Dask cluster.")
    else:
        logging.info(f"Coiled cluster => {cluster.name}")

    try:
        process_all_files(chunk_size, run_mode, src_basename)
    finally:
        if 'dask_client' in locals() and dask_client:
            dask_client.close()
            logging.info("Dask client closed.")
        if 'cluster' in locals() and cluster:
            cluster.close()
            logging.info("Coiled cluster closed.")

    logging.info("All chunk tasks completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SDPT chunk-based rasterizer (TypeCode -> {1,2}); partial TIFs stored under s3_processed_base/<px>/YYYYMMDD."
    )
    parser.add_argument("--src_basename", type=str,
                        help="Process a single shapefile by basename (no extension). Omit to process all files.")
    parser.add_argument("--chunk_size", type=float, default=1.0, help="Chunk size (deg).")
    parser.add_argument("--run_mode", type=str, choices=["default", "test"], default="default",
                        help="default => partial TIF => S3, test => local partial TIFs.")
    parser.add_argument("--client", type=str, choices=["local", "coiled"], default="local",
                        help="Dask client type (local or coiled).")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run a quick pyogrio S3+bbox read test (0.5° probe) and exit.")

    # Input controls
    parser.add_argument("--s3_bucket_raw", type=str, default=S3_BUCKET_RAW_DEFAULT,
                        help="S3 bucket for vector tiles (default: gfw2-data).")
    parser.add_argument("--s3_prefix_raw", type=str, default=S3_PREFIX_RAW_DEFAULT,
                        help="S3 prefix for vector tiles (default: plantations/sdpt_v3/sdpt_v3_vector_tiles/tiles_5d).")
    parser.add_argument("--attr_field", type=str, default=ATTR_FIELD_DEFAULT,
                        help="Attribute field to rasterize (default: TypeCode).")
    parser.add_argument("--show_keys", type=int, default=0,
                        help="Print the first N discovered file basenames and exit (useful for debugging naming).")

    args = parser.parse_args()

    if not any(sys.argv[1:]):
        logging.info("No CLI => processing all tiles locally in test mode for demonstration.")
        main(src_basename=None, chunk_size=1.0, run_mode="test", client="local")
    else:
        main(src_basename=args.src_basename,
             chunk_size=args.chunk_size,
             run_mode=args.run_mode,
             client=args.client,
             smoke_test=args.smoke_test,
             s3_bucket_raw=args.s3_bucket_raw,
             s3_prefix_raw=args.s3_prefix_raw,
             attr_field=args.attr_field,
             show_keys=args.show_keys)
