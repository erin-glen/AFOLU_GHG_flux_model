"""
sdpt_rasterize_attributes.py

Rasterize SDPT plantation attributes in small chunks using Dask, matching the
'simplified/testing' rasterization pipeline in all operational aspects except
for the classification schema (which remains advanced: CSV + heuristic fallback).

Key behavior aligned with the simplified variant:
  - Chunk-by-bbox streaming reads from S3 (pyogrio bbox filter).
  - Robust SHX/sidecar repair path (s3fs + SHAPE_RESTORE_SHX + optional ogr2ogr).
  - Geometry QA: make_valid -> explode -> re-filter.
  - Strict grid alignment guard (origin -180/-90, 0.00025° res).
  - In-memory rasterization with rasterio, all_touched=False, uint8, nodata=0.
  - Idempotent partial writes (skip if exists), 'default' => upload to S3, 'test' => keep local.
  - Smoke test option to verify S3+bbox pyogrio read before connecting to a cluster.
  - Cluster connect via uutil.connect_to_cluster with env-tunable sizing; local fallback.

Classification differences retained:
  - Uses advanced species remap CSV (vernacName -> rotation_code) if available.
  - Falls back to create_remapping.classify + ROTATION_CLASS_CODES.
  - Oil palm detection for tree crops (simpleName contains 'oil palm').

Usage examples:
  python -m src.scripts.preprocessing.sdpt.sdpt_rasterize_attributes --client local --run_mode test --smoke_test
  python -m src.scripts.preprocessing.sdpt.sdpt_rasterize_attributes --tile_id 00N_110E --chunk_bounds "112,-4,114,-2" --run_mode test
"""

import os
import syscon
import logging
import argparse
import warnings
import posixpath
import gc
import tempfile
import subprocess
import shutil
import traceback

import dask
from dask.distributed import Client, LocalCluster
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import shapely
import pandas as pd
from pyogrio import read_dataframe as _read_df
from pyogrio.errors import DataSourceError

# Our universal constants & utilities
import src.scripts.preprocessing.preprocessing_constants as cn
import src.scripts.preprocessing.utilities as uu
from src.scripts.utilities import universal_utilities as uutil

# Advanced reclassification support
from .create_remapping import classify as fallback_classify
from .create_remapping import ROTATION_CLASS_CODES

warnings.filterwarnings("ignore", "Geometry is in a geographic CRS.", UserWarning)

# ---------------------------------------------------------------------------
# Env tuning (safe defaults; can be overridden by shell env)
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", "shp,shx,dbf,prj,cpg")
os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")  # allow opening/creating .shx on the fly
# (Helps when you create new clusters; existing schedulers may ignore.)
os.environ.setdefault("DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL", "None")

# ---------------------------------------------------------------------------
# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Reclassification CSV in S3 (vernacName -> rotation_code)
ADVANCED_REMAP_S3 = (
    "climate/AFOLU_flux_model/organic_soils/inputs/raw/plantations/sdpt/remapping_tables/advanced_remapping.csv"
)

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

# Columns we *prefer* to request; we will gracefully fall back if missing
_READ_COLUMNS = ["geometry", "simpleType", "simpleName", "vernacName"]

# ────────────────────────────────────────────────────────────────
# Helper utilities (mirroring the simplified/testing variant)
def _tile_shp_path(tile_id: str) -> str:
    """Return the /vsis3/ path to a tile_<id>.shp on S3."""
    return (
        f"/vsis3/{cn.s3_bucket_name}/"
        f"{cn.datasets['sdpt']['s3_raw']}/tile_{tile_id}.shp"
    )

def _s3_url_from_vsis3(vsis3_path: str) -> str:
    if vsis3_path.startswith("/vsis3/"):
        parts = vsis3_path.split("/", 3)
        return "s3://" + parts[2] + "/" + parts[3]
    return vsis3_path

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
    for ext in [".shp", ".dbf", ".shx", ".prj", ".cpg"]:
        src = f"{bucket}/{base_no_ext}{ext}"
        dst = f"{local_base}{ext}"
        try:
            if fs.exists(src):
                fs.get(src, dst)
        except Exception:
            # best-effort; missing files are ok
            pass

    shp_path = f"{local_base}.shp"
    if not os.path.exists(shp_path):
        raise DataSourceError(f"Missing .shp: {s3_url}")
    return shp_path

def _safe_read_columns(path, bbox):
    """
    Try reading with our preferred column subset; if layer lacks some, fall back
    to reading all columns for that bbox.
    """
    try:
        return _read_df(path, columns=_READ_COLUMNS, bbox=bbox)
    except ValueError:
        logging.warning(f"{path}: requested columns missing; falling back to all columns for bbox={bbox}.")
        return _read_df(path, bbox=bbox)

def _try_repair_and_read(tile_path: str, bbox: tuple):
    """
    Fallback path for corrupt/missing .shx:
    1) Try s3:// path directly (fsspec).
    2) Download locally; rely on SHAPE_RESTORE_SHX=YES to regenerate .shx.
    3) If still failing and ogr2ogr is available, rewrite shapefile and read.
    """
    s3_url = _s3_url_from_vsis3(tile_path)

    # (1) Retry via s3:// (some environments prefer fsspec)
    try:
        return _safe_read_columns(s3_url, bbox)
    except Exception as e1:
        logging.warning(f"Retry via s3:// failed for {s3_url}: {e1}")

    # (2) Materialize locally and let GDAL rebuild .shx
    with tempfile.TemporaryDirectory() as td:
        try:
            local_shp = _download_shapefile_sidecars_to_tmp(s3_url, td)
            return _safe_read_columns(local_shp, bbox)
        except Exception as e2:
            logging.warning(f"Local open with SHAPE_RESTORE_SHX failed: {e2}")

        # (3) If available, use ogr2ogr to rewrite (= regenerate .shx) then read
        if shutil.which("ogr2ogr"):
            repaired = os.path.join(td, "repaired.shp")
            try:
                subprocess.run(
                    ["ogr2ogr", "-skipfailures", "-f", "ESRI Shapefile", repaired, local_shp],
                    check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                return _safe_read_columns(repaired, bbox)
            except Exception as e3:
                logging.error(f"ogr2ogr repair failed: {e3}")
                raise
        else:
            raise DataSourceError("ogr2ogr not found; cannot repair corrupt shapefile")

def _read_tile_bbox(tile_path: str, bbox):
    """Stream only the features intersecting *bbox*; robust to corrupt .shx."""
    bbox = tuple(bbox)
    try:
        return _safe_read_columns(tile_path, bbox)
    except DataSourceError as e:
        logging.warning(f"pyogrio open failed for {tile_path}: {e} — attempting repair.")
        return _try_repair_and_read(tile_path, bbox)

def list_sdpt_shapefiles():
    """Return the list of SDPT shapefiles stored on S3."""
    prefix = cn.datasets["sdpt"]["s3_raw"]
    return [
        k
        for k in uutil.list_s3_files(cn.s3_bucket_name, prefix)
        if k.lower().endswith(".shp")
    ]

# ────────────────────────────────────────────────────────────────
# Advanced classification (schema retained from the original attributes script)
def load_species_reclassification():
    """Return a mapping of vernacular names to numeric rotation codes.

    Looks for ``advanced_remapping.csv`` in ``cn.local_temp_dir`` and attempts to
    download it from S3 if missing. Returns {} if not available.
    """
    local_csv = os.path.join(cn.local_temp_dir, os.path.basename(ADVANCED_REMAP_S3))

    if not os.path.exists(local_csv):
        try:
            uutil.download_file_from_s3(ADVANCED_REMAP_S3, local_csv, cn.s3_bucket_name)
            logging.info(f"Downloaded CSV => {local_csv}")
        except Exception as exc:  # pragma: no cover - network errors
            logging.warning(f"Failed to download CSV from S3: {exc}")
    else:
        logging.info(f"Local CSV already exists => {local_csv}, skipping download.")

    if not os.path.exists(local_csv):
        logging.warning("Advanced remapping CSV not found; falling back to classification logic.")
        return {}

    df = pd.read_csv(local_csv)
    try:
        mapping = dict(zip(df["vernacName"].str.strip(), df["rotation_code"]))
    except KeyError:
        logging.warning("advanced_remapping.csv missing expected columns")
        mapping = {}

    logging.info(f"Loaded {len(mapping)} species from advanced remapping CSV.")
    return mapping

def assign_rotation_codes(gdf, species_map):
    """
    Vector-aware assignment with targeted row-wise fallback:
      1) Use species_map on vernacName (exact, strip).
      2) For tree crops, set oil_palm when simpleName contains 'oil palm'.
      3) For planted forest, apply fallback_classify(row) then map to ROTATION_CLASS_CODES.
      4) Unknowns => NaN (to be dropped).
    """
    if gdf is None or gdf.empty:
        return gdf

    # Ensure columns exist
    for col in ["simpleType", "simpleName", "vernacName"]:
        if col not in gdf.columns:
            gdf[col] = None

    st = gdf["simpleType"].astype("string").str.strip().str.lower()
    sn = gdf["simpleName"].astype("string").str.strip().str.lower()
    vn = gdf["vernacName"].astype("string").str.strip()

    # 1) vernacName → code
    codes = vn.map(species_map) if species_map else pd.Series(np.nan, index=gdf.index, dtype="float64")

    # 2) tree crops → oil palm detection where still NaN
    is_tree = st == "tree crops"
    oil_mask = is_tree & codes.isna() & sn.str.contains("oil palm", na=False)
    oil_code = ROTATION_CLASS_CODES.get("oil_palm")
    if oil_code is not None and oil_mask.any():
        codes.loc[oil_mask] = oil_code

    # 3) planted forest → heuristic classify for rows still NaN
    pf_mask = (st == "planted forest") & codes.isna()
    if pf_mask.any():
        # Apply only on subset to reduce overhead
        pf_idx = gdf.index[pf_mask]
        sub = gdf.loc[pf_idx]
        cls = sub.apply(lambda r: fallback_classify(r), axis=1)
        mapped = cls.map(ROTATION_CLASS_CODES)
        # Assign back
        codes.loc[pf_idx] = mapped

    gdf = gdf.assign(raster_val=codes)
    gdf = gdf.dropna(subset=["raster_val"])
    if not gdf.empty:
        gdf["raster_val"] = gdf["raster_val"].astype("uint8", copy=False)
    return gdf[["geometry", "raster_val"]]

# ────────────────────────────────────────────────────────────────
# Rasterization (mirrors simplified/testing variant: in-memory, all_touched=False)
def rasterize_chunk_df(subset_gdf, bbox, tile_id, run_mode):
    """Rasterize a GeoDataFrame subset directly in memory → partial GeoTIFF."""

    chunk_str = uutil.boundstr(bbox)
    chunk_px = uutil.calc_chunk_length_pixels(bbox)
    chunk_name = f"{tile_id}__{chunk_str}__sdpt.tif"
    local_dir = cn.datasets["sdpt"]["local_processed"]
    uu.create_directory_if_not_exists(local_dir)
    out_tif = os.path.join(local_dir, chunk_name)

    s3_chunk = posixpath.join(
        cn.datasets["sdpt"]["s3_processed_base"],
        f"{chunk_px}_pixels",
        cn.today_date,
        chunk_name,
    )

    # Skip if this exact partial exists
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
        all_touched=False,  # match simplified/testing
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

# ────────────────────────────────────────────────────────────────
# Single self-contained task (mirrors simplified/testing)
@dask.delayed
def rasterize_bbox_task(tile_id: str, bbox: tuple, run_mode: str, species_map: dict):
    """Robust per-bbox task; never raises to scheduler (logs and returns)."""
    try:
        tile_path = _tile_shp_path(tile_id)
        gdf = _read_tile_bbox(tile_path, bbox)
    except Exception as e:
        logging.error(f"[{tile_id} {bbox}] read failed: {e}\n{traceback.format_exc()}")
        return

    if gdf is None or gdf.empty:
        logging.info(f"No geometries in {bbox} for tile {tile_id} – skipped.")
        return

    # Classification to rotation codes (uint8); drop unknowns
    try:
        gdf = assign_rotation_codes(gdf, species_map)
    except Exception as e:
        logging.error(f"[{tile_id} {bbox}] classification failed: {e}\n{traceback.format_exc()}")
        return

    if gdf is None or gdf.empty:
        logging.info(f"All geometries mapped to None in {bbox} – skipped.")
        return

    try:
        rasterize_chunk_df(gdf, bbox, tile_id, run_mode)
    except Exception as e:
        logging.error(f"[{tile_id} {bbox}] rasterize failed: {e}\n{traceback.format_exc()}")
        return

# ────────────────────────────────────────────────────────────────
def process_tile(tile_id, species_map, chunk_size=2.0, run_mode="default"):
    logging.info(f"Processing tile {tile_id} in ~{chunk_size}° chunks")
    minx, miny, maxx, maxy = uutil.get_10x10_tile_bounds(tile_id)
    chunk_bboxes = uutil.get_chunk_bounds([minx, miny, maxx, maxy], chunk_size)
    return [rasterize_bbox_task(tile_id, tuple(bb), run_mode, species_map) for bb in chunk_bboxes]

def process_tile_with_bounds(tile_id, chunk_bounds, species_map, run_mode="default"):
    if isinstance(chunk_bounds, str):
        chunk_bounds = tuple(map(float, chunk_bounds.split(',')))
    return [rasterize_bbox_task(tile_id, chunk_bounds, run_mode, species_map)]

def process_all_tiles(species_map, chunk_size=2.0, run_mode="default"):
    """Process every SDPT tile sequentially to avoid memory blowout."""
    for shp_key in list_sdpt_shapefiles():
        tile_id = os.path.basename(shp_key)[len("tile_") : -4]
        tasks = process_tile(tile_id, species_map, chunk_size, run_mode)
        if tasks:
            logging.info(f"Computing {len(tasks)} chunk tasks for tile => {tile_id} ...")
            try:
                dask.compute(*tasks)
            except Exception as e:
                # Should be rare due to inside-task catching; continue to next tile
                logging.error(f"Tile {tile_id} encountered an error: {e}\n{traceback.format_exc()}")

# ────────────────────────────────────────────────────────────────
def main(
    tile_id=None,
    chunk_size=2.0,
    chunk_bounds=None,
    run_mode="default",
    client="local",
    smoke_test=False,
):
    """
    If chunk_bounds is provided => only process that bounding box.
    Otherwise => chunk the entire 10x10 tile in N sub-chunks.
    """
    logging.info(f"SDPT chunk-based script => base S3 path {cn.datasets['sdpt']['s3_processed_base']}")

    # Run smoke test BEFORE connecting to any cluster to avoid spurious heartbeats
    if smoke_test:
        shapefiles = list_sdpt_shapefiles()
        if not shapefiles:
            logging.error("No SDPT shapefiles found on S3.")
            return
        test_tile = os.path.basename(shapefiles[0])[len("tile_"):-4]
        bbx = (-180, -90, -179.5, -89.5)
        _ = _read_tile_bbox(_tile_shp_path(test_tile), bbx)
        logging.info(f"pyogrio smoke test OK for tile {test_tile} bbox {bbx}")
        return

    # Connect to cluster only when computing (mirrors simplified/testing)
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

    # Load species reclassification once
    species_map = load_species_reclassification()

    try:
        if tile_id:
            if chunk_bounds:
                logging.info(f"Processing tile => {tile_id}, chunk bounds => {chunk_bounds}")
                tasks = process_tile_with_bounds(tile_id, chunk_bounds, species_map, run_mode)
            else:
                tasks = process_tile(tile_id, species_map, chunk_size, run_mode)

            logging.info(f"Computing {len(tasks)} chunk tasks ...")
            dask.compute(*tasks)
        else:
            logging.info("No tile_id provided => processing all tiles.")
            process_all_tiles(species_map, chunk_size, run_mode)
    finally:
        if dask_client:
            dask_client.close()
            logging.info("Dask client closed.")
        if cluster:
            cluster.close()
            logging.info("Coiled cluster closed.")

    logging.info("All chunk tasks completed successfully.")

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SDPT chunk-based script => partial TIFs stored under s3_processed_base/<px>/YYYYMMDD."
    )
    parser.add_argument("--tile_id", type=str, help="Tile ID (e.g. 00N_110E). Omit to process all tiles.")
    parser.add_argument("--chunk_size", type=float, default=2.0, help="Chunk size (deg).")
    parser.add_argument("--chunk_bounds", type=str, help='Optional single bounding box "min_x,min_y,max_x,max_y" for quick testing.')
    parser.add_argument("--run_mode", type=str, choices=["default", "test"], default="default",
                        help="default => partial TIF => S3, test => local partial TIFs.")
    parser.add_argument("--client", type=str, choices=["local", "coiled"], default="local",
                        help="Dask client type (local or coiled).")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick pyogrio S3+bbox read test and exit.")

    args = parser.parse_args()

    if not any(sys.argv[1:]):
        logging.info("No CLI => processing all tiles locally in test mode for demonstration.")
        main(tile_id=None, chunk_size=2.0, chunk_bounds=None, run_mode="test", client="local")
    else:
        main(tile_id=args.tile_id,
             chunk_size=args.chunk_size,
             chunk_bounds=args.chunk_bounds,
             run_mode=args.run_mode,
             client=args.client,
             smoke_test=args.smoke_test)
