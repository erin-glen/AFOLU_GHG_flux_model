#!/usr/bin/env python
"""
Batch Reproject (EPSG:3395) – OSM roads, canals, GRIP roads, peat tiles,
   and now the 1 km union peat mask.

• Skips tiles already projected (_EPSG3395 suffix, or plain name if you prefer).
• Uses a Coiled / local Dask cluster only when tasks are queued.
"""

import logging, warnings, os, subprocess, posixpath, dask, geopandas as gpd
from dask.distributed import Client, LocalCluster
import src.scripts.preprocessing.preprocessing_constants as cn
import src.scripts.preprocessing.utilities as uu

# ---------- logging & warnings ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore", "Geometry is in a geographic CRS", UserWarning)

# ---------- S3 listing helpers ----------
def list_vector_tiles(prefix, bucket=cn.s3_bucket_name):
    return [k for k in uu.list_s3_files(bucket, prefix) if k.lower().endswith(".shp")]

def list_raster_tiles(prefix, bucket=cn.s3_bucket_name, ext=".tif"):
    return [k for k in uu.list_s3_files(bucket, prefix) if k.lower().endswith(ext)]

# ---------- “already done?” helpers ----------
def _vector_done(layer, dst_prefix, bucket=cn.s3_bucket_name):
    return uu.s3_file_exists(bucket, posixpath.join(dst_prefix, f"{layer}.shp"))

def _raster_done(base, dst_prefix, bucket=cn.s3_bucket_name):
    return uu.s3_file_exists(bucket, posixpath.join(dst_prefix, f"{base}.tif"))

# ---------- delayed reprojection tasks ----------
@dask.delayed
def reproject_vector_shapefile(src_key, dst_prefix, target_crs="EPSG:3395", bucket=cn.s3_bucket_name):
    layer = os.path.splitext(os.path.basename(src_key))[0]
    if _vector_done(layer, dst_prefix, bucket):
        logging.debug(f"[SKIP] {layer}.shp already exists")
        return

    logging.info(f"Reprojecting vector {src_key} => {target_crs}")
    gdf = gpd.read_file(f"/vsis3/{bucket}/{src_key}")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)

    tmp_dir = posixpath.join(cn.local_temp_dir, "batch_reproj_vec")
    os.makedirs(tmp_dir, exist_ok=True)
    local_shp = posixpath.join(tmp_dir, f"{layer}.shp")
    gdf.to_file(local_shp)

    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        lf  = local_shp.replace(".shp", ext)
        key = posixpath.join(dst_prefix, f"{layer}{ext}")
        uu.upload_file_to_s3(lf, bucket, key)
        os.remove(lf)

@dask.delayed
def reproject_raster_tile(src_key, dst_prefix, target_crs="EPSG:3395", bucket=cn.s3_bucket_name):
    base = os.path.splitext(os.path.basename(src_key))[0]
    if _raster_done(base, dst_prefix, bucket):
        logging.debug(f"[SKIP] {base}.tif already exists")
        return

    logging.info(f"Reprojecting raster {src_key} => {target_crs}")
    tmp_dir = posixpath.join(cn.local_temp_dir, "batch_reproj_ras")
    os.makedirs(tmp_dir, exist_ok=True)
    local_in  = posixpath.join(tmp_dir, f"{base}_in.tif")
    local_out = posixpath.join(tmp_dir, f"{base}.tif")

    uu.download_file_from_s3(src_key, local_in, bucket)
    subprocess.run([
        "gdalwarp", "-t_srs", target_crs, "-r", "near",
        "-co", "COMPRESS=LZW", "-co", "TILED=YES", "-overwrite",
        local_in, local_out
    ], check=True)

    uu.upload_file_to_s3(local_out, bucket, posixpath.join(dst_prefix, f"{base}.tif"))
    os.remove(local_in)
    os.remove(local_out)

# ---------- tasks for OSM / GRIP / Peat (previous) ----------
def tasks_osm_roads():
    cfg = cn.datasets["osm"]["roads"]
    pending = []
    for raw_key in list_vector_tiles(cfg["s3_raw"]):
        layer = os.path.splitext(os.path.basename(raw_key))[0]
        if not _vector_done(layer, cfg["s3_projected"]):
            pending.append(reproject_vector_shapefile(raw_key, cfg["s3_projected"]))
    logging.info(f"[OSM Roads] queued {len(pending)} tasks")
    return pending

def tasks_osm_canals():
    cfg = cn.datasets["osm"]["canals"]
    pending = []
    for raw_key in list_vector_tiles(cfg["s3_raw"]):
        layer = os.path.splitext(os.path.basename(raw_key))[0]
        if not _vector_done(layer, cfg["s3_projected"]):
            pending.append(reproject_vector_shapefile(raw_key, cfg["s3_projected"]))
    logging.info(f"[OSM Canals] queued {len(pending)} tasks")
    return pending

def tasks_grip_roads():
    cfg = cn.datasets["grip"]["roads"]
    pending = []
    for raw_key in list_vector_tiles(cfg["s3_raw"]):
        layer = os.path.splitext(os.path.basename(raw_key))[0]
        if not _vector_done(layer, cfg["s3_projected"]):
            pending.append(reproject_vector_shapefile(raw_key, cfg["s3_projected"]))
    logging.info(f"[GRIP Roads] queued {len(pending)} tasks")
    return pending

def tasks_peat():
    # Reproject the “1km” raw peat tiles => e.g. if we have them
    pending = []
    for raw_key in list_raster_tiles(cn.peat_tiles_prefix_1km):
        base = os.path.splitext(os.path.basename(raw_key))[0]
        if not _raster_done(base, cn.peat_tiles_prefix_1km_3395):
            pending.append(reproject_raster_tile(raw_key, cn.peat_tiles_prefix_1km_3395))
    logging.info(f"[Peat] queued {len(pending)} tasks")
    return pending

# ---------- tasks for Union 1km -----------
def tasks_peat_union_1km():
    """
    Reprojects the newly created 1 km union peat mask.
    We'll assume its source is cn.datasets["peat"]["union_mask"]["1km"]
    and target is something like cn.datasets["peat"]["union_mask"]["1km_3395"]
    which you define in constants_and_names.py.
    """
    src_prefix = cn.datasets["peat"]["union_mask"]["1km"]
    dst_prefix = cn.datasets["peat"]["union_mask"]["1km_3395"]  # must exist
    pending = []
    for raw_key in list_raster_tiles(src_prefix):
        base = os.path.splitext(os.path.basename(raw_key))[0]
        # e.g. "00N_010E_union_mask_1km" => reprojected "00N_010E_union_mask_1km.tif"
        if not _raster_done(base, dst_prefix):
            pending.append(reproject_raster_tile(raw_key, dst_prefix))
    logging.info(f"[Peat Union 1km => 3395] queued {len(pending)} tasks")
    return pending

# ---------- main -----------
def main(do_grip=True, do_osm_roads=True, do_osm_canals=True,
         do_peat=False, do_peat_union_1km=False, client_type="local"):
    logging.info("Batch reproject starting…")

    # build tasks *before* starting Dask
    tasks = []
    if do_grip:            tasks += tasks_grip_roads()
    if do_osm_roads:       tasks += tasks_osm_roads()
    if do_osm_canals:      tasks += tasks_osm_canals()
    if do_peat:            tasks += tasks_peat()
    if do_peat_union_1km:  tasks += tasks_peat_union_1km()

    if not tasks:
        logging.info("All requested outputs exist – nothing to do. Exiting.")
        return

    # start cluster only when needed
    if client_type == "coiled":
        client, cluster = uu.setup_coiled_cluster()
        logging.info(f"Using Coiled cluster: {cluster.name}")
    else:
        cluster = LocalCluster()
        client = Client(cluster)
        logging.info("Using local Dask cluster")

    try:
        logging.info(f"Running {len(tasks)} tasks…")
        dask.compute(*tasks)
    finally:
        client.close()
        if client_type == "coiled":
            cluster.close()
        logging.info("Batch reproject finished")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Reproject raw OSM / GRIP vectors, peat tiles, and union 1km mask to EPSG:3395")
    p.add_argument("--grip",             action="store_true", help="Process GRIP roads")
    p.add_argument("--osm_roads",        action="store_true", help="Process OSM roads")
    p.add_argument("--osm_canals",       action="store_true", help="Process OSM canals")
    p.add_argument("--peat",             action="store_true", help="Process peat tiles at 1km")
    p.add_argument("--peat_union_1km",   action="store_true", help="Process union 1km peat mask")
    p.add_argument("--client", choices=["local","coiled"], default="local")
    a = p.parse_args()

    main(do_grip=a.grip, do_osm_roads=a.osm_roads, do_osm_canals=a.osm_canals,
         do_peat=a.peat, do_peat_union_1km=a.peat_union_1km, client_type=a.client)
