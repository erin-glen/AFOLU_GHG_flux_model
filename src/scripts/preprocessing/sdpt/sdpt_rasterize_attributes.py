"""
sdpt_rasterize_attributes.py

Rasterize SDPT plantation attributes in small chunks using Dask:
  1) Download the species reclassification CSV from S3 if not present.
  2) Read a tile shapefile from S3 as a GeoDataFrame.
  3) Classify plantation features and split the tile into bounding boxes.
  4) Rasterize each sub-bounding box in memory to a partial GeoTIFF.
  5) Upload partial rasters to S3 when run in default mode.

Chunk-based approach:
  - Works on sub-bounds (2° × 2°, etc.) to limit memory usage.
  - If ``--tile_id`` is omitted, all tiles under ``sdpt`` raw S3 path are processed.

Usage examples:
  python -m sdpt_rasterize_attributes_csv_chunks --tile_id 00N_110E --chunk_bounds "112,-4,114,-2" --run_mode test
  python -m src.scripts.preprocessing.sdpt.sdpt_rasterize_attributes --client local
"""

import os
import sys
import logging
import argparse
import warnings
import posixpath
import gc
from pyogrio.errors import FeatureError


import dask
import dask_geopandas as dgpd
from dask.distributed import Client, LocalCluster
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
import boto3

# Our universal constants & utilities
import src.scripts.preprocessing.preprocessing_constants as cn
import src.scripts.preprocessing.utilities as uu
from src.scripts.utilities import universal_utilities as uutil

warnings.filterwarnings("ignore", "Geometry is in a geographic CRS.", UserWarning)

# ---------------------------------------------------------------------------
# Logging config
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Reclassification CSV in S3
# The advanced remapping table already contains numeric rotation codes.
ADVANCED_REMAP_S3 = "climate/AFOLU_flux_model/organic_soils/inputs/raw/plantations/sdpt/remapping_tables/advanced_remapping.csv"

# Rasterization settings
RASTER_RES = 0.00025
RASTER_NODATA = 0
# Using a numpy dtype avoids ``rasterio`` ``TypeError`` when rasterising
# attributes directly in memory.
RASTER_DTYPE = np.uint8


def list_sdpt_shapefiles():
    """Return the list of SDPT shapefiles stored on S3."""
    prefix = cn.datasets["sdpt"]["s3_raw"]
    return [
        k
        for k in uutil.list_s3_files(cn.s3_bucket_name, prefix)
        if k.lower().endswith(".shp")
    ]


def load_species_reclassification():
    """Return a mapping of vernacular names to numeric rotation codes.

    The function looks for ``advanced_remapping.csv`` in the temporary
    directory and attempts to download it from S3 if missing.  When the CSV
    cannot be obtained the function returns an empty mapping so that
    classification falls back to the logic defined in ``create_remapping``.
    """
    import pandas as pd

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
        logging.warning(
            "Advanced remapping CSV not found; falling back to classification logic."
        )
        return {}

    df = pd.read_csv(local_csv)
    try:
        mapping = dict(zip(df["vernacName"].str.strip(), df["rotation_code"]))
    except KeyError:
        logging.warning("advanced_remapping.csv missing expected columns")
        mapping = {}

    logging.info(f"Loaded {len(mapping)} species from advanced remapping CSV.")
    return mapping


from .create_remapping import classify as fallback_classify
from .create_remapping import ROTATION_CLASS_CODES


def classify_plantation(row, species_map):
    """Return the numeric rotation code for ``row``.

    If ``vernacName`` exists in ``species_map`` its numeric code is used
    directly.  Otherwise the row is classified using the heuristic logic in
    :mod:`create_remapping`.  Unknown tree crops are ignored.
    """

    simple_type = str(row.get("simpleType", "")).strip().lower()
    simple_name = str(row.get("simpleName", "")).strip().lower()
    vernac_name = str(row.get("vernacName", "")).strip()

    code = species_map.get(vernac_name)
    if code is not None:
        return code

    if simple_type == "tree crops":
        return (
            ROTATION_CLASS_CODES.get("oil_palm") if "oil palm" in simple_name else None
        )

    if simple_type == "planted forest":
        cls = fallback_classify(row)
        return ROTATION_CLASS_CODES.get(cls)

    return None


def rasterize_chunk_shp(shp_path, bbox, tile_id, run_mode):
    """
    1) gdal_rasterize the shapefile chunk => partial TIF
    2) If run_mode='default', upload partial TIF => s3_processed_base/<px>/YYYYMMDD
    3) If run_mode='test', keep partial TIF local
    """
    import subprocess

    chunk_str = uutil.boundstr(bbox)
    chunk_px = uutil.calc_chunk_length_pixels(bbox)
    chunk_name = f"{tile_id}__{chunk_str}__sdpt.tif"
    local_dir = cn.datasets["sdpt"]["local_processed"]
    uu.create_directory_if_not_exists(local_dir)
    out_tif = os.path.join(local_dir, chunk_name)

    # Build the final S3 key for partial TIF
    s3_chunk = posixpath.join(
        cn.datasets["sdpt"]["s3_processed_base"],
        f"{chunk_px}_pixels",
        cn.today_date,
        chunk_name,
    )

    # 1) If run_mode='default', skip if partial TIF already on S3
    #    If run_mode='test', skip if partial TIF local
    if run_mode == "default":
        if uutil.s3_file_exists(cn.s3_bucket_name, s3_chunk):
            logging.info(
                f"Partial TIF => s3://{cn.s3_bucket_name}/{s3_chunk} exists => skipping."
            )
            # remove shapefile pieces
            for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                chunk_file = shp_path.replace(".shp", ext)
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
            return
    else:  # run_mode='test'
        if os.path.exists(out_tif):
            logging.info(f"Partial TIF => {out_tif} exists locally => skipping.")
            for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                chunk_file = shp_path.replace(".shp", ext)
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
            return

    # 2) gdal_rasterize
    minx, miny, maxx, maxy = bbox
    gdal_cmd = [
        "gdal_rasterize",
        "-a",
        "raster_val",
        "-te",
        str(minx),
        str(miny),
        str(maxx),
        str(maxy),
        "-tr",
        str(RASTER_RES),
        str(RASTER_RES),
        "-a_nodata",
        str(RASTER_NODATA),
        "-init",
        str(RASTER_NODATA),
        "-ot",
        "Byte" if RASTER_DTYPE == np.uint8 else str(RASTER_DTYPE),
        "-co",
        "COMPRESS=DEFLATE",
        "-co",
        "TILED=YES",
        shp_path,
        out_tif,
    ]
    logging.info(f"Rasterizing chunk => tile {tile_id} => {bbox}")
    subprocess.run(gdal_cmd, check=True)

    if not os.path.exists(out_tif):
        logging.error(f"Failed to produce partial TIF => {out_tif}")
        return

    # 3) If run_mode='default', upload partial TIF to S3
    if run_mode == "default":
        logging.info(f"Uploading partial TIF => s3://{cn.s3_bucket_name}/{s3_chunk}")
        uutil.upload_file_to_s3(out_tif, cn.s3_bucket_name, s3_chunk)
        os.remove(out_tif)
    else:
        logging.info(f"Test mode => partial TIF => {out_tif} retained locally.")

    # remove chunk shapefile
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        chunk_file = shp_path.replace(".shp", ext)
        if os.path.exists(chunk_file):
            os.remove(chunk_file)

    # free memory used for rasterization
    del gdal_cmd, out_tif
    gc.collect()


def rasterize_chunk_df(subset_gdf, bbox, tile_id, run_mode):
    """Rasterize a GeoDataFrame subset directly in memory.

    This avoids writing temporary shapefiles and uses ``rasterio`` to create
    the partial raster.  The behaviour mirrors :func:`rasterize_chunk_shp` but
    operates on an in-memory dataframe.
    """

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

    if run_mode == "default":
        if uutil.s3_file_exists(cn.s3_bucket_name, s3_chunk):
            logging.info(
                f"Partial TIF => s3://{cn.s3_bucket_name}/{s3_chunk} exists => skipping."
            )
            return
    else:
        if os.path.exists(out_tif):
            logging.info(f"Partial TIF => {out_tif} exists locally => skipping.")
            return

    shapes = [
        (geom, val) for geom, val in zip(subset_gdf.geometry, subset_gdf["raster_val"])
    ]
    if not shapes:
        logging.info(f"No shapes to rasterize in {bbox}, skipping.")
        return

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
        all_touched=True,
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

    with MemoryFile() as memfile:
        with memfile.open(**meta) as dst:
            dst.write(burned, 1)

        if run_mode == "default":
            logging.info(
                f"Uploading partial TIF => s3://{cn.s3_bucket_name}/{s3_chunk}"
            )
            uutil.upload_fileobj_to_s3(memfile, cn.s3_bucket_name, s3_chunk)
        else:
            with open(out_tif, "wb") as local_out:
                local_out.write(memfile.read())
            logging.info(f"Test mode => partial TIF => {out_tif} retained locally.")

    del burned
    gc.collect()


@dask.delayed
def clip_geometries(tile_gdf, bbox):
    minx, miny, maxx, maxy = bbox
    return tile_gdf.cx[minx:maxx, miny:maxy].copy()


@dask.delayed
def classify_features(sub_gdf, species_map):
    if hasattr(species_map, "result"):
        species_map = species_map.result()
    if hasattr(sub_gdf, "compute"):
        sub_gdf = sub_gdf.compute()
    sub_gdf["raster_val"] = sub_gdf.apply(
        lambda r: classify_plantation(r, species_map), axis=1
    )
    sub_gdf.dropna(subset=["raster_val"], inplace=True)
    return sub_gdf[["geometry", "raster_val"]]


def _load_tile_gdf(tile_id):
    """Return the GeoDataFrame for ``tile_id`` or ``None`` on failure."""

    vsis3_tile_shp = (
        f"/vsis3/{cn.s3_bucket_name}/{cn.datasets['sdpt']['s3_raw']}/tile_{tile_id}.shp"
    )
    logging.info(f"Reading tile shapefile => {vsis3_tile_shp}")

    try:
        ddf = dgpd.read_file(vsis3_tile_shp, npartitions=1)
        tile_gdf = ddf.persist()
        count = tile_gdf.map_partitions(len, meta=('length', int)).compute().sum()
        if count == 0:
            logging.info(f"No features found => tile {tile_id}")
            return None
    except Exception as e:
        logging.error(f"Error reading tile_{tile_id}.shp => {e}")
        return None

    return tile_gdf


def process_tile(tile_id, species_map, chunk_size=1.0, run_mode="default"):
    """
    1) Read tile_{tile_id}.shp from /vsis3/ lazily
    2) chunk bounding boxes
    3) classify => build chunk GeoDataFrames
    4) produce tasks => rasterize each chunk
    """
    logging.info(f"Processing entire tile => {tile_id} in ~{chunk_size} deg sub-chunks")

    tile_gdf = _load_tile_gdf(tile_id)
    if tile_gdf is None:
        return []

    # bounding boxes (10x10 deg)
    minx, miny, maxx, maxy = uutil.get_10x10_tile_bounds(tile_id)
    chunk_bboxes = uutil.get_chunk_bounds([minx, miny, maxx, maxy], chunk_size)

    tasks = []
    for bbox in chunk_bboxes:
        classified = classify_features(clip_geometries(tile_gdf, bbox), species_map)
        tasks.append(
            dask.delayed(rasterize_chunk_df)(classified, bbox, tile_id, run_mode)
        )

    return tasks


def process_tile_with_bounds(tile_id, chunk_bounds, species_map, run_mode="default"):
    """
    If user passes a single bounding box => skip the big loop,
    only process that chunk bounding box.
    """
    logging.info(
        f"Processing single bounding box => {chunk_bounds} for tile => {tile_id}"
    )

    # parse chunk_bounds if it's a string
    # might already be a tuple, but let's ensure float-cast
    if isinstance(chunk_bounds, str):
        minx, miny, maxx, maxy = map(float, chunk_bounds.split(","))
        chunk_bounds = (minx, miny, maxx, maxy)

    tile_gdf = _load_tile_gdf(tile_id)
    if tile_gdf is None:
        return []

    classified = classify_features(clip_geometries(tile_gdf, chunk_bounds), species_map)

    return [
        dask.delayed(rasterize_chunk_df)(classified, chunk_bounds, tile_id, run_mode)
    ]


def process_all_tiles(species_map, chunk_size=1.0, run_mode="default"):
    """Process every SDPT tile sequentially to avoid memory blowout."""

    for shp_key in list_sdpt_shapefiles():
        tile_id = os.path.basename(shp_key)[len("tile_") : -4]
        tasks = process_tile(tile_id, species_map, chunk_size, run_mode)
        if tasks:
            logging.info(
                f"Computing {len(tasks)} chunk tasks for tile => {tile_id} ..."
            )
            dask.compute(*tasks)


def main(
    tile_id=None, chunk_size=1.0, chunk_bounds=None, run_mode="default", client="local"
):
    """
    If chunk_bounds is provided => only process that bounding box.
    Otherwise => chunk the entire 10x10 tile in N sub-chunks.
    """
    logging.info(
        f"SDPT chunk-based script => base S3 path {cn.datasets['sdpt']['s3_processed_base']}"
    )
    if client == "coiled":
        cluster, client = uutil.connect_to_cluster(
            cluster_name="sdpt_rasterization",
            n_workers=20,
            region="us-east-1",
            worker_memory="128GiB",
        )
        logging.info(f"Coiled cluster => {cluster.name}")
    else:
        cluster = LocalCluster()
        client = Client(cluster)
        logging.info("Local Dask client started.")

    # Load and broadcast species reclassification table once
    mapping = load_species_reclassification()
    species_map = client.scatter(mapping, broadcast=True)

    tasks = []

    try:
        if tile_id:
            if chunk_bounds:
                logging.info(
                    f"Processing tile => {tile_id}, chunk bounds => {chunk_bounds}"
                )
                tasks = process_tile_with_bounds(
                    tile_id, chunk_bounds, species_map, run_mode
                )
            else:
                tasks = process_tile(tile_id, species_map, chunk_size, run_mode)

            logging.info(f"Computing {len(tasks)} chunk tasks ...")
            dask.compute(*tasks)
        else:
            logging.info("No tile_id provided => processing all tiles.")
            process_all_tiles(species_map, chunk_size, run_mode)

    finally:
        client.close()
        logging.info("Dask client closed.")
        if client == "coiled":
            cluster.close()
            logging.info("Coiled cluster closed.")

    logging.info("All chunk tasks completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SDPT chunk-based script => partial TIFs stored under s3_processed_base/<px>/YYYYMMDD."
    )
    parser.add_argument(
        "--tile_id",
        type=str,
        help="Tile ID (e.g. 00N_110E). Omit to process all tiles.",
    )
    parser.add_argument(
        "--chunk_size", type=float, default=1.0, help="Chunk size (deg)."
    )
    parser.add_argument(
        "--chunk_bounds",
        type=str,
        help='Optional single bounding box "min_x,min_y,max_x,max_y" for quick testing.',
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        choices=["default", "test"],
        default="default",
        help="default => partial TIF => S3, test => local partial TIFs.",
    )
    parser.add_argument(
        "--client",
        type=str,
        choices=["local", "coiled"],
        default="local",
        help="Dask client type (local or coiled).",
    )

    args = parser.parse_args()

    if not any(sys.argv[1:]):
        logging.info(
            "No CLI => processing all tiles locally in test mode for demonstration."
        )
        main(
            tile_id=None,
            chunk_size=1.0,
            chunk_bounds=None,
            run_mode="test",
            client="local",
        )
    else:
        main(
            tile_id=args.tile_id,
            chunk_size=args.chunk_size,
            chunk_bounds=args.chunk_bounds,
            run_mode=args.run_mode,
            client=args.client,
        )

"""
Examples for running in command:

python -m src.preprocessing.sdpt.sdpt_rasterize_attributes_csv_chunks \
  --tile_id 00N_110E \
  --chunk_bounds "112,-4,114,-2" \
  --run_mode test \
  --client local
python -m src.scripts.preprocessing.sdpt.sdpt_rasterize_attributes --client local

test
"""