"""Simplified SDPT rasterisation script.

This variant only rasterises the ``simpleType`` attribute of the SDPT
shapefiles.  ``Planted forest`` features are encoded as ``1`` and
``Tree crops`` as ``2``.  The general workflow mirrors
``sdpt_rasterize_attributes`` but skips the advanced species
reclassification step.
"""

import os
import sys
import logging
import argparse
import warnings
import posixpath
import gc

import dask
import dask_geopandas as dgpd
from dask.distributed import Client, LocalCluster
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

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



def classify_simple_type(row):
    """Map ``simpleType`` to numeric codes.

    Parameters
    ----------
    row : pandas.Series
        Feature attributes with a ``simpleType`` field.

    Returns
    -------
    int or None
        ``1`` for planted forest, ``2`` for tree crops or ``None`` to ignore.
    """

    val = str(row.get("simpleType", "")).strip().lower()
    if val == "planted forest":
        return 1
    if val == "tree crops":
        return 2
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
def clip_geometries(tile_gdf, bbox):
    minx, miny, maxx, maxy = bbox
    return tile_gdf.cx[minx:maxx, miny:maxy].copy()


@dask.delayed
def classify_features(sub_gdf):
    sub_gdf["raster_val"] = sub_gdf.apply(classify_simple_type, axis=1)
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
        tile_gdf = ddf.compute()
    except Exception as e:
        logging.error(f"Error reading tile_{tile_id}.shp => {e}")
        return None

    if tile_gdf.empty:
        logging.info(f"No features found => tile {tile_id}")
        return None

    return tile_gdf


def process_tile(tile_id, chunk_size=2.0, run_mode="default"):
    logging.info(f"Processing entire tile => {tile_id} in ~{chunk_size} deg sub-chunks")

    tile_gdf = _load_tile_gdf(tile_id)
    if tile_gdf is None:
        return []

    minx, miny, maxx, maxy = uutil.get_10x10_tile_bounds(tile_id)
    chunk_bboxes = uutil.get_chunk_bounds([minx, miny, maxx, maxy], chunk_size)

    tasks = []
    for bbox in chunk_bboxes:
        classified = classify_features(clip_geometries(tile_gdf, bbox))
        tasks.append(
            dask.delayed(rasterize_chunk_df)(classified, bbox, tile_id, run_mode)
        )

    # Cleanup explicitly after creating tasks
    del tile_gdf
    gc.collect()

    return tasks



def process_tile_with_bounds(tile_id, chunk_bounds, run_mode="default"):
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

    classified = classify_features(
        clip_geometries(tile_gdf, chunk_bounds)
    )

    return [
        dask.delayed(rasterize_chunk_df)(
            classified, chunk_bounds, tile_id, run_mode
        )
    ]


def process_all_tiles(chunk_size=2.0, run_mode="default"):
    """Process every SDPT tile sequentially to avoid memory blowout."""

    for shp_key in list_sdpt_shapefiles():
        tile_id = os.path.basename(shp_key)[len("tile_") : -4]
        tasks = process_tile(tile_id, chunk_size, run_mode)
        if tasks:
            logging.info(
                f"Computing {len(tasks)} chunk tasks for tile => {tile_id} ..."
            )
            dask.compute(*tasks)


def main(
    tile_id=None, chunk_size=2.0, chunk_bounds=None, run_mode="default", client="local"
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
            n_workers=25,
            region="us-east-1",
            worker_memory="128GiB",
        )
        logging.info(f"Coiled cluster => {cluster.name}")
    else:
        cluster = LocalCluster()
        client = Client(cluster)
        logging.info("Local Dask client started.")

    tasks = []

    try:
        if tile_id:
            if chunk_bounds:
                logging.info(
                    f"Processing tile => {tile_id}, chunk bounds => {chunk_bounds}"
                )
                tasks = process_tile_with_bounds(
                    tile_id, chunk_bounds, run_mode
                )
            else:
                tasks = process_tile(tile_id, chunk_size, run_mode)

            logging.info(f"Computing {len(tasks)} chunk tasks ...")
            dask.compute(*tasks)
        else:
            logging.info("No tile_id provided => processing all tiles.")
            process_all_tiles(chunk_size, run_mode)

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
        "--chunk_size", type=float, default=2.0, help="Chunk size (deg)."
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
            chunk_size=2.0,
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