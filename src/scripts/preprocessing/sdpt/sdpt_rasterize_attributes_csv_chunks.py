"""
sdpt_rasterize_attributes_csv_chunks.py

Rasterize SDPT plantation attributes in small chunks using Dask:
  1) Download the species reclassification CSV from S3 if not present.
  2) Read a tile shapefile from S3 as a GeoDataFrame.
  3) Classify plantation features and split the tile into bounding boxes.
  4) Rasterize each sub-bounding box to a partial GeoTIFF.
  5) Upload partial rasters to S3 when run in default mode.

Chunk-based approach:
  - Works on sub-bounds (2° × 2°, etc.) to limit memory usage.

Usage example:
  python -m sdpt_rasterize_attributes_csv_chunks --tile_id 00N_110E --chunk_bounds "112,-4,114,-2" --run_mode test
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
SDPT_RECLASS_S3 = "climate/AFOLU_flux_model/organic_soils/inputs/raw/plantations/sdpt/updated_classified_planted_forest_species.csv"

# NOTE: The species to rotation mapping derived from this CSV is provisional
# and will likely change when the SDPT dataset is finalized.

# Final integer mapping
FINAL_MAPPING = {
    "oil_palm": 1,
    "unknown_tc": 2,
    "short_rotation": 3,
    "long_rotation": 4,
    "unknown_rotation": 5,
}

# Rasterization settings
RASTER_RES = 0.00025
RASTER_NODATA = 0
RASTER_DTYPE = "Byte"


def load_species_reclassification():
    """Load the species reclassification table.

    Attempts to download the CSV from S3 if it is missing.  If the CSV cannot
    be found, a very small default mapping bundled with the repository is used
    for development or testing purposes.
    """
    import pandas as pd

    local_csv = os.path.join(cn.local_temp_dir, os.path.basename(SDPT_RECLASS_S3))

    if not os.path.exists(local_csv):
        uutil.download_file_from_s3(SDPT_RECLASS_S3, local_csv, cn.s3_bucket_name)
        logging.info(f"Downloaded CSV => {local_csv}")
    else:
        logging.info(f"Local CSV already exists => {local_csv}, skipping download.")

    if not os.path.exists(local_csv):
        logging.warning(
            "Species CSV not found => using default test mapping from repository."
        )
        from .default_species_mapping import DEFAULT_SPECIES_TO_ROTATION

        return DEFAULT_SPECIES_TO_ROTATION

    df = pd.read_csv(local_csv)
    mapping = dict(
        zip(df["vernacName"].str.strip(), df["rotation_category"].str.strip())
    )
    logging.info(f"Loaded {len(mapping)} species from CSV.")
    return mapping


def classify_plantation(row, species_map):
    """
    Classify each row => 'oil_palm', 'unknown_tc', 'short_rotation',
                         'long_rotation', or 'unknown_rotation'.
    """
    simple_type = str(row.get("simpleType", "")).strip().lower()
    simple_name = str(row.get("simpleName", "")).strip().lower()
    vernac_name = str(row.get("vernacName", "")).strip()

    if simple_type == "tree crops":
        return "oil_palm" if "oil palm" in simple_name else "unknown_tc"
    elif simple_type == "planted forest":
        return species_map.get(vernac_name, "unknown_rotation")
    else:
        return None


def rasterize_chunk_shp(shp_path, bbox, tile_id, run_mode):
    """
    1) gdal_rasterize the shapefile chunk => partial TIF
    2) If run_mode='default', upload partial TIF => s3_processed_small
    3) If run_mode='test', keep partial TIF local
    """
    import subprocess

    chunk_name = f"{tile_id}_{int(bbox[0])}_{int(bbox[1])}_chunk.tif"
    local_dir = os.path.dirname(shp_path)
    out_tif = os.path.join(local_dir, chunk_name)

    # Build the final S3 key for partial TIF
    s3_chunk = posixpath.join(
        cn.datasets["sdpt"]["s3_processed_small"],  # e.g. '.../sdpt/YYYYMMDD'
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
        RASTER_DTYPE,
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


def create_shapefile_chunks(tile_id, tile_gdf, species_map, chunk_bounds, local_dir):
    """
    For each bounding box => filter tile_gdf => classify => local .shp => return list[(chunk_shp, bbox), ...]
    """
    results = []
    for bbox in chunk_bounds:
        minx, miny, maxx, maxy = bbox
        subset = tile_gdf.cx[minx:maxx, miny:maxy]
        if subset.empty:
            logging.info(f"No features in chunk => {bbox}, skipping.")
            continue

        # classify
        subset["plantation_type"] = subset.apply(
            lambda r: classify_plantation(r, species_map), axis=1
        )
        subset.dropna(subset=["plantation_type"], inplace=True)
        if subset.empty:
            logging.info(f"No plantation features => {bbox}, skipping.")
            continue

        subset["raster_val"] = subset["plantation_type"].map(FINAL_MAPPING)
        if subset["raster_val"].isnull().all():
            logging.info(f"All mapped to null => {bbox}, skipping.")
            continue

        # Save chunk shapefile
        chunk_name = f"{tile_id}_{int(minx)}_{int(miny)}.shp"
        chunk_path = os.path.join(local_dir, chunk_name)
        subset.to_file(chunk_path)
        results.append((chunk_path, bbox))

    return results


def process_tile(tile_id, chunk_size=2.0, run_mode="default"):
    """
    1) Read tile_{tile_id}.shp from /vsis3/ => .compute() => in-memory GeoDataFrame
    2) chunk bounding boxes
    3) classify => create chunk shapefiles
    4) produce tasks => rasterize each chunk
    """
    logging.info(f"Processing entire tile => {tile_id} in ~{chunk_size} deg sub-chunks")

    vsis3_tile_shp = (
        f"/vsis3/{cn.s3_bucket_name}/{cn.datasets['sdpt']['s3_raw']}/tile_{tile_id}.shp"
    )
    logging.info(f"Reading tile shapefile => {vsis3_tile_shp}")

    try:
        ddf = dgpd.read_file(vsis3_tile_shp, npartitions=1)
        # load entire tile in memory
        tile_gdf = ddf.compute()
    except Exception as e:
        logging.error(f"Error reading tile_{tile_id}.shp => {e}")
        return []

    if tile_gdf.empty:
        logging.info(f"No features found => tile {tile_id}")
        return []

    # bounding boxes (10x10 deg)
    minx, miny, maxx, maxy = uutil.get_10x10_tile_bounds(tile_id)
    chunk_bboxes = uutil.get_chunk_bounds([minx, miny, maxx, maxy], chunk_size)

    # reclassification
    species_map = load_species_reclassification()

    # local outdir
    local_dir = os.path.join(cn.local_temp_dir, f"sdpt_chunks_{tile_id}")
    uu.create_directory_if_not_exists(local_dir)

    # create chunk shapefiles
    chunk_list = create_shapefile_chunks(
        tile_id, tile_gdf, species_map, chunk_bboxes, local_dir
    )
    logging.info(f"Created {len(chunk_list)} chunk shapefiles for tile => {tile_id}")

    tasks = []
    for shp_path, bbox in chunk_list:
        tasks.append(
            dask.delayed(rasterize_chunk_shp)(shp_path, bbox, tile_id, run_mode)
        )

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

    vsis3_tile_shp = (
        f"/vsis3/{cn.s3_bucket_name}/{cn.datasets['sdpt']['s3_raw']}/tile_{tile_id}.shp"
    )
    logging.info(f"Reading tile shapefile => {vsis3_tile_shp}")

    try:
        ddf = dgpd.read_file(vsis3_tile_shp, npartitions=1)
        tile_gdf = ddf.compute()
    except Exception as e:
        logging.error(f"Error reading tile_{tile_id}.shp => {e}")
        return []

    if tile_gdf.empty:
        logging.info(f"No features found => tile {tile_id}")
        return []

    # reclassification
    species_map = load_species_reclassification()

    local_outdir = os.path.join(cn.local_temp_dir, f"sdpt_chunks_{tile_id}")
    uu.create_directory_if_not_exists(local_outdir)

    subset = tile_gdf.cx[
        chunk_bounds[0] : chunk_bounds[2], chunk_bounds[1] : chunk_bounds[3]
    ]
    if subset.empty:
        logging.info(f"No features in user bounding box => {chunk_bounds}, skipping.")
        return []

    subset["plantation_type"] = subset.apply(
        lambda r: classify_plantation(r, species_map), axis=1
    )
    subset.dropna(subset=["plantation_type"], inplace=True)
    if subset.empty:
        logging.info(
            f"No plantation features => bounding box {chunk_bounds}, skipping."
        )
        return []

    subset["raster_val"] = subset["plantation_type"].map(FINAL_MAPPING)
    if subset["raster_val"].isnull().all():
        logging.info(f"All mapped to null => bounding box {chunk_bounds}, skipping.")
        return []

    chunk_name = f"{tile_id}_{int(chunk_bounds[0])}_{int(chunk_bounds[1])}.shp"
    chunk_path = os.path.join(local_outdir, chunk_name)
    subset.to_file(chunk_path)

    return [
        dask.delayed(rasterize_chunk_shp)(chunk_path, chunk_bounds, tile_id, run_mode)
    ]


def main(
    tile_id=None, chunk_size=2.0, chunk_bounds=None, run_mode="default", client="local"
):
    """
    If chunk_bounds is provided => only process that bounding box.
    Otherwise => chunk the entire 10x10 tile in N sub-chunks.
    """
    logging.info(
        f"SDPT chunk-based script => partial TIFs to {cn.datasets['sdpt']['s3_processed_small']}."
    )
    if client == "coiled":
        cluster, client = uutil.connect_to_cluster(
            cluster_name="roads_canals",
            n_workers=20,
            region="us-east-1",
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
                tasks = process_tile_with_bounds(tile_id, chunk_bounds, run_mode)
            else:
                tasks = process_tile(tile_id, chunk_size, run_mode)
        else:
            logging.error(
                "No tile_id provided. (Add your 'process_all_tiles()' logic if needed.)"
            )

        logging.info(f"Computing {len(tasks)} chunk tasks ...")
        dask.compute(*tasks)

    finally:
        client.close()
        logging.info("Dask client closed.")
        if client == "coiled":
            cluster.close()
            logging.info("Coiled cluster closed.")

    logging.info("All chunk tasks completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SDPT chunk-based script => partial TIF => s3_processed_small."
    )
    parser.add_argument("--tile_id", type=str, help="Tile ID (e.g. 00N_110E).")
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
            "No CLI => example tile=00N_110E, chunk_size=2, run_mode=test, local => partial TIFs in local_processed_small."
        )
        main(
            tile_id="00N_110E",
            chunk_size=2.0,
            chunk_bounds="112,-4,114,-2",
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
"""