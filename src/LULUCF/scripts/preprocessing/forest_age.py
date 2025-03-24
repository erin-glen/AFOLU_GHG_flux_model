"""
Run from src/LULUCF

Local:
python -m scripts.preprocessing.forest_age -cn AFOLU_flux_model_scripts -bb 10 49 11 50 -cs 1 --run_local --no_upload

Coiled test:
python -m scripts.utilities.create_cluster -cn AFOLU_flux_model_scripts -n 1
python -m scripts.preprocessing.forest_age -cn AFOLU_flux_model_scripts -bb 10 49 11 50 -cs 1


https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=8f5974e7-3ece-11ef-967a-4ffbfe06208e
https://datapub.gfz-potsdam.de/download/10.5880.GFZ.1.4.2023.006-VEnuo/
Metadata: https://datapub.gfz-potsdam.de/download/10.5880.GFZ.1.4.2023.006-VEnuo/2023-006_Besnard-et-al_Data-Description-v2.1.pdf
Also needs zarr package
"""

import argparse
import boto3
import dask
import numpy as np
import xarray as xr
import pystac
import fsspec
import json
import os
import rasterio

from affine import Affine
from dask.distributed import print
from concurrent.futures import ThreadPoolExecutor
from rioxarray import open_rasterio
from fsspec.implementations.cached import CachingFileSystem
import shutil
from pathlib import Path

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import log_utilities as lu
from ..utilities import universal_utilities as uu

# class S3STACReader:
#     def __init__(self, bucket_url: str, endpoint_url: str = "https://s3.gfz-potsdam.de", anon: bool = True):
#         """
#         Initialize the S3STACReader class.
#
#         Args:
#             bucket_url (str): S3 URL of the STAC catalog.
#             endpoint_url (str): S3 endpoint URL.
#             anon (bool): Set to True for public buckets, False for authenticated access.
#         """
#         self.bucket_url = bucket_url
#         s3_fs = fsspec.filesystem(
#             "s3",
#             anon=True,
#             endpoint_url="https://s3.gfz-potsdam.de"
#         )
#
#         # Wrap it in a caching filesystem
#         self.s3_fs = CachingFileSystem(
#             fs=s3_fs,
#             cache_storage="/tmp/zarr_cache",  # where to store cached chunks
#             block_size=0  # cache whole files (good for Zarr chunks)
#         )
#         self._register_stac_io()
#         self.catalog = self._load_catalog()
#
#     def _register_stac_io(self):
#         """Register a custom STAC I/O handler for reading from S3."""
#         class S3StacIO(pystac.StacIO):
#             s3_fs = self.s3_fs
#
#             def read_text(self, href: str) -> str:
#                 with self.s3_fs.open(href, "r") as f:
#                     return f.read()
#
#             def write_text(self, href: str, txt: str) -> None:
#                 with self.s3_fs.open(href, "w") as f:
#                     f.write(txt)
#
#             def exists(self, href: str) -> bool:
#                 return self.s3_fs.exists(href)
#
#         pystac.StacIO.set_default(S3StacIO)
#
#     def _load_catalog(self) -> pystac.Catalog:
#         """Load the STAC catalog from the S3 bucket."""
#         with self.s3_fs.open(self.bucket_url, 'r') as f:
#             catalog_json = json.load(f)
#         return pystac.Catalog.from_dict(catalog_json)
#
#     def list_collections(self):
#         """List all collections in the catalog."""
#         return [collection.id for collection in self.catalog.get_children()]
#
#     def list_items(self, collection_id: str):
#         """List all items in a specified collection."""
#         collection = self.catalog.get_child(collection_id)
#         return [item.id for item in collection.get_items()]
#
#     def load_zarr_dataset(self, collection_id: str, item_id: str, consolidated: bool = True) -> xr.Dataset:
#         """
#         Load a Zarr dataset from the STAC catalog.
#
#         Args:
#             collection_id (str): ID of the collection.
#             item_id (str): ID of the item.
#             consolidated (bool): Set to True if metadata is consolidated.
#
#         Returns:
#             xr.Dataset: The opened xarray dataset.
#         """
#         collection = self.catalog.get_child(collection_id)
#         item = collection.get_item(item_id)
#         zarr_asset = item.assets["zarr"]
#
#         store = self.s3_fs.get_mapper(zarr_asset.href)
#         return xr.open_zarr(store, consolidated=consolidated)

import json
import pystac
import fsspec
from fsspec.implementations.cached import CachingFileSystem
import xarray as xr
#
#
# class S3STACReader:
#     def __init__(self, bucket_url: str, endpoint_url: str = "https://s3.gfz-potsdam.de", anon: bool = True):
#         """
#         Initialize the S3STACReader class.
#
#         Args:
#             bucket_url (str): S3 URL of the STAC catalog (e.g., s3://bucket/path/catalog.json)
#             endpoint_url (str): S3 endpoint URL
#             anon (bool): Set to True for public buckets
#         """
#         self.bucket_url = bucket_url
#         self.endpoint_url = endpoint_url
#
#         # Use plain S3 FS for both metadata and Zarr access
#         self.s3_fs = fsspec.filesystem("s3", anon=anon, endpoint_url=self.endpoint_url)
#
#         self._register_stac_io()
#         self.catalog = self._load_catalog()
#
#     def _register_stac_io(self):
#         """Register custom STAC I/O using the plain S3 FS."""
#
#         class S3StacIO(pystac.StacIO):
#             def read_text(self_inner, href: str) -> str:
#                 with self.s3_fs.open(href, "r") as f:
#                     return f.read()
#
#             def write_text(self_inner, href: str, txt: str) -> None:
#                 with self.s3_fs.open(href, "w") as f:
#                     f.write(txt)
#
#             def exists(self_inner, href: str) -> bool:
#                 return self.s3_fs.exists(href)
#
#         pystac.StacIO.set_default(S3StacIO)
#
#     def _load_catalog(self) -> pystac.Catalog:
#         """Load the STAC catalog from the S3 bucket."""
#         with self.s3_fs.open(self.bucket_url, "r") as f:
#             catalog_json = json.load(f)
#         return pystac.Catalog.from_dict(catalog_json)
#
#     def list_collections(self):
#         """List all collections in the catalog."""
#         return [collection.id for collection in self.catalog.get_children()]
#
#     def list_items(self, collection_id: str):
#         """List all items in a specified collection."""
#         collection = self.catalog.get_child(collection_id)
#         return [item.id for item in collection.get_items()]
#
#     def load_zarr_dataset(self, collection_id: str, item_id: str, consolidated: bool = True) -> xr.Dataset:
#         """
#         Load a Zarr dataset from the STAC catalog.
#
#         Args:
#             collection_id (str): ID of the collection.
#             item_id (str): ID of the item.
#             consolidated (bool): Set to True if metadata is consolidated.
#
#         Returns:
#             xr.Dataset: The opened xarray dataset.
#         """
#         collection = self.catalog.get_child(collection_id)
#         item = collection.get_item(item_id)
#         zarr_asset = item.assets["zarr"]
#
#         store = self.s3_fs.get_mapper(zarr_asset.href)
#         return xr.open_zarr(store, consolidated=consolidated)

import fsspec
import xarray as xr



def calculate_forest_age(bounds, is_final, no_upload, output_dir_list, stage):
    import os
    import boto3
    import numpy as np
    import rasterio
    import xarray as xr
    from affine import Affine
    import fsspec
    from fsspec.implementations.cached import CachingFileSystem

    from ..utilities import log_utilities as lu
    from ..utilities import universal_utilities as uu
    from ..utilities import constants_and_names as cn

    logger_worker = lu.setup_logging_worker()
    s3 = boto3.client("s3")

    bounds_str = uu.boundstr(bounds)
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)

    if not is_final:
        lu.print_and_log(f"Processing chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

    ### ZARR SETUP (optimized)
    zarr_url = "s3://dog.atlaseo-glm.eo-gridded-data/collections/GAMI/GAMI_v2.1.zarr"
    cache_dir = f"/tmp/zarr_cache_{tile_id}"  # Unique cache per worker chunk
    os.makedirs(cache_dir, exist_ok=True)

    cached_fs = CachingFileSystem(
        fs=fsspec.filesystem("s3", anon=True, endpoint_url="https://s3.gfz-potsdam.de"),
        cache_storage=cache_dir,
        block_size=0,  # cache whole chunks
        check_files=False
    )

    store = cached_fs.get_mapper(zarr_url)
    ds = xr.open_zarr(store, consolidated=True)

    forest_age = ds["forest_age"]

    # Bounding box and buffering
    lon_min, lon_max = bounds[0], bounds[2]
    lat_min, lat_max = bounds[1], bounds[3]
    buffer = cn.resolution * 2

    lu.print_and_log(f"Loading data into memory {bounds_str}: {uu.timestr()}", is_final, logger_worker)
    da_chunk = forest_age.sel(
        time="2010-01-01",
        latitude=slice(lat_max + buffer, lat_min - buffer),
        longitude=slice(lon_min - buffer, lon_max + buffer)
    ).load()  # load only selected chunk

    lu.print_and_log(f"Cleaning {bounds_str}: {uu.timestr()}", is_final, logger_worker)
    da_cleaned = da_chunk.where(da_chunk != -9999, 0)
    da_median = da_cleaned.median(dim="members")

    # Target high-res lat/lon grid
    new_lat = np.arange(lat_max - cn.resolution / 2, lat_min, -cn.resolution)
    new_lon = np.arange(lon_min + cn.resolution / 2, lon_max, cn.resolution)

    lu.print_and_log(f"Interpolating {bounds_str}: {uu.timestr()}", is_final, logger_worker)
    da_resampled = da_median.interp(latitude=new_lat, longitude=new_lon, method="nearest")

    da_2010 = da_resampled.round().clip(0, 100).fillna(0).astype("int8")
    arr_2010 = da_2010.values

    arr_2015 = np.where(arr_2010 != 0, arr_2010 + 5, 0).clip(0, 100).astype("int8")

    transform = Affine.translation(lon_min, lat_max) * Affine.scale(cn.resolution, -cn.resolution)
    crs = "EPSG:4326"
    bucket_name = "gfw2-data"

    # Output paths
    file_2010 = f"/tmp/{tile_id}__{bounds_str}__forest_age_2010.tif"
    file_2015 = f"/tmp/{tile_id}__{bounds_str}__forest_age_2015.tif"

    # Write 2010
    lu.print_and_log(f"Saving 2010 raster {file_2010}: {uu.timestr()}", is_final, logger_worker)
    with rasterio.open(file_2010, 'w', driver='GTiff',
        height=arr_2010.shape[0], width=arr_2010.shape[1],
        count=1, dtype="int8", crs=crs, transform=transform,
        compress="LZW", tiled=True, blockxsize=400, blockysize=400
    ) as dst:
        dst.write(arr_2010, 1)

    # Write 2015
    lu.print_and_log(f"Saving 2015 raster {file_2015}: {uu.timestr()}", is_final, logger_worker)
    with rasterio.open(file_2015, 'w', driver='GTiff',
        height=arr_2015.shape[0], width=arr_2015.shape[1],
        count=1, dtype="int8", crs=crs, transform=transform,
        compress="LZW", tiled=True, blockxsize=400, blockysize=400
    ) as dst:
        dst.write(arr_2015, 1)

    # Upload to S3
    s3_key_2010 = f"{cn.forest_age_2010_dir[cn.full_bucket_prefix_length:]}{os.path.basename(file_2010)}"
    s3_key_2015 = f"{cn.forest_age_2015_dir[cn.full_bucket_prefix_length:]}{os.path.basename(file_2015)}"

    lu.print_and_log(f"Uploading to S3: {s3_key_2010}, {s3_key_2015}", is_final, logger_worker)
    s3.upload_file(file_2010, bucket_name, s3_key_2010)
    s3.upload_file(file_2015, bucket_name, s3_key_2015)

    lu.print_and_log(f"Finished chunk {bounds_str}: {uu.timestr()}", is_final, logger_worker)
    return f"Processed {bounds_str}: {uu.timestr()}"



def main(cluster_name, bounding_box, chunk_size, run_local=None, no_upload=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = f'starting_forest_age_10x10_deg'
    model_type = 'standard'
    age_years = [2010, 2015]

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(bounding_box, "N/A", client, cluster, log_note, run_local, model_type, stage)

    # Starting time for stage
    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Years for age maps: {age_years}")

    chunk_list = uu.get_chunk_bounds_from_bounding_box(bounding_box, chunk_size)

    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_final = False
    if len(chunk_list) > 20:
        is_final = True
        main_logger.info("Running as final model.")

    output_dir_list = [cn.forest_age_2010_dir, cn.forest_age_2015_dir]


    ### Step 2: Create 1x1 degree outputs

    # Creates list of tasks to run (1 task = 1 chunk)
    main_logger.info(f"Creating tasks and starting processing: {uu.timestr()}")
    main_logger.info("Workers' logs to be appended after main function log"+ "\n")

    futures = []

    for chunk in chunk_list:
        future = client.submit(calculate_forest_age, chunk, is_final, no_upload, output_dir_list, stage)
        futures.append(future)

    results = client.gather(futures)
    main_logger.info(results)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hansenize AFOLU model raster inputs.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    run_local = args.run_local
    no_upload = args.no_upload
    log_note = args.log_note

    bounding_box = args.bounding_box
    chunk_size = args.chunk_size

    # Create the cluster with command line arguments
    main(cluster_name, bounding_box, chunk_size, run_local, no_upload, log_note)