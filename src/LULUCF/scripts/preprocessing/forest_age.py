"""
Run from src/LULUCF

Local:
python -m scripts.preprocessing.forest_age -cn AFOLU_flux_model_scripts -bb 10 49 11 50 -cs 1 --run_local --no_upload

Coiled test:
python -m scripts.utilities.create_cluster -cn AFOLU_flux_model_scripts -n 1
python -m scripts.preprocessing.forest_age -bb 10 49 11 50 -cs 1


https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=8f5974e7-3ece-11ef-967a-4ffbfe06208e
https://datapub.gfz-potsdam.de/download/10.5880.GFZ.1.4.2023.006-VEnuo/
Metadata: https://datapub.gfz-potsdam.de/download/10.5880.GFZ.1.4.2023.006-VEnuo/2023-006_Besnard-et-al_Data-Description-v2.1.pdf
Also needs zarr package
"""

import argparse
import dask
import numpy as np
import xarray as xr
import pystac
import fsspec
import json

from affine import Affine
from dask.distributed import print
from concurrent.futures import ThreadPoolExecutor

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import log_utilities as lu
from ..utilities import universal_utilities as uu

class S3STACReader:
    def __init__(self, bucket_url: str, endpoint_url: str = "https://s3.gfz-potsdam.de", anon: bool = True):
        """
        Initialize the S3STACReader class.

        Args:
            bucket_url (str): S3 URL of the STAC catalog.
            endpoint_url (str): S3 endpoint URL.
            anon (bool): Set to True for public buckets, False for authenticated access.
        """
        self.bucket_url = bucket_url
        self.s3_fs = fsspec.filesystem("s3", anon=anon, endpoint_url=endpoint_url)
        self._register_stac_io()
        self.catalog = self._load_catalog()

    def _register_stac_io(self):
        """Register a custom STAC I/O handler for reading from S3."""
        class S3StacIO(pystac.StacIO):
            s3_fs = self.s3_fs

            def read_text(self, href: str) -> str:
                with self.s3_fs.open(href, "r") as f:
                    return f.read()

            def write_text(self, href: str, txt: str) -> None:
                with self.s3_fs.open(href, "w") as f:
                    f.write(txt)

            def exists(self, href: str) -> bool:
                return self.s3_fs.exists(href)

        pystac.StacIO.set_default(S3StacIO)

    def _load_catalog(self) -> pystac.Catalog:
        """Load the STAC catalog from the S3 bucket."""
        with self.s3_fs.open(self.bucket_url, 'r') as f:
            catalog_json = json.load(f)
        return pystac.Catalog.from_dict(catalog_json)

    def list_collections(self):
        """List all collections in the catalog."""
        return [collection.id for collection in self.catalog.get_children()]

    def list_items(self, collection_id: str):
        """List all items in a specified collection."""
        collection = self.catalog.get_child(collection_id)
        return [item.id for item in collection.get_items()]

    def load_zarr_dataset(self, collection_id: str, item_id: str, consolidated: bool = True) -> xr.Dataset:
        """
        Load a Zarr dataset from the STAC catalog.

        Args:
            collection_id (str): ID of the collection.
            item_id (str): ID of the item.
            consolidated (bool): Set to True if metadata is consolidated.

        Returns:
            xr.Dataset: The opened xarray dataset.
        """
        collection = self.catalog.get_child(collection_id)
        item = collection.get_item(item_id)
        zarr_asset = item.assets["zarr"]

        store = self.s3_fs.get_mapper(zarr_asset.href)
        return xr.open_zarr(store, consolidated=consolidated)


def calculate_forest_age(bounds, is_final, no_upload, output_dir_list, stage):

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []

    logger_worker = lu.setup_logging_worker()

    bounds_str = uu.boundstr(bounds)  # String form of chunk bounds
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)

    if not is_final:
        lu.print_and_log(f"Processing data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

    age_bucket_uri = "s3://dog.atlaseo-glm.eo-gridded-data/collections/catalog.json"
    # lu.print_and_log(f"s3 bucket for data: {age_bucket_uri}", is_final, logger_worker)
    reader = S3STACReader(age_bucket_uri)

    # # List collections and items
    # print("Collections:", reader.list_collections())
    # print("GAMI Items:", reader.list_items("GAMI"))

    # Load a Zarr dataset
    GAMI_ds = reader.load_zarr_dataset("GAMI", "GAMI_v2.1")
    # lu.print_and_log(GAMI_ds, is_final, logger_worker)

    GAMI_dask_array = GAMI_ds["forest_age"]

    # Select the 1x1° region with top-left corner at 50N, 10E
    lon_min, lon_max = bounds[0], bounds[2]
    lat_min, lat_max = bounds[1], bounds[3]


    print(f"slicing {bounds}: {uu.timestr()}")
    da_2010 = GAMI_dask_array.sel(time="2010-01-01", latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

    print(f"computing {bounds}: {uu.timestr()}")
    da_subset = da_2010.median(dim="members").persist()
    print(f"Finished computing {bounds}: {uu.timestr()}")

    # Write CRS (already EPSG:4326 per metadata)
    da_subset = da_subset.rio.write_crs("EPSG:4326")

    # Replace -9999 (fill value) with 0
    print(f"replacing -9999s with 0s {bounds}: {uu.timestr()}")
    da_cleaned = da_subset.where(da_subset != -9999, 0)
    resolution = 0.00025

    # Create new target coords at 0.00025° resolution
    new_lat = np.arange(lat_max - resolution / 2, lat_min, -resolution)
    new_lon = np.arange(lon_min + resolution / 2, lon_max, resolution)

    # Interpolate to new resolution
    print(f"resampling {bounds}: {uu.timestr()}")
    da_resampled = da_cleaned.interp(latitude=new_lat, longitude=new_lon, method="nearest")

    print(f"Applying affine transform and CRS {bounds}: {uu.timestr()}")
    transform = Affine.translation(lon_min, lat_max) * Affine.scale(resolution, -resolution)
    da_resampled.rio.write_crs("EPSG:4326", inplace=True)
    da_resampled.rio.write_transform(transform, inplace=True)
    # Round, clip to 0–100, and cast to int8

    print(f"truncating to 100 {bounds}: {uu.timestr()}")
    da_subset_2010 = da_resampled.round().clip(min=0, max=100).fillna(0).astype("int8")  # prevent negative ages just in case

    out_dict = {}

    # Save 2010 map
    print(f"saving 2010 map {bounds}: {uu.timestr()}")
    output_path_2010 = f"50N_010E__{lon_min}_{lat_min}_{lon_max}_{lat_max}__forest_age_2010_int8_30m_near_v5.tif"

    # Output paths without bucket (s3://gfw2-data)
    s3_path_without_bucket = f"{cn.forest_age_2010_dir[cn.full_bucket_prefix_length:]}"

    out_dict['forest_age_2010'] = [da_subset_2010, 'int8', cn.forest_age_2010_pattern, 2010, s3_path_without_bucket]

    # da_subset_2010.rio.to_raster(
    #     output_path_2010,
    #     compress="LZW",
    #     tiled=True,
    #     blockxsize=400,
    #     blockysize=400
    # )

    # Create synthetic 2015 map by adding 5 years
    da_subset_2015 = (da_subset_2010 + 5).clip(min=0, max=100).fillna(0)  # prevent negative ages just in case

    output_path_2015 = f"50N_010E__{lon_min}_{lat_min}_{lon_max}_{lat_max}__forest_age_2015_int8_30m_near_v5.tif"

    # Output paths without bucket (s3://gfw2-data)
    s3_path_without_bucket = f"{cn.forest_age_2015_dir[cn.full_bucket_prefix_length:]}"

    out_dict['forest_age_2015'] = [da_subset_2015, 'int8', cn.forest_age_2015_pattern, 2015, s3_path_without_bucket]

    print(f"Saving {bounds} locally: {uu.timestr()}")
    # da_subset_2015.rio.to_raster(
    #     output_path_2015,
    #     compress="LZW",
    #     tiled=True,
    #     blockxsize=400,
    #     blockysize=400
    # )

    # Converts output numpy arrays to local rasters and puts them in a list of files to upload in parallel
    upload_tasks = uu.save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str,
                                                       out_dict, is_final, logger_worker, 0)

    # Only prints if not a final run
    if not is_final:
        lu.print_and_log(f"Upload tasks created for {bounds_str} in {tile_id}. Ready to upload: {uu.timestr()}",
                         is_final, logger_worker)

    # Executes uploads in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks)

    return (f"Processed {bounds}: {uu.timestr()}")


# Example usage
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

    forest_age_delayed_results = [dask.delayed(calculate_forest_age)
                                  (chunk, is_final, no_upload, output_dir_list, stage) for chunk in chunk_list]

    # Runs analysis and gathers results
    forest_age_results = dask.compute(*forest_age_delayed_results)
    print(forest_age_results)




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