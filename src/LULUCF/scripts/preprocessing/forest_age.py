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

    # Bucket where age maps are stored
    age_bucket_uri = "s3://dog.atlaseo-glm.eo-gridded-data/collections/catalog.json"
    # lu.print_and_log(f"s3 bucket for data: {age_bucket_uri}", is_final, logger_worker)
    reader = S3STACReader(age_bucket_uri)

    # # Lists collections and items
    # print("Collections:", reader.list_collections())
    # print("GAMI Items:", reader.list_items("GAMI"))

    # Loads a Zarr dataset-- GAMI v2.1
    GAMI_ds = reader.load_zarr_dataset("GAMI", "GAMI_v2.1")
    # lu.print_and_log(GAMI_ds, is_final, logger_worker)

    # Gets the forest age dataset
    GAMI_dask_array = GAMI_ds["forest_age"]

    # Bounding box for chunk
    lon_min, lon_max = bounds[0], bounds[2]
    lat_min, lat_max = bounds[1], bounds[3]

    lu.print_and_log(f"slicing {bounds}: {uu.timestr()}", is_final, logger_worker)

    # Gets the age map for the chunk.
    # The chunk needs to be buffered by +/- double the pixel resolution so that the resampled output extends all the way
    # to the edge of the bounding box. Without cn.resolution*2, there was a one pixel-wide border around the edge with
    # no values in it. I believe that's because the resampling was only for pixels with their centers in the targeted
    # slice, so the slice has to be extended a bit so that the outermost resampled pixels have their center within
    # the (expanded) bounding box. The output raster is still the exact right size.
    da_2010 = GAMI_dask_array.sel(
        time="2010-01-01",
        latitude=slice(lat_max + cn.resolution*2, lat_min - cn.resolution*2),
        longitude=slice(lon_min - cn.resolution*2, lon_max + cn.resolution*2)
    )

    lu.print_and_log(f"Computing {bounds}: {uu.timestr()}", is_final, logger_worker)
    da_subset = da_2010.mean(dim="members").persist()
    lu.print_and_log(f"Finished computing {bounds}: {uu.timestr()}", is_final, logger_worker)

    da_subset = da_subset.rio.write_crs("EPSG:4326")

    # Replaces -9999 (fill value) with 0
    lu.print_and_log(f"Replacing -9999s with 0s {bounds}: {uu.timestr()}", is_final, logger_worker)
    da_cleaned = da_subset.where(da_subset != -9999, 0)

    # Creates new target coords at 0.00025° resolution
    new_lat = np.arange(lat_max - cn.resolution / 2, lat_min, -cn.resolution)
    new_lon = np.arange(lon_min + cn.resolution / 2, lon_max, cn.resolution)

    # Interpolates to new resolution
    lu.print_and_log(f"Resampling {bounds}: {uu.timestr()}", is_final, logger_worker)
    da_resampled = da_cleaned.interp(latitude=new_lat, longitude=new_lon, method="nearest")

    lu.print_and_log(f"Applying affine transform and CRS {bounds}: {uu.timestr()}", is_final, logger_worker)
    transform = Affine.translation(lon_min, lat_max) * Affine.scale(cn.resolution, -cn.resolution)
    da_resampled.rio.write_crs("EPSG:4326", inplace=True)
    da_resampled.rio.write_transform(transform, inplace=True)

    lu.print_and_log(f"Truncating to 100 {bounds}: {uu.timestr()}", is_final, logger_worker)
    da_subset_2010 = da_resampled.round().clip(min=0, max=100).fillna(0).astype("int8")  # prevent negative ages just in case

    # Dictionary of outputs
    out_dict = {}

    # Saves 2010 map
    lu.print_and_log(f"Saving 2010 map {bounds}: {uu.timestr()}", is_final, logger_worker)

    # Output path without bucket (s3://gfw2-data)
    s3_path_2010_without_bucket = f"{cn.forest_age_2010_dir[cn.full_bucket_prefix_length:]}"
    s3_path_2015_without_bucket = f"{cn.forest_age_2015_dir[cn.full_bucket_prefix_length:]}"

    # Output dictionary used for uploading to s3
    out_dict['forest_age_2010'] = [da_subset_2010, 'int8', cn.forest_age_2010_pattern, 2010, s3_path_2010_without_bucket]

    # Creates synthetic 2015 map by adding 5 years to non-NoData pixels.
    # Caps maximum age at 100 and prevents negative ages
    lu.print_and_log(f"Creating 2015 age map for {bounds}: {uu.timestr()}", is_final, logger_worker)
    da_subset_2015 = xr.where(
        da_subset_2010 != 0,
        da_subset_2010 + 5,
        0
    ).clip(min=0, max=100).astype("int8")  # prevent negative ages just in case

    out_dict['forest_age_2015'] = [da_subset_2015, 'int8', cn.forest_age_2015_pattern, 2015, s3_path_2015_without_bucket]

    lu.print_and_log(f"Saving {bounds} locally: {uu.timestr()}", is_final, logger_worker)

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
    main_logger.info(forest_age_results)




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