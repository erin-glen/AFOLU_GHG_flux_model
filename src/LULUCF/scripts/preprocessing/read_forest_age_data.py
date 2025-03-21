"""
https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=8f5974e7-3ece-11ef-967a-4ffbfe06208e
https://datapub.gfz-potsdam.de/download/10.5880.GFZ.1.4.2023.006-VEnuo/
Metadata: https://datapub.gfz-potsdam.de/download/10.5880.GFZ.1.4.2023.006-VEnuo/2023-006_Besnard-et-al_Data-Description-v2.1.pdf
Also needs zarr package
"""

import numpy as np
import xarray as xr
import pystac
import fsspec
import json

from affine import Affine

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


# Example usage
if __name__ == "__main__":
    bucket_url = "s3://dog.atlaseo-glm.eo-gridded-data/collections/catalog.json"
    reader = S3STACReader(bucket_url)

    # List collections and items
    print("Collections:", reader.list_collections())
    print("GAMI Items:", reader.list_items("GAMI"))

    # Load a Zarr dataset
    ds = reader.load_zarr_dataset("GAMI", "GAMI_v2.1")
    print(ds)

    # Select the 1x1° region with top-left corner at 50N, 10E
    lat_min, lat_max = 49.0, 50.0
    lon_min, lon_max = 10.0, 11.0

    print("defining data")
    da = ds["forest_age"]

    print("slicing")
    da_2010 = da.sel(time="2010-01-01", latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

    print("computing")
    da_subset = da_2010.mean(dim="members").compute()

    # # Select the 'forest_age' variable
    # print("defining data")
    # da = ds["forest_age"]
    #
    # # Use the ensemble mean across all members
    # da_mean = da.mean(dim="members")
    #
    # # Select the 2010-01-01 time slice
    # da_2010 = da_mean.sel(time="2010-01-01")
    #
    # print("Selected year")
    #
    # print("Creating subset...")
    #
    # # Spatial subset
    # da_subset = da_2010.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    #
    # print("Created subset and computing")
    #
    # # Load data into memory (since it's a Dask array)
    # da_subset = da_subset.compute()

    print("Finished computing")

    # Write CRS (already EPSG:4326 per metadata)
    da_subset = da_subset.rio.write_crs("EPSG:4326")

    # Replace -9999 (fill value) with 0
    print("replacing -9999s with 0s")
    da_cleaned = da_subset.where(da_subset != -9999, 0)

    resolution= 0.00025

    # Create new target coords at 0.00025° resolution
    new_lat = np.arange(lat_max - resolution / 2, lat_min, -resolution)
    new_lon = np.arange(lon_min + resolution / 2, lon_max, resolution)

    # Interpolate to new resolution
    print("resampling")
    da_resampled = da_cleaned.interp(latitude=new_lat, longitude=new_lon, method="nearest")

    print("Applying affine transform and CRS...")
    transform = Affine.translation(lon_min, lat_max) * Affine.scale(resolution, -resolution)
    da_resampled.rio.write_crs("EPSG:4326", inplace=True)
    da_resampled.rio.write_transform(transform, inplace=True)

    # Round, clip to 0–100, and cast to int8
    print("truncating to 100")
    da_subset_2010 = da_resampled.round().clip(min=0, max=100).astype("int8")

    # Save 2010 map
    print("saving 2010 map")
    output_path_2010 = f"50N_010E__{lon_min}_{lat_min}_{lon_max}_{lat_max}__forest_age_2010_int8_30m_near_v3.tif"
    da_subset_2010.rio.to_raster(
        output_path_2010,
        compress="LZW",
        tiled=True,
        blockxsize=400,
        blockysize=400
    )
    print(f"Saved: {output_path_2010}")

    # Create synthetic 2015 map by adding 5 years
    print("creating 2015 map")
    da_subset_2015 = (da_subset_2010 + 5).clip(min=0, max=100)  # prevent negative ages just in case
    output_path_2015 = f"50N_010E__{lon_min}_{lat_min}_{lon_max}_{lat_max}__forest_age_2015_int8_30m_near_v3.tif"

    print("saving 2015 map")
    da_subset_2015.rio.to_raster(
        output_path_2015,
        compress="LZW",
        tiled=True,
        blockxsize=400,
        blockysize=400
    )
    print(f"Saved: {output_path_2015}")