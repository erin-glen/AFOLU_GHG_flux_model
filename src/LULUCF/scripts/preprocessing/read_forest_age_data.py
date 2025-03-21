"""

Also needs zarr package
"""

import xarray as xr
import pystac
import fsspec
import json

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

    print("Writing to raster")

    # Save as GeoTIFF
    output_path = "forest_age_2010_50N_10E_1deg.tif"
    da_subset.rio.to_raster(output_path)

    print(f"Saved: {output_path}")