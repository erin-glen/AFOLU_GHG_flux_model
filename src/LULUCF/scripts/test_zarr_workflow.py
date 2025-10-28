"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn zarr_testing
python -m src.LULUCF.scripts.test_zarr_workflow -cn zarr_testing

Most recent ChatGPT convo: https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68feb594-6d50-8325-a59a-f4c53750be17
"""

from datetime import datetime
import pytz
import numpy as np
import xarray as xr
import dask.array as da
import zarr
import fsspec
import coiled
import time
from datetime import datetime
import json
import dask
import numcodecs
from dask.distributed import Client, print
from dask import delayed
from dask.diagnostics import ProgressBar


from rechunker import rechunk
import os


# ──────────────────────────────
# CONFIGURATION
# ──────────────────────────────

dataset_keys = [
    "carbon_density__AGC__MgC",
    # "carbon_density__BGC__MgC",
    # "flux_NEE",
    # "flux_FIRE",
    # "forest_mask",
]

test_n_years = 1
resolution = 0.00025
lat_size = int(180 / resolution)   # 720000
lon_size = int(360 / resolution)   # 1440000
dtype = "float32"
fill_value = 0.0

# Lat/lon coordinate arrays
lats = np.arange(90.0 - resolution / 2, -90, -resolution)[:lat_size]
lons = np.arange(-180.0 + resolution / 2, 180, resolution)[:lon_size]
years = np.arange(test_n_years)

# Target region
target_box = {
    "lat_min": 1.0,
    "lat_max": 2.0,
    "lon_min": 20.0,
    "lon_max": 21.0
}


# ──────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────
def timestr(format="full"):

    # Define the Eastern Time timezone
    eastern = pytz.timezone('US/Eastern')

    # Get the current time in UTC and convert to Eastern Time
    eastern_time = datetime.now(eastern)

    # Format the time as a string
    if format == "time":
        return eastern_time.strftime("%H:%M:%S")
    else:
        return eastern_time.strftime("%Y%m%d_%H_%M_%S")


def connect_to_Coiled_cluster(cluster_name, run_local, fallback_to_local_on_failure=True):

    # If local run flag is on, doesn't return a cluster or client
    if run_local:
        print("Running locally without Dask/Coiled.")
        return None, None, run_local

    # If no local run flag, it tries to attach to the named cluster
    try:
        # Gets info on all Coiled clusters (including terminated ones)
        all_clusters = coiled.list_clusters()

        # Iterates through clusters and identifies the running one of the correct name to connect to
        for cluster in all_clusters:
            if (cluster.get("name") == cluster_name) and (cluster.get("current_state", {}).get("state") in ['scaling', 'ready']):
                print(f"Connecting to running cluster '{cluster_name}'.")
                cluster = coiled.Cluster(name=cluster_name)
                client = Client(cluster)
                return cluster, client, run_local

        if fallback_to_local_on_failure:
            print(f"Cluster named {cluster_name} not found. Running locally.")
            return None, None, True
        else:
            raise RuntimeError(f"No running cluster named '{cluster_name}' found.")

    except Exception as e:
        if fallback_to_local_on_failure:
            print(f"Error while connecting to Coiled cluster: {e}\nRunning locally instead.")
            return None, None, True
        else:
            raise


def initialize_zarr_with_coords(
    store_url: str,
    dataset_keys: list[str],
    n_years: int,
    resolution: float = 0.00025,
    dtype: str = "float32",
    fill_value: float = np.nan,
    chunks: tuple[int, int, int] = (1, 4000, 4000),
):
    """
    Create a Zarr store on S3 with coordinate arrays (x/y/year),
    spatial_ref metadata, and dataset definitions WITHOUT allocating global arrays.
    """

    # Compute dimensions
    lat_size = int(180 / resolution)
    lon_size = int(360 / resolution)

    # Create coordinate arrays
    lats = np.arange(90.0 - resolution / 2, -90, -resolution)[:lat_size]
    lons = np.arange(-180.0 + resolution / 2, 180, resolution)[:lon_size]
    years = np.arange(n_years)

    # Spatial reference (CRS metadata)
    spatial_attrs = {
        "grid_mapping_name": "latitude_longitude",
        "epsg_code": 4326,
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
    }

    # ──────────────────────────────
    # Configure zstd compression
    # ──────────────────────────────
    compressor = {
        "name": "zstd",
        "configuration": {"level": 3}
    }
    print(f"🧩 Using zstd compression (level=3): {timestr()}")

    # Use Dask arrays filled lazily
    data_vars = {}
    encoding = {}

    for key in dataset_keys:
        dask_data = da.full(
            (n_years, lat_size, lon_size),
            fill_value,
            dtype=dtype,
            chunks=chunks,
        )

        data_vars[key] = xr.DataArray(
            dask_data,
            dims=("year", "y", "x"),
            coords={"year": years, "y": lats, "x": lons},
            name=key,
            attrs={"grid_mapping": "spatial_ref"},
        )

        # Define encoding (compression, dtype, and chunks)
        encoding[key] = {"compressor": compressor}

    # Construct dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "x": lons,
            "y": lats,
            "year": years,
        },
    )
    ds["spatial_ref"] = xr.DataArray(
        np.array(0, dtype="int32"),
        attrs=spatial_attrs,
    )

    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(store_url)

    ds.to_zarr(
        store=mapper,
        compute=False,
        mode='w',
        encoding=encoding,
        zarr_format=3
    )

    # Clean _FillValue in populated rechunked zarr
    ### Need to remove _FillValue attribute in zarr because it's being encoded in some way that is incompatible with xarray while using zarr v3,
    ### per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68f984c6-9aa0-8327-a910-5ad9a8d170fc.
    ### There doesn't seem to be a way to create the zarr with a correctly encoded _FillValue in the first place,
    ### hence this fix after the fact.

    # Open Zarr group in read/write mode
    z = zarr.open_group(store=mapper, mode="r+")
    print(z["carbon_density__AGC__MgC"].filters)  # Should include compressor info

    # Loop through all arrays
    for key in z.array_keys():
        arr = z[key]
        if "_FillValue" in arr.attrs:
            print(f"🔧 Removing _FillValue from {key}")
            del arr.attrs["_FillValue"]

    print(f"Cleaned _FillValue from Zarr metadata: {timestr()}")

    print(f"✅ Initialized spatial Zarr metadata at {store_url}: {timestr()}")


def latlon_to_indices(lat, lon):
    """Convert lat/lon to array indices for (lat descending, lon increasing)."""
    lat_max = 90.0
    lon_min = -180.0
    lat_idx = int(round((lat_max - lat) / resolution))
    lon_idx = int(round((lon - lon_min) / resolution))
    return lat_idx, lon_idx


def write_chunk(dataset_key, year_idx, lat0, lat1, lon0, lon1, store_url):
    """Function to be executed by Coiled worker to write random data into Zarr."""
    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(store_url)
    z = zarr.open(mapper, mode="r+")

    h = lat1 - lat0
    w = lon1 - lon0
    print(f"🧩 Worker writing {dataset_key}[{year_idx}, {lat0}:{lat1}, {lon0}:{lon1}]")

    # Write random data
    data = np.random.rand(h, w).astype("float32")
    z[dataset_key][year_idx, lat0:lat1, lon0:lon1] = data
    return f"✅ Wrote {dataset_key} year={year_idx} region=({lat0}:{lat1}, {lon0}:{lon1})"


def check_region_stats(store_url, dataset_key, year_idx):
    """Check min/max of the region written."""
    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(store_url)
    z = zarr.open(mapper, mode="r")

    lat0, lon0 = latlon_to_indices(target_box["lat_max"], target_box["lon_min"])
    lat1, lon1 = latlon_to_indices(target_box["lat_min"], target_box["lon_max"])

    region_array = z[dataset_key][year_idx, lat0:lat1, lon0:lon1]

    # print(dataset_key)
    # print(mapper)
    # print(z)
    # print(lat0, lon0)
    # print(lat1, lon1)
    # print(region)

    # Non-zero pixels in the array
    non_zero_count = np.count_nonzero(region_array)

    print(f"🔍 {dataset_key} year {year_idx}: min={region_array.min()}, mean={region_array.mean()}, max={region_array.max()}, non-zero cells={non_zero_count}")


# Rechunks a latitudinal band for the entire planet for a given year and dataset.
# The rechunk size is determined by the pre-created rechunked zarr (destination zarr).
# Here, the corresponding source and rechunked zarr datasets are used and have their chunking built-in.
def rechunk_by_lat_band(year_idx, y_start, y_block_size, ny, source_zarr_ds, rechunk_zarr_ds):

    lat_band_start_time = time.time()
    y_end = min(y_start + y_block_size, ny)   # The ending latitudinal band
    print(f"    Rechunking y={y_start}:{y_end}: {timestr()}")

    try:
        block = source_zarr_ds[year_idx, y_start:y_end, :]  # shape (year, specific y range, all x values in that range)
        rechunk_zarr_ds[year_idx, y_start:y_end, :] = block
        lat_band_end_time = time.time()
        print(f"      Rechunked block y={y_start}:{y_end} in {round(lat_band_end_time - lat_band_start_time)} seconds: {timestr()}")

    except Exception as e:
        print(f"      Failed to rechunk block y={y_start}:{y_end}: {e}: {timestr()}")


# ──────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────
if __name__ == "__main__":

    # raw_store_url = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_1_0_3_zarr_testing/small_test_zarr_2vars_2yrs/"
    raw_store_url = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_1_0_3_zarr_testing/mega_zarr/standard_model/annual_intervals/4000_pixels/20251028/"
    # raw_store_url = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_1_0_3_1884_chunks/mega_zarr/standard_model/annual_intervals/4000_pixels/20251027/"
    test_start_time = time.time()

    # # Step 1: Create empty Zarr store (only once)
    # initialize_zarr_with_coords(raw_store_url,
    #     dataset_keys,
    #     n_years,
    #     resolution=0.00025,
    #     dtype="float32",
    #     fill_value=np.nan,
    #     chunks=(1, 4000, 4000)
    # )

    # # Connects to Coiled cluster if not running locally and the named cluster exists
    # cluster, client, run_local = connect_to_Coiled_cluster("zarr_testing", False)

    # # Step 3: Compute region indices
    # lat0, lon0 = latlon_to_indices(target_box["lat_max"], target_box["lon_min"])
    # lat1, lon1 = latlon_to_indices(target_box["lat_min"], target_box["lon_max"])
    #
    #
    # # Step 4: Populate zarr: Submit write tasks for all datasets + all years
    # futures = []
    # for dataset_key in dataset_keys:
    #     for year_idx in range(n_years):
    #         f = client.submit(write_chunk, dataset_key, year_idx, lat0, lat1, lon0, lon1, raw_store_url, pure=False)
    #         futures.append(f)
    #
    # results = client.gather(futures)
    # for r in results:
    #     print(r)
    #
    #
    # # Step 5: Clean _FillValue in populated zarr
    # ### Need to remove _FillValue attribute in zarr because it's being encoded in some way that is incompatible with xarray while using zarr v3,
    # ### per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68f984c6-9aa0-8327-a910-5ad9a8d170fc.
    # ### There doesn't seem to be a way to create the zarr with a correctly encoded _FillValue in the first place,
    # ### hence this fix after the fact.
    #
    # fs = fsspec.filesystem("s3", anon=False)
    # source_mapper = fs.get_mapper(raw_store_url)
    #
    # # Open Zarr group in read/write mode
    # z = zarr.open_group(store=source_mapper, mode="r+")
    # print(z["carbon_density__AGC__MgC"].filters)  # Should include compressor info
    #
    # # Loop through all arrays
    # for key in z.array_keys():
    #     arr = z[key]
    #     if "_FillValue" in arr.attrs:
    #         print(f"🔧 Removing _FillValue from {key}")
    #         del arr.attrs["_FillValue"]
    #
    # print("✅ Cleaned _FillValue from Zarr metadata.")
    #
    # arr = z["carbon_density__AGC__MgC"]
    # attrs = dict(arr.attrs)
    # print(f"Attributes: {attrs}: {timestr()}")
    # print("Type of _FillValue:", type(attrs.get("_FillValue")))


    # Step 6: Check stats for chunk x dataset x year combinations
    fs = fsspec.filesystem("s3", anon=False)
    source_mapper = fs.get_mapper(raw_store_url)
    ds = xr.open_zarr(source_mapper, consolidated=False)
    # print(ds)
    # print(ds.coords)
    print("y range:", ds.y.values.min(), ds.y.values.max())
    print("x range:", ds.x.values.min(), ds.x.values.max())

    check_region_stats(raw_store_url, "carbon_density__AGC__MgC", 0)
    check_region_stats(raw_store_url, "carbon_density__BGC__MgC", 0)
    check_region_stats(raw_store_url, "carbon_density__AGC__MgC", 1)
    check_region_stats(raw_store_url, "carbon_density__BGC__MgC", 1)
    # check_region_min_max("flux_NEE", 2)
    #
    #
    # # Step 7: Rechunk to 10000x10000
    #
    # # Rechunked mega-zarr path
    # rechunk_url = raw_store_url.rstrip("/") + "_rechunked/"
    #
    # # Creates a metadata-only rechunked zarr that will be populated with rechunked data copied in
    # initialize_zarr_with_coords(
    #     rechunk_url,
    #     dataset_keys,
    #     test_n_years,
    #     resolution=0.00025,
    #     dtype="float32",
    #     fill_value=np.nan,
    #     chunks=(1, 10000, 10000)
    # )
    #
    # # Number of latitudinal pixel bands that will be processed in each task.
    # # Can change this to more completely use worker memory.
    # y_block_size = 3500
    #
    # fs = fsspec.filesystem("s3", anon=False)
    # source_mapper = fs.get_mapper(raw_store_url)
    # rechunk_mapper = fs.get_mapper(rechunk_url)
    #
    # # Iterates through datasets
    # rechunk_start_time = time.time()
    # for dataset in dataset_keys:
    #
    #     # Opens original zarr group, then specific dataset
    #     source_zarr_group = zarr.open_group(source_mapper, mode="r")
    #     source_dataset = source_zarr_group[dataset]
    #
    #     # Dataset properties
    #     shape = source_dataset.shape
    #     dtype = source_dataset.dtype
    #     source_n_years, ny, nx = shape
    #
    #     print(f"  Dataset: {dataset}; shape: {shape};  dtype: {dtype}: {timestr()}")
    #
    #     # Opens pre-created rechunked zarr group
    #     zarr_out = zarr.open_group(rechunk_mapper, mode="a")
    #
    #     # Rechunked dataset
    #     rechunk_dataset = zarr_out[dataset]
    #
    #     dataset_start_time = time.time()
    #
    #     # Iterates by year
    #     # for year_idx in range(source_n_years):
    #     for year_idx in range(test_n_years):
    #         year_start_time = time.time()
    #         print(f"  Year {year_idx}: {timestr()}")
    #
    #         # Iterates by latitudinal band
    #         futures = []
    #         for y_start in range(0, ny, y_block_size):
    #
    #             future = client.submit(rechunk_by_lat_band, year_idx, y_start, y_block_size, ny, source_dataset, rechunk_dataset)
    #             futures.append(future)
    #
    #         client.gather(futures)
    #         year_end_time = time.time()
    #         print(f"     {dataset} for year {year_idx} took {round(year_end_time - year_start_time)} seconds: {timestr()}")
    #
    #     dataset_end_time = time.time()
    #     print(f"All years for {dataset} took {round(dataset_end_time - dataset_start_time)} seconds: {timestr()}")
    #
    # rechunk_end_time = time.time()
    # print(f"Rechunking all years for all datasets  took {round(rechunk_end_time - rechunk_start_time)} seconds: {timestr()}")
    #
    #
    # # Step 8: Check stats for chunk x dataset x year combinations in rechunked zarr
    #
    # ds_rechunk = xr.open_zarr(rechunk_mapper, consolidated=False)
    # print(f"rechunked ds: {ds_rechunk}: {timestr()}")
    # print(ds_rechunk.coords)
    #
    # check_region_stats(rechunk_url, "carbon_density__AGC__MgC", 0)
    # check_region_stats(rechunk_url, "carbon_density__BGC__MgC", 0)
    # check_region_stats(rechunk_url, "carbon_density__AGC__MgC", 1)
    # check_region_stats(rechunk_url, "carbon_density__BGC__MgC", 1)
    #
    # test_end_time = time.time()
    # print(f"Done with test. Elapsed time {round(test_end_time - test_start_time)} seconds: {timestr()}")
    #
    # # Optional shutdown
    # cluster.shutdown()