"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

python -m src.utilities.create_cluster -n 1 -t 1 -m 4 -cn zarr_testing
python -m src.LULUCF.scripts.test_zarr_creation -cn zarr_testing
"""

from datetime import datetime
import pytz
import numpy as np
import xarray as xr
import dask.array as da
import zarr
import fsspec
import coiled
from dask.distributed import Client, print

from rechunker import rechunk
import os


# ──────────────────────────────
# CONFIGURATION
# ──────────────────────────────

dataset_keys = [
    "carbon_density_AGC",
    "carbon_density_BGC",
    # "flux_NEE",
    # "flux_FIRE",
    # "forest_mask",
]

n_years = 2
resolution = 0.00025
lat_size = int(180 / resolution)   # 720000
lon_size = int(360 / resolution)   # 1440000
dtype = "float32"
fill_value = 0.0

# Lat/lon coordinate arrays
lats = np.arange(90.0 - resolution / 2, -90, -resolution)[:lat_size]
lons = np.arange(-180.0 + resolution / 2, 180, resolution)[:lon_size]
years = np.arange(n_years)

# Target region: 49–50N, 10–11E
target_box = {
    "lat_min": 49.0,
    "lat_max": 50.0,
    "lon_min": 10.0,
    "lon_max": 11.0
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


def create_global_output_zarr(store_url, dataset_keys, n_years, lat_size, lon_size, chunks):
    """Create a global Zarr group with multiple datasets (year, lat, lon)."""
    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(store_url)

    # Step 1: Explicitly create a group and write root metadata
    root = zarr.open_group(store=mapper, mode="w")
    root.attrs["description"] = "Global outputs with multiple datasets and years"
    print("✅ Root Zarr group created.")

    # Step 2: Create datasets inside the group
    for key in dataset_keys:
        root.create_dataset(
            name=key,
            shape=(n_years, lat_size, lon_size),
            chunks=chunks,
            dtype=dtype,
            fill_value=fill_value,
            overwrite=True,
        )
        print(f"✅ Created dataset: {key} shape={(n_years, lat_size, lon_size)}")

    print(f"✅ Zarr group created at {store_url}")

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

    # Use Dask arrays filled lazily (no memory blowup)
    data_vars = {}
    for key in dataset_keys:
        print(f"{key}: {timestr()}")
        dask_data = da.full(
            (n_years, lat_size, lon_size),
            fill_value,
            dtype=dtype,
            chunks=chunks
        )

        data_vars[key] = xr.DataArray(
            dask_data,
            dims=("year", "y", "x"),
            coords={"year": years, "y": lats, "x": lons},
            name=key,
            attrs={"grid_mapping": "spatial_ref"},
        )

    # Construct dataset
    print(f"Constructing dataset: {timestr()}")
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
        attrs={
            "grid_mapping_name": "latitude_longitude",
            "epsg_code": 4326,
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
        },
    )

    # ds["spatial_ref"].attrs = spatial_attrs
    print(ds)

    # Write metadata only (lazy)
    print(f"Writing metadata: {timestr()}")
    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(store_url)
    ds.to_zarr(mapper, mode="w", compute=False, consolidated=False)

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


def check_region_min_max(store_url, dataset_key, year_idx):
    """Check min/max of the region written."""
    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(store_url)
    z = zarr.open(mapper, mode="r")

    lat0, lon0 = latlon_to_indices(target_box["lat_max"], target_box["lon_min"])
    lat1, lon1 = latlon_to_indices(target_box["lat_min"], target_box["lon_max"])

    region = z[dataset_key][year_idx, lat0:lat1, lon0:lon1]
    print(f"🔍 {dataset_key} year {year_idx}: min={region.min()}, mean={region.mean()}, max={region.max()}")



# Define writing function
def sanitize_attrs(ds: xr.Dataset) -> xr.Dataset:
    clean_global_attrs = {}
    for k, v in ds.attrs.items():
        if isinstance(v, (np.generic, np.ndarray)):
            clean_global_attrs[k] = v.item()
        elif isinstance(v, np.dtype):
            clean_global_attrs[k] = str(v)
        else:
            clean_global_attrs[k] = v
    ds.attrs = clean_global_attrs

    for var in ds.variables:
        new_attrs = {}
        for k, v in ds[var].attrs.items():
            if isinstance(v, (np.generic, np.ndarray)):
                new_attrs[k] = v.item()
            elif isinstance(v, np.dtype):
                new_attrs[k] = str(v)
            else:
                new_attrs[k] = v
        ds[var].attrs = new_attrs

    return ds



def rechunk_one_year_dataset(var_name, year_idx, chunks, source_url, target_url, first_write=False):

    print(f"Rechunking {var_name} year={year_idx} (first_write={first_write}): {timestr()}")

    fs = fsspec.filesystem("s3", anon=False)
    src = xr.open_zarr(fs.get_mapper(source_url), consolidated=False, chunks={})
    tgt_mapper = fs.get_mapper(target_url)
    print(target_url)
    print(src)
    print(tgt_mapper)

    # one_year = src[[var_name]].sel(year=year_idx).expand_dims("year")
    one_year = src[var_name].isel(year=year_idx).expand_dims("year").to_dataset()
    one_year = one_year.chunk(chunks)
    print(one_year)

    # Clean encoding and attrs
    for v in one_year.data_vars:
        one_year[v].encoding.pop("chunks", None)
        one_year[v].encoding.pop("compressor", None)
    one_year = sanitize_attrs(one_year)
    print(one_year)

    # First write initializes the array
    if first_write:
        mode = "w"
        append_dim = None
    else:
        mode = "a"
        append_dim = "year"

    one_year.to_zarr(
        store=tgt_mapper,
        mode=mode,
        consolidated=False,
        append_dim=append_dim,
    )

    print(f"✅ Wrote {var_name} year={year_idx}: {timestr()}")
    return f"✅ Wrote {var_name} year={year_idx}"


# ──────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────
if __name__ == "__main__":

    raw_store_url = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_1_0_3_zarr_testing/small_test_zarr/"

    # Step 1: Create empty Zarr store (only once)
    initialize_zarr_with_coords(raw_store_url, dataset_keys, n_years)

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = connect_to_Coiled_cluster("zarr_testing", False)

    # Step 3: Compute region indices
    lat0, lon0 = latlon_to_indices(target_box["lat_max"], target_box["lon_min"])
    lat1, lon1 = latlon_to_indices(target_box["lat_min"], target_box["lon_max"])

    # Step 4: Populate zarr: Submit write tasks for all datasets + all years--
    futures = []
    for dataset_key in dataset_keys:
        for year_idx in range(n_years):
            f = client.submit(write_chunk, dataset_key, year_idx, lat0, lat1, lon0, lon1, raw_store_url, pure=False)
            futures.append(f)

    results = client.gather(futures)
    for r in results:
        print(r)

    # Step 5: Validate one region
    check_region_min_max(raw_store_url, "carbon_density_AGC", 0)
    check_region_min_max(raw_store_url, "carbon_density_BGC", 0)
    check_region_min_max(raw_store_url, "carbon_density_AGC", 1)
    check_region_min_max(raw_store_url, "carbon_density_BGC", 1)
    # check_region_min_max("flux_NEE", 2)

    # Step 6: Clean _FillValue in populated zarr
    ### Need to remove _FillValue attribute in zarr because it's being encoded in some way that is incompatible with xarray while using zarr v3,
    ### per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68f984c6-9aa0-8327-a910-5ad9a8d170fc.
    ### There doesn't seem to be a way to create the zarr with a correctly encoded _FillValue in the first place,
    ### hence this fix after the fact.

    fs = fsspec.filesystem("s3", anon=False)
    source_mapper = fs.get_mapper(raw_store_url)

    # Open Zarr group in read/write mode
    z = zarr.open_group(store=source_mapper, mode="r+")

    # Loop through all arrays
    for key in z.array_keys():
        arr = z[key]
        if "_FillValue" in arr.attrs:
            print(f"🔧 Removing _FillValue from {key}")
            del arr.attrs["_FillValue"]

    print("✅ Cleaned _FillValue from Zarr metadata.")

    arr = z["carbon_density_AGC"]
    attrs = dict(arr.attrs)
    print("Attributes:", attrs)
    print("Type of _FillValue:", type(attrs.get("_FillValue")))

    ds = xr.open_zarr(source_mapper, consolidated=False, chunks={})
    print(ds)

    # Step 7: Rechunk to 10000x10000

    # Paths
    source_url = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_1_0_3_zarr_testing/small_test_zarr/"
    target_url = source_url.rstrip("/") + "_rechunked/"
    temp_url = source_url.rstrip("/") + "_temp/"

    # FS mappers
    fs = fsspec.filesystem("s3", anon=False)
    source_mapper = fs.get_mapper(source_url)
    target_mapper = fs.get_mapper(target_url)
    temp_mapper = fs.get_mapper(temp_url)

    # Open the dataset
    ds = xr.open_zarr(source_mapper, consolidated=False)

    # Select only variables that have ("year", "y", "x") dimensions
    vars_to_rechunk = [v for v in ds.data_vars if ds[v].dims == ("year", "y", "x")]
    print(f"vars_to_rechunk: {vars_to_rechunk}: {timestr()}")

    # Define new chunk sizes
    target_chunks = {"year": 1, "y": 10000, "x": 10000}

    source_subset = xr.Dataset({v: ds[v] for v in vars_to_rechunk})

    print(source_subset)
    for v in source_subset.data_vars:
        print(v, type(source_subset[v].data))

    # Set up rechunking
    print(f"Creating rechunk plan: {timestr()}")
    rechunk_plan = rechunk(
        source=source_subset,
        target_chunks=target_chunks,
        max_mem="1GB",  # You can adjust this depending on your available RAM
        target_store=target_mapper,
        temp_store=temp_mapper,
    )

    print(f"Executing rechunk plan: {timestr()}")
    # Execute the plan
    rechunk_plan.execute()

    print(f"✅ Rechunking complete: {timestr()}")

    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(target_url)
    ds = xr.open_zarr(mapper, consolidated=False)
    print(f"rechunked ds: {ds}")

    # Print chunks for each variable
    for var in ds.data_vars:
        print(f"{var}: chunks={ds[var].data.chunks}")

    check_region_min_max(target_url, "carbon_density_AGC", 0)
    check_region_min_max(target_url, "carbon_density_BGC", 0)
    check_region_min_max(target_url, "carbon_density_AGC", 1)
    check_region_min_max(target_url, "carbon_density_BGC", 1)










    #
    # # Define target and temp S3 locations
    # rechunk_store_url = raw_store_url.rstrip("/") + "_rechunked/"
    #
    # # Get mappers for S3 stores
    # target_mapper = fs.get_mapper(rechunk_store_url)
    #
    # # Define target chunk sizes
    # target_chunks = {"year": 1, "y": 10_000, "x": 10_000}
    # # target_chunks = (1, 10000, 10000)
    #
    # # Open the source Zarr dataset
    # ds = xr.open_zarr(source_mapper, consolidated=False, chunks={})
    #
    # vars_to_rechunk = [v for v in ds.data_vars if "year" in ds[v].dims]
    # print(f"🧩 Variables to rechunk: {vars_to_rechunk}: {timestr()}")
    #
    # # Step 1: First writes (serial, to initialize)
    # for var in vars_to_rechunk:
    #     print(f"🪶 Initializing Zarr for {var} (year=0): {timestr()}")
    #     rechunk_one_year_dataset(
    #         var_name=var,
    #         year_idx=0,
    #         chunks=target_chunks,
    #         source_url=raw_store_url,
    #         target_url=rechunk_store_url,
    #         first_write=True,
    #     )
    #
    #     fs = fsspec.filesystem("s3", anon=False)
    #     mapper = fs.get_mapper(rechunk_store_url)
    #     ds = xr.open_zarr(mapper, consolidated=False)
    #
    #     # Print chunks for each variable
    #     for var in ds.data_vars:
    #         print(f"{var}: chunks={ds[var].data.chunks}")
    #
    #
    # fs = fsspec.filesystem("s3", anon=False)
    # mapper = fs.get_mapper(rechunk_store_url)
    # ds = xr.open_zarr(mapper, consolidated=False)
    #
    # # Print chunks for each variable
    # for var in ds.data_vars:
    #     print(f"{var}: chunks={ds[var].data.chunks}")
    #
    # # # Step 2: Parallel appends (remaining years)
    # # print(f"Rechunking later years for datasets: {timestr()}")
    # # futures = []
    # # for var in vars_to_rechunk:
    # #     for i, year in enumerate(ds.year.values):
    # #         if i == 0:
    # #             continue  # skip the first year (already written)
    # #         fut = client.submit(
    # #             rechunk_one_year_dataset,
    # #             var_name=var,
    # #             year_idx=int(year),
    # #             chunks=target_chunks,
    # #             source_url=raw_store_url,
    # #             target_url=rechunk_store_url,
    # #             first_write=False,
    # #         )
    # #         futures.append(fut)
    # #
    # # results = client.gather(futures)
    # # for r in results:
    # #     print(r)
    # #
    # # print(f"✅ All dataset-year chunks written: {timestr()}")
    #
    # fs = fsspec.filesystem("s3", anon=False)
    # mapper = fs.get_mapper(rechunk_store_url)
    # ds = xr.open_zarr(mapper, consolidated=False)
    #
    # # Print chunks for each variable
    # for var in ds.data_vars:
    #     print(f"{var}: chunks={ds[var].data.chunks}")
    #
    # # Step 5: Validate one region
    # check_region_min_max(rechunk_store_url, "carbon_density_AGC", 0)
    # check_region_min_max(rechunk_store_url, "carbon_density_BGC", 0)
    # # check_region_min_max(rechunk_store_url, "carbon_density_AGC", 1)
    # # check_region_min_max(rechunk_store_url, "carbon_density_BGC", 1)
