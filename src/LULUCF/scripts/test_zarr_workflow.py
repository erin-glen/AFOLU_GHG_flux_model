"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

python -m src.utilities.create_cluster -n 1 -m 4 -cn zarr_testing
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
    "carbon_density__BGC__MgC",
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




def rechunk_variable_year_by_year(raw_store_url, rechunk_url, var_names=None, years=None, chunk_size=(10000, 10000)):

    # S3 mappers
    fs = fsspec.filesystem("s3", anon=False)
    source_mapper = fs.get_mapper(raw_store_url)
    target_mapper = fs.get_mapper(rechunk_url)

    print(f"🔓 Opening full Zarr dataset: {timestr()}")
    ds = xr.open_zarr(source_mapper, consolidated=False, zarr_format=3)

    if var_names is None:
        var_names = list(ds.data_vars)

    if years is None:
        years = list(ds.year.values)

    # Zstd compression
    compressor = {
        "name": "zstd",
        "configuration": {"level": 3}
    }

    # Open target Zarr for checking existing vars
    target_z = zarr.open_group(target_mapper, mode="r+")
    existing_vars = set(target_z.array_keys())

    for var in var_names:
        print(f"\n📦 Rechunking variable: {var}")
        for year_idx in years:
            print(f"🕒 Year index {year_idx}: {timestr()}")

            # Slice and expand to keep "year" dim
            da_slice = ds[var].isel(year=year_idx).expand_dims(year=[year_idx])

            # Rechunk spatial dims
            da_rechunked = da_slice.chunk({"y": chunk_size[0], "x": chunk_size[1]})
            da_rechunked.encoding.pop("chunks", None)

            # Build encoding dict only if variable does not yet exist
            if var not in existing_vars and year_idx == years[0]:
                encoding = {var: {"compressor": compressor}}
                mode = "a"  # First write
                print(f"🆕 Creating variable {var} with compression")
            else:
                encoding = None  # Don't pass encoding again
                mode = "a"

            # Write to Zarr
            da_rechunked.to_dataset(name=var).to_zarr(
                store=target_mapper,
                mode=mode,
                zarr_format=3,
                encoding=encoding,
                append_dim="year",
            )

            print(f"✅ Wrote {var} year={year_idx} to {rechunk_url}: {timestr()}")



@delayed
def write_slice_to_zarr(dataarray, var, store, year_index):
    dataarray.encoding.pop("chunks", None)
    dataarray = dataarray.expand_dims("year")
    return dataarray.to_dataset(name=var).to_zarr(
        store=store,
        mode="a",
        zarr_format=3,
        append_dim="year"
    )

def rechunk_variable_with_delayed(raw_store_url, rechunk_url, var_names=None, years=None):
    fs = fsspec.filesystem("s3", anon=False)
    source_mapper = fs.get_mapper(raw_store_url)
    target_mapper = fs.get_mapper(rechunk_url)

    ds = xr.open_zarr(source_mapper, consolidated=False, zarr_format=3)

    if var_names is None:
        var_names = list(ds.data_vars)
    if years is None:
        years = list(ds.year.values)

    tasks = []

    for var in var_names:
        for year_idx in years:

            if "year" not in ds[var].dims:
                print(f"⛔ Skipping variable {var} — no 'year' dimension found.")
                continue

            da_slice = ds[var].isel(year=year_idx).copy(deep=False)

            da_slice.attrs = {}
            da_slice = da_slice.reset_coords(drop=True)
            da_rechunked = da_slice.chunk({"y": 10000, "x": 10000})

            task = write_slice_to_zarr(da_rechunked, var, target_mapper, year_idx)
            tasks.append(task)

    print(f"⏳ Submitting {len(tasks)} delayed tasks...")
    dask.compute(*tasks)
    print("✅ All chunks written.")



def write_chunk_delayed(da_slice, target_mapper, var, compressor):
    # Clean encoding and write
    da_slice.encoding.pop("chunks", None)
    return da_slice.to_dataset(name=var).to_zarr(
        store=target_mapper,
        mode="a",
        zarr_format=3,
        append_dim="year",
        encoding={var: {"compressor": compressor}},
    )

def rechunk_variable_with_delayed_one_by_one(
    raw_store_url,
    rechunk_url,
    var_names=None,
    years=None,
    chunk_size=(10000, 10000),
):

    fs = fsspec.filesystem("s3", anon=False)
    source_mapper = fs.get_mapper(raw_store_url)
    target_mapper = fs.get_mapper(rechunk_url)

    print(f"🔓 Opening full Zarr dataset: {timestr()}")
    ds = xr.open_zarr(source_mapper, consolidated=False, zarr_format=3)

    # Clean _FillValue to avoid dtype errors
    for var in ds.data_vars:
        if "_FillValue" in ds[var].attrs:
            print(f"🧹 Removing _FillValue from {var}: {timestr()}")
            ds[var].attrs.pop("_FillValue")

    if var_names is None:
        var_names = list(ds.data_vars)

    if years is None:
        years = list(ds.year.values)

    compressor = {
        "name": "zstd",
        "configuration": {"level": 3}
    }

    for var in var_names:
        if "year" not in ds[var].dims:
            print(f"⛔ Skipping variable {var} — no 'year' dimension found.")
            continue

        print(f"\n📦 Rechunking variable: {var}: {timestr()}")
        for year_idx in years:
            print(f"🕒 Year index {year_idx}: {timestr()}")

            try:
                # Extract slice and break graph dependency
                da_slice = ds[var].isel(year=year_idx).expand_dims(year=[year_idx])
                da_slice = da_slice.chunk({"y": chunk_size[0], "x": chunk_size[1]})
                da_slice = da_slice.persist()  # ✅ Crucial to reduce graph size

                # Clean _FillValue before writing
                da_slice.attrs.pop("_FillValue", None)
                da_slice.encoding.pop("_FillValue", None)

                # Delayed write
                delayed_task = dask.delayed(write_chunk_delayed)(
                    da_slice, target_mapper, var, compressor
                )

                with ProgressBar():
                    dask.compute(delayed_task)

                print(f"✅ Wrote {var} year={year_idx} to {rechunk_url}: {timestr()}")

            except Exception as e:
                print(f"❌ Failed writing {var} year={year_idx}: {e}")



def write_chunk_direct(da_slice, target_mapper, var, compressor, first_year):
    da_slice.encoding.clear()  # Prevent stale encoding like _FillValue

    encoding = {}
    if first_year:
        encoding[var] = {"compressor": compressor}

    return (
        da_slice
        .to_dataset(name=var)
        .to_zarr(
            store=target_mapper,
            mode="a",
            zarr_format=3,
            append_dim="year",
            encoding=encoding if first_year else None,
        )
    )

def rechunk_variable_with_futures(
    client,
    raw_store_url,
    rechunk_url,
    var_names=None,
    years=None,
    chunk_size=(10000, 10000),
):
    fs = fsspec.filesystem("s3", anon=False)
    source_mapper = fs.get_mapper(raw_store_url)
    target_mapper = fs.get_mapper(rechunk_url)

    print(f"🔓 Opening full Zarr dataset: {timestr()}")
    ds = xr.open_zarr(source_mapper, consolidated=False, zarr_format=3)

    if var_names is None:
        var_names = list(ds.data_vars)

    if years is None:
        years = list(ds.year.values)

    compressor = {"name": "zstd", "configuration": {"level": 3}}

    for var in var_names:
        if "year" not in ds[var].dims:
            print(f"⛔ Skipping variable {var} — no 'year' dimension found.")
            continue

        print(f"\n📦 Rechunking variable: {var}")
        for year_idx in years:
            print(f"🕒 Year index {year_idx}: {timestr()}")

            try:
                da_slice = (
                    ds[var]
                    .isel(year=year_idx)
                    .expand_dims(year=[year_idx])
                    .chunk({"y": chunk_size[0], "x": chunk_size[1]})
                )

                future = client.submit(
                    write_chunk_direct,
                    da_slice,
                    target_mapper,
                    var,
                    compressor,
                    year_idx == 0,  # only pass encoding on year=0
                    pure=False,
                )
                future.result()

                print(f"✅ Wrote {var} year={year_idx} to {rechunk_url}: {timestr()}")

            except Exception as e:
                print(f"❌ Failed writing {var} year={year_idx}: {e}")



def rechunk_variable_with_compute_per_slice(raw_store_url, rechunk_url, var_names=None, years=None, chunk_size=(10000, 10000)):
    # === Connect to S3 ===
    fs = fsspec.filesystem("s3", anon=False)
    source_mapper = fs.get_mapper(raw_store_url)
    target_mapper = fs.get_mapper(rechunk_url)

    print(f"🔓 Opening source dataset: {timestr()}")
    ds = xr.open_zarr(source_mapper, consolidated=False, zarr_format=3)

    # Clean _FillValue to avoid dtype issues
    for var in ds.data_vars:
        if "_FillValue" in ds[var].attrs:
            print(f"🧹 Removing _FillValue from {var}")
            ds[var].attrs.pop("_FillValue")

    if var_names is None:
        var_names = list(ds.data_vars)

    if years is None:
        years = list(ds.year.values)

    for var in var_names:
        if "year" not in ds[var].dims:
            print(f"⛔ Skipping variable {var} — no 'year' dimension found.")
            continue

        print(f"\n📦 Rechunking variable: {var}")
        for year_idx in years:
            print(f"🕒 Year index {year_idx}: {timestr()}")
            try:
                # Load 2D slice into memory
                print(f"Slicing {year_idx} for {var}: {timestr()}")
                slice_data = ds[var].isel(year=year_idx).compute()

                # Wrap into 3D DataArray
                print(f"Wrapping {year_idx} for {var}: {timestr()}")
                da = xr.DataArray(
                    slice_data.data[np.newaxis, :, :],
                    dims=("year", "y", "x"),
                    coords={
                        "year": [year_idx],
                        "y": slice_data.y,
                        "x": slice_data.x,
                    },
                    name=var,
                    attrs={"grid_mapping": "spatial_ref"},
                ).chunk({"y": chunk_size[0], "x": chunk_size[1]})

                # Write to Zarr (append one slice)
                print(f"Writing {year_idx} for {var}: {timestr()}")
                with ProgressBar():
                    da.to_dataset().to_zarr(
                        store=target_mapper,
                        mode="a",
                        append_dim="year",
                        zarr_format=3,
                    )

                print(f"✅ Wrote {var} year={year_idx} to {rechunk_url}: {timestr()}")

            except Exception as e:
                print(f"❌ Failed writing {var} year={year_idx}: {e}")



def rechunk_variable_whole(
    raw_store_url,
    rechunk_url,
    var_name=None,
    chunk_size=(1, 10000, 10000),
):
    fs = fsspec.filesystem("s3", anon=False)
    source_mapper = fs.get_mapper(raw_store_url)
    target_mapper = fs.get_mapper(rechunk_url)

    print(f"\n📦 Rechunking variable: {var_name}")
    ds = xr.open_zarr(source_mapper, consolidated=False)

    # Optional: remove problematic attributes like _FillValue
    if "_FillValue" in ds[var_name].attrs:
        ds[var_name].attrs.pop("_FillValue")

    # Slice out only this variable, rechunk, write
    da = ds[var_name].chunk({"year": chunk_size[0], "y": chunk_size[1], "x": chunk_size[2]})
    da.encoding.pop("chunks", None)

    print(f"  🔍 Shape: {da.shape}")
    print(f"  🔍 Writing rechunked variable to store...")

    # Use to_dataset for multi-var compatibility
    da.to_dataset(name=var_name).to_zarr(
        store=target_mapper,
        mode="a",
        append_dim="year",
        compute=True,  # Ensures execution
        zarr_format=3
    )

    print(f"✅ Wrote variable: {var_name}")



def rechunk_variable_in_y_blocks(
        raw_store_url,
        rechunk_url,
        var_name,
        chunk_size=(1, 10_000, 10_000),
        y_block_size=100_000,
):
    """Rechunk a single variable by slicing along 'y' in blocks, loading each into memory."""

    print(f"\n📦 Rechunking variable: {var_name}")

    fs = fsspec.filesystem("s3", anon=False)
    source_mapper = fs.get_mapper(raw_store_url)
    target_mapper = fs.get_mapper(rechunk_url)

    # Open source dataset (assumes Zarr v3)
    ds = xr.open_zarr(source_mapper, consolidated=False, zarr_format=3)

    if var_name not in ds:
        print(f"⛔ Variable {var_name} not found.")
        return

    da = ds[var_name]

    if "_FillValue" in da.attrs:
        print(f"🔧 Removing _FillValue from {var_name}")
        da.attrs.pop("_FillValue")

    # Extract shape info
    n_years = da.sizes.get("year", 1)
    y_total = da.sizes["y"]
    x_total = da.sizes["x"]

    compressor = {"name": "zstd", "configuration": {"level": 3}}

    # Loop over y in blocks (e.g. 100k rows)
    for y_start in range(0, y_total, y_block_size):
        y_end = min(y_start + y_block_size, y_total)
        print(f"  🧩 Rechunking y={y_start}:{y_end}...")

        try:
            # Read block into memory
            block = da.isel(y=slice(y_start, y_end)).compute()

            # Wrap back in DataArray with full dims
            new_da = xr.DataArray(
                block.data,
                dims=("year", "y", "x") if "year" in da.dims else ("y", "x"),
                coords={
                    "year": block.coords.get("year", None),
                    "y": block.coords["y"],
                    "x": block.coords["x"],
                },
                name=var_name,
                attrs=block.attrs,
            )

            # Rechunk to 10k x 10k blocks (or smaller, depending on slice)
            rechunked = new_da.chunk(dict(
                year=chunk_size[0] if "year" in new_da.dims else None,
                y=chunk_size[1],
                x=chunk_size[2],
            ))

            # Clean stale encoding
            rechunked.encoding.pop("chunks", None)

            # Write with append
            rechunked.to_dataset(name=var_name).to_zarr(
                store=target_mapper,
                mode="a",
                append_dim="y",
                encoding={var_name: {"compressor": compressor}},
                compute=True,
                zarr_format=3,
            )

            print(f"    ✅ Wrote y={y_start}:{y_end}")

        except Exception as e:
            print(f"    ❌ Failed y={y_start}:{y_end}: {e}")

    print(f"✅ Finished rechunking {var_name}: {timestr()}")



def rechunk_variable_manual_blocks(
    raw_store_url,
    rechunk_url,
    var_name,
    chunk_size=(1, 10_000, 10_000),
    y_block_size=100_000,
):
    print(f"\n📦 Rechunking variable: {var_name}")

    fs = fsspec.filesystem("s3", anon=False)
    source_mapper = fs.get_mapper(raw_store_url)
    target_mapper = fs.get_mapper(rechunk_url)

    # Direct Zarr group access (no xarray, no Dask)
    source_zg = zarr.open_group(source_mapper, mode="r")
    if var_name not in source_zg:
        print(f"⛔ Variable {var_name} not found.")
        return

    arr = source_zg[var_name]

    shape = arr.shape
    dtype = arr.dtype
    has_year = arr.ndim == 3
    n_years, ny, nx = shape if has_year else (1, *shape)

    print(f"  🔍 Shape: {shape}  dtype: {dtype}")

    # Create output Zarr group if needed
    zarr_out = zarr.open_group(target_mapper, mode="a")

    # Create array in output store
    if var_name not in zarr_out:
        chunks = chunk_size if has_year else chunk_size[1:]
        zarr_out.create_dataset(
            name=var_name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=Zstd(level=3),
            overwrite=False,
        )
        print(f"  ✅ Created dataset in output store with chunks={chunks}")

    out_arr = zarr_out[var_name]

    # Loop through y blocks manually
    for y_start in range(0, ny, y_block_size):
        y_end = min(y_start + y_block_size, ny)
        print(f"  🧩 Processing y={y_start}:{y_end}")

        try:
            if has_year:
                data = arr[:, y_start:y_end, :]  # (year, y, x)
            else:
                data = arr[y_start:y_end, :]     # (y, x)

            # Write to output Zarr
            if has_year:
                out_arr[:, y_start:y_end, :] = data
            else:
                out_arr[y_start:y_end, :] = data

            print(f"    ✅ Wrote block y={y_start}:{y_end}")
        except Exception as e:
            print(f"    ❌ Failed block y={y_start}:{y_end}: {e}")

    print(f"✅ Finished writing {var_name}")



def rechunk_variable_by_year_and_yblock(
    raw_store_url,
    rechunk_url,
    var_name,
    chunk_size=(1, 10_000, 10_000),
    y_block_size=10_000,  # much smaller than before
):
    print(f"\n📦 Rechunking variable: {var_name}")

    fs = fsspec.filesystem("s3", anon=False)
    source_mapper = fs.get_mapper(raw_store_url)
    target_mapper = fs.get_mapper(rechunk_url)

    source_zg = zarr.open_group(source_mapper, mode="r")
    arr = source_zg[var_name]

    shape = arr.shape
    dtype = arr.dtype
    n_years, ny, nx = shape

    print(f"  🔍 Shape: {shape}  dtype: {dtype}")

    # Create output array if not present
    zarr_out = zarr.open_group(target_mapper, mode="a")
    if var_name not in zarr_out:
        zarr_out.create_dataset(
            name=var_name,
            shape=shape,
            chunks=chunk_size,
            dtype=dtype,
            compressor=Zstd(level=3),
            overwrite=False,
        )
        print(f"  ✅ Created dataset in output store")

    out_arr = zarr_out[var_name]

    for year_idx in range(n_years):
        print(f"  🗓️  Year {year_idx}")
        for y_start in range(0, ny, y_block_size):
            y_end = min(y_start + y_block_size, ny)
            print(f"    🧩 y={y_start}:{y_end}")
            try:
                block = arr[year_idx, y_start:y_end, :]  # shape (y_block, x)
                out_arr[year_idx, y_start:y_end, :] = block
                print(f"      ✅ Wrote block y={y_start}:{y_end}")
            except Exception as e:
                print(f"      ❌ Failed block y={y_start}:{y_end}: {e}")

    print(f"✅ Finished writing {var_name}")



# ──────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────
if __name__ == "__main__":

    # raw_store_url = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_1_0_3_zarr_testing/small_test_zarr_2vars_2yrs/"
    # raw_store_url = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_1_0_3_zarr_testing/mega_zarr/standard_model/annual_intervals/4000_pixels/20251027/"
    raw_store_url = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_1_0_3_1884_chunks/mega_zarr/standard_model/annual_intervals/4000_pixels/20251027/"
    start_time = time.time()

    # # Step 1: Create empty Zarr store (only once)
    # initialize_zarr_with_coords(raw_store_url, dataset_keys, n_years)

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


    # Step 6: Validate one region
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

    sys.quit()


    # Step 7: Rechunk to 10000x10000-- one dataset-year at a time

    # === Create paths ===
    rechunk_url = raw_store_url.rstrip("/") + "_rechunked/"

    initialize_zarr_with_coords(
        rechunk_url,
        dataset_keys,
        n_years,
        resolution=0.00025,
        dtype="float32",
        fill_value=np.nan,
        chunks=(1, 10000, 10000)
    )

    # # Graph size of 67 MB
    # rechunk_variable_year_by_year(
    #     raw_store_url=raw_store_url,
    #     rechunk_url=rechunk_url,
    #     var_names=["carbon_density__AGC__MgC"],  # or None for all variables
    #     years=[0, 1, 2],  # or None for all years
    #     chunk_size=(10000, 10000)
    # )

    # # Graph size of 116 MB
    # rechunk_variable_with_delayed(
    #     raw_store_url=raw_store_url,
    #     rechunk_url=rechunk_url
    # )

    # Graph size of 44 MB
    # # Failed after all tasks completed. Graph size of about 44 MB.
    # rechunk_variable_with_delayed_one_by_one(
    #     raw_store_url=raw_store_url,
    #     rechunk_url=rechunk_url,
    #     var_names=["carbon_density__AGC__MgC"],  # or None for all
    #     years=[0],  # or None for all
    #     chunk_size=(10000, 10000)
    # )

    # # Failed early on, but also graph size of 45 MB
    # rechunk_variable_with_futures(
    #     client,
    #     raw_store_url=raw_store_url,
    #     rechunk_url=rechunk_url,
    #     var_names=None,
    #     years=None,  # Or all years
    #     chunk_size=(10000, 10000),
    # )

    # # Graph size of 23 MB but fails at slicing by year-variable
    # rechunk_variable_with_compute_per_slice(
    #     raw_store_url=raw_store_url,
    #     rechunk_url=rechunk_url,
    #     var_names=None,
    #     years=None,  # Or all years
    #     chunk_size=(10000, 10000),
    # )

    # # Graph of 276 MB. Failed soon after reporting that, before starting and Dask tasks
    # rechunk_variable_whole(
    #     raw_store_url=raw_store_url,
    #     rechunk_url=rechunk_url,
    #     var_name="carbon_density__AGC__MgC",
    #     chunk_size=(1, 10000, 10000),
    # )

    # # Step 7: Rechunk to 10000x10000-- all datasets and years together
    #
    # # Paths
    # rechunk_url = raw_store_url.rstrip("/") + "_rechunked/"
    #
    # # === Create S3 mappers ===
    # fs = fsspec.filesystem("s3", anon=False)
    # source_mapper = fs.get_mapper(raw_store_url)
    # target_mapper = fs.get_mapper(rechunk_url)
    #
    # # === Open Zarr v3 dataset ===
    # print(f"Opening zarr dataset for rechunking: {timestr()}")
    # ds = xr.open_zarr(source_mapper, consolidated=False, chunks={})
    #
    # # === Rechunk in memory ===
    # print(f"Rechunking in memory: {timestr()}")
    # rechunked = ds.chunk({"year": 1, "y": 10000, "x": 10000})
    #
    # # === Clean stale encoding to avoid alignment errors ===
    # print(f"Cleaning rechunked output: {timestr()}")
    # for var in rechunked.data_vars:
    #     rechunked[var].encoding.pop("chunks", None)
    #
    # # === Write to new Zarr v3 store on S3 ===
    #
    # compressor = {
    #     "name": "zstd",
    #     "configuration": {"level": 3}
    # }
    #
    # encoding = {
    #     var: {"compressor": compressor}
    #     for var in rechunked.data_vars
    # }
    #
    # print(f"Writing rechunked output to s3: {timestr()}")
    # rechunked.to_zarr(
    #     store=target_mapper,
    #     mode="w",
    #     encoding=encoding,
    #     zarr_format=3
    # )
    #
    # print(f"✅ Rechunked dataset written to {rechunk_url}: {timestr()}")

    # Test the rechunking

    # # Ran tasks but failed after. 34 MB graph.
    # rechunk_variable_in_y_blocks(
    #     raw_store_url=raw_store_url,
    #     rechunk_url=,
    #     var_name="carbon_density__AGC__MgC",
    #     chunk_size=(1, 10_000, 10_000),
    #     y_block_size=100_000,
    # )

    # # Used too much memory. Didn't pursue it.
    # rechunk_variable_manual_blocks(
    #     raw_store_url=raw_store_url,
    #     rechunk_url=rechunk_url,
    #     var_name="carbon_density__AGC__MgC",
    #     chunk_size=(1, 10_000, 10_000),
    #     y_block_size=100_000,
    # )

    rechunk_variable_by_year_and_yblock(
        raw_store_url=raw_store_url,
        rechunk_url=rechunk_url,
        var_name="carbon_density__AGC__MgC",
        chunk_size=(1, 10_000, 10_000),  # final Zarr chunking
        y_block_size=2_000  # controls memory size per read
    )


    fs = fsspec.filesystem("s3", anon=False)
    rechunk_mapper = fs.get_mapper(rechunk_url)
    ds_rechunk = xr.open_zarr(rechunk_mapper, consolidated=False)
    print(f"rechunked ds: {ds_rechunk}: {timestr()}")
    print(ds_rechunk.coords)

    # Open Zarr group in read/write mode
    z = zarr.open_group(store=rechunk_mapper, mode="r+")
    print(z["carbon_density__AGC__MgC"].filters)  # Should include compressor info

    check_region_stats(rechunk_url, "carbon_density__AGC__MgC", 0)
    check_region_stats(rechunk_url, "carbon_density__BGC__MgC", 0)
    check_region_stats(rechunk_url, "carbon_density__AGC__MgC", 1)
    check_region_stats(rechunk_url, "carbon_density__BGC__MgC", 1)

    end_time = time.time()
    print(f"Done with test. Elapsed time {round(end_time - start_time)} seconds: {timestr()}")






    # # Step 7: Rechunk to 10000x10000
    #
    # # Paths
    # source_url = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_1_0_3_zarr_testing/small_test_zarr/"
    # rechunk_url = source_url.rstrip("/") + "_rechunked/"
    # temp_url = source_url.rstrip("/") + "_temp/"
    #
    # # FS mappers
    # fs = fsspec.filesystem("s3", anon=False)
    # source_mapper = fs.get_mapper(source_url)
    # rechunk_mapper = fs.get_mapper(rechunk_url)
    # temp_mapper = fs.get_mapper(temp_url)
    #
    # # Open the dataset
    # ds_raw = xr.open_zarr(source_mapper, consolidated=False)
    #
    # # Select only variables that have ("year", "y", "x") dimensions
    # vars_to_rechunk = [v for v in ds_raw.data_vars if ds_raw[v].dims == ("year", "y", "x")]
    # print(f"vars_to_rechunk: {vars_to_rechunk}: {timestr()}")
    #
    # # Define new chunk sizes
    # target_chunks = {"year": 1, "y": 10000, "x": 10000}
    #
    # source_subset = xr.Dataset({v: ds_raw[v] for v in vars_to_rechunk})
    # print(f"source_subset: {source_subset}: {timestr()}")
    #
    # # Set up rechunking
    # print(f"Creating rechunk plan: {timestr()}")
    # rechunk_plan = rechunk(
    #     source=source_subset,
    #     target_chunks=target_chunks,
    #     max_mem="2GB",  # You can adjust this depending on your available RAM
    #     target_store=rechunk_mapper,
    #     temp_store=temp_mapper,
    # )
    #
    # print(f"Executing rechunk plan: {timestr()}")
    # # Execute the plan
    # rechunk_plan.execute()
    #
    # print(f"✅ Rechunking complete: {timestr()}")
    #
    # fs = fsspec.filesystem("s3", anon=False)
    # mapper = fs.get_mapper(rechunk_url)
    # ds_rechunk = xr.open_zarr(mapper, consolidated=False)
    # print(f"rechunked ds: {ds_rechunk}")
    #
    # for v in ds_rechunk.data_vars:
    #     print(f"{v} has {da.core.numblocks(ds_rechunk[v].data)} blocks")
    #
    # # Print chunks for each variable
    # for var in ds_rechunk.data_vars:
    #     print(f"{var}: chunks={ds_rechunk[var].data.chunks}")
    #
    # check_region_min_max(rechunk_url, "carbon_density_AGC", 0)
    # check_region_min_max(rechunk_url, "carbon_density_BGC", 0)
    # check_region_min_max(rechunk_url, "carbon_density_AGC", 1)
    # check_region_min_max(rechunk_url, "carbon_density_BGC", 1)
    #
    # # --- Cleanup temp store ---
    # print(f"🧹 Deleting temporary store: {temp_url}")
    # fs.rm(temp_url, recursive=True)
    # print(f"✅ Temp store deleted at {timestr()}")






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
