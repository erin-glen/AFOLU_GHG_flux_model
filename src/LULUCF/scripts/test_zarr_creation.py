import fsspec
import zarr
import numpy as np

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import numba_utilities as nu
from src.utilities import universal_utilities as uu
from src.utilities import resize_cluster

# Configuration
resolution = 0.00025
dtype = "float32"
fill_value = 0.0
chunks = (1000, 1000)
shape = (int(180 / resolution), int(360 / resolution))  # (720000, 1440000)

# Target region: 49–50N, 10–11E
target_box = {
    "lat_min": 49.0,
    "lat_max": 50.0,
    "lon_min": 10.0,
    "lon_max": 11.0
}

# Zarr store path on S3
store_url = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_1_0_2_zarr_testing/global_test_0.00025.zarr"

def create_empty_zarr():
    """Create Zarr metadata only (no data) for global float32 grid."""
    print("Creating empty Zarr metadata only...")

    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(store_url)

    # Create metadata-only array (no data is written)
    zarr.create(
        store=mapper,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        fill_value=fill_value,
        overwrite=True
    )

    print("Zarr metadata created:", store_url)


def latlon_to_indices(lat, lon):
    """Convert lat/lon to array indices for (lat descending, lon increasing)."""
    lat_max = 90.0
    lon_min = -180.0
    lat_idx = int(round((lat_max - lat) / resolution))
    lon_idx = int(round((lon - lon_min) / resolution))
    return lat_idx, lon_idx

def write_chunk_to_zarr(lat0, lat1, lon0, lon1, store_url, value=12.0, dtype="float32"):
    """Function that will run on a Dask worker to write to Zarr."""

    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(store_url)
    z = zarr.open(mapper, mode="r+")

    h = lat1 - lat0
    w = lon1 - lon0

    print(f"Worker writing shape ({h}, {w}) to [{lat0}:{lat1}, {lon0}:{lon1}]")
    data = np.random.rand(h, w).astype(dtype)
    z[lat0:lat1, lon0:lon1] = data
    return True


def write_region():
    """Write a 1x1 degree region to the Zarr using Dask + Coiled."""

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster("zarr_testing", False)

    # Compute indices
    lat0, lon0 = latlon_to_indices(target_box["lat_max"], target_box["lon_min"])  # upper-left
    lat1, lon1 = latlon_to_indices(target_box["lat_min"], target_box["lon_max"])  # lower-right

    print(f"Submitting write for region [lat {lat0}:{lat1}, lon {lon0}:{lon1}]")

    future = client.submit(
        write_chunk_to_zarr,
        lat0,
        lat1,
        lon0,
        lon1,
        store_url,
        value=12.0,
        dtype=dtype,
        pure=False,
    )

    result = future.result()
    print("Write task complete:", result)


def check_region_min_max():
    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(store_url)
    z = zarr.open(mapper, mode="r")

    # Get indices for the region
    lat0, lon0 = latlon_to_indices(target_box["lat_max"], target_box["lon_min"])
    lat1, lon1 = latlon_to_indices(target_box["lat_min"], target_box["lon_max"])

    # Read and compute min/max
    region_data = z[lat0:lat1, lon0:lon1]
    print("Min:", region_data.min())
    print("Mean:", region_data.mean())
    print("Max:", region_data.max())


def consolidate_metadata():
    """Consolidate Zarr metadata into a single .zmetadata file."""
    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(store_url)

    print("Consolidating Zarr metadata...")
    zarr.consolidate_metadata(mapper)
    print("Zarr metadata consolidation complete.")

if __name__ == "__main__":
    create_empty_zarr()
    write_region()
    check_region_min_max()
    # consolidate_metadata()
