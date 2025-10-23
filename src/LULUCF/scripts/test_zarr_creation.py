import numpy as np
import zarr
import fsspec
import coiled
from dask.distributed import Client, print


# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import numba_utilities as nu
from src.utilities import universal_utilities as uu
from src.utilities import resize_cluster

# ──────────────────────────────
# CONFIGURATION
# ──────────────────────────────
store_url = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_1_0_2_zarr_testing/global_outputs.zarr"

dataset_keys = [
    "carbon_density_AGC",
    "carbon_density_BGC",
    "flux_NEE",
    "flux_FIRE",
    "forest_mask",
]

n_years = 9
resolution = 0.00025
lat_size = int(180 / resolution)   # 720000
lon_size = int(360 / resolution)   # 1440000
chunks = (1, 1000, 1000)
dtype = "float32"
fill_value = 0.0

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
def create_global_output_zarr(store_url, dataset_keys, n_years, lat_size, lon_size, chunks):
    """Create a global Zarr group with multiple datasets (year, lat, lon)."""
    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(store_url)
    root = zarr.group(store=mapper, overwrite=True)

    for key in dataset_keys:
        root.create_dataset(
            name=key,
            shape=(n_years, lat_size, lon_size),
            chunks=chunks,
            dtype=dtype,
            fill_value=fill_value,
        )
        print(f"✅ Created dataset: {key} shape={(n_years, lat_size, lon_size)}")

    print(f"✅ Zarr group created at {store_url}")


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


def check_region_min_max(dataset_key, year_idx):
    """Check min/max of the region written."""
    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(store_url)
    z = zarr.open(mapper, mode="r")

    lat0, lon0 = latlon_to_indices(target_box["lat_max"], target_box["lon_min"])
    lat1, lon1 = latlon_to_indices(target_box["lat_min"], target_box["lon_max"])

    region = z[dataset_key][year_idx, lat0:lat1, lon0:lon1]
    print(f"🔍 {dataset_key} year {year_idx}: min={region.min():.4f}, mean={region.mean():.4f}, max={region.max():.4f}")


def consolidate_metadata():
    """Consolidate Zarr metadata into a single .zmetadata file."""
    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(store_url)
    print("📦 Consolidating metadata...")
    zarr.consolidate_metadata(mapper)
    print("✅ Metadata consolidated.")


# ──────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────
if __name__ == "__main__":
    # Step 1: Create empty Zarr store (only once)
    create_global_output_zarr(store_url, dataset_keys, n_years, lat_size, lon_size, chunks)

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster("zarr_testing", False)

    # Step 3: Compute region indices
    lat0, lon0 = latlon_to_indices(target_box["lat_max"], target_box["lon_min"])
    lat1, lon1 = latlon_to_indices(target_box["lat_min"], target_box["lon_max"])

    # Step 4: Submit write tasks for all datasets + all years
    futures = []
    for dataset_key in dataset_keys:
        for year_idx in range(n_years):
            f = client.submit(write_chunk, dataset_key, year_idx, lat0, lat1, lon0, lon1, store_url, pure=False)
            futures.append(f)

    results = client.gather(futures)
    for r in results:
        print(r)

    # Step 5: Validate one region
    check_region_min_max("carbon_density_AGC", 0)
    check_region_min_max("flux_NEE", 3)
    check_region_min_max("forest_mask", 7)

    # Step 6: Consolidate metadata
    consolidate_metadata()
