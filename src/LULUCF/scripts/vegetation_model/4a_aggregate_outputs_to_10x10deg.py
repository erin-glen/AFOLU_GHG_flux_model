"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test (Dask part does not work because of client.submit()):
python -m src.LULUCF.scripts.vegetation_model.1_calculate_veg_fluxes -bb 10 49.75 10.25 50 -cs 0.25 --run_local --no_upload --run_date YYYYMMDD

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.1_calculate_veg_fluxes -cn vegetation_model -bb 116.25 -2.25 116.5 -2 -cs 0.25 --run_date YYYYMMDD

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.1_calculate_veg_fluxes -cn vegetation_model -bb -64 -22 -63 -21 -cs 1 --run_date YYYYMMDD --create_zarr

Coiled Cerrado test (174 features):
python -m src.utilities.create_cluster -n 20 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.1_calculate_veg_fluxes -cn vegetation_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__Cerrado_center_in.shp --run_date YYYYMMDD

Coiled large shapefile test (1884 features):
python -m src.utilities.create_cluster -n 100 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.1_calculate_veg_fluxes -cn vegetation_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --run_date YYYYMMDD

Full run:
python -m src.utilities.create_cluster -n 200 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.1_calculate_veg_fluxes -cn LULUCF_model -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --run_date 20250921 --log_note "This is a global run for model v1.0.0 (2016-2024)."

"""

import dask
import xarray as xr
import numpy as np
import rasterio
import tempfile
import fsspec
import time
from dask.distributed import print
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
import s3fs
import zarr

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import numba_utilities as nu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import resize_cluster
from src.utilities.constants_and_names import full_outputs_to_zarr

# STEP 1: Start Dask cluster via Coiled

cluster, client, run_local = uu.connect_to_Coiled_cluster("aggregation_test", False)

# STEP 2: Parameters
zarr_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_1_0_3_1884_chunks/mega_zarr/standard_model/annual_intervals/10000_pixels/20251027/"
# tile_ids = ["20S_020E", "50N_000E"]  # example list – you can supply many more
# tile_ids = ["00N_020E"]  # example list – you can supply many more
# tile_ids = ["30N_020W"]  # example list – you can supply many more
tile_ids = ["40N_000E", "40N_010W", "50N_000E", "50N_010W", "60N_000E"]  # example list – you can supply many more
tile_size_deg = 10
resolution = 0.00025
samples_per_tile = int(tile_size_deg / resolution)  # 40000
output_base = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/version_1_0_3_1884_chunks/10x10_test"

# S3 Filesystem
fs = s3fs.S3FileSystem(anon=False)


# STEP 5: Write single GeoTIFF to S3 using in-memory buffer
def write_geotiff_to_s3(var, year_idx, data, transform, s3_path):
    """Write 2D array to S3 as a GeoTIFF using Rasterio's in-memory buffer."""
    print(f"Writing {var} for year {year_idx} to {s3_path}: {uu.timestr()}")
    upload_start_time = time.time()

    height, width = data.shape

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": data.dtype,
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "LZW",
        "nodata": 0,
        "tiled": True,
        "blockxsize": 400,
        "blockysize": 400,
    }

    # Write to temporary file on disk
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=True) as tmpfile:
        with rasterio.open(tmpfile.name, "w", **profile) as dst:
            dst.write(data, 1)

        # Upload efficiently using multipart S3 upload
        fs.put_file(tmpfile.name, s3_path)

    upload_end_time = time.time()
    print(f"  Wrote {var} for year {year_idx} to {s3_path} in {round(upload_end_time-upload_start_time)} seconds: {uu.timestr()}")


# STEP 6: Main export function
def extract_10x10(var, year_idx, tile_id, raw_path, output_base):
    """Extracts a 10x10° tile from a Zarr store and writes to GeoTIFF on S3."""

    # Convert tile_id to bounding box (W, S, E, N)
    min_x, min_y, max_x, max_y = uu.get_10x10_tile_bounds(tile_id)

    # Open Zarr group using fsspec mapper
    fs = fsspec.filesystem("s3", anon=False)
    zarr_store = zarr.open_group(fs.get_mapper(raw_path), mode="r")

    # Determine pixel indices
    lat_array = zarr_store["y"][:]
    lon_array = zarr_store["x"][:]

    # Get index ranges
    y0 = np.searchsorted(lat_array[::-1], max_y, side='right')
    y1 = np.searchsorted(lat_array[::-1], min_y, side='left')
    x0 = np.searchsorted(lon_array, min_x, side='left')
    x1 = np.searchsorted(lon_array, max_x, side='right')

    # Flip y indices since lat is descending
    y0, y1 = len(lat_array) - y1, len(lat_array) - y0

    if y0 > y1:
        y0, y1 = y1, y0

    year = cn.interval_end_years_annual[year_idx]

    print(f"Extracting {var} for {year} at {tile_id} (x: {x0}-{x1}, y: {y0}-{y1}): {uu.timestr()}")
    extract_start_time = time.time()

    # Load data block (Zarr lazy indexing)
    block = zarr_store[var][year_idx, y0:y1, x0:x1]
    data = block.astype(np.float32)

    # GeoTransform (top-left corner)
    transform = from_origin(min_x, max_y, resolution, resolution)

    extract_end_time = time.time()
    print(f"  Extracted {var} for year {year_idx} in {round(extract_end_time - extract_start_time)} seconds: {uu.timestr()}")

    # Output path
    s3_filename = f"{output_base}/{var}/{year}/{tile_id}.tif"

    # Write to S3
    write_geotiff_to_s3(var, year_idx, data, transform, s3_filename)

    return s3_filename


outputs_to_process = cn.full_outputs_to_zarr
years_to_process = len(cn.interval_end_years_annual)

# outputs_to_process = cn.outputs_to_zarr[0:1]
# years_to_process = len(cn.interval_end_years_annual[0:1])

futures = []

print(f"Starting processing: {uu.timestr()}")

for var in outputs_to_process:

    for year_idx in range(years_to_process):

        for tile_id in tile_ids:

            future = client.submit(extract_10x10,
                                   var, year_idx, tile_id, zarr_path, output_base)
            futures.append(future)

results = client.gather(futures)


print(f"\n✅ All exports completed: {uu.timestr()}")
for r in results:
    print(f"  → {r}")
