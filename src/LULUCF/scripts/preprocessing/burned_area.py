"""

MODIS burned area v6.1 data landing page: https://lpdaac.usgs.gov/products/mcd64a1v061/
Site to download hdfs from: https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/
These are hdf4, not hdf5. This affects what Python libraries to use with them.

Before doing any processing with this script, I copied all the hdfs to s3. I decided to copy all of them to s3
and then work with them there instead of having the script access them directly DAAC because

https://lpdaac.usgs.gov/resources/e-learning/how-access-lp-daac-data-command-line/

To set up wget downloading of hdfs in home directory:
nano .wgetrc | chmod og-rw .wgetrc   # touch didn't work in Ubuntu
echo http-user=REPLACEWITHUSERNAME >> .wgetrc | echo http-password=REPLACEWITHPASSWORD >> .wgetrc

https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67b13996-a308-800a-98a1-dba578305b8d

To download all hdfs in the upper folder (after it iterates through the index.htmls):
wget -r -np -nH --cut-dirs=2 -R "index.html*" -P MCD64A1_data -e robots=off -A "*.hdf" -nd https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/

To download all hdfs for a specific year (all months):
wget -r -np -nH --cut-dirs=2 -R "index.html*" -P MCD64A1_data -e robots=off -A "*.hdf" -nd -I "MOTA/MCD64A1.061/2001*" https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/
2001 alone took 18 minutes

2000-2009:
time wget -r -np -nH --cut-dirs=2 -R "index.html*" -P MCD64A1_data -e robots=off -A "*.hdf" -nd -I "MOTA/MCD64A1.061/200*" https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/
To upload hdfs to s3 for processing:
(base) dagibbs22@USAWDC7F81Q74:~/MODIS/MCD64A1_data$ time aws s3 cp . s3://gfw2-data/fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/ --recursive
321 minutes to upload everything in the 2000s (29472 files)
Deleted 2000-2009 downloaded hdfs.

2010-2019:
time wget -r -np -nH --cut-dirs=2 -R "index.html*" -P MCD64A1_data -e robots=off -A "*.hdf" -nd -I "MOTA/MCD64A1.061/201*" https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/
344 minutes to download everything in the 2010s (32162 files)
aws s3 cp . s3://gfw2-data/fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/ --recursive
Deleted 2010-2019 downloaded hdfs.

2020-2024:
time wget -r -np -nH --cut-dirs=2 -R "index.html*" -P MCD64A1_data -e robots=off -A "*.hdf" -nd -I "MOTA/MCD64A1.061/202*" https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/
267 minutes to download 2020-2024 (16078 fies)
(base) dagibbs22@USAWDC7F81Q74:~/MODIS/MCD64A1_data$ time aws s3 cp . s3://gfw2-data/fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/ --recursive
170 minutes to upload
Deleted 2020-2024 downloaded hdfs.


hdf processing based on https://chatgpt.com/c/67b0d477-1fc0-800a-b41e-44d954cb9b3e

python -m scripts.utilities.create_cluster -n 1 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing.burned_area -cn AFOLU_flux_model_scripts

"""

import os
import re
import io
import boto3
import dask.array as da
import dask.bag as db
import numpy as np
import tempfile
import rasterio
from pyhdf.SD import SD, SDC
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.enums import Compression
from dask.distributed import Client
import coiled
import dask
from dask import delayed

import io
import os
import boto3
import tempfile
from pyhdf.SD import SD, SDC

import argparse
from dask.distributed import print

from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu

# S3 Configuration
BUCKET_NAME = "gfw2-data"
# S3_RAW_PATH = "fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/"
S3_RAW_PATH = "fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/tiny_test/"
# S3_OUTPUT_PATH = "fires/MODIS_burned_area/MCD64A1.061/processed_geotiffs/"
S3_OUTPUT_PATH = "fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/tiny_test/outputs/"


def test_hdf_access():
    global BUCKET_NAME, s3, datasets
    # S3 Bucket Info
    BUCKET_NAME = "gfw2-data"
    S3_KEY = "fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/MCD64A1.A2001001.h00v08.061.2021307220352.hdf"
    # Initialize Boto3 S3 Client
    s3 = boto3.client("s3")
    try:
        print(f"📂 Attempting to open s3://{BUCKET_NAME}/{S3_KEY}")

        # **Step 1: Download HDF4 file from S3 into memory**
        file_stream = io.BytesIO()
        s3.download_fileobj(BUCKET_NAME, S3_KEY, file_stream)
        file_stream.seek(0)  # Reset pointer

        # **Step 2: Write to a Temporary File**
        with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(file_stream.read())  # Write the full file to disk

        print(f"✅ Temporary file created at {tmp_path}")

        # **Step 3: Open the HDF4 file using PyHDF**
        hdf4_file = SD(tmp_path, SDC.READ)

        # **Step 4: List available datasets**
        datasets = hdf4_file.datasets()
        print("✅ Available datasets:", datasets)

        # **Step 5: Read 'Burn Date' dataset**
        if "Burn Date" in datasets:
            burn_date_data = hdf4_file.select("Burn Date")[:]  # Read all data
            print("✅ Sample Burn Date values:", burn_date_data[:5, :5])
        else:
            print("⚠️ 'Burn Date' dataset is missing!")

        # **Step 6: Cleanup - Remove Temporary File**
        os.remove(tmp_path)

    except boto3.exceptions.S3UploadFailedError:
        print(f"❌ FileNotFoundError: s3://{BUCKET_NAME}/{S3_KEY} does not exist!")
    except Exception as e:
        print(f"❌ Error opening HDF4 file: {e}")


def list_hdf_files_from_s3():
    """Lists all MODIS HDF files in the S3 bucket using pagination."""
    s3 = boto3.client("s3")  # Ensure client is initialized inside function
    paginator = s3.get_paginator("list_objects_v2")  # ✅ Use paginator

    hdf_files = []
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=S3_RAW_PATH):
        if "Contents" in page:
            hdf_files.extend([obj["Key"] for obj in page["Contents"] if obj["Key"].endswith(".hdf")])

    print(f"✅ Found {len(hdf_files)} HDF files in S3")
    return hdf_files


# Function to extract year from MODIS HDF filename
def extract_year_from_filename(filename):
    """Extracts the year from the MODIS filename (MCD64A1.AYYYYDDD.hXXvYY...)."""
    match = re.search(r"MCD64A1.A(\d{4})", filename)
    return int(match.group(1)) if match else None

def read_modis_hdf_s3(hdf_s3_key):
    """Downloads an HDF4 file from S3, extracts Burn Date band, and converts to binary burned/not burned."""
    try:
        s3 = boto3.client("s3")
        print(f"📂 Downloading {hdf_s3_key}")

        # Download file from S3 to memory
        file_stream = io.BytesIO()
        s3.download_fileobj(BUCKET_NAME, hdf_s3_key, file_stream)
        file_stream.seek(0)

        # Write to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(file_stream.read())

        print(f"✅ Temporary file created at {tmp_path}")

        # Open HDF4 file
        hdf4_file = SD(tmp_path, SDC.READ)

        # Read 'Burn Date' dataset
        if "Burn Date" in hdf4_file.datasets():
            burn_date_data = hdf4_file.select("Burn Date")[:]

            # ✅ Ensure masking of no-data values (-1)
            burn_date_data = np.where(burn_date_data > 0, 1, 0)  # Burned = 1, Not burned = 0
            burn_date_data = np.where(burn_date_data == -1, np.nan, burn_date_data)  # Mask no-data (-1)

            print(f"✅ Successfully extracted 'Burn Date' from {hdf_s3_key}")
        else:
            print(f"⚠️ 'Burn Date' dataset is missing in {hdf_s3_key}")
            os.remove(tmp_path)
            return da.full((2400, 2400), np.nan, dtype=np.float32)  # Return a full NaN array

        os.remove(tmp_path)
        return da.from_array(burn_date_data, chunks=(1000, 1000))

    except Exception as e:
        print(f"❌ Error reading {hdf_s3_key}: {e}")
        return da.full((2400, 2400), np.nan, dtype=np.float32)  # Ensure return is NaN-filled



# Function to reproject and resample to EPSG:4326 with 0.00025° resolution
def reproject_resample(data, src_crs="ESRI:54008", dst_crs="EPSG:4326", bounds=None, resolution=0.00025):
    """Reprojects and resamples MODIS raster to EPSG:4326 with 0.00025° resolution, handling NaN properly."""
    if bounds is None:
        raise ValueError("🚨 ERROR: Tile bounding box (bounds) must be provided!")

    left, bottom, right, top = bounds  # Extract tile bounding box

    print(f"🌍 Reprojecting tile {bounds} at resolution {resolution}")

    # Calculate width and height based on resolution
    width = int((right - left) / resolution)
    height = int((top - bottom) / resolution)

    print(f"✅ Target Reprojection Size: {width} x {height}")

    dst_data = np.full((height, width), np.nan, dtype=np.float32)  # Initialize with NaN

    transform, _, _ = calculate_default_transform(
        src_crs, dst_crs, width, height, left=left, bottom=bottom, right=right, top=top
    )

    reproject(
        source=data.compute(),  # Convert to NumPy before reprojecting
        destination=dst_data,
        src_transform=transform,
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest
    )

    return np.nan_to_num(dst_data, nan=0)  # Replace NaN with 0 for correct burned area values





# Function to save raster data as Cloud Optimized GeoTIFF to S3
def save_geotiff_to_s3(data, bounds, s3_key):
    """Saves raster data as Cloud Optimized GeoTIFF to S3, ensuring NaN is properly handled."""
    s3 = boto3.client("s3")

    transform = from_bounds(*bounds, data.shape[1], data.shape[0])

    profile = {
        "driver": "COG",
        "dtype": "uint8",
        "nodata": 0,  # ✅ Set nodata to 0 (ensures ocean remains 0)
        "count": 1,
        "height": data.shape[0],
        "width": data.shape[1],
        "transform": transform,
        "crs": "EPSG:4326",
        "compress": "DEFLATE",
        "tiled": True
    }

    with rasterio.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(data.astype(np.uint8), 1)  # Ensure correct dtype

        # Upload to S3
        s3.upload_fileobj(memfile, BUCKET_NAME, s3_key)
        print(f"✅ Saved GeoTIFF to s3://{BUCKET_NAME}/{s3_key}")



def process_year(year_hdf_files):
    """Processes and mosaics HDFs for a single year into 10x10 degree GeoTIFFs."""
    year, hdf_files = year_hdf_files
    print(f"📅 Processing Year: {year} with {len(hdf_files)} files")

    if not hdf_files:
        print(f"⚠️ No HDF files found for year {year}. Skipping...")
        return

    # Convert each file into a Dask task
    print(f"🔄 Creating Dask Delayed tasks for {year}...")
    tasks = [delayed(read_modis_hdf_s3)(f) for f in hdf_files]

    # Compute tasks in parallel
    print(f"🚀 Running Dask Delayed computation for {year}...")
    data_arrays = dask.compute(*tasks)

    # Remove None values from the list
    data_arrays = [arr for arr in data_arrays if arr is not None]

    print(f"✅ Done creating data_arrays for {year}: {len(data_arrays)} elements")

    if not data_arrays:
        print(f"⚠️ No valid data found for year {year}. Skipping...")
        return

    # Stack arrays properly instead of concatenate (to preserve 2D shape)
    print(f"🛠 Concatenating {year}...")
    stacked_data = da.stack(data_arrays, axis=0)  # Shape (N, 2400, 2400)
    print(f"🔍 Stacked Data Shape Before Merging: {stacked_data.shape}")

    merged_data = stacked_data.max(axis=0)  # Merge across time to find all burned areas
    print(f"✅ Merged Data Shape: {merged_data.shape}")

    print(f"💾 Exporting TIFFs for {year}...")

    for lon in range(-180, 180, 10):
        for lat in range(-90, 90, 10):
            bounds = [lon, lat, lon + 10, lat + 10]  # Define tile bounding box

            print(f"📦 Processing tile {lon}, {lat} for {year} with bounds {bounds}...")

            # 🔹 Pass `bounds` correctly
            reprojected_data = reproject_resample(merged_data, "ESRI:54008", dst_crs="EPSG:4326", bounds=bounds)
            print(f"✅ Reprojected Data Shape: {reprojected_data.shape}")

            if reprojected_data.size > 0:
                s3_key = f"{S3_OUTPUT_PATH}{year}/tile_{lon}_{lat}.tif"
                save_geotiff_to_s3(reprojected_data, bounds, s3_key)

    print(f"🎉 Finished processing {year}!")





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Transfer geotifs from GLAD web folders to s3 bucket by year")
    parser.add_argument('-cn', '--cluster_name', type=str, help='Coiled cluster name')
    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    run_local = args.run_local

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # test_hdf_access()

    selected_years = [2000]  # <-- Modify this list as needed

    # Fetch HDF files from S3
    hdf_files = list_hdf_files_from_s3()
    # print(hdf_files)

    # Run pipeline
    hdf_files_by_year = {y: [] for y in selected_years}
    for f in hdf_files:
        y = extract_year_from_filename(f)
        if y in selected_years:
            hdf_files_by_year[y].append(f)
    # print(hdf_files_by_year)

    print(f"✅ Processing {len(hdf_files_by_year)} years...")

    # ✅ Run sequentially instead of using Dask Bag
    for year, hdf_files in hdf_files_by_year.items():
        process_year((year, hdf_files))

    if not run_local:
        # Closes the Dask client if not running locally
        client.close()






