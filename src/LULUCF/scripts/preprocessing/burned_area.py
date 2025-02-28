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
import dask
from dask import delayed
from dask.distributed import Client
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
import tempfile
from pyhdf.SD import SD, SDC
from dask import bag as db

import argparse
from dask.distributed import print

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu
from ..utilities import log_utilities as lu
from ..utilities import numba_utilities as nu
from ..utilities import resize_cluster

# S3 Configuration
BUCKET_NAME = "gfw2-data"
# S3_RAW_PATH = "fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/"
S3_RAW_PATH = "fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/tiny_test/"
# S3_OUTPUT_PATH = "fires/MODIS_burned_area/MCD64A1.061/processed_geotiffs/"
S3_OUTPUT_PATH = "fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/tiny_test/outputs/"
S3_INTERMEDIATE_PATH = "fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/tiny_test/outputs/intermediate_hv_year/"
S3_FINAL_OUTPUT_PATH = "fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/tiny_test/outputs/final_10x10/"

s3 = boto3.client("s3")

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

### 🔹 Utility Functions ###
from collections import defaultdict

def extract_hv_from_filename(filename):
    """Extracts the horizontal (h) and vertical (v) tile numbers from the filename."""
    match = re.search(r"\.h(\d{2})v(\d{2})\.", filename)
    if match:
        h, v = int(match.group(1)), int(match.group(2))
        return h, v
    return None, None

import re

def extract_year_from_filename(filename):
    """Extracts the year (YYYY) from the MODIS burned area filename."""
    match = re.search(r"MCD64A1\.A(\d{4})", filename)
    if match:
        return int(match.group(1))
    return None  # Return None if no match found


def modis_tile_bounds(h, v):
    """Computes geographic bounds for a given MODIS tile index (h, v)."""
    tile_size = 10  # MODIS tiles are roughly 10° x 10°

    lon_min = h * tile_size - 180
    lon_max = lon_min + tile_size
    lat_max = 90 - v * tile_size
    lat_min = lat_max - tile_size

    return [lon_min, lat_min, lon_max, lat_max]

def list_hv_year_files_from_s3(selected_years):
    """Lists and groups all MODIS HDFs by (h, v, year) from S3."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    hv_year_dict = defaultdict(list)  # Use a dictionary to group by (h, v, year)

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=S3_RAW_PATH):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                year = extract_year_from_filename(key)

                if year in selected_years and key.endswith(".hdf"):
                    # Extract h and v from filename (MCD64A1.AYYYYDDD.hXXvYY...)
                    match = re.search(r"h(\d{2})v(\d{2})", key)
                    if match:
                        h, v = match.groups()
                        hv_year = f"{year}_h{h}v{v}"
                        hv_year_dict[hv_year].append(key)  # Group files by h-v-year

    print(f"✅ Found {len(hv_year_dict)} grouped h-v-year datasets in S3")
    return list(hv_year_dict.items())  # Convert dictionary to a list of tuples



def list_year_files_from_s3(selected_years):
    """Lists all final 10x10-degree GeoTIFFs from S3 for selected years."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    year_files = []

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=S3_FINAL_OUTPUT_PATH):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if any(str(year) in key for year in selected_years) and key.endswith(".tif"):
                    year_files.append(key)

    print(f"✅ Found {len(year_files)} final 10x10-degree files for selected years in S3")
    return year_files


def read_geotiff_from_s3(s3_key):
    """Reads a GeoTIFF from S3 and returns it as a NumPy array."""
    s3 = boto3.client("s3")
    file_stream = io.BytesIO()
    s3.download_fileobj(BUCKET_NAME, s3_key, file_stream)
    file_stream.seek(0)

    with rasterio.open(file_stream) as src:
        return src.read(1)  # Read the first band


def save_geotiff_to_s3(data, s3_folder, s3_key, bounds, crs="EPSG:4326"):
    """Saves raster data as a GeoTIFF to S3 with correct bounds."""
    s3 = boto3.client("s3")
    transform = from_bounds(*bounds, data.shape[1], data.shape[0])

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 0,
        "count": 1,
        "height": data.shape[0],
        "width": data.shape[1],
        "transform": transform,
        "crs": crs,
        "compress": "DEFLATE",
        "tiled": True,
    }

    with rasterio.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(data, 1)

        # Upload to S3
        s3.upload_fileobj(memfile, BUCKET_NAME, s3_key)
        print(f"✅ Saved GeoTIFF to s3://{BUCKET_NAME}/{s3_key}")



### 🔹 Step 1: Process & Stack HDFs for Each h-v-Year ###
def read_modis_hdf_s3(hdf_s3_key):
    """Reads an HDF file from S3 and extracts the Burn Date band."""
    s3 = boto3.client("s3")
    file_stream = io.BytesIO()
    s3.download_fileobj(BUCKET_NAME, hdf_s3_key, file_stream)
    file_stream.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf") as tmp_file:
        tmp_file.write(file_stream.read())
        tmp_path = tmp_file.name

    try:
        hdf4_file = SD(tmp_path, SDC.READ)
        if "Burn Date" in hdf4_file.datasets():
            burn_date_data = hdf4_file.select("Burn Date")[:]
            burn_date_data = np.where(burn_date_data > 0, 1, 0)
        else:
            burn_date_data = np.zeros((2400, 2400), dtype=np.uint8)

        return da.from_array(burn_date_data, chunks=(1000, 1000))
    finally:
        os.remove(tmp_path)


def process_hv_year(hv_year_hdf_files):
    """Processes and stacks all HDFs for a given h-v-year."""
    hv_year, hdf_files = hv_year_hdf_files
    print(f"📦 Processing {hv_year} with {len(hdf_files)} files")

    # Extract h, v, and year from filename
    h, v = extract_hv_from_filename(hdf_files[0])
    year = extract_year_from_filename(hdf_files[0])

    if h is None or v is None or year is None:
        print(f"⚠️ Could not extract h, v, or year from {hv_year}. Skipping...")
        return

    # Compute correct geographic bounds
    bounds = modis_tile_bounds(h, v)
    print(f"🌍 Computed bounds for {hv_year}: {bounds}")

    # Process HDFs
    tasks = [delayed(read_modis_hdf_s3)(f) for f in hdf_files]
    dask_arrays = dask.compute(*tasks)
    merged_data = da.stack(dask_arrays, axis=0).max(axis=0)

    s3_key = f"{S3_INTERMEDIATE_PATH}{year}/h{h}v{v}.tif"  # Organize outputs by year
    save_geotiff_to_s3(merged_data.compute(), S3_INTERMEDIATE_PATH, s3_key, bounds)
    print(f"🎉 Finished processing {hv_year}!")



### 🔹 Step 2: Merge h-v-Year Rasters into 10x10-degree Rasters ###
def merge_hv_years_to_10x10(year_files):
    """Merges h-v-year rasters into final 10x10-degree burned area rasters."""
    year = year_files[0].split("/")[-1].split("_")[0]
    print(f"🔄 Merging {len(year_files)} rasters for {year}...")

    tasks = [delayed(read_geotiff_from_s3)(f) for f in year_files]
    dask_arrays = dask.compute(*tasks)
    merged_data = da.stack(dask_arrays, axis=0).max(axis=0)

    s3_key = f"{S3_FINAL_OUTPUT_PATH}{year}.tif"
    save_geotiff_to_s3(merged_data.compute(), S3_FINAL_OUTPUT_PATH, s3_key, bounds=[-180, -90, 180, 90])
    print(f"🎉 Finished merging h-v-year rasters for {year}!")


def main(cluster_name, run_local=False, no_stats=False, no_log=False, no_upload=False,
         # bounding_box=None,
         log_note=None):

    selected_years = [2000]  # <-- Modify this list as needed

    use_shapefile = False
    bounding_box = [-180, -60, 180, 80]

    # Model stage being running
    stage = f'burned_area_{selected_years}'

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # # Creates the log for the main function and populates it with basic run information
    # main_logger, main_log_local_path = lu.populate_main_log_header(bounding_box, use_shapefile, client, cluster, log_note, run_local, stage)
    #
    # # Starting time for stage
    # start_time = uu.timestr()
    # main_logger.info(f"Stage {stage} started at: {start_time}")
    # main_logger.info(f"Years for burned area: {selected_years}")

    # test_hdf_access()


    # main_logger.info(f"Chunks to process: {len(hdf_files_by_year)}")
    #
    # # Determines if the output file names for final versions of outputs should be used
    # is_final = False
    # if len(hdf_files_by_year) > 2000:
    #     is_final = True
    #     main_logger.info("Running as final model.")

    # Accumulates all statistics and output messages from chunk analysis
    # From https://chatgpt.com/share/e/5599b6b0-1aaa-4d54-98d3-c720a436dd9a
    all_stats = []
    return_messages = []

    # ✅ Step 1: Group and process h-v-year stacks
    hv_year_files = list_hv_year_files_from_s3(selected_years)
    print(f"🔹 Processing {len(hv_year_files)} h-v-year stacks...")
    print(f"🔹 Processing {hv_year_files} h-v-year stacks...")
    db.from_sequence(hv_year_files).map(process_hv_year).compute()
    # db.from_sequence(hv_year_files).map(delayed(process_hv_year)).compute()

    # # ✅ Step 2: Merge h-v-Year Stacks into 10x10° Rasters
    # year_files = list_year_files_from_s3(selected_years)
    # # db.from_sequence(year_files).map(merge_hv_years_to_10x10).compute()
    # db.from_sequence(year_files).map(delayed(merge_hv_years_to_10x10)).compute()

    if not run_local:
        # Closes the Dask client if not running locally
        client.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Transfer geotifs from GLAD web folders to s3 bucket by year")
    parser.add_argument('-cn', '--cluster_name', type=str, help='Coiled cluster name')
    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload
    log_note = args.log_note

    main(cluster_name, run_local, no_stats, no_log, no_upload,
         # bounding_box=bounding_box,
        log_note=log_note)








