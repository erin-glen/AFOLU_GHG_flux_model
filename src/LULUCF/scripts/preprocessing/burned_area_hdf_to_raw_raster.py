"""

This script converts monthly stacks of burned area hdfs into annual geotifs of the original extent, projection, and resolution.
Each hdf represents burned area for a given month in a given year, for a given horizontal-vertical (h-v) area.
Annual output rasters show everywhere that was burned in that year (1 for burned).

Part 1: Transfer hdfs for relevant years from DAAC to s3. Download to computer first, then upload to s3.
I couldn't figure out a way to directly transfer from DAAC to s3.
I also was having trouble getting the hdf processing code to use the hdfs directly on DAAC,
so I decided to copy them to s3 for simplicity.

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

Part 2: Run this preprocessing code on the hdfs in s3. The years to run are chosen in main().

hdf processing based on https://chatgpt.com/c/67b0d477-1fc0-800a-b41e-44d954cb9b3e
I have not made this code align with other model components for the most part, e.g., no logs, no output stats, etc.

python -m scripts.utilities.create_cluster -n 1 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing.burned_area_hdf_to_raw_raster -cn AFOLU_flux_model_scripts

python -m scripts.utilities.create_cluster -n 100 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing.burned_area_hdf_to_raw_raster -cn AFOLU_flux_model_scripts

268 h-v stacks for every year 2001-2024, except for 2005, which has 267 h-v stacks (missing h01v08 in original hdf site)
2001-2024: took about 1.5 hours to run, used about 600 Coiled credits, cost about $20 on AWS.

Part 3: Hansenize rasters annual burned area rasters that are in MODIS projection/resolution.
Uses the separate Hansenization preprocessing script for that.

"""

import argparse
import os
import re
import io
import boto3
import dask.array as da
import dask
import numpy as np
import rasterio
import tempfile
from dask import delayed
from dask.distributed import Client, print
from rasterio.transform import from_bounds
from pyhdf.SD import SD, SDC
from dask import bag as db
from rasterio.crs import CRS
from collections import defaultdict

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu

# S3 Configuration
BUCKET_NAME = "gfw2-data"


MODIS_SINUSOIDAL_PROJ4 = (
    "+proj=sinu +lon_0=0 +datum=WGS84 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
)

s3 = boto3.client("s3")

### Utility Functions
def extract_hv_from_filename(filename):
    match = re.search(r"h(\d{2})v(\d{2})", filename)
    if match:
        return int(match.group(1)), int(match.group(2))  # h, v
    return None, None

def modis_tile_bounds(h, v):
    """Returns MODIS tile boundaries in meters using the sinusoidal grid."""
    tile_size_m = 1111950  # MODIS tile size in meters
    x_min = -20015109 + (h * tile_size_m)
    y_max = 10007555 - (v * tile_size_m)
    x_max = x_min + tile_size_m
    y_min = y_max - tile_size_m
    return x_min, y_min, x_max, y_max

def extract_year_from_filename(filename):
    """Extracts the year (YYYY) from the MODIS burned area filename."""
    match = re.search(r"MCD64A1\.A(\d{4})", filename)
    if match:
        return int(match.group(1))
    return None

def extract_bounds_from_hdf(hdf_s3_key):
    """Extracts spatial bounds and CRS from an HDF file."""
    s3 = boto3.client("s3")
    file_stream = io.BytesIO()
    s3.download_fileobj(BUCKET_NAME, hdf_s3_key, file_stream)
    file_stream.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf") as tmp_file:
        tmp_file.write(file_stream.read())
        tmp_path = tmp_file.name

    try:
        hdf4_file = SD(tmp_path, SDC.READ)
        metadata = hdf4_file.attributes()

        if "WESTBOUNDINGCOORDINATE" in metadata:
            west, east = metadata["WESTBOUNDINGCOORDINATE"], metadata["EASTBOUNDINGCOORDINATE"]
            north, south = metadata["NORTHBOUNDINGCOORDINATE"], metadata["SOUTHBOUNDINGCOORDINATE"]
            crs = MODIS_SINUSOIDAL_PROJ4
            return [west, south, east, north], crs

        # Fallback to MODIS Tile grid calculations
        h, v = extract_hv_from_filename(hdf_s3_key)
        return modis_tile_bounds(h, v), MODIS_SINUSOIDAL_PROJ4

    except Exception as e:
        print(f"Error extracting bounds from {hdf_s3_key}: {e}")
        return None, None
    finally:
        os.remove(tmp_path)

def list_hv_year_files_from_s3(year):
    """Lists and groups all MODIS HDFs by (h, v) for a given year."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    hv_year_dict = defaultdict(list)

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=cn.burned_area_hdf_path):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                file_year = extract_year_from_filename(key)
                if file_year == year and key.endswith(".hdf"):
                    h, v = extract_hv_from_filename(key)
                    hv_year = f"{year}_h{h}v{v}"
                    hv_year_dict[hv_year].append(key)

    return list(hv_year_dict.items())

def save_geotiff_to_s3(data, s3_key, bounds, crs=MODIS_SINUSOIDAL_PROJ4):
    """Saves raster data as a GeoTIFF to S3."""
    s3 = boto3.client("s3")

    try:
        crs = CRS.from_string(crs)
    except rasterio.errors.CRSError:
        crs = CRS.from_string(MODIS_SINUSOIDAL_PROJ4)

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
        s3.upload_fileobj(memfile, BUCKET_NAME, s3_key)

### Processing Functions
def read_modis_hdf_s3(hdf_s3_key):
    """Reads an HDF file from S3 and extracts Burn Date band, along with its bounds and CRS."""
    s3 = boto3.client("s3")
    file_stream = io.BytesIO()
    s3.download_fileobj(BUCKET_NAME, hdf_s3_key, file_stream)
    file_stream.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf") as tmp_file:
        tmp_file.write(file_stream.read())
        tmp_path = tmp_file.name

    try:
        hdf4_file = SD(tmp_path, SDC.READ)
        burn_date_data = hdf4_file.select("Burn Date")[:] if "Burn Date" in hdf4_file.datasets() else np.zeros((2400, 2400), dtype=np.uint8)
        burned_data = np.where(burn_date_data > 0, 1, 0)

        bounds, crs = extract_bounds_from_hdf(hdf_s3_key)
        return da.from_array(burned_data, chunks=(1000, 1000)), bounds, crs

    finally:
        os.remove(tmp_path)

def process_hv_year(hv_year_hdf_files):
    """Processes all HDFs for a given h-v-year."""
    hv_year, hdf_files = hv_year_hdf_files
    tasks = [delayed(read_modis_hdf_s3)(f) for f in hdf_files]
    results = dask.compute(*tasks)

    print(f"  Done with reading {hv_year}")

    dask_arrays, bounds_list, crs_list = zip(*results)
    bounds, crs = bounds_list[0], crs_list[0]
    merged_data = da.stack(dask_arrays, axis=0).max(axis=0)

    print(f"  Done merging dask arrays for {hv_year}")

    s3_key = f"{cn.burned_area_hdf_converted_to_raw_raster_path}{hv_year}.tif"
    save_geotiff_to_s3(merged_data.compute(), s3_key, bounds, crs)

    print(f"  Done uploading rasters for {hv_year}")

### Main Function
def main(cluster_name, run_local, selected_years):

    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, False)

    # Iterates through years. All h-v chunks are parallelized within a year.
    # Years are not parallelized because it was too many tasks for Dask, I think.
    # It was working better to iterate through years rather than try to do all h-v-year combos together.
    for year in selected_years:
        hv_year_files = list_hv_year_files_from_s3(year)
        print(f"Processing {year} with {len(hv_year_files)} h-v tiles: {uu.timestr()}...")
        db.from_sequence(hv_year_files).map(process_hv_year).compute()

    # Closes the Dask client if not running locally
    if not run_local:
        client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cn', '--cluster_name', type=str, required=True)
    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    args = parser.parse_args()

    main(args.cluster_name, args.run_local, list(range(2001, 2025)))  # Sequentially process each year









