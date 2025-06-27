"""
Converts global geotifs into global COGs

From https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/685ee215-b624-800a-9ab7-06d1c27a1697
"""

import os
import boto3
import subprocess
from urllib.parse import urlparse
from tqdm import tqdm

# Configuration
input_s3_folder = 's3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/raw__from_Cornell/20250512/year_2020/all_sources/'
output_s3_folder = 's3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/processed/20250512/year_2020/all_sources/global_COG/'
local_input_dir = '/tmp/input_tifs'
local_output_dir = '/tmp/output_cogs'

os.makedirs(local_input_dir, exist_ok=True)
os.makedirs(local_output_dir, exist_ok=True)

# Parse input bucket and prefix
parsed = urlparse(input_s3_folder)
bucket_name = parsed.netloc
prefix = parsed.path.lstrip('/')

# Set up boto3
s3 = boto3.client('s3')

# List all .tif files in the input S3 folder
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
tif_keys = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.tif')]

print(f"Found {len(tif_keys)} TIFF files in {input_s3_folder}: {tif_keys}")

for key in tqdm(tif_keys, desc="Processing GeoTIFFs"):
    filename = os.path.basename(key)
    local_input_path = os.path.join(local_input_dir, filename)
    local_output_path = os.path.join(local_output_dir, filename)

    print(f"Processing {filename}...")

    # Download from S3
    s3.download_file(bucket_name, key, local_input_path)

    # Convert to COG using gdal_translate
    cmd = [
        'gdal_translate', local_input_path, local_output_path,
        '-of', 'COG',
        '-co', 'COMPRESS=DEFLATE',
        '-co', 'PREDICTOR=2',
        '-co', 'BIGTIFF=IF_SAFER',
        '-co', 'OVERVIEW_RESAMPLING=average'
    ]
    subprocess.run(cmd, check=True)

    # Upload to output S3
    output_key = os.path.join(
        urlparse(output_s3_folder).path.lstrip('/'),
        filename
    )
    s3.upload_file(local_output_path, urlparse(output_s3_folder).netloc, output_key)

    # Optional cleanup
    os.remove(local_input_path)
    os.remove(local_output_path)

    print(f"  Finished processing {filename}")

print("All files processed and uploaded as COGs.")
