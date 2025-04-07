"""
Downloads everything in the specified LULUCF output folder in s3 locally.
From https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67f3f252-8624-800a-a4d8-b0e29d05104e

Usage when run from /mnt/c/GIS/git/AFOLU_GHG_flux_model/src/LULUCF$:
python scripts/utilities/download_outputs_local.py v11

The argument is the name of the local subfolder to which everything is downloaded.
"""

import boto3
import os
import sys
from botocore.exceptions import ClientError

# Constants
BUCKET = "gfw2-data"
PREFIX = "climate/AFOLU_flux_model/LULUCF/outputs/version_0_3_0/"
BASE_DEST = "/mnt/c/GIS/AFOLU_flux_model/test_data/output/v0_3_0/"

def main():
    if len(sys.argv) != 2:
        print("Usage: python flatten_s3_download.py <subfolder>")
        sys.exit(1)

    subfolder = sys.argv[1]
    dest_folder = os.path.join(BASE_DEST, subfolder)

    os.makedirs(dest_folder, exist_ok=True)

    s3 = boto3.client('s3')

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX)

    downloaded = 0
    skipped = 0

    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            filename = os.path.basename(key)
            local_path = os.path.join(dest_folder, filename)

            # Skip directories and already existing files
            if not filename or os.path.exists(local_path):
                print(f"  SKIPPING {filename}!!!")
                skipped += 1
                continue

            print(f"  Downloading {key} → {filename}")
            try:
                s3.download_file(BUCKET, key, local_path)
                downloaded += 1
            except ClientError as e:
                print(f" Error downloading {key}: {e}")

    print(f"\n Done. {downloaded} files downloaded, {skipped} skipped.")

if __name__ == "__main__":
    main()
