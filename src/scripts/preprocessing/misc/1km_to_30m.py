import os
import logging
import multiprocessing
import boto3
from utilities import (
    s3_file_exists,
    list_s3_files,
    resample_raster,
)
import src.scripts.preprocessing.preprocessing_constants as cn

"""
This script processes and resamples tiles from S3, checks for existing files, resamples the rasters,
and uploads the results back to S3.
"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# AWS S3 setup
s3_bucket = cn.s3_bucket_name
local_temp_dir = cn.wetlands_temp_dir


def process_tile(tile_id):
    target_tiles = f"{cn.peat_tiles_prefix}{tile_id}_peat_mask_processed.tif"
    grip_density = cn.grip_density_prefix_1km.format(tile_id=tile_id)
    osm_canals_density = cn.osm_canals_density_prefix_1km.format(tile_id=tile_id)

    grip_output = cn.grip_density_prefix_30m.format(tile_id=tile_id)
    osm_canals_output = cn.osm_canals_density_prefix_30m.format(tile_id=tile_id)

    paths = {
        'grip': (grip_density, grip_output),
        'osm_canals': (osm_canals_density, osm_canals_output)
    }

    reference_path = f'/vsis3/{s3_bucket}/{target_tiles}'
    logging.info(f"Checking reference file: {reference_path}")
    if not s3_file_exists(s3_bucket, target_tiles):
        logging.error(f"Reference file {target_tiles} does not exist.")
        return

    for key, (input_path, output_path) in paths.items():
        input_s3_path = f'/vsis3/{s3_bucket}/{input_path}'
        local_output_dir = os.path.join(local_temp_dir, key)
        local_output_path = os.path.join(local_output_dir, os.path.basename(output_path))

        if not os.path.exists(local_output_dir):
            os.makedirs(local_output_dir)

        logging.info(f"Checking local output file for {key}: {local_output_path}")
        if os.path.exists(local_output_path):
            logging.info(f"Local output file {local_output_path} already exists. Skipping resampling for {key}.")
            continue

        logging.info(f"Checking input file for {key}: {input_path}")
        if not s3_file_exists(s3_bucket, input_path):
            logging.error(f"Input file {input_path} does not exist.")
            continue

        logging.info(f"Checking if output file exists for {key}: {output_path}")
        if s3_file_exists(s3_bucket, output_path):
            logging.info(f"Output file {output_path} already exists in S3. Skipping resampling for {key}.")
            continue

        try:
            resample_raster(input_s3_path, local_output_path, reference_path)
            logging.info(f"Resampled {key} for tile {tile_id}")

            logging.info(f"Uploading {local_output_path} to s3://{s3_bucket}/{output_path}")
            compress_and_upload_directory_to_s3(local_output_dir, s3_bucket, os.path.dirname(output_path))
        except Exception as e:
            logging.error(f"Error processing {key} for tile {tile_id}: {e}")


def main(tile_id=None):
    try:
        if tile_id:
            logging.info(
                   f"Listing files in S3 bucket '{s3_bucket}' with prefix '{cn.peat_tiles_prefix}'")
            files = list_s3_files(s3_bucket, cn.peat_tiles_prefix)
            logging.info(f"Files in S3: {files}")

            process_tile(tile_id)
        else:
            s3_client = boto3.client('s3')
            paginator = s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=s3_bucket,
                                               Prefix=cn.peat_tiles_prefix)

            tile_ids = []
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        tile_key = obj['Key']
                        if tile_key.endswith('_peat_mask_processed.tif'):
                            tile_id = tile_key.split('/')[-1].replace('_peat_mask_processed.tif', '')
                            tile_ids.append(tile_id)

            for tile_id in tile_ids:
                process_tile(tile_id)
    finally:
        pass


# Example usage
if __name__ == "__main__":
    multiprocessing.freeze_support()
    # main(tile_id='00N_110E')  # Replace with the tile ID you want to test
    main()  # Uncomment to process all tiles sequentially
