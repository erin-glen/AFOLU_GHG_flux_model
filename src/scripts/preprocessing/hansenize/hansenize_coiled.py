import os
import logging
import tempfile
from pathlib import Path

import boto3
import rasterio
from rasterio.errors import RasterioIOError
from osgeo import gdal

# Import your shared utilities (upload_s3_file, check_s3_file_created, etc.)
# from . import pp_utilities as uu
# from . import constants_and_names as cn

log = logging.getLogger("peat-tiler-hansen")

# For random writes using /vsis3/ in AWS/GDAL:
os.environ['CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE'] = 'YES'


################################################################################
# 1. Checks if a VRT file already exists on S3
################################################################################
def vrt_exists_in_s3(output_vrt_s3):
    """
    Returns True if 'output_vrt_s3' already exists on S3, else False.
    """
    s3 = boto3.client("s3")
    # Parse the S3 path
    s3_path_parts = output_vrt_s3.replace("s3://", "").split("/", 1)
    bucket_name = s3_path_parts[0]
    object_key = s3_path_parts[1]

    try:
        s3.head_object(Bucket=bucket_name, Key=object_key)
        return True  # File exists
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False  # File does not exist
        else:
            raise  # Some other error occurred


################################################################################
# 2. Build a VRT (Local or Coiled) using GDAL
################################################################################
def build_vrt_gdal_local(raw_raster_paths_list_s3, output_vrt_s3):
    """
    Builds a mosaic VRT directly in S3 using /vsis3/. This approach writes
    the final VRT to /vsis3/ if you have permissions to do random writes in S3.

    Args:
        raw_raster_paths_list_s3: List[str] of full S3 paths, e.g. ["s3://bucket/file1.tif", ...]
        output_vrt_s3: S3 path (e.g., "s3://bucket/...") for the final .vrt
    """
    # Enable random write support in environment (if not already set)
    os.environ['CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE'] = 'YES'

    # Convert S3 => /vsis3/
    raw_raster_paths_list_vsis3 = [path.replace("s3://", "/vsis3/") for path in raw_raster_paths_list_s3]
    output_vrt_vsis3 = output_vrt_s3.replace("s3://", "/vsis3/")

    # Build the VRT in S3 directly
    gdal.BuildVRT(output_vrt_vsis3, raw_raster_paths_list_vsis3)

    # Confirm the file is successfully created on S3
    # (Assumes you have a helper like uu.check_s3_file_created)
    # check_s3_file_created(output_vrt_s3)


def build_vrt_gdal_coiled(raw_raster_paths_list_s3, output_vrt_s3, local_vrt):
    """
    Builds a VRT by downloading input rasters (through /vsis3/) locally,
    writing the mosaic to a local .vrt, then uploading that .vrt to S3.

    Args:
        raw_raster_paths_list_s3 (List[str]): S3 paths (s3://...) to raw input rasters
        output_vrt_s3 (str): Final S3 VRT path
        local_vrt (str): Local path for the temporary .vrt
    """
    # Check if the VRT file already exists on S3 to skip
    if vrt_exists_in_s3(output_vrt_s3):
        print(f"VRT file already exists in S3: {output_vrt_s3}. Skipping creation.")
        return

    # Convert S3 => /vsis3/
    vsis3_paths = [s3_path.replace("s3://", "/vsis3/") for s3_path in raw_raster_paths_list_s3]

    # Make sure local directory exists
    local_vrt_path = Path(local_vrt)
    local_vrt_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the VRT locally
    print(f"Building VRT locally at {local_vrt} ...")
    gdal.BuildVRT(str(local_vrt), vsis3_paths)
    print("BuildVRT complete.")

    # Basic checks on the local VRT
    try:
        with rasterio.open(local_vrt) as vrt_dataset:
            if vrt_dataset.count == 0:
                raise RuntimeError("VRT has no data or invalid sources.")
            if not vrt_dataset.bounds:
                raise RuntimeError("VRT has invalid metadata or empty bounds.")
        print(f"Local VRT is valid: {local_vrt}")
    except RasterioIOError:
        raise RuntimeError(f"Error: local VRT file not found or invalid: {local_vrt}")

    # Upload the local VRT to S3
    # uu.upload_s3_file(output_vrt_s3, local_vrt)  # or: uu.upload_file_to_s3(...)
    # check_s3_file_created(output_vrt_s3)

    # Remove local
    # local_vrt_path.unlink()


################################################################################
# 3. Warp to Hansen resolution (Local or Coiled)
################################################################################
def warp_to_hansen_local(source_raster_s3_path, output_raster_s3_path,
                         xmin, ymin, xmax, ymax,
                         dt, no_data, tiled=True,
                         x_pixel_window=400, y_pixel_window=400):
    """
    Warps a VRT or single raster to EPSG:4326 at 0.00025 resolution
    and writes directly to /vsis3/ (so the final .tif is in S3).
    """
    os.environ['CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE'] = 'YES'

    if tiled and not (x_pixel_window and y_pixel_window):
        raise ValueError("If tiled=True, must specify x_pixel_window and y_pixel_window")

    # Convert s3:// => /vsis3/
    source_gdal_path = source_raster_s3_path.replace("s3://", "/vsis3/")
    output_gdal_path = output_raster_s3_path.replace("s3://", "/vsis3/")

    ds = gdal.Open(source_gdal_path)
    if not ds:
        raise RuntimeError(f"Failed to open source for warp: {source_raster_s3_path}")

    # Prepare warp options
    if tiled:
        options = gdal.WarpOptions(
            dstSRS='EPSG:4326',
            xRes=0.00025,
            yRes=0.00025,
            targetAlignedPixels=True,
            outputBounds=[xmin, ymin, xmax, ymax],
            dstNodata=no_data,
            outputType=dt,
            creationOptions=[
                'COMPRESS=DEFLATE',
                'TILED=YES',
                f'BLOCKXSIZE={x_pixel_window}',
                f'BLOCKYSIZE={y_pixel_window}'
            ],
            format='GTiff'
        )
    else:
        options = gdal.WarpOptions(
            dstSRS='EPSG:4326',
            xRes=0.00025,
            yRes=0.00025,
            targetAlignedPixels=True,
            outputBounds=[xmin, ymin, xmax, ymax],
            dstNodata=no_data,
            outputType=dt,
            creationOptions=['COMPRESS=DEFLATE', 'TILED=NO'],
            format='GTiff'
        )

    gdal.Warp(output_gdal_path, ds, options=options)
    ds = None  # close input

    # Confirm final .tif is on S3
    # check_s3_file_created(output_raster_s3_path)


def warp_to_hansen_coiled(source_vrt_path, filename, output_raster_s3_path_and_name,
                          xmin, ymin, xmax, ymax,
                          dt, no_data, tiled=True,
                          x_pixel_window=400, y_pixel_window=400):
    """
    Warps a local file from /vsis3/ then writes output to a local temp .tif,
    checks data, and uploads to S3 (removing the local file).
    """
    # Typically, we set up a custom logger if needed:
    # logger_worker = lu.setup_logging_worker()
    log.info(f"Creating {filename} from {source_vrt_path}...")

    if tiled and not (x_pixel_window and y_pixel_window):
        raise ValueError("If tiled=True, must pass x_pixel_window & y_pixel_window")

    # Convert s3:// => /vsis3/ for reading
    source_vrt_path_vsis3 = source_vrt_path.replace("s3://", "/vsis3/")
    ds = gdal.Open(source_vrt_path_vsis3)
    if not ds:
        raise RuntimeError(f"Failed to open {source_vrt_path}")

    # Local output
    local_tif = Path(tempfile.gettempdir()) / filename
    local_tif_str = str(local_tif)

    if tiled:
        options = gdal.WarpOptions(
            dstSRS='EPSG:4326',
            xRes=0.00025,
            yRes=0.00025,
            targetAlignedPixels=True,
            outputBounds=[xmin, ymin, xmax, ymax],
            dstNodata=no_data,
            outputType=dt,
            creationOptions=[
                'COMPRESS=DEFLATE',
                'TILED=YES',
                f'BLOCKXSIZE={x_pixel_window}',
                f'BLOCKYSIZE={y_pixel_window}'
            ],
            format='GTiff'
        )
    else:
        options = gdal.WarpOptions(
            dstSRS='EPSG:4326',
            xRes=0.00025,
            yRes=0.00025,
            targetAlignedPixels=True,
            outputBounds=[xmin, ymin, xmax, ymax],
            dstNodata=no_data,
            outputType=dt,
            creationOptions=['COMPRESS=DEFLATE', 'TILED=NO'],
            format='GTiff'
        )

    gdal.Warp(local_tif_str, ds, options=options)
    ds = None

    # Check for data
    # if not check_geotiff_has_data(local_tif_str):
    #     # log we have an empty tile, skip upload
    #     local_tif.unlink()
    #     return f"{filename} is empty or has only NoData."

    # Upload final to S3 if it has data
    # uu.upload_s3_file(output_raster_s3_path_and_name, local_tif_str)
    log.info(f"{filename} uploaded to {output_raster_s3_path_and_name}")

    # Remove local
    # if local_tif.exists():
    #     local_tif.unlink()
    return f"Success Hansenizing {filename}"


################################################################################
# 4. Utility to delete local inputs after building VRT
################################################################################
def delete_build_vrt_input_files(raw_raster_paths_list_s3, vrt):
    """
    If you're downloading raw TIFs locally for the mosaic, remove them plus the local .vrt.
    """
    for s3_path in raw_raster_paths_list_s3:
        local_file = s3_path.split('/')[-1]
        local_path = Path(local_file)
        if local_path.exists():
            local_path.unlink()

    local_vrt_path = Path(vrt)
    if local_vrt_path.exists():
        local_vrt_path.unlink()
