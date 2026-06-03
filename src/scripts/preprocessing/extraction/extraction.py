# extraction.py

import os
import posixpath
import re
import json
import logging
import boto3
import botocore
import geopandas as gpd
import pandas as pd  # For merging multiple GeoDataFrames
import rasterio
import rasterio.features
import rasterio.warp
import rasterio.mask
from rasterio.vrt import WarpedVRT
import numpy as np
from contextlib import ExitStack
from shapely.geometry import box
import shapely.speedups
import gc

# Import custom modules (ensure these are correctly set up in your environment)
import src.scripts.preprocessing.utilities as uu  # Utilities module with helper functions
from src.scripts.utilities import universal_utilities as uutil
import src.scripts.preprocessing.preprocessing_constants as cn  # Module containing constants like paths and S3 prefixes
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

EXTRACTION_RASTER_PATTERN = re.compile(r"([0-9]{2}[A-Z]_[0-9]{3}[A-Z])_extraction\.tif$")
RUSSIA_ALLOCATED_MINERAL_RESERVE = "allocated_mineral_reserve"
RUSSIA_PEAT_EXTRACTION_DATES = "peat_extraction_dates"
RUSSIA_SOURCE_FIELD = "russia_source_dataset"
RUSSIA_TYPE_LIC_FIELD = "Type_lic"
RUSSIA_ALLOWED_TYPE_LIC_VALUES = {"extraction"}
DEFAULT_RUSSIA_LICENSE_START_YEAR = 2021
DEFAULT_RUSSIA_LICENSE_END_YEAR = 2024
VECTOR_READ_ENCODINGS = (None, "UTF-8", "CP1251", "ISO-8859-1")

# -------------------- Filtering Functions --------------------

def _empty_like(gdf):
    return gdf.iloc[0:0].copy()


def _read_vector_with_encoding_fallback(path):
    last_error = None
    for encoding in VECTOR_READ_ENCODINGS:
        try:
            if encoding is None:
                return gpd.read_file(path)
            return gpd.read_file(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            logging.warning(
                "Failed reading %s with encoding %s: %s",
                path,
                encoding or "default",
                exc,
            )
    raise last_error


def _normalize_text_series(series):
    return series.astype("string").str.strip().str.lower()


def _require_columns(gdf, columns, label):
    missing = [column for column in columns if column not in gdf.columns]
    if missing:
        raise ValueError(f"{label} is missing required column(s): {missing}")


def _filter_russia_dataset(
    gdf_dataset,
    source_name,
    license_start_year=DEFAULT_RUSSIA_LICENSE_START_YEAR,
    license_end_year=DEFAULT_RUSSIA_LICENSE_END_YEAR,
):
    if source_name == RUSSIA_ALLOCATED_MINERAL_RESERVE:
        logging.info(
            "Excluding Russia %s from production extraction mask; it represents "
            "allocated reserves without extraction license status.",
            source_name,
        )
        return _empty_like(gdf_dataset)

    if source_name != RUSSIA_PEAT_EXTRACTION_DATES:
        raise ValueError(f"Unknown Russia extraction source dataset: {source_name}")

    _require_columns(
        gdf_dataset,
        [RUSSIA_TYPE_LIC_FIELD, "lic_date", "lic_expire", "cancel_dat"],
        source_name,
    )

    start = pd.Timestamp(year=int(license_start_year), month=1, day=1)
    end = pd.Timestamp(year=int(license_end_year), month=12, day=31)
    if start > end:
        raise ValueError(
            "Russia extraction license_start_year must be <= license_end_year."
        )

    license_type = _normalize_text_series(gdf_dataset[RUSSIA_TYPE_LIC_FIELD])
    type_mask = license_type.isin(RUSSIA_ALLOWED_TYPE_LIC_VALUES)

    lic_date = pd.to_datetime(gdf_dataset["lic_date"], errors="coerce")
    lic_expire = pd.to_datetime(gdf_dataset["lic_expire"], errors="coerce")
    cancel_dat = pd.to_datetime(gdf_dataset["cancel_dat"], errors="coerce")

    date_mask = (
        lic_date.notna()
        & lic_expire.notna()
        & (lic_date <= end)
        & (lic_expire >= start)
        & (cancel_dat.isna() | (cancel_dat >= start))
    )

    filtered = gdf_dataset[type_mask & date_mask].copy()
    logging.info(
        "Russia %s filter retained %d of %d features for license window %s-%s "
        "(allowed %s only).",
        source_name,
        len(filtered),
        len(gdf_dataset),
        license_start_year,
        license_end_year,
        sorted(RUSSIA_ALLOWED_TYPE_LIC_VALUES),
    )
    return filtered


def filter_gdf_dataset(
    gdf_dataset,
    dataset,
    source_name=None,
    russia_license_start_year=DEFAULT_RUSSIA_LICENSE_START_YEAR,
    russia_license_end_year=DEFAULT_RUSSIA_LICENSE_END_YEAR,
):
    """
    Apply attribute filtering to the GeoDataFrame based on the dataset.

    Args:
        gdf_dataset (gpd.GeoDataFrame): The input GeoDataFrame.
        dataset (str): The name of the dataset ('finland', 'russia', etc.).

    Returns:
        gpd.GeoDataFrame: The filtered GeoDataFrame.
    """
    try:
        if dataset == 'finland':
            gdf_dataset = gdf_dataset[gdf_dataset['luokka'] == 'turvetuotanto']
            logging.info("Filtering Finland features to luokka == 'turvetuotanto'.")
        elif dataset == 'russia':
            if source_name is None:
                raise ValueError(
                    "Russia extraction filtering requires a source_name so "
                    "allocated reserves and extraction licenses are handled separately."
                )
            gdf_dataset = _filter_russia_dataset(
                gdf_dataset,
                source_name,
                license_start_year=russia_license_start_year,
                license_end_year=russia_license_end_year,
            )
        else:
            logging.info(f"No specific attribute filtering applied for dataset '{dataset}'.")
        return gdf_dataset
    except Exception as e:
        logging.error(f"Error filtering GeoDataFrame for dataset '{dataset}': {e}")
        if dataset == "russia":
            raise
        return gdf_dataset  # Return unfiltered GeoDataFrame in case of error

def filter_raster_data(data, dataset):
    """
    Apply value filtering to the raster data based on the dataset.

    Args:
        data (numpy.ndarray): The input raster data array.
        dataset (str): The name of the dataset ('ireland', etc.).

    Returns:
        numpy.ndarray: The filtered raster data array.
    """
    try:
        if dataset == 'ireland':
            # Values 1 and 2 are Cutaway and Cutover in the raster attribute table.
            logging.info("Filtering Ireland raster values 1 and 2 to binary extraction presence.")
            data = np.where((data == 1) | (data == 2), 1, 0)
        else:
            logging.info(f"No specific raster value filtering applied for dataset '{dataset}'.")
        return data
    except Exception as e:
        logging.error(f"Error filtering raster data for dataset '{dataset}': {e}")
        return data  # Return unfiltered data in case of error

# -------------------- Main Processing Functions --------------------

def rasterize_shapefile_with_ref(gdf, output_raster_path, transform, width, height, fill_value=0, burn_value=1, dtype='uint8', tile_id=None):
    """
    Rasterize a GeoDataFrame using a reference raster's transform and dimensions.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame to rasterize.
        output_raster_path (str): Path to save the rasterized output.
        transform (Affine): Affine transform for the output raster.
        width (int): Width of the output raster.
        height (int): Height of the output raster.
        fill_value (int, optional): Value to fill in the output raster where there are no features. Defaults to 0.
        burn_value (int, optional): Value to burn into the raster where features are present. Defaults to 1.
        dtype (str, optional): Data type of the output raster. Defaults to 'uint8'.
        tile_id (str, optional): Tile ID for logging purposes.

    Returns:
        None
    """
    try:
        # Prepare shapes
        shapes = [(geom, burn_value) for geom in gdf.geometry if geom.is_valid and not geom.is_empty]

        if not shapes:
            logging.warning(f"No shapes to rasterize for tile {tile_id}.")
            return  # Early exit since there's nothing to rasterize

        logging.info(f"Rasterizing {len(shapes)} shapes for tile {tile_id}.")

        # Create a blank raster with the specified shape and dtype
        raster_data = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=fill_value,
            dtype=dtype
        )

        # Define the metadata for the output raster
        out_meta = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": dtype,
            "crs": gdf.crs,
            "transform": transform,
            "nodata": fill_value,
            "compress": "DEFLATE",
            "tiled": True
        }

        # Write the rasterized data to the output file
        with rasterio.open(output_raster_path, "w", **out_meta) as dest:
            dest.write(raster_data, 1)

        logging.info(f"Rasterized shapefile saved to {output_raster_path}")

    except Exception as e:
        logging.error(f"Error rasterizing shapefile for tile {tile_id}: {e}")
        raise  # Re-raise the exception to be caught in the calling function


def extraction_filename(tile_id):
    return f"{tile_id}_extraction.tif"


def extraction_s3_key(prefix, tile_id):
    return posixpath.join(prefix.rstrip("/"), extraction_filename(tile_id))


def source_s3_key(dataset, tile_id):
    return extraction_s3_key(cn.datasets['extraction'][dataset]['s3_processed'], tile_id)


def final_s3_key(tile_id):
    return extraction_s3_key(cn.extraction_final_s3_processed, tile_id)


def source_local_path(dataset, tile_id):
    return os.path.join(cn.datasets['extraction'][dataset]['local_processed'], extraction_filename(tile_id))


def final_local_path(tile_id):
    return os.path.join(cn.extraction_final_local_processed, extraction_filename(tile_id))


def tile_id_from_extraction_key(key):
    match = EXTRACTION_RASTER_PATTERN.search(posixpath.basename(key))
    return match.group(1) if match else None


def get_reference_profile(tile_id):
    s3_input_raster_path = f"/vsis3/{cn.s3_bucket_name}/{cn.peat_tiles_prefix}{tile_id}_peat_mask_processed.tif"
    try:
        with rasterio.Env(AWS_SESSION=boto3.Session()):
            with rasterio.open(s3_input_raster_path) as src:
                return {
                    "crs": src.crs,
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
                    "bounds": src.bounds,
                }
    except Exception as exc:
        bounds = uutil.get_10x10_tile_bounds(tile_id)
        logging.warning(
            "Reference peat tile unavailable for %s (%s); using synthetic "
            "EPSG:4326 10x10 degree model grid.",
            tile_id,
            exc,
        )
        return {
            "crs": rasterio.crs.CRS.from_epsg(4326),
            "transform": rasterio.transform.from_bounds(
                *bounds,
                width=cn.full_raster_dims,
                height=cn.full_raster_dims,
            ),
            "width": cn.full_raster_dims,
            "height": cn.full_raster_dims,
            "bounds": bounds,
        }


def raster_aligns_to_reference(src, reference):
    return (
        src.crs == reference["crs"]
        and src.transform == reference["transform"]
        and src.width == reference["width"]
        and src.height == reference["height"]
    )


def qa_extraction_tile(raster_path, tile_id, source_countries, reference=None):
    unique_values = set()
    nonzero_count = 0

    with rasterio.open(raster_path) as src:
        for _, window in src.block_windows(1):
            data = src.read(1, window=window)
            unique_values.update(int(v) for v in np.unique(data))
            nonzero_count += int(np.count_nonzero(data))

        aligns_to_reference = (
            raster_aligns_to_reference(src, reference) if reference is not None else None
        )
        qa_report = {
            "tile_id": tile_id,
            "source_countries": sorted(source_countries),
            "nonzero_pixel_count": nonzero_count,
            "unique_values": sorted(unique_values),
            "crs": str(src.crs),
            "transform": tuple(src.transform.to_gdal()),
            "width": src.width,
            "height": src.height,
            "aligns_to_reference_grid": aligns_to_reference,
        }

    logging.info("Extraction QA: %s", json.dumps(qa_report, default=str))
    return qa_report


def get_source_raster_path(dataset, tile_id, run_mode='default'):
    local_path = source_local_path(dataset, tile_id)
    if os.path.exists(local_path):
        return local_path

    if run_mode == 'test':
        return None

    s3_key = source_s3_key(dataset, tile_id)
    if uutil.s3_file_exists(cn.s3_bucket_name, s3_key):
        return f"/vsis3/{cn.s3_bucket_name}/{s3_key}"

    return None


def find_source_tiles(tile_id=None, run_mode='default'):
    source_tiles = {}

    if tile_id:
        for dataset in cn.extraction_source_datasets:
            if get_source_raster_path(dataset, tile_id, run_mode) is not None:
                source_tiles.setdefault(tile_id, []).append(dataset)
        return source_tiles

    for dataset in cn.extraction_source_datasets:
        if run_mode == 'test':
            local_dir = cn.datasets['extraction'][dataset]['local_processed']
            keys = []
            if os.path.isdir(local_dir):
                keys = [os.path.join(local_dir, name) for name in os.listdir(local_dir)]
        else:
            try:
                keys = uutil.list_s3_files(
                    cn.s3_bucket_name,
                    cn.datasets['extraction'][dataset]['s3_processed'],
                )
            except Exception as e:
                logging.error(f"Unable to list source extraction outputs for {dataset}: {e}")
                keys = []

        for key in keys:
            tid = tile_id_from_extraction_key(key)
            if tid:
                source_tiles.setdefault(tid, []).append(dataset)

    return source_tiles


def consolidate_final_tile(tile_id, source_countries, run_mode='default'):
    source_paths = []
    for country in source_countries:
        path = get_source_raster_path(country, tile_id, run_mode)
        if path is not None:
            source_paths.append((country, path))

    if not source_paths:
        logging.info(f"No source extraction rasters found for tile {tile_id}. Skipping consolidation.")
        return None

    reference = get_reference_profile(tile_id)
    local_output_path = final_local_path(tile_id)
    os.makedirs(os.path.dirname(local_output_path), exist_ok=True)

    out_meta = {
        "driver": "GTiff",
        "height": reference["height"],
        "width": reference["width"],
        "count": 1,
        "dtype": "uint8",
        "crs": reference["crs"],
        "transform": reference["transform"],
        "nodata": 0,
        "compress": "DEFLATE",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    contributor_pixel_counts = {country: 0 for country, _ in source_paths}
    with rasterio.Env(AWS_SESSION=boto3.Session()):
        with ExitStack() as stack:
            readers = []
            for country, path in source_paths:
                src = stack.enter_context(rasterio.open(path))
                if raster_aligns_to_reference(src, reference):
                    reader = src
                else:
                    logging.warning(
                        f"{country} source tile {tile_id} is not aligned to the reference grid; "
                        "reading through a nearest-neighbor WarpedVRT."
                    )
                    reader = stack.enter_context(WarpedVRT(
                        src,
                        crs=reference["crs"],
                        transform=reference["transform"],
                        width=reference["width"],
                        height=reference["height"],
                        resampling=rasterio.warp.Resampling.nearest,
                        nodata=0,
                        dtype="uint8",
                    ))
                readers.append((country, reader))

            with rasterio.open(local_output_path, "w", **out_meta) as dest:
                for _, window in dest.block_windows(1):
                    union_block = np.zeros(
                        (int(window.height), int(window.width)),
                        dtype=np.uint8,
                    )
                    for country, reader in readers:
                        data = reader.read(1, window=window, boundless=True, fill_value=0)
                        present = data > 0
                        if present.any():
                            contributor_pixel_counts[country] += int(np.count_nonzero(present))
                            union_block = np.maximum(union_block, present.astype(np.uint8))
                    dest.write(union_block, 1, window=window)

    source_countries_with_pixels = [
        country for country, count in contributor_pixel_counts.items() if count > 0
    ]

    if not source_countries_with_pixels:
        logging.warning(f"Consolidated extraction raster for tile {tile_id} contains no data. Skipping upload.")
        uu.delete_file_if_exists(local_output_path)
        return None

    qa_report = qa_extraction_tile(
        local_output_path,
        tile_id,
        source_countries_with_pixels,
        reference=reference,
    )

    if qa_report["unique_values"] not in ([0], [0, 1], [1]):
        logging.warning(
            f"Final extraction raster for tile {tile_id} is not binary: "
            f"{qa_report['unique_values']}"
        )

    if run_mode != 'test':
        s3_output_path = final_s3_key(tile_id)
        uutil.upload_file_to_s3(local_output_path, cn.s3_bucket_name, s3_output_path)
        logging.info(f"Uploaded final binary union to s3://{cn.s3_bucket_name}/{s3_output_path}")
        uu.delete_file_if_exists(local_output_path)

    return qa_report


def consolidate_extraction_outputs(tile_id=None, run_mode='default'):
    source_tiles = find_source_tiles(tile_id=tile_id, run_mode=run_mode)
    if not source_tiles:
        logging.info("No source extraction outputs found for consolidation.")
        return []

    qa_reports = []
    for tid in sorted(source_tiles):
        qa_report = consolidate_final_tile(tid, sorted(set(source_tiles[tid])), run_mode=run_mode)
        if qa_report is not None:
            qa_reports.append(qa_report)

    logging.info(f"Consolidated {len(qa_reports)} final extraction tiles.")
    return qa_reports

def process_vector_dataset(
    dataset,
    tile_id=None,
    run_mode='default',
    russia_license_start_year=DEFAULT_RUSSIA_LICENSE_START_YEAR,
    russia_license_end_year=DEFAULT_RUSSIA_LICENSE_END_YEAR,
):
    """
    Process vector datasets (Finland and Russia).

    Parameters:
        dataset (str): The dataset to process ('finland' or 'russia').
        tile_id (str, optional): Tile ID to process a specific tile. Defaults to None.
        run_mode (str): The mode to run the script ('default' or 'test').

    Returns:
        None
    """
    try:
        logging.info(f"Starting processing routine for {dataset.capitalize()} peat extraction dataset")

        # Initialize an empty GeoDataFrame for Russia
        gdf_dataset = gpd.GeoDataFrame()

        # Load and merge shapefiles for Russia
        if dataset == 'russia':
            # Check if 's3_raw' is a list
            if isinstance(cn.datasets['extraction'][dataset]['s3_raw'], list):
                gdf_list = []
                for s3_prefix in cn.datasets['extraction'][dataset]['s3_raw']:
                    shapefile_name = os.path.basename(s3_prefix)
                    shapefile_path = os.path.join(cn.local_temp_dir, shapefile_name + '.shp')
                    if not os.path.exists(shapefile_path):
                        logging.info(f"{shapefile_name} shapefile not found locally, downloading from S3.")
                        uutil.download_shapefile_from_s3(s3_prefix, cn.local_temp_dir, cn.s3_bucket_name)
                    else:
                        logging.info(f"{shapefile_name} shapefile found locally at {shapefile_path}")
                    # Read the shapefile
                    gdf_part = _read_vector_with_encoding_fallback(shapefile_path)
                    gdf_part[RUSSIA_SOURCE_FIELD] = shapefile_name
                    gdf_part = filter_gdf_dataset(
                        gdf_part,
                        dataset,
                        source_name=shapefile_name,
                        russia_license_start_year=russia_license_start_year,
                        russia_license_end_year=russia_license_end_year,
                    )
                    # Append to the list
                    if not gdf_part.empty:
                        gdf_list.append(gdf_part)
                # Merge the datasets
                if not gdf_list:
                    logging.warning("No Russia extraction features survived filtering. Exiting.")
                    return
                gdf_dataset = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
            else:
                logging.error(f"Expected a list of S3 paths for Russia datasets in 's3_raw'.")
                return
        else:
            # For other datasets (e.g., Finland), proceed as before
            shapefile_s3_prefix = cn.datasets['extraction'][dataset]['s3_raw']
            shapefile_name = os.path.basename(shapefile_s3_prefix)
            shapefile_path = os.path.join(cn.local_temp_dir, shapefile_name + '.shp')
            if not os.path.exists(shapefile_path):
                logging.info(f"{dataset.capitalize()} shapefile not found locally, downloading from S3.")
                uutil.download_shapefile_from_s3(shapefile_s3_prefix, cn.local_temp_dir, cn.s3_bucket_name)
            else:
                logging.info(f"{dataset.capitalize()} shapefile found locally at {shapefile_path}")
            gdf_dataset = _read_vector_with_encoding_fallback(shapefile_path)

        # Apply attribute filtering
        if dataset != "russia":
            gdf_dataset = filter_gdf_dataset(gdf_dataset, dataset)

        # Ensure GeoDataFrame has valid geometries
        gdf_dataset['geometry'] = gdf_dataset['geometry'].buffer(0)

        # Explode multi-part geometries
        gdf_dataset = gdf_dataset.explode(index_parts=False)

        # Load tile index shapefile
        tile_index_path = os.path.join(cn.local_temp_dir, os.path.basename(cn.index_shapefile_prefix) + '.shp')
        if not os.path.exists(tile_index_path):
            logging.info("Global tile index shapefile not found locally. Downloading...")
            uutil.download_shapefile_from_s3(cn.index_shapefile_prefix, cn.local_temp_dir, cn.s3_bucket_name)
            if not os.path.exists(tile_index_path):
                logging.error("Failed to download global tile index shapefile. Exiting.")
                return
        gdf_tiles = gpd.read_file(tile_index_path)

        # Reproject dataset to match tiles CRS if necessary
        if gdf_dataset.crs != gdf_tiles.crs:
            logging.info(f"Reprojecting {dataset} data to match tiles CRS")
            gdf_dataset = gdf_dataset.to_crs(gdf_tiles.crs)

        gdf_dataset = gdf_dataset.reset_index(drop=True)
        gdf_dataset = gdf_dataset[~gdf_dataset.geometry.is_empty & ~gdf_dataset.geometry.isna()]

        # Perform spatial join to find tiles intersecting with the dataset
        tiles_intersecting = gpd.sjoin(gdf_tiles, gdf_dataset, how='inner', predicate='intersects')
        tile_ids = tiles_intersecting['tile_id'].unique()

        logging.info(f"Found {len(tile_ids)} tiles intersecting with {dataset.capitalize()} dataset.")

        if tile_id:
            if tile_id in tile_ids:
                process_vector_tile(dataset, tile_id, gdf_dataset, run_mode)
            else:
                logging.info(f"Tile {tile_id} does not intersect with {dataset.capitalize()} dataset. Skipping.")
        else:
            for tid in tile_ids:
                process_vector_tile(dataset, tid, gdf_dataset, run_mode)

    except Exception as e:
        logging.error(f"Error processing {dataset.capitalize()} dataset: {e}")


def process_vector_tile(dataset, tile_id, gdf_dataset, run_mode='default'):
    """
    Processes a single tile for vector datasets (Finland and Russia).

    Parameters:
        dataset (str): The dataset to process ('finland' or 'russia').
        tile_id (str): ID of the tile to process.
        gdf_dataset (GeoDataFrame): The dataset GeoDataFrame.
        run_mode (str): The mode to run the script ('default' or 'test').

    Returns:
        None
    """
    output_dir = cn.datasets['extraction'][dataset]['local_processed']
    os.makedirs(output_dir, exist_ok=True)

    local_output_path = source_local_path(dataset, tile_id)
    s3_output_path = source_s3_key(dataset, tile_id)

    try:
        if run_mode != 'test':
            if uutil.s3_file_exists(cn.s3_bucket_name, s3_output_path):
                logging.info(f"{s3_output_path} already exists on S3. Skipping processing.")
                return
            else:
                logging.info(f"{s3_output_path} does not exist on S3. Proceeding with processing.")
    except botocore.exceptions.ClientError as e:
        logging.error(f"A ClientError occurred: {e}")
        return
    except (botocore.exceptions.NoCredentialsError, botocore.exceptions.PartialCredentialsError) as e:
        logging.error(f"AWS credentials error: {e}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return

    logging.info(f"Starting processing of the tile {tile_id}")

    try:
        # Get tile bounds and CRS from the input raster tile
        reference = get_reference_profile(tile_id)
        tile_bounds = reference["bounds"]
        tile_transform = reference["transform"]
        tile_width = reference["width"]
        tile_height = reference["height"]
        tile_crs = reference["crs"]

        # Reproject dataset to match the tile's CRS
        if gdf_dataset.crs != tile_crs:
            logging.info(f"Reprojecting {dataset} data to match tile CRS for tile {tile_id}")
            gdf_dataset_tile = gdf_dataset.to_crs(tile_crs)
        else:
            gdf_dataset_tile = gdf_dataset

        # Clip GeoDataFrame to tile bounds using geopandas.clip function
        tile_box = box(*tile_bounds)
        gdf_tile = gpd.clip(gdf_dataset_tile, tile_box)

        # Check and convert if necessary
        logging.info(f"Type of gdf_tile after clipping: {type(gdf_tile)}")
        if isinstance(gdf_tile, gpd.GeoSeries):
            logging.info("Converting GeoSeries to GeoDataFrame")
            gdf_tile = gdf_tile.to_frame(name='geometry')
            gdf_tile = gpd.GeoDataFrame(gdf_tile, geometry='geometry', crs=gdf_dataset_tile.crs)

        # Ensure 'geometry' column exists
        if 'geometry' not in gdf_tile.columns:
            logging.error(f"'geometry' column missing in gdf_tile for tile {tile_id}")
            return

        if gdf_tile.empty:
            logging.info(f"No data in tile {tile_id} after clipping. Skipping.")
            return
        else:
            logging.info(f"Number of geometries in tile {tile_id}: {len(gdf_tile)}")

        # Fix invalid geometries in gdf_tile
        gdf_tile['geometry'] = gdf_tile['geometry'].buffer(0)
        gdf_tile = gdf_tile.make_valid()
        gdf_tile = gdf_tile.explode(index_parts=False).reset_index(drop=True)
        gdf_tile = gdf_tile[~gdf_tile.geometry.is_empty & ~gdf_tile.geometry.isna()]

        # Drop any remaining invalid geometries
        invalid_geoms = gdf_tile[~gdf_tile.is_valid]
        if not invalid_geoms.empty:
            logging.warning(f"Dropping {len(invalid_geoms)} invalid geometries in tile {tile_id}.")
            gdf_tile = gdf_tile[gdf_tile.is_valid]

        if gdf_tile.empty:
            logging.warning(f"All geometries in tile {tile_id} are invalid after cleaning. Skipping.")
            return

        # Rasterize the shapefile using the tile's transform and dimensions
        rasterize_shapefile_with_ref(
            gdf_tile,
            local_output_path,
            transform=tile_transform,
            width=tile_width,
            height=tile_height,
            fill_value=0,
            burn_value=1,
            dtype="uint8",
            tile_id=tile_id
        )

        # Check raster data before uploading
        with rasterio.open(local_output_path) as src:
            data = src.read(1)
            non_zero_count = np.count_nonzero(data)
            logging.info(f"Tile {tile_id} raster has {non_zero_count} non-zero pixels.")

            if non_zero_count == 0:
                logging.warning(f"Raster for tile {tile_id} contains no data. Skipping upload.")
                uu.delete_file_if_exists(local_output_path)
                return  # Skip uploading empty rasters

        if run_mode != 'test':
            # Upload to S3
            uutil.upload_file_to_s3(local_output_path, cn.s3_bucket_name, s3_output_path)
            logging.info(f"Uploaded {local_output_path} to s3://{cn.s3_bucket_name}/{s3_output_path}")

            # Remove local file
            uu.delete_file_if_exists(local_output_path)
            logging.info(f"Intermediate output raster {local_output_path} removed")

        logging.info(f"Tile {tile_id} processed successfully")

        del gdf_tile
        gc.collect()

    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}")

def process_raster_dataset(dataset, tile_id=None, run_mode='default'):
    """
    Process raster datasets (Ireland).

    Parameters:
        dataset (str): The dataset to process ('ireland').
        tile_id (str, optional): Tile ID to process a specific tile. Defaults to None.
        run_mode (str): The mode to run the script ('default' or 'test').

    Returns:
        None
    """
    try:
        # Load the raster dataset from S3
        s3_raster_path = cn.datasets['extraction'][dataset]['s3_raw']
        local_raster_path = os.path.join(cn.local_temp_dir, f"{dataset}_raw.tif")

        # Download the raster if not already present locally
        if not os.path.exists(local_raster_path):
            logging.info(f"Downloading {dataset} raster dataset from S3.")
            uutil.download_file_from_s3(s3_raster_path, local_raster_path, cn.s3_bucket_name)

        # Open the raster dataset
        with rasterio.open(local_raster_path) as raster_dataset:
            raster_crs = raster_dataset.crs

            # Check and set CRS if missing
            if raster_crs is None:
                logging.warning(f"No CRS found for {dataset} raster dataset. Setting CRS manually.")
                # Source raster is distributed in the old Irish Grid when CRS metadata is absent.
                raster_crs = raster_dataset.crs = 'EPSG:29902'

            logging.info(f"{dataset.capitalize()} raster CRS: {raster_crs}")
            logging.info(f"Raster dataset bounds: {raster_dataset.bounds}")

            # Reproject raster bounds to EPSG:4326 (tile index CRS)
            raster_bounds_4326 = rasterio.warp.transform_bounds(
                raster_crs,
                'EPSG:4326',
                *raster_dataset.bounds,
                densify_pts=21
            )
            raster_bbox_4326 = box(*raster_bounds_4326)
            logging.info(f"Raster bounds in EPSG:4326: {raster_bounds_4326}")

        # Load tile index shapefile (in EPSG:4326)
        tile_index_path = os.path.join(cn.local_temp_dir, os.path.basename(cn.index_shapefile_prefix) + '.shp')
        if not os.path.exists(tile_index_path):
            logging.info("Global tile index shapefile not found locally. Downloading...")
            uutil.download_shapefile_from_s3(cn.index_shapefile_prefix, cn.local_temp_dir, cn.s3_bucket_name)
        gdf_tiles = gpd.read_file(tile_index_path)
        logging.info(f"Tile index shapefile CRS: {gdf_tiles.crs}")

        # Ensure tile index is in EPSG:4326
        if gdf_tiles.crs != 'EPSG:4326':
            logging.info("Reprojecting tile index to EPSG:4326")
            gdf_tiles = gdf_tiles.to_crs('EPSG:4326')

        # Create a GeoDataFrame for raster bounds in EPSG:4326
        gdf_raster_bbox = gpd.GeoDataFrame({'geometry': [raster_bbox_4326]}, crs='EPSG:4326')

        # Identify intersecting tiles
        tiles_intersecting_raster = gpd.sjoin(gdf_tiles, gdf_raster_bbox, how='inner', predicate='intersects')
        tile_ids = tiles_intersecting_raster['tile_id'].unique()

        logging.info(f"Found {len(tile_ids)} tiles intersecting with {dataset} dataset: {tile_ids}")

        if tile_id:
            if tile_id in tile_ids:
                process_raster_tile(dataset, tile_id, local_raster_path, run_mode)
            else:
                logging.info(f"Tile {tile_id} does not intersect with {dataset} dataset. Skipping.")
        else:
            for tid in tile_ids:
                process_raster_tile(dataset, tid, local_raster_path, run_mode)

    except Exception as e:
        logging.error(f"Error processing {dataset} dataset: {e}")

def process_raster_tile(dataset, tile_id, local_raster_path, run_mode='default'):
    """
    Process a single tile for raster datasets, including 'hansenize' step.

    Parameters:
        dataset (str): The dataset to process ('ireland').
        tile_id (str): Tile ID to process.
        local_raster_path (str): Path to the local raster file.
        run_mode (str): The mode to run the script ('default' or 'test').

    Returns:
        None
    """
    try:
        logging.info(f"Processing tile {tile_id} for dataset {dataset}")

        # Prepare output paths
        output_dir = cn.datasets['extraction'][dataset]['local_processed']
        os.makedirs(output_dir, exist_ok=True)
        local_output_path = source_local_path(dataset, tile_id)
        s3_output_path = source_s3_key(dataset, tile_id)

        # Check if the output already exists
        if run_mode != 'test':
            if uutil.s3_file_exists(cn.s3_bucket_name, s3_output_path):
                logging.info(f"{s3_output_path} already exists on S3. Skipping processing.")
                return

        # Get tile properties from the peat tile (EPSG:4326)
        reference = get_reference_profile(tile_id)
        tile_bounds = reference["bounds"]
        tile_crs = reference["crs"]
        tile_transform = reference["transform"]
        tile_width = reference["width"]
        tile_height = reference["height"]

        logging.info(f"Tile CRS: {tile_crs}")

        # Open the source raster dataset
        with rasterio.open(local_raster_path) as src_raster:
            # Create a WarpedVRT to match the tile properties (hansenize step)
            with WarpedVRT(
                src_raster,
                crs=tile_crs,
                resampling=rasterio.warp.Resampling.nearest,
                transform=tile_transform,
                width=tile_width,
                height=tile_height,
                nodata=0,
                dtype='uint8'
            ) as vrt:
                # Read the data
                data = vrt.read(1)

                # Apply raster value filtering
                data = filter_raster_data(data, dataset)

                # If the data is empty after filtering, skip processing
                if not data.any():
                    logging.info(f"No data in tile {tile_id} after filtering. Skipping.")
                    return

                # Ensure data is uint8 and nodata is set to 0
                data = data.astype('uint8')
                data[data == vrt.nodata] = 0

                # Update metadata
                out_meta = vrt.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": tile_height,
                    "width": tile_width,
                    "transform": tile_transform,
                    "crs": tile_crs,
                    "count": 1,
                    "dtype": 'uint8',
                    "compress": "DEFLATE",
                    "tiled": True,
                    "nodata": 0
                })

                # Write the output raster
                with rasterio.open(local_output_path, "w", **out_meta) as dest:
                    dest.write(data, 1)

        logging.info(f"Tile {tile_id} processed and saved to {local_output_path}")

        if run_mode != 'test':
            # Upload to S3
            uutil.upload_file_to_s3(local_output_path, cn.s3_bucket_name, s3_output_path)
            logging.info(f"Uploaded {local_output_path} to s3://{cn.s3_bucket_name}/{s3_output_path}")

            # Remove local file
            uu.delete_file_if_exists(local_output_path)
            logging.info(f"Intermediate output raster {local_output_path} removed")

    except Exception as e:
        logging.error(f"Error processing tile {tile_id} for dataset {dataset}: {e}")

def main(
    dataset='finland',
    tile_id=None,
    run_mode='default',
    consolidate=False,
    russia_license_start_year=DEFAULT_RUSSIA_LICENSE_START_YEAR,
    russia_license_end_year=DEFAULT_RUSSIA_LICENSE_END_YEAR,
):
    """
    Main function to orchestrate the processing based on provided arguments.

    Parameters:
        dataset (str): The dataset to process ('finland', 'ireland', 'russia').
        tile_id (str, optional): Tile ID to process a specific tile. Defaults to None.
        run_mode (str, optional): The mode to run the script ('default' or 'test'). Defaults to 'default'.
        consolidate (bool, optional): If True, rebuild the final binary union after processing.

    Returns:
        None
    """
    try:
        logging.info(f"Starting main processing routine for {dataset} peat extraction dataset")

        if dataset in ['finland', 'russia']:
            process_vector_dataset(
                dataset,
                tile_id,
                run_mode,
                russia_license_start_year=russia_license_start_year,
                russia_license_end_year=russia_license_end_year,
            )
        elif dataset == 'ireland':
            process_raster_dataset(dataset, tile_id, run_mode)
        else:
            logging.error(f"Dataset '{dataset}' is not recognized. Please choose 'finland', 'ireland', or 'russia'.")
            return

        if consolidate:
            consolidate_extraction_outputs(tile_id=tile_id, run_mode=run_mode)

    except Exception as e:
        logging.error(f"Error in main processing routine: {e}")
    finally:
        logging.info("Processing completed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    parser = argparse.ArgumentParser(description="Process peat extraction datasets")
    parser.add_argument("--dataset", choices=list(cn.extraction_source_datasets),
                        help="Dataset to process")
    parser.add_argument("--tile_id", default=None, help="Optional tile ID to process")
    parser.add_argument("--run_mode", default="default", choices=["default", "test"],
                        help="Run mode")
    parser.add_argument("--consolidate", action="store_true",
                        help="After processing the selected source, rebuild the final binary union")
    parser.add_argument("--consolidate_only", action="store_true",
                        help="Only rebuild the final binary union from source-country outputs")
    parser.add_argument(
        "--russia_license_start_year",
        type=int,
        default=DEFAULT_RUSSIA_LICENSE_START_YEAR,
        help="First inventory year for Russia license-overlap filtering",
    )
    parser.add_argument(
        "--russia_license_end_year",
        type=int,
        default=DEFAULT_RUSSIA_LICENSE_END_YEAR,
        help="Last inventory year for Russia license-overlap filtering",
    )
    args = parser.parse_args()

    if args.consolidate_only:
        consolidate_extraction_outputs(tile_id=args.tile_id, run_mode=args.run_mode)
    else:
        if args.dataset is None:
            parser.error("--dataset is required unless --consolidate_only is used")
        main(
            dataset=args.dataset,
            tile_id=args.tile_id,
            run_mode=args.run_mode,
            consolidate=args.consolidate,
            russia_license_start_year=args.russia_license_start_year,
            russia_license_end_year=args.russia_license_end_year,
        )

"""
    # Example usage
    # Process Finland dataset
    main(dataset='finland', tile_id=None, run_mode='default')
    # Process Ireland dataset
    main(dataset='ireland', tile_id=None, run_mode='default')
    # Process Russia dataset
    main(dataset='russia', tile_id=None, run_mode='default')
    consolidate_extraction_outputs(tile_id=None, run_mode='default')
"""
