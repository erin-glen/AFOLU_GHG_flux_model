"""Generate Hansen-aligned mangrove extent tiles from yearly Global Mangrove Watch rasters.

This script downloads the yearly mangrove extent rasters that live under
``global-mangrove-extent/version3/smoothed/raster/<year>`` on S3, unions all
available years for each 10x10 degree Hansen tile, hansenizes the result, and
uploads the output back to S3 under the organic soils processed inputs.

The output tiles are binary (0/1) rasters where 1 indicates mangrove presence in
any of the processed years. The rasters share the standard Hansen georeferencing
(0.00025 degree pixels aligned with the Hansen grid).

Example
-------
Process every tile using all available years::

    python -m src.scripts.preprocessing.mangrove.mangrove_extent_union

Process a subset of tiles and restrict to a set of years:

    python -m src.scripts.preprocessing.mangrove.mangrove_extent_union \
        --tile-id 00N_100W --tile-id 00N_110E --years 2018 2019 2020

Execute the workflow on a Coiled cluster::

    python -m src.scripts.preprocessing.mangrove.mangrove_extent_union \
        --client coiled --cluster-name mangrove_extent_union
"""

from __future__ import annotations

import argparse
import logging
import os
import posixpath as pp
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import boto3
import dask
import geopandas as gpd
import numpy as np
import rasterio
from dask import delayed
from dask.distributed import Client, LocalCluster
from rasterio.warp import Resampling, reproject

import src.scripts.preprocessing.preprocessing_constants as cn
from src.scripts.preprocessing.hansenize import hansenize_gdal as hz
from src.scripts.utilities import universal_utilities as uu

LOGGER = logging.getLogger("mangrove-extent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass(frozen=True)
class TileBounds:
    """Lightweight container for Hansen tile bounds."""

    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def as_tuple(self) -> Tuple[float, float, float, float]:
        return self.minx, self.miny, self.maxx, self.maxy


def _list_available_years(s3_client: boto3.client, root_prefix: str) -> List[str]:
    """Return the set of year folder names that exist below *root_prefix*."""

    paginator = s3_client.get_paginator("list_objects_v2")
    years: set[str] = set()
    for page in paginator.paginate(
        Bucket=cn.s3_bucket_name,
        Prefix=root_prefix.rstrip("/") + "/",
        Delimiter="/",
    ):
        for pref in page.get("CommonPrefixes", []):
            folder = pref.get("Prefix", "").rstrip("/")
            if not folder:
                continue
            year = folder.split("/")[-1]
            if year.isdigit():
                years.add(year)
    return sorted(years)


def _gather_tile_sources(
    s3_client: boto3.client,
    root_prefix: str,
    years: Optional[Iterable[str]] = None,
) -> Dict[str, List[str]]:
    """Collect yearly raster paths for each tile.

    Parameters
    ----------
    s3_client
        Boto3 S3 client used for listing objects.
    root_prefix
        Prefix relative to the bucket where the yearly directories live.
    years
        Optional iterable of years (as strings) to restrict the listing.
    """

    allowed_years: Optional[set[str]] = set(str(y) for y in years) if years else None
    prefix = root_prefix.rstrip("/") + "/"
    paginator = s3_client.get_paginator("list_objects_v2")
    tile_map: Dict[str, List[str]] = defaultdict(list)

    for page in paginator.paginate(Bucket=cn.s3_bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.lower().endswith(".tif"):
                continue

            relative = key[len(prefix) :]
            parts = relative.split("/")
            if len(parts) < 2:
                continue

            year = parts[0]
            if allowed_years is not None and year not in allowed_years:
                continue

            filename = parts[-1]
            tile_id = filename.split("__")[0]
            tile_map[tile_id].append(f"s3://{cn.s3_bucket_name}/{key}")

    return tile_map


def _load_tile_bounds() -> Dict[str, TileBounds]:
    """Load tile bounds from the shared Hansen tile index shapefile."""

    local_dir = Path(cn.local_temp_dir) / "tile_index"
    local_dir.mkdir(parents=True, exist_ok=True)
    uu.download_shapefile_from_s3(cn.index_shapefile_prefix, str(local_dir), cn.s3_bucket_name)
    shapefile_path = local_dir / cn.tile_index_shapefile_name
    gdf = gpd.read_file(shapefile_path)

    bounds: Dict[str, TileBounds] = {}
    for _, row in gdf.iterrows():
        tile_id = str(row.get("tile_id"))
        if not tile_id or row.geometry is None:
            continue
        minx, miny, maxx, maxy = row.geometry.bounds
        bounds[tile_id] = TileBounds(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    return bounds


def _union_years(tile_id: str, raster_paths: Sequence[str]) -> Optional[Tuple[np.ndarray, rasterio.profiles.Profile]]:
    """Return a uint8 union array (0/1) and profile for *tile_id*.

    The returned profile uses the metadata of the first raster. Any subsequent
    rasters that do not align with the base profile are reprojected before the
    union.
    """

    base_mask: Optional[np.ndarray] = None
    base_profile: Optional[rasterio.profiles.Profile] = None
    base_transform = None
    base_crs = None
    base_shape: Optional[Tuple[int, int]] = None

    for path in raster_paths:
        try:
            vsipath = path.replace("s3://", "/vsis3/", 1)
            with rasterio.open(vsipath) as src:
                data = src.read(1, masked=True)
                mask = np.asarray(data.filled(0) > 0, dtype=np.uint8)

                if base_mask is None:
                    base_mask = mask
                    base_profile = src.profile.copy()
                    base_transform = src.transform
                    base_crs = src.crs
                    base_shape = mask.shape
                else:
                    if (
                        base_shape != mask.shape
                        or base_transform != src.transform
                        or base_crs != src.crs
                    ):
                        aligned = np.zeros(base_shape, dtype=np.uint8)
                        reproject(
                            source=mask,
                            destination=aligned,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=base_transform,
                            dst_crs=base_crs,
                            resampling=Resampling.nearest,
                            src_nodata=0,
                            dst_nodata=0,
                        )
                        mask = aligned
                    base_mask = np.maximum(base_mask, mask)
        except Exception as exc:  # noqa: BLE001 - log and continue to next raster
            LOGGER.warning("[%s] failed to read %s: %s", tile_id, path, exc)

    if base_mask is None or base_profile is None:
        return None

    base_profile.update(
        dtype="uint8",
        nodata=0,
        count=1,
        compress="DEFLATE",
        tiled=True,
    )
    return base_mask, base_profile


def _write_temp_union(mask: np.ndarray, profile: rasterio.profiles.Profile) -> str:
    """Write *mask* to a temporary GeoTIFF and return the file path."""

    temp_dir = Path(tempfile.gettempdir()) / "mangrove_union"
    temp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=temp_dir)
    os.close(fd)

    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(mask, 1)

    return tmp_path


def _build_output_filename(tile_id: str) -> str:
    return f"{tile_id}__gmw_mangrove_any_year.tif"


def process_tile(
    tile_id: str,
    raster_paths: Sequence[str],
    bounds: Optional[TileBounds],
    *,
    force: bool = False,
) -> Optional[str]:
    """Union yearly rasters for *tile_id* and upload the Hansen tile to S3."""

    dataset_cfg = cn.datasets.get("mangrove_extent", {})
    local_dir = Path(dataset_cfg.get("local_processed", cn.local_temp_dir))
    local_dir.mkdir(parents=True, exist_ok=True)
    s3_output_dir = dataset_cfg.get("s3_processed")
    if not s3_output_dir:
        raise ValueError("mangrove_extent dataset configuration is missing 's3_processed'")

    output_name = _build_output_filename(tile_id)
    s3_key = pp.join(s3_output_dir, output_name)
    if not force and uu.s3_file_exists(cn.s3_bucket_name, s3_key):
        LOGGER.info("[%s] output already exists at s3://%s/%s; skipping.", tile_id, cn.s3_bucket_name, s3_key)
        return None

    union_result = _union_years(tile_id, raster_paths)
    if union_result is None:
        LOGGER.warning("[%s] no valid rasters were read; skipping tile.", tile_id)
        return None

    union_mask, profile = union_result

    temp_union = _write_temp_union(union_mask, profile)
    try:
        if bounds is None:
            LOGGER.warning("[%s] tile bounds missing from index shapefile; skipping upload.", tile_id)
            return None

        local_output = local_dir / output_name
        hz.hansenize_gdal(temp_union, str(local_output), bounds.as_tuple, nodata_value=0, dtype="Byte")

        uu.upload_file_to_s3(str(local_output), cn.s3_bucket_name, s3_key)
        local_output.unlink(missing_ok=True)
        LOGGER.info("[%s] uploaded union tile to s3://%s/%s", tile_id, cn.s3_bucket_name, s3_key)
        return s3_key
    finally:
        Path(temp_union).unlink(missing_ok=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Union yearly GMW mangrove rasters and hansenize them.")
    parser.add_argument(
        "--tile-id",
        action="append",
        dest="tile_ids",
        help="One or more Hansen tile IDs to process. Defaults to all tiles with data.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        help="Optional list of years to include (e.g. 2018 2019). Defaults to all available years.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess tiles even if an output already exists on S3.",
    )
    parser.add_argument(
        "--client",
        choices=["local", "coiled"],
        default="local",
        help="Execution environment for parallel processing (default: local).",
    )
    parser.add_argument(
        "--cluster-name",
        default="mangrove_extent_union",
        help="Name of the Coiled cluster to use when --client=coiled.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    dataset_cfg = cn.datasets.get("mangrove_extent")
    if not dataset_cfg:
        raise RuntimeError("mangrove_extent dataset configuration is missing from preprocessing constants.")

    raw_root = dataset_cfg.get("s3_raw_root")
    if not raw_root:
        raise RuntimeError("mangrove_extent dataset configuration requires 's3_raw_root'.")

    s3_client = boto3.client("s3", region_name=cn.s3_region_name)

    available_years = _list_available_years(s3_client, raw_root)
    LOGGER.info("Available years under %s: %s", raw_root, ", ".join(available_years) or "<none>")

    requested_years = args.years if args.years else available_years
    if requested_years:
        missing_years = sorted(set(requested_years) - set(available_years))
        if missing_years:
            LOGGER.warning("Requested years not found on S3: %s", ", ".join(missing_years))
        requested_years = [y for y in requested_years if y in available_years]

    tile_sources = _gather_tile_sources(s3_client, raw_root, requested_years)
    if not tile_sources:
        LOGGER.warning("No rasters found for requested filters; nothing to process.")
        return

    bounds_lookup = _load_tile_bounds()

    tile_ids: Sequence[str]
    if args.tile_ids:
        requested = set(args.tile_ids)
        tile_ids = [tid for tid in tile_sources.keys() if tid in requested]
        missing_tiles = requested - set(tile_ids)
        if missing_tiles:
            LOGGER.warning("No rasters found for requested tile IDs: %s", ", ".join(sorted(missing_tiles)))
    else:
        tile_ids = sorted(tile_sources.keys())

    if not tile_ids:
        LOGGER.info("No tiles to process after applying filters.")
        return

    LOGGER.info("Processing %d tiles using years: %s", len(tile_ids), ", ".join(requested_years) or "<none>")

    tasks = []
    for tile_id in tile_ids:
        rasters = sorted(tile_sources[tile_id])
        if not rasters:
            LOGGER.info("[%s] no rasters found; skipping.", tile_id)
            continue
        bounds = bounds_lookup.get(tile_id)
        tasks.append(delayed(process_tile)(tile_id, rasters, bounds, force=args.force))

    if not tasks:
        LOGGER.info("No processing tasks were created; exiting.")
        return

    cluster = None
    client: Optional[Client] = None
    try:
        if args.client == "coiled":
            cluster, client, run_local = uu.connect_to_cluster(cluster_name=args.cluster_name)
            if run_local:
                LOGGER.info(
                    "Coiled cluster '%s' unavailable; falling back to a local Dask cluster.",
                    args.cluster_name,
                )
                cluster = LocalCluster(processes=False, dashboard_address=None)
                client = Client(cluster)
            else:
                LOGGER.info("Running on Coiled cluster '%s'.", cluster.name if cluster else args.cluster_name)
        else:
            LOGGER.info("Running tasks on a local Dask cluster.")
            cluster = LocalCluster(processes=False, dashboard_address=None)
            client = Client(cluster)

        dask.compute(*tasks)
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()