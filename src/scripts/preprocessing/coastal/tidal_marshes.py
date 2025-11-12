"""Hansenize coastal tidal marsh extent rasters into Hansen-aligned tiles.

This script reads the tidal marsh extent tiles stored on S3 under
``climate/AFOLU_flux_model/organic_soils/inputs/raw/coastal/tidal_marsh``.
Each raster is a 10x10 degree binary dataset at 10 m resolution.  The script
clips every source tile to the standard Hansen grid, resamples it to the
0.00025 degree Hansen resolution, and uploads the resulting rasters to the
organic soils processed inputs bucket under the tidal marshes directory.

Example
-------
Process every tile with a local Dask cluster::

    python -m src.scripts.preprocessing.coastal.tidal_marshes

Process a small subset of tiles and overwrite existing outputs::

    python -m src.scripts.preprocessing.coastal.tidal_marshes \
        --tile-id 00N_080W --tile-id 10S_120E --reprocess

Run the workflow on an existing Coiled cluster::

    python -m src.scripts.preprocessing.coastal.tidal_marshes \
        --client coiled --cluster-name tidal_marshes
"""

from __future__ import annotations

import argparse
import logging
import posixpath as pp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import boto3
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster
import geopandas as gpd
import rasterio
from rasterio import Env as RasterioEnv
from rasterio.session import AWSSession
from shapely.geometry import box

import src.scripts.preprocessing.preprocessing_constants as cn
from src.scripts.preprocessing.hansenize import hansenize_gdal as hz
from src.scripts.utilities import universal_utilities as uu

LOGGER = logging.getLogger("tidal-marshes")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass(frozen=True)
class TileBounds:
    """Lightweight container describing Hansen tile bounds."""

    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def as_tuple(self) -> Tuple[float, float, float, float]:
        return self.minx, self.miny, self.maxx, self.maxy


def _to_vsis3(path: str) -> str:
    """Convert an S3 URL to a ``/vsis3/`` GDAL path when needed."""

    if path.startswith("/vsis3/"):
        return path
    if path.startswith("s3://"):
        bucket_key = path[len("s3://") :]
        return f"/vsis3/{bucket_key}"
    return path


def _list_raw_rasters(s3_client: boto3.client, prefix: str) -> List[str]:
    """Return S3 URLs to every raster located under *prefix*."""

    paginator = s3_client.get_paginator("list_objects_v2")
    rasters: List[str] = []
    normalized = prefix.rstrip("/") + "/"
    for page in paginator.paginate(Bucket=cn.s3_bucket_name, Prefix=normalized):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if not key or not key.lower().endswith(".tif"):
                continue
            rasters.append(f"s3://{cn.s3_bucket_name}/{key}")
    return rasters


def _load_tile_index() -> gpd.GeoDataFrame:
    """Download and return the Hansen tile index as a GeoDataFrame."""

    local_dir = Path(cn.local_temp_dir) / "tile_index"
    local_dir.mkdir(parents=True, exist_ok=True)
    uu.download_shapefile_from_s3(cn.index_shapefile_prefix, str(local_dir), cn.s3_bucket_name)
    shapefile_path = local_dir / cn.tile_index_shapefile_name

    gdf = gpd.read_file(shapefile_path)
    gdf = gdf[gdf.geometry.notnull()].copy()
    return gdf


def _build_tile_bounds(tile_index: gpd.GeoDataFrame) -> Dict[str, TileBounds]:
    """Convert the Hansen tile index into a lookup of ``TileBounds``."""

    bounds: Dict[str, TileBounds] = {}
    for _, row in tile_index.iterrows():
        tile_id = str(row.get("tile_id"))
        if not tile_id:
            continue
        minx, miny, maxx, maxy = row.geometry.bounds
        bounds[tile_id] = TileBounds(minx=float(minx), miny=float(miny), maxx=float(maxx), maxy=float(maxy))
    return bounds


def _build_dataset_index(raster_paths: Iterable[str]) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame describing the spatial footprint of each raster."""

    records: List[Dict[str, object]] = []
    session = boto3.Session()
    aws_session = AWSSession(session)

    for path in raster_paths:
        vsi_path = _to_vsis3(path)
        try:
            with RasterioEnv(aws_session=aws_session):
                with rasterio.open(vsi_path) as src:
                    geometry = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        except Exception as exc:  # pragma: no cover - S3/driver issues
            LOGGER.warning("Failed to read bounds for %s: %s", path, exc)
            continue

        records.append({"source_path": path, "geometry": geometry})

    if not records:
        return gpd.GeoDataFrame(columns=["source_path", "geometry"], geometry="geometry", crs="EPSG:4326")

    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def _ensure_dataset_config() -> Dict[str, str]:
    cfg = cn.datasets.get("tidal_marshes")
    if not cfg:
        raise RuntimeError("tidal_marshes dataset configuration is missing from preprocessing constants.")
    required_keys = ("s3_raw", "s3_processed", "local_processed")
    for key in required_keys:
        if key not in cfg:
            raise RuntimeError(f"tidal_marshes dataset configuration is missing '{key}'")
    return cfg


def _process_tile(
    tile_id: str,
    source_paths: Sequence[str],
    bounds: Optional[TileBounds],
    dataset_cfg: Dict[str, str],
    *,
    force: bool = False,
) -> Optional[str]:
    """Hansenize a tidal marsh raster and upload the result to S3."""

    if bounds is None:
        LOGGER.warning("[%s] Missing bounds in tile index; skipping tile.", tile_id)
        return None

    if not source_paths:
        LOGGER.info("[%s] No tidal marsh rasters intersect this tile; skipping.", tile_id)
        return None

    output_dir = dataset_cfg["s3_processed"]
    output_name = f"{tile_id}.tif"
    s3_key = pp.join(output_dir, output_name)

    if not force and uu.s3_file_exists(cn.s3_bucket_name, s3_key):
        LOGGER.info("[%s] Output already exists at s3://%s/%s; skipping.", tile_id, cn.s3_bucket_name, s3_key)
        return None

    local_dir = Path(dataset_cfg.get("local_processed", cn.local_temp_dir))
    local_dir.mkdir(parents=True, exist_ok=True)

    local_output = local_dir / output_name
    try:
        vsis3_paths = [_to_vsis3(path) for path in source_paths]
        hz_input: Iterable[str] | str
        if len(vsis3_paths) == 1:
            hz_input = vsis3_paths[0]
        else:
            hz_input = list(vsis3_paths)

        hz.hansenize_gdal(hz_input, str(local_output), bounds.as_tuple, nodata_value=0, dtype="Byte")
        uu.upload_file_to_s3(str(local_output), cn.s3_bucket_name, s3_key)
    finally:
        local_output.unlink(missing_ok=True)

    LOGGER.info("[%s] Uploaded Hansenized raster to s3://%s/%s", tile_id, cn.s3_bucket_name, s3_key)
    return s3_key


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hansenize tidal marsh raster tiles.")
    parser.add_argument(
        "--tile-id",
        action="append",
        dest="tile_ids",
        help="One or more Hansen tile IDs to process. Defaults to all tiles with available rasters.",
    )
    parser.add_argument(
        "--client",
        choices=["local", "coiled"],
        default="local",
        help="Execution environment for parallel processing (default: local).",
    )
    parser.add_argument(
        "--cluster-name",
        default="tidal_marshes",
        help="Name of the Coiled cluster to use when --client=coiled.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess tiles even if an output already exists on S3.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    dataset_cfg = _ensure_dataset_config()
    raw_prefix = dataset_cfg["s3_raw"]

    s3_client = boto3.client("s3", region_name=cn.s3_region_name)
    raster_paths = _list_raw_rasters(s3_client, raw_prefix)
    if not raster_paths:
        LOGGER.warning("No tidal marsh rasters found under %s", raw_prefix)
        return

    tile_index = _load_tile_index()
    if tile_index.empty:
        LOGGER.warning("Global Hansen tile index is empty; aborting.")
        return

    bounds_lookup = _build_tile_bounds(tile_index)

    dataset_index = _build_dataset_index(raster_paths)
    if dataset_index.empty:
        LOGGER.warning("No tidal marsh rasters could be read; nothing to process.")
        return

    if args.tile_ids:
        requested = set(args.tile_ids)
        tile_index = tile_index[tile_index["tile_id"].isin(requested)]
        missing = requested - set(tile_index["tile_id"])
        if missing:
            LOGGER.warning("Requested tiles missing from Hansen index: %s", ", ".join(sorted(missing)))

    if tile_index.empty:
        LOGGER.info("No tiles to process after applying filters.")
        return

    LOGGER.info("Preparing to process %d tidal marsh tiles.", len(tile_index))

    tasks = []
    dataset_sindex = dataset_index.sindex if not dataset_index.empty else None

    for _, row in tile_index.iterrows():
        tile_id = str(row.get("tile_id"))
        tile_geom = row.geometry
        if not tile_id or tile_geom is None:
            continue

        candidate_idx: Iterable[int]
        if dataset_sindex is not None:
            candidate_idx = dataset_sindex.intersection(tile_geom.bounds)
            candidates = dataset_index.iloc[list(candidate_idx)]
        else:
            candidates = dataset_index

        intersecting = candidates[candidates.intersects(tile_geom)]
        source_paths = list(intersecting["source_path"])
        if not source_paths:
            LOGGER.info("[%s] No tidal marsh rasters intersect this tile; skipping.", tile_id)
            continue

        bounds = bounds_lookup.get(tile_id)
        tasks.append(
            delayed(_process_tile)(
                tile_id,
                source_paths,
                bounds,
                dataset_cfg,
                force=args.reprocess,
            )
        )

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