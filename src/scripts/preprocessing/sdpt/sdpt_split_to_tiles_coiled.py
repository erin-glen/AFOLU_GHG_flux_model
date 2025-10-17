"""Split SDPT polygons into 10x10 tiles with optional Dask/Coiled support."""

import argparse
import logging
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import fiona
import geopandas as gpd
import geopandas.io.file
import pandas as pd
from dask import compute, delayed
from packaging.version import InvalidVersion, Version

try:  # Shapely 2.x preferred import path
    from shapely import wkb as shapely_wkb
except ImportError:  # pragma: no cover - Shapely <2.0 fallback
    from shapely import wkb as shapely_wkb

from src.scripts.utilities import universal_utilities as uutil

# --- Robust GDAL Version Patch for ArcGIS Pro Environment ---
# This patch resolves the "Invalid version: '3.8.1e'" error that arises due to ArcGIS Pro's non-standard GDAL versioning.
original_to_file_fiona = geopandas.io.file._to_file_fiona

def patched_to_file_fiona(df, filename, driver, schema, crs=None, mode="w", **kwargs):
    try:
        gdal_version = fiona.env.get_gdal_release_name()
        Version(gdal_version)
    except InvalidVersion:
        gdal_version = "3.8.1"
        kwargs["GEOPANDAS_FIX_GDAL_VERSION"] = gdal_version

    original_version_class = geopandas.io.file.Version
    geopandas.io.file.Version = lambda _: Version("3.8.1")

    try:
        return original_to_file_fiona(df, filename, driver, schema, crs, mode, **kwargs)
    finally:
        geopandas.io.file.Version = original_version_class

geopandas.io.file._to_file_fiona = patched_to_file_fiona
# --- End GDAL Version Patch ---

LOGGER = logging.getLogger(__name__)


def _geom_to_wkb_bytes(geom) -> bytes:
    """Return a bytes representation of *geom* that survives pickling."""

    if hasattr(geom, "to_wkb"):
        data = geom.to_wkb()
    else:  # pragma: no cover - Shapely <2.0 fallback
        data = geom.wkb
    if isinstance(data, memoryview):
        return data.tobytes()
    return data


def get_layer_bounds(gdb_path: str, layer_name: str) -> Optional[Tuple[float, float, float, float]]:
    """Quickly obtain a layer bounding box from ``countries_gdb`` metadata."""

    try:
        with fiona.open(gdb_path, layer=layer_name) as src:
            return src.bounds if src.bounds else None
    except Exception as exc:  # pragma: no cover - logging aids debugging
        LOGGER.error("Error reading layer '%s': %s", layer_name, exc)
        return None


def bounding_boxes_intersect(b1: Sequence[float], b2: Sequence[float]) -> bool:
    """Return ``True`` if bounding boxes *b1* and *b2* intersect."""

    return not (b2[0] > b1[2] or b2[2] < b1[0] or b2[1] > b1[3] or b2[3] < b1[1])


def filter_to_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only polygon geometries for shapefile compatibility."""

    return gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()


def _process_tile(
    tile_id: str,
    tile_geom_wkb: bytes,
    tile_bounds: Tuple[float, float, float, float],
    out_dir: str,
    countries_gdb: str,
    relevant_layers: Sequence[str],
    overwrite: bool = False,
) -> Tuple[str, str]:
    """Process a single tile and return ``(tile_id, status)``."""

    tile_out_path = os.path.join(out_dir, f"tile_{tile_id}.shp")
    if os.path.exists(tile_out_path) and not overwrite:
        return tile_id, "exists"

    tile_geom = shapely_wkb.loads(tile_geom_wkb)
    all_clips: List[gpd.GeoDataFrame] = []

    for layer_name in relevant_layers:
        try:
            ctry_gdf = gpd.read_file(countries_gdb, layer=layer_name, bbox=tile_bounds)
        except Exception as exc:  # pragma: no cover - safeguards remote workers
            LOGGER.error("Tile %s: error reading layer '%s': %s", tile_id, layer_name, exc)
            continue

        if ctry_gdf.empty:
            LOGGER.debug("Tile %s: layer '%s' empty in bbox", tile_id, layer_name)
            continue

        clipped = gpd.clip(ctry_gdf, tile_geom)
        if not clipped.empty:
            clipped["source_layer"] = layer_name
            all_clips.append(clipped)

    if not all_clips:
        return tile_id, "empty"

    merged_gdf = gpd.GeoDataFrame(
        pd.concat(all_clips, ignore_index=True), crs=all_clips[0].crs
    )
    merged_gdf = filter_to_polygons(merged_gdf)

    if merged_gdf.empty:
        return tile_id, "no_polygons"

    os.makedirs(out_dir, exist_ok=True)
    merged_gdf.to_file(tile_out_path)
    return tile_id, f"written:{len(merged_gdf)}"


def build_tile_tasks(
    tiles_gdf: gpd.GeoDataFrame,
    tile_id_field: str,
    countries_gdb: str,
    layer_bounds: Iterable[Tuple[str, Tuple[float, float, float, float]]],
    out_dir: str,
    target_tile_ids: Optional[Sequence[str]] = None,
    overwrite: bool = False,
) -> List["delayed"]:
    """Create Dask delayed tasks for each requested tile."""

    bounds_lookup = list(layer_bounds)
    tasks = []

    for _, tile_row in tiles_gdf.iterrows():
        tile_id = tile_row[tile_id_field]
        if target_tile_ids and tile_id not in target_tile_ids:
            continue

        tile_geom = tile_row.geometry
        tile_bounds = tile_geom.bounds
        relevant_layers = [
            name for name, bounds in bounds_lookup if bounding_boxes_intersect(bounds, tile_bounds)
        ]

        if not relevant_layers:
            LOGGER.info("Tile %s: no intersecting layers; skipping.", tile_id)
            continue

        tile_geom_wkb = _geom_to_wkb_bytes(tile_geom)
        task = delayed(_process_tile)(
            tile_id,
            tile_geom_wkb,
            tile_bounds,
            out_dir,
            countries_gdb,
            relevant_layers,
            overwrite,
        )
        tasks.append(task)

    return tasks


def split_sdpt_by_tiles(
    tile_grid_path: str,
    countries_gdb: str,
    out_dir: str,
    tile_id_field: str = "tile_id",
    target_tile_ids: Optional[Sequence[str]] = None,
    client: str = "local",
    cluster_name: str = "sdpt_split",
    overwrite: bool = False,
) -> List[Tuple[str, str]]:
    """Split SDPT polygons by tile, optionally leveraging Coiled."""

    LOGGER.info("Reading tile grid: %s", tile_grid_path)
    tiles_gdf = gpd.read_file(tile_grid_path)
    if tile_id_field not in tiles_gdf.columns:
        raise ValueError(f"Tile grid missing field '{tile_id_field}'.")

    all_layer_names = fiona.listlayers(countries_gdb)
    LOGGER.info("Found %d layers in '%s'.", len(all_layer_names), countries_gdb)

    layer_bounds: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for layer_name in all_layer_names:
        bounds = get_layer_bounds(countries_gdb, layer_name)
        if bounds is None:
            continue
        layer_bounds.append((layer_name, bounds))

    if not layer_bounds:
        LOGGER.error("No valid layers discovered in %s.", countries_gdb)
        return []

    tasks = build_tile_tasks(
        tiles_gdf,
        tile_id_field,
        countries_gdb,
        layer_bounds,
        out_dir,
        target_tile_ids=target_tile_ids,
        overwrite=overwrite,
    )

    if not tasks:
        LOGGER.warning("No tiles selected for processing.")
        return []

    cluster = dask_client = None
    run_local = client == "local"

    if client == "coiled":
        cluster, dask_client, run_local = uutil.connect_to_cluster(
            cluster_name=cluster_name,
            n_workers=20,
            region="us-east-1",
            run_local=False,
        )
        if run_local:
            LOGGER.warning(
                "Requested coiled client but cluster '%s' unavailable; running locally.",
                cluster_name,
            )
        else:
            LOGGER.info("Using Coiled cluster: %s", cluster.name)

    try:
        LOGGER.info("Submitting %d tile task(s).", len(tasks))
        results: List[Tuple[str, str]] = list(compute(*tasks))
        for tile_id, status in results:
            LOGGER.info("Tile %s => %s", tile_id, status)
        return results
    finally:
        if dask_client and not run_local:
            dask_client.close()
        if cluster and not run_local:
            cluster.close()


def parse_tile_ids(tile_args: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Normalize ``--tile-id`` CLI arguments allowing comma-separated values."""

    if not tile_args:
        return None
    tile_ids: List[str] = []
    for raw in tile_args:
        for part in raw.split(","):
            val = part.strip()
            if val:
                tile_ids.append(val)
    return tile_ids or None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split SDPT polygons into per-tile shapefiles with optional Coiled acceleration. "
            "Outputs are written as tile_<tile_id>.shp in the specified directory."
        )
    )
    parser.add_argument(
        "--tile-grid", required=True, help="Path to the tile grid shapefile (or GeoPackage)."
    )
    parser.add_argument(
        "--countries-gdb", required=True, help="Path to the SDPT geodatabase (file GDB)."
    )
    parser.add_argument(
        "--out-dir", required=True, help="Directory where per-tile shapefiles will be written."
    )
    parser.add_argument(
        "--tile-id-field", default="tile_id", help="Name of the tile identifier column (default: tile_id)."
    )
    parser.add_argument(
        "--tile-id",
        action="append",
        help="Specific tile ID(s) to process. Repeat or pass comma-separated values to limit the run.",
    )
    parser.add_argument(
        "--client",
        choices=["local", "coiled"],
        default="local",
        help=(
            "Execution environment: 'local' uses the default Dask scheduler; "
            "'coiled' connects to a Coiled cluster."
        ),
    )
    parser.add_argument(
        "--cluster-name",
        default="sdpt_split",
        help="Coiled cluster name to connect to when --client=coiled (default: sdpt_split).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild tiles even if the output shapefile already exists.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    tile_ids = parse_tile_ids(args.tile_id)

    split_sdpt_by_tiles(
        tile_grid_path=args.tile_grid,
        countries_gdb=args.countries_gdb,
        out_dir=args.out_dir,
        tile_id_field=args.tile_id_field,
        target_tile_ids=tile_ids,
        client=args.client,
        cluster_name=args.cluster_name,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()