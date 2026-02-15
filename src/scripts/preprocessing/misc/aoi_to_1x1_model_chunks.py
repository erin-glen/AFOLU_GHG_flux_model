"""Create a model-ready 1x1-degree chunk shapefile from an AOI shapefile.

This script selects 1x1 chunk polygons from the canonical fishnet that
intersect a user-supplied AOI. The output is intended for use with the
organic-soils drainage model ``--chunk_shapefile_uri`` input.

Requirements for model readiness:
- output must contain ``chunk_id`` (format W_S_E_N)
- output should contain ``iso``
- CRS should match the 1x1 fishnet CRS (typically EPSG:4326)
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import geopandas as gpd

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu


REQUIRED_COLUMNS = ("chunk_id", "iso")


def _read_vector(path_or_uri: str) -> gpd.GeoDataFrame:
    """Read a local or S3 shapefile into a GeoDataFrame."""
    if path_or_uri.startswith("s3://"):
        bucket, key = uu.split_s3_path(path_or_uri)
        prefix = key[:-4] if key.lower().endswith(".shp") else key
        with tempfile.TemporaryDirectory(prefix="aoi_chunks_") as td:
            return uu.read_shapefile_from_s3(prefix, td, bucket)
    return gpd.read_file(path_or_uri)


def build_model_ready_chunks(
    aoi_path: str,
    output_path: str,
    fishnet_path: str = cn.fishnet_1x1deg_uri,
    keep_all_fishnet_columns: bool = False,
) -> gpd.GeoDataFrame:
    """Return 1x1 chunks intersecting AOI and write them to ``output_path``."""
    fishnet = _read_vector(fishnet_path)
    aoi = _read_vector(aoi_path)

    if fishnet.empty:
        raise ValueError(f"Fishnet has no features: {fishnet_path}")
    if aoi.empty:
        raise ValueError(f"AOI has no features: {aoi_path}")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in fishnet.columns]
    if missing_cols:
        raise ValueError(
            f"Fishnet is missing required model columns: {missing_cols}. "
            "Use a fishnet with at least chunk_id and iso."
        )

    if aoi.crs is None:
        raise ValueError("AOI has no CRS; define CRS before running this script.")
    if fishnet.crs is None:
        raise ValueError("Fishnet has no CRS; cannot align AOI to fishnet grid.")

    if aoi.crs != fishnet.crs:
        aoi = aoi.to_crs(fishnet.crs)

    # Robust unary union across geopandas/shapely versions.
    aoi_union = aoi.geometry.union_all() if hasattr(aoi.geometry, "union_all") else aoi.unary_union
    selected = fishnet[fishnet.intersects(aoi_union)].copy()

    if selected.empty:
        raise ValueError("No 1x1 fishnet chunks intersect the AOI.")

    selected = selected.sort_values("chunk_id").reset_index(drop=True)

    if not keep_all_fishnet_columns:
        selected = selected[["chunk_id", "iso", "geometry"]]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Shapefile sidecars are written automatically by geopandas.
    selected.to_file(out)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create model-ready 1x1-degree chunk shapefile from an AOI shapefile"
    )
    parser.add_argument(
        "--aoi",
        required=True,
        help="Path or s3:// URI to AOI shapefile (.shp)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output shapefile path (e.g. ./tmp/my_aoi_chunks.shp)",
    )
    parser.add_argument(
        "--fishnet",
        default=cn.fishnet_1x1deg_uri,
        help=(
            "Path or s3:// URI to 1x1 fishnet shapefile with chunk_id and iso "
            f"(default: {cn.fishnet_1x1deg_uri})"
        ),
    )
    parser.add_argument(
        "--keep_all_fishnet_columns",
        action="store_true",
        help="Keep all columns from the fishnet in output (default keeps chunk_id, iso, geometry)",
    )

    args = parser.parse_args()

    result = build_model_ready_chunks(
        aoi_path=args.aoi,
        output_path=args.output,
        fishnet_path=args.fishnet,
        keep_all_fishnet_columns=args.keep_all_fishnet_columns,
    )

    print(
        f"Wrote {len(result)} model-ready 1x1 chunks to {os.path.abspath(args.output)}"
    )


if __name__ == "__main__":
    main()

"""
python -m src.scripts.preprocessing.misc.aoi_to_1x1_model_chunks \
  --aoi /path/to/aoi.shp \
  --output /path/to/model_ready_chunks.shp

python -m src.scripts.preprocessing.misc.aoi_to_1x1_model_chunks \
  --aoi s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/wwf/raw/ESSF_OperationalLandscapes.shp \
  --output s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/wwf/fishnet/operational_lanscapes_1x1.shp
  
python -m src.scripts.preprocessing.misc.aoi_to_1x1_model_chunks \
  --aoi s3://my-bucket/my-prefix/aoi.shp \
  --fishnet s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp \
  --output /path/to/model_ready_chunks.shp
"""