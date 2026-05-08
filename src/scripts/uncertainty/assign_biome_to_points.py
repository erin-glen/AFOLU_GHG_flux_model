"""Assign biome labels to point observations by sampling the climate_domain raster.

Reads a CSV with latitude/longitude columns, samples the FAO ecozone raster
(already reclassified to tropical/boreal/temperate) at each point, and writes
the CSV back with a ``biome`` column appended.

Examples
--------
python -m src.scripts.uncertainty.assign_biome_to_points \
    --input  C:/tmp/uncertainty/USDA_testpoints_probability_v3.csv \
    --output C:/tmp/uncertainty/USDA_testpoints_probability_v3_biome.csv

python -m src.scripts.uncertainty.assign_biome_to_points \
    --input  C:/tmp/uncertainty/USDA_testpoints_probability_v3.csv \
    --output C:/tmp/uncertainty/USDA_testpoints_probability_v3_biome.csv \
    --lat-column latitude --lon-column longitude
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities.universal_utilities import xy_to_tile_id

CLIMATE_DOMAIN_DIR = cn.dirs["climate_domain"]
CLIMATE_DOMAIN_PATTERN = cn.patterns["climate_domain"]

ECOZONE_CODE_TO_NAME = {v: k for k, v in cn.ecozone_codes.items()}


def _table_path(path: str | os.PathLike[str]) -> str:
    text = os.fspath(path).replace("\\", "/")
    if text.startswith("s3://"):
        return text
    if text.startswith(f"{cn.project_dir}/") or text.startswith("climate/"):
        return f"s3://{cn.s3_bucket_name}/{text.lstrip('/')}"
    return str(Path(text).expanduser().resolve())


def _is_s3_path(path: str) -> bool:
    return path.startswith("s3://")


def _tile_raster_path(tile_id: str) -> str:
    s3_path = f"{CLIMATE_DOMAIN_DIR}{CLIMATE_DOMAIN_PATTERN.format(tile_id=tile_id)}"
    return s3_path.replace("s3://", "/vsis3/")


def _sample_tile_codes(src: rasterio.DatasetReader, coords: list[tuple[float, float]]) -> np.ndarray:
    """Sample a strip-organized tile with one raster read per touched row."""

    codes = np.zeros(len(coords), dtype=np.int16)
    row_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pos, (x, y) in enumerate(coords):
        row, col = src.index(x, y)
        if 0 <= row < src.height and 0 <= col < src.width:
            row_groups[row].append((pos, col))

    for row, entries in row_groups.items():
        cols = [col for _, col in entries]
        col_min = max(0, min(cols))
        col_max = min(src.width - 1, max(cols))
        data = src.read(
            1,
            window=Window(col_min, row, col_max - col_min + 1, 1),
            boundless=False,
        )
        for pos, col in entries:
            codes[pos] = int(data[0, col - col_min])
    return codes


def assign_biome_column(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    biome_col: str = "biome",
) -> pd.DataFrame:
    """Add a *biome* column by sampling the climate_domain raster at each point.

    Parameters
    ----------
    df : DataFrame
        Must contain *lat_col* and *lon_col* columns.
    lat_col, lon_col : str
        Column names for latitude and longitude.
    biome_col : str
        Name of the new column to create.

    Returns
    -------
    DataFrame
        Copy of *df* with *biome_col* appended.
    """
    out = df.copy()
    biomes = np.full(len(out), "unknown", dtype=object)

    tile_ids = out.apply(
        lambda r: xy_to_tile_id(r[lon_col], r[lat_col]), axis=1
    )

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, tid in enumerate(tile_ids):
        groups[tid].append(idx)

    sorted_groups = sorted(groups.items())
    n_tiles = len(sorted_groups)
    for i, (tid, indices) in enumerate(sorted_groups, 1):
        raster_path = _tile_raster_path(tid)
        try:
            with rasterio.Env(GDAL_CACHEMAX=512):
                with rasterio.open(raster_path) as src:
                    coords = [
                        (float(out.iloc[idx][lon_col]), float(out.iloc[idx][lat_col]))
                        for idx in indices
                    ]
                    for idx, code in zip(indices, _sample_tile_codes(src, coords)):
                        biomes[idx] = ECOZONE_CODE_TO_NAME.get(code, "unknown")
        except Exception as exc:
            print(f"  WARNING: could not open tile {tid} ({raster_path}): {exc}")
            for idx in indices:
                biomes[idx] = "unknown"
        if i % 5 == 0 or i == n_tiles:
            print(f"  [{i:3d}/{n_tiles}] tiles sampled", flush=True)

    out[biome_col] = biomes
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Assign biome labels to point observations from the climate_domain raster.",
    )
    p.add_argument(
        "--input", required=True,
        help="Input CSV with latitude/longitude columns. Local path, s3:// URI, or S3 key.",
    )
    p.add_argument(
        "--output", required=True,
        help="Output CSV path (input + biome column). Local path, s3:// URI, or S3 key.",
    )
    p.add_argument(
        "--lat-column", default="latitude",
        help="Name of the latitude column. Default: latitude",
    )
    p.add_argument(
        "--lon-column", default="longitude",
        help="Name of the longitude column. Default: longitude",
    )
    p.add_argument(
        "--biome-column", default="biome",
        help="Name of the biome column to create. Default: biome",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path = _table_path(args.input)
    if not _is_s3_path(input_path) and not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Read {len(df)} rows from {input_path}")

    for col in [args.lat_column, args.lon_column]:
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found. Available: {df.columns.tolist()}"
            )

    df = assign_biome_column(
        df,
        lat_col=args.lat_column,
        lon_col=args.lon_column,
        biome_col=args.biome_column,
    )

    counts = df[args.biome_column].value_counts()
    print("\nBiome distribution:")
    for name, count in counts.items():
        print(f"  {name}: {count:,}")

    output_path = _table_path(args.output)
    if not _is_s3_path(output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
