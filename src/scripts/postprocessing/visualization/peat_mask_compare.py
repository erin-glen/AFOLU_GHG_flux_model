#!/usr/bin/env python
"""
peatmask_compare.py

Visualize peat mask tiles from multiple datasets side by side.

Usage:
    python -m scripts.visualization.peatmask_compare --tile_id 00N_110E --datasets peatml gpd peatmap ogh

Recommended environment dependencies:
    - rioxarray
    - rasterio
    - matplotlib
"""
from __future__ import annotations

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import rioxarray

import src.scripts.preprocessing.preprocessing_constants as cn

BUCKET = cn.s3_bucket_name


def build_tile_path(ds_key: str, tile_id: str) -> str:
    """Construct the full S3 path for a tile."""
    prefix = cn.datasets["peat"][ds_key]["s3_processed"]
    tile_name = f"{tile_id}_{ds_key}_mask.tif"
    path = os.path.join(prefix, tile_name).replace("\\", "/")
    if not path.startswith("s3://"):
        path = f"s3://{BUCKET}/{path.lstrip('/')}"
    return path


def load_raster(vsis3_path: str):
    """Attempt to load a raster and return the DataArray or None."""
    try:
        return rioxarray.open_rasterio(vsis3_path, masked=True)
    except Exception as exc:  # pragma: no cover - visualization helper
        warnings.warn(f"Failed to load {vsis3_path}: {exc}")
        return None


def plot_datasets(tile_id: str, datasets: list[str]) -> None:
    arrays = []
    labels = []
    for ds in datasets:
        s3_path = build_tile_path(ds, tile_id)
        vsis3_path = s3_path.replace("s3://", "/vsis3/")
        arr = load_raster(vsis3_path)
        if arr is None:
            continue
        arrays.append(arr)
        labels.append(ds)

    if not arrays:
        warnings.warn("No datasets were loaded; nothing to plot")
        return

    fig, axes = plt.subplots(1, len(arrays), figsize=(5 * len(arrays), 5), sharex=True, sharey=True)
    if not isinstance(axes, (list, tuple)):
        axes = [axes]

    extent = arrays[0].rio.bounds()
    cmap = "viridis"
    for ax, arr, label in zip(axes, arrays, labels):
        arr.plot(ax=ax, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(label)
        ax.set_xlim(extent[0], extent[2])
        ax.set_ylim(extent[1], extent[3])

    plt.tight_layout()
    plt.show()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Visualize peat mask tiles")
    parser.add_argument("--tile_id", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args(argv)

    plot_datasets(args.tile_id, args.datasets)


if __name__ == "__main__":
    main()