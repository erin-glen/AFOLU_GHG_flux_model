"""Create a global coarsened OGH organic-soil probability map.

The source OGH tiles are 30 m-ish uint8 probabilities on a 0..100 scale. This
workflow aggregates them to a coarser regular grid and writes a single global
GeoTIFF. The default aggregation is block max so thresholding the coarsened
probability map matches "any native pixel in the output cell meets threshold".
"""

from __future__ import annotations

import argparse
import json
import logging
import posixpath
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import dask
import numpy as np
import rasterio
from dask.distributed import as_completed
from rasterio.transform import from_bounds
from rasterio.windows import Window

from src.scripts.postprocessing.visualization.create_organic_soil_presence_map import (
    DEFAULT_NATIVE_DEG,
    NODATA,
    PROBABILITY_TILE_SUFFIX,
    _aggregation_factor,
    _connect_client,
    _join_s3,
    _s3_to_vsi,
    _split_s3,
    _upload_file,
    _write_json,
    list_probability_tiles,
    parse_tile_ids,
    probability_tile_prefix,
)
from src.scripts.utilities import log_utilities as lu
from src.scripts.utilities import universal_utilities as uu


OUTPUT_ROOT = (
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/"
    "uncertainty/global_organic_soil_probability"
)
DEFAULT_TARGET_DEG = 0.005
DEFAULT_BLOCK_OUT_ROWS = 100


@dataclass(frozen=True)
class ProbabilityTileResult:
    tile_id: str
    bounds: tuple[float, float, float, float]
    data: np.ndarray
    nonzero_pixels: int
    max_probability: int
    status: str


def _label_for_deg(target_deg: float) -> str:
    return str(target_deg).replace(".", "_").rstrip("0").rstrip("_")


def _output_name(probability_date: str, target_deg: float, aggregation: str) -> str:
    label = _label_for_deg(target_deg)
    return (
        f"{label}deg_global_organic_soil_probability__"
        f"{probability_date}__{aggregation}.tif"
    )


def _default_output_prefix(probability_date: str, target_deg: float, aggregation: str) -> str:
    return posixpath.join(
        OUTPUT_ROOT,
        probability_date,
        aggregation,
        f"{_label_for_deg(target_deg)}deg",
    )


def aggregate_probability_block(probability: np.ndarray, factor: int, aggregation: str) -> np.ndarray:
    out_rows = probability.shape[0] // factor
    out_cols = probability.shape[1] // factor
    probability = probability[: out_rows * factor, : out_cols * factor]
    blocks = probability.reshape(out_rows, factor, out_cols, factor)
    if aggregation == "max":
        return blocks.max(axis=(1, 3)).astype(np.uint8)
    if aggregation == "mean":
        return np.rint(blocks.mean(axis=(1, 3))).astype(np.uint8)
    raise ValueError(f"Unsupported aggregation {aggregation!r}")


def aggregate_probability_tile(
    tile_id: str,
    probability_prefix: str,
    native_deg: float,
    target_deg: float,
    block_out_rows: int,
    aggregation: str,
) -> ProbabilityTileResult:
    probability_path = _join_s3(probability_prefix, f"{tile_id}{PROBABILITY_TILE_SUFFIX}")
    bounds = tuple(float(value) for value in uu.get_10x10_tile_bounds(tile_id))
    factor = _aggregation_factor(native_deg, target_deg)
    empty_shape = (int(round(10 / target_deg)), int(round(10 / target_deg)))

    with rasterio.Env():
        try:
            probability_src = rasterio.open(_s3_to_vsi(probability_path))
        except Exception:
            empty = np.zeros(empty_shape, dtype=np.uint8)
            return ProbabilityTileResult(tile_id, bounds, empty, 0, 0, "missing_probability_tile")

        with probability_src:
            out_rows = probability_src.height // factor
            out_cols = probability_src.width // factor
            out = np.zeros((out_rows, out_cols), dtype=np.uint8)
            for out_row0 in range(0, out_rows, block_out_rows):
                out_row1 = min(out_row0 + block_out_rows, out_rows)
                in_row0 = out_row0 * factor
                in_rows = (out_row1 - out_row0) * factor
                window = Window(0, in_row0, out_cols * factor, in_rows)
                probability = probability_src.read(1, window=window, boundless=False)
                out[out_row0:out_row1, :] = aggregate_probability_block(
                    probability,
                    factor,
                    aggregation,
                )

    return ProbabilityTileResult(
        tile_id=tile_id,
        bounds=bounds,
        data=out,
        nonzero_pixels=int((out > 0).sum()),
        max_probability=int(out.max()) if out.size else 0,
        status="ok",
    )


def iter_tile_results(
    tile_ids: list[str],
    client,
    probability_prefix: str,
    native_deg: float,
    target_deg: float,
    block_out_rows: int,
    aggregation: str,
) -> Iterator[ProbabilityTileResult]:
    delayed = [
        dask.delayed(aggregate_probability_tile)(
            tile_id,
            probability_prefix,
            native_deg,
            target_deg,
            block_out_rows,
            aggregation,
        )
        for tile_id in tile_ids
    ]
    if client is None:
        for result in dask.compute(*delayed):
            yield result
        return

    futures = client.compute(delayed)
    for future in as_completed(futures):
        yield future.result()


def _paste_tile(global_arr: np.memmap, tile: ProbabilityTileResult, target_deg: float) -> None:
    min_x, _min_y, _max_x, max_y = tile.bounds
    x0 = int(round((min_x + 180) / target_deg))
    y0 = int(round((90 - max_y) / target_deg))
    rows, cols = tile.data.shape
    global_arr[y0 : y0 + rows, x0 : x0 + cols] = tile.data


def _write_geotiff(path: Path, arr: np.ndarray, target_deg: float) -> None:
    rows, cols = arr.shape
    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_bounds(-180, -90, 180, 90, cols, rows),
        "nodata": NODATA,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "DEFLATE",
        "predictor": 1,
        "bigtiff": "YES",
        "num_threads": "ALL_CPUS",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.update_tags(
            probability_scale="0..100",
            aggregation="coarsened from native OGH probability tiles",
            nodata=str(NODATA),
        )


def run(args: argparse.Namespace) -> dict:
    logger = lu.setup_logging()
    probability_prefix = args.probability_tile_prefix or probability_tile_prefix(args.probability_date)
    tile_ids = parse_tile_ids(args.tile_id)
    if not tile_ids:
        tile_ids = list_probability_tiles(probability_prefix, limit=args.limit_tiles)
    if not tile_ids:
        raise RuntimeError(f"No probability tiles found under {probability_prefix}")

    rows = int(round(180 / args.target_deg))
    cols = int(round(360 / args.target_deg))
    output_prefix = args.output_prefix or _default_output_prefix(
        args.probability_date,
        args.target_deg,
        args.aggregation,
    )
    output_name = args.output_name or _output_name(
        args.probability_date,
        args.target_deg,
        args.aggregation,
    )
    output_path = (
        _join_s3(output_prefix, output_name)
        if output_prefix.startswith("s3://")
        else str(Path(output_prefix) / output_name)
    )

    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="organic_soil_probability_"))
    workdir.mkdir(parents=True, exist_ok=True)
    memmap_path = workdir / "global_probability.dat"
    local_tif = workdir / output_name
    summary_path = workdir / output_name.replace(".tif", ".summary.json")

    logger.info(
        "Creating organic soil probability map | date=%s aggregation=%s tiles=%d output=%s",
        args.probability_date,
        args.aggregation,
        len(tile_ids),
        output_path,
    )

    global_arr = np.memmap(memmap_path, dtype=np.uint8, mode="w+", shape=(rows, cols))
    global_arr[:] = 0

    client = _connect_client(args.cluster_name, args.run_local, args.local_workers)
    completed = 0
    nonzero_pixels = 0
    max_probability = 0
    statuses: dict[str, int] = {}
    try:
        for tile in iter_tile_results(
            tile_ids,
            client,
            probability_prefix,
            args.native_deg,
            args.target_deg,
            args.block_out_rows,
            args.aggregation,
        ):
            _paste_tile(global_arr, tile, args.target_deg)
            completed += 1
            nonzero_pixels += tile.nonzero_pixels
            max_probability = max(max_probability, tile.max_probability)
            statuses[tile.status] = statuses.get(tile.status, 0) + 1
            if completed % args.log_every == 0 or completed == len(tile_ids):
                logger.info(
                    "Completed %d/%d tiles; nonzero pixels=%d; max probability=%d",
                    completed,
                    len(tile_ids),
                    nonzero_pixels,
                    max_probability,
                )
    finally:
        if client is not None:
            client.close()

    global_arr.flush()
    _write_geotiff(local_tif, global_arr, args.target_deg)

    summary = {
        "probability_date": args.probability_date,
        "probability_tile_prefix": probability_prefix,
        "aggregation": args.aggregation,
        "target_deg": args.target_deg,
        "native_deg": args.native_deg,
        "tile_count": len(tile_ids),
        "status_counts": statuses,
        "nonzero_pixels": int(nonzero_pixels),
        "max_probability_0_to_100": int(max_probability),
        "output_path": output_path,
    }
    _write_json(summary_path, summary)

    if output_path.startswith("s3://"):
        _upload_file(local_tif, output_path)
        _upload_file(summary_path, output_path.replace(".tif", ".summary.json"))
    else:
        final_path = Path(output_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if local_tif.resolve() != final_path.resolve():
            shutil.copy2(local_tif, final_path)
            shutil.copy2(summary_path, final_path.with_suffix(".summary.json"))

    logger.info("Wrote organic soil probability map: %s", output_path)
    logger.info("Wrote summary: %s", output_path.replace(".tif", ".summary.json"))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability_date", required=True)
    parser.add_argument("--probability_tile_prefix", default=None)
    parser.add_argument("--aggregation", choices=["max", "mean"], default="max")
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument("--output_name", default=None)
    parser.add_argument("--target_deg", type=float, default=DEFAULT_TARGET_DEG)
    parser.add_argument("--native_deg", type=float, default=DEFAULT_NATIVE_DEG)
    parser.add_argument("--block_out_rows", type=int, default=DEFAULT_BLOCK_OUT_ROWS)
    parser.add_argument("--tile_id", action="append")
    parser.add_argument("--limit_tiles", type=int, default=None)
    parser.add_argument("--cluster_name", "-cn", default="organic_soil_probability_maps")
    parser.add_argument("--run_local", action="store_true")
    parser.add_argument("--local_workers", type=int, default=4)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--log_every", type=int, default=25)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
