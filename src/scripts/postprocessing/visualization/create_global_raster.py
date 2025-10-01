"""
Stage 01: aggregate 10×10° tiles into global rasters at a coarser resolution.

Assumptions (this version)
--------------------------
- All non-integer inputs are **per-pixel totals** (e.g., Mg yr^-1 per native pixel).
- Aggregation for float datasets is **SUM** to the target grid.
- Aggregation for integer datasets is **MODE** (categorical majority).
- No pixel-area products are used; no unit conversions are performed.

Examples
--------
# Aggregate to the default 0.04° target grid (uploads canonical TIFFs):
python -m src.scripts.postprocessing.visualization.create_global_raster \
  -cn create_maps --run_name ogh_sensitivity_1km \
  --model_version 0_8_0 --date_tag 20250923

# Aggregate at 0.01° using a local Dask cluster instead of AWS Batch / ECS:
python -m src.scripts.postprocessing.visualization.create_global_raster \
  -cn local --run_name ogh_sensitivity_1km --run_local \
  --model_version 0_8_0 --date_tag 20250923 --target_deg 0.01
"""

from __future__ import annotations

import argparse
import posixpath
from typing import List, Optional, Tuple

import dask
from dask.distributed import as_completed
import numpy as np

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import log_utilities as lu
from src.scripts.utilities import universal_utilities as uu

from src.scripts.postprocessing.visualization.create_global_map_common import (
    DEFAULT_DATE_TAG,
    DEFAULT_MODEL_VERSION,
    DEFAULT_NATIVE_DEG,
    DEFAULT_TARGET_DEG,
    INTEGER_DATASETS,
    OUTPUT_ROOT,
    assert_grid_divides_world,
    build_download_upload_dict,
    deg_to_label,
    resolve_versioned_paths,
)

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _reaggregate_sum(arr: np.ndarray, native_deg: float, target_deg: float) -> np.ndarray:
    """
    Downsample by summing factor×factor blocks (preserve NaNs).
    If a block has 0 valid cells, output is NaN.
    """
    factor_f = target_deg / native_deg
    if not np.isclose(round(factor_f), factor_f):
        raise ValueError(f"target_deg/native_deg must be an integer; got {target_deg}/{native_deg}.")
    f = int(round(factor_f))

    h, w = arr.shape
    H = (h // f) * f
    W = (w // f) * w  # NOTE: fixed typo below (should be W = (w // f) * f)
    W = (w // f) * f  # correct width trim

    a = arr[:H, :W]
    a4 = a.reshape(H // f, f, W // f, f)

    valid = np.sum(~np.isnan(a4), axis=(1, 3)).astype(np.int32)
    block_sum = np.nansum(a4, axis=(1, 3)).astype(np.float32)
    block_sum[valid == 0] = np.nan
    return block_sum


def _per_pixel_tile_path(items: dict, tile_id: str) -> str:
    """
    Resolve per-pixel total input path for a tile.
    This *replaces* any former 'ha' references with 'pixel'.
    """
    pp_dir = items.get("mg_per_pixel_dir")
    pp_pat = items.get("mg_per_pixel_pattern")
    if not pp_dir or not pp_pat:
        raise RuntimeError(
            "Per-pixel total inputs are required but not configured in build_download_upload_dict: "
            "missing mg_per_pixel_dir and/or mg_per_pixel_pattern."
        )
    return f"{pp_dir}{tile_id}{pp_pat}"


# --------------------------------------------------------------------
# Tile aggregation
# --------------------------------------------------------------------

def agg_tile_to_target(
    tile_id: str,
    bounds: Tuple[float, float, float, float],
    chunk_length_pixels: int,
    per_pixel_total_tile: str,
    native_deg: float,
    target_deg: float,
    is_final: bool,
):
    """Aggregate one 10×10° tile from ``native_deg`` to ``target_deg``.

    - Float datasets (per-pixel totals): SUM to the coarser grid.
    - Integer datasets (categorical): MODE to the coarser grid.
    """

    logger = lu.setup_logging()

    logger.info("Reading tile %s\nper-pixel totals: %s", tile_id, per_pixel_total_tile)

    # Load native chunk (Float32 by default)
    arr = uu.get_tile_dataset_rio(
        per_pixel_total_tile, "Float32", bounds, chunk_length_pixels, is_final, logger
    )[0]

    dataset_name = posixpath.basename(per_pixel_total_tile).split("__")[1]
    is_integer = dataset_name in INTEGER_DATASETS

    if is_integer:
        # Categorical aggregation
        arr = arr.astype(np.int32, copy=False)
        logger.info("Aggregating integer dataset by MODE → %s", dataset_name)
        return uu.reaggregate_mode(arr, native_deg, target_deg)

    # Continuous totals → explicit SUM to target resolution (no unit conversions)
    logger.info("Aggregating continuous dataset by SUM → %s", dataset_name)
    return _reaggregate_sum(arr, native_deg, target_deg)


# --------------------------------------------------------------------
# Execution helpers
# --------------------------------------------------------------------

def _compute_tiles(
    delayed_results: List,
    client,
    logger,
    stage_desc: str,
    tile_ids: List[str],
) -> List[np.ndarray]:
    """Compute tile aggregation tasks, streaming results from the Dask cluster."""

    if not delayed_results:
        return []

    if client is None:
        return list(dask.compute(*delayed_results))

    futures = client.compute(delayed_results, sync=False)
    future_to_index = {future: idx for idx, future in enumerate(futures)}
    tiles: List[Optional[np.ndarray]] = [None] * len(futures)
    completed = 0
    total = len(futures)

    for future in as_completed(list(future_to_index.keys())):
        idx = future_to_index[future]
        tile_id = tile_ids[idx]
        try:
            result = future.result()
        except Exception:
            logger.exception("Tile %s failed during %s", tile_id, stage_desc)
            raise

        tiles[idx] = result
        completed += 1
        if completed % 10 == 0 or completed == total:
            logger.info("Completed %d/%d tiles for %s", completed, total, stage_desc)

    missing = [tile_ids[idx] for idx, tile in enumerate(tiles) if tile is None]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} tile results for {stage_desc}: {', '.join(missing)}"
        )

    return [tile for tile in tiles if tile is not None]


def combine_global_raster(
    tiles: List[np.ndarray],
    bounds_list: List[Tuple[float, float, float, float]],
    res_label: str,
    global_outfile: str,
    global_output_path: str,
    target_deg: float,
    is_final: bool,
):
    """Paste aggregated tiles into a single global array at ``target_deg``."""

    logger = lu.setup_logging()

    rows = int(round(180 / target_deg))
    cols = int(round(360 / target_deg))
    global_raster = np.full((rows, cols), np.nan, dtype=np.float32)

    for tile, bounds in zip(tiles, bounds_list):
        min_x, min_y, max_x, max_y = bounds
        x_start = int(round((min_x + 180) / target_deg))
        x_end = int(round((max_x + 180) / target_deg))
        y_start = int(round((90 - max_y) / target_deg))
        y_end = int(round((90 - min_y) / target_deg))

        th, tw = tile.shape
        assert (y_end - y_start) == th
        assert (x_end - x_start) == tw

        np.copyto(
            global_raster[y_start:y_end, x_start:x_end],
            tile,
            where=~np.isnan(tile),
        )

    global_bounds = (-180, -90, 180, 90)

    uu.save_and_upload_single_raster(
        global_bounds,
        global_raster.shape[1],
        f"{res_label}_global",
        global_raster,
        np.float32,
        global_outfile,
        global_output_path,
        is_final,
        logger,
    )
    return "Success"


# --------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------

def aggregate_main(
    cluster_name: str,
    pixel_resolution: str,
    run_name: str = "ogh_sensitivity_1km",
    run_local: bool = False,
    native_deg: float = DEFAULT_NATIVE_DEG,
    target_deg: float = DEFAULT_TARGET_DEG,
    output_date: str = DEFAULT_DATE_TAG,
    model_version: str = DEFAULT_MODEL_VERSION,
    outputs_root: str = OUTPUT_ROOT,
    base_url: Optional[str] = None,
    outputs_base: Optional[str] = None,
) -> None:
    """Drive Stage 01: aggregation to ``target_deg`` and global mosaic creation."""

    assert_grid_divides_world(target_deg)

    logger = lu.setup_logging_main()
    is_final = not run_local

    cluster, client, run_local = uu.connect_to_cluster(
        cluster_name, run_local=run_local
    )
    is_final = not run_local

    resolved_base_url, resolved_outputs_base = resolve_versioned_paths(
        model_version=model_version,
        outputs_root=outputs_root,
        base_url=base_url,
        outputs_base=outputs_base,
    )

    download_upload_dictionary = build_download_upload_dict(
        pixel_resolution=pixel_resolution,
        run_name=run_name,
        target_deg=target_deg,
        base_url=resolved_base_url,
        output_date=output_date,
        outputs_base=resolved_outputs_base,
    )

    res_label = deg_to_label(target_deg)

    for key, items in download_upload_dictionary.items():
        bounds_list: List[Tuple[float, float, float, float]] = []
        delayed_results: List = []
        tile_ids: List[str] = []

        dataset_name = items["dataset"]
        is_integer = dataset_name in INTEGER_DATASETS

        stage = f"aggregate tiles to {res_label} for {key}"
        start_time = uu.timestr()
        lu.print_and_log(f"Stage {stage} started at: {start_time}", is_final, logger)

        for tile_id in cn.tile_id_list:
            # *** IMPORTANT ***: read per-pixel totals; replace 'ha' paths with 'pixel'
            per_pixel_total_tile = _per_pixel_tile_path(items, tile_id)

            bounds = uu.get_10x10_tile_bounds(tile_id)
            bounds_list.append(bounds)
            chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)
            tile_ids.append(tile_id)

            delayed_results.append(
                dask.delayed(agg_tile_to_target)(
                    tile_id,
                    bounds,
                    chunk_length_pixels,
                    per_pixel_total_tile,
                    native_deg,
                    target_deg,
                    is_final,
                )
            )

        stage_desc = f"{res_label} aggregation for {key}"
        tiles = _compute_tiles(
            delayed_results=delayed_results,
            client=None if run_local else client,
            logger=logger,
            stage_desc=stage_desc,
            tile_ids=tile_ids,
        )

        stage = f"build {res_label} global mosaic for {key}"
        start_time = uu.timestr()
        lu.print_and_log(f"Stage {stage} started at: {start_time}", is_final, logger)

        # Always write the canonical global pattern
        global_outfile = items["global_pattern"]
        global_output_path = items["global_dir"]

        combine_global_raster(
            tiles=list(tiles),
            bounds_list=bounds_list,
            res_label=res_label,
            global_outfile=global_outfile,
            global_output_path=global_output_path,
            target_deg=target_deg,
            is_final=is_final,
        )
        lu.print_and_log(
            f"Global raster saved to {global_output_path}{global_outfile}",
            is_final,
            logger,
        )

    if client is not None:
        client.close()
    if cluster is not None:
        try:
            cluster.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate to a target resolution and build global mosaics (SUM for totals, MODE for integers)."
    )
    parser.add_argument("-cn", "--cluster_name", required=True)
    parser.add_argument("--date_tag", required=True)
    parser.add_argument("-p", "--pixel_resolution", default="40000_pixels")
    parser.add_argument("--run_name", default="ogh_sensitivity_1km")
    parser.add_argument("--run_local", action="store_true")
    parser.add_argument("--native_deg", type=float, default=DEFAULT_NATIVE_DEG)
    parser.add_argument("--target_deg", type=float, default=DEFAULT_TARGET_DEG)
    parser.add_argument(
        "--model_version",
        default=DEFAULT_MODEL_VERSION,
        help="Model version string (underscore separated) used to build S3 paths.",
    )
    parser.add_argument(
        "--outputs_root",
        default=OUTPUT_ROOT,
        help="Root S3 directory for model outputs.",
    )
    parser.add_argument(
        "--base_url",
        default=None,
        help="Optional override for the versioned base URL containing per-tile rasters.",
    )
    parser.add_argument(
        "--outputs_base",
        default=None,
        help="Optional override for the destination of aggregated rasters.",
    )

    args = parser.parse_args()

    aggregate_main(
        cluster_name=args.cluster_name,
        pixel_resolution=args.pixel_resolution,
        run_name=args.run_name,
        run_local=args.run_local,
        native_deg=args.native_deg,
        target_deg=args.target_deg,
        output_date=args.date_tag,
        model_version=args.model_version,
        outputs_root=args.outputs_root,
        base_url=args.base_url,
        outputs_base=args.outputs_base,
    )


if __name__ == "__main__":
    main()
