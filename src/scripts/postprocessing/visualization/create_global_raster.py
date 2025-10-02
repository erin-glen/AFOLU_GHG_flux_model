"""
Stage 01: aggregate 10×10° tiles into global rasters at a coarser resolution.

Assumptions (this version)
--------------------------
- All non-integer inputs are **per-pixel totals** (e.g., Mg yr^-1 per native pixel).
- Aggregation for float datasets is **SUM** to the target grid.
- Aggregation for integer datasets is **MODE** (categorical majority).
- For drained_state specifically: we **reclass to binary** (0 undrained, 1 drained; nodata masked)
  *before* aggregation, then take a binary mode and write UInt8 with nodata=255.
- No unit conversions are performed; no input tiles are modified or overwritten.

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
# Constants / helpers
# --------------------------------------------------------------------

# Nodata flag for binary outputs we write as UInt8
UINT8_NODATA = np.uint8(255)

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
    W = (w // f) * f  # correct width trim

    a = arr[:H, :W]
    a4 = a.reshape(H // f, f, W // f, f)

    valid = np.sum(~np.isnan(a4), axis=(1, 3)).astype(np.int32)
    block_sum = np.nansum(a4, axis=(1, 3)).astype(np.float32)
    block_sum[valid == 0] = np.nan
    return block_sum


def _reaggregate_mode_binary(arr01_nan: np.ndarray, native_deg: float, target_deg: float) -> np.ndarray:
    """
    Binary 'mode' via block sums (ties -> 1). `arr01_nan` contains {0,1} and NaN for nodata.
    Returns float32 with {0,1,NaN}.
    """
    factor_f = target_deg / native_deg
    if not np.isclose(round(factor_f), factor_f):
        raise ValueError(f"target_deg/native_deg must be an integer; got {target_deg}/{native_deg}.")
    f = int(round(factor_f))

    h, w = arr01_nan.shape
    H = (h // f) * f
    W = (w // f) * f

    a = arr01_nan[:H, :W]
    a4 = a.reshape(H // f, f, W // f, f)

    valid = np.sum(~np.isnan(a4), axis=(1, 3)).astype(np.float32)
    ones  = np.nansum(a4, axis=(1, 3)).astype(np.float32)           # sum of 1s
    zeros = valid - ones

    out = np.full((H // f, W // f), np.nan, dtype=np.float32)
    has = valid > 0
    # ties resolve to 1 (as requested: "more accurate mode" favoring drained when equal)
    out[has] = (ones[has] >= zeros[has]).astype(np.float32)
    return out


def _reclass_drained_to_binary(arr: np.ndarray) -> np.ndarray:
    """
    Reclass drained_state values to binary with NaN for nodata:
      0 -> nodata
      20,000,000 -> nodata
      16,000,000 -> 0 (UNDRAINED)
      all other values -> 1 (DRAINED)
    Returns float32 array with values {0,1,NaN}.
    """
    a = arr.astype(np.int64, copy=False)

    nodata_mask = (a == 0) | (a == 20_000_000)
    undrained   = (a == 16_000_000)
    drained     = (~nodata_mask) & (~undrained)

    out = np.full(a.shape, np.nan, dtype=np.float32)
    out[undrained] = 0.0
    out[drained]   = 1.0
    # nodata remains NaN
    return out


def _per_pixel_tile_path(items: dict, tile_id: str) -> str:
    """
    Resolve per-pixel total / state input path for a tile.
    Uses the per_pixel_dir/per_pixel_pattern keys provided by build_download_upload_dict.
    """
    pp_dir = items.get("per_pixel_dir")
    pp_pat = items.get("per_pixel_pattern")
    if not pp_dir or not pp_pat:
        raise RuntimeError(
            "Per-pixel/state inputs are required but not configured in build_download_upload_dict: "
            "missing per_pixel_dir and/or per_pixel_pattern."
        )
    return f"{pp_dir}{tile_id}{pp_pat}"


# --------------------------------------------------------------------
# Tile aggregation
# --------------------------------------------------------------------

def agg_tile_to_target(
    tile_id: str,
    bounds: Tuple[float, float, float, float],
    chunk_length_pixels: int,
    per_pixel_total_or_state_tile: str,
    native_deg: float,
    target_deg: float,
    is_final: bool,
):
    """Aggregate one 10×10° tile from ``native_deg`` to ``target_deg``.

    - Float datasets (per-pixel totals): SUM to the coarser grid.
    - burned_state (integer codes): MODE on native codes.
    - drained_state: **reclass to binary (0/1, NaN for nodata) then binary MODE**.
    """

    logger = lu.setup_logging()

    logger.info("Reading tile %s\ninput: %s", tile_id, per_pixel_total_or_state_tile)

    # Load native chunk (Float32 by default)
    arr = uu.get_tile_dataset_rio(
        per_pixel_total_or_state_tile, "Float32", bounds, chunk_length_pixels, is_final, logger
    )[0]

    # Dataset name is the middle token: "<tile>__<dataset>__<interval>.tif"
    parts = posixpath.basename(per_pixel_total_or_state_tile).split("__")
    if len(parts) == 3:
        _, dataset_name, _ = parts
    elif len(parts) >= 4:
        dataset_name = parts[2]
    else:
        raise ValueError(
            "Expected filenames like '<tile>__<dataset>__<interval>.tif' or "
            "'<tile>__<bounds>__<dataset>__<interval>.tif'. "
            f"Got: {posixpath.basename(per_pixel_total_or_state_tile)}"
        )

    # Integer datasets (categorical)
    if dataset_name in INTEGER_DATASETS:
        if dataset_name == "drained_state":
            # Reclass to binary then binary 'mode'
            arr01_nan = _reclass_drained_to_binary(arr)
            return _reaggregate_mode_binary(arr01_nan, native_deg, target_deg)
        else:
            # e.g., burned_state: keep native codes, aggregate by mode
            return uu.reaggregate_mode(arr.astype(np.int32, copy=False), native_deg, target_deg)

    # Continuous totals → explicit SUM to target resolution (no unit conversions)
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
    *,
    out_dtype: Optional[np.dtype] = None,
    int_nodata: Optional[int] = None,
):
    """
    Paste aggregated tiles into a single global array at ``target_deg`` and save/upload.

    If out_dtype is a float type (default), NaN is used for nodata and pasting uses ~np.isnan(tile).
    If out_dtype is an integer type (e.g., np.uint8), int_nodata must be provided; pasting uses (tile != int_nodata).
    """

    logger = lu.setup_logging()

    rows = int(round(180 / target_deg))
    cols = int(round(360 / target_deg))

    if out_dtype is None or np.issubdtype(out_dtype, np.floating):
        # Float output
        global_raster = np.full((rows, cols), np.nan, dtype=np.float32)
        for tile, bounds in zip(tiles, bounds_list):
            min_x, min_y, max_x, max_y = bounds
            x_start = int(round((min_x + 180) / target_deg))
            x_end   = int(round((max_x + 180) / target_deg))
            y_start = int(round((90 - max_y) / target_deg))
            y_end   = int(round((90 - min_y) / target_deg))

            th, tw = tile.shape
            assert (y_end - y_start) == th
            assert (x_end - x_start) == tw

            np.copyto(
                global_raster[y_start:y_end, x_start:x_end],
                tile.astype(np.float32, copy=False),
                where=~np.isnan(tile),
            )

        save_dtype = np.float32
        nodata_tag = "NaN"
    else:
        # Integer output (e.g., UInt8 for binary drained_state)
        if int_nodata is None:
            raise ValueError("int_nodata must be provided when out_dtype is integer.")
        global_raster = np.full((rows, cols), int_nodata, dtype=out_dtype)
        for tile, bounds in zip(tiles, bounds_list):
            # Cast tile: NaN -> int_nodata
            if np.issubdtype(tile.dtype, np.floating):
                t = np.where(np.isnan(tile), int_nodata, tile).astype(out_dtype, copy=False)
            else:
                t = tile.astype(out_dtype, copy=False)

            min_x, min_y, max_x, max_y = bounds
            x_start = int(round((min_x + 180) / target_deg))
            x_end   = int(round((max_x + 180) / target_deg))
            y_start = int(round((90 - max_y) / target_deg))
            y_end   = int(round((90 - min_y) / target_deg))

            th, tw = t.shape
            assert (y_end - y_start) == th
            assert (x_end - x_start) == tw

            np.copyto(
                global_raster[y_start:y_end, x_start:x_end],
                t,
                where=(t != int_nodata),
            )

        save_dtype = out_dtype
        nodata_tag = int_nodata

    global_bounds = (-180, -90, 180, 90)

    uu.save_and_upload_single_raster(
        global_bounds,
        global_raster.shape[1],
        f"{res_label}_global",
        global_raster,
        save_dtype,
        global_outfile,
        global_output_path,
        is_final,
        logger,
    )
    logger.info("Saved global (%s) with nodata=%s → %s%s", str(save_dtype), str(nodata_tag),
                global_output_path, global_outfile)
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
        is_drained = (dataset_name == "drained_state")

        stage = f"aggregate tiles to {res_label} for {key}"
        start_time = uu.timestr()
        lu.print_and_log(f"Stage {stage} started at: {start_time}", is_final, logger)

        for tile_id in cn.tile_id_list:
            # Read per-pixel/state inputs (never overwritten; read-only)
            tile_path = _per_pixel_tile_path(items, tile_id)

            bounds = uu.get_10x10_tile_bounds(tile_id)
            bounds_list.append(bounds)
            chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)
            tile_ids.append(tile_id)

            delayed_results.append(
                dask.delayed(agg_tile_to_target)(
                    tile_id,
                    bounds,
                    chunk_length_pixels,
                    tile_path,
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

        # Choose output dtype / nodata for the global raster
        if is_drained:
            # Binary UInt8 with nodata=255
            _ = combine_global_raster(
                tiles=list(tiles),
                bounds_list=bounds_list,
                res_label=res_label,
                global_outfile=global_outfile,
                global_output_path=global_output_path,
                target_deg=target_deg,
                is_final=is_final,
                out_dtype=np.uint8,
                int_nodata=int(UINT8_NODATA),
            )
        else:
            _ = combine_global_raster(
                tiles=list(tiles),
                bounds_list=bounds_list,
                res_label=res_label,
                global_outfile=global_outfile,
                global_output_path=global_output_path,
                target_deg=target_deg,
                is_final=is_final,
                out_dtype=None,  # float32 with NaN nodata (default)
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
        description="Aggregate to a target resolution and build global mosaics (SUM for totals, MODE for integers; drained_state → binary)."
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
