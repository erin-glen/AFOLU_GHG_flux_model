"""
Stage 01: aggregate 10×10° tiles into global rasters at a coarser resolution.

What's new in this drop-in update
---------------------------------
- Streamed global mosaic build using a disk-backed memmap (no N×10 GB RAM spike).
- As-completed iteration over Dask futures; no giant gather into a Python list.
- Early cast for `drained_state` tiles to UInt8 (0,1,255 nodata) to cut tile payloads 4×.
- **FIX 1 (GeoTransform)**: Use rasterio.transform.from_bounds (rows & cols) so pixel size Y == X.
- **FIX 2 (S3 writing)**: When writing to s3:// paths, enable GDAL spooling:
    CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE=YES and CPL_TMPDIR=<our working tmpdir>.
- **FIX 3 (OOM on drained_state)**: Memory-tight reclassification (no float64/int64 upcasts; one
    boolean mask at a time; range checks based on 8-digit padding).
- **Typed reads**: Choose "Int32" for integer datasets (burned_state, drained_state), "Float32" otherwise.
- **Local batch knob**: AGG_LOCAL_BATCH env var controls local batch size (default 8) for iterate_tiles.

Assumptions (this version)
--------------------------
- All non-integer inputs are **per-pixel totals** (e.g., Mg yr^-1 per native pixel).
- Aggregation for float datasets is **SUM** to the target grid.
- Aggregation for integer datasets is **MODE** (categorical majority).
- For drained_state specifically: we **reclass to binary** (0 undrained, 1 drained; non-peat masked)
  *before* aggregation, then take a binary mode and write UInt8 with nodata=255.
- No unit conversions are performed; no input tiles are modified or overwritten.

Examples
--------
# Aggregate to 0.01° on a running Dask cluster
python -m src.scripts.postprocessing.visualization.create_global_raster \
  -cn create_maps --run_name ogh_sensitivity_500m_23 \
  --model_version 0_9_7 --date_tag 20251118 --target_deg 0.01 --native_deg 0.00025

# Aggregate at 0.01° using a local Dask scheduler (smaller local batch by default)
AGG_LOCAL_BATCH=8 \
python -m src.scripts.postprocessing.visualization.create_global_raster \
  -cn local --run_name ogh_sensitivity_500m --run_local \
  --model_version 0_9_5 --date_tag 20251117 --target_deg 0.01 --native_deg 0.00025
"""

from __future__ import annotations

import argparse
import os
import posixpath
import tempfile
from typing import Iterator, List, Optional, Tuple

import dask
from dask.distributed import as_completed
import numpy as np

# Write with a correct transform derived from rows & cols
import rasterio
from rasterio.transform import from_bounds

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import log_utilities as lu
from src.scripts.utilities import universal_utilities as uu
from src.scripts.zonal_statistics import zonal_constants as zc

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
STATE_PAD_DIGITS = len(next(iter(zc.ALL_DRAINED_STATE_CODES)))
UNDRAINED_ROOT_CODE = int("16".ljust(STATE_PAD_DIGITS, "0"))


def _reaggregate_sum(arr: np.ndarray, native_deg: float, target_deg: float) -> np.ndarray:
    """
    Downsample by summing factor×factor blocks (preserves NaNs).
    If a block has 0 valid cells, output is NaN.
    """
    factor_f = target_deg / native_deg
    if not np.isclose(round(factor_f), factor_f):
        raise ValueError(f"target_deg/native_deg must be an integer; got {target_deg}/{native_deg}.")
    f = int(round(factor_f))

    h, w = arr.shape
    H = (h // f) * f
    W = (w // f) * f

    a = arr[:H, :W]
    a4 = a.reshape(H // f, f, W // f, f)

    valid = np.sum(~np.isnan(a4), axis=(1, 3)).astype(np.int32)
    block_sum = np.nansum(a4, axis=(1, 3)).astype(np.float32)
    block_sum[valid == 0] = np.nan
    return block_sum


def _reaggregate_mode_binary(arr01_nan: np.ndarray, native_deg: float, target_deg: float) -> np.ndarray:
    """
    Binary 'mode' via block sums (ties -> 1). `arr01_nan` contains {0,1} and NaN for masked.
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
    ones  = np.nansum(a4, axis=(1, 3)).astype(np.float32)  # sum of 1s
    zeros = valid - ones

    out = np.full((H // f, W // f), np.nan, dtype=np.float32)
    has = valid > 0
    # ties resolve to 1 (favor drained when equal)
    out[has] = (ones[has] >= zeros[has]).astype(np.float32)
    return out


def _reclass_drained_to_binary(arr: np.ndarray) -> np.ndarray:
    """
    Reclassify numeric `drained_state` to binary:
      - undrained peat root (16) -> 0.0
      - drained peat roots (11..15) -> 1.0
      - everything else (incl. non-peat 0) -> NaN (masked)

    Robust to 6-, 8-, or 10-digit padded states by detecting observed width.
    """
    a = np.asarray(arr, dtype=np.float32)

    # All zero => non-peat everywhere -> mask (NaN)
    if a.size == 0:
        return np.full(a.shape, np.nan, dtype=np.float32)
    m = float(np.nanmax(a))
    if not np.isfinite(m) or m <= 0.0:
        return np.full(a.shape, np.nan, dtype=np.float32)

    # Derive the effective pad width from the largest code present
    # e.g., 160000 (6-digit) -> width=6, 16000000 (8-digit) -> width=8
    observed_width = max(int(np.floor(np.log10(m))) + 1, 2)
    div = float(10 ** (observed_width - 2))

    und_lo, und_hi = 16.0 * div, 17.0 * div
    drn_lo, drn_hi = 11.0 * div, 16.0 * div

    out = np.full(a.shape, np.nan, dtype=np.float32)

    # One boolean at a time to limit peak memory
    und = (a >= und_lo) & (a < und_hi)
    out[und] = 0.0
    del und

    drn = (a >= drn_lo) & (a < drn_hi)
    out[drn] = 1.0
    del drn

    return out



def _per_pixel_tile_path(items: dict, tile_id: str) -> str:
    pp_dir = items.get("per_pixel_dir")
    pp_pat = items.get("per_pixel_pattern")
    if not pp_dir or not pp_pat:
        raise RuntimeError(
            "Per-pixel/state inputs are required but not configured in build_download_upload_dict: "
            "missing per_pixel_dir and/or per_pixel_pattern."
        )
    return f"{pp_dir}{tile_id}{pp_pat}"


def _join_output_path(base: str, name: str) -> str:
    return f"{base.rstrip('/')}/{name}"


def _to_vsipath_if_s3(path: str) -> str:
    return "/vsis3/" + path[len("s3://") :] if path.startswith("s3://") else path


def _save_global_raster_correct(
    *,
    bounds: Tuple[float, float, float, float],
    arr: np.ndarray,
    dtype: np.dtype,
    dst_base: str,
    dst_name: str,
    logger,
    int_nodata: Optional[int] = None,
    spool_dir: Optional[str] = None,
) -> str:
    """
    Write a single-band GeoTIFF with a correct geotransform derived from rows & cols.
    If destination is s3://..., enable GDAL spooling so random writes work.
    Returns the final destination path.
    """
    minx, miny, maxx, maxy = bounds
    rows, cols = arr.shape
    transform = from_bounds(minx, miny, maxx, maxy, cols, rows)

    dst_path = _join_output_path(dst_base, dst_name)
    vsi_path = _to_vsipath_if_s3(dst_path)

    is_float = np.issubdtype(dtype, np.floating)
    creation_opts = dict(
        driver="GTiff",
        bigtiff="YES",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="DEFLATE",
        predictor=2 if is_float else 1,
        num_threads="ALL_CPUS",
    )

    profile = dict(
        width=cols,
        height=rows,
        count=1,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=None if is_float else int_nodata,
        **creation_opts,
    )

    # If writing to S3, enable GDAL's temp-file spooling for random writes.
    env_kwargs = {}
    if dst_path.startswith("s3://"):
        env_kwargs["CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE"] = "YES"
        if spool_dir:
            env_kwargs["CPL_TMPDIR"] = spool_dir  # ensure enough space on this disk

    logger.info("Writing %s (size %d×%d, dtype=%s) ...",
                dst_path, cols, rows, np.dtype(dtype).name)
    with rasterio.Env(**env_kwargs):
        with rasterio.open(vsi_path, "w", **profile) as dst:
            dst.write(arr, 1)

    logger.info("Saved global (%s) → %s", np.dtype(dtype).name, dst_path)
    return dst_path


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
    """
    Aggregate one 10×10° tile from native_deg to target_deg.
    """
    logger = lu.setup_logging()
    logger.info("Reading tile %s\ninput: %s", tile_id, per_pixel_total_or_state_tile)

    # Determine dataset name up front (so we can choose the read dtype)
    parts = posixpath.basename(per_pixel_total_or_state_tile).split("__")
    if len(parts) == 3:
        _, dataset_name, _ = parts
    elif len(parts) >= 4:
        dataset_name = parts[2]
    else:
        raise ValueError(
            "Expected '<tile>__<dataset>__<interval>.tif' or "
            "'<tile>__<bounds>__<dataset>__<interval>.tif'; "
            f"got: {posixpath.basename(per_pixel_total_or_state_tile)}"
        )

    # Integer datasets are read as Int32; floats as Float32
    dtype_hint = "Int32" if dataset_name in INTEGER_DATASETS else "Float32"

    arr, success = uu.get_tile_dataset_rio(
        per_pixel_total_or_state_tile, dtype_hint, bounds, chunk_length_pixels, is_final, logger
    )

    if dataset_name == "drained_state" and not success:
        logger.warning(
            "Tile %s missing drained_state; treating as non-peat (masked after reclass).",
            tile_id,
        )
        # zeros will not match the [11..16) ranges and thus become NaN in reclass
        arr = np.zeros((chunk_length_pixels, chunk_length_pixels), dtype=np.int32)

    if dataset_name in INTEGER_DATASETS:
        if dataset_name == "drained_state":
            # Reclass to {0,1,NaN} in float32 with minimal temporaries, then binary mode
            arr01_nan = _reclass_drained_to_binary(arr)
            out = _reaggregate_mode_binary(arr01_nan, native_deg, target_deg)  # float32 {0,1,NaN}
            return np.where(np.isnan(out), UINT8_NODATA, out.astype(np.uint8, copy=False))
        else:
            # e.g., burned_state: keep native codes, aggregate by mode
            return uu.reaggregate_mode(
                arr.astype(np.int32, copy=False), native_deg, target_deg
            )

    # Continuous totals → explicit SUM to target resolution (no unit conversions)
    return _reaggregate_sum(arr.astype(np.float32, copy=False), native_deg, target_deg)


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
    """Legacy all-at-once compute; kept for backward compatibility."""
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


def iterate_tiles(
    delayed_results: List,
    client,
    logger,
    stage_desc: str,
    tile_ids: List[str],
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (tile_index, tile_array) as each tile finishes computing."""
    if not delayed_results:
        return
    if client is None:
        # Smaller, configurable local batch to avoid long stalls
        B = int(os.environ.get("AGG_LOCAL_BATCH", "8"))
        for i in range(0, len(delayed_results), B):
            batch = delayed_results[i:i + B]
            results = dask.compute(*batch)
            for j, res in enumerate(results):
                yield (i + j, res)
    else:
        futures = client.compute(delayed_results, sync=False)
        future_to_index = {f: i for i, f in enumerate(futures)}
        completed = 0
        total = len(futures)
        for future in as_completed(list(future_to_index.keys())):
            idx = future_to_index[future]
            try:
                arr = future.result()
            except Exception:
                logger.exception("Tile %s failed during %s", tile_ids[idx], stage_desc)
                raise
            completed += 1
            if completed % 10 == 0 or completed == total:
                logger.info("Completed %d/%d tiles for %s", completed, total, stage_desc)
            yield (idx, arr)


# --------------------------------------------------------------------
# Combine & write
# --------------------------------------------------------------------

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
    """Legacy in-RAM combine; now writes with correct transform and S3-safe spooling."""
    logger = lu.setup_logging()
    rows = int(round(180 / target_deg))
    cols = int(round(360 / target_deg))

    if out_dtype is None or np.issubdtype(out_dtype, np.floating):
        global_raster = np.full((rows, cols), np.nan, dtype=np.float32)
        for tile, bounds in zip(tiles, bounds_list):
            min_x, min_y, max_x, max_y = bounds
            x0 = int(round((min_x + 180) / target_deg))
            x1 = int(round((max_x + 180) / target_deg))
            y0 = int(round((90 - max_y) / target_deg))
            y1 = int(round((90 - min_y) / target_deg))
            t = tile.astype(np.float32, copy=False)
            mask = ~np.isnan(t) if np.issubdtype(t.dtype, np.floating) else np.ones_like(t, dtype=bool)
            np.copyto(global_raster[y0:y1, x0:x1], t, where=mask)
        save_dtype = np.float32
        int_nodata_eff = None
    else:
        if int_nodata is None:
            raise ValueError("int_nodata must be provided when out_dtype is integer.")
        global_raster = np.full((rows, cols), int_nodata, dtype=out_dtype)
        for tile, bounds in zip(tiles, bounds_list):
            if np.issubdtype(tile.dtype, np.floating):
                t = np.where(np.isnan(tile), int_nodata, tile).astype(out_dtype, copy=False)
            else:
                t = tile.astype(out_dtype, copy=False)
            min_x, min_y, max_x, max_y = bounds
            x0 = int(round((min_x + 180) / target_deg))
            x1 = int(round((max_x + 180) / target_deg))
            y0 = int(round((90 - max_y) / target_deg))
            y1 = int(round((90 - min_y) / target_deg))
            np.copyto(global_raster[y0:y1, x0:x1], t, where=(t != int_nodata))
        save_dtype = out_dtype
        int_nodata_eff = int_nodata

    _ = _save_global_raster_correct(
        bounds=(-180, -90, 180, 90),
        arr=global_raster,
        dtype=save_dtype,
        dst_base=global_output_path,
        dst_name=global_outfile,
        logger=logger,
        int_nodata=int_nodata_eff,
        spool_dir=tempfile.gettempdir(),  # safe default
    )
    return "Success"


def combine_global_raster_streaming(
    tiles_iter: "Iterator[Tuple[int, np.ndarray]]",
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
    Stream tiles into a global memmap on disk; then write with correct transform.
    When writing to s3://, enable GDAL spooling and use the same tmpdir for spooling.
    """
    logger = lu.setup_logging()

    rows = int(round(180 / target_deg))
    cols = int(round(360 / target_deg))

    # Use a dedicated working directory; we also point GDAL spooling here.
    tmpdir = tempfile.mkdtemp(prefix=f"{res_label}_global_")
    mm_path = os.path.join(tmpdir, "global_mm.dat")

    if out_dtype is None or np.issubdtype(out_dtype, np.floating):
        save_dtype = np.float32
        global_mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=(rows, cols))
        global_mm[:] = np.nan
        def paste(tile, y0, y1, x0, x1):
            t = tile.astype(np.float32, copy=False)
            mask = ~np.isnan(t) if np.issubdtype(t.dtype, np.floating) else np.ones_like(t, dtype=bool)
            np.copyto(global_mm[y0:y1, x0:x1], t, where=mask)
        int_nodata_eff = None
    else:
        if int_nodata is None:
            raise ValueError("int_nodata must be provided when out_dtype is integer.")
        save_dtype = out_dtype
        global_mm = np.memmap(mm_path, dtype=out_dtype, mode="w+", shape=(rows, cols))
        global_mm[:] = int_nodata
        def paste(tile, y0, y1, x0, x1):
            if np.issubdtype(tile.dtype, np.floating):
                t = np.where(np.isnan(tile), int_nodata, tile).astype(out_dtype, copy=False)
            else:
                t = tile.astype(out_dtype, copy=False)
            np.copyto(global_mm[y0:y1, x0:x1], t, where=(t != int_nodata))
        int_nodata_eff = int_nodata

    flush_every = 16
    seen = 0
    for idx, tile in tiles_iter:
        min_x, min_y, max_x, max_y = bounds_list[idx]
        x0 = int(round((min_x + 180) / target_deg))
        x1 = int(round((max_x + 180) / target_deg))
        y0 = int(round((90 - max_y) / target_deg))
        y1 = int(round((90 - min_y) / target_deg))
        paste(tile, y0, y1, x0, x1)
        seen += 1
        if seen % flush_every == 0:
            global_mm.flush()

    del global_mm
    global_mm = np.memmap(mm_path, dtype=save_dtype, mode="r", shape=(rows, cols))

    _ = _save_global_raster_correct(
        bounds=(-180, -90, 180, 90),
        arr=global_mm,
        dtype=save_dtype,
        dst_base=global_output_path,
        dst_name=global_outfile,
        logger=logger,
        int_nodata=int_nodata_eff,
        spool_dir=tmpdir,  # <-- S3 spooling will use this same directory
    )

    # Cleanup
    try:
        del global_mm
        os.remove(mm_path)
        os.rmdir(tmpdir)
    except Exception:
        pass

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
    assert_grid_divides_world(target_deg)

    logger = lu.setup_logging_main()
    is_final = not run_local

    cluster, client, run_local = uu.connect_to_cluster(cluster_name, run_local=run_local)
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
        is_drained = (dataset_name == "drained_state")

        stage = f"aggregate tiles to {res_label} for {key}"
        lu.print_and_log(f"Stage {stage} started at: {uu.timestr()}", is_final, logger)

        for tile_id in cn.tile_id_list:
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
        tiles_iter = iterate_tiles(
            delayed_results=delayed_results,
            client=None if run_local else client,
            logger=logger,
            stage_desc=stage_desc,
            tile_ids=tile_ids,
        )

        stage = f"build {res_label} global mosaic for {key}"
        lu.print_and_log(f"Stage {stage} started at: {uu.timestr()}", is_final, logger)

        global_outfile = items["global_pattern"]
        global_output_path = items["global_dir"]

        if is_drained:
            _ = combine_global_raster_streaming(
                tiles_iter=tiles_iter,
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
            _ = combine_global_raster_streaming(
                tiles_iter=tiles_iter,
                bounds_list=bounds_list,
                res_label=res_label,
                global_outfile=global_outfile,
                global_output_path=global_output_path,
                target_deg=target_deg,
                is_final=is_final,
                out_dtype=None,
            )

        lu.print_and_log(
            f"Global raster saved to {_join_output_path(global_output_path, global_outfile)}",
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