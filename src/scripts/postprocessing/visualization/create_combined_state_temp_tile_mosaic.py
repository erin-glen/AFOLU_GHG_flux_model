"""Build a global combined_state raster via worker-written aggregated tiles.

This is a robust companion for ``create_global_raster`` when client-side
gathering of 0.5 km combined_state tile arrays is fragile. Workers aggregate
native 10x10 degree tiles to the requested target resolution, upload those
smaller aggregated tiles to a temporary S3 prefix, and return only S3 URIs to
the driver. The driver then streams the temporary tiles into the global mosaic.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import tempfile
from typing import Iterable
from urllib.parse import urlparse

import boto3
import numpy as np
import rasterio
from dask import delayed
from dask.distributed import as_completed
from rasterio.transform import from_bounds

from src.scripts.postprocessing.visualization import create_global_raster as cgr
from src.scripts.postprocessing.visualization.create_global_map_common import (
    DEFAULT_MODEL_VERSION,
    DEFAULT_NATIVE_DEG,
    INTEGER_DATASETS,
    OUTPUT_ROOT,
    build_download_upload_dict,
    deg_to_label,
    resolve_versioned_paths,
)
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import log_utilities as lu
from src.scripts.utilities import universal_utilities as uu


def _s3_parts(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3 URI, got {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _upload_file(local_path: str, s3_uri: str) -> None:
    bucket, key = _s3_parts(s3_uri)
    boto3.client("s3").upload_file(local_path, bucket, key)


def aggregate_combined_state_tile_to_s3(
    tile_id: str,
    src_uri: str,
    dst_uri: str,
    native_deg: float,
    target_deg: float,
) -> str:
    """Aggregate one combined_state tile and upload the smaller result to S3."""
    logger = lu.setup_logging()
    bounds = uu.get_10x10_tile_bounds(tile_id)
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)
    arr = cgr._agg_tile_to_target_windowed(
        tile_id=tile_id,
        chunk_length_pixels=chunk_length_pixels,
        per_pixel_total_or_state_tile=src_uri,
        native_deg=native_deg,
        target_deg=target_deg,
        is_final=True,
        logger=logger,
    ).astype(np.uint32, copy=False)

    transform = from_bounds(*bounds, arr.shape[1], arr.shape[0])
    profile = {
        "driver": "GTiff",
        "width": arr.shape[1],
        "height": arr.shape[0],
        "count": 1,
        "dtype": "uint32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": 0,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "DEFLATE",
        "predictor": 1,
        "bigtiff": "IF_SAFER",
        "num_threads": "ALL_CPUS",
    }

    fd, local_path = tempfile.mkstemp(prefix=f"{tile_id}_combined_state_0p5_", suffix=".tif")
    os.close(fd)
    try:
        with rasterio.open(local_path, "w", **profile) as dst:
            dst.write(arr, 1)
        _upload_file(local_path, dst_uri)
    finally:
        try:
            os.remove(local_path)
        except OSError:
            pass
    return dst_uri


def aggregate_combined_state_fraction_tile_to_s3(
    tile_id: str,
    src_uri: str,
    dst_uri: str,
    native_deg: float,
    target_deg: float,
) -> str:
    """Aggregate one combined_state tile to class-fraction bands and upload."""
    logger = lu.setup_logging()
    bounds = uu.get_10x10_tile_bounds(tile_id)
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)
    arr = cgr._agg_combined_state_class_fractions_windowed(
        tile_id=tile_id,
        chunk_length_pixels=chunk_length_pixels,
        combined_state_tile=src_uri,
        native_deg=native_deg,
        target_deg=target_deg,
        is_final=True,
        logger=logger,
    ).astype(np.float32, copy=False)

    transform = from_bounds(*bounds, arr.shape[2], arr.shape[1])
    profile = {
        "driver": "GTiff",
        "width": arr.shape[2],
        "height": arr.shape[1],
        "count": arr.shape[0],
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": None,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "DEFLATE",
        "predictor": 2,
        "bigtiff": "IF_SAFER",
        "num_threads": "ALL_CPUS",
    }

    fd, local_path = tempfile.mkstemp(prefix=f"{tile_id}_combined_state_fraction_0p5_", suffix=".tif")
    os.close(fd)
    try:
        with rasterio.open(local_path, "w", **profile) as dst:
            for band_idx in range(arr.shape[0]):
                dst.write(arr[band_idx], band_idx + 1)
                if band_idx < len(cgr.COMBINED_STATE_CLASS_BAND_DESCRIPTIONS):
                    dst.set_band_description(
                        band_idx + 1,
                        cgr.COMBINED_STATE_CLASS_BAND_DESCRIPTIONS[band_idx],
                    )
            dst.update_tags(**cgr.COMBINED_STATE_CLASS_FRACTION_TAGS)
        _upload_file(local_path, dst_uri)
    finally:
        try:
            os.remove(local_path)
        except OSError:
            pass
    return dst_uri


def _paste_tile(memmap: np.memmap, tile_uri: str, bounds: tuple[float, float, float, float], target_deg: float) -> None:
    with rasterio.open(cgr._to_vsipath_if_s3(tile_uri)) as src:
        arr = src.read(1, out_dtype="uint32")
    min_x, min_y, max_x, max_y = bounds
    x0 = int(round((min_x + 180) / target_deg))
    x1 = int(round((max_x + 180) / target_deg))
    y0 = int(round((90 - max_y) / target_deg))
    y1 = int(round((90 - min_y) / target_deg))
    memmap[y0:y1, x0:x1] = arr


def _paste_fraction_tile(memmap: np.memmap, tile_uri: str, bounds: tuple[float, float, float, float], target_deg: float) -> None:
    with rasterio.open(cgr._to_vsipath_if_s3(tile_uri)) as src:
        arr = src.read(out_dtype="float32")
    min_x, min_y, max_x, max_y = bounds
    x0 = int(round((min_x + 180) / target_deg))
    x1 = int(round((max_x + 180) / target_deg))
    y0 = int(round((90 - max_y) / target_deg))
    y1 = int(round((90 - min_y) / target_deg))
    memmap[:, y0:y1, x0:x1] = arr


def _iter_tile_uris(prefix: str, tile_ids: Iterable[str]) -> dict[str, str]:
    return {
        tile_id: f"{prefix.rstrip('/')}/{tile_id}__combined_state__0_005deg_tmp.tif"
        for tile_id in tile_ids
    }


def _iter_fraction_tile_uris(prefix: str, tile_ids: Iterable[str]) -> dict[str, str]:
    return {
        tile_id: f"{prefix.rstrip('/')}/{tile_id}__combined_state_class_fraction__0_005deg_tmp.tif"
        for tile_id in tile_ids
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 0.5 km combined_state via worker-written temp tiles.")
    parser.add_argument("-cn", "--cluster_name", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--date_tag", required=True)
    parser.add_argument("--model_version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--native_deg", type=float, default=DEFAULT_NATIVE_DEG)
    parser.add_argument("--target_deg", type=float, required=True)
    parser.add_argument("--inventory_period", default="2021_2024")
    parser.add_argument("--outputs_root", default=OUTPUT_ROOT)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--outputs_base", default=None)
    parser.add_argument("--temp_tile_prefix", required=True)
    parser.add_argument("--dask_batch", type=int, default=16)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    logger = lu.setup_logging_main()
    res_label = deg_to_label(args.target_deg)

    cluster, client, run_local = uu.connect_to_cluster(args.cluster_name, run_local=False)
    if run_local or client is None:
        raise RuntimeError(f"Could not attach to Coiled cluster {args.cluster_name}")

    resolved_base_url, resolved_outputs_base = resolve_versioned_paths(
        model_version=args.model_version,
        outputs_root=args.outputs_root,
        base_url=args.base_url,
        outputs_base=args.outputs_base,
    )
    download_upload = build_download_upload_dict(
        pixel_resolution="40000_pixels",
        run_name=args.run_name,
        target_deg=args.target_deg,
        base_url=resolved_base_url,
        output_date=args.date_tag,
        outputs_base=resolved_outputs_base,
        data_types=["combined_state"],
        inventory_periods=[args.inventory_period],
    )
    key = f"combined_state__{args.inventory_period}"
    items = download_upload[key]
    if items["dataset"] not in INTEGER_DATASETS:
        raise RuntimeError("Expected combined_state dataset.")

    tile_ids = list(cn.tile_id_list)
    temp_tile_uris = _iter_tile_uris(args.temp_tile_prefix, tile_ids)
    temp_fraction_tile_uris = _iter_fraction_tile_uris(args.temp_tile_prefix, tile_ids)

    lu.print_and_log(
        f"Stage aggregate combined_state temp tiles to {res_label} started at: {uu.timestr()}",
        True,
        logger,
    )
    completed = 0
    for start in range(0, len(tile_ids), args.dask_batch):
        batch_ids = tile_ids[start:start + args.dask_batch]
        tasks = [
            delayed(aggregate_combined_state_tile_to_s3)(
                tile_id,
                cgr._per_pixel_tile_path(items, tile_id),
                temp_tile_uris[tile_id],
                args.native_deg,
                args.target_deg,
            )
            for tile_id in batch_ids
        ]
        futures = client.compute(tasks, sync=False, retries=args.retries)
        future_to_tile = {future: tile_id for future, tile_id in zip(futures, batch_ids)}
        for future in as_completed(list(future_to_tile.keys())):
            tile_id = future_to_tile[future]
            try:
                future.result()
            except Exception:
                logger.exception("Temp tile aggregation failed for %s", tile_id)
                raise
            completed += 1
            if completed % 10 == 0 or completed == len(tile_ids):
                lu.print_and_log(
                    f"Completed {completed}/{len(tile_ids)} temp combined_state tiles",
                    True,
                    logger,
                )

    lu.print_and_log(
        f"Stage aggregate combined_state class-fraction temp tiles to {res_label} started at: {uu.timestr()}",
        True,
        logger,
    )
    completed = 0
    for start in range(0, len(tile_ids), args.dask_batch):
        batch_ids = tile_ids[start:start + args.dask_batch]
        tasks = [
            delayed(aggregate_combined_state_fraction_tile_to_s3)(
                tile_id,
                cgr._per_pixel_tile_path(items, tile_id),
                temp_fraction_tile_uris[tile_id],
                args.native_deg,
                args.target_deg,
            )
            for tile_id in batch_ids
        ]
        futures = client.compute(tasks, sync=False, retries=args.retries)
        future_to_tile = {future: tile_id for future, tile_id in zip(futures, batch_ids)}
        for future in as_completed(list(future_to_tile.keys())):
            tile_id = future_to_tile[future]
            try:
                future.result()
            except Exception:
                logger.exception("Temp class-fraction tile aggregation failed for %s", tile_id)
                raise
            completed += 1
            if completed % 10 == 0 or completed == len(tile_ids):
                lu.print_and_log(
                    f"Completed {completed}/{len(tile_ids)} temp class-fraction tiles",
                    True,
                    logger,
                )

    rows = int(round(180 / args.target_deg))
    cols = int(round(360 / args.target_deg))
    tmp_parent = os.environ.get("AGG_GLOBAL_TMPDIR")
    if tmp_parent:
        os.makedirs(tmp_parent, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix=f"{res_label}_combined_state_global_", dir=tmp_parent)
    mm_path = os.path.join(tmpdir, "combined_state_global_mm.dat")
    global_mm = np.memmap(mm_path, dtype=np.uint32, mode="w+", shape=(rows, cols))

    lu.print_and_log(
        f"Stage build {res_label} global combined_state mosaic started at: {uu.timestr()}",
        True,
        logger,
    )
    for idx, tile_id in enumerate(tile_ids, start=1):
        _paste_tile(global_mm, temp_tile_uris[tile_id], uu.get_10x10_tile_bounds(tile_id), args.target_deg)
        if idx % 25 == 0 or idx == len(tile_ids):
            global_mm.flush()
            lu.print_and_log(f"Pasted {idx}/{len(tile_ids)} temp tiles", True, logger)

    global_dir = items["global_dir"]
    global_name = items["global_pattern"]
    cgr._save_global_raster_correct(
        bounds=(-180, -90, 180, 90),
        arr=global_mm,
        dtype=np.uint32,
        dst_base=global_dir,
        dst_name=global_name,
        logger=logger,
        int_nodata=0,
        spool_dir=tmpdir,
    )

    reclass_dir, reclass_name = cgr._combined_state_reclass_output(global_dir, global_name)
    reclass_path = os.path.join(tmpdir, "combined_state_reclass_mm.dat")
    reclass_mm = np.memmap(reclass_path, dtype=np.uint8, mode="w+", shape=(rows, cols))
    cgr._fill_reclassified_memmap(global_mm, reclass_mm)
    cgr._save_global_raster_correct(
        bounds=(-180, -90, 180, 90),
        arr=reclass_mm,
        dtype=np.uint8,
        dst_base=reclass_dir,
        dst_name=reclass_name,
        logger=logger,
        int_nodata=int(cgr.COMBINED_STATE_RECLASS_NODATA),
        spool_dir=tmpdir,
        tags=cgr.COMBINED_STATE_RECLASS_TAGS,
    )

    fraction_path = os.path.join(tmpdir, "combined_state_class_fraction_mm.dat")
    fraction_mm = np.memmap(
        fraction_path,
        dtype=np.float32,
        mode="w+",
        shape=(len(cgr.COMBINED_STATE_CLASS_BAND_DESCRIPTIONS), rows, cols),
    )
    fraction_mm[:] = 0.0

    lu.print_and_log(
        f"Stage build {res_label} global combined_state class-fraction mosaic started at: {uu.timestr()}",
        True,
        logger,
    )
    for idx, tile_id in enumerate(tile_ids, start=1):
        _paste_fraction_tile(
            fraction_mm,
            temp_fraction_tile_uris[tile_id],
            uu.get_10x10_tile_bounds(tile_id),
            args.target_deg,
        )
        if idx % 25 == 0 or idx == len(tile_ids):
            fraction_mm.flush()
            lu.print_and_log(f"Pasted {idx}/{len(tile_ids)} class-fraction temp tiles", True, logger)

    cgr._write_combined_state_fraction_outputs(
        class_fraction=fraction_mm,
        global_output_path=global_dir,
        global_outfile=global_name,
        logger=logger,
        spool_dir=tmpdir,
    )

    del fraction_mm
    del reclass_mm
    del global_mm
    try:
        os.remove(fraction_path)
        os.remove(reclass_path)
        os.remove(mm_path)
        os.rmdir(tmpdir)
    except OSError:
        pass

    client.close()
    try:
        cluster.close()
    except Exception:
        pass

    lu.print_and_log("Combined_state temp-tile global mosaic complete.", True, logger)


if __name__ == "__main__":
    main()
