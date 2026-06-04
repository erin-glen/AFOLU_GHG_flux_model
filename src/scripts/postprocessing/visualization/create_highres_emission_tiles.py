"""Aggregate selected emission tiles to high-resolution per-tile rasters.

This is intended for portrait/map windows where a full global raster would be
too large. It aggregates native 10x10 degree per-pixel emission tiles to a
target resolution and writes one output GeoTIFF per tile/dataset/period.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import tempfile
from typing import Iterable, Optional
from urllib.parse import urlparse

import boto3
import dask
import numpy as np
import rasterio
from dask.distributed import as_completed
from rasterio.transform import from_bounds

from src.scripts.postprocessing.visualization import create_global_raster as cgr
from src.scripts.postprocessing.visualization.create_global_map_common import (
    DEFAULT_DATE_TAG,
    DEFAULT_MODEL_VERSION,
    DEFAULT_NATIVE_DEG,
    OUTPUT_ROOT,
    assert_grid_divides_world,
    build_download_upload_dict,
    deg_to_label,
    resolve_versioned_paths,
)
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import log_utilities as lu
from src.scripts.utilities import universal_utilities as uu


EMISSION_DATASETS = (
    "drained_total_Mg_CO2e_pixel_yr",
    "burned_total_Mg_CO2e_pixel_yr",
)


def _split_cli_items(values: Optional[Iterable[str]]) -> Optional[list[str]]:
    if values is None:
        return None
    items: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                items.append(part)
    return items


def _s3_parts(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3 URI, got {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _s3_exists(uri: str) -> bool:
    bucket, key = _s3_parts(uri)
    try:
        boto3.client("s3").head_object(Bucket=bucket, Key=key)
    except Exception:
        return False
    return True


def _upload_file(local_path: str, s3_uri: str) -> None:
    bucket, key = _s3_parts(s3_uri)
    boto3.client("s3").upload_file(local_path, bucket, key)


def _validate_data_types(data_types: Iterable[str]) -> list[str]:
    requested = list(dict.fromkeys(data_types))
    invalid = [data_type for data_type in requested if data_type not in EMISSION_DATASETS]
    if invalid:
        raise ValueError(
            "This script only supports drained and burned emission totals; "
            f"invalid data types: {invalid}"
        )
    return requested


def _validate_tile_ids(tile_ids: Iterable[str]) -> list[str]:
    requested = list(dict.fromkeys(tile_ids))
    unknown = [tile_id for tile_id in requested if tile_id not in cn.tile_id_list]
    if unknown:
        raise ValueError(f"Unknown tile_ids requested: {unknown}")
    return requested


def _tile_output_uri(
    *,
    outputs_base: str,
    output_prefix: Optional[str],
    res_label: str,
    dataset: str,
    run_name: str,
    interval: str,
    tile_id: str,
) -> str:
    root = output_prefix.rstrip("/") if output_prefix else (
        f"{outputs_base.rstrip('/')}/{res_label}_tile_aggregation"
    )
    return (
        f"{root}/{dataset}/{run_name}/{interval}/"
        f"{tile_id}__{res_label}__{dataset}__{interval}.tif"
    )


def aggregate_emission_tile_to_s3(
    *,
    tile_id: str,
    dataset: str,
    interval: str,
    src_uri: str,
    dst_uri: str,
    run_name: str,
    date_tag: str,
    native_deg: float,
    target_deg: float,
    skip_existing: bool = False,
) -> str:
    """Aggregate one continuous emission tile and upload it to S3."""
    logger = lu.setup_logging()
    if skip_existing and _s3_exists(dst_uri):
        logger.info("Skipping existing output: %s", dst_uri)
        return dst_uri

    bounds = uu.get_10x10_tile_bounds(tile_id)
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)
    logger.info(
        "Aggregating %s %s for %s from %s to %s",
        tile_id,
        dataset,
        interval,
        native_deg,
        target_deg,
    )
    arr = cgr._agg_tile_to_target_windowed(
        tile_id=tile_id,
        chunk_length_pixels=chunk_length_pixels,
        per_pixel_total_or_state_tile=src_uri,
        native_deg=native_deg,
        target_deg=target_deg,
        is_final=True,
        logger=logger,
    ).astype(np.float32, copy=False)

    profile = {
        "driver": "GTiff",
        "width": arr.shape[1],
        "height": arr.shape[0],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_bounds(*bounds, arr.shape[1], arr.shape[0]),
        "nodata": None,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "DEFLATE",
        "predictor": 2,
        "bigtiff": "IF_SAFER",
        "num_threads": "ALL_CPUS",
    }
    tags = {
        "aggregation": "sum",
        "dataset": dataset,
        "inventory_period": interval,
        "native_deg": str(native_deg),
        "target_deg": str(target_deg),
        "tile_id": tile_id,
        "run_name": run_name,
        "date_tag": date_tag,
    }

    tmp_parent = os.environ.get("AGG_TILE_TMPDIR") or os.environ.get("AGG_GLOBAL_TMPDIR")
    if tmp_parent:
        os.makedirs(tmp_parent, exist_ok=True)
    fd, local_path = tempfile.mkstemp(
        prefix=f"{tile_id}_{dataset}_{deg_to_label(target_deg)}_",
        suffix=".tif",
        dir=tmp_parent,
    )
    os.close(fd)
    try:
        with rasterio.open(local_path, "w", **profile) as dst:
            dst.write(arr, 1)
            dst.update_tags(**tags)
        _upload_file(local_path, dst_uri)
    finally:
        try:
            os.remove(local_path)
        except OSError:
            pass
    logger.info("Uploaded %s", dst_uri)
    return dst_uri


def build_tasks(
    *,
    run_name: str,
    output_date: str,
    model_version: str,
    outputs_root: str,
    base_url: Optional[str],
    outputs_base: Optional[str],
    output_prefix: Optional[str],
    pixel_resolution: str,
    data_types: list[str],
    inventory_periods: list[str],
    tile_ids: list[str],
    native_deg: float,
    target_deg: float,
    skip_existing: bool,
) -> list:
    resolved_base_url, resolved_outputs_base = resolve_versioned_paths(
        model_version=model_version,
        outputs_root=outputs_root,
        base_url=base_url,
        outputs_base=outputs_base,
    )
    download_upload = build_download_upload_dict(
        pixel_resolution=pixel_resolution,
        run_name=run_name,
        target_deg=target_deg,
        base_url=resolved_base_url,
        output_date=output_date,
        outputs_base=resolved_outputs_base,
        data_types=data_types,
        inventory_periods=inventory_periods,
    )
    res_label = deg_to_label(target_deg)
    tasks = []
    for key, items in download_upload.items():
        dataset = items["dataset"]
        interval = items["interval"]
        for tile_id in tile_ids:
            src_uri = cgr._per_pixel_tile_path(items, tile_id)
            dst_uri = _tile_output_uri(
                outputs_base=resolved_outputs_base,
                output_prefix=output_prefix,
                res_label=res_label,
                dataset=dataset,
                run_name=run_name,
                interval=interval,
                tile_id=tile_id,
            )
            tasks.append(
                {
                    "key": key,
                    "tile_id": tile_id,
                    "dataset": dataset,
                    "interval": interval,
                    "src_uri": src_uri,
                    "dst_uri": dst_uri,
                    "delayed": dask.delayed(aggregate_emission_tile_to_s3)(
                        tile_id=tile_id,
                        dataset=dataset,
                        interval=interval,
                        src_uri=src_uri,
                        dst_uri=dst_uri,
                        run_name=run_name,
                        date_tag=output_date,
                        native_deg=native_deg,
                        target_deg=target_deg,
                        skip_existing=skip_existing,
                    ),
                }
            )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate selected drained/burned emission tiles to per-tile "
            "high-resolution GeoTIFFs."
        )
    )
    parser.add_argument("-cn", "--cluster_name", default="global_rasters")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--date_tag", default=DEFAULT_DATE_TAG)
    parser.add_argument("--model_version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--outputs_root", default=OUTPUT_ROOT)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--outputs_base", default=None)
    parser.add_argument(
        "--output_prefix",
        default=None,
        help=(
            "Optional S3 root for outputs. Defaults to "
            "<versioned_outputs>/<target>_tile_aggregation/."
        ),
    )
    parser.add_argument("-p", "--pixel_resolution", default="40000_pixels")
    parser.add_argument("--native_deg", type=float, default=DEFAULT_NATIVE_DEG)
    parser.add_argument("--target_deg", type=float, default=0.001)
    parser.add_argument(
        "--data_types",
        nargs="+",
        default=list(EMISSION_DATASETS),
        help="Emission datasets to aggregate. Defaults to drained and burned totals.",
    )
    parser.add_argument(
        "--inventory_periods",
        nargs="+",
        default=["2021_2024"],
        help="Inventory periods to aggregate, e.g. 2021_2024.",
    )
    parser.add_argument(
        "--tile_ids",
        nargs="+",
        required=True,
        help="10x10 degree tile IDs. Supports spaces or commas.",
    )
    parser.add_argument("--dask_batch", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run_local", action="store_true")
    args = parser.parse_args()

    assert_grid_divides_world(args.target_deg)
    data_types = _validate_data_types(_split_cli_items(args.data_types) or [])
    inventory_periods = _split_cli_items(args.inventory_periods) or []
    tile_ids = _validate_tile_ids(_split_cli_items(args.tile_ids) or [])

    logger = lu.setup_logging_main()
    tasks = build_tasks(
        run_name=args.run_name,
        output_date=args.date_tag,
        model_version=args.model_version,
        outputs_root=args.outputs_root,
        base_url=args.base_url,
        outputs_base=args.outputs_base,
        output_prefix=args.output_prefix,
        pixel_resolution=args.pixel_resolution,
        data_types=data_types,
        inventory_periods=inventory_periods,
        tile_ids=tile_ids,
        native_deg=args.native_deg,
        target_deg=args.target_deg,
        skip_existing=args.skip_existing,
    )

    lu.print_and_log(
        f"Prepared {len(tasks)} high-resolution emission tile outputs.",
        True,
        logger,
    )
    for task in tasks:
        lu.print_and_log(f"{task['src_uri']} -> {task['dst_uri']}", True, logger)
    if args.dry_run:
        return

    cluster = None
    client = None
    try:
        cluster, client, run_local = uu.connect_to_cluster(
            args.cluster_name,
            run_local=args.run_local,
        )
        if run_local or client is None:
            completed = 0
            for task in tasks:
                task["delayed"].compute()
                completed += 1
                lu.print_and_log(
                    f"Completed {completed}/{len(tasks)} high-resolution tile outputs.",
                    True,
                    logger,
                )
            return

        completed = 0
        for start in range(0, len(tasks), args.dask_batch):
            batch = tasks[start:start + args.dask_batch]
            futures = client.compute(
                [task["delayed"] for task in batch],
                sync=False,
                retries=args.retries,
            )
            future_to_task = {future: task for future, task in zip(futures, batch)}
            for future in as_completed(list(future_to_task.keys())):
                task = future_to_task[future]
                try:
                    future.result()
                except Exception:
                    logger.exception(
                        "High-resolution tile aggregation failed for %s %s %s",
                        task["tile_id"],
                        task["dataset"],
                        task["interval"],
                    )
                    raise
                completed += 1
                lu.print_and_log(
                    f"Completed {completed}/{len(tasks)} high-resolution tile outputs.",
                    True,
                    logger,
                )
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            try:
                cluster.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
