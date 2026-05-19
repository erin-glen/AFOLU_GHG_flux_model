"""Build a manifest of 1 degree chunks changed between two peat union masks.

This supports delta roads/canals reruns: only chunks whose 30 m union mask
changed need fresh presence and distance rasters. Unchanged chunks can be
copied forward from the previous roads/canals run before aggregation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import posixpath
from typing import Iterable, Optional

import boto3
import dask
import numpy as np
import rasterio
from rasterio.windows import from_bounds

from src.scripts.preprocessing.roads_canals.global_datasets import roads_io
from src.scripts.utilities import universal_utilities as uutil
import src.scripts.preprocessing.preprocessing_constants as cn


LOG = logging.getLogger("changed_peat_chunks")

DEFAULT_OLD_UNION_DATE = "20251110"
DEFAULT_NEW_UNION_DATE = "20260508"
DEFAULT_OUTPUT_DIR = os.path.join("logs", "roads_canals_delta")
FIELDNAMES = [
    "tile_id",
    "chunk_key",
    "chunk_bounds",
    "minx",
    "miny",
    "maxx",
    "maxy",
    "chunk_px",
    "old_peat_px",
    "new_peat_px",
    "added_peat_px",
    "removed_peat_px",
    "changed_peat_px",
    "change_class",
]


def _union_prefix_from_date(date_str: str) -> str:
    return posixpath.join(
        cn.processed_dir,
        "peat_mask",
        "union",
        "30m",
        "tiles",
        str(date_str),
    ) + "/"


def _normalize_s3_prefix(prefix: str) -> str:
    prefix = str(prefix)
    if prefix.startswith("s3://"):
        no_scheme = prefix[5:]
        bucket, _, key = no_scheme.partition("/")
        if bucket != cn.s3_bucket_name:
            raise ValueError(
                f"Expected bucket {cn.s3_bucket_name!r}; got {bucket!r} in {prefix!r}"
            )
        prefix = key
    return prefix.strip("/") + "/"


def _list_union_tiles(s3_client, prefix: str) -> dict[str, str]:
    prefix = _normalize_s3_prefix(prefix)
    tiles: dict[str, str] = {}
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=cn.s3_bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(roads_io.PEAT_30M_PATTERN):
                continue
            tile_id = os.path.basename(key).replace(roads_io.PEAT_30M_PATTERN, "")
            tiles[tile_id] = key
    return tiles


def _vsis3(key: Optional[str]) -> Optional[str]:
    if key is None:
        return None
    return f"/vsis3/{cn.s3_bucket_name}/{key}"


def _chunk_bounds(tile_id: str, chunk_size: float) -> Iterable[list[float]]:
    west, south, east, north = uutil.get_10x10_tile_bounds(tile_id)
    y = south
    eps = 1e-12
    while y < north - eps:
        x = west
        while x < east - eps:
            yield [x, y, x + chunk_size, y + chunk_size]
            x += chunk_size
        y += chunk_size


def _read_chunk(ds, bounds: list[float]) -> Optional[np.ndarray]:
    if ds is None:
        return None
    window = from_bounds(*bounds, transform=ds.transform).round_offsets().round_lengths()
    return ds.read(1, window=window, boundless=True, fill_value=0, masked=False)


def _classify_change(added: int, removed: int) -> str:
    if added and removed:
        return "added_and_removed"
    if added:
        return "added_only"
    if removed:
        return "removed_only"
    return "unchanged"


def _sort_key(row: dict[str, object]) -> tuple:
    return (
        str(row["tile_id"]),
        float(row["miny"]),
        float(row["minx"]),
        float(row["maxy"]),
        float(row["maxx"]),
    )


def _make_manifest_row(
    tile_id: str,
    bounds: list[float],
    *,
    change_class: str,
    old_peat_px: object = "",
    new_peat_px: object = "",
    added_peat_px: object = "",
    removed_peat_px: object = "",
    changed_peat_px: object = "",
) -> dict[str, object]:
    chunk_px = uutil.calc_chunk_length_pixels(bounds)
    return {
        "tile_id": tile_id,
        "chunk_key": roads_io.chunk_manifest_key(tile_id, bounds),
        "chunk_bounds": ",".join(roads_io.fmt_deg(v) for v in bounds),
        "minx": roads_io.fmt_deg(bounds[0]),
        "miny": roads_io.fmt_deg(bounds[1]),
        "maxx": roads_io.fmt_deg(bounds[2]),
        "maxy": roads_io.fmt_deg(bounds[3]),
        "chunk_px": chunk_px,
        "old_peat_px": old_peat_px,
        "new_peat_px": new_peat_px,
        "added_peat_px": added_peat_px,
        "removed_peat_px": removed_peat_px,
        "changed_peat_px": changed_peat_px,
        "change_class": change_class,
    }


def _expand_for_distance(
    changed_rows: list[dict[str, object]],
    valid_tile_ids: set[str],
    neighbor_chunks: int,
    chunk_size: float,
) -> list[dict[str, object]]:
    """Return changed chunks plus neighboring chunks that can see halo effects."""

    expanded: dict[str, dict[str, object]] = {}
    for row in changed_rows:
        minx = float(row["minx"])
        miny = float(row["miny"])
        for dx in range(-neighbor_chunks, neighbor_chunks + 1):
            for dy in range(-neighbor_chunks, neighbor_chunks + 1):
                west = minx + dx * chunk_size
                south = miny + dy * chunk_size
                if west < -180 or west >= 180 or south < -90 or south >= 90:
                    continue
                bounds = [west, south, west + chunk_size, south + chunk_size]
                tile_id = roads_io.tile_id_from_cell(int(round(west)), int(round(south)))
                if valid_tile_ids and tile_id not in valid_tile_ids:
                    continue
                chunk_key = roads_io.chunk_manifest_key(tile_id, bounds)
                if chunk_key == row["chunk_key"]:
                    expanded[chunk_key] = dict(row)
                elif chunk_key not in expanded:
                    expanded[chunk_key] = _make_manifest_row(
                        tile_id,
                        bounds,
                        change_class="distance_halo_neighbor",
                    )

    return sorted(expanded.values(), key=_sort_key)


def _compare_tile(
    tile_id: str,
    old_key: Optional[str],
    new_key: Optional[str],
    chunk_size: float,
    min_changed_pixels: int,
) -> list[dict[str, object]]:
    old_path = _vsis3(old_key)
    new_path = _vsis3(new_key)
    rows: list[dict[str, object]] = []

    old_ds = rasterio.open(old_path) if old_path else None
    new_ds = rasterio.open(new_path) if new_path else None
    try:
        for bounds in _chunk_bounds(tile_id, chunk_size):
            old_arr = _read_chunk(old_ds, bounds)
            new_arr = _read_chunk(new_ds, bounds)

            if old_arr is None and new_arr is None:
                continue
            if old_arr is None:
                old_arr = np.zeros_like(new_arr, dtype=np.uint8)
            if new_arr is None:
                new_arr = np.zeros_like(old_arr, dtype=np.uint8)

            old_mask = old_arr == 1
            new_mask = new_arr == 1
            added = int(np.count_nonzero(new_mask & ~old_mask))
            removed = int(np.count_nonzero(old_mask & ~new_mask))
            changed = added + removed
            if changed < min_changed_pixels:
                continue

            old_peat = int(np.count_nonzero(old_mask))
            new_peat = int(np.count_nonzero(new_mask))
            rows.append(
                _make_manifest_row(
                    tile_id,
                    bounds,
                    old_peat_px=old_peat,
                    new_peat_px=new_peat,
                    added_peat_px=added,
                    removed_peat_px=removed,
                    changed_peat_px=changed,
                    change_class=_classify_change(added, removed),
                )
            )
    finally:
        if old_ds is not None:
            old_ds.close()
        if new_ds is not None:
            new_ds.close()

    return rows


def _compare_tile_task(
    tile_id: str,
    old_key: Optional[str],
    new_key: Optional[str],
    chunk_size: float,
    min_changed_pixels: int,
) -> tuple[str, list[dict[str, object]]]:
    """Dask-friendly wrapper returning the tile id with its changed rows."""

    return tile_id, _compare_tile(
        tile_id,
        old_key,
        new_key,
        chunk_size,
        min_changed_pixels,
    )


def _iter_tile_comparisons(
    all_tile_ids: list[str],
    old_tiles: dict[str, str],
    new_tiles: dict[str, str],
    *,
    chunk_size: float,
    min_changed_pixels: int,
    client_mode: str,
    cluster_name: str,
    n_workers: int,
    worker_memory: str,
    batch_size: int,
):
    if client_mode == "local":
        for idx, tile_id in enumerate(all_tile_ids, start=1):
            LOG.info("Comparing tile %s (%d/%d)", tile_id, idx, len(all_tile_ids))
            yield tile_id, _compare_tile(
                tile_id,
                old_tiles.get(tile_id),
                new_tiles.get(tile_id),
                chunk_size,
                min_changed_pixels,
            )
        return

    cluster, client, run_local = uutil.connect_to_cluster(
        cluster_name=cluster_name,
        n_workers=n_workers,
        region="us-east-1",
        worker_memory=worker_memory,
    )
    if run_local:
        LOG.warning("Coiled cluster unavailable; falling back to local tile comparison.")
        for idx, tile_id in enumerate(all_tile_ids, start=1):
            LOG.info("Comparing tile %s (%d/%d)", tile_id, idx, len(all_tile_ids))
            yield tile_id, _compare_tile(
                tile_id,
                old_tiles.get(tile_id),
                new_tiles.get(tile_id),
                chunk_size,
                min_changed_pixels,
            )
        return

    LOG.info("Comparing tiles on Coiled cluster: %s", cluster.name)
    tasks = [
        dask.delayed(_compare_tile_task)(
            tile_id,
            old_tiles.get(tile_id),
            new_tiles.get(tile_id),
            chunk_size,
            min_changed_pixels,
        )
        for tile_id in all_tile_ids
    ]
    try:
        batch_size = max(int(batch_size), 1)
        for offset in range(0, len(tasks), batch_size):
            batch = tasks[offset:offset + batch_size]
            LOG.info(
                "Submitting tile-comparison batch %d-%d of %d",
                offset + 1,
                min(offset + len(batch), len(tasks)),
                len(tasks),
            )
            for tile_id, rows in dask.compute(*batch):
                yield tile_id, rows
    finally:
        client.close()
        cluster.close()


def build_manifest(
    old_union_prefix: str,
    new_union_prefix: str,
    output: str,
    summary_output: str,
    distance_output: Optional[str] = None,
    chunk_size: float = 1.0,
    tile_ids: Optional[list[str]] = None,
    max_tiles: Optional[int] = None,
    min_changed_pixels: int = 1,
    distance_neighbor_chunks: int = 1,
    client_mode: str = "local",
    cluster_name: str = "roads_canals",
    n_workers: int = 20,
    worker_memory: str = "64GiB",
    batch_size: int = 40,
) -> dict[str, object]:
    s3 = boto3.client("s3")
    old_tiles = _list_union_tiles(s3, old_union_prefix)
    new_tiles = _list_union_tiles(s3, new_union_prefix)
    valid_tile_ids = set(old_tiles) | set(new_tiles)
    all_tile_ids = sorted(set(old_tiles) | set(new_tiles))

    if tile_ids:
        requested = set(tile_ids)
        all_tile_ids = [tid for tid in all_tile_ids if tid in requested]
    if max_tiles is not None:
        all_tile_ids = all_tile_ids[: int(max_tiles)]

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(summary_output)), exist_ok=True)

    summary = {
        "old_union_prefix": _normalize_s3_prefix(old_union_prefix),
        "new_union_prefix": _normalize_s3_prefix(new_union_prefix),
        "chunk_size_degrees": chunk_size,
        "tile_count": len(all_tile_ids),
        "changed_chunk_count": 0,
        "old_peat_px": 0,
        "new_peat_px": 0,
        "added_peat_px": 0,
        "removed_peat_px": 0,
        "changed_peat_px": 0,
        "change_class_counts": {},
        "distance_neighbor_chunks": distance_neighbor_chunks,
        "distance_affected_chunk_count": None,
    }
    changed_rows: list[dict[str, object]] = []

    with open(output, "w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=FIELDNAMES)
        writer.writeheader()

        for tile_id, rows in _iter_tile_comparisons(
            all_tile_ids,
            old_tiles,
            new_tiles,
            chunk_size=chunk_size,
            min_changed_pixels=min_changed_pixels,
            client_mode=client_mode,
            cluster_name=cluster_name,
            n_workers=n_workers,
            worker_memory=worker_memory,
            batch_size=batch_size,
        ):
            LOG.info("Tile %s produced %d changed chunk(s)", tile_id, len(rows))
            for row in rows:
                writer.writerow(row)
                changed_rows.append(row)
                summary["changed_chunk_count"] += 1
                for key in (
                    "old_peat_px",
                    "new_peat_px",
                    "added_peat_px",
                    "removed_peat_px",
                    "changed_peat_px",
                ):
                    summary[key] += int(row[key])
                cls = str(row["change_class"])
                summary["change_class_counts"][cls] = summary["change_class_counts"].get(cls, 0) + 1

    if distance_output:
        distance_rows = _expand_for_distance(
            changed_rows,
            valid_tile_ids=valid_tile_ids,
            neighbor_chunks=distance_neighbor_chunks,
            chunk_size=chunk_size,
        )
        os.makedirs(os.path.dirname(os.path.abspath(distance_output)), exist_ok=True)
        with open(distance_output, "w", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(distance_rows)
        summary["distance_output"] = distance_output
        summary["distance_affected_chunk_count"] = len(distance_rows)

    with open(summary_output, "w") as dst:
        json.dump(summary, dst, indent=2, sort_keys=True)
        dst.write("\n")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a changed 1 degree peat-mask chunk manifest for delta roads/canals reruns."
    )
    parser.add_argument("--old_union_date", default=DEFAULT_OLD_UNION_DATE)
    parser.add_argument("--new_union_date", default=DEFAULT_NEW_UNION_DATE)
    parser.add_argument("--old_union_prefix", default=None, help="Optional S3 prefix overriding --old_union_date")
    parser.add_argument("--new_union_prefix", default=None, help="Optional S3 prefix overriding --new_union_date")
    parser.add_argument("--output", default=None, help="CSV output path")
    parser.add_argument("--distance_output", default=None, help="CSV output path for changed chunks plus halo-neighbor chunks")
    parser.add_argument("--summary_output", default=None, help="JSON summary output path")
    parser.add_argument("--chunk_size", type=float, default=1.0)
    parser.add_argument("--tile_ids", nargs="+", default=None, help="Optional subset of tile IDs")
    parser.add_argument("--max_tiles", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument("--min_changed_pixels", type=int, default=1)
    parser.add_argument("--client", default="local", choices=["local", "coiled"])
    parser.add_argument("--cluster_name", default="roads_canals", help="Coiled cluster name to attach to.")
    parser.add_argument("--n_workers", type=int, default=20, help="Expected Coiled worker count for cluster helper.")
    parser.add_argument("--worker_memory", default="64GiB", help="Expected Coiled worker memory for cluster helper.")
    parser.add_argument("--batch_size", type=int, default=40, help="Tile-comparison batch size for Coiled runs.")
    parser.add_argument(
        "--distance_neighbor_chunks",
        type=int,
        default=1,
        help="Number of neighboring chunk rings to include in the distance-impact manifest",
    )
    parser.add_argument("--loglevel", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.loglevel).upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    old_prefix = args.old_union_prefix or _union_prefix_from_date(args.old_union_date)
    new_prefix = args.new_union_prefix or _union_prefix_from_date(args.new_union_date)

    out_dir = os.path.join(
        DEFAULT_OUTPUT_DIR,
        f"{args.old_union_date}_to_{args.new_union_date}",
    )
    output = args.output or os.path.join(
        out_dir,
        f"changed_peat_chunks_{args.old_union_date}_to_{args.new_union_date}.csv",
    )
    distance_output = args.distance_output or os.path.join(
        out_dir,
        f"distance_affected_peat_chunks_{args.old_union_date}_to_{args.new_union_date}.csv",
    )
    summary_output = args.summary_output or os.path.splitext(output)[0] + "_summary.json"

    summary = build_manifest(
        old_union_prefix=old_prefix,
        new_union_prefix=new_prefix,
        output=output,
        summary_output=summary_output,
        distance_output=distance_output,
        chunk_size=args.chunk_size,
        tile_ids=args.tile_ids,
        max_tiles=args.max_tiles,
        min_changed_pixels=args.min_changed_pixels,
        distance_neighbor_chunks=args.distance_neighbor_chunks,
        client_mode=args.client,
        cluster_name=args.cluster_name,
        n_workers=args.n_workers,
        worker_memory=args.worker_memory,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Manifest written to {output}")
    if distance_output:
        print(f"Distance-impact manifest written to {distance_output}")
    print(f"Summary written to {summary_output}")


if __name__ == "__main__":
    main()
