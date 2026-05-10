"""Assemble a complete roads/canals chunk set for a delta rerun.

The changed chunks should be rerun into the target date folder. This helper
copies unchanged chunks from the previous date folders into that same target
date folder, so aggregation still sees a complete 4000-pixel input set.

The command is dry-run by default. Add ``--apply`` to perform S3 copies.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import posixpath
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from src.scripts.preprocessing.roads_canals.global_datasets import roads_io
import src.scripts.preprocessing.preprocessing_constants as cn


DEFAULT_SOURCE_DATES = {
    "osm_roads": "20251114",
    "osm_canals": "20251115",
    "grip_roads": "20251115",
}
DEFAULT_FEATURE_TYPES = ("osm_roads", "osm_canals", "grip_roads")
DEFAULT_PRODUCTS = ("presence", "distance")


def _read_changed_keys(manifest_path: str) -> set[str]:
    changed: set[str] = set()
    with open(manifest_path, newline="") as src:
        reader = csv.DictReader(src)
        for row_num, row in enumerate(reader, start=2):
            if row.get("chunk_key"):
                changed.add(row["chunk_key"])
                continue
            tile_id = (row.get("tile_id") or "").strip()
            if all(k in row and row[k] not in (None, "") for k in ("minx", "miny", "maxx", "maxy")):
                bounds = [float(row[k]) for k in ("minx", "miny", "maxx", "maxy")]
            elif row.get("chunk_bounds"):
                bounds = [float(v.strip()) for v in row["chunk_bounds"].split(",")]
            else:
                raise ValueError(f"Cannot derive chunk key at {manifest_path}:{row_num}")
            changed.add(roads_io.chunk_manifest_key(tile_id, bounds))
    return changed


def _parse_source_dates(values: Optional[list[str]]) -> dict[str, str]:
    dates = dict(DEFAULT_SOURCE_DATES)
    if not values:
        return dates
    for value in values:
        feature_type, sep, date_str = value.partition("=")
        if not sep or not feature_type or not date_str:
            raise ValueError(
                "--source_date values must look like feature_type=YYYYMMDD "
                f"(got {value!r})"
            )
        dates[feature_type] = date_str
    return dates


def _list_keys(s3, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=cn.s3_bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".tif") or key.endswith(".tiff"):
                keys.append(key)
    return keys


def _object_exists(s3, key: str) -> bool:
    try:
        s3.head_object(Bucket=cn.s3_bucket_name, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def _chunk_key_from_filename(filename: str) -> Optional[str]:
    parts = filename.split("__")
    if len(parts) < 3:
        return None
    return f"{parts[0]}__{parts[1]}"


def assemble(
    changed_manifest: str,
    target_date: str,
    source_dates: dict[str, str],
    feature_types: tuple[str, ...],
    products: tuple[str, ...],
    apply: bool,
    overwrite: bool,
    max_copies: Optional[int],
    summary_output: Optional[str],
) -> dict[str, object]:
    changed_keys = _read_changed_keys(changed_manifest)
    s3 = boto3.client("s3")
    summary: dict[str, object] = {
        "changed_manifest": changed_manifest,
        "target_date": target_date,
        "changed_chunk_count": len(changed_keys),
        "apply": apply,
        "overwrite": overwrite,
        "feature_product_summaries": [],
    }

    total_planned = 0
    total_copied = 0

    for feature_type in feature_types:
        if feature_type not in source_dates:
            raise ValueError(f"No source date configured for {feature_type}")
        source_date = source_dates[feature_type]
        for product in products:
            source_prefix = roads_io.product_prefix(feature_type, product, 4000, source_date)
            target_prefix = roads_io.product_prefix(feature_type, product, 4000, target_date)
            source_keys = _list_keys(s3, source_prefix)

            item = {
                "feature_type": feature_type,
                "product": product,
                "source_date": source_date,
                "source_prefix": source_prefix,
                "target_prefix": target_prefix,
                "source_objects": len(source_keys),
                "skipped_changed": 0,
                "skipped_existing": 0,
                "planned_copies": 0,
                "copied": 0,
                "unparseable_names": 0,
            }

            for source_key in source_keys:
                if max_copies is not None and total_planned >= int(max_copies):
                    break

                filename = posixpath.basename(source_key)
                chunk_key = _chunk_key_from_filename(filename)
                if chunk_key is None:
                    item["unparseable_names"] += 1
                    continue
                if chunk_key in changed_keys:
                    item["skipped_changed"] += 1
                    continue

                target_key = posixpath.join(target_prefix, filename)
                if (not overwrite) and _object_exists(s3, target_key):
                    item["skipped_existing"] += 1
                    continue

                item["planned_copies"] += 1
                total_planned += 1
                if apply:
                    s3.copy_object(
                        CopySource={"Bucket": cn.s3_bucket_name, "Key": source_key},
                        Bucket=cn.s3_bucket_name,
                        Key=target_key,
                    )
                    item["copied"] += 1
                    total_copied += 1

            summary["feature_product_summaries"].append(item)

    summary["planned_copies"] = total_planned
    summary["copied"] = total_copied

    if summary_output:
        os.makedirs(os.path.dirname(os.path.abspath(summary_output)), exist_ok=True)
        with open(summary_output, "w") as dst:
            json.dump(summary, dst, indent=2, sort_keys=True)
            dst.write("\n")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy unchanged roads/canals chunks from previous date folders into a target delta-run date folder."
    )
    parser.add_argument("--changed_manifest", required=True)
    parser.add_argument("--target_date", required=True)
    parser.add_argument(
        "--source_date",
        action="append",
        default=None,
        help=(
            "Feature source date mapping, repeatable. Defaults: "
            "osm_roads=20251114, osm_canals=20251115, grip_roads=20251115"
        ),
    )
    parser.add_argument("--feature_types", nargs="+", default=list(DEFAULT_FEATURE_TYPES))
    parser.add_argument("--products", nargs="+", default=list(DEFAULT_PRODUCTS))
    parser.add_argument("--summary_output", default=None)
    parser.add_argument("--max_copies", type=int, default=None, help="Optional cap for dry-run smoke tests")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target objects")
    parser.add_argument("--apply", action="store_true", help="Perform S3 copies; default is dry-run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = assemble(
        changed_manifest=args.changed_manifest,
        target_date=args.target_date,
        source_dates=_parse_source_dates(args.source_date),
        feature_types=tuple(args.feature_types),
        products=tuple(args.products),
        apply=bool(args.apply),
        overwrite=bool(args.overwrite),
        max_copies=args.max_copies,
        summary_output=args.summary_output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
