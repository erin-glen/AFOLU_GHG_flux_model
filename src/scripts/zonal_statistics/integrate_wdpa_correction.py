# -*- coding: utf-8 -*-
"""Append validated WDPA 12-16 corrections to the completed June baseline.

The source outputs are never modified.  All five intervals are validated and
staged locally before the new isolated S3 destination is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import s3fs

from src.scripts.zonal_statistics.rerun_wdpa_affected_tiles import FULL_TILE_IDS


LOGGER = logging.getLogger("wdpa_correction_integration")

MODEL_VERSION = "1_0_1"
INPUT_RUN_NAME = "ogh_mixed_f1_f15_f2_20260513"
INPUT_RUN_DATE = "20260525"
BASELINE_RUN_NAME = f"{INPUT_RUN_NAME}_starting_cpf_global"
BASELINE_RUN_DATE = "20260610"
CORRECTION_RUN_NAME = f"{INPUT_RUN_NAME}_wdpa_fix_70tiles"
CORRECTION_RUN_DATE = "20260731"
OUTPUT_RUN_NAME_DEFAULT = f"{BASELINE_RUN_NAME}_wdpa_fixed"
OUTPUT_RUN_DATE_DEFAULT = "20260802"

ZONAL_ROOT = (
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/"
    f"version_{MODEL_VERSION}/zonal_stats"
)
BASELINE_ROOT = f"{ZONAL_ROOT}/{BASELINE_RUN_NAME}/{BASELINE_RUN_DATE}"
CORRECTION_ROOT = f"{ZONAL_ROOT}/{CORRECTION_RUN_NAME}/{CORRECTION_RUN_DATE}"

INTERVALS = ("2001_2005", "2006_2010", "2011_2015", "2016_2020", "2021_2024")
AFFECTED_WDPA_CODES = (12, 13, 14, 15, 16)

BASELINE_RIVER_BASINS = (
    "s3://gfw2-data/climate/AFOLU_flux_model/contextual_layer_global_zarr/"
    "river_basins/v2018/20260213_fillValue_removed/river_basins_20260213.zarr"
)
CORRECTION_RIVER_BASINS = (
    "s3://gfw2-data/climate/AFOLU_flux_model/contextual_layer_global_zarr/"
    "river_basins/v2018/20260508_fillValue_removed/river_basins_20260508.zarr"
)


def read_json(fs: s3fs.S3FileSystem, uri: str) -> dict[str, Any]:
    with fs.open(uri, "r") as stream:
        return json.load(stream)


def read_parquet(fs: s3fs.S3FileSystem, uri: str) -> pa.Table:
    with fs.open(uri, "rb") as stream:
        return pq.read_table(stream)


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def s3_etag(fs: s3fs.S3FileSystem, uri: str) -> str:
    info = fs.info(uri)
    return str(info.get("ETag") or info.get("etag") or "").strip('"')


def upload_interval_artifacts(
    fs: s3fs.S3FileSystem,
    *,
    local_dir: Path,
    remote_dir: str,
    expected_parquet_md5: str,
) -> str:
    """Upload and verify data artifacts before publishing the completion marker."""

    required_names = {"part-0.parquet", "_zonal_stats_manifest.json", "_COMPLETE.json"}
    local_files = {path.name: path for path in local_dir.iterdir() if path.is_file()}
    missing = sorted(required_names - set(local_files))
    if missing:
        raise RuntimeError(f"Staged interval is missing required artifacts: {missing}")

    parquet_path = local_files["part-0.parquet"]
    local_parquet_md5 = md5_file(parquet_path)
    if local_parquet_md5 != expected_parquet_md5:
        raise RuntimeError(
            "Staged parquet hash changed before upload: "
            f"expected={expected_parquet_md5} actual={local_parquet_md5}"
        )

    uploaded_etags: dict[str, str] = {}
    for name in sorted(required_names - {"_COMPLETE.json"}):
        path = local_files[name]
        remote_uri = f"{remote_dir}/{name}"
        expected_md5 = md5_file(path)
        fs.put(str(path), remote_uri)
        actual_etag = s3_etag(fs, remote_uri)
        if actual_etag != expected_md5:
            raise RuntimeError(
                f"Uploaded artifact hash differs for {name}: "
                f"local={expected_md5} s3={actual_etag}"
            )
        uploaded_etags[name] = actual_etag

    marker_path = local_files["_COMPLETE.json"]
    marker_uri = f"{remote_dir}/_COMPLETE.json"
    expected_marker_md5 = md5_file(marker_path)
    fs.put(str(marker_path), marker_uri)
    actual_marker_etag = s3_etag(fs, marker_uri)
    if actual_marker_etag != expected_marker_md5:
        raise RuntimeError(
            "Uploaded completion marker hash differs: "
            f"local={expected_marker_md5} s3={actual_marker_etag}"
        )

    return uploaded_etags["part-0.parquet"]


def filter_wdpa(table: pa.Table, *, minimum: int, maximum: int) -> pa.Table:
    wdpa = table.column("wdpa")
    mask = pc.and_(
        pc.greater_equal(wdpa, pa.scalar(minimum, type=wdpa.type)),
        pc.less_equal(wdpa, pa.scalar(maximum, type=wdpa.type)),
    )
    return table.filter(mask)


def validate_values_and_keys(table: pa.Table, *, label: str) -> None:
    values = table.column("value").combine_chunks().to_numpy(zero_copy_only=False)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label} contains non-finite values")
    if np.any(values < 0):
        raise RuntimeError(f"{label} contains negative values")
    key_columns = [name for name in table.column_names if name != "value"]
    keys = table.select(key_columns).to_pandas()
    duplicate_count = int(keys.duplicated(keep=False).sum())
    if duplicate_count:
        raise RuntimeError(f"{label} contains {duplicate_count} duplicate key rows")


def validate_manifest_pair(
    baseline: dict[str, Any],
    correction: dict[str, Any],
    *,
    interval: str,
) -> None:
    common_keys = (
        "model_version",
        "run_name",
        "run_date",
        "interval_type",
        "selected_fluxes",
        "selected_flux_type_labels",
        "selected_contextual_groupers",
        "adm0_zarr_path",
        "pixel_area_zarr_path",
        "pixel_area_var_name",
        "align_tolerance_fraction",
    )
    for key in common_keys:
        if baseline.get(key) != correction.get(key):
            raise RuntimeError(
                f"Source manifests differ for {interval} field {key}: "
                f"baseline={baseline.get(key)!r} correction={correction.get(key)!r}"
            )

    baseline_paths = baseline.get("contextual_grouper_paths") or {}
    correction_paths = correction.get("contextual_grouper_paths") or {}
    for key in ("wdpa", "landmark", "primary_forest", "kba", "drivers_of_loss"):
        if baseline_paths.get(key) != correction_paths.get(key):
            raise RuntimeError(f"Contextual source differs for {interval}: {key}")
    if (
        baseline_paths.get("river_basins") != BASELINE_RIVER_BASINS
        or correction_paths.get("river_basins") != CORRECTION_RIVER_BASINS
    ):
        raise RuntimeError(
            f"River-basin sources differ from the pair proven byte-equivalent: {interval}"
        )

    baseline_tiles = set(baseline.get("processed_tile_ids") or [])
    correction_tiles = set(correction.get("processed_tile_ids") or [])
    expected_correction_tiles = set(FULL_TILE_IDS)
    if len(baseline_tiles) != 208:
        raise RuntimeError(f"Expected 208 baseline tiles for {interval}, found {len(baseline_tiles)}")
    if correction_tiles != expected_correction_tiles:
        missing = sorted(expected_correction_tiles - correction_tiles)
        unexpected = sorted(correction_tiles - expected_correction_tiles)
        raise RuntimeError(
            f"Correction tiles differ from the exact affected-tile manifest for {interval}: "
            f"missing={missing} unexpected={unexpected}"
        )
    if not expected_correction_tiles.issubset(baseline_tiles):
        missing = sorted(expected_correction_tiles - baseline_tiles)
        raise RuntimeError(
            f"Affected correction tiles are missing from the baseline for {interval}: {missing}"
        )


def flux_totals(table: pa.Table) -> dict[str, float]:
    frame = table.select(["flux_type", "value"]).to_pandas()
    return {
        str(name): float(group["value"].astype("float64").sum())
        for name, group in frame.groupby("flux_type", sort=True)
    }


def prepare_interval(
    fs: s3fs.S3FileSystem,
    *,
    interval: str,
    baseline_root: str,
    correction_root: str,
    output_run_name: str,
    output_run_date: str,
    local_root: Path | None,
) -> tuple[dict[str, Any], pa.Table | None]:
    baseline_prefix = f"{baseline_root}/{interval}/combined_state"
    correction_prefix = f"{correction_root}/{interval}/combined_state"
    baseline_manifest = read_json(fs, f"{baseline_prefix}/_zonal_stats_manifest.json")
    correction_manifest = read_json(fs, f"{correction_prefix}/_zonal_stats_manifest.json")
    baseline_marker = read_json(fs, f"{baseline_prefix}/_COMPLETE.json")
    correction_marker = read_json(fs, f"{correction_prefix}/_COMPLETE.json")
    if baseline_marker.get("success") is not True or correction_marker.get("success") is not True:
        raise RuntimeError(f"Source completion marker is not successful: {interval}")
    validate_manifest_pair(baseline_manifest, correction_manifest, interval=interval)

    baseline_uri = f"{baseline_prefix}/part-0.parquet"
    correction_uri = f"{correction_prefix}/part-0.parquet"
    baseline = read_parquet(fs, baseline_uri)
    correction = read_parquet(fs, correction_uri)
    if not baseline.schema.equals(correction.schema, check_metadata=False):
        raise RuntimeError(f"Baseline and correction schemas differ: {interval}")
    if baseline.schema.names != correction.schema.names:
        raise RuntimeError(f"Baseline and correction column order differs: {interval}")

    validate_values_and_keys(baseline, label=f"baseline {interval}")
    validate_values_and_keys(correction, label=f"correction {interval}")
    baseline_affected = filter_wdpa(baseline, minimum=12, maximum=16)
    correction_affected = filter_wdpa(correction, minimum=12, maximum=16)
    if baseline_affected.num_rows:
        raise RuntimeError(f"Baseline unexpectedly contains WDPA 12-16 rows: {interval}")
    if not correction_affected.num_rows:
        raise RuntimeError(f"Correction contains no WDPA 12-16 rows: {interval}")
    correction_codes = set(
        correction_affected.column("wdpa").combine_chunks().to_pylist()
    )
    if not correction_codes.issubset(set(AFFECTED_WDPA_CODES)):
        raise RuntimeError(f"Correction contains unexpected WDPA codes: {interval}")

    integrated = pa.concat_tables([baseline, correction_affected])
    validate_values_and_keys(integrated, label=f"integrated {interval}")
    if integrated.num_rows != baseline.num_rows + correction_affected.num_rows:
        raise RuntimeError(f"Integrated row-count reconciliation failed: {interval}")
    if not integrated.slice(0, baseline.num_rows).equals(baseline):
        raise RuntimeError(f"Baseline rows were not preserved exactly: {interval}")
    if not integrated.slice(baseline.num_rows).equals(correction_affected):
        raise RuntimeError(f"Correction rows were not preserved exactly: {interval}")

    summary: dict[str, Any] = {
        "interval": interval,
        "baseline_prefix": baseline_prefix,
        "correction_prefix": correction_prefix,
        "baseline_rows": baseline.num_rows,
        "correction_rows_total": correction.num_rows,
        "correction_rows_selected": correction_affected.num_rows,
        "integrated_rows": integrated.num_rows,
        "baseline_wdpa_12_16_rows": baseline_affected.num_rows,
        "correction_wdpa_codes_selected": sorted(int(code) for code in correction_codes),
        "correction_flux_totals": flux_totals(correction_affected),
        "baseline_s3_etag": s3_etag(fs, baseline_uri),
        "correction_s3_etag": s3_etag(fs, correction_uri),
        "schema_matches": True,
        "baseline_preserved_exactly": True,
        "correction_preserved_exactly": True,
        "duplicate_key_rows": 0,
        "nonfinite_values": 0,
        "negative_values": 0,
    }

    if local_root is None:
        return summary, integrated

    output_dir = local_root / interval / "combined_state"
    output_dir.mkdir(parents=True, exist_ok=False)
    parquet_path = output_dir / "part-0.parquet"
    pq.write_table(integrated, parquet_path, compression="snappy")
    written = pq.read_table(parquet_path)
    if not written.equals(integrated):
        raise RuntimeError(f"Local parquet round-trip differs: {interval}")

    output_manifest = deepcopy(baseline_manifest)
    output_manifest.update(
        {
            "output_run_name": output_run_name,
            "output_run_date": output_run_date,
            "execution_mode": "baseline_plus_correction_integration",
            "roi_mode": "global_integrated",
            "tile_count": len(baseline_manifest.get("processed_tile_ids") or []),
            "integration": {
                "schema_version": 1,
                "strategy": "append_correction_rows_where_wdpa_between_12_and_16",
                "affected_wdpa_codes": list(AFFECTED_WDPA_CODES),
                "baseline_prefix": baseline_prefix,
                "correction_prefix": correction_prefix,
                "river_basin_sources_proven_byte_equivalent_on": "2026-07-31",
                **summary,
            },
        }
    )
    manifest_path = output_dir / "_zonal_stats_manifest.json"
    manifest_path.write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    marker = {
        "success": True,
        "branch": "combined_state",
        "interval": interval,
        "model_version": MODEL_VERSION,
        "run_name": INPUT_RUN_NAME,
        "run_date": INPUT_RUN_DATE,
        "output_run_name": output_run_name,
        "output_run_date": output_run_date,
        "integration_strategy": "baseline_plus_wdpa_12_16",
        "uploaded_file_count": 2,
    }
    marker_path = output_dir / "_COMPLETE.json"
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")
    summary["output_md5"] = md5_file(parquet_path)
    summary["local_parquet"] = str(parquet_path)
    return summary, None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--baseline-root", default=BASELINE_ROOT)
    parser.add_argument("--correction-root", default=CORRECTION_ROOT)
    parser.add_argument("--output-run-name", default=OUTPUT_RUN_NAME_DEFAULT)
    parser.add_argument("--output-run-date", default=OUTPUT_RUN_DATE_DEFAULT)
    parser.add_argument("--local-output", type=Path, default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    fs = s3fs.S3FileSystem(anon=False)
    output_root = f"{ZONAL_ROOT}/{args.output_run_name}/{args.output_run_date}"
    protected = {args.baseline_root.rstrip("/"), args.correction_root.rstrip("/")}
    if output_root.rstrip("/") in protected:
        raise RuntimeError("Refusing to write to a source prefix")
    if fs.exists(output_root):
        raise RuntimeError(f"Output destination already exists: {output_root}")

    local_root = args.local_output or Path(
        f"/mnt/c/tmp/afolu/wdpa_integration/{args.output_run_name}/{args.output_run_date}"
    )
    if args.execute and local_root.exists():
        raise RuntimeError(f"Local output already exists: {local_root}")

    report: dict[str, Any] = {
        "passed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_root": args.baseline_root,
        "correction_root": args.correction_root,
        "output_root": output_root,
        "output_run_name": args.output_run_name,
        "output_run_date": args.output_run_date,
        "execute": bool(args.execute),
        "intervals": {},
    }
    if args.execute:
        local_root.mkdir(parents=True, exist_ok=False)

    for interval in INTERVALS:
        LOGGER.info("Validating and integrating interval %s", interval)
        summary, _ = prepare_interval(
            fs,
            interval=interval,
            baseline_root=args.baseline_root,
            correction_root=args.correction_root,
            output_run_name=args.output_run_name,
            output_run_date=args.output_run_date,
            local_root=local_root if args.execute else None,
        )
        report["intervals"][interval] = summary

    report["passed"] = True
    if not args.execute:
        print(json.dumps(report, indent=2, sort_keys=True))
        print("Plan/validation only. Add --execute to stage and upload the integrated output.")
        return report

    report_path = local_root / "_wdpa_integration_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if fs.exists(output_root):
        raise RuntimeError(f"Output destination appeared during staging: {output_root}")

    for interval in INTERVALS:
        local_dir = local_root / interval / "combined_state"
        remote_dir = f"{output_root}/{interval}/combined_state"
        expected_md5 = report["intervals"][interval]["output_md5"]
        actual_etag = upload_interval_artifacts(
            fs,
            local_dir=local_dir,
            remote_dir=remote_dir,
            expected_parquet_md5=expected_md5,
        )
        report["intervals"][interval]["output_s3_etag"] = actual_etag

    report["uploaded_and_verified"] = True
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    fs.put(str(report_path), f"{output_root}/_wdpa_integration_report.json")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
