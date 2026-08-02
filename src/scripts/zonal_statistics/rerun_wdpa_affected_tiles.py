# -*- coding: utf-8 -*-
"""Run the isolated WDPA affected-tile correction against the June baseline.

Safe sequence::

    python -m src.scripts.zonal_statistics.rerun_wdpa_affected_tiles --mode smoke --execute
    python -m src.scripts.zonal_statistics.rerun_wdpa_affected_tiles --mode check-smoke
    python -m src.scripts.zonal_statistics.rerun_wdpa_affected_tiles --mode full --execute

The smoke and full runs use current contextual sources. Their schemas are
checked against the completed June 2026 starting-composite-primary-forest
global rerun. Both remote output and local tile staging use isolated paths;
the June baseline is never overwritten.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import platform
from typing import Any, Sequence

import pyarrow.parquet as pq
import s3fs


LOGGER = logging.getLogger("wdpa_affected_tile_rerun")

MODEL_VERSION = "1_0_1"
INPUT_RUN_NAME = "ogh_mixed_f1_f15_f2_20260513"
INPUT_RUN_DATE = "20260525"
BASELINE_RUN_NAME = f"{INPUT_RUN_NAME}_starting_cpf_global"
BASELINE_RUN_DATE = "20260610"
OUTPUT_RUN_DATE_DEFAULT = "20260731"
CLUSTER_NAME_DEFAULT = "organic_soils_zonal_tests"
SMOKE_OUTPUT_RUN_NAME = f"{INPUT_RUN_NAME}_wdpa_fix_smoke"
FULL_OUTPUT_RUN_NAME = f"{INPUT_RUN_NAME}_wdpa_fix_70tiles"

ZONAL_ROOT = (
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/"
    f"version_{MODEL_VERSION}/zonal_stats"
)
BASELINE_ROOT = f"{ZONAL_ROOT}/{BASELINE_RUN_NAME}/{BASELINE_RUN_DATE}"

INTERVALS = {
    2005: "2001_2005",
    2010: "2006_2010",
    2015: "2011_2015",
    2020: "2016_2020",
    2024: "2021_2024",
}

# Exact intersection of the 90 WDPA-affected land tiles with the 208
# combined_state source tiles. The same 208-tile inventory, including these
# 70 affected tiles, is present in all five June baseline periods.
FULL_TILE_IDS = (
    "00N_030E", "00N_080W", "00N_150E", "00N_160E", "10N_000E",
    "10N_070W", "10N_080W", "10N_110E", "10N_120E", "10S_070W",
    "10S_080W", "10S_140E", "20N_070W", "20N_080W", "20N_110E",
    "20N_120E", "30N_080E", "30N_090E", "30N_090W", "30N_100W",
    "30S_140E", "40N_010E", "40N_010W", "40N_020E", "40N_030E",
    "40N_040E", "40N_080W", "40N_090W", "40N_100W", "40N_110W",
    "40N_120E", "40N_120W", "40N_130E", "40N_130W", "40N_140E",
    "40S_140E", "50N_000E", "50N_010E", "50N_010W", "50N_020E",
    "50N_050E", "50N_060E", "50N_060W", "50N_070W", "50N_080W",
    "50N_090W", "50N_100W", "50N_110W", "50N_120W", "50N_130W",
    "50N_140E", "60N_000E", "60N_010E", "60N_010W", "60N_020E",
    "60N_110W", "60N_120W", "60N_130W", "60N_140W", "60N_150W",
    "60N_160W", "70N_010E", "70N_020E", "70N_030E", "70N_110W",
    "70N_120W", "70N_130W", "70N_140W", "70N_150W", "70N_160W",
)

# Covers a code-12-only tile, a code-13-only tile, and a mixed 12/13 tile.
SMOKE_TILE_IDS = ("10N_080W", "60N_020E", "10N_120E")

# Verified 2026-07-31: these two Zarr prefixes contain the same 18,853
# objects. Every metadata object, coordinate chunk, and band_data chunk has
# the same ETag and size. The path difference therefore does not change river
# basin group assignments.
BASELINE_RIVER_BASINS = (
    "s3://gfw2-data/climate/AFOLU_flux_model/contextual_layer_global_zarr/"
    "river_basins/v2018/20260213_fillValue_removed/river_basins_20260213.zarr"
)
CURRENT_RIVER_BASINS = (
    "s3://gfw2-data/climate/AFOLU_flux_model/contextual_layer_global_zarr/"
    "river_basins/v2018/20260508_fillValue_removed/river_basins_20260508.zarr"
)


def read_json(fs: s3fs.S3FileSystem, uri: str) -> dict[str, Any]:
    with fs.open(uri, "r") as stream:
        return json.load(stream)


def baseline_manifest(fs: s3fs.S3FileSystem, interval: str) -> dict[str, Any]:
    return read_json(
        fs,
        f"{BASELINE_ROOT}/{interval}/combined_state/_zonal_stats_manifest.json",
    )


def validate_current_contextual_profile(module: Any, fs: s3fs.S3FileSystem) -> None:
    """Ensure current grouping semantics match the corrected June baseline."""

    manifest = baseline_manifest(fs, "2016_2020")
    baseline_paths = manifest.get("contextual_grouper_paths") or {}
    selected = manifest.get("selected_contextual_groupers") or []
    expected_selected = [
        "wdpa",
        "landmark",
        "primary_forest",
        "kba",
        "river_basins",
        "drivers_of_loss",
    ]
    if selected != expected_selected:
        raise RuntimeError(
            f"June baseline contextual groupers differ: expected={expected_selected} actual={selected}"
        )

    for key in ("wdpa", "landmark", "primary_forest", "kba", "drivers_of_loss"):
        current_path = module.OPTIONAL_CONTEXTUAL_GROUPERS[key]["zarr_path"]
        if current_path != baseline_paths.get(key):
            raise RuntimeError(
                f"Current {key} path differs from the June baseline: "
                f"current={current_path} baseline={baseline_paths.get(key)}"
            )

    primary_name = module.OPTIONAL_CONTEXTUAL_GROUPERS["primary_forest"]["name"]
    if primary_name != "starting_composite_primary_forest":
        raise RuntimeError(
            "Current primary-forest output name does not match the corrected June schema: "
            f"{primary_name}"
        )

    baseline_river = baseline_paths.get("river_basins")
    current_river = module.OPTIONAL_CONTEXTUAL_GROUPERS["river_basins"]["zarr_path"]
    if baseline_river != BASELINE_RIVER_BASINS or current_river != CURRENT_RIVER_BASINS:
        raise RuntimeError(
            "River-basin paths no longer match the pair proven equivalent on 2026-07-31: "
            f"baseline={baseline_river} current={current_river}"
        )


def output_prefix(output_run_name: str, output_run_date: str, interval: str) -> str:
    return f"{ZONAL_ROOT}/{output_run_name}/{output_run_date}/{interval}/combined_state"


def build_zonal_argv(
    *,
    mode: str,
    cluster_name: str,
    output_run_name: str,
    output_run_date: str,
    local_output: str | None,
) -> list[str]:
    if mode not in {"smoke", "full"}:
        raise ValueError(f"Unsupported run mode: {mode}")
    tiles = SMOKE_TILE_IDS if mode == "smoke" else FULL_TILE_IDS
    local = local_output or (
        f"/mnt/c/tmp/afolu/wdpa_zonal_rerun/{output_run_name}/{output_run_date}"
    )
    return [
        "--model_version", MODEL_VERSION,
        "--run_name", INPUT_RUN_NAME,
        "--run_date", INPUT_RUN_DATE,
        "--interval_type", "five_year",
        "--interval_end_years", *[str(year) for year in INTERVALS],
        "--cluster_name", cluster_name,
        "--output_run_name", output_run_name,
        "--output_run_date", output_run_date,
        "--local_output", local,
        "--contextual_groupers", "all",
        "--execution_mode", "tile",
        "--tile_ids", ",".join(tiles),
        "--data_tile_filter", "auto",
        "--chunk_size", "10000",
        "--diagnostics", "off",
        "--keep_local",
        "--keep_tile_stage",
    ]


def assert_isolated_destination(
    fs: s3fs.S3FileSystem,
    *,
    output_run_name: str,
    output_run_date: str,
    resume: bool = False,
) -> None:
    protected = {
        (INPUT_RUN_NAME, INPUT_RUN_DATE),
        (BASELINE_RUN_NAME, BASELINE_RUN_DATE),
    }
    if (output_run_name, output_run_date) in protected:
        raise RuntimeError("Refusing to use a protected production or corrected-baseline prefix.")
    destination = f"{ZONAL_ROOT}/{output_run_name}/{output_run_date}"
    if fs.exists(destination) and not resume:
        raise RuntimeError(
            f"Output destination already exists; refusing to overwrite or mix runs without "
            f"--resume: {destination}"
        )
    if fs.exists(destination):
        LOGGER.info(
            "Resume requested for existing isolated destination. The zonal runner will "
            "validate remote manifests before skipping completed intervals: %s",
            destination,
        )


def _schema(fs: s3fs.S3FileSystem, uri: str):
    with fs.open(uri, "rb") as stream:
        return pq.ParquetFile(stream).schema_arrow


def _affected_area_summary(fs: s3fs.S3FileSystem, uri: str) -> dict[str, Any]:
    counts = {code: 0 for code in range(12, 17)}
    area_ha = {code: 0.0 for code in range(12, 17)}
    with fs.open(uri, "rb") as stream:
        parquet = pq.ParquetFile(stream)
        for batch in parquet.iter_batches(
            columns=["wdpa", "flux_type", "value"],
            batch_size=250_000,
        ):
            frame = batch.to_pandas()
            selected = frame[
                frame["wdpa"].between(12, 16)
                & frame["flux_type"].eq("area__ha")
            ]
            for code, group in selected.groupby("wdpa"):
                code_i = int(code)
                counts[code_i] += len(group)
                area_ha[code_i] += float(group["value"].sum())
    return {"area_row_counts": counts, "area_ha": area_ha}


def validate_output(
    *,
    output_run_name: str,
    output_run_date: str,
    expected_tiles: Sequence[str],
) -> dict[str, Any]:
    fs = s3fs.S3FileSystem(anon=False)
    expected = sorted(expected_tiles)
    summaries: dict[str, Any] = {}
    for interval in INTERVALS.values():
        prefix = output_prefix(output_run_name, output_run_date, interval)
        manifest = read_json(fs, f"{prefix}/_zonal_stats_manifest.json")
        marker = read_json(fs, f"{prefix}/_COMPLETE.json")
        if marker.get("success") is not True:
            raise RuntimeError(f"Completion marker is not successful: {prefix}")
        if sorted(manifest.get("processed_tile_ids") or []) != expected:
            raise RuntimeError(
                f"Processed tiles differ for {interval}: "
                f"expected={expected} actual={manifest.get('processed_tile_ids')}"
            )
        if (
            manifest.get("output_run_name") != output_run_name
            or manifest.get("output_run_date") != output_run_date
        ):
            raise RuntimeError(f"Manifest output isolation metadata differs: {prefix}")
        if manifest.get("execution_mode") != "tile":
            raise RuntimeError(f"Expected tile execution mode: {prefix}")

        result_uri = f"{prefix}/part-0.parquet"
        baseline_uri = f"{BASELINE_ROOT}/{interval}/combined_state/part-0.parquet"
        result_schema = _schema(fs, result_uri)
        baseline_schema = _schema(fs, baseline_uri)
        if not result_schema.equals(baseline_schema, check_metadata=False):
            raise RuntimeError(
                f"Output schema differs from the June baseline for {interval}:\n"
                f"result={result_schema}\nbaseline={baseline_schema}"
            )
        affected = _affected_area_summary(fs, result_uri)
        if not any(value > 0 for value in affected["area_ha"].values()):
            raise RuntimeError(f"No positive WDPA 12-16 area rows found: {result_uri}")
        summaries[interval] = {
            "prefix": prefix,
            "tile_count": len(expected),
            "schema_matches_june_baseline": True,
            **affected,
        }
    return {
        "passed": True,
        "baseline_root": BASELINE_ROOT,
        "output_run_name": output_run_name,
        "output_run_date": output_run_date,
        "intervals": summaries,
    }


def _format_underlying_command(argv: Sequence[str]) -> str:
    return (
        "python -m src.scripts.zonal_statistics.02_run_zonal_stats "
        + " ".join(argv)
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["smoke", "check-smoke", "full"],
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cluster-name", default=CLUSTER_NAME_DEFAULT)
    parser.add_argument("--output-run-name", default=None)
    parser.add_argument("--output-run-date", default=OUTPUT_RUN_DATE_DEFAULT)
    parser.add_argument("--local-output", default=None)
    parser.add_argument("--smoke-output-run-name", default=SMOKE_OUTPUT_RUN_NAME)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an existing isolated output. Completed remote intervals are only "
            "skipped when their manifests match; preserved local tile stages are reused."
        ),
    )
    return parser.parse_args(argv)


def require_linux_for_coiled_execution(*, execute: bool) -> None:
    if execute and platform.system() != "Linux":
        raise RuntimeError(
            "Coiled zonal reruns must be launched from WSL/Linux using the "
            "coiled_20251119 environment."
        )


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    require_linux_for_coiled_execution(
        execute=args.execute and args.mode in {"smoke", "full"}
    )
    if len(FULL_TILE_IDS) != 70 or len(set(FULL_TILE_IDS)) != 70:
        raise RuntimeError("The embedded affected-tile manifest must contain exactly 70 unique tiles.")
    if not set(SMOKE_TILE_IDS).issubset(FULL_TILE_IDS):
        raise RuntimeError("Smoke tiles must be a subset of the 70 affected tiles.")

    output_run_name = args.output_run_name or (
        FULL_OUTPUT_RUN_NAME if args.mode == "full" else args.smoke_output_run_name
    )
    if args.mode == "check-smoke":
        result = validate_output(
            output_run_name=output_run_name,
            output_run_date=args.output_run_date,
            expected_tiles=SMOKE_TILE_IDS,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    fs = s3fs.S3FileSystem(anon=False)
    module = importlib.import_module("src.scripts.zonal_statistics.02_run_zonal_stats")
    validate_current_contextual_profile(module, fs)
    if args.mode == "full":
        # A successful five-period, schema-compatible smoke output gates the
        # expensive 70-tile run.
        validate_output(
            output_run_name=args.smoke_output_run_name,
            output_run_date=args.output_run_date,
            expected_tiles=SMOKE_TILE_IDS,
        )

    zonal_argv = build_zonal_argv(
        mode=args.mode,
        cluster_name=args.cluster_name,
        output_run_name=output_run_name,
        output_run_date=args.output_run_date,
        local_output=args.local_output,
    )
    tile_count = len(SMOKE_TILE_IDS if args.mode == "smoke" else FULL_TILE_IDS)
    print(f"Mode: {args.mode}")
    print(f"Tiles per interval: {tile_count}")
    print(f"Intervals: {list(INTERVALS.values())}")
    print(f"Remote output: {ZONAL_ROOT}/{output_run_name}/{args.output_run_date}")
    print(f"Underlying command: {_format_underlying_command(zonal_argv)}")
    if not args.execute:
        print("Plan only. Add --execute to start the zonal-statistics run.")
        return None

    assert_isolated_destination(
        fs,
        output_run_name=output_run_name,
        output_run_date=args.output_run_date,
        resume=args.resume,
    )
    module.main(zonal_argv)
    result = validate_output(
        output_run_name=output_run_name,
        output_run_date=args.output_run_date,
        expected_tiles=SMOKE_TILE_IDS if args.mode == "smoke" else FULL_TILE_IDS,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
