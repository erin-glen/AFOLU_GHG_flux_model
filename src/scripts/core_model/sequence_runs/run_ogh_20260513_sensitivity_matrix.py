#!/usr/bin/env python3
"""Run the OGH 20260513 publication sensitivity matrix on one Coiled cluster.

This launcher assumes the Coiled cluster already exists and is ready. It runs
the selected 2021-2024 sensitivity scenarios sequentially, streaming each
scenario to the console and a per-step log file.

Typical WSL usage:

    python -m src.scripts.core_model.sequence_runs.run_ogh_20260513_sensitivity_matrix \
      --cluster-name organic_soil_sensitivities

To continue through aggregation and zonal statistics in the same process:

    python -m src.scripts.core_model.sequence_runs.run_ogh_20260513_sensitivity_matrix \
      --cluster-name organic_soil_sensitivities \
      --phases model aggregate zonal
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.scripts.utilities import local_output_paths as lop


DEFAULT_CLUSTER_NAME = "organic_soil_sensitivities"
DEFAULT_MODEL_VERSION = "1_0_1"
DEFAULT_RUN_DATE = "20260525"
DEFAULT_THRESHOLD_CSV = (
    "docs/organic_soil_threshold_profiles/"
    "20260513_mixed_boreal_f1_temperate_f1_5_tropical_f2.csv"
)
CORE_SCENARIOS = (
    "ogh_250m",
    "ogh_750m",
    "ogh_low",
    "ogh_high",
    "gfw",
    "gpd",
)
DECOMPOSITION_SCENARIOS = (
    "ogh_area_low",
    "ogh_area_high",
    "ogh_ef_low",
    "ogh_ef_high",
)
DEFAULT_SCENARIOS = CORE_SCENARIOS + DECOMPOSITION_SCENARIOS
REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class Scenario:
    key: str
    run_name: str
    peat_dataset: str
    drainage_distance_m: int
    emission_factor_variant: str = "default"
    peat_threshold_scenario: str = "baseline"
    uses_mixed_thresholds: bool = False


SCENARIOS: Mapping[str, Scenario] = {
    "ogh_500m": Scenario(
        key="ogh_500m",
        run_name="ogh_sensitivity_500m",
        peat_dataset="ogh",
        drainage_distance_m=500,
        uses_mixed_thresholds=True,
    ),
    "ogh_250m": Scenario(
        key="ogh_250m",
        run_name="ogh_sensitivity_250m",
        peat_dataset="ogh",
        drainage_distance_m=250,
        uses_mixed_thresholds=True,
    ),
    "ogh_750m": Scenario(
        key="ogh_750m",
        run_name="ogh_sensitivity_750m",
        peat_dataset="ogh",
        drainage_distance_m=750,
        uses_mixed_thresholds=True,
    ),
    "ogh_low": Scenario(
        key="ogh_low",
        run_name="ogh_sensitivity_low",
        peat_dataset="ogh",
        drainage_distance_m=500,
        emission_factor_variant="low",
        peat_threshold_scenario="low_area",
        uses_mixed_thresholds=True,
    ),
    "ogh_high": Scenario(
        key="ogh_high",
        run_name="ogh_sensitivity_high",
        peat_dataset="ogh",
        drainage_distance_m=500,
        emission_factor_variant="high",
        peat_threshold_scenario="high_area",
        uses_mixed_thresholds=True,
    ),
    "ogh_area_low": Scenario(
        key="ogh_area_low",
        run_name="ogh_sensitivity_area_low",
        peat_dataset="ogh",
        drainage_distance_m=500,
        peat_threshold_scenario="low_area",
        uses_mixed_thresholds=True,
    ),
    "ogh_area_high": Scenario(
        key="ogh_area_high",
        run_name="ogh_sensitivity_area_high",
        peat_dataset="ogh",
        drainage_distance_m=500,
        peat_threshold_scenario="high_area",
        uses_mixed_thresholds=True,
    ),
    "ogh_ef_low": Scenario(
        key="ogh_ef_low",
        run_name="ogh_sensitivity_ef_low",
        peat_dataset="ogh",
        drainage_distance_m=500,
        emission_factor_variant="low",
        uses_mixed_thresholds=True,
    ),
    "ogh_ef_high": Scenario(
        key="ogh_ef_high",
        run_name="ogh_sensitivity_ef_high",
        peat_dataset="ogh",
        drainage_distance_m=500,
        emission_factor_variant="high",
        uses_mixed_thresholds=True,
    ),
    "gfw": Scenario(
        key="gfw",
        run_name="gfw_standard_model_500m",
        peat_dataset="gfw",
        drainage_distance_m=500,
    ),
    "gpd": Scenario(
        key="gpd",
        run_name="gpd_standard_model_500m",
        peat_dataset="gpd",
        drainage_distance_m=500,
    ),
}


@dataclass(frozen=True)
class Step:
    label: str
    phase: str
    scenario_key: str
    command: list[str]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def shjoin(argv: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in argv)


def safe_log_name(label: str) -> str:
    keep = []
    for char in label:
        keep.append(char if char.isalnum() or char in "._-" else "_")
    return "".join(keep).strip("_")


def split_cli_values(values: Sequence[str | int]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(part for part in str(value).replace(",", " ").split() if part)
    return out


def selected_scenarios(raw: Sequence[str]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    seen: set[str] = set()
    for key in raw:
        if key == "all":
            keys = DEFAULT_SCENARIOS
        elif key == "core":
            keys = CORE_SCENARIOS
        elif key == "decomposition":
            keys = DECOMPOSITION_SCENARIOS
        else:
            keys = (key,)
        for item in keys:
            if item not in SCENARIOS:
                valid = ", ".join(["all", *SCENARIOS])
                raise ValueError(f"Unknown scenario '{item}'. Valid choices: {valid}")
            if item not in seen:
                scenarios.append(SCENARIOS[item])
                seen.add(item)
    return scenarios


def model_step(
    scenario: Scenario,
    *,
    cluster_name: str,
    run_date: str,
    threshold_csv: str,
    peat_threshold: str,
    chunk_size: str,
) -> Step:
    cmd = [
        sys.executable,
        "-m",
        "src.scripts.core_model.0_drainage_emissions_model",
        "--cluster_name",
        cluster_name,
        "--full_model",
        "--chunk_size",
        chunk_size,
        "--start_year",
        "2021",
        "--end_year",
        "2024",
        "--interval_type",
        "five_year",
        "--count_burned_years",
        "--peat_dataset",
        scenario.peat_dataset,
        "--drainage_distance_threshold_m",
        str(scenario.drainage_distance_m),
        "--emission_factor_variant",
        scenario.emission_factor_variant,
        "--create_zarr",
        "--run_date",
        run_date,
        "--run_name",
        scenario.run_name,
    ]
    if scenario.uses_mixed_thresholds:
        cmd.extend(
            [
                "--peat_threshold",
                peat_threshold,
                "--peat_threshold_by_biome",
                threshold_csv,
                "--fscore_metric",
                "mixed",
                "--peat_threshold_scenario",
                scenario.peat_threshold_scenario,
            ]
        )
    return Step(
        label=f"model.{scenario.key}.{scenario.run_name}",
        phase="model",
        scenario_key=scenario.key,
        command=cmd,
    )


def aggregate_step(
    scenario: Scenario,
    *,
    cluster_name: str,
    run_date: str,
    interval_end_years: Sequence[str],
) -> Step:
    return Step(
        label=f"aggregate.{scenario.key}.{scenario.run_name}",
        phase="aggregate",
        scenario_key=scenario.key,
        command=[
            sys.executable,
            "-m",
            "src.scripts.core_model.02_aggregate_soils_outputs",
            "-cn",
            cluster_name,
            "--run_name",
            scenario.run_name,
            "--output_date",
            run_date,
            "--interval_end_years",
            *interval_end_years,
        ],
    )


def zonal_step(
    scenario: Scenario,
    *,
    cluster_name: str,
    run_date: str,
    model_version: str,
    interval_end_years: Sequence[str],
    zonal_chunk_size: str,
    contextual_groupers: Sequence[str],
    overwrite_zonal: bool,
) -> Step:
    cmd = [
        sys.executable,
        "-m",
        "src.scripts.zonal_statistics.02_run_zonal_stats",
        "--model_version",
        model_version,
        "--run_date",
        run_date,
        "--interval_type",
        "five_year",
        "--interval_end_years",
        *interval_end_years,
        "--cluster_name",
        cluster_name,
        "--run_name",
        scenario.run_name,
        "--chunk_size",
        zonal_chunk_size,
        "--contextual_groupers",
        *contextual_groupers,
        "--diagnostics",
        "off",
    ]
    if overwrite_zonal:
        cmd.append("--overwrite_existing")
    return Step(
        label=f"zonal.{scenario.key}.{scenario.run_name}",
        phase="zonal",
        scenario_key=scenario.key,
        command=cmd,
    )


def build_steps(args: argparse.Namespace, scenarios: Sequence[Scenario]) -> list[Step]:
    phases = tuple(args.phases)
    interval_end_years = split_cli_values(args.interval_end_years)
    contextual_groupers = split_cli_values(args.contextual_groupers)
    steps: list[Step] = []

    if "model" in phases:
        steps.extend(
            model_step(
                scenario,
                cluster_name=args.cluster_name,
                run_date=args.run_date,
                threshold_csv=args.threshold_csv,
                peat_threshold=args.peat_threshold,
                chunk_size=args.model_chunk_size,
            )
            for scenario in scenarios
        )
    if "aggregate" in phases:
        steps.extend(
            aggregate_step(
                scenario,
                cluster_name=args.cluster_name,
                run_date=args.run_date,
                interval_end_years=interval_end_years,
            )
            for scenario in scenarios
        )
    if "zonal" in phases:
        steps.extend(
            zonal_step(
                scenario,
                cluster_name=args.cluster_name,
                run_date=args.run_date,
                model_version=args.model_version,
                interval_end_years=interval_end_years,
                zonal_chunk_size=args.zonal_chunk_size,
                contextual_groupers=contextual_groupers,
                overwrite_zonal=not args.no_overwrite_zonal,
            )
            for scenario in scenarios
        )
    return steps


def wait_for_cluster_ready(cluster_name: str, wait_seconds: int) -> None:
    import coiled  # type: ignore

    deadline = time.time() + max(wait_seconds, 0)
    while True:
        clusters = coiled.list_clusters()
        for cluster in clusters:
            if cluster.get("name") != cluster_name:
                continue
            state = cluster.get("current_state", {}).get("state")
            cluster_id = cluster.get("id") or cluster.get("cluster_id")
            if state == "ready":
                print(f"[{now()}] Coiled cluster ready: {cluster_name} ({cluster_id})")
                return
            print(f"[{now()}] Coiled cluster {cluster_name} state={state}; waiting...")
            break
        else:
            print(f"[{now()}] Coiled cluster {cluster_name} not found; waiting...")

        if time.time() >= deadline:
            raise RuntimeError(
                f"Coiled cluster '{cluster_name}' is not ready. Start it first or "
                "increase --cluster-ready-wait-seconds."
            )
        time.sleep(30)


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    scenarios: Sequence[Scenario],
    steps: Sequence[Step],
    results: Sequence[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": now(),
        "repo_root": str(REPO_ROOT),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "steps": [
            {
                "label": step.label,
                "phase": step.phase,
                "scenario_key": step.scenario_key,
                "command": step.command,
            }
            for step in steps
        ],
        "results": list(results),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_step(step: Step, *, env: Mapping[str, str], log_dir: Path, dry_run: bool) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{safe_log_name(step.label)}.log"
    print(f"\n[{now()}] >>> {step.label}")
    print(f"COMMAND: {shjoin(step.command)}")
    print(f"LOGFILE: {log_file}")

    start = time.time()
    if dry_run:
        return {
            "label": step.label,
            "phase": step.phase,
            "scenario_key": step.scenario_key,
            "returncode": 0,
            "elapsed_seconds": 0,
            "log_file": str(log_file),
            "dry_run": True,
        }

    with log_file.open("w", buffering=1, encoding="utf-8") as handle:
        proc = subprocess.Popen(
            step.command,
            cwd=REPO_ROOT,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            handle.write(line)
        returncode = proc.wait()

    elapsed = time.time() - start
    print(f"[{now()}] <<< {step.label} rc={returncode} elapsed={elapsed / 60:0.1f} min")
    return {
        "label": step.label,
        "phase": step.phase,
        "scenario_key": step.scenario_key,
        "returncode": returncode,
        "elapsed_seconds": round(elapsed, 3),
        "log_file": str(log_file),
        "dry_run": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OGH 20260513 publication sensitivity scenarios sequentially."
    )
    parser.add_argument("--cluster-name", default=DEFAULT_CLUSTER_NAME)
    parser.add_argument("--run-date", default=DEFAULT_RUN_DATE)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--threshold-csv", default=DEFAULT_THRESHOLD_CSV)
    parser.add_argument("--peat-threshold", default="0.27")
    parser.add_argument("--model-chunk-size", default="1")
    parser.add_argument("--zonal-chunk-size", default="10000")
    parser.add_argument(
        "--interval-end-years",
        nargs="+",
        default=["2024"],
        help="Interval end years for aggregate/zonal phases.",
    )
    parser.add_argument(
        "--contextual-groupers",
        nargs="+",
        default=["all"],
        help="Contextual groupers for zonal statistics.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help=(
            "Scenario key to run. Repeatable. Use all, core, decomposition, or one of: "
            + ", ".join(SCENARIOS)
        ),
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["model", "aggregate", "zonal"],
        default=["model"],
        help="Pipeline phases to run, in model/aggregate/zonal order.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for logs. Defaults under AFOLU local output root.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Skip steps whose label contains any of these substrings.",
    )
    parser.add_argument("--skip-cluster-check", action="store_true")
    parser.add_argument("--cluster-ready-wait-seconds", type=int, default=900)
    parser.add_argument(
        "--no-overwrite-zonal",
        action="store_true",
        help="Do not pass --overwrite_existing to zonal statistics.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scenarios = selected_scenarios(args.scenario or ["all"])
    steps = build_steps(args, scenarios)
    if args.skip:
        steps = [
            step
            for step in steps
            if not any(skip_text in step.label for skip_text in args.skip)
        ]

    timestamp = run_stamp()
    if args.log_dir is None:
        args.log_dir = Path(lop.pipeline_log_dir("ogh_20260513_sensitivity_matrix")) / args.run_date / timestamp
    log_dir = Path(args.log_dir)
    manifest_path = log_dir / "run_manifest.json"

    print(f"[{now()}] Repo root: {REPO_ROOT}")
    print(f"[{now()}] Log dir: {log_dir}")
    print(f"[{now()}] Selected scenarios: {', '.join(s.key for s in scenarios)}")
    print(f"[{now()}] Selected phases: {', '.join(args.phases)}")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("AFOLU_CLUSTER_READY_WAIT_SECONDS", str(args.cluster_ready_wait_seconds))

    results: list[dict] = []
    write_manifest(manifest_path, args=args, scenarios=scenarios, steps=steps, results=results)

    if not args.dry_run and not args.skip_cluster_check:
        wait_for_cluster_ready(args.cluster_name, args.cluster_ready_wait_seconds)

    start = time.time()
    for step in steps:
        result = run_step(step, env=env, log_dir=log_dir, dry_run=args.dry_run)
        results.append(result)
        write_manifest(manifest_path, args=args, scenarios=scenarios, steps=steps, results=results)
        if result["returncode"] != 0 and not args.continue_on_error:
            break

    elapsed = time.time() - start
    failures = [item for item in results if item["returncode"] != 0]
    print("\n===== OGH SENSITIVITY MATRIX SUMMARY =====")
    print(f"Elapsed: {elapsed / 60:0.1f} min")
    print(f"Manifest: {manifest_path}")
    if failures:
        for failure in failures:
            print(f"FAILED: {failure['label']} rc={failure['returncode']}")
        return int(failures[0]["returncode"])

    print("All requested steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
