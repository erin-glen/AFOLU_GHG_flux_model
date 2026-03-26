#!/usr/bin/env python3
"""
Run the GFW + GPD 500 m soils pipeline sequentially on a single Coiled cluster.

This orchestrator:
- Creates (or attaches to) a Coiled cluster named "drainage_cluster".
- Executes the requested modules in sequence, reusing the same scheduler.
- Streams stdout/stderr to the console and to per-step log files.
- Supports dry-run, skip lists, and continue-on-error semantics.

Usage (examples):
  python run_standard_500m_pipeline.py
  python run_standard_500m_pipeline.py --dry-run
  python run_standard_500m_pipeline.py --continue-on-error --skip 02_zonal_stats[gpd]
  python run_standard_500m_pipeline.py --shutdown-cluster

Notes:
- Duplicate --run_name arguments in the provided shell snippets were removed.
- The cluster is left running by default; pass --shutdown-cluster to close it on exit.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple, Optional


def build_commands(
    output_date: str = "20251120",
    run_date: str = "20251120",
    interval_end_years: str = "2024",
    model_version: str = "0_9_7",
) -> List[Tuple[str, List[str]]]:
    """Return the ordered list of (label, argv) commands to execute."""

    cmds: List[Tuple[str, List[str]]] = []

    # --- 0) Drainage emissions model (GFW and GPD) ---
    base0 = [
        "python",
        "-m",
        "src.scripts.core_model.0_drainage_emissions_model",
        "--cluster_name",
        "drainage_cluster",
        "--full_model",
        "--chunk_size",
        "1",
        "--start_year",
        "2021",
        "--end_year",
        "2024",
        "--count_burned_years",
        "--interval_type",
        "five_year",
    ]

    cmds.append(
        (
            "0_drainage_emissions_model[gfw]",
            base0
            + [
                "--peat_dataset",
                "gfw",
                "--run_name",
                "gfw_standard_model_500m",
            ],
        )
    )
    cmds.append(
        (
            "0_drainage_emissions_model[gpd]",
            base0
            + [
                "--peat_dataset",
                "gpd",
                "--run_name",
                "gpd_standard_model_500m",
            ],
        )
    )

    # --- 2) Per-pixel soils outputs ---
    base2 = [
        "python",
        "-m",
        "src.scripts.core_model.2_per_pixel_soils_outputs",
        "--cluster_name",
        "drainage_cluster",
        "--chunk_size",
        "1",
        "--output_date",
        output_date,
    ]
    cmds.append(
        ("2_per_pixel_soils_outputs[gfw]", base2 + ["--run_name", "gfw_standard_model_500m"])
    )
    cmds.append(
        ("2_per_pixel_soils_outputs[gpd]", base2 + ["--run_name", "gpd_standard_model_500m"])
    )

    # --- 3) Aggregate soils outputs ---
    cmds.append(
        (
            "3_aggregate_soils_outputs[gfw]",
            [
                "python",
                "-m",
                "src.scripts.core_model.3_aggregate_soils_outputs",
                "-cn",
                "drainage_cluster",
                "--run_name",
                "gfw_standard_model_500m",
                "--output_date",
                output_date,
            ],
        )
    )
    cmds.append(
        (
            "3_aggregate_soils_outputs[gpd]",
            [
                "python",
                "-m",
                "src.scripts.core_model.3_aggregate_soils_outputs",
                "-cn",
                "drainage_cluster",
                "--run_name",
                "gpd_standard_model_500m",
                "--output_date",
                output_date,
            ],
        )
    )

    # --- Zonal statistics: build Zarr caches ---
    base_zarr = [
        "python",
        "-m",
        "src.scripts.zonal_statistics.01_build_zarr_caches",
        "--interval_end_years",
        str(interval_end_years),
        "--cluster_name",
        "drainage_cluster",
        "--run_date",
        run_date,
        "--model_version",
        model_version,
        "--tile_pixels",
        "40000",
        "--chunk_size",
        "8000",
    ]
    cmds.append(("01_build_zarr_caches[gfw]", base_zarr + ["--run_name", "gfw_standard_model_500m"]))
    cmds.append(("01_build_zarr_caches[gpd]", base_zarr + ["--run_name", "gpd_standard_model_500m"]))

    # --- Zonal statistics: run zonal stats ---
    base_zstats = [
        "python",
        "-m",
        "src.scripts.zonal_statistics.02_run_zonal_stats",
        "--interval_end_years",
        str(interval_end_years),
        "--cluster_name",
        "drainage_cluster",
        "--run_date",
        run_date,
        "--model_version",
        model_version,
        "--chunk_size",
        "10000",
        "--diagnostics",
        "off",
    ]
    cmds.append(("02_run_zonal_stats[gfw]", base_zstats + ["--run_name", "gfw_standard_model_500m"]))
    cmds.append(("02_run_zonal_stats[gpd]", base_zstats + ["--run_name", "gpd_standard_model_500m"]))

    return cmds


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def shjoin(argv: Iterable[str]) -> str:
    return " ".join(shlex.quote(s) for s in argv)


def run_cmd(
    label: str,
    argv: List[str],
    env: dict,
    log_dir: Path,
    dry_run: bool = False,
) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{label}.log"
    print(f"\n[{now()}] >>> {label}")
    print(f"COMMAND: {shjoin(argv)}")
    print(f"LOGFILE: {log_file.resolve()}")

    if dry_run:
        return 0

    # Stream to console and file
    with log_file.open("w", buffering=1, encoding="utf-8") as lf:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        return proc.wait()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run GFW+GPD 500m pipeline on one Coiled cluster.")
    p.add_argument("--output-date", default="20251120", help="YYYYMMDD for per-pixel + aggregate steps.")
    p.add_argument("--run-date", default="20251120", help="YYYYMMDD for zonal-statistics steps.")
    p.add_argument("--interval-end-years", default="2024", help="Interval end year(s) for zonal-statistics.")
    p.add_argument("--model-version", default="0_9_7", help="Model version for zonal-statistics.")
    p.add_argument("--log-dir", type=Path, default=Path("../logs"), help="Directory for per-step logs.")
    p.add_argument("--dry-run", action="store_true", help="Print the commands without executing them.")
    p.add_argument("--continue-on-error", action="store_true", help="Keep going if a step fails.")
    p.add_argument("--skip", nargs="*", default=[], help="List of step labels to skip (substring match).")
    p.add_argument("--no-cluster", action="store_true", help="Do not create/attach a Coiled cluster explicitly.")
    p.add_argument("--scheduler-address", default=None, help="Override DASK_SCHEDULER_ADDRESS for all steps.")
    p.add_argument("--shutdown-cluster", action="store_true", help="Close the Coiled cluster on exit.")
    args = p.parse_args(argv)

    steps = build_commands(
        output_date=args.output_date,
        run_date=args.run_date,
        interval_end_years=args.interval_end_years,
        model_version=args.model_version,
    )

    # Prepare environment
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    # Create or attach to the Coiled cluster and export scheduler address so all
    # subprocesses connect to the same scheduler (when they call Client()).
    cluster = None
    client = None
    if not args.no_cluster and args.scheduler_address is None:
        try:
            import coiled  # type: ignore
            from dask.distributed import Client  # type: ignore

            print(f"[{now()}] Creating/attaching Coiled cluster: drainage_cluster")
            cluster = coiled.Cluster(name="drainage_cluster", shutdown_on_close=False)
            client = Client(cluster)
            env["DASK_SCHEDULER_ADDRESS"] = client.scheduler.address
            print(f"[{now()}] DASK_SCHEDULER_ADDRESS={env['DASK_SCHEDULER_ADDRESS']}")
            try:
                dash = getattr(cluster, "dashboard_link", None)
                if dash:
                    print(f"[{now()}] Dask dashboard: {dash}")
            except Exception:
                pass
        except Exception as e:
            print(f"[{now()}] WARNING: Could not initialize Coiled cluster: {e}", file=sys.stderr)
            print(
                f"[{now()}] Proceeding without exporting DASK_SCHEDULER_ADDRESS. "
                f"Downstream scripts will rely on their --cluster_name logic.",
                file=sys.stderr,
            )

    if args.scheduler_address is not None:
        env["DASK_SCHEDULER_ADDRESS"] = args.scheduler_address
        print(f"[{now()}] Using provided DASK_SCHEDULER_ADDRESS={env['DASK_SCHEDULER_ADDRESS']}")

    # Execute
    failures = []
    start_all = time.time()
    for label, cmd in steps:
        if any(skip_key in label for skip_key in args.skip):
            print(f"[{now()}] SKIP: {label}")
            continue
        t0 = time.time()
        rc = run_cmd(label, cmd, env=env, log_dir=args.log_dir, dry_run=args.dry_run)
        dt = time.time() - t0
        print(f"[{now()}] <<< {label} finished with rc={rc} in {dt:0.1f}s")
        if rc != 0:
            failures.append((label, rc))
            if not args.continue_on_error:
                break

    total_dt = time.time() - start_all

    # Shutdown policy
    if client is not None:
        try:
            client.close()
        except Exception:
            pass
    if cluster is not None and args.shutdown_cluster:
        try:
            cluster.close()
        except Exception:
            pass

    # Summary
    print("\n===== SUMMARY =====")
    print(f"Total elapsed: {total_dt/60:0.1f} min")
    if failures:
        for label, rc in failures:
            print(f"FAILED: {label} (rc={rc})")
        return failures[0][1]
    else:
        print("All requested steps completed successfully.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
