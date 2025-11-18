#!/usr/bin/env python3
"""
Start or attach to a Coiled cluster and execute two pipeline steps sequentially
against the SAME cluster.

Designed for WSL/Ubuntu shell usage.

Example (uses defaults matching your example):
    python scripts/execute_sequence.py

Or with explicit overrides:
    python -m src.scripts.core_model.execute_sequence \
      --cluster-name zonal_stats \
      --run-date 20251118 \
      --model-version 0_9_5 \
      --run-name ogh_sensitivity_500m_23 \
      --cache-interval-end-years 2024 \
      --tile-pixels 40000 \
      --cache-chunk-size 8000 \
      --zonal-interval-end-years 2021 \
      --zonal-chunk-size 10000 \
      --diagnostics off \
      --shutdown
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import coiled  # type: ignore
except Exception as e:  # pragma: no cover
    print(
        "ERROR: This script requires the 'coiled' package. "
        "Install with: pip install coiled",
        file=sys.stderr,
    )
    raise

# ---------- Utilities ----------

def find_repo_root(start: Path) -> Path:
    """
    Try to locate the repo root (assumes a top-level 'src' directory).
    Falls back to the script's directory if not found.
    """
    path = start.resolve()
    for _ in range(6):
        if (path / "src").exists():
            return path
        if path.parent == path:
            break
        path = path.parent
    return start.resolve()


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    """Run a command, streaming stdout/stderr to the console."""
    print("\n‣ Running:", " ".join(cmd))
    print("  in:", str(cwd))
    if dry_run:
        print("  (dry-run; not executing)")
        return
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


# ---------- Main ----------

def main() -> None:
    # Defaults chosen to mirror your example
    today_ymd = datetime.now().strftime("%Y%m%d")

    p = argparse.ArgumentParser(description="Execute two zonal-stat steps on the same Coiled cluster.")
    # Cluster options
    p.add_argument("--cluster-name", default="zonal_stats", help="Coiled cluster name to create/attach")
    p.add_argument("--software", default=None, help="Optional Coiled software environment name")
    p.add_argument("--region", default=None, help="Optional cloud region (e.g., 'us-east-2')")
    p.add_argument("--n-workers", type=int, default=None, help="If provided, scale cluster to this many workers")
    p.add_argument("--adapt-min", type=int, default=None, help="If provided with --adapt-max, enable adaptive scaling")
    p.add_argument("--adapt-max", type=int, default=None, help="If provided with --adapt-min, enable adaptive scaling")
    p.add_argument("--wait-for-workers", type=int, default=0, help="Block until at least N workers are ready")
    p.add_argument("--keep", action="store_true", help="Keep cluster running on exit (default)")
    p.add_argument("--shutdown", action="store_true", help="Shutdown the cluster at the end of the run")
    # Execution options
    p.add_argument("--python-exe", default=sys.executable, help="Python interpreter to use for -m invocations")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    p.add_argument("--cwd", default=None, help="Working directory (defaults to repo root containing 'src/')")
    # Shared pipeline args
    p.add_argument("--run-date", default=today_ymd, help="Run date, e.g. 20251118")
    p.add_argument("--model-version", default="0_9_5", help="Model version string")
    p.add_argument("--run-name", default="ogh_sensitivity_500m_23", help="Run name")
    # Step 1 (build caches)
    p.add_argument("--cache-interval-end-years", default="2024", help="interval_end_years for build step")
    p.add_argument("--tile-pixels", default="40000", help="tile_pixels for build step")
    p.add_argument("--cache-chunk-size", default="8000", help="chunk_size for build step")
    # Step 2 (zonal stats)
    p.add_argument("--zonal-interval-end-years", default="2021", help="interval_end_years for zonal step")
    p.add_argument("--zonal-chunk-size", default="10000", help="chunk_size for zonal step")
    p.add_argument("--diagnostics", default="off", choices=["on", "off"], help="diagnostics for zonal step")

    args = p.parse_args()

    # Resolve working directory (so `python -m src...` works no matter where we call from)
    script_dir = Path(__file__).resolve().parent
    repo_root = Path(args.cwd) if args.cwd else find_repo_root(script_dir)

    # 1) Start or attach to the cluster (reuse by name).
    print(f"⏳ Starting or attaching to Coiled cluster: {args.cluster_name}")
    cluster_kwargs = {
        "name": args.cluster_name,
        "shutdown_on_close": False,  # keep cluster alive across subprocesses
    }
    if args.software:
        cluster_kwargs["software"] = args.software
    if args.region:
        cluster_kwargs["region"] = args.region

    cluster = coiled.Cluster(**cluster_kwargs)

    # Optionally scale or adapt
    if args.n_workers is not None:
        cluster.scale(args.n_workers)
        print(f"📈 Scaled cluster to {args.n_workers} workers.")
    elif args.adapt_min is not None and args.adapt_max is not None:
        cluster.adapt(minimum=args.adapt_min, maximum=args.adapt_max)
        print(f"📈 Adaptive scaling enabled: min={args.adapt_min}, max={args.adapt_max}")

    # Get a Dask client & optionally wait for workers
    client = cluster.get_client()
    dash = getattr(client, "dashboard_link", None)
    if dash:
        print(f"🔗 Dask dashboard: {dash}")
    if getattr(cluster, "details_url", None):
        print(f"🔗 Coiled cluster page: {cluster.details_url}")

    if args.wait_for_workers and isinstance(args.wait_for_workers, int):
        print(f"⏱️  Waiting for {args.wait_for_workers} workers to be ready...")
        cluster.wait_for_workers(args.wait_for_workers)
        print("✅ Workers ready.")

    # 2) Run the two modules sequentially against the SAME cluster
    # Step 1: build zarr caches
    build_cmd = [
        args.python_exe, "-m", "src.scripts.zonal_statistics.01_build_zarr_caches",
        "--interval_end_years", str(args.cache_interval_end_years),
        "--cluster_name", args.cluster_name,
        "--run_date", str(args.run_date),
        "--model_version", str(args.model_version),
        "--run_name", str(args.run_name),
        "--tile_pixels", str(args.tile_pixels),
        "--chunk_size", str(args.cache_chunk_size),
    ]

    # Step 2: run zonal stats
    zonal_cmd = [
        args.python_exe, "-m", "src.scripts.zonal_statistics.02_run_zonal_stats",
        "--interval_end_years", str(args.zonal_interval_end_years),
        "--cluster_name", args.cluster_name,
        "--run_date", str(args.run_date),
        "--model_version", str(args.model_version),
        "--run_name", str(args.run_name),
        "--chunk_size", str(args.zonal_chunk_size),
        "--diagnostics", str(args.diagnostics),
    ]

    try:
        run_cmd(build_cmd, cwd=repo_root, dry_run=args.dry_run)
        run_cmd(zonal_cmd, cwd=repo_root, dry_run=args.dry_run)
        print("\n🎉 Both steps completed successfully.")
    finally:
        # 3) Cluster lifecycle on exit
        if args.shutdown and not args.dry_run:
            print("🛑 Shutting down cluster...")
            # Explicitly stop the cluster (works even if shutdown_on_close=False)
            cluster.shutdown()  # synchronous in normal usage
            print("✅ Cluster shut down.")
        else:
            keep_note = "keeping cluster running" if not args.dry_run else "no-op (dry-run)"
            print(f"ℹ️  {keep_note}. You can shut it down later via the Coiled UI or by name.")

if __name__ == "__main__":
    main()
