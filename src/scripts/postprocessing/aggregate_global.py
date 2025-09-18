"""Stage A: Aggregate per-region summaries to global totals for a model run.

Key behaviors
-------------
- Run-aware directories:
  * Regional inputs default to ``<project root>/outputs/global/results/<run>``.
  * Global totals default to ``<project root>/outputs/global/aggregated/<run>``.
- Input discovery: picks up every file matching ``--input-pattern`` and
  concatenates them before aggregating.
- Flexible statistics: ``sum`` (default), ``mean``, or ``median`` across any
  group-by columns.
- Outputs: one CSV per invocation written alongside the run, with filenames
  that include the ``--model-run-name`` unless you supply ``--output-name``.

Examples
--------

# --- Aggregate with defaults (sum by scenario and year) ---
python scripts/aggregate_global.py \
  --model-run-name ogh_standard_model

# --- Point at a different project checkout (e.g., notebooks directory) ---
python scripts/aggregate_global.py \
  --project-root /mnt/efs/repos/AFOLU_GHG_flux_model \
  --model-run-name ogh_standard_model

# --- Only read parquet tiles and average across scenarios ---
python scripts/aggregate_global.py \
  --model-run-name ogh_standard_model \
  --input-pattern "*.parquet" \
  --group-by year region \
  --statistic mean

# --- Customise both input and output directories ---
python scripts/aggregate_global.py \
  --model-run-name ogh_standard_model \
  --input-dir /tmp/results/ogh_standard_model \
  --output-dir /tmp/aggregated/ogh_standard_model

# --- Override the output filename (``.csv`` will be appended if missing) ---
python scripts/aggregate_global.py \
  --model-run-name ogh_standard_model \
  --output-name ogh_standard_model__global_totals

# --- Dry-run: inspect discovery + aggregation without writing anything ---
python scripts/aggregate_global.py \
  --model-run-name ogh_standard_model \
  --dry-run
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

SUPPORTED_STATS = ("sum", "mean", "median")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line options."""

    parser = argparse.ArgumentParser(
        description="Aggregate per-region model outputs to global totals.",
    )
    parser.add_argument(
        "--model-run-name",
        required=True,
        help="Identifier for the model run; controls input/output locations.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional override for the project root directory.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing per-region outputs. Defaults to "
            "<project root>/outputs/global/results/<model_run_name>."
        ),
    )
    parser.add_argument(
        "--input-pattern",
        default="*.csv",
        help="Glob pattern used to locate regional results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination for aggregated files. Defaults to "
            "<project root>/outputs/global/aggregated/<model_run_name>."
        ),
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional filename for the aggregated dataset.",
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=("scenario", "year"),
        help="Columns that define unique global totals.",
    )
    parser.add_argument(
        "--value-column",
        default="value",
        help="Column containing numeric values to aggregate.",
    )
    parser.add_argument(
        "--statistic",
        choices=SUPPORTED_STATS,
        default="sum",
        help="Statistic applied to the value column when aggregating.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover inputs and compute aggregates without writing files.",
    )
    return parser.parse_args(argv)


def locate_project_root(project_root: Optional[Path]) -> Path:
    """Return the absolute project root directory."""

    if project_root is not None:
        return project_root.expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def resolve_with_base(base: Path, provided: Optional[Path], default: Path) -> Path:
    """Resolve ``provided`` relative to ``base`` with a sensible default."""

    if provided is None:
        return default

    candidate = provided.expanduser()
    return candidate if candidate.is_absolute() else base / candidate


def default_input_dir(project_root: Path, model_run: str) -> Path:
    return project_root / "outputs" / "global" / "results" / model_run


def default_output_dir(project_root: Path, model_run: str) -> Path:
    return project_root / "outputs" / "global" / "aggregated" / model_run


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_inputs(directory: Path, pattern: str) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")

    files = sorted(path for path in directory.glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' were found in {directory}",
        )
    return files


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ""}:
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension for dataframe input: {path}")


def load_inputs(files: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for file in files:
        frames.append(read_table(file))
    return pd.concat(frames, ignore_index=True)


def aggregate_frame(
    frame: pd.DataFrame,
    group_by: Sequence[str],
    value_column: str,
    statistic: str,
) -> pd.DataFrame:
    missing = [column for column in (*group_by, value_column) if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(f"Columns required for aggregation were missing: {joined}")

    grouped = frame.groupby(list(group_by), dropna=False)[value_column]
    if statistic == "sum":
        aggregated = grouped.sum()
    elif statistic == "mean":
        aggregated = grouped.mean()
    elif statistic == "median":
        aggregated = grouped.median()
    else:  # pragma: no cover - safeguarded by argument parsing
        raise ValueError(f"Unsupported statistic: {statistic}")

    return aggregated.reset_index().sort_values(list(group_by)).reset_index(drop=True)


def determine_output_path(output_dir: Path, output_name: Optional[str], model_run: str) -> Path:
    if output_name:
        destination = output_dir / output_name
    else:
        destination = output_dir / f"global_totals_{model_run}.csv"

    if destination.suffix == "":
        destination = destination.with_suffix(".csv")
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    project_root = locate_project_root(args.project_root)
    input_dir = resolve_with_base(
        project_root,
        args.input_dir,
        default_input_dir(project_root, args.model_run_name),
    )
    output_dir = resolve_with_base(
        project_root,
        args.output_dir,
        default_output_dir(project_root, args.model_run_name),
    )

    try:
        files = discover_inputs(input_dir, args.input_pattern)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    frame = load_inputs(files)

    try:
        aggregated = aggregate_frame(
            frame,
            tuple(args.group_by),
            args.value_column,
            args.statistic,
        )
    except KeyError as exc:
        print(f"Error: {exc}")
        return 1

    if args.dry_run:
        print("Dry-run complete; aggregated dataframe not written to disk.")
        return 0

    ensure_directory(output_dir)
    destination = determine_output_path(output_dir, args.output_name, args.model_run_name)
    aggregated.to_csv(destination, index=False)
    print(f"Wrote aggregated results to {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())