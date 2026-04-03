#!/usr/bin/env python3
"""Build threshold-vs-area curves from area-by-probability-class tables.

Input expected columns:
- adm0_id
- probability_class (1..100)
- area_ha

Output columns:
- threshold
- area_ha

By default this sums all adm0 to global, then computes
area_ha(threshold=t) = sum_{p>=t} area_ha(p).

You can either provide --input directly, or pass --probability-date to read
the default 02b output location:
  s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/uncertainty/
  area_probability/<probability_date>/by_adm0_probability_class/
"""

from __future__ import annotations

import argparse
import posixpath
from pathlib import Path

import pandas as pd

ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
UNCERTAINTY_ROOT = posixpath.join(ROOT, "uncertainty")
DEFAULT_AREA_PROBABILITY_ROOT = posixpath.join(UNCERTAINTY_ROOT, "area_probability")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build probability threshold-vs-area curve from class-area zonal outputs.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=str,
        help="Input CSV, Parquet file, or Parquet directory with class-area values.",
    )
    source.add_argument(
        "--probability-date",
        type=str,
        help="Probability date used by 02b_run_probability_class_area_stats (e.g., 20251105).",
    )
    parser.add_argument(
        "--area-probability-root",
        default=DEFAULT_AREA_PROBABILITY_ROOT,
        help=(
            "Root containing 02b probability-area outputs. "
            "Used with --probability-date. "
            f"Default: {DEFAULT_AREA_PROBABILITY_ROOT}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path for threshold-vs-area curve. "
            "Default with --probability-date: ./area_vs_threshold_<date>.csv; "
            "otherwise ./area_vs_threshold.csv"
        ),
    )
    parser.add_argument("--adm0-id", type=int, default=None, help="Optional ADM0 filter. If omitted, aggregates all ADM0.")
    parser.add_argument("--probability-column", default="probability_class")
    parser.add_argument("--area-column", default="area_ha")
    parser.add_argument("--adm0-column", default="adm0_id")
    return parser.parse_args()


def resolve_input_path(args: argparse.Namespace) -> str:
    if args.input:
        return args.input

    probability_date = str(args.probability_date)
    return posixpath.join(
        args.area_probability_root.rstrip("/"),
        probability_date,
        "by_adm0_probability_class",
    )


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output.resolve()

    if args.probability_date:
        return Path(f"area_vs_threshold_{args.probability_date}.csv").resolve()
    return Path("area_vs_threshold.csv").resolve()


def read_table(path_str: str) -> pd.DataFrame:
    if path_str.startswith("s3://"):
        return pd.read_parquet(path_str)

    path = Path(path_str).expanduser().resolve()
    if path.is_dir():
        parquet_files = sorted(path.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under directory: {path}")
        return pd.concat([pd.read_parquet(fp) for fp in parquet_files], ignore_index=True)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input extension '{suffix}'. Use CSV or Parquet.")


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args)
    output_path = resolve_output_path(args)
    df = read_table(input_path)

    required = [args.adm0_column, args.probability_column, args.area_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available: {df.columns.tolist()}")

    work = df[[args.adm0_column, args.probability_column, args.area_column]].copy()
    work = work.dropna()
    work[args.probability_column] = work[args.probability_column].astype(int)
    work[args.area_column] = work[args.area_column].astype(float)

    if args.adm0_id is not None:
        work = work.loc[work[args.adm0_column] == int(args.adm0_id)].copy()

    # Keep probability classes 1..100, matching zonal output conventions.
    work = work.loc[(work[args.probability_column] >= 1) & (work[args.probability_column] <= 100)]

    grouped = work.groupby(args.probability_column, as_index=False)[args.area_column].sum()
    grouped = grouped.rename(columns={args.probability_column: "probability_class", args.area_column: "area_ha"})

    # Fill missing classes with zero to ensure deterministic thresholds.
    classes = pd.DataFrame({"probability_class": list(range(1, 101))})
    grouped = classes.merge(grouped, on="probability_class", how="left").fillna({"area_ha": 0.0})
    grouped = grouped.sort_values("probability_class", ascending=False).reset_index(drop=True)

    grouped["area_ha"] = grouped["area_ha"].astype(float)
    grouped["threshold"] = grouped["probability_class"].astype(float) / 100.0
    grouped["area_ha"] = grouped["area_ha"].cumsum()

    out = grouped[["threshold", "area_ha"]].sort_values("threshold").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Read class-area table from: {input_path}")
    print(f"Wrote curve with {len(out)} thresholds to {output_path}")


if __name__ == "__main__":
    main()
