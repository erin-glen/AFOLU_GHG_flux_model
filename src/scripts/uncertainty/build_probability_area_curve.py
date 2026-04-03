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
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build probability threshold-vs-area curve from class-area zonal outputs.")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input CSV, Parquet file, or Parquet directory with class-area values.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path for threshold-vs-area curve.")
    parser.add_argument("--adm0-id", type=int, default=None, help="Optional ADM0 filter. If omitted, aggregates all ADM0.")
    parser.add_argument("--probability-column", default="probability_class")
    parser.add_argument("--area-column", default="area_ha")
    parser.add_argument("--adm0-column", default="adm0_id")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
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
    df = read_table(args.input.resolve())

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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Wrote curve with {len(out)} thresholds to {args.output.resolve()}")


if __name__ == "__main__":
    main()
