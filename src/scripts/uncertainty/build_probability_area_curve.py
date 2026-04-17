#!/usr/bin/env python3
"""Build threshold-vs-area curves from area-by-probability-class tables.

Input expected columns:
- adm0_id
- probability_class (1..100)
- area_ha
- biome_id (optional, when --per-biome is used)

Output columns:
- threshold
- area_ha
- biome (only when --per-biome is used, in the combined CSV)

By default this sums all adm0 to global, then computes
area_ha(threshold=t) = sum_{p>=t} area_ha(p).

You can either provide --input directly, or pass --probability-date to read
the default 02b output location:
  s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/uncertainty/
  area_probability/<probability_date>/by_adm0_probability_class/

Examples:
python -m src.scripts.uncertainty.build_probability_area_curve \
  --probability-date 20251105 \
  --output ./area_vs_threshold_20251105.csv

python -m src.scripts.uncertainty.build_probability_area_curve \
  --probability-date 20251105 \
  --per-biome \
  --output ./area_vs_threshold_20251105_biome.csv
"""

from __future__ import annotations

import argparse
import posixpath
from pathlib import Path

import fsspec
import pandas as pd

from src.scripts.utilities import constants_and_names as cn

BIOME_ID_TO_NAME = {v: k for k, v in cn.ecozone_codes.items() if v > 0}

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
    parser.add_argument("--per-biome", action="store_true", default=False,
                        help="Produce separate threshold-vs-area CSVs per biome.")
    parser.add_argument("--biome-column", default="biome_id",
                        help="Column containing biome IDs. Default: biome_id")
    return parser.parse_args()


def resolve_input_path(args: argparse.Namespace) -> str:
    if args.input:
        return args.input

    probability_date = str(args.probability_date)
    subdir = "by_adm0_probability_class_biome" if args.per_biome else "by_adm0_probability_class"
    return posixpath.join(
        args.area_probability_root.rstrip("/"),
        probability_date,
        subdir,
    )


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output.resolve()

    if args.probability_date:
        return Path(f"area_vs_threshold_{args.probability_date}.csv").resolve()
    return Path("area_vs_threshold.csv").resolve()


def _read_parquet_directory(path_str: str) -> pd.DataFrame:
    """Read all parquet files under a local or remote directory path."""
    fs, base_path = fsspec.core.url_to_fs(path_str)
    parquet_paths = sorted(fs.glob(f"{base_path.rstrip('/')}/**/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under directory: {path_str}")

    protocol = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol
    parquet_urls = [f"{protocol}://{p}" if protocol != "file" else p for p in parquet_paths]
    return pd.concat([pd.read_parquet(url) for url in parquet_urls], ignore_index=True)


def read_table(path_str: str) -> pd.DataFrame:
    if path_str.startswith("s3://"):
        if path_str.lower().endswith((".parquet", ".pq")):
            return pd.read_parquet(path_str)
        return _read_parquet_directory(path_str)

    path = Path(path_str).expanduser().resolve()
    if path.is_dir():
        return _read_parquet_directory(str(path))

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input extension '{suffix}'. Use CSV or Parquet.")


def _compute_threshold_curve(work: pd.DataFrame, prob_col: str, area_col: str) -> pd.DataFrame:
    grouped = work.groupby(prob_col, as_index=False)[area_col].sum()
    grouped = grouped.rename(columns={prob_col: "probability_class", area_col: "area_ha"})

    classes = pd.DataFrame({"probability_class": list(range(1, 101))})
    grouped = classes.merge(grouped, on="probability_class", how="left").fillna({"area_ha": 0.0})
    grouped = grouped.sort_values("probability_class", ascending=False).reset_index(drop=True)

    grouped["area_ha"] = grouped["area_ha"].astype(float)
    grouped["threshold"] = grouped["probability_class"].astype(float) / 100.0
    grouped["area_ha"] = grouped["area_ha"].cumsum()

    return grouped[["threshold", "area_ha"]].sort_values("threshold").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args)
    output_path = resolve_output_path(args)
    df = read_table(input_path)

    required = [args.adm0_column, args.probability_column, args.area_column]
    if args.per_biome:
        required.append(args.biome_column)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available: {df.columns.tolist()}")

    keep_cols = [args.adm0_column, args.probability_column, args.area_column]
    if args.per_biome:
        keep_cols.append(args.biome_column)
    work = df[keep_cols].copy()
    work = work.dropna()
    work[args.probability_column] = work[args.probability_column].astype(int)
    work[args.area_column] = work[args.area_column].astype(float)

    if args.adm0_id is not None:
        work = work.loc[work[args.adm0_column] == int(args.adm0_id)].copy()

    work = work.loc[(work[args.probability_column] >= 1) & (work[args.probability_column] <= 100)]

    print(f"Read class-area table from: {input_path}")

    if args.per_biome:
        out_dir = output_path.parent
        out_stem = output_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        combined_parts = []
        for biome_id in sorted(work[args.biome_column].unique()):
            biome_name = BIOME_ID_TO_NAME.get(int(biome_id), f"biome_{biome_id}")
            biome_work = work.loc[work[args.biome_column] == biome_id]
            curve = _compute_threshold_curve(biome_work, args.probability_column, args.area_column)

            biome_path = out_dir / f"{out_stem}_{biome_name}.csv"
            curve.to_csv(biome_path, index=False)
            print(f"  {biome_name}: {len(curve)} thresholds -> {biome_path}")

            curve_with_biome = curve.copy()
            curve_with_biome["biome"] = biome_name
            curve_with_biome["biome_id"] = int(biome_id)
            combined_parts.append(curve_with_biome)

        combined = pd.concat(combined_parts, ignore_index=True)
        combined.to_csv(output_path, index=False)
        print(f"Wrote combined per-biome curve ({len(combined)} rows) to {output_path}")
    else:
        out = _compute_threshold_curve(work, args.probability_column, args.area_column)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        print(f"Wrote curve with {len(out)} thresholds to {output_path}")


if __name__ == "__main__":
    main()
