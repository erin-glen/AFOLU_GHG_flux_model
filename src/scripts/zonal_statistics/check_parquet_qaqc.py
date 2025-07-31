#!/usr/bin/env python
# check_parquet_qaqc.py
"""
Quick QA/QC report for Organic‑Soils or LULUCF zonal‑statistics Parquet outputs.

Features
--------
* Reads a partitioned Parquet “dataset” with pandas (pyarrow engine).
* Works for local paths and S3 URIs (needs AWS credentials via environment/IMDS).
* Optional --interval_end filter to load one or more partitions only.
* Prints:
    - file/partition count and total rows
    - column list with pandas dtypes
    - missing‑value counts and percentages
    - five‑number summary for numeric columns
    - top‑10 value counts for low‑cardinality categorical columns
    - duplicate row‑key check (gadm_adm0, drained_state_nodes, interval_end).
"""

from __future__ import annotations

import argparse
import textwrap

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_integer_dtype, is_string_dtype


def load_parquet_dataset(path: str, interval_filter: list[int] | None = None) -> pd.DataFrame:
    """Load Parquet dataset, optionally filtering on interval_end."""
    kwargs = {"engine": "pyarrow"}  # pyarrow gives predicate push‑down

    if interval_filter:
        filters = [("interval_end", "in", interval_filter)]
        df = pd.read_parquet(path, filters=filters, **kwargs)
    else:
        df = pd.read_parquet(path, **kwargs)

    return df


def numeric_summary(series: pd.Series) -> str:
    q = series.quantile([0, 0.25, 0.5, 0.75, 1])
    return (
        f"min={q.iloc[0]:.3g}, Q1={q.iloc[1]:.3g}, "
        f"median={q.iloc[2]:.3g}, Q3={q.iloc[3]:.3g}, max={q.iloc[4]:.3g}"
    )


def categorical_summary(series: pd.Series, max_levels: int = 10) -> str:
    vc = series.value_counts(dropna=False).head(max_levels)
    items = ", ".join(f"{idx} ({cnt})" for idx, cnt in vc.items())
    return f"top {len(vc)}: {items}"


def print_report(df: pd.DataFrame, show_examples: bool = True) -> None:
    n_rows, n_cols = df.shape
    print("=" * 80)
    print(f"Rows : {n_rows:,}")
    print(f"Cols : {n_cols}")
    print("-" * 80)

    # column‑wise QA
    for col in df.columns:
        ser = df[col]
        na_cnt = ser.isna().sum()
        na_pct = na_cnt / n_rows * 100
        dtype = ser.dtype

        if is_numeric_dtype(dtype):
            summary = numeric_summary(ser.dropna()) if na_cnt < n_rows else "all NA"
        elif is_string_dtype(dtype) or dtype.name == "category":
            summary = categorical_summary(ser)
        else:
            summary = "…"

        print(
            f"{col:<30} {str(dtype):<12} "
            f"NA: {na_cnt:>8} ({na_pct:4.1f} %) | {summary}"
        )

    print("-" * 80)

    # duplicate key check
    key_cols = ["gadm_adm0", "drained_state_nodes", "interval_end"]
    if all(k in df for k in key_cols):
        dups = df.duplicated(subset=key_cols).sum()
        if dups:
            print(f"‼️  Duplicate key rows: {dups:,}")
        else:
            print("No duplicate (gadm_adm0, drained_state_nodes, interval_end) keys.")
    else:
        print("Key columns missing – duplicate check skipped.")

    if show_examples:
        print("-" * 80)
        print("Example rows:")
        with pd.option_context("display.max_columns", None, "display.width", 1000):
            print(df.head(5))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QA/QC inspector for zonal‑statistics Parquet outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples
            --------
            Inspect a local dataset folder:

                python check_parquet_qaqc.py --parquet ./zonal_stats.parquet

            Only load the 2024 and 2019 partitions (fast):

                python check_parquet_qaqc.py --parquet s3://bucket/zonal_stats.parquet \\
                       --interval_end 2024 2019
            """
        ),
    )
    parser.add_argument(
        "--parquet", "-p", required=True, help="Path or S3 URI of the Parquet dataset"
    )
    parser.add_argument(
        "--interval_end",
        nargs="+",
        type=int,
        help="One or more interval_end years to filter on",
    )
    args = parser.parse_args()

    df = load_parquet_dataset(args.parquet, args.interval_end)
    print_report(df)


if __name__ == "__main__":
    main()
