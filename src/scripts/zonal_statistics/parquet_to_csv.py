#!/usr/bin/env python
"""Convert a Parquet dataset to a CSV file for quick inspection."""

from __future__ import annotations

import argparse

import pandas as pd


def parquet_to_csv(parquet_path: str, csv_path: str, columns: list[str] | None = None, nrows: int | None = None) -> None:
    """Load a Parquet file or dataset and write it to CSV.

    Parameters
    ----------
    parquet_path: str
        Path or S3 URI of the source Parquet file or dataset.
    csv_path: str
        Path of the output CSV file.
    columns: list[str] | None
        Optional list of columns to read.
    nrows: int | None
        Optional number of rows to write to the CSV file.
    """
    df = pd.read_parquet(parquet_path, columns=columns, engine="pyarrow")
    if nrows is not None:
        df = df.head(nrows)
    df.to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Parquet dataset to CSV")
    parser.add_argument("--parquet", "-p", required=True, help="Path or S3 URI of the Parquet file or dataset")
    parser.add_argument("--csv", "-c", required=True, help="Destination CSV file path")
    parser.add_argument("--columns", "-C", nargs="+", help="Optional list of columns to include")
    parser.add_argument("--nrows", "-n", type=int, help="Optional number of rows to export")
    args = parser.parse_args()

    parquet_to_csv(args.parquet, args.csv, args.columns, args.nrows)


if __name__ == "__main__":
    main()

"""
python -m src.scripts.zonal_statistics.parquet_to_csv \
-p s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_6_0/zonal_stats/zonal_stats_2024/drained/2024/part-0.parquet \
-c s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_6_0/zonal_stats/zonal_stats_2024/drained/2024/drained.csv 
"""

"""
python -m src.scripts.zonal_statistics.parquet_to_csv \
-p s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_7_0/zonal_stats/ogh_standard_model/20250825/2001_2005/drained/part-0.parquet \
-c s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_7_0/zonal_stats/ogh_standard_model/20250825_keep/2001_2005/drained/drained_old.csv

"""