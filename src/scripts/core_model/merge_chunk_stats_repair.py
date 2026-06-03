"""
Merge tile-repair chunk statistics into an existing full-run workbook.

The core model and 10x10 aggregation stages write timestamped chunk-stat
workbooks. A tile-level repair should not leave a partial workbook as the
implicit source of truth. This helper removes rows for repaired tile/year
combinations from the original workbook, appends the replacement rows, and
recomputes summary sheets when the source workbook has the standard organic
soils 1x1 chunk-stat layout.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import pandas as pd


SUMMARY_SHEETS = {"min_max_summary", "pixel_counts_summary"}


def _split_cli_items(values: list[str] | None) -> list[str]:
    if not values:
        return []
    items: list[str] = []
    for value in values:
        items.extend(
            item.strip()
            for item in str(value).replace(",", " ").split()
            if item.strip()
        )
    return list(dict.fromkeys(items))


def _read_excel(path: str) -> OrderedDict[str, pd.DataFrame]:
    if path.startswith("s3://"):
        import fsspec

        with fsspec.open(path, "rb") as fh:
            sheets = pd.read_excel(fh, sheet_name=None)
    else:
        sheets = pd.read_excel(path, sheet_name=None)
    return OrderedDict((name, df) for name, df in sheets.items())


def _normal_values(values: Iterable[str]) -> set[str]:
    return {str(value).strip() for value in values}


def _row_selection(
    df: pd.DataFrame,
    tile_ids: set[str],
    years: set[str],
) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if "tile_id" in df.columns and tile_ids:
        mask &= df["tile_id"].astype(str).str.strip().isin(tile_ids)
    if "years" in df.columns and years:
        mask &= df["years"].astype(str).str.strip().isin(years)
    return mask


def _align_columns(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(left.columns)
    columns.extend(column for column in right.columns if column not in columns)
    return left.reindex(columns=columns), right.reindex(columns=columns)


def _merge_sheet(
    base_df: pd.DataFrame,
    repair_df: pd.DataFrame,
    tile_ids: set[str],
    years: set[str],
) -> pd.DataFrame:
    if "tile_id" not in base_df.columns and "tile_id" not in repair_df.columns:
        return base_df.copy()

    base_mask = _row_selection(base_df, tile_ids, years)
    repair_mask = _row_selection(repair_df, tile_ids, years)
    base_keep = base_df.loc[~base_mask].copy()
    repair_keep = repair_df.loc[repair_mask].copy()
    base_keep, repair_keep = _align_columns(base_keep, repair_keep)
    return pd.concat([base_keep, repair_keep], ignore_index=True)


def _collect_detail_rows(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    detail_frames = []
    required = {"layer_name", "min_value", "max_value"}
    for name, df in sheets.items():
        if name in SUMMARY_SHEETS:
            continue
        if required <= set(df.columns):
            detail_frames.append(df.copy())
    if not detail_frames:
        return pd.DataFrame()
    return pd.concat(detail_frames, ignore_index=True)


def _recompute_min_max_summary(detail_rows: pd.DataFrame) -> pd.DataFrame:
    if detail_rows.empty:
        return pd.DataFrame(columns=["layer_name", "min_value", "max_value", "count"])
    df = detail_rows.copy()
    df["min_value"] = pd.to_numeric(df["min_value"], errors="coerce")
    df["max_value"] = pd.to_numeric(df["max_value"], errors="coerce")
    return (
        df.groupby("layer_name", dropna=False)
        .agg(
            min_value=("min_value", "min"),
            max_value=("max_value", "max"),
            count=("layer_name", "count"),
        )
        .reset_index()
    )


def _recompute_pixel_counts_summary(detail_rows: pd.DataFrame) -> pd.DataFrame:
    columns = ["tile_id", "layer_name", "total_pixel_count", "tile_layer"]
    if detail_rows.empty or not {"tile_id", "layer_name", "count_value"} <= set(detail_rows.columns):
        return pd.DataFrame(columns=columns)
    df = detail_rows.copy()
    if "in_out" in df.columns:
        df = df[df["in_out"].astype(str) == "output_layer"].copy()
    df["count_value"] = pd.to_numeric(df["count_value"], errors="coerce").fillna(0)
    pixel_counts = (
        df.groupby(["tile_id", "layer_name"], dropna=False)["count_value"]
        .sum()
        .reset_index()
        .rename(columns={"count_value": "total_pixel_count"})
    )
    pixel_counts["tile_layer"] = (
        pixel_counts["tile_id"].astype(str)
        + "__"
        + pixel_counts["layer_name"].astype(str)
        + ".tif"
    )
    return pixel_counts[columns]


def merge_chunk_stats_workbooks(
    base_path: str,
    repair_paths: list[str],
    output_path: str,
    tile_ids: Iterable[str],
    years: Iterable[str],
) -> str:
    """Merge one or more repair workbooks into a full chunk-stat workbook."""

    tile_set = _normal_values(tile_ids)
    year_set = _normal_values(years)
    merged = _read_excel(base_path)

    for repair_path in repair_paths:
        repair = _read_excel(repair_path)
        for sheet_name, repair_df in repair.items():
            if sheet_name in SUMMARY_SHEETS:
                continue
            base_df = merged.get(sheet_name)
            if base_df is None:
                merged[sheet_name] = repair_df.loc[
                    _row_selection(repair_df, tile_set, year_set)
                ].copy()
            else:
                merged[sheet_name] = _merge_sheet(
                    base_df,
                    repair_df,
                    tile_set,
                    year_set,
                )

    detail_rows = _collect_detail_rows(merged)
    if "min_max_summary" in merged:
        merged["min_max_summary"] = _recompute_min_max_summary(detail_rows)
    if "pixel_counts_summary" in merged:
        merged["pixel_counts_summary"] = _recompute_pixel_counts_summary(detail_rows)

    local_output = _write_workbook(merged, output_path)
    return local_output


def _write_workbook(sheets: dict[str, pd.DataFrame], output_path: str) -> str:
    if output_path.startswith("s3://"):
        local_path = os.path.join(
            tempfile.gettempdir(),
            posixpath.basename(output_path.rstrip("/")),
        )
    else:
        local_path = output_path
        parent = Path(local_path).parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(local_path) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    if output_path.startswith("s3://"):
        _upload_to_s3(local_path, output_path)
    return local_path


def _upload_to_s3(local_path: str, output_path: str) -> None:
    import boto3

    no_scheme = output_path.removeprefix("s3://")
    bucket, key = no_scheme.split("/", 1)
    boto3.client("s3").upload_file(local_path, bucket, key)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Merge repaired tile chunk-stat rows into a full workbook."
    )
    parser.add_argument("--base", required=True, help="Original full chunk-stat workbook")
    parser.add_argument(
        "--repair",
        nargs="+",
        required=True,
        help="One or more repair workbooks with replacement rows",
    )
    parser.add_argument("--output", required=True, help="Merged output workbook path")
    parser.add_argument(
        "--tile_ids",
        nargs="+",
        required=True,
        help="Repaired tile IDs. Supports spaces or commas.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        required=True,
        help="Repaired year or period labels, for example 2021_2024.",
    )
    args = parser.parse_args(argv)

    output = merge_chunk_stats_workbooks(
        base_path=args.base,
        repair_paths=args.repair,
        output_path=args.output,
        tile_ids=_split_cli_items(args.tile_ids),
        years=_split_cli_items(args.years),
    )
    print(f"Merged chunk stats written to {output}")


if __name__ == "__main__":
    main()
