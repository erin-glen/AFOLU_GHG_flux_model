# -*- coding: utf-8 -*-
"""
Explore organic-soils zonal stats on S3 with DuckDB (updated for per-interval layout).

New layout written by run_zonal_statistics (v0.7+):
s3://.../version_{model_version}/zonal_stats/{run_name}/{run_date}/{interval}/{drained|burned}/part-*.parquet

Usage:
python -m src.scripts.zonal_statistics.explore_zonal_stats \
  --model_version 0_7_0 \
  --run_name ogh_standard_model \
  --run_date 20250825 \
  --years 2010 2015 2020 2024 \
  --component drained \
  --out_csv /tmp/drained_summary.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import posixpath
from typing import Iterable, List, Sequence, Union

import duckdb

from src.scripts.zonal_statistics.run_zonal_statistics import (
    build_output_parquet,
    build_interval_pairs,
)


# --------------------------- helpers: DuckDB / S3 ---------------------------

def _ensure_httpfs(con: duckdb.DuckDBPyConnection, aws_region: str | None):
    con.execute("INSTALL httpfs; LOAD httpfs;")
    if aws_region:
        con.execute(f"SET s3_region='{aws_region}';")


def _count_globs(con: duckdb.DuckDBPyConnection, globs: Sequence[str]) -> int:
    """Return total number of files that match any of the globs (DuckDB side)."""
    if not globs:
        return 0
    # Use DuckDB glob() then UNION ALL to count across multiple patterns
    union_sql = " UNION ALL ".join([f"SELECT * FROM glob('{g}')" for g in globs])
    return con.execute(f"SELECT COUNT(*) FROM ({union_sql})").fetchone()[0]


def _check_nonempty_globs(con: duckdb.DuckDBPyConnection, globs: Sequence[str], label: str):
    n = _count_globs(con, globs)
    if n == 0:
        raise RuntimeError(f"No Parquet files found for {label}. Checked patterns: {globs}")


def _read_parquet_list_sql(globs: Sequence[str]) -> str:
    """Return SQL literal for DuckDB read_parquet([...]) list."""
    if not globs:
        # Still return a syntactically valid empty list; caller should have validated non-empty
        return "[]"
    items = ", ".join([f"'{g}'" for g in globs])
    return f"[{items}]"


# --------------------------- view construction ---------------------------

def _make_component_view(
    con: duckdb.DuckDBPyConnection,
    name: str,
    globs: Sequence[str],
    kind: str,
):
    """
    Create two views:
      - raw_{name}: direct read of the parquet files with filename included
      - {name}:     normalized/compatible projection with guaranteed columns

    Columns guaranteed in final view:
      interval_end (INTEGER), flux_type (VARCHAR), value (DOUBLE), gadm_adm0 (INTEGER),
      {drained|burned}_state_nodes (INTEGER), {drained|burned}_state_meaning (VARCHAR)
    """
    rp_list = _read_parquet_list_sql(globs)
    con.execute(f"""
        CREATE OR REPLACE VIEW raw_{name} AS
        SELECT * FROM read_parquet(
            {rp_list},
            union_by_name=true,
            filename=1
        );
    """)

    cols = {r[1] for r in con.execute(f"PRAGMA table_info('raw_{name}')").fetchall()}

    filename_expr   = "filename" if "filename" in cols else "NULL"
    value_expr      = "value" if "value" in cols else "CAST(NULL AS DOUBLE)"
    flux_type_expr  = "flux_type" if "flux_type" in cols else "CAST(NULL AS VARCHAR)"
    gadm_expr       = "gadm_adm0" if "gadm_adm0" in cols else "CAST(NULL AS INTEGER)"

    # Prefer new, explicit interval_end column; otherwise parse from the path:
    # .../zonal_stats/{run_name}/{run_date}/{start}_{end}/{drained|burned}/part-*.parquet
    if "interval_end" in cols:
        interval_end_expr = "interval_end"
    else:
        # Extract the _second_ year from "{start}_{end}" just before component folder.
        # Pattern captures two 4-digit groups prior to '/drained/' or '/burned/'.
        # We then cast the second capture as INTEGER.
        interval_end_expr = f"""
            TRY_CAST(
                regexp_extract({filename_expr}, '([0-9]{{4}})_([0-9]{{4}})/(?:drained|burned)/', 2)
            AS INTEGER)
        """

    if kind == "drained":
        nodes_expr   = "drained_state_nodes"   if "drained_state_nodes"   in cols else "CAST(NULL AS INTEGER)"
        meaning_expr = "drained_state_meaning" if "drained_state_meaning" in cols else "CAST(NULL AS VARCHAR)"
        con.execute(f"""
            CREATE OR REPLACE VIEW {name} AS
            SELECT
                {interval_end_expr} AS interval_end,
                {flux_type_expr}    AS flux_type,
                {value_expr}        AS value,
                {gadm_expr}         AS gadm_adm0,
                {nodes_expr}        AS drained_state_nodes,
                {meaning_expr}      AS drained_state_meaning
            FROM raw_{name}
            WHERE {value_expr} IS NOT NULL;
        """)
    else:
        nodes_expr   = "burned_state_nodes"    if "burned_state_nodes"    in cols else "CAST(NULL AS INTEGER)"
        meaning_expr = "burned_state_meaning"  if "burned_state_meaning"  in cols else "CAST(NULL AS VARCHAR)"
        con.execute(f"""
            CREATE OR REPLACE VIEW {name} AS
            SELECT
                {interval_end_expr} AS interval_end,
                {flux_type_expr}    AS flux_type,
                {value_expr}        AS value,
                {gadm_expr}         AS gadm_adm0,
                {nodes_expr}        AS burned_state_nodes,
                {meaning_expr}      AS burned_state_meaning
            FROM raw_{name}
            WHERE {value_expr} IS NOT NULL;
        """)


# --------------------------- S3 path construction ---------------------------

def _interval_folder_strings(years: Iterable[int]) -> List[str]:
    """Return strings like '2001_2005' for requested interval end-years."""
    pairs = build_interval_pairs(list(years))  # -> List[(start,end)]
    return [f"{s}_{e}" for (s, e) in pairs]


def _make_base_prefixes(
    model_version: str,
    run_name: str,
    run_date: str,
    interval_folders: Sequence[str],
) -> List[str]:
    """Return the per-interval base prefixes under which drained/burned live."""
    bases: List[str] = []
    for interval in interval_folders:
        bases.append(build_output_parquet(model_version, run_name, run_date, interval))
    return bases


def _make_globs_for_components(
    base_prefixes: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Return (drained_globs, burned_globs) lists across intervals."""
    drained = [posixpath.join(bp, "drained", "*.parquet") for bp in base_prefixes]
    burned  = [posixpath.join(bp, "burned",  "*.parquet") for bp in base_prefixes]
    return drained, burned


# --------------------------- public: connect & query ---------------------------

def connect_and_register(
    base_prefixes: Union[str, Sequence[str]],
    *,
    aws_region: str | None = None
) -> duckdb.DuckDBPyConnection:
    """
    Connect and register views for both components.

    base_prefixes: either a single base prefix (string) or a list of per-interval
                   base prefixes of the form:
                   s3://.../version_{model_version}/zonal_stats/{run_name}/{run_date}/{interval}/
    """
    # Normalize to list
    if isinstance(base_prefixes, str):
        bases: List[str] = [base_prefixes]
    else:
        bases = list(base_prefixes)

    drained_globs, burned_globs = _make_globs_for_components(bases)

    con = duckdb.connect()
    _ensure_httpfs(con, aws_region)

    _check_nonempty_globs(con, drained_globs, "drained")
    _check_nonempty_globs(con, burned_globs,  "burned")

    _make_component_view(con, "zs_drained", drained_globs, kind="drained")
    _make_component_view(con, "zs_burned",  burned_globs,  kind="burned")
    return con


def quick_totals(con: duckdb.DuckDBPyConnection, component: str = "drained"):
    table = "zs_drained" if component == "drained" else "zs_burned"
    return con.execute(f"""
        SELECT interval_end, flux_type, SUM(value) AS sum_value
        FROM {table}
        GROUP BY 1,2
        ORDER BY 1,2
    """).df()


def export_summary(con: duckdb.DuckDBPyConnection, out_csv: str, component: str = "drained"):
    table = "zs_drained" if component == "drained" else "zs_burned"
    df = con.execute(f"""
        SELECT interval_end, gadm_adm0, flux_type, SUM(value) AS value
        FROM {table}
        GROUP BY 1,2,3
        ORDER BY 1,2,3
    """).df()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


# --------------------------- CLI ---------------------------

def main(argv=None):
    p = argparse.ArgumentParser("Explore organic-soils zonal stats on S3 with DuckDB")
    p.add_argument("--model_version", required=True, help="e.g., 0_7_0")
    p.add_argument("--run_name", required=True, help="Run name used by run_zonal_statistics")
    p.add_argument("--run_date", required=True, help="Run date used by run_zonal_statistics (YYYYMMDD)")
    p.add_argument("--years", nargs="+", required=True, help="Interval end years, e.g. 2010 2015 2020 2024")
    p.add_argument("--component", choices=["drained","burned"], default="drained")
    p.add_argument("--aws_region", default=None, help="AWS region for S3 (e.g., us-east-1)")
    p.add_argument("--out_csv", default=None, help="Optional: write grouped summary to CSV")
    args = p.parse_args(argv)

    # Build per-interval base prefixes
    interval_folders = _interval_folder_strings([int(y) for y in args.years])
    base_prefixes = _make_base_prefixes(args.model_version, args.run_name, args.run_date, interval_folders)

    con = connect_and_register(base_prefixes, aws_region=args.aws_region)

    print(quick_totals(con, component=args.component).head())

    if args.out_csv:
        path = export_summary(con, args.out_csv, component=args.component)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
