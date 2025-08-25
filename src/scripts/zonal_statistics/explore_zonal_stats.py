# -*- coding: utf-8 -*-
"""
Explore organic-soils zonal stats on S3 with DuckDB.

Usage:
python -m src.scripts.zonal_statistics.explore_zonal_stats \
  --model_version 0_6_0 \
  --years 2005 2010 2015 2020 2024 \
  --component drained \
  --out_csv /tmp/drained_summary.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import duckdb

BASE_S3 = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"

def build_prefix(model_version: str, years: list[str]) -> str:
    years_part = "_".join(str(y) for y in years)
    return f"{BASE_S3}/version_{model_version}/zonal_stats/zonal_stats_{years_part}"

def _ensure_httpfs(con: duckdb.DuckDBPyConnection, aws_region: str | None):
    con.execute("INSTALL httpfs; LOAD httpfs;")
    if aws_region:
        con.execute(f"SET s3_region='{aws_region}';")

def _check_nonempty_glob(con: duckdb.DuckDBPyConnection, path_glob: str):
    n_files = con.execute(f"SELECT COUNT(*) FROM glob('{path_glob}')").fetchone()[0]
    if n_files == 0:
        raise RuntimeError(f"No Parquet files found at: {path_glob}")

def _make_component_view(con: duckdb.DuckDBPyConnection, name: str, glob: str, kind: str):
    # read raw with filename; don't rely on hive partition columns
    con.execute(f"""
        CREATE OR REPLACE VIEW raw_{name} AS
        SELECT * FROM read_parquet(
            '{glob}',
            union_by_name=true,
            filename=1
        );
    """)
    cols = {r[1] for r in con.execute(f"PRAGMA table_info('raw_{name}')").fetchall()}

    filename_expr  = "filename" if "filename" in cols else "NULL"
    value_expr     = "value" if "value" in cols else "CAST(NULL AS DOUBLE)"
    flux_type_expr = "flux_type" if "flux_type" in cols else "CAST(NULL AS VARCHAR)"
    gadm_expr      = "gadm_adm0" if "gadm_adm0" in cols else "CAST(NULL AS INTEGER)"

    # derive interval_end from folder name: /drained/YYYY/ or /burned/YYYY/
    if kind == "drained":
        year_regex = f"'/drained/([0-9]{{4}})/'"
        nodes_expr   = "drained_state_nodes"   if "drained_state_nodes"   in cols else "CAST(NULL AS INTEGER)"
        meaning_expr = "drained_state_meaning" if "drained_state_meaning" in cols else "CAST(NULL AS VARCHAR)"
        con.execute(f"""
            CREATE OR REPLACE VIEW {name} AS
            SELECT
                TRY_CAST(regexp_extract({filename_expr}, {year_regex}, 1) AS INTEGER) AS interval_end,
                {flux_type_expr} AS flux_type,
                {value_expr}     AS value,
                {gadm_expr}      AS gadm_adm0,
                {nodes_expr}     AS drained_state_nodes,
                {meaning_expr}   AS drained_state_meaning
            FROM raw_{name}
            WHERE {filename_expr} IS NOT NULL
              AND regexp_extract({filename_expr}, {year_regex}, 1) <> ''
              AND {value_expr} IS NOT NULL;
        """)
    else:
        year_regex = f"'/burned/([0-9]{{4}})/'"
        nodes_expr   = "burned_state_nodes"    if "burned_state_nodes"    in cols else "CAST(NULL AS INTEGER)"
        meaning_expr = "burned_state_meaning"  if "burned_state_meaning"  in cols else "CAST(NULL AS VARCHAR)"
        con.execute(f"""
            CREATE OR REPLACE VIEW {name} AS
            SELECT
                TRY_CAST(regexp_extract({filename_expr}, {year_regex}, 1) AS INTEGER) AS interval_end,
                {flux_type_expr} AS flux_type,
                {value_expr}     AS value,
                {gadm_expr}      AS gadm_adm0,
                {nodes_expr}     AS burned_state_nodes,
                {meaning_expr}   AS burned_state_meaning
            FROM raw_{name}
            WHERE {filename_expr} IS NOT NULL
              AND regexp_extract({filename_expr}, {year_regex}, 1) <> ''
              AND {value_expr} IS NOT NULL;
        """)

def connect_and_register(base_prefix: str, *, aws_region: str | None = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    _ensure_httpfs(con, aws_region)

    drained_glob = f"{base_prefix}/drained/**/*.parquet"
    burned_glob  = f"{base_prefix}/burned/**/*.parquet"
    _check_nonempty_glob(con, drained_glob)
    _check_nonempty_glob(con, burned_glob)

    _make_component_view(con, "zs_drained", drained_glob, kind="drained")
    _make_component_view(con, "zs_burned",  burned_glob,  kind="burned")
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

def main(argv=None):
    p = argparse.ArgumentParser("Explore organic-soils zonal stats on S3 with DuckDB")
    p.add_argument("--model_version", required=True)
    p.add_argument("--years", nargs="+", required=True, help="e.g. 2005 2010 2015 2020 2024")
    p.add_argument("--component", choices=["drained","burned"], default="drained")
    p.add_argument("--aws_region", default=None, help="AWS region for S3 (e.g., us-east-1)")
    p.add_argument("--out_csv", default=None)
    args = p.parse_args(argv)

    prefix = build_prefix(args.model_version, args.years)
    con = connect_and_register(prefix, aws_region=args.aws_region)

    print(quick_totals(con, component=args.component).head())

    if args.out_csv:
        path = export_summary(con, args.out_csv, component=args.component)
        print(f"Wrote {path}")

if __name__ == "__main__":
    main()

"""
python -m src.scripts.zonal_statistics.explore_zonal_stats \
  --model_version 0_7_0 \
  --years 2020 2024 \
  --component drained \
  --out_csv /mnt/c/tmp/drained_summary.csv
"""
