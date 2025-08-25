# -*- coding: utf-8 -*-
"""
Build publication-ready tables from organic-soils zonal statistics.

Outputs (long)
--------------
1) by_country_period.csv
   Columns: interval_end, gadm_adm0[, country, iso3], drained_MgCO2e, burned_MgCO2e, total_MgCO2e

2) by_drained_state_period.csv
   Columns: interval_end, drained_state_nodes, drained_state_meaning, drained_MgCO2e, area_ha

3) by_burned_state_period.csv
   Columns: interval_end, burned_state_nodes,  burned_state_meaning,  burned_MgCO2e, area_ha

Optional wide outputs (when --wide is set)
------------------------------------------
4) by_country_period_wide_drained.csv
5) by_country_period_wide_burned.csv
6) by_country_period_wide_total.csv
   Columns: gadm_adm0[, country, iso3], <metric>_<year> for each requested inventory period

Usage
-----
python -m src.scripts.zonal_statistics.publish_tables \
  --model_version 0_7_0 \
  --years 2005 2010 2015 2020 2024 \
  --aws_region us-east-1 \
  --out_dir /tmp/pub_tables \
  --topn 20 \
  --wide
# Optional: provide a prebuilt lookup instead of auto-building from ISO numeric codes
# --adm0_lookup s3://.../GADM41_adm0_lookup.csv

Notes
-----
- If --adm0_lookup is omitted, we auto-build a lookup (in-memory) from zc.GADM_ADM0_IDS
  using ISO-3166-1 numeric → pycountry (if installed). Otherwise iso3/country may be null.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import posixpath
from typing import Optional, List

import duckdb
import pandas as pd

from src.scripts.zonal_statistics.run_zonal_statistics import build_output_parquet
from src.scripts.zonal_statistics import zonal_constants as zc  # <- for GADM_ADM0_IDS


# ----------------------------- DuckDB helpers -----------------------------

def _ensure_httpfs(con: duckdb.DuckDBPyConnection, aws_region: Optional[str]):
    con.execute("INSTALL httpfs; LOAD httpfs;")
    if aws_region:
        con.execute(f"SET s3_region='{aws_region}';")


def _check_nonempty_glob(con: duckdb.DuckDBPyConnection, path_glob: str, label: str):
    n_files = con.execute(f"SELECT COUNT(*) FROM glob('{path_glob}')").fetchone()[0]
    if n_files == 0:
        raise RuntimeError(f"[{label}] No Parquet files found at: {path_glob}")


def _make_component_view(
    con: duckdb.DuckDBPyConnection, name: str, glob: str, kind: str
):
    """
    Create views:
      raw_<name>   : raw read of Parquet (with filename + hive partition cols if present)
      <name>       : normalized schema with interval_end, flux_type, value, gadm_adm0, and state columns
    """
    con.execute(f"""
        CREATE OR REPLACE VIEW raw_{name} AS
        SELECT * FROM read_parquet(
            '{glob}',
            union_by_name=true,
            filename=1,
            hive_partitioning=1
        );
    """)

    cols = {r[1] for r in con.execute(f"PRAGMA table_info('raw_{name}')").fetchall()}

    # Prefer the real column if present; else extract from Hive-style folder name.
    if "interval_end" in cols:
        interval_expr = "interval_end"
    else:
        interval_expr = "TRY_CAST(regexp_extract(filename, 'interval_end=([0-9]{4})', 1) AS INTEGER)"

    value_expr     = "value" if "value" in cols else "CAST(NULL AS DOUBLE)"
    flux_type_expr = "flux_type" if "flux_type" in cols else "CAST(NULL AS VARCHAR)"
    gadm_expr      = "gadm_adm0" if "gadm_adm0" in cols else "CAST(NULL AS INTEGER)"

    if kind == "drained":
        nodes_expr   = "drained_state_nodes"   if "drained_state_nodes"   in cols else "CAST(NULL AS INTEGER)"
        meaning_expr = "drained_state_meaning" if "drained_state_meaning" in cols else "CAST(NULL AS VARCHAR)"
        con.execute(f"""
            CREATE OR REPLACE VIEW {name} AS
            SELECT
                {interval_expr} AS interval_end,
                CAST({flux_type_expr} AS VARCHAR) AS flux_type,
                CAST({value_expr} AS DOUBLE)      AS value,
                {gadm_expr}                        AS gadm_adm0,
                {nodes_expr}                       AS drained_state_nodes,
                {meaning_expr}                     AS drained_state_meaning
            FROM raw_{name}
            WHERE {interval_expr} IS NOT NULL
              AND {value_expr}    IS NOT NULL;
        """)
    else:
        nodes_expr   = "burned_state_nodes"    if "burned_state_nodes"    in cols else "CAST(NULL AS INTEGER)"
        meaning_expr = "burned_state_meaning"  if "burned_state_meaning"  in cols else "CAST(NULL AS VARCHAR)"
        con.execute(f"""
            CREATE OR REPLACE VIEW {name} AS
            SELECT
                {interval_expr} AS interval_end,
                CAST({flux_type_expr} AS VARCHAR) AS flux_type,
                CAST({value_expr} AS DOUBLE)      AS value,
                {gadm_expr}                        AS gadm_adm0,
                {nodes_expr}                       AS burned_state_nodes,
                {meaning_expr}                     AS burned_state_meaning
            FROM raw_{name}
            WHERE {interval_expr} IS NOT NULL
              AND {value_expr}    IS NOT NULL;
        """)


def _register_all(
    con: duckdb.DuckDBPyConnection, base_prefix: str, aws_region: Optional[str]
):
    _ensure_httpfs(con, aws_region)
    drained_glob = posixpath.join(base_prefix, "drained", "**", "*.parquet")
    burned_glob  = posixpath.join(base_prefix, "burned",  "**", "*.parquet")
    _check_nonempty_glob(con, drained_glob, "drained")
    _check_nonempty_glob(con, burned_glob,  "burned")
    _make_component_view(con, "zs_drained", drained_glob, kind="drained")
    _make_component_view(con, "zs_burned",  burned_glob,  kind="burned")


# --------------------------- GADM lookup logic ----------------------------

def _auto_register_iso_lookup(con: duckdb.DuckDBPyConnection) -> bool:
    """
    Auto-build an adm0 lookup in-memory from zc.GADM_ADM0_IDS using ISO-3166-1 numeric codes.
    Uses pycountry if available; otherwise leaves iso3/country as None.

    Registers a DuckDB view named 'adm0_lookup'.
    """
    try:
        import pycountry  # optional dependency
    except Exception:
        pycountry = None

    rows: List[dict] = []
    # Manual overrides for any non-ISO numeric cases (fill if needed)
    MANUAL = {
        # Example: 383: {"iso3": "XKX", "country": "Kosovo"},
    }

    for code in zc.GADM_ADM0_IDS:
        code = int(code)
        if code == 0:
            rows.append({"gadm_adm0": 0, "iso3": None, "country": "NoData"})
            continue
        if code in MANUAL:
            rows.append({"gadm_adm0": code, **MANUAL[code]})
            continue
        iso3, name = None, None
        if pycountry is not None:
            rec = pycountry.countries.get(numeric=f"{code:03d}")
            if rec:
                iso3 = getattr(rec, "alpha_3", None)
                name = getattr(rec, "name", None)
        rows.append({"gadm_adm0": code, "iso3": iso3, "country": name})

    df = pd.DataFrame(rows).astype({"gadm_adm0": "int32"})
    con.register("adm0_lookup", df)
    return True


def _maybe_register_lookup(con: duckdb.DuckDBPyConnection, adm0_lookup: Optional[str]) -> bool:
    """
    If adm0_lookup CSV is provided, register it. Otherwise, auto-build from ISO numerics.
    Returns True if a view named 'adm0_lookup' is available.
    """
    if adm0_lookup:
        con.execute(f"""
            CREATE OR REPLACE VIEW adm0_lookup AS
            SELECT * FROM read_csv_auto('{adm0_lookup}', header=1);
        """)
        cols = {r[1] for r in con.execute("PRAGMA table_info('adm0_lookup')").fetchall()}
        needed = {"gadm_adm0", "country", "iso3"}
        missing = needed - cols
        if missing:
            raise RuntimeError(f"adm0_lookup is missing columns: {sorted(missing)}")
        return True

    # No CSV provided → build in-memory from zc.GADM_ADM0_IDS
    return _auto_register_iso_lookup(con)


# ------------------------------- SQL makers -------------------------------

def _copy_sql(con: duckdb.DuckDBPyConnection, sql: str, out_path: str):
    out_path_escaped = out_path.replace("'", "''")
    con.execute(f"COPY ({sql}) TO '{out_path_escaped}' (FORMAT CSV, HEADER TRUE)")


def table_by_country_period_sql(with_lookup: bool) -> str:
    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = f.gadm_adm0" if with_lookup else ""
    return f"""
    WITH d AS (
      SELECT interval_end, gadm_adm0, SUM(value) AS drained_MgCO2e
      FROM zs_drained
      WHERE flux_type = 'drained_total_Mg_CO2e'
      GROUP BY 1,2
    ),
    b AS (
      SELECT interval_end, gadm_adm0, SUM(value) AS burned_MgCO2e
      FROM zs_burned
      WHERE flux_type = 'burned_total_Mg_CO2e'
      GROUP BY 1,2
    ),
    f AS (
      SELECT
        COALESCE(d.interval_end, b.interval_end) AS interval_end,
        COALESCE(d.gadm_adm0,  b.gadm_adm0)     AS gadm_adm0,
        d.drained_MgCO2e,
        b.burned_MgCO2e
      FROM d FULL OUTER JOIN b
        ON d.interval_end = b.interval_end
       AND d.gadm_adm0    = b.gadm_adm0
    )
    SELECT
      f.interval_end,
      f.gadm_adm0
      {select_l},
      COALESCE(f.drained_MgCO2e, 0) AS drained_MgCO2e,
      COALESCE(f.burned_MgCO2e, 0) AS burned_MgCO2e,
      COALESCE(f.drained_MgCO2e, 0) + COALESCE(f.burned_MgCO2e, 0) AS total_MgCO2e
    FROM f
    {join_l}
    ORDER BY f.interval_end, total_MgCO2e DESC
    """


def table_by_drained_state_sql() -> str:
    return """
    SELECT
      interval_end,
      drained_state_nodes,
      drained_state_meaning,
      SUM(CASE WHEN flux_type = 'drained_total_Mg_CO2e' THEN value ELSE 0 END) AS drained_MgCO2e,
      SUM(CASE WHEN flux_type = 'area__ha'               THEN value ELSE 0 END) AS area_ha
    FROM zs_drained
    GROUP BY 1,2,3
    ORDER BY interval_end, drained_MgCO2e DESC
    """


def table_by_burned_state_sql() -> str:
    return """
    SELECT
      interval_end,
      burned_state_nodes,
      burned_state_meaning,
      SUM(CASE WHEN flux_type = 'burned_total_Mg_CO2e' THEN value ELSE 0 END) AS burned_MgCO2e,
      SUM(CASE WHEN flux_type = 'area__ha'             THEN value ELSE 0 END) AS area_ha
    FROM zs_burned
    GROUP BY 1,2,3
    ORDER BY interval_end, burned_MgCO2e DESC
    """


def table_topn_country_sql(component: str, topn: int, with_lookup: bool) -> str:
    assert component in {"drained", "burned"}
    base = "zs_drained" if component == "drained" else "zs_burned"
    ftype = "drained_total_Mg_CO2e" if component == "drained" else "burned_total_Mg_CO2e"
    alias = "drained_MgCO2e" if component == "drained" else "burned_MgCO2e"
    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = t.gadm_adm0" if with_lookup else ""
    return f"""
    WITH t AS (
      SELECT interval_end, gadm_adm0, SUM(value) AS {alias}
      FROM {base}
      WHERE flux_type = '{ftype}'
      GROUP BY 1,2
    ),
    ranked AS (
      SELECT
        interval_end,
        gadm_adm0,
        {alias},
        ROW_NUMBER() OVER (PARTITION BY interval_end ORDER BY {alias} DESC) AS rnk
      FROM t
    )
    SELECT
      interval_end,
      rnk AS rank,
      gadm_adm0
      {select_l},
      {alias}
    FROM ranked
    {join_l}
    WHERE rnk <= {topn}
    ORDER BY interval_end, rank
    """


def _wide_sql(measure_col: str, years: List[int], with_lookup: bool) -> str:
    """Build conditional-aggregation wide table SQL from the long by-country view."""
    select_l = ", country, iso3" if with_lookup else ""
    group_l  = ", country, iso3" if with_lookup else ""
    cols = []
    for y in years:
        cols.append(f"SUM(CASE WHEN interval_end = {int(y)} THEN {measure_col} ELSE 0 END) AS {measure_col}_{int(y)}")
    cols_sql = ",\n      ".join(cols)
    return f"""
    SELECT
      gadm_adm0{select_l},
      {cols_sql}
    FROM by_country_long
    GROUP BY gadm_adm0{group_l}
    ORDER BY gadm_adm0
    """


# ---------------------------------- CLI -----------------------------------

def main(argv=None):
    p = argparse.ArgumentParser("Build publication tables from zonal stats")
    p.add_argument("--model_version", required=True)
    p.add_argument("--years", nargs="+", required=True, help="e.g. 2005 2010 2015 2020 2024")
    p.add_argument("--aws_region", default=None, help="AWS region for S3 (e.g., us-east-1)")
    p.add_argument("--out_dir", required=True, help="Local folder or s3://bucket/prefix/")
    p.add_argument("--topn", type=int, default=20, help="Top N countries per period")
    p.add_argument("--adm0_lookup", default=None, help="Optional CSV with columns: gadm_adm0,country,iso3")
    p.add_argument("--wide", action="store_true", help="Also write wide country tables (one column per year)")
    args = p.parse_args(argv)

    years = [int(y) for y in args.years]
    base_prefix = build_output_parquet(args.model_version, years).rstrip("/")

    con = duckdb.connect()
    _register_all(con, base_prefix, aws_region=args.aws_region)
    have_lookup = _maybe_register_lookup(con, args.adm0_lookup)

    # Ensure output directory exists if writing locally
    if not args.out_dir.startswith("s3://"):
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # 1) By country × period (long)
    sql_country = table_by_country_period_sql(with_lookup=have_lookup)
    _copy_sql(con, sql_country, posixpath.join(args.out_dir, "by_country_period.csv"))

    # Create a view for reuse (for wide pivots)
    con.execute(f"CREATE OR REPLACE VIEW by_country_long AS {sql_country}")

    # 2) By drained state × period
    _copy_sql(con, table_by_drained_state_sql(), posixpath.join(args.out_dir, "by_drained_state_period.csv"))

    # 3) By burned state × period
    _copy_sql(con, table_by_burned_state_sql(), posixpath.join(args.out_dir, "by_burned_state_period.csv"))

    # 4) Top-N by country for drained & burned
    _copy_sql(con, table_topn_country_sql("drained", args.topn, have_lookup),
              posixpath.join(args.out_dir, f"top{args.topn}_by_country_drained.csv"))
    _copy_sql(con, table_topn_country_sql("burned", args.topn, have_lookup),
              posixpath.join(args.out_dir, f"top{args.topn}_by_country_burned.csv"))

    # 5) Optional wide exports (three files)
    if args.wide:
        _copy_sql(con, _wide_sql("drained_MgCO2e", years, have_lookup),
                  posixpath.join(args.out_dir, "by_country_period_wide_drained.csv"))
        _copy_sql(con, _wide_sql("burned_MgCO2e", years, have_lookup),
                  posixpath.join(args.out_dir, "by_country_period_wide_burned.csv"))
        _copy_sql(con, _wide_sql("total_MgCO2e", years, have_lookup),
                  posixpath.join(args.out_dir, "by_country_period_wide_total.csv"))

    print("Wrote tables to:", args.out_dir)


if __name__ == "__main__":
    main()


"""
Examples
--------

# Local CSV outputs (long + wide)
python -m src.scripts.zonal_statistics.publish_tables \
  --model_version 0_7_0 \
  --years 2005 2010 2015 2020 2024 \
  --aws_region us-east-1 \
  --out_dir /tmp/pub_tables \
  --topn 20 \
  --wide

# Write CSVs directly to S3 (long + wide), with explicit adm0 lookup CSV
python -m src.scripts.zonal_statistics.publish_tables \
  --model_version 0_7_0 \
  --years 2005 2010 2015 2020 2024 \
  --aws_region us-east-1 \
  --out_dir s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_7_0/zonal_stats/pubs/ \
  --topn 20 \
  --adm0_lookup s3://gfw2-data/.../GADM41_adm0_lookup.csv \
  --wide

# Minimal run (single year, long tables only)
python -m src.scripts.zonal_statistics.publish_tables \
  --model_version 0_7_0 \
  --years 2024 \
  --out_dir /tmp/pub_tables
"""
