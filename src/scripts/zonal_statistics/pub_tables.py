# -*- coding: utf-8 -*-
"""
Build publication-ready tables from organic-soils zonal statistics (updated for v0.7+ layout).

Outputs (long)
--------------
1) by_country_period.csv
   Columns: interval_end, gadm_adm0[, country, iso3], drained_MgCO2e, burned_MgCO2e, total_MgCO2e

2) by_drained_state_period.csv
   Columns: interval_end, drained_state_nodes, drained_state_meaning,
            drained_MgCO2e, area_ha,
            climate_domain, drained_state, emissions_state   # (auto-derived from zonal_constants)

3) by_burned_state_period.csv
   Columns: interval_end, burned_state_nodes, burned_state_meaning,
            burned_MgCO2e, area_ha,
            climate_domain, burned_state, emissions_state    # (auto-derived from zonal_constants)

# Annualized long outputs (period values duplicated to each year in the interval; inclusive)
4) by_country_annual.csv
   Columns: year, gadm_adm0[, country, iso3], drained_MgCO2e, burned_MgCO2e, total_MgCO2e

5) by_drained_state_annual.csv
   Columns: year, drained_state_nodes, drained_state_meaning,
            drained_MgCO2e, area_ha,
            climate_domain, drained_state, emissions_state   # (auto-derived)

6) by_burned_state_annual.csv
   Columns: year, burned_state_nodes, burned_state_meaning,
            burned_MgCO2e, area_ha,
            climate_domain, burned_state, emissions_state    # (auto-derived)

7) top<topn>_by_country_drained.csv
8) top<topn>_by_country_burned.csv

9) top<topn>_by_country_drained_annual.csv
10) top<topn>_by_country_burned_annual.csv

Optional wide outputs (when --wide is set)
------------------------------------------
11) by_country_period_wide_drained.csv
12) by_country_period_wide_burned.csv
13) by_country_period_wide_total.csv
   Columns: gadm_adm0[, country, iso3], <metric>_<periodEndYear> for each requested inventory period

# Annualized wide outputs (when --wide is set)
14) by_country_annual_wide_drained.csv
15) by_country_annual_wide_burned.csv
16) by_country_annual_wide_total.csv
    Columns: gadm_adm0[, country, iso3], <metric>_2001, <metric>_2002, ..., <metric>_2024

Usage
-----
python -m src.scripts.zonal_statistics.publish_tables \
  --model_version 0_7_0 \
  --run_name ogh_standard_model \
  --run_date 20250825 \
  --years 2005 2010 2015 2020 2024 \
  --aws_region us-east-1 \
  --out_dir /tmp/pub_tables \
  --topn 20 \
  --wide
# Optional: provide a prebuilt lookup instead of auto-building from ISO numeric codes
# --adm0_lookup s3://.../GADM41_adm0_lookup.csv

Notes
-----
- Annualized outputs duplicate each period’s values to every year within that period (inclusive of start/end).
- If --adm0_lookup is omitted, we auto-build a lookup (in-memory) from zc.GADM_ADM0_IDS using ISO-3166-1 → pycountry (if installed).
- Drained/burned state **context columns** are derived in-memory from zonal_constants (no external file required).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import posixpath
from typing import Optional, List, Sequence, Iterable, Tuple

import duckdb
import pandas as pd

from src.scripts.zonal_statistics.run_zonal_statistics import (
    build_output_parquet,
    build_interval_pairs,
)
from src.scripts.zonal_statistics import zonal_constants as zc  # <- for node meanings & GADM IDs


# ----------------------------- DuckDB helpers -----------------------------

def _ensure_httpfs(con: duckdb.DuckDBPyConnection, aws_region: Optional[str]):
    con.execute("INSTALL httpfs; LOAD httpfs;")
    if aws_region:
        con.execute(f"SET s3_region='{aws_region}';")


def _count_globs(con: duckdb.DuckDBPyConnection, globs: Sequence[str]) -> int:
    if not globs:
        return 0
    union_sql = " UNION ALL ".join([f"SELECT * FROM glob('{g}')" for g in globs])
    return con.execute(f"SELECT COUNT(*) FROM ({union_sql})").fetchone()[0]


def _check_nonempty_globs(con: duckdb.DuckDBPyConnection, globs: Sequence[str], label: str):
    n_files = _count_globs(con, globs)
    if n_files == 0:
        raise RuntimeError(f"[{label}] No Parquet files found for any of: {list(globs)}")


def _read_parquet_list_sql(globs: Sequence[str]) -> str:
    if not globs:
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
    Create views:
      raw_<name>   : raw read of Parquet (with filename)
      <name>       : normalized schema with interval_end, flux_type, value, gadm_adm0, and state columns
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

    filename_expr  = "filename" if "filename" in cols else "NULL"
    value_expr     = "value" if "value" in cols else "CAST(NULL AS DOUBLE)"
    flux_type_expr = "flux_type" if "flux_type" in cols else "CAST(NULL AS VARCHAR)"
    gadm_expr      = "gadm_adm0" if "gadm_adm0" in cols else "CAST(NULL AS INTEGER)"

    # Prefer embedded interval_end; fallback to parse {start}_{end} preceding '/drained/' or '/burned/'
    if "interval_end" in cols:
        interval_expr = "interval_end"
    else:
        interval_expr = f"""
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
                {interval_expr} AS interval_end,
                CAST({flux_type_expr} AS VARCHAR) AS flux_type,
                CAST({value_expr} AS DOUBLE)      AS value,
                {gadm_expr}                        AS gadm_adm0,
                {nodes_expr}                       AS drained_state_nodes,
                {meaning_expr}                     AS drained_state_meaning
            FROM raw_{name}
            WHERE {value_expr} IS NOT NULL;
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
            WHERE {value_expr} IS NOT NULL;
        """)


# --------------------------- path/glob utilities ---------------------------

def _interval_folder_strings(years: Iterable[int]) -> List[str]:
    """Return strings like '2001_2005' for requested interval end years."""
    pairs: List[Tuple[int, int]] = build_interval_pairs(list(years))
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
        bases.append(build_output_parquet(model_version, run_name, run_date, interval).rstrip("/"))
    return bases


def _make_globs_for_components(base_prefixes: Sequence[str]) -> tuple[list[str], list[str]]:
    """Return (drained_globs, burned_globs) across intervals."""
    drained = [posixpath.join(bp, "drained", "*.parquet") for bp in base_prefixes]
    burned  = [posixpath.join(bp, "burned",  "*.parquet") for bp in base_prefixes]
    return drained, burned


def _register_all(
    con: duckdb.DuckDBPyConnection,
    drained_globs: Sequence[str],
    burned_globs: Sequence[str],
    aws_region: Optional[str],
):
    _ensure_httpfs(con, aws_region)
    _check_nonempty_globs(con, drained_globs, "drained")
    _check_nonempty_globs(con, burned_globs,  "burned")
    _make_component_view(con, "zs_drained", drained_globs, kind="drained")
    _make_component_view(con, "zs_burned",  burned_globs,  kind="burned")


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
    MANUAL = {}

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

    return _auto_register_iso_lookup(con)


# ------------------------- Context from zonal_constants --------------------

def _derive_drained_context_from_zc() -> pd.DataFrame:
    """
    Build a mapping DataFrame from zc.DRAINED_STATE_NODE_MEANINGS to:
      key (padded code), meaning, climate_domain, drained_state, emissions_state
    """
    rows = []
    for key, meaning in zc.DRAINED_STATE_NODE_MEANINGS.items():
        # meaning examples:
        #  - "peat_drained_primary_infra__boreal_forest_poor"
        #  - "peat_undrained"
        #  - "non_peat"
        if "__" in meaning:
            root, emit = meaning.split("__", 1)
            # emit usually like "<domain>_<class>", e.g., "boreal_forest_poor"
            climate_domain = None
            emissions_state = None
            if "_" in emit:
                dom, rest = emit.split("_", 1)
                if dom in {"boreal", "temperate", "tropical"}:
                    climate_domain = dom
                    emissions_state = rest
                else:
                    # Unprefixed (rare): treat entire emit as emissions_state
                    emissions_state = emit
            else:
                # No underscore in emit (rare)
                emissions_state = emit
            drained_state = root
        else:
            climate_domain = None
            drained_state = meaning
            emissions_state = None

        rows.append({
            "key": str(key),
            "meaning": str(meaning),
            "climate_domain": climate_domain,
            "drained_state": drained_state,
            "emissions_state": emissions_state,
        })
    return pd.DataFrame(rows)


def _derive_burned_context_from_zc() -> pd.DataFrame:
    """
    Build a mapping DataFrame from zc.BURNED_STATE_NODE_MEANINGS to:
      key (padded code), meaning, climate_domain, burned_state, emissions_state
    """
    rows = []
    for key, meaning in zc.BURNED_STATE_NODE_MEANINGS.items():
        # meaning examples:
        #  - "boreal__drained"
        #  - "tropical__drained_crop_or_plantation"
        if "__" in meaning:
            dom, state = meaning.split("__", 1)
            climate_domain = dom
            burned_state = state
            emissions_state = state  # no finer split available; use state for emissions_state as well
        else:
            # Fallback (unexpected)
            climate_domain = None
            burned_state = meaning
            emissions_state = meaning
        rows.append({
            "key": str(key),
            "meaning": str(meaning),
            "climate_domain": climate_domain,
            "burned_state": burned_state,
            "emissions_state": emissions_state,
        })
    return pd.DataFrame(rows)


def _register_state_context_views(con: duckdb.DuckDBPyConnection):
    """
    Register in-memory context tables derived from zonal_constants:
      - drained_state_ctx(key, meaning, climate_domain, drained_state, emissions_state)
      - burned_state_ctx (key, meaning, climate_domain, burned_state,  emissions_state)
    """
    ddf = _derive_drained_context_from_zc()
    bdf = _derive_burned_context_from_zc()
    con.register("drained_state_ctx", ddf)
    con.register("burned_state_ctx", bdf)


# ------------------------- Annualization helpers --------------------------

def _build_interval_years(years: List[int]) -> pd.DataFrame:
    """
    Create a 2-col DataFrame mapping each interval_end to all years in that interval.
    Uses build_interval_pairs(), which is already used to form the interval folders.
    """
    pairs = build_interval_pairs(list(years))  # [(start, end), ...]
    rows = []
    for (start, end) in pairs:
        for y in range(int(start), int(end) + 1):
            rows.append({"interval_end": int(end), "year": int(y)})
    df = pd.DataFrame(rows).astype({"interval_end": "int32", "year": "int32"})
    return df


def _register_interval_years(con: duckdb.DuckDBPyConnection, years: List[int]) -> List[int]:
    """
    Register 'interval_years' (interval_end, year) in DuckDB and return the sorted list of all years.
    """
    df = _build_interval_years(years)
    con.register("interval_years", df)
    all_years = sorted(df["year"].unique().tolist())
    return all_years


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
    """
    Period table for drained states with context from drained_state_ctx.
    Join uses meaning first; falls back to padded node key.
    """
    return """
    WITH base AS (
      SELECT
        interval_end,
        drained_state_nodes,
        drained_state_meaning,
        SUM(CASE WHEN flux_type = 'drained_total_Mg_CO2e' THEN value ELSE 0 END) AS drained_MgCO2e,
        SUM(CASE WHEN flux_type = 'area__ha'               THEN value ELSE 0 END) AS area_ha
      FROM zs_drained
      GROUP BY 1,2,3
    )
    SELECT
      base.interval_end,
      base.drained_state_nodes,
      base.drained_state_meaning,
      base.drained_MgCO2e,
      base.area_ha,
      ctx.climate_domain,
      ctx.drained_state,
      ctx.emissions_state
    FROM base
    LEFT JOIN drained_state_ctx AS ctx
      ON (base.drained_state_meaning = ctx.meaning)
      OR (RPAD(CAST(base.drained_state_nodes AS VARCHAR), 8, '0') = ctx.key)
    ORDER BY base.interval_end, base.drained_MgCO2e DESC
    """


def table_by_burned_state_sql() -> str:
    """
    Period table for burned states with context from burned_state_ctx.
    """
    return """
    WITH base AS (
      SELECT
        interval_end,
        burned_state_nodes,
        burned_state_meaning,
        SUM(CASE WHEN flux_type = 'burned_total_Mg_CO2e' THEN value ELSE 0 END) AS burned_MgCO2e,
        SUM(CASE WHEN flux_type = 'area__ha'             THEN value ELSE 0 END) AS area_ha
      FROM zs_burned
      GROUP BY 1,2,3
    )
    SELECT
      base.interval_end,
      base.burned_state_nodes,
      base.burned_state_meaning,
      base.burned_MgCO2e,
      base.area_ha,
      ctx.climate_domain,
      ctx.burned_state,
      ctx.emissions_state
    FROM base
    LEFT JOIN burned_state_ctx AS ctx
      ON (base.burned_state_meaning = ctx.meaning)
      OR (RPAD(CAST(base.burned_state_nodes AS VARCHAR), 8, '0') = ctx.key)
    ORDER BY base.interval_end, base.burned_MgCO2e DESC
    """


def table_topn_country_sql(component: str, topn: int, with_lookup: bool) -> str:
    assert component in {"drained", "burned"}
    base = "zs_drained" if component == "drained" else "zs_burned"
    ftype = "drained_total_Mg_CO2e" if component == "drained" else "burned_total_Mg_CO2e"
    alias = "drained_MgCO2e" if component == "drained" else "burned_MgCO2e"

    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = ranked.gadm_adm0" if with_lookup else ""

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
      ranked.interval_end,
      rnk AS rank,
      ranked.gadm_adm0
      {select_l},
      {alias}
    FROM ranked
    {join_l}
    WHERE rnk <= {topn}
    ORDER BY ranked.interval_end, rank
    """


# -------- Annualized SQL (duplicate each interval's values to all years in that interval)

def table_by_country_annual_sql(with_lookup: bool) -> str:
    base = table_by_country_period_sql(with_lookup=with_lookup)
    select_l = ", base.country, base.iso3" if with_lookup else ""
    return f"""
    WITH base AS ({base})
    SELECT
      iy.year AS year,
      base.gadm_adm0
      {select_l},
      base.drained_MgCO2e,
      base.burned_MgCO2e,
      base.total_MgCO2e
    FROM base
    JOIN interval_years iy
      ON iy.interval_end = base.interval_end
    ORDER BY year, total_MgCO2e DESC
    """


def table_by_drained_state_annual_sql() -> str:
    base = table_by_drained_state_sql()
    return f"""
    WITH base AS ({base})
    SELECT
      iy.year AS year,
      base.drained_state_nodes,
      base.drained_state_meaning,
      base.drained_MgCO2e,
      base.area_ha,
      base.climate_domain,
      base.drained_state,
      base.emissions_state
    FROM base
    JOIN interval_years iy
      ON iy.interval_end = base.interval_end
    ORDER BY year, drained_MgCO2e DESC
    """


def table_by_burned_state_annual_sql() -> str:
    base = table_by_burned_state_sql()
    return f"""
    WITH base AS ({base})
    SELECT
      iy.year AS year,
      base.burned_state_nodes,
      base.burned_state_meaning,
      base.burned_MgCO2e,
      base.area_ha,
      base.climate_domain,
      base.burned_state,
      base.emissions_state
    FROM base
    JOIN interval_years iy
      ON iy.interval_end = base.interval_end
    ORDER BY year, burned_MgCO2e DESC
    """


def _wide_sql(measure_col: str, years: List[int], with_lookup: bool) -> str:
    # Wide pivot over period-end summaries (columns for each interval_end year)
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


def _wide_annual_sql(measure_col: str, years: List[int], with_lookup: bool) -> str:
    # Wide pivot over annualized by-country data (columns for each YEAR, not period-end)
    select_l = ", country, iso3" if with_lookup else ""
    group_l  = ", country, iso3" if with_lookup else ""
    cols = []
    for y in years:
        cols.append(f"SUM(CASE WHEN year = {int(y)} THEN {measure_col} ELSE 0 END) AS {measure_col}_{int(y)}")
    cols_sql = ",\n      ".join(cols)
    return f"""
    SELECT
      gadm_adm0{select_l},
      {cols_sql}
    FROM by_country_annual
    GROUP BY gadm_adm0{group_l}
    ORDER BY gadm_adm0
    """


# ---------------------------------- CLI -----------------------------------

def main(argv=None):
    p = argparse.ArgumentParser("Build publication tables from zonal stats")
    p.add_argument("--model_version", required=True)
    p.add_argument("--run_name", required=True, help="Run name used in run_zonal_statistics")
    p.add_argument("--run_date", required=True, help="Run date used in run_zonal_statistics (YYYYMMDD)")
    p.add_argument("--years", nargs="+", required=True, help="Interval end years, e.g. 2005 2010 2015 2020 2024")
    p.add_argument("--aws_region", default=None, help="AWS region for S3 (e.g., us-east-1)")
    p.add_argument("--out_dir", required=True, help="Local folder or s3://bucket/prefix/")
    p.add_argument("--topn", type=int, default=20, help="Top N countries per period")
    p.add_argument("--adm0_lookup", default=None, help="Optional CSV with columns: gadm_adm0,country,iso3")
    p.add_argument("--wide", action="store_true", help="Also write wide country tables (one column per year)")
    args = p.parse_args(argv)

    years = [int(y) for y in args.years]
    # Build list of per-interval folder names and base prefixes
    interval_folders = _interval_folder_strings(years)
    base_prefixes = _make_base_prefixes(args.model_version, args.run_name, args.run_date, interval_folders)

    drained_globs, burned_globs = _make_globs_for_components(base_prefixes)

    con = duckdb.connect()
    _register_all(con, drained_globs, burned_globs, aws_region=args.aws_region)
    have_lookup = _maybe_register_lookup(con, args.adm0_lookup)

    # Register derived state context (from zonal_constants)
    _register_state_context_views(con)

    # Annualization map
    all_years = _register_interval_years(con, years)  # e.g., [2001,...,2024]

    # Ensure output directory exists if writing locally
    if not args.out_dir.startswith("s3://"):
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # 1) By country × period (long)
    sql_country = table_by_country_period_sql(with_lookup=have_lookup)
    _copy_sql(con, sql_country, posixpath.join(args.out_dir, "by_country_period.csv"))

    # View for reuse (period-wide pivots)
    con.execute(f"CREATE OR REPLACE VIEW by_country_long AS {sql_country}")

    # 1b) By country × year (annualized long)
    sql_country_annual = table_by_country_annual_sql(with_lookup=have_lookup)
    _copy_sql(con, sql_country_annual, posixpath.join(args.out_dir, "by_country_annual.csv"))
    con.execute(f"CREATE OR REPLACE VIEW by_country_annual AS {sql_country_annual}")

    # 2) By drained state × period (with auto context)
    _copy_sql(con, table_by_drained_state_sql(),
              posixpath.join(args.out_dir, "by_drained_state_period.csv"))

    # 2b) By drained state × year (annualized + context)
    _copy_sql(con, table_by_drained_state_annual_sql(),
              posixpath.join(args.out_dir, "by_drained_state_annual.csv"))

    # 3) By burned state × period (with auto context)
    _copy_sql(con, table_by_burned_state_sql(),
              posixpath.join(args.out_dir, "by_burned_state_period.csv"))

    # 3b) By burned state × year (annualized + context)
    _copy_sql(con, table_by_burned_state_annual_sql(),
              posixpath.join(args.out_dir, "by_burned_state_annual.csv"))

    # 4) Top-N by country for drained & burned (period)
    _copy_sql(con, table_topn_country_sql("drained", args.topn, have_lookup),
              posixpath.join(args.out_dir, f"top{args.topn}_by_country_drained.csv"))
    _copy_sql(con, table_topn_country_sql("burned", args.topn, have_lookup),
              posixpath.join(args.out_dir, f"top{args.topn}_by_country_burned.csv"))

    # 5) Optional wide exports (three files for period + three for annual)
    if args.wide:
        _copy_sql(con, _wide_sql("drained_MgCO2e", years, have_lookup),
                  posixpath.join(args.out_dir, "by_country_period_wide_drained.csv"))
        _copy_sql(con, _wide_sql("burned_MgCO2e", years, have_lookup),
                  posixpath.join(args.out_dir, "by_country_period_wide_burned.csv"))
        _copy_sql(con, _wide_sql("total_MgCO2e", years, have_lookup),
                  posixpath.join(args.out_dir, "by_country_period_wide_total.csv"))

        _copy_sql(con, _wide_annual_sql("drained_MgCO2e", all_years, have_lookup),
                  posixpath.join(args.out_dir, "by_country_annual_wide_drained.csv"))
        _copy_sql(con, _wide_annual_sql("burned_MgCO2e", all_years, have_lookup),
                  posixpath.join(args.out_dir, "by_country_annual_wide_burned.csv"))
        _copy_sql(con, _wide_annual_sql("total_MgCO2e", all_years, have_lookup),
                  posixpath.join(args.out_dir, "by_country_annual_wide_total.csv"))

    print("Wrote tables to:", args.out_dir)


if __name__ == "__main__":
    main()


"""
Examples
--------

# Local CSV outputs (long + wide)
python -m src.scripts.zonal_statistics.pub_tables \
  --model_version 0_7_0 \
  --run_name ogh_standard_model \
  --run_date 20250825_keep \
  --years 2005 2010 2015 2020 2024 \
  --out_dir /mnt/c/tmp/pub_tables

# Minimal run (single year, long tables only)
python -m src.scripts.zonal_statistics.pub_tables \
  --model_version 0_7_0 \
  --run_name ogh_standard_model \
  --run_date 20250825 \
  --years 2024 \
  --out_dir /tmp/pub_tables
"""
