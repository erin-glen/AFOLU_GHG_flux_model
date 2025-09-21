# -*- coding: utf-8 -*-
"""
Build publication **tables** and **figures** from organic-soils zonal statistics (one script).

Output root (hardcoded)
-----------------------
All assets are written under: **/mnt/c/tmp/pub_assets**
(Edit the OUT_DIR constant below if you need a different location. If you set OUT_DIR
to an s3:// path, tables/CSVs will write via DuckDB httpfs; PNG/SVG will upload via boto3.)

What this writes (under /mnt/c/tmp/pub_assets)
----------------------------------------------
Tables (CSV via DuckDB COPY):
  - by_country_period.csv
  - by_drained_state_period.csv
  - by_burned_state_period.csv
  - by_country_drained_state_period.csv
  - by_country_burned_state_period.csv
  - top{N}_by_country_{drained|burned}.csv

Figures + Flourish-ready datasets:
  - figures/global_drained_climate_column.{png,svg}
  - figures/data/global_drained_climate_{long,wide}.csv
  - figures/global_burn_climate_column.{png,svg}
  - figures/data/global_burn_climate_{long,wide}.csv
  - figures/drained_landuse_climate_bar.{png,svg}
  - figures/data/drained_landuse_climate_{long,wide}.csv
  - figures/burned_landuse_climate_bar.{png,svg}
  - figures/data/burned_landuse_climate_{long,wide}.csv
  - figures/top_10_country_peat_area_bar.{png,svg}
  - figures/data/top_10_country_peat_area.csv
  - figures/top_10_country_total_emissions_bar.{png,svg}
  - figures/data/top_10_country_total_emissions.csv

Notes
-----
- Uses the same DuckDB registrations/views as your tables workflow to avoid drift.
- Converts Mg → Gt by dividing by 1e9 for figure datasets.
- If OUT_DIR starts with s3://, tables & CSVs write directly via DuckDB httpfs.
  For PNG/SVG to s3://, this script uses boto3. (Install boto3 or set OUT_DIR to a local path.)

Examples (Linux/WSL bash)
-------------------------
cd /mnt/c/gis/git/AFOLU_GHG_flux_model
export PYTHONPATH=/mnt/c/gis/git/AFOLU_GHG_flux_model

# Tables + figures
python -m src.scripts.zonal_statistics.pub_assets \
  --model_version 0_7_0 \
  --run_name ogh_standard_model \
  --run_date 20250825 \
  --years 2005 2010 2015 2020 2024

# Only tables
python -m src.scripts.zonal_statistics.pub_assets \
  --model_version 0_7_5 \
  --run_name ogh_standard_model \
  --run_date 20250914 \
  --years 2024 \
  --do-figures 0

# Only figures (PNG only, no SVG)
python -m src.scripts.zonal_statistics.pub_assets \
  --model_version 0_7_5 \
  --run_name ogh_standard_model \
  --run_date 20250914 \
  --years 2005 2010 2015 2020 2024 \
  --do-tables 0 \
  --no-svg
"""
from __future__ import annotations

import argparse
import os
import posixpath
import re
import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import duckdb
import pandas as pd
import matplotlib.pyplot as plt

from src.scripts.zonal_statistics.run_zonal_stats import (  # existing helpers
    build_output_parquet,
    build_interval_pairs,
)
from src.scripts.zonal_statistics import zonal_constants as zc  # node meanings & GADM IDs

# ----------------------------- config -----------------------------
OUT_DIR = "/mnt/c/tmp/pub_assets"  # hardcoded output root

# ----------------------------- path helpers -----------------------------

def _is_s3(path: str) -> bool:
    return str(path).startswith("s3://")

def _join(base: str, *parts: str) -> str:
    if _is_s3(base):
        return posixpath.join(base, *parts)
    return os.path.join(base, *parts)

def _ensure_parent_dir_local(path: str):
    if _is_s3(path):
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def _to_duckdb_path(path: str) -> str:
    if _is_s3(path):
        return path
    return Path(path).as_posix()

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

def _make_component_view(
    con: duckdb.DuckDBPyConnection,
    name: str,
    globs: Sequence[str],
    kind: str,
):
    """
    Create views:
      raw_<name> : raw read of Parquet (with filename)
      <name>     : normalized schema with interval_end, flux_type, value, gadm_adm0, and state columns
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
        nodes_expr   = "burned_state_nodes"   if "burned_state_nodes"   in cols else "CAST(NULL AS INTEGER)"
        meaning_expr = "burned_state_meaning" if "burned_state_meaning" in cols else "CAST(NULL AS VARCHAR)"
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

def _interval_folder_strings(years: Iterable[int]) -> List[str]:
    pairs: List[Tuple[int, int]] = build_interval_pairs(list(years))
    return [f"{s}_{e}" for (s, e) in pairs]

def _make_base_prefixes(model_version: str, run_name: str, run_date: str,
                        interval_folders: Iterable[str]) -> List[str]:
    bases: List[str] = []
    for interval in interval_folders:
        bases.append(build_output_parquet(model_version, run_name, run_date, interval).rstrip("/"))
    return bases

def _make_globs_for_components(base_prefixes: Sequence[str]) -> tuple[list[str], list[str]]:
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
    try:
        import pycountry  # optional
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

def _make_period_labels_from_years(end_years: Sequence[int]) -> dict[int, str]:
    pairs = build_interval_pairs(list(end_years))  # [(start, end), ...]
    return {e: f"{s}-{e}" for (s, e) in pairs}

# ------------------------- Context from zonal_constants --------------------

def _derive_drained_context_from_zc() -> pd.DataFrame:
    rows = []
    for key, meaning in zc.DRAINED_STATE_NODE_MEANINGS.items():
        if "__" in meaning:
            root, emit = meaning.split("__", 1)
            climate_domain = None
            emissions_state = None
            if "_" in emit:
                dom, rest = emit.split("_", 1)
                if dom in {"boreal", "temperate", "tropical"}:
                    climate_domain = dom
                    emissions_state = rest
                else:
                    emissions_state = emit
            else:
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
    rows = []
    for key, meaning in zc.BURNED_STATE_NODE_MEANINGS.items():
        if "__" in meaning:
            dom, state = meaning.split("__", 1)
            climate_domain = dom
            burned_state = state
            emissions_state = state
        else:
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
    con.register("drained_state_ctx", _derive_drained_context_from_zc())
    con.register("burned_state_ctx",  _derive_burned_context_from_zc())

# ------------------------------- Tables: SQL -------------------------------

def _copy_sql(con: duckdb.DuckDBPyConnection, sql: str, out_path: str):
    _ensure_parent_dir_local(out_path)
    out_path_escaped = _to_duckdb_path(path=out_path).replace("'", "''")
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

def table_by_country_drained_state_sql(with_lookup: bool) -> str:
    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = base.gadm_adm0" if with_lookup else ""
    return f"""
    WITH base AS (
      SELECT
        interval_end,
        gadm_adm0,
        drained_state_nodes,
        drained_state_meaning,
        SUM(CASE WHEN flux_type = 'drained_total_Mg_CO2e' THEN value ELSE 0 END) AS drained_MgCO2e,
        SUM(CASE WHEN flux_type = 'area__ha'               THEN value ELSE 0 END) AS area_ha
      FROM zs_drained
      GROUP BY 1,2,3,4
    )
    SELECT
      base.interval_end,
      base.gadm_adm0
      {select_l},
      base.drained_state_nodes,
      base.drained_state_meaning,
      base.drained_MgCO2e,
      base.area_ha,
      ctx.climate_domain,
      ctx.drained_state,
      ctx.emissions_state
    FROM base
    {join_l}
    LEFT JOIN drained_state_ctx AS ctx
      ON (base.drained_state_meaning = ctx.meaning)
      OR (RPAD(CAST(base.drained_state_nodes AS VARCHAR), 8, '0') = ctx.key)
    ORDER BY base.interval_end, base.drained_MgCO2e DESC
    """

def table_by_country_burned_state_sql(with_lookup: bool) -> str:
    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = base.gadm_adm0" if with_lookup else ""
    return f"""
    WITH base AS (
      SELECT
        interval_end,
        gadm_adm0,
        burned_state_nodes,
        burned_state_meaning,
        SUM(CASE WHEN flux_type = 'burned_total_Mg_CO2e' THEN value ELSE 0 END) AS burned_MgCO2e,
        SUM(CASE WHEN flux_type = 'area__ha'             THEN value ELSE 0 END) AS area_ha
      FROM zs_burned
      GROUP BY 1,2,3,4
    )
    SELECT
      base.interval_end,
      base.gadm_adm0
      {select_l},
      base.burned_state_nodes,
      base.burned_state_meaning,
      base.burned_MgCO2e,
      base.area_ha,
      ctx.climate_domain,
      ctx.burned_state,
      ctx.emissions_state
    FROM base
    {join_l}
    LEFT JOIN burned_state_ctx AS ctx
      ON (base.burned_state_meaning = ctx.meaning)
      OR (RPAD(CAST(base.burned_state_nodes AS VARCHAR), 8, '0') = ctx.key)
    ORDER BY base.interval_end, base.burned_MgCO2e DESC
    """

def table_topn_country_sql(component: str, topn: int, with_lookup: bool) -> str:
    assert component in {"drained", "burned"}
    base  = "zs_drained" if component == "drained" else "zs_burned"
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

# ----------------------------- Figures (data + plot) -----------------------------

CLIMATE_ORDER = ["Boreal", "Temperate", "Tropical"]

CLIMATE_COLORS = {
    "Boreal":    "#4575B4",  # deep blue
    "Temperate": "#FDB863",  # warm amber
    "Tropical":  "#1A9850",  # rich green
}

# Aggregated land-use order for charts
LU_AGG_ORDER = [
    "Other plantation",
    "Oil Palm",
    "Cropland",
    "Forest",
    "Grassland",
    "Settlement",
    "Otherland",
    "Wetland",
    "Extraction",
]

def sql_drained_by_climate() -> str:
    return """
    WITH joined AS (
      SELECT
        z.interval_end,
        COALESCE(ctx.climate_domain, 'Unspecified') AS climate_domain,
        SUM(CASE WHEN z.flux_type = 'drained_total_Mg_CO2e' THEN z.value ELSE 0 END) AS drained_MgCO2e
      FROM zs_drained z
      LEFT JOIN drained_state_ctx AS ctx
        ON (z.drained_state_meaning = ctx.meaning)
        OR (RPAD(CAST(z.drained_state_nodes AS VARCHAR), 8, '0') = ctx.key)
      GROUP BY 1,2
    )
    SELECT interval_end, climate_domain, drained_MgCO2e / 1e9 AS drained_GtCO2e
    FROM joined
    ORDER BY interval_end, climate_domain;
    """

def sql_burned_by_climate() -> str:
    return """
    WITH joined AS (
      SELECT
        z.interval_end,
        COALESCE(ctx.climate_domain, 'Unspecified') AS climate_domain,
        SUM(CASE WHEN z.flux_type = 'burned_total_Mg_CO2e' THEN z.value ELSE 0 END) AS burned_MgCO2e
      FROM zs_burned z
      LEFT JOIN burned_state_ctx AS ctx
        ON (z.burned_state_meaning = ctx.meaning)
        OR (RPAD(CAST(z.burned_state_nodes AS VARCHAR), 8, '0') = ctx.key)
      GROUP BY 1,2
    )
    SELECT interval_end, climate_domain, burned_MgCO2e / 1e9 AS burned_GtCO2e
    FROM joined
    ORDER BY interval_end, climate_domain;
    """

# Totals across all selected intervals (no interval_end in GROUP BY)
def sql_drained_landuse_climate_totals() -> str:
    return """
    WITH joined AS (
      SELECT
        COALESCE(ctx.climate_domain, 'Unspecified') AS climate_domain,
        COALESCE(ctx.emissions_state, 'Unspecified') AS emissions_state,
        SUM(CASE WHEN z.flux_type = 'drained_total_Mg_CO2e' THEN z.value ELSE 0 END) AS drained_MgCO2e
      FROM zs_drained z
      LEFT JOIN drained_state_ctx AS ctx
        ON (z.drained_state_meaning = ctx.meaning)
        OR (RPAD(CAST(z.drained_state_nodes AS VARCHAR), 8, '0') = ctx.key)
      GROUP BY 1,2
    )
    SELECT climate_domain, emissions_state, drained_MgCO2e / 1e9 AS drained_GtCO2e
    FROM joined;
    """

def sql_burned_landuse_climate_totals() -> str:
    return """
    WITH joined AS (
      SELECT
        COALESCE(ctx.climate_domain, 'Unspecified') AS climate_domain,
        COALESCE(ctx.emissions_state, 'Unspecified') AS emissions_state,
        SUM(CASE WHEN z.flux_type = 'burned_total_Mg_CO2e' THEN z.value ELSE 0 END) AS burned_MgCO2e
      FROM zs_burned z
      LEFT JOIN burned_state_ctx AS ctx
        ON (z.burned_state_meaning = ctx.meaning)
        OR (RPAD(CAST(z.burned_state_nodes AS VARCHAR), 8, '0') = ctx.key)
      GROUP BY 1,2
    )
    SELECT climate_domain, emissions_state, burned_MgCO2e / 1e9 AS burned_GtCO2e
    FROM joined;
    """

# Top-N peat area composition for latest interval (drained vs undrained)
def sql_topn_peat_area_comp_latest(latest_year: int, topn: int, with_lookup: bool) -> str:
    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = r.gadm_adm0" if with_lookup else ""
    return f"""
    WITH base AS (
      SELECT
        gadm_adm0,
        CASE
          WHEN drained_state_meaning LIKE 'peat_drained%%'   THEN 'drained'
          WHEN drained_state_meaning LIKE 'peat_undrained%%' THEN 'undrained'
          ELSE 'other'
        END AS peat_state,
        SUM(value) AS area_ha
      FROM zs_drained
      WHERE flux_type = 'area__ha' AND interval_end = {latest_year}
      GROUP BY 1, 2
    ),
    agg AS (
      SELECT
        gadm_adm0,
        SUM(CASE WHEN peat_state='drained'   THEN area_ha ELSE 0 END) AS drained_ha,
        SUM(CASE WHEN peat_state='undrained' THEN area_ha ELSE 0 END) AS undrained_ha
      FROM base
      GROUP BY 1
    ),
    r AS (
      SELECT
        gadm_adm0,
        drained_ha,
        undrained_ha,
        (drained_ha + undrained_ha) AS total_peat_ha
      FROM agg
      WHERE (drained_ha + undrained_ha) > 0
    )
    SELECT
      r.gadm_adm0
      {select_l},
      drained_ha   / 1e6 AS drained_area_mha,
      undrained_ha / 1e6 AS undrained_area_mha,
      total_peat_ha/ 1e6 AS total_area_mha
    FROM r
    {join_l}
    ORDER BY total_area_mha DESC
    LIMIT {topn};
    """

# Top-N combined emissions (drained + burned), totals across selected intervals
def sql_topn_total_emissions_split(topn: int, with_lookup: bool) -> str:
    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = f.gadm_adm0" if with_lookup else ""
    return f"""
    WITH d AS (
      SELECT gadm_adm0, SUM(value) AS drained_MgCO2e
      FROM zs_drained
      WHERE flux_type = 'drained_total_Mg_CO2e'
      GROUP BY 1
    ),
    b AS (
      SELECT gadm_adm0, SUM(value) AS burned_MgCO2e
      FROM zs_burned
      WHERE flux_type = 'burned_total_Mg_CO2e'
      GROUP BY 1
    ),
    f AS (
      SELECT
        COALESCE(d.gadm_adm0, b.gadm_adm0) AS gadm_adm0,
        COALESCE(d.drained_MgCO2e, 0) AS drained_MgCO2e,
        COALESCE(b.burned_MgCO2e, 0) AS burned_MgCO2e,
        COALESCE(d.drained_MgCO2e, 0) + COALESCE(b.burned_MgCO2e, 0) AS total_MgCO2e
      FROM d FULL OUTER JOIN b
      ON d.gadm_adm0 = b.gadm_adm0
    )
    SELECT
      f.gadm_adm0
      {select_l},
      burned_MgCO2e / 1e9  AS burned_GtCO2e,
      drained_MgCO2e / 1e9 AS drained_GtCO2e,
      total_MgCO2e / 1e9   AS total_GtCO2e
    FROM f
    {join_l}
    ORDER BY total_GtCO2e DESC
    LIMIT {topn};
    """

def _titlecase_domain(s: str | None) -> str:
    if s is None:
        return "Unspecified"
    t = s.strip().title()
    if t.startswith("Boreal"): return "Boreal"
    if t.startswith("Temperate"): return "Temperate"
    if t.startswith("Tropical"): return "Tropical"
    return t or "Unspecified"

def _pivot_wide(df_long: pd.DataFrame, value_col: str, index_col: str = "Year") -> pd.DataFrame:
    wide = (
        df_long
        .pivot_table(index=index_col, columns="Climate", values=value_col, aggfunc="sum", fill_value=0.0)
        .reindex(columns=CLIMATE_ORDER, fill_value=0.0)
        .reset_index()
    )
    return wide

def _write_csv_df(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, path: str):
    _ensure_parent_dir_local(path)
    tmp_name = f"df_{uuid.uuid4().hex}"
    con.register(tmp_name, df)
    out = _to_duckdb_path(path).replace("'", "''")
    con.execute(f"COPY {tmp_name} TO '{out}' (FORMAT CSV, HEADER TRUE)")
    try:
        con.unregister(tmp_name)
    except Exception:
        pass

def _save_binary(fig: plt.Figure, path: str, fmt: str, dpi: int | None = None,
                 width: float | None = None, height: float | None = None):
    if width and height:
        fig.set_size_inches(width, height)
    if _is_s3(path):
        try:
            import boto3
        except Exception as e:
            raise RuntimeError("Writing images to s3:// requires boto3. "
                               "Install boto3 or use a local OUT_DIR.") from e
        import tempfile
        suffix = ".png" if fmt == "png" else ".svg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            if fmt == "png":
                fig.savefig(tmp.name, dpi=(dpi or 300), bbox_inches="tight")
            else:
                fig.savefig(tmp.name, format="svg", bbox_inches="tight")
            tmp.flush()
            no_scheme = path[5:]
            bucket, key = no_scheme.split("/", 1)
            boto3.client("s3").upload_file(tmp.name, bucket, key)
    else:
        _ensure_parent_dir_local(path)
        if fmt == "png":
            fig.savefig(path, dpi=(dpi or 300), bbox_inches="tight")
        else:
            fig.savefig(path, format="svg", bbox_inches="tight")

def _stacked_bar(
    df_long: pd.DataFrame,
    value_col: str,
    title: str | None = None,
    index_col: str = "Year",
    x_order: Optional[Sequence[str]] = None,
    xlabel: str = "Year",
) -> plt.Figure:
    df_long = df_long.copy()
    df_long["Climate"] = pd.Categorical(df_long["Climate"], CLIMATE_ORDER, ordered=True)

    x_vals = list(x_order) if x_order is not None else sorted(df_long[index_col].unique())
    wide = _pivot_wide(df_long, value_col, index_col=index_col).set_index(index_col).reindex(index=x_vals)

    totals = wide.sum(axis=1).values
    y_max = float(max(totals)) * 1.12 if len(totals) else 1.0

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bottom = None
    for climate in CLIMATE_ORDER:
        vals = wide[climate].values
        ax.bar(
            [str(x) for x in wide.index],
            vals,
            bottom=bottom,
            label=climate,
            color=CLIMATE_COLORS.get(climate),
        )
        bottom = vals if bottom is None else bottom + vals

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Annual Emissions (Gt CO₂e/year)")
    if title:
        ax.set_title(title, pad=10)

    ax.legend(
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.10),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.2,
    )

    ax.set_ylim(0, y_max)
    pad = y_max * 0.015
    for xpos, total in zip(range(len(totals)), totals):
        ax.text(xpos, total + pad, f"{total:.2f}", ha="center", va="bottom", fontsize=9)

    ax.margins(x=0.02)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig

# --- Land-use reclass + horizontal stacked bar ---

_LU_RECLASS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(oil[_\- ]?palm|oilpalm)$"), "Oil Palm"),
    (re.compile(r"^(short[_\- ]?rotation|long[_\- ]?rotation|plantation.*|planted.*|tree[_\- ]?crop.*)$"),
     "Other plantation"),
    (re.compile(r"^cropland.*$"), "Cropland"),
    (re.compile(r"^forest.*$"), "Forest"),
    (re.compile(r"^(grassland|pasture|rangeland).*$"), "Grassland"),
    (re.compile(r"^(settlement|built[_\- ]?up|urban).*$"), "Settlement"),
    (re.compile(r"^wetland.*$"), "Wetland"),
    (re.compile(r"^(extraction|peat[_\- ]?extraction|cutover).*$"), "Extraction"),
    (re.compile(r"^(otherland|other)$"), "Otherland"),
]

def _normalize_emissions_state(s: Optional[str]) -> str:
    if s is None:
        return "other"
    return re.sub(r"[\s\-]+", "_", s.strip().lower())

def _reclass_emissions_state(s: Optional[str]) -> str:
    key = _normalize_emissions_state(s)
    for pat, lbl in _LU_RECLASS_PATTERNS:
        if pat.match(key):
            return lbl
    return (s.strip().replace("_", " ").title() if s else "Otherland")

def _aggregate_landuse(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df = df.copy()
    df["Climate"] = df["climate_domain"].apply(_titlecase_domain)
    df = df[df["Climate"].isin(CLIMATE_ORDER)]
    df["LandUse"] = df["emissions_state"].apply(_reclass_emissions_state)
    df["LandUse"] = df["LandUse"].str.replace("_", " ", regex=False)
    out = (
        df.groupby(["LandUse", "Climate"], as_index=False)[value_col]
          .sum()
          .sort_values([value_col], ascending=False)
    )
    totals = out.groupby("LandUse")[value_col].sum().sort_values(ascending=False).index.tolist()
    order = [lu for lu in LU_AGG_ORDER if lu in totals] + [lu for lu in totals if lu not in LU_AGG_ORDER]
    out["LandUse"] = pd.Categorical(out["LandUse"], order, ordered=True)
    out = out.sort_values(["LandUse", "Climate"])
    return out

def _pivot_wide_lu(df_long: pd.DataFrame, value_col: str) -> pd.DataFrame:
    wide = (
        df_long
        .pivot_table(index="LandUse", columns="Climate", values=value_col, aggfunc="sum", fill_value=0.0)
        .reindex(columns=CLIMATE_ORDER, fill_value=0.0)
        .reset_index()
    )
    return wide

def _stacked_hbar(df_long: pd.DataFrame, value_col: str, xlabel: str) -> plt.Figure:
    df_long = df_long.copy()
    df_long["Climate"] = pd.Categorical(df_long["Climate"], CLIMATE_ORDER, ordered=True)
    order = (
        df_long.groupby("LandUse")[value_col]
               .sum()
               .sort_values(ascending=False)
               .index.tolist()
    )
    wide = _pivot_wide_lu(df_long, value_col).set_index("LandUse").reindex(order)

    totals = wide.sum(axis=1).values
    x_max = float(max(totals)) if len(totals) else 1.0
    right_pad = x_max * 0.08

    height = max(3.2, 0.55 * len(order) + 1.0)
    fig, ax = plt.subplots(figsize=(7.5, height))

    left = None
    for climate in CLIMATE_ORDER:
        vals = wide[climate].values
        ax.barh(
            wide.index.astype(str),
            vals,
            left=left,
            label=climate,
            color=CLIMATE_COLORS.get(climate),
        )
        left = vals if left is None else left + vals

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Land Use")

    ax.legend(
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.12),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.2,
    )

    ax.set_xlim(0, x_max + right_pad)
    for y, total in zip(range(len(totals)), totals):
        ax.text(total + (x_max * 0.01), y, f"{total:.2f}", ha="left", va="center", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig

def _hbar_stacked_two_series(
    labels: list[str],
    left_vals: list[float],     # series A
    right_vals: list[float],    # series B (stacked to the right)
    xlabel: str,
    legend_labels: tuple[str, str],
    colors: tuple[str, str],
) -> plt.Figure:
    totals = [lv + rv for lv, rv in zip(left_vals, right_vals)]
    order = sorted(range(len(labels)), key=lambda i: totals[i], reverse=True)
    labs   = [labels[i] for i in order]
    lvals  = [left_vals[i] for i in order]
    rvals  = [right_vals[i] for i in order]
    tots   = [totals[i] for i in order]
    y = list(range(len(labs)))

    x_max = max(tots) if tots else 1.0
    height = max(3.0, 0.5 * len(labs) + 1.0)
    fig, ax = plt.subplots(figsize=(7.5, height))

    ax.barh(y, lvals, color=colors[0], label=legend_labels[0])
    ax.barh(y, rvals, left=lvals, color=colors[1], label=legend_labels[1])

    ax.set_yticks(y)
    ax.set_yticklabels(labs)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")

    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.10),
              frameon=False, handlelength=1.6, columnspacing=1.2)

    right_pad = x_max * 0.08
    ax.set_xlim(0, x_max + right_pad)
    for yy, tot in zip(y, tots):
        ax.text(tot + (x_max * 0.01), yy, f"{tot:.2f}", ha="left", va="center", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig

# ---------------------------------- CLI -----------------------------------

def main(argv=None):
    p = argparse.ArgumentParser("Build publication tables and climate-domain figures")
    p.add_argument("--model_version", required=True)
    p.add_argument("--run_name", required=True, help="Run name used in run_zonal_statistics")
    p.add_argument("--run_date", required=True, help="Run date (YYYYMMDD)")
    p.add_argument("--years", nargs="+", required=True, help="Interval end years, e.g. 2005 2010 2015 2020 2024")
    p.add_argument("--aws_region", default=None, help="AWS region for S3 (e.g., us-east-1)")
    # Tables options
    p.add_argument("--adm0_lookup", default=None, help="Optional CSV with columns: gadm_adm0,country,iso3")
    p.add_argument("--topn", type=int, default=20, help="Top N countries per period")
    p.add_argument("--do-tables", type=int, default=1, help="1=write tables, 0=skip")
    # Figures options
    p.add_argument("--do-figures", type=int, default=1, help="1=write figures, 0=skip")
    p.add_argument("--data-only", action="store_true", help="Figures: write CSVs only (no images)")
    p.add_argument("--no-png", action="store_true", help="Figures: skip PNG outputs")
    p.add_argument("--no-svg", action="store_true", help="Figures: skip SVG outputs")
    args = p.parse_args(argv)

    years = [int(y) for y in args.years]
    interval_folders = _interval_folder_strings(years)
    base_prefixes = _make_base_prefixes(args.model_version, args.run_name, args.run_date, interval_folders)
    drained_globs, burned_globs = _make_globs_for_components(base_prefixes)

    con = duckdb.connect()
    _register_all(con, drained_globs, burned_globs, aws_region=args.aws_region)
    have_lookup = _maybe_register_lookup(con, args.adm0_lookup)
    _register_state_context_views(con)

    # Build shared labels for Inventory Period axis
    period_labels = _make_period_labels_from_years(years)   # {2005: "2000-2005", ...}
    x_order = [period_labels[y] for y in years]
    inv_col = "Inventory Period"

    # -------------------- Tables --------------------
    if args.do_tables:
        out_tables_dir = OUT_DIR
        _copy_sql(con, table_by_country_period_sql(with_lookup=have_lookup),
                  _join(out_tables_dir, "by_country_period.csv"))

        _copy_sql(con, table_by_drained_state_sql(),
                  _join(out_tables_dir, "by_drained_state_period.csv"))

        _copy_sql(con, table_by_burned_state_sql(),
                  _join(out_tables_dir, "by_burned_state_period.csv"))

        _copy_sql(con, table_by_country_drained_state_sql(with_lookup=have_lookup),
                  _join(out_tables_dir, "by_country_drained_state_period.csv"))

        _copy_sql(con, table_by_country_burned_state_sql(with_lookup=have_lookup),
                  _join(out_tables_dir, "by_country_burned_state_period.csv"))

        _copy_sql(con, table_topn_country_sql("drained", args.topn, have_lookup),
                  _join(out_tables_dir, f"top{args.topn}_by_country_drained.csv"))

        _copy_sql(con, table_topn_country_sql("burned", args.topn, have_lookup),
                  _join(out_tables_dir, f"top{args.topn}_by_country_burned.csv"))

    # -------------------- Figures --------------------
    if args.do_figures:
        # ---------- drained (by climate over time) ----------
        df_d = con.execute(sql_drained_by_climate()).df()
        df_d["Climate"] = df_d["climate_domain"].apply(_titlecase_domain)
        df_d = df_d[["interval_end", "Climate", "drained_GtCO2e"]].rename(columns={"interval_end": "Year"})
        df_d = df_d[df_d["Climate"].isin(CLIMATE_ORDER)]
        df_d[inv_col] = df_d["Year"].map(period_labels)

        drained_long_csv = _join(OUT_DIR, "figures", "data", "global_drained_climate_long.csv")
        drained_wide_csv = _join(OUT_DIR, "figures", "data", "global_drained_climate_wide.csv")

        d_long = df_d[[inv_col, "Climate", "drained_GtCO2e"]]
        _write_csv_df(con, d_long, drained_long_csv)
        d_wide = _pivot_wide(d_long, "drained_GtCO2e", index_col=inv_col)
        _write_csv_df(con, d_wide, drained_wide_csv)

        if not args.data_only:
            fig_d = _stacked_bar(
                d_long, "drained_GtCO2e",
                title=None, index_col=inv_col, x_order=x_order, xlabel="Inventory Period"
            )
            if not args.no_png:
                _save_binary(fig_d, _join(OUT_DIR, "figures", "global_drained_climate_column.png"),
                             fmt="png", dpi=300, width=7.5, height=4.5)
            if not args.no_svg:
                _save_binary(fig_d, _join(OUT_DIR, "figures", "global_drained_climate_column.svg"),
                             fmt="svg", width=7.5, height=4.5)

        # ---------- burned (by climate over time) ----------
        df_b = con.execute(sql_burned_by_climate()).df()
        df_b["Climate"] = df_b["climate_domain"].apply(_titlecase_domain)
        df_b = df_b[["interval_end", "Climate", "burned_GtCO2e"]].rename(columns={"interval_end": "Year"})
        df_b = df_b[df_b["Climate"].isin(CLIMATE_ORDER)]
        df_b[inv_col] = df_b["Year"].map(period_labels)

        burned_long_csv = _join(OUT_DIR, "figures", "data", "global_burn_climate_long.csv")
        burned_wide_csv = _join(OUT_DIR, "figures", "data", "global_burn_climate_wide.csv")

        b_long = df_b[[inv_col, "Climate", "burned_GtCO2e"]]
        _write_csv_df(con, b_long, burned_long_csv)
        b_wide = _pivot_wide(b_long, "burned_GtCO2e", index_col=inv_col)
        _write_csv_df(con, b_wide, burned_wide_csv)

        if not args.data_only:
            fig_b = _stacked_bar(
                b_long, "burned_GtCO2e",
                title=None, index_col=inv_col, x_order=x_order, xlabel="Inventory Period"
            )
            if not args.no_png:
                _save_binary(fig_b, _join(OUT_DIR, "figures", "global_burn_climate_column.png"),
                             fmt="png", dpi=300, width=7.5, height=4.5)
            if not args.no_svg:
                _save_binary(fig_b, _join(OUT_DIR, "figures", "global_burn_climate_column.svg"),
                             fmt="svg", width=7.5, height=4.5)

        # ---------- drained: Land Use × Climate (totals across all periods) ----------
        d_lu_raw = con.execute(sql_drained_landuse_climate_totals()).df()
        d_lu_long = _aggregate_landuse(d_lu_raw, "drained_GtCO2e")
        d_lu_long_csv = _join(OUT_DIR, "figures", "data", "drained_landuse_climate_long.csv")
        d_lu_wide_csv = _join(OUT_DIR, "figures", "data", "drained_landuse_climate_wide.csv")
        _write_csv_df(con, d_lu_long[["LandUse", "Climate", "drained_GtCO2e"]], d_lu_long_csv)
        _write_csv_df(con, _pivot_wide_lu(d_lu_long, "drained_GtCO2e"), d_lu_wide_csv)

        if not args.data_only:
            fig_dlu = _stacked_hbar(d_lu_long, "drained_GtCO2e", xlabel="Total Emissions (Gt CO₂e)")
            if not args.no_png:
                _save_binary(fig_dlu, _join(OUT_DIR, "figures", "drained_landuse_climate_bar.png"),
                             fmt="png", dpi=300)
            if not args.no_svg:
                _save_binary(fig_dlu, _join(OUT_DIR, "figures", "drained_landuse_climate_bar.svg"),
                             fmt="svg")

        # ---------- burned: Land Use × Climate (totals across all periods) ----------
        b_lu_raw = con.execute(sql_burned_landuse_climate_totals()).df()
        b_lu_long = _aggregate_landuse(b_lu_raw, "burned_GtCO2e")
        b_lu_long_csv = _join(OUT_DIR, "figures", "data", "burned_landuse_climate_long.csv")
        b_lu_wide_csv = _join(OUT_DIR, "figures", "data", "burned_landuse_climate_wide.csv")
        _write_csv_df(con, b_lu_long[["LandUse", "Climate", "burned_GtCO2e"]], b_lu_long_csv)
        _write_csv_df(con, _pivot_wide_lu(b_lu_long, "burned_GtCO2e"), b_lu_wide_csv)

        if not args.data_only:
            fig_blu = _stacked_hbar(b_lu_long, "burned_GtCO2e", xlabel="Total Emissions (Gt CO₂e)")
            if not args.no_png:
                _save_binary(fig_blu, _join(OUT_DIR, "figures", "burned_landuse_climate_bar.png"),
                             fmt="png", dpi=300)
            if not args.no_svg:
                _save_binary(fig_blu, _join(OUT_DIR, "figures", "burned_landuse_climate_bar.svg"),
                             fmt="svg")

        # ---------- Top-N by country: PEAT AREA split (latest interval only) ----------
        latest_year = max(years)
        df_area = con.execute(sql_topn_peat_area_comp_latest(latest_year, args.topn, have_lookup)).df()
        if "iso3" in df_area.columns:
            df_area["label"] = df_area["iso3"].fillna(df_area["gadm_adm0"].astype(str))
        else:
            df_area["label"] = df_area["gadm_adm0"].astype(str)

        area_csv = _join(OUT_DIR, "figures", "data", "top_10_country_peat_area.csv")
        _write_csv_df(
            con,
            df_area[["label", "drained_area_mha", "undrained_area_mha", "total_area_mha"]]
                  .rename(columns={"label": "iso3_or_code"}),
            area_csv,
        )

        if not args.data_only:
            fig_area = _hbar_stacked_two_series(
                labels=df_area["label"].tolist(),
                left_vals=df_area["drained_area_mha"].tolist(),
                right_vals=df_area["undrained_area_mha"].tolist(),
                xlabel="Total Area (million ha)",
                legend_labels=("Drained peat area", "Undrained peat area"),
                colors=("#FB6A29", "#3E3753"),
            )
            if not args.no_png:
                _save_binary(fig_area, _join(OUT_DIR, "figures", "top_10_country_peat_area_bar.png"),
                             fmt="png", dpi=300)
            if not args.no_svg:
                _save_binary(fig_area, _join(OUT_DIR, "figures", "top_10_country_peat_area_bar.svg"),
                             fmt="svg")

        # ---------- Top-N by country: TOTAL EMISSIONS split (drained + burned) ----------
        df_emsplit = con.execute(sql_topn_total_emissions_split(args.topn, have_lookup)).df()
        if "iso3" in df_emsplit.columns:
            df_emsplit["label"] = df_emsplit["iso3"].fillna(df_emsplit["gadm_adm0"].astype(str))
        else:
            df_emsplit["label"] = df_emsplit["gadm_adm0"].astype(str)

        em_split_csv = _join(OUT_DIR, "figures", "data", "top_10_country_total_emissions.csv")
        _write_csv_df(
            con,
            df_emsplit[["label", "burned_GtCO2e", "drained_GtCO2e", "total_GtCO2e"]]
                     .rename(columns={"label": "iso3_or_code"}),
            em_split_csv,
        )

        if not args.data_only:
            # Orange for burned (left), purple for drained (right) to match earlier examples
            fig_emsplit = _hbar_stacked_two_series(
                labels=df_emsplit["label"].tolist(),
                left_vals=df_emsplit["burned_GtCO2e"].tolist(),
                right_vals=df_emsplit["drained_GtCO2e"].tolist(),
                xlabel="Total Emissions (Gt CO₂e)",
                legend_labels=("Total burned emissions", "Total drained emissions"),
                colors=("#FB6A29", "#3E3753"),
            )
            if not args.no_png:
                _save_binary(fig_emsplit, _join(OUT_DIR, "figures", "top_10_country_total_emissions_bar.png"),
                             fmt="png", dpi=300)
            if not args.no_svg:
                _save_binary(fig_emsplit, _join(OUT_DIR, "figures", "top_10_country_total_emissions_bar.svg"),
                             fmt="svg")

    print("Assets written to:", OUT_DIR)

if __name__ == "__main__":
    main()
