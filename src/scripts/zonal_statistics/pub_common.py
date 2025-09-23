# -*- coding: utf-8 -*-
"""
Public helpers for building publication tables & figures from organic-soils zonal statistics.

This module intentionally exposes only PUBLIC functions:
- register_state_context_views(con): creates 'drained_state_ctx' and 'burned_state_ctx'
- table_* SQL builders
- sql_* SQL builders (for figure datasets)

Assumptions (created by the driver before using these SQLs):
  - DuckDB views:
      * zs_drained (columns incl. interval_end, flux_type, value, gadm_adm0,
                    drained_state_meaning, drained_state_nodes)
      * zs_burned  (columns incl. interval_end, flux_type, value, gadm_adm0,
                    burned_state_meaning,  burned_state_nodes)
  - Optional DuckDB view 'adm0_lookup' with columns: gadm_adm0, country, iso3
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.scripts.zonal_statistics import zonal_constants as zc


# ---------------------------------------------------------------------------
# Public: register_state_context_views
# ---------------------------------------------------------------------------

def register_state_context_views(con) -> None:
    """
    Register state-context lookup views used by the SQL builders below.

    Creates two DuckDB views by registering in-memory DataFrames:
      - drained_state_ctx (key, meaning, climate_domain, drained_state, emissions_state)
      - burned_state_ctx  (key, meaning, climate_domain, burned_state,  emissions_state)
    """
    # ----- Build drained context from zonal_constants -----
    d_rows = []
    for key, meaning in zc.DRAINED_STATE_NODE_MEANINGS.items():
        # Meaning encodes domain and state as e.g. "peat_drained__tropical_oil_palm"
        climate_domain = None
        drained_state = None
        emissions_state = None

        if "__" in meaning:
            left, right = meaning.split("__", 1)
            drained_state = left
            # pull domain if present at the start of 'right'
            if "_" in right:
                dom, rest = right.split("_", 1)
                if dom in {"boreal", "temperate", "tropical"}:
                    climate_domain = dom
                    emissions_state = rest
                else:
                    emissions_state = right
            else:
                emissions_state = right
        else:
            drained_state = meaning

        d_rows.append({
            "key": f"{key}",
            "meaning": f"{meaning}",
            "climate_domain": climate_domain,
            "drained_state": drained_state,
            "emissions_state": emissions_state,
        })
    d_df = pd.DataFrame(d_rows)

    # ----- Build burned context from zonal_constants -----
    b_rows = []
    for key, meaning in zc.BURNED_STATE_NODE_MEANINGS.items():
        climate_domain = None
        burned_state = None
        emissions_state = None

        if "__" in meaning:
            dom, state = meaning.split("__", 1)
            climate_domain = dom
            burned_state = state
            emissions_state = state
        else:
            burned_state = meaning
            emissions_state = meaning

        b_rows.append({
            "key": f"{key}",
            "meaning": f"{meaning}",
            "climate_domain": climate_domain,
            "burned_state": burned_state,
            "emissions_state": emissions_state,
        })
    b_df = pd.DataFrame(b_rows)

    # Register as DuckDB relations (temp)
    con.register("drained_state_ctx", d_df)
    con.register("burned_state_ctx",  b_df)


# ---------------------------------------------------------------------------
# Public: TABLE SQL builders
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public: FIGURE SQL builders
# ---------------------------------------------------------------------------

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


def sql_drained_landuse_climate_avgs(n_periods: int) -> str:
    return f"""
    WITH joined AS (
      SELECT
        COALESCE(ctx.climate_domain, 'Unspecified')  AS climate_domain,
        COALESCE(ctx.emissions_state, 'Unspecified') AS emissions_state,
        SUM(CASE WHEN z.flux_type = 'drained_total_Mg_CO2e' THEN z.value ELSE 0 END) AS drained_MgCO2e
      FROM zs_drained z
      LEFT JOIN drained_state_ctx AS ctx
        ON (z.drained_state_meaning = ctx.meaning)
        OR (RPAD(CAST(z.drained_state_nodes AS VARCHAR), 8, '0') = ctx.key)
      GROUP BY 1,2
    )
    SELECT
      climate_domain,
      emissions_state,
      (drained_MgCO2e / {n_periods}) / 1e9 AS drained_avg_GtCO2e_per_yr
    FROM joined;
    """


def sql_burned_landuse_climate_avgs(n_periods: int) -> str:
    return f"""
    WITH joined AS (
      SELECT
        COALESCE(ctx.climate_domain, 'Unspecified')  AS climate_domain,
        COALESCE(ctx.emissions_state, 'Unspecified') AS emissions_state,
        SUM(CASE WHEN z.flux_type = 'burned_total_Mg_CO2e' THEN z.value ELSE 0 END) AS burned_MgCO2e
      FROM zs_burned z
      LEFT JOIN burned_state_ctx AS ctx
        ON (z.burned_state_meaning = ctx.meaning)
        OR (RPAD(CAST(z.burned_state_nodes AS VARCHAR), 8, '0') = ctx.key)
      GROUP BY 1,2
    )
    SELECT
      climate_domain,
      emissions_state,
      (burned_MgCO2e / {n_periods}) / 1e9 AS burned_avg_GtCO2e_per_yr
    FROM joined;
    """


def sql_topn_total_emissions_split_avg(topn: int, with_lookup: bool, n_periods: int) -> str:
    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = ranked.gadm_adm0" if with_lookup else ""
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
    ),
    avgd AS (
      SELECT
        gadm_adm0,
        (drained_MgCO2e / {n_periods}) / 1e9 AS drained_avg_GtCO2e_per_yr,
        (burned_MgCO2e  / {n_periods}) / 1e9 AS burned_avg_GtCO2e_per_yr,
        (total_MgCO2e   / {n_periods}) / 1e9 AS total_avg_GtCO2e_per_yr
      FROM f
    ),
    ranked AS (
      SELECT
        gadm_adm0,
        drained_avg_GtCO2e_per_yr,
        burned_avg_GtCO2e_per_yr,
        total_avg_GtCO2e_per_yr,
        ROW_NUMBER() OVER (ORDER BY total_avg_GtCO2e_per_yr DESC) AS rnk
      FROM avgd
    )
    SELECT
      ranked.gadm_adm0
      {select_l},
      burned_avg_GtCO2e_per_yr,
      drained_avg_GtCO2e_per_yr,
      total_avg_GtCO2e_per_yr
    FROM ranked
    {join_l}
    WHERE rnk <= {topn}
    ORDER BY rnk;
    """


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


def sql_global_totals_by_period_long() -> str:
    """
    Long-format global totals by inventory period and component.
    Columns: interval_end, component ('Drained'|'Burned'), GtCO2e
    """
    return """
    WITH d AS (
      SELECT interval_end, SUM(value) AS Mg
      FROM zs_drained
      WHERE flux_type = 'drained_total_Mg_CO2e'
      GROUP BY 1
    ),
    b AS (
      SELECT interval_end, SUM(value) AS Mg
      FROM zs_burned
      WHERE flux_type = 'burned_total_Mg_CO2e'
      GROUP BY 1
    )
    SELECT interval_end, 'Drained' AS component, d.Mg / 1e9 AS GtCO2e FROM d
    UNION ALL
    SELECT interval_end, 'Burned'  AS component, b.Mg / 1e9 AS GtCO2e FROM b
    ORDER BY interval_end, component;
    """


def sql_topn_avg_component_emissions(component: str, topn: int, with_lookup: bool, n_periods: int) -> str:
    """
    Top-N by average annual emissions for a single component ('drained' or 'burned').
    Returns: gadm_adm0[, country, iso3], <comp>_avg_GtCO2e_per_yr
    """
    assert component in {"drained", "burned"}
    base   = "zs_drained" if component == "drained" else "zs_burned"
    ftype  = "drained_total_Mg_CO2e" if component == "drained" else "burned_total_Mg_CO2e"
    alias  = "drained_avg_GtCO2e_per_yr" if component == "drained" else "burned_avg_GtCO2e_per_yr"
    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = ranked.gadm_adm0" if with_lookup else ""
    return f"""
    WITH t AS (
      SELECT gadm_adm0, SUM(value) AS Mg
      FROM {base}
      WHERE flux_type = '{ftype}'
      GROUP BY 1
    ),
    avgd AS (
      SELECT gadm_adm0, (Mg / {n_periods}) / 1e9 AS avg_Gt_per_yr
      FROM t
    ),
    ranked AS (
      SELECT gadm_adm0, avg_Gt_per_yr,
             ROW_NUMBER() OVER (ORDER BY avg_Gt_per_yr DESC) AS rnk
      FROM avgd
    )
    SELECT ranked.gadm_adm0{select_l}, avg_Gt_per_yr AS {alias}
    FROM ranked
    {join_l}
    WHERE rnk <= {topn}
    ORDER BY rnk;
    """


def sql_country_emissions_intensity_avg(
    n_periods: int,
    with_lookup: bool,
    topn: Optional[int] = None,
    min_area_ha: float = 10000.0
) -> str:
    """
    Drained emissions intensity = (avg annual drained emissions) / (latest drained peat area).
    Returns: gadm_adm0[, country, iso3], intensity_tCO2e_per_ha_yr,
             total_avg_GtCO2e_per_yr, latest_drained_area_mha
    """
    select_l = ", l.country, l.iso3" if with_lookup else ""
    # IMPORTANT: join ISO to alias 'e' (final SELECT source)
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = e.gadm_adm0" if with_lookup else ""
    limit    = f"LIMIT {int(topn)}" if topn is not None else ""
    return f"""
    WITH area_by_period AS (
      SELECT interval_end, gadm_adm0,
             SUM(CASE WHEN drained_state_meaning LIKE 'peat_drained%%' THEN value ELSE 0 END) AS drained_ha
      FROM zs_drained
      WHERE flux_type = 'area__ha'
      GROUP BY 1,2
    ),
    latest_area AS (
      SELECT a.gadm_adm0, a.drained_ha AS latest_drained_ha
      FROM area_by_period a
      JOIN (SELECT gadm_adm0, MAX(interval_end) AS max_end FROM area_by_period GROUP BY 1) mx
        ON a.gadm_adm0 = mx.gadm_adm0 AND a.interval_end = mx.max_end
    ),
    d AS (
      SELECT gadm_adm0, SUM(value) AS drained_Mg_per_periods
      FROM zs_drained
      WHERE flux_type = 'drained_total_Mg_CO2e'
      GROUP BY 1
    ),
    avg_em AS (
      SELECT gadm_adm0, (drained_Mg_per_periods / {n_periods}) AS avg_Mg_per_yr
      FROM d
    )
    SELECT
      e.gadm_adm0{select_l},
      (e.avg_Mg_per_yr / NULLIF(a.latest_drained_ha, 0))        AS intensity_tCO2e_per_ha_yr,
      (e.avg_Mg_per_yr / 1e9)                                   AS total_avg_GtCO2e_per_yr,
      (a.latest_drained_ha / 1e6)                               AS latest_drained_area_mha
    FROM avg_em e
    JOIN latest_area a ON a.gadm_adm0 = e.gadm_adm0
    {join_l}
    WHERE a.latest_drained_ha >= {min_area_ha}
    ORDER BY intensity_tCO2e_per_ha_yr DESC
    {limit};
    """


def sql_country_emissions_vs_area_avg(n_periods: int, with_lookup: bool) -> str:
    """
    Scatter-ready dataset: average-annual total emissions vs average drained area by country.
    Returns: gadm_adm0[, country, iso3], total_avg_GtCO2e_per_yr, avg_drained_area_mha
    """
    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = f.gadm_adm0" if with_lookup else ""
    return f"""
    WITH area_by_period AS (
      SELECT interval_end, gadm_adm0,
             SUM(CASE
                   WHEN drained_state_meaning LIKE 'peat_drained%%' THEN value
                   ELSE 0 END) AS drained_ha
      FROM zs_drained
      WHERE flux_type = 'area__ha'
      GROUP BY 1,2
    ),
    area_avg AS (
      SELECT gadm_adm0, AVG(drained_ha) AS avg_drained_ha
      FROM area_by_period
      GROUP BY 1
    ),
    d AS (
      SELECT gadm_adm0, SUM(value) AS drained_Mg
      FROM zs_drained
      WHERE flux_type = 'drained_total_Mg_CO2e'
      GROUP BY 1
    ),
    b AS (
      SELECT gadm_adm0, SUM(value) AS burned_Mg
      FROM zs_burned
      WHERE flux_type = 'burned_total_Mg_CO2e'
      GROUP BY 1
    ),
    f AS (
      SELECT
        COALESCE(d.gadm_adm0, b.gadm_adm0) AS gadm_adm0,
        COALESCE(d.drained_Mg, 0) AS drained_Mg,
        COALESCE(b.burned_Mg, 0)  AS burned_Mg
      FROM d FULL OUTER JOIN b ON d.gadm_adm0 = b.gadm_adm0
    ),
    avg_em AS (
      SELECT gadm_adm0, (drained_Mg + burned_Mg) / {n_periods} AS avg_Mg_per_yr
      FROM f
    )
    SELECT
      f.gadm_adm0{select_l},
      (avg_Mg_per_yr / 1e9)           AS total_avg_GtCO2e_per_yr,
      (a.avg_drained_ha / 1e6)        AS avg_drained_area_mha
    FROM avg_em f
    LEFT JOIN area_avg a
      ON a.gadm_adm0 = f.gadm_adm0
    {join_l}
    WHERE a.avg_drained_ha IS NOT NULL AND a.avg_drained_ha > 0
    ORDER BY total_avg_GtCO2e_per_yr DESC;
    """
