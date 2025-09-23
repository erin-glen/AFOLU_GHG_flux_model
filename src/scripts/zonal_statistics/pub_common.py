# -*- coding: utf-8 -*-
"""
Utilities for building publication tables & figures from organic-soils zonal statistics.

Pure utilities: plotting helpers, constants, SQL string builders.
No DuckDB setup or registration; the driver wires everything.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence, List, Dict

import pandas as pd
import matplotlib.pyplot as plt

# Attempt to import pycountry lazily for ISO lookups. The dependency is optional and
# the rest of the module works without it, so we swallow any import-time errors and
# simply operate without country names/ISO codes when unavailable.
try:  # pragma: no cover - import guard is environment dependent
    import pycountry  # type: ignore
except Exception:  # pragma: no cover - best effort optional dependency
    pycountry = None  # type: ignore

# ----------------------------- Plot constants -----------------------------

CLIMATE_ORDER = ["Boreal", "Temperate", "Tropical"]

CLIMATE_COLORS = {
    "Boreal":    "#4575B4",  # deep blue
    "Temperate": "#FDB863",  # warm amber
    "Tropical":  "#1A9850",  # rich green
}

PROCESS_ORDER = ["Drained", "Burned"]
PROCESS_COLORS = {"Drained": "#3E3753", "Burned": "#FB6A29"}

# ----------------------------- Small helpers ------------------------------

def titlecase_domain(s: Optional[str]) -> str:
    if s is None:
        return "Unspecified"
    t = s.strip().title()
    if t.startswith("Boreal"): return "Boreal"
    if t.startswith("Temperate"): return "Temperate"
    if t.startswith("Tropical"): return "Tropical"
    return t or "Unspecified"

def country_label(df: pd.DataFrame) -> pd.Series:
    """Prefer iso3 if present/non-empty, else fallback to numeric code."""
    if "iso3" in df.columns:
        iso = df["iso3"].astype("string")
        return iso.where(iso.str.len().fillna(0) > 0, df["gadm_adm0"].astype(str))
    return df["gadm_adm0"].astype(str)


def build_adm0_lookup_df(manual_overrides: Optional[Dict[int, Dict[str, Optional[str]]]] = None) -> pd.DataFrame:
    """Return a DataFrame with gadm_adm0 → (iso3, country) best-effort mappings."""

    from src.scripts.zonal_statistics import zonal_constants as zc

    overrides: Dict[int, Dict[str, Optional[str]]] = manual_overrides or {}

    rows: list[dict[str, Optional[str] | int]] = []
    seen: set[int] = set()

    for raw_code in zc.GADM_ADM0_IDS.tolist():
        code = int(raw_code)
        if code in seen:
            continue
        seen.add(code)

        if code == 0:
            rows.append({"gadm_adm0": 0, "iso3": None, "country": "NoData"})
            continue

        if code in overrides:
            entry = overrides[code]
            rows.append({
                "gadm_adm0": code,
                "iso3": entry.get("iso3"),
                "country": entry.get("country"),
            })
            continue

        iso3: Optional[str] = None
        country: Optional[str] = None

        if pycountry is not None:
            lookup_code = f"{code:03d}"
            record = pycountry.countries.get(numeric=lookup_code)
            if record is None:
                # Fallback to historic countries (e.g., defunct ISO assignments)
                record = getattr(pycountry, "historic_countries", None)
                if record is not None:
                    record = record.get(numeric=lookup_code)  # type: ignore[assignment]

            if record is not None:
                iso3 = getattr(record, "alpha_3", None)
                country = (
                    getattr(record, "common_name", None)
                    or getattr(record, "official_name", None)
                    or getattr(record, "name", None)
                )

        rows.append({"gadm_adm0": code, "iso3": iso3, "country": country})

    df = pd.DataFrame(rows, columns=["gadm_adm0", "iso3", "country"])
    if not df.empty:
        df = df.astype({"gadm_adm0": "int32"})
    return df

def pivot_wide(df_long: pd.DataFrame, value_col: str, index_col: str) -> pd.DataFrame:
    return (
        df_long
        .pivot_table(index=index_col, columns="Climate", values=value_col,
                     aggfunc="sum", fill_value=0.0, observed=False)
        .reindex(columns=CLIMATE_ORDER, fill_value=0.0)
        .reset_index()
    )

# ----------------------------- Land-use reclass ---------------------------

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

def aggregate_landuse(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Reclass emissions_state to coarse land-uses and aggregate by (LandUse × Climate)."""
    df = df.copy()
    df["Climate"] = df["climate_domain"].apply(titlecase_domain)
    df = df[df["Climate"].isin(CLIMATE_ORDER)]
    df["LandUse"] = df["emissions_state"].apply(_reclass_emissions_state).str.replace("_", " ", regex=False)
    out = (
        df.groupby(["LandUse", "Climate"], as_index=False, observed=False)[value_col]
          .sum()
    )
    totals = out.groupby("LandUse", observed=False)[value_col].sum().sort_values(ascending=False).index.tolist()
    out["LandUse"] = pd.Categorical(out["LandUse"], totals, ordered=True)
    return out.sort_values(["LandUse", "Climate"])

# ----------------------------- Plot helpers -------------------------------

def stacked_column_by_category(
    df_long: pd.DataFrame,
    index_col: str,
    category_col: str,
    value_col: str,
    category_order: Sequence[str],
    color_map: dict[str, str],
    xlabel: str,
    ylabel: str,
    width: float = 7.5,
    height: float = 4.5,
    legend_above: bool = False,
) -> plt.Figure:
    df = df_long.copy()
    df[category_col] = pd.Categorical(df[category_col], category_order, ordered=True)
    x_vals = list(dict.fromkeys(df[index_col].tolist()))
    wide = (
        df.pivot_table(index=index_col, columns=category_col, values=value_col,
                       aggfunc="sum", fill_value=0.0, observed=False)
          .reindex(index=x_vals)
          .reindex(columns=category_order, fill_value=0.0)
    )
    totals = wide.sum(axis=1).values
    y_max = float(max(totals)) * 1.12 if len(totals) else 1.0

    fig, ax = plt.subplots(figsize=(width, height))
    bottom = None
    for cat in category_order:
        vals = wide[cat].values
        ax.bar([str(x) for x in wide.index], vals, bottom=bottom, label=cat, color=color_map.get(cat))
        bottom = vals if bottom is None else bottom + vals
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if legend_above:
        ax.legend(ncol=min(len(category_order), 5), loc="upper center",
                  bbox_to_anchor=(0.5, 1.18), frameon=False, handlelength=1.6, columnspacing=1.2)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
    else:
        ax.legend(ncol=min(len(category_order), 4), loc="upper left",
                  bbox_to_anchor=(0.0, 1.10), frameon=False, handlelength=1.6, columnspacing=1.2)
        fig.tight_layout(rect=(0, 0, 1, 0.88))

    ax.set_ylim(0, y_max)
    for xpos, total in zip(range(len(totals)), totals):
        ax.text(xpos, total + (y_max * 0.015), f"{total:.2f}", ha="center", va="bottom", fontsize=9)
    return fig

def stacked_hbar(df_long: pd.DataFrame, value_col: str, xlabel: str) -> plt.Figure:
    df = df_long.copy()
    df["Climate"] = pd.Categorical(df["Climate"], CLIMATE_ORDER, ordered=True)
    wide = (
        df.pivot_table(index="LandUse", columns="Climate", values=value_col,
                       aggfunc="sum", fill_value=0.0, observed=False)
          .reindex(columns=CLIMATE_ORDER, fill_value=0.0)
    )
    order = list(wide.sum(axis=1).sort_values(ascending=False).index)
    wide = wide.reindex(order)

    totals = wide.sum(axis=1).values
    x_max = float(max(totals)) if len(totals) else 1.0
    height = max(3.2, 0.55 * len(order) + 1.0)
    fig, ax = plt.subplots(figsize=(7.5, height))

    left = None
    for climate in CLIMATE_ORDER:
        vals = wide[climate].values
        ax.barh(wide.index.astype(str), vals, left=left, color=CLIMATE_COLORS.get(climate), label=climate)
        left = vals if left is None else left + vals

    ax.set_xlabel(xlabel); ax.set_ylabel("Land Use")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.12),
              frameon=False, handlelength=1.6, columnspacing=1.2)
    for y, total in zip(range(len(totals)), totals):
        ax.text(total + (x_max * 0.01), y, f"{total:.2f}", ha="left", va="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig

def hbar_two_series(labels: List[str], left_vals: List[float], right_vals: List[float],
                    xlabel: str, legends: tuple[str, str], colors: tuple[str, str]) -> plt.Figure:
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
    ax.barh(y, lvals, color=colors[0], label=legends[0])
    ax.barh(y, rvals, left=lvals, color=colors[1], label=legends[1])
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.10), frameon=False)
    for yy, tot in zip(y, tots):
        ax.text(tot + (x_max * 0.01), yy, f"{tot:.2f}", ha="left", va="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig

def barh_single(labels: List[str], values: List[float], xlabel: str, color: str,
                *, sort_desc: bool = True) -> plt.Figure:
    """Horizontal bar chart for a single series.

    Parameters
    ----------
    labels
        Category labels corresponding to ``values``.
    values
        Numeric values to plot.
    xlabel
        Label for the X axis.
    color
        Bar color (all bars use the same color).
    sort_desc
        Whether to sort bars descending by value (default) or preserve the
        original order provided by ``labels`` and ``values``.
    """

    if sort_desc:
        order = sorted(range(len(labels)), key=lambda i: values[i], reverse=True)
    else:
        order = list(range(len(labels)))
    labs  = [labels[i] for i in order]
    vals  = [values[i] for i in order]
    y = list(range(len(labs)))
    height = max(3.0, 0.5 * len(labs) + 1.0)
    fig, ax = plt.subplots(figsize=(7.5, height))
    ax.barh(y, vals, color=color)
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    x_max = max(vals) if vals else 1.0
    for yy, v in zip(y, vals):
        ax.text(v + (x_max * 0.01), yy, f"{v:.2f}", ha="left", va="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig

# ----------------------------- SQL builders -------------------------------

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

# ----------------------------- Figure SQL -------------------------------

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

def sql_global_peat_area_split(latest_year: int) -> str:
    return f"""
    WITH base AS (
      SELECT
        CASE
          WHEN drained_state_meaning LIKE 'peat_drained%%'   THEN 'drained'
          WHEN drained_state_meaning LIKE 'peat_undrained%%' THEN 'undrained'
          ELSE 'other'
        END AS peat_state,
        SUM(value) AS area_ha
      FROM zs_drained
      WHERE flux_type = 'area__ha' AND interval_end = {latest_year}
      GROUP BY 1
    ),
    agg AS (
      SELECT peat_state, SUM(area_ha) AS area_ha
      FROM base
      GROUP BY 1
    ),
    states AS (
      SELECT * FROM (VALUES ('drained'), ('undrained')) AS s(peat_state)
    )
    SELECT
      s.peat_state,
      COALESCE(a.area_ha, 0) / 1e6 AS area_mha
    FROM states s
    LEFT JOIN agg a ON a.peat_state = s.peat_state
    ORDER BY s.peat_state;
    """

def sql_global_component_emissions_avg(n_periods: int) -> str:
    return f"""
    WITH d AS (
      SELECT 'Drained' AS component, SUM(value) AS total_Mg
      FROM zs_drained
      WHERE flux_type = 'drained_total_Mg_CO2e'
    ),
    b AS (
      SELECT 'Burned' AS component, SUM(value) AS total_Mg
      FROM zs_burned
      WHERE flux_type = 'burned_total_Mg_CO2e'
    ),
    unioned AS (
      SELECT * FROM d
      UNION ALL
      SELECT * FROM b
    ),
    components AS (
      SELECT * FROM (VALUES ('Drained'), ('Burned')) AS t(component)
    )
    SELECT
      c.component,
      COALESCE(u.total_Mg, 0) / NULLIF({n_periods}, 0) / 1e9 AS avg_GtCO2e_per_yr
    FROM components c
    LEFT JOIN unioned u ON u.component = c.component
    ORDER BY c.component;
    """

def sql_global_totals_by_period_long() -> str:
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
    select_l = ", l.country, l.iso3" if with_lookup else ""
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