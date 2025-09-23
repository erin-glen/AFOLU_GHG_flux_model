# -*- coding: utf-8 -*-
"""
Build publication **tables** and **figures** from organic-soils zonal statistics (driver).

- PNG-only outputs
- Self-sufficient: sets up DuckDB httpfs + views (no private imports from pub_common)
- Imports pub_common as `pc` for SQL builders only
- No line charts
"""

from __future__ import annotations

import argparse
import os
import posixpath
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import duckdb
import pandas as pd
import matplotlib.pyplot as plt

import src.scripts.zonal_statistics.pub_common as pc
from src.scripts.zonal_statistics.run_zonal_stats import (
    build_output_parquet,
    build_interval_pairs,
)

# ----------------------------- config -----------------------------
OUT_DIR = "/mnt/c/tmp/pub_assets"  # hardcoded output root

# ----------------------------- local path helpers -----------------------------

def _is_s3(path: str) -> bool:
    return str(path).startswith("s3://")

def _join(base: str, *parts: str) -> str:
    return posixpath.join(base, *parts) if _is_s3(base) else os.path.join(base, *parts)

def _ensure_parent_dir_local(path: str):
    if _is_s3(path):
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def _to_duckdb_path(path: str) -> str:
    return path if _is_s3(path) else Path(path).as_posix()

# ----------------------------- DuckDB setup (httpfs + views) -----------------------------

def _ensure_httpfs(con: duckdb.DuckDBPyConnection, aws_region: str | None):
    con.execute("INSTALL httpfs; LOAD httpfs;")
    if aws_region:
        con.execute(f"SET s3_region='{aws_region}';")

def _count_globs(con: duckdb.DuckDBPyConnection, globs: Sequence[str]) -> int:
    if not globs: return 0
    union_sql = " UNION ALL ".join([f"SELECT * FROM glob('{g}')" for g in globs])
    return con.execute(f"SELECT COUNT(*) FROM ({union_sql})").fetchone()[0]

def _read_parquet_list_sql(globs: Sequence[str]) -> str:
    if not globs: return "[]"
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

def _register_components(con, drained_globs, burned_globs, aws_region):
    _ensure_httpfs(con, aws_region)
    if _count_globs(con, drained_globs) == 0:
        raise RuntimeError(f"[drained] No Parquet files found under: {drained_globs}")
    if _count_globs(con, burned_globs) == 0:
        raise RuntimeError(f"[burned] No Parquet files found under: {burned_globs}")
    _make_component_view(con, "zs_drained", drained_globs, kind="drained")
    _make_component_view(con, "zs_burned",  burned_globs,  kind="burned")

# ----------------------------- CSV / image writers -----------------------------

def _copy_sql(con: duckdb.DuckDBPyConnection, sql: str, out_path: str):
    _ensure_parent_dir_local(out_path)
    out_path_escaped = _to_duckdb_path(out_path).replace("'", "''")
    con.execute(f"COPY ({sql}) TO '{out_path_escaped}' (FORMAT CSV, HEADER TRUE)")

def _write_csv_df(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, path: str):
    _ensure_parent_dir_local(path)
    tmp_name = f"df_{id(df)}"
    con.register(tmp_name, df)
    out = _to_duckdb_path(path).replace("'", "''")
    con.execute(f"COPY {tmp_name} TO '{out}' (FORMAT CSV, HEADER TRUE)")
    try:
        con.unregister(tmp_name)
    except Exception:
        pass

def _save_png(fig: plt.Figure, path: str, dpi: int = 300, width: float | None = None, height: float | None = None):
    if width and height:
        fig.set_size_inches(width, height)
    _ensure_parent_dir_local(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")

# ----------------------------- plotting helpers (local) -----------------------------

CLIMATE_ORDER = ["Boreal", "Temperate", "Tropical"]
CLIMATE_COLORS = {"Boreal": "#4575B4", "Temperate": "#FDB863", "Tropical": "#1A9850"}
PROCESS_ORDER = ["Drained", "Burned"]
PROCESS_COLORS = {"Drained": "#3E3753", "Burned": "#FB6A29"}

_LU_RECLASS_PATTERNS = [
    (r"^(oil[_\- ]?palm|oilpalm)$", "Oil Palm"),
    (r"^(short[_\- ]?rotation|long[_\- ]?rotation|plantation.*|planted.*|tree[_\- ]?crop.*)$", "Other plantation"),
    (r"^cropland.*$", "Cropland"),
    (r"^forest.*$", "Forest"),
    (r"^(grassland|pasture|rangeland).*$", "Grassland"),
    (r"^(settlement|built[_\- ]?up|urban).*$", "Settlement"),
    (r"^wetland.*$", "Wetland"),
    (r"^(extraction|peat[_\- ]?extraction|cutover).*$", "Extraction"),
    (r"^(otherland|other)$", "Otherland"),
]

import re as _re

def _titlecase_domain(s: str | None) -> str:
    if not s: return "Unspecified"
    t = s.strip().title()
    if t.startswith("Boreal"): return "Boreal"
    if t.startswith("Temperate"): return "Temperate"
    if t.startswith("Tropical"): return "Tropical"
    return t or "Unspecified"

def _country_label(df: pd.DataFrame) -> pd.Series:
    if "iso3" in df.columns:
        iso = df["iso3"].astype("string")
        return iso.where(iso.str.len().fillna(0) > 0, df["gadm_adm0"].astype(str))
    return df["gadm_adm0"].astype(str)

def _pivot_wide(df_long: pd.DataFrame, value_col: str, index_col: str) -> pd.DataFrame:
    wide = (
        df_long
        .pivot_table(index=index_col, columns="Climate", values=value_col,
                     aggfunc="sum", fill_value=0.0, observed=False)
        .reindex(columns=CLIMATE_ORDER, fill_value=0.0)
        .reset_index()
    )
    return wide

def _stacked_column_by_category(
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
    else:
        ax.legend(ncol=min(len(category_order), 4), loc="upper left",
                  bbox_to_anchor=(0.0, 1.10), frameon=False, handlelength=1.6, columnspacing=1.2)
    ax.set_ylim(0, y_max)
    pad = y_max * 0.015
    for xpos, total in zip(range(len(totals)), totals):
        ax.text(xpos, total + pad, f"{total:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88 if not legend_above else 0.86))
    return fig

def _reclass_landuse(s: str | None) -> str:
    key = "other" if s is None else _re.sub(r"[\s\-]+", "_", s.strip().lower())
    for pat, lbl in _LU_RECLASS_PATTERNS:
        if _re.match(pat, key): return lbl
    return (s.strip().replace("_", " ").title() if s else "Otherland")

def _aggregate_landuse(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df = df.copy()
    df["Climate"] = df["climate_domain"].apply(_titlecase_domain)
    df = df[df["Climate"].isin(CLIMATE_ORDER)]
    df["LandUse"] = df["emissions_state"].apply(_reclass_landuse)
    df["LandUse"] = df["LandUse"].str.replace("_", " ", regex=False)
    out = (
        df.groupby(["LandUse", "Climate"], as_index=False, observed=False)[value_col]
          .sum()
          .sort_values([value_col], ascending=False)
    )
    # order by total
    totals = out.groupby("LandUse", observed=False)[value_col].sum().sort_values(ascending=False).index.tolist()
    out["LandUse"] = pd.Categorical(out["LandUse"], totals, ordered=True)
    return out.sort_values(["LandUse", "Climate"])

def _stacked_hbar(df_long: pd.DataFrame, value_col: str, xlabel: str) -> plt.Figure:
    df = df_long.copy()
    df["Climate"] = pd.Categorical(df["Climate"], CLIMATE_ORDER, ordered=True)
    wide = (
        df.pivot_table(index="LandUse", columns="Climate", values=value_col,
                       aggfunc="sum", fill_value=0.0, observed=False)
          .reindex(columns=CLIMATE_ORDER, fill_value=0.0)
          .sort_values(by=CLIMATE_ORDER, ascending=False)
    )
    totals = wide.sum(axis=1).values
    order = list(wide.sum(axis=1).sort_values(ascending=False).index)
    wide = wide.reindex(order)

    x_max = float(max(totals)) if len(totals) else 1.0
    right_pad = x_max * 0.08
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
    ax.set_xlim(0, x_max + right_pad)
    for y, total in zip(range(len(totals)), totals):
        ax.text(total + (x_max * 0.01), y, f"{total:.2f}", ha="left", va="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig

def _hbar_two_series(labels: list[str], left_vals: list[float], right_vals: list[float],
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
    ax.set_xlabel(xlabel); ax.set_ylabel("")
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.10),
              frameon=False, handlelength=1.6, columnspacing=1.2)
    right_pad = x_max * 0.08; ax.set_xlim(0, x_max + right_pad)
    for yy, tot in zip(y, tots):
        ax.text(tot + (x_max * 0.01), yy, f"{tot:.2f}", ha="left", va="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig

def _barh_single(labels: list[str], values: list[float], xlabel: str, color: str) -> plt.Figure:
    order = sorted(range(len(labels)), key=lambda i: values[i], reverse=True)
    labs  = [labels[i] for i in order]
    vals  = [values[i] for i in order]
    y = list(range(len(labs)))
    height = max(3.0, 0.5 * len(labs) + 1.0)
    fig, ax = plt.subplots(figsize=(7.5, height))
    ax.barh(y, vals, color=color)
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis()
    ax.set_xlabel(xlabel); ax.set_ylabel("")
    x_max = max(vals) if vals else 1.0
    right_pad = x_max * 0.08; ax.set_xlim(0, x_max + right_pad)
    for yy, v in zip(y, vals):
        ax.text(v + (x_max * 0.01), yy, f"{v:.2f}", ha="left", va="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig

# ---------------------------------- CLI -----------------------------------

def _interval_folder_strings(years: Iterable[int]) -> List[str]:
    pairs: List[Tuple[int, int]] = build_interval_pairs(list(years))
    return [f"{s}_{e}" for (s, e) in pairs]

def _make_base_prefixes(model_version: str, run_name: str, run_date: str,
                        interval_folders: Iterable[str]) -> List[str]:
    return [build_output_parquet(model_version, run_name, run_date, interval).rstrip("/")
            for interval in interval_folders]

def _make_globs_for_components(base_prefixes: Sequence[str]) -> tuple[list[str], list[str]]:
    drained = [posixpath.join(bp, "drained", "*.parquet") for bp in base_prefixes]
    burned  = [posixpath.join(bp, "burned",  "*.parquet") for bp in base_prefixes]
    return drained, burned

def main(argv=None):
    p = argparse.ArgumentParser("Build publication tables and figures (PNG-only)")
    p.add_argument("--model_version", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--run_date", required=True)
    p.add_argument("--years", nargs="+", required=True)
    p.add_argument("--aws_region", default=None)
    p.add_argument("--adm0_lookup", default=None)
    p.add_argument("--topn", type=int, default=20)
    p.add_argument("--do-tables", type=int, default=1)
    p.add_argument("--do-figures", type=int, default=1)
    p.add_argument("--data-only", action="store_true")
    args = p.parse_args(argv)

    years = [int(y) for y in args.years]
    interval_folders = _interval_folder_strings(years)
    base_prefixes = _make_base_prefixes(args.model_version, args.run_name, args.run_date, interval_folders)
    drained_globs, burned_globs = _make_globs_for_components(base_prefixes)

    con = duckdb.connect()
    _register_components(con, drained_globs, burned_globs, aws_region=args.aws_region)
    pc.register_state_context_views(con)

    # Optional ISO lookup
    if args.adm0_lookup:
        con.execute(f"""
            CREATE OR REPLACE VIEW adm0_lookup AS
            SELECT * FROM read_csv_auto('{args.adm0_lookup}', header=1);
        """)

    # Inventory period labels
    pairs = build_interval_pairs(list(years))
    period_labels = {e: f"{s}-{e}" for (s, e) in pairs}
    x_order = [period_labels[y] for y in years]
    inv_col = "Inventory Period"
    n_periods = len(years)

    # -------------------- Tables --------------------
    if args.do_tables:
        out_tables_dir = OUT_DIR
        _copy_sql(con, pc.table_by_country_period_sql(with_lookup=bool(args.adm0_lookup)),
                  _join(out_tables_dir, "by_country_period.csv"))
        _copy_sql(con, pc.table_by_drained_state_sql(),
                  _join(out_tables_dir, "by_drained_state_period.csv"))
        _copy_sql(con, pc.table_by_burned_state_sql(),
                  _join(out_tables_dir, "by_burned_state_period.csv"))
        _copy_sql(con, pc.table_by_country_drained_state_sql(with_lookup=bool(args.adm0_lookup)),
                  _join(out_tables_dir, "by_country_drained_state_period.csv"))
        _copy_sql(con, pc.table_by_country_burned_state_sql(with_lookup=bool(args.adm0_lookup)),
                  _join(out_tables_dir, "by_country_burned_state_period.csv"))
        _copy_sql(con, pc.table_topn_country_sql("drained", args.topn, bool(args.adm0_lookup)),
                  _join(out_tables_dir, f"top{args.topn}_by_country_drained.csv"))
        _copy_sql(con, pc.table_topn_country_sql("burned", args.topn, bool(args.adm0_lookup)),
                  _join(out_tables_dir, f"top{args.topn}_by_country_burned.csv"))

    # -------------------- Figures --------------------
    if args.do_figures:

        # --- Drained by climate (per period) ---
        df_d = con.execute(pc.sql_drained_by_climate()).df()
        df_d["Climate"] = df_d["climate_domain"].apply(_titlecase_domain)
        df_d = df_d[df_d["Climate"].isin(CLIMATE_ORDER)]
        df_d = df_d.rename(columns={"interval_end": "Year"})
        df_d[inv_col] = df_d["Year"].map(period_labels)
        d_long = df_d[[inv_col, "Climate", "drained_GtCO2e"]]
        _write_csv_df(con, d_long, _join(OUT_DIR, "figures", "data", "global_drained_climate_long.csv"))
        _write_csv_df(con, _pivot_wide(d_long, "drained_GtCO2e", inv_col),
                      _join(OUT_DIR, "figures", "data", "global_drained_climate_wide.csv"))
        if not args.data_only:
            fig = _stacked_column_by_category(
                d_long, inv_col, "Climate", "drained_GtCO2e",
                CLIMATE_ORDER, CLIMATE_COLORS,
                xlabel="Inventory Period", ylabel="Annual Emissions (Gt CO₂e/year)"
            )
            _save_png(fig, _join(OUT_DIR, "figures", "global_drained_climate_column.png"), dpi=300, width=7.5, height=4.5)

        # --- Burned by climate (per period) ---
        df_b = con.execute(pc.sql_burned_by_climate()).df()
        df_b["Climate"] = df_b["climate_domain"].apply(_titlecase_domain)
        df_b = df_b[df_b["Climate"].isin(CLIMATE_ORDER)]
        df_b = df_b.rename(columns={"interval_end": "Year"})
        df_b[inv_col] = df_b["Year"].map(period_labels)
        b_long = df_b[[inv_col, "Climate", "burned_GtCO2e"]]
        _write_csv_df(con, b_long, _join(OUT_DIR, "figures", "data", "global_burn_climate_long.csv"))
        _write_csv_df(con, _pivot_wide(b_long, "burned_GtCO2e", inv_col),
                      _join(OUT_DIR, "figures", "data", "global_burn_climate_wide.csv"))
        if not args.data_only:
            fig = _stacked_column_by_category(
                b_long, inv_col, "Climate", "burned_GtCO2e",
                CLIMATE_ORDER, CLIMATE_COLORS,
                xlabel="Inventory Period", ylabel="Annual Emissions (Gt CO₂e/year)"
            )
            _save_png(fig, _join(OUT_DIR, "figures", "global_burn_climate_column.png"), dpi=300, width=7.5, height=4.5)

        # --- Drained LU × Climate (avg annual) ---
        d_lu_raw = con.execute(pc.sql_drained_landuse_climate_avgs(n_periods)).df()
        d_lu = _aggregate_landuse(d_lu_raw, "drained_avg_GtCO2e_per_yr")
        _write_csv_df(con, d_lu[["LandUse", "Climate", "drained_avg_GtCO2e_per_yr"]],
                      _join(OUT_DIR, "figures", "data", "drained_landuse_climate_long.csv"))
        wide = (
            d_lu.pivot_table(index="LandUse", columns="Climate", values="drained_avg_GtCO2e_per_yr",
                             aggfunc="sum", fill_value=0.0, observed=False)
                .reindex(columns=CLIMATE_ORDER, fill_value=0.0).reset_index()
        )
        _write_csv_df(con, wide, _join(OUT_DIR, "figures", "data", "drained_landuse_climate_wide.csv"))
        if not args.data_only:
            fig = _stacked_hbar(d_lu, "drained_avg_GtCO2e_per_yr", xlabel="Average Annual Emissions (Gt CO₂e/year)")
            _save_png(fig, _join(OUT_DIR, "figures", "drained_landuse_climate_bar.png"), dpi=300)

        # --- Burned LU × Climate (avg annual) ---
        b_lu_raw = con.execute(pc.sql_burned_landuse_climate_avgs(n_periods)).df()
        b_lu = _aggregate_landuse(b_lu_raw, "burned_avg_GtCO2e_per_yr")
        _write_csv_df(con, b_lu[["LandUse", "Climate", "burned_avg_GtCO2e_per_yr"]],
                      _join(OUT_DIR, "figures", "data", "burned_landuse_climate_long.csv"))
        wide = (
            b_lu.pivot_table(index="LandUse", columns="Climate", values="burned_avg_GtCO2e_per_yr",
                             aggfunc="sum", fill_value=0.0, observed=False)
                .reindex(columns=CLIMATE_ORDER, fill_value=0.0).reset_index()
        )
        _write_csv_df(con, wide, _join(OUT_DIR, "figures", "data", "burned_landuse_climate_wide.csv"))
        if not args.data_only:
            fig = _stacked_hbar(b_lu, "burned_avg_GtCO2e_per_yr", xlabel="Average Annual Emissions (Gt CO₂e/year)")
            _save_png(fig, _join(OUT_DIR, "figures", "burned_landuse_climate_bar.png"), dpi=300)

        # --- Top-N peat area (latest) ---
        latest_year = max(years)
        df_area = con.execute(pc.sql_topn_peat_area_comp_latest(latest_year, args.topn, bool(args.adm0_lookup))).df()
        df_area["label"] = _country_label(df_area)
        _write_csv_df(con,
                      df_area[["label", "drained_area_mha", "undrained_area_mha", "total_area_mha"]]
                            .rename(columns={"label": "iso3_or_code"}),
                      _join(OUT_DIR, "figures", "data", "top_10_country_peat_area.csv"))
        if not args.data_only:
            fig = _hbar_two_series(
                labels=df_area["label"].tolist(),
                left_vals=df_area["drained_area_mha"].tolist(),
                right_vals=df_area["undrained_area_mha"].tolist(),
                xlabel="Total Area (million ha)",
                legends=("Drained peat area", "Undrained peat area"),
                colors=("#FB6A29", "#3E3753"),
            )
            _save_png(fig, _join(OUT_DIR, "figures", "top_10_country_peat_area_bar.png"), dpi=300)

        # --- Top-N total emissions split (avg annual) ---
        df_emsplit = con.execute(pc.sql_topn_total_emissions_split_avg(args.topn, bool(args.adm0_lookup), n_periods)).df()
        df_emsplit["label"] = _country_label(df_emsplit)
        _write_csv_df(con,
                      df_emsplit[["label", "burned_avg_GtCO2e_per_yr", "drained_avg_GtCO2e_per_yr", "total_avg_GtCO2e_per_yr"]]
                                  .rename(columns={"label": "iso3_or_code"}),
                      _join(OUT_DIR, "figures", "data", "top_10_country_total_emissions.csv"))
        if not args.data_only:
            fig = _hbar_two_series(
                labels=df_emsplit["label"].tolist(),
                left_vals=df_emsplit["burned_avg_GtCO2e_per_yr"].tolist(),
                right_vals=df_emsplit["drained_avg_GtCO2e_per_yr"].tolist(),
                xlabel="Average Annual Emissions (Gt CO₂e/year)",
                legends=("Average burned emissions", "Average drained emissions"),
                colors=("#FB6A29", "#3E3753"),
            )
            _save_png(fig, _join(OUT_DIR, "figures", "top_10_country_total_emissions_bar.png"), dpi=300)

        # --- Global totals over time (stacked drained+burned) ---
        df_gt = con.execute(pc.sql_global_totals_by_period_long()).df()
        df_gt[inv_col] = df_gt["interval_end"].map(period_labels)
        gt_long = df_gt[[inv_col, "component", "GtCO2e"]].rename(columns={"component": "Component"})
        _write_csv_df(con, gt_long, _join(OUT_DIR, "figures", "data", "global_total_emissions_long.csv"))
        gt_wide = (
            gt_long.pivot_table(index=inv_col, columns="Component", values="GtCO2e",
                                aggfunc="sum", fill_value=0.0, observed=False)
                    .reset_index()
        )
        _write_csv_df(con, gt_wide, _join(OUT_DIR, "figures", "data", "global_total_emissions_wide.csv"))
        if not args.data_only:
            fig = _stacked_column_by_category(
                gt_long.rename(columns={"component": "Component"}),
                inv_col, "Component", "GtCO2e",
                PROCESS_ORDER, PROCESS_COLORS,
                xlabel="Inventory Period", ylabel="Annual Emissions (Gt CO₂e/year)"
            )
            _save_png(fig, _join(OUT_DIR, "figures", "global_total_emissions_column.png"), dpi=300)

        # --- Top-N drained avg annual ---
        df_topd = con.execute(pc.sql_topn_avg_component_emissions("drained", args.topn, bool(args.adm0_lookup), n_periods)).df()
        df_topd["label"] = _country_label(df_topd)
        _write_csv_df(con,
                      df_topd[["label", "drained_avg_GtCO2e_per_yr"]].rename(columns={"label": "iso3_or_code"}),
                      _join(OUT_DIR, "figures", "data", "top_10_country_drained_avg_emissions.csv"))
        if not args.data_only:
            fig = _barh_single(
                labels=df_topd["label"].tolist(),
                values=df_topd["drained_avg_GtCO2e_per_yr"].tolist(),
                xlabel="Average Annual Emissions (Gt CO₂e/year)",
                color=PROCESS_COLORS["Drained"],
            )
            _save_png(fig, _join(OUT_DIR, "figures", "top_10_country_drained_avg_emissions_bar.png"), dpi=300)

        # --- Top-N burned avg annual ---
        df_topb = con.execute(pc.sql_topn_avg_component_emissions("burned", args.topn, bool(args.adm0_lookup), n_periods)).df()
        df_topb["label"] = _country_label(df_topb)
        _write_csv_df(con,
                      df_topb[["label", "burned_avg_GtCO2e_per_yr"]].rename(columns={"label": "iso3_or_code"}),
                      _join(OUT_DIR, "figures", "data", "top_10_country_burned_avg_emissions.csv"))
        if not args.data_only:
            fig = _barh_single(
                labels=df_topb["label"].tolist(),
                values=df_topb["burned_avg_GtCO2e_per_yr"].tolist(),
                xlabel="Average Annual Emissions (Gt CO₂e/year)",
                color=PROCESS_COLORS["Burned"],
            )
            _save_png(fig, _join(OUT_DIR, "figures", "top_10_country_burned_avg_emissions_bar.png"), dpi=300)

        # --- Drained emissions intensity (t CO2e/ha/yr), Top-N ---
        df_int = con.execute(pc.sql_country_emissions_intensity_avg(n_periods, bool(args.adm0_lookup), args.topn, 10000.0)).df()
        df_int["label"] = _country_label(df_int)
        _write_csv_df(con,
                      df_int[["label", "intensity_tCO2e_per_ha_yr", "total_avg_GtCO2e_per_yr", "latest_drained_area_mha"]]
                           .rename(columns={"label": "iso3_or_code"}),
                      _join(OUT_DIR, "figures", "data", "top_10_country_emissions_intensity.csv"))
        if not args.data_only:
            fig = _barh_single(
                labels=df_int["label"].tolist(),
                values=df_int["intensity_tCO2e_per_ha_yr"].tolist(),
                xlabel="Drained Emissions Intensity (t CO₂e/ha/year)",
                color="#5C6BC0",
            )
            _save_png(fig, _join(OUT_DIR, "figures", "top_10_country_emissions_intensity_bar.png"), dpi=300)

        # --- Drained LU share by climate (100% stacked; drained only) ---
        d_share_src = con.execute(pc.sql_drained_landuse_climate_avgs(n_periods)).df()
        d_share_src["Climate"] = d_share_src["climate_domain"].apply(_titlecase_domain)
        d_share_src = d_share_src[d_share_src["Climate"].isin(CLIMATE_ORDER)]
        d_lu = _aggregate_landuse(d_share_src, "drained_avg_GtCO2e_per_yr")
        share = d_lu.copy()
        share["total_climate"] = share.groupby("Climate", observed=False)["drained_avg_GtCO2e_per_yr"].transform("sum")
        share["Share (%)"] = (share["drained_avg_GtCO2e_per_yr"] / share["total_climate"].replace({0: pd.NA})) * 100.0
        _write_csv_df(con, share[["LandUse", "Climate", "Share (%)"]],
                      _join(OUT_DIR, "figures", "data", "drained_landuse_share_by_climate_100pct.csv"))
        if not args.data_only:
            # order land-uses by total share descending
            lu_order = (
                share.groupby("LandUse", observed=False)["Share (%)"]
                     .sum().sort_values(ascending=False).index.tolist()
            )
            fig = _stacked_column_by_category(
                share, index_col="Climate", category_col="LandUse", value_col="Share (%)",
                category_order=lu_order, color_map={}, xlabel="Climate Domain",
                ylabel="Share of Avg Annual Emissions (%)", legend_above=True
            )
            _save_png(fig, _join(OUT_DIR, "figures", "drained_landuse_share_by_climate_100pct.png"), dpi=300)

        # --- Country scatter: avg annual vs avg drained area ---
        df_sc = con.execute(pc.sql_country_emissions_vs_area_avg(n_periods, bool(args.adm0_lookup))).df()
        df_sc["label"] = _country_label(df_sc)
        _write_csv_df(con,
                      df_sc[["label", "total_avg_GtCO2e_per_yr", "avg_drained_area_mha"]]
                           .rename(columns={"label": "iso3_or_code"}),
                      _join(OUT_DIR, "figures", "data", "country_emissions_vs_area.csv"))
        if not args.data_only:
            fig, ax = plt.subplots(figsize=(7.5, 5.4))
            ax.scatter(df_sc["avg_drained_area_mha"], df_sc["total_avg_GtCO2e_per_yr"], alpha=0.8)
            for xi, yi, lab in zip(df_sc["avg_drained_area_mha"], df_sc["total_avg_GtCO2e_per_yr"], df_sc["label"]):
                ax.annotate(str(lab), (xi, yi), xytext=(4, 3), textcoords="offset points", fontsize=8)
            ax.set_xlabel("Avg Drained Peat Area (Mha)")
            ax.set_ylabel("Avg Annual Emissions (Gt CO₂e/year)")
            fig.tight_layout()
            _save_png(fig, _join(OUT_DIR, "figures", "country_emissions_vs_area_scatter.png"), dpi=300)

    print("Assets written to:", OUT_DIR)

if __name__ == "__main__":
    main()
