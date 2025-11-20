# -*- coding: utf-8 -*-
"""
Build publication **tables** and **figures** from organic-soils zonal statistics (driver).

Design:
- PNG-only outputs
- Utilities in `pub_common` (this script imports and uses them)
- This driver alone does DuckDB setup + light view registration:
    * read Parquet to create: zs_drained, zs_burned
    * create tiny in-memory lookups: drained_state_ctx, burned_state_ctx
    * create adm0_lookup ONLY if --adm0_lookup is provided (optional)
- No line charts.

Outputs mirror the main drivers and are organized under::

    /mnt/c/tmp/pub_assets/version_<model_version>/<run_name>/<run_date>/

Usage examples:
  cd /mnt/c/gis/git/AFOLU_GHG_flux_model

  python -m src.scripts.zonal_statistics.pub_assets \
    --model_version 0_9_7 \
    --run_name ogh_sensitivity_500m_10 \
    --run_date 20251118 \
    --years 2005 2010 2015 2020 2024 \
    --topn 10
"""

from __future__ import annotations

import argparse
import os
import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Optional, Callable, Mapping

import duckdb
import pandas as pd
import matplotlib.pyplot as plt

import src.scripts.zonal_statistics.pub_common as pc
from src.scripts.zonal_statistics import zonal_constants as zc
from src.scripts.zonal_statistics.run_zonal_stats import (
    build_output_parquet,
    build_interval_pairs,
)

# ----------------------------- config -----------------------------
OUT_DIR_ROOT = "/mnt/c/tmp/pub_assets"  # hardcoded output root
OUT_DIR = OUT_DIR_ROOT

# ----------------------------- path helpers -----------------------------

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


def build_output_dir(model_version: str, run_name: str, run_date: str) -> str:
    """Return the publication output directory for a run."""

    return _join(OUT_DIR_ROOT, f"version_{model_version}", run_name, run_date)

# ----------------------------- DuckDB setup -----------------------------

def _ensure_httpfs(con: duckdb.DuckDBPyConnection, aws_region: Optional[str]):
    con.execute("INSTALL httpfs; LOAD httpfs;")
    if aws_region:
        con.execute(f"SET s3_region='{aws_region}';")

def _count_globs(con: duckdb.DuckDBPyConnection, globs: Sequence[str]) -> int:
    if not globs:
        return 0
    union_sql = " UNION ALL ".join([f"SELECT * FROM glob('{g}')" for g in globs])
    return con.execute(f"SELECT COUNT(*) FROM ({union_sql})").fetchone()[0]

def _read_parquet_list_sql(globs: Sequence[str]) -> str:
    if not globs:
        return "[]"
    items = ", ".join([f"'{g}'" for g in globs])
    return f"[{items}]"

def _make_component_view(con: duckdb.DuckDBPyConnection, name: str,
                         globs: Sequence[str], kind: str):
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

def _register_components(con: duckdb.DuckDBPyConnection,
                         drained_globs: Sequence[str], burned_globs: Sequence[str],
                         aws_region: Optional[str]):
    _ensure_httpfs(con, aws_region)
    if _count_globs(con, drained_globs) == 0:
        raise RuntimeError(f"[drained] No Parquet files found under: {drained_globs}")
    if _count_globs(con, burned_globs) == 0:
        raise RuntimeError(f"[burned] No Parquet files found under: {burned_globs}")
    _make_component_view(con, "zs_drained", drained_globs, kind="drained")
    _make_component_view(con, "zs_burned",  burned_globs,  kind="burned")

# ----------------------------- Lookup registrations (driver-only) -----------------------------

def _register_state_context_views(con: duckdb.DuckDBPyConnection):
    """Small in-memory lookups from zonal_constants: drained_state_ctx, burned_state_ctx."""
    # drained
    d_rows = []
    for key, meaning in zc.DRAINED_STATE_NODE_MEANINGS.items():
        climate_domain = None
        drained_state = None
        emissions_state = None
        if "__" in meaning:
            left, right = meaning.split("__", 1)
            drained_state = left
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
    con.register("drained_state_ctx", pd.DataFrame(d_rows))

    # burned
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
    con.register("burned_state_ctx", pd.DataFrame(b_rows))

def _ensure_adm0_lookup(con: duckdb.DuckDBPyConnection, csv_path: Optional[str]) -> bool:
    """Register adm0_lookup view from CSV (if provided) or pycountry fallbacks."""

    if csv_path:
        con.execute(f"""
            CREATE OR REPLACE VIEW adm0_lookup AS
            SELECT * FROM read_csv_auto('{csv_path}', header=1);
        """)
        return True

    df = pc.build_adm0_lookup_df()
    if df.empty:
        return False
    con.register("adm0_lookup", df)
    return True

# ----------------------------- Writers -----------------------------

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

# ----------------------------- Plot metadata & helpers -----------------------------

# Choose a palette globally (applies to all climate charts)
pc.set_climate_palette("brewer_dark2")  # or "brewer_set2" / "okabe_ito"

@dataclass(frozen=True)
class ComponentPlotMeta:
    component: str
    sql_builder: Callable[[], str]
    value_column: str
    color_set: Mapping[str, str]
    file_stub: str


CLIMATE_COMPONENT_PLOTS: tuple[ComponentPlotMeta, ...] = (
    ComponentPlotMeta(
        component="drained",
        sql_builder=pc.sql_drained_by_climate,
        value_column="drained_GtCO2e",
        color_set=pc.CLIMATE_COLORS,
        file_stub="global_drained_climate",
    ),
    ComponentPlotMeta(
        component="burned",
        sql_builder=pc.sql_burned_by_climate,
        value_column="burned_GtCO2e",
        color_set=pc.CLIMATE_COLORS,
        file_stub="global_burn_climate",
    ),
)


def _build_component_climate_plot(
    con: duckdb.DuckDBPyConnection,
    meta: ComponentPlotMeta,
    inventory_col: str,
    period_labels: Mapping[int, str],
    data_only: bool,
):
    df = con.execute(meta.sql_builder()).df()
    df["Climate"] = df["climate_domain"].apply(pc.titlecase_domain)
    df = df[df["Climate"].isin(pc.CLIMATE_ORDER)]
    df = df.rename(columns={"interval_end": "Year"})
    df[inventory_col] = df["Year"].map(period_labels)
    long_df = df[[inventory_col, "Climate", meta.value_column]]

    data_dir = _join(OUT_DIR, "figures", "data")
    _write_csv_df(con, long_df, _join(data_dir, f"{meta.file_stub}_long.csv"))
    _write_csv_df(
        con,
        pc.pivot_wide(long_df, meta.value_column, inventory_col),
        _join(data_dir, f"{meta.file_stub}_wide.csv"),
    )

    if data_only:
        return

    fig = pc.stacked_column_by_category(
        long_df,
        inventory_col,
        "Climate",
        meta.value_column,
        pc.CLIMATE_ORDER,
        meta.color_set,
        xlabel="Inventory Period",
        ylabel="Annual Emissions (Gt CO₂e/year)",
        width=7.5,
        height=4.5,
    )
    _save_png(fig, _join(OUT_DIR, "figures", f"{meta.file_stub}_column.png"), dpi=300)

# ----------------------------- Parquet path builders -----------------------------

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

# ----------------------------- CLI ----------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser("Build publication tables and figures (PNG-only)")
    p.add_argument("--model_version", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--run_date", required=True)
    p.add_argument("--years", nargs="+", required=True)
    p.add_argument("--aws_region", default=None)
    p.add_argument("--adm0_lookup", default=None, help="Optional CSV with gadm_adm0,country,iso3")
    p.add_argument("--topn", type=int, default=20)
    p.add_argument("--do-tables", type=int, default=1)
    p.add_argument("--do-figures", type=int, default=1)
    p.add_argument("--data-only", action="store_true")
    args = p.parse_args(argv)

    global OUT_DIR
    OUT_DIR = build_output_dir(args.model_version, args.run_name, args.run_date)

    years = [int(y) for y in args.years]
    interval_folders = _interval_folder_strings(years)
    base_prefixes = _make_base_prefixes(args.model_version, args.run_name, args.run_date, interval_folders)
    drained_globs, burned_globs = _make_globs_for_components(base_prefixes)

    con = duckdb.connect()
    _register_components(con, drained_globs, burned_globs, aws_region=args.aws_region)
    _register_state_context_views(con)
    have_lookup = _ensure_adm0_lookup(con, args.adm0_lookup)

    # Inventory period labels
    pairs = build_interval_pairs(list(years))
    period_labels = {e: f"{s}-{e}" for (s, e) in pairs}
    inv_col = "Inventory Period"
    n_periods = len(years)

    # -------------------- Tables --------------------
    if args.do_tables:
        out_tables_dir = OUT_DIR
        _copy_sql(con, pc.table_by_country_period_sql(with_lookup=have_lookup),
                  _join(out_tables_dir, "by_country_period.csv"))
        _copy_sql(con, pc.table_by_drained_state_sql(),
                  _join(out_tables_dir, "by_drained_state_period.csv"))
        _copy_sql(con, pc.table_by_burned_state_sql(),
                  _join(out_tables_dir, "by_burned_state_period.csv"))
        _copy_sql(con, pc.table_by_country_drained_state_sql(with_lookup=have_lookup),
                  _join(out_tables_dir, "by_country_drained_state_period.csv"))
        _copy_sql(con, pc.table_by_country_burned_state_sql(with_lookup=have_lookup),
                  _join(out_tables_dir, "by_country_burned_state_period.csv"))
        _copy_sql(con, pc.table_topn_country_sql("drained", args.topn, have_lookup),
                  _join(out_tables_dir, f"top{args.topn}_by_country_drained.csv"))
        _copy_sql(con, pc.table_topn_country_sql("burned", args.topn, have_lookup),
                  _join(out_tables_dir, f"top{args.topn}_by_country_burned.csv"))

    # -------------------- Figures --------------------
    if args.do_figures:

        # 1) Drained & burned by climate (per period)
        for meta in CLIMATE_COMPONENT_PLOTS:
            _build_component_climate_plot(con, meta, inv_col, period_labels, args.data_only)

        # C) Total emissions (drained+burned) by climate × period
        df_tot = con.execute(pc.sql_total_by_climate()).df()
        df_tot["Climate"] = df_tot["climate_domain"].apply(pc.titlecase_domain)
        df_tot = df_tot[df_tot["Climate"].isin(pc.CLIMATE_ORDER)].rename(columns={"interval_end": "Year"})
        df_tot[inv_col] = df_tot["Year"].map(period_labels)
        tot_long = df_tot[[inv_col, "Climate", "total_GtCO2e"]]

        _write_csv_df(con, tot_long, _join(OUT_DIR, "figures", "data", "global_total_by_climate_long.csv"))
        _write_csv_df(con, pc.pivot_wide(tot_long, "total_GtCO2e", inv_col),
                      _join(OUT_DIR, "figures", "data", "global_total_by_climate_wide.csv"))
        if not args.data_only:
            fig = pc.stacked_column_by_category(
                tot_long, inv_col, "Climate", "total_GtCO2e",
                pc.CLIMATE_ORDER, pc.CLIMATE_COLORS,
                xlabel="Inventory Period", ylabel="Annual Emissions (Gt CO₂e/year)"
            )
            _save_png(fig, _join(OUT_DIR, "figures", "global_total_by_climate_column.png"), dpi=300)

        # 2) Drained: Land Use × Climate (avg annual across selected periods)
        d_lu_raw = con.execute(pc.sql_drained_landuse_climate_avgs(n_periods)).df()
        d_lu = pc.aggregate_landuse(d_lu_raw, "drained_avg_GtCO2e_per_yr")
        _write_csv_df(con, d_lu[["LandUse", "Climate", "drained_avg_GtCO2e_per_yr"]],
                      _join(OUT_DIR, "figures", "data", "drained_landuse_climate_long.csv"))
        wide = (
            d_lu.pivot_table(index="LandUse", columns="Climate", values="drained_avg_GtCO2e_per_yr",
                             aggfunc="sum", fill_value=0.0, observed=False)
                .reindex(columns=pc.CLIMATE_ORDER, fill_value=0.0).reset_index()
        )
        _write_csv_df(con, wide, _join(OUT_DIR, "figures", "data", "drained_landuse_climate_wide.csv"))
        if not args.data_only:
            fig = pc.stacked_hbar(d_lu, "drained_avg_GtCO2e_per_yr",
                                  xlabel="Average Annual Emissions (Gt CO₂e/year)")
            _save_png(fig, _join(OUT_DIR, "figures", "drained_landuse_climate_bar.png"), dpi=300)

        # 3) Burned: Land Use × Climate (avg annual across selected periods)
        b_lu_raw = con.execute(pc.sql_burned_landuse_climate_avgs(n_periods)).df()
        b_lu = pc.aggregate_landuse(b_lu_raw, "burned_avg_GtCO2e_per_yr")
        _write_csv_df(con, b_lu[["LandUse", "Climate", "burned_avg_GtCO2e_per_yr"]],
                      _join(OUT_DIR, "figures", "data", "burned_landuse_climate_long.csv"))
        wide = (
            b_lu.pivot_table(index="LandUse", columns="Climate", values="burned_avg_GtCO2e_per_yr",
                             aggfunc="sum", fill_value=0.0, observed=False)
                .reindex(columns=pc.CLIMATE_ORDER, fill_value=0.0).reset_index()
        )
        _write_csv_df(con, wide, _join(OUT_DIR, "figures", "data", "burned_landuse_climate_wide.csv"))
        if not args.data_only:
            fig = pc.stacked_hbar(b_lu, "burned_avg_GtCO2e_per_yr",
                                  xlabel="Average Annual Emissions (Gt CO₂e/year)")
            _save_png(fig, _join(OUT_DIR, "figures", "burned_landuse_climate_bar.png"), dpi=300)

        # D) Component split within each climate (avg over selected periods)
        df_cs = con.execute(pc.sql_component_split_by_climate_avg(n_periods)).df()
        df_cs["Climate"] = df_cs["climate_domain"].apply(pc.titlecase_domain)
        df_cs = df_cs[df_cs["Climate"].isin(pc.CLIMATE_ORDER)]
        df_cs["Component"] = pd.Categorical(df_cs["component"], pc.PROCESS_ORDER, ordered=True)

        _write_csv_df(con, df_cs[["Climate", "Component", "avg_GtCO2e_per_yr"]],
                      _join(OUT_DIR, "figures", "data", "component_split_by_climate_avg.csv"))
        if not args.data_only:
            fig = pc.stacked_column_by_category(
                df_cs.rename(columns={"avg_GtCO2e_per_yr": "Value"}),
                index_col="Climate",
                category_col="Component",
                value_col="Value",
                category_order=pc.PROCESS_ORDER,
                color_map=pc.PROCESS_COLORS,
                xlabel="Climate",
                ylabel="Average Annual Emissions (Gt CO₂e/year)"
            )
            _save_png(fig, _join(OUT_DIR, "figures", "component_split_by_climate_bar.png"), dpi=300)

        # 4) Top-N by country: PEAT AREA split (latest interval only)
        latest_year = max(years)
        df_area = con.execute(pc.sql_topn_peat_area_comp_latest(latest_year, args.topn, have_lookup)).df()
        df_area["label"] = pc.country_label(df_area)
        _write_csv_df(con,
                      df_area[["label", "drained_area_mha", "undrained_area_mha", "total_area_mha"]]
                            .rename(columns={"label": "iso3_or_code"}),
                      _join(OUT_DIR, "figures", "data", "top_10_country_peat_area.csv"))
        if not args.data_only:
            fig = pc.hbar_two_series(
                labels=df_area["label"].tolist(),
                left_vals=df_area["drained_area_mha"].tolist(),
                right_vals=df_area["undrained_area_mha"].tolist(),
                xlabel="Total Area (million ha)",
                legends=("Drained peat area", "Undrained peat area"),
                colors=("#FB6A29", "#3E3753"),
            )
            _save_png(fig, _join(OUT_DIR, "figures", "top_10_country_peat_area_bar.png"), dpi=300)

        # 5) Top-N by country: AVERAGE annual TOTAL EMISSIONS split (drained + burned)
        df_emsplit = con.execute(pc.sql_topn_total_emissions_split_avg(args.topn, have_lookup, n_periods)).df()
        df_emsplit["label"] = pc.country_label(df_emsplit)
        _write_csv_df(con,
                      df_emsplit[["label", "burned_avg_GtCO2e_per_yr", "drained_avg_GtCO2e_per_yr", "total_avg_GtCO2e_per_yr"]]
                                  .rename(columns={"label": "iso3_or_code"}),
                      _join(OUT_DIR, "figures", "data", "top_10_country_total_emissions.csv"))
        if not args.data_only:
            fig = pc.hbar_two_series(
                labels=df_emsplit["label"].tolist(),
                left_vals=df_emsplit["burned_avg_GtCO2e_per_yr"].tolist(),
                right_vals=df_emsplit["drained_avg_GtCO2e_per_yr"].tolist(),
                xlabel="Average Annual Emissions (Gt CO₂e/year)",
                legends=("Average burned emissions", "Average drained emissions"),
                colors=("#FB6A29", "#3E3753"),
            )
            _save_png(fig, _join(OUT_DIR, "figures", "top_10_country_total_emissions_bar.png"), dpi=300)

        # 6) Global totals over time (stacked drained+burned)
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
            fig = pc.stacked_column_by_category(
                gt_long, inv_col, "Component", "GtCO2e",
                pc.PROCESS_ORDER, pc.PROCESS_COLORS,
                xlabel="Inventory Period", ylabel="Annual Emissions (Gt CO₂e/year)"
            )
            _save_png(fig, _join(OUT_DIR, "figures", "global_total_emissions_column.png"), dpi=300)

        # 7) Top-N average-annual by component (separate charts)
        df_topd = con.execute(pc.sql_topn_avg_component_emissions("drained", args.topn, have_lookup, n_periods)).df()
        df_topd["label"] = pc.country_label(df_topd)
        _write_csv_df(con,
                      df_topd[["label", "drained_avg_GtCO2e_per_yr"]].rename(columns={"label": "iso3_or_code"}),
                      _join(OUT_DIR, "figures", "data", "top_10_country_drained_avg_emissions.csv"))
        if not args.data_only:
            fig = pc.barh_single(
                labels=df_topd["label"].tolist(),
                values=df_topd["drained_avg_GtCO2e_per_yr"].tolist(),
                xlabel="Average Annual Emissions (Gt CO₂e/year)",
                color=pc.PROCESS_COLORS["Drained"],
            )
            _save_png(fig, _join(OUT_DIR, "figures", "top_10_country_drained_avg_emissions_bar.png"), dpi=300)

        df_topb = con.execute(pc.sql_topn_avg_component_emissions("burned", args.topn, have_lookup, n_periods)).df()
        df_topb["label"] = pc.country_label(df_topb)
        _write_csv_df(con,
                      df_topb[["label", "burned_avg_GtCO2e_per_yr"]].rename(columns={"label": "iso3_or_code"}),
                      _join(OUT_DIR, "figures", "data", "top_10_country_burned_avg_emissions.csv"))
        if not args.data_only:
            fig = pc.barh_single(
                labels=df_topb["label"].tolist(),
                values=df_topb["burned_avg_GtCO2e_per_yr"].tolist(),
                xlabel="Average Annual Emissions (Gt CO₂e/year)",
                color=pc.PROCESS_COLORS["Burned"],
            )
            _save_png(fig, _join(OUT_DIR, "figures", "top_10_country_burned_avg_emissions_bar.png"), dpi=300)

        # F) Drained emissions intensity by climate (avg over periods / latest drained area)
        df_ic = con.execute(pc.sql_drained_intensity_by_climate_avg(n_periods)).df()
        df_ic["Climate"] = df_ic["climate_domain"].apply(pc.titlecase_domain)
        df_ic = df_ic[df_ic["Climate"].isin(pc.CLIMATE_ORDER)]
        df_ic = df_ic.sort_values("intensity_tCO2e_per_ha_yr", ascending=False)

        _write_csv_df(con, df_ic[["Climate", "intensity_tCO2e_per_ha_yr"]],
                      _join(OUT_DIR, "figures", "data", "drained_intensity_by_climate.csv"))
        if not args.data_only:
            fig = pc.barh_single(
                labels=df_ic["Climate"].tolist(),
                values=df_ic["intensity_tCO2e_per_ha_yr"].tolist(),
                xlabel="Drained Emissions Intensity (t CO₂e/ha/year)",
                color="#5C6BC0",
                sort_desc=False,
            )
            _save_png(fig, _join(OUT_DIR, "figures", "drained_intensity_by_climate_bar.png"), dpi=300)

        # 9) Country scatter and land-use share figures removed per workflow update

    print("Assets written to:", OUT_DIR)

if __name__ == "__main__":
    main()