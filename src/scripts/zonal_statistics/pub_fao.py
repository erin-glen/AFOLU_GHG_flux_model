# -*- coding: utf-8 -*-
"""Build FAO-style comparison figures from zonal statistics and FAOSTAT.

This driver mirrors the pub_assets/pub_compare conventions but focuses on
cropland and grassland cuts of drained emissions for the most recent inventory
period *present in the FAO subset*.

It compares:
- Global drained peat area (Cropland + Grassland) against FAOSTAT "Area".
- Global drained emissions by gas (CO₂, N₂O) against FAOSTAT "Emissions (CO2)"
  and "Emissions (N2O)".
- Global land-use split of drained peat area (Cropland vs Grassland) against
  FAOSTAT area for "Cropland organic soils" and "Grassland organic soils".

Inputs
------
1) Zonal statistics FAO tiles (Parquet):
   <base>/<start>_<end>/drained/fao_stat/*.parquet
   with flux_type values like:
     - drained_co2_Mg_CO2...
     - drained_n2o_Mg_CO2e...
     - area__ha / area_ha

2) FAOSTAT CSV (drained organic soils domain "GV"):
   s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/fao_stat/
       FAOSTAT_data_en_11-22-2025.csv

Outputs
-------
PNG figures and CSVs under:
  /mnt/c/tmp/pub_fao/version_<model_version>/<run_name>/<run_date>/

CSV tables under figures/data/ include both Model and FAOSTAT rows, with a
Source column.
"""

from __future__ import annotations

import argparse
import posixpath
from typing import Iterable, List, Sequence, Tuple, Optional

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.scripts.zonal_statistics.pub_common as pc
import src.scripts.zonal_statistics.pub_assets as pa
from src.scripts.zonal_statistics.run_zonal_stats import (
    build_interval_pairs,
    build_output_parquet,
)

# ----------------------------- config -----------------------------

OUT_DIR_ROOT = "/mnt/c/tmp/pub_fao"
OUT_DIR = OUT_DIR_ROOT

# Default FAOSTAT CSV (S3)
DEFAULT_FAO_CSV = (
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/fao_stat/"
    "FAOSTAT_data_en_11-22-2025.csv"
)

# ----------------------------- path helpers -----------------------------

_join = pa._join
_save_png = pa._save_png
_write_csv_df = pa._write_csv_df


def build_output_dir(model_version: str, run_name: str, run_date: str) -> str:
    return _join(OUT_DIR_ROOT, f"version_{model_version}", run_name, run_date)


def _interval_folder_strings(years: Iterable[int]) -> List[str]:
    pairs: List[Tuple[int, int]] = build_interval_pairs(list(years))
    return [f"{s}_{e}" for (s, e) in pairs]


def _make_base_prefixes(
    model_version: str,
    run_name: str,
    run_date: str,
    interval_folders: Iterable[str],
) -> List[str]:
    return [
        build_output_parquet(model_version, run_name, run_date, interval).rstrip("/")
        for interval in interval_folders
    ]


def _make_drained_globs(base_prefixes: Sequence[str]) -> list[str]:
    # FAO exports are nested under a "fao_stat" subfolder instead of the
    # top-level drained directory used by other publication scripts.
    return [
        posixpath.join(bp, "drained", "fao_stat", "*.parquet")
        for bp in base_prefixes
    ]


def _register_drained(
    con: duckdb.DuckDBPyConnection,
    drained_globs: Sequence[str],
    aws_region: Optional[str],
) -> None:
    """Create zs_drained view from FAO Parquet outputs."""
    pa._ensure_httpfs(con, aws_region)
    drained_count = pa._count_globs(con, drained_globs)
    print(f"[drained] Searching {len(drained_globs)} globs; matched {drained_count} files")
    if drained_count == 0:
        raise RuntimeError(f"[drained] No Parquet files found under: {drained_globs}")
    pa._make_component_view(con, "zs_drained", drained_globs, kind="drained")


# ----------------------------- Queries -----------------------------


def sql_components_latest() -> str:
    """
    Aggregate drained N2O, CO2 and peat area by emissions_state, then keep only
    the *latest* interval_end present in the data.

    Critical bit: match gas flux types using prefix so that names like
    `drained_co2_Mg_CO2` and `drained_n2o_Mg_CO2e` are picked up.
    """
    return """
    WITH base AS (
      SELECT
        z.interval_end,
        COALESCE(ctx.emissions_state, 'Unspecified') AS emissions_state,
        LOWER(z.flux_type) AS flux_type,
        z.value
      FROM zs_drained z
      LEFT JOIN drained_state_ctx AS ctx
        ON (z.drained_state_meaning = ctx.meaning)
        OR (RPAD(CAST(z.drained_state_nodes AS VARCHAR), 8, '0') = ctx.key)
    ),
    agg AS (
      SELECT
        interval_end,
        emissions_state,
        -- capture drained_n2o_Mg_CO2e, drained_n2o*, etc.
        SUM(
          CASE WHEN flux_type LIKE 'drained_n2o%%'
               THEN value ELSE 0 END
        ) AS drained_n2o,
        -- capture drained_co2_Mg_CO2, drained_co2*, etc.
        SUM(
          CASE WHEN flux_type LIKE 'drained_co2%%'
               THEN value ELSE 0 END
        ) AS drained_co2,
        -- peat area (ha)
        SUM(
          CASE WHEN flux_type IN ('area__ha', 'area_ha')
               THEN value ELSE 0 END
        ) AS area_ha
      FROM base
      GROUP BY 1, 2
    ),
    latest AS (
      SELECT MAX(interval_end) AS interval_end
      FROM agg
    )
    SELECT a.*
    FROM agg a
    JOIN latest l USING (interval_end)
    ORDER BY emissions_state;
    """


# ----------------------------- FAOSTAT helpers -----------------------------


def _load_fao_for_year(
    con: duckdb.DuckDBPyConnection,
    year: int,
    path: str,
) -> pd.DataFrame:
    """
    Load FAOSTAT drained-organic-soils records for the given year and the
    cropland/grassland items we care about.
    """
    path_escaped = path.replace("'", "''")
    print(f"[FAOSTAT] Reading FAO CSV from: {path}")
    sql = f"""
      SELECT
        "Area Code (M49)" AS area_code,
        "Area"            AS area,
        "Item",
        "Element",
        "Year",
        "Unit",
        "Value"
      FROM read_csv_auto('{path_escaped}', header=1)
      WHERE "Domain Code" = 'GV'
        AND "Year" = {year}
        AND "Item" IN ('Cropland organic soils', 'Grassland organic soils');
    """
    try:
        df = con.execute(sql).df()
    except Exception as exc:  # pragma: no cover - IO environment dependent
        print(f"[FAOSTAT] Failed to read FAO CSV from {path!r}: {exc}")
        return pd.DataFrame()
    return df


def _compute_fao_metrics(fao_df: pd.DataFrame) -> tuple[float, float, float, pd.DataFrame]:
    """
    Compute FAOSTAT global metrics:

    Returns
    -------
    (co2_gt, n2o_gt, total_area_mha, landuse_split_df)

    where `landuse_split_df` has columns LandUse, Area_Mha.
    If fao_df is empty, returns zeros and empty split DF.
    """
    if fao_df.empty:
        return 0.0, 0.0, 0.0, pd.DataFrame(columns=["LandUse", "Area_Mha"])

    # Map FAO item names to our LandUse categories
    item_to_landuse = {
        "Cropland organic soils": "Cropland",
        "Grassland organic soils": "Grassland",
    }

    fao_df = fao_df.copy()
    fao_df["LandUse"] = fao_df["Item"].map(item_to_landuse)

    # Emissions (kt) -> Gt CO2e
    co2_kt = fao_df.loc[fao_df["Element"] == "Emissions (CO2)", "Value"].sum()
    n2o_kt = fao_df.loc[fao_df["Element"] == "Emissions (N2O)", "Value"].sum()
    co2_gt = co2_kt * 1e-6
    n2o_gt = n2o_kt * 1e-6

    # Area (ha) -> Mha
    area_mask = fao_df["Element"] == "Area"
    total_area_mha = fao_df.loc[area_mask, "Value"].sum() / 1e6

    landuse_split = (
        fao_df.loc[area_mask]
        .groupby("LandUse", as_index=False, observed=False)["Value"]
        .sum()
        .rename(columns={"Value": "Area_Mha"})
    )
    landuse_split["Area_Mha"] = landuse_split["Area_Mha"] / 1e6

    return co2_gt, n2o_gt, total_area_mha, landuse_split


# ----------------------------- Plotting -----------------------------


def _plot_emissions(emissions: pd.DataFrame) -> plt.Figure:
    """
    Grouped bar chart by Gas (CO2, N2O) × Source (Model, FAOSTAT).
    """
    if emissions.empty:
        raise ValueError("emissions DataFrame is empty in _plot_emissions")

    gas_order = ["CO₂", "N₂O"]
    source_order = ["Model", "FAOSTAT"]

    # Pivot to Gas × Source wide format
    pivot = (
        emissions.pivot_table(
            index="Gas",
            columns="Source",
            values="Emissions_GtCO2e",
            aggfunc="sum",
            fill_value=0.0,
            observed=False,
        )
        .reindex(index=gas_order, fill_value=0.0)
    )

    # Ensure columns in desired order, but keep any extras
    existing_sources = [s for s in source_order if s in pivot.columns]
    extra_sources = [s for s in pivot.columns if s not in existing_sources]
    col_order = existing_sources + extra_sources
    pivot = pivot.reindex(columns=col_order, fill_value=0.0)

    x = np.arange(len(gas_order))
    n_src = len(col_order)
    bar_width = min(0.35, 0.8 / max(n_src, 1))

    color_map = pc.resolve_colors(col_order, palette="okabe_ito")

    with pc.use_theme(pc.THEME_LIGHT_GRID):
        fig, ax = plt.subplots(figsize=(6.0, 4.0))

        for j, src in enumerate(col_order):
            y = pivot[src].to_numpy()
            offsets = x + (j - (n_src - 1) / 2.0) * bar_width
            ax.bar(offsets, y, width=bar_width, label=src,
                   color=color_map[src], edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(gas_order)
        ax.set_ylabel("Emissions (Gt CO₂e/year)")
        ax.set_xlabel("Gas")
        pc.tidy_axes(ax, grid="y")
        pc.fmt_si(ax, axis="y")
        ax.legend(frameon=False, title="Source")

        return fig


def _plot_area(area: pd.DataFrame) -> plt.Figure:
    """
    Simple bar chart of total peat area by Source.
    """
    if area.empty:
        raise ValueError("area DataFrame is empty in _plot_area")

    # We expect a single Category repeated; use x = Source.
    ordered_sources = ["Model", "FAOSTAT"]
    src_order = [s for s in ordered_sources if s in area["Source"].unique()]
    src_order += [s for s in area["Source"].unique() if s not in src_order]

    df = area.copy()
    df["Source"] = pd.Categorical(df["Source"], categories=src_order, ordered=True)
    df = df.sort_values("Source")

    with pc.use_theme(pc.THEME_LIGHT_GRID):
        fig, ax = plt.subplots(figsize=(4.8, 3.8))
        colors = pc.resolve_colors(src_order, palette="okabe_ito")
        xs = np.arange(len(df))
        vals = df["Area_Mha"].to_numpy()

        bars = ax.bar(
            xs,
            vals,
            color=[colors[s] for s in df["Source"]],
            edgecolor="white",
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(df["Source"])
        ax.set_ylabel("Area (million ha)")
        ax.set_xlabel("")
        pc.tidy_axes(ax, grid="y")
        pc.fmt_si(ax, axis="y")

        # Label values on top
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                v,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        return fig


def _plot_landuse_split(landuse: pd.DataFrame) -> plt.Figure:
    """
    Grouped bar chart of peat area by LandUse × Source.
    """
    if landuse.empty:
        raise ValueError("landuse DataFrame is empty in _plot_landuse_split")

    landuse_order = ["Cropland", "Grassland"]
    source_order = ["Model", "FAOSTAT"]

    lu = landuse.copy()
    lu["LandUse"] = pd.Categorical(lu["LandUse"], categories=landuse_order, ordered=True)
    lu = lu.sort_values(["LandUse", "Source"])

    pivot = (
        lu.pivot_table(
            index="LandUse",
            columns="Source",
            values="Area_Mha",
            aggfunc="sum",
            fill_value=0.0,
            observed=False,
        )
        .reindex(index=landuse_order, fill_value=0.0)
    )

    existing_sources = [s for s in source_order if s in pivot.columns]
    extra_sources = [s for s in pivot.columns if s not in existing_sources]
    col_order = existing_sources + extra_sources
    pivot = pivot.reindex(columns=col_order, fill_value=0.0)

    x = np.arange(len(landuse_order))
    n_src = len(col_order)
    bar_width = min(0.35, 0.8 / max(n_src, 1))

    colors = pc.resolve_colors(col_order, palette="okabe_ito")

    with pc.use_theme(pc.THEME_LIGHT_GRID):
        fig, ax = plt.subplots(figsize=(5.4, 3.8))

        for j, src in enumerate(col_order):
            y = pivot[src].to_numpy()
            offsets = x + (j - (n_src - 1) / 2.0) * bar_width
            ax.bar(offsets, y, width=bar_width,
                   label=src, color=colors[src], edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(landuse_order)
        ax.set_ylabel("Area (million ha)")
        ax.set_xlabel("")
        pc.tidy_axes(ax, grid="y")
        pc.fmt_si(ax, axis="y")
        ax.legend(frameon=False, title="Source")

        # Label values on top of each bar
        for j, src in enumerate(col_order):
            y = pivot[src].to_numpy()
            offsets = x + (j - (n_src - 1) / 2.0) * bar_width
            for xx, v in zip(offsets, y):
                ax.text(
                    xx,
                    v,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        return fig


# ----------------------------- Driver -----------------------------


def main(argv=None):
    p = argparse.ArgumentParser("Build FAO-style cropland/grassland comparison figures")
    p.add_argument("--model_version", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--run_date", required=True)
    p.add_argument("--years", nargs="+", required=True)
    p.add_argument("--aws_region", default=None)
    p.add_argument(
        "--fao_csv",
        default=DEFAULT_FAO_CSV,
        help="Path to FAOSTAT CSV (local or s3://...). "
             "Default is the S3 GV drained-organic-soils CSV.",
    )
    p.add_argument("--data-only", action="store_true")
    args = p.parse_args(argv)

    global OUT_DIR
    OUT_DIR = build_output_dir(args.model_version, args.run_name, args.run_date)

    years = [int(y) for y in args.years]
    interval_folders = _interval_folder_strings(years)
    base_prefixes = _make_base_prefixes(
        args.model_version,
        args.run_name,
        args.run_date,
        interval_folders,
    )
    drained_globs = _make_drained_globs(base_prefixes)

    print("Resolved drained globs:")
    for g in drained_globs:
        print("  -", g)

    con = duckdb.connect()
    try:
        _register_drained(con, drained_globs, aws_region=args.aws_region)
        pa._register_state_context_views(con)

        # Component data for the latest interval present in zs_drained
        df = con.execute(sql_components_latest()).df()

        if df.empty:
            meta = con.execute(
                "SELECT COUNT(*) AS n_rows, "
                "MIN(interval_end) AS min_year, "
                "MAX(interval_end) AS max_year "
                "FROM zs_drained"
            ).df().iloc[0]
            flux_types = con.execute(
                "SELECT DISTINCT flux_type "
                "FROM zs_drained "
                "ORDER BY flux_type LIMIT 20"
            ).df()["flux_type"].tolist()
            raise RuntimeError(
                "FAO comparison: no rows returned from sql_components_latest().\n"
                f"  zs_drained rows: {meta['n_rows']}\n"
                f"  interval_end range: {meta['min_year']} – {meta['max_year']}\n"
                f"  sample flux_type values: {flux_types}"
            )

        latest_year = int(df["interval_end"].max())

        # Land-use reclassification & agricultural subset (Cropland / Grassland)
        df["LandUse"] = df["emissions_state"].apply(pc._reclass_emissions_state)
        ag_uses = df[df["LandUse"].isin(["Cropland", "Grassland"])].copy()

        # --- Model aggregates ---

        total_co2 = ag_uses["drained_co2"].sum()
        total_n2o = ag_uses["drained_n2o"].sum()
        model_emissions = pd.DataFrame(
            {
                "Source": ["Model", "Model"],
                "Gas": ["CO₂", "N₂O"],
                "Emissions_GtCO2e": [total_co2 / 1e9, total_n2o / 1e9],
                "interval_end": [latest_year, latest_year],
            }
        )

        total_area_mha = ag_uses["area_ha"].sum() / 1e6
        model_area = pd.DataFrame(
            {
                "Source": ["Model"],
                "Category": ["Cropland + Grassland peat area"],
                "Area_Mha": [total_area_mha],
                "interval_end": [latest_year],
            }
        )

        model_landuse_split = (
            ag_uses.groupby("LandUse", as_index=False, observed=False)["area_ha"]
            .sum()
            .rename(columns={"area_ha": "Area_Mha"})
        )
        model_landuse_split["Area_Mha"] = model_landuse_split["Area_Mha"] / 1e6
        model_landuse_split["Source"] = "Model"
        model_landuse_split["interval_end"] = latest_year

        # --- FAOSTAT aggregates ---

        fao_raw = _load_fao_for_year(con, latest_year, args.fao_csv)
        if fao_raw.empty:
            print(
                f"[FAOSTAT] No FAO rows found for year {latest_year} in {args.fao_csv!r}. "
                "Only model data will be exported."
            )
            fao_emissions = pd.DataFrame(columns=model_emissions.columns)
            fao_area = pd.DataFrame(columns=model_area.columns)
            fao_landuse_split = pd.DataFrame(columns=model_landuse_split.columns)
        else:
            fao_co2_gt, fao_n2o_gt, fao_total_area_mha, fao_lu_split = _compute_fao_metrics(fao_raw)

            fao_emissions = pd.DataFrame(
                {
                    "Source": ["FAOSTAT", "FAOSTAT"],
                    "Gas": ["CO₂", "N₂O"],
                    "Emissions_GtCO2e": [fao_co2_gt, fao_n2o_gt],
                    "interval_end": [latest_year, latest_year],
                }
            )

            fao_area = pd.DataFrame(
                {
                    "Source": ["FAOSTAT"],
                    "Category": ["Cropland + Grassland peat area"],
                    "Area_Mha": [fao_total_area_mha],
                    "interval_end": [latest_year],
                }
            )

            fao_lu_split = fao_lu_split.copy()
            fao_lu_split["Source"] = "FAOSTAT"
            fao_lu_split["interval_end"] = latest_year

            # Align columns with model_landuse_split
            missing_cols = [c for c in model_landuse_split.columns if c not in fao_lu_split.columns]
            for c in missing_cols:
                fao_lu_split[c] = np.nan
            fao_lu_split = fao_lu_split[model_landuse_split.columns]

            fao_landuse_split = fao_lu_split

        # Combined tables
        emissions = pd.concat([model_emissions, fao_emissions], ignore_index=True)
        area = pd.concat([model_area, fao_area], ignore_index=True)
        landuse_split = pd.concat([model_landuse_split, fao_landuse_split], ignore_index=True)

        print(f"Writing outputs to: {OUT_DIR}")

        # Raw components (model only for now)
        _write_csv_df(con, ag_uses, _join(OUT_DIR, "figures", "data", "ag_raw_components.csv"))
        print("  - figures/data/ag_raw_components.csv")

        # Aggregated comparison tables
        _write_csv_df(con, emissions, _join(OUT_DIR, "figures", "data", "emissions_by_gas.csv"))
        print("  - figures/data/emissions_by_gas.csv")

        _write_csv_df(con, area, _join(OUT_DIR, "figures", "data", "peat_area.csv"))
        print("  - figures/data/peat_area.csv")

        _write_csv_df(con, landuse_split, _join(OUT_DIR, "figures", "data", "landuse_area_split.csv"))
        print("  - figures/data/landuse_area_split.csv")

        if args.data_only:
            print("Data-only mode: figures skipped.")
            return

        # Figures
        fig = _plot_emissions(emissions)
        _save_png(fig, _join(OUT_DIR, "figures", "emissions_by_gas.png"), dpi=300)
        print("  - figures/emissions_by_gas.png")

        fig = _plot_area(area)
        _save_png(fig, _join(OUT_DIR, "figures", "peat_area.png"), dpi=300)
        print("  - figures/peat_area.png")

        if not landuse_split.empty:
            fig = _plot_landuse_split(landuse_split)
            _save_png(fig, _join(OUT_DIR, "figures", "landuse_area_split.png"), dpi=300)
            print("  - figures/landuse_area_split.png")
        else:
            print("  - [skipped] landuse_area_split.png (no Cropland/Grassland rows)")

        print("Done.")

    finally:
        con.close()


if __name__ == "__main__":
    main()
