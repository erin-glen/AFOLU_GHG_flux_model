# -*- coding: utf-8 -*-
"""
Build FAO-style comparison figures from zonal statistics and FAOSTAT.

This driver compares multiple model runs (e.g., OGH, GFW, GPD) plus FAOSTAT
for Cropland+Grassland drained peat emissions, using CO₂+N₂O zonal statistics.

Key points
----------
* For each model run, we use the *drained_co2_n2o* zonal-statistics tiles for
  the latest inventory interval (e.g., 2021–2024 for a 2024 end year).
  These tiles are already in *average annual* units, so we just convert Mg
  to Gt CO₂e/year.

* For FAOSTAT, we read the GV domain for the *same year window* as the latest
  interval (e.g., 2021–2024) and compute **average annual** emissions:
    - Emissions (CO2)   -> kt CO2       -> Gt CO₂e/year
    - Emissions (N2O)   -> kt N2O mass  -> kt CO2e via GWP -> Gt CO₂e/year
  plus area (Area) -> ha -> Mha (also averaged over the same window).

* Land-use comparability:
    - We classify model pixels from the unified state semantics exposed via the
      `emissions_state` compatibility label (derived from combined-state routing)
      using `pc._reclass_emissions_state`.
    - For this FAO comparison we explicitly fold **Oil Palm into Cropland**:
          "Oil Palm" -> "Cropland"
    - We then restrict to LandUse in {"Cropland", "Grassland"}.
      (So Cropland includes oil-palm drained peat.)

Outputs
-------
PNG figures and CSVs under:
  /mnt/c/tmp/pub_fao/version_<model_version>/<run_name>/<run_date>/

Where <run_name>/<run_date> come from the *first* --run entry (typically OGH).

CSV tables under figures/data/ include all model runs and FAOSTAT, with a
Source column.

Example usage
-------------
  cd /mnt/c/gis/git/AFOLU_GHG_flux_model

  python -m src.scripts.zonal_statistics.pub_fao \
    --years 2024 \
    --run "ogh_sensitivity_500m_10=0_9_7:20251118|OGH" \
    --run "gfw_standard_model_500m=0_9_7:20251120|GFW" \
    --run "gpd_standard_model_500m=0_9_7:20251120|GPD"
"""

from __future__ import annotations

import argparse
import posixpath
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.scripts.zonal_statistics.pub_scripts.pub_common as pc
import src.scripts.zonal_statistics.pub_scripts.pub_assets as pa
from src.scripts.zonal_statistics.run_zonal_stats import (
    build_interval_pairs,
    build_output_parquet,
)

# ----------------------------- config -----------------------------

OUT_DIR_ROOT = "/mnt/c/tmp/pub_fao"
OUT_DIR = OUT_DIR_ROOT

# Default FAOSTAT CSV (S3)
DEFAULT_FAO_CSV = (
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/"
    "fao_stat/FAOSTAT_data_en_11-22-2025.csv"
)

# Global Warming Potential used to convert FAOSTAT N2O (kt N2O) to CO2e.
# Set this to the same GWP you use in the AFOLU model:
#   AR4: 265, AR5: 298, AR6: 273
FAO_N2O_GWP = 273.0

# ----------------------------- color config -----------------------------
# All figure colors live here so they're easy to tweak without touching
# the plotting logic.

# 1) Gas colors for CO₂ vs N₂O in the emissions stacked bar
FAO_GAS_ORDER = ["CO₂", "N₂O"]
FAO_GAS_COLORS = {
    # Okabe–Ito CVD-safe pair:
    # CO₂ = deep blue, N₂O = warm orange/red
    "CO₂": "#0072B2",
    "N₂O": "#D55E00",
}

# 2) Land-use colors (Cropland vs Grassland) for peat area split
FAO_LANDUSE_ORDER = ["Cropland", "Grassland"]
FAO_LANDUSE_COLORS = {
    # Consistent with pub_common land-use styling:
    "Cropland": "#E17C05",   # orange
    "Grassland": "#A6761D",  # brown
}

# 3) Palette used for *sources* (OGH, GFW, GPD, FAOSTAT) in the area chart
FAO_SOURCE_PALETTE = "okabe_ito"

# Path helpers borrowed from pub_assets
_join = pa._join
_save_png = pa._save_png
_write_csv_df = pa._write_csv_df


RunSpec = pc.RunSpec


@dataclass
class ModelMetrics:
    source: str           # e.g., "OGH", "GFW", "GPD"
    co2_gt: float         # Gt CO2e/year (CO2)
    n2o_gt: float         # Gt CO2e/year (N2O)
    total_area_mha: float # Cropland+Grassland drained peat area (Mha)
    landuse_split: pd.DataFrame  # columns: LandUse, Area_Mha
    components: pd.DataFrame     # per-emissions_state components (ag subset)
    latest_year: int


# ----------------------------- helpers -----------------------------

def build_output_dir(model_version: str, run_name: str, run_date: str) -> str:
    return _join(OUT_DIR_ROOT, f"version_{model_version}", run_name, run_date)


def _interval_pair_for_latest(years: Sequence[int]) -> Tuple[int, int]:
    """Return (start, end) pair for the interval whose end is max(years)."""
    years_int = [int(y) for y in years]
    pairs = build_interval_pairs(years_int)
    latest = max(years_int)
    for start, end in pairs:
        if end == latest:
            return start, end
    # Fallback: just return the last pair
    return pairs[-1]


def _parse_run_specs(entries: Sequence[str]) -> list[RunSpec]:
    return pc.parse_run_specs(entries, spec_name="--run")


# ----------------------------- model (zonal stats) -----------------------------

def _make_drained_glob_candidates_for_run(spec: RunSpec, years: Sequence[int]) -> Tuple[list[Tuple[str, list[str]]], Tuple[int, int]]:
    """
    Build drained-component glob candidates for the latest inventory interval
    of this run, in preferred order:
      1) drained_co2_n2o branch (legacy FAO-targeted output)
      2) combined_state branch (new zonal-stats default)
      3) drained/fao_stat branch (older legacy fallback)
    """
    start, end = _interval_pair_for_latest(years)
    interval_folder = f"{start}_{end}"

    base_prefix = build_output_parquet(
        spec.model_version,
        spec.run_name,
        spec.run_date,
        interval_folder,
    ).rstrip("/")

    candidates = [
        ("drained_co2_n2o", [posixpath.join(base_prefix, "drained_co2_n2o", "*.parquet")]),
        ("combined_state", [posixpath.join(base_prefix, "combined_state", "*.parquet")]),
        ("drained/fao_stat", [posixpath.join(base_prefix, "drained", "fao_stat", "*.parquet")]),
    ]
    return candidates, (start, end)


def _register_model_views_for_run(
    con: duckdb.DuckDBPyConnection,
    spec: RunSpec,
    years: Sequence[int],
    aws_region: Optional[str],
) -> Tuple[int, int]:
    """
    Register zs_drained + drained_state_ctx for a given run's drained_co2_n2o tiles.

    Returns the (start, end) interval pair used.
    """
    pa._ensure_httpfs(con, aws_region)

    candidates, interval_pair = _make_drained_glob_candidates_for_run(spec, years)
    drained_globs: list[str] = []
    source_label = ""
    n_files = 0
    for candidate_label, candidate_globs in candidates:
        candidate_count = pa._count_globs(con, candidate_globs)
        if candidate_count > 0:
            source_label = candidate_label
            drained_globs = candidate_globs
            n_files = candidate_count
            break
    if n_files == 0:
        all_globs = [g for _, globs in candidates for g in globs]
        raise RuntimeError(
            f"[FAO] No drained CO2+N2O Parquet files found for run '{spec.run_name}'. "
            f"Tried globs: {all_globs}"
        )

    print(
        f"[FAO] {spec.label}: using {source_label} tiles "
        f"({n_files} file(s) across {len(drained_globs)} interval(s))"
    )
    print(f"  Globs for {spec.label}")
    for g in drained_globs:
        print(f"    - {g}")

    pa._make_component_view(con, "zs_drained", drained_globs, kind="drained")
    pa._register_state_context_views(con)
    return interval_pair


def sql_components_latest() -> str:
    """
    Aggregate drained N2O, CO2 and peat area by emissions_state, then keep only
    the *latest* interval_end present in the data.

    Critical bit: match gas flux types using prefix so that names like
    `drained_co2_Mg_CO2...` and `drained_n2o_Mg_CO2e...` are picked up.
    """
    return f"""
    WITH base AS (
      SELECT
        z.interval_end,
        COALESCE(ctx.emissions_state, 'Unspecified') AS emissions_state,
        LOWER(z.flux_type) AS flux_type,
        z.value
      FROM zs_drained z
      LEFT JOIN drained_state_ctx AS ctx
        ON (z.drained_state_meaning = ctx.meaning)
        OR ({pc._rpad_sql('z.drained_state_nodes')} = ctx.key)
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


def _compute_model_metrics_for_run(
    spec: RunSpec,
    years: Sequence[int],
    aws_region: Optional[str],
) -> ModelMetrics:
    """
    Compute Cropland+Grassland drained peat metrics for a model run:
    CO₂, N₂O, total area, and land-use split.

    Land-use classification:
      - Based on unified-state semantics exposed as `emissions_state` via
        pc._reclass_emissions_state.
      - "Oil Palm" is explicitly folded into "Cropland" for FAO comparison.
      - Filtered to LandUse in {"Cropland", "Grassland"}.
    """
    con = duckdb.connect()
    try:
        _register_model_views_for_run(con, spec, years, aws_region)
        df = con.execute(sql_components_latest()).df()
    finally:
        con.close()

    if df.empty:
        raise RuntimeError(
            f"[FAO] No component rows returned for run '{spec.run_name}' "
            "from sql_components_latest()."
        )

    latest_year = int(df["interval_end"].max())

    # Land-use classification from unified-state semantics exposed as
    # `emissions_state` (NOT drained_state)
    df = df.copy()
    df["LandUse_raw"] = df["emissions_state"].apply(pc._reclass_emissions_state)

    # Fold Oil Palm into Cropland for FAO comparison
    df["LandUse"] = df["LandUse_raw"]
    df["LandUse"] = df["LandUse"].replace({"Oil Palm": "Cropland"})
    mask_oil_palm = df["LandUse_raw"].astype(str).str.lower().str.contains("oil palm")
    df.loc[mask_oil_palm, "LandUse"] = "Cropland"

    ag_uses = df[df["LandUse"].isin(["Cropland", "Grassland"])].copy()
    if ag_uses.empty:
        return ModelMetrics(
            source=spec.label,
            co2_gt=0.0,
            n2o_gt=0.0,
            total_area_mha=0.0,
            landuse_split=pd.DataFrame(columns=["LandUse", "Area_Mha"]),
            components=ag_uses,
            latest_year=latest_year,
        )

    ag_uses[["drained_co2", "drained_n2o", "area_ha"]] = ag_uses[
        ["drained_co2", "drained_n2o", "area_ha"]
    ].fillna(0.0)

    total_co2_gt = ag_uses["drained_co2"].sum() / 1e9
    total_n2o_gt = ag_uses["drained_n2o"].sum() / 1e9
    total_area_mha = ag_uses["area_ha"].sum() / 1e6

    landuse_split = (
        ag_uses.groupby("LandUse", as_index=False, observed=False)["area_ha"]
        .sum()
        .rename(columns={"area_ha": "Area_Mha"})
    )
    landuse_split["Area_Mha"] = landuse_split["Area_Mha"] / 1e6

    return ModelMetrics(
        source=spec.label,
        co2_gt=total_co2_gt,
        n2o_gt=total_n2o_gt,
        total_area_mha=total_area_mha,
        landuse_split=landuse_split,
        components=ag_uses,
        latest_year=latest_year,
    )


# ----------------------------- FAOSTAT helpers -----------------------------

def _load_fao_for_years(
    con: duckdb.DuckDBPyConnection,
    years: Sequence[int],
    path: str,
    aws_region: Optional[str],
) -> pd.DataFrame:
    """
    Load FAOSTAT drained-organic-soils records for the given years and the
    cropland/grassland items we care about (GV domain).
    """
    years = sorted(set(int(y) for y in years))
    if not years:
        return pd.DataFrame()

    pa._ensure_httpfs(con, aws_region)
    years_list = ", ".join(str(y) for y in years)
    yrs_label = f"{years[0]}–{years[-1]}" if len(years) > 1 else f"{years[0]}"

    path_escaped = path.replace("'", "''")
    print(f"[FAOSTAT] Reading FAO CSV for years {yrs_label} from: {path}")
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
        AND "Year" IN ({years_list})
        AND "Item" IN ('Cropland organic soils', 'Grassland organic soils');
    """
    try:
        df = con.execute(sql).df()
    except Exception as exc:  # pragma: no cover
        print(f"[FAOSTAT] Failed to read FAO CSV from {path!r}: {exc}")
        return pd.DataFrame()
    return df


def _compute_fao_metrics(
    fao_df: pd.DataFrame,
    years: Optional[Sequence[int]] = None,
) -> tuple[float, float, float, pd.DataFrame]:
    """
    Compute FAOSTAT global metrics from the GV drained-organic-soils subset.

    All returned emissions are in **Gt CO2e/year**, averaged over the
    provided year window (e.g., 2021–2024):

      * CO2: FAOSTAT reports Emissions (CO2) in kt CO2.
        We convert kt -> Gt CO₂e by multiplying by 1e-6 and dividing
        by the number of years.

      * N2O: FAOSTAT reports Emissions (N2O) in kt N2O (mass).
        We first convert kt N2O -> kt CO2e using FAO_N2O_GWP,
        then kt -> Gt and average over years.

      * Area: FAOSTAT "Area" is in ha; we convert to million ha (Mha)
        and average over years.

    Parameters
    ----------
    fao_df : DataFrame
        FAOSTAT rows already filtered to Domain Code 'GV' and the relevant
        Items ('Cropland organic soils', 'Grassland organic soils') and
        years of interest.
    years : sequence of int, optional
        Used to restrict to a specific subset of years and to determine
        how many years to average over. If None, the unique years present
        in fao_df are used.

    Returns
    -------
    (co2_gt, n2o_gt, total_area_mha, landuse_split_df)

    where `landuse_split_df` has columns:
        LandUse, Area_Mha
    """
    if fao_df.empty:
        return 0.0, 0.0, 0.0, pd.DataFrame(columns=["LandUse", "Area_Mha"])

    # Map FAO item names to our LandUse categories
    item_to_landuse = {
        "Cropland organic soils": "Cropland",
        "Grassland organic soils": "Grassland",
    }

    df = fao_df.copy()
    df["LandUse"] = df["Item"].map(item_to_landuse)

    # Restrict to requested years if provided
    if years is not None:
        yrs = {int(y) for y in years}
        df = df[df["Year"].isin(yrs)]
        if df.empty:
            return 0.0, 0.0, 0.0, pd.DataFrame(columns=["LandUse", "Area_Mha"])

    # Determine number of unique years for averaging
    if "Year" in df.columns:
        unique_years = sorted(df["Year"].dropna().unique().tolist())
        n_years = len(unique_years)
    else:
        n_years = len(set(years)) if years is not None else 1
    if n_years == 0:
        n_years = 1

    # Convenience masks
    co2_mask = df["Element"] == "Emissions (CO2)"
    n2o_mask = df["Element"] == "Emissions (N2O)"
    area_mask = df["Element"] == "Area"

    # ---- Emissions ----
    # CO2: kt CO2 -> Gt CO2e/year
    co2_kt_total = df.loc[co2_mask, "Value"].sum()

    # N2O: kt N2O -> kt CO2e via GWP -> Gt CO2e/year
    n2o_kt_total = df.loc[n2o_mask, "Value"].sum()

    co2_gt = (co2_kt_total / n_years) * 1e-6
    n2o_gt = (n2o_kt_total / n_years) * FAO_N2O_GWP * 1e-6

    # ---- Area ----
    # Area in ha -> average over years -> Mha
    area_total_ha = df.loc[area_mask, "Value"].sum()
    total_area_mha = (area_total_ha / n_years) / 1e6

    landuse_split = (
        df.loc[area_mask]
        .groupby("LandUse", as_index=False, observed=False)["Value"]
        .sum()
        .rename(columns={"Value": "Area_Mha"})
    )
    landuse_split["Area_Mha"] = landuse_split["Area_Mha"] / n_years / 1e6

    return co2_gt, n2o_gt, total_area_mha, landuse_split


# ----------------------------- plotting -----------------------------

def _plot_emissions(emissions: pd.DataFrame) -> plt.Figure:
    """
    Stacked bar chart by Source with CO₂ + N₂O segments.

    Styled to match other publication figures:
      - Legend on top (centered).
      - Themed with THEME_LIGHT_GRID + y-axis grid.
    """
    if emissions.empty:
        raise ValueError("emissions DataFrame is empty in _plot_emissions")

    df = emissions.copy()
    df = df[df["Gas"].isin(FAO_GAS_ORDER)]
    if df.empty:
        raise ValueError("No CO₂/N₂O rows available for plotting in _plot_emissions")

    df["Gas"] = pd.Categorical(df["Gas"], categories=FAO_GAS_ORDER, ordered=True)

    # Complete gas color map from top-level config
    gas_colors = pc.resolve_colors(
        FAO_GAS_ORDER,
        color_map=FAO_GAS_COLORS,
        palette="okabe_ito",
    )

    theme = {**pc.THEME_LIGHT_GRID, "axes.grid.axis": "y"}
    with pc.use_theme(theme):
        fig = pc.stacked_column_by_category(
            df_long=df,
            index_col="Source",
            category_col="Gas",
            value_col="Emissions_GtCO2e",
            category_order=FAO_GAS_ORDER,
            color_map=gas_colors,
            xlabel="Source",
            ylabel="Emissions (Gt CO₂e/year)",
            width=7.0,
            height=4.6,
            legend_above=True,
        )
        return fig


def _plot_area(area: pd.DataFrame) -> plt.Figure:
    """
    Simple bar chart of total Cropland+Grassland peat area by Source.
    """
    if area.empty:
        raise ValueError("area DataFrame is empty in _plot_area")

    src_order = list(area["Source"].unique())
    df = area.copy()
    df["Source"] = pd.Categorical(df["Source"], categories=src_order, ordered=True)
    df = df.sort_values("Source")

    theme = {**pc.THEME_LIGHT_GRID, "axes.grid.axis": "y"}
    with pc.use_theme(theme):
        fig, ax = plt.subplots(figsize=(4.8, 3.8))
        colors = pc.resolve_colors(src_order, palette=FAO_SOURCE_PALETTE)
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

        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                v,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.tight_layout()
        return fig


def _plot_landuse_split(landuse: pd.DataFrame) -> plt.Figure:
    """
    Stacked bar chart of peat area by Source with LandUse (Cropland/Grassland) segments.

    Uses top-level FAO_LANDUSE_* config and legend on top, to match
    other publication figures.
    """
    if landuse.empty:
        raise ValueError("landuse DataFrame is empty in _plot_landuse_split")

    df = landuse.copy()
    df = df[df["LandUse"].isin(FAO_LANDUSE_ORDER)]
    if df.empty:
        raise ValueError("No Cropland/Grassland rows available for plotting in _plot_landuse_split")

    df["LandUse"] = pd.Categorical(df["LandUse"], categories=FAO_LANDUSE_ORDER, ordered=True)

    lu_colors = pc.resolve_colors(
        FAO_LANDUSE_ORDER,
        color_map=FAO_LANDUSE_COLORS,
        palette="okabe_ito",
    )

    theme = {**pc.THEME_LIGHT_GRID, "axes.grid.axis": "y"}
    with pc.use_theme(theme):
        fig = pc.stacked_column_by_category(
            df_long=df,
            index_col="Source",
            category_col="LandUse",
            value_col="Area_Mha",
            category_order=FAO_LANDUSE_ORDER,
            color_map=lu_colors,
            xlabel="Source",
            ylabel="Area (million ha)",
            width=7.0,
            height=4.5,
            legend_above=True,
        )
        return fig


# ----------------------------- driver -----------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        "Build FAO-style cropland/grassland comparison figures "
        "(OGH / GFW / GPD vs FAOSTAT)"
    )
    p.add_argument("--years", nargs="+", required=True)
    p.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Run specification: run_name=model_version:run_date or "
            "run_name=model_version:run_date|Label (e.g. "
            "\"ogh_sensitivity_500m_10=0_9_7:20251118|OGH\"). "
            "Provide once per model source."
        ),
    )
    p.add_argument("--aws_region", default=None)
    p.add_argument(
        "--fao_csv",
        default=DEFAULT_FAO_CSV,
        help=(
            "Path to FAOSTAT CSV (local or s3://...). "
            "Default is the S3 GV drained-organic-soils CSV."
        ),
    )
    p.add_argument("--data-only", action="store_true")
    args = p.parse_args(argv)

    years = sorted({int(y) for y in args.years})
    if not years:
        raise SystemExit("At least one inventory year must be provided")

    run_specs = _parse_run_specs(args.run)
    if not run_specs:
        raise SystemExit("At least one --run specification is required")

    # Use the first run (typically OGH) to define the output directory
    primary = run_specs[0]
    global OUT_DIR
    OUT_DIR = build_output_dir(primary.model_version, primary.run_name, primary.run_date)

    print("FAO comparison – model sources:")
    for spec in run_specs:
        print(
            f"  - {spec.label}: run_name={spec.run_name}, "
            f"version={spec.model_version}, run_date={spec.run_date}"
        )
    print("Output directory:", OUT_DIR)

    # Determine FAO averaging window from the latest interval
    start_year, end_year = _interval_pair_for_latest(years)
    fao_years = list(range(start_year, end_year + 1))
    if len(fao_years) > 1:
        print(
            f"Comparison period (model interval): {start_year}_{end_year} "
            f"(FAO averaged over years {fao_years[0]}–{fao_years[-1]})"
        )
    else:
        print(
            f"Comparison period (model interval): {start_year}_{end_year} "
            f"(FAO year {fao_years[0]})"
        )

    # --- model metrics for each run ---
    model_metrics: list[ModelMetrics] = []
    for spec in run_specs:
        metrics = _compute_model_metrics_for_run(spec, years, args.aws_region)
        model_metrics.append(metrics)

    # Assemble combined tables for model sources
    emissions_rows = []
    area_rows = []
    landuse_rows = []
    components_frames = []

    for m in model_metrics:
        emissions_rows.extend(
            [
                {
                    "Source": m.source,
                    "Gas": "CO₂",
                    "Emissions_GtCO2e": m.co2_gt,
                    "interval_end": m.latest_year,
                },
                {
                    "Source": m.source,
                    "Gas": "N₂O",
                    "Emissions_GtCO2e": m.n2o_gt,
                    "interval_end": m.latest_year,
                },
            ]
        )

        area_rows.append(
            {
                "Source": m.source,
                "Category": "Cropland + Grassland peat area",
                "Area_Mha": m.total_area_mha,
                "interval_end": m.latest_year,
            }
        )

        if not m.landuse_split.empty:
            lu = m.landuse_split.copy()
            lu["Source"] = m.source
            lu["interval_end"] = m.latest_year
            landuse_rows.append(lu)

        if not m.components.empty:
            df_c = m.components.copy()
            df_c["Source"] = m.source
            components_frames.append(df_c)

    emissions = pd.DataFrame(emissions_rows)
    area = pd.DataFrame(area_rows)
    landuse_split = (
        pd.concat(landuse_rows, ignore_index=True) if landuse_rows else pd.DataFrame()
    )
    ag_components = (
        pd.concat(components_frames, ignore_index=True)
        if components_frames
        else pd.DataFrame()
    )

    # --- FAOSTAT metrics over the same window (average 2021–2024 etc.) ---
    con_fao = duckdb.connect()
    try:
        fao_raw = _load_fao_for_years(con_fao, fao_years, args.fao_csv, args.aws_region)
    finally:
        con_fao.close()

    latest_year = max(years)

    if fao_raw.empty:
        print(
            f"[FAOSTAT] No FAO GV rows found for years {fao_years} "
            f"in {args.fao_csv!r}. FAOSTAT will not appear in comparison."
        )
    else:
        fao_co2_gt, fao_n2o_gt, fao_area_mha, fao_lu_split = _compute_fao_metrics(
            fao_raw, fao_years
        )

        emissions = pd.concat(
            [
                emissions,
                pd.DataFrame(
                    {
                        "Source": ["FAOSTAT", "FAOSTAT"],
                        "Gas": ["CO₂", "N₂O"],
                        "Emissions_GtCO2e": [fao_co2_gt, fao_n2o_gt],
                        "interval_end": [latest_year, latest_year],
                    }
                ),
            ],
            ignore_index=True,
        )

        area = pd.concat(
            [
                area,
                pd.DataFrame(
                    {
                        "Source": ["FAOSTAT"],
                        "Category": ["Cropland + Grassland peat area"],
                        "Area_Mha": [fao_area_mha],
                        "interval_end": [latest_year],
                    }
                ),
            ],
            ignore_index=True,
        )

        if not fao_lu_split.empty:
            fao_lu_split = fao_lu_split.copy()
            fao_lu_split["Source"] = "FAOSTAT"
            fao_lu_split["interval_end"] = latest_year
            landuse_split = pd.concat(
                [landuse_split, fao_lu_split], ignore_index=True
            )

    print("Writing outputs to:", OUT_DIR)

    writer_con = duckdb.connect()
    try:
        _write_csv_df(
            writer_con,
            ag_components,
            _join(OUT_DIR, "figures", "data", "ag_raw_components.csv"),
        )
        print("  - figures/data/ag_raw_components.csv")

        _write_csv_df(
            writer_con,
            emissions,
            _join(OUT_DIR, "figures", "data", "emissions_by_gas.csv"),
        )
        print("  - figures/data/emissions_by_gas.csv")

        _write_csv_df(
            writer_con,
            area,
            _join(OUT_DIR, "figures", "data", "peat_area.csv"),
        )
        print("  - figures/data/peat_area.csv")

        _write_csv_df(
            writer_con,
            landuse_split,
            _join(OUT_DIR, "figures", "data", "landuse_area_split.csv"),
        )
        print("  - figures/data/landuse_area_split.csv")

    finally:
        writer_con.close()

    if args.data_only:
        print("Data-only mode: figures skipped.")
        return

    # --- Figures ---
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


if __name__ == "__main__":
    main()