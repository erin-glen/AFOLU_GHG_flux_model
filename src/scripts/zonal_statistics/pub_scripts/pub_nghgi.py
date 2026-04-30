# -*- coding: utf-8 -*-
"""
Compare model drained organic-soil outputs to country NGHGI CRT reports.

For each model run and inventory interval (e.g. 2001-2005 through 2021-2024):

  * Model side:
      zs_drained zonal-stats tiles -> table_nghgi_comparison_subset_sql
      gives (iso3, land_use, interval_end, drained_area_ha,
      drained_on_site_co2_Mg_CO2_yr).

  * NGHGI side:
      Two compiled CSVs produced from UNFCCC CRT workbooks (see README in the
      organic_soil_nghgi directory):
        * organic_soil_compiled.csv       -> CRT Table 4(II)
          (drained organic-soil area + CO2/N2O/CH4)
        * organic_soil_cstock_compiled.csv -> CRT Tables 4.A-4.F
          (organic-soil carbon stock change, used when Table 4(II) CO2 = 'IE')

Values for each country x land-use x interval are averaged across the years in
the interval window so the comparison is on a per-year basis.

Outputs
-------
PNG figures + CSVs under:
    <OUT_DIR_ROOT>/version_<model_version>/<run_name>/<run_date>/

Where <run_name>/<run_date> come from the *first* --run entry.

Example usage
-------------
    cd /mnt/c/gis/git/AFOLU_GHG_flux_model

    python -m src.scripts.zonal_statistics.pub_scripts.pub_nghgi \\
        --years 2005 2010 2015 2020 2024 \\
        --run "zarr_test_full=0_1_2:20260403|GFW" \\
        --nghgi_dir /mnt/c/GIS/Data/Global/Wetlands/organic_soil_nghgi
"""

from __future__ import annotations

import argparse
import os
import posixpath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.scripts.zonal_statistics.pub_scripts.pub_assets as pa
import src.scripts.zonal_statistics.pub_scripts.pub_common as pc
import src.scripts.zonal_statistics.pub_scripts.extract_organic_soil_jrc as jrc_loader
from src.scripts.zonal_statistics.run_zonal_stats import (
    build_interval_pairs,
    build_output_parquet,
)


# ----------------------------- config -----------------------------

OUT_DIR_ROOT = os.environ.get("AFOLU_PUB_NGHGI_DIR", "/mnt/c/tmp/pub_nghgi")
OUT_DIR = OUT_DIR_ROOT

DEFAULT_NGHGI_DIR = os.environ.get("AFOLU_NGHGI_DATA_DIR", "/mnt/c/GIS/Data/Global/Wetlands/organic_soil_nghgi")
DEFAULT_JRC_DIR = jrc_loader.DEFAULT_JRC_DIR
NGHGI_TABLE_4II_NAME = "organic_soil_compiled.csv"
NGHGI_CSTOCK_NAME = "organic_soil_cstock_compiled.csv"

# Prefer File 1 (Table 4(II)) em_co2_kt when numeric; fall back to converting
# cstock_soil_organic_ktC from File 2. kt C -> kt CO2 uses 44/12; sign flips
# because a negative carbon stock change equals a positive CO2 emission.
C_TO_CO2 = 44.0 / 12.0

# NGHGI category-code prefix -> model LandUse label (from
# pub_common._reclass_emissions_state). Peat extraction in the NGHGI
# falls under 4(II).D (Wetlands) and is mapped to Wetland; the model
# may surface it as "Extraction" which we fold into Wetland for
# comparability.
NGHGI_CATEGORY_TO_LANDUSE: Dict[str, str] = {
    "4(II).A": "Forest",
    "4(II).B": "Cropland",
    "4(II).C": "Grassland",
    "4(II).D": "Wetland",
    "4(II).E": "Settlement",
    "4(II).F": "Otherland",
}

# Model LandUse values that should be folded into a coarser label for the
# NGHGI comparison. NGHGI reports at IPCC land-use level so we match that.
MODEL_LANDUSE_FOLD: Dict[str, str] = {
    "Oil Palm": "Cropland",
    "Other plantation": "Cropland",
    "Extraction": "Wetland",
    "Undrained (unclassified)": "Wetland",
}

LANDUSE_ORDER = ["Forest", "Cropland", "Grassland", "Wetland", "Settlement", "Otherland"]

LANDUSE_COLORS = {
    "Forest":     "#117733",
    "Cropland":   "#E17C05",
    "Grassland":  "#A6761D",
    "Wetland":    "#0072B2",
    "Settlement": "#CC79A7",
    "Otherland":  "#6E6E6E",
}

# Path helpers borrowed from pub_assets
_join = pa._join
_save_png = pa._save_png
_write_csv_df = pa._write_csv_df


RunSpec = pc.RunSpec


# ----------------------------- helpers -----------------------------

def build_output_dir(model_version: str, run_name: str, run_date: str) -> str:
    return _join(OUT_DIR_ROOT, f"version_{model_version}", run_name, run_date)


def _parse_run_specs(entries: Sequence[str]) -> list[RunSpec]:
    return pc.parse_run_specs(entries, spec_name="--run")


def _nghgi_category_to_landuse(code: Optional[str]) -> Optional[str]:
    """Map NGHGI category_code (e.g. '4(II).A.1.b') to a coarse LandUse."""
    if not code:
        return None
    for prefix, landuse in NGHGI_CATEGORY_TO_LANDUSE.items():
        if code == prefix or code.startswith(prefix + "."):
            return landuse
    return None


def _fold_model_landuse(landuse: Optional[str]) -> Optional[str]:
    if landuse is None:
        return None
    return MODEL_LANDUSE_FOLD.get(landuse, landuse)


# ----------------------------- model side -----------------------------

def _model_parquet_globs(spec: RunSpec, interval_folder: str) -> list[str]:
    """
    Candidate parquet globs for a run x interval, combined_state branch.
    """
    base_prefix = build_output_parquet(
        spec.model_version,
        spec.run_name,
        spec.run_date,
        interval_folder,
    ).rstrip("/")
    return [posixpath.join(base_prefix, "combined_state", "*.parquet")]


def _model_country_landuse_for_interval(
    spec: RunSpec,
    interval_folder: str,
    aws_region: Optional[str],
    adm0_lookup_csv: Optional[str],
) -> pd.DataFrame:
    """
    Run table_nghgi_comparison_subset_sql for a single (run, interval_folder)
    and return a DataFrame with columns:
        interval_end, gadm_adm0, country, iso3, land_use,
        drained_area_ha, undrained_area_ha, drained_on_site_co2_Mg_CO2_yr

    Returns an empty DataFrame if the parquet tiles don't exist for that
    interval (allows partial model runs to skip cleanly).
    """
    con = duckdb.connect()
    try:
        pa._ensure_httpfs(con, aws_region)
        globs = _model_parquet_globs(spec, interval_folder)

        if pa._count_globs(con, globs) == 0:
            print(
                f"  [skip] {spec.label} {interval_folder}: no combined_state "
                f"parquet files at {globs[0]}"
            )
            return pd.DataFrame()

        print(f"  [model] {spec.label} {interval_folder}: reading {globs[0]}")
        pa._make_component_view(con, "zs_drained", globs, kind="drained")
        pa._register_state_context_views(con)
        has_lookup = pa._ensure_adm0_lookup(con, adm0_lookup_csv)

        sql = pc.table_nghgi_comparison_subset_sql(with_lookup=has_lookup)
        df = con.execute(sql).df()
    finally:
        con.close()

    if df.empty:
        return df

    # Fold plantation/extraction labels onto the coarser IPCC land-use set
    df["land_use_folded"] = df["land_use"].apply(_fold_model_landuse)
    agg = (
        df.groupby(
            ["interval_end", "gadm_adm0", "iso3", "country", "land_use_folded"],
            as_index=False,
            dropna=False,
        )[["drained_area_ha", "undrained_area_ha", "drained_on_site_co2_Mg_CO2_yr"]]
        .sum()
        .rename(columns={"land_use_folded": "land_use"})
    )
    return agg


def load_model_country_landuse(
    specs: Sequence[RunSpec],
    years: Sequence[int],
    aws_region: Optional[str],
    adm0_lookup_csv: Optional[str],
) -> pd.DataFrame:
    """
    For each run x interval (derived from years), run the NGHGI-comparison SQL
    and concatenate. Tags each row with run label + interval_start.
    """
    pairs = build_interval_pairs([int(y) for y in years])

    frames: list[pd.DataFrame] = []
    for spec in specs:
        for start, end in pairs:
            interval_folder = f"{start}_{end}"
            df = _model_country_landuse_for_interval(
                spec, interval_folder, aws_region, adm0_lookup_csv
            )
            if df.empty:
                continue
            df = df.copy()
            df["run_label"] = spec.label
            df["run_name"] = spec.run_name
            df["model_version"] = spec.model_version
            df["run_date"] = spec.run_date
            df["interval_start"] = start
            df["interval_end"] = end
            df["interval"] = interval_folder
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ----------------------------- NGHGI side -----------------------------

def _to_numeric_col(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _load_nghgi_compiled(nghgi_dir: str) -> pd.DataFrame:
    """
    Load organic_soil_compiled.csv (CRT Table 4(II)). Keep only 'Total organic
    soils' and 'Drained organic soils' rows so we can compute area (drained)
    and gas emissions per country x land-use x year.
    """
    path = os.path.join(nghgi_dir, NGHGI_TABLE_4II_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"NGHGI Table 4(II) CSV not found at {path}. "
            "Run the compile_nghgi step or pass --nghgi_dir."
        )
    df = pd.read_csv(path, low_memory=False)

    # Restrict to the organic-soil soil-type rows
    df = df[df["soil_type"].isin(["Total organic soils", "Drained organic soils"])].copy()

    # Coerce numerics (flag columns kept separate so sentinels stay inspectable)
    for col in ("area_kha", "em_co2_kt", "em_n2o_kt", "em_ch4_kt"):
        df[col] = _to_numeric_col(df[col])

    df["land_use"] = df["category_code"].apply(_nghgi_category_to_landuse)
    df = df[df["land_use"].notna()].copy()
    df["year"] = _to_numeric_col(df["year"]).astype("Int64")
    return df


def _load_nghgi_cstock(nghgi_dir: str) -> pd.DataFrame:
    """
    Load organic_soil_cstock_compiled.csv (CRT Tables 4.A-4.F). Used as the
    fallback source for organic-soil CO2 when Table 4(II) reports 'IE'.
    """
    path = os.path.join(nghgi_dir, NGHGI_CSTOCK_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"NGHGI C-stock CSV not found at {path}. "
            "Run the compile_nghgi step or pass --nghgi_dir."
        )
    df = pd.read_csv(path, low_memory=False)

    for col in ("area_organic_kha", "cstock_soil_organic_ktC"):
        df[col] = _to_numeric_col(df[col])

    # Map `4.X` prefix to the same LandUse values using the 4(II) mapping
    def _cstock_cat_to_landuse(code: Optional[str]) -> Optional[str]:
        if not code:
            return None
        # cstock codes look like '4.A', '4.A.1', etc.
        lu_map = {
            "4.A": "Forest", "4.B": "Cropland", "4.C": "Grassland",
            "4.D": "Wetland", "4.E": "Settlement", "4.F": "Otherland",
        }
        for prefix, lu in lu_map.items():
            if code == prefix or code.startswith(prefix + "."):
                return lu
        return None

    df["land_use"] = df["category_code"].apply(_cstock_cat_to_landuse)
    df = df[df["land_use"].notna()].copy()
    df["year"] = _to_numeric_col(df["year"]).astype("Int64")
    return df


def nghgi_by_iso_landuse_interval(
    nghgi_t4ii: pd.DataFrame,
    nghgi_cstock: pd.DataFrame,
    interval_pairs: Sequence[Tuple[int, int]],
) -> pd.DataFrame:
    """
    Average NGHGI values over each (start_year, end_year) interval.

    Returns one row per (iso3, land_use, interval_end) with:
        nghgi_area_drained_organic_ha
        nghgi_em_co2_Mg_yr   (preferred: T4(II) numeric; fallback: cstock * 44/12)
        nghgi_em_co2_source  ('T4II' | 'T4land_cstock' | None)
        nghgi_em_n2o_kt_yr
        nghgi_em_ch4_kt_yr
        nghgi_years_available  (count of years with any numeric value in window)
    """
    # --------- Drained organic-soil area (kha -> ha) from T4(II) ---------
    area_df = nghgi_t4ii[nghgi_t4ii["soil_type"] == "Drained organic soils"].copy()

    # --------- Gas emissions: prefer Drained sub-row, fall back to Total ---------
    # The model only covers drained-peat emissions, so we prefer the "Drained
    # organic soils" sub-row. When a country reports CO2 only on the aggregate
    # "Total organic soils" row (without splitting into drained/rewetted), we
    # fall back to that value.
    gas_drained = nghgi_t4ii[nghgi_t4ii["soil_type"] == "Drained organic soils"].copy()
    gas_total = nghgi_t4ii[nghgi_t4ii["soil_type"] == "Total organic soils"].copy()

    # Aggregate each land-use's sub-categories (.1 + .2 etc.) up to the land-use
    def _collapse_to_toplevel(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        # CRT 4(II) hierarchy: 4(II).X.1 ("remaining") and 4(II).X.2 ("converted")
        # are the canonical non-overlapping partition for each land-use X.  Their
        # children (.X.2.a, .X.2.b ...) carry the same values broken out further,
        # so including both parent + children would double-count.
        #
        # Strategy: use exactly-2-dot rows (.X.1 / .X.2).  When a .X.1 or .X.2
        # parent is NaN, fill from the sum of its children.
        df = df.copy()
        df["n_dots"] = df["category_code"].str.count(r"\.")
        # ".sub" = the 1-or-2 digit after the land-use letter (1 or 2)
        df["sub"] = df["category_code"].str.extract(r"4\(II\)\.[A-F]\.(\d)")

        level2 = df[df["n_dots"] == 2]
        level3 = df[df["n_dots"] >= 3]

        grp_keys = ["iso3", "year", "land_use", "sub"]
        parent_vals = (
            level2.groupby(grp_keys, as_index=False, observed=False)[value_col]
            .sum(min_count=1)
            .rename(columns={value_col: "parent"})
        )
        child_vals = (
            level3.groupby(grp_keys, as_index=False, observed=False)[value_col]
            .sum(min_count=1)
            .rename(columns={value_col: "child"})
        )

        merged = parent_vals.merge(child_vals, on=grp_keys, how="outer")
        merged[value_col] = merged["parent"].where(merged["parent"].notna(), merged["child"])

        grouped = (
            merged.groupby(["iso3", "year", "land_use"], as_index=False, observed=False)[value_col]
            .sum(min_count=1)
        )
        return grouped

    area_per_year = _collapse_to_toplevel(area_df, "area_kha")
    area_per_year = area_per_year.rename(columns={"area_kha": "nghgi_area_drained_organic_kha"})

    # Gas emissions: collapse Drained and Total separately, then prefer Drained
    def _prefer_drained_gas(drained_df, total_df, col):
        drn = _collapse_to_toplevel(drained_df, col).rename(columns={col: "drn"})
        tot = _collapse_to_toplevel(total_df, col).rename(columns={col: "tot"})
        m = drn.merge(tot, on=["iso3", "year", "land_use"], how="outer")
        m[col] = m["drn"].where(m["drn"].notna(), m["tot"])
        return m[["iso3", "year", "land_use", col]]

    co2_per_year = _prefer_drained_gas(gas_drained, gas_total, "em_co2_kt")
    n2o_per_year = _prefer_drained_gas(gas_drained, gas_total, "em_n2o_kt")
    ch4_per_year = _prefer_drained_gas(gas_drained, gas_total, "em_ch4_kt")

    co2_per_year = co2_per_year.rename(columns={"em_co2_kt": "nghgi_em_co2_kt"})
    n2o_per_year = n2o_per_year.rename(columns={"em_n2o_kt": "nghgi_em_n2o_kt"})
    ch4_per_year = ch4_per_year.rename(columns={"em_ch4_kt": "nghgi_em_ch4_kt"})

    # --------- CO2 + area fallback via Tables 4.A-4.F ---------
    # Only the top-level land-use rows ('4.A', '4.B', ...) carry the total
    # organic area / C-stock change per land use; deeper sub-categories would
    # double-count when summed.
    cstock_top = nghgi_cstock[nghgi_cstock["category_code"].str.count(r"\.") == 1].copy()
    cstock_per_year = (
        cstock_top.groupby(["iso3", "year", "land_use"], as_index=False, observed=False)
        .agg(
            cstock_soil_organic_ktC=("cstock_soil_organic_ktC", "sum"),
            nghgi_area_organic_t4land_kha=("area_organic_kha", "sum"),
        )
    )
    # -kt C * 44/12 => kt CO2. Flip sign so a net C loss reads as a positive emission.
    cstock_per_year["nghgi_em_co2_from_cstock_kt"] = (
        -cstock_per_year["cstock_soil_organic_ktC"] * C_TO_CO2
    )
    cstock_per_year = cstock_per_year.drop(columns=["cstock_soil_organic_ktC"])

    # Merge per-year NGHGI
    annual = (
        area_per_year
        .merge(co2_per_year, on=["iso3", "year", "land_use"], how="outer")
        .merge(n2o_per_year, on=["iso3", "year", "land_use"], how="outer")
        .merge(ch4_per_year, on=["iso3", "year", "land_use"], how="outer")
        .merge(cstock_per_year, on=["iso3", "year", "land_use"], how="outer")
    )

    # --------- Window-average over each interval pair ---------
    rows: list[dict] = []
    for start, end in interval_pairs:
        window = annual[(annual["year"] >= start) & (annual["year"] <= end)].copy()
        if window.empty:
            continue
        grouped = (
            window.groupby(["iso3", "land_use"], as_index=False, observed=False)
            .agg(
                nghgi_area_drained_organic_kha=("nghgi_area_drained_organic_kha", "mean"),
                nghgi_area_organic_t4land_kha=("nghgi_area_organic_t4land_kha", "mean"),
                nghgi_em_co2_kt=("nghgi_em_co2_kt", "mean"),
                nghgi_em_n2o_kt=("nghgi_em_n2o_kt", "mean"),
                nghgi_em_ch4_kt=("nghgi_em_ch4_kt", "mean"),
                nghgi_em_co2_from_cstock_kt=("nghgi_em_co2_from_cstock_kt", "mean"),
                nghgi_years_available=("year", "nunique"),
            )
        )
        grouped["interval_start"] = start
        grouped["interval_end"] = end
        grouped["interval"] = f"{start}_{end}"
        rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    # CO2 preference: use T4(II) value if present, else cstock conversion
    t4ii_co2 = out["nghgi_em_co2_kt"]
    cstock_co2 = out["nghgi_em_co2_from_cstock_kt"]
    out["nghgi_em_co2_kt_preferred"] = t4ii_co2.where(t4ii_co2.notna(), cstock_co2)
    out["nghgi_em_co2_source"] = np.where(
        t4ii_co2.notna(),
        "T4II",
        np.where(cstock_co2.notna(), "T4land_cstock", None),
    )

    # Area columns (kha -> ha)
    t4ii_area = out["nghgi_area_drained_organic_kha"]
    t4land_area = out["nghgi_area_organic_t4land_kha"]

    # Drained organic area reported in T4(II); NaN when country reports 'IE'/'NE'
    out["nghgi_area_drained_organic_ha"] = t4ii_area * 1_000.0
    # Total organic area (drained + undrained + rewetted) from T4.A-F
    out["nghgi_area_total_organic_ha"] = t4land_area * 1_000.0
    # Undrained proxy = total - drained, clipped at 0. NaN when either side missing.
    out["nghgi_area_undrained_organic_ha"] = (
        (t4land_area - t4ii_area).clip(lower=0) * 1_000.0
    )

    # kt CO2 -> Mg CO2 so units line up with the model
    out["nghgi_em_co2_Mg_yr"] = out["nghgi_em_co2_kt_preferred"] * 1_000.0

    return out


# ----------------------------- joining -----------------------------

def join_model_nghgi(
    model_df: pd.DataFrame,
    nghgi_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Outer-join model + NGHGI on (iso3, land_use, interval_end). Keep rows where
    *either* side has data so that coverage gaps are visible.
    """
    if model_df.empty and nghgi_df.empty:
        return pd.DataFrame()
    if model_df.empty:
        return nghgi_df.assign(
            model_drained_area_ha=np.nan,
            model_drained_co2_Mg_yr=np.nan,
        )
    if nghgi_df.empty:
        m = model_df.copy()
        m["nghgi_area_drained_organic_ha"] = np.nan
        m["nghgi_em_co2_Mg_yr"] = np.nan
        m["nghgi_em_co2_source"] = None
        m["nghgi_years_available"] = np.nan
        return m

    model_join = (
        model_df
        .rename(columns={
            "drained_area_ha": "model_drained_area_ha",
            "drained_on_site_co2_Mg_CO2_yr": "model_drained_co2_Mg_yr",
        })
        .loc[:, [
            "run_label", "run_name", "model_version", "run_date",
            "interval", "interval_start", "interval_end",
            "gadm_adm0", "iso3", "country", "land_use",
            "model_drained_area_ha", "undrained_area_ha", "model_drained_co2_Mg_yr",
        ]]
    )

    joined = model_join.merge(
        nghgi_df[[
            "iso3", "land_use", "interval_end",
            "nghgi_area_drained_organic_ha",
            "nghgi_area_undrained_organic_ha",
            "nghgi_area_total_organic_ha",
            "nghgi_em_co2_Mg_yr",
            "nghgi_em_co2_kt", "nghgi_em_n2o_kt", "nghgi_em_ch4_kt",
            "nghgi_em_co2_from_cstock_kt", "nghgi_em_co2_source",
            "nghgi_years_available",
        ]],
        on=["iso3", "land_use", "interval_end"],
        how="outer",
    )

    # Derived: difference, ratio
    joined["area_diff_ha"] = joined["model_drained_area_ha"] - joined["nghgi_area_drained_organic_ha"]
    joined["area_ratio_model_over_nghgi"] = (
        joined["model_drained_area_ha"] / joined["nghgi_area_drained_organic_ha"]
    )
    joined["co2_diff_Mg_yr"] = joined["model_drained_co2_Mg_yr"] - joined["nghgi_em_co2_Mg_yr"]
    joined["co2_ratio_model_over_nghgi"] = (
        joined["model_drained_co2_Mg_yr"] / joined["nghgi_em_co2_Mg_yr"]
    )

    return joined


# ----------------------------- plotting -----------------------------

def _plot_scatter_model_vs_nghgi(
    df: pd.DataFrame,
    value_model: str,
    value_nghgi: str,
    unit_label: str,
    title: str,
) -> plt.Figure:
    df = df.copy()
    df = df.dropna(subset=[value_model, value_nghgi])
    df = df[(df[value_model] > 0) & (df[value_nghgi] > 0)]

    theme = {**pc.THEME_LIGHT_GRID, "axes.grid.axis": "both"}
    with pc.use_theme(theme):
        fig, ax = plt.subplots(figsize=(6.5, 6.0))

        for lu in LANDUSE_ORDER:
            sub = df[df["land_use"] == lu]
            if sub.empty:
                continue
            ax.scatter(
                sub[value_nghgi],
                sub[value_model],
                label=lu,
                color=LANDUSE_COLORS.get(lu, "#444"),
                alpha=0.75,
                edgecolor="white",
                linewidth=0.5,
                s=42,
            )

        if not df.empty:
            lo = min(df[value_nghgi].min(), df[value_model].min())
            hi = max(df[value_nghgi].max(), df[value_model].max())
            lo = max(lo, 1e-6)
            ax.plot([lo, hi], [lo, hi], color="#888", linestyle="--", linewidth=1.0, label="1:1")
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_xlabel(f"NGHGI {unit_label}")
        ax.set_ylabel(f"Model {unit_label}")
        ax.set_title(title)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        pc.tidy_axes(ax, grid="both")
        fig.tight_layout()
        return fig


def _plot_topn_country_comparison(
    df: pd.DataFrame,
    value_model: str,
    value_nghgi: str,
    ylabel: str,
    title: str,
    topn: int = 15,
) -> plt.Figure:
    """Side-by-side bars: top N countries by NGHGI value."""
    df = df.copy()
    df = df.dropna(subset=[value_nghgi])
    df = df[df[value_nghgi] > 0]
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No comparable rows", ha="center", va="center")
        return fig

    per_country = (
        df.groupby(["iso3", "country"], as_index=False, observed=False)
        .agg({value_model: "sum", value_nghgi: "sum"})
        .sort_values(value_nghgi, ascending=False)
        .head(topn)
    )

    theme = {**pc.THEME_LIGHT_GRID, "axes.grid.axis": "y"}
    with pc.use_theme(theme):
        fig, ax = plt.subplots(figsize=(max(8, topn * 0.55), 4.5))
        xs = np.arange(len(per_country))
        w = 0.38
        labels = per_country["iso3"].where(
            per_country["iso3"].notna(), per_country["country"]
        ).fillna("?")

        ax.bar(xs - w / 2, per_country[value_nghgi], width=w, color="#4E79A7", label="NGHGI")
        ax.bar(xs + w / 2, per_country[value_model], width=w, color="#F28E2B", label="Model")

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False)
        pc.tidy_axes(ax, grid="y")
        pc.fmt_si(ax, axis="y")
        fig.tight_layout()
        return fig


def _plot_topn_area_stacked(
    df: pd.DataFrame,
    topn: int = 15,
    title: str = "Drained + undrained organic-soil area",
    yscale: str = "linear",
) -> plt.Figure:
    """
    Side-by-side stacked bars per country for NGHGI-reporting countries:
    NGHGI [drained | undrained] vs Model [drained | undrained].

    Ranks by NGHGI total organic area so the plot focuses on countries with
    inventory data (model-only countries are excluded).

    NGHGI 'undrained' is T4.A-F total organic - T4(II) drained, so it also
    carries any rewetted/managed-intact peat. Countries with only partial
    reporting show only the drained segment on the NGHGI side.
    """
    df = df.copy()

    per_country = (
        df.groupby(["iso3", "country"], as_index=False, observed=False)
        .agg({
            "model_drained_area_ha": "sum",
            "undrained_area_ha": "sum",
            "nghgi_area_drained_organic_ha": "sum",
            "nghgi_area_undrained_organic_ha": "sum",
            "nghgi_area_total_organic_ha": "sum",
        })
    )

    # Rank by NGHGI reporting extent (total organic if available, else drained)
    per_country["nghgi_rank"] = per_country["nghgi_area_total_organic_ha"].fillna(
        per_country["nghgi_area_drained_organic_ha"]
    )
    per_country = per_country.dropna(subset=["nghgi_rank"])
    per_country = per_country[per_country["nghgi_rank"] > 0]
    per_country = per_country.sort_values("nghgi_rank", ascending=False).head(topn)
    if per_country.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No comparable rows", ha="center", va="center")
        return fig

    labels = per_country["iso3"].where(
        per_country["iso3"].notna(), per_country["country"]
    ).fillna("?").tolist()

    nghgi_drained = per_country["nghgi_area_drained_organic_ha"].fillna(0).to_numpy()
    nghgi_undrain = per_country["nghgi_area_undrained_organic_ha"].fillna(0).to_numpy()
    model_drained = per_country["model_drained_area_ha"].fillna(0).to_numpy()
    model_undrain = per_country["undrained_area_ha"].fillna(0).to_numpy()

    # Solid = drained, hatched lighter = undrained. NGHGI blue / Model orange.
    c_nghgi_drained = "#2E5E8C"
    c_nghgi_undrain = "#A3BED3"
    c_model_drained = "#C26A0F"
    c_model_undrain = "#F3C17A"

    theme = {**pc.THEME_LIGHT_GRID, "axes.grid.axis": "y"}
    with pc.use_theme(theme):
        fig, ax = plt.subplots(figsize=(max(9, topn * 0.7), 5.0))
        xs = np.arange(len(per_country))
        w = 0.38

        ax.bar(xs - w / 2, nghgi_drained, width=w, color=c_nghgi_drained, label="NGHGI drained")
        ax.bar(
            xs - w / 2, nghgi_undrain, width=w, bottom=nghgi_drained,
            color=c_nghgi_undrain, label="NGHGI undrained (total − drained)",
        )
        ax.bar(xs + w / 2, model_drained, width=w, color=c_model_drained, label="Model drained")
        ax.bar(
            xs + w / 2, model_undrain, width=w, bottom=model_drained,
            color=c_model_undrain, label="Model undrained",
        )

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("organic-soil area (ha)")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
        pc.tidy_axes(ax, grid="y")
        if yscale == "log":
            ax.set_yscale("log")
            ymax = max(
                (nghgi_drained + nghgi_undrain).max(),
                (model_drained + model_undrain).max(),
            )
            ax.set_ylim(1e3, ymax * 3)
        else:
            pc.fmt_si(ax, axis="y")
        fig.tight_layout()
        return fig


# ----------------------------- validation -----------------------------

def _aggregate_raw_cstock_top_level(raw_cstock: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the raw cstock CSV to the same per-(iso3, year, land_use) shape
    that JRC reports: top-level land-use rows (4.A, 4.B, ...) only.

    Uses sum(min_count=1) so a group of all-NaN values (country reports IE/NE)
    stays NaN rather than collapsing to 0 — otherwise the comparison flags
    sentinel-only countries as "raw=0 vs jrc=numeric" disagreements.
    """
    top = raw_cstock[raw_cstock["category_code"].str.count(r"\.") == 1].copy()
    return (
        top.groupby(["iso3", "year", "land_use"], as_index=False, observed=False)
        .agg(
            raw_area_organic_kha=("area_organic_kha", lambda s: s.sum(min_count=1)),
            raw_cstock_organic_ktC=("cstock_soil_organic_ktC", lambda s: s.sum(min_count=1)),
        )
    )


def _plot_validation_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    unit_label: str,
    title: str,
) -> plt.Figure:
    theme = {**pc.THEME_LIGHT_GRID, "axes.grid.axis": "both"}
    with pc.use_theme(theme):
        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        for lu in LANDUSE_ORDER:
            sub = df[df["land_use"] == lu]
            if sub.empty:
                continue
            ax.scatter(
                sub[x_col], sub[y_col],
                label=lu, color=LANDUSE_COLORS.get(lu, "#444"),
                alpha=0.6, s=22, edgecolor="white", linewidth=0.4,
            )
        if not df.empty:
            lo = float(min(df[x_col].min(), df[y_col].min()))
            hi = float(max(df[x_col].max(), df[y_col].max()))
            ax.plot([lo, hi], [lo, hi], color="#888", linestyle="--", linewidth=1.0, label="1:1")
        ax.set_xlabel(f"Raw extract  {unit_label}")
        ax.set_ylabel(f"JRC aggregate  {unit_label}")
        ax.set_title(title)
        ax.legend(loc="best", frameon=False, fontsize=9)
        pc.tidy_axes(ax, grid="both")
        fig.tight_layout()
        return fig


def run_jrc_validation(
    nghgi_dir: str,
    jrc_dir: str,
    out_dir: str,
    rel_diff_threshold: float = 0.10,
) -> int:
    """
    Compare raw CRT-extracted Tables 4.A-F against the JRC AnnexI 2026
    aggregates for the country-year-land-use rows present in both. Writes
    one scatter PNG per variable and a CSV of the joined rows with diffs.
    """
    print("\n[validation] Loading raw cstock CSV")
    raw_cstock = _load_nghgi_cstock(nghgi_dir)
    raw_agg = _aggregate_raw_cstock_top_level(raw_cstock)
    raw_iso = sorted(raw_agg["iso3"].dropna().unique())
    print(f"  raw iso3 ({len(raw_iso)}): {', '.join(raw_iso)}")

    print("\n[validation] Loading JRC AnnexI 2026 land-use tables")
    annexi_dir, _btr1 = jrc_loader.default_jrc_paths(jrc_dir)
    jrc_lu = jrc_loader.load_jrc_landuse_tables(annexi_dir)
    jrc_iso = sorted(jrc_lu["iso3"].dropna().unique())
    print(f"  jrc iso3 ({len(jrc_iso)}): {', '.join(jrc_iso)}")

    overlap_iso = sorted(set(raw_iso) & set(jrc_iso))
    raw_only = sorted(set(raw_iso) - set(jrc_iso))
    print(f"\n[validation] Overlap iso3 ({len(overlap_iso)}): {', '.join(overlap_iso)}")
    print(f"  raw-only (kept on raw side permanently): {', '.join(raw_only)}")

    jrc_agg = jrc_lu.rename(
        columns={
            "area_organic_kha": "jrc_area_organic_kha",
            "cstock_organic_ktC": "jrc_cstock_organic_ktC",
        }
    )[["iso3", "year", "land_use", "jrc_area_organic_kha", "jrc_cstock_organic_ktC"]]

    # Normalize Int64 dtypes so the merge keys align
    raw_agg["year"] = pd.to_numeric(raw_agg["year"], errors="coerce").astype("Int64")
    jrc_agg["year"] = pd.to_numeric(jrc_agg["year"], errors="coerce").astype("Int64")

    merged = raw_agg.merge(jrc_agg, on=["iso3", "year", "land_use"], how="inner")
    merged = merged[merged["iso3"].isin(overlap_iso)].copy()

    # Relative diff = (jrc - raw) / max(|raw|, |jrc|); avoids div-by-zero on cstock
    for var, raw_col, jrc_col in [
        ("area", "raw_area_organic_kha", "jrc_area_organic_kha"),
        ("cstock", "raw_cstock_organic_ktC", "jrc_cstock_organic_ktC"),
    ]:
        denom = np.maximum(merged[raw_col].abs(), merged[jrc_col].abs())
        merged[f"{var}_abs_diff"] = merged[jrc_col] - merged[raw_col]
        merged[f"{var}_rel_diff"] = np.where(
            denom > 0, (merged[jrc_col] - merged[raw_col]) / denom, np.nan
        )

    out_data = _join(out_dir, "figures", "validation", "data")
    writer_con = duckdb.connect()
    try:
        _write_csv_df(writer_con, merged, _join(out_data, "validation_overlap.csv"))
    finally:
        writer_con.close()

    print(f"\n[validation] Joined overlap rows: {len(merged):,}")
    for var, raw_col, jrc_col, unit_label, fig_label in [
        ("area",   "raw_area_organic_kha",   "jrc_area_organic_kha",
         "organic-soil area (kha)",          "area_organic"),
        ("cstock", "raw_cstock_organic_ktC", "jrc_cstock_organic_ktC",
         "organic-soil cstock (kt C)",       "cstock_organic"),
    ]:
        sub = merged.dropna(subset=[raw_col, jrc_col])
        n = len(sub)
        n_disagree = int((sub[f"{var}_rel_diff"].abs() > rel_diff_threshold).sum())
        worst = sub.assign(_a=sub[f"{var}_rel_diff"].abs()).nlargest(5, "_a")
        print(
            f"  [{var}] both-numeric rows: {n:,}  "
            f">{rel_diff_threshold:.0%} disagree: {n_disagree}"
        )
        if n_disagree:
            print(f"    top-5 worst: ")
            for _, r in worst.iterrows():
                print(
                    f"      {r['iso3']} {int(r['year']):d} {r['land_use']:<10s}  "
                    f"raw={r[raw_col]:>12.3f}  jrc={r[jrc_col]:>12.3f}  "
                    f"rel_diff={r[f'{var}_rel_diff']:+.2%}"
                )

        fig = _plot_validation_scatter(
            sub, raw_col, jrc_col,
            unit_label=unit_label,
            title=f"Raw extract vs JRC aggregate -- {unit_label}",
        )
        _save_png(
            fig,
            _join(out_dir, "figures", "validation", f"scatter_{fig_label}_raw_vs_jrc.png"),
            dpi=200,
        )

    print("\n[validation] Wrote:")
    print(f"  - figures/validation/scatter_area_organic_raw_vs_jrc.png")
    print(f"  - figures/validation/scatter_cstock_organic_raw_vs_jrc.png")
    print(f"  - figures/validation/data/validation_overlap.csv")
    return 0


# ----------------------------- driver -----------------------------

def main(argv=None):
    global OUT_DIR_ROOT, OUT_DIR
    p = argparse.ArgumentParser(
        "Compare model drained organic-soil outputs to country NGHGI CRT reports."
    )
    p.add_argument(
        "--years",
        nargs="+",
        required=False,
        default=None,
        type=int,
        help=(
            "One or more inventory end years (e.g. 2005 2010 2015 2020 2024). "
            "Required unless --validate_jrc is set."
        ),
    )
    p.add_argument(
        "--run",
        action="append",
        required=False,
        default=None,
        help=(
            "Run spec: run_name=model_version:run_date[|Label] "
            "(e.g. 'zarr_test_full=0_1_2:20260403|GFW'). One per model source. "
            "Required unless --validate_jrc is set."
        ),
    )
    p.add_argument(
        "--nghgi_dir",
        default=DEFAULT_NGHGI_DIR,
        help=(
            f"Directory with compiled NGHGI CSVs "
            f"({NGHGI_TABLE_4II_NAME}, {NGHGI_CSTOCK_NAME}). "
            f"Default: {DEFAULT_NGHGI_DIR}"
        ),
    )
    p.add_argument(
        "--jrc_dir",
        default=DEFAULT_JRC_DIR,
        help=(
            f"Directory containing the JRC-aggregated drop "
            f"(AnnexI_2026/, BTR1_2024_Table3D.xlsx). Default: {DEFAULT_JRC_DIR}"
        ),
    )
    p.add_argument(
        "--validate_jrc",
        action="store_true",
        help=(
            "Validation-only mode: compare raw-extract Tables 4.A-F against "
            "the JRC AnnexI 2026 aggregates on overlapping iso3/year/land-use "
            "rows, write scatter PNGs and a diff CSV under figures/validation/, "
            "then exit. Skips the model side and the regular figure pipeline."
        ),
    )
    p.add_argument("--aws_region", default=None)
    p.add_argument(
        "--adm0_lookup_csv",
        default=None,
        help="Optional CSV with gadm_adm0 -> iso3/country overrides.",
    )
    p.add_argument("--data-only", action="store_true")
    p.add_argument(
        "--out_dir_root",
        default=OUT_DIR_ROOT,
        help=f"Output root (default: {OUT_DIR_ROOT}).",
    )
    args = p.parse_args(argv)

    OUT_DIR_ROOT = args.out_dir_root

    if args.validate_jrc:
        drop_label = os.path.basename(os.path.normpath(args.jrc_dir)) or "jrc"
        out_dir = _join(OUT_DIR_ROOT, "validation", drop_label)
        print("NGHGI validation mode (raw extract vs JRC AnnexI 2026)")
        print("  nghgi_dir:", args.nghgi_dir)
        print("  jrc_dir  :", args.jrc_dir)
        print("  output dir:", out_dir)
        return run_jrc_validation(args.nghgi_dir, args.jrc_dir, out_dir)

    if not args.years:
        p.error("--years is required (unless --validate_jrc is set)")
    if not args.run:
        p.error("--run is required (unless --validate_jrc is set)")

    years = sorted({int(y) for y in args.years})
    run_specs = _parse_run_specs(args.run)
    interval_pairs = build_interval_pairs(years)

    primary = run_specs[0]
    OUT_DIR = build_output_dir(primary.model_version, primary.run_name, primary.run_date)

    print("NGHGI comparison")
    print("  intervals:", [f"{s}_{e}" for s, e in interval_pairs])
    for spec in run_specs:
        print(f"  model: {spec.label} (run={spec.run_name} v={spec.model_version} date={spec.run_date})")
    print("  nghgi_dir:", args.nghgi_dir)
    print("  output dir:", OUT_DIR)

    # --- Model side ---
    print("\n[1/3] Reading model zonal stats")
    model_df = load_model_country_landuse(
        run_specs, years, args.aws_region, args.adm0_lookup_csv
    )
    if model_df.empty:
        print("  No model rows found for any (run, interval). Aborting.")
        return 1
    print(f"  model rows: {len(model_df):,}")

    # --- NGHGI side ---
    print("\n[2/3] Loading compiled NGHGI CSVs")
    t4ii = _load_nghgi_compiled(args.nghgi_dir)
    cstock = _load_nghgi_cstock(args.nghgi_dir)
    print(f"  Table 4(II) rows: {len(t4ii):,}  (organic-soil rows only)")
    print(f"  Table 4.A-F rows: {len(cstock):,}")

    nghgi_df = nghgi_by_iso_landuse_interval(t4ii, cstock, interval_pairs)
    print(f"  NGHGI rows after averaging: {len(nghgi_df):,}")

    # --- Join ---
    print("\n[3/3] Joining model + NGHGI")
    joined = join_model_nghgi(model_df, nghgi_df)
    print(f"  joined rows: {len(joined):,}")

    # --- Write tables ---
    out_data = _join(OUT_DIR, "figures", "data")
    writer_con = duckdb.connect()
    try:
        _write_csv_df(writer_con, model_df,  _join(out_data, "model_country_landuse.csv"))
        _write_csv_df(writer_con, nghgi_df,  _join(out_data, "nghgi_country_landuse.csv"))
        _write_csv_df(writer_con, joined,    _join(out_data, "model_vs_nghgi.csv"))
        print("  wrote:")
        print("    - model_country_landuse.csv")
        print("    - nghgi_country_landuse.csv")
        print("    - model_vs_nghgi.csv")
    finally:
        writer_con.close()

    if args.data_only:
        print("\nData-only mode: figures skipped.")
        return 0

    # --- Figures ---
    print("\nBuilding figures")

    # One set of figures per interval (clearer comparison than pooling all years)
    for interval in sorted(joined["interval"].dropna().unique()):
        sub = joined[joined["interval"] == interval].copy()

        # Scatter: area
        fig = _plot_scatter_model_vs_nghgi(
            sub,
            value_model="model_drained_area_ha",
            value_nghgi="nghgi_area_drained_organic_ha",
            unit_label="drained organic-soil area (ha)",
            title=f"Model vs NGHGI drained organic-soil area ({interval})",
        )
        _save_png(fig, _join(OUT_DIR, "figures", f"scatter_area_{interval}.png"), dpi=200)

        # Scatter: CO2
        fig = _plot_scatter_model_vs_nghgi(
            sub,
            value_model="model_drained_co2_Mg_yr",
            value_nghgi="nghgi_em_co2_Mg_yr",
            unit_label="drained organic-soil CO2 (Mg CO2/yr)",
            title=f"Model vs NGHGI drained organic-soil CO2 ({interval})",
        )
        _save_png(fig, _join(OUT_DIR, "figures", f"scatter_co2_{interval}.png"), dpi=200)

        # Top-N country bar: area
        fig = _plot_topn_country_comparison(
            sub,
            value_model="model_drained_area_ha",
            value_nghgi="nghgi_area_drained_organic_ha",
            ylabel="drained organic-soil area (ha)",
            title=f"Top countries by NGHGI drained organic-soil area ({interval})",
        )
        _save_png(fig, _join(OUT_DIR, "figures", f"topn_area_{interval}.png"), dpi=200)

        # Top-N country bar: CO2
        fig = _plot_topn_country_comparison(
            sub,
            value_model="model_drained_co2_Mg_yr",
            value_nghgi="nghgi_em_co2_Mg_yr",
            ylabel="drained organic-soil CO2 (Mg CO2/yr)",
            title=f"Top countries by NGHGI drained organic-soil CO2 ({interval})",
        )
        _save_png(fig, _join(OUT_DIR, "figures", f"topn_co2_{interval}.png"), dpi=200)

        # Stacked drained + undrained area (linear)
        fig = _plot_topn_area_stacked(
            sub,
            title=f"NGHGI vs Model organic-soil area, drained + undrained ({interval})",
        )
        _save_png(fig, _join(OUT_DIR, "figures", f"topn_area_stacked_{interval}.png"), dpi=200)

        # Stacked drained + undrained area (log-y for cross-country readability)
        fig = _plot_topn_area_stacked(
            sub,
            title=f"NGHGI vs Model organic-soil area, drained + undrained ({interval}, log scale)",
            yscale="log",
        )
        _save_png(fig, _join(OUT_DIR, "figures", f"topn_area_stacked_log_{interval}.png"), dpi=200)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
