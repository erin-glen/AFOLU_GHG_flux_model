# -*- coding: utf-8 -*-
"""
Generate publication figures & Flourish-ready datasets for climate-domain stacks.

Figures (saved under --out_dir):
  1) global_drained_climate_column.(png|svg)
     Data: figures/data/global_drained_climate_{long|wide}.csv
  2) global_burn_climate_bar.(png|svg)
     Data: figures/data/global_burn_climate_{long|wide}.csv

Notes
-----
- Reuses your DuckDB registration from pub_tables.py to keep one source of truth.
- Converts Mg -> Gt by dividing by 1e9.
- Climate domain order is Boreal, Temperate, Tropical (title-cased).
- CSVs are written in both long and wide formats (wide is convenient for Flourish).

Examples
--------
python -m src.scripts.zonal_statistics.pub_figures \
  --model_version 0_7_0 \
  --run_name ogh_standard_model \
  --run_date 20250825 \
  --years 2005 2010 2015 2020 2024 \
  --out_dir /mnt/c/tmp/pub_figs

# Write only CSVs (no images)
python -m src.scripts.zonal_statistics.pub_figures \
  --model_version 0_7_5 --run_name ogh_standard_model --run_date 20250914 \
  --years 2005 2010 2015 2020 2024 \
  --out_dir s3://your-bucket/outputs/publication \
  --data-only

# PNG only (no SVG)
python -m src.scripts.zonal_statistics.pub_figures \
  --model_version 0_7_5 --run_name ogh_standard_model --run_date 20250914 \
  --years 2005 2010 2015 2020 2024 \
  --out_dir /mnt/c/tmp/pub_figs \
  --no-svg
"""

from __future__ import annotations
import argparse
import posixpath
from pathlib import Path
from typing import Iterable, List, Tuple

import duckdb
import pandas as pd
import matplotlib.pyplot as plt

# --- Reuse helpers from pub_tables to avoid drift ---
from .pub_tables import (  # type: ignore
    _register_all,
    _register_state_context_views,
    _make_globs_for_components,
    build_interval_pairs,
    build_output_parquet,
)


# --------------------------- small utilities ---------------------------

CLIMATE_ORDER = ["Boreal", "Temperate", "Tropical"]

def _interval_folder_strings(years: Iterable[int]) -> List[str]:
    pairs: List[Tuple[int, int]] = build_interval_pairs(list(years))
    return [f"{s}_{e}" for (s, e) in pairs]

def _make_base_prefixes(model_version: str, run_name: str, run_date: str,
                        interval_folders: Iterable[str]) -> List[str]:
    bases: List[str] = []
    for interval in interval_folders:
        bases.append(build_output_parquet(model_version, run_name, run_date, interval).rstrip("/"))
    return bases

def _ensure_dir(path: str):
    if path.startswith("s3://"):
        return
    Path(path).mkdir(parents=True, exist_ok=True)

def _titlecase_domain(s: str | None) -> str:
    if s is None:
        return "Unspecified"
    t = s.strip().title()
    # Normalize common forms
    if t.startswith("Boreal"):
        return "Boreal"
    if t.startswith("Temperate"):
        return "Temperate"
    if t.startswith("Tropical"):
        return "Tropical"
    return t or "Unspecified"

def _pivot_wide(df_long: pd.DataFrame, value_col: str) -> pd.DataFrame:
    wide = (
        df_long
        .pivot_table(index="Year", columns="Climate", values=value_col, aggfunc="sum", fill_value=0.0)
        .reindex(columns=CLIMATE_ORDER, fill_value=0.0)
        .reset_index()
    )
    return wide

def _write_csv(df: pd.DataFrame, path: str):
    _ensure_dir(posixpath.dirname(path))
    df.to_csv(path, index=False)

def _write_png(fig: plt.Figure, path: str, dpi: int = 300, width: float | None = None, height: float | None = None):
    _ensure_dir(posixpath.dirname(path))
    if width and height:
        fig.set_size_inches(width, height)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")

def _write_svg(fig: plt.Figure, path: str, width: float | None = None, height: float | None = None):
    _ensure_dir(posixpath.dirname(path))
    if width and height:
        fig.set_size_inches(width, height)
    fig.savefig(path, format="svg", bbox_inches="tight")

# --------------------------- SQL makers ---------------------------

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
    SELECT
      interval_end,
      climate_domain,
      drained_MgCO2e / 1e9 AS drained_GtCO2e
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
    SELECT
      interval_end,
      climate_domain,
      burned_MgCO2e / 1e9 AS burned_GtCO2e
    FROM joined
    ORDER BY interval_end, climate_domain;
    """

# --------------------------- plotting recipes ---------------------------

def _stacked_bar(df_long: pd.DataFrame, value_col: str, title: str) -> plt.Figure:
    # df_long: columns Year, Climate, value_col
    # ensure order
    df_long = df_long.copy()
    df_long["Climate"] = pd.Categorical(df_long["Climate"], CLIMATE_ORDER, ordered=True)
    years = list(df_long["Year"].unique())
    years.sort()

    # build wide for plotting
    wide = _pivot_wide(df_long, value_col)
    wide = wide.set_index("Year")
    # ensure consistent order
    wide = wide.reindex(index=years)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bottom = None
    for climate in CLIMATE_ORDER:
        vals = wide[climate].values
        ax.bar(wide.index.astype(str), vals, bottom=bottom, label=climate)
        bottom = (vals if bottom is None else bottom + vals)

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Emissions (Gt CO₂e/year)")
    ax.set_title(title)
    ax.legend(title=None, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.15))
    ax.margins(x=0.02)
    fig.tight_layout()
    return fig

# --------------------------- main entry ---------------------------

def main(argv=None):
    ap = argparse.ArgumentParser("Build climate-domain stacked figures + CSVs")
    ap.add_argument("--model_version", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--run_date", required=True)
    ap.add_argument("--years", nargs="+", required=True, help="Interval end years e.g. 2005 2010 2015 2020 2024")
    ap.add_argument("--aws_region", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data-only", action="store_true", help="Write CSVs only (no images)")
    ap.add_argument("--no-png", action="store_true", help="Skip PNG outputs")
    ap.add_argument("--no-svg", action="store_true", help="Skip SVG outputs")
    args = ap.parse_args(argv)

    years = [int(y) for y in args.years]
    interval_folders = _interval_folder_strings(years)
    base_prefixes = _make_base_prefixes(args.model_version, args.run_name, args.run_date, interval_folders)
    drained_globs, burned_globs = _make_globs_for_components(base_prefixes)

    con = duckdb.connect()
    _register_all(con, drained_globs, burned_globs, aws_region=args.aws_region)
    _register_state_context_views(con)

    # ---------- drained ----------
    df_d = con.execute(sql_drained_by_climate()).df()
    df_d["Climate"] = df_d["climate_domain"].apply(_titlecase_domain)
    df_d = df_d[["interval_end", "Climate", "drained_GtCO2e"]].rename(columns={"interval_end": "Year"})
    df_d = df_d[df_d["Climate"].isin(CLIMATE_ORDER)]  # keep canonical three

    # Write CSVs
    drained_long_csv = posixpath.join(args.out_dir, "figures/data/global_drained_climate_long.csv")
    drained_wide_csv = posixpath.join(args.out_dir, "figures/data/global_drained_climate_wide.csv")
    _write_csv(df_d, drained_long_csv)
    _write_csv(_pivot_wide(df_d, "drained_GtCO2e"), drained_wide_csv)

    if not args.data_only:
        fig_d = _stacked_bar(df_d, "drained_GtCO2e", "Annual Emissions (Gt CO₂e/year)")
        if not args.no_png:
            _write_png(fig_d, posixpath.join(args.out_dir, "figures/global_drained_climate_column.png"))
        if not args.no_svg:
            _write_svg(fig_d, posixpath.join(args.out_dir, "figures/global_drained_climate_column.svg"))

    # ---------- burned ----------
    df_b = con.execute(sql_burned_by_climate()).df()
    df_b["Climate"] = df_b["climate_domain"].apply(_titlecase_domain)
    df_b = df_b[["interval_end", "Climate", "burned_GtCO2e"]].rename(columns={"interval_end": "Year"})
    df_b = df_b[df_b["Climate"].isin(CLIMATE_ORDER)]

    burned_long_csv = posixpath.join(args.out_dir, "figures/data/global_burn_climate_long.csv")
    burned_wide_csv = posixpath.join(args.out_dir, "figures/data/global_burn_climate_wide.csv")
    _write_csv(df_b, burned_long_csv)
    _write_csv(_pivot_wide(df_b, "burned_GtCO2e"), burned_wide_csv)

    if not args.data_only:
        fig_b = _stacked_bar(df_b, "burned_GtCO2e", "Annual Emissions (Gt CO₂e/year)")
        if not args.no_png:
            _write_png(fig_b, posixpath.join(args.out_dir, "figures/global_burn_climate_bar.png"))
        if not args.no_svg:
            _write_svg(fig_b, posixpath.join(args.out_dir, "figures/global_burn_climate_bar.svg"))

    print("Wrote figures & data to:", args.out_dir)


if __name__ == "__main__":
    main()
