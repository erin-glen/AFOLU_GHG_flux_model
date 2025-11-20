# -*- coding: utf-8 -*-
"""Build comparison figures across multiple zonal-statistics runs.

Arguments mirror ``pub_assets`` conventions:

* ``--years`` – one or more inventory period end years (``YYYY``).
* ``--run`` – repeatable run specification in the form
  ``run_name=model_version:run_date`` (optionally ``|Custom Label``).
  ``run_date`` **must** be formatted as ``YYYYMMDD`` to align with
  :func:`_make_base_prefixes` and downstream file discovery.
  When adding a custom label (anything after ``|``) quote the entire
  argument so your shell does not treat the pipe character as a command
  separator; e.g. ``--run "gfw_standard_model_1km=0_8_5:20251006|GFW 1 km"``.
* ``--aws_region`` – optional AWS region for S3 access (mirrors
  ``pub_assets`` default behaviour).
* ``--data-only`` – skip figure generation and export CSV data only.
* ``--flag-abs-mha`` – absolute spread threshold (Mha) for FLAGging countries
  in the disagreement summary (default 0.1 = 100,000 ha).
* ``--flag-rel-fold`` – relative spread threshold (fold-change) for FLAGging
  countries in the disagreement summary (default 10x).

Each comparison defined below expects a specific set of run names. When a
comparison's requirements are not fully met, the script will skip that
comparison and emit a note summarizing the missing runs. Provide
additional ``--run`` entries if you need to compare multiple model
versions or reruns of the same scenario.

Outputs are grouped under ``/mnt/c/tmp/pub_assets/comparisons/<run_dates>/<run_names>/``
to mirror the main driver folder hierarchy.

Usage example:

  cd /mnt/c/gis/git/AFOLU_GHG_flux_model

  python -m src.scripts.zonal_statistics.pub_compare_runs \
    --years 2005 2010 2015 2020 2024 \
    --run ogh_sensitivity_0km=0_8_5:20251006 \
    --run ogh_sensitivity_1km=0_8_5:20251006 \
    --run ogh_sensitivity_2km=0_8_5:20251006 \
    --run "ogh_sensitivity_high=0_8_5:20251006|OGH High" \
    --run "ogh_sensitivity_low=0_8_5:20251006|OGH Low" \
    --run "gfw_standard_model_1km=0_8_5:20251006|GFW 1 km" \
    --run "gpd_standard_model_1km=0_8_5:20251007|GPD 1 km" \
    --run "gpd_standard_model_1km_pml=0_8_5:20251007|GPD 1 km (PML)"

To generate a subset, provide only the runs required for the comparisons
you care about. For example, the following command builds the inventory
input comparison while skipping the OGH sensitivity plots:

  python -m src.scripts.zonal_statistics.pub_compare_runs \
    --years 2024 \
    --run ogh_sensitivity_1km=0_8_5:20251002 \
    --run "gfw_standard_model_1km=0_8_5:20251006|GFW 1 km" \
    --run "gpd_standard_model_1km=0_8_5:20251007|GPD 1 km" \
    --run "gpd_standard_model_1km_pml=0_8_5:20251008|GPD 1 km (PML)"

Each run name may specify a custom label after ``|`` that will be used in
the exported tables and figure titles. If you skip the custom label the
script will derive one automatically from the run name. In addition to the
comparison summaries, the script exports run-level tables mirroring the
``pub_assets`` outputs (by country, drained state, burned state, and their
country/state intersections). Country-level tables include the run metadata
columns (``run_name``, ``Run``, ``model_version``, ``run_date``). Country
tables also carry best-effort ISO3 and country name lookups using the same
helper logic as ``pub_assets``.

NEW: For the **Inventory Input Source Comparison**, this script now also exports
per-country **disagreement** tables for total peat area (Mha) across the source
runs (GFW / GPD / GPD-PML / OGH), including min/median/max, absolute spread,
fold-change, log10 spread, and the min/max contributing dataset, plus a flagged
subset based on user-tunable thresholds.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence
import math

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

import src.scripts.zonal_statistics.pub_common as pc
import src.scripts.zonal_statistics.pub_assets as pa


OUT_DIR_ROOT = pa.OUT_DIR_ROOT

_join = pa._join
_save_png = pa._save_png
_write_csv_df = pa._write_csv_df
_register_components = pa._register_components
_register_state_context_views = pa._register_state_context_views
_ensure_adm0_lookup = pa._ensure_adm0_lookup
_interval_folder_strings = pa._interval_folder_strings
_make_base_prefixes = pa._make_base_prefixes
_make_globs_for_components = pa._make_globs_for_components

STACK_COMPONENT_COLORS = {"Drained": "#4f81bd", "Burned": "#c0504d"}


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    model_version: str
    run_date: str
    label: str


@dataclass
class RunMetrics:
    drained_area_mha: float
    undrained_area_mha: float
    drained_emissions_gt: float
    burned_emissions_gt: float

    @property
    def peat_area_mha(self) -> float:
        return self.drained_area_mha + self.undrained_area_mha

    @property
    def total_emissions_gt(self) -> float:
        return self.drained_emissions_gt + self.burned_emissions_gt


@dataclass
class RunBreakouts:
    by_country: pd.DataFrame
    by_drained_state: pd.DataFrame
    by_burned_state: pd.DataFrame
    by_country_drained_state: pd.DataFrame
    by_country_burned_state: pd.DataFrame


@dataclass(frozen=True)
class MetricUncertainty:
    drained_low: float | None = None
    drained_high: float | None = None
    burned_low: float | None = None
    burned_high: float | None = None
    total_low: float | None = None
    total_high: float | None = None

    def has_bounds(self) -> bool:
        return any(
            v is not None
            for v in (
                self.drained_low,
                self.drained_high,
                self.burned_low,
                self.burned_high,
                self.total_low,
                self.total_high,
            )
        )


@dataclass(frozen=True)
class RunRecord:
    spec: RunSpec
    metrics: RunMetrics
    breakouts: RunBreakouts
    color: str
    uncertainty: MetricUncertainty | None = None


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    units: str
    extractor: Callable[[RunMetrics], float]


@dataclass(frozen=True)
class ComparisonSpec:
    key: str
    label: str
    run_names: tuple[str, ...]
    metric_keys: tuple[str, ...]


def _comparison_out_dir(run_specs: Mapping[str, RunSpec]) -> str:
    """Build comparison output directory segmented by run dates and names."""

    run_dates = sorted({spec.run_date for spec in run_specs.values()})
    run_names = sorted(run_specs.keys())
    date_slug = "__".join(run_dates) if run_dates else "unspecified_dates"
    name_slug = "__".join(run_names) if run_names else "unspecified_runs"
    return _join(OUT_DIR_ROOT, "comparisons", date_slug, name_slug)


METRIC_SPECS: Mapping[str, MetricSpec] = {
    "peat_total_area": MetricSpec(
        key="peat_total_area",
        label="Total peat area",
        units="million ha",
        extractor=lambda m: m.peat_area_mha,
    ),
    "peat_drained_area": MetricSpec(
        key="peat_drained_area",
        label="Drained peat area",
        units="million ha",
        extractor=lambda m: m.drained_area_mha,
    ),
    "drained_emissions": MetricSpec(
        key="drained_emissions",
        label="Drained emissions",
        units="Gt CO₂e/year",
        extractor=lambda m: m.drained_emissions_gt,
    ),
    "burned_emissions": MetricSpec(
        key="burned_emissions",
        label="Burned emissions",
        units="Gt CO₂e/year",
        extractor=lambda m: m.burned_emissions_gt,
    ),
    "total_emissions": MetricSpec(
        key="total_emissions",
        label="Total emissions",
        units="Gt CO₂e/year",
        extractor=lambda m: m.total_emissions_gt,
    ),
}


COMPARISONS: Sequence[ComparisonSpec] = (
    ComparisonSpec(
        key="ogh_resolution",
        label="OGH Sensitivity (Spatial Resolution)",
        run_names=("ogh_sensitivity_1km", "ogh_sensitivity_2km", "ogh_sensitivity_0km"),
        metric_keys=("peat_drained_area", "drained_emissions"),
    ),
    ComparisonSpec(
        key="inventory_source",
        label="Inventory Input Source Comparison",
        run_names=(
            "ogh_sensitivity_1km",
            "gfw_standard_model_1km",
            "gpd_standard_model_1km",
            "gpd_standard_model_1km_pml",
        ),
        metric_keys=("peat_total_area", "peat_drained_area", "drained_emissions", "burned_emissions"),
    ),
    ComparisonSpec(
        key="ogh_sensitivity_range",
        label="OGH Sensitivity (High/Low Emissions)",
        run_names=("ogh_sensitivity_1km", "ogh_sensitivity_high", "ogh_sensitivity_low"),
        metric_keys=("drained_emissions", "burned_emissions", "total_emissions"),
    ),
)


def _partition_comparisons(
    run_specs: Mapping[str, RunSpec],
) -> tuple[list[ComparisonSpec], Mapping[str, tuple[str, ...]]]:
    active: list[ComparisonSpec] = []
    missing: dict[str, tuple[str, ...]] = {}
    for comp in COMPARISONS:
        missing_runs = tuple(run for run in comp.run_names if run not in run_specs)
        if missing_runs:
            missing[comp.key] = missing_runs
        else:
            active.append(comp)
    return active, missing


def _default_label(run_name: str) -> str:
    label = run_name.replace("_", " ").replace("-", " ").strip()
    return label or run_name


def _parse_run_specs(entries: Sequence[str]) -> Mapping[str, RunSpec]:
    specs: dict[str, RunSpec] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --run specification (expected name=model_version:run_date): {entry}")
        raw_name, rest = entry.split("=", 1)
        run_name = raw_name.strip()
        if not run_name:
            raise ValueError(f"Invalid run name in --run specification: {entry}")

        if "|" in rest:
            config_part, label_part = rest.split("|", 1)
            label = label_part.strip() or _default_label(run_name)
        else:
            config_part = rest
            label = _default_label(run_name)

        parts = [p.strip() for p in config_part.split(":") if p.strip()]
        if len(parts) != 2:
            raise ValueError(
                "Invalid --run specification (expected model_version:run_date or model_version:run_date|Label): "
                f"{entry}"
            )
        model_version, run_date = parts
        specs[run_name] = RunSpec(run_name=run_name, model_version=model_version, run_date=run_date, label=label)
    return specs


def _parse_chunk_stat_paths(entries: Sequence[str] | None) -> Mapping[str, str]:
    mapping: dict[str, str] = {}
    if not entries:
        return mapping

    for entry in entries:
        if "=" not in entry:
            raise ValueError("Invalid --chunk-stats specification (expected run_name=path)")
        run_name, path = entry.split("=", 1)
        run_name = run_name.strip()
        path = path.strip()
        if not run_name or not path:
            raise ValueError("Invalid --chunk-stats specification (expected run_name=path)")
        mapping[run_name] = path
    return mapping


def _assign_colors(run_names: Iterable[str]) -> Mapping[str, str]:
    cmap = plt.get_cmap("tab10")
    ordered = sorted(dict.fromkeys(run_names))
    return {run: mcolors.to_hex(cmap(i % cmap.N)) for i, run in enumerate(ordered)}


def _add_run_columns(df: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
    meta = {
        "run_name": spec.run_name,
        "Run": spec.label,
        "model_version": spec.model_version,
        "run_date": spec.run_date,
    }
    df = df.copy()
    for col, value in meta.items():
        df[col] = value
    ordered_cols = list(meta.keys()) + [c for c in df.columns if c not in meta]
    return df[ordered_cols]


def _compute_run_data(
    spec: RunSpec, years: Sequence[int], aws_region: str | None, chunk_stats_path: str | None
) -> tuple[RunMetrics, RunBreakouts, MetricUncertainty | None]:
    interval_folders = _interval_folder_strings(years)
    base_prefixes = _make_base_prefixes(spec.model_version, spec.run_name, spec.run_date, interval_folders)
    drained_globs, burned_globs = _make_globs_for_components(base_prefixes)

    con = duckdb.connect()
    uncertainty: MetricUncertainty | None = None
    try:
        _register_components(con, drained_globs, burned_globs, aws_region=aws_region)
        _register_state_context_views(con)
        have_lookup = _ensure_adm0_lookup(con, None)

        latest_year = max(years)
        area_df = con.execute(pc.sql_global_peat_area_split(latest_year)).df()
        area_map = {row["peat_state"]: float(row["area_mha"]) for _, row in area_df.iterrows()}
        drained_area = float(area_map.get("drained", 0.0))
        undrained_area = float(area_map.get("undrained", 0.0))

        n_periods = len(years)
        emissions_df = con.execute(pc.sql_global_component_emissions_avg(n_periods)).df()
        emissions_map = {row["component"]: float(row["avg_GtCO2e_per_yr"]) for _, row in emissions_df.iterrows()}
        drained_emissions = float(emissions_map.get("Drained", 0.0))
        burned_emissions = float(emissions_map.get("Burned", 0.0))

        by_country = con.execute(pc.table_by_country_period_sql(with_lookup=have_lookup)).df()
        by_drained_state = con.execute(pc.table_by_drained_state_sql()).df()
        by_burned_state = con.execute(pc.table_by_burned_state_sql()).df()
        by_country_drained_state = con.execute(
            pc.table_by_country_drained_state_sql(with_lookup=have_lookup)
        ).df()
        by_country_burned_state = con.execute(
            pc.table_by_country_burned_state_sql(with_lookup=have_lookup)
        ).df()

        if chunk_stats_path:
            loaded = _load_chunk_stats_bounds(chunk_stats_path, years)
            if loaded is not None:
                stats_override, bounds = loaded
                uncertainty = bounds if bounds.has_bounds() else None
                drained_emissions = float(stats_override.get("drained_emissions_gt", drained_emissions))
                burned_emissions = float(stats_override.get("burned_emissions_gt", burned_emissions))
    finally:
        con.close()

    metrics = RunMetrics(
        drained_area_mha=drained_area,
        undrained_area_mha=undrained_area,
        drained_emissions_gt=drained_emissions,
        burned_emissions_gt=burned_emissions,
    )

    breakouts = RunBreakouts(
        by_country=_add_run_columns(by_country, spec),
        by_drained_state=_add_run_columns(by_drained_state, spec),
        by_burned_state=_add_run_columns(by_burned_state, spec),
        by_country_drained_state=_add_run_columns(by_country_drained_state, spec),
        by_country_burned_state=_add_run_columns(by_country_burned_state, spec),
    )

    return metrics, breakouts, uncertainty


def _summary_column(metric: MetricSpec) -> str:
    return f"{metric.label} ({metric.units})"


def _build_comparison_summary(comp: ComparisonSpec, records: Mapping[str, RunRecord]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for run_name in comp.run_names:
        record = records[run_name]
        row: dict[str, float | str] = {"Run": record.spec.label}
        for metric_key in comp.metric_keys:
            metric = METRIC_SPECS[metric_key]
            row[_summary_column(metric)] = metric.extractor(record.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_metric_long_df(comp: ComparisonSpec, metric: MetricSpec, records: Mapping[str, RunRecord]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for run_name in comp.run_names:
        record = records[run_name]
        rows.append({
            "run_key": run_name,
            "Run": record.spec.label,
            "Value": metric.extractor(record.metrics),
            "Metric": metric.label,
            "Units": metric.units,
        })
    return pd.DataFrame(rows)


def _plot_metric(df: pd.DataFrame, metric: MetricSpec, comp: ComparisonSpec, colors: Sequence[str]) -> plt.Figure:
    labels = df["Run"].tolist()
    values = df["Value"].tolist()
    y_positions = list(range(len(labels)))
    height = max(3.2, 0.55 * len(labels) + 1.0)
    fig, ax = plt.subplots(figsize=(7.5, height))
    ax.barh(y_positions, values, color=colors)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric.label} ({metric.units})")
    ax.set_title(comp.label)

    x_max = max(values) if values else 0.0
    pad = x_max * 0.03 if x_max else 0.05
    for ypos, val in zip(y_positions, values):
        ax.text(val + pad, ypos, f"{val:.2f}", ha="left", va="center", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _build_emission_stack_df(comp: ComparisonSpec, records: Mapping[str, RunRecord]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for run_name in comp.run_names:
        record = records[run_name]
        drained = record.metrics.drained_emissions_gt
        burned = record.metrics.burned_emissions_gt
        total = drained + burned
        unc = record.uncertainty
        rows.append(
            {
                "run_key": run_name,
                "Run": record.spec.label,
                "Drained": drained,
                "Burned": burned,
                "Total": total,
                "Total_low": unc.total_low if unc else None,
                "Total_high": unc.total_high if unc else None,
                "Drained_low": unc.drained_low if unc else None,
                "Drained_high": unc.drained_high if unc else None,
                "Burned_low": unc.burned_low if unc else None,
                "Burned_high": unc.burned_high if unc else None,
            }
        )
    return pd.DataFrame(rows)


def _plot_stacked_total_with_error(df: pd.DataFrame, comp: ComparisonSpec) -> plt.Figure:
    x = list(range(len(df)))
    fig, ax = plt.subplots(figsize=(max(6.5, 1.8 * len(x)), 5.0))

    drained = ax.bar(x, df["Drained"], color=STACK_COMPONENT_COLORS["Drained"], label="Drained")
    burned = ax.bar(
        x,
        df["Burned"],
        bottom=df["Drained"],
        color=STACK_COMPONENT_COLORS["Burned"],
        label="Burned",
    )

    if df[["Total_low", "Total_high"]].notna().any().any():
        totals = df["Total"]
        lower = totals - df["Total_low"].fillna(totals)
        upper = df["Total_high"].fillna(totals) - totals
        ax.errorbar(
            x,
            totals,
            yerr=[lower, upper],
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["Run"], rotation=20, ha="right")
    ax.set_ylabel("Gt CO₂e/year")
    ax.set_title(f"{comp.label}: Total emissions with uncertainty")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.bar_label(drained, fmt="{:.2f}", label_type="center", color="white", fontsize=8)
    ax.bar_label(burned, fmt="{:.2f}", label_type="center", color="white", fontsize=8)

    fig.tight_layout()
    return fig


def _plot_small_multiple_stacks(df: pd.DataFrame, comp: ComparisonSpec) -> plt.Figure:
    metrics = (
        ("Drained", "Drained emissions"),
        ("Burned", "Burned emissions"),
        ("Total", "Total emissions"),
    )
    fig, axes = plt.subplots(1, len(metrics), figsize=(max(9.0, 2.6 * len(df)), 4.8), sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, metrics):
        x = list(range(len(df)))
        if col == "Total":
            ax.bar(x, df["Drained"], color=STACK_COMPONENT_COLORS["Drained"], label="Drained")
            ax.bar(
                x,
                df["Burned"],
                bottom=df["Drained"],
                color=STACK_COMPONENT_COLORS["Burned"],
                label="Burned",
            )
            if df[["Total_low", "Total_high"]].notna().any().any():
                totals = df["Total"]
                lower = totals - df["Total_low"].fillna(totals)
                upper = df["Total_high"].fillna(totals) - totals
                ax.errorbar(x, totals, yerr=[lower, upper], fmt="none", ecolor="black", capsize=4)
        else:
            color = STACK_COMPONENT_COLORS[col]
            ax.bar(x, df[col], color=color)
            low_col = f"{col}_low"
            high_col = f"{col}_high"
            if df[[low_col, high_col]].notna().any().any():
                lower = df[col] - df[low_col].fillna(df[col])
                upper = df[high_col].fillna(df[col]) - df[col]
                ax.errorbar(x, df[col], yerr=[lower, upper], fmt="none", ecolor="black", capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(df["Run"], rotation=20, ha="right")
        ax.set_title(label)
        ax.set_ylabel("Gt CO₂e/year")
        ax.set_ylim(bottom=0)
        if col == "Total":
            ax.legend()

    fig.suptitle(f"{comp.label}: Drained vs. burned uncertainty panels")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _collect_breakouts(records: Mapping[str, RunRecord], attr: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_name in sorted(records):
        df = getattr(records[run_name].breakouts, attr)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# === NEW: Helpers for per-country disagreement across runs (inventory_source) ===

def _first_present_col(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """Return the first column present in df from candidates (case-insensitive)."""
    if df is None or df.empty:
        return None
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _parse_dateish_to_order(series: pd.Series) -> pd.Series:
    """Convert various year/date strings to an integer order key YYYYMMDD0000-ish."""
    def _one(v):
        if pd.isna(v):
            return pd.NA
        s = str(v)
        digits = "".join(ch for ch in s if ch.isdigit())
        if not digits:
            return pd.NA
        if len(digits) >= 8:
            return int(digits[:8])
        if len(digits) >= 4:
            return int(digits[:4]) * 10000
        return pd.NA
    return series.map(_one)


def _latest_period_filter(df: pd.DataFrame, years: Sequence[int]) -> pd.DataFrame:
    """
    Keep only rows for the latest requested inventory period if we can identify a period column.
    Tries year columns, then date-ish columns, then run_date; otherwise returns df unchanged.
    """
    if df is None or df.empty:
        return df

    for c in ["interval_end_year", "period_end_year", "inventory_year", "year"]:
        if c in df.columns:
            latest = max(years)
            return df[df[c] == latest].copy()

    for c in ["interval_end", "period_end", "interval_end_date", "period", "interval_label"]:
        if c in df.columns:
            order = _parse_dateish_to_order(df[c])
            if order.notna().any():
                max_order = order.max()
                return df[order == max_order].copy()

    if "run_date" in df.columns:
        order = _parse_dateish_to_order(df["run_date"])
        if order.notna().any():
            max_order = order.max()
            return df[order == max_order].copy()

    return df


def _get_first_value(row: pd.Series, candidates: Sequence[str]) -> float | None:
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            try:
                return float(row[col])
            except (TypeError, ValueError):
                continue
    return None


def _component_stats(row: pd.Series) -> tuple[float | None, float | None, float | None]:
    mean = _get_first_value(row, [
        "mean_gtco2e_per_yr",
        "avg_gtco2e_per_yr",
        "mean",
        "average",
        "median",
        "value",
    ])
    vmin = _get_first_value(row, ["min", "lower", "low", "p05", "p5"])
    vmax = _get_first_value(row, ["max", "upper", "high", "p95", "p90"])
    return mean, vmin, vmax


def _load_chunk_stats_bounds(path: str, years: Sequence[int]) -> tuple[dict[str, float], MetricUncertainty] | None:
    """
    Load per-component mean/min/max emissions (Gt CO2e/yr) from a chunk_stats CSV/Parquet.
    Returns (metrics_override, uncertainty) keyed by component name.
    """

    try:
        if path.lower().endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - file handling
        print(f"[chunk_stats] Failed to read {path}: {exc}")
        return None

    if df is None or df.empty:
        return None

    df = _latest_period_filter(df, years)

    comp_col = _first_present_col(df, ["component", "metric", "flux_component", "name"])
    if not comp_col:
        print(f"[chunk_stats] No component column detected in {path}; columns={list(df.columns)}")
        return None

    stats: dict[str, float] = {}
    bounds = MetricUncertainty()

    comp_series = df[comp_col].astype(str).str.lower() if comp_col in df.columns else pd.Series(dtype=str)
    for comp_name, label in (("drained", "Drained"), ("burned", "Burned"), ("total", "Total")):
        subset = df[comp_series.str.contains(comp_name)] if comp_col in df.columns else pd.DataFrame()
        if subset.empty:
            continue
        row = subset.iloc[0]
        mean, vmin, vmax = _component_stats(row)
        key_mean = f"{label.lower()}_emissions_gt" if label != "Total" else "total_emissions_gt"
        if mean is not None:
            stats[key_mean] = mean
        if label == "Drained":
            bounds = MetricUncertainty(
                drained_low=vmin,
                drained_high=vmax,
                burned_low=bounds.burned_low,
                burned_high=bounds.burned_high,
                total_low=bounds.total_low,
                total_high=bounds.total_high,
            )
        elif label == "Burned":
            bounds = MetricUncertainty(
                drained_low=bounds.drained_low,
                drained_high=bounds.drained_high,
                burned_low=vmin,
                burned_high=vmax,
                total_low=bounds.total_low,
                total_high=bounds.total_high,
            )
        else:
            bounds = MetricUncertainty(
                drained_low=bounds.drained_low,
                drained_high=bounds.drained_high,
                burned_low=bounds.burned_low,
                burned_high=bounds.burned_high,
                total_low=vmin,
                total_high=vmax,
            )

    # Derive totals if only components are present
    if "total_emissions_gt" not in stats and {"drained_emissions_gt", "burned_emissions_gt"} <= set(stats):
        stats["total_emissions_gt"] = stats["drained_emissions_gt"] + stats["burned_emissions_gt"]

    if bounds.total_low is None and bounds.drained_low is not None and bounds.burned_low is not None:
        bounds = MetricUncertainty(
            drained_low=bounds.drained_low,
            drained_high=bounds.drained_high,
            burned_low=bounds.burned_low,
            burned_high=bounds.burned_high,
            total_low=bounds.drained_low + bounds.burned_low,
            total_high=bounds.total_high,
        )
    if bounds.total_high is None and bounds.drained_high is not None and bounds.burned_high is not None:
        bounds = MetricUncertainty(
            drained_low=bounds.drained_low,
            drained_high=bounds.drained_high,
            burned_low=bounds.burned_low,
            burned_high=bounds.burned_high,
            total_low=bounds.total_low,
            total_high=bounds.drained_high + bounds.burned_high,
        )

    return stats, bounds


def _normalize_country_keys(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Standardize to columns 'iso3' (if available) and 'country' for grouping.
    Returns the normalized frame and the list of key columns present.
    """
    if df is None or df.empty:
        return df, []

    iso3_col = _first_present_col(df, ["iso3", "adm0_iso3", "country_iso3", "ISO3", "iso_a3"])
    name_col = _first_present_col(df, ["country", "adm0_name", "country_name", "name"])

    keys: list[str] = []
    out = df.copy()
    if iso3_col:
        out["iso3"] = out[iso3_col]
        keys.append("iso3")
    if name_col:
        out["country"] = out[name_col]
        keys.append("country")
    if not keys:
        for fallback in df.columns:
            if isinstance(fallback, str) and fallback.lower().endswith("name"):
                out["country"] = out[fallback]
                keys.append("country")
                break
    return out, keys


def _select_area_column(df: pd.DataFrame) -> tuple[str | None, float]:
    """
    Find a column that represents area and return (column_name, to_mha_factor).
    Priority: *_mha  -> factor 1
              *_ha   -> factor 1e-6
              *_km2  -> factor 1/10000
    """
    col = _first_present_col(df, ["peat_area_mha", "total_peat_area_mha", "area_mha", "drained_area_mha", "undrained_area_mha"])
    if col:
        return col, 1.0
    col = _first_present_col(df, ["peat_area_ha", "total_peat_area_ha", "area_ha", "ha", "drained_area_ha", "undrained_area_ha"])
    if col:
        return col, 1e-6
    col = _first_present_col(df, ["area_km2", "km2", "area_km²", "km²", "drained_area_km2", "undrained_area_km2"])
    if col:
        return col, 1.0 / 10000.0
    return None, 1.0


def _normalize_drained_state_values(series: pd.Series) -> pd.Series:
    """Map a variety of encodings to {'drained','undrained','other'}."""
    def _one(v):
        if pd.isna(v):
            return "other"
        if isinstance(v, (bool,)):
            return "drained" if v else "undrained"
        s = str(v).strip().lower()
        if s in {"drained", "d", "true", "t", "1", "yes", "y"}:
            return "drained"
        if s in {"undrained", "u", "false", "f", "0", "no", "n"}:
            return "undrained"
        return "other"
    return series.map(_one)


def _extract_country_peat_area_from_by_country(bc: pd.DataFrame, years: Sequence[int]) -> pd.DataFrame:
    """
    Try to read total peat area directly from by-country table.
    Returns standardized columns: ['iso3?','country?','run_name','Run','value_mha']
    """
    if bc is None or bc.empty:
        return pd.DataFrame()

    bc = _latest_period_filter(bc, years)
    bc, keys = _normalize_country_keys(bc)
    if not keys:
        return pd.DataFrame()

    d_col = _first_present_col(bc, ["drained_area_mha", "drained_area_ha", "drained_area_km2"])
    u_col = _first_present_col(bc, ["undrained_area_mha", "undrained_area_ha", "undrained_area_km2"])
    if d_col and u_col:
        _, d_factor = _select_area_column(pd.DataFrame({d_col: bc[d_col]}))
        _, u_factor = _select_area_column(pd.DataFrame({u_col: bc[u_col]}))
        tmp = bc.copy()
        tmp["__dr_mha"] = tmp[d_col] * d_factor
        tmp["__un_mha"] = tmp[u_col] * u_factor
        grouped = (
            tmp.groupby(keys + ["run_name", "Run"], observed=True)[["__dr_mha", "__un_mha"]]
            .sum()
            .reset_index()
        )
        grouped["value_mha"] = grouped["__dr_mha"] + grouped["__un_mha"]
        return grouped[keys + ["run_name", "Run", "value_mha"]]

    area_col, factor = _select_area_column(bc)
    if not area_col:
        return pd.DataFrame()

    state_col = _first_present_col(bc, ["drained_state", "peat_state", "is_drained", "drained", "state"])
    tmp = bc.copy()
    tmp["__area_mha"] = tmp[area_col] * factor

    if state_col and state_col in tmp.columns:
        tmp["__state"] = _normalize_drained_state_values(tmp[state_col])
        sub = tmp[tmp["__state"].isin(["drained", "undrained"])].copy()
        if not sub.empty:
            grouped = (
                sub.groupby(keys + ["run_name", "Run"], observed=True)["__area_mha"]
                .sum()
                .reset_index()
                .rename(columns={"__area_mha": "value_mha"})
            )
            return grouped

    grouped = (
        tmp.groupby(keys + ["run_name", "Run"], observed=True)["__area_mha"]
        .sum()
        .reset_index()
        .rename(columns={"__area_mha": "value_mha"})
    )
    return grouped


def _extract_country_peat_area_from_drained_state(bcd: pd.DataFrame, years: Sequence[int]) -> pd.DataFrame:
    """
    Fallback: use by-country-by-drained-state; sum drained+undrained.
    Returns standardized columns: ['iso3?','country?','run_name','Run','value_mha']
    """
    if bcd is None or bcd.empty:
        return pd.DataFrame()

    bcd = _latest_period_filter(bcd, years)
    bcd, keys = _normalize_country_keys(bcd)
    if not keys:
        return pd.DataFrame()

    area_col, factor = _select_area_column(bcd)
    if not area_col:
        return pd.DataFrame()

    state_col = _first_present_col(bcd, ["drained_state", "peat_state", "is_drained", "drained", "state"])
    if not state_col:
        return pd.DataFrame()

    tmp = bcd.copy()
    tmp["__area_mha"] = tmp[area_col] * factor
    tmp["__state"] = _normalize_drained_state_values(tmp[state_col])

    sub = tmp[tmp["__state"].isin(["drained", "undrained"])].copy()
    if sub.empty:
        sub = tmp

    grouped = (
        sub.groupby(keys + ["run_name", "Run"], observed=True)["__area_mha"]
        .sum()
        .reset_index()
        .rename(columns={"__area_mha": "value_mha"})
    )
    return grouped


def _build_country_area_base(records: Mapping[str, RunRecord], years: Sequence[int]) -> pd.DataFrame:
    """
    Build a normalized per-country, per-run table with peat area in Mha.
    Tries multiple breakouts and unit encodings. Returns empty DataFrame if nothing matches.
    """
    bc = _collect_breakouts(records, "by_country")
    base = _extract_country_peat_area_from_by_country(bc, years)
    if base is not None and not base.empty:
        return base

    bcd = _collect_breakouts(records, "by_country_drained_state")
    base = _extract_country_peat_area_from_drained_state(bcd, years)
    if base is not None and not base.empty:
        return base

    bcb = _collect_breakouts(records, "by_country_burned_state")
    base = _extract_country_peat_area_from_drained_state(bcb, years)
    if base is not None and not base.empty:
        return base

    return pd.DataFrame()


def _compute_country_disagreement(
    base: pd.DataFrame,
    run_label_order: Sequence[str],
    abs_flag_mha: float,
    rel_flag_fold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Given a normalized long table with columns: ['iso3?','country?','run_name','Run','value_mha'],
    pivot to wide, compute disagreement metrics, and return (wide, summary, flagged).
    """
    keys = [k for k in ["iso3", "country"] if k in base.columns]
    if not keys:
        keys = ["country"]

    base = base.copy()
    base["Run"] = base["Run"].astype(str)
    col_order = [lbl for lbl in run_label_order if lbl in set(base["Run"])]

    wide = base.pivot_table(
        index=keys,
        columns="Run",
        values="value_mha",
        aggfunc="sum",
        observed=True,
    ).reset_index()

    for col in col_order:
        if col not in wide.columns:
            wide[col] = pd.NA

    run_cols = [c for c in wide.columns if c not in keys]

    def _row_metrics(row: pd.Series) -> pd.Series:
        vals = row[run_cols].dropna().astype(float)
        n = int(vals.count())
        if n < 2:
            return pd.Series({
                "n_datasets": n, "min_mha": pd.NA, "median_mha": pd.NA, "max_mha": pd.NA,
                "abs_spread_mha": pd.NA, "fold_change": pd.NA, "log10_spread": pd.NA,
                "min_run": pd.NA, "max_run": pd.NA, "flag": ""
            })
        vmin = float(vals.min())
        vmax = float(vals.max())
        abs_spread = vmax - vmin
        pos = vals[vals > 0.0]
        if not pos.empty:
            min_pos = float(pos.min())
            fold = (vmax / min_pos) if min_pos > 0.0 else math.inf
            log10s = (math.log10(fold) if math.isfinite(fold) and fold > 0.0 else pd.NA)
        else:
            fold = math.inf if vmax > 0.0 else pd.NA
            log10s = pd.NA

        min_run = vals.idxmin()
        max_run = vals.idxmax()

        rel_ok = (isinstance(log10s, float)) and (log10s >= math.log10(rel_flag_fold))
        abs_ok = abs_spread >= abs_flag_mha
        flag = "FLAG" if (rel_ok or abs_ok) else ""

        return pd.Series({
            "n_datasets": n,
            "min_mha": vmin,
            "median_mha": float(vals.median()),
            "max_mha": vmax,
            "abs_spread_mha": abs_spread,
            "fold_change": fold,
            "log10_spread": log10s,
            "min_run": min_run,
            "max_run": max_run,
            "flag": flag,
        })

    summary = wide.copy()
    metrics = summary.apply(_row_metrics, axis=1)
    summary = pd.concat([summary[keys], metrics], axis=1)

    flagged = summary[(summary["n_datasets"] >= 2) & (summary["flag"] == "FLAG")].copy()

    wide = wide[keys + col_order]
    summary_cols = keys + [
        "n_datasets", "min_mha", "median_mha", "max_mha",
        "abs_spread_mha", "fold_change", "log10_spread", "min_run", "max_run", "flag"
    ]
    summary = summary[summary_cols]

    return wide, summary, flagged


def _export_inventory_source_area_disagreement(
    comp: ComparisonSpec,
    records: Mapping[str, RunRecord],
    years: Sequence[int],
    abs_flag_mha: float,
    rel_flag_fold: float,
    writer_con: duckdb.DuckDBPyConnection,
    out_dir: str,
) -> None:
    """
    Compute and write CSVs for the inventory_source comparison:
    - per-country wide table of peat area (Mha) per run (label),
    - per-country summary disagreement metrics,
    - flagged subset.
    """
    sub_records = {rn: records[rn] for rn in comp.run_names if rn in records}
    if len(sub_records) < 2:
        print("Skipping country disagreement export: fewer than two matching runs were provided.")
        return

    base = _build_country_area_base(sub_records, years)
    if base is None or base.empty:
        print(
            "[inventory_source] No country-level peat area columns detected in provided breakouts. "
            "Searched for: area_mha / area_ha / area_km2 or drained/undrained area columns; "
            "and by_country/by_country_drained_state/by_country_burned_state tables."
        )
        return

    labels_in_order = [sub_records[rn].spec.label for rn in comp.run_names]
    base = base[base["run_name"].isin(sub_records.keys())].copy()

    wide, summary, flagged = _compute_country_disagreement(
        base=base,
        run_label_order=labels_in_order,
        abs_flag_mha=abs_flag_mha,
        rel_flag_fold=rel_flag_fold,
    )

    _write_csv_df(writer_con, wide,   _join(out_dir, f"{comp.key}_country_peat_area_wide.csv"))
    _write_csv_df(writer_con, summary,_join(out_dir, f"{comp.key}_country_peat_area_summary.csv"))
    _write_csv_df(writer_con, flagged,_join(out_dir, f"{comp.key}_country_peat_area_flagged.csv"))
    print(
        f"[inventory_source] wrote: "
        f"{comp.key}_country_peat_area_wide.csv, "
        f"{comp.key}_country_peat_area_summary.csv, "
        f"{comp.key}_country_peat_area_flagged.csv"
    )


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser("Build comparison figures across multiple runs")
    parser.add_argument("--years", nargs="+", required=True, help="Inventory period end years (e.g., 2005 2010 2015)")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Run specification: run_name=model_version:run_date or run_name=model_version:run_date|Label. "
            "Provide once per run."
        ),
    )
    parser.add_argument("--aws_region", default=None, help="Optional AWS region for S3 access")
    parser.add_argument("--data-only", action="store_true", help="Export CSV data only (skip figures)")
    # NEW thresholds for disagreement flags
    parser.add_argument(
        "--flag-abs-mha",
        type=float,
        default=0.1,
        help="Absolute spread flag threshold in million hectares (default 0.1 Mha = 100,000 ha).",
    )
    parser.add_argument(
        "--flag-rel-fold",
        type=float,
        default=10.0,
        help="Relative spread flag threshold as fold-change (default 10x).",
    )
    parser.add_argument(
        "--chunk-stats",
        action="append",
        help=(
            "Optional path to chunk_stats summary per run (run_name=path). "
            "When provided, drained/burned totals and error bars will use these values."
        ),
    )
    args = parser.parse_args(argv)

    try:
        years = [int(y) for y in args.years]
    except ValueError as exc:  # pragma: no cover - CLI validation
        raise SystemExit(f"Invalid --years value: {exc}")

    if not years:
        raise SystemExit("At least one inventory year must be provided")

    run_specs = _parse_run_specs(args.run)
    chunk_stat_paths = _parse_chunk_stat_paths(args.chunk_stats)
    out_dir = _comparison_out_dir(run_specs)
    active_comparisons, skipped_comparisons = _partition_comparisons(run_specs)
    if skipped_comparisons:
        for key, missing_runs in sorted(skipped_comparisons.items()):
            missing_list = ", ".join(missing_runs)
            print(
                f"Skipping comparison '{key}' because the following runs were not provided: {missing_list}"
            )

    if not active_comparisons:
        raise SystemExit(
            "No comparisons can be generated because none of the required run combinations were provided."
        )

    color_map = _assign_colors(run_specs.keys())

    records: dict[str, RunRecord] = {}
    for run_name, spec in run_specs.items():
        metrics, breakouts, uncertainty = _compute_run_data(
            spec, years, args.aws_region, chunk_stat_paths.get(run_name)
        )
        records[run_name] = RunRecord(
            spec=spec,
            metrics=metrics,
            breakouts=breakouts,
            color=color_map[run_name],
            uncertainty=uncertainty,
        )

    out_data_dir = _join(out_dir, "figures", "comparisons", "data")
    writer_con = duckdb.connect()
    try:
        for comp in active_comparisons:
            summary_df = _build_comparison_summary(comp, records)
            summary_path = _join(out_data_dir, f"{comp.key}_summary.csv")
            _write_csv_df(writer_con, summary_df, summary_path)

            for metric_key in comp.metric_keys:
                metric = METRIC_SPECS[metric_key]
                metric_df = _build_metric_long_df(comp, metric, records)
                metric_path = _join(out_data_dir, f"{comp.key}_{metric.key}.csv")
                _write_csv_df(writer_con, metric_df, metric_path)

                if args.data_only:
                    continue

                colors = [records[rn].color for rn in comp.run_names]
                fig = _plot_metric(metric_df, metric, comp, colors)
                fig_path = _join(out_dir, "figures", "comparisons", f"{comp.key}_{metric.key}.png")
                _save_png(fig, fig_path, dpi=300)
                plt.close(fig)

            if comp.key == "ogh_sensitivity_range":
                stack_df = _build_emission_stack_df(comp, records)
                stack_path = _join(out_data_dir, f"{comp.key}_drained_burned_stack.csv")
                _write_csv_df(writer_con, stack_df, stack_path)

                if args.data_only:
                    continue

                stack_fig = _plot_stacked_total_with_error(stack_df, comp)
                stack_fig_path = _join(out_dir, "figures", "comparisons", f"{comp.key}_total_stack_error.png")
                _save_png(stack_fig, stack_fig_path, dpi=300)
                plt.close(stack_fig)

                small_fig = _plot_small_multiple_stacks(stack_df, comp)
                small_fig_path = _join(out_dir, "figures", "comparisons", f"{comp.key}_small_multiples.png")
                _save_png(small_fig, small_fig_path, dpi=300)
                plt.close(small_fig)

        breakout_specs = (
            ("by_country_period", "by_country"),
            ("by_drained_state_period", "by_drained_state"),
            ("by_burned_state_period", "by_burned_state"),
            ("by_country_drained_state_period", "by_country_drained_state"),
            ("by_country_burned_state_period", "by_country_burned_state"),
        )
        for file_stub, attr in breakout_specs:
            breakout_df = _collect_breakouts(records, attr)
            breakout_path = _join(out_data_dir, f"runs_{file_stub}.csv")
            _write_csv_df(writer_con, breakout_df, breakout_path)

        # NEW: per-country disagreement for inventory-source peat area (Mha)
        for comp in active_comparisons:
            if comp.key == "inventory_source":
                _export_inventory_source_area_disagreement(
                    comp=comp,
                    records=records,
                    years=years,
                    abs_flag_mha=float(args.flag_abs_mha),
                    rel_flag_fold=float(args.flag_rel_fold),
                    writer_con=writer_con,
                    out_dir=out_data_dir,
                )
    finally:
        writer_con.close()

    print("Comparison assets written to:", out_dir)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()