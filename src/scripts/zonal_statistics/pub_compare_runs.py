# -*- coding: utf-8 -*-
"""Build comparison figures across multiple zonal-statistics runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

import src.scripts.zonal_statistics.pub_common as pc
import src.scripts.zonal_statistics.pub_assets as pa


OUT_DIR = pa.OUT_DIR

_join = pa._join
_save_png = pa._save_png
_write_csv_df = pa._write_csv_df
_register_components = pa._register_components
_register_state_context_views = pa._register_state_context_views
_interval_folder_strings = pa._interval_folder_strings
_make_base_prefixes = pa._make_base_prefixes
_make_globs_for_components = pa._make_globs_for_components


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


@dataclass(frozen=True)
class RunRecord:
    spec: RunSpec
    metrics: RunMetrics
    color: str


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
        run_names=("ogh_sensitivity_1km", "gfw_standard_1km", "gdp_standard_1km"),
        metric_keys=("peat_total_area", "peat_drained_area", "drained_emissions", "burned_emissions"),
    ),
    ComparisonSpec(
        key="ogh_sensitivity_range",
        label="OGH Sensitivity (High/Low Emissions)",
        run_names=("ogh_sensitivity_1km", "ogh_sensitivity_high", "ogh_sensitivity_low"),
        metric_keys=("drained_emissions", "burned_emissions", "total_emissions"),
    ),
)


REQUIRED_RUNS = {run for comp in COMPARISONS for run in comp.run_names}


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


def _assign_colors(run_names: Iterable[str]) -> Mapping[str, str]:
    cmap = plt.get_cmap("tab10")
    ordered = sorted(dict.fromkeys(run_names))
    return {run: mcolors.to_hex(cmap(i % cmap.N)) for i, run in enumerate(ordered)}


def _compute_run_metrics(spec: RunSpec, years: Sequence[int], aws_region: str | None) -> RunMetrics:
    interval_folders = _interval_folder_strings(years)
    base_prefixes = _make_base_prefixes(spec.model_version, spec.run_name, spec.run_date, interval_folders)
    drained_globs, burned_globs = _make_globs_for_components(base_prefixes)

    con = duckdb.connect()
    try:
        _register_components(con, drained_globs, burned_globs, aws_region=aws_region)
        _register_state_context_views(con)

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
    finally:
        con.close()

    return RunMetrics(
        drained_area_mha=drained_area,
        undrained_area_mha=undrained_area,
        drained_emissions_gt=drained_emissions,
        burned_emissions_gt=burned_emissions,
    )


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
    args = parser.parse_args(argv)

    try:
        years = [int(y) for y in args.years]
    except ValueError as exc:  # pragma: no cover - CLI validation
        raise SystemExit(f"Invalid --years value: {exc}")

    if not years:
        raise SystemExit("At least one inventory year must be provided")

    run_specs = _parse_run_specs(args.run)
    missing_runs = sorted(REQUIRED_RUNS.difference(run_specs.keys()))
    if missing_runs:
        raise SystemExit(
            "Missing --run specification(s) for required runs: " + ", ".join(missing_runs)
        )

    color_map = _assign_colors(run_specs.keys())

    records: dict[str, RunRecord] = {}
    for run_name, spec in run_specs.items():
        metrics = _compute_run_metrics(spec, years, args.aws_region)
        records[run_name] = RunRecord(spec=spec, metrics=metrics, color=color_map[run_name])

    out_data_dir = _join(OUT_DIR, "figures", "comparisons", "data")
    writer_con = duckdb.connect()
    try:
        for comp in COMPARISONS:
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
                fig_path = _join(OUT_DIR, "figures", "comparisons", f"{comp.key}_{metric.key}.png")
                _save_png(fig, fig_path, dpi=300)
                plt.close(fig)
    finally:
        writer_con.close()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()