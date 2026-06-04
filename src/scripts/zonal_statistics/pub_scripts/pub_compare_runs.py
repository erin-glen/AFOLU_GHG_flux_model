
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
  separator; e.g. ``--run "gfw_standard_model_500m=0_9_7:20251006|GFW 500 m"``.
* ``--aws_region`` – optional AWS region for S3 access (mirrors
  ``pub_assets`` default behaviour).
* ``--data-only`` – skip figure generation and export CSV data only.
* ``--flag-abs-mha`` – absolute spread threshold (Mha) for FLAGging countries
  in the disagreement summary (default 0.1 = 100,000 ha).
* ``--flag-rel-fold`` – relative spread threshold (fold-change) for FLAGging
  countries in the disagreement summary (default 10x).
* ``--chunk-stats`` – *optional* per-run overrides for chunk_stats location;
  see notes below.

Each comparison defined below expects a specific set of run names. When a
comparison's requirements are not fully met, the script will skip that
comparison and emit a note summarizing the missing runs. Provide additional
``--run`` entries if you need to compare multiple model versions or reruns of
the same scenario.

Outputs are grouped under ``/mnt/c/tmp/pub_assets/comparisons/<run_names>/``
where ``<run_names>`` is all run names joined with ``__``.

Usage example (inventory + OGH sensitivity combined):

  cd /mnt/c/gis/git/AFOLU_GHG_flux_model

  python -m src.scripts.zonal_statistics.pub_scripts.pub_compare_runs \
    --years 2024 \
    --run "ogh_mixed_f1_f15_f2_20260513=1_0_1:20260525|OGH baseline" \
    --run "gfw_standard_model_500m=1_0_1:20260525|GFW" \
    --run "gpd_standard_model_500m=1_0_1:20260525|GPD" \
    --run "ogh_sensitivity_250m=1_0_1:20260525|250 m" \
    --run "ogh_sensitivity_750m=1_0_1:20260525|750 m" \
    --run "ogh_sensitivity_low=1_0_1:20260525|Low envelope" \
    --run "ogh_sensitivity_high=1_0_1:20260525|High envelope" \
    --run "ogh_sensitivity_area_low=1_0_1:20260525|Area low" \
    --run "ogh_sensitivity_area_high=1_0_1:20260525|Area high" \
    --run "ogh_sensitivity_ef_low=1_0_1:20260525|EF low" \
    --run "ogh_sensitivity_ef_high=1_0_1:20260525|EF high"


  python -m src.scripts.zonal_statistics.pub_scripts.pub_compare_runs \
    --years 2024 \
    --run "ogh_mixed_f1_f15_f2_20260513=1_0_1:20260525|OGH" \
    --run "gfw_standard_model_500m=1_0_1:20260525|GFW" \
    --run "gpd_standard_model_500m=1_0_1:20260525|GPD"

  python -m src.scripts.zonal_statistics.pub_scripts.pub_compare_runs \
    --years 2024 \
    --run "ogh_sensitivity_250m=1_0_1:20260525|250 m" \
    --run "ogh_mixed_f1_f15_f2_20260513=1_0_1:20260525|500 m baseline" \
    --run "ogh_sensitivity_750m=1_0_1:20260525|750 m"

  python -m src.scripts.zonal_statistics.pub_scripts.pub_compare_runs \
    --years 2024 \
    --run "ogh_sensitivity_low=1_0_1:20260525|Low envelope" \
    --run "ogh_mixed_f1_f15_f2_20260513=1_0_1:20260525|Baseline" \
    --run "ogh_sensitivity_high=1_0_1:20260525|High envelope"


OGH sensitivity comparisons (distance and high/low emissions) are designed to
prefer chunk-statistics inputs when available. This script will *auto-discover*
the 1×1 chunk_stats Excel file for the OGH sensitivity runs based on
``run_name``, ``model_version`` and ``run_date``, assuming the standard AFOLU
output layout on S3:

  s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/
    version_<model_version>/chunk_stats/<dir_name>/<run_date>/*.xlsx

where ``dir_name`` is:

  - ``ogh_sensitivity_250m_10``
  - ``ogh_sensitivity_750m_10``
  - ``ogh_sensitivity_low_10``
  - ``ogh_sensitivity_high_10``

If no such Excel file is found for a sensitivity run, the script falls back to
using zonal statistics only for that run.

The **Inventory Input Source Comparison** uses the following runs and, by
default, reads *only* zonal-statistics outputs:

  - ``ogh_mixed_f1_f15_f2_20260513``
  - ``gfw_standard_model_500m``
  - ``gpd_standard_model_500m``

Chunk stats for these runs are used *only* if you explicitly pass
``--chunk-stats run_name=...`` for them.

If your chunk_stats live somewhere else (e.g. a different bucket or local
path), you can either:

* override the root via ``AFOLU_CHUNK_STATS_ROOT``, or
* pass explicit paths via ``--chunk-stats run_name=path``. The path may be a
  specific .xlsx file or a directory/prefix containing a single .xlsx.

Chunk stats are expected to be Excel outputs containing an ``other_outputs_1x1``
sheet with ``layer_name`` values
``drained_total_Mg_CO2e_ha_yr`` and ``burned_total_Mg_CO2e_ha_yr``; the script
automatically extracts the ``sum_value`` totals (converted from Mg to Gt).

Example OGH high/low comparison:

  python -m src.scripts.zonal_statistics.pub_scripts.pub_compare_runs \
    --years 2024 \
    --run "ogh_sensitivity_low=1_0_1:20260525|Low envelope" \
    --run "ogh_mixed_f1_f15_f2_20260513=1_0_1:20260525|OGH baseline" \
    --run "ogh_sensitivity_high=1_0_1:20260525|High envelope"

For distance and high/low comparisons, this script produces a *single* stacked
bar chart each, showing total emissions (Gt CO₂e/year) split into drained and
burned components. The mid-point (500 m baseline) is always plotted in the
middle, with the lower sensitivity on the left and higher on the right.

For the **Inventory Input Source Comparison**, this script also exports
per-country **disagreement** tables for total peat area (Mha) across the source
runs (OGH / GFW / GPD), including min/median/max, absolute spread, fold-change,
log10 spread, and the min/max contributing dataset, plus a flagged subset based
on user-tunable thresholds.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence
import math
import os
from pathlib import Path
import glob

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

import src.scripts.zonal_statistics.pub_scripts.pub_common as pc
import src.scripts.zonal_statistics.pub_scripts.pub_assets as pa
from src.scripts.utilities import local_output_paths as lop
from src.scripts.zonal_statistics.run_zonal_stats import build_interval_pairs


def _default_comparison_root() -> str:
    legacy_assets_root = os.environ.get("AFOLU_PUB_ASSETS_DIR")
    if legacy_assets_root:
        return pc._join(legacy_assets_root, "comparisons")
    return lop.publication_root("comparisons")


OUT_DIR_ROOT = os.environ.get("AFOLU_PUB_COMPARE_DIR", _default_comparison_root())

_join = pc._join
_save_png = pc._save_png
_write_csv_df = pc._write_csv_df
_register_components = pa._register_components
_register_state_context_views = pa._register_state_context_views
_ensure_adm0_lookup = pa._ensure_adm0_lookup
_interval_folder_strings = pa._interval_folder_strings
_make_base_prefixes = pa._make_base_prefixes
_make_globs_for_components = pa._make_globs_for_components

# Match the drained/burned palette used throughout pub_assets/pub_common
STACK_COMPONENT_COLORS = pc.PROCESS_COLORS
STACK_COMPONENT_ORDER = ("Drained", "Burned")

PEAT_AREA_COLORS = {
    "Drained": STACK_COMPONENT_COLORS.get("Drained", "#4c78a8"),
    "Undrained": "#9ca3af",
}

# Default categorical palette for run-level comparisons. Update this constant to
# swap palettes without plumbing a CLI flag (palette names mirror pc.PALETTES).
RUN_COLOR_PALETTE = "tol_bright"

# Root location for chunk_stats Excel summaries.
# IMPORTANT: default to the S3 outputs root, not the local pub_assets root.
CHUNK_STATS_ROOT = os.environ.get(
    "AFOLU_CHUNK_STATS_ROOT",
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs",
)

# Runs for which chunk_stats paths are auto-discovered (sensitivities that may
# only have a chunk_stats 4000 px summary, not zonal stats). This list does NOT
# force chunk usage: the per-comparison resolution is availability-driven
# (_resolve_comparison_source in main) -- zonal is used whenever every run in a
# comparison has zonal stats, and chunk is only the fallback. Listing a run here
# just pre-resolves its chunk_stats path so it can serve as that fallback.
SENSITIVITY_CHUNK_RUNS = {
    "ogh_sensitivity_250m",
    "ogh_sensitivity_750m",
    "ogh_sensitivity_low",
    "ogh_sensitivity_high",
}


RunSpec = pc.RunSpec


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
    by_climate_component: pd.DataFrame
    by_period_drained_climate: pd.DataFrame
    by_period_burned_climate: pd.DataFrame
    by_period_total_climate: pd.DataFrame


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
    """Build comparison output directory segmented by run names only."""
    run_names = sorted(run_specs.keys())
    name_slug = "__".join(run_names) if run_names else "unspecified_runs"
    return _join(OUT_DIR_ROOT, name_slug)


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
        key="ogh_distance",
        label="OGH Sensitivity (Distance Threshold (meters))",
        run_names=(
            "ogh_sensitivity_250m",
            "ogh_mixed_f1_f15_f2_20260513",
            "ogh_sensitivity_750m",
        ),
        # summary CSV has drained area + drained + total; figure uses total stack
        metric_keys=("peat_drained_area", "drained_emissions", "total_emissions"),
    ),
    ComparisonSpec(
        key="inventory_source",
        label="Inventory Input Source Comparison",
        run_names=(
            "ogh_mixed_f1_f15_f2_20260513",
            "gfw_standard_model_500m",
            "gpd_standard_model_500m",
        ),
        metric_keys=("peat_total_area", "peat_drained_area", "drained_emissions", "burned_emissions"),
    ),
    ComparisonSpec(
        key="ogh_sensitivity_range",
        label="OGH Sensitivity (High/Low Emissions)",
        # Low – Baseline – High so baseline is always in the middle
        run_names=("ogh_sensitivity_low", "ogh_mixed_f1_f15_f2_20260513", "ogh_sensitivity_high"),
        metric_keys=("drained_emissions", "burned_emissions", "total_emissions"),
    ),
    ComparisonSpec(
        key="ogh_area_effect",
        label="OGH Sensitivity (Mapped Area Threshold Only)",
        run_names=(
            "ogh_sensitivity_area_low",
            "ogh_mixed_f1_f15_f2_20260513",
            "ogh_sensitivity_area_high",
        ),
        metric_keys=("peat_drained_area", "drained_emissions", "burned_emissions", "total_emissions"),
    ),
    ComparisonSpec(
        key="ogh_ef_effect",
        label="OGH Sensitivity (Emission Factor Only)",
        run_names=(
            "ogh_sensitivity_ef_low",
            "ogh_mixed_f1_f15_f2_20260513",
            "ogh_sensitivity_ef_high",
        ),
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


def _parse_run_specs(entries: Sequence[str]) -> Mapping[str, RunSpec]:
    return {s.run_name: s for s in pc.parse_run_specs(entries, spec_name="--run")}


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


def _parse_category_labels(entries: Sequence[str] | None) -> Mapping[str, str]:
    labels: dict[str, str] = {}
    if not entries:
        return labels

    for entry in entries:
        if "=" not in entry:
            raise ValueError("Invalid label override (expected run_name=Label)")
        run_name, label = entry.split("=", 1)
        run_name = run_name.strip()
        label = label.strip()
        if not run_name or not label:
            raise ValueError("Invalid label override (expected run_name=Label)")
        labels[run_name] = label

    return labels


def _glob_s3_xlsx(prefix: str) -> list[str]:
    """Best-effort S3 glob for *.xlsx under the given prefix directory."""
    try:
        import fsspec  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - environment-specific
        print(f"[chunk_stats] S3 support not available (fsspec import failed): {exc}")
        return []

    fs = fsspec.filesystem("s3")
    pattern = prefix.rstrip("/") + "/*.xlsx"
    try:
        matches = fs.glob(pattern)
    except Exception as exc:  # pragma: no cover - environment-specific
        print(f"[chunk_stats] Failed to glob S3 pattern {pattern!r}: {exc}")
        return []

    results: list[str] = []
    for p in matches:
        if isinstance(p, str) and p.startswith("s3://"):
            results.append(p)
        else:
            results.append("s3://" + str(p))
    return results


def _resolve_chunk_stats_path_for_run(spec: RunSpec, raw_path: str) -> str:
    """
    Resolve a chunk_stats location to a concrete file.

    Accepts:
      * a direct file path (local or s3://...*.xlsx);
      * a directory/prefix (local or s3://.../dir[/run_date]);
      * a prefix without extension.
    """
    # S3: treat raw_path as either file or prefix
    if raw_path.startswith("s3://"):
        base = raw_path.rstrip("/")
        if base.lower().endswith(".xlsx"):
            return base

        candidates = _glob_s3_xlsx(base)
        if not candidates and spec.run_date:
            candidates = _glob_s3_xlsx(f"{base}/{spec.run_date}")

        if not candidates:
            raise FileNotFoundError(
                f"Could not find any .xlsx chunk_stats file for run '{spec.run_name}' under S3 prefix '{raw_path}'."
            )
        return sorted(set(candidates))[-1]

    # Local filesystem
    base_path = Path(raw_path)

    if base_path.is_file():
        return str(base_path)

    candidates: list[str] = []
    search_roots: list[Path] = []

    if base_path.is_dir():
        search_roots.append(base_path)
        if spec.run_date:
            child = base_path / spec.run_date
            if child.is_dir():
                search_roots.append(child)
    else:
        if spec.run_date:
            child = base_path / spec.run_date
            if child.is_dir():
                search_roots.append(child)

    for root in search_roots:
        candidates.extend(glob.glob(str(root / "*.xlsx")))

    if not candidates:
        for ext in (".xlsx", ".parquet", ".csv"):
            candidate = base_path.with_suffix(ext)
            if candidate.is_file():
                candidates.append(str(candidate))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find any .xlsx chunk_stats file for run '{spec.run_name}' under '{raw_path}'."
        )

    return sorted(set(candidates))[-1]


def _guess_chunk_stats_path(spec: RunSpec) -> str | None:
    """
    Auto-discover chunk_stats under S3 (or a custom CHUNK_STATS_ROOT) for a given run.

    Default pattern (most runs):
      {CHUNK_STATS_ROOT}/version_{model_version}/chunk_stats/{run_name}_10/{run_date}/*.xlsx

    Returns a concrete file path or None if nothing is found.
    """

    root = CHUNK_STATS_ROOT.rstrip("/")

    dir_name = f"{spec.run_name}_10"

    rel_candidates = [
        f"version_{spec.model_version}/chunk_stats/{dir_name}/{spec.run_date}",
        f"version_{spec.model_version}/chunk_stats/{dir_name}",
    ]

    for rel in rel_candidates:
        base_dir = f"{root}/{rel}"
        try:
            resolved = _resolve_chunk_stats_path_for_run(spec, base_dir)
        except FileNotFoundError:
            continue
        else:
            return resolved

    print(
        f"[chunk_stats] Auto-discovery failed for run '{spec.run_name}' "
        f"(model_version={spec.model_version}, run_date={spec.run_date}) "
        f"under candidates: {[f'{root}/{r}' for r in rel_candidates]}"
    )
    return None


def _assign_colors(run_names: Iterable[str]) -> Mapping[str, str]:
    ordered = sorted(dict.fromkeys(run_names))
    if RUN_COLOR_PALETTE:
        return pc.resolve_colors(ordered, palette=RUN_COLOR_PALETTE)

    cmap = plt.get_cmap("tab10")
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
    spec: RunSpec,
    years: Sequence[int],
    aws_region: str | None,
    chunk_stats_path: str | None,
    chunk_stats_strict: bool = False,
) -> tuple[RunMetrics, RunBreakouts, MetricUncertainty | None]:
    """
    Load run-level metrics.

    If chunk_stats_path is provided and successfully read, drained/burned emissions (and
    uncertainties, if available) come from the chunk_stats file, and zonal stats are
    skipped entirely for that run. If chunk_stats_path is auto-discovered and fails to
    load, a warning is printed and the run falls back to zonal stats; if it was provided
    explicitly via --chunk-stats (chunk_stats_strict=True), failures are fatal errors.
    """
    interval_folders = _interval_folder_strings(years)
    base_prefixes = _make_base_prefixes(spec.model_version, spec.run_name, spec.run_date, interval_folders)
    drained_globs, burned_globs = _make_globs_for_components(base_prefixes)

    con = duckdb.connect()
    uncertainty: MetricUncertainty | None = None
    drained_area = math.nan
    undrained_area = math.nan
    drained_emissions = math.nan
    burned_emissions = math.nan
    by_country = pd.DataFrame()
    by_drained_state = pd.DataFrame()
    by_burned_state = pd.DataFrame()
    by_country_drained_state = pd.DataFrame()
    by_country_burned_state = pd.DataFrame()
    by_climate_component = pd.DataFrame()
    by_period_drained_climate = pd.DataFrame()
    by_period_burned_climate = pd.DataFrame()
    by_period_total_climate = pd.DataFrame()

    stats_override: dict[str, float] | None = None
    if chunk_stats_path:
        loaded = _load_chunk_stats_bounds(chunk_stats_path, years)
        if loaded is None:
            msg = (
                f"[chunk_stats] Failed to load chunk_stats for {spec.run_name}; "
                f"ensure the Excel/CSV/Parquet path exists and contains drained/burned layers. "
                f"Path={chunk_stats_path}"
            )
            if chunk_stats_strict:
                raise RuntimeError(msg)
            else:
                print(msg + " (continuing without chunk_stats override)")
        else:
            stats_override, bounds = loaded
            uncertainty = bounds if bounds.has_bounds() else None
            drained_emissions = float(stats_override.get("drained_emissions_gt", drained_emissions))
            burned_emissions = float(stats_override.get("burned_emissions_gt", burned_emissions))

    try:
        if stats_override is None:
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
            by_climate_component = con.execute(pc.sql_component_split_by_climate_avg(n_periods)).df()
            by_period_drained_climate = con.execute(pc.sql_drained_by_climate()).df()
            by_period_burned_climate = con.execute(pc.sql_burned_by_climate()).df()
            by_period_total_climate = con.execute(pc.sql_total_by_climate()).df()
        else:
            print(
                f"[chunk_stats] Skipping zonal stats for {spec.run_name}; "
                "using chunk_stats totals only."
            )
    finally:
        con.close()

    if stats_override is not None and (math.isnan(drained_area) or math.isnan(undrained_area)):
        print(
            f"[chunk_stats] Using chunk_stats-only totals for {spec.run_name}; "
            "zonal stats not loaded for area or country breakouts."
        )

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
        by_climate_component=_add_run_columns(by_climate_component, spec),
        by_period_drained_climate=_add_run_columns(by_period_drained_climate, spec),
        by_period_burned_climate=_add_run_columns(by_period_burned_climate, spec),
        by_period_total_climate=_add_run_columns(by_period_total_climate, spec),
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
        rows.append(
            {
                "run_key": run_name,
                "Run": record.spec.label,
                "Value": metric.extractor(record.metrics),
                "Metric": metric.label,
                "Units": metric.units,
            }
        )
    return pd.DataFrame(rows)


def _plot_metric(df: pd.DataFrame, metric: MetricSpec, comp: ComparisonSpec, colors: Sequence[str]) -> plt.Figure:
    return pc.barh_single(
        df["Run"].tolist(),
        df["Value"].tolist(),
        xlabel=f"{metric.label} ({metric.units})",
        color=colors,
        sort_desc=False,
    )


def _plot_horizontal_stack(
    df: pd.DataFrame,
    component_order: Sequence[str],
    component_colors: Mapping[str, str],
    xlabel: str,
    title: str,
    legend_columns: int = 2,
) -> plt.Figure:
    return pc.stacked_hbar_generic(
        df, label_col="Run",
        component_order=component_order,
        component_colors=component_colors,
        xlabel=xlabel,
        legend_columns=legend_columns,
    )


def _build_inventory_climate_component_df(
    comp: ComparisonSpec,
    records: Mapping[str, RunRecord],
    component: str,
    years: Sequence[int],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for run_name in comp.run_names:
        record = records[run_name]
        df = record.breakouts.by_climate_component
        if df is None or df.empty:
            continue
        sub = df[df["component"].str.lower() == component.lower()].copy()
        if sub.empty:
            continue
        period_col = _first_present_col(sub, ["interval_end", "period_end_year", "inventory_year", "year"])
        if period_col:
            sub = sub[sub[period_col].isin(years)]
        if sub.empty:
            continue
        sub["Climate"] = sub["climate_domain"].apply(pc.titlecase_domain)
        sub = sub[sub["Climate"].isin(pc.CLIMATE_ORDER)]
        sub["run_key"] = run_name
        sub["Run"] = record.spec.label
        sub = sub.rename(columns={"avg_GtCO2e_per_yr": "Value"})
        rows.append(sub[["run_key", "Run", "Climate", "Value"]])

    if not rows:
        return pd.DataFrame(columns=["run_key", "Run", "Climate", "Value"])

    df_out = pd.concat(rows, ignore_index=True)
    df_out["Run"] = pd.Categorical(
        df_out["Run"],
        [records[rn].spec.label for rn in comp.run_names],
        ordered=True,
    )
    return df_out.sort_values(["Run", "Climate"]).reset_index(drop=True)


def _build_inventory_climate_stack_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a wide-format dataframe for stacked climate emissions in the
    Inventory Input Source comparison. Fixes the issue where extra
    all-zero rows appear because 'Run' is categorical and pivot_table
    expands to the full cartesian product when observed=False.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    # Ensure run order matches the comparison order
    run_order = (
        df["Run"]
        .dropna()
        .unique()
        .tolist()
    )

    # Pivot only on ACTUALLY OBSERVED combinations
    wide = (
        df.pivot_table(
            index=["run_key", "Run"],
            columns="Climate",
            values="Value",
            aggfunc="sum",
            fill_value=0.0,
            observed=True,  # <-- CRITICAL FIX
        )
        .reindex(columns=pc.CLIMATE_ORDER, fill_value=0.0)
        .reset_index()
    )

    # Enforce run order
    if "Run" in wide.columns:
        wide["Run"] = pd.Categorical(wide["Run"], categories=run_order, ordered=True)
        wide = wide.sort_values("Run")

    # Add total
    wide["Total"] = wide[pc.CLIMATE_ORDER].sum(axis=1)

    return wide



def _plot_inventory_climate_component(df: pd.DataFrame, component_label: str) -> plt.Figure | None:
    if df is None or df.empty:
        return None

    theme = {**pc.THEME_LIGHT_GRID, "axes.grid.axis": "y"}
    with pc.use_theme(theme):
        fig = pc.stacked_column_by_category(
            df_long=df,
            index_col="Run",
            category_col="Climate",
            value_col="Value",
            category_order=pc.CLIMATE_ORDER,
            color_map=pc.CLIMATE_COLORS,
            xlabel="Inventory Input Source",
            ylabel=f"{component_label} emissions (Gt CO₂e/year)",
            width=7.8,
            height=4.6,
            legend_above=True,
        )
        fig.tight_layout()
        return fig


def _inventory_period_labels(years: Sequence[int]) -> dict[int, str]:
    pairs = build_interval_pairs(list(years))
    return {end: f"{start}-{end}" for start, end in pairs}


def _build_inventory_period_climate_df(
    comp: ComparisonSpec,
    records: Mapping[str, RunRecord],
    component: str,
    period_labels: Mapping[int, str],
    period_order: Sequence[str],
) -> pd.DataFrame:
    attr_map = {
        "drained": "by_period_drained_climate",
        "burned": "by_period_burned_climate",
        "total": "by_period_total_climate",
    }
    value_map = {
        "drained": "drained_GtCO2e",
        "burned": "burned_GtCO2e",
        "total": "total_GtCO2e",
    }

    attr = attr_map[component.lower()]
    value_col = value_map[component.lower()]

    rows: list[pd.DataFrame] = []
    for run_name in comp.run_names:
        record = records[run_name]
        df = getattr(record.breakouts, attr, pd.DataFrame())
        if df is None or df.empty:
            continue

        sub = df.copy()
        sub["Climate"] = sub["climate_domain"].apply(pc.titlecase_domain)
        sub = sub[sub["Climate"].isin(pc.CLIMATE_ORDER)]
        sub["Inventory period"] = sub["interval_end"].map(period_labels)
        sub = sub[~sub["Inventory period"].isna()]
        sub["Inventory period"] = pd.Categorical(sub["Inventory period"], period_order, ordered=True)
        sub["run_key"] = run_name
        sub["Run"] = record.spec.label
        sub = sub.rename(columns={value_col: "Value"})
        rows.append(sub[["run_key", "Run", "Inventory period", "Climate", "Value"]])

    if not rows:
        return pd.DataFrame(columns=["run_key", "Run", "Inventory period", "Climate", "Value"])

    df_out = pd.concat(rows, ignore_index=True)
    df_out["Run"] = pd.Categorical(
        df_out["Run"],
        [records[rn].spec.label for rn in comp.run_names],
        ordered=True,
    )
    return df_out.sort_values(["Run", "Inventory period", "Climate"]).reset_index(drop=True)


def _plot_inventory_period_climate(
    df: pd.DataFrame, component_label: str, run_label: str
) -> plt.Figure | None:
    if df is None or df.empty:
        return None

    theme = {**pc.THEME_LIGHT_GRID, "axes.grid.axis": "y"}
    with pc.use_theme(theme):
        fig = pc.stacked_column_by_category(
            df_long=df,
            index_col="Inventory period",
            category_col="Climate",
            value_col="Value",
            category_order=pc.CLIMATE_ORDER,
            color_map=pc.CLIMATE_COLORS,
            xlabel="Inventory period",
            ylabel=f"{component_label} emissions (Gt CO₂e/year)",
            width=7.5,
            height=4.5,
            legend_above=True,
        )
        fig.tight_layout()
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


def _build_peat_area_stack_df(comp: ComparisonSpec, records: Mapping[str, RunRecord]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for run_name in comp.run_names:
        record = records[run_name]
        drained = record.metrics.drained_area_mha
        undrained = record.metrics.undrained_area_mha
        rows.append(
            {
                "run_key": run_name,
                "Run": record.spec.label,
                "Drained": drained,
                "Undrained": undrained,
                "Total": drained + undrained,
            }
        )
    return pd.DataFrame(rows)


def _apply_category_labels(df: pd.DataFrame, overrides: Mapping[str, str]) -> pd.DataFrame:
    if not overrides:
        return df

    df = df.copy()
    df["Run"] = df["run_key"].map(overrides).fillna(df["Run"])
    return df


def _plot_stacked_total(
    df: pd.DataFrame,
    comp: ComparisonSpec,
    component_colors: Mapping[str, str] = STACK_COMPONENT_COLORS,
    xlabel: str | None = None,
) -> plt.Figure:
    """
    Single vertical stacked bar chart for total emissions split drained/burned.

    Bars are ordered exactly as in df; for the sensitivity-range comparison this
    is Low – Baseline – High so the baseline appears in the middle.
    """
    x = list(range(len(df)))
    colors = pc.resolve_colors(STACK_COMPONENT_ORDER, component_colors)

    with pc.use_theme(pc.THEME_LIGHT_GRID):
        fig, ax = plt.subplots(figsize=(max(6.5, 2.2 * len(x)), 5.0))

        drained_bar = ax.bar(x, df["Drained"], label="Drained", color=colors["Drained"])
        burned_bar = ax.bar(
            x,
            df["Burned"],
            bottom=df["Drained"],
            label="Burned",
            color=colors["Burned"],
        )

        # Optional error bars on Total (if present)
        if df[["Total_low", "Total_high"]].notna().any().any():
            totals = df["Total"]
            total_low = df["Total_low"].fillna(totals)
            total_high = df["Total_high"].fillna(totals)
            lower = totals - total_low
            upper = total_high - totals
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
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel("Gt CO₂e/year")
        ax.set_axisbelow(True)
        pc.tidy_axes(ax, grid="y")
        pc.fmt_si(ax, axis="y")
        ax.set_ylim(bottom=0.0)
        ax.legend(frameon=False)

        # Label drained + burned segments
        for bar, val in zip(drained_bar, df["Drained"]):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height * 0.5,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )
        for bar, val_d, val_b in zip(burned_bar, df["Drained"], df["Burned"]):
            if val_b <= 0:
                continue
            bottom = val_d
            height = val_b
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bottom + height * 0.5,
                f"{val_b:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

        fig.tight_layout()
        return fig


def _collect_breakouts(
    records: Mapping[str, RunRecord], attr: str, run_filter: Iterable[str] | None = None
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_name in sorted(records):
        if run_filter is not None and run_name not in run_filter:
            continue
        df = getattr(records[run_name].breakouts, attr)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# === Helpers for per-country disagreement across runs (inventory_source) ===


def _first_present_col(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
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
    mean = _get_first_value(
        row,
        [
            "mean_gtco2e_per_yr",
            "avg_gtco2e_per_yr",
            "mean",
            "average",
            "median",
            "value",
        ],
    )
    vmin = _get_first_value(row, ["min", "lower", "low", "p05", "p5"])
    vmax = _get_first_value(row, ["max", "upper", "high", "p95", "p90"])
    return mean, vmin, vmax


def _load_chunk_stats_bounds(path: str, years: Sequence[int]) -> tuple[dict[str, float], MetricUncertainty] | None:
    def _read_chunk_table(chunk_path: str) -> pd.DataFrame | None:
        try:
            return pd.read_excel(chunk_path, sheet_name="other_outputs_1x1")
        except Exception:
            pass
        try:
            if chunk_path.lower().endswith(".parquet"):
                return pd.read_parquet(chunk_path)
            return pd.read_csv(chunk_path)
        except Exception as exc:  # pragma: no cover - IO handling
            print(f"[chunk_stats] Failed to read {chunk_path}: {exc}")
            return None

    df = _read_chunk_table(path)
    if df is None or df.empty:
        return None

    df = _latest_period_filter(df, years)

    layer_col = _first_present_col(df, ["layer_name", "layer", "metric"])
    if layer_col and "sum_value" in df.columns:
        keep_layers = {
            "drained_total_Mg_CO2e_ha_yr": "drained_emissions_gt",
            "burned_total_Mg_CO2e_ha_yr": "burned_emissions_gt",
        }
        lower = df[layer_col].astype(str).str.lower()
        stats: dict[str, float] = {}
        bounds = MetricUncertainty()

        for raw_name, key in keep_layers.items():
            subset = df[lower == raw_name.lower()]
            if subset.empty:
                continue
            total_mg = float(subset["sum_value"].sum())
            stats[key] = total_mg / 1_000_000_000.0

        if not stats:
            print(
                f"[chunk_stats] No drained/burned layers found in {path}; "
                f"available layers={sorted(df[layer_col].dropna().unique())}"
            )
        else:
            if {"drained_emissions_gt", "burned_emissions_gt"} <= set(stats):
                stats["total_emissions_gt"] = stats["drained_emissions_gt"] + stats["burned_emissions_gt"]
            return stats, bounds

    comp_col = _first_present_col(df, ["component", "metric", "flux_component", "name"])
    if not comp_col:
        print(f"[chunk_stats] No component column detected in {path}; columns={list(df.columns)}")
        return None

    stats: dict[str, float] = {}
    bounds = MetricUncertainty()
    comp_series = df[comp_col].astype(str).str.lower()

    for comp_name, label in (("drained", "Drained"), ("burned", "Burned"), ("total", "Total")):
        subset = df[comp_series.str.contains(comp_name)]
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
    col = _first_present_col(
        df,
        [
            "peat_area_mha",
            "total_peat_area_mha",
            "area_mha",
            "drained_area_mha",
            "undrained_area_mha",
        ],
    )
    if col:
        return col, 1.0
    col = _first_present_col(
        df,
        [
            "peat_area_ha",
            "total_peat_area_ha",
            "area_ha",
            "ha",
            "drained_area_ha",
            "undrained_area_ha",
        ],
    )
    if col:
        return col, 1e-6
    col = _first_present_col(
        df,
        [
            "area_km2",
            "km2",
            "area_km²",
            "km²",
            "drained_area_km2",
            "undrained_area_km2",
        ],
    )
    if col:
        return col, 1.0 / 10000.0
    return None, 1.0


def _normalize_drained_state_values(series: pd.Series) -> pd.Series:
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
            return pd.Series(
                {
                    "n_datasets": n,
                    "min_mha": pd.NA,
                    "median_mha": pd.NA,
                    "max_mha": pd.NA,
                    "abs_spread_mha": pd.NA,
                    "fold_change": pd.NA,
                    "log10_spread": pd.NA,
                    "min_run": pd.NA,
                    "max_run": pd.NA,
                    "flag": "",
                }
            )
        vmin = float(vals.min())
        vmax = float(vals.max())
        abs_spread = vmax - vmin
        pos = vals[vals > 0.0]
        if not pos.empty:
            min_pos = float(pos.min())
            fold = (vmax / min_pos) if min_pos > 0.0 else math.inf
            log10s = math.log10(fold) if math.isfinite(fold) and fold > 0.0 else pd.NA
        else:
            fold = math.inf if vmax > 0.0 else pd.NA
            log10s = pd.NA

        min_run = vals.idxmin()
        max_run = vals.idxmax()

        rel_ok = isinstance(log10s, float) and (log10s >= math.log10(rel_flag_fold))
        abs_ok = abs_spread >= abs_flag_mha
        flag = "FLAG" if (rel_ok or abs_ok) else ""

        return pd.Series(
            {
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
            }
        )

    summary = wide.copy()
    metrics = summary.apply(_row_metrics, axis=1)
    summary = pd.concat([summary[keys], metrics], axis=1)

    flagged = summary[(summary["n_datasets"] >= 2) & (summary["flag"] == "FLAG")].copy()

    wide = wide[keys + col_order]
    summary_cols = keys + [
        "n_datasets",
        "min_mha",
        "median_mha",
        "max_mha",
        "abs_spread_mha",
        "fold_change",
        "log10_spread",
        "min_run",
        "max_run",
        "flag",
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

    _write_csv_df(writer_con, wide, _join(out_dir, f"{comp.key}_country_peat_area_wide.csv"))
    _write_csv_df(writer_con, summary, _join(out_dir, f"{comp.key}_country_peat_area_summary.csv"))
    _write_csv_df(writer_con, flagged, _join(out_dir, f"{comp.key}_country_peat_area_flagged.csv"))
    print(
        f"[inventory_source] wrote: "
        f"{comp.key}_country_peat_area_wide.csv, "
        f"{comp.key}_country_peat_area_summary.csv, "
        f"{comp.key}_country_peat_area_flagged.csv"
    )


def main(argv: Sequence[str] | None = None):
    global OUT_DIR_ROOT
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
    parser.add_argument(
        "--inventory-label",
        action="append",
        help=(
            "Optional category label override for the inventory source comparison (run_name=Label). "
            "Only affects that comparison's outputs."
        ),
    )
    parser.add_argument(
        "--emission-factor-label",
        action="append",
        help=(
            "Optional category label override for the emission factor comparison (run_name=Label). "
            "Only affects that comparison's outputs."
        ),
    )
    parser.add_argument(
        "--distance-label",
        action="append",
        help=(
            "Optional category label override for the distance threshold comparison (run_name=Label). "
            "Only affects that comparison's outputs."
        ),
    )
    parser.add_argument("--aws_region", default=None, help="Optional AWS region for S3 access")
    parser.add_argument("--data-only", action="store_true", help="Export CSV data only (skip figures)")
    parser.add_argument(
        "--out-dir-root",
        default=OUT_DIR_ROOT,
        help=f"Output root (default: {OUT_DIR_ROOT}).",
    )
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
            "Optional chunk_stats location per run (run_name=path). "
            "Path may be a specific Excel/CSV/Parquet file or a directory/prefix; "
            "for directories/prefixes the script will search for any .xlsx file "
            "in that folder (and a <folder>/<run_date> child for local FS, or "
            "<prefix>/<run_date> on S3). "
            "If omitted, the script will auto-discover chunk_stats for OGH "
            "sensitivity runs under "
            "CHUNK_STATS_ROOT/version_<ver>/chunk_stats/<dir_name>/<run_date>."
        ),
    )
    args = parser.parse_args(argv)

    OUT_DIR_ROOT = args.out_dir_root

    try:
        years = [int(y) for y in args.years]
    except ValueError as exc:  # pragma: no cover - CLI validation
        raise SystemExit(f"Invalid --years value: {exc}")

    if not years:
        raise SystemExit("At least one inventory year must be provided")

    period_labels = _inventory_period_labels(years)
    period_order = [period_labels[end] for end in sorted(period_labels)]

    run_specs = _parse_run_specs(args.run)
    inventory_labels = _parse_category_labels(args.inventory_label)
    emission_factor_labels = _parse_category_labels(args.emission_factor_label)
    distance_labels = _parse_category_labels(args.distance_label)

    user_chunk_stat_paths_raw = _parse_chunk_stat_paths(args.chunk_stats)
    chunk_stat_config: dict[str, tuple[str | None, bool]] = {}

    for run_name, spec in run_specs.items():
        if run_name in user_chunk_stat_paths_raw:
            # Explicit override – treat strictly
            raw = user_chunk_stat_paths_raw[run_name]
            try:
                resolved = _resolve_chunk_stats_path_for_run(spec, raw)
            except FileNotFoundError as exc:
                raise SystemExit(f"Could not resolve --chunk-stats path for run '{run_name}': {exc}")
            chunk_stat_config[run_name] = (resolved, True)
        elif run_name in SENSITIVITY_CHUNK_RUNS:
            guessed = _guess_chunk_stats_path(spec)
            if guessed:
                chunk_stat_config[run_name] = (guessed, False)
                print(f"[chunk_stats] Auto-discovered chunk_stats for {run_name}: {guessed}")

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

    # Resolution-consistency policy (availability-driven): each comparison is
    # rendered at a SINGLE resolution so the ~3% chunk_stats coarsening bias
    # cannot contaminate the deltas.
    #   * if every run in the comparison has zonal stats (40000 px / ~28 m) ->
    #     use zonal for all (preferred: finest, matches the headline figures);
    #   * else if every run has chunk_stats (4000 px / ~278 m) -> use chunk for
    #     all (the systematic coarsening bias then cancels in the relative deltas);
    #   * else -> skip the comparison (cannot render it at one resolution).
    # SENSITIVITY_CHUNK_RUNS / --chunk-stats only PRE-RESOLVE chunk_stats paths;
    # they no longer force chunk when zonal is also available. So once zonal stats
    # are produced for the sensitivity runs, the comparisons automatically switch
    # to all-zonal with no code change.
    _probe_con = duckdb.connect()
    pa._ensure_httpfs(_probe_con, args.aws_region)
    zonal_avail_cache: dict[str, bool] = {}

    def _has_zonal(run_name: str) -> bool:
        if run_name not in zonal_avail_cache:
            spec = run_specs[run_name]
            base_prefixes = _make_base_prefixes(
                spec.model_version, spec.run_name, spec.run_date,
                _interval_folder_strings(years),
            )
            drained_globs, _ = _make_globs_for_components(base_prefixes)
            try:
                zonal_avail_cache[run_name] = pa._count_globs(_probe_con, drained_globs) > 0
            except Exception:
                zonal_avail_cache[run_name] = False
        return zonal_avail_cache[run_name]

    chunk_path_cache: dict[str, str | None] = {}

    def _chunk_path_for(run_name: str) -> str | None:
        if run_name not in chunk_path_cache:
            if run_name in chunk_stat_config:
                chunk_path_cache[run_name] = chunk_stat_config[run_name][0]
            else:
                chunk_path_cache[run_name] = _guess_chunk_stats_path(run_specs[run_name])
        return chunk_path_cache[run_name]

    # Records are loaded lazily and cached per (run_name, use_chunk); the baseline
    # may be loaded at both resolutions (zonal for the inventory comparison, chunk
    # for the sensitivity comparisons).
    record_cache: dict[tuple[str, bool], RunRecord] = {}

    def _get_record(run_name: str, use_chunk: bool) -> RunRecord:
        key = (run_name, use_chunk)
        if key not in record_cache:
            spec = run_specs[run_name]
            if use_chunk:
                cs_path = _chunk_path_for(run_name)
                cs_strict = chunk_stat_config.get(run_name, (None, False))[1]
            else:
                cs_path, cs_strict = None, False
            metrics, breakouts, uncertainty = _compute_run_data(
                spec, years, args.aws_region, cs_path, chunk_stats_strict=cs_strict
            )
            record_cache[key] = RunRecord(
                spec=spec,
                metrics=metrics,
                breakouts=breakouts,
                color=color_map[run_name],
                uncertainty=uncertainty,
            )
        return record_cache[key]

    def _resolve_comparison_source(comp: ComparisonSpec) -> str | None:
        """Choose a single resolution shared by every run in the comparison.

        Returns 'zonal', 'chunk', or None (skip). Prefers zonal; falls back to
        chunk only when zonal is missing for one or more runs but chunk_stats is
        available for all of them.
        """
        runs = comp.run_names
        if all(_has_zonal(rn) for rn in runs):
            return "zonal"
        if all(_chunk_path_for(rn) for rn in runs):
            return "chunk"
        return None

    out_data_dir = _join(out_dir, "figures", "comparisons", "data")
    writer_con = duckdb.connect()
    try:
        for comp in active_comparisons:
            # Load every run in this comparison at a single, consistent resolution
            # (zonal preferred; chunk fallback) so the deltas are not biased by a
            # mix of 40000 px and 4000 px sources.
            source = _resolve_comparison_source(comp)
            if source is None:
                missing_zonal = [rn for rn in comp.run_names if not _has_zonal(rn)]
                print(
                    f"Skipping comparison '{comp.key}': cannot render all of its runs at a "
                    f"single resolution. Runs without zonal stats: "
                    f"{', '.join(missing_zonal) or 'none'}; and chunk_stats is not available "
                    f"for all runs either. Produce zonal stats for every run (preferred), or "
                    f"chunk_stats for every run."
                )
                continue
            use_chunk = source == "chunk"
            records = {rn: _get_record(rn, use_chunk) for rn in comp.run_names}

            label_overrides: Mapping[str, str] = {}
            if comp.key == "inventory_source":
                label_overrides = inventory_labels
            elif comp.key in {"ogh_sensitivity_range", "ogh_ef_effect"}:
                label_overrides = emission_factor_labels
            elif comp.key == "ogh_distance":
                label_overrides = distance_labels

            summary_df = _build_comparison_summary(comp, records)
            summary_path = _join(out_data_dir, f"{comp.key}_summary.csv")
            _write_csv_df(writer_con, summary_df, summary_path)

            # Per-metric long-form CSVs for all comparisons
            for metric_key in comp.metric_keys:
                metric = METRIC_SPECS[metric_key]
                metric_df = _build_metric_long_df(comp, metric, records)
                metric_path = _join(out_data_dir, f"{comp.key}_{metric.key}.csv")
                _write_csv_df(writer_con, metric_df, metric_path)

                # Only inventory comparison gets per-metric bar figures
                if args.data_only:
                    continue
                if comp.key == "inventory_source":
                    colors = [records[rn].color for rn in comp.run_names]
                    fig = _plot_metric(metric_df, metric, comp, colors)
                    fig_path = _join(out_dir, "figures", "comparisons", f"{comp.key}_{metric.key}.png")
                    _save_png(fig, fig_path)
                    plt.close(fig)

            if comp.key == "inventory_source":
                total_stack_df = _build_emission_stack_df(comp, records)
                total_stack_df = _apply_category_labels(total_stack_df, label_overrides)
                total_stack_path = _join(out_data_dir, f"{comp.key}_total_emissions_stack.csv")
                _write_csv_df(writer_con, total_stack_df, total_stack_path)

                area_stack_df = _build_peat_area_stack_df(comp, records)
                area_stack_df = _apply_category_labels(area_stack_df, label_overrides)
                area_stack_path = _join(out_data_dir, f"{comp.key}_peat_area_stack.csv")
                _write_csv_df(writer_con, area_stack_df, area_stack_path)

                for component in ("Drained", "Burned"):
                    climate_df = _build_inventory_climate_component_df(comp, records, component, years)
                    climate_stack_df = _build_inventory_climate_stack_df(climate_df)
                    climate_stack_path = _join(
                        out_data_dir,
                        f"{comp.key}_{component.lower()}_emissions_by_climate_stack.csv",
                    )
                    climate_stack_df = _apply_category_labels(climate_stack_df, label_overrides)
                    _write_csv_df(writer_con, climate_stack_df, climate_stack_path)

                if not args.data_only:
                    total_stack_fig = _plot_horizontal_stack(
                        total_stack_df,
                        STACK_COMPONENT_ORDER,
                        STACK_COMPONENT_COLORS,
                        "Total emissions (Gt CO₂e/year)",
                        comp.label,
                    )
                    total_stack_fig_path = _join(
                        out_dir,
                        "figures",
                        "comparisons",
                        f"{comp.key}_total_emissions_stack.png",
                    )
                    _save_png(total_stack_fig, total_stack_fig_path)
                    plt.close(total_stack_fig)

                    area_stack_fig = _plot_horizontal_stack(
                        area_stack_df,
                        ("Drained", "Undrained"),
                        PEAT_AREA_COLORS,
                        "Peat area (million ha)",
                        comp.label,
                    )
                    area_stack_fig_path = _join(
                        out_dir,
                        "figures",
                        "comparisons",
                        f"{comp.key}_peat_area_stack.png",
                    )
                    _save_png(area_stack_fig, area_stack_fig_path)
                    plt.close(area_stack_fig)

                    for component in ("Drained", "Burned"):
                        climate_df = _build_inventory_climate_component_df(comp, records, component, years)
                        climate_stack_df = _build_inventory_climate_stack_df(climate_df)
                        if climate_stack_df.empty:
                            continue
                        climate_components = [c for c in pc.CLIMATE_ORDER if c in climate_stack_df.columns]
                        climate_stack_fig = _plot_horizontal_stack(
                            climate_stack_df,
                            climate_components,
                            pc.CLIMATE_COLORS,
                            f"{component} emissions (Gt CO₂e/year)",
                            f"{comp.label} – {component} by climate",
                            legend_columns=3,
                        )
                        climate_stack_fig_path = _join(
                            out_dir,
                            "figures",
                            "comparisons",
                            f"{comp.key}_{component.lower()}_emissions_by_climate_stack.png",
                        )
                        _save_png(climate_stack_fig, climate_stack_fig_path)
                        plt.close(climate_stack_fig)

                for component in ("Drained", "Burned"):
                    climate_df = _build_inventory_climate_component_df(comp, records, component, years)
                    long_path = _join(
                        out_data_dir,
                        f"{comp.key}_{component.lower()}_by_climate_long.csv",
                    )
                    _write_csv_df(writer_con, climate_df, long_path)
                    wide_path = _join(
                        out_data_dir,
                        f"{comp.key}_{component.lower()}_by_climate_wide.csv",
                    )
                    _write_csv_df(
                        writer_con,
                        pc.pivot_wide(climate_df[["Run", "Climate", "Value"]], "Value", "Run"),
                        wide_path,
                    )

                    if args.data_only:
                        continue

                    fig = _plot_inventory_climate_component(climate_df, component)
                    if fig is not None:
                        fig_path = _join(
                            out_dir,
                            "figures",
                            "comparisons",
                            f"{comp.key}_{component.lower()}_by_climate.png",
                        )
                        _save_png(fig, fig_path)
                        plt.close(fig)

                for component in ("Drained", "Burned", "Total"):
                    period_climate_df = _build_inventory_period_climate_df(
                        comp, records, component, period_labels, period_order
                    )

                    long_path = _join(
                        out_data_dir,
                        f"{comp.key}_{component.lower()}_by_period_climate_long.csv",
                    )
                    _write_csv_df(writer_con, period_climate_df, long_path)

                    wide = (
                        period_climate_df.pivot_table(
                            index=["Run", "Inventory period"],
                            columns="Climate",
                            values="Value",
                            aggfunc="sum",
                            fill_value=0.0,
                            observed=False,
                        )
                        .reindex(columns=pc.CLIMATE_ORDER, fill_value=0.0)
                        .reset_index()
                    )
                    wide_path = _join(
                        out_data_dir,
                        f"{comp.key}_{component.lower()}_by_period_climate_wide.csv",
                    )
                    _write_csv_df(writer_con, wide, wide_path)

                    if args.data_only:
                        continue

                    for run_name in comp.run_names:
                        run_df = period_climate_df[period_climate_df["run_key"] == run_name]
                        fig = _plot_inventory_period_climate(
                            run_df, component, records[run_name].spec.label
                        )
                        if fig is None:
                            continue
                        fig_path = _join(
                            out_dir,
                            "figures",
                            "comparisons",
                            f"{comp.key}_{component.lower()}_by_period_climate__{run_name}.png",
                        )
                        _save_png(fig, fig_path)
                        plt.close(fig)

            if args.data_only:
                continue

            # Single stacked total-emissions chart for sensitivity comparisons
            if comp.key in {"ogh_distance", "ogh_sensitivity_range", "ogh_area_effect", "ogh_ef_effect"}:
                stack_df = _build_emission_stack_df(comp, records)
                stack_df = _apply_category_labels(stack_df, label_overrides)
                stack_path = _join(out_data_dir, f"{comp.key}_drained_burned_stack.csv")
                _write_csv_df(writer_con, stack_df, stack_path)

                xlabel = None
                if comp.key == "ogh_distance":
                    xlabel = "Distance threshold"
                elif comp.key == "ogh_sensitivity_range":
                    xlabel = "Area threshold + emission factor"
                elif comp.key == "ogh_area_effect":
                    xlabel = "Mapped area threshold"
                elif comp.key == "ogh_ef_effect":
                    xlabel = "Emission factor"

                stack_fig = _plot_stacked_total(stack_df, comp, xlabel=xlabel)
                stack_fig_path = _join(out_dir, "figures", "comparisons", f"{comp.key}_total_stack.png")
                _save_png(stack_fig, stack_fig_path)
                plt.close(stack_fig)

        # Breakout exports want the zonal (40000 px) view, which carries the
        # per-country/state/climate detail. Use the zonal variant of each run
        # where it was loaded; chunk-only runs fall back to their chunk record
        # (empty breakouts), exactly as before this resolution split.
        breakout_records: dict[str, RunRecord] = {}
        for run_name in run_specs:
            if (run_name, False) in record_cache:
                breakout_records[run_name] = record_cache[(run_name, False)]
            elif (run_name, True) in record_cache:
                breakout_records[run_name] = record_cache[(run_name, True)]

        breakout_specs = (
            ("by_country_period", "by_country"),
            ("by_drained_state_period", "by_drained_state"),
            ("by_burned_state_period", "by_burned_state"),
            ("by_country_drained_state_period", "by_country_drained_state"),
            ("by_country_burned_state_period", "by_country_burned_state"),
            ("by_climate_component", "by_climate_component"),
            ("by_period_drained_climate", "by_period_drained_climate"),
            ("by_period_burned_climate", "by_period_burned_climate"),
            ("by_period_total_climate", "by_period_total_climate"),
        )
        climate_breakouts = {
            "by_climate_component",
            "by_period_drained_climate",
            "by_period_burned_climate",
            "by_period_total_climate",
        }
        inventory_comp = next((c for c in active_comparisons if c.key == "inventory_source"), None)
        inventory_runs: Iterable[str] | None = inventory_comp.run_names if inventory_comp else None

        for file_stub, attr in breakout_specs:
            if attr in climate_breakouts and inventory_runs is None:
                breakout_df = pd.DataFrame()
            else:
                run_filter = inventory_runs if attr in climate_breakouts else None
                breakout_df = _collect_breakouts(breakout_records, attr, run_filter=run_filter)
            breakout_path = _join(out_data_dir, f"runs_{file_stub}.csv")
            _write_csv_df(writer_con, breakout_df, breakout_path)

        for comp in active_comparisons:
            if comp.key == "inventory_source":
                _export_inventory_source_area_disagreement(
                    comp=comp,
                    records=breakout_records,
                    years=years,
                    abs_flag_mha=float(args.flag_abs_mha),
                    rel_flag_fold=float(args.flag_rel_fold),
                    writer_con=writer_con,
                    out_dir=out_data_dir,
                )
    finally:
        writer_con.close()

    print("Comparison assets written to:", out_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
