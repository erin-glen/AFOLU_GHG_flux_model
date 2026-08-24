# -*- coding: utf-8 -*-
"""Reusable, offline publication-figure constructors.

The functions in this module intentionally stop at returning a Matplotlib
``Figure``.  They do not save files, query remote data, or mutate the frozen
input tables.  Export format, physical dimensions, and QA are handled by the
offline publication orchestrator.

These constructors promote the approved plotting lineage used for the August
2026 corrected organic-soils publication figures.  In particular, Figure 2b
retains the manuscript's stacked bars, Figure 9d prints three decimal places, and
Figure 13 reads its corrected baseline and sensitivity endpoints from the
frozen values table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import pandas as pd

from src.scripts.zonal_statistics.pub_scripts import pub_common as pc


FrameSource: TypeAlias = pd.DataFrame | str | Path


def _read_frame(source: FrameSource) -> pd.DataFrame:
    """Return a defensive copy of a DataFrame or a locally stored CSV."""

    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_csv(Path(source))


def _require_columns(df: pd.DataFrame, required: set[str], figure: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{figure} input is missing columns: {', '.join(missing)}")


def build_figure_01a(source: FrameSource) -> plt.Figure:
    """Build Figure 1a: global annual emissions split by process."""

    df = _read_frame(source)
    required = {"Inventory Period", "Component", "GtCO2e"}
    _require_columns(df, required, "Figure 1a")
    return pc.stacked_column_by_category(
        df,
        index_col="Inventory Period",
        category_col="Component",
        value_col="GtCO2e",
        category_order=pc.PROCESS_ORDER,
        color_map=pc.PROCESS_COLORS,
        xlabel="Inventory Period",
        ylabel="Annual Emissions (Gt CO$_2$e/year)",
    )


def build_figure_02b(source: FrameSource) -> plt.Figure:
    """Build Figure 2b using the manuscript's illustrative stacked layout."""

    df = _read_frame(source)
    required = {"Climate", "Component", "intensity_tCO2e_per_ha_yr"}
    _require_columns(df, required, "Figure 2b")

    if set(df["Climate"]) != set(pc.CLIMATE_ORDER):
        raise ValueError("Figure 2b input does not contain every climate")
    df["Climate"] = pd.Categorical(df["Climate"], pc.CLIMATE_ORDER, ordered=True)
    df["Component"] = pd.Categorical(df["Component"], pc.PROCESS_ORDER, ordered=True)
    df = df.sort_values(["Climate", "Component"])
    return pc.stacked_column_by_category(
        df,
        index_col="Climate",
        category_col="Component",
        value_col="intensity_tCO2e_per_ha_yr",
        category_order=pc.PROCESS_ORDER,
        color_map=pc.PROCESS_COLORS,
        xlabel="Climate",
        ylabel="Emissions Intensity (t CO$_2$e/ha/year)",
        legend_above=True,
        bar_width=0.55,
        segment_edgecolor="white",
        segment_linewidth=0.6,
        # The caption explains that the visual stack is illustrative because
        # drained and burned intensities use separate denominators.
        show_totals=False,
    )


def build_figure_09d(source: FrameSource) -> plt.Figure:
    """Build Figure 9d with approved three-decimal value labels."""

    df = _read_frame(source)
    required = {"iso3_or_code", "burned_avg_GtCO2e_per_yr"}
    _require_columns(df, required, "Figure 9d")
    df = df.sort_values(
        "burned_avg_GtCO2e_per_yr", ascending=False, kind="stable"
    )
    labels = df["iso3_or_code"].astype(str).tolist()
    values = df["burned_avg_GtCO2e_per_yr"].astype(float).tolist()
    y_positions = list(range(len(labels)))
    height = max(3.0, 0.5 * len(labels) + 1.0)

    with pc.use_theme({**pc.THEME_LIGHT_GRID, "axes.grid.axis": "x"}):
        fig, ax = plt.subplots(figsize=(7.5, height))
        ax.barh(y_positions, values, color=pc.PROCESS_COLORS["Burned"])
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Average Annual Emissions (Gt CO$_2$e/year)")
        pc.tidy_axes(ax, grid="x")
        pc.fmt_si(ax, axis="x")

        x_max = max(values) if values else 1.0
        ax.set_xlim(0, x_max * 1.10)
        for y_position, value in zip(y_positions, values, strict=True):
            ax.text(
                value + (x_max * 0.01),
                y_position,
                f"{value:.3f}",
                ha="left",
                va="center",
                fontsize=9,
            )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def build_figure_10b(source: FrameSource) -> plt.Figure:
    """Build Figure 10b: burned emissions by land use and climate."""

    df = _read_frame(source)
    required = {"LandUse", "Climate", "burned_avg_GtCO2e_per_yr"}
    _require_columns(df, required, "Figure 10b")
    df = pc.merge_burned_drained_other(
        df, "burned_avg_GtCO2e_per_yr"
    )
    original_palette = pc.CLIMATE_COLORS.copy()
    try:
        pc.set_climate_palette("brewer_dark2")
        return pc.stacked_hbar(
            df,
            "burned_avg_GtCO2e_per_yr",
            xlabel="Annual Emissions, 2021-2024 (Gt CO$_2$e/year)",
        )
    finally:
        pc.CLIMATE_COLORS.clear()
        pc.CLIMATE_COLORS.update(original_palette)


def build_figure_10a(source: FrameSource) -> plt.Figure:
    """Build Figure 10a with publication-facing land-use labels."""

    df = _read_frame(source)
    required = {"LandUse", "Climate", "drained_avg_GtCO2e_per_yr"}
    _require_columns(df, required, "Figure 10a")
    # Keep the model's internal category key unchanged in the frozen table;
    # only the plotted label receives normal word spacing.
    df["LandUse"] = df["LandUse"].astype(str).replace(
        {"Otherland": "Other land"}
    )
    original_palette = pc.CLIMATE_COLORS.copy()
    try:
        pc.set_climate_palette("brewer_dark2")
        return pc.stacked_hbar(
            df,
            "drained_avg_GtCO2e_per_yr",
            xlabel="Annual Emissions, 2021-2024 (Gt CO$_2$e/year)",
        )
    finally:
        pc.CLIMATE_COLORS.clear()
        pc.CLIMATE_COLORS.update(original_palette)


def build_figure_11a(source: FrameSource) -> plt.Figure:
    """Build Figure 11a through the canonical comparison renderer."""

    df = _read_frame(source)
    required = {"Run", "Drained", "Undrained"}
    _require_columns(df, required, "Figure 11a")

    # Import lazily so this module remains a lightweight offline constructor.
    # The comparison module is reused unchanged; no publication data are read.
    from src.scripts.zonal_statistics.pub_scripts import pub_compare_runs as pcr

    return pcr._plot_horizontal_stack(
        df,
        ("Drained", "Undrained"),
        pcr.PEAT_AREA_COLORS,
        "Mapped area (million ha)",
        "Inventory Input Source Comparison",
    )


@dataclass(frozen=True)
class SensitivityFactor:
    """One one-at-a-time Figure 13 sensitivity range."""

    key: str
    label: str
    low: float
    high: float

    @property
    def swing(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class Figure13Controls:
    """Numerical controls required to draw and independently QA Figure 13."""

    baseline: float
    combined_low: float
    combined_high: float
    factors: tuple[SensitivityFactor, ...]


_FIGURE_13_LABELS = {
    "emission_factors": "Tier 1 emission factors\n(low / high Tier 1 bounds)",
    "extent_threshold": (
        "Organic-soil extent threshold\n(validation scenario bounds)"
    ),
    "extent_product": "Extent product\n(GPD / GFW)",
    "drainage_distance": "Drainage distance\n(250 / 750 m)",
}

_EXPECTED_FIGURE_13_BASELINE = 2.689369853477171
_EXPECTED_FIGURE_13_ENDPOINTS = {
    "emission_factors": (1.4970864689605823, 3.974086335837641),
    "extent_threshold": (1.3328539456591912, 3.078938639723742),
    "extent_product": (2.6417334212766637, 3.289116169690267),
    "drainage_distance": (2.5753522569346567, 2.758452351110751),
}
_EXPECTED_FIGURE_13_COMBINED = (0.6740910775276678, 4.5159091200562065)


def load_figure_13_controls(source: FrameSource) -> Figure13Controls:
    """Load and validate the frozen Figure 13 values table."""

    df = _read_frame(source)
    required = {
        "row_type",
        "key",
        "low_gt_co2e_yr",
        "baseline_gt_co2e_yr",
        "high_gt_co2e_yr",
    }
    _require_columns(df, required, "Figure 13")

    baseline_rows = df.loc[df["row_type"] == "baseline"]
    combined_rows = df.loc[df["row_type"] == "combined"]
    factor_rows = df.loc[df["row_type"] == "one_at_a_time"]
    if len(baseline_rows) != 1 or len(combined_rows) != 1:
        raise ValueError("Figure 13 requires exactly one baseline and combined row")

    baseline = float(baseline_rows.iloc[0]["baseline_gt_co2e_yr"])
    combined_low = float(combined_rows.iloc[0]["low_gt_co2e_yr"])
    combined_high = float(combined_rows.iloc[0]["high_gt_co2e_yr"])

    factor_by_key = {str(row["key"]): row for _, row in factor_rows.iterrows()}
    if set(factor_by_key) != set(_FIGURE_13_LABELS):
        raise ValueError(
            "Figure 13 factor keys differ from the approved frozen controls"
        )
    factors = tuple(
        SensitivityFactor(
            key=key,
            label=_FIGURE_13_LABELS[key],
            low=float(factor_by_key[key]["low_gt_co2e_yr"]),
            high=float(factor_by_key[key]["high_gt_co2e_yr"]),
        )
        for key in _FIGURE_13_LABELS
    )
    controls = Figure13Controls(
        baseline=baseline,
        combined_low=combined_low,
        combined_high=combined_high,
        factors=factors,
    )
    _validate_figure_13_controls(controls)
    return controls


def _validate_figure_13_controls(controls: Figure13Controls) -> None:
    tolerance = 1e-12
    if abs(controls.baseline - _EXPECTED_FIGURE_13_BASELINE) > tolerance:
        raise ValueError(f"Unexpected corrected Figure 13 baseline: {controls.baseline}")
    if (
        abs(controls.combined_low - _EXPECTED_FIGURE_13_COMBINED[0]) > tolerance
        or abs(controls.combined_high - _EXPECTED_FIGURE_13_COMBINED[1])
        > tolerance
    ):
        raise ValueError("Unexpected Figure 13 combined sensitivity endpoints")
    for factor in controls.factors:
        expected_low, expected_high = _EXPECTED_FIGURE_13_ENDPOINTS[factor.key]
        if (
            abs(factor.low - expected_low) > tolerance
            or abs(factor.high - expected_high) > tolerance
        ):
            raise ValueError(f"Unexpected Figure 13 endpoints for {factor.key}")
        if not factor.low < controls.baseline < factor.high:
            raise ValueError(f"Figure 13 baseline is outside {factor.key} endpoints")
    swings = [factor.swing for factor in controls.factors]
    if swings != sorted(swings, reverse=True):
        raise ValueError("Figure 13 factors are not ordered by descending swing")


def _draw_figure_13_split(
    ax: plt.Axes,
    baseline: float,
    y: float,
    low: float,
    high: float,
    height: float,
    *,
    edge: str | None = None,
) -> None:
    low_color, high_color = "#9ca3af", "#3E3753"
    ax.barh(
        y,
        baseline - low,
        left=low,
        color=low_color,
        height=height,
        zorder=3,
        edgecolor=edge,
        linewidth=0.9 if edge else 0,
    )
    ax.barh(
        y,
        high - baseline,
        left=baseline,
        color=high_color,
        height=height,
        zorder=3,
        edgecolor=edge,
        linewidth=0.9 if edge else 0,
    )
    ax.text(low - 0.05, y, f"{low:.2f}", va="center", ha="right", fontsize=9.5)
    ax.text(high + 0.05, y, f"{high:.2f}", va="center", ha="left", fontsize=9.5)


def build_figure_13(source: FrameSource) -> plt.Figure:
    """Build corrected Figure 13 from the frozen, validated values table."""

    controls = load_figure_13_controls(source)
    baseline = controls.baseline
    combined = (controls.combined_low, controls.combined_high)
    x_right, separator_x, swing_x = 6.0, 5.0, 5.5
    combined_y = 0.0
    factor_y = [1.0, 2.0, 3.0, 4.0]

    with pc.use_theme(pc.THEME_LIGHT_GRID):
        fig, ax = plt.subplots(figsize=(9.0, 5.4))
        ax.axhspan(
            combined_y - 0.5,
            combined_y + 0.5,
            color="#f1f2f4",
            zorder=0,
        )
        combined_separator = ax.axhline(
            combined_y + 0.7, color="#cfd3d8", lw=1.0, zorder=1
        )
        combined_separator.set_gid("decorative")

        _draw_figure_13_split(
            ax,
            baseline,
            combined_y,
            combined[0],
            combined[1],
            0.5,
            edge="black",
        )
        ax.text(
            swing_x,
            combined_y,
            f"{combined[1] - combined[0]:.2f}",
            ha="center",
            va="center",
            fontsize=9.5,
        )
        ax.text(
            (combined[0] + combined[1]) / 2,
            combined_y + 0.42,
            "emission-factor and extent-threshold bounds applied together",
            ha="center",
            va="top",
            fontsize=8,
            style="italic",
            color="#555555",
        )

        for y, factor in zip(factor_y, controls.factors, strict=True):
            _draw_figure_13_split(
                ax, baseline, y, factor.low, factor.high, 0.5
            )
            ax.text(
                swing_x,
                y,
                f"{factor.swing:.2f}",
                ha="center",
                va="center",
                fontsize=9.5,
            )

        # The dashed baseline is a meaningful data line, so it follows the
        # Frontiers 2-point minimum while decorative separators remain thinner.
        ax.axvline(baseline, color="black", ls="--", lw=2.0, zorder=4)
        ax.text(
            baseline,
            -0.95,
            f"baseline\n{baseline:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"),
            zorder=5,
        )
        swing_separator = ax.axvline(separator_x, color="#cfd3d8", lw=1.0, zorder=1)
        swing_separator.set_gid("decorative")
        ax.text(
            swing_x,
            -0.95,
            "Swing\n(Gt CO$_2$e yr$^{-1}$)",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#444444",
        )

        ax.set_yticks([combined_y] + factor_y)
        ax.set_yticklabels(
            ["Combined scenario\n(EF + extent threshold)"]
            + [factor.label for factor in controls.factors]
        )
        ax.set_ylim(max(factor_y) + 1.1, -1.4)
        ax.set_xlim(0, x_right)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xlabel(
            "Total emissions from disturbed organic soils (Gt CO$_2$e/year)"
        )
        pc.tidy_axes(ax, grid="x")
        ax.spines["left"].set_visible(False)

        handles = [
            plt.Rectangle((0, 0), 1, 1, color="#9ca3af"),
            plt.Rectangle((0, 0), 1, 1, color="#3E3753"),
        ]
        ax.legend(
            handles,
            [
                "Lower estimate (below baseline)",
                "Higher estimate (above baseline)",
            ],
            ncol=2,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            frameon=False,
            fontsize=9,
        )
        fig.tight_layout()
    return fig


PILOT_BUILDERS = {
    "Figure_01a": build_figure_01a,
    "Figure_02b": build_figure_02b,
    "Figure_09d": build_figure_09d,
    "Figure_10a": build_figure_10a,
    "Figure_10b": build_figure_10b,
    "Figure_11a": build_figure_11a,
    "Figure_13": build_figure_13,
}
