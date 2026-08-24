from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import pandas as pd
import pytest

from src.scripts.zonal_statistics.pub_scripts import publication_figure_renderers as pfr
from src.scripts.zonal_statistics.pub_scripts import pub_common as pc


def test_figure_02b_retains_stacked_layout() -> None:
    df = pd.DataFrame(
        {
            "Climate": ["Boreal", "Boreal", "Temperate", "Temperate", "Tropical", "Tropical"],
            "Component": ["Drained", "Burned"] * 3,
            "intensity_tCO2e_per_ha_yr": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    fig = pfr.build_figure_02b(df)
    try:
        ax = fig.axes[0]
        assert len(ax.patches) == 6
        # Each climate has one stacked column: burned starts above drained.
        assert [round(patch.get_y(), 8) for patch in ax.patches] == [
            0.0,
            0.0,
            0.0,
            1.0,
            3.0,
            5.0,
        ]
        centers = sorted(round(patch.get_x() + patch.get_width() / 2, 2) for patch in ax.patches)
        assert centers == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "Drained",
            "Burned",
        ]
    finally:
        plt.close(fig)


def test_figure_09d_uses_three_decimal_labels() -> None:
    df = pd.DataFrame(
        {
            "iso3_or_code": ["AAA", "BBB"],
            "burned_avg_GtCO2e_per_yr": [0.234651, 0.000587],
        }
    )

    fig = pfr.build_figure_09d(df)
    try:
        assert [text.get_text() for text in fig.axes[0].texts] == ["0.235", "0.001"]
    finally:
        plt.close(fig)


def _figure_13_frame() -> pd.DataFrame:
    rows = [
        {
            "row_type": "baseline",
            "key": "corrected_ogh_2021_2024",
            "low_gt_co2e_yr": None,
            "baseline_gt_co2e_yr": 2.689369853477171,
            "high_gt_co2e_yr": None,
        },
        {
            "row_type": "combined",
            "key": "ef_plus_extent_threshold",
            "low_gt_co2e_yr": 0.6740910775276678,
            "baseline_gt_co2e_yr": 2.689369853477171,
            "high_gt_co2e_yr": 4.5159091200562065,
        },
    ]
    endpoints = {
        "emission_factors": (1.4970864689605823, 3.974086335837641),
        "extent_threshold": (1.3328539456591912, 3.078938639723742),
        "extent_product": (2.6417334212766637, 3.289116169690267),
        "drainage_distance": (2.5753522569346567, 2.758452351110751),
    }
    rows.extend(
        {
            "row_type": "one_at_a_time",
            "key": key,
            "low_gt_co2e_yr": low,
            "baseline_gt_co2e_yr": 2.689369853477171,
            "high_gt_co2e_yr": high,
        }
        for key, (low, high) in endpoints.items()
    )
    return pd.DataFrame(rows)


def test_figure_13_controls_and_minimum_text_and_baseline_line() -> None:
    controls = pfr.load_figure_13_controls(_figure_13_frame())
    assert controls.baseline == pytest.approx(2.689369853477171, abs=1e-15)
    assert controls.factors[0].key == "emission_factors"
    assert [factor.swing for factor in controls.factors] == sorted(
        (factor.swing for factor in controls.factors), reverse=True
    )

    fig = pfr.build_figure_13(_figure_13_frame())
    try:
        ax = fig.axes[0]
        visible_sizes = [text.get_fontsize() for text in ax.texts if text.get_visible()]
        visible_sizes.extend(label.get_fontsize() for label in ax.get_xticklabels())
        visible_sizes.extend(label.get_fontsize() for label in ax.get_yticklabels())
        assert min(visible_sizes) >= 8.0
        baseline_lines = [
            line
            for line in ax.lines
            if len(line.get_xdata()) == 2
            and all(float(x) == pytest.approx(controls.baseline) for x in line.get_xdata())
        ]
        assert len(baseline_lines) == 1
        assert baseline_lines[0].get_linewidth() >= 2.0
    finally:
        plt.close(fig)


def test_figure_13_rejects_stale_baseline() -> None:
    df = _figure_13_frame()
    df.loc[df["row_type"] == "baseline", "baseline_gt_co2e_yr"] = 2.67
    with pytest.raises(ValueError, match="Unexpected corrected Figure 13 baseline"):
        pfr.load_figure_13_controls(df)


def test_figure_10b_uses_canonical_dark2_palette_without_global_mutation() -> None:
    original_global = pc.CLIMATE_COLORS.copy()
    df = pd.DataFrame(
        {
            "LandUse": ["Undrained"] * 3,
            "Climate": ["Boreal", "Temperate", "Tropical"],
            "burned_avg_GtCO2e_per_yr": [1.0, 2.0, 3.0],
        }
    )
    fig = pfr.build_figure_10b(df)
    try:
        observed = [
            to_hex(container.patches[0].get_facecolor())
            for container in fig.axes[0].containers
        ]
        expected = [
            pc.CLIMATE_PALETTES["brewer_dark2"][climate].lower()
            for climate in pc.CLIMATE_ORDER
        ]
        assert observed == expected
        assert pc.CLIMATE_COLORS == original_global
    finally:
        plt.close(fig)


def test_figure_10a_uses_publication_label_without_changing_values() -> None:
    df = pd.DataFrame(
        {
            "LandUse": ["Otherland"] * 3,
            "Climate": ["Boreal", "Temperate", "Tropical"],
            "drained_avg_GtCO2e_per_yr": [1.0, 2.0, 3.0],
        }
    )

    fig = pfr.build_figure_10a(df)
    try:
        labels = {label.get_text() for label in fig.axes[0].get_yticklabels()}
        assert labels == {"Other land"}
        assert sum(patch.get_width() for patch in fig.axes[0].patches) == pytest.approx(
            df["drained_avg_GtCO2e_per_yr"].sum()
        )
    finally:
        plt.close(fig)


def test_figure_13_uses_publication_units_and_tier_typography() -> None:
    fig = pfr.build_figure_13(_figure_13_frame())
    try:
        ax = fig.axes[0]
        ylabels = [label.get_text() for label in ax.get_yticklabels()]
        assert any("Tier 1" in label for label in ylabels)
        assert all("Tier-1" not in label for label in ylabels)
        assert "Swing\n(Gt CO$_2$e yr$^{-1}$)" in [
            text.get_text() for text in ax.texts
        ]
    finally:
        plt.close(fig)


def test_figure_10b_merges_generic_drained_into_drained_other() -> None:
    df = pd.DataFrame(
        {
            "LandUse": [
                "Undrained",
                "Drained Other",
                "Drained Crop Or Plantation",
                "Drained",
                "Drained",
            ],
            "Climate": ["Boreal", "Tropical", "Tropical", "Boreal", "Temperate"],
            "burned_avg_GtCO2e_per_yr": [0.40, 0.08, 0.04, 0.007, 0.010],
        }
    )

    fig = pfr.build_figure_10b(df)
    try:
        labels = {label.get_text() for label in fig.axes[0].get_yticklabels()}
        assert labels == {
            "Undrained",
            "Drained Other",
            "Drained Crop Or Plantation",
        }
        assert "Drained" not in labels
        assert sum(patch.get_width() for patch in fig.axes[0].patches) == pytest.approx(
            df["burned_avg_GtCO2e_per_yr"].sum()
        )
    finally:
        plt.close(fig)


def test_figure_11a_routes_through_existing_comparison_renderer(monkeypatch) -> None:
    from src.scripts.zonal_statistics.pub_scripts import pub_compare_runs as pcr

    sentinel = plt.figure()
    observed: dict[str, object] = {}

    def fake_plot(df, component_order, component_colors, xlabel, title):
        observed.update(
            df=df,
            component_order=component_order,
            component_colors=component_colors,
            xlabel=xlabel,
            title=title,
        )
        return sentinel

    monkeypatch.setattr(pcr, "_plot_horizontal_stack", fake_plot)
    source = pd.DataFrame(
        {"Run": ["OGH"], "Drained": [1.0], "Undrained": [2.0]}
    )
    try:
        assert pfr.build_figure_11a(source) is sentinel
        assert observed["component_order"] == ("Drained", "Undrained")
        assert observed["xlabel"] == "Mapped area (million ha)"
        assert observed["df"].equals(source)
    finally:
        plt.close(sentinel)


@pytest.mark.parametrize(
    ("builder", "columns"),
    [
        (pfr.build_figure_01a, {"Inventory Period", "Component"}),
        (pfr.build_figure_02b, {"Climate", "Component"}),
        (pfr.build_figure_09d, {"iso3_or_code"}),
        (pfr.build_figure_10a, {"LandUse", "Climate"}),
        (pfr.build_figure_10b, {"LandUse", "Climate"}),
        (pfr.build_figure_11a, {"Run", "Drained"}),
    ],
)
def test_builders_reject_missing_columns(builder, columns) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        builder(pd.DataFrame(columns=sorted(columns)))
