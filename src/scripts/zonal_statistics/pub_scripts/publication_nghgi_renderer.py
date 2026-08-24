# -*- coding: utf-8 -*-
"""Offline adapter for the tracked NGHGI publication renderer.

The canonical renderer lives in the Organic-Soils publication repository and
normally writes PNGs.  This adapter loads that tracked implementation, captures
the Matplotlib figure before any raster is written, and returns it to the
format-neutral publication exporter.  The plotting code itself is not copied
or reimplemented here.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_RENDERER_SHA256 = (
    "5941fa9f1dc8750aedb9a545f81f0b530215bc09e1ceb7d1f0aea14e781bf997"
)
EXPECTED_COUNTRY_ORDER = (
    "IDN",
    "SWE",
    "MYS",
    "FIN",
    "RUS",
    "DEU",
    "USA",
    "NOR",
    "CAN",
    "IRL",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_renderer(path: Path) -> ModuleType:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Tracked NGHGI renderer not found: {path}")
    observed_hash = _sha256(path)
    if observed_hash != EXPECTED_RENDERER_SHA256:
        raise ValueError(
            "Tracked NGHGI renderer hash differs from the approved frozen "
            f"lineage: {observed_hash}"
        )
    spec = importlib.util.spec_from_file_location(
        "organic_soils_tracked_nghgi_renderer", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import tracked NGHGI renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture_saved_figure(callback) -> plt.Figure:
    captured: list[plt.Figure] = []

    def capture_savefig(figure, *_args, **_kwargs):
        captured.append(figure)

    with patch.object(matplotlib.figure.Figure, "savefig", capture_savefig):
        callback()
    if len(captured) != 1:
        raise RuntimeError(
            f"Expected one figure from tracked renderer, captured {len(captured)}"
        )
    return captured[0]


def _apply_frontiers_artist_minima(fig: plt.Figure) -> None:
    for text_artist in fig.findobj(match=plt.Text):
        if text_artist.get_visible() and text_artist.get_text().strip():
            if float(text_artist.get_fontsize()) < 8.0:
                text_artist.set_fontsize(8.0)

    # The horizontal dumbbell connectors encode model-vs-NGHGI differences.
    # Grid lines are managed by the axes and are intentionally excluded.
    for axis in fig.axes:
        grid_lines = set(axis.get_xgridlines()) | set(axis.get_ygridlines())
        for line in axis.lines:
            if line not in grid_lines and line.get_linestyle() not in (
                None,
                "",
                "None",
                "none",
                " ",
            ):
                line.set_linewidth(max(2.0, float(line.get_linewidth())))
                line.set_gid("meaningful")


def build_figure_15(
    data_root: Path,
    renderer_path: Path,
    *,
    endpoint: int = 2024,
) -> tuple[plt.Figure, pd.DataFrame, tuple[str, ...]]:
    """Return Figure 15, its plotted evidence, and the fixed country order."""

    data_root = data_root.resolve()
    renderer = _load_renderer(renderer_path)
    tables, order, evidence = renderer._load_main(data_root, endpoint)
    observed_order = tuple(order)
    if observed_order != EXPECTED_COUNTRY_ORDER:
        raise ValueError(
            "NGHGI country order differs from the approved corrected order: "
            f"{observed_order}"
        )

    figure = _capture_saved_figure(
        lambda: renderer._plot_main(tables, order, Path("captured_figure_15.png"))
    )
    _apply_frontiers_artist_minima(figure)
    return figure, evidence.copy(), observed_order


def validate_frozen_ratio_evidence(
    generated: pd.DataFrame,
    ratio_csv: Path,
    *,
    endpoint: int = 2024,
) -> None:
    """Assert the renderer evidence matches the frozen Figure 15 ratio table."""

    expected = pd.read_csv(ratio_csv)
    expected = expected.loc[
        (expected["figure"] == "Figure 15")
        & (expected["endpoint"].astype(int) == endpoint)
    ].copy()
    keys = ["figure", "endpoint", "order", "iso3", "metric", "model_scope", "unit"]
    values = ["model_value", "nghgi_value", "ratio_model_over_nghgi"]
    left = generated.sort_values(keys).reset_index(drop=True)
    right = expected.sort_values(keys).reset_index(drop=True)
    if len(left) != len(right) or not left[keys].equals(right[keys]):
        raise ValueError("Generated Figure 15 evidence keys differ from frozen ratios")
    for column in values:
        difference = (left[column].astype(float) - right[column].astype(float)).abs()
        tolerance = 1e-8 * right[column].astype(float).abs().clip(lower=1.0)
        if bool((difference > tolerance).any()):
            raise ValueError(
                f"Generated Figure 15 evidence differs in {column}"
            )
