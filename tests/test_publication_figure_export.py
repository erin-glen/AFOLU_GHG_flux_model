from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageCms
import pytest

from src.scripts.zonal_statistics.pub_scripts import publication_figure_export as pfe


def _compliant_figure():
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.set_xlabel("Value", fontsize=8)
    ax.set_ylabel("Group", fontsize=8)
    ax.tick_params(labelsize=8)
    line = ax.plot([0, 1], [0, 1], linewidth=2.0)[0]
    line.set_gid("meaningful")
    fig.subplots_adjust(left=0.20, right=0.95, bottom=0.25, top=0.92)
    return fig


def test_tiff_export_is_exact_rgb_lzw_srgb_and_restores_figure(tmp_path: Path):
    fig = _compliant_figure()
    original_size = tuple(fig.get_size_inches())
    original_dpi = fig.dpi
    original_canvas = fig.canvas
    output = tmp_path / "Figure_01a.tif"
    repeat_output = tmp_path / "Figure_01a_repeat.tif"
    try:
        dimensions = pfe.save_publication_figure(
            fig,
            str(output),
            output_format="tif",
            dpi=300,
            target_width_mm=85,
            aspect_ratio=2.0,
        )
        assert dimensions == (1004, 502)
        assert tuple(fig.get_size_inches()) == pytest.approx(original_size)
        assert fig.dpi == original_dpi
        assert fig.canvas is original_canvas
        repeated_dimensions = pfe.save_publication_figure(
            fig,
            str(repeat_output),
            output_format="tif",
            dpi=300,
            target_width_mm=85,
            aspect_ratio=2.0,
        )
        assert repeated_dimensions == dimensions
        assert output.read_bytes() == repeat_output.read_bytes()
    finally:
        plt.close(fig)

    with Image.open(output) as image:
        assert image.format == "TIFF"
        assert image.mode == "RGB"
        assert image.size == (1004, 502)
        assert image.tag_v2.get(259) == 5
        assert image.info["dpi"] == pytest.approx((300, 300), abs=0.1)
        icc = image.info.get("icc_profile")
        assert isinstance(icc, bytes)
        assert len(icc) >= 128
        assert icc[24:36] == pfe._FIXED_ICC_TIMESTAMP
        assert icc[36:40] == b"acsp"
        try:
            description = ImageCms.getProfileDescription(
                ImageCms.getOpenProfile(io.BytesIO(icc))
            ).casefold()
        except ImportError:
            description = ""
        assert "srgb" in description or b"srgb" in icc.lower()
        assert image.getpixel((0, 0)) == (255, 255, 255)


def test_artist_validation_fails_small_text_and_meaningful_line():
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "small", fontsize=7.9)
    ax.plot([0, 1], [0, 1], linewidth=1.9)
    try:
        issues = pfe.publication_figure_validation_issues(fig)
        assert any("small" in issue for issue in issues)
        assert any("meaningful line" in issue for issue in issues)
        with pytest.raises(ValueError, match="Publication figure validation failed"):
            pfe.validate_publication_figure(fig)
    finally:
        plt.close(fig)


def test_artist_validation_exempts_hidden_marker_only_and_decorative_lines():
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "hidden", fontsize=4, visible=False)
    ax.plot([0, 1], [0, 1], linestyle="none", marker="o", linewidth=0.2)
    decorative = ax.axvline(0.5, linewidth=0.5)
    decorative.set_gid("decorative")
    try:
        assert pfe.publication_figure_validation_issues(fig) == []
    finally:
        plt.close(fig)


def test_extent_validation_rejects_active_label_off_canvas() -> None:
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    ax.set_xlabel("Clipped label", fontsize=8)
    ax.set_position((0.15, -0.08, 0.8, 0.8))
    try:
        issues = pfe.publication_figure_extent_issues(fig)
        assert any("Clipped label" in issue for issue in issues)
    finally:
        plt.close(fig)


def test_extent_validation_ignores_locator_ticks_outside_view() -> None:
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    ax.set_xticks([0.0, 0.5, 1.0, 2.0])
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.25, top=0.9)
    try:
        assert pfe.publication_figure_extent_issues(fig) == []
    finally:
        plt.close(fig)

def test_custom_hook_and_bad_arguments(tmp_path: Path):
    fig = _compliant_figure()
    called = []

    def hook(actual):
        called.append(actual)
        return []

    try:
        pfe.save_publication_figure(
            fig,
            str(tmp_path / "ok.tif"),
            target_width_mm=85,
            aspect_ratio=2,
            validation_hooks=(hook,),
        )
        assert called == [fig]
        with pytest.raises(ValueError, match="requires target_width_mm"):
            pfe.save_publication_figure(fig, str(tmp_path / "bad.tif"))
        with pytest.raises(ValueError, match="uses target_width_mm"):
            pfe.save_publication_figure(
                fig,
                str(tmp_path / "bad2.tif"),
                target_width_mm=85,
                aspect_ratio=2,
                width=3,
            )
        with pytest.raises(ValueError, match="PNG export uses"):
            pfe.save_publication_figure(
                fig,
                str(tmp_path / "bad.png"),
                target_width_mm=85,
                aspect_ratio=2,
            )
    finally:
        plt.close(fig)
