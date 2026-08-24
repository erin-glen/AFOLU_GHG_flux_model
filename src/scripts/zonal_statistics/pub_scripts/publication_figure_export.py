# -*- coding: utf-8 -*-
"""Format-neutral, offline export helpers for publication figures.

Legacy PNG behavior remains in :mod:`pub_common`.  This companion module owns
the strict final-size TIFF path so existing publication scripts are unaffected
until they opt in.  TIFFs are rendered from the Matplotlib Agg canvas, never
resampled from a prior PNG.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Callable, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.legend import Legend
from PIL import Image, ImageCms

from src.scripts.zonal_statistics.pub_scripts import pub_common as pc


ValidationHook = Callable[[plt.Figure], Sequence[str] | None]

_FIXED_ICC_TIMESTAMP = struct.pack(">6H", 2000, 1, 1, 0, 0, 0)


def _deterministic_icc_profile(profile_bytes: bytes) -> bytes:
    """Normalize the ICC creation timestamp so exports are byte-reproducible."""

    if len(profile_bytes) < 128 or profile_bytes[36:40] != b"acsp":
        raise ValueError("Invalid ICC profile")
    normalized = bytearray(profile_bytes)
    normalized[24:36] = _FIXED_ICC_TIMESTAMP
    return bytes(normalized)


def _srgb_profile_bytes() -> bytes:
    """Return a complete sRGB ICC profile for publication raster exports."""

    try:
        profile = ImageCms.createProfile("sRGB")
        return _deterministic_icc_profile(
            ImageCms.ImageCmsProfile(profile).tobytes()
        )
    except (AttributeError, ImportError, OSError):
        # Some Windows Pillow builds omit LittleCMS.  Embedding the operating
        # system's standard profile is equivalent and avoids silently writing
        # an untagged RGB TIFF.
        windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        candidates = (
            windir
            / "System32"
            / "spool"
            / "drivers"
            / "color"
            / "sRGB Color Space Profile.icm",
            Path("/usr/share/color/icc/colord/sRGB.icc"),
            Path("/usr/share/color/icc/sRGB.icc"),
        )
        for candidate in candidates:
            if candidate.is_file():
                profile_bytes = candidate.read_bytes()
                if (
                    len(profile_bytes) >= 128
                    and profile_bytes[36:40] == b"acsp"
                    and b"srgb" in profile_bytes.lower()
                ):
                    return _deterministic_icc_profile(profile_bytes)
        raise RuntimeError(
            "An sRGB ICC profile is required for publication TIFF export"
        )


def publication_figure_validation_issues(
    fig: plt.Figure,
    *,
    min_text_pt: float = 8.0,
    min_line_pt: float = 2.0,
) -> list[str]:
    """Return visible-text and meaningful-line compliance issues.

    Grid lines, marker-only artists, and lines explicitly tagged with
    ``gid='decorative'`` are not treated as meaningful data lines.  This keeps
    the two-point rule focused on data encodings and connectors.
    """

    issues: list[str] = []
    for text_artist in fig.findobj(mpl.text.Text):
        if not text_artist.get_visible() or not text_artist.get_text().strip():
            continue
        size = float(text_artist.get_fontsize())
        if size + 1e-9 < min_text_pt:
            issues.append(
                f"visible text {text_artist.get_text()!r} is {size:g} pt "
                f"(< {min_text_pt:g} pt)"
            )

    for axis_index, axis in enumerate(fig.axes):
        grid_lines = set(axis.get_xgridlines()) | set(axis.get_ygridlines())
        for line_index, line in enumerate(axis.lines):
            if not line.get_visible() or line in grid_lines:
                continue
            if line.get_gid() == "decorative":
                continue
            linestyle = line.get_linestyle()
            if linestyle in (None, "", "None", "none", " "):
                continue
            if len(line.get_xdata()) < 2 or len(line.get_ydata()) < 2:
                continue
            linewidth = float(line.get_linewidth())
            if linewidth + 1e-9 < min_line_pt:
                issues.append(
                    f"meaningful line axes[{axis_index}].lines[{line_index}] "
                    f"is {linewidth:g} pt (< {min_line_pt:g} pt)"
                )
    return issues


def validate_publication_figure(
    fig: plt.Figure,
    *,
    min_text_pt: float = 8.0,
    min_line_pt: float = 2.0,
    validation_hooks: Sequence[ValidationHook] = (),
) -> None:
    """Raise when a figure violates artist-level publication requirements."""

    issues = publication_figure_validation_issues(
        fig,
        min_text_pt=min_text_pt,
        min_line_pt=min_line_pt,
    )
    for hook in validation_hooks:
        hook_issues = hook(fig)
        if hook_issues:
            issues.extend(str(issue) for issue in hook_issues)
    if issues:
        raise ValueError("Publication figure validation failed: " + "; ".join(issues))


def _active_tick_label_ids(fig: plt.Figure) -> tuple[set[int], set[int]]:
    """Return IDs for draw-active and merely allocated tick labels."""

    active: set[int] = set()
    allocated: set[int] = set()
    for axes in fig.axes:
        for coordinate_axis in (axes.xaxis, axes.yaxis):
            for tick in (*coordinate_axis.majorTicks, *coordinate_axis.minorTicks):
                allocated.update((id(tick.label1), id(tick.label2)))
            # Mirrors Axis.draw's mathematical view filtering; locator-created
            # labels outside the displayed interval are not drawn content.
            for tick in coordinate_axis._update_ticks():
                active.update((id(tick.label1), id(tick.label2)))
    return active, allocated


def _extent_outside_canvas(
    bounds,
    *,
    canvas_width: int,
    canvas_height: int,
    tolerance_px: float,
) -> bool:
    x0, y0, width, height = bounds
    return (
        x0 < -tolerance_px
        or y0 < -tolerance_px
        or x0 + width > canvas_width + tolerance_px
        or y0 + height > canvas_height + tolerance_px
    )


def _publication_figure_extent_issues_on_canvas(
    fig: plt.Figure,
    *,
    renderer,
    canvas_size: tuple[int, int],
    tolerance_px: float = 0.5,
) -> list[str]:
    if tolerance_px < 0:
        raise ValueError("tolerance_px must be non-negative")
    canvas_width, canvas_height = canvas_size
    active_ticks, allocated_ticks = _active_tick_label_ids(fig)
    issues: list[str] = []
    for text_artist in fig.findobj(mpl.text.Text):
        if not text_artist.get_visible() or not text_artist.get_text().strip():
            continue
        artist_id = id(text_artist)
        if artist_id in allocated_ticks and artist_id not in active_ticks:
            continue
        bounds = tuple(
            float(value)
            for value in text_artist.get_window_extent(renderer).bounds
        )
        if _extent_outside_canvas(
            bounds,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            tolerance_px=tolerance_px,
        ):
            issues.append(
                f"visible text {text_artist.get_text()!r} extends off canvas: "
                f"bounds={bounds}, canvas={(canvas_width, canvas_height)}"
            )
    for legend in fig.findobj(Legend):
        if not legend.get_visible():
            continue
        bounds = tuple(
            float(value) for value in legend.get_window_extent(renderer).bounds
        )
        if _extent_outside_canvas(
            bounds,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            tolerance_px=tolerance_px,
        ):
            issues.append(
                "visible legend extends off canvas: "
                f"bounds={bounds}, canvas={(canvas_width, canvas_height)}"
            )
    return issues


def publication_figure_extent_issues(
    fig: plt.Figure,
    *,
    tolerance_px: float = 0.5,
) -> list[str]:
    """Draw through Agg and report active labels or legends off canvas."""

    original_canvas = fig.canvas
    try:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        return _publication_figure_extent_issues_on_canvas(
            fig,
            renderer=canvas.get_renderer(),
            canvas_size=canvas.get_width_height(),
            tolerance_px=tolerance_px,
        )
    finally:
        fig.set_canvas(original_canvas)

def _save_publication_tiff(
    fig: plt.Figure,
    path: str,
    *,
    dpi: int,
    target_width_mm: float,
    aspect_ratio: float,
    validate_extents: bool,
) -> tuple[int, int]:
    """Render *fig* through Agg directly to an RGB8, LZW sRGB TIFF."""

    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if target_width_mm <= 0:
        raise ValueError("target_width_mm must be positive")
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be positive")

    width_px = int(round(target_width_mm / 25.4 * dpi))
    height_px = int(round(width_px / aspect_ratio))
    if width_px < 1 or height_px < 1:
        raise ValueError("Publication TIFF dimensions must be at least one pixel")

    original_size = tuple(float(value) for value in fig.get_size_inches())
    original_dpi = float(fig.dpi)
    original_facecolor = fig.patch.get_facecolor()
    original_alpha = fig.patch.get_alpha()
    original_canvas = fig.canvas
    try:
        fig.set_size_inches(width_px / dpi, height_px / dpi, forward=False)
        fig.set_dpi(dpi)
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        if validate_extents:
            extent_issues = _publication_figure_extent_issues_on_canvas(
                fig,
                renderer=canvas.get_renderer(),
                canvas_size=canvas.get_width_height(),
            )
            if extent_issues:
                raise ValueError(
                    "Publication figure extent validation failed: "
                    + "; ".join(extent_issues)
                )
        rgba = Image.frombuffer(
            "RGBA",
            canvas.get_width_height(),
            canvas.buffer_rgba(),
            "raw",
            "RGBA",
            0,
            1,
        ).copy()
        if rgba.size != (width_px, height_px):
            raise RuntimeError(
                "Agg rendered unexpected dimensions: "
                f"expected {(width_px, height_px)}, observed {rgba.size}"
            )
        rgb = Image.new("RGB", rgba.size, "white")
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        pc._ensure_parent_dir_local(path)
        rgb.save(
            path,
            format="TIFF",
            compression="tiff_lzw",
            dpi=(dpi, dpi),
            icc_profile=_srgb_profile_bytes(),
        )
    finally:
        fig.set_size_inches(*original_size, forward=False)
        fig.set_dpi(original_dpi)
        fig.patch.set_facecolor(original_facecolor)
        fig.patch.set_alpha(original_alpha)
        fig.set_canvas(original_canvas)
    return width_px, height_px


def save_publication_figure(
    fig: plt.Figure,
    path: str,
    *,
    output_format: Optional[str] = None,
    dpi: int = 300,
    width: Optional[float] = None,
    height: Optional[float] = None,
    target_width_mm: Optional[float] = None,
    aspect_ratio: Optional[float] = None,
    validate: bool = True,
    min_text_pt: float = 8.0,
    min_line_pt: float = 2.0,
    validation_hooks: Sequence[ValidationHook] = (),
) -> Optional[tuple[int, int]]:
    """Save a publication figure without changing legacy PNG behavior.

    TIFF output is rendered from the Matplotlib Agg canvas at the requested
    physical width and oracle aspect ratio.  PNG requests delegate to the
    unchanged legacy helper in :mod:`pub_common`.
    """

    fmt = (output_format or Path(path).suffix.lstrip(".")).lower()
    if fmt in {"tif", "tiff"}:
        if width is not None or height is not None:
            raise ValueError("TIFF export uses target_width_mm and aspect_ratio")
        if target_width_mm is None or aspect_ratio is None:
            raise ValueError(
                "TIFF export requires target_width_mm and aspect_ratio"
            )
        if validate:
            validate_publication_figure(
                fig,
                min_text_pt=min_text_pt,
                min_line_pt=min_line_pt,
                validation_hooks=validation_hooks,
            )
        return _save_publication_tiff(
            fig,
            path,
            dpi=dpi,
            target_width_mm=target_width_mm,
            aspect_ratio=aspect_ratio,
            validate_extents=validate,
        )
    if fmt == "png":
        if target_width_mm is not None or aspect_ratio is not None:
            raise ValueError("PNG export uses width and height in inches")
        pc._save_png(fig, path, dpi=dpi, width=width, height=height)
        return None
    raise ValueError(f"Unsupported publication figure format: {fmt!r}")
