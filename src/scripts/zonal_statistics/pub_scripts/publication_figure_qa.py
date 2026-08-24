"""Offline QA for publication TIFFs against approved PNG appearance oracles.

This module intentionally separates automated file/geometry checks from the
required human visual gate.  A successful run is therefore reported as
``PENDING_VISUAL_APPROVAL`` until the generated comparison sheets have been
inspected; there is no command-line or API option for overriding that state.

The checks are image-only and offline.  They do not access model outputs,
remote object stores, LaTeX, or publisher PDFs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont


VISUAL_APPROVAL_STATUS = "PENDING_VISUAL_APPROVAL"
PASS = "PASS"
FAIL = "FAIL"
APPROVED_EXCEPTION = "APPROVED_EXCEPTION"
PASS_WITH_APPROVED_EXCEPTIONS = "PASS_WITH_APPROVED_EXCEPTIONS"
APPROVED_GEOMETRY_CHECKS = frozenset(
    {
        "normalized_nonwhite_bbox_matches_oracle",
        "normalized_color_encoding_bbox_matches_oracle",
        "normalized_dark_content_bbox_matches_oracle",
    }
)
GEOMETRY_REFLOW_EXCEPTION_POLICY = {
    "policy_id": "PUBFIG-GEOMETRY-REFLOW-20260806",
    "approved_by": "Erin Glen",
    "approved_date": "2026-08-06",
    "decision": (
        "Prioritize final-size text of at least 8 pt and meaningful lines of "
        "at least 2 pt at 85 or 180 mm; accept documented normalized geometry "
        "reflow relative to the approved PNG oracle."
    ),
    "eligible_checks": sorted(APPROVED_GEOMETRY_CHECKS),
    "eligibility": (
        "Only comparable oracle/generated masks with retained numeric side "
        "deltas are eligible. A mask or content class missing from one image "
        "remains a hard failure."
    ),
    "nonwaived_hard_gates": [
        "TIFF format and metadata",
        "exact core palette",
        "canvas and content containment",
        "visible text >= 8 pt",
        "meaningful lines >= 2 pt",
        "active text and legend extents",
        "numerical controls",
    ],
}


@dataclass(frozen=True)
class FigureQASpec:
    """Inputs and publication geometry for one figure asset."""

    asset_id: str
    oracle_path: Path
    tiff_path: Path
    target_width_px: int
    target_dpi: float = 300.0
    allow_approved_geometry_reflow: bool = False

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        oracle_root: Path | None = None,
        generated_root: Path | None = None,
    ) -> "FigureQASpec":
        oracle = Path(str(value["oracle_path"]))
        generated = Path(str(value["tiff_path"]))
        if oracle_root is not None and not oracle.is_absolute():
            oracle = oracle_root / oracle
        if generated_root is not None and not generated.is_absolute():
            generated = generated_root / generated
        allow_geometry = value.get("allow_approved_geometry_reflow", False)
        if not isinstance(allow_geometry, bool):
            raise ValueError(
                "allow_approved_geometry_reflow must be a boolean"
            )
        return cls(
            asset_id=str(value["asset_id"]),
            oracle_path=oracle,
            tiff_path=generated,
            target_width_px=int(value["target_width_px"]),
            target_dpi=float(value.get("target_dpi", 300.0)),
            allow_approved_geometry_reflow=allow_geometry,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten_rgb(image: Image.Image) -> Image.Image:
    """Return an RGB view, compositing transparency onto white when present."""

    if image.mode == "RGB":
        return image.copy()
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    return image.convert("RGB")


def _normalized_nonwhite_bbox(
    image: Image.Image, *, threshold: int = 250
) -> tuple[float, float, float, float] | None:
    """Return normalized bounds of pixels visibly different from white."""

    rgb = np.asarray(_flatten_rgb(image), dtype=np.uint8)
    mask = np.any(rgb < threshold, axis=2)
    if not bool(mask.any()):
        return None
    rows, columns = np.nonzero(mask)
    width, height = image.size
    # Right/bottom are exclusive so a full-canvas bbox normalizes to exactly 1.
    return (
        float(columns.min()) / width,
        float(rows.min()) / height,
        float(columns.max() + 1) / width,
        float(rows.max() + 1) / height,
    )


def _normalized_mask_bbox(
    mask: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """Return normalized bounds of a two-dimensional boolean mask."""

    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if not bool(mask.any()):
        return None
    rows, columns = np.nonzero(mask)
    height, width = mask.shape
    return (
        float(columns.min()) / width,
        float(rows.min()) / height,
        float(columns.max() + 1) / width,
        float(rows.max() + 1) / height,
    )


def _rgb_codes(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.uint32, copy=False)
    return (values[..., 0] << 16) | (values[..., 1] << 8) | values[..., 2]


def _decode_rgb_code(code: int) -> tuple[int, int, int]:
    return ((code >> 16) & 255, (code >> 8) & 255, code & 255)


def _core_palette(
    image: Image.Image,
    *,
    minimum_fraction: float = 0.001,
    minimum_pixels: int = 50,
) -> tuple[list[dict[str, Any]], set[int]]:
    """Identify stable solid encoding colors and every color present."""

    rgb = np.asarray(_flatten_rgb(image), dtype=np.uint8)
    unique, counts = np.unique(_rgb_codes(rgb), return_counts=True)
    total = int(rgb.shape[0] * rgb.shape[1])
    threshold = max(minimum_pixels, int(np.ceil(total * minimum_fraction)))
    records: list[dict[str, Any]] = []
    for raw_code, raw_count in zip(unique, counts, strict=True):
        code, count = int(raw_code), int(raw_count)
        color = _decode_rgb_code(code)
        chroma = max(color) - min(color)
        if (
            count < threshold
            or min(color) >= 235
            or max(color) <= 55
            or chroma < 8
        ):
            continue
        records.append(
            {"rgb": list(color), "count": count, "fraction": count / total}
        )
    records.sort(key=lambda item: (-int(item["count"]), item["rgb"]))
    return records, {int(value) for value in unique}


def _palette_mask(
    image: Image.Image,
    palette: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    rgb = np.asarray(_flatten_rgb(image), dtype=np.uint8)
    codes = _rgb_codes(rgb)
    palette_codes = {
        (int(item["rgb"][0]) << 16)
        | (int(item["rgb"][1]) << 8)
        | int(item["rgb"][2])
        for item in palette
    }
    if not palette_codes:
        return np.zeros(codes.shape, dtype=bool)
    return np.isin(codes, np.fromiter(palette_codes, dtype=np.uint32))


def _dark_content_mask(image: Image.Image) -> np.ndarray:
    """Mask neutral dark text, ticks, spines and line work, not chromatic fills."""

    rgb = np.asarray(_flatten_rgb(image), dtype=np.int16)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    return (rgb.max(axis=2) <= 145) & (chroma <= 8)


def _mask_geometry_evidence(
    oracle_mask: np.ndarray,
    generated_mask: np.ndarray,
    *,
    tolerance: float,
) -> tuple[bool, dict[str, Any]]:
    oracle_count = int(oracle_mask.sum())
    generated_count = int(generated_mask.sum())
    oracle_bbox = _normalized_mask_bbox(oracle_mask)
    generated_bbox = _normalized_mask_bbox(generated_mask)
    base = {
        "oracle_pixel_count": oracle_count,
        "generated_pixel_count": generated_count,
        "oracle": list(oracle_bbox) if oracle_bbox else None,
        "generated": list(generated_bbox) if generated_bbox else None,
        "tolerance_per_side": tolerance,
    }
    if oracle_bbox is None and generated_bbox is None:
        return True, {
            **base,
            "comparable": False,
            "reason": "mask_absent_in_both_images",
            "absolute_side_differences": None,
        }
    if oracle_bbox is None or generated_bbox is None:
        return False, {
            **base,
            "comparable": False,
            "reason": "mask_missing_from_one_image",
            "absolute_side_differences": None,
        }
    deltas = [
        abs(generated_side - oracle_side)
        for oracle_side, generated_side in zip(
            oracle_bbox, generated_bbox, strict=True
        )
    ]
    return all(delta <= tolerance for delta in deltas), {
        **base,
        "comparable": True,
        "reason": None,
        "absolute_side_differences": deltas,
        "maximum_side_difference": max(deltas),
    }


def _dpi_from_image(image: Image.Image) -> tuple[float | None, float | None]:
    dpi = image.info.get("dpi")
    if isinstance(dpi, (tuple, list)) and len(dpi) >= 2:
        try:
            return float(dpi[0]), float(dpi[1])
        except (TypeError, ValueError):
            pass

    # TIFF stores resolution as XResolution/YResolution and ResolutionUnit.
    tags = getattr(image, "tag_v2", {})
    try:
        x_resolution = float(tags.get(282))
        y_resolution = float(tags.get(283))
        unit = int(tags.get(296, 2))
    except (TypeError, ValueError):
        return None, None
    if unit == 3:  # pixels per centimetre
        x_resolution *= 2.54
        y_resolution *= 2.54
    if unit not in (2, 3):
        return None, None
    return x_resolution, y_resolution


def _icc_description(icc_bytes: bytes | None) -> str:
    if not icc_bytes:
        return ""
    try:
        profile = ImageCms.getOpenProfile(io.BytesIO(icc_bytes))
        return ImageCms.getProfileDescription(profile).strip()
    except (ImportError, OSError, TypeError, ValueError):
        return ""


def _compression_details(image: Image.Image) -> tuple[str, int | None]:
    name = str(image.info.get("compression", ""))
    raw_tag: int | None = None
    try:
        tag_value = image.tag_v2.get(259)
        raw_tag = int(tag_value) if tag_value is not None else None
    except (AttributeError, TypeError, ValueError):
        pass
    return name, raw_tag


def _white_background_metrics(
    image: Image.Image, *, threshold: int = 250, corner_size: int = 5
) -> dict[str, Any]:
    rgb = np.asarray(_flatten_rgb(image), dtype=np.uint8)
    height, width, _ = rgb.shape
    size = max(1, min(corner_size, height, width))
    corners = {
        "top_left": rgb[:size, :size],
        "top_right": rgb[:size, width - size :],
        "bottom_left": rgb[height - size :, :size],
        "bottom_right": rgb[height - size :, width - size :],
    }
    corner_white = {
        key: bool(np.all(value >= threshold)) for key, value in corners.items()
    }
    white_pixel_fraction = float(np.mean(np.all(rgb >= threshold, axis=2)))
    return {
        "threshold": threshold,
        "corner_sample_px": size,
        "corners_white": corner_white,
        "all_corners_white": all(corner_white.values()),
        "white_pixel_fraction": white_pixel_fraction,
    }


def _check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
    return {"name": name, "status": PASS if passed else FAIL, **evidence}


def _eligible_for_geometry_exception(
    check: Mapping[str, Any],
    *,
    allowed: bool,
) -> bool:
    return (
        allowed
        and check.get("status") == FAIL
        and check.get("name") in APPROVED_GEOMETRY_CHECKS
        and check.get("absolute_side_differences") is not None
        and check.get("comparable") is not False
    )


def _apply_geometry_exception(
    check: Mapping[str, Any],
    *,
    allowed: bool,
) -> dict[str, Any]:
    copied = dict(check)
    if not _eligible_for_geometry_exception(copied, allowed=allowed):
        return copied
    copied.update(
        {
            "raw_status": FAIL,
            "status": APPROVED_EXCEPTION,
            "exception_policy_id": GEOMETRY_REFLOW_EXCEPTION_POLICY["policy_id"],
            "exception_scope": "normalized_geometry_reflow_only",
            "raw_evidence_retained": True,
        }
    )
    return copied


def _automated_status_from_checks(
    checks: Sequence[Mapping[str, Any]],
) -> str:
    if not checks:
        return FAIL
    statuses = {str(check.get("status")) for check in checks}
    allowed = {PASS, APPROVED_EXCEPTION}
    if not statuses.issubset(allowed):
        return FAIL
    if APPROVED_EXCEPTION in statuses:
        return PASS_WITH_APPROVED_EXCEPTIONS
    return PASS


def _aggregate_automated_status(
    results: Sequence[Mapping[str, Any]],
) -> str:
    if not results:
        return FAIL
    statuses = {str(result.get("automated_status")) for result in results}
    allowed = {PASS, PASS_WITH_APPROVED_EXCEPTIONS}
    if not statuses.issubset(allowed):
        return FAIL
    if PASS_WITH_APPROVED_EXCEPTIONS in statuses:
        return PASS_WITH_APPROVED_EXCEPTIONS
    return PASS


def automated_status_is_success(status: str) -> bool:
    return status in {PASS, PASS_WITH_APPROVED_EXCEPTIONS}


def validate_figure(
    spec: FigureQASpec,
    *,
    aspect_ratio_tolerance: float = 0.01,
    bbox_side_tolerance: float = 0.02,
    dpi_tolerance: float = 1.0,
) -> dict[str, Any]:
    """Validate one generated TIFF and compare geometry with its PNG oracle.

    ``aspect_ratio_tolerance`` and ``bbox_side_tolerance`` are fractions, so
    the defaults implement the publication plan's one- and two-percent gates.
    """

    oracle_path = spec.oracle_path.resolve()
    tiff_path = spec.tiff_path.resolve()
    if not oracle_path.is_file():
        raise FileNotFoundError(f"Approved PNG oracle not found: {oracle_path}")
    if not tiff_path.is_file():
        raise FileNotFoundError(f"Generated TIFF not found: {tiff_path}")

    with Image.open(oracle_path) as oracle_source:
        oracle = _flatten_rgb(oracle_source)
        oracle_format = oracle_source.format
    with Image.open(tiff_path) as generated_source:
        generated_format = generated_source.format
        generated_mode = generated_source.mode
        generated_bands = generated_source.getbands()
        generated_info = dict(generated_source.info)
        generated_dpi = _dpi_from_image(generated_source)
        compression_name, compression_tag = _compression_details(generated_source)
        generated = _flatten_rgb(generated_source)

    expected_height_px = int(
        round(spec.target_width_px * oracle.height / oracle.width)
    )
    expected_dimensions = (spec.target_width_px, expected_height_px)
    oracle_ratio = oracle.width / oracle.height
    generated_ratio = generated.width / generated.height
    aspect_delta = abs(generated_ratio / oracle_ratio - 1.0)

    oracle_bbox = _normalized_nonwhite_bbox(oracle)
    generated_bbox = _normalized_nonwhite_bbox(generated)
    if oracle_bbox is None or generated_bbox is None:
        bbox_deltas: list[float] | None = None
        bbox_pass = oracle_bbox == generated_bbox
    else:
        bbox_deltas = [
            abs(generated_side - oracle_side)
            for oracle_side, generated_side in zip(
                oracle_bbox, generated_bbox, strict=True
            )
        ]
        bbox_pass = all(delta <= bbox_side_tolerance for delta in bbox_deltas)

    oracle_palette, oracle_present_colors = _core_palette(oracle)
    generated_palette, generated_present_colors = _core_palette(generated)
    oracle_core_codes = {
        (int(item["rgb"][0]) << 16)
        | (int(item["rgb"][1]) << 8)
        | int(item["rgb"][2])
        for item in oracle_palette
    }
    generated_core_codes = {
        (int(item["rgb"][0]) << 16)
        | (int(item["rgb"][1]) << 8)
        | int(item["rgb"][2])
        for item in generated_palette
    }
    missing_palette_codes = sorted(oracle_core_codes - generated_present_colors)
    unexpected_palette_codes = sorted(
        generated_core_codes - oracle_present_colors
    )
    palette_comparable = bool(oracle_core_codes or generated_core_codes)
    palette_pass = not missing_palette_codes and not unexpected_palette_codes

    color_geometry_pass, color_geometry = _mask_geometry_evidence(
        _palette_mask(oracle, oracle_palette),
        _palette_mask(generated, generated_palette),
        tolerance=bbox_side_tolerance,
    )
    dark_geometry_pass, dark_geometry = _mask_geometry_evidence(
        _dark_content_mask(oracle),
        _dark_content_mask(generated),
        tolerance=bbox_side_tolerance,
    )
    icc_bytes = generated_info.get("icc_profile")
    icc_description = _icc_description(icc_bytes)
    # ICC signatures are stored at bytes 36:40. Parse the description when
    # ImageCms is available, but retain deterministic validation on runtimes
    # where LittleCMS support is absent.
    icc_parseable = (
        isinstance(icc_bytes, bytes)
        and len(icc_bytes) >= 128
        and icc_bytes[36:40] == b"acsp"
    )
    icc_is_srgb = icc_parseable and (
        "srgb" in icc_description.casefold() or b"srgb" in icc_bytes.lower()
    )

    background = _white_background_metrics(generated)
    x_dpi, y_dpi = generated_dpi
    dpi_pass = (
        x_dpi is not None
        and y_dpi is not None
        and abs(x_dpi - spec.target_dpi) <= dpi_tolerance
        and abs(y_dpi - spec.target_dpi) <= dpi_tolerance
    )
    no_alpha = "A" not in generated_bands and "transparency" not in generated_info
    lzw = compression_tag == 5 or compression_name.casefold() in {
        "tiff_lzw",
        "lzw",
    }
    content_clear_of_edges = generated_bbox is None or (
        generated_bbox[0] > 0.0
        and generated_bbox[1] > 0.0
        and generated_bbox[2] < 1.0
        and generated_bbox[3] < 1.0
    )

    raw_checks = [
        _check(
            "approved_oracle_is_png",
            oracle_format == "PNG",
            observed=oracle_format,
        ),
        _check(
            "generated_format_is_tiff",
            generated_format == "TIFF",
            observed=generated_format,
        ),
        _check("generated_mode_is_rgb8", generated_mode == "RGB", observed=generated_mode),
        _check("generated_has_no_alpha", no_alpha, observed_bands=list(generated_bands)),
        _check(
            "generated_uses_lzw",
            lzw,
            observed_name=compression_name,
            observed_tag=compression_tag,
        ),
        _check(
            "generated_has_embedded_srgb",
            icc_is_srgb,
            icc_present=bool(icc_bytes),
            icc_parseable=icc_parseable,
            icc_description=icc_description,
        ),
        _check(
            "generated_dimensions_match_target",
            generated.size == expected_dimensions,
            expected=list(expected_dimensions),
            observed=list(generated.size),
        ),
        _check(
            "generated_dpi_matches_target",
            dpi_pass,
            expected=spec.target_dpi,
            tolerance=dpi_tolerance,
            observed=[x_dpi, y_dpi],
        ),
        _check(
            "aspect_ratio_matches_oracle",
            aspect_delta <= aspect_ratio_tolerance,
            tolerance=aspect_ratio_tolerance,
            relative_difference=aspect_delta,
            oracle=oracle_ratio,
            generated=generated_ratio,
        ),
        _check(
            "normalized_nonwhite_bbox_matches_oracle",
            bbox_pass,
            tolerance_per_side=bbox_side_tolerance,
            oracle=list(oracle_bbox) if oracle_bbox else None,
            generated=list(generated_bbox) if generated_bbox else None,
            absolute_side_differences=bbox_deltas,
        ),
        _check(
            "core_palette_matches_oracle",
            palette_pass,
            comparable=palette_comparable,
            oracle_core_palette=oracle_palette,
            generated_core_palette=generated_palette,
            missing_oracle_core_colors=[
                list(_decode_rgb_code(code)) for code in missing_palette_codes
            ],
            unexpected_generated_core_colors=[
                list(_decode_rgb_code(code))
                for code in unexpected_palette_codes
            ],
            method="frequent solid non-background colors; exact RGB identity",
        ),
        _check(
            "normalized_color_encoding_bbox_matches_oracle",
            color_geometry_pass,
            **color_geometry,
        ),
        _check(
            "normalized_dark_content_bbox_matches_oracle",
            dark_geometry_pass,
            **dark_geometry,
        ),
        _check(
            "content_does_not_touch_canvas_edge",
            content_clear_of_edges,
            generated=list(generated_bbox) if generated_bbox else None,
        ),

        _check(
            "background_is_white",
            bool(background["all_corners_white"])
            and float(background["white_pixel_fraction"]) > 0.0,
            **background,
        ),
    ]

    checks = [
        _apply_geometry_exception(
            check,
            allowed=spec.allow_approved_geometry_reflow,
        )
        for check in raw_checks
    ]
    hard_failure_checks = [
        str(check["name"]) for check in checks if check["status"] == FAIL
    ]
    approved_exception_checks = [
        str(check["name"])
        for check in checks
        if check["status"] == APPROVED_EXCEPTION
    ]
    automated_status = _automated_status_from_checks(checks)
    return {
        "asset_id": spec.asset_id,
        "automated_status": automated_status,
        "visual_approval_status": VISUAL_APPROVAL_STATUS,
        "hard_failure_checks": hard_failure_checks,
        "approved_exception_checks": approved_exception_checks,
        "approved_exception_policy_id": (
            GEOMETRY_REFLOW_EXCEPTION_POLICY["policy_id"]
            if approved_exception_checks
            else None
        ),
        "allow_approved_geometry_reflow": spec.allow_approved_geometry_reflow,
        "oracle_path": str(oracle_path),
        "tiff_path": str(tiff_path),
        "oracle_sha256": _sha256(oracle_path),
        "tiff_sha256": _sha256(tiff_path),
        "target_width_px": spec.target_width_px,
        "target_height_px": expected_height_px,
        "target_dpi": spec.target_dpi,
        "checks": checks,
    }


def create_comparison_sheet(
    spec: FigureQASpec,
    output_path: Path,
    *,
    scale: float,
) -> Path:
    """Create a labeled PNG showing oracle and TIFF at a fixed scale."""

    if scale <= 0:
        raise ValueError("scale must be positive")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(spec.oracle_path) as oracle_source:
        oracle = _flatten_rgb(oracle_source)
    with Image.open(spec.tiff_path) as tiff_source:
        generated = _flatten_rgb(tiff_source)

    # Compare at the TIFF's exact publication geometry.  The oracle is an
    # appearance reference and is resampled only in this disposable QA sheet.
    panel_size = (
        max(1, int(round(generated.width * scale))),
        max(1, int(round(generated.height * scale))),
    )
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    oracle_panel = oracle.resize(panel_size, resampling)
    generated_panel = generated.resize(panel_size, resampling)

    font = ImageFont.load_default()
    label_height = 36
    margin = 20
    gap = 28
    sheet = Image.new(
        "RGB",
        (
            2 * panel_size[0] + gap + 2 * margin,
            panel_size[1] + label_height + 2 * margin,
        ),
        "white",
    )
    sheet.paste(oracle_panel, (margin, margin + label_height))
    right_x = margin + panel_size[0] + gap
    sheet.paste(generated_panel, (right_x, margin + label_height))
    draw = ImageDraw.Draw(sheet)
    scale_label = f"{scale * 100:g}%"
    draw.text((margin, margin), f"Approved PNG oracle - {scale_label}", fill="black", font=font)
    draw.text(
        (right_x, margin),
        f"Generated TIFF - {scale_label}",
        fill="black",
        font=font,
    )
    sheet.save(output_path, format="PNG", dpi=(spec.target_dpi, spec.target_dpi))
    return output_path


def _check_status(result: Mapping[str, Any], name: str) -> str:
    check = _check_record(result, name)
    return str(check.get("status")) if check else "NOT_RECORDED"


def _check_record(
    result: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any] | None:
    for check in result.get("checks", []):
        if check.get("name") == name:
            return check
    return None


def _geometry_delta_summary(check: Mapping[str, Any] | None) -> str:
    if not check or check.get("status") != APPROVED_EXCEPTION:
        return ""
    deltas = check.get("absolute_side_differences")
    tolerance = check.get("tolerance_per_side")
    maximum = max(float(value) for value in deltas)
    formatted = ", ".join(f"{float(value):.6f}" for value in deltas)
    return (
        f"; raw side deltas=[{formatted}], max={maximum:.6f}, "
        f"tolerance={float(tolerance):.6f}, policy="
        f"{check['exception_policy_id']}"
    )


def _comparison_link(
    result: Mapping[str, Any],
    key: str,
    output_root: Path,
) -> str:
    value = result.get("comparison_sheets", {}).get(key)
    if not value:
        return "MISSING"
    path = Path(str(value)).resolve()
    try:
        return path.relative_to(output_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_visual_approval_checklist(
    results: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> Path:
    """Write comparison evidence with deliberately blank visual decisions."""

    output_root = Path(output_root)
    path = output_root / "VISUAL_APPROVAL_CHECKLIST.md"
    automated_status = _aggregate_automated_status(results)
    lines = [
        "# Publication Figure Pilot Visual Approval Checklist",
        "",
        f"**Automated status: {automated_status}**",
        f"**Visual status: {VISUAL_APPROVAL_STATUS}**",
        "",
        "This file records comparison evidence only. It does not grant visual approval.",
        "",
        "## Approved geometry-reflow exception",
        "",
        f"- Policy: `{GEOMETRY_REFLOW_EXCEPTION_POLICY['policy_id']}`",
        "- Approved by Erin Glen on 2026-08-06.",
        f"- Decision: {GEOMETRY_REFLOW_EXCEPTION_POLICY['decision']}",
        (
            "- `APPROVED_EXCEPTION` means the 2% geometry tolerance was not met; "
            "it does not mean the metric passed. Raw bboxes and deltas remain in "
            "`pilot_figure_qa.json`."
        ),
        (
            "- This exception never covers missing content, clipping, palette "
            "changes, TIFF defects, text/line minimums, active artist extents, "
            "or numerical controls."
        ),
        "",
        "## Required viewing procedure",
        "",
        (
            "1. Open each `100pct` sheet in a print-aware image application "
            "that honors its 300-dpi metadata; use physical/print-size view, "
            "not browser CSS 100%."
        ),
        (
            "2. Confirm each panel's physical width from the recorded pixels "
            "and dpi (approximately 85 mm for 1,004 px or 180 mm for 2,126 px)."
        ),
        (
            "3. Inspect the paired `150pct` sheet for clipping, text collisions, "
            "palette changes, line weight, marker shape, legends, axes, and "
            "numerical labels."
        ),
        (
            "4. Any hard FAIL requires HOLD. APPROVED_EXCEPTION requires visual "
            "inspection but is not itself a hard failure."
        ),
        "",
        "## Per-asset evidence",
        "",
    ]
    metric_names = (
        ("Core palette", "core_palette_matches_oracle"),
        (
            "Color-encoding geometry",
            "normalized_color_encoding_bbox_matches_oracle",
        ),
        (
            "Dark text/line geometry",
            "normalized_dark_content_bbox_matches_oracle",
        ),
        (
            "Aggregate visual geometry",
            "normalized_nonwhite_bbox_matches_oracle",
        ),
        ("Canvas-edge clearance", "content_does_not_touch_canvas_edge"),
    )
    for result in results:
        asset_id = str(result.get("asset_id"))
        width_mm = (
            float(result.get("target_width_px", 0))
            / float(result.get("target_dpi", 300))
            * 25.4
        )
        lines.extend(
            [
                f"### {asset_id}",
                "",
                f"- Automated status: **{result.get('automated_status')}**",
                (
                    "- Hard failures: "
                    + (
                        ", ".join(result.get("hard_failure_checks", []))
                        or "NONE"
                    )
                ),
                (
                    "- Approved exceptions: "
                    + (
                        ", ".join(result.get("approved_exception_checks", []))
                        or "NONE"
                    )
                ),
                f"- Target final width: {width_mm:.1f} mm",
            ]
        )
        for label, name in metric_names:
            check = _check_record(result, name)
            lines.append(
                f"- {label}: `{_check_status(result, name)}`"
                f"{_geometry_delta_summary(check)}"
            )
        lines.extend(
            [
                (
                    "- [100% publication-size comparison]"
                    f"({_comparison_link(result, 'publication_size', output_root)})"
                ),
                (
                    "- [150% enlarged comparison]"
                    f"({_comparison_link(result, 'enlarged_150_percent', output_root)})"
                ),
                "- Erin visual decision: [ ] APPROVE  [ ] HOLD",
                "- Erin initials/date: ______________________________",
                (
                    "- Notes: "
                    "________________________________________________________________"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Overall Erin visual decision",
            "",
            "[ ] APPROVE PILOT  [ ] HOLD PILOT",
            "",
            "Erin initials/date: ______________________________",
            "",
            (
                "Artist-level and numerical hard-gate evidence is stored in "
                "`pilot_manifest.json`; any absent or non-PASS record is HOLD."
            ),
            (
                "The visual status remains `PENDING_VISUAL_APPROVAL` until a "
                "separate reviewed manifest records the decision."
            ),
            "",
        ]
    )
    path.write_text(chr(10).join(lines), encoding="utf-8")
    return path


def _flatten_checks(result: Mapping[str, Any]) -> dict[str, Any]:
    statuses = {
        str(check.get("name")): str(check.get("status"))
        for check in result.get("checks", [])
    }
    hard_failed = [
        str(check["name"])
        for check in result.get("checks", [])
        if check.get("status") == FAIL
    ]
    approved_exceptions = [
        str(check["name"])
        for check in result.get("checks", [])
        if check.get("status") == APPROVED_EXCEPTION
    ]
    return {
        "asset_id": result.get("asset_id"),
        "automated_status": result.get("automated_status"),
        "visual_approval_status": result.get("visual_approval_status"),
        "oracle_path": result.get("oracle_path"),
        "tiff_path": result.get("tiff_path"),
        "oracle_sha256": result.get("oracle_sha256"),
        "tiff_sha256": result.get("tiff_sha256"),
        "target_width_px": result.get("target_width_px"),
        "target_height_px": result.get("target_height_px"),
        "target_dpi": result.get("target_dpi"),
        "hard_failed_checks": ";".join(hard_failed),
        "approved_exception_checks": ";".join(approved_exceptions),
        "approved_exception_policy_id": (
            GEOMETRY_REFLOW_EXCEPTION_POLICY["policy_id"]
            if approved_exceptions
            else ""
        ),
        "core_palette_status": statuses.get(
            "core_palette_matches_oracle", "NOT_RECORDED"
        ),
        "color_encoding_bbox_status": statuses.get(
            "normalized_color_encoding_bbox_matches_oracle", "NOT_RECORDED"
        ),
        "dark_content_bbox_status": statuses.get(
            "normalized_dark_content_bbox_matches_oracle", "NOT_RECORDED"
        ),
        "aggregate_bbox_status": statuses.get(
            "normalized_nonwhite_bbox_matches_oracle", "NOT_RECORDED"
        ),
    }


def write_reports(results: Sequence[Mapping[str, Any]], output_root: Path) -> dict[str, Path]:
    """Write detailed JSON and a compact one-row-per-asset CSV report."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "pilot_figure_qa.json"
    csv_path = output_root / "pilot_figure_qa.csv"

    automated_status = _aggregate_automated_status(results)
    hard_failure_count = sum(
        len(item.get("hard_failure_checks", [])) for item in results
    )
    approved_exception_count = sum(
        len(item.get("approved_exception_checks", [])) for item in results
    )
    summary = {
        "automated_status": automated_status,
        "visual_approval_status": VISUAL_APPROVAL_STATUS,
        "asset_count": len(results),
        "approved_exception_policy": GEOMETRY_REFLOW_EXCEPTION_POLICY,
        "hard_failure_count": hard_failure_count,
        "approved_exception_count": approved_exception_count,
        "results": list(results),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = [_flatten_checks(result) for result in results]
    fieldnames = list(rows[0]) if rows else [
        "asset_id",
        "automated_status",
        "visual_approval_status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    checklist_path = write_visual_approval_checklist(results, output_root)
    return {
        "json": json_path,
        "csv": csv_path,
        "visual_checklist": checklist_path,
    }


def run_pilot_qa(specs: Iterable[FigureQASpec], output_root: Path) -> dict[str, Any]:
    """Run automated checks and create 100% and 150% comparison evidence."""

    output_root = Path(output_root)
    comparisons_root = output_root / "comparison_sheets"
    results: list[dict[str, Any]] = []
    for spec in specs:
        result = validate_figure(spec)
        sheet_100 = create_comparison_sheet(
            spec,
            comparisons_root / f"{spec.asset_id}_comparison_100pct.png",
            scale=1.0,
        )
        sheet_150 = create_comparison_sheet(
            spec,
            comparisons_root / f"{spec.asset_id}_comparison_150pct.png",
            scale=1.5,
        )
        result["comparison_sheets"] = {
            "publication_size": str(sheet_100.resolve()),
            "enlarged_150_percent": str(sheet_150.resolve()),
        }
        results.append(result)

    report_paths = write_reports(results, output_root)
    automated_status = _aggregate_automated_status(results)
    hard_failure_count = sum(
        len(item.get("hard_failure_checks", [])) for item in results
    )
    approved_exception_count = sum(
        len(item.get("approved_exception_checks", [])) for item in results
    )
    return {
        "automated_status": automated_status,
        "visual_approval_status": VISUAL_APPROVAL_STATUS,
        "asset_count": len(results),
        "approved_exception_policy": GEOMETRY_REFLOW_EXCEPTION_POLICY,
        "hard_failure_count": hard_failure_count,
        "approved_exception_count": approved_exception_count,
        "report_paths": {
            key: str(path.resolve()) for key, path in report_paths.items()
        },
        "results": results,
    }


def _load_specs(
    manifest_path: Path,
    *,
    oracle_root: Path | None,
    generated_root: Path | None,
) -> list[FigureQASpec]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_specs = payload.get("figures", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_specs, list):
        raise ValueError("QA manifest must be a JSON list or contain a 'figures' list")
    return [
        FigureQASpec.from_mapping(
            item,
            oracle_root=oracle_root,
            generated_root=generated_root,
        )
        for item in raw_specs
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path)
    parser.add_argument("--generated-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = _load_specs(
        args.manifest,
        oracle_root=args.oracle_root,
        generated_root=args.generated_root,
    )
    result = run_pilot_qa(specs, args.output_root)
    print(json.dumps({key: result[key] for key in ("automated_status", "visual_approval_status", "asset_count", "report_paths")}, indent=2))
    return (
        0
        if automated_status_is_success(result["automated_status"])
        else 1
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
