import csv
import json
from pathlib import Path

from PIL import Image, ImageCms, ImageDraw
import pytest

from src.scripts.zonal_statistics.pub_scripts import publication_figure_qa as qa


def _srgb_bytes() -> bytes:
    try:
        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except ImportError:
        profile_path = Path(
            r"C:\Windows\System32\spool\drivers\color\sRGB Color Space Profile.icm"
        )
        return profile_path.read_bytes()


def _save_tiff_image(
    path: Path,
    image: Image.Image,
    *,
    dpi: tuple[int, int] = (300, 300),
    compression: str = "tiff_lzw",
    include_icc: bool = True,
) -> None:
    kwargs = {"format": "TIFF", "compression": compression, "dpi": dpi}
    if include_icc:
        kwargs["icc_profile"] = _srgb_bytes()
    image.save(path, **kwargs)

def _make_oracle(path: Path, size: tuple[int, int] = (100, 60)) -> None:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 8, 89, 49), fill=(31, 119, 180))
    image.save(path, format="PNG", dpi=(300, 300))


def _make_tiff(
    path: Path,
    *,
    size: tuple[int, int] = (100, 60),
    dpi: tuple[int, int] = (300, 300),
    compression: str = "tiff_lzw",
    include_icc: bool = True,
) -> None:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 8, 89, 49), fill=(31, 119, 180))
    _save_tiff_image(
        path,
        image,
        dpi=dpi,
        compression=compression,
        include_icc=include_icc,
    )


def test_validate_figure_accepts_compliant_tiff(tmp_path):
    oracle_path = tmp_path / "approved.png"
    tiff_path = tmp_path / "Figure_01a.tif"
    _make_oracle(oracle_path)
    _make_tiff(tiff_path)

    result = qa.validate_figure(
        qa.FigureQASpec("Figure_01a", oracle_path, tiff_path, target_width_px=100)
    )

    assert result["automated_status"] == qa.PASS
    assert result["visual_approval_status"] == qa.VISUAL_APPROVAL_STATUS
    assert all(check["status"] == qa.PASS for check in result["checks"])


def test_validate_figure_reports_noncompliant_tiff_metadata(tmp_path):
    oracle_path = tmp_path / "approved.png"
    tiff_path = tmp_path / "bad.tif"
    _make_oracle(oracle_path)
    _make_tiff(
        tiff_path,
        size=(101, 60),
        dpi=(150, 150),
        compression="raw",
        include_icc=False,
    )

    result = qa.validate_figure(
        qa.FigureQASpec("Figure_bad", oracle_path, tiff_path, target_width_px=100)
    )
    by_name = {check["name"]: check["status"] for check in result["checks"]}

    assert result["automated_status"] == qa.FAIL
    assert result["visual_approval_status"] == qa.VISUAL_APPROVAL_STATUS
    assert by_name["generated_dimensions_match_target"] == qa.FAIL
    assert by_name["generated_dpi_matches_target"] == qa.FAIL
    assert by_name["generated_uses_lzw"] == qa.FAIL
    assert by_name["generated_has_embedded_srgb"] == qa.FAIL


def test_palette_change_fails_even_when_geometry_matches(tmp_path):
    oracle_path = tmp_path / "approved.png"
    tiff_path = tmp_path / "changed.tif"
    oracle = Image.new("RGB", (100, 60), "white")
    generated = Image.new("RGB", (100, 60), "white")
    ImageDraw.Draw(oracle).rectangle((10, 8, 89, 49), fill=(31, 119, 180))
    ImageDraw.Draw(generated).rectangle((10, 8, 89, 49), fill=(214, 39, 40))
    oracle.save(oracle_path, format="PNG", dpi=(300, 300))
    _save_tiff_image(tiff_path, generated)

    result = qa.validate_figure(
        qa.FigureQASpec("palette", oracle_path, tiff_path, 100)
    )
    by_name = {check["name"]: check for check in result["checks"]}

    assert result["automated_status"] == qa.FAIL
    palette = by_name["core_palette_matches_oracle"]
    assert palette["status"] == qa.FAIL
    assert palette["missing_oracle_core_colors"] == [[31, 119, 180]]
    assert palette["unexpected_generated_core_colors"] == [[214, 39, 40]]
    assert (
        by_name["normalized_color_encoding_bbox_matches_oracle"]["status"]
        == qa.PASS
    )


def test_shifted_color_encoding_bbox_fails_two_percent_gate(tmp_path):
    oracle_path = tmp_path / "approved.png"
    tiff_path = tmp_path / "shifted.tif"
    oracle = Image.new("RGB", (100, 60), "white")
    generated = Image.new("RGB", (100, 60), "white")
    ImageDraw.Draw(oracle).rectangle((10, 8, 59, 49), fill=(31, 119, 180))
    ImageDraw.Draw(generated).rectangle((25, 8, 74, 49), fill=(31, 119, 180))
    oracle.save(oracle_path, format="PNG", dpi=(300, 300))
    _save_tiff_image(tiff_path, generated)

    result = qa.validate_figure(
        qa.FigureQASpec("shifted", oracle_path, tiff_path, 100)
    )
    by_name = {check["name"]: check for check in result["checks"]}

    assert by_name["core_palette_matches_oracle"]["status"] == qa.PASS
    shifted = by_name["normalized_color_encoding_bbox_matches_oracle"]
    assert shifted["status"] == qa.FAIL
    assert shifted["maximum_side_difference"] > 0.02


def test_explicit_opt_in_converts_comparable_geometry_to_exception(tmp_path):
    oracle_path = tmp_path / "approved.png"
    tiff_path = tmp_path / "shifted.tif"
    oracle = Image.new("RGB", (100, 60), "white")
    generated = Image.new("RGB", (100, 60), "white")
    ImageDraw.Draw(oracle).rectangle((10, 8, 59, 49), fill=(31, 119, 180))
    ImageDraw.Draw(generated).rectangle((25, 8, 74, 49), fill=(31, 119, 180))
    oracle.save(oracle_path, format="PNG", dpi=(300, 300))
    _save_tiff_image(tiff_path, generated)

    result = qa.validate_figure(
        qa.FigureQASpec(
            "approved_reflow",
            oracle_path,
            tiff_path,
            100,
            allow_approved_geometry_reflow=True,
        )
    )
    by_name = {check["name"]: check for check in result["checks"]}

    assert result["automated_status"] == qa.PASS_WITH_APPROVED_EXCEPTIONS
    assert result["visual_approval_status"] == qa.VISUAL_APPROVAL_STATUS
    assert result["hard_failure_checks"] == []
    assert set(result["approved_exception_checks"]) == {
        "normalized_nonwhite_bbox_matches_oracle",
        "normalized_color_encoding_bbox_matches_oracle",
    }
    for name in result["approved_exception_checks"]:
        check = by_name[name]
        assert check["status"] == qa.APPROVED_EXCEPTION
        assert check["raw_status"] == qa.FAIL
        assert check["raw_evidence_retained"] is True
        assert max(check["absolute_side_differences"]) > 0.02
        assert check["exception_policy_id"] == (
            qa.GEOMETRY_REFLOW_EXCEPTION_POLICY["policy_id"]
        )
    assert by_name["core_palette_matches_oracle"]["status"] == qa.PASS
    assert by_name["content_does_not_touch_canvas_edge"]["status"] == qa.PASS


def test_opt_in_does_not_waive_palette_failure(tmp_path):
    oracle_path = tmp_path / "approved.png"
    tiff_path = tmp_path / "bad_palette.tif"
    oracle = Image.new("RGB", (100, 60), "white")
    generated = Image.new("RGB", (100, 60), "white")
    ImageDraw.Draw(oracle).rectangle((10, 8, 59, 49), fill=(31, 119, 180))
    ImageDraw.Draw(generated).rectangle((25, 8, 74, 49), fill=(214, 39, 40))
    oracle.save(oracle_path, format="PNG", dpi=(300, 300))
    _save_tiff_image(tiff_path, generated)

    result = qa.validate_figure(
        qa.FigureQASpec(
            "palette_hard_fail",
            oracle_path,
            tiff_path,
            100,
            allow_approved_geometry_reflow=True,
        )
    )
    by_name = {check["name"]: check for check in result["checks"]}

    assert result["automated_status"] == qa.FAIL
    assert by_name["core_palette_matches_oracle"]["status"] == qa.FAIL
    assert "core_palette_matches_oracle" in result["hard_failure_checks"]


def test_opt_in_does_not_waive_missing_content_mask(tmp_path):
    oracle_path = tmp_path / "approved.png"
    tiff_path = tmp_path / "missing-label.tif"
    oracle = Image.new("RGB", (100, 60), "white")
    generated = Image.new("RGB", (100, 60), "white")
    for image in (oracle, generated):
        ImageDraw.Draw(image).rectangle(
            (20, 8, 89, 49),
            fill=(31, 119, 180),
        )
    ImageDraw.Draw(oracle).rectangle((3, 20, 7, 40), fill=(0, 0, 0))
    oracle.save(oracle_path, format="PNG", dpi=(300, 300))
    _save_tiff_image(tiff_path, generated)

    result = qa.validate_figure(
        qa.FigureQASpec(
            "missing_opt_in",
            oracle_path,
            tiff_path,
            100,
            allow_approved_geometry_reflow=True,
        )
    )
    by_name = {check["name"]: check for check in result["checks"]}
    dark = by_name["normalized_dark_content_bbox_matches_oracle"]

    assert result["automated_status"] == qa.FAIL
    assert dark["status"] == qa.FAIL
    assert dark["reason"] == "mask_missing_from_one_image"
    assert "normalized_dark_content_bbox_matches_oracle" in result[
        "hard_failure_checks"
    ]

def test_missing_peripheral_dark_content_fails(tmp_path):
    oracle_path = tmp_path / "approved.png"
    tiff_path = tmp_path / "missing-label.tif"
    oracle = Image.new("RGB", (100, 60), "white")
    generated = Image.new("RGB", (100, 60), "white")
    for image in (oracle, generated):
        ImageDraw.Draw(image).rectangle(
            (20, 8, 89, 49),
            fill=(31, 119, 180),
        )
    ImageDraw.Draw(oracle).rectangle((3, 20, 7, 40), fill=(0, 0, 0))
    oracle.save(oracle_path, format="PNG", dpi=(300, 300))
    _save_tiff_image(tiff_path, generated)

    result = qa.validate_figure(
        qa.FigureQASpec("missing", oracle_path, tiff_path, 100)
    )
    by_name = {check["name"]: check for check in result["checks"]}

    assert by_name["core_palette_matches_oracle"]["status"] == qa.PASS
    assert (
        by_name["normalized_color_encoding_bbox_matches_oracle"]["status"]
        == qa.PASS
    )
    dark = by_name["normalized_dark_content_bbox_matches_oracle"]
    assert dark["status"] == qa.FAIL
    assert dark["reason"] == "mask_missing_from_one_image"

def test_run_pilot_qa_writes_reports_and_two_comparison_scales(tmp_path):
    oracle_path = tmp_path / "approved.png"
    tiff_path = tmp_path / "Figure_01a.tif"
    output_root = tmp_path / "qa"
    _make_oracle(oracle_path)
    _make_tiff(tiff_path)
    spec = qa.FigureQASpec(
        "Figure_01a", oracle_path, tiff_path, target_width_px=100
    )

    result = qa.run_pilot_qa([spec], output_root)

    assert result["automated_status"] == qa.PASS
    assert result["visual_approval_status"] == "PENDING_VISUAL_APPROVAL"
    sheet_100 = output_root / "comparison_sheets/Figure_01a_comparison_100pct.png"
    sheet_150 = output_root / "comparison_sheets/Figure_01a_comparison_150pct.png"
    assert sheet_100.is_file()
    assert sheet_150.is_file()
    with Image.open(sheet_100) as actual, Image.open(sheet_150) as enlarged:
        assert enlarged.width > actual.width
        assert enlarged.height > actual.height

    json_report = json.loads((output_root / "pilot_figure_qa.json").read_text())
    assert json_report["visual_approval_status"] == "PENDING_VISUAL_APPROVAL"
    assert json_report["results"][0]["comparison_sheets"]["publication_size"].endswith(
        "Figure_01a_comparison_100pct.png"
    )
    with (output_root / "pilot_figure_qa.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["asset_id"] == "Figure_01a"
    assert rows[0]["visual_approval_status"] == "PENDING_VISUAL_APPROVAL"
    checklist_path = output_root / "VISUAL_APPROVAL_CHECKLIST.md"
    assert checklist_path.is_file()
    checklist = checklist_path.read_text(encoding="utf-8")
    assert "Visual status: PENDING_VISUAL_APPROVAL" in checklist
    assert "Figure_01a" in checklist
    assert "100% publication-size comparison" in checklist
    assert "150% enlarged comparison" in checklist
    assert "Erin visual decision: [ ] APPROVE  [ ] HOLD" in checklist
    assert "Status: APPROVED" not in checklist
    assert result["report_paths"]["visual_checklist"].endswith(
        "VISUAL_APPROVAL_CHECKLIST.md"
    )


def test_geometry_reflow_opt_in_requires_boolean(tmp_path):
    payload = {
        "asset_id": "x",
        "oracle_path": "a.png",
        "tiff_path": "a.tif",
        "target_width_px": 100,
        "allow_approved_geometry_reflow": "false",
    }
    with pytest.raises(ValueError, match="must be a boolean"):
        qa.FigureQASpec.from_mapping(payload)


def test_approved_exception_status_is_success_but_visual_stays_pending():
    assert qa.automated_status_is_success(qa.PASS)
    assert qa.automated_status_is_success(
        qa.PASS_WITH_APPROVED_EXCEPTIONS
    )
    assert not qa.automated_status_is_success(qa.FAIL)
    assert qa.VISUAL_APPROVAL_STATUS == "PENDING_VISUAL_APPROVAL"


def test_opted_in_report_records_exceptions_and_pending_visual_status(tmp_path):
    oracle_path = tmp_path / "approved.png"
    tiff_path = tmp_path / "shifted.tif"
    output_root = tmp_path / "qa"
    oracle = Image.new("RGB", (100, 60), "white")
    generated = Image.new("RGB", (100, 60), "white")
    ImageDraw.Draw(oracle).rectangle((10, 8, 59, 49), fill=(31, 119, 180))
    ImageDraw.Draw(generated).rectangle((25, 8, 74, 49), fill=(31, 119, 180))
    oracle.save(oracle_path, format="PNG", dpi=(300, 300))
    _save_tiff_image(tiff_path, generated)
    spec = qa.FigureQASpec(
        "approved_reflow",
        oracle_path,
        tiff_path,
        100,
        allow_approved_geometry_reflow=True,
    )

    result = qa.run_pilot_qa([spec], output_root)
    checklist = (output_root / "VISUAL_APPROVAL_CHECKLIST.md").read_text(
        encoding="utf-8"
    )
    with (output_root / "pilot_figure_qa.csv").open(
        newline="",
        encoding="utf-8",
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert result["automated_status"] == qa.PASS_WITH_APPROVED_EXCEPTIONS
    assert result["visual_approval_status"] == qa.VISUAL_APPROVAL_STATUS
    assert result["hard_failure_count"] == 0
    assert result["approved_exception_count"] == 2
    assert qa.GEOMETRY_REFLOW_EXCEPTION_POLICY["policy_id"] in checklist
    assert "APPROVED_EXCEPTION` means the 2% geometry tolerance" in checklist
    assert "raw side deltas=" in checklist
    assert "Erin visual decision: [ ] APPROVE  [ ] HOLD" in checklist
    assert "Visual status: APPROVED" not in checklist
    assert rows[0]["hard_failed_checks"] == ""
    assert "normalized_color_encoding_bbox_matches_oracle" in rows[0][
        "approved_exception_checks"
    ]

def test_manifest_cannot_override_visual_approval_status(tmp_path):
    oracle_path = tmp_path / "approved.png"
    tiff_path = tmp_path / "Figure_01a.tif"
    _make_oracle(oracle_path)
    _make_tiff(tiff_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "asset_id": "Figure_01a",
                    "oracle_path": oracle_path.name,
                    "tiff_path": tiff_path.name,
                    "target_width_px": 100,
                    "visual_approval_status": "APPROVED",
                }
            ]
        ),
        encoding="utf-8",
    )

    specs = qa._load_specs(
        manifest, oracle_root=tmp_path, generated_root=tmp_path
    )
    result = qa.run_pilot_qa(specs, tmp_path / "qa")

    assert result["visual_approval_status"] == "PENDING_VISUAL_APPROVAL"
