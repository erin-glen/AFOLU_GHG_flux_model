from pathlib import Path

from PIL import Image, ImageDraw

from src.scripts.zonal_statistics.pub_scripts import publication_figure_qa as qa
from src.scripts.zonal_statistics.pub_scripts.publication_figure_export import (
    _srgb_profile_bytes,
)


def test_content_touching_canvas_edge_is_rejected(tmp_path: Path) -> None:
    oracle = Image.new("RGB", (100, 60), "white")
    ImageDraw.Draw(oracle).rectangle((10, 8, 89, 49), fill="#3E3753")
    oracle_path = tmp_path / "oracle.png"
    oracle.save(oracle_path, format="PNG")

    generated = Image.new("RGB", (100, 60), "white")
    ImageDraw.Draw(generated).rectangle((0, 8, 89, 49), fill="#3E3753")
    generated_path = tmp_path / "generated.tif"
    generated.save(
        generated_path,
        format="TIFF",
        compression="tiff_lzw",
        dpi=(300, 300),
        icc_profile=_srgb_profile_bytes(),
    )

    result = qa.validate_figure(
        qa.FigureQASpec(
            "edge_case",
            oracle_path,
            generated_path,
            target_width_px=100,
            allow_approved_geometry_reflow=True,
        ),
        bbox_side_tolerance=1.0,
    )
    checks = {item["name"]: item["status"] for item in result["checks"]}
    assert checks["content_does_not_touch_canvas_edge"] == qa.FAIL
    assert result["automated_status"] == qa.FAIL
