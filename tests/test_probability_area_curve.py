import sys

import pandas as pd

from src.scripts.uncertainty import build_probability_area_curve as curve


def test_derive_global_output_path_removes_biome_suffix(tmp_path):
    output_path = tmp_path / "area_vs_threshold_20260508_biome.csv"

    assert curve.derive_global_output_path(output_path) == (
        tmp_path / "area_vs_threshold_20260508.csv"
    )


def test_per_biome_run_writes_global_companion_curve(tmp_path, monkeypatch):
    input_path = tmp_path / "class_area.csv"
    pd.DataFrame(
        {
            "adm0_id": [1, 1],
            "probability_class": [10, 20],
            "biome_id": [1, 2],
            "area_ha": [5.0, 7.0],
        }
    ).to_csv(input_path, index=False)

    per_biome_output = tmp_path / "area_vs_threshold_20260508_biome.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_probability_area_curve",
            "--input",
            str(input_path),
            "--per-biome",
            "--output",
            str(per_biome_output),
        ],
    )

    curve.main()

    global_output = tmp_path / "area_vs_threshold_20260508.csv"
    assert per_biome_output.exists()
    assert global_output.exists()

    global_curve = pd.read_csv(global_output)
    area_at_10 = global_curve.loc[global_curve["threshold"] == 0.10, "area_ha"].iloc[0]
    area_at_20 = global_curve.loc[global_curve["threshold"] == 0.20, "area_ha"].iloc[0]

    assert area_at_10 == 12.0
    assert area_at_20 == 7.0
