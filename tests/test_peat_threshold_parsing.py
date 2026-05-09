from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest

from src.scripts.utilities import constants_and_names as cn


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/scripts/core_model/0_drainage_emissions_model.py"
)
SPEC = spec_from_file_location("drainage_model", MODULE_PATH)
drainage_model = module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(drainage_model)


def test_parse_baseline_thresholds_selects_fscore_metric(tmp_path):
    path = tmp_path / "biome_thresholds_summary.csv"
    pd.DataFrame(
        {
            "biome": ["boreal", "temperate", "tropical"],
            "best_f1_threshold": [0.29, 0.20, 0.31],
            "best_f2_threshold": [0.18, 0.15, 0.17],
        }
    ).to_csv(path, index=False)

    parsed = drainage_model.parse_biome_thresholds(
        str(path),
        fallback=0.10,
        fscore_metric="f2",
        threshold_scenario="baseline",
    )

    assert parsed[cn.ecozone_codes["boreal"]] == 18.0
    assert parsed[cn.ecozone_codes["temperate"]] == 15.0
    assert parsed[cn.ecozone_codes["tropical"]] == 17.0
    assert parsed[cn.ecozone_codes["unknown"]] == 10.0


def test_parse_json_thresholds_ignore_csv_selection_args():
    parsed = drainage_model.parse_biome_thresholds(
        '{"boreal": 0.29, "temperate": 0.20, "tropical": 0.31}',
        fallback=0.10,
        fscore_metric="not-a-metric",
        threshold_scenario="not-a-scenario",
    )

    assert parsed[cn.ecozone_codes["boreal"]] == pytest.approx(29.0)
    assert parsed[cn.ecozone_codes["temperate"]] == 20.0
    assert parsed[cn.ecozone_codes["tropical"]] == 31.0
    assert parsed[cn.ecozone_codes["unknown"]] == 10.0


def test_parse_baseline_scenario_csv_uses_operational_threshold(tmp_path):
    path = tmp_path / "scenario_bounds_thresholds_f2.csv"
    pd.DataFrame(
        {
            "metric": ["F1", "F2"],
            "biome": ["boreal", "boreal"],
            "operational_threshold": [0.295571, 0.189001],
            "low_area_threshold": [0.43, 0.38],
            "high_area_threshold": [0.23, 0.15],
        }
    ).to_csv(path, index=False)

    parsed = drainage_model.parse_biome_thresholds(
        str(path),
        fallback=0.10,
        fscore_metric="f2",
        threshold_scenario="baseline",
    )

    assert parsed[cn.ecozone_codes["boreal"]] == pytest.approx(18.9001)
    assert parsed[cn.ecozone_codes["unknown"]] == 10.0


def test_parse_scenario_thresholds_selects_low_and_high_area(tmp_path):
    path = tmp_path / "scenario_bounds_thresholds_f1.csv"
    pd.DataFrame(
        {
            "metric": ["F1", "F1", "F1"],
            "biome": ["boreal", "temperate", "tropical"],
            "operational_threshold": [0.295571, 0.206218, 0.313584],
            "low_area_threshold": [0.43, 0.34, 0.42],
            "high_area_threshold": [0.23, 0.14, 0.23],
        }
    ).to_csv(path, index=False)

    low_area = drainage_model.parse_biome_thresholds(
        str(path),
        fallback=0.10,
        fscore_metric="f1",
        threshold_scenario="low",
    )
    high_area = drainage_model.parse_biome_thresholds(
        str(path),
        fallback=0.10,
        fscore_metric="f1",
        threshold_scenario="high_area",
    )

    assert low_area[cn.ecozone_codes["boreal"]] == 43.0
    assert low_area[cn.ecozone_codes["temperate"]] == 34.0
    assert low_area[cn.ecozone_codes["tropical"]] == 42.0
    assert high_area[cn.ecozone_codes["boreal"]] == 23.0
    assert high_area[cn.ecozone_codes["temperate"]] == pytest.approx(14.0)
    assert high_area[cn.ecozone_codes["tropical"]] == 23.0


def test_parse_scenario_thresholds_rejects_wrong_metric_file(tmp_path):
    path = tmp_path / "scenario_bounds_thresholds_f1.csv"
    pd.DataFrame(
        {
            "metric": ["F1"],
            "biome": ["boreal"],
            "operational_threshold": [0.295571],
            "low_area_threshold": [0.43],
            "high_area_threshold": [0.23],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="metric column"):
        drainage_model.parse_biome_thresholds(
            str(path),
            fscore_metric="f2",
            threshold_scenario="baseline",
        )


def test_parse_scenario_thresholds_requires_requested_column(tmp_path):
    path = tmp_path / "bad_thresholds.csv"
    pd.DataFrame({"biome": ["boreal"], "operational_threshold": [0.29]}).to_csv(
        path, index=False
    )

    with pytest.raises(KeyError, match="low_area"):
        drainage_model.parse_biome_thresholds(
            str(path),
            fscore_metric="f1",
            threshold_scenario="low_area",
        )
