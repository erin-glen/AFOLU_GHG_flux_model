import importlib.util
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


MODULE_PATH = Path("src/scripts/zonal_statistics/03_qaqc_combined_state_alignment.py")
spec = importlib.util.spec_from_file_location("combined_state_qa", MODULE_PATH)
combined_state_qa = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(combined_state_qa)


def _combined_frame() -> pd.DataFrame:
    packed = int(
        combined_state_qa.zc.pack_combined_state(
            np.array([[np.uint32(0)]], dtype=np.uint32),
            np.array([[np.uint32(0)]], dtype=np.uint32),
        )[0, 0]
    )
    return pd.DataFrame(
        {
            "gadm_adm0": [840, 840, 840],
            "interval_end": [2024, 2024, 2024],
            "flux_type": [
                "area__ha",
                "drained_total_Mg_CO2e",
                "burned_total_Mg_CO2e",
            ],
            "value": [10.0, 1.5, 0.25],
            "combined_state_nodes": [packed, packed, packed],
        }
    )


def test_load_interval_frames_self_projects_when_legacy_branches_absent() -> None:
    frame = _combined_frame()
    with (
        mock.patch.object(
            combined_state_qa,
            "_resolve_paths",
            lambda *args: ("combined_path", "drained_path", "burned_path"),
        ),
        mock.patch.object(
            combined_state_qa,
            "_exists_parquet",
            lambda path: path == "combined_path",
        ),
        mock.patch.object(
            combined_state_qa,
            "_read_parquet_df",
            lambda path, cols: frame[[c for c in cols if c in frame.columns]].copy(),
        ),
    ):
        frames = combined_state_qa._load_interval_frames("1_0_0", "run", "20260510", "2021_2024")

    assert not frames.combined.empty
    assert not frames.drained.empty
    assert not frames.burned.empty
    assert "drained_state_nodes" in frames.drained.columns
    assert "burned_state_nodes" in frames.burned.columns


def test_evaluate_interval_passes_for_single_combined_branch() -> None:
    frame = _combined_frame()
    with (
        mock.patch.object(
            combined_state_qa,
            "_resolve_paths",
            lambda *args: ("combined_path", "drained_path", "burned_path"),
        ),
        mock.patch.object(
            combined_state_qa,
            "_exists_parquet",
            lambda path: path == "combined_path",
        ),
        mock.patch.object(
            combined_state_qa,
            "_read_parquet_df",
            lambda path, cols: frame[[c for c in cols if c in frame.columns]].copy(),
        ),
        mock.patch.object(
            combined_state_qa,
            "_read_manifest",
            lambda path: {
                "selected_fluxes": ["drained_total", "burned_total"],
            }
            if path == "combined_path"
            else None,
        ),
    ):
        result = combined_state_qa.evaluate_interval(
            model_version="1_0_0",
            run_name="run",
            run_date="20260510",
            interval="2021_2024",
            value_tol=1e-6,
            relative_tol=1e-6,
            area_value_tol=1e-6,
            area_relative_tol=1e-5,
        )

    assert result["pass"] is True
    assert result["decode_self_consistency"]["decode_drained_mismatch_rows"] == 0
    assert result["decode_self_consistency"]["decode_burned_mismatch_rows"] == 0
