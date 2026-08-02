from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.scripts.zonal_statistics import rerun_wdpa_affected_tiles as rerun


def test_embedded_tile_manifests_are_exact_and_smoke_is_representative():
    assert len(rerun.FULL_TILE_IDS) == 70
    assert len(set(rerun.FULL_TILE_IDS)) == 70
    assert set(rerun.SMOKE_TILE_IDS).issubset(rerun.FULL_TILE_IDS)
    assert rerun.SMOKE_TILE_IDS == ("10N_080W", "60N_020E", "10N_120E")


def test_build_zonal_argv_is_isolated_and_runs_all_five_periods():
    argv = rerun.build_zonal_argv(
        mode="smoke",
        cluster_name="organic_soils_zonal_tests",
        output_run_name=rerun.SMOKE_OUTPUT_RUN_NAME,
        output_run_date="20260731",
        local_output=None,
    )
    joined = " ".join(argv)
    assert f"--run_name {rerun.INPUT_RUN_NAME}" in joined
    assert f"--run_date {rerun.INPUT_RUN_DATE}" in joined
    assert f"--output_run_name {rerun.SMOKE_OUTPUT_RUN_NAME}" in joined
    assert "--output_run_date 20260731" in joined
    assert "/mnt/c/tmp/afolu/wdpa_zonal_rerun/" in joined
    assert "--contextual_groupers all" in joined
    assert "--execution_mode tile" in joined
    assert "--interval_end_years 2005 2010 2015 2020 2024" in joined
    assert "--keep_local" in argv
    assert "--keep_tile_stage" in argv
    assert ",".join(rerun.SMOKE_TILE_IDS) in argv


def test_current_contextual_profile_matches_june_with_equivalent_river(monkeypatch):
    exact_paths = {
        "wdpa": "s3://baseline/wdpa",
        "landmark": "s3://baseline/landmark",
        "primary_forest": "s3://baseline/primary",
        "kba": "s3://baseline/kba",
        "drivers_of_loss": "s3://baseline/drivers",
    }
    module = SimpleNamespace(
        OPTIONAL_CONTEXTUAL_GROUPERS={
            **{key: {"zarr_path": path, "name": key} for key, path in exact_paths.items()},
            "river_basins": {
                "zarr_path": rerun.CURRENT_RIVER_BASINS,
                "name": "river_basins",
            },
        }
    )
    module.OPTIONAL_CONTEXTUAL_GROUPERS["primary_forest"]["name"] = (
        "starting_composite_primary_forest"
    )
    manifest = {
        "selected_contextual_groupers": [
            "wdpa",
            "landmark",
            "primary_forest",
            "kba",
            "river_basins",
            "drivers_of_loss",
        ],
        "contextual_grouper_paths": {
            **exact_paths,
            "river_basins": rerun.BASELINE_RIVER_BASINS,
        },
    }
    monkeypatch.setattr(rerun, "baseline_manifest", lambda fs, interval: manifest)
    rerun.validate_current_contextual_profile(module, object())


def test_prepare_sources_mode_has_been_removed():
    choices = rerun.parse_args(["--mode", "smoke"]).mode
    assert choices == "smoke"


class _ExistingDestinationFS:
    @staticmethod
    def exists(path):
        return True


def test_existing_isolated_destination_requires_explicit_resume():
    with pytest.raises(RuntimeError, match="without --resume"):
        rerun.assert_isolated_destination(
            _ExistingDestinationFS(),
            output_run_name=rerun.FULL_OUTPUT_RUN_NAME,
            output_run_date="20260731",
        )

    rerun.assert_isolated_destination(
        _ExistingDestinationFS(),
        output_run_name=rerun.FULL_OUTPUT_RUN_NAME,
        output_run_date="20260731",
        resume=True,
    )


def test_resume_never_allows_a_protected_destination():
    with pytest.raises(RuntimeError, match="protected"):
        rerun.assert_isolated_destination(
            _ExistingDestinationFS(),
            output_run_name=rerun.BASELINE_RUN_NAME,
            output_run_date=rerun.BASELINE_RUN_DATE,
            resume=True,
        )


def test_resume_flag_is_explicit():
    assert rerun.parse_args(["--mode", "full"]).resume is False
    assert rerun.parse_args(["--mode", "full", "--resume"]).resume is True


def test_coiled_execution_requires_wsl_linux(monkeypatch):
    monkeypatch.setattr(rerun.platform, "system", lambda: "Windows")

    with pytest.raises(RuntimeError, match="WSL/Linux"):
        rerun.require_linux_for_coiled_execution(execute=True)

    rerun.require_linux_for_coiled_execution(execute=False)
