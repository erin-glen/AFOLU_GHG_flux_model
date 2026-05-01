from src.scripts.core_model.sequence_runs import run_standard_500m_pipeline as pipeline
from src.scripts.utilities import constants_and_names as cn


def _command_by_label(commands, label):
    return dict(commands)[label]


def test_standard_pipeline_uses_matching_run_date_and_current_modules():
    commands = pipeline.build_commands(
        run_date="20260417",
        interval_end_years=["2024"],
        model_version=cn.model_version_underscore,
    )

    drainage_cmd = _command_by_label(commands, "0_drainage_emissions_model[gfw]")
    aggregate_cmd = _command_by_label(commands, "02_aggregate_soils_outputs[gfw]")
    zonal_cmd = _command_by_label(commands, "02_run_zonal_stats[gfw]")

    assert drainage_cmd[drainage_cmd.index("--run_date") + 1] == "20260417"
    assert aggregate_cmd[aggregate_cmd.index("--output_date") + 1] == "20260417"
    assert aggregate_cmd[2] == "src.scripts.core_model.02_aggregate_soils_outputs"
    assert aggregate_cmd[aggregate_cmd.index("--interval_end_years") + 1] == "2024"
    assert zonal_cmd[zonal_cmd.index("--run_date") + 1] == "20260417"
    assert zonal_cmd[zonal_cmd.index("--model_version") + 1] == cn.model_version_underscore
    assert "--zarr_chunk_size_pixels" not in zonal_cmd


def test_standard_pipeline_legacy_stage_points_to_legacy_module():
    commands = pipeline.build_commands(
        run_date="20260417",
        include_legacy_per_pixel_stage=True,
    )

    legacy_cmd = _command_by_label(commands, "legacy_2_per_pixel_soils_outputs[gfw]")

    assert legacy_cmd[2] == "src.scripts.core_model.legacy.2_per_pixel_soils_outputs"
    assert legacy_cmd[legacy_cmd.index("--output_date") + 1] == "20260417"


def test_standard_pipeline_log_dir_defaults_to_afolu_namespace():
    assert pipeline.DEFAULT_LOG_DIR.as_posix().endswith("/afolu/logs/pipelines/standard_500m")
