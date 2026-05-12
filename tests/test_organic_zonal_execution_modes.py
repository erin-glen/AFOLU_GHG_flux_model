import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr
import pyarrow as pa
import pyarrow.dataset as ds


MODULE_PATH = Path("src/scripts/zonal_statistics/02_run_zonal_stats.py")
spec = importlib.util.spec_from_file_location("organic_zonal", MODULE_PATH)
organic_zonal = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(organic_zonal)


class FakeS3:
    def __init__(self, matches):
        self.matches = matches
        self.patterns = []

    def glob(self, pattern):
        self.patterns.append(pattern)
        return list(self.matches)


def test_auto_mode_resolves_tile_for_global_and_explicit_tiles() -> None:
    args_global = SimpleNamespace(
        tile_ids=None,
        bounding_box=None,
        execution_mode="auto",
        auto_tile_threshold_tiles=8,
    )
    plan_global = organic_zonal.resolve_execution_plan(args_global)
    assert plan_global["execution_mode_resolved"] == "tile"
    assert plan_global["tile_source"] == "canonical_global_roster"

    args_explicit = SimpleNamespace(
        tile_ids=["00N_010E,00N_020E"],
        bounding_box=None,
        execution_mode="auto",
        auto_tile_threshold_tiles=8,
    )
    plan_explicit = organic_zonal.resolve_execution_plan(args_explicit)
    assert plan_explicit["execution_mode_resolved"] == "tile"
    assert plan_explicit["tile_source"] == "explicit_ids"
    assert sorted(plan_explicit["tile_ids_to_process"]) == ["00N_010E", "00N_020E"]


def test_aggregated_tile_prefix_and_tile_id_extraction() -> None:
    prefix = organic_zonal.build_aggregated_tile_prefix(
        model_version="1_0_0",
        dataset="combined_state",
        run_name="ogh_biome_thresholds",
        interval_type="five_year",
        interval="2021_2024",
        pixel_resolution="40000_pixels",
        run_date="20260510",
    )
    assert prefix.endswith(
        "/version_1_0_0/combined_state/ogh_biome_thresholds/"
        "five_year_intervals/2021_2024/40000_pixels/20260510/"
    )
    assert organic_zonal.extract_tile_id_from_path(
        f"{prefix}70N_100W__combined_state__2021_2024.tif"
    ) == "70N_100W"


def test_interval_execution_plan_filters_to_aggregated_data_tiles() -> None:
    args = SimpleNamespace(
        model_version="1_0_0",
        run_name="ogh_biome_thresholds",
        run_date="20260510",
        interval_type="five_year",
        data_tile_filter="auto",
        data_tile_filter_dataset="combined_state",
        data_tile_filter_pixel_resolution="40000_pixels",
    )
    base_plan = {
        "execution_mode_resolved": "tile",
        "tile_ids_to_process": ["00N_010E", "00N_020E", "00N_030E"],
        "tile_count": 3,
        "tile_source": "canonical_global_roster",
        "explicit_tile_ids": [],
        "bbox": None,
        "exact_tile_mask_required": False,
        "roi_mode": "global",
        "is_global_request": True,
    }
    fs = FakeS3(
        [
            "s3://bucket/path/00N_020E__combined_state__2021_2024.tif",
            "s3://bucket/path/99N_999E__not_a_real_tile.tif",
        ]
    )
    logger = organic_zonal.logging.getLogger("test")
    plan = organic_zonal.resolve_interval_execution_plan(
        base_plan=base_plan,
        args=args,
        interval="2021_2024",
        fs_s3=fs,
        logger=logger,
    )
    assert plan["tile_ids_to_process"] == ["00N_020E"]
    assert plan["tile_count"] == 1
    assert plan["base_tile_count"] == 3
    assert plan["data_tile_filter_available_count"] == 2
    assert plan["data_tile_filter_dropped_count"] == 2
    assert plan["tile_source"] == "canonical_global_roster+aggregated_combined_state"


def test_finalize_interval_tile_outputs_reaggregates_and_decodes(tmp_path: Path) -> None:
    tile_stage_dir = tmp_path / "stage"
    tile_stage_dir.mkdir()
    combined = int(
        organic_zonal.zc.pack_combined_state(
            np.array([[np.uint32(0)]], dtype=np.uint32),
            np.array([[np.uint32(0)]], dtype=np.uint32),
        )[0, 0]
    )
    frame = pd.DataFrame(
        {
            "adm0": [840, 840],
            "combined_state_nodes": [combined, combined],
            "flux_type": ["drained_total_Mg_CO2e", "drained_total_Mg_CO2e"],
            "interval_end": [2024, 2024],
            "tile_id": ["00N_010E", "00N_020E"],
            "value": [1.5, 2.5],
        }
    )
    ds.write_dataset(pa.Table.from_pandas(frame, preserve_index=False), base_dir=str(tile_stage_dir), format="parquet")

    out = organic_zonal.finalize_interval_tile_outputs(tile_stage_dir)
    assert out.shape[0] == 1
    assert out.iloc[0]["value"] == 4.0
    assert "tile_id" not in out.columns
    assert "drained_state_nodes" in out.columns
    assert "burned_state_nodes" in out.columns


def test_resolve_combined_state_nodes_prefers_emissions_and_pack_fallback() -> None:
    x = [0.0, 1.0]
    y = [0.0, 1.0]
    ref = xr.DataArray(np.ones((2, 2), dtype=np.float32), dims=("y", "x"), coords={"x": x, "y": y})

    emissions_ds = xr.Dataset(
        {
            "emissions_state": xr.DataArray(np.ones((2, 2), dtype=np.uint32), dims=("y", "x"), coords={"x": x, "y": y}),
        }
    )
    logger = organic_zonal.logging.getLogger("test")
    arr = organic_zonal.resolve_combined_state_nodes(emissions_ds, ref, 0.49, False, "memory", "int", logger)
    assert arr.dtype == np.uint32

    pack_ds = xr.Dataset(
        {
            "drained_state": xr.DataArray(np.zeros((2, 2), dtype=np.uint32), dims=("y", "x"), coords={"x": x, "y": y}),
            "burned_state": xr.DataArray(np.zeros((2, 2), dtype=np.uint32), dims=("y", "x"), coords={"x": x, "y": y}),
        }
    )
    packed = organic_zonal.resolve_combined_state_nodes(pack_ds, ref, 0.49, False, "memory", "int", logger)
    assert packed.dtype == np.uint32


def test_script_no_longer_uses_drained_burned_output_prefixes() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '"combined_state"' in source
    assert "dest_d" not in source
    assert "dest_b" not in source


def test_build_bbox_mask_clips_to_bbox_subset() -> None:
    ref = xr.DataArray(
        np.ones((4, 4), dtype=np.float32),
        dims=("y", "x"),
        coords={"x": [0.5, 1.5, 2.5, 3.5], "y": [3.5, 2.5, 1.5, 0.5]},
    )
    mask = organic_zonal.build_bbox_mask(ref, [1.0, 1.0, 3.0, 3.0])
    # selected coords are x={1.5,2.5}, y={2.5,1.5}
    assert int(mask.sum()) == 4


def test_resolve_requested_contextual_groupers_canonical_deduped() -> None:
    resolved = organic_zonal.resolve_requested_contextual_groupers(
        ["drivers_of_loss", "KBA", "wdpa", "kba", "WDPA"]
    )
    assert resolved == ["wdpa", "kba", "drivers_of_loss"]


def test_default_flux_selection_includes_offsite_and_total_co2_sums_sources() -> None:
    assert "drained_co2_onsite" in organic_zonal.ordered_dataset_keys(None)
    assert "drained_co2_offsite" in organic_zonal.ordered_dataset_keys(None)
    assert "drained_co2" not in organic_zonal.ordered_dataset_keys(None)
    assert organic_zonal.ordered_dataset_keys(["drained_co2"]) == ["drained_co2_onsite"]
    assert (
        organic_zonal.FLUX_SPECS["drained_co2_onsite"]["label"]
        == "drained_co2_onsite_Mg_CO2"
    )
    assert organic_zonal.FLUX_DATASETS["drained_total_co2"]["source_var"] == [
        "drained_co2_Mg_CO2_ha_yr",
        "drained_co2_offsite_Mg_CO2_ha_yr",
    ]
    assert (
        organic_zonal.zc.ZONAL_FLUX_LABELS_BY_KEY["drained_co2_offsite"]
        == "drained_co2_offsite_Mg_CO2"
    )

    dsx = xr.Dataset(
        {
            "drained_co2_Mg_CO2_ha_yr": xr.DataArray(
                np.array([[2.0]], dtype=np.float32), dims=("y", "x")
            ),
            "drained_co2_offsite_Mg_CO2_ha_yr": xr.DataArray(
                np.array([[0.5]], dtype=np.float32), dims=("y", "x")
            ),
        }
    )
    arr = organic_zonal.dataset_from_mega(
        organic_zonal.FLUX_DATASETS["drained_total_co2"],
        dsx,
        dataset_key="drained_total_co2",
    )
    assert np.isclose(float(arr.values[0, 0]), 2.5)


def test_drivers_of_loss_contextual_grouper_registry() -> None:
    spec = organic_zonal.OPTIONAL_CONTEXTUAL_GROUPERS["drivers_of_loss"]
    assert spec["name"] == "drivers_of_TCL_1_km"
    assert spec["zarr_path"].endswith(
        "/contextual_layer_global_zarr/drivers_of_TCL_1_km/v20250414/"
        "update2023_20241218__run_20260507_fillValue_removed/"
        "drivers_of_TCL_1_km_20260507.zarr"
    )
    assert spec["dtype"] == np.uint8
    assert spec["expected_groups"].dtype == np.uint8
    assert spec["expected_groups"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]


def test_finalize_interval_tile_outputs_preserves_optional_contextual_columns(tmp_path: Path) -> None:
    tile_stage_dir = tmp_path / "stage_ctx"
    tile_stage_dir.mkdir()
    combined = int(
        organic_zonal.zc.pack_combined_state(
            np.array([[np.uint32(0)]], dtype=np.uint32),
            np.array([[np.uint32(0)]], dtype=np.uint32),
        )[0, 0]
    )
    frame = pd.DataFrame(
        {
            "adm0": [840, 840],
            "combined_state_nodes": [combined, combined],
            "flux_type": ["drained_total_Mg_CO2e", "drained_total_Mg_CO2e"],
            "interval_end": [2024, 2024],
            "tile_id": ["00N_010E", "00N_020E"],
            "wdpa": [0, 0],
            "kba": [1, 1],
            "value": [1.0, 2.0],
        }
    )
    ds.write_dataset(pa.Table.from_pandas(frame, preserve_index=False), base_dir=str(tile_stage_dir), format="parquet")
    out = organic_zonal.finalize_interval_tile_outputs(tile_stage_dir)
    assert "wdpa" in out.columns
    assert "kba" in out.columns
    assert out.iloc[0]["value"] == 3.0


def test_manifest_match_includes_contextual_grouper_identity() -> None:
    base = {
        "model_version": "1",
        "run_name": "r",
        "run_date": "20260101",
        "interval": "2020_2024",
        "interval_type": "five_year",
        "branch": "combined_state",
        "selected_fluxes": ["drained_total"],
        "selected_contextual_groupers": [],
        "contextual_grouper_paths": {},
        "align_tolerance_fraction": 0.49,
        "force_align": False,
        "roi_mode": "global",
        "bounding_box": None,
        "tile_ids": None,
        "execution_mode": "roi",
        "tile_source": "none",
        "tile_count": 0,
        "adm0_zarr_path": "a",
        "pixel_area_zarr_path": "p",
    }
    changed = dict(base)
    changed["selected_contextual_groupers"] = ["wdpa"]
    changed["contextual_grouper_paths"] = {"wdpa": "path"}
    assert organic_zonal.manifests_match(base, base)
    assert not organic_zonal.manifests_match(base, changed)
