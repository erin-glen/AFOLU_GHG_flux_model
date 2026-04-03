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
