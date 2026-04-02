from pathlib import Path

import numpy as np

from src.scripts.utilities import constants_and_names as cn
from src.scripts.zonal_statistics import zonal_constants as zc


def test_pack_unpack_combined_state_roundtrip_matches_registered_codes() -> None:
    drained_codes = np.array([0, *map(int, sorted(zc.ALL_DRAINED_STATE_CODES))], dtype=np.uint32)
    burned_codes = np.array([0, *map(int, sorted(zc.ALL_BURNED_STATE_CODES))], dtype=np.uint32)
    drained_grid = np.repeat(drained_codes[:, None], burned_codes.shape[0], axis=1)
    burned_grid = np.repeat(burned_codes[None, :], drained_codes.shape[0], axis=0)

    packed = zc.pack_combined_state(drained_grid, burned_grid)
    drained_decoded, burned_decoded = zc.unpack_combined_state_to_legacy(packed)

    np.testing.assert_array_equal(drained_decoded, drained_grid)
    np.testing.assert_array_equal(burned_decoded, burned_grid)


def test_combined_state_group_values_cover_registry_outputs() -> None:
    group_values = zc.COMBINED_STATE_GROUP_VALUES
    assert group_values.dtype == np.uint32
    assert np.unique(group_values).size == group_values.size

    drained_codes = np.array([0, *map(int, sorted(zc.ALL_DRAINED_STATE_CODES))], dtype=np.uint32)
    burned_codes = np.array([0, *map(int, sorted(zc.ALL_BURNED_STATE_CODES))], dtype=np.uint32)
    drained_grid = np.repeat(drained_codes[:, None], burned_codes.shape[0], axis=1)
    burned_grid = np.repeat(burned_codes[None, :], drained_codes.shape[0], axis=0)
    packed = zc.pack_combined_state(drained_grid, burned_grid)

    assert set(np.unique(packed).tolist()).issubset(set(group_values.tolist()))


def test_new_write_contracts_use_combined_state_names() -> None:
    assert "combined_state" in cn.drainage_outputs_to_zarr
    assert "combined_state" in cn.drainage_output_dtypes
    assert "emissions_state" not in cn.drainage_outputs_to_zarr

    core_model_source = Path("src/scripts/core_model/0_drainage_emissions_model.py").read_text()
    assert 'outputs["combined_state"]' in core_model_source
    assert 'outputs["emissions_state"]' not in core_model_source

    zonal_source = Path("src/scripts/zonal_statistics/02_run_zonal_stats.py").read_text()
    assert '"combined_state_nodes"' in zonal_source
