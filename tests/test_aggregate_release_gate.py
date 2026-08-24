from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/scripts/core_model/02_aggregate_soils_outputs.py"
)
SPEC = spec_from_file_location("aggregate_soils_release_gate", MODULE_PATH)
aggregate = module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(aggregate)


class _Logger:
    def warning(self, *args, **kwargs):
        pass


def _write_test_store(path: Path, status: str) -> None:
    ds = xr.Dataset(
        {"organic_soil": (("year", "y", "x"), np.zeros((2, 1, 1), dtype=np.uint8))},
        coords={"year": np.asarray([2020, 2024], dtype=np.int32), "y": [0.0], "x": [0.0]},
        attrs={"run_status": status},
    )
    ds.to_zarr(path, mode="w", zarr_format=3)


def test_aggregation_rejects_incomplete_new_model_store(tmp_path):
    zarr_path = tmp_path / "incomplete.zarr"
    _write_test_store(zarr_path, "running")

    with pytest.raises(RuntimeError, match="not marked complete"):
        aggregate.ready_mega_zarr_year_index(str(zarr_path), _Logger())


def test_aggregation_uses_actual_completed_store_year_coordinate(tmp_path):
    zarr_path = tmp_path / "complete.zarr"
    _write_test_store(zarr_path, "complete")

    assert aggregate.ready_mega_zarr_year_index(
        str(zarr_path),
        _Logger(),
    ) == [2020, 2024]
