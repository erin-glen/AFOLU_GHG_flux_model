import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import xarray as xr
from zarr.errors import ContainsGroupError
from zarr.storage import FsspecStore

from src.scripts.utilities import drainage_zarr_utilities as dzu


class _Logger:
    def info(self, *args, **kwargs):
        pass



class DrainageZarrUtilitiesTests(unittest.TestCase):
    def test_annual_year_index_contains_only_complete_global_inputs(self):
        self.assertEqual(dzu.full_model_year_index("annual"), [2024])

    def test_global_coords_use_canonical_float64_grid(self):
        x, y, resolution = dzu.global_coords()

        self.assertEqual(x.dtype, np.dtype("float64"))
        self.assertEqual(y.dtype, np.dtype("float64"))
        self.assertEqual(x.size, 1_440_000)
        self.assertEqual(y.size, 720_000)
        self.assertAlmostEqual(float(x[1] - x[0]), resolution, places=12)
        self.assertAlmostEqual(float(y[0] - y[1]), resolution, places=12)
        self.assertAlmostEqual(float(x[0]), -179.999875, places=12)
        self.assertAlmostEqual(float(x[-1]), 179.999875, places=12)
        self.assertAlmostEqual(float(y[0]), 89.999875, places=12)
        self.assertAlmostEqual(float(y[-1]), -89.999875, places=12)

    def test_make_zarr_store_uses_zarr_v3_fsspec_store_for_s3(self):
        fake_fs = SimpleNamespace(async_impl=True, asynchronous=True)

        with patch.object(dzu.fsspec, "filesystem", return_value=fake_fs) as filesystem:
            store = dzu.make_zarr_store(
                "s3://example-bucket/path/to/mega.zarr",
                read_only=True,
            )

        filesystem.assert_called_once_with("s3", anon=False, asynchronous=True)
        self.assertIsInstance(store, FsspecStore)
        self.assertEqual(store.path, "example-bucket/path/to/mega.zarr")
        self.assertTrue(store.read_only)

    def test_initialize_global_mega_zarr_local_smoke(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zarr_path = Path(tmp_dir) / "mega.zarr"

            with (
                patch.object(dzu, "global_coords", return_value=(np.arange(4), np.arange(3), 1.0)),
                patch.object(dzu, "global_grid_shape", return_value=(3, 4)),
            ):
                dzu.initialize_global_mega_zarr(
                    str(zarr_path),
                    outputs_to_zarr=["drained_soil", "drained_total_Mg_CO2e_ha_yr"],
                    years=[2024],
                    chunk_size_pixels=2,
                    interval_type="five_year",
                    logger=_Logger(),
                )
                ds = xr.open_zarr(str(zarr_path), consolidated=False)
                self.assertEqual(ds.attrs["run_status"], "initialized")
                dzu.set_mega_zarr_run_status(
                    str(zarr_path),
                    "complete",
                    successful_task_count=1,
                )
                completed = xr.open_zarr(str(zarr_path), consolidated=False)
                self.assertEqual(completed.attrs["run_status"], "complete")
                self.assertEqual(completed.attrs["successful_task_count"], 1)

        self.assertEqual(
            set(ds.data_vars),
            {"drained_soil", "drained_total_Mg_CO2e_ha_yr"},
        )
        self.assertEqual(ds.sizes, {"year": 1, "y": 3, "x": 4})

    def test_initialize_global_mega_zarr_refuses_existing_store(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zarr_path = Path(tmp_dir) / "mega.zarr"
            kwargs = {
                "zarr_path": str(zarr_path),
                "outputs_to_zarr": ["organic_soil"],
                "years": [2024],
                "chunk_size_pixels": 2,
                "interval_type": "five_year",
                "logger": _Logger(),
            }
            with (
                patch.object(
                    dzu,
                    "global_coords",
                    return_value=(np.arange(4), np.arange(3), 1.0),
                ),
                patch.object(dzu, "global_grid_shape", return_value=(3, 4)),
            ):
                dzu.initialize_global_mega_zarr(**kwargs)
                with self.assertRaises((FileExistsError, ValueError, ContainsGroupError)):
                    dzu.initialize_global_mega_zarr(**kwargs)


if __name__ == "__main__":
    unittest.main()
