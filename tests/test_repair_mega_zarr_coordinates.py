import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr
import zarr

from src.scripts.utilities import drainage_zarr_utilities as dzu
from src.scripts.utilities import repair_mega_zarr_coordinates as repair


class RepairMegaZarrCoordinatesTests(unittest.TestCase):
    def test_repair_preserves_data_arrays_and_refreshes_consolidated_metadata(self):
        expected_x = np.array([-1.5, -0.5, 0.5, 1.5], dtype="float64")
        expected_y = np.array([1.0, 0.0, -1.0], dtype="float64")

        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "mega.zarr"
            backup = Path(tmp_dir) / "coordinate_backup"
            group = zarr.open_group(str(source), mode="w", zarr_format=3)
            group.create_array(
                "x",
                data=expected_x.astype("float32"),
                chunks=(2,),
                fill_value=np.nan,
                attributes={"_FillValue": np.nan},
                dimension_names=("x",),
            )
            group.create_array(
                "y",
                data=expected_y.astype("float32"),
                chunks=(2,),
                fill_value=np.nan,
                attributes={"_FillValue": np.nan},
                dimension_names=("y",),
            )
            group.create_array(
                "year",
                data=np.array([2024], dtype="int32"),
                chunks=(1,),
                dimension_names=("year",),
            )
            group.create_array(
                "combined_state",
                shape=(1, 3, 4),
                dtype="uint32",
                chunks=(1, 3, 4),
                fill_value=0,
                dimension_names=("year", "y", "x"),
            )
            group["combined_state"][:] = np.arange(12, dtype="uint32").reshape(1, 3, 4)
            zarr.consolidate_metadata(str(source))

            with (
                patch.object(dzu, "global_grid_shape", return_value=(3, 4)),
                patch.object(
                    dzu,
                    "global_coords",
                    return_value=(expected_x, expected_y, 1.0),
                ),
            ):
                result = repair.inspect_or_repair(
                    str(source),
                    apply=True,
                    backup_path=str(backup),
                )

            self.assertTrue(result["consolidated_metadata_refreshed"])
            self.assertTrue(
                result["validation"]["consolidated_and_unconsolidated_reads_match"]
            )
            self.assertTrue(result["validation"]["non_coordinate_metadata_unchanged"])
            self.assertTrue((backup / "backup_manifest.json").exists())
            self.assertTrue((backup / "repair_result.json").exists())

            for consolidated in (None, True, False):
                dataset = xr.open_zarr(str(source), consolidated=consolidated)
                np.testing.assert_array_equal(dataset["x"].values, expected_x)
                np.testing.assert_array_equal(dataset["y"].values, expected_y)
                np.testing.assert_array_equal(
                    dataset["combined_state"].values,
                    np.arange(12, dtype="uint32").reshape(1, 3, 4),
                )

            repaired_group = zarr.open_group(str(source), mode="r")
            for coordinate in ("x", "y"):
                self.assertNotIn("_FillValue", repaired_group[coordinate].attrs)


if __name__ == "__main__":
    unittest.main()
