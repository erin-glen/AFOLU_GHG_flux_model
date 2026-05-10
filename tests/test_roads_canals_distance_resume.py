import importlib
import unittest
from unittest import mock

from src.scripts.preprocessing.roads_canals.global_datasets import roads_io


distance_mod = importlib.import_module(
    "src.scripts.preprocessing.roads_canals.global_datasets.1_2_distance_from_presence_mosaic"
)


class RoadsCanalsDistanceResumeTests(unittest.TestCase):
    def test_open_rioxarray_with_retries_retries_transient_failure(self):
        attempts = []

        def flaky_open(path, **kwargs):
            attempts.append((path, kwargs))
            if len(attempts) == 1:
                raise RuntimeError("temporary /vsis3 read failure")
            return "opened"

        with (
            mock.patch.object(roads_io.rxr, "open_rasterio", flaky_open),
            mock.patch.object(roads_io.time, "sleep", lambda _: None),
        ):
            result = roads_io.open_rioxarray_with_retries(
                "/vsis3/bucket/key.tif",
                masked=True,
                attempts=2,
                delay_seconds=0,
            )

        self.assertEqual(result, "opened")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0][0], "/vsis3/bucket/key.tif")
        self.assertIs(attempts[0][1]["masked"], True)

    def test_distance_chunk_skips_existing_output_before_heavy_work(self):
        class ExistingObjectClient:
            def head_object(self, Bucket, Key):
                return {}

        def fail_if_called(*args, **kwargs):
            raise AssertionError("heavy processing should not run for existing outputs")

        with (
            mock.patch.object(distance_mod.roads_io, "ensure_s3_client", lambda: ExistingObjectClient()),
            mock.patch.object(distance_mod.uutil, "calc_chunk_length_pixels", lambda bounds: 4000),
            mock.patch.object(
                distance_mod.roads_io,
                "distance_raster_name",
                lambda tile_id, bounds, feature_type: "10N_000E__0_0_1_1__osm_roads_distance.tif",
            ),
            mock.patch.object(
                distance_mod.roads_io,
                "build_s3_uri",
                lambda feature_type, product, chunk_px, date_str, filename: (
                    f"s3://bucket/{filename}",
                    filename,
                ),
            ),
            mock.patch.object(distance_mod.roads_io, "local_product_dir", fail_if_called),
            mock.patch.object(distance_mod, "_mosaic_presence", fail_if_called),
        ):
            result = distance_mod._process_chunk_distance(
                tile_id="10N_000E",
                chunk_bounds=[0, 0, 1, 1],
                feature_type="osm_roads",
                date_str="20260509",
                halo_m=1000,
            ).compute(scheduler="single-threaded")

        self.assertEqual(result["status"], "skip_existing")
        self.assertEqual(
            result["s3"],
            ["s3://bucket/10N_000E__0_0_1_1__osm_roads_distance.tif"],
        )

    def test_log_batch_results_raises_on_reported_task_errors(self):
        with self.assertRaisesRegex(RuntimeError, "tile=10N_000E"):
            distance_mod._log_batch_results(
                (
                    {
                        "tile": "10N_000E",
                        "bounds": [0, 0, 1, 1],
                        "status": "error",
                        "s3": [],
                        "msgs": "mask_open_failed:boom",
                    },
                )
            )


if __name__ == "__main__":
    unittest.main()
