import unittest
import numpy as np
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import numba_utilities as nu


class TestUniversalUtilities(unittest.TestCase):
    def test_get_chunk_bounds(self):
        bounds = [0, 0, 20, 20]
        expected = [
            [0, 0, 10, 10],
            [10, 0, 20, 10],
            [0, 10, 10, 20],
            [10, 10, 20, 20],
        ]
        result = uu.get_chunk_bounds(bounds, 10)
        self.assertEqual(result, expected)

    def test_get_10x10_tile_bounds(self):
        self.assertEqual(uu.get_10x10_tile_bounds("02N_010E"), (10, -8, 20, 2))
        self.assertEqual(uu.get_10x10_tile_bounds("10S_050W"), (-50, -20, -40, -10))

    def test_boundstr_and_length(self):
        b = [0, 0, 10, 10]
        self.assertEqual(uu.boundstr(b), "0_0_10_10")
        self.assertEqual(uu.calc_chunk_length_pixels(b), 40000)

    def test_xy_to_tile_id(self):
        self.assertEqual(uu.xy_to_tile_id(15.2, 12.8), "20N_010E")
        self.assertEqual(uu.xy_to_tile_id(-5.0, -3.0), "00N_010W")

    def test_fill_missing_input_layers_with_no_data(self):
        layers = {"a": np.ones((2, 2), dtype=np.uint8)}
        uint8_list = ["a", "b"]
        int16_list = ["c"]
        int32_list = []
        float32_list = ["d"]
        out = uu.fill_missing_input_layers_with_no_data(
            layers, uint8_list, int16_list, int32_list, float32_list,
            "bstr", "tid", False, None
        )
        self.assertIn("b", out)
        self.assertTrue((out["b"] == 0).all() and out["b"].dtype == np.uint8)
        self.assertIn("c", out)
        self.assertEqual(out["c"].dtype, np.int16)
        self.assertIn("d", out)
        self.assertEqual(out["d"].dtype, np.float32)


class TestNumbaUtilities(unittest.TestCase):
    def test_accrete_node(self):
        self.assertEqual(nu.accrete_node(1, 3), 13)

    def test_create_typed_dicts(self):
        layers = {
            "u8": np.zeros((1, 1), dtype=np.uint8),
            "i16": np.zeros((1, 1), dtype=np.int16),
            "i32": np.zeros((1, 1), dtype=np.int32),
            "f32": np.zeros((1, 1), dtype=np.float32),
        }
        d_u8, d_i16, d_i32, d_f32 = nu.create_typed_dicts(layers)
        self.assertIn("u8", d_u8)
        self.assertIn("i16", d_i16)
        self.assertIn("i32", d_i32)
        self.assertIn("f32", d_f32)

    def test_calculate_drainage_emissions_co2e(self):
        args = (
            np.float32(1.0), np.float32(2.0), np.float32(3.0), np.float32(4.0),
            np.float32(0.5), np.float32(0.1),
            np.float32(3.67), np.float32(1.571), np.float32(265.0), np.float32(28.0),
            np.float32(2.0)
        )
        result = nu.calculate_drainage_emissions_co2e(*args)
        expected_total = (
            (1.0 * 3.67) + (2.0 * 1.571 * 265.0 / 1000.0) +
            (3.0 / 1000.0 * 28.0) + (4.0 / 1000.0 * 28.0 * 0.1) + (0.5 * 3.67)
        ) * 2.0
        self.assertTrue(np.isclose(result[-1], expected_total))


if __name__ == "__main__":
    unittest.main()