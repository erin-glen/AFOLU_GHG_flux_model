import unittest
import numpy as np
from src.scripts.utilities import numba_utilities as nu

class TestCalculateBurnedAreaEmissions(unittest.TestCase):
    def test_total_burned_emissions(self):
        burn_co2, burn_co, burn_ch4, total = nu.calculate_burned_area_emissions(
            np.float32(1.0),
            np.float32(1000.0),
            np.float32(1.0),
            np.float32(1.0),
            np.float32(2.0),
            np.float32(3.0),
            np.float32(2.0),
            np.float32(3.0),
        )
        self.assertTrue(np.isclose(burn_co2, 1.0))
        self.assertTrue(np.isclose(burn_co, 4.0))
        self.assertTrue(np.isclose(burn_ch4, 9.0))
        self.assertTrue(np.isclose(total, 14.0))

if __name__ == '__main__':
    unittest.main()