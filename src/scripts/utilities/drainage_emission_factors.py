"""Drainage emission factors lookup tables.

Values are keyed by ecozone and land-cover combinations.
All values are in tonnes per hectare per year.
The order is [co2, n2o, ch4_land, ch4_ditch, co2_offsite, frac_ditch].
Low and high tables duplicate defaults and can be updated.
"""

import numpy as np
from numba import types
from numba.typed import Dict

ZERO_ARRAY = np.zeros(6, dtype=np.float32)

DEFAULT = {
    'boreal_forest_poor':        [0.25, 0.22, 7.0, 217.0, 0.12, 0.025],
    'boreal_forest_rich':        [0.95, 3.2,  2.0, 217.0, 0.12, 0.025],
    'boreal_grassland':          [5.7,  9.5,  1.4, 1165.0,0.12, 0.05],
    'boreal_cropland':           [7.9,  13.0, 0.0, 1165.0,0.12, 0.05],
    'boreal_extraction':         [2.8,  0.3,  6.1, 542.0, 0.12, 0.05],

    'temperate_forest':          [2.6,  2.8,  2.5, 217.0, 0.31, 0.05],
    'temperate_grassland_poor':  [5.3,  4.3,  1.8, 1165.0,0.31, 0.05],
    'temperate_grassland_rich':  [6.1,  8.2, 16.0, 1165.0,0.31, 0.05],
    'temperate_cropland':        [10.5,13.0, 0.0, 1165.0,0.31, 0.05],
    'temperate_extraction':      [3.0,  0.3,  6.1, 542.0, 0.31, 0.05],

    'tropical_long_rotation':    [15.0,2.4,  2.7, 2259.0,0.82, 0.02],
    'tropical_short_rotation':   [20.0,2.4,  2.7, 2259.0,0.82, 0.02],
    'tropical_oil_palm':         [11.0,1.2,  0.0, 2259.0,0.82, 0.02],
    'tropical_sago_palm':        [1.5, 3.3, 26.2, 2259.0,0.82, 0.02],
    'tropical_forest':           [5.3, 2.4,  4.9, 2259.0,0.82, 0.02],
    'tropical_grassland':        [9.6, 5.0,  7.0, 2259.0,0.82, 0.02],
    'tropical_cropland':         [14.0,5.0,  7.0, 2259.0,0.82, 0.02],
    'tropical_extraction':       [2.0, 0.0,  0.0, 2259.0,0.82, 0.02],
}

LOW = DEFAULT.copy()
HIGH = DEFAULT.copy()


def _to_typed(table: dict) -> Dict:
    d = Dict.empty(key_type=types.unicode_type, value_type=types.float32[:])
    for k, vals in table.items():
        d[k] = np.array(vals, dtype=np.float32)
    return d

DEFAULT_TABLE = _to_typed(DEFAULT)
LOW_TABLE = _to_typed(LOW)
HIGH_TABLE = _to_typed(HIGH)