"""Emission factors for burned area calculations.

Values are keyed by ecozone and drainage state.
The order is [gef_co2, gef_co, gef_ch4, mass_burnt].
Low and high tables duplicate defaults and can be updated later.
"""

import numpy as np
from numba import types
from numba.typed import Dict

ZERO_ARRAY = np.zeros(4, dtype=np.float32)

DEFAULT = {
    'boreal_drained':   [1650.0, 110.0, 12.0, 250.0],
    'boreal_undrained': [1450.0, 90.0, 10.0, 75.0],
    'temperate_drained':   [1650.0, 110.0, 12.0, 200.0],
    'temperate_undrained': [1450.0, 90.0, 10.0, 50.0],
    'tropical_drained_crop_or_plantation': [1700.0, 200.0, 15.0, 150.0],
    'tropical_drained_other': [1600.0, 180.0, 14.0, 300.0],
    'tropical_undrained': [0.0, 0.0, 0.0, 0.0],
    'other': [0.0, 0.0, 0.0, 0.0],
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