import numpy as np

from src.scripts.utilities import numba_utilities as nu


def test_burned_area_emissions_use_fuel_consumed_without_extra_cf():
    fuel_consumed = np.float32(155.0)
    gef_co2 = np.float32(464.0 * (44.0 / 12.0))
    gef_co = np.float32(210.0)
    gef_ch4 = np.float32(21.0)
    gwp_ch4 = np.float32(27.0)

    burn_co2, burn_co, burn_ch4, burn_total = nu.calculate_burned_area_emissions(
        fuel_consumed,
        gef_co2,
        gef_co,
        gef_ch4,
        gwp_ch4,
    )

    expected_co2 = fuel_consumed * gef_co2 * 1e-3
    expected_co = fuel_consumed * gef_co * 1e-3
    expected_ch4 = fuel_consumed * gef_ch4 * 1e-3 * gwp_ch4

    assert np.isclose(burn_co2, expected_co2)
    assert np.isclose(burn_co, expected_co)
    assert np.isclose(burn_ch4, expected_ch4)
    assert np.isclose(burn_total, expected_co2 + expected_ch4)
    assert not np.isclose(burn_co2, expected_co2 * 0.75)
