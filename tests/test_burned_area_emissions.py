from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

from src.scripts.utilities import burned_area_emission_factors as baf
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import drainage_emission_factors as defac
from src.scripts.utilities import numba_utilities as nu


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/scripts/core_model/0_drainage_emissions_model.py"
)
SPEC = spec_from_file_location("drainage_model", MODULE_PATH)
drainage_model = module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(drainage_model)


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


def test_burned_co_output_is_raw_co_not_co2e():
    layers = {
        "peat": np.array([[1]], dtype=np.uint8),
        "land_cover": np.array([[cn.ipcc_codes["forest"]]], dtype=np.uint8),
        "planted_forest_type": np.array([[0]], dtype=np.uint8),
        "extraction": np.array([[0]], dtype=np.uint8),
        "mangrove_extent": np.array([[0]], dtype=np.uint8),
        "tidal_marsh": np.array([[0]], dtype=np.uint8),
        "burned_area_combined_2001": np.array([[1]], dtype=np.uint8),
        "climate_domain": np.array([[cn.ecozone_codes["temperate"]]], dtype=np.int16),
        "descals_type": np.array([[0]], dtype=np.int16),
        "dadap": np.array([[0.0]], dtype=np.float32),
        "osm_roads": np.array([[0.0]], dtype=np.float32),
        "osm_canals": np.array([[0.0]], dtype=np.float32),
        "engert": np.array([[0.0]], dtype=np.float32),
        "grip": np.array([[0.0]], dtype=np.float32),
    }
    typed_uint8, typed_int16, _, typed_float32 = nu.create_typed_dicts(layers)

    _, out_float32 = drainage_model.calculate_drainage_and_emissions(
        typed_uint8,
        typed_int16,
        typed_float32,
        defac.DEFAULT_TABLE,
        baf.DEFAULT_TABLE,
        False,
    )

    assert "burned_co_Mg_CO_ha" in out_float32
    assert "burned_co_Mg_CO2e_ha" not in out_float32
    assert np.isclose(
        out_float32["burned_co_Mg_CO_ha"][0, 0],
        66.0 * 207.0 * 1e-3,
    )
