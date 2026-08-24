from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

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


def _drained_soil_for_threshold(threshold_m, land_cover=None, **float_overrides):
    if land_cover is None:
        land_cover = cn.ipcc_codes["forest"]
    layers = {
        "peat": np.array([[1]], dtype=np.uint8),
        "land_cover": np.array([[land_cover]], dtype=np.uint8),
        "planted_forest_type": np.array([[0]], dtype=np.uint8),
        "extraction": np.array([[0]], dtype=np.uint8),
        "mangrove_extent": np.array([[0]], dtype=np.uint8),
        "tidal_marsh": np.array([[0]], dtype=np.uint8),
        "burned_area_combined_2001": np.array([[0]], dtype=np.uint8),
        "climate_domain": np.array([[cn.ecozone_codes["temperate"]]], dtype=np.int16),
        "descals_type": np.array([[0]], dtype=np.int16),
        "dadap": np.array([[0.0]], dtype=np.float32),
        "osm_roads": np.array([[0.0]], dtype=np.float32),
        "osm_canals": np.array([[0.0]], dtype=np.float32),
        "engert": np.array([[0.0]], dtype=np.float32),
        "grip": np.array([[0.0]], dtype=np.float32),
    }
    for key, value in float_overrides.items():
        layers[key] = np.array([[value]], dtype=np.float32)

    typed_uint8, typed_int16, _, typed_float32 = nu.create_typed_dicts(layers)
    out_uint32, _ = drainage_model.calculate_drainage_and_emissions(
        typed_uint8,
        typed_int16,
        typed_float32,
        defac.DEFAULT_TABLE,
        baf.DEFAULT_TABLE,
        False,
        threshold_m,
    )
    return int(out_uint32["drained_soil"][0, 0])


def test_osm_roads_use_configured_distance_threshold():
    assert _drained_soil_for_threshold(250, osm_roads=300.0) == 1
    assert _drained_soil_for_threshold(500, osm_roads=300.0) == 2


def test_osm_canals_use_inclusive_distance_threshold_boundary():
    assert _drained_soil_for_threshold(250, osm_canals=250.0) == 2


def test_grip_roads_use_configured_distance_threshold():
    assert _drained_soil_for_threshold(500, grip=750.0) == 1
    assert _drained_soil_for_threshold(750, grip=750.0) == 2


def test_dadap_drains_independent_of_distance_threshold():
    assert _drained_soil_for_threshold(250, dadap=1.0) == 2
    assert _drained_soil_for_threshold(750, dadap=1.0) == 2


def test_engert_drains_independent_of_distance_threshold():
    assert _drained_soil_for_threshold(250, engert=1.0) == 2
    assert _drained_soil_for_threshold(750, engert=1.0) == 2


def test_land_cover_nodata_cannot_be_routed_to_otherland():
    assert _drained_soil_for_threshold(500, land_cover=0, osm_roads=100.0) == 0


def test_pipeline_rejects_land_cover_nodata_on_organic_soil():
    layers = {
        "peat": np.array([[1, 0]], dtype=np.uint8),
        "land_cover": np.array([[0, 0]], dtype=np.uint8),
    }

    with pytest.raises(
        drainage_model.uu.RequiredInputRasterError,
        match="Refusing to route unknown class values to Otherland",
    ):
        drainage_model.validate_land_cover_on_organic_soil(
            layers,
            tile_id="00N_110E",
            bounds_string="test-window",
            iv_start=2011,
            iv_end=2015,
        )


def test_pipeline_rejects_unconfigured_climate_domain_source_code():
    layers = {
        "climate_domain": np.array([[1, 4]], dtype=np.int16),
        "land_cover": np.array([[0, cn.ipcc_codes["forest"]]], dtype=np.uint8),
    }

    with pytest.raises(
        drainage_model.uu.RequiredInputRasterError,
        match="outside its configured legend",
    ):
        drainage_model.validate_climate_domain_source_codes(
            layers,
            tile_id="00N_110E",
            bounds_string="test-window",
        )


def test_exported_organic_soil_is_binary_peat_presence():
    drained_soil = np.array([[0, 1, 2]], dtype=np.uint32)

    organic_soil = drainage_model.organic_soil_from_drained_soil(drained_soil)

    np.testing.assert_array_equal(
        organic_soil,
        np.array([[0, 1, 1]], dtype=np.uint8),
    )
    assert organic_soil.dtype == np.uint8


@pytest.mark.parametrize("threshold_m", [0, -1, 1000, np.inf])
def test_drainage_distance_threshold_validation(threshold_m):
    with pytest.raises(ValueError, match="drainage_distance_threshold_m"):
        drainage_model.validate_drainage_distance_threshold_m(threshold_m)
