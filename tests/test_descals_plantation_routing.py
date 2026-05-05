from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

from src.scripts.utilities import burned_area_emission_factors as baf
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import drainage_emission_factors as defac
from src.scripts.utilities import numba_utilities as nu
from src.scripts.zonal_statistics import zonal_constants as zc


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/scripts/core_model/0_drainage_emissions_model.py"
)
SPEC = spec_from_file_location("drainage_model", MODULE_PATH)
drainage_model = module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(drainage_model)


def _meaning(mapping, code):
    if int(code) == 0:
        return "unburned"
    return mapping[str(int(code)).zfill(zc.PAD_DIGITS)]


def _run_one_pixel(
    *,
    peat=1,
    planted_forest_type=0,
    descals_type=0,
    extraction=0,
    land_cover=None,
    burned=0,
):
    if land_cover is None:
        land_cover = cn.ipcc_codes["forest"]

    layers = {
        "peat": np.array([[peat]], dtype=np.uint8),
        "land_cover": np.array([[land_cover]], dtype=np.uint8),
        "planted_forest_type": np.array([[planted_forest_type]], dtype=np.uint8),
        "extraction": np.array([[extraction]], dtype=np.uint8),
        "mangrove_extent": np.array([[0]], dtype=np.uint8),
        "tidal_marsh": np.array([[0]], dtype=np.uint8),
        "burned_area_combined_2001": np.array([[burned]], dtype=np.uint8),
        "climate_domain": np.array([[cn.ecozone_codes["tropical"]]], dtype=np.int16),
        "descals_type": np.array([[descals_type]], dtype=np.int16),
        "dadap": np.array([[0.0]], dtype=np.float32),
        "osm_roads": np.array([[0.0]], dtype=np.float32),
        "osm_canals": np.array([[0.0]], dtype=np.float32),
        "engert": np.array([[0.0]], dtype=np.float32),
        "grip": np.array([[0.0]], dtype=np.float32),
    }
    td8, td16, _, td32f = nu.create_typed_dicts(layers)
    out_u32, _ = drainage_model.calculate_drainage_and_emissions(
        td8,
        td16,
        td32f,
        defac.DEFAULT_TABLE,
        baf.DEFAULT_TABLE,
        False,
        drainage_model.DEFAULT_DRAINAGE_DISTANCE_THRESHOLD_M,
    )
    return (
        int(out_u32["drained_soil"][0, 0]),
        _meaning(zc.DRAINED_STATE_NODE_MEANINGS, out_u32["drained_state"][0, 0]),
        _meaning(zc.BURNED_STATE_NODE_MEANINGS, out_u32["burned_state"][0, 0]),
    )


def test_descals_only_pixel_routes_to_drainage_plantation_oil_palm():
    soil, drained_label, _ = _run_one_pixel(descals_type=1)
    assert soil == 2
    assert drained_label == "peat_drained_plantation__tropical_oil_palm"


def test_descals_overrides_sdpt_short_rotation_to_oil_palm():
    _, drained_label, _ = _run_one_pixel(
        descals_type=1,
        planted_forest_type=cn.plantation_type_codes["short_rotation"],
    )
    assert drained_label == "peat_drained_plantation__tropical_oil_palm"


def test_descals_overrides_sdpt_long_rotation_to_oil_palm():
    _, drained_label, _ = _run_one_pixel(
        descals_type=1,
        planted_forest_type=cn.plantation_type_codes["long_rotation"],
    )
    assert drained_label == "peat_drained_plantation__tropical_oil_palm"


def test_descals_class_value_two_still_routes_to_oil_palm():
    _, drained_label, _ = _run_one_pixel(descals_type=2)
    assert drained_label == "peat_drained_plantation__tropical_oil_palm"


def test_sdpt_short_rotation_unchanged_when_descals_absent():
    _, drained_label, _ = _run_one_pixel(
        planted_forest_type=cn.plantation_type_codes["short_rotation"],
    )
    assert drained_label == "peat_drained_plantation__tropical_short_rotation"


def test_sdpt_long_rotation_unchanged_when_descals_absent():
    _, drained_label, _ = _run_one_pixel(
        planted_forest_type=cn.plantation_type_codes["long_rotation"],
    )
    assert drained_label == "peat_drained_plantation__tropical_long_rotation"


def test_tropical_burned_descals_pixel_routes_to_crop_or_plantation():
    _, _, burned_label = _run_one_pixel(descals_type=2, burned=1)
    assert burned_label == "tropical__drained_crop_or_plantation"


def test_extraction_precedence_overrides_plantation_in_tropical_drainage():
    _, drained_label, _ = _run_one_pixel(descals_type=2, extraction=1)
    assert drained_label == "peat_drained_extraction__tropical_extraction"


def test_extraction_layer_is_binarized_before_model_consumption():
    layers = {"extraction": np.array([[0, 1, 2, 7]], dtype=np.uint8)}
    drainage_model.binarize_extraction_layer(layers)
    assert layers["extraction"].dtype == np.uint8
    np.testing.assert_array_equal(
        layers["extraction"],
        np.array([[0, 1, 1, 1]], dtype=np.uint8),
    )
