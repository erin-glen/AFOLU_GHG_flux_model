import numpy as np

from src.scripts.utilities import burned_area_emission_factors as baf
from src.scripts.utilities import drainage_emission_factors as defac


EXPECTED_DRAINAGE_KEYS = {
    "boreal_forest_poor",
    "boreal_forest_rich",
    "boreal_grassland",
    "boreal_cropland",
    "boreal_extraction",
    "boreal_settlement",
    "boreal_wetland",
    "boreal_otherland",
    "temperate_forest",
    "temperate_grassland_poor",
    "temperate_grassland_rich",
    "temperate_cropland",
    "temperate_extraction",
    "temperate_settlement",
    "temperate_wetland",
    "temperate_otherland",
    "tropical_long_rotation",
    "tropical_short_rotation",
    "tropical_oil_palm",
    "tropical_sago_palm",
    "tropical_forest",
    "tropical_grassland",
    "tropical_cropland",
    "tropical_extraction",
    "tropical_settlement",
    "tropical_wetland",
    "tropical_otherland",
    "coastal_mangrove",
    "coastal_tidal_marsh",
}

EXPECTED_BURNED_KEYS = {
    "boreal_drained",
    "boreal_undrained",
    "temperate_drained",
    "temperate_undrained",
    "tropical_drained_crop_or_plantation",
    "tropical_drained_other",
    "tropical_undrained",
    "other",
}


def test_all_drainage_factor_variants_cover_every_model_route():
    for table in (defac.DEFAULT, defac.LOW, defac.HIGH):
        assert set(table) == EXPECTED_DRAINAGE_KEYS
        for values in table.values():
            array = np.asarray(values, dtype=np.float32)
            assert array.shape == (6,)
            assert np.isfinite(array).all()
            assert 0 <= array[5] <= 1


def test_all_burned_factor_variants_cover_every_model_route():
    for table in (baf.DEFAULT, baf.LOW, baf.HIGH):
        assert set(table) == EXPECTED_BURNED_KEYS
        for values in table.values():
            array = np.asarray(values, dtype=np.float32)
            assert array.shape == (4,)
            assert np.isfinite(array).all()
            assert (array >= 0).all()
