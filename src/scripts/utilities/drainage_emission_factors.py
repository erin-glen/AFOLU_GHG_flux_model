"""Drainage emission factor tables for the organic-soils model.

Values are separated from the main model so they can be updated or
replaced easily.  `LOW` and `HIGH` tables are placeholders for
sensitivity analysis and should be filled with values from the
literature.
"""
from __future__ import annotations

import numpy as np

from . import constants_and_names as cn

# ---------------------------------------------------------------------
# base dictionaries ---------------------------------------------------
# Each ecozone contains per-land-cover factor sets. Values are in
# tonne C/ha/yr for CO2 and kg/ha/yr for other gases.
# ---------------------------------------------------------------------

DEFAULT = {
    "boreal": {
        "co2_offsite": 0.12,
        "forest_poor": {
            "co2": 0.25,
            "n2o": 0.22,
            "ch4_land": 7.0,
            "ch4_ditch": 217.0,
            "frac_ditch": 0.025,
        },
        "forest_rich": {
            "co2": 0.95,
            "n2o": 3.2,
            "ch4_land": 2.0,
            "ch4_ditch": 217.0,
            "frac_ditch": 0.025,
        },
        "grassland": {
            "co2": 5.7,
            "n2o": 9.5,
            "ch4_land": 1.4,
            "ch4_ditch": 1165.0,
            "frac_ditch": 0.05,
        },
        "cropland": {
            "co2": 7.9,
            "n2o": 13.0,
            "ch4_land": 0.0,
            "ch4_ditch": 1165.0,
            "frac_ditch": 0.05,
        },
        "extraction": {
            "co2": 2.8,
            "n2o": 0.30,
            "ch4_land": 6.1,
            "ch4_ditch": 542.0,
            "frac_ditch": 0.05,
        },
    },
    "temperate": {
        "co2_offsite": 0.31,
        "forest": {
            "co2": 2.6,
            "n2o": 2.8,
            "ch4_land": 2.5,
            "ch4_ditch": 217.0,
            "frac_ditch": 0.05,
        },
        "grassland_poor": {
            "co2": 5.3,
            "n2o": 4.3,
            "ch4_land": 1.8,
            "ch4_ditch": 1165.0,
            "frac_ditch": 0.05,
        },
        "grassland_rich": {
            "co2": 6.1,
            "n2o": 8.2,
            "ch4_land": 16.0,
            "ch4_ditch": 1165.0,
            "frac_ditch": 0.05,
        },
        "cropland": {
            "co2": 10.5,
            "n2o": 13.0,
            "ch4_land": 0.0,
            "ch4_ditch": 1165.0,
            "frac_ditch": 0.05,
        },
        "extraction": {
            "co2": 3.0,
            "n2o": 0.3,
            "ch4_land": 6.1,
            "ch4_ditch": 542.0,
            "frac_ditch": 0.05,
        },
    },
    "tropical": {
        "co2_offsite": 0.82,
        "common": {"ch4_ditch": 2259.0, "frac_ditch": 0.02},
        "long_rotation": {
            "co2": 15.0,
            "n2o": 2.4,
            "ch4_land": 2.7,
        },
        "short_rotation": {
            "co2": 20.0,
            "n2o": 2.4,
            "ch4_land": 2.7,
        },
        "oil_palm": {
            "co2": 11.0,
            "n2o": 1.2,
            "ch4_land": 0.0,
        },
        "sago_palm": {
            "co2": 1.5,
            "n2o": 3.3,
            "ch4_land": 26.2,
        },
        "forest": {
            "co2": 5.3,
            "n2o": 2.4,
            "ch4_land": 4.9,
        },
        "grassland": {
            "co2": 9.6,
            "n2o": 5.0,
            "ch4_land": 7.0,
        },
        "cropland": {
            "co2": 14.0,
            "n2o": 5.0,
            "ch4_land": 7.0,
        },
        "extraction": {
            "co2": 2.0,
            "n2o": 0.0,
            "ch4_land": 0.0,
        },
    },
}

# Placeholder structures for sensitivity analyses --------------------
LOW = {}
HIGH = {}

ZERO_FACTORS = {
    "co2": 0.0,
    "n2o": 0.0,
    "ch4_land": 0.0,
    "ch4_ditch": 0.0,
    "frac_ditch": 0.0,
    "co2_offsite": 0.0,
}


# ---------------------------------------------------------------------
# helper function -----------------------------------------------------

def get_factors(
    ecozone: int,
    land_cover: int,
    nutrient: int,
    extraction: int,
    planted_forest_type: int,
    scenario: str = "default",
):
    """Return drainage emission factors for a pixel."""
    table = {"default": DEFAULT, "low": LOW, "high": HIGH}.get(scenario, DEFAULT)

    if ecozone == cn.ecozone_codes["boreal"]:
        t = table.get("boreal", {})
        base = {"co2_offsite": t.get("co2_offsite", 0.0)}
        if land_cover == cn.ipcc_codes["forest"]:
            if nutrient == cn.nutrient_status_codes["poor"]:
                f = t.get("forest_poor", ZERO_FACTORS)
            elif nutrient == cn.nutrient_status_codes["rich"]:
                f = t.get("forest_rich", ZERO_FACTORS)
            else:
                f = ZERO_FACTORS
        elif land_cover == cn.ipcc_codes["grassland"]:
            f = t.get("grassland", ZERO_FACTORS)
        elif land_cover == cn.ipcc_codes["cropland"]:
            f = t.get("cropland", ZERO_FACTORS)
        elif extraction > 0:
            f = t.get("extraction", ZERO_FACTORS)
        else:
            f = ZERO_FACTORS
        return {**f, **base}

    elif ecozone == cn.ecozone_codes["temperate"]:
        t = table.get("temperate", {})
        base = {"co2_offsite": t.get("co2_offsite", 0.0)}
        if land_cover == cn.ipcc_codes["forest"]:
            f = t.get("forest", ZERO_FACTORS)
        elif land_cover == cn.ipcc_codes["grassland"]:
            if nutrient == cn.nutrient_status_codes["poor"]:
                f = t.get("grassland_poor", ZERO_FACTORS)
            elif nutrient == cn.nutrient_status_codes["rich"]:
                f = t.get("grassland_rich", ZERO_FACTORS)
            else:
                f = ZERO_FACTORS
        elif land_cover == cn.ipcc_codes["cropland"]:
            f = t.get("cropland", ZERO_FACTORS)
        elif extraction > 0:
            f = t.get("extraction", ZERO_FACTORS)
        else:
            f = ZERO_FACTORS
        return {**f, **base}

    elif ecozone == cn.ecozone_codes["tropical"]:
        t = table.get("tropical", {})
        base = {
            "co2_offsite": t.get("co2_offsite", 0.0),
            "ch4_ditch": t.get("common", {}).get("ch4_ditch", 0.0),
            "frac_ditch": t.get("common", {}).get("frac_ditch", 0.0),
        }
        if planted_forest_type == cn.plantation_type_codes["long_rotation"]:
            f = t.get("long_rotation", ZERO_FACTORS)
        elif planted_forest_type == cn.plantation_type_codes["short_rotation"]:
            f = t.get("short_rotation", ZERO_FACTORS)
        elif planted_forest_type == cn.plantation_type_codes["oil_palm"]:
            f = t.get("oil_palm", ZERO_FACTORS)
        elif planted_forest_type == cn.plantation_type_codes["sago_palm"]:
            f = t.get("sago_palm", ZERO_FACTORS)
        elif land_cover == cn.ipcc_codes["forest"]:
            f = t.get("forest", ZERO_FACTORS)
        elif land_cover == cn.ipcc_codes["grassland"]:
            f = t.get("grassland", ZERO_FACTORS)
        elif land_cover == cn.ipcc_codes["cropland"]:
            f = t.get("cropland", ZERO_FACTORS)
        elif extraction > 0:
            f = t.get("extraction", ZERO_FACTORS)
        else:
            f = ZERO_FACTORS
        return {**f, **base}

    return ZERO_FACTORS

__all__ = ["DEFAULT", "LOW", "HIGH", "get_factors", "ZERO_FACTORS"]