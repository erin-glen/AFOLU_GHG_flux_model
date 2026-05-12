import numpy as np

"""Node code utilities for organic soils zonal statistics."""

PAD_DIGITS = 8
# Backward compatibility: historical callers refer to STATE_CODE_PAD_DIGITS
# when zfill-ing emitted node codes. Keep the public alias to avoid breaking
# downstream scripts (e.g., pub_common/pub_assets).
STATE_CODE_PAD_DIGITS = PAD_DIGITS

ZONAL_FLUX_LABELS_BY_KEY: dict[str, str] = {
    "drained_total": "drained_total_Mg_CO2e",
    "drained_co2_onsite": "drained_co2_onsite_Mg_CO2",
    # Compatibility for archived zonal parquets/manifests.
    "drained_co2": "drained_co2_Mg_CO2",
    "drained_co2_offsite": "drained_co2_offsite_Mg_CO2",
    "drained_n2o": "drained_n2o_Mg_CO2e",
    "drained_total_co2": "drained_total_co2_Mg_CO2",
    "drained_total_ch4": "drained_total_ch4_Mg_CO2e",
    "burned_total": "burned_total_Mg_CO2e",
    "burned_total_co2": "burned_total_co2_Mg_CO2",
    "burned_total_ch4": "burned_total_ch4_Mg_CO2e",
}

def _pad_right(code: str) -> str:
    """Right‑pad ``code`` with zeros to the standard length."""
    return code.ljust(PAD_DIGITS, "0")

_drain_root = {
    "11": "peat_drained_primary_infra",          # dadap / canals
    "12": "peat_drained_secondary_infra",        # roads / ENGERT / GRIP
    "13": "peat_drained_cropland_settlement",
    "14": "peat_drained_plantation",
    "15": "peat_drained_extraction",
    "16": "peat_undrained",
    "0":  "non_peat",
}

_classification_suffix_labels = {
    "": "",
    "91": "__coastal_mangrove",
    "92": "__coastal_tidal_marsh",
}

_emissions_by_suffix = {
    "": {
        # BOREAL
        "111": "boreal_extraction",
        "1131": "boreal_forest_poor",
        "1132": "boreal_forest_rich",
        "114": "boreal_grassland",
        "115": "boreal_cropland",
        "116": "boreal_settlement",
        "117": "boreal_wetland",
        "118": "boreal_otherland",
        # TEMPERATE
        "121": "temperate_extraction",
        "123": "temperate_forest",
        "1241": "temperate_grassland_poor",
        "1242": "temperate_grassland_rich",
        "125": "temperate_cropland",
        "126": "temperate_settlement",
        "127": "temperate_wetland",
        "128": "temperate_otherland",
        # TROPICAL
        "131": "tropical_extraction",
        "1321": "tropical_long_rotation",
        "1322": "tropical_short_rotation",
        "1323": "tropical_oil_palm",
        "133": "tropical_forest",
        "134": "tropical_grassland",
        "135": "tropical_cropland",
        "136": "tropical_settlement",
        "137": "tropical_wetland",
        "138": "tropical_otherland",
    },
    "91": {
        "1191": "boreal_coastal_mangrove",
        "1291": "temperate_coastal_mangrove",
        "1391": "tropical_coastal_mangrove",
        "191": "other_domain_coastal_mangrove",
        # Drainage model emits ``…91`` for non‑domain coastal cases (no climate digit).
        "91": "coastal_mangrove",
    },
    "92": {
        "1192": "boreal_coastal_tidal_marsh",
        "1292": "temperate_coastal_tidal_marsh",
        "1392": "tropical_coastal_tidal_marsh",
        "192": "other_domain_coastal_tidal_marsh",
        # Drainage model emits ``…92`` for non‑domain coastal cases (no climate digit).
        "92": "coastal_tidal_marsh",
    },
}

# Ecozone digits appended to undrained-peat codes by the drainage model.
# No EF lookup is performed for undrained peat (EF = 0); the digit only carries
# climate-domain info through to the zonal output.
_undrained_ecozones = {
    "1": "boreal",
    "2": "temperate",
    "3": "tropical",
    "4": "other_domain",
}


def _build_drained_state_mapping() -> dict[str, str]:
    mapping: dict[str, str] = {}

    # Fallback (non-domain) codes for root "16" and the non_peat root "0".
    # 16000000 / 16910000 / 16920000 should not normally be emitted by the
    # current encoder, but are kept for safety and for legacy outputs.
    for root in ("16", "0"):
        base_label = _drain_root[root]
        for suffix, suffix_label in _classification_suffix_labels.items():
            if suffix:
                code = suffix if root == "0" else root + suffix
            else:
                code = root
            mapping[_pad_right(code)] = base_label + suffix_label

    # Domain-qualified undrained-peat codes: root "16" + optional coastal
    # suffix + ecozone digit.
    for suffix, suffix_label in _classification_suffix_labels.items():
        coastal_part = suffix_label.lstrip("_")
        class_code = "16" + suffix
        for ecozone_digit, ecozone_label in _undrained_ecozones.items():
            full_code = class_code + ecozone_digit
            if coastal_part:
                meaning = f"peat_undrained__{ecozone_label}_{coastal_part}"
            else:
                meaning = f"peat_undrained__{ecozone_label}"
            mapping[_pad_right(full_code)] = meaning

    for root in ("11", "12", "13", "14", "15"):
        base_label = _drain_root[root]
        for suffix in _classification_suffix_labels.keys():
            class_code = root + suffix
            for emit_code, emit_label in _emissions_by_suffix[suffix].items():
                mapping[_pad_right(class_code + emit_code)] = (
                    f"{base_label}__{emit_label}"
                )

    return mapping


DRAINED_STATE_NODE_MEANINGS = _build_drained_state_mapping()

_burn_state_labels = {
    "111": "boreal__drained",
    "112": "boreal__undrained",
    "221": "temperate__drained",
    "222": "temperate__undrained",
    "331": "tropical__drained_crop_or_plantation",
    "332": "tropical__drained_other",
    "333": "tropical__undrained",
    "44": "other_domain__other",
}

BURNED_STATE_NODE_MEANINGS: dict[str, str] = {
    _pad_right(code): label for code, label in _burn_state_labels.items()
}

ALL_DRAINED_STATE_CODES = frozenset(DRAINED_STATE_NODE_MEANINGS.keys())
ALL_BURNED_STATE_CODES = frozenset(BURNED_STATE_NODE_MEANINGS.keys())

# ---------------------------------------------------------------------
# Unified combined-state packing (uint32)
# ---------------------------------------------------------------------
# bits 0-7: drained_state_id
# bits 8-11: burned_state_id
# bit 12: has_drainage_component
# bit 13: has_burned_component
COMBINED_STATE_DRAINED_BITS = 8
COMBINED_STATE_BURNED_BITS = 4
COMBINED_STATE_DRAINED_MASK = (1 << COMBINED_STATE_DRAINED_BITS) - 1
COMBINED_STATE_BURNED_MASK = (1 << COMBINED_STATE_BURNED_BITS) - 1
COMBINED_STATE_BURNED_SHIFT = COMBINED_STATE_DRAINED_BITS
COMBINED_STATE_HAS_DRAINED_BIT = 12
COMBINED_STATE_HAS_BURNED_BIT = 13

_SORTED_DRAINED_CODES = sorted(ALL_DRAINED_STATE_CODES)
_SORTED_BURNED_CODES = sorted(ALL_BURNED_STATE_CODES)

DRAINED_STATE_CODE_TO_ID = {code: idx + 1 for idx, code in enumerate(_SORTED_DRAINED_CODES)}
BURNED_STATE_CODE_TO_ID = {code: idx + 1 for idx, code in enumerate(_SORTED_BURNED_CODES)}
DRAINED_STATE_ID_TO_CODE = {idx: code for code, idx in DRAINED_STATE_CODE_TO_ID.items()}
BURNED_STATE_ID_TO_CODE = {idx: code for code, idx in BURNED_STATE_CODE_TO_ID.items()}


def pack_combined_state(drained_code: np.ndarray, burned_code: np.ndarray) -> np.ndarray:
    """Pack drained/burned node-code rasters into one uint32 combined-state raster."""
    out = np.zeros(drained_code.shape, dtype=np.uint32)
    for code_str, idx in DRAINED_STATE_CODE_TO_ID.items():
        code = np.uint32(int(code_str))
        mask = drained_code == code
        if np.any(mask):
            out[mask] |= np.uint32(idx)
            out[mask] |= np.uint32(1 << COMBINED_STATE_HAS_DRAINED_BIT)

    for code_str, idx in BURNED_STATE_CODE_TO_ID.items():
        code = np.uint32(int(code_str))
        mask = burned_code == code
        if np.any(mask):
            out[mask] |= np.uint32(idx << COMBINED_STATE_BURNED_SHIFT)
            out[mask] |= np.uint32(1 << COMBINED_STATE_HAS_BURNED_BIT)

    return out


def unpack_combined_state_to_legacy(combined_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode combined-state raster back to drained/burned node-code rasters (uint32)."""
    drained_id = (combined_state & np.uint32(COMBINED_STATE_DRAINED_MASK)).astype(np.uint16)
    burned_id = (
        (combined_state >> np.uint32(COMBINED_STATE_BURNED_SHIFT))
        & np.uint32(COMBINED_STATE_BURNED_MASK)
    ).astype(np.uint16)

    drained_out = np.zeros(combined_state.shape, dtype=np.uint32)
    burned_out = np.zeros(combined_state.shape, dtype=np.uint32)

    for idx, code_str in DRAINED_STATE_ID_TO_CODE.items():
        mask = drained_id == np.uint16(idx)
        if np.any(mask):
            drained_out[mask] = np.uint32(int(code_str))
    for idx, code_str in BURNED_STATE_ID_TO_CODE.items():
        mask = burned_id == np.uint16(idx)
        if np.any(mask):
            burned_out[mask] = np.uint32(int(code_str))

    return drained_out, burned_out

# Import-time bit-space assertions for packed combined-state ids.
assert len(DRAINED_STATE_CODE_TO_ID) <= (1 << COMBINED_STATE_DRAINED_BITS), (
    "drained state registry exceeds 8-bit packed id capacity"
)
assert len(BURNED_STATE_CODE_TO_ID) <= (1 << COMBINED_STATE_BURNED_BITS), (
    "burned state registry exceeds 4-bit packed id capacity"
)

COMBINED_STATE_GROUP_VALUES = np.array(
    sorted(
        {
            0,
            *[
                int(pack_combined_state(
                    np.array([[np.uint32(int(d_code))]], dtype=np.uint32),
                    np.array([[np.uint32(int(b_code))]], dtype=np.uint32),
                )[0, 0])
                for d_code in ["0", *sorted(ALL_DRAINED_STATE_CODES)]
                for b_code in ["0", *sorted(ALL_BURNED_STATE_CODES)]
            ],
        }
    ),
    dtype=np.uint32,
)

# One-release compatibility aliases (temporary).
EMISSIONS_STATE_DRAINED_BITS = COMBINED_STATE_DRAINED_BITS
EMISSIONS_STATE_BURNED_BITS = COMBINED_STATE_BURNED_BITS
EMISSIONS_STATE_DRAINED_MASK = COMBINED_STATE_DRAINED_MASK
EMISSIONS_STATE_BURNED_MASK = COMBINED_STATE_BURNED_MASK
EMISSIONS_STATE_BURNED_SHIFT = COMBINED_STATE_BURNED_SHIFT
EMISSIONS_STATE_HAS_DRAINED_BIT = COMBINED_STATE_HAS_DRAINED_BIT
EMISSIONS_STATE_HAS_BURNED_BIT = COMBINED_STATE_HAS_BURNED_BIT


def pack_emissions_state(drained_code: np.ndarray, burned_code: np.ndarray) -> np.ndarray:
    return pack_combined_state(drained_code, burned_code)


def unpack_emissions_state_to_legacy(combined_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return unpack_combined_state_to_legacy(combined_state)

GADM_ADM0_IDS = np.array(sorted({
    0, 4, 8, 10, 12, 16, 20, 24, 28, 31, 32, 36,
    40, 44, 48, 50, 51, 52, 56, 60, 64, 68, 70, 72,
    74, 76, 84, 86, 90, 92, 96, 100, 104, 108, 112, 116,
    120, 124, 132, 136, 140, 144, 148, 152, 156, 158, 162, 166,
    170, 174, 175, 178, 180, 184, 188, 191, 192, 196, 203, 204,
    208, 212, 214, 218, 222, 226, 231, 232, 233, 234, 238, 239,
    242, 246, 248, 250, 254, 258, 260, 262, 266, 268, 270, 275,
    276, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324, 328,
    332, 334, 336, 340, 348, 352, 356, 360, 364, 368, 372, 376,
    380, 384, 388, 392, 398, 400, 404, 408, 410, 414, 417, 418,
    422, 426, 428, 430, 434, 438, 440, 442, 450, 454, 458, 462,
    466, 470, 474, 478, 480, 484, 492, 496, 498, 499, 500, 504,
    508, 512, 516, 520, 524, 528, 531, 533, 534, 535, 540, 548,
    554, 558, 562, 566, 570, 574, 578, 580, 581, 583, 584, 585,
    586, 591, 598, 600, 604, 608, 612, 616, 620, 624, 626, 630,
    634, 638, 642, 643, 646, 652, 654, 659, 660, 662, 663, 666,
    670, 674, 678, 682, 686, 688, 690, 694, 702, 703, 704, 705,
    706, 710, 716, 724, 728, 729, 732, 740, 744, 748, 752, 756,
    760, 762, 764, 768, 772, 776, 780, 784, 788, 792, 795, 796,
    798, 800, 804, 807, 818, 826, 831, 832, 833, 834, 840, 850,
    854, 858, 860, 862, 876, 882, 887, 894,
}), dtype=np.uint16)
