import numpy as np

"""Node code utilities for organic soils zonal statistics."""

PAD_DIGITS = 8
# Backward compatibility: historical callers refer to STATE_CODE_PAD_DIGITS
# when zfill-ing emitted node codes. Keep the public alias to avoid breaking
# downstream scripts (e.g., pub_common/pub_assets).
STATE_CODE_PAD_DIGITS = PAD_DIGITS

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

def _build_drained_state_mapping() -> dict[str, str]:
    mapping: dict[str, str] = {}

    for root in ("16", "0"):
        base_label = _drain_root[root]
        for suffix, suffix_label in _classification_suffix_labels.items():
            if suffix:
                code = suffix if root == "0" else root + suffix
            else:
                code = root
            mapping[_pad_right(code)] = base_label + suffix_label

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