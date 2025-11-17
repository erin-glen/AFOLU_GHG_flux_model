import numpy as np

"""Node code utilities for organic soils zonal statistics.

Updated for the simplified decision tree:
- `drained_state` encodes *only* classification (peat drained trigger or undrained)
  with an optional coastal tag (91=coastal_mangrove, 92=coastal_tidal_marsh).
- Emission-path digits are no longer present in `drained_state`.
- Two explicit categorical outputs now exist and are described here:
  * `coastal_mask`: 0=non_coastal, 1=mangrove, 2=tidal_marsh
  * `drain_source`: 0=non_peat, 1=canals, 2=roads_grip, 3=crop_settlement,
                   4=plantation_descals, 5=extraction, 6=peat_undrained
"""

# Must match the model's MAX_STATE_DIGITS (kept at 8 in the simplified tree).
_PAD_DIGITS = 8


def _pad_right(code: str) -> str:
    """Right‑pad ``code`` with zeros to the standard length."""
    return code.ljust(_PAD_DIGITS, "0")


# Root meanings (classification only)
# 11..15 are drained peat triggers; 16 is undrained peat; 0 is non‑peat.
_drain_root = {
    "11": "peat_drained_primary_infra",          # canals
    "12": "peat_drained_secondary_infra",        # roads / GRIP
    "13": "peat_drained_cropland_settlement",
    "14": "peat_drained_plantation",
    "15": "peat_drained_extraction",
    "16": "peat_undrained",
    "0":  "non_peat",
}

# Optional classification suffix (only present for peat pixels)
_classification_suffix_labels = {
    "": "",
    "91": "__coastal_mangrove",
    "92": "__coastal_tidal_marsh",
}


def _build_drained_state_mapping() -> dict[str, str]:
    """
    Build the mapping for classification‑only `drained_state`.

    Codes emitted by the simplified tree:
      - '0' (non‑peat)
      - '11','12','13','14','15' (drained peat by trigger), '16' (undrained peat)
      - Optional coastal suffix appended as '9{1|2}', e.g., '1191', '1692'
    All keys are right‑padded to _PAD_DIGITS with zeros.
    """
    mapping: dict[str, str] = {}

    # Non‑peat root (no coastal suffix on non‑peat in the simplified tree)
    mapping[_pad_right("0")] = _drain_root["0"]

    # Peat roots with and without coastal suffix
    for root in ("11", "12", "13", "14", "15", "16"):
        base_label = _drain_root[root]
        # Without coastal suffix
        mapping[_pad_right(root)] = base_label
        # With coastal suffix (only if suffix is non‑empty)
        for suffix, suffix_label in _classification_suffix_labels.items():
            if suffix:
                mapping[_pad_right(root + suffix)] = base_label + suffix_label

    return mapping


DRAINED_STATE_NODE_MEANINGS: dict[str, str] = _build_drained_state_mapping()

# Burned-state meanings are unchanged (emission logic unaffected by the simplification)
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

# Convenience sets of all valid state codes (padded)
ALL_DRAINED_STATE_CODES = frozenset(DRAINED_STATE_NODE_MEANINGS.keys())
ALL_BURNED_STATE_CODES = frozenset(BURNED_STATE_NODE_MEANINGS.keys())

# --- New: explicit categorical layer meanings from the simplified model ---

# drain_source layer (uint), values defined by the decision tree:
# 0=non_peat, 1=canals, 2=roads/GRIP, 3=crop/settlement,
# 4=plantation/DeScals, 5=extraction, 6=peat_undrained
DRAIN_SOURCE_MEANINGS: dict[int, str] = {
    0: "non_peat",
    1: "canals",
    2: "roads_grip",
    3: "cropland_settlement",
    4: "plantation_descals",
    5: "extraction",
    6: "peat_undrained",
}

# coastal_mask layer (uint), values defined by the decision tree:
# 0=non_coastal, 1=coastal_mangrove, 2=coastal_tidal_marsh
COASTAL_MASK_MEANINGS: dict[int, str] = {
    0: "non_coastal",
    1: "coastal_mangrove",
    2: "coastal_tidal_marsh",
}

# Optional helper arrays if you need fast membership checks / histogram bins
ALL_DRAIN_SOURCE_CODES = np.array(sorted(DRAIN_SOURCE_MEANINGS.keys()), dtype=np.uint8)
ALL_COASTAL_MASK_CODES = np.array(sorted(COASTAL_MASK_MEANINGS.keys()), dtype=np.uint8)

# GADM IDs unchanged
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
