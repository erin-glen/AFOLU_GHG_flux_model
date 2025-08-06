import numpy as np

"""Node code utilities for organic soils zonal statistics."""

_PAD_DIGITS = 8

def _pad_right(code: str) -> str:
    """Right‑pad ``code`` with zeros to the standard length."""
    return code.ljust(_PAD_DIGITS, "0")

_drain_root = {
    "11": "peat_drained_primary_infra",          # dadap / canals
    "12": "peat_drained_secondary_infra",        # roads / ENGERT / GRIP
    "13": "peat_drained_cropland_settlement",
    "14": "peat_drained_plantation",
    "15": "peat_drained_extraction",
    "16": "peat_undrained",
    "2":  "non_peat",
}

_drain_emit = {
    # BOREAL
    "1111": "boreal_forest_poor",
    "1112": "boreal_forest_rich",
    "112":  "boreal_grassland",
    "113":  "boreal_cropland",
    "114":  "boreal_extraction",
    "115":  "boreal_settlement",
    "116":  "boreal_wetland",
    "117":  "boreal_otherland",
    # TEMPERATE
    "121":  "temperate_forest",
    "1221": "temperate_grassland_poor",
    "1222": "temperate_grassland_rich",
    "123":  "temperate_cropland",
    "124":  "temperate_extraction",
    "125":  "temperate_settlement",
    "126":  "temperate_wetland",
    "127":  "temperate_otherland",
    # TROPICAL
    "1311": "tropical_long_rotation",
    "1312": "tropical_short_rotation",
    "1313": "tropical_oil_palm",
    "132":  "tropical_forest",
    "133":  "tropical_grassland",
    "134":  "tropical_cropland",
    "135":  "tropical_extraction",
    "138":  "tropical_settlement",
    "139":  "tropical_wetland",
    "13":  "tropical_otherland",
}

_burn_root = {
    "1": "boreal",
    "2": "temperate",
    "3": "tropical",
    "4": "other_domain",
}

_burn_emit = {
    "1": {"11": "drained", "12": "undrained"},
    "2": {"21": "drained", "22": "undrained"},
    "3": {
        "31": "drained_crop_or_plantation",
        "32": "drained_other",
        "33": "undrained",
    },
    "4": {"4": "other"},
}

## ── lookup tables ───────────────────────────────────────────────────
DRAINED_STATE_NODE_MEANINGS: dict[str, str] = {
    _pad_right(code): label
    for code, label in _drain_root.items()
    if code in ("16", "2")
}
DRAINED_STATE_NODE_MEANINGS.update(
    {
        _pad_right(root + emit_code): f"{_drain_root[root]}__{emit_label}"
        for root in ("11", "12", "13", "14", "15")
        for emit_code, emit_label in _drain_emit.items()
    }
)

BURNED_STATE_NODE_MEANINGS: dict[str, str] = {
    _pad_right(root + sub): f"{_burn_root[root]}__{label}"
    for root, sub_dict in _burn_emit.items()
    for sub, label in sub_dict.items()
}

ALL_DRAINED_STATE_CODES = frozenset(DRAINED_STATE_NODE_MEANINGS.keys())
ALL_BURNED_STATE_CODES = frozenset(BURNED_STATE_NODE_MEANINGS.keys())

GADM_ADM0_IDS = np.array([
    0.0, 4.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 28.0, 31.0, 32.0, 36.0,
    40.0, 44.0, 48.0, 50.0, 51.0, 52.0, 56.0, 60.0, 64.0, 68.0, 70.0, 72.0,
    74.0, 76.0, 84.0, 86.0, 90.0, 92.0, 96.0, 100.0, 104.0, 108.0, 112.0,
    116.0, 120.0, 124.0, 132.0, 136.0, 140.0, 144.0, 148.0, 152.0, 156.0,
    158.0, 162.0, 166.0, 170.0, 174.0, 175.0, 178.0, 180.0, 184.0, 188.0,
    191.0, 192.0, 196.0, 203.0, 204.0, 208.0, 212.0, 214.0, 218.0, 222.0,
    226.0, 231.0, 232.0, 233.0, 234.0, 238.0, 239.0, 242.0, 246.0, 248.0,
    250.0, 254.0, 258.0, 260.0, 262.0, 266.0, 268.0, 270.0, 275.0, 276.0,
    288.0, 292.0, 296.0, 300.0, 304.0, 308.0, 312.0, 316.0, 320.0, 324.0,
    328.0, 332.0, 334.0, 336.0, 340.0, 348.0, 352.0, 356.0, 360.0, 364.0,
    368.0, 372.0, 376.0, 380.0, 384.0, 388.0, 392.0, 398.0, 400.0, 404.0,
    408.0, 410.0, 414.0, 417.0, 418.0, 422.0, 426.0, 428.0, 430.0, 434.0,
    438.0, 440.0, 442.0, 450.0, 454.0, 458.0, 462.0, 466.0, 470.0, 474.0,
    478.0, 480.0, 484.0, 492.0, 496.0, 498.0, 499.0, 500.0, 504.0, 508.0,
    512.0, 516.0, 520.0, 524.0, 528.0, 531.0, 533.0, 534.0, 535.0, 540.0,
    548.0, 554.0, 558.0, 562.0, 566.0, 70.0, 574.0, 578.0, 580.0, 581.0,
    583.0, 584.0, 585.0, 586.0, 591.0, 598.0, 600.0, 604.0, 608.0, 612.0,
    616.0, 620.0, 624.0, 626.0, 630.0, 634.0, 638.0, 642.0, 643.0, 646.0,
    652.0, 654.0, 659.0, 660.0, 662.0, 663.0, 666.0, 670.0, 674.0, 678.0,
    682.0, 686.0, 688.0, 690.0, 694.0, 702.0, 703.0, 704.0, 705.0, 706.0,
    710.0, 716.0, 724.0, 728.0, 729.0, 732.0, 740.0, 744.0, 748.0, 752.0,
    756.0, 760.0, 762.0, 764.0, 768.0, 772.0, 776.0, 780.0, 784.0, 788.0,
    792.0, 795.0, 796.0, 798.0, 800.0, 804.0, 807.0, 818.0, 826.0, 831.0,
    832.0, 833.0, 834.0, 840.0, 850.0, 854.0, 858.0, 860.0, 862.0, 876.0,
    882.0, 887.0, 894.0
], dtype=np.uint16)