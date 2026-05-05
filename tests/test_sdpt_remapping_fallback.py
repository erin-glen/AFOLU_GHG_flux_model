from src.scripts.preprocessing.sdpt.create_remapping import (
    ROTATION_CLASS_CODES,
    classify,
)


def test_sdpt_fallback_classifies_eucalyptus_as_short_rotation():
    row = {
        "simpleType": "planted forest",
        "simpleName": "",
        "sciName": "Eucalyptus",
    }

    assert classify(row) == "short_rotation"


def test_sdpt_fallback_classifies_pinus_as_long_rotation():
    row = {
        "simpleType": "planted forest",
        "simpleName": "",
        "sciName": "Pinus",
    }

    assert classify(row) == "long_rotation"


def test_sdpt_fallback_missing_sci_name_does_not_crash():
    row = {
        "simpleType": "planted forest",
        "simpleName": "",
    }

    assert classify(row) == "long_rotation"


def test_sdpt_fallback_prioritizes_oil_palm_simple_name():
    row = {
        "simpleType": "tree crops",
        "simpleName": "oil palm",
        "sciName": "Pinus",
    }

    assert classify(row) == "oil_palm"


def test_sdpt_rotation_class_codes_are_model_convention():
    assert ROTATION_CLASS_CODES == {
        "oil_palm": 1,
        "short_rotation": 2,
        "long_rotation": 3,
    }
