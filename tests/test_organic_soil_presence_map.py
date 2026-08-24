from pathlib import Path

import numpy as np
import pytest

from src.scripts.postprocessing.visualization import create_organic_soil_presence_map as maps
from src.scripts.utilities import constants_and_names as cn


def test_load_threshold_config_scales_registry_values() -> None:
    config = maps.load_threshold_config(
        Path("docs/organic_soil_threshold_registry.csv"),
        "20260508",
        "f2",
        "per-biome",
    )

    assert config.metric == "F2"
    assert config.method == "per-biome"
    assert config.fallback_threshold == pytest.approx(15.302119944)
    assert config.thresholds_by_biome_code[cn.ecozone_codes["boreal"]] == pytest.approx(18.9001100747)
    assert config.thresholds_by_biome_code[cn.ecozone_codes["temperate"]] == pytest.approx(15.302119944)
    assert config.thresholds_by_biome_code[cn.ecozone_codes["tropical"]] == pytest.approx(17.0864019769)


def test_load_threshold_config_global_omits_biome_thresholds() -> None:
    config = maps.load_threshold_config(
        Path("docs/organic_soil_threshold_registry.csv"),
        "20251105_legacy",
        "f1",
        "global",
    )

    assert config.method == "global"
    assert config.fallback_threshold == pytest.approx(21.9017404469)
    assert config.thresholds_by_biome_code == {}


def test_aggregate_presence_block_global() -> None:
    config = maps.ThresholdConfig(
        version="test",
        metric="F1",
        method="global",
        fallback_threshold=50.0,
        thresholds_by_biome_code={},
    )
    probability = np.array(
        [
            [0, 10, 20, 30],
            [40, 51, 20, 30],
            [0, 10, 20, 30],
            [40, 50, 20, 99],
        ],
        dtype=np.uint8,
    )

    out = maps.aggregate_presence_block(
        probability,
        None,
        factor=2,
        threshold_config=config,
    )

    # Equality is included in the model's documented >= threshold contract.
    np.testing.assert_array_equal(out, np.array([[1, 0], [1, 1]], dtype=np.uint8))


def test_aggregate_presence_block_per_biome_uses_fallback_for_unknown() -> None:
    config = maps.ThresholdConfig(
        version="test",
        metric="F1",
        method="per-biome",
        fallback_threshold=80.0,
        thresholds_by_biome_code={
            cn.ecozone_codes["boreal"]: 30.0,
            cn.ecozone_codes["temperate"]: 60.0,
        },
    )
    probability = np.array(
        [
            [40, 20, 70, 10],
            [20, 20, 50, 10],
            [70, 10, 70, 10],
            [70, 10, 79, 10],
        ],
        dtype=np.uint8,
    )
    climate = np.array(
        [
            [cn.ecozone_codes["boreal"], cn.ecozone_codes["boreal"], cn.ecozone_codes["temperate"], cn.ecozone_codes["temperate"]],
            [cn.ecozone_codes["boreal"], cn.ecozone_codes["boreal"], cn.ecozone_codes["temperate"], cn.ecozone_codes["temperate"]],
            [cn.ecozone_codes["unknown"], cn.ecozone_codes["unknown"], cn.ecozone_codes["unknown"], cn.ecozone_codes["unknown"]],
            [cn.ecozone_codes["unknown"], cn.ecozone_codes["unknown"], cn.ecozone_codes["unknown"], cn.ecozone_codes["unknown"]],
        ],
        dtype=np.int16,
    )

    out = maps.aggregate_presence_block(
        probability,
        climate,
        factor=2,
        threshold_config=config,
    )

    np.testing.assert_array_equal(out, np.array([[1, 1], [0, 0]], dtype=np.uint8))
