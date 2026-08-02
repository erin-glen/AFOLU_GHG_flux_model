import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path("src/scripts/zonal_statistics/validate_wdpa_engineer_match.py")
spec = importlib.util.spec_from_file_location("wdpa_engineer_validation", MODULE_PATH)
validation = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = validation
spec.loader.exec_module(validation)


def test_default_compatibility_tolerance_accepts_observed_float_source_residual() -> None:
    engineer = 15_727_667.062699437
    zonal = 15_727_779.0  # 7.12 ppm above the engineer result

    passed, difference, tolerance = validation._numeric_match(
        zonal,
        engineer,
        absolute_tolerance=validation.DEFAULT_EMISSIONS_ABSOLUTE_TOLERANCE,
        relative_tolerance=validation.DEFAULT_EMISSIONS_RELATIVE_TOLERANCE,
    )

    assert passed
    assert difference > 0
    assert tolerance == validation.DEFAULT_EMISSIONS_RELATIVE_TOLERANCE * engineer


def test_compatibility_tolerance_still_rejects_material_wdpa_gap() -> None:
    engineer = 1_000_000.0
    zonal = 990_000.0  # one-percent discrepancy

    passed, _, _ = validation._numeric_match(
        zonal,
        engineer,
        absolute_tolerance=validation.DEFAULT_EMISSIONS_ABSOLUTE_TOLERANCE,
        relative_tolerance=validation.DEFAULT_EMISSIONS_RELATIVE_TOLERANCE,
    )

    assert not passed


def test_coiled_execution_requires_wsl_linux(monkeypatch) -> None:
    monkeypatch.setattr(validation.platform, "system", lambda: "Windows")

    with pytest.raises(RuntimeError, match="WSL/Linux"):
        validation.require_linux_for_coiled_execution(execute_zonal=True)

    validation.require_linux_for_coiled_execution(execute_zonal=False)
