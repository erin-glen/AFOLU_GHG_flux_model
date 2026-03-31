from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTANTS_FILE = REPO_ROOT / "src/scripts/utilities/constants_and_names.py"
ZONAL_RUN_FILE = REPO_ROOT / "src/scripts/zonal_statistics/02_run_zonal_stats.py"


def _read_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _literal_assignment(module: ast.Module, name: str):
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == name and node.value is not None:
                return ast.literal_eval(node.value)
    raise AssertionError(f"Could not find assignment for {name}")


def _outputs_to_zarr() -> list[str]:
    module = _read_module(CONSTANTS_FILE)
    outputs = _literal_assignment(module, "drainage_outputs_to_zarr")
    assert isinstance(outputs, list)
    return outputs


def _datasets_config() -> dict[str, dict]:
    module = _read_module(ZONAL_RUN_FILE)
    datasets = _literal_assignment(module, "DATASETS")
    assert isinstance(datasets, dict)
    return datasets


def _flux_specs() -> dict[str, dict]:
    module = _read_module(ZONAL_RUN_FILE)
    specs = _literal_assignment(module, "FLUX_SPECS")
    assert isinstance(specs, dict)
    return specs


def test_state_outputs_present_and_named_consistently():
    outputs = set(_outputs_to_zarr())
    datasets = _datasets_config()

    expected_state_datasets = {
        "emissions_state_nodes": ("emissions_state", "emissions_state_nodes"),
        "drained_state_nodes": ("drained_state", "drained_state_nodes"),
        "burned_state_nodes": ("burned_state", "burned_state_nodes"),
    }

    for key, (source_var, alias) in expected_state_datasets.items():
        spec = datasets[key]
        assert spec["source_var"] == source_var
        assert spec["kind"] == "state"
        assert spec["state_alias"] == alias
        assert source_var in outputs


def test_flux_datasets_reference_available_driver_outputs():
    outputs = set(_outputs_to_zarr())
    datasets = _datasets_config()

    for key, spec in datasets.items():
        if not key.startswith(("drained_", "burned_")):
            continue
        source_var = spec["source_var"]
        if isinstance(source_var, list):
            missing = [name for name in source_var if name not in outputs]
            assert not missing, f"{key} missing output vars: {missing}"
        else:
            assert source_var in outputs, f"{key} missing output var: {source_var}"


def test_flux_specs_cover_all_flux_datasets():
    datasets = _datasets_config()
    flux_specs = _flux_specs()

    flux_dataset_keys = {
        key
        for key, spec in datasets.items()
        if spec.get("kind") in {"flux_per_ha_yr", "flux_per_ha_yr_sum"}
    }
    assert flux_dataset_keys == set(flux_specs.keys())
