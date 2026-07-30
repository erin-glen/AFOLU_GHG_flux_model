import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = REPO_ROOT / "portable" / "input_dataset_maps"
SCRIPT_NAME = "create_input_dataset_maps.py"
CONFIG_NAME = "input_dataset_maps.config.json"
EXPECTED_FILES = {
    SCRIPT_NAME,
    CONFIG_NAME,
    "requirements.txt",
    "environment.yml",
    "README.md",
    "smoke_test.py",
    "run_maps.ps1",
    "run_maps.sh",
    "run_maps.bat",
}


def _read(name: str) -> str:
    return (BUNDLE_DIR / name).read_text(encoding="utf-8")


def _requirement_names(text: str) -> set[str]:
    names: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith(("-r", "--requirement")):
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)", line)
        if match:
            names.add(match.group(1).lower().replace("_", "-"))
    return names


def _all_mapping_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _all_mapping_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_mapping_keys(child)


def test_portable_bundle_contains_complete_handoff() -> None:
    assert BUNDLE_DIR.is_dir()
    assert EXPECTED_FILES <= {path.name for path in BUNDLE_DIR.iterdir() if path.is_file()}


def test_portable_script_parses_and_has_no_repo_import() -> None:
    script = _read(SCRIPT_NAME)
    tree = ast.parse(script, filename=str(BUNDLE_DIR / SCRIPT_NAME))

    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert "src" not in imported_roots
    assert "src.scripts" not in script
    assert "constants_and_names" not in script
    compile(tree, str(BUNDLE_DIR / SCRIPT_NAME), "exec")


def test_portable_config_has_eight_explicit_sources_and_no_model_resolver() -> None:
    config = json.loads(_read(CONFIG_NAME))

    assert isinstance(config, dict)
    assert "model_input" not in set(_all_mapping_keys(config))
    assert "model_context" not in config
    assert isinstance(config.get("datasets"), list)
    assert len(config["datasets"]) == 8

    source_entries = []
    for dataset in config["datasets"]:
        assert isinstance(dataset, dict)
        assert "source" in dataset
        assert "model_input" not in dataset
        source = dataset["source"]
        if isinstance(source, str):
            values = [source]
        else:
            assert isinstance(source, list)
            values = source
        assert values and all(isinstance(value, str) and value.strip() for value in values)
        source_entries.extend(values)

    assert len(source_entries) == 8


def test_portable_script_validates_from_outside_repo_and_resolves_eight_sources(
    tmp_path: Path,
) -> None:
    copied_bundle = tmp_path / "copied_bundle"
    shutil.copytree(BUNDLE_DIR, copied_bundle)
    outside_cwd = tmp_path / "unrelated_working_directory"
    outside_cwd.mkdir()
    output_dir = tmp_path / "unused_outputs"

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(copied_bundle / SCRIPT_NAME),
            "--config",
            str(copied_bundle / CONFIG_NAME),
            "--output-dir",
            str(output_dir),
            "--validate-only",
        ],
        cwd=outside_cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["validated"] is True
    assert len(payload["datasets"]) == 8
    resolved = [
        source
        for dataset in payload["datasets"]
        for source in dataset["resolved_sources"]
    ]
    assert len(resolved) == 8
    assert all(source.startswith("s3://") for source in resolved)
    assert not output_dir.exists()


def test_bundled_offline_smoke_test_runs_from_outside_repo(tmp_path: Path) -> None:
    copied_bundle = tmp_path / "copied_bundle"
    shutil.copytree(BUNDLE_DIR, copied_bundle)
    outside_cwd = tmp_path / "unrelated_working_directory"
    outside_cwd.mkdir()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(copied_bundle / "smoke_test.py")],
        cwd=outside_cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "PASS: standalone generator produced 8 aligned PNGs" in result.stdout


def test_portable_dependency_files_cover_standalone_imports() -> None:
    requirements = _requirement_names(_read("requirements.txt"))
    assert {"numpy", "rasterio", "matplotlib", "s3fs", "boto3"} <= requirements

    environment = _read("environment.yml").lower()
    assert "python" in environment
    assert "pip" in environment
    assert "requirements.txt" in environment or requirements <= _requirement_names(environment)


def test_shell_launcher_is_self_relative_and_syntax_checks_when_available() -> None:
    launcher = BUNDLE_DIR / "run_maps.sh"
    text = launcher.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash")
    assert "set -" in text and "e" in text.splitlines()[1]
    assert "BASH_SOURCE" in text or 'dirname "$0"' in text
    assert SCRIPT_NAME in text
    assert CONFIG_NAME in text
    assert '"$@"' in text

    bash = shutil.which("bash")
    if bash and os.name != "nt":
        result = subprocess.run(
            [bash, "-n", str(launcher)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_powershell_launcher_is_self_relative_and_parses_when_available() -> None:
    launcher = BUNDLE_DIR / "run_maps.ps1"
    text = launcher.read_text(encoding="utf-8")

    assert "$PSScriptRoot" in text
    assert SCRIPT_NAME in text
    assert CONFIG_NAME in text
    assert "ValueFromRemainingArguments" in text
    assert "$RemainingArgs" in text
    assert "@PythonArgs" in text

    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell:
        escaped_path = str(launcher).replace("'", "''")
        parse_command = (
            "$tokens = $null; $errors = $null; "
            "[System.Management.Automation.Language.Parser]::ParseFile("
            f"'{escaped_path}', [ref]$tokens, [ref]$errors) | Out-Null; "
            "if ($errors.Count -gt 0) { "
            "$errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
        )
        result = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", parse_command],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_batch_launcher_is_self_relative_and_forwards_arguments(tmp_path: Path) -> None:
    text = _read("run_maps.bat")
    lower = text.lower()

    assert "@echo off" in lower
    assert "%~dp0" in lower
    assert SCRIPT_NAME.lower() in lower
    assert CONFIG_NAME.lower() in lower
    assert "%*" in text

    if os.name == "nt" and shutil.which("cmd.exe"):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [
                "cmd.exe",
                "/d",
                "/c",
                "call",
                str(BUNDLE_DIR / "run_maps.bat"),
                "-ValidateOnly",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert '"validated": true' in result.stdout


def test_readme_describes_bundle_local_launch_paths() -> None:
    readme = _read("README.md")

    assert SCRIPT_NAME in readme
    assert CONFIG_NAME in readme
    assert "run_maps.ps1" in readme
    assert "run_maps.sh" in readme
    assert "run_maps.bat" in readme
    assert "requirements.txt" in readme
