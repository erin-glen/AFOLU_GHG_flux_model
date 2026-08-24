from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from src.scripts.zonal_statistics.pub_scripts import (
    publication_figure_provenance as provenance,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def test_pins_approved_checksum_hash_and_dirty_dependency() -> None:
    assert provenance.APPROVED_SHA256SUMS_SHA256 == (
        "5bb4dd0f67d9ce92e3c3e2c16f6eb7439d4388927d2008300473b2511289bb34"
    )
    assert any(
        path.endswith("/pub_compare_runs.py")
        for path in provenance.GENERATOR_RELATIVE_PATHS
    )


def test_approved_checksum_mismatch_fails(tmp_path: Path) -> None:
    checksum_file = tmp_path / "SHA256SUMS.txt"
    checksum_file.write_text("not the approved checksum list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Approved SHA256SUMS.txt hash mismatch"):
        provenance.verify_approved_sha256s(checksum_file)


def test_records_working_tree_bytes_and_git_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "publication-test@example.invalid")
    _git(repo, "config", "user.name", "Publication Test")

    _write(repo / "clean.py", "CLEAN = True\n")
    _write(repo / "dirty.py", "VALUE = 'committed'\n")
    _git(repo, "add", "clean.py", "dirty.py")
    _git(repo, "commit", "-m", "fixture")
    head = _git(repo, "rev-parse", "HEAD")

    _write(repo / "dirty.py", "VALUE = 'working tree'\n")
    _write(repo / "new.py", "UNTRACKED = True\n")
    external = tmp_path / "external_renderer.py"
    _write(external, "RENDERER = 'frozen'\n")

    checksum_file = repo / "SHA256SUMS.txt"
    checksum_bytes = b"frozen input hashes\n"
    checksum_file.write_bytes(checksum_bytes)
    monkeypatch.setattr(
        provenance,
        "APPROVED_SHA256SUMS_SHA256",
        hashlib.sha256(checksum_bytes).hexdigest(),
    )

    result = provenance.build_publication_figure_provenance(
        repo_root=repo,
        approved_sha256s_path=checksum_file,
        generator_relative_paths=("clean.py", "dirty.py", "new.py"),
        external_dependency_paths=(external,),
    )

    repository = result["repository"]
    assert repository["head_commit"] == head
    assert repository["working_tree_dirty"] is True
    assert repository["head_contains_all_recorded_file_bytes"] is False
    assert "HEAD identifies the repository base only" in repository["head_scope_note"]

    records = {
        record["path"]: record
        for record in result["generator_and_dependency_files"]
    }
    assert records["clean.py"]["git_status"] == ""
    assert records["clean.py"]["tracked"] is True
    assert records["clean.py"]["head_contains_current_bytes"] is True
    assert records["dirty.py"]["git_status"] == " M"
    assert records["dirty.py"]["tracked"] is True
    assert records["dirty.py"]["head_contains_current_bytes"] is False
    assert records["new.py"]["git_status"] == "??"
    assert records["new.py"]["tracked"] is False
    assert records["new.py"]["head_contains_current_bytes"] is False
    for relative, record in records.items():
        path = repo / relative
        assert record["bytes"] == path.stat().st_size
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    approved = result["approved_sha256s"]
    assert approved["bytes"] == len(checksum_bytes)
    assert approved["sha256"] == hashlib.sha256(checksum_bytes).hexdigest()
    assert result["external_dependency_files"] == [
        {
            "path": str(external.resolve()),
            "bytes": external.stat().st_size,
            "sha256": hashlib.sha256(external.read_bytes()).hexdigest(),
        }
    ]


def test_generator_path_cannot_escape_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "publication-test@example.invalid")
    _git(repo, "config", "user.name", "Publication Test")
    _write(repo / "tracked.py", "VALUE = 1\n")
    _git(repo, "add", "tracked.py")
    _git(repo, "commit", "-m", "fixture")
    outside = tmp_path / "outside.py"
    _write(outside, "VALUE = 2\n")
    checksum = repo / "SHA256SUMS.txt"
    checksum.write_bytes(b"approved\n")
    monkeypatch.setattr(
        provenance,
        "APPROVED_SHA256SUMS_SHA256",
        hashlib.sha256(checksum.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="escapes repository"):
        provenance.build_publication_figure_provenance(
            repo_root=repo,
            approved_sha256s_path=checksum,
            generator_relative_paths=("../outside.py",),
        )
