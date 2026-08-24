# -*- coding: utf-8 -*-
"""Offline provenance records for corrected publication-figure exports."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Sequence


APPROVED_SHA256SUMS_SHA256 = (
    "5bb4dd0f67d9ce92e3c3e2c16f6eb7439d4388927d2008300473b2511289bb34"
)

# Current working-tree bytes for every file in this list are recorded.  Git
# HEAD is reported separately and is never presented as containing modified or
# untracked files.
GENERATOR_RELATIVE_PATHS = (
    "src/scripts/zonal_statistics/pub_scripts/regenerate_publication_figures.py",
    "src/scripts/zonal_statistics/pub_scripts/publication_figure_renderers.py",
    "src/scripts/zonal_statistics/pub_scripts/publication_figure_export.py",
    "src/scripts/zonal_statistics/pub_scripts/publication_figure_qa.py",
    "src/scripts/zonal_statistics/pub_scripts/publication_figure_provenance.py",
    "src/scripts/zonal_statistics/pub_scripts/publication_nghgi_renderer.py",
    "src/scripts/zonal_statistics/pub_scripts/pub_common.py",
    "src/scripts/zonal_statistics/pub_scripts/pub_compare_runs.py",
)

HEAD_SCOPE_NOTE = (
    "Git HEAD identifies the repository base only. Per-file SHA-256 values "
    "record the actual working-tree bytes used, including modified and "
    "untracked files; HEAD must not be cited as containing those bytes."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, display_path: str | None = None) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Provenance input is not a file: {path}")
    return {
        "path": display_path or str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def verify_approved_sha256s(path: Path) -> dict[str, object]:
    """Verify and describe the one approved frozen-input checksum list."""

    record = _file_record(path)
    observed = str(record["sha256"])
    if observed != APPROVED_SHA256SUMS_SHA256:
        raise ValueError(
            "Approved SHA256SUMS.txt hash mismatch: expected "
            f"{APPROVED_SHA256SUMS_SHA256}, observed {observed}"
        )
    return record


def _run_git(repo_root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _relative_repo_file(repo_root: Path, relative: str) -> tuple[Path, str]:
    normalized = Path(*relative.replace("\\", "/").split("/"))
    if normalized.is_absolute():
        raise ValueError(f"Generator path must be repository-relative: {relative}")
    path = (repo_root / normalized).resolve()
    try:
        canonical = path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Generator path escapes repository: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Generator/dependency file is missing: {path}")
    return path, canonical


def _git_file_state(repo_root: Path, path: Path, relative: str) -> dict[str, object]:
    tracked_result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    tracked = tracked_result.returncode == 0
    status_result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            relative,
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if status_result.returncode:
        detail = status_result.stderr.strip() or status_result.stdout.strip()
        raise RuntimeError(f"git status failed: {detail}")
    status_lines = status_result.stdout.splitlines()
    status = status_lines[0][:2] if status_lines else ""

    head_blob = None
    current_blob = _run_git(repo_root, "hash-object", "--", str(path))
    if tracked:
        result = subprocess.run(
            ["git", "rev-parse", f"HEAD:{relative}"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode == 0:
            head_blob = result.stdout.strip()
    return {
        "git_status": status,
        "tracked": tracked,
        "head_blob": head_blob,
        "working_tree_blob": current_blob,
        "head_contains_current_bytes": bool(head_blob and head_blob == current_blob),
    }


def build_publication_figure_provenance(
    *,
    repo_root: Path,
    approved_sha256s_path: Path,
    generator_relative_paths: Sequence[str] = GENERATOR_RELATIVE_PATHS,
    external_dependency_paths: Sequence[Path] = (),
) -> dict[str, object]:
    """Build an offline, byte-specific provenance record for figure rendering."""

    repo_root = repo_root.resolve()
    if not (repo_root / ".git").exists():
        raise ValueError(f"Not a Git working tree: {repo_root}")

    generator_records: list[dict[str, object]] = []
    for relative in generator_relative_paths:
        path, canonical = _relative_repo_file(repo_root, relative)
        record = _file_record(path, display_path=canonical)
        record.update(_git_file_state(repo_root, path, canonical))
        generator_records.append(record)

    repository_dirty = bool(
        _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=normal")
    )
    all_in_head = all(
        bool(record["head_contains_current_bytes"]) for record in generator_records
    )
    external_records = [_file_record(Path(path)) for path in external_dependency_paths]

    return {
        "schema_version": 1,
        "approved_sha256s": verify_approved_sha256s(approved_sha256s_path),
        "repository": {
            "path": str(repo_root),
            "head_commit": _run_git(repo_root, "rev-parse", "HEAD"),
            "working_tree_dirty": repository_dirty,
            "head_contains_all_recorded_file_bytes": all_in_head,
            "head_scope_note": HEAD_SCOPE_NOTE,
        },
        "generator_and_dependency_files": generator_records,
        "external_dependency_files": external_records,
    }
