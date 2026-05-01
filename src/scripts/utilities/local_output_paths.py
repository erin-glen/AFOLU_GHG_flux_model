"""Shared local output paths for durable AFOLU artifacts.

This module intentionally avoids importing project constants so it can be used
by constants modules and lightweight CLI utilities without circular imports.
"""

from __future__ import annotations

import os
import platform
import posixpath
from typing import Callable, Mapping, Optional


LOCAL_OUTPUT_ROOT_ENV = "AFOLU_LOCAL_OUTPUT_ROOT"


def _as_posix_path(path: str | os.PathLike[str]) -> str:
    """Return a stable slash-normalized path string."""

    text = os.fspath(path).replace("\\", "/")
    if text.endswith("/") and not _is_root_path(text):
        text = text.rstrip("/")
    return text


def _is_root_path(path: str) -> bool:
    return path == "/" or (len(path) == 3 and path[1:] == ":/")


def default_local_output_root(
    *,
    platform_system: Optional[str] = None,
    path_exists: Callable[[str], bool] = os.path.exists,
) -> str:
    """Return the default durable local output root for the current host."""

    system = platform_system or platform.system()
    if system == "Windows":
        return "C:/tmp/afolu"
    if path_exists("/mnt/c/tmp"):
        return "/mnt/c/tmp/afolu"
    return "/tmp/afolu"


def local_output_root(env: Optional[Mapping[str, str]] = None) -> str:
    """Return the durable AFOLU local output root.

    ``AFOLU_LOCAL_OUTPUT_ROOT`` overrides the platform default.
    """

    env_map = os.environ if env is None else env
    root = env_map.get(LOCAL_OUTPUT_ROOT_ENV) or default_local_output_root()
    return _as_posix_path(root)


def local_output_path(*parts: str, root: Optional[str] = None) -> str:
    """Join path parts under the durable AFOLU local output root."""

    base = _as_posix_path(root or local_output_root())
    return posixpath.join(base, *parts)


def env_or_default(env_var: str, *default_parts: str, env: Optional[Mapping[str, str]] = None) -> str:
    """Return an explicit legacy env-var path or a helper-derived default."""

    env_map = os.environ if env is None else env
    explicit = env_map.get(env_var)
    if explicit:
        return _as_posix_path(explicit)
    return local_output_path(*default_parts)


def publication_root(kind: str, env_var: Optional[str] = None) -> str:
    """Return the local root for a publication output family."""

    if env_var:
        return env_or_default(env_var, "publications", kind)
    return local_output_path("publications", kind)


def publication_run_dir(
    kind: str,
    model_version: str,
    run_name: str,
    run_date: str,
    env_var: Optional[str] = None,
) -> str:
    """Return ``publications/<kind>/version_<model>/<run>/<date>``."""

    return posixpath.join(
        publication_root(kind, env_var=env_var),
        f"version_{model_version}",
        run_name,
        run_date,
    )


def zonal_stats_staging_dir(model_version: str, run_name: str, run_date: str) -> str:
    """Return the local staging directory for zonal statistics."""

    return local_output_path("staging", "zonal_stats", model_version, run_name, run_date)


def probability_area_stats_staging_dir(probability_date: str, contextual_date: str) -> str:
    """Return the local staging directory for probability-area statistics."""

    return local_output_path(
        "staging",
        "probability_area_stats",
        probability_date,
        contextual_date,
    )


def pipeline_log_dir(pipeline_name: str = "standard_500m") -> str:
    """Return the local log directory for a named pipeline."""

    return local_output_path("logs", "pipelines", pipeline_name)


def chunk_stats_root(run_name: Optional[str] = None, run_date: Optional[str] = None) -> str:
    """Return the local root for chunk-statistics outputs."""

    parts = ["chunk_stats"]
    if run_name:
        parts.append(run_name)
    if run_date:
        parts.append(run_date)
    return local_output_path(*parts)


def display_output_root() -> str:
    """Return the display-raster output mirror root."""

    return local_output_path("visualization", "global_maps", "display")


def world_boundaries_dir() -> str:
    """Return the local world-boundaries cache directory."""

    return local_output_path("visualization", "world_boundaries")
