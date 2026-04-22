"""Common helpers shared by the global raster aggregation and display scripts."""

from __future__ import annotations

import os
import posixpath
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.scripts.utilities import constants_and_names as cn


DEFAULT_NATIVE_DEG = 0.00025
DEFAULT_TARGET_DEG = 0.04

DATA_TYPES = [
    "burned_total_Mg_CO2e_pixel_yr",
    "drained_total_Mg_CO2e_pixel_yr",
    "combined_state",
]

INTEGER_DATASETS: set[str] = {"combined_state"}

INVENTORY_PERIODS = ["2021_2024"]

OUTPUT_ROOT = posixpath.join(cn.full_bucket_prefix, cn.project_dir, "outputs")
DEFAULT_MODEL_VERSION = getattr(
    cn, "model_version_underscore", cn.model_version.replace(".", "_")
)

BASE_URL = posixpath.join(OUTPUT_ROOT, f"version_{DEFAULT_MODEL_VERSION}")
OUTPUTS_BASE = BASE_URL

DEFAULT_DATE_TAG = "20260418"

DISPLAY_OUT_ROOT = os.environ.get("DISPLAY_OUT_ROOT", "/tmp/create_global_maps/display")


def deg_to_label(deg: float) -> str:
    """Convert 0.04 -> '0_04deg'; 0.01 -> '0_01deg'; 0.005 -> '0_005deg'."""

    s = f"{deg:.5f}".rstrip("0").rstrip(".")
    return f"{s.replace('.', '_')}deg"


def assert_grid_divides_world(target_deg: float) -> None:
    """Ensure ``target_deg`` divides 180°×360° evenly."""

    rows = round(180 / target_deg)
    cols = round(360 / target_deg)
    if not (
        np.isclose(rows * target_deg, 180.0)
        and np.isclose(cols * target_deg, 360.0)
    ):
        raise ValueError(
            f"--target_deg={target_deg} must divide 180 and 360 degrees evenly for a global grid."
        )


def ensure_dir(path_like: str | Path) -> None:
    """Create ``path_like`` (and parents) if it does not exist."""

    Path(path_like).mkdir(parents=True, exist_ok=True)


def gdalize_s3_url(s3_url: str) -> str:
    """Map ``s3://bucket/key`` to ``/vsis3/bucket/key`` for GDAL compatibility."""

    if s3_url.startswith("s3://"):
        return "/vsis3/" + s3_url[len("s3://") :]
    return s3_url


def to_local_mirror(path_like: str, root: str = DISPLAY_OUT_ROOT) -> str:
    """Mirror an S3 or ``/vsis3/`` path under the local ``DISPLAY_OUT_ROOT`` tree."""

    if path_like.startswith("/vsis3/"):
        rel = path_like[len("/vsis3/") :].lstrip("/")
    elif path_like.startswith("s3://"):
        rel = path_like[len("s3://") :].lstrip("/")
    else:
        rel = path_like.lstrip("/")

    return posixpath.join(root.rstrip("/"), rel)


def build_versioned_outputs_root(model_version: str, outputs_root: str = OUTPUT_ROOT) -> str:
    root = outputs_root.rstrip("/")
    return posixpath.join(root, f"version_{model_version}")


def resolve_versioned_paths(
    model_version: str,
    outputs_root: str,
    base_url: Optional[str] = None,
    outputs_base: Optional[str] = None,
) -> Tuple[str, str]:
    """Return ``(base_url, outputs_base)`` with sensible defaults."""

    versioned_root = build_versioned_outputs_root(model_version, outputs_root)
    resolved_base = (base_url or versioned_root).rstrip("/")
    resolved_outputs = (outputs_base or versioned_root).rstrip("/")
    return resolved_base, resolved_outputs


def get_input_datasets(
    pixel_resolution: str,
    data_types: Optional[List[str]] = None,
    inventory_periods: Optional[List[str]] = None,
    run_name: str = "ogh_sensitivity_1km",
    base_url: str = BASE_URL,
    output_date: str = DEFAULT_DATE_TAG,
) -> List[str]:
    """Return list of S3 folders for input rasters."""

    data_types = data_types or DATA_TYPES
    inventory_periods = inventory_periods or INVENTORY_PERIODS

    paths: List[str] = []
    for period in inventory_periods:
        for dtype in data_types:
            path = (
                f"{base_url}/{dtype}/{run_name}/"
                f"five_year_intervals/{period}/{pixel_resolution}/{output_date}"
            )
            paths.append(path if path.endswith("/") else path + "/")
    return paths


def build_download_upload_dict(
    pixel_resolution: str,
    run_name: str,
    target_deg: float,
    base_url: str = BASE_URL,
    output_date: str = DEFAULT_DATE_TAG,
    outputs_base: str = OUTPUTS_BASE,
    data_types: Optional[List[str]] = None,
    inventory_periods: Optional[List[str]] = None,
) -> dict:
    """Build dictionaries describing canonical input/output paths per dataset."""

    res_label = deg_to_label(target_deg)
    data_types = data_types or DATA_TYPES
    inventory_periods = inventory_periods or INVENTORY_PERIODS

    dictionary: dict[str, dict[str, str]] = {}
    for path in get_input_datasets(
        pixel_resolution,
        data_types,
        inventory_periods,
        run_name,
        base_url,
        output_date,
    ):
        parts = path.rstrip("/").split("/")

        # Paths follow:
        # ``.../version_<ver>/<dataset>/<run_name>/five_year_intervals/<interval>/<pixels>/<date>/``
        # ``parts`` therefore looks like:
        # ["s3:", "", "gfw2-data", ..., "version_<ver>", "<dataset>", "<run_name>",
        #  "five_year_intervals", "<interval>", "<pixel_res>", "<date>"]
        try:
            dataset = parts[-6]
            interval = parts[-3]
        except IndexError as exc:  # pragma: no cover - defensive; path template should hold
            raise ValueError(f"Unexpected input path format: {path}") from exc

        key = f"{dataset}__{interval}"

        if dataset.endswith("_ha") or dataset.endswith("_ha_yr"):
            raise ValueError(
                "Per-hectare datasets are no longer supported by the visualization pipeline: "
                f"received '{dataset}' in path {path}"
            )

        per_pixel_dir = path
        per_pixel_pattern = f"__{dataset}__{interval}.tif"

        # Mirror the main drivers: include the run name between dataset and interval so
        # global artifacts stay isolated per run.
        out_dir = (
            f"{outputs_base}/{res_label}_output_aggregation/"
            f"{dataset}/{run_name}/{interval}/"
        )

        dictionary[key] = {
            "dataset": dataset,
            "interval": interval,
            "per_pixel_dir": per_pixel_dir,
            "per_pixel_pattern": per_pixel_pattern,
            "global_dir": out_dir,
            "global_pattern": f"{res_label}_global__{dataset}_{interval}.tif",
        }
    return dictionary