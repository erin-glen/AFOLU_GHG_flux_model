"""Run a bounded WDPA pre-flight against the engineer's organic-soil output.

This command is deliberately safe by default.  Without ``--execute-zonal`` it
only validates the pinned comparison artifacts, exercises the WDPA category
contract on synthetic data, and prints the planned AOIs.  With
``--execute-zonal`` it runs isolated country-bounding-box zonal jobs, compares
country emissions with the engineer parquet, and writes CSV/JSON QA reports.

Run from the repository root in the WSL ``coiled_20251119`` environment::

    python -m src.scripts.zonal_statistics.validate_wdpa_engineer_match \
      --profile lean \
      --execute-zonal \
      --launch-cluster \
      --cluster-name wdpa_preflight

The default lean profile covers Singapore (fast control), Netherlands (known
WDPA code 13), Poland (largest absolute reference gap), and Philippines
(largest relative reference gap).  ``--profile full`` covers all nine countries
whose old reference difference exceeded one percent, plus Singapore.

The current AFOLU zonal output's ``area__ha`` is full administrative area,
whereas the engineer's ``area_ha`` is organic-soil-masked area.  Emissions are
therefore the default pass/fail gate.  Use ``--require-area-match`` only after a
distinct ``organic_soil_area__ha`` flux type has been added to the zonal output.

The engineer pipeline uses a float64 hectare-area Zarr and float64 accumulation,
while the AFOLU zonal path uses its corrected float32 square-metre area Zarr.
Literal floating-point equality is reported separately but is not the default
gate.  The compatibility gate defaults to 0.001% relative difference (10 ppm),
which is still three orders of magnitude tighter than the one-percent WDPA
anomaly threshold this pre-flight is intended to detect.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import util as importlib_util
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr

from src.scripts.utilities import constants_and_names as cn


ENGINEER_ORGANIC_SHA256 = (
    "ff64efe55389aea6aa66eac76aaeb5233eb2931868d4252264492faf17ef73ff"
)
ENGINEER_REQUIRED_COLUMNS = {
    "aoi_id",
    "aoi_type",
    "interval_end_year",
    "gross_emissions_MgCO2e",
    "area_ha",
}
REFERENCE_REQUIRED_COLUMNS = {
    "adm0",
    "year",
    "org_soil_emis__all_gases__MgCO2e_yr",
}
KNOWN_HIGH_DIFFERENCE_COUNTRIES = {
    "CHE",
    "COL",
    "DEU",
    "DNK",
    "EST",
    "NLD",
    "PHL",
    "POL",
    "SVK",
}
INTERVAL_DIRECTORIES = {2020: "2016_2020", 2024: "2021_2024"}
EMISSIONS_FLUX_TYPES = {
    "drained_total_Mg_CO2e",
    "burned_total_Mg_CO2e",
}
ORGANIC_AREA_FLUX_TYPE = "organic_soil_area__ha"
DEFAULT_EMISSIONS_ABSOLUTE_TOLERANCE = 0.1
DEFAULT_EMISSIONS_RELATIVE_TOLERANCE = 1e-5


@dataclass(frozen=True)
class AoiSpec:
    iso3: str
    gadm_adm0: int
    bbox: tuple[float, float, float, float]
    purpose: str


# Bboxes are padded beyond the GADM country extent.  Neighboring countries are
# harmless because the comparison filters the result by numeric ADM0 code.
AOIS: dict[str, AoiSpec] = {
    "SGP": AoiSpec("SGP", 702, (103.5, 1.1, 104.2, 1.6), "compact control"),
    "NLD": AoiSpec("NLD", 528, (3.0, 50.5, 7.5, 54.0), "known WDPA code 13"),
    "POL": AoiSpec("POL", 616, (14.0, 48.8, 24.5, 55.0), "largest absolute gap"),
    "PHL": AoiSpec("PHL", 608, (116.5, 4.0, 127.0, 21.5), "largest relative gap"),
    "CHE": AoiSpec("CHE", 756, (5.8, 45.7, 10.6, 48.0), "flagged country"),
    "COL": AoiSpec("COL", 170, (-82.5, -4.5, -66.5, 13.8), "flagged country"),
    "DEU": AoiSpec("DEU", 276, (5.5, 47.0, 15.5, 55.2), "flagged country"),
    "DNK": AoiSpec("DNK", 208, (8.0, 54.4, 15.5, 58.0), "flagged country"),
    "EST": AoiSpec("EST", 233, (21.5, 57.3, 28.5, 60.0), "flagged country"),
    "SVK": AoiSpec("SVK", 703, (16.7, 47.5, 22.7, 49.7), "flagged country"),
}

PROFILES: dict[str, tuple[str, ...]] = {
    "lean": ("SGP", "NLD", "POL", "PHL"),
    "full": ("SGP", "NLD", "CHE", "DNK", "SVK", "EST", "DEU", "POL", "PHL", "COL"),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _country_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        ~frame["aoi_id"].astype("string").str.contains(".", regex=False)
    ].copy()


def validate_engineer_artifact(path: Path, *, enforce_sha256: bool) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Engineer organic-soil parquet not found: {path}")
    actual_sha256 = _sha256(path)
    if enforce_sha256 and actual_sha256 != ENGINEER_ORGANIC_SHA256:
        raise ValueError(
            "Engineer parquet SHA-256 does not match the pinned v20260730 attachment: "
            f"expected={ENGINEER_ORGANIC_SHA256} actual={actual_sha256}"
        )

    frame = pd.read_parquet(path)
    missing_columns = ENGINEER_REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Engineer parquet is missing columns: {sorted(missing_columns)}")
    if len(frame) != 19_538:
        raise ValueError(f"Engineer parquet row count changed: expected=19538 actual={len(frame)}")
    if frame.duplicated(["aoi_id", "interval_end_year"]).any():
        raise ValueError("Engineer parquet has duplicate (aoi_id, interval_end_year) keys")
    if frame[list(ENGINEER_REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Engineer parquet has null values in required columns")
    if set(frame["interval_end_year"].unique()) != {2020, 2024}:
        raise ValueError("Engineer parquet must contain exactly interval years 2020 and 2024")
    if set(frame["aoi_type"].unique()) != {"admin"}:
        raise ValueError("Engineer parquet contains an unexpected aoi_type")
    if (frame["gross_emissions_MgCO2e"] < 0).any() or (frame["area_ha"] < 0).any():
        raise ValueError("Engineer parquet contains negative emissions or area")

    countries = _country_rows(frame)
    if len(countries) != 244 or countries["aoi_id"].nunique() != 122:
        raise ValueError(
            "Engineer country-level coverage changed: "
            f"rows={len(countries)} countries={countries['aoi_id'].nunique()}"
        )
    print(
        "PASS artifact: "
        f"rows={len(frame)} country_year_rows={len(countries)} sha256={actual_sha256}"
    )
    return countries


def validate_reference_signature(reference_path: Path, engineer_countries: pd.DataFrame) -> None:
    if not reference_path.is_file():
        raise FileNotFoundError(f"Combined reference parquet not found: {reference_path}")
    reference = pd.read_parquet(reference_path, columns=sorted(REFERENCE_REQUIRED_COLUMNS))
    reference["interval_end_year"] = reference["year"].astype(int)
    reference = reference.loc[reference["interval_end_year"].isin(INTERVAL_DIRECTORIES)]
    reference = (
        reference.groupby(["adm0", "interval_end_year"], as_index=False)[
            "org_soil_emis__all_gases__MgCO2e_yr"
        ]
        .sum()
        .rename(
            columns={
                "adm0": "aoi_id",
                "org_soil_emis__all_gases__MgCO2e_yr": "reference_emissions",
            }
        )
    )
    comparison = reference.merge(
        engineer_countries[
            ["aoi_id", "interval_end_year", "gross_emissions_MgCO2e"]
        ],
        on=["aoi_id", "interval_end_year"],
        how="inner",
        validate="one_to_one",
    )
    difference = comparison["gross_emissions_MgCO2e"] - comparison["reference_emissions"]
    denominator = comparison["reference_emissions"].abs().replace(0, np.nan)
    comparison["absolute_percent_difference"] = difference.abs() / denominator * 100.0
    high_difference = set(
        comparison.loc[comparison["absolute_percent_difference"] > 1.0, "aoi_id"]
    )
    if high_difference != KNOWN_HIGH_DIFFERENCE_COUNTRIES:
        raise ValueError(
            "The attached reference/engineer discrepancy signature changed: "
            f"expected={sorted(KNOWN_HIGH_DIFFERENCE_COUNTRIES)} "
            f"actual={sorted(high_difference)}"
        )
    significant = comparison["absolute_percent_difference"] > 0.01
    if not (difference.loc[significant] > 0).all():
        raise ValueError(
            "At least one significant discrepancy is not engineer > reference; "
            "the WDPA-dropout signature no longer holds"
        )
    print(
        "PASS reference signature: nine countries exceed 1%; all differences "
        "above 0.01% are engineer > reference"
    )


def _load_organic_zonal_module():
    module_path = _repo_root() / "src/scripts/zonal_statistics/02_run_zonal_stats.py"
    spec = importlib_util.spec_from_file_location("organic_zonal_preflight", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load zonal module from {module_path}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_synthetic_wdpa_contract() -> None:
    module = _load_organic_zonal_module()
    registered = module.OPTIONAL_CONTEXTUAL_GROUPERS["wdpa"]["expected_groups"]
    if registered.tolist() != list(range(17)):
        raise ValueError(f"WDPA registry is not 0..16: {registered.tolist()}")

    valid = xr.DataArray(
        np.arange(17, dtype=np.uint8).reshape(1, 17),
        dims=("y", "x"),
        name="wdpa",
    )
    mask = xr.ones_like(valid, dtype=bool)
    valid_unexpected = module.unexpected_group_count_tasks(
        [valid], [registered], mask
    )[0]
    old_contract_unexpected = module.unexpected_group_count_tasks(
        [valid], [np.arange(12, dtype=np.uint8)], mask
    )[0]
    sentinel = xr.DataArray(
        np.array([[17]], dtype=np.uint8), dims=("y", "x"), name="wdpa"
    )
    sentinel_unexpected = module.unexpected_group_count_tasks(
        [sentinel], [registered], xr.ones_like(sentinel, dtype=bool)
    )[0]
    valid_count, old_count, sentinel_count = (
        int(np.asarray(value).item())
        for value in module.dask.compute(
            valid_unexpected,
            old_contract_unexpected,
            sentinel_unexpected,
        )
    )
    if valid_count != 0 or old_count != 5 or sentinel_count != 1:
        raise AssertionError(
            "Synthetic WDPA contract check failed: "
            f"valid={valid_count} old_contract={old_count} sentinel={sentinel_count}"
        )
    print(
        "PASS synthetic WDPA contract: codes 0..16 accepted, old contract drops "
        "exactly 12..16, code 17 rejected"
    )


def _run_command(command: list[str]) -> None:
    print(f"RUN {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=_repo_root(), check=True)


def _cluster_state(cluster_name: str) -> tuple[str | None, str | None]:
    import coiled

    for cluster in coiled.list_clusters(workspace=cn.Coiled_workspace)[:100]:
        if cluster.get("name") == cluster_name:
            state = cluster.get("current_state", {}).get("state")
            cluster_id = cluster.get("id") or cluster.get("cluster_id")
            return state, str(cluster_id) if cluster_id is not None else None
    return None, None


def ensure_cluster_ready(
    cluster_name: str,
    *,
    launch: bool,
    workers: int,
    memory_gib: int,
    timeout_seconds: int = 900,
) -> bool:
    launched_here = False
    if launch:
        _run_command(
            [
                sys.executable,
                "-m",
                "src.scripts.utilities.create_cluster",
                "-n",
                str(workers),
                "-m",
                str(memory_gib),
                "-cn",
                cluster_name,
                "--spot-policy",
                "on-demand",
            ]
        )
        launched_here = True

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        state, cluster_id = _cluster_state(cluster_name)
        print(f"Cluster {cluster_name}: state={state} id={cluster_id}")
        if state == "ready":
            return launched_here
        if state in {"error", "stopped"}:
            raise RuntimeError(f"Cluster {cluster_name} entered terminal state {state}")
        time.sleep(10)
    raise TimeoutError(f"Cluster {cluster_name} was not ready within {timeout_seconds}s")


def terminate_cluster(cluster_name: str) -> None:
    _run_command(
        [
            sys.executable,
            "-m",
            "src.scripts.utilities.terminate_cluster",
            cluster_name,
        ]
    )


def run_country_zonal(
    spec: AoiSpec,
    *,
    cluster_name: str,
    output_base_name: str,
    output_run_date: str,
    local_root: Path,
) -> Path:
    local_output = local_root / spec.iso3
    output_run_name = f"{output_base_name}_{spec.iso3.lower()}"
    command = [
        sys.executable,
        "-m",
        "src.scripts.zonal_statistics.02_run_zonal_stats",
        "--model_version",
        "1_0_1",
        "--run_date",
        "20260525",
        "--interval_type",
        "five_year",
        "--interval_end_years",
        "2020",
        "2024",
        "--cluster_name",
        cluster_name,
        "--run_name",
        "ogh_mixed_f1_f15_f2_20260513",
        "--output_run_name",
        output_run_name,
        "--output_run_date",
        output_run_date,
        "--local_output",
        str(local_output),
        "--keep_local",
        "--chunk_size",
        "10000",
        "--datasets",
        "drained_total",
        "burned_total",
        "--contextual_groupers",
        "all",
        "--execution_mode",
        "roi",
        "--diagnostics",
        "off",
        "--bounding_box",
        *(str(value) for value in spec.bbox),
    ]
    _run_command(command)
    return local_output


def _numeric_match(
    actual: float,
    expected: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[bool, float, float]:
    difference = actual - expected
    tolerance = max(absolute_tolerance, relative_tolerance * abs(expected))
    return abs(difference) <= tolerance, difference, tolerance


def compare_country_output(
    spec: AoiSpec,
    *,
    local_output: Path,
    engineer_countries: pd.DataFrame,
    require_area_match: bool,
    emissions_absolute_tolerance: float,
    emissions_relative_tolerance: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year, interval_directory in INTERVAL_DIRECTORIES.items():
        interval_path = local_output / "combined_state" / interval_directory
        completion_path = interval_path / "_COMPLETE.json"
        if not completion_path.is_file():
            raise FileNotFoundError(f"Missing completion marker: {completion_path}")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if not completion.get("success"):
            raise ValueError(f"Completion marker is not successful: {completion_path}")
        parquet_paths = sorted(interval_path.glob("*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"No output parquet found in {interval_path}")
        zonal = pd.concat((pd.read_parquet(path) for path in parquet_paths), ignore_index=True)
        country = zonal.loc[zonal["gadm_adm0"].astype(int).eq(spec.gadm_adm0)].copy()
        if country.empty:
            raise ValueError(
                f"No zonal rows found for {spec.iso3} ADM0={spec.gadm_adm0} in {interval_path}"
            )
        unexpected_wdpa = sorted(
            set(country["wdpa"].dropna().astype(int).unique()) - set(range(17))
        )
        if unexpected_wdpa:
            raise ValueError(f"Unexpected WDPA values for {spec.iso3}: {unexpected_wdpa}")

        engineer_match = engineer_countries.loc[
            engineer_countries["aoi_id"].eq(spec.iso3)
            & engineer_countries["interval_end_year"].eq(year)
        ]
        if len(engineer_match) != 1:
            raise ValueError(
                f"Expected one engineer row for {(spec.iso3, year)}, found {len(engineer_match)}"
            )
        engineer_row = engineer_match.iloc[0]
        zonal_emissions = float(
            country.loc[country["flux_type"].isin(EMISSIONS_FLUX_TYPES), "value"].sum()
        )
        engineer_emissions = float(engineer_row["gross_emissions_MgCO2e"])
        emissions_pass, emissions_difference, emissions_tolerance = _numeric_match(
            zonal_emissions,
            engineer_emissions,
            absolute_tolerance=emissions_absolute_tolerance,
            relative_tolerance=emissions_relative_tolerance,
        )
        emissions_relative_difference = (
            emissions_difference / engineer_emissions
            if engineer_emissions != 0
            else (0.0 if emissions_difference == 0 else np.nan)
        )

        full_area = float(
            country.loc[country["flux_type"].eq("area__ha"), "value"].sum()
        )
        organic_area_rows = country.loc[
            country["flux_type"].eq(ORGANIC_AREA_FLUX_TYPE), "value"
        ]
        organic_area = float(organic_area_rows.sum()) if not organic_area_rows.empty else None
        engineer_area = float(engineer_row["area_ha"])
        if organic_area is None:
            area_pass = not require_area_match
            area_difference = None
            area_tolerance = None
            area_status = "not_available_current_zonal_schema"
        else:
            area_pass, area_difference, area_tolerance = _numeric_match(
                organic_area,
                engineer_area,
                absolute_tolerance=0.01,
                relative_tolerance=1e-9,
            )
            area_status = "compared"

        result = {
            "aoi_id": spec.iso3,
            "gadm_adm0": spec.gadm_adm0,
            "purpose": spec.purpose,
            "interval_end_year": year,
            "engineer_emissions_MgCO2e": engineer_emissions,
            "zonal_emissions_MgCO2e": zonal_emissions,
            "emissions_difference_MgCO2e": emissions_difference,
            "emissions_relative_difference": emissions_relative_difference,
            "emissions_percent_difference": emissions_relative_difference * 100.0,
            "emissions_tolerance_MgCO2e": emissions_tolerance,
            "emissions_exact_match": bool(emissions_difference == 0.0),
            "emissions_pass": emissions_pass,
            "engineer_organic_area_ha": engineer_area,
            "zonal_full_admin_area_ha": full_area,
            "zonal_organic_soil_area_ha": organic_area,
            "organic_area_difference_ha": area_difference,
            "organic_area_tolerance_ha": area_tolerance,
            "area_status": area_status,
            "area_pass": area_pass,
            "wdpa_codes_present": ",".join(
                str(value) for value in sorted(country["wdpa"].dropna().astype(int).unique())
            ),
            "pass": bool(emissions_pass and area_pass),
        }
        rows.append(result)
        print(
            f"{'PASS' if result['pass'] else 'FAIL'} {spec.iso3} {year}: "
            f"emissions={zonal_emissions:.6f} engineer={engineer_emissions:.6f} "
            f"diff={emissions_difference:.6f} "
            f"pct_diff={emissions_relative_difference * 100.0:.9f}%; "
            f"exact={result['emissions_exact_match']}; area_status={area_status}"
        )
    return rows


def _planned_pixel_equivalents(specs: Iterable[AoiSpec]) -> float:
    return sum(
        (spec.bbox[2] - spec.bbox[0]) * (spec.bbox[3] - spec.bbox[1])
        for spec in specs
    )


def parse_args() -> argparse.Namespace:
    default_downloads = Path("/mnt/c/Users/Erin.Glen/Downloads")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="lean")
    parser.add_argument(
        "--engineer-organic-parquet",
        type=Path,
        default=default_downloads / "admin-land_ghg_inventory-organic_soil.parquet",
    )
    parser.add_argument(
        "--reference-parquet",
        type=Path,
        default=default_downloads / "LULUCF__v1_0_0__for_figures__wide__20260617.parquet",
    )
    parser.add_argument("--no-enforce-engineer-sha256", action="store_true")
    parser.add_argument(
        "--execute-zonal",
        action="store_true",
        help="Run the isolated zonal jobs. Without this flag, only cheap preflight checks run.",
    )
    parser.add_argument(
        "--compare-local-root",
        type=Path,
        help="Skip zonal execution and compare existing per-country local outputs under this root.",
    )
    parser.add_argument("--cluster-name", default="wdpa_preflight")
    parser.add_argument("--launch-cluster", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--memory-gib", type=int, default=64)
    parser.add_argument(
        "--keep-cluster",
        action="store_true",
        help="Do not terminate a cluster launched by this command.",
    )
    parser.add_argument(
        "--require-area-match",
        action="store_true",
        help=(
            "Fail unless zonal output contains organic_soil_area__ha matching the engineer. "
            "Current production schema has full-area area__ha only."
        ),
    )
    parser.add_argument(
        "--emissions-absolute-tolerance",
        type=float,
        default=DEFAULT_EMISSIONS_ABSOLUTE_TOLERANCE,
        help="Absolute emissions compatibility tolerance in Mg CO2e (default: 0.1).",
    )
    parser.add_argument(
        "--emissions-relative-tolerance",
        type=float,
        default=DEFAULT_EMISSIONS_RELATIVE_TOLERANCE,
        help="Relative emissions compatibility tolerance (default: 1e-5, or 0.001%%).",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=None,
        help="Local WSL staging/report root. Defaults under /mnt/c/tmp/afolu.",
    )
    parser.add_argument("--output-base-name", default=None)
    args = parser.parse_args()
    if args.execute_zonal and args.compare_local_root is not None:
        parser.error("--execute-zonal and --compare-local-root are mutually exclusive")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    if args.emissions_absolute_tolerance < 0:
        parser.error("--emissions-absolute-tolerance must be non-negative")
    if args.emissions_relative_tolerance < 0:
        parser.error("--emissions-relative-tolerance must be non-negative")
    return args


def require_linux_for_coiled_execution(*, execute_zonal: bool) -> None:
    if execute_zonal and platform.system() != "Linux":
        raise RuntimeError(
            "Coiled engineer-comparison runs must be launched from WSL/Linux using the "
            "coiled_20251119 environment."
        )


def main() -> int:
    args = parse_args()
    require_linux_for_coiled_execution(execute_zonal=args.execute_zonal)
    started = time.monotonic()
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    output_run_date = now.strftime("%Y%m%d")
    output_base_name = args.output_base_name or f"wdpa_engineer_preflight_{args.profile}_{timestamp}"
    local_root = args.local_root or Path(
        f"/mnt/c/tmp/afolu/wdpa_engineer_preflight/{output_base_name}"
    )

    engineer_countries = validate_engineer_artifact(
        args.engineer_organic_parquet,
        enforce_sha256=not args.no_enforce_engineer_sha256,
    )
    validate_reference_signature(args.reference_parquet, engineer_countries)
    validate_synthetic_wdpa_contract()

    specs = [AOIS[iso3] for iso3 in PROFILES[args.profile]]
    missing_targets = sorted(
        set(PROFILES[args.profile]) - set(engineer_countries["aoi_id"].unique())
    )
    if missing_targets:
        raise ValueError(f"Engineer parquet is missing profile targets: {missing_targets}")

    equivalent_square_degrees = _planned_pixel_equivalents(specs)
    print(f"Profile {args.profile}: {[spec.iso3 for spec in specs]}")
    print(
        "Planned bbox area (cost proxy): "
        f"{equivalent_square_degrees:.1f} square-degrees x 2 intervals"
    )
    if not args.execute_zonal and args.compare_local_root is None:
        estimate = "30-60 minutes" if args.profile == "lean" else "60-120 minutes"
        print(f"PLAN ONLY: estimated zonal runtime on the recommended cluster is {estimate}.")
        print("Add --execute-zonal --launch-cluster to run the full sequence.")
        return 0

    local_root.mkdir(parents=True, exist_ok=True)
    launched_here = False
    comparison_rows: list[dict[str, Any]] = []
    try:
        if args.execute_zonal:
            workers = args.workers or (50 if args.profile == "lean" else 75)
            launched_here = ensure_cluster_ready(
                args.cluster_name,
                launch=args.launch_cluster,
                workers=workers,
                memory_gib=args.memory_gib,
            )
            for spec in specs:
                country_local_output = run_country_zonal(
                    spec,
                    cluster_name=args.cluster_name,
                    output_base_name=output_base_name,
                    output_run_date=output_run_date,
                    local_root=local_root,
                )
                comparison_rows.extend(
                    compare_country_output(
                        spec,
                        local_output=country_local_output,
                        engineer_countries=engineer_countries,
                        require_area_match=args.require_area_match,
                        emissions_absolute_tolerance=args.emissions_absolute_tolerance,
                        emissions_relative_tolerance=args.emissions_relative_tolerance,
                    )
                )
        else:
            for spec in specs:
                comparison_rows.extend(
                    compare_country_output(
                        spec,
                        local_output=args.compare_local_root / spec.iso3,
                        engineer_countries=engineer_countries,
                        require_area_match=args.require_area_match,
                        emissions_absolute_tolerance=args.emissions_absolute_tolerance,
                        emissions_relative_tolerance=args.emissions_relative_tolerance,
                    )
                )
    finally:
        if launched_here and not args.keep_cluster:
            terminate_cluster(args.cluster_name)

    comparison = pd.DataFrame(comparison_rows)
    csv_path = local_root / "wdpa_engineer_country_comparison.csv"
    json_path = local_root / "wdpa_engineer_preflight_report.json"
    comparison.to_csv(csv_path, index=False)
    passed = bool(comparison["pass"].all())
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "aoi_specs": [asdict(spec) for spec in specs],
        "engineer_parquet": str(args.engineer_organic_parquet),
        "engineer_sha256": _sha256(args.engineer_organic_parquet),
        "reference_parquet": str(args.reference_parquet),
        "output_base_name": output_base_name,
        "output_run_date": output_run_date,
        "require_area_match": bool(args.require_area_match),
        "emissions_absolute_tolerance_MgCO2e": args.emissions_absolute_tolerance,
        "emissions_relative_tolerance": args.emissions_relative_tolerance,
        "emissions_relative_tolerance_percent": args.emissions_relative_tolerance * 100.0,
        "all_emissions_exact_match": bool(comparison["emissions_exact_match"].all()),
        "maximum_absolute_emissions_percent_difference": float(
            comparison["emissions_percent_difference"].abs().max()
        ),
        "comparison_csv": str(csv_path),
        "rows_compared": len(comparison),
        "passed": passed,
        "elapsed_seconds": time.monotonic() - started,
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Comparison CSV: {csv_path}")
    print(f"QA report: {json_path}")
    print(f"OVERALL {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
