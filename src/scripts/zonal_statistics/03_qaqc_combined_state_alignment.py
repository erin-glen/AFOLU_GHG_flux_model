# -*- coding: utf-8 -*-
"""QA/QC gate for combined-state zonal outputs.

For archived runs, compares `combined_state` zonal outputs from
02_run_zonal_stats against legacy `drained` and `burned` branch outputs. For
current single-branch runs where those legacy branches are intentionally absent,
the gate validates combined-state encoding and compares drained/burned
self-projections derived from the combined branch.

Example:
python -m src.scripts.zonal_statistics.03_qaqc_combined_state_alignment \
  --model_version 0_1_1 \
  --run_name zarr_test \
  --run_date 20260330 \
  --intervals 2021_2024
"""

from __future__ import annotations

import argparse
import json
import posixpath
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import s3fs

from src.scripts.zonal_statistics import zonal_constants as zc

ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
DEFAULT_EMISSIONS_ABS_TOL = 1e-6
DEFAULT_EMISSIONS_REL_TOL = 1e-6
DEFAULT_AREA_ABS_TOL = 1e-6
DEFAULT_AREA_REL_TOL = 1e-5
MANIFEST_FILENAME = "_zonal_stats_manifest.json"


@dataclass
class BranchFrames:
    combined: pd.DataFrame
    drained: pd.DataFrame
    burned: pd.DataFrame


def _read_parquet_df(path: str, cols: List[str]) -> pd.DataFrame:
    dset = ds.dataset(path, format="parquet")
    existing = [c for c in cols if c in dset.schema.names]
    if not existing:
        return pd.DataFrame(columns=cols)
    return dset.to_table(columns=existing).to_pandas()


def _exists_parquet(path: str) -> bool:
    try:
        dset = ds.dataset(path, format="parquet")
        return len(dset.files) > 0
    except Exception:
        return False


def _read_manifest(path: str) -> Optional[Dict[str, Any]]:
    fs_s3 = s3fs.S3FileSystem(anon=False)
    manifest_path = posixpath.join(path.rstrip("/"), MANIFEST_FILENAME)
    try:
        if not fs_s3.exists(manifest_path):
            return None
        with fs_s3.open(manifest_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _derive_expected_flux_types(manifest: Optional[Dict[str, Any]]) -> Set[str]:
    selected_fluxes = manifest.get("selected_fluxes", []) if manifest else []
    out: Set[str] = {"area__ha"}
    for flux_key in selected_fluxes:
        label = zc.ZONAL_FLUX_LABELS_BY_KEY.get(flux_key)
        if label:
            out.add(label)
    return out


def _resolve_paths(model_version: str, run_name: str, run_date: str, interval: str) -> Tuple[str, str, str]:
    root = posixpath.join(ROOT, f"version_{model_version}", "zonal_stats", run_name, run_date, interval)
    combined = posixpath.join(root, "combined_state")
    drained = posixpath.join(root, "drained")
    drained_alt = posixpath.join(root, "drained_co2_n2o")
    burned = posixpath.join(root, "burned")

    if not _exists_parquet(combined):
        legacy_combined = posixpath.join(root, "emissions_state")
        if _exists_parquet(legacy_combined):
            combined = legacy_combined

    if not _exists_parquet(drained) and _exists_parquet(drained_alt):
        drained = drained_alt

    return combined, drained, burned


def _load_interval_frames(model_version: str, run_name: str, run_date: str, interval: str) -> BranchFrames:
    combined_path, drained_path, burned_path = _resolve_paths(model_version, run_name, run_date, interval)

    combined = _read_parquet_df(
        combined_path,
        ["gadm_adm0", "interval_end", "flux_type", "value", "combined_state_nodes", "emissions_state_nodes", "drained_state_nodes", "burned_state_nodes"],
    )
    if combined.empty:
        raise FileNotFoundError(f"No combined-state parquet rows found for interval={interval}: {combined_path}")
    if "combined_state_nodes" not in combined.columns:
        if "emissions_state_nodes" in combined.columns:
            combined["combined_state_nodes"] = combined["emissions_state_nodes"]
        else:
            raise ValueError("Combined parquet missing both combined_state_nodes and emissions_state_nodes")

    if "drained_state_nodes" not in combined.columns or "burned_state_nodes" not in combined.columns:
        arr = combined["combined_state_nodes"].fillna(0).astype("uint32").to_numpy(copy=False)
        d_dec, b_dec = zc.unpack_combined_state_to_legacy(arr)
        if "drained_state_nodes" not in combined.columns:
            combined["drained_state_nodes"] = d_dec.astype("uint32", copy=False)
        if "burned_state_nodes" not in combined.columns:
            combined["burned_state_nodes"] = b_dec.astype("uint32", copy=False)

    if _exists_parquet(drained_path):
        drained = _read_parquet_df(
            drained_path,
            ["gadm_adm0", "interval_end", "flux_type", "value", "drained_state_nodes"],
        )
    else:
        drained = combined[
            ["gadm_adm0", "interval_end", "flux_type", "value", "drained_state_nodes"]
        ].copy()

    if _exists_parquet(burned_path):
        burned = _read_parquet_df(
            burned_path,
            ["gadm_adm0", "interval_end", "flux_type", "value", "burned_state_nodes"],
        )
    else:
        burned = combined[
            ["gadm_adm0", "interval_end", "flux_type", "value", "burned_state_nodes"]
        ].copy()

    if drained.empty:
        raise FileNotFoundError(f"No drained/self-projected parquet rows found for interval={interval}: {drained_path}")
    if burned.empty:
        raise FileNotFoundError(f"No burned/self-projected parquet rows found for interval={interval}: {burned_path}")

    return BranchFrames(combined=combined, drained=drained, burned=burned)


def _decode_consistency(combined: pd.DataFrame) -> Dict[str, int]:
    # NOTE: This checks self-consistency of combined output encoding/decoding only;
    # it is not an independent parity check against legacy branch outputs.
    arr = combined["combined_state_nodes"].fillna(0).astype("uint32").to_numpy(copy=False)
    d_dec, b_dec = zc.unpack_combined_state_to_legacy(arr)

    out = {
        "combined_rows": int(len(combined)),
        "decode_drained_mismatch_rows": 0,
        "decode_burned_mismatch_rows": 0,
    }

    if "drained_state_nodes" in combined.columns:
        d_obs = combined["drained_state_nodes"].fillna(0).astype("uint32").to_numpy(copy=False)
        out["decode_drained_mismatch_rows"] = int(np.count_nonzero(d_obs != d_dec))

    if "burned_state_nodes" in combined.columns:
        b_obs = combined["burned_state_nodes"].fillna(0).astype("uint32").to_numpy(copy=False)
        out["decode_burned_mismatch_rows"] = int(np.count_nonzero(b_obs != b_dec))

    return out


def _aggregate_and_compare(
    combined: pd.DataFrame,
    branch: pd.DataFrame,
    state_col: str,
    value_tol: float,
    relative_tol: float,
    area_value_tol: float,
    area_relative_tol: float,
    expected_flux_types: Set[str],
) -> Dict[str, object]:
    keys = ["gadm_adm0", "interval_end", "flux_type", state_col]
    combined_flux_types_before = set(combined["flux_type"].dropna().unique().tolist())
    branch_flux_types_before = set(branch["flux_type"].dropna().unique().tolist())
    expected_present_flux_types = expected_flux_types & (combined_flux_types_before | branch_flux_types_before)
    missing_expected_flux_types_combined = sorted(expected_present_flux_types - combined_flux_types_before)
    missing_expected_flux_types_branch = sorted(expected_present_flux_types - branch_flux_types_before)

    c = combined.loc[combined["flux_type"].isin(expected_flux_types), ["gadm_adm0", "interval_end", "flux_type", "value", state_col]].copy()
    c = c.groupby(keys, dropna=False, as_index=False)["value"].sum().rename(columns={"value": "combined_value"})

    b = branch.loc[branch["flux_type"].isin(expected_flux_types), ["gadm_adm0", "interval_end", "flux_type", "value", state_col]].copy()
    b = b.groupby(keys, dropna=False, as_index=False)["value"].sum().rename(columns={"value": "branch_value"})

    combined_flux_types_after = set(c["flux_type"].dropna().unique().tolist())
    branch_flux_types_after = set(b["flux_type"].dropna().unique().tolist())
    compared_flux_types = sorted(combined_flux_types_after & branch_flux_types_after)
    dropped_combined_flux_types = sorted(combined_flux_types_before - combined_flux_types_after)
    dropped_branch_flux_types = sorted(branch_flux_types_before - branch_flux_types_after)

    c = c[c["flux_type"].isin(compared_flux_types)].copy()
    b = b[b["flux_type"].isin(compared_flux_types)].copy()

    m = c.merge(b, on=keys, how="outer").fillna({"combined_value": 0.0, "branch_value": 0.0})
    m["abs_delta"] = (m["combined_value"] - m["branch_value"]).abs()
    area_mask = m["flux_type"].eq("area__ha")
    m["is_match"] = False
    m.loc[~area_mask, "is_match"] = np.isclose(
        m.loc[~area_mask, "combined_value"],
        m.loc[~area_mask, "branch_value"],
        atol=value_tol,
        rtol=relative_tol,
    )
    m.loc[area_mask, "is_match"] = np.isclose(
        m.loc[area_mask, "combined_value"],
        m.loc[area_mask, "branch_value"],
        atol=area_value_tol,
        rtol=area_relative_tol,
    )

    mismatches = m[~m["is_match"]]
    mismatch_counts_by_flux_type = {
        str(k): int(v)
        for k, v in mismatches.groupby("flux_type", dropna=False).size().sort_values(ascending=False).items()
    }
    max_delta = float(m["abs_delta"].max()) if not m.empty else 0.0
    total_abs_delta = float(m["abs_delta"].sum()) if not m.empty else 0.0
    top_mismatches = (
        mismatches.sort_values("abs_delta", ascending=False)
        .head(10)[keys + ["combined_value", "branch_value", "abs_delta"]]
        .to_dict(orient="records")
    )

    return {
        "rows_compared": int(len(m)),
        "mismatch_rows": int(len(mismatches)),
        "max_abs_delta": max_delta,
        "total_abs_delta": total_abs_delta,
        "compared_flux_types": compared_flux_types,
        "dropped_combined_flux_types": dropped_combined_flux_types,
        "dropped_branch_flux_types": dropped_branch_flux_types,
        "missing_expected_flux_types_combined": missing_expected_flux_types_combined,
        "missing_expected_flux_types_branch": missing_expected_flux_types_branch,
        "mismatch_counts_by_flux_type": mismatch_counts_by_flux_type,
        "top_mismatches": top_mismatches,
    }


def evaluate_interval(
    model_version: str,
    run_name: str,
    run_date: str,
    interval: str,
    value_tol: float,
    relative_tol: float,
    area_value_tol: float,
    area_relative_tol: float,
) -> Dict[str, object]:
    combined_path, drained_path, burned_path = _resolve_paths(model_version, run_name, run_date, interval)
    frames = _load_interval_frames(model_version, run_name, run_date, interval)
    combined = frames.combined.copy()
    combined_manifest = _read_manifest(combined_path)
    drained_manifest = _read_manifest(drained_path)
    burned_manifest = _read_manifest(burned_path)
    drained_expected_flux_types = _derive_expected_flux_types(drained_manifest)
    burned_expected_flux_types = _derive_expected_flux_types(burned_manifest)
    combined_expected_flux_types = _derive_expected_flux_types(combined_manifest)
    combined_expected_drained_flux_types = {"area__ha"} | {
        flux for flux in combined_expected_flux_types if flux.startswith("drained_")
    }
    combined_expected_burned_flux_types = {"area__ha"} | {
        flux for flux in combined_expected_flux_types if flux.startswith("burned_")
    }
    drained_expected_flux_types |= combined_expected_drained_flux_types
    burned_expected_flux_types |= combined_expected_burned_flux_types

    # Ensure drained/burned node columns exist in combined (decode if missing).
    if "drained_state_nodes" not in combined.columns or "burned_state_nodes" not in combined.columns:
        arr = combined["combined_state_nodes"].fillna(0).astype("uint32").to_numpy(copy=False)
        d_dec, b_dec = zc.unpack_combined_state_to_legacy(arr)
        if "drained_state_nodes" not in combined.columns:
            combined["drained_state_nodes"] = d_dec.astype("uint32", copy=False)
        if "burned_state_nodes" not in combined.columns:
            combined["burned_state_nodes"] = b_dec.astype("uint32", copy=False)

    decode_stats = _decode_consistency(combined)
    drained_stats = _aggregate_and_compare(
        combined,
        frames.drained,
        "drained_state_nodes",
        value_tol,
        relative_tol,
        area_value_tol,
        area_relative_tol,
        drained_expected_flux_types,
    )
    burned_stats = _aggregate_and_compare(
        combined,
        frames.burned,
        "burned_state_nodes",
        value_tol,
        relative_tol,
        area_value_tol,
        area_relative_tol,
        burned_expected_flux_types,
    )

    interval_pass = (
        drained_stats["rows_compared"] > 0
        and burned_stats["rows_compared"] > 0
        and len(drained_stats["missing_expected_flux_types_combined"]) == 0
        and len(drained_stats["missing_expected_flux_types_branch"]) == 0
        and len(burned_stats["missing_expected_flux_types_combined"]) == 0
        and len(burned_stats["missing_expected_flux_types_branch"]) == 0
        and drained_stats["mismatch_rows"] == 0
        and burned_stats["mismatch_rows"] == 0
    )

    return {
        "interval": interval,
        "pass": interval_pass,
        "decode_self_consistency": decode_stats,
        "drained_alignment": drained_stats,
        "burned_alignment": burned_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="QA/QC gate for combined-state zonal outputs")
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--run_date", required=True)
    parser.add_argument("--intervals", nargs="+", required=True, help="Inventory intervals, e.g. 2021_2024")
    parser.add_argument("--value_tolerance", type=float, default=DEFAULT_EMISSIONS_ABS_TOL)
    parser.add_argument("--relative_tolerance", type=float, default=DEFAULT_EMISSIONS_REL_TOL)
    parser.add_argument("--area_value_tolerance", type=float, default=DEFAULT_AREA_ABS_TOL)
    parser.add_argument("--area_relative_tolerance", type=float, default=DEFAULT_AREA_REL_TOL)
    parser.add_argument("--report_json", default=None, help="Optional path to write JSON report")
    args = parser.parse_args()

    results = []
    for interval in args.intervals:
        results.append(
            evaluate_interval(
                model_version=args.model_version,
                run_name=args.run_name,
                run_date=args.run_date,
                interval=interval,
                value_tol=float(args.value_tolerance),
                relative_tol=float(args.relative_tolerance),
                area_value_tol=float(args.area_value_tolerance),
                area_relative_tol=float(args.area_relative_tolerance),
            )
        )

    proceed = all(r["pass"] for r in results)
    summary = {
        "proceed_with_deprecation": proceed,
        "interval_results": results,
    }

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("YES - Proceed with deprecation" if proceed else "NO - Do not deprecate yet")

    if not proceed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
