# -*- coding: utf-8 -*-
"""QA/QC gate for combined-state zonal outputs.

Compares `combined_state` zonal outputs from 02_run_zonal_stats against legacy
`drained` and `burned` branch outputs and returns a YES/NO recommendation for
proceeding with deprecation of legacy state-node outputs.

Example:
python -m src.scripts.zonal_statistics.03_qaqc_combined_state_alignment \
  --model_version 0_9_7 \
  --run_name ogh_standard_model \
  --run_date 20251118 \
  --intervals 2021_2024
"""

from __future__ import annotations

import argparse
import json
import posixpath
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from src.scripts.zonal_statistics import zonal_constants as zc

ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"


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
    drained = _read_parquet_df(
        drained_path,
        ["gadm_adm0", "interval_end", "flux_type", "value", "drained_state_nodes"],
    )
    burned = _read_parquet_df(
        burned_path,
        ["gadm_adm0", "interval_end", "flux_type", "value", "burned_state_nodes"],
    )

    if combined.empty:
        raise FileNotFoundError(f"No combined-state parquet rows found for interval={interval}: {combined_path}")
    if drained.empty:
        raise FileNotFoundError(f"No drained parquet rows found for interval={interval}: {drained_path}")
    if burned.empty:
        raise FileNotFoundError(f"No burned parquet rows found for interval={interval}: {burned_path}")

    if "combined_state_nodes" not in combined.columns:
        if "emissions_state_nodes" in combined.columns:
            combined["combined_state_nodes"] = combined["emissions_state_nodes"]
        else:
            raise ValueError("Combined parquet missing both combined_state_nodes and emissions_state_nodes")

    return BranchFrames(combined=combined, drained=drained, burned=burned)


def _decode_consistency(combined: pd.DataFrame) -> Dict[str, int]:
    arr = combined["combined_state_nodes"].fillna(0).astype("uint32").to_numpy(copy=False)
    d_dec, b_dec = zc.unpack_emissions_state_to_legacy(arr)

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
) -> Dict[str, float]:
    keys = ["gadm_adm0", "interval_end", "flux_type", state_col]

    c = combined[["gadm_adm0", "interval_end", "flux_type", "value", state_col]].copy()
    c = c.groupby(keys, dropna=False, as_index=False)["value"].sum().rename(columns={"value": "combined_value"})

    b = branch[["gadm_adm0", "interval_end", "flux_type", "value", state_col]].copy()
    b = b.groupby(keys, dropna=False, as_index=False)["value"].sum().rename(columns={"value": "branch_value"})

    m = c.merge(b, on=keys, how="outer").fillna({"combined_value": 0.0, "branch_value": 0.0})
    m["abs_delta"] = (m["combined_value"] - m["branch_value"]).abs()

    mismatches = m[m["abs_delta"] > value_tol]
    max_delta = float(m["abs_delta"].max()) if not m.empty else 0.0
    total_abs_delta = float(m["abs_delta"].sum()) if not m.empty else 0.0

    return {
        "rows_compared": int(len(m)),
        "mismatch_rows": int(len(mismatches)),
        "max_abs_delta": max_delta,
        "total_abs_delta": total_abs_delta,
    }


def evaluate_interval(model_version: str, run_name: str, run_date: str, interval: str, value_tol: float) -> Dict[str, object]:
    frames = _load_interval_frames(model_version, run_name, run_date, interval)
    combined = frames.combined.copy()

    # Ensure drained/burned node columns exist in combined (decode if missing).
    if "drained_state_nodes" not in combined.columns or "burned_state_nodes" not in combined.columns:
        arr = combined["combined_state_nodes"].fillna(0).astype("uint32").to_numpy(copy=False)
        d_dec, b_dec = zc.unpack_emissions_state_to_legacy(arr)
        if "drained_state_nodes" not in combined.columns:
            combined["drained_state_nodes"] = d_dec.astype("uint32", copy=False)
        if "burned_state_nodes" not in combined.columns:
            combined["burned_state_nodes"] = b_dec.astype("uint32", copy=False)

    decode_stats = _decode_consistency(combined)
    drained_stats = _aggregate_and_compare(combined, frames.drained, "drained_state_nodes", value_tol)
    burned_stats = _aggregate_and_compare(combined, frames.burned, "burned_state_nodes", value_tol)

    interval_pass = (
        decode_stats["decode_drained_mismatch_rows"] == 0
        and decode_stats["decode_burned_mismatch_rows"] == 0
        and drained_stats["mismatch_rows"] == 0
        and burned_stats["mismatch_rows"] == 0
    )

    return {
        "interval": interval,
        "pass": interval_pass,
        "decode_consistency": decode_stats,
        "drained_alignment": drained_stats,
        "burned_alignment": burned_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="QA/QC gate for combined-state zonal outputs")
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--run_date", required=True)
    parser.add_argument("--intervals", nargs="+", required=True, help="Inventory intervals, e.g. 2021_2024")
    parser.add_argument("--value_tolerance", type=float, default=1e-6)
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
