#!/usr/bin/env python3
"""
Create threshold-response diagnostics from a hold-out validation table and,
optionally, compute validation-based area bounds for an operational threshold.

Expected columns in the input CSV:
    - organic: binary reference label (0 = non-organic, 1 = organic)
    - predicted_organic_probability: continuous predicted probability for the
      organic class

Core outputs:
    1. threshold_metrics.csv: confusion-matrix counts and threshold-response
       metrics across all unique score thresholds;
    2. selected_thresholds.csv: direct evaluation of any user-specified
       thresholds;
    3. f1_f2_vs_threshold.png: F1 and F2 vs. probability threshold;
    4. precision_recall_vs_threshold.png: precision and recall vs. probability
       threshold.

The threshold metrics table now includes the following fields for the organic
class at each threshold:
    - precision, recall, F1, F2, accuracy
    - UA/PA aliases (ua, pa)
    - commission_error, omission_error
    - specificity, balanced_accuracy
    - Matthews correlation coefficient (mcc)
    - TP, FP, FN, TN

Optional area-bound outputs (when --mapped-area is supplied):
    5. extent_bounds_summary.csv: lower and upper area bounds based on
       A_low ≈ A_map × UA and A_high ≈ A_map / PA, where UA is mapped to
       precision and PA to recall at the operational threshold.

Optional threshold-matching outputs (when --area-curve-table is supplied):
    6. area_bound_threshold_matches.csv: the threshold in the area-vs-threshold
       table whose mapped area is closest to each target bound, while preferring
       the methodologically appropriate direction relative to the operational
       threshold.
    7. area_vs_threshold_with_bounds.png: area-vs-threshold curve annotated with
       the mapped area at the operational threshold, target bounds, and matched
       thresholds.

Examples
--------
python src/scripts/uncertainty/fscore_threshold_curves_bounds.py \
    --input "/mnt/c/tmp/uncertainty/USDA_testpoints_probability.csv" \
    --output-dir "/mnt/c/tmp/uncertainty/" \
    --report-thresholds 0.23

python src/scripts/uncertainty/fscore_threshold_curves_bounds.py \
    --input "/mnt/c/tmp/uncertainty/USDA_testpoints_probability.csv" \
    --output-dir "/mnt/c/tmp/uncertainty/" \
    --report-thresholds 0.23 \
    --mapped-area 120.5 \
    --mapped-area-unit Mha \
    --bounds-threshold 0.23

python src/scripts/uncertainty/fscore_threshold_curves_bounds.py \
    --input "/mnt/c/tmp/uncertainty/USDA_testpoints_probability.csv" \
    --output-dir "/mnt/c/tmp/uncertainty/" \
    --report-thresholds 0.23 \
    --mapped-area 120.5 \
    --mapped-area-unit Mha \
    --bounds-threshold 0.23 \
    --area-curve-table "/mnt/c/tmp/uncertainty/area_vs_threshold_dummy.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OPERATIONAL_THRESHOLD = 0.23


METRIC_COLUMNS_ORDER = [
    "precision",
    "recall",
    "ua",
    "pa",
    "commission_error",
    "omission_error",
    "specificity",
    "balanced_accuracy",
    "mcc",
    "f1",
    "f2",
    "accuracy",
]


SUMMARY_COLUMNS = [
    "threshold",
    "precision",
    "recall",
    "ua",
    "pa",
    "commission_error",
    "omission_error",
    "specificity",
    "balanced_accuracy",
    "mcc",
    "f1",
    "f2",
    "accuracy",
    "tp",
    "fp",
    "fn",
    "tn",
]


BEST_PRINT_COLUMNS = [
    "threshold",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "mcc",
    "f1",
    "f2",
    "accuracy",
]


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise division that returns 0 where the denominator is 0."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.zeros_like(numerator, dtype=float)
    mask = denominator != 0
    out[mask] = numerator[mask] / denominator[mask]
    return out


def nan_where_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise division that returns NaN where the denominator is 0."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full_like(numerator, np.nan, dtype=float)
    mask = denominator != 0
    out[mask] = numerator[mask] / denominator[mask]
    return out


def fbeta_score(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    """Compute the F-beta score from precision and recall."""
    beta2 = beta ** 2
    return safe_divide((1 + beta2) * precision * recall, beta2 * precision + recall)


def matthews_correlation(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    tn: np.ndarray,
) -> np.ndarray:
    """Compute Matthews correlation coefficient from confusion-matrix counts."""
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return safe_divide(numerator, denominator)


def add_derived_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add threshold-response metrics derived from confusion-matrix counts."""
    out = df.copy()

    tp = out["tp"].to_numpy(dtype=float)
    fp = out["fp"].to_numpy(dtype=float)
    fn = out["fn"].to_numpy(dtype=float)
    tn = out["tn"].to_numpy(dtype=float)
    total = tp + fp + fn + tn

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    accuracy = safe_divide(tp + tn, total)
    f1 = fbeta_score(precision, recall, beta=1.0)
    f2 = fbeta_score(precision, recall, beta=2.0)
    balanced_accuracy = (recall + specificity) / 2.0
    mcc = matthews_correlation(tp, fp, fn, tn)

    out["precision"] = precision
    out["recall"] = recall
    out["ua"] = precision
    out["pa"] = recall
    out["commission_error"] = nan_where_divide(fp, tp + fp)
    out["omission_error"] = nan_where_divide(fn, tp + fn)
    out["specificity"] = specificity
    out["balanced_accuracy"] = balanced_accuracy
    out["mcc"] = mcc
    out["f1"] = f1
    out["f2"] = f2
    out["accuracy"] = accuracy

    return out


def evaluate_single_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Evaluate confusion-matrix counts and metrics at one threshold."""
    y_pred = (scores >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    row = pd.DataFrame(
        [
            {
                "threshold": float(threshold),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        ]
    )
    row = add_derived_metric_columns(row)
    for count_col in ["tp", "fp", "fn", "tn"]:
        row[count_col] = row[count_col].astype(int)
    return row.iloc[0].to_dict()


def compute_threshold_metrics(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Compute exact threshold-response metrics at all unique score thresholds."""
    if y_true.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional arrays.")
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length.")
    if len(y_true) == 0:
        raise ValueError("No valid rows remain after filtering missing values.")

    unique_labels = np.unique(y_true)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError(
            "The reference label column must be binary (0/1). "
            f"Found values: {unique_labels.tolist()}"
        )

    n_total = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = n_total - n_pos

    order = np.argsort(-scores, kind="mergesort")
    scores_sorted = scores[order]
    y_sorted = y_true[order]

    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)

    is_last_in_tied_block = np.r_[scores_sorted[1:] != scores_sorted[:-1], True]
    last_idx = np.flatnonzero(is_last_in_tied_block)

    thresholds = scores_sorted[last_idx]
    tp = tp_cum[last_idx]
    fp = fp_cum[last_idx]
    fn = n_pos - tp
    tn = n_neg - fp

    metrics = pd.DataFrame(
        {
            "threshold": thresholds,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    )

    boundary_rows: list[dict[str, float | int]] = []

    if float(metrics["threshold"].max()) < 1.0:
        boundary_rows.append(
            {
                "threshold": 1.0,
                "tp": 0,
                "fp": 0,
                "fn": n_pos,
                "tn": n_neg,
            }
        )

    if float(metrics["threshold"].min()) > 0.0:
        boundary_rows.append(
            {
                "threshold": 0.0,
                "tp": n_pos,
                "fp": n_neg,
                "fn": 0,
                "tn": 0,
            }
        )

    if boundary_rows:
        metrics = pd.concat([pd.DataFrame(boundary_rows), metrics], ignore_index=True)

    metrics = add_derived_metric_columns(metrics)
    metrics = metrics.drop_duplicates(subset="threshold", keep="first")
    metrics = metrics.sort_values("threshold", ascending=False).reset_index(drop=True)
    return metrics


def add_threshold_reference_lines(
    ax: plt.Axes,
    report_thresholds: Iterable[float],
    operational_threshold: float,
    best_f1_threshold: float,
    best_f2_threshold: float,
) -> None:
    """Add common vertical reference lines to a threshold-response plot."""
    seen: set[tuple[float, str, str]] = set()

    for threshold in report_thresholds:
        key = (float(threshold), "--", "Reported threshold")
        if key not in seen:
            ax.axvline(float(threshold), linestyle="--", linewidth=1, alpha=0.6, label="Reported threshold")
            seen.add(key)

    operational_key = (float(operational_threshold), "-.", f"Operational threshold = {operational_threshold:.4f}")
    if operational_key not in seen:
        ax.axvline(
            float(operational_threshold),
            linestyle="-.",
            linewidth=1.2,
            alpha=0.9,
            label=f"Operational threshold = {operational_threshold:.4f}",
        )
        seen.add(operational_key)

    best_f1_key = (float(best_f1_threshold), ":", f"Best F1 threshold = {best_f1_threshold:.4f}")
    if best_f1_key not in seen:
        ax.axvline(
            float(best_f1_threshold),
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
            label=f"Best F1 threshold = {best_f1_threshold:.4f}",
        )
        seen.add(best_f1_key)

    best_f2_key = (float(best_f2_threshold), (0, (5, 2)), f"Best F2 threshold = {best_f2_threshold:.4f}")
    if best_f2_key not in seen:
        ax.axvline(
            float(best_f2_threshold),
            linestyle=(0, (5, 2)),
            linewidth=1.2,
            alpha=0.9,
            label=f"Best F2 threshold = {best_f2_threshold:.4f}",
        )
        seen.add(best_f2_key)


def plot_metric_curves(
    metrics: pd.DataFrame,
    output_path: Path,
    report_thresholds: Iterable[float],
    operational_threshold: float,
    best_f1_threshold: float,
    best_f2_threshold: float,
    metric_columns: Sequence[str],
    metric_labels: Sequence[str],
    title: str,
    y_label: str = "Score",
) -> None:
    """Save a figure of selected metrics as a function of threshold."""
    if len(metric_columns) != len(metric_labels):
        raise ValueError("metric_columns and metric_labels must have the same length.")

    fig, ax = plt.subplots(figsize=(8, 5))

    for column, label in zip(metric_columns, metric_labels):
        ax.plot(metrics["threshold"], metrics[column], label=label)

    add_threshold_reference_lines(
        ax=ax,
        report_thresholds=report_thresholds,
        operational_threshold=operational_threshold,
        best_f1_threshold=best_f1_threshold,
        best_f2_threshold=best_f2_threshold,
    )

    ax.set_xlabel("Probability threshold")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def resolve_bounds_threshold(bounds_threshold: float | None, report_thresholds: Sequence[float]) -> float:
    """Resolve which threshold to use for the mapped-area bound calculation."""
    if bounds_threshold is not None:
        return float(bounds_threshold)
    if report_thresholds:
        return float(report_thresholds[0])
    return DEFAULT_OPERATIONAL_THRESHOLD


def compute_extent_bounds(
    mapped_area: float,
    area_unit: str,
    operational_threshold: float,
    operational_metrics: dict[str, float],
) -> pd.DataFrame:
    """Compute lower and upper area bounds from mapped area and validation metrics."""
    if mapped_area < 0:
        raise ValueError("Mapped area must be non-negative.")

    ua_precision = float(operational_metrics["ua"])
    pa_recall = float(operational_metrics["pa"])

    lower_area = mapped_area * ua_precision
    upper_area = np.inf if pa_recall <= 0 else mapped_area / pa_recall

    summary = pd.DataFrame(
        [
            {
                "operational_threshold": operational_threshold,
                "mapped_area": mapped_area,
                "area_unit": area_unit,
                "ua_precision": ua_precision,
                "pa_recall": pa_recall,
                "commission_error": float(operational_metrics["commission_error"])
                if pd.notna(operational_metrics["commission_error"])
                else np.nan,
                "omission_error": float(operational_metrics["omission_error"])
                if pd.notna(operational_metrics["omission_error"])
                else np.nan,
                "specificity": float(operational_metrics["specificity"]),
                "balanced_accuracy": float(operational_metrics["balanced_accuracy"]),
                "mcc": float(operational_metrics["mcc"]),
                "lower_bound_area": lower_area,
                "upper_bound_area": upper_area,
                "lower_bound_multiplier": ua_precision,
                "upper_bound_multiplier": np.inf if pa_recall <= 0 else 1.0 / pa_recall,
                "bound_method": "A_low≈A_map×UA; A_high≈A_map/PA",
                "ua_definition": "organic-class precision at operational threshold",
                "pa_definition": "organic-class recall at operational threshold",
                "tp": int(operational_metrics["tp"]),
                "fp": int(operational_metrics["fp"]),
                "fn": int(operational_metrics["fn"]),
                "tn": int(operational_metrics["tn"]),
                "accuracy": float(operational_metrics["accuracy"]),
                "f1": float(operational_metrics["f1"]),
                "f2": float(operational_metrics["f2"]),
            }
        ]
    )
    return summary


def read_area_curve_table(
    area_curve_path: Path,
    threshold_column: str,
    area_column: str,
) -> pd.DataFrame:
    """Read and validate a threshold-vs-area table."""
    if not area_curve_path.exists():
        raise FileNotFoundError(f"Area-curve table not found: {area_curve_path}")

    area_df = pd.read_csv(area_curve_path)
    missing_columns = [col for col in [threshold_column, area_column] if col not in area_df.columns]
    if missing_columns:
        raise KeyError(
            "Required columns were not found in the area-curve table: "
            + ", ".join(missing_columns)
            + f"\nAvailable columns: {area_df.columns.tolist()}"
        )

    area_df = area_df[[threshold_column, area_column]].copy().dropna()
    area_df = area_df.rename(columns={threshold_column: "threshold", area_column: "area"})
    area_df["threshold"] = area_df["threshold"].astype(float)
    area_df["area"] = area_df["area"].astype(float)
    area_df = area_df.sort_values("threshold").drop_duplicates(subset="threshold", keep="last").reset_index(drop=True)
    return area_df


def monotonicity_status(area_df: pd.DataFrame) -> str:
    """Return a short diagnostic about whether area declines with threshold."""
    diffs = np.diff(area_df["area"].to_numpy())
    if np.all(diffs <= 0):
        return "nonincreasing"
    if np.all(diffs >= 0):
        return "nondecreasing"
    return "nonmonotonic"


def match_target_area_to_threshold(
    area_df: pd.DataFrame,
    target_area: float,
    operational_threshold: float,
    direction: str,
    target_name: str,
) -> dict[str, float | int | str]:
    """Find the threshold whose mapped area is closest to the target area."""
    if direction not in {"higher_threshold", "lower_threshold"}:
        raise ValueError("direction must be 'higher_threshold' or 'lower_threshold'.")

    if direction == "higher_threshold":
        directional = area_df.loc[area_df["threshold"] >= operational_threshold].copy()
    else:
        directional = area_df.loc[area_df["threshold"] <= operational_threshold].copy()

    status = "matched_with_directional_candidates"
    candidates = directional
    if candidates.empty:
        candidates = area_df.copy()
        status = "no_directional_candidates_used_nearest_overall"

    candidates["abs_area_difference"] = (candidates["area"] - target_area).abs()
    candidates["abs_threshold_distance_from_operational"] = (candidates["threshold"] - operational_threshold).abs()
    candidates = candidates.sort_values(
        by=["abs_area_difference", "abs_threshold_distance_from_operational", "threshold"],
        ascending=[True, True, True],
    )
    best = candidates.iloc[0]

    return {
        "target_name": target_name,
        "target_area": float(target_area),
        "direction_preference": direction,
        "selected_threshold": float(best["threshold"]),
        "selected_area": float(best["area"]),
        "signed_area_difference": float(best["area"] - target_area),
        "abs_area_difference": float(abs(best["area"] - target_area)),
        "relative_area_difference": float(abs(best["area"] - target_area) / target_area) if target_area != 0 else np.nan,
        "threshold_distance_from_operational": float(abs(best["threshold"] - operational_threshold)),
        "candidate_count_used": int(len(candidates)),
        "status": status,
    }


def match_area_bounds_to_thresholds(
    area_df: pd.DataFrame,
    bounds_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Match lower/upper target areas to thresholds in an area-vs-threshold table."""
    if bounds_summary.empty:
        raise ValueError("bounds_summary is empty.")

    row = bounds_summary.iloc[0]
    operational_threshold = float(row["operational_threshold"])
    lower_target = float(row["lower_bound_area"])
    upper_target = float(row["upper_bound_area"])

    matches = [
        match_target_area_to_threshold(
            area_df=area_df,
            target_area=lower_target,
            operational_threshold=operational_threshold,
            direction="higher_threshold",
            target_name="lower_bound_area",
        ),
        match_target_area_to_threshold(
            area_df=area_df,
            target_area=upper_target,
            operational_threshold=operational_threshold,
            direction="lower_threshold",
            target_name="upper_bound_area",
        ),
    ]

    out = pd.DataFrame(matches)
    out.insert(0, "operational_threshold", operational_threshold)
    out.insert(1, "mapped_area", float(row["mapped_area"]))
    out.insert(2, "area_unit", row["area_unit"])
    out.insert(3, "area_curve_monotonicity", monotonicity_status(area_df))
    return out


def plot_area_curve_with_bounds(
    area_df: pd.DataFrame,
    bounds_summary: pd.DataFrame,
    matches_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot area vs threshold, annotated with the operational threshold and bounds."""
    row = bounds_summary.iloc[0]
    operational_threshold = float(row["operational_threshold"])
    mapped_area = float(row["mapped_area"])
    lower_target = float(row["lower_bound_area"])
    upper_target = float(row["upper_bound_area"])
    area_unit = str(row["area_unit"])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        area_df["threshold"],
        area_df["area"],
        color="#1f77b4",
        linewidth=2.2,
        label="Area vs threshold",
        zorder=3,
    )

    lower_fill = min(lower_target, upper_target)
    upper_fill = max(lower_target, upper_target)
    ax.axhspan(
        lower_fill,
        upper_fill,
        facecolor="#a1d99b",
        alpha=0.18,
        label=f"Validation range [{lower_fill:.2f}, {upper_fill:.2f}] {area_unit}",
        zorder=0,
    )

    ax.axhline(
        mapped_area,
        color="#ff7f0e",
        linestyle=(0, (8, 3)),
        linewidth=2.0,
        label=f"Mapped area = {mapped_area:.4f} {area_unit}",
        zorder=2,
    )
    ax.axhline(
        lower_target,
        color="#2ca02c",
        linestyle=":",
        linewidth=2.0,
        label=f"Lower bound = {lower_target:.4f} {area_unit}",
        zorder=2,
    )
    ax.axhline(
        upper_target,
        color="#d62728",
        linestyle=":",
        linewidth=2.0,
        label=f"Upper bound = {upper_target:.4f} {area_unit}",
        zorder=2,
    )

    ax.axvline(
        operational_threshold,
        color="#9467bd",
        linestyle="-.",
        linewidth=1.8,
        label=f"Operational threshold = {operational_threshold:.4f}",
        zorder=2,
    )

    target_to_color = {
        "lower_bound_area": "#2ca02c",
        "upper_bound_area": "#d62728",
    }
    target_to_label = {
        "lower_bound_area": "Matched lower-bound threshold",
        "upper_bound_area": "Matched upper-bound threshold",
    }
    seen_match_labels: set[str] = set()
    for _, match in matches_df.iterrows():
        target_name = str(match["target_name"])
        color = target_to_color.get(target_name, "#7f7f7f")
        base_label = target_to_label.get(target_name, f"Matched threshold ({target_name})")
        label = base_label if base_label not in seen_match_labels else "_nolegend_"
        seen_match_labels.add(base_label)

        selected_threshold = float(match["selected_threshold"])
        selected_area = float(match["selected_area"])
        ax.axvline(
            selected_threshold,
            color=color,
            linestyle=(0, (3, 2, 1, 2)),
            linewidth=1.6,
            alpha=0.9,
            label=label,
            zorder=1,
        )
        ax.scatter(
            [selected_threshold],
            [selected_area],
            color=color,
            edgecolor="white",
            linewidth=0.7,
            s=48,
            zorder=4,
        )

    ax.set_xlabel("Probability threshold")
    ax.set_ylabel(f"Mapped area ({area_unit})")
    ax.set_title("Area-versus-threshold curve with validation-based bounds")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_scalar_for_console(value: object) -> object:
    """Format scalar values for compact, readable console output."""
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return "nan"
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        if float(value).is_integer():
            return int(value)
        return f"{float(value):.6f}"
    return value



def format_vertical_series(series: pd.Series, key_order: Sequence[str] | None = None) -> str:
    """Return a one-record vertical string representation for console output."""
    if key_order is not None:
        series = series[[key for key in key_order if key in series.index]]
    formatted = pd.Series({key: format_scalar_for_console(value) for key, value in series.items()}, dtype=object)
    return formatted.to_string()


def print_vertical_dataframe_block(
    title: str,
    df: pd.DataFrame,
    key_order: Sequence[str],
    record_label_column: str | None = None,
) -> None:
    """Pretty-print one or more records in a vertical, row-wise console format."""
    print(title)
    if df.empty:
        print("<no rows>")
        return

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        if len(df) > 1:
            if record_label_column and record_label_column in row.index:
                print(f"[{i}] {record_label_column}={row[record_label_column]}")
            else:
                print(f"[{i}]")
        print(format_vertical_series(row, key_order=key_order))
        if i < len(df):
            print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create threshold-response diagnostics and optional extent bounds from a validation CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files. Defaults to '<input directory>/threshold_curve_outputs'.",
    )
    parser.add_argument(
        "--label-column",
        default="organic",
        help="Name of the binary reference-label column. Default: organic",
    )
    parser.add_argument(
        "--score-column",
        default="predicted_organic_probability",
        help="Name of the probability-score column. Default: predicted_organic_probability",
    )
    parser.add_argument(
        "--report-thresholds",
        nargs="*",
        type=float,
        default=[DEFAULT_OPERATIONAL_THRESHOLD],
        help=f"Specific thresholds to evaluate directly and report. Default: {DEFAULT_OPERATIONAL_THRESHOLD}",
    )
    parser.add_argument(
        "--mapped-area",
        type=float,
        default=None,
        help="Known mapped organic-soil area at the operational threshold. Optional.",
    )
    parser.add_argument(
        "--mapped-area-unit",
        default="Mha",
        help="Unit label for the mapped area. Default: Mha",
    )
    parser.add_argument(
        "--bounds-threshold",
        type=float,
        default=None,
        help=(
            "Operational threshold to use for the area-bound calculation. "
            "Defaults to the first value in --report-thresholds, or 0.23 if none is given."
        ),
    )
    parser.add_argument(
        "--area-curve-table",
        type=Path,
        default=None,
        help="Optional CSV with threshold and mapped-area columns for threshold matching.",
    )
    parser.add_argument(
        "--area-curve-threshold-column",
        default="threshold",
        help="Threshold column name in --area-curve-table. Default: threshold",
    )
    parser.add_argument(
        "--area-curve-area-column",
        default="area_ha",
        help="Mapped-area column name in --area-curve-table. Default: area_ha",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = args.output_dir.resolve() if args.output_dir else input_path.parent / "threshold_curve_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    missing_columns = [col for col in [args.label_column, args.score_column] if col not in df.columns]
    if missing_columns:
        raise KeyError(
            "Required columns were not found in the input file: "
            + ", ".join(missing_columns)
            + f"\nAvailable columns: {df.columns.tolist()}"
        )

    subset = df[[args.label_column, args.score_column]].copy()
    n_before = len(subset)
    subset = subset.dropna()
    n_after = len(subset)
    n_dropped = n_before - n_after

    y_true = subset[args.label_column].astype(int).to_numpy()
    scores = subset[args.score_column].astype(float).to_numpy()

    metrics = compute_threshold_metrics(y_true=y_true, scores=scores)
    for count_col in ["tp", "fp", "fn", "tn"]:
        metrics[count_col] = metrics[count_col].astype(int)
    metrics = metrics[["threshold", "tp", "fp", "fn", "tn", *METRIC_COLUMNS_ORDER]]
    metrics_path = output_dir / "threshold_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    report_thresholds = list(args.report_thresholds)
    reported_rows = [evaluate_single_threshold(y_true, scores, t) for t in report_thresholds]
    reported = pd.DataFrame(reported_rows)
    for count_col in ["tp", "fp", "fn", "tn"]:
        reported[count_col] = reported[count_col].astype(int)
    reported = reported[["threshold", "tp", "fp", "fn", "tn", *METRIC_COLUMNS_ORDER]]
    reported = reported.sort_values("threshold", ascending=False)
    reported_path = output_dir / "selected_thresholds.csv"
    reported.to_csv(reported_path, index=False)

    best_f1 = metrics.loc[metrics["f1"].idxmax()]
    best_f2 = metrics.loc[metrics["f2"].idxmax()]
    operational_threshold = resolve_bounds_threshold(args.bounds_threshold, report_thresholds)

    fscore_figure_path = output_dir / "f1_f2_vs_threshold.png"
    plot_metric_curves(
        metrics=metrics,
        output_path=fscore_figure_path,
        report_thresholds=report_thresholds,
        operational_threshold=operational_threshold,
        best_f1_threshold=float(best_f1["threshold"]),
        best_f2_threshold=float(best_f2["threshold"]),
        metric_columns=["f1", "f2"],
        metric_labels=["F1", "F2"],
        title="Threshold-response curves for organic-soil classification",
    )

    pr_figure_path = output_dir / "precision_recall_vs_threshold.png"
    plot_metric_curves(
        metrics=metrics,
        output_path=pr_figure_path,
        report_thresholds=report_thresholds,
        operational_threshold=operational_threshold,
        best_f1_threshold=float(best_f1["threshold"]),
        best_f2_threshold=float(best_f2["threshold"]),
        metric_columns=["precision", "recall"],
        metric_labels=["Precision (UA)", "Recall (PA)"],
        title="Precision and recall vs. probability threshold",
    )

    bounds_path: Path | None = None
    matches_path: Path | None = None
    area_plot_path: Path | None = None

    bounds_summary: pd.DataFrame | None = None
    matches_df: pd.DataFrame | None = None

    if args.mapped_area is not None:
        operational_metrics = evaluate_single_threshold(y_true, scores, operational_threshold)
        bounds_summary = compute_extent_bounds(
            mapped_area=float(args.mapped_area),
            area_unit=args.mapped_area_unit,
            operational_threshold=operational_threshold,
            operational_metrics=operational_metrics,
        )
        bounds_path = output_dir / "extent_bounds_summary.csv"
        bounds_summary.to_csv(bounds_path, index=False)

    if args.area_curve_table is not None:
        if bounds_summary is None:
            raise ValueError("--area-curve-table requires --mapped-area so the target bounds can be computed.")
        area_df = read_area_curve_table(
            area_curve_path=args.area_curve_table.resolve(),
            threshold_column=args.area_curve_threshold_column,
            area_column=args.area_curve_area_column,
        )
        matches_df = match_area_bounds_to_thresholds(area_df=area_df, bounds_summary=bounds_summary)
        matches_path = output_dir / "area_bound_threshold_matches.csv"
        matches_df.to_csv(matches_path, index=False)

        area_plot_path = output_dir / "area_vs_threshold_with_bounds.png"
        plot_area_curve_with_bounds(
            area_df=area_df,
            bounds_summary=bounds_summary,
            matches_df=matches_df,
            output_path=area_plot_path,
        )

    print("Finished.")
    print(f"Input file: {input_path}")
    print(f"Rows read: {n_before}")
    print(f"Rows used after dropping missing values: {n_after}")
    print(f"Rows dropped for missing label/score: {n_dropped}")
    print()
    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {reported_path}")
    print(f"Wrote: {fscore_figure_path}")
    print(f"Wrote: {pr_figure_path}")
    if bounds_path is not None:
        print(f"Wrote: {bounds_path}")
    if matches_path is not None:
        print(f"Wrote: {matches_path}")
    if area_plot_path is not None:
        print(f"Wrote: {area_plot_path}")
    print()
    print("Best exact F1 threshold:")
    print(best_f1[BEST_PRINT_COLUMNS].to_string())
    print()
    print("Best exact F2 threshold:")
    print(best_f2[BEST_PRINT_COLUMNS].to_string())
    print()
    print_vertical_dataframe_block(
        title="User-specified thresholds:",
        df=reported,
        key_order=SUMMARY_COLUMNS,
        record_label_column="threshold",
    )

    if bounds_summary is not None:
        print()
        print_vertical_dataframe_block(
            title="Validation-based extent bounds:",
            df=bounds_summary,
            key_order=[
                "operational_threshold",
                "mapped_area",
                "area_unit",
                "ua_precision",
                "pa_recall",
                "commission_error",
                "omission_error",
                "specificity",
                "balanced_accuracy",
                "mcc",
                "f1",
                "f2",
                "accuracy",
                "tp",
                "fp",
                "fn",
                "tn",
                "lower_bound_area",
                "upper_bound_area",
                "lower_bound_multiplier",
                "upper_bound_multiplier",
                "bound_method",
                "ua_definition",
                "pa_definition",
            ],
        )

    if matches_df is not None:
        print()
        print_vertical_dataframe_block(
            title="Area-bound threshold matches:",
            df=matches_df,
            key_order=[
                "target_name",
                "target_area",
                "direction_preference",
                "selected_threshold",
                "selected_area",
                "signed_area_difference",
                "abs_area_difference",
                "relative_area_difference",
                "threshold_distance_from_operational",
                "candidate_count_used",
                "status",
            ],
            record_label_column="target_name",
        )


if __name__ == "__main__":
    main()
