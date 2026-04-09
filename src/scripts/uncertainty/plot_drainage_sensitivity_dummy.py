#!/usr/bin/env python3
"""
Create a dummy sensitivity figure for drainage-emissions scenario combinations.

This utility is intentionally synthetic: it does NOT ingest model outputs.
It creates a simple 2x2 scenario matrix representing how total drainage
emissions might vary when combining:
    - low vs high peat threshold
    - low vs high emission-factor (EF) bounds

Outputs
-------
1. dummy_drainage_sensitivity_scenarios.csv
2. dummy_drainage_sensitivity_figure.png

Example
-------
python src/scripts/uncertainty/plot_drainage_sensitivity_dummy.py \
  --output-dir /mnt/c/tmp/uncertainty \
  --base-emissions 120.5 \
  --unit "MtCO2e/yr"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a synthetic 2x2 drainage-emissions sensitivity figure for "
            "low/high threshold combined with low/high EF bounds."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where dummy CSV and figure outputs are written.",
    )
    parser.add_argument(
        "--base-emissions",
        type=float,
        default=100.0,
        help="Baseline emissions level used to build dummy scenarios. Default: 100.0",
    )
    parser.add_argument(
        "--unit",
        default="MtCO2e/yr",
        help="Unit label for scenario values. Default: MtCO2e/yr",
    )
    parser.add_argument(
        "--high-threshold-reduction-frac",
        type=float,
        default=0.18,
        help=(
            "Fractional emissions reduction applied when using the high threshold "
            "(synthetic assumption). Default: 0.18"
        ),
    )
    parser.add_argument(
        "--high-ef-increase-frac",
        type=float,
        default=0.22,
        help=(
            "Fractional emissions increase applied when using the high EF bound "
            "(synthetic assumption). Default: 0.22"
        ),
    )
    return parser.parse_args()


def build_dummy_scenarios(
    base_emissions: float,
    high_threshold_reduction_frac: float,
    high_ef_increase_frac: float,
) -> pd.DataFrame:
    """Create a 2x2 synthetic scenario table."""
    threshold_multiplier = {
        "low_threshold": 1.0,
        "high_threshold": 1.0 - high_threshold_reduction_frac,
    }
    ef_multiplier = {
        "low_ef": 1.0,
        "high_ef": 1.0 + high_ef_increase_frac,
    }

    rows = []
    for threshold_case, threshold_mult in threshold_multiplier.items():
        for ef_case, ef_mult in ef_multiplier.items():
            scenario_value = base_emissions * threshold_mult * ef_mult
            rows.append(
                {
                    "threshold_case": threshold_case,
                    "ef_case": ef_case,
                    "threshold_multiplier": threshold_mult,
                    "ef_multiplier": ef_mult,
                    "dummy_emissions": scenario_value,
                }
            )
    return pd.DataFrame(rows)


def plot_dummy_scenarios(
    scenarios: pd.DataFrame,
    unit: str,
    output_path: Path,
) -> None:
    """Render a heatmap-style 2x2 dummy sensitivity figure."""
    pivot = scenarios.pivot(index="threshold_case", columns="ef_case", values="dummy_emissions")
    pivot = pivot.reindex(index=["low_threshold", "high_threshold"], columns=["low_ef", "high_ef"])
    z = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    image = ax.imshow(z, cmap="YlOrRd", aspect="auto")

    for row_idx in range(z.shape[0]):
        for col_idx in range(z.shape[1]):
            val = z[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{val:.2f}\n{unit}",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
                fontweight="bold",
            )

    ax.set_xticks(np.arange(2), labels=["Low EF bound", "High EF bound"])
    ax.set_yticks(np.arange(2), labels=["Low threshold", "High threshold"])
    ax.set_xlabel("Emission-factor scenario")
    ax.set_ylabel("Threshold scenario")
    ax.set_title("Dummy drainage-emissions sensitivity matrix\n(threshold x EF bound)")

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Dummy emissions ({unit})")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_dummy_scenarios(
        base_emissions=float(args.base_emissions),
        high_threshold_reduction_frac=float(args.high_threshold_reduction_frac),
        high_ef_increase_frac=float(args.high_ef_increase_frac),
    )

    csv_path = output_dir / "dummy_drainage_sensitivity_scenarios.csv"
    fig_path = output_dir / "dummy_drainage_sensitivity_figure.png"

    scenarios.to_csv(csv_path, index=False)
    plot_dummy_scenarios(scenarios=scenarios, unit=args.unit, output_path=fig_path)

    print("Finished dummy drainage sensitivity outputs.")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {fig_path}")


if __name__ == "__main__":
    main()
