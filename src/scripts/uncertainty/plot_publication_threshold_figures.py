#!/usr/bin/env python3
"""Regenerate publication-ready per-biome threshold figures from existing curve CSVs.

This is a *presentation-only* script. It does not recompute any metric, threshold,
area, or bound; it re-plots the CSVs already produced by
``fscore_threshold_curves_bounds.py`` / ``build_probability_area_curve.py`` with
publication styling and two deliberate, documented adjustments that make the
shapes legible without altering the underlying numbers:

1. F-score panels are x-limited to the operationally relevant range [0, 0.6].
   The high-threshold tail is sparse/noisy and contains no selected threshold.
2. Area panels use a log y-axis and start the x-axis at 0.03, so the curve is
   readable across orders of magnitude and the lowest-probability "quantization
   floor" (see note below) does not flatten the operational region.

NOTE ON CURVE SHAPE (for the paper methods/caption)
---------------------------------------------------
The OpenGeoHub ensemble organic-soil probability surface
(``organic.soils_ensemble.organic_p_30m_..._v20260513.tif``) is encoded on a
discrete integer 0-100 grid with an *effective* resolution of ~2%: the value 1
is never emitted and even values predominate (even:odd count ratio ~3:1 in the
raw source). Our preprocessing warps this surface to the Hansen grid with
nearest-neighbour resampling (``hansenize_coiled.warp_to_hansen_coiled``,
GDAL default ``resampleAlg``), which preserves source DNs exactly. Consequently
the threshold-response curves inherit faint stair-stepping, and the lowest class
holds a large pile of background area. Operational thresholds are selected in the
smooth, well-populated central range and are unaffected. No smoothing is applied
to the curves here; the steps are real and disclosed rather than hidden.

Inputs (under --results-dir, default = the 20260513 threshold_curves_biome_f1 dir):
- <biome>/threshold_metrics.csv            (threshold, precision, recall, f1, f2, ...)
- area_vs_threshold_<date>_biome_<biome>.csv (threshold, area_ha)
- peat_thresholds_mixed_*.csv               (selected threshold, mapped area, bounds)
- biome_thresholds_summary.csv              (best_f1_threshold, best_f2_threshold)

Outputs (under --out-dir, default = <results-dir>/publication_figures):
- pub_fscore_<biome>.{png,pdf}, pub_area_<biome>.{png,pdf}   (6 single panels)
- pub_threshold_panels_combined.{png,pdf}                    (3x2 grid)

Example:
python -m src.scripts.uncertainty.plot_publication_threshold_figures \
  --results-dir "C:/tmp/afolu/uncertainty/ogh_probability/20260513/threshold_curves_biome_f1" \
  --mixed-thresholds-csv "C:/tmp/afolu/uncertainty/ogh_probability/20260513/peat_thresholds_mixed_boreal_f1_temperate_f1_5_tropical_f2.csv" \
  --date 20260513
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Biomes in display order, with the metric basis used to select each one's
# operational threshold in the mixed deliverable.
BIOMES = [
    ("boreal", "Boreal", "F1"),
    ("temperate", "Temperate", "F1.5"),
    ("tropical", "Tropical", "F2"),
]

HA_PER_MHA = 1e6
FSCORE_XLIM = (0.0, 0.6)
AREA_XMIN = 0.03  # drop the <0.03 quantization floor (value-2 pile) from the area view

# Consistent palette
C_F1 = "#1f77b4"
C_F2 = "#ff7f0e"
C_SEL = "#2ca02c"      # selected/operational threshold
C_BESTF1 = "#1f77b4"
C_BESTF2 = "#ff7f0e"
C_AREA = "#1f77b4"
C_BOUND = "#7f7f7f"


def _styling() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_mixed(path: Path) -> dict:
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        b = str(r["biome"]).strip().lower()
        if b in {"boreal", "temperate", "tropical"}:
            out[b] = dict(
                operational=float(r["operational_threshold"]),
                mapped=float(r["mapped_area_Mha"]),
                low=float(r["lower_bound_area_Mha"]),
                high=float(r["upper_bound_area_Mha"]),
            )
    return out


def load_best(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        b = str(r["biome"]).strip().lower()
        out[b] = dict(
            best_f1=float(r["best_f1_threshold"]),
            best_f2=float(r["best_f2_threshold"]),
        )
    return out


def plot_fscore(ax, metrics: pd.DataFrame, sel: dict, best: dict, basis: str, title: str) -> None:
    m = metrics.sort_values("threshold")
    ax.plot(m["threshold"], m["f1"], "-", color=C_F1, lw=1.6, marker="o", ms=2.5, label="F1")
    ax.plot(m["threshold"], m["f2"], "-", color=C_F2, lw=1.6, marker="s", ms=2.5, label="F2")

    if best:
        ax.axvline(best["best_f1"], color=C_BESTF1, ls=":", lw=1.0, alpha=0.8,
                   label=f"best F1 = {best['best_f1']:.2f}")
        ax.axvline(best["best_f2"], color=C_BESTF2, ls="--", lw=1.0, alpha=0.8,
                   label=f"best F2 = {best['best_f2']:.2f}")
    ax.axvline(sel["operational"], color=C_SEL, ls="-", lw=1.8,
               label=f"selected = {sel['operational']:.2f} ({basis})")

    ax.set_xlim(*FSCORE_XLIM)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"{title} — threshold response")
    ax.legend(loc="upper right", framealpha=0.9)


def plot_area(ax, area: pd.DataFrame, sel: dict, title: str) -> None:
    a = area.sort_values("threshold").copy()
    a["area_Mha"] = a["area_ha"] / HA_PER_MHA
    a = a[a["threshold"] >= AREA_XMIN]

    ax.plot(a["threshold"], a["area_Mha"], "-", color=C_AREA, lw=1.8, label="Area vs threshold")
    ax.set_yscale("log")

    ax.axhspan(sel["low"], sel["high"], color=C_SEL, alpha=0.12,
               label=f"validation range [{sel['low']:.1f}, {sel['high']:.1f}] Mha")
    ax.axhline(sel["mapped"], color="#d62728", ls="--", lw=1.4,
               label=f"mapped = {sel['mapped']:.1f} Mha")
    ax.axhline(sel["low"], color=C_BOUND, ls=":", lw=1.0)
    ax.axhline(sel["high"], color=C_BOUND, ls=":", lw=1.0)
    ax.axvline(sel["operational"], color=C_SEL, ls="-", lw=1.8,
               label=f"selected threshold = {sel['operational']:.2f}")

    ax.set_xlim(AREA_XMIN, 1.0)
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Mapped area (Mha, log scale)")
    ax.set_title(f"{title} — area vs threshold")
    ax.legend(loc="upper right", framealpha=0.9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="threshold_curves_biome_f1 directory containing per-biome subdirs and area CSVs.")
    ap.add_argument("--mixed-thresholds-csv", type=Path, required=True,
                    help="peat_thresholds_mixed_*.csv with selected thresholds, mapped area, and bounds.")
    ap.add_argument("--date", default="20260513", help="Probability date tag used in the area CSV filenames.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory. Default: <results-dir>/publication_figures")
    args = ap.parse_args()

    results = args.results_dir.resolve()
    out_dir = (args.out_dir or (results / "publication_figures")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _styling()

    mixed = load_mixed(args.mixed_thresholds_csv)
    best = load_best(results / "biome_thresholds_summary.csv")

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 13.5), constrained_layout=True)

    for row, (key, label, basis) in enumerate(BIOMES):
        metrics = pd.read_csv(results / key / "threshold_metrics.csv")
        area = pd.read_csv(results / f"area_vs_threshold_{args.date}_biome_{key}.csv")
        sel = mixed[key]
        b = best.get(key, {})

        # combined grid
        plot_fscore(axes[row, 0], metrics, sel, b, basis, label)
        plot_area(axes[row, 1], area, sel, label)

        # single panels
        f1, a1 = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
        plot_fscore(a1, metrics, sel, b, basis, label)
        for ext in ("png", "pdf"):
            f1.savefig(out_dir / f"pub_fscore_{key}.{ext}")
        plt.close(f1)

        f2, a2 = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
        plot_area(a2, area, sel, label)
        for ext in ("png", "pdf"):
            f2.savefig(out_dir / f"pub_area_{key}.{ext}")
        plt.close(f2)

    fig.suptitle(
        "Per-biome threshold-response and area-versus-threshold curves "
        "(OpenGeoHub organic-soil probability, 20260513)",
        fontsize=12,
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"pub_threshold_panels_combined.{ext}")
    plt.close(fig)

    print(f"Wrote publication figures to {out_dir}")
    for p in sorted(out_dir.glob("pub_*")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
