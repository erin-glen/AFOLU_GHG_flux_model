"""Render experimental alternatives for publication Figure 10a.

This module is intentionally separate from the canonical publication renderer.
It reads the frozen full-disaggregation zonal-statistics master and produces two
review-only alternatives:

1. Emissions by assigned emission-factor class, with drained area and mean
   emission rate shown alongside the climate-stacked bars.
2. A matrix showing how the model's first-matching drainage pathways feed the
   assigned emission-factor classes.

The script does not query S3, mutate publication inputs, or update a submission
package.  All source tables and QA metadata are written beside the test images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont


MM_PER_INCH = 25.4
PUBLICATION_WIDTH_MM = 180.0
CLIMATE_ORDER = ["boreal", "temperate", "tropical"]
CLIMATE_LABELS = {
    "boreal": "Boreal",
    "temperate": "Temperate",
    "tropical": "Tropical",
}
CLIMATE_COLORS = {
    "boreal": "#7570B3",
    "temperate": "#D95F02",
    "tropical": "#1B9E77",
}
PATHWAY_ORDER = [
    "peat_drained_primary_infra",
    "peat_drained_secondary_infra",
    "peat_drained_extraction",
    "peat_drained_cropland_settlement",
    "peat_drained_plantation",
]
PATHWAY_LABELS = {
    "peat_drained_primary_infra": "Primary canal evidence",
    "peat_drained_secondary_infra": "Secondary road evidence",
    "peat_drained_extraction": "Extraction",
    "peat_drained_cropland_settlement": "Cropland or settlement",
    "peat_drained_plantation": "Plantation",
}
DISPLAY_LAND_USE = {
    "Otherland": "Other land",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master",
        type=Path,
        required=True,
        help="Frozen corrected full-disaggregation master parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for review-only images, source tables, and QA metadata.",
    )
    parser.add_argument("--interval-end", type=int, default=2024)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_source(master: Path, interval_end: int) -> pd.DataFrame:
    if not master.is_file():
        raise FileNotFoundError(master)
    query = """
        SELECT
            lower(climate_domain) AS climate_domain,
            land_use,
            drainage_class,
            sum(area__ha) / 1e6 AS area_mha,
            sum(drained_total_Mg_CO2e) / 1e9 AS emissions_gt
        FROM read_parquet(?)
        WHERE interval_end = ?
          AND drainage_class LIKE 'peat_drained%'
        GROUP BY ALL
        ORDER BY land_use, drainage_class, climate_domain
    """
    with duckdb.connect() as con:
        frame = con.execute(query, [str(master), interval_end]).fetchdf()
    if frame.empty:
        raise ValueError(f"No drained rows found for interval_end={interval_end}")
    frame["mean_rate_t_ha_yr"] = np.divide(
        frame["emissions_gt"] * 1e3,
        frame["area_mha"],
        out=np.zeros(len(frame), dtype=float),
        where=frame["area_mha"].to_numpy() > 0,
    )
    return frame


def build_class_tables(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_climate = (
        source.groupby(["land_use", "climate_domain"], as_index=False)[
            ["area_mha", "emissions_gt"]
        ]
        .sum()
        .sort_values(["land_use", "climate_domain"])
    )
    totals = (
        source.groupby("land_use", as_index=False)[["area_mha", "emissions_gt"]]
        .sum()
        .sort_values("emissions_gt", ascending=False)
        .reset_index(drop=True)
    )
    totals["mean_rate_t_ha_yr"] = totals["emissions_gt"] * 1e3 / totals["area_mha"]
    totals["global_emissions_share_pct"] = (
        100 * totals["emissions_gt"] / totals["emissions_gt"].sum()
    )
    totals["display_land_use"] = totals["land_use"].replace(DISPLAY_LAND_USE)
    return class_climate, totals


def render_alternative_1(
    class_climate: pd.DataFrame,
    totals: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    """Render climate contributions plus area and mean-rate context."""
    configure_style()
    width_in = PUBLICATION_WIDTH_MM / MM_PER_INCH
    height_in = 4.9
    fig = plt.figure(figsize=(width_in, height_in), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[5.4, 1.05, 1.35],
        left=0.17,
        right=0.985,
        top=0.79,
        bottom=0.13,
        wspace=0.04,
    )
    ax = fig.add_subplot(grid[0, 0])
    area_ax = fig.add_subplot(grid[0, 1], sharey=ax)
    rate_ax = fig.add_subplot(grid[0, 2], sharey=ax)

    order = totals["land_use"].tolist()
    y = np.arange(len(order))
    left = np.zeros(len(order), dtype=float)
    for climate in CLIMATE_ORDER:
        lookup = (
            class_climate[class_climate["climate_domain"] == climate]
            .set_index("land_use")["emissions_gt"]
            .reindex(order, fill_value=0.0)
            .to_numpy()
        )
        ax.barh(
            y,
            lookup,
            left=left,
            color=CLIMATE_COLORS[climate],
            edgecolor="white",
            linewidth=0.45,
            height=0.70,
            label=CLIMATE_LABELS[climate],
        )
        left += lookup

    ax.set_yticks(y, totals["display_land_use"])
    ax.invert_yaxis()
    ax.set_xlabel("Annual drainage emissions (Gt CO$_2$e yr$^{-1}$)")
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=3,
        frameon=False,
        borderaxespad=0,
        handlelength=1.3,
        columnspacing=1.2,
        title=None,
    )

    for i, total in enumerate(totals["emissions_gt"]):
        label = f"{total:.2f}" if total >= 0.01 else "<0.01"
        ax.text(total + 0.009, i, label, va="center", ha="left", fontsize=8)
    ax.set_xlim(0, max(totals["emissions_gt"]) * 1.20)

    for table_ax, heading, values, fmt in [
        (area_ax, "Drained area\n(Mha)", totals["area_mha"], "{:.1f}"),
        (
            rate_ax,
            "Mean rate\n(t CO$_2$e ha$^{-1}$ yr$^{-1}$)",
            totals["mean_rate_t_ha_yr"],
            "{:.1f}",
        ),
    ]:
        table_ax.set_xlim(0, 1)
        table_ax.set_ylim(ax.get_ylim())
        table_ax.set_xticks([])
        table_ax.tick_params(axis="y", left=False, labelleft=False)
        table_ax.spines[:].set_visible(False)
        table_ax.axvline(0.02, color="#D0D0D0", linewidth=0.7)
        table_ax.text(
            0.52,
            1.015,
            heading,
            transform=table_ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
        for i, value in enumerate(values):
            table_ax.text(0.52, i, fmt.format(value), ha="center", va="center", fontsize=8.5)

    fig.suptitle(
        "Drainage emissions, area, and mean rate by class",
        x=0.17,
        y=0.975,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.17,
        0.925,
        "Class totals for 2021–2024 reflect drained area × the assigned Tier 1 rate.",
        ha="left",
        fontsize=8.5,
        color="#444444",
    )
    fig.text(
        0.17,
        0.025,
        "The negligible fallback/other-domain contribution is retained in the table totals but omitted from the color stack.",
        ha="left",
        fontsize=7.2,
        color="#555555",
    )
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def build_pathway_matrix(source: pd.DataFrame, class_order: list[str]) -> pd.DataFrame:
    matrix = source.pivot_table(
        index="drainage_class",
        columns="land_use",
        values="emissions_gt",
        aggfunc="sum",
        fill_value=0.0,
    )
    matrix = matrix.reindex(index=PATHWAY_ORDER, columns=class_order, fill_value=0.0)
    matrix.index = [PATHWAY_LABELS[value] for value in matrix.index]
    matrix.columns = [DISPLAY_LAND_USE.get(value, value) for value in matrix.columns]
    return matrix


def render_alternative_2(matrix: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """Render first-matching pathway by emission-factor-class matrix."""
    configure_style()
    width_in = PUBLICATION_WIDTH_MM / MM_PER_INCH
    height_in = 4.75
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0.27, right=0.93, top=0.78, bottom=0.28)

    cmap = LinearSegmentedColormap.from_list(
        "white_to_green", ["#FFFFFF", "#D7EFE7", "#72C1AA", "#1B7F68"]
    )
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=cmap, aspect="auto", vmin=0, vmax=values.max())

    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns)
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    plt.setp(ax.get_xticklabels(), rotation=32, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="both", length=0)
    ax.set_ylabel("First-matching drainage pathway", labelpad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    threshold = values.max() * 0.48
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if value < 0.0005:
                label = ""
            elif value < 0.01:
                label = "<0.01"
            else:
                label = f"{value:.2f}"
            if label:
                ax.text(
                    col,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.7,
                    color="white" if value > threshold else "#202020",
                )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Annual drainage emissions (Gt CO$_2$e yr$^{-1}$)", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Drainage pathways and assigned emission-factor classes",
        x=0.27,
        y=0.965,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.text(
        0.27,
        0.875,
        "Cell values are Gt CO$_2$e yr$^{-1}$ for 2021–2024.",
        ha="left",
        fontsize=8.5,
        color="#444444",
    )
    fig.text(
        0.27,
        0.035,
        "Pathways are priority assignments in the model, not exclusive causal attribution.",
        ha="left",
        fontsize=7.5,
        color="#555555",
    )
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def make_contact_sheet(image_paths: list[Path], output_path: Path) -> None:
    opened = [Image.open(path).convert("RGB") for path in image_paths]
    width = max(image.width for image in opened)
    label_height = 48
    gap = 28
    total_height = sum(image.height + label_height for image in opened) + gap * (len(opened) + 1)
    sheet = Image.new("RGB", (width + 2 * gap, total_height), "#E8E8E8")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    y = gap
    for index, (path, image) in enumerate(zip(image_paths, opened), start=1):
        draw.text((gap, y + 12), f"Alternative {index}: {path.name}", fill="black", font=font)
        y += label_height
        sheet.paste(image, (gap, y))
        y += image.height + gap
    sheet.save(output_path, dpi=(150, 150))
    for image in opened:
        image.close()


def image_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        alpha_opaque = True
        if "A" in image.getbands():
            extrema = image.getchannel("A").getextrema()
            alpha_opaque = extrema == (255, 255)
        return {
            "path": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "width_px": image.width,
            "height_px": image.height,
            "mode": image.mode,
            "dpi": list(image.info.get("dpi", ())),
            "alpha_fully_opaque": alpha_opaque,
        }


def main() -> int:
    args = parse_args()
    master = args.master.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = load_source(master, args.interval_end)
    class_climate, totals = build_class_tables(source)
    matrix = build_pathway_matrix(source, totals["land_use"].tolist())

    source_path = output_dir / "figure10a_source_long.csv"
    class_path = output_dir / "figure10a_class_summary.csv"
    matrix_path = output_dir / "figure10a_pathway_matrix.csv"
    source.to_csv(source_path, index=False)
    totals.drop(columns="display_land_use").to_csv(class_path, index=False)
    matrix.to_csv(matrix_path)

    alternative_1 = output_dir / "Figure_10a_alternative_1_contribution_context.png"
    alternative_2 = output_dir / "Figure_10a_alternative_2_pathway_matrix.png"
    contact_sheet = output_dir / "Figure_10a_alternatives_contact_sheet.png"
    render_alternative_1(class_climate, totals, alternative_1, args.dpi)
    render_alternative_2(matrix, alternative_2, args.dpi)
    make_contact_sheet([alternative_1, alternative_2], contact_sheet)

    total_area = float(source["area_mha"].sum())
    total_emissions = float(source["emissions_gt"].sum())
    class_area = float(totals["area_mha"].sum())
    class_emissions = float(totals["emissions_gt"].sum())
    matrix_emissions = float(matrix.to_numpy().sum())
    other = source[source["climate_domain"] == "other_domain"]
    other_area = float(other["area_mha"].sum())
    other_emissions = float(other["emissions_gt"].sum())

    checks = {
        "class_area_conserves": abs(class_area - total_area) <= 1e-10,
        "class_emissions_conserve": abs(class_emissions - total_emissions) <= 1e-12,
        "matrix_emissions_conserve": abs(matrix_emissions - total_emissions) <= 1e-12,
        "expected_pathways_present": set(source["drainage_class"]) == set(PATHWAY_ORDER),
        "no_negative_area": bool((source["area_mha"] >= 0).all()),
        "no_negative_emissions": bool((source["emissions_gt"] >= 0).all()),
    }
    qa = {
        "status": "PASS" if all(checks.values()) else "HOLD",
        "purpose": "Review-only Figure 10a alternatives; no submission-package mutation",
        "master": {
            "path": str(master),
            "sha256": sha256_file(master),
            "interval_end": args.interval_end,
        },
        "controls": {
            "drained_area_mha": total_area,
            "drained_emissions_gt_co2e_yr": total_emissions,
            "mean_rate_t_co2e_ha_yr": total_emissions * 1e3 / total_area,
            "fallback_other_domain_area_mha": other_area,
            "fallback_other_domain_emissions_gt_co2e_yr": other_emissions,
        },
        "semantics": {
            "alternative_1": "Assigned land-use emission-factor classes with climate contributions, area, and mean rate.",
            "alternative_2": "First-matching operational drainage pathways by assigned emission-factor class; not causal attribution.",
        },
        "checks": checks,
        "source_files": [
            {"path": path.name, "sha256": sha256_file(path), "rows": len(frame)}
            for path, frame in [
                (source_path, source),
                (class_path, totals),
                (matrix_path, matrix),
            ]
        ],
        "images": [image_metadata(path) for path in [alternative_1, alternative_2, contact_sheet]],
    }
    qa_path = output_dir / "figure10a_alternatives_qa.json"
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps({"status": qa["status"], "output_dir": str(output_dir)}, indent=2))
    return 0 if qa["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
