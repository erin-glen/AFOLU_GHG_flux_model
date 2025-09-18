# -*- coding: utf-8 -*-
"""Render global quick-look figures from aggregated tables.

This renderer mirrors the ``drainage_emissions_model`` directory structure:
aggregated inputs live under ``outputs/global/aggregated/<run>`` and figures
are written to ``figures/global/<run>``.  Each export includes the
``--model-run-name`` so multiple runs can coexist.

---------------------------------------------------------------------------
Quick examples
---------------------------------------------------------------------------

# 1) Default scatter of global totals (sum across all scenarios/years)
python scripts/render_global_maps.py \
  --model-run-name ogh_standard_model

# 2) Target a custom aggregated file (e.g., alternate statistic)
python scripts/render_global_maps.py \
  --model-run-name ogh_standard_model \
  --data-file /tmp/aggregated/ogh_standard_model/custom_totals.csv

# 3) Filter to a specific scenario/year before plotting
python scripts/render_global_maps.py \
  --model-run-name ogh_standard_model \
  --scenario baseline \
  --year 2030

# 4) Annotate each point and tweak styling
python scripts/render_global_maps.py \
  --model-run-name ogh_standard_model \
  --label-column region_id \
  --cmap plasma \
  --marker-size 60

# 5) Send figures to a scratch directory at higher DPI
python scripts/render_global_maps.py \
  --model-run-name ogh_standard_model \
  --output-dir /tmp/figures/ogh_standard_model \
  --dpi 300

# 6) Dry-run to verify discovery and filtering without saving
python scripts/render_global_maps.py \
  --model-run-name ogh_standard_model \
  --dry-run

Notes
-----
• Inputs: ``global_totals_<run>.csv`` is used by default unless ``--data-file``
  is provided.
• Outputs: filenames follow ``global_map_<run>_<scenario>_<year>.png`` with
  "all-scenarios" / "all-years" placeholders for unfiltered runs.
• Latitude/longitude/value column names can be configured to match the
  aggregated dataset schema produced by :mod:`scripts.aggregate_global`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render global maps using aggregated model outputs.",
    )
    parser.add_argument(
        "--model-run-name",
        required=True,
        help="Identifier for the model run; controls default paths and filenames.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional override for the project root directory.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help=(
            "Aggregated dataset to plot. Defaults to "
            "<project root>/outputs/global/aggregated/<model_run_name>/"
            "global_totals_<model_run_name>.csv."
        ),
    )
    parser.add_argument(
        "--latitude-column",
        default="latitude",
        help="Column containing latitude coordinates.",
    )
    parser.add_argument(
        "--longitude-column",
        default="longitude",
        help="Column containing longitude coordinates.",
    )
    parser.add_argument(
        "--value-column",
        default="value",
        help="Numeric column plotted as the colour scale.",
    )
    parser.add_argument(
        "--scenario-column",
        default="scenario",
        help="Column storing scenario identifiers.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional scenario filter applied before plotting.",
    )
    parser.add_argument(
        "--year-column",
        default="year",
        help="Column storing temporal identifiers.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Optional year filter applied before plotting.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Optional column whose values will be annotated next to markers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for rendered figures. Defaults to "
            "<project root>/figures/global/<model_run_name>."
        ),
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional filename for the produced figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Dots-per-inch used when saving the figure.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=30.0,
        help="Marker size passed to :func:`matplotlib.pyplot.scatter`.",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colour map used to colourise the scatter markers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process data but skip writing any files.",
    )
    return parser.parse_args(argv)


def locate_project_root(project_root: Optional[Path]) -> Path:
    if project_root is not None:
        return project_root.expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def resolve_with_base(base: Path, provided: Optional[Path], default: Path) -> Path:
    if provided is None:
        return default

    candidate = provided.expanduser()
    return candidate if candidate.is_absolute() else base / candidate


def default_data_file(project_root: Path, model_run: str) -> Path:
    return (
        project_root
        / "outputs"
        / "global"
        / "aggregated"
        / model_run
        / f"global_totals_{model_run}.csv"
    )


def default_output_dir(project_root: Path, model_run: str) -> Path:
    return project_root / "figures" / "global" / model_run


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def apply_optional_filter(frame: pd.DataFrame, column: str, value: Optional[object]) -> pd.DataFrame:
    if value is None:
        return frame
    if column not in frame.columns:
        raise KeyError(f"Column '{column}' was not present in the dataset.")
    return frame.loc[frame[column] == value]


def build_title(model_run: str, scenario: Optional[str], year: Optional[int]) -> str:
    parts = [f"Global emissions – {model_run}"]
    if scenario is not None:
        parts.append(f"scenario: {scenario}")
    if year is not None:
        parts.append(f"year: {year}")
    return " – ".join(parts)


def determine_output_path(
    output_dir: Path,
    output_name: Optional[str],
    model_run: str,
    scenario: Optional[str],
    year: Optional[int],
) -> Path:
    if output_name:
        destination = output_dir / output_name
    else:
        scenario_part = scenario if scenario is not None else "all-scenarios"
        year_part = str(year) if year is not None else "all-years"
        destination = output_dir / f"global_map_{model_run}_{scenario_part}_{year_part}.png"

    if destination.suffix == "":
        destination = destination.with_suffix(".png")
    return destination


def render_map(
    frame: pd.DataFrame,
    latitude_column: str,
    longitude_column: str,
    value_column: str,
    cmap: str,
    marker_size: float,
    label_column: Optional[str],
    title: str,
) -> plt.Figure:
    for column in (latitude_column, longitude_column, value_column):
        if column not in frame.columns:
            raise KeyError(f"Column '{column}' was not present in the dataset.")

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    scatter = ax.scatter(
        frame[longitude_column],
        frame[latitude_column],
        c=frame[value_column],
        cmap=cmap,
        s=marker_size,
        edgecolor="black",
        linewidth=0.4,
    )
    colour_bar = fig.colorbar(scatter, ax=ax)
    colour_bar.set_label(value_column)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_title(title)
    ax.grid(True, linewidth=0.2, linestyle="--", alpha=0.5)

    if label_column and label_column in frame.columns:
        for _, row in frame.iterrows():
            ax.annotate(
                str(row[label_column]),
                (row[longitude_column], row[latitude_column]),
                textcoords="offset points",
                xytext=(2, 2),
                fontsize=7,
            )

    return fig


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    project_root = locate_project_root(args.project_root)
    data_file = resolve_with_base(
        project_root,
        args.data_file,
        default_data_file(project_root, args.model_run_name),
    )
    output_dir = resolve_with_base(
        project_root,
        args.output_dir,
        default_output_dir(project_root, args.model_run_name),
    )

    if not data_file.exists():
        print(f"Error: aggregated dataset not found: {data_file}")
        return 1

    frame = pd.read_csv(data_file)

    try:
        filtered = apply_optional_filter(frame, args.scenario_column, args.scenario)
        filtered = apply_optional_filter(filtered, args.year_column, args.year)
    except KeyError as exc:
        print(f"Error: {exc}")
        return 1

    if filtered.empty:
        print("No rows left after applying filters; skipping render.")
        return 0

    title = build_title(args.model_run_name, args.scenario, args.year)

    try:
        fig = render_map(
            filtered,
            args.latitude_column,
            args.longitude_column,
            args.value_column,
            args.cmap,
            args.marker_size,
            args.label_column,
            title,
        )
    except KeyError as exc:
        print(f"Error: {exc}")
        return 1

    destination = determine_output_path(
        output_dir,
        args.output_name,
        args.model_run_name,
        args.scenario,
        args.year,
    )

    if args.dry_run:
        print(f"Dry-run enabled – figure would be written to {destination}")
        plt.close(fig)
        return 0

    ensure_directory(output_dir)
    fig.savefig(destination, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote global map to {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())