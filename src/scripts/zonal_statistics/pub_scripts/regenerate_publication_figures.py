# -*- coding: utf-8 -*-
"""Regenerate corrected publication figures from frozen local CSVs.

This orchestrator is deliberately offline.  The first supported scope is the
required seven-asset visual pilot; the remaining assets are not rendered until
that pilot receives recorded human approval.  No S3, Coiled, map, LaTeX, or PDF
code path is imported or invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.scripts.zonal_statistics.pub_scripts import pub_common as pc
from src.scripts.zonal_statistics.pub_scripts import publication_figure_export as pfe
from src.scripts.zonal_statistics.pub_scripts import publication_figure_provenance as pfp
from src.scripts.zonal_statistics.pub_scripts import publication_figure_qa as pfq
from src.scripts.zonal_statistics.pub_scripts import publication_figure_renderers as pfr
from src.scripts.zonal_statistics.pub_scripts import publication_nghgi_renderer as pngr


STATUS = "PENDING_VISUAL_APPROVAL"
EXPECTED_GLOBAL_AREA_MHA = 1144.3268381581924
EXPECTED_BASELINE_GT_CO2E_YR = 2.689369853477171
EXPECTED_NGHGI_HASHES = {
    "model_vs_nghgi.csv": "183142e69c1fd419c7eed2855cc7493e84e02fd86d2aff98e3446d6eefc5d624",
    "model_vs_nghgi_t3d.csv": "2edbbd41a8946a75b3045ca28b6af038aa245c805b986bb302f88e0833be1192",
    "model_vs_nghgi_t3d_cropland_grassland.csv": "2bd2809b0c5fc15a4d58fa1f840546ec49052a590abbe0a53ff9c97e18f06509",
    "nghgi_publication_manifest.json": "176cfd963cc6ff21f9247d10c82134952546b55ca0b30c7efb1d08323206c652",
    "nghgi_publication_ratios.csv": "0046e54cf4526ae13b30445f4e8ecbe510c42e8fc804d8de532e4dbe76584c5a",
}


@dataclass(frozen=True)
class PilotAsset:
    asset_id: str
    upload_filename: str
    oracle_filename: str
    source_relative_path: str | None
    builder_key: str
    target_width_mm: float
    change_description: str


PILOT_ASSETS = (
    PilotAsset(
        "Figure_01a",
        "Figure_01a.tif",
        "global_total_emissions_column.png",
        "data/core/global_total_emissions_long.csv",
        "Figure_01a",
        85.0,
        "Corrected global totals; approved palette and stacked-column layout retained.",
    ),
    PilotAsset(
        "Figure_02b",
        "Figure_02b.tif",
        "intensity_by_climate_component_column.png",
        "data/core/intensity_by_climate_component.csv",
        "Figure_02b",
        85.0,
        "Corrected process intensities with the manuscript's stacked layout retained.",
    ),
    PilotAsset(
        "Figure_09d",
        "Figure_09d.tif",
        "top_10_country_burned_avg_emissions_bar.png",
        "data/core/top_10_country_burned_avg_emissions.csv",
        "Figure_09d",
        85.0,
        "Corrected ranking values with three-decimal annotations.",
    ),
    PilotAsset(
        "Figure_10b",
        "Figure_10b.tif",
        "burned_landuse_climate_bar.png",
        "data/core/burned_landuse_climate_long.csv",
        "Figure_10b",
        85.0,
        "Corrected burned land-use/climate values; approved layout retained.",
    ),
    PilotAsset(
        "Figure_11a",
        "Figure_11a.tif",
        "inventory_source_peat_area_stack.png",
        "data/comparisons/inventory_source_peat_area_stack.csv",
        "Figure_11a",
        85.0,
        "Corrected mapped-area comparison; canonical comparison renderer retained.",
    ),
    PilotAsset(
        "Figure_13",
        "Figure_13.tif",
        "sensitivity_tornado.png",
        "data/figure13/sensitivity_tornado_values.csv",
        "Figure_13",
        180.0,
        "Corrected 2.689369853477171 Gt CO2e/yr baseline and sensitivity bounds.",
    ),
    PilotAsset(
        "Figure_15",
        "Figure_15.tif",
        "nghgi_comparison_3panel.png",
        None,
        "Figure_15",
        180.0,
        "Corrected fixed country order and model-to-NGHGI ratios.",
    ),
)

FINAL_AXES_POSITIONS = {
    "Figure_01a": (0.19, 0.28, 0.79, 0.50),
    "Figure_02b": (0.19, 0.24, 0.79, 0.55),
    "Figure_09d": (0.11, 0.22, 0.865, 0.77),
    "Figure_10b": (0.505, 0.36, 0.395, 0.43),  # shallow 85-mm panel
    "Figure_11a": (0.24, 0.38, 0.61, 0.40),
}
FINAL_LEGEND_COLUMNS = {
    "Figure_01a": 2,
    "Figure_02b": 2,
    "Figure_10b": 3,
    "Figure_11a": 2,
}


def _prepare_final_layout(
    figure,
    asset: PilotAsset,
    *,
    dpi: int,
    aspect_ratio: float,
) -> tuple[int, int]:
    target_width_px = 1004 if asset.target_width_mm == 85.0 else 2126
    target_height_px = int(round(target_width_px / aspect_ratio))
    figure.set_size_inches(
        target_width_px / dpi,
        target_height_px / dpi,
        forward=False,
    )
    if asset.target_width_mm == 85.0:
        # Enforce the eight-point floor without flattening the approved hierarchy.
        for text_artist in figure.findobj(match=plt.Text):
            if text_artist.get_visible() and text_artist.get_text().strip():
                if float(text_artist.get_fontsize()) < 8.0:
                    text_artist.set_fontsize(8.0)
        axis = figure.axes[0]
        if asset.asset_id == "Figure_10b":
            # Compact-panel exception: keep all F10 text at the approved floor.
            for text_artist in figure.findobj(match=plt.Text):
                if text_artist.get_visible() and text_artist.get_text().strip():
                    text_artist.set_fontsize(8.0)
        position = FINAL_AXES_POSITIONS.get(asset.asset_id)
        if position is not None:
            axis.set_position(position)
        if asset.asset_id == "Figure_01a":
            axis.set_ylabel("Annual Emissions\n(Gt CO$_2$e/year)", labelpad=2.0)
            axis.set_xlabel(axis.get_xlabel(), labelpad=0.0)
            tick_positions = axis.get_xticks()
            tick_labels = [
                label.get_text().replace("-", "-\n", 1) for label in axis.get_xticklabels()
            ]
            axis.set_xticks(tick_positions, labels=tick_labels, fontsize=8.0)
        elif asset.asset_id == "Figure_02b":
            axis.set_ylabel("Emissions Intensity\n(t CO$_2$e/ha/year)")
            axis.set_xlabel(axis.get_xlabel(), labelpad=1.0)
        elif asset.asset_id == "Figure_09d":
            axis.set_xlabel(
                "Average Annual Emissions\n(Gt CO$_2$e/year)",
                labelpad=3.0,
            )
            current_max = max(float(patch.get_width()) for patch in axis.patches)
            axis.set_xlim(0.0, current_max * 1.16)
            axis.spines["left"].set_visible(False)
            axis.tick_params(axis="y", length=0)
            for patch in axis.patches:
                patch.set_zorder(3)
        elif asset.asset_id == "Figure_10b":
            tick_positions = axis.get_yticks()
            tick_labels = [
                label.get_text().replace("\n", " ")
                for label in axis.get_yticklabels()
            ]
            axis.set_yticks(tick_positions, labels=tick_labels, fontsize=8.0)
            axis.set_xlabel(
                "Annual Emissions, 2021-2024 (Gt CO$_2$e/year)",
                fontsize=8.5,
                labelpad=0.0,
            )
            axis.xaxis.set_label_coords(-0.02, -0.45)
            axis.set_ylabel(
                "Land Use",
                fontsize=8.5,
                rotation=0,
                ha="left",
                va="center",
                labelpad=0.0,
            )
            axis.yaxis.set_label_coords(-1.20, 1.11)
        legend = axis.get_legend()
        if legend is not None:
            legend_fontsize = (
                8.0
                if asset.asset_id == "Figure_10b"
                else max(
                    [
                        8.0,
                        *[
                            float(text.get_fontsize())
                            for text in legend.get_texts()
                        ],
                    ]
                )
            )
            handles, labels = axis.get_legend_handles_labels()
            legend.remove()
            figure.legend(
                handles,
                labels,
                ncol=FINAL_LEGEND_COLUMNS[asset.asset_id],
                loc=(
                    "upper left"
                    if asset.asset_id == "Figure_10b"
                    else "upper center"
                ),
                bbox_to_anchor=(
                    (0.25, 1.02)
                    if asset.asset_id == "Figure_10b"
                    else (0.5, 1.02)
                ),
                frameon=False,
                handlelength=1.4,
                columnspacing=0.9,
                fontsize=legend_fontsize,
            )
    elif asset.asset_id == "Figure_13":
        axis = figure.axes[0]
        left, bottom, width, height = axis.get_position().bounds
        axis.set_position((left + 0.04, bottom, width - 0.04, height))
    elif asset.asset_id == "Figure_15":
        # The approved PNG used a tight outer crop. Shift the canonical axes
        # left by two percentage points on the fixed final canvas to preserve
        # that normalized geometry without raster cropping or upscaling.
        figure.subplots_adjust(left=0.065, right=0.985, top=0.79, bottom=0.15, wspace=0.07)
    return target_width_px, target_height_px


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_sha256s(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        digest, relative = line.split(maxsplit=1)
        records[relative.replace("\\", "/")] = digest.lower()
    return records


def _verify_file(path: Path, expected_hash: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = _sha256(path)
    if observed != expected_hash.lower():
        raise ValueError(
            f"{label} hash mismatch: expected {expected_hash}, observed {observed}"
        )
    return observed


def _git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _assert_numerical_controls(input_root: Path) -> dict[str, object]:
    totals = pd.read_csv(input_root / "data/core/global_total_emissions_long.csv")
    latest = totals.loc[totals["Inventory Period"] == "2021-2024", "GtCO2e"].sum()
    if abs(float(latest) - EXPECTED_BASELINE_GT_CO2E_YR) > 1e-12:
        raise ValueError(f"Corrected 2021-2024 baseline mismatch: {latest}")

    area = pd.read_csv(
        input_root / "data/comparisons/inventory_source_peat_area_stack.csv"
    )
    ogh = area.loc[area["Run"] == "OGH 500 m"]
    if len(ogh) != 1:
        raise ValueError("Figure 11a must contain exactly one OGH 500 m row")
    observed_area = float(ogh.iloc[0]["Drained"] + ogh.iloc[0]["Undrained"])
    if abs(observed_area - EXPECTED_GLOBAL_AREA_MHA) > 1e-6:
        raise ValueError(f"Corrected global mapped area mismatch: {observed_area}")

    controls = pfr.load_figure_13_controls(
        input_root / "data/figure13/sensitivity_tornado_values.csv"
    )
    return {
        "latest_total_gt_co2e_yr": float(latest),
        "global_mapped_area_mha": observed_area,
        "figure_13_baseline_gt_co2e_yr": controls.baseline,
        "figure_13_combined_low_gt_co2e_yr": controls.combined_low,
        "figure_13_combined_high_gt_co2e_yr": controls.combined_high,
    }


def _verify_frozen_inputs(
    input_root: Path,
    nghgi_input_root: Path,
) -> dict[str, object]:
    checksum_path = input_root / "SHA256SUMS.txt"
    checksums = _parse_sha256s(checksum_path)
    required_relatives: set[str] = {"SHA256SUMS.txt"}
    for asset in PILOT_ASSETS:
        required_relatives.add(f"figures/{asset.oracle_filename}")
        if asset.source_relative_path:
            required_relatives.add(asset.source_relative_path)

    verified: dict[str, str] = {}
    for relative in sorted(required_relatives - {"SHA256SUMS.txt"}):
        if relative not in checksums:
            raise ValueError(f"Approved package checksum list omits {relative}")
        verified[relative] = _verify_file(
            input_root / Path(relative), checksums[relative], relative
        )

    nghgi_verified = {
        filename: _verify_file(
            nghgi_input_root / filename,
            expected,
            f"frozen NGHGI {filename}",
        )
        for filename, expected in sorted(EXPECTED_NGHGI_HASHES.items())
    }
    return {
        "approved_package_sha256s_hash": _sha256(checksum_path),
        "approved_files": verified,
        "nghgi_files": nghgi_verified,
    }


def _build_asset(
    asset: PilotAsset,
    *,
    input_root: Path,
    nghgi_input_root: Path,
    nghgi_renderer: Path,
):
    if asset.builder_key == "Figure_15":
        figure, evidence, country_order = pngr.build_figure_15(
            nghgi_input_root, nghgi_renderer
        )
        pngr.validate_frozen_ratio_evidence(
            evidence,
            nghgi_input_root / "nghgi_publication_ratios.csv",
        )
        extra = {
            "country_order": list(country_order),
            "ratio_evidence_rows": len(evidence),
        }
        return figure, extra

    source = input_root / str(asset.source_relative_path)
    builder = pfr.PILOT_BUILDERS[asset.builder_key]
    return builder(source), {}


def render_pilot(
    *,
    input_root: Path,
    nghgi_input_root: Path,
    output_root: Path,
    nghgi_renderer: Path,
    dpi: int = 300,
    output_format: str = "tif",
) -> dict[str, object]:
    """Render and automatically validate the seven-asset visual pilot."""

    input_root = input_root.resolve()
    nghgi_input_root = nghgi_input_root.resolve()
    output_root = output_root.resolve()
    nghgi_renderer = nghgi_renderer.resolve()
    if output_format.lower() not in {"tif", "tiff"}:
        raise ValueError("The corrected publication package requires TIFF output")
    if dpi != 300:
        raise ValueError("The corrected publication pilot is fixed at 300 dpi")

    repository_root = Path(__file__).resolve().parents[4]
    provenance = pfp.build_publication_figure_provenance(
        repo_root=repository_root,
        approved_sha256s_path=input_root / "SHA256SUMS.txt",
        external_dependency_paths=(nghgi_renderer,),
    )
    frozen_evidence = _verify_frozen_inputs(input_root, nghgi_input_root)
    numerical_controls = _assert_numerical_controls(input_root)
    _verify_file(
        nghgi_renderer,
        pngr.EXPECTED_RENDERER_SHA256,
        "tracked NGHGI renderer",
    )

    figures_root = output_root / "figure_uploads"
    qa_root = output_root / "internal_qa"
    if any((figures_root / asset.upload_filename).exists() for asset in PILOT_ASSETS):
        raise FileExistsError(
            f"Pilot output already exists; choose a fresh output root: {output_root}"
        )
    figures_root.mkdir(parents=True, exist_ok=True)
    qa_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    qa_specs: list[pfq.FigureQASpec] = []
    for asset in PILOT_ASSETS:
        oracle = input_root / "figures" / asset.oracle_filename
        with Image.open(oracle) as oracle_image:
            oracle_width, oracle_height = oracle_image.size
        aspect_ratio = oracle_width / oracle_height
        output = figures_root / asset.upload_filename
        figure, extra = _build_asset(
            asset,
            input_root=input_root,
            nghgi_input_root=nghgi_input_root,
            nghgi_renderer=nghgi_renderer,
        )
        expected_width, expected_height = _prepare_final_layout(
            figure,
            asset,
            dpi=dpi,
            aspect_ratio=aspect_ratio,
        )
        artist_issues = pfe.publication_figure_validation_issues(
            figure,
            min_text_pt=8.0,
            min_line_pt=2.0,
        )
        extent_issues = pfe.publication_figure_extent_issues(figure)
        if artist_issues or extent_issues:
            raise ValueError(
                "Hard publication artist gate failed: "
                + "; ".join([*artist_issues, *extent_issues])
            )
        artist_gate_evidence = {
            "status": pfq.PASS,
            "minimum_visible_text_pt": 8.0,
            "minimum_meaningful_line_pt": 2.0,
            "artist_issue_count": 0,
            "active_extent_issue_count": 0,
        }
        try:
            dimensions = pc.save_publication_figure(
                figure,
                str(output),
                output_format=output_format,
                dpi=dpi,
                target_width_mm=asset.target_width_mm,
                aspect_ratio=aspect_ratio,
            )
        finally:
            plt.close(figure)
        if dimensions is None:
            raise RuntimeError("TIFF exporter did not return output dimensions")
        width_px, height_px = dimensions
        if height_px != expected_height:
            raise ValueError(
                f"{asset.asset_id} height mismatch: {height_px} vs {expected_height}"
            )
        if width_px != expected_width:
            raise ValueError(
                f"{asset.asset_id} width mismatch: {width_px} vs {expected_width}"
            )

        source_hash = (
            _sha256(input_root / str(asset.source_relative_path))
            if asset.source_relative_path
            else None
        )
        record = {
            **asdict(asset),
            "source_path": (
                str((input_root / str(asset.source_relative_path)).resolve())
                if asset.source_relative_path
                else str(nghgi_input_root),
            ),
            "source_sha256": source_hash,
            "oracle_path": str(oracle.resolve()),
            "oracle_sha256": _sha256(oracle),
            "oracle_dimensions_px": [oracle_width, oracle_height],
            "oracle_aspect_ratio": aspect_ratio,
            "output_path": str(output.resolve()),
            "output_sha256": _sha256(output),
            "output_dimensions_px": [width_px, height_px],
            "dpi": dpi,
            "artist_gate": artist_gate_evidence,
            **extra,
        }
        records.append(record)
        qa_specs.append(
            pfq.FigureQASpec(
                asset.asset_id,
                oracle,
                output,
                target_width_px=expected_width,
                target_dpi=float(dpi),
                allow_approved_geometry_reflow=True,
            )
        )

    qa_result = pfq.run_pilot_qa(qa_specs, qa_root)
    qa_by_asset = {
        str(item["asset_id"]): item for item in qa_result["results"]
    }
    for record in records:
        item = qa_by_asset[str(record["asset_id"])]
        record["qa"] = {
            "automated_status": item["automated_status"],
            "hard_failure_checks": item["hard_failure_checks"],
            "approved_exception_checks": item["approved_exception_checks"],
            "approved_exception_policy_id": item[
                "approved_exception_policy_id"
            ],
            "allow_approved_geometry_reflow": item[
                "allow_approved_geometry_reflow"
            ],
        }
    manifest = {
        "status": STATUS,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "seven-asset visual pilot",
        "asset_count": len(records),
        "format": "TIFF",
        "dpi": dpi,
        "boundaries": {
            "remote_queries": False,
            "coiled": False,
            "maps": False,
            "latex_or_pdf_changes": False,
            "remaining_assets_rendered": False,
        },
        "repository": {
            "afolu_path": str(repository_root),
            "afolu_commit": _git_commit(repository_root),
            "afolu_commit_scope": "repository base only; see provenance",
            "nghgi_renderer_path": str(nghgi_renderer),
            "nghgi_renderer_sha256": _sha256(nghgi_renderer),
            "nghgi_repository_commit": _git_commit(nghgi_renderer.parents[1]),
        },
        "provenance": provenance,
        "frozen_inputs": frozen_evidence,
        "numerical_controls": numerical_controls,
        "numerical_controls_status": pfq.PASS,
        "approved_exception_policy": pfq.GEOMETRY_REFLOW_EXCEPTION_POLICY,
        "approved_exception_count": qa_result["approved_exception_count"],
        "hard_failure_count": qa_result["hard_failure_count"],
        "qa_report_paths": qa_result["report_paths"],
        "automated_qa_status": qa_result["automated_status"],
        "visual_approval_status": qa_result["visual_approval_status"],
        "assets": records,
    }
    manifest_path = output_root / "pilot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--nghgi-input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nghgi-renderer", type=Path, required=True)
    parser.add_argument("--format", choices=("tif", "tiff"), default="tif")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--scope",
        choices=("pilot",),
        default="pilot",
        help="Full rendering remains locked until the pilot visual gate is approved.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = render_pilot(
        input_root=args.input_root,
        nghgi_input_root=args.nghgi_input_root,
        output_root=args.output_root,
        nghgi_renderer=args.nghgi_renderer,
        dpi=args.dpi,
        output_format=args.format,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "asset_count": result["asset_count"],
                "automated_qa_status": result["automated_qa_status"],
                "visual_approval_status": result["visual_approval_status"],
            },
            indent=2,
        )
    )
    return (
        0
        if pfq.automated_status_is_success(result["automated_qa_status"])
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
