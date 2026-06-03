from pathlib import Path

import pandas as pd

from src.scripts.core_model.merge_chunk_stats_repair import (
    merge_chunk_stats_workbooks,
)


def _write_book(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


def test_merge_chunk_stats_replaces_tile_year_rows_and_recomputes_summaries(tmp_path):
    base = tmp_path / "base.xlsx"
    repair = tmp_path / "repair.xlsx"
    out = tmp_path / "merged.xlsx"

    base_detail = pd.DataFrame(
        [
            {
                "chunk_id": "80_50_81_51",
                "tile_id": "60N_080E",
                "layer_name": "drained_total_Mg_CO2e_ha_yr",
                "years": "2021_2024",
                "in_out": "output_layer",
                "min_value": 1,
                "mean_value": 1,
                "max_value": 1,
                "count_value": 10,
                "sum_value": 100,
                "data_type": "float32",
            },
            {
                "chunk_id": "20_60_21_61",
                "tile_id": "70N_020E",
                "layer_name": "drained_total_Mg_CO2e_ha_yr",
                "years": "2016_2020",
                "in_out": "output_layer",
                "min_value": 3,
                "mean_value": 3,
                "max_value": 3,
                "count_value": 30,
                "sum_value": 300,
                "data_type": "float32",
            },
            {
                "chunk_id": "0_0_1_1",
                "tile_id": "00N_000E",
                "layer_name": "drained_total_Mg_CO2e_ha_yr",
                "years": "2021_2024",
                "in_out": "output_layer",
                "min_value": 5,
                "mean_value": 5,
                "max_value": 5,
                "count_value": 50,
                "sum_value": 500,
                "data_type": "float32",
            },
        ]
    )
    repair_detail = pd.DataFrame(
        [
            {
                "chunk_id": "80_50_81_51",
                "tile_id": "60N_080E",
                "layer_name": "drained_total_Mg_CO2e_ha_yr",
                "years": "2021_2024",
                "in_out": "output_layer",
                "min_value": 2,
                "mean_value": 2,
                "max_value": 2,
                "count_value": 20,
                "sum_value": 200,
                "data_type": "float32",
            },
            {
                "chunk_id": "20_60_21_61",
                "tile_id": "70N_020E",
                "layer_name": "drained_total_Mg_CO2e_ha_yr",
                "years": "2016_2020",
                "in_out": "output_layer",
                "min_value": 4,
                "mean_value": 4,
                "max_value": 4,
                "count_value": 40,
                "sum_value": 400,
                "data_type": "float32",
            },
        ]
    )

    _write_book(
        base,
        {
            "other_outputs_1x1": base_detail,
            "min_max_summary": pd.DataFrame(),
            "pixel_counts_summary": pd.DataFrame(),
        },
    )
    _write_book(repair, {"other_outputs_1x1": repair_detail})

    merge_chunk_stats_workbooks(
        base_path=str(base),
        repair_paths=[str(repair)],
        output_path=str(out),
        tile_ids=["60N_080E"],
        years=["2021_2024"],
    )

    merged = pd.read_excel(out, sheet_name="other_outputs_1x1")
    repaired = merged[
        (merged["tile_id"] == "60N_080E") & (merged["years"] == "2021_2024")
    ]

    assert repaired["sum_value"].tolist() == [200]
    assert merged["sum_value"].sum() == 200 + 300 + 500

    counts = pd.read_excel(out, sheet_name="pixel_counts_summary")
    repaired_count = counts[
        (counts["tile_id"] == "60N_080E")
        & (counts["layer_name"] == "drained_total_Mg_CO2e_ha_yr")
    ]["total_pixel_count"].item()
    assert repaired_count == 20

    summary = pd.read_excel(out, sheet_name="min_max_summary")
    row = summary[summary["layer_name"] == "drained_total_Mg_CO2e_ha_yr"].iloc[0]
    assert row["min_value"] == 2
    assert row["max_value"] == 5
    assert row["count"] == 3
