from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.scripts.zonal_statistics import rebuild_lulucf_wide_organic_soils as rebuild


def _master_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "iso3": "AAA",
        "interval_end": 2020,
        "climate_domain": "boreal",
        "drainage_class": "peat_undrained",
        "burned_state_meaning": None,
        "area__ha": 0.0,
        "drained_total_Mg_CO2e": None,
        "drained_co2_onsite_Mg_CO2": None,
        "drained_co2_offsite_Mg_CO2": None,
        "drained_total_co2_Mg_CO2": None,
        "drained_total_ch4_Mg_CO2e": None,
        "drained_n2o_Mg_CO2e": None,
        "burned_total_Mg_CO2e": None,
        "burned_total_co2_Mg_CO2": None,
        "burned_total_ch4_Mg_CO2e": None,
    }
    row.update(overrides)
    return row


def _write_master(path: Path, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    pq.write_table(table, path)


def _synthetic_master_rows() -> list[dict[str, object]]:
    return [
        _master_row(
            climate_domain="boreal",
            drainage_class="peat_drained_plantation",
            area__ha=10.0,
            drained_total_Mg_CO2e=100.0,
            drained_co2_onsite_Mg_CO2=60.0,
            drained_co2_offsite_Mg_CO2=10.0,
            drained_total_co2_Mg_CO2=70.0,
            drained_total_ch4_Mg_CO2e=20.0,
            drained_n2o_Mg_CO2e=10.0,
        ),
        _master_row(
            climate_domain="temperate",
            drainage_class="peat_undrained",
            burned_state_meaning="temperate__undrained",
            area__ha=20.0,
            burned_total_Mg_CO2e=50.0,
            burned_total_co2_Mg_CO2=40.0,
            burned_total_ch4_Mg_CO2e=10.0,
        ),
        _master_row(
            climate_domain="tropical",
            drainage_class="peat_drained_primary_infra",
            burned_state_meaning="tropical__drained_other",
            area__ha=30.0,
            drained_total_Mg_CO2e=80.0,
            drained_co2_onsite_Mg_CO2=40.0,
            drained_co2_offsite_Mg_CO2=10.0,
            drained_total_co2_Mg_CO2=50.0,
            drained_total_ch4_Mg_CO2e=20.0,
            drained_n2o_Mg_CO2e=10.0,
            burned_total_Mg_CO2e=30.0,
            burned_total_co2_Mg_CO2=24.0,
            burned_total_ch4_Mg_CO2e=6.0,
        ),
        _master_row(
            climate_domain="other_domain",
            drainage_class="peat_undrained",
            area__ha=40.0,
        ),
        _master_row(
            climate_domain="Unspecified",
            drainage_class="non_peat",
            area__ha=1000.0,
        ),
        # aggregate_master requires both production endpoints.
        _master_row(
            interval_end=2024,
            climate_domain="Unspecified",
            drainage_class="non_peat",
            area__ha=2000.0,
        ),
    ]


def _row(
    aggregation: pd.DataFrame, endpoint: int, climate: str
) -> pd.Series:
    selected = aggregation.loc[
        aggregation["interval_end"].eq(endpoint)
        & aggregation["climate_domain"].eq(climate)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def test_aggregate_master_maps_climates_excludes_nonpeat_and_classifies_states(
    tmp_path: Path,
) -> None:
    master_path = tmp_path / "master.parquet"
    _write_master(master_path, _synthetic_master_rows())

    aggregation = rebuild.aggregate_master(
        master_path, include_nonpeat_area=False
    )
    endpoint_2020 = aggregation.loc[aggregation["interval_end"].eq(2020)]
    assert set(endpoint_2020["climate_domain"]) == {
        "Boreal",
        "Temperate",
        "Subtropical/tropical",
        "Unassigned",
    }

    drained = _row(aggregation, 2020, "Boreal")
    assert drained["org_soil_drained_only__all_gases__MgCO2e_yr"] == 100.0
    assert drained["org_soil_drained_only__CO2_onsite__MgCO2_yr"] == 60.0
    assert drained["org_soil_drained_only__CO2_offsite__MgCO2_yr"] == 10.0
    assert drained["org_soil_drained_only__CO2__MgCO2_yr"] == 70.0
    assert drained["org_soil_drained_only__CH4__MgCO2e_yr"] == 20.0
    assert drained["org_soil_drained_only__N2O__MgCO2e_yr"] == 10.0
    assert drained["org_soil_burned_only__all_gases__MgCO2e_yr"] == 0.0
    assert drained["org_soil_drained_burned__all_gases__MgCO2e_yr"] == 0.0

    burned = _row(aggregation, 2020, "Temperate")
    assert burned["org_soil_burned_only__all_gases__MgCO2e_yr"] == 50.0
    assert burned["org_soil_burned_only__CO2__MgCO2_yr"] == 40.0
    assert burned["org_soil_burned_only__CH4__MgCO2e_yr"] == 10.0
    assert burned["org_soil_drained_only__all_gases__MgCO2e_yr"] == 0.0
    assert burned["org_soil_drained_burned__all_gases__MgCO2e_yr"] == 0.0

    combined = _row(aggregation, 2020, "Subtropical/tropical")
    assert combined["org_soil_drained_burned__all_gases__MgCO2e_yr"] == 110.0
    assert combined["org_soil_drained_burned__CO2__MgCO2_yr"] == 74.0
    assert combined["org_soil_drained_burned__CH4__MgCO2e_yr"] == 26.0
    assert combined["org_soil_drained_burned__N2O__MgCO2e_yr"] == 10.0
    assert combined["LULUCF_gross_emissions__all_gases__MgCO2e_yr"] == 110.0

    # other_domain and Unspecified collapse into one key, while the non-peat
    # row is excluded from the corrected organic-area definition.
    unassigned = _row(aggregation, 2020, "Unassigned")
    assert unassigned["org_soil__area_ha"] == 40.0
    assert unassigned["org_soil_emis__all_gases__MgCO2e_yr"] == 0.0

    legacy = rebuild.aggregate_master(master_path, include_nonpeat_area=True)
    assert _row(legacy, 2020, "Unassigned")["org_soil__area_ha"] == 1040.0


def test_aggregate_master_rejects_unexpected_climate(tmp_path: Path) -> None:
    rows = _synthetic_master_rows()
    rows[0]["climate_domain"] = "polar"
    master_path = tmp_path / "unexpected_climate.parquet"
    _write_master(master_path, rows)

    with pytest.raises(ValueError, match="Unexpected master climate values"):
        rebuild.aggregate_master(master_path, include_nonpeat_area=False)


def test_key_frame_expands_years_to_production_endpoints() -> None:
    years = list(range(2016, 2025))
    dimensions = {
        "country_name": ["Example"] * len(years),
        "adm0": ["AAA"] * len(years),
        "region_L1": ["Example"] * len(years),
        "region_L2_L3": ["Example"] * len(years),
        "year": [str(year) for year in years],
        "land_state_node": ["-999"] * len(years),
        "land_state_meaning": ["Unassigned"] * len(years),
        "land_state_broad_class": ["Unassigned"] * len(years),
        "land_state_detailed_class": ["Unassigned"] * len(years),
        "tall_veg_type": ["Unassigned"] * len(years),
        "climate_domain": ["Unassigned"] * len(years),
    }
    table = pa.Table.from_pydict(dimensions)

    keys = rebuild._key_frame(table, np.ones(len(years), dtype=bool))
    assert keys["interval_end"].tolist() == [2020] * 5 + [2024] * 4
    with pytest.raises(ValueError, match="Unsupported wide-artifact year"):
        rebuild.endpoint_for_year(2015)


def test_engineer_checks_filters_to_three_character_adm0_ids(
    tmp_path: Path,
) -> None:
    rebuilt = pa.Table.from_pydict(
        {
            "country_name": ["Example", "Example"],
            "adm0": ["AAA", "AAA"],
            "region_L1": ["Example", "Example"],
            "region_L2_L3": ["Example", "Example"],
            "year": ["2020", "2024"],
            "land_state_node": ["-999", "-999"],
            "land_state_meaning": ["Unassigned", "Unassigned"],
            "land_state_broad_class": ["Unassigned", "Unassigned"],
            "land_state_detailed_class": ["Unassigned", "Unassigned"],
            "tall_veg_type": ["Unassigned", "Unassigned"],
            "climate_domain": ["Unassigned", "Unassigned"],
            "org_soil__area_ha": pa.array([10.0, 20.0], type=pa.float32()),
            "org_soil_emis__all_gases__MgCO2e_yr": pa.array(
                [100.0, 200.0], type=pa.float32()
            ),
        }
    )
    engineer = pd.DataFrame(
        {
            "interval_end_year": [2020, 2024, 2020, 2024, 2020],
            "area_ha": [10.0, 20.0, 999.0, 999.0, 999.0],
            "gross_emissions_MgCO2e": [100.0, 200.0, 999.0, 999.0, 999.0],
            "aoi_id": ["AAA", "AAA", "AAA_1", "AAA_1", "BBB"],
            "aoi_type": ["admin", "admin", "admin", "admin", "basin"],
        }
    )
    engineer_path = tmp_path / "engineer.parquet"
    pq.write_table(
        pa.Table.from_pandas(engineer, preserve_index=False), engineer_path
    )

    comparison, summary = rebuild.engineer_checks(rebuilt, engineer_path)

    assert summary["engineer_rows"] == 2
    assert summary["matched_rows"] == 2
    assert summary["emission_pass_rows"] == 2
    assert summary["area_pass_rows"] == 2
    assert set(comparison["adm0"]) == {"AAA"}
