import duckdb
import pandas as pd

from src.scripts.zonal_statistics.pub_scripts import pub_common as pc


def _flux_and_area_rows(
    *,
    interval_end: int,
    gadm_adm0: int,
    state_col: str,
    meaning_col: str,
    state_nodes: int,
    state_meaning: str | None,
    flux_type: str,
    flux_value: float,
    area_ha: float,
) -> list[dict[str, object]]:
    return [
        {
            "interval_end": interval_end,
            "gadm_adm0": gadm_adm0,
            state_col: state_nodes,
            meaning_col: state_meaning,
            "flux_type": flux_type,
            "value": flux_value,
        },
        {
            "interval_end": interval_end,
            "gadm_adm0": gadm_adm0,
            state_col: state_nodes,
            meaning_col: state_meaning,
            "flux_type": "area__ha",
            "value": area_ha,
        },
    ]


def _register_lulucf_fixture(con: duckdb.DuckDBPyConnection) -> None:
    drained_ctx = pd.DataFrame(
        [
            {
                "key": "12121000",
                "meaning": "peat_drained_secondary_infra__temperate_extraction",
                "climate_domain": "temperate",
                "drained_state": "peat_drained_secondary_infra",
                "emissions_state": "extraction",
            },
            {
                "key": "15921192",
                "meaning": "peat_drained_extraction__boreal_coastal_tidal_marsh",
                "climate_domain": "boreal",
                "drained_state": "peat_drained_extraction",
                "emissions_state": "coastal_tidal_marsh",
            },
            {
                "key": "13125000",
                "meaning": "peat_drained_cropland_settlement__temperate_cropland",
                "climate_domain": "temperate",
                "drained_state": "peat_drained_cropland_settlement",
                "emissions_state": "cropland",
            },
            {
                "key": "16000000",
                "meaning": "peat_undrained",
                "climate_domain": None,
                "drained_state": "peat_undrained",
                "emissions_state": None,
            },
            {
                "key": "00000000",
                "meaning": "non_peat",
                "climate_domain": None,
                "drained_state": "non_peat",
                "emissions_state": None,
            },
        ]
    )
    burned_ctx = pd.DataFrame(
        [
            {
                "key": "11200000",
                "meaning": "boreal__undrained",
                "climate_domain": "boreal",
                "burned_state": "undrained",
                "emissions_state": "undrained",
            },
            {
                "key": "33300000",
                "meaning": "tropical__undrained",
                "climate_domain": "tropical",
                "burned_state": "undrained",
                "emissions_state": "undrained",
            },
        ]
    )
    drained_rows: list[dict[str, object]] = []
    drained_rows += _flux_and_area_rows(
        interval_end=2024,
        gadm_adm0=840,
        state_col="drained_state_nodes",
        meaning_col="drained_state_meaning",
        state_nodes=12121000,
        state_meaning="peat_drained_secondary_infra__temperate_extraction",
        flux_type="drained_total_Mg_CO2e",
        flux_value=5.0,
        area_ha=6.0,
    )
    drained_rows += _flux_and_area_rows(
        interval_end=2024,
        gadm_adm0=840,
        state_col="drained_state_nodes",
        meaning_col="drained_state_meaning",
        state_nodes=15921192,
        state_meaning="peat_drained_extraction__boreal_coastal_tidal_marsh",
        flux_type="drained_total_Mg_CO2e",
        flux_value=3.0,
        area_ha=4.0,
    )
    drained_rows += _flux_and_area_rows(
        interval_end=2024,
        gadm_adm0=840,
        state_col="drained_state_nodes",
        meaning_col="drained_state_meaning",
        state_nodes=13125000,
        state_meaning="peat_drained_cropland_settlement__temperate_cropland",
        flux_type="drained_total_Mg_CO2e",
        flux_value=7.0,
        area_ha=8.0,
    )
    drained_rows += _flux_and_area_rows(
        interval_end=2024,
        gadm_adm0=840,
        state_col="drained_state_nodes",
        meaning_col="drained_state_meaning",
        state_nodes=16000000,
        state_meaning="peat_undrained",
        flux_type="drained_total_Mg_CO2e",
        flux_value=0.0,
        area_ha=1000.0,
    )
    drained_rows += _flux_and_area_rows(
        interval_end=2024,
        gadm_adm0=840,
        state_col="drained_state_nodes",
        meaning_col="drained_state_meaning",
        state_nodes=0,
        state_meaning="non_peat",
        flux_type="drained_total_Mg_CO2e",
        flux_value=0.0,
        area_ha=9000.0,
    )

    burned_rows: list[dict[str, object]] = []
    burned_rows += _flux_and_area_rows(
        interval_end=2024,
        gadm_adm0=840,
        state_col="burned_state_nodes",
        meaning_col="burned_state_meaning",
        state_nodes=11200000,
        state_meaning="boreal__undrained",
        flux_type="burned_total_Mg_CO2e",
        flux_value=20.0,
        area_ha=10.0,
    )
    burned_rows += _flux_and_area_rows(
        interval_end=2024,
        gadm_adm0=840,
        state_col="burned_state_nodes",
        meaning_col="burned_state_meaning",
        state_nodes=33300000,
        state_meaning="tropical__undrained",
        flux_type="burned_total_Mg_CO2e",
        flux_value=0.0,
        area_ha=11.0,
    )
    burned_rows += _flux_and_area_rows(
        interval_end=2024,
        gadm_adm0=840,
        state_col="burned_state_nodes",
        meaning_col="burned_state_meaning",
        state_nodes=0,
        state_meaning="unburned",
        flux_type="burned_total_Mg_CO2e",
        flux_value=0.0,
        area_ha=9000.0,
    )

    con.register("drained_state_ctx", drained_ctx)
    con.register("burned_state_ctx", burned_ctx)
    con.register("zs_drained", pd.DataFrame(drained_rows))
    con.register("zs_burned", pd.DataFrame(burned_rows))


def test_stats_for_lulucf_paper_excludes_background_area_and_keeps_real_components() -> None:
    con = duckdb.connect()
    _register_lulucf_fixture(con)

    df = con.execute(pc.table_stats_for_lulucf_paper_sql(with_lookup=False)).df()
    by_component = df.groupby("component", as_index=True)[["flux_Mg_CO2e_yr", "area_ha"]].sum()

    assert "Unspecified" not in set(df["climate_domain"])
    assert by_component.loc["Drainage"].to_dict() == {
        "flux_Mg_CO2e_yr": 7.0,
        "area_ha": 8.0,
    }
    assert by_component.loc["Extraction"].to_dict() == {
        "flux_Mg_CO2e_yr": 8.0,
        "area_ha": 10.0,
    }
    assert by_component.loc["Fire"].to_dict() == {
        "flux_Mg_CO2e_yr": 20.0,
        "area_ha": 21.0,
    }
    assert df["area_ha"].sum() == 39.0


def test_stats_for_lulucf_paper_classifies_extraction_root_as_extraction() -> None:
    con = duckdb.connect()
    _register_lulucf_fixture(con)

    df = con.execute(pc.table_stats_for_lulucf_paper_sql(with_lookup=False)).df()

    coastal_extraction = df[
        (df["climate_domain"] == "boreal")
        & (df["component"] == "Extraction")
    ]
    assert coastal_extraction["flux_Mg_CO2e_yr"].sum() == 3.0
    assert coastal_extraction["area_ha"].sum() == 4.0
