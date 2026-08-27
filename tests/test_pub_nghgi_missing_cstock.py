import numpy as np
import pandas as pd

from src.scripts.zonal_statistics.pub_scripts import pub_nghgi


def test_all_nan_cstock_fallback_stays_missing():
    nghgi_t4ii = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "year": 2001,
                "land_use": "Forest",
                "category_code": "4(II).A.1",
                "soil_type": "Drained organic soils",
                "area_kha": np.nan,
                "em_co2_kt": np.nan,
                "em_n2o_kt": np.nan,
                "em_ch4_kt": np.nan,
            }
        ]
    )
    nghgi_cstock = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "year": 2001,
                "land_use": "Forest",
                "category_code": "4.A",
                "area_organic_kha": np.nan,
                "cstock_soil_organic_ktC": np.nan,
            }
        ]
    )

    out = pub_nghgi.nghgi_by_iso_landuse_interval(
        nghgi_t4ii,
        nghgi_cstock,
        [(2001, 2001)],
    )

    row = out.loc[(out["iso3"] == "AAA") & (out["land_use"] == "Forest")].iloc[0]
    assert pd.isna(row["nghgi_area_total_organic_ha"])
    assert pd.isna(row["nghgi_em_co2_from_cstock_kt"])
    assert pd.isna(row["nghgi_em_co2_Mg_yr"])
    assert row["nghgi_em_co2_source"] is None


def test_undrained_area_uses_raw_area_when_jrc_overrides_cstock():
    nghgi_t4ii = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "year": 2001,
                "land_use": "Forest",
                "category_code": "4(II).A.1",
                "soil_type": "Drained organic soils",
                "area_kha": 90.0,
                "em_co2_kt": np.nan,
                "em_n2o_kt": np.nan,
                "em_ch4_kt": np.nan,
            }
        ]
    )
    nghgi_cstock = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "year": 2001,
                "land_use": "Forest",
                "category_code": "4.A",
                "area_organic_kha": 100.0,
                "cstock_soil_organic_ktC": -3.0,
            }
        ]
    )
    jrc_lu = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "year": 2001,
                "land_use": "Forest",
                "area_total_kha": 200.0,
                "area_organic_kha": 80.0,
                "cstock_organic_ktC": -4.0,
                "source": "JRC_AnnexI_2026",
            }
        ]
    )

    out = pub_nghgi.nghgi_by_iso_landuse_interval(
        nghgi_t4ii,
        nghgi_cstock,
        [(2001, 2001)],
        jrc_lu=jrc_lu,
    )

    row = out.loc[(out["iso3"] == "AAA") & (out["land_use"] == "Forest")].iloc[0]
    assert row["nghgi_area_total_organic_ha"] == 80_000.0
    assert row["nghgi_area_undrained_organic_ha"] == 10_000.0
    assert row["nghgi_area_total_organic_source"] == "JRC_AnnexI_2026"
    assert row["nghgi_area_undrained_organic_source"] == "raw_T4land_minus_raw_T4II"
    assert row["nghgi_em_co2_source"] == "T4land_cstock"
    assert np.isclose(row["nghgi_em_co2_Mg_yr"], 4.0 * pub_nghgi.C_TO_CO2 * 1_000.0)


def test_metric_specific_availability_counts_by_interval():
    nghgi_t4ii = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "year": 2001,
                "land_use": "Forest",
                "category_code": "4(II).A.1",
                "soil_type": "Drained organic soils",
                "area_kha": 10.0,
                "em_co2_kt": np.nan,
                "em_n2o_kt": 1.0,
                "em_ch4_kt": np.nan,
            },
            {
                "iso3": "AAA",
                "year": 2002,
                "land_use": "Forest",
                "category_code": "4(II).A.1",
                "soil_type": "Drained organic soils",
                "area_kha": np.nan,
                "em_co2_kt": 5.0,
                "em_n2o_kt": np.nan,
                "em_ch4_kt": 2.0,
            },
        ]
    )
    nghgi_cstock = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "year": 2001,
                "land_use": "Forest",
                "category_code": "4.A",
                "area_organic_kha": 15.0,
                "cstock_soil_organic_ktC": -3.0,
            },
            {
                "iso3": "AAA",
                "year": 2002,
                "land_use": "Forest",
                "category_code": "4.A",
                "area_organic_kha": 20.0,
                "cstock_soil_organic_ktC": np.nan,
            },
        ]
    )

    out = pub_nghgi.nghgi_by_iso_landuse_interval(
        nghgi_t4ii,
        nghgi_cstock,
        [(2001, 2002)],
    )

    row = out.loc[(out["iso3"] == "AAA") & (out["land_use"] == "Forest")].iloc[0]
    expected_co2_kt = ((3.0 * pub_nghgi.C_TO_CO2) + 5.0) / 2.0
    assert np.isclose(row["nghgi_em_co2_Mg_yr"], expected_co2_kt * 1_000.0)
    assert row["nghgi_em_co2_source"] == "mixed_T4II_T4land_cstock"
    assert row["nghgi_area_drained_years_available"] == 1
    assert row["nghgi_area_total_years_available"] == 2
    assert row["nghgi_area_undrained_basis_years_available"] == 1
    assert row["nghgi_area_undrained_years_available"] == 1
    assert row["nghgi_em_co2_t4ii_years_available"] == 1
    assert row["nghgi_em_co2_cstock_years_available"] == 1
    assert row["nghgi_em_co2_years_available"] == 2
    assert row["nghgi_em_n2o_years_available"] == 1
    assert row["nghgi_em_ch4_years_available"] == 1
    assert row["nghgi_area_undrained_organic_ha"] == 5_000.0


def test_availability_matrix_has_year_metric_rows():
    nghgi_t4ii = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "year": 2001,
                "land_use": "Forest",
                "category_code": "4(II).A.1",
                "soil_type": "Drained organic soils",
                "area_kha": 10.0,
                "em_co2_kt": np.nan,
                "em_n2o_kt": 1.0,
                "em_ch4_kt": np.nan,
            },
            {
                "iso3": "AAA",
                "year": 2002,
                "land_use": "Forest",
                "category_code": "4(II).A.1",
                "soil_type": "Drained organic soils",
                "area_kha": np.nan,
                "em_co2_kt": 5.0,
                "em_n2o_kt": np.nan,
                "em_ch4_kt": np.nan,
            },
        ]
    )
    nghgi_cstock = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "year": 2001,
                "land_use": "Forest",
                "category_code": "4.A",
                "area_organic_kha": 15.0,
                "cstock_soil_organic_ktC": -3.0,
            },
            {
                "iso3": "AAA",
                "year": 2002,
                "land_use": "Forest",
                "category_code": "4.A",
                "area_organic_kha": 20.0,
                "cstock_soil_organic_ktC": np.nan,
            },
        ]
    )

    annual = pub_nghgi.nghgi_annual_by_iso_landuse(nghgi_t4ii, nghgi_cstock)
    matrix = pub_nghgi.build_nghgi_availability_matrix(annual, [(2001, 2002)])

    n2o_2001 = matrix.loc[
        (matrix["metric"] == "n2o_t4ii") & (matrix["year"] == 2001)
    ].iloc[0]
    n2o_2002 = matrix.loc[
        (matrix["metric"] == "n2o_t4ii") & (matrix["year"] == 2002)
    ].iloc[0]
    total_2002 = matrix.loc[
        (matrix["metric"] == "area_total_organic") & (matrix["year"] == 2002)
    ].iloc[0]
    undrained_2002 = matrix.loc[
        (matrix["metric"] == "area_undrained_organic") & (matrix["year"] == 2002)
    ].iloc[0]

    assert bool(n2o_2001["has_value"])
    assert not bool(n2o_2002["has_value"])
    assert bool(total_2002["has_value"])
    assert total_2002["source"] == "raw_CRT_extract"
    assert not bool(undrained_2002["has_value"])


def test_t3d_metric_specific_availability_counts_and_matrix():
    jrc_t3d = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "year": 2001,
                "area_ha": 10.0,
                "n2o_kt": np.nan,
                "source": "JRC_BTR1_2024",
            },
            {
                "iso3": "AAA",
                "year": 2002,
                "area_ha": np.nan,
                "n2o_kt": 2.0,
                "source": "JRC_BTR1_2024",
            },
        ]
    )

    annual = pub_nghgi.nghgi_t3d_annual(jrc_t3d)
    out = pub_nghgi.nghgi_t3d_by_iso_interval(
        jrc_t3d,
        [(2001, 2002)],
        annual_df=annual,
    )
    row = out.loc[out["iso3"] == "AAA"].iloc[0]
    assert row["nghgi_t3d_area_years_available"] == 1
    assert row["nghgi_t3d_n2o_years_available"] == 1
    assert row["nghgi_t3d_area_ha"] == 10.0
    assert row["nghgi_t3d_n2o_kt"] == 2.0

    matrix = pub_nghgi.build_nghgi_availability_matrix(
        pd.DataFrame(),
        [(2001, 2002)],
        t3d_annual_df=annual,
    )
    n2o_2002 = matrix.loc[
        (matrix["metric"] == "n2o_t3d") & (matrix["year"] == 2002)
    ].iloc[0]
    assert bool(n2o_2002["has_value"])
    assert n2o_2002["land_use"] == "All"


def test_join_preserves_nghgi_only_interval_metadata():
    model_df = pd.DataFrame(
        [
            {
                "run_label": "Run",
                "run_name": "run",
                "model_version": "0",
                "run_date": "20260101",
                "interval": "2001_2005",
                "interval_start": 2001,
                "interval_end": 2005,
                "gadm_adm0": "AAA",
                "iso3": "AAA",
                "country": "A",
                "land_use": "Forest",
                "drained_area_ha": 1.0,
                "undrained_area_ha": 0.0,
                "drained_on_site_co2_Mg_CO2_yr": 2.0,
                "drained_n2o_Mg_CO2e_yr": 3.0,
            }
        ]
    )
    nghgi_df = pd.DataFrame(
        [
            {
                "iso3": "BBB",
                "land_use": "Cropland",
                "interval": "2001_2005",
                "interval_start": 2001,
                "interval_end": 2005,
                "nghgi_area_drained_organic_ha": 10.0,
                "nghgi_em_co2_Mg_yr": 20.0,
                "nghgi_years_available": 1,
            }
        ]
    )

    joined = pub_nghgi.join_model_nghgi(model_df, nghgi_df)
    row = joined.loc[joined["iso3"] == "BBB"].iloc[0]
    assert row["interval"] == "2001_2005"
    assert row["interval_start"] == 2001
    assert row["run_label"] == "Run"
    assert pd.isna(row["model_drained_area_ha"])


def test_t3d_join_preserves_nghgi_only_interval_metadata():
    model_df = pd.DataFrame(
        [
            {
                "run_label": "Run",
                "run_name": "run",
                "model_version": "0",
                "run_date": "20260101",
                "interval": "2001_2005",
                "interval_start": 2001,
                "interval_end": 2005,
                "gadm_adm0": "AAA",
                "iso3": "AAA",
                "country": "A",
                "land_use": "Cropland",
                "drained_area_ha": 1.0,
                "drained_n2o_Mg_CO2e_yr": 3.0,
            }
        ]
    )
    t3d_df = pd.DataFrame(
        [
            {
                "iso3": "BBB",
                "interval": "2001_2005",
                "interval_start": 2001,
                "interval_end": 2005,
                "nghgi_t3d_area_ha": 10.0,
                "nghgi_t3d_n2o_kt": 1.0,
                "nghgi_t3d_n2o_Mg_CO2e_yr": 265_000.0,
                "nghgi_t3d_source": "JRC_BTR1_2024",
                "nghgi_t3d_years_available": 1,
            }
        ]
    )

    joined = pub_nghgi.join_model_nghgi_t3d(model_df, t3d_df)
    row = joined.loc[joined["iso3"] == "BBB"].iloc[0]
    assert row["interval"] == "2001_2005"
    assert row["interval_start"] == 2001
    assert row["run_label"] == "Run"
    assert pd.isna(row["model_t3d_drained_area_ha"])


def test_matched_sum_keeps_reported_zero():
    df = pd.DataFrame(
        [
            {"iso3": "AAA", "model": 5.0, "nghgi": 0.0},
            {"iso3": "BBB", "model": 7.0, "nghgi": np.nan},
        ]
    )

    out = pub_nghgi._matched_sum(df, "model", "nghgi")

    assert out["iso3"].tolist() == ["AAA"]
    assert out.loc[0, "model"] == 5.0
    assert out.loc[0, "nghgi"] == 0.0


def test_join_model_nghgi_zero_denominators_have_undefined_ratios():
    model_df = pd.DataFrame(
        [
            {
                "run_label": "Run",
                "run_name": "run",
                "model_version": "0",
                "run_date": "20260101",
                "interval": "2001_2005",
                "interval_start": 2001,
                "interval_end": 2005,
                "gadm_adm0": "AAA",
                "iso3": "AAA",
                "country": "A",
                "land_use": "Forest",
                "drained_area_ha": 1.0,
                "undrained_area_ha": 0.0,
                "drained_on_site_co2_Mg_CO2_yr": 2.0,
                "drained_n2o_Mg_CO2e_yr": 3.0,
            }
        ]
    )
    nghgi_df = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "land_use": "Forest",
                "interval": "2001_2005",
                "interval_start": 2001,
                "interval_end": 2005,
                "nghgi_area_drained_organic_ha": 0.0,
                "nghgi_em_co2_Mg_yr": 0.0,
            }
        ]
    )

    joined = pub_nghgi.join_model_nghgi(model_df, nghgi_df)

    assert pd.isna(joined.loc[0, "area_ratio_model_over_nghgi"])
    assert pd.isna(joined.loc[0, "co2_ratio_model_over_nghgi"])
