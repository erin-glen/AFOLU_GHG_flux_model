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
