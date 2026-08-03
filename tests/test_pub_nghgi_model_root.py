from __future__ import annotations

import pandas as pd
import pytest

from src.scripts.zonal_statistics.pub_scripts import pub_nghgi
from src.scripts.zonal_statistics.pub_scripts.pub_common import RunSpec


def _spec() -> RunSpec:
    return RunSpec(
        run_name="corrected_run",
        model_version="1_0_1",
        run_date="20260802",
        label="Corrected",
    )


def test_model_parquet_globs_use_isolated_root() -> None:
    assert pub_nghgi._model_parquet_globs(
        _spec(),
        "2021_2024",
        model_zonal_root="/mnt/c/tmp/afolu/corrected/",
    ) == [
        "/mnt/c/tmp/afolu/corrected/2021_2024/combined_state/*.parquet"
    ]


def test_model_parquet_globs_normalize_windows_separators() -> None:
    assert pub_nghgi._model_parquet_globs(
        _spec(),
        "2021_2024",
        model_zonal_root=r"C:\tmp\corrected run",
    ) == ["C:/tmp/corrected run/2021_2024/combined_state/*.parquet"]


def test_model_parquet_globs_preserve_canonical_default(monkeypatch) -> None:
    monkeypatch.setattr(
        pub_nghgi,
        "build_output_parquet",
        lambda *_args: "s3://bucket/canonical/2021_2024/",
    )

    assert pub_nghgi._model_parquet_globs(_spec(), "2021_2024") == [
        "s3://bucket/canonical/2021_2024/combined_state/*.parquet"
    ]


def test_validate_model_interval_end_accepts_matching_endpoint() -> None:
    pub_nghgi._validate_model_interval_end(
        pd.DataFrame({"interval_end": [2024, 2024]}), "2021_2024"
    )


def test_validate_model_interval_end_rejects_wrong_endpoint() -> None:
    with pytest.raises(ValueError, match="expected interval_end=2024"):
        pub_nghgi._validate_model_interval_end(
            pd.DataFrame({"interval_end": [2020]}), "2021_2024"
        )


@pytest.mark.parametrize("invalid_value", [None, "not-a-year"])
def test_validate_model_interval_end_rejects_invalid_values(
    invalid_value: object,
) -> None:
    with pytest.raises(ValueError, match="invalid interval_end"):
        pub_nghgi._validate_model_interval_end(
            pd.DataFrame({"interval_end": [2024, invalid_value]}), "2021_2024"
        )


def test_validate_model_interval_end_rejects_fractional_endpoint() -> None:
    with pytest.raises(ValueError, match="non-integral interval_end"):
        pub_nghgi._validate_model_interval_end(
            pd.DataFrame({"interval_end": [2024.5]}), "2021_2024"
        )


def test_missing_isolated_interval_fails_without_httpfs(monkeypatch) -> None:
    monkeypatch.setattr(pub_nghgi.pa, "_count_globs", lambda *_args: 0)

    def fail_httpfs(*_args):
        raise AssertionError("local model routing must not install httpfs")

    monkeypatch.setattr(pub_nghgi.pa, "_ensure_httpfs", fail_httpfs)

    with pytest.raises(FileNotFoundError, match="2021_2024"):
        pub_nghgi._model_country_landuse_for_interval(
            _spec(),
            "2021_2024",
            aws_region=None,
            adm0_lookup_csv=None,
            model_zonal_root="/mnt/c/tmp/afolu/corrected",
        )


def test_load_model_country_landuse_forwards_isolated_root(monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_load(
        _spec_arg,
        interval_folder,
        _aws_region,
        _adm0_lookup_csv,
        model_zonal_root=None,
    ):
        calls.append((interval_folder, model_zonal_root))
        return pd.DataFrame(
            {
                "interval_end": [2024],
                "gadm_adm0": [1],
                "iso3": ["AAA"],
                "country": ["Example"],
                "land_use": ["Forest"],
                "drained_area_ha": [1.0],
                "undrained_area_ha": [2.0],
                "drained_on_site_co2_Mg_CO2_yr": [3.0],
                "drained_n2o_Mg_CO2e_yr": [4.0],
            }
        )

    monkeypatch.setattr(
        pub_nghgi, "_model_country_landuse_for_interval", fake_load
    )

    result = pub_nghgi.load_model_country_landuse(
        [_spec()],
        [2024],
        aws_region=None,
        adm0_lookup_csv=None,
        model_zonal_root="/mnt/c/tmp/afolu/corrected",
    )

    assert calls == [
        ("2021_2024", "/mnt/c/tmp/afolu/corrected")
    ]
    assert result.loc[0, "run_name"] == "corrected_run"
    assert result.loc[0, "interval"] == "2021_2024"
