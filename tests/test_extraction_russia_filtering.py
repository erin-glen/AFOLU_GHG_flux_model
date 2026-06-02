import geopandas as gpd
import pytest
from shapely.geometry import Point

from src.scripts.preprocessing.extraction import extraction


def _gdf(rows):
    return gpd.GeoDataFrame(
        rows,
        geometry=[Point(i, i) for i in range(len(rows))],
        crs="EPSG:4326",
    )


def test_russia_filter_excludes_allocated_mineral_reserves():
    gdf = _gdf([{"name": "reserve"}])

    filtered = extraction.filter_gdf_dataset(
        gdf,
        "russia",
        source_name=extraction.RUSSIA_ALLOCATED_MINERAL_RESERVE,
    )

    assert filtered.empty
    assert filtered.crs == gdf.crs


def test_russia_filter_keeps_extraction_licenses_overlapping_interval_only():
    gdf = _gdf(
        [
            {
                "Type_lic": "extraction",
                "lic_date": "2020-01-01",
                "lic_expire": "2025-01-01",
                "cancel_dat": None,
                "label": "active",
            },
            {
                "Type_lic": " exploring ",
                "lic_date": "2020-01-01",
                "lic_expire": "2025-01-01",
                "cancel_dat": None,
                "label": "exploring",
            },
            {
                "Type_lic": "extraction",
                "lic_date": "2010-01-01",
                "lic_expire": "2020-12-31",
                "cancel_dat": None,
                "label": "expired_before_interval",
            },
            {
                "Type_lic": "extraction",
                "lic_date": "2010-01-01",
                "lic_expire": "2025-01-01",
                "cancel_dat": "2020-12-31",
                "label": "cancelled_before_interval",
            },
            {
                "Type_lic": "EXTRACTION",
                "lic_date": "2024-01-01",
                "lic_expire": "2024-06-01",
                "cancel_dat": "2024-06-01",
                "label": "overlaps_interval",
            },
            {
                "Type_lic": "extraction",
                "lic_date": "2025-01-01",
                "lic_expire": "2030-01-01",
                "cancel_dat": None,
                "label": "starts_after_interval",
            },
        ]
    )

    filtered = extraction.filter_gdf_dataset(
        gdf,
        "russia",
        source_name=extraction.RUSSIA_PEAT_EXTRACTION_DATES,
        russia_license_start_year=2021,
        russia_license_end_year=2024,
    )

    assert filtered["label"].tolist() == ["active", "overlaps_interval"]


def test_russia_filter_requires_source_name():
    gdf = _gdf([])

    with pytest.raises(ValueError, match="requires a source_name"):
        extraction.filter_gdf_dataset(gdf, "russia")


def test_russia_filter_requires_license_columns():
    gdf = _gdf([{"Type_lic": "extraction"}])

    with pytest.raises(ValueError, match="missing required column"):
        extraction.filter_gdf_dataset(
            gdf,
            "russia",
            source_name=extraction.RUSSIA_PEAT_EXTRACTION_DATES,
        )
