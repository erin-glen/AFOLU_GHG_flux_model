from src.scripts.preprocessing.hansenize import hansenize_coiled
from src.scripts.preprocessing.peat import peat_masks


def test_tile_intersects_source_bounds():
    source_bounds = (-180.0005, -56.0005, 180.0005, 66.0005)

    assert peat_masks._tile_intersects_bounds("70N_000E", source_bounds)
    assert not peat_masks._tile_intersects_bounds("80N_000E", source_bounds)


def test_filter_tiles_to_source_bounds(monkeypatch):
    monkeypatch.setattr(
        peat_masks,
        "_source_raster_bounds",
        lambda _path: (-180.0005, -56.0005, 180.0005, 66.0005),
    )

    filtered = peat_masks._filter_tiles_to_source_bounds(
        ["70N_000E", "80N_000E"],
        "https://example.org/source.tif",
    )

    assert filtered == ["70N_000E"]


def test_resolve_raw_path_preserves_explicit_https_url_when_raw_date_is_set():
    url = "https://s3.opengeohub.org/global/soil.types/organic.soils.tif"
    ds = {"s3_raw": "climate/AFOLU_flux_model/organic_soils/inputs/raw/soils/OGH/20260508/source.tif"}

    assert peat_masks.resolve_raw_path(ds, raw_date="20260509", raw_path=url) == url


def test_resolve_raw_path_dates_configured_relative_source():
    ds = {"s3_raw": "climate/AFOLU_flux_model/organic_soils/inputs/raw/soils/OGH/20260508/source.tif"}

    resolved = peat_masks.resolve_raw_path(ds, raw_date="20260509")

    assert resolved == (
        f"s3://{peat_masks.BUCKET}/"
        "climate/AFOLU_flux_model/organic_soils/inputs/raw/soils/OGH/20260509/source.tif"
    )


def test_source_file_exists_uses_bucket_from_s3_uri(monkeypatch):
    calls = []

    def fake_s3_file_exists(bucket, key):
        calls.append((bucket, key))
        return True

    monkeypatch.setattr(peat_masks.uutil, "s3_file_exists", fake_s3_file_exists)

    assert peat_masks._source_file_exists("s3://other-bucket/path/to/source.tif")
    assert calls == [("other-bucket", "path/to/source.tif")]


def test_http_tif_is_single_file_source():
    url = "https://s3.opengeohub.org/global/soil.types/source.tif?download=1"

    assert peat_masks._is_single_tif_path(url)
    assert peat_masks._source_file_exists(url)


def test_gdal_input_path_translates_https_to_vsicurl():
    assert hansenize_coiled._gdal_input_path("https://example.org/source.tif") == (
        "/vsicurl/https://example.org/source.tif"
    )
    assert hansenize_coiled._gdal_input_path("s3://bucket/path/source.tif") == (
        "/vsis3/bucket/path/source.tif"
    )
    assert hansenize_coiled._gdal_input_path("/vsicurl/https://example.org/source.tif") == (
        "/vsicurl/https://example.org/source.tif"
    )
