import logging

import numpy as np
import pytest

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu


@pytest.mark.parametrize(
    ("start_year", "end_year", "expected_path"),
    [
        (2001, 2005, "/five_year/2005/"),
        (2006, 2010, "/five_year/2010/"),
        (2011, 2015, "/five_year/2015/"),
        (2016, 2020, "/five_year/2020/"),
        (2021, 2024, "/annual/2024/"),
        (2024, 2024, "/annual/2024/"),
    ],
)
def test_land_cover_source_matches_inventory_interval(
    start_year,
    end_year,
    expected_path,
):
    inputs = cn.get_dynamic_download_dict(
        "{tile_id}",
        start_year,
        end_year,
    )

    assert expected_path in inputs["land_cover"]


def test_sparse_historical_annual_land_cover_is_not_supported():
    with pytest.raises(ValueError, match="Unsupported land-cover period 2015-2015"):
        cn.get_dynamic_download_dict("{tile_id}", 2015, 2015)


@pytest.mark.parametrize("peat_dataset", cn.peat_dataset_choices)
def test_dynamic_inputs_have_only_the_selected_peat_layer(peat_dataset):
    inputs = cn.get_dynamic_download_dict(
        "{tile_id}",
        2021,
        2024,
        peat_dataset=peat_dataset,
    )

    assert "peat" in inputs
    assert "ogh" not in inputs
    if peat_dataset == "ogh":
        assert "ogh_unthresholded" in inputs["peat"]


class _FakePaginator:
    def __init__(self, pages):
        self.pages = pages

    def paginate(self, **kwargs):
        assert kwargs["Bucket"] == "example-bucket"
        assert kwargs["Prefix"] == "model/land_cover/"
        return self.pages


class _FakeS3Client:
    def __init__(self, pages):
        self.pages = pages

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return _FakePaginator(self.pages)


def test_required_tile_coverage_accepts_complete_prefix():
    client = _FakeS3Client(
        [
            {"Contents": [{"Key": "model/land_cover/00N_110E__lc_ipcc.tif"}]},
            {"Contents": [{"Key": "model/land_cover/60N_110W__lc_ipcc.tif"}]},
        ]
    )

    count = uu.validate_required_s3_tile_coverage(
        "s3://example-bucket/model/land_cover/{tile_id}__lc_ipcc.tif",
        ["00N_110E", "60N_110W"],
        layer_name="land_cover",
        s3_client=client,
    )

    assert count == 2


def test_tile_listing_rejects_zero_byte_raster_objects():
    client = _FakeS3Client(
        [
            {
                "Contents": [
                    {
                        "Key": "model/land_cover/00N_110E__lc_ipcc.tif",
                        "Size": 0,
                    }
                ]
            }
        ]
    )

    with pytest.raises(
        uu.RequiredInputRasterError,
        match="zero-byte raster objects",
    ):
        uu.list_existing_s3_tile_ids(
            "s3://example-bucket/model/land_cover/{tile_id}__lc_ipcc.tif",
            ["00N_110E"],
            s3_client=client,
        )


def test_tile_set_fingerprint_is_order_and_duplicate_independent():
    expected = uu.tile_id_set_sha256(["00N_110E", "60N_110W"])

    assert uu.tile_id_set_sha256(
        ["60N_110W", "00N_110E", "00N_110E"]
    ) == expected


def test_tile_set_fingerprint_gate_rejects_count_or_membership_drift():
    expected = uu.tile_id_set_sha256(["00N_110E", "60N_110W"])

    with pytest.raises(
        uu.RequiredInputRasterError,
        match="footprint drifted from the audited reference",
    ):
        uu.validate_tile_set_fingerprint(
            ["00N_110E"],
            expected_count=2,
            expected_sha256=expected,
            layer_name="land_cover",
        )

    with pytest.raises(uu.RequiredInputRasterError, match="sha256="):
        uu.validate_tile_set_fingerprint(
            ["00N_110E", "70N_020E"],
            expected_count=2,
            expected_sha256=expected,
            layer_name="land_cover",
        )


def test_required_layer_flag_reaches_parallel_window_reader(monkeypatch):
    observed = {}

    def fake_reader(uri, dtype, bounds, chunk_px, logger, is_final, required):
        observed[uri] = required
        return np.zeros((chunk_px, chunk_px), dtype=np.uint8)

    monkeypatch.setattr(uu, "open_window_as_array", fake_reader)
    futures = uu.queue_chunk_downloads(
        [0, 0, 0.1, 0.1],
        {
            "peat": ("s3://example/peat.tif", "Byte"),
            "optional": ("s3://example/optional.tif", "Byte"),
        },
        400,
        logging.getLogger(__name__),
        required_layers={"peat"},
    )
    for future in futures:
        future.result()

    assert observed == {
        "s3://example/peat.tif": True,
        "s3://example/optional.tif": False,
    }


def test_chunk_pixel_length_rounds_grid_aligned_float_bounds():
    assert uu.calc_chunk_length_pixels([25.0, 64.9, 25.1, 65.0]) == 400


def test_chunk_pixel_length_rejects_non_square_or_off_grid_bounds():
    with pytest.raises(ValueError, match="positive square aligned"):
        uu.calc_chunk_length_pixels([25.0, 64.9, 25.1, 65.0001])


def test_required_tile_coverage_rejects_sparse_prefix():
    client = _FakeS3Client(
        [{"Contents": [{"Key": "model/land_cover/00N_110E__lc_ipcc.tif"}]}]
    )

    with pytest.raises(
        uu.RequiredInputRasterError,
        match=r"1 of 2 requested tiles are missing.*60N_110W",
    ):
        uu.validate_required_s3_tile_coverage(
            "s3://example-bucket/model/land_cover/{tile_id}__lc_ipcc.tif",
            ["00N_110E", "60N_110W"],
            layer_name="land_cover",
            s3_client=client,
        )


def test_required_raster_read_fails_instead_of_returning_zeros(monkeypatch):
    def fail_open(*args, **kwargs):
        raise RuntimeError("not found")

    monkeypatch.setattr(uu, "rio_open", fail_open)

    with pytest.raises(
        uu.RequiredInputRasterError,
        match="could not be read: not found",
    ):
        uu.open_window_as_array(
            "s3://example-bucket/model/land_cover/60N_110W__lc_ipcc.tif",
            "Byte",
            [-110, 50, -109, 51],
            4000,
            logging.getLogger(__name__),
            required=True,
        )


def test_raster_upload_failure_is_not_reported_as_success(monkeypatch, tmp_path):
    class _FailingS3Client:
        def upload_file(self, *args, **kwargs):
            raise RuntimeError("simulated upload failure")

    local_raster = tmp_path / "output.tif"
    local_raster.write_bytes(b"not-a-real-raster")
    monkeypatch.setattr(uu.boto3, "client", lambda *args, **kwargs: _FailingS3Client())

    with pytest.raises(RuntimeError, match="simulated upload failure"):
        uu.upload_raster_to_s3(
            str(local_raster),
            "example-bucket",
            "outputs/output.tif",
        )
    assert local_raster.exists()
