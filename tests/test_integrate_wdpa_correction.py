from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np
import pyarrow as pa
import pytest

from src.scripts.zonal_statistics import integrate_wdpa_correction as integrate


def _table(wdpa, values):
    return pa.table(
        {
            "flux_type": pa.array(["area__ha"] * len(wdpa)),
            "gadm_adm0": pa.array(np.arange(1, len(wdpa) + 1), type=pa.uint32()),
            "wdpa": pa.array(wdpa, type=pa.uint8()),
            "value": pa.array(values, type=pa.float32()),
        }
    )


def test_filter_wdpa_selects_only_correction_codes():
    source = _table([0, 11, 12, 13, 16], [1, 2, 3, 4, 5])
    selected = integrate.filter_wdpa(source, minimum=12, maximum=16)
    assert selected.column("wdpa").to_pylist() == [12, 13, 16]
    assert selected.column("value").to_pylist() == [3.0, 4.0, 5.0]


def test_value_and_key_validation_rejects_duplicates():
    duplicate = pa.table(
        {
            "flux_type": pa.array(["area__ha", "area__ha"]),
            "gadm_adm0": pa.array([1, 1], type=pa.uint32()),
            "wdpa": pa.array([12, 12], type=pa.uint8()),
            "value": pa.array([1.0, 2.0], type=pa.float32()),
        }
    )
    with pytest.raises(RuntimeError, match="duplicate key"):
        integrate.validate_values_and_keys(duplicate, label="duplicate")


@pytest.mark.parametrize("value, message", [(np.nan, "non-finite"), (-1.0, "negative")])
def test_value_and_key_validation_rejects_invalid_values(value, message):
    invalid = _table([12], [value])
    with pytest.raises(RuntimeError, match=message):
        integrate.validate_values_and_keys(invalid, label="invalid")


def test_output_defaults_are_isolated_from_sources():
    assert integrate.OUTPUT_RUN_NAME_DEFAULT not in {
        integrate.BASELINE_RUN_NAME,
        integrate.CORRECTION_RUN_NAME,
    }
    assert integrate.OUTPUT_RUN_DATE_DEFAULT == "20260802"


def _manifest_pair():
    common = {
        "model_version": "1_0_1",
        "run_name": "run",
        "run_date": "date",
        "interval_type": "five_year",
        "selected_fluxes": ["area"],
        "selected_flux_type_labels": ["area__ha"],
        "selected_contextual_groupers": [
            "wdpa", "landmark", "primary_forest", "kba", "river_basins",
            "drivers_of_loss",
        ],
        "adm0_zarr_path": "adm0",
        "pixel_area_zarr_path": "area",
        "pixel_area_var_name": "area",
        "align_tolerance_fraction": 0.1,
    }
    shared_paths = {
        "wdpa": "wdpa",
        "landmark": "landmark",
        "primary_forest": "primary_forest",
        "kba": "kba",
        "drivers_of_loss": "drivers_of_loss",
    }
    extras = [f"baseline_extra_{index:03d}" for index in range(138)]
    baseline = {
        **deepcopy(common),
        "processed_tile_ids": [*integrate.FULL_TILE_IDS, *extras],
        "contextual_grouper_paths": {
            **shared_paths,
            "river_basins": integrate.BASELINE_RIVER_BASINS,
        },
    }
    correction = {
        **deepcopy(common),
        "processed_tile_ids": list(integrate.FULL_TILE_IDS),
        "contextual_grouper_paths": {
            **shared_paths,
            "river_basins": integrate.CORRECTION_RIVER_BASINS,
        },
    }
    return baseline, correction


def test_manifest_validation_requires_exact_affected_tile_set():
    baseline, correction = _manifest_pair()
    integrate.validate_manifest_pair(baseline, correction, interval="2001_2005")

    correction["processed_tile_ids"][-1] = "baseline_extra_000"
    with pytest.raises(RuntimeError, match="exact affected-tile manifest"):
        integrate.validate_manifest_pair(baseline, correction, interval="2001_2005")


class _RecordingFS:
    def __init__(self, *, corrupt_name=None):
        self.corrupt_name = corrupt_name
        self.objects = {}
        self.put_order = []

    def put(self, local_path, remote_uri):
        name = remote_uri.rsplit("/", 1)[-1]
        self.put_order.append(name)
        self.objects[remote_uri] = integrate.Path(local_path).read_bytes()

    def info(self, remote_uri):
        name = remote_uri.rsplit("/", 1)[-1]
        etag = hashlib.md5(self.objects[remote_uri]).hexdigest()
        if name == self.corrupt_name:
            etag = "0" * 32
        return {"ETag": f'"{etag}"'}


def _write_staged_interval(directory):
    payloads = {
        "part-0.parquet": b"parquet",
        "_zonal_stats_manifest.json": b"{}",
        "_COMPLETE.json": b'{"success": true}',
    }
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)
    return hashlib.md5(payloads["part-0.parquet"]).hexdigest()


def test_upload_publishes_completion_marker_last(tmp_path):
    expected_parquet_md5 = _write_staged_interval(tmp_path)
    fs = _RecordingFS()

    actual_etag = integrate.upload_interval_artifacts(
        fs,
        local_dir=tmp_path,
        remote_dir="s3://bucket/output",
        expected_parquet_md5=expected_parquet_md5,
    )

    assert actual_etag == expected_parquet_md5
    assert fs.put_order == [
        "_zonal_stats_manifest.json",
        "part-0.parquet",
        "_COMPLETE.json",
    ]


def test_upload_failure_never_publishes_completion_marker(tmp_path):
    expected_parquet_md5 = _write_staged_interval(tmp_path)
    fs = _RecordingFS(corrupt_name="part-0.parquet")

    with pytest.raises(RuntimeError, match="Uploaded artifact hash differs"):
        integrate.upload_interval_artifacts(
            fs,
            local_dir=tmp_path,
            remote_dir="s3://bucket/output",
            expected_parquet_md5=expected_parquet_md5,
        )

    assert "_COMPLETE.json" not in fs.put_order
