from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.scripts.zonal_statistics import discover_wdpa_affected_tiles as discovery


def _array(values: list[list[int]], *, x=None, y=None, dtype="uint8") -> xr.DataArray:
    data = np.asarray(values, dtype=dtype)
    return xr.DataArray(
        da.from_array(data, chunks=(1, 1)),
        dims=("y", "x"),
        coords={
            "x": np.asarray(x if x is not None else [0.5, 1.5], dtype="float64"),
            "y": np.asarray(y if y is not None else [-0.5, -1.5], dtype="float64"),
        },
    )


def test_normalize_tile_ids_defaults_and_validates_explicit_ids() -> None:
    canonical = ["00N_010E", "00N_000E"]
    assert discovery.normalize_tile_ids(None, canonical) == ["00N_000E", "00N_010E"]
    assert discovery.normalize_tile_ids(["00N_010E,00N_000E"], canonical) == canonical[::-1]
    with pytest.raises(ValueError, match="not in the canonical model roster"):
        discovery.normalize_tile_ids(["99N_999E"], canonical)


def test_candidate_and_land_histograms_find_only_adm0_intersection() -> None:
    wdpa = _array([[12, 13], [14, 0]])
    adm0 = _array([[1, 0], [2, 3]], dtype="uint32")

    candidate_lazy = discovery.build_candidate_histograms(wdpa, ["00N_000E"])
    land_lazy, missing_lazy = discovery.build_land_histograms(
        wdpa,
        adm0,
        ["00N_000E"],
    )
    candidate, land, missing = dask.compute(candidate_lazy, land_lazy, missing_lazy)

    candidate_hist = candidate["00N_000E"]
    land_hist = land["00N_000E"]
    assert candidate_hist[12] == 1
    assert candidate_hist[13] == 1
    assert candidate_hist[14] == 1
    assert land_hist[12] == 1
    assert land_hist[13] == 0
    assert land_hist[14] == 1
    assert int(missing["00N_000E"]) == 0


def test_align_adm0_to_wdpa_accepts_subpixel_coordinate_shift() -> None:
    wdpa = _array([[12, 0], [0, 0]])
    adm0 = _array(
        [[1, 2], [3, 4]],
        x=[0.50001, 1.50001],
        y=[-0.50001, -1.50001],
        dtype="uint32",
    )
    aligned = discovery.align_adm0_to_wdpa(wdpa, adm0)
    assert np.array_equal(aligned.x.values, wdpa.x.values)
    assert np.array_equal(aligned.y.values, wdpa.y.values)
    assert np.array_equal(aligned.compute().values, np.array([[1, 2], [3, 4]]))


def test_build_scan_frame_reports_candidates_land_and_unexpected_values() -> None:
    candidate = np.zeros(256, dtype=np.int64)
    candidate[[0, 12, 13, 17]] = [100, 5, 2, 1]
    land = np.zeros(256, dtype=np.int64)
    land[[0, 12, 13]] = [50, 3, 0]

    frame = discovery.build_scan_frame(
        ["00N_000E"],
        {"00N_000E": candidate},
        {"00N_000E": land},
        affected_codes=[12, 13, 14, 15, 16],
        registered_codes=range(17),
    )
    row = frame.iloc[0]
    assert row["candidate_affected_pixels"] == 7
    assert row["affected_land_pixels"] == 3
    assert row["unexpected_wdpa_pixels"] == 1
    assert bool(row["is_candidate"])
    assert bool(row["is_affected_land"])


def test_write_outputs_emits_direct_zonal_tile_arguments(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "tile_id": "00N_000E",
                "candidate_affected_pixels": 5,
                "affected_land_pixels": 3,
                "is_candidate": True,
                "is_affected_land": True,
            },
            {
                "tile_id": "00N_010E",
                "candidate_affected_pixels": 2,
                "affected_land_pixels": 0,
                "is_candidate": True,
                "is_affected_land": False,
            },
        ]
    )
    output_dir = tmp_path / "manifest"
    paths = discovery.write_outputs(
        frame,
        output_dir=output_dir,
        metadata={"schema_version": 1},
    )

    assert paths["tile_ids"].read_text(encoding="utf-8") == "00N_000E\n"
    assert paths["tile_args"].read_text(encoding="utf-8") == (
        "--execution_mode tile --tile_ids 00N_000E --data_tile_filter auto\n"
    )
    affected = pd.read_csv(paths["affected_csv"])
    assert affected["tile_id"].tolist() == ["00N_000E"]
    assert paths["manifest"].is_file()


def test_compute_collections_handles_single_future_returned_for_dictionary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        submitted = None

        def compute(self, value):
            self.submitted = value
            return "single-dictionary-future"

        @staticmethod
        def gather(future):
            assert future == "single-dictionary-future"
            return {"tile": np.array([1, 2, 3])}

    progress_calls = []
    monkeypatch.setattr(discovery, "progress", progress_calls.append)
    client = FakeClient()
    result = discovery.compute_collections(
        {"tile": da.from_array(np.array([1, 2, 3]), chunks=3)},
        client=client,
        label="test",
    )

    assert list(client.submitted) == ["tile"]
    assert progress_calls == ["single-dictionary-future"]
    assert result["tile"].tolist() == [1, 2, 3]
