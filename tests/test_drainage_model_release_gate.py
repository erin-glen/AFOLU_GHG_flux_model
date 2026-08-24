from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from src.scripts.utilities import constants_and_names as cn


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/scripts/core_model/0_drainage_emissions_model.py"
)
SPEC = spec_from_file_location("drainage_model_release_gate", MODULE_PATH)
drainage_model = module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(drainage_model)


def test_all_five_year_periods_are_exactly_canonical():
    intervals, start, end, interval_type = drainage_model.compute_intervals(
        2015,
        2020,
        cn.intervals_annual,
        True,
    )

    assert intervals == cn.five_year_inventory_periods
    assert (start, end, interval_type) == (2001, 2024, cn.intervals_five_year)


def test_five_year_subset_preserves_canonical_boundaries():
    intervals, start, end, interval_type = drainage_model.compute_intervals(
        2006,
        2020,
        cn.intervals_five_year,
        False,
    )

    assert intervals == [(2006, 2010), (2011, 2015), (2016, 2020)]
    assert (start, end, interval_type) == (2006, 2020, cn.intervals_five_year)


def test_default_five_year_run_selects_latest_complete_period():
    intervals, start, end, _ = drainage_model.compute_intervals(
        None,
        None,
        cn.intervals_five_year,
        False,
    )

    assert intervals == [(2021, 2024)]
    assert (start, end) == (2021, 2024)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (2015, 2020),
        (2019, 2024),
        (2001, 2019),
        (2020, 2024),
    ],
)
def test_partial_or_shifted_five_year_periods_are_rejected(start, end):
    with pytest.raises(ValueError, match="canonical inventory-period boundary"):
        drainage_model.compute_intervals(
            start,
            end,
            cn.intervals_five_year,
            False,
        )


def test_sparse_historical_annual_periods_are_rejected():
    with pytest.raises(ValueError, match="lack complete global land-cover coverage"):
        drainage_model.compute_intervals(
            2015,
            2024,
            cn.intervals_annual,
            False,
        )


def test_no_upload_rejects_every_megazarr_write_mode():
    for options in (
        {"create_zarr": True, "update_existing_zarr": False, "mega_zarr_path": None},
        {"create_zarr": False, "update_existing_zarr": True, "mega_zarr_path": None},
        {
            "create_zarr": False,
            "update_existing_zarr": False,
            "mega_zarr_path": "s3://example/mega.zarr",
        },
    ):
        with pytest.raises(ValueError, match="--no_upload forbids all remote writes"):
            drainage_model.normalize_zarr_write_options(no_upload=True, **options)


def test_run_size_does_not_implicitly_enable_megazarr_writes():
    assert (
        drainage_model.normalize_zarr_write_options(
            no_upload=False,
            create_zarr=False,
            update_existing_zarr=False,
            mega_zarr_path=None,
        )
        is False
    )
