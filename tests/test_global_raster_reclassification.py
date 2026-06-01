import numpy as np

from src.scripts.postprocessing.visualization import create_global_raster as cgr
from src.scripts.zonal_statistics import zonal_constants as zc


def _first_code_with_prefix(mapping: dict[int, str], prefix: str) -> np.uint32:
    for code in mapping.values():
        if code.startswith(prefix):
            return np.uint32(int(code))
    raise AssertionError(f"No state code found with prefix {prefix}")


def test_reclassify_combined_state_four_publication_classes() -> None:
    drained_code = _first_code_with_prefix(zc.DRAINED_STATE_ID_TO_CODE, "11")
    undrained_code = _first_code_with_prefix(zc.DRAINED_STATE_ID_TO_CODE, "16")
    nonpeat_code = _first_code_with_prefix(zc.DRAINED_STATE_ID_TO_CODE, "0")
    burned_code = np.uint32(int(next(iter(zc.BURNED_STATE_ID_TO_CODE.values()))))

    drained = np.array(
        [[undrained_code, drained_code, undrained_code, drained_code, 0, nonpeat_code]],
        dtype=np.uint32,
    )
    burned = np.array(
        [[0, 0, burned_code, burned_code, 0, 0]],
        dtype=np.uint32,
    )
    packed = zc.pack_combined_state(drained, burned)
    packed[0, 4] = 0

    out = cgr._reclassify_combined_state(packed)

    expected = np.array(
        [[
            cgr.COMBINED_STATE_RECLASS_UNDRAINED,
            cgr.COMBINED_STATE_RECLASS_DRAINED_ONLY,
            cgr.COMBINED_STATE_RECLASS_BURNED_ONLY,
            cgr.COMBINED_STATE_RECLASS_DRAINED_BURNED,
            cgr.COMBINED_STATE_RECLASS_NODATA,
            cgr.COMBINED_STATE_RECLASS_NODATA,
        ]],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(out, expected)


def test_combined_state_reclass_output_uses_companion_dataset_folder() -> None:
    out_dir, out_name = cgr._combined_state_reclass_output(
        "s3://bucket/path/0_01deg_output_aggregation/combined_state/run/2021_2024/",
        "0_01deg_global__combined_state_2021_2024.tif",
    )

    assert out_dir == (
        "s3://bucket/path/0_01deg_output_aggregation/"
        "combined_state_reclassified/run/2021_2024"
    )
    assert out_name == "0_01deg_global__combined_state_reclassified_2021_2024.tif"


def test_combined_state_class_fractions_preserve_minority_drained_presence() -> None:
    drained_code = _first_code_with_prefix(zc.DRAINED_STATE_ID_TO_CODE, "11")

    drained = np.zeros((4, 4), dtype=np.uint32)
    burned = np.zeros((4, 4), dtype=np.uint32)
    drained[0, 0] = drained_code
    packed = zc.pack_combined_state(drained, burned)

    modal = cgr._mode_per_component(packed, native_deg=1.0, target_deg=4.0)
    modal_reclass = cgr._reclassify_combined_state(modal)
    fractions = cgr._aggregate_combined_state_class_fractions(
        packed,
        native_deg=1.0,
        target_deg=4.0,
    )
    presence_reclass = cgr._presence_reclass_from_class_fractions(fractions)

    assert int(modal_reclass[0, 0]) == int(cgr.COMBINED_STATE_RECLASS_NODATA)
    assert fractions.shape == (4, 1, 1)
    assert float(fractions[1, 0, 0]) == np.float32(1 / 16)
    assert int(presence_reclass[0, 0]) == int(cgr.COMBINED_STATE_RECLASS_DRAINED_ONLY)


def test_combined_state_class_fractions_preserve_mixed_states() -> None:
    drained_code = _first_code_with_prefix(zc.DRAINED_STATE_ID_TO_CODE, "11")
    undrained_code = _first_code_with_prefix(zc.DRAINED_STATE_ID_TO_CODE, "16")
    burned_code = np.uint32(int(next(iter(zc.BURNED_STATE_ID_TO_CODE.values()))))

    drained = np.full((4, 4), undrained_code, dtype=np.uint32)
    burned = np.zeros((4, 4), dtype=np.uint32)
    drained[0, 0] = drained_code
    burned[0, 1] = burned_code
    drained[0, 2] = drained_code
    burned[0, 2] = burned_code
    packed = zc.pack_combined_state(drained, burned)

    fractions = cgr._aggregate_combined_state_class_fractions(
        packed,
        native_deg=1.0,
        target_deg=4.0,
    )
    presence_reclass = cgr._presence_reclass_from_class_fractions(fractions)

    expected = np.array([13 / 16, 1 / 16, 1 / 16, 1 / 16], dtype=np.float32)
    np.testing.assert_allclose(fractions[:, 0, 0], expected)
    assert int(presence_reclass[0, 0]) == int(cgr.COMBINED_STATE_RECLASS_DRAINED_BURNED)


def test_combined_state_fraction_and_presence_outputs_use_companion_folders() -> None:
    base = "s3://bucket/path/0_01deg_output_aggregation/combined_state/run/2021_2024/"
    name = "0_01deg_global__combined_state_2021_2024.tif"

    frac_dir, frac_name = cgr._combined_state_class_fraction_output(base, name)
    presence_dir, presence_name = cgr._combined_state_presence_reclass_output(base, name)

    assert frac_dir == (
        "s3://bucket/path/0_01deg_output_aggregation/"
        "combined_state_class_fraction/run/2021_2024"
    )
    assert frac_name == "0_01deg_global__combined_state_class_fraction_2021_2024.tif"
    assert presence_dir == (
        "s3://bucket/path/0_01deg_output_aggregation/"
        "combined_state_presence_reclassified/run/2021_2024"
    )
    assert presence_name == "0_01deg_global__combined_state_presence_reclassified_2021_2024.tif"
