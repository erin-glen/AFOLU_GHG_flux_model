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
