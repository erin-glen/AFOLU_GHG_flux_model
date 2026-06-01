import importlib

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import local_output_paths as lop


def test_default_local_output_root_windows() -> None:
    assert lop.default_local_output_root(platform_system="Windows") == "C:/tmp/afolu"


def test_default_local_output_root_wsl_mount() -> None:
    assert (
        lop.default_local_output_root(
            platform_system="Linux",
            path_exists=lambda path: path == "/mnt/c/tmp",
        )
        == "/mnt/c/tmp/afolu"
    )


def test_default_local_output_root_plain_posix() -> None:
    assert (
        lop.default_local_output_root(
            platform_system="Linux",
            path_exists=lambda path: False,
        )
        == "/tmp/afolu"
    )


def test_local_output_root_env_override(monkeypatch) -> None:
    monkeypatch.setenv("AFOLU_LOCAL_OUTPUT_ROOT", r"D:\afolu_outputs")
    assert lop.local_output_root() == "D:/afolu_outputs"


def test_publication_builders_default_under_local_root(monkeypatch) -> None:
    monkeypatch.setenv("AFOLU_LOCAL_OUTPUT_ROOT", "C:/tmp/afolu")
    for env_name in ("AFOLU_PUB_ASSETS_DIR", "AFOLU_PUB_FAO_DIR", "AFOLU_PUB_NGHGI_DIR"):
        monkeypatch.delenv(env_name, raising=False)

    from src.scripts.zonal_statistics.pub_scripts import pub_assets, pub_fao, pub_nghgi

    pub_assets = importlib.reload(pub_assets)
    pub_fao = importlib.reload(pub_fao)
    pub_nghgi = importlib.reload(pub_nghgi)

    assert pub_assets.build_output_dir("0_1_4", "run_a", "20260417") == (
        "C:/tmp/afolu/publications/assets/version_0_1_4/run_a/20260417"
    )
    assert pub_fao.build_output_dir("0_1_4", "run_a", "20260417") == (
        "C:/tmp/afolu/publications/fao/version_0_1_4/run_a/20260417"
    )
    assert pub_nghgi.build_output_dir("0_1_4", "run_a", "20260417") == (
        "C:/tmp/afolu/publications/nghgi/version_0_1_4/run_a/20260417"
    )


def test_publication_builders_honor_legacy_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("AFOLU_LOCAL_OUTPUT_ROOT", "C:/tmp/afolu")
    monkeypatch.setenv("AFOLU_PUB_ASSETS_DIR", "D:/legacy/pub_assets")
    monkeypatch.setenv("AFOLU_PUB_FAO_DIR", "D:/legacy/pub_fao")
    monkeypatch.setenv("AFOLU_PUB_NGHGI_DIR", "D:/legacy/pub_nghgi")

    from src.scripts.zonal_statistics.pub_scripts import pub_assets, pub_fao, pub_nghgi

    pub_assets = importlib.reload(pub_assets)
    pub_fao = importlib.reload(pub_fao)
    pub_nghgi = importlib.reload(pub_nghgi)

    assert pub_assets.build_output_dir("0_1_4", "run_a", "20260417") == (
        "D:/legacy/pub_assets/version_0_1_4/run_a/20260417"
    )
    assert pub_fao.build_output_dir("0_1_4", "run_a", "20260417") == (
        "D:/legacy/pub_fao/version_0_1_4/run_a/20260417"
    )
    assert pub_nghgi.build_output_dir("0_1_4", "run_a", "20260417") == (
        "D:/legacy/pub_nghgi/version_0_1_4/run_a/20260417"
    )


def test_comparison_output_builder_defaults_to_publications_comparisons(monkeypatch) -> None:
    monkeypatch.setenv("AFOLU_LOCAL_OUTPUT_ROOT", "C:/tmp/afolu")
    monkeypatch.delenv("AFOLU_PUB_ASSETS_DIR", raising=False)
    monkeypatch.delenv("AFOLU_PUB_COMPARE_DIR", raising=False)

    from src.scripts.zonal_statistics.pub_scripts import pub_compare_runs

    pub_compare_runs = importlib.reload(pub_compare_runs)
    out_dir = pub_compare_runs._comparison_out_dir({"run_b": object(), "run_a": object()})

    assert out_dir == "C:/tmp/afolu/publications/comparisons/run_a__run_b"


def test_zonal_and_probability_staging_defaults(monkeypatch) -> None:
    monkeypatch.setenv("AFOLU_LOCAL_OUTPUT_ROOT", "C:/tmp/afolu")

    organic_zonal = importlib.import_module("src.scripts.zonal_statistics.02_run_zonal_stats")
    probability = importlib.import_module("src.scripts.zonal_statistics.02b_run_probability_class_area_stats")

    assert organic_zonal.default_local_output("0_1_4", "run_a", "20260417") == (
        "C:/tmp/afolu/staging/zonal_stats/0_1_4/run_a/20260417"
    )
    assert probability.default_local_output("20251105", "20250925") == (
        "C:/tmp/afolu/staging/probability_area_stats/20251105/20250925"
    )


def test_corrected_pixel_area_zarr_defaults() -> None:
    build_zarr = importlib.import_module("src.scripts.zonal_statistics.01_build_zarr_caches")
    organic_zonal = importlib.import_module("src.scripts.zonal_statistics.02_run_zonal_stats")
    probability = importlib.import_module("src.scripts.zonal_statistics.02b_run_probability_class_area_stats")

    assert cn.pixel_area_zarr_label == "20260531_fillValue_removed"
    assert cn.pixel_area_zarr_var == "band_data"
    assert cn.pixel_area_zarr_path.endswith(
        "/contextual_layer_global_zarr/pixel_area/"
        "20260531_fillValue_removed/global_pixel_area_20260531.zarr"
    )
    assert build_zarr.PIXEL_AREA_ZARR == cn.pixel_area_zarr_path
    assert organic_zonal.PIXEL_AREA_ZARR == cn.pixel_area_zarr_path
    assert probability.PIXEL_AREA_ZARR == cn.pixel_area_zarr_path
    assert probability.pixel_area_zarr_path() == cn.pixel_area_zarr_path
