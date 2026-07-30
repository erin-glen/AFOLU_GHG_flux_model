import json
from pathlib import Path

import numpy as np
from botocore.exceptions import ClientError
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_bounds

from src.scripts.postprocessing.visualization import create_input_dataset_maps as maps


def _write_tif(
    path: Path,
    data: np.ndarray,
    *,
    bounds: tuple[float, float, float, float],
    crs: str = "EPSG:4326",
    nodata: float | int | None = None,
) -> Path:
    array = np.asarray(data)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError("Synthetic raster data must be 2D or band-first 3D")

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[2],
        height=array.shape[1],
        count=array.shape[0],
        dtype=array.dtype,
        crs=crs,
        transform=from_bounds(*bounds, array.shape[2], array.shape[1]),
        nodata=nodata,
    ) as dst:
        dst.write(array)
    return path


def _base_config(source: Path) -> dict:
    return {
        "target": {
            "crs": "EPSG:4326",
            "bounds": [0, 0, 4, 2],
            "width": 4,
            "height": 2,
        },
        "datasets": [
            {
                "name": "Example continuous layer",
                "source": str(source),
                "kind": "continuous",
                "resampling": "bilinear",
                "style": {"cmap": "viridis", "vmin": 0, "vmax": 10},
            }
        ],
    }


def test_tile_template_resolution_uses_intersecting_model_tiles() -> None:
    config = {
        "target": {
            "crs": "EPSG:4326",
            "bounds": [-60, -10, -49.5, 0],
            "width": 21,
            "height": 20,
        },
        "datasets": [
            {
                "name": "Tiled input",
                "source": "tiles/{tile_id}.tif",
                "kind": "continuous",
            }
        ],
    }
    grid = maps.validate_config(config)

    assert maps.tile_ids_for_target(grid) == ["00N_060W", "00N_050W"]
    assert maps.resolve_dataset_sources(config["datasets"][0], config, grid) == [
        "tiles/00N_060W.tif",
        "tiles/00N_050W.tif",
    ]


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"target": {"bounds": [4, 0, 0, 2], "width": 4, "height": 2}}, "ordered"),
        ({"target": {"bounds": [0, 0, 4, 2], "width": 0, "height": 2}}, "positive integers"),
    ],
)
def test_validate_config_rejects_invalid_target(update: dict, message: str, tmp_path: Path) -> None:
    config = _base_config(tmp_path / "unused.tif")
    config.update(update)

    with pytest.raises(ValueError, match=message):
        maps.validate_config(config)


def test_validate_config_rejects_ambiguous_and_unsafe_dataset_settings(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "unused.tif")
    config["datasets"][0].update(
        {
            "kind": "categorical",
            "resampling": "bilinear",
            "style": {"colors": {"1": "#ff0000"}},
        }
    )

    with pytest.raises(ValueError, match="categorical and cannot use"):
        maps.validate_config(config)

    config = _base_config(tmp_path / "unused.tif")
    config["datasets"].append(
        {
            **config["datasets"][0],
            "name": "Second name",
            "output_name": "example-continuous-layer",
        }
    )
    with pytest.raises(ValueError, match="Duplicate dataset output name"):
        maps.validate_config(config)


def test_align_dataset_mosaics_local_tiles_exactly(tmp_path: Path) -> None:
    left = _write_tif(
        tmp_path / "left.tif",
        np.array([[1, 2], [3, 4]], dtype="float32"),
        bounds=(0, 0, 2, 2),
        nodata=-9999,
    )
    right = _write_tif(
        tmp_path / "right.tif",
        np.array([[5, 6], [7, 8]], dtype="float32"),
        bounds=(2, 0, 4, 2),
        nodata=-9999,
    )
    outside = _write_tif(
        tmp_path / "outside.tif",
        np.full((2, 2), 99, dtype="float32"),
        bounds=(10, 10, 12, 12),
        nodata=-9999,
    )
    dataset = {
        "name": "Exact local mosaic",
        "source": [str(left), str(right), str(outside)],
        "kind": "continuous",
        "resampling": "nearest",
        "mosaic_method": "last",
        "style": {"vmin": 1, "vmax": 8},
    }
    grid = maps.build_target_grid(
        {"crs": "EPSG:4326", "bounds": [0, 0, 4, 2], "width": 4, "height": 2}
    )

    aligned = maps.align_dataset(
        dataset,
        [str(left), str(right), str(outside)],
        grid,
        cache_dir=None,
        aws_profile=None,
    )

    np.testing.assert_array_equal(
        aligned.data,
        np.array([[1, 2, 5, 6], [3, 4, 7, 8]], dtype="float32"),
    )
    assert [record["uri"] for record in aligned.source_records] == [str(left), str(right)]
    assert aligned.skipped_nonintersecting == [str(outside)]
    assert aligned.missing_sources == []


def test_source_nodata_can_be_kept_as_visible_class_zero(tmp_path: Path) -> None:
    source = _write_tif(
        tmp_path / "zero_class.tif",
        np.array([[0, 1]], dtype="uint8"),
        bounds=(0, 0, 2, 1),
        nodata=0,
    )
    grid = maps.build_target_grid(
        {"crs": "EPSG:4326", "bounds": [0, 0, 2, 1], "width": 2, "height": 1}
    )
    dataset = {
        "name": "Binary class",
        "source": str(source),
        "kind": "categorical",
        "resampling": "nearest",
        "style": {"colors": {"0": "#ffffff", "1": "#ff0000"}},
    }

    respected = maps.align_dataset(
        dataset, [str(source)], grid, cache_dir=None, aws_profile=None
    )
    assert np.isnan(respected.data[0, 0])

    dataset["respect_source_nodata"] = False
    visible = maps.align_dataset(
        dataset, [str(source)], grid, cache_dir=None, aws_profile=None
    )
    np.testing.assert_array_equal(visible.data, [[0, 1]])


def test_raw_masks_run_before_value_transforms_and_missing_errors_are_specific() -> None:
    transformed = maps._apply_value_rules(
        np.array([[-9999, 2]], dtype="float32"),
        {
            "mask_values": [-9999],
            "scale": 10,
            "value_transform": {"type": "threshold_gte", "value": 10},
        },
    )
    assert np.isnan(transformed[0, 0])
    assert transformed[0, 1] == 1

    missing = ClientError(
        {"Error": {"Code": "NoSuchKey"}, "ResponseMetadata": {"HTTPStatusCode": 404}},
        "GetObject",
    )
    denied = ClientError(
        {"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}},
        "GetObject",
    )
    assert maps._is_missing_source_error(missing)
    assert not maps._is_missing_source_error(denied)


def test_render_rgba_preserves_categorical_and_continuous_transparency() -> None:
    categorical = np.array([[0, 1], [2, np.nan]], dtype="float32")
    categorical_dataset = {
        "name": "Categories",
        "kind": "categorical",
        "style": {
            "colors": {"1": "#ff0000", "2": "#00ff00"},
            "unknown_color": "#0000ff80",
            "nodata_color": "#00000000",
        },
    }

    categorical_rgba, categorical_style = maps.render_rgba(
        categorical, categorical_dataset
    )

    np.testing.assert_array_equal(categorical_rgba[0, 1], [255, 0, 0, 255])
    np.testing.assert_array_equal(categorical_rgba[1, 0], [0, 255, 0, 255])
    np.testing.assert_array_equal(categorical_rgba[0, 0], [0, 0, 255, 128])
    np.testing.assert_array_equal(categorical_rgba[1, 1], [0, 0, 0, 0])
    assert categorical_style["category_pixel_counts"] == {"1": 1, "2": 1}

    continuous = np.array([[0, 1], [np.nan, 2]], dtype="float32")
    continuous_dataset = {
        "name": "Continuous",
        "kind": "continuous",
        "style": {
            "cmap": "viridis",
            "vmin": 0,
            "vmax": 2,
            "nodata_color": "#00000000",
        },
    }

    continuous_rgba, continuous_style = maps.render_rgba(
        continuous, continuous_dataset
    )

    assert continuous_rgba.shape == (2, 2, 4)
    np.testing.assert_array_equal(continuous_rgba[..., 3], [[255, 255], [0, 255]])
    assert continuous_style["vmin"] == 0
    assert continuous_style["vmax"] == 2


def test_run_local_end_to_end_writes_exact_outputs_and_guards_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    width, height = 7, 5
    bounds = (0, 0, 7, 5)
    continuous_data = np.arange(width * height, dtype="float32").reshape(height, width)
    continuous_data[0, 0] = -9999
    categorical_data = np.where(
        np.indices((height, width))[1] < 3,
        1,
        2,
    ).astype("uint8")
    categorical_data[-1, -1] = 255

    continuous_path = _write_tif(
        tmp_path / "continuous.tif",
        continuous_data,
        bounds=bounds,
        nodata=-9999,
    )
    categorical_path = _write_tif(
        tmp_path / "categorical.tif",
        categorical_data,
        bounds=bounds,
        nodata=255,
    )
    config = {
        "target": {
            "crs": "EPSG:4326",
            "bounds": list(bounds),
            "width": width,
            "height": height,
        },
        "write_aligned_geotiff": True,
        "datasets": [
            {
                "name": "Canopy height",
                "source": str(continuous_path),
                "kind": "continuous",
                "resampling": "bilinear",
                "style": {"cmap": "viridis", "vmin": 0, "vmax": 34},
            },
            {
                "name": "Land cover",
                "source": str(categorical_path),
                "kind": "categorical",
                "resampling": "nearest",
                "style": {
                    "colors": {"1": "#ff0000", "2": "#00ff00"},
                    "nodata_color": "#00000000",
                },
            },
        ],
    }
    config_path = tmp_path / "maps.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "outputs"

    def fail_if_s3_is_used(*_args, **_kwargs):
        raise AssertionError("Local-only test unexpectedly attempted to use S3")

    monkeypatch.setattr(maps, "_boto_session", fail_if_s3_is_used)
    monkeypatch.setattr(maps.s3fs, "S3FileSystem", fail_if_s3_is_used)

    result = maps.run(config_path, output_dir, no_cache=True)

    png_paths = [output_dir / "01_canopy_height.png", output_dir / "02_land_cover.png"]
    tif_paths = [
        output_dir / "aligned" / "01_canopy_height.tif",
        output_dir / "aligned" / "02_land_cover.tif",
    ]
    for png_path in png_paths:
        assert png_path.exists()
        with Image.open(png_path) as image:
            assert image.mode == "RGBA"
            assert image.size == (width, height)

    expected_transform = from_bounds(*bounds, width, height)
    for tif_path in tif_paths:
        with rasterio.open(tif_path) as src:
            assert (src.width, src.height) == (width, height)
            assert src.crs == rasterio.crs.CRS.from_epsg(4326)
            assert src.transform == expected_transform
            assert src.dtypes == ("float32",)

    with rasterio.open(tif_paths[0]) as src:
        aligned_continuous = src.read(1)
    with rasterio.open(tif_paths[1]) as src:
        aligned_categorical = src.read(1)
    assert np.isnan(aligned_continuous[0, 0])
    np.testing.assert_array_equal(aligned_continuous[1:, :], continuous_data[1:, :])
    assert np.isnan(aligned_categorical[-1, -1])
    assert set(np.unique(aligned_categorical[np.isfinite(aligned_categorical)])) == {1.0, 2.0}

    manifest_path = output_dir / maps.MANIFEST_FILENAME
    on_disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk_manifest == result
    assert on_disk_manifest["target"]["width"] == width
    assert on_disk_manifest["target"]["height"] == height
    assert [layer["name"] for layer in on_disk_manifest["layers"]] == [
        "Canopy height",
        "Land cover",
    ]
    for layer, png_path, tif_path in zip(
        on_disk_manifest["layers"], png_paths, tif_paths
    ):
        assert layer["png"] == str(png_path.resolve())
        assert layer["aligned_geotiff"] == str(tif_path.resolve())
        assert len(layer["png_sha256"]) == 64
        assert len(layer["aligned_geotiff_sha256"]) == 64

    original_manifest = manifest_path.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        maps.run(config_path, output_dir, no_cache=True)
    assert manifest_path.read_bytes() == original_manifest

    overwritten = maps.run(config_path, output_dir, no_cache=True, overwrite=True)
    assert len(overwritten["layers"]) == 2
