"""Render aligned, designer-ready maps from model raster inputs.

The script reads a JSON manifest containing one shared target grid and an
ordered list of raster layers. Every source is warped onto the exact same CRS,
bounds, width, height, and affine transform before it is rendered. Outputs are
borderless RGBA PNGs; aligned GeoTIFFs can also be retained for QA.

Sources may be local paths, exact S3 URIs, S3/local globs, or model 10-degree
tile templates containing ``{tile_id}``. All inputs are declared explicitly in
the JSON configuration, so this copy has no dependency on the AFOLU model repo.

Example
-------
python create_input_dataset_maps.py \
  --config input_dataset_maps.config.json \
  --output-dir outputs/example_run

This is intentionally a local Rasterio workflow. A five- or six-layer AOI
export does not need a Coiled cluster.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")

from matplotlib import image as mpl_image
from matplotlib.colors import LogNorm, Normalize, PowerNorm, to_rgba
import numpy as np
import rasterio
from botocore.exceptions import BotoCoreError, ClientError
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.transform import Affine, from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
import s3fs


LOGGER = logging.getLogger(__name__)
MANIFEST_FILENAME = "input_dataset_maps.manifest.json"
DEFAULT_WINDOWS_CACHE = Path("C:/tmp/afolu/input_dataset_map_cache")

SUPPORTED_RESAMPLING = {
    name: getattr(Resampling, name)
    for name in (
        "nearest",
        "bilinear",
        "cubic",
        "average",
        "mode",
        "max",
        "min",
        "med",
        "q1",
        "q3",
        "sum",
    )
    if hasattr(Resampling, name)
}

SUPPORTED_MOSAIC_METHODS = {"first", "last", "min", "max"}
SUPPORTED_STRETCHES = {"linear", "log", "sqrt", "asinh"}
SUPPORTED_VALUE_TRANSFORMS = {"none", "binary_nonzero", "threshold_gte"}


@dataclass(frozen=True)
class TargetGrid:
    crs: CRS
    bounds: tuple[float, float, float, float]
    width: int
    height: int
    transform: Affine


@dataclass(frozen=True)
class ResolvedSource:
    uri: str
    read_path: str
    cached: bool


@dataclass
class AlignedLayer:
    data: np.ndarray
    source_records: list[dict[str, Any]]
    missing_sources: list[str]
    skipped_nonintersecting: list[str]


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _as_float_tuple(values: Sequence[Any], *, field: str, length: int) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != length:
        raise ValueError(f"{field} must contain exactly {length} numbers")
    try:
        return tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain exactly {length} numbers") from exc


def build_target_grid(target: Mapping[str, Any]) -> TargetGrid:
    """Validate and build the single grid shared by every output layer."""

    if not isinstance(target, Mapping):
        raise ValueError("target must be a JSON object")
    bounds = _as_float_tuple(target.get("bounds", ()), field="target.bounds", length=4)
    west, south, east, north = bounds
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("target.bounds values must be finite")
    if not (west < east and south < north):
        raise ValueError("target.bounds must be ordered [west, south, east, north]")

    try:
        width = int(target["width"])
        height = int(target["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("target.width and target.height must be positive integers") from exc
    if width <= 0 or height <= 0:
        raise ValueError("target.width and target.height must be positive integers")

    try:
        crs = CRS.from_user_input(target.get("crs", "EPSG:4326"))
    except Exception as exc:  # Rasterio raises several CRS parsing exception types.
        raise ValueError(f"Invalid target.crs: {target.get('crs')!r}") from exc

    return TargetGrid(
        crs=crs,
        bounds=(west, south, east, north),
        width=width,
        height=height,
        transform=from_bounds(west, south, east, north, width, height),
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    if not slug:
        raise ValueError(f"Dataset name cannot be converted to a filename: {value!r}")
    return slug


def _source_values(dataset: Mapping[str, Any]) -> list[str]:
    raw = dataset.get("source")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = [str(item) for item in raw if str(item).strip()]
        if values:
            return values
    return []


def _finite_float(value: Any, *, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be a finite number")
    return converted


def validate_config(config: Mapping[str, Any]) -> TargetGrid:
    """Validate the manifest before any output files are written."""

    if not isinstance(config, Mapping):
        raise ValueError("Configuration root must be a JSON object")
    grid = build_target_grid(config.get("target", {}))
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty JSON array")

    names: set[str] = set()
    stems: set[str] = set()
    for index, dataset in enumerate(datasets, start=1):
        label = f"datasets[{index - 1}]"
        if not isinstance(dataset, Mapping):
            raise ValueError(f"{label} must be a JSON object")
        name = str(dataset.get("name", "")).strip()
        if not name:
            raise ValueError(f"{label}.name is required")
        if name in names:
            raise ValueError(f"Duplicate dataset name: {name!r}")
        names.add(name)

        sources = _source_values(dataset)
        if not sources:
            raise ValueError(f"Dataset {name!r} must define source")

        try:
            band = int(dataset.get("band", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Dataset {name!r} band must be a positive integer") from exc
        if band <= 0:
            raise ValueError(f"Dataset {name!r} band must be a positive integer")

        kind = str(dataset.get("kind", "continuous")).lower()
        if kind not in {"continuous", "categorical"}:
            raise ValueError(f"Dataset {name!r} kind must be continuous or categorical")
        resampling = str(dataset.get("resampling", "nearest")).lower()
        if resampling not in SUPPORTED_RESAMPLING:
            raise ValueError(
                f"Dataset {name!r} has unsupported resampling {resampling!r}; "
                f"choose from {sorted(SUPPORTED_RESAMPLING)}"
            )
        if kind == "categorical" and resampling not in {"nearest", "mode", "max", "min"}:
            raise ValueError(
                f"Dataset {name!r} is categorical and cannot use {resampling!r} resampling"
            )
        mosaic = str(dataset.get("mosaic_method", "last")).lower()
        if mosaic not in SUPPORTED_MOSAIC_METHODS:
            raise ValueError(
                f"Dataset {name!r} mosaic_method must be one of {sorted(SUPPORTED_MOSAIC_METHODS)}"
            )

        transform_cfg = dataset.get("value_transform", {"type": "none"})
        if isinstance(transform_cfg, str):
            transform_name = transform_cfg
        elif isinstance(transform_cfg, Mapping):
            transform_name = str(transform_cfg.get("type", "none"))
        else:
            raise ValueError(f"Dataset {name!r} value_transform must be a string or object")
        if transform_name not in SUPPORTED_VALUE_TRANSFORMS:
            raise ValueError(
                f"Dataset {name!r} has unsupported value_transform {transform_name!r}"
            )
        if transform_name == "threshold_gte":
            if not isinstance(transform_cfg, Mapping) or "value" not in transform_cfg:
                raise ValueError(
                    f"Dataset {name!r} threshold_gte value_transform requires value"
                )
            _finite_float(
                transform_cfg["value"],
                field=f"Dataset {name!r} threshold_gte value",
            )

        for field in ("source_nodata", "scale", "offset", "valid_min", "valid_max"):
            if dataset.get(field) is not None:
                _finite_float(dataset[field], field=f"Dataset {name!r} {field}")
        mask_values = dataset.get("mask_values", [])
        if not isinstance(mask_values, Sequence) or isinstance(mask_values, (str, bytes)):
            raise ValueError(f"Dataset {name!r} mask_values must be an array")
        for value in mask_values:
            _finite_float(value, field=f"Dataset {name!r} mask_values entry")
        if "respect_source_nodata" in dataset and not isinstance(
            dataset["respect_source_nodata"], bool
        ):
            raise ValueError(f"Dataset {name!r} respect_source_nodata must be true or false")

        style = dataset.get("style", {})
        if not isinstance(style, Mapping):
            raise ValueError(f"Dataset {name!r} style must be a JSON object")
        if kind == "continuous":
            stretch = str(style.get("stretch", "linear")).lower()
            if stretch not in SUPPORTED_STRETCHES:
                raise ValueError(
                    f"Dataset {name!r} style.stretch must be one of {sorted(SUPPORTED_STRETCHES)}"
                )
            try:
                matplotlib.colormaps.get_cmap(str(style.get("cmap", "viridis")))
            except ValueError as exc:
                raise ValueError(
                    f"Dataset {name!r} has unknown colormap {style.get('cmap')!r}"
                ) from exc
            percentiles = _as_float_tuple(
                style.get("percentiles", (2, 98)),
                field=f"Dataset {name!r} style.percentiles",
                length=2,
            )
            if not (0 <= percentiles[0] < percentiles[1] <= 100):
                raise ValueError(
                    f"Dataset {name!r} style.percentiles must be ordered within 0..100"
                )
            numeric_limits = {}
            for field in ("vmin", "vmax", "asinh_scale"):
                if style.get(field) is not None:
                    numeric_limits[field] = _finite_float(
                        style[field], field=f"Dataset {name!r} style.{field}"
                    )
            if (
                "vmin" in numeric_limits
                and "vmax" in numeric_limits
                and numeric_limits["vmin"] >= numeric_limits["vmax"]
            ):
                raise ValueError(f"Dataset {name!r} style requires vmin < vmax")
            if numeric_limits.get("asinh_scale", 1) <= 0:
                raise ValueError(f"Dataset {name!r} style.asinh_scale must be positive")
        else:
            colors = style.get("colors")
            if not isinstance(colors, Mapping) or not colors:
                raise ValueError(
                    f"Dataset {name!r} is categorical and requires style.colors"
                )
            for code, color in colors.items():
                try:
                    float(code)
                    to_rgba(color)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Dataset {name!r} has invalid category color {code!r}: {color!r}"
                    ) from exc
        for field in ("nodata_color", "unknown_color"):
            if style.get(field) is not None:
                try:
                    to_rgba(style[field])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Dataset {name!r} style.{field} is not a valid color"
                    ) from exc

        output_stem = _slugify(str(dataset.get("output_name", name)))
        if output_stem in stems:
            raise ValueError(f"Duplicate dataset output name: {output_stem!r}")
        stems.add(output_stem)

    return grid


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    validate_config(config)
    return config


def _bbox_intersects(
    left: Sequence[float], right: Sequence[float]
) -> bool:
    lw, ls, le, ln = left
    rw, rs, re_, rn = right
    return not (le <= rw or re_ <= lw or ln <= rs or rn <= ls)


def tile_ids_for_target(grid: TargetGrid) -> list[str]:
    """Return model 10-degree tile IDs intersecting the target grid."""

    geographic_bounds = transform_bounds(
        grid.crs,
        "EPSG:4326",
        *grid.bounds,
        densify_pts=21,
    )
    west, south, east, north = geographic_bounds
    if east - west >= 359.999 or west < -180.000001 or east > 180.000001:
        raise ValueError("Target bounds crossing the antimeridian are not supported")

    tile_ids: list[str] = []
    # The model tile ID stores the tile's north edge and west edge. For example,
    # 00N_060W spans lon -60..-50 and lat -10..0.
    for tile_north in range(-80, 91, 10):
        tile_south = tile_north - 10
        lat_suffix = "N" if tile_north >= 0 else "S"
        lat_label = f"{abs(tile_north):02d}{lat_suffix}"
        for tile_west in range(-180, 180, 10):
            tile_east = tile_west + 10
            if not _bbox_intersects(
                (west, south, east, north),
                (tile_west, tile_south, tile_east, tile_north),
            ):
                continue
            lon_suffix = "E" if tile_west >= 0 else "W"
            lon_label = f"{abs(tile_west):03d}{lon_suffix}"
            tile_ids.append(f"{lat_label}_{lon_label}")
    return tile_ids


def _has_magic(path: str) -> bool:
    return any(character in path for character in "*?[")


def _normalize_s3_match(path: str) -> str:
    return path if path.startswith("s3://") else "s3://" + path.lstrip("/")


def _expand_glob(path: str, aws_profile: str | None) -> list[str]:
    if not _has_magic(path):
        return [path]
    if path.startswith("s3://"):
        kwargs: dict[str, Any] = {"anon": False}
        if aws_profile:
            kwargs["profile"] = aws_profile
        fs = s3fs.S3FileSystem(**kwargs)
        return [_normalize_s3_match(match) for match in sorted(fs.glob(path[5:]))]
    return sorted(glob.glob(path, recursive=True))


def resolve_dataset_sources(
    dataset: Mapping[str, Any],
    grid: TargetGrid,
    *,
    aws_profile: str | None = None,
) -> list[str]:
    tile_ids = tile_ids_for_target(grid)
    raw_paths: list[str] = []
    for source in _source_values(dataset):
        if "{tile_id}" in source:
            raw_paths.extend(source.format(tile_id=tile_id) for tile_id in tile_ids)
        else:
            raw_paths.append(source)

    expanded: list[str] = []
    for raw_path in raw_paths:
        expanded.extend(_expand_glob(raw_path, aws_profile))
    unique_paths = list(dict.fromkeys(expanded))
    if not unique_paths:
        raise ValueError(f"Dataset {dataset['name']!r} did not resolve any source paths")
    return unique_paths


def _split_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _cache_path(cache_dir: Path, uri: str) -> Path:
    _bucket, key = _split_s3(uri)
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
    name = Path(key).name or "raster.tif"
    return cache_dir / f"{digest}_{name}"


def _boto_session(aws_profile: str | None):
    import boto3

    return boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()


def materialize_source(
    uri: str,
    *,
    cache_dir: Path | None,
    aws_profile: str | None,
) -> ResolvedSource:
    if not uri.startswith("s3://"):
        return ResolvedSource(uri=uri, read_path=uri, cached=False)
    if cache_dir is None:
        return ResolvedSource(
            uri=uri,
            read_path="/vsis3/" + uri[len("s3://") :],
            cached=False,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = _cache_path(cache_dir, uri)
    if not local_path.exists() or local_path.stat().st_size == 0:
        bucket, key = _split_s3(uri)
        temp_path = local_path.with_suffix(local_path.suffix + ".part")
        temp_path.unlink(missing_ok=True)
        LOGGER.info("Caching %s -> %s", uri, local_path)
        try:
            _boto_session(aws_profile).client("s3").download_file(
                bucket,
                key,
                str(temp_path),
            )
            temp_path.replace(local_path)
        finally:
            temp_path.unlink(missing_ok=True)
    return ResolvedSource(uri=uri, read_path=str(local_path), cached=True)


def _is_missing_source_error(exc: BaseException) -> bool:
    """Distinguish missing objects/files from auth, network, and format failures."""

    if isinstance(exc, ClientError):
        response = exc.response or {}
        code = str(response.get("Error", {}).get("Code", "")).lower()
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in {"404", "nosuchkey", "notfound"} or status == 404
    if isinstance(exc, FileNotFoundError):
        return True
    if isinstance(exc, (RasterioIOError, OSError)):
        message = str(exc).lower()
        missing_markers = (
            "no such file",
            "not found",
            "does not exist",
            "http response code: 404",
            "status=404",
        )
        return any(marker in message for marker in missing_markers)
    return False


def _rasterio_env(aws_profile: str | None):
    options = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    }
    if not aws_profile:
        return rasterio.Env(**options)
    from rasterio.session import AWSSession

    return rasterio.Env(AWSSession(_boto_session(aws_profile)), **options)


def _bounds_in_target(src: rasterio.io.DatasetReader, grid: TargetGrid) -> tuple[float, ...]:
    if src.crs is None:
        raise ValueError(f"Source raster has no CRS: {src.name}")
    return transform_bounds(src.crs, grid.crs, *src.bounds, densify_pts=21)


def _combine(dest: np.ndarray, incoming: np.ndarray, method: str) -> None:
    incoming_valid = np.isfinite(incoming)
    if method == "last":
        dest[incoming_valid] = incoming[incoming_valid]
        return
    dest_valid = np.isfinite(dest)
    if method == "first":
        fill = incoming_valid & ~dest_valid
        dest[fill] = incoming[fill]
        return
    fill = incoming_valid & ~dest_valid
    dest[fill] = incoming[fill]
    overlap = incoming_valid & dest_valid
    if method == "min":
        dest[overlap] = np.minimum(dest[overlap], incoming[overlap])
    elif method == "max":
        dest[overlap] = np.maximum(dest[overlap], incoming[overlap])


def _apply_value_rules(data: np.ndarray, dataset: Mapping[str, Any]) -> np.ndarray:
    out = data.astype("float32", copy=True)
    # Validity rules describe aligned source values and must run before scale,
    # offset, or classification transforms alter sentinels and thresholds.
    for mask_value in dataset.get("mask_values", []):
        out[np.isclose(out, float(mask_value), equal_nan=False)] = np.nan
    if dataset.get("valid_min") is not None:
        out[out < float(dataset["valid_min"])] = np.nan
    if dataset.get("valid_max") is not None:
        out[out > float(dataset["valid_max"])] = np.nan

    valid = np.isfinite(out)
    scale = float(dataset.get("scale", 1.0))
    offset = float(dataset.get("offset", 0.0))
    out[valid] = out[valid] * scale + offset

    transform_cfg = dataset.get("value_transform", {"type": "none"})
    if isinstance(transform_cfg, str):
        transform_name = transform_cfg
        transform_cfg = {"type": transform_name}
    else:
        transform_name = str(transform_cfg.get("type", "none"))
    if transform_name == "binary_nonzero":
        out[valid] = (out[valid] != 0).astype("float32")
    elif transform_name == "threshold_gte":
        threshold = float(transform_cfg["value"])
        out[valid] = (out[valid] >= threshold).astype("float32")
    return out


def align_dataset(
    dataset: Mapping[str, Any],
    source_uris: Sequence[str],
    grid: TargetGrid,
    *,
    cache_dir: Path | None,
    aws_profile: str | None,
) -> AlignedLayer:
    """Warp and mosaic one dataset onto the canonical target grid."""

    dest = np.full((grid.height, grid.width), np.nan, dtype="float32")
    band = int(dataset.get("band", 1))
    resampling_name = str(dataset.get("resampling", "nearest")).lower()
    resampling = SUPPORTED_RESAMPLING[resampling_name]
    mosaic_method = str(dataset.get("mosaic_method", "last")).lower()
    allow_missing = bool(
        dataset.get(
            "allow_missing_sources",
            any("{tile_id}" in source for source in _source_values(dataset)),
        )
    )
    source_nodata_override = dataset.get("source_nodata")
    respect_source_nodata = bool(dataset.get("respect_source_nodata", True))

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    skipped: list[str] = []
    with _rasterio_env(aws_profile):
        for uri in source_uris:
            try:
                resolved = materialize_source(
                    uri,
                    cache_dir=cache_dir,
                    aws_profile=aws_profile,
                )
                try:
                    source_reader = rasterio.open(resolved.read_path)
                except RasterioIOError:
                    if not resolved.cached:
                        raise
                    # A non-empty cache entry can still be truncated. Remove it
                    # and make one clean download attempt before surfacing the
                    # source error.
                    LOGGER.warning("Refreshing unreadable cached raster: %s", resolved.read_path)
                    Path(resolved.read_path).unlink(missing_ok=True)
                    resolved = materialize_source(
                        uri,
                        cache_dir=cache_dir,
                        aws_profile=aws_profile,
                    )
                    source_reader = rasterio.open(resolved.read_path)
                with source_reader as src:
                    if band > src.count:
                        raise ValueError(
                            f"Dataset {dataset['name']!r} requests band {band}, but {uri} has {src.count} bands"
                        )
                    projected_bounds = _bounds_in_target(src, grid)
                    if not _bbox_intersects(projected_bounds, grid.bounds):
                        skipped.append(uri)
                        continue

                    vrt_kwargs: dict[str, Any] = {
                        "crs": grid.crs,
                        "transform": grid.transform,
                        "width": grid.width,
                        "height": grid.height,
                        "resampling": resampling,
                        "dtype": "float32",
                        "nodata": np.nan,
                    }
                    if source_nodata_override is not None:
                        vrt_kwargs["src_nodata"] = float(source_nodata_override)
                    elif not respect_source_nodata:
                        # NaN cannot collide with integer class 0 and remains a
                        # suitable source nodata marker for floating rasters.
                        vrt_kwargs["src_nodata"] = np.nan
                    with WarpedVRT(src, **vrt_kwargs) as vrt:
                        warped = vrt.read(band, masked=True, out_dtype="float32")
                    incoming = np.asarray(warped.filled(np.nan), dtype="float32")
                    _combine(dest, incoming, mosaic_method)
                    records.append(
                        {
                            "uri": uri,
                            "cached": resolved.cached,
                            "source_crs": src.crs.to_string() if src.crs else None,
                            "source_width": src.width,
                            "source_height": src.height,
                            "source_nodata": src.nodata,
                        }
                    )
            except (RasterioIOError, OSError, BotoCoreError, ClientError) as exc:
                if allow_missing and _is_missing_source_error(exc):
                    LOGGER.warning("Skipping missing source for %s: %s", dataset["name"], uri)
                    missing.append(uri)
                    continue
                raise RuntimeError(
                    f"Could not access source for dataset {dataset['name']!r}: {uri}. "
                    "The failure was not an ordinary missing tile."
                ) from exc

    dest = _apply_value_rules(dest, dataset)
    if not np.isfinite(dest).any() and not bool(dataset.get("allow_empty", False)):
        raise RuntimeError(
            f"Dataset {dataset['name']!r} has no valid pixels in the requested AOI"
        )
    return AlignedLayer(
        data=dest,
        source_records=records,
        missing_sources=missing,
        skipped_nonintersecting=skipped,
    )


def _color_bytes(color: Any) -> np.ndarray:
    rgba = to_rgba(color)
    return np.array([round(channel * 255) for channel in rgba], dtype="uint8")


def _continuous_limits(
    data: np.ndarray,
    style: Mapping[str, Any],
) -> tuple[float, float]:
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return 0.0, 1.0
    percentiles = _as_float_tuple(
        style.get("percentiles", (2, 98)),
        field="style.percentiles",
        length=2,
    )
    if not (0 <= percentiles[0] < percentiles[1] <= 100):
        raise ValueError("style.percentiles must be ordered within 0..100")
    vmin = float(style["vmin"]) if style.get("vmin") is not None else float(
        np.percentile(valid, percentiles[0])
    )
    vmax = float(style["vmax"]) if style.get("vmax") is not None else float(
        np.percentile(valid, percentiles[1])
    )
    if not (math.isfinite(vmin) and math.isfinite(vmax)):
        raise ValueError("Continuous style limits must be finite")
    if vmin == vmax:
        delta = max(abs(vmin) * 1e-6, 1e-6)
        vmin -= delta
        vmax += delta
    if vmin > vmax:
        raise ValueError("Continuous style requires vmin < vmax")
    return vmin, vmax


def render_rgba(
    data: np.ndarray,
    dataset: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert an aligned numeric array to an RGBA image of identical shape."""

    kind = str(dataset.get("kind", "continuous")).lower()
    style = dataset.get("style", {})
    mask = ~np.isfinite(data)
    if kind == "categorical":
        rgba = np.empty((*data.shape, 4), dtype="uint8")
        rgba[:] = _color_bytes(style.get("unknown_color", "#00000000"))
        counts: dict[str, int] = {}
        for raw_code, color in style["colors"].items():
            code = float(raw_code)
            category_mask = np.isfinite(data) & np.isclose(data, code)
            rgba[category_mask] = _color_bytes(color)
            counts[str(raw_code)] = int(category_mask.sum())
        style_record: dict[str, Any] = {
            "kind": "categorical",
            "category_pixel_counts": counts,
        }
    else:
        vmin, vmax = _continuous_limits(data, style)
        stretch = str(style.get("stretch", "linear")).lower()
        values = data.astype("float64", copy=True)
        render_mask = mask.copy()
        if stretch == "log":
            render_mask |= values <= 0
            if vmin <= 0:
                positive = values[(~render_mask) & np.isfinite(values)]
                if positive.size == 0:
                    raise ValueError(
                        f"Dataset {dataset['name']!r} log stretch has no positive values"
                    )
                vmin = float(positive.min())
            norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
        elif stretch == "sqrt":
            norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax, clip=True)
        elif stretch == "asinh":
            scale = float(style.get("asinh_scale", max(abs(vmin), abs(vmax)) / 10 or 1.0))
            if scale <= 0:
                raise ValueError("style.asinh_scale must be positive")
            values = np.arcsinh(values / scale)
            norm = Normalize(
                vmin=np.arcsinh(vmin / scale),
                vmax=np.arcsinh(vmax / scale),
                clip=True,
            )
        else:
            norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

        cmap_name = str(style.get("cmap", "viridis"))
        try:
            colormap = matplotlib.colormaps.get_cmap(cmap_name)
        except ValueError as exc:
            raise ValueError(
                f"Dataset {dataset['name']!r} has unknown colormap {cmap_name!r}"
            ) from exc
        rgba = np.asarray(colormap(norm(values), bytes=True), dtype="uint8")
        mask = render_mask
        style_record = {
            "kind": "continuous",
            "cmap": cmap_name,
            "stretch": stretch,
            "vmin": vmin,
            "vmax": vmax,
        }

    rgba[mask] = _color_bytes(style.get("nodata_color", "#00000000"))
    return rgba, style_record


def _atomic_save_png(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".png", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        mpl_image.imsave(temp_path, rgba, format="png")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_write_aligned_geotiff(path: Path, data: np.ndarray, grid: TargetGrid) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tif", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        profile: dict[str, Any] = {
            "driver": "GTiff",
            "width": grid.width,
            "height": grid.height,
            "count": 1,
            "dtype": "float32",
            "crs": grid.crs,
            "transform": grid.transform,
            "nodata": np.nan,
            "compress": "DEFLATE",
            "predictor": 3,
        }
        if grid.width >= 16 and grid.height >= 16:
            profile.update(
                tiled=True,
                blockxsize=max(16, min(512, grid.width) // 16 * 16),
                blockysize=max(16, min(512, grid.height) // 16 * 16),
            )
        with rasterio.open(temp_path, "w", **profile) as dst:
            dst.write(data.astype("float32", copy=False), 1)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_stats(data: np.ndarray) -> dict[str, Any]:
    valid = data[np.isfinite(data)]
    stats: dict[str, Any] = {
        "valid_pixels": int(valid.size),
        "nodata_pixels": int(data.size - valid.size),
    }
    if valid.size:
        stats.update(
            minimum=float(valid.min()),
            maximum=float(valid.max()),
            mean=float(valid.mean()),
            p02=float(np.percentile(valid, 2)),
            p50=float(np.percentile(valid, 50)),
            p98=float(np.percentile(valid, 98)),
        )
    return stats


def _output_paths(
    config: Mapping[str, Any],
    output_dir: Path,
) -> list[tuple[Path, Path]]:
    prefix_order = bool(config.get("prefix_output_order", True))
    paths = []
    for index, dataset in enumerate(config["datasets"], start=1):
        stem = _slugify(str(dataset.get("output_name", dataset["name"])))
        filename_stem = f"{index:02d}_{stem}" if prefix_order else stem
        paths.append(
            (
                output_dir / f"{filename_stem}.png",
                output_dir / "aligned" / f"{filename_stem}.tif",
            )
        )
    return paths


def _preflight_outputs(
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    write_aligned: bool,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    candidates = [output_dir / MANIFEST_FILENAME]
    for png_path, tif_path in _output_paths(config, output_dir):
        candidates.append(png_path)
        if write_aligned:
            candidates.append(tif_path)
    existing = [str(path) for path in candidates if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing map outputs. Use --overwrite to replace them:\n"
            + "\n".join(existing)
        )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".json", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=False)
            file.write("\n")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def run(
    config_path: Path,
    output_dir: Path,
    *,
    cache_dir: Path | None = None,
    no_cache: bool = False,
    write_aligned_override: bool | None = None,
    overwrite: bool = False,
    validate_only: bool = False,
    aws_profile_override: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    grid = build_target_grid(config["target"])
    aws_profile = aws_profile_override or config.get("aws_profile")
    if no_cache:
        resolved_cache_dir = None
    elif cache_dir is not None:
        resolved_cache_dir = cache_dir
    elif config.get("cache_dir"):
        resolved_cache_dir = Path(str(config["cache_dir"]))
    elif os.name == "nt":
        resolved_cache_dir = DEFAULT_WINDOWS_CACHE
    else:
        resolved_cache_dir = None

    write_aligned = (
        bool(config.get("write_aligned_geotiff", False))
        if write_aligned_override is None
        else write_aligned_override
    )
    resolved_by_dataset: list[list[str]] = []
    for dataset in config["datasets"]:
        sources = resolve_dataset_sources(
            dataset,
            grid,
            aws_profile=aws_profile,
        )
        resolved_by_dataset.append(sources)
        LOGGER.info("Resolved %d source(s) for %s", len(sources), dataset["name"])

    if validate_only:
        return {
            "validated": True,
            "target": _target_record(grid),
            "datasets": [
                {"name": dataset["name"], "resolved_sources": sources}
                for dataset, sources in zip(config["datasets"], resolved_by_dataset)
            ],
        }

    _preflight_outputs(
        config,
        output_dir,
        write_aligned=write_aligned,
        overwrite=overwrite,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_records: list[dict[str, Any]] = []
    for index, (dataset, sources, output_paths) in enumerate(
        zip(config["datasets"], resolved_by_dataset, _output_paths(config, output_dir)),
        start=1,
    ):
        png_path, tif_path = output_paths
        LOGGER.info("Rendering %d/%d: %s", index, len(config["datasets"]), dataset["name"])
        aligned = align_dataset(
            dataset,
            sources,
            grid,
            cache_dir=resolved_cache_dir,
            aws_profile=aws_profile,
        )
        rgba, style_record = render_rgba(aligned.data, dataset)
        if rgba.shape[:2] != (grid.height, grid.width):
            raise AssertionError(
                f"Rendered shape {rgba.shape[:2]} does not match target {(grid.height, grid.width)}"
            )
        _atomic_save_png(png_path, rgba)
        if write_aligned:
            _atomic_write_aligned_geotiff(tif_path, aligned.data, grid)

        record = {
            "order": index,
            "name": dataset["name"],
            "title": dataset.get("title", dataset["name"]),
            "png": str(png_path.resolve()),
            "png_sha256": _sha256(png_path),
            "width": grid.width,
            "height": grid.height,
            "kind": dataset.get("kind", "continuous"),
            "source_mode": "explicit_source",
            "resampling": dataset.get("resampling", "nearest"),
            "mosaic_method": dataset.get("mosaic_method", "last"),
            "value_transform": dataset.get("value_transform", {"type": "none"}),
            "respect_source_nodata": bool(dataset.get("respect_source_nodata", True)),
            "scale": float(dataset.get("scale", 1.0)),
            "offset": float(dataset.get("offset", 0.0)),
            "mask_values": dataset.get("mask_values", []),
            "valid_min": dataset.get("valid_min"),
            "valid_max": dataset.get("valid_max"),
            "style": style_record,
            "statistics": _array_stats(aligned.data),
            "sources": aligned.source_records,
            "missing_sources": aligned.missing_sources,
            "skipped_nonintersecting_sources": aligned.skipped_nonintersecting,
        }
        if write_aligned:
            record["aligned_geotiff"] = str(tif_path.resolve())
            record["aligned_geotiff_sha256"] = _sha256(tif_path)
        layer_records.append(record)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "description": config.get("description"),
        "output_directory": str(output_dir.resolve()),
        "target": _target_record(grid),
        "cache_directory": str(resolved_cache_dir.resolve()) if resolved_cache_dir else None,
        "raw_or_transformed": (
            "Each layer reads the raw source by default; value_transform is recorded per layer."
        ),
        "layers": layer_records,
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    _write_json_atomic(manifest_path, manifest)
    LOGGER.info("Wrote %d aligned maps and %s", len(layer_records), manifest_path)
    return manifest


def _target_record(grid: TargetGrid) -> dict[str, Any]:
    return {
        "crs": grid.crs.to_string(),
        "bounds": list(grid.bounds),
        "width": grid.width,
        "height": grid.height,
        "transform": list(grid.transform)[:6],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON layer manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache S3 GeoTIFFs locally (default on Windows: C:/tmp/afolu/input_dataset_map_cache)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Read S3 rasters directly through /vsis3 instead of using the cache",
    )
    parser.add_argument(
        "--write-aligned-geotiff",
        action="store_true",
        default=None,
        help="Retain one aligned float32 GeoTIFF per layer",
    )
    parser.add_argument("--aws-profile", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the manifest and resolve source names without reading rasters",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _configure_logging(args.verbose)
    result = run(
        args.config,
        args.output_dir,
        cache_dir=args.cache_dir,
        no_cache=args.no_cache,
        write_aligned_override=args.write_aligned_geotiff,
        overwrite=args.overwrite,
        validate_only=args.validate_only,
        aws_profile_override=args.aws_profile,
    )
    if args.validate_only:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
