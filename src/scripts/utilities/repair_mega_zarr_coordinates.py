"""Repair drainage mega-zarr x/y coordinates without rewriting model data.

The command is read-only unless ``--apply`` is passed. Applied repairs require
an unused backup path. Only the ``x`` and ``y`` array subtrees are replaced,
then the root consolidated metadata is refreshed. All non-coordinate array
metadata is snapshotted and checked after the repair.

Example dry run::

    python -m src.scripts.utilities.repair_mega_zarr_coordinates \
      --zarr-path s3://bucket/path/mega.zarr

Example applied repair::

    python -m src.scripts.utilities.repair_mega_zarr_coordinates \
      --zarr-path s3://bucket/path/mega.zarr \
      --backup-path s3://bucket/path/mega_coordinate_backup_YYYYMMDD \
      --apply
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import posixpath
from typing import Any

import fsspec
import numpy as np
import xarray as xr
import zarr
from zarr.storage import FsspecStore

from src.scripts.utilities import drainage_zarr_utilities as dzu


LOGGER = logging.getLogger(__name__)
COORDINATE_NAMES = ("x", "y")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _json_safe_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _json_safe(value: Any) -> Any:
    """Recursively normalize metadata values for strict JSON manifests."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return _json_safe_scalar(value)


def _array_metadata_snapshot(group: zarr.Group) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for name in sorted(group.array_keys()):
        array = group[name]
        snapshot[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "chunks": list(array.chunks),
            "dimension_names": list(array.metadata.dimension_names or ()),
            "attributes": dict(array.attrs),
            "fill_value": _json_safe_scalar(array.metadata.fill_value),
            "codecs": [str(codec) for codec in array.metadata.codecs],
        }
    return snapshot


def _coordinate_layouts(group: zarr.Group) -> dict[str, dict[str, Any]]:
    layouts: dict[str, dict[str, Any]] = {}
    for name in COORDINATE_NAMES:
        if name not in group:
            raise ValueError(f"Mega-zarr is missing required coordinate array: {name}")
        array = group[name]
        # ``_FillValue`` is CF serialization metadata, not a meaningful
        # property of an index coordinate.  Carrying it into a Zarr v3
        # coordinate makes some xarray/zarr combinations attempt to decode a
        # Zarr ``Float64`` data-type object as an enum (``.value``), which
        # prevents the entire dataset from opening.
        attributes = dict(array.attrs)
        attributes.pop("_FillValue", None)
        layouts[name] = {
            "chunks": tuple(array.chunks),
            "filters": array.filters,
            "compressors": array.compressors,
            "serializer": array.serializer,
            "attributes": attributes,
            "chunk_key_encoding": array.metadata.chunk_key_encoding,
        }
    return layouts


def _validate_source_shape(group: zarr.Group) -> None:
    expected_height, expected_width = dzu.global_grid_shape()
    actual_x = tuple(group["x"].shape)
    actual_y = tuple(group["y"].shape)
    if actual_x != (expected_width,) or actual_y != (expected_height,):
        raise ValueError(
            "Refusing coordinate repair because the source grid shape is not canonical: "
            f"x={actual_x}, y={actual_y}, expected_x={(expected_width,)}, "
            f"expected_y={(expected_height,)}"
        )
    for name in group.array_keys():
        if name in (*COORDINATE_NAMES, "year"):
            continue
        array = group[name]
        dimension_names = tuple(array.metadata.dimension_names or ())
        if "x" not in dimension_names or "y" not in dimension_names:
            continue
        x_axis = dimension_names.index("x")
        y_axis = dimension_names.index("y")
        if array.shape[x_axis] != expected_width or array.shape[y_axis] != expected_height:
            raise ValueError(
                f"Refusing coordinate repair because data array {name!r} has grid shape "
                f"{array.shape}, inconsistent with the canonical global grid"
            )


def _coordinate_object_records(fs, root: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in COORDINATE_NAMES:
        axis_root = posixpath.join(root.rstrip("/"), name)
        details = fs.find(axis_root, detail=True)
        if not details:
            raise FileNotFoundError(f"No stored objects found for coordinate array {name}: {axis_root}")
        for path, info in sorted(details.items()):
            records.append(
                {
                    "path": path,
                    "relative_path": posixpath.relpath(path, root),
                    "size": info.get("size"),
                    "etag": info.get("ETag") or info.get("etag"),
                    "last_modified": info.get("LastModified") or info.get("last_modified"),
                }
            )
    return records


def _write_json(fs, path: str, payload: dict[str, Any]) -> None:
    parent = posixpath.dirname(path)
    if parent:
        fs.makedirs(parent, exist_ok=True)
    body = json.dumps(
        _json_safe(payload),
        indent=2,
        sort_keys=True,
        default=_json_default,
        allow_nan=False,
    ).encode("utf-8")
    with fs.open(path, "wb") as file_obj:
        file_obj.write(body)


def _copy_object(fs, source: str, destination: str) -> None:
    parent = posixpath.dirname(destination)
    if parent:
        fs.makedirs(parent, exist_ok=True)
    fs.copy(source, destination)


def _backup_coordinates(
    *,
    fs,
    source_root: str,
    backup_root: str,
    source_uri: str,
    backup_uri: str,
    metadata_before: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if fs.exists(backup_root):
        raise FileExistsError(f"Backup path already exists; refusing to overwrite it: {backup_uri}")

    records = _coordinate_object_records(fs, source_root)
    for record in records:
        destination = posixpath.join(backup_root, record["relative_path"])
        _copy_object(fs, record["path"], destination)

    root_metadata = posixpath.join(source_root, "zarr.json")
    if fs.exists(root_metadata):
        _copy_object(fs, root_metadata, posixpath.join(backup_root, "root", "zarr.json"))

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_zarr": source_uri,
        "backup_path": backup_uri,
        "coordinate_objects": records,
        "array_metadata_before": metadata_before,
    }
    _write_json(fs, posixpath.join(backup_root, "backup_manifest.json"), manifest)
    return manifest


def _restore_coordinates(*, fs, source_root: str, backup_root: str) -> None:
    for name in COORDINATE_NAMES:
        source_axis_root = posixpath.join(source_root, name)
        backup_axis_root = posixpath.join(backup_root, name)
        if fs.exists(source_axis_root):
            fs.rm(source_axis_root, recursive=True)
        backup_objects = fs.find(backup_axis_root)
        if not backup_objects:
            raise FileNotFoundError(f"Backup coordinate objects are missing: {backup_axis_root}")
        for backup_object in backup_objects:
            relative = posixpath.relpath(backup_object, backup_root)
            _copy_object(fs, backup_object, posixpath.join(source_root, relative))
    backup_root_metadata = posixpath.join(backup_root, "root", "zarr.json")
    if fs.exists(backup_root_metadata):
        _copy_object(fs, backup_root_metadata, posixpath.join(source_root, "zarr.json"))
    fs.invalidate_cache(source_root)


def _create_coordinate_array(
    group: zarr.Group,
    name: str,
    values: np.ndarray,
    layout: dict[str, Any],
) -> None:
    group.create_array(
        name,
        data=values,
        chunks=layout["chunks"],
        filters=layout["filters"],
        compressors=layout["compressors"],
        serializer=layout["serializer"],
        # Coordinates have no missing cells, so do not explicitly encode NaN
        # as their storage fill value.
        fill_value=None,
        attributes=layout["attributes"],
        chunk_key_encoding=layout["chunk_key_encoding"],
        dimension_names=(name,),
    )


@contextmanager
def _fresh_read_store(zarr_path: str):
    if not zarr_path.startswith("s3://"):
        yield zarr_path
        return
    fs = fsspec.filesystem(
        "s3",
        anon=False,
        asynchronous=True,
        skip_instance_cache=True,
    )
    store = FsspecStore(
        fs,
        read_only=True,
        path=zarr_path.removeprefix("s3://").rstrip("/"),
    )
    try:
        yield store
    finally:
        s3_session = getattr(fs, "_s3", None)
        if s3_session is not None:
            fs.close_session(fs.loop, s3_session)


def _create_and_validate_candidate(
    candidate_path: str,
    expected_x: np.ndarray,
    expected_y: np.ndarray,
    layouts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_group = zarr.open_group(
        store=dzu.make_zarr_store(candidate_path),
        mode="w",
        zarr_format=3,
    )
    _create_coordinate_array(candidate_group, "x", expected_x, layouts["x"])
    _create_coordinate_array(candidate_group, "y", expected_y, layouts["y"])

    with _fresh_read_store(candidate_path) as fresh_store:
        fresh_group = zarr.open_group(store=fresh_store, mode="r")
        actual_x = fresh_group["x"][:]
        actual_y = fresh_group["y"][:]
        x_object_chunks = int(np.prod(fresh_group["x"].cdata_shape))
        y_object_chunks = int(np.prod(fresh_group["y"].cdata_shape))
    if actual_x.dtype != np.dtype("float64") or not np.array_equal(actual_x, expected_x):
        raise ValueError("Fresh candidate x coordinate failed independent validation")
    if actual_y.dtype != np.dtype("float64") or not np.array_equal(actual_y, expected_y):
        raise ValueError("Fresh candidate y coordinate failed independent validation")
    return {
        "path": candidate_path,
        "x_object_chunks": x_object_chunks,
        "y_object_chunks": y_object_chunks,
        "validated_with_uncached_store": True,
    }


def _install_candidate_coordinates(
    *,
    fs,
    source_root: str,
    candidate_root: str,
) -> None:
    for name in COORDINATE_NAMES:
        source_axis_root = posixpath.join(source_root, name)
        candidate_axis_root = posixpath.join(candidate_root, name)
        candidate_objects = fs.find(candidate_axis_root)
        if not candidate_objects:
            raise FileNotFoundError(f"Candidate coordinate objects are missing: {candidate_axis_root}")
        if fs.exists(source_axis_root):
            fs.rm(source_axis_root, recursive=True)
        for candidate_object in candidate_objects:
            relative = posixpath.relpath(candidate_object, candidate_root)
            _copy_object(fs, candidate_object, posixpath.join(source_root, relative))
        fs.invalidate_cache(source_axis_root)


def _validate_repair(
    zarr_path: str,
    expected_x: np.ndarray,
    expected_y: np.ndarray,
    non_coordinate_metadata_before: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    coordinate_reads: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, consolidated in (("unconsolidated", False), ("auto", None)):
        with _fresh_read_store(zarr_path) as fresh_store:
            dataset = xr.open_zarr(
                fresh_store,
                consolidated=consolidated,
            )
            try:
                coordinate_reads[label] = (dataset["x"].values, dataset["y"].values)
            finally:
                dataset.close()

    actual_x, actual_y = coordinate_reads["unconsolidated"]
    auto_x, auto_y = coordinate_reads["auto"]
    if not np.array_equal(auto_x, actual_x) or not np.array_equal(auto_y, actual_y):
        raise ValueError("Consolidated and unconsolidated coordinate reads do not match")

    for name, actual, expected in (
        ("x", actual_x, expected_x),
        ("y", actual_y, expected_y),
    ):
        if actual.dtype != np.dtype("float64"):
            raise ValueError(f"Repaired {name} coordinate has dtype {actual.dtype}, expected float64")
        if not np.array_equal(actual, expected):
            mismatch_count = int(np.count_nonzero(actual != expected))
            raise ValueError(
                f"Repaired {name} coordinate does not match the canonical grid; "
                f"mismatched values={mismatch_count}"
            )

    with _fresh_read_store(zarr_path) as fresh_store:
        group = zarr.open_group(store=fresh_store, mode="r")
        for name in COORDINATE_NAMES:
            array = group[name]
            if "_FillValue" in array.attrs:
                raise ValueError(
                    f"Repaired {name} coordinate still has a _FillValue attribute; "
                    "xarray may be unable to open the Zarr v3 store"
                )

    with _fresh_read_store(zarr_path) as fresh_store:
        group = zarr.open_group(store=fresh_store, mode="r")
        metadata_after = _array_metadata_snapshot(group)
    non_coordinate_metadata_after = {
        name: value for name, value in metadata_after.items() if name not in COORDINATE_NAMES
    }
    if non_coordinate_metadata_after != non_coordinate_metadata_before:
        raise ValueError("Non-coordinate array metadata changed during coordinate repair")

    return {
        "x_dtype": str(actual_x.dtype),
        "y_dtype": str(actual_y.dtype),
        "x_size": int(actual_x.size),
        "y_size": int(actual_y.size),
        "x_first": float(actual_x[0]),
        "x_last": float(actual_x[-1]),
        "y_first": float(actual_y[0]),
        "y_last": float(actual_y[-1]),
        "x_first_step": float(actual_x[1] - actual_x[0]),
        "y_first_step": float(actual_y[0] - actual_y[1]),
        "non_coordinate_array_count": len(non_coordinate_metadata_after),
        "non_coordinate_metadata_unchanged": True,
        "consolidated_and_unconsolidated_reads_match": True,
    }


def inspect_or_repair(zarr_path: str, *, apply: bool, backup_path: str | None) -> dict[str, Any]:
    store = dzu.make_zarr_store(zarr_path, read_only=not apply)
    group = zarr.open_group(store=store, mode="r+" if apply else "r")
    _validate_source_shape(group)
    metadata_before = _array_metadata_snapshot(group)
    layouts = _coordinate_layouts(group)
    expected_x, expected_y, resolution = dzu.global_coords()
    report: dict[str, Any] = {
        "zarr_path": zarr_path,
        "mode": "apply" if apply else "dry-run",
        "canonical_resolution": resolution,
        "current_coordinates": {
            name: metadata_before[name] for name in COORDINATE_NAMES
        },
        "expected_coordinates": {
            "dtype": "float64",
            "x_shape": list(expected_x.shape),
            "y_shape": list(expected_y.shape),
            "x_first_step": float(expected_x[1] - expected_x[0]),
            "y_first_step": float(expected_y[0] - expected_y[1]),
        },
    }
    if not apply:
        return report
    if not backup_path:
        raise ValueError("--backup-path is required with --apply")

    fs, source_root = fsspec.core.url_to_fs(zarr_path)
    backup_fs, backup_root = fsspec.core.url_to_fs(backup_path)
    source_protocol = fsspec.utils.get_protocol(zarr_path)
    backup_protocol = fsspec.utils.get_protocol(backup_path)
    if source_protocol != backup_protocol:
        raise ValueError("Source and backup must use the same filesystem protocol")
    if type(fs) is not type(backup_fs):
        raise ValueError("Source and backup resolved to different filesystem implementations")
    if backup_root.rstrip("/").startswith(source_root.rstrip("/") + "/"):
        raise ValueError("Backup path must not be located inside the source zarr")

    non_coordinate_metadata_before = {
        name: value for name, value in metadata_before.items() if name not in COORDINATE_NAMES
    }
    backup_manifest = _backup_coordinates(
        fs=fs,
        source_root=source_root,
        backup_root=backup_root,
        source_uri=zarr_path,
        backup_uri=backup_path,
        metadata_before=metadata_before,
    )
    report["backup"] = {
        "path": backup_path,
        "coordinate_object_count": len(backup_manifest["coordinate_objects"]),
    }

    try:
        candidate_path = backup_path.rstrip("/") + "/corrected_coordinates.zarr"
        candidate_root = posixpath.join(backup_root, "corrected_coordinates.zarr")
        report["candidate"] = _create_and_validate_candidate(
            candidate_path,
            expected_x,
            expected_y,
            layouts,
        )
        _install_candidate_coordinates(
            fs=fs,
            source_root=source_root,
            candidate_root=candidate_root,
        )
        zarr.consolidate_metadata(store=dzu.make_zarr_store(zarr_path))
        fs.invalidate_cache(posixpath.join(source_root, "zarr.json"))
        report["consolidated_metadata_refreshed"] = True
        report["validation"] = _validate_repair(
            zarr_path,
            expected_x,
            expected_y,
            non_coordinate_metadata_before,
        )
    except Exception as exc:
        LOGGER.exception("Coordinate repair failed; restoring x/y from backup")
        _restore_coordinates(fs=fs, source_root=source_root, backup_root=backup_root)
        _write_json(
            fs,
            posixpath.join(backup_root, "repair_failure_rolled_back.json"),
            {
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_zarr": zarr_path,
                "backup_path": backup_path,
                "error": repr(exc),
                "rollback_completed": True,
            },
        )
        raise

    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(fs, posixpath.join(backup_root, "repair_result.json"), report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect or repair float32 coordinates in a drainage mega-zarr"
    )
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument(
        "--backup-path",
        help="Unused backup prefix for the original x/y objects; required with --apply",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Back up and replace x/y coordinates. Without this flag the command is read-only.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    report = inspect_or_repair(
        args.zarr_path,
        apply=args.apply,
        backup_path=args.backup_path,
    )
    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
