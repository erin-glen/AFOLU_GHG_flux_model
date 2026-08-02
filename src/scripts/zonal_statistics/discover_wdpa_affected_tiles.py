# -*- coding: utf-8 -*-
"""Discover 10x10-degree tiles affected by the WDPA category-contract fix.

The failed production zonal run accepted WDPA codes 0-11 even though the
production WDPA Zarr contains valid codes 12-16.  This script scans that exact
Zarr, identifies canonical model tiles containing those codes, and then checks
which candidates intersect ``adm0 > 0``.  It writes an auditable CSV/JSON
manifest plus a comma-separated tile list that can be passed directly to
``02_run_zonal_stats.py --tile_ids``.

The script is safe by default: without ``--execute`` it only prints the plan.

Attach to an existing Coiled cluster from WSL::

    python -m src.scripts.zonal_statistics.discover_wdpa_affected_tiles \
      --execute \
      --cluster-name organic_soils_zonal_tests

Launch and automatically terminate a small on-demand cluster::

    python -m src.scripts.zonal_statistics.discover_wdpa_affected_tiles \
      --execute \
      --launch-cluster \
      --cluster-name wdpa_tile_discovery \
      --workers 25 \
      --memory-gib 32

For a cheap local smoke test, explicitly select one or more tiles::

    python -m src.scripts.zonal_statistics.discover_wdpa_affected_tiles \
      --execute --run-local --tile-ids 50N_000E
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import dask
import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from dask.diagnostics import ProgressBar
from distributed import Client, progress

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu


LOGGER = logging.getLogger("wdpa_affected_tile_discovery")
DEFAULT_AFFECTED_CODES = (12, 13, 14, 15, 16)
DEFAULT_ADM0_DATE = "20250925"
DEFAULT_ADM0_ZARR = (
    f"{cn.contextual_layer_global_zarr_root}/GADM4_1_adm0_global/"
    f"{DEFAULT_ADM0_DATE}/global_GADM41_adm0_{DEFAULT_ADM0_DATE}.zarr"
)
UINT8_HISTOGRAM_EDGES = np.arange(257, dtype=np.float64) - 0.5


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def normalize_tile_ids(
    raw_tile_args: Sequence[str] | None,
    canonical_tile_ids: Sequence[str],
) -> list[str]:
    """Normalize optional comma/space-separated tile arguments.

    With no explicit selection, the canonical model tile roster is returned.
    Explicit IDs must be members of that roster so discovery and zonal
    execution cannot silently use different spatial domains.
    """

    canonical = sorted({str(tile_id) for tile_id in canonical_tile_ids})
    if not raw_tile_args:
        return canonical

    requested: set[str] = set()
    for item in raw_tile_args:
        requested.update(part.strip() for part in str(item).split(",") if part.strip())
    unknown = sorted(requested - set(canonical))
    if unknown:
        raise ValueError(
            f"Tile IDs are not in the canonical model roster: {unknown}. "
            f"Canonical tile source: {cn.tile_id_list_source}"
        )
    return sorted(requested)


def _select_xy_data_array(
    dataset: xr.Dataset,
    *,
    preferred_variable: str | None,
    source_label: str,
) -> xr.DataArray:
    if preferred_variable is not None:
        if preferred_variable not in dataset.data_vars:
            raise KeyError(
                f"Variable '{preferred_variable}' not found in {source_label}; "
                f"available={list(dataset.data_vars)}"
            )
        array = dataset[preferred_variable]
    else:
        candidates = [
            dataset[name]
            for name in dataset.data_vars
            if {"x", "y"}.issubset(dataset[name].dims)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one x/y data variable in {source_label}; "
                f"available={list(dataset.data_vars)}"
            )
        array = candidates[0]

    for dim in tuple(array.dims):
        if dim in {"x", "y"}:
            continue
        if array.sizes[dim] != 1:
            raise ValueError(
                f"Refusing to collapse non-singleton dimension '{dim}' in "
                f"{source_label}: size={array.sizes[dim]}"
            )
        array = array.isel({dim: 0}, drop=True)
    return array.transpose("y", "x")


def open_zarr_data_array(
    path: str,
    *,
    variable: str | None,
    source_label: str,
) -> xr.DataArray:
    dataset = xr.open_zarr(
        path,
        consolidated=None,
        storage_options={"anon": False},
    )
    return _select_xy_data_array(
        dataset,
        preferred_variable=variable,
        source_label=source_label,
    )


def select_tile(array: xr.DataArray, tile_id: str) -> xr.DataArray:
    """Select one canonical tile using the zonal runner's exact bounds."""

    west, south, east, north = uu.get_10x10_tile_bounds(tile_id)
    x0, x1 = float(array.x.values[0]), float(array.x.values[-1])
    y0, y1 = float(array.y.values[0]), float(array.y.values[-1])
    x_slice = slice(west, east) if x0 < x1 else slice(east, west)
    y_slice = slice(south, north) if y0 < y1 else slice(north, south)
    selected = array.sel(x=x_slice, y=y_slice)
    if selected.sizes.get("x", 0) == 0 or selected.sizes.get("y", 0) == 0:
        raise ValueError(
            f"Tile {tile_id} with bounds {(west, south, east, north)} selects no pixels."
        )
    return selected


def _pixel_step(array: xr.DataArray) -> float:
    xvals = np.asarray(array.x.values)
    yvals = np.asarray(array.y.values)
    steps: list[float] = []
    if xvals.size >= 2:
        steps.append(abs(float(xvals[1]) - float(xvals[0])))
    if yvals.size >= 2:
        steps.append(abs(float(yvals[1]) - float(yvals[0])))
    if not steps:
        raise ValueError("Cannot infer pixel step from a single-pixel tile selection.")
    return min(steps)


def align_adm0_to_wdpa(
    wdpa_tile: xr.DataArray,
    adm0_tile: xr.DataArray,
    *,
    tolerance_fraction: float = 0.49,
) -> xr.DataArray:
    """Align ADM0 to WDPA with the same nearest-neighbour tolerance as zonal stats."""

    same_shape = wdpa_tile.sizes == adm0_tile.sizes
    same_x = same_shape and np.array_equal(wdpa_tile.x.values, adm0_tile.x.values)
    same_y = same_shape and np.array_equal(wdpa_tile.y.values, adm0_tile.y.values)
    if same_x and same_y:
        return adm0_tile

    tolerance = float(tolerance_fraction) * _pixel_step(wdpa_tile)
    return adm0_tile.reindex_like(wdpa_tile, method="nearest", tolerance=tolerance)


def uint8_histogram(array: xr.DataArray) -> da.Array:
    """Return a lazy 256-bin histogram for a uint8 categorical array."""

    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    counts, _ = da.histogram(array.data, bins=UINT8_HISTOGRAM_EDGES)
    return counts.astype(np.int64)


def build_candidate_histograms(
    wdpa: xr.DataArray,
    tile_ids: Sequence[str],
) -> dict[str, da.Array]:
    return {
        tile_id: uint8_histogram(select_tile(wdpa, tile_id))
        for tile_id in tile_ids
    }


def build_land_histograms(
    wdpa: xr.DataArray,
    adm0: xr.DataArray,
    tile_ids: Sequence[str],
    *,
    tolerance_fraction: float = 0.49,
) -> tuple[dict[str, da.Array], dict[str, da.Array]]:
    histograms: dict[str, da.Array] = {}
    missing_counts: dict[str, da.Array] = {}
    for tile_id in tile_ids:
        wdpa_tile = select_tile(wdpa, tile_id)
        adm0_tile = align_adm0_to_wdpa(
            wdpa_tile,
            select_tile(adm0, tile_id),
            tolerance_fraction=tolerance_fraction,
        )
        if np.issubdtype(adm0_tile.dtype, np.floating):
            missing = da.isnan(adm0_tile.data)
        else:
            missing = da.zeros_like(adm0_tile.data, dtype=bool)
        masked_values = da.where((adm0_tile.data > 0) & ~missing, wdpa_tile.data, np.uint8(0))
        masked = xr.DataArray(masked_values, dims=wdpa_tile.dims, coords=wdpa_tile.coords)
        histograms[tile_id] = uint8_histogram(masked)
        missing_counts[tile_id] = missing.sum(dtype=np.int64)
    return histograms, missing_counts


def compute_collections(
    collections: Mapping[str, dask.base.DaskMethodsMixin],
    *,
    client: Client | None,
    label: str,
) -> dict[str, Any]:
    LOGGER.info("%s: submitting %d tile tasks", label, len(collections))
    if client is not None:
        # ``Client.compute`` preserves the outer collection.  For a dictionary
        # it therefore returns one Future whose result is the computed
        # dictionary, rather than a dictionary of Futures.
        future = client.compute(dict(collections))
        progress(future)
        return dict(client.gather(future))
    with ProgressBar():
        return dict(dask.compute(dict(collections))[0])


def _histogram_as_int64(value: Any) -> np.ndarray:
    histogram = np.asarray(value, dtype=np.int64)
    if histogram.shape != (256,):
        raise ValueError(f"Expected a 256-bin uint8 histogram, got {histogram.shape}")
    return histogram


def build_scan_frame(
    tile_ids: Sequence[str],
    candidate_histograms: Mapping[str, Any],
    land_histograms: Mapping[str, Any],
    *,
    affected_codes: Sequence[int],
    registered_codes: Sequence[int],
) -> pd.DataFrame:
    affected_codes = tuple(sorted({int(code) for code in affected_codes}))
    registered_set = {int(code) for code in registered_codes}
    records: list[dict[str, Any]] = []
    for tile_id in tile_ids:
        candidate = _histogram_as_int64(candidate_histograms[tile_id])
        land_value = land_histograms.get(tile_id)
        land = (
            _histogram_as_int64(land_value)
            if land_value is not None
            else np.zeros(256, dtype=np.int64)
        )
        west, south, east, north = uu.get_10x10_tile_bounds(tile_id)
        record: dict[str, Any] = {
            "tile_id": tile_id,
            "west": west,
            "south": south,
            "east": east,
            "north": north,
            "scanned_pixels": int(candidate.sum()),
            "candidate_affected_pixels": int(candidate[list(affected_codes)].sum()),
            "affected_land_pixels": int(land[list(affected_codes)].sum()),
            "unexpected_wdpa_pixels": int(
                sum(candidate[code] for code in range(256) if code not in registered_set)
            ),
        }
        for code in affected_codes:
            record[f"wdpa_{code}_pixels"] = int(candidate[code])
            record[f"wdpa_{code}_land_pixels"] = int(land[code])
        record["is_candidate"] = record["candidate_affected_pixels"] > 0
        record["is_affected_land"] = record["affected_land_pixels"] > 0
        records.append(record)
    return pd.DataFrame.from_records(records).sort_values("tile_id").reset_index(drop=True)


def _s3_root(path: str) -> str:
    return path[5:] if path.startswith("s3://") else path


def zarr_metadata_signature(path: str) -> dict[str, Any]:
    """Capture a lightweight object signature for the exact Zarr metadata used."""

    fs = s3fs.S3FileSystem(anon=False)
    root = _s3_root(path).rstrip("/")
    for suffix in (".zmetadata", "zarr.json", ".zgroup"):
        key = f"{root}/{suffix}"
        try:
            info = fs.info(key)
        except FileNotFoundError:
            continue
        return {
            "metadata_object": f"s3://{key}",
            "etag": _json_safe(info.get("ETag") or info.get("etag")),
            "size": _json_safe(info.get("size")),
            "last_modified": _json_safe(info.get("LastModified") or info.get("last_modified")),
        }
    return {"metadata_object": None, "etag": None, "size": None, "last_modified": None}


def write_outputs(
    scan_frame: pd.DataFrame,
    *,
    output_dir: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    affected = scan_frame.loc[scan_frame["is_affected_land"]].copy()
    all_csv = output_dir / "wdpa_tile_scan_all.csv"
    affected_csv = output_dir / "wdpa_affected_tiles.csv"
    tile_ids_path = output_dir / "wdpa_affected_tile_ids.txt"
    tile_args_path = output_dir / "wdpa_affected_tile_args.txt"
    manifest_path = output_dir / "wdpa_affected_tiles.json"

    scan_frame.to_csv(all_csv, index=False)
    affected.to_csv(affected_csv, index=False)
    tile_ids = affected["tile_id"].astype(str).tolist()
    comma_separated = ",".join(tile_ids)
    tile_ids_path.write_text(comma_separated + ("\n" if comma_separated else ""), encoding="utf-8")
    tile_args_path.write_text(
        f"--execution_mode tile --tile_ids {comma_separated} --data_tile_filter auto\n"
        if comma_separated
        else "# No affected land tiles were discovered.\n",
        encoding="utf-8",
    )

    payload = dict(metadata)
    payload.update(
        {
            "scanned_tile_count": int(len(scan_frame)),
            "candidate_tile_count": int(scan_frame["is_candidate"].sum()),
            "affected_land_tile_count": int(scan_frame["is_affected_land"].sum()),
            "candidate_affected_pixels": int(scan_frame["candidate_affected_pixels"].sum()),
            "affected_land_pixels": int(scan_frame["affected_land_pixels"].sum()),
            "affected_tile_ids": tile_ids,
            "affected_tiles": affected.to_dict(orient="records"),
            "files": {
                "all_tile_scan_csv": all_csv.name,
                "affected_tiles_csv": affected_csv.name,
                "affected_tile_ids": tile_ids_path.name,
                "zonal_tile_arguments": tile_args_path.name,
            },
        }
    )
    manifest_path.write_text(json.dumps(payload, indent=2, default=_json_safe) + "\n", encoding="utf-8")
    return {
        "all_csv": all_csv,
        "affected_csv": affected_csv,
        "tile_ids": tile_ids_path,
        "tile_args": tile_args_path,
        "manifest": manifest_path,
    }


def _connect(args: argparse.Namespace) -> tuple[Any, Client | None, bool]:
    if args.run_local:
        return None, None, False
    if platform.system() != "Linux":
        raise RuntimeError(
            "Coiled discovery must be launched from WSL/Linux using the "
            "coiled_20251119 environment."
        )

    if args.launch_cluster:
        from src.scripts.utilities.create_cluster import create_cluster

        cluster = create_cluster(
            cluster_name=args.cluster_name,
            n_workers=args.workers,
            threads_per_worker=1,
            worker_memory=args.memory_gib,
            spot_policy="on-demand",
        )
        client = Client(cluster)
        uu.upload_repo_source_to_dask(client)
        uu.patch_zarr_asyncarray_config_on_workers(client, LOGGER)
        return cluster, client, True

    cluster, client, fell_back_local = uu.connect_to_cluster(
        cluster_name=args.cluster_name,
        run_local=False,
    )
    if fell_back_local or client is None:
        raise RuntimeError(
            f"Ready Coiled cluster '{args.cluster_name}' was not found. Launch it first, "
            "use --launch-cluster, or explicitly use --run-local for a small tile subset."
        )
    uu.patch_zarr_asyncarray_config_on_workers(client, LOGGER)
    return cluster, client, False


def run(args: argparse.Namespace) -> dict[str, Path] | None:
    affected_codes = tuple(sorted({int(code) for code in args.affected_codes}))
    registered_codes = tuple(int(code) for code in np.asarray(cn.WDPA_codes).tolist())
    invalid_affected = sorted(set(affected_codes) - set(registered_codes))
    if invalid_affected:
        raise ValueError(
            f"Affected codes are outside the registered WDPA contract: {invalid_affected}; "
            f"registered={list(registered_codes)}"
        )
    tile_ids = normalize_tile_ids(args.tile_ids, cn.tile_id_list)
    output_dir = args.output_dir or Path(
        f"/mnt/c/tmp/afolu/wdpa_affected_tiles/{_utc_timestamp()}"
    )

    LOGGER.info("WDPA source: %s", args.wdpa_zarr)
    LOGGER.info("ADM0 source: %s", args.adm0_zarr)
    LOGGER.info("Canonical tile source: %s", cn.tile_id_list_source)
    LOGGER.info("Tiles selected: %d", len(tile_ids))
    LOGGER.info("Affected codes: %s", list(affected_codes))
    LOGGER.info("Output directory: %s", output_dir)
    if not args.execute:
        print("Plan only; no Zarr pixels were read and no output files were written.")
        print("Add --execute and either --cluster-name <ready-cluster> or --launch-cluster.")
        return None
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Choose a fresh --output-dir."
        )

    cluster = None
    client: Client | None = None
    launched_cluster = False
    try:
        cluster, client, launched_cluster = _connect(args)
        wdpa = open_zarr_data_array(
            args.wdpa_zarr,
            variable=args.wdpa_variable,
            source_label="WDPA",
        )
        if wdpa.dtype != np.uint8:
            raise TypeError(f"Expected uint8 WDPA data, found {wdpa.dtype}")

        candidate_lazy = build_candidate_histograms(wdpa, tile_ids)
        candidate_histograms = compute_collections(
            candidate_lazy,
            client=client,
            label="WDPA candidate scan",
        )
        candidate_tile_ids = [
            tile_id
            for tile_id in tile_ids
            if int(_histogram_as_int64(candidate_histograms[tile_id])[list(affected_codes)].sum()) > 0
        ]
        LOGGER.info(
            "Candidate scan complete: %d of %d tiles contain WDPA codes %s",
            len(candidate_tile_ids),
            len(tile_ids),
            list(affected_codes),
        )

        land_histograms: dict[str, Any] = {}
        if candidate_tile_ids:
            adm0 = open_zarr_data_array(
                args.adm0_zarr,
                variable=args.adm0_variable,
                source_label="ADM0",
            )
            land_lazy, missing_lazy = build_land_histograms(
                wdpa,
                adm0,
                candidate_tile_ids,
                tolerance_fraction=args.align_tolerance_fraction,
            )
            combined_land_scan = {
                **{f"histogram::{tile_id}": value for tile_id, value in land_lazy.items()},
                **{f"missing::{tile_id}": value for tile_id, value in missing_lazy.items()},
            }
            combined_land_results = compute_collections(
                combined_land_scan,
                client=client,
                label="WDPA land-intersection and ADM0 coverage scan",
            )
            land_histograms = {
                tile_id: combined_land_results[f"histogram::{tile_id}"]
                for tile_id in candidate_tile_ids
            }
            missing_counts = {
                tile_id: combined_land_results[f"missing::{tile_id}"]
                for tile_id in candidate_tile_ids
            }
            missing = {
                tile_id: int(np.asarray(value).item())
                for tile_id, value in missing_counts.items()
                if int(np.asarray(value).item()) > 0
            }
            if missing:
                raise RuntimeError(
                    "ADM0 nearest-neighbour alignment left uncovered WDPA pixels; refusing "
                    f"to emit a potentially incomplete manifest: {missing}"
                )

        scan_frame = build_scan_frame(
            tile_ids,
            candidate_histograms,
            land_histograms,
            affected_codes=affected_codes,
            registered_codes=registered_codes,
        )
        unexpected = scan_frame.loc[
            scan_frame["unexpected_wdpa_pixels"] > 0,
            ["tile_id", "unexpected_wdpa_pixels"],
        ]
        if not unexpected.empty:
            raise RuntimeError(
                "WDPA values outside the registered 0-16 contract were found; refusing "
                f"to emit an incomplete manifest:\n{unexpected.to_string(index=False)}"
            )

        metadata = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Tiles affected by legacy WDPA expected_groups 0-11 dropping valid codes 12-16",
            "wdpa_zarr": args.wdpa_zarr,
            "wdpa_metadata_signature": zarr_metadata_signature(args.wdpa_zarr),
            "adm0_zarr": args.adm0_zarr,
            "adm0_metadata_signature": zarr_metadata_signature(args.adm0_zarr),
            "canonical_tile_source": cn.tile_id_list_source,
            "registered_wdpa_codes": list(registered_codes),
            "affected_wdpa_codes": list(affected_codes),
            "land_mask": "adm0 > 0",
            "align_tolerance_fraction": float(args.align_tolerance_fraction),
            "cluster_name": None if args.run_local else args.cluster_name,
            "execution": "local" if args.run_local else "coiled",
        }
        paths = write_outputs(scan_frame, output_dir=output_dir, metadata=metadata)
        LOGGER.info(
            "Discovery complete: candidates=%d affected_land_tiles=%d affected_land_pixels=%d",
            int(scan_frame["is_candidate"].sum()),
            int(scan_frame["is_affected_land"].sum()),
            int(scan_frame["affected_land_pixels"].sum()),
        )
        for label, path in paths.items():
            LOGGER.info("Output %s: %s", label, path)
        return paths
    finally:
        if client is not None:
            client.close()
        if launched_cluster and cluster is not None and not args.keep_cluster:
            LOGGER.info("Terminating cluster launched by this script: %s", args.cluster_name)
            cluster.shutdown()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Execute the global/category scan.")
    parser.add_argument("--wdpa-zarr", "--wdpa_zarr", default=cn.WDPA_zarr_path)
    parser.add_argument("--adm0-zarr", "--adm0_zarr", default=DEFAULT_ADM0_ZARR)
    parser.add_argument("--wdpa-variable", "--wdpa_variable", default=None)
    parser.add_argument("--adm0-variable", "--adm0_variable", default=None)
    parser.add_argument(
        "--affected-codes",
        "--affected_codes",
        nargs="+",
        type=int,
        default=list(DEFAULT_AFFECTED_CODES),
    )
    parser.add_argument(
        "--tile-ids",
        "--tile_ids",
        action="append",
        help="Optional comma-separated canonical tile IDs. Omit for the full model roster.",
    )
    parser.add_argument(
        "--align-tolerance-fraction",
        "--align_tolerance_fraction",
        type=float,
        default=0.49,
    )
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=None)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-local", "--run_local", action="store_true")
    mode.add_argument("--launch-cluster", "--launch_cluster", action="store_true")
    parser.add_argument("--cluster-name", "--cluster_name", default="wdpa_tile_discovery")
    parser.add_argument("--workers", type=int, default=25)
    parser.add_argument("--memory-gib", "--memory_gib", type=int, choices=[16, 32, 64, 128], default=32)
    parser.add_argument(
        "--keep-cluster",
        "--keep_cluster",
        action="store_true",
        help="Keep a cluster launched by this script. Existing clusters are never terminated.",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not 0 < args.align_tolerance_fraction < 1:
        parser.error("--align-tolerance-fraction must be between 0 and 1")
    if args.keep_cluster and not args.launch_cluster:
        parser.error("--keep-cluster is only valid with --launch-cluster")
    if args.run_local and not args.tile_ids:
        LOGGER.warning(
            "A full local scan reads hundreds of gigabytes after decompression. "
            "Coiled is strongly recommended."
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
