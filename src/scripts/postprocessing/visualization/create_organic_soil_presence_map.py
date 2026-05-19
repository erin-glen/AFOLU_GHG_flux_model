"""Create a global 1 km organic-soil presence map from OGH probability tiles.

This is a lightweight side workflow for map production and QA. It does not run
the emissions model. It reads the threshold registry, thresholds the 30 m OGH
probability tiles either globally or by biome, aggregates native presence to a
0.01 degree grid by block ``any``, then writes a single global GeoTIFF.

Example:

python -m src.scripts.postprocessing.visualization.create_organic_soil_presence_map \
  --cluster_name organic_soil_maps \
  --organic_soil_version 20260508 \
  --fscore_metric f1 \
  --threshold_method per-biome
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import posixpath
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import boto3
import dask
import numpy as np
import rasterio
import s3fs
from dask.distributed import Client, LocalCluster, as_completed
from rasterio.transform import from_bounds
from rasterio.windows import Window

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import log_utilities as lu
from src.scripts.utilities import universal_utilities as uu


OUTPUT_ROOT = (
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/"
    "uncertainty/global_organic_soil_presence"
)
PROBABILITY_TILE_PREFIX_TEMPLATE = (
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/"
    "peat_mask/OGH/tiles_unthresholded/{date}/"
)
PROBABILITY_TILE_SUFFIX = "_ogh_unthresholded_mask.tif"
CLIMATE_DOMAIN_PREFIX = cn.dirs["climate_domain"].rstrip("/") + "/"
CLIMATE_DOMAIN_PATTERN = cn.patterns["climate_domain"]

DEFAULT_REGISTRY = Path("docs/organic_soil_threshold_registry.csv")
DEFAULT_TARGET_DEG = 0.01
DEFAULT_NATIVE_DEG = 0.00025
DEFAULT_BLOCK_OUT_ROWS = 100
NODATA = 255

THRESHOLD_METHOD_ALIASES = {
    "global": "global",
    "per-biome": "per-biome",
    "per_biome": "per-biome",
    "biome": "per-biome",
}


@dataclass(frozen=True)
class ThresholdConfig:
    version: str
    metric: str
    method: str
    fallback_threshold: float
    thresholds_by_biome_code: dict[int, float]


@dataclass(frozen=True)
class TileResult:
    tile_id: str
    bounds: tuple[float, float, float, float]
    data: np.ndarray
    present_pixels: int
    status: str


def _s3_to_vsi(path: str) -> str:
    return "/vsis3/" + path[len("s3://") :] if path.startswith("s3://") else path


def _split_s3(path: str) -> tuple[str, str]:
    if not path.startswith("s3://"):
        raise ValueError(f"Expected s3:// path, got {path!r}")
    bucket, key = path[5:].split("/", 1)
    return bucket, key


def _join_s3(prefix: str, name: str) -> str:
    return f"{prefix.rstrip('/')}/{name}"


def _normalize_metric(metric: str) -> str:
    key = str(metric).strip().upper()
    if key not in {"F1", "F2", "MIXED"}:
        raise ValueError(f"--fscore_metric must be f1, f2, or mixed, got {metric!r}")
    return key


def _normalize_method(method: str) -> str:
    key = str(method).strip().lower()
    try:
        return THRESHOLD_METHOD_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            "--threshold_method must be one of: global, per-biome"
        ) from exc


def _read_registry(registry_path: Path) -> list[dict[str, str]]:
    with registry_path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def load_threshold_config(
    registry_path: Path,
    organic_soil_version: str,
    fscore_metric: str,
    threshold_method: str,
) -> ThresholdConfig:
    """Load version/metric thresholds from the registry and scale to 0..100."""

    metric = _normalize_metric(fscore_metric)
    method = _normalize_method(threshold_method)
    rows = [
        row
        for row in _read_registry(registry_path)
        if row["organic_soil_version"] == organic_soil_version
        and row["optimization_metric"].upper() == metric
    ]
    if not rows:
        raise ValueError(
            f"No threshold rows found for version={organic_soil_version!r}, metric={metric!r}"
        )

    global_rows = [row for row in rows if row["threshold_scope"] == "global" and row["biome"] == "all"]
    if len(global_rows) != 1:
        raise ValueError(
            f"Expected one global threshold row for version={organic_soil_version}, metric={metric}; "
            f"found {len(global_rows)}"
        )
    fallback = _scale_threshold(float(global_rows[0]["best_threshold"]))

    if method == "global":
        return ThresholdConfig(
            version=organic_soil_version,
            metric=metric,
            method=method,
            fallback_threshold=fallback,
            thresholds_by_biome_code={},
        )

    thresholds: dict[int, float] = {}
    for row in rows:
        if row["threshold_scope"] != "biome":
            continue
        biome = row["biome"].strip().lower()
        if biome not in cn.ecozone_codes:
            raise ValueError(f"Unknown biome in threshold registry: {biome!r}")
        thresholds[int(cn.ecozone_codes[biome])] = _scale_threshold(float(row["best_threshold"]))

    required = {
        cn.ecozone_codes["boreal"],
        cn.ecozone_codes["temperate"],
        cn.ecozone_codes["tropical"],
    }
    missing = sorted(required.difference(thresholds))
    if missing:
        raise ValueError(
            f"Missing per-biome thresholds for version={organic_soil_version}, metric={metric}: {missing}"
        )

    return ThresholdConfig(
        version=organic_soil_version,
        metric=metric,
        method=method,
        fallback_threshold=fallback,
        thresholds_by_biome_code=thresholds,
    )


def _scale_threshold(value: float) -> float:
    """Registry thresholds are 0..1; OGH tile values are uint8 0..100."""

    return value * 100.0 if value <= 1.0 else value


def probability_tile_prefix(date: str) -> str:
    return PROBABILITY_TILE_PREFIX_TEMPLATE.format(date=date)


def _tile_id_from_probability_name(name: str) -> Optional[str]:
    if not name.endswith(PROBABILITY_TILE_SUFFIX):
        return None
    return name[: -len(PROBABILITY_TILE_SUFFIX)]


def list_probability_tiles(prefix: str, limit: Optional[int] = None) -> list[str]:
    fs = s3fs.S3FileSystem(anon=False)
    files = fs.glob(prefix.rstrip("/") + f"/*{PROBABILITY_TILE_SUFFIX}")
    tile_ids = []
    for path in files:
        tile_id = _tile_id_from_probability_name(posixpath.basename(path))
        if tile_id:
            tile_ids.append(tile_id)
    tile_ids = sorted(set(tile_ids))
    if limit is not None:
        tile_ids = tile_ids[: int(limit)]
    return tile_ids


def parse_tile_ids(raw_values: Optional[list[str]]) -> list[str]:
    if not raw_values:
        return []
    tiles: list[str] = []
    for value in raw_values:
        tiles.extend(item.strip() for item in value.replace(",", " ").split() if item.strip())
    return sorted(set(tiles))


def _empty_aggregated_tile(
    probability_path: str,
    native_deg: float,
    target_deg: float,
) -> np.ndarray:
    with rasterio.open(_s3_to_vsi(probability_path)) as src:
        factor = _aggregation_factor(native_deg, target_deg)
        return np.zeros((src.height // factor, src.width // factor), dtype=np.uint8)


def _aggregation_factor(native_deg: float, target_deg: float) -> int:
    factor_f = target_deg / native_deg
    if not np.isclose(round(factor_f), factor_f):
        raise ValueError(f"target_deg/native_deg must be an integer; got {target_deg}/{native_deg}")
    return int(round(factor_f))


def aggregate_presence_block(
    probability: np.ndarray,
    climate_domain: Optional[np.ndarray],
    *,
    factor: int,
    threshold_config: ThresholdConfig,
) -> np.ndarray:
    """Aggregate a native-resolution block to presence by block ``any``."""

    rows, cols = probability.shape
    out_rows = rows // factor
    out_cols = cols // factor
    probability = probability[: out_rows * factor, : out_cols * factor]

    if threshold_config.method == "global":
        present = probability >= threshold_config.fallback_threshold
    else:
        if climate_domain is None:
            present = probability >= threshold_config.fallback_threshold
        else:
            climate_domain = climate_domain[: out_rows * factor, : out_cols * factor]
            present = probability >= threshold_config.fallback_threshold
            for code, biome_threshold in threshold_config.thresholds_by_biome_code.items():
                mask = climate_domain == code
                present[mask] = probability[mask] >= biome_threshold

    return present.reshape(out_rows, factor, out_cols, factor).any(axis=(1, 3)).astype(np.uint8)


def aggregate_probability_tile(
    tile_id: str,
    probability_prefix: str,
    threshold_config: ThresholdConfig,
    native_deg: float,
    target_deg: float,
    block_out_rows: int,
) -> TileResult:
    probability_path = _join_s3(probability_prefix, f"{tile_id}{PROBABILITY_TILE_SUFFIX}")
    climate_path = _join_s3(CLIMATE_DOMAIN_PREFIX, CLIMATE_DOMAIN_PATTERN.format(tile_id=tile_id))
    bounds = tuple(float(value) for value in uu.get_10x10_tile_bounds(tile_id))
    factor = _aggregation_factor(native_deg, target_deg)

    with rasterio.Env():
        try:
            probability_src = rasterio.open(_s3_to_vsi(probability_path))
        except Exception:
            empty = np.zeros((int(round(10 / target_deg)), int(round(10 / target_deg))), dtype=np.uint8)
            return TileResult(tile_id, bounds, empty, 0, "missing_probability_tile")

        with probability_src:
            out_rows = probability_src.height // factor
            out_cols = probability_src.width // factor
            out = np.zeros((out_rows, out_cols), dtype=np.uint8)

            climate_src = None
            if threshold_config.method == "per-biome":
                try:
                    climate_src = rasterio.open(_s3_to_vsi(climate_path))
                except Exception:
                    climate_src = None

            try:
                for out_row0 in range(0, out_rows, block_out_rows):
                    out_row1 = min(out_row0 + block_out_rows, out_rows)
                    in_row0 = out_row0 * factor
                    in_rows = (out_row1 - out_row0) * factor
                    in_cols = out_cols * factor
                    window = Window(0, in_row0, in_cols, in_rows)
                    probability = probability_src.read(1, window=window, out_dtype="uint8", masked=False)
                    climate = None
                    if climate_src is not None:
                        climate = climate_src.read(1, window=window, out_dtype="int16", masked=False)
                    out[out_row0:out_row1, :] = aggregate_presence_block(
                        probability,
                        climate,
                        factor=factor,
                        threshold_config=threshold_config,
                    )
            finally:
                if climate_src is not None:
                    climate_src.close()

    return TileResult(tile_id, bounds, out, int(out.sum()), "ok")


def iter_tile_results(
    tile_ids: list[str],
    client: Optional[Client],
    threshold_config: ThresholdConfig,
    probability_prefix: str,
    native_deg: float,
    target_deg: float,
    block_out_rows: int,
) -> Iterator[TileResult]:
    delayed = [
        dask.delayed(aggregate_probability_tile)(
            tile_id,
            probability_prefix,
            threshold_config,
            native_deg,
            target_deg,
            block_out_rows,
        )
        for tile_id in tile_ids
    ]
    if client is None:
        for result in dask.compute(*delayed):
            yield result
        return

    futures = client.compute(delayed, sync=False)
    for future in as_completed(futures):
        yield future.result()


def _paste_tile(global_arr: np.memmap, tile: TileResult, target_deg: float) -> None:
    min_x, min_y, max_x, max_y = tile.bounds
    x0 = int(round((min_x + 180) / target_deg))
    y0 = int(round((90 - max_y) / target_deg))
    rows, cols = tile.data.shape
    global_arr[y0 : y0 + rows, x0 : x0 + cols] = tile.data


def _output_name(config: ThresholdConfig, target_deg: float) -> str:
    label = str(target_deg).replace(".", "_").rstrip("0").rstrip("_")
    method = config.method.replace("-", "_")
    return (
        f"{label}deg_global_organic_soil_presence__"
        f"{config.version}__{method}__{config.metric.lower()}.tif"
    )


def _default_output_prefix(config: ThresholdConfig, target_deg: float) -> str:
    method = config.method.replace("-", "_")
    label = str(target_deg).replace(".", "_").rstrip("0").rstrip("_")
    return posixpath.join(
        OUTPUT_ROOT,
        config.version,
        method,
        config.metric.lower(),
        f"{label}deg",
    )


def _write_geotiff(path: Path, arr: np.ndarray, target_deg: float) -> None:
    rows, cols = arr.shape
    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_bounds(-180, -90, 180, 90, cols, rows),
        "nodata": NODATA,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "DEFLATE",
        "predictor": 1,
        "bigtiff": "YES",
        "num_threads": "ALL_CPUS",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.update_tags(
            class_0="not organic soil presence",
            class_1="organic soil presence",
            nodata=str(NODATA),
        )


def _upload_file(local_path: Path, s3_path: str) -> None:
    bucket, key = _split_s3(s3_path)
    boto3.client("s3").upload_file(str(local_path), bucket, key)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _connect_client(cluster_name: str, run_local: bool, n_workers: int) -> Optional[Client]:
    if run_local:
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True)
        return Client(cluster)
    _cluster, client, effective_run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local=False)
    if effective_run_local:
        logging.getLogger(__name__).warning(
            "Coiled cluster %s unavailable; continuing with the local Dask scheduler.",
            cluster_name,
        )
        return None
    return client


def run(args: argparse.Namespace) -> dict:
    logger = lu.setup_logging()
    registry_path = Path(args.threshold_registry)
    config = load_threshold_config(
        registry_path,
        args.organic_soil_version,
        args.fscore_metric,
        args.threshold_method,
    )
    probability_prefix = args.probability_tile_prefix or probability_tile_prefix(
        args.probability_date or config.version.split("_", 1)[0]
    )

    tile_ids = parse_tile_ids(args.tile_id)
    if not tile_ids:
        tile_ids = list_probability_tiles(probability_prefix, limit=args.limit_tiles)
    if not tile_ids:
        raise RuntimeError(f"No probability tiles found under {probability_prefix}")

    rows = int(round(180 / args.target_deg))
    cols = int(round(360 / args.target_deg))
    output_prefix = args.output_prefix or _default_output_prefix(config, args.target_deg)
    output_name = args.output_name or _output_name(config, args.target_deg)
    output_path = _join_s3(output_prefix, output_name) if output_prefix.startswith("s3://") else str(Path(output_prefix) / output_name)

    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="organic_soil_presence_"))
    workdir.mkdir(parents=True, exist_ok=True)
    memmap_path = workdir / "global_presence.dat"
    local_tif = workdir / output_name
    summary_path = workdir / output_name.replace(".tif", ".summary.json")

    logger.info(
        "Creating organic soil presence map | version=%s metric=%s method=%s tiles=%d output=%s",
        config.version,
        config.metric,
        config.method,
        len(tile_ids),
        output_path,
    )

    global_arr = np.memmap(memmap_path, dtype=np.uint8, mode="w+", shape=(rows, cols))
    global_arr[:] = 0

    client = _connect_client(args.cluster_name, args.run_local, args.local_workers)
    completed = 0
    present_pixels = 0
    statuses: dict[str, int] = {}
    try:
        for tile in iter_tile_results(
            tile_ids,
            client,
            config,
            probability_prefix,
            args.native_deg,
            args.target_deg,
            args.block_out_rows,
        ):
            _paste_tile(global_arr, tile, args.target_deg)
            completed += 1
            present_pixels += tile.present_pixels
            statuses[tile.status] = statuses.get(tile.status, 0) + 1
            if completed % args.log_every == 0 or completed == len(tile_ids):
                logger.info("Completed %d/%d tiles; present 1km pixels=%d", completed, len(tile_ids), present_pixels)
    finally:
        if client is not None:
            client.close()

    global_arr.flush()
    _write_geotiff(local_tif, global_arr, args.target_deg)

    summary = {
        "organic_soil_version": config.version,
        "probability_tile_prefix": probability_prefix,
        "fscore_metric": config.metric,
        "threshold_method": config.method,
        "fallback_threshold_0_to_100": config.fallback_threshold,
        "thresholds_by_biome_code_0_to_100": config.thresholds_by_biome_code,
        "target_deg": args.target_deg,
        "native_deg": args.native_deg,
        "tile_count": len(tile_ids),
        "status_counts": statuses,
        "present_pixels": int(present_pixels),
        "present_1km_pixels": int(present_pixels),
        "output_path": output_path,
    }
    _write_json(summary_path, summary)

    if output_path.startswith("s3://"):
        _upload_file(local_tif, output_path)
        _upload_file(summary_path, output_path.replace(".tif", ".summary.json"))
    else:
        final_path = Path(output_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if local_tif.resolve() != final_path.resolve():
            shutil.copy2(local_tif, final_path)
            shutil.copy2(summary_path, final_path.with_suffix(".summary.json"))

    logger.info("Wrote organic soil presence map: %s", output_path)
    logger.info("Wrote summary: %s", output_path.replace(".tif", ".summary.json"))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--organic_soil_version", required=True, help="Registry version, e.g. 20260508 or 20251105_legacy.")
    parser.add_argument("--probability_date", default=None, help="Probability tile date folder. Defaults to version date.")
    parser.add_argument(
        "--fscore_metric",
        required=True,
        choices=["f1", "f2", "mixed", "F1", "F2", "MIXED"],
    )
    parser.add_argument("--threshold_method", default="per-biome", choices=["global", "per-biome", "per_biome", "biome"])
    parser.add_argument("--threshold_registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--probability_tile_prefix", default=None, help="Override source probability tile prefix.")
    parser.add_argument("--output_prefix", default=None, help="S3 or local output prefix.")
    parser.add_argument("--output_name", default=None, help="Override output GeoTIFF filename.")
    parser.add_argument("--target_deg", type=float, default=DEFAULT_TARGET_DEG)
    parser.add_argument("--native_deg", type=float, default=DEFAULT_NATIVE_DEG)
    parser.add_argument("--block_out_rows", type=int, default=DEFAULT_BLOCK_OUT_ROWS)
    parser.add_argument("--tile_id", action="append", help="Optional tile IDs for smoke tests; comma or space separated.")
    parser.add_argument("--limit_tiles", type=int, default=None, help="Optional first-N tile limit for smoke tests.")
    parser.add_argument("--cluster_name", "-cn", default="organic_soil_maps")
    parser.add_argument("--run_local", action="store_true")
    parser.add_argument("--local_workers", type=int, default=2)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--log_every", type=int, default=10)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
