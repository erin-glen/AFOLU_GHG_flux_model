"""
02_aggregate_soils_outputs.py
Aggregate chunk-level organic-soils model outputs to 10x10 degree tiles.

Reads per-dataset slices from the global mega-zarr and writes GeoTIFFs to S3
under ``{BASE_URL}/{dataset}/...``.  The canonical packed-state dataset is
``combined_state``; archived mega-zarr stores that still use the legacy name
``emissions_state`` are handled via a read-time fallback (see
``create_10x10_outputs_from_zarr``).  New outputs always use the canonical
``combined_state`` path.
"""

import argparse
import dask
import fsspec
import numpy as np
import zarr

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import drainage_zarr_utilities as dzu
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu

DATA_TYPES = list(cn.drainage_outputs_to_zarr)
if "combined_state" not in DATA_TYPES:
    # Keep aggregation forward-compatible when constants lag behind model outputs.
    DATA_TYPES.append("combined_state")


version = cn.model_version_underscore
BASE_URL = f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_{version}"
DEFAULT_OUTPUT_DATE = "20251007"


def get_inventory_periods(interval_type: str) -> list[str]:
    """Return inventory period labels for the selected interval type."""
    if interval_type == cn.intervals_annual:
        last_year = cn.five_year_inventory_periods[-1][1]
        return [
            f"{year}_{year}"
            for year in range(cn.annual_land_cover_start_year, last_year + 1)
        ]
    return [f"{start}_{end}" for start, end in cn.five_year_inventory_periods]


def get_output_folders(
    interval_type: str,
    output_pixel_resolution: str,
    run_name: str = "ogh_standard_model",
    output_date: str = DEFAULT_OUTPUT_DATE,
) -> list:
    """Return list of S3 folders for organic soil outputs."""
    interval_folder = f"{interval_type}_intervals"
    inventory_periods = get_inventory_periods(interval_type)
    paths = []
    for period in inventory_periods:
        for dtype in DATA_TYPES:
            path = (
                f"{BASE_URL}/{dtype}/{run_name}/"
                f"{interval_folder}/{period}/{output_pixel_resolution}/{output_date}"
            )
            paths.append(path)
    return paths


def build_year_indices(interval_type: str) -> tuple[list[int], dict[int, int]]:
    year_index = dzu.full_model_year_index(interval_type)
    year_lookup = {year: idx for idx, year in enumerate(year_index)}
    return year_index, year_lookup


def load_zarr_window(zarr_path: str, dataset: str, bounds, year_idx: int) -> np.ndarray:
    return dzu.open_zarr_window(zarr_path, dataset, bounds, year_idx)


def create_10x10_outputs_from_zarr(
    dataset: str,
    year_idx: int,
    year_value: int,
    period: str,
    tile_id: str,
    zarr_path: str,
    interval_type: str,
    output_pixel_resolution: str,
    run_name: str,
    output_date: str,
    is_final: bool,
    no_upload: bool,
    logger,
):
    """Read a single dataset/tile/period slice from the mega-zarr and write 10x10 GeoTIFFs.

    Output path contract
    --------------------
    New outputs are written under ``{BASE_URL}/{dataset}/...`` where *dataset*
    is the canonical name (e.g. ``combined_state``).  No new outputs are ever
    written under the legacy ``emissions_state`` path.

    Legacy read fallback
    --------------------
    When *dataset* is ``combined_state`` and the mega-zarr does not contain that
    variable (i.e. the store predates the rename), the function falls back to
    reading ``emissions_state`` and emits a warning.  The output is still
    published under the canonical ``combined_state`` path.
    """
    bounds = uu.get_10x10_tile_bounds(tile_id)
    chunk_px = uu.calc_chunk_length_pixels(bounds)
    bstr = uu.boundstr(bounds)

    dataset_for_read = dataset
    try:
        data_per_ha = load_zarr_window(zarr_path, dataset_for_read, bounds, year_idx)
    except Exception:
        if dataset == "combined_state":
            dataset_for_read = "emissions_state"
            logger.warning("Legacy input naming used for archived store: emissions_state -> combined_state")
            data_per_ha = load_zarr_window(zarr_path, dataset_for_read, bounds, year_idx)
        else:
            raise
    has_data = np.any(np.isfinite(data_per_ha) & (data_per_ha != 0))
    if not has_data:
        logger.info(
            "Skipping %s %s %s (no nonzero data in tile).",
            tile_id,
            dataset,
            period,
        )
        return f"Skipped empty {tile_id} {dataset} {period}", []

    dtype_str = cn.drainage_output_dtypes.get(dataset, "float32")
    is_numeric = dtype_str == "float32"

    data_per_pixel = None
    if is_numeric:
        pixel_area_uri = f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tile_id}.tif"
        pixel_area = uu.get_tile_dataset_rio(
            pixel_area_uri, "Float32", bounds, chunk_px, is_final, logger
        )[0]
        data_per_pixel = data_per_ha * pixel_area * cn.m2_to_ha

    output_interval_folder = f"{interval_type}_intervals"
    output_dir_per_ha = (
        f"{BASE_URL}/{dataset}/{run_name}/{output_interval_folder}/"
        f"{period}/{output_pixel_resolution}/{output_date}/"
    )
    output_name_per_ha = f"{tile_id}__{dataset}__{period}.tif"

    output_name_per_pixel = None
    output_dir_per_pixel = None
    if is_numeric:
        dataset_pixel = dataset.replace("_ha_", "_pixel_")
        output_dir_per_pixel = (
            f"{BASE_URL}/{dataset_pixel}/{run_name}/{output_interval_folder}/"
            f"{period}/{output_pixel_resolution}/{output_date}/"
        )
        output_name_per_pixel = f"{tile_id}__{dataset_pixel}__{period}.tif"

    if not no_upload:
        uu.save_and_upload_single_raster(
            bounds,
            chunk_px,
            tile_id,
            data_per_ha,
            data_per_ha.dtype.name,
            output_name_per_ha,
            output_dir_per_ha,
            is_final,
            logger,
        )
        if is_numeric and data_per_pixel is not None:
            uu.save_and_upload_single_raster(
                bounds,
                chunk_px,
                tile_id,
                data_per_pixel,
                data_per_pixel.dtype.name,
                output_name_per_pixel,
                output_dir_per_pixel,
                is_final,
                logger,
            )

    chunk_stats = []
    chunk_stats.append(
        uu.calculate_stats(
            data_per_ha,
            output_name_per_ha,
            bstr,
            tile_id,
            "output_layer",
            array_per_pixel=data_per_pixel if is_numeric else None,
            iv_start=None,
            iv_end=year_value,
        )
    )
    if is_numeric and output_name_per_pixel:
        chunk_stats.append(
            uu.calculate_stats(
                data_per_pixel,
                output_name_per_pixel,
                bstr,
                tile_id,
                "output_layer",
                array_per_pixel=data_per_pixel,
                iv_start=None,
                iv_end=year_value,
            )
        )

    return f"Success for {tile_id} {dataset} {period}", chunk_stats


def robust_create_10x10_outputs(*args, logger, **kwargs):
    dataset = args[0] if args else kwargs.get("dataset", "unknown")
    tile_id = args[4] if len(args) > 4 else kwargs.get("tile_id", "unknown")
    try:
        msg, stats = create_10x10_outputs_from_zarr(*args, **kwargs, logger=logger)
        logger.info(f"Successfully aggregated: {tile_id} {dataset}")
        return msg, stats
    except Exception as e:
        logger.error(f"Error aggregating {tile_id} {dataset}: {e}")
        return f"Error: {tile_id} {dataset} - {e}", None


def main(
    cluster_name,
    run_local: bool = False,
    no_upload: bool = False,
    no_log: bool = False,
    pixel_resolution: str = "4000_pixels",
    run_name: str = "ogh_standard_model",
    output_date: str = DEFAULT_OUTPUT_DATE,
    interval_type: str = cn.intervals_five_year,
):
    logger = lu.setup_logging_main()

    cluster, client, run_local = uu.connect_to_cluster(
        cluster_name=cluster_name, run_local=run_local
    )

    stage = (
        f"LULUCF_flux_postprocessing__outputs_aggregated_to_10x10deg_{pixel_resolution}"
    )

    start_time = uu.timestr()
    lu.print_and_log(f"Stage {stage} started at: {start_time}", False, logger)

    chunk_size_pixels = int(pixel_resolution.replace("_pixels", ""))
    output_pixel_resolution = f"{cn.full_raster_dims}_pixels"
    year_index, year_lookup = build_year_indices(interval_type)
    inventory_periods = get_inventory_periods(interval_type)

    zarr_path = dzu.create_mega_zarr_path(
        cn.drainage_outputs_path_mega_zarr,
        chunk_size_pixels,
        interval_type,
        run_name,
        output_date,
        logger,
    )
    fs = fsspec.filesystem("s3", anon=False)
    zarr.open_group(fs.get_mapper(zarr_path), mode="r")

    tile_ids = cn.tile_id_list
    is_final = len(tile_ids) > 20

    tasks = []
    for dataset in DATA_TYPES:
        for period in inventory_periods:
            end_year = int(period.split("_")[-1])
            year_idx = year_lookup.get(end_year)
            if year_idx is None:
                logger.warning(f"Skipping period {period}; not in zarr year index")
                continue
            for tile_id in tile_ids:
                tasks.append(
                    dask.delayed(robust_create_10x10_outputs)(
                        dataset,
                        year_idx,
                        year_index[year_idx],
                        period,
                        tile_id,
                        zarr_path,
                        interval_type,
                        output_pixel_resolution,
                        run_name,
                        output_date,
                        is_final,
                        no_upload,
                        logger=logger,
                    )
                )

    delayed_results = [
        task for task in tasks
    ]

    results = dask.compute(*delayed_results)
    lu.print_and_log(results, is_final, logger)

    success_count, all_stats = uu.count_successful_chunks(
        tile_ids, is_final, logger, results
    )

    output_folders = get_output_folders(
        interval_type, output_pixel_resolution, run_name, output_date
    )
    output_folders_pixel = [
        folder.replace("_ha_", "_pixel_")
        for folder in output_folders
        if "_ha_" in folder
    ]

    if not no_upload:
        for folder in output_folders:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(
                folder
            )
            lu.print_and_log(
                f"Aggregated 10x10 deg outputs in {folder}: {file_count}",
                is_final,
                logger,
            )
        for folder in output_folders_pixel:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(
                folder
            )
            lu.print_and_log(
                f"Aggregated 10x10 deg per-pixel outputs in {folder}: {file_count}",
                is_final,
                logger,
            )

    if success_count > 0:
        uu.aggregate_10x10_chunk_stats(all_stats, stage, no_upload, logger)

    end_time = uu.timestr()
    lu.print_and_log(f"Stage {stage} ended at: {end_time}", is_final, logger)
    uu.stage_duration(start_time, end_time, stage)

    if not run_local:
        lu.compile_worker_logs(no_log, cluster, stage, start_time, logger)

    if not run_local:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate LULUCF model outputs to 10x10 degree geotifs."
    )
    parser.add_argument("-cn", "--cluster_name", required=True, help="Cluster name")

    parser.add_argument("--run_local", action="store_true", help="Run locally without Dask/Coiled")
    parser.add_argument("--no_log", action="store_true", help="Do not create the combined log")
    parser.add_argument("--no_upload", action="store_true", help="Do not save and upload outputs to S3")
    parser.add_argument(
        "--pixel_resolution",
        choices=["4000_pixels", "8000_pixels"],
        default="4000_pixels",
        help="Input raster resolution to process",
    )
    parser.add_argument("--run_name", default="ogh_standard_model", help="Model run name")
    parser.add_argument(
        "--interval_type",
        choices=[cn.intervals_five_year, cn.intervals_annual],
        default=cn.intervals_five_year,
        help="Interval type for zarr outputs",
    )
    parser.add_argument(
        "--output_date",
        default=DEFAULT_OUTPUT_DATE,
        help="Date tag for selecting input datasets (YYYYMMDD)",
    )

    args = parser.parse_args()

    main(
        args.cluster_name,
        args.run_local,
        args.no_upload,
        args.no_log,
        args.pixel_resolution,
        args.run_name,
        args.output_date,
        args.interval_type,
    )

"""
python -m src.scripts.core_model.02_aggregate_soils_outputs -cn drainage_cluster --run_name ogh_sensitivity_500m_10 --output_date 20251118
"""
