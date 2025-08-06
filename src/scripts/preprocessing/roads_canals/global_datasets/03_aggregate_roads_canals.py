import argparse
import posixpath
import dask

from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu
from src.scripts.preprocessing import preprocessing_constants as pcn
from src.scripts.utilities.constants_and_names import (
    today_date,
    full_bucket_prefix,
)

DEFAULT_FEATURE_TYPES = ["osm_roads", "osm_canals", "grip_roads"]


def get_input_datasets(
    feature_types=DEFAULT_FEATURE_TYPES,
    pixel_resolution: str = "8000_pixels",
    date: str = today_date,
) -> list:
    """Return list of S3 folders for road and canal outputs."""
    paths = []
    for ft in feature_types:
        group, sub = ft.split("_", 1)
        paths.append(
            posixpath.join(
                full_bucket_prefix,
                pcn.datasets[group][sub]["s3_processed_base"],
                pixel_resolution,
                date,
            )
        )
    return paths


def robust_merge_small_tiles(s3_name_dict, is_final, no_upload, no_log, logger):
    """Wrapper for :func:`merge_small_tiles_gdal` with error handling."""
    folder, output_names = next(iter(s3_name_dict.items()))
    out_file = output_names[0]
    try:
        uu.merge_small_tiles_gdal(s3_name_dict, is_final, no_upload, no_log)
        logger.info(f"Successfully merged: {out_file}")
        return f"Success: {out_file}"
    except Exception as e:
        logger.error(f"Error merging {out_file}: {e}")
        return f"Failed: {out_file} - {e}"


def main(
    cluster_name,
    feature_types=DEFAULT_FEATURE_TYPES,
    run_local: bool = False,
    no_upload: bool = False,
    no_log: bool = False,
    pixel_resolution: str = "8000_pixels",
    date: str = today_date,
):
    logger = lu.setup_logging_main()

    is_final = False

    cluster, client, run_local = uu.connect_to_cluster(
        cluster_name, run_local=run_local
    )

    stage = (
        f"roads_canals_preprocessing__outputs_aggregated_to_10x10deg_{pixel_resolution}_{date}"
    )

    start_time = uu.timestr()
    lu.print_and_log(f"Stage {stage} started at: {start_time}", is_final, logger)

    input_datasets = get_input_datasets(feature_types, pixel_resolution, date)

    list_of_s3_name_dicts_total = uu.create_list_for_aggregation(input_datasets, logger)

    delayed_results = [
        dask.delayed(robust_merge_small_tiles)(
            s3_name_dict, is_final, no_upload, no_log, logger
        )
        for s3_name_dict in list_of_s3_name_dicts_total
    ]

    results = dask.compute(*delayed_results)
    lu.print_and_log(results, is_final, logger)

    output_folders = [path.replace(pixel_resolution, "40000_pixels") for path in input_datasets]

    for folder in output_folders:
        geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(folder)
        lu.print_and_log(
            f"Aggregated 10x10 deg outputs in {folder}: {file_count}", is_final, logger
        )

    end_time = uu.timestr()
    lu.print_and_log(f"Stage {stage} ended at: {end_time}", is_final, logger)
    uu.stage_duration(start_time, end_time, stage)

    if not run_local:
        lu.compile_worker_logs(no_log, cluster, stage, start_time, logger)

    if not run_local:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate road and canal outputs to 10x10 degree geotifs."
    )
    parser.add_argument("-cn", "--cluster_name", required=True, help="Cluster name")

    parser.add_argument(
        "-d",
        "--date",
        default=today_date,
        help="Date (YYYYMMDD) of road and canal outputs",
    )

    parser.add_argument(
        "--feature_types",
        nargs="+",
        default=DEFAULT_FEATURE_TYPES,
        help="Feature types to process (e.g., osm_roads osm_canals grip_roads)",
    )

    parser.add_argument("--run_local", action="store_true", help="Run locally without Dask/Coiled")
    parser.add_argument("--no_log", action="store_true", help="Do not create the combined log")
    parser.add_argument("--no_upload", action="store_true", help="Do not save and upload outputs to S3")
    parser.add_argument(
        "--pixel_resolution",
        default="8000_pixels",
        help="Input raster resolution to process",
    )

    args = parser.parse_args()

    main(
        args.cluster_name,
        args.feature_types,
        args.run_local,
        args.no_upload,
        args.no_log,
        args.pixel_resolution,
        args.date,
    )

"""
python -m src.scripts.preprocessing.roads_canals.global_datasets.03_aggregate_roads_canals -cn aggregate
"""