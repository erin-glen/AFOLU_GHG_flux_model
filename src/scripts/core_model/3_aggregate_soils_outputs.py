import argparse
import dask
import re

from src.scripts.utilities import constants_and_names as cn

from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu

DATA_TYPES = [
    # "burned_ch4_Mg_CO2e_ha",
    # "burned_co2_Mg_CO2_ha",
    # "burned_co_Mg_CO2e_ha",
    "burned_state",
    "burned_total_Mg_CO2e_ha",
    # "burned_total_Mg_CO2e_pixel",
    # "drained_ch4_ditch_Mg_CO2e_ha_yr",
    # "drained_ch4_land_Mg_CO2e_ha_yr",
    # "drained_co2_Mg_CO2_ha_yr",
    # "drained_co2_offsite_Mg_CO2_ha_yr",
    # "drained_n2o_Mg_CO2e_ha_yr",
    "drained_total_Mg_CO2e_ha_yr",
    # "drained_total_Mg_CO2e_pixel",
    # "drained_soil",
    "drained_state",
]

INVENTORY_PERIODS = [
    "2001_2005",
    "2006_2010",
    "2011_2015",
    "2016_2020",
    "2021_2023",
]

BASE_URL = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_4_4"
OUTPUT_DATE = "20250714"


def get_input_datasets(pixel_resolution: str = "4000_pixels") -> list:
    """Return list of S3 folders for organic soil outputs."""
    paths = []
    for period in INVENTORY_PERIODS:
        for dtype in DATA_TYPES:
            path = (
                f"{BASE_URL}/{dtype}/ogh_standard_model/"
                f"five_year_intervals/{period}/{pixel_resolution}/{OUTPUT_DATE}"
            )
            paths.append(path)
    return paths


def robust_merge_small_tiles(s3_name_dict, is_final, no_upload, no_log, logger):
    """Wrapper for :func:`merge_small_tiles_gdal` with error handling."""
    folder, output_names = next(iter(s3_name_dict.items()))
    out_file = output_names[0]
    try:
        msg, stats = uu.merge_small_tiles_gdal(s3_name_dict, is_final, no_upload, no_log)
        logger.info(f"Successfully merged: {out_file}")
        return msg, stats
    except Exception as e:
        logger.error(f"Error merging {out_file}: {e}")
        return f"Failed: {out_file} - {e}", None


def main(
    cluster_name,
    run_local: bool = False,
    no_upload: bool = False,
    no_log: bool = False,
    pixel_resolution: str = "4000_pixels",
):
    logger = lu.setup_logging_main()

    is_final = False

    cluster, client, run_local = uu.connect_to_cluster(
        cluster_name=cluster_name, run_local=run_local
    )

    stage = f"LULUCF_flux_postprocessing__outputs_aggregated_to_10x10deg_{pixel_resolution}"

    start_time = uu.timestr()
    lu.print_and_log(f"Stage {stage} started at: {start_time}", is_final, logger)

    input_datasets = get_input_datasets(pixel_resolution)

    list_of_s3_name_dicts_total = uu.create_list_for_aggregation(input_datasets, logger)

    delayed_results = [
        dask.delayed(robust_merge_small_tiles)(
            s3_name_dict, is_final, no_upload, no_log, logger
        )
        for s3_name_dict in list_of_s3_name_dicts_total
    ]

    results = dask.compute(*delayed_results)
    lu.print_and_log(results, is_final, logger)

    # extract tile ids for reporting
    tile_ids = set()
    for entry in list_of_s3_name_dicts_total:
        out_name = list(entry.values())[0][0]
        match = re.search(cn.tile_id_pattern, out_name)
        if match:
            tile_ids.add(match.group())

    chunk_list = sorted(tile_ids)

    success_count, all_stats = uu.count_successful_chunks(chunk_list, is_final, logger, results)

    output_folders = [path.replace(pixel_resolution, "40000_pixels") for path in input_datasets]

    for folder in output_folders:
        geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(folder)
        lu.print_and_log(
            f"Aggregated 10x10 deg outputs in {folder}: {file_count}", is_final, logger
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

    args = parser.parse_args()

    main(
        args.cluster_name,
        args.run_local,
        args.no_upload,
        args.no_log,
        args.pixel_resolution,
    )

"""
 python -m src.scripts.core_model.3_aggregate_soils_outputs -cn aggregate
"""
