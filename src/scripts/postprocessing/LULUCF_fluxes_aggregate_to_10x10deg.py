import argparse
import dask

# Project imports
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu


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
    run_local=False,
    no_upload=False,
    no_log=False,
    pixel_resolution="4000_pixels",
):

    logger = lu.setup_logging_main()

    is_final = False

    # Connect to cluster or run locally
    cluster, client = uu.connect_to_cluster(cluster_name, run_local=run_local)

    stage = (
        f"LULUCF_flux_postprocessing__outputs_aggregated_to_10x10deg_{pixel_resolution}"
    )

    start_time = uu.timestr()
    lu.print_and_log(f"Stage {stage} started at: {start_time}", is_final, logger)

    # Hardcoded datasets
    input_datasets = [
        # f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/sdpt/{pixel_resolution}/20250531"
        # f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_roads_density/{pixel_resolution}/20250526"
        # f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_canals_density/{pixel_resolution}/20250526",
        # f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/grip_density/{pixel_resolution}/20250526",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_ch4_co2e/ogh_standard_model/five_years_intervals/2000_2005/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_co2/ogh_standard_model/five_years_intervals/2000_2005/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_co_co2e/ogh_standard_model/five_years_intervals/2000_2005/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_state/ogh_standard_model/five_years_intervals/2000_2005/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_total_co2e/ogh_standard_model/five_years_intervals/2000_2005/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_ch4_ditch_co2e/ogh_standard_model/five_years_intervals/2000_2005/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_ch4_land_co2e/ogh_standard_model/five_years_intervals/2000_2005/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_co2/ogh_standard_model/five_years_intervals/2000_2005/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_co2_offsite/ogh_standard_model/five_years_intervals/2000_2005/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_n2o_co2e/ogh_standard_model/five_years_intervals/2000_2005/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_total_co2e/ogh_standard_model/five_years_intervals/2000_2005/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/soil/ogh_standard_model/five_years_intervals/2000_2005/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/state/ogh_standard_model/five_years_intervals/2000_2005/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_ch4_co2e/ogh_standard_model/five_years_intervals/2005_2010/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_co2/ogh_standard_model/five_years_intervals/2005_2010/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_co_co2e/ogh_standard_model/five_years_intervals/2005_2010/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_state/ogh_standard_model/five_years_intervals/2005_2010/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_total_co2e/ogh_standard_model/five_years_intervals/2005_2010/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_ch4_ditch_co2e/ogh_standard_model/five_years_intervals/2005_2010/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_ch4_land_co2e/ogh_standard_model/five_years_intervals/2005_2010/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_co2/ogh_standard_model/five_years_intervals/2005_2010/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_co2_offsite/ogh_standard_model/five_years_intervals/2005_2010/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_n2o_co2e/ogh_standard_model/five_years_intervals/2005_2010/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_total_co2e/ogh_standard_model/five_years_intervals/2005_2010/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/soil/ogh_standard_model/five_years_intervals/2005_2010/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/state/ogh_standard_model/five_years_intervals/2005_2010/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_ch4_co2e/ogh_standard_model/five_years_intervals/2010_2015/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_co2/ogh_standard_model/five_years_intervals/2010_2015/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_co_co2e/ogh_standard_model/five_years_intervals/2010_2015/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_state/ogh_standard_model/five_years_intervals/2010_2015/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_total_co2e/ogh_standard_model/five_years_intervals/2010_2015/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_ch4_ditch_co2e/ogh_standard_model/five_years_intervals/2010_2015/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_ch4_land_co2e/ogh_standard_model/five_years_intervals/2010_2015/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_co2/ogh_standard_model/five_years_intervals/2010_2015/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_co2_offsite/ogh_standard_model/five_years_intervals/2010_2015/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_n2o_co2e/ogh_standard_model/five_years_intervals/2010_2015/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_total_co2e/ogh_standard_model/five_years_intervals/2010_2015/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/soil/ogh_standard_model/five_years_intervals/2010_2015/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/state/ogh_standard_model/five_years_intervals/2010_2015/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_ch4_co2e/ogh_standard_model/five_years_intervals/2015_2020/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_co2/ogh_standard_model/five_years_intervals/2015_2020/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_co_co2e/ogh_standard_model/five_years_intervals/2015_2020/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_state/ogh_standard_model/five_years_intervals/2015_2020/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_total_co2e/ogh_standard_model/five_years_intervals/2015_2020/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_ch4_ditch_co2e/ogh_standard_model/five_years_intervals/2015_2020/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_ch4_land_co2e/ogh_standard_model/five_years_intervals/2015_2020/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_co2/ogh_standard_model/five_years_intervals/2015_2020/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_co2_offsite/ogh_standard_model/five_years_intervals/2015_2020/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_n2o_co2e/ogh_standard_model/five_years_intervals/2015_2020/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_total_co2e/ogh_standard_model/five_years_intervals/2015_2020/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/soil/ogh_standard_model/five_years_intervals/2015_2020/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/state/ogh_standard_model/five_years_intervals/2015_2020/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_ch4_co2e/ogh_standard_model/five_years_intervals/2020_2023/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_co2/ogh_standard_model/five_years_intervals/2020_2023/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_co_co2e/ogh_standard_model/five_years_intervals/2020_2023/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_state/ogh_standard_model/five_years_intervals/2020_2023/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/burned_total_co2e/ogh_standard_model/five_years_intervals/2020_2023/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_ch4_ditch_co2e/ogh_standard_model/five_years_intervals/2020_2023/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_ch4_land_co2e/ogh_standard_model/five_years_intervals/2020_2023/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_co2/ogh_standard_model/five_years_intervals/2020_2023/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_co2_offsite/ogh_standard_model/five_years_intervals/2020_2023/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_n2o_co2e/ogh_standard_model/five_years_intervals/2020_2023/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/drained_total_co2e/ogh_standard_model/five_years_intervals/2020_2023/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/soil/ogh_standard_model/five_years_intervals/2020_2023/{pixel_resolution}/20250605",
        f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_6/state/ogh_standard_model/five_years_intervals/2020_2023/{pixel_resolution}/20250605"
    ]

    # Generate aggregation tasks
    list_of_s3_name_dicts_total = uu.create_list_for_aggregation(input_datasets, logger)

    # Execute aggregation tasks in parallel
    delayed_results = [
        dask.delayed(robust_merge_small_tiles)(
            s3_name_dict, is_final, no_upload, no_log, logger
        )
        for s3_name_dict in list_of_s3_name_dicts_total
    ]

    results = dask.compute(*delayed_results)
    lu.print_and_log(results, is_final, logger)

    output_folders = [
        path.replace(pixel_resolution, "40000_pixels") for path in input_datasets
    ]

    # Confirm aggregated outputs in S3
    for folder in output_folders:
        geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(
            folder
        )
        lu.print_and_log(
            f"Aggregated 10x10 deg outputs in {folder}: {file_count}", is_final, logger
        )

    end_time = uu.timestr()
    lu.print_and_log(f"Stage {stage} ended at: {end_time}", is_final, logger)
    uu.stage_duration(start_time, end_time, stage)

    log_note = f"{stage} run"
    if not run_local:
        lu.compile_worker_logs(
            no_log,
            cluster,
            stage,
            start_time,
            logger,
        )

    if not run_local:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate LULUCF model outputs to 10x10 degree geotifs."
    )
    parser.add_argument("-cn", "--cluster_name", required=True, help="Cluster name")

    parser.add_argument(
        "--run_local", action="store_true", help="Run locally without Dask/Coiled"
    )
    parser.add_argument(
        "--no_log", action="store_true", help="Do not create the combined log"
    )
    parser.add_argument(
        "--no_upload", action="store_true", help="Do not save and upload outputs to S3"
    )
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