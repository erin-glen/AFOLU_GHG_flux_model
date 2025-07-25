import argparse
import concurrent.futures
import dask
import os
import sys
import time

from concurrent.futures import ThreadPoolExecutor

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu

# ----------------------------------------------------------------------
# input dataset configuration
# ----------------------------------------------------------------------

DATA_TYPES = [
    # "burned_ch4_Mg_CO2e_ha",
    # "burned_co2_Mg_CO2_ha",
    # "burned_co_Mg_CO2e_ha",
    "burned_total_Mg_CO2e_ha",
    # "drained_ch4_ditch_Mg_CO2e_ha_yr",
    # "drained_ch4_land_Mg_CO2e_ha_yr",
    # "drained_co2_Mg_CO2_ha_yr",
    # "drained_co2_offsite_Mg_CO2_ha_yr",
    # "drained_n2o_Mg_CO2e_ha_yr",
    "drained_total_Mg_CO2e_ha_yr",
]

INVENTORY_PERIODS = [
    "2001_2005",
    "2006_2010",
    "2011_2015",
    "2016_2020",
    "2021_2024",
]

version = cn.model_version_underscore
BASE_URL = f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_{version}"
OUTPUT_DATE = "20250724"
PIXEL_RES = "4000_pixels"


def get_input_folders(pixel_resolution: str = PIXEL_RES) -> list:
    paths = []
    for period in INVENTORY_PERIODS:
        for dtype in DATA_TYPES:
            path = (
                f"{BASE_URL}/{dtype}/ogh_standard_model/"
                f"five_year_intervals/{period}/{pixel_resolution}/{OUTPUT_DATE}/"
            )
            paths.append(path)
    return paths


# ----------------------------------------------------------------------
# per-chunk processing
# ----------------------------------------------------------------------

def create_per_pixel_soils_outputs(bounds, input_dirs, is_final, no_upload, stage):
    chunk_stats = []
    logger = lu.setup_logging_worker()
    start_time = time.time()

    uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_final, logger)

    bstr = uu.boundstr(bounds)
    tid = uu.xy_to_tile_id(bounds[0], bounds[3])
    chunk_px = uu.calc_chunk_length_pixels(bounds)

    # build download dictionary and map each dataset key to its output folder
    download_dict = {}
    outdirs = {}
    for folder in input_dirs:
        parts = folder.strip("/").split("/")
        dtype = parts[-6]
        interval = parts[-3]
        key = f"{dtype}_{interval}"
        download_dict[key] = [f"{folder}{tid}__{bstr}__{dtype}__{interval}.tif"]
        outdirs[key] = folder.replace("_ha", "_pixel")

    futures = uu.prepare_to_download_chunk(bounds, download_dict, chunk_px, is_final, logger)

    lu.print_and_log(
        f"Waiting for requests for data in chunk {bstr} in {tid}: {uu.timestr()}",
        False,
        logger,
    )

    layers = {}
    for future in concurrent.futures.as_completed(futures):
        layer = futures[future]
        data, status = future.result()
        if not status:
            lu.print_and_log(
                f"Failed to fetch {layer}; using zeros", is_final, logger
            )
        layers[layer] = data

    lu.print_and_log(
        f"Calculating per-pixel outputs in {bstr} in {tid}: {uu.timestr()}",
        False,
        logger,
    )
    uu.rename_s3_task_file(stage, bounds, "calculating_", is_final, logger)

    pixel_area_uri = f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tid}.tif"
    pixel_area_chunk = uu.get_tile_dataset_rio(
        pixel_area_uri, "Float32", bounds, chunk_px, is_final, logger
    )[0]

    upload_tasks = []
    for key, arr in layers.items():
        dtype, year_start, year_end = key.rsplit("_", 2)
        interval = f"{year_start}_{year_end}"
        dtype_pixel = dtype.replace("_ha", "_pixel")
        out_arr = arr * pixel_area_chunk * cn.m2_to_ha
        chunk_stats.append(
            uu.calculate_stats(arr, key, bstr, tid, "output_layer", out_arr)
        )
        outdir = outdirs[key]
        outfile = f"{tid}__{bstr}__{dtype_pixel}__{interval}.tif"
        upload_tasks.append(
            (
                bounds,
                chunk_px,
                tid,
                out_arr,
                out_arr.dtype.name,
                outfile,
                outdir,
                is_final,
                logger,
                0,
            )
        )

    if not no_upload:
        lu.print_and_log(
            f"Upload tasks created for {bstr} in {tid}. Uploading now: {uu.timestr()}",
            False,
            logger,
        )
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(lambda args: uu.save_and_upload_single_raster(*args), upload_tasks)

        lu.print_and_log(
            f"Uploads completed for {bstr} in {tid} using {cn.outputs_path}: {uu.timestr()}",
            is_final,
            logger,
        )

    end_time = time.time()
    lu.print_and_log(
        f"{bstr} took {round(end_time - start_time)} seconds: {uu.timestr()}",
        False,
        logger,
    )

    uu.delete_s3_task_file(stage, bounds, is_final, logger)
    return f"Success for {bstr}: {uu.timestr()}", chunk_stats


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------

def main(
        cluster_name,
        run_local=False,
        no_stats=False,
        no_log=False,
        no_upload=False,
        chunk_shapefile_uri=None,
        bounding_box=None,
        chunk_size=None,
        first_chunks=None,
        log_note=None,
):
    stage = "per_pixel_soils_outputs"
    model_type = "ogh_standard_model"

    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    main_logger, main_log_local_path = lu.populate_main_log_header(
        bounding_box,
        chunk_shapefile_uri if chunk_shapefile_uri else False,
        client,
        cluster,
        log_note,
        run_local,
        model_type,
        stage,
    )

    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"no_upload: {no_upload}")

    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)
    chunk_list, chunk_size_pixels = uu.create_chunk_list(
        bounding_box,
        chunk_shapefile_uri,
        chunk_size,
        first_chunks,
        fishnet_iso_df,
        main_logger,
    )

    if chunk_size_pixels != 4000:
        sys.exit("This stage can only be run on 1x1 degree (4000 pixel) chunks.")

    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    is_final = True if len(chunk_list) > 20 else False

    input_dirs = get_input_folders()

    main_logger.info("Creating task txts in s3...")
    uu.create_s3_task_files(stage, chunk_list)

    delayed_results = [
        dask.delayed(create_per_pixel_soils_outputs)(
            chunk,
            input_dirs,
            is_final,
            no_upload,
            stage,
        )
        for chunk in chunk_list
    ]

    results = dask.compute(*delayed_results)

    success_count, all_stats = uu.count_successful_chunks(
        chunk_list, is_final, main_logger, results
    )
    uu.stage_duration(start_time, uu.timestr(), stage)

    if (not no_stats) and success_count > 0:
        uu.compile_1x1_chunk_stats(all_stats, chunk_shapefile_uri, stage, no_upload, main_logger)

    if not no_upload:
        output_dirs = [p.replace("_ha/", "_pixel/") for p in input_dirs]
        for folder in output_dirs:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(folder)
            main_logger.info(f"Output rasters in {folder}: {file_count}")

    if not run_local:
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with log compilation")
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create per-pixel organic soil outputs")
    parser.add_argument("-cn", "--cluster_name", help="Coiled cluster name")
    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W, S, E, N")
    parser.add_argument("-cs", "--chunk_size", type=float, help="Chunk size (degrees)")
    parser.add_argument("-cshp", "--chunk_shapefile_uri", help="s3 path to chunk shapefile")
    parser.add_argument("-f", "--first_chunks", type=int, help="Process only first N chunks")
    parser.add_argument("-ln", "--log_note", help="Note for log")
    parser.add_argument("--run_local", action="store_true", help="Run locally without Dask")
    parser.add_argument("--no_stats", action="store_true", help="Skip chunk stats spreadsheet")
    parser.add_argument("--no_log", action="store_true", help="Do not create combined log")
    parser.add_argument("--no_upload", action="store_true", help="Do not upload outputs to S3")

    args = parser.parse_args()

    main(
        args.cluster_name,
        run_local=args.run_local,
        no_stats=args.no_stats,
        no_log=args.no_log,
        no_upload=args.no_upload,
        chunk_shapefile_uri=args.chunk_shapefile_uri,
        bounding_box=args.bounding_box,
        chunk_size=args.chunk_size,
        first_chunks=args.first_chunks,
        log_note=args.log_note,
    )

    """
python -m src.scripts.core_model.2_per_pixel_soils_outputs \
  --cluster_name per_pixel \
  --bounding_box 110 -10 120 0 \
  --chunk_size 1 \
  --log_note "Testing per-pixel outputs" 
  
python -m src.scripts.core_model.2_per_pixel_soils_outputs \
  --cluster_name per_pixel \
  --chunk_size 1 


    """