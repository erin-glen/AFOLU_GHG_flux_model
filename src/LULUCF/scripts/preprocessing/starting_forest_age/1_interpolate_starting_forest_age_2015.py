"""
Run from src/LULUCF

Local:
Has age data (should not have any 0s):
python -m scripts.preprocessing.starting_forest_age.1_interpolate_starting_forest_age_2015 -cn AFOLU_flux_model_scripts -bb 10 49 11 50 -cs 1 --run_local --no_upload
Does not have age data (should output a raster full of 0s):
python -m scripts.preprocessing.starting_forest_age.1_interpolate_starting_forest_age_2015 -cn AFOLU_flux_model_scripts -bb -28 -60 -27 -59 -cs 1 --run_local

Coiled tiny test:
python -m scripts.utilities.create_cluster -cn AFOLU_flux_model_scripts -n 1
python -m scripts.preprocessing.starting_forest_age.1_interpolate_starting_forest_age_2015 -cn AFOLU_flux_model_scripts -bb 10 49 11 50 -cs 1
python -m scripts.preprocessing.starting_forest_age.1_interpolate_starting_forest_age_2015 -cn AFOLU_flux_model_scripts -cshp -f 5

Full run:
python -m scripts.utilities.create_cluster -n 60 -t 5 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing.starting_forest_age.1_interpolate_starting_forest_age_2015 -cn AFOLU_flux_model_scripts -cshp -ln "This is intended to be the definitive interpolated forest age for 2015."
"""

import argparse
import numpy as np
import rasterio
import rasterio.merge
import dask
import fsspec

from rasterio.windows import from_bounds, transform
from scipy.ndimage import distance_transform_edt


# Project imports
from ...utilities import constants_and_names as cn
from ...utilities import log_utilities as lu
from ...utilities import universal_utilities as uu
from ...utilities import resize_cluster


def interpolate_starting_forest_age(bounds, is_final, no_upload, output_dir_list, stage):

    chunk_stats = []

    logger_worker = lu.setup_logging_worker()

    try:

        uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_final, logger_worker)

        bounds_str = uu.boundstr(bounds)
        tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])
        chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)

        ### Part 1: Identifies chunks adjacent to focal chunk and downloads focal and adjacent chunks

        input_dir = cn.forest_age_2015_dir
        output_dir = output_dir_list[0]

        # The buffer determines which chunks adjacent to the focal chunk are downloaded, so it doesn't matter exactly how
        # big the buffer it as long as it's <1 deg.
        # 10 pixels at 120 px/deg.
        buffer_deg = 1 / 120 * 10

        # Bounding box of the focal tile with the buffer
        buffered_bounds = (
            bounds[0] - buffer_deg, # West - buffer
            bounds[1] - buffer_deg, # South - buffer
            bounds[2] + buffer_deg, # East + buffer
            bounds[3] + buffer_deg, # North + buffer
        )

        # Determines which chunks intersect buffered bounds
        all_tile_bounds = uu.get_adjacent_1x1_chunks(buffered_bounds)  # A list of chunk bounds to download: focal chunk and all 8 adjacent ones

        # List of tuples of tile_ids and chunk bounds, e.g., [('50N_000E', (9, 47, 10, 48)), ('50N_000E', (9, 48, 10, 49)), ('50N_000E', (9, 49, 10, 50)),...]
        tile_ids_and_bounds = [(uu.xy_to_tile_id(xmin, ymax), (xmin, ymin, xmax, ymax)) for (xmin, ymin, xmax, ymax) in all_tile_bounds]
        lu.print_and_log(f" {len(tile_ids_and_bounds)} chunks to analyze for {bounds_str} (including the focal chunk): {tile_ids_and_bounds}: {uu.timestr()}", False, logger_worker)

        fs = fsspec.filesystem("s3")
        src_datasets = []

        # Tries to load each chunk identified above (focal, plus 8 adjacent chunks)
        for neighbor_tile_id, tile_bounds in tile_ids_and_bounds:
            neighbor_bounds_str = uu.boundstr(tile_bounds)
            tile_path = f"{input_dir}{neighbor_tile_id}__{neighbor_bounds_str}__{cn.forest_age_2015_pattern}.tif"
            tile_path = tile_path.replace("CHUNK_SIZE", str(chunk_length_pixels))

            try:
                if fs.exists(tile_path):
                    f = fs.open(tile_path, "rb")
                    src = rasterio.open(f)
                    src_datasets.append(src)
            except Exception as e:
                # Logs when a chunk can't be opened. Doesn't fail when no chunks are found, because that chunk might not exist.
                lu.print_and_log(f"Tile not found or error opening {tile_path}: {e}: {uu.timestr()}", False, logger_worker)

        lu.print_and_log(f"{len(src_datasets)} chunks found for {bounds_str} (including the focal chunk): {uu.timestr()}", is_final, logger_worker)

        if not src_datasets:
            lu.print_and_log(f"Focal chunk {bounds_str} and adjacent chunks do not have data: {uu.timestr()}", is_final, logger_worker)


        ### Part 2: Merges focal and adjacent age rasters, interpolates to fill in all missing pixels, and clips to focal chunk extent
        ### NOTE: does not calculate chunk stats for input chunks because the individual focal chunks are never
        ### isolated as inputs; they are merged with adjacent chunks. Thus, there is never really a chance
        ### to calculate input chunk stats, and it's not worth revising the workflow to include that.

        uu.rename_s3_task_file(stage, bounds, "calculating_", is_final, logger_worker)

        # Merges the focal and adjacent chunks into a mosaic
        mosaic_data, mosaic_transform = rasterio.merge.merge(src_datasets, bounds=buffered_bounds)
        mosaic_data = mosaic_data[0]  # Only one band

        # Creates window for focal chunk bounds within the mosaic
        crop_window = from_bounds(*bounds, transform=mosaic_transform).round_offsets().round_lengths()

        # Interpolates over all 0 values (0 = NoData)

        # Mask of non-zero values
        mask = mosaic_data != 0

        # Separate processing routes for chunks that have no age values in them vs. those that do
        # Mosaics with no age values-- entire mosaic (focal chunk + adjacent chunks) is filled with 0s
        if not np.any(mask):

            lu.print_and_log(f"Focal chunk {bounds_str} and adjacent chunks do not have non-zero values. Filling with 0s: {uu.timestr()}", False, logger_worker)

            # Fills focal chunk extent with 0s
            filled_crop = np.zeros((int(crop_window.height), int(crop_window.width)), dtype=mosaic_data.dtype)

        # Mosaics with age values
        else:

            lu.print_and_log(f"Focal chunk {bounds_str} or adjacent chunks have non-zero values. Interpolating ages: {uu.timestr()}", False, logger_worker)

            distance, indices = distance_transform_edt(~mask, return_indices=True)
            filled_data = mosaic_data[tuple(indices)]

            # Crops the mosaic to the focal chunk extent
            filled_crop = filled_data[
                          int(crop_window.row_off):int(crop_window.row_off + crop_window.height),
                          int(crop_window.col_off):int(crop_window.col_off + crop_window.width)
                          ]


        transform = rasterio.transform.from_origin(
            west=bounds[0],
            north=bounds[3],
            xsize=mosaic_transform.a,
            ysize=abs(mosaic_transform.e)
        )

        # Updates raster metadata
        profile = src_datasets[0].profile
        profile.update(
            height=int(crop_window.height),
            width=int(crop_window.width),
            transform=transform,
            # nodata=0,  # I want the chunks without any age pixels to show 0s rather than NoData
            compress='lzw'
        )


        ### Part 3: Saves and uploads the output raster

        lu.print_and_log(f" Saving and uploading {bounds_str}: {uu.timestr()}", is_final, logger_worker)
        uu.rename_s3_task_file(stage, bounds, "uploading_", is_final, logger_worker)

        if is_final:
            file_name = f"{tile_id}__{bounds_str}__{cn.forest_age_2015_interpolated_pattern}.tif"
        else:
            file_name = f"{tile_id}__{bounds_str}__{cn.forest_age_2015_interpolated_pattern}__{uu.timestr()}.tif"

        # Saves filled in focal chunk locally
        if run_local and no_upload:
            output_tmp_path = f"/mnt/c/GIS/AFOLU_flux_model/forest_age/filled_in/{file_name}"
        else:
            output_tmp_path = f"/tmp/{file_name}"

        with rasterio.open(output_tmp_path, "w", **profile) as dst:
            dst.write(filled_crop, 1)

        # Optional: Uploads to S3
        s3_path = f"{output_dir}{file_name}"

        if not no_upload:
            with fs.open(s3_path, "wb") as f_out:
                with open(output_tmp_path, "rb") as f_in:
                    f_out.write(f_in.read())

        return_message = f"Success creating filled forest age for {bounds_str}: {uu.timestr()}"

        chunk_stats.append(uu.calculate_stats(filled_crop, cn.forest_age_2015_interpolated_pattern, bounds_str, tile_id, 'output_layer'))

        # Removes task tracking file from S3 once task is successful
        uu.delete_s3_task_file(stage, bounds, is_final, logger_worker)

    except Exception as e:

        return_message = f"Error processing chunk {bounds}: {e}: {uu.timestr()}"

        lu.print_and_log(return_message, False, logger_worker)
        uu.rename_s3_task_file(stage, bounds, "error_", is_final, logger_worker)

    return return_message, chunk_stats


def main(cluster_name, run_local=False, no_stats=False, no_log=False, no_upload=False,
         use_shapefile=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    age_years = [2015]  # Could also apply to age in 2000 once that is ready
    stage = f'interpolate_forest_age_{age_years[0]}__1x1_deg'
    model_type = 'standard'

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(bounding_box, "N/A", client, cluster, log_note, run_local, model_type, stage)

    # Starting time for stage
    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Years for age maps: {age_years}")

    # Returns a dataframe of chunk_id and ISO for the GADM3.6 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso()

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, use_shapefile, chunk_size, first_chunks, fishnet_iso_df, main_logger)

    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_final = False
    if len(chunk_list) > 20:
        is_final = True
        main_logger.info("Running as final model.")

    # Creates list of output directories specific to the run
    output_dir_list = [cn.forest_age_2015_interpolated_dir]
    output_dir_list = [path.replace("CHUNK_SIZE", str(chunk_size_pixels)) for path in output_dir_list]


    ### Step 2: Create 1x1 degree outputs

    # Creates list of tasks to run (1 task = 1 chunk)
    main_logger.info(f"Creating tasks and starting processing: {uu.timestr()}")
    main_logger.info("Workers' logs to be appended after main function log"+ "\n")

    # Makes a txt for each task in the list. These are deleted as tasks are completed.
    main_logger.info("Creating task txts in s3...")
    uu.create_s3_task_files(stage, chunk_list)

    delayed_results_1x1_deg = [dask.delayed(interpolate_starting_forest_age)
                       (chunk, is_final, no_upload, output_dir_list, stage)
                       for chunk in chunk_list]

    # Runs analysis and gathers results
    results_1x1_deg = dask.compute(*delayed_results_1x1_deg)

    success_count_1x1, all_1x1_stats = uu.count_successful_chunks(chunk_list, is_final, main_logger, results_1x1_deg)

    # Iterates through output folders and counts the number of output rasters (only if uploads enabled)
    if not no_upload:
        for output_folder in output_dir_list:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
            main_logger.info(f"Output rasters in {output_folder}: {file_count}")

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


    ### Step 3: Chunk stats for 1x1 degree outputs, aggregates logs

    # Resizes cluster down to 1 worker for chunk stats and log aggregation since that only needs a minimal remainder of the
    # cluster, not all the workers.
    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        # TODO Or maybe just have it terminate the cluster altogether, rather than resize it. Need to make sure that chunk stats and log still work, though.
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster("AFOLU_flux_model_scripts", 1)

    # Prepares 1x1 deg chunk stats spreadsheet: min, mean, max, and sum for all input and output chunks,
    # and min and max values across all chunks for all inputs and outputs
    # only if not suppressed by the --no_stats flag and at least one chunk was successfully (wasn't skipped).
    if (not no_stats) and (success_count_1x1 > 0):
        uu.aggregate_1x1_chunk_stats(all_1x1_stats, stage, no_upload, main_logger)

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats", main_logger)

    # Sets it so that no worker logs are created if doing a local run
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats and worker log compilation", main_logger)

        # Adds the workers' logs to the main log and uploads to s3
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)

    # Closes the Dask client if not running locally
    if not run_local:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create carbon pools in 2000.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cshp', '--use_shapefile', action='store_true', help='Use shapefile to determine chunks')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    bounding_box = args.bounding_box
    chunk_size = args.chunk_size
    use_shapefile = args.use_shapefile
    first_chunks = args.first_chunks
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    main(cluster_name, run_local, no_stats, no_log, no_upload, use_shapefile,
         bounding_box=bounding_box, chunk_size=chunk_size,
         first_chunks=first_chunks, log_note=log_note)
