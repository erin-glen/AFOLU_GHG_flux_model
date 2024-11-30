"""
Run from src/LULUCF

# 30 workers with 10 threads each should be able to process 315 outputs simultaneously (35 workers * (8 threads/worker + 1 bonus thread that's always there)).
# Making 10x10 aggregate tiles takes basically no memory, so each worker can handle lots of tasks at the same time, it seems.
# 20 and 40 threads/worker caused many gdal_merge errors. Even 10 threads/worker caused at least one error.
python -m scripts.utilities.create_cluster -n 35 -t 8
python -m scripts.postprocessing.LULUCF_fluxes_aggregate_to_10x10deg -cn AFOLU_flux_model_scripts -d 20241121

"""

import argparse
import dask

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import log_utilities as lu
from ..utilities import universal_utilities as uu


def main(cluster_name, date, run_local=False, no_upload=False, no_log=False):

    logger = lu.setup_logging()

    is_final = False

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Model stage being running
    stage = 'drainage_postprocessing__outputs_aggregated_to_10x10deg'

    # Starting time for stage
    start_time = uu.timestr()
    #TODO Logging in any of the main() functions does not work; nothing sent to logger
    lu.print_and_log(f"Stage {stage} started at: {start_time}", is_final, logger)

    # Folders to process
    #TODO centralize the output folder list in constants_and_names and reuse in other post-processing steps
    drainage_output_folders = [
        f"{cn.s3_out_dir}/total_emissions/2020/8000_pixels/{date}/",
        f"{cn.s3_out_dir}/soil/2020/8000_pixels/{date}/",
        f"{cn.s3_out_dir}/state/2020/8000_pixels/{date}/"
    ]

    # Starting time for stage
    start_time = uu.timestr()
    lu.print_and_log(f"Stage {stage} started at: {start_time}", is_final, logger)

    # Creates the list of aggregated 10x10 rasters that will be created (list of dictionaries of input s3 folder and output aggregated raster name.
    # These are the basis for the tasks.
    list_of_s3_name_dicts_total = uu.create_list_for_aggregation(drainage_output_folders)

    # For testing. Limits the number of output rasters
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[0:1]  # First 1 tile
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[0:3]  # First 3 tiles
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[40:41] # 10N_130E; Internal chunks missing and padding needed on right; FID40
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[0:2]  # 00N_000E
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[16:17] # 00N_110E
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[41:42]  # 10S_010E; No padding needed; FID41
    # print(list_of_s3_name_dicts_total)

    # TODO Output as COGs, not just geotifs? Need to ask AJ first.
    # Each task is a single 10x10 deg aggregated geotif
    delayed_result = [dask.delayed(uu.merge_small_tiles_gdal)(s3_name_dict, is_final, no_upload, no_log) for s3_name_dict in list_of_s3_name_dicts_total]

    results = dask.compute(*delayed_result)
    lu.print_and_log(results, is_final, logger)

    drainage_aggreg_folders = [path.replace("4000", "80000") for path in drainage_output_folders]

    for drainage_aggreg_folder in drainage_aggreg_folders:

        geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(drainage_aggreg_folder)
        # print(geotiff_files)
        lu.print_and_log(f"Aggregated 10x10 deg outputs in {drainage_aggreg_folder}: {file_count}", is_final, logger)

    # Ending time for stage
    end_time = uu.timestr()
    lu.print_and_log(f"Stage {stage} ended at: {end_time}", is_final, logger)
    uu.stage_duration(start_time, end_time, stage)

    # Creates combined log if not deactivated
    #TODO outputs log file with header and footer but no other info (filtered content is empty). Don't know why.
    log_note = f"{stage} run"
    # # TODO fix the compile and upload log function
    # lu.compile_and_upload_log(no_log, client, cluster, stage, len(list_of_s3_name_dicts_total), '10x10deg', start_time, end_time, end_time,
    #                           0, 0, 'N/A', log_note)

    if not run_local:
        # Closes the Dask client if not running locally
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate 1x1 degree outputs from LULUCF model to 10x10 degree geotifs.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-d', '--date', help='Date in YYYYMMDD to process')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    # Create the cluster with command line arguments
    main(args.cluster_name, args.date, args.run_local, args.no_upload, args.no_log)