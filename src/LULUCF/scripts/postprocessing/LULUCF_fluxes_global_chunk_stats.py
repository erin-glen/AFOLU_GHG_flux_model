"""
Run from src/LULUCF

python -m scripts.utilities.create_cluster -n 1 -t 4
python -m scripts.utilities.create_cluster -n 200 -t 4 #In practice, the tasks seem so short that it can't actually use that many threads at once
python -m scripts.postprocessing.LULUCF_fluxes_global_chunk_stats -cn AFOLU_flux_model_scripts -d 20241121

"""

import argparse
import dask
import os
import re

from dask.distributed import print

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import log_utilities as lu
from ..utilities import universal_utilities as uu

# Calculates statistics for 1x1 degree rasters and summarizes them in a spreadsheet
# Per https://chatgpt.com/share/e/674105d3-6924-800a-ba00-a942ca95ac32
# def get_chunk_stats(tile_to_process_uri, fishnet_iso_df):
def get_chunk_stats(tile_to_process_uri):

    is_final = False
    logger = lu.setup_logging()

    # Extracts s3 directory path and file name
    bucket_and_path = tile_to_process_uri[len("s3://"):]
    bucket_and_dir, file_name = os.path.split(bucket_and_path)

    # Extracts the tile_id from the file name
    tile_id = re.search(cn.tile_id_pattern, file_name)[0]

    # Extracts the chunk bounds from the file name (alternatively, could use raster metadata to get chunk bounds but this was simple)
    # Form is __8_-1_9_0__
    bounds = re.search(cn.small_chunk_pattern, file_name)[0][2:-2]  # To drop the __ and __ at the start and end
    # Converts chunk bounds into list, e.g., [8, -1, 9, 0]
    bounds_list = list(map(int, re.findall(r"-?\d+", bounds)))

    # Converts chunk bounds into string
    bounds_str = uu.boundstr(bounds_list)

    lu.print_and_log(f"Calculating chunk stats in {bounds_str} for {file_name} in {tile_id}: {uu.timestr()}", is_final, logger)

    # The relevant pixel area (m^2) file in s3
    pixel_area_uri = f"{cn.pixel_area_path}{cn.pixel_area_pattern}_{tile_id}.tif"

    # Gets numpy arrays of the model output being analyzed and the area (m^2) per pixel
    pixel_area_chunk = uu.get_tile_dataset_rio(pixel_area_uri, 'Float32', bounds_list, 4000, is_final, logger)

    try:
        tile_to_process_chunk_per_ha = uu.get_tile_dataset_rio(tile_to_process_uri, 'Float32', bounds_list, 4000, is_final, logger)
    except Exception as e:
        return f"Failed to pixel area raster for {bounds_list}: {e}"

    # Converts per hectare values to per pixel values in the numpy array
    tile_to_process_chunk_per_pixel = tile_to_process_chunk_per_ha * pixel_area_chunk * cn.m2_to_ha

    #  Calculates stats for the output layers from create_starting_C_densities as a dictionary with chunk attributes,
    # and joins the ISO to each entry
    # stats = uu.calculate_stats(tile_to_process_chunk_per_ha, file_name,
    #                            bounds, tile_id, "output_layer", fishnet_iso_df, tile_to_process_chunk_per_pixel)
    stats = uu.calculate_stats(tile_to_process_chunk_per_ha, file_name,
                               bounds, tile_id, "output_layer", tile_to_process_chunk_per_pixel)

    return stats


def main(cluster_name, date, run_local=False, no_upload=False, no_log=False):

    logger = lu.setup_logging()

    is_final = False

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Model stage being running
    stage = 'LULUCF_flux_postprocessing__chunk_totals'

    # Starting time for stage
    start_time = uu.timestr()
    print(f"Stage {stage} started at: {start_time}")

    # Folders to process
    #TODO centralize the output folder list in constants_and_names and reuse in other post-processing steps
    LULUCF_output_folders = [
        # f"{cn.outputs_path}{cn.AGC_density_path_part}/2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.AGC_density_path_part}/2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.AGC_density_path_part}/2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.AGC_density_path_part}/2020/4000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.BGC_density_path_part}/2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.BGC_density_path_part}/2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.BGC_density_path_part}/2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.BGC_density_path_part}/2020/4000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.deadwood_c_density_path_part}/2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.deadwood_c_density_path_part}/2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.deadwood_c_density_path_part}/2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.deadwood_c_density_path_part}/2020/4000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.litter_c_density_path_part}/2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.litter_c_density_path_part}/2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.litter_c_density_path_part}/2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.litter_c_density_path_part}/2020/4000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.agc_net_flux_pattern}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.agc_net_flux_pattern}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.agc_net_flux_pattern}/2010_2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.agc_net_flux_pattern}/2015_2020/4000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.bgc_net_flux_pattern}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.bgc_net_flux_pattern}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.bgc_net_flux_pattern}/2010_2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.bgc_net_flux_pattern}/2015_2020/4000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.deadwood_c_net_flux_pattern}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.deadwood_c_net_flux_pattern}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.deadwood_c_net_flux_pattern}/2010_2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.deadwood_c_net_flux_pattern}/2015_2020/4000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.litter_c_net_flux_pattern}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.litter_c_net_flux_pattern}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.litter_c_net_flux_pattern}/2010_2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.litter_c_net_flux_pattern}/2015_2020/4000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.ch4_flux_pattern}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.ch4_flux_pattern}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.ch4_flux_pattern}/2010_2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.ch4_flux_pattern}/2015_2020/4000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.n2o_flux_pattern}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.n2o_flux_pattern}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.n2o_flux_pattern}/2010_2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.n2o_flux_pattern}/2015_2020/4000_pixels/{date}/",



        # f"{cn.outputs_path}{cn.gross_emis_all_C_pools_CO2_only_pattern}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.gross_emis_all_C_pools_CO2_only_pattern}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.gross_emis_all_C_pools_CO2_only_pattern}/2010_2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.gross_emis_all_C_pools_CO2_only_pattern}/2015_2020/4000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.gross_emis_non_CO2_only_pattern}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.gross_emis_non_CO2_only_pattern}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.gross_emis_non_CO2_only_pattern}/2010_2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.gross_emis_non_CO2_only_pattern}/2015_2020/4000_pixels/{date}/",

        # This is about 196,272 rasters to analyze!
        f"{cn.outputs_path}{cn.gross_emis_all_C_pools_all_gases_pattern}/2000_2005/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.gross_emis_all_C_pools_all_gases_pattern}/2005_2010/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.gross_emis_all_C_pools_all_gases_pattern}/2010_2015/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.gross_emis_all_C_pools_all_gases_pattern}/2015_2020/4000_pixels/{date}/",

        f"{cn.outputs_path}{cn.gross_removals_all_C_pools_pattern}/2000_2005/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.gross_removals_all_C_pools_pattern}/2005_2010/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.gross_removals_all_C_pools_pattern}/2010_2015/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.gross_removals_all_C_pools_pattern}/2015_2020/4000_pixels/{date}/",

        # f"{cn.outputs_path}{cn.net_flux_all_C_pools_CO2_only_pattern}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.net_flux_all_C_pools_CO2_only_pattern}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.net_flux_all_C_pools_CO2_only_pattern}/2010_2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.net_flux_all_C_pools_CO2_only_pattern}/2015_2020/4000_pixels/{date}/",

        f"{cn.outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/2000_2005/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/2005_2010/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/2010_2015/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/2015_2020/4000_pixels/{date}/"

        # f"{cn.outputs_path}{cn.land_state_node_path_part}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.land_state_node_path_part}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.land_state_node_path_part}/2010_2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.land_state_node_path_part}/2015_2020/4000_pixels/{date}/"
    ]

    tiles_to_process = []

    # Iterates through folders to process to create a consolidated list of tiles to process,
    # with the count of tiles in each folder
    for LULUCF_output_folder in LULUCF_output_folders:

        # geotiff_files = uu.list_raster_names_in_folder(LULUCF_output_folder)

        geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(LULUCF_output_folder)
        # print(geotiff_files)
        lu.print_and_log(f"Output rasters in {LULUCF_output_folder} to process: {file_count}", is_final, logger)
        tiles_to_process.append(geotiff_files)

    # Converts nested list of tiles [[...], [...], [...],...] to flat list [...]
    tiles_to_process = uu.flatten_list(tiles_to_process)

    lu.print_and_log(f"Output rasters to process in {len(LULUCF_output_folders)} folders: {len(tiles_to_process)}", is_final, logger)

    # For testing. Limits the number of output rasters
    # tiles_to_process = tiles_to_process[0:1]  # First 1 tile
    # tiles_to_process = tiles_to_process[0:3]  # First 3 tiles
    # tiles_to_process = tiles_to_process[0:20]  # First 20 tiles
    # tiles_to_process = tiles_to_process[0:100]  # First 100 tiles
    # tiles_to_process = tiles_to_process[15000:15005]  # Some middle tiles
    # tiles_to_process = tiles_to_process[13000:14000]  # Some middle tiles
    # print(tiles_to_process)

    # Returns a dataframe of chunk_id and ISO, to be joined with chunk stats
    fishnet_iso_df = uu.fishnet_with_GADM_iso()

    # For local runs
    if run_local:
        print("Using dask delayed for local run")
        # Distributes tasks and processes them
        delayed_result = [dask.delayed(get_chunk_stats)(tile_to_process, fishnet_iso_df) for tile_to_process in tiles_to_process]
        results = dask.compute(*delayed_result)

    # This approach handles large task lists (graphs) better than [dask.delayed(get_chunk_stats ... )]
    else:
        print("Using futures for large task list")
        futures = []
        for tile_to_process in tiles_to_process:
            # future = client.submit(get_chunk_stats, tile_to_process, fishnet_iso_df)
            future = client.submit(get_chunk_stats, tile_to_process)
            futures.append(future)

        # Collect the results once they are finished
        results = client.gather(futures)

    # Filters out None values in case of errors
    results = [res for res in results if res is not None]
    # print(results)

    # Creates a chunk stats spreadsheet and optionally uploads it to s3
    uu.calculate_chunk_stats(results, stage, no_upload)

    # Ending time for stage
    end_time = uu.timestr()
    print(f"Stage {stage} ended at: {end_time}")
    uu.stage_duration(start_time, end_time, stage)

    # Creates combined log if not deactivated
    if not run_local:
        log_note = f"{stage} run"
        lu.compile_and_upload_log(no_log, client, cluster, stage, len(tiles_to_process), '1x1deg',
                                  start_time, end_time, end_time,
                                  0, 0, 'N/A', log_note)

    if not run_local:
        # Closes the Dask client if not running locally
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate LULUCF fluxes.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-d', '--date', help='Date in YYYYMMDD to process')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    # Create the cluster with command line arguments
    main(args.cluster_name, args.date, args.run_local, args.no_upload, args.no_log)