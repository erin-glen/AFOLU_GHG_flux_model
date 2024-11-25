"""
Run from src/LULUCF

Test:
For tile indexes:
python -m scripts.utilities.create_cluster -n 15  #15 workers with 3 threads each should be able to make tile indexes for all 44 outputs in one pass
python -m scripts.core_model.LULUCF_fluxes_postprocessing -cn AFOLU_flux_model_scripts -d 20241102

For 1x1 deg chunk aggregation:
python -m scripts.utilities.create_cluster -n 200
python -m scripts.postprocessing.LULUCF_fluxes_global_totals -cn AFOLU_flux_model_scripts -d 20241121

"""


import argparse
import dask
import rasterio
import numpy as np
import boto3

from dask.distributed import print
import os
import re

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import log_utilities as lu
from ..utilities import universal_utilities as uu

# Per https://chatgpt.com/share/e/674105d3-6924-800a-ba00-a942ca95ac32
def global_totals_local(tile_to_process_uri, no_upload):

    is_final = False
    logger = lu.setup_logging()

    # Extract directory path and file name
    bucket_and_path = tile_to_process_uri[len("s3://"):]
    bucket_and_dir, file_name = os.path.split(bucket_and_path)

    # print(bucket_and_dir)
    # print(file_name)
    tile_id = re.search(cn.tile_id_pattern, file_name)[0]
    # print(tile_id)

    bounds = re.search(cn.small_chunk_pattern, file_name)[0]
    # print(bounds)
    bounds_list = list(map(int, re.findall(r"-?\d+", bounds)))
    # print(bounds_list)

    pixel_area_uri = f"{cn.pixel_area_path}{cn.pixel_area_pattern}_{tile_id}.tif"
    # print(pixel_area_uri)

    pixel_area_chunk = uu.get_tile_dataset_rio(pixel_area_uri, 'Float32', bounds_list, 4000, is_final, logger)
    tile_to_process_chunk = uu.get_tile_dataset_rio(tile_to_process_uri, 'Float32', bounds_list, 4000, is_final, logger)
    # print(pixel_area_chunk)
    # print(tile_to_process_chunk)

    tile_to_process_chunk_per_pixel = tile_to_process_chunk * pixel_area_chunk * cn.m2_to_ha
    # print(tile_to_process_chunk_per_pixel)

    chunk_per_ha_average = {"chunk_per_ha_average": np.mean(tile_to_process_chunk)}
    chunk_per_ha_min = {"chunk_per_ha_min": np.min(tile_to_process_chunk)}
    chunk_per_ha_max = {"chunk_per_ha_max": np.max(tile_to_process_chunk)}
    chunk_per_pixel_sum = {"chunk_per_pixel_sum": np.sum(tile_to_process_chunk_per_pixel)}
    chunk_non_zero_count = {"chunk_non_zero_count": np.count_nonzero(tile_to_process_chunk_per_pixel)}

    print(chunk_per_ha_average)
    print(chunk_per_ha_min)
    print(chunk_per_ha_max)
    print(chunk_per_pixel_sum)
    print(chunk_non_zero_count)

    return chunk_per_ha_average, chunk_per_ha_min, chunk_per_ha_max, chunk_per_pixel_sum, chunk_non_zero_count






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
    # Folders to process
    #TODO centralize the output folder list in constants_and_names and reuse in other post-processing steps
    LULUCF_output_folders = [
        # f"{cn.outputs_path}{cn.AGC_density_path_part}/2005/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.AGC_density_path_part}/2010/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.AGC_density_path_part}/2015/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.AGC_density_path_part}/2020/40000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.BGC_density_path_part}/2005/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.BGC_density_path_part}/2010/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.BGC_density_path_part}/2015/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.BGC_density_path_part}/2020/40000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.deadwood_c_density_path_part}/2005/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.deadwood_c_density_path_part}/2010/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.deadwood_c_density_path_part}/2015/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.deadwood_c_density_path_part}/2020/40000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.litter_c_density_path_part}/2005/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.litter_c_density_path_part}/2010/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.litter_c_density_path_part}/2015/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.litter_c_density_path_part}/2020/40000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.agc_net_flux_pattern}/2000_2005/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.agc_net_flux_pattern}/2005_2010/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.agc_net_flux_pattern}/2010_2015/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.agc_net_flux_pattern}/2015_2020/40000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.bgc_net_flux_pattern}/2000_2005/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.bgc_net_flux_pattern}/2005_2010/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.bgc_net_flux_pattern}/2010_2015/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.bgc_net_flux_pattern}/2015_2020/40000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.deadwood_c_net_flux_pattern}/2000_2005/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.deadwood_c_net_flux_pattern}/2005_2010/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.deadwood_c_net_flux_pattern}/2010_2015/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.deadwood_c_net_flux_pattern}/2015_2020/40000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.litter_c_net_flux_pattern}/2000_2005/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.litter_c_net_flux_pattern}/2005_2010/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.litter_c_net_flux_pattern}/2010_2015/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.litter_c_net_flux_pattern}/2015_2020/40000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.ch4_flux_pattern}/2000_2005/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.ch4_flux_pattern}/2005_2010/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.ch4_flux_pattern}/2010_2015/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.ch4_flux_pattern}/2015_2020/40000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.n2o_flux_pattern}/2000_2005/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.n2o_flux_pattern}/2005_2010/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.n2o_flux_pattern}/2010_2015/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.n2o_flux_pattern}/2015_2020/40000_pixels/{date}/",



        # f"{cn.outputs_path}{cn.gross_emis_all_C_pools_CO2_only_pattern}/2000_2005/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.gross_emis_all_C_pools_CO2_only_pattern}/2005_2010/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.gross_emis_all_C_pools_CO2_only_pattern}/2010_2015/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.gross_emis_all_C_pools_CO2_only_pattern}/2015_2020/40000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.gross_emis_non_CO2_only_pattern}/2000_2005/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.gross_emis_non_CO2_only_pattern}/2005_2010/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.gross_emis_non_CO2_only_pattern}/2010_2015/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.gross_emis_non_CO2_only_pattern}/2015_2020/40000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.gross_emis_all_C_pools_all_gases_pattern}/2000_2005/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.gross_emis_all_C_pools_all_gases_pattern}/2005_2010/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.gross_emis_all_C_pools_all_gases_pattern}/2010_2015/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.gross_emis_all_C_pools_all_gases_pattern}/2015_2020/40000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.gross_removals_all_C_pools_pattern}/2000_2005/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.gross_removals_all_C_pools_pattern}/2005_2010/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.gross_removals_all_C_pools_pattern}/2010_2015/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.gross_removals_all_C_pools_pattern}/2015_2020/40000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.net_flux_all_C_pools_CO2_only_pattern}/2000_2005/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.net_flux_all_C_pools_CO2_only_pattern}/2005_2010/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.net_flux_all_C_pools_CO2_only_pattern}/2010_2015/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.net_flux_all_C_pools_CO2_only_pattern}/2015_2020/40000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/2000_2005/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/2005_2010/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/2010_2015/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/2015_2020/4000_pixels/{date}/"

        # f"{cn.outputs_path}{cn.land_state_node_path_part}/2000_2005/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.land_state_node_path_part}/2005_2010/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.land_state_node_path_part}/2010_2015/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.land_state_node_path_part}/2015_2020/40000_pixels/{date}/"
    ]

    tiles_to_process = []

    # Iterates through folders to process to create a consolidated list of tiles to process, with the count of them
    for LULUCF_output_folder in LULUCF_output_folders:

        # geotiff_files = uu.list_raster_names_in_folder(LULUCF_output_folder)

        geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(LULUCF_output_folder)
        # print(geotiff_files)
        lu.print_and_log(f"Output rasters in {LULUCF_output_folder} to process: {file_count}", is_final, logger)
        tiles_to_process.append(geotiff_files)

    # Converts nested list of tiles [[...], [...], [...],...] to flat list [...]
    tiles_to_process = uu.flatten_list(tiles_to_process)

    # print(tiles_to_process)
    # print(tiles_to_process[-100:])
    lu.print_and_log(f"Tiles to process: {len(tiles_to_process)}", is_final, logger)
    # os.quit()

    # For testing. Limits the number of output rasters
    tiles_to_process = tiles_to_process[0:1]  # First 1 tile
    # list_of_s3_name_dicts_total = tiles_to_process[0:3]  # First 3 tiles
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[40:41] # 10N_130E; Internal chunks missing and padding needed on right; FID40
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[0:2]  # 00N_000E
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[16:17] # 00N_110E
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[41:42]  # 10S_010E; No padding needed; FID41
    # print(list_of_s3_name_dicts_total)

    # delayed_result = [dask.delayed(uu.merge_small_tiles_gdal)(s3_name_dict, no_upload) for s3_name_dict in list_of_s3_name_dicts_total]
    delayed_result = [dask.delayed(global_totals_local)(tile_to_process, no_upload) for tile_to_process in tiles_to_process]

    results = dask.compute(*delayed_result)

    # Filter out None values in case of errors
    results = [res for res in results if res is not None]
    print(results)

    # # Sum the totals
    # total_sum = sum(results)
    # print(f"Total sum of all raster values: {total_sum}")
    #
    # # Ending time for stage
    # end_time = uu.timestr()
    # print(f"Stage {stage} ended at: {end_time}")
    # uu.stage_duration(start_time, end_time, stage)

    # # Creates combined log if not deactivated
    # log_note = f"{stage} run"
    # lu.compile_and_upload_log(no_log, client, cluster, stage, 0, '10x10deg', start_time, end_time, end_time,
    #                           0, 0, 'N/A', log_note)

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