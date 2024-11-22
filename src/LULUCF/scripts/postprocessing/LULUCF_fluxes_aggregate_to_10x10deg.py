"""
Run from src/LULUCF

For 1x1 deg output aggregation:
python -m scripts.utilities.create_cluster -n 100
python -m scripts.postprocessing.LULUCF_fluxes_aggregate_to_10x10deg -cn AFOLU_flux_model_scripts -d 20241121

"""


import argparse
import dask

from dask.distributed import print

# Project imports
from src.LULUCF.scripts.utilities import constants_and_names as cn
from src.LULUCF.scripts.utilities import universal_utilities as uu
from src.LULUCF.scripts.utilities import resize_cluster

def main(cluster_name, date, run_local=False, no_upload=False):

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Model stage being running
    stage = 'LULUCF_flux_postprocessing__aggregated_outputs_to_10x10deg'

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
        f"{cn.outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/2015_2020/4000_pixels/{date}/",

        f"{cn.outputs_path}{cn.land_state_node_path_part}/2000_2005/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.land_state_node_path_part}/2005_2010/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.land_state_node_path_part}/2010_2015/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.land_state_node_path_part}/2015_2020/4000_pixels/{date}/"
    ]

    # Starting time for stage
    start_time = uu.timestr()
    print(f"Stage {stage} started at: {start_time}")

    # Creates the list of aggregated 10x10 rasters that will be created (list of dictionaries of input s3 folder and output aggregated raster name.
    # These are the basis for the tasks.
    list_of_s3_name_dicts_total = uu.create_list_for_aggregation(LULUCF_output_folders)

    # For testing. Limits the number of output rasters
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[0:3]  # First 3 tiles
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[40:41] # 10N_130E; Internal chunks missing and padding needed on right; FID40
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[0:2]  # 00N_000E
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[16:17] # 00N_110E
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[41:42]  # 10S_010E; No padding needed; FID41
    # print(list_of_s3_name_dicts_total)

    delayed_result = [dask.delayed(uu.merge_small_tiles_gdal)(s3_name_dict, no_upload) for s3_name_dict in list_of_s3_name_dicts_total]

    results = dask.compute(*delayed_result)
    print(results)

    # Ending time for stage
    end_time = uu.timestr()
    print(f"Stage {stage} ended at: {end_time}")
    uu.stage_duration(start_time, end_time, stage)

    if not run_local:
        # Closes the Dask client if not running locally
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate 1x1 degree outputs from LULUCF model to 10x10 degree geotifs.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-d', '--date', help='Date in YYYYMMDD to process')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    # Create the cluster with command line arguments
    main(args.cluster_name, args.date, args.run_local, args.no_upload)