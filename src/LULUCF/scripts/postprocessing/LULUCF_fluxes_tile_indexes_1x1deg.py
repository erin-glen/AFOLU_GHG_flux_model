"""
Run from src/LULUCF

# 6 workers with 5 threads each should be able to process 39 outputs in one pass (3 workers * (12 threads/worker + 1 bonus thread that's always there)).
# Making tile indexes takes basically no memory, so each worker can handle lots of tasks at the same time, it seems.
python -m scripts.utilities.create_cluster -n 3 -t 12
python -m scripts.postprocessing.LULUCF_fluxes_tile_indexes_1x1deg -cn AFOLU_flux_model_scripts -d 20241121
"""


import argparse
import dask

from dask.distributed import print

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import log_utilities as lu
from ..utilities import universal_utilities as uu

def main(cluster_name, date, run_local=False, no_upload=False, no_log=False):

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Model stage being running
    stage = 'LULUCF_flux_postprocessing__tile_index_1x1_deg'

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

        f"{cn.outputs_path}{cn.gross_emis_all_C_pools_CO2_only_pattern}/2000_2005/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.gross_emis_all_C_pools_CO2_only_pattern}/2005_2010/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.gross_emis_all_C_pools_CO2_only_pattern}/2010_2015/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.gross_emis_all_C_pools_CO2_only_pattern}/2015_2020/4000_pixels/{date}/",

        f"{cn.outputs_path}{cn.gross_emis_non_CO2_only_pattern}/2000_2005/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.gross_emis_non_CO2_only_pattern}/2005_2010/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.gross_emis_non_CO2_only_pattern}/2010_2015/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.gross_emis_non_CO2_only_pattern}/2015_2020/4000_pixels/{date}/",

        f"{cn.outputs_path}{cn.gross_emis_all_C_pools_all_gases_pattern}/2000_2005/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.gross_emis_all_C_pools_all_gases_pattern}/2005_2010/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.gross_emis_all_C_pools_all_gases_pattern}/2010_2015/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.gross_emis_all_C_pools_all_gases_pattern}/2015_2020/4000_pixels/{date}/",

        f"{cn.outputs_path}{cn.gross_removals_all_C_pools_pattern}/2000_2005/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.gross_removals_all_C_pools_pattern}/2005_2010/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.gross_removals_all_C_pools_pattern}/2010_2015/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.gross_removals_all_C_pools_pattern}/2015_2020/4000_pixels/{date}/",

        f"{cn.outputs_path}{cn.net_flux_all_C_pools_CO2_only_pattern}/2000_2005/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.net_flux_all_C_pools_CO2_only_pattern}/2005_2010/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.net_flux_all_C_pools_CO2_only_pattern}/2010_2015/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.net_flux_all_C_pools_CO2_only_pattern}/2015_2020/4000_pixels/{date}/",

        f"{cn.outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/2000_2005/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/2005_2010/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/2010_2015/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/2015_2020/4000_pixels/{date}/",

        f"{cn.outputs_path}{cn.land_state_node_path_part}/2000_2005/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.land_state_node_path_part}/2005_2010/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.land_state_node_path_part}/2010_2015/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.land_state_node_path_part}/2015_2020/4000_pixels/{date}/"
    ]

    # Creates dictionary of s3 tile set paths with corresponding tile index shapefile names
    s3_in_folders_list_of_dicts = []

    for path in LULUCF_output_folders:
        # Extracts the portion after 'cn.outputs_path'
        path_suffix = path.replace(cn.outputs_path, "")

        # Replaces '/' with '__'
        value = path_suffix.rstrip('/').replace("/", "__")

        s3_in_folders_list_of_dicts.append({path: value})

    # Make raster footprint shapefiles from output rasters
    # Takes over 1 hour on global LULUCF output 1x1 tile set
    delayed_result = [dask.delayed(uu.make_tile_footprint_shp)(input_dict, no_upload) for input_dict in s3_in_folders_list_of_dicts]

    # Actually runs analysis
    results = dask.compute(*delayed_result)
    print(results)

    # Ending time for stage
    end_time = uu.timestr()
    print(f"Stage {stage} ended at: {end_time}")
    uu.stage_duration(start_time, end_time, stage)

    # Creates combined log if not deactivated
    #TODO log for this stage is untested.
    log_note = f"{stage} run"
    lu.compile_and_upload_log(no_log, client, cluster, stage, 0, '1x1deg', start_time, end_time, end_time,
                              'N/A', 'N/A', 'N/A', log_note)

    if not run_local:
        # Closes the Dask client if not running locally
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create tile index shapefiles of 1x1 degree geotif outputs from LULUCF model.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-d', '--date', help='Date in YYYYMMDD to process')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    # Create the cluster with command line arguments
    main(args.cluster_name, args.date, args.run_local, args.no_upload, args.no_log)