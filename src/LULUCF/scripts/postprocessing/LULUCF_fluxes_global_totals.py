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
import s3fs

from dask.distributed import print

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import log_utilities as lu
from ..utilities import universal_utilities as uu


def global_totals_local(s3_name_dict, no_upload):
    """
    Downloads the relevant raster and pixel_area raster using s3fs, multiplies them, and calculates the sum of the resulting raster.

    Args:
        s3_name_dict (dict): Dictionary containing paths to relevant raster and pixel_area raster.
        no_upload (bool): If True, disables uploading results to S3.

    Returns:
        float: Sum of all pixel values in the multiplied raster.
    """
    # Initialize S3 filesystem
    s3 = s3fs.S3FileSystem()

    try:
        # Step 1: Access the input raster
        raster_path = f"s3://{s3_name_dict['bucket']}/{s3_name_dict['input_raster']}"
        pixel_area_path = f"s3://{s3_name_dict['bucket']}/{s3_name_dict['pixel_area_raster']}"

        # Step 2: Open rasters with rasterio using s3fs
        with rasterio.open(s3.open(raster_path, mode='rb')) as raster, \
             rasterio.open(s3.open(pixel_area_path, mode='rb')) as pixel_area:

            # Ensure rasters have the same dimensions and transformation
            if raster.shape != pixel_area.shape:
                raise ValueError("Raster and pixel area raster dimensions do not match.")

            raster_data = raster.read(1)  # Read the first band
            pixel_area_data = pixel_area.read(1)  # Read the first band

            # Multiply raster data with pixel area
            multiplied_data = raster_data * pixel_area_data

            # Calculate the sum of the multiplied raster
            total_sum = np.nansum(multiplied_data)  # Use NaN-safe summation

        return total_sum

    except Exception as e:
        print(f"Error processing {s3_name_dict}: {e}")
        return None


def main(cluster_name, date, run_local=False, no_upload=False, no_log=False):

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Model stage being running
    stage = 'LULUCF_flux_postprocessing__global_totals'

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
        # f"{cn.outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/2010_2015/40000_pixels/{date}/",
        f"{cn.outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/2015_2020/40000_pixels/{date}/"

        # f"{cn.outputs_path}{cn.land_state_node_path_part}/2000_2005/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.land_state_node_path_part}/2005_2010/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.land_state_node_path_part}/2010_2015/40000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.land_state_node_path_part}/2015_2020/40000_pixels/{date}/"
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

    # delayed_result = [dask.delayed(uu.merge_small_tiles_gdal)(s3_name_dict, no_upload) for s3_name_dict in list_of_s3_name_dicts_total]
    delayed_result = [dask.delayed(global_totals_local)(s3_name_dict, no_upload) for s3_name_dict in list_of_s3_name_dicts_total]

    results = dask.compute(*delayed_result)

    # Filter out None values in case of errors
    results = [res for res in results if res is not None]

    # Sum the totals
    total_sum = sum(results)
    print(f"Total sum of all raster values: {total_sum}")

    # Ending time for stage
    end_time = uu.timestr()
    print(f"Stage {stage} ended at: {end_time}")
    uu.stage_duration(start_time, end_time, stage)

    # Creates combined log if not deactivated
    log_note = f"{stage} run"
    lu.compile_and_upload_log(no_log, client, cluster, stage, 0, '10x10deg', start_time, end_time, end_time,
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