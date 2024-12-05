"""
Run from src/LULUCF

python -m scripts.utilities.create_cluster -n 1 -m 32 -cn AFOLU_flux_model_scripts
python -m scripts.postprocessing.create_global_4km_maps -cn AFOLU_flux_model_scripts


"""
import os
from osgeo import gdal
import numpy as np
import argparse
import subprocess
import dask
from dask.distributed import print
from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu
from ..utilities import log_utilities as lu

########################################################################################################################

def agg_4x4(chunk, cropland_emissions_kg_input_dir, cropland_emissions_Mg_output_dir):

    is_final = False

    logger = lu.setup_logging()

    print("In Dask function")

    input_tile = chunk
    print(input_tile)

    input_tile_path = f"{cropland_emissions_kg_input_dir}{input_tile}"
    output_tile = input_tile.replace("kg", "Mg")

    print(input_tile_path)

    # Get bounds and chunk_length_pixels to read in input data
    tile_id = uu.string_to_tile_id(input_tile_path)
    bounds = uu.get_10x10_tile_bounds(tile_id)
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)

    print(tile_id)
    print(bounds)
    print(chunk_length_pixels)

    # Read in the raster
    print("Getting cropland raster")
    kg_tile_chunk = uu.get_tile_dataset_rio(input_tile_path, 'Float32', bounds, chunk_length_pixels, is_final, logger)

    print(kg_tile_chunk)

    # # Create an array with the conversion value
    # conversion_array = np.full(kg_tile_chunk.shape, 1e-3, dtype=np.float32)

    kg_to_Mg = 1e-3

    # print(conversion_array)

    # Multiply the input tile by the conversion array to get the Mg values
    print("Performing kg to Mg conversion")
    Mg_tile_chunk = kg_tile_chunk * kg_to_Mg

    print(Mg_tile_chunk)

    # Upload raster to s3
    data_type = Mg_tile_chunk.dtype.name
    uu.save_and_upload_single_raster(bounds, chunk_length_pixels, tile_id, Mg_tile_chunk, data_type, output_tile,
                                     cropland_emissions_Mg_output_dir, is_final, logger)

def main(cluster_name, cluster_type):

    run_local = False

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    #-------------------------------------------------------------------------------------------------------------------
    # Step 1: Get local or coiled cluster
    logger = lu.setup_logging()
    is_final = False

    # if cluster_type == 'full':
    #     # Full cluster with 40 workers
    #     client = uu.get_client_from_cluster_type('coiled', cluster_name, 40, 4, "16GiB")
    # elif cluster_type == 'test':
    #     # Test cluster with 1 worker
    #     client = uu.get_client_from_cluster_type('coiled', cluster_name, 1, 2, "8GiB")
    # elif cluster_type == 'local':
    #     # Local cluster with multiple workers
    #     client = uu.get_client_from_cluster_type('local')
    # else:
    #     print("set cluster_type to one of the following: 'full', 'test', 'local'")
    #
    # client


    # -------------------------------------------------------------------------------------------------------------------
    # Step 2: Create download/ upload dictionary from list of processes to run
    #TODO: Pass in which fluxes and which years you want to process as command line arguments and add to download_upload_dictionary accordingly.
    # For now hardcoding the download_upload_dictionary. Update flux path/patterns from cn.
    download_upload_dictionary = {
        "cropland_emissions_kg_ha_yr" : {
            'tile_dir': "s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/processed/20241204/year_2020/all_sources/kg/including_peatland/2019/physical_area/",
            # TODO: Fix this path
            'tile_pattern': cn.global_cropland_mean_rate_physical_area_all_crops_peat_2019_processed_pattern,
            '4km_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/0_04deg_output_aggregation/cropland_emissions_all_crops_all_gases__MgCO2e_yr/2020/",
            '4km_pattern': '0_04deg_global__all_GHGs_cropland_physical_area_CO2eq_all_crops_2019__MgCO2e_yr.tif'
        },
        "net_flux_2000_2005_mg_ha_yr": {
            'tile_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_ha_yr/2000_2005/40000_pixels/20241203/",
            'tile_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_ha_yr_2000_2005.tif",
            '4km_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/0_04deg_output_aggregation/net_flux_all_C_pools_all_gases__MgCO2e_yr/2000-2005/",
            '4km_pattern': '0_04deg_global__net_flux_all_C_pools_all_gases__MgCO2e_yr_2000_2005.tif'
        },
        "net_flux_2005_2010_mg_ha_yr": {
            'tile_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_ha_yr/2005_2010/40000_pixels/20241203/",
            'tile_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_ha_yr_2005_2010.tif",
            '4km_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/0_04deg_output_aggregation/net_flux_all_C_pools_all_gases__MgCO2e_yr/2005-2010/",
            '4km_pattern': '0_04deg_global__net_flux_all_C_pools_all_gases__MgCO2e_yr_2005_2010.tif'
        },
        "net_flux_2010_2015_mg_ha_yr": {
            'tile_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_ha_yr/2010_2015/40000_pixels/20241203/",
            'tile_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_ha_yr_2010_2015.tif",
            '4km_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/0_04deg_output_aggregation/net_flux_all_C_pools_all_gases__MgCO2e_yr/2010-2015/",
            '4km_pattern': '0_04deg_global__net_flux_all_C_pools_all_gases__MgCO2e_yr_2010_2015.tif'
        },
        "net_flux_2015_2020_mg_ha_yr": {
            'tile_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_ha_yr/2015_2020/40000_pixels/20241203/",
            'tile_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_ha_yr_2015_2020.tif",
            '4km_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/0_04deg_output_aggregation/net_flux_all_C_pools_all_gases__MgCO2e_yr/2015-2020/",
            '4km_pattern': '0_04deg_global__net_flux_all_C_pools_all_gases__MgCO2e_yr_2015_2020.tif'
        }
    }

    # -------------------------------------------------------------------------------------------------------------------
    # Step 3: Convert cropland emissions from kg per hectare per year to mg per hectare per year
    #TODO: Move this step to hansenize function for cropland emissions
    #Model stage being run
    stage = 'convert_cropland_emissions_units_from_kg_to_mg'

    # Starting time for stage
    start_time = uu.timestr()
    print(f"Stage {stage} started at: {start_time}")

    # Input/ output dirs
    cropland_emissions_kg_input_dir = download_upload_dictionary["cropland_emissions_kg_ha_yr"]["tile_dir"]
    cropland_emissions_Mg_output_dir = "s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/processed/20241204/year_2020/all_sources/mean_rate/including_peatland/2019/physical_area/"

    # Get list of all tiles in the cropland emissions kg s3 folder
    cropland_emissions_kg_tiles_list = uu.list_raster_names_in_s3_folder(cropland_emissions_kg_input_dir)
    cropland_emissions_kg_tiles_list = [cropland_emissions_kg_tiles_list[0]]
    print(cropland_emissions_kg_tiles_list)

    # Creates list of tasks to run (1 task = 1 chunk)
    print(f"Creating tasks and starting processing: {uu.timestr()}")
    delayed_results = [dask.delayed(agg_4x4)(chunk, cropland_emissions_kg_input_dir, cropland_emissions_Mg_output_dir) for chunk in cropland_emissions_kg_tiles_list]

    # Runs analysis and gathers results
    results = dask.compute(*delayed_results)

    print(results)

    client.close()

    # # Submit futures to create and upload cropland emissions tiles converted into Mg units
    # tile_futures = []
    # for tile in cropland_emissions_kg_tiles_list:
    #     tile_future = client.submit(uu.kg_to_Mg_conversion, tile, cropland_emissions_kg_input_dir, cropland_emissions_Mg_output_dir)
    #     tile_futures.append(tile_future)
    #
    # # Collect the results once they are finished
    # tile_results = client.gather(tile_futures)

    # # DECONSTRUCTING kg_to_Mg_conversion FUNCTION FOR TESTING
    # cropland_emissions_kg_tiles_list = cropland_emissions_kg_tiles_list[0]
    # input_tile = cropland_emissions_kg_tiles_list
    # input_folder = cropland_emissions_kg_input_dir
    # output_folder = cropland_emissions_Mg_output_dir
    #
    # input_tile_path = f"{input_folder}{input_tile}"
    # output_tile = input_tile.replace("kg", "Mg")
    #
    # # Get bounds and chunk_length_pixels to read in input data
    # tile_id = uu.string_to_tile_id(input_tile_path)
    # bounds = uu.get_10x10_tile_bounds(tile_id)
    # chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)
    #
    # # Read in the raster
    # print("Getting cropland raster")
    # kg_tile_chunk = uu.get_tile_dataset_rio(input_tile_path, 'Float32', bounds, chunk_length_pixels, is_final, logger)
    #
    # #Create an array with the conversion value
    # conversion_array = np.full(kg_tile_chunk.shape, 1e-3 , dtype=np.float32)
    #
    # # Multiply the input tile by the conversion array to get the Mg values
    # print("Performing kg to Mg conversion")
    # Mg_tile_chunk = kg_tile_chunk * conversion_array
    #
    # # Upload raster to s3
    # data_type = Mg_tile_chunk.dtype.name
    # uu.save_and_upload_single_raster(bounds, chunk_length_pixels, tile_id, Mg_tile_chunk, data_type, output_tile, output_folder, is_final, logger)
    # #TODO: The save_and_upload_single_raster function (modified from save_and_upload_small_raster_set()) is not working
    # # when I run locally I get this error /home/melrose94/anaconda3/envs/coiled_updated/lib/python3.10/multiprocessing/resource_tracker.py:224: UserWarning: resource_tracker: There appear to be 24 leaked semaphore objects to clean up at shutdown
    # # warnings.warn('resource_tracker: There appear to be %d '. Thought this might be a memory issue so I tried running on coiled and it steps into the function,
    # # but the process is killed when it tries to step into rasterio.open


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocessing cropland emissions.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-ct', '--cluster_type', action='store', help='Run locally with Dask (local), test with 1 worker in coiled (test), or run with full coiled cluster (full)')

    args = parser.parse_args()

    # Create the cluster with command line arguments
    main(args.cluster_name, args.cluster_type)