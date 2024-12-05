"""
Run from AFOLU_GHG_flux_model

Local:
python -m src.LULUCF.scripts.postprocessing.create_global_4km_maps -cn create_global_4km_maps -ct local

Test:
python -m src.LULUCF.scripts.postprocessing.create_global_4km_maps -cn create_global_4km_maps -ct test

Full run:
python -m src.LULUCF.scripts.postprocessing.create_global_4km_maps -cn create_global_4km_maps -ct full

#QC
cluster_name = 'create_global_4km_maps'
cluster_type = 'local'

"""
import os
from osgeo import gdal
import numpy as np
import argparse
import subprocess
from dask.distributed import print
from src.LULUCF.scripts.utilities import constants_and_names as cn
from src.LULUCF.scripts.utilities import log_utilities as lu
from src.LULUCF.scripts.utilities import universal_utilities as uu

########################################################################################################################

def main(cluster_name, cluster_type):

    #-------------------------------------------------------------------------------------------------------------------
    # Step 1: Get local or coiled cluster
    logger = lu.setup_logging()
    is_final = False

    if cluster_type == 'full':
        # Full cluster with 40 workers
        client = uu.get_client_from_cluster_type('coiled', cluster_name, 40, 4, "16GiB")
    elif cluster_type == 'test':
        # Test cluster with 1 worker
        client = uu.get_client_from_cluster_type('coiled', cluster_name, 1, 2, "8GiB")
    elif cluster_type == 'local':
        # Local cluster with multiple workers
        client = uu.get_client_from_cluster_type('local')
    else:
        print("set cluster_type to one of the following: 'full', 'test', 'local'")

    client


    # -------------------------------------------------------------------------------------------------------------------
    # Step 2: Create download/ upload dictionary from list of processes to run
    #TODO: Pass in which fluxes and which years you want to process as command line arguments and add to download_upload_dictionary accordingly.
    # For now hardcoding the download_upload_dictionary. Update flux path/patterns from cn.
    download_upload_dictionary = {
        "cropland_emissions_kg_ha_yr" : {
            'tile_dir': "s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/processed/20241204/year_2020/all_sources//mean_rate/including_peatland/2019/physical_area/",
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
    # Model stage being running
    stage = 'convert_cropland_emissions_units_from_kg_to_mg'

    # Starting time for stage
    start_time = uu.timestr()
    print(f"Stage {stage} started at: {start_time}")

    # Input/ output dirs
    cropland_emissions_kg_input_dir = download_upload_dictionary["cropland_emissions_kg_ha_yr"]["tile_dir"]
    cropland_emissions_Mg_output_dir = "s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/processed/20241204/year_2020/all_sources/mean_rate/including_peatland/2019/physical_area/"

    # Get list of all tiles in the cropland emissions kg s3 folder
    cropland_emissions_kg_tiles_list = uu.list_raster_names_in_s3_folder(cropland_emissions_kg_input_dir)

    # Testing:
    cropland_emissions_kg_tiles_list = cropland_emissions_kg_tiles_list[0]
    input_tile = cropland_emissions_kg_tiles_list
    input_folder = cropland_emissions_kg_input_dir
    output_folder = cropland_emissions_Mg_output_dir

    input_tile_path = f"{input_folder}{input_tile}"
    output_tile = input_tile.replace("kg", "Mg")
    output_tile_path = f"{output_folder}{output_tile}"

    # Get bounds and chunk_length_pixels to create numpy array of cropland emissions data
    tile_id = uu.string_to_tile_id(input_tile_path)
    bounds = uu.get_10x10_tile_bounds(tile_id)
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)

    # Open the input raster
    kg_tile_chunk = uu.get_tile_dataset_rio(input_tile_path, 'Float32', bounds, chunk_length_pixels, is_final, logger)

    #Create an array with the conversion value
    conversion_array = np.full(kg_tile_chunk.shape, 1e-3 , dtype=np.float32)

    # Multiply the input tile by the conversion array to get the Mg values
    Mg_tile_chunk = kg_tile_chunk * conversion_array

    # Convert data array to raster
    profile_kwargs = {'compress': 'lzw'}
    Mg_tile_chunk.rio.to_raster(f"/tmp/{output_tile}", **profile_kwargs)

    # Upload raster to s3
    uu.upload_s3_file(output_tile_path, f"/tmp/{output_tile}")

    #Function to convert
    # def kg_to_Mg_conversion(input_tile, input_folder, output_folder):
    #     # Input/ output paths
    #     input_tile_path = f"{input_folder}{input_tile}"
    #     output_tile = input_tile.replace("kg", "Mg")
    #     output_tile_path = f"{output_folder}{output_tile}"
    #
    #     # Get bounds and chunk_length_pixels to create numpy array of cropland emissions data
    #     tile_id = uu.string_to_tile_id(input_tile_path)
    #     bounds = uu.get_10x10_tile_bounds(tile_id)
    #     chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)
    #
    #     # Open the input raster
    #     kg_tile_chunk = uu.get_tile_dataset_rio(input_tile_path, 'Float32', bounds, chunk_length_pixels, is_final, logger)
    #
    #     #Create an array with the conversion value
    #     conversion_array = np.full(kg_tile_chunk.shape, 1e-3 , dtype=np.float32)
    #
    #     # Multiply the input tile by the conversion array to get the Mg values
    #     Mg_tile_chunk = kg_tile_chunk * conversion_array
    #
    #     # Convert data array to raster
    #     profile_kwargs = {'compress': 'lzw'}
    #     Mg_tile_chunk.rio.to_raster(f"/tmp/{output_tile}", **profile_kwargs)
    #
    #     # Upload raster to s3
    #     uu.upload_s3_file(output_tile_path, f"/tmp/{output_tile}")
    #
    # tile_futures = []
    # for tile in cropland_emissions_kg_tiles_list:
    #     tile_future = client.submit(kg_to_Mg_conversion, tile, cropland_emissions_kg_input_dir, cropland_emissions_Mg_output_dir)
    #     tile_futures.append(tile_future)
    #
    # # Collect the results once they are finished
    # tile_results = client.gather(tile_futures)



