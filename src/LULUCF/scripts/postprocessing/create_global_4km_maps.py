"""
Run from src/LULUCF

python -m scripts.utilities.create_cluster -n 4 -m 32 -c 4 -t 1 -i 10 -cn global_4km_raster_test
python -m scripts.postprocessing.create_global_4km_maps -cn global_4km_raster_test

"""
import numpy as np
import argparse
import dask
from dask.distributed import print
from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu
from ..utilities import log_utilities as lu

########################################################################################################################

def agg_4x4(tile_id, bounds, chunk_length_pixels, pixel_area_tile, mg_ha_yr_tile, per_pixel_output_tile, per_pixel_output_path):

    is_final = False
    logger = lu.setup_logging()



    # Gets numpy arrays of the model output being analyzed and the area (m^2) per pixel
    print(f"Getting rasters for {tile_id}: \n {pixel_area_tile} \n {mg_ha_yr_tile}")
    pixel_area_tile_chunk = uu.get_tile_dataset_rio(pixel_area_tile, 'Float32', bounds, chunk_length_pixels, is_final, logger)
    mg_ha_yr_tile_chunk = uu.get_tile_dataset_rio(mg_ha_yr_tile, 'Float32', bounds, chunk_length_pixels, is_final, logger)

    # Converts per hectare values to per pixel values in the numpy array
    mg_yr_per_pixel_tile_chunk = mg_ha_yr_tile_chunk * pixel_area_tile_chunk * cn.m2_to_ha

    # Upload per-pixel raster to s3
    data_type = mg_yr_per_pixel_tile_chunk.dtype.name
    uu.save_and_upload_single_raster(bounds, chunk_length_pixels, tile_id, mg_yr_per_pixel_tile_chunk, data_type,
                                     per_pixel_output_tile, per_pixel_output_path, is_final, logger)
    #TODO Eventually create per-pixel tile and upload to s3 in a different script

    # Reaggregate into 0.04x0.04 degree resolution
    mg_yr_per_pixel_agg_4x4_tile_chunk = uu.reaggregate_resolution(mg_yr_per_pixel_tile_chunk, 0.00025, 0.04)

    return mg_yr_per_pixel_agg_4x4_tile_chunk


def combine_global_raster(tiles, bounds_list, tile_id, global_4km_outfile, global_4km_output_path):
    #Courtest of chatGPT
    """
    Combines multiple 0.04x0.04 degree tiles into a single global raster.

    Parameters:
        tiles (list): List of numpy arrays for all tiles.
        bounds_list (list): List of bounds corresponding to each tile.

    Returns:
        numpy array: Combined global raster.
    """

    is_final = False
    logger = lu.setup_logging()




    # Define global raster size (360x180 degrees at 0.04 resolution)
    global_shape = (3600, 7200)
    global_raster = np.zeros(global_shape, dtype=np.float32)

    # Insert each tile into the global raster
    for tile, bounds in zip(tiles, bounds_list):
        min_x, min_y, max_x, max_y = bounds
        x_start = int((min_x + 180) / 0.04)
        y_start = int((min_y + 90) / 0.04)
        x_end = x_start + tile.shape[1]
        y_end = y_start + tile.shape[0]

        # Insert the tile into the global raster
        global_raster[y_start:y_end, x_start:x_end] += tile

    # Save the global raster
    global_bounds = (-180, -90, 180, 90)
    uu.save_and_upload_single_raster(global_bounds, global_raster.shape[1], tile_id, global_raster,
                                     np.float32, global_4km_outfile, global_4km_output_path, is_final,
                                     logger)

    return "Success"
    #TODO update with checking to see if the file exists in s3


def main(cluster_name):
    # -------------------------------------------------------------------------------------------------------------------
    # Step 1: Connects to Coiled cluster if not running locally
    run_local = False
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # -------------------------------------------------------------------------------------------------------------------
    # Step 2: Create download/ upload dictionary from list of processes to run
    #TODO: Pass in which fluxes and which years you want to process as command line arguments and add to download_upload_dictionary accordingly.
    # For now hardcoding the download_upload_dictionary. Update flux path/patterns from cn.
    download_upload_dictionary = {
        "cropland_emissions_mg_ha_yr" : {
            'mg_ha_yr_dir': "s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/processed/20241204/year_2020/all_sources/mean_rate/including_peatland/2019/physical_area/",
            'mg_ha_yr_pattern': "_all_GHGs_cropland_mean_rate_physical_area_CO2eq_all_crops_2019_Mg_ha_CO2.tif",
            'mg_per_pixel_dir': "s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/processed/20241204/year_2020/all_sources/per_pixel/including_peatland/2019/physical_area/",
            'mg_per_pixel_pattern': "_all_GHGs_cropland_mean_rate_physical_area_CO2eq_all_crops_2019_Mg_CO2.tif",
            '4km_dir': "s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/processed/20241204/year_2020/all_sources/0_04deg_output_aggregation/cropland_emissions_all_crops_all_gases__MgCO2e_yr/2019/",
            '4km_pattern': '0_04deg_global__all_GHGs_cropland_physical_area_CO2eq_all_crops_2019__MgCO2e_yr.tif'
        },
        "net_flux_2000_2005_mg_ha_yr": {
            'mg_ha_yr_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_ha_yr/2000_2005/40000_pixels/20241203/",
            'mg_ha_yr_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_ha_yr_2000_2005.tif",
            'mg_per_pixel_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_yr/2000_2005/40000_pixels/20241203/",
            'mg_per_pixel_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_yr_2000_2005.tif",
            '4km_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/0_04deg_output_aggregation/net_flux_all_C_pools_all_gases__MgCO2e_yr/2000-2005/",
            '4km_pattern': '0_04deg_global__net_flux_all_C_pools_all_gases__MgCO2e_yr_2000_2005.tif'
        },
        "net_flux_2005_2010_mg_ha_yr": {
            'mg_ha_yr_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_ha_yr/2005_2010/40000_pixels/20241203/",
            'mg_ha_yr_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_ha_yr_2005_2010.tif",
            'mg_per_pixel_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_yr/2005_2010/40000_pixels/20241203/",
            'mg_per_pixel_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_yr_2005_2010.tif",
            '4km_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/0_04deg_output_aggregation/net_flux_all_C_pools_all_gases__MgCO2e_yr/2005-2010/",
            '4km_pattern': '0_04deg_global__net_flux_all_C_pools_all_gases__MgCO2e_yr_2005_2010.tif'
        },
        "net_flux_2010_2015_mg_ha_yr": {
            'mg_ha_yr_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_ha_yr/2010_2015/40000_pixels/20241203/",
            'mg_ha_yr_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_ha_yr_2010_2015.tif",
            'mg_per_pixel_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_yr/2010_2015/40000_pixels/20241203/",
            'mg_per_pixel_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_yr_2010_2015.tif",
            '4km_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/0_04deg_output_aggregation/net_flux_all_C_pools_all_gases__MgCO2e_yr/2010-2015/",
            '4km_pattern': '0_04deg_global__net_flux_all_C_pools_all_gases__MgCO2e_yr_2010_2015.tif'
        },
        "net_flux_2015_2020_mg_ha_yr": {
            'mg_ha_yr_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_ha_yr/2015_2020/40000_pixels/20241203/",
            'mg_ha_yr_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_ha_yr_2015_2020.tif",
            'mg_per_pixel_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_yr/2015_2020/40000_pixels/20241203/",
            'mg_per_pixel_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_yr_2015_2020.tif",
            '4km_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/0_04deg_output_aggregation/net_flux_all_C_pools_all_gases__MgCO2e_yr/2015-2020/",
            '4km_pattern': '0_04deg_global__net_flux_all_C_pools_all_gases__MgCO2e_yr_2015_2020.tif'
        }
    } #TODO REMOVE 4KM PATTERN
        #TODO: After net flux and cropland per pixel tiles are created, update paths in constants and names, remove per-pixel creation step here,
        #  after per-pixel creation step is its own script, remove mg_ha_yr_dir/pattern from download_upload dictionary
    # -------------------------------------------------------------------------------------------------------------------
    # Step 3: Create per-pixel rasters and aggregate into 0.04x0.04 degrees for each tile
    # Model stage being run
    stage = 'create 0.04x0.04 tile rasters'

    # Starting time for stage
    start_time = uu.timestr()
    print(f"Stage {stage} started at: {start_time}")

    # Creating per-pixel rasters
    for key, items in download_upload_dictionary.items():
        bounds_list = []
        delayed_results = []
        for tile_id in cn.tile_id_list:
            mg_ha_yr_tile = f"{items['mg_ha_yr_dir']}{tile_id}{items['mg_ha_yr_pattern']}"
            print(mg_ha_yr_tile)
            pixel_area_tile = f"{cn.pixel_area_path}{cn.pixel_area_pattern}_{tile_id}.tif"
            print(pixel_area_tile)
            per_pixel_tile_outfile = f"{tile_id}{items['mg_per_pixel_pattern']}"
            print(per_pixel_tile_outfile)
            per_pixel_output_path = items["mg_per_pixel_dir"]
            print(per_pixel_output_path)

            # Get bounds and chunk_length_pixels to read in input data
            bounds = uu.get_10x10_tile_bounds(tile_id)
            bounds_list.append(bounds)
            print(bounds)

            chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)
            print(chunk_length_pixels)

            #Submit dask task to create per-pixel raster and aggregate into 0.04x0.04 degrees for each tile
            delayed_results.append(dask.delayed(agg_4x4)(tile_id, bounds, chunk_length_pixels, pixel_area_tile, mg_ha_yr_tile, per_pixel_tile_outfile, per_pixel_output_path))

        #Check
        print(bounds_list)
        print(delayed_results)

        # -------------------------------------------------------------------------------------------------------------------
        # Step 4: Combine 0.04x0.04 degrees tiles into global raster
        # Model stage being run
        stage = 'create 0.04x0.04 degree global rasters'

        # Starting time for stage
        start_time = uu.timestr()
        print(f"Stage {stage} started at: {start_time}")

        # Compute results
        tiles = dask.compute(*delayed_results)
        print(tiles)

        # Combine results into global raster
        tile_id = "0_04deg_global"
        global_4km_outfile = f"{tile_id}{items['mg_per_pixel_pattern']}"
        global_4km_output_path = items['4km_dir']

        global_raster = combine_global_raster(tiles, bounds_list, tile_id, global_4km_outfile, global_4km_output_path)
        print(global_raster)




    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregating AFOLU model output into global ~4km rasters.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')

    args = parser.parse_args()

    # Create the cluster with command line arguments
    main(args.cluster_name)