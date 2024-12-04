"""
Run from AFOLU_GHG_flux_model

Local:
python -m src.LULUCF.scripts.preprocessing.hansenize -cn Hansenize_cropland_emissions_data -ct local -p cropland_emissions

Test:
python -m src.LULUCF.scripts.preprocessing.hansenize -cn Hansenize_drivers_data -ct test -p drivers --delete_local_files

Full run:
python -m src.LULUCF.scripts.preprocessing.hansenize -cn Hansenize_drivers_data -ct full -p drivers --delete_local_files

#QC
cluster_name = 'Hansenize_drivers_data'
cluster_type = 'test'
process = ['drivers']
delete_local_files = True


"""
import os
import argparse
from dask.distributed import print
from src.LULUCF.scripts.utilities import constants_and_names as cn
from src.LULUCF.scripts.utilities import universal_utilities as uu

########################################################################################################################

def main(cluster_name, cluster_type, process, delete_local_files):

    #-------------------------------------------------------------------------------------------------------------------
    # Step 1: Get local or coiled cluster
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

    #-------------------------------------------------------------------------------------------------------------------
    #Step 2: Create download/ upload dictionary from list of processes to run
    # Create empty dictionary
    download_upload_dictionary ={}

    # Add drivers data
    if 'drivers' in process:
        download_upload_dictionary["drivers"] = {
            'raw_dir': cn.drivers_raw_dir,
            'raw_pattern': cn.drivers_pattern,
            'vrt': "drivers.vrt",
            'processed_dir': cn.drivers_processed_dir,
            #'processed_dir': "s3://gfw2-data/drivers_of_loss/1_km/processed/coiled_test/",
            'processed_pattern': cn.drivers_pattern
        }

    # Add Robinson et al secondary natural forest growth rates
    if 'secondary_natural_forest' in process:
        download_upload_dictionary["secondary_natural_forest_0_5"] = {
            'raw_dir': cn.secondary_natural_forest_raw_dir,
            'raw_pattern': cn.secondary_natural_forest_0_5_pattern,
            'vrt': "secondary_natural_forest_0_5.vrt",
            'processed_dir': cn.secondary_natural_forest_0_5_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_0_5_pattern
        }

        download_upload_dictionary["secondary_natural_forest_6_10"] = {
            'raw_dir': cn.secondary_natural_forest_raw_dir,
            'raw_pattern': cn.secondary_natural_forest_6_10_pattern,
            'vrt': "secondary_natural_forest_6_10.vrt",
            'processed_dir': cn.secondary_natural_forest_6_10_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_6_10_pattern
        }

        download_upload_dictionary["secondary_natural_forest_11_15"] = {
            'raw_dir': cn.secondary_natural_forest_raw_dir,
            'raw_pattern': cn.secondary_natural_forest_11_15_pattern,
            'vrt': "secondary_natural_forest_11_15.vrt",
            'processed_dir': cn.secondary_natural_forest_11_15_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_11_15_pattern
        }

        download_upload_dictionary["secondary_natural_forest_16_20"] = {
            'raw_dir': cn.secondary_natural_forest_raw_dir,
            'raw_pattern': cn.secondary_natural_forest_16_20_pattern,
            'vrt': "secondary_natural_forest_16_20.vrt",
            'processed_dir': cn.secondary_natural_forest_16_20_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_16_20_pattern
        }

        download_upload_dictionary["secondary_natural_forest_21_100"] = {
            'raw_dir': cn.secondary_natural_forest_raw_dir,
            'raw_pattern': cn.secondary_natural_forest_21_100_pattern,
            'vrt': "secondary_natural_forest_21_100.vrt",
            'processed_dir': cn.secondary_natural_forest_21_100_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_21_100_pattern
        }

    if 'cropland_emissions' in process:
        # download_upload_dictionary["global_cropland_mean_rate_harvest_area_all_crops_peat_2006"] = {
        #     'raw_dir': cn.global_cropland_emissions_raw_dir,
        #     'raw_pattern': cn.global_cropland_mean_rate_harvest_area_all_crops_peat_2006_raw_pattern,
        #     'vrt': "global_cropland_mean_rate_harvest_area_all_crops_peat_2006.vrt",
        #     'processed_dir': cn.global_cropland_mean_rate_harvest_area_all_crops_peat_2006_processed_dir,
        #     'processed_pattern': cn.global_cropland_mean_rate_harvest_area_all_crops_peat_2006_processed_pattern
        # }
        # download_upload_dictionary["global_cropland_mean_rate_harvest_area_all_crops_peat_2019"] = {
        #     'raw_dir': cn.global_cropland_emissions_raw_dir,
        #     'raw_pattern': cn.global_cropland_mean_rate_harvest_area_all_crops_peat_2019_raw_pattern,
        #     'vrt': "global_cropland_mean_rate_harvest_area_all_crops_peat_2019.vrt",
        #     'processed_dir': cn.global_cropland_mean_rate_harvest_area_all_crops_peat_2019_processed_dir,
        #     'processed_pattern': cn.global_cropland_mean_rate_harvest_area_all_crops_peat_2019_processed_pattern
        # }
        # download_upload_dictionary["global_cropland_mean_rate_harvest_area_all_crops_nonpeat_2006"] = {
        #     'raw_dir': cn.global_cropland_emissions_raw_dir,
        #     'raw_pattern': cn.global_cropland_mean_rate_harvest_area_all_crops_nonpeat_2006_raw_pattern,
        #     'vrt': "global_cropland_mean_rate_harvest_area_all_crops_nonpeat_2006.vrt",
        #     'processed_dir': cn.global_cropland_mean_rate_harvest_area_all_crops_nonpeat_2006_processed_dir,
        #     'processed_pattern': cn.global_cropland_mean_rate_harvest_area_all_crops_nonpeat_2006_processed_pattern
        # }
        #
        # download_upload_dictionary["global_cropland_mean_rate_harvest_area_all_crops_nonpeat_2019"] = {
        #     'raw_dir': cn.global_cropland_emissions_raw_dir,
        #     'raw_pattern': cn.global_cropland_mean_rate_harvest_area_all_crops_nonpeat_2019_raw_pattern,
        #     'vrt': "global_cropland_mean_rate_harvest_area_all_crops_nonpeat_2019.vrt",
        #     'processed_dir': cn.global_cropland_mean_rate_harvest_area_all_crops_nonpeat_2019_processed_dir,
        #     'processed_pattern': cn.global_cropland_mean_rate_harvest_area_all_crops_nonpeat_2019_processed_pattern
        # }
        #
        # download_upload_dictionary["global_cropland_mean_rate_physical_area_all_crops_peat_2006"] = {
        #     'raw_dir': cn.global_cropland_emissions_raw_dir,
        #     'raw_pattern': cn.global_cropland_mean_rate_physical_area_all_crops_peat_2006_raw_pattern,
        #     'vrt': "global_cropland_mean_rate_physical_area_all_crops_peat_2006.vrt",
        #     'processed_dir': cn.global_cropland_mean_rate_physical_area_all_crops_peat_2006_processed_dir,
        #     'processed_pattern': cn.global_cropland_mean_rate_physical_area_all_crops_peat_2006_processed_pattern
        # }

        download_upload_dictionary["global_cropland_mean_rate_physical_area_all_crops_peat_2019"] = {
            'raw_dir': cn.global_cropland_emissions_raw_dir,
            'raw_pattern': cn.global_cropland_mean_rate_physical_area_all_crops_peat_2019_raw_pattern,
            'vrt': "global_cropland_mean_rate_physical_area_all_crops_peat_2019.vrt",
            'processed_dir': cn.global_cropland_mean_rate_physical_area_all_crops_peat_2019_processed_dir,
            'processed_pattern': cn.global_cropland_mean_rate_physical_area_all_crops_peat_2019_processed_pattern
        }

        # download_upload_dictionary["global_cropland_mean_rate_physical_area_all_crops_nonpeat_2006"] = {
        #     'raw_dir': cn.global_cropland_emissions_raw_dir,
        #     'raw_pattern': cn.global_cropland_mean_rate_physical_area_all_crops_nonpeat_2006_raw_pattern,
        #     'vrt': "global_cropland_mean_rate_physical_area_all_crops_nonpeat_2006.vrt",
        #     'processed_dir': cn.global_cropland_mean_rate_physical_area_all_crops_nonpeat_2006_processed_dir,
        #     'processed_pattern': cn.global_cropland_mean_rate_physical_area_all_crops_nonpeat_2006_processed_pattern
        # }
        #
        # download_upload_dictionary["global_cropland_mean_rate_physical_area_all_crops_nonpeat_2019"] = {
        #     'raw_dir': cn.global_cropland_emissions_raw_dir,
        #     'raw_pattern': cn.global_cropland_mean_rate_physical_area_all_crops_nonpeat_2019_raw_pattern,
        #     'vrt': "global_cropland_mean_rate_physical_area_all_crops_nonpeat_2019.vrt",
        #     'processed_dir': cn.global_cropland_mean_rate_physical_area_all_crops_nonpeat_2019_processed_dir,
        #     'processed_pattern': cn.global_cropland_mean_rate_physical_area_all_crops_nonpeat_2019_processed_pattern
        # }
        #
        # download_upload_dictionary["global_cropland_total_amount_all_crops_peat_2006"] = {
        #     'raw_dir': cn.global_cropland_emissions_raw_dir,
        #     'raw_pattern': cn.global_cropland_total_amount_all_crops_peat_2006_raw_pattern,
        #     'vrt': "global_cropland_total_amount_all_crops_peat_2006.vrt",
        #     'processed_dir': cn.global_cropland_total_amount_all_crops_peat_2006_processed_dir,
        #     'processed_pattern': cn.global_cropland_total_amount_all_crops_peat_2006_processed_pattern
        # }
        #
        # download_upload_dictionary["global_cropland_total_amount_all_crops_peat_2019"] = {
        #     'raw_dir': cn.global_cropland_emissions_raw_dir,
        #     'raw_pattern': cn.global_cropland_total_amount_all_crops_peat_2019_raw_pattern,
        #     'vrt': "global_cropland_total_amount_all_crops_peat_2019.vrt",
        #     'processed_dir': cn.global_cropland_total_amount_all_crops_peat_2019_processed_dir,
        #     'processed_pattern': cn.global_cropland_total_amount_all_crops_peat_2019_processed_pattern
        # }
        #
        # download_upload_dictionary["global_cropland_total_amount_all_crops_nonpeat_2006"] = {
        #     'raw_dir': cn.global_cropland_emissions_raw_dir,
        #     'raw_pattern': cn.global_cropland_total_amount_all_crops_nonpeat_2006_raw_pattern,
        #     'vrt': "global_cropland_total_amount_all_crops_nonpeat_2006.vrt",
        #     'processed_dir': cn.global_cropland_total_amount_all_crops_nonpeat_2006_processed_dir,
        #     'processed_pattern': cn.global_cropland_total_amount_all_crops_nonpeat_2006_processed_pattern
        # }
        #
        # download_upload_dictionary["global_cropland_total_amount_all_crops_nonpeat_2019"] = {
        #     'raw_dir': cn.global_cropland_emissions_raw_dir,
        #     'raw_pattern': cn.global_cropland_total_amount_all_crops_nonpeat_2019_raw_pattern,
        #     'vrt': "global_cropland_total_amount_all_crops_nonpeat_2019.vrt",
        #     'processed_dir': cn.global_cropland_total_amount_all_crops_nonpeat_2019_processed_dir,
        #     'processed_pattern': cn.global_cropland_total_amount_all_crops_nonpeat_2019_processed_pattern
        # }

    #-------------------------------------------------------------------------------------------------------------------
    #Step 3: Create a VRT for each dataset
    vrt_futures = []

    for key,items in download_upload_dictionary.items():
        path = items["raw_dir"]
        pattern = items["raw_pattern"]
        vrt = items["vrt"]
        output_vrt_s3 = f"{path}{vrt}"

        # Add output_vrt_s3 to dictionary
        download_upload_dictionary[key]["output_vrt_s3"] = output_vrt_s3

        # Find all files in s3 that match the raw pattern (with terminal .tif)
        # Add list of raw file s3 paths to download_upload_dictionary
        input_raster_list_s3  = uu.list_s3_files_with_pattern(path, pattern)
        if input_raster_list_s3:
            download_upload_dictionary[key]["raw_raster_list"] = input_raster_list_s3

        #Create a vrt from all raw input rasters
        print(f"Attempting to build vrt for {key}:")
        if cluster_type == 'full' or cluster_type == 'test':
            # If running in coiled, download all raw input files, build vrt in cluster, and upload to s3
            future = client.submit(uu.build_vrt_gdal_coiled, input_raster_list_s3, output_vrt_s3, vrt)
        elif cluster_type == 'local':
            # If running locally, save vrt directly to s3 using vsis3 (does not work in coiled)
            future = client.submit(uu.build_vrt_gdal_local, input_raster_list_s3, output_vrt_s3)
        vrt_futures.append(future)

    # Collect the results once they are finished
    vrt_results = client.gather(vrt_futures)

    ###########################################################################################################
    #Step 4: Get GDAL datatype of each VRT
    for key,items in download_upload_dictionary.items():
        vrt = items['vrt']
        output_vrt_s3 = items['output_vrt_s3']

        # Get raster data type from vrt
        print(f"Attempting to get data type from {output_vrt_s3}")
        if cluster_type == 'full' or cluster_type == 'test':
            # If running in coiled, get data type from downloading vrt
            dt = uu.get_dtype_from_coiled(output_vrt_s3, vrt)
        elif cluster_type == 'local':
            # If running locally, get data type from vrt in s3
            dt = uu.get_dtype_from_s3(output_vrt_s3)

        # Add GDAL data type to download_upload dictionary
        if dt:
            gdal_dt = next(key for key, value in uu.gdal_dtype_mapping.items() if value == dt)  # Convert dt into GDAL data type
            download_upload_dictionary[key]["dt"] = gdal_dt
            print(f"vrt for {key} has data type: {dt} ({gdal_dt})")

    ###########################################################################################################
    #Step 5: Use warp_to_hansen to preprocess each dataset into 10x10 degree tiles
    #TODO see LULUCF model (take a bounding box as a command line argument, and make chunks instead of tile_id)
    for tile_id in cn.tile_id_list:
        tile_futures = []
        for key,items in download_upload_dictionary.items():
            filename = f"{tile_id}_{items['processed_pattern']}"
            output_tile_s3 = f"{items['processed_dir']}{filename}"
            xmin, ymin, xmax, ymax = uu.get_10x10_tile_bounds(tile_id)
            dt = items['dt']

            # Create 10 x 10 degree hansenized tile for each dataset in dictionary
            print(f"Attempting to create {key} tile for {tile_id}:")
            if cluster_type == 'full' or cluster_type == 'test':
                vrt = items["vrt"]
                tile_future = client.submit(uu.warp_to_hansen_coiled, vrt, filename, output_tile_s3,  xmin, ymin, xmax, ymax, dt, 0, True, 400, 400)
            elif cluster_type == 'local':
                input_vrt_s3 = f"{items['raw_dir']}{items['vrt']}"
                tile_future = client.submit(uu.warp_to_hansen_local, input_vrt_s3, output_tile_s3, xmin, ymin, xmax, ymax, dt, 0, True, 400, 400)
            tile_futures.append(tile_future)

        # Collect the results once they are finished
        tile_results = client.gather(tile_futures)

    ###########################################################################################################
    #Step 6: Delete files that were downloaded
    # Remove vrt and raw rasters after tile creation step
    if delete_local_files and (cluster_type == 'full' or cluster_type == 'test'):
        for key,items in download_upload_dictionary.items():
            print(f"Deleting local copy of input rasters and vrt for {key}:")
            raw_raster_list = items["raw_raster_list"]
            vrt = items["vrt"]
            uu.delete_build_vrt_input_files(raw_raster_list, vrt)

    ###########################################################################################################
    #Step 7: Close the cluster
    client.close()
    ###########################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hansenize AFOLU model raster inputs.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-ct', '--cluster_type', action='store', help='Run locally with Dask (local), test with 1 worker in coiled (test), or run with full coiled cluster (full)')
    parser.add_argument('-p', '--processes', action='store', nargs='+', help='What datasets do you want to hansenize? Options: drivers, secondary_natural_forest')
    parser.add_argument('--delete_local_files', action='store_true', help='When running in Coiled, deletes raw input rasters and vrt after hansenized tiles have been uploaded')

    args = parser.parse_args()

    # Create the cluster with command line arguments
    main(args.cluster_name, args.cluster_type, args.processes, args.delete_local_files)