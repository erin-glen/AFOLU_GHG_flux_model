"""
Run from src/LULUCF

Local:
python -m src.LULUCF.scripts.preprocessing.hansenize -ct local -p drivers

Coiled (Test):
python -m scripts.utilities.create_cluster -cn hansenize_drivers_test -n 1 -m 8 -t 8
python -m src.LULUCF.scripts.preprocessing.hansenize_restructure -cn hansenize_drivers_test -ct coiled -p drivers

#QC
cluster_name = 'Hansenize_drivers_data'
cluster_type = 'test'
process = ['drivers']


"""
import os
import argparse
import boto3
from dask.distributed import print
from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu


########################################################################################################################

from osgeo import gdal

def main(cluster_name, cluster_type, process, run_local):

    # Step 1: Create download/ upload dictionary from list of processes to run
    # Create empty dictionary
    download_upload_dictionary = {}

    # Add drivers data
    if 'drivers' in process:
        download_upload_dictionary["drivers"] = {
            'raw_dir': cn.drivers_raw_dir,
            'raw_pattern': cn.drivers_pattern,
            'vrt': f"/tmp/drivers.vrt",
            # 'processed_dir': cn.drivers_processed_dir,
            # TODO: Switch back processed dir
            'processed_dir': "s3://gfw2-data/drivers_of_loss/1_km/processed/coiled_test/",
            'processed_pattern': cn.drivers_pattern
        }

    # Add Robinson et al secondary natural forest growth rates
    if 'secondary_natural_forest' in process:
        download_upload_dictionary["secondary_natural_forest_0_5"] = {
            'raw_dir': cn.secondary_natural_forest_raw_dir,
            'raw_pattern': cn.secondary_natural_forest_0_5_pattern,
            'vrt': f"/tmp/secondary_natural_forest_0_5.vrt",
            'processed_dir': cn.secondary_natural_forest_0_5_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_0_5_pattern
        }

        download_upload_dictionary["secondary_natural_forest_6_10"] = {
            'raw_dir': cn.secondary_natural_forest_raw_dir,
            'raw_pattern': cn.secondary_natural_forest_6_10_pattern,
            'vrt': f"/tmp/secondary_natural_forest_6_10.vrt",
            'processed_dir': cn.secondary_natural_forest_6_10_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_6_10_pattern
        }

        download_upload_dictionary["secondary_natural_forest_11_15"] = {
            'raw_dir': cn.secondary_natural_forest_raw_dir,
            'raw_pattern': cn.secondary_natural_forest_11_15_pattern,
            'vrt': f"/tmp/secondary_natural_forest_11_15.vrt",
            'processed_dir': cn.secondary_natural_forest_11_15_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_11_15_pattern
        }

        download_upload_dictionary["secondary_natural_forest_16_20"] = {
            'raw_dir': cn.secondary_natural_forest_raw_dir,
            'raw_pattern': cn.secondary_natural_forest_16_20_pattern,
            'vrt': f"/tmp/secondary_natural_forest_16_20.vrt",
            'processed_dir': cn.secondary_natural_forest_16_20_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_16_20_pattern
        }

        download_upload_dictionary["secondary_natural_forest_21_100"] = {
            'raw_dir': cn.secondary_natural_forest_raw_dir,
            'raw_pattern': cn.secondary_natural_forest_21_100_pattern,
            'vrt': f"/tmp/secondary_natural_forest_21_100.vrt",
            'processed_dir': cn.secondary_natural_forest_21_100_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_21_100_pattern
        }

    if 'AGB2015' in process:
        download_upload_dictionary["AGB2015"] = {
            'raw_dir': cn.agb_2015_path_raw,
            'raw_pattern': cn.agb_2015_pattern_raw,
            'vrt': f"/tmp/agb2015.vrt",
            'processed_dir': cn.agb_2015_path_processed,
            'processed_pattern': cn.agb_2015_pattern
        }

    # if 'cropland_fertilizer' in process:
    #     download_upload_dictionary[""] = {
    #         'raw_dir': cn.x_raw_dir,
    #         'raw_pattern': cn.x_pattern,
    #         'vrt': "x.vrt",
    #         'processed_dir': cn.x_processed_dir,
    #         'processed_pattern': cn.x_pattern
    #     }
    #
    # if 'cropland_manure' in process:
    #     download_upload_dictionary[""] = {
    #         'raw_dir': cn.x_raw_dir,
    #         'raw_pattern': cn.x_pattern,
    #         'vrt': "x.vrt",
    #         'processed_dir': cn.x_processed_dir,
    #         'processed_pattern': cn.x_pattern
    #     }
    #
    # if 'cropland_peatland' in process:
    #     download_upload_dictionary[""] = {
    #         'raw_dir': cn.x_raw_dir,
    #         'raw_pattern': cn.x_pattern,
    #         'vrt': "x.vrt",
    #         'processed_dir': cn.x_processed_dir,
    #         'processed_pattern': cn.x_pattern
    #     }
    #
    # if 'cropland_residues' in process:
    #     download_upload_dictionary[""] = {
    #         'raw_dir': cn.x_raw_dir,
    #         'raw_pattern': cn.x_pattern,
    #         'vrt': "x.vrt",
    #         'processed_dir': cn.x_processed_dir,
    #         'processed_pattern': cn.x_pattern
    #     }
    #
    # if 'cropland_residues_burnt' in process:
    #     download_upload_dictionary[""] = {
    #         'raw_dir': cn.x_raw_dir,
    #         'raw_pattern': cn.x_pattern,
    #         'vrt': "x.vrt",
    #         'processed_dir': cn.x_processed_dir,
    #         'processed_pattern': cn.x_pattern
    #     }
    #
    # if 'cropland_rice' in process:
    #     download_upload_dictionary[""] = {
    #         'raw_dir': cn.x_raw_dir,
    #         'raw_pattern': cn.x_pattern,
    #         'vrt': "x.vrt",
    #         'processed_dir': cn.x_processed_dir,
    #         'processed_pattern': cn.x_pattern
    #     }

    #-------------------------------------------------------------------------------------------------------------------
    # COILED PIPELINE
    if cluster_type == 'coiled':
        # Step 2: Attach to coiled cluster by cluster_name
        # (cluster created via command line prior to running this module)

        if not run_local:
            # Connects to Coiled cluster if not running locally
            cluster, client = uu.connect_to_Coiled_cluster(cluster_name, False)
            client


        # Step 3: Create a VRT for each dataset

        for key, items in download_upload_dictionary.items():

            # Add output_vrt_s3 to dictionary
            output_vrt_s3 = f"{items['raw_dir']}{os.path.basename(items['vrt'])}"
            # print("output_vrt_s3:", output_vrt_s3)
            download_upload_dictionary[key]["output_vrt_s3"] = output_vrt_s3
            print("download_upload_dictionary:", download_upload_dictionary)

            # Find all files in s3 that match the raw pattern (w/ '*.tif') and add s3 paths to download_upload_dictionary
            input_raster_list_s3 = uu.list_s3_files_with_pattern(items["raw_dir"], items["raw_pattern"])
            if input_raster_list_s3:
                download_upload_dictionary[key]["raw_raster_list"] = input_raster_list_s3

            # Create a vrt from all raw input rasters
            print(f"Building vrt for {key}...")
            uu.build_vrt_gdal_coiled(input_raster_list_s3, output_vrt_s3, items["vrt"])


        # Step 4: Get GDAL datatype of each dataset using the first tile in that dataset
        for key, items in download_upload_dictionary.items():

            # Dictionary that matches format expected by function that gets name of first tile in an s3 folder
            simple_dict = {}
            simple_dict_key = key
            simple_dict_value = items["raw_dir"]
            simple_dict[simple_dict_key] = simple_dict_value

            # Path of first tile in the dataset
            first_tile = uu.first_file_name_in_s3_folder(simple_dict)

            # Gets datatype of first tile in input dataset and converts it to GDAL format
            download_dict_with_data_types = uu.add_file_type_to_dict(first_tile)
            dtype = download_dict_with_data_types[simple_dict_key][1]
            gdal_dtype = uu.string_to_gdal_dtype_mapping.get(dtype)

            # Adds the dtype of the dataset to the processing dictionary
            # download_upload_dictionary[key]["dt"] = gdal.GDT_Int16  #TODO placeholder
            download_upload_dictionary[key]["dt"] = gdal_dtype  #TODO placeholder
            print(download_upload_dictionary)

    # # -------------------------------------------------------------------------------------------------------------------
    # # LOCAL PIPELINE
    # # Step 2: Get local cluster
    # elif cluster_type == 'local':
    #
    #     # Step 2: Get local dask cluster with multiple workers
    #     client = uu.get_client_from_cluster_type('local')
    #     client
    #
    #
    #     # Step 3: Create a VRT for each dataset
    #     vrt_futures = []
    #
    #     for key, items in download_upload_dictionary.items():
    #
    #         # Add output_vrt_s3 to dictionary
    #         output_vrt_s3 = f"{items["raw_dir"]}{items["vrt"]}"
    #         download_upload_dictionary[key]["output_vrt_s3"] = output_vrt_s3
    #
    #         # Find all files in s3 that match the raw pattern (w/ '*.tif') and add s3 paths to download_upload_dictionary
    #         input_raster_list_s3 = uu.list_s3_files_with_pattern(items["raw_dir"], items["raw_pattern"])
    #         if input_raster_list_s3:
    #             download_upload_dictionary[key]["raw_raster_list"] = input_raster_list_s3
    #
    #         # Create a vrt from all raw input rasters
    #         print(f"Attempting to build vrt for {key}:")
    #
    #         # If running locally, save vrt directly to s3 using vsis3 (does not work in coiled)
    #         future = client.submit(uu.build_vrt_gdal_local, input_raster_list_s3, output_vrt_s3)
    #         vrt_futures.append(future)
    #
    #     # Collect the results once they are finished
    #     vrt_results = client.gather(vrt_futures)
    #
    #
    #     # Step 4: Get GDAL datatype of each VRT
    #     for key, items in download_upload_dictionary.items():
    #
    #         # Get raster data type from vrt
    #         print(f"Attempting to get data type from {items['output_vrt_s3']}")
    #
    #         # If running locally, get data type from vrt in s3
    #          dt = uu.get_dtype_from_s3(items['output_vrt_s3'])
    #
    #         # Add GDAL data type to download_upload dictionary
    #         if dt:
    #             gdal_dt = next(key for key, value in uu.gdal_dtype_mapping.items() if value == dt)  # Convert dt into GDAL data type
    #             download_upload_dictionary[key]["dt"] = gdal_dt
    #             print(f"vrt for {key} has data type: {dt} ({gdal_dt})")
    #

    # -------------------------------------------------------------------------------------------------------------------
    else:
        print("Set cluster_type to one of the following: 'coiled', 'local'")


    ###########################################################################################################
    #Step 5: Use warp_to_hansen to preprocess each dataset into 10x10 degree tiles

    # Iterates through all input datasets
    for key,items in download_upload_dictionary.items():

        # Separate tile_futures list for each dataset being processed
        tile_futures = []

        # Iterates through all tiles in a given dataset
        for tile_id in cn.tile_id_list:

            output_filename = f"{tile_id}_{items['processed_pattern']}.tif"
            # print(output_filename)
            output_tile_s3 = f"{items['processed_dir']}{output_filename}"
            # print(output_tile_s3)
            xmin, ymin, xmax, ymax = uu.get_10x10_tile_bounds(tile_id)
            dt = items['dt']

            # Create 10 x 10 degree hansenized tile for each dataset in dictionary
            if cluster_type == 'coiled' or cluster_type == 'test':

                # if not os.path.exists(vrt):
                #     raise FileNotFoundError(f"Outside warping function: VRT file not found at {vrt}")
                # else:
                #     print("VRT found outside warping function")

                tile_future = client.submit(uu.warp_to_hansen_coiled, output_vrt_s3, output_filename, output_tile_s3,
                                            xmin, ymin, xmax, ymax, dt, 0, True, 400, 400)
                tile_futures.append(tile_future)


            if cluster_type == 'local':
                input_vrt_s3 = f"{items['raw_dir']}{items['vrt']}"
                tile_future = client.submit(uu.warp_to_hansen_local, input_vrt_s3, output_tile_s3,
                                            xmin, ymin, xmax, ymax, dt, 0, True, 400, 400)
                tile_futures.append(tile_future)

        print(f"Tiles to process: {len(tile_futures)}")

        # Collect the results once they are finished
        tile_results = client.gather(tile_futures)
        print(tile_results)
        # TODO see LULUCF model (take a bounding box as a command line argument, and make chunks instead of tile_id)


    ###########################################################################################################
    # Closes the Dask client if not running locally
    if not run_local:
        client.close()
    ###########################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hansenize AFOLU model raster inputs.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-ct', '--cluster_type', action='store', help='Run locally with Dask (local), test with 1 worker in coiled (test), or run with full coiled cluster (full)')
    parser.add_argument('-p', '--processes', action='store', nargs='+', help='What datasets do you want to hansenize? Options: drivers, secondary_natural_forest, AGB2015')
    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')

    args = parser.parse_args()

    # Create the cluster with command line arguments
    main(args.cluster_name, args.cluster_type, args.processes, args.run_local)