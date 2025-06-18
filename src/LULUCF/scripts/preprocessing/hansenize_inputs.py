"""
Run from src/LULUCF

Coiled test area without land (i.e. no data):
python -m scripts.utilities.create_cluster -cn AFOLU_preprocessing -n 1
python -m scripts.preprocessing.hansenize_inputs -cn AFOLU_preprocessing -ct coiled -p secondary_natural_forest -bb -120 30 -110 40 -cs 10

Coiled test area with data:
python -m scripts.utilities.create_cluster -cn hansenize_mangroves_test -n 2 -t 2 -m 16
python -m scripts.preprocessing.hansenize_inputs -cn hansenize_mangroves_test -ct coiled -p mangroves -bb -120 30 -110 40 -cs 10

Coiled full run:
python -m scripts.utilities.create_cluster -cn AFOLU_preprocessing -n 20 -t 12
python -m scripts.preprocessing.hansenize_inputs -cn AFOLU_preprocessing -ct coiled -p secondary_natural_forest -bb -180 -60 180 80 -cs 10

"""
import os
import sys
import argparse
import dask
from dask.distributed import print
from ..utilities import constants_and_names as cn
from ..utilities import log_utilities as lu
from ..utilities import universal_utilities as uu


########################################################################################################################

def main(cluster_name, cluster_type, process, bounding_box, chunk_size, run_local, no_upload):

    # Step 1: Create download/ upload dictionary from list of processes to run
    # Create empty dictionary
    download_upload_dictionary = {}

    start_time = uu.timestr()

    # Add drivers data
    if 'drivers' in process:
        download_upload_dictionary["drivers"] = {
            'raw_dir': cn.drivers_raw_dir,
            'raw_pattern': cn.drivers_pattern,
            'vrt': f"/tmp/drivers.vrt",
            'processed_dir': cn.drivers_processed_dir,
            'processed_pattern': cn.drivers_pattern
        }

    # Add Robinson et al. secondary natural forest growth rates
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
        download_upload_dictionary["secondary_natural_forest_21_40"] = {
            'raw_dir': cn.secondary_natural_forest_raw_dir,
            'raw_pattern': cn.secondary_natural_forest_21_40_pattern,
            'vrt': f"/tmp/secondary_natural_forest_21_40.vrt",
            'processed_dir': cn.secondary_natural_forest_21_40_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_21_40_pattern
        }
        download_upload_dictionary["secondary_natural_forest_41_60"] = {
            'raw_dir': cn.secondary_natural_forest_raw_dir,
            'raw_pattern': cn.secondary_natural_forest_41_60_pattern,
            'vrt': f"/tmp/secondary_natural_forest_41_60.vrt",
            'processed_dir': cn.secondary_natural_forest_41_60_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_41_60_pattern
        }
        download_upload_dictionary["secondary_natural_forest_61_80"] = {
            'raw_dir': cn.secondary_natural_forest_raw_dir,
            'raw_pattern': cn.secondary_natural_forest_61_80_pattern,
            'vrt': f"/tmp/secondary_natural_forest_61_80.vrt",
            'processed_dir': cn.secondary_natural_forest_61_80_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_61_80_pattern
        }
        download_upload_dictionary["secondary_natural_forest_81_100"] = {
            'raw_dir': cn.secondary_natural_forest_raw_dir,
            'raw_pattern': cn.secondary_natural_forest_81_100_pattern,
            'vrt': f"/tmp/secondary_natural_forest_81_100.vrt",
            'processed_dir': cn.secondary_natural_forest_81_100_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_81_100_pattern
        }

    if 'AGB2015' in process:
        download_upload_dictionary["AGB2015"] = {
            'raw_dir': cn.agb_2015_dir_raw,
            'raw_pattern': cn.agb_2015_pattern_raw,
            'vrt': f"/tmp/agb2015.vrt",
            'processed_dir': cn.agb_2015_dir_processed,
            'processed_pattern': cn.agb_2015_pattern
        }

    if 'climate_zone' in process:
        download_upload_dictionary["climate_zone"] = {
            'raw_dir': cn.climate_zone_raw_dir,
            'raw_pattern': cn.climate_zone_raw_pattern,
            'vrt': f"/tmp/climate_zone.vrt",
            'processed_dir': cn.climate_zone_processed_dir,
            'processed_pattern': cn.climate_zone_pattern
        }

    if 'mangroves' in process:
        for year in cn.mangrove_extent_years:
            download_upload_dictionary[f"mangrove_extent_{year}"] = {
                'raw_dir': f"{cn.mangrove_extent_raw_dir}{year}/",
                'raw_pattern': cn.mangrove_extent_raw_pattern,
                'vrt': f"/tmp/mangrove_extent_{year}_v3.vrt",
                'processed_dir': f"{cn.mangrove_extent_processed_dir}{year}/",
                'processed_pattern': f"{year}_{cn.mangrove_extent_processed_pattern}"
            }

    if 'cropland_fertilizer' in process:
        download_upload_dictionary[""] = {
            'raw_dir': cn.x_raw_dir,
            'raw_pattern': cn.x_pattern,
            'vrt': "x.vrt",
            'processed_dir': cn.x_processed_dir,
            'processed_pattern': cn.x_pattern
        }

    if 'cropland_manure' in process:
        download_upload_dictionary[""] = {
            'raw_dir': cn.x_raw_dir,
            'raw_pattern': cn.x_pattern,
            'vrt': "x.vrt",
            'processed_dir': cn.x_processed_dir,
            'processed_pattern': cn.x_pattern
        }

    if 'cropland_peatland' in process:
        download_upload_dictionary[""] = {
            'raw_dir': cn.x_raw_dir,
            'raw_pattern': cn.x_pattern,
            'vrt': "x.vrt",
            'processed_dir': cn.x_processed_dir,
            'processed_pattern': cn.x_pattern
        }

    if 'cropland_residues' in process:
        download_upload_dictionary[""] = {
            'raw_dir': cn.x_raw_dir,
            'raw_pattern': cn.x_pattern,
            'vrt': "x.vrt",
            'processed_dir': cn.x_processed_dir,
            'processed_pattern': cn.x_pattern
        }

    if 'cropland_residues_burnt' in process:
        download_upload_dictionary[""] = {
            'raw_dir': cn.x_raw_dir,
            'raw_pattern': cn.x_pattern,
            'vrt': "x.vrt",
            'processed_dir': cn.x_processed_dir,
            'processed_pattern': cn.x_pattern
        }

    if 'cropland_rice' in process:
        download_upload_dictionary[""] = {
            'raw_dir': cn.x_raw_dir,
            'raw_pattern': cn.x_pattern,
            'vrt': "x.vrt",
            'processed_dir': cn.x_processed_dir,
            'processed_pattern': cn.x_pattern
        }

    #-------------------------------------------------------------------------------------------------------------------
    # COILED PIPELINE
    if cluster_type == 'coiled':
        #TODO get rid of cluster type, only runnin in cluster instead of locally

        if not run_local:
            # Connects to Coiled cluster if the named cluster exists
            cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, False, False)
            client

            # Creates the log for the main function and populates it with basic run information
            main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, f"Preprocessing: {process}", run_local,
                                                                           'standard', f'Hansenize: {process}')

            # Step 1: Create chunk list
            # Makes list of chunks to analyze from the bounding box and chunk size (deg)
            # Output list form is [[110, -10, 120, 0], [...], [...], ...]  (W, S, E, N)
            main_logger.info("STEP 1: Using bounding box and chunk size to determine number of chunks\n")
            chunk_list = uu.get_chunk_bounds_from_bounding_box(bounding_box, chunk_size)
            main_logger.info(f"Chunks identified: {len(chunk_list)}")


            # Step 2: Create a VRT for each dataset
            vrt_futures = []    #creates VRTs in parallel for each dataset
            for key, items in download_upload_dictionary.items():
                main_logger.info(f"STEP 2: Creating a VRT for {key}\n")

                # Add output_vrt_s3 to dictionary
                output_vrt_s3 = f"{items['raw_dir']}{os.path.basename(items['vrt'])}"
                main_logger.info(f"Adding output_vrt_s3 path to dictionary: {output_vrt_s3}")
                download_upload_dictionary[key]['output_vrt_s3'] = output_vrt_s3

                # Find all files in s3 that match the raw pattern (w/ '*.tif') and add s3 paths to download_upload_dictionary
                input_raster_list_s3 = uu.list_s3_files_with_pattern(items['raw_dir'], items['raw_pattern'])
                main_logger.info(f"Raw data folder: {items['raw_dir']}")
                main_logger.info(f"Raw data pattern: {items['raw_pattern']}")
                main_logger.info(f"There are {len(input_raster_list_s3)} rasters in the raw data folder to include in the vrt")
                if input_raster_list_s3:
                    download_upload_dictionary[key]['raw_raster_list'] = input_raster_list_s3

                # Create a vrt from all raw input rasters
                main_logger.info(f"Submitting VRT build for {key} ({uu.timestr('time')})")
                vrt_future = client.submit(uu.build_vrt_gdal_coiled, input_raster_list_s3, output_vrt_s3, items['vrt'])
                vrt_futures.append((key, vrt_future))

            # Wait for all VRTs to finish
            for key, future in vrt_futures:
                future.result()
            main_logger.info(f"\nStep 2 complete. All VRTs built ({uu.timestr('time')})\n")

            # Step 4: Get GDAL datatype of each dataset using the first tile in that dataset
            #TODO: Add logging and make dask task
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
                download_upload_dictionary[key]["dt"] = gdal_dtype
                main_logger.info(f"data type for {key} is {gdal_dtype}") #TODO so it is byte instead of 1

    else:
        sys.exit("Set cluster_type to 'coiled'")


    ###########################################################################################################
    #Step 5: Use warp_to_hansen to preprocess each dataset into 10x10 degree tiles

    # Iterates through all input datasets
    for key, items in download_upload_dictionary.items():

        # Separate tile_futures list for each dataset being processed
        tile_futures = []

        # Iterates through all tiles in a given dataset
        for chunk in chunk_list:

            tile_id = uu.xy_to_tile_id(chunk[0], chunk[3])  # tile_id in YYN/S_XXXE/W

            output_filename = f"{tile_id}_{items['processed_pattern']}"  #TODO add.tif extention and make sure all patterns don't have tif extension
            # print(output_filename)
            output_tile_s3 = f"{items['processed_dir']}{output_filename}"
            # print(output_tile_s3)
            xmin, ymin, xmax, ymax = uu.get_10x10_tile_bounds(tile_id)
            dt = items['dt']

            output_vrt_s3 = f"{items['raw_dir']}{os.path.basename(items['vrt'])}"
            main_logger.info(f"Using {output_vrt_s3} for Hansenization")

            # Create 10 x 10 degree hansenized tile for each dataset in dictionary
            if cluster_type == 'coiled':

                tile_future = client.submit(uu.warp_to_hansen_coiled, output_vrt_s3, output_filename, output_tile_s3,
                                            xmin, ymin, xmax, ymax, dt, 0, True, 400, 400)
                tile_futures.append(tile_future)

        main_logger.info(f"Tiles to process: {len(tile_futures)}")

        # Collect the results once they are finished
        tile_results = client.gather(tile_futures)
        main_logger.info(tile_results)
        main_logger.info(f"Completed Hansenizing tile set {key} from {items['raw_dir']}: {uu.timestr()}")
        uu.stage_duration(start_time, uu.timestr(), f"Hansenize_{key}", main_logger)


        # Step 6: Creates a tile index shapefile of the output rasters to check completeness of Hansenization

        main_logger.info(f"Making index shapefile for {key} from {items['raw_dir']}: {uu.timestr()}")

        # Creates a list of dictionaries of s3 tile set path with corresponding tile index shapefile names,
        # e.g., [{'s3://gfw2-data/climate/ESA_CCI_biomass/v5_01/2015/AGB/processed/20250217/': 'AGB_2015_ESA_CCI_Mg_AGB_ha'}]
        tile_index_dict = []

        # The key for the dictionary: the s3 path with a tile set that will be indexed
        path = items['processed_dir'].replace(cn.outputs_path, "")

        # The value for the dictionary: the pattern to use for naming the output shapefile
        value = items['processed_pattern']

        # Creates the dictionary:
        # e.g., {'s3://gfw2-data/climate/ESA_CCI_biomass/v5_01/2015/AGB/processed/20250217/': 'AGB_2015_ESA_CCI_Mg_AGB_ha'}
        tile_index_dict.append({path: value})
        # print(tile_index_dict)

        # Makes raster footprint shapefile from output raster set
        delayed_result = [dask.delayed(uu.make_tile_footprint_shp)(input_dict, no_upload)
                          for input_dict in tile_index_dict]

        # Actually runs analysis
        results = dask.compute(*delayed_result)
        main_logger.info(results)

        main_logger.info(f"Finished making index shapefile for {path} from {items['raw_dir']}: {uu.timestr()}" + "\n" + "\n")
        uu.stage_duration(start_time, uu.timestr(), f"shapefile_index_for_Hansenized_{key}", main_logger)

    # Closes the Dask client if not running locally
    if not run_local:
        client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hansenize AFOLU model raster inputs.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-ct', '--cluster_type', action='store', help='Run locally with Dask (local), or run with coiled cluster (coiled)')
    parser.add_argument('-p', '--processes', action='store', nargs='+', help='What datasets do you want to hansenize?')
    #parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    cluster_type = args.cluster_type
    processes = args.processes
    #run_local = args.run_local
    no_upload = args.no_upload
    #todo: remove all references to run_local

    bounding_box = args.bounding_box
    chunk_size = args.chunk_size

    # Create the cluster with command line arguments
    main(cluster_name, cluster_type, processes, bounding_box, chunk_size, False, no_upload)