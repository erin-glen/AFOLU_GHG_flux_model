"""
Run from src/LULUCF/

Local:
python -m scripts.preprocessing.starting_carbon_pools.0_create_starting_carbon_pools -cn AFOLU_flux_model_scripts -bb 116 -3 116.25 -2.75 -cs 0.25 --run_local --no_stats --no_upload --year YYYY

python -m scripts.utilities.create_cluster -n 1 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing.starting_carbon_pools.0_create_starting_carbon_pools -cn AFOLU_flux_model_scripts -bb 116 -3 116.25 -2.75 -cs 0.25 --no_stats --year YYYY
python -m scripts.preprocessing.starting_carbon_pools.0_create_starting_carbon_pools -cn AFOLU_flux_model_scripts -cshp -f 1 --year YYYY

python -m scripts.utilities.create_cluster -n 60 -t 10 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing..starting_carbon_pools.0_create_starting_carbon_pools -cn AFOLU_flux_model_scripts -cshp --year 2000 -ln "This is intended to be the definitive global run for carbon pool 2000 creation."
Max memory usage: ~18 GB/worker
Time: 23:17 through calculation; 40:25 through tile stats; Credits: 170; Cost: $6.00

python -m scripts.utilities.create_cluster -n 50 -t 12 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing.starting_carbon_pools.0_create_starting_carbon_pools -cn AFOLU_flux_model_scripts -cshp --year 2015 -ln "This is intended to be the definitive global run for carbon pool 2015 creation."
Max memory usage: ~20 GB/worker
Time: 29:26 through calculation; 31:11 through tile stats; Credits: 116; Cost: $3.94
-t 14 seemed high, so may be better to go back down to -t 12.

NOTE: Maybe there's some way to configure this to output 10x10 deg tiles but I can't figure it out.
Instead, it creates 1x1 deg tiles and then merges them to 10x10 deg tiles.

To create a vrt of the 10x10 deg outputs, do:
aws s3 ls s3://gfw2-data/climate/ESA_CCI_biomass/v5_01/2015/year_2015_derived_carbon_pools/litter_C_density_MgC_ha/40000_pixels/ --recursive | grep .tif$ | awk '{print "/vsis3/gfw2-data/"$4}' > litter_C_2015_file_list.txt
gdalbuildvrt -input_file_list litter_C_2015_file_list.txt deadwood_C2015_mosaic.vrt
"""

import argparse
import concurrent.futures
import dask
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor

from dask.distributed import print
from numba import jit

# Project imports
from ...utilities import constants_and_names as cn
from ...utilities import universal_utilities as uu
from ...utilities import log_utilities as lu
from ...utilities import numba_utilities as nu
from ...utilities import resize_cluster


# Function to create initial (year 2000) non-soil carbon pool densities
# Operates pixel by pixel, so uses numba (Python compiled to C++).
@jit(nopython=True)
def create_starting_C_densities(in_dict_uint8, in_dict_uint16, in_dict_int16,
                                in_dict_int32, in_dict_float32, mangrove_C_ratio_array, year):

    # Separate dictionaries for output numpy arrays of each datatype, named by output data type.
    # This is because a dictionary in a Numba function cannot have arrays with multiple data types, so each dictionary has to store only one data type,
    # just like inputs to the function.
    out_dict_float32 = {}

    # print(in_dict_uint8)
    # print(in_dict_uint16)
    # print(in_dict_int16)
    # print(in_dict_int32)
    # print(in_dict_float32)

    # Input blocks
    r_s_ratio_block = in_dict_float32[cn.r_s_ratio_pattern]
    elevation_block = in_dict_int16[cn.elevation_pattern]
    climate_domain_block = in_dict_int16[cn.climate_domain_pattern]
    precipitation_block = in_dict_int32[cn.precipitation_pattern]
    continent_ecozone_block = in_dict_int16[cn.continent_ecozone_pattern]

    # Gets a fallback value for continent_ecozone for the chunk in case some pixels don't have one
    continent_ecozone_fallback = nu.backup_continent_ecozone(continent_ecozone_block)

    # AGB block sources (mangrove and non-mangrove) depend on the starting year
    # Numba can't handle two different possible datatypes for agb_non_mang_block,
    # so both options need to be recast to the same type
    if year == 2000:
        agb_non_mang_block = in_dict_int16[cn.agb_2000_pattern].astype(np.int16)
        mangrove_agb_block = in_dict_float32[cn.mangrove_agb_2000_pattern]
    elif year == 2015:
        agb_non_mang_block = in_dict_uint16[cn.agb_2015_pattern].astype(np.int16)
        mangrove_agb_block = in_dict_float32[cn.mangrove_agb_2000_pattern]
    else:
        out_dict_float32[f"{cn.agc_dens_pattern}_{year}"] = np.full(in_dict_float32[cn.r_s_ratio_pattern].shape, 9999).astype('float32')
        return out_dict_float32

    mangrove_in_chunk = True  # Flag for whether chunk has mangrove in it
    agb_non_mang_in_chunk = True  # Flag for whether chunk has non-mangrove AGB in it

    # Checks if the chunk has various inputs by seeing if the max value is 0.
    # If the max value is 0, it assumed that input doesn't exist.
    if agb_non_mang_block.max() == 0:
        agb_non_mang_in_chunk = False
    if mangrove_agb_block.max() == 0:
        mangrove_in_chunk = False

    # Output blocks
    # Need to specify the output datatype or it will default to float32
    agc_out_block = np.zeros(in_dict_float32[cn.r_s_ratio_pattern].shape).astype('float32')
    bgc_out_block = np.zeros(in_dict_float32[cn.r_s_ratio_pattern].shape).astype('float32')
    deadwood_c_out_block = np.zeros(in_dict_float32[cn.r_s_ratio_pattern].shape).astype('float32')
    litter_c_out_block = np.zeros(in_dict_float32[cn.r_s_ratio_pattern].shape).astype('float32')

    # Iterates through all pixels in the chunk
    for row in range(continent_ecozone_block.shape[0]):
        for col in range(continent_ecozone_block.shape[1]):

            # Input values for this specific cell
            agb_non_mang_cell = agb_non_mang_block[row, col]
            mangrove_agb_cell = mangrove_agb_block[row, col]
            elevation = elevation_block[row, col]
            climate_domain = climate_domain_block[row, col]
            precipitation = precipitation_block[row, col]
            r_s_ratio = r_s_ratio_block[row, col]
            continent_ecozone_cell = continent_ecozone_block[row, col]

            # If mangrove AGB is present, AGC is calculated from it, overwriting any AGC that is based on non-mang AGB that is already there
            if (mangrove_in_chunk) and (mangrove_agb_cell > 0):  # Only uses AGB if chunk exists and there is a value in that pixel
                agc_out_block[row, col] = mangrove_agb_cell * cn.biomass_to_carbon_mangrove

            # If non-mang AGB is present, AGC is calculated from it
            elif (agb_non_mang_in_chunk) and (agb_non_mang_cell > 0):  # Only uses AGB if chunk exists and there is a value in that pixel
                agc_out_block[row, col] = agb_non_mang_cell * cn.biomass_to_carbon_non_mangrove

            else:
                agc_out_block[row, col] = 0

            # Separate branches for assigning BGC, deadwood C, and litter C ratios depending on whether the pixel has mangroves.
            # Calculation of BGC, deadwood C, and litter C are done after the decision tree assigns the ratios.

            # Mangrove carbon pool ratio branch
            # From IPCC 2013 Wetland Supplement
            if (mangrove_in_chunk) and (mangrove_agb_cell > 0):  # Only replaces non-mangrove AGB if mangrove chunk exists and if mangrove value in that pixel

                # Applies the continent_ecozne fallback value when there isn't a value for the pixel
                if continent_ecozone_cell == 0:
                    continent_ecozone_cell = continent_ecozone_fallback

                bgc_ratio = mangrove_C_ratio_array[np.where(mangrove_C_ratio_array[:, 0] == continent_ecozone_cell)][0, 1]
                deadwood_c_ratio = mangrove_C_ratio_array[np.where(mangrove_C_ratio_array[:, 0] == continent_ecozone_cell)][0, 2]
                litter_c_ratio = mangrove_C_ratio_array[np.where(mangrove_C_ratio_array[:, 0] == continent_ecozone_cell)][0, 3]


            # Non-mangrove carbon pool ratio branch
            # Deadwood and litter carbon as fractions of AGC are from
            # https://cdm.unfccc.int/methodologies/ARmethodologies/tools/ar-am-tool-12-v3.0.pdf
            # "Clean Development Mechanism A/R Methodological Tool:
            # Estimation of carbon stocks and change in carbon stocks in dead wood and litter in A/R CDM project activities version 03.0"
            # Tables on pages 18 (deadwood) and 19 (litter).
            # They depend on the climate domain, elevation, and precipitation.
            elif (agb_non_mang_in_chunk) and (agb_non_mang_cell > 0):  # Non-mangrove

                # If no mapped R:S (=0), uses the global default value instead
                if r_s_ratio == 0:
                    r_s_ratio = cn.default_r_s_non_mang
                bgc_ratio = r_s_ratio  # Uses R:S for BGC

                if climate_domain == 1:  # Tropical/subtropical
                    if elevation <= 2000:  # Low elevation
                        if precipitation <= 1000:  # Low precipitation or no precip raster
                            deadwood_c_ratio = cn.tropical_low_elev_low_precip_deadwood_c_ratio
                            litter_c_ratio = cn.tropical_low_elev_low_precip_litter_c_ratio
                        elif ((precipitation > 1000) and (precipitation <= 1600)):  # Medium precipitation
                            deadwood_c_ratio = cn.tropical_low_elev_med_precip_deadwood_c_ratio
                            litter_c_ratio = cn.tropical_low_elev_med_precip_litter_c_ratio
                        else:  # High precipitation
                            deadwood_c_ratio = cn.tropical_low_elev_high_precip_deadwood_c_ratio
                            litter_c_ratio = cn.tropical_low_elev_high_precip_litter_c_ratio
                    else:  # High elevation
                        deadwood_c_ratio = cn.tropical_high_elev_deadwood_c_ratio
                        litter_c_ratio = cn.tropical_high_elev_litter_c_ratio
                else:  # Temperate/boreal
                    deadwood_c_ratio = cn.non_tropical_deadwood_c_ratio
                    litter_c_ratio = cn.non_tropical_litter_c_ratio

            else:

                # Ridiculous default BGC, deadwood C, and litter C ratios that will make it very clear if they are being used instead of
                # something being assigned in the decision tree above
                bgc_ratio = -5
                deadwood_c_ratio = -10
                litter_c_ratio = -20

            # Actually calculates BGC, deadwood C, and litter C using the ratios assigned in the above decision tree
            bgc_out_block[row, col] = agc_out_block[row, col] * bgc_ratio
            deadwood_c_out_block[row, col] = agc_out_block[row, col] * deadwood_c_ratio
            litter_c_out_block[row, col] = agc_out_block[row, col] * litter_c_ratio

    # Adds the output arrays to the dictionary with the appropriate data type
    # Outputs need .copy() so that previous intervals' arrays in dictionary aren't overwritten because arrays in dictionaries are mutable (courtesy of ChatGPT).
    out_dict_float32[f"{cn.agc_dens_pattern}_{year}"] = agc_out_block.copy()
    out_dict_float32[f"{cn.bgc_dens_pattern}_{year}"] = bgc_out_block.copy()
    out_dict_float32[f"{cn.deadwood_c_dens_pattern}_{year}"] = deadwood_c_out_block.copy()
    out_dict_float32[f"{cn.litter_c_dens_pattern}_{year}"] = litter_c_out_block.copy()

    # return output dictionary/ies
    return out_dict_float32



# All steps for creating starting non-soil carbon pools in a chunk: download chunks, calculate carbon densities, upload to s3
def create_and_upload_starting_C_densities(bounds, mangrove_C_ratio_array, download_dict_with_data_types, year,
                                           fishnet_iso_df, is_final, no_upload, output_folders, stage):

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []

    logger_worker = lu.setup_logging_worker()

    try:

        uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_final, logger_worker)

        bounds_str = uu.boundstr(bounds)  # String form of chunk bounds
        tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
        chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)

        ### Part 1: Downloads chunk.
        ### No checks about whether the chunk has data because the way the chunk_list is constructed,
        ### every chunk is relevant and should be processed, so they don't need to be checked.

        # Replaces the placeholder tile_id in the download data dictionary from main with the tile_id for this chunk
        updated_download_dict = uu.replace_tile_id_in_dict(download_dict_with_data_types, tile_id)

        # If a particular tile doesn't exist for an input, an array of 0s of the correct size and datatype is returned instead.
        # Thus, this returns a complete set of inputs (missing chunks filled).
        # Note: If running in a local Dask cluster, prints to console may be duplicated. Doesn't happen with a Coiled cluster of the same size (1 worker).
        # Seems to be a problem with local Dask getting overwhelmed by so many futures being created and downloaded from s3.
        futures = uu.prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_final, logger_worker)
        # print(futures)

        lu.print_and_log(f"Waiting for requests for data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

        # Dictionary that stores the dataset name (key) and downloaded data and download status (values)
        layers = {}

        # Ensures futures stores Future objects
        # Revised with https://chatgpt.com/share/e/67bde66c-d9a0-800a-a524-a9ef88c641a2 to return status messages for chunks
        for future in concurrent.futures.as_completed(futures):
            layer = futures[future]  # Gets the corresponding key
            data, status = future.result()  # Unpacks the tuple result
            if 'success' not in status: # Prints and logs any inputs that couldn't be accessed and are downloaded as all 0s
                lu.print_and_log(f"{status}", is_final, logger_worker)
            layers[layer] = data

        # # Test prints
        # print(layers)
        # print(layers['AGB_2015_ESA_CCI_Mg_AGB_ha'].max())
        # print(layers['AGB_2015_ESA_CCI_Mg_AGB_ha'].dtype)


        ### Part 2: Calculates min, mean, and max for each input chunk.
        ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.

        # Calculates stats for the input layers
        for key, array in layers.items():
            chunk_stats.append(uu.calculate_stats(array, key, bounds_str, tile_id, 'input_layer'))
        # print(chunk_stats)


        ### Part 3: Creates a separate dictionary for each chunk datatype so that they can be passed to Numba as separate arguments.
        ### Numba functions can accept (and return) dictionaries of arrays as long as each dictionary only has arrays of one data type (e.g., uint8, float32).
        ### Note: need to add new code if inputs with other data types are added

        lu.print_and_log(f"Creating typed dictionaries for chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

        # Creates the typed dictionaries for all input layers (including those that originally had no data)
        typed_dict_uint8, typed_dict_uint16, typed_dict_int16, typed_dict_int32, typed_dict_float32 = nu.create_typed_dicts(layers)

        # print("uint8_typed_list:", typed_dict_uint8)
        # print("uint16_typed_list:", typed_dict_uint16)
        # print("int16_typed_list:", typed_dict_int16)
        # print("int32_typed_list:", typed_dict_int32)
        # print("float32_typed_list:", typed_dict_float32)


        ### Part 4: Creates starting carbon pool densities

        lu.print_and_log(f"Creating starting C densities for {year} in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker) # Prints during full runs
        uu.rename_s3_task_file(stage, bounds, "calculating_", is_final, logger_worker)

        # Create AGC, BGC, deadwood C and litter C densities in selected starting year
        out_dict_float32 = create_starting_C_densities(
            typed_dict_uint8, typed_dict_uint16, typed_dict_int16, typed_dict_int32, typed_dict_float32,
            mangrove_C_ratio_array, year
        )

        # Fresh non-Numba-constrained dictionary that stores all output numpy arrays of all datatypes.
        # The dictionaries by datatype that are returned from the numba function have limitations on them,
        # e.g., they can't be combined with other datatypes. This prevents the addition of attributes needed for uploading to s3.
        # So the trick here is to copy the numba-exported arrays into normal Python arrays to which we can do anything in Python.
        out_dict_all_dtypes = {}

        # Transfers the dictionaries of numpy arrays for each data type to a new, Pythonic array
        out_dicts = [out_dict_float32]

        # Loop through each dictionary and update out_dict_all_dtypes
        for out_dict in out_dicts:
            for key, value in out_dict.items():
                out_dict_all_dtypes[key] = value

            # Clear memory of unneeded arrays
            del out_dict


        ### Part 5: Calculates per ha min, per ha mean, per ha max, and per pixel sum for each output chunk.
        ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.
        ### Also useful for a quick sum of outputs without doing zonal stats

        # Deletes all unnecessary input dictionaries before the memory-intensive derived output calculations
        # Suggested by ChatGPT: https://chatgpt.com/share/e/672bbf2e-ebbc-800a-aae3-3d92f5a1d663
        in_dicts = [layers, typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32]
        [in_dict.clear() for in_dict in in_dicts]

        # The relevant pixel area (m^2) file in s3
        pixel_area_uri = f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tile_id}.tif"

        # Gets numpy arrays of the model output being analyzed and the area (m^2) per pixel
        pixel_area_chunk = uu.get_tile_dataset_rio(pixel_area_uri, bounds, chunk_length_pixels, 'Float32')
        pixel_area_chunk = pixel_area_chunk[0]  # Converts downloaded tuple (array, status) to just the array

        # Calculates stats for the output layers from create_starting_C_densities as a dictionary with chunk attributes
        for key, array_per_ha in out_dict_all_dtypes.items():

            # Converts per hectare values to per pixel values for the output numpy array
            output_per_pixel = array_per_ha * pixel_area_chunk * cn.m2_to_ha

            chunk_stats.append(uu.calculate_stats(array_per_ha, key, bounds_str, tile_id, 'output_layer', output_per_pixel))


        ### Part 6: Saves numpy arrays as rasters and uploads to s3

        uu.rename_s3_task_file(stage, bounds, "uploading_", is_final, logger_worker)

        # Only saves arrays to geotifs and uploads them to s3 if enabled
        if not no_upload:

            out_no_data_val = 0  # NoData value for output raster (optional)

            # Adds metadata used for uploading outputs to s3 to the dictionary
            for key, value in out_dict_all_dtypes.items():
                data_type = value.dtype.name

                # Retrieves the file name pattern and date(s) covered for the output file for use in s3 folder construction
                out_pattern, year_range = uu.strip_and_extract_years(key)

                # Retrieves the relevant output s3 path for this specific output  (list of one element)
                matched_output_s3_folder = [item for item in output_folders if out_pattern in item][0]

                # Output paths without bucket (s3://gfw2-data)
                s3_path_without_bucket = f"{matched_output_s3_folder[cn.full_bucket_prefix_length:]}"

                # Dictionary with metadata for each array
                out_dict_all_dtypes[key] = [value, data_type, out_pattern, year_range, s3_path_without_bucket]

            # Converts output numpy arrays to local rasters and puts them in a list of files to upload in parallel
            upload_tasks = uu.save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str,
                                                               out_dict_all_dtypes, is_final, logger_worker, out_no_data_val)

            # Only prints if not a final run
            lu.print_and_log(f"Upload tasks created for {bounds_str} in {tile_id}. Uploading now: {uu.timestr()}", is_final, logger_worker)

            # Executes uploads in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                executor.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks)

            # Only prints if not a final run
            lu.print_and_log(f"Uploads completed for {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

        # Clears memory of unneeded arrays
        del out_dict_all_dtypes

        return_message = f"Success creating initial carbon pools for {bounds_str}: {uu.timestr()}"

        # Removes task tracking file from S3 once task is successful
        uu.delete_s3_task_file(stage, bounds, is_final, logger_worker)

    except Exception as e:

        return_message = f"Error processing chunk {bounds}: {e}: {uu.timestr()}"

        lu.print_and_log(return_message, False, logger_worker)
        uu.rename_s3_task_file(stage, bounds, "error_", is_final, logger_worker)

    return return_message, chunk_stats  # Return both the success message and the statistics


def main(cluster_name, year, run_local=False, no_stats=False, no_log=False, no_upload= False,
         use_shapefile=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = f'starting_carbon_pools_{year}_1x1_deg'
    model_type = 'standard'

    # Determines if argument for year is valid
    if year in [2000, 2015]:
        print("Year selection valid")
    else:
        print("Year selection not valid")
        sys.exit()

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(bounding_box, use_shapefile, client, cluster, log_note, run_local, model_type, stage)

    # Starting time for stage
    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Year for carbon pools: {year}")

    # Returns a dataframe of chunk_id and ISO for the GADM3.6 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso()

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, use_shapefile, chunk_size, first_chunks, fishnet_iso_df, main_logger)

    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_final = False
    if len(chunk_list) > 20:
        is_final = True
        main_logger.info("Running as final model.")

    # This is just a placeholder tile_id that is used to obtain the datatype of each tile set.
    # It is overwritten when chunks are assigned and analyzed.
    # Using this placeholder allows the full path and tile name to be specified up front, which simplifies things.
    # Otherwise, we'd have just the path but not the file name now and would have to add in the file name later
    # (probably at the chunk level).
    sample_tile_id = "00N_000E"

    # Dictionary of data to download (inputs to model)
    download_dict = {
        cn.elevation_pattern: f"{cn.elevation_dir}{sample_tile_id}_{cn.elevation_pattern}.tif",
        cn.climate_domain_pattern: f"{cn.climate_domain_dir}{sample_tile_id}_{cn.climate_domain_pattern}.tif",
        cn.precipitation_pattern: f"{cn.precipitation_dir}{sample_tile_id}_{cn.precipitation_pattern}.tif",
        cn.r_s_ratio_pattern: f"{cn.r_s_ratio_dir}{sample_tile_id}_{cn.r_s_ratio_pattern}.tif",
        cn.continent_ecozone_pattern: f"{cn.continent_ecozone_dir}{sample_tile_id}_{cn.continent_ecozone_pattern}.tif"
    }

    # Dictionary of data to download
    if year == 2000:
        download_dict[cn.agb_2000_pattern] = f"{cn.agb_2000_dir}{sample_tile_id}_{cn.agb_2000_pattern}.tif"
        download_dict[cn.mangrove_agb_2000_pattern] = f"{cn.mangrove_agb_2000_dir}{sample_tile_id}_{cn.mangrove_agb_2000_pattern}.tif"
        output_dir_list = [cn.agc_2000_dir, cn.bgc_2000_dir, cn.deadwood_c_2000_dir, cn.litter_c_2000_dir]

    elif year == 2015:
        download_dict[cn.agb_2015_pattern] = f"{cn.agb_2015_dir_processed}{sample_tile_id}_{cn.agb_2015_pattern}.tif"
        ##TODO Using mangrove AGB2000 for 2015 model start! Need to use something else for 2015!!!!!!
        download_dict[cn.mangrove_agb_2000_pattern] = f"{cn.mangrove_agb_2000_dir}{sample_tile_id}_{cn.mangrove_agb_2000_pattern}.tif"
        output_dir_list = [cn.agc_2015_dir, cn.bgc_2015_dir, cn.deadwood_c_2015_dir, cn.litter_c_2015_dir]

    else:
        print(f"Year input {year} not valid. Terminating.")
        sys.exit()

    # Creates list of output directories specific to the run
    output_dir_list = [path.replace("DATE", uu.timestr()[:8]) for path in output_dir_list]
    output_dir_list = [path.replace("CHUNK_SIZE", str(chunk_size_pixels)) for path in output_dir_list]
    # print(output_dir_list)

    # Returns the first tile in each input so that the datatype can be determined.
    # This is done up front, once per tile set, rather than on each chunk, since
    # all tiles have the same datatype for each input-- it only needs to be done once at the very beginning of the stage.
    main_logger.info(f"Getting tile_id of first tile in each tile set: {uu.timestr()}")
    first_tiles = uu.first_file_name_in_s3_folder(download_dict)
    # print(first_tiles)

    # Creates a download dictionary with the datatype of each input in the values.
    # This is supplied to each chunk that is being analyzed.
    # This also serves as a check of whether all inputs are being found (s3 paths correct)
    main_logger.info(f"Getting datatype of first tile in each tile set: {uu.timestr()}")
    download_dict_with_data_types = uu.add_file_type_to_dict(first_tiles)
    # print(download_dict_with_data_types)

    # Creates numpy array of ratios of BGC, deadwood C, and litter C relative to AGC. Relevant columns must be specified.
    mangrove_C_ratio_array = uu.convert_lookup_table_to_array(cn.RF_C_ratio_spreadsheet_full_path, cn.mangrove_rate_ratio_tab,
                                                              ['gainEcoCon', 'BGC_AGC', 'deadwood_AGC', 'litter_AGC'])

    # Makes a txt for each task in the list. These are deleted as tasks are completed.
    main_logger.info("Creating task txts in s3...")
    uu.create_s3_task_files(stage, chunk_list)


    ### Step 2: Create 1x1 degree outputs

    # Creates list of tasks to run (1 task = 1 chunk)
    main_logger.info(f"Creating tasks and starting processing: {uu.timestr()}")
    main_logger.info("Workers' logs to be appended after main function log"+ "\n")

    C_pool_1x1_deg_delayed_results = [dask.delayed(create_and_upload_starting_C_densities)
                       (chunk, mangrove_C_ratio_array, download_dict_with_data_types, year,
                        fishnet_iso_df, is_final, no_upload, output_dir_list, stage)
                       for chunk in chunk_list]

    # Runs analysis and gathers results
    C_pool_1x1_deg_results = dask.compute(*C_pool_1x1_deg_delayed_results)

    success_count_1x1, all_1x1_stats = uu.count_successful_chunks(chunk_list, is_final, main_logger, C_pool_1x1_deg_results)

    # Iterates through output folders and counts the number of output rasters (only if uploads enabled)
    if not no_upload:
        for output_folder in output_dir_list:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
            main_logger.info(f"Output rasters in {output_folder}: {file_count}")
            # print(geotiff_files)

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


    ### Step 3: Chunk stats for 1x1 degree outputs, aggregates logs

    # Resizes cluster down to 1 worker for chunk stats and log aggregation since that only needs a minimal remainder of the
    # cluster, not all the workers.
    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        # TODO Or maybe just have it terminate the cluster altogether, rather than resize it. Need to make sure that chunk stats and log still work, though.
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster("AFOLU_flux_model_scripts", 1)

    # Prepares 1x1 deg chunk stats spreadsheet: min, mean, max, and sum for all input and output chunks,
    # and min and max values across all chunks for all inputs and outputs
    # only if not suppressed by the --no_stats flag and at least one chunk was successfully (wasn't skipped).
    if (not no_stats) and (success_count_1x1 > 0):
        uu.compile_1x1_chunk_stats(all_1x1_stats, stage, no_upload, main_logger)

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats", main_logger)

    # Sets it so that no worker logs are created if doing a local run
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats, and worker log compilation", main_logger)

        # Adds the workers' logs to the main log and uploads to s3
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)

    # Closes the Dask client if not running locally
    if not run_local:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create carbon pools in 2000.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cshp', '--use_shapefile', action='store_true', help='Use shapefile to determine chunks')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('--year', type=int, required=True, help='Year for carbon pools')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    bounding_box = args.bounding_box
    chunk_size = args.chunk_size
    use_shapefile = args.use_shapefile
    first_chunks = args.first_chunks
    year = args.year
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    main(cluster_name, year, run_local, no_stats, no_log, no_upload, use_shapefile,
         bounding_box=bounding_box, chunk_size=chunk_size,
         first_chunks=first_chunks, log_note=log_note)

