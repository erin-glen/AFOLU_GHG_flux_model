"""
Run from src/LULUCF/

Local:
python -m scripts.preprocessing.gmw_smooth_mangrove_extent_timeseries.0_smooth_mangrove_extent.py -bb 116 -3 116.25 -2.75 -cs 0.25 --run_local --no_stats --no_upload

todo:
- see if it can scale from 1 degree to 10 degrees after testing 

"""
import argparse
import sys
import numpy as np
from numba import jit
import dask
from dask.distributed import print
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# Project imports
from ...utilities import constants_and_names as cn
from ...utilities import universal_utilities as uu
from ...utilities import log_utilities as lu
from ...utilities import numba_utilities as nu
from ...utilities import resize_cluster

#Function to smooth the mangrove data:
    # fills in 0 data if there is mangrove extent (1) in the previous year and the following year
    # then removes false positives (1) if there is no mangrove extent (0) in the previous year and the following year
@jit(nopython=True)
def smooth_mangrove_data(in_dict_uint8):
    # Dictionary for output numpy arrays
    out_dict_uint8 = {}

    #Input data blocks: hansenized mangrove extent
    mangrove_extent_1996_block = in_dict_uint8["mangrove_extent_1996"]  #TODO fill in pattern from cn
    mangrove_extent_2007_block = in_dict_uint8["mangrove_extent_2007"]
    mangrove_extent_2008_block = in_dict_uint8["mangrove_extent_2008"]
    mangrove_extent_2009_block = in_dict_uint8["mangrove_extent_2009"]
    mangrove_extent_2010_block = in_dict_uint8["mangrove_extent_2010"]
    mangrove_extent_2015_block = in_dict_uint8["mangrove_extent_2015"]
    mangrove_extent_2016_block = in_dict_uint8["mangrove_extent_2016"]
    mangrove_extent_2017_block = in_dict_uint8["mangrove_extent_2017"]
    mangrove_extent_2018_block = in_dict_uint8["mangrove_extent_2018"]
    mangrove_extent_2019_block = in_dict_uint8["mangrove_extent_2019"]
    mangrove_extent_2020_block = in_dict_uint8["mangrove_extent_2020"]

    # Output data blocks: smoothed mangrove extent (empty blocks)
    # Initial and final year of consecutive years are not modified, only the years in between.
    # So smoothed data for those years will be identical to raw data and copied directly from input blocls
    mangrove_extent_2008_out_block = np.zeros(in_dict_uint8["mangrove_extent_2008"].shape).astype('uint8') #TODO fix pattern with cn
    mangrove_extent_2009_out_block = np.zeros(in_dict_uint8["mangrove_extent_2009"].shape).astype('uint8')
    mangrove_extent_2016_out_block = np.zeros(in_dict_uint8["mangrove_extent_2016"].shape).astype('uint8')
    mangrove_extent_2017_out_block = np.zeros(in_dict_uint8["mangrove_extent_2017"].shape).astype('uint8')
    mangrove_extent_2018_out_block = np.zeros(in_dict_uint8["mangrove_extent_2018"].shape).astype('uint8')
    mangrove_extent_2019_out_block = np.zeros(in_dict_uint8["mangrove_extent_2019"].shape).astype('uint8')

    # STEP 1: Fill in missing mangrove extent years (i.e. "false negatives") using raw data
    # Iterates through all pixels in the chunk
    for row in range(mangrove_extent_2007_block.shape[0]):
        for col in range(mangrove_extent_2007_block.shape[1]):

            # Input values for specific cell
            mangrove_extent_2007 = mangrove_extent_2007_block[row, col]
            mangrove_extent_2008 = mangrove_extent_2008_block[row, col]
            mangrove_extent_2009 = mangrove_extent_2009_block[row, col]
            mangrove_extent_2010 = mangrove_extent_2010_block[row, col]
            mangrove_extent_2015 = mangrove_extent_2015_block[row, col]
            mangrove_extent_2016 = mangrove_extent_2016_block[row, col]
            mangrove_extent_2017 = mangrove_extent_2017_block[row, col]
            mangrove_extent_2018 = mangrove_extent_2018_block[row, col]
            mangrove_extent_2019 = mangrove_extent_2019_block[row, col]
            mangrove_extent_2020 = mangrove_extent_2020_block[row, col]

            # Not modifying 2007 since it is the start of the consecutive time period (2007 - 2010)
            # Adding in mangrove extent in 2008 if there is mangrove extent in 2007 and 2009
            if mangrove_extent_2008 == 1:
                mangrove_extent_2008_out_block[row, col] = 1
            elif (mangrove_extent_2007 == 1 and mangrove_extent_2009 == 1):
                mangrove_extent_2008_out_block[row, col] = 1
            else:
                mangrove_extent_2008_out_block[row, col] = 0
            # TODO: raise error if neither 0 nor 1 throughout

            # Adding in mangrove extent in 2009 if there is mangrove extent in 2008 and 2010
            if mangrove_extent_2009 == 1:
                mangrove_extent_2009_out_block[row, col] = 1
            elif (mangrove_extent_2008 == 1 and mangrove_extent_2010 == 1):
                mangrove_extent_2009_out_block[row, col] = 1
            else:
                mangrove_extent_2009_out_block[row, col] = 0

            # Not modifying 2010 since it is the end of the consecutive time period (2007 - 2010)
            # Not modifying 2015 since it is the start of the consecutive time period (2015 - 2020)
            # Adding in mangrove extent in 2016 if there is mangrove extent in 2015 and 2017
            if mangrove_extent_2016 == 1:
                mangrove_extent_2016_out_block[row, col] = 1
            elif (mangrove_extent_2015 == 1 and mangrove_extent_2017 == 1):
                mangrove_extent_2016_out_block[row, col] = 1
            else:
                mangrove_extent_2016_out_block[row, col] = 0

            # Adding in mangrove extent in 2017 if there is mangrove extent in 2016 and 2018
            if mangrove_extent_2017 == 1:
                mangrove_extent_2017_out_block[row, col] = 1
            elif (mangrove_extent_2016 == 1 and mangrove_extent_2018 == 1):
                mangrove_extent_2017_out_block[row, col] = 1
            else:
                mangrove_extent_2017_out_block[row, col] = 0

            # Adding in mangrove extent in 2018 if there is mangrove extent in 2017 and 2019
            if mangrove_extent_2018 == 1:
                mangrove_extent_2018_out_block[row, col] = 1
            elif (mangrove_extent_2017 == 1 and mangrove_extent_2019 == 1):
                mangrove_extent_2018_out_block[row, col] = 1
            else:
                mangrove_extent_2018_out_block[row, col] = 0

            # Adding in mangrove extent in 2019 if there is mangrove extent in 2018 and 2020
            if mangrove_extent_2019 == 1:
                mangrove_extent_2019_out_block[row, col] = 1
            elif (mangrove_extent_2018 == 1 and mangrove_extent_2020 == 1):
                mangrove_extent_2019_out_block[row, col] = 1
            else:
                mangrove_extent_2019_out_block[row, col] = 0
            # Not modifying 2020 since it is the end of the consecutive time period (2015 - 2020)

    # TODO: Check that this is writing over out_block correctly
    # STEP 2: Remove "false positive" in mangrove extent years using processed out_block data
    # Iterates through all pixels in the chunk
    for row in range(mangrove_extent_2007_block.shape[0]):
        for col in range(mangrove_extent_2007_block.shape[1]):

            # Input values for specific cell (use smoothed data)
            mangrove_extent_2007 = mangrove_extent_2007_block[row, col]
            mangrove_extent_2008 = mangrove_extent_2008_out_block[row, col]
            mangrove_extent_2009 = mangrove_extent_2009_out_block[row, col]
            mangrove_extent_2010 = mangrove_extent_2010[row, col]
            mangrove_extent_2015 = mangrove_extent_2015[row, col]
            mangrove_extent_2016 = mangrove_extent_2016_out_block[row, col]
            mangrove_extent_2017 = mangrove_extent_2017_out_block[row, col]
            mangrove_extent_2018 = mangrove_extent_2018_out_block[row, col]
            mangrove_extent_2019 = mangrove_extent_2019_out_block[row, col]
            mangrove_extent_2020 = mangrove_extent_2020[row, col]

            # Removing mangrove extent in 2008 if there is no mangrove extent in 2007 and 2009
            if (mangrove_extent_2008 == 1 and mangrove_extent_2007 == 0 and mangrove_extent_2009 == 0):
                mangrove_extent_2008_out_block[row, col] = 0

            # Removing mangrove extent in 2009 if there is no mangrove extent in 2008 and 2010
            if (mangrove_extent_2009 == 1 and mangrove_extent_2008 == 0 and mangrove_extent_2010 == 0):
                mangrove_extent_2009_out_block[row, col] = 0

            # Removing mangrove extent in 2016 if there is no mangrove extent in 2015 and 2017
            if (mangrove_extent_2016 == 1 and mangrove_extent_2015 == 0 and mangrove_extent_2017 == 0):
                mangrove_extent_2016_out_block[row, col] = 0

            # Removing mangrove extent in 2017 if there is no mangrove extent in 2016 and 2018
            if (mangrove_extent_2017 == 1 and mangrove_extent_2016 == 0 and mangrove_extent_2018 == 0):
                mangrove_extent_2017_out_block[row, col] = 0

            # Removing mangrove extent in 2018 if there is no mangrove extent in 2017 and 2019
            if (mangrove_extent_2018_ == 1 and mangrove_extent_2017 == 0 and mangrove_extent_2019 == 0):
                mangrove_extent_2018_out_block[row, col] = 0

            # Removing mangrove extent in 2019 if there is no mangrove extent in 2018 and 2020
            if (mangrove_extent_2019 == 1 and mangrove_extent_2018 == 0 and mangrove_extent_2020 == 0):
                mangrove_extent_2019_out_block[row, col] = 0

    # Adds the output arrays to the output data dictionary  #TODO use cn pattern
    out_dict_uint8["mangrove_extent_1996_smoothed"] = mangrove_extent_1996_block.copy()     #same as raw data
    out_dict_uint8["mangrove_extent_2007_smoothed"] = mangrove_extent_2007_block.copy()     #same as raw data
    out_dict_uint8["mangrove_extent_2008_smoothed"] = mangrove_extent_2008_out_block.copy()
    out_dict_uint8["mangrove_extent_2009_smoothed"] = mangrove_extent_2009_out_block.copy()
    out_dict_uint8["mangrove_extent_2010_smoothed"] = mangrove_extent_2010_block.copy()     #same as raw data
    out_dict_uint8["mangrove_extent_2015_smoothed"] = mangrove_extent_2015_block.copy()     #same as raw data
    out_dict_uint8["mangrove_extent_2016_smoothed"] = mangrove_extent_2016_out_block.copy()
    out_dict_uint8["mangrove_extent_2017_smoothed"] = mangrove_extent_2017_out_block.copy()
    out_dict_uint8["mangrove_extent_2018_smoothed"] = mangrove_extent_2018_out_block.copy()
    out_dict_uint8["mangrove_extent_2019_smoothed"] = mangrove_extent_2019_out_block.copy()
    out_dict_uint8["mangrove_extent_2020_smoothed"] = mangrove_extent_2020_block.copy()     #same as raw data

    return out_dict_uint8

# All steps for creating smoothed mangrove data: download chunks, smooth data, upload to s3
def preprocess_and_upload_smoothed_mangrove_data(bounds, download_dict_with_data_types, is_final, no_upload, output_folders, stage):

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []

    logger_worker = lu.setup_logging_worker()

    uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_final, logger_worker)

    bounds_str = uu.boundstr(bounds)  # String form of chunk bounds
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)


    ### Part 1: Downloads chunk.

    # Replaces the placeholder tile_id in the download data dictionary from main with the tile_id for this chunk
    updated_download_dict = uu.replace_tile_id_in_dict(download_dict_with_data_types, tile_id)

    # If a particular tile doesn't exist for an input, an array of 0s of the correct size and datatype is returned instead.
    futures = uu.prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_final, logger_worker)
    print(futures)

    lu.print_and_log(f"Waiting for requests for data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

    # Dictionary that stores the dataset name (key) and downloaded data and download status (values)
    layers = {}

    # Ensures futures stores Future objects
    for future in concurrent.futures.as_completed(futures):
        layer = futures[future]
        data, status = future.result()
        if 'success' not in status: # Prints and logs any inputs that couldn't be accessed and are downloaded as all 0s
            lu.print_and_log(f"{status}", is_final, logger_worker)
        layers[layer] = data

    # Test prints
    print(layers)
    # print(layers[''].max())
    # print(layers[''].dtype)


    ### Part 2: Numba functions can accept (and return) dictionaries of arrays as long as each dictionary only has arrays of one data type

    lu.print_and_log(f"Creating typed dictionaries for chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

    # Creates the typed dictionaries for all input layers (including those that originally had no data)
    typed_dict_uint8, typed_dict_uint16, typed_dict_int16, typed_dict_int32, typed_dict_float32 = nu.create_typed_dicts(layers)
    print("uint8_typed_list:", typed_dict_uint8)
    #TODO: Create only uint8 or delete other types


    ### Part 3: Creates mangrove extent rasters

    lu.print_and_log(f"Creating preprocessed mangrove data in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker) # Prints during full runs
    uu.rename_s3_task_file(stage, bounds, "calculating_", is_final, logger_worker)

    # Create smoothed mangrove data
    out_dict_uint8 = smooth_mangrove_data(in_dict_uint8)

    # Fresh non-Numba-constrained dictionary that stores all output numpy arrays of all datatypes.
    # The dictionaries by datatype that are returned from the numba function have limitations on them,
    # e.g., they can't be combined with other datatypes. This prevents the addition of attributes needed for uploading to s3.
    # So the trick here is to copy the numba-exported arrays into normal Python arrays to which we can do anything in Python.
    out_dict_all_dtypes = {}

    # Transfers the dictionaries of numpy arrays for each data type to a new, Pythonic array
    out_dicts = [out_dict_uint8]

    # Loop through each dictionary and update out_dict_all_dtypes
    for out_dict in out_dicts:
        for key, value in out_dict.items():
            out_dict_all_dtypes[key] = value

        # Clear memory of unneeded arrays
        del out_dict

    ##TODO calculate extent of mangrove for each year and min and max value for each chunk


    ### Part 4: Saves numpy arrays as rasters and uploads to s3

    uu.rename_s3_task_file(stage, bounds, "uploading_", is_final, logger_worker)

    # Only saves arrays to geotifs and uploads them to s3 if enabled
    if not no_upload:

        out_no_data_val = 0  # NoData value for output raster (optional)

        # Adds metadata used for uploading outputs to s3 to the dictionary
        for key, value in out_dict_all_dtypes.items():

            data_type = value.dtype.name
            # print("key:", key)

            # Retrieves the file name pattern and date(s) covered for the output file for use in s3 folder construction
            out_pattern, year_range = uu.strip_and_extract_years(key)
            # print("out_pattern:", out_pattern)
            # print("year_range:", year_range)

            # Gets the core filename pattern and pixel meaning
            out_pattern_without_pixel_meaning, pixel_meaning = uu.strip_pixel_meaning(out_pattern)
            # print("out_pattern_without_pixel_meaning:", out_pattern_without_pixel_meaning)

            # Retrieves the relevant output s3 path for this specific output  (list of one element)
            matched_output_s3_folder = [item for item in output_folders if out_pattern_without_pixel_meaning in item][0]
            # print("matched_output_s3_folder:", matched_output_s3_folder)

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

    return_message = f"Success creating smoothed mangrove extent raster for {bounds_str}: {uu.timestr()}"

    # Removes task tracking file from S3 once task is successful
    uu.delete_s3_task_file(stage, bounds, is_final, logger_worker)

    return return_message, chunk_stats  # Return both the success message and the statistics