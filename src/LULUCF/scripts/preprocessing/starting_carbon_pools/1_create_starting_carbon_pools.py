"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model/

Local:
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.1_create_starting_carbon_pools -bb 116 -3 116.25 -2.75 -cs 0.25 --run_local --no_stats --no_upload --year YYYY

Needs 4GB Coiled workers with 1 thread for 1x1 deg chunks; 2GB workers are too small.

Coiled small test:
python -m src.utilities.create_cluster -n 1 -t 1 -m 4 -cn LULUCF_preprocessing
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.1_create_starting_carbon_pools -cn LULUCF_preprocessing -bb 114 -4 115 -3 -cs 1  --year YYYY

Coiled shapefile test:
python -m src.utilities.create_cluster -n 1 -t 1 -m 4 -cn LULUCF_preprocessing
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.1_create_starting_carbon_pools -cn LULUCF_preprocessing -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -f 1 --year YYYY

Full run 2000:
python -m src.utilities.create_cluster -n 200 -t 1 -m 4 -cn LULUCF_preprocessing
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.1_create_starting_carbon_pools -cn LULUCF_preprocessing --year 2000 -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -ln "This is intended to be the definitive global run for carbon pool 2000 creation using GADM v4.1, raw and LC masked versions."
Peak memory per worker: ~2.8 GB
Time for numba processing for each task: ~1 second (based on scanning the console)
Time for total processing for each task: 15-25 seconds (based on scanning the console)
Time until chunk stats: 24:26 (no non-soil C sum), 25:43 (with non-soil C sum)
Time after chunk stats: 25:56 (no non-soil C sum), 27:06 (with non-soil C sum)
Coiled credits: 92 (no non-soil C sum), 99 (with non-soil C sum) (200/hr for 200 m8g.medium workers, according to dashboard)
AWS cost: $4.10 (no non-soil C sum), $4.50 (with non-soil C sum)

Full run 2015:
python -m src.utilities.create_cluster -n 200 -t 1 -m 4 -cn LULUCF_preprocessing
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.1_create_starting_carbon_pools -cn LULUCF_preprocessing --year 2015 -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -ln "This is intended to be the definitive global run for carbon pool 2015 creation using GADM v4.1, raw and LC masked versions."
Peak memory per worker: ~2.8 GB
Time for numba processing for each task: ~1 second (based on scanning the console)
Time for total processing for each task: 15-25 seconds (based on scanning the console)
Time until chunk stats: 24:53 (with non-soil C sum)
Time after chunk stats: 26:11 (with non-soil C sum)
Coiled credits: 91.5 (with non-soil C sum) (200/hr for 200 m8g.medium workers, according to dashboard)
AWS cost: $4.21 (with non-soil C sum)


To create a vrt of the 10x10 deg outputs, do:
aws s3 ls s3://gfw2-data/climate/ESA_CCI_biomass/v5_01/2015/year_2015_derived_carbon_pools/litter_C_density_MgC_ha/40000_pixels/ --recursive | grep .tif$ | awk '{print "/vsis3/gfw2-data/"$4}' > litter_C_2015_file_list.txt
gdalbuildvrt -input_file_list litter_C_2015_file_list.txt deadwood_C2015_mosaic.vrt
"""

import argparse
import concurrent.futures
import dask
import numpy as np
import os
import psutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from dask.distributed import print
from numba import jit

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import numba_utilities as nu
from src.utilities import universal_utilities as uu
from src.utilities import resize_cluster


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
    r_s_ratio_block = in_dict_float32[cn.r_s_ratio_non_mang_pattern]
    elevation_block = in_dict_int16[cn.elevation_pattern]
    climate_domain_block = in_dict_int16[cn.climate_domain_pattern]
    precipitation_block = in_dict_int32[cn.precipitation_pattern]
    continent_ecozone_block = in_dict_int16[cn.continent_ecozone_pattern]
    climate_zone_block = in_dict_uint8[cn.climate_zone_pattern]
    LC_composite_block = in_dict_uint8[cn.land_cover_pattern]  # LC composite of the year being processed. Pattern is the same regardless of starting year.
    veg_height_block = in_dict_uint8[cn.vegetation_height_pattern]  # Veg height of the year being processed. Pattern is the same regardless of starting year.
    mangrove_extent_block = in_dict_uint8[cn.mangrove_extent_processed_pattern]  # Veg height of the year being processed. Pattern is the same regardless of starting year.

    # AGB block sources (mangrove and non-mangrove) depend on the starting year
    # Numba can't handle two different possible datatypes for agb_non_mang_block,
    # so both options need to be recast to the same type
    if year == 2000:
        agb_non_mang_block = in_dict_int16[cn.agb_2000_pattern].astype(np.int16)
        mangrove_agb_block = in_dict_float32[cn.mangrove_agb_2000_pattern]
    elif year == 2015: # No mangrove-specific AGB for 2015. Need to supply something for mangroves for completeness.
        agb_non_mang_block = in_dict_uint16[cn.agb_2015_pattern].astype(np.int16)
        mangrove_agb_block = np.zeros(in_dict_float32[cn.r_s_ratio_non_mang_pattern].shape).astype('float32')
    else:
        out_dict_float32[f"{cn.agc_raw_dens_pattern}_{year}"] = np.full(in_dict_float32[cn.r_s_ratio_non_mang_pattern].shape, 9999).astype('float32')
        return out_dict_float32

    mangrove_in_chunk = True  # Flag for whether chunk has mangrove in it
    agb_non_mang_in_chunk = True  # Flag for whether chunk has non-mangrove AGB in it

    # Checks if the chunk has various inputs by seeing if the max value is 0.
    # If the max value is 0, it assumes that input doesn't exist.
    if agb_non_mang_block.max() == 0:
        agb_non_mang_in_chunk = False
    if (mangrove_agb_block.max() == 0) and (mangrove_extent_block.max() == 0):
        mangrove_in_chunk = False

    # print("agb_non_mang_in_chunk:", agb_non_mang_in_chunk)
    # print("mangrove_in_chunk:", mangrove_in_chunk)

    # Output blocks
    # Need to specify the output datatype or it will default to float32
    agc_raw_out_block = np.zeros(in_dict_float32[cn.r_s_ratio_non_mang_pattern].shape).astype('float32')
    bgc_raw_out_block = np.zeros(in_dict_float32[cn.r_s_ratio_non_mang_pattern].shape).astype('float32')
    deadwood_c_raw_out_block = np.zeros(in_dict_float32[cn.r_s_ratio_non_mang_pattern].shape).astype('float32')
    litter_c_raw_out_block = np.zeros(in_dict_float32[cn.r_s_ratio_non_mang_pattern].shape).astype('float32')
    non_soil_c_raw_out_block = np.zeros(in_dict_float32[cn.r_s_ratio_non_mang_pattern].shape).astype('float32')

    agc_LC_masked_out_block = np.zeros(in_dict_float32[cn.r_s_ratio_non_mang_pattern].shape).astype('float32')
    bgc_LC_masked_out_block = np.zeros(in_dict_float32[cn.r_s_ratio_non_mang_pattern].shape).astype('float32')
    deadwood_c_LC_masked_out_block = np.zeros(in_dict_float32[cn.r_s_ratio_non_mang_pattern].shape).astype('float32')
    litter_c_LC_masked_out_block = np.zeros(in_dict_float32[cn.r_s_ratio_non_mang_pattern].shape).astype('float32')
    non_soil_c_LC_masked_out_block = np.zeros(in_dict_float32[cn.r_s_ratio_non_mang_pattern].shape).astype('float32')

    # Gets a fallback value for continent_ecozone for the chunk in case some pixels don't have one
    continent_ecozone_fallback = nu.fallback_conteco_climzone_value(continent_ecozone_block, 2020)

    # Sets a fallback value for climate zone for the chunk in case any pixels fall outside the climate zone boundary.
    # fallback_value is only used if the chunk doesn't have any climate_zone pixels in it at all.
    climate_zone_fallback = nu.fallback_conteco_climzone_value(climate_zone_block, 5)

    # Iterates through all pixels in the chunk
    for row in range(continent_ecozone_block.shape[0]):
        for col in range(continent_ecozone_block.shape[1]):

            ### Part 1: Pixel-level values

            # Input values for this specific cell
            agb_non_mang_cell = agb_non_mang_block[row, col]
            mangrove_agb_cell = mangrove_agb_block[row, col]
            elevation = elevation_block[row, col]
            climate_domain = climate_domain_block[row, col]
            precipitation = precipitation_block[row, col]
            r_s_ratio = r_s_ratio_block[row, col]
            continent_ecozone_cell = continent_ecozone_block[row, col]
            climate_zone_cell = climate_zone_block[row, col]
            LC_composite_cell = LC_composite_block[row, col]
            veg_height_cell = veg_height_block[row, col]
            mangrove_extent_cell = mangrove_extent_block[row, col]

            # Applies the continent_ecozne fallback value when there isn't a value for the pixel
            if continent_ecozone_cell == 0:
                continent_ecozone_cell = continent_ecozone_fallback

            # Applies the climate_zone fallback value when there isn't a value for the pixel
            if climate_zone_cell == 0:
                climate_zone_cell = climate_zone_fallback

            # Criteria for classifying pixels as mangrove pixels: If mangrove AGB or GMWv3 is present
            # mangrove_pixel = (mangrove_in_chunk) and ((mangrove_agb_cell > 0) or (mangrove_extent_cell > 0))
            mangrove_pixel = (mangrove_agb_cell > 0) or (mangrove_extent_cell > 0)


            ### Part 2: Calculation of raw carbon density outputs (not masked by veg height/land cover)

            # If mangrove pixel, AGC is calculated from it, overwriting any AGC that is based on non-mang AGB that is already there
            if mangrove_pixel == True:  # Only uses AGB if chunk exists and there is a value in that pixel
                if year == 2000:
                    agc_raw_out_cell = mangrove_agb_cell * cn.biomass_to_carbon_mangrove
                elif year == 2015:
                    agc_raw_out_cell = agb_non_mang_cell * cn.biomass_to_carbon_mangrove
                else:
                    raise ValueError("start_year not valid: must be 2000 or 2015")

            # If non-mang AGB is present, AGC is calculated from it
            # elif (agb_non_mang_in_chunk) and (agb_non_mang_cell > 0):  # Only uses AGB if chunk exists and there is a value in that pixel
            elif (agb_non_mang_cell > 0):  # Only uses AGB if chunk exists and there is a value in that pixel
                agc_raw_out_cell = agb_non_mang_cell * cn.biomass_to_carbon_non_mangrove

            else:
                agc_raw_out_cell = 0

            # Separate branches for assigning BGC, deadwood C, and litter C ratios depending on whether the pixel has mangroves.
            # Calculation of BGC, deadwood C, and litter C are done after the decision tree assigns the ratios.

            # Mangrove carbon pool ratio branch
            # From IPCC 2013 Wetland Supplement
            if mangrove_pixel == True:  # Only replaces non-mangrove AGB if mangrove chunk exists and if mangrove value in that pixel

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
            # elif (agb_non_mang_in_chunk) and (agb_non_mang_cell > 0):  # Non-mangrove
            elif (agb_non_mang_cell > 0):  # Non-mangrove

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

            # Actually calculates BGC, deadwood C, and litter C using the ratios assigned in the above decision tree for raw outputs
            bgc_raw_out_cell = agc_raw_out_cell * bgc_ratio
            deadwood_c_raw_out_cell = agc_raw_out_cell * deadwood_c_ratio
            litter_c_raw_out_cell = agc_raw_out_cell * litter_c_ratio


            ### Part 3: Calculation of carbon density outputs masked by veg height/land cover

            short_veg_LC, tall_veg_LC = nu.classify_veg_height(LC_composite_cell)

            # Carbon density for short vegetation (Mg C/ha) based on climate zone (IPCC default)
            short_veg_AGC, short_veg_BGC = nu.calc_short_veg_removals(climate_zone_cell)

            # Assigns carbon densities based on vegetation height and composite landcover.
            # Tall vegetation and/or mangrove (i.e. mangrove AGB pixels without tall vegetation keep all C pools)
            #TODO Need to include GMWv3 1996 extent as another condition for keeping raw carbon pool outputs.
            # Otherwise, pixels that are treated as mangrove in the model because they have GMWv3 but don't have
            # mangrove AGB or tall vegetation get assigned other C pool values.
            # e.g., 114.41835E, 3.4814S (in 00N_110E) has WHRC AGC of 47.47, mangrove AGC of 0, and a
            # LC-masked AGC of 2.914 because the composite LC is short veg (code 124). However, there is GMW in 1996/2000,
            # so this pixel should maintain the tree-relevant C-pools rather then be reclassified as short veg.
            # This requires adding GMWv3 1996 as another input to this script and adding another or statement here.
            # It can be tested as the coordinate mentioned above.
            if (veg_height_cell >= cn.tree_threshold) or (mangrove_pixel == True):
                agc_LC_masked_out_cell = agc_raw_out_cell
                bgc_LC_masked_out_cell = bgc_raw_out_cell
                deadwood_c_LC_masked_out_cell = deadwood_c_raw_out_cell
                litter_c_LC_masked_out_cell = litter_c_raw_out_cell
            elif short_veg_LC:  # Short vegetation
                agc_LC_masked_out_cell = short_veg_AGC
                bgc_LC_masked_out_cell = short_veg_BGC
                deadwood_c_LC_masked_out_cell = 0
                litter_c_LC_masked_out_cell = 0
            elif LC_composite_cell == cn.cropland:  # Cropland
                agc_LC_masked_out_cell = cn.cropland_agc_dens
                bgc_LC_masked_out_cell = 0
                deadwood_c_LC_masked_out_cell = 0
                litter_c_LC_masked_out_cell = 0
            else:  # Anything else
                agc_LC_masked_out_cell = 0
                bgc_LC_masked_out_cell = 0
                deadwood_c_LC_masked_out_cell = 0
                litter_c_LC_masked_out_cell = 0

            # Assigns cell outputs to blocks
            agc_raw_out_block[row, col] = agc_raw_out_cell
            bgc_raw_out_block[row, col] = bgc_raw_out_cell
            deadwood_c_raw_out_block[row, col] = deadwood_c_raw_out_cell
            litter_c_raw_out_block[row, col] = litter_c_raw_out_cell
            non_soil_c_raw_out_block[row, col] = (agc_raw_out_cell + bgc_raw_out_cell +
                                                  deadwood_c_raw_out_cell + litter_c_raw_out_cell)

            agc_LC_masked_out_block[row, col] = agc_LC_masked_out_cell
            bgc_LC_masked_out_block[row, col] = bgc_LC_masked_out_cell
            deadwood_c_LC_masked_out_block[row, col] = deadwood_c_LC_masked_out_cell
            litter_c_LC_masked_out_block[row, col] = litter_c_LC_masked_out_cell
            non_soil_c_LC_masked_out_block[row, col] = (agc_LC_masked_out_cell + bgc_LC_masked_out_cell +
                                                        deadwood_c_LC_masked_out_cell + litter_c_LC_masked_out_cell)

    # Adds the output arrays to the dictionary with the appropriate data type
    # Outputs need .copy() so that previous intervals' arrays in dictionary aren't overwritten because arrays in dictionaries are mutable (courtesy of ChatGPT).
    out_dict_float32[f"{cn.agc_raw_dens_pattern}{cn.C_density_pixel_meaning}_{year}"] = agc_raw_out_block.copy()
    out_dict_float32[f"{cn.bgc_raw_dens_pattern}{cn.C_density_pixel_meaning}_{year}"] = bgc_raw_out_block.copy()
    out_dict_float32[f"{cn.deadwood_c_raw_dens_pattern}{cn.C_density_pixel_meaning}_{year}"] = deadwood_c_raw_out_block.copy()
    out_dict_float32[f"{cn.litter_c_raw_dens_pattern}{cn.C_density_pixel_meaning}_{year}"] = litter_c_raw_out_block.copy()
    out_dict_float32[f"{cn.non_soil_c_raw_dens_pattern}{cn.C_density_pixel_meaning}_{year}"] = non_soil_c_raw_out_block.copy()

    out_dict_float32[f"{cn.agc_LC_masked_dens_pattern}{cn.C_density_pixel_meaning}_{year}"] = agc_LC_masked_out_block.copy()
    out_dict_float32[f"{cn.bgc_LC_masked_dens_pattern}{cn.C_density_pixel_meaning}_{year}"] = bgc_LC_masked_out_block.copy()
    out_dict_float32[f"{cn.deadwood_c_LC_masked_dens_pattern}{cn.C_density_pixel_meaning}_{year}"] = deadwood_c_LC_masked_out_block.copy()
    out_dict_float32[f"{cn.litter_c_LC_masked_dens_pattern}{cn.C_density_pixel_meaning}_{year}"] = litter_c_LC_masked_out_block.copy()
    out_dict_float32[f"{cn.non_soil_c_LC_masked_dens_pattern}{cn.C_density_pixel_meaning}_{year}"] = non_soil_c_LC_masked_out_block.copy()

    # return output dictionary/ies
    return out_dict_float32



# All steps for creating starting non-soil carbon pools in a chunk: download chunks, calculate carbon densities, upload to s3
def create_and_upload_starting_C_densities(bounds, mangrove_C_ratio_array, download_dict_with_data_types, year,
                                           is_final, no_upload, output_folders, stage):

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []

    process = psutil.Process(os.getpid())

    logger_worker = lu.setup_logging_worker()

    chunk_start_time = time.time()

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
        if 'success' not in status: # Prints and logs any inputs that couldn't be accessed (downloaded as all 0s) or had to be padded
            lu.print_and_log(f"{status}: {uu.timestr()}", False, logger_worker)
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
    numba_start = time.time()

    # Create AGC, BGC, deadwood C and litter C densities in selected starting year
    out_dict_float32 = create_starting_C_densities(
        typed_dict_uint8, typed_dict_uint16, typed_dict_int16, typed_dict_int32, typed_dict_float32,
        mangrove_C_ratio_array, year
    )

    numba_end = time.time()
    lu.print_and_log(f"Done calculating carbon densities in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)
    lu.print_and_log(f"Memory usage after numba calculations completed for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB", is_final, logger_worker)
    lu.print_and_log(f"Calculated carbon densities in {bounds_str} in {tile_id} in {round(numba_end-numba_start)} seconds: {uu.timestr()}", False, logger_worker)


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

    chunk_end_time = time.time()
    lu.print_and_log(f"{bounds_str} took {round(chunk_end_time - chunk_start_time)} seconds: {uu.timestr()}", False, logger_worker)

    return_message = f"Success creating initial carbon pools for {bounds_str}: {uu.timestr()}"

    # Removes task tracking file from S3 once task is successful
    uu.delete_s3_task_file(stage, bounds, is_final, logger_worker)

    return return_message, chunk_stats  # Return both the success message and the statistics


def main(cluster_name, year, run_local=False, no_stats=False, no_log=False, no_upload= False,
         chunk_shapefile_uri=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):

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

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    # Starting time for stage
    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Year for carbon pools: {year}")

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size, first_chunks, fishnet_iso_df, main_logger)

    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_final = False
    # is_final = True  # For simulating a large run
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
        cn.climate_zone_pattern: f"{cn.climate_zone_processed_dir}{sample_tile_id}_{cn.climate_zone_pattern}.tif",
        cn.precipitation_pattern: f"{cn.precipitation_dir}{sample_tile_id}_{cn.precipitation_pattern}.tif",
        cn.r_s_ratio_non_mang_pattern: f"{cn.r_s_ratio_non_mang_dir}{sample_tile_id}_{cn.r_s_ratio_non_mang_pattern}.tif",
        cn.continent_ecozone_pattern: f"{cn.continent_ecozone_dir}{sample_tile_id}_{cn.continent_ecozone_pattern}.tif"
    }

    # Dictionary of data to download
    if year == 2000:
        download_dict[cn.agb_2000_pattern] = f"{cn.agb_2000_dir}{sample_tile_id}_{cn.agb_2000_pattern}.tif"
        download_dict[cn.mangrove_agb_2000_pattern] = f"{cn.mangrove_agb_2000_dir}{sample_tile_id}_{cn.mangrove_agb_2000_pattern}.tif"
        download_dict[cn.land_cover_pattern] = f"{cn.land_cover_5_year_path}2000/{sample_tile_id}.tif"
        download_dict[cn.vegetation_height_pattern] = f"{cn.vegetation_height_5_year_path}2000/{sample_tile_id}_{cn.vegetation_height_5_year_pattern}_2000.tif"
        download_dict[cn.mangrove_extent_processed_pattern] = f"{cn.mangrove_extent_processed_dir}1996/{sample_tile_id}__{cn.mangrove_extent_processed_pattern}_1996.tif"

        output_dir_list = [cn.agc_2000_raw_dir, cn.bgc_2000_raw_dir, cn.deadwood_c_2000_raw_dir, cn.litter_c_2000_raw_dir, cn.non_soil_c_2000_raw_dir,
                           cn.agc_2000_LC_masked_dir, cn.bgc_2000_LC_masked_dir, cn.deadwood_c_2000_LC_masked_dir, cn.litter_c_2000_LC_masked_dir, cn.non_soil_c_2000_LC_masked_dir]

    elif year == 2015:   # No mangrove-specific AGB for 2015
        download_dict[cn.agb_2015_pattern] = f"{cn.agb_2015_dir_processed}{sample_tile_id}_{cn.agb_2015_pattern}.tif"
        download_dict[cn.land_cover_pattern] = f"{cn.land_cover_annual_path}2015/{sample_tile_id}.tif"
        download_dict[cn.vegetation_height_pattern] = f"{cn.vegetation_height_annual_path}2015/{sample_tile_id}.tif"
        download_dict[cn.mangrove_extent_processed_pattern] = f"{cn.mangrove_extent_processed_dir}2015/{sample_tile_id}__{cn.mangrove_extent_processed_pattern}_2015.tif"

        output_dir_list = [cn.agc_2015_raw_dir, cn.bgc_2015_raw_dir, cn.deadwood_c_2015_raw_dir, cn.litter_c_2015_raw_dir, cn.non_soil_c_2015_raw_dir,
                           cn.agc_2015_LC_masked_dir, cn.bgc_2015_LC_masked_dir, cn.deadwood_c_2015_LC_masked_dir, cn.litter_c_2015_LC_masked_dir, cn.non_soil_c_2015_LC_masked_dir]

    else:
        print(f"Year input {year} not valid. Terminating.")
        sys.exit()
    # print(download_dict)

    # Creates list of output directories specific to the run
    output_dir_list = [path.replace("CHUNK_SIZE", str(chunk_size_pixels)) for path in output_dir_list]
    output_dir_list = [path.replace("PER_HA_OR_PIXEL", cn.C_density_pixel_meaning) for path in output_dir_list]
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
                        is_final, no_upload, output_dir_list, stage)
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
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster(cluster_name, 1)

    # Prepares 1x1 deg chunk stats spreadsheet: min, mean, max, and sum for all input and output chunks,
    # and min and max values across all chunks for all inputs and outputs
    # only if not suppressed by the --no_stats flag and at least one chunk was successfully (wasn't skipped).
    if (not no_stats) and (success_count_1x1 > 0):
        uu.compile_1x1_chunk_stats(all_1x1_stats, chunk_shapefile_uri, stage, no_upload, main_logger)

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
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('--year', type=int, required=True, help='Year for carbon pools: must be 2000 or 2015')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    bounding_box = args.bounding_box
    chunk_size = args.chunk_size
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_chunks = args.first_chunks
    year = args.year
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    main(cluster_name, year, run_local, no_stats, no_log, no_upload, chunk_shapefile_uri,
         bounding_box=bounding_box, chunk_size=chunk_size,
         first_chunks=first_chunks, log_note=log_note)

