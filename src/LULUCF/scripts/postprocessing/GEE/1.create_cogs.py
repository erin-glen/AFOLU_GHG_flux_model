"""
Script to create global, stacked COGS:
1) builds global VRT per dataset per year from all 10 x 10 tiles in input s3 dir,
2) builds a global COG for per dataset per year, and
3) combine annual COGs into single, stacked global COG per dataset

#TODO: Update creation option based on GDAL type. Check blocksize. Right now using COs for float. Look up optimal COs by datatype.
"""
import os
import argparse
import time
import subprocess

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import universal_utilities as uu
from src.utilities import log_utilities as lu

#TODO: move to UU
def create_cog_from_vrt(output_vrt_s3_path, tmp_cog_path, dt, output_cog_s3_path):
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.vrt,.ovr,.aux.xml")

    logger_worker = lu.setup_logging_worker()

    # Check if the COG already exists in S3
    if uu.exists_in_s3(output_cog_s3_path):
        return lu.print_and_log(f"COG file already exists in S3: {output_cog_s3_path}. Skipping creation.", False, logger_worker)

    #TODO can change to gdal python package instead of subprocess
    cmd = [
        "gdal_translate",
        output_vrt_s3_path.replace("s3://", "/vsis3/"),
        tmp_cog_path,
        "-of", "COG",
        "-ot", f"{dt}",
        "-co", "COMPRESS=DEFLATE",
        "-co", "BLOCKSIZE=1024",
        "-co", "PREDICTOR=3",
        "-co", "BIGTIFF=IF_SAFER",
        "-co", f"OVERVIEW_RESAMPLING=AVERAGE",
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    lu.print_and_log(f"Built local COG: {tmp_cog_path}", False, logger_worker)

    # Upload COG to S3
    uu.upload_s3_file(output_cog_s3_path, tmp_cog_path)

    # If successfully uploaded to s3, delete local COG #TODO: Make microservice to delete local file if uploaded to s3 to avoid duplicate code
    if uu.exists_in_s3(output_cog_s3_path):
        lu.print_and_log(f"Uploaded COG to S3: {output_cog_s3_path}", False, logger_worker)
        try:
            os.remove(tmp_cog_path)
        except Exception as e:
            lu.print_and_log(f"Warning: could not delete {tmp_cog_path}: {e}", False, logger_worker)

def stack_annual_cogs(dataset_name, dt, year_to_cog_s3, tmp_stacked_vrt, tmp_stacked_cog, output_stacked_vrt_s3_path, output_stacked_cog_s3_path):
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.vrt,.ovr,.aux.xml")

    logger_worker = lu.setup_logging_worker()

    # Check if the stacked COG already exists in S3
    if uu.exists_in_s3(output_stacked_cog_s3_path):
        return lu.print_and_log( f"{dataset_name} stacked COG already exists in S3: {output_stacked_cog_s3_path}. Skipping creation.",False, logger_worker)

    # Build the stacked VRT locally
    years_sorted = sorted(year_to_cog_s3.keys())    # ensure chronological order
    inputs_vsis3 = [year_to_cog_s3[y].replace("s3://", "/vsis3/") for y in years_sorted]
    cmd_vrt = ["gdalbuildvrt", "-overwrite", "-separate", tmp_stacked_vrt, *inputs_vsis3]
    subprocess.run(cmd_vrt,stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    lu.print_and_log(f"Built stacked local VRT: {tmp_stacked_vrt}", False, logger_worker)

    # TODO can change to gdal python package instead of subprocess
    # Translate the stacked VRT to multiband COG
    cmd_cog = [
        "gdal_translate",
        tmp_stacked_vrt,
        tmp_stacked_cog,
        "-of", "COG",
        "-ot", f"{dt}",
        "-co", "COMPRESS=DEFLATE",
        "-co", "BLOCKSIZE=1024",
        "-co", "PREDICTOR=3",
        "-co", "BIGTIFF=IF_SAFER",
        "-co", f"OVERVIEW_RESAMPLING=AVERAGE",
    ]
    subprocess.run(cmd_cog, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    lu.print_and_log(f"Built stacked local COG: {tmp_stacked_cog}", False, logger_worker)

    # Upload stacked VRT to s3
    uu.upload_s3_file(output_stacked_vrt_s3_path, tmp_stacked_vrt)
    if uu.exists_in_s3(output_stacked_vrt_s3_path):
        lu.print_and_log(f"Uploaded stacked VRT to S3: {output_stacked_vrt_s3_path}", False, logger_worker)
        try:
            os.remove(tmp_stacked_vrt)
        except Exception as e:
            lu.print_and_log(f"Warning: could not delete {tmp_stacked_vrt}: {e}", False, logger_worker)

    # Upload stacked COG to s3
    uu.upload_s3_file(output_stacked_cog_s3_path, tmp_stacked_cog)
    if uu.exists_in_s3(output_stacked_cog_s3_path):
        lu.print_and_log(f"Uploaded stacked COG to S3: {output_stacked_cog_s3_path}", False, logger_worker)
        try:
            os.remove(tmp_stacked_cog)
        except Exception as e:
            lu.print_and_log(f"Warning: could not delete {tmp_stacked_cog}: {e}", False, logger_worker)


def main(cluster_name, process):
    # Connects to Coiled cluster and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, False)
    client

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, f"Global COG creation",
                                                                   run_local, 'standard', f'Global COG creation')

    #TODO: Change to cn paths when switching back to global AFOLU ouput
    wwf_emissions_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_3_WWF_sites/gross_emissions__all_C_pools__all_gases__MgCO2e/standard_model/annual_intervals/YYYY/_pixel_yr/40000_pixels/20251211/"
    wwf_removals_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_3_WWF_sites/gross_removals__all_C_pools__MgCO2/standard_model/annual_intervals/YYYY/_pixel_yr/40000_pixels/20251211/"
    wwf_net_flux_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_3_WWF_sites/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/annual_intervals/YYYY/_pixel_yr/40000_pixels/20251211/"

    wwf_emissions_pattern = "gross_emissions__all_C_pools__all_gases__MgCO2e_pixel_yr"
    wwf_removals_pattern = "gross_removals__all_C_pools__MgCO2_pixel_yr"
    wwf_net_flux_pattern = "net_flux__all_C_pools__all_gases__MgCO2e_pixel_yr"

    # ------------------------------------------------------------------------------------------------------------------

    # Step 1: Create download/ upload dictionary for the datasets we want to create global COGs for.
    download_upload_dictionary = {}

    # Datasets to create global COGs for
    if 'emissions' in process:
        for year in cn.years_annual:
            download_upload_dictionary[f"emissions_{year}"] = {
                'raw_dir': wwf_emissions_path.replace("YYYY", f"{year}"),
                'raw_pattern': wwf_emissions_pattern,
                'vrt': f"/tmp/WWF_{wwf_emissions_pattern}_{year}.vrt",
                'cog': f"/tmp/WWF_{wwf_emissions_pattern}_{year}.tif"
            }
    if 'removals' in process:
        for year in cn.years_annual:
            download_upload_dictionary[f"removals_{year}"] = {
                'raw_dir': wwf_removals_path.replace("YYYY", f"{year}"),
                'raw_pattern': wwf_removals_pattern,
                'vrt': f"/tmp/WWF_{wwf_removals_pattern}_{year}.vrt",
                'cog': f"/tmp/WWF_{wwf_removals_pattern}_{year}.tif"
            }
    if 'net_flux' in process:
        for year in cn.years_annual:
            download_upload_dictionary[f"net_flux_{year}"] = {
                'raw_dir': wwf_net_flux_path.replace("YYYY", f"{year}"),
                'raw_pattern': wwf_net_flux_pattern,
                'vrt': f"/tmp/WWF_{wwf_net_flux_pattern}_{year}.vrt",
                'cog': f"/tmp/WWF_{wwf_net_flux_pattern}_{year}.tif"
            }

    # -------------------------------------------------------------------------------------------------------------------

    # Step 2: Create a VRT per year for each dataset (i.e. 9 VRTs per dataset)
    start_time = time.time()
    main_logger.info(f"STEP 2 - Building VRTs")

    # Create VRTs in parallel for each year x dataset using futures
    vrt_futures = []
    keys_to_remove = []
    for key, items in download_upload_dictionary.items():

        # Add output VRT s3 path to the dictionary
        output_vrt_s3 = f"{items['raw_dir']}{os.path.basename(items['vrt'])}"
        download_upload_dictionary[key]['output_vrt_s3'] = output_vrt_s3
        main_logger.info(f" S3 data folder: {items['raw_dir']}")     # TODO: comment out or delete
        main_logger.info(f" File pattern: {items['raw_pattern']}")   # TODO: comment out or delete
        main_logger.info(f" Local vrt path: {items['vrt']}")         # TODO: comment out or delete
        main_logger.info(f" S3 vrt path: {output_vrt_s3}")           # TODO: comment out or delete

        # Find all files in s3 that match the raw pattern (w/ '*.tif') and add to the dictionary
        input_raster_list_s3 = uu.list_s3_files_with_pattern(items['raw_dir'], items['raw_pattern'])
        if input_raster_list_s3:
            download_upload_dictionary[key]['raw_raster_list'] = input_raster_list_s3
            main_logger.info(f" Input raster list: {items['raw_raster_list']}")  # TODO: comment out or delete
            main_logger.info(f" {key} - there are {len(input_raster_list_s3)} rasters in the raw data folder to include in the vrt")
        else:
            main_logger.warning(f"{key} - there were no rasters found in {items['raw_dir']}. Skipping VRT/COG creation.")
            keys_to_remove.append(key)
            continue

        # Create a VRT from all input rasters
        main_logger.info(f" Submitting VRT build for {key}: {uu.timestr('time')}")
        vrt_future = client.submit(uu.build_vrt_gdal_coiled, input_raster_list_s3, output_vrt_s3, items['vrt'])
        vrt_futures.append((key, vrt_future))

    # Wait for all VRTs to finish before moving on to Step 3
    for key, future in vrt_futures:
        future.result()

    # Remove datasets from the download_upload dictionary that don't have any rasters in their raw_dir
    for key in keys_to_remove:
        download_upload_dictionary.pop(key, None)

    end_time = time.time()
    main_logger.info(f"STEP 2 Complete - All VRTs built in {round(end_time-start_time)} seconds\n")

    #-------------------------------------------------------------------------------------------------------------------

    # Step 3: Get GDAL datatype of each dataset using the first tile in that dataset
    start_time = time.time()
    main_logger.info(f"STEP 3 - Getting GDAL datatypes")

    for key, items in download_upload_dictionary.items():
        main_logger.info(f" Getting GDAL datatype for {key}")

        # Dictionary that matches format expected by function that gets name of first tile in an s3 folder
        simple_dict = {}
        simple_dict[key] = items["raw_dir"]

        # Path of first tile in the dataset
        first_tile = uu.first_file_name_in_s3_folder(simple_dict)

        # Gets datatype of first tile in input dataset and converts it to GDAL format
        download_dict_with_data_types = uu.add_file_type_to_dict(first_tile)
        dtype = download_dict_with_data_types[key][1]
        gdal_dtype = uu.string_to_gdal_dtype_mapping.get(dtype)

        # Adds the dtype of the dataset to the processing dictionary
        download_upload_dictionary[key]["dt"] = gdal_dtype
        main_logger.info(f" Data type for {key} is {uu.gdal_to_string_dtype_mapping.get(gdal_dtype)}")

    end_time = time.time()
    main_logger.info(f"STEP 3 Complete - All GDAL datatypes added to dictionary in {round(end_time-start_time)} seconds\n")

    # -------------------------------------------------------------------------------------------------------------------

    # Step 4: Building global COG for each dataset (per year)
    start_time = time.time()
    main_logger.info(f"STEP 4 - Building global COGs")

    cog_futures = []
    for key, items in download_upload_dictionary.items():

        # Add output COG s3 path to the dictionary
        output_cog_s3 = f"{items['raw_dir']}{os.path.basename(items['cog'])}"
        download_upload_dictionary[key]['output_cog_s3'] = output_cog_s3

        main_logger.info(f" Submitting global COG build for {key}")
        cog_future = client.submit(create_cog_from_vrt, items["output_vrt_s3"], items["cog"], items["dt"], items["output_cog_s3"])
        cog_futures.append((key, cog_future))

    # Wait for all COGs to finish before moving on to Step 5
    for key, future in cog_futures:
        future.result()

    end_time = time.time()
    main_logger.info(f"STEP 4 Complete - All global COGs built in {round(end_time - start_time)} seconds\n")

    # -------------------------------------------------------------------------------------------------------------------

    # Step 5: Combine annual timeseries into a single, stacked global COG
    start_time = time.time()
    main_logger.info("STEP 5 - Creating single, stacked global COG from annual COGs")

    # Limit stacking to datasets with timeseries
    allowed_datasets = {"emissions", "removals", "net_flux"}
    dataset_to_year_cog = {}
    for key, items in download_upload_dictionary.items():
        dataset, year_str = key.rsplit("_", 1)
        if dataset not in allowed_datasets:
            continue
        dataset_to_year_cog.setdefault(dataset, {})
        dataset_to_year_cog[dataset][int(year_str)] = items["output_cog_s3"]

    stack_futures = []
    for dataset, year_map in dataset_to_year_cog.items():
        years_sorted = sorted(year_map.keys())
        min_year, max_year = years_sorted[0], years_sorted[-1]
        stacked_interval = f"{min_year}_{max_year}"

        # Local temp stacked outputs
        dataset_items = download_upload_dictionary[f"{dataset}_{min_year}"]
        dt = dataset_items["dt"]
        tmp_stacked_vrt_path = dataset_items["vrt"].replace(str(min_year), stacked_interval)
        tmp_stacked_cog_path = dataset_items["cog"].replace(str(min_year), stacked_interval)

        # S3 stacked outputs:
        raw_dir_prefix = dataset_items["raw_dir"].split("/annual_intervals", 1)[0].rstrip("/")
        stacked_vrt_filename = os.path.basename(dataset_items["vrt"]).replace(str(min_year), stacked_interval)
        stacked_cog_filename = os.path.basename(dataset_items["cog"]).replace(str(min_year), stacked_interval)
        output_stacked_vrt_s3_path = f"{raw_dir_prefix}/stacked_interval/{stacked_vrt_filename}"
        output_stacked_cog_s3_path = f"{raw_dir_prefix}/stacked_interval/{stacked_cog_filename}"
        #TODO: Add tmp_stacked_vrt_path, etc to dataset_to_year_cog dictionary?

        # Submit stacked VRT + COG build
        main_logger.info(f" Submitting stacked VRT + COG build for {dataset}: {uu.timestr('time')}")
        stack_future = client.submit(stack_annual_cogs, dataset, dt, year_map, tmp_stacked_vrt_path, tmp_stacked_cog_path, output_stacked_vrt_s3_path, output_stacked_cog_s3_path)
        stack_futures.append((dataset, stack_future))

    for dataset, future in stack_futures:
        future.result()

    end_time = time.time()
    main_logger.info(f"STEP 5 Complete - All stacked COGs built in {round(end_time - start_time)} seconds\n")

    # -------------------------------------------------------------------------------------------------------------------

    # Closes the client if not running locally
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create global COGs")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-p', '--processes', action='store', nargs='+', help='What datasets do you want to convert to global COGs?')

    args = parser.parse_args()
    cluster_name = args.cluster_name
    processes = args.processes

    main(cluster_name, processes)