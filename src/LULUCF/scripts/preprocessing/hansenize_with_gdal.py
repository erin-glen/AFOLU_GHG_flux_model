import dask
from dask.distributed import Client, LocalCluster
import coiled
import os
from osgeo import gdal
import tempfile
from pathlib import Path
from dask.distributed import print
from src.LULUCF.scripts.utilities import constants_and_names as cn
from src.LULUCF.scripts.utilities import universal_utilities as uu

############################################################################################################
# Connects to local or Coiled cluster
cluster_type = 'test'
cluster_name = 'hansenize_gdal_test'
no_upload = False  #TODO Change to False when running final version
save_local_file = True #TODO Change to False when running final version
process = 'drivers' #TODO add text input file or command line arguments to determine which inputs to preprocess



if cluster_type == 'full':
    # Full cluster with 40 workers
    coiled_cluster = coiled.Cluster(
        n_workers=40,
        use_best_zone=True,
        compute_purchase_option="spot_with_fallback",
        idle_timeout="10 minutes",
        region="us-east-1",
        name=cluster_name,
        workspace='wri-forest-research',
        worker_cpu=4,
        worker_memory="16GiB"
    )
    client = coiled_cluster.get_client()

elif cluster_type == 'test':
    # Test cluster with 1 worker
    coiled_cluster = coiled.Cluster(
        n_workers=1,
        use_best_zone=True,
        compute_purchase_option="spot_with_fallback",
        idle_timeout="10 minutes",
        region="us-east-1",
        name=cluster_name,
        workspace='wri-forest-research',
        worker_cpu=2,
        worker_memory="8GiB"
    )
    client = coiled_cluster.get_client()
elif cluster_type == 'local':
    # Local cluster with multiple workers
    local_cluster = LocalCluster()
    client = Client(local_cluster)
    # Took 32.5 minutes to process drivers data locally (uint8)
else:
    print("set cluster_type to one of the following: 'full', 'test', 'local'")

client

###########################################################################################################
#Step 1: Create download dictionary
download_upload_dictionary ={}
if process == 'drivers':
    download_upload_dictionary["drivers"] = {
        'raw_dir': cn.drivers_raw_dir,
        'raw_pattern': cn.drivers_pattern,
        'vrt': "drivers.vrt",
        #'processed_dir': cn.drivers_processed_dir,
        #TODO: Switch back processed dir
        'processed_dir': "s3://gfw2-data/drivers_of_loss/1_km/processed/coiled_test/",
        'processed_pattern': cn.drivers_pattern
    }

if process == 'secondary_natural_forest':
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

#Step 2: Create a VRT for each dataset that needs to be hansenized
vrt_futures = []

for key,items in download_upload_dictionary.items():
    path = items["raw_dir"]
    pattern = items["raw_pattern"]
    vrt = items["vrt"]
    output_vrt_s3 = f"{path}{vrt}"

    # Find all files in s3 that match the raw pattern and add raw file s3 paths to download_upload_dictionary
    input_raster_list_s3  = uu.list_s3_files_with_pattern(path, pattern)
    if input_raster_list_s3:
        download_upload_dictionary[key]["raw_raster_list"] = input_raster_list_s3

    #Create a vrt of all raw input rasters
    print(f"Attempting to build vrt for {key}:")
    if cluster_type == 'full' or cluster_type == 'test':
        # If running in coiled, download all raw input files and build vrt in cluster
        future = client.submit(uu.build_vrt_gdal_coiled, input_raster_list_s3, output_vrt_s3, vrt, no_upload, save_local_file)
    elif cluster_type == 'local':
        # If running locally, save vrt directly to s3 using vsis3 (does not work in coiled)
        future = client.submit(uu.build_vrt_gdal_local, input_raster_list_s3, output_vrt_s3)
    vrt_futures.append(future)

# Collect the results once they are finished
vrt_results = client.gather(vrt_futures)


#Step 3: Get GDAL datatype of each VRT
for key,items in download_upload_dictionary.items():
    path = items["raw_dir"]
    vrt = items["vrt"]
    output_vrt = f"{path}{vrt}"  #s3 path to upload vrt file

    # Get raster data type from vrt
    if cluster_type == 'full' or cluster_type == 'test':
        dt = uu.get_dtype_from_raster(vrt)  #If running in coiled, uses local vrt
    elif cluster_type == 'local':
        dt = uu.get_dtype_from_s3(output_vrt)   #If running locally, uses vrt in s3

    # Add GDAL data type to download_upload dictionary
    if dt:
        gdal_dt = next(key for key, value in uu.gdal_dtype_mapping.items() if value == dt)  # Get GDAL data type
        download_upload_dictionary[key]["dt"] = gdal_dt
        print(f"vrt for {key} has data type: {dt} ({gdal_dt})")


#Step 4: Use warp_to_hansen to preprocess each dataset into 10x10 degree tiles
#TODO see LULUCF model (take a bounding box as a command line argument, and make chunks)
for tile_id in cn.tile_id_list:
    tile_futures = []
    for key,items in download_upload_dictionary.items():
        filename = f"{tile_id}_{items['processed_pattern']}"
        output_tile_s3 = f"{items['processed_dir']}{filename}"
        xmin, ymin, xmax, ymax = uu.get_10x10_tile_bounds(tile_id)
        dt = items['dt']

        if cluster_type == 'full' or cluster_type == 'test':
            vrt = items["vrt"]
            tile_future = client.submit(uu.warp_to_hansen_coiled, vrt, filename, output_tile_s3,  xmin, ymin, xmax, ymax, dt, 0, True, 400, 400)
        elif cluster_type == 'local':
            input_vrt_s3 = f"{items['raw_dir']}{items['vrt']}"
            tile_future = client.submit(uu.warp_to_hansen_local, input_vrt_s3, output_tile_s3, xmin, ymin, xmax, ymax, dt, 0, True, 400, 400)
        tile_futures.append(tile_future)

    # Collect the results once they are finished
    tile_results = client.gather(tile_futures)

#Step 5: Delete local files
# Remove vrt after tile creation step
#os.remove(str(Path(source_raster_path))) #vrt
#remove local files in raster list
#TODO: Add upload_to_s3/ no_upload_to_s3 and delete_local_copy_after_upload options

#TODO: Change for loop/ tile_futures /tile_results structure so parallelizes more tasks with Coiled?
#Note: Get this warning each time for the first tile to finish so there is always 1 tile missing:
#ERROR 1: DoSinglePartPUT of /vsis3/gfw2-data/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/20241004/rate_6_10/00N_010E_natural_forest_mean_growth_rate__Mg_AGC_ha_yr__6_10_years.tif failed
#ERROR 3: /vsis3/gfw2-data/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/20241004/rate_6_10/00N_010E_natural_forest_mean_growth_rate__Mg_AGC_ha_yr__6_10_years.tif: I/O error




#dask delayed methods
# vrt_task = uu.build_vrt_gdal(raster_list, output_vrt)
# dask.compute(vrt_task)

# tasks = []
# for tile_id in cn.tile_id_list:
#     for key,items in download_upload_dictionary.items():
#         output_vrt = f"{items['raw_dir']}{items['vrt']}"
#         output_tile = f"{items['processed_dir']}{tile_id}_{items['processed_pattern']}"
#         xmin, ymin, xmax, ymax = uu.get_10x10_tile_bounds(tile_id)
#         dt = items['dt']
#         task = dask.delayed(uu.warp_to_hansen)(output_vrt, output_tile, xmin, ymin, xmax, ymax, dt, 0, False)
#         tasks.append(task)
#         print(f"Submitting dask delayed task to hansenize {output_tile}")
# results = dask.compute(tasks)