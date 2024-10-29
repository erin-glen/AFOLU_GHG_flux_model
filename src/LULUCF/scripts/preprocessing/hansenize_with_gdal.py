import dask
from dask.distributed import Client, LocalCluster
import coiled
import os
from osgeo import gdal
from dask.distributed import print
from src.LULUCF.scripts.utilities import constants_and_names as cn
from src.LULUCF.scripts.utilities import universal_utilities as uu

############################################################################################################
# Connects to local or Coiled cluster
cluster_type = 'local'
cluster_name = 'hansenize_gdal_test'
no_upload = True  #TODO Change to False when running final version
remove_local_file = False #TODO Change to True when running final version

#Set the environment variable to enable random writes for S3
os.environ['CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE'] = 'YES'

#Set process
process = 'drivers'
#TODO add text input file or command line arguments to determine which inputs to preprocess


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

    # Find all files that match the raw pattern
    raster_list  = uu.list_s3_files_with_pattern(path, pattern)
    if raster_list:
        download_upload_dictionary[key]["raw_raster_list"] = raster_list

    #Create a vrt of all raw input rasters
    output_vrt = f"{path}{vrt}"
    future = client.submit(uu.build_vrt_gdal, raster_list, vrt, output_vrt, no_upload, remove_local_file)
    vrt_futures.append(future)

# Collect the results once they are finished
vrt_results = client.gather(vrt_futures)


#Step 3: Get GDAL datatype of each VRT
for key,items in download_upload_dictionary.items():
   #path = items["raw_dir"]
    vrt = items["vrt"]

    # Add datatype to download_upload dictionary
    #output_vrt = f"{path}{vrt}"
    dt = uu.get_dtype_from_local(vrt)
    if dt:
        gdal_dt = next(key for key, value in uu.gdal_dtype_mapping.items() if value == dt)  # Get GDAL data type
        download_upload_dictionary[key]["dt"] = gdal_dt
        print(f"vrt for {key} has data type: {dt} ({gdal_dt})")


#Step 3: Use warp_to_hansen to preprocess each dataset into 10x10 degree tiles
#TODO see LULUCF model (take a bounding box as a command line argument, and make chunks)
for tile_id in cn.tile_id_list:
    tile_futures = []
    for key,items in download_upload_dictionary.items():
        vrt = items["vrt"]  #Saved locally with this file name so no need to download from s3
        out_pattern = items['processed_pattern']
        out_dir = items['processed_dir']

        filename = f"{tile_id}_{out_pattern}"
        output_tile_s3 = f"{out_dir}{filename}"
        xmin, ymin, xmax, ymax = uu.get_10x10_tile_bounds(tile_id)
        dt = items['dt']
        tile_future = client.submit(uu.warp_to_hansen, vrt, filename, output_tile_s3, xmin, ymin, xmax, ymax, dt, 0, True, 400, 400)
        tile_futures.append(tile_future)

    # Collect the results once they are finished
    tile_results = client.gather(tile_futures)
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