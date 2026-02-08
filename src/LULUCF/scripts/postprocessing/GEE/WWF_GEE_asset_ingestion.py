"""
Script to create WWF GEE assets:
1) uploads data from s3 storage to GCS bucket directly (with option to filter which tiles to upload, passed in as a .txt file)
2) ingest data into GEE as ee asset (each dataset x year is its own ee.Image asset)

run from /mnt/c/GIS/git/AFOLU_GHG_flux_model
To run locally:
python -m src.LULUCF.scripts.postprocessing.GEE.WWF_GEE_asset_ingestion -d net_flux -b lulucf -f WWF -r users/melrose/ -t /mnt/c/GIS/rasters/AFOLU_cogs/operational_landscapes_1x1_tile_ids.txt --skip_existing
python -m src.LULUCF.scripts.postprocessing.GEE.WWF_GEE_asset_ingestion -d net_flux -b lulucf -f WWF -r users/melrose/ -t /mnt/c/GIS/rasters/AFOLU_cogs/operational_landscapes_1x1_tile_ids.txt --skip_existing --clean_gcs --make_public

To run in coiled:
python -m src.utilities.create_cluster -cn GEE_net_flux_2016 -n 10 -m 4 --gcp_project --gcp_credentials_file
python -m src.LULUCF.scripts.postprocessing.GEE.WWF_GEE_asset_ingestion -cn GEE_net_flux_2016 -d net_flux -y 2016 -b lulucf -f WWF -r users/melrose/ -t /mnt/c/GIS/rasters/AFOLU_cogs/operational_landscapes_1x1_tile_ids.txt --skip_existing
*repeated on different clusters for 2016-2024 to run in parallel

python -m src.utilities.create_cluster -cn GEE_mineral_soils_2010 -n 10 -m 4 --gcp_project --gcp_credentials_file
python -m src.LULUCF.scripts.postprocessing.GEE.WWF_GEE_asset_ingestion -cn GEE_mineral_soils_2010 -d mineral_soil -y 2010 -b lulucf -f WWF -r projects/wri-datalab/global_afolu/ -t /mnt/c/GIS/rasters/AFOLU_cogs/operational_landscapes_1x1_tile_ids.txt --skip_existing
*repeated on different clusters for 2010, 2015, 2020, and 2022 to run in parallel

Notes:
    - Make sure to set up your Google Cloud account credentials in your environment and refresh token prior to run.
    - It took ~40 minutes to upload 378 1x1 tiles (tiles that overlap with WWF project sites) for 2010 mineral soils locally
    - It took ~12 minutes to upload 5,175 1x1 tiles (tiles that overlap with WWF operational landscapes) for net flux, all years in Coiled.
      Ran each year in a separate cluster to parallelize all years (~18 credits).

"""
from __future__ import annotations

import argparse
import os
import posixpath
import ee
from google.cloud import storage
import time
import boto3
from dask.distributed import as_completed
import json
from google.oauth2.credentials import Credentials

from src.utilities import constants_and_names as cn
from src.utilities import universal_utilities as uu
from src.utilities import log_utilities as lu

#ee.Authenticate()
gee_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
google_credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
ee_initialized = False

# @lru_cache(maxsize=1)
# def get_gcs_client():
#     if google_credentials:
#         with open(google_credentials, "r", encoding="utf-8") as f:
#             info = json.load(f)
#
#         # IMPORTANT: don't force scopes here; forcing causes invalid_scope
#         creds = Credentials.from_authorized_user_info(info)
#
#         return storage.Client(project=gee_project, credentials=creds)
#
#     # local ADC fallback
#     return storage.Client(project=gee_project)

# Read tile_ids from .txt file
def read_tile_ids(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")}

# List s3 files. Option to match list of tile_ids
def list_s3_tiles_from_tile_ids(s3_path, tile_ids):
    s3 = boto3.client("s3")
    bucket_name, prefix = uu.split_s3_path(s3_path)

    tile_ids = set(tile_ids) if tile_ids else None
    matching_tiles = []
    token = None

    while True:
        kwargs = {"Bucket": bucket_name, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        response = s3.list_objects_v2(**kwargs)

        for object in response.get("Contents", []):
            key = object["Key"]
            if not (key.lower().endswith(".tif") or key.lower().endswith(".tiff")):
                continue
            if tile_ids and not any(tile_id in key for tile_id in tile_ids):
                continue
            matching_tiles.append(key)

        if response.get("IsTruncated"):
            token = response["NextContinuationToken"]
        else:
            break

    return matching_tiles

# Checks to see if tile already exists in GCS
def gcs_blob_exists(gcs_storage_client, bucket, blob_name):
    return gcs_storage_client.bucket(bucket).blob(blob_name).exists(client=gcs_storage_client)

# Upload tile to GCS storage without writing local file
def upload_s3_to_gcs(s3_client, gcs_storage_client, s3_bucket, s3_file, gcs_bucket, gcs_blob):
    obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_file)
    body = obj["Body"]  # streaming file-like object
    blob = gcs_storage_client.bucket(gcs_bucket).blob(gcs_blob)
    blob.upload_from_file(body, rewind=False)

def ensure_gcp_creds_on_workers(client):
    """
    Ensure every worker has a credentials JSON file written to a writable path
    and that GOOGLE_APPLICATION_CREDENTIALS points to it.
    """
    def _write():
        import os, base64

        b64 = os.environ.get("GCP_CREDENTIALS_B64")
        if not b64:
            return {"ok": False, "reason": "missing GCP_CREDENTIALS_B64 on worker"}

        # Prefer the env var path, but fall back to /tmp if it's not writable
        dest = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "/tmp/gcp.json"

        # Try to create the directory; if permission denied (e.g., /opt), fall back to /tmp
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        except PermissionError:
            dest = "/tmp/gcp.json"
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = dest
            os.makedirs(os.path.dirname(dest), exist_ok=True)

        with open(dest, "wb") as f:
            f.write(base64.b64decode(b64))

        return {
            "ok": True,
            "dest": dest,
            "exists": os.path.exists(dest),
            "size": os.path.getsize(dest),
        }

    return client.run(_write)


# One pair of s3 and GCS clients per worker to parallelize tasks using dask
# def worker_clients():
#     return boto3.Session().client("s3"), get_gcs_client()

# Upload all s3 tiles from list -> gs://gcs_bucket/gcs_folder/<basename> (skips existing files)
# Returns count of # of files uploaded vs skipped for log
# def gcs_upload(s3_path, s3_files, gcs_bucket, gcs_folder):
#     s3_client, gcs_storage_client = worker_clients()
#     s3_bucket, _ = uu.split_s3_path(s3_path)
#     uploaded = 0
#     skipped = 0
#
#     for s3_file in s3_files:
#         filename = os.path.basename(s3_file)
#         gcs_blob = "/".join([p for p in [gcs_folder, filename] if p])
#
#         if gcs_blob_exists(gcs_storage_client, gcs_bucket, gcs_blob):
#             skipped += 1
#             continue
#
#         upload_s3_to_gcs(s3_client, gcs_storage_client, s3_bucket, s3_file, gcs_bucket, gcs_blob)
#         uploaded += 1
#
#     return {"uploaded": uploaded, "skipped": skipped, "total": len(s3_files)}
#
def chunked(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]

def make_gcs_client_from_env():
    gee_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    google_credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "/tmp/gcp.json"

    if not os.path.exists(google_credentials):
        raise FileNotFoundError(
            f"Credentials file not found on this machine: {google_credentials}. "
            "If running on Coiled, ensure ensure_gcp_creds_on_workers(client) ran successfully."
        )

    with open(google_credentials, "r", encoding="utf-8") as f:
        info = json.load(f)

    creds = Credentials.from_authorized_user_info(info)
    return storage.Client(project=gee_project, credentials=creds)



def gcs_upload(s3_path, s3_files, gcs_bucket, gcs_folder):
    s3_client = boto3.Session().client("s3")
    gcs_storage_client = make_gcs_client_from_env()

    s3_bucket, _ = uu.split_s3_path(s3_path)
    uploaded = 0
    skipped = 0

    for s3_file in s3_files:
        filename = os.path.basename(s3_file)
        gcs_blob = "/".join([p for p in [gcs_folder, filename] if p])

        if gcs_blob_exists(gcs_storage_client, gcs_bucket, gcs_blob):
            skipped += 1
            continue

        upload_s3_to_gcs(s3_client, gcs_storage_client, s3_bucket, s3_file, gcs_bucket, gcs_blob)
        uploaded += 1

    return {"uploaded": uploaded, "skipped": skipped, "total": len(s3_files)}

# List tiles in GCS bucket
def list_gcs_tifs(gcs_storage_client, bucket, prefix):
    out = []
    for blob in gcs_storage_client.list_blobs(bucket, prefix=prefix):
        name = blob.name
        if name.lower().endswith(".tif") or name.lower().endswith(".tiff"):
            out.append(name)
    return out

# Remove s3 tiles from upload list that already exist in GCS storage
def filter_s3_list_by_existing_gcs(s3_files, gcs_bucket, gcs_folder):
    gcs_storage_client = make_gcs_client_from_env()
    existing = {os.path.basename(tile) for tile in list_gcs_tifs(gcs_storage_client, gcs_bucket, gcs_folder)}
    if not existing:
        return s3_files
    return [file for file in s3_files if os.path.basename(file) not in existing]

# Formats GEE asset path into ee folder structure
def ensure_ee_folder(folder_id):
    parts = folder_id.split("/")
    for i in range(2, len(parts) + 1):
        path = "/".join(parts[:i])
        try:
            ee.data.createAsset({"type": "FOLDER"}, to_cloud_asset_name(path))
        except Exception:
            pass

# Checks if GEE asset already exists
def ee_asset_exists(asset_id):
    try:
        ee.data.getAsset(to_cloud_asset_name(asset_id))
        return True
    except Exception:
        return False

# Kicks of GEE task to ingest all tiles in GCS storage as a singe, mosaiced ee.Image asset
def start_image_ingestion_from_gcs(gcs_uris, asset_id, band_name="b1"):
    # Sources needs a single tile per ImageSource
    sources = [{"uris": [uri]} for uri in gcs_uris]

    # Ingestion manifest for all tiles
    manifest = {
        "name": to_cloud_asset_name(asset_id),
        "tilesets": [{"sources": sources}],
        "bands": [{"id": band_name}],
    }
    return ee.data.startIngestion(None, manifest)

# Makes sure GEE asset name is formatted correctly
def to_cloud_asset_name(asset_id):
    asset_id = asset_id.strip("/")
    if asset_id.startswith("users/") or asset_id.startswith("projects/wri-datalab/"):
        return f"projects/earthengine-legacy/assets/{asset_id}"
    elif asset_id.startswith("projects/"):
        return asset_id
    else:
        raise ValueError(f"Bad asset id: {asset_id}")

def get_asset_id_for_item(item):
    ee_folder = posixpath.join(item["ee_dir"])
    asset_name = f"{item['ee_pattern']}__{item['year']}"
    return posixpath.join(ee_folder, asset_name)

# Main function to submit ee.Image asset ingestion task for each dataset x year
def ingest_dataset_year(item, tile_ids, skip_existing):
    global ee_initialized
    if not ee_initialized:
        ee.Initialize(project=gee_project)
        ee_initialized = True
    gcs_storage_client = make_gcs_client_from_env()

    gcs_bucket = item["gcs_bucket"]
    gcs_folder = item["gcs_folder"]
    year = item["year"]

    blob_names = list_gcs_tifs(gcs_storage_client, gcs_bucket, gcs_folder)

    if tile_ids:
        blob_names = [blob for blob in blob_names if any(tile_id in blob for tile_id in tile_ids)]
    if not blob_names:
        return {"asset_id": None, "task_id": None, "n_tiles": 0}
    blob_names = sorted(blob_names)
    gcs_uris = [f"gs://{gcs_bucket}/{blob}" for blob in blob_names]

    # GEE folder
    ee_folder = posixpath.join(item["ee_dir"])
    ensure_ee_folder(ee_folder)

    # Asset name
    asset_name = f"{item['ee_pattern']}__{year}"
    asset_id = posixpath.join(ee_folder, asset_name)

    if skip_existing and ee_asset_exists(asset_id):
        return {"asset_id": asset_id, "task_id": None, "n_tiles": len(gcs_uris), "skipped": True}

    task_id = start_image_ingestion_from_gcs(gcs_uris=gcs_uris, asset_id=asset_id)
    return {"asset_id": asset_id, "task_id": task_id, "n_tiles": len(gcs_uris), "skipped": False}

# Deletes tile in the gcs_folder (dataset x year).
# If tile_ids is provided, only deletes tiles whose names contain those tile_ids
def delete_gcs_dataset_year(gcs_bucket, gcs_folder, tile_ids):
    gcs_storage_client = make_gcs_client_from_env()
    bucket = gcs_storage_client.bucket(gcs_bucket)

    deleted = 0
    kept = 0
    seen = 0

    for blob in gcs_storage_client.list_blobs(gcs_bucket, prefix=gcs_folder.rstrip("/") + "/"):
        name = blob.name
        if not (name.lower().endswith(".tif") or name.lower().endswith(".tiff")):
            continue
        seen += 1
        if tile_ids and not any(tile_id in name for tile_id in tile_ids):
            kept += 1
            continue
        bucket.blob(name).delete()
        deleted += 1

    return {"seen": seen, "deleted": deleted, "kept": kept}

# Makes an EE asset publicly readable
def make_gee_asset_public(asset_id):
    ee.data.setAssetAcl(to_cloud_asset_name(asset_id), {"all_users_can_read": True})



def main(cluster_name, datasets, gcs_bucket, gcs_folder, gee_repo, years, tile_ids, skip_existing, clean_gcs, make_public):

    # Connects to Coiled cluster if the named cluster exists
    if cluster_name:
        run_local = False
    else:
        run_local = True

    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)
    client

    if not run_local:
        ensure_gcp_creds_on_workers(client)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers= lu.populate_main_log_header(client, cluster, "GEE asset creation", run_local, 'standard', 'GEE asset creation')

    #TODO: Change to cn paths when switching to global AFOLU ouput
    wwf_emissions_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_5__standard__global/gross_emissions__all_C_pools__all_gases__MgCO2e/annual_intervals/YYYY/_ha_yr/4000_pixels/20260130/"
    wwf_removals_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_5__standard__global/gross_removals__all_C_pools__MgCO2/annual_intervals/YYYY/_ha_yr/4000_pixels/20260130/"
    wwf_net_flux_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_5__standard__global/net_flux__all_C_pools__all_gases__MgCO2e/annual_intervals/YYYY/_ha_yr/4000_pixels/20260130/"
    wwf_mineral_soil_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/version_1_0_0__standard__global/SOC_change__mineral_soil_extent__0-30cm_MgC/YYYY/_ha_yr/4000_pixels/20251224/"

    wwf_emissions_gee_folder = "WWF_annual_emissions"
    wwf_removals_gee_folder = "WWF_annual_removals"
    wwf_net_flux_gee_folder = "WWF_annual_net_flux"
    wwf_mineral_soil_gee_folder = "WWF_mineral_soils"

    wwf_emissions_gee_pattern = "emissions__all_C_pools__all_gases__MgCO2e_per_hectare_per_year"
    wwf_removals_gee_pattern = "removals__all_C_pools__MgCO2_per_hectare_per_year"
    wwf_net_flux_gee_pattern = "net_flux__all_C_pools__all_gases__MgCO2e_per_hectare_per_year"
    wwf_mineral_soil_gee_pattern = "SOC_change__mineral_soil_extent__0-30cm_MgC_per_hectare_per_year"

    # ------------------------------------------------------------------------------------------------------------------

    # Step 1: Create download/ upload dictionary for the datasets we want to create GEE assets for.

    gcs_folder = gcs_folder.rstrip("/") if gcs_folder else ""
    tile_ids = read_tile_ids(tile_ids) if tile_ids else None

    # Default to all available years if years are not provided by user
    if years is None:
        years = {
            "emissions": cn.interval_end_years_annual,
            "removals": cn.interval_end_years_annual,
            "net_flux": cn.interval_end_years_annual,
            "mineral_soil": cn.SOC_change_intervals,
        }

    # Datasets to pass in arguments for tile upload + GEE asset creation
    download_upload_dictionary = {}

    for dataset in datasets:
        ds_years = years if isinstance(years, list) else years[dataset]
        if not ds_years:
            raise ValueError(f"No years found for dataset={dataset}. Pass --years or add defaults.")

        for year in ds_years:
            if dataset == "emissions":
                s3_dir = wwf_emissions_path.replace("YYYY", str(year))
                gee_dir = wwf_emissions_gee_folder
                gee_pattern = wwf_emissions_gee_pattern
            elif dataset == "removals":
                s3_dir = wwf_removals_path.replace("YYYY", str(year))
                gee_dir = wwf_removals_gee_folder
                gee_pattern = wwf_removals_gee_pattern
            elif dataset == "net_flux":
                s3_dir = wwf_net_flux_path.replace("YYYY", str(year))
                gee_dir = wwf_net_flux_gee_folder
                gee_pattern = wwf_net_flux_gee_pattern
            elif dataset == "mineral_soil":
                s3_dir = wwf_mineral_soil_path.replace("YYYY", str(year))
                gee_dir = wwf_mineral_soil_gee_folder
                gee_pattern = wwf_mineral_soil_gee_pattern
            else:
                raise ValueError(f"Unknown dataset: {dataset}")

            download_upload_dictionary[f"{dataset}_{year}"] = {
                "dataset": dataset,
                "year": year,
                "s3_dir": s3_dir.rstrip("/") + "/",
                "gcs_bucket": gcs_bucket,
                "gcs_folder": "/".join([p for p in [gcs_folder, dataset, str(year)] if p]),     # GCS layout: <bucket>/<gcs_folder>/<dataset>/<year>/
                "ee_dir": posixpath.join(gee_repo, gcs_folder, gee_dir),
                "ee_pattern": gee_pattern
            }


    # -------------------------------------------------------------------------------------------------------------------

    # Step 2: Upload tiles from s3 storage directly to GCS bucket storage
    start_time = time.time()
    main_logger.info(f"STEP 2 - Uploading data from s3 to GCS bucket")

    # Create list of rasters to upload to GCS (filtered if tile_ids if provided) (existing tiles in GCS skipped from upload)
    keys_to_remove = []
    for key, items in download_upload_dictionary.items():
        s3_raster_list = list_s3_tiles_from_tile_ids(items["s3_dir"], tile_ids)
        if not s3_raster_list:
            main_logger.warning(f"{key} - There were no rasters found in s3. Skipping upload.")
            keys_to_remove.append(key)
            continue
        if skip_existing:
            s3_raster_list = filter_s3_list_by_existing_gcs(s3_raster_list, items["gcs_bucket"], items["gcs_folder"])

        download_upload_dictionary[key]['s3_raster_list'] = s3_raster_list
        main_logger.info(f" {key} - There are {len(s3_raster_list)} rasters in s3 to upload to GCS")

    # Remove datasets from the download_upload dictionary that don't have any rasters in their s3_dir
    for key in keys_to_remove:
        download_upload_dictionary.pop(key, None)


    # Upload data from s3 to GCS using futures (Coiled) or locally
    total_to_upload = sum(len(i["s3_raster_list"]) for i in download_upload_dictionary.values())
    if total_to_upload == 0:
        main_logger.info(" Nothing to upload (all tiles missing or already in GCS).")
    else:
        uploaded_so_far = 0
        batch_size = 25     # edit this (smaller -> more progress updates; larger -> less overhead)
        if not run_local:
            gcs_futures = []
            for key, items in download_upload_dictionary.items():
                for batch in chunked(items["s3_raster_list"], batch_size):
                    gcs_future = client.submit(gcs_upload, items["s3_dir"], batch, items["gcs_bucket"], items["gcs_folder"])
                    gcs_futures.append((key, gcs_future))

            # Wait for all GCS uploads to finish before moving on to asset ingestion
            future_to_key = {future: key for key, future in gcs_futures}
            for future in as_completed(list(future_to_key)):
                result = future.result()
                uploaded_so_far += result["uploaded"]
                percent = (uploaded_so_far / total_to_upload) * 100
                main_logger.info("STEP 2 progress: %.1f%% (%s/%s uploaded)", percent, uploaded_so_far, total_to_upload)

        else:
            for key, items in download_upload_dictionary.items():
                for batch in chunked(items["s3_raster_list"], batch_size):
                    result = gcs_upload(items["s3_dir"], batch, items["gcs_bucket"], items["gcs_folder"])
                    uploaded_so_far += result["uploaded"]
                    percent = (uploaded_so_far / total_to_upload) * 100
                    main_logger.info("STEP 2 progress: %.1f%% (%s/%s uploaded)", percent, uploaded_so_far, total_to_upload)

    end_time = time.time()
    main_logger.info(f"STEP 2 Complete - All data uploaded to GCS storage in {round(end_time - start_time)/60.0} minutes\n")

    # -------------------------------------------------------------------------------------------------------------------

    # Step 3: Ingest all tiles in each dataset×year folder as its own ee.Image asset
    start_time = time.time()
    main_logger.info("STEP 3 - Ingesting dataset × year assets into Earth Engine")

    # Earth Engine uses existing credentials on the machine
    if not gee_project:
        raise ValueError("GOOGLE_CLOUD_PROJECT is not set")
    ee.Initialize(project=gee_project)

    for key, item in download_upload_dictionary.items():
        result = ingest_dataset_year(item, tile_ids, skip_existing)
        if result.get("asset_id") is None:
            main_logger.warning("%s - no tiles found in GCS to ingest", key)
        elif result.get("skipped"):
            main_logger.info("%s - skip existing asset: %s (%s tiles)", key, result["asset_id"], result["n_tiles"])
        else:
            main_logger.info("%s - started ingestion: %s (task=%s, %s tiles)", key, result["asset_id"], result["task_id"], result["n_tiles"])

    end_time = time.time()
    main_logger.info("STEP 3 Complete - all ingestions completed in %s seconds\n", round(end_time - start_time))


    # -------------------------------------------------------------------------------------------------------------------

    # Step 4: Delete tiles in GCS storage after asset has successfully been ingested and QCed
    start_time = time.time()
    main_logger.info("STEP 4 - Optional: Deleting tiles from GCS storage")

    if clean_gcs:
        for key, item in download_upload_dictionary.items():
            try:
                stats = delete_gcs_dataset_year(item["gcs_bucket"], item["gcs_folder"], tile_ids)
                main_logger.info("%s - deleted tiles in GCS storage (seen=%s, deleted=%s, kept=%s)",
                                 key, stats["seen"], stats["deleted"], stats["kept"])
            except Exception as e:
                main_logger.warning("%s - failed to delete tiles in gs://%s/%s: %s",
                                    key, item["gcs_bucket"], item["gcs_folder"], e)

    end_time = time.time()
    main_logger.info("STEP 4 Complete - tiles deleted in %s seconds\n", round(end_time - start_time))

    # -------------------------------------------------------------------------------------------------------------------

    # Step 5: Make asset publicly accessible after asset has successfully been ingested and QCed
    start_time = time.time()
    main_logger.info("STEP 5 - Optional: Make GEE assets publicly accessible")

    if make_public:
        if not gee_project:
            raise ValueError("GOOGLE_CLOUD_PROJECT is not set")
        ee.Initialize(project=gee_project)

        for key, item in download_upload_dictionary.items():
            asset_id = get_asset_id_for_item(item)
            try:
                make_gee_asset_public(asset_id)
                main_logger.info("%s - set public access: %s", key, asset_id)
            except Exception as e:
                main_logger.warning("%s - failed to set public access on %s: %s", key, asset_id, e)

    end_time = time.time()
    main_logger.info("STEP 5 Complete - all assets set to public access in %s seconds\n", round(end_time - start_time))

    # -------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S3 -> GCS upload --> GEE asset ingestion")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-d', '--datasets', required=True, nargs='+', help='What datasets do you want to ingest as GEE assets? Options: emissions, removals, net_flux, mineral_soil')
    parser.add_argument('-b', '--gcs_bucket', required=True, help="GCS bucket name (ex: my-bucket)")
    parser.add_argument('-f', '--gcs_folder',  help="Folder in GCS bucket (ex: wwf)")
    parser.add_argument('-r', '--gee_repo', help="GEE repo to ingest asset to (ex: my-asset-repo)")
    parser.add_argument('-y', '--years', nargs='+', help="Which year(s) to run? Defaults to use all available years if not specified.")
    parser.add_argument('-t', "--tile_ids", help="Optional text file with tile ids to filter to (one per line)")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--clean_gcs", action="store_true")
    parser.add_argument("--make_public", action="store_true")

    args = parser.parse_args()
    cluster_name = args.cluster_name
    datasets = args.datasets
    gcs_bucket = args.gcs_bucket
    gcs_folder = args.gcs_folder
    gee_repo = args.gee_repo
    years = args.years
    tile_ids = args.tile_ids
    skip_existing = args.skip_existing
    clean_gcs = args.clean_gcs
    make_public = args.make_public

    main(cluster_name, datasets, gcs_bucket, gcs_folder, gee_repo, years, tile_ids, skip_existing, clean_gcs, make_public)