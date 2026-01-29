"""
Script to create WWF GEE assets:
1) uploads data from s3 storage to GCS bucket (with option to filter which tiles to upload)
2) ingest data into GEE (each dataset x year is its own ee.Image asset)

run from /mnt/c/GIS/git/AFOLU_GHG_flux_model
python -m src.LULUCF.scripts.postprocessing.GEE.WWF_GEE_asset_ingestion -d mineral_soil -y 2010 -b lulucf -f WWF -r users/melrose -t /mnt/c/GIS/rasters/AFOLU_cogs/tile_ids.txt --skip_existing

"""
from __future__ import annotations

import argparse
import os
import posixpath
import ee
from functools import lru_cache
from google.cloud import storage
import time
import boto3

from src.utilities import constants_and_names as cn
from src.utilities import universal_utilities as uu
from src.utilities import log_utilities as lu

# Export these from your bashrc to use here
gee_project = os.environ.get("GOOGLE_CLOUD_PROJECT")

# def norm_prefix(prefix: str | None) -> str:
#     return (prefix or "").strip("/")
#
#
# def _parse_years(raw: str | None, default_years: list[int]) -> list[int]:
#     if not raw:
#         return default_years
#     years: set[int] = set()
#     for tok in raw.replace(" ", "").split(","):
#         if not tok:
#             continue
#         if "-" in tok:
#             a, b = tok.split("-", 1)
#             years.update(range(int(a), int(b) + 1))
#         else:
#             years.add(int(tok))
#     return sorted(years)
#
#
# Read tile_ids from .txt file
def read_tile_ids(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")}

#
# def _tile_ok(name_or_key: str, tile_ids: set[str] | None) -> bool:
#     if not tile_ids:
#         return True
#     return any(tid in name_or_key for tid in tile_ids)
#
#
# def _is_tif(key: str) -> bool:
#     k = key.lower()
#     return k.endswith(".tif") or k.endswith(".tiff")
#
#
# def _ensure_ee_folder(folder_id: str) -> None:
#     # folder_id like users/melrose/WWF/mineral_soil/2010
#     parts = folder_id.split("/")
#     for i in range(2, len(parts) + 1):
#         path = "/".join(parts[:i])
#         try:
#             ee.data.createAsset({"type": "FOLDER"}, path)
#         except Exception:
#             # Folder likely exists (or parent missing in older EE backends) — ignore.
#             pass
#
#
# def _ee_asset_exists(asset_id: str) -> bool:
#     try:
#         ee.data.getAsset(asset_id)
#         return True
#     except Exception:
#         return False
#
#
# def _start_ingestion_from_gcs(*, gcs_uri: str, asset_id: str) -> str:
#     """
#     Ingest a single GeoTIFF from GCS -> EE asset via EE Python API.
#
#     Assumption: these rasters are single-band; if not, ingestion may need a bands manifest.
#     """
#     manifest = {
#         "name": asset_id,
#         "tilesets": [{"sources": [{"uris": [gcs_uri]}]}],
#         "bands": [{"id": "b1"}],
#     }
#     task_id = ee.data.startIngestion(None, manifest)
#     return task_id
#
#
# def _gs_uri(bucket: str, *parts: str) -> str:
#     clean = [p.strip("/") for p in parts if p and p.strip("/")]
#     if clean:
#         return f"gs://{bucket}/" + "/".join(clean)
#     return f"gs://{bucket}"

# List s3 files with tile_ids
def list_s3_tifs_from_tile_ids(s3_path, tile_ids):
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

def gcs_blob_exists(gcs_storage_client, bucket, blob_name):
    return gcs_storage_client.bucket(bucket).blob(blob_name).exists(client=gcs_storage_client)


# Upload S3 file to GCS storage without writing local file
def upload_s3_to_gcs(s3_client, gcs_storage_client, s3_bucket, s3_file, gcs_bucket, gcs_blob):
    object = s3_client.get_object(Bucket=s3_bucket, Key=s3_file)
    body = object["Body"]  # streaming file-like object
    blob = gcs_storage_client.bucket(gcs_bucket).blob(gcs_blob)
    blob.upload_from_file(body, rewind=False)

@lru_cache(maxsize=1)
# One pair of s3 and GCS clients per worker to parallelize tasks using dask
def worker_clients():
    return boto3.Session().client("s3"), storage.Client()

# Upload all s3 files from list -> gs://gcs_bucket/gcs_folder/<basename> (skips existing files)
# Returns count of # of files uploaded vs skipped for log
def gcs_upload(s3_path, s3_files, gcs_bucket, gcs_folder):
    s3_client, gcs_storage_client = worker_clients()

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

def main(cluster_name, datasets, gcs_bucket, gcs_folder, gee_repo, years, tile_ids, skip_existing):

    # Connects to Coiled cluster and the named cluster exists
    if cluster_name:
        run_local = False
    else:
        run_local = True

    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)
    client

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers= lu.populate_main_log_header(client, cluster, "GEE asset creation", run_local, 'standard', 'GEE asset creation')

    #TODO: Change to cn paths when switching to global AFOLU ouput
    wwf_emissions_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_3_WWF_sites/gross_emissions__all_C_pools__all_gases__MgCO2e/standard_model/annual_intervals/YYYY/_pixel_yr/40000_pixels/20251211/"
    wwf_removals_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_3_WWF_sites/gross_removals__all_C_pools__MgCO2/standard_model/annual_intervals/YYYY/_pixel_yr/40000_pixels/20251211/"
    wwf_net_flux_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_3_WWF_sites/net_flux__all_C_pools__all_gases__MgCO2e/standard_model/annual_intervals/YYYY/_pixel_yr/40000_pixels/20251211/"
    wwf_mineral_soil_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/version_1_0_0__standard__global/SOC_change__mineral_soil_extent__0-30cm_MgC/YYYY/_ha_yr/4000_pixels/20251224/"

    # wwf_emissions_pattern = "gross_emissions__all_C_pools__all_gases__MgCO2e_pixel_yr"
    # wwf_removals_pattern = "gross_removals__all_C_pools__MgCO2_pixel_yr"
    # wwf_net_flux_pattern = "net_flux__all_C_pools__all_gases__MgCO2e_pixel_yr"
    # wwf_mineral_soil_pattern = "SOC_change__mineral_soil_extent__0-30cm_MgC_ha_yr"

    wwf_emissions_gee_folder = "WWF_annual_emissions"
    wwf_removals_gee_folder = "WWF_annual_removals"
    wwf_net_flux_gee_folder = "WWF_annual_net_flux"
    wwf_mineral_soil_gee_folder = "WWF_mineral_soils"

    wwf_emissions_gee_pattern = "emissions__all_C_pools__all_gases__MgCO2e_per_pixel"
    wwf_removals_gee_pattern = "removals__all_C_pools__MgCO2_per_pixel"
    wwf_net_flux_gee_pattern = "net_flux__all_C_pools__all_gases__MgCO2e_per_pixel"
    wwf_mineral_soil_gee_pattern = "SOC_change__mineral_soil_extent__0-30cm_MgC_per_hectare"

    # ------------------------------------------------------------------------------------------------------------------
    # Step 1: Create download/ upload/ ingestion dictionary for the datasets we want to create GEE assets for.

    gcs_folder = gcs_folder.rstrip("/") if gcs_folder else ""
    tile_ids = read_tile_ids(tile_ids) if tile_ids else None

    # Default to all available years if years not provided by users
    if years is None:
        years = {
            "emissions": list(cn.interval_end_years_annual),
            "removals": list(cn.interval_end_years_annual),
            "net_flux": list(cn.interval_end_years_annual),
            "mineral_soil": [2010, 2015, 2020, 2022],
        }

    # Datasets to create GEE assets for
    download_upload_dictionary = {}

    for dataset in datasets:
        ds_years = years if isinstance(years, list) else years[dataset]
        if not ds_years:
            raise ValueError(f"No years found for dataset={dataset}. Pass --years or add defaults.")

        for year in ds_years:
            if dataset == "emissions":
                s3_dir = wwf_emissions_path.replace("YYYY", str(year))
                #s3_pattern = wwf_emissions_pattern
                gee_dir = wwf_emissions_gee_folder
                gee_pattern = wwf_emissions_gee_pattern
            elif dataset == "removals":
                s3_dir = wwf_removals_path.replace("YYYY", str(year))
                #s3_pattern = wwf_removals_pattern
                gee_dir = wwf_removals_gee_folder
                gee_pattern = wwf_removals_gee_pattern
            elif dataset == "net_flux":
                s3_dir = wwf_net_flux_path.replace("YYYY", str(year))
                #s3_pattern = wwf_net_flux_pattern
                gee_dir = wwf_net_flux_gee_folder
                gee_pattern = wwf_net_flux_gee_pattern
            elif dataset == "mineral_soil":
                s3_dir = wwf_mineral_soil_path.replace("YYYY", str(year))
                #s3_pattern = wwf_mineral_soil_pattern
                gee_dir = wwf_mineral_soil_gee_folder
                gee_pattern = wwf_mineral_soil_gee_pattern
            else:
                raise ValueError(f"Unknown dataset: {dataset}")

            download_upload_dictionary[f"{dataset}_{year}"] = {
                "dataset": dataset,
                "year": year,
                "s3_dir": s3_dir.rstrip("/") + "/",
                #"s3_pattern": s3_pattern,
                "gcs_bucket": gcs_bucket,
                "gcs_folder": "/".join([p for p in [gcs_folder, dataset, str(year)] if p]), # GCS layout: <bucket>/<gcs_folder>/<dataset>/<year>/
                "ee_dir": posixpath.join(gee_repo, gcs_folder, gee_dir),
                "ee_pattern": gee_pattern
            }


    # -------------------------------------------------------------------------------------------------------------------

    # Step 2: Upload tiles from s3 storage directly to GCS bucket storage

    # Earth Engine: uses existing credentials on the machine/container
    ee.Initialize(project=gee_project)

    start_time = time.time()
    main_logger.info(f"STEP 2 - Uploading data from s3 to GCS bucket")

    # Create list of rasters to upload to GCS (filtered if tile_ids provided)
    keys_to_remove = []
    for key, items in download_upload_dictionary.items():
        s3_raster_list = list_s3_tifs_from_tile_ids(items["s3_dir"], tile_ids)
        if not s3_raster_list:
            main_logger.warning(f"{key} - There were no rasters found. Skipping upload.")
            keys_to_remove.append(key)
            continue

        download_upload_dictionary[key]['s3_raster_list'] = s3_raster_list
        main_logger.info(f" {key} - There are {len(s3_raster_list)} rasters in s3 to upload to GCS")

    # Remove datasets from the download_upload dictionary that don't have any rasters in their s3_dir
    for key in keys_to_remove:
        download_upload_dictionary.pop(key, None)


    # Upload all data from s3 to GCS using futures (Coiled) or locally
    if not run_local:
        gcs_futures = []
        for key, items in download_upload_dictionary.items():
            main_logger.info(f" Uploading data for {key} to GCS: {uu.timestr('time')}")
            gcs_future = client.submit(
                gcs_upload,
                s3_path=items["s3_dir"],
                s3_files=items["s3_raster_list"],
                gcs_bucket=items["gcs_bucket"],
                gcs_folder=items["gcs_folder"]
            )
            gcs_futures.append((key, gcs_future))

        # Wait for all GCS uploads to finish before moving on to asset ingestion
        for key, future in gcs_futures:
            result = future.result()
            main_logger.info("%s - uploaded=%s skipped=%s total=%s", key, result["uploaded"], result["skipped"], result["total"])

    else:
        for key, items in download_upload_dictionary.items():
            main_logger.info(f" Uploading data for {key} to GCS: {uu.timestr('time')}")
            result = gcs_upload(
                s3_path=items["s3_dir"],
                s3_files=items["s3_raster_list"],
                gcs_bucket=items["gcs_bucket"],
                gcs_folder=items["gcs_folder"]
            )
            main_logger.info("%s - uploaded=%s skipped=%s total=%s", key, result["uploaded"], result["skipped"], result["total"])

    end_time = time.time()
    main_logger.info(f"STEP 2 Complete - All data uploaded to GCS storage in {round(end_time - start_time)} seconds\n")


    # # -----------------------------
    # # Step 3: Ingest all tiles per dataset/year folder -> EE
    # # -----------------------------
    # for _, job in download_upload_dictionary.items():
    #     gcs_bucket_name = job["gcs_bucket"]
    #     gcs_prefix_key = job["gcs_prefix"]
    #     ee_folder = job["ee_folder"]
    #     ds = job["dataset"]
    #     yr = job["year"]
    #
    #     _ensure_ee_folder(ee_folder)
    #     tiles = _list_gcs_tifs(gcs, bucket=gcs_bucket_name, prefix=gcs_prefix_key)
    #     tiles = [u for u in tiles if _tile_ok(u, tile_ids_set)]
    #
    #     if not tiles:
    #         main_logger.warning(f"No tiles found to ingest for {ds} {yr}: gs://{gcs_bucket_name}/{gcs_prefix_key}/")
    #         continue
    #
    #     main_logger.info(f"Ingest {ds} {yr}: {len(tiles)} tiles -> {ee_folder}/")
    #
    #     for gcs_uri in tiles:
    #         name = os.path.splitext(os.path.basename(gcs_uri))[0]
    #         asset_id = posixpath.join(ee_folder, name)
    #
    #         if skip_existing and _ee_asset_exists(asset_id):
    #             continue
    #
    #         task_id = _start_ingestion_from_gcs(gcs_uri=gcs_uri, asset_id=asset_id)
    #         main_logger.info(f"Started ingestion: {asset_id} (task={task_id})")
    #
    # main_logger.info("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S3 -> GCS upload --> GEE asset ingestion")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-d', '--datasets', required=True, nargs='+', help='What datasets do you want to ingest as GEE assets? Options: emissions, removals, net_flux, mineral_soil')
    parser.add_argument('-b', '--gcs_bucket', required=True, help="GCS bucket name (ex: my-bucket)")
    parser.add_argument('-f', '--gcs_folder',  help="Folder in GCS bucket (ex: wwf)")
    parser.add_argument('-r', '--gee_repo', help="GEE repo to ingest asset to (ex: my-asset-repo)")
    parser.add_argument('-y', '--years', nargs='+', type=int, help="Which year(s) to run? Defaults to use all available years if not specified.")
    parser.add_argument('-t', "--tile_ids", help="Optional text file with tile ids to filter to (one per line)")
    parser.add_argument("--skip_existing", action="store_true")

    args = parser.parse_args()
    cluster_name = args.cluster_name
    datasets = args.datasets
    gcs_bucket = args.gcs_bucket
    gcs_folder = args.gcs_folder
    gee_repo = args.gee_repo
    years = args.years
    tile_ids = args.tile_ids
    skip_existing = args.skip_existing

    main(cluster_name, datasets, gcs_bucket, gcs_folder, gee_repo, years, tile_ids, skip_existing)