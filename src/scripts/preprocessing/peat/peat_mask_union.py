#!/usr/bin/env python
"""
pp_union_peatmask.py

1) Checks if the 30m (~0.00025°) union mask already exists on S3. If so,
   skip re-union. Otherwise, create the union from gfw/gpd/peatmap/peatml/ogh tiles.

2) Optionally (--resample 1km) resample the union to 1 km, either from
   the newly created 30m union or from the existing one if it was found.

Usage:
  # Just do union at 30m if missing:
  python -m src.scripts.preprocessing.pp_union_peatmask --dataset_list gfw gpd peatmap peatml ogh

  # Single tile, also do 1km resample:
  python -m src.scripts.preprocessing.pp_union_peatmask --tile_id 20N_020W --resample 1km

  # Local or Coiled:
  python -m src.scripts.preprocessing.pp_union_peatmask --client local
"""

import os
import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling
from dask import delayed
from dask.distributed import Client, LocalCluster
import dask

import src.scripts.preprocessing.preprocessing_constants as cn
from src.scripts.utilities import universal_utilities as uu

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("peat-union")

BUCKET = cn.s3_bucket_name

# Sample tile for 1 km alignment
SAMPLE_1KM_TILE = f"/vsis3/{BUCKET}/{cn.peat_tiles_prefix_1km}00N_110E_peat_mask_processed.tif"


def get_tile_path(ds_key, tile_id):
    ds = cn.datasets["peat"][ds_key]
    s3_base = ds["s3_processed"]
    tile_name = f"{tile_id}_{ds_key}_mask.tif"
    tile_path = os.path.join(s3_base, tile_name).replace("\\", "/")
    if not tile_path.startswith("s3://"):
        tile_path = f"s3://{BUCKET}/{tile_path.lstrip('/')}"
    return tile_path

def get_union_output_path(tile_id, resolution="30m"):
    if resolution == "1km":
        union_dir = cn.datasets["peat"]["union_mask"]["1km"]
        out_name = f"{tile_id}_union_mask_1km.tif"
    else:
        union_dir = cn.datasets["peat"]["union_mask"]["30m"]
        out_name = f"{tile_id}_union_mask.tif"

    out_path = os.path.join(union_dir, out_name).replace("\\", "/")
    if not out_path.startswith("s3://"):
        out_path = f"s3://{BUCKET}/{out_path.lstrip('/')}"
    return out_path


@dask.delayed
def union_tile(tile_id, ds_list, run_mode="default", do_resample=False):
    """
    1) Check if 30 m union already exists on S3. If so, skip union step.
    2) If union not found, read each dataset tile and create union (0/1).
    3) Optionally resample union to 1 km, either from newly created local file or
       by downloading the existing 30 m union from S3 if it was found.
    """
    log.info(f"[union|{tile_id}] Checking 30m union presence, do_resample={do_resample}")
    out_30m_path = get_union_output_path(tile_id, "30m")
    s3_30m_key = out_30m_path.replace(f"s3://{BUCKET}/", "", 1)

    local_temp = Path(tempfile.gettempdir()) / "union_peat"
    local_temp.mkdir(parents=True, exist_ok=True)
    local_30m = local_temp / f"{tile_id}_union_30m.tif"

    # Check if union tile already on S3
    union_exists = uu.s3_file_exists(BUCKET, s3_30m_key)

    if union_exists and run_mode != "test":
        log.info(f"[union|{tile_id}] 30m union already exists => skipping union step.")
        # We'll download the existing union file if we need for resampling
        if do_resample:
            vsis3_path = out_30m_path.replace("s3://", "/vsis3/")
            # Download or open directly. We'll just open directly for warp
            log.info(f"[union|{tile_id}] will open existing 30m union from {vsis3_path}")
        else:
            # If no resample => done
            return f"[union|{tile_id}] union tile found => skip."
    else:
        # union tile doesn't exist => create it
        arrays = []
        profile = None
        for ds_key in ds_list:
            tile_path = get_tile_path(ds_key, tile_id)
            s3_key_for_check = tile_path.replace(f"s3://{BUCKET}/", "", 1)
            if not uu.s3_file_exists(BUCKET, s3_key_for_check):
                log.warning(f"[union|{tile_id}] MISSING tile for {ds_key}: {tile_path}")
                continue

            vsis3_path = tile_path.replace("s3://", "/vsis3/")
            try:
                with rasterio.open(vsis3_path) as src:
                    arr = src.read(1)
                    arrays.append(arr > 0)
                    if profile is None:
                        profile = src.profile
            except Exception as e:
                log.warning(f"[union|{tile_id}] Error reading {ds_key}: {tile_path}, {e}")
                continue

        if not arrays:
            log.info(f"[union|{tile_id}] All datasets missing => skipping union.")
            return f"[union|{tile_id}] no data from any dataset"

        union_bool = np.any(np.stack(arrays, axis=0), axis=0)
        union_uint8 = union_bool.astype("uint8")

        out_profile = profile.copy()
        out_profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            compress="DEFLATE",
            tiled=True,
            nodata=0
        )

        with rasterio.open(local_30m, "w", **out_profile) as dst:
            dst.write(union_uint8, 1)

        if run_mode != "test":
            uu.upload_file_to_s3(str(local_30m), BUCKET, s3_30m_key)
            log.info(f"[union|{tile_id}] 30m union created and uploaded => {out_30m_path}")

    # 2) Resample to 1 km if needed
    if do_resample:
        # If we didn't create local_30m (because it existed), we open from S3
        if union_exists and run_mode != "test":
            # open directly from /vsis3/ for warp
            local_or_vsis3_30m = out_30m_path.replace("s3://", "/vsis3/")
        else:
            local_or_vsis3_30m = str(local_30m)

        local_1km = local_temp / f"{tile_id}_union_1km.tif"
        resample_union_to_1km(
            input_path=local_or_vsis3_30m,
            sample_1km_tile=SAMPLE_1KM_TILE,
            output_path=str(local_1km)
        )

        out_1km_path = get_union_output_path(tile_id, "1km")
        s3_1km_key = out_1km_path.replace(f"s3://{BUCKET}/", "", 1)

        if run_mode != "test":
            uu.upload_file_to_s3(str(local_1km), BUCKET, s3_1km_key)
            log.info(f"[union|{tile_id}] 1 km union uploaded => {out_1km_path}")
            local_1km.unlink()

    # remove local 30m if we have it
    if (not union_exists or run_mode=="test") and local_30m.exists():
        local_30m.unlink()

    return f"[union|{tile_id}] done"


def resample_union_to_1km(input_path, sample_1km_tile, output_path):
    """
    Resample from ~30 m (0.00025°) to 1 km using nearest neighbor,
    using the sample_1km_tile's alignment.
    """
    with rasterio.open(sample_1km_tile) as ref:
        target_crs = ref.crs
        target_transform = ref.transform
        target_width = ref.width
        target_height = ref.height

    with rasterio.open(input_path) as src:
        data_30m = src.read(1)
        src_profile = src.profile
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
            dst_width=target_width,
            dst_height=target_height
        )
        kwargs = src_profile.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "driver": "GTiff",
            "dtype": "uint8",
            "compress": "DEFLATE",
            "nodata": 0,
            "tiled": True
        })

        data_1km = np.zeros((height, width), dtype="uint8")
        from rasterio.warp import reproject, Resampling
        reproject(
            source=data_30m,
            destination=data_1km,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )

    with rasterio.open(output_path, "w", **kwargs) as dst:
        dst.write(data_1km, 1)


def build_tasks(tile_ids, ds_list, run_mode="default", do_resample=False):
    tasks = []
    for tid in tile_ids:
        tasks.append(union_tile(tid, ds_list, run_mode, do_resample))
    return tasks

def main(tile_id=None, dataset_list=None, client="coiled", run_mode="default", resample=None):
    """
    If 30m union exists, skip re-union. If --resample=1km, do 1km step from
    existing or newly created 30m. 'none' => no resample.
    """
    ds_list = dataset_list or ["gfw", "gpd", "peatmap", "peatml", "ogh"]

    if client == "local":
        cluster = LocalCluster(processes=False, dashboard_address=None)
        client_obj = Client(cluster)
        log.info("Running locally.")
    else:
        cluster, client_obj = uu.connect_to_cluster(
            cluster_name="peat_union",
            n_workers=20,
            region="us-east-1",
        )
        log.info(f"Running on Coiled: {cluster.name}")

    tile_ids = [tile_id] if tile_id else cn.tile_id_list
    do_resample_1km = (resample == "1km")

    log.info(f"[union] Datasets: {ds_list}, Tiles: {len(tile_ids)}, do_resample_1km={do_resample_1km}")
    tasks = build_tasks(tile_ids, ds_list, run_mode, do_resample_1km)
    dask.compute(*tasks)

    client_obj.close()
    if client == "coiled":
        cluster.close()
    log.info("All union tasks completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Union peat masks at ~30m, optionally resample to 1km, skipping re-union if 30m tile found."
    )
    parser.add_argument("--tile_id", help="Single tile ID (optional)")
    parser.add_argument(
        "--dataset_list", nargs="+", default=None,
        help="Datasets to union. Default: gfw gpd peatmap peatml ogh"
    )
    parser.add_argument(
        "--client", default="coiled", choices=["local","coiled"],
        help="Run environment (coiled or local)."
    )
    parser.add_argument(
        "--run_mode", default="default", choices=["default","test"],
        help="default => do S3 upload, test => skip."
    )
    parser.add_argument(
        "--resample", choices=["none", "1km"], default="none",
        help="If '1km', also resample the union mask to 1km. If 30m union exists, skip union logic."
    )
    args = parser.parse_args()

    main(
        tile_id=args.tile_id,
        dataset_list=args.dataset_list,
        client=args.client,
        run_mode=args.run_mode,
        resample=args.resample
    )