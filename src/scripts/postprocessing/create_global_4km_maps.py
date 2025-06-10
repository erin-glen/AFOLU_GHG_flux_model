"""
Aggregate 10x10° raster outputs into global ~4 km maps.
"""

import argparse
import dask
import numpy as np

from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu
from ..utilities import log_utilities as lu
from .LULUCF_fluxes_aggregate_to_10x10deg import get_input_datasets


def agg_4x4(tile_id, bounds, chunk_length_pixels, pixel_area_tile, mg_ha_yr_tile,
            per_pixel_output_tile, per_pixel_output_path):
    """Convert per-hectare tile to per-pixel and aggregate to 0.04°."""
    is_final = False
    logger = lu.setup_logging()

    logger.info(f"Getting rasters for {tile_id}\n{pixel_area_tile}\n{mg_ha_yr_tile}")
    pixel_area_tile_chunk = uu.get_tile_dataset_rio(
        pixel_area_tile, "Float32", bounds, chunk_length_pixels, is_final, logger
    )[0]
    mg_ha_yr_tile_chunk = uu.get_tile_dataset_rio(
        mg_ha_yr_tile, "Float32", bounds, chunk_length_pixels, is_final, logger
    )[0]

    mg_yr_per_pixel_tile_chunk = (
        mg_ha_yr_tile_chunk * pixel_area_tile_chunk * cn.m2_to_ha
    )

    data_type = mg_yr_per_pixel_tile_chunk.dtype.name
    uu.save_and_upload_single_raster(
        bounds,
        chunk_length_pixels,
        tile_id,
        mg_yr_per_pixel_tile_chunk,
        data_type,
        per_pixel_output_tile,
        per_pixel_output_path,
        is_final,
        logger,
    )

    return uu.reaggregate_resolution(mg_yr_per_pixel_tile_chunk, 0.00025, 0.04)


def combine_global_raster(tiles, bounds_list, tile_id, global_4km_outfile, global_4km_output_path):
    """Combine multiple 0.04° tiles into a single global raster."""
    is_final = False
    logger = lu.setup_logging()

    global_shape = (int(180 / 0.04), int(360 / 0.04))
    global_raster = np.zeros(global_shape, dtype=np.float32)

    for tile, bounds in zip(tiles, bounds_list):
        min_x, min_y, max_x, max_y = bounds
        x_start = int((min_x + 180) / 0.04)
        x_end = int((max_x + 180) / 0.04)
        y_start = int((90 - max_y) / 0.04)
        y_end = int((90 - min_y) / 0.04)

        tile_height, tile_width = tile.shape
        assert (y_end - y_start) == tile_height
        assert (x_end - x_start) == tile_width

        global_raster[y_start:y_end, x_start:x_end] += tile

    global_bounds = (-180, -90, 180, 90)
    uu.save_and_upload_single_raster(
        global_bounds,
        global_raster.shape[1],
        tile_id,
        global_raster,
        np.float32,
        global_4km_outfile,
        global_4km_output_path,
        is_final,
        logger,
    )

    return "Success"


def build_download_upload_dict(pixel_resolution: str) -> dict:
    """Create download/upload dictionary from drainage output paths."""
    dictionary = {}
    for path in get_input_datasets(pixel_resolution):
        parts = path.rstrip("/").split("/")
        dataset = parts[8]
        interval = parts[11]
        key = f"{dataset}_{interval}"
        mg_ha_yr_dir = path if path.endswith("/") else f"{path}/"
        mg_ha_yr_pattern = f"__{dataset}_{interval}.tif"
        mg_per_pixel_dir = mg_ha_yr_dir.replace(pixel_resolution, "per_pixel")
        mg_per_pixel_pattern = f"__{dataset}_per_pixel_{interval}.tif"
        out_dir = (
            "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/0_04deg_output_aggregation/"
            f"{dataset}/{interval}/"
        )
        dictionary[key] = {
            "mg_ha_yr_dir": mg_ha_yr_dir,
            "mg_ha_yr_pattern": mg_ha_yr_pattern,
            "mg_per_pixel_dir": mg_per_pixel_dir,
            "mg_per_pixel_pattern": mg_per_pixel_pattern,
            "4km_dir": out_dir,
            "4km_pattern": f"0_04deg_global__{dataset}_{interval}.tif",
        }
    return dictionary


def main(cluster_name: str, pixel_resolution: str):
    run_local = False
    cluster, client = uu.connect_to_cluster(cluster_name, run_local)

    download_upload_dictionary = build_download_upload_dict(pixel_resolution)

    for key, items in download_upload_dictionary.items():
        bounds_list = []
        delayed_results = []
        for tile_id in cn.tile_id_list:
            stage = f"create 0.04x0.04 deg tile rasters for {key}"
            start_time = uu.timestr()
            print(f"Stage {stage} started at: {start_time}")

            mg_ha_yr_tile = f"{items['mg_ha_yr_dir']}{tile_id}{items['mg_ha_yr_pattern']}"
            pixel_area_tile = f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tile_id}.tif"
            per_pixel_tile_outfile = f"{tile_id}{items['mg_per_pixel_pattern']}"
            per_pixel_output_path = items["mg_per_pixel_dir"]

            bounds = uu.get_10x10_tile_bounds(tile_id)
            bounds_list.append(bounds)
            chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)

            delayed_results.append(
                dask.delayed(agg_4x4)(
                    tile_id,
                    bounds,
                    chunk_length_pixels,
                    pixel_area_tile,
                    mg_ha_yr_tile,
                    per_pixel_tile_outfile,
                    per_pixel_output_path,
                )
            )

        stage = f"create 0.04x0.04 degree global raster for {key}"
        start_time = uu.timestr()
        print(f"Stage {stage} started at: {start_time}")

        tiles = dask.compute(*delayed_results)

        tile_id = "0_04deg_global"
        global_4km_outfile = f"{tile_id}{items['mg_per_pixel_pattern']}"
        global_4km_output_path = items["4km_dir"]

        result = combine_global_raster(
            tiles,
            bounds_list,
            tile_id,
            global_4km_outfile,
            global_4km_output_path,
        )
        print(result)

    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate AFOLU model output into global ~4km rasters."
    )
    parser.add_argument("-cn", "--cluster_name", required=True, help="Coiled cluster name")
    parser.add_argument(
        "-p",
        "--pixel_resolution",
        default="40000_pixels",
        help="Input raster resolution",
    )
    args = parser.parse_args()

    main(args.cluster_name, args.pixel_resolution)