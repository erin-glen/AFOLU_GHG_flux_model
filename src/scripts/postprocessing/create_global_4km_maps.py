"""Aggregate 10x10° raster outputs into global ~4 km maps.

    The script converts per‑hectare drainage emission outputs to per‑pixel values
    using pixel area rasters and then aggregates them to approximately 4 km
    resolution.  Passing ``--skip_pixel_area`` will bypass the conversion step and
    simply average the per‑hectare values when creating the 4 km products.  When
    pixel area rasters are skipped the output filenames omit the ``per_pixel``
    suffix used in the default workflow.
"""

import argparse
import dask
import numpy as np
import posixpath

from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu
from ..utilities import log_utilities as lu

DATA_TYPES = [
    # "burned_ch4_co2e",
    # "burned_co2"
    # "burned_co_co2e",
    "burned_state",
    # "burned_emission_state",
    "burned_total_co2e",
    # "drained_ch4_ditch_co2e",
    # "drained_ch4_land_co2e",
    # "drained_co2",
    # "drained_co2_offsite",
    # "drained_n2o_co2e",
    "drained_total_co2e",
    "emission_state",
    "soil",
    "state",
]

INTEGER_DATASETS = {
    "state",
    "burned_state",
    "emission_state",
    "burned_emission_state",
    "soil",
}

INVENTORY_PERIODS = [
    # "2000_2005",
    # "2005_2010",
    # "2010_2015",
    # "2015_2020",
    "2020_2023"
]

BASE_URL = (
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/"
    "outputs/version_0_3_9"
)
OUTPUT_DATE = "20250610"


def get_input_datasets(
    pixel_resolution: str,
    data_types: list[str] | None = None,
    inventory_periods: list[str] | None = None,
) -> list[str]:
    """Return list of S3 folders for organic soil outputs."""
    data_types = data_types or DATA_TYPES
    inventory_periods = inventory_periods or INVENTORY_PERIODS

    paths = []
    for period in inventory_periods:
        for dtype in data_types:
            path = (
                f"{BASE_URL}/{dtype}/ogh_standard_model/"
                f"five_year_intervals/{period}/{pixel_resolution}/{OUTPUT_DATE}"
            )
            paths.append(path)
    return paths


def agg_4x4(
    tile_id,
    bounds,
    chunk_length_pixels,
    pixel_area_tile,
    mg_ha_yr_tile,
    per_pixel_output_tile,
    per_pixel_output_path,
    use_pixel_area=True,
):
    """Aggregate a 10×10° tile to 0.04° resolution.

    If ``use_pixel_area`` is ``True`` the function converts per‑hectare values to
    per‑pixel values using the pixel‑area raster before aggregating. When
    ``use_pixel_area`` is ``False`` it simply averages the per‑hectare values to
    the coarser resolution.
    """
    is_final = False
    logger = lu.setup_logging()

    logger.info(
        f"Getting rasters for {tile_id}\n{pixel_area_tile}\n{mg_ha_yr_tile}"
    )

    mg_ha_yr_tile_chunk = uu.get_tile_dataset_rio(
        mg_ha_yr_tile, "Float32", bounds, chunk_length_pixels, is_final, logger
    )[0]

    dataset_name = posixpath.basename(mg_ha_yr_tile).split("__")[1]
    is_integer = dataset_name in INTEGER_DATASETS
    if is_integer:
        mg_ha_yr_tile_chunk = mg_ha_yr_tile_chunk.astype(np.int32)

    if use_pixel_area and not is_integer:
        pixel_area_tile_chunk = uu.get_tile_dataset_rio(
            pixel_area_tile, "Float32", bounds, chunk_length_pixels, is_final, logger
        )[0]

        mg_per_pixel_tile_chunk = (
            mg_ha_yr_tile_chunk * pixel_area_tile_chunk * cn.m2_to_ha
        )

        data_type = mg_per_pixel_tile_chunk.dtype.name
        uu.save_and_upload_single_raster(
            bounds,
            chunk_length_pixels,
            tile_id,
            mg_per_pixel_tile_chunk,
            data_type,
            per_pixel_output_tile,
            per_pixel_output_path,
            is_final,
            logger,
        )

        return uu.reaggregate_resolution(
            mg_per_pixel_tile_chunk, 0.00025, 0.04
        )
    if is_integer:
        return uu.reaggregate_mode(mg_ha_yr_tile_chunk, 0.00025, 0.04)

    # When pixel area is not used, aggregate the per-hectare array by averaging
    summed = uu.reaggregate_resolution(mg_ha_yr_tile_chunk, 0.00025, 0.04)
    factor = int(round(0.04 / 0.00025))
    return summed / float(factor * factor)


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
        key = f"{dataset}__{interval}"
        mg_ha_yr_dir = path if path.endswith("/") else f"{path}/"
        mg_ha_yr_pattern = f"__{dataset}__{interval}.tif"
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


def main(
    cluster_name: str,
    pixel_resolution: str,
    run_local: bool = False,
    use_pixel_area: bool = True,
):
    logger = lu.setup_logging_main()
    is_final = not run_local

    cluster, client = uu.connect_to_cluster(cluster_name, run_local)

    download_upload_dictionary = build_download_upload_dict(pixel_resolution)

    for key, items in download_upload_dictionary.items():
        bounds_list = []
        delayed_results = []
        for tile_id in cn.tile_id_list:
            stage = f"create 0.04x0.04 deg tile rasters for {key}"
            start_time = uu.timestr()
            lu.print_and_log(
                f"Stage {stage} started at: {start_time}", is_final, logger
            )

            mg_ha_yr_tile = f"{items['mg_ha_yr_dir']}{tile_id}{items['mg_ha_yr_pattern']}"
            pixel_area_tile = None
            if use_pixel_area:
                pixel_area_tile = (
                    f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tile_id}.tif"
                )
            lu.print_and_log(
                f"Processing {tile_id}:\nmg_ha_yr_tile: {mg_ha_yr_tile}\n"
                f"pixel_area_tile: {pixel_area_tile}",
                is_final,
                logger,
            )
            if use_pixel_area:
                per_pixel_tile_outfile = f"{tile_id}{items['mg_per_pixel_pattern']}"
                per_pixel_output_path = items["mg_per_pixel_dir"]
            else:
                per_pixel_tile_outfile = None
                per_pixel_output_path = None

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
                    use_pixel_area,
                )
            )

        stage = f"create 0.04x0.04 degree global raster for {key}"
        start_time = uu.timestr()
        lu.print_and_log(
            f"Stage {stage} started at: {start_time}", is_final, logger
        )

        tiles = dask.compute(*delayed_results)

        tile_id = "0_04deg_global"
        if use_pixel_area:
            global_4km_outfile = f"{tile_id}{items['mg_per_pixel_pattern']}"
        else:
            global_4km_outfile = items["4km_pattern"]
        global_4km_output_path = items["4km_dir"]

        result = combine_global_raster(
            tiles,
            bounds_list,
            tile_id,
            global_4km_outfile,
            global_4km_output_path,
        )
        lu.print_and_log(
            f"Global raster saved to {global_4km_output_path}{global_4km_outfile}",
            is_final,
            logger,
        )

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
    parser.add_argument(
        "--run_local",
        action="store_true",
        help="Run locally without Dask/Coiled",
    )
    parser.add_argument(
        "--skip_pixel_area",
        action="store_true",
        help="Do not use pixel area rasters (output values remain per hectare)",
    )
    args = parser.parse_args()

    main(
        args.cluster_name,
        args.pixel_resolution,
        args.run_local,
        not args.skip_pixel_area,
    )