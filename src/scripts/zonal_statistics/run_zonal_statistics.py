# -*- coding: utf-8 -*-
"""Run LULUCF zonal statistics.

This script converts the workflow developed in
`LULUCF_output_zonal_stats_20250613__basic_working_zonal_stats.ipynb`
into a command line utility.  It can run locally or connect to a Coiled
Dask cluster for distributed execution.
"""

import argparse
import re
from io import BytesIO

import fsspec
import numpy as np
import pandas as pd
import requests
import s3fs
import xarray as xr
from flox import xarray_reduce, ReindexArrayType, ReindexStrategy
from dask.distributed import Client, LocalCluster
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities.universal_utilities import timestr
from src.scripts.utilities.lulucf_constants_and_names import C_to_CO2


def create_state_node_df(state_node_lookup_table_local: str, state_node_lookup_table_s3: str, sheet_name: str) -> pd.DataFrame:
    """Load the state node lookup table from S3 falling back to a local file."""
    try:
        response = requests.get(state_node_lookup_table_s3, timeout=10)
        response.raise_for_status()
        state_node_df = pd.read_excel(BytesIO(response.content), sheet_name=sheet_name)
    except (requests.exceptions.RequestException, Exception) as e:  # pragma: no cover - network access not available
        print(f"Failed to download file from S3. Falling back to local file. Error: {e}")
        state_node_df = pd.read_excel(state_node_lookup_table_local, sheet_name=sheet_name)
    return state_node_df


def list_folder_uris(base_uri: str) -> pd.Series:
    """List GeoTIFF files in an s3 folder and return them as a Series."""
    fs = s3fs.S3FileSystem(anon=False)
    all_files = fs.ls(base_uri)
    tif_files = [f"s3://{f}" for f in all_files if f.endswith(".tif")]
    return pd.Series(tif_files)


def parse_pattern_from_uri(uri_series: pd.Series) -> str | None:
    """Extract the file name pattern from a URI."""
    uri = uri_series.values.tolist()[0]
    pattern = r"__([a-zA-Z0-9_]+(?:__?[a-zA-Z0-9_]+)*)_pixel_yr_\d{4}_\d{4}\.tif$"
    match = re.search(pattern, uri)
    return match.group(1) if match else None


def make_xarray_chunks(tile_uris: pd.Series, chunk_size: int) -> xr.Dataset:
    """Open multiple GeoTIFFs into a chunked Xarray dataset."""
    return (
        xr.open_mfdataset(tile_uris.values.tolist(), parallel=True, chunks={"x": chunk_size, "y": chunk_size}).squeeze()
    )


def safe_crop(ds: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Crop one dataset to another's extent."""
    return ds.sel(x=ref.x, y=ref.y, method="nearest")


def convert_to_coord_dict(flux_results: xr.Dataset, interval: str) -> dict:
    """Convert flox results to a coordinate dictionary."""
    print(f"   Postprocessing {interval}: {timestr()}")
    sparse_data = flux_results.data
    dim_names = flux_results.dims
    indices = sparse_data.coords
    values = sparse_data.data
    coord_dict = {dim: flux_results.coords[dim].values[indices[i]] for i, dim in enumerate(dim_names)}
    coord_dict["value"] = values
    return coord_dict


def classify_node(state_node: int) -> str:
    """Classify a state node code into a broader category."""
    node_str = str(state_node)
    first_digit = int(node_str[0])
    one_digit_map = {1: "forest_gain", 2: "forest_loss", 4: "cropland", 5: "grassland"}
    three_digit_map = {311: "forest_loss", 312: "forest_loss", 321: "disturbed_forest", 322: "stable_forest"}
    if first_digit == 3:
        prefix = int(node_str[:3])
        return three_digit_map.get(prefix, "unknown_3x")
    return one_digit_map.get(first_digit, "unknown")


def create_interval_df(
    coord_dict: dict,
    state_node_df: pd.DataFrame,
    flux_type_dict: dict,
    interval_end_year: int,
) -> pd.DataFrame:
    """Convert flox output to a processed dataframe."""
    df = pd.DataFrame(coord_dict)
    df["flux_type"] = df["flux_type"].replace(flux_type_dict)
    df["node_grp"] = df["state_nodes"].apply(classify_node)
    df["interval_end"] = interval_end_year
    df = df.merge(state_node_df[["state_nodes", "meaning"]], on="state_nodes", how="left")
    df.loc[df["flux_type"].eq("area__ha"), "value"] = df["value"] / 10000
    return df


def calculate_interval_flux_densities(df: pd.DataFrame, contextual_layer_names: list[str]) -> pd.DataFrame:
    """Calculate per-hectare flux densities."""
    area_df = df[df["flux_type"] == "area__ha"].copy()
    flux_df = df[df["flux_type"] != "area__ha"].copy()
    merged = pd.merge(
        flux_df,
        area_df[contextual_layer_names + ["interval_end", "value"]],
        on=contextual_layer_names,
        how="left",
        suffixes=("", "_area"),
    )
    merged["value_per_ha"] = merged["value"] / merged["value_area"] / C_to_CO2
    new_rows = merged.copy()
    new_rows["flux_type"] = new_rows["flux_type"] + "__C_per_ha"
    new_rows["value"] = new_rows["value_per_ha"]
    new_rows = new_rows.drop(columns=["value_area", "interval_end_area", "value_per_ha"])
    return pd.concat([df, new_rows], ignore_index=True)


def ensure_zarr_exists(uri_list: pd.Series, zarr_path: str, chunk_size: int) -> None:
    """Create a zarr from URIs if it does not already exist."""
    fs, path = fsspec.core.url_to_fs(zarr_path)
    if fs.exists(path):
        return
    ds = make_xarray_chunks(uri_list, chunk_size)
    ds.to_zarr(zarr_path, mode="w")


def main(args: argparse.Namespace) -> None:
    cluster, client, run_local = uu.connect_to_cluster(
        cluster_name=args.cluster_name,
        run_local=args.run_local,
    )
    if run_local:
        cluster = LocalCluster()
        client = Client(cluster)
        print("Running locally.")
    else:
        print(f"Using coiled cluster: {cluster.name}")

    # Paths derived from arguments
    output_path = f"s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/{args.model_version}/"

    gross_emis_CO2_folder = (
        f"{output_path}gross_emissions__all_C_pools__CO2_only__MgCO2/standard_model/annual_intervals/INTERVAL/_pixel_yr/40000_pixels/{args.run_date}/"
    )
    gross_emis_all_gases_folder = (
        f"{output_path}gross_emissions__all_C_pools_all_gases__MgCO2e/standard_model/annual_intervals/INTERVAL/_pixel_yr/40000_pixels/{args.run_date}/"
    )
    gross_remv_all_pools_folder = (
        f"{output_path}gross_removals__all_C_pools__MgCO2/standard_model/annual_intervals/INTERVAL/_pixel_yr/40000_pixels/{args.run_date}/"
    )
    net_flux_all_pools_CO2_folder = (
        f"{output_path}net_flux__all_C_pools__CO2_only__MgCO2/standard_model/annual_intervals/INTERVAL/_pixel_yr/40000_pixels/{args.run_date}/"
    )
    node_folder = f"{output_path}land_state_node/standard_model/annual_intervals/INTERVAL/40000_pixels/{args.run_date}/"

    zarr_s3_path = f"s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/{args.model_version}/zarr/{args.run_date}/"

    adm0_folder = "s3://gfw2-data/gadm_administrative_boundaries/v4.1/v4.1.64_from_gfw-data-lake/raster/epsg-4326/10/40000/adm0/gdal-geotiff/"
    adm0_zarr_name = (
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/GADM4_1_adm0_global/20250604/global_GADM41_adm0_20250604.zarr"
    )
    pixel_area_folder = "s3://gfw2-data/analyses/umd_area_2013__from_gfw-data-lake/v1.10/raster/epsg-4326/10/40000/area_m/gdal-geotiff/"
    pixel_area_zarr_name = (
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/pixel_area/20250604/global_pixel_area_20250604.zarr"
    )
    primary_forest_IFL_folder = "s3://gfw2-data/climate/carbon_model/ifl_primary_merged/processed/20200724/"
    primary_forest_IFL_zarr_name = (
        "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/IFL2000_tropical_primary_forest_2001/20250609/ifl_primary_forest_merged.zarr"
    )

    state_node_lookup_table_local = "./src/LULUCF/LULUCF_state_node_lookup_table.xlsx"
    state_node_lookup_table_s3 = "http://gfw2-data.s3.amazonaws.com/climate/AFOLU_flux_model/LULUCF/state_node_lookup_tables/LULUCF_state_node_lookup_table.xlsx"
    sheet = "v030_20250430"

    state_node_df = create_state_node_df(state_node_lookup_table_local, state_node_lookup_table_s3, sheet)

    # Ensure contextual zarrs exist
    ensure_zarr_exists(list_folder_uris(adm0_folder), adm0_zarr_name, args.chunk_size)
    ensure_zarr_exists(list_folder_uris(pixel_area_folder), pixel_area_zarr_name, args.chunk_size)
    ensure_zarr_exists(list_folder_uris(primary_forest_IFL_folder), primary_forest_IFL_zarr_name, args.chunk_size)

    adm0 = xr.open_zarr(adm0_zarr_name).band_data
    pixel_area = xr.open_zarr(pixel_area_zarr_name).band_data
    primary_forest_IFL = xr.open_zarr(primary_forest_IFL_zarr_name).band_data

    contextual_layer_names = ["state_nodes", "gadm_adm0", "primary_forest_IFL"]

    node_codes = np.array(
        [
            1110000,
            1120000,
            1210000,
            1220000,
            2111000,
            2112000,
            2121100,
            2121200,
            2122100,
            2122200,
            2123100,
            2123200,
            2124100,
            2124200,
            2125100,
            2125200,
            2211100,
            2211200,
            2212110,
            2212120,
            2212210,
            2212220,
            2213110,
            2213120,
            2213210,
            2213220,
            2214100,
            2214200,
            2215100,
            2215200,
            2221100,
            2221200,
            2222100,
            2222200,
            2223100,
            2223200,
            3110000,
            3120000,
            3211111,
            3211112,
            3211121,
            3211122,
            3211211,
            3211212,
            3211221,
            3211222,
            3212111,
            3212112,
            3212121,
            3212122,
            3212211,
            3212212,
            3212221,
            3212222,
            3221110,
            3221120,
            3221210,
            3221220,
            3222111,
            3222112,
            3222121,
            3222122,
            3222210,
            3222220,
            4100000,
            4210000,
            4220000,
            4310000,
            4320000,
            5100000,
            5210000,
            5220000,
            5310000,
            5320000,
        ],
        dtype=np.uint32,
    )

    gadm_adm0_ids = np.array(
        [
            0.0,
            4.0,
            8.0,
            10.0,
            12.0,
            16.0,
            20.0,
            24.0,
            28.0,
            31.0,
            32.0,
            36.0,
            40.0,
            44.0,
            48.0,
            50.0,
            51.0,
            52.0,
            56.0,
            60.0,
            64.0,
            68.0,
            70.0,
            72.0,
            74.0,
            76.0,
            84.0,
            86.0,
            90.0,
            92.0,
            96.0,
            100.0,
            104.0,
            108.0,
            112.0,
            116.0,
            120.0,
            124.0,
            132.0,
            136.0,
            140.0,
            144.0,
            148.0,
            152.0,
            156.0,
            158.0,
            162.0,
            166.0,
            170.0,
            174.0,
            175.0,
            178.0,
            180.0,
            184.0,
            188.0,
            191.0,
            192.0,
            196.0,
            203.0,
            204.0,
            208.0,
            212.0,
            214.0,
            218.0,
            222.0,
            226.0,
            231.0,
            232.0,
            233.0,
            234.0,
            238.0,
            239.0,
            242.0,
            246.0,
            248.0,
            250.0,
            254.0,
            258.0,
            260.0,
            262.0,
            266.0,
            268.0,
            270.0,
            275.0,
            276.0,
            288.0,
            292.0,
            296.0,
            300.0,
            304.0,
            308.0,
            312.0,
            316.0,
            320.0,
            324.0,
            328.0,
            332.0,
            334.0,
            336.0,
            340.0,
            348.0,
            352.0,
            356.0,
            360.0,
            364.0,
            368.0,
            372.0,
            376.0,
            380.0,
            384.0,
            388.0,
            392.0,
            398.0,
            400.0,
            404.0,
            408.0,
            410.0,
            414.0,
            417.0,
            418.0,
            422.0,
            426.0,
            428.0,
            430.0,
            434.0,
            438.0,
            440.0,
            442.0,
            450.0,
            454.0,
            458.0,
            462.0,
            466.0,
            470.0,
            474.0,
            478.0,
            480.0,
            484.0,
            492.0,
            496.0,
            498.0,
            499.0,
            500.0,
            504.0,
            508.0,
            512.0,
            516.0,
            520.0,
            524.0,
            528.0,
            531.0,
            533.0,
            534.0,
            535.0,
            540.0,
            548.0,
            554.0,
            558.0,
            562.0,
            566.0,
            70.0,
            574.0,
            578.0,
            580.0,
            581.0,
            583.0,
            584.0,
            585.0,
            586.0,
            591.0,
            598.0,
            600.0,
            604.0,
            608.0,
            612.0,
            616.0,
            620.0,
            624.0,
            626.0,
            630.0,
            634.0,
            638.0,
            642.0,
            643.0,
            646.0,
            652.0,
            654.0,
            659.0,
            660.0,
            662.0,
            663.0,
            666.0,
            670.0,
            674.0,
            678.0,
            682.0,
            686.0,
            688.0,
            690.0,
            694.0,
            702.0,
            703.0,
            704.0,
            705.0,
            706.0,
            710.0,
            716.0,
            724.0,
            728.0,
            729.0,
            732.0,
            740.0,
            744.0,
            748.0,
            752.0,
            756.0,
            760.0,
            762.0,
            764.0,
            768.0,
            772.0,
            776.0,
            780.0,
            784.0,
            788.0,
            792.0,
            795.0,
            796.0,
            798.0,
            800.0,
            804.0,
            807.0,
            818.0,
            826.0,
            831.0,
            832.0,
            833.0,
            834.0,
            840.0,
            850.0,
            854.0,
            858.0,
            860.0,
            862.0,
            876.0,
            882.0,
            887.0,
            894.0,
        ],
        dtype=np.uint16,
    )

    primary_forest_IFL_codes = np.array([0, 1], dtype=np.uint8)

    combined_df = pd.DataFrame()

    for interval_end_year in args.interval_end_years:
        interval = f"{interval_end_year - 1}_{interval_end_year}"
        print(f"Processing {interval}: {timestr()}")

        gross_emis_CO2_zarr_name = f"{zarr_s3_path}{interval}/gross_emissions__all_C_pools__CO2_only__MgCO2_{interval}.zarr"
        gross_emis_all_gases_zarr_name = f"{zarr_s3_path}{interval}/gross_emissions__all_C_pools__all_gases__MgCO2e_pixel_yr_{interval}.zarr"
        gross_remv_all_pools_zarr_name = f"{zarr_s3_path}{interval}/gross_removals__all_C_pools__MgCO2_pixel_yr_{interval}.zarr"
        net_flux_all_pools_CO2_zarr_name = f"{zarr_s3_path}{interval}/net_flux__all_C_pools__CO2_only__MgCO2_pixel_yr_{interval}.zarr"
        node_zarr_name = f"{zarr_s3_path}{interval}/land_state_node_{interval}.zarr"

        gross_emis_CO2 = xr.open_zarr(gross_emis_CO2_zarr_name).band_data
        gross_emis_all_gases = xr.open_zarr(gross_emis_all_gases_zarr_name).band_data
        gross_remv_all_pools = xr.open_zarr(gross_remv_all_pools_zarr_name).band_data
        net_flux_all_pools_CO2 = xr.open_zarr(net_flux_all_pools_CO2_zarr_name).band_data
        state_nodes = xr.open_zarr(node_zarr_name).band_data

        reference = state_nodes
        adm0_aligned = safe_crop(adm0, reference)
        primary_forest_IFL_aligned = safe_crop(primary_forest_IFL, reference)
        pixel_area_aligned = safe_crop(pixel_area, reference)
        gross_emis_CO2_aligned = safe_crop(gross_emis_CO2, reference)
        gross_emis_all_gases_aligned = safe_crop(gross_emis_all_gases, reference)
        gross_remv_all_pools_aligned = safe_crop(gross_remv_all_pools, reference)
        net_flux_all_pools_CO2_aligned = safe_crop(net_flux_all_pools_CO2, reference)

        flux_type_dict = {
            0: parse_pattern_from_uri(list_folder_uris(gross_emis_CO2_folder.replace("INTERVAL", interval))),
            1: parse_pattern_from_uri(list_folder_uris(gross_emis_all_gases_folder.replace("INTERVAL", interval))),
            2: parse_pattern_from_uri(list_folder_uris(gross_remv_all_pools_folder.replace("INTERVAL", interval))),
            3: parse_pattern_from_uri(list_folder_uris(net_flux_all_pools_CO2_folder.replace("INTERVAL", interval))),
            4: "area__ha",
        }

        flux_cube = xr.DataArray(
            xr.dask.array.stack(
                [
                    gross_emis_CO2_aligned,
                    gross_emis_all_gases_aligned,
                    gross_remv_all_pools_aligned,
                    net_flux_all_pools_CO2_aligned,
                    pixel_area_aligned,
                ]
            ),
            dims=("flux_type", "y", "x"),
        )

        flux_cube, adm0_aligned, state_nodes, primary_forest_IFL_aligned = xr.align(
            flux_cube,
            adm0_aligned,
            state_nodes,
            primary_forest_IFL_aligned,
            join="override",
        )

        adm0_aligned.name = "gadm_adm0"
        primary_forest_IFL_aligned.name = "primary_forest_IFL"
        state_nodes.name = "state_nodes"

        flux_results = xarray_reduce(
            flux_cube,
            *(adm0_aligned, state_nodes, primary_forest_IFL_aligned),
            func="sum",
            expected_groups=(gadm_adm0_ids, node_codes, primary_forest_IFL_codes),
            reindex=ReindexStrategy(blockwise=False, array_type=ReindexArrayType.SPARSE_COO),
            fill_value=0,
        ).compute()

        coord_dict = convert_to_coord_dict(flux_results, interval)
        df = create_interval_df(coord_dict, state_node_df, flux_type_dict, interval_end_year)
        df = calculate_interval_flux_densities(df, contextual_layer_names)
        combined_df = pd.concat([combined_df, df])

    combined_df = combined_df.reset_index(drop=True)
    combined_df.to_csv(args.output_csv, index=False)

    if client:
        client.close()
    if cluster:
        cluster.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LULUCF zonal statistics")
    parser.add_argument("--model_version", required=True, help="Model version string")
    parser.add_argument("--run_date", required=True, help="Model run date")
    parser.add_argument("--interval_end_years", nargs="+", type=int, required=True, help="Interval end years")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Chunk size for reading rasters")
    parser.add_argument("--output_csv", required=True, help="Path of the output CSV")
    parser.add_argument(
        "--cluster_name",
        default="zonal_stats",
        help="Name of the Coiled cluster to attach to",
    )
    parser.add_argument(
        "--run_local",
        action="store_true",
        help="Run locally without Dask/Coiled",
    )

    main(parser.parse_args())