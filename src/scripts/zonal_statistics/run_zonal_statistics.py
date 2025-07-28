# -*- coding: utf-8 -*-
"""Run organic‑soils zonal statistics.

This script converts the workflow developed in
``LULUCF_output_zonal_stats_20250613__basic_working_zonal_stats.ipynb``
into a command line utility.  It can run locally or connect to a Coiled
Dask cluster for distributed execution.  The paths and flux layers have
been updated for the organic soils model.
"""

import argparse, logging, re

import fsspec
import s3fs
import numpy as np, pandas as pd, xarray as xr
import dask.array as da
from flox import xarray_reduce, ReindexArrayType, ReindexStrategy

# LocalCluster not needed; uu.connect_to_cluster already handles local fallback.
# `Client` import is unused (uu.connect_to_cluster returns an
# initialized client).  Remove to silence lint warnings.

# Runtime utilities
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities.universal_utilities import timestr

# Absolute import so the script can run as `python run_zonal_statistics.py`
import zonal_constants as zc

# constant no longer referenced after previous unit-alignment patch

# ╭────────────────────────────────────────────────────────────────────────────╮
# │  ORGANIC SOILS ZONAL‑STATISTICS – PATH MANIFEST (single source of truth)  │
# ╰────────────────────────────────────────────────────────────────────────────╯
ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"

#  The three CLI flags --model_version, --run_date, and the per-loop
#  interval string "{interval}" are interpolated later with .format().

# Literal template – evaluated later with .format(...)
OUTPUT_BASE = "{root}/version_{model_version}/"

DRAINED_TOTAL_MG_CO2E_PIXEL = (
    OUTPUT_BASE + "drained_total_Mg_CO2e_pixel_yr/"
    "ogh_standard_model/five_year_intervals/{interval}/40000_pixels/{run_date}/"
)
BURNED_TOTAL_MG_CO2E_PIXEL = (
    OUTPUT_BASE + "burned_total_Mg_CO2e_pixel/"
    "ogh_standard_model/five_year_intervals/{interval}/40000_pixels/{run_date}/"
)

# Zarr cache folders (one per interval)
ZARR_CACHE_PREFIX = OUTPUT_BASE + "zarr/{run_date}/{interval}/"

ZARR_PATHS = {
    "drained_total_Mg_CO2e_pixel": ZARR_CACHE_PREFIX
    + "drained_total_Mg_CO2e_pixel_yr_{interval}.zarr",
    "burned_total_Mg_CO2e_pixel": ZARR_CACHE_PREFIX
    + "burned_total_Mg_CO2e_pixel_{interval}.zarr",
    "state_nodes": ZARR_CACHE_PREFIX + "land_state_node_{interval}.zarr",
}

# Contextual layers (static)
ADM0_GTIFF_FOLDER = "s3://gfw2-data/gadm_administrative_boundaries/v4.1/v4.1.64_from_gfw-data-lake/raster/epsg-4326/10/40000/adm0/gdal-geotiff/"
ADM0_ZARR = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/GADM4_1_adm0_global/20250604/global_GADM41_adm0_20250604.zarr"
PIXEL_AREA_GTIFF_FOLDER = "s3://gfw2-data/analyses/umd_area_2013__from_gfw-data-lake/v1.10/raster/epsg-4326/10/40000/area_m/gdal-geotiff/"
PIXEL_AREA_ZARR = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/pixel_area/20250604/global_pixel_area_20250604.zarr"
IFL_PRIMARY_GTIFF_FOLDER = (
    "s3://gfw2-data/climate/carbon_model/ifl_primary_merged/processed/20200724/"
)
IFL_PRIMARY_ZARR = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/IFL2000_tropical_primary_forest_2001/20250609/ifl_primary_forest_merged.zarr"

# Lookup table
STATE_NODE_XLSX_LOCAL = "./src/LULUCF/LULUCF_state_node_lookup_table.xlsx"
STATE_NODE_XLSX_S3 = "https://gfw2-data.s3.amazonaws.com/climate/AFOLU_flux_model/LULUCF/state_node_lookup_tables/LULUCF_state_node_lookup_table.xlsx"
# ===  END PATH MANIFEST  ======================================================


def create_state_node_df(
    state_node_lookup_table_local: str, state_node_lookup_table_s3: str, sheet_name: str
) -> pd.DataFrame:
    """Load the state node lookup table.

    The function first attempts to read the file from the ``state_node_lookup_table_s3``
    URL.  If that fails, it falls back to ``state_node_lookup_table_local``.  Any
    ``pandas`` supported URL or file path is accepted.
    """
    try:
        logging.info("Reading state-node lookup from %s", state_node_lookup_table_s3)
        return pd.read_excel(state_node_lookup_table_s3, sheet_name=sheet_name)
    except Exception as exc:  # pragma: no cover - network/IO errors
        logging.warning(
            "Failed to load %s: %s – using local file %s",
            state_node_lookup_table_s3,
            exc,
            state_node_lookup_table_local,
        )
        return pd.read_excel(state_node_lookup_table_local, sheet_name=sheet_name)


def list_folder_uris(base_uri: str) -> pd.Series:
    """List GeoTIFF files in an s3 folder and return them as a Series."""
    fs = s3fs.S3FileSystem(anon=False)
    all_files = fs.ls(base_uri)
    tif_files = [f"s3://{f}" for f in all_files if f.endswith(".tif")]
    return pd.Series(tif_files, dtype="string")


def parse_pattern_from_uri(uri_series: pd.Series) -> str:
    """Extract the dataset name from the first URI in ``uri_series``.

    The filenames are expected to end with ``_<start>_<end>.tif`` where
    ``start`` and ``end`` are four digit years.  Examples include organic soils
    outputs such as ``drained_total_Mg_CO2e_ha`` and ``burned_total_Mg_CO2e_ha``.
    """
    if uri_series.empty:
        raise ValueError("No GeoTIFFs found – cannot derive flux-type pattern")
    uri = uri_series.iloc[0]
    patterns = [
        r"__([A-Za-z0-9_]+)_\d{4}_\d{4}\.tif$",
        r"__([A-Za-z0-9_]+)_pixel_yr_\d{4}_\d{4}\.tif$",
    ]
    for p in patterns:
        m = re.search(p, uri)
        if m:
            return m.group(1)
    raise ValueError(f"Cannot parse flux-type from {uri}")


def make_xarray_chunks(tile_uris: pd.Series, chunk_size: int) -> xr.Dataset:
    """Open multiple GeoTIFFs into a chunked Xarray dataset."""
    return xr.open_mfdataset(
        tile_uris.values.tolist(),
        parallel=True,
        chunks={"x": chunk_size, "y": chunk_size},
    ).squeeze()


def safe_crop(ds: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Crop one dataset to another's extent **without** shifting a pixel."""
    return ds.sel(x=ref.x, y=ref.y)  # exact index match


def convert_to_coord_dict(flux_results: xr.DataArray, interval: str) -> dict:
    """Convert flox results to a coordinate dictionary."""
    logging.info("   Post-processing %s : %s", interval, timestr())
    sparse = flux_results.data  # sparse COO array
    dim_names = flux_results.dims
    indices = sparse.coords
    values = sparse.data
    coord_dict = {
        dim: flux_results.coords[dim].values[indices[i]]
        for i, dim in enumerate(dim_names)
    }
    coord_dict["value"] = values
    return coord_dict


def classify_node(state_node: int) -> str:
    """Group state node codes into broad categories."""
    node_str = str(state_node)
    first_digit = int(node_str[0])

    one_digit_map = {
        1: "forest_gain",
        2: "forest_loss",
        4: "cropland",
        5: "grassland",
    }

    three_digit_map = {
        311: "forest_loss",
        312: "forest_loss",
        321: "disturbed_forest",
        322: "stable_forest",
    }

    if first_digit == 3:
        prefix = int(node_str[:3])
        return three_digit_map.get(prefix, "unknown_3x")
    return one_digit_map.get(first_digit, "unknown")


def create_interval_df(
    coord_dict: dict,
    flux_type_dict: dict,
    interval_end_year: int,
    state_node_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert flox output to a processed dataframe."""
    df = pd.DataFrame(coord_dict)
    df["flux_type"] = df["flux_type"].replace(flux_type_dict)
    df["node_grp"] = df["state_nodes"].apply(classify_node)
    df["interval_end"] = interval_end_year
    df = df.merge(
        state_node_df[["state_nodes", "meaning"]], on="state_nodes", how="left"
    )
    df.loc[df["flux_type"].eq("area__ha"), "value"] = df["value"] / 10000
    return df


def calculate_interval_flux_densities(
    df: pd.DataFrame, contextual_layer_names: list[str]
) -> pd.DataFrame:
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
    # Guard against divide-by-zero for degenerate zones
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["value_per_ha"] = merged["value"] / merged["value_area"]
    merged.loc[merged["value_area"] == 0, "value_per_ha"] = np.nan
    new_rows = merged.copy()
    new_rows["flux_type"] = new_rows["flux_type"] + "__CO2_per_ha"
    new_rows["value"] = new_rows["value_per_ha"]
    new_rows = new_rows.drop(
        columns=["value_area", "interval_end_area", "value_per_ha"]
    )
    return pd.concat([df, new_rows], ignore_index=True)


def ensure_zarr_exists(uri_list: pd.Series, zarr_path: str, chunk_size: int) -> None:
    """Create a zarr from URIs if it does not already exist."""
    fs, path = fsspec.core.url_to_fs(zarr_path)
    if fs.exists(path):
        return
    ds = make_xarray_chunks(uri_list, chunk_size)
    ds.to_zarr(zarr_path, mode="w")


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO if not args.debug else logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cluster, client, _ = uu.connect_to_cluster(
        cluster_name=args.cluster_name,
        run_local=args.run_local,
    )

    logging.info(
        "Connected to cluster %s", cluster.name if cluster else "local-threaded"
    )

    # Resolve manifest placeholders ------------------------------------------------
    OUTPUT_KW = dict(
        root=ROOT,
        model_version=args.model_version,
        run_date=args.run_date,
    )

    adm0_folder, adm0_zarr_name = ADM0_GTIFF_FOLDER, ADM0_ZARR
    pixel_area_folder, pixel_area_zarr_name = PIXEL_AREA_GTIFF_FOLDER, PIXEL_AREA_ZARR
    primary_forest_IFL_folder, primary_forest_IFL_zarr_name = (
        IFL_PRIMARY_GTIFF_FOLDER,
        IFL_PRIMARY_ZARR,
    )

    # Reserved for future use
    # state_node_lookup_table_local, state_node_lookup_table_s3 = (
    #     STATE_NODE_XLSX_LOCAL,
    #     STATE_NODE_XLSX_S3,
    # )
    # sheet = "v030_20250430"

    # Ensure contextual zarrs exist
    ensure_zarr_exists(list_folder_uris(adm0_folder), adm0_zarr_name, args.chunk_size)
    ensure_zarr_exists(
        list_folder_uris(pixel_area_folder), pixel_area_zarr_name, args.chunk_size
    )
    ensure_zarr_exists(
        list_folder_uris(primary_forest_IFL_folder),
        primary_forest_IFL_zarr_name,
        args.chunk_size,
    )

    adm0 = xr.open_zarr(adm0_zarr_name).band_data
    pixel_area = xr.open_zarr(pixel_area_zarr_name).band_data
    primary_forest_IFL = xr.open_zarr(primary_forest_IFL_zarr_name).band_data
    state_node_df = create_state_node_df(
        args.state_node_xlsx_local,
        args.state_node_xlsx_s3,
        args.state_node_sheet,
    )

    contextual_layer_names = ["state_nodes", "gadm_adm0", "primary_forest_IFL"]

    node_codes = zc.NODE_CODES
    gadm_adm0_ids = zc.GADM_ADM0_IDS
    primary_forest_IFL_codes = zc.PRIMARY_FOREST_IFL_CODES

    combined_df = pd.DataFrame()

    for interval_end_year in args.interval_end_years:
        interval = f"{interval_end_year - 1}_{interval_end_year}"
        logging.info("Processing interval %s : %s", interval, timestr())

        drained_total_zarr_name = ZARR_PATHS["drained_total_Mg_CO2e_pixel"].format(
            interval=interval, **OUTPUT_KW
        )
        burned_total_zarr_name = ZARR_PATHS["burned_total_Mg_CO2e_pixel"].format(
            interval=interval, **OUTPUT_KW
        )
        node_zarr_name = ZARR_PATHS["state_nodes"].format(
            interval=interval, **OUTPUT_KW
        )

        drained_total = xr.open_zarr(drained_total_zarr_name).band_data
        burned_total = xr.open_zarr(burned_total_zarr_name).band_data
        state_nodes = xr.open_zarr(node_zarr_name).band_data

        reference = state_nodes
        adm0_aligned = safe_crop(adm0, reference)
        primary_forest_IFL_aligned = safe_crop(primary_forest_IFL, reference)
        pixel_area_aligned = safe_crop(pixel_area, reference)
        drained_total_aligned = safe_crop(drained_total, reference)
        burned_total_aligned = safe_crop(burned_total, reference)

        # Grab one URI from each folder to label flux_type
        flux_type_dict = {
            0: parse_pattern_from_uri(
                list_folder_uris(
                    DRAINED_TOTAL_MG_CO2E_PIXEL.format(interval=interval, **OUTPUT_KW)
                )
            ),
            1: parse_pattern_from_uri(
                list_folder_uris(
                    BURNED_TOTAL_MG_CO2E_PIXEL.format(interval=interval, **OUTPUT_KW)
                )
            ),
            2: "area__ha",
        }

        flux_cube = xr.DataArray(
            da.stack(
                [
                    drained_total_aligned,
                    burned_total_aligned,
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

        flux_type_ids = np.arange(3, dtype=np.uint8)
        flux_results = xarray_reduce(
            flux_cube,
            *(adm0_aligned, state_nodes, primary_forest_IFL_aligned),
            func="sum",
            expected_groups=(
                gadm_adm0_ids,
                node_codes,
                primary_forest_IFL_codes,
            ),
            reindex=ReindexStrategy(
                blockwise=False, array_type=ReindexArrayType.SPARSE_COO
            ),
            fill_value=np.nan,
        ).compute()

        coord_dict = convert_to_coord_dict(flux_results, interval)
        df = create_interval_df(
            coord_dict,
            flux_type_dict,
            interval_end_year,
            state_node_df,
        )
        df = calculate_interval_flux_densities(df, contextual_layer_names)
        combined_df = pd.concat([combined_df, df])

    combined_df = combined_df.reset_index(drop=True)
    logging.info("Writing Parquet output to %s", args.output_parquet)
    combined_df.to_parquet(
        args.output_parquet, index=False, partition_cols=["interval_end"]
    )
    logging.info("Parquet write complete – %d rows", len(combined_df))

    if client:
        client.close()
    if cluster:
        cluster.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run organic‑soils zonal statistics")
    parser.add_argument("--model_version", required=True, help="Model version string")
    parser.add_argument("--run_date", required=True, help="Model run date")
    parser.add_argument(
        "--interval_end_years",
        nargs="+",
        type=int,
        required=True,
        help="Interval end years",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=10000, help="Chunk size for reading rasters"
    )
    parser.add_argument("--output_parquet", required=True, help="Output Parquet folder")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
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
    parser.add_argument(
        "--state_node_xlsx_local",
        default=STATE_NODE_XLSX_LOCAL,
        help="Local path to state-node lookup table",
    )
    parser.add_argument(
        "--state_node_xlsx_s3",
        default=STATE_NODE_XLSX_S3,
        help="S3 URL for state-node lookup table",
    )
    parser.add_argument(
        "--state_node_sheet",
        default="v030_20250430",
        help="Sheet name within the lookup table",
    )
    main(parser.parse_args())

# TODO (2025-07-09): node_codes / gadm_adm0_ids lists should eventually
# move to a shared constants module or be generated dynamically from
# the rasters, so that ontology updates propagate automatically.