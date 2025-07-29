# -*- coding: utf-8 -*-
"""Run organic‑soils zonal statistics.

This script converts the workflow developed in
``LULUCF_output_zonal_stats_20250613__basic_working_zonal_stats.ipynb``
into a command line utility.  It can run locally or connect to a Coiled
Dask cluster for distributed execution.  The paths and flux layers have
been updated for the organic soils model.
"""

import argparse, logging, re, sys

import fsspec
import s3fs
import numpy as np, pandas as pd, xarray as xr
import dask.array as da
import zarr

# ── flox compatibility (works for flox 0.5  →  ≥0.10) ─────────────────────
try:
    # future‑proof in case flox later re‑exports from its root
    from flox import xarray_reduce, ReindexArrayType, ReindexStrategy  # type: ignore
except ImportError:
    from flox.xarray import xarray_reduce                              # noqa: F401
    try:
        from flox.reindex import ReindexArrayType, ReindexStrategy     # noqa: F401
    except ImportError:
        # flox ≤0.8 – no sparse‑reindex API; fall back to defaults
        ReindexArrayType = None
        ReindexStrategy = None
# ──────────────────────────────────────────────────────────────────────────


# LocalCluster not needed; uu.connect_to_cluster already handles local fallback.
# `Client` import is unused (uu.connect_to_cluster returns an
# initialized client).  Remove to silence lint warnings.

# Runtime utilities
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities.universal_utilities import timestr

# Absolute import so the script can run as `python run_zonal_statistics.py`
import src.scripts.zonal_statistics.zonal_constants as zc

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
ADM0_GTIFF_FOLDER = "s3://gfw2-data/gadm_administrative_boundaries/v4.1/v4.1.64__from_gfw-data-lake/raster/epsg-4326/10/40000/adm0/gdal-geotiff/"
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


"""
def create_state_node_df(
    state_node_lookup_table_local: str, state_node_lookup_table_s3: str, sheet_name: str
) -> pd.DataFrame:
    ""Load the state node lookup table from S3 falling back to a local file.""
    pass  # Deprecated – retained for reference
"""


def list_folder_uris(base_uri: str) -> pd.Series:
    """Return GeoTIFF URIs within ``base_uri`` (recursively)."""
    fs = s3fs.S3FileSystem(anon=False)
    pattern = base_uri.rstrip("/") + "/**/*.tif"
    tif_files = [
        (f if f.startswith("s3://") else f"s3://{f}") for f in fs.glob(pattern)
    ]
    if not tif_files:
        raise FileNotFoundError(f"No GeoTIFFs found in {base_uri}")
    return pd.Series(tif_files, dtype="string")


def parse_pattern_from_uri(uri_series: pd.Series) -> str:
    """Extract the dataset name from the first URI in ``uri_series``.

    The filenames are expected to end with ``_<start>_<end>.tif`` where
    ``start`` and ``end`` are four digit years.  Examples include organic soils
    outputs such as ``drained_total_Mg_CO2e_ha`` and ``burned_total_Mg_CO2e_ha``.
    """
    if uri_series.empty:
        raise ValueError("No GeoTIFFs found – cannot derive flux‑type pattern")

    patterns = [
        r"__([A-Za-z0-9_]+)_\d{4}_\d{4}\.tif$",
        r"__([A-Za-z0-9_]+)_pixel_yr_\d{4}_\d{4}\.tif$",
    ]
    matches = {
        re.search(pat, u).group(1)
        for u in uri_series
        for pat in patterns
        if re.search(pat, u)
    }
    if len(matches) != 1:
        raise ValueError(f"Mixed or unparseable flux‑type patterns: {matches}")
    return matches.pop()


def make_xarray_chunks(tile_uris: pd.Series, chunk_size: int) -> xr.Dataset:
    """Open multiple GeoTIFFs into a chunked Xarray dataset."""
    return xr.open_mfdataset(
        tile_uris.values.tolist(),
        parallel=True,
        chunks={"x": chunk_size, "y": chunk_size},
    ).squeeze()


def safe_crop(ds: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Crop one dataset to another's extent **without** shifting a pixel.
       Abort if grids are not identical (guard against silent mis‑alignment)."""
    if not (ds.x.equals(ref.x) and ds.y.equals(ref.y)):
        raise ValueError("Raster grid mis‑aligned with reference; aborting.")
    return ds.sel(x=ref.x, y=ref.y)


def crop_to_bbox(ds: xr.DataArray, bbox: list[float]) -> xr.DataArray:
    """Slice ``ds`` to ``bbox`` handling coordinate order."""
    west, south, east, north = bbox
    x_slice = slice(west, east) if ds.x[0] < ds.x[-1] else slice(east, west)
    y_slice = slice(south, north) if ds.y[0] < ds.y[-1] else slice(north, south)
    return ds.sel(x=x_slice, y=y_slice)

# ───────────────────────────────────────────────────────────────────────────
# New helper – read only the AOI region *before* chunking
# ───────────────────────────────────────────────────────────────────────────
def open_zarr_region(path: str, bbox: list[float] | None, chunk_size: int) -> xr.DataArray:
    """Open a Zarr store and return a 2‑D ``DataArray``.

    The helper tries to be tolerant of input variations:
    - the first variable containing both ``x`` and ``y`` dims is selected
    - a leading ``band`` dimension is dropped when present
    - when ``bbox`` is supplied and ``x``/``y`` exist the data are cropped
    - chunking is applied only to existing ``x``/``y`` dims
    """

    mapper = fsspec.get_mapper(path, anon=False, check=False)
    ds = xr.open_zarr(mapper)

    # pick variable
    if isinstance(ds, xr.DataArray):
        da = ds
    else:
        vars_xy = [v for v in ds.data_vars.values() if {"x", "y"}.issubset(v.dims)]
        da = vars_xy[0] if vars_xy else next(iter(ds.data_vars.values()))

    # drop band if needed
    if "band" in da.dims:
        da = da.isel(band=0, drop=True)

    # optional crop
    if bbox is not None and {"x", "y"}.issubset(da.dims):
        west, south, east, north = bbox
        x_slice = slice(west, east) if west < east else slice(east, west)
        y_slice = slice(south, north) if south < north else slice(north, south)
        da = da.sel(x=x_slice, y=y_slice)
    elif bbox is not None:
        logging.warning("Skipping spatial crop for %s – x/y dims not present.", path)

    chunk_dict = {d: chunk_size for d in ("x", "y") if d in da.dims}
    return da.chunk(chunk_dict)


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


def create_interval_df(
    coord_dict: dict,
    flux_type_dict: dict,
    interval_end_year: int,
) -> pd.DataFrame:
    """Convert flox output to a processed dataframe."""
    df = pd.DataFrame(coord_dict)
    df["flux_type"] = df["flux_type"].replace(flux_type_dict)
    df["interval_end"] = interval_end_year
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
    """Create a zarr from URIs if it does not already exist.

    If the store exists but lacks consolidated metadata, create ``.zmetadata``.
    """
    logging.debug("Ensuring Zarr store %s", zarr_path)
    fs, path = fsspec.core.url_to_fs(zarr_path)
    group_exists = fs.exists(f"{path}/.zgroup")
    metadata_exists = fs.exists(f"{path}/.zmetadata")

    if group_exists and metadata_exists:
        return

    if uri_list.empty:
        raise FileNotFoundError(f"No GeoTIFFs found for {zarr_path}")

    if not group_exists:
        logging.debug("Creating new Zarr store %s", zarr_path)
        ds = make_xarray_chunks(uri_list, chunk_size)
        # Rechunk to ensure uniform chunk sizes for the Zarr output
        ds = ds.chunk({"x": chunk_size, "y": chunk_size})
        ds.to_zarr(zarr_path, mode="w")

    if not metadata_exists:
        logging.debug("Consolidating metadata for %s", zarr_path)
        zarr.convenience.consolidate_metadata(fs.get_mapper(path))


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO if not args.debug else logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.debug("Starting run with args: %s", args)

    cluster, client, _ = uu.connect_to_cluster(
        cluster_name=args.cluster_name,
        run_local=args.run_local,
    )
    logging.info(
        "Connected to cluster %s", cluster.name if cluster else "local-threaded"
    )
    if client:
        logging.debug("Dask client info: %s", client)

    bbox = None
    if args.bounding_box:
        bbox = [float(x) for x in args.bounding_box]
        logging.debug("Using bounding box: %s", bbox)
    elif args.tile_ids:
        tiles: list[str] = []
        for item in args.tile_ids:
            tiles.extend(t.strip() for t in item.split(",") if t.strip())
        if tiles:
            bounds = [uu.get_10x10_tile_bounds(t) for t in tiles]
            west = min(b[0] for b in bounds)
            south = min(b[1] for b in bounds)
            east = max(b[2] for b in bounds)
            north = max(b[3] for b in bounds)
            bbox = [west, south, east, north]
            logging.debug("Calculated bounding box from tiles %s: %s", tiles, bbox)

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
    logging.debug("Checking contextual layer adm0")
    ensure_zarr_exists(list_folder_uris(adm0_folder), adm0_zarr_name, args.chunk_size)
    logging.debug("Checking contextual layer pixel_area")
    ensure_zarr_exists(
        list_folder_uris(pixel_area_folder), pixel_area_zarr_name, args.chunk_size
    )
    logging.debug("Checking contextual layer primary_forest_IFL")
    ensure_zarr_exists(
        list_folder_uris(primary_forest_IFL_folder),
        primary_forest_IFL_zarr_name,
        args.chunk_size,
    )

    logging.debug("Opening contextual layers")
    adm0 = open_zarr_region(adm0_zarr_name, bbox, args.chunk_size)
    pixel_area = open_zarr_region(pixel_area_zarr_name, bbox, args.chunk_size)
    primary_forest_IFL = open_zarr_region(
        primary_forest_IFL_zarr_name, bbox, args.chunk_size
    )

    contextual_layer_names = ["state_nodes", "gadm_adm0", "primary_forest_IFL"]

    node_codes = zc.NODE_CODES
    gadm_adm0_ids = zc.GADM_ADM0_IDS
    primary_forest_IFL_codes = zc.PRIMARY_FOREST_IFL_CODES

    first_write = True   # incremental Parquet write flag

    for interval_end_year in args.interval_end_years:
        interval = f"{interval_end_year - 1}_{interval_end_year}"
        logging.info("Processing interval %s : %s", interval, timestr())
        logging.debug("Opening flux zarrs for interval %s", interval)

        drained_total_zarr_name = ZARR_PATHS["drained_total_Mg_CO2e_pixel"].format(
            interval=interval, **OUTPUT_KW
        )
        burned_total_zarr_name = ZARR_PATHS["burned_total_Mg_CO2e_pixel"].format(
            interval=interval, **OUTPUT_KW
        )
        node_zarr_name = ZARR_PATHS["state_nodes"].format(
            interval=interval, **OUTPUT_KW
        )

        # --- build flux‑layer caches if missing ----------------------------
        drained_folder = DRAINED_TOTAL_MG_CO2E_PIXEL.format(interval=interval, **OUTPUT_KW)
        burned_folder  = BURNED_TOTAL_MG_CO2E_PIXEL.format(interval=interval, **OUTPUT_KW)
        ensure_zarr_exists(
            list_folder_uris(drained_folder), drained_total_zarr_name, args.chunk_size
        )
        ensure_zarr_exists(
            list_folder_uris(burned_folder), burned_total_zarr_name, args.chunk_size
        )
        # -------------------------------------------------------------------

        drained_total = open_zarr_region(
            drained_total_zarr_name, bbox, args.chunk_size
        )
        burned_total = open_zarr_region(
            burned_total_zarr_name, bbox, args.chunk_size
        )
        state_nodes = open_zarr_region(node_zarr_name, bbox, args.chunk_size)
        logging.debug("Flux layers opened for interval %s", interval)

        reference = state_nodes
        adm0_aligned = safe_crop(adm0, reference)
        primary_forest_IFL_aligned = safe_crop(primary_forest_IFL, reference)
        pixel_area_aligned = safe_crop(pixel_area, reference)
        drained_total_aligned = safe_crop(drained_total, reference)
        burned_total_aligned = safe_crop(burned_total, reference)
        logging.debug("Datasets aligned for interval %s", interval)

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
        logging.debug("Flux cube stacked for interval %s", interval)

        flux_cube, adm0_aligned, state_nodes, primary_forest_IFL_aligned = xr.align(
            flux_cube,
            adm0_aligned,
            state_nodes,
            primary_forest_IFL_aligned,
            join="override",
        )
        logging.debug("Arrays aligned for reduction for interval %s", interval)

        adm0_aligned.name = "gadm_adm0"
        primary_forest_IFL_aligned.name = "primary_forest_IFL"
        state_nodes.name = "state_nodes"

        # Build reduction kwargs based on flox version
        _xr_kwargs = {}
        if ReindexStrategy is not None:
            _xr_kwargs["reindex"] = ReindexStrategy(
                blockwise=False, array_type=ReindexArrayType.SPARSE_COO
            )
        else:
            logging.warning(
                "Sparse re-index helpers missing – using dense aggregation; "
                "memory use will be higher. Consider installing flox >= 0.10."
            )

        flux_type_ids = np.arange(3, dtype=np.uint8)
        logging.debug("Running flox reduce for interval %s", interval)
        flux_results = xarray_reduce(
            flux_cube,
            *(adm0_aligned, state_nodes, primary_forest_IFL_aligned),
            func="sum",
            expected_groups=(
                gadm_adm0_ids,
                node_codes,
                primary_forest_IFL_codes,
            ),
            fill_value=np.nan,
            **_xr_kwargs,
        ).compute()
        logging.debug("Flox reduce complete for interval %s", interval)

        coord_dict = convert_to_coord_dict(flux_results, interval)
        df = create_interval_df(coord_dict, flux_type_dict, interval_end_year)
        df = calculate_interval_flux_densities(df, contextual_layer_names)
        df.to_parquet(
            args.output_parquet,
            partition_cols=["interval_end"],
            append=not first_write,
            index=False,
            compression="zstd",
            engine="pyarrow",
        )
        logging.info("Wrote results for interval %s", interval)
        first_write = False
        del df

    logging.info("Parquet write complete – partitions written to %s", args.output_parquet)

    if client:
        logging.debug("Closing Dask client")
        client.close()
    if cluster:
        logging.debug("Closing cluster")
        cluster.close()


def main(argv=None):
    """Parse CLI args and dispatch to :func:`run`.

    When ``argv`` is empty the function runs a small local smoke test
    over a 1×1‑degree tile using default S3 paths.
    """

    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("No CLI args: running 1×1‑degree local smoke test")
        argv = [
            "--model_version",
            "test",
            "--run_date",
            "20250101",
            "--interval_end_years",
            "2020",
            "--output_parquet",
            "zonal_stats_test.parquet",
            "--run_local",
            "--bounding_box",
            "112",
            "-2",
            "113",
            "-1",
        ]

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
        "--chunk_size",
        type=int,
        default=4096,
        help="Tile chunk in pixels (lower -> less per-task memory)",
    )
    parser.add_argument("--output_parquet", required=True, help="Output Parquet folder")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run_local",
        action="store_true",
        help="Run locally without Dask/Coiled",
    )
    mode.add_argument(
        "--cluster_name",
        default="zonal_stats",
        help="Name of the Coiled cluster to attach to",
    )
    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W S E N")
    parser.add_argument("--tile_ids", action="append", help="Comma separated tile IDs")

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()

# TODO (2025-07-09): node_codes / gadm_adm0_ids lists should eventually
# move to a shared constants module or be generated dynamically from
# the rasters, so that ontology updates propagate automatically.


"""
python -m src.scripts.zonal_statistics.run_zonal_statistics \
       --interval_end_years 2024 \
       --cluster_name zonal_stats \
       --run_date 20250724 \
       --tile_ids 00N_110E \
       --model_version 0_5_0 \
       --output_parquet zonal_stats_test.parquet
"""