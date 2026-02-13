"""
Zonal stats for vegetation

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Coiled small tests (needs 32 GB because of per-ha and per-pixel outputs):
python -m src.utilities.create_cluster -n 1 -m 64 -cn vegetation_zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.vegetation_zonal_stats -cn vegetation_zonal_stats -bb 10 49 11 50 -fv 2 -ft 2 -mt standard -mpd global --input_date YYYYMMDD

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -m 64 -cn vegetation_zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.vegetation_zonal_stats -cn vegetation_zonal_stats -bb -64 -22 -63 -21 -fv 3 -ft 3 -mt standard -mpd global --input_date YYYYMMDD

Coiled Cerrado test (174 features):
python -m src.utilities.create_cluster -n 50 -m 64 -cn vegetation_zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.vegetation_zonal_stats -cn vegetation_zonal_stats -mt standard -mpd global -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__Cerrado_center_in.shp --input_date YYYYMMDD

Coiled large shapefile test (1884 features):
python -m src.utilities.create_cluster -n 50 -m 64 -cn vegetation_zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.vegetation_zonal_stats -cn vegetation_zonal_stats -mt standard -mpd global -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD

Full run:
python -m src.utilities.create_cluster -n 50 -m 64 -cn vegetation_zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.vegetation_zonal_stats -cn vegetation_zonal_stats -mt standard -mpd global -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --input_date YYYYMMDD --log_note "10x10 deg tile creation for vegetation model v1.0.5 (2016-2024)."
"""

import argparse
import pandas as pd
import os
import time
from dask.distributed import print
import xarray as xr
import numpy as np
from flox.xarray import xarray_reduce
from flox import ReindexArrayType, ReindexStrategy
from io import BytesIO
import requests

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import resize_cluster


# Creates a Pandas dataframe with the state_nodes codes and meanings from an Excel spreadsheet
def create_state_node_df(state_node_lookup_table_local, state_node_lookup_table_s3, sheet_name):

    try:
        # Tries fetching the file from the S3 URL
        # print(f"Attempting to download file from URL: {spreadsheet}")
        response = requests.get(state_node_lookup_table_s3, timeout=10)
        response.raise_for_status()
        state_node_df = pd.read_excel(BytesIO(response.content), sheet_name=sheet_name)

    except (requests.exceptions.RequestException, Exception) as e:
        print(f"Failed to download file from S3. Falling back to local file. Error: {e}")

        print(f"Reading file from local path: {state_node_lookup_table_local}")
        state_node_df = pd.read_excel(state_node_lookup_table_local, sheet_name=sheet_name)

    return state_node_df

# Crops one input to the other input's extent.
# ref is the reference dataset that is being cropped to.
# From long chat in https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/684749fe-7b30-800a-ba8b-c502377f2c3a
def safe_crop(ds, ref):
    return ds.sel(x=ref.x, y=ref.y, method="nearest")


# Fix floating-point precision issues
def round_coords(ds, decimals=5):
    ds = ds.assign_coords({
        'x': np.round(ds.coords['x'].values, decimals),
        'y': np.round(ds.coords['y'].values, decimals)
    })
    return ds


# Converts results of flox to coordinate dictionary.
# This code came from Solomon Negusse and I haven't changed it in any substantial way.
def convert_to_coord_dict(flux_results, main_logger):

    main_logger.info(f"  Postprocessing: {uu.timestr()}")
    sparse_data = flux_results.data

    dim_names = flux_results.dims
    indices = sparse_data.coords  # tuple of arrays with indices into each dim
    values = sparse_data.data  # non-zero values

    coord_dict = {
        dim: flux_results.coords[dim].values[indices[i]]
        for i, dim in enumerate(dim_names)
    }
    coord_dict["value"] = values

    return coord_dict


# Converts flox output to dataframe and does some processing of it:
# replaces the numeric flux type with the name
# classifies specific flux types to larger groupings
# adds the interval end year to the dataframe
# adds the state node meaning to the dataframe
# converts area from m^2 to ha
def create_df(coord_dict, state_node_df, merge_keys, tile_id):

    df = pd.DataFrame(coord_dict)
    # print(df)

    df['tile_id'] = str(tile_id)

    # Splits df into pixel area and other analysis layers
    df_area = (
        df[df['analysis_layer'] == 'pixel_area_ha']
          .rename(columns={'value': 'pixel_area_ha'})
          [merge_keys + ['pixel_area_ha']]
    )

    # Non-pixel area analysis layers
    df_other = df[df['analysis_layer'] != 'pixel_area_ha']

    # Removs _ha_yr from analysis layer names
    df_other.loc[:, 'analysis_layer'] = df_other['analysis_layer'].str.replace('_ha_yr', '', regex=False)

    # Merge area values into flux rows
    df_with_areas = df_other.merge(df_area, on=merge_keys, how='left')

    # Adds the state_node meaning and classifications to the dataframe
    df_with_areas = df_with_areas.merge(state_node_df[['state_nodes', 'meaning', 'broad_class', 'detailed_class']],
              left_on='land_state_node', right_on='state_nodes',
              how='left')
    # print("merged:", df_with_areas)

    # Replaces the year index with the actual reporting year
    df_with_areas['year'] = df_with_areas['year'] + 2016

    # Deletes redundant state node column
    df_with_areas = df_with_areas.drop(columns=['state_nodes'])

    # Converts numeric codes to ISO codes
    # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/698a53aa-8674-832c-b734-4bd8afc6a6df
    # Based on https://github.com/wri/project-zeno-data-infra/blob/main/notebooks/grasslands_areas_gadm_2000-2022.ipynb originally
    if 'adm0' in df_with_areas.columns:
        df_with_areas['adm0'] = df_with_areas['adm0'].map(cn.numeric_to_alpha3)
        df_with_areas['country_name'] = df_with_areas['adm0'].map(cn.iso_to_country)
        df_with_areas['region'] = df_with_areas['adm0'].map(cn.iso_to_region)

    # Map cont_eco to continent and ecozone
    # From https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/698a53aa-8674-832c-b734-4bd8afc6a6df
    df_with_areas['continent'] = df_with_areas['cont_eco'].map(lambda x: cn.cont_eco_to_text.get(x, {}).get('continent'))
    df_with_areas['continent_ecozone'] = df_with_areas['cont_eco'].map(lambda x: cn.cont_eco_to_text.get(x, {}).get('ecozone'))

    df_with_areas['flux_Mg_ha'] = df_with_areas['value'] / df_with_areas['pixel_area_ha'].replace(0, pd.NA)

    return df_with_areas


def main(cluster_name, input_date, model_type, no_log, no_upload, chunk_shapefile_uri=False, bounding_box=None,
         first_variables_to_process=None, first_tiles_to_process=None, model_path_description=None, log_note=None):


    ### Step 1: Preparation

    # Model stage being run
    stage = 'vegetation_zonal_statistics'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, False)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    main_logger.info(f"Stage {stage} started at: {uu.timestr()}")
    main_logger.info(f"Vegetation model version: {cn.veg_model_version}")
    main_logger.info(f"Vegatation model path descriptor: {model_path_description}")
    main_logger.info(f"Start year: {cn.first_model_year_annual}; end year: {cn.last_model_year_annual}")
    main_logger.info(f"Input date: {input_date}")
    main_logger.info(f"no_upload: {no_upload}")

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_size_deg = 1   # Chunk size for geotifs is set at 1x1 deg
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size_deg, None, fishnet_iso_df, main_logger)

    # Gets a list of unique tile_ids from the chunk list
    tile_ids = []
    for chunk in chunk_list:
        tile_id = uu.xy_to_tile_id(chunk[0], chunk[3])  # tile_id in YYN/S_XXXE/W
        tile_ids.append(tile_id)

    unique_tile_ids = sorted(list(set(tile_ids)))

    # Outputs to performs zonal stats on
    full_list_of_vars = [cn.gross_emis_all_C_pools_CO2_only_pattern, cn.ch4_gross_emis_pattern, cn.n2o_gross_emis_pattern,
                         cn.gross_emis_all_C_pools_non_CO2_only_pattern, cn.gross_emis_all_C_pools_all_gases_pattern,
                         cn.gross_removals_all_C_pools_pattern,
                         cn.net_flux_all_C_pools_CO2_only_pattern, cn.net_flux_all_C_pools_all_gases_pattern,
                         cn.non_soil_c_modeled_dens_pattern]

    full_list_of_vars_with_units = [
        zu.add_units_year_to_pattern(var_name, 9999)[0]  # Dummy year since we don't need the year to access the datasets in the zarr
        for var_name in full_list_of_vars
    ]

    # Limits the processed variables to the supplied number (for testing)
    if first_variables_to_process:
        vars_to_process = full_list_of_vars_with_units[0:first_variables_to_process]
    else:
        vars_to_process = full_list_of_vars_with_units
    main_logger.info(f"Variables to run zonal stats on for: {vars_to_process} ({len(vars_to_process)} out of {len(full_list_of_vars_with_units)})")

    if first_tiles_to_process:
        tile_ids_to_process = unique_tile_ids[0:first_tiles_to_process]
    else:
        tile_ids_to_process = unique_tile_ids
    tile_ids_to_process = ['00N_020E'] #TODO for testing
    main_logger.info(f"tile_ids to perform zonal stats on: {tile_ids_to_process} ({len(tile_ids_to_process)} out of {len(unique_tile_ids)})")

    # lat-long chunk size for source zarr
    source_zarr_chunk_size = cn.chunk_dims  #4000x4000

    # The zarr path that's being used
    zarr_path = zu.create_zarr_path(cn.veg_outputs_path_mega_zarr, source_zarr_chunk_size, 'annual',
                                         model_type, cn.veg_model_version_underscore, model_path_description,
                                         input_date, main_logger)
    main_logger.info(f"Zonal stats from zarr ({source_zarr_chunk_size} pixel chunks): {zarr_path}")

    # Creates dataframe of state_node codes and meanings
    state_node_df = create_state_node_df(cn.state_node_lookup_table_local, cn.state_node_lookup_table_s3, cn.sheet)
    node_codes = np.array(list(state_node_df['state_nodes']), dtype=np.uint32)
    main_logger.info(f"State nodes are from: {cn.state_node_lookup_table_local} or {cn.state_node_lookup_table_s3}, sheet {cn.sheet}")
    main_logger.info(f"State nodes are: {node_codes}")


    ### Step 2: Prepare input zarrs

    prep_start_time = time.time()

    adm0_xr = xr.open_zarr(cn.adm0_zarr_path, consolidated=False).rename_vars(band_data='adm0')
    pixel_area_xr = xr.open_zarr(cn.pixel_area_zarr_path, consolidated=False).rename_vars(band_data='pixel_area')
    WDPA_xr = xr.open_zarr(cn.wdpa_zarr_path, consolidated=False).rename_vars(band_data='WDPA')
    BRA_biomes_xr = xr.open_zarr(cn.BRA_biomes_zarr_path, consolidated=False).rename_vars(band_data='BRA_biomes')
    cont_eco_xr = xr.open_zarr(cn.cont_eco_zarr_path, consolidated=False).rename_vars(band_data='cont_eco')

    ds = xr.open_zarr(zarr_path, consolidated=False)

    ds_selected_analysis_vars = ds[vars_to_process]

    main_logger.info(f"Rounding coordinates: {uu.timestr()}")
    reference = round_coords(pixel_area_xr["pixel_area"])
    ds_selected_analysis_vars = round_coords(ds_selected_analysis_vars)
    adm0_xr = round_coords(adm0_xr)
    WDPA_xr = round_coords(WDPA_xr)
    BRA_biomes_xr = round_coords(BRA_biomes_xr)
    land_state_node = round_coords(ds["land_state_node"])

    main_logger.info(f"Cropping: {uu.timestr()}")
    pixel_area_aligned = reference
    adm0_aligned = safe_crop(adm0_xr, reference)
    WDPA_aligned = safe_crop(WDPA_xr, reference)
    BRA_biomes_aligned = safe_crop(BRA_biomes_xr, reference)
    cont_eco_aligned = safe_crop(cont_eco_xr, reference)
    land_state_node_aligned = safe_crop(land_state_node, reference)
    ds_selected_analysis_vars_aligned = safe_crop(ds_selected_analysis_vars, reference)

    main_logger.info(f"Selecting datasets: {uu.timestr()}")
    # List of selected variable names (already aligned and cropped)
    selected_datasets = list(ds_selected_analysis_vars_aligned.data_vars)

    # Expand pixel_area to match shape of flux variables
    pixel_area_expanded = pixel_area_aligned.expand_dims(year=ds_selected_analysis_vars_aligned.year)

    # Use the exact same x/y coordinates for both
    x_coords = reference.coords['x']
    y_coords = reference.coords['y']

    # Replace coords in both sources
    main_logger.info(f"Replacing coordinates: {uu.timestr()}")
    ds_selected_analysis_vars_aligned = ds_selected_analysis_vars_aligned.assign_coords(x=x_coords, y=y_coords)

    # Multiply each flux var by pixel_area
    main_logger.info(f"Calculating per-pixel values for analysis layers: {uu.timestr()}")
    flux_layers = []
    for var in selected_datasets:
        flux_scaled = ((ds_selected_analysis_vars_aligned[var] * pixel_area_expanded) / 10000).astype("float32")
        flux_layers.append(flux_scaled)

    # Convert pixel_area from m² to hectares, then adds to the list of layers to analyze (to get area of contextual layers)
    main_logger.info(f"Calculating pixel area: {uu.timestr()}")
    pixel_area_layer = (pixel_area_expanded / 10000).astype("float32")
    flux_layers.append(pixel_area_layer)

    # Also updates the list of analysis layer names
    selected_datasets.append("pixel_area_ha")

    # Stack into one flux cube: shape (analysis_layer, year, y, x)
    main_logger.info(f"Stacking analysis layers into flux cube: {uu.timestr()}")
    flux_cube = xr.concat(flux_layers, dim="analysis_layer")

    # Set the analysis_layer coordinate names
    flux_cube = flux_cube.assign_coords(
        analysis_layer=("analysis_layer", selected_datasets)
    )
    flux_cube = round_coords(flux_cube)
    print(flux_cube)

    prep_end_time = time.time()
    main_logger.info(f"  Finished zonal stats prep, took {round(prep_end_time - prep_start_time)} seconds: {uu.timestr()}")


    # Part 3: Do zonal stats tile by tile

    main_logger.info(f"Starting zonal stats: {uu.timestr()}")
    combined_list = []

    for tile_id in tile_ids_to_process:
        main_logger.info(f"Processing {tile_id}: {uu.timestr()}")
        tile_start_time = time.time()

        west, south, east, north = uu.get_10x10_tile_bounds(tile_id)

        # Subset the flux cube by x/y coordinates
        main_logger.info(f"  Subsetting: {uu.timestr()}")
        flux_cube_subset = flux_cube.sel(
            x=slice(west, east),
            y=slice(north, south)  # Note: y typically decreases from top to bottom
        )

        adm0_aligned_subset = adm0_aligned.sel(x=slice(west, east), y=slice(north, south))
        WDPA_aligned_subset = WDPA_aligned.sel(x=slice(west, east), y=slice(north, south))
        BRA_biomes_aligned_subset = BRA_biomes_aligned.sel(x=slice(west, east), y=slice(north, south))
        cont_eco_aligned_subset = cont_eco_aligned.sel(x=slice(west, east), y=slice(north, south))
        land_state_node_aligned_subset = land_state_node_aligned.sel(x=slice(west, east), y=slice(north, south))
        pixel_area_expanded_subset = pixel_area_expanded.sel(x=slice(west, east), y=slice(north, south))

        # print("Flux cube x range:", flux_cube_subset.coords['x'].values.min(), flux_cube_subset.coords['x'].values.max(), "len:", len(flux_cube_subset.coords['x']))
        # print("Pixel area x range:", pixel_area_expanded_subset.coords['x'].values.min(), pixel_area_expanded_subset.coords['x'].values.max(), "len:", len(pixel_area_expanded_subset.coords['x']))
        # print("land_state_node x range:", land_state_node_aligned_subset.coords['x'].values.min(), land_state_node_aligned_subset.coords['x'].values.max(), "len:", len(land_state_node_aligned_subset.coords['x']))

        # # For datasets that don't have global coverage
        # try:
        #     print("ADM0 x range:", adm0_aligned_subset["adm0"].coords['x'].values.min(), adm0_aligned_subset["adm0"].coords['x'].values.max(), "len:", len(adm0_aligned_subset["adm0"].coords['x']))
        # except:
        #     print("  ADM0 not in chunk extent")

        # try:
        #     print("IFL x range:", primary_forest_IFL_aligned_subset["primary_forest_IFL"].coords['x'].values.min(), primary_forest_IFL_aligned_subset["primary_forest_IFL"].coords['x'].values.max(), "len:", len(primary_forest_IFL_aligned_subset["primary_forest_IFL"].coords['x']))
        # except:
        #     print("  IFL/primary forest not in chunk extent")

        # try:
        #     print("WDPA x range:", WDPA_aligned_subset["WDPA"].coords['x'].values.min(), WDPA_aligned_subset["WDPA"].coords['x'].values.max(), "len:", len(WDPA_aligned_subset["WDPA"].coords['x']))
        # except:
        #     print("  WDPA not in chunk extent")

        # try:
        #     print("BRA biomes x range:", BRA_biomes_aligned_subset["BRA_biomes"].coords['x'].values.min(), BRA_biomes_aligned_subset["BRA_biomes"].coords['x'].values.max(), "len:", len(BRA_biomes_aligned_subset["BRA_biomes"].coords['x']))
        # except:
        #     print("  BRA biomes not in chunk extent")

        # try:
        #     print("cont_eco x range:", cont_eco_aligned_subset["cont_eco"].coords['x'].values.min(), cont_eco_aligned_subset["cont_eco"].coords['x'].values.max(), "len:", len(cont_eco_aligned_subset["cont_eco"].coords['x']))
        # except:
        #     print("  cont_eco not in chunk extent")

        # Final alignment
        main_logger.info(f"  Aligning: {uu.timestr()}")
        (flux_cube_subset,
         pixel_area_expanded_subset,
         adm0_aligned_subset,
         WDPA_aligned_subset,
         cont_eco_aligned_subset,
         land_state_node_aligned_subset,
         ) = xr.align(
            flux_cube_subset,
            pixel_area_expanded_subset,
            adm0_aligned_subset["adm0"],
            WDPA_aligned_subset["WDPA"],
            cont_eco_aligned_subset["cont_eco"],
            land_state_node_aligned_subset,
            join="override"
        )

        flux_cube_subset = flux_cube_subset.persist()

        main_logger.info(f"  Computing: {uu.timestr()}")
        results = xarray_reduce(
            flux_cube_subset,
            *(
                adm0_aligned_subset,
                land_state_node_aligned_subset,
                WDPA_aligned_subset,
                cont_eco_aligned_subset,
                flux_cube_subset["year"]
            ),
            func='sum',
            expected_groups=(
                cn.gadm_adm0_ids,
                node_codes,
                cn.WDPA_codes,
                cn.cont_eco_codes,
                flux_cube_subset.year.values,
            ),
            group_dims=["year"],
            reindex=ReindexStrategy(blockwise=False, array_type=ReindexArrayType.SPARSE_COO),
            fill_value=0
        ).compute()

        # Contextual layers to use to merge pixel_area against other analysis layers (to calculate flux/ha)
        contextual_layers = [
            'adm0',
            'land_state_node',
            'WDPA',
            'cont_eco',
            'year'
        ]

        coord_dict = convert_to_coord_dict(results, main_logger)
        # coord_dict
        df = create_df(coord_dict, state_node_df, contextual_layers, tile_id)
        main_logger.info(f"  Rows in {tile_id} dataframe: {len(df.index)}")

        # Appends the tile-level df to a list of dfs
        combined_list.append(df)

        tile_df_name = f'veg_model_zonal_stats_{tile_id}_v{cn.veg_model_version_underscore}_{time.strftime('%Y%m%d_%H_%M_%S')}'
        df.to_parquet(f"/mnt/c/GIS/AFOLU_flux_model/flox_zonal_stats__TEMP/{tile_df_name}.parquet")
        df.to_csv(f"/mnt/c/GIS/AFOLU_flux_model/flox_zonal_stats__TEMP/{tile_df_name}.csv")

        tile_end_time = time.time()

        main_logger.info(f"  Done with {tile_id}, took {round(tile_end_time) - round(tile_start_time)} seconds: {uu.timestr()}")


    # Combines all the tile-level dfs in the list into a single df
    combined_df = pd.concat(combined_list, axis=0, ignore_index=True)

    main_logger.info(f"Rows in combined dataframe: {len(combined_df.index)}")
    main_logger.info(combined_df.head())

    combined_df_name = f'veg_model_zonal_stats_v{cn.veg_model_version_underscore}_{time.strftime('%Y%m%d_%H_%M_%S')}'
    combined_df.to_parquet(f"/mnt/c/GIS/AFOLU_flux_model/flox_zonal_stats__TEMP/{combined_df_name}.parquet")
    if len(combined_df.index) < 900_000:  # Only writes combined file to Excel if it's not giant
        combined_df.to_csv(f"/mnt/c/GIS/AFOLU_flux_model/flox_zonal_stats__TEMP/{combined_df_name}.csv")


    end_time = time.time()
    main_logger.info(f"  Finished zonal stats, took {round(end_time - prep_start_time)} seconds: {uu.timestr()}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create 10x10 deg per-ha and per-pixel output geotifs")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-id', '--input_date', required=True, help='Date of run, in YYYYMMDD')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-fv', '--first_variables_to_process', type=int, help='Number of variables to process from raw mega-zarr (for testing)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-ft', '--first_tiles_to_process', type=int, help='Number of tiles to process (for testing)')
    parser.add_argument('-mt', '--model_type', default='standard', help='Type of model run (e.g., standard).')
    parser.add_argument('-mpd', '--model_path_description', help='Description of model run (e.g., global, test, X_area).')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    input_date = args.input_date
    bounding_box = args.bounding_box
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_tiles_to_process = args.first_tiles_to_process
    first_variables_to_process = args.first_variables_to_process
    model_type = args.model_type
    model_path_description = args.model_path_description
    log_note = args.log_note

    no_log = args.no_log
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, input_date, model_type, no_log, no_upload, chunk_shapefile_uri, bounding_box=bounding_box,
         first_variables_to_process=first_variables_to_process,
         first_tiles_to_process=first_tiles_to_process, model_path_description=model_path_description, log_note=log_note)
