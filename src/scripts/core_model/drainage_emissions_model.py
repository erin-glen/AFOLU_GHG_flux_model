"""
drainage_emissions_model.py
Organic‑soils drainage and fire emissions model

* full decision‑tree logic (drainage & burned‑area) kept intact
* parallel execution via Coiled / Dask (mirrors LULUCF model)
* outputs saved to S3 using universal_utilities.save_and_upload_small_raster_set
"""

from __future__ import annotations
import argparse
import concurrent.futures
import os
import sys
from datetime import datetime

import dask.bag
import numpy as np
from numba import jit, types
from numba.typed import Dict

# project utilities
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu
from src.scripts.utilities import numba_utilities as nu

# ----------------------------------------------------------------------
# constants pulled into locals for Numba speed
# ----------------------------------------------------------------------
c_to_co2 = np.float32(cn.c_to_co2)
n2o_n_to_n2o = np.float32(cn.n2o_n_to_n2o)
gwp_ch4 = np.float32(cn.gwp_ch4)
gwp_n2o = np.float32(cn.gwp_n2o)
gwp_co = np.float32(cn.gwp_co)
combustion_factor = np.float32(cn.combustion_factor)

forest_code = cn.ipcc_codes["forest"]
cropland_code = cn.ipcc_codes["cropland"]
settlement_code = cn.ipcc_codes["settlement"]
wetland_code = cn.ipcc_codes["wetland"]
grassland_code = cn.ipcc_codes["grassland"]
otherland_code = cn.ipcc_codes["otherland"]

boreal_code = cn.ecozone_codes["boreal"]
temperate_code = cn.ecozone_codes["temperate"]
tropical_code = cn.ecozone_codes["tropical"]

poor_nutrient_code = cn.nutrient_status_codes["poor"]
rich_nutrient_code = cn.nutrient_status_codes["rich"]

long_rotation_code = cn.plantation_type_codes["long_rotation"]
short_rotation_code = cn.plantation_type_codes["short_rotation"]
oil_palm_code = cn.plantation_type_codes["oil_palm"]
sago_palm_code = cn.plantation_type_codes["sago_palm"]

# ----------------------------------------------------------------------
# full decision‑tree function (unchanged logic, ASCII comments only)
# ----------------------------------------------------------------------
@jit(nopython=True)
def calculate_drainage_and_emissions(
    in_dict_uint8, in_dict_int16, in_dict_float32
):
    """
    1) Drainage classification & state
    2) Drainage‑based emissions (CO2, N2O, CH4, off‑site CO2, total CO2e)
    3) Burned‑area emissions (CO2, CO, CH4, total CO2e) if burned layer present
    Returns two numba typed dicts: uint32 layers, float32 layers
    """

    out_dict_uint32 = Dict.empty(types.unicode_type, types.uint32[:, :])
    out_dict_float32 = Dict.empty(types.unicode_type, types.float32[:, :])

    # required inputs --------------------------------------------------
    peat_block = in_dict_uint8["peat"]
    land_cover_block = in_dict_uint8["land_cover"]
    planted_forest_type_block = in_dict_uint8["planted_forest_type"]
    dadap_block = in_dict_float32["dadap"]
    osm_roads_block = in_dict_float32["osm_roads"]
    osm_canals_block = in_dict_float32["osm_canals"]
    engert_block = in_dict_float32["engert"]
    grip_block = in_dict_float32["grip"]
    extraction_block = in_dict_uint8["extraction"]
    ecozone_block = in_dict_int16["climate_domain"]
    nutrient_block = in_dict_uint8["nutrient_status"]
    descals_type_block = in_dict_int16["descals_type"]

    # optional burned‑area mask
    burned_block = None
    for k in in_dict_uint8.keys():
        if k.startswith("burned_area_combined_"):
            burned_block = in_dict_uint8[k]
            break

    rows, cols = peat_block.shape
    pixel_area_block = in_dict_float32.get(
        "pixel_area_ha", np.ones((rows, cols), dtype=np.float32)
    )

    # output arrays ----------------------------------------------------
    soil_block = np.zeros((rows, cols), dtype=np.uint32)
    state_block = np.zeros((rows, cols), dtype=np.uint32)

    drained_co2_out = np.zeros((rows, cols), dtype=np.float32)
    drained_n2o_out = np.zeros((rows, cols), dtype=np.float32)
    drained_ch4_land_out = np.zeros((rows, cols), dtype=np.float32)
    drained_ch4_ditch_out = np.zeros((rows, cols), dtype=np.float32)
    drained_co2_offsite_out = np.zeros((rows, cols), dtype=np.float32)
    drained_total_co2e_out = np.zeros((rows, cols), dtype=np.float32)

    burned_state_out = np.zeros((rows, cols), dtype=np.uint32)
    burned_co2_out = np.zeros((rows, cols), dtype=np.float32)
    burned_co_out = np.zeros((rows, cols), dtype=np.float32)
    burned_ch4_out = np.zeros((rows, cols), dtype=np.float32)
    burned_total_co2e_out = np.zeros((rows, cols), dtype=np.float32)

    # main pixel loop --------------------------------------------------
    for row in range(rows):
        for col in range(cols):

            # pixel values
            pixel_area_ha = pixel_area_block[row, col]
            peat = peat_block[row, col]
            land_cover = land_cover_block[row, col]
            planted_forest_type = planted_forest_type_block[row, col]
            dadap = dadap_block[row, col]
            osm_roads = osm_roads_block[row, col]
            osm_canals = osm_canals_block[row, col]
            engert = engert_block[row, col]
            grip = grip_block[row, col]
            extraction = extraction_block[row, col]
            ecozone = ecozone_block[row, col]
            nutrient = nutrient_block[row, col]
            descals_type = descals_type_block[row, col]

            # initialize
            ef_co2 = np.float32(0.0)
            ef_n2o = np.float32(0.0)
            ef_ch4_land = np.float32(0.0)
            ef_ch4_ditch = np.float32(0.0)
            ef_co2_offsite = np.float32(0.0)
            frac_ditch = np.float32(0.0)
            node = 0
            drained = False

            # A) Drainage classification ----------------------------------
            if peat == 1:
                node = nu.accrete_node(node, 1)
                if dadap > 0 or osm_canals > 0:
                    node = nu.accrete_node(node, 1)
                    drained = True
                elif engert > 0 or grip > 0 or osm_roads > 0:
                    node = nu.accrete_node(node, 2)
                    drained = True
                elif land_cover in (cropland_code, settlement_code):
                    node = nu.accrete_node(node, 3)
                    drained = True
                elif planted_forest_type > 0 or descals_type > 0:
                    node = nu.accrete_node(node, 4)
                    drained = True
                elif extraction > 0:
                    node = nu.accrete_node(node, 5)
                    drained = True
                else:
                    node = nu.accrete_node(node, 6)

                soil_block[row, col] = 2 if drained else 1
            else:
                node = nu.accrete_node(node, 2)
                soil_block[row, col] = 0  # not peat

            state_block[row, col] = node

            # B) Drainage emission factors --------------------------------
            if soil_block[row, col] == 2:  # only if drained
                node = nu.accrete_node(node, 1)

                # BOREAL ---------------------------------------------------
                if ecozone == boreal_code:
                    node = nu.accrete_node(node, 1)
                    ef_co2_offsite = 0.12
                    if land_cover == forest_code:
                        node = nu.accrete_node(node, 1)
                        if nutrient == poor_nutrient_code:
                            node = nu.accrete_node(node, 1)
                            ef_co2 = 0.25
                            ef_n2o = 0.22
                            ef_ch4_land = 7.0
                            ef_ch4_ditch = 217.0
                            frac_ditch = 0.025
                        elif nutrient == rich_nutrient_code:
                            node = nu.accrete_node(node, 2)
                            ef_co2 = 0.95
                            ef_n2o = 3.2
                            ef_ch4_land = 2.0
                            ef_ch4_ditch = 217.0
                            frac_ditch = 0.025
                    elif land_cover == grassland_code:
                        node = nu.accrete_node(node, 2)
                        ef_co2 = 5.7
                        ef_n2o = 9.5
                        ef_ch4_land = 1.4
                        ef_ch4_ditch = 1165.0
                        frac_ditch = 0.05
                    elif land_cover == cropland_code:
                        node = nu.accrete_node(node, 3)
                        ef_co2 = 7.9
                        ef_n2o = 13.0
                        ef_ch4_land = 0.0
                        ef_ch4_ditch = 1165.0
                        frac_ditch = 0.05
                    elif extraction > 0:
                        node = nu.accrete_node(node, 4)
                        ef_co2 = 2.8
                        ef_n2o = 0.30
                        ef_ch4_land = 6.1
                        ef_ch4_ditch = 542.0
                        frac_ditch = 0.05

                # TEMPERATE -----------------------------------------------
                elif ecozone == temperate_code:
                    node = nu.accrete_node(node, 2)
                    ef_co2_offsite = 0.31
                    if land_cover == forest_code:
                        node = nu.accrete_node(node, 1)
                        ef_co2 = 2.6
                        ef_n2o = 2.8
                        ef_ch4_land = 2.5
                        ef_ch4_ditch = 217.0
                        frac_ditch = 0.05
                    elif land_cover == grassland_code:
                        node = nu.accrete_node(node, 2)
                        ef_ch4_ditch = 1165.0
                        if nutrient == poor_nutrient_code:
                            node = nu.accrete_node(node, 1)
                            ef_co2 = 5.3
                            ef_n2o = 4.3
                            ef_ch4_land = 1.8
                            frac_ditch = 0.05
                        elif nutrient == rich_nutrient_code:
                            node = nu.accrete_node(node, 2)
                            ef_co2 = 6.1
                            ef_n2o = 8.2
                            ef_ch4_land = 16.0
                            frac_ditch = 0.05
                    elif land_cover == cropland_code:
                        node = nu.accrete_node(node, 3)
                        ef_co2 = 10.5
                        ef_n2o = 13.0
                        ef_ch4_land = 0.0
                        ef_ch4_ditch = 1165.0
                        frac_ditch = 0.05
                    elif extraction > 0:
                        node = nu.accrete_node(node, 4)
                        ef_co2 = 3.0
                        ef_n2o = 0.3
                        ef_ch4_land = 6.1
                        ef_ch4_ditch = 542.0
                        frac_ditch = 0.05

                # TROPICAL -------------------------------------------------
                elif ecozone == tropical_code:
                    node = nu.accrete_node(node, 3)
                    ef_co2_offsite = 0.82
                    ef_ch4_ditch = 2259.0
                    frac_ditch = 0.02
                    if planted_forest_type > 0:
                        node = nu.accrete_node(node, 1)
                        if planted_forest_type == long_rotation_code:
                            ef_co2 = 15.0
                            ef_n2o = 2.4
                            ef_ch4_land = 2.7
                        elif planted_forest_type == short_rotation_code:
                            ef_co2 = 20.0
                            ef_n2o = 2.4
                            ef_ch4_land = 2.7
                        elif planted_forest_type == oil_palm_code:
                            ef_co2 = 11.0
                            ef_n2o = 1.2
                            ef_ch4_land = 0.0
                        elif planted_forest_type == sago_palm_code:
                            ef_co2 = 1.5
                            ef_n2o = 3.3
                            ef_ch4_land = 26.2
                    elif land_cover == forest_code:
                        node = nu.accrete_node(node, 2)
                        ef_co2 = 5.3
                        ef_n2o = 2.4
                        ef_ch4_land = 4.9
                    elif land_cover == grassland_code:
                        node = nu.accrete_node(node, 3)
                        ef_co2 = 9.6
                        ef_n2o = 5.0
                        ef_ch4_land = 7.0
                    elif land_cover == cropland_code:
                        node = nu.accrete_node(node, 4)
                        ef_co2 = 14.0
                        ef_n2o = 5.0
                        ef_ch4_land = 7.0
                    elif extraction > 0:
                        node = nu.accrete_node(node, 5)
                        ef_co2 = 2.0
                        ef_n2o = 0.0

                # calculate drainage emissions ---------------------------
                (
                    co2_em,
                    n2o_co2e,
                    ch4_land_co2e,
                    ch4_ditch_co2e,
                    co2_off,
                    total_co2e,
                ) = nu.calculate_drainage_emissions_co2e(
                    ef_co2,
                    ef_n2o,
                    ef_ch4_land,
                    ef_ch4_ditch,
                    ef_co2_offsite,
                    frac_ditch,
                    c_to_co2,
                    n2o_n_to_n2o,
                    gwp_n2o,
                    gwp_ch4,
                    pixel_area_ha,
                )

                drained_co2_out[row, col] = co2_em
                drained_n2o_out[row, col] = n2o_co2e
                drained_ch4_land_out[row, col] = ch4_land_co2e
                drained_ch4_ditch_out[row, col] = ch4_ditch_co2e
                drained_co2_offsite_out[row, col] = co2_off
                drained_total_co2e_out[row, col] = total_co2e

            # C) Burned‑area emissions -------------------------------------
            burned_node = 0
            if burned_block is not None:
                burned_val = burned_block[row, col]
                if burned_val > 0 and soil_block[row, col] in (1, 2):

                    if ecozone == boreal_code:
                        burned_node = nu.accrete_node(burned_node, 1)
                        if soil_block[row, col] == 2:
                            gef_co2, gef_co, gef_ch4 = 1650.0, 110.0, 12.0
                            mass_burnt = 250.0
                        else:
                            gef_co2, gef_co, gef_ch4 = 1450.0, 90.0, 10.0
                            mass_burnt = 75.0

                    elif ecozone == temperate_code:
                        burned_node = nu.accrete_node(burned_node, 2)
                        if soil_block[row, col] == 2:
                            gef_co2, gef_co, gef_ch4 = 1650.0, 110.0, 12.0
                            mass_burnt = 200.0
                        else:
                            gef_co2, gef_co, gef_ch4 = 1450.0, 90.0, 10.0
                            mass_burnt = 50.0

                    elif ecozone == tropical_code:
                        burned_node = nu.accrete_node(burned_node, 3)
                        if soil_block[row, col] == 2:
                            if (
                                land_cover in (cropland_code,)
                                or planted_forest_type > 0
                            ):
                                gef_co2, gef_co, gef_ch4 = 1700.0, 200.0, 15.0
                                mass_burnt = 150.0
                            else:
                                gef_co2, gef_co, gef_ch4 = 1600.0, 180.0, 14.0
                                mass_burnt = 300.0
                        else:
                            gef_co2 = gef_co = gef_ch4 = 0.0
                            mass_burnt = 0.0
                    else:
                        burned_node = nu.accrete_node(burned_node, 4)
                        gef_co2 = gef_co = gef_ch4 = 0.0
                        mass_burnt = 0.0

                    (
                        burn_co2,
                        burn_co,
                        burn_ch4,
                        burn_total_co2e,
                    ) = nu.calculate_burned_area_emissions(
                        np.float32(pixel_area_ha),
                        np.float32(mass_burnt),
                        combustion_factor,
                        np.float32(gef_co2),
                        np.float32(gef_co),
                        np.float32(gef_ch4),
                        gwp_co,
                        gwp_ch4,
                    )

                    burned_co2_out[row, col] = burn_co2
                    burned_co_out[row, col] = burn_co
                    burned_ch4_out[row, col] = burn_ch4
                    burned_total_co2e_out[row, col] = burn_total_co2e

            burned_state_out[row, col] = burned_node

    # pack outputs ----------------------------------------------------------
    out_dict_uint32["soil"] = soil_block
    out_dict_uint32["state"] = state_block
    out_dict_uint32["burned_state"] = burned_state_out

    out_dict_float32["drained_co2"] = drained_co2_out
    out_dict_float32["drained_n2o_co2e"] = drained_n2o_out
    out_dict_float32["drained_ch4_land_co2e"] = drained_ch4_land_out
    out_dict_float32["drained_ch4_ditch_co2e"] = drained_ch4_ditch_out
    out_dict_float32["drained_co2_offsite"] = drained_co2_offsite_out
    out_dict_float32["drained_total_co2e"] = drained_total_co2e_out
    out_dict_float32["burned_co2"] = burned_co2_out
    out_dict_float32["burned_co_co2e"] = burned_co_out
    out_dict_float32["burned_ch4_co2e"] = burned_ch4_out
    out_dict_float32["burned_total_co2e"] = burned_total_co2e_out

    return out_dict_uint32, out_dict_float32


# ----------------------------------------------------------------------
# helper: combine burned layers across interval
# ----------------------------------------------------------------------
def combine_burned_area(layers: dict, iv_start: int, iv_end: int):
    shape = None
    combined = None
    for yr in range(iv_start, iv_end + 1):
        key = f"{cn.burned_area_final_pattern}_{yr}"
        if key in layers:
            arr = layers[key]
            if shape is None:
                shape = arr.shape
                combined = np.zeros(shape, dtype=np.uint8)
            combined += (arr > 0).astype(np.uint8)
    if combined is not None:
        combined[combined > 0] = 1
        layers[f"burned_area_combined_{iv_start}_{iv_end}"] = combined


# ----------------------------------------------------------------------
# per‑chunk wrapper
# ----------------------------------------------------------------------
def calculate_and_upload_drainage(
    bounds,
    typed_dict,
    is_final,
    no_upload,
    iv_start,
    iv_end,
    use_actual_pixel_area=False,
):

    logger = lu.setup_logging_worker()
    bstr = uu.boundstr(bounds)
    tid = uu.xy_to_tile_id(bounds[0], bounds[3])
    chunk_px = uu.calc_chunk_length_pixels(bounds)
    chunk_stats = []

    # replace {tile_id} with actual
    import copy
    import re

    dwn = copy.deepcopy(typed_dict)
    for k, (uri, dt) in dwn.items():
        dwn[k] = (re.sub(cn.tile_id_pattern, tid, uri), dt)

    if not uu.check_for_tile(dwn, is_final, logger):
        return f"Skipped {bstr} (tile absent)", chunk_stats

    # ensure burned keys exist
    for yr in range(iv_start, iv_end + 1):
        dwn.setdefault(f"{cn.burned_area_final_pattern}_{yr}", (None, "Byte"))

    # download chunk windows in parallel
    futs = uu.queue_chunk_downloads(
        bounds, dwn, chunk_px, logger, is_final=is_final
    )
    layers = {}
    for fut in concurrent.futures.as_completed(futs):
        layers[futs[fut]] = fut.result()

    # fill missing layers with zeros
    uint8 = [
        "peat",
        "planted_forest_type",
        "extraction",
        "nutrient_status",
        "land_cover",
    ]
    uint8 += [
        f"{cn.burned_area_final_pattern}_{yr}"
        for yr in range(iv_start, iv_end + 1)
    ]
    int16 = ["climate_domain", "descals_type"]
    float32 = [
        "dadap",
        "osm_roads",
        "osm_canals",
        "engert",
        "grip",
        "pixel_area_ha",
    ]
    layers = uu.fill_missing_input_layers_with_no_data(
        layers, uint8, int16, [], float32, bstr, tid, is_final, logger
    )

    # stats for inputs
    for k, arr in layers.items():
        chunk_stats.append(uu.calculate_stats(arr, k, bstr, tid, "input"))

    combine_burned_area(layers, iv_start, iv_end)

    # create typed dicts for numba
    td8, td16, td32, td32f = nu.create_typed_dicts(layers)

    lu.print_and_log(
        f"starting drainage calc {tid} {bstr} {iv_start}-{iv_end}",
        is_final,
        logger,
    )
    out_u32, out_f32 = calculate_drainage_and_emissions(td8, td16, td32f)
    outputs = {**out_u32, **out_f32}

    # stats for outputs
    for k, arr in outputs.items():
        chunk_stats.append(uu.calculate_stats(arr, k, bstr, tid, "output"))

    # upload rasters
    if not no_upload:
        year_tag = f"{iv_start}" if iv_start == iv_end else f"{iv_start}_{iv_end}"
        interval_tag = "annual" if iv_start == iv_end else "five_years"
        for k, arr in outputs.items():
            outputs[k] = [arr, arr.dtype.name, k, year_tag]
        uu.save_and_upload_small_raster_set(
            bounds,
            chunk_px,
            tid,
            bstr,
            outputs,
            is_final,
            logger,
            interval_type=interval_tag,
            model_type="standard_model",
            no_data_val=0,
        )

    return f"Success {bstr} {iv_start}-{iv_end}", chunk_stats


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def run_drainage_model(
    cluster_name=None,
    bounding_box=None,
    chunk_size=None,
    run_local=False,
    no_stats=False,
    no_log=False,
    no_upload=False,
    start_year=None,
    end_year=None,
    interval_type="annual",
    use_actual_pixel_area=False,
):

    stage = "drainage_model"
    start_ts = uu.timestr()
    cluster, client = uu.connect_to_cluster(
        cluster_name=cluster_name, run_local=run_local
    )

    main_logger, _ = lu.populate_main_log_header(
        bounding_box=bounding_box,
        use_shapefile=False,
        client=client,
        cluster=cluster,
        log_note="Organic-soils drainage model",
        run_local=run_local,
        model_type="organic_soils",
        stage=stage,
    )

    bounding_box = bounding_box or [110, -10, 120, 0]
    chunk_size = chunk_size or 2
    chunks = uu.get_chunk_bounds(bounding_box, chunk_size)
    is_final = len(chunks) > 20

    start_year = start_year or 2020
    end_year = end_year or start_year
    if interval_type == "five_year":
        intervals = [
            (y, min(y + 4, end_year)) for y in range(start_year, end_year + 1, 5)
        ]
    else:
        intervals = [(y, y) for y in range(start_year, end_year + 1)]

    # figure out data types from first chunk
    sample_tid = uu.xy_to_tile_id(chunks[0][0], chunks[0][3])
    sample_dict = cn.get_dynamic_download_dict(sample_tid, start_year, end_year)
    if use_actual_pixel_area:
        sample_dict["pixel_area_ha"] = os.path.join(
            cn.pixel_area_ha_dir, f"{sample_tid}_pixel_area_ha.tif"
        )
    typed_dict = uu.add_file_type_to_dict(sample_dict)

    # build task list & run with dask.bag
    bag_items = [(bds, iv[0], iv[1]) for iv in intervals for bds in chunks]
    bag = dask.bag.from_sequence(bag_items, npartitions=len(bag_items))

    def _wrap(t):
        return calculate_and_upload_drainage(
            t[0],
            typed_dict,
            is_final,
            no_upload,
            t[1],
            t[2],
            use_actual_pixel_area,
        )

    results = bag.map(_wrap).compute()

    successes = sum("Success" in r[0] for r in results)
    skips = sum("Skipped" in r[0] for r in results)
    all_stats = [stat for _, lst in results for stat in lst]

    main_logger.info(f"successes={successes} | skips={skips}")
    if not no_stats:
        uu.calculate_chunk_stats(all_stats, stage)

    uu.stage_duration(start_ts, uu.timestr(), stage)
    if not run_local:
        client.close()
        cluster.close()


# ----------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------
def main(argv=None):
    """
    CLI entry point with an IDE‑friendly quick‑test.

    * If you pass arguments, they are honored (same flags as before).
    * If you run the script with no args (for example, click Run in an IDE),
      it spins up a tiny 1 x 1 degree chunk locally so you can debug fast.
    """
    argv = argv if argv is not None else sys.argv[1:]

    # quick test when no CLI args ------------------------------------
    if not argv:
        print("No CLI args: running 1‑tile local smoke test")
        run_drainage_model(
            cluster_name=None,
            bounding_box=[112, -2, 113, -1],  # 1 x 1 degree tile
            chunk_size=1,
            run_local=True,
            no_stats=False,
            no_log=False,
            no_upload=False,
            start_year=2015,
            end_year=2019,
            interval_type="five_year",
            use_actual_pixel_area=False,
        )
        return

    # normal CLI parsing --------------------------------------------
    p = argparse.ArgumentParser("Organic‑soils drainage model")
    p.add_argument("--cluster_name")
    p.add_argument(
        "--bounding_box",
        "-bb",
        nargs=4,
        type=float,
        metavar=("W", "S", "E", "N"),
    )
    p.add_argument("--chunk_size", "-cs", type=float)
    p.add_argument("--run_local", action="store_true")
    p.add_argument("--no_stats", action="store_true")
    p.add_argument("--no_log", action="store_true")
    p.add_argument("--no_upload", action="store_true")
    p.add_argument("--start_year", type=int, required=True)
    p.add_argument("--end_year", type=int, required=True)
    p.add_argument(
        "--interval_type", choices=["annual", "five_year"], default="annual"
    )
    p.add_argument("--use_actual_pixel_area", action="store_true")
    args = p.parse_args(argv)

    run_drainage_model(
        cluster_name=args.cluster_name,
        bounding_box=args.bounding_box,
        chunk_size=args.chunk_size,
        run_local=args.run_local,
        no_stats=args.no_stats,
        no_log=args.no_log,
        no_upload=args.no_upload,
        start_year=args.start_year,
        end_year=args.end_year,
        interval_type=args.interval_type,
        use_actual_pixel_area=args.use_actual_pixel_area,
    )


if __name__ == "__main__":
    main()
