"""
drainage_emissions_model.py
Organic‑soils drainage and fire emissions model

* full decision‑tree logic (drainage & burned‑area) kept intact
* parallel execution via Coiled / Dask (mirrors LULUCF model)
* outputs saved to S3 using universal_utilities.save_and_upload_small_raster_set
* optionally target specific 10x10 degree tiles via ``--tile_ids``
* or process the entire list of tiles via ``--full_model``
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
from src.scripts.utilities import drainage_emission_factors as defac
from src.scripts.utilities import burned_area_emission_factors as baf

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
unknown_ecozone_code = cn.ecozone_codes["unknown"]

poor_nutrient_code = cn.nutrient_status_codes["poor"]
rich_nutrient_code = cn.nutrient_status_codes["rich"]
unknown_nutrient_code = cn.nutrient_status_codes["unknown"]


long_rotation_code = cn.plantation_type_codes["long_rotation"]
short_rotation_code = cn.plantation_type_codes["short_rotation"]
oil_palm_code = cn.plantation_type_codes["oil_palm"]

# --- climate-domain → ecozone LUT (Numba-friendly; mirrors cn.climate_domain_remap)
_domain_max_key = max(cn.climate_domain_remap.keys()) if cn.climate_domain_remap else 0
domain_to_ecozone_lut = np.zeros(_domain_max_key + 1, dtype=np.uint8)
for _k, _v in cn.climate_domain_remap.items():
    domain_to_ecozone_lut[int(_k)] = np.uint8(_v)


# helpers for numba dict lookups
@jit(nopython=True)
def lookup_efs(key, table):
    """Return (values, is_missing) from drainage factor table."""
    if key in table:
        return table[key], False
    return defac.ZERO_ARRAY, True


@jit(nopython=True)
def lookup_befs(key, table):
    """Return (values, is_missing) from burned-area factor table."""
    if key in table:
        return table[key], False
    return baf.ZERO_ARRAY, True

# ----------------------------------------------------------------------
# full decision‑tree function (unchanged logic, ASCII comments only)
# ----------------------------------------------------------------------
@jit(nopython=True)
def calculate_drainage_and_emissions(
    in_dict_uint8,
    in_dict_int16,
    in_dict_float32,
    drainage_table,
    burned_table,
    mark_missing,
    count_burned_years,
    domain_to_ecozone_lut,
):
    """
    1) Drainage classification & state
    2) Drainage‑based emissions (CO2, N2O, CH4, off‑site CO2, total CO2e)
    3) Burned‑area emissions (CO2, CO, CH4, total CO2e) if burned layer present.
       When ``count_burned_years`` is ``True``, burned emissions are multiplied
       by the number of burned years for each pixel within the interval.
    Returns two numba typed dicts:
    - uint32 layers: categorical classification (drained_soil, drained_state,
      burned_state, burned_years_count)
    - float32 layers: emissions totals
    """

    out_dict_uint32 = Dict.empty(types.unicode_type, types.uint32[:, :])
    out_dict_float32 = Dict.empty(types.unicode_type, types.float32[:, :])

    # The maximum allowed digits for classification nodes.
    max_digits_state = 8

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
    continent_ecozone_block = in_dict_int16["continent_ecozone"]
    climate_zone_block = in_dict_uint8["climate_zone"]
    descals_type_block = in_dict_int16["descals_type"]

    # optional climate-domain block (prefer when present)
    climate_domain_block = None
    for k in in_dict_uint8.keys():
        if k == "climate_domain":
            climate_domain_block = in_dict_uint8[k]
            break

    # optional burned‑area mask
    burned_block = None
    for k in in_dict_uint8.keys():
        if k.startswith("burned_area_combined_"):
            burned_block = in_dict_uint8[k]
            break

    rows, cols = peat_block.shape

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
    burned_years_count_out = np.zeros((rows, cols), dtype=np.uint32)

    # main pixel loop --------------------------------------------------
    for row in range(rows):
        for col in range(cols):

            # pixel values
            peat = peat_block[row, col]
            land_cover = land_cover_block[row, col]
            planted_forest_type = planted_forest_type_block[row, col]
            dadap = dadap_block[row, col]
            osm_roads = osm_roads_block[row, col]
            osm_canals = osm_canals_block[row, col]
            engert = engert_block[row, col]
            grip = grip_block[row, col]
            extraction = extraction_block[row, col]
            continent_ecozone = continent_ecozone_block[row, col]
            climate_zone = climate_zone_block[row, col]

            # Determine simplified ecozone (prefer climate-domain remap; fallback to climate-zone)
            if climate_domain_block is not None:
                cd = int(climate_domain_block[row, col])
                if 0 <= cd < len(domain_to_ecozone_lut):
                    ecozone = domain_to_ecozone_lut[cd]
                else:
                    ecozone = unknown_ecozone_code
            else:
                if 1 <= climate_zone <= 4:
                    ecozone = tropical_code
                elif 5 <= climate_zone <= 8:
                    ecozone = temperate_code
                elif climate_zone >= 9:
                    ecozone = boreal_code
                else:
                    ecozone = unknown_ecozone_code
            descals_type = descals_type_block[row, col]

            # default nutrient status by ecozone
            if ecozone == boreal_code:
                nutrient = poor_nutrient_code
            elif ecozone == temperate_code:
                nutrient = rich_nutrient_code
            else:
                nutrient = unknown_nutrient_code

            # initialize
            ef_co2 = np.float32(0.0)
            ef_n2o = np.float32(0.0)
            ef_ch4_land = np.float32(0.0)
            ef_ch4_ditch = np.float32(0.0)
            ef_co2_offsite = np.float32(0.0)
            frac_ditch = np.float32(0.0)
            node = 0
            emission_node = 0
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


            # B) Drainage emission factors --------------------------------
            if soil_block[row, col] == 2:  # only if drained
                emission_node = nu.accrete_node(emission_node, 1)
                key = ""

                # BOREAL ---------------------------------------------------
                if ecozone == boreal_code:
                    emission_node = nu.accrete_node(emission_node, 1)
                    if land_cover == forest_code:
                        emission_node = nu.accrete_node(emission_node, 1)
                        if nutrient == poor_nutrient_code:
                            emission_node = nu.accrete_node(emission_node, 1)
                            key = "boreal_forest_poor"
                        elif nutrient == rich_nutrient_code:
                            emission_node = nu.accrete_node(emission_node, 2)
                            key = "boreal_forest_rich"
                    elif land_cover == grassland_code:
                        emission_node = nu.accrete_node(emission_node, 2)
                        key = "boreal_grassland"
                    elif land_cover == cropland_code:
                        emission_node = nu.accrete_node(emission_node, 3)
                        key = "boreal_cropland"
                    elif extraction > 0:
                        emission_node = nu.accrete_node(emission_node, 4)
                        key = "boreal_extraction"
                    elif land_cover == settlement_code:
                        emission_node = nu.accrete_node(emission_node, 5)
                        key = "boreal_settlement"
                    elif land_cover == wetland_code:
                        emission_node = nu.accrete_node(emission_node, 6)
                        key = "boreal_wetland"
                    else:
                        emission_node = nu.accrete_node(emission_node, 7)
                        key = "boreal_otherland"

                # TEMPERATE -----------------------------------------------
                elif ecozone == temperate_code:
                    emission_node = nu.accrete_node(emission_node, 2)
                    if land_cover == forest_code:
                        emission_node = nu.accrete_node(emission_node, 1)
                        key = "temperate_forest"
                    elif land_cover == grassland_code:
                        emission_node = nu.accrete_node(emission_node, 2)
                        if nutrient == poor_nutrient_code:
                            emission_node = nu.accrete_node(emission_node, 1)
                            key = "temperate_grassland_poor"
                        elif nutrient == rich_nutrient_code:
                            emission_node = nu.accrete_node(emission_node, 2)
                            key = "temperate_grassland_rich"
                    elif land_cover == cropland_code:
                        emission_node = nu.accrete_node(emission_node, 3)
                        key = "temperate_cropland"
                    elif extraction > 0:
                        emission_node = nu.accrete_node(emission_node, 4)
                        key = "temperate_extraction"
                    elif land_cover == settlement_code:
                        emission_node = nu.accrete_node(emission_node, 5)
                        key = "temperate_settlement"
                    elif land_cover == wetland_code:
                        emission_node = nu.accrete_node(emission_node, 6)
                        key = "temperate_wetland"
                    else:
                        emission_node = nu.accrete_node(emission_node, 7)
                        key = "temperate_otherland"

                # TROPICAL -------------------------------------------------
                elif ecozone == tropical_code:
                    emission_node = nu.accrete_node(emission_node, 3)
                    if planted_forest_type > 0:
                        emission_node = nu.accrete_node(emission_node, 1)
                        if planted_forest_type == long_rotation_code:
                            emission_node = nu.accrete_node(emission_node, 1)
                            key = "tropical_long_rotation"
                        elif planted_forest_type == short_rotation_code:
                            emission_node = nu.accrete_node(emission_node, 2)
                            key = "tropical_short_rotation"
                        elif planted_forest_type == oil_palm_code:
                            emission_node = nu.accrete_node(emission_node, 3)
                            key = "tropical_oil_palm"
                    elif land_cover == forest_code:
                        emission_node = nu.accrete_node(emission_node, 2)
                        key = "tropical_forest"
                    elif land_cover == grassland_code:
                        emission_node = nu.accrete_node(emission_node, 3)
                        key = "tropical_grassland"
                    elif land_cover == cropland_code:
                        emission_node = nu.accrete_node(emission_node, 4)
                        key = "tropical_cropland"
                    elif extraction > 0:
                        emission_node = nu.accrete_node(emission_node, 5)
                        key = "tropical_extraction"
                    elif land_cover == settlement_code:
                        emission_node = nu.accrete_node(emission_node, 8)
                        key = "tropical_settlement"
                    elif land_cover == wetland_code:
                        emission_node = nu.accrete_node(emission_node, 9)
                        key = "tropical_wetland"
                    else:
                        emission_node = nu.accrete_node(emission_node, 0)
                        key = "tropical_otherland"

                vals, missing = lookup_efs(key, drainage_table)
                ef_co2 = vals[0]
                ef_n2o = vals[1]
                ef_ch4_land = vals[2]
                ef_ch4_ditch = vals[3]
                ef_co2_offsite = vals[4]
                frac_ditch = vals[5]

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
                )

                drained_co2_out[row, col] = co2_em
                drained_n2o_out[row, col] = n2o_co2e
                drained_ch4_land_out[row, col] = ch4_land_co2e
                drained_ch4_ditch_out[row, col] = ch4_ditch_co2e
                drained_co2_offsite_out[row, col] = co2_off
                drained_total_co2e_out[row, col] = total_co2e

                # Optional sentinel for missing drainage EF
                if missing and mark_missing:
                    sentinel = 90 + int(ecozone)
                    emission_node = nu.accrete_node(
                        emission_node, sentinel
                    )

            if emission_node > 0:
                node = nu.accrete_node(node, emission_node)

            if node > (10 ** max_digits_state) - 1:
                raise ValueError("Maximum state digits exceeded")

            state_block[row, col] = nu.pad_to_6_digits(node, max_digits_state)

            # C) Burned‑area emissions -------------------------------------
            burned_node = 0
            burned_emission_node = 0
            if burned_block is not None:
                burned_val = burned_block[row, col]
                burned_years_count = (
                    burned_val if count_burned_years else (1 if burned_val > 0 else 0)
                )
                burned_years_count_out[row, col] = burned_years_count
                if burned_val > 0 and soil_block[row, col] in (1, 2):

                    if ecozone == boreal_code:
                        burned_node = nu.accrete_node(burned_node, 1)
                        burned_emission_node = nu.accrete_node(burned_emission_node, 1)
                        if soil_block[row, col] == 2:
                            bkey = "boreal_drained"
                            burned_emission_node = nu.accrete_node(burned_emission_node, 1)
                        else:
                            bkey = "boreal_undrained"
                            burned_emission_node = nu.accrete_node(burned_emission_node, 2)

                    elif ecozone == temperate_code:
                        burned_node = nu.accrete_node(burned_node, 2)
                        burned_emission_node = nu.accrete_node(burned_emission_node, 2)
                        if soil_block[row, col] == 2:
                            bkey = "temperate_drained"
                            burned_emission_node = nu.accrete_node(burned_emission_node, 1)
                        else:
                            bkey = "temperate_undrained"
                            burned_emission_node = nu.accrete_node(burned_emission_node, 2)

                    elif ecozone == tropical_code:
                        burned_node = nu.accrete_node(burned_node, 3)
                        burned_emission_node = nu.accrete_node(burned_emission_node, 3)
                        if soil_block[row, col] == 2:
                            if (
                                land_cover == cropland_code
                                or planted_forest_type > 0
                            ):
                                bkey = "tropical_drained_crop_or_plantation"
                                burned_emission_node = nu.accrete_node(burned_emission_node, 1)
                            else:
                                bkey = "tropical_drained_other"
                                burned_emission_node = nu.accrete_node(burned_emission_node, 2)
                        else:
                            bkey = "tropical_undrained"
                            burned_emission_node = nu.accrete_node(burned_emission_node, 3)
                    else:
                        burned_node = nu.accrete_node(burned_node, 4)
                        bkey = "other"
                        burned_emission_node = nu.accrete_node(burned_emission_node, 4)

                    bvals, bmissing = lookup_befs(bkey, burned_table)
                    gef_co2 = bvals[0]
                    gef_co = bvals[1]
                    gef_ch4 = bvals[2]
                    mass_burnt = bvals[3]

                    multiplier = burned_years_count
                    (
                        burn_co2,
                        burn_co,
                        burn_ch4,
                        burn_total_co2e,
                    ) = nu.calculate_burned_area_emissions(
                        np.float32(mass_burnt) * np.float32(multiplier),
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

                    # Optional sentinel for missing burned EFs
                    if bmissing and mark_missing:
                        if ecozone == boreal_code:
                            sentinel = 94
                        elif ecozone == temperate_code:
                            sentinel = 95
                        elif ecozone == tropical_code:
                            sentinel = 96
                        else:
                            sentinel = 97
                        burned_emission_node = nu.accrete_node(
                            burned_emission_node, sentinel
                        )

            if burned_emission_node > 0:
                burned_node = nu.accrete_node(burned_node, burned_emission_node)

            if burned_node > (10 ** max_digits_state) - 1:
                raise ValueError("Maximum burned state digits exceeded")

            burned_state_out[row, col] = nu.pad_to_6_digits(burned_node, max_digits_state)

    # pack outputs ----------------------------------------------------------
    out_dict_uint32["drained_soil"] = soil_block
    out_dict_uint32["drained_state"] = state_block
    out_dict_uint32["burned_state"] = burned_state_out
    out_dict_uint32["burned_years_count"] = burned_years_count_out

    out_dict_float32["drained_co2_Mg_CO2_ha_yr"] = drained_co2_out
    out_dict_float32["drained_n2o_Mg_CO2e_ha_yr"] = drained_n2o_out
    out_dict_float32["drained_ch4_land_Mg_CO2e_ha_yr"] = drained_ch4_land_out
    out_dict_float32["drained_ch4_ditch_Mg_CO2e_ha_yr"] = drained_ch4_ditch_out
    out_dict_float32["drained_co2_offsite_Mg_CO2_ha_yr"] = drained_co2_offsite_out
    out_dict_float32["drained_total_Mg_CO2e_ha_yr"] = drained_total_co2e_out
    out_dict_float32["burned_co2_Mg_CO2_ha"] = burned_co2_out
    out_dict_float32["burned_co_Mg_CO2e_ha"] = burned_co_out
    out_dict_float32["burned_ch4_Mg_CO2e_ha"] = burned_ch4_out
    out_dict_float32["burned_total_Mg_CO2e_ha"] = burned_total_co2e_out
    return out_dict_uint32, out_dict_float32


# ----------------------------------------------------------------------
# helper: combine burned layers across interval
# ----------------------------------------------------------------------
def combine_burned_area(
    layers: dict,
    iv_start: int,
    iv_end: int,
    count_burned_years: bool = False,
):
    """Combine annual burned area layers over an interval.

    Parameters
    ----------
    layers : dict
        Dictionary of layer arrays.
    iv_start, iv_end : int
        Interval start and end years.
    count_burned_years : bool, optional
        If ``True``, return the number of burned years per pixel. Otherwise a
        binary mask is returned.
    """
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
        if not count_burned_years:
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
    closing_year,
    peat_dataset="ogh",
    mark_missing=False,
    count_burned_years=False,
):
    """Process a single chunk for a given interval.

    Parameters
    ----------
    bounds : list[float]
        Bounding box for the chunk.
    typed_dict : dict
        Dictionary of input raster paths keyed by layer name.
    is_final : bool
        Whether the run covers the full domain.
    no_upload : bool
        If ``True``, skip uploading outputs to S3.
    iv_start, iv_end : int
        Interval start and end years.
    closing_year : int
        Land cover composite year for this interval.
    peat_dataset : str, optional
        Peat mask dataset name.
    mark_missing : bool, optional
        Append sentinel digits for missing emission factors if ``True``.
    count_burned_years : bool, optional
        If ``True``, count the number of burned years within the interval and
        multiply burned emissions accordingly.
    """

    logger = lu.setup_logging_worker()
    bstr = uu.boundstr(bounds)
    tid = uu.xy_to_tile_id(bounds[0], bounds[3])
    chunk_px = uu.calc_chunk_length_pixels(bounds)
    chunk_stats = []

    lu.print_and_log(
        f"Processing {bstr} {iv_start}-{iv_end} using land cover {closing_year}",
        is_final,
        logger,
    )

    # replace {tile_id} with actual
    import copy
    import re

    dwn = copy.deepcopy(typed_dict)
    for k, (uri, dt) in list(dwn.items()):
        dwn[k] = (re.sub(cn.tile_id_pattern, tid, uri), dt)

    # Drop burned area years outside the current interval
    for k in list(dwn.keys()):
        if k.startswith(cn.burned_area_final_pattern):
            try:
                yr = int(k.split("_")[-1])
            except ValueError:
                continue
            if yr < iv_start or yr > iv_end:
                dwn.pop(k)
    lu.print_and_log(
        f"Using burned area years {[k for k in dwn if k.startswith(cn.burned_area_final_pattern)]}",
        is_final,
        logger,
    )

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
        "land_cover",
        "climate_zone",
        "climate_domain",
    ]
    uint8 += [
        f"{cn.burned_area_final_pattern}_{yr}"
        for yr in range(iv_start, iv_end + 1)
    ]
    int16 = ["continent_ecozone", "descals_type"]
    float32 = [
        "dadap",
        "osm_roads",
        "osm_canals",
        "engert",
        "grip",
    ]
    layers = uu.fill_missing_input_layers_with_no_data(
        layers, uint8, int16, [], float32, bstr, tid, is_final, logger
    )

    # stats for inputs
    for k, arr in layers.items():
        chunk_stats.append(
            uu.calculate_stats(arr, k, bstr, tid, "input_layer", iv_start=iv_start, iv_end=iv_end)
        )

    combine_burned_area(layers, iv_start, iv_end, count_burned_years)

    # create typed dicts for numba
    td8, td16, td32, td32f = nu.create_typed_dicts(layers)

    lu.print_and_log(
        f"starting drainage calc {tid} {bstr} {iv_start}-{iv_end}",
        is_final,
        logger,
    )
    out_u32, out_f32 = calculate_drainage_and_emissions(
        td8,
        td16,
        td32f,
        defac.DEFAULT_TABLE,
        baf.DEFAULT_TABLE,
        mark_missing,
        count_burned_years,
        domain_to_ecozone_lut,
    )
    outputs = {**out_u32, **out_f32}

    # burned-area emissions are totals for the whole inventory period; convert
    # to annual values based on the number of years in the period
    interval_length = iv_end - iv_start + 1
    burned_layers = {
        "burned_co2_Mg_CO2_ha": "burned_co2_Mg_CO2_ha_yr",
        "burned_co_Mg_CO2e_ha": "burned_co_Mg_CO2e_ha_yr",
        "burned_ch4_Mg_CO2e_ha": "burned_ch4_Mg_CO2e_ha_yr",
        "burned_total_Mg_CO2e_ha": "burned_total_Mg_CO2e_ha_yr",
    }
    for old, new in burned_layers.items():
        if old in outputs:
            outputs[new] = outputs.pop(old) / interval_length

    # stats for outputs, with explicit layer categorization
    drainage_classification_layers = ["drained_soil", "drained_state"]
    burned_classification_layers = ["burned_state"]
    numeric_layers = ["burned_years_count"]

    pixel_area_uri = f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tid}.tif"
    pixel_area_chunk = uu.get_tile_dataset_rio(
        pixel_area_uri,
        "Float32",
        bounds,
        chunk_px,
        is_final,
        logger,
    )[0]

    for k, arr in outputs.items():
        if k in drainage_classification_layers or k in burned_classification_layers:
            chunk_stats.append(
                uu.calculate_stats(
                    arr,
                    k,
                    bstr,
                    tid,
                    "output_layer",
                    iv_start=iv_start,
                    iv_end=iv_end,
                )
            )
        elif k in numeric_layers:
            chunk_stats.append(
                uu.calculate_stats(
                    arr,
                    k,
                    bstr,
                    tid,
                    "output_layer_numeric",
                    iv_start=iv_start,
                    iv_end=iv_end,
                )
            )
        else:
            per_pixel = arr * pixel_area_chunk * cn.m2_to_ha
            chunk_stats.append(
                uu.calculate_stats(
                    arr,
                    k,
                    bstr,
                    tid,
                    "output_layer",
                    per_pixel,
                    iv_start,
                    iv_end,
                )
            )


    # upload rasters
    if not no_upload:
        if iv_start == iv_end:
            year_tag = f"{iv_start}"
        else:
            # Paths reflect the land cover year closing the period.
            end_for_path = iv_end
            year_tag = f"{iv_start}_{end_for_path}"
        interval_tag = (
            cn.intervals_annual
            if iv_start == iv_end
            else cn.intervals_five_year
        )
        for k, arr in outputs.items():
            outputs[k] = [arr, arr.dtype.name, k, year_tag]
        upload_tasks = uu.save_and_upload_small_raster_set(
            bounds,
            chunk_px,
            tid,
            bstr,
            outputs,
            is_final,
            logger,
            interval_type=interval_tag,
            model_type=f"{peat_dataset}_standard_model",
            no_data_val=0,
        )
        lu.print_and_log(
            f"Upload tasks created for {bstr} in {tid}. Uploading now: {uu.timestr()}",
            False,
            logger,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            ex.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks)
        lu.print_and_log(
            f"Uploads completed for {bstr} in {tid} using {cn.outputs_path}: {uu.timestr()}",
            is_final,
            logger,
        )

    return f"Success {bstr} {iv_start}-{iv_end}", chunk_stats


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------


def compute_intervals(start_year, end_year, interval_type, all_five_year_periods):
    """Return list of interval tuples and normalized settings."""
    if all_five_year_periods:
        interval_type = cn.intervals_five_year
        start_year = cn.five_year_inventory_periods[0][0]
        end_year = cn.five_year_inventory_periods[-1][1]
    else:
        start_year = start_year or cn.annual_land_cover_start_year
        if end_year is None:
            if interval_type == cn.intervals_five_year:
                end_year = start_year + 4
            else:
                end_year = start_year
        end_year = min(end_year, cn.five_year_inventory_periods[-1][1])
        if (
            interval_type == cn.intervals_annual
            and start_year < cn.annual_land_cover_start_year
        ):
            start_year = cn.annual_land_cover_start_year
        if end_year < start_year:
            end_year = start_year

    if interval_type == cn.intervals_five_year:
        # Snap a 2019 start year to the final interval (2020–2024)
        if start_year == 2019 and end_year >= 2024:
            start_year = 2020
        intervals = [
            (y, min(y + 4, end_year)) for y in range(start_year, end_year + 1, 5)
        ]
    else:
        intervals = [(y, y) for y in range(start_year, end_year + 1)]

    return intervals, start_year, end_year, interval_type


def run_drainage_model(
    cluster_name=None,
    bounding_box=None,
    chunk_size=None,
    chunk_shapefile_uri=None,
    first_chunks=None,
    run_local=False,
    no_stats=False,
    no_log=False,
    no_upload=False,
    start_year=None,
    end_year=None,
    all_five_year_periods=False,
    interval_type="annual",
    tile_ids=None,
    peat_dataset="ogh",
    mark_missing=False,
    count_burned_years=False,
):

    stage = "drainage_model"
    start_ts = uu.timestr()
    cluster, client, run_local = uu.connect_to_cluster(
        cluster_name=cluster_name, run_local=run_local
    )

    shapefile_provided = bool(chunk_shapefile_uri)
    chunk_shapefile_uri = chunk_shapefile_uri or cn.fishnet_1x1deg_uri

    main_logger, _ = lu.populate_main_log_header(
        bounding_box=None if tile_ids else bounding_box,
        use_shapefile=chunk_shapefile_uri if chunk_shapefile_uri else False,
        client=client,
        cluster=cluster,
        log_note="Organic-soils drainage model",
        run_local=run_local,
        model_type="organic_soils",
        stage=stage,
    )

    if chunk_shapefile_uri:
        if shapefile_provided:
            main_logger.info(
                f"Using user-supplied chunk shapefile: {chunk_shapefile_uri}"
            )
        else:
            main_logger.info(
                f"Using default chunk shapefile: {chunk_shapefile_uri}"
            )

    chunk_size = chunk_size or 1
    if tile_ids:
        chunks = []
        for tid in tile_ids:
            bds = uu.get_10x10_tile_bounds(tid)
            chunks.extend(uu.get_chunk_bounds(bds, chunk_size))
    elif chunk_shapefile_uri:
        fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)
        chunks, _ = uu.create_chunk_list(
            bounding_box,
            chunk_shapefile_uri,
            chunk_size,
            first_chunks,
            fishnet_iso_df,
            main_logger,
        )
    else:
        bounding_box = bounding_box or [110, -10, 120, 0]
        chunks = uu.get_chunk_bounds(bounding_box, chunk_size)
    is_final = len(chunks) > 20

    # Normalize interval settings and compute interval list
    intervals, start_year, end_year, interval_type = compute_intervals(
        start_year,
        end_year,
        interval_type,
        all_five_year_periods,
    )

    # build task list & run with dask.bag
    bag_items = [(bds, iv[0], iv[1]) for iv in intervals for bds in chunks]
    bag = dask.bag.from_sequence(bag_items, npartitions=len(bag_items))

    typed_dict_cache = {}

    def _wrap(t):
        final_year = cn.five_year_inventory_periods[-1][1]
        closing_year = t[2]
        bstr = uu.boundstr(t[0])
        main_logger.info(
            f"{bstr} interval {t[1]}-{t[2]} uses land cover {closing_year}"
        )

        key = (t[1], closing_year)
        if key not in typed_dict_cache:
            download_dict = cn.get_dynamic_download_dict(
                cn.sample_tile_id,
                t[1],
                closing_year,
                peat_dataset=peat_dataset,
            )
            first_tiles = uu.first_file_name_in_s3_folder(download_dict)
            typed_dict_cache[key] = uu.add_file_type_to_dict(first_tiles)

        typed_dict = typed_dict_cache[key]

        return calculate_and_upload_drainage(
            t[0],
            typed_dict,
            is_final,
            no_upload,
            t[1],
            t[2],
            closing_year,
            peat_dataset,
            mark_missing,
            count_burned_years,
        )

    results = bag.map(_wrap).compute()

    # Summarize chunk results and gather per-chunk statistics
    success_count, all_stats = uu.count_successful_chunks(
        bag_items, is_final, main_logger, results
    )

    # Aggregate per‑chunk statistics and merge with the fishnet shapefile
    if (not no_stats) and (success_count > 0):
        uu.compile_1x1_chunk_stats(
            all_stats, chunk_shapefile_uri, stage, no_upload, main_logger
        )


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
            chunk_shapefile_uri=None,
            first_chunks=None,
            run_local=True,
            no_stats=False,
            no_log=False,
            no_upload=False,
            start_year=2015,
            end_year=2020,
            interval_type=cn.intervals_five_year,
            all_five_year_periods=False,
            peat_dataset="ogh",
            mark_missing=False,
            count_burned_years=False,
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
    p.add_argument(
        "--tile_ids",
        action="append",
        help="Comma separated 10x10 tile IDs (e.g. 00N_110E). Can be used multiple times.",
    )
    p.add_argument(
        "--full_model",
        action="store_true",
        help="Process all available 10x10 degree tiles",
    )
    p.add_argument("--chunk_size", "-cs", type=float)
    p.add_argument(
        "-cshp",
        "--chunk_shapefile_uri",
        help="S3 location for shapefile of 1x1 deg chunk footprints",
    )
    p.add_argument(
        "-f",
        "--first_chunks",
        type=int,
        help="Number of chunks to process from shapefile",
    )
    p.add_argument("--run_local", action="store_true")
    p.add_argument("--no_stats", action="store_true")
    p.add_argument("--no_log", action="store_true")
    p.add_argument("--no_upload", action="store_true")
    p.add_argument("--start_year", type=int)
    p.add_argument("--end_year", type=int)
    p.add_argument(
        "--interval_type",
        choices=[cn.intervals_annual, cn.intervals_five_year],
        default=cn.intervals_annual,
    )
    p.add_argument(
        "--all_five_year_periods",
        action="store_true",
        help=(
            "Process all available five year intervals "
            "(sets interval_type to five_year)"
        ),
    )
    p.add_argument(
        "--peat_dataset",
        default="ogh",
        choices=cn.peat_dataset_choices,
        help="Peat mask dataset to use",
    )
    p.add_argument(
        "--mark_missing_factors",
        action="store_true",
        help=(
            "Append sentinel digits (91-97) to state codes when EF "
            "look-ups are missing; emissions remain zero."
        ),
    )
    p.add_argument(
        "--count_burned_years",
        action="store_true",
        help="Multiply burned emissions by the number of burned years in each interval",
    )
    args = p.parse_args(argv)

    tile_ids = []
    if args.full_model:
        tile_ids = list(cn.tile_id_list)
    elif args.tile_ids:
        for item in args.tile_ids:
            tile_ids.extend(t.strip() for t in item.split(",") if t.strip())

    run_drainage_model(
        cluster_name=args.cluster_name,
        bounding_box=args.bounding_box,
        chunk_size=args.chunk_size,
        chunk_shapefile_uri=args.chunk_shapefile_uri,
        first_chunks=args.first_chunks,
        run_local=args.run_local,
        no_stats=args.no_stats,
        no_log=args.no_log,
        no_upload=args.no_upload,
        start_year=args.start_year,
        end_year=args.end_year,
        interval_type=args.interval_type,
        all_five_year_periods=args.all_five_year_periods,
        tile_ids=tile_ids,
        peat_dataset=args.peat_dataset,
        mark_missing=args.mark_missing_factors,
        count_burned_years=args.count_burned_years,
    )


if __name__ == "__main__":
    main()

"""
python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster \
  --bounding_box 110 -10 120 0 \
  --chunk_size 1 \
  --start_year 2016 \
  --end_year 2020 \
  --interval_type five_year

python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster \
  --chunk_shapefile_uri s3://path/to/fishnet.shp \
  --first_chunks 10 \
  --start_year 2016 \
  --end_year 2020 \
  --interval_type five_year

python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster \
  --tile_ids 00N_110E,00N_120E \
  --chunk_size 1 \
  --start_year 2016 \
  --end_year 2020 \
  --interval_type five_year \
  --peat_dataset peatmap

python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster \
  --tile_ids 00N_110E,10N_020E,20N_020W,60N_010W,60N_110W \
  --chunk_size 1 \
  --start_year 2001 \
  --end_year 2024 \
  --interval_type five_year \
  --mark_missing_factors
  
python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster \
  --bounding_box 110 -10 120 0 \
  --chunk_size 1 \
  --start_year 2001 \
  --end_year 2024 \
  --all_five_year_periods \
  --count_burned_years

python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster \
  --tile_ids 00N_080W,00N_070W,00N_110E,10N_110E,10N_010E,10N_020E,20N_020W,60N_010W,60N_110W,70N_050E \
  --chunk_size 1 \
  --start_year 2001 \
  --end_year 2024 \
  --all_five_year_periods \
  --mark_missing_factors \
  --count_burned_years

python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster \
  --full_model \
  --chunk_size 1 \
  --start_year 2001 \
  --end_year 2024 \
  --all_five_year_periods \
  --mark_missing_factors \
  --count_burned_years
"""