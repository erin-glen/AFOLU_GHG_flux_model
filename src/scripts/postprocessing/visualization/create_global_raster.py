"""
Stage 01: aggregate 10×10° tiles into global rasters at a coarser resolution.

What's new in this drop-in update
---------------------------------
- Streamed global mosaic build using a disk-backed memmap (no N×10 GB RAM spike).
- As-completed iteration over Dask futures; no giant gather into a Python list.
- **FIX 1 (GeoTransform)**: Use rasterio.transform.from_bounds (rows & cols) so pixel size Y == X.
- **FIX 2 (S3 writing)**: When writing to s3:// paths, enable GDAL spooling:
    CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE=YES and CPL_TMPDIR=<our working tmpdir>.
- **Typed reads**: Choose "Int32" for integer datasets (combined_state), "Float32" otherwise.
- **Local batch knob**: AGG_LOCAL_BATCH env var controls local batch size (default 8) for iterate_tiles.

Assumptions (this version)
--------------------------
- All non-integer inputs are **per-pixel totals** (e.g., Mg yr^-1 per native pixel).
- Aggregation for float datasets is **SUM** to the target grid.
- For the bit-packed `combined_state` (uint32): the backward-compatible
  `combined_state` output remains a component-wise modal state. In addition,
  the canonical coarse state summary is written as class fractions computed
  from native pixels before any modal collapse.
- Whenever `combined_state` is aggregated, also write a UInt8
  `combined_state_reclassified` companion raster from the modal state:
  1=undrained organic soil, 2=drained only, 3=burned only, 4=drained+burned,
  255=nodata/non-organic.
- Whenever `combined_state` is aggregated, also write:
  `combined_state_class_fraction`: 4-band Float32 native-pixel fractions with
  bands [undrained, drained only, burned only, drained+burned].
  `combined_state_presence_reclassified`: UInt8 display classes derived from
  any nonzero class fraction, so minority drained/burned pixels remain visible
  at coarse resolution.
- No unit conversions are performed; no input tiles are modified or overwritten.

Examples
--------
# Aggregate to 0.01° on a running Dask cluster
python -m src.scripts.postprocessing.visualization.create_global_raster \
  -cn create_maps --run_name ogh_sensitivity_500m_23 \
  --model_version 0_9_7 --date_tag 20251118 --target_deg 0.01 --native_deg 0.00025

# Aggregate at 0.01° using a local Dask scheduler (smaller local batch by default)
AGG_LOCAL_BATCH=8 \
python -m src.scripts.postprocessing.visualization.create_global_raster -cn drainage_cluster --run_name ogh_biome_thresholds --model_version 0_1_4 --date_tag 20260417 --target_deg 0.01 --native_deg 0.00025
"""

from __future__ import annotations

import argparse
import os
import posixpath
import tempfile
from typing import Iterator, List, Optional, Tuple

import dask
from dask.distributed import as_completed
import numpy as np

# Write with a correct transform derived from rows & cols
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import log_utilities as lu
from src.scripts.utilities import universal_utilities as uu
from src.scripts.zonal_statistics import zonal_constants as zc

from src.scripts.postprocessing.visualization.create_global_map_common import (
    DEFAULT_DATE_TAG,
    DEFAULT_MODEL_VERSION,
    DEFAULT_NATIVE_DEG,
    DEFAULT_TARGET_DEG,
    INTEGER_DATASETS,
    OUTPUT_ROOT,
    assert_grid_divides_world,
    build_download_upload_dict,
    deg_to_label,
    resolve_versioned_paths,
)

# --------------------------------------------------------------------
# Constants / helpers
# --------------------------------------------------------------------

COMBINED_STATE_RECLASS_DATASET = "combined_state_reclassified"
COMBINED_STATE_CLASS_FRACTION_DATASET = "combined_state_class_fraction"
COMBINED_STATE_PRESENCE_RECLASS_DATASET = "combined_state_presence_reclassified"
COMBINED_STATE_RECLASS_NODATA = np.uint8(255)
COMBINED_STATE_RECLASS_UNDRAINED = np.uint8(1)
COMBINED_STATE_RECLASS_DRAINED_ONLY = np.uint8(2)
COMBINED_STATE_RECLASS_BURNED_ONLY = np.uint8(3)
COMBINED_STATE_RECLASS_DRAINED_BURNED = np.uint8(4)

COMBINED_STATE_CLASS_BAND_DESCRIPTIONS = (
    "undrained organic soil fraction",
    "drained only organic soil fraction",
    "burned only organic soil fraction",
    "drained+burned organic soil fraction",
)

_DRAINED_STATE_ID_LUT_SIZE = 1 << zc.COMBINED_STATE_DRAINED_BITS

COMBINED_STATE_RECLASS_TAGS = {
    "class_1": "undrained organic soil",
    "class_2": "drained only",
    "class_3": "burned only",
    "class_4": "drained+burned",
    "nodata": str(int(COMBINED_STATE_RECLASS_NODATA)),
}

COMBINED_STATE_CLASS_FRACTION_TAGS = {
    "band_1": COMBINED_STATE_CLASS_BAND_DESCRIPTIONS[0],
    "band_2": COMBINED_STATE_CLASS_BAND_DESCRIPTIONS[1],
    "band_3": COMBINED_STATE_CLASS_BAND_DESCRIPTIONS[2],
    "band_4": COMBINED_STATE_CLASS_BAND_DESCRIPTIONS[3],
    "units": "fraction of native pixels in coarse pixel",
}

COMBINED_STATE_PRESENCE_RECLASS_TAGS = {
    **COMBINED_STATE_RECLASS_TAGS,
    "aggregation": "presence from combined_state_class_fraction",
}


def _split_cli_items(values: Optional[List[str]]) -> Optional[List[str]]:
    """Normalize repeated, comma-delimited, or space-delimited CLI values."""
    if not values:
        return None

    items: List[str] = []
    for value in values:
        items.extend(
            item.strip()
            for item in str(value).replace(",", " ").split()
            if item.strip()
        )
    return list(dict.fromkeys(items)) or None


def _reaggregate_sum(arr: np.ndarray, native_deg: float, target_deg: float) -> np.ndarray:
    """
    Downsample by summing factor×factor blocks (preserves NaNs).
    If a block has 0 valid cells, output is NaN.
    """
    factor_f = target_deg / native_deg
    if not np.isclose(round(factor_f), factor_f):
        raise ValueError(f"target_deg/native_deg must be an integer; got {target_deg}/{native_deg}.")
    f = int(round(factor_f))

    h, w = arr.shape
    H = (h // f) * f
    W = (w // f) * f

    a = arr[:H, :W]
    a4 = a.reshape(H // f, f, W // f, f)

    valid = np.sum(~np.isnan(a4), axis=(1, 3)).astype(np.int32)
    block_sum = np.nansum(a4, axis=(1, 3)).astype(np.float32)
    block_sum[valid == 0] = np.nan
    return block_sum


def _mode_per_component(arr: np.ndarray, native_deg: float, target_deg: float) -> np.ndarray:
    """
    Aggregate a bit-packed ``combined_state`` uint32 tile by taking the mode of
    each component (drained_id and burned_id) independently within each block,
    then repacking. The has-drained / has-burned bits are set wherever the
    aggregated component is nonzero.
    """
    c = np.asarray(arr, dtype=np.uint32)
    drained_id = (c & np.uint32(zc.COMBINED_STATE_DRAINED_MASK)).astype(np.uint8)
    burned_id = (
        (c >> np.uint32(zc.COMBINED_STATE_BURNED_SHIFT))
        & np.uint32(zc.COMBINED_STATE_BURNED_MASK)
    ).astype(np.uint8)

    drained_mode = uu.reaggregate_mode(drained_id, native_deg, target_deg)
    burned_mode = uu.reaggregate_mode(burned_id, native_deg, target_deg)

    out = drained_mode.astype(np.uint32)
    out |= burned_mode.astype(np.uint32) << np.uint32(zc.COMBINED_STATE_BURNED_SHIFT)
    out |= (drained_mode > 0).astype(np.uint32) << np.uint32(zc.COMBINED_STATE_HAS_DRAINED_BIT)
    out |= (burned_mode > 0).astype(np.uint32) << np.uint32(zc.COMBINED_STATE_HAS_BURNED_BIT)
    return out


def _aggregation_factor(native_deg: float, target_deg: float) -> int:
    factor_f = target_deg / native_deg
    if not np.isclose(round(factor_f), factor_f):
        raise ValueError(f"target_deg/native_deg must be an integer; got {target_deg}/{native_deg}.")
    return int(round(factor_f))


def _mode_uint8_blocks(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-mode aggregate for small uint8 category rasters."""
    rows, cols = arr.shape
    new_rows = rows // factor
    new_cols = cols // factor
    trimmed = arr[: new_rows * factor, : new_cols * factor]
    blocks = trimmed.reshape(new_rows, factor, new_cols, factor)

    best_vals = np.zeros((new_rows, new_cols), dtype=np.uint8)
    best_counts = np.full((new_rows, new_cols), -1, dtype=np.int32)
    for val in np.unique(trimmed):
        counts = np.sum(blocks == val, axis=(1, 3), dtype=np.int32)
        update = counts > best_counts
        best_vals[update] = val
        best_counts[update] = counts[update]
    return best_vals


def _sum_uint8_blocks(arr: np.ndarray, factor: int) -> np.ndarray:
    """Return block sums for uint8/bool masks."""
    rows, cols = arr.shape
    new_rows = rows // factor
    new_cols = cols // factor
    trimmed = arr[: new_rows * factor, : new_cols * factor]
    blocks = trimmed.reshape(new_rows, factor, new_cols, factor)
    return np.sum(blocks, axis=(1, 3), dtype=np.uint32)


def _mode_per_component_windowed(
    path: str,
    chunk_length_pixels: int,
    native_deg: float,
    target_deg: float,
    logger,
) -> np.ndarray:
    factor = _aggregation_factor(native_deg, target_deg)
    out_rows = chunk_length_pixels // factor
    out_cols = chunk_length_pixels // factor
    output = np.zeros((out_rows, out_cols), dtype=np.uint32)
    output_rows_per_read = int(os.environ.get("AGG_TILE_OUTPUT_ROWS", "100"))
    input_rows_per_read = max(factor, output_rows_per_read * factor)

    with rasterio.Env():
        with rasterio.open(path) as src:
            for row0 in range(0, chunk_length_pixels, input_rows_per_read):
                rows = min(input_rows_per_read, chunk_length_pixels - row0)
                rows -= rows % factor
                if rows <= 0:
                    continue
                c = src.read(
                    1,
                    window=Window(0, row0, chunk_length_pixels, rows),
                    out_dtype="uint32",
                )
                drained_id = (c & np.uint32(zc.COMBINED_STATE_DRAINED_MASK)).astype(np.uint8)
                burned_id = (
                    (c >> np.uint32(zc.COMBINED_STATE_BURNED_SHIFT))
                    & np.uint32(zc.COMBINED_STATE_BURNED_MASK)
                ).astype(np.uint8)
                drained_mode = _mode_uint8_blocks(drained_id, factor)
                burned_mode = _mode_uint8_blocks(burned_id, factor)

                packed = drained_mode.astype(np.uint32)
                packed |= burned_mode.astype(np.uint32) << np.uint32(zc.COMBINED_STATE_BURNED_SHIFT)
                packed |= (drained_mode > 0).astype(np.uint32) << np.uint32(zc.COMBINED_STATE_HAS_DRAINED_BIT)
                packed |= (burned_mode > 0).astype(np.uint32) << np.uint32(zc.COMBINED_STATE_HAS_BURNED_BIT)

                out0 = row0 // factor
                out1 = out0 + packed.shape[0]
                output[out0:out1, :] = packed
                logger.debug("Aggregated combined_state rows %d-%d", row0, row0 + rows)
    return output


def _reaggregate_sum_windowed(
    path: str,
    chunk_length_pixels: int,
    native_deg: float,
    target_deg: float,
    logger,
) -> np.ndarray:
    factor = _aggregation_factor(native_deg, target_deg)
    out_rows = chunk_length_pixels // factor
    out_cols = chunk_length_pixels // factor
    output = np.empty((out_rows, out_cols), dtype=np.float32)
    output_rows_per_read = int(os.environ.get("AGG_TILE_OUTPUT_ROWS", "100"))
    input_rows_per_read = max(factor, output_rows_per_read * factor)

    with rasterio.Env():
        with rasterio.open(path) as src:
            for row0 in range(0, chunk_length_pixels, input_rows_per_read):
                rows = min(input_rows_per_read, chunk_length_pixels - row0)
                rows -= rows % factor
                if rows <= 0:
                    continue
                arr = src.read(
                    1,
                    window=Window(0, row0, chunk_length_pixels, rows),
                    out_dtype="float32",
                )
                agg = _reaggregate_sum(arr, native_deg, target_deg)
                out0 = row0 // factor
                out1 = out0 + agg.shape[0]
                output[out0:out1, :] = agg
                logger.debug("Aggregated continuous rows %d-%d", row0, row0 + rows)
    return output


def _combined_state_class_fractions_windowed(
    path: str,
    chunk_length_pixels: int,
    native_deg: float,
    target_deg: float,
    logger,
) -> np.ndarray:
    """Aggregate combined_state to class-fraction bands without modal collapse."""
    factor = _aggregation_factor(native_deg, target_deg)
    out_rows = chunk_length_pixels // factor
    out_cols = chunk_length_pixels // factor
    output = np.zeros(
        (len(COMBINED_STATE_CLASS_BAND_DESCRIPTIONS), out_rows, out_cols),
        dtype=np.float32,
    )
    output_rows_per_read = int(os.environ.get("AGG_TILE_OUTPUT_ROWS", "100"))
    input_rows_per_read = max(factor, output_rows_per_read * factor)

    with rasterio.Env():
        with rasterio.open(path) as src:
            for row0 in range(0, chunk_length_pixels, input_rows_per_read):
                rows = min(input_rows_per_read, chunk_length_pixels - row0)
                rows -= rows % factor
                if rows <= 0:
                    continue
                arr = src.read(
                    1,
                    window=Window(0, row0, chunk_length_pixels, rows),
                    out_dtype="uint32",
                )
                fractions = _aggregate_combined_state_class_fractions(
                    arr,
                    native_deg,
                    target_deg,
                )
                out0 = row0 // factor
                out1 = out0 + fractions.shape[1]
                output[:, out0:out1, :] = fractions
                logger.debug(
                    "Aggregated combined_state class fractions rows %d-%d",
                    row0,
                    row0 + rows,
                )
    return output


def _build_drained_state_luts() -> Tuple[np.ndarray, np.ndarray]:
    """Return LUTs identifying organic-soil and drained-state ids."""
    organic = np.zeros(_DRAINED_STATE_ID_LUT_SIZE, dtype=bool)
    drained = np.zeros(_DRAINED_STATE_ID_LUT_SIZE, dtype=bool)
    for idx, code in zc.DRAINED_STATE_ID_TO_CODE.items():
        if not 0 <= idx < _DRAINED_STATE_ID_LUT_SIZE:
            continue
        if code.startswith("16"):
            organic[idx] = True
        elif code.startswith(("11", "12", "13", "14", "15")):
            organic[idx] = True
            drained[idx] = True
    return organic, drained


_ORGANIC_BY_DRAINED_ID, _DRAINED_BY_DRAINED_ID = _build_drained_state_luts()


def _combined_state_component_masks(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(organic, drained, burned)`` masks decoded from packed state."""
    packed = np.asarray(arr, dtype=np.uint32)
    drained_id = (packed & np.uint32(zc.COMBINED_STATE_DRAINED_MASK)).astype(np.uint16)
    burned_id = (
        (packed >> np.uint32(zc.COMBINED_STATE_BURNED_SHIFT))
        & np.uint32(zc.COMBINED_STATE_BURNED_MASK)
    ).astype(np.uint16)

    valid_drained_id = drained_id < np.uint16(_DRAINED_STATE_ID_LUT_SIZE)
    organic = np.zeros(packed.shape, dtype=bool)
    drained = np.zeros(packed.shape, dtype=bool)
    organic[valid_drained_id] = _ORGANIC_BY_DRAINED_ID[drained_id[valid_drained_id]]
    drained[valid_drained_id] = _DRAINED_BY_DRAINED_ID[drained_id[valid_drained_id]]

    burned = burned_id > 0
    organic |= burned
    return organic, drained, burned


def _combined_state_class_masks(arr: np.ndarray) -> np.ndarray:
    """
    Decode packed combined_state into four mutually exclusive class masks.

    Band order is:
        0: undrained organic soil, not burned
        1: drained only
        2: burned only
        3: drained + burned
    """
    organic, drained, burned = _combined_state_component_masks(arr)
    masks = np.empty((4, *organic.shape), dtype=np.uint8)
    masks[0] = organic & ~drained & ~burned
    masks[1] = organic & drained & ~burned
    masks[2] = organic & ~drained & burned
    masks[3] = organic & drained & burned
    return masks


def _aggregate_combined_state_class_fractions(
    arr: np.ndarray,
    native_deg: float,
    target_deg: float,
) -> np.ndarray:
    """Aggregate native combined_state to four class-fraction bands."""
    factor = _aggregation_factor(native_deg, target_deg)
    denominator = np.float32(factor * factor)
    masks = _combined_state_class_masks(arr)
    out_rows = arr.shape[0] // factor
    out_cols = arr.shape[1] // factor
    fractions = np.empty((masks.shape[0], out_rows, out_cols), dtype=np.float32)
    for idx in range(masks.shape[0]):
        fractions[idx] = _sum_uint8_blocks(masks[idx], factor).astype(np.float32) / denominator
    return fractions


def _presence_reclass_from_class_fractions(fractions: np.ndarray) -> np.ndarray:
    """
    Derive a display class from any nonzero class fraction.

    Unlike the modal reclassification, this preserves minority drained/burned
    presence in coarse cells so it stays consistent with summed emissions.
    """
    class_fraction = np.asarray(fractions, dtype=np.float32)
    if class_fraction.shape[0] != 4:
        raise ValueError(
            "Expected class fractions with shape (4, rows, cols); "
            f"got {class_fraction.shape}"
        )

    has_undrained = class_fraction[0] > 0
    has_drained = (class_fraction[1] > 0) | (class_fraction[3] > 0)
    has_burned = (class_fraction[2] > 0) | (class_fraction[3] > 0)
    organic = has_undrained | has_drained | has_burned

    out = np.full(class_fraction.shape[1:], COMBINED_STATE_RECLASS_NODATA, dtype=np.uint8)
    out[organic & ~has_drained & ~has_burned] = COMBINED_STATE_RECLASS_UNDRAINED
    out[has_drained & ~has_burned] = COMBINED_STATE_RECLASS_DRAINED_ONLY
    out[~has_drained & has_burned] = COMBINED_STATE_RECLASS_BURNED_ONLY
    out[has_drained & has_burned] = COMBINED_STATE_RECLASS_DRAINED_BURNED
    return out


def _reclassify_combined_state(arr: np.ndarray) -> np.ndarray:
    """
    Collapse packed combined_state values into publication map classes.

    Output classes:
        1: undrained organic soil, not burned
        2: drained only
        3: burned only
        4: drained + burned
        255: nodata / non-organic / no state
    """
    organic, drained, burned = _combined_state_component_masks(arr)
    out = np.full(organic.shape, COMBINED_STATE_RECLASS_NODATA, dtype=np.uint8)
    out[organic & ~drained & ~burned] = COMBINED_STATE_RECLASS_UNDRAINED
    out[organic & drained & ~burned] = COMBINED_STATE_RECLASS_DRAINED_ONLY
    out[organic & ~drained & burned] = COMBINED_STATE_RECLASS_BURNED_ONLY
    out[organic & drained & burned] = COMBINED_STATE_RECLASS_DRAINED_BURNED
    return out


def _fill_reclassified_memmap(
    src: np.ndarray,
    dst: np.memmap,
    *,
    block_rows: int = 512,
) -> None:
    """Write the combined-state class raster in row blocks to keep memory bounded."""
    rows = src.shape[0]
    for row0 in range(0, rows, block_rows):
        row1 = min(row0 + block_rows, rows)
        dst[row0:row1, :] = _reclassify_combined_state(src[row0:row1, :])
        dst.flush()


def _fill_presence_reclassified_memmap(
    fractions: np.ndarray,
    dst: np.memmap,
    *,
    block_rows: int = 512,
) -> None:
    """Write presence-style class raster from fraction bands in row blocks."""
    rows = fractions.shape[1]
    for row0 in range(0, rows, block_rows):
        row1 = min(row0 + block_rows, rows)
        dst[row0:row1, :] = _presence_reclass_from_class_fractions(
            fractions[:, row0:row1, :]
        )
        dst.flush()


def _combined_state_companion_output(
    global_output_path: str,
    global_outfile: str,
    dataset_name: str,
) -> Tuple[str, str]:
    """Return output directory and filename for a combined-state companion raster."""
    out_dir = global_output_path.rstrip("/")
    marker = "/combined_state/"
    if marker in out_dir:
        out_dir = out_dir.replace(marker, f"/{dataset_name}/")
    else:
        out_dir = f"{out_dir}/{dataset_name}"

    if "__combined_state_" in global_outfile:
        out_name = global_outfile.replace(
            "__combined_state_",
            f"__{dataset_name}_",
        )
    else:
        stem, ext = os.path.splitext(global_outfile)
        out_name = f"{stem}__{dataset_name}{ext or '.tif'}"
    return out_dir, out_name


def _combined_state_reclass_output(global_output_path: str, global_outfile: str) -> Tuple[str, str]:
    """Return output directory and filename for the modal combined-state class raster."""
    return _combined_state_companion_output(
        global_output_path,
        global_outfile,
        COMBINED_STATE_RECLASS_DATASET,
    )


def _combined_state_class_fraction_output(
    global_output_path: str,
    global_outfile: str,
) -> Tuple[str, str]:
    """Return output directory and filename for class-fraction bands."""
    return _combined_state_companion_output(
        global_output_path,
        global_outfile,
        COMBINED_STATE_CLASS_FRACTION_DATASET,
    )


def _combined_state_presence_reclass_output(
    global_output_path: str,
    global_outfile: str,
) -> Tuple[str, str]:
    """Return output directory and filename for presence-style class raster."""
    return _combined_state_companion_output(
        global_output_path,
        global_outfile,
        COMBINED_STATE_PRESENCE_RECLASS_DATASET,
    )


def _per_pixel_tile_path(items: dict, tile_id: str) -> str:
    pp_dir = items.get("per_pixel_dir")
    pp_pat = items.get("per_pixel_pattern")
    if not pp_dir or not pp_pat:
        raise RuntimeError(
            "Per-pixel/state inputs are required but not configured in build_download_upload_dict: "
            "missing per_pixel_dir and/or per_pixel_pattern."
        )
    return f"{pp_dir}{tile_id}{pp_pat}"


def _join_output_path(base: str, name: str) -> str:
    return f"{base.rstrip('/')}/{name}"


def _to_vsipath_if_s3(path: str) -> str:
    return "/vsis3/" + path[len("s3://") :] if path.startswith("s3://") else path


def _save_global_raster_correct(
    *,
    bounds: Tuple[float, float, float, float],
    arr: np.ndarray,
    dtype: np.dtype,
    dst_base: str,
    dst_name: str,
    logger,
    int_nodata: Optional[int] = None,
    spool_dir: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
) -> str:
    """
    Write a single-band GeoTIFF with a correct geotransform derived from rows & cols.
    If destination is s3://..., enable GDAL spooling so random writes work.
    Returns the final destination path.
    """
    minx, miny, maxx, maxy = bounds
    rows, cols = arr.shape
    transform = from_bounds(minx, miny, maxx, maxy, cols, rows)

    dst_path = _join_output_path(dst_base, dst_name)
    vsi_path = _to_vsipath_if_s3(dst_path)

    is_float = np.issubdtype(dtype, np.floating)
    creation_opts = dict(
        driver="GTiff",
        bigtiff="YES",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="DEFLATE",
        predictor=2 if is_float else 1,
        num_threads="ALL_CPUS",
    )

    profile = dict(
        width=cols,
        height=rows,
        count=1,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=None if is_float else int_nodata,
        **creation_opts,
    )

    # If writing to S3, enable GDAL's temp-file spooling for random writes.
    env_kwargs = {}
    if dst_path.startswith("s3://"):
        env_kwargs["CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE"] = "YES"
        if spool_dir:
            env_kwargs["CPL_TMPDIR"] = spool_dir  # ensure enough space on this disk

    logger.info("Writing %s (size %d×%d, dtype=%s) ...",
                dst_path, cols, rows, np.dtype(dtype).name)
    with rasterio.Env(**env_kwargs):
        with rasterio.open(vsi_path, "w", **profile) as dst:
            dst.write(arr, 1)
            if tags:
                dst.update_tags(**tags)

    logger.info("Saved global (%s) → %s", np.dtype(dtype).name, dst_path)
    return dst_path


def _save_global_multiband_raster_correct(
    *,
    bounds: Tuple[float, float, float, float],
    arr: np.ndarray,
    dtype: np.dtype,
    dst_base: str,
    dst_name: str,
    logger,
    nodata: Optional[float | int] = None,
    spool_dir: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
    band_descriptions: Optional[Tuple[str, ...]] = None,
) -> str:
    """Write a multi-band GeoTIFF with a correct geotransform."""
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array (bands, rows, cols); got {arr.shape}")

    minx, miny, maxx, maxy = bounds
    bands, rows, cols = arr.shape
    transform = from_bounds(minx, miny, maxx, maxy, cols, rows)

    dst_path = _join_output_path(dst_base, dst_name)
    vsi_path = _to_vsipath_if_s3(dst_path)

    is_float = np.issubdtype(dtype, np.floating)
    creation_opts = dict(
        driver="GTiff",
        bigtiff="YES",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="DEFLATE",
        predictor=2 if is_float else 1,
        num_threads="ALL_CPUS",
    )

    profile = dict(
        width=cols,
        height=rows,
        count=bands,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
        **creation_opts,
    )

    env_kwargs = {}
    if dst_path.startswith("s3://"):
        env_kwargs["CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE"] = "YES"
        if spool_dir:
            env_kwargs["CPL_TMPDIR"] = spool_dir

    logger.info(
        "Writing %s (size %dÃ—%d, bands=%d, dtype=%s) ...",
        dst_path,
        cols,
        rows,
        bands,
        np.dtype(dtype).name,
    )
    with rasterio.Env(**env_kwargs):
        with rasterio.open(vsi_path, "w", **profile) as dst:
            for band_idx in range(bands):
                dst.write(arr[band_idx].astype(dtype, copy=False), band_idx + 1)
                if band_descriptions and band_idx < len(band_descriptions):
                    dst.set_band_description(band_idx + 1, band_descriptions[band_idx])
            if tags:
                dst.update_tags(**tags)

    logger.info("Saved global multiband (%s) â†’ %s", np.dtype(dtype).name, dst_path)
    return dst_path


def _write_combined_state_fraction_outputs(
    *,
    class_fraction: np.ndarray,
    global_output_path: str,
    global_outfile: str,
    logger,
    spool_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """Write canonical class fractions and a presence-style display raster."""
    fraction_dir, fraction_name = _combined_state_class_fraction_output(
        global_output_path,
        global_outfile,
    )
    fraction_path = _save_global_multiband_raster_correct(
        bounds=(-180, -90, 180, 90),
        arr=class_fraction,
        dtype=np.float32,
        dst_base=fraction_dir,
        dst_name=fraction_name,
        logger=logger,
        nodata=None,
        spool_dir=spool_dir,
        tags=COMBINED_STATE_CLASS_FRACTION_TAGS,
        band_descriptions=COMBINED_STATE_CLASS_BAND_DESCRIPTIONS,
    )

    tmp_parent = spool_dir or tempfile.gettempdir()
    presence_path = os.path.join(tmp_parent, "combined_state_presence_reclass_mm.dat")
    presence_mm = np.memmap(
        presence_path,
        dtype=np.uint8,
        mode="w+",
        shape=class_fraction.shape[1:],
    )
    try:
        _fill_presence_reclassified_memmap(class_fraction, presence_mm)
        presence_dir, presence_name = _combined_state_presence_reclass_output(
            global_output_path,
            global_outfile,
        )
        presence_output_path = _save_global_raster_correct(
            bounds=(-180, -90, 180, 90),
            arr=presence_mm,
            dtype=np.uint8,
            dst_base=presence_dir,
            dst_name=presence_name,
            logger=logger,
            int_nodata=int(COMBINED_STATE_RECLASS_NODATA),
            spool_dir=spool_dir,
            tags=COMBINED_STATE_PRESENCE_RECLASS_TAGS,
        )
    finally:
        try:
            del presence_mm
            os.remove(presence_path)
        except Exception:
            pass

    return fraction_path, presence_output_path


# --------------------------------------------------------------------
# Tile aggregation
# --------------------------------------------------------------------

def _agg_tile_to_target_windowed(
    tile_id: str,
    chunk_length_pixels: int,
    per_pixel_total_or_state_tile: str,
    native_deg: float,
    target_deg: float,
    is_final: bool,
    logger,
):
    parts = posixpath.basename(per_pixel_total_or_state_tile).split("__")
    if len(parts) == 3:
        _, dataset_name, _ = parts
    elif len(parts) >= 4:
        dataset_name = parts[2]
    else:
        raise ValueError(
            "Expected '<tile>__<dataset>__<interval>.tif' or "
            "'<tile>__<bounds>__<dataset>__<interval>.tif'; "
            f"got: {posixpath.basename(per_pixel_total_or_state_tile)}"
        )

    path = _to_vsipath_if_s3(per_pixel_total_or_state_tile)

    try:
        if dataset_name == "combined_state":
            return _mode_per_component_windowed(
                path,
                chunk_length_pixels,
                native_deg,
                target_deg,
                logger,
            )
        return _reaggregate_sum_windowed(
            path,
            chunk_length_pixels,
            native_deg,
            target_deg,
            logger,
        )
    except Exception as exc:
        lu.print_and_log(
            f"WARNING: {per_pixel_total_or_state_tile} failed ({exc}) -> zeros",
            is_final,
            logger,
        )
        factor = _aggregation_factor(native_deg, target_deg)
        out_shape = (chunk_length_pixels // factor, chunk_length_pixels // factor)
        if dataset_name == "combined_state":
            logger.warning(
                "Tile %s missing combined_state; treating as all-zero (nodata after aggregation).",
                tile_id,
            )
            return np.zeros(out_shape, dtype=np.uint32)
        return np.zeros(out_shape, dtype=np.float32)


def _agg_combined_state_class_fractions_windowed(
    tile_id: str,
    chunk_length_pixels: int,
    combined_state_tile: str,
    native_deg: float,
    target_deg: float,
    is_final: bool,
    logger,
) -> np.ndarray:
    path = _to_vsipath_if_s3(combined_state_tile)
    try:
        return _combined_state_class_fractions_windowed(
            path,
            chunk_length_pixels,
            native_deg,
            target_deg,
            logger,
        )
    except Exception as exc:
        lu.print_and_log(
            f"WARNING: {combined_state_tile} class-fraction aggregation failed ({exc}) -> zeros",
            is_final,
            logger,
        )
        factor = _aggregation_factor(native_deg, target_deg)
        out_shape = (
            len(COMBINED_STATE_CLASS_BAND_DESCRIPTIONS),
            chunk_length_pixels // factor,
            chunk_length_pixels // factor,
        )
        logger.warning(
            "Tile %s missing combined_state; treating class fractions as zero.",
            tile_id,
        )
        return np.zeros(out_shape, dtype=np.float32)


def agg_tile_to_target(
    tile_id: str,
    bounds: Tuple[float, float, float, float],
    chunk_length_pixels: int,
    per_pixel_total_or_state_tile: str,
    native_deg: float,
    target_deg: float,
    is_final: bool,
):
    """
    Aggregate one 10×10° tile from native_deg to target_deg.
    """
    logger = lu.setup_logging()
    logger.info("Reading tile %s\ninput: %s", tile_id, per_pixel_total_or_state_tile)
    return _agg_tile_to_target_windowed(
        tile_id,
        chunk_length_pixels,
        per_pixel_total_or_state_tile,
        native_deg,
        target_deg,
        is_final,
        logger,
    )

    # Determine dataset name up front (so we can choose the read dtype)
    parts = posixpath.basename(per_pixel_total_or_state_tile).split("__")
    if len(parts) == 3:
        _, dataset_name, _ = parts
    elif len(parts) >= 4:
        dataset_name = parts[2]
    else:
        raise ValueError(
            "Expected '<tile>__<dataset>__<interval>.tif' or "
            "'<tile>__<bounds>__<dataset>__<interval>.tif'; "
            f"got: {posixpath.basename(per_pixel_total_or_state_tile)}"
        )

    # Integer datasets are read as Int32; floats as Float32
    dtype_hint = "Int32" if dataset_name in INTEGER_DATASETS else "Float32"

    arr, success = uu.get_tile_dataset_rio(
        per_pixel_total_or_state_tile, dtype_hint, bounds, chunk_length_pixels, is_final, logger
    )

    if dataset_name == "combined_state" and not success:
        logger.warning(
            "Tile %s missing combined_state; treating as all-zero (nodata after aggregation).",
            tile_id,
        )
        arr = np.zeros((chunk_length_pixels, chunk_length_pixels), dtype=np.uint32)

    if dataset_name == "combined_state":
        return _mode_per_component(arr, native_deg, target_deg)

    # Continuous totals → explicit SUM to target resolution (no unit conversions)
    return _reaggregate_sum(arr.astype(np.float32, copy=False), native_deg, target_deg)


def agg_combined_state_class_fractions_to_target(
    tile_id: str,
    bounds: Tuple[float, float, float, float],
    chunk_length_pixels: int,
    combined_state_tile: str,
    native_deg: float,
    target_deg: float,
    is_final: bool,
) -> np.ndarray:
    """Aggregate one combined_state tile to four class-fraction bands."""
    logger = lu.setup_logging()
    logger.info(
        "Reading tile %s for combined_state class fractions\ninput: %s",
        tile_id,
        combined_state_tile,
    )
    return _agg_combined_state_class_fractions_windowed(
        tile_id,
        chunk_length_pixels,
        combined_state_tile,
        native_deg,
        target_deg,
        is_final,
        logger,
    )


# --------------------------------------------------------------------
# Execution helpers
# --------------------------------------------------------------------

def _compute_tiles(
    delayed_results: List,
    client,
    logger,
    stage_desc: str,
    tile_ids: List[str],
) -> List[np.ndarray]:
    """Legacy all-at-once compute; kept for backward compatibility."""
    if not delayed_results:
        return []
    if client is None:
        return list(dask.compute(*delayed_results))

    futures = client.compute(delayed_results, sync=False)
    future_to_index = {future: idx for idx, future in enumerate(futures)}
    tiles: List[Optional[np.ndarray]] = [None] * len(futures)
    completed = 0
    total = len(futures)

    for future in as_completed(list(future_to_index.keys())):
        idx = future_to_index[future]
        tile_id = tile_ids[idx]
        try:
            result = future.result()
        except Exception:
            logger.exception("Tile %s failed during %s", tile_id, stage_desc)
            raise

        tiles[idx] = result
        completed += 1
        if completed % 10 == 0 or completed == total:
            logger.info("Completed %d/%d tiles for %s", completed, total, stage_desc)

    missing = [tile_ids[idx] for idx, tile in enumerate(tiles) if tile is None]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} tile results for {stage_desc}: {', '.join(missing)}"
        )

    return [tile for tile in tiles if tile is not None]


def iterate_tiles(
    delayed_results: List,
    client,
    logger,
    stage_desc: str,
    tile_ids: List[str],
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (tile_index, tile_array) as each tile finishes computing."""
    if not delayed_results:
        return
    if client is None:
        # Smaller, configurable local batch to avoid long stalls
        B = int(os.environ.get("AGG_LOCAL_BATCH", "8"))
        for i in range(0, len(delayed_results), B):
            batch = delayed_results[i:i + B]
            results = dask.compute(*batch)
            for j, res in enumerate(results):
                yield (i + j, res)
    else:
        B = int(os.environ.get("AGG_DASK_BATCH", "0"))
        if B > 0:
            for start in range(0, len(delayed_results), B):
                batch = delayed_results[start:start + B]
                futures = client.compute(batch, sync=False)
                future_to_index = {f: start + i for i, f in enumerate(futures)}
                completed = 0
                total = len(futures)
                for future in as_completed(list(future_to_index.keys())):
                    idx = future_to_index[future]
                    try:
                        arr = future.result()
                    except Exception:
                        logger.exception("Tile %s failed during %s", tile_ids[idx], stage_desc)
                        raise
                    completed += 1
                    overall_completed = start + completed
                    if overall_completed % 10 == 0 or overall_completed == len(delayed_results):
                        logger.info(
                            "Completed %d/%d tiles for %s",
                            overall_completed,
                            len(delayed_results),
                            stage_desc,
                        )
                    yield (idx, arr)
            return

        futures = client.compute(delayed_results, sync=False)
        future_to_index = {f: i for i, f in enumerate(futures)}
        completed = 0
        total = len(futures)
        for future in as_completed(list(future_to_index.keys())):
            idx = future_to_index[future]
            try:
                arr = future.result()
            except Exception:
                logger.exception("Tile %s failed during %s", tile_ids[idx], stage_desc)
                raise
            completed += 1
            if completed % 10 == 0 or completed == total:
                logger.info("Completed %d/%d tiles for %s", completed, total, stage_desc)
            yield (idx, arr)


# --------------------------------------------------------------------
# Combine & write
# --------------------------------------------------------------------

def combine_global_raster(
    tiles: List[np.ndarray],
    bounds_list: List[Tuple[float, float, float, float]],
    res_label: str,
    global_outfile: str,
    global_output_path: str,
    target_deg: float,
    is_final: bool,
    *,
    out_dtype: Optional[np.dtype] = None,
    int_nodata: Optional[int] = None,
    write_combined_state_reclass: bool = False,
):
    """Legacy in-RAM combine; now writes with correct transform and S3-safe spooling."""
    logger = lu.setup_logging()
    rows = int(round(180 / target_deg))
    cols = int(round(360 / target_deg))

    if out_dtype is None or np.issubdtype(out_dtype, np.floating):
        global_raster = np.full((rows, cols), np.nan, dtype=np.float32)
        for tile, bounds in zip(tiles, bounds_list):
            min_x, min_y, max_x, max_y = bounds
            x0 = int(round((min_x + 180) / target_deg))
            x1 = int(round((max_x + 180) / target_deg))
            y0 = int(round((90 - max_y) / target_deg))
            y1 = int(round((90 - min_y) / target_deg))
            t = tile.astype(np.float32, copy=False)
            mask = ~np.isnan(t) if np.issubdtype(t.dtype, np.floating) else np.ones_like(t, dtype=bool)
            np.copyto(global_raster[y0:y1, x0:x1], t, where=mask)
        save_dtype = np.float32
        int_nodata_eff = None
    else:
        if int_nodata is None:
            raise ValueError("int_nodata must be provided when out_dtype is integer.")
        global_raster = np.full((rows, cols), int_nodata, dtype=out_dtype)
        for tile, bounds in zip(tiles, bounds_list):
            if np.issubdtype(tile.dtype, np.floating):
                t = np.where(np.isnan(tile), int_nodata, tile).astype(out_dtype, copy=False)
            else:
                t = tile.astype(out_dtype, copy=False)
            min_x, min_y, max_x, max_y = bounds
            x0 = int(round((min_x + 180) / target_deg))
            x1 = int(round((max_x + 180) / target_deg))
            y0 = int(round((90 - max_y) / target_deg))
            y1 = int(round((90 - min_y) / target_deg))
            np.copyto(global_raster[y0:y1, x0:x1], t, where=(t != int_nodata))
        save_dtype = out_dtype
        int_nodata_eff = int_nodata

    _ = _save_global_raster_correct(
        bounds=(-180, -90, 180, 90),
        arr=global_raster,
        dtype=save_dtype,
        dst_base=global_output_path,
        dst_name=global_outfile,
        logger=logger,
        int_nodata=int_nodata_eff,
        spool_dir=tempfile.gettempdir(),  # safe default
    )
    if write_combined_state_reclass:
        reclass_dir, reclass_name = _combined_state_reclass_output(
            global_output_path,
            global_outfile,
        )
        _ = _save_global_raster_correct(
            bounds=(-180, -90, 180, 90),
            arr=_reclassify_combined_state(global_raster),
            dtype=np.uint8,
            dst_base=reclass_dir,
            dst_name=reclass_name,
            logger=logger,
            int_nodata=int(COMBINED_STATE_RECLASS_NODATA),
            spool_dir=tempfile.gettempdir(),
            tags=COMBINED_STATE_RECLASS_TAGS,
        )
    return "Success"


def combine_global_raster_streaming(
    tiles_iter: "Iterator[Tuple[int, np.ndarray]]",
    bounds_list: List[Tuple[float, float, float, float]],
    res_label: str,
    global_outfile: str,
    global_output_path: str,
    target_deg: float,
    is_final: bool,
    *,
    out_dtype: Optional[np.dtype] = None,
    int_nodata: Optional[int] = None,
    write_combined_state_reclass: bool = False,
):
    """
    Stream tiles into a global memmap on disk; then write with correct transform.
    When writing to s3://, enable GDAL spooling and use the same tmpdir for spooling.
    """
    logger = lu.setup_logging()

    rows = int(round(180 / target_deg))
    cols = int(round(360 / target_deg))

    # Use a dedicated working directory; we also point GDAL spooling here.
    tmp_parent = os.environ.get("AGG_GLOBAL_TMPDIR")
    if tmp_parent:
        os.makedirs(tmp_parent, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix=f"{res_label}_global_", dir=tmp_parent)
    mm_path = os.path.join(tmpdir, "global_mm.dat")
    skip_init = os.environ.get("AGG_SKIP_GLOBAL_INIT", "0") == "1"

    def init_memmap(arr: np.memmap, value, label: str) -> None:
        if skip_init:
            logger.info(
                "Skipping full %s memmap initialization at %s; tiles will overwrite the global grid.",
                label,
                mm_path,
            )
            return
        init_rows = int(os.environ.get("AGG_GLOBAL_INIT_ROWS", "1024"))
        logger.info(
            "Initializing %s memmap at %s (shape=%s, rows_per_block=%d)",
            label,
            mm_path,
            arr.shape,
            init_rows,
        )
        for row0 in range(0, arr.shape[0], init_rows):
            row1 = min(row0 + init_rows, arr.shape[0])
            arr[row0:row1, :] = value
            arr.flush()
            if row1 == arr.shape[0] or row1 % (init_rows * 8) == 0:
                logger.info("Initialized %d/%d rows for %s", row1, arr.shape[0], label)

    if out_dtype is None or np.issubdtype(out_dtype, np.floating):
        save_dtype = np.float32
        global_mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=(rows, cols))
        init_memmap(global_mm, np.nan, "float")
        def paste(tile, y0, y1, x0, x1):
            t = tile.astype(np.float32, copy=False)
            if skip_init:
                global_mm[y0:y1, x0:x1] = t
            else:
                mask = ~np.isnan(t) if np.issubdtype(t.dtype, np.floating) else np.ones_like(t, dtype=bool)
                np.copyto(global_mm[y0:y1, x0:x1], t, where=mask)
        int_nodata_eff = None
    else:
        if int_nodata is None:
            raise ValueError("int_nodata must be provided when out_dtype is integer.")
        save_dtype = out_dtype
        global_mm = np.memmap(mm_path, dtype=out_dtype, mode="w+", shape=(rows, cols))
        init_memmap(global_mm, int_nodata, "integer")
        def paste(tile, y0, y1, x0, x1):
            if np.issubdtype(tile.dtype, np.floating):
                t = np.where(np.isnan(tile), int_nodata, tile).astype(out_dtype, copy=False)
            else:
                t = tile.astype(out_dtype, copy=False)
            if skip_init:
                global_mm[y0:y1, x0:x1] = t
            else:
                np.copyto(global_mm[y0:y1, x0:x1], t, where=(t != int_nodata))
        int_nodata_eff = int_nodata

    flush_every = 16
    seen = 0
    for idx, tile in tiles_iter:
        min_x, min_y, max_x, max_y = bounds_list[idx]
        x0 = int(round((min_x + 180) / target_deg))
        x1 = int(round((max_x + 180) / target_deg))
        y0 = int(round((90 - max_y) / target_deg))
        y1 = int(round((90 - min_y) / target_deg))
        paste(tile, y0, y1, x0, x1)
        seen += 1
        if seen % flush_every == 0:
            global_mm.flush()

    del global_mm
    global_mm = np.memmap(mm_path, dtype=save_dtype, mode="r", shape=(rows, cols))

    _ = _save_global_raster_correct(
        bounds=(-180, -90, 180, 90),
        arr=global_mm,
        dtype=save_dtype,
        dst_base=global_output_path,
        dst_name=global_outfile,
        logger=logger,
        int_nodata=int_nodata_eff,
        spool_dir=tmpdir,  # <-- S3 spooling will use this same directory
    )

    if write_combined_state_reclass:
        reclass_dir, reclass_name = _combined_state_reclass_output(
            global_output_path,
            global_outfile,
        )
        reclass_path = os.path.join(tmpdir, "combined_state_reclass_mm.dat")
        reclass_mm = np.memmap(reclass_path, dtype=np.uint8, mode="w+", shape=(rows, cols))
        _fill_reclassified_memmap(global_mm, reclass_mm)
        _ = _save_global_raster_correct(
            bounds=(-180, -90, 180, 90),
            arr=reclass_mm,
            dtype=np.uint8,
            dst_base=reclass_dir,
            dst_name=reclass_name,
            logger=logger,
            int_nodata=int(COMBINED_STATE_RECLASS_NODATA),
            spool_dir=tmpdir,
            tags=COMBINED_STATE_RECLASS_TAGS,
        )
        del reclass_mm
        try:
            os.remove(reclass_path)
        except Exception:
            pass

    # Cleanup
    try:
        del global_mm
        os.remove(mm_path)
        os.rmdir(tmpdir)
    except Exception:
        pass

    return "Success"


def combine_global_fraction_stack_streaming(
    tiles_iter: "Iterator[Tuple[int, np.ndarray]]",
    bounds_list: List[Tuple[float, float, float, float]],
    res_label: str,
    global_outfile: str,
    global_output_path: str,
    target_deg: float,
    is_final: bool,
) -> str:
    """Stream four-band combined-state class fractions into a global raster."""
    logger = lu.setup_logging()

    rows = int(round(180 / target_deg))
    cols = int(round(360 / target_deg))
    bands = len(COMBINED_STATE_CLASS_BAND_DESCRIPTIONS)

    tmp_parent = os.environ.get("AGG_GLOBAL_TMPDIR")
    if tmp_parent:
        os.makedirs(tmp_parent, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix=f"{res_label}_class_fraction_global_", dir=tmp_parent)
    mm_path = os.path.join(tmpdir, "combined_state_class_fraction_mm.dat")
    global_mm = np.memmap(
        mm_path,
        dtype=np.float32,
        mode="w+",
        shape=(bands, rows, cols),
    )

    init_rows = int(os.environ.get("AGG_GLOBAL_INIT_ROWS", "1024"))
    logger.info(
        "Initializing combined_state class-fraction memmap at %s (shape=%s)",
        mm_path,
        global_mm.shape,
    )
    for row0 in range(0, rows, init_rows):
        row1 = min(row0 + init_rows, rows)
        global_mm[:, row0:row1, :] = 0.0
        global_mm.flush()
        if row1 == rows or row1 % (init_rows * 8) == 0:
            logger.info("Initialized %d/%d rows for class fractions", row1, rows)

    flush_every = 16
    seen = 0
    for idx, tile in tiles_iter:
        min_x, min_y, max_x, max_y = bounds_list[idx]
        x0 = int(round((min_x + 180) / target_deg))
        x1 = int(round((max_x + 180) / target_deg))
        y0 = int(round((90 - max_y) / target_deg))
        y1 = int(round((90 - min_y) / target_deg))
        if tile.shape[0] != bands:
            raise ValueError(
                f"Expected {bands} class-fraction bands for tile index {idx}; "
                f"got shape {tile.shape}"
            )
        global_mm[:, y0:y1, x0:x1] = tile.astype(np.float32, copy=False)
        seen += 1
        if seen % flush_every == 0:
            global_mm.flush()

    global_mm.flush()
    _write_combined_state_fraction_outputs(
        class_fraction=global_mm,
        global_output_path=global_output_path,
        global_outfile=global_outfile,
        logger=logger,
        spool_dir=tmpdir,
    )

    try:
        del global_mm
        os.remove(mm_path)
        os.rmdir(tmpdir)
    except Exception:
        pass

    return "Success"


# --------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------

def aggregate_main(
    cluster_name: str,
    pixel_resolution: str,
    run_name: str = "ogh_sensitivity_1km",
    run_local: bool = False,
    native_deg: float = DEFAULT_NATIVE_DEG,
    target_deg: float = DEFAULT_TARGET_DEG,
    output_date: str = DEFAULT_DATE_TAG,
    model_version: str = DEFAULT_MODEL_VERSION,
    outputs_root: str = OUTPUT_ROOT,
    base_url: Optional[str] = None,
    outputs_base: Optional[str] = None,
    data_types: Optional[List[str]] = None,
    inventory_periods: Optional[List[str]] = None,
    tile_ids: Optional[List[str]] = None,
) -> None:
    assert_grid_divides_world(target_deg)

    logger = lu.setup_logging_main()
    is_final = not run_local

    cluster, client, run_local = uu.connect_to_cluster(cluster_name, run_local=run_local)
    is_final = not run_local

    resolved_base_url, resolved_outputs_base = resolve_versioned_paths(
        model_version=model_version,
        outputs_root=outputs_root,
        base_url=base_url,
        outputs_base=outputs_base,
    )

    download_upload_dictionary = build_download_upload_dict(
        pixel_resolution=pixel_resolution,
        run_name=run_name,
        target_deg=target_deg,
        base_url=resolved_base_url,
        output_date=output_date,
        outputs_base=resolved_outputs_base,
        data_types=data_types,
        inventory_periods=inventory_periods,
    )

    if tile_ids:
        unknown_tile_ids = [tile_id for tile_id in tile_ids if tile_id not in cn.tile_id_list]
        if unknown_tile_ids:
            raise ValueError(f"Unknown tile_ids requested: {unknown_tile_ids}")
        selected_tile_ids = list(dict.fromkeys(tile_ids))
    else:
        selected_tile_ids = list(cn.tile_id_list)

    res_label = deg_to_label(target_deg)
    tile_agg_func = agg_tile_to_target
    if __name__ == "__main__":
        from src.scripts.postprocessing.visualization import create_global_raster as cgr_module
        tile_agg_func = cgr_module.agg_tile_to_target

    for key, items in download_upload_dictionary.items():
        bounds_list: List[Tuple[float, float, float, float]] = []
        delayed_results: List = []
        tile_ids_for_key: List[str] = []

        dataset_name = items["dataset"]
        is_combined = (dataset_name == "combined_state")

        stage = f"aggregate tiles to {res_label} for {key}"
        lu.print_and_log(f"Stage {stage} started at: {uu.timestr()}", is_final, logger)

        for tile_id in selected_tile_ids:
            tile_path = _per_pixel_tile_path(items, tile_id)
            bounds = uu.get_10x10_tile_bounds(tile_id)
            bounds_list.append(bounds)
            chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)
            tile_ids_for_key.append(tile_id)

            delayed_results.append(
                dask.delayed(tile_agg_func)(
                    tile_id,
                    bounds,
                    chunk_length_pixels,
                    tile_path,
                    native_deg,
                    target_deg,
                    is_final,
                )
            )

        stage_desc = f"{res_label} aggregation for {key}"
        tiles_iter = iterate_tiles(
            delayed_results=delayed_results,
            client=None if run_local else client,
            logger=logger,
            stage_desc=stage_desc,
            tile_ids=tile_ids_for_key,
        )

        stage = f"build {res_label} global mosaic for {key}"
        lu.print_and_log(f"Stage {stage} started at: {uu.timestr()}", is_final, logger)

        global_outfile = items["global_pattern"]
        global_output_path = items["global_dir"]

        if is_combined:
            _ = combine_global_raster_streaming(
                tiles_iter=tiles_iter,
                bounds_list=bounds_list,
                res_label=res_label,
                global_outfile=global_outfile,
                global_output_path=global_output_path,
                target_deg=target_deg,
                is_final=is_final,
                out_dtype=np.uint32,
                int_nodata=0,  # 0 = no drained and no burned component present
                write_combined_state_reclass=True,
            )

            fraction_delayed_results: List = []
            for tile_id, bounds in zip(tile_ids_for_key, bounds_list):
                tile_path = _per_pixel_tile_path(items, tile_id)
                chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)
                fraction_delayed_results.append(
                    dask.delayed(agg_combined_state_class_fractions_to_target)(
                        tile_id,
                        bounds,
                        chunk_length_pixels,
                        tile_path,
                        native_deg,
                        target_deg,
                        is_final,
                    )
                )

            fraction_stage_desc = f"{res_label} class-fraction aggregation for {key}"
            lu.print_and_log(
                f"Stage build {res_label} class-fraction global mosaic for {key} started at: {uu.timestr()}",
                is_final,
                logger,
            )
            fraction_tiles_iter = iterate_tiles(
                delayed_results=fraction_delayed_results,
                client=None if run_local else client,
                logger=logger,
                stage_desc=fraction_stage_desc,
                tile_ids=tile_ids_for_key,
            )
            _ = combine_global_fraction_stack_streaming(
                tiles_iter=fraction_tiles_iter,
                bounds_list=bounds_list,
                res_label=res_label,
                global_outfile=global_outfile,
                global_output_path=global_output_path,
                target_deg=target_deg,
                is_final=is_final,
            )
        else:
            _ = combine_global_raster_streaming(
                tiles_iter=tiles_iter,
                bounds_list=bounds_list,
                res_label=res_label,
                global_outfile=global_outfile,
                global_output_path=global_output_path,
                target_deg=target_deg,
                is_final=is_final,
                out_dtype=None,
            )

        lu.print_and_log(
            f"Global raster saved to {_join_output_path(global_output_path, global_outfile)}",
            is_final,
            logger,
        )

    if client is not None:
        client.close()
    if cluster is not None:
        try:
            cluster.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate to a target resolution and build global mosaics "
            "(SUM for totals; combined_state aggregated by component-wise mode, "
            "repacked, and accompanied by a four-class reclassified raster)."
        )
    )
    parser.add_argument("-cn", "--cluster_name", required=True)
    parser.add_argument("--date_tag", required=True)
    parser.add_argument("-p", "--pixel_resolution", default="40000_pixels")
    parser.add_argument("--run_name", default="ogh_sensitivity_1km")
    parser.add_argument("--run_local", action="store_true")
    parser.add_argument("--native_deg", type=float, default=DEFAULT_NATIVE_DEG)
    parser.add_argument("--target_deg", type=float, default=DEFAULT_TARGET_DEG)
    parser.add_argument(
        "--model_version",
        default=DEFAULT_MODEL_VERSION,
        help="Model version string (underscore separated) used to build S3 paths.",
    )
    parser.add_argument(
        "--outputs_root",
        default=OUTPUT_ROOT,
        help="Root S3 directory for model outputs.",
    )
    parser.add_argument(
        "--base_url",
        default=None,
        help="Optional override for the versioned base URL containing per-tile rasters.",
    )
    parser.add_argument(
        "--outputs_base",
        default=None,
        help="Optional override for the destination of aggregated rasters.",
    )
    parser.add_argument(
        "--data_types",
        nargs="+",
        default=None,
        help="Optional subset of per-pixel/state datasets to aggregate.",
    )
    parser.add_argument(
        "--inventory_periods",
        nargs="+",
        default=None,
        help="Optional subset of inventory periods, e.g. 2021_2024.",
    )
    parser.add_argument(
        "--tile_ids",
        nargs="+",
        default=None,
        help="Optional subset of 10x10 tile IDs. Supports spaces or commas.",
    )

    args = parser.parse_args()

    aggregate_main(
        cluster_name=args.cluster_name,
        pixel_resolution=args.pixel_resolution,
        run_name=args.run_name,
        run_local=args.run_local,
        native_deg=args.native_deg,
        target_deg=args.target_deg,
        output_date=args.date_tag,
        model_version=args.model_version,
        outputs_root=args.outputs_root,
        base_url=args.base_url,
        outputs_base=args.outputs_base,
        data_types=_split_cli_items(args.data_types),
        inventory_periods=_split_cli_items(args.inventory_periods),
        tile_ids=_split_cli_items(args.tile_ids),
    )


if __name__ == "__main__":
    main()
