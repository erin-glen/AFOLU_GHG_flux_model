"""
Reclassify a packed ``combined_state`` GeoTIFF (Stage 01 aggregation output)
into a uint8 binary drained / undrained organic-soil raster suitable for
direct visualisation or as input to ``create_global_displays.py``.

Input encoding (uint32, packed — see ``zonal_constants``):
    bits 0-7 : drained_state_id  (index into DRAINED_STATE_ID_TO_CODE)
    bits 8-11: burned_state_id
    bit 12   : has_drainage_component (set whenever drained_id > 0,
               including for undrained peat)
    bit 13   : has_burned_component

Output encoding (uint8), matching the ``drained_state`` binary contract
consumed by ``create_global_displays.py``:
    1   : drained organic soil   (code prefix 11/12/13/14/15)
    0   : undrained organic soil (code prefix 16)
    255 : nodata                 (non-peat or outside organic-soil extent)

Examples
--------
# Default: write ``...__drained_binary_<interval>.tif`` next to the source
python -m src.scripts.postprocessing.visualization.build_drained_binary_raster \
  --src C:/tmp/global_drained_state_biome/0_01deg_global__combined_state_2021_2024.tif

# Explicit destination
python -m src.scripts.postprocessing.visualization.build_drained_binary_raster \
  --src .../combined_state.tif --dst .../drained_binary.tif --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

from src.scripts.utilities import log_utilities as lu
from src.scripts.zonal_statistics.zonal_constants import (
    COMBINED_STATE_DRAINED_MASK,
    DRAINED_STATE_ID_TO_CODE,
)

DRAINED_PREFIXES = ("11", "12", "13", "14", "15")
UNDRAINED_PREFIX = "16"

OUT_DRAINED = np.uint8(1)
OUT_UNDRAINED = np.uint8(0)
OUT_NODATA = np.uint8(255)

# drained_id is a packed 8-bit field, so the LUT never needs more than 256 entries.
_LUT_SIZE = 1 << 8


def build_drained_id_lut() -> np.ndarray:
    """Return a uint8 LUT mapping ``drained_id`` to {0, 1, 255}."""
    lut = np.full(_LUT_SIZE, OUT_NODATA, dtype=np.uint8)
    for idx, code in DRAINED_STATE_ID_TO_CODE.items():
        if not 0 <= idx < _LUT_SIZE:
            continue
        if code.startswith(UNDRAINED_PREFIX):
            lut[idx] = OUT_UNDRAINED
        elif code.startswith(DRAINED_PREFIXES):
            lut[idx] = OUT_DRAINED
        # Codes starting with "0" are non-peat — leave as nodata.
    # drained_id == 0 means the drainage component was never populated.
    lut[0] = OUT_NODATA
    return lut


def reclassify(src_path: Path, dst_path: Path, logger) -> None:
    lut = build_drained_id_lut()
    drained_mask = np.uint32(COMBINED_STATE_DRAINED_MASK)

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(
            dtype="uint8",
            nodata=int(OUT_NODATA),
            count=1,
            compress="DEFLATE",
            predictor=1,
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )
        logger.info(
            "Reclassifying %s (%dx%d, %s) -> %s",
            src_path, src.width, src.height, src.dtypes[0], dst_path,
        )

        n_drained = n_undrained = n_nodata = 0
        with rasterio.open(dst_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                arr = src.read(1, window=window)
                drained_id = (arr.astype(np.uint32, copy=False) & drained_mask).astype(np.uint8)
                out = lut[drained_id]
                dst.write(out, 1, window=window)
                n_drained += int(np.sum(out == OUT_DRAINED))
                n_undrained += int(np.sum(out == OUT_UNDRAINED))
                n_nodata += int(np.sum(out == OUT_NODATA))

    total_peat = n_drained + n_undrained
    logger.info(
        "Pixel counts: drained=%d undrained=%d nodata=%d (drained share of peat=%.1f%%)",
        n_drained, n_undrained, n_nodata,
        100.0 * n_drained / total_peat if total_peat else float("nan"),
    )
    logger.info("Wrote binary drained/undrained raster: %s", dst_path)


def _default_destination(src: Path) -> Path:
    stem = src.stem
    if "__combined_state" in stem:
        stem = stem.replace("__combined_state", "__drained_binary")
    else:
        stem = f"{stem}__drained_binary"
    return src.with_name(stem + ".tif")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reclassify a packed combined_state GeoTIFF into a uint8 binary "
            "drained/undrained organic-soil raster (1=drained, 0=undrained, 255=nodata)."
        )
    )
    parser.add_argument("--src", required=True, help="Input combined_state GeoTIFF (uint32).")
    parser.add_argument(
        "--dst",
        default=None,
        help="Output path. Default: alongside source with '__combined_state' -> '__drained_binary'.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite dst if it exists.")
    args = parser.parse_args()

    logger = lu.setup_logging_main()

    src = Path(args.src).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    dst = Path(args.dst).expanduser().resolve() if args.dst else _default_destination(src)
    if dst.exists() and not args.overwrite:
        raise FileExistsError(f"{dst} exists; pass --overwrite to replace.")
    dst.parent.mkdir(parents=True, exist_ok=True)

    reclassify(src, dst, logger)


if __name__ == "__main__":
    main()
