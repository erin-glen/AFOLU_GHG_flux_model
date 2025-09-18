# -*- coding: utf-8 -*-
"""
Render global mosaics as Robinson-projected JPEGs (and optional GIFs)
without loading huge rasters into memory.

This script reprojects and resamples to a **display grid** sized to your
figure (panel_width × dpi × proj_scale) using a WarpedVRT, so memory use
is bounded by your chosen output image size rather than native mosaic size.

---------------------------------------------------------------------------
Quick examples
---------------------------------------------------------------------------

# 1) Typical run (reads mosaics from S3 in place, 0.04° grid)
python -m src.scripts.postprocessing.render_global_maps \
  --date_tag 20250825 \
  --run_name ogh_standard_model \
  --read_from_s3 \
  --dpi 300 --proj_scale 2.0 --resampling cubic \
  --low_p 2 --high_p 98 --interpolation bilinear

# 2) Finer aggregated grid (0.01°) – still safe: we render to display size
python -m src.scripts.postprocessing.render_global_maps \
  --date_tag 20250825 \
  --run_name ogh_standard_model \
  --target_deg 0.01 \
  --read_from_s3 \
  --proj_scale 2.0 --resampling cubic \
  --low_p 2 --high_p 98 --interpolation bilinear

# 3) No borders / faster cartography:
python -m src.scripts.postprocessing.render_global_maps \
  --date_tag 20250825 --run_name ogh_standard_model \
  --read_from_s3 --no_borders

# 4) Log stretch for strictly-positive "gross emissions" layers:
python -m src.scripts.postprocessing.render_global_maps \
  --date_tag 20250825 --run_name ogh_standard_model \
  --read_from_s3 --log_emissions --low_p 5 --high_p 99

# 5) Custom output base (if your mosaics live somewhere else):
python -m src.scripts.postprocessing.render_global_maps \
  --date_tag 20250825 --run_name ogh_standard_model \
  --outputs_base s3://gfw2-data/climate/AFOLU_GHG_flux_model/organic_soils/outputs \
  --read_from_s3

# 6) Specify datasets/intervals explicitly (comma-separated):
python -m src.scripts.postprocessing.render_global_maps \
  --datasets burned_total_Mg_CO2e_pixel_yr,drained_total_Mg_CO2e_pixel_yr \
  --intervals 2021_2024 \
  --run_name ogh_standard_model \
  --read_from_s3

Notes
-----
• The renderer looks for mosaics under:
  {outputs_base}/{RES}_output_aggregation/{dataset}/{interval}/
  and picks the latest file matching:
  {RES}_global__{dataset}_{interval}*.tif
  where RES is e.g. "0_04deg" or "0_01deg".

• Units: mosaics are assumed to be in **Mg (t) CO₂(e) per cell per year**.
  Use `--unit_scale kt|Mt|Gt` to set the legend units (default: kt).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import posixpath
from pathlib import Path
from typing import Iterable, List, Tuple

import boto3
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, calculate_default_transform

# Project constants (colors, CRS, paths) – with safe fallbacks
from src.scripts.utilities import constants_and_names as cn


# ---------------------------------------------------------------------------
# Defaults & palettes
# ---------------------------------------------------------------------------

DEFAULT_OUTPUTS_BASE = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"
DEFAULT_TARGET_DEG = 0.04
DEFAULT_DATASETS = [
    "burned_total_Mg_CO2e_pixel_yr",
    "drained_total_Mg_CO2e_pixel_yr",
]
DEFAULT_INTERVALS = ["2021_2024"]

# Brewer BrBG-like diverging palette (greens → browns)
NET_PALETTE = [
    (0, 60, 48),
    (1, 102, 94),
    (53, 151, 143),
    (128, 205, 193),
    (199, 234, 229),
    (246, 232, 195),
    (223, 194, 125),
    (191, 129, 45),
    (140, 81, 10),
    (84, 48, 5),
]
REMOVALS_PALETTE = NET_PALETTE[0:5]
EMISSIONS_PALETTE = NET_PALETTE[5:]

# Resampling mapping for GDAL / WarpedVRT (exposed in CLI)
RESAMPLING_MAP = {
    "nearest":  Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic":    Resampling.cubic,
    "lanczos":  Resampling.lanczos,
    "average":  Resampling.average,  # mean
    "mode":     Resampling.mode,     # categorical majority
    "gauss":    Resampling.gauss,
    "min":      Resampling.min,
    "max":      Resampling.max,
    "med":      Resampling.med,      # median
    "q1":       Resampling.q1,
    "q3":       Resampling.q3,
    "sum":      Resampling.sum,
    "rms":      Resampling.rms,
}

# Matplotlib interpolation choices
IMSHOW_INTERP = {"none": "none", "nearest": "nearest", "bilinear": "bilinear"}

# Fallbacks if not present in constants
ROBINSON_CRS = getattr(cn, "Robinson_crs", "ESRI:54030")
OCEAN_RGB = getattr(cn, "ocean_color", (235, 235, 235))
LAND_RGB = getattr(cn, "land_bkgrnd", (245, 245, 245))
BORDER_RGB = getattr(cn, "boundary_color", (150, 150, 150))
BORDER_W = getattr(cn, "boundary_width", 0.2)
PANEL_DIMS = getattr(cn, "panel_dims", (12, 6))
DPI_DEFAULT = getattr(cn, "dpi_jpeg", 300)
LEGEND_FONTSIZE = getattr(cn, "legend_fontsize", 9)
CBAR_RECT = getattr(cn, "colorbar_dimensions", [0.14, 0.17, 0.02, 0.13])
PRES_TEXT = getattr(cn, "pres_text", "Preliminary land use vegetation fluxes")

SHP_ORIG = getattr(cn, "original_shapefile_path", "/mnt/c/GIS/world-administrative-boundaries_simple.shp")
SHP_REPROJ = getattr(cn, "reprojected_shapefile_path", "/mnt/c/GIS/world-administrative-boundaries_simple__reproj.shp")

_S3_CLIENT = getattr(cn, "s3_client", boto3.client("s3"))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _is_s3_path(path: str) -> bool:
    return path.startswith("s3://")


def _split_s3(url: str) -> Tuple[str, str]:
    assert url.startswith("s3://")
    rest = url[len("s3://") :]
    b, _, k = rest.partition("/")
    return b, k


def _upload_dir_to_s3(local_dir: Path, s3_prefix: str):
    bucket, prefix = _split_s3(s3_prefix)
    for f in Path(local_dir).rglob("*"):
        if f.is_file():
            rel = f.relative_to(local_dir).as_posix()
            _S3_CLIENT.upload_file(str(f), bucket, f"{prefix.rstrip('/')}/{rel}")


def deg_to_label(deg: float) -> str:
    """Convert 0.04 -> '0_04deg'; 0.01 -> '0_01deg'."""
    s = f"{deg:.5f}".rstrip("0").rstrip(".")
    return f"{s.replace('.', '_')}deg"


def _density_to_pixel_name(dataset: str) -> str:
    """Convert *_ha or *_ha_yr → *_pixel or *_pixel_yr (if needed)."""
    return dataset.replace("_ha_yr", "_pixel_yr").replace("_ha", "_pixel")


def _rgb_to_mpl(rgb):  # 0..255 → 0..1
    return tuple(v / 255 for v in rgb)


def _rgb_palette_to_mpl(rgb_palette):
    return [_rgb_to_mpl(rgb) for rgb in rgb_palette]


def _gdalize_s3_url(s3_url: str) -> str:
    """s3://bucket/key.tif -> /vsis3/bucket/key.tif"""
    if s3_url.startswith("s3://"):
        return "/vsis3/" + s3_url[len("s3://") :]
    return s3_url


def _list_s3_keys(prefix_s3: str) -> Iterable[str]:
    """Yield keys under s3:// prefix."""
    bucket, prefix = _split_s3(prefix_s3)
    token = None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix)
        if token:
            kw["ContinuationToken"] = token
        resp = _S3_CLIENT.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            yield obj["Key"]
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")


def _find_global_tif(global_dir: str, expected_basename: str, read_from_s3: bool) -> str:
    """
    Resolve the actual global mosaic file and return a path **without** the '.tif' extension.
    Tries exact name; if missing, picks the lexicographically-latest file matching the prefix.
    """
    if read_from_s3:
        bucket, prefix = _split_s3(global_dir)
        exact_key = f"{prefix.rstrip('/')}/{expected_basename}.tif"
        try:
            _S3_CLIENT.head_object(Bucket=bucket, Key=exact_key)
            return f"s3://{bucket}/{exact_key[:-4]}"
        except Exception:
            pass

        # Fallback: search
        start = f"{prefix.rstrip('/')}/{expected_basename}"
        matches = [k for k in _list_s3_keys(global_dir) if k.startswith(start) and k.endswith(".tif")]
        if not matches:
            raise FileNotFoundError(f"No global mosaic under {global_dir} with base '{expected_basename}'.")
        chosen = sorted(matches)[-1]
        return f"s3://{bucket}/{chosen[:-4]}"
    else:
        exact = Path(global_dir) / f"{expected_basename}.tif"
        if exact.exists():
            return str(exact)[:-4]
        matches = sorted([str(p) for p in Path(global_dir).glob(f"{expected_basename}*.tif")])
        if not matches:
            raise FileNotFoundError(f"No global mosaic in {global_dir} with base '{expected_basename}'.")
        return matches[-1][:-4]


def _open_display_vrt(
    src_ds,
    dst_crs: str,
    out_px_w: int,
    out_px_h: int,
    resampling: str,
    dst_nodata: float | int = 0,
):
    """
    Build a WarpedVRT that reprojects/decimates the source into exactly
    out_px_w × out_px_h pixels in dst_crs. Keeps memory bounded.
    """
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_ds.crs,
        dst_crs,
        src_ds.width,
        src_ds.height,
        *src_ds.bounds,
        dst_width=out_px_w,
        dst_height=out_px_h,
    )

    vrt = WarpedVRT(
        src_ds,
        crs=dst_crs,
        transform=dst_transform,
        width=dst_width,
        height=dst_height,
        resampling=RESAMPLING_MAP.get(resampling, Resampling.cubic),
        nodata=dst_nodata,
    )
    return vrt


def _choose_recipe(dataset_name: str):
    """
    Get ('diverging' | 'emissions' | 'removals', legend_title).
    """
    d = dataset_name.lower()
    if "net" in d:
        return "diverging", "Net greenhouse gas flux"
    if "removal" in d or "sink" in d:
        return "removals", "Gross CO\u2082 removals"
    return "emissions", "Gross greenhouse gas emissions"


def _compute_stretch(
    data: np.ndarray,
    mode: str,
    low_p: float,
    high_p: float,
    symmetric: bool,
) -> Tuple[float, float, float | None]:
    """
    Return (vmin, vmax, vcenter) for color normalization.
    For diverging: TwoSlopeNorm centered at 0 (if symmetric) else min/max percentiles.
    """
    nz = data[np.isfinite(data)]
    if nz.size == 0:
        return 0.0, 1.0, 0.0 if mode == "diverging" else None

    if mode == "diverging":
        if symmetric:
            pos = nz[nz > 0.0]
            neg = -nz[nz < 0.0]  # absolute values
            pos_q = np.percentile(pos, high_p) if pos.size else 0.0
            neg_q = np.percentile(neg, high_p) if neg.size else 0.0
            vmax = float(max(pos_q, neg_q))
            vmin = -vmax
            return vmin, vmax, 0.0
        else:
            vmin = float(np.percentile(nz, low_p))
            vmax = float(np.percentile(nz, high_p))
            return vmin, vmax, 0.0

    elif mode == "emissions":
        pos = nz[nz > 0.0]
        if pos.size == 0:
            return 0.0, 1.0, None
        vmin = float(np.percentile(pos, low_p))
        vmax = float(np.percentile(pos, high_p))
        return vmin, vmax, None

    else:  # removals (negative)
        neg = -nz[nz < 0.0]  # absolute magnitudes
        if neg.size == 0:
            return 0.0, 1.0, None
        vmin = float(np.percentile(neg, low_p))
        vmax = float(np.percentile(neg, high_p))
        return vmin, vmax, None


def _legend(
    fig: plt.Figure,
    img,
    mode: str,
    vmin: float,
    vmax: float,
    vcenter: float | None,
    legend_title: str,
    unit_label: str,
):
    cax = fig.add_axes(CBAR_RECT)
    cb = plt.colorbar(img, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=LEGEND_FONTSIZE)

    def fmt(v):
        if abs(v) >= 100:
            return f"{v:.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        return f"{v:.2f}"

    if mode == "diverging" and vcenter is not None:
        cb.set_ticks([vmin, vcenter, vmax])
        cb.set_ticklabels([fmt(vmin), "0", fmt(vmax)])
    else:
        cb.set_ticks([vmin, vmax])
        cb.set_ticklabels([fmt(vmin), fmt(vmax)])

    title = f"{legend_title}\n{unit_label} yr$^{{-1}}$"
    cax.text(0, 1.1, title, fontsize=LEGEND_FONTSIZE, ha="left", va="bottom", transform=cax.transAxes)


def _draw_borders(ax, shp):
    # land fill
    for geom in shp.geometry:
        try:
            if hasattr(geom, "exterior") and geom.exterior is not None:
                x, y = geom.exterior.xy
                ax.fill(x, y, color=_rgb_to_mpl(LAND_RGB), zorder=1)
            else:
                for part in getattr(geom, "geoms", []):
                    px, py = part.exterior.xy
                    ax.fill(px, py, color=_rgb_to_mpl(LAND_RGB), zorder=1)
        except Exception:
            continue
    # boundaries
    try:
        shp.boundary.plot(ax=ax, edgecolor=_rgb_to_mpl(BORDER_RGB), linewidth=BORDER_W, zorder=3)
    except Exception:
        pass


def render_one(
    dataset_name: str,
    interval: str,
    base_tif_noext: str,
    out_dir: str,
    read_from_s3: bool,
    # display controls
    target_crs: str,
    panel_w: float,
    panel_h: float,
    dpi: int,
    proj_scale: float,
    resampling: str,
    interpolation: str,
    # stretch / units
    low_p: float,
    high_p: float,
    symmetric: bool,
    unit_scale: str,
    log_emissions: bool,
    # cartography
    no_borders: bool,
    shp_gdf: gpd.GeoDataFrame | None,
):
    os.makedirs(out_dir, exist_ok=True)

    # Choose recipe & palette
    mode, legend_root = _choose_recipe(dataset_name)
    palette = NET_PALETTE if mode == "diverging" else (EMISSIONS_PALETTE if mode == "emissions" else REMOVALS_PALETTE)
    colors_mpl = _rgb_palette_to_mpl(palette)
    cmap = LinearSegmentedColormap.from_list("custom", colors_mpl)
    cmap.set_bad(_rgb_to_mpl(OCEAN_RGB))  # masked values show background

    # Units
    unit_scale = unit_scale.lower()
    scale_div = {"kt": 1e3, "mt": 1e6, "gt": 1e9}[unit_scale]
    unit_label = unit_scale

    base_path = base_tif_noext + ".tif"
    if read_from_s3 and base_path.startswith("s3://"):
        base_path = _gdalize_s3_url(base_path)

    with rasterio.open(base_path) as src:
        out_px_w = max(512, int(panel_w * dpi * proj_scale))
        out_px_h = max(256, int(panel_h * dpi * proj_scale))
        out_px_w = min(out_px_w, 12000)
        out_px_h = min(out_px_h, 8000)

        with _open_display_vrt(
            src_ds=src,
            dst_crs=target_crs,
            out_px_w=out_px_w,
            out_px_h=out_px_h,
            resampling=resampling,
            dst_nodata=0,
        ) as vrt:
            band = vrt.read(1, out_dtype="float32")
            data = band.astype(np.float64) / scale_div

            if log_emissions and mode == "emissions":
                data = np.where(data > 0, np.log10(1.0 + data), 0.0)

            masked = np.ma.masked_where(data == 0.0, data)

            vmin, vmax, vcenter = _compute_stretch(masked.filled(0.0), mode, low_p, high_p, symmetric)
            if mode == "diverging":
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0 if vcenter is not None else 0.0, vmax=vmax)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)

            fig, ax = plt.subplots(figsize=(panel_w, panel_h))
            ax.set_facecolor(_rgb_to_mpl(OCEAN_RGB))
            ax.set_xticks([]); ax.set_yticks([])

            if not no_borders and shp_gdf is not None:
                _draw_borders(ax, shp_gdf)

            left, bottom, right, top = array_bounds(vrt.height, vrt.width, vrt.transform)
            extent = [left, right, bottom, top]

            img = ax.imshow(
                masked,
                cmap=cmap,
                norm=norm,
                extent=extent,
                origin="upper",
                interpolation=IMSHOW_INTERP.get(interpolation, "bilinear"),
                zorder=2,
            )

            _legend(fig, img, mode, vmin, vmax, vcenter, legend_root, unit_label)

            label = interval
            ax.text(0.5, 0.07, str(label), transform=ax.transAxes,
                    ha="center", va="top", fontsize=18, weight="bold", color="black")

            out_base = Path(out_dir) / f"{dataset_name}__{interval}"
            out_main = f"{out_base}.jpeg"
            plt.savefig(out_main, dpi=dpi, bbox_inches="tight", pad_inches=0)

            ax.text(0.98, 0.04, PRES_TEXT, transform=ax.transAxes, fontsize=7,
                    ha="right", va="top", color="black")
            out_pres = f"{out_base}__for_pres.jpeg"
            plt.savefig(out_pres, dpi=dpi, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

    return True


def display_main(
    date_tag: str,
    read_from_s3: bool,
    run_name: str,
    datasets: List[str],
    intervals: List[str],
    target_deg: float,
    outputs_base: str,
    # display controls
    panel_w: float,
    panel_h: float,
    dpi: int,
    proj_scale: float,
    resampling: str,
    interpolation: str,
    # stretch / units
    low_p: float,
    high_p: float,
    symmetric: bool,
    unit_scale: str,
    log_emissions: bool,
    # cartography
    no_borders: bool,
):
    """
    Render all requested datasets/intervals for a given target resolution label.
    """
    res_label = deg_to_label(target_deg)
    logging.info(f"Rendering for res='{res_label}', outputs_base='{outputs_base}'")

    # Optional shapefile
    shp_gdf = None
    if not no_borders:
        try:
            shp_path = SHP_REPROJ if Path(SHP_REPROJ).exists() else SHP_ORIG
            shp_gdf = gpd.read_file(shp_path).to_crs(ROBINSON_CRS)
        except Exception as e:
            logging.warning(f"Could not read shapefile ({e}); continuing without borders.")
            shp_gdf = None

    for dataset in datasets:
        dataset_out = _density_to_pixel_name(dataset)
        for interval in intervals:
            # Global mosaic directory for this dataset/interval
            global_dir = posixpath.join(
                outputs_base.rstrip("/"),
                f"{res_label}_output_aggregation",
                dataset,
                interval,
                ""
            )
            expected_basename = f"{res_label}_global__{dataset_out}_{interval}"

            # Resolve the actual raster (handles timestamped suffixes)
            base_noext = _find_global_tif(global_dir, expected_basename, read_from_s3)

            # Stage locally, then upload to the final S3 path (or copy to local)
            final_display_dir = posixpath.join(global_dir, "display", interval, dataset)
            local_display_dir = Path("/tmp/afolu_global_maps_display") / res_label / interval / dataset
            local_display_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"Rendering {dataset} {interval}")
            render_one(
                dataset_name=dataset,
                interval=interval,
                base_tif_noext=base_noext,
                out_dir=str(local_display_dir),
                read_from_s3=read_from_s3,
                target_crs=ROBINSON_CRS,
                panel_w=panel_w,
                panel_h=panel_h,
                dpi=dpi,
                proj_scale=proj_scale,
                resampling=resampling,
                interpolation=interpolation,
                low_p=low_p,
                high_p=high_p,
                symmetric=symmetric,
                unit_scale=unit_scale,
                log_emissions=log_emissions,
                no_borders=no_borders,
                shp_gdf=shp_gdf,
            )

            # Upload staged files to S3 (or mirror locally)
            if _is_s3_path(final_display_dir):
                _upload_dir_to_s3(local_display_dir, final_display_dir)
                logging.info("Uploaded display files to %s", final_display_dir)
            else:
                Path(final_display_dir).mkdir(parents=True, exist_ok=True)
                for f in local_display_dir.rglob("*"):
                    if f.is_file():
                        dst = Path(final_display_dir) / f.relative_to(local_display_dir)
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(f.read_bytes())
                logging.info("Wrote display files to %s", final_display_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_list(arg: str) -> List[str]:
    return [x.strip() for x in arg.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Render Robinson-projected JPEGs (and GIFs) from global mosaics without huge memory use."
    )
    parser.add_argument("--date_tag", default="20250825", help="Informational tag; not used to locate mosaics.")
    parser.add_argument("--read_from_s3", action="store_true", help="Read COGs directly via /vsis3/ (no local download).")
    parser.add_argument("--run_name", default="ogh_standard_model", help="Run name used when aggregating.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                        help="Comma-separated dataset names to render.")
    parser.add_argument("--intervals", default=",".join(DEFAULT_INTERVALS),
                        help="Comma-separated intervals to render (e.g., 2021_2024).")
    parser.add_argument("--target_deg", type=float, default=DEFAULT_TARGET_DEG,
                        help="Target-degree label that mosaics were aggregated to (e.g., 0.04, 0.01).")
    parser.add_argument("--outputs_base", default=DEFAULT_OUTPUTS_BASE,
                        help="Root under which mosaics were saved (…/organic_soils/outputs).")

    # cartography & quality
    parser.add_argument("--no_borders", action="store_true", help="Skip country borders for a cleaner/faster render.")
    parser.add_argument("--proj_scale", type=float, default=2.0,
                        help="Oversampling factor for smoother reprojection (1.0–2.5 recommended).")
    parser.add_argument("--resampling", choices=tuple(RESAMPLING_MAP.keys()), default="cubic",
                        help="WarpedVRT resampling during reprojection/downsampling.")
    parser.add_argument("--low_p", type=float, default=2.0, help="Lower percentile for stretch (e.g., 2).")
    parser.add_argument("--high_p", type=float, default=98.0, help="Upper percentile for stretch (e.g., 98).")
    parser.add_argument("--symmetric", dest="symmetric", action="store_true",
                        help="Symmetric diverging stretch (center=0).")
    parser.add_argument("--no_symmetric", dest="symmetric", action="store_false")
    parser.set_defaults(symmetric=True)
    parser.add_argument("--log_emissions", action="store_true",
                        help="Use log10(1+x) stretch for strictly-positive emissions layers.")
    parser.add_argument("--dpi", type=int, default=DPI_DEFAULT)
    parser.add_argument("--panel_width", type=float, default=PANEL_DIMS[0])
    parser.add_argument("--panel_height", type=float, default=PANEL_DIMS[1])
    parser.add_argument("--unit_scale", choices=("kt", "Mt", "Gt"), default="kt",
                        help="Legend unit scale (kt, Mt, or Gt per year).")
    parser.add_argument("--interpolation", choices=tuple(IMSHOW_INTERP.keys()), default="bilinear",
                        help="Matplotlib image interpolation.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    display_main(
        date_tag=args.date_tag,
        read_from_s3=args.read_from_s3,
        run_name=args.run_name,
        datasets=_parse_list(args.datasets),
        intervals=_parse_list(args.intervals),
        target_deg=args.target_deg,
        outputs_base=args.outputs_base,
        panel_w=args.panel_width,
        panel_h=args.panel_height,
        dpi=args.dpi,
        proj_scale=args.proj_scale,
        resampling=args.resampling,
        interpolation=args.interpolation,
        low_p=args.low_p,
        high_p=args.high_p,
        symmetric=args.symmetric,
        unit_scale=args.unit_scale,
        log_emissions=args.log_emissions,
        no_borders=args.no_borders,
    )


if __name__ == "__main__":
    main()
