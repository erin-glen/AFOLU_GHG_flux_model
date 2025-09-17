# -*- coding: utf-8 -*-
"""
Stage B: High-quality rendering of global mosaics into Robinson-projected JPEGs/GIFs.

Improvements:
- Higher reprojection resolution via --proj_scale (default 2.0).
- Resampling choice: --resampling nearest|bilinear|cubic (default bilinear).
- Robust color stretch with --low_p/--high_p (default 2/98).
- Symmetric diverging normalization around 0 for net maps (--symmetric).
- Optional log scale for emissions (--log_emissions).
- Higher default DPI (450) and customizable panel size.

Examples
--------
python -m src.scripts.postprocessing.render_global_maps \
  --date_tag 20250825 \
  --run_name ogh_standard_model \
  --target_deg 0.04 \
  --read_from_s3 \
  --no_borders \
  --proj_scale 3.0 \
  --resampling cubic \
  --low_p 2 --high_p 98 --symmetric \
  --dpi 600 --panel_width 14 --panel_height 7 \
  --unit_scale kt
"""
from __future__ import annotations

import argparse
import math
import os
import posixpath
import re
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import boto3
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from affine import Affine

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from PIL import Image
import geopandas as gpd

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import log_utilities as lu

# ---------- safe constants ----------
def _cn(name: str, default):
    return getattr(cn, name, default)

_S3_CLIENT = _cn("s3_client", boto3.client("s3"))

# ---------- defaults ----------
DEFAULT_TARGET_DEG = 0.04
INVENTORY_PERIODS = ["2021_2024"]
DATA_TYPES = [
    "burned_total_Mg_CO2e_pixel_yr",
    "drained_total_Mg_CO2e_pixel_yr",
]
LOCAL_DISPLAY_ROOT = Path("/tmp/afolu_global_maps_display")

_BASE_URL_FALLBACK = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_7_0"
BASE_URL = _cn("outputs_path", _BASE_URL_FALLBACK)
OUTPUTS_BASE = posixpath.dirname(BASE_URL)

# ---------- helpers ----------
def deg_to_label(deg: float) -> str:
    s = f"{deg:.5f}".rstrip("0").rstrip(".")
    return f"{s.replace('.', '_')}deg"

def assert_grid_divides_world(target_deg: float):
    rows = round(180 / target_deg); cols = round(360 / target_deg)
    if not (np.isclose(rows*target_deg, 180.0) and np.isclose(cols*target_deg, 360.0)):
        raise ValueError("--target_deg must divide 180 and 360 evenly.")

def _split_s3(url: str) -> Tuple[str, str]:
    assert url.startswith("s3://")
    rest = url[len("s3://"):]
    b, _, k = rest.partition("/")
    return b, k

def _join_s3(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"

def _list_s3_keys(prefix_s3: str):
    bucket, prefix = _split_s3(prefix_s3)
    token = None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix)
        if token: kw["ContinuationToken"] = token
        resp = _S3_CLIENT.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            yield obj["Key"]
        if not resp.get("IsTruncated"): break
        token = resp.get("NextContinuationToken")

def _ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def _gdalize_s3_url(s3_url: str) -> str:
    return "/vsis3/" + s3_url[len("s3://"):] if s3_url.startswith("s3://") else s3_url

def _strip_ext(path_noext_or_with_ext: str) -> str:
    return path_noext_or_with_ext[:-4] if path_noext_or_with_ext.lower().endswith(".tif") else path_noext_or_with_ext

def _is_s3_path(path: str) -> bool:
    return path.startswith("s3://")

def _upload_dir_to_s3(local_dir: Path, s3_prefix: str):
    bucket, prefix = _split_s3(s3_prefix)
    for f in local_dir.rglob("*"):
        if not f.is_file(): continue
        if any(part.startswith("_reproj_cache") for part in f.parts): continue
        rel = f.relative_to(local_dir).as_posix()
        key = f"{prefix.rstrip('/')}/{rel}"
        _S3_CLIENT.upload_file(str(f), bucket, key)

def _find_global_tif(global_dir: str, expected_basename: str, read_from_s3: bool) -> str:
    if read_from_s3:
        bucket, prefix = _split_s3(global_dir)
        exact = f"{prefix.rstrip('/')}/{expected_basename}.tif"
        try:
            _S3_CLIENT.head_object(Bucket=bucket, Key=exact)
            return _strip_ext(_join_s3(bucket, exact))
        except Exception:
            pass
        start = f"{prefix.rstrip('/')}/{expected_basename}"
        candidates = [k for k in _list_s3_keys(global_dir) if k.startswith(start) and k.endswith(".tif")]
        if not candidates:
            raise FileNotFoundError(f"No global mosaic found under {global_dir} with base '{expected_basename}'.")
        chosen = sorted(candidates)[-1]
        return _strip_ext(_join_s3(bucket, chosen))
    else:
        exact = Path(global_dir) / f"{expected_basename}.tif"
        if exact.exists(): return _strip_ext(str(exact))
        prefixp = Path(global_dir) / expected_basename
        matches = sorted([str(p) for p in prefixp.parent.glob(prefixp.name + "*.tif")])
        if not matches:
            raise FileNotFoundError(f"No global mosaic found in {global_dir} with base '{expected_basename}'.")
        return _strip_ext(matches[-1])

def build_download_upload_dict(
    pixel_resolution: str,
    run_name: str,
    base_url: str,
    output_date: str,
    outputs_base: str,
    data_types: list[str] | None = None,
    inventory_periods: list[str] | None = None,
) -> dict:
    data_types = data_types or DATA_TYPES
    inventory_periods = inventory_periods or INVENTORY_PERIODS
    d = {}
    for period in inventory_periods:
        for dataset in data_types:
            src_dir = (
                f"{base_url}/{dataset}/{run_name}/"
                f"five_year_intervals/{period}/{pixel_resolution}/{output_date}/"
            )
            out_dir = f"{outputs_base}/{{res_label}}_output_aggregation/{dataset}/{period}/"
            d[f"{dataset}__{period}"] = {
                "src_dir": src_dir,
                "src_pattern": f"__{dataset}__{period}.tif",
                "global_dir": out_dir,
            }
    return d

# ---------- reprojection with scale ----------
_RESAMPLING_MAP = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
}

def _reproject_if_needed(out_tif: str, in_tif: str, target_crs: str,
                         scale: float = 2.0, resampling: str = "bilinear") -> str:
    """
    Reproject (and optionally oversample) to target_crs; set nodata=0 in output.
    scale=2.0 doubles width/height (smaller pixel size => smoother display).
    """
    if os.path.exists(out_tif):
        return out_tif
    rsmp = _RESAMPLING_MAP.get(resampling, Resampling.bilinear)

    with rasterio.open(in_tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        # upscale by 'scale'
        width = int(math.ceil(width * scale))
        height = int(math.ceil(height * scale))
        transform = transform * Affine.scale(1.0/scale, 1.0/scale)

        profile = src.profile.copy()
        compression = profile.get("compress", "deflate")
        profile.update(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            nodata=0,
            compress=compression,
            tiled=True,
            blockxsize=min(512, width),
            blockysize=min(512, height),
        )
        with rasterio.open(out_tif, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=rsmp,
            )
    return out_tif

# ---------- shapefile ----------
def _load_shapefile_or_none(no_borders: bool) -> Optional[gpd.GeoDataFrame]:
    if no_borders:
        return None
    rob_crs = _cn("Robinson_crs", "ESRI:54030")
    reproj_path = _cn("reprojected_shapefile_path", None)
    orig_path = _cn("original_shapefile_path", None)
    try:
        if reproj_path and Path(reproj_path).exists():
            shp = gpd.read_file(reproj_path)
            if shp.crs is None or str(shp.crs) != rob_crs:
                shp = shp.to_crs(rob_crs)
            return shp
        if orig_path and Path(orig_path).exists():
            return gpd.read_file(orig_path).to_crs(rob_crs)
    except Exception:
        pass
    return None

def _plot_country_layers(ax, shapefile):
    if shapefile is None:
        return
    land_bkgrnd = _cn("land_bkgrnd", (245, 245, 245))
    boundary_color = _cn("boundary_color", (150, 150, 150))
    boundary_width = _cn("boundary_width", 0.2)
    for geom in shapefile.geometry:
        try:
            if hasattr(geom, "exterior"):
                x, y = geom.exterior.xy
                ax.fill(x, y, color=tuple(v/255 for v in land_bkgrnd), zorder=1)
            else:
                for part in geom.geoms:
                    px, py = part.exterior.xy
                    ax.fill(px, py, color=tuple(v/255 for v in land_bkgrnd), zorder=1)
        except Exception:
            continue
    shapefile.boundary.plot(ax=ax, edgecolor=tuple(v/255 for v in boundary_color),
                            linewidth=boundary_width, zorder=3)

# ---------- color & normalization ----------
def _palette_brbg():
    return [
        (0, 60, 48), (1, 102, 94), (53, 151, 143), (128, 205, 193), (199, 234, 229),
        (246, 232, 195), (223, 194, 125), (191, 129, 45), (140, 81, 10), (84, 48, 5)
    ]

def _to_mpl(rgb_list):
    return [tuple(v/255 for v in rgb) for rgb in rgb_list]

def _robust_range(arr, low_p: float, high_p: float):
    return (np.percentile(arr, low_p), np.percentile(arr, high_p)) if arr.size else (0.0, 0.0)

def _choose_recipe(dataset_name: str):
    lower = dataset_name.lower()
    if "net" in lower:
        return "diverging", "Net greenhouse gas flux"
    if "removal" in lower or "sink" in lower:
        return "removals", "Gross CO$_2$ removals"
    return "emissions", "Gross greenhouse gas emissions"

def _make_norm_and_range(data: np.ndarray, mode: str,
                         low_p: float, high_p: float,
                         symmetric: bool, log_emissions: bool):
    """
    Returns (masked_data, norm, vmin, vcenter, vmax)
    """
    # mask zeros for percentiles
    nz = data[data != 0]

    if mode == "diverging":
        # symmetric around 0 (recommended)
        if symmetric:
            v = np.percentile(np.abs(nz), high_p) if nz.size else 0.0
            vmin, vcenter, vmax = -v, 0.0, v
        else:
            vmin = np.percentile(nz, low_p) if nz.size else 0.0
            vmax = np.percentile(nz, high_p) if nz.size else 0.0
            vcenter = 0.0
        masked = np.ma.masked_where(data == 0, data)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        return masked, norm, vmin, vcenter, vmax

    if mode == "emissions":
        pos = data[data > 0]
        if pos.size == 0:
            vmin = vmax = 0.0
            masked = np.ma.masked_all_like(data)
            norm = Normalize(vmin=0, vmax=1)
            return masked, norm, vmin, None, vmax
        vmin = max(np.percentile(pos, low_p), np.finfo(float).eps)
        vmax = np.percentile(pos, high_p)
        masked = np.ma.masked_where(data <= 0, data)
        norm = LogNorm(vmin=vmin, vmax=vmax) if log_emissions else Normalize(vmin=vmin, vmax=vmax)
        return masked, norm, vmin, None, vmax

    # removals (negative)
    neg = data[data < 0]
    if neg.size == 0:
        vmin = vmax = 0.0
        masked = np.ma.masked_all_like(data)
        norm = Normalize(vmin=0, vmax=1)
        return masked, norm, vmin, None, vmax
    vmin = np.percentile(neg, high_p)  # closest to zero (less negative)
    vmax = np.percentile(neg, low_p)   # most negative
    masked = np.ma.masked_where(data >= 0, data)
    norm = Normalize(vmin=vmin, vmax=vmax)
    return masked, norm, vmin, None, vmax

# ---------- drawing ----------
def _legend(fig, img, mode, vmin, vcenter, vmax, unit_suffix: str, fontsize: int):
    cax_dims = _cn("colorbar_dimensions", [0.14, 0.17, 0.02, 0.13])
    cax = fig.add_axes(cax_dims)
    cb = plt.colorbar(img, cax=cax, orientation="vertical")

    if mode == "diverging":
        cb.set_ticks([vmin, vcenter, vmax])
        cb.set_ticklabels([f"{vmin:,.0f}", "0", f"{vmax:,.0f}"], fontsize=fontsize)
    else:
        locator = MaxNLocator(nbins=4)
        ticks = locator.tick_values(vmin, vmax)
        cb.set_ticks([ticks[0], ticks[-1]])
        cb.set_ticklabels([f"{ticks[0]:,.0f}", f"{ticks[-1]:,.0f}"], fontsize=fontsize)

    cax.text(0, 1.1, unit_suffix, fontsize=fontsize, ha="left", va="bottom",
             transform=cax.transAxes)

def _draw_frame(ax, extent, masked, cmap, norm, ocean_rgb, interpolation: str = "none"):
    ax.set_facecolor(tuple(v/255 for v in ocean_rgb))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xticklabels([]); ax.set_yticklabels([])
    img = ax.imshow(masked, cmap=cmap, norm=norm, extent=extent, origin="upper",
                    zorder=2, interpolation=interpolation)
    return img

def _save_two_versions(ax, base_out: Path, label: str | int, dpi: int, pres_text: str):
    out_1 = f"{base_out}__{label}.jpeg"
    out_2 = f"{base_out}__{label}__for_pres.jpeg"
    ax.text(0.5, 0.07, str(label), transform=ax.transAxes, ha="center", va="top",
            fontsize=18, weight="bold", color="black")
    plt.savefig(out_1, dpi=dpi, bbox_inches="tight", pad_inches=0)
    ax.text(0.98, 0.04, pres_text, transform=ax.transAxes, fontsize=7,
            ha="right", va="top", color="black")
    plt.savefig(out_2, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()
    return [out_1, out_2]

def _gif_from_frames(base_out: Path, first_label, last_label, frames: list[str]):
    if not frames: return []
    imgs = [Image.open(p) for p in frames]
    out_fast = f"{base_out}__{first_label}_{last_label}__fast.gif"
    out_slow = f"{base_out}__{first_label}_{last_label}__slow.gif"
    imgs[0].save(out_fast, save_all=True, append_images=imgs[1:], duration=1000, loop=0)
    imgs[0].save(out_slow, save_all=True, append_images=imgs[1:], duration=2500, loop=0)
    return [out_fast, out_slow]

# ---------- display ----------
def render_one(
    dataset_name: str,
    interval: str,
    base_tif_noext: str,
    out_dir: str,
    palette_rgb,
    low_p: float,
    high_p: float,
    shapefile_gdf,
    out_name_base: str,
    years: list[int] | None,
    symmetric: bool,
    log_emissions: bool,
    proj_scale: float,
    resampling: str,
    target_crs: str,
    read_from_s3: bool,
    dpi: int,
    panel_w: float,
    panel_h: float,
    ocean_color,
    unit_scale: str,
    interpolation: str,
):
    logger = lu.setup_logging()
    _ensure_dir(out_dir)
    work_dir = Path(out_dir) / "_reproj_cache"
    _ensure_dir(work_dir)

    # s3 -> /vsis3/
    if read_from_s3 and base_tif_noext.startswith("s3://"):
        base_tif_noext = _gdalize_s3_url(base_tif_noext)

    colors = LinearSegmentedColormap.from_list("custom", _to_mpl(palette_rgb))

    mode, title_base = _choose_recipe(dataset_name)
    labels = years or [interval]

    pres_text = _cn("pres_text", f"Preliminary (subset)")
    ocean_rgb = ocean_color
    legend_fontsize = _cn("legend_fontsize", 9)
    unit_suffix = {
        "kt": "kt CO$_2$e yr$^{-1}$",
        "Mt": "Mt CO$_2$e yr$^{-1}$",
        "Gt": "Gt CO$_2$e yr$^{-1}$",
    }.get(unit_scale, "kt CO$_2$e yr$^{-1}$")
    divisor = {"kt": 1e3, "Mt": 1e6, "Gt": 1e9}.get(unit_scale, 1e3)

    created = []
    for label in labels:
        tif_unproj = base_tif_noext + ".tif"
        tif_proj = work_dir / (Path(base_tif_noext).name + f"_reproj_x{proj_scale}_{resampling}.tif")
        tif_proj = _reproject_if_needed(str(tif_proj), tif_unproj, target_crs, proj_scale, resampling)

        with rasterio.open(tif_proj) as src:
            data = src.read(1).astype(np.float64) / divisor
            bounds = src.bounds

        data = np.nan_to_num(data, nan=0.0)

        masked, norm, vmin, vcenter, vmax = _make_norm_and_range(
            data, mode, low_p, high_p, symmetric, log_emissions
        )

        fig, ax = plt.subplots(figsize=(panel_w, panel_h))
        _plot_country_layers(ax, shapefile_gdf)
        img = _draw_frame(ax,
                          [bounds.left, bounds.right, bounds.bottom, bounds.top],
                          masked, colors, norm, ocean_rgb, interpolation=interpolation)

        # Legend
        title_text = f"{title_base}\n{unit_suffix}"
        _legend(fig, img, mode, vmin, vcenter if vcenter is not None else 0, vmax,
                unit_suffix="", fontsize=legend_fontsize)
        # Title in upper-left outside colorbar
        ax.text(0.02, 0.97, title_text, transform=ax.transAxes, ha="left", va="top",
                fontsize=legend_fontsize+1)

        base_out = Path(out_dir) / out_name_base
        outs = _save_two_versions(ax, base_out, label, dpi, pres_text)
        created.extend(outs)

    # Optional GIF if multiple labels
    if len(labels) > 1 and isinstance(labels[0], int):
        created.extend(_gif_from_frames(Path(out_dir) / out_name_base, labels[0], labels[-1], created[-len(labels):]))

    logger.info(f"Rendered {dataset_name} {interval} ({len(created)} files)")
    return created

def display_main(
    date_tag: str,
    read_from_s3: bool,
    run_name: str,
    pixel_resolution: str,
    target_deg: float = DEFAULT_TARGET_DEG,
    no_borders: bool = False,
    proj_scale: float = 2.0,
    resampling: str = "bilinear",
    low_p: float = 2.0,
    high_p: float = 98.0,
    symmetric: bool = True,
    log_emissions: bool = False,
    dpi: int = 450,
    panel_width: float = 12.0,
    panel_height: float = 6.0,
    unit_scale: str = "kt",
    interpolation: str = "none",
):
    assert_grid_divides_world(target_deg)
    res_label = deg_to_label(target_deg)
    target_crs = _cn("Robinson_crs", "ESRI:54030")

    logger = lu.setup_logging_main()
    d = build_download_upload_dict(pixel_resolution, run_name, BASE_URL, date_tag, OUTPUTS_BASE,
                                   data_types=DATA_TYPES, inventory_periods=INVENTORY_PERIODS)

    shp = _load_shapefile_or_none(no_borders=no_borders)

    net_palette = _palette_brbg()
    removals_palette = net_palette[0:5]
    emissions_palette = net_palette[5:]
    ocean_color = _cn("ocean_color", (235, 235, 235))

    for key, items in d.items():
        dataset, interval = key.split("__")

        expected_dataset_out = dataset.replace("_ha_yr", "_pixel_yr").replace("_ha", "_pixel") \
            if (dataset.endswith("_ha") or dataset.endswith("_ha_yr")) else dataset
        expected_basename = f"{res_label}_global__{expected_dataset_out}_{interval}"
        global_dir = items["global_dir"].format(res_label=res_label)

        base_noext = _find_global_tif(global_dir, expected_basename, read_from_s3)

        final_display_dir = posixpath.join(global_dir, "display", interval, dataset)
        local_display_dir = LOCAL_DISPLAY_ROOT / res_label / interval / dataset
        _ensure_dir(local_display_dir)

        mode, _ = _choose_recipe(dataset)
        palette = net_palette if mode == "diverging" else (emissions_palette if mode == "emissions" else removals_palette)
        out_base = f"{dataset}__{interval}"

        created_local = render_one(
            dataset_name=dataset,
            interval=interval,
            base_tif_noext=base_noext,
            out_dir=str(local_display_dir),
            palette_rgb=palette,
            low_p=low_p,
            high_p=high_p,
            shapefile_gdf=shp,
            out_name_base=out_base,
            years=None,
            symmetric=symmetric,
            log_emissions=log_emissions,
            proj_scale=proj_scale,
            resampling=resampling,
            target_crs=target_crs,
            read_from_s3=read_from_s3,
            dpi=dpi,
            panel_w=panel_width,
            panel_h=panel_height,
            ocean_color=ocean_color,
            unit_scale=unit_scale,
            interpolation=interpolation,
        )

        if _is_s3_path(final_display_dir):
            _upload_dir_to_s3(local_display_dir, final_display_dir)
            logger.info(f"Uploaded {len(created_local)} files to {final_display_dir}")
        else:
            _ensure_dir(final_display_dir)
            for f in Path(local_display_dir).rglob("*"):
                if f.is_file():
                    dst = Path(final_display_dir) / f.relative_to(local_display_dir)
                    _ensure_dir(dst.parent)
                    dst.write_bytes(f.read_bytes())
            logger.info(f"Wrote display files to {final_display_dir}")

def main():
    p = argparse.ArgumentParser(description="Render high-quality global display maps.")
    p.add_argument("--date_tag", required=True)
    p.add_argument("--read_from_s3", action="store_true")
    p.add_argument("--run_name", default="ogh_standard_model")
    p.add_argument("-p", "--pixel_resolution", default="40000_pixels")
    p.add_argument("--target_deg", type=float, default=DEFAULT_TARGET_DEG)
    p.add_argument("--no_borders", action="store_true")
    p.add_argument("--proj_scale", type=float, default=2.0)
    p.add_argument("--resampling", choices=["nearest", "bilinear", "cubic"], default="bilinear")
    p.add_argument("--low_p", type=float, default=2.0)
    p.add_argument("--high_p", type=float, default=98.0)
    p.add_argument("--symmetric", action="store_true", default=True)
    p.add_argument("--no_symmetric", dest="symmetric", action="store_false")
    p.add_argument("--log_emissions", action="store_true")
    p.add_argument("--dpi", type=int, default=450)
    p.add_argument("--panel_width", type=float, default=12.0)
    p.add_argument("--panel_height", type=float, default=6.0)
    p.add_argument("--unit_scale", choices=["kt","Mt","Gt"], default="kt")
    p.add_argument("--interpolation", choices=["none","nearest","bilinear"], default="none")
    args = p.parse_args()

    display_main(
        date_tag=args.date_tag,
        read_from_s3=args.read_from_s3,
        run_name=args.run_name,
        pixel_resolution=args.pixel_resolution,
        target_deg=args.target_deg,
        no_borders=args.no_borders,
        proj_scale=args.proj_scale,
        resampling=args.resampling,
        low_p=args.low_p,
        high_p=args.high_p,
        symmetric=args.symmetric,
        log_emissions=args.log_emissions,
        dpi=args.dpi,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
        unit_scale=args.unit_scale,
        interpolation=args.interpolation,
    )

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Stage B: Render Robinson-projected JPEGs (and GIFs) from aggregated global rasters.

Improvements over earlier display stage
---------------------------------------
- Higher reprojection quality via --proj_scale (default 2.0) and --resampling (bilinear|cubic).
- Robust stretch: --low_p/--high_p (default 2/98 percentiles).
- Symmetric diverging normalization for net flux (--symmetric).
- Optional log scaling for emissions-only layers (--log_emissions).
- Higher default DPI (450) and configurable panel size (--panel_width/--panel_height).
- Unit scaling (--unit_scale kt|Mt|Gt) for legend readability.
- Optional --no_borders to skip drawing country outlines.

Examples
--------

# --- Display only ---
# Render JPEGs for existing 0.04° mosaics, reading directly from S3 via /vsis3:
python -m src.scripts.postprocessing.render_global_maps \
  --date_tag 20250825 \
  --run_name ogh_standard_model \
  --target_deg 0.01 \
  --read_from_s3 \
  --base_url s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_7_0
  

# Render at 0.01° (must match aggregated resolution), with smoother reprojection:
python -m src.scripts.postprocessing.render_global_maps \
  --date_tag 20250825 \
  --run_name ogh_standard_model \
  --target_deg 0.01 \
  --read_from_s3 \
  --proj_scale 3.0 \
  --resampling cubic \
  --dpi 600

# Skip drawing country boundaries and use symmetric color stretch:
python -m src.scripts.postprocessing.render_global_maps \
  --date_tag 20250825 \
  --run_name ogh_standard_model \
  --read_from_s3 \
  --no_borders \
  --symmetric

# Use log scaling for emissions-only layers (better low-value visibility):
python -m src.scripts.postprocessing.render_global_maps \
  --date_tag 20250825 \
  --run_name ogh_standard_model \
  --read_from_s3 \
  --log_emissions \
  --base_url s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_7_0

# Change output units (Mt instead of kt) and panel size:
python -m src.scripts.postprocessing.render_global_maps \
  --date_tag 20250825 \
  --run_name ogh_standard_model \
  --read_from_s3 \
  --unit_scale Mt \
  --panel_width 14 \
  --panel_height 7
"""
