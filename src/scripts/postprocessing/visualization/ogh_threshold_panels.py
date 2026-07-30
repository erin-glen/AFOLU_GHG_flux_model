"""Compare raw OGH organic-soil probability to its thresholded binary masks.

Builds a grid with one row per biome example; each row is:

    [ raw probability ] [ Low-area mask ] [ Baseline mask ] [ High-area mask ]

The raw panel shows the continuous OGH organic-soil probability surface (uint8,
0-100). The three binary panels show the organic-soil/other mask produced by the
model's per-biome probability cutoffs for the publication sensitivity scenarios:

  - Low estimate  -> ``low_area_threshold``  (HIGHER cutoff -> LESS organic-soil area)
  - Baseline      -> ``operational_threshold``
  - High estimate -> ``high_area_threshold`` (LOWER cutoff -> MORE organic-soil area)

This mirrors the model's thresholding in
``src/scripts/core_model/0_drainage_emissions_model.py`` (``layers["peat"] =
peat_layer >= thresh``). Per ``parse_biome_thresholds`` there, the CSV
thresholds are expressed on a 0-1 scale and rescaled to the raster's native
0-100 range (``thresh * 100``); this script applies the same rescaling.

A light Natural Earth basemap (rivers, lakes, country borders) and a per-row
locator inset are drawn for context, and each view auto-crops to the
organic-soil-rich part of its tile. Row heights follow each view's true
geographic aspect, so equatorial rows are wider and high-latitude rows taller.

Default example (one row per biome)::

    python -m src.scripts.postprocessing.visualization.ogh_threshold_panels

Custom examples (``tile:biome`` comma-separated, top to bottom)::

    python -m src.scripts.postprocessing.visualization.ogh_threshold_panels \
        --examples "60N_070E:boreal,00N_010E:tropical,60N_010W:temperate"

Notes
-----
* Tiles are read at a decimated resolution (``--max-dim``) for display. Binary
  panels threshold that same decimated array so all panels in a row are mutually
  consistent; production masks threshold at native 100 m, so per-view areas here
  are illustrative.
* ``nodata`` is 0 in the source raster, so genuine zero-probability pixels and
  off-tile pixels are both treated as unmapped.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds

from src.scripts.zonal_statistics.pub_scripts import pub_common as pc
from src.scripts.utilities import local_output_paths as lop

DEFAULT_S3_BASE = (
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/"
    "peat_mask/OGH/tiles_unthresholded/20260513"
)
DEFAULT_TILE_SUFFIX = "_ogh_unthresholded_mask.tif"
DEFAULT_LOCAL_DIR = "C:/tmp"

DEFAULT_THRESHOLD_CSV = (
    "docs/organic_soil_threshold_profiles/"
    "20260513_mixed_boreal_f1_temperate_f1_5_tropical_f2.csv"
)

# Default: one row per biome (top to bottom). Each view is a fixed SQUARE window
# centred on the peatland -- square => identical panel size at every latitude.
DEFAULT_EXAMPLES = [
    {"tile": "60N_070E", "biome": "boreal",    "center": (74.5, 57.9), "lat_span": 4.2},
    {"tile": "60N_010E", "biome": "temperate", "center": (15.0, 53.0), "lat_span": 4.2},
    {"tile": "00N_100E", "biome": "tropical",  "center": (103.5, -2.1), "lat_span": 4.2},
]
REGION_NAMES = {
    "60N_070E": "West Siberia",
    "00N_100E": "Sumatra",
    "60N_010E": "N. Germany & Poland",
}

# Scenario -> (CSV column, key, human label). Order Low -> Baseline -> High.
SCENARIOS: Sequence[tuple[str, str, str]] = (
    ("low_area_threshold", "low", "Low estimate"),
    ("operational_threshold", "baseline", "Baseline"),
    ("high_area_threshold", "high", "High estimate"),
)

EARTH_RADIUS_M = 6_371_007.181

PROB_CMAP = "viridis"
OS_COLOR = "#1F6F3E"
LAND_COLOR = "#F4F3EE"
RIVER_COLOR = "#5B8DB8"
LAKE_FACE = "#CFE0EF"
LAKE_EDGE = "#9DBBD6"
BORDER_COLOR = "#BFBFBF"
CUTOFF_MARK = "#D7263D"

NE_CACHE = "C:/tmp/ne_cache"
NE_LAYERS = {
    "rivers": "ne_10m_rivers_lake_centerlines",
    "lakes": "ne_10m_lakes",
    "countries": "ne_110m_admin_0_countries",
}
NE_URLS = {
    "ne_10m_rivers_lake_centerlines": [
        "https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_lake_centerlines.zip",
        "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_rivers_lake_centerlines.zip",
    ],
    "ne_10m_lakes": [
        "https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip",
        "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_lakes.zip",
    ],
    "ne_110m_admin_0_countries": [
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip",
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip",
    ],
}


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
def _resolve_tile_path(tile_id: str, local_dir: str, s3_base: str) -> str:
    """Return a LOCAL path to the tile, downloading from S3 once if needed.

    Decimated reads of these 40000x40000, no-overview, 400px-block tiles over
    /vsis3 can crash GDAL's curl layer on Windows; caching locally is stable.
    """
    fname = f"{tile_id}{DEFAULT_TILE_SUFFIX}"
    local_path = os.path.join(local_dir, fname).replace("\\", "/")
    if os.path.isfile(local_path):
        return local_path
    if not s3_base.startswith("s3://"):
        alt = os.path.join(s3_base, fname).replace("\\", "/")
        if os.path.isfile(alt):
            return alt
        raise FileNotFoundError(f"Tile not found locally: {local_path} or {alt}")
    import boto3

    rest = s3_base[len("s3://"):].rstrip("/")
    bucket, _, prefix = rest.partition("/")
    key = f"{prefix}/{fname}" if prefix else fname
    os.makedirs(local_dir, exist_ok=True)
    print(f"Caching s3://{bucket}/{key} -> {local_path}")
    boto3.client("s3").download_file(bucket, key, local_path)
    return local_path


def _load_biome_thresholds(csv_path: str, biome: str) -> dict[str, float]:
    """{scenario_key: cutoff on 0-100 scale}. CSV is 0-1; model rescales *100."""
    df = pd.read_csv(csv_path)
    if "biome" not in df.columns:
        raise KeyError(f"CSV missing 'biome' column. Found: {list(df.columns)}")
    rows = df.loc[df["biome"].astype(str).str.lower() == biome.lower()]
    if rows.empty:
        raise ValueError(f"Biome {biome!r} not in {csv_path}; "
                         f"have {sorted(df['biome'].astype(str).unique())}")
    row = rows.iloc[0]
    out: dict[str, float] = {}
    for col, key, _label in SCENARIOS:
        if col not in row or pd.isna(row[col]):
            raise KeyError(f"CSV missing/empty {col!r} for biome {biome!r}")
        out[key] = float(row[col]) * 100.0
    return out


def _read_window(path, bbox, max_dim, resampling):
    with rasterio.open(path) as ds:
        if bbox is not None:
            window = from_bounds(*bbox, ds.transform).intersection(
                Window(0, 0, ds.width, ds.height))
        else:
            window = Window(0, 0, ds.width, ds.height)
        win_h, win_w = int(round(window.height)), int(round(window.width))
        scale = max(1, int(np.ceil(max(win_h, win_w) / max_dim)))
        out_h, out_w = max(1, win_h // scale), max(1, win_w // scale)
        arr = ds.read(1, window=window, out_shape=(out_h, out_w),
                      resampling=resampling).astype("float32")
        l, b, r, t = rasterio.windows.bounds(window, ds.transform)
        nodata = 0 if ds.nodata is None else int(ds.nodata)
    arr[arr == nodata] = np.nan
    return arr, (l, r, b, t)


def _square_bbox(center, lat_span):
    """Square view window (l,b,r,t) centred on (lon,lat). lon_span is widened by
    1/cos(lat) so the panel renders square once the geographic aspect is applied."""
    clon, clat = center
    lon_span = lat_span / max(np.cos(np.radians(clat)), 1e-3)
    return (clon - lon_span / 2, clat - lat_span / 2,
            clon + lon_span / 2, clat + lat_span / 2)


def _auto_crop_bbox(prob, extent, cutoff, min_frac=0.02, margin_deg=0.4):
    l, r, b, t = extent
    valid = ~np.isnan(prob)
    pos = valid & (prob >= cutoff)
    row_frac = pos.sum(1) / np.maximum(valid.sum(1), 1)
    col_frac = pos.sum(0) / np.maximum(valid.sum(0), 1)
    rows = np.where(row_frac >= min_frac)[0]
    cols = np.where(col_frac >= min_frac)[0]
    if rows.size == 0 or cols.size == 0:
        return None
    nrows, ncols = prob.shape
    lat = t + (rows / nrows) * (b - t)
    lon = l + (cols / ncols) * (r - l)
    return (max(l, lon.min() - margin_deg), max(b, lat.min() - margin_deg),
            min(r, lon.max() + margin_deg), min(t, lat.max() + margin_deg))


# --------------------------------------------------------------------------- #
# Basemap
# --------------------------------------------------------------------------- #
def _ensure_ne_layer(stem: str) -> Optional[str]:
    os.makedirs(NE_CACHE, exist_ok=True)
    dst = os.path.join(NE_CACHE, f"{stem}.zip")
    if not os.path.isfile(dst):
        for url in NE_URLS[stem]:
            try:
                urllib.request.urlretrieve(url, dst)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  basemap fetch failed ({stem}): {type(exc).__name__}")
    return dst if os.path.isfile(dst) else None


def _load_basemap():
    try:
        import geopandas as gpd
    except Exception:  # noqa: BLE001
        print("  geopandas unavailable; skipping basemap.")
        return {}
    bm = {}
    for name, stem in NE_LAYERS.items():
        path = _ensure_ne_layer(stem)
        if path:
            try:
                bm[name] = gpd.read_file(f"zip://{path}")
            except Exception as exc:  # noqa: BLE001
                print(f"  could not read {stem}: {type(exc).__name__}")
    return bm


def _draw_basemap(ax, bm, extent, *, on_top: bool):
    # aspect=None stops geopandas forcing its own aspect; we set it last.
    l, r, b, t = extent
    if not bm:
        return
    if "countries" in bm and not on_top:
        sub = bm["countries"].cx[l:r, b:t]
        if len(sub):
            sub.boundary.plot(ax=ax, color=BORDER_COLOR, lw=0.5, zorder=1.5, aspect=None)
    if "lakes" in bm:
        sub = bm["lakes"].cx[l:r, b:t]
        if len(sub):
            if on_top:
                sub.boundary.plot(ax=ax, color="white", lw=0.5, alpha=0.7, zorder=4, aspect=None)
            else:
                sub.plot(ax=ax, facecolor=LAKE_FACE, edgecolor=LAKE_EDGE, lw=0.4, zorder=2, aspect=None)
    if "rivers" in bm:
        sub = bm["rivers"].cx[l:r, b:t]
        if len(sub):
            color = "white" if on_top else RIVER_COLOR
            sub.plot(ax=ax, color=color, lw=0.6, alpha=0.7 if on_top else 0.9, zorder=4, aspect=None)


def _place_cutoff_labels(cax, items):
    """Mark cutoffs on a vertical 0-100 colorbar (horizontal lines) and label
    them to the right, spreading labels in y with leader lines so close cutoffs
    (e.g. tropical 15/16/20%) don't overlap."""
    items = sorted(items)
    min_sep = 13.0  # data units (0-100) between adjacent label centres
    lab_y = [v for v, _ in items]
    for i in range(1, len(lab_y)):
        if lab_y[i] - lab_y[i - 1] < min_sep:
            lab_y[i] = lab_y[i - 1] + min_sep
    over = lab_y[-1] - 100.0
    if over > 0:
        lab_y = [y - over for y in lab_y]
    for (val, label), ly in zip(items, lab_y):
        cax.axhline(val, color=CUTOFF_MARK, lw=1.1)
        cax.annotate(label, xy=(1.0, val), xycoords=("axes fraction", "data"),
                     xytext=(1.8, ly), textcoords=("axes fraction", "data"),
                     ha="left", va="center", fontsize=6.2, color=CUTOFF_MARK,
                     arrowprops=dict(arrowstyle="-", color=CUTOFF_MARK, lw=0.5,
                                     shrinkA=0, shrinkB=0))


# --------------------------------------------------------------------------- #
# Area + stats + axis styling
# --------------------------------------------------------------------------- #
def _row_pixel_area_ha(extent, shape) -> np.ndarray:
    l, r, b, t = extent
    nrows, ncols = shape
    dlon_rad = np.radians((r - l) / ncols)
    lat_edges = np.linspace(t, b, nrows + 1)
    sin_edges = np.sin(np.radians(lat_edges))
    band = EARTH_RADIUS_M ** 2 * dlon_rad * (sin_edges[:-1] - sin_edges[1:])
    return np.abs(band) / 1e4


def _stats(prob, cutoff, row_area_ha):
    valid = ~np.isnan(prob)
    pos = valid & (prob >= cutoff)
    area = np.broadcast_to(row_area_ha[:, None], prob.shape)
    os_mha = float(area[pos].sum() / 1e6)
    land_mha = float(area[valid].sum() / 1e6)
    pct = (100.0 * os_mha / land_mha) if land_mha else 0.0
    return np.where(pos, 1.0, np.nan), os_mha, pct, land_mha


def _view_aspect(extent):
    l, r, b, t = extent
    return 1.0 / max(np.cos(np.radians(0.5 * (b + t))), 1e-3)


def _style_map_axes(ax, extent, show_x=True, show_y=True):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(_view_aspect(extent))
    ax.grid(False)
    ax.set_facecolor(LAND_COLOR)
    ax.set_xticks(np.linspace(extent[0], extent[1], 3) if show_x else [])
    ax.set_yticks(np.linspace(extent[2], extent[3], 3) if show_y else [])
    ax.tick_params(labelsize=6.5)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
COLUMN_HEADERS = ("Raw probability (%)", "Low estimate", "Baseline", "High estimate")
_OVERLAY_BBOX = dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.72, edgecolor="none")


def _render_row(fig, axes_row, ex, bm, *, top_row: bool, bottom_row: bool):
    prob, extent, thr = ex["prob"], ex["extent"], ex["thresholds"]
    biome, tile = ex["biome"], ex["tile"]
    row_area_ha = _row_pixel_area_ha(extent, prob.shape)

    prob_cmap = plt.get_cmap(PROB_CMAP).copy()
    prob_cmap.set_bad(alpha=0.0)
    os_cmap = ListedColormap([OS_COLOR])
    os_cmap.set_bad(alpha=0.0)

    region = REGION_NAMES.get(tile, tile)

    # Raw probability panel + per-row vertical colorbar (red = scenario cutoffs).
    ax0 = axes_row[0]
    im = ax0.imshow(prob, extent=extent, origin="upper", cmap=prob_cmap,
                    vmin=0, vmax=100, zorder=1)
    _draw_basemap(ax0, bm, extent, on_top=True)
    _style_map_axes(ax0, extent, show_x=bottom_row, show_y=False)
    if top_row:
        ax0.set_title(COLUMN_HEADERS[0], fontsize=10.5, weight="bold")
    # Biome/region as a rotated label on the left (one per row).
    ax0.annotate(f"{biome.title()} · {region}", xy=(-0.36, 0.5),
                 xycoords="axes fraction", rotation=90, ha="center", va="center",
                 fontsize=10, weight="bold", annotation_clip=False)
    cax = ax0.inset_axes([-0.17, 0.03, 0.05, 0.94])
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cax.yaxis.set_ticks_position("left")
    cbar.set_ticks([0, 50, 100])
    cax.tick_params(labelsize=5.8, length=2)
    _place_cutoff_labels(cbar.ax, [(thr[k], f"{thr[k]:.0f}%") for _c, k, _l in SCENARIOS])

    stats = []
    for ax, header, (_c, key, label) in zip(axes_row[1:], COLUMN_HEADERS[1:], SCENARIOS):
        cutoff = thr[key]
        mask, os_mha, pct, land = _stats(prob, cutoff, row_area_ha)
        stats.append({"biome": biome, "tile": tile, "scenario": key, "label": label,
                      "cutoff_pct": round(cutoff, 2), "organic_soil_Mha": round(os_mha, 3),
                      "pct_of_land_in_view": round(pct, 1)})
        _draw_basemap(ax, bm, extent, on_top=False)
        ax.imshow(mask, extent=extent, origin="upper", cmap=os_cmap,
                  vmin=0, vmax=1, alpha=0.85, zorder=3)
        _style_map_axes(ax, extent, show_x=bottom_row, show_y=False)
        if top_row:
            ax.set_title(f"{header}", fontsize=10.5, weight="bold")
        # Data label overlaid in-panel (keeps rows tight, no caption band).
        ax.text(0.035, 0.045, f"{os_mha:,.1f} Mha · {pct:.0f}% of land",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7.3,
                zorder=5, bbox=_OVERLAY_BBOX)
    return stats


def build_grid(examples, bm):
    n = len(examples)
    # Row height tracks each view's true aspect; with square views these are all
    # ~1, so every panel is the same size.
    ratios = [_view_aspect(e["extent"]) * (e["extent"][3] - e["extent"][2])
              / (e["extent"][1] - e["extent"][0]) for e in examples]

    # Solve fig height so grid cells are exactly the right shape (no whitespace).
    fig_w, wspace, hspace = 14.0, 0.08, 0.07
    left, right, top, bottom = 0.095, 0.99, 0.87, 0.04
    cell_w = fig_w * (right - left) / (4 + 3 * wspace)
    row_units = sum(ratios) + hspace * (n - 1) * (sum(ratios) / n)
    fig_h = cell_w * row_units / (top - bottom)

    stats_rows = []
    with pc.use_theme(pc.THEME_PANEL):
        fig, axes = plt.subplots(
            n, 4, figsize=(fig_w, fig_h), squeeze=False,
            gridspec_kw={"height_ratios": ratios, "hspace": hspace, "wspace": wspace})
        fig.patch.set_facecolor("white")
        for i, ex in enumerate(examples):
            stats_rows += _render_row(fig, axes[i], ex, bm,
                                      top_row=(i == 0), bottom_row=(i == n - 1))
        fig.suptitle(
            "OGH organic-soil probability vs. thresholded extent, by biome\n"
            "Low estimate (high cutoff) → Baseline → High estimate (low cutoff)",
            fontsize=12, weight="bold", y=0.975)
        # Inset colorbars are incompatible with tight_layout; set margins directly.
        fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right)
    return fig, pd.DataFrame(stats_rows)


def _parse_examples(spec: str):
    """Parse "tile:biome" (auto-crop) or "tile:biome:lon:lat:latspan" (square crop)."""
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(":")]
        if len(parts) not in (2, 5):
            raise ValueError(
                f"--examples entry must be tile:biome or tile:biome:lon:lat:latspan, "
                f"got {chunk!r}")
        ex = {"tile": parts[0], "biome": parts[1].lower()}
        if len(parts) == 5:
            ex["center"] = (float(parts[2]), float(parts[3]))
            ex["lat_span"] = float(parts[4])
        out.append(ex)
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", default=None,
                        help='Comma-separated rows "tile:biome" (auto-crop) or '
                             '"tile:biome:lon:lat:latspan" (square crop). '
                             'Default: one square-cropped row per biome.')
    parser.add_argument("--threshold-csv", default=DEFAULT_THRESHOLD_CSV)
    parser.add_argument("--s3-base", default=DEFAULT_S3_BASE)
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--no-auto-crop", action="store_true")
    parser.add_argument("--no-basemap", action="store_true")
    parser.add_argument("--max-dim", type=int, default=1800)
    parser.add_argument("--resampling", default="nearest",
                        choices=["nearest", "average", "bilinear"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    specs = _parse_examples(args.examples) if args.examples else DEFAULT_EXAMPLES
    resampling = Resampling[args.resampling]
    bm = {} if args.no_basemap else _load_basemap()

    examples = []
    for spec in specs:
        tile, biome = spec["tile"], spec["biome"]
        path = _resolve_tile_path(tile, args.local_dir, args.s3_base)
        thr = _load_biome_thresholds(args.threshold_csv, biome)
        if spec.get("center"):
            bbox = _square_bbox(spec["center"], spec["lat_span"])
        elif not args.no_auto_crop:
            coarse, c_ext = _read_window(path, None, 700, Resampling.nearest)
            bbox = _auto_crop_bbox(coarse, c_ext, cutoff=min(thr.values()))
        else:
            bbox = None
        prob, extent = _read_window(path, bbox, args.max_dim, resampling)
        print(f"{tile} ({biome}): view {tuple(round(x,2) for x in extent)}  "
              f"grid {prob.shape}  cutoffs " + "/".join(f"{thr[k]:.0f}" for _c, k, _l in SCENARIOS))
        examples.append({"tile": tile, "biome": biome, "prob": prob,
                         "extent": extent, "thresholds": thr})

    fig, stats = build_grid(examples, bm)

    biomes = "_".join(s["biome"] for s in specs)
    out_path = args.out or pc._join(
        lop.publication_root("figures"), "organic_soil_threshold_maps",
        f"ogh_threshold_grid_{biomes}.png")
    pc._save_png(fig, out_path, dpi=args.dpi)
    plt.close(fig)

    csv_path = out_path.rsplit(".", 1)[0] + "_stats.csv"
    pc._ensure_parent_dir_local(csv_path)
    stats.to_csv(csv_path, index=False)

    print("\nPer-scenario organic-soil extent (within each view):")
    print(stats.to_string(index=False))
    print(f"\nFigure: {out_path}\nStats:  {csv_path}")


if __name__ == "__main__":
    main()
