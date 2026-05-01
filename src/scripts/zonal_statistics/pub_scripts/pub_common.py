# -*- coding: utf-8 -*-
"""
Utilities for building publication tables & figures from organic-soils zonal statistics.

Pure utilities: plotting helpers, constants, SQL string builders.
No DuckDB setup or registration; the driver wires everything.

Additions:
- Publication themes (rcParams) + context manager: THEME_LIGHT_GRID, THEME_PANEL, THEME_GRAYSCALE, use_theme()
- Axis utilities: tidy_axes(), fmt_si()
- Color utilities: PALETTES, categorical_color_map(), landuse_colors(), resolve_colors()

New SQL helpers added in this revision:
- sql_total_by_climate(): drained+burned totals by climate × period
- sql_component_split_by_climate_avg(n_periods): avg drained vs burned per climate
- sql_component_intensity_by_climate_avg(n_periods): emissions intensity per climate split by component
- sql_drained_intensity_by_climate_avg(n_periods): drained emissions intensity per climate
"""

from __future__ import annotations

import argparse
import os
import posixpath
import re
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, List, Dict, Mapping, Tuple

import duckdb
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from src.scripts.utilities import local_output_paths as lop
from src.scripts.zonal_statistics import zonal_constants as zc

# Attempt to import pycountry lazily for ISO lookups. The dependency is optional and
# the rest of the module works without it, so we swallow any import-time errors and
# simply operate without country names/ISO codes when unavailable.
try:  # pragma: no cover - import guard is environment dependent
    import pycountry  # type: ignore
except Exception:  # pragma: no cover - best effort optional dependency
    pycountry = None  # type: ignore

# cycler is often available via matplotlib dependency; keep optional
try:  # pragma: no cover
    from cycler import cycler  # type: ignore
except Exception:  # pragma: no cover
    cycler = None  # type: ignore

# ----------------------------- Shared constants ---------------------------

# AR6 100-year GWP for N2O. Used by every pub_* script that converts NGHGI
# kt N2O (mass) to CO2e for like-for-like comparison with the model's
# drained_n2o_Mg_CO2e flux. Update once here when the inventory cycle
# moves to AR7 GWPs.
N2O_GWP: float = 273.0

# Stoichiometric ratio for converting organic-soil C-stock change (kt C in
# CRT Tables 4.A-F) to CO2 emissions. Sign is flipped at the call site
# because a negative cstock change (carbon loss) corresponds to a positive
# CO2 emission.
C_TO_CO2: float = 44.0 / 12.0


# ----------------------------- IO helpers ---------------------------------

def _is_s3(path: str) -> bool:
    return str(path).startswith("s3://")


def _join(base: str, *parts: str) -> str:
    if _is_s3(base):
        return posixpath.join(base, *parts)
    return os.path.join(base, *parts).replace("\\", "/")


def _ensure_parent_dir_local(path: str) -> None:
    if _is_s3(path):
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _to_duckdb_path(path: str) -> str:
    return path if _is_s3(path) else Path(path).as_posix()


def _save_png(
    fig: plt.Figure,
    path: str,
    dpi: int = 300,
    width: Optional[float] = None,
    height: Optional[float] = None,
) -> None:
    if width and height:
        fig.set_size_inches(width, height)
    _ensure_parent_dir_local(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _write_csv_df(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, path: str) -> None:
    _ensure_parent_dir_local(path)
    tmp_name = f"df_{id(df)}"
    con.register(tmp_name, df)
    out = _to_duckdb_path(path).replace("'", "''")
    con.execute(f"COPY {tmp_name} TO '{out}' (FORMAT CSV, HEADER TRUE)")
    try:
        con.unregister(tmp_name)
    except Exception:
        pass


# ----------------------------- CLI helpers --------------------------------

def add_publication_root_arg(
    parser: argparse.ArgumentParser, kind: str, env_var: str
) -> str:
    """
    Add the standard ``--out-dir-root`` flag with the publication-root
    default (``$<env_var>`` if set, else ``lop.publication_root(kind)``).
    Returns the resolved default for callers that want to mirror it into
    a module-level variable.
    """
    default = os.environ.get(env_var) or lop.publication_root(kind)
    parser.add_argument(
        "--out-dir-root",
        default=default,
        help=f"Output root (default: {default}).",
    )
    return default


# ----------------------------- Plot constants -----------------------------

CLIMATE_ORDER = ["Boreal", "Temperate", "Tropical"]

# New: curated climate palettes (CVD-safe)
CLIMATE_PALETTES = {
    "brewer_set2": {      # soft, modern
        "Boreal":    "#8DA0CB",
        "Temperate": "#FC8D62",
        "Tropical":  "#66C2A5",
    },
    "brewer_dark2": {     # rich, high contrast
        "Boreal":    "#7570B3",
        "Temperate": "#D95F02",
        "Tropical":  "#1B9E77",
    },
    "okabe_ito": {       # canonical CVD-safe triad
        "Boreal":    "#0072B2",
        "Temperate": "#E69F00",
        "Tropical":  "#009E73",
    },
}

# Default climate colors (switch here if you want a different default)
CLIMATE_COLORS = CLIMATE_PALETTES["brewer_set2"].copy()

PROCESS_ORDER = ["Drained", "Burned"]
PROCESS_COLORS = {"Drained": "#3E3753", "Burned": "#FB6A29"}

STATE_PAD_DIGITS = zc.STATE_CODE_PAD_DIGITS


def default_run_label(run_name: str) -> str:
    """Return a readable default label for a run name."""
    label = run_name.replace("_", " ").replace("-", " ").strip()
    return label or run_name


def parse_run_spec_entries(
    entries: Sequence[str],
    *,
    spec_name: str = "--run",
) -> List[Tuple[str, str, str, str]]:
    """
    Parse entries in the form:
      run_name=model_version:run_date
      run_name=model_version:run_date|Label

    Returns a list of tuples:
      (run_name, model_version, run_date, label)
    """
    parsed: List[Tuple[str, str, str, str]] = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid {spec_name} specification (expected run_name=model_version:run_date): {entry}"
            )
        raw_name, rest = entry.split("=", 1)
        run_name = raw_name.strip()
        if not run_name:
            raise ValueError(f"Invalid run name in {spec_name} specification: {entry}")

        if "|" in rest:
            config_part, label_part = rest.split("|", 1)
            label = label_part.strip() or default_run_label(run_name)
        else:
            config_part = rest
            label = default_run_label(run_name)

        parts = [p.strip() for p in config_part.split(":") if p.strip()]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid {spec_name} specification "
                "(expected model_version:run_date or model_version:run_date|Label): "
                f"{entry}"
            )
        model_version, run_date = parts
        parsed.append((run_name, model_version, run_date, label))
    return parsed


@dataclass(frozen=True)
class RunSpec:
    """Parsed model-run specification."""
    run_name: str
    model_version: str
    run_date: str
    label: str


def parse_run_specs(
    entries: Sequence[str],
    *,
    spec_name: str = "--run",
) -> List[RunSpec]:
    """Parse ``--run`` CLI entries into :class:`RunSpec` objects."""
    return [
        RunSpec(run_name=rn, model_version=mv, run_date=rd, label=lbl)
        for rn, mv, rd, lbl in parse_run_spec_entries(entries, spec_name=spec_name)
    ]


def set_climate_palette(name: str) -> dict:
    """
    Mutate CLIMATE_COLORS in-place so any references captured earlier
    (e.g., in ComponentPlotMeta) pick up the new palette automatically.
    """
    new_map = CLIMATE_PALETTES.get(name)
    if not new_map:
        raise ValueError(f"Unknown climate palette: {name}. "
                         f"Choose from: {', '.join(CLIMATE_PALETTES.keys())}")
    CLIMATE_COLORS.clear()
    CLIMATE_COLORS.update(new_map)
    return CLIMATE_COLORS


# ----------------------------- Color utilities ----------------------------

# Paul Tol / Okabe–Ito inspired, color-vision-deficiency-safe sets
PALETTES: Dict[str, List[str]] = {
    "tol_bright": [
        "#4477AA", "#EE6677", "#228833", "#CCBB44",
        "#66CCEE", "#AA3377", "#BBBBBB", "#000000",
    ],
    "tol_muted": [
        "#332288", "#88CCEE", "#44AA99", "#117733",
        "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499",
    ],
    "okabe_ito": [
        "#0072B2", "#E69F00", "#009E73", "#D55E00",
        "#CC79A7", "#F0E442", "#56B4E9", "#000000",
    ],
    "grayscale": ["#111111", "#555555", "#888888", "#BBBBBB"],
}

# Land-use color pins (get stable, meaningful hues for common classes)
DEFAULT_LANDUSE_OVERRIDES: Dict[str, str] = {
    "Cropland":        "#E17C05",
    "Other plantation":"#1B9E77",
    "Oil Palm":        "#66A61E",
    "Forest":          "#7570B3",
    "Grassland":       "#A6761D",
    "Settlement":      "#666666",
    "Extraction":      "#E7298A",
    "Wetland":         "#1B9E77",
    "Otherland":       "#BBBBBB",
}

def categorical_color_map(
    categories: Sequence[str],
    *,
    palette: str = "tol_muted",
    overrides: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """
    Build a stable {category: color} map.
    - Deterministic assignment using SHA1 hash into a palette.
    - 'overrides' let you pin specific categories to specific colors.
    - Works even if 'categories' changes between runs: unchanged labels keep colors.
    """
    base = PALETTES.get(palette, PALETTES["tol_muted"])
    cmap: Dict[str, str] = {}
    if overrides:
        cmap.update(dict(overrides))

    for c in categories:
        if c in cmap:
            continue
        h = int(hashlib.sha1(str(c).encode("utf-8")).hexdigest(), 16)
        cmap[c] = base[h % len(base)]
    return cmap

def resolve_colors(
    categories: Sequence[str],
    color_map: Optional[Mapping[str, str]] = None,
    *,
    palette: str = "okabe_ito",
    overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """
    Ensure a complete color map for 'categories', preserving any provided colors
    and deterministically filling the rest from 'palette'.
    """
    provided = dict(color_map) if color_map else {}
    missing = [c for c in categories if c not in provided]
    if missing:
        auto_map = categorical_color_map(missing, palette=palette, overrides=overrides or {})
        provided.update({k: auto_map[k] for k in missing})
    return provided

def landuse_colors(categories: Sequence[str]) -> Dict[str, str]:
    """Stable, readable colors for land-use classes."""
    return categorical_color_map(categories, palette="okabe_ito", overrides=DEFAULT_LANDUSE_OVERRIDES)

# ----------------------------- Theme utilities ----------------------------

def _maybe_cycle(colors: List[str]) -> Dict[str, object]:
    if cycler is None:
        return {}
    return {"axes.prop_cycle": cycler(color=colors)}

_BASE_THEME: Dict[str, object] = {
    # Export behavior
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    # Typography (journal-friendly)
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    # Lines/ticks
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    # Spacing
    "axes.titlepad": 8.0,
    "axes.labelpad": 6.0,
}

THEME_LIGHT_GRID: Dict[str, object] = {
    **_BASE_THEME,
    "axes.grid": True,
    "axes.grid.axis": "y",       # default helpful for column/bar charts
    "grid.linewidth": 0.6,
    "grid.color": "#D6D6D6",
}

THEME_PANEL: Dict[str, object] = {
    **_BASE_THEME,
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "axes.grid.axis": "both",
    "grid.linewidth": 0.5,
    "grid.color": "#E1E1E1",
}

THEME_GRAYSCALE: Dict[str, object] = {
    **_BASE_THEME,
    **_maybe_cycle(PALETTES["grayscale"]),
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#CFCFCF",
}

@contextmanager
def use_theme(theme: Mapping[str, object]):
    """Temporarily apply rcParams (use around plotting code)."""
    with mpl.rc_context(theme):
        yield

def tidy_axes(ax: plt.Axes, *, grid: Optional[str] = "y", minor: bool = False) -> plt.Axes:
    """
    Standardize spines, ticks, and gridlines on an Axes.
    grid: 'x', 'y', 'both', or None
    """
    # gridlines
    if grid:
        ax.grid(True, axis=grid, which="major")
        if minor:
            ax.minorticks_on()
            ax.grid(True, axis=grid, which="minor", linewidth=0.4, alpha=0.6)
    # spines & ticks
    for side in ("top", "right"):
        if side in ax.spines:
            ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", which="both", direction="out")
    return ax

def fmt_si(ax: plt.Axes, *, axis: str = "y", unit: Optional[str] = None) -> plt.Axes:
    """
    Apply SI-style tick formatter and optionally set unit label if none is set.
    axis: 'x' or 'y'
    """
    from matplotlib.ticker import ScalarFormatter
    fmt = ScalarFormatter(useOffset=False, useMathText=True)
    fmt.set_powerlimits((-3, 4))
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
        if unit and not ax.get_ylabel():
            ax.set_ylabel(unit)
    else:
        ax.xaxis.set_major_formatter(fmt)
        if unit and not ax.get_xlabel():
            ax.set_xlabel(unit)
    return ax

# ----------------------------- Small helpers ------------------------------

def titlecase_domain(s: Optional[str]) -> str:
    if s is None:
        return "Unspecified"
    t = s.strip().title()
    if t.startswith("Boreal"): return "Boreal"
    if t.startswith("Temperate"): return "Temperate"
    if t.startswith("Tropical"): return "Tropical"
    return t or "Unspecified"

def country_label(df: pd.DataFrame) -> pd.Series:
    """Prefer iso3 if present/non-empty, else fallback to numeric code."""
    if "iso3" in df.columns:
        iso = df["iso3"].astype("string")
        return iso.where(iso.str.len().fillna(0) > 0, df["gadm_adm0"].astype(str))
    return df["gadm_adm0"].astype(str)


def _rpad_sql(expr: str) -> str:
    """Return an RPAD expression using the shared state-node width."""
    return f"RPAD(CAST({expr} AS VARCHAR), {STATE_PAD_DIGITS}, '0')"

def build_adm0_lookup_df(manual_overrides: Optional[Dict[int, Dict[str, Optional[str]]]] = None) -> pd.DataFrame:
    """Return a DataFrame with gadm_adm0 → (iso3, country) best-effort mappings."""
    overrides: Dict[int, Dict[str, Optional[str]]] = manual_overrides or {}
    rows: list[dict[str, Optional[str] | int]] = []
    seen: set[int] = set()

    for raw_code in zc.GADM_ADM0_IDS.tolist():
        code = int(raw_code)
        if code in seen:
            continue
        seen.add(code)

        if code == 0:
            rows.append({"gadm_adm0": 0, "iso3": None, "country": "NoData"})
            continue

        if code in overrides:
            entry = overrides[code]
            rows.append({
                "gadm_adm0": code,
                "iso3": entry.get("iso3"),
                "country": entry.get("country"),
            })
            continue

        iso3: Optional[str] = None
        country: Optional[str] = None

        if pycountry is not None:
            lookup_code = f"{code:03d}"
            record = pycountry.countries.get(numeric=lookup_code)
            if record is None:
                # Fallback to historic countries (e.g., defunct ISO assignments)
                record = getattr(pycountry, "historic_countries", None)
                if record is not None:
                    record = record.get(numeric=lookup_code)  # type: ignore[assignment]

            if record is not None:
                iso3 = getattr(record, "alpha_3", None)
                country = (
                    getattr(record, "common_name", None)
                    or getattr(record, "official_name", None)
                    or getattr(record, "name", None)
                )

        rows.append({"gadm_adm0": code, "iso3": iso3, "country": country})

    df = pd.DataFrame(rows, columns=["gadm_adm0", "iso3", "country"])
    if not df.empty:
        df = df.astype({"gadm_adm0": "int32"})
    return df

def pivot_wide(df_long: pd.DataFrame, value_col: str, index_col: str) -> pd.DataFrame:
    return (
        df_long
        .pivot_table(index=index_col, columns="Climate", values=value_col,
                     aggfunc="sum", fill_value=0.0, observed=False)
        .reindex(columns=CLIMATE_ORDER, fill_value=0.0)
        .reset_index()
    )

# ----------------------------- Land-use reclass ---------------------------

_LU_RECLASS_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Coastal wetlands (mangroves & tidal marshes) → Wetland
    # Handles:
    #   coastal_mangrove
    #   other_domain_coastal_mangrove
    #   coastal_tidal_marsh
    #   other_domain_coastal_tidal_marsh
    # and any future variants that normalize to '*coastal_*mangrove' or '*coastal_*tidal_marsh'.
    (re.compile(r"^.*coastal[_\- ]?(?:mangrove|tidal[_\- ]?marsh).*$"), "Wetland"),

    (re.compile(r"^(oil[_\- ]?palm|oilpalm)$"), "Oil Palm"),
    (re.compile(
        r"^(short[_\- ]?rotation|long[_\- ]?rotation|plantation.*|planted.*|"
        r"tree[_\- ]?crop.*)$"
    ), "Other plantation"),
    (re.compile(r"^cropland.*$"), "Cropland"),
    (re.compile(r"^forest.*$"), "Forest"),
    (re.compile(r"^(grassland|pasture|rangeland).*$"), "Grassland"),
    (re.compile(r"^(settlement|built[_\- ]?up|urban).*$"), "Settlement"),
    (re.compile(r"^wetland.*$"), "Wetland"),
    (re.compile(r"^(extraction|peat[_\- ]?extraction|cutover).*$"), "Extraction"),
    (re.compile(r"^(otherland|other)$"), "Otherland"),
]


def _normalize_emissions_state(s: Optional[str]) -> str:
    if s is None:
        return "other"
    return re.sub(r"[\s\-]+", "_", s.strip().lower())

def _reclass_emissions_state(s: Optional[str]) -> str:
    key = _normalize_emissions_state(s)
    for pat, lbl in _LU_RECLASS_PATTERNS:
        if pat.match(key):
            return lbl
    return (s.strip().replace("_", " ").title() if s else "Otherland")

def aggregate_landuse(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Reclass emissions_state to coarse land-uses and aggregate by (LandUse × Climate)."""
    df = df.copy()
    df["Climate"] = df["climate_domain"].apply(titlecase_domain)
    df = df[df["Climate"].isin(CLIMATE_ORDER)]
    df["LandUse"] = df["emissions_state"].apply(_reclass_emissions_state).str.replace("_", " ", regex=False)
    out = (
        df.groupby(["LandUse", "Climate"], as_index=False, observed=False)[value_col]
          .sum()
    )
    totals = out.groupby("LandUse", observed=False)[value_col].sum().sort_values(ascending=False).index.tolist()
    out["LandUse"] = pd.Categorical(out["LandUse"], totals, ordered=True)
    return out.sort_values(["LandUse", "Climate"])

# ----------------------------- Plot helpers -------------------------------

def stacked_column_by_category(
    df_long: pd.DataFrame,
    index_col: str,
    category_col: str,
    value_col: str,
    category_order: Sequence[str],
    color_map: dict[str, str],
    xlabel: str,
    ylabel: str,
    width: float = 7.5,
    height: float = 4.5,
    legend_above: bool = False,
    # NEW:
    bar_width: float = 0.58,              # <— slimmer columns (was implicit 0.8)
    segment_edgecolor: str = "white",     # clean separators between stacked segments
    segment_linewidth: float = 0.5,
) -> plt.Figure:
    df = df_long.copy()
    df[category_col] = pd.Categorical(df[category_col], category_order, ordered=True)
    x_vals = list(dict.fromkeys(df[index_col].tolist()))
    wide = (
        df.pivot_table(index=index_col, columns=category_col, values=value_col,
                       aggfunc="sum", fill_value=0.0, observed=False)
          .reindex(index=x_vals)
          .reindex(columns=category_order, fill_value=0.0)
    )
    totals = wide.sum(axis=1).values
    y_max = float(max(totals)) * 1.12 if len(totals) else 1.0

    # ensure a complete color map
    colors = resolve_colors(category_order, color_map)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_axisbelow(True)  # keep grid behind bars
    bottom = None
    for cat in category_order:
        vals = wide[cat].values
        ax.bar([str(x) for x in wide.index], vals,
               width=bar_width,
               bottom=bottom,
               label=cat,
               color=colors[cat],
               edgecolor=segment_edgecolor,
               linewidth=segment_linewidth)
        bottom = vals if bottom is None else bottom + vals

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    tidy_axes(ax, grid="y")
    fmt_si(ax, axis="y")
    ax.margins(x=0.04)  # a touch of breathing room

    if legend_above:
        ax.legend(ncol=min(len(category_order), 5), loc="upper center",
                  bbox_to_anchor=(0.5, 1.18), frameon=False, handlelength=1.6, columnspacing=1.2)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
    else:
        ax.legend(ncol=min(len(category_order), 4), loc="upper left",
                  bbox_to_anchor=(0.0, 1.10), frameon=False, handlelength=1.6, columnspacing=1.2)
        fig.tight_layout(rect=(0, 0, 1, 0.88))

    ax.set_ylim(0, y_max)
    for xpos, total in zip(range(len(totals)), totals):
        ax.text(xpos, total + (y_max * 0.015), f"{total:.2f}", ha="center", va="bottom", fontsize=9)
    return fig


def stacked_hbar(df_long: pd.DataFrame, value_col: str, xlabel: str) -> plt.Figure:
    """Stacked horizontal bars by Climate within LandUse."""
    df = df_long.copy()
    df["Climate"] = pd.Categorical(df["Climate"], CLIMATE_ORDER, ordered=True)
    wide = (
        df.pivot_table(index="LandUse", columns="Climate", values=value_col,
                       aggfunc="sum", fill_value=0.0, observed=False)
          .reindex(columns=CLIMATE_ORDER, fill_value=0.0)
    )
    order = list(wide.sum(axis=1).sort_values(ascending=False).index)
    wide = wide.reindex(order)

    totals = wide.sum(axis=1).values
    x_max = float(max(totals)) if len(totals) else 1.0
    height = max(3.2, 0.55 * len(order) + 1.0)
    fig, ax = plt.subplots(figsize=(7.5, height))

    # Ensure climate colors are complete (in case of extra domains)
    colors = resolve_colors(CLIMATE_ORDER, CLIMATE_COLORS)

    left = None
    for climate in CLIMATE_ORDER:
        vals = wide[climate].values
        ax.barh(wide.index.astype(str), vals, left=left, color=colors[climate], label=climate)
        left = vals if left is None else left + vals

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Land Use")
    tidy_axes(ax, grid="x")
    fmt_si(ax, axis="x")

    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.12),
              frameon=False, handlelength=1.6, columnspacing=1.2)
    for y, total in zip(range(len(totals)), totals):
        ax.text(total + (x_max * 0.01), y, f"{total:.2f}", ha="left", va="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig

def hbar_two_series(labels: List[str], left_vals: List[float], right_vals: List[float],
                    xlabel: str, legends: tuple[str, str], colors: tuple[str, str]) -> plt.Figure:
    """Two-series horizontal stacked bars (e.g., drained vs undrained)."""
    totals = [lv + rv for lv, rv in zip(left_vals, right_vals)]
    order = sorted(range(len(labels)), key=lambda i: totals[i], reverse=True)
    labs   = [labels[i] for i in order]
    lvals  = [left_vals[i] for i in order]
    rvals  = [right_vals[i] for i in order]
    tots   = [totals[i] for i in order]
    y = list(range(len(labs)))

    x_max = max(tots) if tots else 1.0
    height = max(3.0, 0.5 * len(labs) + 1.0)
    fig, ax = plt.subplots(figsize=(7.5, height))
    ax.barh(y, lvals, color=colors[0], label=legends[0])
    ax.barh(y, rvals, left=lvals, color=colors[1], label=legends[1])
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    tidy_axes(ax, grid="x")
    fmt_si(ax, axis="x")

    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.10), frameon=False)
    for yy, tot in zip(y, tots):
        ax.text(tot + (x_max * 0.01), yy, f"{tot:.2f}", ha="left", va="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig

def barh_single(labels: List[str], values: List[float], xlabel: str,
                color, *, sort_desc: bool = True) -> plt.Figure:
    """Horizontal bar chart for a single series."""
    if sort_desc:
        order = sorted(range(len(labels)), key=lambda i: values[i], reverse=True)
    else:
        order = list(range(len(labels)))
    labs  = [labels[i] for i in order]
    vals  = [values[i] for i in order]
    y = list(range(len(labs)))
    height = max(3.0, 0.5 * len(labs) + 1.0)
    fig, ax = plt.subplots(figsize=(7.5, height))
    ax.barh(y, vals, color=color)
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    tidy_axes(ax, grid="x")
    fmt_si(ax, axis="x")

    x_max = max(vals) if vals else 1.0
    for yy, v in zip(y, vals):
        ax.text(v + (x_max * 0.01), yy, f"{v:.2f}", ha="left", va="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def stacked_hbar_generic(
    df: pd.DataFrame,
    label_col: str,
    component_order: Sequence[str],
    component_colors: Mapping[str, str],
    xlabel: str,
    *,
    legend_columns: int = 2,
    width: float = 8.0,
) -> plt.Figure:
    """Generalized horizontal stacked bar chart.

    *df* must have a *label_col* column for y-axis labels and one numeric
    column per entry in *component_order*.
    """
    labels = df[label_col].tolist()
    y_positions = list(range(len(labels)))
    height = max(3.2, 0.55 * len(labels) + 1.0)

    totals = df[list(component_order)].sum(axis=1).tolist()
    x_max = max(totals) if totals else 0.0

    colors = resolve_colors(component_order, component_colors)

    theme = {**THEME_LIGHT_GRID, "axes.grid.axis": "x"}
    with use_theme(theme):
        fig, ax = plt.subplots(figsize=(width, height))

        left = [0.0] * len(labels)
        for component in component_order:
            vals = df[component].tolist()
            ax.barh(y_positions, vals, left=left, color=colors[component], label=component)
            left = [l + v for l, v in zip(left, vals)]

        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_axisbelow(True)
        tidy_axes(ax, grid="x")
        fmt_si(ax, axis="x")

        pad = x_max * 0.03 if x_max else 0.05
        for ypos, total in zip(y_positions, totals):
            ax.text(total + pad, ypos, f"{total:.2f}", ha="left", va="center", fontsize=9)

        ax.legend(
            ncol=legend_columns,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.10),
            frameon=False,
            handlelength=1.6,
            columnspacing=1.2,
        )

        fig.tight_layout(rect=(0, 0, 1, 0.9))
        return fig

# ----------------------------- SQL builders -------------------------------

def table_by_country_period_sql(with_lookup: bool) -> str:
    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = f.gadm_adm0" if with_lookup else ""
    return f"""
    WITH d AS (
      SELECT interval_end, gadm_adm0, SUM(value) AS drained_MgCO2e
      FROM zs_drained
      WHERE flux_type = 'drained_total_Mg_CO2e'
      GROUP BY 1,2
    ),
    b AS (
      SELECT interval_end, gadm_adm0, SUM(value) AS burned_MgCO2e
      FROM zs_burned
      WHERE flux_type = 'burned_total_Mg_CO2e'
      GROUP BY 1,2
    ),
    f AS (
      SELECT
        COALESCE(d.interval_end, b.interval_end) AS interval_end,
        COALESCE(d.gadm_adm0,  b.gadm_adm0)     AS gadm_adm0,
        d.drained_MgCO2e,
        b.burned_MgCO2e
      FROM d FULL OUTER JOIN b
        ON d.interval_end = b.interval_end
       AND d.gadm_adm0    = b.gadm_adm0
    )
    SELECT
      f.interval_end,
      f.gadm_adm0
      {select_l},
      COALESCE(f.drained_MgCO2e, 0) AS drained_MgCO2e,
      COALESCE(f.burned_MgCO2e, 0) AS burned_MgCO2e,
      COALESCE(f.drained_MgCO2e, 0) + COALESCE(f.burned_MgCO2e, 0) AS total_MgCO2e
    FROM f
    {join_l}
    ORDER BY f.interval_end, total_MgCO2e DESC
    """

def table_by_drained_state_sql() -> str:
    return f"""
    WITH base AS (
      SELECT
        interval_end,
        drained_state_nodes,
        drained_state_meaning,
        SUM(CASE WHEN flux_type = 'drained_total_Mg_CO2e' THEN value ELSE 0 END) AS drained_MgCO2e,
        SUM(CASE WHEN flux_type = 'area__ha'               THEN value ELSE 0 END) AS area_ha
      FROM zs_drained
      GROUP BY 1,2,3
    )
    SELECT
      base.interval_end,
      base.drained_state_nodes,
      base.drained_state_meaning,
      base.drained_MgCO2e,
      base.area_ha,
      ctx.climate_domain,
      ctx.drained_state,
      ctx.emissions_state
    FROM base
    LEFT JOIN drained_state_ctx AS ctx
      ON (base.drained_state_meaning = ctx.meaning)
      OR ({_rpad_sql('base.drained_state_nodes')} = ctx.key)
    ORDER BY base.interval_end, base.drained_MgCO2e DESC
    """

def table_by_burned_state_sql() -> str:
    return f"""
    WITH base AS (
      SELECT
        interval_end,
        burned_state_nodes,
        burned_state_meaning,
        SUM(CASE WHEN flux_type = 'burned_total_Mg_CO2e' THEN value ELSE 0 END) AS burned_MgCO2e,
        SUM(CASE WHEN flux_type = 'area__ha'             THEN value ELSE 0 END) AS area_ha
      FROM zs_burned
      GROUP BY 1,2,3
    )
    SELECT
      base.interval_end,
      base.burned_state_nodes,
      base.burned_state_meaning,
      base.burned_MgCO2e,
      base.area_ha,
      ctx.climate_domain,
      ctx.burned_state,
      ctx.emissions_state
    FROM base
    LEFT JOIN burned_state_ctx AS ctx
      ON (base.burned_state_meaning = ctx.meaning)
      OR ({_rpad_sql('base.burned_state_nodes')} = ctx.key)
    ORDER BY base.interval_end, base.burned_MgCO2e DESC
    """

def table_by_country_drained_state_sql(with_lookup: bool) -> str:
    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = base.gadm_adm0" if with_lookup else ""
    return f"""
    WITH base AS (
      SELECT
        interval_end,
        gadm_adm0,
        drained_state_nodes,
        drained_state_meaning,
        SUM(CASE WHEN flux_type = 'drained_total_Mg_CO2e' THEN value ELSE 0 END) AS drained_MgCO2e,
        SUM(CASE WHEN flux_type = 'area__ha'               THEN value ELSE 0 END) AS area_ha
      FROM zs_drained
      GROUP BY 1,2,3,4
    )
    SELECT
      base.interval_end,
      base.gadm_adm0
      {select_l},
      base.drained_state_nodes,
      base.drained_state_meaning,
      base.drained_MgCO2e,
      base.area_ha,
      ctx.climate_domain,
      ctx.drained_state,
      ctx.emissions_state
    FROM base
    {join_l}
    LEFT JOIN drained_state_ctx AS ctx
      ON (base.drained_state_meaning = ctx.meaning)
      OR ({_rpad_sql('base.drained_state_nodes')} = ctx.key)
    ORDER BY base.interval_end, base.drained_MgCO2e DESC
    """

def table_by_country_burned_state_sql(with_lookup: bool) -> str:
    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = base.gadm_adm0" if with_lookup else ""
    return f"""
    WITH base AS (
      SELECT
        interval_end,
        gadm_adm0,
        burned_state_nodes,
        burned_state_meaning,
        SUM(CASE WHEN flux_type = 'burned_total_Mg_CO2e' THEN value ELSE 0 END) AS burned_MgCO2e,
        SUM(CASE WHEN flux_type = 'area__ha'             THEN value ELSE 0 END) AS area_ha
      FROM zs_burned
      GROUP BY 1,2,3,4
    )
    SELECT
      base.interval_end,
      base.gadm_adm0
      {select_l},
      base.burned_state_nodes,
      base.burned_state_meaning,
      base.burned_MgCO2e,
      base.area_ha,
      ctx.climate_domain,
      ctx.burned_state,
      ctx.emissions_state
    FROM base
    {join_l}
    LEFT JOIN burned_state_ctx AS ctx
      ON (base.burned_state_meaning = ctx.meaning)
      OR ({_rpad_sql('base.burned_state_nodes')} = ctx.key)
    ORDER BY base.interval_end, base.burned_MgCO2e DESC
    """

def table_stats_for_lulucf_paper_sql(with_lookup: bool) -> str:
    """
    Country × climate × component table by inventory period.

    Output columns:
      - interval_end
      - gadm_adm0[, country, iso3]
      - climate_domain
      - component (Drainage|Extraction|Fire)
      - flux_Mg_CO2e_yr
      - area_ha
    """
    select_l = (
        ", COALESCE(l.country, CAST(base.gadm_adm0 AS VARCHAR)) AS country,"
        " COALESCE(l.iso3, CAST(base.gadm_adm0 AS VARCHAR)) AS iso3"
        if with_lookup else ""
    )
    join_l = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = base.gadm_adm0" if with_lookup else ""
    return f"""
    WITH drained_labeled AS (
      SELECT
        z.interval_end,
        COALESCE(
          TRY_CAST(z.gadm_adm0 AS INTEGER),
          TRY_CAST(regexp_extract(CAST(z.gadm_adm0 AS VARCHAR), '(\\d+)', 1) AS INTEGER)
        ) AS gadm_adm0,
        COALESCE(ctx.climate_domain, 'Unspecified') AS climate_domain,
        CASE
          WHEN regexp_matches(lower(COALESCE(ctx.emissions_state, '')),
                              '^(extraction|peat[_\\- ]?extraction|cutover).*$')
          THEN 'Extraction'
          ELSE 'Drainage'
        END AS component,
        SUM(CASE WHEN z.flux_type = 'drained_total_Mg_CO2e' THEN z.value ELSE 0 END) AS flux_Mg_CO2e_yr,
        SUM(CASE WHEN z.flux_type = 'area__ha' THEN z.value ELSE 0 END) AS area_ha
      FROM zs_drained z
      LEFT JOIN drained_state_ctx AS ctx
        ON (z.drained_state_meaning = ctx.meaning)
        OR ({_rpad_sql('z.drained_state_nodes')} = ctx.key)
      GROUP BY 1,2,3,4
    ),
    burned_base AS (
      SELECT
        z.interval_end,
        COALESCE(
          TRY_CAST(z.gadm_adm0 AS INTEGER),
          TRY_CAST(regexp_extract(CAST(z.gadm_adm0 AS VARCHAR), '(\\d+)', 1) AS INTEGER)
        ) AS gadm_adm0,
        COALESCE(ctx.climate_domain, 'Unspecified') AS climate_domain,
        'Fire' AS component,
        SUM(CASE WHEN z.flux_type = 'burned_total_Mg_CO2e' THEN z.value ELSE 0 END) AS flux_Mg_CO2e_yr,
        SUM(CASE WHEN z.flux_type = 'area__ha' THEN z.value ELSE 0 END) AS area_ha
      FROM zs_burned z
      LEFT JOIN burned_state_ctx AS ctx
        ON (z.burned_state_meaning = ctx.meaning)
        OR ({_rpad_sql('z.burned_state_nodes')} = ctx.key)
      GROUP BY 1,2,3,4
    ),
    base AS (
      SELECT
        interval_end,
        gadm_adm0,
        climate_domain,
        component,
        SUM(flux_Mg_CO2e_yr) AS flux_Mg_CO2e_yr,
        SUM(area_ha) AS area_ha
      FROM (
        SELECT interval_end, gadm_adm0, climate_domain, component, flux_Mg_CO2e_yr, area_ha
        FROM drained_labeled
        UNION ALL
        SELECT interval_end, gadm_adm0, climate_domain, component, flux_Mg_CO2e_yr, area_ha
        FROM burned_base
      ) unioned
      GROUP BY 1,2,3,4
      HAVING SUM(flux_Mg_CO2e_yr) <> 0 OR SUM(area_ha) <> 0
    )
    SELECT
      base.interval_end,
      base.gadm_adm0
      {select_l},
      base.climate_domain,
      base.component,
      base.flux_Mg_CO2e_yr,
      base.area_ha
    FROM base
    {join_l}
    ORDER BY base.interval_end, base.gadm_adm0, base.climate_domain, base.component
    """

def table_topn_country_sql(component: str, topn: int, with_lookup: bool) -> str:
    assert component in {"drained", "burned"}
    base  = "zs_drained" if component == "drained" else "zs_burned"
    ftype = "drained_total_Mg_CO2e" if component == "drained" else "burned_total_Mg_CO2e"
    alias = "drained_MgCO2e" if component == "drained" else "burned_MgCO2e"

    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = ranked.gadm_adm0" if with_lookup else ""

    return f"""
    WITH t AS (
      SELECT interval_end, gadm_adm0, SUM(value) AS {alias}
      FROM {base}
      WHERE flux_type = '{ftype}'
      GROUP BY 1,2
    ),
    ranked AS (
      SELECT
        interval_end,
        gadm_adm0,
        {alias},
        ROW_NUMBER() OVER (PARTITION BY interval_end ORDER BY {alias} DESC) AS rnk
      FROM t
    )
    SELECT
      ranked.interval_end,
      rnk AS rank,
      ranked.gadm_adm0
      {select_l},
      {alias}
    FROM ranked
    {join_l}
    WHERE rnk <= {topn}
    ORDER BY ranked.interval_end, rank
    """

def table_nghgi_comparison_subset_sql(with_lookup: bool) -> str:
    """
    Country × inventory period × land-use subset intended for NGHGI comparisons.

    Output columns:
      - interval_end
      - gadm_adm0[, country, iso3]
      - land_use
      - drained_area_ha
      - drained_on_site_co2_Mg_CO2_yr
    """
    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = base.gadm_adm0" if with_lookup else ""
    return f"""
    WITH joined AS (
      SELECT
        z.*,
        ctx.drained_state,
        ctx.combined_state,
        ctx.emissions_state,
        CASE
          WHEN lower(COALESCE(ctx.drained_state, z.drained_state_meaning, '')) LIKE 'peat_drained%' THEN 'drained'
          WHEN lower(COALESCE(ctx.drained_state, z.drained_state_meaning, '')) LIKE 'peat_undrained%' THEN 'undrained'
          ELSE 'other'
        END AS peat_state
      FROM zs_drained z
      LEFT JOIN drained_state_ctx AS ctx
        ON (z.drained_state_meaning = ctx.meaning)
        OR ({_rpad_sql('z.drained_state_nodes')} = ctx.key)
    ),
    base AS (
      SELECT
        j.interval_end,
        j.gadm_adm0,
        CASE
          WHEN regexp_matches(lower(COALESCE(j.combined_state, j.emissions_state, '')),
                              '^.*coastal[_\\- ]?(mangrove|tidal[_\\- ]?marsh).*$')
          THEN 'Wetland'
          WHEN regexp_matches(lower(COALESCE(j.combined_state, j.emissions_state, '')), '^(oil[_\\- ]?palm|oilpalm)$')
          THEN 'Oil Palm'
          WHEN regexp_matches(lower(COALESCE(j.combined_state, j.emissions_state, '')),
                              '^(short[_\\- ]?rotation|long[_\\- ]?rotation|plantation.*|planted.*|tree[_\\- ]?crop.*)$')
          THEN 'Other plantation'
          WHEN regexp_matches(lower(COALESCE(j.combined_state, j.emissions_state, '')), '^cropland.*$')
          THEN 'Cropland'
          WHEN regexp_matches(lower(COALESCE(j.combined_state, j.emissions_state, '')), '^forest.*$')
          THEN 'Forest'
          WHEN regexp_matches(lower(COALESCE(j.combined_state, j.emissions_state, '')), '^(grassland|pasture|rangeland).*$')
          THEN 'Grassland'
          WHEN regexp_matches(lower(COALESCE(j.combined_state, j.emissions_state, '')), '^(settlement|built[_\\- ]?up|urban).*$')
          THEN 'Settlement'
          WHEN regexp_matches(lower(COALESCE(j.combined_state, j.emissions_state, '')), '^wetland.*$')
          THEN 'Wetland'
          WHEN regexp_matches(lower(COALESCE(j.combined_state, j.emissions_state, '')),
                              '^(extraction|peat[_\\- ]?extraction|cutover).*$')
          THEN 'Extraction'
          WHEN regexp_matches(lower(COALESCE(j.combined_state, j.emissions_state, '')), '^(otherland|other)$')
          THEN 'Otherland'
          WHEN j.peat_state = 'undrained'
          THEN 'Undrained (unclassified)'
          ELSE COALESCE(j.combined_state, j.emissions_state, 'Unspecified')
        END AS land_use,
        SUM(
          CASE
            WHEN lower(j.flux_type) IN ('area__ha', 'area_ha') AND j.peat_state = 'drained'
            THEN j.value ELSE 0
          END
        ) AS drained_area_ha,
        SUM(
          CASE
            WHEN lower(j.flux_type) IN ('area__ha', 'area_ha') AND j.peat_state = 'undrained'
            THEN j.value ELSE 0
          END
        ) AS undrained_area_ha,
        SUM(
          CASE
            WHEN lower(j.flux_type) LIKE 'drained_co2%' AND lower(j.flux_type) NOT LIKE '%offsite%'
                 AND j.peat_state = 'drained'
            THEN j.value ELSE 0
          END
        ) AS drained_on_site_co2_Mg_CO2_yr,
        SUM(
          CASE
            WHEN lower(j.flux_type) LIKE 'drained_n2o%'
                 AND j.peat_state = 'drained'
            THEN j.value ELSE 0
          END
        ) AS drained_n2o_Mg_CO2e_yr
      FROM joined j
      GROUP BY 1,2,3
    )
    SELECT
      base.interval_end,
      base.gadm_adm0
      {select_l},
      base.land_use,
      base.drained_area_ha,
      base.undrained_area_ha,
      base.drained_on_site_co2_Mg_CO2_yr,
      base.drained_n2o_Mg_CO2e_yr
    FROM base
    {join_l}
    WHERE base.drained_area_ha <> 0
       OR base.undrained_area_ha <> 0
       OR base.drained_on_site_co2_Mg_CO2_yr <> 0
       OR base.drained_n2o_Mg_CO2e_yr <> 0
    ORDER BY base.interval_end, base.gadm_adm0, base.land_use
    """

# ----------------------------- Figure SQL --------------------------------

def sql_drained_by_climate() -> str:
    return f"""
    WITH joined AS (
      SELECT
        z.interval_end,
        COALESCE(ctx.climate_domain, 'Unspecified') AS climate_domain,
        SUM(CASE WHEN z.flux_type = 'drained_total_Mg_CO2e' THEN z.value ELSE 0 END) AS drained_MgCO2e
      FROM zs_drained z
      LEFT JOIN drained_state_ctx AS ctx
        ON (z.drained_state_meaning = ctx.meaning)
        OR ({_rpad_sql('z.drained_state_nodes')} = ctx.key)
      GROUP BY 1,2
    )
    SELECT interval_end, climate_domain, drained_MgCO2e / 1e9 AS drained_GtCO2e
    FROM joined
    ORDER BY interval_end, climate_domain;
    """

def sql_burned_by_climate() -> str:
    return f"""
    WITH joined AS (
      SELECT
        z.interval_end,
        COALESCE(ctx.climate_domain, 'Unspecified') AS climate_domain,
        SUM(CASE WHEN z.flux_type = 'burned_total_Mg_CO2e' THEN z.value ELSE 0 END) AS burned_MgCO2e
      FROM zs_burned z
      LEFT JOIN burned_state_ctx AS ctx
        ON (z.burned_state_meaning = ctx.meaning)
        OR ({_rpad_sql('z.burned_state_nodes')} = ctx.key)
      GROUP BY 1,2
    )
    SELECT interval_end, climate_domain, burned_MgCO2e / 1e9 AS burned_GtCO2e
    FROM joined
    ORDER BY interval_end, climate_domain;
    """

def sql_drained_landuse_climate_avgs(n_periods: int) -> str:
    """
    Drained Land Use × Climate using ONLY the latest inventory period
    (i.e., the maximum interval_end present in zs_drained, which is 2024
    for your current run).

    Note: 'n_periods' is no longer used in the calculation; we keep the
    output column name 'drained_avg_GtCO2e_per_yr' for compatibility with
    the existing pub_assets plotting code.
    """
    return f"""
    WITH latest AS (
      SELECT MAX(interval_end) AS max_end
      FROM zs_drained
    ),
    joined AS (
      SELECT
        COALESCE(ctx.climate_domain, 'Unspecified')  AS climate_domain,
        COALESCE(ctx.emissions_state, 'Unspecified') AS emissions_state,
        SUM(
          CASE
            WHEN z.flux_type = 'drained_total_Mg_CO2e' THEN z.value
            ELSE 0
          END
        ) AS drained_MgCO2e
      FROM zs_drained z
      JOIN latest lt
        ON z.interval_end = lt.max_end
      LEFT JOIN drained_state_ctx AS ctx
        ON (z.drained_state_meaning = ctx.meaning)
        OR ({_rpad_sql('z.drained_state_nodes')} = ctx.key)
      GROUP BY 1,2
    )
    SELECT
      climate_domain,
      emissions_state,
      drained_MgCO2e / 1e9 AS drained_avg_GtCO2e_per_yr
    FROM joined;
    """


def sql_burned_landuse_climate_avgs(n_periods: int) -> str:
    """
    Burned Land Use × Climate using ONLY the latest inventory period
    (i.e., the maximum interval_end present in zs_burned, which is 2024
    for your current run).

    Note: 'n_periods' is no longer used in the calculation; we keep the
    output column name 'burned_avg_GtCO2e_per_yr' for compatibility with
    the existing pub_assets plotting code.
    """
    return f"""
    WITH latest AS (
      SELECT MAX(interval_end) AS max_end
      FROM zs_burned
    ),
    joined AS (
      SELECT
        COALESCE(ctx.climate_domain, 'Unspecified')  AS climate_domain,
        COALESCE(ctx.emissions_state, 'Unspecified') AS emissions_state,
        SUM(
          CASE
            WHEN z.flux_type = 'burned_total_Mg_CO2e' THEN z.value
            ELSE 0
          END
        ) AS burned_MgCO2e
      FROM zs_burned z
      JOIN latest lt
        ON z.interval_end = lt.max_end
      LEFT JOIN burned_state_ctx AS ctx
        ON (z.burned_state_meaning = ctx.meaning)
        OR ({_rpad_sql('z.burned_state_nodes')} = ctx.key)
      GROUP BY 1,2
    )
    SELECT
      climate_domain,
      emissions_state,
      burned_MgCO2e / 1e9 AS burned_avg_GtCO2e_per_yr
    FROM joined;
    """


# --- NEW for C: total emissions (drained+burned) by climate × period -----

def sql_total_by_climate() -> str:
    return f"""
    WITH d AS (
      SELECT
        z.interval_end,
        COALESCE(ctx.climate_domain, 'Unspecified') AS climate_domain,
        SUM(CASE WHEN z.flux_type = 'drained_total_Mg_CO2e' THEN z.value ELSE 0 END) AS Mg
      FROM zs_drained z
      LEFT JOIN drained_state_ctx AS ctx
        ON (z.drained_state_meaning = ctx.meaning)
        OR ({_rpad_sql('z.drained_state_nodes')} = ctx.key)
      GROUP BY 1,2
    ),
    b AS (
      SELECT
        z.interval_end,
        COALESCE(ctx.climate_domain, 'Unspecified') AS climate_domain,
        SUM(CASE WHEN z.flux_type = 'burned_total_Mg_CO2e' THEN z.value ELSE 0 END) AS Mg
      FROM zs_burned z
      LEFT JOIN burned_state_ctx AS ctx
        ON (z.burned_state_meaning = ctx.meaning)
        OR ({_rpad_sql('z.burned_state_nodes')} = ctx.key)
      GROUP BY 1,2
    )
    SELECT
      COALESCE(d.interval_end, b.interval_end) AS interval_end,
      COALESCE(d.climate_domain, b.climate_domain) AS climate_domain,
      (COALESCE(d.Mg, 0) + COALESCE(b.Mg, 0)) / 1e9 AS total_GtCO2e
    FROM d
    FULL OUTER JOIN b
      ON d.interval_end = b.interval_end
     AND d.climate_domain = b.climate_domain
    ORDER BY interval_end, climate_domain;
    """

# --- NEW for D: component split within each climate (avg over periods) ---

def sql_component_split_by_climate_avg(n_periods: int) -> str:
    return f"""
    WITH d AS (
      SELECT
        COALESCE(ctx.climate_domain, 'Unspecified') AS climate_domain,
        SUM(CASE WHEN z.flux_type='drained_total_Mg_CO2e' THEN z.value ELSE 0 END) AS Mg
      FROM zs_drained z
      LEFT JOIN drained_state_ctx AS ctx
        ON (z.drained_state_meaning = ctx.meaning)
        OR ({_rpad_sql('z.drained_state_nodes')} = ctx.key)
      GROUP BY 1
    ),
    b AS (
      SELECT
        COALESCE(ctx.climate_domain, 'Unspecified') AS climate_domain,
        SUM(CASE WHEN z.flux_type='burned_total_Mg_CO2e' THEN z.value ELSE 0 END) AS Mg
      FROM zs_burned z
      LEFT JOIN burned_state_ctx AS ctx
        ON (z.burned_state_meaning = ctx.meaning)
        OR ({_rpad_sql('z.burned_state_nodes')} = ctx.key)
      GROUP BY 1
    )
    SELECT climate_domain, 'Drained' AS component, (COALESCE(d.Mg,0) / NULLIF({n_periods},0)) / 1e9 AS avg_GtCO2e_per_yr
      FROM d
    UNION ALL
    SELECT climate_domain, 'Burned'  AS component, (COALESCE(b.Mg,0) / NULLIF({n_periods},0)) / 1e9 AS avg_GtCO2e_per_yr
      FROM b
    ORDER BY climate_domain, component;
    """

# --- NEW for F: drained emissions intensity by climate -------------------

def sql_drained_intensity_by_climate_avg(n_periods: int) -> str:
    return f"""
    WITH area_by_period AS (
      SELECT
        COALESCE(ctx.climate_domain, 'Unspecified') AS climate_domain,
        z.interval_end,
        SUM(CASE
              WHEN z.flux_type='area__ha' AND z.drained_state_meaning LIKE 'peat_drained%%'
              THEN z.value ELSE 0 END) AS drained_ha
      FROM zs_drained z
      LEFT JOIN drained_state_ctx AS ctx
        ON (z.drained_state_meaning = ctx.meaning)
        OR ({_rpad_sql('z.drained_state_nodes')} = ctx.key)
      GROUP BY 1,2
    ),
    latest_area AS (
      SELECT ap.climate_domain, ap.drained_ha AS latest_drained_ha
      FROM area_by_period ap
      JOIN (
        SELECT climate_domain, MAX(interval_end) AS max_end
        FROM area_by_period GROUP BY 1
      ) mx
        ON ap.climate_domain = mx.climate_domain AND ap.interval_end = mx.max_end
    ),
    em AS (
      SELECT
        COALESCE(ctx.climate_domain, 'Unspecified') AS climate_domain,
        SUM(CASE WHEN z.flux_type='drained_total_Mg_CO2e' THEN z.value ELSE 0 END) AS sum_Mg
      FROM zs_drained z
      LEFT JOIN drained_state_ctx AS ctx
        ON (z.drained_state_meaning = ctx.meaning)
        OR ({_rpad_sql('z.drained_state_nodes')} = ctx.key)
      GROUP BY 1
    )
    SELECT
      e.climate_domain,
      ( (e.sum_Mg / NULLIF({n_periods},0)) / NULLIF(a.latest_drained_ha, 0) ) AS intensity_tCO2e_per_ha_yr
    FROM em e
    JOIN latest_area a ON a.climate_domain = e.climate_domain;
    """

def sql_component_intensity_by_climate_avg(n_periods: int) -> str:
    return f"""
    WITH area_drained AS (
      SELECT
        COALESCE(ctx.climate_domain, 'Unspecified') AS climate_domain,
        z.interval_end,
        SUM(CASE
              WHEN z.flux_type='area__ha' AND z.drained_state_meaning LIKE 'peat_drained%'
              THEN z.value ELSE 0 END) AS drained_ha
      FROM zs_drained z
      LEFT JOIN drained_state_ctx AS ctx
        ON (z.drained_state_meaning = ctx.meaning)
        OR ({_rpad_sql('z.drained_state_nodes')} = ctx.key)
      GROUP BY 1,2
    ),
    area_burned AS (
      SELECT
        COALESCE(ctx.climate_domain, 'Unspecified') AS climate_domain,
        z.interval_end,
        SUM(CASE WHEN z.flux_type='area__ha' THEN z.value ELSE 0 END) AS burned_ha
      FROM zs_burned z
      LEFT JOIN burned_state_ctx AS ctx
        ON (z.burned_state_meaning = ctx.meaning)
        OR ({_rpad_sql('z.burned_state_nodes')} = ctx.key)
      GROUP BY 1,2
    ),
    latest_area AS (
      SELECT
        COALESCE(d.climate_domain, b.climate_domain) AS climate_domain,
        COALESCE(d.drained_ha, 0) AS latest_drained_ha,
        COALESCE(b.burned_ha, 0) AS latest_burned_ha
      FROM (
        SELECT ad.climate_domain, ad.drained_ha
        FROM area_drained ad
        JOIN (
          SELECT climate_domain, MAX(interval_end) AS max_end
          FROM area_drained GROUP BY 1
        ) mxd ON ad.climate_domain = mxd.climate_domain AND ad.interval_end = mxd.max_end
      ) d
      FULL OUTER JOIN (
        SELECT ab.climate_domain, ab.burned_ha
        FROM area_burned ab
        JOIN (
          SELECT climate_domain, MAX(interval_end) AS max_end
          FROM area_burned GROUP BY 1
        ) mxb ON ab.climate_domain = mxb.climate_domain AND ab.interval_end = mxb.max_end
      ) b
        ON d.climate_domain = b.climate_domain
    ),
    em_drained AS (
      SELECT
        COALESCE(ctx.climate_domain, 'Unspecified') AS climate_domain,
        SUM(CASE WHEN z.flux_type='drained_total_Mg_CO2e' THEN z.value ELSE 0 END) AS sum_Mg
      FROM zs_drained z
      LEFT JOIN drained_state_ctx AS ctx
        ON (z.drained_state_meaning = ctx.meaning)
        OR ({_rpad_sql('z.drained_state_nodes')} = ctx.key)
      GROUP BY 1
    ),
    em_burned AS (
      SELECT
        COALESCE(ctx.climate_domain, 'Unspecified') AS climate_domain,
        SUM(CASE WHEN z.flux_type='burned_total_Mg_CO2e' THEN z.value ELSE 0 END) AS sum_Mg
      FROM zs_burned z
      LEFT JOIN burned_state_ctx AS ctx
        ON (z.burned_state_meaning = ctx.meaning)
        OR ({_rpad_sql('z.burned_state_nodes')} = ctx.key)
      GROUP BY 1
    )
    SELECT
      la.climate_domain,
      'Drained' AS component,
      ( (ed.sum_Mg / NULLIF({n_periods},0)) / NULLIF(la.latest_drained_ha, 0) ) AS intensity_tCO2e_per_ha_yr
    FROM latest_area la
    LEFT JOIN em_drained ed ON la.climate_domain = ed.climate_domain
    UNION ALL
    SELECT
      la.climate_domain,
      'Burned' AS component,
      ( (eb.sum_Mg / NULLIF({n_periods},0)) / NULLIF(la.latest_burned_ha, 0) ) AS intensity_tCO2e_per_ha_yr
    FROM latest_area la
    LEFT JOIN em_burned eb ON la.climate_domain = eb.climate_domain;
    """

def sql_topn_total_emissions_split_avg(topn: int, with_lookup: bool, n_periods: int) -> str:
    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = ranked.gadm_adm0" if with_lookup else ""
    return f"""
    WITH d AS (
      SELECT gadm_adm0, SUM(value) AS drained_MgCO2e
      FROM zs_drained
      WHERE flux_type = 'drained_total_Mg_CO2e'
      GROUP BY 1
    ),
    b AS (
      SELECT gadm_adm0, SUM(value) AS burned_MgCO2e
      FROM zs_burned
      WHERE flux_type = 'burned_total_Mg_CO2e'
      GROUP BY 1
    ),
    f AS (
      SELECT
        COALESCE(d.gadm_adm0, b.gadm_adm0) AS gadm_adm0,
        COALESCE(d.drained_MgCO2e, 0) AS drained_MgCO2e,
        COALESCE(b.burned_MgCO2e, 0) AS burned_MgCO2e,
        COALESCE(d.drained_MgCO2e, 0) + COALESCE(b.burned_MgCO2e, 0) AS total_MgCO2e
      FROM d FULL OUTER JOIN b
      ON d.gadm_adm0 = b.gadm_adm0
    ),
    avgd AS (
      SELECT
        gadm_adm0,
        (drained_MgCO2e / {n_periods}) / 1e9 AS drained_avg_GtCO2e_per_yr,
        (burned_MgCO2e  / {n_periods}) / 1e9 AS burned_avg_GtCO2e_per_yr,
        (total_MgCO2e   / {n_periods}) / 1e9 AS total_avg_GtCO2e_per_yr
      FROM f
    ),
    ranked AS (
      SELECT
        gadm_adm0,
        drained_avg_GtCO2e_per_yr,
        burned_avg_GtCO2e_per_yr,
        total_avg_GtCO2e_per_yr,
        ROW_NUMBER() OVER (ORDER BY total_avg_GtCO2e_per_yr DESC) AS rnk
      FROM avgd
    )
    SELECT
      ranked.gadm_adm0
      {select_l},
      burned_avg_GtCO2e_per_yr,
      drained_avg_GtCO2e_per_yr,
      total_avg_GtCO2e_per_yr
    FROM ranked
    {join_l}
    WHERE rnk <= {topn}
    ORDER BY rnk;
    """

def sql_topn_peat_area_comp_latest(latest_year: int, topn: int, with_lookup: bool) -> str:
    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = r.gadm_adm0" if with_lookup else ""
    return f"""
    WITH base AS (
      SELECT
        gadm_adm0,
        CASE
          WHEN drained_state_meaning LIKE 'peat_drained%%'   THEN 'drained'
          WHEN drained_state_meaning LIKE 'peat_undrained%%' THEN 'undrained'
          ELSE 'other'
        END AS peat_state,
        SUM(value) AS area_ha
      FROM zs_drained
      WHERE flux_type = 'area__ha' AND interval_end = {latest_year}
      GROUP BY 1, 2
    ),
    agg AS (
      SELECT
        gadm_adm0,
        SUM(CASE WHEN peat_state='drained'   THEN area_ha ELSE 0 END) AS drained_ha,
        SUM(CASE WHEN peat_state='undrained' THEN area_ha ELSE 0 END) AS undrained_ha
      FROM base
      GROUP BY 1
    ),
    r AS (
      SELECT
        gadm_adm0,
        drained_ha,
        undrained_ha,
        (drained_ha + undrained_ha) AS total_peat_ha
      FROM agg
      WHERE (drained_ha + undrained_ha) > 0
    )
    SELECT
      r.gadm_adm0
      {select_l},
      drained_ha   / 1e6 AS drained_area_mha,
      undrained_ha / 1e6 AS undrained_area_mha,
      total_peat_ha/ 1e6 AS total_area_mha
    FROM r
    {join_l}
    ORDER BY total_area_mha DESC
    LIMIT {topn};
    """

def sql_global_peat_area_split(latest_year: int) -> str:
    return f"""
    WITH base AS (
      SELECT
        CASE
          WHEN drained_state_meaning LIKE 'peat_drained%%'   THEN 'drained'
          WHEN drained_state_meaning LIKE 'peat_undrained%%' THEN 'undrained'
          ELSE 'other'
        END AS peat_state,
        SUM(value) AS area_ha
      FROM zs_drained
      WHERE flux_type = 'area__ha' AND interval_end = {latest_year}
      GROUP BY 1
    ),
    agg AS (
      SELECT peat_state, SUM(area_ha) AS area_ha
      FROM base
      GROUP BY 1
    ),
    states AS (
      SELECT * FROM (VALUES ('drained'), ('undrained')) AS s(peat_state)
    )
    SELECT
      s.peat_state,
      COALESCE(a.area_ha, 0) / 1e6 AS area_mha
    FROM states s
    LEFT JOIN agg a ON a.peat_state = s.peat_state
    ORDER BY s.peat_state;
    """

def sql_global_component_emissions_avg(n_periods: int) -> str:
    return f"""
    WITH d AS (
      SELECT 'Drained' AS component, SUM(value) AS total_Mg
      FROM zs_drained
      WHERE flux_type = 'drained_total_Mg_CO2e'
    ),
    b AS (
      SELECT 'Burned' AS component, SUM(value) AS total_Mg
      FROM zs_burned
      WHERE flux_type = 'burned_total_Mg_CO2e'
    ),
    unioned AS (
      SELECT * FROM d
      UNION ALL
      SELECT * FROM b
    ),
    components AS (
      SELECT * FROM (VALUES ('Drained'), ('Burned')) AS t(component)
    )
    SELECT
      c.component,
      COALESCE(u.total_Mg, 0) / NULLIF({n_periods}, 0) / 1e9 AS avg_GtCO2e_per_yr
    FROM components c
    LEFT JOIN unioned u ON u.component = c.component
    ORDER BY c.component;
    """

def sql_global_totals_by_period_long() -> str:
    return """
    WITH d AS (
      SELECT interval_end, SUM(value) AS Mg
      FROM zs_drained
      WHERE flux_type = 'drained_total_Mg_CO2e'
      GROUP BY 1
    ),
    b AS (
      SELECT interval_end, SUM(value) AS Mg
      FROM zs_burned
      WHERE flux_type = 'burned_total_Mg_CO2e'
      GROUP BY 1
    )
    SELECT interval_end, 'Drained' AS component, d.Mg / 1e9 AS GtCO2e FROM d
    UNION ALL
    SELECT interval_end, 'Burned'  AS component, b.Mg / 1e9 AS GtCO2e FROM b
    ORDER BY interval_end, component;
    """

def sql_topn_avg_component_emissions(component: str, topn: int, with_lookup: bool, n_periods: int) -> str:
    assert component in {"drained", "burned"}
    base   = "zs_drained" if component == "drained" else "zs_burned"
    ftype  = "drained_total_Mg_CO2e" if component == "drained" else "burned_total_Mg_CO2e"
    alias  = "drained_avg_GtCO2e_per_yr" if component == "drained" else "burned_avg_GtCO2e_per_yr"
    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = ranked.gadm_adm0" if with_lookup else ""
    return f"""
    WITH t AS (
      SELECT gadm_adm0, SUM(value) AS Mg
      FROM {base}
      WHERE flux_type = '{ftype}'
      GROUP BY 1
    ),
    avgd AS (
      SELECT gadm_adm0, (Mg / {n_periods}) / 1e9 AS avg_Gt_per_yr
      FROM t
    ),
    ranked AS (
      SELECT gadm_adm0, avg_Gt_per_yr,
             ROW_NUMBER() OVER (ORDER BY avg_Gt_per_yr DESC) AS rnk
      FROM avgd
    )
    SELECT ranked.gadm_adm0{select_l}, avg_Gt_per_yr AS {alias}
    FROM ranked
    {join_l}
    WHERE rnk <= {topn}
    ORDER BY rnk;
    """

def sql_country_emissions_intensity_avg(
    n_periods: int,
    with_lookup: bool,
    topn: Optional[int] = None,
    min_area_ha: float = 10000.0
) -> str:
    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = e.gadm_adm0" if with_lookup else ""
    limit    = f"LIMIT {int(topn)}" if topn is not None else ""
    return f"""
    WITH area_by_period AS (
      SELECT interval_end, gadm_adm0,
             SUM(CASE WHEN drained_state_meaning LIKE 'peat_drained%%' THEN value ELSE 0 END) AS drained_ha
      FROM zs_drained
      WHERE flux_type = 'area__ha'
      GROUP BY 1,2
    ),
    latest_area AS (
      SELECT a.gadm_adm0, a.drained_ha AS latest_drained_ha
      FROM area_by_period a
      JOIN (SELECT gadm_adm0, MAX(interval_end) AS max_end FROM area_by_period GROUP BY 1) mx
        ON a.gadm_adm0 = mx.gadm_adm0 AND a.interval_end = mx.max_end
    ),
    d AS (
      SELECT gadm_adm0, SUM(value) AS drained_Mg_per_periods
      FROM zs_drained
      WHERE flux_type = 'drained_total_Mg_CO2e'
      GROUP BY 1
    ),
    avg_em As (
      SELECT gadm_adm0, (drained_Mg_per_periods / {n_periods}) AS avg_Mg_per_yr
      FROM d
    )
    SELECT
      e.gadm_adm0{select_l},
      (e.avg_Mg_per_yr / NULLIF(a.latest_drained_ha, 0))        AS intensity_tCO2e_per_ha_yr,
      (e.avg_Mg_per_yr / 1e9)                                   AS total_avg_GtCO2e_per_yr,
      (a.latest_drained_ha / 1e6)                               AS latest_drained_area_mha
    FROM avg_em e
    JOIN latest_area a ON a.gadm_adm0 = e.gadm_adm0
    {join_l}
    WHERE a.latest_drained_ha >= {min_area_ha}
    ORDER BY intensity_tCO2e_per_ha_yr DESC
    {limit};
    """

def sql_country_emissions_vs_area_avg(n_periods: int, with_lookup: bool) -> str:
    select_l = ", l.country, l.iso3" if with_lookup else ""
    join_l   = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = f.gadm_adm0" if with_lookup else ""
    return f"""
    WITH area_by_period AS (
      SELECT interval_end, gadm_adm0,
             SUM(CASE
                   WHEN drained_state_meaning LIKE 'peat_drained%%' THEN value
                   ELSE 0 END) AS drained_ha
      FROM zs_drained
      WHERE flux_type = 'area__ha'
      GROUP BY 1,2
    ),
    area_avg AS (
      SELECT gadm_adm0, AVG(drained_ha) AS avg_drained_ha
      FROM area_by_period
      GROUP BY 1
    ),
    d AS (
      SELECT gadm_adm0, SUM(value) AS drained_Mg
      FROM zs_drained
      WHERE flux_type = 'drained_total_Mg_CO2e'
      GROUP BY 1
    ),
    b AS (
      SELECT gadm_adm0, SUM(value) AS burned_Mg
      FROM zs_burned
      WHERE flux_type = 'burned_total_Mg_CO2e'
      GROUP BY 1
    ),
    f AS (
      SELECT
        COALESCE(d.gadm_adm0, b.gadm_adm0) AS gadm_adm0,
        COALESCE(d.drained_Mg, 0) AS drained_Mg,
        COALESCE(b.burned_Mg, 0)  AS burned_Mg
      FROM d FULL OUTER JOIN b ON d.gadm_adm0 = b.gadm_adm0
    ),
    avg_em AS (
      SELECT gadm_adm0, (drained_Mg + burned_Mg) / {n_periods} AS avg_Mg_per_yr
      FROM f
    )
    SELECT
      f.gadm_adm0{select_l},
      (avg_Mg_per_yr / 1e9)           AS total_avg_GtCO2e_per_yr,
      (a.avg_drained_ha / 1e6)        AS avg_drained_area_mha
    FROM avg_em f
    LEFT JOIN area_avg a
      ON a.gadm_adm0 = f.gadm_adm0
    {join_l}
    WHERE a.avg_drained_ha IS NOT NULL AND a.avg_drained_ha > 0
    ORDER BY total_avg_GtCO2e_per_yr DESC;
    """
