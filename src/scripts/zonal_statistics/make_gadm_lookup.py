# -*- coding: utf-8 -*-
"""
Generate a lookup table mapping numeric GADM 4.1 adm0 raster codes to ISO3 and country names.

This script samples your adm0 Zarr raster at representative points within each
GADM 4.1 country polygon to discover the integer code used in the raster,
then writes a tidy CSV suitable for joining in publication tables.

Output columns:
- gadm_adm0 : uint32      (integer code in the raster)
- iso3      : string      (ISO3; falls back to GADM's GID_0 when ISO3 field not present)
- country   : string      (country/territory name)

Typical usage:
python -m src.scripts.zonal_statistics.make_gadm41_lookup \
  --adm0_zarr s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/GADM4_1_adm0_global/20250604/global_GADM41_adm0_20250604.zarr \
  --gadm_vector /path/to/gadm_4.1/gadm_410.gpkg \
  --out_csv s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/reference/GADM41_adm0_lookup.csv

Notes:
- The script assumes the adm0 Zarr is in EPSG:4326 with coords named x (lon), y (lat)
  — this matches the contextual layer used in your pipeline.
- For S3 IO, credentials are taken from the environment (AWS_*); you can also set
  DuckDB/HTTPFS if you later want to validate results, but not required here.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import s3fs
import shapely
import xarray as xr


# ---------------------------- I/O helpers ---------------------------------

def open_zarr_adm0(zarr_path: str) -> xr.DataArray:
    ds = xr.open_zarr(zarr_path, consolidated=None, storage_options={"anon": False})
    # Heuristics to find the 2-D variable with (x,y)
    if isinstance(ds, xr.DataArray):
        da = ds
    else:
        vars_xy = [v for v in ds.data_vars.values() if {"x", "y"}.issubset(v.dims)]
        if not vars_xy:
            raise RuntimeError(f"No 2-D (x,y) variable found in {zarr_path}")
        da = vars_xy[0]
    if "band" in da.dims:
        da = da.isel(band=0, drop=True)
    # Standardize dtype
    if da.dtype != np.uint32:
        da = da.astype("uint32")
    return da


def load_gadm_adm0(gadm_vector_path: str) -> gpd.GeoDataFrame:
    """
    Load GADM 4.1 adm0 polygons and pick best available ISO3 and name fields.
    The function is forgiving about field names present in different distributions.
    """
    gdf = gpd.read_file(gadm_vector_path)

    cols = {c.lower(): c for c in gdf.columns}
    # Common candidates in GADM4.1 and derivative distributions
    iso_candidates = ["iso3", "gid_0", "ADM0_A3".lower(), "ADM0_ISO3".lower(), "SOV_A3".lower()]
    name_candidates = ["name_0", "country", "name", "NAME_ENGLI".lower(), "shapeName".lower()]

    def pick(colset: List[str], label: str) -> str:
        for c in colset:
            if c in cols:
                return cols[c]
        raise RuntimeError(
            f"Could not find a suitable '{label}' column. "
            f"Looked for any of: {colset}. Columns present: {sorted(gdf.columns)}"
        )

    iso_field = pick(iso_candidates, "ISO3/GID_0")
    name_field = pick(name_candidates, "country name")

    # Normalize
    out = gdf[[iso_field, name_field, "geometry"]].rename(
        columns={iso_field: "iso3", name_field: "country"}
    ).copy()
    out["iso3"] = out["iso3"].astype(str).str.upper()
    out["country"] = out["country"].astype(str)

    # Dissolve to one multipart polygon per ISO3 (some datasets already are)
    out = out.dissolve(by="iso3", as_index=False, aggfunc={"country": "first"})
    out = out.set_crs("EPSG:4326", allow_override=True)  # ensure crs is set
    return out


def write_csv(df: pd.DataFrame, path: str) -> None:
    if path.startswith("s3://"):
        fs = fsspec.filesystem("s3")
        with fs.open(path, "w") as f:
            df.to_csv(f, index=False)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


# ---------------------------- core logic ----------------------------------

def sample_raster_at_points(da: xr.DataArray, points: List[Tuple[float, float]]) -> List[int]:
    """
    Nearest-neighbor sample of integer codes at (lon, lat) pairs.
    """
    vals: List[int] = []
    for lon, lat in points:
        v = da.sel(x=float(lon), y=float(lat), method="nearest").values.item()
        vals.append(int(v))
    return vals


def representative_points_for_iso3(gdf_iso: gpd.GeoDataFrame, max_per_iso: int = 12) -> Dict[str, List[Tuple[float, float]]]:
    """
    Generate up to `max_per_iso` representative (lon,lat) points per ISO3.
    Uses polygon representative_point() for robust interior samples across multi-geometries.
    """
    reps: Dict[str, List[Tuple[float, float]]] = {}
    for _, row in gdf_iso.iterrows():
        iso3 = row["iso3"]
        geom = row.geometry
        pts: List[Tuple[float, float]] = []

        # If multipolygon, take up to max_per_iso parts; else just one
        geoms = list(geom.geoms) if isinstance(geom, shapely.geometry.multipolygon.MultiPolygon) else [geom]
        # Cap to max_per_iso and prefer larger parts first
        geoms = sorted(geoms, key=lambda g: g.area if g is not None else 0.0, reverse=True)[:max_per_iso]

        for g in geoms:
            rp = g.representative_point()
            pts.append((rp.x, rp.y))
            if len(pts) >= max_per_iso:
                break

        if not pts:
            # Fallback: centroid if representative_point failed (rare)
            c = geom.centroid
            pts = [(c.x, c.y)]
        reps[iso3] = pts
    return reps


def make_lookup(adm0_da: xr.DataArray, gadm_iso_gdf: gpd.GeoDataFrame, max_samples_per_iso: int = 12) -> pd.DataFrame:
    """
    Build mapping: gadm_adm0 (int) -> iso3, country.

    Strategy: for each ISO3 polygon set, sample multiple interior points against the raster,
    then pick the majority integer code as the mapping for that ISO3.
    """
    reps = representative_points_for_iso3(gadm_iso_gdf, max_per_iso=max_samples_per_iso)

    rows = []
    for iso3, pts in reps.items():
        codes = sample_raster_at_points(adm0_da, pts)
        # Majority vote
        vals, counts = np.unique(np.array(codes, dtype=np.uint32), return_counts=True)
        majority_code = int(vals[np.argmax(counts)])
        rows.append({"iso3": iso3, "gadm_adm0": majority_code})

    df_codes = pd.DataFrame(rows)
    df = df_codes.merge(gadm_iso_gdf[["iso3", "country"]], on="iso3", how="left")

    # Sanity: ensure uniqueness (1 code per ISO3)
    dup = df.duplicated(subset=["iso3"], keep=False)
    if dup.any():
        raise RuntimeError("Non-unique mapping produced for some ISO3 codes.")

    # Optional: verify that every raster code >0 is covered
    unique_codes = np.array(np.unique(adm0_da.values)).astype(np.uint32)
    unique_codes = unique_codes[unique_codes > 0]
    missing = sorted(set(unique_codes.tolist()) - set(df["gadm_adm0"].tolist()))
    if missing:
        logging.warning("Raster contains %d adm0 codes not assigned to any ISO3 (examples: %s)",
                        len(missing), missing[:10])

    # Sort nicely
    df["gadm_adm0"] = df["gadm_adm0"].astype(np.uint32)
    df = df[["gadm_adm0", "iso3", "country"]].sort_values(["iso3"]).reset_index(drop=True)
    return df


# ---------------------------- CLI ----------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser("Generate GADM 4.1 adm0 lookup (raster code -> ISO3,country)")
    ap.add_argument("--adm0_zarr", required=True, help="Path to the adm0 Zarr (EPSG:4326; x,y coords)")
    ap.add_argument("--gadm_vector", required=True, help="Path to GADM 4.1 Adm0 vector (gpkg/shp/geojson; local or s3://)")
    ap.add_argument("--out_csv", required=True, help="Output CSV path (local or s3://)")
    ap.add_argument("--max_samples_per_iso", type=int, default=12, help="Rep points per ISO3 (default: 12)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    logging.info("Opening adm0 Zarr: %s", args.adm0_zarr)
    adm0_da = open_zarr_adm0(args.adm0_zarr)

    logging.info("Loading GADM Adm0 vector: %s", args.gadm_vector)
    gadm = load_gadm_adm0(args.gadm_vector)

    logging.info("Building lookup via representative-point sampling")
    df = make_lookup(adm0_da, gadm, max_samples_per_iso=args.max_samples_per_iso)

    logging.info("Writing CSV: %s", args.out_csv)
    write_csv(df, args.out_csv)

    logging.info("Done. Rows: %d", len(df))


if __name__ == "__main__":
    main()
