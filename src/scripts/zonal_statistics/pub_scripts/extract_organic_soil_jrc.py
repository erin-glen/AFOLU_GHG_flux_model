# -*- coding: utf-8 -*-
"""
Loader for the JRC pre-aggregated NGHGI organic-soil tables.

Source data: ``<jrc_dir>/AnnexI_2026/`` (six 4.A-F xlsx + one Table 3.D xlsx)
plus ``<jrc_dir>/BTR1_2024_Table3D.xlsx``. See the README at
``<jrc_dir>/README.md`` for provenance.

Two products:

  * ``load_jrc_landuse_tables`` -- Tables 4.A-4.F top-level "Total <land-use>"
    rows per country/year/coarse-land-use. Columns:
        iso3, year, land_use, area_total_kha, area_organic_kha,
        cstock_organic_ktC, source

  * ``load_jrc_table_3d`` -- Table 3.D.1.f "Cultivation of organic soils
    (i.e. histosols)" per country/year. Columns:
        iso3, year, area_ha, n2o_kt, source

Tables 4.A-C/E/F label the category column "Land Use Category"; Table 4.D
uses "GHG Source/Sink Category". Both are renamed to ``category`` here.

Country values arrive as ``"Australia (AUS)"`` -- the trailing ``(XXX)`` is
taken as the iso3.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple

import pandas as pd


DEFAULT_JRC_DIR = os.environ.get(
    "AFOLU_NGHGI_JRC_DIR",
    "/mnt/c/GIS/Data/Global/Wetlands/organic_soil_nghgi/JRC_aggregations/2026-04-30",
)


JRC_CATEGORY_TO_LANDUSE: Dict[str, str] = {
    "4.A": "Forest",
    "4.B": "Cropland",
    "4.C": "Grassland",
    "4.D": "Wetland",
    "4.E": "Settlement",
    "4.F": "Otherland",
}


ANNEXI_TABLE_FILES: Dict[str, str] = {
    "Table4.A": "NGHGI_data_Table_4_A_-_Forest_Land_2026-04-28.xlsx",
    "Table4.B": "NGHGI_data_Table_4_B_-_Cropland_2026-04-28.xlsx",
    "Table4.C": "NGHGI_data_Table_4_C_-_Grassland_2026-04-28.xlsx",
    "Table4.D": "NGHGI_data_Table_4_D_-_Wetlands_2026-04-28.xlsx",
    "Table4.E": "NGHGI_data_Table_4_E_-_Settlements_2026-04-28.xlsx",
    "Table4.F": "NGHGI_data_Table_4_F_-_Other_Land_2026-04-28.xlsx",
}
ANNEXI_TABLE3D_FILE = (
    "NGHGI_data_Table_3_D_-_Direct_and_indirect_N2O_emissions_from_agricultural_soils_2026-04-28.xlsx"
)
BTR1_TABLE3D_FILE = "BTR1_2024_Table3D.xlsx"


_COUNTRY_ISO3_RE = re.compile(r"\(([A-Z]{3})\)\s*$")
_CATEGORY_PREFIX_RE = re.compile(r"^(4\.[A-F])\.")
_TOTAL_LU_PATTERN = r"^4\.[A-F]\.\s*Total\b"
_T3D_1F_PATTERN = r"^3\.D\.1\.f\."


def _parse_iso3(country: object) -> Optional[str]:
    if not isinstance(country, str):
        return None
    m = _COUNTRY_ISO3_RE.search(country.strip())
    return m.group(1) if m else None


def _parse_category_prefix(label: object) -> Optional[str]:
    if not isinstance(label, str):
        return None
    m = _CATEGORY_PREFIX_RE.match(label.strip())
    return m.group(1) if m else None


def _read_xlsx_sheet1(path: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="Sheet1", engine="openpyxl")


def load_jrc_landuse_tables(annexi_dir: str) -> pd.DataFrame:
    """
    Read all six AnnexI 2026 land-use xlsx files and return one tidy frame
    with the top-level ("4.X. Total ...") rows only.
    """
    frames = []
    for label, fn in ANNEXI_TABLE_FILES.items():
        path = os.path.join(annexi_dir, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(f"JRC file not found: {path}")
        df = _read_xlsx_sheet1(path)

        cat_col = (
            "Land Use Category"
            if "Land Use Category" in df.columns
            else "GHG Source/Sink Category"
        )
        df = df.rename(
            columns={
                "Country": "country",
                "Inventory_year": "inventory_year",
                "Sheet": "sheet",
                "Year": "year",
                cat_col: "category",
                "Total Area (kha)": "area_total_kha",
                "Organic Soil Area (kha)": "area_organic_kha",
                "CSC Soils-Organic (kt C)": "cstock_organic_ktC",
            }
        )
        keep_mask = df["category"].astype(str).str.contains(
            _TOTAL_LU_PATTERN, regex=True, na=False
        )
        df = df.loc[keep_mask].copy()

        df["iso3"] = df["country"].map(_parse_iso3)
        df["category_prefix"] = df["category"].map(_parse_category_prefix)
        df["land_use"] = df["category_prefix"].map(JRC_CATEGORY_TO_LANDUSE)
        df = df.dropna(subset=["iso3", "land_use"]).copy()
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        for col in ("area_total_kha", "area_organic_kha", "cstock_organic_ktC"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["source"] = "JRC_AnnexI_2026"
        frames.append(
            df[
                [
                    "iso3",
                    "year",
                    "land_use",
                    "area_total_kha",
                    "area_organic_kha",
                    "cstock_organic_ktC",
                    "source",
                ]
            ]
        )

    return pd.concat(frames, ignore_index=True)


def load_jrc_table_3d(
    annexi_dir: str,
    btr1_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read Table 3.D.1.f "Cultivation of organic soils" rows from the AnnexI 2026
    file and (optionally) the BTR1 2024 file. When both sources cover the same
    (iso3, year), AnnexI 2026 wins (newer cycle).
    """
    annexi_path = os.path.join(annexi_dir, ANNEXI_TABLE3D_FILE)
    if not os.path.exists(annexi_path):
        raise FileNotFoundError(f"JRC Table 3.D file not found: {annexi_path}")
    sources = [(annexi_path, "JRC_AnnexI_2026")]
    if btr1_path and os.path.exists(btr1_path):
        sources.append((btr1_path, "JRC_BTR1_2024"))

    frames = []
    for path, src in sources:
        df = _read_xlsx_sheet1(path)
        df = df.rename(
            columns={
                "Country": "country",
                "Inventory_year": "inventory_year",
                "Sheet": "sheet",
                "Year": "year",
                "Emission Category": "category",
                "Area (ha)": "area_ha",
                "N2O emissions (kt)": "n2o_kt",
            }
        )
        keep_mask = df["category"].astype(str).str.contains(
            _T3D_1F_PATTERN, regex=True, na=False
        )
        df = df.loc[keep_mask].copy()
        df["iso3"] = df["country"].map(_parse_iso3)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["area_ha"] = pd.to_numeric(df["area_ha"], errors="coerce")
        df["n2o_kt"] = pd.to_numeric(df["n2o_kt"], errors="coerce")
        df["source"] = src
        df = df.dropna(subset=["iso3"]).copy()
        frames.append(df[["iso3", "year", "area_ha", "n2o_kt", "source"]])

    out = pd.concat(frames, ignore_index=True)

    # Lexicographic sort puts JRC_AnnexI_2026 < JRC_BTR1_2024, so keep="first"
    # prefers the AnnexI 2026 cycle when both sources cover the same country-year.
    out = out.sort_values(["iso3", "year", "source"], ascending=[True, True, True])
    out = out.drop_duplicates(subset=["iso3", "year"], keep="first")
    return out.reset_index(drop=True)


def default_jrc_paths(jrc_dir: str = DEFAULT_JRC_DIR) -> Tuple[str, Optional[str]]:
    """Resolve the AnnexI subdir and BTR1 xlsx inside a dated JRC drop folder."""
    annexi_dir = os.path.join(jrc_dir, "AnnexI_2026")
    btr1 = os.path.join(jrc_dir, BTR1_TABLE3D_FILE)
    return annexi_dir, (btr1 if os.path.exists(btr1) else None)
