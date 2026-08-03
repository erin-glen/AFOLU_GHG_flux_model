# -*- coding: utf-8 -*-
"""Surgically rebuild organic-soil fields in a combined LULUCF parquet.

The input wide parquet is immutable.  The script aggregates a corrected
organic-soils master to the existing ADM0 x endpoint x climate sentinel grain,
replaces 15 organic fields, and recomputes seven dependent fields.  It refuses
to write unless the old master first reverse-reproduces the attached baseline
and every preservation, engineer, and arithmetic gate passes.

Without ``--execute`` the complete rebuild runs in memory and writes nothing.
An existing output or report is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DIMENSION_COLUMNS: Tuple[str, ...] = (
    "country_name",
    "adm0",
    "region_L1",
    "region_L2_L3",
    "year",
    "land_state_node",
    "land_state_meaning",
    "land_state_broad_class",
    "land_state_detailed_class",
    "tall_veg_type",
    "climate_domain",
)

ORGANIC_COLUMNS: Tuple[str, ...] = (
    "org_soil_drained_only__all_gases__MgCO2e_yr",
    "org_soil_drained_only__CO2_onsite__MgCO2_yr",
    "org_soil_drained_only__CO2_offsite__MgCO2_yr",
    "org_soil_drained_only__CO2__MgCO2_yr",
    "org_soil_drained_only__CH4__MgCO2e_yr",
    "org_soil_drained_only__N2O__MgCO2e_yr",
    "org_soil_burned_only__all_gases__MgCO2e_yr",
    "org_soil_burned_only__CO2__MgCO2_yr",
    "org_soil_burned_only__CH4__MgCO2e_yr",
    "org_soil_drained_burned__all_gases__MgCO2e_yr",
    "org_soil_drained_burned__CO2__MgCO2_yr",
    "org_soil_drained_burned__CH4__MgCO2e_yr",
    "org_soil_drained_burned__N2O__MgCO2e_yr",
    "org_soil__area_ha",
    "org_soil_emis__all_gases__MgCO2e_yr",
)

DEPENDENT_COLUMNS: Tuple[str, ...] = (
    "soil_emis__all_gases__MgCO2e_yr",
    "LULUCF_gross_emissions__CO2__MgCO2_yr",
    "LULUCF_gross_emissions__CH4__MgCO2e_yr",
    "LULUCF_gross_emissions__N2O__MgCO2e_yr",
    "LULUCF_gross_emissions__non_CO2__MgCO2e_yr",
    "LULUCF_gross_emissions__all_gases__MgCO2e_yr",
    "LULUCF_net_flux__MgCO2e_yr",
)

TARGET_COLUMNS: Tuple[str, ...] = ORGANIC_COLUMNS + DEPENDENT_COLUMNS
SENTINEL_KEYS: Tuple[str, ...] = ("adm0", "interval_end", "climate_domain")
WIDE_GRAIN: Tuple[str, ...] = (
    "adm0",
    "year",
    "land_state_node",
    "tall_veg_type",
    "climate_domain",
)

MASTER_MEASURES: Tuple[str, ...] = (
    "area__ha",
    "drained_total_Mg_CO2e",
    "drained_co2_onsite_Mg_CO2",
    "drained_co2_offsite_Mg_CO2",
    "drained_total_co2_Mg_CO2",
    "drained_total_ch4_Mg_CO2e",
    "drained_n2o_Mg_CO2e",
    "burned_total_Mg_CO2e",
    "burned_total_co2_Mg_CO2",
    "burned_total_ch4_Mg_CO2e",
)
MASTER_REQUIRED = {
    "iso3",
    "interval_end",
    "climate_domain",
    "drainage_class",
    "burned_state_meaning",
    *MASTER_MEASURES,
}
ALLOWED_MASTER_CLIMATES = {
    "boreal",
    "temperate",
    "tropical",
    "other_domain",
    "Unspecified",
}

REPORT_NAMES: Tuple[str, ...] = (
    "lulucf_wide_wdpa_rebuild_report.json",
    "old_master_reproduction_summary.csv",
    "engineer_country_endpoint_comparison.csv",
    "target_column_change_summary.csv",
    "global_endpoint_summary.csv",
    "dependent_identity_summary.csv",
)


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def endpoint_for_year(year: int) -> int:
    if 2016 <= int(year) <= 2020:
        return 2020
    if 2021 <= int(year) <= 2024:
        return 2024
    raise ValueError(f"Unsupported wide-artifact year: {year}")


def _expressions(*, include_nonpeat_area: bool) -> Dict[str, str]:
    drained = "drainage_class LIKE 'peat_drained%'"
    burned = "burned_state_meaning IS NOT NULL"
    peat = "drainage_class <> 'non_peat'"
    area = (
        "SUM(COALESCE(area__ha, 0))"
        if include_nonpeat_area
        else f"SUM(CASE WHEN {peat} THEN COALESCE(area__ha, 0) ELSE 0 END)"
    )
    organic_total = (
        f"SUM(CASE WHEN {peat} THEN COALESCE(drained_total_Mg_CO2e, 0) + "
        "COALESCE(burned_total_Mg_CO2e, 0) ELSE 0 END)"
    )
    co2 = (
        f"SUM(CASE WHEN {peat} THEN COALESCE(drained_total_co2_Mg_CO2, 0) + "
        "COALESCE(burned_total_co2_Mg_CO2, 0) ELSE 0 END)"
    )
    ch4 = (
        f"SUM(CASE WHEN {peat} THEN COALESCE(drained_total_ch4_Mg_CO2e, 0) + "
        "COALESCE(burned_total_ch4_Mg_CO2e, 0) ELSE 0 END)"
    )
    n2o = (
        f"SUM(CASE WHEN {peat} THEN COALESCE(drained_n2o_Mg_CO2e, 0) "
        "ELSE 0 END)"
    )
    non_co2 = (
        f"SUM(CASE WHEN {peat} THEN COALESCE(drained_total_ch4_Mg_CO2e, 0) + "
        "COALESCE(burned_total_ch4_Mg_CO2e, 0) + "
        "COALESCE(drained_n2o_Mg_CO2e, 0) ELSE 0 END)"
    )
    gas_total = (
        f"SUM(CASE WHEN {peat} THEN COALESCE(drained_total_co2_Mg_CO2, 0) + "
        "COALESCE(burned_total_co2_Mg_CO2, 0) + "
        "COALESCE(drained_total_ch4_Mg_CO2e, 0) + "
        "COALESCE(burned_total_ch4_Mg_CO2e, 0) + "
        "COALESCE(drained_n2o_Mg_CO2e, 0) ELSE 0 END)"
    )
    return {
        "org_soil_drained_only__all_gases__MgCO2e_yr": (
            f"SUM(CASE WHEN {drained} AND NOT ({burned}) THEN "
            "COALESCE(drained_total_Mg_CO2e, 0) ELSE 0 END)"
        ),
        "org_soil_drained_only__CO2_onsite__MgCO2_yr": (
            f"SUM(CASE WHEN {drained} AND NOT ({burned}) THEN "
            "COALESCE(drained_co2_onsite_Mg_CO2, 0) ELSE 0 END)"
        ),
        "org_soil_drained_only__CO2_offsite__MgCO2_yr": (
            f"SUM(CASE WHEN {drained} AND NOT ({burned}) THEN "
            "COALESCE(drained_co2_offsite_Mg_CO2, 0) ELSE 0 END)"
        ),
        "org_soil_drained_only__CO2__MgCO2_yr": (
            f"SUM(CASE WHEN {drained} AND NOT ({burned}) THEN "
            "COALESCE(drained_total_co2_Mg_CO2, 0) ELSE 0 END)"
        ),
        "org_soil_drained_only__CH4__MgCO2e_yr": (
            f"SUM(CASE WHEN {drained} AND NOT ({burned}) THEN "
            "COALESCE(drained_total_ch4_Mg_CO2e, 0) ELSE 0 END)"
        ),
        "org_soil_drained_only__N2O__MgCO2e_yr": (
            f"SUM(CASE WHEN {drained} AND NOT ({burned}) THEN "
            "COALESCE(drained_n2o_Mg_CO2e, 0) ELSE 0 END)"
        ),
        "org_soil_burned_only__all_gases__MgCO2e_yr": (
            f"SUM(CASE WHEN drainage_class = 'peat_undrained' AND {burned} THEN "
            "COALESCE(burned_total_Mg_CO2e, 0) ELSE 0 END)"
        ),
        "org_soil_burned_only__CO2__MgCO2_yr": (
            f"SUM(CASE WHEN drainage_class = 'peat_undrained' AND {burned} THEN "
            "COALESCE(burned_total_co2_Mg_CO2, 0) ELSE 0 END)"
        ),
        "org_soil_burned_only__CH4__MgCO2e_yr": (
            f"SUM(CASE WHEN drainage_class = 'peat_undrained' AND {burned} THEN "
            "COALESCE(burned_total_ch4_Mg_CO2e, 0) ELSE 0 END)"
        ),
        "org_soil_drained_burned__all_gases__MgCO2e_yr": (
            f"SUM(CASE WHEN {drained} AND {burned} THEN "
            "COALESCE(drained_total_Mg_CO2e, 0) + "
            "COALESCE(burned_total_Mg_CO2e, 0) ELSE 0 END)"
        ),
        "org_soil_drained_burned__CO2__MgCO2_yr": (
            f"SUM(CASE WHEN {drained} AND {burned} THEN "
            "COALESCE(drained_total_co2_Mg_CO2, 0) + "
            "COALESCE(burned_total_co2_Mg_CO2, 0) ELSE 0 END)"
        ),
        "org_soil_drained_burned__CH4__MgCO2e_yr": (
            f"SUM(CASE WHEN {drained} AND {burned} THEN "
            "COALESCE(drained_total_ch4_Mg_CO2e, 0) + "
            "COALESCE(burned_total_ch4_Mg_CO2e, 0) ELSE 0 END)"
        ),
        "org_soil_drained_burned__N2O__MgCO2e_yr": (
            f"SUM(CASE WHEN {drained} AND {burned} THEN "
            "COALESCE(drained_n2o_Mg_CO2e, 0) ELSE 0 END)"
        ),
        "org_soil__area_ha": area,
        "org_soil_emis__all_gases__MgCO2e_yr": organic_total,
        "soil_emis__all_gases__MgCO2e_yr": organic_total,
        "LULUCF_gross_emissions__CO2__MgCO2_yr": co2,
        "LULUCF_gross_emissions__CH4__MgCO2e_yr": ch4,
        "LULUCF_gross_emissions__N2O__MgCO2e_yr": n2o,
        "LULUCF_gross_emissions__non_CO2__MgCO2e_yr": non_co2,
        "LULUCF_gross_emissions__all_gases__MgCO2e_yr": gas_total,
        # Organic sentinel rows have no removals, so net equals gas total.
        "LULUCF_net_flux__MgCO2e_yr": gas_total,
    }


def build_master_aggregation_sql(
    master_path: str | Path, *, include_nonpeat_area: bool
) -> str:
    measures = ",\n        ".join(
        f'{expression} AS "{column}"'
        for column, expression in _expressions(
            include_nonpeat_area=include_nonpeat_area
        ).items()
    )
    return f"""
    SELECT
        iso3 AS adm0,
        CAST(interval_end AS INTEGER) AS interval_end,
        CASE climate_domain
            WHEN 'boreal' THEN 'Boreal'
            WHEN 'temperate' THEN 'Temperate'
            WHEN 'tropical' THEN 'Subtropical/tropical'
            WHEN 'other_domain' THEN 'Unassigned'
            WHEN 'Unspecified' THEN 'Unassigned'
        END AS climate_domain,
        {measures}
    FROM read_parquet({_sql_literal(master_path)})
    WHERE interval_end IN (2020, 2024)
    GROUP BY ALL
    ORDER BY adm0, interval_end, climate_domain
    """


def aggregate_master(
    path: str | Path, *, include_nonpeat_area: bool
) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        described = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({_sql_literal(path)})"
        ).fetchall()
        columns = {str(row[0]) for row in described}
        missing = MASTER_REQUIRED - columns
        if missing:
            raise ValueError(f"Master is missing columns: {sorted(missing)}")
        climates = {
            row[0]
            for row in con.execute(
                "SELECT DISTINCT climate_domain FROM read_parquet(?) "
                "WHERE interval_end IN (2020, 2024)",
                [str(path)],
            ).fetchall()
        }
        unexpected = climates - ALLOWED_MASTER_CLIMATES
        if unexpected or None in climates:
            raise ValueError(f"Unexpected master climate values: {sorted(unexpected)}")
        null_iso = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?) "
            "WHERE interval_end IN (2020, 2024) "
            "AND (iso3 IS NULL OR TRIM(iso3) = '')",
            [str(path)],
        ).fetchone()[0]
        if null_iso:
            raise ValueError(f"Master contains {null_iso} endpoint rows without ISO3")
        result = con.execute(
            build_master_aggregation_sql(
                path, include_nonpeat_area=include_nonpeat_area
            )
        ).df()
    finally:
        con.close()

    if result.empty or result.duplicated(list(SENTINEL_KEYS)).any():
        raise ValueError("Master aggregation is empty or contains duplicate keys")
    if set(result["interval_end"].astype(int)) != {2020, 2024}:
        raise ValueError("Master must contain endpoint years 2020 and 2024")
    target = result.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Master aggregation contains invalid target values")
    return result


def _float32(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(
        table[column].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float32,
    )


def _dimensions(table: pa.Table) -> pd.DataFrame:
    return table.select(DIMENSION_COLUMNS).to_pandas()


def _sentinel(dimensions: pd.DataFrame) -> np.ndarray:
    return (
        dimensions["land_state_node"].eq("-999")
        & dimensions["tall_veg_type"].eq("Unassigned")
    ).to_numpy(dtype=bool)


def validate_baseline(table: pa.Table) -> Tuple[np.ndarray, Dict[str, object]]:
    missing = set(DIMENSION_COLUMNS + TARGET_COLUMNS) - set(table.column_names)
    if missing or table.num_columns != 97:
        raise ValueError(
            f"Invalid baseline schema: missing={sorted(missing)}, "
            f"columns={table.num_columns}"
        )
    for column in TARGET_COLUMNS:
        if table.schema.field(column).type != pa.float32() or table[column].null_count:
            raise ValueError(f"Target must be non-null float32: {column}")
    dimensions = _dimensions(table)
    duplicates = int(dimensions.duplicated(list(WIDE_GRAIN)).sum())
    if duplicates:
        raise ValueError(f"Baseline has {duplicates} duplicate grain rows")
    sentinel = _sentinel(dimensions)
    years = pd.to_numeric(dimensions.loc[sentinel, "year"], errors="raise").astype(int)
    if set(years.unique()) != set(range(2016, 2025)):
        raise ValueError(f"Unexpected sentinel years: {sorted(years.unique())}")
    for column in ORGANIC_COLUMNS:
        if np.max(np.abs(_float32(table, column)[~sentinel]), initial=0.0) != 0.0:
            raise ValueError(f"Organic values exist outside sentinel rows: {column}")
    removals = _float32(table, "LULUCF_gross_removals__MgCO2_yr")
    if np.max(np.abs(removals[sentinel]), initial=0.0) != 0.0:
        raise ValueError("Sentinel rows contain removals; net-flux replacement is unsafe")
    endpoint_rows = sentinel & dimensions["year"].isin(["2020", "2024"]).to_numpy()
    return sentinel, {
        "rows": table.num_rows,
        "columns": table.num_columns,
        "duplicate_wide_keys": duplicates,
        "sentinel_rows": int(sentinel.sum()),
        "sentinel_endpoint_rows": int(endpoint_rows.sum()),
        "status": "PASS",
    }


def _key_frame(table: pa.Table, sentinel: np.ndarray) -> pd.DataFrame:
    dimensions = _dimensions(table)
    years = pd.to_numeric(dimensions.loc[sentinel, "year"], errors="raise").astype(int)
    return pd.DataFrame(
        {
            "_row_id": np.flatnonzero(sentinel),
            "adm0": dimensions.loc[sentinel, "adm0"].astype(str).to_numpy(),
            "interval_end": years.map(endpoint_for_year).to_numpy(dtype=np.int64),
            "climate_domain": dimensions.loc[sentinel, "climate_domain"].astype(str).to_numpy(),
        }
    )


def _match_aggregation(
    table: pa.Table, sentinel: np.ndarray, aggregation: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    keys = _key_frame(table, sentinel)
    wide_keys = keys.loc[:, list(SENTINEL_KEYS)].drop_duplicates()
    master_keys = aggregation.loc[:, list(SENTINEL_KEYS)].drop_duplicates()
    coverage = wide_keys.merge(
        master_keys,
        on=list(SENTINEL_KEYS),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    left = int(coverage["_merge"].eq("left_only").sum())
    right = int(coverage["_merge"].eq("right_only").sum())
    if left or right:
        raise ValueError(
            f"Sentinel-key mismatch: missing_from_master={left}, "
            f"missing_from_wide={right}"
        )
    matched = keys.merge(
        aggregation,
        on=list(SENTINEL_KEYS),
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False,
    ).sort_values("_row_id", kind="stable")
    if not matched["_merge"].eq("both").all():
        raise ValueError("Annual sentinel mapping is incomplete")
    if matched.loc[:, list(TARGET_COLUMNS)].isna().any().any():
        raise ValueError("Annual sentinel mapping contains null values")
    return matched, {
        "annual_sentinel_rows": len(keys),
        "wide_endpoint_keys": len(wide_keys),
        "master_endpoint_keys": len(master_keys),
        "missing_from_master": left,
        "missing_from_wide": right,
        "status": "PASS",
    }


def _ulp(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if (left < 0).any() or (right < 0).any():
        raise ValueError("ULP gate expects non-negative values")
    return np.abs(
        left.view(np.uint32).astype(np.int64)
        - right.view(np.uint32).astype(np.int64)
    )


def reverse_reproduce_old(
    baseline: pa.Table, old: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    dimensions = _dimensions(baseline)
    endpoint = _sentinel(dimensions) & dimensions["year"].isin(["2020", "2024"]).to_numpy()
    keys = pd.DataFrame(
        {
            "_row_id": np.flatnonzero(endpoint),
            "adm0": dimensions.loc[endpoint, "adm0"].astype(str).to_numpy(),
            "interval_end": pd.to_numeric(
                dimensions.loc[endpoint, "year"], errors="raise"
            ).astype(int).to_numpy(),
            "climate_domain": dimensions.loc[endpoint, "climate_domain"].astype(str).to_numpy(),
        }
    )
    matched = keys.merge(
        old,
        on=list(SENTINEL_KEYS),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if len(matched) != len(keys) or not matched["_merge"].eq("both").all():
        raise ValueError("Old master does not reproduce the 690 baseline keys")
    summaries = []
    row_ids = matched["_row_id"].to_numpy(dtype=np.int64)
    for column in TARGET_COLUMNS:
        actual = _float32(baseline, column)[row_ids]
        expected = matched[column].to_numpy(dtype=np.float64).astype(np.float32)
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        relative = difference / np.maximum(np.abs(actual.astype(np.float64)), 1.0)
        ulp = _ulp(actual, expected)
        exact = actual.view(np.uint32) == expected.view(np.uint32)
        summaries.append(
            {
                "column": column,
                "rows": len(actual),
                "exact_float32_rows": int(exact.sum()),
                "mismatched_float32_rows": int((~exact).sum()),
                "max_absolute_difference": float(difference.max(initial=0.0)),
                "max_relative_difference": float(relative.max(initial=0.0)),
                "max_float32_ulp_distance": int(ulp.max(initial=0)),
            }
        )
    result = pd.DataFrame(summaries)
    max_ulp = int(result["max_float32_ulp_distance"].max())
    max_relative = float(result["max_relative_difference"].max())
    if max_ulp > 1 or max_relative > 2e-7:
        raise ValueError(
            f"Old-source reproduction failed: ULP={max_ulp}, rel={max_relative}"
        )
    return result, {
        "endpoint_rows": len(keys),
        "columns_checked": len(TARGET_COLUMNS),
        "max_float32_ulp_distance": max_ulp,
        "max_relative_difference": max_relative,
        "gate_max_ulp": 1,
        "gate_max_relative": 2e-7,
        "status": "PASS",
    }


def rebuild_table(
    baseline: pa.Table, sentinel: np.ndarray, corrected: pd.DataFrame
) -> Tuple[pa.Table, Dict[str, object]]:
    matched, key_summary = _match_aggregation(baseline, sentinel, corrected)
    row_ids = matched["_row_id"].to_numpy(dtype=np.int64)
    rebuilt = baseline
    for column in TARGET_COLUMNS:
        values = _float32(rebuilt, column).copy()
        values[row_ids] = matched[column].to_numpy(dtype=np.float64).astype(np.float32)
        rebuilt = rebuilt.set_column(
            rebuilt.schema.get_field_index(column),
            rebuilt.schema.field(column),
            pa.array(values, type=pa.float32()),
        )
    return rebuilt, key_summary


def preservation_checks(
    baseline: pa.Table, rebuilt: pa.Table, sentinel: np.ndarray
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if baseline.num_rows != rebuilt.num_rows:
        raise ValueError("Row count changed")
    if baseline.column_names != rebuilt.column_names:
        raise ValueError("Column order changed")
    if not baseline.schema.equals(rebuilt.schema, check_metadata=True):
        raise ValueError("Schema or schema metadata changed")
    untouched = [column for column in baseline.column_names if column not in TARGET_COLUMNS]
    changed_untouched = [
        column for column in untouched if not baseline[column].equals(rebuilt[column])
    ]
    if changed_untouched:
        raise ValueError(f"Non-target columns changed: {changed_untouched}")

    rows = []
    outside_changes = 0
    for column in TARGET_COLUMNS:
        old = _float32(baseline, column)
        new = _float32(rebuilt, column)
        changed = old.view(np.uint32) != new.view(np.uint32)
        changed_outside = int((changed & ~sentinel).sum())
        outside_changes += changed_outside
        sentinel_values = new[sentinel].astype(np.float64)
        if not np.isfinite(new.astype(np.float64)).all():
            raise ValueError(f"Non-finite values in {column}")
        if (sentinel_values < 0).any():
            raise ValueError(f"Negative rebuilt sentinel values in {column}")
        rows.append(
            {
                "column": column,
                "changed_cells": int(changed.sum()),
                "changed_sentinel_cells": int((changed & sentinel).sum()),
                "changed_non_sentinel_cells": changed_outside,
                "old_sum_float64": float(old.astype(np.float64).sum()),
                "new_sum_float64": float(new.astype(np.float64).sum()),
                "delta_sum_float64": float(
                    new.astype(np.float64).sum() - old.astype(np.float64).sum()
                ),
            }
        )
    if outside_changes:
        raise ValueError(f"{outside_changes} target cells changed outside sentinel rows")
    return pd.DataFrame(rows), {
        "rows_preserved": True,
        "column_order_preserved": True,
        "schema_metadata_preserved": True,
        "non_target_columns_checked": len(untouched),
        "changed_non_target_columns": changed_untouched,
        "changed_target_cells_outside_sentinel": outside_changes,
        "status": "PASS",
    }


def _country_endpoints(table: pa.Table) -> pd.DataFrame:
    dimensions = _dimensions(table)
    mask = _sentinel(dimensions) & dimensions["year"].isin(["2020", "2024"]).to_numpy()
    rows = pd.DataFrame(
        {
            "adm0": dimensions.loc[mask, "adm0"].astype(str).to_numpy(),
            "interval_end": pd.to_numeric(
                dimensions.loc[mask, "year"], errors="raise"
            ).astype(int).to_numpy(),
            "output_area_ha": _float32(table, "org_soil__area_ha")[mask].astype(np.float64),
            "output_MgCO2e": _float32(
                table, "org_soil_emis__all_gases__MgCO2e_yr"
            )[mask].astype(np.float64),
        }
    )
    return rows.groupby(["adm0", "interval_end"], as_index=False).agg(
        output_area_ha=("output_area_ha", "sum"),
        output_MgCO2e=("output_MgCO2e", "sum"),
    )


def engineer_checks(
    rebuilt: pa.Table, engineer_path: str | Path
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    output = _country_endpoints(rebuilt)
    engineer = pq.read_table(engineer_path).to_pandas()
    required = {
        "interval_end_year",
        "area_ha",
        "gross_emissions_MgCO2e",
        "aoi_id",
        "aoi_type",
    }
    if required - set(engineer.columns):
        raise ValueError("Engineer comparator schema is incomplete")
    country_rows = engineer["aoi_type"].eq("admin") & engineer[
        "aoi_id"
    ].astype("string").str.fullmatch(r"[A-Z]{3}", na=False)
    engineer = engineer.loc[country_rows].rename(
        columns={
            "aoi_id": "adm0",
            "interval_end_year": "interval_end",
            "area_ha": "engineer_area_ha",
            "gross_emissions_MgCO2e": "engineer_MgCO2e",
        }
    )[
        ["adm0", "interval_end", "engineer_area_ha", "engineer_MgCO2e"]
    ]
    if engineer.empty:
        raise ValueError("Engineer comparator contains no ADM0 country rows")
    engineer["adm0"] = engineer["adm0"].astype(str)
    if engineer.duplicated(["adm0", "interval_end"]).any():
        raise ValueError("Engineer comparator has duplicate keys")
    result = engineer.merge(
        output,
        on=["adm0", "interval_end"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    matched = result["_merge"].eq("both")
    result["emission_difference_MgCO2e"] = (
        result["output_MgCO2e"] - result["engineer_MgCO2e"]
    )
    result["emission_absolute_difference_MgCO2e"] = result[
        "emission_difference_MgCO2e"
    ].abs()
    emission_denominator = result["engineer_MgCO2e"].abs()
    result["emission_absolute_percent_difference"] = np.where(
        emission_denominator > 0,
        result["emission_absolute_difference_MgCO2e"] / emission_denominator * 100,
        np.where(result["emission_absolute_difference_MgCO2e"].eq(0), 0.0, np.inf),
    )
    result["emission_tolerance_MgCO2e"] = np.maximum(
        0.1, emission_denominator * 1e-5
    )
    result["emission_pass"] = matched & (
        result["emission_absolute_difference_MgCO2e"]
        <= result["emission_tolerance_MgCO2e"]
    )
    result["area_difference_ha"] = result["output_area_ha"] - result["engineer_area_ha"]
    result["area_absolute_difference_ha"] = result["area_difference_ha"].abs()
    area_denominator = result["engineer_area_ha"].abs()
    result["area_absolute_percent_difference"] = np.where(
        area_denominator > 0,
        result["area_absolute_difference_ha"] / area_denominator * 100,
        np.where(result["area_absolute_difference_ha"].eq(0), 0.0, np.inf),
    )
    result["area_pass"] = matched & (
        result["area_absolute_percent_difference"] <= 0.05
    )

    matched_rows = result.loc[matched]
    engineer_only = int(result["_merge"].eq("left_only").sum())
    output_only_rows = result.loc[result["_merge"].eq("right_only")]
    output_only_max = float(
        output_only_rows[["output_area_ha", "output_MgCO2e"]]
        .fillna(0.0)
        .abs()
        .to_numpy(dtype=np.float64)
        .max(initial=0.0)
    )
    emission_pass = int(matched_rows["emission_pass"].sum())
    area_pass = int(matched_rows["area_pass"].sum())
    if (
        engineer_only
        or emission_pass != len(matched_rows)
        or area_pass != len(matched_rows)
        or output_only_max != 0.0
    ):
        raise ValueError(
            "Engineer gate failed: "
            f"engineer_only={engineer_only}, emission={emission_pass}/"
            f"{len(matched_rows)}, area={area_pass}/{len(matched_rows)}, "
            f"output_only_max={output_only_max}"
        )
    result = result.sort_values(["_merge", "adm0", "interval_end"]).reset_index(drop=True)
    return result, {
        "engineer_rows": len(engineer),
        "output_country_endpoint_rows": len(output),
        "matched_rows": len(matched_rows),
        "engineer_only_rows": engineer_only,
        "output_only_zero_rows": len(output_only_rows),
        "emission_pass_rows": emission_pass,
        "area_pass_rows": area_pass,
        "emission_gate": "max(0.1 Mg, 0.001%)",
        "area_gate_percent": 0.05,
        "max_emission_absolute_difference_MgCO2e": float(
            matched_rows["emission_absolute_difference_MgCO2e"].max()
        ),
        "max_emission_absolute_percent_difference": float(
            matched_rows["emission_absolute_percent_difference"].max()
        ),
        "max_area_absolute_difference_ha": float(
            matched_rows["area_absolute_difference_ha"].max()
        ),
        "max_area_absolute_percent_difference": float(
            matched_rows["area_absolute_percent_difference"].max()
        ),
        "status": "PASS",
    }


def global_checks(
    rebuilt: pa.Table, corrected: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    output = _country_endpoints(rebuilt).groupby("interval_end", as_index=False).agg(
        output_area_ha=("output_area_ha", "sum"),
        output_MgCO2e=("output_MgCO2e", "sum"),
    )
    master = corrected.groupby("interval_end", as_index=False).agg(
        master_area_ha=("org_soil__area_ha", "sum"),
        master_MgCO2e=("org_soil_emis__all_gases__MgCO2e_yr", "sum"),
    )
    result = master.merge(
        output, on="interval_end", how="outer", indicator=True, validate="one_to_one"
    )
    if not result["_merge"].eq("both").all():
        raise ValueError("Global endpoint coverage mismatch")
    for measure in ("area_ha", "MgCO2e"):
        difference = (result[f"output_{measure}"] - result[f"master_{measure}"]).abs()
        result[f"{measure}_absolute_difference"] = difference
        result[f"{measure}_relative_difference"] = difference / np.maximum(
            result[f"master_{measure}"].abs(), 1.0
        )
    max_relative = float(
        result[["area_ha_relative_difference", "MgCO2e_relative_difference"]]
        .to_numpy(dtype=np.float64)
        .max(initial=0.0)
    )
    if max_relative > 1e-7:
        raise ValueError(f"Global residual exceeds 1e-7: {max_relative}")
    return result, {
        "max_relative_difference": max_relative,
        "gate_max_relative": 1e-7,
        "status": "PASS",
    }


def identity_checks(
    rebuilt: pa.Table, sentinel: np.ndarray
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    requested = set(TARGET_COLUMNS) | {"LULUCF_gross_removals__MgCO2_yr"}
    values = {
        column: _float32(rebuilt, column)[sentinel].astype(np.float64)
        for column in requested
    }
    identities = {
        "organic_total_equals_categories": (
            values["org_soil_emis__all_gases__MgCO2e_yr"],
            values["org_soil_drained_only__all_gases__MgCO2e_yr"]
            + values["org_soil_burned_only__all_gases__MgCO2e_yr"]
            + values["org_soil_drained_burned__all_gases__MgCO2e_yr"],
        ),
        "soil_total_equals_organic_total": (
            values["soil_emis__all_gases__MgCO2e_yr"],
            values["org_soil_emis__all_gases__MgCO2e_yr"],
        ),
        "gross_co2_equals_organic_components": (
            values["LULUCF_gross_emissions__CO2__MgCO2_yr"],
            values["org_soil_drained_only__CO2__MgCO2_yr"]
            + values["org_soil_burned_only__CO2__MgCO2_yr"]
            + values["org_soil_drained_burned__CO2__MgCO2_yr"],
        ),
        "gross_ch4_equals_organic_components": (
            values["LULUCF_gross_emissions__CH4__MgCO2e_yr"],
            values["org_soil_drained_only__CH4__MgCO2e_yr"]
            + values["org_soil_burned_only__CH4__MgCO2e_yr"]
            + values["org_soil_drained_burned__CH4__MgCO2e_yr"],
        ),
        "gross_n2o_equals_organic_components": (
            values["LULUCF_gross_emissions__N2O__MgCO2e_yr"],
            values["org_soil_drained_only__N2O__MgCO2e_yr"]
            + values["org_soil_drained_burned__N2O__MgCO2e_yr"],
        ),
        "non_co2_equals_ch4_plus_n2o": (
            values["LULUCF_gross_emissions__non_CO2__MgCO2e_yr"],
            values["LULUCF_gross_emissions__CH4__MgCO2e_yr"]
            + values["LULUCF_gross_emissions__N2O__MgCO2e_yr"],
        ),
        "gross_all_equals_co2_plus_non_co2": (
            values["LULUCF_gross_emissions__all_gases__MgCO2e_yr"],
            values["LULUCF_gross_emissions__CO2__MgCO2_yr"]
            + values["LULUCF_gross_emissions__non_CO2__MgCO2e_yr"],
        ),
        "net_equals_gross_plus_negative_removals": (
            values["LULUCF_net_flux__MgCO2e_yr"],
            values["LULUCF_gross_emissions__all_gases__MgCO2e_yr"]
            + values["LULUCF_gross_removals__MgCO2_yr"],
        ),
    }
    rows = []
    for name, (left, right) in identities.items():
        difference = np.abs(left - right)
        relative = difference / np.maximum(np.abs(left), 1.0)
        rows.append(
            {
                "identity": name,
                "rows": len(left),
                "max_absolute_residual": float(difference.max(initial=0.0)),
                "max_relative_residual": float(relative.max(initial=0.0)),
            }
        )
    result = pd.DataFrame(rows)
    max_relative = float(result["max_relative_residual"].max())
    if max_relative > 2e-7:
        raise ValueError(f"Identity residual exceeds 2e-7: {max_relative}")
    return result, {
        "identities_checked": len(identities),
        "max_relative_residual": max_relative,
        "gate_max_relative": 2e-7,
        "status": "PASS",
    }


def invariance_check(rebuilt: pa.Table, sentinel: np.ndarray) -> Dict[str, object]:
    dimensions = _dimensions(rebuilt)
    frame = dimensions.loc[sentinel, ["adm0", "year", "climate_domain"]].copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["period"] = np.where(frame["year"] <= 2020, "2016_2020", "2021_2024")
    for column in ORGANIC_COLUMNS:
        frame[column] = _float32(rebuilt, column)[sentinel].view(np.uint32)
    grouped = frame.groupby(["adm0", "climate_domain", "period"], observed=True)
    violations = sum(
        int((grouped[column].nunique(dropna=False) > 1).sum())
        for column in ORGANIC_COLUMNS
    )
    if violations:
        raise ValueError(f"Within-period invariance violations: {violations}")
    return {
        "organic_columns_checked": len(ORGANIC_COLUMNS),
        "within_period_invariance_violations": violations,
        "status": "PASS",
    }


def _record(path: Path) -> Dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_reports(
    report_dir: Path,
    report: Mapping[str, object],
    old: pd.DataFrame,
    engineer: pd.DataFrame,
    changes: pd.DataFrame,
    global_table: pd.DataFrame,
    identities: pd.DataFrame,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "old_master_reproduction_summary.csv": old,
        "engineer_country_endpoint_comparison.csv": engineer,
        "target_column_change_summary.csv": changes,
        "global_endpoint_summary.csv": global_table,
        "dependent_identity_summary.csv": identities,
    }
    for name, frame in frames.items():
        path = report_dir / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite report: {path}")
        frame.to_csv(path, index=False)
    report_path = report_dir / "lulucf_wide_wdpa_rebuild_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, object]:
    baseline_path = Path(args.baseline_wide).resolve()
    corrected_path = Path(args.corrected_master).resolve()
    old_path = Path(args.old_master).resolve()
    engineer_path = Path(args.engineer_reference).resolve()
    output_path = Path(args.output).resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else output_path.parent

    for path in (baseline_path, corrected_path, old_path, engineer_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path == baseline_path:
        raise ValueError("Output must differ from the immutable baseline")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_path}")
    for name in REPORT_NAMES:
        if (report_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite report: {report_dir / name}")

    baseline = pq.read_table(baseline_path)
    sentinel, baseline_summary = validate_baseline(baseline)
    old_aggregation = aggregate_master(old_path, include_nonpeat_area=True)
    old_table, old_summary = reverse_reproduce_old(baseline, old_aggregation)
    corrected = aggregate_master(corrected_path, include_nonpeat_area=False)
    rebuilt, key_summary = rebuild_table(baseline, sentinel, corrected)
    changes, preservation = preservation_checks(baseline, rebuilt, sentinel)
    engineer_table, engineer_summary = engineer_checks(rebuilt, engineer_path)
    global_table, global_summary = global_checks(rebuilt, corrected)
    identity_table, identity_summary = identity_checks(rebuilt, sentinel)
    invariance = invariance_check(rebuilt, sentinel)

    report: Dict[str, object] = {
        "status": "PASS",
        "mode": "execute" if args.execute else "dry-run",
        "command": [sys.executable, *sys.argv],
        "inputs": {
            "baseline_wide": _record(baseline_path),
            "corrected_master": _record(corrected_path),
            "old_master": _record(old_path),
            "engineer_reference": _record(engineer_path),
        },
        "baseline": baseline_summary,
        "old_master_reproduction": old_summary,
        "sentinel_key_reconciliation": key_summary,
        "preservation": preservation,
        "engineer_comparison": engineer_summary,
        "global_endpoint_reconciliation": global_summary,
        "dependent_identities": identity_summary,
        "interval_invariance": invariance,
        "area_definition": "SUM(area__ha) WHERE drainage_class <> 'non_peat'",
        "target_columns": list(TARGET_COLUMNS),
        "output": {"path": str(output_path)},
    }
    if not args.execute:
        print(json.dumps(report, indent=2))
        return report

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        pq.write_table(
            rebuilt,
            temporary,
            compression="snappy",
            use_dictionary=True,
            row_group_size=rebuilt.num_rows,
        )
        written_file = pq.ParquetFile(temporary)
        try:
            written = written_file.read()
            row_groups = written_file.metadata.num_row_groups
        finally:
            # ParquetFile retains an OS-level file handle.  Windows will not
            # permit the atomic rename below until that handle is released.
            written_file.close()
        if not rebuilt.schema.equals(written.schema, check_metadata=True):
            raise ValueError("Round-trip schema or metadata mismatch")
        if not rebuilt.equals(written, check_metadata=True):
            raise ValueError("Round-trip value mismatch")
        roundtrip = {
            "rows": written.num_rows,
            "columns": written.num_columns,
            "row_groups": row_groups,
            "schema_metadata_preserved": True,
            "all_values_roundtrip_equal": True,
            "status": "PASS",
        }
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    report["roundtrip"] = roundtrip
    report["output"] = _record(output_path)
    _write_reports(
        report_dir,
        report,
        old_table,
        engineer_table,
        changes,
        global_table,
        identity_table,
    )
    print(json.dumps(report, indent=2))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild organic-soil fields in a combined LULUCF wide parquet"
    )
    parser.add_argument("--baseline-wide", required=True)
    parser.add_argument("--corrected-master", required=True)
    parser.add_argument("--old-master", required=True)
    parser.add_argument("--engineer-reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write only after all in-memory QA gates pass",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
