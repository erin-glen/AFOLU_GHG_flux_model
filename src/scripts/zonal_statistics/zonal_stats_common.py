# -*- coding: utf-8 -*-
"""Shared helpers for zonal-statistics scripts."""

from __future__ import annotations

import posixpath
from typing import List, Tuple

from src.scripts.utilities import constants_and_names as cn

ROOT = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs"


def build_output_parquet(model_version: str, run_name: str, run_date: str, interval: str) -> str:
    """Return the S3 prefix for zonal statistics output."""
    return (
        posixpath.join(
            ROOT,
            f"version_{model_version}",
            "zonal_stats",
            run_name,
            run_date,
            interval,
        ).rstrip("/")
        + "/"
    )


def build_interval_pairs(end_years: List[int]) -> List[Tuple[int, int]]:
    """Map inventory interval end years to (start, end) pairs."""
    mapping = {end: (start, end) for start, end in cn.five_year_inventory_periods}
    pairs = []
    for year in end_years:
        if year not in mapping:
            raise ValueError(
                f"Interval end year {year} not supported. Valid options: {sorted(mapping)}"
            )
        pairs.append(mapping[year])
    return pairs
