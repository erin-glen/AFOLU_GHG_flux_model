# -*- coding: utf-8 -*-
"""
Master zonal-stats export.

Combine the per-inventory-period ``combined_state`` zonal-stats parquets for a
run into a single master table at FULL disaggregation -- every contextual
overlay retained -- and write it to S3 (or a chosen directory) as parquet
and/or CSV.

Grain (one row per unique combination):
    inventory period x country x climate x land_use x drainage_class
    x drained_state x burned_state x <every contextual overlay column>

Each row carries area + all flux measures as columns (``flux_type`` pivoted).
The export reads the per-interval ``combined_state`` parquets directly (the
authoritative source that matches the published figures); it does NOT use the
``all_inventory_periods`` aggregation.

Decoding (land_use, climate_domain, drainage_class, country/iso3) reuses the
same helpers as pub_assets/pub_nghgi, so the master cannot drift from the
figure/NGHGI outputs.

Example
-------
    python -m src.scripts.zonal_statistics.pub_scripts.pub_master \\
        --model_version 1_0_1 \\
        --run_name ogh_mixed_f1_f15_f2_20260513 \\
        --run_date 20260525 \\
        --years 2005 2010 2015 2020 2024
"""
from __future__ import annotations

import argparse
import posixpath
from typing import List, Optional, Sequence

import duckdb

import src.scripts.zonal_statistics.pub_scripts.pub_assets as pa
import src.scripts.zonal_statistics.pub_scripts.pub_common as pc
from src.scripts.zonal_statistics.run_zonal_stats import ROOT, build_interval_pairs

# Structural (non-measure, non-contextual) columns of combined_state. Anything
# else that is not flux_type/value is treated as a contextual overlay grouping
# column and carried through automatically.
_STRUCTURAL_COLS = {
    "flux_type",
    "value",
    "interval_end",
    "interval_start",
    "inventory_period",
    "gadm_adm0",
    "combined_state_nodes",
    "drained_state_nodes",
    "burned_state_nodes",
    "drained_state_meaning",
    "burned_state_meaning",
}

# Preferred output order for the pivoted flux measures; only those actually
# present in the source are emitted.
_MEASURE_ORDER = [
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
]


def _run_dir(model_version: str, run_name: str, run_date: str) -> str:
    return posixpath.join(
        ROOT, f"version_{model_version}", "zonal_stats", run_name, run_date
    )


def _source_globs(model_version: str, run_name: str, run_date: str, years: Sequence[int]) -> List[str]:
    rd = _run_dir(model_version, run_name, run_date)
    pairs = build_interval_pairs([int(y) for y in years])
    return [f"{rd}/{s}_{e}/combined_state/*.parquet" for (s, e) in pairs]


def _period_case(years: Sequence[int], col: str, value: str) -> str:
    """Build a CASE mapping interval_end -> inventory_period label or interval_start."""
    pairs = build_interval_pairs([int(y) for y in years])
    whens = []
    for s, e in pairs:
        lit = f"'{s}_{e}'" if value == "label" else str(s)
        whens.append(f"WHEN {e} THEN {lit}")
    return f"CASE {col} " + " ".join(whens) + " END"


def build_master_sql(
    src_list_sql: str,
    contextual_cols: Sequence[str],
    measures: Sequence[str],
    years: Sequence[int],
    have_lookup: bool,
) -> str:
    period_case = _period_case(years, "z.interval_end", "label")
    start_case = _period_case(years, "z.interval_end", "start")
    lu_case = pc.landuse_case_sql(
        "ctx.combined_state",
        "ctx.emissions_state",
        "lower(COALESCE(ctx.drained_state, z.drained_state_meaning, '')) LIKE 'peat_undrained%'",
    )
    rpad_nodes = pc._rpad_sql("z.drained_state_nodes")
    select_country = "l.country, l.iso3," if have_lookup else ""
    join_country = "LEFT JOIN adm0_lookup l ON l.gadm_adm0 = z.gadm_adm0" if have_lookup else ""
    ctx_select = "".join(f"        z.{c},\n" for c in contextual_cols)
    measure_cols = ",\n        ".join(
        f"SUM(CASE WHEN z.flux_type='{m}' THEN z.value END) AS {m}" for m in measures
    )
    return f"""
      SELECT
        {period_case} AS inventory_period,
        {start_case}  AS interval_start,
        z.interval_end,
        z.gadm_adm0, {select_country}
        COALESCE(NULLIF(ctx.climate_domain, ''), 'Unspecified') AS climate_domain,
        {lu_case} AS land_use,
        ctx.drained_state AS drainage_class,
        z.drained_state_meaning,
        z.burned_state_meaning,
        z.combined_state_nodes,
        z.drained_state_nodes,
        z.burned_state_nodes,
{ctx_select}        {measure_cols}
      FROM read_parquet({src_list_sql}) z
      LEFT JOIN drained_state_ctx AS ctx
        ON (z.drained_state_meaning = ctx.meaning)
        OR ({rpad_nodes} = ctx.key)
      {join_country}
      GROUP BY ALL
    """


def main(argv=None):
    p = argparse.ArgumentParser("Master full-disaggregation zonal-stats export")
    p.add_argument("--model_version", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--run_date", required=True)
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--aws_region", default="us-east-1")
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (S3 or local). Default: the run's zonal_stats "
        "run/date directory, beside the per-interval parquets.",
    )
    p.add_argument(
        "--basename",
        default="master_zonal_full_disaggregation",
        help="Output file basename (without extension).",
    )
    p.add_argument(
        "--formats",
        nargs="+",
        default=["parquet", "csv"],
        choices=["parquet", "csv"],
        help="Output format(s) to write.",
    )
    p.add_argument("--adm0_lookup", default=None, help="Optional CSV with gadm_adm0,country,iso3")
    args = p.parse_args(argv)

    out_dir = args.out_dir or _run_dir(args.model_version, args.run_name, args.run_date)
    globs = _source_globs(args.model_version, args.run_name, args.run_date, args.years)
    src_list_sql = "[" + ", ".join(f"'{g}'" for g in globs) + "]"

    con = duckdb.connect()
    pa._ensure_httpfs(con, args.aws_region)
    pa._register_state_context_views(con)
    have_lookup = pa._ensure_adm0_lookup(con, args.adm0_lookup)
    print(f"adm0 lookup available: {have_lookup}")

    if pa._count_globs(con, globs) == 0:
        raise FileNotFoundError(
            f"No combined_state parquet files found for {args.run_name} "
            f"{args.run_date} under {out_dir}"
        )

    # Discover columns: contextual overlays (everything that is not a structural
    # column or a measure) and which measures are actually present.
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({src_list_sql})"
    ).df()
    src_cols = list(schema["column_name"])
    contextual_cols = [c for c in src_cols if c not in _STRUCTURAL_COLS]
    present_flux = {
        r[0] for r in con.execute(
            f"SELECT DISTINCT flux_type FROM read_parquet({src_list_sql})"
        ).fetchall()
    }
    measures = [m for m in _MEASURE_ORDER if m in present_flux]
    extra = sorted(present_flux - set(_MEASURE_ORDER) - {"area__ha"})
    measures += [m for m in extra if m not in measures]
    print(f"contextual overlay columns: {contextual_cols}")
    print(f"flux measures pivoted     : {measures}")

    select_sql = build_master_sql(src_list_sql, contextual_cols, measures, args.years, have_lookup)

    for fmt in args.formats:
        ext = "parquet" if fmt == "parquet" else "csv"
        out = f"{out_dir.rstrip('/')}/{args.basename}.{ext}"
        copy_opts = "(FORMAT PARQUET)" if fmt == "parquet" else "(FORMAT CSV, HEADER)"
        print(f"[{fmt}] writing -> {out}")
        con.execute(f"COPY ({select_sql}) TO '{out}' {copy_opts};")

    # Reconciliation
    out_any = f"{out_dir.rstrip('/')}/{args.basename}.{args.formats[0]}"
    reader = "read_parquet" if args.formats[0] == "parquet" else "read_csv_auto"
    n = con.execute(f"SELECT COUNT(*) FROM {reader}('{out_any}')").fetchone()[0]
    print(f"\nrows: {n:,}")
    df = con.execute(
        f"SELECT interval_end, "
        f"SUM(drained_total_Mg_CO2e)/1e9 AS drained_GtCO2e, "
        f"SUM(burned_total_Mg_CO2e)/1e9 AS burned_GtCO2e "
        f"FROM {reader}('{out_any}') GROUP BY 1 ORDER BY 1"
    ).df()
    print(df.to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
