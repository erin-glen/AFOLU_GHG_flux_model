#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_zarr_variable.py  –  inspect / confirm which variable a workflow would
                           select from one or more Zarr stores.

For every input Zarr:
  • lists all variables with their shapes, dtypes and dimensions
  • shows the variable that *would* be selected automatically
  • lets you override the choice with --var NAME

No project‑specific imports or dependencies beyond Xarray.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import textwrap

import xarray as xr


def choose_var(ds: xr.Dataset | xr.DataArray,
               explicit: str | None,
               label: str) -> xr.DataArray:
    """
    Return the variable that would be used from *ds*.

    • If *explicit* is given, ensure it exists and return it.
    • Otherwise return the first variable that carries both x and y dims.
    """
    if isinstance(ds, xr.DataArray):  # single‑variable Zarr
        return ds

    if explicit:
        if explicit not in ds:
            raise KeyError(
                f"{label}: variable '{explicit}' not found.  "
                f"Available: {list(ds.data_vars)}"
            )
        return ds[explicit]

    for v in ds.data_vars.values():
        if {"x", "y"}.issubset(v.dims):
            return v

    raise ValueError(f"{label}: no variable with x & y dimensions found.")


def describe_store(uri: str, explicit_var: str | None) -> None:
    label = Path(uri).name
    print(f"\n┏━ {uri}")
    ds = xr.open_zarr(uri, consolidated=True)

    # Print catalogue of variables
    if isinstance(ds, xr.DataArray):
        cat = {ds.name: ds}
    else:
        cat = ds.data_vars

    print("┃ available variables:")
    for name, var in cat.items():
        dims = "×".join(f"{d}:{var.sizes[d]}" for d in var.dims)
        print(f"┃   • {name:<25}  shape=({dims})  dtype={str(var.dtype)}")

    # Decide which variable would be picked
    picked = choose_var(ds, explicit_var, label)
    print("┗▶ selected variable:", picked.name,
          f"(dims={picked.dims}, dtype={picked.dtype})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        Inspect one or more Zarr stores and report which variable a workflow
        would pick by default.

        Selection rule (unless --var is supplied):
          first variable that contains BOTH 'x' and 'y' dimensions.

        Exit status is 0 if every store has a clear selection, otherwise 1.
        """)
    )
    p.add_argument("zarr", nargs="+", help="Path or s3:// URI of a Zarr store")
    p.add_argument("--var", help="Variable name to select (applied to all stores)")
    args = p.parse_args(argv)

    okay = True
    for uri in args.zarr:
        try:
            describe_store(uri, args.var)
        except (KeyError, ValueError) as e:
            print("✖", e, file=sys.stderr)
            okay = False

    sys.exit(0 if okay else 1)


if __name__ == "__main__":
    main()


"""
# ① Let the script decide which variable would be chosen
python -m src.scripts.zonal_statistics.qaqc \
       s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_5_0/zarr/20250724/2021_2024/drained_total_Mg_CO2e_pixel_yr_2021_2024.zarr \
       s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_5_0/zarr/20250724/2021_2024/drained_state_node_2021_2024.zarr

# ② Force a specific variable name for *all* stores
python -m src.scripts.zonal_statistics.qaqc \
       --var drained_total_Mg_CO2e_pixel_yr \
       s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_5_0/zarr/20250724/2021_2024/drained_total_Mg_CO2e_pixel_yr_2021_2024.zarr

"""