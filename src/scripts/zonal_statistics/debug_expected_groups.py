#!/usr/bin/env python
import xarray as xr, dask.array as da, numpy as np, pandas as pd
from flox.xarray import xarray_reduce
import fsspec, argparse

def observed(arr):
    return np.unique(da.unique(arr.data)).compute()

def open_arr(uri, chunk=4000):
    return xr.open_zarr(fsspec.get_mapper(uri, anon=False),
                        consolidated=True).squeeze()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total_zarr")
    p.add_argument("--node_zarr")
    p.add_argument("--adm0_zarr")
    p.add_argument("--bbox", nargs=4, type=float, help="W S E N degrees")
    a = p.parse_args()

    total = open_arr(a.total_zarr)
    nodes = open_arr(a.node_zarr).astype("u4")
    adm0  = open_arr(a.adm0_zarr).astype("u4")

    if a.bbox:
        w,s,e,n = a.bbox
        total = total.sel(x=slice(w,e), y=slice(n,s))  # y desc
        nodes = nodes.sel(x=slice(w,e), y=slice(n,s))
        adm0  =  adm0.sel(x=slice(w,e), y=slice(n,s))

    # Run with global groups
    res_all = xarray_reduce(total, adm0, nodes,
                            func="sum",
                            expected_groups=(
                                np.arange(900,dtype="u4"),  # fake 0–899
                                np.arange(0,44000001,1000000,dtype="u4")
                            )).compute()

    # Run with observed only
    res_obs = xarray_reduce(total, adm0, nodes,
                            func="sum",
                            expected_groups=(
                                observed(adm0), observed(nodes)
                            )).compute()

    print("┌─ With global expected_groups:", res_all.data.size,
          "rows; non‑zero", (res_all.data>0).sum())
    print("└─ With observed groups:       ", res_obs.data.size,
          "rows; non‑zero", (res_obs.data>0).sum())

if __name__ == "__main__":
    main()

"""
python -m src.scripts.zonal_statistics.debug_expected_groups \
  --total_zarr s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_5_0/zarr/20250724/2021_2024/drained_total_Mg_CO2e_pixel_yr_2021_2024.zarr \
  --node_zarr  s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_5_0/zarr/20250724/2021_2024/drained_state_node_2021_2024.zarr \
  --adm0_zarr  s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/GADM4_1_adm0_global/20250604/global_GADM41_adm0_20250604.zarr \
  --bbox 110 -10 120 0
  """