import xarray as xr
import numpy as np

paths = [
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_5_0/zarr/20250724/2021_2024/drained_total_Mg_CO2e_pixel_yr_2021_2024.zarr",
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_5_0/zarr/20250724/2021_2024/burned_total_Mg_CO2e_pixel_2021_2024.zarr",
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_5_0/zarr/20250724/2021_2024/drained_state_node_2021_2024.zarr",
    "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_5_0/zarr/20250724/2021_2024/burned_state_node_2021_2024.zarr",
    "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/GADM4_1_adm0_global/20250604/global_GADM41_adm0_20250604.zarr",
    "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/contextual_layer_global_zarr/pixel_area/20250730/global_pixel_area_20250730.zarr",
]

bbox = [110, -10, 120, 0]

def open_zarr_crop(path, bbox):
    ds = xr.open_zarr(path)
    west, south, east, north = bbox
    cropped = ds.sel(x=slice(west, east), y=slice(south, north))
    # Select the main data variable explicitly
    if isinstance(cropped, xr.Dataset):
        for var in cropped.data_vars:
            if "x" in cropped[var].dims and "y" in cropped[var].dims:
                return cropped[var]
    return cropped

for path in paths:
    da = open_zarr_crop(path, bbox)
    print(f"\nChecking: {path}")
    mean_val = da.mean().values
    print(f"Mean: {mean_val}")
    data = da.values
    if np.issubdtype(data.dtype, np.number):
        valid_values = data[~np.isnan(data)]
        unique_vals = np.unique(valid_values)
        print(f"Unique values (up to 10): {unique_vals[:10]}")
    else:
        print(f"Non-numeric data type detected: {data.dtype}")



"""
python -m src.scripts.zonal_statistics.test_zarr
"""
