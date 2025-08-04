import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

zarr_path = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_5_0/zarr/20250724/2021_2024/drained_total_Mg_CO2e_pixel_yr_2021_2024.zarr"
ds = xr.open_zarr(zarr_path, consolidated=True)

# Explicitly define your bbox
west, south, east, north = 110, -10, 120, 0

# Ensure correct ordering based on coordinate descending order:
x_slice = slice(west, east) if ds.x[0] < ds.x[-1] else slice(east, west)
y_slice = slice(south, north) if ds.y[0] < ds.y[-1] else slice(north, south)

ds_crop = ds['band_data'].sel(x=x_slice, y=y_slice)

print(ds_crop)
print("Mean value:", ds_crop.mean().compute().values)

# Quickly visualize:
ds_crop.plot(robust=True, cmap='viridis')
plt.title("Cropped Data Visualization")
plt.show()
