from src.scripts.zonal_statistics.run_zonal_statistics import open_zarr_region
import numpy as np

path = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_5_0/zarr/20250724/2021_2024/drained_total_Mg_CO2e_pixel_yr_2021_2024.zarr"
bbox = [110, -10, 120, 0]

da = open_zarr_region(path, bbox=bbox, chunk_size=4000)

print(da)
print("Mean:", da.mean().compute().values)
print("Unique values:", np.unique(da.values[~np.isnan(da.values)])[:10])
