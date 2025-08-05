import pandas as pd

PATH = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_5_0/zonal_stats/zonal_stats_2024/drained/2024/"  # a *directory* not a file!

# ➜ only the columns we plan to inspect
cols = ["gadm_adm0", "drained_state_nodes", "flux_type", "value"]

# ➜ only one partition (interval) and one flux type
filters = [
    ("interval_end", "==", 2024),
    ("flux_type", "==", "drained_total_Mg_CO2e_pixel_yr"),
]

df = pd.read_parquet(PATH, columns=cols, filters=filters, engine="pyarrow")
print(len(df), "rows loaded")
