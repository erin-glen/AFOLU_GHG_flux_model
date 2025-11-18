"""
python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster \
  --full_model \
  --chunk_size 1 \
  --start_year 2021 \
  --end_year 2024 \
  --count_burned_years \
  --interval_type five_year \
  --peat_dataset gfw \
  --run_name gfw_standard_model_500m_23

python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster \
  --full_model \
  --chunk_size 1 \
  --start_year 2021 \
  --end_year 2024 \
  --count_burned_years \
  --interval_type five_year \
  --peat_dataset gfw \
  --run_name gpd_standard_model_500m_23

python -m src.scripts.core_model.2_per_pixel_soils_outputs \
  --cluster_name drainage_cluster \
  --chunk_size 1 \
  --run_name ogh_sensitivity_1km\
  --output_date 20251116

"""