# AFOLU GHG Flux Model

This repository contains the organic-soils AFOLU model and downstream post-processing pipelines.

## Current production workflow (500 m core model → 10x10 zonal stats)

The current workflow is:

1. **Run the core model** (`0_drainage_emissions_model`) on model chunks.
2. **Aggregate chunk outputs to 10x10 tiles** (`3_aggregate_soils_outputs`) from the model mega-zarr.
3. **Run zonal stats** (`zonal_statistics/02_run_zonal_stats`) directly from the model mega-zarr.

`zonal_statistics/01_build_zarr_caches` is now mainly for contextual-layer prep
(`pixel_area`, `adm0`, `ogh_unthresholded_probability`). Its legacy model-output
cache path only runs when flux datasets are explicitly passed via `--datasets`.
The active pixel-area source is the corrected global zarr at
`s3://gfw2-data/climate/AFOLU_flux_model/contextual_layer_global_zarr/pixel_area/20260531_fillValue_removed/global_pixel_area_20260531.zarr`;
do not rebuild pixel area from the older data-lake GeoTIFF source.

Default (contextual-only) run example:

```bash
python -m src.scripts.zonal_statistics.01_build_zarr_caches \
  --cluster_name zarr_build \
  --chunk_size 8000
```

Legacy flux-cache example (explicit):

```bash
python -m src.scripts.zonal_statistics.01_build_zarr_caches \
  --cluster_name zarr_build \
  --include_legacy_model_output_caches \
  --model_version 0_9_7 \
  --run_date 20251118 \
  --interval_end_years 2024 \
  --run_name ogh_standard_model \
  --datasets drained_total burned_total
```

### Is `2_per_pixel_soils_outputs.py` still required?

Usually **no** for the current zonal-statistics workflow.

- Zonal statistics now reads the model mega-zarr directly and performs per-hectare→per-pixel conversion at read time.
- Therefore, running `2_per_pixel_soils_outputs.py` is a **legacy / optional** step and is not part of the default pipeline here.

Use step 2 only if you explicitly need standalone per-pixel outputs at the original chunk resolution for a separate analysis.
