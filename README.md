# AFOLU GHG Flux Model

This repository contains the organic-soils AFOLU model and downstream post-processing pipelines.

## Current production workflow (500 m core model → 10x10 zonal stats)

The current workflow is:

1. **Run the core model** (`0_drainage_emissions_model`) on model chunks.
2. **Aggregate chunk outputs to 10x10 tiles** (`3_aggregate_soils_outputs`) from the model mega-zarr.
3. **Build zonal-stat Zarr caches** from the 10x10 outputs (`zonal_statistics/01_build_zarr_caches`).
4. **Run zonal stats** (`zonal_statistics/02_run_zonal_stats`) using those caches.

### Is `2_per_pixel_soils_outputs.py` still required?

Usually **no** for the current zonal-statistics workflow.

- The cache builder (`01_build_zarr_caches`) reads 10x10 tile outputs (for example folders like `drained_co2_Mg_CO2_pixel_yr/.../40000_pixels/...`), which are produced by the **10x10 aggregation step**.
- The 10x10 aggregation step (`3_aggregate_soils_outputs`) already derives per-pixel flux layers for numeric outputs while writing 10x10 tiles.
- Therefore, running `2_per_pixel_soils_outputs.py` is generally a **legacy / optional** step and is not part of the default pipeline here.

Use step 2 only if you explicitly need standalone per-pixel outputs at the original chunk resolution for a separate analysis.
