# AFOLU GHG Flux Model

This repository contains the organic-soils AFOLU model and downstream post-processing pipelines.

## Current production workflow (500 m core model → 10x10 zonal stats)

The current workflow is:

1. **Run the core model** (`0_drainage_emissions_model`) on model chunks.
2. **Aggregate chunk outputs to 10x10 tiles** (`3_aggregate_soils_outputs`) from the model mega-zarr.
3. **Run zonal stats** (`zonal_statistics/02_run_zonal_stats`) directly from the model mega-zarr.

`zonal_statistics/01_build_zarr_caches` is now mainly for contextual-layer prep
(`pixel_area`, `adm0`). Its legacy model-output cache build path is optional.

### Is `2_per_pixel_soils_outputs.py` still required?

Usually **no** for the current zonal-statistics workflow.

- Zonal statistics now reads the model mega-zarr directly and performs per-hectare→per-pixel conversion at read time.
- Therefore, running `2_per_pixel_soils_outputs.py` is a **legacy / optional** step and is not part of the default pipeline here.

Use step 2 only if you explicitly need standalone per-pixel outputs at the original chunk resolution for a separate analysis.
