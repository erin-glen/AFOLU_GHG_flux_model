# Organic Soil Threshold Registry

`organic_soil_threshold_registry.csv` records the selected organic soil thresholds and validation scores by data version. These values are version-specific because the probability rasters, validation extracts, and area curves can change with each OpenGeoHub delivery.

Each organic soil version should include:

- Global thresholds for F1 and F2 maximization.
- Per-biome thresholds for boreal, temperate, and tropical biomes for F1 and F2 maximization.
- The selected threshold, snapped threshold used for mapped-area lookup, mapped area in million hectares, and validation confusion-matrix metrics.
- Source paths for the threshold metrics and area summary artifacts used to populate the row.

The current registry contains:

- `20260508`: current OpenGeoHub organic soil probability outputs from `C:/tmp/afolu/uncertainty/ogh_probability/20260508`.
- `20251105_legacy`: most recent comparable legacy outputs from `C:/tmp/afolu/analysis/legacy/uncertainty`.

For a new delivery, rerun the uncertainty threshold workflow, then append one row per `(threshold_scope, biome, optimization_metric)` combination. Use `threshold_scale=0_to_1` for probability thresholds and keep `organic_soil_version` aligned with the delivery date when possible.
