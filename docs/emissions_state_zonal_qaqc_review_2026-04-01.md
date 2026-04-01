# Emissions-state integration review (2026-04-01)

## Scope reviewed
Workflow path requested:

1. `src/scripts/core_model/0_drainage_emissions_model.py`
2. `src/scripts/core_model/02_aggregate_soils_outputs.py`
3. `src/scripts/zonal_statistics/01_build_zarr_caches.py`
4. `src/scripts/zonal_statistics/02_run_zonal_stats.py`
5. `src/scripts/zonal_statistics/pub_scripts/pub_assets.py`

## What is already integrated

### Core model and aggregation are dual-writing `emissions_state`
- The drainage model now packs a unified state raster into `outputs["emissions_state"]`, derived from drained + burned state rasters.
- Aggregation includes `emissions_state` in its output type list even if constants lag.

**Conclusion:** upstream production of unified state looks wired in.

### Zonal runner can decode unified state when legacy state rasters are absent
- `02_run_zonal_stats.py` includes `emissions_state_nodes` in `DATASETS`.
- If `drained_state` and/or `burned_state` are missing from mega-zarr, it falls back to decoding from `emissions_state` using `unpack_emissions_state_to_legacy`.

**Conclusion:** the reducer path already supports compatibility-mode QA/QC by deriving legacy branches from unified state.

## Main integration gap for QA/QC

### Publication outputs are still branch-centric
`pub_assets.py` still builds publication inputs as two branches:
- drained (`zs_drained`) with `drained_state_*`
- burned (`zs_burned`) with `burned_state_*`

State lookup views also remain split (`drained_state_ctx`, `burned_state_ctx`), and exported state tables are still:
- `by_drained_state_period.csv`
- `by_burned_state_period.csv`
- country variants

There is no first-class `by_emissions_state_*` output yet.

**Impact:** you can *run* zonal stats from `emissions_state`, but downstream QA/QC still lacks a canonical unified-state artifact to validate transition quality directly.

## Recommended path forward

### Phase 1 (safe, immediate): keep compatibility outputs, add explicit unified QA artifacts
1. **Naming bridge:** keep wire format `emissions_state` but introduce alias language `combined_state` in docs/logs/CLI help so users can discover intent without breaking existing runs.
2. **Zonal outputs:** add optional emissions-state parquet outputs:
   - `by_adm0__emissions_state__flux_type__interval_end`
   - `by_adm0__emissions_state__area__interval_end` (if you keep area separate)
3. **Publication exports:** add new CSVs:
   - `by_emissions_state_period.csv`
   - `by_country_emissions_state_period.csv`
4. **Maintain legacy drained/burned exports** for one release window to preserve parity checks.

### Phase 2 (Q/A hardening): enforce decode parity and mapping completeness
1. **Decode parity check in zonal run:** when both direct legacy rasters and unified raster are present, sample-check or full-check that:
   - `decode(emissions_state).drained == drained_state`
   - `decode(emissions_state).burned == burned_state`
   (with explicit mismatch counters and fail threshold)
2. **Coverage checks:** fail publication if unified-state keys are unmapped to semantic labels.
3. **Regression artifact:** emit compact QA report per interval (JSON/CSV):
   - unique code counts
   - unmapped code list
   - decode mismatch totals
   - branch totals parity deltas (drained/burned)

### Phase 3 (migration): promote unified outputs to canonical
1. Switch dashboard/consumer defaults to `by_emissions_state_*`.
2. Keep drained/burned exports as derived compatibility views.
3. Deprecate legacy branch-native state dependencies after one stable release cycle.

## Concrete implementation sequence

1. **Extend `pub_common.py` SQL builders** with `table_by_emissions_state_sql()` and country variant.
2. **In `pub_assets.py`**, register a new `emissions_state_ctx` lookup sourced from `zonal_constants` unified mapping (or deterministic decode + join strategy).
3. **In `pub_assets.py` writer block**, emit the two new `by_emissions_state_*` CSVs in parallel with existing exports.
4. **In `02_run_zonal_stats.py`**, optionally materialize emissions-state zonal parquet directly (feature flag) so publication does not depend only on decoded branches.
5. Add a small **smoke QA command set** in docs/README for a single interval (e.g., `2024`) to compare:
   - branch totals from direct legacy
   - branch totals from decoded unified
   - unified-state table row-count + key coverage

## Suggested acceptance criteria

- For a test interval, drained and burned total emissions match pre-change baseline within tolerance.
- Zero unexpected unmapped unified-state codes in publication tables.
- If legacy rasters exist, decode mismatch count is zero (or below agreed threshold, with explicit waiver).
- `by_emissions_state_period.csv` and country variant are generated in `pub_assets` output.

