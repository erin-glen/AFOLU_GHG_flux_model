# OGH Probability Publication Rerun Checklist

This runbook inventories the steps needed when a refreshed OpenGeoHub organic
soil probability raster arrives and the organic-soils model must be rerun for
publication outputs.

Use placeholders consistently:

- `{RUN_DATE}`: production run date tag, `YYYYMMDD`.
- `{OGH_PROB_DATE}`: processed OGH unthresholded probability tile date tag.
- `{OGH_SOURCE_TIF_URL}`: HTTPS URL for the delivered OpenGeoHub probability
  GeoTIFF/COG.
- `{MODEL_VERSION}`: underscore model version, for example `0_1_4`.
- `{THRESHOLD_DIR}`: local directory containing threshold diagnostics.
- `{BASELINE_BIOME_THRESHOLDS}`: CSV from `biome_thresholds_summary.csv` or
  equivalent JSON thresholds for the baseline/max-F1 option.
- `{SCENARIO_BOUNDS_THRESHOLDS}`: CSV from `scenario_bounds_thresholds_f1.csv`
  or `scenario_bounds_thresholds_f2.csv`, containing baseline, low-area, and
  high-area threshold columns.

## Current 2026-05 Rerun Status

- OGH unthresholded probability tiles are using date tag `20260508`.
- The broad binary OGH mask used for the refreshed union was built with
  threshold `9` for roads/canals coverage.
- The refreshed 30 m peat union mask is using date tag `20260508`.
- Roads/canals were fully rerun from the refreshed peat union mask, without
  reprojecting source road/canal vectors, and the aggregated distance products
  are using date tag `20260509`.
- `dirs["osm_roads"]`, `dirs["osm_canals"]`, and `dirs["grip"]` now point to
  the `distance/40000_pixels/20260509` folders.
- Before launching the full all-period production run and sensitivity matrix,
  run one global OGH baseline gate for `2021_2024` and carry that single run
  through aggregation, zonal statistics, publication QA/comparison scripts, and
  global map aggregation.
- The `20260510` OGH baseline gate completed through 0.01-degree global raster
  aggregation for `2021_2024`; display rendering was intentionally skipped for
  this gate.
- The gate also produced the 0.01-degree `combined_state_reclassified` raster
  with four organic-state map classes.

## Critical Findings

1. OGH thresholding is now applied in the core model, not only during tiling.
   `--peat_dataset ogh` resolves to the unthresholded OGH probability tiles and
   applies `--peat_threshold` or `--peat_threshold_by_biome` at run time.

2. Scalar `--peat_threshold` currently expects the native OGH raster scale
   (`0..100`). The biome CSV/JSON parser rescales `0..1` values automatically
   only for `--peat_threshold_by_biome`. For scalar sensitivity runs, pass
   `10` for 10 percent, not `0.10`, unless the parser is changed first.

3. Roads/canals preprocessing is still masked to organic-soil extent, via the
   30 m peat union mask. If the updated OGH probability surface materially
   expands the organic-soil extent used for analysis, rerun the peat union and
   roads/canals presence/distance products so newly organic pixels do not get
   missing/zero drainage-distance inputs.

4. `01_build_zarr_caches.py` has an `--organic_probability_date` /
   `--probability_date` flag. Pass it explicitly for every rerun so the
   contextual Zarr and processed GeoTIFF source date stay aligned.

5. Publication comparison scripts expect specific run names. Either keep those
   names, or update the hard-coded comparison definitions in
   `pub_compare_runs.py` before producing final figures.

6. Several preprocessing scripts derive output dates from `today_date` or
   script-level constants rather than CLI arguments. Before production, either
   patch these scripts to accept explicit date tags or verify the resolved S3
   input/output prefixes in a smoke run.

## Required Preflight Updates

1. Version the new OGH input paths.
   - Update raw OGH paths and processed date tags in
     `src/scripts/preprocessing/preprocessing_constants.py`.
   - Update `peat_mask_dirs["ogh_unthresholded"]` and, if creating a binary OGH
     tile for roads extent, `peat_mask_dirs["ogh"]` in
     `src/scripts/utilities/constants_and_names.py`.
   - Pass explicit date tags with `--date` and
     `--organic_probability_date` so multi-day reruns do not drift with
     `today_date`. Use `--raw_date` only when reading the configured dated S3
     source path instead of an explicit `--raw_path`.
   - If OpenGeoHub provides a public range-readable GeoTIFF/COG URL, prefer
     passing it directly with `--raw_path {OGH_SOURCE_TIF_URL}` instead of
     downloading and re-uploading the raw raster to `gfw2-data`.
   - If rerunning the peat union, pass `--dataset_date ogh={OGH_BINARY_DATE}`
     and any other refreshed input dates so it does not silently use the
     default historical inputs.
   - Check that `peat_masks.py` and `peat_mask_union.py` are not writing an
     unintended nested date prefix. Both combine configured S3 prefixes with a
     runtime date in some paths.

2. Choose canonical run names before launching the model.
   - Baseline all periods: `ogh_biome_thresholds`.
   - Inventory-source sensitivities: `gfw_standard_model_500m`,
     `gpd_standard_model_500m`.
   - Drainage-distance sensitivities, if using existing pub comparison defaults:
     `ogh_sensitivity_250m`, `ogh_sensitivity_500m`, `ogh_sensitivity_750m`.
   - Threshold plus emission-factor envelope: `ogh_sensitivity_low`,
     `ogh_sensitivity_high`.

3. Decide whether to run an OGH 500 m baseline alias.
   - `pub_compare_runs.py` currently uses `ogh_biome_thresholds` for inventory
     source comparison, but `ogh_sensitivity_500m` for distance and high/low
     comparisons. Either run a 2021-2024 baseline alias as `ogh_sensitivity_500m`
     or edit `COMPARISONS` in `pub_compare_runs.py` to use one baseline name.

## Stage 1: Process Updated OGH Probability Tiles

Primary script:

- `src/scripts/preprocessing/peat/peat_masks.py`

Command:

```bash
python -m src.scripts.preprocessing.peat.peat_masks \
  --dataset ogh_unthresholded \
  --date {OGH_PROB_DATE} \
  --raw_path {OGH_SOURCE_TIF_URL} \
  --client coiled
```

Expected output:

- 10x10 degree OGH unthresholded probability tiles under the intended
  `peat_mask/OGH/tiles_unthresholded/{OGH_PROB_DATE}/` prefix after the date
  path harmonization above.
- When the source GeoTIFF/COG covers less than the canonical global tile roster,
  `peat_masks.py` filters out non-overlapping tiles before scheduling work.
  For OGH outputs, all-zero warped tiles are skipped by default; pass
  `--upload_empty_tiles` only when a complete zero-filled tile set is required.

Optional, only if roads/canals need a refreshed broad organic-soil mask:

```bash
python -m src.scripts.preprocessing.peat.peat_masks \
  --dataset ogh \
  --date {OGH_BINARY_DATE} \
  --raw_path {OGH_SOURCE_TIF_URL} \
  --client coiled
```

Use the low-threshold/high-area envelope, or another conservative binary extent,
for this roads/canals mask so the distance products cover every pixel that may
be organic in any publication run.

## Stage 2: Rebuild Contextual Zarrs

Primary script:

- `src/scripts/zonal_statistics/01_build_zarr_caches.py`

Command:

```bash
python -m src.scripts.zonal_statistics.01_build_zarr_caches \
  --cluster_name zarr_build \
  --organic_probability_date {OGH_PROB_DATE} \
  --chunk_size 8000
```

This ensures the contextual zarrs needed by zonal statistics:

- pixel area
- ADM0
- OGH unthresholded probability
- climate domain

The probability date should be supplied explicitly with
`--organic_probability_date` even when the current default happens to match.

## Stage 3: Probability Area Stats and Area Curves

Primary scripts:

- `src/scripts/zonal_statistics/02b_run_probability_class_area_stats.py`
- `src/scripts/uncertainty/build_probability_area_curve.py`

Run probability-class area by ADM0 and biome:

```bash
python -m src.scripts.zonal_statistics.02b_run_probability_class_area_stats \
  --contextual_date 20250925 \
  --probability_date {OGH_PROB_DATE} \
  --include_biome \
  --overwrite_existing
```

Build per-biome and global area-vs-threshold curves from the single biome
class-area reduction:

```bash
python -m src.scripts.uncertainty.build_probability_area_curve \
  --probability-date {OGH_PROB_DATE} \
  --per-biome \
  --output {THRESHOLD_DIR}/area_vs_threshold_{OGH_PROB_DATE}_biome.csv
```

This also writes `{THRESHOLD_DIR}/area_vs_threshold_{OGH_PROB_DATE}.csv`
by default, derived from the same biome class-area table. Do not run a second
non-biome `02b_run_probability_class_area_stats` reduction unless doing a
deliberate QA comparison.

## Stage 4: Threshold Tuning

Primary scripts:

- `src/scripts/uncertainty/assign_biome_to_points.py`
- `src/scripts/uncertainty/fscore_threshold_curves_bounds.py`

If the validation point table does not already contain biome labels:

```bash
python -m src.scripts.uncertainty.assign_biome_to_points \
  --input {VALIDATION_POINTS_CSV} \
  --output {VALIDATION_POINTS_WITH_BIOME_CSV}
```

Run global and per-biome threshold diagnostics:

```bash
python -m src.scripts.uncertainty.fscore_threshold_curves_bounds \
  --input {VALIDATION_POINTS_WITH_BIOME_CSV} \
  --output-dir {THRESHOLD_DIR} \
  --biome-column biome \
  --report-thresholds {BASELINE_THRESHOLD_0_TO_1} \
  --area-curve-table {THRESHOLD_DIR}/area_vs_threshold_{OGH_PROB_DATE}.csv \
  --biome-area-curves {THRESHOLD_DIR}/area_vs_threshold_{OGH_PROB_DATE}_biome.csv \
  --biome-bounds-metric f1 \
  --mapped-area-unit Mha
```

The per-biome curve supplies biome-specific mapped area for threshold matching.
The global area curve is now derived from the same biome class-area output so
threshold tuning can use both products without a second raster reduction.

Outputs to retain:

- `threshold_metrics.csv`
- `selected_thresholds.csv`
- `extent_bounds_summary.csv`, if mapped area is supplied
- `biome_thresholds_summary.csv`
- per-biome `extent_bounds_summary.csv`
- per-biome `area_bound_threshold_matches.csv`

Decisions to record:

- Baseline threshold: max-F1 threshold, by biome if supported by the final
  validation summary.
- Low-area envelope: high threshold, paired with low emission factors.
- High-area envelope: low threshold, paired with high emission factors.
- Drainage-distance sensitivity values, all less than 1000 m unless the
  distance rasters and model validation are changed.

## Stage 5: Roads/Canals Decision and Optional Rerun

Current behavior:

- `src/scripts/preprocessing/roads_canals/global_datasets/roads_io.py` loads
  the 30 m peat union mask.
- `src/scripts/preprocessing/roads_canals/global_datasets/1_1_binary_roads_presence.py`
  writes presence only inside that mask.
- `src/scripts/preprocessing/roads_canals/global_datasets/1_2_distance_from_presence_mosaic.py`
  computes distance, then masks distance back to the peat union.
- The core model reads the aggregated distance rasters configured in
  `src/scripts/utilities/constants_and_names.py`.

Rerun roads/canals if either condition is true:

- the updated OGH surface adds organic area outside the current union mask, or
- any high drainage threshold is `>= 1000 m`.

If rerunning, use these scripts in order:

1. Build refreshed 30 m peat union:

```bash
python -m src.scripts.preprocessing.peat.peat_mask_union \
  --dataset_list gfw gpd peatmap peatml ogh \
  --output_date {PEAT_UNION_DATE} \
  --dataset_date ogh={OGH_BINARY_DATE} \
  --client coiled
```

2. Reproject vectors only if the source OSM/GRIP vectors changed:

```bash
python -m src.scripts.preprocessing.roads_canals.global_datasets.01_reproject_roads_canals \
  --grip --osm_roads --osm_canals \
  --client coiled
```

3. Build 30 m presence rasters for each feature type:

```bash
python -m src.scripts.preprocessing.roads_canals.global_datasets.1_1_binary_roads_presence \
  --feature_type osm_roads \
  --client coiled \
  --resolution 30m \
  --product presence \
  --chunk_size 1.0 \
  --batch_size 20
```

Repeat for `osm_canals` and `grip_roads`.

4. Build distance rasters for each feature type:

```bash
python -m src.scripts.preprocessing.roads_canals.global_datasets.1_2_distance_from_presence_mosaic \
  --feature_type osm_roads \
  --client coiled \
  --date {ROADS_PRESENCE_DATE} \
  --halo_m 1000 \
  --maxdist 1000 \
  --batch_size 20
```

Repeat for `osm_canals` and `grip_roads`.

5. Aggregate road/canal chunks to 10x10 degree rasters:

```bash
python -m src.scripts.preprocessing.roads_canals.global_datasets.03_aggregate_roads_canals \
  -cn drainage_cluster \
  --products presence distance \
  --pixel_resolution 4000_pixels \
  --date {ROADS_DISTANCE_DATE}
```

6. Update `dirs["osm_roads"]`, `dirs["osm_canals"]`, and `dirs["grip"]` in
   `src/scripts/utilities/constants_and_names.py` to point to the new aggregated
   `distance/40000_pixels/{ROADS_DISTANCE_DATE}` folders before model runs.

Optional delta workflow if only the peat union mask changed:

The core model should still use aggregated 10x10 degree distance rasters. To
avoid rerunning every roads/canals chunk, assemble a complete 4000-pixel
roads/canals folder from old unchanged chunks plus newly rerun changed chunks.

1. Identify 1x1 degree chunks where the union mask changed:

```bash
python -m src.scripts.preprocessing.roads_canals.global_datasets.04_changed_peat_chunks \
  --old_union_date 20251110 \
  --new_union_date 20260508 \
  --output logs/roads_canals_delta/20251110_to_20260508/changed_peat_chunks_20251110_to_20260508.csv \
  --distance_output logs/roads_canals_delta/20251110_to_20260508/distance_affected_peat_chunks_20251110_to_20260508.csv
```

The `changed_peat_chunks` manifest is used for presence. The
`distance_affected_peat_chunks` manifest includes changed chunks plus one ring
of neighboring chunks so distance values are refreshed where a 1 km halo can
cross chunk boundaries.

2. Copy unchanged previous presence chunks into a clean target date folder:

```bash
python -m src.scripts.preprocessing.roads_canals.global_datasets.05_assemble_delta_roads_canals \
  --changed_manifest logs/roads_canals_delta/20251110_to_20260508/changed_peat_chunks_20251110_to_20260508.csv \
  --target_date {ROADS_DELTA_DATE} \
  --products presence \
  --summary_output logs/roads_canals_delta/20251110_to_20260508/assemble_presence_{ROADS_DELTA_DATE}.json \
  --apply
```

3. Rerun presence only for changed chunks:

```bash
python -m src.scripts.preprocessing.roads_canals.global_datasets.1_1_binary_roads_presence \
  --feature_type osm_roads \
  --client coiled \
  --resolution 30m \
  --product presence \
  --chunk_manifest logs/roads_canals_delta/20251110_to_20260508/changed_peat_chunks_20251110_to_20260508.csv \
  --batch_size 20 \
  --date {ROADS_DELTA_DATE}
```

Repeat for `osm_canals` and `grip_roads`.

4. Copy unchanged previous distance chunks into the same clean target date
   folder:

```bash
python -m src.scripts.preprocessing.roads_canals.global_datasets.05_assemble_delta_roads_canals \
  --changed_manifest logs/roads_canals_delta/20251110_to_20260508/distance_affected_peat_chunks_20251110_to_20260508.csv \
  --target_date {ROADS_DELTA_DATE} \
  --products distance \
  --summary_output logs/roads_canals_delta/20251110_to_20260508/assemble_distance_{ROADS_DELTA_DATE}.json \
  --apply
```

5. Rerun distance only for changed chunks, after target-date presence is
   complete:

```bash
python -m src.scripts.preprocessing.roads_canals.global_datasets.1_2_distance_from_presence_mosaic \
  --feature_type osm_roads \
  --client coiled \
  --date {ROADS_DELTA_DATE} \
  --chunk_manifest logs/roads_canals_delta/20251110_to_20260508/distance_affected_peat_chunks_20251110_to_20260508.csv \
  --halo_m 1000 \
  --maxdist 1000 \
  --batch_size 20
```

Repeat for `osm_canals` and `grip_roads`.

6. Aggregate the complete target-date 4000-pixel folders to 10x10 degree
   rasters:

```bash
python -m src.scripts.preprocessing.roads_canals.global_datasets.03_aggregate_roads_canals \
  -cn drainage_cluster \
  --products distance \
  --pixel_resolution 4000_pixels \
  --date {ROADS_DELTA_DATE}
```

Aggregate `presence` too if QA maps or counts need the 10x10 presence rasters.

## Stage 6: Core Model Run Matrix

Primary script:

- `src/scripts/core_model/0_drainage_emissions_model.py`

Single-period global gate before the full matrix:

```bash
python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster \
  --full_model \
  --chunk_size 1 \
  --start_year 2021 \
  --end_year 2024 \
  --interval_type five_year \
  --count_burned_years \
  --peat_dataset ogh \
  --peat_threshold {BASELINE_FALLBACK_THRESHOLD_0_TO_100} \
  --peat_threshold_by_biome {BASELINE_BIOME_THRESHOLDS} \
  --fscore_metric f1 \
  --peat_threshold_scenario baseline \
  --drainage_distance_threshold_m 500 \
  --emission_factor_variant default \
  --create_zarr \
  --run_date {RUN_DATE} \
  --run_name ogh_biome_thresholds
```

For the 20260510 test gate, use `--peat_threshold 10`,
`{BASELINE_BIOME_THRESHOLDS}` =
`/mnt/c/tmp/afolu/uncertainty/ogh_probability/20260508/threshold_curves_biome_f1/biome_thresholds_summary.csv`,
`{RUN_DATE}=20260510`, and run only `2024` through Stages 7-10 before
starting the full matrix.

Baseline, all inventory periods:

```bash
python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster \
  --full_model \
  --chunk_size 1 \
  --all_five_year_periods \
  --count_burned_years \
  --peat_dataset ogh \
  --peat_threshold {BASELINE_FALLBACK_THRESHOLD_0_TO_100} \
  --peat_threshold_by_biome {BASELINE_BIOME_THRESHOLDS} \
  --fscore_metric f1 \
  --peat_threshold_scenario baseline \
  --drainage_distance_threshold_m 500 \
  --emission_factor_variant default \
  --create_zarr \
  --run_date {RUN_DATE} \
  --run_name ogh_biome_thresholds
```

Sensitivities, 2021-2024 only:

```bash
# GFW inventory source
python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster --full_model --chunk_size 1 \
  --start_year 2021 --end_year 2024 --interval_type five_year \
  --count_burned_years --peat_dataset gfw \
  --drainage_distance_threshold_m 500 \
  --emission_factor_variant default --create_zarr \
  --run_date {RUN_DATE} --run_name gfw_standard_model_500m

# GPD inventory source
python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster --full_model --chunk_size 1 \
  --start_year 2021 --end_year 2024 --interval_type five_year \
  --count_burned_years --peat_dataset gpd \
  --drainage_distance_threshold_m 500 \
  --emission_factor_variant default --create_zarr \
  --run_date {RUN_DATE} --run_name gpd_standard_model_500m
```

Use the OGH baseline threshold configuration for low/high drainage distance:

```bash
python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster --full_model --chunk_size 1 \
  --start_year 2021 --end_year 2024 --interval_type five_year \
  --count_burned_years --peat_dataset ogh \
  --peat_threshold {BASELINE_FALLBACK_THRESHOLD_0_TO_100} \
  --peat_threshold_by_biome {BASELINE_BIOME_THRESHOLDS} \
  --fscore_metric f1 \
  --peat_threshold_scenario baseline \
  --drainage_distance_threshold_m {LOW_DRAINAGE_THRESHOLD_M} \
  --emission_factor_variant default --create_zarr \
  --run_date {RUN_DATE} --run_name ogh_sensitivity_250m

python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster --full_model --chunk_size 1 \
  --start_year 2021 --end_year 2024 --interval_type five_year \
  --count_burned_years --peat_dataset ogh \
  --peat_threshold {BASELINE_FALLBACK_THRESHOLD_0_TO_100} \
  --peat_threshold_by_biome {BASELINE_BIOME_THRESHOLDS} \
  --fscore_metric f1 \
  --peat_threshold_scenario baseline \
  --drainage_distance_threshold_m {HIGH_DRAINAGE_THRESHOLD_M} \
  --emission_factor_variant default --create_zarr \
  --run_date {RUN_DATE} --run_name ogh_sensitivity_750m
```

Run a 2021-2024 OGH baseline alias only if needed for existing pub comparison
defaults:

```bash
python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster --full_model --chunk_size 1 \
  --start_year 2021 --end_year 2024 --interval_type five_year \
  --count_burned_years --peat_dataset ogh \
  --peat_threshold {BASELINE_FALLBACK_THRESHOLD_0_TO_100} \
  --peat_threshold_by_biome {BASELINE_BIOME_THRESHOLDS} \
  --fscore_metric f1 \
  --peat_threshold_scenario baseline \
  --drainage_distance_threshold_m 500 \
  --emission_factor_variant default --create_zarr \
  --run_date {RUN_DATE} --run_name ogh_sensitivity_500m
```

Run low/high area plus emission-factor envelopes:

```bash
# Lower-bound envelope: high threshold, low emission factors
python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster --full_model --chunk_size 1 \
  --start_year 2021 --end_year 2024 --interval_type five_year \
  --count_burned_years --peat_dataset ogh \
  --peat_threshold_by_biome {SCENARIO_BOUNDS_THRESHOLDS} \
  --fscore_metric f1 \
  --peat_threshold_scenario low_area \
  --drainage_distance_threshold_m 500 \
  --emission_factor_variant low --create_zarr \
  --run_date {RUN_DATE} --run_name ogh_sensitivity_low

# Upper-bound envelope: low threshold, high emission factors
python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster --full_model --chunk_size 1 \
  --start_year 2021 --end_year 2024 --interval_type five_year \
  --count_burned_years --peat_dataset ogh \
  --peat_threshold_by_biome {SCENARIO_BOUNDS_THRESHOLDS} \
  --fscore_metric f1 \
  --peat_threshold_scenario high_area \
  --drainage_distance_threshold_m 500 \
  --emission_factor_variant high --create_zarr \
  --run_date {RUN_DATE} --run_name ogh_sensitivity_high
```

## Stage 7: Aggregate 10x10 Degree Tiles

Primary script:

- `src/scripts/core_model/02_aggregate_soils_outputs.py`

Run for every completed model run:

```bash
python -m src.scripts.core_model.02_aggregate_soils_outputs \
  -cn drainage_cluster \
  --run_name {RUN_NAME} \
  --output_date {RUN_DATE} \
  --interval_end_years {YEARS}
```

Use:

- baseline: `2005 2010 2015 2020 2024`
- sensitivities: `2024`

## Stage 8: Zonal Statistics

Primary scripts:

- `src/scripts/zonal_statistics/02_run_zonal_stats.py`
- `src/scripts/zonal_statistics/03_qaqc_combined_state_alignment.py`

Run zonal stats for every completed model run:

```bash
python -m src.scripts.zonal_statistics.02_run_zonal_stats \
  --model_version {MODEL_VERSION} \
  --run_date {RUN_DATE} \
  --interval_type five_year \
  --interval_end_years {YEARS} \
  --cluster_name drainage_cluster \
  --run_name {RUN_NAME} \
  --chunk_size 10000 \
  --diagnostics off \
  --overwrite_existing
```

By default, zonal stats now discovers the 10x10 tiles with aggregated
`combined_state` rasters for each `{RUN_NAME}` / `{RUN_DATE}` /
inventory period and only loops over those tiles. Use
`--data_tile_filter off` only when intentionally processing the full
canonical tile roster.

Optional combined-state QA for runs/intervals that need a publication gate:

```bash
python -m src.scripts.zonal_statistics.03_qaqc_combined_state_alignment \
  --model_version {MODEL_VERSION} \
  --run_name {RUN_NAME} \
  --run_date {RUN_DATE} \
  --intervals {INTERVAL_FOLDERS}
```

## Stage 9: Publication Tables and Figures

Primary scripts:

- `src/scripts/zonal_statistics/pub_scripts/pub_assets.py`
- `src/scripts/zonal_statistics/pub_scripts/pub_compare_runs.py`
- `src/scripts/zonal_statistics/pub_scripts/pub_fao.py`
- `src/scripts/zonal_statistics/pub_scripts/pub_nghgi.py`
- `src/scripts/zonal_statistics/pub_scripts/extract_organic_soil_jrc.py`

Run per-run publication assets for every run:

```bash
python -m src.scripts.zonal_statistics.pub_scripts.pub_assets \
  --model_version {MODEL_VERSION} \
  --run_name {RUN_NAME} \
  --run_date {RUN_DATE} \
  --years {YEARS} \
  --topn 20
```

Run cross-run comparisons:

```bash
python -m src.scripts.zonal_statistics.pub_scripts.pub_compare_runs \
  --years 2024 \
  --run "ogh_biome_thresholds={MODEL_VERSION}:{RUN_DATE}|OGH" \
  --run "gfw_standard_model_500m={MODEL_VERSION}:{RUN_DATE}|GFW" \
  --run "gpd_standard_model_500m={MODEL_VERSION}:{RUN_DATE}|GPD" \
  --run "ogh_sensitivity_250m={MODEL_VERSION}:{RUN_DATE}|Low drainage" \
  --run "ogh_sensitivity_500m={MODEL_VERSION}:{RUN_DATE}|Baseline drainage" \
  --run "ogh_sensitivity_750m={MODEL_VERSION}:{RUN_DATE}|High drainage" \
  --run "ogh_sensitivity_low={MODEL_VERSION}:{RUN_DATE}|Low envelope" \
  --run "ogh_sensitivity_high={MODEL_VERSION}:{RUN_DATE}|High envelope"
```

Run FAOSTAT comparison:

```bash
python -m src.scripts.zonal_statistics.pub_scripts.pub_fao \
  --years 2024 \
  --run "ogh_biome_thresholds={MODEL_VERSION}:{RUN_DATE}|OGH" \
  --run "gfw_standard_model_500m={MODEL_VERSION}:{RUN_DATE}|GFW" \
  --run "gpd_standard_model_500m={MODEL_VERSION}:{RUN_DATE}|GPD"
```

Run NGHGI comparison:

```bash
python -m src.scripts.zonal_statistics.pub_scripts.pub_nghgi \
  --years 2005 2010 2015 2020 2024 \
  --run "ogh_biome_thresholds={MODEL_VERSION}:{RUN_DATE}|OGH"
```

Recommended NGHGI preflight:

```bash
python -m src.scripts.zonal_statistics.pub_scripts.pub_nghgi \
  --validate_jrc
```

## Stage 10: Global Raster Aggregation

Primary scripts:

- `src/scripts/postprocessing/visualization/create_global_raster.py`
- `src/scripts/postprocessing/visualization/build_drained_binary_raster.py`

Aggregate 0.01-degree global rasters for the three inventory source versions at minimum:

```bash
python -m src.scripts.postprocessing.visualization.create_global_raster \
  -cn drainage_cluster \
  --run_name {RUN_NAME} \
  --model_version {MODEL_VERSION} \
  --date_tag {RUN_DATE} \
  --target_deg 0.01 \
  --native_deg 0.00025
```

When `combined_state` is included, this command also writes a companion
`combined_state_reclassified` GeoTIFF with these UInt8 classes:
`1=undrained organic soil`, `2=drained only`, `3=burned only`,
`4=drained+burned`, and `255=nodata/non-organic`.

Do not render display assets for this rerun gate; the 0.01-degree aggregated
rasters are the required map artifacts.

Run at least for:

- `ogh_biome_thresholds`
- `gfw_standard_model_500m`
- `gpd_standard_model_500m`

Add sensitivity maps only if the publication needs maps for those scenarios.

## Final QA Checklist

- Confirm no stale S3 date tags remain in constants or script-level defaults.
- Confirm `{RUN_DATE}`, `{OGH_PROB_DATE}`, and `{MODEL_VERSION}` are recorded
  in a run manifest.
- Confirm OGH probability zarr date matches the new OGH tiles.
- Confirm threshold values are on the expected scale for the core model.
- Confirm roads/canals distance date folders in `constants_and_names.py` match
  the distance rasters used by the run.
- Confirm all expected intervals exist in model mega-zarr, aggregated 10x10
  outputs, and zonal-statistics parquet.
- Confirm `pub_compare_runs.py` recognizes the chosen run names.
- Archive threshold diagnostics, area curves, command logs, chunk stats, zonal
  manifests, and publication CSVs with the run.
