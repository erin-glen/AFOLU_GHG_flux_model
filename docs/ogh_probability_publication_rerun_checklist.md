# OGH Probability Publication Rerun Checklist

This runbook inventories the steps needed when a refreshed OpenGeoHub organic
soil probability raster arrives and the organic-soils model must be rerun for
publication outputs.

Use placeholders consistently:

- `{RUN_DATE}`: production run date tag, `YYYYMMDD`.
- `{OGH_PROB_DATE}`: processed OGH unthresholded probability tile date tag.
- `{MODEL_VERSION}`: underscore model version, for example `0_1_4`.
- `{THRESHOLD_DIR}`: local directory containing threshold diagnostics.
- `{BASELINE_BIOME_THRESHOLDS}`: CSV from `biome_thresholds_summary.csv` or
  equivalent JSON thresholds for the baseline/max-F1 option.
- `{LOW_AREA_THRESHOLDS}`: high probability thresholds for the low-area envelope.
- `{HIGH_AREA_THRESHOLDS}`: low probability thresholds for the high-area envelope.

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

4. `01_build_zarr_caches.py` hard-codes `ORGANIC_PROBABILITY_DATE`. Either
   update that constant before running, or add a CLI flag for the probability
   date before the production rerun.

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
   - Pass explicit date tags with `--date`, `--raw_date`, and
     `--organic_probability_date` so multi-day reruns do not drift with
     `today_date`.
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
  --raw_date {OGH_PROB_DATE} \
  --client coiled
```

Expected output:

- 10x10 degree OGH unthresholded probability tiles under the intended
  `peat_mask/OGH/tiles_unthresholded/{OGH_PROB_DATE}/` prefix after the date
  path harmonization above.

Optional, only if roads/canals need a refreshed broad organic-soil mask:

```bash
python -m src.scripts.preprocessing.peat.peat_masks \
  --dataset ogh \
  --date {OGH_BINARY_DATE} \
  --raw_date {OGH_PROB_DATE} \
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

The probability date is currently driven by the script constant, so confirm it
matches `{OGH_PROB_DATE}` before running.

## Stage 3: Probability Area Stats and Area Curves

Primary scripts:

- `src/scripts/zonal_statistics/02b_run_probability_class_area_stats.py`
- `src/scripts/uncertainty/build_probability_area_curve.py`

Run global probability-class area by ADM0:

```bash
python -m src.scripts.zonal_statistics.02b_run_probability_class_area_stats \
  --contextual_date 20250925 \
  --probability_date {OGH_PROB_DATE} \
  --overwrite_existing
```

Run the same by ADM0 and biome:

```bash
python -m src.scripts.zonal_statistics.02b_run_probability_class_area_stats \
  --contextual_date 20250925 \
  --probability_date {OGH_PROB_DATE} \
  --include_biome \
  --overwrite_existing
```

Build area-vs-threshold curves:

```bash
python -m src.scripts.uncertainty.build_probability_area_curve \
  --probability-date {OGH_PROB_DATE} \
  --output {THRESHOLD_DIR}/area_vs_threshold_{OGH_PROB_DATE}.csv

python -m src.scripts.uncertainty.build_probability_area_curve \
  --probability-date {OGH_PROB_DATE} \
  --per-biome \
  --output {THRESHOLD_DIR}/area_vs_threshold_{OGH_PROB_DATE}_biome.csv
```

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

When `--area-curve-table` is supplied without `--mapped-area`, the diagnostic
script infers mapped area from the area curve at the operational/report
threshold.

Outputs to retain:

- `threshold_metrics.csv`
- `selected_thresholds.csv`
- `extent_bounds_summary.csv`, if mapped area is supplied
- `area_bound_threshold_matches.csv`
- `biome_thresholds_summary.csv`
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

## Stage 6: Core Model Run Matrix

Primary script:

- `src/scripts/core_model/0_drainage_emissions_model.py`

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
  --peat_threshold_by_biome {LOW_AREA_THRESHOLDS} \
  --fscore_metric f1 \
  --drainage_distance_threshold_m 500 \
  --emission_factor_variant low --create_zarr \
  --run_date {RUN_DATE} --run_name ogh_sensitivity_low

# Upper-bound envelope: low threshold, high emission factors
python -m src.scripts.core_model.0_drainage_emissions_model \
  --cluster_name drainage_cluster --full_model --chunk_size 1 \
  --start_year 2021 --end_year 2024 --interval_type five_year \
  --count_burned_years --peat_dataset ogh \
  --peat_threshold_by_biome {HIGH_AREA_THRESHOLDS} \
  --fscore_metric f1 \
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

## Stage 10: Global Map Aggregation

Primary scripts:

- `src/scripts/postprocessing/visualization/create_global_raster.py`
- `src/scripts/postprocessing/visualization/create_global_displays.py`
- `src/scripts/postprocessing/visualization/build_drained_binary_raster.py`

Aggregate global rasters for the three inventory source versions at minimum:

```bash
python -m src.scripts.postprocessing.visualization.create_global_raster \
  -cn drainage_cluster \
  --run_name {RUN_NAME} \
  --model_version {MODEL_VERSION} \
  --date_tag {RUN_DATE} \
  --target_deg 0.01 \
  --native_deg 0.00025
```

Render display assets:

```bash
python -m src.scripts.postprocessing.visualization.create_global_displays \
  --date_tag {RUN_DATE} \
  --run_name {RUN_NAME} \
  --model_version {MODEL_VERSION} \
  --read_from_s3
```

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
