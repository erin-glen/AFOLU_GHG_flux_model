# Aligned input-dataset maps

`src.scripts.postprocessing.visualization.create_input_dataset_maps` creates a
set of borderless raster maps for design and publication work. Every output
uses one explicitly defined area of interest (AOI), CRS, affine transform,
width, and height, so the PNGs can be stacked without manual registration.

The script is intended for a small figure package, typically several model
inputs or outputs over one regional AOI. It runs locally with Rasterio; a Coiled
cluster is not needed for this use case.

## Quick start

From the repository root, first validate the configuration and resolve its S3
paths without reading the rasters:

```bash
python -m src.scripts.postprocessing.visualization.create_input_dataset_maps \
  --config docs/input_dataset_maps.example.json \
  --output-dir C:/tmp/afolu/input_dataset_maps/borneo \
  --validate-only
```

Then render the maps:

```bash
python -m src.scripts.postprocessing.visualization.create_input_dataset_maps \
  --config docs/input_dataset_maps.example.json \
  --output-dir C:/tmp/afolu/input_dataset_maps/borneo
```

The example writes eight `1800 x 1500` PNGs for a Central Kalimantan AOI in
Borneo: six model inputs plus the final state code and total drained emissions
for the production baseline. It also requests aligned GeoTIFFs for spatial QA.

## Output contract

The top-level `target` object defines the only output grid. For every dataset,
the script uses a `WarpedVRT` to produce exactly the target:

- CRS;
- bounds in `[west, south, east, north]` order;
- width and height;
- affine transform; and
- array shape.

The rendered PNG is the same width and height as the aligned array. It has no
axes, title, legend, margin, or automatic tight cropping. Nodata is transparent
unless a different `style.nodata_color` is configured. This makes each PNG a
registration-ready design asset, not a finished cartographic panel.

The output manifest, `input_dataset_maps.manifest.json`, records the target
grid, resolved sources, per-layer statistics, rendering limits, missing inputs,
and SHA-256 hashes. Ordered filenames such as `01_land_cover.png` preserve the
dataset order in the JSON configuration.

## GeoTIFF and COG scope in version 1

Set top-level `write_aligned_geotiff` to `true`, or pass
`--write-aligned-geotiff`, to retain one aligned float32 GeoTIFF per layer under
`<output-dir>/aligned/`. These files use DEFLATE compression and are tiled when
their dimensions permit it.

Version 1 does **not** create or validate Cloud Optimized GeoTIFFs (COGs). It
does not build overview pyramids, run a COG validator, or upload outputs to S3.
The aligned GeoTIFFs are local QA/interchange artifacts. A later COG workflow
should be a separate, explicit conversion step so that COG layout and overview
resampling can be tested independently.

## Raw inputs versus model-transformed layers

A dataset with `model_input` is resolved through
`constants_and_names.get_dynamic_download_dict`. This ties the figure to the
same current S3 catalog used by the organic-soils model. It does **not** execute
the core model or automatically reproduce downstream transformations.

For example:

- with `model_context.peat_dataset` set to `ogh`, `model_input: "peat"`
  resolves the raw, unthresholded OGH probability raster rather than the
  model's thresholded peat mask;
- climate-domain remapping performed in the core model is not applied;
- extraction values are not automatically binarized; and
- annual burned-area inputs are not combined across years.

The first six example layers intentionally show raw model inputs: they use no
`value_transform`, `scale`, or `offset`. The final two layers read the completed
production-baseline `2021_2024` global outputs for model version `1_0_1`, run
`ogh_mixed_f1_f15_f2_20260513`: the publication-facing reclassified state code
and total drained emissions. The example uses the completed 0.005-degree
versions to retain more regional detail. Warping/resampling onto the shared
display grid still occurs and is recorded in the manifest. If a figure needs another
model-consumed representation, configure `value_transform` explicitly and
describe that choice in the figure metadata. Supported transforms are `none`,
`binary_nonzero`, and `threshold_gte`.

Do not mask zero globally. Several stored model rasters declare zero as nodata,
while zero is also the useful display class for absence in layers such as
burned area and planted-forest type. The script honors raster nodata metadata
by default. Set `respect_source_nodata` to `false` only when the figure should
render that stored zero class; the Borneo example does this for those two
categorical layers. Configure `source_nodata`, `mask_values`, `valid_min`, and
`valid_max` per dataset only when their source semantics justify it.

## Model-input resolver

When any layer uses `model_input`, the configuration must include:

```json
"model_context": {
  "interval_start_year": 2021,
  "interval_end_year": 2024,
  "peat_dataset": "ogh"
}
```

The standard resolver keys are:

- `land_cover`
- `peat`
- `dadap`
- `engert`
- `grip`
- `osm_roads`
- `osm_canals`
- `planted_forest_type`
- `extraction`
- `climate_domain`
- `descals_type`
- `mangrove_extent`
- `tidal_marsh`
- `ogh`
- `burned_area_final_<year>` for each year in the requested interval

The available keys are derived at runtime, and an invalid key produces an
error listing the keys valid for that context. When `peat_dataset` is `ogh`,
`peat` is the core-model input key for the unthresholded OGH surface; the `ogh`
key resolves the same source and normally should not be included as a second
layer.

The script calculates all model 10-degree tile IDs intersecting the target AOI
and mosaics them on the target grid. AOIs crossing the antimeridian are not
supported in version 1.

## Dataset configuration

Each entry in `datasets` requires a unique `name` and exactly one of:

- `model_input`: a key from the current model input dictionary; or
- `source`: a local path, exact S3 URI, local/S3 glob, list of sources, or a
  path template containing `{tile_id}`.

Common fields are:

| Field | Purpose |
| --- | --- |
| `title` | Human-readable title recorded in the manifest. |
| `output_name` | Optional filename stem; otherwise `name` is used. |
| `band` | One-based raster band, default `1`. |
| `kind` | `categorical` or `continuous`. |
| `resampling` | Rasterio resampling method. Use `nearest` for class codes and usually `bilinear` or `average` for continuous display surfaces. |
| `mosaic_method` | `first`, `last`, `min`, or `max`; default `last`. |
| `source_nodata` | Optional source nodata override. |
| `respect_source_nodata` | Honor the raster's stored nodata value, default `true`. Set `false` only when a stored nodata value such as class 0 must remain visible. |
| `allow_missing_sources` | Whether a missing intersecting source may be skipped. Model-resolved and tile-template layers default to allowing missing tiles. |
| `allow_empty` | Permit an all-nodata result; default `false`. |
| `scale`, `offset` | Optional numeric display transformations. |
| `value_transform` | `none`, `binary_nonzero`, or `{ "type": "threshold_gte", "value": ... }`. |
| `mask_values`, `valid_min`, `valid_max` | Dataset-specific validity filters applied after alignment and before scale/offset/value transforms. |

Categorical styles require a `colors` object mapping numeric class codes to
Matplotlib-compatible colors. Codes not listed use `unknown_color`. Continuous
styles accept `cmap`, `stretch` (`linear`, `log`, `sqrt`, or `asinh`), explicit
`vmin`/`vmax`, or a two-value `percentiles` range used to derive limits.

Categorical layers must use `nearest`, `mode`, `min`, or `max` resampling.
Nearest-neighbor is the safest default because it does not invent class codes.

## S3 access and Windows caching

On Windows, S3 GeoTIFFs are cached by default under:

```text
C:/tmp/afolu/input_dataset_map_cache
```

This avoids known instability when large, no-overview rasters are read through
GDAL `/vsis3` on Windows. Use `--cache-dir` to select another location. Cache
filenames include a hash of the full S3 URI, so different keys do not collide.
If an existing S3 key is replaced in place, remove its cached copy before
rerunning; the version-1 cache does not compare S3 ETags.

On WSL/Linux, direct `/vsis3` reading is the default. A cache can still be
requested with `--cache-dir`. `--no-cache` explicitly disables caching on any
platform. AWS credentials follow the normal boto3/GDAL credential chain, and
`--aws-profile` can select a named profile.

## No-overwrite behavior

The default is safe: if the manifest or any requested output already exists,
the script stops before rendering. Prefer a fresh output directory for each
design revision. Pass `--overwrite` only when replacing the named outputs is
intentional.

Writes are staged to temporary files and then moved into place. `--overwrite`
does not remove unrelated or obsolete files already present in the output
directory.

## Scaling guidance

The normal small regional workflow should remain local. The script has no
Coiled client or cluster argument and does not become distributed merely by
launching it near a cluster.

If a later workflow expands to many AOIs or global/high-resolution products,
parallelization should be added deliberately around independent AOI jobs. For
this repository, any Coiled launch and Dask client work must run from WSL
`Ubuntu-24.04` in the `coiled_20251119` conda environment, using
`src.scripts.utilities.create_cluster`. Do not launch from the Windows
`coiled_env` environment. Follow `AGENTS.md` for current worker sizing,
readiness checks, logging, and cluster cleanup.
