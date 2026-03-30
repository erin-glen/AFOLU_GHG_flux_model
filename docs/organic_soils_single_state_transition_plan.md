# Organic soils model: migration from dual state outputs to a single state output

## Context

The current organic soils driver emits two separate categorical state rasters:

- `drained_state` (drainage classification + drainage EF routing context)
- `burned_state` (burned emissions routing context)

This is implemented in the core driver and carried through zonal-stat cache building and publication tables.

## Current implementation overview

### Core model behavior

- The main per-pixel function computes and outputs both `drained_state` and `burned_state`, plus emissions arrays.  
- The chunk wrapper validates both state outputs against registered state dictionaries in `zonal_constants`, and can drop all-zero `burned_state`.  
- Output typing and zarr inclusion currently include both state variables.

### Zonal-stat pipeline behavior

- Cache builders and zonal runners expect *both* `drained_state_nodes` and `burned_state_nodes` zarr inputs.
- Published assets/tables maintain separate drained-state and burned-state outputs and join each to separate context views.

## Why a single state can simplify operations

1. **Less preprocessing and harmonization overhead**
   - One categorical raster to cache, load, align, and join in zonal workflows.
2. **Lower risk of state misalignment bugs**
   - Today, drained and burned state arrays are cropped/aligned separately in some zonal code paths.
3. **Simpler publication schema for downstream users**
   - One “state” axis reduces table proliferation and lessens custom post-processing for zonal stats consumers.
4. **Potential storage/read performance gains**
   - Fewer zarr groups and fewer raster artifacts in the output tree.

## Key design choices (before implementation)

## 1) What does “single state” represent?

Choose one of two models:

- **Option A: unified event-aware state code** (recommended)
  - One code namespace that can represent:
    - drained-only context,
    - burned-only context,
    - both contexts in a combined code,
    - and non-peat / no-emission states.
  - Preserves analytical fidelity for both drainage and fire analyses.

- **Option B: drainage-only state + derived burned context**
  - Keep only drainage state in outputs, infer burned context from emissions or masks during post-processing.
  - Simpler implementation, but higher risk of losing burned categorization fidelity.

## 1a) Can we preserve *all* information in one output field?

Yes — but we should not rely on longer decimal-string concatenation.  
The current implementation stores state as `uint32` arrays, so capacity is numerical (32 bits), not textual.  
To avoid information loss and keep compatibility with downstream tooling, use a **bit-packed unified state code** in one `uint32` raster.

Recommended packed layout (example):

- bits 0–7   : `drained_state_id` (0–255)
- bits 8–11  : `burned_state_id` (0–15)
- bit 12     : `has_drainage_component`
- bit 13     : `has_burned_component`
- bits 14–31 : reserved for future expansion/versioning

Why this is sufficient now:

- Current drained-state ontology is ~176 unique node meanings (fits in 8 bits).
- Current burned-state ontology is 8 unique node meanings (fits in 4 bits).
- We retain full drained and burned context in one integer output without collapsing categories.

Decoding logic is deterministic and cheap (`bitwise and` / `shift`), so zonal and publication pipelines can recover either component when needed.

## 2) Backward compatibility strategy

- Keep legacy fields as aliases for one release cycle (`drained_state`, `burned_state`) while introducing new `emissions_state` (or similar), then deprecate.
- Update zonal and publication scripts to support both schemas during transition.

## Risks and potential issues

1. **Loss of burned-state semantics**
   - Burned state currently has a dedicated ontology (`BURNED_STATE_NODE_MEANINGS`). A collapse can erase distinctions unless explicitly encoded.
2. **Lookup table breakage in publication layer**
   - `pub_assets.py`, `pub_tables.py`, and `pub_common.py` create separate drained/burned state context joins; these joins will fail or produce null context unless refactored.
3. **Zarr contract breakage**
   - Cache configs and zonal runners explicitly reference `drained_state_nodes` and `burned_state_nodes`. Removing one/both without compatibility logic breaks runs.
4. **Comparability across historical runs**
   - Existing run comparisons expect separate state breakout files (`by_drained_state_*`, `by_burned_state_*`). Trend reporting and regression tests may fail.
5. **Code-width and node namespace collisions**
   - Current validation assumes drained and burned codes are validated against separate registries. Unified namespace needs collision-safe encoding and validation.
6. **Consumer impact**
   - Any downstream notebook/BI pipeline expecting separate state tables will need migration guidance.
7. **Ambiguity if decimal concatenation is used**
   - Packing both ontologies by string/decimal concatenation can exceed practical pad-width assumptions and create parsing ambiguity over time.
   - A bit-packed schema avoids this risk.

## Recommended migration plan

### Phase 0 — Decision and specification (no code behavior change)

1. Define canonical unified state ontology and encoding rules (bit-packed `uint32`, not decimal concatenation).
2. Publish a short schema contract:
   - field name,
   - integer width + bit layout,
   - node-meaning map,
   - compatibility aliases.
3. Add migration acceptance criteria (row-count parity, emissions parity, state coverage).

### Phase 1 — Dual-write in core model

1. In `0_drainage_emissions_model.py`, compute and output new `emissions_state` alongside existing `drained_state` and `burned_state`.
2. Add validation against a new unified state mapping in `zonal_constants.py`.
3. Extend `constants_and_names.py` output lists and dtype maps with `emissions_state`.

### Phase 2 — Zonal pipeline compatibility

1. Update zarr cache builders/runners to prefer `emissions_state_nodes` when present.
2. Keep fallback support for legacy drained/burned nodes.
3. Add a compatibility transform that can reconstruct coarse drained/burned classes from unified state where possible.

### Phase 3 — Publication layer refactor

1. Refactor context view registration (`pub_assets.py` / `pub_tables.py`) to derive one state context view.
2. Replace separate by-drained/by-burned exports with:
   - `by_emissions_state_period.csv`
   - `by_country_emissions_state_period.csv`
3. Keep legacy exports (`by_drained_state_*`, `by_burned_state_*`) generated from compatibility logic for one full release window to support QA/QC.

### Phase 4 — Validation and cutover

1. Run side-by-side production-like runs for at least one interval (e.g., `2021_2024`).
2. Validate:
   - total emissions parity (global and by country),
   - peat area parity,
   - state coverage completeness (no unexpected unmapped codes).
3. Announce deprecation date for legacy state outputs.
4. Remove legacy state paths after one stable release cycle.

## Minimum test plan for the migration

1. **Unit tests**
   - Unified-state code generation for representative pixel scenarios.
   - Mapping completeness and uniqueness checks.
2. **Integration tests**
   - Zonal stats run succeeds with unified-only inputs.
   - Publication SQL emits expected non-null context fields.
3. **Regression checks**
   - Emissions totals unchanged (within tolerance) versus current dual-state workflow.
   - Country-level outputs unchanged for aggregate metrics.

## Suggested immediate next steps

1. Draft unified ontology in `zonal_constants.py` (no wiring yet).
2. Implement dual-write of `emissions_state` in core model.
3. Add compatibility loading in zonal cache/runners.
4. Execute one trial interval and compare outputs before deprecating legacy state columns.

## Decisions captured from review

1. **Legacy exports (resolved):** keep both `by_drained_state_*` and `by_burned_state_*` for one release cycle for QA/QC, then remove after validation.
2. **Identifier/output behavior (resolved):** keep downstream-facing behavior consistent with current conventions during transition.  
   - Practically: continue publishing the same familiar breakout tables/columns for one release while the unified internal field is introduced.
3. **Versioning bits (clarified):** reserve high bits now, but keep them set to `0` initially.  
   - This does **not** change current analysis behavior; it only future-proofs the field so a future schema update can be signaled without renaming the output.
4. **Cutover criteria (clarified):** this means defining objective pass/fail checks before removing legacy outputs.  
   - Example criteria:  
     - Global drained+burned totals must match legacy workflow exactly (or within an agreed tiny tolerance).  
     - Country-level totals must remain within agreed tolerance.  
     - No unmapped unified-state codes are allowed in publication outputs.
