# pub_nghgi — model vs NGHGI organic-soil comparison

`pub_nghgi.py` compares this repository's drained organic-soil model outputs
against country-reported greenhouse-gas inventories (NGHGI) from the UNFCCC
Common Reporting Tables (CRT) for 36+ Annex I parties and ~30 BTR1 reporters.

The companion loader `extract_organic_soil_jrc.py` ingests the JRC-aggregated
data drop; per-country CRT extracts are loaded from compiled CSVs prepared
upstream of this repo.

```text
src/scripts/zonal_statistics/pub_scripts/
├── pub_nghgi.py                    main driver
├── extract_organic_soil_jrc.py     JRC xlsx loader
└── README_pub_nghgi.md             this file
```

---

## 1. Inputs

### Model side: zonal-statistics parquets on S3

`s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_<v>/zonal_stats/<run_name>/<run_date>/<interval>/combined_state/*.parquet`

`combined_state/` is the unified output that merges what used to be separate
`drained/` and `drained_co2_n2o/` directories. It contains rows of
`(gadm_adm0, drained_state_meaning, flux_type, value, ...)`.

The flux types pub_nghgi reads:

| flux_type pattern              | What it represents                          |
|--------------------------------|---------------------------------------------|
| `area__ha`                     | Pixel area in hectares (drained + undrained)|
| `drained_co2%` (excl. offsite) | Drained-peat CO₂ in Mg CO₂/yr               |
| `drained_n2o%`                 | Drained-peat N₂O in **Mg CO₂e/yr**          |

The SQL is built by `pub_common.table_nghgi_comparison_subset_sql`. It maps
the model's `combined_state` / `emissions_state` strings to coarse IPCC
land-use categories (Forest, Cropland, Grassland, Wetland, Settlement,
Otherland) with:

* `Oil Palm` and `Other plantation` folded into **Cropland**
* `Extraction` and `Undrained (unclassified)` folded into **Wetland**

### NGHGI side: two compiled sources

**Raw extracts** (12 countries, deeper variable detail) at:
`C:\GIS\Data\Global\Wetlands\organic_soil_nghgi\`

| File                                | CRT origin   | Key columns                                         |
|-------------------------------------|--------------|-----------------------------------------------------|
| `organic_soil_compiled.csv`         | Table 4(II)  | drained-organic area, gas-specific emissions, EFs   |
| `organic_soil_cstock_compiled.csv`  | Tables 4.A–F | total-organic area, organic-soil C-stock change     |

Countries: CAN, CHN, DEU, FIN, IDN, IRL, KAZ, MYS, NOR, RUS, SWE, USA.
Extracted from raw CRT workbooks under `WRI/WRI/{2024_ETF,2025_AnnexI}/<ISO3>/`.
The extraction scripts referenced in the source README are not in this repo —
they were ephemeral one-offs.

**JRC pre-aggregated drop** (36 Annex I + 88 BTR1, shallower variable detail) at:
`C:\GIS\Data\Global\Wetlands\organic_soil_nghgi\JRC_aggregations\<date>\`

Loaded by `extract_organic_soil_jrc.py`. Two products:

* `load_jrc_landuse_tables(annexi_dir)` — Tables 4.A–F top-level "Total"
  rows per `(iso3, year, land_use)`: area_total_kha, area_organic_kha,
  cstock_organic_ktC.
* `load_jrc_table_3d(annexi_dir, btr1_path)` — Table 3.D.1.f "Cultivation
  of organic soils (i.e. histosols)" per `(iso3, year)`: area_ha, n2o_kt.

The 2026-04-30 drop covers years 1990–2024 for AnnexI and 1990–2023 for BTR1.

---

## 2. Country source routing

The two NGHGI sources cover overlapping but distinct iso3 sets. pub_nghgi
routes per-country to keep the comparison consistent:

| Bucket                          | Count | Members                                  | Source for area + cstock-CO₂ | Source for T4(II) drained / EFs |
|---------------------------------|-------|------------------------------------------|------------------------------|---------------------------------|
| **Overlap** (raw + JRC)         | 6     | CAN, DEU, FIN, IRL, NOR, SWE             | **JRC 2026** (overrides raw) | Raw 2025                         |
| **JRC-only**                    | 30    | AUS, AUT, BEL, BGR, BLR, CHE, CYP, CZE, DNK, ESP, EST, GRC, HRV, ISL, ITA, JPN, LIE, LTU, LUX, LVA, MCO, MLT, NLD, NZL, POL, ROU, SVK, SVN, TUR, UKR | JRC 2026                     | (not available)                  |
| **Raw-only** (excluded by JRC)  | 6     | CHN, IDN, KAZ, MYS, RUS, USA             | Raw 2025                     | Raw 2025                         |

Russia is excluded from the EEA-aggregated JRC drop post-invasion; the
others are non-Annex I or BTR1 parties not covered by the AnnexI 2026
distribution. `--no_jrc` opts out of JRC entirely (raw-only fallback).

JRC 2026 overrides raw 2025 for the 6 overlap countries because it reflects
each country's most recent submission cycle. Validation against the 6
overlap iso3s found ~13% area / ~14% cstock rows differ by >10% between
2025 and 2026 cycles — country revisions, not extraction errors. Run
`--validate_jrc` to regenerate the overlap scatter PNGs and diff CSV.

---

## 3. Comparison metrics and figure inventory

Five "main" comparisons aggregated across IPCC LULUCF categories (4.A–F),
plus two "Table 3.D" comparisons restricted to cultivated organic soils:

| Figure                                | Model side                                | NGHGI side                                                               | Unit          |
|---------------------------------------|-------------------------------------------|--------------------------------------------------------------------------|---------------|
| `topn_compare_total_area_<int>`       | drained + undrained peat extent           | Tables 4.A–F `area_organic_kha` (total organic, includes drained + undrained) | Mha           |
| `topn_compare_drained_area_<int>`     | drained area                              | Table 4(II) "Drained organic soils" `area_kha`                          | Mha           |
| `topn_compare_undrained_area_<int>`   | undrained area                            | total organic (4.A–F) − drained (4(II))                                 | Mha           |
| `topn_compare_co2_<int>`              | drained `drained_co2%` excl. offsite      | Table 4(II) `em_co2_kt` numeric **else** Tables 4.A–F `cstock_soil_organic_ktC × −44/12` | Mt CO₂e/yr    |
| `topn_compare_n2o_<int>`              | drained `drained_n2o%` (already CO₂e)     | Table 4(II) `em_n2o_kt × N2O_GWP × 1000`                                | Mt CO₂e/yr    |
| `topn_compare_t3d_area_<int>`         | drained area filtered to `T3D_N2O_MODEL_LANDUSE` | Table 3.D.1.f `Area (ha)`                                          | Mha           |
| `topn_compare_t3d_n2o_<int>`          | drained N₂O filtered to `T3D_N2O_MODEL_LANDUSE` | Table 3.D.1.f `n2o_kt × N2O_GWP × 1000`                            | Mt CO₂e/yr    |

Plus two non-replaced figures:

| Figure                  | Purpose                                                                |
|-------------------------|------------------------------------------------------------------------|
| `scatter_area_<int>`    | All-country distribution view (model vs NGHGI on log-log axes)         |
| `scatter_co2_<int>`     | Same, for CO₂                                                          |

Each `topn_compare_*` figure shows the same 10 focus countries in the same
order (defined below) with a color scheme distinct to that figure for
visual differentiation.

---

## 4. Key assumptions and parameters

### `FOCUS_COUNTRIES_ISO3`

```python
("CAN", "DEU", "FIN", "IDN", "IRL", "MYS", "NOR", "RUS", "SWE", "USA")
```

The 10 raw-extract countries with full Table 4(II) reporting. Used to
restrict every `topn_compare_*` figure to the same country set so
cross-figure comparison is meaningful.

The y-axis order is computed once per run from each country's NGHGI total
organic-soil area (latest interval, descending), then applied identically
to every figure. Largest reporter at top of plot, smallest at bottom.

### `T3D_N2O_MODEL_LANDUSE`

```python
("Cropland",)   # 2026-04-30 default
```

Land uses summed on the model side for the Table 3.D.1.f "Cultivation of
organic soils" comparison. Cropland-only because the model classifies
most countries' cultivated histosols as Grassland, but most of that
"drained Grassland on organic soils" is non-cultivated peatland that
countries do not report under 3.D.1.f. Including grassland inflated the
model side wildly.

`--t3d_landuse Cropland Grassland` overrides for the broader IPCC 2006
Vol. 4 Ch. 11 scope (cultivated cropland + managed grassland histosols)
as a sensitivity test.

### `N2O_GWP`

```python
N2O_GWP = 273.0
```

AR6 100-year GWP. Converts NGHGI N₂O (kt N₂O mass) to CO₂e for like-for-like
comparison with the model's `drained_n2o_Mg_CO2e` flux. Matches `pub_fao`
convention. AR4 (265) and AR5 (298) are alternatives if you need to match a
specific country's reporting cycle exactly.

### `C_TO_CO2`

```python
C_TO_CO2 = 44.0 / 12.0
```

Stoichiometric ratio used to convert organic-soil C-stock change (kt C in
Tables 4.A–F) to CO₂ emissions. Sign is flipped because a negative cstock
change (carbon loss) corresponds to a positive CO₂ emission.

### Inventory interval averaging

Each model run produces parquet outputs in 4-year intervals (e.g.,
`2021_2024/`). The NGHGI side averages per-country numeric values across
the same window — e.g., `--years 2024` triggers comparison against the
2021–2024 mean of NGHGI annual values where reported.

---

## 5. Methodological choices

### Apples-to-apples matching of land-use categories

Most NGHGI reporters cover only a subset of IPCC LULUCF categories per
metric (CAN reports drained area only for Forest + Wetland; FIN no
Cropland/Grassland drained; USA same; etc.). The model produces non-zero
values in every `(iso3, land_use)` cell.

Per-country roll-up therefore uses **matched-scope summation**: for each
`(model, nghgi)` column pair, the model side sums across only the
`(iso3, land_use)` tuples where the NGHGI metric has reported numeric
data on that axis. Otherwise the model side inflates by including
categories the country does not report.

Affects all five main figures. T3D figures are already country-level
(no LU breakdown), so the issue does not apply there.

`0` is treated as "not reported" because the upstream
`nghgi_by_iso_landuse_interval` aggregator silently maps NaN→0 in some
sub-aggregations; distinguishing real zero from missing is not possible
without an upstream fix to that aggregator.

### CO₂ source preference (Table 4(II) vs Tables 4.A–F)

For each `(iso3, land_use, year)`, the NGHGI CO₂ value is:

1. Table 4(II) `em_co2_kt` if numeric (some non-Annex I countries report CO₂
   directly in Table 4(II));
2. Otherwise derived from Tables 4.A–F: `−cstock_soil_organic_ktC × 44/12`
   (most Annex I countries flag T4(II) CO₂ as `IE` and put the value here).

The chosen source per country-year-LU is recorded in the
`nghgi_em_co2_source` column of `nghgi_country_landuse.csv`.

### NGHGI undrained area derivation

Undrained organic-soil area is not reported directly by countries. We
derive it as:

```
undrained = max(0, total_organic_4.A-F  −  drained_4(II))
```

where both sides are summed up from the per-LU rows. Where T4(II) drained
is not reported, undrained is `NaN` (not a derivable proxy).

### Model land-use folding for NGHGI comparability

Within `pub_common.table_nghgi_comparison_subset_sql`:

* Coastal mangrove / tidal marsh → **Wetland**
* Oil palm, other plantation, planted, tree-crop → **Oil Palm** / **Other plantation**
  (then folded to **Cropland** in pub_nghgi.MODEL_LANDUSE_FOLD)
* Settlement, built-up, urban → **Settlement**
* Peat extraction, cutover → **Extraction** (folded to **Wetland**)
* Otherwise model's native land-use string

This matches the IPCC LULUCF taxonomy that countries report under.

---

## 6. Outputs

For each `--years` end-year and `--run`, pub_nghgi writes:

```
<OUT_DIR_ROOT>/version_<v>/<run_name>/<run_date>/
  figures/
    scatter_area_<interval>.png
    scatter_co2_<interval>.png
    topn_compare_total_area_<interval>.png
    topn_compare_drained_area_<interval>.png
    topn_compare_undrained_area_<interval>.png
    topn_compare_co2_<interval>.png
    topn_compare_n2o_<interval>.png
    topn_compare_t3d_area_<interval>.png
    topn_compare_t3d_n2o_<interval>.png
    data/
      model_country_landuse.csv          per (iso3, land_use, interval)
      nghgi_country_landuse.csv          per (iso3, land_use, interval)
      model_vs_nghgi.csv                 outer-joined comparison frame
      nghgi_t3d_country.csv              per (iso3, interval), 3.D.1.f only
      model_vs_nghgi_t3d.csv             outer-joined 3.D comparison
```

`--validate_jrc` mode writes to a separate path:

```
<OUT_DIR_ROOT>/validation/<jrc_drop_label>/
  figures/validation/
    scatter_area_organic_raw_vs_jrc.png
    scatter_cstock_organic_raw_vs_jrc.png
    data/validation_overlap.csv
```

---

## 7. Usage

### Standard run

```bash
cd /mnt/c/gis/git/AFOLU_GHG_flux_model
python -m src.scripts.zonal_statistics.pub_scripts.pub_nghgi \
    --years 2024 \
    --run "ogh_biome_thresholds=0_1_4:20260417|OGH"
```

### Multi-source comparison

```bash
python -m src.scripts.zonal_statistics.pub_scripts.pub_nghgi \
    --years 2005 2010 2015 2020 2024 \
    --run "ogh_sensitivity_500m_10=0_9_7:20251118|OGH" \
    --run "gfw_standard_model_500m=0_9_7:20251120|GFW" \
    --run "gpd_standard_model_500m=0_9_7:20251120|GPD"
```

The first `--run` determines the output directory.

### Sensitivity flags

| Flag                              | Effect                                                                                  |
|-----------------------------------|-----------------------------------------------------------------------------------------|
| `--no_jrc`                        | Skip the JRC drop entirely (raw-only 12-country comparison)                             |
| `--t3d_landuse Cropland Grassland`| Broaden the 3.D.1.f model scope to include managed grassland histosols                  |
| `--validate_jrc`                  | Validation-only mode: raw vs JRC overlap scatter + diff CSV; skip the model side        |
| `--data-only`                     | Skip figure generation, write CSVs only                                                 |

### Environment overrides

| Variable                  | Default                                                                                 |
|---------------------------|-----------------------------------------------------------------------------------------|
| `AFOLU_PUB_NGHGI_DIR`     | `/mnt/c/tmp/pub_nghgi`                                                                  |
| `AFOLU_NGHGI_DATA_DIR`    | `/mnt/c/GIS/Data/Global/Wetlands/organic_soil_nghgi`                                    |
| `AFOLU_NGHGI_JRC_DIR`     | `/mnt/c/GIS/Data/Global/Wetlands/organic_soil_nghgi/JRC_aggregations/2026-04-30`        |

---

## 8. Known caveats and findings

### Russia (RUS) is a 3.D.1.f outlier by design

Russia's BTR1 reports Table 3.D.1.f as actively cultivated histosols only;
the model counts every drained-peatland pixel labeled cropland or grassland
across the country. Even with cropland-only model scope, RUS is ~22× NGHGI
on cultivation N₂O. Treat as a definitional gap, not a model bug.

### Canada (CAN) total-area divergence is a CRT taxonomy gap

Canada's Tables 4.A–F barely classify any Canadian peatland as "organic
soils" under the LULUCF taxonomy — most boreal peatland is reported
elsewhere or not at all. The model finds extensive Canadian peat, giving
~3000× model/NGHGI on total organic area. The matching logic does not
fix this because the sparse-but-reported categories (Wetland, Settlement)
on the NGHGI side are tiny relative to the model's full Canadian peat
extent. Real methodological gap, surface as caveat in any presentation.

### Model classifies most cultivated histosols as Grassland

For most NGHGI-reporting countries, the unified-state SQL classifies
drained cultivated histosols as **Grassland rather than Cropland**.
Grassland share of model 3.D.1.f-equivalent N₂O: BLR 98%, GBR 99%,
DEU/NLD/COL/SWE 95–96%, FIN 86%, POL 76%, NZL 66%, UKR 64%, USA 55%,
RUS 45%, IDN 30%, MYS 8%.

Implication: cropland-only model scope (the default for `topn_compare_t3d_*`)
under-reports for EU/Annex-I countries (DEU/IRL ~0.0× NGHGI). The
under-reporting is a model land-use classification artifact, not a
peat-detection failure. Including grassland inflates against country
reporting practice. The current default is the most defensible single
choice; document the artifact when presenting.

### OGH peat probability raster excludes Greenland and >76°N

The upstream OGH peat-probability raster has spatial bounds [−56°S, +76°N]
and is nodata-masked over all of Greenland even within that band. Country
totals for GRL, SVB (Svalbard), and the high Canadian Arctic Archipelago
report as zero, not estimated. Affects model side only; NGHGI side is
unaffected because Greenland reports under DNK and high-Arctic territories
are usually not reported as organic soils. Surface as caveat for any
country totals near the Arctic.

### NaN→0 quirk in `nghgi_by_iso_landuse_interval`

The aggregator's per-(iso3, year, land_use) sums use plain `.sum()` in
several sub-aggregations, which silently maps NaN → 0. So a country that
reports `IE` / `NE` / `NA` for a category appears as a numeric `0` in the
joined frame rather than a missing value. The matching logic in §5 treats
0 and NaN identically as "not reported" to compensate; a future cleanup
should add `min_count=1` to the upstream aggregations and propagate sentinel
flags through to the figure-building step.

### Non-stacked, linear, top-10

Stacked drained+undrained or CO₂+N₂O figures were tried and removed:
log scale on stacked bars distorts segment proportionality; linear scale
on stacked bars has cross-country range that buries smaller reporters.
Each metric is now its own figure on linear scale, with the same 10 focus
countries in the same order, allowing direct visual comparison across the
figure set.

---

## 9. Memory / decision log

The current architecture and the design rationale for the cropland-only
T3D scope, the 6/30/6 country-routing split, and the matched-scope
aggregation are tracked in
`~/.claude/projects/<project>/memory/project_pub_nghgi_jrc.md`. Update
that memory when reporting practice or scope decisions change.

Branch: `feature/organic_soils_time` (commits `27a8c86`..`a06858c` make up
the JRC integration + figure-format work, 2026-04-15..2026-04-30).
