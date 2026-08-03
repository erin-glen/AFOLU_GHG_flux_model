# pub_nghgi: model vs NGHGI organic-soil comparison

`pub_nghgi.py` compares this repository's drained organic-soil model outputs
with country-reported greenhouse-gas inventory data from UNFCCC Common
Reporting Tables (CRTs) and JRC pre-aggregated inventory extracts.

The pipeline is intended for publication figures and diagnostic CSVs. It is
not a generic NGHGI parser: upstream extraction/aggregation is expected to
have already produced the raw compiled CSVs and the JRC workbook drop.

```text
src/scripts/zonal_statistics/pub_scripts/
|-- pub_nghgi.py                 main comparison and figure driver
|-- extract_organic_soil_jrc.py  loader for the JRC xlsx drop
`-- README_pub_nghgi.md          this file
```

## Quick Start

Run the latest interval for one model source:

```bash
cd /mnt/c/GIS/git/AFOLU_GHG_flux_model

python -m src.scripts.zonal_statistics.pub_scripts.pub_nghgi \
  --years 2024 \
  --run "ogh_biome_thresholds=0_1_4:20260417|OGH"
```

Run all inventory intervals and write only CSVs:

```bash
python -m src.scripts.zonal_statistics.pub_scripts.pub_nghgi \
  --years 2005 2010 2015 2020 2024 \
  --run "ogh_biome_thresholds=0_1_4:20260417|OGH" \
  --data-only
```

Validate the JRC Annex I 2026 aggregate against the raw CRT extracts:

```bash
python -m src.scripts.zonal_statistics.pub_scripts.pub_nghgi \
  --validate_jrc
```

## Inputs

### Model zonal statistics

The model side is read from `combined_state` parquet tiles on S3:

```text
s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/
  version_<model_version>/zonal_stats/<run_name>/<run_date>/
  <interval>/combined_state/*.parquet
```

The interval folder is named `<start>_<end>`, such as `2021_2024`. The SQL
used for this comparison is `pub_common.table_nghgi_comparison_subset_sql`.
It returns country, interval, coarse land-use, drained area, undrained area,
drained on-site CO2, and drained N2O.

Model N2O is stored upstream as CO2e using the project-wide AR6 GWP100 value
for N2O (`273`). In this comparison pipeline, that original value is preserved
in `drained_n2o_Mg_CO2e_yr_model_gwp`, and the comparison value
`drained_n2o_Mg_CO2e_yr` is rescaled to the inventory GWP described below.

### Raw NGHGI compiled CSVs

Default directory:

```text
C:/GIS/Data/Global/Wetlands/organic_soil_nghgi
```

Required files:

| File | CRT origin | Used for |
| --- | --- | --- |
| `organic_soil_compiled.csv` | Table 4(II) | Drained organic-soil area, CO2, N2O, CH4, implied EFs |
| `organic_soil_cstock_compiled.csv` | Tables 4.A-4.F | Total organic-soil area and organic-soil carbon-stock change |

The current raw extract covers the detailed 12-country set:
`CAN, CHN, DEU, FIN, IDN, IRL, KAZ, MYS, NOR, RUS, SWE, USA`.

### JRC pre-aggregated drop

Default directory:

```text
C:/GIS/Data/Global/Wetlands/organic_soil_nghgi/JRC_aggregations/2026-04-30
```

`extract_organic_soil_jrc.py` loads two products:

| Loader | Source | Used for |
| --- | --- | --- |
| `load_jrc_landuse_tables()` | Annex I 2026 Tables 4.A-4.F | Top-level total area, organic-soil area, and organic-soil carbon-stock change by country, year, and land-use |
| `load_jrc_table_3d()` | Annex I 2026 Table 3.D plus BTR1 2024 Table 3.D | Table 3.D.1.f cultivation-of-organic-soils area and N2O |

When Annex I 2026 and BTR1 2024 both cover the same `(iso3, year)` in
Table 3.D, the Annex I 2026 row is retained.

## Source Routing

By default, JRC data are used where available.

| Country bucket | Members | Source use |
| --- | --- | --- |
| Raw + JRC overlap | `CAN, DEU, FIN, IRL, NOR, SWE` | JRC 2026 replaces raw Tables 4.A-4.F total organic area and C-stock CO2. Raw Table 4(II) remains the source for drained area and gases. |
| JRC-only | Annex I countries present in the JRC drop but absent from the raw extract | JRC 2026 supplies total organic area and C-stock CO2. Table 4(II) drained area and gases are unavailable. |
| Raw-only | `CHN, IDN, KAZ, MYS, RUS, USA` | Raw compiled CSVs supply both Table 4(II) and Tables 4.A-4.F values. |

The undrained-area proxy deliberately does not subtract raw Table 4(II)
drained area from a JRC 2026 total. Instead, it uses raw Tables 4.A-4.F total
organic area minus raw Table 4(II) drained area. This avoids creating
artificial negative undrained values from mixed inventory cycles while still
allowing the total-area and C-stock CO2 comparison to use newer JRC data.

Use `--no_jrc` to run the raw-only 12-country comparison.

## Core Methods

### Inventory intervals

`--years` accepts inventory end years only. Valid values are the end years in
`cn.five_year_inventory_periods`, currently:

```text
2005, 2010, 2015, 2020, 2024
```

Each NGHGI metric is averaged across the corresponding inventory window, so
`--years 2024` compares the model's `2021_2024` interval with the mean of
reported annual inventory values for 2021-2024.

Every interval output carries metric-specific `*_years_available` columns.
Those counts are the number of non-null annual reported values inside the
model window for that metric. A comparison value should be treated as chartable
only when the corresponding count is at least 1.

### Land-use matching

NGHGI category codes are mapped to the six coarse IPCC LULUCF classes:

| NGHGI category | Land-use label |
| --- | --- |
| `4(II).A`, `4.A` | Forest |
| `4(II).B`, `4.B` | Cropland |
| `4(II).C`, `4.C` | Grassland |
| `4(II).D`, `4.D` | Wetland |
| `4(II).E`, `4.E` | Settlement |
| `4(II).F`, `4.F` | Otherland |

Model labels are folded to the same reporting level:

| Model label | Comparison label |
| --- | --- |
| Oil Palm | Cropland |
| Other plantation | Cropland |
| Extraction | Wetland |
| Undrained (unclassified) | Wetland |

### Table 4(II) hierarchy collapse

Table 4(II) is hierarchical. The pipeline avoids double-counting by summing
the non-overlapping parent rows (`4(II).X.1` and `4(II).X.2`) and only falling
back to deeper children when a parent is missing. All sums use `min_count=1`
so all-missing `IE`/`NE` groups remain `NaN` instead of becoming numeric zero.

### Table 4(II) gas sub-row preference

For CO2, N2O, and CH4, Table 4(II) can contain both "Total organic soils" and
"Drained organic soils" sub-rows. The comparison prefers the Drained row when
it is numeric and falls back to Total only when Drained is missing. This keeps
the NGHGI gas comparison as close as possible to the model's drained-peat
emissions while still supporting countries that only report the aggregate row.

Drained area uses the Drained row only. Total organic area comes from Tables
4.A-4.F, not from the Table 4(II) Total row.

### CO2 source preference

For each `(iso3, land_use, year)`, NGHGI CO2 is:

1. Table 4(II) `em_co2_kt` when numeric.
2. Otherwise, Tables 4.A-4.F organic-soil C-stock change converted as
   `-cstock_soil_organic_ktC * 44/12`.

The selected source is stored in `nghgi_em_co2_source`.

### N2O GWP convention

For the NGHGI comparison outputs, both model and inventory N2O are normalized
to the current inventory convention: AR5 GWP100 for N2O = `265`.

The output keeps enough metadata to avoid ambiguity:

| Column | Meaning |
| --- | --- |
| `drained_n2o_Mg_CO2e_yr_model_gwp` | Original model-side N2O CO2e using project-wide GWP100 = 273 |
| `drained_n2o_Mg_CO2e_yr` | Model N2O rescaled to inventory GWP100 = 265 |
| `model_n2o_gwp_original` | Original model GWP value |
| `comparison_n2o_gwp` | GWP used for comparison outputs |
| `nghgi_n2o_gwp`, `nghgi_t3d_n2o_gwp` | GWP used to convert NGHGI N2O mass to CO2e |

The upstream model output is not changed; normalization happens only in this
comparison layer.

### Undrained organic-soil area

Countries do not report undrained organic-soil area directly. The proxy is:

```text
undrained = raw Tables 4.A-4.F total organic area - raw Table 4(II) drained area
```

Values more than `1 ha` below zero are treated as source inconsistencies:
`nghgi_area_undrained_negative_flag = True`, the unclipped value is retained in
`nghgi_area_undrained_organic_unclipped_ha`, and the chartable undrained value
is left missing. Smaller negatives are treated as rounding noise and clipped
to zero.

The interval count `nghgi_area_undrained_years_available` requires same-year
raw total area and same-year raw drained area. `nghgi_area_undrained_basis_years_available`
records how many years had both inputs before the negative-consistency check.

### Matched-scope country rollups

Country bars are built by summing the model side only across land-use cells
where the NGHGI side has a numeric value for that metric. This keeps the
model from being inflated by categories a country did not report.

Reported zeroes are retained. A zero inventory value is information, and it
should reveal model false positives in the paired country bars.

## Table 3.D.1.f Scope

Table 3.D.1.f is "Cultivation of organic soils (i.e. histosols)" in the
Agriculture sector. Its inventory scope can include cultivated cropland and
managed grassland histosols, but countries often define reportable cultivated
grassland more narrowly than the model's coarse "drained Grassland on organic
soil" class.

For that reason, the default headline comparison is Cropland-only on the
model side:

```python
T3D_N2O_MODEL_LANDUSE = ("Cropland",)
```

When `--t3d_landuse` is omitted, the pipeline also writes an automatic
Cropland+Grassland sensitivity:

```text
model_vs_nghgi_t3d_cropland_grassland.csv
topn_compare_t3d_area_cropland_grassland_<interval>.png
topn_compare_t3d_n2o_cropland_grassland_<interval>.png
```

Pass `--t3d_landuse Cropland Grassland` if the broader scope should be the
explicit primary output for a run.

## Outputs

Default output root:

```text
<AFOLU_PUB_NGHGI_DIR or publication_root("nghgi")>/
  version_<model_version>/<run_name>/<run_date>/
```

The first `--run` entry determines the output directory.

### Data CSVs

With the default T3D settings, the data folder contains:

```text
figures/data/
  model_country_landuse.csv
  nghgi_country_landuse.csv
  nghgi_availability_matrix.csv
  model_vs_nghgi.csv
  nghgi_t3d_country.csv
  model_vs_nghgi_t3d.csv
  model_vs_nghgi_t3d_cropland_grassland.csv   # default T3D sensitivity
```

`model_vs_nghgi.csv` and the T3D joined CSVs are outer joins. NGHGI-only rows
are expected and retain interval metadata.

`nghgi_availability_matrix.csv` is a long annual matrix with one row per
country, land-use, interval, year, and metric. Important columns:

| Column | Meaning |
| --- | --- |
| `inventory_table` | Source table or derived product (`Table 4(II)`, `Tables 4.A-4.F`, `Table 3.D.1.f`, etc.) |
| `metric` | Availability metric, such as `area_drained_organic`, `co2_preferred`, `n2o_t4ii`, or `n2o_t3d` |
| `gas` | Gas when applicable (`CO2`, `N2O`, `CH4`) |
| `source` | Raw/JRC/source-preference label when a value is present |
| `has_value` | Whether that annual metric is non-null |
| `value` | The annual value used to compute interval means |

### Figures

With the default T3D settings, each requested interval writes:

```text
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
  topn_compare_t3d_area_cropland_grassland_<interval>.png
  topn_compare_t3d_n2o_cropland_grassland_<interval>.png
```

The `topn_compare_*` figures use the same focus-country set and ordering for
visual comparability:

```python
("CAN", "DEU", "FIN", "IDN", "IRL", "MYS", "NOR", "RUS", "SWE", "USA")
```

The order is computed from latest-interval NGHGI total organic-soil area and
then reused across figures.

### JRC validation mode

`--validate_jrc` writes:

```text
validation/<jrc_drop_label>/
  figures/validation/
    scatter_area_organic_raw_vs_jrc.png
    scatter_cstock_organic_raw_vs_jrc.png
    data/validation_overlap.csv
```

This mode skips the model side and does not require `--years` or `--run`.

## CLI Reference

| Flag | Meaning |
| --- | --- |
| `--years 2024 [2015 ...]` | Inventory end years to compare. Required unless `--validate_jrc` is set. |
| `--run "run_name=model_version:run_date|Label"` | Model run spec. May be repeated. Required unless `--validate_jrc` is set. |
| `--model-zonal-root <path>` | Optional isolated run root containing `<interval>/combined_state/*.parquet`; requires exactly one `--run`, requires every requested interval with a matching embedded endpoint, and bypasses the canonical S3 input location. It cannot be combined with `--validate_jrc`. |
| `--nghgi_dir <path>` | Directory containing the two compiled raw NGHGI CSVs. |
| `--jrc_dir <path>` | Directory containing `AnnexI_2026/` and optional `BTR1_2024_Table3D.xlsx`. |
| `--no_jrc` | Disable JRC replacement and run raw-only. |
| `--t3d_landuse Cropland Grassland` | Override the model land-use set used for the primary T3D comparison. |
| `--validate_jrc` | Run raw-vs-JRC validation and exit. |
| `--data-only` | Write CSVs and skip figures. |
| `--out-dir-root <path>` | Override the publication output root for this invocation. |
| `--aws_region <region>` | Optional AWS region for S3 reads. |
| `--adm0_lookup_csv <path>` | Optional `gadm_adm0,iso3,country` lookup override. |

## Environment Variables

| Variable | Default |
| --- | --- |
| `AFOLU_PUB_NGHGI_DIR` | `publication_root("nghgi")` |
| `AFOLU_NGHGI_DATA_DIR` | `/mnt/c/GIS/Data/Global/Wetlands/organic_soil_nghgi` |
| `AFOLU_NGHGI_JRC_DIR` | `/mnt/c/GIS/Data/Global/Wetlands/organic_soil_nghgi/JRC_aggregations/2026-04-30` |

`publication_root("nghgi")` is resolved by
`src/scripts/utilities/local_output_paths.py`, usually under `C:/tmp/afolu`
on Windows.

## Known Caveats

### Cultivated grassland is not a clean model class

Including all modeled Grassland in Table 3.D.1.f usually overstates the
model side because natural or protected grassland organic soils can be outside
countries' cultivated/agricultural-management definitions. The Cropland-only
primary and Cropland+Grassland sensitivity are intended to bracket this
classification issue rather than hide it.

### Total peat extent and country reporting scope can diverge sharply

Some countries report a narrow inventory subset of organic soils relative to
the model's mapped peat extent. Canada is the most visible example: sparse
reported organic-soil area in Tables 4.A-4.F should be interpreted as a CRT
taxonomy/reporting-scope issue, not automatically as a model extent failure.

### Remaining negative undrained flags are diagnostics

The mixed-cycle artifact from subtracting raw drained area from JRC totals has
been removed. Any remaining `nghgi_area_undrained_negative_flag = True` rows
come from the raw-basis subtraction itself and should be inspected as inventory
or extraction consistency issues.

### Arctic mask limitations

The current OGH peat-probability raster excludes Greenland and areas north of
about 76 degrees N. Model-side country totals near the high Arctic can
therefore be zero or incomplete even when an inventory reports some area.

## Maintenance Checklist

When the comparison method changes, update this README together with:

1. `pub_nghgi.py` constants and CLI help.
2. Regression tests under `tests/test_pub_nghgi_missing_cstock.py`.
3. Any downstream slide/deck notes that describe T3D scope or N2O GWP.
4. The JRC drop README if a new dated aggregation folder becomes the default.
