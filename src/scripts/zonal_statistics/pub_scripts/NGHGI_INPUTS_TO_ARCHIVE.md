NGHGI benchmark inputs to archive

This file lists the benchmark inputs required to make the NGHGI comparison auditable. It does not contain the data themselves.

Archive the exact inputs used to generate the NGHGI comparison figures and CSV outputs:

organic_soil_compiled.csv
Source: compiled raw UNFCCC CRT Table 4(II) organic-soil records.
Required fields include country or ISO3, year, category code, soil type, drained organic-soil area, CO2, N2O, CH4, and any original notation-key fields retained by the compiler.
organic_soil_cstock_compiled.csv
Source: compiled raw UNFCCC land-use Tables 4.A--4.F.
Required fields include country or ISO3, year, category code, organic-soil area, organic-soil carbon-stock change, and any original notation-key fields retained by the compiler.
JRC-derived land-use table used by extract_organic_soil_jrc.load_jrc_landuse_tables
Source: JRC Annex I 2026 Table 4.A--4.F aggregation files.
Required fields include ISO3, year, land use, total area, organic-soil area, organic-soil carbon-stock change, and source label.
JRC/BTR-derived Table 3.D.1.f table used by extract_organic_soil_jrc.load_jrc_table_3d
Source: JRC Annex I 2026 Table 3.D.1.f and BTR1 2024 Table 3.D files where used.
Required fields include ISO3, year, area, N2O emissions, and source label.
Provenance README
For each archived file, record the original source URL or data portal, download or extraction date, inventory submission cycle, inventory table, and any filtering or notation-key handling applied before the data entered pub_nghgi.py.

The manuscript should cite the archived data package or derived benchmark-input archive before journal submission.
