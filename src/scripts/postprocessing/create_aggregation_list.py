pixel_resolution = '{pixel_resolution}'
output_date = '20250605'

inventory_periods = [
    '2000_2005',
    '2005_2010',
    '2010_2015',
    '2015_2020',
    '2020_2023'
]

data_types = [
    'burned_ch4_co2e',
    'burned_co2',
    'burned_co_co2e',
    'burned_state',
    'burned_total_co2e',
    'drained_ch4_ditch_co2e',
    'drained_ch4_land_co2e',
    'drained_co2',
    'drained_co2_offsite',
    'drained_n2o_co2e',
    'drained_total_co2e',
    'soil',
    'state'
]

base_url = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_3_5"

paths = []

for period in inventory_periods:
    for data_type in data_types:
        path = f'f"{base_url}/{data_type}/ogh_standard_model/five_years_intervals/{period}/{pixel_resolution}/{output_date}",'
        paths.append(path)

# Complete list of paths with f-string, double quotes, and commas
for path in paths:
    print(path)
