# ## Part 5: Calculates combined gross fluxes and net fluxes.
# ## Useful for QC-- to see if there are any egregiously incorrect or unexpected values.
# ## Doing this outside numba function to minimize pixel-level calculations and chunks being returned by numba function.
#
# lu.print_and_log(f"Summing derivative outputs in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)
#
# for interval_end_year in interval_end_years:
#
#     year_range = f"{interval_end_year - interval_year_diff}_{interval_end_year}"
#
#     # Gross emissions across all carbon pools
#     out_dict_all_dtypes[f"{cn.gross_emis_all_C_pools_CO2_only_pattern}_{year_range}"] = (
#             out_dict_all_dtypes[f"{cn.agc_gross_emis_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.bgc_gross_emis_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.deadwood_c_gross_emis_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.litter_c_gross_emis_pattern}_{year_range}"])
#
#     # Gross emissions for non-CO2 emissions
#     out_dict_all_dtypes[f"{cn.gross_emis_all_C_pools_non_CO2_only_pattern}_{year_range}"] = (
#             out_dict_all_dtypes[f"{cn.ch4_flux_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.n2o_flux_pattern}_{year_range}"])
#
#     # Gross emissions for all carbon pools and all gases
#     out_dict_all_dtypes[f"{cn.gross_emis_all_C_pools_all_gases_pattern}_{year_range}"] = (
#         out_dict_all_dtypes[f"{cn.gross_emis_all_C_pools_CO2_only_pattern}_{year_range}"]
#         + out_dict_all_dtypes[f"{cn.gross_emis_all_C_pools_non_CO2_only_pattern}_{year_range}"]
#     )
#
#     # Gross removals across all carbon pools
#     out_dict_all_dtypes[f"{cn.gross_removals_all_C_pools_pattern}_{year_range}"] = (
#             out_dict_all_dtypes[f"{cn.agc_gross_removals_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.bgc_gross_removals_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.deadwood_c_gross_removals_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.litter_c_gross_removals_pattern}_{year_range}"])
#
#     # Net flux for each carbon pool
#     out_dict_all_dtypes[f"{cn.agc_net_flux_pattern}_{year_range}"] = out_dict_all_dtypes[f"{cn.agc_gross_emis_pattern}_{year_range}"] + out_dict_all_dtypes[f"{cn.agc_gross_removals_pattern}_{year_range}"]
#     out_dict_all_dtypes[f"{cn.bgc_net_flux_pattern}_{year_range}"] = out_dict_all_dtypes[f"{cn.bgc_gross_emis_pattern}_{year_range}"] + out_dict_all_dtypes[f"{cn.bgc_gross_removals_pattern}_{year_range}"]
#     out_dict_all_dtypes[f"{cn.deadwood_c_net_flux_pattern}_{year_range}"] = out_dict_all_dtypes[f"{cn.deadwood_c_gross_emis_pattern}_{year_range}"] + out_dict_all_dtypes[f"{cn.deadwood_c_gross_removals_pattern}_{year_range}"]
#     out_dict_all_dtypes[f"{cn.litter_c_net_flux_pattern}_{year_range}"] = out_dict_all_dtypes[f"{cn.litter_c_gross_emis_pattern}_{year_range}"] + out_dict_all_dtypes[f"{cn.litter_c_gross_removals_pattern}_{year_range}"]
#
#     # Net flux across all carbon pools but for CO2 only
#     out_dict_all_dtypes[f"{cn.net_flux_all_C_pools_CO2_only_pattern}_{year_range}"] = (
#             out_dict_all_dtypes[f"{cn.agc_net_flux_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.bgc_net_flux_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.deadwood_c_net_flux_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.litter_c_net_flux_pattern}_{year_range}"])
#
#     # Net flux across all carbon pools, plus non-pool non-CO2 emissions
#     out_dict_all_dtypes[f"{cn.net_flux_all_C_pools_all_gases_pattern}_{year_range}"] = (
#             out_dict_all_dtypes[f"{cn.net_flux_all_C_pools_CO2_only_pattern}_{year_range}"]
#             + out_dict_all_dtypes[f"{cn.gross_emis_all_C_pools_non_CO2_only_pattern}_{year_range}"])
#
# lu.print_and_log(f"Done summing derivative outputs in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)
# print(f"After creating summative outputs for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB")
