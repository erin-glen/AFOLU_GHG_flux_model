"""
Global:
python -m src.synthesis.scripts.create_sector_level_0_04deg_global_display_maps
-veg /mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/vegetation/v1_0_5__standard__global/net_flux__all_C_pools__all_gases__MgCO2e_0_04deg_yr_v1_0_5_2016_2024_mean_global_reproj.tif
-os /mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/LULUCF_totals/0_01deg_global__drained_burned_total_Mg_CO2e_pixel_yr_2021_2024.tif
-cl s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/raw__from_Cornell/20250828/year_2020/all_sources/Global_grid_all_GHGs_cropland_total_amount_CO2eq_all_crops_NonPeatland_2019_kg_CO2.tif
-ls s3://gfw2-data/climate/AFOLU_flux_model/livestock_emissions/raw__from_Cornell/20251223/Total_GHG_Emissions/Tot_CO2eq_kg_livestock_GHG_emissions.tif

For Central Africa:
python -m src.LULUCF.scripts.vegetation_model.create_sector_level_0_04deg_global_display_maps--center_latitude 0 --center_longitude 20 --lat_height 20 -bbd central_Africa

Run locally (not in Coiled)

A zoomed in map can be created by supplying central lat-long arguments, as well as a north-south extent for the map to include.
The aspect ratio used in the global map of 2:1 (width:height) is maintained, and the east-west extent is determined
from that information. That keeps all zoomed in maps in the same shape as the global map for simplicity.

With https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67634e63-bbcc-800a-8267-004e88ced2e4
Continued at https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68d6d26f-b054-8323-98bb-731a86582e74
This specific code at https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69778c22-2538-8325-a70e-1a2b70312505
"""

import argparse
from pathlib import Path
import time
import os
import rasterio
import math
import numpy as np
import re
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from shapely.geometry import Polygon, MultiPolygon, box, mapping
from scipy.stats import percentileofscore
from pyproj import Transformer

from src.utilities import constants_and_names as cn
from src.utilities import map_utilities as mu
from src.utilities import universal_utilities as uu
from src.utilities import log_utilities as lu

# Reprojects global geotifs to the projection/extent/resolution that the vegetation geotifs use, if not already reprojected
def reproject_to_vegetation(geotif_to_reproj, local_reproj_folder, net_all_gases_geotif_local):

    # Extracts the file name with extension, then removes extension
    filename_with_ext = os.path.basename(geotif_to_reproj)
    filename = os.path.splitext(filename_with_ext)[0]

    path_unproj = geotif_to_reproj
    path_reproj = f"{local_reproj_folder}/{filename}_reproj.tif"

    if not os.path.exists(path_reproj):

        print("  Reprojected raster does not exist. Reprojecting now...")
        print(f"   Unprojected raster: {path_unproj}")
        print(f"   Reprojected raster: {path_reproj}")

        # Opens reference raster to extract desired CRS, transform, and shape
        with rasterio.open(net_all_gases_geotif_local) as ref:
            dst_crs = ref.crs
            dst_transform = ref.transform
            dst_width = ref.width
            dst_height = ref.height

        # Opens source raster to reproject
        with rasterio.open(path_unproj) as src:

            # Prepare output metadata
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height,
                'nodata': 0,
                'compress': 'lzw'
            })

            # Reprojects and write to file
            with rasterio.open(path_reproj, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest  # or bilinear/cubic as needed
                    )

    else:
        print("  Reprojected raster already exists")

    return path_reproj

# Converts geotif from kg to megagrams (tonnes)
def convert_kg_to_Mg(path_reproj):

    # Unit-converted raster
    converted_path = path_reproj.replace("kg", "Mg")

    if not os.path.exists(converted_path):

        print("  Unit-converted raster does not exist. Converting kg to Mg...")

        with rasterio.open(path_reproj) as src:
            data = src.read(1)
            meta = src.meta.copy()
            nodata = src.nodata

            # Masks nodata values (e.g., 0) to avoid dividing them
            data = np.where(data == nodata, nodata, data / 1000.0)

            with rasterio.open(converted_path, 'w', **meta) as dst:
                dst.write(data.astype('float32'), 1)

    else:
        print("  Unit-converted raster already exists")

    return converted_path

# Sums the vegetation net flux and other dataset
def add_veg_and_other_data(output_sum_path, converted_path, net_all_gases_geotif_local):

    with rasterio.open(net_all_gases_geotif_local) as src_a, rasterio.open(converted_path) as src_b:
        data_a = src_a.read(1)
        data_b = src_b.read(1)

        # Add rasters directly — no masking
        data_sum = data_a + data_b

        # Copy metadata from one of the sources (assumed identical)
        meta = src_a.meta.copy()
        meta.update(dtype='float32')

        with rasterio.open(output_sum_path, 'w', **meta) as dst:
            dst.write(data_sum.astype('float32'), 1)

        # All non-zero values (used for calculating legend values)
        non_zero_values = data_sum[data_sum != 0]

    return non_zero_values


def map_AFOLU_totals(net_all_gases_geotif_local,
                     organic_soil_local,
                     mineral_soil_s3,
                     cropland_geotif_s3,
                     livestock_geotif_s3,
                     net_colors_rgb, country_shapefile, bounding_box, bounding_box_description, main_logger):

    start_time = time.time()

    # Folders for local outputs
    LULUCF_reproj_folder = Path(cn.local_jpeg_folder_LULUCF)
    LULUCF_reproj_folder.mkdir(parents=True, exist_ok=True)
    LULUCF_local_jpeg_non_pres_folder = Path(f"{cn.local_jpeg_folder_LULUCF}output_jpegs_and_gifs_{bounding_box_description}/jpegs_non_pres")
    LULUCF_local_jpeg_non_pres_folder.mkdir(parents=True, exist_ok=True)
    LULUCF_local_jpeg_pres_folder = Path(f"{cn.local_jpeg_folder_LULUCF}output_jpegs_and_gifs_{bounding_box_description}/jpegs_pres")
    LULUCF_local_jpeg_pres_folder.mkdir(parents=True, exist_ok=True)
    LULUCF_local_gif_folder = Path(f"{cn.local_jpeg_folder_LULUCF}output_jpegs_and_gifs_{bounding_box_description}/gifs")
    LULUCF_local_gif_folder.mkdir(parents=True, exist_ok=True)

    cropland_reproj_folder = Path(cn.local_jpeg_folder_cropland)
    cropland_reproj_folder.mkdir(parents=True, exist_ok=True)
    livestock_reproj_folder = Path(cn.local_jpeg_folder_livestock)
    livestock_reproj_folder.mkdir(parents=True, exist_ok=True)

    AFOLU_local_jpeg_non_pres_folder = Path(f"{cn.local_jpeg_folder_AFOLU}output_jpegs_and_gifs_{bounding_box_description}/jpegs_non_pres")
    AFOLU_local_jpeg_non_pres_folder.mkdir(parents=True, exist_ok=True)
    AFOLU_local_jpeg_pres_folder = Path(f"{cn.local_jpeg_folder_AFOLU}output_jpegs_and_gifs_{bounding_box_description}/jpegs_pres")
    AFOLU_local_jpeg_pres_folder.mkdir(parents=True, exist_ok=True)
    AFOLU_local_gif_folder = Path(f"{cn.local_jpeg_folder_AFOLU}output_jpegs_and_gifs_{bounding_box_description}/gifs")
    AFOLU_local_gif_folder.mkdir(parents=True, exist_ok=True)

    out_maps_for_gif = []

    # If bounding_box was given in degrees, transforms to match the raster CRS (Robinson)
    if bounding_box is not None:
        bounding_box_proj = mu.transform_bbox_to_robinson(bounding_box)
    else:
        bounding_box_proj = None

    data_to_add = {}

    print(organic_soil_local)

    if organic_soil_local:
        data_to_add['organic_soil'] = [organic_soil_local, LULUCF_reproj_folder, cn.veg_organic_soil_pres_text]

    if mineral_soil_s3:
        data_to_add['mineral_soil'] = [mineral_soil_s3, LULUCF_reproj_folder, cn.veg_mineral_soil_pres_text]

    if cropland_geotif_s3:
        data_to_add['cropland'] = [cropland_geotif_s3, cropland_reproj_folder, cn.veg_cropland_pres_text]

    if livestock_geotif_s3:
        data_to_add['livestock'] = [livestock_geotif_s3, livestock_reproj_folder, cn.veg_livestock_pres_text]

    main_logger.info(f"Vegetation net flux: {net_all_gases_geotif_local}")
    main_logger.info(f"Inputs to add to vegetation: {data_to_add}")

    veg_analysis_years = f"{cn.interval_end_years_annual[0]}_{cn.last_model_year_annual}"

    # Version of the vegetation model being used
    veg_version = re.search(r'v\d+_\d+_\d+', net_all_gases_geotif_local).group(0)

    # Loads base vegetation raster once
    with rasterio.open(net_all_gases_geotif_local) as src_veg:
        veg_meta = src_veg.meta.copy()
        total_across_LULUCF = src_veg.read(1).astype('float32')  # base raster to accumulate into for LULUCF total
        total_across_AFOLU = src_veg.read(1).astype('float32')  # base raster to accumulate into for AFOLU total

    # Iterates through non-vegetation layers to combine them with vegetation individually
    for key, value in data_to_add.items():

        input_s3_path = value[0]
        local_reproj_folder = value[1]
        presentation_slide_text = value[2]

        main_logger.info(f"\n---Reprojecting {key} to vegetation projection")

        # Date of the other dataset being used
        additional_data_date = re.search(r'/(\d{8})/', input_s3_path).group(1)
        value.append(additional_data_date)

        # Reprojects to match vegetation net flux (if not already reprojected)
        path_reproj = reproject_to_vegetation(input_s3_path, local_reproj_folder, net_all_gases_geotif_local)
        # print("path_reproj:", path_reproj)

        # Converts from kg to Mg (if not already converted, like for cropland and livestock)
        unit_converted_path = convert_kg_to_Mg(path_reproj)
        # print("unit_converted_path:", unit_converted_path)

        # Loads unit-converted raster and adds it to the running LULUCF total
        if "soil" in unit_converted_path:
            with rasterio.open(unit_converted_path) as src:
                data = src.read(1).astype('float32')
                total_across_LULUCF += data

        # Loads unit-converted raster and adds it to the running AFOLU total
        with rasterio.open(unit_converted_path) as src:
            data = src.read(1).astype('float32')
            total_across_AFOLU += data


        main_logger.info(f"Combining vegetation net flux and {key} emissions")

        output_name = f"vegetation_net_flux_all_pools_all_gases_{veg_version}__{key}_{additional_data_date}__{veg_analysis_years}__Mg_CO2e"
        output_sum_path = f"{cn.local_jpeg_folder_AFOLU}/{output_name}.tif"

        # Sums the vegetation net flux and other data
        non_zero_values = add_veg_and_other_data(output_sum_path, unit_converted_path, net_all_gases_geotif_local)


        main_logger.info(f"\n\n---Preparing legend")

        # Calculates min, center and max across all years
        percentile_for_saturation = 1
        breaks_all_yrs = np.percentile(non_zero_values, [1, (100-percentile_for_saturation)])  # The min and max percentiles at which colors saturate

        lower_lim_all_yrs = breaks_all_yrs[0]
        global_neutral = 0
        upper_lim_all_yrs = breaks_all_yrs[-1]

        main_logger.info(f"Across vegetation+{key}:")
        main_logger.info(f"  lower limit ({percentile_for_saturation} percentile):", lower_lim_all_yrs)
        main_logger.info(f"  neutral:", global_neutral)
        main_logger.info(f"  upper limit ({(100-percentile_for_saturation)} percentile):", upper_lim_all_yrs)

        # Creates the min and max values for the legend in kt CO2e (converts legend units from Mg (t) to kt with 10**3-- data doesn't change).
        # Rounds data_min down and data_max up for legend.
        rounded_lower_lim_all_yrs = math.ceil(lower_lim_all_yrs / 10 ** 3 * 100) / 100  # Rounds up
        rounded_upper_lim_all_yrs = math.floor(upper_lim_all_yrs / 10 ** 3 * 100) / 100  # Rounds down
        tick_labels = [f"< {rounded_lower_lim_all_yrs:.0f}  (sink)",  # Spaces are to horizontally align the text explanations
                       "0        (neutral)",
                       f"> {rounded_upper_lim_all_yrs:.0f}  (source)"]
        # print(tick_labels)


        main_logger.info(f"\n\n---Mapping vegetation + {key}")

        # Reads raster data for year
        with rasterio.open(output_sum_path) as src:

            if bounding_box_proj is not None:
                minx, miny, maxx, maxy = bounding_box_proj

                window = from_bounds(minx, miny, maxx, maxy, src.transform)

                data = src.read(1, window=window)

                # Update extent from the window
                left, bottom, right, top = rasterio.windows.bounds(window, src.transform)
                raster_extent = (left, right, bottom, top)

            else:
                data = src.read(1)
                b = src.bounds
                raster_extent = (b.left, b.right, b.bottom, b.top)


        # Calculates the percentile for 0 for the year (neutral, no flux) for mapping
        main_logger.info(f"  Calculating percentiles and breaks")
        percentile_0 = mu.percentile_for_0(data)
        main_logger.info(f"  0 is at the {percentile_0}th percentile of the raster.")
        percentiles = [percentile_0 / 6, percentile_0 / 4, percentile_0 / 2, percentile_0 / 1.3, percentile_0 / 1.05,
                       percentile_0 * 1.05, percentile_0 * 1.1, percentile_0 * 1.2, percentile_0 * 1.3, percentile_0 * 1.5]
        # print("percentiles:", percentiles)

        # Converts RGB color palette to matplotlib color palette
        colors_matplotlib = mu.rgb_to_mpl_palette(net_colors_rgb)

        # Matches percentile breaks with colors for the map.
        # Normalizes percentiles to a 0-1 scale.
        percentiles_normalized = np.linspace(0, 1, len(percentiles))
        # print("percentiles_normalized:", percentiles_normalized)
        cmap = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized, colors_matplotlib)))

        main_logger.info(f"  Masking raster to non-0 values")
        masked_data = np.ma.masked_where(data == 0, data)

        # For map (not legend)
        norm = TwoSlopeNorm(
            vmin=lower_lim_all_yrs,
            vcenter=global_neutral,
            vmax=upper_lim_all_yrs
        )

        main_logger.info(f"  Plotting map ")
        ax, fig = mu.create_plot()

        # Sets the ocean color
        mu.set_ocean_color(ax)

        # Limits shapefile to focal extent (if requested)
        if bounding_box_proj is not None:
            bbox_geom = box(*bounding_box_proj)
            country_shapefile = country_shapefile.clip(bbox_geom)

        # Plots the country polygons first
        mu.plot_country_polygons(ax, country_shapefile)

        # Raster extent
        extent = list(raster_extent)

        # Plots the raster next
        img = mu.plot_raster(ax, cmap, extent, masked_data, norm)

        # Plots the country boundaries on top
        mu.plot_country_boundaries(ax, country_shapefile)

        # Explicitly sets the bounding box for the plot image
        if bounding_box_proj is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

        # Title
        title_text = f"Vegetation and {key}\nkt CO$_2$e in {veg_analysis_years}"

        # Creates legend
        mu.create_divergent_legend_asymmetric(fig, rounded_lower_lim_all_yrs, rounded_upper_lim_all_yrs,
                                           title_text, tick_labels,
                                              veg_analysis_years, net_colors_rgb, percentiles, percentile_0)

        # Removes axis ticks and labels
        mu.remove_ticks(ax)

        core_jpeg_name = f"{output_name}__{uu.timestr()[0:8]}"
        if bounding_box_description:  # Adds bounding box description to file name, if supplied
            core_jpeg_name = f"{core_jpeg_name}_{bounding_box_description}"
        jpeg_path = f"{AFOLU_local_jpeg_non_pres_folder}/{core_jpeg_name}.jpeg"
        jpeg_for_pres_path = f"{AFOLU_local_jpeg_pres_folder}/{core_jpeg_name}__for_pres.jpeg"

        # Saves two versions of the map: without and with a source note in the bottom right
        veg_addtl_pres_text = presentation_slide_text.replace("YYYYMMDD", additional_data_date)
        out_jpeg_for_pres = mu.save_pres_non_pres_jpegs(ax, jpeg_path, jpeg_for_pres_path, "", veg_addtl_pres_text)

        end_time = time.time()
        main_logger.info(f"vegetation+{key} {bounding_box_description} took {round(end_time - start_time)} seconds: {uu.timestr()}")



    # print("\n\n---Mapping vegetation + the following subsectors:")
    # print(", ".join(data_to_add.keys()))
    #
    # # Iteratively collects that names of datasets and dates for the non-vegetation data
    # non_veg_dates = ''
    # for key, values in data_to_add.items():
    #     non_veg_dates = f"{non_veg_dates}_{key}_{values[3]}"
    #
    # # Final combined output
    # output_name = f"vegetation_net_flux_all_pools_all_gases_{veg_version}_{non_veg_dates}__{veg_analysis_years}__Mg_CO2e"
    # final_total_path = f"{cn.local_jpeg_folder_AFOLU}/{output_name}.tif"
    # with rasterio.open(final_total_path, 'w', **veg_meta) as dst:
    #     dst.write(total_across_AFOLU.astype('float32'), 1)
    #
    # non_zero_values = total_across_AFOLU[total_across_AFOLU != 0]
    #
    # print(f"\n\n---Preparing legend")
    #
    # # Calculates min, center and max across all years
    # percentile_for_saturation = 1
    # breaks_all_yrs = np.percentile(non_zero_values, [1, (100 - percentile_for_saturation)])  # The min and max percentiles at which colors saturate
    #
    # lower_lim_all_yrs = breaks_all_yrs[0]
    # global_neutral = 0
    # upper_lim_all_yrs = breaks_all_yrs[-1]
    #
    # print(f"Across vegetation+{non_veg_dates}:")
    # print(f"  lower limit ({percentile_for_saturation} percentile):", lower_lim_all_yrs)
    # print(f"  neutral:", global_neutral)
    # print(f"  upper limit ({(100 - percentile_for_saturation)} percentile):", upper_lim_all_yrs)
    #
    # # Creates the min and max values for the legend in kt CO2e (converts legend units from Mg (t) to kt with 10**3-- data doesn't change).
    # # Rounds data_min down and data_max up for legend.
    # rounded_lower_lim_all_yrs = math.ceil(lower_lim_all_yrs / 10 ** 3 * 100) / 100  # Rounds up
    # rounded_upper_lim_all_yrs = math.floor(upper_lim_all_yrs / 10 ** 3 * 100) / 100  # Rounds down
    # tick_labels = [f"< {rounded_lower_lim_all_yrs:.0f}  (sink)",
    #                # Spaces are to horizontally align the text explanations
    #                "0        (neutral)",
    #                f"> {rounded_upper_lim_all_yrs:.0f}  (source)"]
    # # print(tick_labels)
    #
    # print(f"\n\n---Mapping vegetation + {non_veg_dates}")
    #
    # # Reads raster data
    # with rasterio.open(final_total_path) as src:
    #
    #     if bounding_box_proj is not None:
    #         minx, miny, maxx, maxy = bounding_box_proj
    #
    #         window = from_bounds(minx, miny, maxx, maxy, src.transform)
    #
    #         data = src.read(1, window=window)
    #
    #         # Update extent from the window
    #         left, bottom, right, top = rasterio.windows.bounds(window, src.transform)
    #         raster_extent = (left, right, bottom, top)
    #
    #     else:
    #         data = src.read(1)
    #         b = src.bounds
    #         raster_extent = (b.left, b.right, b.bottom, b.top)
    #
    # # Calculates the percentile for 0 for the year (neutral, no flux) for mapping
    # print(f"  Calculating percentiles and breaks")
    # percentile_0 = mu.percentile_for_0(data)
    # print(f"  0 is at the {percentile_0}th percentile of the raster.")
    # percentiles = [percentile_0 / 6, percentile_0 / 4, percentile_0 / 2, percentile_0 / 1.3, percentile_0 / 1.05,
    #                percentile_0 * 1.05, percentile_0 * 1.1, percentile_0 * 1.2, percentile_0 * 1.3, percentile_0 * 1.5]
    # # print("percentiles:", percentiles)
    #
    # # Converts RGB color palette to matplotlib color palette
    # colors_matplotlib = mu.rgb_to_mpl_palette(net_colors_rgb)
    #
    # # Matches percentile breaks with colors for the map.
    # # Normalizes percentiles to a 0-1 scale.
    # percentiles_normalized = np.linspace(0, 1, len(percentiles))
    # # print("percentiles_normalized:", percentiles_normalized)
    # cmap = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized, colors_matplotlib)))
    #
    # print(f"  Masking raster to non-0 values")
    # masked_data = np.ma.masked_where(data == 0, data)
    #
    # # For map (not legend)
    # norm = TwoSlopeNorm(
    #     vmin=lower_lim_all_yrs,
    #     vcenter=global_neutral,
    #     vmax=upper_lim_all_yrs
    # )
    #
    # print(f"  Plotting map ")
    # ax, fig = mu.create_plot()
    #
    # # Sets the ocean color
    # mu.set_ocean_color(ax)
    #
    # # Limits shapefile to focal extent (if requested)
    # if bounding_box_proj is not None:
    #     bbox_geom = box(*bounding_box_proj)
    #     country_shapefile = country_shapefile.clip(bbox_geom)
    #
    # # Plots the country polygons first
    # mu.plot_country_polygons(ax, country_shapefile)
    #
    # # Raster extent
    # extent = list(raster_extent)
    #
    # # Plots the raster next
    # img = mu.plot_raster(ax, cmap, extent, masked_data, norm)
    #
    # # Plots the country boundaries on top
    # mu.plot_country_boundaries(ax, country_shapefile)
    #
    # # Explicitly sets the bounding box for the plot image
    # if bounding_box_proj is not None:
    #     ax.set_xlim(extent[0], extent[1])
    #     ax.set_ylim(extent[2], extent[3])
    #
    # # Title
    # title_text = f"Vegetation, cropland, and livestock\nkt CO$_2$e in {veg_analysis_years}"
    #
    # # Creates legend
    # mu.create_divergent_legend_asymmetric(fig, rounded_lower_lim_all_yrs, rounded_upper_lim_all_yrs,
    #                                       title_text, tick_labels,
    #                                       veg_analysis_years, net_colors_rgb, percentiles, percentile_0)
    #
    # # Removes axis ticks and labels
    # mu.remove_ticks(ax)
    #
    # core_jpeg_name = f"{output_name}__{uu.timestr()[0:8]}"
    # if bounding_box_description:  # Adds bounding box description to file name, if supplied
    #     core_jpeg_name = f"{core_jpeg_name}_{bounding_box_description}"
    # jpeg_path = f"{local_jpeg_non_pres_folder}/{core_jpeg_name}.jpeg"
    # jpeg_for_pres_path = f"{local_jpeg_pres_folder}/{core_jpeg_name}__for_pres.jpeg"
    #
    # # Saves two versions of the map: without and with a source note in the bottom right
    # veg_addtl_pres_text = cn.veg_cropland_livestock_pres_text.replace("cropland: YYYYMMDD", f"cropland: {data_to_add['cropland'][3]}")
    # veg_addtl_pres_text = veg_addtl_pres_text.replace("livestock: YYYYMMDD", f"livestock: {data_to_add['livestock'][3]}")
    # out_jpeg_for_pres = mu.save_pres_non_pres_jpegs(ax, jpeg_path, jpeg_for_pres_path, "", veg_addtl_pres_text)
    #
    # end_time = time.time()
    # print(f"vegetation+{data_to_add} {bounding_box_description} took {round(end_time - start_time)} seconds: {uu.timestr()}")


def main(net_all_gases_geotif_local,
         organic_soil_local=None,
         mineral_soil_s3=None,
         cropland_geotif_s3=None,
         livestock_geotif_s3=None,
         center_latitude=None, center_longitude=None, lat_height=None, bounding_box_description=None):

    # Model stage being run
    stage = 'summative_4x4km_maps'
    log_note = '4x4 km maps for various AFOLU components'

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header("NA", "NA", log_note, True, "NA", stage)

    # Reprojects simplified country boundary shapefile, if needed
    country_shapefile = mu.check_and_reproject_shapefile(
        shapefile_path=cn.original_shapefile_path,
        target_crs=cn.Robinson_crs,
        reprojected_shapefile_path=cn.reprojected_shapefile_path
    )

    # Creates bounding box in degrees from given map center and desired latitude range (optional)
    if center_latitude is not None and center_longitude is not None and lat_height is not None:
        bounding_box = mu.calculate_bbox_centered(
            center_lat=center_latitude,
            center_lon=center_longitude,
            lat_height=lat_height,
            aspect_ratio=2.0  # panel_dims = (12, 6), same as global map for simplicity
        )
        print(f"Using custom bounding box: {bounding_box}")
    else:
        bounding_box = None
        print("No bounding box specified; using global extent.")

    # Generates jpegs for net flux, gross emissions, and gross removals
    map_AFOLU_totals(net_all_gases_geotif_local,
                     organic_soil_local,
                     mineral_soil_s3,
                     cropland_geotif_s3,
                     livestock_geotif_s3,
                     cn.net_colors_rgb, country_shapefile, bounding_box, bounding_box_description, main_logger)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create jpegs of 0.04x0.04 deg output maps.")
    parser.add_argument('-clat', '--center_latitude', type=float, help='Latitude to center output maps (optional)')
    parser.add_argument('-clon', '--center_longitude', type=float, help='Longitude to center output maps (optional)')
    parser.add_argument('-lh', '--lat_height', type=float, help='Latitude to show around lat center (value is total north/south) (optional)')
    parser.add_argument('-bbd', '--bounding_box_description', default='global', help='Description of bounding box (if used) to include in output names.')

    parser.add_argument('-veg', '--net_all_gases_geotif_local', help='Local vegetation net flux file to use')
    parser.add_argument('-os', '--organic_soil_local', help='Local organic soil emissions file (eventually should come from s3 but added drained and burned together locally')
    parser.add_argument('-ms', '--mineral_soil_s3', help='s3 path for mineral soil net flux')
    parser.add_argument('-cl', '--cropland_geotif_s3', help='s3 path for cropland management emissions')
    parser.add_argument('-ls', '--livestock_geotif_s3', help='s3 path for livestock emissions')


    args = parser.parse_args()
    center_latitude = args.center_latitude
    center_longitude = args.center_longitude
    lat_height = args.lat_height
    bounding_box_description = args.bounding_box_description

    net_all_gases_geotif_local = args.net_all_gases_geotif_local
    organic_soil_local = args.organic_soil_local
    mineral_soil_s3 = args.mineral_soil_s3
    cropland_geotif_s3 = args.cropland_geotif_s3
    livestock_geotif_s3 = args.livestock_geotif_s3

    main(net_all_gases_geotif_local,
         organic_soil_local=organic_soil_local,  # Created by downloading and summing drainage and burning. Erin hasn't made a combined map yet.
         mineral_soil_s3=mineral_soil_s3,
         cropland_geotif_s3=cropland_geotif_s3,
         livestock_geotif_s3=livestock_geotif_s3,
         center_latitude=center_latitude, center_longitude=center_longitude, lat_height=lat_height, bounding_box_description=bounding_box_description)

