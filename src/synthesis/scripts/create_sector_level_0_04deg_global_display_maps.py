"""
Global:
ython -m src.synthesis.scripts.create_sector_level_0_04deg_global_display_maps -mpd global
--input_vegetation_date 20251224
-cl s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/raw__from_Cornell/20250828/year_2020/all_sources/Global_grid_all_GHGs_cropland_total_amount_CO2eq_all_crops_NonPeatland_2019_kg_CO2.tif
-ls s3://gfw2-data/climate/AFOLU_flux_model/livestock_emissions/raw__from_Cornell/20251223/Total_GHG_Emissions/Tot_CO2eq_kg_livestock_GHG_emissions.tif
-veg /mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/v1_0_4__standard__global/net_flux__all_C_pools__all_gases__MgCO2e_0_04deg_yr_v1_0_4_2020_global_reproj.tif

For Central Africa:
python -m src.LULUCF.scripts.vegetation_model.create_sector_level_0_04deg_global_display_maps -mt standard -mpd global --input_date YYYYMMDD --center_latitude 0 --center_longitude 20 --lat_height 20 -bbd central_Africa

Run locally (not in Coiled)

A zoomed in map can be created by supplying central lat-long arguments, as well as a north-south extent for the map to include.
The aspect ratio used in the global map of 2:1 (width:height) is maintained, and the east-west extent is determined
from that information. That keeps all zoomed in maps in the same shape as the global map for simplicity.

With https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67634e63-bbcc-800a-8267-004e88ced2e4
Continued at https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68d6d26f-b054-8323-98bb-731a86582e74
"""

import argparse
from pathlib import Path
import time
import os
import rasterio
import math
import numpy as np
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from shapely.geometry import Polygon, MultiPolygon, box, mapping
from scipy.stats import percentileofscore
from pyproj import Transformer

from src.utilities import constants_and_names as cn
from src.utilities import map_utilities as mu
from src.utilities import universal_utilities as uu

def map_AFOLU_totals(net_all_gases_geotif_local, cropland_geotif_s3, livestock_geotif_s3, local_reproj_folder,
                 local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
                 net_colors_rgb, country_shapefile, bounding_box, bounding_box_description):

    series_start_time = time.time()

    out_maps_for_gif = []

    # If bounding_box was given in degrees, transforms to match the raster CRS (Robinson)
    if bounding_box is not None:
        bounding_box_proj = mu.transform_bbox_to_robinson(bounding_box)
    else:
        bounding_box_proj = None

    all_valid_values = []

    print(f"\n\n---Reprojecting cropland for 2020")

    # Extract the file name with extension
    filename_with_ext = os.path.basename(cropland_geotif_s3)

    # Removes the extension
    filename = os.path.splitext(filename_with_ext)[0]

    path_unproj = cropland_geotif_s3
    path_reproj = f"{local_reproj_folder}/{filename}_reproj.tif"

    print(f"Unprojected raster: {path_unproj}")
    print(f"Reprojected raster: {path_reproj}")

    # Reprojects cropland raster to match the vegetation net flux raster, if not already reprojected
    # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69778c22-2538-8325-a70e-1a2b70312505
    if not os.path.exists(path_reproj):
        print("  Reprojected raster does not exist. Reprojecting now...")

        # Input file paths
        src_path = path_unproj  # File you want to reproject
        ref_path = net_all_gases_geotif_local  # File whose projection and grid you want to match
        dst_path = path_reproj  # File to write the reprojected result

        # Open reference raster to extract desired CRS, transform, and shape
        with rasterio.open(ref_path) as ref:
            dst_crs = ref.crs
            dst_transform = ref.transform
            dst_width = ref.width
            dst_height = ref.height

        # Open source raster to reproject
        with rasterio.open(src_path) as src:

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

            # Reproject and write to file
            with rasterio.open(dst_path, 'w', **kwargs) as dst:
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


    # Unit-converted raster
    converted_path = path_reproj.replace("kg_CO2_reproj.tif", "Mg_CO2e_reproj.tif")

    if not os.path.exists(converted_path):
        print("  Unit-converted raster does not exist. Converting kg to Mg now...")

        with rasterio.open(path_reproj) as src:
            data = src.read(1)
            meta = src.meta.copy()
            nodata = src.nodata

            # Mask nodata values (e.g., 0) to avoid dividing them
            data = np.where(data == nodata, nodata, data / 1000.0)

            with rasterio.open(converted_path, 'w', **meta) as dst:
                dst.write(data.astype('float32'), 1)


    print("Combining vegetation net flux and cropland emissions")

    # Inputs
    raster_a_path = net_all_gases_geotif_local  # e.g. vegetation net flux raster (already reprojected)
    raster_b_path = converted_path  # e.g. cropland raster reprojected and divided by 1000
    output_sum_path = f"{cn.local_jpeg_folder_AFOLU}/veg_cropland_combined.tif"

    with rasterio.open(raster_a_path) as src_a, rasterio.open(raster_b_path) as src_b:
        data_a = src_a.read(1)
        data_b = src_b.read(1)

        # Add rasters directly — no masking
        data_sum = data_a + data_b

        # Copy metadata from one of the sources (assumed identical)
        meta = src_a.meta.copy()
        meta.update(dtype='float32')

        with rasterio.open(output_sum_path, 'w', **meta) as dst:
            dst.write(data_sum.astype('float32'), 1)

        valid = data_sum[data_sum != 0]
        if valid.size > 0:
            all_valid_values.append(valid)

    # Calculates min, center and max across all years
    all_valid_values = np.concatenate(all_valid_values)

    percentile_for_saturation = 1
    breaks_all_yrs = np.percentile(all_valid_values, [1, (100-percentile_for_saturation)])  # The min and max percentiles at which colors saturate

    lower_lim_all_yrs = breaks_all_yrs[0]
    global_neutral = 0
    upper_lim_all_yrs = breaks_all_yrs[-1]

    print("Across vegetation+cropland:")
    print(f"  lower limit ({percentile_for_saturation} percentile):", lower_lim_all_yrs)
    print(f"  neutral:", global_neutral)
    print(f"  upper limit ({(100-percentile_for_saturation)} percentile):", upper_lim_all_yrs)

    # Creates the min and max values for the legend in kt CO2e (converts legend units from Mg (t) to kt with 10**3-- data doesn't change).
    # Rounds data_min down and data_max up for legend.
    rounded_lower_lim_all_yrs = math.ceil(lower_lim_all_yrs / 10 ** 3 * 100) / 100  # Rounds up
    rounded_upper_lim_all_yrs = math.floor(upper_lim_all_yrs / 10 ** 3 * 100) / 100  # Rounds down
    tick_labels = [f"< {rounded_lower_lim_all_yrs:.0f}  (sink)",  # Spaces are to horizontally align the text explanations
                   "0        (neutral)",
                   f"> {rounded_upper_lim_all_yrs:.0f}  (source)"]
    print(tick_labels)



    print(f"\n\n---Mapping cropland+vegetation")

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
    percentile_0 = mu.percentile_for_0(data)
    print(f"  0 is at the {percentile_0}th percentile of the raster.")
    percentiles = [percentile_0 / 6, percentile_0 / 4, percentile_0 / 2, percentile_0 / 1.3, percentile_0 / 1.05,
                   percentile_0 * 1.05, percentile_0 * 1.1, percentile_0 * 1.2, percentile_0 * 1.3, percentile_0 * 1.5]
    # print("percentiles:", percentiles)

    print(f"  Calculating percentiles and breaks")

    # Converts RGB color palette to matplotlib color palette
    colors_matplotlib = mu.rgb_to_mpl_palette(net_colors_rgb)

    # Matches percentile breaks with colors for the map.
    # Normalizes percentiles to a 0-1 scale.
    percentiles_normalized = np.linspace(0, 1, len(percentiles))
    # print("percentiles_normalized:", percentiles_normalized)
    cmap = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized, colors_matplotlib)))

    print(f"  Masking raster to non-0 values")
    masked_data = np.ma.masked_where(data == 0, data)

    # For map (not legend)
    norm = TwoSlopeNorm(
        vmin=lower_lim_all_yrs,
        vcenter=global_neutral,
        vmax=upper_lim_all_yrs
    )

    print(f"  Plotting map ")
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
    title_text = f"Net greenhouse gas flux vegetation+cropland\nAll vegetation pools, all gases\nkt CO$_2$e yr$^{{-1}}$"

    # Creates legend
    mu.create_divergent_legend_asymmetric(fig, rounded_lower_lim_all_yrs, rounded_upper_lim_all_yrs,
                                       title_text, tick_labels,
                                       2020, net_colors_rgb, percentiles, percentile_0)

    # Removes axis ticks and labels
    mu.remove_ticks(ax)

    core_jpeg_name = f"veg_cropland__2020__v{cn.veg_model_version_underscore}__{uu.timestr()[0:8]}"
    if bounding_box_description:  # Adds bounding box description to file name, if supplied
        core_jpeg_name = f"{core_jpeg_name}_{bounding_box_description}"
    jpeg_path = f"{local_jpeg_non_pres_folder}/{core_jpeg_name}.jpeg"
    jpeg_for_pres_path = f"{local_jpeg_pres_folder}/{core_jpeg_name}__for_pres.jpeg"

    # Saves two versions of the map: without and with a source note in the bottom right
    out_jpeg_for_pres = mu.save_pres_non_pres_jpegs(ax, jpeg_path, jpeg_for_pres_path, 2020)

    series_end_time = time.time()
    print(f"vegetation+cropland took {round(series_end_time - series_start_time)} seconds: {uu.timestr()}")


def main(net_all_gases_geotif_local,input_vegetation_date, cropland_geotif_s3=None, livestock_geotif_s3=None, vegetation_model_path_description=None,
         center_latitude=None, center_longitude=None, lat_height=None, bounding_box_description=None):

    # Defines desired percentiles for colors. Specifies where colors transition in the data.
    # Setting neutral ends of sink and source is empirically based on the 0 value being around the 82nd percentile.
    # From some experimentation, it's better not to encode a neutral percentile (or associated color) here or below.
    # It dampens the colors around the neutral value (low emissions and removals) even more.
    # net_percentiles = [5, 30, 60, 70, 81,   # Sink
    #                    83, 88, 97, 94, 99]  # Source
    removals_percentiles = [5, 25, 50, 75, 99]
    emissions_percentiles = [5, 25, 50, 75, 99]

    # Colors in RGB. Gross emissions and removals are subset of net flux palette.
    # From https://colorbrewer2.org/#type=diverging&scheme=BrBG&n=10
    net_colors_rgb = [(0, 60, 48), (1, 102, 94), (53, 151, 143), (128, 205, 193), (199, 234, 229),  # Used for removals
                         (246, 232, 195), (223, 194, 125), (191, 129, 45), (140, 81, 10), (84, 48, 5)  # Used for emissions
                         ]
    removals_colors_rgb = net_colors_rgb[0:5]
    emissions_colors_rgb = net_colors_rgb[5:]

    # Datasets that need to be expanded to all output years
    vegetation_dir =f"{cn.veg_outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/annual_intervals/2020/{cn.flux_aggreg_pixel_meaning}/global/{input_vegetation_date}/"


    # Folders for local outputs
    cropland_reproj_folder = Path(cn.local_jpeg_folder_cropland)
    cropland_reproj_folder.mkdir(parents=True, exist_ok=True)

    local_jpeg_non_pres_folder = Path(f"{cn.local_jpeg_folder_AFOLU}output_jpegs_and_gifs_{bounding_box_description}/jpegs_non_pres")
    local_jpeg_non_pres_folder.mkdir(parents=True, exist_ok=True)
    local_jpeg_pres_folder = Path(f"{cn.local_jpeg_folder_AFOLU}output_jpegs_and_gifs_{bounding_box_description}/jpegs_pres")
    local_jpeg_pres_folder.mkdir(parents=True, exist_ok=True)
    local_gif_folder = Path(f"{cn.local_jpeg_folder_AFOLU}output_jpegs_and_gifs_{bounding_box_description}/gifs")
    local_gif_folder.mkdir(parents=True, exist_ok=True)

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
    map_AFOLU_totals(net_all_gases_geotif_local, cropland_geotif_s3, livestock_geotif_s3, cropland_reproj_folder,
                 local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
                 net_colors_rgb, country_shapefile, bounding_box, bounding_box_description)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create jpegs of 0.04x0.04 deg output maps.")
    parser.add_argument('-clat', '--center_latitude', type=float, help='Latitude to center output maps (optional)')
    parser.add_argument('-clon', '--center_longitude', type=float, help='Longitude to center output maps (optional)')
    parser.add_argument('-lh', '--lat_height', type=float, help='Latitude to show around lat center (value is total north/south) (optional)')
    parser.add_argument('-bbd', '--bounding_box_description', default='global', help='Description of bounding box (if used) to include in output names.')
    parser.add_argument('-veg', '--net_all_gases_geotif_local', help='Local vegetation net flux file to use')
    parser.add_argument('-ivd', '--input_vegetation_date', help='Date of vegetation model run, in YYYYMMDD')
    parser.add_argument('-cl', '--cropland_geotif_s3', help='s3 path for cropland management emissions')
    parser.add_argument('-ls', '--livestock_geotif_s3', help='s3 path for livestock emissions')
    parser.add_argument('-mpd', '--vegetation_model_path_description', help='Description of model run (e.g., global, test, X_area).')

    args = parser.parse_args()
    center_latitude = args.center_latitude
    center_longitude = args.center_longitude
    lat_height = args.lat_height
    bounding_box_description = args.bounding_box_description
    net_all_gases_geotif_local = args.net_all_gases_geotif_local
    input_vegetation_date = args.input_vegetation_date
    cropland_geotif_s3 = args.cropland_geotif_s3
    livestock_geotif_s3 = args.livestock_geotif_s3
    vegetation_model_path_description = args.vegetation_model_path_description

    main(net_all_gases_geotif_local, input_vegetation_date, cropland_geotif_s3=cropland_geotif_s3, livestock_geotif_s3=livestock_geotif_s3,
         vegetation_model_path_description=vegetation_model_path_description,
         center_latitude=center_latitude, center_longitude=center_longitude, lat_height=lat_height, bounding_box_description=bounding_box_description)

