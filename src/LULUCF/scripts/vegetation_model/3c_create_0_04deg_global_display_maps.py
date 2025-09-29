"""
python -m src.LULUCF.scripts.vegetation_model.3c_create_0_04deg_global_display_maps --input_date YYYYMMDD

With https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67634e63-bbcc-800a-8267-004e88ced2e4
Continued at https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68d6d26f-b054-8323-98bb-731a86582e74
"""

import argparse
import math
import os
import boto3
import rasterio
import geopandas as gpd
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Polygon, MultiPolygon
from scipy.stats import percentileofscore

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu


def rgb_to_mpl(rgb):
    """
    Converts RGB from 0-255 range to matplotlib-compatible 0-1 range.
    :param rgb: Tuple of (R, G, B) in 0-255 range.
    :return: Tuple of (R, G, B) in 0-1 range.
    """
    return tuple(val / 255 for val in rgb)

def download_s3_file(s3_uri, local_path):
    """
    Downloads a file from S3 to a local path.

    Parameters:
    - s3_uri (str): The full S3 URI, e.g., 's3://bucket/key/file.tif'
    - local_path (str or Path): Local path to save the file
    """
    print(f"  Downloading {s3_uri} to {local_path}")

    # Parse S3 URI
    assert s3_uri.startswith("s3://"), f"Invalid S3 URI: {s3_uri}"
    parts = s3_uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1]

    # Ensure local directory exists
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    # Use boto3 to download
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(local_path))

def reproject_raster(tif_unproj_s3, tif_reproj_local):
    """
    Reprojects raster to Robinson projection if the output doesn't already exist.
    Only supports local output; input can be S3.
    """

    if not os.path.exists(tif_reproj_local):
        print("  Reprojected raster does not exist. Reprojecting now...")

        with rasterio.open(tif_unproj_s3) as src:
            transform, width, height = calculate_default_transform(
                src.crs, cn.Robinson_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': cn.Robinson_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'nodata': 0,
                'compress': 'lzw'  # Optional: add compression
            })

            with rasterio.open(tif_reproj_local, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=cn.Robinson_crs,
                    resampling=Resampling.nearest
                )
    else:
        print("  Reprojected raster already exists locally. Skipping reprojection.")

def check_and_reproject_shapefile(shapefile_path, target_crs, reprojected_shapefile_path):
    """
    Checks if the shapefile is already projected to the target CRS.
    If not, reprojects the shapefile, saves it, and returns the reprojected shapefile.

    Parameters:
    - shapefile_path (str): Path to the input shapefile.
    - target_crs (str): The target CRS in PROJ format (e.g., "EPSG:4326" or "ESRI:54030").
    - reprojected_shapefile_path (str): Path to save the reprojected shapefile.

    Returns:
    - geopandas.GeoDataFrame: The original or reprojected shapefile.
    """

    # Checks if the reprojected shapefile already exists
    if os.path.exists(reprojected_shapefile_path):
        print(f"  Reprojected shapefile already exists at {reprojected_shapefile_path}.")
        return gpd.read_file(reprojected_shapefile_path)

    # Loads the shapefile
    shapefile = gpd.read_file(shapefile_path)

    # Checks if the shapefile is already in the target CRS
    if shapefile.crs == target_crs:
        print(f"  Shapefile is already projected to {target_crs}.")
        return shapefile

    # Reprojects the shapefile
    print(f"  Reprojecting shapefile from {shapefile.crs} to {target_crs}.")
    shapefile = shapefile.to_crs(target_crs)

    # Saves the reprojected shapefile for future use
    shapefile.to_file(reprojected_shapefile_path)
    print(f"  Reprojected shapefile saved to {reprojected_shapefile_path}.")

    return shapefile

def create_plot():
    """
    Creates matplotlib plot
    :return: ax and fig
    """
    fig, ax = plt.subplots(figsize=cn.panel_dims)
    return ax, fig

def remove_ticks(ax):
    """
    Removes ticks from matplotlib plot
    :param ax: graph
    :return: N/A
    """
    # Set map aesthetics
    # NOTE: can't use ax.set_axis_off() to remove axis ticks and labels because it also changes the background color back to white
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.set_yticklabels([])  # Remove y-axis labels

def create_divergent_legend(fig, img, vmin, vcenter, vmax, title_text, tick_labels, year):
    """
    Creates a vertical colorbar legend with a left-aligned title above it.
    :param fig: The figure
    :param img: The image
    :param vmin: minimum value to use in scaling legend colors
    :param vcenter: middle value to use in scaling legend colors
    :param vmax: maximum value to use in scaling legend colors
    :param title_text: Title for legend
    :param tick_labels: Tick labels for legend
    :return: N/A
    """

    print(f"  Creating legend for {year}")

    # Add a vertical colorbar (legend) in the bottom-left of the map
    cbar_ax = fig.add_axes(cn.colorbar_dimensions)  # [left, bottom, width, height]
    cb = plt.colorbar(img, cax=cbar_ax, orientation="vertical")

    # Set custom ticks and labels for the colorbar
    cb.set_ticks([vmin, vcenter, vmax])  # Set the ticks at the minimum, zero, and maximum
    cb.set_ticklabels(tick_labels, fontsize=cn.legend_fontsize)  # Format the labels

    # Add a left-aligned, multi-row title above the colorbar
    cbar_ax.text(
        0, 1.1,  # Adjust the x (horizontal) and y (vertical) coordinates for the title position
        title_text,
        fontsize=cn.legend_fontsize,
        ha="left",  # Horizontally align the text to the left
        va="bottom",  # Vertically align the text
        transform=cbar_ax.transAxes  # Use axes coordinates for positioning
    )

def create_unidirection_legend(fig, img, vmin, vmax, title_text, tick_labels, year):
    """
    Creates a vertical colorbar legend with a left-aligned title above it.
    :param fig: The figure
    :param img: The image
    :param vmin: minimum value to use in scaling legend colors
    :param vmax: maximum value to use in scaling legend colors
    :param title_text: Title for legend
    :param tick_labels: Tick labels for legend
    :return: N/A
    """

    print(f"  Creating legend for {year}")

    # Add a vertical colorbar (legend) in the bottom-left of the map
    cbar_ax = fig.add_axes(cn.colorbar_dimensions)  # [left, bottom, width, height]
    cb = plt.colorbar(img, cax=cbar_ax, orientation="vertical")

    # Set custom ticks and labels for the colorbar
    cb.set_ticks([vmin, vmax])  # Set the ticks at the minimum, zero, and maximum
    cb.set_ticklabels(tick_labels, fontsize=cn.legend_fontsize)  # Format the labels

    # Add a left-aligned, multi-row title above the colorbar
    cbar_ax.text(
        0, 1.1,  # Adjust the x (horizontal) and y (vertical) coordinates for the title position
        title_text,
        fontsize=cn.legend_fontsize,
        ha="left",  # Horizontally align the text to the left
        va="bottom",  # Vertically align the text
        transform=cbar_ax.transAxes  # Use axes coordinates for positioning
    )

def rgb_to_mpl_palette(rgb_palette):
    """
    Converts a list of RGB colors from 0-255 range to 0-1 range for Matplotlib.

    Parameters:
    - rgb_palette (list of tuples): List of RGB tuples (R, G, B) in 0-255 range.

    Returns:
    - list: List of RGB tuples (R, G, B) in 0-1 range.
    """
    return [tuple(val / 255 for val in rgb) for rgb in rgb_palette]

def percentile_for_0(data):

    # Masks invalid values (e.g., NoData or zero values)
    valid_data = data[data != 0]  # Excludes zeros (or use np.ma.masked_invalid for general NoData masking)

    # Ensures valid_data is not empty
    if len(valid_data) == 0:
        raise ValueError("No valid data found in the raster.")

    # Calculates the percentile of 0
    percentile_0 = percentileofscore(valid_data, 0, kind="mean")

    return percentile_0

def set_ocean_color(ax):
    # Sets the background color of the map
    ax.set_facecolor(rgb_to_mpl(cn.ocean_color))  # Set the background color

def plot_country_polygons(ax, shapefile):
    """
    Plots the shapefile polygons or multipolygons with a specified color. zorder sets the order of drawing.
    :param ax: figure
    :param shapefile: shapefile to draw
    :return: N/A
    """

    for geom in shapefile.geometry:
        if isinstance(geom, Polygon):
            # Single Polygon
            x, y = geom.exterior.xy
            ax.fill(x, y, color=rgb_to_mpl(cn.land_bkgrnd), zorder=1)
        elif isinstance(geom, MultiPolygon):
            # MultiPolygon: Iterate through each Polygon in the MultiPolygon
            for part in geom.geoms:
                x, y = part.exterior.xy
                ax.fill(x, y, color=rgb_to_mpl(cn.land_bkgrnd), zorder=1)

def plot_raster(ax, cmap, extent, masked_data, norm):
    """
    Plots raster
    :param ax: figure
    :param cmap: colormap
    :param extent: raster extent
    :param masked_data: masked data (no NoData/0s) to plot
    :param norm: data normalization
    :return: image
    """

    img = ax.imshow(masked_data, cmap=cmap, norm=norm, extent=extent, origin='upper', zorder=2)
    return img

def plot_country_boundaries(ax, shapefile):

    # Overlaya shapefile boundaries (e.g., country borders)
    # zorder determines the order of appearance in the figure
    shapefile.boundary.plot(ax=ax, edgecolor=rgb_to_mpl(cn.boundary_color), linewidth=cn.boundary_width, zorder=3)

def save_jpeg(out_jpeg, year):

    print(f"  Saving {out_jpeg} for {year}")
    plt.savefig(out_jpeg, dpi=cn.dpi_jpeg, bbox_inches="tight", pad_inches=0)

def save_pres_non_pres_jpegs(ax, out_jpeg, out_jpeg_for_pres, year):

    # Adds year label inside the plot, bottom center
    ax.text(
        0.5, 0.07, str(year),   # x=50% (center), slightly into the panel space
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=18, weight="bold", color="black"
    )

    # Saves jpeg without journal name and update notes in bottom right
    save_jpeg(out_jpeg, year)

    # Note in bottom right of panel
    ax.text(0.98, 0.04, cn.pres_text, transform=ax.transAxes, fontsize=7,
            ha="right", va="top", color="black")

    # Saves jpeg with journal name and update notes in bottom right
    save_jpeg(out_jpeg_for_pres, year)
    plt.close()

    return out_jpeg_for_pres

# Creates gifs of timeseries (fast and slow)
def create_gif(gif_base_name, out_folder, out_maps_for_gif):
    # Open all images
    frames = [Image.open(img) for img in out_maps_for_gif]
    # Save as GIF
    frames[0].save(
        f"{out_folder}/{gif_base_name}__fast.gif",
        save_all=True,
        append_images=frames[1:],  # add the rest
        duration=1000,  # ms per frame
        loop=0  # 0 = infinite loop
    )
    frames[0].save(
        f"{out_folder}/{gif_base_name}__slow.gif",
        save_all=True,
        append_images=frames[1:],  # add the rest
        duration=2500,  # ms per frame
        loop=0  # 0 = infinite loop
    )

# Makes jpegs and gifs of net fluxes
def map_net_flux(input_date, s3_folders,
                 local_reproj_folder, local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
                 colors, percentiles):

    series_start_time = time.time()

    out_maps_for_gif = []

    # Iterates through modeled years
    for i, year in enumerate(cn.years_annual[1:]):
    # for i, year in enumerate(cn.years_annual[2:3]): # For testing a specific year

        # The s3 folder to process for this year
        s3_folder = s3_folders[i]

        # All the components of the input s3 path
        parts = s3_folder.strip('/').split('/')

        # Gets the segment for the input pattern
        pattern_idx = parts.index(f"version_{cn.model_version_underscore}")
        pattern_segment = parts[pattern_idx + 1]

        # Gets the segment for the input interval
        interval_idx = parts.index(f"annual_intervals")
        interval_segment = parts[interval_idx + 1]

        # Names before and after reprojection
        year_file = f"{pattern_segment}{cn.flux_aggreg_pixel_meaning}_{interval_segment}_global"
        year_path_unproj = f"{s3_folder}{year_file}.tif"
        year_path_reproj = f"{local_reproj_folder}/{year_file}_reproj.tif"

        print(f"\n\n---Mapping {pattern_segment} for {year} from {year_file}")

        # Reprojects raster, if needed
        reproject_raster(year_path_unproj, year_path_reproj)

        # Reads raster data
        with rasterio.open(year_path_reproj) as src:
            data = src.read(1)  # Read the first band
            raster_extent = src.bounds

        # Calculates the percentile for 0 (no flux)
        percentile_0 = percentile_for_0(data)
        print(f"  0 is at the {percentile_0}th percentile of the raster for {year}.")

        # Matches percentile breaks with colors.
        # Normalizes percentiles to a 0-1 scale.
        print(f"  Calculating percentiles and breaks for {year}")

        # Converts RGB color palette to matplotlib color palette
        colors_matplotlib = rgb_to_mpl_palette(colors)

        # Makes percentiles for the breakpoints and prepares colormap
        percentiles_normalized = np.linspace(0, 1, len(percentiles))
        cmap = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized, colors_matplotlib)))

        # Calculates breaks in the data based on the percentiles
        breaks = np.percentile(data[data != 0], percentiles)  # Ignores NoData values
        print(f"  Breaks for {year}: {breaks}")

        # Min, center and max values for the colormap (not the min and max values for the raster)
        vmin, vcenter, vmax = breaks[0], breaks[len(breaks) // 2], breaks[-1]  # Uses the median as the center
        print(f"  vcenter: {vcenter}")

        print(f"  Masking raster for {year} to non-0 values")
        masked_data = np.ma.masked_where(data == 0, data)
        data_min = masked_data.min()  # Minimum of the valid data
        data_max = masked_data.max()  # Maximum of the valid data
        print(f"  Min and max for {year} (Mg): {data_min}, {data_max}")

        print(f"  Normalizing for {year}")
        # Normalizes the data for the colormap
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        print(f"  Plotting map for {year}")
        ax, fig = create_plot()

        # Sets the ocean color
        set_ocean_color(ax)

        # Plots the country polygons first
        plot_country_polygons(ax, shapefile)

        # Raster extent
        extent = [raster_extent.left, raster_extent.right, raster_extent.bottom, raster_extent.top]

        # Plots the raster next
        img = plot_raster(ax, cmap, extent, masked_data, norm)

        # Plots the country boundaries on top
        plot_country_boundaries(ax, shapefile)

        # Creates the legend in kt CO2e (converts legend units from Mg (t) to kt with 10**3-- data doesn't change).
        # Rounds data_min down and data_max up for legend.
        rounded_min = math.ceil(data_min/10**3 * 100) / 100  # Rounds up
        rounded_max = math.floor(data_max/10**3 * 100) / 100  # Rounds down
        tick_labels = [f"< {rounded_min:.0f}  (sink)",   # Spaces are to horizontally align the text explanations
                        "0           (neutral)",
                       f"> {rounded_max:.0f}  (source)"]

        # Modifies the legend title based on the input.
        if "all_gases" in pattern_segment:
            title_text = f"Net greenhouse gas flux\nAll vegetation pools, all gases\nkt CO$_2$e yr$^{{-1}}$"
        else:
            title_text = f"Net greenhouse gas flux\nAll vegetation pools, CO2 only\nkt CO$_2$e yr$^{{-1}}$"

        create_divergent_legend(fig, img, vmin, vcenter, vmax, title_text, tick_labels, year)

        # Removes axis ticks and labels
        remove_ticks(ax)

        pattern_segment_revised = pattern_segment.replace("MgCO2", "ktCO2")  # Replaces Mg with the mapped unit of kt
        core_jpeg_name = f"veg_{pattern_segment_revised}__{year}__v{cn.model_version_underscore}"  #
        jpeg_path = f"{local_jpeg_non_pres_folder}/{core_jpeg_name}.jpeg"
        jpeg_for_pres_path = f"{local_jpeg_pres_folder}/{core_jpeg_name}__for_pres.jpeg"

        # Saves two versions of the map: without and with a source note in the bottom right
        out_jpeg_for_pres = save_pres_non_pres_jpegs(ax, jpeg_path, jpeg_for_pres_path, year)

        out_maps_for_gif.append(out_jpeg_for_pres)

    # Creates gifs of timeseries
    gif_base_name = f"veg_{pattern_segment_revised}__{cn.years_annual[1]}_{cn.years_annual[-1]}__v{cn.model_version_underscore}"
    create_gif(gif_base_name, local_gif_folder, out_maps_for_gif)

    series_end_time = time.time()
    print(f"{pattern_segment} took {round(series_end_time - series_start_time)} seconds: {uu.timestr()}")


# Makes jpeg of gross fluxes
def map_gross(input_date, s3_folders,
                 local_reproj_folder, local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
                 colors, percentiles):

    series_start_time = time.time()

    out_maps_for_gif = []

    # Iterates through modeled years
    for i, year in enumerate(cn.years_annual[1:]):
    # for i, year in enumerate(cn.years_annual[1:2]): # For testing a specific year

        # The s3 folder to process for this year
        s3_folder = s3_folders[i]

        # All the components of the input s3 path
        parts = s3_folder.strip('/').split('/')

        # Gets the segment for the input pattern
        pattern_idx = parts.index(f"version_{cn.model_version_underscore}")
        pattern_segment = parts[pattern_idx + 1]

        # Gets the segment for the input interval
        interval_idx = parts.index(f"annual_intervals")
        interval_segment = parts[interval_idx + 1]

        # Names before and after reprojection
        year_file = f"{pattern_segment}{cn.flux_aggreg_pixel_meaning}_{interval_segment}_global"
        year_path_unproj = f"{s3_folder}{year_file}.tif"

        # For reasons I couldn't figure out, gross removals and CO2-only emissions just wouldn't work for some files
        # using the reprojected file names I wanted. So, these two need special reprojected file names.
        # From https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant
        if ("removals" in pattern_segment) or ("all_C_pools__CO2_only" in pattern_segment):
            year_path_reproj = f"{local_reproj_folder}/{pattern_segment}{cn.flux_aggreg_pixel_meaning}_{year}_reproj.tif"
        else:
            year_path_reproj = f"{local_reproj_folder}/{year_file}_reproj.tif"

        print(f"\n\n---Mapping {pattern_segment} for {year} from {year_file}")

        # Reprojects raster, if needed
        reproject_raster(year_path_unproj, year_path_reproj)

        # Reads raster data
        with rasterio.open(year_path_reproj) as src:
            data = src.read(1)  # Read the first band
            raster_extent = src.bounds

        # Matches percentile breaks with colors.
        # Normalizes percentiles to a 0-1 scale.
        print(f"  Calculating percentiles and breaks for {year}")

        # Converts RGB color palette to matplotlib color palette
        colors_matplotlib = rgb_to_mpl_palette(colors)

        # Makes percentiles for the breakpoints and prepares colormap
        percentiles_normalized = np.linspace(0, 1, len(percentiles))
        cmap = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized, colors_matplotlib)))

        # Calculates breaks in the data based on the percentiles
        breaks = np.percentile(data[data != 0], percentiles)  # Ignores NoData values
        print("  Breaks:", breaks)

        # Min and max values for the colormap (not the min and max values for the raster)
        vmin, vmax = breaks[0], breaks[-1]
        print(f"  vmin: {vmin}, vmax: {vmax}")

        print(f"  Masking raster to non-0 values for {year}")
        if "removals" in year_path_reproj:
            masked_data = np.ma.masked_where(data >= 0, data)

            # This colors all 0-value pixels, leaving non-0s white.
            # It clearly shows that more of Australia is non-0, but I just can't get it to be symbolized in any masking.
            # masked_data = np.ma.masked_where(data < 0, data)
        elif "emis" in year_path_reproj:
            masked_data = np.ma.masked_where(data <= 0, data)
        else:
            masked_data = np.ma.masked_where(data == 0, data)
            print("Not using either emissions or removals")
        data_min = masked_data.min()  # Minimum of the valid data
        data_max = masked_data.max()  # Maximum of the valid data

        print(f"  Normalizing for {year}")
        # Normalizes the data for the colormap
        norm = Normalize(vmin=vmin, vmax=vmax)

        print(f"  Plotting map for {year}")
        ax, fig = create_plot()

        # Sets the ocean color
        set_ocean_color(ax)

        # Plots the country polygons first
        plot_country_polygons(ax, shapefile)

        # Raster extent
        extent = [raster_extent.left, raster_extent.right, raster_extent.bottom, raster_extent.top]

        # Plots the raster next
        img = plot_raster(ax, cmap, extent, masked_data, norm)

        # Plots the country boundaries on top
        plot_country_boundaries(ax, shapefile)

        # Creates the legend in kt CO2e (converts legend units from Mg (t) to kt with 10**3-- data doesn't change).
        # Rounds data_min down and data_max up for legend.
        rounded_min = math.floor(data_min/10**3 * 100) / 100  # Round down
        rounded_max = math.ceil(data_max/10**3 * 100) / 100  # Round up
        # print(data_min, rounded_min)
        # print(data_max, rounded_max)

        # Legend labels depend on what exact input is displayed
        if "removals" in pattern_segment:
            tick_labels = [f"< {rounded_min:.0f}", 0]
            title_text = f"Gross removals\nAll vegetation pools\nkt CO$_2$ yr$^{{-1}}$"
        elif "all_gases" in pattern_segment:
            tick_labels = [0, f"> {rounded_max:.0f}"]
            title_text = f"Gross emissions\nAll vegetation pools, all gases\nkt CO$_2$e yr$^{{-1}}$"
        elif "non_CO2_only" in pattern_segment:
            tick_labels = [0, f"> {rounded_max:.0f}"]
            title_text = f"Gross emissions\nAll vegetation pools, non-CO$_2$ only\nkt CO$_2$e yr$^{{-1}}$"
        elif "CO2_only" in pattern_segment:
            tick_labels = [0, f"> {rounded_max:.0f}"]
            title_text = f"Gross emissions\nAll vegetation pools, CO$_2$ only\nkt CO$_2$ yr$^{{-1}}$"
        else:
            tick_labels = ["N/A", "N/A"]
            title_text = ""
            print("Can't generate tick labels")

        create_unidirection_legend(fig, img, vmin, vmax, title_text, tick_labels, year)

        # Removes axis ticks and labels
        remove_ticks(ax)

        pattern_segment_revised = pattern_segment.replace("MgCO2", "ktCO2")  # Replaces Mg with the mapped unit of kt
        core_jpeg_name = f"veg_{pattern_segment_revised}__{year}__v{cn.model_version_underscore}"  #
        jpeg_path = f"{local_jpeg_non_pres_folder}/{core_jpeg_name}.jpeg"
        jpeg_for_pres_path = f"{local_jpeg_pres_folder}/{core_jpeg_name}__for_pres.jpeg"

        # Saves two versions of the map: without and with a source note in the bottom right
        out_jpeg_for_pres = save_pres_non_pres_jpegs(ax, jpeg_path, jpeg_for_pres_path, year)

        out_maps_for_gif.append(out_jpeg_for_pres)

    # Creates gifs of timeseries
    gif_base_name = f"veg_{pattern_segment_revised}__{cn.years_annual[1]}_{cn.years_annual[-1]}__v{cn.model_version_underscore}"
    create_gif(gif_base_name, local_gif_folder, out_maps_for_gif)

    series_end_time = time.time()
    print(f"{pattern_segment} took {round(series_end_time - series_start_time)} seconds: {uu.timestr()}")



def create_three_panel_map():
    """
    Creates a three-panel map showing emissions, removals, and net flux.
    """
    print("Creating three-panel map")

    # Loads individual panel images
    emissions_img = plt.imread(emissions_jpeg)
    removals_img = plt.imread(removals_jpeg)
    net_img = plt.imread(net_jpeg)

    # Panel titles and images
    panel_labels = ["a", "b", "c"]
    images = [emissions_img, removals_img, net_img]

    three_panel_dims = (cn.panel_dims[0], cn.panel_dims[1] * len(images))

    # Sets up the figure
    fig, axes = plt.subplots(nrows=len(images), ncols=1, figsize=three_panel_dims)

    # Removes spaces between panels
    fig.subplots_adjust(hspace=0, wspace=0)

    # Adds each panel to the figure
    for ax, img, label in zip(axes, images, panel_labels):
        ax.imshow(img, aspect='auto')
        ax.axis("off")  # Removes axis ticks
        # Adds panel label in the top-left corner
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=10, fontweight="bold",
                ha="left", va="top", color="black")

    # Saves jpeg
    save_jpeg(out_jpeg)
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create jpegs of 0.04x0.04 deg output maps.")
    parser.add_argument('-id', '--input_date', help='Date of run, in YYYYMMDD')

    args = parser.parse_args()

    input_date = args.input_date

    # Defines desired percentiles for colors
    net_percentiles = [5, 25, 50, 75, 89, 91, 92, 93, 94, 99]  # Specifies where colors transition in the data
    removals_percentiles = [5, 25, 50, 75, 99]
    emissions_percentiles = [5, 25, 50, 75, 99]

    # Colors in RGB. Gross emissions and removals are subset of net flux palette.
    # From https://colorbrewer2.org/#type=diverging&scheme=BrBG&n=10
    net_color_palette = [(0, 60, 48), (1, 102, 94), (53, 151, 143), (128, 205, 193), (199, 234, 229),  # Used for removals
                         (246, 232, 195), (223, 194, 125), (191, 129, 45), (140, 81, 10), (84, 48, 5)  # Used for emissions
                         ]
    removals_colors = net_color_palette[0:5]
    emissions_colors = net_color_palette[5:]

    local_folder = f"/mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/v1_0_0_2016_2024_global/"

    basic_dirs_to_expand = [
        f"{cn.outputs_path}{cn.gross_emis_all_C_pools_CO2_only_pattern}/{cn.model_type_placholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        f"{cn.outputs_path}{cn.gross_emis_all_C_pools_non_CO2_only_pattern}/{cn.model_type_placholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        f"{cn.outputs_path}{cn.gross_emis_all_C_pools_all_gases_pattern}/{cn.model_type_placholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        f"{cn.outputs_path}{cn.gross_removals_all_C_pools_pattern}/{cn.model_type_placholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        f"{cn.outputs_path}{cn.net_flux_all_C_pools_CO2_only_pattern}/{cn.model_type_placholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        f"{cn.outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/{cn.model_type_placholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"
    ]

    # Creates a list of output directories for all outputs and intervals based on specifics of the model run
    inputs_by_interval_dir_list = uu.create_output_dir_name_list(basic_dirs_to_expand, "annual", cn.first_model_year_annual,
                                                                  "global", "standard_model", cn.interval_end_years_annual,
                                                                 [1, 1, 1, 1, 1, 1, 1, 1, 1], input_date,
                                                                 True, cn.flux_aggreg_pixel_meaning)

    gross_emis_CO2_only_input_folders_s3 = []
    gross_emis_non_CO2_input_folders_s3 = []
    gross_emis_all_gases_input_folders_s3 = []
    gross_removals_input_folders_s3 = []
    net_CO2_only_input_folders_s3 = []
    net_all_gases_input_folders_s3 = []
    other_input_folders_s3 = []

    for input in inputs_by_interval_dir_list:
        if cn.gross_emis_all_C_pools_CO2_only_pattern in input:
            gross_emis_CO2_only_input_folders_s3.append(input)
        elif cn.gross_emis_all_C_pools_non_CO2_only_pattern in input:
            gross_emis_non_CO2_input_folders_s3.append(input)
        elif cn.gross_emis_all_C_pools_all_gases_pattern in input:
            gross_emis_all_gases_input_folders_s3.append(input)
        elif cn.gross_removals_all_C_pools_pattern in input:
            gross_removals_input_folders_s3.append(input)
        elif cn.net_flux_all_C_pools_CO2_only_pattern in input:
            net_CO2_only_input_folders_s3.append(input)
        elif cn.net_flux_all_C_pools_all_gases_pattern in input:
            net_all_gases_input_folders_s3.append(input)
        else:
            other_input_folders_s3.append(input)

    # print(gross_emis_CO2_only_input_folders_s3)
    # print(gross_emis_non_CO2_input_folders_s3)
    # print(gross_emis_all_gases_input_folders_s3)
    # print(gross_removals_input_folders_s3)
    # print(net_CO2_only_input_folders_s3)
    # print(net_all_gases_input_folders_s3)

    local_reproj_folder = Path(local_folder)
    local_reproj_folder.mkdir(parents=True, exist_ok=True)
    local_jpeg_non_pres_folder = Path(f"{local_folder}output_jpegs_and_gifs/jpegs_non_pres")
    local_jpeg_non_pres_folder.mkdir(parents=True, exist_ok=True)
    local_jpeg_pres_folder = Path(f"{local_folder}output_jpegs_and_gifs/jpegs_pres")
    local_jpeg_pres_folder.mkdir(parents=True, exist_ok=True)
    local_gif_folder = Path(f"{local_folder}output_jpegs_and_gifs/gifs")
    local_gif_folder.mkdir(parents=True, exist_ok=True)

    # Reprojects shapefile, if needed
    shapefile = check_and_reproject_shapefile(
        shapefile_path=cn.original_shapefile_path,
        target_crs=cn.Robinson_crs,
        reprojected_shapefile_path=cn.reprojected_shapefile_path
    )

    # Generates jpegs for gross emissions, removals and net flux

    map_gross(input_date, gross_emis_CO2_only_input_folders_s3, local_reproj_folder,
                     local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
                     emissions_colors, emissions_percentiles)

    map_gross(input_date, gross_emis_non_CO2_input_folders_s3, local_reproj_folder,
                     local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
                     emissions_colors, emissions_percentiles)

    map_gross(input_date, gross_emis_all_gases_input_folders_s3, local_reproj_folder,
                     local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
                     emissions_colors, emissions_percentiles)

    map_gross(input_date, gross_removals_input_folders_s3, local_reproj_folder,
                 local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
                 removals_colors, removals_percentiles)

    map_net_flux(input_date, net_CO2_only_input_folders_s3, local_reproj_folder,
                 local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
                 net_color_palette, net_percentiles)

    map_net_flux(input_date, net_all_gases_input_folders_s3, local_reproj_folder,
                 local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
                 net_color_palette, net_percentiles)

    # # Generates three-panel map
    # create_three_panel_map()
