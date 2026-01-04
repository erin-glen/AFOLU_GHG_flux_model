"""
python -m src.LULUCF.scripts.vegetation_model.4_create_0_04deg_global_display_maps -mt standard -mpd global --input_date YYYYMMDD

Run locally (not in Coiled)

A zoomed in map can be created by supplying central lat-long arguments, as well as a north-south extent for the map to include.
The aspect ratio used in the global map of 2:1 (width:height) is maintained, and the east-west extent is determined
from that information. That keeps all zoomed in maps in the same shape as the global map for simplicity.

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
import pyproj

from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Polygon, MultiPolygon, box, mapping
from scipy.stats import percentileofscore
from rasterio.windows import from_bounds
from pyproj import Transformer

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

def calculate_bbox_centered(center_lat, center_lon, lat_height, aspect_ratio=2.0):
    """
    Given a center point (lat/lon), vertical height in degrees latitude,
    and desired width:height aspect ratio, returns a bounding box in degrees
    that maintains the visual proportions in the map projection.

    Returns: (lon_min, lat_min, lon_max, lat_max)
    """
    # Compute vertical range
    lat_min = center_lat - lat_height / 2
    lat_max = center_lat + lat_height / 2

    # Setup projection to Robinson (or your map projection)
    src_crs = "EPSG:4326"
    dst_crs = cn.Robinson_crs  # e.g. 'ESRI:54030'
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    # Project the vertical extent at the center longitude
    _, y_min = transformer.transform(center_lon, lat_min)
    _, y_max = transformer.transform(center_lon, lat_max)
    height_m = abs(y_max - y_min)

    # Desired width in projected meters
    width_m = height_m * aspect_ratio

    # Estimate how many degrees of longitude gives that width
    # Use small step to compute meters per degree lon at center_lat
    test_dx = 1.0
    x0, _ = transformer.transform(center_lon, center_lat)
    x1, _ = transformer.transform(center_lon + test_dx, center_lat)
    meters_per_degree_lon = abs(x1 - x0)

    # Required lon range in degrees
    lon_width = width_m / meters_per_degree_lon
    lon_min = center_lon - lon_width / 2
    lon_max = center_lon + lon_width / 2

    return (lon_min, lat_min, lon_max, lat_max)

def transform_bbox_to_robinson(bbox_deg, src_crs="EPSG:4326", dst_crs=None):
    """
    Transforms a bounding box from lat/lon (EPSG:4326) to Robinson projection.
    bbox_deg: (minx, miny, maxx, maxy) in degrees
    dst_crs: destination CRS (defaults to cn.Robinson_crs)
    Returns: (minx, miny, maxx, maxy) in meters (Robinson)
    """
    if dst_crs is None:
        dst_crs = cn.Robinson_crs  # e.g., 'ESRI:54030'

    minx, miny, maxx, maxy = bbox_deg
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    # Transform corners
    xmin_t, ymin_t = transformer.transform(minx, miny)
    xmax_t, ymax_t = transformer.transform(maxx, maxy)

    # Return projected bounding box
    return (min(xmin_t, xmax_t), min(ymin_t, ymax_t),
            max(xmin_t, xmax_t), max(ymin_t, ymax_t))

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

def create_divergent_legend_asymmetric(fig, vmin, vcenter, vmax, title_text, tick_labels, year, colors_rgb):
    """
    Creates a vertical asymmetric colorbar legend where 0 is not visually centered.

    Parameters:
        fig: Matplotlib figure
        vmin: Minimum data value (e.g., -14)
        vcenter: Center value (e.g., 0)
        vmax: Maximum data value (e.g., 22)
        title_text: Text for the legend title
        tick_labels: Labels for the three ticks (min, 0, max)
        year: Year string, for logging/debug
    """
    print(f"  Creating asymmetric legend for {year}")

    # Converts net flux RGB palette to hex
    # per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68d6d26f-b054-8323-98bb-731a86582e74
    net_color_palette_hex = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in colors_rgb]

    neutral_rgb = tuple(round((colors_rgb[4][i] + colors_rgb[5][i]) / 2) for i in range(3))
    neutral_hex = "#{:02X}{:02X}{:02X}".format(*neutral_rgb)

    # Compute center position in normalized [0–1] space
    neutral_pos = abs(vmin) / (vmax - vmin)

    print(f"  Neutral tick position for legend: {neutral_pos}")

    # Creates the colormap manually with asymmetry.
    # Manually setting legend percentiles for now, and they don't bear any relationship to the actual data.
    # TODO make these percentiles actually reflect the percentiles of the values in the data. Haven't tried it at all.
    colors = [
        (0.0, net_color_palette_hex[0]),         # sink color
        (0.05, net_color_palette_hex[1]),         # sink color
        (0.20, net_color_palette_hex[2]),         # sink color
        (0.25, net_color_palette_hex[3]),         # sink color
        (0.36, net_color_palette_hex[4]),         # sink color
        (neutral_pos, neutral_hex),         # near neutral, midpoint of adjacent colors per ChatGPT
        (0.40, net_color_palette_hex[5]),         # source color
        (0.50, net_color_palette_hex[6]),         # source color
        (0.60, net_color_palette_hex[7]),         # source color
        (0.90, net_color_palette_hex[8]),         # source color
        (1.0, net_color_palette_hex[9]),          # source color
    ]

    cmap = LinearSegmentedColormap.from_list("asymmetric_div", colors)

    # Create fake gradient image for legend (not shown, just for colorbar)
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)

    # Create colorbar axis
    cbar_ax = fig.add_axes(cn.colorbar_dimensions)  # e.g., [left, bottom, width, height]

    im = cbar_ax.imshow(gradient, cmap=cmap, aspect='auto', origin='lower')
    cbar_ax.axis('off')

    # Add ticks using a separate axis
    cb_ax = fig.add_axes([
        cn.colorbar_dimensions[0] + cn.colorbar_dimensions[2] + 0.01,  # shift to right
        cn.colorbar_dimensions[1],
        0.03,  # width
        cn.colorbar_dimensions[3]
    ])

    cb = plt.colorbar(im, cax=cb_ax, orientation="vertical")
    cb.set_ticks([0.0, neutral_pos, 1.0])
    cb.set_ticklabels(tick_labels, fontsize=cn.legend_fontsize)

    # Add title above the bar (optional)
    cb_ax.text(
        0, 1.05,  # x, y in axes coords
        title_text,
        fontsize=cn.legend_fontsize,
        ha="left",
        va="bottom",
        transform=cb_ax.transAxes
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
def map_net_flux(s3_folders,
                 local_reproj_folder, local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
                 colors_rgb, percentiles, country_shapefile, bounding_box=None, bounding_box_description=None):

    series_start_time = time.time()

    out_maps_for_gif = []

    # If bounding_box was given in degrees, transforms to match the raster CRS (Robinson)
    if bounding_box is not None:
        bounding_box_proj = transform_bbox_to_robinson(bounding_box)
    else:
        bounding_box_proj = None

    print("Pre-scanning rasters to determine full time series color scale...")

    all_valid_values = []

    # First pass through years to get the range of values across years to standardize legend across years
    # for i, year in enumerate(cn.years_annual[1:]):
    for i, year in enumerate(cn.years_annual[2:3]):
        s3_folder = s3_folders[i]
        parts = s3_folder.strip('/').split('/')

        pattern_idx = parts.index(f"version_{cn.veg_model_version_underscore}__{model_type}__{model_path_description}")
        pattern_segment = parts[pattern_idx + 1]

        interval_idx = parts.index("annual_intervals")
        interval_segment = parts[interval_idx + 1]

        year_file = f"{pattern_segment}{cn.flux_aggreg_pixel_meaning}_v{cn.veg_model_version_underscore}_{interval_segment}_global"
        year_path_reproj = f"{local_reproj_folder}/{year_file}_reproj.tif"

        with rasterio.open(year_path_reproj) as src:
            if bounding_box_proj is not None:
                window = from_bounds(*bounding_box_proj, src.transform)
                data = src.read(1, window=window)
            else:
                data = src.read(1)

        valid = data[data != 0]
        if valid.size > 0:
            all_valid_values.append(valid)

    # Calculates min, center and max across all years
    all_valid_values = np.concatenate(all_valid_values)

    global_breaks = np.percentile(all_valid_values, [1, 99])  # The min and max percentiles at which colors saturate
    # global_breaks = np.percentile(all_valid_values, [0.5, 99.5])

    global_vmin = global_breaks[0]
    global_vcenter = 0
    global_vmax = global_breaks[-1]

    print("Global scale:")
    print("  vmin:", global_vmin)
    print("  vcenter:", global_vcenter)
    print("  vmax:", global_vmax)

    # Creates the legend in kt CO2e (converts legend units from Mg (t) to kt with 10**3-- data doesn't change).
    # Rounds data_min down and data_max up for legend.
    rounded_min = math.ceil(global_vmin / 10 ** 3 * 100) / 100  # Rounds up
    rounded_max = math.floor(global_vmax / 10 ** 3 * 100) / 100  # Rounds down
    print(rounded_min)
    print(rounded_max)
    tick_labels = [f"< {rounded_min:.0f}  (sink)",  # Spaces are to horizontally align the text explanations
                   "0           (neutral)",
                   f"> {rounded_max:.0f}  (source)"]
    print(tick_labels)

    # Iterates through modeled years
    # for i, year in enumerate(cn.years_annual[1:]):
    for i, year in enumerate(cn.years_annual[2:3]): # For testing a specific year

        # The s3 folder to process for this year
        s3_folder = s3_folders[i]

        # All the components of the input s3 path
        parts = s3_folder.strip('/').split('/')

        # Gets the segment for the input pattern
        pattern_idx = parts.index(f"version_{cn.veg_model_version_underscore}__{model_type}__{model_path_description}")
        pattern_segment = parts[pattern_idx + 1]

        # Gets the segment for the input interval
        interval_idx = parts.index(f"annual_intervals")
        interval_segment = parts[interval_idx + 1]

        # Names before and after reprojection
        year_file = f"{pattern_segment}{cn.flux_aggreg_pixel_meaning}_v{cn.veg_model_version_underscore}_{interval_segment}_global"
        year_path_unproj = f"{s3_folder}{year_file}.tif"
        year_path_reproj = f"{local_reproj_folder}/{year_file}_reproj.tif"

        print(f"\n\n---Mapping {pattern_segment} for {year} from {year_file}")

        print(f"Unprojected raster: {year_path_unproj}")
        print(f"Reprojected raster: {year_path_reproj}")

        # Reprojects raster, if needed
        reproject_raster(year_path_unproj, year_path_reproj)

        # # Reads raster data
        # with rasterio.open(year_path_reproj) as src:
        #     data = src.read(1)  # Read the first band
        #     raster_extent = src.bounds

        with rasterio.open(year_path_reproj) as src:

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

        # Calculates the percentile for 0 (no flux)
        percentile_0 = percentile_for_0(data)
        print(f"  0 is at the {percentile_0}th percentile of the raster for {year}.")

        # Matches percentile breaks with colors.
        # Normalizes percentiles to a 0-1 scale.
        print(f"  Calculating percentiles and breaks for {year}")

        # Converts RGB color palette to matplotlib color palette
        colors_matplotlib = rgb_to_mpl_palette(colors_rgb)

        # Makes percentiles for the breakpoints and prepares colormap
        percentiles_normalized = np.linspace(0, 1, len(percentiles))
        cmap = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized, colors_matplotlib)))

        # # Calculates breaks in the data based on the percentiles
        # breaks = np.percentile(data[data != 0], percentiles)  # Ignores NoData values
        # print(f"  Breaks for {year}: {breaks}")

        # # Min, center and max values for the colormap (not the min and max values for the raster)
        # vmin, vcenter, vmax = breaks[0], breaks[len(breaks) // 2], breaks[-1]  # Uses the median as the center
        # print(f"  vcenter: {vcenter}")

        print(f"  Masking raster for {year} to non-0 values")
        masked_data = np.ma.masked_where(data == 0, data)
        # data_min = masked_data.min()  # Minimum of the valid data
        # data_max = masked_data.max()  # Maximum of the valid data
        # print(f"  Min and max for {year} (Mg): {data_min}, {data_max}")
        #
        # print(f"  Normalizing map for {year}")
        # # Normalizes the data for the colormap
        # norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        norm = TwoSlopeNorm(
            vmin=global_vmin,
            # vcenter=global_vcenter,
            vcenter=0,
            vmax=global_vmax
        )

        print(f"  Plotting map for {year}")
        ax, fig = create_plot()

        # Sets the ocean color
        set_ocean_color(ax)

        if bounding_box_proj is not None:
            bbox_geom = box(*bounding_box_proj)
            country_shapefile = country_shapefile.clip(bbox_geom)

        # Plots the country polygons first
        plot_country_polygons(ax, country_shapefile)

        # Raster extent
        extent = list(raster_extent)

        # Plots the raster next
        img = plot_raster(ax, cmap, extent, masked_data, norm)

        # Plots the country boundaries on top
        plot_country_boundaries(ax, country_shapefile)

        # Explicitly sets the bounding box for the plot image
        if bounding_box_proj is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

        # Modifies the legend title based on the input.
        if "all_gases" in pattern_segment:
            title_text = f"Net greenhouse gas flux\nAll vegetation pools, all gases\nkt CO$_2$e yr$^{{-1}}$"
        else:
            title_text = f"Net greenhouse gas flux\nAll vegetation pools, CO2 only\nkt CO$_2$e yr$^{{-1}}$"

        VIS_VMIN = -14
        VIS_VMAX = 22
        VIS_VCENTER = 0
        create_divergent_legend_asymmetric(fig, VIS_VMIN, VIS_VCENTER, VIS_VMAX, title_text, tick_labels, year, colors_rgb)

        # Removes axis ticks and labels
        remove_ticks(ax)

        pattern_segment_revised = pattern_segment.replace("MgCO2", "ktCO2")  # Replaces Mg with the mapped unit of kt
        core_jpeg_name = f"veg_{pattern_segment_revised}__{year}__v{cn.veg_model_version_underscore}__{uu.timestr()[0:8]}"
        if bounding_box_description:  # Adds bounding box description to file name, if supplied
            core_jpeg_name = f"{core_jpeg_name}_{bounding_box_description}"
        jpeg_path = f"{local_jpeg_non_pres_folder}/{core_jpeg_name}.jpeg"
        jpeg_for_pres_path = f"{local_jpeg_pres_folder}/{core_jpeg_name}__for_pres.jpeg"

        # Saves two versions of the map: without and with a source note in the bottom right
        out_jpeg_for_pres = save_pres_non_pres_jpegs(ax, jpeg_path, jpeg_for_pres_path, year)

        out_maps_for_gif.append(out_jpeg_for_pres)

    # Creates gifs of timeseries
    gif_base_name = f"veg_{pattern_segment_revised}__{cn.years_annual[1]}_{cn.years_annual[-1]}__v{cn.veg_model_version_underscore}"
    create_gif(gif_base_name, local_gif_folder, out_maps_for_gif)

    series_end_time = time.time()
    print(f"{pattern_segment} took {round(series_end_time - series_start_time)} seconds: {uu.timestr()}")


# Makes jpeg of gross fluxes
def map_gross(s3_folders,
              local_reproj_folder, local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
              colors, percentiles, country_shapefile, bounding_box):

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
        pattern_idx = parts.index(f"version_{cn.veg_model_version_underscore}")
        pattern_segment = parts[pattern_idx + 1]

        # Gets the segment for the input interval
        interval_idx = parts.index(f"annual_intervals")
        interval_segment = parts[interval_idx + 1]

        # Names before and after reprojection
        year_file = f"{pattern_segment}{cn.flux_aggreg_pixel_meaning}_{interval_segment}_{cn.veg_model_version_underscore}__global"
        year_path_unproj = f"{s3_folder}{year_file}.tif"

        # For reasons I couldn't figure out, gross removals and CO2-only emissions just wouldn't work for some files
        # using the reprojected file names I wanted. So, these two need special reprojected file names.
        # From https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant
        if ("removals" in pattern_segment) or ("all_C_pools__CO2_only" in pattern_segment):
            year_path_reproj = f"{local_reproj_folder}/{pattern_segment}{cn.flux_aggreg_pixel_meaning}_{year}_reproj.tif"
        else:
            year_path_reproj = f"{local_reproj_folder}/{year_file}_reproj.tif"


        print(f"\n\n---Mapping {pattern_segment} for {year} from {year_file}")

        print(f"Unprojected raster: {year_path_unproj}")
        print(f"Reprojected raster: {year_path_reproj}")

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
        plot_country_polygons(ax, country_shapefile)

        # Raster extent
        extent = [raster_extent.left, raster_extent.right, raster_extent.bottom, raster_extent.top]

        # Plots the raster next
        img = plot_raster(ax, cmap, extent, masked_data, norm)

        # Plots the country boundaries on top
        plot_country_boundaries(ax, country_shapefile)

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
        core_jpeg_name = f"veg_{pattern_segment_revised}__{year}__v{cn.veg_model_version_underscore}__{uu.timestr()[0:8]}"
        jpeg_path = f"{local_jpeg_non_pres_folder}/{core_jpeg_name}.jpeg"
        jpeg_for_pres_path = f"{local_jpeg_pres_folder}/{core_jpeg_name}__for_pres.jpeg"

        # Saves two versions of the map: without and with a source note in the bottom right
        out_jpeg_for_pres = save_pres_non_pres_jpegs(ax, jpeg_path, jpeg_for_pres_path, year)

        out_maps_for_gif.append(out_jpeg_for_pres)

    # Creates gifs of timeseries
    gif_base_name = f"veg_{pattern_segment_revised}__{cn.years_annual[1]}_{cn.years_annual[-1]}__v{cn.veg_model_version_underscore}"
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


def main(input_date, model_type, model_path_description=None,
         center_latitude=None, center_longitude=None, lat_height=None, bounding_box_description=None):

    # Defines desired percentiles for colors. Specifies where colors transition in the data.
    # Setting neutral ends of sink and source is empirically based on the 0 value being around the 82nd percentile.
    # From some experimentation, it's better not to encode a neutral percentile (or associated color) here or below.
    # It dampens the colors around the neutral value (low emissions and removals) even more.
    net_percentiles = [5, 30, 60, 70, 81,   # Sink
                       83, 88, 97, 94, 99]  # Source
    removals_percentiles = [5, 25, 50, 75, 99]
    emissions_percentiles = [5, 25, 50, 75, 99]

    # Colors in RGB. Gross emissions and removals are subset of net flux palette.
    # From https://colorbrewer2.org/#type=diverging&scheme=BrBG&n=10
    net_color_palette = [(0, 60, 48), (1, 102, 94), (53, 151, 143), (128, 205, 193), (199, 234, 229),  # Used for removals
                         (246, 232, 195), (223, 194, 125), (191, 129, 45), (140, 81, 10), (84, 48, 5)  # Used for emissions
                         ]
    removals_colors = net_color_palette[0:5]
    emissions_colors = net_color_palette[5:]

    # Datasets that need to be expanded to all output years
    basic_dirs_to_expand = [
        # f"{cn.veg_outputs_path}{cn.gross_emis_all_C_pools_CO2_only_pattern}/{cn.model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        # f"{cn.veg_outputs_path}{cn.gross_emis_all_C_pools_non_CO2_only_pattern}/{cn.model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        # f"{cn.veg_outputs_path}{cn.gross_emis_all_C_pools_all_gases_pattern}/{cn.model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        # f"{cn.veg_outputs_path}{cn.gross_removals_all_C_pools_pattern}/{cn.model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        # f"{cn.veg_outputs_path}{cn.net_flux_all_C_pools_CO2_only_pattern}/{cn.model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        f"{cn.veg_outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"
    ]

    # Creates a list of output directories for all outputs and intervals based on specifics of the model run
    inputs_by_interval_dir_list = uu.create_output_dir_name_list(basic_dirs_to_expand, "annual", cn.first_model_year_annual,"global",
                                                                 model_type, cn.veg_model_version_underscore, model_path_description,
                                                                 cn.interval_end_years_annual,
                                                                 [1, 1, 1, 1, 1, 1, 1, 1, 1], input_date,
                                                                 True, cn.flux_aggreg_pixel_meaning)

    # s3 folders to process
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

    # Folders for local outputs
    local_reproj_folder = Path(cn.local_jpeg_folder)
    local_reproj_folder.mkdir(parents=True, exist_ok=True)
    local_jpeg_non_pres_folder = Path(f"{cn.local_jpeg_folder}output_jpegs_and_gifs/jpegs_non_pres")
    local_jpeg_non_pres_folder.mkdir(parents=True, exist_ok=True)
    local_jpeg_pres_folder = Path(f"{cn.local_jpeg_folder}output_jpegs_and_gifs/jpegs_pres")
    local_jpeg_pres_folder.mkdir(parents=True, exist_ok=True)
    local_gif_folder = Path(f"{cn.local_jpeg_folder}output_jpegs_and_gifs/gifs")
    local_gif_folder.mkdir(parents=True, exist_ok=True)

    # Reprojects simplified country boundary shapefile, if needed
    country_shapefile = check_and_reproject_shapefile(
        shapefile_path=cn.original_shapefile_path,
        target_crs=cn.Robinson_crs,
        reprojected_shapefile_path=cn.reprojected_shapefile_path
    )

    # Creates bounding box in degrees from given map center and desired latitude range (optional)
    if center_latitude is not None and center_longitude is not None and lat_height is not None:
        bounding_box = calculate_bbox_centered(
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

    map_net_flux(net_all_gases_input_folders_s3, local_reproj_folder,
                 local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
                 net_color_palette, net_percentiles, country_shapefile, bounding_box, bounding_box_description)

    # map_net_flux(net_CO2_only_input_folders_s3, local_reproj_folder,
    #              local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
    #              net_color_palette, net_percentiles, country_shapefile, bounding_box, bounding_box_description)
    #
    # map_gross(gross_emis_CO2_only_input_folders_s3, local_reproj_folder,
    #                  local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
    #                  emissions_colors, emissions_percentiles, country_shapefile, bounding_box, bounding_box_description)
    #
    # map_gross(gross_emis_non_CO2_input_folders_s3, local_reproj_folder,
    #                  local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
    #                  emissions_colors, emissions_percentiles, country_shapefile, bounding_box, bounding_box_description)
    #
    # map_gross(gross_emis_all_gases_input_folders_s3, local_reproj_folder,
    #                  local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
    #                  emissions_colors, emissions_percentiles, country_shapefile, bounding_box, bounding_box_description)
    #
    # map_gross(gross_removals_input_folders_s3, local_reproj_folder,
    #              local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
    #              removals_colors, removals_percentiles, country_shapefile, bounding_box, bounding_box_description)

    # # Generates three-panel map
    # create_three_panel_map()


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Create jpegs of 0.04x0.04 deg output maps.")
    parser.add_argument('-clat', '--center_latitude', type=float, help='Latitude to center output maps (optional)')
    parser.add_argument('-clon', '--center_longitude', type=float, help='Longitude to center output maps (optional)')
    parser.add_argument('-lh', '--lat_height', type=float, help='Latitude to show around lat center (value is total north/south) (optional)')
    parser.add_argument('-bbd', '--bounding_box_description', help='Description of bounding box (if used) to include in output names.')
    parser.add_argument('-id', '--input_date', help='Date of run, in YYYYMMDD')
    parser.add_argument('-mt', '--model_type', default='standard', help='Type of model run (e.g., standard).')
    parser.add_argument('-mpd', '--model_path_description', help='Description of model run (e.g., global, test, X_area).')

    args = parser.parse_args()
    center_latitude = args.center_latitude
    center_longitude = args.center_longitude
    lat_height = args.lat_height
    bounding_box_description = args.bounding_box_description
    input_date = args.input_date
    model_type = args.model_type
    model_path_description = args.model_path_description

    main(input_date, model_type, model_path_description=model_path_description,
         center_latitude=center_latitude, center_longitude=center_longitude, lat_height=lat_height, bounding_box_description=bounding_box_description)

