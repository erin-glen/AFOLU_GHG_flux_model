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

def create_divergent_legend_asymmetric(fig, vmin, vmax, title_text, tick_labels,
                                       year, colors_rgb, percentiles, percentile_0):
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
    print(f"  Creating legend for {year}")

    # Converts net flux RGB palette to hex
    # per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68d6d26f-b054-8323-98bb-731a86582e74
    net_colors_rgb_hex = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in colors_rgb]

    # Calculates color (RGB and hex) for neutral value-- for legend only. Not used in mop because it makes the map look worse.
    neutral_rgb = tuple(round((colors_rgb[4][i] + colors_rgb[5][i]) / 2) for i in range(3))
    neutral_hex = "#{:02X}{:02X}{:02X}".format(*neutral_rgb)

    # Computes neutral (no flux) position in normalized [0–1] space for display on legend
    neutral_pos = abs(vmin) / (vmax - vmin)
    print(f"  Neutral tick position for legend: {neutral_pos}")

    # Creates the colormap manually with asymmetry.
    # Determining what percentile of the legend each color should be at was pretty convoluted.
    # Long ChatGPT conversation (https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68d6d26f-b054-8323-98bb-731a86582e74)
    # which ended up getting me an asymmetric legend and the basis for this percentile-color list
    # but not the actual percentiles.
    # I messed around for a while to figure out how to calculate the percentile for each color.
    # I am still not sure this is entirely right (in the sense that the percentiles here may not exactly match the
    # percentiles used for colors on the map) but this should be somewhat close or at least generally representative.
    # I tried it with a global map and a Central Africa map and the legend looked okay at both scales.
    # Basically, I tried to determine at what point on the legend each color should go relative to the neutral value,
    # hence, everything is in reference to neutral_pos (when flux=0).
    colors = [
        (0.0, net_colors_rgb_hex[0]),         # sink color
        ((1-((percentile_0-percentiles[1])/percentile_0))*neutral_pos, net_colors_rgb_hex[1]),         # sink color
        ((1-((percentile_0-percentiles[2])/percentile_0))*neutral_pos, net_colors_rgb_hex[2]),         # sink color
        ((1-((percentile_0-percentiles[3])/percentile_0))*neutral_pos, net_colors_rgb_hex[3]),         # sink color
        ((1-((percentile_0-percentiles[4])/percentile_0))*neutral_pos, net_colors_rgb_hex[4]),         # sink color
        (neutral_pos, neutral_hex),         # near neutral, midpoint of adjacent colors per ChatGPT
        (neutral_pos+(1-neutral_pos)*((percentiles[5]-percentile_0)/percentile_0), net_colors_rgb_hex[5]),         # source color
        (neutral_pos+(1-neutral_pos)*((percentiles[6]-percentile_0)/percentile_0), net_colors_rgb_hex[6]),         # source color
        (neutral_pos+(1-neutral_pos)*((percentiles[7]-percentile_0)/percentile_0), net_colors_rgb_hex[7]),         # source color
        (neutral_pos+(1-neutral_pos)*((percentiles[8]-percentile_0)/percentile_0), net_colors_rgb_hex[8]),         # source color
        (1.0, net_colors_rgb_hex[9]),          # source color
    ]
    print(f"legend breakpoints and associated colors: {colors}")

    # Makes color map for legend
    cmap = LinearSegmentedColormap.from_list("asymmetric_div", colors)

    # Adds ticks using a separate axis
    cbar_ax = fig.add_axes([          # [left, bottom, width, height]
        cn.colorbar_dimensions[0] + cn.colorbar_dimensions[2],
        cn.colorbar_dimensions[1],
        cn.colorbar_dimensions[2],
        cn.colorbar_dimensions[3]
    ])

    cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax, orientation="vertical")
    cb.set_ticks([0.0, neutral_pos, 1.0])
    cb.set_ticklabels(tick_labels, fontsize=cn.legend_fontsize)

    # Adds title above the bar
    cbar_ax.text(
        0, 1.05,  # x, y in axes coords
        title_text,
        fontsize=cn.legend_fontsize,
        ha="left",
        va="bottom",
        transform=cbar_ax.transAxes
    )

def create_unidirection_legend(fig, img, lower_lim_all_yrs, upper_lim_all_yrs, title_text, tick_labels,
                               year, colors_rgb, percentiles):
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

    # Converts net flux RGB palette to hex
    # per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68d6d26f-b054-8323-98bb-731a86582e74
    net_colors_rgb_hex = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in colors_rgb]

    # Add a vertical colorbar (legend) in the bottom-left of the map
    cbar_ax = fig.add_axes([          # [left, bottom, width, height]
        cn.colorbar_dimensions[0] + cn.colorbar_dimensions[2],
        cn.colorbar_dimensions[1],
        cn.colorbar_dimensions[2],
        cn.colorbar_dimensions[3]
    ])
    cb = plt.colorbar(img, cax=cbar_ax, orientation="vertical")

    # Set custom ticks and labels for the colorbar
    cb.set_ticks([lower_lim_all_yrs, upper_lim_all_yrs])  # Set the ticks at the minimum, zero, and maximum
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

    # Turn off interpolation to prevent showing boundaries around pixel patches
    # (matters mostly for areas with sparse emissions in gross emissions map, like boreal forest)
    # https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6964d5d4-12b4-8325-aec3-c7bbed008ac9
    img = ax.imshow(masked_data, cmap=cmap, norm=norm, extent=extent,
                    origin='upper', interpolation='none', zorder=2)
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
    ax.text(0.99, 0.07, cn.pres_text, transform=ax.transAxes, fontsize=7,   #Vertical, horizontal
            ha="right", va="top", color="black")

    # Saves jpeg with journal name and update notes in bottom right
    save_jpeg(out_jpeg_for_pres, year)
    plt.close()

    return out_jpeg_for_pres

# Creates gifs of timeseries (fast and slow)
def create_gif(out_maps_for_gif, output_gif_path):
    """
    Create a GIF from JPEGs using a consistent color palette across frames.

    Parameters:
        jpeg_folder (str): Path to folder containing .jpeg frames.
        output_gif_path (str): Where to save the output .gif.
    """
    # Step 1: Loads JPEG frames
    frames = [Image.open(f).convert("RGB") for f in out_maps_for_gif]

    # Step 2: Builds a global color palette.
    # I was finding that even if the color palettes looked the same in each jpeg,
    # they were different by year in the gif for some reason.
    # Using a single color palette fixes that.
    print("Generating global color palette from all frames...")
    combined_height = frames[0].height * len(frames)
    combined = Image.new("RGB", (frames[0].width, combined_height))
    for i, frame in enumerate(frames):
        combined.paste(frame, (0, i * frames[0].height))

    # Gets adaptive palette from composite image
    palette_image = combined.convert("P", palette=Image.ADAPTIVE, colors=256)
    global_palette = palette_image.getpalette()

    # Step 3: Applies palette to each frame.
    # Needs this specific dithering to prevent the gif legend from being blocky,
    # per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/695e7ea8-60dc-832b-ba01-c4852bedbc57.
    # It was looking blocky otherwise.
    palettized_frames = [
        f.quantize(palette=palette_image, dither=Image.FLOYDSTEINBERG)
        for f in frames
    ]

    # Step 4: Saves GIF
    print(f"Saving animated GIF to: {output_gif_path}")
    palettized_frames[0].save(
        f"{output_gif_path}_fast.gif",
        save_all=True,
        append_images=palettized_frames[1:],
        duration=1000,
        loop=0,
        optimize=False,
        disposal=2  # Clears previous frame
    )

    palettized_frames[0].save(
        f"{output_gif_path}_slow.gif",
        save_all=True,
        append_images=palettized_frames[1:],
        duration=2500,
        loop=0,
        optimize=False,
        disposal=2  # Clears previous frame
    )

# Makes jpegs and gifs of net fluxes
def map_net_flux(s3_folders, model_type, model_path_description,
                 local_reproj_folder, local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
                 colors_rgb, country_shapefile, bounding_box=None, bounding_box_description=None):

    series_start_time = time.time()

    out_maps_for_gif = []

    # If bounding_box was given in degrees, transforms to match the raster CRS (Robinson)
    if bounding_box is not None:
        bounding_box_proj = transform_bbox_to_robinson(bounding_box)
    else:
        bounding_box_proj = None

    all_valid_values = []

    # First pass: Reprojects input rasters
    for i, year in enumerate(cn.years_annual[1:]):
    # for i, year in enumerate(cn.years_annual[2:3]): # For testing a specific year

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

    # Second pass: Gets the range of values across years to standardize legend across years
    print("Pre-scanning rasters to determine full time series upper and lower limits...")
    for i, year in enumerate(cn.years_annual[1:]):
    # for i, year in enumerate(cn.years_annual[2:3]):

        # print(f"Scanning {year}")
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

    percentile_for_saturation = 1
    breaks_all_yrs = np.percentile(all_valid_values, [1, (100-percentile_for_saturation)])  # The min and max percentiles at which colors saturate

    lower_lim_all_yrs = breaks_all_yrs[0]
    global_neutral = 0
    upper_lim_all_yrs = breaks_all_yrs[-1]

    print("Across all years:")
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
    # print(tick_labels)

    # Final pass: Iterates through modeled years to create the jpegs
    for i, year in enumerate(cn.years_annual[1:]):
    # for i, year in enumerate(cn.years_annual[2:4]): # For testing a specific year

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

        # Names after reprojection
        year_file = f"{pattern_segment}{cn.flux_aggreg_pixel_meaning}_v{cn.veg_model_version_underscore}_{interval_segment}_global"
        year_path_reproj = f"{local_reproj_folder}/{year_file}_reproj.tif"

        print(f"\n\n---Mapping {pattern_segment} for {year} from {year_file}")

        # Reads raster data for year
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

        # Calculates the percentile for 0 for the year (neutral, no flux) for mapping
        percentile_0 = percentile_for_0(data)
        print(f"  0 is at the {percentile_0}th percentile of the raster for {year}.")
        percentiles = [percentile_0/6, percentile_0/4, percentile_0/2, percentile_0/1.3, percentile_0/1.05,
                       percentile_0*1.05, percentile_0*1.1, percentile_0*1.2, percentile_0*1.3, percentile_0*1.5]
        # print("percentiles:", percentiles)

        print(f"  Calculating percentiles and breaks for {year}")

        # Converts RGB color palette to matplotlib color palette
        colors_matplotlib = rgb_to_mpl_palette(colors_rgb)

        # Matches percentile breaks with colors for the map.
        # Normalizes percentiles to a 0-1 scale.
        percentiles_normalized = np.linspace(0, 1, len(percentiles))
        # print("percentiles_normalized:", percentiles_normalized)
        cmap = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized, colors_matplotlib)))

        print(f"  Masking raster for {year} to non-0 values")
        masked_data = np.ma.masked_where(data == 0, data)

        # For map (not legend)
        norm = TwoSlopeNorm(
            vmin=lower_lim_all_yrs,
            vcenter=global_neutral,
            vmax=upper_lim_all_yrs
        )

        print(f"  Plotting map for {year}")
        ax, fig = create_plot()

        # Sets the ocean color
        set_ocean_color(ax)

        # Limits shapefile to focal extent (if requested)
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

        # Creates legend
        create_divergent_legend_asymmetric(fig, rounded_lower_lim_all_yrs, rounded_upper_lim_all_yrs,
                                           title_text, tick_labels,
                                           year, colors_rgb, percentiles, percentile_0)

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
    create_gif(
        out_maps_for_gif,
        output_gif_path=f"{local_gif_folder}/{gif_base_name}"
    )

    series_end_time = time.time()
    print(f"{pattern_segment} took {round(series_end_time - series_start_time)} seconds: {uu.timestr()}")


# Makes jpeg of gross fluxes
def map_gross(s3_folders, model_type, model_path_description,
              local_reproj_folder, local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
              colors_rgb, percentiles, country_shapefile, bounding_box=None, bounding_box_description=None):

    series_start_time = time.time()

    out_maps_for_gif = []

    # If bounding_box was given in degrees, transforms to match the raster CRS (Robinson)
    if bounding_box is not None:
        bounding_box_proj = transform_bbox_to_robinson(bounding_box)
    else:
        bounding_box_proj = None

    all_valid_values = []

    # First pass: Reprojects input rasters
    for i, year in enumerate(cn.years_annual[1:]):
    # for i, year in enumerate(cn.years_annual[2:3]): # For testing a specific year

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

    pattern_segment = ''

    # Second pass: Gets the range of values across years to standardize legend across years
    print("Pre-scanning rasters to determine full time series upper and lower limits...")
    for i, year in enumerate(cn.years_annual[1:]):
    # for i, year in enumerate(cn.years_annual[2:3]):

        # print(f"Scanning {year}")
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

    # Calculates min and max across all years
    all_valid_values = np.concatenate(all_valid_values)

    percentile_for_saturation = 1
    breaks_all_yrs = np.percentile(all_valid_values, [1, (100 - percentile_for_saturation)])  # The min and max percentiles at which colors saturate

    lower_lim_all_yrs = breaks_all_yrs[0]
    upper_lim_all_yrs = breaks_all_yrs[-1]

    print("Across all years:")
    print(f"  lower limit ({percentile_for_saturation} percentile):", lower_lim_all_yrs)
    print(f"  upper limit ({(100 - percentile_for_saturation)} percentile):", upper_lim_all_yrs)

    # Creates the legend in kt CO2e (converts legend units from Mg (t) to kt with 10**3-- data doesn't change).
    # Rounds data_min down and data_max up for legend.
    rounded_lower_lim_all_yrs = math.ceil(lower_lim_all_yrs / 10 ** 3 * 100) / 100  # Rounds up
    rounded_upper_lim_all_yrs = math.floor(upper_lim_all_yrs / 10 ** 3 * 100) / 100  # Rounds down

    # Legend labels depend on what exact input is displayed
    if "removals" in pattern_segment:
        tick_labels = [f"< {rounded_lower_lim_all_yrs:.0f}", 0]
        title_text = f"Gross removals\nAll vegetation pools\nkt CO$_2$ yr$^{{-1}}$"
    elif "all_gases" in pattern_segment:
        tick_labels = [0, f"> {rounded_upper_lim_all_yrs:.0f}"]
        title_text = f"Gross emissions\nAll vegetation pools, all gases\nkt CO$_2$e yr$^{{-1}}$"
    elif "non_CO2_only" in pattern_segment:
        tick_labels = [0, f"> {rounded_upper_lim_all_yrs:.0f}"]
        title_text = f"Gross emissions\nAll vegetation pools, non-CO$_2$ only\nkt CO$_2$e yr$^{{-1}}$"
    elif "CO2_only" in pattern_segment:
        tick_labels = [0, f"> {rounded_upper_lim_all_yrs:.0f}"]
        title_text = f"Gross emissions\nAll vegetation pools, CO$_2$ only\nkt CO$_2$ yr$^{{-1}}$"
    else:
        tick_labels = ["N/A", "N/A"]
        title_text = ""
        print("Can't generate tick labels")
    print(tick_labels)

    # Final pass: Iterates through modeled years to create the jpegs
    for i, year in enumerate(cn.years_annual[1:]):
    # for i, year in enumerate(cn.years_annual[2:4]): # For testing a specific year

        # The s3 folder to process for this year
        s3_folder = s3_folders[i]

        # All the components of the input s3 path
        parts = s3_folder.strip('/').split('/')

        # Gets the segment for the input interval
        interval_idx = parts.index(f"annual_intervals")
        interval_segment = parts[interval_idx + 1]

        # # Names before and after reprojection
        # year_file = f"{pattern_segment}{cn.flux_aggreg_pixel_meaning}_{interval_segment}_{cn.veg_model_version_underscore}__global"
        # year_path_unproj = f"{s3_folder}{year_file}.tif"
        #
        # # For reasons I couldn't figure out, gross removals and CO2-only emissions just wouldn't work for some files
        # # using the reprojected file names I wanted. So, these two need special reprojected file names.
        # # From https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant
        # if ("removals" in pattern_segment) or ("all_C_pools__CO2_only" in pattern_segment):
        #     year_path_reproj = f"{local_reproj_folder}/{pattern_segment}{cn.flux_aggreg_pixel_meaning}_{year}_reproj.tif"
        # else:
        #     year_path_reproj = f"{local_reproj_folder}/{year_file}_reproj.tif"

        # Names after reprojection
        year_file = f"{pattern_segment}{cn.flux_aggreg_pixel_meaning}_v{cn.veg_model_version_underscore}_{interval_segment}_global"
        year_path_reproj = f"{local_reproj_folder}/{year_file}_reproj.tif"

        print(f"\n\n---Mapping {pattern_segment} for {year} from {year_file}")

        # Reads raster data for year
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

        # Matches percentile breaks with colors.
        # Normalizes percentiles to a 0-1 scale.
        print(f"  Calculating percentiles and breaks for {year}")

        # Converts RGB color palette to matplotlib color palette
        colors_matplotlib = rgb_to_mpl_palette(colors_rgb)

        # Matches percentile breaks with colors for the map.
        # Normalizes percentiles to a 0-1 scale.
        percentiles_normalized = np.linspace(0, 1, len(percentiles))
        # print("percentiles_normalized:", percentiles_normalized)
        cmap = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized, colors_matplotlib)))

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

        print(f"  Normalizing for {year}")
        # Normalizes the data for the colormap
        norm = Normalize(vmin=lower_lim_all_yrs, vmax=upper_lim_all_yrs)

        print(f"  Plotting map for {year}")
        ax, fig = create_plot()

        # Sets the ocean color
        set_ocean_color(ax)

        # Limits shapefile to focal extent (if requested)
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


        create_unidirection_legend(fig, img, lower_lim_all_yrs, upper_lim_all_yrs,
                                   title_text, tick_labels,
                                   year, colors_rgb, percentiles)

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
    create_gif(
        out_maps_for_gif,
        output_gif_path=f"{local_gif_folder}/{gif_base_name}"
    )

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

