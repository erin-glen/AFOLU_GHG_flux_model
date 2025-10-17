import os
import fiona
import geopandas as gpd
import pandas as pd
from packaging.version import Version, InvalidVersion
import geopandas.io.file

# --- Robust GDAL Version Patch for ArcGIS Pro Environment ---
# This patch resolves the "Invalid version: '3.8.1e'" error that arises due to ArcGIS Pro's non-standard GDAL versioning.
original_to_file_fiona = geopandas.io.file._to_file_fiona

def patched_to_file_fiona(df, filename, driver, schema, crs=None, mode="w", **kwargs):
    try:
        gdal_version = fiona.env.get_gdal_release_name()
        Version(gdal_version)
    except InvalidVersion:
        gdal_version = "3.8.1"
        kwargs['GEOPANDAS_FIX_GDAL_VERSION'] = gdal_version

    original_version_class = geopandas.io.file.Version
    geopandas.io.file.Version = lambda _: Version("3.8.1")

    try:
        return original_to_file_fiona(df, filename, driver, schema, crs, mode, **kwargs)
    finally:
        geopandas.io.file.Version = original_version_class

geopandas.io.file._to_file_fiona = patched_to_file_fiona
# --- End GDAL Version Patch ---

def get_layer_bounds(gdb_path, layer_name):
    """
    Quickly obtain bounding box from layer metadata.
    """
    try:
        with fiona.open(gdb_path, layer=layer_name) as src:
            return src.bounds if src.bounds else None
    except Exception as e:
        print(f"Error reading layer '{layer_name}': {e}")
        return None

def bounding_boxes_intersect(b1, b2):
    """
    Check if two bounding boxes intersect.
    """
    return not (b2[0] > b1[2] or b2[2] < b1[0] or b2[1] > b1[3] or b2[3] < b1[1])

def filter_to_polygons(gdf):
    """
    Keep only polygon and multipolygon geometries for shapefile compatibility.
    """
    return gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

def split_sdpt_by_tiles(tile_grid_path, countries_gdb, out_dir, tile_id_field="tile_id"):
    """
    Processes SDPT polygons by tiles from GDB layers.
    """
    print(f"Reading tile grid: {tile_grid_path}")
    tiles_gdf = gpd.read_file(tile_grid_path)
    if tile_id_field not in tiles_gdf.columns:
        raise ValueError(f"Tile grid missing field '{tile_id_field}'.")

    layer_names = fiona.listlayers(countries_gdb)
    print(f"Found {len(layer_names)} layers in '{countries_gdb}'.")

    country_bb_df = pd.DataFrame([
        {"layer_name": layer_name, "bounds": get_layer_bounds(countries_gdb, layer_name)}
        for layer_name in layer_names
        if get_layer_bounds(countries_gdb, layer_name) is not None
    ])

    for idx, tile_row in tiles_gdf.iterrows():
        tile_geom = tile_row.geometry
        tile_id = tile_row[tile_id_field]
        tile_bounds = tile_geom.bounds

        os.makedirs(out_dir, exist_ok=True)
        tile_out_path = os.path.join(out_dir, f"tile_{tile_id}.shp")

        if os.path.exists(tile_out_path):
            print(f"Tile {tile_id}: already exists. Skipping.")
            continue

        relevant_layers = country_bb_df[
            country_bb_df["bounds"].apply(lambda b: bounding_boxes_intersect(b, tile_bounds))
        ]

        if relevant_layers.empty:
            print(f"Tile {tile_id}: no intersecting layers. Skipping.")
            continue

        all_clips = []

        for _, layer_row in relevant_layers.iterrows():
            layer_name = layer_row["layer_name"]
            print(f" - Processing tile {tile_id} with layer '{layer_name}'")

            try:
                ctry_gdf = gpd.read_file(countries_gdb, layer=layer_name, bbox=tile_bounds)
                if ctry_gdf.empty:
                    print(f"   Layer '{layer_name}' has no features in bbox. Skipping.")
                    continue

                clipped = gpd.clip(ctry_gdf, tile_geom)

                if not clipped.empty:
                    clipped["source_layer"] = layer_name
                    all_clips.append(clipped)

            except Exception as e:
                print(f"   Error processing layer '{layer_name}': {e}")

        if all_clips:
            merged_gdf = gpd.GeoDataFrame(pd.concat(all_clips, ignore_index=True), crs=all_clips[0].crs)

            # --- NEW FIX: Only polygons allowed for shapefile format ---
            merged_gdf = filter_to_polygons(merged_gdf)

            if merged_gdf.empty:
                print(f"Tile {tile_id}: no polygon features after filtering. Skipping write.")
                continue

            print(f"Writing {len(merged_gdf)} polygon features to {tile_out_path}")
            merged_gdf.to_file(tile_out_path)
        else:
            print(f"Tile {tile_id}: no features after clipping.")

def main():
    tile_grid_path = r"C:\tmp\peat_index\Global_Peatlands_project.shp"
    countries_gdb = r"C:\tmp\sdpt_v3_final.gdb\sdpt_v3_final.gdb"
    out_dir = r"C:\GIS\Data\Global\Plantation\sdpt_by_tiles"
    tile_id_field = "tile_id"

    split_sdpt_by_tiles(tile_grid_path, countries_gdb, out_dir, tile_id_field)

if __name__ == "__main__":
    main()
