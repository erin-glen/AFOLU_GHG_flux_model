import os
import fiona
import geopandas as gpd
import pandas as pd
from shapely.geometry import box


def get_layer_bounds(gdb_path, layer_name):
    """
    Return the bounding box (minx, miny, maxx, maxy) for a layer in a FileGDB,
    using Fiona's metadata for the layer (src.bounds).

    This approach does NOT read all geometries, so it's much faster for large layers.
    Returns None if there's an error or the layer has no valid bounds.
    """
    try:
        with fiona.open(gdb_path, layer=layer_name) as src:
            # src.bounds => (minx, miny, maxx, maxy) from metadata
            if src.bounds is not None:
                return src.bounds
            else:
                return None
    except Exception as e:
        print(f"Error reading layer '{layer_name}' in GDB '{gdb_path}': {e}")
        return None


def bounding_boxes_intersect(b1, b2):
    """
    b1, b2: (minx, miny, maxx, maxy)
    Returns True if bounding boxes overlap.
    """
    return not (
        b2[0] > b1[2] or  # b2.minx > b1.maxx
        b2[2] < b1[0] or  # b2.maxx < b1.minx
        b2[1] > b1[3] or  # b2.miny > b1.maxy
        b2[3] < b1[1]     # b2.maxy < b1.miny
    )


def split_sdpt_by_tiles(tile_grid_path,
                        countries_gdb,
                        out_dir,
                        tile_id_field="tile_id"):
    """
    Reads a tile grid (shapefile or other supported format), loops over each tile,
    then clips relevant country layers (i.e., those that pass a bounding-box check)
    in a file geodatabase. Writes out tile-based shapefiles.
    """

    # 1) Load tile grid
    print(f"Reading tile grid: {tile_grid_path}")
    tiles_gdf = gpd.read_file(tile_grid_path)
    if tile_id_field not in tiles_gdf.columns:
        raise ValueError(f"Tile grid missing field '{tile_id_field}'.")

    # 2) Identify layers within the GDB and get bounding boxes from metadata
    layer_names = fiona.listlayers(countries_gdb)
    print(f"Found {len(layer_names)} layers in '{countries_gdb}'.")

    country_bb_list = []
    for layer_name in layer_names:
        layer_bounds = get_layer_bounds(countries_gdb, layer_name)
        if layer_bounds is None:
            print(f"Warning: Could not read bounding box for layer '{layer_name}'. Skipping.")
            continue

        country_bb_list.append({
            "layer_name": layer_name,
            "bounds": layer_bounds  # (minx, miny, maxx, maxy)
        })

    # Convert to DataFrame for convenience
    country_bb_df = pd.DataFrame(country_bb_list)

    # 3) For each tile, clip polygons from relevant GDB layers
    for idx, tile_row in tiles_gdf.iterrows():
        tile_geom = tile_row.geometry
        tile_id = tile_row[tile_id_field]
        tile_bounds = tile_geom.bounds  # (minx, miny, maxx, maxy)

        # Collect all clipped polygons across layers
        all_clips = []

        # 3.1) Filter layers whose bounding box intersects this tile's bounding box
        relevant_layers = country_bb_df[
            country_bb_df["bounds"].apply(lambda b: bounding_boxes_intersect(b, tile_bounds))
        ]

        if len(relevant_layers) == 0:
            print(f"Tile {tile_id}: no intersecting GDB layers. Skipping.")
            continue

        # 3.2) For each relevant layer, read & clip
        for idx2, row2 in relevant_layers.iterrows():
            layer_name = row2["layer_name"]
            print(f" - Clipping tile {tile_id} with layer '{layer_name}'")
            try:
                # Now we read ALL geometries only for layers that intersect
                ctry_gdf = gpd.read_file(countries_gdb, layer=layer_name)
                if ctry_gdf.empty:
                    continue
                # Clip to tile geometry
                clipped = gpd.clip(ctry_gdf, tile_geom)
                if not clipped.empty:
                    clipped["source_layer"] = layer_name
                    all_clips.append(clipped)
            except Exception as e:
                print(f"   Error clipping layer '{layer_name}' for tile {tile_id}: {e}")

        # 3.3) Merge and write out
        if len(all_clips) > 0:
            merged_clips = gpd.GeoDataFrame(
                pd.concat(all_clips, ignore_index=True),
                crs=all_clips[0].crs
            )
            os.makedirs(out_dir, exist_ok=True)
            out_name = f"tile_{tile_id}.shp"
            tile_out_path = os.path.join(out_dir, out_name)

            if os.path.exists(tile_out_path):
                os.remove(tile_out_path)

            print(f"Writing {len(merged_clips)} features to {tile_out_path}")
            merged_clips.to_file(tile_out_path)
        else:
            print(f"Tile {tile_id}: no features after clipping. Skipping write.")


def main():
    # Hardcode or parameterize the paths
    tile_grid_path = r"C:\tmp\Global_Peatlands.shp"
    countries_gdb = r"C:\GIS\Data\Global\Plantation\sdpt_v21_v09152024_public.gdb\sdpt_v21_v09152024_public.gdb"
    out_dir = r"C:\GIS\Data\Global\Plantation\sdpt_by_tiles"
    tile_id_field = "tile_id"  # Adapt to your tile grid schema

    split_sdpt_by_tiles(
        tile_grid_path=tile_grid_path,
        countries_gdb=countries_gdb,
        out_dir=out_dir,
        tile_id_field=tile_id_field
    )


if __name__ == "__main__":
    main()
