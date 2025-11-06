"""
Summarizes SDPTv3 country gdb layers by:
1) optionally, filters data to specific conditions (i.e. simpleType == "Planted forest")
2) gets all unique values in the summary attribute (i.e. "final_id")
3) counts the number of features for each unique value, and
4) sums total area for each unique value
Outputs a csv with final results for all countries combined

Notes:
    - Processing locally with local copy of gdb took ~1.5 min for NZL
    - Processing locally with s3 copy of gdb took ~4.5 min for NZL without GDAL s3 read options
    - Processing locally with s3 copy of gdb took ~2.25 min for NZL with GDAL s3 read options
    - Processing in coiled with s3 copy of gdb took ~1.25 min for NZL with GDAL s3 read options

There are 168 country layer in the gfw_only copy of SDPTv3. To get filtered summary stats for all countries using 20
workers took
------------------------------------------------------------------------------------------------------------------------
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

# Local test
python -m src.LULUCF.scripts.preprocessing.sdptv3_removal_factor_update.feature_count_and_area --run_local

# Coiled test
python -m src.utilities.create_cluster -cn sdpt -n 1 -m 8
python -m src.LULUCF.scripts.preprocessing.sdptv3_removal_factor_update.feature_count_and_area -cn sdpt

# Coiled run
python -m src.utilities.create_cluster -cn sdpt -n 20 -m 16
python -m src.LULUCF.scripts.preprocessing.sdptv3_removal_factor_update.feature_count_and_area -cn sdpt
"""
from __future__ import annotations
import os
import sys
import argparse
import time
import warnings
from pathlib import Path
import fiona
import geopandas as gpd
import pandas as pd
from pyproj import Geod
import operator as op
import functools
from tqdm import tqdm
import dask

# Project imports
from src.utilities import universal_utilities as uu
from src.utilities import log_utilities as lu

#-----------------------------------------------------------------------------------------------------------------------
# User inputs
#-----------------------------------------------------------------------------------------------------------------------
#gdb_path = r"/vsis3/gfw2-data/plantations/sdpt_v3/oct2025_updates/oct212025_updates/vector_gdb_gfw_only/sdpt_v3_final_gfw_only.gdb"
gdb_path = r"/mnt/c/GIS/shapefiles/SDPTv3/sdpt_v3_final_gfw_only.gdb/sdpt_v3_final_gfw_only.gdb"
output_path = r"/mnt/c/GIS/shapefiles/SDPTv3/stats/planted_forest_summary_20251103.csv"

# The main attribute to group results by
attribute_col = "final_id"

# Columns to keep in the output csv (other than the attribute_col)
keep_cols = [
    "iso3",
    "country",
    "simpleType",
    "simpleName",
    "vernacName",
    "sciName",
    "sciName1",
    "sciName2",
    "leafStatus",
    "leafType",
    "woodType"
]

# Optional: filters applied to the whole country layer before grouping
where = "simpleType = 'Planted forest' AND sciName1 <> 'Unknown' AND sciName2 IS NULL"  #TODO update everywhere instead of filters
filters = {
    # AND conditions
    "all": [
        ("simpleType", "==", "Planted forest"),
        ("sciName1", "!=", "Unknown"),
        ("sciName2", "isnull", True),
    ],
    # OR conditions
    "any": [
        # ("iso3", "in", ["NZL", "AUS"]),
    ],
    # NOT conditions
    "none": [
        # ("ownership", "contains", "private"),
    ],
}
#-----------------------------------------------------------------------------------------------------------------------
# GDAL options for faster s3 reads
#-----------------------------------------------------------------------------------------------------------------------
# Avoid directory listing / metadata storms
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_LIST_DIR"] = "NO"

# Only probe file types we actually need (helps a lot with FileGDBs)
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = (".gdb,.gdbtable,.gdbtablx,.spx,.freelist,.dat,.atx,.xml,.indexes")

# Cache upstream reads in-process
os.environ["VSI_CACHE"] = "TRUE"
os.environ["VSI_CACHE_SIZE"] = str(64 * 1024 * 1024)  # 64 MB; try 128–256MB for RAM

# Larger HTTP range request chunks (fewer GETs for faster reads)
os.environ["CPL_VSIL_CURL_CHUNK_SIZE"] = str(1000 * 1024 * 1024)

#-----------------------------------------------------------------------------------------------------------------------
# Dask config to avoid worker TTL
#-----------------------------------------------------------------------------------------------------------------------
dask.config.set({
    # allow slower S3/TCP and long native calls without tripping connection timeouts
    "distributed.comm.timeouts.connect": "120s",
    "distributed.comm.timeouts.tcp": "600s",
    "distributed.scheduler.worker-ttl": None,

    # scheduler waits longer before considering a worker unresponsive
    # (only affects this process unless also set on the scheduler at cluster creation)
    "distributed.scheduler.worker-timeout": "1800s",  # 30 minutes
})

#-----------------------------------------------------------------------------------------------------------------------
# Utilities     #TODO: Move to UU
#-----------------------------------------------------------------------------------------------------------------------
# Returns all layer names in the geodatabase
def list_gdb_layers(gdb_path):
    try:
        return list(fiona.listlayers(str(gdb_path)))
    except Exception as e:
        raise RuntimeError(f"Could not list layers in {gdb_path}.") from e

# Reads a single layer as a GeoDataFrame
def read_gdb_layer(gdb_path, layer_name):
    cols = [attribute_col] + keep_cols + ["geometry"]
    try:
        return gpd.read_file(str(gdb_path), layer=layer_name, engine="pyogrio", columns=cols, where=where, use_arrow=True)
    except Exception:
        # Fallback to default engine if pyogrio not available
        return gpd.read_file(str(gdb_path), layer=layer_name, columns=cols, where=where, use_arrow=True)

# Applies filter conditions to a geopandas GeoDataFrame. Returns a filtered copy.
def apply_gdf_filters(gdf, filter_dict):
    ops = {
        "==": lambda s, v: s == v,
        "!=": lambda s, v: s != v,
        "in": lambda s, v: s.isin(v),
        "notin": lambda s, v: ~s.isin(v),
        "contains": lambda s, v: s.astype("string").str.contains(str(v), case=True, na=False),
        "icontains": lambda s, v: s.astype("string").str.contains(str(v), case=False, na=False),
        "isnull": lambda s, v=True: s.isna() if v else s.notna(),
        ">": lambda s, v: s > v, "gt": lambda s, v: s > v,
        ">=": lambda s, v: s >= v, "ge": lambda s, v: s >= v,
        "<": lambda s, v: s < v, "lt": lambda s, v: s < v,
        "<=": lambda s, v: s <= v, "le": lambda s, v: s <= v,
    }

    def series_for(col):
        if col in gdf.columns:
            return gdf[col]
        return pd.Series(False, index=gdf.index)

    def to_mask(cond):
        col, op_name, *vals = cond
        fn = ops[op_name.lower()]
        s = series_for(col)
        return fn(s, *vals) if vals else fn(s)

    TRUE = pd.Series(True, index=gdf.index)
    FALSE = pd.Series(False, index=gdf.index)

    all_mask = functools.reduce(op.and_, (to_mask(c) for c in filter_dict.get("all", [])), TRUE) if filter_dict.get("all") else TRUE
    any_mask = functools.reduce(op.or_, (to_mask(c) for c in filter_dict.get("any", [])), FALSE) if filter_dict.get("any") else TRUE
    none_mask = ~functools.reduce(op.or_, (to_mask(c) for c in filter_dict.get("none", [])), FALSE) if filter_dict.get("none") else TRUE

    return gdf.loc[all_mask & any_mask & none_mask].copy()

# Computes geodesic area (square meters) for a single geometry using WGS84 ellipsoid.
def geodesic_area(geom):
    if geom is None or geom.is_empty:
        return 0.0

    geom_type = geom.geom_type
    if geom_type == "Polygon":
        area, _ = Geod(ellps="WGS84").geometry_area_perimeter(geom)
        return abs(area)
    elif geom_type == "MultiPolygon":
        total_area = 0.0
        for polygon in geom.geoms:
            area, _ = Geod(ellps="WGS84").geometry_area_perimeter(polygon)
            total_area += abs(area)
        return total_area
    else:
        return 0.0

# For a country layer, compute feature counts and total area grouped by the attribute column.
# Returns a dataframe with columns: attribute_col, keep_cols, feature_count, area_ha
def summarize_country_layer(gdf, layer, attribute_col, keep_cols):
    # Ensure CRS is WGS84 for geodesic computation
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Make sure attribute_col is included in grouping columns
    cols = list(dict.fromkeys([attribute_col] + list(keep_cols)))

    # Compute per-feature geodesic area (m2)
    gdf = gdf.copy()
    gdf["area_sq_m"] = gdf.geometry.apply(geodesic_area)

    # Reduce to needed columns for grouping
    missing = [c for c in cols if c not in gdf.columns]
    if missing:
        warnings.warn(f"Missing columns {missing} in layer '{layer}'. They will be omitted from grouping.")
        cols = [c for c in cols if c in gdf.columns]

    # Group by keep_cols (including attribute_col)
    agg = (gdf.groupby(cols, dropna=False, as_index=False)
        .agg(feature_count=("area_sq_m", "size"), area_sq_m=("area_sq_m", "sum"))
    )
    agg["area_ha"] = agg["area_sq_m"] / 10_000.0    # convert to hectares

    return agg[cols + ["feature_count", "area_ha"]]

def get_country_layer_stats(gdb_path, layer):

    logger_worker = lu.setup_logging_worker()

    start_time = time.time()
    try:
        gdf = read_gdb_layer(gdb_path, layer)
        if gdf is None or gdf.empty:
            lu.print_and_log(f"Layer '{layer}' is empty. Skipping.", False, logger_worker)
            return None

        # 1) Optional filters
        gdf_f = apply_gdf_filters(gdf, filters) if filters else gdf
        if gdf_f.empty:
            lu.print_and_log(f"Layer '{layer}' filtered to zero rows. Skipping.", False, logger_worker)
            return None

        # 2) Summarize/collapse by keep_cols (includes attribute_col)
        summary_df = summarize_country_layer(gdf_f, layer, attribute_col, keep_cols)
        end_time = time.time()
        lu.print_and_log(f"Finished summarizing {layer} in {round(end_time - start_time)} seconds", False, logger_worker)
        del gdf, gdf_f

        return summary_df if not summary_df.empty else None

    except Exception as e:
        warnings.warn(f"Failed to process layer '{layer}': {e}")
        return None

def main(cluster_name, run_local):

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    gdb = Path(gdb_path)
    layers = list_gdb_layers(gdb)

    # layers = layers[:1]  #TODO: comment out to run the whole GDB
    print(f"Found {len(layers)} layers in {gdb.name}. Processing…")

    start_time = time.time()
    if run_local:
        results: list[pd.DataFrame] = []
        for layer in tqdm(layers):
            country_stats = get_country_layer_stats(str(gdb), layer)
            results.append(country_stats)
    else:
        layer_futures = []
        for layer in tqdm(layers):
            layer_future = client.submit(get_country_layer_stats, str(gdb), layer, retries=3)
            layer_futures.append(layer_future)
        results = client.gather(layer_futures)
    end_time = time.time()
    print(f"Finished summarizing {len(layers)} layers in {round(end_time - start_time)} seconds")

    # Filter out None/empty and concat locally
    dfs = [df for df in results if df is not None and not df.empty]
    if not dfs:
        sys.exit("No summaries were produced. Check your columns/filters.")

    global_df = pd.concat(dfs, ignore_index=True)

    # Write CSV
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    global_df.to_csv(out_path, index=False)
    print(f"Wrote global summary to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDPTv3 feature count and area summary")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    run_local = args.run_local


    main(cluster_name, run_local)
