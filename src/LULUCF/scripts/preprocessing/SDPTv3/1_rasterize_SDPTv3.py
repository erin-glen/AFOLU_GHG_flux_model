"""
Builds a global OGR VRT that unions (concatenates) polygon feature classes from a zipped global .gdb file on S3.

Args:
    gdb_s3_path: S3 URI to the ZIP that contains multiple country .gdb directories (e.g., "s3://bucket/path/world_gdb.zip")
    output_vrt_s3_path: S3 URI where the resulting .vrt should be uploaded (e.g., "s3://bucket/path/global_union.vrt")
    local_vrt_path: Local temp path to write the VRT before upload (e.g., "/tmp/global_union.vrt")
    main_logger: your pipeline logger

Returns:
    str: Success message

Behavior:
    - Skips work if the VRT already exists at S3.
    - Opens .gdb with OGR and creates a single OGR VRT using <OGRVRTUnionLayer>.
    - Validates the VRT by opening it with OGR and ensuring it has features.
    - Uploads the VRT to S3 and removes the local temp file.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test:
python -m src.LULUCF.scripts.preprocessing.SDPTv3.1_rasterize_SDPTv3 --run_local

"""

import os
import argparse
from osgeo import ogr, gdal
import xml.etree.ElementTree as ET

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import universal_utilities as uu
from src.utilities import log_utilities as lu


try:
    FLATTEN = ogr.wkbFlatten        # some builds expose this
except AttributeError:
    FLATTEN = ogr.GT_Flatten        # Python API alias

# optional: guard constants that might not exist on older GDALs
WKB_CURVEPOLYGON    = getattr(ogr, "wkbCurvePolygon", None)
WKB_MULTISURFACE    = getattr(ogr, "wkbMultiSurface", None)

def geom_name(gtype: int) -> str:
    try:
        return ogr.GeometryTypeToName(gtype)
    except Exception:
        return str(gtype)

def is_polygonish(gtype: int) -> bool:
    base = FLATTEN(gtype)
    if base in (ogr.wkbPolygon, ogr.wkbMultiPolygon):
        return True
    if WKB_CURVEPOLYGON is not None and base == WKB_CURVEPOLYGON:
        return True
    if WKB_MULTISURFACE is not None and base == WKB_MULTISURFACE:
        return True
    return False

def build_global_union_vrt_from_gdb(gdb_s3_path, output_vrt_s3_path, local_vrt_path, main_logger,
                                    require_polygons = True, layer_name_prefix = None, include_layers = None, exclude_layers = None):
# TODO: Get rid of require_polygons, layer_name_prefix, include_layers, and exclude_layers as input arguments

    logger_worker = lu.setup_logging_worker()
    lu.print_and_log(f"Starting global union VRT build from {gdb_s3_path}: {uu.timestr('time')}", False, logger_worker)

    # Check if the VRT file already exists in S3
    if uu.vrt_exists_in_s3(output_vrt_s3_path):
        return main_logger.info(f"VRT file already exists in S3: {output_vrt_s3_path}. Skipping creation.")

    # STEP 1: Convert s3 path to VSI path for OGR
    #TODO: Update so that it can also take a local path
    vsis3_path = gdb_s3_path.replace("s3://", "/vsis3/")

    # Use vsizip along with vsis3 to access the compressed global SDPTv3 geodatabase in s3
    # Example: s3://gfw2-data/plantations/sdpt_v3/sdpt_v3_final.gdb.zip -> /vsizip//vsis3/gfw2-data/plantations/sdpt_v3/sdpt_v3_final.gdb.zip
    # Note: OGR can only open the zip root if it contains a single .gdb
    if gdb_s3_path.lower().endswith(".zip"):
        vsi_zip_root = f"/vsizip/{vsis3_path}"
        print(f"vsi_zip_root: {vsi_zip_root}")

        SDPTv3 = None
        vsi_path = None

        # 1) Try opening the ZIP root directly (works if it contains a single .gdb)
        ds_try = ogr.Open(vsi_zip_root)
        if ds_try is not None:
            SDPTv3 = ds_try
            vsi_path = vsi_zip_root

        # 2) If that failed, guess the inner .gdb directory from the zip filename
        if SDPTv3 is None:
            base_no_zip = os.path.basename(gdb_s3_path)[:-4]  # strip ".zip", e.g. "sdpt_v3_final.gdb"
            vsi_guess = f"{vsi_zip_root.rstrip('/')}/{base_no_zip}"
            print(f"vsi_guess_inner: {vsi_guess}")
            ds_try = ogr.Open(vsi_guess)
            if ds_try is not None:
                SDPTv3 = ds_try
                vsi_path = vsi_guess

        # 3) Final fallback: list the ZIP root once (non-recursive) and try the first *.gdb entry
        if SDPTv3 is None:
            root_entries = gdal.ReadDir(vsi_zip_root) or []
            gdb_candidates = [e for e in root_entries if e.lower().endswith(".gdb")]
            print(f"zip root entries: {root_entries}")
            print(f"gdb_candidates: {gdb_candidates}")
            if not gdb_candidates:
                raise RuntimeError("No .gdb directory found inside the ZIP (couldn’t auto-detect).")
            vsi_path = f"{vsi_zip_root.rstrip('/')}/{gdb_candidates[0]}"
            SDPTv3 = ogr.Open(vsi_path)
            if SDPTv3 is None:
                raise RuntimeError(f"Found inner GDB '{gdb_candidates[0]}' but could not open it at {vsi_path}")

    # Otherwise, use only vsis3 if global SDPTv3 geodatabase is uncompressed on s3
    # Example: s3://gfw2-data/plantations/sdpt_v3/sdpt_v3_final.gdb -> /vsis3/gfw2-data/plantations/sdpt_v3/sdpt_v3_final.gdb
    elif gdb_s3_path.lower().endswith(".gdb"):
        vsi_path = vsis3_path
        SDPTv3 = ogr.Open(vsi_path)
        if SDPTv3 is None:
            raise RuntimeError(f"Could not open FileGDB at {vsi_path}")
    else:
        raise ValueError("gdb_s3_path must end with '.gdb' (directory) or '.zip' (zip containing a .gdb).")

    print(f"Opened datasource at {vsi_path}; driver={SDPTv3.GetDriver().GetName()} layers={SDPTv3.GetLayerCount()}")

# Step 3: Inspect the GDB to list all layers and polygon types
    layers_info = []
    unique_geom_type_names = set()
    unique_polygon_type_names = set()
    #TODO: Comment out these 3 lines after inspecting gdb

    include_layers = set(include_layers) if include_layers else None
    exclude_layers = set(exclude_layers) if exclude_layers else set()

    sources = []  # (datasource_path, layer_name)
    total_seen = 0
    total_added = 0

    polygon_types = {ogr.wkbPolygon, ogr.wkbMultiPolygon, ogr.wkbPolygon25D, ogr.wkbMultiPolygon25D}

    for i in range(SDPTv3.GetLayerCount()):

        layer = SDPTv3.GetLayerByIndex(i)

        if layer is None:
            main_logger.info(f"Skipping null layer: {layer}")
            continue

        total_seen += 1
        layer_name = layer.GetName()
        main_logger.info(f"Inspecting layer: {layer_name}")

        # TODO: After inspecting gdb, comment out from here...
        gtype = layer.GetGeomType()
        gname = geom_name(gtype)

        layers_info.append({"name": layer_name, "geom_type": gname})
        unique_geom_type_names.add(gname)

        if is_polygonish(gtype):
            unique_polygon_type_names.add(geom_name(FLATTEN(gtype)))
            unique_polygon_type_names.add(geom_name(gtype))
        # TODO: ... to here

        # Name filters
        if include_layers is not None and layer_name not in include_layers:
            continue
        if layer_name in exclude_layers:
            continue
        if layer_name_prefix and not layer_name.startswith(layer_name_prefix):
            continue

        # Geometry filter
        if require_polygons:
            gtype = layer.GetGeomType()
            if not is_polygonish(gtype):
                continue
        sources.append((vsi_path, layer_name))
        total_added += 1

    # TODO: After inspecting gdb, comment out from here...
    # Log summaries
    all_layer_names = [d["name"] for d in layers_info]
    lu.print_and_log(f"All layers in GDB ({len(all_layer_names)}): {all_layer_names}", False, logger_worker)
    lu.print_and_log(f"Unique geometry types: {sorted(unique_geom_type_names)}", False, logger_worker)
    lu.print_and_log(f"Unique polygon geometry types: {sorted(unique_polygon_type_names)}", False, logger_worker)

    # (Optional) also push to main_logger
    main_logger.info(f"Layers found: {len(all_layer_names)}")
    main_logger.info(f"Unique geom types: {sorted(unique_geom_type_names)}")
    main_logger.info(f"Unique polygon geom types: {sorted(unique_polygon_type_names)}")
    # TODO: ... to here

    SDPTv3 = None
    main_logger.info(f"Found {total_seen} layer(s) in GDB; adding {total_added} layer(s) to union VRT.")

    if not sources:
        raise RuntimeError("No eligible polygon feature classes found in the GDB (check filters).")

    # Step 4: Build OGR VRT with OGRVRTUnionLayer
    vrt_SDPTv3 = ET.Element("OGRVRTDataSource")
    union = ET.SubElement(vrt_SDPTv3, "OGRVRTUnionLayer", name="global_union_polygons")
    ET.SubElement(union, "FieldStrategy").text = "Union"  # union field schemas

    for src_SDPTv3_path, src_layer_name in sources:
        vlayer = ET.SubElement(union, "OGRVRTLayer", name=f"{os.path.basename(src_SDPTv3_path)}__{src_layer_name}")
        src = ET.SubElement(vlayer, "SrcDataSource")
        src.text = src_SDPTv3_path
        ET.SubElement(vlayer, "SrcLayer").text = src_layer_name

    # Step 5: Write VRT locally
    tree = ET.ElementTree(vrt_SDPTv3)
    try:
        ET.indent(tree, space="  ", level=0)  # pretty (py>=3.9)
    except Exception:
        pass
    tree.write(local_vrt_path, encoding="UTF-8", xml_declaration=True)
    lu.print_and_log(f"Wrote union VRT: {local_vrt_path} — {uu.timestr('time')}", True, logger_worker)

    # Step 6: Validate by opening VRT
    vrt_check = ogr.Open(local_vrt_path)
    if vrt_check is None:
        raise RuntimeError("Generated VRT could not be opened.")
    vlayer = vrt_check.GetLayerByName("global_union_polygons")
    if vlayer is None:
        raise RuntimeError("Union layer 'global_union_polygons' missing in VRT.")
    try:
        _ = vlayer.GetFeatureCount()  # may trigger remote IO
    except Exception:
        main_logger.warning("Could not count features on the VRT; continuing.")

    vrt_check = None

    # Step 7: Upload + cleanup
    uu.upload_s3_file(output_vrt_s3_path, local_vrt_path)
    if uu.check_s3_file_created(output_vrt_s3_path, main_logger):
        try:
            os.remove(local_vrt_path)
            if not os.path.exists(local_vrt_path):
                main_logger.info(f"Deleted local VRT file: {local_vrt_path}")
        except Exception as e:
            main_logger.warning(f"Error deleting local VRT file: {local_vrt_path} — {e}")

    lu.print_and_log(f"Uploaded union VRT to {output_vrt_s3_path}: {uu.timestr('time')}", True, logger_worker)
    return f"Success: union VRT uploaded to {output_vrt_s3_path}"


def main(cluster_name, run_local):

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)
    client

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, f"Rasterizing SDPTv3",
                                                        run_local,'standard', f'Rasterizing SDPTv3')

    # TODO: Update in constants and names
    gdb_s3_path = "s3://gfw2-data/plantations/sdpt_v3/sdpt_v3_final.gdb.zip"
    out_vrt_s3_path = "s3://gfw2-data/plantations/sdpt_v3/SDPTv3.vrt"
    local_vrt_path = "/tmp/SDPTv3.vrt"


    #STEP 1: Create VRT of unionized polygons for SDPTv3
    main_logger.info(f"Submitting VRT build for SDPTv3: {uu.timestr('time')}\n")
    if not run_local:
        vrt_future = client.submit(build_global_union_vrt_from_gdb, gdb_s3_path, out_vrt_s3_path, local_vrt_path, main_logger, require_polygons=True)
        vrt_future.result()
    else:
        build_global_union_vrt_from_gdb(gdb_s3_path, out_vrt_s3_path, local_vrt_path, main_logger, require_polygons = True)

    # Closes the client if not running locally
    if not run_local:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create SDPTv3 tiles")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--run_date', help='Date of run, in YYYYMMDD')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    #run_date = args.run_date
    #bounding_box = args.bounding_box
    #fishnet_path = args.cshp

    tile_id_field = "chunk_id"

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload


    main(cluster_name, run_local)