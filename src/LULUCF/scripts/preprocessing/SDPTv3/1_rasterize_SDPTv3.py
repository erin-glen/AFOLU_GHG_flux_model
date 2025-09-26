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


Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test:
python -m src.LULUCF.scripts.preprocessing.SDPTv3.1_rasterize_SDPTv3 --run_local

python -m src.utilities.create_cluster -n 1 -t 1 -m 8 -cn SDPTv3
python -m src.LULUCF.scripts.preprocessing.SDPTv3.1_rasterize_SDPTv3 -cn SDPTv3

"""

import os
import argparse
from osgeo import ogr, gdal, osr
import xml.etree.ElementTree as ET
import shutil
import tempfile

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import universal_utilities as uu
from src.utilities import log_utilities as lu

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS",".gdb,.gdbtable,.gdbtablx,.gdbindexes,.spx,.freelist,.zip,.vrt,.gpkg,.shp,.dbf,.shx,.prj,.cpg")
os.environ.setdefault("CPL_VSIL_CURL_CHUNK_SIZE", "16777216")  # 16 MB (8–32MB is a good range, larger reads reduce round trips)

########################################################################################################################
# Helper Utilities
########################################################################################################################
#Create vsis3 path from s3 path
def vsis3_from_s3(path: str) -> str:
    return path.replace("s3://", "/vsis3/")

#Checks to see if tile has at least one feature. Uses fast count when available, otherwise finds a single feature.
def layer_has_any_feature(layer):
    try:
        if layer.TestCapability(ogr.OLCFastFeatureCount):
            return layer.GetFeatureCount() > 0
    except Exception:
        pass
    layer.ResetReading()
    f = layer.GetNextFeature()
    layer.ResetReading()
    return f is not None

#Creates bounding box for tile clip
def bbox_wkt(w, s, e, n):
    return f"POLYGON(({w} {s}, {w} {n}, {e} {n}, {e} {s}, {w} {s}))"

# Cleans up all lical files from shp tile creation
def clean_local_shp_tmp(shp_dir, tmp_dir, main_logger):
    try:
        shutil.rmtree(shp_dir, ignore_errors=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception as e:
        main_logger.warning(f"Cleanup warning: {e}")
    return



########################################################################################################################
# Main Utilities
########################################################################################################################
# Function to build a global union VRT for the entire SDPTv3 gdb
    # Downloads and unzips SDPTv3 .gdb from s3. Opens .gdb with OGR and creates a single OGR VRT using <OGRVRTUnionLayer>.
    # Uploads the VRT to S3 and removes the local temp file. Note: Skips this step if the VRT already exists in s3.
def build_global_union_vrt_from_gdb(gdb_s3_path, output_vrt_s3_path, main_logger):

    logger_worker = lu.setup_logging_worker()
    lu.print_and_log(f"Starting global union VRT build from {gdb_s3_path}: {uu.timestr('time')}", False, logger_worker)

    # Check if the VRT file already exists in S3. If so, skip VRT creation.
    if uu.exists_in_s3(output_vrt_s3_path):
        lu.print_and_log(f"VRT file already exists in S3: {output_vrt_s3_path}. Skipping creation.", False, logger_worker)
        return

    # Step 1: Convert s3 path to VSI path for OGR and opens file to build VRT schema (.zip or .gdb only)
    vsis3_path = gdb_s3_path.replace("s3://", "/vsis3/")

    # Use vsizip and vsis3 to access the compressed (.zip) SDPTv3 .gdb in s3
    # Note: OGR can only open the zip root if it contains a single .gdb
    #TODO: Only keep which step worked for zip
    SDPTv3 = None
    vsi_path = None
    if gdb_s3_path.lower().endswith(".zip"):
        vsi_zip_root = f"/vsizip/{vsis3_path}"
        ds_try = ogr.Open(vsi_zip_root)
        if ds_try is not None:
            SDPTv3 = ds_try
            vsi_path = vsi_zip_root
    # Otherwise, use only vsis3 if SDPTv3 .gdb is uncompressed on s3
    elif gdb_s3_path.lower().endswith(".gdb"):
        vsi_path = vsis3_path
        SDPTv3 = ogr.Open(vsi_path)
        if SDPTv3 is None:
            raise RuntimeError(f"Could not open FileGDB at {vsi_path}")
    else:
        raise ValueError("gdb_s3_path must end with '.gdb' (directory) or '.zip' (zip containing a .gdb).")

    print(f"Opened datasource at {vsi_path}; driver={SDPTv3.GetDriver().GetName()} layers={SDPTv3.GetLayerCount()}")

    # Step 2: Inspect each layer in the .gdb and add to VRT sources
    sources = []    # (datasource_path, layer_name)
    total_added = 0

    # There are currently 164 layers in sdptv3
    for i in range(SDPTv3.GetLayerCount()):
        layer = SDPTv3.GetLayerByIndex(i)
        layer_name = layer.GetName()
        sources.append((vsi_path, layer_name))
        total_added += 1
        lu.print_and_log(f"Added layer: {layer_name}", True, logger_worker)
    SDPTv3 = None   # close file
    main_logger.info(f"Added {total_added} layer(s) to union VRT.")

    if not sources:
        raise RuntimeError("No eligible polygon feature classes found in the GDB (check filters).")

    # Step 3: Build global union VRT with OGRVRTUnionLayer
    vrt_SDPTv3 = ET.Element("OGRVRTDataSource")
    union = ET.SubElement(vrt_SDPTv3, "OGRVRTUnionLayer", name="global_union_polygons")
    ET.SubElement(union, "FieldStrategy").text = "Union"

    for src_SDPTv3_path, src_layer_name in sources:
        vlayer = ET.SubElement(union, "OGRVRTLayer", name=f"{os.path.basename(src_SDPTv3_path)}__{src_layer_name}")
        src = ET.SubElement(vlayer, "SrcDataSource")
        src.text = src_SDPTv3_path
        ET.SubElement(vlayer, "SrcLayer").text = src_layer_name

    # Step 4: Write VRT locally
    local_vrt_path = "/tmp/sdpt_v3.vrt"
    tree = ET.ElementTree(vrt_SDPTv3)
    try:
        ET.indent(tree, space="  ", level=0)
    except Exception:
        pass
    tree.write(local_vrt_path, encoding="UTF-8", xml_declaration=True)
    lu.print_and_log(f"Wrote union VRT at {local_vrt_path}", True, logger_worker)

    # Step 5: Upload vrt to s3, check that it exists and then cleanup local files
    uu.upload_s3_file(output_vrt_s3_path, local_vrt_path)
    if uu.check_s3_file_created(output_vrt_s3_path, main_logger):
        try:
            os.remove(local_vrt_path)
            if not os.path.exists(local_vrt_path):
                lu.print_and_log(f"Deleted local VRT file: {local_vrt_path}", True, logger_worker)
        except Exception as e:
            lu.print_and_log(f"Error deleting local VRT file: {local_vrt_path} — {e}", False, logger_worker)

    lu.print_and_log(f"Uploaded union VRT to {output_vrt_s3_path}: {uu.timestr('time')}", False, logger_worker)

    return


# Function to create either 1 x 1 degree or 10 x 10 degree shapefiles from the SDPTv3 union VRT. Because all country
# layers have already been unioned togehter, all polygons (regardless of country) will be included in the output tile.
def clip_vrt_to_bbox_shapefile(vrt_s3_path, tile_id, west, south, east, north, out_s3_prefix, main_logger, overwrite = True, keep_fields=None):
    
    logger_worker = lu.setup_logging_worker()
    lu.print_and_log(f"Starting SDPTv3 shapefile creation for {tile_id}: [{west}, {south}, {east}, {north}]: {uu.timestr('time')}", False, logger_worker)

    #Step 1: If overwrite is False, checks if all files for tile already exist in s3. If so, skips tile creation.
    exts = [".shp", ".shx", ".dbf", ".prj", ".cpg"]
    tile_name = f"{tile_id}_sdptv3"

    if not overwrite:
        files = [f"{out_s3_prefix}{tile_name}{ext}" for ext in exts]
        check = all(uu.exists_in_s3(f) for f in files)
        if check:
            lu.print_and_log(f"SDPTv3 shapefile for {tile_id} already exists in S3 for. Skipping creation.", False, logger_worker)
            return

    # Step 2: Open VRT and make sure union layer exists
    layer_name = "global_union_polygons"
    vsi_vrt = vrt_s3_path.replace("s3://", "/vsis3/")
    src_ds = ogr.Open(vsi_vrt)
    if src_ds is None:
        raise RuntimeError(f"Could not open VRT: {vsi_vrt}")

    src_lyr = src_ds.GetLayerByName(layer_name)
    if src_lyr is None:
        raise RuntimeError(f"Layer '{layer_name}' not found in VRT.")
    src_ds = None  # close file

    # Step 3: Create temp workspace for tile
    tmp_dir = tempfile.mkdtemp(prefix="vrtclip_")
    shp_dir = os.path.join(tmp_dir, tile_name)
    os.makedirs(shp_dir, exist_ok=True)
    shp_path = os.path.join(shp_dir, f"{tile_name}.shp")

    #This will overwrite any existing local files for a tile if they already exist locally
    for f in os.listdir(shp_dir):
        if f.startswith(tile_name):
            try:
                os.remove(os.path.join(shp_dir, f))
            except:
                pass

    # Step 4: Clip vectors to counding box
    clip_wkt = f"POLYGON(({west} {south}, {west} {north}, {east} {north}, {east} {south}, {west} {south}))"

    select_fields = None
    if keep_fields:
        src_ds = ogr.Open(vsi_vrt)
        lyr = src_ds.GetLayerByName(layer_name)
        have = {lyr.GetLayerDefn().GetFieldDefn(i).GetName()
                for i in range(lyr.GetLayerDefn().GetFieldCount())}
        select_fields = [f for f in keep_fields if f in have]
        src_ds = None

    opts = gdal.VectorTranslateOptions(
        format="ESRI Shapefile",
        spatSRS="EPSG:4326",
        spatFilter=[west, south, east, north],  # fast preselect
        dstSRS="EPSG:4326",
        reproject=True,
        clipDst=clip_wkt,  # geometric clip
        layers=["global_union_polygons"],
        selectFields=select_fields,  # <— keep only what you need
        layerCreationOptions=["ENCODING=UTF-8"],
    )
    out_ds = gdal.VectorTranslate(shp_path, vsi_vrt, options=opts)
    if out_ds is not None:
        out_ds = None   # close

    # Checks to see if the tile contains any features
    empty = True
    check = ogr.Open(shp_path)
    if check:
        layer_0 = check.GetLayer(0)
        if layer_0 and layer_has_any_feature(layer_0):
            empty = False
        check = None  # close

    # If nothing in this tile, do not upload and clean up local temp
    if empty:
        clean_local_shp_tmp(shp_dir, tmp_dir, main_logger)
        lu.print_and_log(f"No features in {tile_id}; skipping upload.: {uu.timestr('time')}", False, logger_worker)
        return

    # Make sure shp is projected into WGS84
    srs = osr.SpatialReference();
    srs.ImportFromEPSG(4326)
    with open(os.path.join(shp_dir, f"{tile_name}.prj"), "w", encoding="utf-8") as fp:
        fp.write(srs.ExportToWkt())

    # Step 5: Upload all related files to s3 and check that they have been successfully uploaded
    uploaded = []
    for ext in exts:
        p = os.path.join(shp_dir, f"{tile_name}{ext}")
        if os.path.exists(p):
            s3_key = out_s3_prefix + os.path.basename(p)
            uu.upload_s3_file(s3_key, p)
            uu.check_s3_file_created(s3_key, main_logger)
            uploaded.append(s3_key)

    # Step 6: Cleanup local files
    clean_local_shp_tmp(shp_dir, tmp_dir, main_logger)
    lu.print_and_log(f"Uploaded shapefile for {tile_id} ({len(uploaded)} files) to {out_s3_prefix}: {uu.timestr('time')}", True, logger_worker)
    return


def main(cluster_name, bounding_box = None, chunk_size = None, run_local = False):

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)
    client

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, f"Rasterizing SDPTv3",
                                                        run_local,'standard', f'Rasterizing SDPTv3')
    # TODO: Update in constants and names
    gdb_s3_path = "s3://gfw2-data/plantations/sdpt_v3/sdpt_v3_final.gdb.zip" #TODO: Point to .dgb instead of .zip to help speed things up
    out_vrt_s3_path = "s3://gfw2-data/plantations/sdpt_v3/vrt/sdpt_v3.vrt" #TODO: CHANGE BACK
    out_shp_s3_path = "s3://gfw2-data/plantations/sdpt_v3/sdpt_v3_vector_tiles/tiles_10x10/"

    tile_id = '-71_-55_-70_-54'
    west = -71
    south = -54
    east = -70
    north = -55
    #TODO: Remove after fishnet logic

    #STEP 1: Create a global VRT of unionized polygons for SDPTv3
    main_logger.info(f"STEP 1: Submitting VRT build for SDPTv3: {uu.timestr('time')}\n")
    if not run_local:
        vrt_future = client.submit(build_global_union_vrt_from_gdb, gdb_s3_path, out_vrt_s3_path, main_logger)
        vrt_future.result()
    else:
        build_global_union_vrt_from_gdb(gdb_s3_path, out_vrt_s3_path, main_logger)
    main_logger.info(f"STEP 1: VRT creation complete: {uu.timestr('time')}\n")

    #STEP 2: Create SDPTv3 shapefile tiles using fishnet
    # if chunk_size == 1:
    #     chunk_shapefile_uri = cn.fishnet_1x1deg_uri
    #     tile_id_field = "chunk_id"
    # elif chunk_size == 10:
    #     #chunk_shapefile_uri = hansen_tile_footprint
    #     tile_id_field = "Name"
    #     tile_id =
    #     #grab only the last 8 chars --> Hansen_GFC2014_treecover2000_00N_000E
    #     #west, south, east, north = uu.get_10x10_tile_bounds(tile_id)

    overwrite = True

    if not run_local:
        shp_future = client.submit(clip_vrt_to_bbox_shapefile, out_vrt_s3_path, tile_id, west, south, east, north, out_shp_s3_path, main_logger, overwrite)
        shp_future.result()
    else:
        clip_vrt_to_bbox_shapefile(out_vrt_s3_path, tile_id, west, south, east, north, out_shp_s3_path, main_logger, overwrite)



    # Closes the client if not running locally
    if not run_local:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create SDPTv3 vector tiles")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    #bounding_box = args.bounding_box
    #fishnet_path = args.cshp




    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload


    main(cluster_name, run_local)