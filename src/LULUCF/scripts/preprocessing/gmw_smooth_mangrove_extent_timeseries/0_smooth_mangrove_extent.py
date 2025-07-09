import argparse
import numpy as np
from numba import jit
import dask
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# ---- Utility imports ----
import luigi_utils as lu
import universal_utils as uu
import numba_utils as nu
import constants as cn
import resize_cluster


@jit(nopython=True)
def smooth_mangrove_data(in_dict_uint8):
    out_dict_uint8 = {}

    m96 = in_dict_uint8["mangrove_extent_1996"]
    m07 = in_dict_uint8["mangrove_extent_2007"]
    m08 = in_dict_uint8["mangrove_extent_2008"]
    m09 = in_dict_uint8["mangrove_extent_2009"]
    m10 = in_dict_uint8["mangrove_extent_2010"]
    m15 = in_dict_uint8["mangrove_extent_2015"]
    m16 = in_dict_uint8["mangrove_extent_2016"]
    m17 = in_dict_uint8["mangrove_extent_2017"]
    m18 = in_dict_uint8["mangrove_extent_2018"]
    m19 = in_dict_uint8["mangrove_extent_2019"]
    m20 = in_dict_uint8["mangrove_extent_2020"]

    m07_out = np.zeros_like(m07)
    m08_out = np.zeros_like(m08)
    m09_out = np.zeros_like(m09)
    m10_out = np.zeros_like(m10)
    m15_out = np.zeros_like(m15)
    m16_out = np.zeros_like(m16)
    m17_out = np.zeros_like(m17)
    m18_out = np.zeros_like(m18)
    m19_out = np.zeros_like(m19)
    m20_out = np.zeros_like(m20)

    for row in range(m07.shape[0]):
        for col in range(m07.shape[1]):
            m07_out[row, col] = 1 if m07[row, col] == 1 else 0
            m08_out[row, col] = 1 if m08[row, col] == 1 or (m07[row, col] == 1 and m09[row, col] == 1) else 0
            m09_out[row, col] = 1 if m09[row, col] == 1 or (m08_out[row, col] == 1 and m10[row, col] == 1) else 0
            m10_out[row, col] = 1 if m10[row, col] == 1 else m09_out[row, col]
            #TODO add removing the false positives

            m15_out[row, col] = 1 if m15[row, col] == 1 else 0
            m16_out[row, col] = 1 if m16[row, col] == 1 or (m15[row, col] == 1 and m17[row, col] == 1) else 0
            m17_out[row, col] = 1 if m17[row, col] == 1 or (m16_out[row, col] == 1 and m18[row, col] == 1) else 0
            m18_out[row, col] = 1 if m18[row, col] == 1 or (m17_out[row, col] == 1 and m19[row, col] == 1) else 0
            m19_out[row, col] = 1 if m19[row, col] == 1 or (m18_out[row, col] == 1 and m20[row, col] == 1) else 0
            m20_out[row, col] = 1 if m20[row, col] == 1 else 0

    out_dict_uint8["mangrove_extent_1996_processed"] = m96.copy()
    out_dict_uint8["mangrove_extent_2007_processed"] = m07_out
    out_dict_uint8["mangrove_extent_2008_processed"] = m08_out
    out_dict_uint8["mangrove_extent_2009_processed"] = m09_out
    out_dict_uint8["mangrove_extent_2010_processed"] = m10_out
    out_dict_uint8["mangrove_extent_2015_processed"] = m15_out
    out_dict_uint8["mangrove_extent_2016_processed"] = m16_out
    out_dict_uint8["mangrove_extent_2017_processed"] = m17_out
    out_dict_uint8["mangrove_extent_2018_processed"] = m18_out
    out_dict_uint8["mangrove_extent_2019_processed"] = m19_out
    out_dict_uint8["mangrove_extent_2020_processed"] = m20_out

    return out_dict_uint8


def main(cluster_name, run_local=False, no_stats=False, no_log=False, no_upload=False,
         chunk_shapefile_uri=None, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):

    stage = 'starting_mangroves_1x1_deg'
    model_type = 'standard'
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)
    chunk_shapefile_uri = chunk_shapefile_uri or cn.fishnet_1x1deg_uri
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)
    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")

    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size, first_chunks, fishnet_iso_df, main_logger)

    sample_tile_id = "00N_000E"
    download_dict = {
        cn.mangrove_extent_1996_pattern: f"{cn.mangrove_extent_1996}{sample_tile_id}_{cn.mangrove_extent_1996_pattern}.tif",
        cn.mangrove_extent_2007_pattern: f"{cn.mangrove_extent_2007}{sample_tile_id}_{cn.mangrove_extent_2007_pattern}.tif",
        cn.mangrove_extent_2008_pattern: f"{cn.mangrove_extent_2008}{sample_tile_id}_{cn.mangrove_extent_2008_pattern}.tif",
        cn.mangrove_extent_2009_pattern: f"{cn.mangrove_extent_2009}{sample_tile_id}_{cn.mangrove_extent_2009_pattern}.tif",
        cn.mangrove_extent_2010_pattern: f"{cn.mangrove_extent_2010}{sample_tile_id}_{cn.mangrove_extent_2010_pattern}.tif",
        cn.mangrove_extent_2015_pattern: f"{cn.mangrove_extent_2015}{sample_tile_id}_{cn.mangrove_extent_2015_pattern}.tif",
        cn.mangrove_extent_2016_pattern: f"{cn.mangrove_extent_2016}{sample_tile_id}_{cn.mangrove_extent_2016_pattern}.tif",
        cn.mangrove_extent_2017_pattern: f"{cn.mangrove_extent_2017}{sample_tile_id}_{cn.mangrove_extent_2017_pattern}.tif",
        cn.mangrove_extent_2018_pattern: f"{cn.mangrove_extent_2018}{sample_tile_id}_{cn.mangrove_extent_2018_pattern}.tif",
        cn.mangrove_extent_2019_pattern: f"{cn.mangrove_extent_2019}{sample_tile_id}_{cn.mangrove_extent_2019_pattern}.tif",
        cn.mangrove_extent_2020_pattern: f"{cn.mangrove_extent_2020}{sample_tile_id}_{cn.mangrove_extent_2020_pattern}.tif",
    }

    output_dir_list = [
        cn.mangrove_1996_processed_dir, cn.mangrove_2007_processed_dir, cn.mangrove_2008_processed_dir,
        cn.mangrove_2009_processed_dir, cn.mangrove_2010_processed_dir, cn.mangrove_2015_processed_dir,
        cn.mangrove_2016_processed_dir, cn.mangrove_2017_processed_dir, cn.mangrove_2018_processed_dir,
        cn.mangrove_2019_processed_dir, cn.mangrove_2020_processed_dir
    ]
    output_dir_list = [p.replace("CHUNK_SIZE", str(chunk_size_pixels)).replace("PER_HA_OR_PIXEL", cn.mangrove_pixel_meaning) for p in output_dir_list]

    first_tiles = uu.first_file_name_in_s3_folder(download_dict)
    download_dict_with_data_types = uu.add_file_type_to_dict(first_tiles)
    uu.create_s3_task_files(stage, chunk_list)

    mangrove_results = [dask.delayed(uu.preprocess_and_upload_mangrove_extents)(
        chunk, download_dict_with_data_types, len(chunk_list) > 20, no_upload, output_dir_list, stage)
        for chunk in chunk_list]

    mangrove_1x1_deg_results = dask.compute(*mangrove_results)
    success_count_1x1, all_1x1_stats = uu.count_successful_chunks(chunk_list, len(chunk_list) > 20, main_logger, mangrove_1x1_deg_results)

    if not no_upload:
        for output_folder in output_dir_list:
            _, count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
            main_logger.info(f"Output rasters in {output_folder}: {count}")

    if not no_stats and success_count_1x1 > 0:
        uu.compile_1x1_chunk_stats(all_1x1_stats, chunk_shapefile_uri, stage, no_upload, main_logger)

    if not run_local:
        workers = client.scheduler_info()["workers"]
        if len(workers) > 10:
            resize_cluster.resize_coiled_cluster(cluster_name, 1)

        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smooth mangrove extent rasters")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='S3 location of chunk shapefile')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process')
    parser.add_argument('-ln', '--log_note', help='Note for log')

    parser.add_argument('--run_local', action='store_true')
    parser.add_argument('--no_stats', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--no_upload', action='store_true')
    args = parser.parse_args()

    main(
        cluster_name=args.cluster_name,
        run_local=args.run_local,
        no_stats=args.no_stats,
        no_log=args.no_log,
        no_upload=args.no_upload,
        chunk_shapefile_uri=args.chunk_shapefile_uri,
        bounding_box=args.bounding_box,
        chunk_size=args.chunk_size,
        first_chunks=args.first_chunks,
        log_note=args.log_note
    )