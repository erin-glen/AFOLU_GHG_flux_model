import os
import subprocess

# Folder containing your raster tiles
input_folder = r"C:\tmp\soil_outputs_10x10_30m"

# Output files
vrt_file = r"C:\tmp\soil_outputs_10x10_30m\tmp.vrt"
merged_tif = r"C:\tmp\soil_outputs_10x10_30m\merged_raster.tif"

# Build VRT (virtual raster mosaic)
subprocess.run([
    'gdalbuildvrt',
    vrt_file,
    os.path.join(input_folder, '*.tif')
], check=True)

# Translate VRT to GeoTIFF with BIGTIFF option
subprocess.run([
    'gdal_translate',
    '-of', 'GTiff',
    '-co', 'COMPRESS=LZW',
    '-co', 'BIGTIFF=YES',
    vrt_file,
    merged_tif
], check=True)

print(f"Merged raster created at {merged_tif}")
