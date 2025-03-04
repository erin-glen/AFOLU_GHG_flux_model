"""

This document describes the steps used to process MODIS burned area monthly h-v hdf stacks into annual burned area geotif composites.
There are several steps to the workflow, handled in different scripts.

---Step 0: Transfer hdfs for relevant years from DAAC to s3. Download to computer first, then upload to s3.
I couldn't figure out a way to directly transfer from DAAC to s3.
I also was having trouble getting the hdf processing code to use the hdfs directly on DAAC,
so I decided to copy them to s3 for simplicity.
I also couldn't figure out an easy way to do this in Python programatically, so I did it using the command line.

MODIS burned area v6.1 data landing page: https://lpdaac.usgs.gov/products/mcd64a1v061/
Site to download hdfs from: https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/
These are hdf4, not hdf5. This affects what Python libraries to use with them.

Before doing any processing with this script, I copied all the hdfs to s3. I decided to copy all of them to s3
and then work with them there instead of having the script access them directly DAAC because

https://lpdaac.usgs.gov/resources/e-learning/how-access-lp-daac-data-command-line/

To set up wget downloading of hdfs in home directory:
nano .wgetrc | chmod og-rw .wgetrc   # touch didn't work in Ubuntu
echo http-user=REPLACEWITHUSERNAME >> .wgetrc | echo http-password=REPLACEWITHPASSWORD >> .wgetrc

https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67b13996-a308-800a-98a1-dba578305b8d

To download all hdfs in the upper folder (after it iterates through the index.htmls):
wget -r -np -nH --cut-dirs=2 -R "index.html*" -P MCD64A1_data -e robots=off -A "*.hdf" -nd https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/

To download all hdfs for a specific year (all months):
wget -r -np -nH --cut-dirs=2 -R "index.html*" -P MCD64A1_data -e robots=off -A "*.hdf" -nd -I "MOTA/MCD64A1.061/2001*" https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/
2001 alone took 18 minutes

2000-2009:
time wget -r -np -nH --cut-dirs=2 -R "index.html*" -P MCD64A1_data -e robots=off -A "*.hdf" -nd -I "MOTA/MCD64A1.061/200*" https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/
To upload hdfs to s3 for processing:
(base) dagibbs22@USAWDC7F81Q74:~/MODIS/MCD64A1_data$ time aws s3 cp . s3://gfw2-data/fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/ --recursive
321 minutes to upload everything in the 2000s (29472 files)
Deleted 2000-2009 downloaded hdfs.

2010-2019:
time wget -r -np -nH --cut-dirs=2 -R "index.html*" -P MCD64A1_data -e robots=off -A "*.hdf" -nd -I "MOTA/MCD64A1.061/201*" https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/
344 minutes to download everything in the 2010s (32162 files)
aws s3 cp . s3://gfw2-data/fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/ --recursive
Deleted 2010-2019 downloaded hdfs.

2020-2024:
time wget -r -np -nH --cut-dirs=2 -R "index.html*" -P MCD64A1_data -e robots=off -A "*.hdf" -nd -I "MOTA/MCD64A1.061/202*" https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/
267 minutes to download 2020-2024 (16078 fies)
(base) dagibbs22@USAWDC7F81Q74:~/MODIS/MCD64A1_data$ time aws s3 cp . s3://gfw2-data/fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/ --recursive
170 minutes to upload
Deleted 2020-2024 downloaded hdfs.


---Step 1: Run this preprocessing code on the hdfs in s3. The years to run are chosen in main().

hdf processing based on https://chatgpt.com/c/67b0d477-1fc0-800a-b41e-44d954cb9b3e
I have not made this code align with other model components for the most part, e.g., no logs, no output stats, etc.

python -m scripts.utilities.create_cluster -n 1 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing.burned_area.burned_area_hdf_to_raw_raster -cn AFOLU_flux_model_scripts

python -m scripts.utilities.create_cluster -n 100 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing.burned_area.burned_area_hdf_to_raw_raster -cn AFOLU_flux_model_scripts

268 h-v stacks for every year 2001-2024, except for 2005, which has 267 h-v stacks (missing h01v08 in original hdf site)
2001-2024: took about 1.5 hours to run, used about 600 Coiled credits, cost about $20 on AWS.

To download a set of monthly raw h-v hdfs locally from s3 for one year for checking:
aws s3 cp s3://gfw2-data/fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/ . --recursive --exclude "*" --include "*A2024*h24v02*"


---Step 2: Reproject and resample annual burned area rasters that are in MODIS projection/resolution
to 0.00025x0.00025 deg in WGS84 (but does not actually put in Hansen tiles).
Uses the separate 2_burned_area_raw_raster_to_WGS_raster.py


---Step 3: Convert to final Hansen tiles (10x10 deg, 0.00025x0.00025 deg, WGS84) using hansenize_restructure.py for relevant years.
"""




