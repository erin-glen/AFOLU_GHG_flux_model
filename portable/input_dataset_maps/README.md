# Portable AFOLU input-dataset map generator

This folder is a self-contained handoff for generating aligned, designer-ready
maps. It does not import the AFOLU repository, require Coiled/Dask, or need to
be run from a particular working directory. The bundled example produces eight
maps over the same Borneo area: six inputs, final state code, and total drained
emissions.

The only organization-specific prerequisite is read access to the private
`s3://gfw2-data` objects listed in `input_dataset_maps.config.json`.

## Recommended setup (Windows)

Install Miniconda or Anaconda, unzip this folder, and open Anaconda Prompt or
PowerShell in the folder. Then run:

```powershell
conda env create --file environment.yml
conda activate afolu-input-dataset-maps
python smoke_test.py
```

The smoke test is local-only: it creates tiny temporary rasters and confirms
that the standalone script can produce eight aligned PNGs and GeoTIFFs. It does
not connect to AWS.

If conda is unavailable, Python 3.11 can use the pinned pip environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --requirement requirements.txt
python smoke_test.py
```

Conda is recommended on Windows because it handles Rasterio/GDAL binaries more
consistently.

## Authenticate to AWS

Use the normal organization AWS credentials. If AWS CLI v2 is installed, an
AWS SSO profile can be authenticated like this:

```powershell
aws sso login --profile YOUR_PROFILE
```

The profile must be able to read the configured S3 objects. Exact paths need
`s3:GetObject`; S3 wildcard paths additionally need `s3:ListBucket`. The
example config uses six tile templates that each resolve to one exact key for
this AOI, plus two exact global paths, so it does not require a bucket listing.

## Validate, then run

First confirm that the config parses and resolves all eight paths without
downloading the rasters:

```powershell
.\run_maps.ps1 -ValidateOnly -AwsProfile YOUR_PROFILE
```

Then generate the maps:

```powershell
.\run_maps.ps1 -AwsProfile YOUR_PROFILE
```

If the default AWS profile is already authenticated, omit `-AwsProfile`. Each
run gets a new timestamped folder under `outputs`, so an analyst does not
accidentally overwrite a prior result. The first run caches about 309 MB of
source data under `cache`; allow at least 1 GB of free disk space.

Command Prompt users can pass the same PowerShell options through the batch
launcher:

```bat
run_maps.bat -ValidateOnly -AwsProfile YOUR_PROFILE
run_maps.bat -AwsProfile YOUR_PROFILE
```

On macOS, Linux, or WSL:

```bash
chmod +x run_maps.sh
./run_maps.sh --validate-only --aws-profile YOUR_PROFILE
./run_maps.sh --aws-profile YOUR_PROFILE
```

## Results

Every successful run contains:

- Eight numbered, transparent-background PNGs with identical pixel dimensions.
- Eight aligned float32 GeoTIFFs under `aligned/` for QA or GIS use.
- `input_dataset_maps.manifest.json`, which records source paths, styles,
  dimensions, statistics, and SHA-256 hashes.

The PNGs are the files to pass to the design team. The GeoTIFFs and manifest
make the result auditable.

## Adapt the example

Make a copy of `input_dataset_maps.config.json` and edit only that copy.

- `target.bounds` is `[west, south, east, north]` in the target CRS.
- `target.width` and `target.height` control the shared output dimensions.
- Dataset order controls the numeric filename prefixes.
- `source` can be a local path, exact S3 URI, wildcard, or a 10-degree tile
  template containing `{tile_id}`.
- Use `kind: "categorical"` with `resampling: "nearest"` for class rasters.
- Use `kind: "continuous"` with `resampling: "bilinear"` for continuous data.

Run the copied config explicitly:

```powershell
.\run_maps.ps1 -Config C:\path\to\my.config.json -AwsProfile YOUR_PROFILE
```

The bundled S3 paths are intentionally frozen to the versions used for the
verified example. Update them deliberately when moving to a newer model run.

## Direct command and troubleshooting

The launchers are conveniences. The equivalent direct command is:

```powershell
python create_input_dataset_maps.py --config input_dataset_maps.config.json --output-dir outputs\manual_run --cache-dir cache --aws-profile YOUR_PROFILE
```

- `NoCredentialsError` means no usable AWS session was found; authenticate or
  pass the correct profile.
- `AccessDenied` means the selected identity lacks permission for that object.
- Rasterio/GDAL installation errors are usually resolved by using
  `environment.yml` instead of pip on Windows.
- Re-running the same output directory is blocked by default. Choose a new
  directory, or use `-Overwrite` only when replacement is intentional.
- Delete `cache` only when the source objects changed at the same S3 paths or
  disk space must be reclaimed.

`PROVENANCE.json` records the repository commit and model-output version used to
build this handoff. `SHA256SUMS.txt` can be used to confirm that transferred
files were not altered.
