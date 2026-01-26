"""
# S3 to GEE asset publishing (via GCS staging)

This script automates downloading a subset of model outputs from S3, staging them to
Google Cloud Storage (GCS), and uploading them to Google Earth Engine (GEE) assets.

Important:
- The Earth Engine CLI command `earthengine upload image` ingests images *from GCS*,
  not from local disk. Therefore a `gs://...` URI is required.
- This script DOES NOT set assets public. Use the companion script
  `src/scripts/postprocessing/gee_set_public.py` after ingestion completes.

## Script

`src/scripts/postprocessing/s3_to_GEE.py`

## Requirements

- AWS credentials with access to the S3 bucket.
- Earth Engine CLI (`earthengine`) authenticated for the target account.
- Google Cloud CLI tooling authenticated for GCS writes:
  - Either `gcloud` (recommended) OR `gsutil` available on PATH.
- A writable GCS bucket/prefix provided via `--gcs-root`.

## Example usage

```bash
python -m src.scripts.postprocessing.s3_to_GEE \
  --run-name wwf_run \
  --output-date 20260120 \
  --inventory-periods all \
  --datasets "drained burned" \
  --include-pixel-outputs \
  --gee-root users/erineglen/organic_soils \
  --gcs-root gs://MY_STAGING_BUCKET/organic_soils_stage \
  --include-ext .tif \
  --gee-upload-args="--pyramiding_policy=MEAN" \
  --skip-existing
```

After uploads finish, publish assets with:

```bash
python -m src.scripts.postprocessing.gee_set_public \
  --asset-root users/erineglen/organic_soils/wwf_run/2021_2024/40000_pixels/20260120 \
  --recursive
```

By default the script expects the S3 layout
`{s3_root}/{dataset}/{run_name}/{interval_type}_intervals/`
`{period}/{output_pixel_resolution}/{output_date}/...`.
If your data uses a different layout, override `--s3-template`.

## Notes

- `--s3-root` defaults to the shared outputs root (`cn.outputs_path`).
- `--aws-region` defaults to the shared AWS region (`cn.s3_region_name`).
- Use `--dry-run` to preview actions without downloading, staging, or uploading.
- For non-default AWS accounts or regions, set `--aws-profile`/`--aws-region`.
- `--run-date` is a legacy alias for `--output-date`.
- Use `--inventory-periods` for five-year outputs (supports `all`, a single year
  like `2021`, or `start_end` labels) or `--years` to map years into inventory
  periods (annual intervals use the year directly).
- Dataset aliases: use `drained`, `burned`, or `all` to expand to the
  model output dataset names defined in `cn.drainage_outputs_to_zarr`.
- Use `--include-pixel-outputs` to include the per-pixel datasets created
  by `3_aggregate_soils_outputs` (e.g., `_pixel_` variants of `_ha_` outputs).
- IMPORTANT: Do not delete staged GCS objects until EE ingestion tasks complete.
"""

from __future__ import annotations

import argparse
import logging
import os
import posixpath
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import boto3

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import log_utilities as lu

LOG = logging.getLogger("flm_logger")


@dataclass(frozen=True)
class S3Location:
    bucket: str
    prefix: str


def parse_years(raw: str) -> List[int]:
    years: List[int] = []
    tokens = [t for t in re.split(r"[\s,]+", raw.strip()) if t]
    for token in tokens:
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                raise ValueError(f"Invalid year range: {token}")
            years.extend(range(start, end + 1))
        else:
            years.append(int(token))
    return sorted(set(years))


def parse_periods(raw: str) -> List[str]:
    return [t for t in re.split(r"[\s,]+", raw.strip()) if t]


def build_period_lookup() -> dict[int, str]:
    return {end: f"{start}_{end}" for start, end in cn.five_year_inventory_periods}


def period_for_year(year: int) -> str:
    for start, end in cn.five_year_inventory_periods:
        if start <= year <= end:
            return f"{start}_{end}"
    raise ValueError(f"No inventory period found for year {year}.")


def periods_from_years(years: Sequence[int], interval_type: str) -> List[str]:
    if interval_type == cn.intervals_annual:
        return [str(year) for year in years]
    return sorted({period_for_year(year) for year in years})


def normalize_inventory_periods(raw: str, interval_type: str) -> List[str]:
    tokens = parse_periods(raw)
    if interval_type == cn.intervals_annual:
        if any(token.lower() == "all" for token in tokens):
            raise ValueError("Token 'all' is only supported for five-year intervals.")
        years = parse_years(" ".join(tokens))
        return [str(year) for year in years]

    if any(token.lower() == "all" for token in tokens):
        return [f"{start}_{end}" for start, end in cn.five_year_inventory_periods]

    period_lookup = build_period_lookup()
    normalized: List[str] = []
    for token in tokens:
        if token.isdigit():
            year = int(token)
            normalized.append(period_for_year(year))
            continue
        match = re.match(r"^(?P<start>\d{4})[-_](?P<end>\d{4})$", token)
        if match:
            start = int(match.group("start"))
            end = int(match.group("end"))
            if period_lookup.get(end) != f"{start}_{end}":
                raise ValueError(f"Unknown inventory period: {token}")
            normalized.append(period_lookup[end])
            continue
        raise ValueError(f"Invalid inventory period token: {token}")

    return sorted(set(normalized))


def build_dataset_catalog(include_pixel_outputs: bool) -> List[str]:
    catalog = list(cn.drainage_outputs_to_zarr)
    if include_pixel_outputs:
        for name in cn.drainage_outputs_to_zarr:
            if cn.drainage_output_dtypes.get(name) == "float32" and "_ha_" in name:
                catalog.append(name.replace("_ha_", "_pixel_"))
    return catalog


def parse_datasets(raw: str, dataset_catalog: Sequence[str]) -> List[str]:
    tokens = [t for t in re.split(r"[\s,]+", raw.strip()) if t]
    catalog = list(dataset_catalog)
    expanded: List[str] = []
    unknown: List[str] = []
    for token in tokens:
        if token == "all":
            expanded.extend(catalog)
            continue
        if token == "drained":
            expanded.extend([name for name in catalog if name.startswith("drained_")])
            continue
        if token == "burned":
            expanded.extend([name for name in catalog if name.startswith("burned_")])
            continue
        if token in catalog:
            expanded.append(token)
            continue
        unknown.append(token)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(sorted(set(unknown)))}")
    return sorted(set(expanded))


def split_s3_uri(uri: str) -> S3Location:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri}")
    without_scheme = uri[5:]
    parts = without_scheme.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return S3Location(bucket=bucket, prefix=prefix)


def normalize_gcs_root(gcs_root: str) -> str:
    gcs_root = gcs_root.rstrip("/")
    if not gcs_root.startswith("gs://"):
        raise ValueError("--gcs-root must start with gs://")
    return gcs_root


def list_s3_objects(s3_client, bucket: str, prefix: str) -> Iterable[str]:
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            yield item["Key"]


def run_command(cmd: Sequence[str], dry_run: bool, allow_fail: bool = False) -> None:
    LOG.info("Running: %s", " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        if allow_fail:
            LOG.warning("Command failed (ignored): %s", exc)
            return
        raise


def ensure_gee_folders(folder_id: str, dry_run: bool) -> None:
    """
    Ensure all parent folders up to and including folder_id exist.

    Example folder_id:
      users/you/root/run/2021_2024/40000_pixels/20260120/dataset_name
    """
    parts = folder_id.split("/")
    # idx must include len(parts) so the deepest folder is created.
    for idx in range(2, len(parts) + 1):
        folder = "/".join(parts[:idx])
        run_command(["earthengine", "create", "folder", folder], dry_run, allow_fail=True)


def download_objects(
    s3_client,
    bucket: str,
    prefix: str,
    local_dir: str,
    include_ext: str | None,
    dry_run: bool,
) -> List[str]:
    downloaded: List[str] = []
    for key in list_s3_objects(s3_client, bucket, prefix):
        if include_ext and not key.endswith(include_ext):
            continue
        rel_key = key[len(prefix) :].lstrip("/")
        local_path = os.path.join(local_dir, rel_key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        LOG.info("Downloading s3://%s/%s -> %s", bucket, key, local_path)
        if not dry_run:
            s3_client.download_file(bucket, key, local_path)
        downloaded.append(local_path)
    return downloaded


def build_asset_id(
    gee_root: str,
    run_name: str,
    output_date: str,
    period: str,
    dataset: str,
    output_pixel_resolution: str,
    rel_path: str,
) -> str:
    rel_no_ext = os.path.splitext(rel_path)[0].replace(os.sep, "/")
    return posixpath.join(
        gee_root,
        run_name,
        period,
        output_pixel_resolution,
        output_date,
        dataset,
        rel_no_ext,
    )


def detect_gcs_tool(preference: str) -> str:
    """
    Returns one of: 'gcloud', 'gsutil'.
    """
    preference = (preference or "auto").lower()
    if preference in {"gcloud", "gsutil"}:
        return preference
    if preference != "auto":
        raise ValueError("--gcs-tool must be one of: auto, gcloud, gsutil")

    if shutil.which("gcloud"):
        return "gcloud"
    if shutil.which("gsutil"):
        return "gsutil"
    raise RuntimeError(
        "Neither 'gcloud' nor 'gsutil' was found on PATH. "
        "Install Google Cloud SDK (gcloud) or gsutil, and authenticate for GCS access."
    )


def stage_directory_to_gcs(
    local_dir: str,
    gcs_dir: str,
    tool: str,
    dry_run: bool,
) -> None:
    """
    Stage an entire local directory to a GCS prefix, preserving relative paths.

    - gsutil: gsutil -m rsync -r <local_dir> <gcs_dir>
    - gcloud: gcloud storage rsync --recursive <local_dir> <gcs_dir>
    """
    if tool == "gsutil":
        cmd = ["gsutil", "-m", "rsync", "-r", local_dir, gcs_dir]
    elif tool == "gcloud":
        cmd = ["gcloud", "storage", "rsync", "--recursive", local_dir, gcs_dir]
    else:
        raise ValueError(f"Unsupported GCS tool: {tool}")
    run_command(cmd, dry_run)


def stage_file_to_gcs(
    local_path: str,
    gcs_uri: str,
    tool: str,
    dry_run: bool,
) -> None:
    """
    Stage a single file to an exact GCS URI.

    - gsutil: gsutil -q cp <local_path> <gcs_uri>
    - gcloud: gcloud storage cp <local_path> <gcs_uri>
    """
    if tool == "gsutil":
        cmd = ["gsutil", "-q", "cp", local_path, gcs_uri]
    elif tool == "gcloud":
        cmd = ["gcloud", "storage", "cp", local_path, gcs_uri]
    else:
        raise ValueError(f"Unsupported GCS tool: {tool}")
    run_command(cmd, dry_run)


def upload_image_from_gcs(
    source_uri: str,
    asset_id: str,
    extra_args: Sequence[str],
    dry_run: bool,
) -> None:
    """
    Upload (ingest) an image from GCS into EE.
    This starts an asynchronous EE ingestion task.
    """
    if not source_uri.startswith("gs://"):
        raise ValueError(f"earthengine upload image requires a gs:// URI; got: {source_uri}")

    ensure_gee_folders(posixpath.dirname(asset_id), dry_run)
    run_command(
        ["earthengine", "upload", "image", f"--asset_id={asset_id}", *extra_args, source_uri],
        dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download model outputs from S3, stage to GCS, upload to GEE as assets."
    )
    parser.add_argument(
        "--s3-root",
        default=cn.outputs_path,
        help=("Base s3://bucket/prefix root. " f"Defaults to {cn.outputs_path}."),
    )
    parser.add_argument(
        "--s3-template",
        default=(
            "{s3_root}/{dataset}/{run_name}/{interval_type}_intervals/"
            "{period}/{output_pixel_resolution}/{output_date}"
        ),
        help=(
            "Template for S3 prefix. Available fields: s3_root, run_name, output_date, "
            "run_date, dataset, interval_type, period, output_pixel_resolution."
        ),
    )
    parser.add_argument("--run-name", required=True, help="Model run name.")
    parser.add_argument("--output-date", required=False, help="Output date (YYYYMMDD).")
    parser.add_argument("--run-date", help="Legacy alias for --output-date (YYYYMMDD).")
    parser.add_argument(
        "--years",
        help=(
            "Years to process (e.g. '2018,2019' or '2018-2020'). "
            "Converted to inventory periods unless --interval-type is annual."
        ),
    )
    parser.add_argument(
        "--inventory-periods",
        help=(
            "Inventory periods to process (e.g. 'all', '2001_2005 2021_2024', or years like '2021'). "
            "Five-year intervals normalize to canonical start_end labels."
        ),
    )
    parser.add_argument(
        "--datasets",
        required=True,
        help=(
            "Dataset names to process (comma- or space-separated). "
            "Use drained, burned, or all to expand to model output datasets."
        ),
    )
    parser.add_argument(
        "--include-pixel-outputs",
        action="store_true",
        help="Include per-pixel datasets produced alongside per-ha outputs.",
    )
    parser.add_argument(
        "--interval-type",
        default=cn.intervals_five_year,
        help=(
            "Interval type for outputs (e.g. five_year, annual). "
            f"Default: {cn.intervals_five_year}."
        ),
    )
    parser.add_argument(
        "--output-pixel-resolution",
        default="40000_pixels",
        help="Pixel resolution folder name (default: 40000_pixels).",
    )
    parser.add_argument("--local-root", default="./gee_uploads", help="Local root directory for downloads.")
    parser.add_argument("--gee-root", required=True, help="GEE asset root (e.g. users/you/afolu).")

    parser.add_argument(
        "--gcs-root",
        help=(
            "GCS staging root (e.g. gs://my-bucket/some/prefix). "
            "Required unless --dry-run."
        ),
    )
    parser.add_argument(
        "--gcs-tool",
        default="auto",
        choices=["auto", "gcloud", "gsutil"],
        help="Tool to use for staging to GCS (default: auto).",
    )
    parser.add_argument(
        "--gcs-stage-mode",
        default="rsync",
        choices=["rsync", "cp"],
        help=(
            "How to stage files to GCS. "
            "'rsync' stages the whole directory efficiently; 'cp' copies files one-by-one. "
            "Default: rsync."
        ),
    )

    parser.add_argument("--include-ext", default=".tif", help="Only download files with this extension (default: .tif).")
    parser.add_argument("--gee-upload-args", default="", help="Extra args for 'earthengine upload image' (quoted string).")

    parser.add_argument("--dry-run", action="store_true", help="Log actions without executing.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip uploads if asset already exists.",
    )
    parser.add_argument(
        "--cleanup-local",
        action="store_true",
        help="Delete local downloads after staging to GCS (does not delete GCS objects).",
    )
    parser.add_argument("--aws-profile", help="AWS profile name to use.")
    parser.add_argument(
        "--aws-region",
        default=cn.s3_region_name,
        help=f"AWS region for session (default: {cn.s3_region_name}).",
    )

    # Backward-compatibility: previously used by some commands, now a no-op.
    parser.add_argument(
        "--no-public",
        action="store_true",
        help="DEPRECATED (no-op): this script never sets assets public. Use gee_set_public.py.",
    )
    return parser.parse_args()


def asset_exists(asset_id: str, dry_run: bool) -> bool:
    if dry_run:
        return False

    result = subprocess.run(
        ["earthengine", "asset", "info", asset_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def main() -> None:
    global LOG
    LOG = lu.setup_logging_main()
    args = parse_args()

    if args.run_date and args.output_date and args.run_date != args.output_date:
        raise ValueError("--run-date must match --output-date when both are provided.")
    output_date = args.output_date or args.run_date
    if not output_date:
        raise ValueError("Missing required --output-date.")

    if not args.years and not args.inventory_periods:
        raise ValueError("Provide --years or --inventory-periods.")

    years = parse_years(args.years) if args.years else []
    periods = (
        normalize_inventory_periods(args.inventory_periods, args.interval_type)
        if args.inventory_periods
        else periods_from_years(years, args.interval_type)
    )
    dataset_catalog = build_dataset_catalog(args.include_pixel_outputs)
    datasets = parse_datasets(args.datasets, dataset_catalog)

    session_kwargs = {}
    if args.aws_profile:
        session_kwargs["profile_name"] = args.aws_profile
    if args.aws_region:
        session_kwargs["region_name"] = args.aws_region
    session = boto3.Session(**session_kwargs)
    s3_client = session.client("s3")

    s3_root = args.s3_root.rstrip("/")
    if not s3_root.startswith("s3://"):
        raise ValueError("--s3-root must start with s3://")

    s3_root_loc = split_s3_uri(s3_root)
    extra_args = shlex.split(args.gee_upload_args)

    gcs_root = None
    gcs_tool = None
    if not args.dry_run:
        if not args.gcs_root:
            raise ValueError(
                "earthengine upload image requires sources in GCS. "
                "Provide --gcs-root gs://<bucket>/<prefix> (or use --dry-run)."
            )
        gcs_root = normalize_gcs_root(args.gcs_root)
        gcs_tool = detect_gcs_tool(args.gcs_tool)

    for period in periods:
        for dataset in datasets:
            s3_prefix_template = args.s3_template.format(
                s3_root=s3_root,
                run_name=args.run_name,
                run_date=output_date,
                output_date=output_date,
                dataset=dataset,
                interval_type=args.interval_type,
                period=period,
                output_pixel_resolution=args.output_pixel_resolution,
            )
            loc = split_s3_uri(s3_prefix_template)
            if loc.bucket != s3_root_loc.bucket:
                raise ValueError("S3 template bucket must match --s3-root bucket")

            local_dir = os.path.join(
                args.local_root,
                args.run_name,
                period,
                args.output_pixel_resolution,
                output_date,
                dataset,
            )

            downloaded = download_objects(
                s3_client=s3_client,
                bucket=loc.bucket,
                prefix=loc.prefix.rstrip("/") + "/",
                local_dir=local_dir,
                include_ext=args.include_ext,
                dry_run=args.dry_run,
            )
            if not downloaded:
                LOG.warning("No files found for s3://%s/%s", loc.bucket, loc.prefix)
                continue

            # Stage to GCS (directory-level by default).
            gcs_dir = None
            if not args.dry_run:
                assert gcs_root is not None and gcs_tool is not None
                gcs_dir = posixpath.join(
                    gcs_root,
                    args.run_name,
                    period,
                    args.output_pixel_resolution,
                    output_date,
                    dataset,
                )
                LOG.info(
                    "Staging local_dir=%s -> %s (tool=%s, mode=%s)",
                    local_dir,
                    gcs_dir,
                    gcs_tool,
                    args.gcs_stage_mode,
                )
                if args.gcs_stage_mode == "rsync":
                    stage_directory_to_gcs(local_dir=local_dir, gcs_dir=gcs_dir, tool=gcs_tool, dry_run=args.dry_run)
                else:
                    for local_path in downloaded:
                        rel_path = os.path.relpath(local_path, local_dir).replace(os.sep, "/")
                        gcs_uri = posixpath.join(gcs_dir, rel_path)
                        stage_file_to_gcs(local_path=local_path, gcs_uri=gcs_uri, tool=gcs_tool, dry_run=args.dry_run)

            # Upload each file (from its gs:// URI) to EE.
            for local_path in downloaded:
                rel_path = os.path.relpath(local_path, local_dir)
                asset_id = build_asset_id(
                    args.gee_root,
                    args.run_name,
                    output_date,
                    period,
                    dataset,
                    args.output_pixel_resolution,
                    rel_path,
                )
                if args.skip_existing and asset_exists(asset_id, args.dry_run):
                    LOG.info("Skipping existing asset %s", asset_id)
                    continue

                if args.dry_run:
                    gcs_uri = posixpath.join(
                        "gs://<gcs-root-required>",
                        args.run_name,
                        period,
                        args.output_pixel_resolution,
                        output_date,
                        dataset,
                        rel_path.replace(os.sep, "/"),
                    )
                else:
                    assert gcs_dir is not None
                    gcs_uri = posixpath.join(gcs_dir, rel_path.replace(os.sep, "/"))

                upload_image_from_gcs(
                    source_uri=gcs_uri,
                    asset_id=asset_id,
                    extra_args=extra_args,
                    dry_run=args.dry_run,
                )

            if args.cleanup_local and not args.dry_run:
                try:
                    LOG.info("Cleaning up local downloads: %s", local_dir)
                    shutil.rmtree(local_dir, ignore_errors=True)
                except Exception as exc:
                    LOG.warning("Failed to cleanup local dir %s: %s", local_dir, exc)


if __name__ == "__main__":
    main()