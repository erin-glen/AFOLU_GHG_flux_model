
"""
# S3 to GEE asset publishing

This script automates downloading a subset of model outputs from S3, uploading them to
Google Earth Engine (GEE) assets, and optionally setting those assets public.

## Script

`src/scripts/utilities/s3_to_gee_assets.py`

## Requirements

- AWS credentials with access to the S3 bucket.
- Earth Engine CLI (`earthengine`) authenticated for the target account.

## Example usage

```bash
python -m src.scripts.utilities.s3_to_gee_assets \
  --s3-root s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_0_1_0 \
  --run-name wwf_run\
  --run-date 20260120 \
  --years 2001-2024 \
  --datasets drained burned \
  --gee-root users/erineglen/organic_soils \
  --include-ext .tif \
  --gee-upload-args "--pyramiding_policy=MEAN" \
  --skip-existing
```

By default the script expects the S3 layout
`{s3_root}/{run_name}/{run_date}/{year}/{dataset}/...`.
If your data uses a different layout, override `--s3-template`.

## Notes

- Use `--dry-run` to preview actions without downloading or uploading.
- Use `--no-public` to skip setting ACLs.
- For non-default AWS accounts or regions, set `--aws-profile`/`--aws-region`.
"""

from __future__ import annotations

import argparse
import logging
import os
import posixpath
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import boto3

LOG = logging.getLogger(__name__)


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


def parse_datasets(raw: str) -> List[str]:
    return [t for t in re.split(r"[\s,]+", raw.strip()) if t]


def split_s3_uri(uri: str) -> S3Location:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri}")
    without_scheme = uri[5:]
    parts = without_scheme.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return S3Location(bucket=bucket, prefix=prefix)


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


def ensure_gee_folders(asset_id: str, dry_run: bool) -> None:
    parts = asset_id.split("/")
    for idx in range(2, len(parts)):
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
        rel_key = key[len(prefix):].lstrip("/")
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
    run_date: str,
    year: int,
    dataset: str,
    rel_path: str,
) -> str:
    rel_no_ext = os.path.splitext(rel_path)[0].replace(os.sep, "/")
    return posixpath.join(gee_root, run_name, run_date, str(year), dataset, rel_no_ext)


def upload_and_publish(
    local_path: str,
    asset_id: str,
    extra_args: Sequence[str],
    make_public: bool,
    dry_run: bool,
) -> None:
    ensure_gee_folders(posixpath.dirname(asset_id), dry_run)
    run_command(
        ["earthengine", "upload", "image", f"--asset_id={asset_id}", *extra_args, local_path],
        dry_run,
    )
    if make_public:
        run_command(["earthengine", "acl", "set", "public", asset_id], dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download model outputs from S3 to local, upload to GEE as assets, and set public."
        )
    )
    parser.add_argument("--s3-root", required=True, help="Base s3://bucket/prefix root.")
    parser.add_argument(
        "--s3-template",
        default="{s3_root}/{run_name}/{run_date}/{year}/{dataset}",
        help=(
            "Template for S3 prefix. Available fields: s3_root, run_name, run_date, year, dataset."
        ),
    )
    parser.add_argument("--run-name", required=True, help="Model run name.")
    parser.add_argument("--run-date", required=True, help="Run date (YYYYMMDD).")
    parser.add_argument(
        "--years",
        required=True,
        help="Years to process (e.g. '2018,2019' or '2018-2020').",
    )
    parser.add_argument(
        "--datasets",
        required=True,
        help="Dataset names to process (comma- or space-separated).",
    )
    parser.add_argument(
        "--local-root",
        default="./gee_uploads",
        help="Local root directory for downloads.",
    )
    parser.add_argument(
        "--gee-root",
        required=True,
        help="GEE asset root (e.g. users/you/afolu).",
    )
    parser.add_argument(
        "--include-ext",
        default=".tif",
        help="Only download files with this extension (default: .tif).",
    )
    parser.add_argument(
        "--gee-upload-args",
        default="",
        help="Extra args for 'earthengine upload image' (quoted string).",
    )
    parser.add_argument("--no-public", action="store_true", help="Skip setting assets public.")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without executing.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip uploads if asset already exists.",
    )
    parser.add_argument("--aws-profile", help="AWS profile name to use.")
    parser.add_argument("--aws-region", help="AWS region for session.")
    return parser.parse_args()


def asset_exists(asset_id: str, dry_run: bool) -> bool:
    if dry_run:
        return False
    result = subprocess.run(
        ["earthengine", "ls", asset_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    years = parse_years(args.years)
    datasets = parse_datasets(args.datasets)

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

    for year in years:
        for dataset in datasets:
            s3_prefix_template = args.s3_template.format(
                s3_root=s3_root,
                run_name=args.run_name,
                run_date=args.run_date,
                year=year,
                dataset=dataset,
            )
            loc = split_s3_uri(s3_prefix_template)
            if loc.bucket != s3_root_loc.bucket:
                raise ValueError("S3 template bucket must match --s3-root bucket")
            local_dir = os.path.join(
                args.local_root, args.run_name, args.run_date, str(year), dataset
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

            for local_path in downloaded:
                rel_path = os.path.relpath(local_path, local_dir)
                asset_id = build_asset_id(
                    args.gee_root,
                    args.run_name,
                    args.run_date,
                    year,
                    dataset,
                    rel_path,
                )
                if args.skip_existing and asset_exists(asset_id, args.dry_run):
                    LOG.info("Skipping existing asset %s", asset_id)
                    continue
                upload_and_publish(
                    local_path=local_path,
                    asset_id=asset_id,
                    extra_args=extra_args,
                    make_public=not args.no_public,
                    dry_run=args.dry_run,
                )


if __name__ == "__main__":
    main()