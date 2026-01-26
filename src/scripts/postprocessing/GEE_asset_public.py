"""
# Publish (set public ACL) for existing Earth Engine assets

This script walks an Earth Engine asset folder/collection and sets assets public.
It is intended to be run AFTER ingestion tasks have completed (i.e., assets exist).

## Script

`src/scripts/postprocessing/gee_set_public.py`

## Requirements

- Earth Engine CLI (`earthengine`) authenticated for the target account.

## Example usage

Dry-run preview:

```bash
python -m src.scripts.postprocessing.gee_set_public \
  --asset-root users/erineglen/organic_soils/wwf_run/2021_2024/40000_pixels/20260120 \
  --recursive \
  --dry-run
```

Apply public ACLs:

```bash
python -m src.scripts.postprocessing.gee_set_public \
  --asset-root users/erineglen/organic_soils/wwf_run/2021_2024/40000_pixels/20260120 \
  --recursive
```

Filter to a specific dataset folder (optional):

```bash
python -m src.scripts.postprocessing.gee_set_public \
  --asset-root users/erineglen/organic_soils/wwf_run/2021_2024/40000_pixels/20260120/burned_ch4_Mg_CO2e_ha_yr \
  --recursive
```

Notes:
- This script is intentionally conservative: it will set public ACL on any *leaf* assets it finds
  (typically Images). Folder/collection ACLs are not relied on.
- If you run this while ingestion is still in progress, assets may not appear in listings yet.
"""

from __future__ import annotations

import argparse
import logging
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from src.scripts.utilities import log_utilities as lu

LOG = logging.getLogger("flm_logger")


LEGACY_PREFIX = "projects/earthengine-legacy/assets/"
ASSETS_MARKER = "/assets/"


def normalize_asset_id(asset_id: str) -> str:
    asset_id = asset_id.strip()
    if asset_id.startswith(LEGACY_PREFIX):
        return asset_id[len(LEGACY_PREFIX) :]
    # For non-legacy project-style IDs, keep as-is.
    return asset_id


def run_command(cmd: Sequence[str], dry_run: bool, allow_fail: bool = False) -> subprocess.CompletedProcess:
    LOG.info("Running: %s", " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        # Mimic CompletedProcess for callers that expect it.
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    try:
        return subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        if allow_fail:
            LOG.warning("Command failed (ignored): %s", exc)
            return exc
        raise


def ee_ls(asset_id: str, dry_run: bool) -> Optional[List[str]]:
    """
    Returns a list of child asset IDs for folders/collections, or None if listing fails.
    """
    asset_id = normalize_asset_id(asset_id)
    if dry_run:
        # In dry-run we don't have a reliable listing; return None so we treat as leaf.
        return None

    proc = subprocess.run(
        ["earthengine", "ls", asset_id],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None

    children: List[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        children.append(normalize_asset_id(line))
    return children


def asset_exists(asset_id: str) -> bool:
    asset_id = normalize_asset_id(asset_id)
    proc = subprocess.run(
        ["earthengine", "asset", "info", asset_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def set_public(asset_id: str, dry_run: bool) -> None:
    asset_id = normalize_asset_id(asset_id)
    run_command(["earthengine", "acl", "set", "public", asset_id], dry_run=dry_run, allow_fail=True)


def iter_assets_recursive(
    root: str,
    dry_run: bool,
    match: Optional[re.Pattern[str]] = None,
    exclude: Optional[re.Pattern[str]] = None,
) -> Iterable[str]:
    """
    Depth-first walk of assets under root using `earthengine ls`.

    For each node:
    - If it can be listed (folder/collection), recurse into children.
    - Otherwise, yield it as a leaf asset.

    We do NOT yield containers by default (only leaves), because public ACLs on containers
    are not assumed to apply to child assets.
    """
    root = normalize_asset_id(root)
    stack: List[str] = [root]

    while stack:
        current = stack.pop()
        current_norm = normalize_asset_id(current)

        if exclude and exclude.search(current_norm):
            continue
        if match and not match.search(current_norm):
            # If a container doesn't match, its children might; still need to traverse.
            pass

        children = ee_ls(current_norm, dry_run=dry_run)
        if children is None:
            # Leaf
            if match and not match.search(current_norm):
                continue
            yield current_norm
            continue

        # Container
        # Always traverse children; filter at leaf stage.
        # Push in reverse to get stable-ish ordering.
        for child in reversed(children):
            stack.append(child)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set public ACL on Earth Engine assets under a root folder/collection."
    )
    parser.add_argument(
        "--asset-root",
        required=True,
        help=(
            "Root Earth Engine asset folder/collection to traverse, e.g. "
            "users/you/root/run/2021_2024/40000_pixels/20260120"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Traverse recursively. If omitted, only assets directly under --asset-root are processed.",
    )
    parser.add_argument(
        "--match",
        help="Regex to include only matching leaf asset IDs (applied to full asset ID).",
    )
    parser.add_argument(
        "--exclude",
        help="Regex to exclude matching asset IDs (applied to full asset ID).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without executing.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=0,
        help=(
            "If >0, wait up to this many seconds for --asset-root to exist before starting. "
            "Useful if you run immediately after submitting ingestion tasks."
        ),
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=10,
        help="Polling interval for --wait-seconds (default: 10).",
    )
    parser.add_argument(
        "--max-assets",
        type=int,
        default=0,
        help="If >0, stop after processing this many leaf assets (safety valve).",
    )
    return parser.parse_args()


def main() -> None:
    global LOG
    LOG = lu.setup_logging_main()
    args = parse_args()

    match_re = re.compile(args.match) if args.match else None
    exclude_re = re.compile(args.exclude) if args.exclude else None

    root = normalize_asset_id(args.asset_root)

    if not args.dry_run and args.wait_seconds > 0:
        deadline = time.time() + args.wait_seconds
        while time.time() < deadline:
            if asset_exists(root):
                break
            time.sleep(max(1, args.poll_seconds))
        else:
            LOG.warning("Timed out waiting for asset root to exist: %s", root)

    # Build the list of targets
    targets: List[str] = []
    if args.recursive:
        targets.extend(iter_assets_recursive(root, dry_run=args.dry_run, match=match_re, exclude=exclude_re))
    else:
        children = ee_ls(root, dry_run=args.dry_run) or []
        # Only process leaves directly under root.
        for child in children:
            # A child might itself be a folder; treat non-listable items as leaves.
            if exclude_re and exclude_re.search(child):
                continue
            if match_re and not match_re.search(child):
                continue
            # If ls(child) succeeds, it's a container; skip unless recursive.
            if ee_ls(child, dry_run=args.dry_run) is None:
                targets.append(child)

    if args.max_assets > 0:
        targets = targets[: args.max_assets]

    if not targets:
        LOG.warning("No leaf assets found to publish under: %s", root)
        return

    LOG.info("Publishing %d assets (public ACL): %s", len(targets), root)
    for asset_id in targets:
        set_public(asset_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
