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
  --asset-root users/erineglen/organic_soils/wwf_run \
  --recursive \
  --dry-run
```

Apply public ACLs:

```bash
python -m src.scripts.postprocessing.gee_set_public \
  --asset-root users/erineglen/organic_soils/wwf_run \
  --recursive
```

Notes:
- Conservative: sets public ACL on leaf assets only (typically Images).
- Uses `earthengine ls -l` to avoid calling `ls` on every leaf image (much faster).
- Adds per-command timeouts + retries so a single stalled CLI call can't hang forever.
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


def normalize_asset_id(asset_id: str) -> str:
    asset_id = asset_id.strip()
    if asset_id.startswith(LEGACY_PREFIX):
        return asset_id[len(LEGACY_PREFIX) :]
    return asset_id


def _normalize_asset_type(asset_type: str) -> str:
    # Make type checks robust to minor formatting differences (spaces/underscores/case).
    return re.sub(r"[^A-Za-z0-9]+", "", (asset_type or "")).upper()


# Types returned by `earthengine ls -l` that should be treated as containers.
_CONTAINER_TYPES = {"FOLDER", "IMAGECOLLECTION"}


def _is_container_type(asset_type: str) -> bool:
    return _normalize_asset_type(asset_type) in _CONTAINER_TYPES


def _fmt_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_command(
    cmd: Sequence[str],
    *,
    dry_run: bool,
    timeout_s: int,
    retries: int,
    allow_fail: bool = False,
    log_cmd: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run a subprocess with timeout + retries.

    - If allow_fail=False, raises RuntimeError after exhausting retries.
    - If allow_fail=True, returns the last CompletedProcess even if nonzero/timeout.
    """
    if log_cmd:
        LOG.info("Running: %s", _fmt_cmd(cmd))

    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    retries = max(1, retries)
    last: Optional[subprocess.CompletedProcess] = None

    for attempt in range(1, retries + 1):
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s)
            last = proc

            if proc.returncode == 0:
                return proc

            # Non-zero return code
            if attempt < retries:
                backoff = min(30, 2 ** (attempt - 1))
                LOG.warning(
                    "Command failed (rc=%s), retrying in %ss (%d/%d): %s | stderr: %s",
                    proc.returncode,
                    backoff,
                    attempt,
                    retries,
                    _fmt_cmd(cmd),
                    (proc.stderr or "").strip(),
                )
                time.sleep(backoff)
                continue

            # final attempt
            if allow_fail:
                LOG.warning(
                    "Command failed (rc=%s) (ignored): %s | stderr: %s",
                    proc.returncode,
                    _fmt_cmd(cmd),
                    (proc.stderr or "").strip(),
                )
                return proc

            raise RuntimeError(
                f"Command failed (rc={proc.returncode}): {_fmt_cmd(cmd)}\n"
                f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
            )

        except subprocess.TimeoutExpired:
            # Timeout
            last = subprocess.CompletedProcess(cmd, 124, stdout="", stderr=f"Timed out after {timeout_s}s")

            if attempt < retries:
                backoff = min(30, 2 ** (attempt - 1))
                LOG.warning(
                    "Command timed out after %ss, retrying in %ss (%d/%d): %s",
                    timeout_s,
                    backoff,
                    attempt,
                    retries,
                    _fmt_cmd(cmd),
                )
                time.sleep(backoff)
                continue

            if allow_fail:
                LOG.warning("Command timed out after %ss (ignored): %s", timeout_s, _fmt_cmd(cmd))
                return last

            raise RuntimeError(f"Command timed out after {timeout_s}s: {_fmt_cmd(cmd)}")

    # Should be unreachable.
    return last or subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Unknown failure")


@dataclass(frozen=True)
class ListedChild:
    asset_id: str
    asset_type: str  # e.g., Image, ImageCollection, Folder, Table, ...


def ee_ls_typed(asset_id: str, *, dry_run: bool, timeout_s: int, retries: int) -> Optional[List[ListedChild]]:
    """
    Returns typed children for folders/collections, or None if listing fails (treat as leaf).

    Uses `earthengine ls -l` (long format includes type).
    """
    asset_id = normalize_asset_id(asset_id)
    if dry_run:
        # In dry-run we don't have a reliable listing.
        return None

    # Prefer long format (includes type).
    proc = run_command(
        ["earthengine", "ls", "-l", asset_id],
        dry_run=dry_run,
        timeout_s=timeout_s,
        retries=retries,
        allow_fail=True,
        log_cmd=False,
    )

    if proc.returncode != 0:
        # Fallback: try non-long listing (older CLI / edge cases). Type will be unknown.
        proc2 = run_command(
            ["earthengine", "ls", asset_id],
            dry_run=dry_run,
            timeout_s=timeout_s,
            retries=retries,
            allow_fail=True,
            log_cmd=False,
        )
        if proc2.returncode != 0:
            return None
        children: List[ListedChild] = []
        for line in (proc2.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            children.append(ListedChild(asset_id=normalize_asset_id(line), asset_type=""))
        return children

    children: List[ListedChild] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        # Expected: "<Type> <asset_id>"
        parts = line.split(None, 1)
        if len(parts) == 1:
            child_type = ""
            child_id = parts[0]
        else:
            child_type, child_id = parts[0], parts[1]
        children.append(ListedChild(asset_id=normalize_asset_id(child_id), asset_type=child_type))

    return children


def asset_exists(asset_id: str, *, timeout_s: int, retries: int) -> bool:
    asset_id = normalize_asset_id(asset_id)
    retries = max(1, retries)
    for _ in range(retries):
        try:
            proc = subprocess.run(
                ["earthengine", "asset", "info", asset_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_s,
            )
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            continue
    return False


def set_public(asset_id: str, *, dry_run: bool, timeout_s: int, retries: int) -> None:
    asset_id = normalize_asset_id(asset_id)
    run_command(
        ["earthengine", "acl", "set", "public", asset_id],
        dry_run=dry_run,
        timeout_s=timeout_s,
        retries=retries,
        allow_fail=True,
    )


def iter_assets_recursive(
    root: str,
    *,
    dry_run: bool,
    timeout_s: int,
    retries: int,
    match: Optional[re.Pattern[str]] = None,
    exclude: Optional[re.Pattern[str]] = None,
    scan_log_seconds: int = 30,
) -> Iterable[str]:
    """
    Depth-first walk of assets under root.

    - Lists containers with `earthengine ls -l` and uses returned type to avoid `ls` calls on leaf assets.
    - Yields leaf assets only (Images, Tables, etc).
    - Containers are traversed even if they don't match `match` (children might).
    """
    root = normalize_asset_id(root)
    stack: List[str] = [root]

    containers_seen = 0
    leaves_found = 0
    last_log = time.time()

    while stack:
        current = stack.pop()
        current_norm = normalize_asset_id(current)

        if exclude and exclude.search(current_norm):
            continue

        # Periodic scan progress log so it never looks "stuck".
        if scan_log_seconds > 0 and (time.time() - last_log) >= scan_log_seconds:
            LOG.info(
                "Scanning... containers_seen=%d leaves_found=%d stack=%d last=%s",
                containers_seen,
                leaves_found,
                len(stack),
                current_norm,
            )
            last_log = time.time()

        children = ee_ls_typed(current_norm, dry_run=dry_run, timeout_s=timeout_s, retries=retries)
        if children is None:
            # Treat as leaf
            if match and not match.search(current_norm):
                continue
            leaves_found += 1
            yield current_norm
            continue

        containers_seen += 1

        # Push only containers; yield leaves immediately (no extra ls calls).
        # Push in reverse to keep stable-ish ordering.
        for child in reversed(children):
            cid = child.asset_id

            if exclude and exclude.search(cid):
                continue

            if _is_container_type(child.asset_type) or child.asset_type == "":
                # Unknown type -> push to stack (fallback to discover via listing).
                stack.append(cid)
                continue

            # Leaf
            if match and not match.search(cid):
                continue
            leaves_found += 1
            yield cid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set public ACL on Earth Engine assets under a root folder/collection."
    )
    parser.add_argument(
        "--asset-root",
        required=True,
        help=(
            "Root Earth Engine asset folder/collection to traverse, e.g. "
            "users/you/root/run or users/you/root/run/2021_2024/40000_pixels/20260120"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Traverse recursively. If omitted, only leaf assets directly under --asset-root are processed.",
    )
    parser.add_argument("--match", help="Regex to include only matching leaf asset IDs (applied to full asset ID).")
    parser.add_argument("--exclude", help="Regex to exclude matching asset IDs (applied to full asset ID).")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without executing.")
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=0,
        help=(
            "If >0, wait up to this many seconds for --asset-root to exist before starting. "
            "Useful if you run immediately after submitting ingestion tasks."
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=10, help="Polling interval for --wait-seconds (default: 10).")
    parser.add_argument("--max-assets", type=int, default=0, help="If >0, stop after processing this many leaf assets.")

    # New: keep it from ever hanging silently/forever.
    parser.add_argument(
        "--cmd-timeout",
        type=int,
        default=300,
        help="Timeout (seconds) for each earthengine CLI command (default: 300).",
    )
    parser.add_argument("--retries", type=int, default=2, help="Retries for transient CLI failures/timeouts (default: 2).")
    parser.add_argument(
        "--scan-log-seconds",
        type=int,
        default=30,
        help="Emit a scan progress log line every N seconds (0 disables).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Set ACLs as leaves are discovered (no pre-scan list, no total count).",
    )
    parser.add_argument(
        "--sleep-between",
        type=float,
        default=0.0,
        help="Optional sleep (seconds) between ACL updates (helps if you hit rate limits).",
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
            if asset_exists(root, timeout_s=args.cmd_timeout, retries=args.retries):
                break
            time.sleep(max(1, args.poll_seconds))
        else:
            LOG.warning("Timed out waiting for asset root to exist: %s", root)

    if args.recursive:
        if args.stream:
            LOG.info("Publishing assets (streaming) under: %s", root)
            published = 0
            for asset_id in iter_assets_recursive(
                root,
                dry_run=args.dry_run,
                timeout_s=args.cmd_timeout,
                retries=args.retries,
                match=match_re,
                exclude=exclude_re,
                scan_log_seconds=args.scan_log_seconds,
            ):
                published += 1
                if args.max_assets > 0 and published > args.max_assets:
                    break
                LOG.info("(%d) Setting public ACL: %s", published, asset_id)
                set_public(asset_id, dry_run=args.dry_run, timeout_s=args.cmd_timeout, retries=args.retries)
                if args.sleep_between > 0:
                    time.sleep(args.sleep_between)
            if published == 0:
                LOG.warning("No leaf assets found to publish under: %s", root)
            return

        # Non-streaming (keeps the "total count" UX)
        LOG.info("Scanning for leaf assets under: %s", root)
        targets = list(
            iter_assets_recursive(
                root,
                dry_run=args.dry_run,
                timeout_s=args.cmd_timeout,
                retries=args.retries,
                match=match_re,
                exclude=exclude_re,
                scan_log_seconds=args.scan_log_seconds,
            )
        )
    else:
        # Non-recursive: only leaves directly under root.
        children = ee_ls_typed(root, dry_run=args.dry_run, timeout_s=args.cmd_timeout, retries=args.retries)
        if children is None:
            # Root is a leaf; treat it as the only target.
            targets = [root]
        else:
            targets = []
            for child in children:
                cid = child.asset_id
                if exclude_re and exclude_re.search(cid):
                    continue
                if match_re and not match_re.search(cid):
                    continue
                if _is_container_type(child.asset_type) or child.asset_type == "":
                    # Container or unknown -> skip unless recursive.
                    continue
                targets.append(cid)

    if args.max_assets > 0:
        targets = targets[: args.max_assets]

    if not targets:
        LOG.warning("No leaf assets found to publish under: %s", root)
        return

    total = len(targets)
    LOG.info("Publishing %d assets (public ACL): %s", total, root)
    for idx, asset_id in enumerate(targets, start=1):
        LOG.info("(%d/%d) Setting public ACL: %s", idx, total, asset_id)
        set_public(asset_id, dry_run=args.dry_run, timeout_s=args.cmd_timeout, retries=args.retries)
        if args.sleep_between > 0:
            time.sleep(args.sleep_between)


if __name__ == "__main__":
    main()
