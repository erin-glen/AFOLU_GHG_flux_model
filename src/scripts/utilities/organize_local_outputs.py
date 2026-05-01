"""Plan or apply migration of legacy local AFOLU outputs into C:/tmp/afolu.

Default execution is a dry-run report. Use ``--apply`` to move known outputs
and write a CSV manifest. Unknown top-level entries are reported unless
``--include-unknown`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from src.scripts.utilities import local_output_paths as lop


@dataclass(frozen=True)
class MigrationEntry:
    source: Path
    destination: Optional[Path]
    type: str
    action: str


def _display_raster_name(name: str) -> bool:
    lower = name.lower()
    suffix = Path(name).suffix.lower()
    if suffix not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        return False
    return (
        lower.startswith("compare_drained_binary")
        or lower == "biome_combined_state.tif"
        or lower.startswith("global_drained_state")
        or "display" in lower
    )


def _threshold_csv_name(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(".csv") and ("threshold" in lower or lower.startswith("area_vs_threshold"))


def _known_destination(name: str, target_root: Path) -> Optional[Path]:
    lower = name.lower()
    if lower == "pub_assets":
        return target_root / "publications" / "assets"
    if lower == "pub_fao":
        return target_root / "publications" / "fao"
    if lower.startswith("pub_nghgi"):
        return target_root / "publications" / "nghgi" / "legacy" / name
    if lower.startswith("global_drained_state") or lower == "state_map_v2" or _display_raster_name(name):
        return target_root / "visualization" / "legacy" / name
    if lower == "uncertainty" or _threshold_csv_name(name):
        return target_root / "analysis" / "legacy" / name
    return None


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a.absolute() == b.absolute()


def plan_moves(source_root: Path | str, target_root: Path | str, *, include_unknown: bool = False) -> list[MigrationEntry]:
    """Return planned top-level moves from ``source_root`` into ``target_root``."""

    source = Path(source_root)
    target = Path(target_root)
    if not source.exists():
        raise FileNotFoundError(f"Source root does not exist: {source}")

    entries: list[MigrationEntry] = []
    for child in sorted(source.iterdir(), key=lambda p: p.name.lower()):
        if _same_path(child, target):
            entries.append(MigrationEntry(child, None, "target_root", "skip"))
            continue

        dest = _known_destination(child.name, target)
        if dest is not None:
            action = "conflict" if dest.exists() else "move"
            entries.append(MigrationEntry(child, dest, "known", action))
            continue

        if include_unknown:
            dest = target / "unknown" / "legacy" / child.name
            action = "conflict" if dest.exists() else "move"
            entries.append(MigrationEntry(child, dest, "unknown", action))
        else:
            entries.append(MigrationEntry(child, None, "unknown", "report"))

    return entries


def apply_plan(entries: Iterable[MigrationEntry]) -> list[MigrationEntry]:
    """Move entries whose planned action is ``move`` and return result rows."""

    results: list[MigrationEntry] = []
    for entry in entries:
        if entry.action != "move" or entry.destination is None:
            results.append(entry)
            continue

        if entry.destination.exists():
            results.append(replace(entry, action="conflict"))
            continue

        entry.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(entry.source), str(entry.destination))
        results.append(replace(entry, action="moved"))
    return results


def default_manifest_path(target_root: Path | str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return Path(target_root) / "migration_manifests" / f"local_output_migration_{timestamp}.csv"


def write_manifest(entries: Iterable[MigrationEntry], manifest_path: Path | str) -> Path:
    """Write a CSV manifest containing source, destination, type, and action."""

    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source", "destination", "type", "action"])
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "source": str(entry.source),
                    "destination": "" if entry.destination is None else str(entry.destination),
                    "type": entry.type,
                    "action": entry.action,
                }
            )
    return path


def print_report(entries: Iterable[MigrationEntry]) -> None:
    print("source,destination,type,action")
    for entry in entries:
        destination = "" if entry.destination is None else str(entry.destination)
        print(f"{entry.source},{destination},{entry.type},{entry.action}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or apply local AFOLU output organization.")
    parser.add_argument("--source-root", default="C:/tmp", help="Legacy local output root to scan.")
    parser.add_argument("--target-root", default=lop.local_output_root(), help="Organized AFOLU local output root.")
    parser.add_argument("--dry-run", action="store_true", help="Report planned moves only (default).")
    parser.add_argument("--apply", action="store_true", help="Move planned entries and write a CSV manifest.")
    parser.add_argument("--include-unknown", action="store_true", help="Also move unknown top-level entries.")
    parser.add_argument("--manifest", default=None, help="CSV manifest path. Defaults under target-root for --apply.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    entries = plan_moves(args.source_root, args.target_root, include_unknown=args.include_unknown)

    if not args.apply:
        print_report(entries)
        return 0

    results = apply_plan(entries)
    manifest_path = Path(args.manifest) if args.manifest else default_manifest_path(args.target_root)
    write_manifest(results, manifest_path)
    print_report(results)
    print(f"Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
