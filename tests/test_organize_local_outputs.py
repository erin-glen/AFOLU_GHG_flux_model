import csv

from src.scripts.utilities import organize_local_outputs as org


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_dry_run_plan_does_not_move_files(tmp_path) -> None:
    source = tmp_path / "legacy_tmp"
    target = tmp_path / "afolu"
    (source / "pub_assets").mkdir(parents=True)
    _touch(source / "area_vs_threshold_20251105.csv")
    _touch(source / "mystery.txt")

    entries = org.plan_moves(source, target)

    assert (source / "pub_assets").exists()
    assert (source / "area_vs_threshold_20251105.csv").exists()
    assert (source / "mystery.txt").exists()
    assert {
        (entry.source.name, entry.destination.relative_to(target).as_posix() if entry.destination else "", entry.action)
        for entry in entries
    } == {
        ("area_vs_threshold_20251105.csv", "analysis/legacy/area_vs_threshold_20251105.csv", "move"),
        ("mystery.txt", "", "report"),
        ("pub_assets", "publications/assets", "move"),
    }


def test_apply_moves_known_mappings_and_writes_manifest(tmp_path) -> None:
    source = tmp_path / "legacy_tmp"
    target = tmp_path / "afolu"
    (source / "pub_assets").mkdir(parents=True)
    (source / "pub_fao").mkdir()
    (source / "pub_nghgi_final").mkdir()
    (source / "state_map_v2").mkdir()
    _touch(source / "compare_drained_binary.png")
    _touch(source / "area_vs_threshold_20251105.csv")
    _touch(source / "unknown.txt")

    planned = org.plan_moves(source, target)
    results = org.apply_plan(planned)
    manifest = org.write_manifest(results, target / "manifest.csv")

    assert not (source / "pub_assets").exists()
    assert (target / "publications" / "assets").exists()
    assert (target / "publications" / "fao").exists()
    assert (target / "publications" / "nghgi" / "legacy" / "pub_nghgi_final").exists()
    assert (target / "visualization" / "legacy" / "state_map_v2").exists()
    assert (target / "visualization" / "legacy" / "compare_drained_binary.png").exists()
    assert (target / "analysis" / "legacy" / "area_vs_threshold_20251105.csv").exists()
    assert (source / "unknown.txt").exists()

    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    assert {"source", "destination", "type", "action"} == set(rows[0])
    moved = {row["source"].split("\\")[-1].split("/")[-1]: row["action"] for row in rows}
    assert moved["pub_assets"] == "moved"
    assert moved["unknown.txt"] == "report"


def test_unknown_entries_move_only_when_requested(tmp_path) -> None:
    source = tmp_path / "legacy_tmp"
    target = tmp_path / "afolu"
    _touch(source / "unknown.txt")

    report_only = org.plan_moves(source, target, include_unknown=False)
    include_unknown = org.plan_moves(source, target, include_unknown=True)

    assert report_only[0].destination is None
    assert report_only[0].action == "report"
    assert include_unknown[0].destination == target / "unknown" / "legacy" / "unknown.txt"
    assert include_unknown[0].action == "move"
