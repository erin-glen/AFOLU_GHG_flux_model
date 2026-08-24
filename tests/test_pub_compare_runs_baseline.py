from src.scripts.zonal_statistics.pub_scripts import pub_compare_runs as pcr


def test_comparisons_for_baseline_preserves_default_contract():
    comparisons = pcr._comparisons_for_baseline(pcr.DEFAULT_BASELINE_RUN_NAME)

    assert comparisons == tuple(pcr.COMPARISONS)


def test_comparisons_for_baseline_replaces_only_canonical_baseline():
    corrected = "ogh_mixed_f1_f15_f2_20260513_starting_cpf_global_wdpa_fixed"

    comparisons = pcr._comparisons_for_baseline(corrected)

    for original, updated in zip(pcr.COMPARISONS, comparisons, strict=True):
        expected = tuple(
            corrected if run_name == pcr.DEFAULT_BASELINE_RUN_NAME else run_name
            for run_name in original.run_names
        )
        assert updated.run_names == expected
        assert updated.key == original.key
        assert updated.label == original.label
        assert updated.metric_keys == original.metric_keys


def test_partition_comparisons_accepts_corrected_baseline_specs():
    corrected = "ogh_mixed_f1_f15_f2_20260513_starting_cpf_global_wdpa_fixed"
    comparisons = pcr._comparisons_for_baseline(corrected)
    inventory = next(comp for comp in comparisons if comp.key == "inventory_source")
    run_specs = {run_name: object() for run_name in inventory.run_names}

    active, missing = pcr._partition_comparisons(run_specs, comparisons)

    assert [comp.key for comp in active] == ["inventory_source"]
    assert "inventory_source" not in missing


def test_comparisons_for_baseline_rejects_empty_name():
    for invalid in ("", "   "):
        try:
            pcr._comparisons_for_baseline(invalid)
        except ValueError as exc:
            assert "non-empty" in str(exc)
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("Expected an empty baseline run name to fail")
