import pytest

from loc_gs.stdloc_native.bank_level_resampling import build_large_edit_resampling_policy


def test_bank_level_resampling_requires_saturation_and_conflict_controls_for_large_edits():
    with pytest.raises(ValueError, match="large-edit"):
        build_large_edit_resampling_policy(
            source_count=1000,
            replacement_fraction=0.2,
            saturation_enabled=False,
            conflict_graph_available=True,
        )

    policy = build_large_edit_resampling_policy(
        source_count=1000,
        replacement_fraction=0.2,
        saturation_enabled=True,
        conflict_graph_available=True,
        min_safe_core_fraction=0.3,
    )

    assert policy["max_edits"] == 200
    assert policy["safe_core_min_count"] == 300
    assert policy["stage"] == "bank_level_resampling"
    assert policy["requires_train_dev_gate"] is True
