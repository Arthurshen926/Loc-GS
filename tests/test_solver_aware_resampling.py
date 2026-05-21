import pytest
import torch

from loc_gs.stdloc_native.solver_aware_resampling import (
    apply_selected_local_edits,
    parse_edit_selection,
    solver_aware_local_edit,
)


def test_solver_aware_local_edit_replaces_only_non_safe_source_with_evidence_candidate():
    result = solver_aware_local_edit(
        source_idx=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        candidate_pool=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long),
        utility=torch.tensor([0.9, 0.8, 0.2, 0.3, 0.95, 0.99], dtype=torch.float32),
        evidence_mask=torch.tensor([True, True, True, True, True, False]),
        safe_core=torch.tensor([0, 1], dtype=torch.long),
        max_edits=2,
    )

    assert result["sampled_idx"].numel() == 4
    assert {0, 1}.issubset(set(result["sampled_idx"].tolist()))
    assert 4 in result["sampled_idx"].tolist()
    assert 5 not in result["sampled_idx"].tolist()
    assert result["metadata"]["safe_core_dropped_count"] == 0
    assert result["metadata"]["num_admissible_replacements"] == 1


def test_solver_aware_local_edit_honors_admissibility_callback():
    result = solver_aware_local_edit(
        source_idx=torch.tensor([0, 1, 2], dtype=torch.long),
        candidate_pool=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        utility=torch.tensor([0.9, 0.8, 0.1, 0.95], dtype=torch.float32),
        evidence_mask=torch.ones(4, dtype=torch.bool),
        safe_core=torch.tensor([], dtype=torch.long),
        is_admissible=lambda add_id, drop_id: add_id != 3,
        max_edits=1,
    )

    assert result["sampled_idx"].tolist() == [0, 1, 2]
    assert result["metadata"]["num_rejected_by_admissibility"] == 1


def test_parse_edit_selection_uses_one_based_ranges_and_prefixes():
    assert parse_edit_selection("1-3,5,prefix:2", total_edits=6) == [0, 1, 2, 4]
    assert parse_edit_selection("all", total_edits=3) == [0, 1, 2]

    with pytest.raises(ValueError, match="outside"):
        parse_edit_selection("4", total_edits=3)


def test_apply_selected_local_edits_can_skip_prefix_steps():
    edits = [
        {"add_id": 4, "drop_id": 2, "utility_gain": 0.8},
        {"add_id": 5, "drop_id": 1, "utility_gain": 0.7},
        {"add_id": 6, "drop_id": 3, "utility_gain": 0.6},
    ]

    result = apply_selected_local_edits(
        source_idx=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        edits=edits,
        utility=torch.tensor([0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7], dtype=torch.float32),
        selected_edit_indices=[0, 2],
    )

    assert result["sampled_idx"].tolist() == [0, 1, 4, 6]
    assert result["metadata"]["applied_edit_count"] == 2
    assert result["metadata"]["skipped_conflict_count"] == 0
    assert result["metadata"]["same_budget"] is True


def test_apply_selected_local_edits_reports_unsatisfied_dependencies():
    edits = [
        {"add_id": 2, "drop_id": 0, "utility_gain": 0.8},
        {"add_id": 3, "drop_id": 2, "utility_gain": 0.7},
    ]

    result = apply_selected_local_edits(
        source_idx=torch.tensor([0, 1], dtype=torch.long),
        edits=edits,
        utility=torch.tensor([0.2, 0.3, 0.8, 0.9], dtype=torch.float32),
        selected_edit_indices=[1],
        strict_conflicts=False,
    )

    assert result["sampled_idx"].tolist() == [0, 1]
    assert result["metadata"]["applied_edit_count"] == 0
    assert result["metadata"]["skipped_conflict_count"] == 1

    with pytest.raises(ValueError, match="cannot drop"):
        apply_selected_local_edits(
            source_idx=torch.tensor([0, 1], dtype=torch.long),
            edits=edits,
            utility=torch.tensor([0.2, 0.3, 0.8, 0.9], dtype=torch.float32),
            selected_edit_indices=[1],
            strict_conflicts=True,
        )
