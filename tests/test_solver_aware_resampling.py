import pytest
import torch

from loc_gs.stdloc_native.solver_aware_resampling import (
    apply_selected_local_edits,
    parse_edit_selection,
    score_hard_query_cvar_utility,
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


def test_solver_aware_local_edit_can_scan_next_drop_for_admissible_replacement():
    result = solver_aware_local_edit(
        source_idx=torch.tensor([0, 1, 2], dtype=torch.long),
        candidate_pool=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        utility=torch.tensor([0.4, 0.2, 0.1, 0.95], dtype=torch.float32),
        evidence_mask=torch.ones(4, dtype=torch.bool),
        safe_core=torch.tensor([], dtype=torch.long),
        is_admissible=lambda add_id, drop_id: drop_id != 2,
        max_edits=1,
        max_drop_scan=2,
    )

    assert result["sampled_idx"].tolist() == [0, 2, 3]
    assert result["metadata"]["num_rejected_by_admissibility"] == 1
    assert result["edits"][0]["drop_id"] == 1


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


def test_hard_query_cvar_rejects_positive_mean_when_tail_regresses():
    result = score_hard_query_cvar_utility(
        {
            101: {"support": 1.0, "viable_tuple_mass": 0.5, "logdet_H": 0.0, "ambiguity": 0.0},
            102: {"support": 0.75, "viable_tuple_mass": 0.0, "logdet_H": 0.0, "ambiguity": 0.0},
            103: {"support": -2.0, "viable_tuple_mass": 0.0, "logdet_H": 0.0, "ambiguity": 0.0},
        },
        weights={"support": 1.0, "viable_tuple_mass": 1.0, "logdet_H": 0.0, "ambiguity": -1.0},
        alpha=1.0 / 3.0,
        min_score=0.0,
    )

    assert result["metadata"]["query_count"] == 3
    assert result["metadata"]["mean"] > 0.0
    assert result["metadata"]["cvar"] == pytest.approx(-2.0)
    assert result["metadata"]["accepted"] is False
    assert result["score"] < 0.0


def test_hard_query_cvar_accepts_string_image_id_query_ids():
    result = score_hard_query_cvar_utility(
        {
            "seq-000001.jpg": {"support": 0.25, "viable_tuple_mass": 0.5, "logdet_H": 0.25, "ambiguity": 0.1},
            "seq-000002.jpg": {"support": 0.5, "viable_tuple_mass": 0.25, "logdet_H": 0.25, "ambiguity": 0.0},
        },
        weights={"support": 1.0, "viable_tuple_mass": 1.0, "logdet_H": 0.5, "ambiguity": -1.0},
        alpha=0.5,
        min_score=0.5,
    )

    assert result["metadata"]["query_count"] == 2
    assert result["metadata"]["alpha"] == pytest.approx(0.5)
    assert result["metadata"]["accepted"] is True
    assert set(result["per_query_utility"]) == {"seq-000001.jpg", "seq-000002.jpg"}
