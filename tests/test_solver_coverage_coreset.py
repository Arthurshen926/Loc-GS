import torch

from loc_gs.stdloc_native.solver_coverage_coreset import (
    build_solver_coverage_tables,
    solver_coverage_local_edit,
)


def _constraints():
    return {
        "hard_query_ids": ["q0", "q1"],
        "candidate_gain": {
            "3": {"q0": {"support": 2.0, "viable_tuple_mass": 1.0, "logdet_H": 1.0}},
            "4": {"q1": {"support": 2.0, "viable_tuple_mass": 1.0, "logdet_H": 1.0}},
            "5": {"q0": {"support": 1.0, "viable_tuple_mass": 0.1, "dense_worsen_risk": 3.0}},
        },
        "source_loss": {
            "0": {"q0": {"support": 1.0, "viable_tuple_mass": 1.0, "logdet_H": 1.0}},
            "1": {"q1": {"support": 1.0, "viable_tuple_mass": 1.0, "logdet_H": 1.0}},
            "2": {},
        },
        "thresholds": {
            "cvar_weights": {
                "support": 1.0,
                "viable_tuple_mass": 1.0,
                "logdet_H": 1.0,
                "min_eigenvalue": 1.0,
                "dense_worsen_risk": -1.0,
                "ambiguity": -1.0,
            }
        },
    }


def test_build_solver_coverage_tables_parses_string_query_ids():
    tables = build_solver_coverage_tables(_constraints())

    assert tables.hard_query_ids == ("q0", "q1")
    assert tables.candidate_gain[3]["q0"]["support"] == 2.0
    assert tables.source_loss[1]["q1"]["viable_tuple_mass"] == 1.0


def test_solver_coverage_local_edit_prefers_uncovered_queries_and_preserves_budget():
    tables = build_solver_coverage_tables(_constraints())
    result = solver_coverage_local_edit(
        source_idx=torch.tensor([0, 1, 2]),
        candidate_pool=torch.tensor([3, 4, 5]),
        utility=torch.tensor([0.2, 0.2, 0.0, 0.9, 0.8, 0.95]),
        evidence_mask=torch.ones(6, dtype=torch.bool),
        coverage_tables=tables,
        max_edits=2,
        min_coverage_gain=0.1,
    )

    sampled = set(result["sampled_idx"].tolist())
    assert len(sampled) == 3
    assert {3, 4}.issubset(sampled)
    assert 5 not in sampled
    assert result["metadata"]["coverage_policy"] == "solver_coverage_coreset"


def test_solver_coverage_local_edit_protects_source_support_for_low_coverage_query():
    tables = build_solver_coverage_tables(_constraints())
    result = solver_coverage_local_edit(
        source_idx=torch.tensor([0, 1, 2]),
        candidate_pool=torch.tensor([3]),
        utility=torch.tensor([0.0, 0.0, 0.1, 0.9]),
        evidence_mask=torch.ones(4, dtype=torch.bool),
        coverage_tables=tables,
        max_edits=1,
        min_coverage_gain=0.1,
    )

    sampled = set(result["sampled_idx"].tolist())
    assert 1 in sampled
    assert 3 in sampled
    assert result["metadata"]["coverage_protected_drop_count"] >= 1


def test_solver_coverage_local_edit_limits_drop_cost_search_window():
    source_count = 200
    payload = {
        "hard_query_ids": ["q0"],
        "candidate_gain": {
            str(source_count): {"q0": {"support": 2.0, "viable_tuple_mass": 1.0, "logdet_H": 1.0}},
            str(source_count + 1): {"q0": {"support": 1.0}},
        },
        "source_loss": {"0": {"q0": {"support": 1.0}}},
        "thresholds": {},
    }
    tables = build_solver_coverage_tables(payload)

    result = solver_coverage_local_edit(
        source_idx=torch.arange(source_count),
        candidate_pool=torch.tensor([source_count, source_count + 1]),
        utility=torch.cat([torch.zeros(source_count), torch.tensor([0.9, 0.8])]),
        evidence_mask=torch.ones(source_count + 2, dtype=torch.bool),
        coverage_tables=tables,
        max_edits=1,
        max_drop_scan=4,
    )

    assert source_count in result["sampled_idx"].tolist()
    assert result["metadata"]["drop_cost_evaluations"] <= 64


def test_solver_coverage_local_edit_saturates_easy_query_gain():
    tables = build_solver_coverage_tables(
        {
            "hard_query_ids": ["easy", "hard"],
            "candidate_gain": {
                "3": {"easy": {"support": 1000.0}},
                "4": {"hard": {"support": 2.0}},
            },
            "source_loss": {
                "0": {"easy": {"support": 100.0}},
                "1": {"hard": {"support": 1.0}},
                "2": {},
            },
            "thresholds": {},
        }
    )

    result = solver_coverage_local_edit(
        source_idx=torch.tensor([0, 1, 2]),
        candidate_pool=torch.tensor([3, 4]),
        utility=torch.tensor([0.0, 0.0, 0.1, 0.99, 0.5]),
        evidence_mask=torch.ones(5, dtype=torch.bool),
        coverage_tables=tables,
        max_edits=1,
        coverage_saturation_mode="native_percentile",
        coverage_saturation_percentile=0.5,
        hard_query_min_gain=0.1,
    )

    sampled = set(result["sampled_idx"].tolist())
    assert 4 in sampled
    assert 3 not in sampled
    assert result["metadata"]["coverage_saturation"]["enabled"] is True
    assert result["metadata"]["coverage_saturation"]["saturated_query_count"] == 1


def test_solver_coverage_local_edit_uses_saturation_target_to_protect_drops():
    tables = build_solver_coverage_tables(
        {
            "hard_query_ids": ["easy", "hard"],
            "candidate_gain": {
                "3": {"hard": {"support": 2.0}},
            },
            "source_loss": {
                "0": {"easy": {"support": 100.0}},
                "1": {"hard": {"support": 10.0}},
                "2": {},
            },
            "thresholds": {},
        }
    )

    result = solver_coverage_local_edit(
        source_idx=torch.tensor([0, 1, 2]),
        candidate_pool=torch.tensor([3]),
        utility=torch.tensor([0.0, 0.0, 0.1, 0.9]),
        evidence_mask=torch.ones(4, dtype=torch.bool),
        coverage_tables=tables,
        max_edits=1,
        coverage_saturation_mode="native_percentile",
        coverage_saturation_percentile=0.5,
        saturation_drop_protection_weight=1.0,
    )

    sampled = set(result["sampled_idx"].tolist())
    assert 1 in sampled
    assert 2 not in sampled
    assert 3 in sampled
    assert result["metadata"]["coverage_saturation"]["drop_protection_weight"] == 1.0


def test_solver_coverage_local_edit_penalizes_pairwise_negative_conflicts():
    tables = build_solver_coverage_tables(
        {
            "hard_query_ids": ["q0"],
            "candidate_gain": {
                "2": {"q0": {"support": 5.0}},
                "3": {"q0": {"support": 4.0}},
            },
            "source_loss": {
                "0": {"q0": {"support": 0.1}},
                "1": {"q0": {"support": 0.1}},
            },
        }
    )
    graph = {
        "edge_weights": {(1, 2): 100.0},
        "unary_risk": {},
    }

    result = solver_coverage_local_edit(
        source_idx=torch.tensor([0, 1]),
        candidate_pool=torch.tensor([2, 3]),
        utility=torch.tensor([0.1, 0.1, 10.0, 9.0]),
        evidence_mask=torch.ones(4, dtype=torch.bool),
        coverage_tables=tables,
        max_edits=1,
        max_drop_scan=1,
        negative_support_graph=graph,
        negative_conflict_pair_weight=1.0,
    )

    sampled = set(result["sampled_idx"].tolist())
    assert 3 in sampled
    assert 2 not in sampled
    assert result["metadata"]["negative_support_graph"]["enabled"] is True
