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
