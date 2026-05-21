import torch

from loc_gs.stdloc_native.solver_admissibility import (
    build_safe_core_from_tuple_bank,
    is_replacement_admissible,
)


def test_build_safe_core_from_tuple_bank_keeps_landmarks_in_best_hard_query_tuples():
    tuple_bank = {
        "queries": [
            {
                "query_id": 0,
                "tuples": [
                    {"landmark_ids": [0, 1, 2, 3], "viable": True, "geometry_logdet": 5.0},
                    {"landmark_ids": [4, 5, 6, 7], "viable": True, "geometry_logdet": 1.0},
                ],
            },
            {
                "query_id": 1,
                "tuples": [
                    {"landmark_ids": [8, 9, 10, 11], "viable": True, "geometry_logdet": 9.0},
                ],
            },
        ]
    }

    safe_core, meta = build_safe_core_from_tuple_bank(
        tuple_bank,
        hard_query_ids={0},
        source_idx=torch.arange(12),
        top_tuples_per_query=1,
    )

    assert safe_core.tolist() == [0, 1, 2, 3]
    assert meta["hard_query_count"] == 1
    assert meta["safe_core_count"] == 4


def test_replacement_admissibility_rejects_logdet_drop_on_hard_queries():
    source_loss = {
        2: {
            0: {"support": 1.0, "viable_tuple_mass": 1.0, "logdet_H": 2.0, "ambiguity": 0.0},
        }
    }
    candidate_gain = {
        4: {
            0: {"support": 1.0, "viable_tuple_mass": 1.0, "logdet_H": 0.5, "ambiguity": 0.0},
        }
    }

    decision = is_replacement_admissible(
        add_id=4,
        drop_id=2,
        hard_query_ids={0},
        candidate_gain=candidate_gain,
        source_loss=source_loss,
        min_logdet_delta=-0.2,
    )

    assert decision.admissible is False
    assert "logdet_drop" in decision.reasons


def test_replacement_admissibility_accepts_equal_support_and_better_logdet():
    source_loss = {
        2: {
            0: {"support": 1.0, "viable_tuple_mass": 1.0, "logdet_H": 2.0, "ambiguity": 0.2},
        }
    }
    candidate_gain = {
        4: {
            0: {"support": 1.0, "viable_tuple_mass": 1.1, "logdet_H": 2.5, "ambiguity": 0.1},
        }
    }

    decision = is_replacement_admissible(
        add_id=4,
        drop_id=2,
        hard_query_ids={0},
        candidate_gain=candidate_gain,
        source_loss=source_loss,
        min_logdet_delta=-0.2,
    )

    assert decision.admissible is True
    assert decision.delta["logdet_H"] == 0.5


def test_replacement_admissibility_can_require_candidate_gain():
    decision = is_replacement_admissible(
        add_id=4,
        drop_id=2,
        hard_query_ids={0},
        candidate_gain={},
        source_loss={},
        require_candidate_gain=True,
    )

    assert decision.admissible is False
    assert "missing_candidate_gain" in decision.reasons
