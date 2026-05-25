import torch
import pytest

from loc_gs.stdloc_native.solver_admissibility import (
    build_safe_core_from_tuple_bank,
    is_replacement_admissible,
    make_replacement_admissibility_checker,
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


def test_build_safe_core_from_tuple_bank_accepts_feedback_bank_v2_string_query_ids():
    tuple_bank = {
        "queries": [
            {
                "query_id": "seq1/frame00001.png",
                "tuples": [
                    {"landmark_ids": [10, 11, 12, 13], "viable": True, "geometry_logdet": 4.0},
                ],
            },
        ]
    }

    safe_core, meta = build_safe_core_from_tuple_bank(
        tuple_bank,
        hard_query_ids={"seq1/frame00001.png"},
        source_idx=torch.arange(20),
        top_tuples_per_query=1,
    )

    assert safe_core.tolist() == [10, 11, 12, 13]
    assert meta["hard_query_count"] == 1
    assert meta["used_query_count"] == 1


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


def test_replacement_admissibility_checker_reuses_normalized_constraints():
    source_loss = {
        2: {
            0: {"support": 1.0, "viable_tuple_mass": 1.0, "logdet_H": 2.0, "ambiguity": 0.0},
        }
    }
    candidate_gain = {
        4: {
            0: {"support": 1.0, "viable_tuple_mass": 1.0, "logdet_H": 2.5, "ambiguity": 0.0},
        }
    }

    checker = make_replacement_admissibility_checker(
        hard_query_ids={0},
        candidate_gain=candidate_gain,
        source_loss=source_loss,
        min_logdet_delta=0.0,
    )

    first = checker(add_id=4, drop_id=2)
    second = checker(add_id=4, drop_id=2)

    assert first.admissible is True
    assert second.admissible is True
    assert second.delta["logdet_H"] == pytest.approx(0.5)


def test_replacement_admissibility_accepts_feedback_bank_v2_string_query_ids():
    source_loss = {
        2: {
            "seq1/frame00001.png": {"support": 1.0, "viable_tuple_mass": 1.0, "logdet_H": 2.0, "ambiguity": 0.1},
        }
    }
    candidate_gain = {
        4: {
            "seq1/frame00001.png": {"support": 1.0, "viable_tuple_mass": 1.2, "logdet_H": 2.4, "ambiguity": 0.05},
        }
    }

    decision = is_replacement_admissible(
        add_id=4,
        drop_id=2,
        hard_query_ids={"seq1/frame00001.png"},
        candidate_gain=candidate_gain,
        source_loss=source_loss,
        min_logdet_delta=0.0,
    )

    assert decision.admissible is True
    assert decision.delta["logdet_H"] == pytest.approx(0.4)


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


def test_replacement_admissibility_rejects_dense_worsen_risk_rise():
    source_loss = {
        2: {
            "hard/image.png": {
                "support": 1.0,
                "viable_tuple_mass": 1.0,
                "logdet_H": 2.0,
                "min_eigenvalue": 0.4,
                "dense_worsen_risk": 0.1,
                "ambiguity": 0.1,
            },
        }
    }
    candidate_gain = {
        4: {
            "hard/image.png": {
                "support": 1.0,
                "viable_tuple_mass": 1.0,
                "logdet_H": 2.2,
                "min_eigenvalue": 0.5,
                "dense_worsen_risk": 0.6,
                "ambiguity": 0.1,
            },
        }
    }

    decision = is_replacement_admissible(
        add_id=4,
        drop_id=2,
        hard_query_ids={"hard/image.png"},
        candidate_gain=candidate_gain,
        source_loss=source_loss,
        max_dense_worsen_delta=0.0,
    )

    assert decision.admissible is False
    assert "dense_worsen_rise" in decision.reasons
    assert decision.delta["dense_worsen_risk"] == pytest.approx(0.5)


def test_replacement_admissibility_rejects_min_eigen_drop():
    source_loss = {
        2: {
            0: {
                "support": 1.0,
                "viable_tuple_mass": 1.0,
                "logdet_H": 2.0,
                "min_eigenvalue": 0.5,
                "dense_worsen_risk": 0.1,
                "ambiguity": 0.1,
            },
        }
    }
    candidate_gain = {
        4: {
            0: {
                "support": 1.0,
                "viable_tuple_mass": 1.0,
                "logdet_H": 2.0,
                "min_eigenvalue": 0.1,
                "dense_worsen_risk": 0.1,
                "ambiguity": 0.1,
            },
        }
    }

    decision = is_replacement_admissible(
        add_id=4,
        drop_id=2,
        hard_query_ids={0},
        candidate_gain=candidate_gain,
        source_loss=source_loss,
        min_min_eigen_delta=0.0,
    )

    assert decision.admissible is False
    assert "min_eigen_drop" in decision.reasons
    assert decision.delta["min_eigenvalue"] == pytest.approx(-0.4)


def test_replacement_admissibility_rejects_hard_query_cvar_tail_regression():
    source_loss = {
        2: {
            "easy": {"support": 0.0, "viable_tuple_mass": 0.0, "logdet_H": 0.0, "ambiguity": 0.0},
            "hard": {"support": 0.0, "viable_tuple_mass": 0.0, "logdet_H": 0.0, "ambiguity": 0.0},
        }
    }
    candidate_gain = {
        4: {
            "easy": {"support": 3.0, "viable_tuple_mass": 0.0, "logdet_H": 0.0, "ambiguity": 0.0},
            "hard": {"support": -2.0, "viable_tuple_mass": 0.0, "logdet_H": 0.0, "ambiguity": 0.0},
        }
    }

    decision = is_replacement_admissible(
        add_id=4,
        drop_id=2,
        hard_query_ids={"easy", "hard"},
        candidate_gain=candidate_gain,
        source_loss=source_loss,
        min_support_delta=-10.0,
        min_logdet_delta=-10.0,
        cvar_alpha=0.5,
        min_cvar_score=0.0,
        cvar_weights={
            "support": 1.0,
            "viable_tuple_mass": 0.0,
            "logdet_H": 0.0,
            "min_eigenvalue": 0.0,
            "ambiguity": 0.0,
            "dense_worsen_risk": 0.0,
        },
    )

    assert decision.admissible is False
    assert "hard_query_cvar" in decision.reasons
    assert decision.cvar is not None
    assert decision.cvar["metadata"]["cvar"] == pytest.approx(-2.0)
