import pytest
import torch

from loc_gs.stdloc_native.query_conditioned_support import (
    build_query_conditioned_solver_constraints,
)


def _episode_payload():
    return {
        "query_id": torch.tensor([10, 10, 11], dtype=torch.long),
        "candidate_landmark_ids": torch.tensor(
            [
                [0, 2, 3],
                [1, 2, 3],
                [0, 3, 2],
            ],
            dtype=torch.long,
        ),
        "candidate_cosine": torch.tensor(
            [
                [0.90, 0.80, 0.40],
                [0.70, 0.60, 0.95],
                [0.85, 0.90, 0.20],
            ],
            dtype=torch.float32,
        ),
        "candidate_reprojection_error": torch.tensor(
            [
                [1.0, 2.0, 1.0],
                [1.5, 2.5, 9.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        "candidate_visible": torch.ones((3, 3), dtype=torch.bool),
        "candidate_pnp_inlier": torch.tensor(
            [
                [True, True, True],
                [True, True, False],
                [True, True, True],
            ]
        ),
        "metadata": {"split_name": "selfmap_train_rendered"},
    }


def test_query_conditioned_constraints_emit_hard_query_candidate_gain_and_source_loss():
    constraints = build_query_conditioned_solver_constraints(
        _episode_payload(),
        source_idx=torch.tensor([0, 1], dtype=torch.long),
        num_gaussians=4,
        hard_query_ids=[10],
        reprojection_threshold_px=3.0,
        score_threshold=0.5,
        min_candidate_positive=0.5,
    )

    assert constraints["hard_query_ids"] == [10]
    assert constraints["candidate_gain"]["2"]["10"]["support"] > 0.0
    assert constraints["candidate_gain"].get("3", {}).get("10", {}).get("support", 0.0) == 0.0
    assert constraints["source_loss"]["0"]["10"]["support"] > 0.0
    assert constraints["source_loss"]["1"]["10"]["support"] > 0.0
    assert constraints["metadata"]["query_group_mode"] == "query_id"
    assert constraints["metadata"]["split_name"] == "selfmap_train_rendered"


def test_query_conditioned_constraints_map_base_landmark_ids_to_gaussian_ids():
    payload = _episode_payload()
    payload["base_gaussian_id"] = torch.tensor([100, 101, 102, 103], dtype=torch.long)

    constraints = build_query_conditioned_solver_constraints(
        payload,
        source_idx=torch.tensor([100, 101], dtype=torch.long),
        num_gaussians=104,
        hard_query_ids=[10],
        reprojection_threshold_px=3.0,
        score_threshold=0.5,
    )

    assert "102" in constraints["candidate_gain"]
    assert "100" in constraints["source_loss"]
    assert "2" not in constraints["candidate_gain"]


def test_query_conditioned_constraints_require_real_query_id():
    payload = _episode_payload()
    payload.pop("query_id")

    with pytest.raises(KeyError, match="query_id"):
        build_query_conditioned_solver_constraints(
            payload,
            source_idx=torch.tensor([0, 1], dtype=torch.long),
            num_gaussians=4,
            hard_query_ids=[10],
        )
