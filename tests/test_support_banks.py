import pytest
import torch

from loc_gs.stdloc_native.support_banks import (
    build_named_support_banks,
    build_observable_query_features,
    build_query_support_banks,
    route_query_to_banks,
)


def test_query_support_banks_separate_viewpoint_clusters_and_aggregate_landmarks():
    result = build_query_support_banks(
        query_ids=["q0", "q1", "q2", "q3"],
        query_features=torch.tensor([[0.0], [0.1], [10.0], [10.1]], dtype=torch.float32),
        landmark_support={
            11: {"q0": 2.0, "q1": 1.0},
            22: {"q2": 3.0, "q3": 1.0},
            33: {"q1": 0.5, "q2": 0.5},
        },
        bank_count=2,
    )

    assert len(result["banks"]) == 2
    left, right = result["banks"]
    assert left["query_ids"] == ["q0", "q1"]
    assert right["query_ids"] == ["q2", "q3"]
    assert left["landmark_weights"][11] > left["landmark_weights"].get(22, 0.0)
    assert right["landmark_weights"][22] > right["landmark_weights"].get(11, 0.0)

    routed = route_query_to_banks(torch.tensor([9.9]), result["router"], topk=1)
    assert routed == [1]


def test_observable_query_features_reject_gt_and_result_fields():
    observations = [
        {
            "query_id": "q0",
            "keypoint_count": 100,
            "detector_centroid_yx": [0.4, 0.6],
            "topk_match_entropy": 0.3,
            "sparse_inliers": 40,
            "pnp_confidence": 0.8,
            "global_descriptor": [1.0, 0.0],
            "gt_pose": [0, 0, 0],
        }
    ]

    with pytest.raises(ValueError, match="forbidden"):
        build_observable_query_features(observations)


def test_named_support_banks_build_fixed_safe_artifact_and_top2_router():
    observations = [
        {"query_id": "easy", "keypoint_count": 200, "detector_centroid_yx": [0.5, 0.5], "topk_match_entropy": 0.1, "sparse_inliers": 80, "pnp_confidence": 0.9},
        {"query_id": "amb", "keypoint_count": 180, "detector_centroid_yx": [0.4, 0.5], "topk_match_entropy": 0.9, "sparse_inliers": 30, "pnp_confidence": 0.4},
        {"query_id": "occ", "keypoint_count": 40, "detector_centroid_yx": [0.7, 0.2], "topk_match_entropy": 0.4, "sparse_inliers": 8, "pnp_confidence": 0.2},
        {"query_id": "tail", "keypoint_count": 90, "detector_centroid_yx": [0.1, 0.9], "topk_match_entropy": 0.7, "sparse_inliers": 12, "pnp_confidence": 0.3},
    ]
    support = {
        1: {"easy": 3.0, "amb": 0.1},
        2: {"amb": 4.0},
        3: {"occ": 5.0},
        4: {"tail": 6.0},
    }

    banks = build_named_support_banks(
        query_observations=observations,
        landmark_support=support,
        native_safe_core_ids=[1],
        hard_query_ids=["tail"],
    )

    assert [bank["name"] for bank in banks["banks"]] == [
        "native_safe_core",
        "ambiguity_safe",
        "occlusion_robust",
        "hard_tail_recovery",
    ]
    assert banks["metadata"]["paper_safe_scope"] == "offline_train_or_selfmap_router"
    routed = route_query_to_banks(banks["query_features"][2], banks["router"], topk=2)
    assert len(routed) == 2


def test_named_support_banks_filters_hard_queries_not_in_observations():
    banks = build_named_support_banks(
        query_observations=[
            {"query_id": "seen", "keypoint_count": 100, "detector_centroid_yx": [0.5, 0.5], "topk_match_entropy": 0.2, "sparse_inliers": 30, "pnp_confidence": 0.5},
        ],
        landmark_support={7: {"unseen": 3.0}},
        native_safe_core_ids=[],
        hard_query_ids=["unseen"],
    )

    hard_tail = next(bank for bank in banks["banks"] if bank["name"] == "hard_tail_recovery")
    assert hard_tail["query_ids"] == []
    assert hard_tail["landmark_weights"] == {}
    assert banks["metadata"]["unmatched_hard_query_count"] == 1


def test_route_query_to_banks_skips_empty_banks_when_mask_is_available():
    router = {
        "mode": "nearest_centroid",
        "centroids": torch.tensor([[0.0], [0.1], [10.0]], dtype=torch.float32),
        "bank_ids": [0, 1, 2],
        "non_empty_bank_ids": [0, 2],
    }

    assert route_query_to_banks(torch.tensor([0.05]), router, topk=2) == [0, 2]
