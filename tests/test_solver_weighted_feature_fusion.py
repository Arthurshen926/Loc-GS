import torch
import torch.nn.functional as F
import pytest

from loc_gs.stdloc_native.solver_weighted_feature_fusion import solver_weighted_landmark_feature_fusion
from loc_gs.stdloc_native.solver_weighted_feature_fusion import solver_weighted_fusion_from_pair_cache
from loc_gs.stdloc_native.solver_weighted_feature_fusion import guard_solver_weighted_fusion_from_validation_profile


def test_solver_weighted_feature_fusion_blends_supported_landmark_and_keeps_fallback():
    native = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        p=2,
        dim=-1,
    )
    observations = F.normalize(
        torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        p=2,
        dim=-1,
    )

    fused, metadata = solver_weighted_landmark_feature_fusion(
        native_descriptors=native,
        landmark_ids=torch.tensor([0, 0, 1], dtype=torch.long),
        observed_descriptors=observations,
        solver_weights=torch.tensor([1.0, 3.0, 2.0], dtype=torch.float32),
        geometry_weights=torch.tensor([1.0, 1.0, 0.5], dtype=torch.float32),
        visibility_weights=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        trust_alpha=0.5,
        min_total_weight=0.1,
    )

    expected_landmark0_observation = F.normalize(torch.tensor([3.0, 1.0]), p=2, dim=0)
    expected_landmark0 = F.normalize(0.5 * native[0] + 0.5 * expected_landmark0_observation, p=2, dim=0)

    assert torch.allclose(fused[0], expected_landmark0, atol=1e-6)
    assert not torch.allclose(fused[1], native[1])
    assert torch.allclose(fused[2], native[2])
    assert metadata["fused_landmark_count"] == 2
    assert metadata["fallback_landmark_count"] == 1


def test_solver_weighted_fusion_from_pair_cache_uses_query_descriptors_for_low_error_pairs():
    cache = {
        "base_landmark_desc": F.normalize(torch.eye(3, dtype=torch.float32), p=2, dim=-1),
        "base_gaussian_id": torch.tensor([100, 101, 102], dtype=torch.long),
        "query_desc": F.normalize(
            torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
            p=2,
            dim=-1,
        ),
        "landmark_id": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "candidate_mask": torch.tensor([[True, True], [True, True]]),
        "cosine": torch.tensor([[0.9, 0.2], [0.8, 0.7]], dtype=torch.float32),
        "reprojection_error": torch.tensor([[1.0, 10.0], [2.0, 1.0]], dtype=torch.float32),
        "metadata": {"source_split_name": "train", "feedback_bank_split_name": "selfmap_train_rendered"},
    }

    result = solver_weighted_fusion_from_pair_cache(
        cache,
        trust_alpha=0.5,
        reprojection_threshold_px=3.0,
        min_cosine=0.5,
        min_native_cosine=-1.0,
    )

    assert result["gaussian_ids"].tolist() == [100, 101, 102]
    assert result["metadata"]["positive_pair_count"] == 3
    assert result["metadata"]["descriptor_mode"] == "solver_weighted_pair_cache_fusion"
    assert not torch.allclose(result["descriptors"][0], cache["base_landmark_desc"][0])
    assert not torch.allclose(result["descriptors"][1], cache["base_landmark_desc"][1])
    assert torch.allclose(result["base_descriptors"], F.normalize(cache["base_landmark_desc"].float(), p=2, dim=-1))


def test_solver_weighted_fusion_can_filter_pairs_with_active_fusion_plan():
    native = F.normalize(torch.eye(3, dtype=torch.float32), p=2, dim=-1)
    cache = {
        "base_landmark_desc": native,
        "base_gaussian_id": torch.tensor([10, 11, 12], dtype=torch.long),
        "query_desc": F.normalize(
            torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            p=2,
            dim=-1,
        ),
        "landmark_id": torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
        "candidate_mask": torch.ones(2, 2, dtype=torch.bool),
        "cosine": torch.full((2, 2), 0.9, dtype=torch.float32),
        "reprojection_error": torch.ones(2, 2, dtype=torch.float32),
        "row_source_view_id": ["view_a", "view_b"],
        "metadata": {"source_split_name": "selfmap_train"},
    }
    active_plan = {
        "split_name": "selfmap_train",
        "landmark_fusion_plan": {
            "10": {"selected_view_ids": ["view_a"]},
            "12": {"selected_view_ids": ["view_b"]},
        },
    }

    result = solver_weighted_fusion_from_pair_cache(
        cache,
        trust_alpha=0.5,
        reprojection_threshold_px=4.0,
        min_cosine=0.5,
        min_native_cosine=-1.0,
        active_fusion_plan=active_plan,
    )

    assert result["metadata"]["active_fusion_plan_enabled"] is True
    assert result["metadata"]["active_fusion_plan_positive_pair_count"] == 2
    assert result["metadata"]["positive_pair_count"] == 2
    assert not torch.allclose(result["descriptors"][0], native[0])
    assert torch.allclose(result["descriptors"][1], native[1])
    assert not torch.allclose(result["descriptors"][2], native[2])


def test_solver_weighted_fusion_requires_row_source_view_id_for_active_plan():
    native = F.normalize(torch.eye(2, dtype=torch.float32), p=2, dim=-1)
    cache = {
        "base_landmark_desc": native,
        "base_gaussian_id": torch.tensor([10, 11], dtype=torch.long),
        "query_desc": native[:1],
        "landmark_id": torch.tensor([[0, 1]], dtype=torch.long),
        "candidate_mask": torch.ones(1, 2, dtype=torch.bool),
        "cosine": torch.ones(1, 2, dtype=torch.float32),
        "reprojection_error": torch.ones(1, 2, dtype=torch.float32),
        "metadata": {"source_split_name": "selfmap_train"},
    }

    with pytest.raises(ValueError, match="row_source_view_id"):
        solver_weighted_fusion_from_pair_cache(
            cache,
            active_fusion_plan={"split_name": "selfmap_train", "landmark_fusion_plan": {"10": {"selected_view_ids": ["v"]}}},
        )


def test_solver_weighted_fusion_from_pair_cache_can_guard_sparse_or_large_shift_landmarks():
    native = F.normalize(torch.eye(3, dtype=torch.float32), p=2, dim=-1)
    cache = {
        "base_landmark_desc": native,
        "base_gaussian_id": torch.tensor([10, 11, 12], dtype=torch.long),
        "query_desc": F.normalize(
            torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            p=2,
            dim=-1,
        ),
        "landmark_id": torch.tensor([[0], [0], [1]], dtype=torch.long),
        "candidate_mask": torch.tensor([[True], [True], [True]]),
        "cosine": torch.tensor([[0.9], [0.9], [0.9]], dtype=torch.float32),
        "reprojection_error": torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32),
        "metadata": {"source_split_name": "train"},
    }

    result = solver_weighted_fusion_from_pair_cache(
        cache,
        trust_alpha=0.5,
        reprojection_threshold_px=3.0,
        min_cosine=0.5,
        min_observations_per_landmark=2,
        min_native_cosine=0.99,
    )

    assert torch.allclose(result["descriptors"], native)
    assert result["metadata"]["eligible_landmark_count"] == 1
    assert result["metadata"]["updated_landmark_count"] == 0
    assert result["metadata"]["reverted_low_observation_count"] == 1
    assert result["metadata"]["reverted_shift_count"] == 1


def test_solver_weighted_fusion_defaults_to_conservative_v9_trust_region():
    native = F.normalize(torch.eye(2, dtype=torch.float32), p=2, dim=-1)
    cache = {
        "base_landmark_desc": native,
        "query_desc": F.normalize(torch.tensor([[0.0, 1.0]], dtype=torch.float32), p=2, dim=-1),
        "landmark_id": torch.tensor([[0]], dtype=torch.long),
        "candidate_mask": torch.tensor([[True]]),
        "cosine": torch.tensor([[0.99]], dtype=torch.float32),
        "reprojection_error": torch.tensor([[0.1]], dtype=torch.float32),
        "metadata": {"source_split_name": "train"},
    }

    result = solver_weighted_fusion_from_pair_cache(cache)

    assert result["metadata"]["trust_alpha"] == 0.1
    assert result["metadata"]["min_native_cosine"] == 0.95
    assert result["metadata"]["descriptor_mode"] == "solver_weighted_pair_cache_fusion"


def test_solver_weighted_fusion_uses_pair_cache_split_name_for_audit_metadata():
    native = F.normalize(torch.eye(2, dtype=torch.float32), p=2, dim=-1)
    cache = {
        "base_landmark_desc": native,
        "query_desc": F.normalize(torch.tensor([[1.0, 0.0]], dtype=torch.float32), p=2, dim=-1),
        "landmark_id": torch.tensor([[0]], dtype=torch.long),
        "candidate_mask": torch.tensor([[True]]),
        "cosine": torch.tensor([[0.99]], dtype=torch.float32),
        "reprojection_error": torch.tensor([[0.1]], dtype=torch.float32),
        "metadata": {"split_name": "train_dev_seed13"},
    }

    result = solver_weighted_fusion_from_pair_cache(cache)

    assert result["metadata"]["source_split_name"] == "train_dev_seed13"


def test_guard_solver_weighted_fusion_reverts_regression_landmarks_and_keeps_improvements():
    native = F.normalize(torch.eye(4, dtype=torch.float32), p=2, dim=-1)
    artifact = {
        "descriptors": F.normalize(
            native
            + torch.tensor(
                [
                    [0.0, 0.5, 0.0, 0.0],
                    [0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.5],
                    [0.0, 0.0, 0.5, 0.0],
                ],
                dtype=torch.float32,
            ),
            p=2,
            dim=-1,
        ),
        "base_descriptors": native,
        "gaussian_ids": torch.tensor([10, 11, 12, 13], dtype=torch.long),
        "metadata": {"source_split_name": "train_dev"},
    }
    pair_cache = {
        "landmark_id": torch.tensor([[0], [2], [1], [3]], dtype=torch.long),
        "candidate_mask": torch.ones(4, 1, dtype=torch.bool),
        "cosine": torch.full((4, 1), 0.9),
        "reprojection_error": torch.ones(4, 1),
        "metadata": {
            "split_name": "train_dev",
            "processed_images": 2,
            "query_keypoint_count": 4,
            "topk": 1,
            "reprojection_threshold_px": 4.0,
        },
    }
    profile = {
        "split_name": "train_dev",
        "paired_queries": [
            {"query_id": "q_regress", "delta_sparse_te_cm": 30.0, "label": "regression"},
            {"query_id": "q_improve", "delta_sparse_te_cm": -40.0, "label": "improvement"},
        ],
    }

    guarded = guard_solver_weighted_fusion_from_validation_profile(
        artifact,
        pair_cache,
        profile,
        regression_margin_cm=20.0,
        improvement_margin_cm=20.0,
    )

    assert torch.allclose(guarded["descriptors"][0], native[0])
    assert torch.allclose(guarded["descriptors"][2], native[2])
    assert not torch.allclose(guarded["descriptors"][1], native[1])
    assert not torch.allclose(guarded["descriptors"][3], native[3])
    assert guarded["metadata"]["guarded_landmark_count"] == 2
    assert guarded["metadata"]["regression_query_count"] == 1
    assert guarded["metadata"]["improvement_query_count"] == 1


def test_guard_solver_weighted_fusion_preserves_landmark_with_stronger_improvement_credit():
    native = F.normalize(torch.eye(2, dtype=torch.float32), p=2, dim=-1)
    artifact = {
        "descriptors": F.normalize(native + torch.tensor([[0.0, 0.5], [0.5, 0.0]]), p=2, dim=-1),
        "base_descriptors": native,
        "gaussian_ids": torch.tensor([10, 11], dtype=torch.long),
        "metadata": {"source_split_name": "train_dev"},
    }
    pair_cache = {
        "landmark_id": torch.tensor([[0], [0]], dtype=torch.long),
        "candidate_mask": torch.ones(2, 1, dtype=torch.bool),
        "cosine": torch.full((2, 1), 0.9),
        "reprojection_error": torch.ones(2, 1),
        "metadata": {
            "split_name": "train_dev",
            "processed_images": 2,
            "query_keypoint_count": 2,
            "topk": 1,
        },
    }
    profile = {
        "split_name": "train_dev",
        "paired_queries": [
            {"query_id": "q_regress", "delta_sparse_te_cm": 25.0},
            {"query_id": "q_improve", "delta_sparse_te_cm": -50.0},
        ],
    }

    guarded = guard_solver_weighted_fusion_from_validation_profile(
        artifact,
        pair_cache,
        profile,
        regression_margin_cm=20.0,
        improvement_margin_cm=20.0,
    )

    assert not torch.allclose(guarded["descriptors"][0], native[0])
    assert guarded["metadata"]["guarded_landmark_count"] == 0


def test_guard_solver_weighted_fusion_can_limit_risk_to_protected_regressions():
    native = F.normalize(torch.eye(3, dtype=torch.float32), p=2, dim=-1)
    artifact = {
        "descriptors": F.normalize(native + 0.25 * torch.roll(native, shifts=1, dims=1), p=2, dim=-1),
        "base_descriptors": native,
        "gaussian_ids": torch.tensor([10, 11, 12], dtype=torch.long),
        "metadata": {"source_split_name": "train_dev"},
    }
    pair_cache = {
        "landmark_id": torch.tensor([[0], [1], [2]], dtype=torch.long),
        "candidate_mask": torch.ones(3, 1, dtype=torch.bool),
        "cosine": torch.full((3, 1), 0.9),
        "reprojection_error": torch.ones(3, 1),
        "metadata": {"split_name": "train_dev", "processed_images": 3, "query_keypoint_count": 3, "topk": 1},
    }
    profile = {
        "split_name": "train_dev",
        "paired_queries": [
            {"query_id": "hard_regress", "delta_sparse_te_cm": 100.0, "protected_by_baseline": False},
            {"query_id": "protected_regress", "delta_sparse_te_cm": 30.0, "protected_by_baseline": True},
            {"query_id": "hard_improve", "delta_sparse_te_cm": -30.0, "hard_by_baseline": True},
        ],
    }

    guarded = guard_solver_weighted_fusion_from_validation_profile(
        artifact,
        pair_cache,
        profile,
        guard_query_scope="protected_regressions",
        regression_margin_cm=20.0,
        improvement_margin_cm=20.0,
    )

    assert not torch.allclose(guarded["descriptors"][0], native[0])
    assert torch.allclose(guarded["descriptors"][1], native[1])
    assert not torch.allclose(guarded["descriptors"][2], native[2])
    assert guarded["metadata"]["guard_query_scope"] == "protected_regressions"
    assert guarded["metadata"]["regression_query_count"] == 1


def test_guard_solver_weighted_fusion_uses_v2_direct_attribution_risk_and_support():
    native = F.normalize(torch.eye(2, dtype=torch.float32), p=2, dim=-1)
    artifact = {
        "descriptors": F.normalize(native + torch.tensor([[0.0, 0.5], [0.5, 0.0]]), p=2, dim=-1),
        "base_descriptors": native,
        "gaussian_ids": torch.tensor([10, 11], dtype=torch.long),
        "metadata": {"source_split_name": "train_dev"},
    }
    pair_cache = {
        "landmark_id": torch.tensor([[0], [1]], dtype=torch.long),
        "candidate_mask": torch.ones(2, 1, dtype=torch.bool),
        "cosine": torch.full((2, 1), 0.9),
        "reprojection_error": torch.ones(2, 1),
        "metadata": {"split_name": "train_dev", "processed_images": 1, "query_keypoint_count": 2, "topk": 1},
    }
    profile = {
        "schema": "loc_gs_sparse_pnp_validation_profile_v2",
        "split_name": "train_dev",
        "attribution_status": "attributed",
        "landmark_regression_risk": {"10": 100.0, "11": 1.0},
        "validated_per_query_support": {"q_hard": {"11": 100.0}},
    }

    guarded = guard_solver_weighted_fusion_from_validation_profile(
        artifact,
        pair_cache,
        profile,
        regression_margin_cm=20.0,
        improvement_margin_cm=20.0,
    )

    assert torch.allclose(guarded["descriptors"][0], native[0])
    assert not torch.allclose(guarded["descriptors"][1], native[1])
    assert guarded["metadata"]["direct_profile_regression_risk_landmark_count"] == 2
    assert guarded["metadata"]["direct_profile_support_credit_landmark_count"] == 1
    assert guarded["metadata"]["validation_profile_attribution_status"] == "attributed"
