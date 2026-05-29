import torch
import torch.nn.functional as F

from loc_gs.stdloc_native.solver_weighted_feature_fusion import solver_weighted_landmark_feature_fusion
from loc_gs.stdloc_native.solver_weighted_feature_fusion import solver_weighted_fusion_from_pair_cache


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
