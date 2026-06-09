import pytest
import torch

from loc_gs.scripts.prune_ulfloc_descriptor_conflicts import (
    build_argparser,
    protect_scores_from_sparse_validation_profile,
    source_anchor_idx_from_sparse_validation_profile,
)
from loc_gs.stdloc_native.descriptor_conflict_pruning import (
    compute_descriptor_conflict_scores,
    prune_pairwise_descriptor_conflicts,
    prune_descriptor_conflicts,
)


def test_descriptor_conflict_pruning_keeps_source_and_removes_far_descriptor_clone():
    sampled_idx = torch.tensor([10, 20, 30], dtype=torch.long)
    source_anchor_idx = torch.tensor([10], dtype=torch.long)
    features = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    xyz = torch.zeros((31, 3), dtype=torch.float32)
    xyz[10] = torch.tensor([0.0, 0.0, 0.0])
    xyz[20] = torch.tensor([5.0, 0.0, 0.0])
    xyz[30] = torch.tensor([8.0, 0.0, 0.0])

    result = prune_descriptor_conflicts(
        sampled_idx=sampled_idx,
        features=features,
        xyz=xyz,
        source_anchor_idx=source_anchor_idx,
        min_cosine=0.95,
        min_spatial_distance_m=1.0,
    )

    assert result.keep_mask.tolist() == [True, False, True]
    assert result.pruned_sampled_idx.tolist() == [10, 30]
    assert result.metadata["source_kept_count"] == 1
    assert result.metadata["descriptor_conflict_pruned_count"] == 1


def test_descriptor_conflict_pruning_does_not_remove_near_duplicate_support():
    sampled_idx = torch.tensor([10, 20], dtype=torch.long)
    source_anchor_idx = torch.tensor([10], dtype=torch.long)
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    xyz = torch.zeros((21, 3), dtype=torch.float32)
    xyz[10] = torch.tensor([0.0, 0.0, 0.0])
    xyz[20] = torch.tensor([0.2, 0.0, 0.0])

    result = prune_descriptor_conflicts(
        sampled_idx=sampled_idx,
        features=features,
        xyz=xyz,
        source_anchor_idx=source_anchor_idx,
        min_cosine=0.95,
        min_spatial_distance_m=1.0,
    )

    assert result.keep_mask.tolist() == [True, True]
    assert result.metadata["descriptor_conflict_pruned_count"] == 0


def test_descriptor_conflict_pruning_keeps_solver_protected_far_conflict():
    sampled_idx = torch.tensor([10, 20, 30], dtype=torch.long)
    source_anchor_idx = torch.tensor([10], dtype=torch.long)
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    xyz = torch.zeros((31, 3), dtype=torch.float32)
    xyz[20] = torch.tensor([5.0, 0.0, 0.0])
    xyz[30] = torch.tensor([6.0, 0.0, 0.0])
    protect_scores = torch.zeros(31, dtype=torch.float32)
    protect_scores[20] = 0.8
    protect_scores[30] = 0.1

    result = prune_descriptor_conflicts(
        sampled_idx=sampled_idx,
        features=features,
        xyz=xyz,
        source_anchor_idx=source_anchor_idx,
        min_cosine=0.95,
        min_spatial_distance_m=1.0,
        protect_scores=protect_scores,
        min_protect_score=0.5,
    )

    assert result.keep_mask.tolist() == [True, True, False]
    assert result.metadata["descriptor_conflict_candidate_count"] == 1
    assert result.metadata["solver_protected_conflict_count"] == 1


def test_pairwise_descriptor_conflict_pruning_removes_lower_supported_far_twin_without_source_anchor():
    sampled_idx = torch.tensor([10, 20, 30], dtype=torch.long)
    features = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    xyz = torch.zeros((31, 3), dtype=torch.float32)
    xyz[10] = torch.tensor([0.0, 0.0, 0.0])
    xyz[20] = torch.tensor([5.0, 0.0, 0.0])
    xyz[30] = torch.tensor([0.5, 0.0, 0.0])
    protect_scores = torch.zeros(31, dtype=torch.float32)
    protect_scores[10] = 1.0
    protect_scores[20] = 0.1

    result = prune_pairwise_descriptor_conflicts(
        sampled_idx=sampled_idx,
        features=features,
        xyz=xyz,
        protect_scores=protect_scores,
        min_protect_score=0.5,
        min_cosine=0.95,
        min_spatial_distance_m=1.0,
        max_prune_count=1,
        top_k=1,
    )

    assert result.pruned_sampled_idx.tolist() == [10, 30]
    assert result.keep_mask.tolist() == [True, False, True]
    assert result.metadata["method"] == "pairwise_descriptor_conflict_pruning"
    assert result.metadata["descriptor_conflict_pruned_count"] == 1


def test_descriptor_conflict_scores_report_nearest_source_cosine_and_distance():
    sampled_idx = torch.tensor([1, 2], dtype=torch.long)
    source_anchor_idx = torch.tensor([1], dtype=torch.long)
    features = torch.tensor([[1.0, 0.0], [0.6, 0.8]], dtype=torch.float32)
    xyz = torch.zeros((3, 3), dtype=torch.float32)
    xyz[2] = torch.tensor([3.0, 0.0, 0.0])

    scores = compute_descriptor_conflict_scores(
        sampled_idx=sampled_idx,
        features=features,
        xyz=xyz,
        source_anchor_idx=source_anchor_idx,
        min_cosine=0.5,
        min_spatial_distance_m=1.0,
    )

    assert torch.allclose(scores.nearest_source_cosine, torch.tensor([1.0, 0.6]))
    assert torch.allclose(scores.nearest_source_distance_m, torch.tensor([0.0, 3.0]))
    assert scores.conflict_score[0].item() == 0.0
    assert scores.conflict_score[1].item() > 0.0


def test_prune_ulfloc_descriptor_conflict_argparser_accepts_required_paths():
    args = build_argparser().parse_args(
        [
            "--input_model_path",
            "/models/in",
            "--input_log_dir",
            "/models/in/log",
            "--output_model_path",
            "/models/out",
            "--conflict_mode",
            "source_anchor",
            "--source_anchor_idx",
            "/models/native/keypoints_sampled_idx.pkl",
            "--cfg",
            "/models/in/log/config.yaml",
            "--output_log_name",
            "conflict_pruned",
            "--min_cosine",
            "0.96",
            "--min_spatial_distance_m",
            "2.0",
            "--protect_feedback",
            "/feedback/solver_feedback.pkl",
            "--protect_key",
            "landmark_support",
            "--min_protect_score",
            "0.2",
        ]
    )

    assert str(args.input_model_path) == "/models/in"
    assert str(args.input_log_dir) == "/models/in/log"
    assert str(args.output_model_path) == "/models/out"
    assert str(args.source_anchor_idx) == "/models/native/keypoints_sampled_idx.pkl"
    assert args.output_log_name == "conflict_pruned"
    assert args.min_cosine == 0.96
    assert args.min_spatial_distance_m == 2.0
    assert str(args.protect_feedback) == "/feedback/solver_feedback.pkl"
    assert args.protect_key == "landmark_support"
    assert args.min_protect_score == 0.2


def test_prune_ulfloc_descriptor_conflict_argparser_accepts_pairwise_mode_without_source_anchor():
    args = build_argparser().parse_args(
        [
            "--input_model_path",
            "/models/in",
            "--input_log_dir",
            "/models/in/log",
            "--output_model_path",
            "/models/out",
            "--cfg",
            "/models/in/log/config.yaml",
            "--output_log_name",
            "pairwise_conflict_pruned",
            "--conflict_mode",
            "pairwise",
            "--pairwise_top_k",
            "2",
        ]
    )

    assert args.conflict_mode == "pairwise"
    assert args.source_anchor_idx is None
    assert args.pairwise_top_k == 2


def test_sparse_validation_profile_exports_solver_validated_anchors_and_protect_scores():
    profile = {
        "split_name": "train_dev_seed13_20p_validation",
        "protected_per_query_support": {
            "q_easy.png": {"10": 2.0, "20": 1.0},
        },
        "validated_per_query_support": {
            "q_hard.png": {"20": 3.0, "30": 4.0},
        },
    }

    anchors = source_anchor_idx_from_sparse_validation_profile(profile)
    scores = protect_scores_from_sparse_validation_profile(profile, num_gaussians=40)

    assert anchors.tolist() == [10, 20, 30]
    assert scores[10].item() == pytest.approx(2.0)
    assert scores[20].item() == pytest.approx(4.0)
    assert scores[30].item() == pytest.approx(4.0)
    assert scores.sum().item() == pytest.approx(10.0)


def test_sparse_validation_profile_helpers_reject_test_split():
    profile = {
        "split_name": "test",
        "protected_per_query_support": {"q.png": {"10": 1.0}},
    }

    with pytest.raises(ValueError, match="test split"):
        source_anchor_idx_from_sparse_validation_profile(profile)
    with pytest.raises(ValueError, match="test split"):
        protect_scores_from_sparse_validation_profile(profile, num_gaussians=20)


def test_prune_ulfloc_descriptor_conflict_argparser_accepts_sparse_validation_profile_anchors():
    args = build_argparser().parse_args(
        [
            "--input_model_path",
            "/models/in",
            "--input_log_dir",
            "/models/in/log",
            "--output_model_path",
            "/models/out",
            "--cfg",
            "/models/in/log/config.yaml",
            "--output_log_name",
            "validation_anchor_conflict_pruned",
            "--conflict_mode",
            "source_anchor",
            "--sparse_validation_profile",
            "/profiles/sparse_pnp_validation_profile.json",
            "--source_anchor_from_sparse_validation_profile",
            "--protect_sparse_validation_profile",
        ]
    )

    assert str(args.sparse_validation_profile) == "/profiles/sparse_pnp_validation_profile.json"
    assert args.source_anchor_from_sparse_validation_profile is True
    assert args.protect_sparse_validation_profile is True
