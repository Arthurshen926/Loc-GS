import torch
import torch.nn.functional as F

from loc_gs.stdloc_native.solver_feedback_feature_fusion_v2 import (
    fuse_landmark_descriptors,
    fuse_landmark_descriptors_contrastive,
)


def test_negative_views_are_excluded_and_native_anchor_kept():
    native = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32), p=2, dim=1)
    observations = {
        0: [
            {"descriptor": torch.tensor([1.0, 0.0]), "positive_weight": 1.0, "negative_weight": 0.0},
            {"descriptor": torch.tensor([0.0, 1.0]), "positive_weight": 0.0, "negative_weight": 5.0},
        ],
        1: [
            {"descriptor": torch.tensor([0.0, 1.0]), "positive_weight": 1.0, "negative_weight": 0.0},
        ],
    }

    fused, meta = fuse_landmark_descriptors(native, observations, trust_alpha=0.1, min_native_cosine=0.95)

    assert fused.shape == native.shape
    assert float((fused[0] * native[0]).sum()) >= 0.95
    assert meta["negative_view_excluded_count"] == 1
    assert meta["fused_landmark_count"] == 2


def test_trust_region_reverts_large_descriptor_shift():
    native = F.normalize(torch.tensor([[1.0, 0.0]], dtype=torch.float32), p=2, dim=1)
    observations = {
        0: [
            {"descriptor": torch.tensor([0.0, 1.0]), "positive_weight": 1.0, "negative_weight": 0.0},
        ]
    }

    fused, meta = fuse_landmark_descriptors(native, observations, trust_alpha=1.0, min_native_cosine=0.95)

    assert torch.allclose(fused, native)
    assert meta["trust_region_fallback_count"] == 1
    assert meta["fused_landmark_count"] == 0


def test_fallback_when_positive_views_are_insufficient_after_negative_exclusion():
    native = F.normalize(torch.tensor([[1.0, 0.0]], dtype=torch.float32), p=2, dim=1)
    observations = {
        0: [
            {"descriptor": torch.tensor([0.0, 1.0]), "positive_weight": 0.1, "negative_weight": 0.5},
        ]
    }

    fused, meta = fuse_landmark_descriptors(native, observations)

    assert torch.allclose(fused, native)
    assert meta["negative_view_excluded_count"] == 1
    assert meta["insufficient_positive_view_count"] == 1
    assert meta["fallback_count"] == 1


def test_contrastive_fusion_attracts_inlier_views_and_repels_outlier_views():
    native = F.normalize(torch.tensor([[0.8, 0.6]], dtype=torch.float32), p=2, dim=1)
    positive_view = F.normalize(torch.tensor([1.0, 0.0], dtype=torch.float32), p=2, dim=0)
    negative_view = F.normalize(torch.tensor([0.0, 1.0], dtype=torch.float32), p=2, dim=0)
    observations = {
        0: [
            {"descriptor": positive_view, "positive_weight": 2.0, "negative_weight": 0.0},
            {"descriptor": negative_view, "positive_weight": 0.0, "negative_weight": 4.0},
        ]
    }

    fused, meta = fuse_landmark_descriptors_contrastive(
        native,
        observations,
        min_native_cosine=0.0,
        contrastive_steps=40,
        contrastive_lr=0.2,
        anchor_weight=0.1,
        negative_weight_scale=1.0,
        negative_margin=0.0,
    )

    before_positive = float((native[0] * positive_view).sum())
    before_negative = float((native[0] * negative_view).sum())
    after_positive = float((fused[0] * positive_view).sum())
    after_negative = float((fused[0] * negative_view).sum())

    assert after_positive > before_positive
    assert after_negative < before_negative
    assert meta["descriptor_mode"] == "solver_feedback_contrastive_fusion_v1"
    assert meta["positive_view_selected_count"] == 1
    assert meta["negative_view_selected_count"] == 1
    assert meta["negative_view_excluded_count"] == 1
