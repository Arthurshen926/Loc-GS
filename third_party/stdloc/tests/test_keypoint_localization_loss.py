import torch

from utils.localization_loss import (
    keypoint_reprojection_loss,
    locability_weighted_feature_loss,
    pose_guided_reprojection_loss,
)


def test_keypoint_reprojection_loss_is_low_for_aligned_descriptor_peak():
    rendered = torch.zeros(2, 4, 4)
    target = torch.zeros(2, 4, 4)
    rendered[0] = 1.0
    target[0] = 1.0
    rendered[:, 2, 1] = torch.tensor([0.0, 1.0])
    target[:, 2, 1] = torch.tensor([0.0, 1.0])
    scores = torch.zeros(4, 4)
    scores[2, 1] = 1.0

    loss, stats = keypoint_reprojection_loss(
        rendered,
        target,
        scores,
        max_keypoints=1,
        temperature=0.01,
    )

    assert loss.item() < 1e-3
    assert stats["keypoints"].item() == 1


def test_keypoint_reprojection_loss_increases_for_shifted_descriptor_peak():
    rendered = torch.zeros(2, 4, 4)
    target = torch.zeros(2, 4, 4)
    rendered[0] = 1.0
    target[0] = 1.0
    rendered[:, 1, 3] = torch.tensor([0.0, 1.0])
    target[:, 2, 1] = torch.tensor([0.0, 1.0])
    scores = torch.zeros(4, 4)
    scores[2, 1] = 1.0

    loss, _stats = keypoint_reprojection_loss(
        rendered,
        target,
        scores,
        max_keypoints=1,
        temperature=0.01,
    )

    assert loss.item() > 0.3


def test_pose_guided_reprojection_loss_is_low_for_geometrically_aligned_match():
    rendered = torch.zeros(2, 4, 4)
    target = torch.zeros(2, 4, 4)
    rendered[0] = 1.0
    target[0] = 1.0
    rendered[:, 2, 1] = torch.tensor([0.0, 1.0])
    target[:, 2, 1] = torch.tensor([0.0, 1.0])
    scores = torch.zeros(4, 4)
    scores[2, 1] = 1.0
    depth = torch.ones(4, 4)
    pose = torch.eye(4)
    K = torch.eye(3)

    loss, stats = pose_guided_reprojection_loss(
        rendered,
        target,
        scores,
        depth,
        pose,
        pose,
        K,
        max_keypoints=1,
        temperature=0.01,
        target_sigma_px=0.05,
    )

    assert loss.item() < 1e-3
    assert stats["keypoints"].item() == 1


def test_pose_guided_reprojection_loss_increases_for_wrong_geometric_match():
    rendered = torch.zeros(2, 4, 4)
    target = torch.zeros(2, 4, 4)
    rendered[0] = 1.0
    target[0] = 1.0
    rendered[:, 1, 3] = torch.tensor([0.0, 1.0])
    target[:, 2, 1] = torch.tensor([0.0, 1.0])
    scores = torch.zeros(4, 4)
    scores[2, 1] = 1.0
    depth = torch.ones(4, 4)
    pose = torch.eye(4)
    K = torch.eye(3)

    loss, _stats = pose_guided_reprojection_loss(
        rendered,
        target,
        scores,
        depth,
        pose,
        pose,
        K,
        max_keypoints=1,
        temperature=0.01,
        target_sigma_px=0.05,
    )

    assert loss.item() > 0.3


def test_locability_weighted_feature_loss_focuses_high_locability_pixels():
    target = torch.zeros(1, 2, 2)
    rendered = target.clone()
    rendered[:, 0, 0] = 2.0
    rendered[:, 1, 1] = 2.0
    locability = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)

    weighted, stats = locability_weighted_feature_loss(
        rendered,
        target,
        locability,
        min_weight=0.1,
        gamma=1.0,
    )
    uniform = (rendered - target).abs().mean()

    assert weighted > uniform
    assert stats["selected_weight_mean"] > stats["background_weight_mean"]


def test_locability_weighted_feature_loss_downweights_background_errors():
    target = torch.zeros(1, 2, 2)
    rendered = target.clone()
    rendered[:, 1, 1] = 2.0
    locability = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)

    weighted, _stats = locability_weighted_feature_loss(
        rendered,
        target,
        locability,
        min_weight=0.1,
        gamma=1.0,
    )
    uniform = (rendered - target).abs().mean()

    assert weighted < uniform


def test_locability_weighted_feature_loss_respects_top_ratio_budget():
    target = torch.zeros(1, 2, 2)
    locability = torch.tensor([[[0.9, 0.8], [0.1, 0.0]]], dtype=torch.float32)

    rendered_top = target.clone()
    rendered_top[:, 0, 0] = 2.0
    weighted_top, stats = locability_weighted_feature_loss(
        rendered_top,
        target,
        locability,
        min_weight=0.05,
        gamma=1.0,
        top_ratio=0.25,
    )

    rendered_background = target.clone()
    rendered_background[:, 0, 1] = 2.0
    weighted_background, _stats = locability_weighted_feature_loss(
        rendered_background,
        target,
        locability,
        min_weight=0.05,
        gamma=1.0,
        top_ratio=0.25,
    )
    uniform = (rendered_top - target).abs().mean()

    assert weighted_top > uniform
    assert weighted_background < uniform
    assert torch.isclose(stats["selected_fraction"], torch.tensor(0.25))
    assert stats["selected_weight_mean"] > stats["background_weight_mean"]


def test_locability_weighted_feature_loss_detaches_selection_scores():
    target = torch.zeros(1, 2, 2)
    rendered = target.clone()
    rendered[:, 0, 0] = 2.0
    rendered.requires_grad_(True)
    locability = torch.tensor(
        [[[0.9, 0.8], [0.1, 0.0]]], dtype=torch.float32, requires_grad=True
    )

    weighted, _stats = locability_weighted_feature_loss(
        rendered,
        target,
        locability,
        min_weight=0.05,
        gamma=1.0,
        top_ratio=0.25,
    )
    weighted.backward()

    assert locability.grad is None
