import torch
import torch.nn.functional as F

from loc_gs.losses.differentiable_pnp import DifferentiablePnPMatchLoss


def test_differentiable_pnp_loss_backpropagates_to_matching_inputs():
    torch.manual_seed(7)
    query_descs = F.normalize(torch.randn(1, 4, 8), dim=-1)
    query_keypoints = torch.tensor(
        [[[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]]],
        dtype=torch.float32,
    )
    query_mask = torch.ones(1, 4, dtype=torch.bool)
    rendered_desc = F.normalize(torch.randn(1, 8, 4, 4), dim=1).requires_grad_(True)
    depth = torch.full((1, 4, 4), 4.0, requires_grad=True)
    locability = torch.zeros(1, 1, 4, 4, requires_grad=True)
    pose = torch.eye(4).unsqueeze(0)
    K = torch.tensor(
        [[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    loss_fn = DifferentiablePnPMatchLoss(
        temperature=0.2,
        pnp_iterations=2,
        locability_prior_weight=0.1,
    )
    out = loss_fn(
        query_descs=query_descs,
        query_keypoints_yx=query_keypoints,
        query_mask=query_mask,
        rendered_desc=rendered_desc,
        depth_map=depth,
        render_pose_w2c=pose,
        gt_pose_w2c=pose,
        K=K,
        locability_map=locability,
    )

    assert out["total"].ndim == 0
    assert out["pose_w2c"].shape == (1, 4, 4)
    out["total"].backward()
    assert rendered_desc.grad is not None
    assert torch.isfinite(rendered_desc.grad).all()
    assert depth.grad is not None
    assert torch.isfinite(depth.grad).all()
    assert locability.grad is not None


def test_match_target_uses_gt_pose_for_shifted_render_pose():
    channels = 16
    height = 4
    width = 4
    rendered_desc = torch.eye(channels, dtype=torch.float32).T.reshape(1, channels, height, width)
    depth = torch.ones(1, height, width)
    render_pose = torch.eye(4).unsqueeze(0)
    gt_pose = torch.eye(4).unsqueeze(0)
    gt_pose[:, 0, 3] = -1.0
    K = torch.eye(3)
    query_keypoints = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
    query_mask = torch.ones(1, 1, dtype=torch.bool)
    good_query_desc = rendered_desc.flatten(2).transpose(1, 2)[:, [2], :]
    bad_query_desc = rendered_desc.flatten(2).transpose(1, 2)[:, [1], :]

    loss_fn = DifferentiablePnPMatchLoss(
        temperature=0.05,
        pnp_iterations=0,
        pose_weight=0.0,
        quality_weight=0.0,
        reprojection_weight=0.0,
        observability_weight=0.0,
        locability_weight=0.0,
        target_sigma_px=0.25,
    )

    good = loss_fn(
        query_descs=good_query_desc,
        query_keypoints_yx=query_keypoints,
        query_mask=query_mask,
        rendered_desc=rendered_desc,
        depth_map=depth,
        render_pose_w2c=render_pose,
        gt_pose_w2c=gt_pose,
        K=K,
    )
    bad = loss_fn(
        query_descs=bad_query_desc,
        query_keypoints_yx=query_keypoints,
        query_mask=query_mask,
        rendered_desc=rendered_desc,
        depth_map=depth,
        render_pose_w2c=render_pose,
        gt_pose_w2c=gt_pose,
        K=K,
    )

    assert good["match"] < bad["match"]
