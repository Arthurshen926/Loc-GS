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


def test_locability_target_prior_map_is_accepted_and_backpropagates():
    query_descs = F.normalize(torch.randn(1, 4, 8), dim=-1)
    query_keypoints = torch.tensor(
        [[[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]]],
        dtype=torch.float32,
    )
    query_mask = torch.ones(1, 4, dtype=torch.bool)
    rendered_desc = F.normalize(torch.randn(1, 8, 4, 4), dim=1)
    depth = torch.full((1, 4, 4), 4.0)
    locability = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
    target_prior = torch.zeros(1, 1, 4, 4)
    target_prior[:, :, :2, :2] = 1.0
    pose = torch.eye(4).unsqueeze(0)
    K = torch.tensor(
        [[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    out = DifferentiablePnPMatchLoss(
        temperature=0.2,
        pnp_iterations=1,
        locability_target_prior_weight=1.0,
    )(
        query_descs=query_descs,
        query_keypoints_yx=query_keypoints,
        query_mask=query_mask,
        rendered_desc=rendered_desc,
        depth_map=depth,
        render_pose_w2c=pose,
        gt_pose_w2c=pose,
        K=K,
        locability_map=locability,
        locability_target_prior_map=target_prior,
    )

    out["total"].backward()
    assert torch.isfinite(out["locability"])
    assert locability.grad is not None


def test_queries_without_valid_gt_projection_do_not_supervise_pnp():
    channels = 8
    height = 4
    width = 4
    query_descs = F.normalize(torch.randn(1, 4, channels), dim=-1)
    query_keypoints = torch.tensor(
        [[[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]]],
        dtype=torch.float32,
    )
    rendered_desc = F.normalize(torch.randn(1, channels, height, width), dim=1)
    depth = torch.ones(1, height, width)
    render_pose = torch.eye(4).unsqueeze(0)
    gt_pose = torch.eye(4).unsqueeze(0)
    gt_pose[:, 0, 3] = 100.0
    K = torch.tensor(
        [[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    out = DifferentiablePnPMatchLoss(
        temperature=0.2,
        pnp_iterations=1,
        pose_weight=1.0,
        match_weight=1.0,
        quality_weight=1.0,
        reprojection_weight=1.0,
        observability_weight=1.0,
        locability_weight=0.0,
    )(
        query_descs=query_descs,
        query_keypoints_yx=query_keypoints,
        query_mask=torch.ones(1, 4, dtype=torch.bool),
        rendered_desc=rendered_desc,
        depth_map=depth,
        render_pose_w2c=render_pose,
        gt_pose_w2c=gt_pose,
        K=K,
    )

    assert out["valid_queries"].item() == 0.0
    assert out["match"].item() == 0.0
    assert out["pose"].item() == 0.0


def test_queries_without_nearby_gt_target_projection_do_not_supervise_pnp():
    channels = 4
    height = 4
    width = 4
    query_descs = F.normalize(torch.randn(1, 1, channels), dim=-1)
    query_keypoints = torch.tensor([[[3.0, 3.0]]], dtype=torch.float32)
    rendered_desc = F.normalize(torch.randn(1, channels, height, width), dim=1)
    depth = torch.zeros(1, height, width)
    depth[:, 0, 0] = 1.0
    pose = torch.eye(4).unsqueeze(0)
    K = torch.eye(3)

    out = DifferentiablePnPMatchLoss(
        temperature=0.2,
        pnp_iterations=1,
        pose_weight=1.0,
        match_weight=1.0,
        quality_weight=1.0,
        reprojection_weight=1.0,
        observability_weight=1.0,
        locability_weight=0.0,
        target_sigma_px=0.25,
    )(
        query_descs=query_descs,
        query_keypoints_yx=query_keypoints,
        query_mask=torch.ones(1, 1, dtype=torch.bool),
        rendered_desc=rendered_desc,
        depth_map=depth,
        render_pose_w2c=pose,
        gt_pose_w2c=pose,
        K=K,
    )

    assert out["valid_queries"].item() == 0.0
    assert out["match"].item() == 0.0
    assert out["pose"].item() == 0.0


def test_gt_depth_alpha_visibility_masks_pnp_targets():
    channels = 4
    height = 4
    width = 4
    query_descs = F.normalize(torch.randn(1, 1, channels), dim=-1)
    query_keypoints = torch.tensor([[[1.0, 1.0]]], dtype=torch.float32)
    rendered_desc = F.normalize(torch.randn(1, channels, height, width), dim=1)
    depth = torch.ones(1, height, width)
    pose = torch.eye(4).unsqueeze(0)
    K = torch.eye(3)
    gt_depth = torch.full((1, height, width), 5.0)
    gt_alpha = torch.ones(1, height, width)

    out = DifferentiablePnPMatchLoss(
        temperature=0.2,
        pnp_iterations=1,
        pose_weight=1.0,
        match_weight=1.0,
        quality_weight=1.0,
        reprojection_weight=1.0,
        observability_weight=1.0,
        locability_weight=0.0,
        target_sigma_px=0.25,
        occlusion_depth_tolerance=0.05,
        occlusion_depth_rel_tolerance=0.0,
    )(
        query_descs=query_descs,
        query_keypoints_yx=query_keypoints,
        query_mask=torch.ones(1, 1, dtype=torch.bool),
        rendered_desc=rendered_desc,
        depth_map=depth,
        render_pose_w2c=pose,
        gt_pose_w2c=pose,
        K=K,
        gt_depth_map=gt_depth,
        gt_alpha_map=gt_alpha,
    )

    assert out["valid_queries"].item() == 0.0
    assert out["match"].item() == 0.0


def test_topk_pnp_path_keeps_loss_differentiable():
    query_descs = F.normalize(torch.randn(1, 4, 8), dim=-1)
    query_keypoints = torch.tensor(
        [[[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]]],
        dtype=torch.float32,
    )
    rendered_desc = F.normalize(torch.randn(1, 8, 4, 4), dim=1).requires_grad_(True)
    depth = torch.full((1, 4, 4), 4.0, requires_grad=True)
    pose = torch.eye(4).unsqueeze(0)
    K = torch.tensor(
        [[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    out = DifferentiablePnPMatchLoss(temperature=0.2, pnp_iterations=1, topk_pnp=2)(
        query_descs=query_descs,
        query_keypoints_yx=query_keypoints,
        query_mask=torch.ones(1, 4, dtype=torch.bool),
        rendered_desc=rendered_desc,
        depth_map=depth,
        render_pose_w2c=pose,
        gt_pose_w2c=pose,
        K=K,
    )

    out["total"].backward()
    assert torch.isfinite(out["total"])
    assert rendered_desc.grad is not None
    assert torch.isfinite(rendered_desc.grad).all()


def test_topk_pnp_candidates_respect_gt_visibility_mask():
    channels = 16
    height = 4
    width = 4
    rendered_desc = torch.eye(channels, dtype=torch.float32).T.reshape(1, channels, height, width)
    # Make the query descriptor prefer the bottom-right pixel, which is not
    # GT-visible.  PnP should still be forced to use the visible top-left target.
    query_descs = rendered_desc.flatten(2).transpose(1, 2)[:, [15], :]
    query_keypoints = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
    depth = torch.ones(1, height, width)
    pose = torch.eye(4).unsqueeze(0)
    K = torch.eye(3)
    gt_alpha = torch.zeros(1, height, width)
    gt_alpha[:, 0, 0] = 1.0

    out = DifferentiablePnPMatchLoss(
        temperature=0.05,
        pnp_iterations=1,
        topk_pnp=1,
        pose_weight=0.0,
        match_weight=0.0,
        quality_weight=0.0,
        reprojection_weight=1.0,
        observability_weight=0.0,
        locability_weight=0.0,
        target_sigma_px=0.25,
        gt_alpha_threshold=0.5,
    )(
        query_descs=query_descs,
        query_keypoints_yx=query_keypoints,
        query_mask=torch.ones(1, 1, dtype=torch.bool),
        rendered_desc=rendered_desc,
        depth_map=depth,
        render_pose_w2c=pose,
        gt_pose_w2c=pose,
        K=K,
        gt_alpha_map=gt_alpha,
    )

    assert out["valid_queries"].item() == 1.0
    assert out["reprojection"].item() < 1e-4


def test_topk_pnp_does_not_reweight_invisible_fill_candidates():
    channels = 16
    height = 4
    width = 4
    rendered_desc = torch.eye(channels, dtype=torch.float32).T.reshape(1, channels, height, width)
    # Only the top-left rendered point is GT-visible for this query, but the
    # descriptor strongly prefers the bottom-right point.  If top-k has to pad
    # with masked candidates, those candidates must keep zero probability.
    query_descs = rendered_desc.flatten(2).transpose(1, 2)[:, [15], :]
    query_keypoints = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
    depth = torch.ones(1, height, width)
    pose = torch.eye(4).unsqueeze(0)
    K = torch.eye(3)
    gt_alpha = torch.zeros(1, height, width)
    gt_alpha[:, 0, 0] = 1.0

    out = DifferentiablePnPMatchLoss(
        temperature=0.05,
        pnp_iterations=1,
        topk_pnp=2,
        pose_weight=0.0,
        match_weight=0.0,
        quality_weight=0.0,
        reprojection_weight=1.0,
        observability_weight=0.0,
        locability_weight=0.0,
        target_sigma_px=0.25,
        gt_alpha_threshold=0.5,
    )(
        query_descs=query_descs,
        query_keypoints_yx=query_keypoints,
        query_mask=torch.ones(1, 1, dtype=torch.bool),
        rendered_desc=rendered_desc,
        depth_map=depth,
        render_pose_w2c=pose,
        gt_pose_w2c=pose,
        K=K,
        gt_alpha_map=gt_alpha,
    )

    assert out["valid_queries"].item() == 1.0
    assert out["reprojection"].item() < 1e-4


def test_differentiable_pnp_loss_handles_amp_half_inputs_on_cuda():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")
    query_descs = F.normalize(torch.randn(1, 4, 8, device=device), dim=-1).half()
    query_keypoints = torch.tensor(
        [[[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]]],
        dtype=torch.float16,
        device=device,
    )
    query_mask = torch.ones(1, 4, dtype=torch.bool, device=device)
    rendered_desc = F.normalize(torch.randn(1, 8, 4, 4, device=device), dim=1).half().requires_grad_(True)
    depth = torch.full((1, 4, 4), 4.0, device=device, dtype=torch.float16, requires_grad=True)
    pose = torch.eye(4, device=device).unsqueeze(0).half()
    K = torch.tensor(
        [[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]],
        dtype=torch.float16,
        device=device,
    )

    out = DifferentiablePnPMatchLoss(temperature=0.2, pnp_iterations=1)(
        query_descs=query_descs,
        query_keypoints_yx=query_keypoints,
        query_mask=query_mask,
        rendered_desc=rendered_desc,
        depth_map=depth,
        render_pose_w2c=pose,
        gt_pose_w2c=pose,
        K=K,
    )

    assert torch.isfinite(out["total"])
