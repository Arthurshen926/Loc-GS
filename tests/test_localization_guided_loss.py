import torch
import torch.nn.functional as F

from loc_gs.losses.localization_loss import LocalizationGuidedLoss


def test_localization_loss_builds_geometry_target_and_backprops():
    loss_fn = LocalizationGuidedLoss(
        temperature=0.2,
        target_sigma_px=0.5,
        min_depth=0.05,
        max_depth=10.0,
    )

    rendered_raw = torch.tensor(
        [[[[2.0, 0.1], [0.1, 0.1]], [[0.1, 2.0], [0.1, 0.1]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    rendered_desc = F.normalize(rendered_raw, p=2, dim=1)
    query_descs = F.normalize(
        torch.tensor([[[1.0, 0.0]]], dtype=torch.float32),
        p=2,
        dim=-1,
    )
    query_keypoints_yx = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
    query_mask = torch.tensor([[True]])
    depth = torch.ones(1, 2, 2, dtype=torch.float32, requires_grad=True)
    locability = torch.zeros(1, 1, 2, 2, dtype=torch.float32, requires_grad=True)

    render_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    gt_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    gt_pose[:, 0, 3] = 0.1
    K = torch.eye(3, dtype=torch.float32)

    out = loss_fn(
        query_descs=query_descs,
        query_keypoints_yx=query_keypoints_yx,
        query_mask=query_mask,
        rendered_desc=rendered_desc,
        depth_map=depth,
        render_pose_w2c=render_pose,
        gt_pose_w2c=gt_pose,
        K=K,
        locability_map=locability,
    )

    assert out["geometry_target"].shape == (1, 1, 4)
    assert out["geometry_target"][0, 0].argmax().item() == 0
    assert torch.isfinite(out["total"])

    out["total"].backward()

    assert rendered_raw.grad is not None
    assert rendered_raw.grad.abs().sum() > 0
    assert depth.grad is not None
    assert depth.grad.abs().sum() > 0
    assert locability.grad is not None
    assert locability.grad.abs().sum() > 0
