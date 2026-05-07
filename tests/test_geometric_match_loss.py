import torch
import torch.nn.functional as F

from loc_gs.losses.geometric_match import geometric_keypoint_match_loss


def test_geometric_keypoint_match_prefers_descriptor_at_keypoint_pixel():
    channels = 16
    rendered = torch.eye(channels, dtype=torch.float32).T.reshape(1, channels, 4, 4)
    query_keypoints = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
    query_mask = torch.ones(1, 1, dtype=torch.bool)
    flat = rendered.flatten(2).transpose(1, 2)
    good_query = flat[:, [6], :]
    bad_query = flat[:, [4], :]

    good = geometric_keypoint_match_loss(
        query_descs=good_query,
        query_keypoints_yx=query_keypoints,
        query_mask=query_mask,
        rendered_desc=rendered,
        temperature=0.05,
        target_sigma_px=0.25,
    )
    bad = geometric_keypoint_match_loss(
        query_descs=bad_query,
        query_keypoints_yx=query_keypoints,
        query_mask=query_mask,
        rendered_desc=rendered,
        temperature=0.05,
        target_sigma_px=0.25,
    )

    assert good["match"] < bad["match"]
    assert good["top1_1px"] == 1.0
    assert bad["top1_1px"] == 0.0


def test_geometric_keypoint_match_backpropagates_to_desc_and_locability():
    torch.manual_seed(11)
    rendered = F.normalize(torch.randn(1, 8, 4, 4), dim=1).requires_grad_(True)
    query = F.normalize(torch.randn(1, 3, 8), dim=-1)
    query_keypoints = torch.tensor([[[1.0, 1.0], [1.0, 2.0], [2.0, 1.0]]])
    query_mask = torch.ones(1, 3, dtype=torch.bool)
    locability = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)

    out = geometric_keypoint_match_loss(
        query_descs=query,
        query_keypoints_yx=query_keypoints,
        query_mask=query_mask,
        rendered_desc=rendered,
        locability_map=locability,
        locability_weight=0.1,
    )
    out["total"].backward()

    assert rendered.grad is not None
    assert torch.isfinite(rendered.grad).all()
    assert locability.grad is not None
    assert torch.isfinite(locability.grad).all()


def test_geometric_keypoint_match_ignores_queries_without_valid_target():
    rendered = F.normalize(torch.randn(1, 4, 4, 4), dim=1)
    query = F.normalize(torch.randn(1, 2, 4), dim=-1)
    query_keypoints = torch.tensor([[[1.0, 1.0], [3.0, 3.0]]])
    query_mask = torch.ones(1, 2, dtype=torch.bool)
    valid_pixels = torch.zeros(1, 4, 4, dtype=torch.bool)
    valid_pixels[:, 0:2, 0:2] = True

    out = geometric_keypoint_match_loss(
        query_descs=query,
        query_keypoints_yx=query_keypoints,
        query_mask=query_mask,
        rendered_desc=rendered,
        valid_pixels=valid_pixels,
        target_sigma_px=0.25,
    )

    assert out["valid_queries"] == 1.0
