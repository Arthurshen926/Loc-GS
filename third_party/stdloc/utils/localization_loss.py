import math

import torch
import torch.nn.functional as F


def _pixel_grid_yx(height, width, device, dtype):
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([y.reshape(-1), x.reshape(-1)], dim=-1)


def _sample_descriptors(descriptor_map, points_yx):
    channels, height, width = descriptor_map.shape
    if points_yx.numel() == 0:
        return descriptor_map.new_zeros((0, channels))

    y = points_yx[:, 0]
    x = points_yx[:, 1]
    x_norm = 2.0 * x / max(width - 1, 1) - 1.0
    y_norm = 2.0 * y / max(height - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        descriptor_map.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
    return F.normalize(sampled, p=2, dim=-1)


def _select_keypoints(score_map, height, width, max_keypoints, min_score, device, dtype):
    score_map = score_map.to(device=device, dtype=dtype)
    if score_map.shape[-2:] != (height, width):
        score_map = F.interpolate(
            score_map.view(1, 1, *score_map.shape[-2:]),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).view(height, width)

    flat_scores = score_map.reshape(-1)
    k = min(int(max_keypoints), flat_scores.numel())
    if k <= 0:
        return score_map.new_zeros((0, 2))

    values, flat_ids = torch.topk(flat_scores, k=k, largest=True)
    keep = values > float(min_score)
    if not keep.any():
        keep = torch.zeros_like(values, dtype=torch.bool)
        keep[0] = True
    flat_ids = flat_ids[keep]
    return torch.stack([flat_ids // width, flat_ids % width], dim=-1).to(
        device=device, dtype=dtype
    )


def _unproject_depth_to_world(depth_map, pose_w2c, K):
    height, width = depth_map.shape[-2:]
    device = depth_map.device
    dtype = depth_map.dtype
    coords_yx = _pixel_grid_yx(height, width, device, dtype)
    xy1 = torch.stack(
        [
            coords_yx[:, 1] + 0.5,
            coords_yx[:, 0] + 0.5,
            torch.ones_like(coords_yx[:, 0]),
        ],
        dim=-1,
    )
    rays = torch.linalg.inv(K.to(device=device, dtype=dtype)) @ xy1.T
    depth = depth_map.reshape(-1)
    pts_cam = (rays.T * depth[:, None]).to(dtype)
    pts_cam_h = torch.cat([pts_cam, torch.ones_like(depth[:, None])], dim=-1)
    pose_c2w = torch.linalg.inv(pose_w2c.to(device=device, dtype=dtype))
    pts_world = (pose_c2w @ pts_cam_h.T).T[:, :3]
    valid = torch.isfinite(depth) & (depth > 0)
    return pts_world, coords_yx, valid


def _project_world_to_yx(points_world, pose_w2c, K):
    device = points_world.device
    dtype = points_world.dtype
    pts_h = torch.cat(
        [points_world, torch.ones((points_world.shape[0], 1), device=device, dtype=dtype)],
        dim=-1,
    )
    pts_cam = (pose_w2c.to(device=device, dtype=dtype) @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2].clamp_min(1e-8)
    pix = (K.to(device=device, dtype=dtype) @ (pts_cam / z[:, None]).T).T
    projected = torch.stack([pix[:, 1] - 0.5, pix[:, 0] - 0.5], dim=-1)
    valid = torch.isfinite(projected).all(dim=-1) & (pts_cam[:, 2] > 0)
    return projected, valid


def keypoint_reprojection_loss(
    rendered_desc,
    target_desc,
    score_map,
    max_keypoints=64,
    temperature=0.07,
    min_score=0.0,
):
    channels, height, width = rendered_desc.shape
    keypoints = _select_keypoints(
        score_map,
        height,
        width,
        max_keypoints,
        min_score,
        rendered_desc.device,
        rendered_desc.dtype,
    )
    if keypoints.numel() == 0:
        zero = rendered_desc.sum() * 0.0
        return zero, {"keypoints": zero.detach(), "entropy": zero.detach()}

    query_desc = _sample_descriptors(target_desc, keypoints)
    rendered_flat = F.normalize(rendered_desc, p=2, dim=0).flatten(1)
    logits = query_desc @ rendered_flat
    probs = F.softmax(logits / max(float(temperature), 1e-6), dim=-1)

    coords = _pixel_grid_yx(height, width, rendered_desc.device, rendered_desc.dtype)
    expected = probs @ coords
    reproj = torch.linalg.norm(expected - keypoints, dim=-1)
    loss = reproj.mean() / max(math.sqrt(height * height + width * width), 1.0)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
    return loss, {
        "keypoints": torch.as_tensor(keypoints.shape[0], device=rendered_desc.device),
        "entropy": entropy.detach(),
    }


def locability_weighted_feature_loss(
    rendered_desc,
    target_desc,
    locability_map,
    min_weight=0.1,
    gamma=1.0,
    top_ratio=0.0,
    mask=None,
):
    """Prioritize feature reconstruction on localization-relevant pixels.

    The weights are normalized to mean one over the valid region, so the term
    changes where reconstruction capacity is spent rather than trivially
    changing the loss scale.
    """
    channels, height, width = rendered_desc.shape
    device = rendered_desc.device
    dtype = rendered_desc.dtype
    locability = locability_map.to(device=device, dtype=dtype).squeeze().detach()
    if locability.shape[-2:] != (height, width):
        locability = F.interpolate(
            locability.view(1, 1, *locability.shape[-2:]),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).view(height, width)

    if mask is not None:
        valid = mask.to(device=device, dtype=dtype).squeeze()
        if valid.shape[-2:] != (height, width):
            valid = F.interpolate(
                valid.view(1, 1, *valid.shape[-2:]),
                size=(height, width),
                mode="nearest",
            ).view(height, width)
        valid = (valid > 0.5).to(dtype)
    else:
        valid = torch.ones((height, width), device=device, dtype=dtype)

    locability_score = locability.clamp(0.0, 1.0).pow(float(gamma))
    selected = locability > 0.5
    if top_ratio and float(top_ratio) > 0.0:
        valid_bool = valid > 0
        selected = torch.zeros((height, width), device=device, dtype=torch.bool)
        valid_ids = valid_bool.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if valid_ids.numel() > 0:
            keep = max(
                1,
                min(valid_ids.numel(), math.ceil(valid_ids.numel() * float(top_ratio))),
            )
            top_local = torch.topk(
                locability_score.reshape(-1)[valid_ids],
                k=keep,
            ).indices
            selected.reshape(-1)[valid_ids[top_local]] = True
        locability_score = locability_score * selected.to(dtype)

    weights = float(min_weight) + locability_score
    weights = weights * valid
    norm = weights.sum().clamp_min(1e-6) / valid.sum().clamp_min(1.0)
    weights = weights / norm

    per_pixel_error = (rendered_desc - target_desc).abs().mean(dim=0)
    loss = (per_pixel_error * weights * valid).sum() / valid.sum().clamp_min(1.0)

    background = ~selected
    selected_valid = selected & (valid > 0)
    background_valid = background & (valid > 0)
    return loss, {
        "weight_mean": weights[valid > 0].mean().detach(),
        "selected_fraction": (
            selected_valid.to(dtype).sum().div(valid.sum().clamp_min(1.0)).detach()
        ),
        "selected_weight_mean": (
            weights[selected_valid].mean().detach()
            if selected_valid.any()
            else weights.new_tensor(0.0)
        ),
        "background_weight_mean": (
            weights[background_valid].mean().detach()
            if background_valid.any()
            else weights.new_tensor(0.0)
        ),
    }


def pose_guided_reprojection_loss(
    rendered_desc,
    target_desc,
    score_map,
    depth_map,
    render_pose_w2c,
    gt_pose_w2c,
    K,
    max_keypoints=64,
    temperature=0.07,
    target_sigma_px=2.0,
    min_score=0.0,
    min_depth=0.05,
    max_depth=100.0,
    locability_map=None,
    locability_weight=0.0,
):
    channels, height, width = rendered_desc.shape
    device = rendered_desc.device
    dtype = rendered_desc.dtype
    keypoints = _select_keypoints(
        score_map, height, width, max_keypoints, min_score, device, dtype
    )
    if keypoints.numel() == 0:
        zero = rendered_desc.sum() * 0.0
        return zero, {"keypoints": zero.detach(), "entropy": zero.detach()}

    depth_map = depth_map.to(device=device, dtype=dtype).squeeze()
    if depth_map.shape[-2:] != (height, width):
        depth_map = F.interpolate(
            depth_map.view(1, 1, *depth_map.shape[-2:]),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).view(height, width)

    query_desc = _sample_descriptors(target_desc, keypoints)
    rendered_flat = F.normalize(rendered_desc, p=2, dim=0).flatten(1)
    logits = query_desc @ rendered_flat

    if locability_map is not None and locability_weight != 0.0:
        prior = locability_map.to(device=device, dtype=dtype)
        if prior.shape[-2:] != (height, width):
            prior = F.interpolate(
                prior.view(1, 1, *prior.shape[-2:]),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).view(height, width)
        logits = logits + float(locability_weight) * prior.reshape(1, -1)

    world_points, _coords_yx, valid_depth = _unproject_depth_to_world(
        depth_map, render_pose_w2c, K
    )
    projected_yx, valid_proj = _project_world_to_yx(world_points, gt_pose_w2c, K)
    valid_depth = valid_depth & (depth_map.reshape(-1) >= float(min_depth))
    valid_depth = valid_depth & (depth_map.reshape(-1) <= float(max_depth))
    valid_pixels = valid_depth & valid_proj

    if not valid_pixels.any():
        zero = rendered_desc.sum() * 0.0
        return zero, {"keypoints": zero.detach(), "entropy": zero.detach()}

    reproj_error = torch.linalg.norm(
        projected_yx[None, :, :] - keypoints[:, None, :],
        dim=-1,
    )
    masked_logits = logits / max(float(temperature), 1e-6)
    masked_logits = masked_logits.masked_fill(~valid_pixels[None, :], -1.0e4)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    probs = log_probs.exp()

    target_logits = -0.5 * (reproj_error / max(float(target_sigma_px), 1e-6)) ** 2
    target_logits = target_logits.masked_fill(~valid_pixels[None, :], -1.0e4)
    geometry_target = F.softmax(target_logits, dim=-1)
    match_loss = -(geometry_target.detach() * log_probs).sum(dim=-1).mean()

    reproj = (probs * reproj_error).sum(dim=-1)
    diag = max(math.sqrt(height * height + width * width), 1.0)
    reproj_loss = reproj.mean() / diag
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return match_loss + reproj_loss, {
        "keypoints": torch.as_tensor(keypoints.shape[0], device=device),
        "match": match_loss.detach(),
        "reprojection": reproj_loss.detach(),
        "entropy": entropy.detach(),
    }
