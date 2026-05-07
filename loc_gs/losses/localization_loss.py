from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_pixel_grid(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([y.reshape(-1), x.reshape(-1)], dim=-1)


def _rigid_inverse(pose_w2c: torch.Tensor) -> torch.Tensor:
    rot = pose_w2c[:, :3, :3]
    trans = pose_w2c[:, :3, 3:4]
    rot_inv = rot.transpose(1, 2)
    trans_inv = -rot_inv @ trans
    inv = torch.zeros_like(pose_w2c)
    inv[:, :3, :3] = rot_inv
    inv[:, :3, 3:4] = trans_inv
    inv[:, 3, 3] = 1.0
    return inv


def unproject_dense_depth_to_world(
    depth_map: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unproject every rendered depth pixel to world coordinates.

    Returns:
        world_points: [B, H*W, 3]
        pixel_grid_yx: [H*W, 2]
        valid_depth: [B, H*W]
    """
    B, H, W = depth_map.shape
    device = depth_map.device
    dtype = depth_map.dtype
    grid_yx = _make_pixel_grid(H, W, device, dtype)
    y = grid_yx[:, 0].view(1, -1).expand(B, -1)
    x = grid_yx[:, 1].view(1, -1).expand(B, -1)
    z = depth_map.reshape(B, -1)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (x - cx) / fx.clamp_min(1e-8) * z
    y_cam = (y - cy) / fy.clamp_min(1e-8) * z

    ones = torch.ones_like(z)
    pts_cam_h = torch.stack([x_cam, y_cam, z, ones], dim=-1)
    cam_to_world = _rigid_inverse(pose_w2c)
    pts_world = torch.bmm(cam_to_world, pts_cam_h.transpose(1, 2)).transpose(1, 2)
    valid = torch.isfinite(z) & (z > 0.0)
    return pts_world[:, :, :3], grid_yx, valid


def project_world_to_image_yx(
    world_points: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project world points into a camera.

    Args:
        world_points: [B, P, 3]
        pose_w2c: [B, 4, 4]
        K: [3, 3]

    Returns:
        projected_yx: [B, P, 2]
        valid_z: [B, P]
    """
    B, P, _ = world_points.shape
    ones = torch.ones(B, P, 1, device=world_points.device, dtype=world_points.dtype)
    pts_h = torch.cat([world_points, ones], dim=-1)
    pts_cam = torch.bmm(pose_w2c, pts_h.transpose(1, 2)).transpose(1, 2)[:, :, :3]

    z = pts_cam[:, :, 2].clamp_min(1e-8)
    x = K[0, 0] * (pts_cam[:, :, 0] / z) + K[0, 2]
    y = K[1, 1] * (pts_cam[:, :, 1] / z) + K[1, 2]
    projected = torch.stack([y, x], dim=-1)
    return projected, pts_cam[:, :, 2] > 1e-8


def projection_jacobian_observability(
    world_points: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
    min_depth: float,
) -> torch.Tensor:
    """Return ||d pi(TX) / d se(3)|| for each world point.

    The score is high for points whose image projection changes strongly under
    small camera pose updates, which makes them useful for geometric
    localization.  It is differentiable w.r.t. point positions/depth.
    """
    B, P, _ = world_points.shape
    ones = torch.ones(B, P, 1, device=world_points.device, dtype=world_points.dtype)
    pts_h = torch.cat([world_points, ones], dim=-1)
    pts_cam = torch.bmm(pose_w2c, pts_h.transpose(1, 2)).transpose(1, 2)[:, :, :3]

    x = pts_cam[:, :, 0]
    y = pts_cam[:, :, 1]
    z = pts_cam[:, :, 2].clamp_min(min_depth)
    z2 = z.square().clamp_min(1e-8)

    fx, fy = K[0, 0], K[1, 1]
    du_dx = fx / z
    du_dz = -fx * x / z2
    dv_dy = fy / z
    dv_dz = -fy * y / z2

    # Translation columns.
    j_tx_u = du_dx
    j_ty_u = torch.zeros_like(du_dx)
    j_tz_u = du_dz
    j_tx_v = torch.zeros_like(dv_dy)
    j_ty_v = dv_dy
    j_tz_v = dv_dz

    # Rotation columns for a small camera-frame se(3) update.
    j_rx_u = -du_dz * y
    j_rx_v = dv_dy * z - dv_dz * y
    j_ry_u = -du_dx * z + du_dz * x
    j_ry_v = dv_dz * x
    j_rz_u = du_dx * y
    j_rz_v = -dv_dy * x

    jac_sq = (
        j_tx_u.square()
        + j_ty_u.square()
        + j_tz_u.square()
        + j_tx_v.square()
        + j_ty_v.square()
        + j_tz_v.square()
        + j_rx_u.square()
        + j_rx_v.square()
        + j_ry_u.square()
        + j_ry_v.square()
        + j_rz_u.square()
        + j_rz_v.square()
    )
    return torch.sqrt(jac_sq.clamp_min(1e-12))


def sample_descriptors_at_keypoints(
    descriptor_map: torch.Tensor,
    keypoints_yx: torch.Tensor,
) -> torch.Tensor:
    """Bilinearly sample descriptors at coarse-resolution keypoints."""
    C, H, W = descriptor_map.shape
    Kp = keypoints_yx.shape[0]
    if Kp == 0:
        return descriptor_map.new_zeros((0, C))
    y = keypoints_yx[:, 0]
    x = keypoints_yx[:, 1]
    x_norm = 2.0 * x / max(W - 1, 1) - 1.0
    y_norm = 2.0 * y / max(H - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, Kp, 1, 2)
    sampled = F.grid_sample(
        descriptor_map.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
    return F.normalize(sampled, p=2, dim=1)


@torch.no_grad()
def extract_keypoints_from_superpoint_logits(
    detector_logits: torch.Tensor,
    confidence_threshold: float,
    nms_radius: int,
    max_keypoints: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract SuperPoint keypoints in coarse feature-map coordinates."""
    _, Hc, Wc = detector_logits.shape
    probs = F.softmax(detector_logits.float(), dim=0)
    heatmap = F.pixel_shuffle(probs[:64].unsqueeze(0), 8).squeeze(0).squeeze(0)
    if nms_radius > 0:
        kernel = 2 * nms_radius + 1
        pooled = F.max_pool2d(
            heatmap.view(1, 1, *heatmap.shape),
            kernel_size=kernel,
            stride=1,
            padding=nms_radius,
        ).view_as(heatmap)
        heatmap = heatmap * (heatmap == pooled).float()

    mask = heatmap > confidence_threshold
    if mask.any():
        ys, xs = torch.where(mask)
        scores = heatmap[ys, xs]
    else:
        k = min(max_keypoints, heatmap.numel())
        scores, flat = heatmap.reshape(-1).topk(k)
        ys = flat // heatmap.shape[1]
        xs = flat % heatmap.shape[1]

    order = scores.argsort(descending=True)[:max_keypoints]
    ys = ys[order].float().clamp(0, Hc * 8 - 1)
    xs = xs[order].float().clamp(0, Wc * 8 - 1)
    scores = scores[order]
    return torch.stack([ys / 8.0, xs / 8.0], dim=-1), scores


@torch.no_grad()
def prepare_superpoint_queries(
    descriptor_map: torch.Tensor,
    detector_logits: torch.Tensor,
    max_keypoints: int,
    confidence_threshold: float,
    nms_radius: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create padded query descriptor/keypoint tensors from teacher SP outputs."""
    B, C, _, _ = descriptor_map.shape
    desc_out = descriptor_map.new_zeros((B, max_keypoints, C))
    keypoint_out = descriptor_map.new_zeros((B, max_keypoints, 2))
    mask_out = torch.zeros(B, max_keypoints, device=descriptor_map.device, dtype=torch.bool)

    for b in range(B):
        keypoints, _scores = extract_keypoints_from_superpoint_logits(
            detector_logits[b],
            confidence_threshold=confidence_threshold,
            nms_radius=nms_radius,
            max_keypoints=max_keypoints,
        )
        n = min(keypoints.shape[0], max_keypoints)
        if n == 0:
            continue
        keypoints = keypoints[:n].to(device=descriptor_map.device, dtype=descriptor_map.dtype)
        desc = sample_descriptors_at_keypoints(descriptor_map[b], keypoints)
        desc_out[b, :n] = desc
        keypoint_out[b, :n] = keypoints
        mask_out[b, :n] = True

    return desc_out, keypoint_out, mask_out


class LocalizationGuidedLoss(nn.Module):
    """Differentiable soft matching + reprojection proxy for localization."""

    def __init__(
        self,
        temperature: float = 0.07,
        target_sigma_px: float = 2.0,
        min_depth: float = 0.05,
        max_depth: float = 20.0,
        locability_prior_weight: float = 1.0,
        reproj_charbonnier_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.target_sigma_px = target_sigma_px
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.locability_prior_weight = locability_prior_weight
        self.reproj_charbonnier_eps = reproj_charbonnier_eps

    def forward(
        self,
        query_descs: torch.Tensor,
        query_keypoints_yx: torch.Tensor,
        query_mask: torch.Tensor,
        rendered_desc: torch.Tensor,
        depth_map: torch.Tensor,
        render_pose_w2c: torch.Tensor,
        gt_pose_w2c: torch.Tensor,
        K: torch.Tensor,
        locability_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, C, H, W = rendered_desc.shape
        _, Kq, _ = query_descs.shape
        P = H * W

        rendered_desc = F.normalize(rendered_desc.float(), p=2, dim=1)
        query_descs = F.normalize(query_descs.float(), p=2, dim=-1)
        depth_map = depth_map.float()
        K = K.to(device=rendered_desc.device, dtype=rendered_desc.dtype)
        render_pose_w2c = render_pose_w2c.to(device=rendered_desc.device, dtype=rendered_desc.dtype)
        gt_pose_w2c = gt_pose_w2c.to(device=rendered_desc.device, dtype=rendered_desc.dtype)

        desc_flat = rendered_desc.flatten(2)
        logits = torch.bmm(query_descs, desc_flat)
        if locability_map is not None:
            loc_flat = locability_map.float().flatten(2)
            logits = logits + self.locability_prior_weight * loc_flat.expand(B, Kq, P)

        world_points, _grid_yx, valid_depth = unproject_dense_depth_to_world(
            depth_map,
            render_pose_w2c,
            K,
        )
        valid_depth = valid_depth & (depth_map.reshape(B, P) >= self.min_depth) & (
            depth_map.reshape(B, P) <= self.max_depth
        )
        projected_yx, valid_proj = project_world_to_image_yx(world_points, gt_pose_w2c, K)
        valid_pixels = valid_depth & valid_proj

        reproj_error = torch.linalg.norm(
            projected_yx[:, None, :, :] - query_keypoints_yx.float()[:, :, None, :],
            dim=-1,
        )
        valid_pairs = query_mask[:, :, None] & valid_pixels[:, None, :]

        masked_logits = logits / max(self.temperature, 1e-6)
        masked_logits = masked_logits.masked_fill(~valid_pixels[:, None, :], -1.0e4)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        probs = log_probs.exp()

        target_logits = -0.5 * (reproj_error / max(self.target_sigma_px, 1e-6)) ** 2
        target_logits = target_logits.masked_fill(~valid_pixels[:, None, :], -1.0e4)
        geometry_target = F.softmax(target_logits, dim=-1)

        valid_queries = query_mask & valid_pixels.any(dim=1, keepdim=True).expand(-1, Kq)
        denom = valid_queries.float().sum().clamp_min(1.0)

        match_per_query = -(geometry_target.detach() * log_probs).sum(dim=-1)
        match_loss = (match_per_query * valid_queries.float()).sum() / denom

        robust_error = torch.sqrt(reproj_error.square() + self.reproj_charbonnier_eps ** 2)
        reproj_per_query = (probs * robust_error).sum(dim=-1)
        reproj_loss = (reproj_per_query * valid_queries.float()).sum() / denom

        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = (entropy * valid_queries.float()).sum() / denom

        obs = projection_jacobian_observability(
            world_points,
            gt_pose_w2c,
            K,
            min_depth=self.min_depth,
        )
        obs = obs.masked_fill(~valid_pixels, 0.0)
        obs = obs / obs.amax(dim=1, keepdim=True).clamp_min(1e-6)
        observability = (probs * obs[:, None, :]).sum(dim=-1)
        observability_loss = -(
            observability * valid_queries.float()
        ).sum() / denom

        total = match_loss + reproj_loss
        return {
            "total": total,
            "match": match_loss,
            "reprojection": reproj_loss,
            "observability": observability_loss,
            "entropy": entropy,
            "geometry_target": geometry_target,
            "match_prob": probs,
            "valid_queries": valid_queries.float().sum(),
        }
