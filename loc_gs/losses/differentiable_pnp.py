from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from loc_gs.losses.localization_loss import (
    projection_jacobian_observability,
    project_world_to_image_yx,
    unproject_dense_depth_to_world,
)


def _skew(vec: torch.Tensor) -> torch.Tensor:
    x, y, z = vec.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=-1,
    ).reshape(*vec.shape[:-1], 3, 3)


def se3_exp(delta: torch.Tensor) -> torch.Tensor:
    """Small SE(3) exponential for left-multiplicative pose updates."""
    trans = delta[..., :3]
    rot = delta[..., 3:]
    theta = torch.linalg.norm(rot, dim=-1, keepdim=True).clamp_min(1e-8)
    K = _skew(rot / theta)
    eye = torch.eye(3, device=delta.device, dtype=delta.dtype).expand(delta.shape[0], -1, -1)
    theta_m = theta[..., None]
    sin_t = torch.sin(theta_m)
    cos_t = torch.cos(theta_m)
    R = eye + sin_t * K + (1.0 - cos_t) * (K @ K)
    V = eye + ((1.0 - cos_t) / theta_m) * K + ((theta_m - sin_t) / theta_m) * (K @ K)
    T = torch.eye(4, device=delta.device, dtype=delta.dtype).unsqueeze(0).repeat(delta.shape[0], 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = (V @ trans.unsqueeze(-1)).squeeze(-1)
    return T


def _project_points_xy(
    world_points: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, _ = world_points.shape
    ones = torch.ones(B, N, 1, device=world_points.device, dtype=world_points.dtype)
    pts_h = torch.cat([world_points, ones], dim=-1)
    pts_cam = torch.bmm(pose_w2c, pts_h.transpose(1, 2)).transpose(1, 2)[:, :, :3]
    z = pts_cam[:, :, 2].clamp_min(1e-6)
    x = K[0, 0] * (pts_cam[:, :, 0] / z) + K[0, 2]
    y = K[1, 1] * (pts_cam[:, :, 1] / z) + K[1, 2]
    return torch.stack([x, y], dim=-1), pts_cam, pts_cam[:, :, 2] > 1e-6


def _projection_jacobian_xy(points_cam: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    x = points_cam[:, :, 0]
    y = points_cam[:, :, 1]
    z = points_cam[:, :, 2].clamp_min(1e-6)
    z2 = z.square().clamp_min(1e-8)
    fx, fy = K[0, 0], K[1, 1]

    j = points_cam.new_zeros((*points_cam.shape[:2], 2, 6))
    j[:, :, 0, 0] = fx / z
    j[:, :, 0, 2] = -fx * x / z2
    j[:, :, 1, 1] = fy / z
    j[:, :, 1, 2] = -fy * y / z2

    j[:, :, 0, 3] = -fx * x * y / z2
    j[:, :, 0, 4] = fx + fx * x.square() / z2
    j[:, :, 0, 5] = -fx * y / z
    j[:, :, 1, 3] = -fy - fy * y.square() / z2
    j[:, :, 1, 4] = fy * x * y / z2
    j[:, :, 1, 5] = fy * x / z
    return j


def differentiable_pnp_gauss_newton(
    world_points: torch.Tensor,
    target_keypoints_yx: torch.Tensor,
    init_pose_w2c: torch.Tensor,
    K: torch.Tensor,
    weights: torch.Tensor,
    iterations: int = 4,
    damping: float = 1e-3,
    max_translation_step: float = 0.25,
    max_rotation_step: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    pose = init_pose_w2c
    target_xy = torch.stack([target_keypoints_yx[..., 1], target_keypoints_yx[..., 0]], dim=-1)
    eye6 = torch.eye(6, device=world_points.device, dtype=world_points.dtype).unsqueeze(0)

    residual = world_points.new_zeros((*weights.shape, 2))
    for _ in range(max(int(iterations), 0)):
        projected_xy, points_cam, valid_z = _project_points_xy(world_points, pose, K)
        residual = projected_xy - target_xy
        valid_weights = weights * valid_z.to(weights.dtype)
        jac = _projection_jacobian_xy(points_cam, K)
        sqrt_w = valid_weights.clamp_min(0.0).sqrt()
        jac_w = jac * sqrt_w[:, :, None, None]
        res_w = residual * sqrt_w[:, :, None]
        H = torch.einsum("bnpj,bnpk->bjk", jac_w, jac_w)
        b = torch.einsum("bnpj,bnp->bj", jac_w, res_w)
        H = H + float(damping) * eye6
        delta = torch.linalg.solve(H, -b.unsqueeze(-1)).squeeze(-1)
        trans = delta[:, :3]
        rot = delta[:, 3:]
        trans_scale = (
            float(max_translation_step)
            / trans.norm(dim=-1, keepdim=True).clamp_min(float(max_translation_step))
        )
        rot_scale = (
            float(max_rotation_step)
            / rot.norm(dim=-1, keepdim=True).clamp_min(float(max_rotation_step))
        )
        delta = torch.cat([trans * trans_scale, rot * rot_scale], dim=-1)
        pose = se3_exp(delta) @ pose
    return pose, residual


class DifferentiablePnPMatchLoss(nn.Module):
    """Soft descriptor matching followed by unrolled differentiable PnP.

    Gradients flow through the soft correspondence distribution, the rendered
    depth-derived 3D points, the unrolled pose solve, and the locability prior.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        pnp_iterations: int = 4,
        pnp_damping: float = 1e-3,
        max_translation_step: float = 0.25,
        max_rotation_step: float = 0.25,
        pose_weight: float = 1.0,
        match_weight: float = 0.5,
        quality_weight: float = 0.5,
        reprojection_weight: float = 0.5,
        observability_weight: float = 0.02,
        locability_weight: float = 0.05,
        entropy_weight: float = 0.0,
        locability_prior_weight: float = 0.1,
        target_sigma_px: float = 2.0,
        min_depth: float = 0.05,
        max_depth: float = 100.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.pnp_iterations = pnp_iterations
        self.pnp_damping = pnp_damping
        self.max_translation_step = max_translation_step
        self.max_rotation_step = max_rotation_step
        self.pose_weight = pose_weight
        self.match_weight = match_weight
        self.quality_weight = quality_weight
        self.reprojection_weight = reprojection_weight
        self.observability_weight = observability_weight
        self.locability_weight = locability_weight
        self.entropy_weight = entropy_weight
        self.locability_prior_weight = locability_prior_weight
        self.target_sigma_px = target_sigma_px
        self.min_depth = min_depth
        self.max_depth = max_depth

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
    ) -> dict[str, torch.Tensor]:
        B, C, H, W = rendered_desc.shape
        P = H * W
        dtype = rendered_desc.dtype
        K = K.to(device=rendered_desc.device, dtype=dtype)
        render_pose_w2c = render_pose_w2c.to(device=rendered_desc.device, dtype=dtype)
        gt_pose_w2c = gt_pose_w2c.to(device=rendered_desc.device, dtype=dtype)

        rendered_desc = F.normalize(rendered_desc.float(), p=2, dim=1)
        query_descs = F.normalize(query_descs.float(), p=2, dim=-1)
        depth_map = depth_map.float()

        world_points, _grid_yx, valid_depth = unproject_dense_depth_to_world(
            depth_map,
            render_pose_w2c,
            K,
        )
        depth_flat = depth_map.reshape(B, P)
        valid_pixels = valid_depth & (depth_flat >= self.min_depth) & (depth_flat <= self.max_depth)

        logits = torch.bmm(query_descs, rendered_desc.flatten(2))
        if locability_map is not None:
            loc = locability_map.float()
            if loc.shape[-2:] != (H, W):
                loc = F.interpolate(loc, size=(H, W), mode="bilinear", align_corners=False)
            logits = logits + self.locability_prior_weight * loc.flatten(2).expand(B, query_descs.shape[1], P)
        logits = logits / max(float(self.temperature), 1e-6)
        logits = logits.masked_fill(~valid_pixels[:, None, :], -1e4)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        confidence = probs.max(dim=-1).values
        pnp_weights = query_mask.float() * confidence.detach()
        expected_world = torch.bmm(probs, world_points)
        pose_w2c, pnp_residual = differentiable_pnp_gauss_newton(
            expected_world,
            query_keypoints_yx.float(),
            render_pose_w2c,
            K,
            pnp_weights,
            iterations=self.pnp_iterations,
            damping=self.pnp_damping,
            max_translation_step=self.max_translation_step,
            max_rotation_step=self.max_rotation_step,
        )

        target_pixels_yx, valid_target_proj = project_world_to_image_yx(world_points, gt_pose_w2c, K)
        target_in_frame = (
            (target_pixels_yx[..., 0] >= 0.0)
            & (target_pixels_yx[..., 0] <= float(H - 1))
            & (target_pixels_yx[..., 1] >= 0.0)
            & (target_pixels_yx[..., 1] <= float(W - 1))
        )
        valid_target = valid_target_proj & target_in_frame
        image_diag = float((H * H + W * W) ** 0.5)
        target_err = torch.linalg.norm(
            target_pixels_yx[:, None, :, :] - query_keypoints_yx.float()[:, :, None, :],
            dim=-1,
        )
        target_err = target_err.masked_fill(~valid_target[:, None, :], 1e4)
        target_logits = -0.5 * (target_err / max(float(self.target_sigma_px), 1e-6)).square()
        target_logits = target_logits.masked_fill(~valid_pixels[:, None, :], -1e4)
        geometry_target = F.softmax(target_logits, dim=-1)

        valid_queries = query_mask & valid_pixels.any(dim=1, keepdim=True).expand_as(query_mask)
        denom = valid_queries.float().sum().clamp_min(1.0)
        match_loss = -((geometry_target.detach() * log_probs).sum(dim=-1) * valid_queries.float()).sum() / denom

        robust_target = torch.sqrt(target_err.square() + 1e-6)
        quality_loss = ((probs * robust_target).sum(dim=-1) * valid_queries.float()).sum() / denom
        quality_loss = quality_loss / max(image_diag, 1.0)
        pnp_reproj = (torch.linalg.norm(pnp_residual, dim=-1) * valid_queries.float()).sum() / denom
        pnp_reproj = pnp_reproj / max(image_diag, 1.0)

        pred_c2w = torch.linalg.inv(pose_w2c)
        gt_c2w = torch.linalg.inv(gt_pose_w2c)
        translation_loss = torch.linalg.norm(pred_c2w[:, :3, 3] - gt_c2w[:, :3, 3], dim=-1).mean()
        rel_rot = pose_w2c[:, :3, :3] @ gt_c2w[:, :3, :3]
        cos_angle = ((rel_rot.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
        rotation_loss = (1.0 - cos_angle).mean()
        pose_loss = translation_loss + rotation_loss

        obs = projection_jacobian_observability(world_points, gt_pose_w2c, K, self.min_depth)
        obs = obs.masked_fill(~(valid_pixels & valid_target), 0.0)
        obs = obs / obs.amax(dim=1, keepdim=True).clamp_min(1e-6)
        observability_loss = -((probs * obs[:, None, :]).sum(dim=-1) * valid_queries.float()).sum() / denom

        entropy = -(probs * log_probs).sum(dim=-1)
        entropy_loss = (entropy * valid_queries.float()).sum() / denom

        locability_loss = rendered_desc.new_tensor(0.0)
        if locability_map is not None:
            inlier_support = (
                probs.detach() * torch.exp(-target_err.detach() / max(float(self.target_sigma_px), 1e-6))
            ).sum(dim=1)
            inlier_support = inlier_support / inlier_support.amax(dim=1, keepdim=True).clamp_min(1e-6)
            loc = locability_map.float()
            if loc.shape[-2:] != (H, W):
                loc = F.interpolate(loc, size=(H, W), mode="bilinear", align_corners=False)
            locability_loss = F.binary_cross_entropy(
                loc.flatten(2).squeeze(1).clamp(1e-4, 1.0 - 1e-4),
                inlier_support.clamp(0.0, 1.0),
            )

        total = (
            self.pose_weight * pose_loss
            + self.match_weight * match_loss
            + self.quality_weight * quality_loss
            + self.reprojection_weight * pnp_reproj
            + self.observability_weight * observability_loss
            + self.locability_weight * locability_loss
            + self.entropy_weight * entropy_loss
        )
        return {
            "total": total,
            "pose": pose_loss,
            "translation": translation_loss,
            "rotation": rotation_loss,
            "match": match_loss,
            "quality": quality_loss,
            "reprojection": pnp_reproj,
            "observability": observability_loss,
            "locability": locability_loss,
            "entropy": entropy_loss,
            "pose_w2c": pose_w2c,
            "match_prob": probs,
            "valid_queries": valid_queries.float().sum(),
        }
