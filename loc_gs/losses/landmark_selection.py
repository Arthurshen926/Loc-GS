from __future__ import annotations

import torch
import torch.nn.functional as F

from loc_gs.losses.localization_loss import project_world_to_image_yx


def locability_budget_loss(locability: torch.Tensor, target_count: int) -> torch.Tensor:
    loc = locability.reshape(-1).float().clamp(0.0, 1.0)
    if loc.numel() == 0:
        return loc.sum() * 0.0
    target_ratio = min(max(float(target_count) / float(loc.numel()), 0.0), 1.0)
    return (loc.mean() - target_ratio) ** 2


def descriptor_ambiguity_loss(
    descriptors: torch.Tensor,
    locability: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    if descriptors.shape[0] < 2:
        return descriptors.sum() * 0.0
    desc = F.normalize(descriptors.float(), p=2, dim=-1)
    loc = locability.reshape(-1).float().clamp(0.0, 1.0)
    sim = desc @ desc.T
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    pair_weight = loc[:, None] * loc[None, :]
    penalty = F.relu(sim - float(margin)).square()
    penalty = penalty.masked_fill(eye, 0.0)
    pair_weight = pair_weight.masked_fill(eye, 0.0)
    return (penalty * pair_weight).sum() / pair_weight.sum().clamp_min(1.0)


def key_gaussian_isotropy_loss(
    scales: torch.Tensor,
    locability: torch.Tensor,
) -> torch.Tensor:
    if scales.numel() == 0:
        return scales.sum() * 0.0
    s = scales.float().abs().clamp_min(1e-6)
    anisotropy = (s.max(dim=-1).values / s.min(dim=-1).values - 1.0).square()
    weights = locability.reshape(-1).float().clamp(0.0, 1.0)
    return (anisotropy * weights).sum() / weights.sum().clamp_min(1.0)


def _camera_centers_from_w2c(poses_w2c: torch.Tensor) -> torch.Tensor:
    rot = poses_w2c[:, :3, :3]
    trans = poses_w2c[:, :3, 3]
    return -(rot.transpose(1, 2) @ trans.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def splatloc_saliency_prior(
    world_points: torch.Tensor,
    poses_w2c: torch.Tensor,
    K: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """SplatLoc-inspired visibility/angular-span prior normalized to [0, 1]."""
    if world_points.numel() == 0:
        return world_points.new_empty((0,))
    pts = world_points.float()
    poses = poses_w2c.to(device=pts.device, dtype=pts.dtype)
    K = K.to(device=pts.device, dtype=pts.dtype)
    centers = _camera_centers_from_w2c(poses)
    visibility = pts.new_zeros(pts.shape[0])
    direction_sum = pts.new_zeros(pts.shape[0], 3)
    direction_sq = pts.new_zeros(pts.shape[0])
    for pose, center in zip(poses, centers):
        proj, valid_z = project_world_to_image_yx(pts.unsqueeze(0), pose.unsqueeze(0), K)
        proj = proj[0]
        visible = (
            valid_z[0]
            & (proj[:, 0] >= 0)
            & (proj[:, 0] <= height - 1)
            & (proj[:, 1] >= 0)
            & (proj[:, 1] <= width - 1)
        )
        visibility += visible.float()
        dirs = F.normalize(center.view(1, 3) - pts, p=2, dim=-1)
        direction_sum += dirs * visible.float().view(-1, 1)
        direction_sq += visible.float()
    mean_dir = direction_sum / direction_sq.clamp_min(1.0).view(-1, 1)
    angular_span = (1.0 - mean_dir.norm(dim=-1)).clamp_min(0.0)
    score = visibility + angular_span
    score = torch.where(visibility > 0, score, torch.zeros_like(score))
    if score.max() <= score.min():
        return (score > 0).float()
    return (score - score.min()) / (score.max() - score.min()).clamp_min(1e-6)
