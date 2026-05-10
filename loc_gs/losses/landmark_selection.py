from __future__ import annotations

from collections.abc import Mapping

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


def normalize_score01(
    score: torch.Tensor,
    valid: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Robust min-max normalization for landmark score components."""
    if score.numel() == 0:
        return score.float()
    value = score.float()
    finite = torch.isfinite(value)
    if valid is not None:
        finite = finite & valid.to(device=value.device, dtype=torch.bool)
    out = torch.zeros_like(value)
    if not finite.any():
        return out
    selected = value[finite]
    lo = selected.amin()
    hi = selected.amax()
    if (hi - lo) <= float(eps):
        out[finite] = 1.0
        return out
    out[finite] = ((selected - lo) / (hi - lo).clamp_min(float(eps))).clamp(0.0, 1.0)
    return out


def geometric_mean_score(
    components: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    eps: float = 1e-4,
) -> torch.Tensor:
    """Combine normalized score components with a weighted geometric mean."""
    active: list[tuple[torch.Tensor, float]] = []
    for name, value in components.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0.0:
            continue
        active.append((normalize_score01(value).clamp(0.0, 1.0), weight))
    if not active:
        first = next(iter(components.values()))
        return torch.ones_like(first, dtype=torch.float32)
    log_score = torch.zeros_like(active[0][0], dtype=torch.float32)
    total_weight = 0.0
    for value, weight in active:
        log_score = log_score + weight * torch.log(value.clamp_min(float(eps)))
        total_weight += weight
    return torch.exp(log_score / max(total_weight, float(eps))).clamp(0.0, 1.0)


def superpoint_detector_saliency(detector_logits: torch.Tensor) -> torch.Tensor:
    """Coarse-grid SuperPoint saliency using non-dustbin probability mass."""
    if detector_logits.ndim != 3 or detector_logits.shape[0] != 65:
        raise ValueError("detector_logits must have shape [65, H, W]")
    probs = F.softmax(detector_logits.float(), dim=0)
    return (1.0 - probs[64]).clamp(0.0, 1.0)


def depth_consistency_score(
    depth_map: torch.Tensor,
    valid: torch.Tensor | None = None,
    window_size: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Prefer locally stable surface depth, a cheap geometry-consistency proxy."""
    if depth_map.ndim != 2:
        raise ValueError("depth_map must have shape [H, W]")
    depth = depth_map.float()
    valid_mask = torch.isfinite(depth) & (depth > 0.0)
    if valid is not None:
        valid_mask = valid_mask & valid.to(device=depth.device, dtype=torch.bool)
    kernel = max(1, int(window_size))
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    weight = valid_mask.float().view(1, 1, *depth.shape)
    depth_safe = torch.where(valid_mask, depth, torch.zeros_like(depth)).view(1, 1, *depth.shape)
    count = F.avg_pool2d(weight, kernel_size=kernel, stride=1, padding=pad) * float(kernel * kernel)
    mean = (
        F.avg_pool2d(depth_safe, kernel_size=kernel, stride=1, padding=pad)
        * float(kernel * kernel)
        / count.clamp_min(1.0)
    )
    sq = (
        F.avg_pool2d(depth_safe.square(), kernel_size=kernel, stride=1, padding=pad)
        * float(kernel * kernel)
        / count.clamp_min(1.0)
    )
    var = (sq - mean.square()).clamp_min(0.0)
    rel_std = var.sqrt().view_as(depth) / mean.view_as(depth).abs().clamp_min(float(eps))
    score = torch.exp(-rel_std)
    return torch.where(valid_mask, score, torch.zeros_like(score)).clamp(0.0, 1.0)


def descriptor_local_distinctiveness(
    descriptor_map: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize descriptors that look too similar to immediate image neighbors."""
    if descriptor_map.ndim != 3:
        raise ValueError("descriptor_map must have shape [C, H, W]")
    desc = F.normalize(descriptor_map.float(), p=2, dim=0)
    _, height, width = desc.shape
    max_neighbor = torch.full((height, width), -1.0, device=desc.device, dtype=desc.dtype)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = torch.roll(desc, shifts=(dy, dx), dims=(1, 2))
        sim = (desc * shifted).sum(dim=0)
        if dy > 0:
            sim[:dy, :] = -1.0
        elif dy < 0:
            sim[dy:, :] = -1.0
        if dx > 0:
            sim[:, :dx] = -1.0
        elif dx < 0:
            sim[:, dx:] = -1.0
        max_neighbor = torch.maximum(max_neighbor, sim)
    score = (1.0 - max_neighbor.clamp(-1.0, 1.0)) * 0.5
    if valid is not None:
        score = torch.where(valid.to(device=score.device, dtype=torch.bool), score, torch.zeros_like(score))
    return normalize_score01(score)


def gaussian_geometry_score(
    scales: torch.Tensor,
    opacity: torch.Tensor | None = None,
) -> torch.Tensor:
    """SplatLoc-inspired geometry reliability from opacity and scale isotropy."""
    if scales.numel() == 0:
        return scales.new_empty((0,))
    s = scales.float().abs().clamp_min(1e-6)
    anisotropy = s.max(dim=-1).values / s.min(dim=-1).values
    isotropy = (1.0 / anisotropy.clamp_min(1.0)).clamp(0.0, 1.0)
    if opacity is None or opacity.numel() == 0:
        return isotropy
    return (isotropy * normalize_score01(opacity.reshape(-1).float())).clamp(0.0, 1.0)


def spatially_balanced_topk(
    score: torch.Tensor,
    positions: torch.Tensor,
    k: int,
    grid_size: int = 8,
    exclude: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select high-score landmarks while spreading them over a coarse 3D grid."""
    score = score.float().view(-1)
    if score.numel() == 0 or int(k) <= 0:
        return torch.empty(0, dtype=torch.long, device=score.device)
    keep = min(int(k), int(score.numel()))
    valid = torch.isfinite(score)
    pos = positions.to(device=score.device, dtype=torch.float32)
    if pos.ndim != 2 or pos.shape[0] != score.numel() or pos.shape[1] < 3:
        raise ValueError("positions must have shape [N, 3+] and match score length")
    valid = valid & torch.isfinite(pos[:, :3]).all(dim=1)
    if exclude is not None:
        valid = valid & ~exclude.to(device=score.device, dtype=torch.bool).view(-1)
    eligible = torch.where(valid)[0]
    if eligible.numel() <= keep:
        return eligible[torch.argsort(score[eligible], descending=True)]
    grid = int(grid_size)
    if grid <= 1:
        return eligible[torch.topk(score[eligible], k=keep).indices]

    xyz = pos[eligible, :3]
    lo = xyz.amin(dim=0)
    hi = xyz.amax(dim=0)
    norm = (xyz - lo) / (hi - lo).clamp_min(1e-6)
    bins = torch.clamp((norm * float(grid)).long(), min=0, max=grid - 1)
    cell_ids = bins[:, 0] * (grid * grid) + bins[:, 1] * grid + bins[:, 2]
    unique_cells = torch.unique(cell_ids, sorted=False)
    cell_best = []
    for cell in unique_cells:
        local = torch.where(cell_ids == cell)[0]
        if local.numel() == 0:
            continue
        best = local[torch.argmax(score[eligible[local]])]
        cell_best.append(best)
    if not cell_best:
        return eligible[torch.topk(score[eligible], k=keep).indices]

    first_pass_local = torch.stack(cell_best, dim=0)
    first_order = torch.argsort(score[eligible[first_pass_local]], descending=True)
    selected = eligible[first_pass_local[first_order[:keep]]]
    if selected.numel() < keep:
        mask = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
        mask[selected] = True
        remaining = torch.where(valid & ~mask)[0]
        topup = min(keep - int(selected.numel()), int(remaining.numel()))
        if topup > 0:
            selected = torch.cat([selected, remaining[torch.topk(score[remaining], k=topup).indices]], dim=0)
    return selected[:keep]


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
