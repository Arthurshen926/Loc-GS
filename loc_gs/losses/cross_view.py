from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn.functional as F

from loc_gs.losses.localization_loss import (
    project_world_to_image_yx,
    unproject_dense_depth_to_world,
)


def projective_view_overlap(
    world_points: torch.Tensor,
    pose_a_w2c: torch.Tensor,
    pose_b_w2c: torch.Tensor,
    K: torch.Tensor,
    height: int,
    width: int,
) -> float:
    """Fraction of points visible in both views."""
    pts = world_points.unsqueeze(0)
    pose_a = pose_a_w2c.unsqueeze(0)
    pose_b = pose_b_w2c.unsqueeze(0)
    proj_a, valid_a = project_world_to_image_yx(pts, pose_a, K)
    proj_b, valid_b = project_world_to_image_yx(pts, pose_b, K)
    in_a = (
        valid_a[0]
        & (proj_a[0, :, 0] >= 0)
        & (proj_a[0, :, 0] <= height - 1)
        & (proj_a[0, :, 1] >= 0)
        & (proj_a[0, :, 1] <= width - 1)
    )
    in_b = (
        valid_b[0]
        & (proj_b[0, :, 0] >= 0)
        & (proj_b[0, :, 0] <= height - 1)
        & (proj_b[0, :, 1] >= 0)
        & (proj_b[0, :, 1] <= width - 1)
    )
    denom = max(int((in_a | in_b).sum().item()), 1)
    return float((in_a & in_b).sum().item() / denom)


@dataclass
class DescriptorMemoryBank:
    num_embeddings: int
    dim: int
    momentum: float = 0.99
    device: torch.device | str = "cpu"

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        self.embeddings = torch.zeros(self.num_embeddings, self.dim, device=device)
        self.valid_mask = torch.zeros(self.num_embeddings, dtype=torch.bool, device=device)

    def to(self, device: torch.device | str) -> "DescriptorMemoryBank":
        self.embeddings = self.embeddings.to(device)
        self.valid_mask = self.valid_mask.to(device)
        return self

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[ids.to(device=self.embeddings.device, dtype=torch.long)]

    @torch.no_grad()
    def update(self, ids: torch.Tensor, descriptors: torch.Tensor) -> None:
        ids = ids.to(device=self.embeddings.device, dtype=torch.long)
        desc = F.normalize(descriptors.detach().to(self.embeddings.device).float(), p=2, dim=-1)
        valid_old = self.valid_mask[ids]
        old = self.embeddings[ids]
        mixed = torch.where(
            valid_old[:, None],
            float(self.momentum) * old + (1.0 - float(self.momentum)) * desc,
            desc,
        )
        self.embeddings[ids] = F.normalize(mixed, p=2, dim=-1)
        self.valid_mask[ids] = True


def memory_bank_contrastive_loss(
    descriptors: torch.Tensor,
    ids: torch.Tensor,
    bank: DescriptorMemoryBank,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Contrast current descriptors against previously stored prototypes."""
    ids = ids.to(device=bank.embeddings.device, dtype=torch.long)
    desc = F.normalize(descriptors.float().to(bank.embeddings.device), p=2, dim=-1)
    candidate_ids = torch.where(bank.valid_mask)[0]
    if candidate_ids.numel() == 0 or desc.numel() == 0:
        return desc.sum() * 0.0
    target_pos = torch.searchsorted(candidate_ids, ids)
    in_range = target_pos < candidate_ids.numel()
    target_pos = target_pos.clamp_max(max(candidate_ids.numel() - 1, 0))
    is_valid_target = in_range & (candidate_ids[target_pos] == ids)
    if not bool(is_valid_target.any()):
        return desc.sum() * 0.0
    desc = desc[is_valid_target]
    target_pos = target_pos[is_valid_target]
    logits = desc @ bank.embeddings[candidate_ids].T / max(float(temperature), 1e-6)
    return F.cross_entropy(logits, target_pos)


def _sample_descriptors_at_yx(desc_map: torch.Tensor, points_yx: torch.Tensor) -> torch.Tensor:
    B, C, H, W = desc_map.shape
    y = points_yx[..., 0]
    x = points_yx[..., 1]
    grid = torch.stack(
        [
            2.0 * x / max(W - 1, 1) - 1.0,
            2.0 * y / max(H - 1, 1) - 1.0,
        ],
        dim=-1,
    ).view(B, -1, 1, 2)
    sampled = F.grid_sample(
        desc_map.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return F.normalize(sampled.squeeze(-1).transpose(1, 2), p=2, dim=-1)


def _sample_map_at_yx(value_map: torch.Tensor, points_yx: torch.Tensor) -> torch.Tensor:
    if value_map.dim() != 3:
        raise ValueError("value_map must have shape [B, H, W]")
    B, H, W = value_map.shape
    y = points_yx[..., 0]
    x = points_yx[..., 1]
    grid = torch.stack(
        [
            2.0 * x / max(W - 1, 1) - 1.0,
            2.0 * y / max(H - 1, 1) - 1.0,
        ],
        dim=-1,
    ).view(B, -1, 1, 2)
    sampled = F.grid_sample(
        value_map.unsqueeze(1).float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[:, 0, :, 0]


def _near_target_mask(
    target_ids: torch.Tensor,
    num_classes: int,
    batch: int,
    height: int,
    width: int,
    radius: float,
) -> torch.Tensor:
    mask = F.one_hot(target_ids, num_classes=num_classes).bool()
    if radius <= 0.0:
        return mask
    bank_ids = torch.arange(num_classes, device=target_ids.device)
    bank_b = bank_ids // (height * width)
    bank_rem = bank_ids % (height * width)
    bank_y = bank_rem // width
    bank_x = bank_rem % width
    target_b = target_ids // (height * width)
    target_rem = target_ids % (height * width)
    target_y = target_rem // width
    target_x = target_rem % width
    radius_i = max(0, int(math.ceil(float(radius))))
    near = (
        (bank_b.unsqueeze(0) == target_b.unsqueeze(1))
        & ((bank_y.unsqueeze(0) - target_y.unsqueeze(1)).abs() <= radius_i)
        & ((bank_x.unsqueeze(0) - target_x.unsqueeze(1)).abs() <= radius_i)
    )
    return mask | near


def cross_view_projective_contrastive_loss(
    desc_a: torch.Tensor,
    depth_a: torch.Tensor,
    pose_a_w2c: torch.Tensor,
    desc_b: torch.Tensor,
    pose_b_w2c: torch.Tensor,
    K: torch.Tensor,
    positive_desc_b: Optional[torch.Tensor] = None,
    depth_b: Optional[torch.Tensor] = None,
    alpha_b: Optional[torch.Tensor] = None,
    valid_a: Optional[torch.Tensor] = None,
    locability_a: Optional[torch.Tensor] = None,
    max_samples: int = 512,
    temperature: float = 0.07,
    hard_negative_weight: float = 0.0,
    hard_negative_margin: float = 0.1,
    hard_negative_exclusion_radius: float = 1.0,
    depth_tolerance: float = 0.2,
    depth_rel_tolerance: float = 0.05,
    alpha_threshold: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Cross-view InfoNCE using geometry-projected positives in view B."""
    if desc_a.requires_grad:
        desc_a.retain_grad()
    if desc_b.requires_grad:
        desc_b.retain_grad()
    if positive_desc_b is not None and positive_desc_b.requires_grad:
        positive_desc_b.retain_grad()
    B, C, H, W = desc_a.shape
    desc_a_n = F.normalize(desc_a.float(), p=2, dim=1)
    desc_b_n = F.normalize(desc_b.float(), p=2, dim=1)
    positive_desc_b_n = (
        F.normalize(positive_desc_b.float(), p=2, dim=1)
        if positive_desc_b is not None
        else None
    )
    K = K.to(device=desc_a.device, dtype=desc_a.dtype)
    world_points, _grid_yx, valid_depth = unproject_dense_depth_to_world(depth_a.float(), pose_a_w2c.float(), K.float())
    proj_b, valid_z = project_world_to_image_yx(world_points, pose_b_w2c.float(), K.float())
    pts_h = torch.cat(
        [world_points, torch.ones_like(world_points[..., :1])],
        dim=-1,
    )
    depth_in_b = (pose_b_w2c.float() @ pts_h.transpose(1, 2)).transpose(1, 2)[..., 2]
    in_frame = (
        valid_depth
        & valid_z
        & (proj_b[..., 0] >= 0.0)
        & (proj_b[..., 0] <= H - 1)
        & (proj_b[..., 1] >= 0.0)
        & (proj_b[..., 1] <= W - 1)
    )
    if depth_b is not None:
        sampled_depth_b = _sample_map_at_yx(depth_b.float(), proj_b)
        tol = max(float(depth_tolerance), 0.0) + max(float(depth_rel_tolerance), 0.0) * depth_in_b.abs()
        depth_consistent = (
            torch.isfinite(sampled_depth_b)
            & (sampled_depth_b > 0.05)
            & ((sampled_depth_b - depth_in_b).abs() <= tol)
        )
        in_frame = in_frame & depth_consistent
    if alpha_b is not None:
        sampled_alpha_b = _sample_map_at_yx(alpha_b.float(), proj_b)
        in_frame = in_frame & (sampled_alpha_b > float(alpha_threshold))
    if valid_a is not None:
        in_frame = in_frame & valid_a.reshape(B, -1).to(device=in_frame.device, dtype=torch.bool)

    anchors = []
    targets = []
    positive_targets = []
    weights = []
    desc_a_flat = desc_a_n.flatten(2).transpose(1, 2)
    for b in range(B):
        ids = torch.where(in_frame[b])[0]
        if ids.numel() == 0:
            continue
        if locability_a is not None:
            loc = locability_a[b].reshape(-1)[ids].float()
            order = loc.argsort(descending=True)
            ids = ids[order[: max(1, min(int(max_samples), ids.numel()))]]
            weights.append(loc[order[: ids.numel()]])
        else:
            ids = ids[: max(1, min(int(max_samples), ids.numel()))]
            weights.append(torch.ones(ids.numel(), device=desc_a.device))
        anchors.append(desc_a_flat[b, ids])
        py = proj_b[b, ids, 0].round().long().clamp(0, H - 1)
        px = proj_b[b, ids, 1].round().long().clamp(0, W - 1)
        targets.append(py * W + px + b * H * W)
        if positive_desc_b_n is not None:
            positive_targets.append(
                _sample_descriptors_at_yx(
                    positive_desc_b_n[b : b + 1],
                    proj_b[b : b + 1, ids],
                ).squeeze(0)
            )

    if not anchors:
        zero = desc_a.sum() * 0.0 + desc_b.sum() * 0.0
        return {
            "total": zero,
            "info_nce": zero,
            "hard_negative": zero,
            "valid_samples": torch.tensor(0.0, device=desc_a.device),
        }

    anchor_desc = torch.cat(anchors, dim=0)
    target_ids = torch.cat(targets, dim=0)
    sample_weights = torch.cat(weights, dim=0).to(anchor_desc.device).clamp_min(0.0)
    sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-6)
    if positive_desc_b_n is not None:
        bank = positive_desc_b_n.flatten(2).transpose(1, 2).reshape(B * H * W, C)
        positive_desc = torch.cat(positive_targets, dim=0)
        pos_logits = (anchor_desc * positive_desc).sum(dim=-1) / max(float(temperature), 1e-6)
        neg_logits = anchor_desc @ bank.T / max(float(temperature), 1e-6)
        neg_mask = _near_target_mask(
            target_ids,
            neg_logits.shape[1],
            B,
            H,
            W,
            hard_negative_exclusion_radius,
        )
        neg_logits = neg_logits.masked_fill(neg_mask, -1e4)
        logits = torch.cat([pos_logits[:, None], neg_logits], dim=1)
        ce = F.cross_entropy(
            logits,
            torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device),
            reduction="none",
        )
    else:
        bank = desc_b_n.flatten(2).transpose(1, 2).reshape(B * H * W, C)
        logits = anchor_desc @ bank.T / max(float(temperature), 1e-6)
        ce = F.cross_entropy(logits, target_ids, reduction="none")
        pos_logits = logits.gather(1, target_ids[:, None]).squeeze(1)
        neg_mask = _near_target_mask(
            target_ids,
            logits.shape[1],
            B,
            H,
            W,
            hard_negative_exclusion_radius,
        )
        neg_logits = logits.masked_fill(neg_mask, -1e4)
    info_nce = (ce * sample_weights).mean()

    hard_loss = anchor_desc.new_tensor(0.0)
    if hard_negative_weight > 0.0 and neg_logits.shape[1] > 1:
        hardest = neg_logits.max(dim=1).values
        hard_loss = F.softplus(hardest - pos_logits + float(hard_negative_margin)).mean()
    total = info_nce + float(hard_negative_weight) * hard_loss
    return {
        "total": total,
        "info_nce": info_nce,
        "hard_negative": hard_loss,
        "valid_samples": torch.tensor(float(anchor_desc.shape[0]), device=desc_a.device),
    }
