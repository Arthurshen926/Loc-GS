from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _sample_desc(desc_map: torch.Tensor, keypoints_yx: torch.Tensor) -> torch.Tensor:
    B, C, H, W = desc_map.shape
    y = keypoints_yx[..., 0]
    x = keypoints_yx[..., 1]
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


def external_match_supervision_loss(
    desc_a: torch.Tensor,
    desc_b: torch.Tensor,
    kpts_a_yx: torch.Tensor,
    kpts_b_yx: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    negative_kpts_b_yx: Optional[torch.Tensor] = None,
    temperature: float = 0.07,
    hard_negative_weight: float = 0.0,
    hard_negative_margin: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Differentiable proxy loss for external detector-free match labels."""
    if desc_a.requires_grad:
        desc_a.retain_grad()
    if desc_b.requires_grad:
        desc_b.retain_grad()
    B = desc_a.shape[0]
    if kpts_a_yx.numel() == 0 or kpts_b_yx.numel() == 0:
        zero = desc_a.sum() * 0.0 + desc_b.sum() * 0.0
        return {
            "total": zero,
            "positive": zero,
            "hard_negative": zero,
            "valid_matches": torch.tensor(0.0, device=desc_a.device),
        }
    if kpts_a_yx.dim() == 2:
        kpts_a_yx = kpts_a_yx.unsqueeze(0).expand(B, -1, -1)
    if kpts_b_yx.dim() == 2:
        kpts_b_yx = kpts_b_yx.unsqueeze(0).expand(B, -1, -1)
    kpts_a_yx = kpts_a_yx.to(device=desc_a.device, dtype=desc_a.dtype)
    kpts_b_yx = kpts_b_yx.to(device=desc_a.device, dtype=desc_a.dtype)
    desc_pos_a = _sample_desc(desc_a, kpts_a_yx)
    desc_pos_b = _sample_desc(desc_b, kpts_b_yx)
    pos_cos = (desc_pos_a * desc_pos_b).sum(dim=-1)
    if scores is None:
        weights = torch.ones_like(pos_cos)
    else:
        weights = scores.to(device=desc_a.device, dtype=desc_a.dtype)
        if weights.dim() == 1:
            weights = weights.unsqueeze(0).expand_as(pos_cos)
        weights = weights.clamp_min(0.0)
    if valid_mask is not None:
        valid = valid_mask.to(device=desc_a.device, dtype=torch.bool)
        weights = weights * valid.float()
    else:
        valid = torch.ones_like(weights, dtype=torch.bool)
    if not bool(valid.any()):
        zero = desc_a.sum() * 0.0 + desc_b.sum() * 0.0
        return {
            "total": zero,
            "positive": zero,
            "hard_negative": zero,
            "valid_matches": torch.tensor(0.0, device=desc_a.device),
        }
    weights = weights / weights.mean().clamp_min(1e-6)
    positive = ((1.0 - pos_cos) * weights).mean()

    hard_negative = desc_a.new_tensor(0.0)
    if hard_negative_weight > 0.0 and negative_kpts_b_yx is not None and negative_kpts_b_yx.numel() > 0:
        if negative_kpts_b_yx.dim() == 2:
            negative_kpts_b_yx = negative_kpts_b_yx.unsqueeze(0).expand(B, -1, -1)
        negative_kpts_b_yx = negative_kpts_b_yx.to(device=desc_a.device, dtype=desc_a.dtype)
        neg_desc = _sample_desc(desc_b, negative_kpts_b_yx)
        if neg_desc.shape[1] == desc_pos_a.shape[1]:
            neg_cos = (desc_pos_a * neg_desc).sum(dim=-1)
        else:
            neg_cos = torch.einsum("bnc,bmc->bnm", desc_pos_a, neg_desc).amax(dim=-1)
        hard_raw = F.softplus((neg_cos - pos_cos + float(hard_negative_margin)) / max(float(temperature), 1e-6))
        hard_negative = (hard_raw * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

    total = positive + float(hard_negative_weight) * hard_negative
    return {
        "total": total,
        "positive": positive,
        "hard_negative": hard_negative,
        "valid_matches": torch.tensor(float(valid.sum().item()), device=desc_a.device),
    }
