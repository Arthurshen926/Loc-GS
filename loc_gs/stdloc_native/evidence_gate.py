from __future__ import annotations

from typing import Any

import torch


def _vector(values: torch.Tensor | Any | None, *, size: int | None = None, default: float = 0.0) -> torch.Tensor:
    if values is None:
        if size is None:
            raise ValueError("size is required when values is None")
        return torch.full((int(size),), float(default), dtype=torch.float32)
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1).cpu()
    if size is not None and tensor.shape[0] != int(size):
        raise ValueError("evidence vectors must have the same length")
    return tensor


def _source_mask(source_idx: torch.Tensor | Any | None, size: int) -> torch.Tensor:
    mask = torch.zeros((int(size),), dtype=torch.bool)
    if source_idx is None:
        return mask
    ids = torch.as_tensor(source_idx, dtype=torch.long).reshape(-1).cpu()
    valid = ids[(ids >= 0) & (ids < int(size))]
    if valid.numel():
        mask[valid] = True
    return mask


def build_evidence_gate(
    *,
    selector: torch.Tensor | Any | None = None,
    positive_support: torch.Tensor | Any | None = None,
    hard_negative_risk: torch.Tensor | Any | None = None,
    alpha_reliability: torch.Tensor | Any | None = None,
    multiview_stability: torch.Tensor | Any | None = None,
    pose_utility: torch.Tensor | Any | None = None,
    source_idx: torch.Tensor | Any | None = None,
    keep_source: bool = False,
    min_positive_support: float = 0.0,
    max_hard_negative_risk: float = 1.0,
    min_alpha_reliability: float = 0.0,
    min_multiview_stability: float = 0.0,
    min_pose_utility: float = 0.0,
) -> dict[str, Any]:
    """Build an evidence gate for non-native LSF candidate admission."""

    size_candidates = [
        torch.as_tensor(item).reshape(-1).shape[0]
        for item in (selector, positive_support, hard_negative_risk, alpha_reliability, multiview_stability, pose_utility)
        if item is not None
    ]
    if not size_candidates:
        raise ValueError("at least one evidence vector is required")
    size = int(size_candidates[0])
    if any(int(item) != size for item in size_candidates):
        raise ValueError("all evidence vectors must have the same length")
    selector_t = _vector(selector, size=size, default=0.0).clamp(0.0, 1.0)
    positive = _vector(positive_support, size=size, default=1.0).clamp(0.0, 1.0)
    risk = _vector(hard_negative_risk, size=size, default=0.0).clamp(0.0, 1.0)
    alpha = _vector(alpha_reliability, size=size, default=1.0).clamp(0.0, 1.0)
    stability = _vector(multiview_stability, size=size, default=1.0).clamp(0.0, 1.0)
    pose = _vector(pose_utility, size=size, default=1.0).clamp(0.0, 1.0)
    reject_positive = positive < float(min_positive_support)
    reject_risk = risk > float(max_hard_negative_risk)
    reject_alpha = alpha < float(min_alpha_reliability)
    reject_stability = stability < float(min_multiview_stability)
    reject_pose = pose < float(min_pose_utility)
    mask = ~(reject_positive | reject_risk | reject_alpha | reject_stability | reject_pose)
    source = _source_mask(source_idx, size)
    source_forced = source & ~mask
    if bool(keep_source):
        mask = mask | source
    score = (
        0.25 * selector_t
        + 0.30 * positive
        + 0.20 * pose
        + 0.15 * alpha
        + 0.10 * stability
        - 0.30 * risk
    ).clamp_min(0.0)
    score = torch.where(mask, score, torch.zeros_like(score))
    metadata = {
        "candidate_count": size,
        "accepted_count": int(mask.sum().item()),
        "rejected_by_positive_support": int(reject_positive.sum().item()),
        "rejected_by_hard_negative": int(reject_risk.sum().item()),
        "rejected_by_alpha_reliability": int(reject_alpha.sum().item()),
        "rejected_by_multiview_stability": int(reject_stability.sum().item()),
        "rejected_by_pose_utility": int(reject_pose.sum().item()),
        "source_forced_count": int(source_forced.sum().item()) if bool(keep_source) else 0,
        "thresholds": {
            "min_positive_support": float(min_positive_support),
            "max_hard_negative_risk": float(max_hard_negative_risk),
            "min_alpha_reliability": float(min_alpha_reliability),
            "min_multiview_stability": float(min_multiview_stability),
            "min_pose_utility": float(min_pose_utility),
        },
    }
    return {
        "mask": mask,
        "score": score.float(),
        "metadata": metadata,
    }

