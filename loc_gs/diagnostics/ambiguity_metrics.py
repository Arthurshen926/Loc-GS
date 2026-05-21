from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def descriptor_ambiguity_risk(
    descriptors: torch.Tensor | Any,
    *,
    cosine_threshold: float = 0.95,
    eps: float = 1e-6,
) -> dict[str, float | int]:
    """Compute a descriptor near-duplicate risk for a selected landmark set."""

    desc = torch.as_tensor(descriptors, dtype=torch.float32).reshape(-1, torch.as_tensor(descriptors).shape[-1]).cpu()
    count = int(desc.shape[0])
    if count < 2:
        return {
            "landmark_count": count,
            "pair_count": 0,
            "near_duplicate_pairs": 0,
            "max_cosine": 0.0,
            "mean_excess_cosine": 0.0,
            "ambiguity_risk": 0.0,
        }
    desc = F.normalize(desc, p=2, dim=-1, eps=float(eps))
    cosine = desc @ desc.T
    mask = torch.triu(torch.ones_like(cosine, dtype=torch.bool), diagonal=1)
    values = cosine[mask]
    threshold = float(cosine_threshold)
    near = values >= threshold
    excess = (values[near] - threshold).clamp_min(0.0)
    pair_count = int(values.numel())
    near_count = int(near.sum().item())
    risk = (float(near_count) / float(pair_count)) if pair_count else 0.0
    mean_excess = float(excess.mean().item()) if excess.numel() else 0.0
    return {
        "landmark_count": count,
        "pair_count": pair_count,
        "near_duplicate_pairs": near_count,
        "max_cosine": float(values.max().item()) if values.numel() else 0.0,
        "mean_excess_cosine": mean_excess,
        "ambiguity_risk": risk + mean_excess,
    }

