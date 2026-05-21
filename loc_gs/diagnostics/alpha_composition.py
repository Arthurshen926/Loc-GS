from __future__ import annotations

from typing import Any

import torch


def _normalize_weights(weights: torch.Tensor | Any, eps: float = 1e-8) -> torch.Tensor:
    tensor = torch.as_tensor(weights, dtype=torch.float32).reshape(torch.as_tensor(weights).shape).cpu()
    if tensor.dim() < 2:
        raise ValueError("composition weights must have shape [..., contributors]")
    tensor = tensor.clamp_min(0.0)
    denom = tensor.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    return tensor / denom


def composition_entropy(weights: torch.Tensor | Any, *, eps: float = 1e-8) -> torch.Tensor:
    """Return normalized entropy for alpha/ray composition weights."""

    probs = _normalize_weights(weights, eps=eps)
    entropy = -(probs * (probs + float(eps)).log()).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(probs.shape[-1]), dtype=torch.float32)).clamp_min(float(eps))
    return (entropy / max_entropy).clamp(0.0, 1.0)


def composition_support_aggregation(
    contributor_ids: torch.Tensor | Any,
    contribution_weights: torch.Tensor | Any,
    gaussian_support: torch.Tensor | Any,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Aggregate per-Gaussian support to rendered rays/pixels by contribution weight."""

    ids = torch.as_tensor(contributor_ids, dtype=torch.long).cpu()
    weights = _normalize_weights(torch.as_tensor(contribution_weights, dtype=torch.float32).cpu(), eps=eps)
    if ids.shape != weights.shape:
        raise ValueError("contributor_ids and contribution_weights must have the same shape")
    support = torch.as_tensor(gaussian_support, dtype=torch.float32).reshape(-1).cpu().clamp(0.0, 1.0)
    if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= support.numel()):
        raise IndexError("contributor_ids outside gaussian_support")
    return (weights * support[ids]).sum(dim=-1).clamp(0.0, 1.0)


def alpha_composition_reliability(
    contribution_weights: torch.Tensor | Any,
    *,
    opacity: torch.Tensor | Any | None = None,
    depth_stability: torch.Tensor | Any | None = None,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Proxy reliability for alpha-composited rendered features/rays.

    Exact raster composition weights are ideal. When unavailable, callers can
    feed proxy contribution weights and mark downstream reports accordingly.
    """

    weights = _normalize_weights(contribution_weights, eps=eps)
    dominance = weights.max(dim=-1).values.clamp(0.0, 1.0)
    entropy = composition_entropy(weights, eps=eps)
    low_entropy = (1.0 - entropy).clamp(0.0, 1.0)
    reliability = dominance * low_entropy
    if opacity is not None:
        opacity_tensor = torch.as_tensor(opacity, dtype=torch.float32).reshape(reliability.shape).cpu().clamp(0.0, 1.0)
        reliability = reliability * opacity_tensor
    if depth_stability is not None:
        depth_tensor = torch.as_tensor(depth_stability, dtype=torch.float32).reshape(reliability.shape).cpu().clamp(0.0, 1.0)
        reliability = reliability * depth_tensor
    return {
        "dominance": dominance,
        "entropy": entropy,
        "low_entropy": low_entropy,
        "reliability": reliability.clamp(0.0, 1.0),
    }

