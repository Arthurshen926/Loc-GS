from __future__ import annotations

import torch
import torch.nn.functional as F


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.float().sum() * 0.0


def _weights_like(value: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.ones(reference.shape[0], device=reference.device, dtype=torch.float32)
    weights = torch.as_tensor(value, device=reference.device, dtype=torch.float32).reshape(-1)
    if weights.numel() != reference.shape[0]:
        raise ValueError(f"weights has {weights.numel()} items, expected {reference.shape[0]}")
    return weights.clamp_min(0.0)


def _weighted_mean(loss: torch.Tensor, weights: torch.Tensor, eps: float) -> torch.Tensor:
    denom = weights.sum()
    if denom <= float(eps):
        return _zero(loss)
    return (loss.reshape(-1) * weights.reshape(-1)).sum() / denom.clamp_min(float(eps))


def hard_negative_margin_loss(
    query_desc: torch.Tensor,
    pos_desc: torch.Tensor,
    neg_desc: torch.Tensor,
    *,
    margin: float = 0.2,
    weights: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Triplet-style descriptor loss for feedback-bank hard negatives."""

    if query_desc.shape != pos_desc.shape or query_desc.shape != neg_desc.shape:
        raise ValueError("query_desc, pos_desc, and neg_desc must have the same shape")
    if query_desc.numel() == 0:
        return _zero(query_desc)
    query = F.normalize(query_desc.float(), p=2, dim=-1)
    pos = F.normalize(pos_desc.float(), p=2, dim=-1)
    neg = F.normalize(neg_desc.float(), p=2, dim=-1)
    pos_sim = (query * pos).sum(dim=-1)
    neg_sim = (query * neg).sum(dim=-1)
    raw = F.relu(float(margin) + neg_sim - pos_sim)
    return _weighted_mean(raw, _weights_like(weights, raw), eps)


def residual_trust_region_loss(
    base_desc: torch.Tensor,
    residual_desc: torch.Tensor,
    alpha_max: float,
    *,
    weights: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize descriptor residuals that exceed the protected alpha cap."""

    if base_desc.shape != residual_desc.shape:
        raise ValueError("base_desc and residual_desc must have the same shape")
    if base_desc.numel() == 0:
        return _zero(base_desc)
    residual_norm = (residual_desc.float() - base_desc.float()).norm(dim=-1)
    cap = max(float(alpha_max), 0.0)
    raw = F.relu(residual_norm - cap).square()
    return _weighted_mean(raw, _weights_like(weights, raw), eps)


def feedback_weighted_descriptor_loss(
    query_desc: torch.Tensor,
    pos_desc: torch.Tensor,
    neg_desc: torch.Tensor,
    *,
    base_desc: torch.Tensor,
    residual_desc: torch.Tensor,
    hard_negative_weight: torch.Tensor | None = None,
    margin: float = 0.2,
    alpha_max: float = 0.03,
    margin_weight: float = 1.0,
    residual_trust_region_weight: float = 1.0,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Combine hard-negative margin and protected residual trust-region terms."""

    weights = _weights_like(hard_negative_weight, query_desc)
    margin_term = hard_negative_margin_loss(
        query_desc,
        pos_desc,
        neg_desc,
        margin=margin,
        weights=weights,
        eps=eps,
    )
    trust_region = residual_trust_region_loss(
        base_desc,
        residual_desc,
        alpha_max=alpha_max,
        weights=weights,
        eps=eps,
    )
    total = float(margin_weight) * margin_term + float(residual_trust_region_weight) * trust_region
    return {
        "loss": total,
        "margin": margin_term,
        "trust_region": trust_region,
    }
