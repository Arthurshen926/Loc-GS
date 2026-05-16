from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.float().sum() * 0.0


def _candidate_mask(selection_logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    valid = torch.isfinite(selection_logits.float())
    if mask is not None:
        valid = valid & mask.to(device=selection_logits.device, dtype=torch.bool)
    return valid


def _as_candidate_tensor(value: torch.Tensor | float | int, selection_logits: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=selection_logits.device, dtype=torch.float32)
    target_shape = tuple(selection_logits.shape)
    if tuple(tensor.shape) == target_shape:
        return tensor
    if tensor.ndim >= selection_logits.ndim and tuple(tensor.shape[: selection_logits.ndim]) == target_shape:
        extra = tensor.reshape(*target_shape, -1)
        if tensor.ndim >= selection_logits.ndim + 2 and tensor.shape[-1] == tensor.shape[-2]:
            matrix = tensor.reshape(*target_shape, tensor.shape[-2], tensor.shape[-1])
            return matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        return extra.square().sum(dim=-1).sqrt()
    return torch.broadcast_to(tensor, target_shape).float()


def _normalize01(value: torch.Tensor, valid: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.zeros_like(value, dtype=torch.float32)
    finite = valid & torch.isfinite(value)
    if not finite.any():
        return out
    selected = value[finite]
    lo = selected.amin()
    hi = selected.amax()
    scale = (hi - lo).clamp_min(float(eps))
    if (hi - lo) <= float(eps):
        out[finite] = 1.0
    else:
        out[finite] = ((selected - lo) / scale).clamp(0.0, 1.0)
    return out


def pnp_information_proxy_loss(
    selection_logits: torch.Tensor,
    inlier_probability: torch.Tensor,
    pose_information: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Encourage selecting high-inlier, high-pose-information landmarks.

    `pose_information` can be a scalar proxy per candidate, a vector proxy, or a
    per-candidate square information matrix. Matrices are reduced by trace.
    """

    logits = selection_logits.float()
    valid = _candidate_mask(logits, mask)
    if not valid.any():
        return _zero_like_loss(logits)
    inlier = _as_candidate_tensor(inlier_probability, logits).clamp(0.0, 1.0)
    info = _as_candidate_tensor(pose_information, logits).clamp_min(0.0)
    info_norm = _normalize01(info, valid, eps)
    target_weight = (inlier * info_norm).masked_fill(~valid, 0.0)
    denom = target_weight.sum()
    if denom <= float(eps):
        return _zero_like_loss(logits)
    return F.binary_cross_entropy_with_logits(
        logits,
        torch.ones_like(logits),
        weight=target_weight,
        reduction="sum",
    ) / denom.clamp_min(float(eps))


def hard_negative_suppression_loss(
    selection_logits: torch.Tensor,
    hard_negative_risk: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize selecting candidates that look matchable but fail geometry."""

    logits = selection_logits.float()
    valid = _candidate_mask(logits, mask)
    if not valid.any():
        return _zero_like_loss(logits)
    risk = _as_candidate_tensor(hard_negative_risk, logits).clamp(0.0, 1.0).masked_fill(~valid, 0.0)
    denom = risk.sum()
    if denom <= float(eps):
        return _zero_like_loss(logits)
    return F.binary_cross_entropy_with_logits(
        logits,
        torch.zeros_like(logits),
        weight=risk,
        reduction="sum",
    ) / denom.clamp_min(float(eps))


def selection_budget_loss(
    selection_logits: torch.Tensor,
    *,
    target_fraction: float | None = None,
    target_count: int | float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match the expected selected fraction/count under sigmoid scores."""

    logits = selection_logits.float()
    valid = _candidate_mask(logits, mask)
    if not valid.any():
        return _zero_like_loss(logits)
    probs = torch.sigmoid(logits)
    count = valid.float().sum().clamp_min(1.0)
    selected_fraction = (probs.masked_fill(~valid, 0.0).sum() / count).clamp(0.0, 1.0)
    if target_fraction is None:
        if target_count is None:
            target_fraction = 0.5
        else:
            target_fraction = float(target_count) / float(count.item())
    target = min(max(float(target_fraction), 0.0), 1.0)
    return (selected_fraction - logits.new_tensor(target)).square()


def coverage_regularization_loss(
    selection_logits: torch.Tensor,
    positions_xy: torch.Tensor,
    *,
    visibility_score: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    sigma: float = 0.15,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Discourage selected landmarks from collapsing into one image region."""

    logits = selection_logits.float()
    if positions_xy.shape[: logits.ndim] != logits.shape or positions_xy.shape[-1] < 2:
        raise ValueError("positions_xy must have shape selection_logits.shape + (2,)")
    valid = _candidate_mask(logits, mask)
    if not valid.any():
        return _zero_like_loss(logits)
    probs = torch.sigmoid(logits).masked_fill(~valid, 0.0)
    if visibility_score is not None:
        visibility = _as_candidate_tensor(visibility_score, logits).clamp(0.0, 1.0)
        probs = probs * visibility.masked_fill(~valid, 0.0)
    flat_probs = probs.reshape(-1)
    flat_valid = valid.reshape(-1)
    if int((flat_probs > eps).sum().item()) < 2:
        return _zero_like_loss(logits)
    pos = positions_xy[..., :2].to(device=logits.device, dtype=torch.float32).reshape(-1, 2)
    pos = pos.masked_fill(~flat_valid[:, None], 0.0)
    dist2 = torch.cdist(pos, pos).square()
    scale = max(float(sigma), float(eps))
    close_penalty = torch.exp(-dist2 / (2.0 * scale * scale))
    pair_weight = flat_probs[:, None] * flat_probs[None, :]
    eye = torch.eye(pair_weight.shape[0], dtype=torch.bool, device=pair_weight.device)
    pair_weight = pair_weight.masked_fill(eye, 0.0)
    close_penalty = close_penalty.masked_fill(eye, 0.0)
    denom = pair_weight.sum()
    if denom <= float(eps):
        return _zero_like_loss(logits)
    return (pair_weight * close_penalty).sum() / denom.clamp_min(float(eps))


def combined_pose_info_selector_loss(
    selection_logits: torch.Tensor,
    *,
    inlier_probability: torch.Tensor,
    hard_negative_risk: torch.Tensor,
    pose_information: torch.Tensor,
    positions_xy: torch.Tensor | None = None,
    visibility_score: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    budget_target_fraction: float | None = None,
    budget_target_count: int | float | None = None,
    pnp_information_weight: float = 1.0,
    hard_negative_weight: float = 1.0,
    budget_weight: float = 1.0,
    coverage_weight: float = 0.25,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Combine pose-info, hard-negative, budget, and coverage terms."""

    logits = selection_logits.float()
    pnp = pnp_information_proxy_loss(
        logits,
        inlier_probability,
        pose_information,
        mask=mask,
        eps=eps,
    )
    hard_negative = hard_negative_suppression_loss(
        logits,
        hard_negative_risk,
        mask=mask,
        eps=eps,
    )
    budget = selection_budget_loss(
        logits,
        target_fraction=budget_target_fraction,
        target_count=budget_target_count,
        mask=mask,
    )
    if positions_xy is None:
        coverage = _zero_like_loss(logits)
    else:
        coverage = coverage_regularization_loss(
            logits,
            positions_xy,
            visibility_score=visibility_score,
            mask=mask,
            eps=eps,
        )
    total = (
        float(pnp_information_weight) * pnp
        + float(hard_negative_weight) * hard_negative
        + float(budget_weight) * budget
        + float(coverage_weight) * coverage
    )
    return {
        "loss": total,
        "pnp_information": pnp,
        "hard_negative": hard_negative,
        "budget": budget,
        "coverage": coverage,
    }
