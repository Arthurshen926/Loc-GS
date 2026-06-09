from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MatchCompetitionPruneResult:
    pruned_sampled_idx: torch.Tensor
    pruned_features: torch.Tensor
    keep_mask: torch.Tensor
    selected_risk_scores: torch.Tensor
    metadata: dict[str, Any]


def _as_sampled(values: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.long).reshape(-1).cpu()
    if int(tensor.numel()) == 0:
        raise ValueError("sampled_idx must not be empty")
    if int(tensor.min().item()) < 0:
        raise ValueError("sampled_idx must contain non-negative ids")
    return tensor


def _as_features(values: torch.Tensor, *, rows: int) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError("features must have shape [N, C]")
    if int(tensor.shape[0]) != int(rows):
        raise ValueError(f"features row count {tensor.shape[0]} does not match sampled_idx length {rows}")
    return tensor.cpu()


def _selected_scores(scores: torch.Tensor, sampled: torch.Tensor, *, name: str) -> torch.Tensor:
    vector = torch.as_tensor(scores, dtype=torch.float32).reshape(-1).cpu()
    max_gid = int(sampled.max().item())
    if int(vector.numel()) <= max_gid:
        raise ValueError(f"{name} length {vector.numel()} does not cover sampled gid {max_gid}")
    return vector[sampled].to(torch.float32)


def prune_match_competition(
    *,
    sampled_idx: torch.Tensor,
    features: torch.Tensor,
    risk_scores: torch.Tensor,
    protect_scores: torch.Tensor | None = None,
    min_protect_score: float = 0.0,
    min_risk_score: float = 0.0,
    max_prune_count: int = 0,
    min_keep_count: int = 0,
) -> MatchCompetitionPruneResult:
    sampled = _as_sampled(sampled_idx)
    feats = _as_features(features, rows=int(sampled.numel()))
    selected_risk = _selected_scores(risk_scores, sampled, name="risk_scores")
    selected_protect = (
        torch.zeros_like(selected_risk)
        if protect_scores is None
        else _selected_scores(protect_scores, sampled, name="protect_scores")
    )
    risky = selected_risk >= float(min_risk_score)
    if float(min_risk_score) <= 0.0:
        risky = selected_risk > 0.0
    protected = selected_protect >= float(min_protect_score) if protect_scores is not None else torch.zeros_like(risky)
    candidates = torch.where(risky & ~protected)[0]
    protected_risky = risky & protected

    prune_limit = int(candidates.numel())
    if int(max_prune_count) > 0:
        prune_limit = min(prune_limit, int(max_prune_count))
    if int(min_keep_count) > 0:
        prune_limit = min(prune_limit, max(0, int(sampled.numel()) - int(min_keep_count)))

    keep = torch.ones((int(sampled.numel()),), dtype=torch.bool)
    pruned_positions = torch.empty((0,), dtype=torch.long)
    if prune_limit > 0:
        order = torch.argsort(selected_risk[candidates], descending=True)
        pruned_positions = candidates[order[:prune_limit]]
        keep[pruned_positions] = False

    pruned_idx = sampled[keep].contiguous()
    pruned_features = feats[keep].contiguous()
    metadata = {
        "method": "match_competition_pruning",
        "input_sampled_count": int(sampled.numel()),
        "output_sampled_count": int(pruned_idx.numel()),
        "risky_selected_count": int(risky.sum().item()),
        "solver_protected_risky_count": int(protected_risky.sum().item()),
        "match_competition_candidate_count": int(candidates.numel()),
        "match_competition_pruned_count": int(pruned_positions.numel()),
        "min_risk_score": float(min_risk_score),
        "min_protect_score": float(min_protect_score),
        "max_prune_count": int(max_prune_count),
        "min_keep_count": int(min_keep_count),
        "max_selected_risk_score": float(selected_risk.max().item()) if selected_risk.numel() else 0.0,
        "mean_pruned_risk_score": float(selected_risk[pruned_positions].mean().item())
        if pruned_positions.numel()
        else 0.0,
    }
    return MatchCompetitionPruneResult(
        pruned_sampled_idx=pruned_idx,
        pruned_features=pruned_features,
        keep_mask=keep,
        selected_risk_scores=selected_risk,
        metadata=metadata,
    )
