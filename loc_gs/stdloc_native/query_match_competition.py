from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class QueryMatchCompetitionResult:
    stealer_scores: torch.Tensor
    support_match_strength: torch.Tensor
    top1_gaussian_ids: torch.Tensor
    support_best_gaussian_ids: torch.Tensor
    support_best_scores: torch.Tensor
    support_best_rank: torch.Tensor
    metadata: dict[str, float | int]


def _as_feature_matrix(values: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [N, C]")
    if int(tensor.shape[1]) <= 0:
        raise ValueError(f"{name} must have at least one channel")
    return F.normalize(tensor.cpu(), dim=1)


def _as_sampled_idx(values: torch.Tensor, *, rows: int) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.long).reshape(-1).cpu()
    if int(tensor.numel()) != int(rows):
        raise ValueError(f"sampled_idx length {tensor.numel()} does not match landmark feature rows {rows}")
    if int(tensor.numel()) == 0:
        raise ValueError("sampled_idx must not be empty")
    if int(tensor.min().item()) < 0:
        raise ValueError("sampled_idx must contain non-negative Gaussian ids")
    return tensor


def _support_weights(
    query_support: Mapping[int, float] | Mapping[str, float],
    *,
    num_gaussians: int,
) -> torch.Tensor:
    weights = torch.zeros((int(num_gaussians),), dtype=torch.float32)
    for raw_gid, raw_value in query_support.items():
        try:
            gid = int(raw_gid)
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if gid < 0 or gid >= int(num_gaussians) or value <= 0.0:
            continue
        weights[gid] += value
    return weights


def compute_query_match_competition(
    *,
    query_features: torch.Tensor,
    sampled_idx: torch.Tensor,
    landmark_features: torch.Tensor,
    query_support: Mapping[int, float] | Mapping[str, float],
    top_k: int = 10,
    score_margin: float = 0.05,
) -> QueryMatchCompetitionResult:
    """Score selected landmarks that steal query matches from solver-validated support.

    This is intentionally query-level. Static descriptor similarity alone can
    miss the real failure mode: for a given query descriptor, a non-support
    landmark can become top-1 while a solver-validated support landmark is still
    nearby in top-K. Those top-1 non-support landmarks are competition risks.
    """

    query = _as_feature_matrix(query_features, name="query_features")
    landmarks = _as_feature_matrix(landmark_features, name="landmark_features")
    sampled = _as_sampled_idx(sampled_idx, rows=int(landmarks.shape[0]))
    max_gid = int(sampled.max().item())
    support = _support_weights(query_support, num_gaussians=max_gid + 1)
    selected_support_weights = support[sampled]
    support_mask = selected_support_weights > 0.0
    support_positions = torch.where(support_mask)[0]

    empty_ids = torch.full((int(query.shape[0]),), -1, dtype=torch.long)
    empty_scores = torch.full((int(query.shape[0]),), -float("inf"), dtype=torch.float32)
    empty_ranks = torch.full((int(query.shape[0]),), int(sampled.numel()) + 1, dtype=torch.long)
    if int(support_positions.numel()) == 0:
        return QueryMatchCompetitionResult(
            stealer_scores=torch.zeros((max_gid + 1,), dtype=torch.float32),
            support_match_strength=torch.zeros((max_gid + 1,), dtype=torch.float32),
            top1_gaussian_ids=empty_ids,
            support_best_gaussian_ids=empty_ids,
            support_best_scores=empty_scores,
            support_best_rank=empty_ranks,
            metadata={
                "query_feature_count": int(query.shape[0]),
                "selected_count": int(sampled.numel()),
                "support_present_count": 0,
                "support_top1_count": 0,
                "support_topk_count": 0,
                "unique_support_topk_count": 0,
                "feature_weak_query_feature_count": int(query.shape[0]),
                "stealer_count": 0,
                "support_rank_p50": float(int(sampled.numel()) + 1),
                "support_rank_p90": float(int(sampled.numel()) + 1),
                "mean_top1_minus_support_score": 0.0,
                "mean_support_best_score": 0.0,
                "support_match_strength_nonzero_count": 0,
                "support_match_strength_total": 0.0,
                "support_match_strength_max": 0.0,
            },
        )

    corr = query @ landmarks.T
    top1_scores, top1_positions = corr.max(dim=1)
    top1_gids = sampled[top1_positions]

    support_corr = corr[:, support_positions]
    support_best_scores, local_support_pos = support_corr.max(dim=1)
    support_best_positions = support_positions[local_support_pos]
    support_best_gids = sampled[support_best_positions]
    support_best_weights = support[support_best_gids]
    support_best_rank = (corr > support_best_scores[:, None]).sum(dim=1).to(torch.long) + 1

    k = max(1, int(top_k))
    support_top1 = support_best_rank <= 1
    support_topk = support_best_rank <= k
    support_rank_float = support_best_rank.to(torch.float32)
    unique_support_topk = torch.unique(support_best_gids[support_topk]) if bool(support_topk.any()) else torch.empty(0)
    top1_is_support = support[top1_gids] > 0.0
    steal_mask = (~top1_is_support) & support_topk & (top1_scores <= support_best_scores + float(score_margin))

    stealer_scores = torch.zeros((max_gid + 1,), dtype=torch.float32)
    if bool(steal_mask.any()):
        steal_gids = top1_gids[steal_mask]
        steal_values = support_best_weights[steal_mask] * (
            float(score_margin) + top1_scores[steal_mask] - support_best_scores[steal_mask]
        ).clamp_min(0.0)
        stealer_scores.scatter_add_(0, steal_gids.to(torch.long), steal_values.to(torch.float32))

    support_match_strength = torch.zeros((max_gid + 1,), dtype=torch.float32)
    if bool(support_topk.any()):
        rank_weight = ((float(k) + 1.0 - support_best_rank.to(torch.float32)) / float(k)).clamp_min(0.0)
        support_values = (
            support_best_weights[support_topk]
            * support_best_scores[support_topk].clamp_min(0.0)
            * rank_weight[support_topk]
        )
        support_match_strength.scatter_add_(
            0,
            support_best_gids[support_topk].to(torch.long),
            support_values.to(torch.float32),
        )

    return QueryMatchCompetitionResult(
        stealer_scores=stealer_scores,
        support_match_strength=support_match_strength,
        top1_gaussian_ids=top1_gids.to(torch.long),
        support_best_gaussian_ids=support_best_gids.to(torch.long),
        support_best_scores=support_best_scores.to(torch.float32),
        support_best_rank=support_best_rank.to(torch.long),
        metadata={
            "query_feature_count": int(query.shape[0]),
            "selected_count": int(sampled.numel()),
            "support_present_count": int(support_positions.numel()),
            "support_top1_count": int(support_top1.sum().item()),
            "support_topk_count": int(support_topk.sum().item()),
            "unique_support_topk_count": int(unique_support_topk.numel()),
            "feature_weak_query_feature_count": int((~support_topk).sum().item()),
            "stealer_count": int((stealer_scores > 0.0).sum().item()),
            "steal_event_count": int(steal_mask.sum().item()),
            "support_rank_p50": float(torch.quantile(support_rank_float, 0.50).item()),
            "support_rank_p90": float(torch.quantile(support_rank_float, 0.90).item()),
            "mean_top1_minus_support_score": float((top1_scores - support_best_scores).mean().item()),
            "mean_support_best_score": float(support_best_scores.mean().item()),
            "support_match_strength_nonzero_count": int((support_match_strength > 0.0).sum().item()),
            "support_match_strength_total": float(support_match_strength.sum().item()),
            "support_match_strength_max": float(support_match_strength.max().item())
            if int(support_match_strength.numel()) > 0
            else 0.0,
            "top_k": int(k),
            "score_margin": float(score_margin),
        },
    )
