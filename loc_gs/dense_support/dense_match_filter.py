from __future__ import annotations

from typing import Any

import torch


def filter_dense_matches(
    *,
    query_yx: torch.Tensor | Any,
    reference_yx: torch.Tensor | Any,
    scores: torch.Tensor | Any,
    reliability: torch.Tensor | Any,
    min_reliability: float = 0.5,
    topk: int | None = None,
    reweight_scores: bool = False,
) -> dict[str, Any]:
    """Filter dense correspondences with a rendered support/reliability mask."""

    query = torch.as_tensor(query_yx, dtype=torch.float32).reshape(-1, 2).cpu()
    ref = torch.as_tensor(reference_yx, dtype=torch.float32).reshape(-1, 2).cpu()
    score = torch.as_tensor(scores, dtype=torch.float32).reshape(-1).cpu()
    rel = torch.as_tensor(reliability, dtype=torch.float32).reshape(-1).cpu().clamp(0.0, 1.0)
    count = int(query.shape[0])
    if ref.shape[0] != count or score.shape[0] != count or rel.shape[0] != count:
        raise ValueError("query_yx, reference_yx, scores, and reliability must have matching lengths")
    keep = rel >= float(min_reliability)
    indices = torch.where(keep)[0]
    filtered_scores = score[indices] * rel[indices] if bool(reweight_scores) else score[indices]
    if topk is not None and int(topk) > 0 and indices.numel() > int(topk):
        order = torch.argsort(filtered_scores, descending=True, stable=True)[: int(topk)]
        indices = indices[order]
        filtered_scores = filtered_scores[order]
    metadata = {
        "input_count": count,
        "kept_count": int(indices.numel()),
        "dropped_count": int(count - indices.numel()),
        "min_reliability": float(min_reliability),
        "topk": int(topk) if topk is not None else None,
        "reweight_scores": bool(reweight_scores),
    }
    return {
        "indices": indices.long(),
        "query_yx": query[indices],
        "reference_yx": ref[indices],
        "scores": filtered_scores.float(),
        "reliability": rel[indices],
        "metadata": metadata,
    }

