from __future__ import annotations

from typing import Any

import torch


def _vector(value: torch.Tensor | Any, count: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1).cpu()
    if tensor.numel() != count:
        raise ValueError(f"{name} must match dense match count")
    return tensor.clamp(0.0, 1.0)


def score_dense_matches(
    *,
    descriptor_scores: torch.Tensor | Any,
    local_geometry: torch.Tensor | Any,
    lsf_support: torch.Tensor | Any,
    alpha_dominance: torch.Tensor | Any,
    ambiguity: torch.Tensor | Any,
    depth_uncertainty: torch.Tensor | Any,
    pose_leverage: torch.Tensor | Any,
) -> torch.Tensor:
    """Score dense correspondences with support-consistent verification signals."""

    desc = torch.as_tensor(descriptor_scores, dtype=torch.float32).reshape(-1).cpu().clamp(0.0, 1.0)
    count = int(desc.numel())
    geom = _vector(local_geometry, count, "local_geometry")
    support = _vector(lsf_support, count, "lsf_support")
    alpha = _vector(alpha_dominance, count, "alpha_dominance")
    ambiguity_t = _vector(ambiguity, count, "ambiguity")
    depth = _vector(depth_uncertainty, count, "depth_uncertainty")
    leverage = _vector(pose_leverage, count, "pose_leverage")
    positive = 0.30 * desc + 0.22 * geom + 0.22 * support + 0.16 * alpha + 0.10 * leverage
    negative = 0.24 * ambiguity_t + 0.18 * depth
    return (positive - negative).clamp(0.0, 1.0)


def verify_dense_matches(
    *,
    query_yx: torch.Tensor | Any,
    reference_yx: torch.Tensor | Any,
    descriptor_scores: torch.Tensor | Any,
    local_geometry: torch.Tensor | Any,
    lsf_support: torch.Tensor | Any,
    alpha_dominance: torch.Tensor | Any,
    ambiguity: torch.Tensor | Any,
    depth_uncertainty: torch.Tensor | Any,
    pose_leverage: torch.Tensor | Any,
    min_verifier_score: float = 0.5,
    topk: int | None = None,
    reweight_scores: bool = False,
) -> dict[str, Any]:
    """Filter and optionally reweight dense matches before dense refinement."""

    query = torch.as_tensor(query_yx, dtype=torch.float32).reshape(-1, 2).cpu()
    ref = torch.as_tensor(reference_yx, dtype=torch.float32).reshape(-1, 2).cpu()
    desc = torch.as_tensor(descriptor_scores, dtype=torch.float32).reshape(-1).cpu().clamp(0.0, 1.0)
    count = int(query.shape[0])
    if ref.shape[0] != count or desc.numel() != count:
        raise ValueError("query_yx, reference_yx, and descriptor_scores must have matching lengths")
    verifier = score_dense_matches(
        descriptor_scores=desc,
        local_geometry=local_geometry,
        lsf_support=lsf_support,
        alpha_dominance=alpha_dominance,
        ambiguity=ambiguity,
        depth_uncertainty=depth_uncertainty,
        pose_leverage=pose_leverage,
    )
    indices = torch.where(verifier >= float(min_verifier_score))[0]
    scores = desc[indices] * verifier[indices] if bool(reweight_scores) else desc[indices]
    if topk is not None and int(topk) > 0 and indices.numel() > int(topk):
        order = torch.argsort(scores, descending=True, stable=True)[: int(topk)]
        indices = indices[order]
        scores = scores[order]
    return {
        "indices": indices.long(),
        "query_yx": query[indices],
        "reference_yx": ref[indices],
        "scores": scores.float(),
        "verifier_scores": verifier[indices].float(),
        "metadata": {
            "input_count": int(count),
            "kept_count": int(indices.numel()),
            "dropped_count": int(count - indices.numel()),
            "valid_dense_match_ratio": float(indices.numel() / count) if count else 0.0,
            "min_verifier_score": float(min_verifier_score),
            "reweight_scores": bool(reweight_scores),
            "verifier": "lsf_alpha_local_geometry",
        },
    }


def _value(row: dict[str, Any], name: str) -> float | None:
    raw = row.get(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def audit_dense_transition(
    rows: list[dict[str, Any]],
    *,
    sparse_correct_te_cm: float = 5.0,
    sparse_correct_re_deg: float = 5.0,
    dense_wrong_te_cm: float = 10.0,
    dense_wrong_re_deg: float = 5.0,
    worsen_margin_cm: float = 5.0,
) -> dict[str, Any]:
    """Summarize sparse-to-dense transitions for dense-verifier validation."""

    sparse_correct_dense_wrong = 0
    dense_worsened = 0
    dense_improved = 0
    valid = 0
    for row in rows:
        sparse_te = _value(row, "sparse_te_cm")
        sparse_re = _value(row, "sparse_re_deg")
        dense_te = _value(row, "dense_te_cm")
        dense_re = _value(row, "dense_re_deg")
        if sparse_te is None or sparse_re is None or dense_te is None or dense_re is None:
            continue
        valid += 1
        if (
            sparse_te <= float(sparse_correct_te_cm)
            and sparse_re <= float(sparse_correct_re_deg)
            and (dense_te > float(dense_wrong_te_cm) or dense_re > float(dense_wrong_re_deg))
        ):
            sparse_correct_dense_wrong += 1
        delta = dense_te - sparse_te
        if delta >= float(worsen_margin_cm):
            dense_worsened += 1
        elif delta <= -float(worsen_margin_cm):
            dense_improved += 1
    return {
        "query_count": int(valid),
        "sparse_correct_dense_wrong_count": int(sparse_correct_dense_wrong),
        "dense_worsened_count": int(dense_worsened),
        "dense_improved_count": int(dense_improved),
        "dense_verifier_gate": "transition_audit",
    }
