from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _as_bool_like(value: torch.Tensor | None, reference: torch.Tensor, default: bool) -> torch.Tensor:
    if value is None:
        return torch.full(reference.shape, bool(default), dtype=torch.bool, device=reference.device)
    tensor = value.to(device=reference.device, dtype=torch.bool)
    if tensor.shape != reference.shape:
        raise ValueError(f"expected shape {tuple(reference.shape)}, got {tuple(tensor.shape)}")
    return tensor


def label_selfmap_pairs(
    reprojection_error: torch.Tensor,
    *,
    visible: torch.Tensor | None = None,
    pnp_inlier: torch.Tensor | None = None,
    reprojection_threshold_px: float = 4.0,
) -> torch.Tensor:
    """Label self-map candidates that are useful for PnP supervision."""

    errors = reprojection_error.float()
    finite = torch.isfinite(errors)
    visibility = _as_bool_like(visible, errors, True)
    inlier = _as_bool_like(pnp_inlier, errors, True)
    return finite & visibility & inlier & (errors <= float(reprojection_threshold_px))


def build_gaussian_advantage_labels(
    landmark_ids: torch.Tensor,
    pair_labels: torch.Tensor,
    *,
    num_landmarks: int | None = None,
    pair_scores: torch.Tensor | None = None,
    false_positive_score_threshold: float | None = None,
    false_positive_weight: float = 1.0,
    smoothing: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Aggregate pair labels into suppressible per-Gaussian advantage targets."""

    lm = landmark_ids.long().reshape(-1)
    labels = pair_labels.to(device=lm.device, dtype=torch.bool).reshape(-1)
    if lm.numel() != labels.numel():
        raise ValueError("landmark_ids and pair_labels must have the same number of elements")
    if num_landmarks is None:
        num_landmarks = int(lm.max().item()) + 1 if lm.numel() else 0
    num_landmarks = int(num_landmarks)
    if num_landmarks < 0:
        raise ValueError("num_landmarks must be non-negative")
    if lm.numel() and (int(lm.min().item()) < 0 or int(lm.max().item()) >= num_landmarks):
        raise IndexError("landmark_ids contain values outside num_landmarks")
    tp = torch.zeros(num_landmarks, dtype=torch.float32, device=lm.device)
    fp = torch.zeros_like(tp)
    ones = torch.ones(lm.numel(), dtype=torch.float32, device=lm.device)
    if labels.any():
        tp.scatter_add_(0, lm[labels], ones[labels])
    fp_mask = ~labels
    if pair_scores is not None and false_positive_score_threshold is not None:
        scores = pair_scores.to(device=lm.device, dtype=torch.float32).reshape(-1)
        if scores.numel() != lm.numel():
            raise ValueError("pair_scores and landmark_ids must have the same number of elements")
        fp_mask = fp_mask & (scores >= float(false_positive_score_threshold))
    if fp_mask.any():
        fp.scatter_add_(0, lm[fp_mask], ones[fp_mask])
    weight = max(float(false_positive_weight), 0.0)
    smooth = max(float(smoothing), 0.0)
    advantage = tp - weight * fp
    denom = tp + weight * fp + 2.0 * smooth
    target = torch.full_like(tp, 0.5)
    observed = denom > 0.0
    target[observed] = (tp[observed] + smooth) / denom[observed].clamp_min(1e-8)
    return {
        "target": target.clamp(0.0, 1.0),
        "advantage": advantage,
        "tp_count": tp,
        "fp_count": fp,
    }


def _first_positive_or_dustbin(pair_label: torch.Tensor) -> torch.Tensor:
    if pair_label.dim() != 2:
        raise ValueError("pair_label must have shape [num_queries, topk]")
    num_queries, topk = int(pair_label.shape[0]), int(pair_label.shape[1])
    labels = torch.full((num_queries,), topk, dtype=torch.long, device=pair_label.device)
    if num_queries == 0 or topk == 0:
        return labels
    any_pos = pair_label.any(dim=1)
    first_pos = pair_label.float().argmax(dim=1).long()
    labels[any_pos] = first_pos[any_pos]
    return labels


def build_selfmap_episode_cache(
    *,
    query_id: torch.Tensor,
    keypoint_yx: torch.Tensor,
    candidate_landmark_ids: torch.Tensor,
    candidate_cosine: torch.Tensor,
    candidate_reprojection_error: torch.Tensor,
    candidate_visible: torch.Tensor | None = None,
    candidate_pnp_inlier: torch.Tensor | None = None,
    episode_success: torch.Tensor | None = None,
    reprojection_threshold_px: float = 4.0,
    num_landmarks: int | None = None,
    false_positive_weight: float = 1.0,
    false_positive_score_threshold: float | None = None,
) -> dict[str, Any]:
    """Pack fine-grained self-localization rehearsal supervision."""

    lm = candidate_landmark_ids.long()
    if lm.dim() != 2:
        raise ValueError("candidate_landmark_ids must have shape [num_queries, topk]")
    num_queries, topk = int(lm.shape[0]), int(lm.shape[1])
    cosine = candidate_cosine.float()
    errors = candidate_reprojection_error.float()
    if cosine.shape != lm.shape or errors.shape != lm.shape:
        raise ValueError("candidate_cosine and candidate_reprojection_error must match candidate_landmark_ids")
    query = query_id.long().reshape(-1)
    if query.shape[0] != num_queries:
        raise ValueError("query_id must have one value per query keypoint")
    yx = keypoint_yx.float()
    if yx.shape != (num_queries, 2):
        raise ValueError(f"keypoint_yx must have shape {(num_queries, 2)}")
    visible = _as_bool_like(candidate_visible, errors, True)
    pnp_inlier = _as_bool_like(candidate_pnp_inlier, errors, True)
    pair_label = label_selfmap_pairs(
        errors,
        visible=visible,
        pnp_inlier=pnp_inlier,
        reprojection_threshold_px=float(reprojection_threshold_px),
    )
    advantage = build_gaussian_advantage_labels(
        lm.reshape(-1),
        pair_label.reshape(-1),
        num_landmarks=num_landmarks,
        pair_scores=cosine.reshape(-1),
        false_positive_score_threshold=false_positive_score_threshold,
        false_positive_weight=false_positive_weight,
    )
    success: torch.Tensor
    if episode_success is None:
        success = torch.ones(num_queries, dtype=torch.bool, device=lm.device)
    else:
        success = episode_success.to(device=lm.device, dtype=torch.bool).reshape(-1)
        if success.numel() == 1:
            success = success.expand(num_queries).clone()
        if success.shape[0] != num_queries:
            raise ValueError("episode_success must be scalar or one value per query")
    return {
        "query_id": query.detach().cpu(),
        "keypoint_yx": yx.detach().cpu(),
        "candidate_landmark_ids": lm.detach().cpu(),
        "candidate_cosine": cosine.detach().cpu(),
        "candidate_reprojection_error": errors.detach().cpu(),
        "candidate_visible": visible.detach().cpu(),
        "candidate_pnp_inlier": pnp_inlier.detach().cpu(),
        "episode_success": success.detach().cpu(),
        "pair_label": pair_label.detach().cpu(),
        "listwise_label": _first_positive_or_dustbin(pair_label).detach().cpu(),
        "gaussian_advantage_target": advantage["target"].detach().cpu(),
        "gaussian_advantage": advantage["advantage"].detach().cpu(),
        "gaussian_tp_count": advantage["tp_count"].detach().cpu(),
        "gaussian_fp_count": advantage["fp_count"].detach().cpu(),
        "metadata": {
            "format": "selfmap_episode_v1",
            "topk": topk,
            "queries": num_queries,
            "reprojection_threshold_px": float(reprojection_threshold_px),
            "false_positive_weight": float(false_positive_weight),
            "false_positive_score_threshold": false_positive_score_threshold,
        },
    }


def save_selfmap_episode_cache(payload: dict[str, Any], path: str | Path) -> None:
    required = {
        "query_id",
        "keypoint_yx",
        "candidate_landmark_ids",
        "candidate_cosine",
        "candidate_reprojection_error",
        "pair_label",
        "listwise_label",
        "gaussian_advantage_target",
    }
    missing = required - set(payload)
    if missing:
        raise KeyError(f"self-map episode cache is missing fields: {sorted(missing)}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
