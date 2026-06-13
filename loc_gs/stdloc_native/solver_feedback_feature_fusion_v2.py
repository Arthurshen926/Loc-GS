from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F


def _nonnegative_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if not math.isfinite(number):
        return float(default)
    return max(0.0, float(number))


def fuse_landmark_descriptors(
    native_descriptors: torch.Tensor | Any,
    observations: Mapping[int, list[Mapping[str, Any]]],
    *,
    trust_alpha: float = 0.1,
    min_native_cosine: float = 0.95,
    min_positive_weight: float = 1e-6,
    min_positive_views: int = 1,
) -> tuple[torch.Tensor, dict[str, int | float | str]]:
    """Fuse sampled ULF landmark descriptors using split-safe solver feedback views.

    ``observations`` is keyed by native descriptor row index. Each observation row
    must provide a descriptor plus positive/negative feedback weights. Rows whose
    negative weight exceeds their positive weight are excluded before fusion.
    """

    native = torch.as_tensor(native_descriptors, dtype=torch.float32).detach().cpu()
    if native.dim() != 2:
        raise ValueError("native_descriptors must have shape [landmarks, descriptor_dim]")
    native = F.normalize(native, p=2, dim=-1)
    fused = native.clone()

    alpha = max(0.0, min(1.0, float(trust_alpha)))
    min_shift_cosine = max(-1.0, min(1.0, float(min_native_cosine)))
    min_pos_weight = max(0.0, float(min_positive_weight))
    min_pos_views = max(1, int(min_positive_views))

    observation_count = 0
    positive_observation_count = 0
    negative_view_excluded_count = 0
    observed_landmark_count = 0
    fused_landmark_count = 0
    insufficient_positive_view_count = 0
    trust_region_fallback_count = 0
    native_cosines: list[float] = []

    for raw_gid, rows in observations.items():
        try:
            idx = int(raw_gid)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= int(native.shape[0]):
            continue
        if not isinstance(rows, list):
            continue
        observed_landmark_count += 1
        weighted_descriptors: list[torch.Tensor] = []
        weights: list[float] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            observation_count += 1
            pos = _nonnegative_float(row.get("positive_weight", row.get("weight", 0.0)))
            neg = _nonnegative_float(row.get("negative_weight", row.get("risk", 0.0)))
            if neg > pos:
                negative_view_excluded_count += 1
                continue
            if pos <= min_pos_weight:
                continue
            if "descriptor" not in row:
                raise KeyError("positive observation is missing descriptor")
            desc = torch.as_tensor(row["descriptor"], dtype=torch.float32).reshape(-1).detach().cpu()
            if int(desc.numel()) != int(native.shape[1]):
                raise ValueError("observation descriptor dimension must match native descriptors")
            if not bool(torch.isfinite(desc).all()):
                continue
            weighted_descriptors.append(F.normalize(desc, p=2, dim=0))
            weights.append(float(pos))
            positive_observation_count += 1
        if len(weighted_descriptors) < min_pos_views:
            insufficient_positive_view_count += 1
            continue

        stack = torch.stack(weighted_descriptors, dim=0)
        weight = torch.tensor(weights, dtype=torch.float32).reshape(-1, 1)
        observed_mean = F.normalize((stack * weight).sum(dim=0), p=2, dim=0)
        candidate = F.normalize((1.0 - alpha) * native[idx] + alpha * observed_mean, p=2, dim=0)
        native_cosine = float((candidate * native[idx]).sum().clamp(-1.0, 1.0).item())
        native_cosines.append(native_cosine)
        if native_cosine < min_shift_cosine:
            trust_region_fallback_count += 1
            continue
        fused[idx] = candidate
        fused_landmark_count += 1

    fallback_count = insufficient_positive_view_count + trust_region_fallback_count
    return fused, {
        "descriptor_mode": "solver_feedback_feature_fusion_v2",
        "native_landmark_count": int(native.shape[0]),
        "descriptor_dim": int(native.shape[1]),
        "observation_count": int(observation_count),
        "positive_observation_count": int(positive_observation_count),
        "observed_landmark_count": int(observed_landmark_count),
        "fused_landmark_count": int(fused_landmark_count),
        "fallback_count": int(fallback_count),
        "insufficient_positive_view_count": int(insufficient_positive_view_count),
        "trust_region_fallback_count": int(trust_region_fallback_count),
        "negative_view_excluded_count": int(negative_view_excluded_count),
        "trust_alpha": float(alpha),
        "min_native_cosine": float(min_shift_cosine),
        "min_positive_weight": float(min_pos_weight),
        "min_positive_views": int(min_pos_views),
        "native_cosine_min": float(min(native_cosines)) if native_cosines else 1.0,
        "native_cosine_mean": float(sum(native_cosines) / len(native_cosines)) if native_cosines else 1.0,
    }


def _normalized_observation_descriptor(row: Mapping[str, Any], descriptor_dim: int) -> torch.Tensor | None:
    if "descriptor" not in row:
        return None
    desc = torch.as_tensor(row["descriptor"], dtype=torch.float32).reshape(-1).detach().cpu()
    if int(desc.numel()) != int(descriptor_dim):
        raise ValueError("observation descriptor dimension must match native descriptors")
    if not bool(torch.isfinite(desc).all()):
        return None
    return F.normalize(desc, p=2, dim=0)


def _weighted_top_views(
    items: list[tuple[torch.Tensor, float]],
    *,
    max_views: int,
) -> tuple[list[torch.Tensor], list[float]]:
    if max_views <= 0:
        return [], []
    selected = sorted(items, key=lambda item: float(item[1]), reverse=True)[: int(max_views)]
    return [item[0] for item in selected], [float(item[1]) for item in selected]


def fuse_landmark_descriptors_contrastive(
    native_descriptors: torch.Tensor | Any,
    observations: Mapping[int, list[Mapping[str, Any]]],
    *,
    min_native_cosine: float = 0.95,
    min_positive_weight: float = 1e-6,
    min_negative_weight: float = 1e-6,
    min_positive_views: int = 1,
    max_positive_views: int = 8,
    max_negative_views: int = 8,
    contrastive_steps: int = 20,
    contrastive_lr: float = 0.1,
    anchor_weight: float = 1.0,
    negative_weight_scale: float = 1.0,
    negative_margin: float = 0.2,
) -> tuple[torch.Tensor, dict[str, int | float | str]]:
    """Fuse descriptors by selecting solver-reliable views and repelling bad views.

    Positive PnP-inlier views are used as attraction samples. High-score PnP
    outlier views are never averaged into the descriptor; they act only as
    contrastive negatives during a small anchored optimization.
    """

    native = torch.as_tensor(native_descriptors, dtype=torch.float32).detach().cpu()
    if native.dim() != 2:
        raise ValueError("native_descriptors must have shape [landmarks, descriptor_dim]")
    native = F.normalize(native, p=2, dim=-1)
    fused = native.clone()

    descriptor_dim = int(native.shape[1])
    min_shift_cosine = max(-1.0, min(1.0, float(min_native_cosine)))
    min_pos_weight = max(0.0, float(min_positive_weight))
    min_neg_weight = max(0.0, float(min_negative_weight))
    min_pos_views = max(1, int(min_positive_views))
    max_pos_views = max(0, int(max_positive_views))
    max_neg_views = max(0, int(max_negative_views))
    steps = max(1, int(contrastive_steps))
    lr = max(0.0, float(contrastive_lr))
    anchor = max(0.0, float(anchor_weight))
    neg_scale = max(0.0, float(negative_weight_scale))
    margin = max(-1.0, min(1.0, float(negative_margin)))

    observation_count = 0
    positive_view_candidate_count = 0
    positive_view_selected_count = 0
    negative_view_candidate_count = 0
    negative_view_selected_count = 0
    negative_view_excluded_count = 0
    observed_landmark_count = 0
    fused_landmark_count = 0
    insufficient_positive_view_count = 0
    trust_region_fallback_count = 0
    contrastive_optimized_landmark_count = 0
    native_cosines: list[float] = []

    for raw_gid, rows in observations.items():
        try:
            idx = int(raw_gid)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= int(native.shape[0]):
            continue
        if not isinstance(rows, list):
            continue
        observed_landmark_count += 1

        positive_items: list[tuple[torch.Tensor, float]] = []
        negative_items: list[tuple[torch.Tensor, float]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            observation_count += 1
            pos = _nonnegative_float(row.get("positive_weight", row.get("weight", 0.0)))
            neg = _nonnegative_float(row.get("negative_weight", row.get("risk", 0.0)))
            desc = _normalized_observation_descriptor(row, descriptor_dim)
            if desc is None:
                if pos > min_pos_weight:
                    raise KeyError("positive observation is missing descriptor")
                continue
            if neg > pos:
                negative_view_excluded_count += 1
            if pos > min_pos_weight and pos >= neg:
                positive_items.append((desc, pos))
                positive_view_candidate_count += 1
            if neg > min_neg_weight and neg > pos:
                negative_items.append((desc, neg))
                negative_view_candidate_count += 1

        positive_views, positive_weights = _weighted_top_views(positive_items, max_views=max_pos_views)
        negative_views, negative_weights = _weighted_top_views(negative_items, max_views=max_neg_views)
        positive_view_selected_count += len(positive_views)
        negative_view_selected_count += len(negative_views)
        if len(positive_views) < min_pos_views:
            insufficient_positive_view_count += 1
            continue

        positive_stack = torch.stack(positive_views, dim=0)
        positive_weight = torch.tensor(positive_weights, dtype=torch.float32)
        positive_weight = positive_weight / positive_weight.sum().clamp_min(1e-12)
        if negative_views:
            negative_stack = torch.stack(negative_views, dim=0)
            negative_weight = torch.tensor(negative_weights, dtype=torch.float32)
            negative_weight = negative_weight / negative_weight.sum().clamp_min(1e-12)
        else:
            negative_stack = torch.empty(0, descriptor_dim, dtype=torch.float32)
            negative_weight = torch.empty(0, dtype=torch.float32)

        candidate = native[idx].clone().detach()
        if lr > 0.0:
            for _step in range(steps):
                candidate_var = candidate.detach().clone().requires_grad_(True)
                descriptor = F.normalize(candidate_var, p=2, dim=0)
                pos_cos = positive_stack @ descriptor
                positive_loss = ((1.0 - pos_cos) * positive_weight).sum()
                if int(negative_stack.shape[0]) > 0 and neg_scale > 0.0:
                    neg_cos = negative_stack @ descriptor
                    negative_loss = (F.relu(neg_cos - margin) * negative_weight).sum()
                else:
                    negative_loss = torch.tensor(0.0, dtype=torch.float32)
                anchor_loss = 1.0 - torch.clamp((descriptor * native[idx]).sum(), -1.0, 1.0)
                loss = positive_loss + neg_scale * negative_loss + anchor * anchor_loss
                grad = torch.autograd.grad(loss, candidate_var, allow_unused=False)[0]
                candidate = F.normalize(candidate_var.detach() - lr * grad.detach(), p=2, dim=0)
        else:
            candidate = F.normalize((positive_stack * positive_weight.reshape(-1, 1)).sum(dim=0), p=2, dim=0)

        native_cosine = float((candidate * native[idx]).sum().clamp(-1.0, 1.0).item())
        native_cosines.append(native_cosine)
        if native_cosine < min_shift_cosine:
            trust_region_fallback_count += 1
            continue
        fused[idx] = candidate
        fused_landmark_count += 1
        contrastive_optimized_landmark_count += 1

    fallback_count = insufficient_positive_view_count + trust_region_fallback_count
    return fused, {
        "descriptor_mode": "solver_feedback_contrastive_fusion_v1",
        "view_pruning_mode": "positive_inlier_topk_negative_outlier_contrastive",
        "native_landmark_count": int(native.shape[0]),
        "descriptor_dim": int(native.shape[1]),
        "observation_count": int(observation_count),
        "positive_view_candidate_count": int(positive_view_candidate_count),
        "positive_view_selected_count": int(positive_view_selected_count),
        "negative_view_candidate_count": int(negative_view_candidate_count),
        "negative_view_selected_count": int(negative_view_selected_count),
        "negative_view_excluded_count": int(negative_view_excluded_count),
        "observed_landmark_count": int(observed_landmark_count),
        "fused_landmark_count": int(fused_landmark_count),
        "contrastive_optimized_landmark_count": int(contrastive_optimized_landmark_count),
        "fallback_count": int(fallback_count),
        "insufficient_positive_view_count": int(insufficient_positive_view_count),
        "trust_region_fallback_count": int(trust_region_fallback_count),
        "min_native_cosine": float(min_shift_cosine),
        "min_positive_weight": float(min_pos_weight),
        "min_negative_weight": float(min_neg_weight),
        "min_positive_views": int(min_pos_views),
        "max_positive_views": int(max_pos_views),
        "max_negative_views": int(max_neg_views),
        "contrastive_steps": int(steps),
        "contrastive_lr": float(lr),
        "anchor_weight": float(anchor),
        "negative_weight_scale": float(neg_scale),
        "negative_margin": float(margin),
        "native_cosine_min": float(min(native_cosines)) if native_cosines else 1.0,
        "native_cosine_mean": float(sum(native_cosines) / len(native_cosines)) if native_cosines else 1.0,
    }
