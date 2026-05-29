from __future__ import annotations

from typing import Any
from pathlib import Path

import torch
import torch.nn.functional as F


def _weights(value: torch.Tensor | Any | None, count: int) -> torch.Tensor:
    if value is None:
        return torch.ones(count, dtype=torch.float32)
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1).cpu()
    if tensor.numel() != count:
        raise ValueError("weight tensors must match observed descriptor count")
    return tensor.clamp_min(0.0)


def solver_weighted_landmark_feature_fusion(
    *,
    native_descriptors: torch.Tensor | Any,
    landmark_ids: torch.Tensor | Any,
    observed_descriptors: torch.Tensor | Any,
    solver_weights: torch.Tensor | Any,
    geometry_weights: torch.Tensor | Any | None = None,
    visibility_weights: torch.Tensor | Any | None = None,
    trust_alpha: float = 0.1,
    min_total_weight: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fuse landmark descriptors with self-localization solver feedback weights."""

    native = F.normalize(torch.as_tensor(native_descriptors, dtype=torch.float32).cpu(), p=2, dim=-1)
    ids = torch.as_tensor(landmark_ids, dtype=torch.long).reshape(-1).cpu()
    observed = F.normalize(torch.as_tensor(observed_descriptors, dtype=torch.float32).cpu(), p=2, dim=-1)
    if observed.dim() != 2 or native.dim() != 2:
        raise ValueError("native_descriptors and observed_descriptors must have shape [N,D]")
    if ids.numel() != observed.shape[0]:
        raise ValueError("landmark_ids must match observed_descriptors rows")
    if observed.shape[1] != native.shape[1]:
        raise ValueError("descriptor dimensions must match")
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= native.shape[0]):
        raise IndexError("landmark_ids outside native descriptor table")

    count = int(ids.numel())
    total_weight = (
        _weights(solver_weights, count)
        * _weights(geometry_weights, count)
        * _weights(visibility_weights, count)
    )
    accum = torch.zeros_like(native)
    denom = torch.zeros(native.shape[0], dtype=torch.float32)
    for row, gid in enumerate(ids.tolist()):
        weight = float(total_weight[row].item())
        if weight <= 0.0:
            continue
        accum[int(gid)] += observed[row] * weight
        denom[int(gid)] += weight

    fused = native.clone()
    supported = torch.where(denom >= float(min_total_weight))[0]
    alpha = max(0.0, min(1.0, float(trust_alpha)))
    if supported.numel() > 0 and alpha > 0.0:
        observed_mean = F.normalize(accum[supported] / denom[supported].clamp_min(1e-12).unsqueeze(1), p=2, dim=-1)
        fused[supported] = F.normalize((1.0 - alpha) * native[supported] + alpha * observed_mean, p=2, dim=-1)
    metadata = {
        "descriptor_mode": "solver_weighted_landmark_fusion",
        "observation_count": int(count),
        "fused_landmark_count": int(supported.numel()),
        "fallback_landmark_count": int(native.shape[0] - supported.numel()),
        "trust_alpha": float(alpha),
        "min_total_weight": float(min_total_weight),
    }
    return fused, metadata


def _load_pair_cache(pair_cache: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(pair_cache, (str, Path)):
        payload = torch.load(Path(pair_cache), map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("pair cache must contain a dict")
        return payload
    return dict(pair_cache)


def _reject_test_split(metadata: dict[str, Any]) -> None:
    for key in ("source_split_name", "feedback_bank_split_name", "split_name"):
        value = str(metadata.get(key, "")).lower()
        if value == "test" or value.endswith("_test"):
            raise ValueError("solver-weighted feature fusion cannot use test split pair caches")


def solver_weighted_fusion_from_pair_cache(
    pair_cache: dict[str, Any] | str | Path,
    *,
    trust_alpha: float = 0.1,
    reprojection_threshold_px: float = 4.0,
    min_cosine: float = -1.0,
    min_observations_per_landmark: int = 1,
    min_native_cosine: float = 0.95,
    disallow_test_split: bool = True,
) -> dict[str, Any]:
    """Build solver-weighted fused descriptors from a listwise SceneMatch pair cache."""

    payload = _load_pair_cache(pair_cache)
    metadata = dict(payload.get("metadata", {}))
    if bool(disallow_test_split):
        _reject_test_split(metadata)
    base = F.normalize(torch.as_tensor(payload["base_landmark_desc"], dtype=torch.float32).cpu(), p=2, dim=-1)
    query_desc = torch.as_tensor(payload["query_desc"], dtype=torch.float32).cpu()
    landmark_ids = torch.as_tensor(payload["landmark_id"], dtype=torch.long).cpu()
    if query_desc.dim() != 2 or landmark_ids.dim() != 2:
        raise ValueError("pair cache query_desc must be [N,D] and landmark_id must be [N,K]")
    if query_desc.shape[0] != landmark_ids.shape[0]:
        raise ValueError("query_desc rows must match landmark_id rows")
    rows, topk = int(landmark_ids.shape[0]), int(landmark_ids.shape[1])
    mask = torch.ones(rows, topk, dtype=torch.bool)
    if "candidate_mask" in payload:
        mask = torch.as_tensor(payload["candidate_mask"], dtype=torch.bool).cpu()
        if mask.shape != landmark_ids.shape:
            raise ValueError("candidate_mask must match landmark_id")
    cosine = torch.ones(rows, topk, dtype=torch.float32)
    if "cosine" in payload:
        cosine = torch.as_tensor(payload["cosine"], dtype=torch.float32).cpu()
        if cosine.shape != landmark_ids.shape:
            raise ValueError("cosine must match landmark_id")
    reproj = torch.zeros(rows, topk, dtype=torch.float32)
    if "reprojection_error" in payload:
        reproj = torch.as_tensor(payload["reprojection_error"], dtype=torch.float32).cpu()
        if reproj.shape != landmark_ids.shape:
            raise ValueError("reprojection_error must match landmark_id")
        mask = mask & torch.isfinite(reproj) & (reproj <= float(reprojection_threshold_px))
    mask = mask & (cosine >= float(min_cosine))
    row_ids, col_ids = torch.where(mask)
    if row_ids.numel() == 0:
        fused = base.clone()
        positive_count = 0
        observation_counts = torch.zeros(base.shape[0], dtype=torch.long)
    else:
        selected_landmarks = landmark_ids[row_ids, col_ids]
        observed = query_desc[row_ids]
        reproj_weight = torch.exp(-reproj[row_ids, col_ids].clamp_min(0.0) / max(float(reprojection_threshold_px), 1e-6))
        solver_weight = cosine[row_ids, col_ids].clamp_min(0.0) * reproj_weight
        fused, _ = solver_weighted_landmark_feature_fusion(
            native_descriptors=base,
            landmark_ids=selected_landmarks,
            observed_descriptors=observed,
            solver_weights=solver_weight,
            trust_alpha=float(trust_alpha),
            min_total_weight=1e-8,
        )
        positive_count = int(row_ids.numel())
        observation_counts = torch.bincount(selected_landmarks.reshape(-1), minlength=base.shape[0]).long()
    active = observation_counts > 0
    min_obs = max(1, int(min_observations_per_landmark))
    eligible = observation_counts >= min_obs
    fused = F.normalize(fused, p=2, dim=-1)
    native_cosine = (fused * base).sum(dim=-1).clamp(-1.0, 1.0)
    min_shift_cosine = min(max(float(min_native_cosine), -1.0), 1.0)
    shifted_too_far = eligible & (native_cosine < min_shift_cosine)
    keep = eligible & ~shifted_too_far
    guarded = base.clone()
    guarded[keep] = fused[keep]
    changed = keep & ((guarded - base).abs().sum(dim=-1) > 1e-7)
    gaussian_ids = torch.as_tensor(
        payload.get("base_gaussian_id", torch.arange(base.shape[0], dtype=torch.long)),
        dtype=torch.long,
    ).reshape(-1).cpu()
    if gaussian_ids.numel() != base.shape[0]:
        raise ValueError("base_gaussian_id must match base_landmark_desc rows")
    return {
        "descriptors": guarded,
        "base_descriptors": base,
        "gaussian_ids": gaussian_ids,
        "metadata": {
            "descriptor_mode": "solver_weighted_pair_cache_fusion",
            "base_landmark_count": int(base.shape[0]),
            "descriptor_dim": int(base.shape[1]),
            "positive_pair_count": int(positive_count),
            "active_landmark_count": int(active.sum().item()),
            "eligible_landmark_count": int(eligible.sum().item()),
            "updated_landmark_count": int(changed.sum().item()),
            "reverted_low_observation_count": int((active & ~eligible).sum().item()),
            "reverted_shift_count": int(shifted_too_far.sum().item()),
            "trust_alpha": float(trust_alpha),
            "reprojection_threshold_px": float(reprojection_threshold_px),
            "min_cosine": float(min_cosine),
            "min_observations_per_landmark": int(min_obs),
            "min_native_cosine": float(min_shift_cosine),
            "native_cosine_mean": float(native_cosine[active].mean().item()) if bool(active.any()) else 1.0,
            "native_cosine_min": float(native_cosine[active].min().item()) if bool(active.any()) else 1.0,
            "source_split_name": str(metadata.get("source_split_name", "")),
            "feedback_bank_split_name": str(metadata.get("feedback_bank_split_name", "")),
        },
    }
