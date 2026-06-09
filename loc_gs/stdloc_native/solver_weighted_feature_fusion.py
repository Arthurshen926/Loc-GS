from __future__ import annotations

import json
import math
from typing import Any, Mapping
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


def _load_artifact(artifact: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(artifact, (str, Path)):
        payload = torch.load(Path(artifact), map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("descriptor artifact must contain a dict")
        return payload
    return dict(artifact)


def _load_validation_profile(profile: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(profile, (str, Path)):
        payload = json.loads(Path(profile).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("validation profile must contain a JSON object")
        return payload
    return dict(profile)


def _load_active_fusion_plan(plan: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(plan, (str, Path)):
        payload = json.loads(Path(plan).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("active fusion plan must contain a JSON object")
        return payload
    return dict(plan)


def _reject_test_split(metadata: dict[str, Any]) -> None:
    for key in ("source_split_name", "feedback_bank_split_name", "split_name"):
        value = str(metadata.get(key, "")).lower()
        if value == "test" or value.endswith("_test"):
            raise ValueError("solver-weighted feature fusion cannot use test split pair caches")


def _reject_test_profile(profile: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    for source in (profile, metadata):
        for key in ("split_name", "split", "source_split_name", "feedback_bank_split_name"):
            value = str(source.get(key, "")).strip().lower()
            if value == "test" or value.endswith("_test"):
                raise ValueError("test split validation/profile data is not allowed for guarded descriptor fusion")


def _reject_test_active_plan(plan: Mapping[str, Any]) -> None:
    for key in ("split_name", "split", "source_split_name", "feedback_bank_split_name"):
        value = str(plan.get(key, "")).strip().lower()
        if value == "test" or value.endswith("_test"):
            raise ValueError("test split active fusion plan is not allowed for descriptor fusion")


def _active_plan_lookup(plan: Mapping[str, Any]) -> dict[int, set[str]]:
    raw_plan = plan.get("landmark_fusion_plan", plan.get("descriptor_fusion", {}).get("landmark_fusion_plan", {}))
    if not isinstance(raw_plan, Mapping):
        raise ValueError("active fusion plan must contain landmark_fusion_plan")
    lookup: dict[int, set[str]] = {}
    for raw_gid, raw_row in raw_plan.items():
        try:
            gid = int(raw_gid)
        except (TypeError, ValueError):
            continue
        if gid < 0 or not isinstance(raw_row, Mapping):
            continue
        views = raw_row.get("selected_view_ids", raw_row.get("view_ids", []))
        if isinstance(views, str):
            selected = {views} if views.strip() else set()
        elif isinstance(views, (list, tuple, set)):
            selected = {str(view) for view in views if str(view).strip()}
        else:
            selected = set()
        if selected:
            lookup[gid] = selected
    return lookup


def _active_plan_candidate_mask(
    *,
    payload: Mapping[str, Any],
    landmark_ids: torch.Tensor,
    gaussian_ids: torch.Tensor,
    active_fusion_plan: Mapping[str, Any] | str | Path,
    disallow_test_split: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    plan = _load_active_fusion_plan(active_fusion_plan)
    if bool(disallow_test_split):
        _reject_test_active_plan(plan)
    lookup = _active_plan_lookup(plan)
    rows, topk = int(landmark_ids.shape[0]), int(landmark_ids.shape[1])
    raw_view_ids = payload.get("row_source_view_id", payload.get("row_query_id", payload.get("query_image_name")))
    if raw_view_ids is None:
        raise ValueError("pair cache must contain row_source_view_id when active_fusion_plan is used")
    row_view_ids = [str(item) for item in raw_view_ids]
    if len(row_view_ids) != rows:
        raise ValueError("row_source_view_id length must match pair cache rows")
    active_mask = torch.zeros(rows, topk, dtype=torch.bool)
    valid_landmark = (landmark_ids >= 0) & (landmark_ids < int(gaussian_ids.numel()))
    for row in range(rows):
        view_id = row_view_ids[row]
        for col in range(topk):
            if not bool(valid_landmark[row, col]):
                continue
            gid = int(gaussian_ids[int(landmark_ids[row, col])].item())
            allowed_views = lookup.get(gid)
            if allowed_views and view_id in allowed_views:
                active_mask[row, col] = True
    metadata = {
        "active_fusion_plan_enabled": True,
        "active_fusion_plan_landmark_count": int(len(lookup)),
        "active_fusion_plan_positive_pair_count": int(active_mask.sum().item()),
        "active_fusion_plan_split_name": str(plan.get("split_name", plan.get("split", ""))),
    }
    return active_mask, metadata


def _profile_delta_cm(row: Mapping[str, Any]) -> float | None:
    for key in ("delta_sparse_te_cm", "delta_te_cm", "candidate_minus_baseline_te_cm"):
        value = row.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return float(number)
    return None


def _profile_landmark_scalar_table(raw: object, *, gaussian_to_row: Mapping[int, int], landmark_count: int) -> torch.Tensor:
    out = torch.zeros(int(landmark_count), dtype=torch.float32)
    if not isinstance(raw, Mapping):
        return out
    for raw_gid, raw_value in raw.items():
        try:
            gid = int(raw_gid)
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value) or value <= 0.0:
            continue
        row = gaussian_to_row.get(gid)
        if row is None and 0 <= gid < int(landmark_count):
            row = int(gid)
        if row is None or not (0 <= int(row) < int(landmark_count)):
            continue
        out[int(row)] = max(float(out[int(row)].item()), float(value))
    return out


def _profile_query_support_to_landmark_scores(
    raw: object,
    *,
    gaussian_to_row: Mapping[int, int],
    landmark_count: int,
) -> torch.Tensor:
    out = torch.zeros(int(landmark_count), dtype=torch.float32)
    if not isinstance(raw, Mapping):
        return out
    for raw_query_map in raw.values():
        if not isinstance(raw_query_map, Mapping):
            continue
        values = _profile_landmark_scalar_table(
            raw_query_map,
            gaussian_to_row=gaussian_to_row,
            landmark_count=landmark_count,
        )
        out += values
    return out


def _query_positive_landmark_weights(
    *,
    landmark_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
    cosine: torch.Tensor,
    reprojection_error: torch.Tensor,
    start: int,
    stop: int,
    reprojection_threshold_px: float,
    min_cosine: float,
    landmark_count: int,
) -> torch.Tensor:
    ids = landmark_ids[start:stop]
    mask = candidate_mask[start:stop].bool()
    scores = cosine[start:stop].float()
    reproj = reprojection_error[start:stop].float()
    selected = mask & torch.isfinite(scores) & torch.isfinite(reproj)
    selected = selected & (scores >= float(min_cosine)) & (reproj <= float(reprojection_threshold_px))
    if not bool(selected.any()):
        return torch.zeros(landmark_count, dtype=torch.float32)
    selected_ids = ids[selected].reshape(-1).long()
    if selected_ids.numel() == 0:
        return torch.zeros(landmark_count, dtype=torch.float32)
    selected_scores = scores[selected].clamp_min(0.0).reshape(-1)
    selected_reproj = reproj[selected].clamp_min(0.0).reshape(-1)
    reproj_scale = max(float(reprojection_threshold_px), 1e-6)
    weights = selected_scores * torch.exp(-selected_reproj / reproj_scale)
    valid = (selected_ids >= 0) & (selected_ids < int(landmark_count)) & torch.isfinite(weights) & (weights > 0.0)
    out = torch.zeros(landmark_count, dtype=torch.float32)
    if bool(valid.any()):
        out.scatter_add_(0, selected_ids[valid], weights[valid])
    return out


def solver_weighted_fusion_from_pair_cache(
    pair_cache: dict[str, Any] | str | Path,
    *,
    trust_alpha: float = 0.1,
    reprojection_threshold_px: float = 4.0,
    min_cosine: float = -1.0,
    min_observations_per_landmark: int = 1,
    min_native_cosine: float = 0.95,
    disallow_test_split: bool = True,
    active_fusion_plan: Mapping[str, Any] | str | Path | None = None,
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
    gaussian_ids = torch.as_tensor(
        payload.get("base_gaussian_id", torch.arange(base.shape[0], dtype=torch.long)),
        dtype=torch.long,
    ).reshape(-1).cpu()
    if gaussian_ids.numel() != base.shape[0]:
        raise ValueError("base_gaussian_id must match base_landmark_desc rows")
    active_metadata: dict[str, Any] = {
        "active_fusion_plan_enabled": False,
        "active_fusion_plan_landmark_count": 0,
        "active_fusion_plan_positive_pair_count": 0,
    }
    if active_fusion_plan is not None:
        active_mask, active_metadata = _active_plan_candidate_mask(
            payload=payload,
            landmark_ids=landmark_ids,
            gaussian_ids=gaussian_ids,
            active_fusion_plan=active_fusion_plan,
            disallow_test_split=bool(disallow_test_split),
        )
        mask = mask & active_mask
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
    source_split = str(metadata.get("source_split_name", metadata.get("split_name", "")))
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
            "source_split_name": source_split,
            "feedback_bank_split_name": str(metadata.get("feedback_bank_split_name", "")),
            **active_metadata,
        },
    }


def guard_solver_weighted_fusion_from_validation_profile(
    descriptor_artifact: dict[str, Any] | str | Path,
    pair_cache: dict[str, Any] | str | Path,
    validation_profile: Mapping[str, Any] | str | Path,
    *,
    regression_margin_cm: float = 20.0,
    improvement_margin_cm: float = 20.0,
    reprojection_threshold_px: float | None = None,
    min_cosine: float = -1.0,
    min_regression_support: float = 1e-6,
    improvement_credit_ratio: float = 1.0,
    guard_query_scope: str = "all_regressions",
    disallow_test_split: bool = True,
) -> dict[str, Any]:
    """Revert descriptor updates implicated by real sparse-PnP regressions.

    This is an offline map/log artifact guard.  It uses only paired validation
    rows from a non-test split and never changes per-query inference behavior.
    Query-to-cache alignment follows the pair-cache generation order: each
    processed image contributes a fixed number of sparse keypoint rows.
    """

    artifact = _load_artifact(descriptor_artifact)
    payload = _load_pair_cache(pair_cache)
    profile = _load_validation_profile(validation_profile)
    metadata = dict(payload.get("metadata", {}))
    if bool(disallow_test_split):
        _reject_test_profile(profile, metadata)
    scope = str(guard_query_scope).strip().lower()
    if scope not in {"all_regressions", "protected_regressions"}:
        raise ValueError("guard_query_scope must be 'all_regressions' or 'protected_regressions'")

    descriptors = F.normalize(torch.as_tensor(artifact["descriptors"], dtype=torch.float32).cpu(), p=2, dim=-1)
    base = F.normalize(torch.as_tensor(artifact.get("base_descriptors"), dtype=torch.float32).cpu(), p=2, dim=-1)
    if descriptors.shape != base.shape or descriptors.dim() != 2:
        raise ValueError("descriptor artifact must contain descriptors/base_descriptors with identical [N,D] shape")
    landmark_count = int(descriptors.shape[0])
    gaussian_ids = torch.as_tensor(
        artifact.get("gaussian_ids", torch.arange(landmark_count, dtype=torch.long)),
        dtype=torch.long,
    ).reshape(-1).cpu()
    if int(gaussian_ids.numel()) != landmark_count:
        raise ValueError("descriptor artifact gaussian_ids must match descriptors")
    gaussian_to_row = {int(gid): int(index) for index, gid in enumerate(gaussian_ids.tolist())}

    direct_risk = _profile_landmark_scalar_table(
        profile.get("landmark_regression_risk", {}),
        gaussian_to_row=gaussian_to_row,
        landmark_count=landmark_count,
    )
    direct_benefit = _profile_query_support_to_landmark_scores(
        profile.get("protected_per_query_support", {}),
        gaussian_to_row=gaussian_to_row,
        landmark_count=landmark_count,
    )
    direct_benefit += _profile_query_support_to_landmark_scores(
        profile.get("validated_per_query_support", {}),
        gaussian_to_row=gaussian_to_row,
        landmark_count=landmark_count,
    )

    landmark_ids = torch.as_tensor(payload["landmark_id"], dtype=torch.long).cpu()
    if landmark_ids.dim() != 2:
        raise ValueError("pair cache landmark_id must be [rows, topk]")
    row_count, topk = int(landmark_ids.shape[0]), int(landmark_ids.shape[1])
    candidate_mask = torch.ones(row_count, topk, dtype=torch.bool)
    if "candidate_mask" in payload:
        candidate_mask = torch.as_tensor(payload["candidate_mask"], dtype=torch.bool).cpu()
    cosine = torch.ones(row_count, topk, dtype=torch.float32)
    if "cosine" in payload:
        cosine = torch.as_tensor(payload["cosine"], dtype=torch.float32).cpu()
    reproj = torch.zeros(row_count, topk, dtype=torch.float32)
    if "reprojection_error" in payload:
        reproj = torch.as_tensor(payload["reprojection_error"], dtype=torch.float32).cpu()
    if candidate_mask.shape != landmark_ids.shape or cosine.shape != landmark_ids.shape or reproj.shape != landmark_ids.shape:
        raise ValueError("pair cache candidate tensors must match landmark_id shape")

    paired_queries = profile.get("paired_queries")
    has_direct_attribution = bool((direct_risk > 0).any() or (direct_benefit > 0).any())
    if (not isinstance(paired_queries, list) or not paired_queries) and not has_direct_attribution:
        raise ValueError("validation profile must contain a non-empty paired_queries list")
    paired_query_count = len(paired_queries) if isinstance(paired_queries, list) else 0
    processed = int(metadata.get("processed_images", max(1, paired_query_count)))
    if processed <= 0:
        raise ValueError("pair cache metadata processed_images must be positive")
    if row_count % processed != 0:
        raise ValueError("pair cache rows must divide evenly by processed_images")
    if isinstance(paired_queries, list) and len(paired_queries) > processed:
        raise ValueError("validation profile has more paired queries than pair cache processed images")
    rows_per_query = row_count // processed
    threshold = (
        float(reprojection_threshold_px)
        if reprojection_threshold_px is not None
        else float(metadata.get("reprojection_threshold_px", 4.0))
    )

    risk = direct_risk.clone()
    benefit = direct_benefit.clone()
    regression_query_count = 0
    improvement_query_count = 0
    used_regression_pair_count = 0
    used_improvement_pair_count = 0
    for query_index, raw_row in enumerate(paired_queries if isinstance(paired_queries, list) else []):
        if not isinstance(raw_row, Mapping):
            continue
        delta = _profile_delta_cm(raw_row)
        if delta is None:
            continue
        is_regression = delta >= float(regression_margin_cm)
        is_improvement = -delta >= float(improvement_margin_cm)
        if is_regression and scope == "protected_regressions" and not bool(raw_row.get("protected_by_baseline", False)):
            is_regression = False
        if not is_regression and not is_improvement:
            continue
        start = int(query_index) * rows_per_query
        stop = start + rows_per_query
        weights = _query_positive_landmark_weights(
            landmark_ids=landmark_ids,
            candidate_mask=candidate_mask,
            cosine=cosine,
            reprojection_error=reproj,
            start=start,
            stop=stop,
            reprojection_threshold_px=threshold,
            min_cosine=float(min_cosine),
            landmark_count=landmark_count,
        )
        support_count = int((weights > 0).sum().item())
        if support_count == 0:
            continue
        if is_regression:
            risk += weights * float(delta)
            regression_query_count += 1
            used_regression_pair_count += support_count
        if is_improvement:
            benefit += weights * float(-delta)
            improvement_query_count += 1
            used_improvement_pair_count += support_count

    changed = ((descriptors - base).abs().sum(dim=-1) > 1e-7)
    guard = changed & (risk >= float(min_regression_support)) & (risk > benefit * float(improvement_credit_ratio))
    guarded = descriptors.clone()
    guarded[guard] = base[guard]
    native_cosine = (guarded * base).sum(dim=-1).clamp(-1.0, 1.0)
    parent_metadata = dict(artifact.get("metadata", {}))
    source_split = str(parent_metadata.get("source_split_name", metadata.get("split_name", "")))
    out_metadata = {
        **parent_metadata,
        "descriptor_mode": "solver_weighted_pair_cache_fusion_guarded_by_sparse_pnp_validation",
        "parent_descriptor_mode": str(parent_metadata.get("descriptor_mode", "")),
        "guarded_landmark_count": int(guard.sum().item()),
        "guard_candidate_landmark_count": int(((risk >= float(min_regression_support)) & changed).sum().item()),
        "changed_landmark_count_before_guard": int(changed.sum().item()),
        "changed_landmark_count_after_guard": int(((guarded - base).abs().sum(dim=-1).gt(1e-7)).sum().item()),
        "regression_query_count": int(regression_query_count),
        "improvement_query_count": int(improvement_query_count),
        "used_regression_landmark_support_count": int(used_regression_pair_count),
        "used_improvement_landmark_support_count": int(used_improvement_pair_count),
        "direct_profile_regression_risk_landmark_count": int((direct_risk > 0).sum().item()),
        "direct_profile_support_credit_landmark_count": int((direct_benefit > 0).sum().item()),
        "validation_profile_attribution_status": str(profile.get("attribution_status", "unknown")),
        "regression_margin_cm": float(regression_margin_cm),
        "improvement_margin_cm": float(improvement_margin_cm),
        "improvement_credit_ratio": float(improvement_credit_ratio),
        "guard_query_scope": scope,
        "min_regression_support": float(min_regression_support),
        "guard_reprojection_threshold_px": float(threshold),
        "guard_min_cosine": float(min_cosine),
        "validation_profile_query_count": int(len(paired_queries)) if isinstance(paired_queries, list) else 0,
        "pair_cache_processed_images": int(processed),
        "pair_cache_rows_per_query": int(rows_per_query),
        "source_split_name": source_split,
        "native_cosine_mean_after_guard": float(native_cosine.mean().item()) if native_cosine.numel() else 1.0,
        "native_cosine_min_after_guard": float(native_cosine.min().item()) if native_cosine.numel() else 1.0,
        "official_test_used": False,
        "branch_selection": False,
    }
    return {
        "descriptors": guarded,
        "base_descriptors": base,
        "gaussian_ids": gaussian_ids,
        "metadata": out_metadata,
    }
