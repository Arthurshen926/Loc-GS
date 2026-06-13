from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import torch


SCHEMA_VERSION = "ulfloc_solver_feedback_v1"
POSITIVE_ROLES = {"protected_support", "positive_inlier"}
DEFAULT_NATIVE_FUSION_NEGATIVE_ROLES = {"harmful_negative"}
KNOWN_NEGATIVE_ROLES = {"harmful_negative", "risky_competitor_negative"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if number == number else float(default)


def _audit(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    audit = payload.get("split_audit", {})
    return audit if isinstance(audit, Mapping) else {}


def _split_name(payload: Mapping[str, Any]) -> str:
    split = str(payload.get("split_name", payload.get("split", ""))).strip()
    if not split:
        audit = _audit(payload)
        split = str(audit.get("split_name", audit.get("split", ""))).strip()
    if not split:
        raise ValueError("split_name is required")
    return split


def _validate_split(payload: Mapping[str, Any], split: str) -> None:
    audit = _audit(payload)
    if split.lower() == "test":
        raise ValueError("refusing to build ULF native fusion feedback from test split")
    if bool(audit.get("test_split_used", False)) or bool(audit.get("official_test_used", False)):
        raise ValueError("refusing to build ULF native fusion feedback from test split")
    audit_split = str(audit.get("split_name", audit.get("split", ""))).strip()
    if audit_split.lower() == "test":
        raise ValueError("refusing to build ULF native fusion feedback from test split")
    if audit_split and audit_split != split:
        raise ValueError(f"split_name mismatch: expected {split}, got audit split {audit_split}")


def _positive_score(row: Mapping[str, Any]) -> float:
    margin = max(0.0, _float(row.get("descriptor_margin"), 0.0))
    score = max(0.0, _float(row.get("descriptor_score"), 0.0))
    reproj = max(0.0, _float(row.get("reprojection_error_px"), 0.0))
    return float((1.0 + min(1.0, score) + min(1.0, margin)) / (1.0 + reproj / 4.0))


def _negative_score(row: Mapping[str, Any]) -> float:
    score = max(0.0, _float(row.get("descriptor_score"), 0.0))
    margin = max(0.0, _float(row.get("descriptor_margin"), 0.0))
    reproj = max(0.0, _float(row.get("reprojection_error_px"), 0.0))
    low_margin = 1.0 / (1.0 + margin / 0.15)
    return float((0.25 + min(1.0, score)) * min(2.0, reproj / 8.0) * low_margin)


def _normalize(values: dict[int, float]) -> dict[int, float]:
    max_value = max(values.values(), default=0.0)
    if max_value <= 0.0:
        return {}
    return {gid: float(value) / float(max_value) for gid, value in values.items() if value > 0.0}


def _weights_from_pos_neg(
    positive: dict[int, float],
    negative: dict[int, float],
    *,
    num_gaussians: int,
    alpha: float,
    risk_penalty: float,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor:
    count = int(num_gaussians)
    weights = torch.ones((count,), dtype=torch.float32)
    pos_norm = _normalize(positive)
    neg_norm = _normalize(negative)
    for gid in sorted(set(pos_norm) | set(neg_norm)):
        if gid < 0 or gid >= count:
            continue
        signal = float(pos_norm.get(gid, 0.0)) - float(risk_penalty) * float(neg_norm.get(gid, 0.0))
        weights[gid] = float(1.0 + float(alpha) * signal)
    return weights.clamp(float(min_weight), float(max_weight))


def _view_weights(
    view_positive: Mapping[str, dict[int, float]],
    view_negative: Mapping[str, dict[int, float]],
    *,
    num_gaussians: int,
    alpha: float,
    risk_penalty: float,
    min_weight: float,
    max_weight: float,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    entry_count = 0
    negative_entry_count = 0
    for view_id in sorted(set(view_positive) | set(view_negative)):
        weights = _weights_from_pos_neg(
            view_positive.get(view_id, {}),
            view_negative.get(view_id, {}),
            num_gaussians=num_gaussians,
            alpha=alpha,
            risk_penalty=risk_penalty,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        changed = torch.where(torch.abs(weights - 1.0) > 1e-6)[0].to(torch.long)
        if changed.numel() == 0:
            continue
        out[str(view_id)] = {
            "landmark_ids": changed,
            "weights": weights[changed].to(torch.float32),
        }
        entry_count += int(changed.numel())
        negative_entry_count += int((weights[changed] < 1.0).sum().item())
    return out, {
        "view_count": int(len(out)),
        "entry_count": int(entry_count),
        "negative_entry_count": int(negative_entry_count),
        "min_weight": float(min_weight),
        "max_weight": float(max_weight),
    }


def _view_pruning_weights(
    view_negative: Mapping[str, dict[int, float]],
    *,
    negative_threshold: float,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    entry_count = 0
    for view_id, rows in sorted(view_negative.items()):
        ids = sorted(int(gid) for gid, value in rows.items() if float(value) > float(negative_threshold))
        if not ids:
            continue
        out[str(view_id)] = {
            "landmark_ids": torch.as_tensor(ids, dtype=torch.long),
            "weights": torch.zeros((len(ids),), dtype=torch.float32),
        }
        entry_count += len(ids)
    return out, {
        "view_count": int(len(out)),
        "entry_count": int(entry_count),
        "negative_entry_count": int(entry_count),
        "min_weight": 0.0,
        "max_weight": 1.0,
        "negative_threshold": float(negative_threshold),
    }


def build_ulfloc_native_fusion_feedback_from_v4(
    feedback_v4: Mapping[str, Any],
    *,
    scene: str,
    num_gaussians: int,
    alpha: float = 0.5,
    risk_penalty: float = 1.0,
    min_weight: float = 0.25,
    max_weight: float = 1.75,
    fusion_policy: str = "soft_weight",
    pruning_negative_threshold: float = 0.0,
    negative_roles: tuple[str, ...] | list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Build ULF-compatible feedback consumed by native train-view feature fusion.

    The returned payload is meant for ULF-Loc's ``utils.gsfeature_fusion``. It
    preserves the fixed landmark set and changes only the per-landmark/per-view
    aggregation weights used while sampling descriptors from original train
    images.
    """

    count = int(num_gaussians)
    if count <= 0:
        raise ValueError(f"num_gaussians must be positive, got {num_gaussians}")
    split = _split_name(feedback_v4)
    _validate_split(feedback_v4, split)

    positive: dict[int, float] = defaultdict(float)
    negative: dict[int, float] = defaultdict(float)
    view_positive: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    view_negative: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    neutral_count = 0
    positive_count = 0
    negative_count = 0
    ignored_negative_count = 0
    active_negative_roles = (
        {str(role).strip() for role in negative_roles if str(role).strip()}
        if negative_roles is not None
        else set(DEFAULT_NATIVE_FUSION_NEGATIVE_ROLES)
    )
    unknown_negative_roles = sorted(active_negative_roles - KNOWN_NEGATIVE_ROLES)
    if unknown_negative_roles:
        raise ValueError(f"unknown negative_roles: {unknown_negative_roles}")
    for raw in feedback_v4.get("correspondences", []):
        if not isinstance(raw, Mapping):
            continue
        row_split = str(raw.get("split_name", split)).strip()
        if row_split.lower() == "test":
            raise ValueError("refusing to build ULF native fusion feedback from test split")
        if row_split and row_split != split:
            raise ValueError(f"split_name mismatch: expected {split}, got row split {row_split}")
        try:
            gid = int(raw.get("gaussian_id", raw.get("matched_gaussian_id")))
        except (TypeError, ValueError):
            continue
        if gid < 0 or gid >= count:
            continue
        view_id = str(raw.get("image_id", raw.get("query_id", raw.get("source_view_id", "")))).strip()
        role = str(raw.get("label_role", "")).strip()
        if role in POSITIVE_ROLES:
            value = _positive_score(raw)
            positive[gid] += value
            if view_id:
                view_positive[view_id][gid] += value
            positive_count += 1
        elif role in active_negative_roles:
            value = _negative_score(raw)
            negative[gid] += value
            if view_id:
                view_negative[view_id][gid] += value
            negative_count += 1
        elif role in KNOWN_NEGATIVE_ROLES:
            ignored_negative_count += 1
        else:
            neutral_count += 1

    policy = str(fusion_policy).strip().lower()
    if policy not in {"soft_weight", "view_pruning"}:
        raise ValueError(f"unsupported fusion_policy: {fusion_policy}")
    if policy == "view_pruning":
        weights = torch.ones((count,), dtype=torch.float32)
        view_landmark_weights, view_summary = _view_pruning_weights(
            view_negative,
            negative_threshold=float(pruning_negative_threshold),
        )
        weight_application = "absolute"
    else:
        weights = _weights_from_pos_neg(
            positive,
            negative,
            num_gaussians=count,
            alpha=float(alpha),
            risk_penalty=float(risk_penalty),
            min_weight=float(min_weight),
            max_weight=float(max_weight),
        )
        view_landmark_weights, view_summary = _view_weights(
            view_positive,
            view_negative,
            num_gaussians=count,
            alpha=float(alpha),
            risk_penalty=float(risk_penalty),
            min_weight=float(min_weight),
            max_weight=float(max_weight),
        )
        weight_application = "blend"
    observed_count = torch.zeros((count,), dtype=torch.long)
    for gid in set(positive) | set(negative):
        if 0 <= gid < count:
            observed_count[gid] = 1

    return {
        "schema_version": SCHEMA_VERSION,
        "scene": str(scene),
        "split_name": split,
        "num_landmarks": count,
        "landmark_weights": weights,
        "view_landmark_weights": view_landmark_weights,
        "observed_count": observed_count,
        "metadata": {
            "source_schema_version": str(feedback_v4.get("schema_version", "")),
            "positive_count": int(positive_count),
            "negative_count": int(negative_count),
            "ignored_negative_count": int(ignored_negative_count),
            "neutral_outlier_count": int(neutral_count),
            "positive_landmark_count": int(len(positive)),
            "negative_landmark_count": int(len(negative)),
            "hyperparameters": {
                "alpha": float(alpha),
                "risk_penalty": float(risk_penalty),
                "min_weight": float(min_weight),
                "max_weight": float(max_weight),
                "fusion_policy": policy,
                "pruning_negative_threshold": float(pruning_negative_threshold),
                "negative_roles": sorted(active_negative_roles),
            },
            "fusion_policy": policy,
            "weight_application": weight_application,
            "view_landmark_weight_summary": view_summary,
            "role": "native_train_view_feature_fusion_feedback",
        },
    }
