from __future__ import annotations

import math
from typing import Any, Mapping

import torch


SCHEMA_VERSION = "ulfloc_solver_feedback_detector_targets_v1"


def _split_name(payload: Mapping[str, Any]) -> str:
    split = str(payload.get("split_name", payload.get("split", ""))).strip()
    split_audit = payload.get("split_audit")
    if not split and isinstance(split_audit, Mapping):
        split = str(split_audit.get("split_name", split_audit.get("split", ""))).strip()
    return split


def _reject_test_split(payload: Mapping[str, Any], *, label: str) -> str:
    split = _split_name(payload)
    split_audit = payload.get("split_audit")
    audit_split = ""
    test_split_used = False
    if isinstance(split_audit, Mapping):
        audit_split = str(split_audit.get("split_name", "")).strip()
        test_split_used = bool(split_audit.get("test_split_used", False))
    def is_test_alias(value: str) -> bool:
        lowered = str(value).strip().lower()
        return lowered == "test" or lowered == "official_test" or lowered.endswith("_test")

    if is_test_alias(split) or is_test_alias(audit_split) or test_split_used:
        raise ValueError(f"refusing to build ULF-Loc solver-feedback detector targets from test split {label}")
    return split


def validate_detector_target_inputs(
    teacher_payload: Mapping[str, Any] | None,
    impact: Mapping[str, Any],
    sp_teacher_payload: Mapping[str, Any] | None = None,
) -> tuple[str, str, str]:
    teacher_split = _reject_test_split(teacher_payload, label="visibility teacher") if teacher_payload else ""
    sp_teacher_split = _reject_test_split(sp_teacher_payload, label="SuperPoint teacher") if sp_teacher_payload else ""
    impact_split = _reject_test_split(impact, label="impact attribution")
    known = [
        split
        for split in (teacher_split, sp_teacher_split, impact_split)
        if split and split.lower() != "unknown"
    ]
    if len(set(known)) > 1:
        raise ValueError(f"split_name mismatch between detector target inputs: {known}")
    split = known[0] if known else (teacher_split or impact_split or "unknown")
    return split, teacher_split or "unknown", impact_split or "unknown"


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _point_yx(row: Mapping[str, Any]) -> tuple[float, float] | None:
    for key in ("keypoint_yx", "projected_yx", "yx"):
        point = row.get(key)
        if torch.is_tensor(point) and point.numel() >= 2:
            flat = point.detach().cpu().reshape(-1)
            y = _finite_float(flat[0].item())
            x = _finite_float(flat[1].item())
            if y is not None and x is not None:
                return y, x
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            y = _finite_float(point[0])
            x = _finite_float(point[1])
            if y is not None and x is not None:
                return y, x
    point = row.get("keypoint_xy", row.get("projected_xy", row.get("xy")))
    if torch.is_tensor(point) and point.numel() >= 2:
        flat = point.detach().cpu().reshape(-1)
        x = _finite_float(flat[0].item())
        y = _finite_float(flat[1].item())
        if y is not None and x is not None:
            return y, x
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        x = _finite_float(point[0])
        y = _finite_float(point[1])
        if y is not None and x is not None:
            return y, x
    return None


def _int_value(value: Any, *, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _targets_mapping(payload_or_targets: Mapping[str, Any]) -> Mapping[str, Any]:
    targets = payload_or_targets.get("targets")
    if isinstance(targets, Mapping):
        return targets
    return payload_or_targets


def _as_point_tensor(value: Any) -> torch.Tensor:
    points = torch.as_tensor(value if value is not None else [], dtype=torch.float32)
    if points.numel() == 0:
        return torch.empty(0, 2, dtype=torch.float32)
    return points.reshape(-1, 2)


def _points_from_entry(
    entry: Mapping[str, Any],
    *,
    yx_keys: tuple[str, ...],
    xy_keys: tuple[str, ...] = (),
) -> torch.Tensor:
    for key in yx_keys:
        if key in entry:
            return _as_point_tensor(entry.get(key))
    for key in xy_keys:
        if key in entry:
            xy = _as_point_tensor(entry.get(key))
            if xy.numel() == 0:
                return xy
            return torch.stack((xy[:, 1], xy[:, 0]), dim=1).to(dtype=torch.float32)
    return torch.empty(0, 2, dtype=torch.float32)


def _weights_from_entry(
    entry: Mapping[str, Any],
    *,
    keys: tuple[str, ...],
    count: int,
) -> torch.Tensor:
    value = None
    for key in keys:
        if key in entry:
            value = entry.get(key)
            break
    weights = torch.as_tensor(value if value is not None else [], dtype=torch.float32).reshape(-1)
    if weights.numel() != int(count):
        weights = torch.ones(int(count), dtype=torch.float32)
    return weights.clamp_min(0.0)


def _teacher_entry(entry: Mapping[str, Any], *, height: int, width: int) -> tuple[dict[str, Any], dict[str, int | float]]:
    gids = torch.as_tensor(entry.get("gaussian_ids", []), dtype=torch.long).reshape(-1)
    yx = _as_point_tensor(entry.get("keypoint_yx", []))
    if yx.shape[0] != gids.numel():
        raise ValueError("visibility teacher keypoint_yx rows must match gaussian_ids")
    weights = torch.as_tensor(entry.get("support_weights", []), dtype=torch.float32).reshape(-1)
    if weights.numel() != gids.numel():
        weights = torch.ones(gids.numel(), dtype=torch.float32)
    weights = weights.clamp_min(0.0)
    out = {
        "gaussian_ids": gids.clone(),
        "keypoint_yx": yx.clone(),
        "support_weights": weights.clone(),
        "teacher_support_weights": weights.clone(),
        "positive_count": int(gids.numel()),
        "height": int(height),
        "width": int(width),
        "target_role": "superpoint_teacher_plus_visibility_plus_solver_residual",
    }
    return out, {
        "positive_count": int(gids.numel()),
        "positive_weight_sum": float(weights.sum().item()) if weights.numel() else 0.0,
    }


def _sp_teacher_entry(entry: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    yx = _points_from_entry(
        entry,
        yx_keys=("sp_teacher_keypoint_yx", "sp_keypoint_yx", "keypoint_yx", "yx"),
        xy_keys=("sp_teacher_keypoint_xy", "sp_keypoint_xy", "keypoint_xy", "xy"),
    )
    weights = _weights_from_entry(
        entry,
        keys=("sp_teacher_weights", "sp_keypoint_scores", "keypoint_scores", "scores", "support_weights"),
        count=int(yx.shape[0]),
    )
    return yx, weights


def compose_detector_targets_with_metrics(
    teacher_targets: Mapping[str, Mapping[str, Any]],
    impact: Mapping[str, Any],
    *,
    height: int,
    width: int,
    sp_teacher_targets: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError("height and width must be positive")
    teacher_payload = (
        teacher_targets
        if any(key in teacher_targets for key in ("split_name", "split", "split_audit", "targets"))
        else None
    )
    sp_teacher_payload = (
        sp_teacher_targets
        if isinstance(sp_teacher_targets, Mapping)
        and any(key in sp_teacher_targets for key in ("split_name", "split", "split_audit", "targets"))
        else None
    )
    validate_detector_target_inputs(teacher_payload, impact, sp_teacher_payload)

    targets = _targets_mapping(teacher_targets)
    sp_targets = _targets_mapping(sp_teacher_targets) if isinstance(sp_teacher_targets, Mapping) else {}
    detector_positive = impact.get("detector_positive", {})
    detector_negative = impact.get("detector_negative", {})
    if not isinstance(detector_positive, Mapping):
        detector_positive = {}
    if not isinstance(detector_negative, Mapping):
        detector_negative = {}

    metrics: dict[str, Any] = {
        "schema_version": "ulfloc_solver_feedback_detector_targets_metrics_v1",
        "height": int(height),
        "width": int(width),
        "target_role": "stdloc_visibility_teacher_with_solver_feedback_suppression",
        "detector_target_storage": "points",
        "image_count": 0,
        "positive_detector_point_count": 0,
        "visibility_positive_point_count": 0,
        "sp_teacher_point_count": 0,
        "solver_positive_detector_point_count": 0,
        "negative_suppression_point_count": 0,
        "sp_teacher_weight_sum": 0.0,
        "teacher_positive_weight_sum": 0.0,
        "negative_suppression_weight_sum": 0.0,
        "input_negative_count": 0,
        "invalid_negative_count": 0,
        "out_of_bounds_negative_count": 0,
        "impact_detector_positive_query_count": int(len(detector_positive)),
        "impact_detector_negative_query_count": int(len(detector_negative)),
    }
    composed: dict[str, dict[str, Any]] = {}
    query_ids = sorted(
        set(str(key) for key in targets)
        | set(str(key) for key in sp_targets)
        | set(str(key) for key in detector_positive)
        | set(str(key) for key in detector_negative)
    )

    for query_id in query_ids:
        raw_teacher = targets.get(query_id, {})
        if not isinstance(raw_teacher, Mapping):
            raw_teacher = {}
        entry, teacher_metrics = _teacher_entry(raw_teacher, height=int(height), width=int(width))
        raw_sp_teacher = sp_targets.get(query_id, {})
        if not isinstance(raw_sp_teacher, Mapping):
            raw_sp_teacher = {}
        sp_yx, sp_weights = _sp_teacher_entry(raw_sp_teacher)
        in_bounds = torch.ones(sp_yx.shape[0], dtype=torch.bool)
        if sp_yx.numel() > 0:
            in_bounds = (
                (sp_yx[:, 0] >= 0.0)
                & (sp_yx[:, 0] < float(height))
                & (sp_yx[:, 1] >= 0.0)
                & (sp_yx[:, 1] < float(width))
                & torch.isfinite(sp_yx).all(dim=1)
            )
            sp_yx = sp_yx[in_bounds]
            sp_weights = sp_weights[in_bounds]
        neg_yx: list[list[float]] = []
        neg_weights: list[float] = []
        neg_gids: list[int] = []
        solver_pos_yx: list[list[float]] = []
        solver_pos_weights: list[float] = []
        solver_pos_gids: list[int] = []
        positive_rows = detector_positive.get(query_id, [])
        if not isinstance(positive_rows, (list, tuple)):
            positive_rows = []
        for row in positive_rows:
            if not isinstance(row, Mapping):
                continue
            point = _point_yx(row)
            if point is None:
                continue
            y, x = point
            if not (0.0 <= y < float(height) and 0.0 <= x < float(width)):
                continue
            weight = _finite_float(row.get("weight", row.get("support", 1.0)))
            if weight is None:
                weight = 1.0
            weight = max(0.0, float(weight))
            if weight <= 0.0:
                continue
            solver_pos_yx.append([float(y), float(x)])
            solver_pos_weights.append(float(weight))
            solver_pos_gids.append(_int_value(row.get("gaussian_id", row.get("landmark_id", -1)), default=-1))
        rows = detector_negative.get(query_id, [])
        if not isinstance(rows, (list, tuple)):
            rows = []
        for row in rows:
            metrics["input_negative_count"] += 1
            if not isinstance(row, Mapping):
                metrics["invalid_negative_count"] += 1
                continue
            point = _point_yx(row)
            if point is None:
                metrics["invalid_negative_count"] += 1
                continue
            y, x = point
            if not (0.0 <= y < float(height) and 0.0 <= x < float(width)):
                metrics["out_of_bounds_negative_count"] += 1
                continue
            weight = _finite_float(row.get("weight", row.get("risk", 1.0)))
            if weight is None:
                weight = 1.0
            weight = max(0.0, float(weight))
            if weight <= 0.0:
                metrics["invalid_negative_count"] += 1
                continue
            neg_yx.append([float(y), float(x)])
            neg_weights.append(float(weight))
            neg_gids.append(_int_value(row.get("gaussian_id", row.get("landmark_id", -1)), default=-1))

        entry.update(
            {
                "sp_teacher_keypoint_yx": sp_yx.clone(),
                "sp_teacher_weights": sp_weights.clone(),
                "sp_teacher_count": int(sp_yx.shape[0]),
                "solver_positive_gaussian_ids": torch.tensor(solver_pos_gids, dtype=torch.long),
                "solver_positive_keypoint_yx": torch.tensor(solver_pos_yx, dtype=torch.float32).reshape(-1, 2),
                "solver_positive_weights": torch.tensor(solver_pos_weights, dtype=torch.float32),
                "solver_positive_count": int(len(solver_pos_gids)),
                "negative_gaussian_ids": torch.tensor(neg_gids, dtype=torch.long),
                "negative_keypoint_yx": torch.tensor(neg_yx, dtype=torch.float32).reshape(-1, 2),
                "negative_weights": torch.tensor(neg_weights, dtype=torch.float32),
                "negative_count": int(len(neg_gids)),
            }
        )
        composed[str(query_id)] = entry
        metrics["image_count"] += 1
        metrics["positive_detector_point_count"] += int(teacher_metrics["positive_count"])
        metrics["visibility_positive_point_count"] += int(teacher_metrics["positive_count"])
        metrics["sp_teacher_point_count"] += int(sp_yx.shape[0])
        metrics["negative_suppression_point_count"] += int(len(neg_gids))
        metrics["teacher_positive_weight_sum"] += float(teacher_metrics["positive_weight_sum"])
        metrics["sp_teacher_weight_sum"] += float(sp_weights.sum().item()) if sp_weights.numel() else 0.0
        metrics["solver_positive_detector_point_count"] += int(len(solver_pos_gids))
        metrics["negative_suppression_weight_sum"] += float(sum(neg_weights))

    metrics["teacher_positive_weight_sum"] = float(metrics["teacher_positive_weight_sum"])
    metrics["sp_teacher_weight_sum"] = float(metrics["sp_teacher_weight_sum"])
    metrics["negative_suppression_weight_sum"] = float(metrics["negative_suppression_weight_sum"])
    return composed, metrics


def compose_detector_targets(
    teacher_targets: Mapping[str, Mapping[str, Any]],
    impact: Mapping[str, Any],
    *,
    height: int,
    width: int,
    sp_teacher_targets: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    targets, _metrics = compose_detector_targets_with_metrics(
        teacher_targets,
        impact,
        height=int(height),
        width=int(width),
        sp_teacher_targets=sp_teacher_targets,
    )
    return targets
