from __future__ import annotations

import math
from typing import Any, Mapping

import torch


SCHEMA_VERSION = "ulfloc_superpoint_teacher_v1"


def validate_superpoint_teacher_split_name(split_name: str) -> str:
    split = str(split_name or "unknown").strip() or "unknown"
    lowered = split.lower()
    if lowered == "test" or lowered == "official_test" or lowered.endswith("_test"):
        raise ValueError("refusing to build ULF-Loc SuperPoint teacher from test split")
    return split


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _point_yx(row: Mapping[str, Any]) -> tuple[float, float] | None:
    point = row.get("keypoint_yx", row.get("yx"))
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
    point = row.get("keypoint_xy", row.get("xy"))
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


def _score(row: Mapping[str, Any]) -> float:
    for key in ("sp_teacher_weight", "keypoint_score", "score", "weight"):
        value = _finite_float(row.get(key))
        if value is not None:
            return max(0.0, float(value))
    return 1.0


def build_superpoint_teacher_targets_with_metrics(
    detections: Mapping[str, Any],
    *,
    height: int,
    width: int,
    split_name: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    split = validate_superpoint_teacher_split_name(split_name)
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError("height and width must be positive")
    metrics: dict[str, Any] = {
        "schema_version": "ulfloc_superpoint_teacher_metrics_v1",
        "split_name": split,
        "height": int(height),
        "width": int(width),
        "image_count": 0,
        "input_keypoint_count": 0,
        "kept_keypoint_count": 0,
        "invalid_keypoint_count": 0,
        "out_of_bounds_keypoint_count": 0,
        "score_sum": 0.0,
    }
    targets: dict[str, dict[str, Any]] = {}
    for image_name in sorted(str(key) for key in detections):
        rows = detections.get(image_name, [])
        if not isinstance(rows, (list, tuple)):
            rows = []
        yx_rows: list[list[float]] = []
        weights: list[float] = []
        for row in rows:
            metrics["input_keypoint_count"] += 1
            if not isinstance(row, Mapping):
                metrics["invalid_keypoint_count"] += 1
                continue
            point = _point_yx(row)
            if point is None:
                metrics["invalid_keypoint_count"] += 1
                continue
            y, x = point
            if not (0.0 <= y < float(height) and 0.0 <= x < float(width)):
                metrics["out_of_bounds_keypoint_count"] += 1
                continue
            weight = _score(row)
            if weight <= 0.0:
                metrics["invalid_keypoint_count"] += 1
                continue
            yx_rows.append([float(y), float(x)])
            weights.append(float(weight))
        yx = torch.tensor(yx_rows, dtype=torch.float32).reshape(-1, 2)
        score = torch.tensor(weights, dtype=torch.float32)
        targets[image_name] = {
            "sp_teacher_keypoint_yx": yx,
            "sp_teacher_weights": score,
            "sp_teacher_count": int(yx.shape[0]),
            "height": int(height),
            "width": int(width),
            "target_role": "superpoint_teacher",
        }
        metrics["image_count"] += 1
        metrics["kept_keypoint_count"] += int(yx.shape[0])
        metrics["score_sum"] += float(score.sum().item()) if score.numel() else 0.0
    metrics["score_sum"] = float(metrics["score_sum"])
    return targets, metrics


def build_superpoint_teacher_targets(
    detections: Mapping[str, Any],
    *,
    height: int,
    width: int,
    split_name: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    return build_superpoint_teacher_targets_with_metrics(
        detections,
        height=int(height),
        width=int(width),
        split_name=split_name,
    )
