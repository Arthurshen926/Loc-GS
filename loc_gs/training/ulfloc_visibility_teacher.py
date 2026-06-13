from __future__ import annotations

import math
from typing import Any, Mapping

import torch


SCHEMA_VERSION = "ulfloc_visibility_teacher_v1"


def validate_visibility_teacher_split_name(split_name: str) -> str:
    normalized = str(split_name).strip()
    if not normalized:
        raise ValueError("split_name is required")
    if normalized.lower() == "test":
        raise ValueError("test split is not allowed for ULF-Loc visibility teacher targets")
    return normalized


def _bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
        return bool(default)
    return bool(value)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _point_yx(row: Mapping[str, Any]) -> tuple[float, float] | None:
    for key in ("keypoint_yx", "projected_yx", "yx"):
        point = row.get(key)
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            y = _finite_float(point[0])
            x = _finite_float(point[1])
            if y is not None and x is not None:
                return y, x
    point = row.get("keypoint_xy", row.get("projected_xy", row.get("xy")))
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        x = _finite_float(point[0])
        y = _finite_float(point[1])
        if y is not None and x is not None:
            return y, x
    return None


def _gaussian_id(row: Mapping[str, Any]) -> int | None:
    value = row.get("gaussian_id", row.get("landmark_id"))
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _empty_entry(*, height: int, width: int) -> dict[str, Any]:
    return {
        "gaussian_ids": torch.empty(0, dtype=torch.long),
        "keypoint_yx": torch.empty(0, 2, dtype=torch.float32),
        "support_weights": torch.empty(0, dtype=torch.float32),
        "positive_count": 0,
        "height": int(height),
        "width": int(width),
        "target_role": "stdloc_visibility_teacher",
    }


def build_visibility_teacher_targets_with_metrics(
    projections: Mapping[str, list[Mapping[str, Any]]],
    *,
    height: int,
    width: int,
    max_points_per_image: int = 0,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError("height and width must be positive")

    targets: dict[str, dict[str, Any]] = {}
    metrics = {
        "schema_version": "ulfloc_visibility_teacher_metrics_v1",
        "height": int(height),
        "width": int(width),
        "image_count": int(len(projections)),
        "input_projection_count": 0,
        "kept_projection_count": 0,
        "invisible_projection_count": 0,
        "mask_invalid_projection_count": 0,
        "out_of_bounds_projection_count": 0,
        "invalid_projection_count": 0,
        "empty_image_count": 0,
        "support_weight_sum": 0.0,
        "dropped_by_image_cap_count": 0,
        "max_points_per_image": int(max_points_per_image),
        "detector_target_storage": "points",
        "target_role": "stdloc_visibility_teacher",
    }

    for image_id, rows in projections.items():
        entry = _empty_entry(height=int(height), width=int(width))
        gids: list[int] = []
        yx: list[list[float]] = []
        weights: list[float] = []

        for row in rows:
            metrics["input_projection_count"] += 1
            if not isinstance(row, Mapping):
                metrics["invalid_projection_count"] += 1
                continue
            if not _bool(row.get("visible", False), default=False):
                metrics["invisible_projection_count"] += 1
                continue
            if not _bool(row.get("mask_valid", True), default=True):
                metrics["mask_invalid_projection_count"] += 1
                continue
            point = _point_yx(row)
            gid = _gaussian_id(row)
            if point is None or gid is None:
                metrics["invalid_projection_count"] += 1
                continue
            y, x = point
            if not (0.0 <= y < float(height) and 0.0 <= x < float(width)):
                metrics["out_of_bounds_projection_count"] += 1
                continue
            weight = _finite_float(row.get("weight", row.get("support_weight", 1.0)))
            if weight is None:
                weight = 1.0
            gids.append(int(gid))
            yx.append([float(y), float(x)])
            weights.append(float(max(0.0, weight)))

        cap = int(max_points_per_image)
        if cap > 0 and len(gids) > cap:
            order = sorted(
                range(len(gids)),
                key=lambda idx: (-float(weights[idx]), int(gids[idx]), float(yx[idx][0]), float(yx[idx][1])),
            )[:cap]
            metrics["dropped_by_image_cap_count"] += int(len(gids) - cap)
            gids = [gids[idx] for idx in order]
            yx = [yx[idx] for idx in order]
            weights = [weights[idx] for idx in order]

        if gids:
            entry = {
                "gaussian_ids": torch.tensor(gids, dtype=torch.long),
                "keypoint_yx": torch.tensor(yx, dtype=torch.float32).reshape(-1, 2),
                "support_weights": torch.tensor(weights, dtype=torch.float32),
                "positive_count": int(len(gids)),
                "height": int(height),
                "width": int(width),
                "target_role": "stdloc_visibility_teacher",
            }
        else:
            metrics["empty_image_count"] += 1
        metrics["kept_projection_count"] += int(len(gids))
        metrics["support_weight_sum"] += float(sum(weights))
        targets[str(image_id)] = entry

    metrics["support_weight_sum"] = float(metrics["support_weight_sum"])
    return targets, metrics


def build_visibility_teacher_targets(
    projections: Mapping[str, list[Mapping[str, Any]]],
    *,
    height: int,
    width: int,
) -> dict[str, dict[str, Any]]:
    targets, _metrics = build_visibility_teacher_targets_with_metrics(projections, height=height, width=width)
    return targets
