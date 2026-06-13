from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F


SCHEMA_VERSION = "ulfloc_multiview_feature_observation_cache_v1"
DESCRIPTOR_SOURCE = "ulf_original_multiview_feature_observation"


def validate_multiview_observation_split_name(split_name: str) -> str:
    split = str(split_name or "unknown").strip() or "unknown"
    lowered = split.lower()
    if lowered == "test" or lowered == "official_test" or lowered.endswith("_test"):
        raise ValueError("refusing to build ULF multiview feature observation cache from test split")
    return split


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return float(number)


def _nonnegative(value: Any, default: float = 0.0) -> float:
    return max(0.0, _finite_float(value, default=default))


def _descriptor(row: Mapping[str, Any]) -> torch.Tensor | None:
    if "descriptor" not in row:
        return None
    desc = torch.as_tensor(row["descriptor"], dtype=torch.float32).reshape(-1)
    if desc.numel() == 0 or not bool(torch.isfinite(desc).all()):
        return None
    return F.normalize(desc, p=2, dim=0)


def _point_yx(row: Mapping[str, Any]) -> list[float] | None:
    if "keypoint_yx" in row:
        point = torch.as_tensor(row["keypoint_yx"], dtype=torch.float32).reshape(-1)
        if point.numel() >= 2:
            return [float(point[0].item()), float(point[1].item())]
    if "keypoint_xy" in row:
        point = torch.as_tensor(row["keypoint_xy"], dtype=torch.float32).reshape(-1)
        if point.numel() >= 2:
            return [float(point[1].item()), float(point[0].item())]
    return None


def _rows_by_view(raw: Mapping[str, Any] | list[Any]) -> list[tuple[str, Mapping[str, Any]]]:
    rows: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(raw, Mapping):
        for view_id, value in raw.items():
            if isinstance(value, (list, tuple)):
                for row in value:
                    if isinstance(row, Mapping):
                        rows.append((str(view_id), row))
            elif isinstance(value, Mapping):
                for landmark_key, item in value.items():
                    if isinstance(item, Mapping):
                        merged = dict(item)
                        merged.setdefault("gaussian_id", landmark_key)
                        rows.append((str(view_id), merged))
        return rows
    for item in raw:
        if isinstance(item, Mapping):
            rows.append((str(item.get("source_view_id", item.get("view_id", item.get("image_id", "")))), item))
    return rows


def build_multiview_feature_observation_cache(
    observations: Mapping[str, Any] | list[Any],
    *,
    sampled_idx: torch.Tensor | Any,
    scene: str,
    split_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    split = validate_multiview_observation_split_name(split_name)
    sampled = torch.as_tensor(sampled_idx, dtype=torch.long).reshape(-1).detach().cpu()
    if sampled.numel() == 0:
        raise ValueError("sampled_idx must be non-empty")
    sampled_set = {int(gid) for gid in sampled.tolist()}
    metrics: dict[str, Any] = {
        "schema_version": "ulfloc_multiview_feature_observation_cache_metrics_v1",
        "scene": str(scene),
        "split_name": split,
        "sampled_count": int(sampled.numel()),
        "input_observation_count": 0,
        "kept_observation_count": 0,
        "invalid_observation_count": 0,
        "not_sampled_observation_count": 0,
        "zero_weight_observation_count": 0,
        "observed_landmark_count": 0,
        "observed_view_count": 0,
        "descriptor_dim": 0,
        "observation_descriptor_source": DESCRIPTOR_SOURCE,
    }
    grouped: dict[int, list[dict[str, Any]]] = {}
    observed_views: set[str] = set()
    for view_id, row in _rows_by_view(observations):
        metrics["input_observation_count"] += 1
        gid_value = row.get("gaussian_id", row.get("landmark_id"))
        try:
            gid = int(gid_value)
        except (TypeError, ValueError):
            metrics["invalid_observation_count"] += 1
            continue
        if gid not in sampled_set:
            metrics["not_sampled_observation_count"] += 1
            continue
        desc = _descriptor(row)
        if desc is None:
            metrics["invalid_observation_count"] += 1
            continue
        positive = _nonnegative(
            row.get("positive_weight", row.get("geometry_weight", row.get("weight", row.get("cosine_weight", 1.0)))),
            default=1.0,
        )
        if positive <= 0.0:
            metrics["zero_weight_observation_count"] += 1
            continue
        negative = _nonnegative(row.get("negative_weight", row.get("risk", 0.0)))
        point_yx = _point_yx(row)
        item: dict[str, Any] = {
            "descriptor": desc.detach().cpu(),
            "positive_weight": float(positive),
            "negative_weight": float(negative),
            "source_view_id": str(view_id),
            "gaussian_id": int(gid),
            "descriptor_source": DESCRIPTOR_SOURCE,
        }
        if point_yx is not None:
            item["keypoint_yx"] = point_yx
        if row.get("depth_m", row.get("depth")) is not None:
            item["depth_m"] = float(_finite_float(row.get("depth_m", row.get("depth")), default=0.0))
        grouped.setdefault(int(gid), []).append(item)
        observed_views.add(str(view_id))
        metrics["kept_observation_count"] += 1
        metrics["descriptor_dim"] = int(desc.numel())
    metrics["observed_landmark_count"] = int(len(grouped))
    metrics["observed_view_count"] = int(len(observed_views))
    cache = {
        "schema_version": SCHEMA_VERSION,
        "scene": str(scene),
        "split_name": split,
        "sampled_idx": sampled,
        "observations": grouped,
        "observation_format": "explicit_multiview_feature_observations",
        "observation_id_space": "gaussian_id",
        "observation_descriptor_source": DESCRIPTOR_SOURCE,
        "split_audit": {
            "schema_version": "ulfloc_multiview_feature_observation_cache_split_audit_v1",
            "split_name": split,
            "test_split_used": False,
            "official_test_used": False,
        },
    }
    return cache, metrics
