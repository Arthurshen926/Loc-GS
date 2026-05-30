from __future__ import annotations

from collections import defaultdict
import math
from statistics import median
from typing import Any, Mapping, Sequence


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _is_non_base(label: Any) -> bool:
    text = str(label or "").strip()
    return bool(text and text != "base")


def _median(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values]
    return float(median(clean)) if clean else None


def _collect_optional(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[list[float], int]:
    values: list[float] = []
    missing = 0
    for row in rows:
        value = _optional_float(row.get(key))
        if value is None:
            missing += 1
        else:
            values.append(value)
    return values, missing


def summarize_phase0_action_safety(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize whether render-control actions fire on Phase0 case types."""

    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("case_type", "unknown"))].append(row)
    by_type: dict[str, dict[str, Any]] = {}
    for case_type, items in sorted(buckets.items()):
        deltas, missing_delta_count = _collect_optional(items, "sparse_conditioned_delta_cm")
        by_type[case_type] = {
            "count": int(len(items)),
            "non_base_action_count": int(sum(_is_non_base(row.get("sparse_conditioned_label")) for row in items)),
            "regression_20cm_count": int(sum(delta >= 20.0 for delta in deltas)),
            "improvement_20cm_count": int(sum(delta <= -20.0 for delta in deltas)),
            "missing_delta_count": int(missing_delta_count),
            "median_sparse_conditioned_delta_cm": _median(deltas),
        }
    return {
        "schema": "loc_gs_clean_render_phase1_action_safety_v1",
        "case_count": int(sum(len(items) for items in buckets.values())),
        "by_type": by_type,
        "diagnostic_only": True,
    }


def summarize_selected_render_health(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize selected-render health from train/self-map case-analysis rows."""

    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("category", "unknown"))].append(row)
    by_category: dict[str, dict[str, Any]] = {}
    for category, items in sorted(buckets.items()):
        visible, missing_visible_count = _collect_optional(items, "visible_ratio")
        near, missing_near_count = _collect_optional(items, "near_occluder_ratio")
        feature, missing_feature_count = _collect_optional(items, "feature_cosine_median")
        deltas, missing_delta_count = _collect_optional(items, "delta_sc_minus_none_cm")
        by_category[category] = {
            "count": int(len(items)),
            "non_base_selected_count": int(sum(_is_non_base(row.get("selected_label")) for row in items)),
            "median_visible_ratio": _median(visible),
            "median_near_occluder_ratio": _median(near),
            "median_feature_cosine": _median(feature),
            "median_delta_sc_minus_none_cm": _median(deltas),
            "regression_20cm_count": int(sum(delta >= 20.0 for delta in deltas)),
            "improvement_20cm_count": int(sum(delta <= -20.0 for delta in deltas)),
            "missing_visible_ratio_count": int(missing_visible_count),
            "missing_near_occluder_ratio_count": int(missing_near_count),
            "missing_feature_cosine_count": int(missing_feature_count),
            "missing_delta_count": int(missing_delta_count),
        }
    return {
        "schema": "loc_gs_clean_render_phase1_selected_render_health_v1",
        "case_count": int(sum(len(items) for items in buckets.values())),
        "by_category": by_category,
        "diagnostic_only": True,
    }
