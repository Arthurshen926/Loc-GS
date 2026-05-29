from __future__ import annotations

from typing import Any, Mapping


def _value(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def select_passing_v13_components(
    component_metrics: Mapping[str, Mapping[str, Any]],
    *,
    min_macro_te_gain_cm: float = 0.5,
    min_strict_recall_gain: float = 0.005,
    max_hard_scene_regression_cm: float = 0.0,
    max_p95_regression_cm: float = 0.0,
    min_dense_worsened_reduction_rate: float = 0.20,
    max_latency_delta_fraction: float = 0.10,
    max_candidate_regression_20cm_count: int = 0,
    max_candidate_regression_50cm_count: int = 0,
) -> dict[str, Any]:
    """Apply fixed v13 combination gates to independently validated components."""

    passing: list[str] = []
    rejected: dict[str, list[str]] = {}
    for name, metrics in sorted(component_metrics.items()):
        reasons: list[str] = []
        if _value(metrics, "macro_dense_median_te_delta_cm") > -float(min_macro_te_gain_cm):
            reasons.append("macro_te_gain_too_small")
        r10 = _value(metrics, "macro_dense_r10_delta")
        r5 = _value(metrics, "macro_dense_r5_delta")
        if max(r10, r5) < float(min_strict_recall_gain):
            reasons.append("strict_recall_gain_too_small")
        if "macro_dense_r2_delta" in metrics and _value(metrics, "macro_dense_r2_delta") < 0.0:
            reasons.append("dense_r2_regression")
        if _value(metrics, "oldhospital_dense_median_te_delta_cm") > float(max_hard_scene_regression_cm):
            reasons.append("oldhospital_regression")
        if _value(metrics, "stmaryschurch_dense_median_te_delta_cm") > float(max_hard_scene_regression_cm):
            reasons.append("stmaryschurch_regression")
        if _value(metrics, "p95_te_delta_cm") > float(max_p95_regression_cm):
            reasons.append("p95_regression")
        if "dense_worsened_reduction_rate" in metrics:
            if _value(metrics, "dense_worsened_reduction_rate") < float(min_dense_worsened_reduction_rate):
                reasons.append("dense_worsened_reduction_too_small")
        if "latency_delta_fraction" in metrics:
            if _value(metrics, "latency_delta_fraction") > float(max_latency_delta_fraction):
                reasons.append("latency_regression")
        if "candidate_regression_20cm_count" in metrics:
            if _value(metrics, "candidate_regression_20cm_count") > float(max_candidate_regression_20cm_count):
                reasons.append("candidate_regression_20cm")
        if "candidate_regression_50cm_count" in metrics:
            if _value(metrics, "candidate_regression_50cm_count") > float(max_candidate_regression_50cm_count):
                reasons.append("candidate_regression_50cm")
        if reasons:
            rejected[str(name)] = reasons
        else:
            passing.append(str(name))
    return {
        "passing_components": passing,
        "rejected_components": rejected,
        "gate": {
            "min_macro_te_gain_cm": float(min_macro_te_gain_cm),
            "min_strict_recall_gain": float(min_strict_recall_gain),
            "max_hard_scene_regression_cm": float(max_hard_scene_regression_cm),
            "max_p95_regression_cm": float(max_p95_regression_cm),
            "min_dense_worsened_reduction_rate": float(min_dense_worsened_reduction_rate),
            "max_latency_delta_fraction": float(max_latency_delta_fraction),
            "max_candidate_regression_20cm_count": int(max_candidate_regression_20cm_count),
            "max_candidate_regression_50cm_count": int(max_candidate_regression_50cm_count),
            "combination_policy": "only_components_that_pass_independent_gates",
        },
    }
