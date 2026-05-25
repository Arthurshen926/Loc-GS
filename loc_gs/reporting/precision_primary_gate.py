from __future__ import annotations

from typing import Any, Mapping, Sequence


_METRICS = (
    "median_te_cm",
    "median_re_deg",
    "recall_5cm_5deg",
    "recall_2cm_2deg",
)


def _dense(row: Mapping[str, Any]) -> Mapping[str, Any]:
    dense = row.get("dense", {})
    if isinstance(dense, Mapping):
        return dense
    metrics = row.get("metrics", {})
    if isinstance(metrics, Mapping):
        nested = metrics.get("dense", {})
        if isinstance(nested, Mapping):
            return nested
    return {}


def _metric(row: Mapping[str, Any], name: str) -> float | None:
    dense = _dense(row)
    if name not in dense or dense[name] is None:
        return None
    return float(dense[name])


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def evaluate_precision_primary_gate(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    *,
    required_positive_scenes: Sequence[str] = (),
    neutral_scenes: Sequence[str] = (),
    max_recall_drop: float = 0.002,
    min_median_te_gain_cm: float = 0.0,
    min_median_re_gain_deg: float = 0.0,
    max_abs_median_te_cm: float = 100.0,
    max_abs_median_re_deg: float = 5.0,
) -> dict[str, Any]:
    """Evaluate a precision-primary diagnostic gate.

    This report is intended for analysis, not post-hoc test selection. Pose
    median precision is primary, recall is a bounded safety check, and unusable
    absolute pose quality cannot be promoted as a success.
    """

    required_positive = {str(scene) for scene in required_positive_scenes}
    neutral = {str(scene) for scene in neutral_scenes}
    scenes = sorted((set(baseline) & set(candidate)) | required_positive | neutral)
    scene_results: dict[str, dict[str, Any]] = {}
    macro_deltas: dict[str, list[float]] = {name: [] for name in _METRICS}

    for scene in scenes:
        reasons: list[str] = []
        base = baseline.get(scene)
        cand = candidate.get(scene)
        deltas: dict[str, float | None] = {name: None for name in _METRICS}
        candidate_metrics: dict[str, float | None] = {name: None for name in _METRICS}
        if base is None:
            reasons.append("missing_baseline")
        if cand is None:
            reasons.append("missing_candidate")
        if base is not None and cand is not None:
            for name in _METRICS:
                before = _metric(base, name)
                after = _metric(cand, name)
                candidate_metrics[name] = after
                if before is None or after is None:
                    reasons.append(f"missing_{name}")
                    continue
                delta = float(after - before)
                deltas[name] = delta
                macro_deltas[name].append(delta)

        te_delta = deltas["median_te_cm"]
        re_delta = deltas["median_re_deg"]
        r5_delta = deltas["recall_5cm_5deg"]
        r2_delta = deltas["recall_2cm_2deg"]
        precision_improved = (
            (te_delta is not None and te_delta < -float(min_median_te_gain_cm))
            or (re_delta is not None and re_delta < -float(min_median_re_gain_deg))
        )
        recall_safe = (
            (r5_delta is not None and r5_delta >= -float(max_recall_drop))
            and (r2_delta is not None and r2_delta >= -float(max_recall_drop))
        )
        abs_te = candidate_metrics["median_te_cm"]
        abs_re = candidate_metrics["median_re_deg"]
        absolute_pose_usable = (
            abs_te is not None
            and abs_re is not None
            and abs_te <= float(max_abs_median_te_cm)
            and abs_re <= float(max_abs_median_re_deg)
        )
        if base is not None and cand is not None:
            if not precision_improved:
                reasons.append("missing_precision_gain")
            if not recall_safe:
                reasons.append("recall_safety_regression")
            if not absolute_pose_usable:
                reasons.append("absolute_pose_not_usable")
            if scene in required_positive and not precision_improved:
                reasons.append("missing_required_precision_gain")

        scene_results[scene] = {
            "scene": scene,
            "role": "required_positive" if scene in required_positive else "neutral" if scene in neutral else "tracked",
            "passed": not reasons,
            "reasons": sorted(set(reasons)),
            "delta": deltas,
            "candidate_dense": candidate_metrics,
            "precision_improved": bool(precision_improved),
            "recall_safe": bool(recall_safe),
            "absolute_pose_usable": bool(absolute_pose_usable),
        }

    macro_delta = {name: _mean(values) for name, values in macro_deltas.items()}
    macro_precision_improved = (
        bool(macro_deltas["median_te_cm"] and macro_delta["median_te_cm"] < -float(min_median_te_gain_cm))
        or bool(macro_deltas["median_re_deg"] and macro_delta["median_re_deg"] < -float(min_median_re_gain_deg))
    )
    macro_recall_safe = (
        (not macro_deltas["recall_5cm_5deg"] or macro_delta["recall_5cm_5deg"] >= -float(max_recall_drop))
        and (not macro_deltas["recall_2cm_2deg"] or macro_delta["recall_2cm_2deg"] >= -float(max_recall_drop))
    )
    macro_reasons: list[str] = []
    if not macro_precision_improved:
        macro_reasons.append("macro_missing_precision_gain")
    if not macro_recall_safe:
        macro_reasons.append("macro_recall_safety_regression")

    passed = all(row["passed"] for row in scene_results.values()) and not macro_reasons
    return {
        "policy": "precision_primary_pose_gate",
        "passed": bool(passed),
        "scene_results": scene_results,
        "macro_delta": macro_delta,
        "macro_reasons": macro_reasons,
        "thresholds": {
            "max_recall_drop": float(max_recall_drop),
            "min_median_te_gain_cm": float(min_median_te_gain_cm),
            "min_median_re_gain_deg": float(min_median_re_gain_deg),
            "max_abs_median_te_cm": float(max_abs_median_te_cm),
            "max_abs_median_re_deg": float(max_abs_median_re_deg),
        },
        "paper_safety_note": (
            "Precision-primary diagnostics may motivate a future fixed recipe, "
            "but must not be used for test-set reselection."
        ),
    }
