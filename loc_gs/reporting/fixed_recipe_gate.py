from __future__ import annotations

from typing import Any, Mapping, Sequence


_DENSE_METRICS = ("median_te_cm", "recall_5cm_5deg", "recall_2cm_2deg")
_RECALL_POLICIES = {"hard", "warn"}


def _check_recall_policy(recall_policy: str) -> str:
    policy = str(recall_policy)
    if policy not in _RECALL_POLICIES:
        raise ValueError(f"recall_policy must be one of {sorted(_RECALL_POLICIES)}, got {recall_policy!r}")
    return policy


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


def _list_field(row: Mapping[str, Any] | None, key: str) -> list[Any]:
    if row is None:
        return []
    value = row.get(key, [])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def evaluate_fixed_recipe_gate(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    *,
    required_positive_scenes: Sequence[str] = (),
    neutral_scenes: Sequence[str] = (),
    max_median_te_regression_cm: float = 0.0,
    max_recall_drop: float = 0.0,
    recall_policy: str = "hard",
) -> dict[str, Any]:
    """Evaluate the fixed train-dev gate before promoting a recipe to paper-facing eval.

    The gate is intentionally scene-level: hard scenes must stay neutral or better
    even when the macro score improves.
    """

    recall_policy = _check_recall_policy(recall_policy)
    required_positive = {str(scene) for scene in required_positive_scenes}
    neutral = {str(scene) for scene in neutral_scenes}
    scenes = sorted((set(baseline.keys()) & set(candidate.keys())) | required_positive | neutral)
    scene_results: dict[str, dict[str, Any]] = {}
    macro_deltas: dict[str, list[float]] = {name: [] for name in _DENSE_METRICS}

    for scene in scenes:
        reasons: list[str] = []
        warnings: list[str] = []
        base = baseline.get(scene)
        cand = candidate.get(scene)
        deltas: dict[str, float | None] = {name: None for name in _DENSE_METRICS}
        if base is None:
            reasons.append("missing_baseline")
        if cand is None:
            reasons.append("missing_candidate")
        if base is not None and cand is not None:
            for name in _DENSE_METRICS:
                before = _metric(base, name)
                after = _metric(cand, name)
                if before is None or after is None:
                    reasons.append(f"missing_{name}")
                    continue
                delta = float(after - before)
                deltas[name] = delta
                macro_deltas[name].append(delta)
            median_delta = deltas["median_te_cm"]
            r5_delta = deltas["recall_5cm_5deg"]
            r2_delta = deltas["recall_2cm_2deg"]
            if median_delta is not None and median_delta > float(max_median_te_regression_cm):
                reasons.append("median_te_regression")
            recall_regressions: list[str] = []
            if r5_delta is not None and r5_delta < -float(max_recall_drop):
                recall_regressions.append("recall_5cm_5deg_regression")
            if r2_delta is not None and r2_delta < -float(max_recall_drop):
                recall_regressions.append("recall_2cm_2deg_regression")
            if recall_policy == "hard":
                reasons.extend(recall_regressions)
            else:
                warnings.extend(recall_regressions)
            if scene in required_positive:
                positive = (
                    (median_delta is not None and median_delta < 0.0)
                    or (r5_delta is not None and r5_delta > 0.0)
                    or (r2_delta is not None and r2_delta > 0.0)
                )
                if not positive:
                    reasons.append("missing_required_positive_gain")

        scene_results[scene] = {
            "scene": scene,
            "role": "required_positive" if scene in required_positive else "neutral" if scene in neutral else "tracked",
            "passed": not reasons,
            "reasons": sorted(set(reasons)),
            "warnings": sorted(set(warnings)),
            "delta": {name: value for name, value in deltas.items()},
        }

    macro_delta = {name: _mean(values) for name, values in macro_deltas.items()}
    macro_reasons: list[str] = []
    macro_warnings: list[str] = []
    if macro_deltas["median_te_cm"] and macro_delta["median_te_cm"] > float(max_median_te_regression_cm):
        macro_reasons.append("macro_median_te_regression")
    macro_recall_regressions: list[str] = []
    if macro_deltas["recall_5cm_5deg"] and macro_delta["recall_5cm_5deg"] < -float(max_recall_drop):
        macro_recall_regressions.append("macro_recall_5cm_5deg_regression")
    if macro_deltas["recall_2cm_2deg"] and macro_delta["recall_2cm_2deg"] < -float(max_recall_drop):
        macro_recall_regressions.append("macro_recall_2cm_2deg_regression")
    if recall_policy == "hard":
        macro_reasons.extend(macro_recall_regressions)
    else:
        macro_warnings.extend(macro_recall_regressions)

    passed = all(row["passed"] for row in scene_results.values()) and not macro_reasons
    return {
        "passed": bool(passed),
        "scene_results": scene_results,
        "macro_delta": macro_delta,
        "macro_reasons": sorted(set(macro_reasons)),
        "macro_warnings": sorted(set(macro_warnings)),
        "required_positive_scenes": sorted(required_positive),
        "neutral_scenes": sorted(neutral),
        "thresholds": {
            "max_median_te_regression_cm": float(max_median_te_regression_cm),
            "max_recall_drop": float(max_recall_drop),
            "recall_policy": recall_policy,
        },
    }


def build_scene_level_acceptance_recipe(
    baseline: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    candidate_order: Sequence[str] = (),
    required_positive_scenes: Sequence[str] = (),
    neutral_scenes: Sequence[str] = (),
    max_median_te_regression_cm: float = 0.0,
    max_recall_drop: float = 0.0,
    recall_policy: str = "hard",
) -> dict[str, Any]:
    """Select one fixed map per scene using a predeclared train-dev gate.

    This is a reconstruction-time acceptance report, not a per-query branch
    selector. Candidate order must be fixed before looking at test results.
    """

    recall_policy = _check_recall_policy(recall_policy)
    ordered_names = [str(name) for name in candidate_order if name in candidates]
    ordered_names.extend(str(name) for name in candidates if str(name) not in ordered_names)
    required_positive = {str(scene) for scene in required_positive_scenes}
    neutral = {str(scene) for scene in neutral_scenes}
    scenes = sorted(set(baseline.keys()) | required_positive | neutral)
    selected: dict[str, Mapping[str, Any]] = {}
    selected_runs: dict[str, dict[str, Any]] = {}
    candidate_evaluations: dict[str, dict[str, Any]] = {}

    for scene in scenes:
        base = baseline.get(scene)
        scene_evals: dict[str, Any] = {}
        selected_name = "native"
        selected_reason = "no_candidate_passed"
        selected_row: Mapping[str, Any] | None = base

        if base is None:
            selected_reason = "missing_baseline"
            selected_row = None
        else:
            for name in ordered_names:
                cand = candidates.get(name, {}).get(scene)
                report = evaluate_fixed_recipe_gate(
                    {scene: base},
                    {scene: cand} if cand is not None else {},
                    required_positive_scenes=(scene,) if scene in required_positive else (),
                    neutral_scenes=(scene,) if scene in neutral else (),
                    max_median_te_regression_cm=max_median_te_regression_cm,
                    max_recall_drop=max_recall_drop,
                    recall_policy=recall_policy,
                )
                scene_report = report["scene_results"].get(scene, {})
                scene_evals[name] = {
                    "passed": bool(scene_report.get("passed", False)),
                    "reasons": list(scene_report.get("reasons", [])),
                    "warnings": list(scene_report.get("warnings", [])),
                    "delta": dict(scene_report.get("delta", {})),
                }
                if bool(scene_report.get("passed", False)):
                    selected_name = name
                    selected_reason = "candidate_passed_predeclared_gate"
                    selected_row = cand
                    break

        if selected_row is not None:
            selected[scene] = selected_row
        candidate_evaluations[scene] = scene_evals
        selected_runs[scene] = {
            "scene": scene,
            "selected": selected_name,
            "reason": selected_reason,
            "run_dir": str(selected_row.get("run_dir", "")) if isinstance(selected_row, Mapping) else "",
            "data_roots": _list_field(selected_row, "data_roots") if isinstance(selected_row, Mapping) else [],
            "hyperparameters": dict(selected_row.get("hyperparameters", {}))
            if isinstance(selected_row, Mapping) and isinstance(selected_row.get("hyperparameters", {}), Mapping)
            else {},
            "split_audit": dict(selected_row.get("split_audit", {}))
            if isinstance(selected_row, Mapping) and isinstance(selected_row.get("split_audit", {}), Mapping)
            else {},
        }

    global_gate = evaluate_fixed_recipe_gate(
        baseline,
        selected,
        required_positive_scenes=required_positive_scenes,
        neutral_scenes=neutral_scenes,
        max_median_te_regression_cm=max_median_te_regression_cm,
        max_recall_drop=max_recall_drop,
        recall_policy=recall_policy,
    )
    return {
        "policy": "scene_level_reconstruction_time_acceptance",
        "selection_scope": "scene_level_map_acceptance_not_per_query",
        "candidate_order": ordered_names,
        "selected_runs": selected_runs,
        "candidate_evaluations": candidate_evaluations,
        "global_gate": global_gate,
        "thresholds": {
            "max_median_te_regression_cm": float(max_median_te_regression_cm),
            "max_recall_drop": float(max_recall_drop),
            "recall_policy": recall_policy,
        },
        "required_positive_scenes": sorted(required_positive),
        "neutral_scenes": sorted(neutral),
        "paper_safety_note": (
            "Train-dev acceptance is diagnostic until the recipe is frozen before "
            "any Cambridge test/full paper-facing evaluation."
        ),
    }
