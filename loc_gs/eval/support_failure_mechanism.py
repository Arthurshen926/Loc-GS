from __future__ import annotations

import math
from typing import Any


def _stage_dict(payload: dict[str, Any] | None, stage: str = "dense") -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    value = payload.get(stage)
    if isinstance(value, dict):
        return value
    pose = payload.get("pose")
    if isinstance(pose, dict) and isinstance(pose.get(stage), dict):
        return pose[stage]
    return payload


def _summary_dict(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    value = payload.get("summary")
    if isinstance(value, dict):
        return value
    transition = payload.get("dense_transition")
    if isinstance(transition, dict):
        return transition
    return payload


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _metric(payload: dict[str, Any] | None, keys: tuple[str, ...], *, stage: str = "dense") -> float | None:
    data = _stage_dict(payload, stage)
    for key in keys:
        value = _as_float(data.get(key))
        if value is not None:
            return value
    return None


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return round(float(candidate) - float(baseline), 12)


def _solver_summary(report: dict[str, Any], key: str) -> dict[str, Any]:
    value = report.get(key, {})
    if isinstance(value, dict):
        summary = value.get("summary", {})
        return summary if isinstance(summary, dict) else {}
    return {}


def _solver_delta(report: dict[str, Any], key: str, source_key: str | None = None) -> float | int | None:
    delta = report.get("summary_delta", {})
    if isinstance(delta, dict) and key in delta:
        value = _as_float(delta.get(key))
        if value is not None:
            if key.endswith("_sum") or key.endswith("_count"):
                return int(round(value))
            return round(value, 12)
    source_key = source_key or key
    source = _as_float(_solver_summary(report, "source").get(source_key))
    candidate = _as_float(_solver_summary(report, "candidate").get(source_key))
    out = _delta(candidate, source)
    if out is None:
        return None
    if source_key.endswith("_sum") or source_key.endswith("_count"):
        return int(round(out))
    return out


def _transition_delta(
    baseline_transition: dict[str, Any] | None,
    candidate_transition: dict[str, Any] | None,
    key: str,
) -> int | None:
    aliases = {
        "r5_lost_count": ("r5_lost_count", "lost_r5_count"),
        "r2_lost_count": ("r2_lost_count", "lost_r2_count"),
    }
    keys = aliases.get(key, (key,))
    baseline_summary = _summary_dict(baseline_transition)
    candidate_summary = _summary_dict(candidate_transition)
    baseline = next((baseline_summary.get(item) for item in keys if item in baseline_summary), None)
    candidate = next((candidate_summary.get(item) for item in keys if item in candidate_summary), None)
    baseline_value = _as_float(baseline)
    candidate_value = _as_float(candidate)
    if baseline_value is None or candidate_value is None:
        return None
    return int(round(candidate_value - baseline_value))


def build_support_failure_mechanism_report(
    solver_report: dict[str, Any],
    *,
    baseline_metrics: dict[str, Any] | None = None,
    candidate_metrics: dict[str, Any] | None = None,
    baseline_transition: dict[str, Any] | None = None,
    candidate_transition: dict[str, Any] | None = None,
    scene: str | None = None,
) -> dict[str, Any]:
    """Combine solver, pose, and dense-transition deltas into one mechanism diagnostic.

    This is a paper-safety aid, not an official evaluator.  It is meant to
    expose cases where support count is preserved while solver quality or dense
    reliability worsens.
    """

    source_summary = _solver_summary(solver_report, "source")
    candidate_summary = _solver_summary(solver_report, "candidate")
    support_delta = _solver_delta(solver_report, "support_count_sum")
    viable_mass_delta = _solver_delta(solver_report, "viable_tuple_mass")
    logdet_delta = _solver_delta(solver_report, "mean_logdet_H")
    ambiguity_delta = _solver_delta(solver_report, "mean_ambiguity_risk")

    baseline_te = _metric(baseline_metrics, ("median_te_cm", "median_te", "median_translation_cm"))
    candidate_te = _metric(candidate_metrics, ("median_te_cm", "median_te", "median_translation_cm"))
    baseline_r5 = _metric(baseline_metrics, ("recall_5cm_5deg", "recall_5cm_5d", "r5"))
    candidate_r5 = _metric(candidate_metrics, ("recall_5cm_5deg", "recall_5cm_5d", "r5"))
    baseline_r2 = _metric(baseline_metrics, ("recall_2cm_2deg", "recall_2cm_2d", "r2"))
    candidate_r2 = _metric(candidate_metrics, ("recall_2cm_2deg", "recall_2cm_2d", "r2"))

    dense_worsened_delta = _transition_delta(baseline_transition, candidate_transition, "worsened_count")
    r5_lost_delta = _transition_delta(baseline_transition, candidate_transition, "r5_lost_count")
    r2_lost_delta = _transition_delta(baseline_transition, candidate_transition, "r2_lost_count")

    deltas = {
        "support_count_delta": support_delta,
        "viable_tuple_mass_delta": viable_mass_delta,
        "mean_logdet_H_delta": logdet_delta,
        "ambiguity_risk_delta": ambiguity_delta,
        "dense_worsened_count_delta": dense_worsened_delta,
        "dense_r5_lost_count_delta": r5_lost_delta,
        "dense_r2_lost_count_delta": r2_lost_delta,
        "dense_median_te_cm_delta": _delta(candidate_te, baseline_te),
        "dense_r5_delta": _delta(candidate_r5, baseline_r5),
        "dense_r2_delta": _delta(candidate_r2, baseline_r2),
    }

    eps = 1e-9
    support_preserved = support_delta is not None and float(support_delta) >= -eps
    solver_degraded = (
        (viable_mass_delta is not None and float(viable_mass_delta) < -eps)
        or (logdet_delta is not None and float(logdet_delta) < -eps)
        or (ambiguity_delta is not None and float(ambiguity_delta) > eps)
    )
    pose_degraded = (
        (deltas["dense_median_te_cm_delta"] is not None and float(deltas["dense_median_te_cm_delta"]) > eps)
        or (deltas["dense_r5_delta"] is not None and float(deltas["dense_r5_delta"]) < -eps)
        or (deltas["dense_r2_delta"] is not None and float(deltas["dense_r2_delta"]) < -eps)
        or (dense_worsened_delta is not None and int(dense_worsened_delta) > 0)
    )

    flags: list[str] = []
    if support_preserved and solver_degraded:
        flags.append("support_preserved_but_solver_degraded")
    if pose_degraded and solver_degraded:
        flags.append("pose_degraded_with_solver_drop")
    if dense_worsened_delta is not None and int(dense_worsened_delta) > 0:
        flags.append("dense_worsened_count_increased")

    split_audit = source_summary.get("split_audit", {})
    split_audit_status = split_audit.get("audit_status", "unknown") if isinstance(split_audit, dict) else "unknown"
    query_group_mode = str(source_summary.get("query_group_mode", "unknown"))
    paper_state = "eligible_mechanism_diagnostic"
    if split_audit_status != "passed" or query_group_mode.startswith("synthetic"):
        paper_state = "diagnostic_only"

    return {
        "scene": str(scene or solver_report.get("scene") or "unknown"),
        "mechanism_table": {
            "source": {
                "support_count": source_summary.get("support_count_sum"),
                "viable_tuple_mass": source_summary.get("viable_tuple_mass"),
                "mean_logdet_H": source_summary.get("mean_logdet_H"),
                "ambiguity_risk": source_summary.get("mean_ambiguity_risk"),
                "dense_median_te_cm": baseline_te,
                "dense_r5": baseline_r5,
                "dense_r2": baseline_r2,
                "dense_worsened_count": _summary_dict(baseline_transition).get("worsened_count"),
            },
            "candidate": {
                "support_count": candidate_summary.get("support_count_sum"),
                "viable_tuple_mass": candidate_summary.get("viable_tuple_mass"),
                "mean_logdet_H": candidate_summary.get("mean_logdet_H"),
                "ambiguity_risk": candidate_summary.get("mean_ambiguity_risk"),
                "dense_median_te_cm": candidate_te,
                "dense_r5": candidate_r5,
                "dense_r2": candidate_r2,
                "dense_worsened_count": _summary_dict(candidate_transition).get("worsened_count"),
            },
            "deltas": deltas,
        },
        "verdict": {
            "support_count_not_sufficient": bool(support_preserved and solver_degraded),
            "solver_degraded": bool(solver_degraded),
            "pose_degraded": bool(pose_degraded),
            "flags": flags,
            "summary": (
                "support-count preservation is insufficient"
                if support_preserved and solver_degraded
                else "support-count failure mechanism not established"
            ),
        },
        "paper_safety": {
            "split_audit_status": split_audit_status,
            "query_group_mode": query_group_mode,
            "paper_facing_state": paper_state,
        },
    }


def support_failure_mechanism_markdown(report: dict[str, Any]) -> str:
    deltas = report["mechanism_table"]["deltas"]
    lines = [
        f"# Support Failure Mechanism: {report.get('scene', 'unknown')}",
        "",
        "This report is a mechanism diagnostic, not an official pose leaderboard.",
        f"- verdict: `{report['verdict']['summary']}`",
        f"- split audit: `{report['paper_safety']['split_audit_status']}`",
        f"- query grouping: `{report['paper_safety']['query_group_mode']}`",
        "",
        "| metric | delta |",
        "| --- | ---: |",
    ]
    for key in (
        "support_count_delta",
        "viable_tuple_mass_delta",
        "mean_logdet_H_delta",
        "ambiguity_risk_delta",
        "dense_worsened_count_delta",
        "dense_median_te_cm_delta",
        "dense_r5_delta",
        "dense_r2_delta",
    ):
        value = deltas.get(key)
        if value is None:
            text = "n/a"
        elif isinstance(value, int):
            text = f"{value:+d}"
        else:
            text = f"{float(value):+.6f}"
        lines.append(f"| {key} | {text} |")
    if report["verdict"]["support_count_not_sufficient"]:
        lines.extend(
            [
                "",
                "The candidate preserves landmark support count, but solver-level quality drops.",
                "This is direct evidence that support-count preservation is insufficient for Loc-GS selection.",
            ]
        )
    return "\n".join(lines) + "\n"
