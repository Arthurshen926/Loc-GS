from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from loc_gs.eval.timing import timing_profile_digest
from loc_gs.localization.pose_metrics import pose_error_summary


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stage_values(rows: Iterable[dict[str, Any]], stage: str, key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _as_float(row.get(f"{stage}_{key}"))
        if value is not None and np.isfinite(value):
            values.append(value)
    return values


def _stage_inliers(rows: Iterable[dict[str, Any]], stage: str) -> list[int]:
    values: list[int] = []
    for row in rows:
        value = row.get(f"{stage}_inliers")
        if value is None:
            continue
        values.append(int(value))
    return values


def _with_metric_aliases(summary: dict[str, float]) -> dict[str, float]:
    out = {
        "median_te_cm": float(summary.get("median_te", float("inf"))),
        "median_re_deg": float(summary.get("median_ae", float("inf"))),
        "avg_inliers": float(summary.get("avg_inliers", 0.0)),
    }
    aliases = {
        "recall_50cm_5deg": "recall_50cm_5d",
        "recall_15cm_5deg": None,
        "recall_10cm_5deg": "recall_10cm_5d",
        "recall_5cm_5deg": "recall_5cm_5d",
        "recall_2cm_2deg": "recall_2cm_2d",
    }
    for alias, key in aliases.items():
        if key is None:
            continue
        out[alias] = float(summary.get(key, 0.0))
    return out


def _recall_15cm_5deg(te_cm: list[float], re_deg: list[float]) -> float:
    if not te_cm or len(te_cm) != len(re_deg):
        return 0.0
    te = np.asarray(te_cm, dtype=np.float64)
    re = np.asarray(re_deg, dtype=np.float64)
    return float(((te <= 15.0) & (re <= 5.0)).mean())


def lsf_pose_stage_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Summarize sparse initial pose and dense final pose with LSF recall keys."""

    payload: dict[str, dict[str, float]] = {}
    for stage in ("sparse", "dense"):
        te = _stage_values(rows, stage, "te_cm")
        re = _stage_values(rows, stage, "re_deg")
        inliers = _stage_inliers(rows, stage)
        summary = _with_metric_aliases(pose_error_summary(te, re, inliers))
        summary["recall_15cm_5deg"] = _recall_15cm_5deg(te, re)
        payload[stage] = summary
    return payload


def dense_transition_summary(
    rows: list[dict[str, Any]],
    *,
    tolerance_cm: float = 0.0,
    recall_te_cm: float = 5.0,
    recall_re_deg: float = 5.0,
) -> dict[str, Any]:
    """Count whether dense refinement improves, worsens, recovers, or loses queries."""

    improved = worsened = unchanged = 0
    recovered = lost = 0
    examples: list[dict[str, Any]] = []
    tol = max(float(tolerance_cm), 0.0)
    for index, row in enumerate(rows):
        sparse_te = _as_float(row.get("sparse_te_cm"))
        dense_te = _as_float(row.get("dense_te_cm"))
        sparse_re = _as_float(row.get("sparse_re_deg"))
        dense_re = _as_float(row.get("dense_re_deg"))
        if sparse_te is None or dense_te is None:
            continue
        delta = dense_te - sparse_te
        if delta < -tol:
            improved += 1
        elif delta > tol:
            worsened += 1
        else:
            unchanged += 1
        sparse_ok = (
            sparse_te <= float(recall_te_cm)
            and sparse_re is not None
            and sparse_re <= float(recall_re_deg)
        )
        dense_ok = (
            dense_te <= float(recall_te_cm)
            and dense_re is not None
            and dense_re <= float(recall_re_deg)
        )
        if (not sparse_ok) and dense_ok:
            recovered += 1
        if sparse_ok and (not dense_ok):
            lost += 1
        if len(examples) < 10 and abs(delta) > tol:
            examples.append(
                {
                    "query_id": str(row.get("query_id", f"query_{index:06d}")),
                    "sparse_te_cm": sparse_te,
                    "dense_te_cm": dense_te,
                    "delta_te_cm": delta,
                }
            )
    total = improved + worsened + unchanged
    return {
        "query_count": int(total),
        "improved_count": int(improved),
        "worsened_count": int(worsened),
        "unchanged_count": int(unchanged),
        "recovered_r5_count": int(recovered),
        "lost_r5_count": int(lost),
        "examples": examples,
    }


def summarize_lsf_eval(
    rows: list[dict[str, Any]],
    *,
    scene: str,
    method: str,
    landmark_count: int | None = None,
    timing_profile: dict[str, Any] | None = None,
    offline_costs: dict[str, Any] | None = None,
    map_size_mb: float | None = None,
) -> dict[str, Any]:
    """Build one audit-friendly LSF report from query rows and profile payloads."""

    report: dict[str, Any] = {
        "scene": str(scene),
        "method": str(method),
        "query_count": int(len(rows)),
        "pose": lsf_pose_stage_summary(rows),
        "dense_transition": dense_transition_summary(rows),
    }
    if landmark_count is not None:
        report["landmark_budget"] = {"landmark_count": int(landmark_count)}
    if timing_profile is not None:
        report["online_timing"] = timing_profile
        report["online_timing_digest"] = timing_profile_digest(timing_profile)
    if offline_costs is not None:
        report["offline_costs"] = dict(offline_costs)
    if map_size_mb is not None:
        report["map_size_mb"] = float(map_size_mb)
    return report

