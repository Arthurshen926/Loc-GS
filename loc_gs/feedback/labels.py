from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from loc_gs.feedback.schema import FeedbackMatchRecord


def _records(records: Iterable[FeedbackMatchRecord | dict[str, Any]]) -> list[FeedbackMatchRecord]:
    return [FeedbackMatchRecord.from_mapping(record) for record in records]


def _value(stage: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = stage.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def derive_inlier_labels(
    records: Iterable[FeedbackMatchRecord | dict[str, Any]],
    *,
    reprojection_error_px_max: float = 4.0,
) -> list[int]:
    labels: list[int] = []
    for record in _records(records):
        reproj_ok = (
            record.reprojection_error_px is not None
            and float(record.reprojection_error_px) <= float(reprojection_error_px_max)
        )
        labels.append(1 if record.pnp_inlier and record.pnp_success and reproj_ok else 0)
    return labels


def derive_hard_negative_labels(
    records: Iterable[FeedbackMatchRecord | dict[str, Any]],
    *,
    descriptor_score_min: float = 0.5,
    reprojection_error_px_min: float = 8.0,
) -> list[int]:
    labels: list[int] = []
    for record in _records(records):
        descriptor_ok = record.descriptor_score is not None and float(record.descriptor_score) >= descriptor_score_min
        reproj_bad = (
            record.reprojection_error_px is not None
            and float(record.reprojection_error_px) >= float(reprojection_error_px_min)
        )
        labels.append(1 if descriptor_ok and (not record.pnp_inlier) and reproj_bad else 0)
    return labels


def derive_landmark_reliability(
    records: Iterable[FeedbackMatchRecord | dict[str, Any]],
    *,
    descriptor_score_min: float = 0.5,
    reprojection_error_px_min: float = 8.0,
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[FeedbackMatchRecord]] = defaultdict(list)
    for record in _records(records):
        key = record.matched_landmark_id or record.matched_gaussian_id
        if key:
            grouped[str(key)].append(record)

    out: dict[str, dict[str, float | int]] = {}
    for key, group in grouped.items():
        count = len(group)
        inlier_count = sum(derive_inlier_labels(group))
        hard_negative_count = sum(
            derive_hard_negative_labels(
                group,
                descriptor_score_min=descriptor_score_min,
                reprojection_error_px_min=reprojection_error_px_min,
            )
        )
        visibility_values = [record.visibility_score for record in group if record.visibility_score is not None]
        depth_values = [record.depth_consistency for record in group if record.depth_consistency is not None]
        inlier_rate = inlier_count / count if count else 0.0
        hard_negative_rate = hard_negative_count / count if count else 0.0
        mean_visibility = sum(float(v) for v in visibility_values) / len(visibility_values) if visibility_values else 0.0
        mean_depth = sum(float(v) for v in depth_values) / len(depth_values) if depth_values else 0.0
        reliability_score = max(0.0, inlier_rate * (0.5 + 0.25 * mean_visibility + 0.25 * mean_depth) - 0.5 * hard_negative_rate)
        out[key] = {
            "count": count,
            "inlier_count": inlier_count,
            "inlier_rate": inlier_rate,
            "hard_negative_count": hard_negative_count,
            "hard_negative_rate": hard_negative_rate,
            "mean_visibility_score": mean_visibility,
            "mean_depth_consistency": mean_depth,
            "reliability_score": reliability_score,
        }
    return out


def derive_scene_reliability_baseline_relative(
    candidate_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    *,
    stage: str = "dense",
) -> dict[str, Any]:
    candidate_stage = candidate_summary.get(stage, candidate_summary)
    baseline_stage = baseline_summary.get(stage, baseline_summary)
    if not isinstance(candidate_stage, dict) or not isinstance(baseline_stage, dict):
        raise ValueError("candidate and baseline summaries must contain metric dictionaries")
    candidate_te = _value(candidate_stage, "median_te_cm", "median_te")
    baseline_te = _value(baseline_stage, "median_te_cm", "median_te")
    candidate_re = _value(candidate_stage, "median_re_deg", "median_ae", "median_re")
    baseline_re = _value(baseline_stage, "median_re_deg", "median_ae", "median_re")
    candidate_r5 = _value(candidate_stage, "recall_5cm_5deg", "recall_5cm_5d", "r5")
    baseline_r5 = _value(baseline_stage, "recall_5cm_5deg", "recall_5cm_5d", "r5")
    candidate_r2 = _value(candidate_stage, "recall_2cm_2deg", "recall_2cm_2d", "r2")
    baseline_r2 = _value(baseline_stage, "recall_2cm_2deg", "recall_2cm_2d", "r2")

    median_te_delta = None if candidate_te is None or baseline_te is None else candidate_te - baseline_te
    median_re_delta = None if candidate_re is None or baseline_re is None else candidate_re - baseline_re
    recall_5_delta = None if candidate_r5 is None or baseline_r5 is None else candidate_r5 - baseline_r5
    recall_2_delta = None if candidate_r2 is None or baseline_r2 is None else candidate_r2 - baseline_r2
    paper_safe = bool(
        median_te_delta is not None
        and recall_5_delta is not None
        and median_te_delta <= 0.0
        and recall_5_delta >= 0.0
    )
    return {
        "scene": str(candidate_summary.get("scene", baseline_summary.get("scene", ""))),
        "stage": stage,
        "median_te_delta_cm": median_te_delta,
        "median_re_delta_deg": median_re_delta,
        "recall_5cm_5deg_delta": recall_5_delta,
        "recall_2cm_2deg_delta": recall_2_delta,
        "paper_safe_improvement": paper_safe,
    }
