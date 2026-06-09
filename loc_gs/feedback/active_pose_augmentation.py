from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from loc_gs.feedback.ray_attributed_solver_feedback import normalize_contributors, validate_split_name


SCHEMA_VERSION = "solver_feedback_active_pose_augmentation_v1"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _clamp01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _float(value, default)))


def _record_split(record: Mapping[str, Any], default: str) -> str:
    split = str(record.get("split_name", record.get("split", default))).strip() or default
    return validate_split_name(split)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item).strip()]
    return []


def _candidate_pose_id(record: Mapping[str, Any], index: int) -> str:
    pose_id = str(record.get("pose_id", record.get("view_id", record.get("source_view_id", "")))).strip()
    return pose_id or f"candidate_{index:06d}"


def _candidate_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    query_ids = sorted(set(_string_list(record.get("query_ids", record.get("query_id")))))
    covered_cells = record.get("covered_cells", [])
    covered_depth_bins = record.get("covered_depth_bins", [])
    return {
        "alpha_valid_ratio": _clamp01(record.get("alpha_valid_ratio"), 1.0),
        "stable_mask_ratio": _clamp01(record.get("stable_mask_ratio"), 1.0),
        "selected_landmark_visible_count": _int(record.get("selected_landmark_visible_count"), 0),
        "candidate_landmark_visible_count": _int(record.get("candidate_landmark_visible_count"), 0),
        "feature_variance": max(0.0, _float(record.get("feature_variance"), 0.0)),
        "artifact_score": _clamp01(record.get("artifact_score", record.get("ray_artifact_score")), 0.0),
        "novelty_score": _clamp01(record.get("novelty_score"), 0.0),
        "expected_query_gain": max(0.0, _float(record.get("expected_query_gain"), 0.0)),
        "query_ids": query_ids,
        "covered_cell_count": len(covered_cells) if isinstance(covered_cells, Sequence) else 0,
        "covered_depth_bin_count": len(set(covered_depth_bins)) if isinstance(covered_depth_bins, Sequence) else 0,
    }


def _reject_reason(
    metrics: Mapping[str, Any],
    *,
    min_alpha_valid: float,
    min_stable_mask_ratio: float,
    min_selected_landmark_visible: int,
    min_candidate_landmark_visible: int,
    min_feature_variance: float,
    max_artifact_score: float,
) -> str:
    reasons: list[str] = []
    if float(metrics["alpha_valid_ratio"]) < float(min_alpha_valid):
        reasons.append("alpha_valid_ratio")
    if float(metrics["stable_mask_ratio"]) < float(min_stable_mask_ratio):
        reasons.append("stable_mask_ratio")
    if int(metrics["selected_landmark_visible_count"]) < int(min_selected_landmark_visible):
        reasons.append("selected_landmark_visible_count")
    if int(metrics["candidate_landmark_visible_count"]) < int(min_candidate_landmark_visible):
        reasons.append("candidate_landmark_visible_count")
    if float(metrics["feature_variance"]) < float(min_feature_variance):
        reasons.append("feature_variance")
    if float(metrics["artifact_score"]) > float(max_artifact_score):
        reasons.append("artifact_score")
    return ",".join(reasons)


def _pose_row(record: Mapping[str, Any], *, index: int, split_name: str) -> dict[str, Any]:
    _record_split(record, split_name)
    pose_id = _candidate_pose_id(record, index)
    metrics = _candidate_metrics(record)
    return {
        "pose_id": pose_id,
        "anchor_view_id": str(record.get("anchor_view_id", record.get("anchor_id", ""))).strip(),
        "pose_source": str(record.get("pose_source", record.get("source_role", "unknown"))).strip() or "unknown",
        "split_name": split_name,
        **metrics,
    }


def _priority(
    row: Mapping[str, Any],
    *,
    selected_queries: set[str],
    hard_query_ids: set[str],
) -> tuple[int, int, int, float, float, int, int, int, int, float]:
    query_ids = set(_string_list(row.get("query_ids")))
    uncovered = query_ids.difference(selected_queries)
    hard_uncovered = uncovered.intersection(hard_query_ids)
    hard_any = query_ids.intersection(hard_query_ids)
    return (
        1 if hard_uncovered else 0,
        1 if hard_any else 0,
        len(uncovered),
        float(row.get("expected_query_gain", 0.0)),
        float(row.get("novelty_score", 0.0)),
        int(row.get("covered_cell_count", 0)),
        int(row.get("covered_depth_bin_count", 0)),
        int(row.get("selected_landmark_visible_count", 0)),
        int(row.get("candidate_landmark_visible_count", 0)),
        -float(row.get("artifact_score", 0.0)),
    )


def select_active_pose_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    split_name: str,
    hard_query_ids: Iterable[str] | None = None,
    max_poses: int = 32,
    max_per_anchor: int = 4,
    min_alpha_valid: float = 0.5,
    min_stable_mask_ratio: float = 0.0,
    min_selected_landmark_visible: int = 1,
    min_candidate_landmark_visible: int = 1,
    min_feature_variance: float = 0.0,
    max_artifact_score: float = 0.5,
) -> dict[str, Any]:
    """Select active render poses using hard constraints and lexicographic utility.

    This is intentionally not a learned policy and not a weighted objective. It is a
    deterministic controller that keeps only poses whose rendered view is likely to
    expose sparse-supported landmarks, then prioritizes hard-query and new-query
    coverage before secondary quality signals.
    """

    split = validate_split_name(split_name)
    hard_queries = {str(item) for item in (hard_query_ids or [])}
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, record in enumerate(candidates):
        row = _pose_row(record, index=index, split_name=split)
        reason = _reject_reason(
            row,
            min_alpha_valid=float(min_alpha_valid),
            min_stable_mask_ratio=float(min_stable_mask_ratio),
            min_selected_landmark_visible=int(min_selected_landmark_visible),
            min_candidate_landmark_visible=int(min_candidate_landmark_visible),
            min_feature_variance=float(min_feature_variance),
            max_artifact_score=float(max_artifact_score),
        )
        if reason:
            rejected.append({**row, "reject_reason": reason})
        else:
            eligible.append(row)

    selected: list[dict[str, Any]] = []
    selected_queries: set[str] = set()
    anchor_counts: dict[str, int] = defaultdict(int)
    remaining = list(eligible)
    while remaining and len(selected) < int(max_poses):
        ranked = sorted(
            enumerate(remaining),
            key=lambda item: (
                _priority(item[1], selected_queries=selected_queries, hard_query_ids=hard_queries),
                item[1]["pose_id"],
            ),
            reverse=True,
        )
        chosen_index: int | None = None
        per_anchor_skips: list[int] = []
        for index, row in ranked:
            anchor = str(row.get("anchor_view_id", ""))
            if anchor and anchor_counts[anchor] >= int(max_per_anchor):
                per_anchor_skips.append(index)
                continue
            chosen_index = index
            break
        if chosen_index is None:
            for index in sorted(per_anchor_skips, reverse=True):
                row = remaining.pop(index)
                rejected.append({**row, "reject_reason": "max_per_anchor"})
            break
        row = remaining.pop(chosen_index)
        selected.append({**row, "selection_rank": len(selected)})
        selected_queries.update(_string_list(row.get("query_ids")))
        anchor = str(row.get("anchor_view_id", ""))
        if anchor:
            anchor_counts[anchor] += 1

    for row in remaining:
        rejected.append({**row, "reject_reason": "max_poses"})

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "split_name": split,
        "candidate_pose_count": int(len(eligible) + len(rejected)),
        "eligible_pose_count": int(len(eligible)),
        "selected_pose_count": int(len(selected)),
        "rejected_pose_count": int(len(rejected)),
        "selected_query_count": int(len(selected_queries)),
        "selected_hard_query_count": int(len(selected_queries.intersection(hard_queries))),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "split_name": split,
        "selected_poses": selected,
        "rejected_poses": rejected,
        "metrics": metrics,
        "hyperparameters": {
            "max_poses": int(max_poses),
            "max_per_anchor": int(max_per_anchor),
            "min_alpha_valid": float(min_alpha_valid),
            "min_stable_mask_ratio": float(min_stable_mask_ratio),
            "min_selected_landmark_visible": int(min_selected_landmark_visible),
            "min_candidate_landmark_visible": int(min_candidate_landmark_visible),
            "min_feature_variance": float(min_feature_variance),
            "max_artifact_score": float(max_artifact_score),
        },
    }


def _contributors(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    contributors = record.get("contributors", record.get("ray_contributors", []))
    if not isinstance(contributors, Sequence) or isinstance(contributors, (str, bytes)):
        raise TypeError("observation contributors must be a sequence")
    return list(contributors)


def candidate_poses_from_observations(
    observations: Iterable[Mapping[str, Any]],
    *,
    split_name: str,
) -> list[dict[str, Any]]:
    """Infer active-pose candidates by grouping solver observations by source view.

    This is the fallback path when a renderer-specific candidate-pose generator has
    not materialized a separate pose table yet. It does not invent poses; it only
    summarizes already rendered/observed source views so the same active selection
    gates can be applied.
    """

    split = validate_split_name(split_name)
    by_view: dict[str, dict[str, Any]] = {}
    for record in observations:
        _record_split(record, split)
        source_view_id = str(record.get("source_view_id", record.get("view_id", ""))).strip()
        if not source_view_id:
            continue
        normalized = normalize_contributors(_contributors(record))
        if not normalized:
            continue
        row = by_view.setdefault(
            source_view_id,
            {
                "pose_id": source_view_id,
                "anchor_view_id": str(record.get("anchor_view_id", source_view_id)).strip() or source_view_id,
                "pose_source": str(record.get("source_role", record.get("pose_source", "observation_aggregate"))),
                "split_name": split,
                "query_ids": set(),
                "candidate_landmark_ids": set(),
                "selected_landmark_ids": set(),
                "covered_cells": set(),
                "covered_depth_bins": set(),
                "artifact_scores": [],
                "feature_variances": [],
                "positive_support": 0.0,
            },
        )
        query_id = str(record.get("query_id", record.get("image_id", ""))).strip()
        if query_id:
            row["query_ids"].add(query_id)
        cell = record.get("image_cell", record.get("cell"))
        if isinstance(cell, Sequence) and not isinstance(cell, (str, bytes)) and len(cell) >= 2:
            row["covered_cells"].add((int(cell[0]), int(cell[1])))
        if "depth_bin" in record:
            row["covered_depth_bins"].add(_int(record.get("depth_bin")))
        artifact = _artifact(record)
        row["artifact_scores"].append(float(artifact))
        if "feature_variance" in record:
            row["feature_variances"].append(max(0.0, _float(record.get("feature_variance"), 0.0)))
        pnp_inlier = _bool(record.get("pnp_inlier", False))
        quality = _quality(record, reprojection_threshold_px=4.0) if pnp_inlier and artifact <= 0.3 else 0.0
        for item in normalized:
            gid = int(item["gaussian_id"])
            row["candidate_landmark_ids"].add(gid)
            if quality > 0.0:
                row["selected_landmark_ids"].add(gid)
                row["positive_support"] = float(row["positive_support"]) + quality * float(item["weight"])

    candidates: list[dict[str, Any]] = []
    for row in sorted(by_view.values(), key=lambda item: str(item["pose_id"])):
        artifact_values = list(row["artifact_scores"])
        feature_values = list(row["feature_variances"])
        artifact_score = sum(artifact_values) / len(artifact_values) if artifact_values else 0.0
        feature_variance = sum(feature_values) / len(feature_values) if feature_values else 0.0
        candidates.append(
            {
                "pose_id": str(row["pose_id"]),
                "anchor_view_id": str(row["anchor_view_id"]),
                "pose_source": str(row["pose_source"]),
                "split_name": split,
                "query_ids": sorted(row["query_ids"]),
                "alpha_valid_ratio": max(0.0, min(1.0, 1.0 - artifact_score)),
                "stable_mask_ratio": max(0.0, min(1.0, 1.0 - artifact_score)),
                "selected_landmark_visible_count": int(len(row["selected_landmark_ids"])),
                "candidate_landmark_visible_count": int(len(row["candidate_landmark_ids"])),
                "feature_variance": float(feature_variance),
                "artifact_score": float(artifact_score),
                "novelty_score": float(min(1.0, len(row["query_ids"]) / 4.0)),
                "expected_query_gain": float(row["positive_support"]),
                "covered_cells": [list(cell) for cell in sorted(row["covered_cells"])],
                "covered_depth_bins": sorted(row["covered_depth_bins"]),
            }
        )
    return candidates


def _quality(record: Mapping[str, Any], reprojection_threshold_px: float) -> float:
    descriptor = _clamp01(record.get("descriptor_score"), 1.0)
    margin = _clamp01(record.get("descriptor_margin"), 0.5)
    local_geometry = _clamp01(record.get("local_geometry_score"), 1.0)
    reproj = _float(record.get("reprojection_error_px"), 0.0)
    reproj_quality = max(0.0, min(1.0, 1.0 - reproj / max(float(reprojection_threshold_px), 1e-6)))
    return float(descriptor * (0.5 + 0.5 * margin) * local_geometry * reproj_quality)


def _artifact(record: Mapping[str, Any]) -> float:
    return _clamp01(record.get("ray_artifact_score", record.get("artifact_score", record.get("artifact_risk"))), 0.0)


def build_view_landmark_reliability(
    observations: Iterable[Mapping[str, Any]],
    *,
    split_name: str,
    num_gaussians: int,
    reprojection_threshold_px: float = 4.0,
    min_positive_observations: int = 2,
    max_negative_observations: int = 0,
    max_fusion_artifact_score: float = 0.3,
    compact: bool = False,
    reliability_sample_limit: int = 0,
) -> dict[str, Any]:
    split = validate_split_name(split_name)
    count = int(num_gaussians)
    if count < 0:
        raise ValueError(f"num_gaussians must be non-negative, got {count}")

    rows: dict[tuple[str, int], dict[str, Any]] = {}
    query_ids: set[str] = set()
    source_views: set[str] = set()
    observation_count = 0
    for record in observations:
        _record_split(record, split)
        normalized = normalize_contributors(_contributors(record))
        if not normalized:
            continue
        observation_count += 1
        query_id = str(record.get("query_id", record.get("image_id", ""))).strip()
        source_view_id = str(record.get("source_view_id", record.get("view_id", ""))).strip()
        if query_id:
            query_ids.add(query_id)
        if source_view_id:
            source_views.add(source_view_id)
        source_view_id = source_view_id or "unknown_view"
        pnp_inlier = _bool(record.get("pnp_inlier", False))
        reproj = _float(record.get("reprojection_error_px"), 0.0)
        quality = _quality(record, float(reprojection_threshold_px)) if pnp_inlier else 0.0
        artifact = _artifact(record)
        high_score_outlier = (not pnp_inlier) and _float(record.get("descriptor_score"), 0.0) >= 0.5
        bad_reproj = reproj > float(reprojection_threshold_px)
        for item in normalized:
            gid = int(item["gaussian_id"])
            if gid >= count:
                raise ValueError(f"gaussian_id {gid} is outside num_gaussians={count}")
            key = (source_view_id, gid)
            row = rows.setdefault(
                key,
                {
                    "source_view_id": source_view_id,
                    "gaussian_id": gid,
                    "positive_observations": 0,
                    "negative_observations": 0,
                    "support_mass": 0.0,
                    "artifact_mass": 0.0,
                    "reprojection_error_sum": 0.0,
                    "reprojection_error_count": 0,
                    "query_ids": set(),
                },
            )
            weight = float(item["weight"])
            if query_id:
                row["query_ids"].add(query_id)
            row["artifact_mass"] = float(row["artifact_mass"]) + artifact * weight
            if pnp_inlier and quality > 0.0 and artifact <= float(max_fusion_artifact_score):
                row["positive_observations"] = int(row["positive_observations"]) + 1
                row["support_mass"] = float(row["support_mass"]) + quality * weight
                row["reprojection_error_sum"] = float(row["reprojection_error_sum"]) + reproj
                row["reprojection_error_count"] = int(row["reprojection_error_count"]) + 1
            if high_score_outlier or artifact > float(max_fusion_artifact_score) or (pnp_inlier and bad_reproj):
                row["negative_observations"] = int(row["negative_observations"]) + 1

    reliability_rows: list[dict[str, Any]] = []
    reliability_sample: list[dict[str, Any]] = []
    fusion_plan: dict[str, dict[str, Any]] = {}
    view_landmark_pair_count = 0
    selected_pair_count = 0
    deboosted_pair_count = 0
    for (_source_view_id, gid), row in sorted(rows.items()):
        reproj_count = int(row["reprojection_error_count"])
        mean_reproj = float(row["reprojection_error_sum"]) / reproj_count if reproj_count else None
        positive = int(row["positive_observations"])
        negative = int(row["negative_observations"])
        status = "ignored"
        if (
            positive >= int(min_positive_observations)
            and negative <= int(max_negative_observations)
            and float(row["artifact_mass"]) <= float(max_fusion_artifact_score)
        ):
            status = "selected_for_fusion"
        elif negative > int(max_negative_observations) or float(row["artifact_mass"]) > float(max_fusion_artifact_score):
            status = "deboosted"
        output_row = {
            "source_view_id": str(row["source_view_id"]),
            "gaussian_id": int(gid),
            "status": status,
            "positive_observations": positive,
            "negative_observations": negative,
            "support_mass": float(row["support_mass"]),
            "artifact_mass": float(row["artifact_mass"]),
            "mean_reprojection_error_px": mean_reproj,
            "query_ids": sorted(row["query_ids"]),
        }
        view_landmark_pair_count += 1
        if status == "selected_for_fusion":
            selected_pair_count += 1
        elif status == "deboosted":
            deboosted_pair_count += 1
        if compact:
            if int(reliability_sample_limit) > 0 and len(reliability_sample) < int(reliability_sample_limit):
                reliability_sample.append(output_row)
            elif (
                int(reliability_sample_limit) > 0
                and status == "selected_for_fusion"
                and not any(row["status"] == "selected_for_fusion" for row in reliability_sample)
            ):
                reliability_sample[-1] = output_row
        else:
            reliability_rows.append(output_row)
        if status == "selected_for_fusion":
            plan = fusion_plan.setdefault(
                str(gid),
                {
                    "gaussian_id": int(gid),
                    "selected_view_ids": [],
                    "support_mass": 0.0,
                    "positive_observations": 0,
                },
            )
            plan["selected_view_ids"].append(str(row["source_view_id"]))
            plan["support_mass"] = float(plan["support_mass"]) + float(row["support_mass"])
            plan["positive_observations"] = int(plan["positive_observations"]) + positive

    for plan in fusion_plan.values():
        plan["selected_view_ids"] = sorted(plan["selected_view_ids"])

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "split_name": split,
        "num_gaussians": int(count),
        "observation_count": int(observation_count),
        "query_count": int(len(query_ids)),
        "source_view_count": int(len(source_views)),
        "view_landmark_pair_count": int(view_landmark_pair_count),
        "selected_fusion_view_landmark_pairs": int(selected_pair_count),
        "deboosted_view_landmark_pairs": int(deboosted_pair_count),
        "fusion_landmark_count": int(len(fusion_plan)),
        "compact_output": bool(compact),
        "view_landmark_reliability_sample_count": int(len(reliability_sample)),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "split_name": split,
        "view_landmark_reliability": reliability_rows,
        "view_landmark_reliability_sample": reliability_sample,
        "landmark_fusion_plan": fusion_plan,
        "metrics": metrics,
        "hyperparameters": {
            "reprojection_threshold_px": float(reprojection_threshold_px),
            "min_positive_observations": int(min_positive_observations),
            "max_negative_observations": int(max_negative_observations),
            "max_fusion_artifact_score": float(max_fusion_artifact_score),
            "compact": bool(compact),
            "reliability_sample_limit": int(reliability_sample_limit),
        },
    }


def build_active_pose_augmentation(
    *,
    candidate_poses: Iterable[Mapping[str, Any]],
    observations: Iterable[Mapping[str, Any]],
    num_gaussians: int,
    scene: str,
    split_name: str,
    hard_query_ids: Iterable[str] | None = None,
    max_poses: int = 32,
    max_per_anchor: int = 4,
    min_alpha_valid: float = 0.5,
    min_stable_mask_ratio: float = 0.0,
    min_selected_landmark_visible: int = 1,
    min_candidate_landmark_visible: int = 1,
    min_feature_variance: float = 0.0,
    max_artifact_score: float = 0.5,
    reprojection_threshold_px: float = 4.0,
    min_positive_observations: int = 2,
    max_negative_observations: int = 0,
    max_fusion_artifact_score: float = 0.3,
    compact: bool = False,
    reliability_sample_limit: int = 0,
) -> dict[str, Any]:
    split = validate_split_name(split_name)
    pose_result = select_active_pose_candidates(
        candidate_poses,
        split_name=split,
        hard_query_ids=hard_query_ids,
        max_poses=max_poses,
        max_per_anchor=max_per_anchor,
        min_alpha_valid=min_alpha_valid,
        min_stable_mask_ratio=min_stable_mask_ratio,
        min_selected_landmark_visible=min_selected_landmark_visible,
        min_candidate_landmark_visible=min_candidate_landmark_visible,
        min_feature_variance=min_feature_variance,
        max_artifact_score=max_artifact_score,
    )
    reliability = build_view_landmark_reliability(
        observations,
        split_name=split,
        num_gaussians=int(num_gaussians),
        reprojection_threshold_px=reprojection_threshold_px,
        min_positive_observations=min_positive_observations,
        max_negative_observations=max_negative_observations,
        max_fusion_artifact_score=max_fusion_artifact_score,
        compact=compact,
        reliability_sample_limit=reliability_sample_limit,
    )
    metrics = {
        **dict(pose_result["metrics"]),
        **{
            key: value
            for key, value in reliability["metrics"].items()
            if key not in {"schema_version", "split_name", "num_gaussians"}
        },
        "num_gaussians": int(num_gaussians),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scene": str(scene),
        "split_name": split,
        "selected_poses": pose_result["selected_poses"],
        "rejected_poses": pose_result["rejected_poses"],
        "view_landmark_reliability": reliability["view_landmark_reliability"],
        "view_landmark_reliability_sample": reliability["view_landmark_reliability_sample"],
        "landmark_fusion_plan": reliability["landmark_fusion_plan"],
        "metrics": metrics,
        "hyperparameters": {
            "pose_selection": pose_result["hyperparameters"],
            "view_landmark_reliability": reliability["hyperparameters"],
        },
    }
