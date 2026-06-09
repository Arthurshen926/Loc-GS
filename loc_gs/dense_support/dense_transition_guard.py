from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from loc_gs.diagnostics.match_visualization import project_points


@dataclass(frozen=True)
class DenseTransitionGuardPolicy:
    """Observable sparse-to-dense transition policy.

    The default is intentionally conservative for paper safety: weak or missing
    sparse evidence records risk but does not veto dense. High-confidence sparse
    evidence can reject a dense update when observable consistency checks fail.
    """

    high_confidence_min_inliers: int = 80
    high_confidence_min_inlier_ratio: float = 0.0
    high_confidence_min_score: float = 0.75
    max_translation_delta_m: float = 0.35
    max_rotation_delta_deg: float = 5.0
    max_sparse_reprojection_error_px: float = 8.0
    max_sparse_reprojection_worsening_px: float = 2.0
    min_sparse_reprojection_retained_ratio: float = 0.90
    min_dense_valid_match_ratio: float = 0.25
    max_artifact_depth_risk_score: float = 0.65
    max_sparse_anchor_count: int = 256
    reject_weak_sparse_failures: bool = False


def _policy_dict(policy: DenseTransitionGuardPolicy) -> dict[str, Any]:
    return {
        key: (int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value)
        for key, value in asdict(policy).items()
    }


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _finite_int(value: Any) -> int | None:
    parsed = _finite_float(value)
    return None if parsed is None else int(parsed)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested_get(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = mapping
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _first_float(mapping: Mapping[str, Any], paths: Sequence[str | Sequence[str]]) -> float | None:
    for path in paths:
        raw = _nested_get(mapping, tuple(path.split("."))) if isinstance(path, str) else _nested_get(mapping, path)
        parsed = _finite_float(raw)
        if parsed is not None:
            return parsed
    return None


def _first_int(mapping: Mapping[str, Any], paths: Sequence[str | Sequence[str]]) -> int | None:
    for path in paths:
        raw = _nested_get(mapping, tuple(path.split("."))) if isinstance(path, str) else _nested_get(mapping, path)
        parsed = _finite_int(raw)
        if parsed is not None:
            return parsed
    return None


def _pose_w2c(value: Any, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.size != 16:
        raise ValueError(f"{name} must be a 4x4 world-to-camera pose")
    return pose.reshape(4, 4)


def _camera_center(pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    return -pose[:3, :3].T @ pose[:3, 3]


def _rotation_delta_deg(a_w2c: np.ndarray, b_w2c: np.ndarray) -> float:
    a = np.asarray(a_w2c, dtype=np.float64).reshape(4, 4)[:3, :3]
    b = np.asarray(b_w2c, dtype=np.float64).reshape(4, 4)[:3, :3]
    relative = a @ b.T
    cos_angle = float((np.trace(relative) - 1.0) * 0.5)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _array_from_keys(mapping: Mapping[str, Any], keys: Sequence[str], shape_tail: tuple[int, ...]) -> np.ndarray | None:
    for key in keys:
        if key not in mapping:
            continue
        array = np.asarray(mapping[key], dtype=np.float64)
        if array.size == 0:
            return np.empty((0, *shape_tail), dtype=np.float64)
        try:
            return array.reshape(-1, *shape_tail)
        except ValueError as exc:
            raise ValueError(f"{key} must have trailing shape {shape_tail}") from exc
    return None


def _selected_indices(sparse_inliers: Mapping[str, Any], count: int, max_count: int) -> np.ndarray:
    if "inliers" in sparse_inliers:
        indices = np.asarray(sparse_inliers["inliers"], dtype=np.int64).reshape(-1)
    elif "sparse_inlier_indices" in sparse_inliers:
        indices = np.asarray(sparse_inliers["sparse_inlier_indices"], dtype=np.int64).reshape(-1)
    else:
        indices = np.arange(count, dtype=np.int64)
    valid = indices[(indices >= 0) & (indices < count)]
    if int(max_count) > 0 and valid.shape[0] > int(max_count):
        valid = valid[: int(max_count)]
    return valid


def _inlier_count(sparse_inliers: Mapping[str, Any], query_xy: np.ndarray | None, points: np.ndarray | None) -> int:
    explicit = _first_int(
        sparse_inliers,
        (
            "sparse_inlier_count",
            "solver_inlier_count",
            "inlier_count",
            "num_inliers",
            "inliers_count",
        ),
    )
    if explicit is not None:
        return max(0, int(explicit))
    if "inliers" in sparse_inliers:
        return int(np.asarray(sparse_inliers["inliers"]).reshape(-1).shape[0])
    if query_xy is None or points is None:
        return 0
    return int(min(query_xy.shape[0], points.shape[0]))


def _inlier_ratio(sparse_inliers: Mapping[str, Any], count: int) -> float | None:
    ratio = _first_float(
        sparse_inliers,
        (
            "sparse_inlier_ratio",
            "solver_inlier_ratio",
            "inlier_ratio",
            "inlier_fraction",
        ),
    )
    if ratio is not None:
        return float(np.clip(ratio, 0.0, 1.0))
    total = _first_int(sparse_inliers, ("total_match_count", "match_count", "num_matches", "candidate_count"))
    if total is not None and total > 0:
        return float(np.clip(float(count) / float(total), 0.0, 1.0))
    return None


def _sparse_confidence_score(sparse_inliers: Mapping[str, Any]) -> float | None:
    score = _first_float(
        sparse_inliers,
        (
            "sparse_confidence_score",
            "pose_confidence",
            "confidence",
            "sparse_pose_confidence",
        ),
    )
    if score is not None:
        return float(np.clip(score, 0.0, 1.0))
    nested = sparse_inliers.get("sparse_confidence")
    if isinstance(nested, Mapping):
        return _first_float(nested, ("confidence", "score", "sparse_pose_confidence"))
    return None


def _sparse_support(
    sparse_inliers: Mapping[str, Any],
    *,
    query_xy: np.ndarray | None,
    points: np.ndarray | None,
    policy: DenseTransitionGuardPolicy,
) -> dict[str, Any]:
    count = _inlier_count(sparse_inliers, query_xy, points)
    ratio = _inlier_ratio(sparse_inliers, count)
    score = _sparse_confidence_score(sparse_inliers)
    ratio_ok = ratio is None or ratio >= float(policy.high_confidence_min_inlier_ratio)
    count_high = count >= int(policy.high_confidence_min_inliers) and ratio_ok
    score_high = score is not None and score >= float(policy.high_confidence_min_score)
    if count <= 0 and score is None:
        strength = "none"
    elif count_high or score_high:
        strength = "high"
    else:
        strength = "weak"
    return {
        "strength": strength,
        "inlier_count": int(count),
        "inlier_ratio": ratio,
        "confidence_score": score,
        "high_confidence_min_inliers": int(policy.high_confidence_min_inliers),
        "high_confidence_min_inlier_ratio": float(policy.high_confidence_min_inlier_ratio),
    }


def _image_size(sparse_inliers: Mapping[str, Any], query_xy: np.ndarray | None) -> tuple[int, int] | None:
    raw = sparse_inliers.get("image_size")
    if raw is not None:
        values = np.asarray(raw).reshape(-1)
        if values.shape[0] >= 2:
            return max(1, int(values[0])), max(1, int(values[1]))
    width = _first_int(sparse_inliers, ("image_width", "width"))
    height = _first_int(sparse_inliers, ("image_height", "height"))
    if width is not None and height is not None:
        return max(1, int(width)), max(1, int(height))
    if query_xy is not None and query_xy.size:
        finite = query_xy[np.isfinite(query_xy).all(axis=1)]
        if finite.size:
            return (
                max(1, int(np.ceil(float(finite[:, 0].max()) + 2.0))),
                max(1, int(np.ceil(float(finite[:, 1].max()) + 2.0))),
            )
    return None


def _sparse_reprojection_check(
    *,
    sparse_pose: np.ndarray,
    dense_pose: np.ndarray,
    sparse_inliers: Mapping[str, Any],
    query_xy: np.ndarray | None,
    points: np.ndarray | None,
    policy: DenseTransitionGuardPolicy,
) -> tuple[dict[str, Any], bool]:
    intrinsic = _array_from_keys(sparse_inliers, ("intrinsic", "K", "camera_intrinsic"), (3, 3))
    if query_xy is None or points is None or intrinsic is None:
        return {
            "status": "unavailable",
            "reason": "missing_query_points_or_intrinsic",
            "retained_ratio": None,
            "evaluated_count": 0,
        }, False
    count = min(query_xy.shape[0], points.shape[0])
    if count <= 0:
        return {"status": "unavailable", "reason": "no_sparse_anchors", "retained_ratio": None, "evaluated_count": 0}, False
    selected = _selected_indices(sparse_inliers, count, int(policy.max_sparse_anchor_count))
    if selected.shape[0] == 0:
        return {"status": "unavailable", "reason": "no_selected_sparse_anchors", "retained_ratio": None, "evaluated_count": 0}, False
    selected_xy = np.asarray(query_xy[selected], dtype=np.float64).reshape(-1, 2)
    selected_points = np.asarray(points[selected], dtype=np.float64).reshape(-1, 3)
    image_size = _image_size(sparse_inliers, selected_xy)
    if image_size is None:
        return {"status": "unavailable", "reason": "missing_image_size", "retained_ratio": None, "evaluated_count": 0}, False
    sparse_projected, sparse_valid = project_points(
        selected_points,
        sparse_pose,
        intrinsic.reshape(3, 3),
        width=int(image_size[0]),
        height=int(image_size[1]),
    )
    dense_projected, dense_valid = project_points(
        selected_points,
        dense_pose,
        intrinsic.reshape(3, 3),
        width=int(image_size[0]),
        height=int(image_size[1]),
    )
    sparse_errors = np.linalg.norm(np.asarray(sparse_projected, dtype=np.float64) - selected_xy, axis=1)
    dense_errors = np.linalg.norm(np.asarray(dense_projected, dtype=np.float64) - selected_xy, axis=1)
    sparse_reference = sparse_valid & np.isfinite(sparse_errors)
    dense_finite = dense_valid & np.isfinite(dense_errors)
    tolerances = np.maximum(
        float(policy.max_sparse_reprojection_error_px),
        sparse_errors + float(policy.max_sparse_reprojection_worsening_px),
    )
    retained = sparse_reference & dense_finite & (dense_errors <= tolerances)
    denominator = int(sparse_reference.sum())
    retained_ratio = float(retained.sum() / denominator) if denominator > 0 else None
    failed = retained_ratio is not None and retained_ratio < float(policy.min_sparse_reprojection_retained_ratio)
    finite_sparse = sparse_errors[sparse_reference]
    finite_dense = dense_errors[dense_finite]
    return {
        "status": "ok" if denominator > 0 else "unavailable",
        "reason": None if denominator > 0 else "no_valid_sparse_pose_projection",
        "candidate_anchor_count": int(count),
        "evaluated_count": int(selected.shape[0]),
        "valid_sparse_anchor_count": int(denominator),
        "retained_count": int(retained.sum()),
        "retained_ratio": retained_ratio,
        "min_retained_ratio": float(policy.min_sparse_reprojection_retained_ratio),
        "sparse_median_error_px": float(np.median(finite_sparse)) if finite_sparse.size else None,
        "dense_median_error_px": float(np.median(finite_dense)) if finite_dense.size else None,
        "max_sparse_reprojection_error_px": float(policy.max_sparse_reprojection_error_px),
        "max_sparse_reprojection_worsening_px": float(policy.max_sparse_reprojection_worsening_px),
    }, bool(failed)


def _pose_delta_check(
    sparse_pose: np.ndarray,
    dense_pose: np.ndarray,
    policy: DenseTransitionGuardPolicy,
) -> tuple[dict[str, Any], dict[str, bool]]:
    translation_delta = float(np.linalg.norm(_camera_center(sparse_pose) - _camera_center(dense_pose)))
    rotation_delta = _rotation_delta_deg(sparse_pose, dense_pose)
    failed = {
        "pose_translation_trust_region": bool(translation_delta > float(policy.max_translation_delta_m)),
        "pose_rotation_trust_region": bool(rotation_delta > float(policy.max_rotation_delta_deg)),
    }
    return {
        "translation_delta_m": translation_delta,
        "rotation_delta_deg": rotation_delta,
        "max_translation_delta_m": float(policy.max_translation_delta_m),
        "max_rotation_delta_deg": float(policy.max_rotation_delta_deg),
    }, failed


def _valid_match_ratio(dense_stats: Mapping[str, Any]) -> float | None:
    ratio = _first_float(
        dense_stats,
        (
            "valid_dense_match_ratio",
            "dense_valid_match_ratio",
            "valid_match_ratio",
            "metadata.valid_dense_match_ratio",
            "verifier.metadata.valid_dense_match_ratio",
        ),
    )
    if ratio is not None:
        return float(np.clip(ratio, 0.0, 1.0))
    kept = _first_float(dense_stats, ("kept_count", "valid_dense_match_count", "metadata.kept_count"))
    total = _first_float(dense_stats, ("input_count", "dense_match_count", "metadata.input_count"))
    if kept is not None and total is not None and total > 0.0:
        return float(np.clip(kept / total, 0.0, 1.0))
    return None


def _artifact_depth_risk_score(dense_stats: Mapping[str, Any]) -> float | None:
    direct = _first_float(
        dense_stats,
        (
            "artifact_depth_risk_score",
            "artifact_depth_risk",
            "depth_risk_score",
            "artifact_score",
            "render_artifact_score",
            "dense_damage_risk_score",
            "dense_damage_risk.risk",
            "artifact_depth_risk.risk",
            "render_reliability.risk",
            "risk",
        ),
    )
    ratio_candidates: list[float] = []
    for key in (
        "near_occluder_ratio",
        "far_surface_or_hole_ratio",
        "missing_depth_ratio",
        "preflight.near_occluder_ratio",
        "preflight.far_surface_or_hole_ratio",
        "preflight.missing_depth_ratio",
    ):
        value = _first_float(dense_stats, (key,))
        if value is not None:
            ratio_candidates.append(float(np.clip(value, 0.0, 1.0)))
    if direct is not None:
        ratio_candidates.append(float(np.clip(direct, 0.0, 1.0)))
    if not ratio_candidates:
        return None
    return float(max(ratio_candidates))


def _dense_stats_check(
    dense_stats: Mapping[str, Any],
    policy: DenseTransitionGuardPolicy,
) -> tuple[dict[str, Any], dict[str, bool]]:
    valid_ratio = _valid_match_ratio(dense_stats)
    risk_score = _artifact_depth_risk_score(dense_stats)
    failed = {
        "dense_valid_match_ratio": bool(
            valid_ratio is not None and valid_ratio < float(policy.min_dense_valid_match_ratio)
        ),
        "artifact_depth_risk": bool(
            risk_score is not None and risk_score > float(policy.max_artifact_depth_risk_score)
        ),
    }
    return {
        "valid_dense_match_ratio": valid_ratio,
        "min_dense_valid_match_ratio": float(policy.min_dense_valid_match_ratio),
        "artifact_depth_risk_score": risk_score,
        "max_artifact_depth_risk_score": float(policy.max_artifact_depth_risk_score),
    }, failed


def _reasons(failed_checks: Mapping[str, bool], sparse_strength: str, reject: bool) -> list[str]:
    reasons: list[str] = []
    if sparse_strength == "none":
        reasons.append("no_sparse_evidence")
    elif sparse_strength == "weak":
        reasons.append("weak_sparse_evidence")
    for key, failed in failed_checks.items():
        if failed:
            reasons.append(f"{key}_failed")
    if reject:
        reasons.append("high_confidence_sparse_veto")
    elif not any(failed_checks.values()):
        reasons.append("dense_transition_within_observable_trust_region")
    else:
        reasons.append("dense_failures_recorded_without_sparse_veto")
    return reasons


def should_accept_dense(
    sparse_pose: Any,
    dense_pose: Any,
    sparse_inliers: Mapping[str, Any] | None,
    dense_stats: Mapping[str, Any] | None,
    policy: DenseTransitionGuardPolicy = DenseTransitionGuardPolicy(),
) -> dict[str, Any]:
    """Decide whether a dense pose should replace a sparse pose without GT.

    Inputs must come from inference-observable state: sparse PnP inlier
    geometry, the proposed dense pose, and dense matching/render diagnostics.
    """

    sparse = _pose_w2c(sparse_pose, "sparse_pose")
    dense = _pose_w2c(dense_pose, "dense_pose")
    sparse_map = _mapping(sparse_inliers)
    dense_map = _mapping(dense_stats)
    query_xy = _array_from_keys(sparse_map, ("query_xy", "sparse_query_xy", "keypoints_xy"), (2,))
    points = _array_from_keys(sparse_map, ("p3d", "points_world", "sparse_points_world", "xyz"), (3,))
    support = _sparse_support(sparse_map, query_xy=query_xy, points=points, policy=policy)
    pose_delta, pose_failed = _pose_delta_check(sparse, dense, policy)
    sparse_reprojection, sparse_reprojection_failed = _sparse_reprojection_check(
        sparse_pose=sparse,
        dense_pose=dense,
        sparse_inliers=sparse_map,
        query_xy=query_xy,
        points=points,
        policy=policy,
    )
    dense_observables, dense_failed = _dense_stats_check(dense_map, policy)
    failed_checks = {
        **pose_failed,
        "sparse_reprojection_preservation": bool(sparse_reprojection_failed),
        **dense_failed,
    }
    sparse_strength = str(support["strength"])
    can_veto = sparse_strength == "high" or (
        sparse_strength == "weak" and bool(policy.reject_weak_sparse_failures)
    )
    reject = bool(can_veto and any(failed_checks.values()))
    return {
        "schema": "loc_gs_dense_transition_guard_v1",
        "decision": "reject_dense_keep_sparse" if reject else "accept_dense",
        "selected_pose_source": "sparse" if reject else "dense",
        "sparse_support_strength": sparse_strength,
        "dense_worsened_proxy": bool(sparse_strength == "high" and any(failed_checks.values())),
        "reasons": _reasons(failed_checks, sparse_strength, reject),
        "failed_checks": failed_checks,
        "checks": {
            "sparse_support": support,
            "pose_delta": pose_delta,
            "sparse_reprojection": sparse_reprojection,
            "dense_observables": dense_observables,
        },
        "policy": _policy_dict(policy),
        "uses_gt": False,
        "inference_observable_only": True,
        "diagnostic_only": True,
        "paper_safe_main_method": False,
    }


def _query_id(row: Mapping[str, Any], index: int) -> str:
    for key in ("query_id", "image_name", "image_id", "name"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return f"query_{index:06d}"


def _summary_support(row: Mapping[str, Any], policy: DenseTransitionGuardPolicy) -> dict[str, Any]:
    sparse_map = {
        "sparse_inlier_count": _first_int(row, ("sparse_inlier_count", "solver_inlier_count", "inlier_count")) or 0,
        "sparse_inlier_ratio": _first_float(row, ("sparse_inlier_ratio", "solver_inlier_ratio", "inlier_ratio")),
    }
    score = _first_float(row, ("sparse_confidence_score", "pose_confidence", "confidence"))
    if score is not None:
        sparse_map["confidence"] = score
    return _sparse_support(sparse_map, query_xy=None, points=None, policy=policy)


def _summary_decision(row: Mapping[str, Any], policy: DenseTransitionGuardPolicy) -> dict[str, Any]:
    support = _summary_support(row, policy)
    translation_delta = _first_float(row, ("translation_delta_m", "pose_translation_delta_m"))
    rotation_delta = _first_float(row, ("rotation_delta_deg", "pose_rotation_delta_deg"))
    retained_ratio = _first_float(
        row,
        (
            "retained_sparse_inlier_ratio",
            "sparse_inlier_retained_ratio",
            "sparse_reprojection_retained_ratio",
            "checks.sparse_reprojection.retained_ratio",
        ),
    )
    dense_observables, dense_failed = _dense_stats_check(row, policy)
    failed_checks = {
        "pose_translation_trust_region": bool(
            translation_delta is not None and translation_delta > float(policy.max_translation_delta_m)
        ),
        "pose_rotation_trust_region": bool(
            rotation_delta is not None and rotation_delta > float(policy.max_rotation_delta_deg)
        ),
        "sparse_reprojection_preservation": bool(
            retained_ratio is not None and retained_ratio < float(policy.min_sparse_reprojection_retained_ratio)
        ),
        **dense_failed,
    }
    sparse_strength = str(support["strength"])
    can_veto = sparse_strength == "high" or (
        sparse_strength == "weak" and bool(policy.reject_weak_sparse_failures)
    )
    reject = bool(can_veto and any(failed_checks.values()))
    return {
        "schema": "loc_gs_dense_transition_guard_v1",
        "decision": "reject_dense_keep_sparse" if reject else "accept_dense",
        "selected_pose_source": "sparse" if reject else "dense",
        "sparse_support_strength": sparse_strength,
        "dense_worsened_proxy": bool(sparse_strength == "high" and any(failed_checks.values())),
        "reasons": _reasons(failed_checks, sparse_strength, reject),
        "failed_checks": failed_checks,
        "checks": {
            "sparse_support": support,
            "pose_delta": {
                "translation_delta_m": translation_delta,
                "rotation_delta_deg": rotation_delta,
                "max_translation_delta_m": float(policy.max_translation_delta_m),
                "max_rotation_delta_deg": float(policy.max_rotation_delta_deg),
            },
            "sparse_reprojection": {
                "status": "summary",
                "retained_ratio": retained_ratio,
                "min_retained_ratio": float(policy.min_sparse_reprojection_retained_ratio),
            },
            "dense_observables": dense_observables,
        },
        "policy": _policy_dict(policy),
        "uses_gt": False,
        "inference_observable_only": True,
        "diagnostic_only": True,
        "paper_safe_main_method": False,
        "source": "summary_record",
    }


def _existing_guard(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("dense_transition_guard", "transition_guard", "guard"):
        value = row.get(key)
        if isinstance(value, Mapping) and "decision" in value:
            return value
    if "decision" in row and "failed_checks" in row:
        return row
    return None


def evaluate_dense_transition_record(
    row: Mapping[str, Any],
    *,
    policy: DenseTransitionGuardPolicy = DenseTransitionGuardPolicy(),
) -> dict[str, Any]:
    guard = _existing_guard(row)
    if guard is not None:
        decision = dict(guard)
        decision.setdefault("schema", "loc_gs_dense_transition_guard_v1")
        decision.setdefault("uses_gt", False)
        decision.setdefault("diagnostic_only", True)
        decision.setdefault("inference_observable_only", True)
        decision.setdefault("dense_worsened_proxy", decision.get("decision") == "reject_dense_keep_sparse")
        decision.setdefault("source", "embedded_guard")
        return decision
    if all(key in row for key in ("sparse_pose", "dense_pose")):
        return should_accept_dense(
            row["sparse_pose"],
            row["dense_pose"],
            _mapping(row.get("sparse_inliers")),
            _mapping(row.get("dense_stats")),
            policy=policy,
        )
    return _summary_decision(row, policy)


def build_dense_transition_report(
    records: Sequence[Mapping[str, Any]],
    *,
    policy: DenseTransitionGuardPolicy = DenseTransitionGuardPolicy(),
    scene: str = "unknown",
    split_name: str = "unknown",
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    failed_check_counts: Counter[str] = Counter()
    for index, row in enumerate(records):
        decision = evaluate_dense_transition_record(row, policy=policy)
        decision = dict(decision)
        decision["query_index"] = int(index)
        decision["query_id"] = _query_id(row, index)
        decisions.append(decision)
        reason_counts.update(str(reason) for reason in decision.get("reasons", []))
        failed_check_counts.update(
            key for key, failed in _mapping(decision.get("failed_checks")).items() if bool(failed)
        )
    accepted = sum(1 for decision in decisions if decision.get("decision") == "accept_dense")
    rejected = sum(1 for decision in decisions if decision.get("decision") == "reject_dense_keep_sparse")
    proxy = sum(1 for decision in decisions if bool(decision.get("dense_worsened_proxy")))
    return {
        "schema": "loc_gs_dense_transition_guard_report_v1",
        "scene": str(scene),
        "split_name": str(split_name),
        "uses_gt": False,
        "inference_observable_only": True,
        "diagnostic_only": True,
        "paper_safe_main_method": False,
        "summary": {
            "record_count": int(len(records)),
            "accepted_count": int(accepted),
            "rejected_count": int(rejected),
            "dense_worsened_proxy_count": int(proxy),
            "reason_counts": dict(sorted(reason_counts.items())),
            "failed_check_counts": dict(sorted(failed_check_counts.items())),
        },
        "policy": _policy_dict(policy),
        "decisions": decisions,
    }


def load_dense_transition_records(path: str | Any) -> list[dict[str, Any]]:
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = None
        for key in ("results", "records", "queries", "decisions"):
            value = payload.get(key)
            if isinstance(value, list):
                rows = value
                break
        if rows is None:
            rows = [payload]
    else:
        raise TypeError(f"dense transition records must be a list or JSON object: {path}")
    return [dict(row) for row in rows if isinstance(row, Mapping)]
