from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from loc_gs.diagnostics.match_visualization import project_points


@dataclass(frozen=True)
class AnchorConditionedPatchDensePolicy:
    """Policy for combining query-side dense patches with sparse anchors."""

    anchor_weight: float = 3.0
    anchor_max_count: int = 64
    anchor_max_reprojection_px: float = 8.0
    anchor_target_weight_fraction: float = 0.0
    anchor_max_weight: float = 256.0
    min_refine_median_gain_px: float = 0.05
    min_refine_p90_gain_px: float = 0.10
    max_refine_translation_without_gain_m: float = 0.01
    max_refine_rotation_without_gain_deg: float = 0.05


def _finite_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _effective_anchor_weight(
    *,
    dense_match_count: int,
    selected_anchor_count: int,
    policy: AnchorConditionedPatchDensePolicy,
) -> float:
    base = max(0.0, float(policy.anchor_weight))
    anchor_count = max(0, int(selected_anchor_count))
    if anchor_count <= 0:
        return base
    target_fraction = min(max(0.0, float(policy.anchor_target_weight_fraction)), 0.95)
    if target_fraction > 0.0:
        dense_mass = float(max(0, int(dense_match_count)))
        target_anchor_mass = dense_mass * target_fraction / max(1.0e-6, 1.0 - target_fraction)
        base = max(base, target_anchor_mass / float(anchor_count))
    max_weight = _finite_float(policy.anchor_max_weight, default=None)
    if max_weight is not None and max_weight > 0.0:
        base = min(base, float(max_weight))
    return float(base)


def _reprojection_errors(
    *,
    query_xy: np.ndarray,
    points_world: np.ndarray,
    pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    projected, valid = project_points(
        points_world,
        pose_w2c,
        intrinsic,
        width=int(image_size[0]),
        height=int(image_size[1]),
    )
    errors = np.linalg.norm(np.asarray(projected, dtype=np.float64) - np.asarray(query_xy, dtype=np.float64), axis=1)
    errors[~valid] = np.inf
    return errors.astype(np.float64)


def build_anchor_conditioned_refinement_matches(
    dense_merged: Mapping[str, Any],
    sparse_capture: Mapping[str, Any],
    *,
    reference_pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    policy: AnchorConditionedPatchDensePolicy = AnchorConditionedPatchDensePolicy(),
) -> dict[str, Any]:
    """Add reference-consistent sparse anchors to dense patch refinement matches.

    Dense patch matches handle query-side occlusion/local corruption. Sparse
    anchors keep the local patch solution globally tied to the PnP support.
    """

    dense_xy = np.asarray(dense_merged.get("xy", np.empty((0, 2), dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
    dense_xyz = np.asarray(dense_merged.get("xyz", np.empty((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
    dense_weights = np.ones((dense_xy.shape[0],), dtype=np.float32)
    diagnostics: dict[str, Any] = {
        "schema": "loc_gs_acpd_sparse_anchor_refinement_v1",
        "enabled": bool(float(policy.anchor_weight) > 0.0),
        "dense_match_count": int(dense_xy.shape[0]),
        "candidate_anchor_count": 0,
        "selected_anchor_count": 0,
        "sparse_anchor_weight": float(policy.anchor_weight),
        "effective_sparse_anchor_weight": float(policy.anchor_weight),
        "sparse_anchor_max_count": int(policy.anchor_max_count),
        "sparse_anchor_max_reprojection_px": float(policy.anchor_max_reprojection_px),
        "sparse_anchor_target_weight_fraction": float(policy.anchor_target_weight_fraction),
        "sparse_anchor_max_weight": float(policy.anchor_max_weight),
        "diagnostic_only": True,
    }
    if float(policy.anchor_weight) <= 0.0:
        return {"xy": dense_xy, "xyz": dense_xyz, "weights": dense_weights, "diagnostics": diagnostics}

    sparse_xy_all = np.asarray(sparse_capture.get("query_xy", np.empty((0, 2), dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
    sparse_xyz_all = np.asarray(sparse_capture.get("p3d", np.empty((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
    sparse_inliers = np.asarray(sparse_capture.get("inliers", np.empty((0,), dtype=np.int32)), dtype=np.int64).reshape(-1)
    max_index = min(int(sparse_xy_all.shape[0]), int(sparse_xyz_all.shape[0]))
    valid_inliers = sparse_inliers[(sparse_inliers >= 0) & (sparse_inliers < max_index)]
    diagnostics["candidate_anchor_count"] = int(valid_inliers.shape[0])
    if valid_inliers.shape[0] == 0:
        return {"xy": dense_xy, "xyz": dense_xyz, "weights": dense_weights, "diagnostics": diagnostics}

    anchor_xy = sparse_xy_all[valid_inliers]
    anchor_xyz = sparse_xyz_all[valid_inliers]
    errors = _reprojection_errors(
        query_xy=anchor_xy,
        points_world=anchor_xyz,
        pose_w2c=np.asarray(reference_pose_w2c, dtype=np.float32).reshape(4, 4),
        intrinsic=np.asarray(intrinsic, dtype=np.float32).reshape(3, 3),
        image_size=image_size,
    )
    keep = np.isfinite(errors) & (errors <= float(policy.anchor_max_reprojection_px))
    kept_indices = np.flatnonzero(keep).astype(np.int64)
    if kept_indices.shape[0] > int(policy.anchor_max_count) > 0:
        order = np.argsort(errors[kept_indices], kind="mergesort")
        kept_indices = kept_indices[order[: int(policy.anchor_max_count)]]
    selected_xy = anchor_xy[kept_indices]
    selected_xyz = anchor_xyz[kept_indices]
    diagnostics["selected_anchor_count"] = int(selected_xy.shape[0])
    finite_errors = errors[np.isfinite(errors)]
    diagnostics["candidate_median_reprojection_error_px"] = float(np.median(finite_errors)) if finite_errors.size else None
    diagnostics["selected_median_reprojection_error_px"] = (
        float(np.median(errors[kept_indices])) if kept_indices.shape[0] else None
    )
    if selected_xy.shape[0] == 0:
        return {"xy": dense_xy, "xyz": dense_xyz, "weights": dense_weights, "diagnostics": diagnostics}

    anchor_weight = _effective_anchor_weight(
        dense_match_count=int(dense_xy.shape[0]),
        selected_anchor_count=int(selected_xy.shape[0]),
        policy=policy,
    )
    diagnostics["effective_sparse_anchor_weight"] = float(anchor_weight)
    xy = np.concatenate([dense_xy, selected_xy.astype(np.float32)], axis=0)
    xyz = np.concatenate([dense_xyz, selected_xyz.astype(np.float32)], axis=0)
    weights = np.concatenate(
        [dense_weights, np.full((selected_xy.shape[0],), float(anchor_weight), dtype=np.float32)],
        axis=0,
    )
    return {"xy": xy, "xyz": xyz, "weights": weights, "diagnostics": diagnostics}


def assess_anchor_conditioned_update(
    local_refine: Mapping[str, Any] | None,
    *,
    policy: AnchorConditionedPatchDensePolicy = AnchorConditionedPatchDensePolicy(),
) -> dict[str, Any]:
    """Accept a dense patch update only when it gives non-trivial no-GT gain.

    Dense patch refinement can create a tiny reprojection-objective gain by
    moving the pose away from an already-good dense estimate. This gate keeps
    patch dense as a corrective module: if it moves the pose, the local
    reference objective must improve enough to justify the update.
    """

    if local_refine is None:
        return {
            "schema": "loc_gs_acpd_refinement_acceptance_v1",
            "decision": "accept_anchor_conditioned_update",
            "reason": "no_reference_refinement_diagnostics",
            "diagnostic_only": True,
        }
    local_refine = dict(local_refine)
    success = bool(local_refine.get("success", True))
    before_median = _finite_float(local_refine.get("before_median_reprojection_error_px"))
    after_median = _finite_float(local_refine.get("after_median_reprojection_error_px"))
    before_p90 = _finite_float(local_refine.get("before_p90_reprojection_error_px"))
    after_p90 = _finite_float(local_refine.get("after_p90_reprojection_error_px"))
    translation_delta = _finite_float(local_refine.get("delta_translation_norm_m"), default=0.0) or 0.0
    rotation_delta = _finite_float(local_refine.get("delta_rotation_norm_deg"), default=0.0) or 0.0
    median_gain = None if before_median is None or after_median is None else before_median - after_median
    p90_gain = None if before_p90 is None or after_p90 is None else before_p90 - after_p90
    enough_median_gain = median_gain is not None and median_gain >= float(policy.min_refine_median_gain_px)
    enough_p90_gain = p90_gain is not None and p90_gain >= float(policy.min_refine_p90_gain_px)
    moved_without_gain = (
        translation_delta > float(policy.max_refine_translation_without_gain_m)
        or rotation_delta > float(policy.max_refine_rotation_without_gain_deg)
    )
    failed_checks = {
        "reference_refine_failed": not success,
        "insufficient_objective_gain_for_motion": bool(moved_without_gain and not (enough_median_gain or enough_p90_gain)),
    }
    decision = "accept_anchor_conditioned_update" if not any(failed_checks.values()) else "reject_anchor_conditioned_update"
    return {
        "schema": "loc_gs_acpd_refinement_acceptance_v1",
        "decision": decision,
        "failed_checks": failed_checks,
        "median_gain_px": None if median_gain is None else float(median_gain),
        "p90_gain_px": None if p90_gain is None else float(p90_gain),
        "translation_delta_m": float(translation_delta),
        "rotation_delta_deg": float(rotation_delta),
        "min_refine_median_gain_px": float(policy.min_refine_median_gain_px),
        "min_refine_p90_gain_px": float(policy.min_refine_p90_gain_px),
        "max_refine_translation_without_gain_m": float(policy.max_refine_translation_without_gain_m),
        "max_refine_rotation_without_gain_deg": float(policy.max_refine_rotation_without_gain_deg),
        "diagnostic_only": True,
    }


def summarize_anchor_conditioned_patch_dense(
    *,
    decision: str,
    patch_ids: Sequence[int],
    merged_match_count: int,
    candidate_match_count: int,
    final_inlier_count: int,
    anchor_diagnostics: Mapping[str, Any] | None,
    acpd_acceptance: Mapping[str, Any] | None = None,
    transition_control: Mapping[str, Any] | None = None,
    base_trust: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create compact ACPD audit metadata without GT-dependent signals."""

    anchor_diagnostics = dict(anchor_diagnostics or {})
    candidate_anchor_count = int(anchor_diagnostics.get("candidate_anchor_count", 0) or 0)
    selected_anchor_count = int(anchor_diagnostics.get("selected_anchor_count", 0) or 0)
    return {
        "schema": "loc_gs_anchor_conditioned_patch_dense_v1",
        "decision": str(decision),
        "anchor_conditioned": True,
        "query_side_occlusion_module": "patch_dense_matching",
        "map_side_artifact_module": "sparse_conditioned_render_control",
        "uses_gt": False,
        "is_oracle_branch_selector": False,
        "patch_ids": [int(value) for value in patch_ids],
        "patch_count": int(len(list(patch_ids))),
        "merged_match_count": int(merged_match_count),
        "candidate_match_count": int(candidate_match_count),
        "final_inlier_count": int(final_inlier_count),
        "candidate_anchor_count": candidate_anchor_count,
        "selected_anchor_count": selected_anchor_count,
        "anchor_selected_ratio": float(selected_anchor_count / max(1, candidate_anchor_count)),
        "anchor_weight": float(anchor_diagnostics.get("sparse_anchor_weight", 0.0) or 0.0),
        "effective_anchor_weight": float(
            anchor_diagnostics.get("effective_sparse_anchor_weight", anchor_diagnostics.get("sparse_anchor_weight", 0.0)) or 0.0
        ),
        "acpd_acceptance_decision": None if acpd_acceptance is None else acpd_acceptance.get("decision"),
        "transition_decision": None if transition_control is None else transition_control.get("decision"),
        "transition_selected_fraction": None if transition_control is None else transition_control.get("selected_fraction"),
        "base_trust_decision": None if base_trust is None else base_trust.get("decision"),
        "diagnostic_only": True,
    }
