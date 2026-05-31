#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

from loc_gs.diagnostics.match_visualization import (
    draw_match_canvas,
    pose_error_cm_deg,
    project_points,
    summarize_match_quality,
    write_json,
)
from loc_gs.dense_support.sparse_conditioned_dense_preflight import (
    DenseTransitionPolicy,
    SLCDPRepairSearchConfig,
    SLCDPThresholds,
    _camera_center as _slcdp_camera_center,
    _rotation_delta_deg as _slcdp_rotation_delta_deg,
    compute_sparse_ray_gaussian_gating_mask,
    compute_sparse_ray_depth_diagnostics,
    evaluate_dense_step_acceptance,
    generate_fast_landmark_guided_pose_candidates,
    generate_landmark_guided_pose_candidates,
    generate_pose_repair_candidates,
    select_sparse_conditioned_dense_transition,
    select_soft_sparse_conditioned_dense_transition,
    select_repaired_pose_candidate,
    sparse_landmark_conditioned_preflight,
    interpolate_w2c_poses,
)
from loc_gs.dense_support.clean_render_generator import select_clean_render_candidate
from loc_gs.dense_support.apd_dense import APDDensePolicy, run_anchor_patch_dense_refinement
from loc_gs.dense_support.patch_dense_candidates import PatchDenseCandidatePolicy, generate_patch_dense_candidates
from loc_gs.dense_support.sparse_anchor_residual import SparseAnchorResidualPolicy, compute_sparse_anchor_residual_group
from loc_gs.diagnostics.apd_dense_damage_risk import compute_dense_damage_risk
from loc_gs.reporting.artifact_audit import write_artifact_audit_bundle


REPO_ROOT = Path(__file__).resolve().parents[2]
STDLOC_ROOT = REPO_ROOT / "third_party/stdloc"
SPARSE_CONDITIONED_RENDER_CONTROL = "sparse_conditioned"
SPARSE_CONDITIONED_EFFECTIVE_RENDER_CONTROL = "fast_gating_guided_pose"


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _repo_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    return raw if raw.is_absolute() else REPO_ROOT / raw


def _camera_collection_get(cameras: Any, index: int) -> Any:
    if hasattr(cameras, "dataset"):
        return cameras.dataset[int(index)]
    return cameras[int(index)]


class _LazyCameraByName:
    def __init__(self, cameras: Any, camera_infos: Sequence[Any]):
        self._cameras = cameras
        self._name_to_index = {str(info.image_name): int(index) for index, info in enumerate(camera_infos)}
        self._cache: dict[int, Any] = {}

    def get(self, image_name: str, default: Any = None) -> Any:
        index = self._name_to_index.get(str(image_name))
        if index is None:
            return default
        if index not in self._cache:
            self._cache[index] = _camera_collection_get(self._cameras, index)
        return self._cache[index]


def _parse_float_steps(value: str, *, default: tuple[float, ...]) -> tuple[float, ...]:
    text = str(value or "").strip()
    if not text:
        return default
    steps = tuple(float(item) for item in text.split(",") if item.strip())
    return steps or default


def _should_apply_slcdp_gating(
    *,
    render_control_mode: str,
    behavior_render_control_mode: str,
    candidate: dict[str, Any],
) -> bool:
    if behavior_render_control_mode not in {"gaussian_gating", "gating_guided_pose", "fast_gating_guided_pose"}:
        return False
    if render_control_mode == SPARSE_CONDITIONED_RENDER_CONTROL:
        return bool(candidate.get("slcdp_apply_gating", False))
    return True


def _should_accept_sparse_conditioned_base_fast_path(preflight: Mapping[str, Any] | None) -> bool:
    """Skip sparse-conditioned candidate rendering when native dense render is already safe."""

    if not isinstance(preflight, Mapping):
        return False
    if preflight.get("decision") != "accept_dense":
        return False
    if not bool(preflight.get("sparse_confident", False)):
        return False
    visible_ratio = float(preflight.get("visible_ratio", 0.0) or 0.0)
    near_ratio = float(preflight.get("near_occluder_ratio", 1.0) or 1.0)
    coverage_cells = int(preflight.get("coverage_grid_cells", 0) or 0)
    return visible_ratio >= 0.35 and near_ratio <= 0.50 and coverage_cells >= 4


def _dense_pose_quality(capture: Mapping[str, Any] | None) -> dict[str, Any]:
    """Summarize dense PnP quality without GT pose or evaluation feedback."""

    if not isinstance(capture, Mapping):
        return {
            "match_count": 0,
            "solver_inlier_count": 0,
            "solver_inlier_ratio": 0.0,
            "median_reprojection_error_px": None,
            "p90_reprojection_error_px": None,
        }
    query_xy = np.asarray(capture.get("query_xy", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    p3d = np.asarray(capture.get("p3d", np.empty((0, 3))), dtype=np.float64).reshape(-1, 3)
    match_count = int(min(query_xy.shape[0], p3d.shape[0]))
    inliers = np.asarray(capture.get("inliers", np.empty(0)), dtype=np.int64).reshape(-1)
    inliers = inliers[(inliers >= 0) & (inliers < match_count)]
    quality: dict[str, Any] = {
        "match_count": match_count,
        "solver_inlier_count": int(inliers.shape[0]),
        "solver_inlier_ratio": float(inliers.shape[0] / max(match_count, 1)),
        "median_reprojection_error_px": None,
        "p90_reprojection_error_px": None,
    }
    if match_count == 0:
        return quality
    try:
        pose = np.asarray(capture["pose_w2c"], dtype=np.float64).reshape(4, 4)
        K = np.asarray(capture["K"], dtype=np.float64).reshape(3, 3)
    except (KeyError, TypeError, ValueError):
        return quality
    homog = np.concatenate([p3d[:match_count], np.ones((match_count, 1), dtype=np.float64)], axis=1)
    camera = (pose @ homog.T).T[:, :3]
    depth = camera[:, 2]
    finite = np.isfinite(camera).all(axis=1) & (depth > 1e-8)
    projected = np.full((match_count, 2), np.nan, dtype=np.float64)
    if finite.any():
        projected_h = (K @ camera[finite].T).T
        projected[finite] = projected_h[:, :2] / np.maximum(projected_h[:, 2:3], 1e-8)
    errors = np.linalg.norm(projected - query_xy[:match_count], axis=1)
    finite_errors = errors[np.isfinite(errors)]
    if finite_errors.size:
        quality["median_reprojection_error_px"] = float(np.median(finite_errors))
        quality["p90_reprojection_error_px"] = float(np.percentile(finite_errors, 90.0))
    return quality


def _is_high_confidence_dense_quality(quality: Mapping[str, Any] | None) -> bool:
    if not isinstance(quality, Mapping):
        return False
    try:
        match_count = int(quality.get("match_count", 0) or 0)
        inlier_count = int(quality.get("solver_inlier_count", 0) or 0)
        inlier_ratio = float(quality.get("solver_inlier_ratio", 0.0) or 0.0)
        median_error = float(quality.get("median_reprojection_error_px"))
        p90_error = float(quality.get("p90_reprojection_error_px"))
    except (TypeError, ValueError):
        return False
    return (
        match_count >= 1000
        and inlier_count >= 500
        and inlier_ratio >= 0.92
        and median_error <= 2.5
        and p90_error <= 5.0
    )


def _quality_float(quality: Mapping[str, Any], key: str, default: float) -> float:
    try:
        value = quality.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _anchor_residual_for_pose(
    *,
    sparse_capture: Mapping[str, Any],
    pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    try:
        return compute_sparse_anchor_residual_group(
            sparse_query_xy=sparse_capture["query_xy"],
            sparse_points_world=sparse_capture["p3d"],
            pose_w2c=np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4),
            intrinsic=np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
            sparse_inlier_indices=sparse_capture["inliers"],
            image_size=image_size,
            policy=SparseAnchorResidualPolicy(max_anchor_count=256),
        )
    except (KeyError, TypeError, ValueError):
        return {
            "valid_anchor_count": 0,
            "median_error_px": None,
            "p90_error_px": None,
            "group_loss": None,
        }


def _dense_damage_risk_observation(
    *,
    sparse_capture: Mapping[str, Any],
    dense_capture: Mapping[str, Any],
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    sparse_pose = np.asarray(sparse_capture["pose_w2c"], dtype=np.float64).reshape(4, 4)
    dense_pose = np.asarray(dense_capture["pose_w2c"], dtype=np.float64).reshape(4, 4)
    dense_quality = dense_capture.get("dense_pose_quality")
    if not isinstance(dense_quality, Mapping):
        dense_quality = _dense_pose_quality(dense_capture)
    sparse_anchor = _anchor_residual_for_pose(
        sparse_capture=sparse_capture,
        pose_w2c=sparse_pose,
        intrinsic=intrinsic,
        image_size=image_size,
    )
    dense_anchor = _anchor_residual_for_pose(
        sparse_capture=sparse_capture,
        pose_w2c=dense_pose,
        intrinsic=intrinsic,
        image_size=image_size,
    )
    translation_delta = float(np.linalg.norm(_slcdp_camera_center(dense_pose) - _slcdp_camera_center(sparse_pose)))
    rotation_delta = float(_slcdp_rotation_delta_deg(sparse_pose, dense_pose))
    return {
        "sparse_inlier_count": int(np.asarray(sparse_capture.get("inliers", [])).reshape(-1).shape[0]),
        "sparse_pose_anchor_median_px": sparse_anchor.get("median_error_px"),
        "sparse_pose_anchor_p90_px": sparse_anchor.get("p90_error_px"),
        "sparse_pose_anchor_valid_count": sparse_anchor.get("valid_anchor_count"),
        "base_dense_pose_anchor_median_px": dense_anchor.get("median_error_px"),
        "base_dense_pose_anchor_p90_px": dense_anchor.get("p90_error_px"),
        "base_dense_pose_anchor_valid_count": dense_anchor.get("valid_anchor_count"),
        "base_dense_vs_sparse_translation_delta_m": translation_delta,
        "base_dense_vs_sparse_rotation_delta_deg": rotation_delta,
        "base_dense_pose_p90_reprojection_error_px": dense_quality.get("p90_reprojection_error_px"),
        "base_dense_pose_median_reprojection_error_px": dense_quality.get("median_reprojection_error_px"),
        "base_dense_pose_solver_inlier_ratio": dense_quality.get("solver_inlier_ratio"),
        "base_dense_pose_solver_inlier_count": dense_quality.get("solver_inlier_count"),
    }


def _evaluate_apd_no_regression_switch(
    *,
    native_capture: Mapping[str, Any],
    apd_pose_w2c: np.ndarray,
    apd_refine_success: bool,
    min_inlier_ratio_delta: float,
    max_median_reproj_increase_px: float,
    max_p90_reproj_increase_px: float,
    min_inlier_count_ratio: float,
) -> dict[str, Any]:
    """Decide whether APD refined pose can replace native dense pose.

    This gate is no-GT and protects normal cases by requiring APD pose quality
    to stay close to native dense quality on the same correspondence set.
    """

    native_quality = dict(
        native_capture.get("dense_pose_quality") or _dense_pose_quality(native_capture)
    )
    apd_capture = dict(native_capture)
    apd_capture["pose_w2c"] = np.asarray(apd_pose_w2c, dtype=np.float32).reshape(4, 4)
    apd_quality = _dense_pose_quality(apd_capture)

    native_inlier_ratio = _quality_float(native_quality, "solver_inlier_ratio", 0.0)
    apd_inlier_ratio = _quality_float(apd_quality, "solver_inlier_ratio", 0.0)
    native_inlier_count = int(_quality_float(native_quality, "solver_inlier_count", 0.0))
    apd_inlier_count = int(_quality_float(apd_quality, "solver_inlier_count", 0.0))
    native_median = _quality_float(native_quality, "median_reprojection_error_px", float("inf"))
    apd_median = _quality_float(apd_quality, "median_reprojection_error_px", float("inf"))
    native_p90 = _quality_float(native_quality, "p90_reprojection_error_px", float("inf"))
    apd_p90 = _quality_float(apd_quality, "p90_reprojection_error_px", float("inf"))
    native_high_confidence = _is_high_confidence_dense_quality(native_quality)

    checks = {
        "refinement_success": bool(apd_refine_success),
        "inlier_ratio_not_worse": bool(apd_inlier_ratio >= native_inlier_ratio - float(min_inlier_ratio_delta)),
        "inlier_count_not_worse": bool(apd_inlier_count >= int(np.floor(native_inlier_count * float(min_inlier_count_ratio)))),
        "median_reprojection_not_worse": bool(apd_median <= native_median + float(max_median_reproj_increase_px)),
        "p90_reprojection_not_worse": bool(apd_p90 <= native_p90 + float(max_p90_reproj_increase_px)),
    }
    if native_high_confidence:
        checks.update(
            {
                "high_confidence_native_protection": bool(
                    apd_inlier_ratio >= native_inlier_ratio
                    and apd_inlier_count >= native_inlier_count
                    and apd_median <= native_median
                    and apd_p90 <= native_p90
                ),
            }
        )
    accept = bool(all(checks.values()))
    reason = "accept_apd_pose" if accept else "reject_apd_pose_no_regression_gate"
    return {
        "schema": "loc_gs_apd_no_regression_gate_v1",
        "decision": reason,
        "accept_apd_pose": bool(accept),
        "checks": checks,
        "native_dense_quality": native_quality,
        "apd_dense_quality": apd_quality,
        "native_high_confidence": bool(native_high_confidence),
        "thresholds": {
            "min_inlier_ratio_delta": float(min_inlier_ratio_delta),
            "max_median_reproj_increase_px": float(max_median_reproj_increase_px),
            "max_p90_reproj_increase_px": float(max_p90_reproj_increase_px),
            "min_inlier_count_ratio": float(min_inlier_count_ratio),
        },
        "diagnostic_only": True,
    }


def _should_reject_sparse_conditioned_repair_for_dense_quality_regression(
    base_quality: Mapping[str, Any] | None,
    repair_quality: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Reject a repair when a coherent native dense solution would be damaged.

    This is a non-GT diagnostic trust check. It protects cases where native dense
    has many geometrically coherent inliers but the sparse-conditioned render
    worsens the dense PnP tail and consensus ratio.
    """

    if not isinstance(base_quality, Mapping) or not isinstance(repair_quality, Mapping):
        return {"reject_repair": False, "reason": "missing_dense_quality"}
    base_match_count = int(_quality_float(base_quality, "match_count", 0.0))
    base_inlier_count = int(_quality_float(base_quality, "solver_inlier_count", 0.0))
    base_ratio = _quality_float(base_quality, "solver_inlier_ratio", 0.0)
    base_median = _quality_float(base_quality, "median_reprojection_error_px", float("inf"))
    base_p90 = _quality_float(base_quality, "p90_reprojection_error_px", float("inf"))
    repair_match_count = int(_quality_float(repair_quality, "match_count", 0.0))
    repair_ratio = _quality_float(repair_quality, "solver_inlier_ratio", 0.0)
    repair_median = _quality_float(repair_quality, "median_reprojection_error_px", float("inf"))
    repair_p90 = _quality_float(repair_quality, "p90_reprojection_error_px", float("inf"))
    protective_base = (
        base_match_count >= 1000
        and base_inlier_count >= 500
        and base_ratio >= 0.70
        and base_median <= 2.5
    )
    if not protective_base:
        return {
            "reject_repair": False,
            "reason": "base_dense_quality_not_protective",
            "base_quality": dict(base_quality),
            "repair_quality": dict(repair_quality),
        }
    tail_degraded = repair_p90 >= base_p90 + 50.0
    consensus_degraded = repair_ratio <= base_ratio - 0.02
    core_median_degraded = repair_median >= max(base_median + 5.0, base_median * 5.0)
    core_consensus_collapsed = repair_ratio <= base_ratio - 0.15
    match_support_collapsed = repair_match_count <= int(0.80 * max(base_match_count, 1))
    core_degraded = bool(core_median_degraded or (core_consensus_collapsed and match_support_collapsed))
    reject = bool((tail_degraded and consensus_degraded) or core_degraded)
    if core_degraded:
        reason = "repair_degrades_dense_quality_core"
    elif tail_degraded and consensus_degraded:
        reason = "repair_degrades_dense_quality_tail"
    else:
        reason = "repair_dense_quality_not_worse"
    return {
        "reject_repair": reject,
        "reason": reason,
        "tail_degraded": bool(tail_degraded),
        "consensus_degraded": bool(consensus_degraded),
        "core_median_degraded": bool(core_median_degraded),
        "core_consensus_collapsed": bool(core_consensus_collapsed),
        "match_support_collapsed": bool(match_support_collapsed),
        "base_quality": dict(base_quality),
        "repair_quality": dict(repair_quality),
    }


def _should_reject_sparse_conditioned_dense_for_low_quality(
    *,
    selected_label: str,
    repair_selection: Mapping[str, Any] | None,
    sparse_capture: Mapping[str, Any] | None,
    dense_quality: Mapping[str, Any] | None,
    min_sparse_inliers: int = DenseTransitionPolicy.strong_sparse_inlier_count,
) -> dict[str, Any]:
    """Reject repaired dense updates with poor no-GT dense consensus.

    Sparse-conditioned render control can make anchor visibility look much
    better while dense matching still produces a noisy, weak-consensus PnP.
    For strong sparse anchors, such a dense update is not allowed to become the
    final pose unless the dense correspondence set is internally coherent.
    """

    if str(selected_label) == "base":
        return {"reject_dense": False, "reason": "base_candidate", "diagnostic_only": True}
    if not isinstance(repair_selection, Mapping) or str(repair_selection.get("decision", "")) != "accept_repaired_dense_pose":
        return {"reject_dense": False, "reason": "no_accepted_repair", "diagnostic_only": True}
    if not isinstance(sparse_capture, Mapping):
        return {"reject_dense": False, "reason": "missing_sparse_capture", "diagnostic_only": True}
    sparse_inliers = np.asarray(sparse_capture.get("inliers", np.empty(0)), dtype=np.int64).reshape(-1)
    sparse_inlier_count = int(sparse_inliers.shape[0])
    if sparse_inlier_count < int(min_sparse_inliers):
        return {
            "reject_dense": False,
            "reason": "sparse_support_not_strong",
            "sparse_inlier_count": sparse_inlier_count,
            "min_sparse_inliers": int(min_sparse_inliers),
            "diagnostic_only": True,
        }
    if not isinstance(dense_quality, Mapping):
        return {"reject_dense": False, "reason": "missing_dense_quality", "diagnostic_only": True}

    match_count = int(_quality_float(dense_quality, "match_count", 0.0))
    inlier_count = int(_quality_float(dense_quality, "solver_inlier_count", 0.0))
    inlier_ratio = _quality_float(dense_quality, "solver_inlier_ratio", 0.0)
    median_error = _quality_float(dense_quality, "median_reprojection_error_px", float("inf"))
    p90_error = _quality_float(dense_quality, "p90_reprojection_error_px", float("inf"))
    weak_consensus = inlier_ratio < 0.60
    poor_core = median_error > 12.0
    poor_tail = p90_error > 80.0
    low_support = match_count < 256 or inlier_count < int(min_sparse_inliers)
    reject = bool(low_support or (weak_consensus and (poor_core or poor_tail)))
    if low_support:
        reason = "strong_sparse_low_dense_support"
    elif reject:
        reason = "strong_sparse_low_quality_dense_repair"
    else:
        reason = "dense_quality_acceptable"
    return {
        "schema": "loc_gs_slcdp_dense_quality_guard_v1",
        "decision": "reject_dense_keep_sparse" if reject else "accept_dense_update",
        "reject_dense": reject,
        "reason": reason,
        "selected_label": str(selected_label),
        "sparse_inlier_count": sparse_inlier_count,
        "min_sparse_inliers": int(min_sparse_inliers),
        "dense_quality": dict(dense_quality),
        "failed_checks": {
            "low_support": bool(low_support),
            "weak_consensus": bool(weak_consensus),
            "poor_core": bool(poor_core),
            "poor_tail": bool(poor_tail),
        },
        "diagnostic_only": True,
    }


def _should_reject_gating_only_repair_for_weak_sparse_support(
    *,
    selected_label: str,
    sparse_capture: Mapping[str, Any] | None,
    min_sparse_inliers: int = DenseTransitionPolicy.strong_sparse_inlier_count,
) -> dict[str, Any]:
    """Reject zero-motion ray gating when sparse support is too weak.

    Ray-depth gating uses sparse inlier rays to decide which Gaussians are
    artifacts. With too few inliers, that prior is not reliable enough to
    override native dense. Guided-pose candidates are evaluated separately.
    """

    if str(selected_label) != "gated_base":
        return {
            "reject_repair": False,
            "reason": "not_gating_only",
            "selected_label": str(selected_label),
        }
    if not isinstance(sparse_capture, Mapping):
        return {
            "reject_repair": True,
            "reason": "missing_sparse_capture_for_gating_only",
            "selected_label": str(selected_label),
            "sparse_inlier_count": 0,
            "min_sparse_inliers": int(min_sparse_inliers),
        }
    inliers = np.asarray(sparse_capture.get("inliers", []), dtype=np.int64).reshape(-1)
    points = sparse_capture.get("p3d")
    if points is not None:
        point_count = int(np.asarray(points).reshape(-1, 3).shape[0])
        inliers = inliers[(inliers >= 0) & (inliers < point_count)]
    else:
        inliers = inliers[inliers >= 0]
    inlier_count = int(inliers.size)
    reject = inlier_count < int(min_sparse_inliers)
    return {
        "reject_repair": bool(reject),
        "reason": "weak_sparse_support_gating_only" if reject else "sparse_support_sufficient_for_gating_only",
        "selected_label": str(selected_label),
        "sparse_inlier_count": inlier_count,
        "min_sparse_inliers": int(min_sparse_inliers),
    }


def _should_reuse_sparse_conditioned_base_dense(
    preflight: Mapping[str, Any] | None,
    base_dense_capture: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Decide if sparse-conditioned control should preserve native dense."""

    if _should_accept_sparse_conditioned_base_fast_path(preflight):
        return {"reuse_base": True, "reason": "base_preflight_safe"}
    quality = None
    if isinstance(base_dense_capture, Mapping):
        quality = base_dense_capture.get("dense_pose_quality")
        if quality is None:
            quality = _dense_pose_quality(base_dense_capture)
    if _is_high_confidence_dense_quality(quality):
        return {
            "reuse_base": True,
            "reason": "base_dense_high_confidence",
            "dense_pose_quality": dict(quality),
        }
    return {
        "reuse_base": False,
        "reason": "base_preflight_requires_repair",
        "dense_pose_quality": dict(quality) if isinstance(quality, Mapping) else None,
    }


def _should_apply_sparse_conditioned_transition_control(
    *,
    selected_label: str,
    repair_selection: Mapping[str, Any] | None,
) -> bool:
    """Run sparse-conditioned step acceptance only after a non-base repair is selected."""

    if str(selected_label) == "base":
        return False
    if not isinstance(repair_selection, Mapping):
        return False
    return str(repair_selection.get("decision", "")) == "accept_repaired_dense_pose"


def _reuse_sparse_conditioned_base_dense_capture(
    base_dense_capture: Mapping[str, Any],
    base_rendered: Mapping[str, Any],
    *,
    reuse_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reuse an already computed base dense result for a sparse-conditioned base decision."""

    preflight = dict(base_rendered.get("preflight") or {})
    render_control = dict(base_rendered.get("render_control") or {})
    reuse_decision = dict(reuse_decision or {})
    reused = dict(base_dense_capture)
    reused["dense_pose_quality"] = dict(
        reused.get("dense_pose_quality") or _dense_pose_quality(reused)
    )
    reused["slcdp_render_control"] = {
        **render_control,
        "base_dense_reused": True,
        "base_reuse_reason": reuse_decision.get("reason", "base_preflight_safe"),
    }
    reused["slcdp_preflight"] = preflight
    reused["slcdp_repair_search"] = {
        "enabled": True,
        "selection": {
            "schema": "loc_gs_slcdp_repair_selection_v1",
            "decision": "accept_original_dense_pose",
            "selected_label": "base",
            "best_label": "base",
            "score_gain": 0.0,
            "candidate_count": 1,
            "base_reuse_reason": reuse_decision.get("reason", "base_preflight_safe"),
            "dense_pose_quality": reuse_decision.get("dense_pose_quality"),
            "diagnostic_only": True,
        },
        "candidates": [
            {
                "label": "base",
                "selection_mode": base_rendered.get("selection_mode"),
                "render_rank": base_rendered.get("render_rank"),
                "translation_cam_m": base_rendered.get("translation_cam_m"),
                "rotation_deg": base_rendered.get("rotation_deg"),
                "fast_score": base_rendered.get("fast_score"),
                "fast_pool_candidate_count": base_rendered.get("fast_pool_candidate_count"),
                "preflight": preflight,
                "render_control": render_control,
            }
        ],
        "render_candidate_count": 1,
        "dense_skipped": False,
        "dense_reused_from_base": True,
    }
    return reused


def _skip_sparse_conditioned_dense_from_base_render(
    *,
    sparse_pose: np.ndarray,
    K: np.ndarray,
    base_rendered: Mapping[str, Any],
) -> dict[str, Any]:
    preflight = dict(base_rendered.get("preflight") or {})
    render_control = dict(base_rendered.get("render_control") or {})
    empty = np.empty((0, 2), dtype=np.float32)
    return {
        "query_xy": empty,
        "rendered_xy": empty,
        "p3d": np.empty((0, 3), dtype=np.float32),
        "pose_w2c": np.asarray(sparse_pose, dtype=np.float32).reshape(4, 4),
        "inliers": np.empty(0, dtype=np.int32),
        "K": K,
        "render": (base_rendered.get("render_pkg") or {}).get("render"),
        "ray_depth_diagnostics": [],
        "slcdp_render_control": render_control,
        "slcdp_preflight": preflight,
        "slcdp_repair_search": {
            "enabled": True,
            "selection": {
                "schema": "loc_gs_slcdp_repair_selection_v1",
                "decision": "skip_dense_keep_sparse",
                "selected_label": "base",
                "best_label": "base",
                "score_gain": 0.0,
                "candidate_count": 1,
                "reason": "base_preflight_sparse_confident_render_not_explanatory",
                "diagnostic_only": True,
            },
            "candidates": [
                {
                    "label": "base",
                    "selection_mode": base_rendered.get("selection_mode"),
                    "render_rank": base_rendered.get("render_rank"),
                    "translation_cam_m": base_rendered.get("translation_cam_m"),
                    "rotation_deg": base_rendered.get("rotation_deg"),
                    "fast_score": base_rendered.get("fast_score"),
                    "fast_pool_candidate_count": base_rendered.get("fast_pool_candidate_count"),
                    "preflight": preflight,
                    "render_control": render_control,
                }
            ],
            "render_candidate_count": 1,
            "dense_skipped": True,
        },
    }


def _slcdp_candidate_audit_rows(rendered_candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": item.get("label"),
            "selection_mode": item.get("selection_mode"),
            "render_rank": item.get("render_rank"),
            "translation_cam_m": item.get("translation_cam_m"),
            "rotation_deg": item.get("rotation_deg"),
            "fast_score": item.get("fast_score"),
            "fast_pool_candidate_count": item.get("fast_pool_candidate_count"),
            "preflight": item.get("preflight"),
            "render_control": item.get("render_control"),
        }
        for item in rendered_candidates
    ]


def _reuse_base_dense_after_rejected_sparse_conditioned_repair(
    *,
    base_dense_capture: Mapping[str, Any],
    base_rendered: Mapping[str, Any],
    repair_selection: Mapping[str, Any],
    rendered_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reused = _reuse_sparse_conditioned_base_dense_capture(
        base_dense_capture,
        base_rendered,
        reuse_decision={"reason": "repair_rejected_no_accept"},
    )
    selection = dict(repair_selection)
    selection["original_decision"] = selection.get("decision")
    selection["decision"] = "reuse_base_dense_no_repair_accept"
    selection["selected_label"] = "base"
    selection.setdefault("best_label", repair_selection.get("best_label", "base"))
    selection["diagnostic_only"] = True
    reused["slcdp_repair_search"] = {
        "enabled": True,
        "selection": selection,
        "candidates": _slcdp_candidate_audit_rows(rendered_candidates),
        "render_candidate_count": int(len(rendered_candidates)),
        "dense_skipped": False,
        "dense_reused_from_base": True,
    }
    return reused


def _reuse_base_dense_after_rejected_sparse_conditioned_transition(
    *,
    base_dense_capture: Mapping[str, Any],
    base_rendered: Mapping[str, Any],
    repair_selection: Mapping[str, Any],
    rendered_candidates: Sequence[Mapping[str, Any]],
    transition_control: Mapping[str, Any],
) -> dict[str, Any]:
    transition_decision = str(transition_control.get("decision", ""))
    reuse_reason = (
        "repair_dense_quality_regression"
        if transition_decision == "reject_repair_dense_quality_keep_base"
        else "repair_transition_rejected"
    )
    reused = _reuse_sparse_conditioned_base_dense_capture(
        base_dense_capture,
        base_rendered,
        reuse_decision={"reason": reuse_reason},
    )
    selection = dict(repair_selection)
    selection["original_decision"] = selection.get("decision")
    selection["decision"] = (
        "reuse_base_dense_quality_regression"
        if transition_decision == "reject_repair_dense_quality_keep_base"
        else "reuse_base_dense_transition_reject"
    )
    selection["rejected_selected_label"] = repair_selection.get("selected_label")
    selection["selected_label"] = "base"
    selection.setdefault("best_label", repair_selection.get("best_label", "base"))
    selection["diagnostic_only"] = True
    reused["slcdp_repair_search"] = {
        "enabled": True,
        "selection": selection,
        "candidates": _slcdp_candidate_audit_rows(rendered_candidates),
        "render_candidate_count": int(len(rendered_candidates)),
        "dense_skipped": False,
        "dense_reused_from_base": True,
    }
    reused["slcdp_transition_control"] = _compact_transition_result(dict(transition_control))
    return reused


def _resolve_slcdp_effective_options(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve CLI options, including the fixed sparse-conditioned dense preset."""

    translation_steps = _parse_float_steps(
        args.slcdp_repair_translation_steps_m,
        default=SLCDPRepairSearchConfig.translation_steps_m,
    )
    rotation_steps = _parse_float_steps(
        args.slcdp_repair_rotation_steps_deg,
        default=SLCDPRepairSearchConfig.rotation_steps_deg,
    )
    guided_steps = _parse_float_steps(args.slcdp_guided_pose_steps_m, default=(0.5, 1.0, 2.0, 5.0))
    transition_fractions = _parse_float_steps(
        args.slcdp_transition_line_search_fractions,
        default=DenseTransitionPolicy.line_search_fractions,
    )
    options: dict[str, Any] = {
        "slcdp_repair_search": bool(args.slcdp_repair_search),
        "slcdp_repair_skip_if_no_accept": bool(args.slcdp_repair_skip_if_no_accept),
        "slcdp_repair_max_candidates": int(args.slcdp_repair_max_candidates),
        "slcdp_repair_translation_steps_m": translation_steps,
        "slcdp_repair_rotation_steps_deg": rotation_steps,
        "slcdp_repair_require_accept": not bool(args.slcdp_repair_allow_non_accept),
        "slcdp_repair_min_score_gain": float(args.slcdp_repair_min_score_gain),
        "slcdp_repair_translation_penalty_per_m": float(args.slcdp_repair_translation_penalty_per_m),
        "slcdp_allow_low_confidence_gated_base": bool(args.slcdp_allow_low_confidence_gated_base),
        "slcdp_low_confidence_gated_base_min_score_gain": float(args.slcdp_low_confidence_gated_base_min_score_gain),
        "slcdp_render_control": str(args.slcdp_render_control),
        "slcdp_guided_pose_steps_m": guided_steps,
        "slcdp_gating_radius_px": float(args.slcdp_gating_radius_px),
        "slcdp_gating_depth_margin_m": float(args.slcdp_gating_depth_margin_m),
        "slcdp_gating_footprint_radius_scale": float(args.slcdp_gating_footprint_radius_scale),
        "slcdp_gating_max_footprint_radius_px": float(args.slcdp_gating_max_footprint_radius_px),
        "slcdp_gating_min_footprint_radius_px": float(args.slcdp_gating_min_footprint_radius_px),
        "slcdp_gating_min_depth_m": float(args.slcdp_gating_min_depth_m),
        "slcdp_gating_max_depth_m": float(args.slcdp_gating_max_depth_m),
        "slcdp_gating_min_opacity": float(args.slcdp_gating_min_opacity),
        "slcdp_gating_chunk_size": int(args.slcdp_gating_chunk_size),
        "slcdp_fast_guided_max_render_candidates": int(args.slcdp_fast_guided_max_render_candidates),
        "slcdp_fast_guided_conflict_radius_px": float(args.slcdp_fast_guided_conflict_radius_px),
        "slcdp_fast_guided_depth_margin_m": float(args.slcdp_fast_guided_depth_margin_m),
        "slcdp_transition_control": bool(args.slcdp_transition_control),
        "slcdp_soft_transition_control": bool(args.slcdp_soft_transition_control),
        "slcdp_transition_max_reprojection_error_px": float(args.slcdp_transition_max_reprojection_error_px),
        "slcdp_transition_min_retained_ratio": float(args.slcdp_transition_min_retained_ratio),
        "slcdp_transition_max_translation_delta_m": float(args.slcdp_transition_max_translation_delta_m),
        "slcdp_transition_max_rotation_delta_deg": float(args.slcdp_transition_max_rotation_delta_deg),
        "slcdp_transition_line_search_fractions": transition_fractions,
        "apd_dense": bool(getattr(args, "apd_dense", False)),
        "apd_include_patch_candidates": bool(getattr(args, "apd_include_patch_candidates", False)),
        "apd_dense_group_weight": float(getattr(args, "apd_dense_group_weight", 1.0)),
        "apd_anchor_group_weight": float(getattr(args, "apd_anchor_group_weight", 1.0)),
        "apd_max_translation_delta_m": float(getattr(args, "apd_max_translation_delta_m", 0.0)),
        "apd_max_rotation_delta_deg": float(getattr(args, "apd_max_rotation_delta_deg", 0.0)),
        "apd_anchor_monotonic": bool(getattr(args, "apd_anchor_monotonic", False)),
        "apd_anchor_monotonic_epsilon_px": float(getattr(args, "apd_anchor_monotonic_epsilon_px", 1.0)),
        "apd_use_refined_pose": bool(getattr(args, "apd_use_refined_pose", False)),
        "apd_risk_weighted": bool(getattr(args, "apd_risk_weighted", False)),
        "apd_risk_max_beta": float(getattr(args, "apd_risk_max_beta", 1.0)),
        "apd_no_regression_gate": bool(getattr(args, "apd_no_regression_gate", True)),
        "apd_gate_min_inlier_ratio_delta": float(getattr(args, "apd_gate_min_inlier_ratio_delta", 0.02)),
        "apd_gate_max_median_reproj_increase_px": float(getattr(args, "apd_gate_max_median_reproj_increase_px", 1.0)),
        "apd_gate_max_p90_reproj_increase_px": float(getattr(args, "apd_gate_max_p90_reproj_increase_px", 3.0)),
        "apd_gate_min_inlier_count_ratio": float(getattr(args, "apd_gate_min_inlier_count_ratio", 0.90)),
    }
    if str(args.slcdp_render_control) == SPARSE_CONDITIONED_RENDER_CONTROL:
        options.update(
            {
                "slcdp_repair_search": False,
                "slcdp_repair_skip_if_no_accept": True,
                "slcdp_repair_max_candidates": 37,
                "slcdp_repair_translation_steps_m": SLCDPRepairSearchConfig.translation_steps_m,
                "slcdp_repair_rotation_steps_deg": SLCDPRepairSearchConfig.rotation_steps_deg,
                "slcdp_repair_require_accept": True,
                "slcdp_repair_min_score_gain": 2.0,
                "slcdp_repair_translation_penalty_per_m": 0.75,
                "slcdp_allow_low_confidence_gated_base": False,
                "slcdp_low_confidence_gated_base_min_score_gain": 0.25,
                "slcdp_guided_pose_steps_m": (0.5, 1.0, 2.0, 5.0),
                "slcdp_gating_radius_px": 6.0,
                "slcdp_gating_depth_margin_m": 1.0,
                "slcdp_gating_footprint_radius_scale": 1.5,
                "slcdp_gating_max_footprint_radius_px": 96.0,
                "slcdp_gating_min_footprint_radius_px": 12.0,
                "slcdp_gating_min_depth_m": 5.0,
                "slcdp_gating_max_depth_m": 60.0,
                "slcdp_gating_min_opacity": 0.01,
                "slcdp_fast_guided_max_render_candidates": 4,
                "slcdp_fast_guided_conflict_radius_px": 8.0,
                "slcdp_fast_guided_depth_margin_m": 0.0,
                "slcdp_transition_control": True,
                "slcdp_transition_max_reprojection_error_px": 8.0,
                "slcdp_transition_min_retained_ratio": 0.875,
                "slcdp_transition_max_translation_delta_m": 0.35,
                "slcdp_transition_max_rotation_delta_deg": 5.0,
                "slcdp_transition_line_search_fractions": (1.0, 0.75, 0.5, 0.25, 0.0),
            }
        )
    if bool(getattr(args, "apd_dense", False)):
        options["slcdp_transition_control"] = False
        options["slcdp_soft_transition_control"] = False
    return options


def _prepare_stdloc() -> Any:
    stdloc_path = str(STDLOC_ROOT)
    if stdloc_path not in sys.path:
        sys.path.insert(0, stdloc_path)
    import stdloc as stdloc_module

    return stdloc_module


def _tensor_to_image(tensor: torch.Tensor, *, size: tuple[int, int] | None = None) -> Image.Image:
    array = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if array.dim() == 3 and array.shape[0] in {1, 3, 4}:
        array = array[:3].permute(1, 2, 0)
    image = Image.fromarray((array.numpy() * 255.0).astype(np.uint8)).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size)
    return image


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _load_results(run_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        payload = payload["results"]
    if not isinstance(payload, list):
        raise TypeError(f"results.json must contain a list: {run_dir}")
    return [dict(item) for item in payload]


def _build_context(run_dir: Path, *, split_override: str | None = None) -> dict[str, Any]:
    stdloc = _prepare_stdloc()
    manifest = _load_manifest(run_dir)
    scene = str(manifest.get("scene") or run_dir.name)
    data_roots = manifest.get("data_roots") or []
    scene_root = _repo_path(data_roots[0]) if data_roots else Path("/mnt/pool/sqy/Cambridge_stdloc") / scene
    map_path = _repo_path(manifest.get("map_path", ""))
    cfg_path = _repo_path(manifest.get("hyperparameters", {}).get("cfg", run_dir / "stdloc_spgs_cambridge_dense1.yaml"))
    if not cfg_path.exists():
        cfg_path = run_dir / "stdloc_spgs_cambridge_dense1.yaml"

    parser = argparse.ArgumentParser(add_help=False)
    model = stdloc.ModelParams(parser, sentinel=False)
    stdloc.PipelineParams(parser)
    parsed = parser.parse_args(
        [
            "-s",
            str(scene_root),
            "-m",
            str(map_path),
            "-r",
            "1",
            "-f",
            "sp",
            "-g",
            "3dgs",
            "--images",
            "processed" if (scene_root / "processed").exists() else ".",
            "--data_device",
            "cpu",
        ]
    )
    dataset = model.extract(parsed)
    gaussians = stdloc.GaussianModel(dataset.sh_degree)
    scene_obj = stdloc.Scene(
        dataset,
        gaussians,
        load_iteration=-1,
        shuffle=False,
        preload_cameras=False,
        dataloader_num_workers=0,
        pin_memory=False,
    )
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    config["dense"]["norm_before_render"] = dataset.norm_before_render
    config["feature_type"] = dataset.feature_type
    config["longest_edge"] = dataset.longest_edge
    config["model_path"] = dataset.model_path
    localizer = stdloc.STDLoc(gaussians, config)
    split_name = str(split_override or manifest.get("split") or "test")
    cameras = scene_obj.getTrainCameras() if split_name == "train" else scene_obj.getTestCameras()
    camera_infos = scene_obj.scene_info.train_cameras if split_name == "train" else scene_obj.scene_info.test_cameras
    camera_by_name = _LazyCameraByName(cameras, camera_infos)
    return {
        "stdloc": stdloc,
        "manifest": manifest,
        "scene": scene,
        "scene_root": str(scene_root),
        "map_path": str(map_path),
        "config": config,
        "localizer": localizer,
        "split": split_name,
        "cameras": cameras,
        "camera_by_name": camera_by_name,
    }


def _inlier_mask(count: int, inliers: np.ndarray) -> np.ndarray:
    mask = np.zeros((count,), dtype=bool)
    ids = np.asarray(inliers).reshape(-1)
    ids = ids[(ids >= 0) & (ids < count)].astype(np.int64)
    mask[ids] = True
    return mask


class _MaskedGaussianProxy:
    """Temporary render-only view of a GaussianModel with a boolean keep mask."""

    def __init__(self, base: Any, keep_mask: torch.Tensor):
        self.base = base
        self.keep_mask = keep_mask.bool()
        self.active_sh_degree = getattr(base, "active_sh_degree", 0)

    @property
    def get_xyz(self):
        return self.base.get_xyz[self.keep_mask]

    @property
    def get_opacity(self):
        return self.base.get_opacity[self.keep_mask]

    @property
    def get_scaling(self):
        return self.base.get_scaling[self.keep_mask]

    @property
    def get_rotation(self):
        return self.base.get_rotation[self.keep_mask]

    @property
    def get_features(self):
        return self.base.get_features[self.keep_mask]

    @property
    def get_loc_feature(self):
        return self.base.get_loc_feature[self.keep_mask]

    @property
    def get_locability(self):
        return self.base.get_locability[self.keep_mask]


def _compact_gating_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    compact: dict[str, Any] = {}
    for key, value in result.items():
        if key in {"keep_mask", "conflict_mask"}:
            continue
        if key == "conflict_indices":
            indices = np.asarray(value).reshape(-1)
            compact["conflict_indices_sample"] = [int(v) for v in indices[:20]]
            compact["conflict_indices_sample_count"] = int(min(indices.shape[0], 20))
            continue
        compact[key] = value
    return compact


def _compact_transition_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "schema": result.get("schema"),
        "decision": result.get("decision"),
        "selected_fraction": result.get("selected_fraction"),
        "sparse_support_strength": result.get("sparse_support_strength"),
        "selected_acceptance": result.get("selected_acceptance"),
        "policy": result.get("policy"),
        "candidate_count": len(result.get("candidates", []) or []),
        "candidates": [
            {
                "fraction": item.get("fraction"),
                "acceptance": item.get("acceptance"),
            }
            for item in (result.get("candidates", []) or [])
        ],
        "diagnostic_only": bool(result.get("diagnostic_only", True)),
    }


def _compact_apd_result(result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, Mapping):
        return None
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), Mapping) else {}
    pool = diagnostics.get("candidate_pool") if isinstance(diagnostics.get("candidate_pool"), Mapping) else {}
    refine = diagnostics.get("refinement") if isinstance(diagnostics.get("refinement"), Mapping) else {}
    anchor_monotonic = diagnostics.get("anchor_monotonic") if isinstance(diagnostics.get("anchor_monotonic"), Mapping) else {}
    score_rows = []
    for row in pool.get("score_summaries", []) or []:
        if not isinstance(row, Mapping):
            continue
        score_rows.append(
            {
                "candidate_count": row.get("candidate_count"),
                "weight_mean": row.get("weight_mean"),
                "anchor_consistency_median": row.get("anchor_consistency_median"),
                "geometry_score": row.get("geometry_score"),
                "match_quality_median": row.get("match_quality_median"),
            }
        )
    return {
        "schema": result.get("schema"),
        "uses_gt": bool(result.get("uses_gt", False)),
        "dense_refine_success": bool(result.get("dense_refine_success", False)),
        "num_global_candidates": result.get("num_global_candidates"),
        "num_clean_render_candidates": result.get("num_clean_render_candidates"),
        "num_patch_candidates": result.get("num_patch_candidates"),
        "num_anchor_residuals": result.get("num_anchor_residuals"),
        "candidate_pool": {
            "candidate_count": pool.get("candidate_count"),
            "source_counts": pool.get("source_counts"),
            "score_summaries": score_rows,
        },
        "refinement": {
            "success": refine.get("success"),
            "dense_candidate_count": refine.get("dense_candidate_count"),
            "anchor_residual_count": refine.get("anchor_residual_count"),
            "weighted_correspondence_count": refine.get("weighted_correspondence_count"),
            "dense_weight_sum": refine.get("dense_weight_sum"),
            "anchor_weight_sum": refine.get("anchor_weight_sum"),
            "group_normalized": refine.get("group_normalized"),
            "local_refinement": {
                "success": (refine.get("refinement") or {}).get("success") if isinstance(refine.get("refinement"), Mapping) else None,
                "delta_translation_norm_m": (refine.get("refinement") or {}).get("delta_translation_norm_m")
                if isinstance(refine.get("refinement"), Mapping)
                else None,
                "delta_rotation_norm_deg": (refine.get("refinement") or {}).get("delta_rotation_norm_deg")
                if isinstance(refine.get("refinement"), Mapping)
                else None,
                "before_median_reprojection_error_px": (refine.get("refinement") or {}).get("before_median_reprojection_error_px")
                if isinstance(refine.get("refinement"), Mapping)
                else None,
                "after_median_reprojection_error_px": (refine.get("refinement") or {}).get("after_median_reprojection_error_px")
                if isinstance(refine.get("refinement"), Mapping)
                else None,
            },
        },
        "diagnostics": {
            "anchor_monotonic": {
                "enabled": anchor_monotonic.get("enabled"),
                "success": anchor_monotonic.get("success"),
                "reason": anchor_monotonic.get("reason"),
                "selected_alpha": anchor_monotonic.get("selected_alpha"),
                "sparse_anchor_median_px": anchor_monotonic.get("sparse_anchor_median_px"),
                "selected_anchor_median_px": anchor_monotonic.get("selected_anchor_median_px"),
                "epsilon_px": anchor_monotonic.get("epsilon_px"),
            } if anchor_monotonic else None,
        },
    }


def _select_render_label_for_dense(
    *,
    clean_render_selection: Mapping[str, Any],
    repair_selection: Mapping[str, Any],
    apd_dense: bool,
) -> str:
    if bool(apd_dense):
        return str(clean_render_selection.get("selected_label", "base") or "base")
    selected_label = str(repair_selection.get("selected_label", "base") or "base")
    if repair_selection.get("decision") not in {"accept_repaired_dense_pose", "accept_original_dense_pose"}:
        return "base"
    return selected_label


def _clean_render_visual_payload(
    *,
    selection: Mapping[str, Any],
    rendered_candidates: Sequence[Mapping[str, Any]],
    ray_depth_rows_fn: Any,
) -> dict[str, Any]:
    """Keep non-serializable clean-render images for diagnostic visualization."""

    if not rendered_candidates:
        return {"selection": dict(selection), "base": None, "selected": None}
    base_idx = int(selection.get("base_index", 0) or 0)
    selected_idx = int(selection.get("selected_index", base_idx) or base_idx)
    base_idx = max(0, min(base_idx, len(rendered_candidates) - 1))
    selected_idx = max(0, min(selected_idx, len(rendered_candidates) - 1))

    def _pack(item: Mapping[str, Any]) -> dict[str, Any]:
        depth = item.get("depth")
        pose = np.asarray(item.get("pose_w2c"), dtype=np.float32).reshape(4, 4)
        return {
            "label": str(item.get("label", "candidate")),
            "render": item.get("render_pkg", {}).get("render") if isinstance(item.get("render_pkg"), Mapping) else None,
            "preflight": item.get("preflight"),
            "render_control": item.get("render_control"),
            "ray_depth_diagnostics": ray_depth_rows_fn(depth, pose) if depth is not None else [],
        }

    return {
        "selection": dict(selection),
        "base": _pack(rendered_candidates[base_idx]),
        "selected": _pack(rendered_candidates[selected_idx]),
    }


def _capture_sparse(ctx: dict[str, Any], query_image: torch.Tensor, fovx: float, fovy: float) -> dict[str, Any]:
    stdloc = ctx["stdloc"]
    loc = ctx["localizer"]
    fine_map, coarse_map = loc.get_feature_map(query_image)
    H, W = fine_map.shape[-2:]
    heat = loc.detector(fine_map)
    kp_scores = stdloc.simple_nms(heat, loc.config["sparse"].get("nms", 4)).flatten()
    _, kp_ids = torch.topk(kp_scores, loc.config["sparse"].get("detect_num", 2048))
    kp_ids = kp_ids[(kp_scores > 0)[kp_ids]]
    kp_mask = torch.zeros_like(kp_scores, dtype=torch.bool)
    kp_mask[kp_ids] = True
    sampled_features = fine_map.reshape(fine_map.shape[0], -1)[:, kp_mask]
    landmark_features = F.normalize(loc.landmarks.get_loc_feature.squeeze(), dim=-1)
    corr = torch.matmul(sampled_features.T, landmark_features.T)
    corr = stdloc.apply_landmark_prior(
        corr,
        loc.landmark_prior,
        weight=loc.config["sparse"].get("landmark_prior_weight", 0.0),
    )
    if loc.config["sparse"]["dual_softmax"] is True:
        corr = stdloc.dual_softmax(corr, temp=loc.config["sparse"]["dual_softmax_temp"])
    if loc.config["sparse"]["mnn_match"] is True:
        _b, im_idx, gs_ids = stdloc.mnn_match(corr[None], thr=loc.config["sparse"]["threshold"])
        im_idx = im_idx.reshape(-1)
        gs_ids = gs_ids.reshape(-1)
        scores = corr[im_idx, gs_ids] if im_idx.numel() else corr.new_empty(0)
    else:
        im_idx, gs_ids, scores = stdloc.topk_match(
            corr[None],
            loc.config["sparse"]["topk"],
            thr=loc.config["sparse"]["threshold"],
        )
        im_idx = im_idx.reshape(-1)
        gs_ids = gs_ids.reshape(-1)
        scores = scores.reshape(-1)
    grid = torch.stack([torch.arange(H * W) % W, torch.arange(H * W) // W], dim=1)
    p2d = grid[kp_mask.cpu()][im_idx.cpu()].numpy().astype(np.float32)
    p3d = loc.landmarks.get_xyz[gs_ids].detach().cpu().numpy().astype(np.float32)
    K = stdloc.get_intrinsic(fovx, fovy, W, H)
    pose, inliers = stdloc.solve_pose(
        p2d + 0.5,
        p3d,
        K,
        loc.config["sparse"]["solver"],
        loc.config["sparse"]["reprojection_error"],
        loc.config["sparse"]["confidence"],
        loc.config["sparse"]["max_iterations"],
        loc.config["sparse"]["min_iterations"],
    )
    return {
        "fine_map": fine_map,
        "coarse_map": coarse_map,
        "query_xy": p2d + 0.5,
        "p3d": p3d,
        "gs_ids": gs_ids.detach().cpu().numpy().astype(np.int64),
        "scores": scores.detach().cpu().numpy().astype(np.float32),
        "pose_w2c": pose,
        "inliers": np.asarray(inliers).reshape(-1).astype(np.int32),
        "K": K,
        "width": W,
        "height": H,
    }


def _capture_dense(
    ctx: dict[str, Any],
    query_coarse: torch.Tensor,
    query_fine: torch.Tensor,
    sparse_pose: np.ndarray,
    fovx: float,
    fovy: float,
    *,
    sparse_capture: dict[str, Any] | None = None,
    slcdp_repair_search: bool = False,
    slcdp_repair_skip_if_no_accept: bool = False,
    slcdp_repair_max_candidates: int = 37,
    slcdp_repair_translation_steps_m: tuple[float, ...] = SLCDPRepairSearchConfig.translation_steps_m,
    slcdp_repair_rotation_steps_deg: tuple[float, ...] = SLCDPRepairSearchConfig.rotation_steps_deg,
    slcdp_repair_require_accept: bool = True,
    slcdp_repair_min_score_gain: float = 0.05,
    slcdp_repair_translation_penalty_per_m: float = 0.0,
    slcdp_allow_low_confidence_gated_base: bool = False,
    slcdp_low_confidence_gated_base_min_score_gain: float = 0.25,
    slcdp_render_control: str = "none",
    slcdp_guided_pose_steps_m: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0),
    slcdp_gating_radius_px: float = 4.0,
    slcdp_gating_depth_margin_m: float = 1.0,
    slcdp_gating_footprint_radius_scale: float = 0.0,
    slcdp_gating_max_footprint_radius_px: float = 64.0,
    slcdp_gating_min_footprint_radius_px: float = 0.0,
    slcdp_gating_min_depth_m: float = 0.0,
    slcdp_gating_max_depth_m: float = 0.0,
    slcdp_gating_min_opacity: float = 0.0,
    slcdp_gating_chunk_size: int = 65536,
    slcdp_fast_guided_max_render_candidates: int = 4,
    slcdp_fast_guided_conflict_radius_px: float = 8.0,
    slcdp_fast_guided_depth_margin_m: float = 1.0,
    slcdp_transition_control: bool = False,
    slcdp_soft_transition_control: bool = False,
    slcdp_transition_max_reprojection_error_px: float = 8.0,
    slcdp_transition_min_retained_ratio: float = 0.90,
    slcdp_transition_max_translation_delta_m: float = 0.35,
    slcdp_transition_max_rotation_delta_deg: float = 5.0,
    slcdp_transition_line_search_fractions: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0),
    slcdp_base_dense_capture: Mapping[str, Any] | None = None,
    apd_dense: bool = False,
    apd_include_patch_candidates: bool = False,
    apd_dense_group_weight: float = 1.0,
    apd_anchor_group_weight: float = 1.0,
    apd_max_translation_delta_m: float = 0.0,
    apd_max_rotation_delta_deg: float = 0.0,
    apd_anchor_monotonic: bool = False,
    apd_anchor_monotonic_epsilon_px: float = 1.0,
    apd_use_refined_pose: bool = False,
    apd_risk_weighted: bool = False,
    apd_risk_max_beta: float = 1.0,
    apd_no_regression_gate: bool = True,
    apd_gate_min_inlier_ratio_delta: float = 0.02,
    apd_gate_max_median_reproj_increase_px: float = 1.0,
    apd_gate_max_p90_reproj_increase_px: float = 3.0,
    apd_gate_min_inlier_count_ratio: float = 0.90,
) -> dict[str, Any]:
    stdloc = ctx["stdloc"]
    loc = ctx["localizer"]
    Hf, Wf = query_fine.shape[-2:]
    coarse_query = query_coarse
    Hc, Wc = coarse_query.shape[-2:]
    W = Hf // Hc
    C = loc.feature_extractor.feature_dim
    K = stdloc.get_intrinsic(fovx, fovy, Wf, Hf)
    render_control_mode = str(slcdp_render_control or "none")
    behavior_render_control_mode = (
        SPARSE_CONDITIONED_EFFECTIVE_RENDER_CONTROL
        if render_control_mode == SPARSE_CONDITIONED_RENDER_CONTROL
        else render_control_mode
    )
    gaussian_xyz_np: np.ndarray | None = None
    gaussian_scale_np: np.ndarray | None = None
    gaussian_opacity_np: np.ndarray | None = None

    def _load_gaussian_geometry() -> None:
        nonlocal gaussian_xyz_np, gaussian_scale_np, gaussian_opacity_np
        if gaussian_xyz_np is None:
            gaussian_xyz_np = loc.gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
        if gaussian_scale_np is None:
            gaussian_scale_np = loc.gaussians.get_scaling.detach().cpu().numpy().astype(np.float32)
        if gaussian_opacity_np is None:
            gaussian_opacity_np = loc.gaussians.get_opacity.detach().cpu().numpy().reshape(-1).astype(np.float32)

    def _ray_depth_rows(render_depth: torch.Tensor | np.ndarray, render_pose: np.ndarray) -> list[dict[str, Any]]:
        if sparse_capture is None:
            return []
        try:
            depth_np = render_depth.detach().cpu().numpy() if isinstance(render_depth, torch.Tensor) else np.asarray(render_depth)
            return compute_sparse_ray_depth_diagnostics(
                sparse_query_xy=sparse_capture["query_xy"],
                sparse_points_world=sparse_capture["p3d"],
                sparse_pose_w2c=np.asarray(render_pose, dtype=np.float32).reshape(4, 4),
                intrinsic=K,
                render_depth=np.asarray(depth_np).squeeze(),
                sparse_inlier_indices=sparse_capture["inliers"],
                thresholds=SLCDPThresholds(),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            return [{"error": str(exc)}]

    def _render_pc_for_candidate(candidate: dict[str, Any]) -> tuple[Any, dict[str, Any] | None]:
        candidate_pose = np.asarray(candidate["pose_w2c"], dtype=np.float32).reshape(4, 4)
        if (
            not _should_apply_slcdp_gating(
                render_control_mode=render_control_mode,
                behavior_render_control_mode=behavior_render_control_mode,
                candidate=candidate,
            )
            or sparse_capture is None
        ):
            return loc.gaussians, None
        _load_gaussian_geometry()
        protected = None
        if sparse_capture.get("gs_ids") is not None:
            gs_ids = np.asarray(sparse_capture["gs_ids"], dtype=np.int64).reshape(-1)
            inliers = np.asarray(sparse_capture.get("inliers", []), dtype=np.int64).reshape(-1)
            inliers = inliers[(inliers >= 0) & (inliers < gs_ids.shape[0])]
            protected = gs_ids[inliers] if inliers.size else None
        try:
            gating = compute_sparse_ray_gaussian_gating_mask(
                gaussian_xyz=gaussian_xyz_np,
                gaussian_scale_world=gaussian_scale_np,
                gaussian_opacity=gaussian_opacity_np,
                sparse_query_xy=sparse_capture["query_xy"],
                sparse_points_world=sparse_capture["p3d"],
                sparse_pose_w2c=candidate_pose,
                intrinsic=K,
                sparse_inlier_indices=sparse_capture["inliers"],
                image_size=(int(Wf), int(Hf)),
                radius_px=float(slcdp_gating_radius_px),
                depth_margin_m=float(slcdp_gating_depth_margin_m),
                footprint_radius_scale=float(slcdp_gating_footprint_radius_scale),
                max_footprint_radius_px=float(slcdp_gating_max_footprint_radius_px),
                min_conflict_footprint_radius_px=float(slcdp_gating_min_footprint_radius_px),
                min_conflict_depth_m=float(slcdp_gating_min_depth_m),
                max_conflict_depth_m=(
                    None
                    if float(slcdp_gating_max_depth_m) <= 0.0
                    else float(slcdp_gating_max_depth_m)
                ),
                min_opacity=float(slcdp_gating_min_opacity),
                protected_gaussian_indices=protected,
                chunk_size=int(slcdp_gating_chunk_size),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            return loc.gaussians, {"error": str(exc), "diagnostic_only": True}
        keep_mask_np = np.asarray(gating["keep_mask"], dtype=bool)
        if int(keep_mask_np.sum()) <= 0:
            return loc.gaussians, {**_compact_gating_result(gating), "disabled_reason": "empty_keep_mask"}
        keep_mask = torch.as_tensor(keep_mask_np, device=loc.gaussians.get_xyz.device)
        return _MaskedGaussianProxy(loc.gaussians, keep_mask), _compact_gating_result(gating)

    def _render_and_preflight(candidate: dict[str, Any]) -> dict[str, Any]:
        candidate_pose = np.asarray(candidate["pose_w2c"], dtype=np.float32).reshape(4, 4)
        render_pc, gating_metadata = _render_pc_for_candidate(candidate)
        render = stdloc.render_from_pose_gsplat(
            render_pc,
            torch.tensor(candidate_pose, device="cuda"),
            fovx,
            fovy,
            Wf,
            Hf,
            render_mode="RGB+ED",
            norm_feat_bf_render=loc.config["dense"]["norm_before_render"],
            rasterize_mode="antialiased",
        )
        candidate_depth = render["depth"].squeeze()
        candidate_fine = render["feature_map"]
        if candidate_fine is None or (candidate_fine == 0).all():
            preflight = {"decision": "retry_sparse_or_patch_dense", "reason": "empty_render_feature"}
        elif sparse_capture is None:
            preflight = None
        else:
            try:
                preflight = sparse_landmark_conditioned_preflight(
                    sparse_query_xy=sparse_capture["query_xy"],
                    sparse_points_world=sparse_capture["p3d"],
                    sparse_pose_w2c=candidate_pose,
                    intrinsic=K,
                    render_depth=candidate_depth.detach().cpu().numpy(),
                    query_features=query_fine.detach().cpu().numpy(),
                    render_features=candidate_fine.detach().cpu().numpy(),
                    sparse_inlier_indices=sparse_capture["inliers"],
                    sparse_scores=sparse_capture.get("scores"),
                    thresholds=SLCDPThresholds(),
                )
            except (RuntimeError, ValueError, KeyError) as exc:
                preflight = {"decision": "retry_sparse_or_patch_dense", "reason": str(exc)}
        return {
            "label": str(candidate.get("label", "candidate")),
            "pose_w2c": candidate_pose,
            "selection_mode": candidate.get("selection_mode"),
            "render_rank": candidate.get("render_rank"),
            "translation_cam_m": candidate.get("translation_cam_m"),
            "rotation_deg": candidate.get("rotation_deg"),
            "fast_score": candidate.get("fast_score"),
            "fast_pool_candidate_count": candidate.get("fast_pool_candidate_count"),
            "render_pkg": render,
            "depth": candidate_depth,
            "fine_render": candidate_fine,
            "preflight": preflight,
            "render_control": {
                "mode": render_control_mode,
                "behavior_mode": behavior_render_control_mode,
                "gaussian_gating": gating_metadata,
                "candidate_label": str(candidate.get("label", "candidate")),
                "gating_applied": bool(gating_metadata is not None and "disabled_reason" not in gating_metadata),
                "gating_requested": bool(candidate.get("slcdp_apply_gating", False)),
            },
        }

    repair_config = SLCDPRepairSearchConfig(
        translation_steps_m=tuple(float(v) for v in slcdp_repair_translation_steps_m),
        rotation_steps_deg=tuple(float(v) for v in slcdp_repair_rotation_steps_deg),
        max_candidates=int(slcdp_repair_max_candidates),
        min_score_gain=float(slcdp_repair_min_score_gain),
        translation_penalty_per_m=float(slcdp_repair_translation_penalty_per_m),
    )
    if behavior_render_control_mode in {"fast_guided_pose", "fast_gating_guided_pose"} and sparse_capture is not None:
        rendered_candidates: list[dict[str, Any]] | None = None
        if render_control_mode == SPARSE_CONDITIONED_RENDER_CONTROL:
            base_candidate = {
                "label": "base",
                "pose_w2c": np.asarray(sparse_pose, dtype=np.float32).reshape(4, 4),
                "translation_cam_m": (0.0, 0.0, 0.0),
                "rotation_deg": (0.0, 0.0, 0.0),
                "selection_mode": "sparse_conditioned_native_base",
                "render_rank": 0,
                "slcdp_apply_gating": False,
            }
            base_rendered = _render_and_preflight(base_candidate)
            base_reuse_decision = _should_reuse_sparse_conditioned_base_dense(
                base_rendered.get("preflight"),
                slcdp_base_dense_capture,
            )
            if bool(base_reuse_decision.get("reuse_base")):
                if slcdp_base_dense_capture is not None:
                    return _reuse_sparse_conditioned_base_dense_capture(
                        slcdp_base_dense_capture,
                        base_rendered,
                        reuse_decision=base_reuse_decision,
                    )
                rendered_candidates = [base_rendered]
                candidates = []
            else:
                selected_inliers = np.asarray(sparse_capture["inliers"], dtype=np.int64).reshape(-1)
                selected_inliers = selected_inliers[(selected_inliers >= 0) & (selected_inliers < len(sparse_capture["p3d"]))]
                guide_points = sparse_capture["p3d"][selected_inliers] if selected_inliers.size else sparse_capture["p3d"]
                _load_gaussian_geometry()
                guided_candidates = generate_fast_landmark_guided_pose_candidates(
                    sparse_pose,
                    sparse_points_world=guide_points,
                    gaussian_xyz=gaussian_xyz_np,
                    gaussian_scale_world=gaussian_scale_np,
                    intrinsic=K,
                    image_size=(int(Wf), int(Hf)),
                    steps_m=tuple(float(v) for v in slcdp_guided_pose_steps_m),
                    max_render_candidates=int(slcdp_fast_guided_max_render_candidates),
                    conflict_radius_px=float(slcdp_fast_guided_conflict_radius_px),
                    depth_margin_m=float(slcdp_fast_guided_depth_margin_m),
                    footprint_radius_scale=float(slcdp_gating_footprint_radius_scale),
                    max_footprint_radius_px=float(slcdp_gating_max_footprint_radius_px),
                    pool_max_candidates=int(slcdp_repair_max_candidates),
                    chunk_size=int(slcdp_gating_chunk_size),
                )
                rendered_candidates = [base_rendered]
                candidates = []
                gated_candidates: list[dict[str, Any]] = []
                for candidate in guided_candidates:
                    gated = dict(candidate)
                    gated["slcdp_apply_gating"] = True
                    gated["selection_mode"] = f"sparse_conditioned_{gated.get('selection_mode', 'guided')}"
                    gated["label"] = f"gated_{gated.get('label', 'candidate')}"
                    gated_candidates.append(gated)
                candidates = gated_candidates
        else:
            selected_inliers = np.asarray(sparse_capture["inliers"], dtype=np.int64).reshape(-1)
            selected_inliers = selected_inliers[(selected_inliers >= 0) & (selected_inliers < len(sparse_capture["p3d"]))]
            guide_points = sparse_capture["p3d"][selected_inliers] if selected_inliers.size else sparse_capture["p3d"]
            _load_gaussian_geometry()
            guided_candidates = generate_fast_landmark_guided_pose_candidates(
                sparse_pose,
                sparse_points_world=guide_points,
                gaussian_xyz=gaussian_xyz_np,
                gaussian_scale_world=gaussian_scale_np,
                intrinsic=K,
                image_size=(int(Wf), int(Hf)),
                steps_m=tuple(float(v) for v in slcdp_guided_pose_steps_m),
                max_render_candidates=int(slcdp_fast_guided_max_render_candidates),
                conflict_radius_px=float(slcdp_fast_guided_conflict_radius_px),
                depth_margin_m=float(slcdp_fast_guided_depth_margin_m),
                footprint_radius_scale=float(slcdp_gating_footprint_radius_scale),
                max_footprint_radius_px=float(slcdp_gating_max_footprint_radius_px),
                pool_max_candidates=int(slcdp_repair_max_candidates),
                chunk_size=int(slcdp_gating_chunk_size),
            )
            candidates = guided_candidates
        repair_enabled = True
    elif behavior_render_control_mode in {"guided_pose", "gating_guided_pose"} and sparse_capture is not None:
        selected_inliers = np.asarray(sparse_capture["inliers"], dtype=np.int64).reshape(-1)
        selected_inliers = selected_inliers[(selected_inliers >= 0) & (selected_inliers < len(sparse_capture["p3d"]))]
        guide_points = sparse_capture["p3d"][selected_inliers] if selected_inliers.size else sparse_capture["p3d"]
        candidates = generate_landmark_guided_pose_candidates(
            sparse_pose,
            sparse_points_world=guide_points,
            steps_m=tuple(float(v) for v in slcdp_guided_pose_steps_m),
            max_candidates=int(slcdp_repair_max_candidates),
        )
        repair_enabled = True
    elif bool(slcdp_repair_search):
        candidates = generate_pose_repair_candidates(sparse_pose, repair_config)
        repair_enabled = True
    else:
        candidates = [
            {
                "label": "base",
                "pose_w2c": np.asarray(sparse_pose, dtype=np.float32).reshape(4, 4),
                "translation_cam_m": (0.0, 0.0, 0.0),
                "rotation_deg": (0.0, 0.0, 0.0),
            }
        ]
        repair_enabled = False
    if "rendered_candidates" not in locals() or rendered_candidates is None:
        rendered_candidates = []
    rendered_candidates.extend(_render_and_preflight(candidate) for candidate in candidates)
    clean_render_selection = select_clean_render_candidate(rendered_candidates)
    clean_render_visualization = _clean_render_visual_payload(
        selection=clean_render_selection,
        rendered_candidates=rendered_candidates,
        ray_depth_rows_fn=_ray_depth_rows,
    )
    repair_selection = select_repaired_pose_candidate(
        rendered_candidates,
        require_accept=bool(slcdp_repair_require_accept),
        min_score_gain=float(repair_config.min_score_gain),
        translation_penalty_per_m=float(repair_config.translation_penalty_per_m),
        allow_low_confidence_gated_base=bool(slcdp_allow_low_confidence_gated_base),
        low_confidence_gated_base_min_score_gain=float(slcdp_low_confidence_gated_base_min_score_gain),
    )
    selected_label = _select_render_label_for_dense(
        clean_render_selection=clean_render_selection,
        repair_selection=repair_selection,
        apd_dense=bool(apd_dense),
    )
    if (
        not bool(apd_dense)
        and
        render_control_mode == SPARSE_CONDITIONED_RENDER_CONTROL
        and str(repair_selection.get("decision")) == "accept_repaired_dense_pose"
    ):
        weak_support_gating_guard = _should_reject_gating_only_repair_for_weak_sparse_support(
            selected_label=selected_label,
            sparse_capture=sparse_capture,
        )
        repair_selection = dict(repair_selection)
        repair_selection["weak_support_gating_guard"] = weak_support_gating_guard
        if bool(weak_support_gating_guard.get("reject_repair")):
            repair_selection.update(
                {
                    "decision": "accept_original_dense_pose",
                    "original_decision": "accept_repaired_dense_pose",
                    "rejected_selected_label": selected_label,
                    "selected_label": "base",
                    "selected_score": float(repair_selection.get("base_score", repair_selection.get("selected_score", 0.0))),
                }
            )
            selected_label = "base"
    if (
        not bool(apd_dense)
        and bool(repair_enabled)
        and repair_selection["decision"] == "skip_dense_keep_sparse"
        and bool(slcdp_repair_skip_if_no_accept)
    ):
        base = rendered_candidates[0]
        if render_control_mode == SPARSE_CONDITIONED_RENDER_CONTROL and slcdp_base_dense_capture is not None:
            return _reuse_base_dense_after_rejected_sparse_conditioned_repair(
                base_dense_capture=slcdp_base_dense_capture,
                base_rendered=base,
                repair_selection=repair_selection,
                rendered_candidates=rendered_candidates,
            )
        empty = np.empty((0, 2), dtype=np.float32)
        return {
            "query_xy": empty,
            "rendered_xy": empty,
            "p3d": np.empty((0, 3), dtype=np.float32),
            "pose_w2c": np.asarray(sparse_pose, dtype=np.float32).reshape(4, 4),
            "inliers": np.empty(0, dtype=np.int32),
            "K": K,
            "render": base["render_pkg"].get("render"),
            "ray_depth_diagnostics": _ray_depth_rows(base["depth"], base["pose_w2c"]),
            "slcdp_render_control": base.get("render_control"),
            "slcdp_preflight": base.get("preflight"),
            "slcdp_repair_search": {
                "enabled": bool(repair_enabled),
                "clean_render_generation": clean_render_selection,
                "selection": repair_selection,
                "candidates": _slcdp_candidate_audit_rows(rendered_candidates),
                "render_candidate_count": int(len(rendered_candidates)),
                "dense_skipped": True,
            },
            "_clean_render_visualization": clean_render_visualization,
        }
    selected = next((item for item in rendered_candidates if item["label"] == selected_label), rendered_candidates[0])
    render_pose = np.asarray(selected["pose_w2c"], dtype=np.float32).reshape(4, 4)
    render_pkg = selected["render_pkg"]
    depth = render_pkg["depth"].squeeze()
    fine_render = selected["fine_render"]
    slcdp_preflight = selected.get("preflight")
    slcdp_repair_metadata = {
        "enabled": bool(repair_enabled),
        "clean_render_generation": clean_render_selection,
        "selection": repair_selection,
        "candidates": [
            {
                "label": item["label"],
                "selection_mode": item.get("selection_mode"),
                "render_rank": item.get("render_rank"),
                "translation_cam_m": item.get("translation_cam_m"),
                "rotation_deg": item.get("rotation_deg"),
                "fast_score": item.get("fast_score"),
                "fast_pool_candidate_count": item.get("fast_pool_candidate_count"),
                "preflight": item.get("preflight"),
                "render_control": item.get("render_control"),
            }
            for item in rendered_candidates
        ],
        "render_candidate_count": int(len(rendered_candidates)),
        "dense_skipped": False,
    }
    if fine_render is None or (fine_render == 0).all():
        empty = np.empty((0, 2), dtype=np.float32)
        return {
            "query_xy": empty,
            "rendered_xy": empty,
            "p3d": np.empty((0, 3), dtype=np.float32),
            "pose_w2c": sparse_pose,
            "inliers": np.empty(0, dtype=np.int32),
            "K": K,
            "render": render_pkg.get("render"),
            "ray_depth_diagnostics": _ray_depth_rows(depth, render_pose),
            "slcdp_render_control": selected.get("render_control"),
            "slcdp_preflight": {"decision": "retry_sparse_or_patch_dense", "reason": "empty_render_feature"},
            "slcdp_repair_search": slcdp_repair_metadata,
            "_clean_render_visualization": clean_render_visualization,
        }
    coarse_render = F.interpolate(fine_render[None], size=(Hc, Wc), mode="bilinear", align_corners=False)[0]
    coarse_render = F.normalize(coarse_render, dim=0)
    coarse_corr = torch.matmul(
        coarse_query.permute(1, 2, 0).reshape(1, -1, C),
        coarse_render.reshape(1, C, -1),
    )
    coarse_corr = stdloc.dual_softmax(coarse_corr, temp=loc.config["dense"]["coarse_dual_softmax_temp"])
    c_b, c_i, c_j = stdloc.mnn_match(coarse_corr, thr=loc.config["dense"]["coarse_threshold"])
    c_b = c_b.reshape(-1)
    c_i = c_i.reshape(-1)
    c_j = c_j.reshape(-1)
    if c_i.numel() < 3:
        empty = np.empty((0, 2), dtype=np.float32)
        return {
            "query_xy": empty,
            "rendered_xy": empty,
            "p3d": np.empty((0, 3), dtype=np.float32),
            "pose_w2c": sparse_pose,
            "inliers": np.empty(0, dtype=np.int32),
            "K": K,
            "render": render_pkg.get("render"),
            "ray_depth_diagnostics": _ray_depth_rows(depth, render_pose),
            "slcdp_render_control": selected.get("render_control"),
            "slcdp_preflight": slcdp_preflight,
            "slcdp_repair_search": slcdp_repair_metadata,
            "_clean_render_visualization": clean_render_visualization,
        }

    query_windows = F.unfold(query_fine, (W, W), stride=W).reshape(1, C, W * W, -1)[c_b, :, :, c_i].permute(0, 2, 1)
    rendered_windows = F.unfold(fine_render, (W, W), stride=W).reshape(1, C, W * W, -1)[c_b, :, :, c_j].permute(0, 2, 1)
    fine_scores = torch.matmul(query_windows, rendered_windows.transpose(-2, -1))
    fine_corr = stdloc.dual_softmax(fine_scores, temp=loc.config["dense"]["fine_dual_softmax_temp"])
    f_b, f_i, f_j = stdloc.mnn_match(fine_corr, thr=loc.config["dense"]["fine_threshold"])
    f_b = f_b.reshape(-1)
    f_i = f_i.reshape(-1)
    f_j = f_j.reshape(-1)
    if f_i.numel() < 3:
        empty = np.empty((0, 2), dtype=np.float32)
        return {
            "query_xy": empty,
            "rendered_xy": empty,
            "p3d": np.empty((0, 3), dtype=np.float32),
            "pose_w2c": sparse_pose,
            "inliers": np.empty(0, dtype=np.int32),
            "K": K,
            "render": render_pkg.get("render"),
            "ray_depth_diagnostics": _ray_depth_rows(depth, render_pose),
            "slcdp_render_control": selected.get("render_control"),
            "slcdp_preflight": slcdp_preflight,
            "slcdp_repair_search": slcdp_repair_metadata,
            "_clean_render_visualization": clean_render_visualization,
        }
    if loc.config["dense"].get("subpixel_refine", False):
        temp = loc.config["dense"].get("subpixel_temperature", 0.1)
        query_offsets = stdloc.soft_argmax_offsets(fine_scores[f_b, :, f_j], W, temperature=temp)
        rendered_offsets = stdloc.soft_argmax_offsets(fine_scores[f_b, f_i, :], W, temperature=temp)
    else:
        query_offsets = torch.stack([f_i % W, f_i // W], dim=1).float()
        rendered_offsets = torch.stack([f_j % W, f_j // W], dim=1).float()
    query_origins = torch.stack([c_i[f_b] % Wc * W, c_i[f_b] // Wc * W], dim=1).float()
    rendered_origins = torch.stack([c_j[f_b] % Wc * W, c_j[f_b] // Wc * W], dim=1).float()
    query_xy = query_origins + query_offsets
    rendered_xy = rendered_origins + rendered_offsets
    p3d = stdloc.lift_2d_to_3d(
        rendered_xy,
        torch.tensor(K, device="cuda"),
        torch.tensor(np.linalg.inv(render_pose), device="cuda"),
        depth,
        interpolation=loc.config["dense"].get("depth_interpolation", "nearest"),
    )
    p2d_np = query_xy.detach().cpu().numpy().astype(np.float32)
    p3d_np = p3d.detach().cpu().numpy().astype(np.float32)
    fine_scores_np = fine_corr[f_b, f_i, f_j].detach().cpu().numpy().astype(np.float32) if f_i.numel() else np.empty((0,), dtype=np.float32)
    pose, inliers = stdloc.solve_pose(
        p2d_np + 0.5,
        p3d_np,
        K,
        loc.config["dense"]["solver"],
        loc.config["dense"]["reprojection_error"],
        loc.config["dense"]["confidence"],
        loc.config["dense"]["max_iterations"],
        loc.config["dense"]["min_iterations"],
    )
    transition_control = None
    final_pose = pose
    apd_result = None
    apd_switch = None
    if bool(apd_dense) and sparse_capture is not None:
        apd_policy = APDDensePolicy(
            dense_group_weight=float(apd_dense_group_weight),
            anchor_group_weight=float(apd_anchor_group_weight),
            max_translation_delta_m=None if float(apd_max_translation_delta_m) <= 0.0 else float(apd_max_translation_delta_m),
            max_rotation_delta_deg=None if float(apd_max_rotation_delta_deg) <= 0.0 else float(apd_max_rotation_delta_deg),
            anchor_monotonic=bool(apd_anchor_monotonic),
            anchor_monotonic_epsilon_px=float(apd_anchor_monotonic_epsilon_px),
        )
        sparse_ref_xy, _sparse_ref_valid = project_points(
            sparse_capture["p3d"],
            render_pose,
            K,
            width=int(Wf),
            height=int(Hf),
        )
        apd_anchors = {
            "query_xy": sparse_capture["query_xy"],
            "render_xy": sparse_ref_xy,
            "p3d": sparse_capture["p3d"],
            "inliers": sparse_capture["inliers"],
        }
        current_dense_candidates = {
            "query_xy": p2d_np + 0.5,
            "p3d": p3d_np,
            "match_scores": fine_scores_np,
        }
        selected_is_clean = render_control_mode != "none" and str(selected.get("label", selected_label)) != "base"
        patch_candidates = None
        if bool(apd_include_patch_candidates):
            patch_candidates = generate_patch_dense_candidates(
                query_fine.detach().cpu().numpy(),
                fine_render.detach().cpu().numpy(),
                depth.detach().cpu().numpy(),
                apd_anchors,
                policy=PatchDenseCandidatePolicy(),
                render_pose_w2c=render_pose,
                intrinsic=K,
            )
        apd_result = run_anchor_patch_dense_refinement(
            sparse_pose=sparse_pose,
            dense_reference_pose=pose,
            sparse_anchors=apd_anchors,
            intrinsic=K,
            image_size=(int(Wf), int(Hf)),
            global_dense_candidates=None if selected_is_clean else current_dense_candidates,
            clean_render_dense_candidates=current_dense_candidates if selected_is_clean else None,
            patch_dense_candidates=patch_candidates,
            policy=apd_policy,
        )
        apd_pose = np.asarray(apd_result["final_pose"], dtype=np.float32).reshape(4, 4)
        apd_switch = {
            "schema": "loc_gs_apd_pose_switch_v1",
            "mode": "apd_diagnostic_only_keep_native",
            "accept_apd_pose": False,
            "reason": "apd_pose_not_enabled_as_final",
            "apd_use_refined_pose": bool(apd_use_refined_pose),
            "apd_risk_weighted": bool(apd_risk_weighted),
            "apd_no_regression_gate": bool(apd_no_regression_gate),
            "diagnostic_only": True,
        }
        if bool(apd_use_refined_pose):
            native_capture = {
                "query_xy": p2d_np + 0.5,
                "p3d": p3d_np,
                "inliers": np.asarray(inliers).reshape(-1).astype(np.int32),
                "pose_w2c": pose,
                "K": K,
            }
            native_capture["dense_pose_quality"] = _dense_pose_quality(native_capture)
            if bool(apd_risk_weighted):
                risk_observation = _dense_damage_risk_observation(
                    sparse_capture=sparse_capture,
                    dense_capture=native_capture,
                    intrinsic=K,
                    image_size=(int(Wf), int(Hf)),
                )
                risk_report = compute_dense_damage_risk(risk_observation)
                apd_confidence = 1.0 if bool(apd_result.get("dense_refine_success", False)) else 0.0
                beta = float(
                    np.clip(
                        float(risk_report.get("risk", 0.0))
                        * max(float(apd_risk_max_beta), 0.0)
                        * apd_confidence,
                        0.0,
                        1.0,
                    )
                )
                final_pose = interpolate_w2c_poses(pose, apd_pose, beta)
                apd_switch = {
                    "schema": "loc_gs_apd_pose_switch_v1",
                    "mode": "apd_use_refined_pose_risk_weighted",
                    "accept_apd_pose": bool(beta > 0.0),
                    "reason": "risk_weighted_apd_pose_blend",
                    "beta": beta,
                    "risk": float(risk_report.get("risk", 0.0)),
                    "risk_report": risk_report,
                    "risk_observation": risk_observation,
                    "apd_confidence": apd_confidence,
                    "apd_risk_max_beta": float(apd_risk_max_beta),
                    "native_dense_default": True,
                    "apd_use_refined_pose": True,
                    "apd_risk_weighted": True,
                    "apd_no_regression_gate": bool(apd_no_regression_gate),
                    "diagnostic_only": True,
                }
            elif bool(apd_no_regression_gate):
                gate = _evaluate_apd_no_regression_switch(
                    native_capture=native_capture,
                    apd_pose_w2c=apd_pose,
                    apd_refine_success=bool(apd_result.get("dense_refine_success", False)),
                    min_inlier_ratio_delta=float(apd_gate_min_inlier_ratio_delta),
                    max_median_reproj_increase_px=float(apd_gate_max_median_reproj_increase_px),
                    max_p90_reproj_increase_px=float(apd_gate_max_p90_reproj_increase_px),
                    min_inlier_count_ratio=float(apd_gate_min_inlier_count_ratio),
                )
                accept_apd = bool(gate.get("accept_apd_pose", False))
                apd_switch = {
                    "schema": "loc_gs_apd_pose_switch_v1",
                    "mode": "apd_use_refined_pose_with_gate",
                    "accept_apd_pose": accept_apd,
                    "reason": str(gate.get("decision", "reject_apd_pose_no_regression_gate")),
                    "gate": gate,
                    "apd_use_refined_pose": True,
                    "apd_risk_weighted": False,
                    "apd_no_regression_gate": True,
                    "diagnostic_only": True,
                }
                if accept_apd:
                    final_pose = apd_pose
            else:
                final_pose = apd_pose
                apd_switch = {
                    "schema": "loc_gs_apd_pose_switch_v1",
                    "mode": "apd_use_refined_pose_no_gate",
                    "accept_apd_pose": True,
                    "reason": "apd_pose_forced_without_gate",
                    "apd_use_refined_pose": True,
                    "apd_risk_weighted": False,
                    "apd_no_regression_gate": False,
                    "diagnostic_only": True,
                }
    apply_transition_control = True
    if render_control_mode == SPARSE_CONDITIONED_RENDER_CONTROL:
        apply_transition_control = _should_apply_sparse_conditioned_transition_control(
            selected_label=str(selected.get("label", selected_label)),
            repair_selection=repair_selection,
        )
    if (bool(slcdp_transition_control) or bool(slcdp_soft_transition_control)) and bool(apply_transition_control) and sparse_capture is not None:
        transition_policy = DenseTransitionPolicy(
            max_reprojection_error_px=float(slcdp_transition_max_reprojection_error_px),
            min_retained_ratio=float(slcdp_transition_min_retained_ratio),
            max_translation_delta_m=float(slcdp_transition_max_translation_delta_m),
            max_rotation_delta_deg=float(slcdp_transition_max_rotation_delta_deg),
            line_search_fractions=tuple(float(v) for v in slcdp_transition_line_search_fractions),
        )
        if bool(slcdp_soft_transition_control):
            transition_control = select_soft_sparse_conditioned_dense_transition(
                sparse_query_xy=sparse_capture["query_xy"],
                sparse_points_world=sparse_capture["p3d"],
                sparse_pose_w2c=sparse_pose,
                dense_pose_w2c=pose,
                intrinsic=K,
                sparse_inlier_indices=sparse_capture["inliers"],
                preflight=slcdp_preflight,
                policy=transition_policy,
                image_size=(int(Wf), int(Hf)),
            )
        else:
            transition_control = select_sparse_conditioned_dense_transition(
                sparse_query_xy=sparse_capture["query_xy"],
                sparse_points_world=sparse_capture["p3d"],
                sparse_pose_w2c=sparse_pose,
                dense_pose_w2c=pose,
                intrinsic=K,
                sparse_inlier_indices=sparse_capture["inliers"],
                policy=transition_policy,
                image_size=(int(Wf), int(Hf)),
            )
        if (
            render_control_mode == SPARSE_CONDITIONED_RENDER_CONTROL
            and slcdp_base_dense_capture is not None
            and str(transition_control.get("decision")) == "reject_dense_keep_sparse"
        ):
            base_rendered = next((item for item in rendered_candidates if item.get("label") == "base"), rendered_candidates[0])
            return _reuse_base_dense_after_rejected_sparse_conditioned_transition(
                base_dense_capture=slcdp_base_dense_capture,
                base_rendered=base_rendered,
                repair_selection=repair_selection,
                rendered_candidates=rendered_candidates,
                transition_control=transition_control,
            )
        final_pose = np.asarray(transition_control["selected_pose_w2c"], dtype=np.float32).reshape(4, 4)
    result = {
        "query_xy": p2d_np + 0.5,
        "rendered_xy": rendered_xy.detach().cpu().numpy().astype(np.float32),
        "p3d": p3d_np,
        "raw_pose_w2c": pose,
        "pose_w2c": final_pose,
        "inliers": np.asarray(inliers).reshape(-1).astype(np.int32),
        "K": K,
        "render": render_pkg.get("render"),
        "ray_depth_diagnostics": _ray_depth_rows(depth, render_pose),
        "slcdp_render_control": selected.get("render_control"),
        "slcdp_preflight": slcdp_preflight,
        "slcdp_repair_search": slcdp_repair_metadata,
        "slcdp_transition_control": _compact_transition_result(transition_control),
        "_clean_render_visualization": clean_render_visualization,
        "apd_dense": _compact_apd_result(apd_result),
        "apd_pose_switch": apd_switch,
    }
    result["dense_pose_quality"] = _dense_pose_quality(result)
    if (
        render_control_mode == SPARSE_CONDITIONED_RENDER_CONTROL
        and sparse_capture is not None
        and str(selected.get("label", selected_label)) != "base"
    ):
        low_quality_guard = _should_reject_sparse_conditioned_dense_for_low_quality(
            selected_label=str(selected.get("label", selected_label)),
            repair_selection=repair_selection,
            sparse_capture=sparse_capture,
            dense_quality=result["dense_pose_quality"],
        )
        result["slcdp_dense_quality_guard"] = low_quality_guard
        if bool(low_quality_guard.get("reject_dense")):
            empty = np.empty((0, 2), dtype=np.float32)
            result.update(
                {
                    "query_xy": empty,
                    "rendered_xy": empty,
                    "p3d": np.empty((0, 3), dtype=np.float32),
                    "pose_w2c": np.asarray(sparse_pose, dtype=np.float32).reshape(4, 4),
                    "inliers": np.empty(0, dtype=np.int32),
                    "slcdp_transition_control": _compact_transition_result(
                        {
                            **low_quality_guard,
                            "selected_fraction": 0.0,
                            "sparse_support_strength": "strong",
                            "selected_acceptance": {
                                "decision": "reject_dense_keep_sparse",
                                "reason": low_quality_guard.get("reason"),
                            },
                            "policy": {
                                "source": "sparse_conditioned_dense_quality_guard",
                            },
                            "candidates": [],
                        }
                    ),
                }
            )
            return result
    if (
        render_control_mode == SPARSE_CONDITIONED_RENDER_CONTROL
        and slcdp_base_dense_capture is not None
        and str(selected.get("label", selected_label)) != "base"
    ):
        base_quality = slcdp_base_dense_capture.get("dense_pose_quality")
        if base_quality is None:
            base_quality = _dense_pose_quality(slcdp_base_dense_capture)
        quality_guard = _should_reject_sparse_conditioned_repair_for_dense_quality_regression(
            base_quality,
            result["dense_pose_quality"],
        )
        if bool(quality_guard.get("reject_repair")):
            base_rendered = next((item for item in rendered_candidates if item.get("label") == "base"), rendered_candidates[0])
            quality_transition = dict(transition_control or {})
            quality_transition.update(
                {
                    "decision": "reject_repair_dense_quality_keep_base",
                    "selected_pose_source": "base_dense",
                    "dense_quality_guard": quality_guard,
                    "diagnostic_only": True,
                }
            )
            return _reuse_base_dense_after_rejected_sparse_conditioned_transition(
                base_dense_capture=slcdp_base_dense_capture,
                base_rendered=base_rendered,
                repair_selection=repair_selection,
                rendered_candidates=rendered_candidates,
                transition_control=quality_transition,
            )
    return result


def _case_rows(cases_csv: Path, *, category: str, max_cases: int) -> list[dict[str, Any]]:
    rows = []
    with cases_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            categories = str(row.get("categories", ""))
            if category and category not in categories.split(";"):
                continue
            rows.append(dict(row))
    rows.sort(key=lambda r: float(r.get("candidate_dense_te_cm") or 0.0), reverse=True)
    return rows[: int(max_cases)]


def _write_matches_csv(path: Path, query_xy: np.ndarray, reference_xy: np.ndarray, errors: np.ndarray, inlier_mask: np.ndarray, good_mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "query_x", "query_y", "reference_x", "reference_y", "gt_reprojection_error_px", "solver_inlier", "gt_good"],
        )
        writer.writeheader()
        for i in range(int(query_xy.shape[0])):
            writer.writerow(
                {
                    "index": i,
                    "query_x": float(query_xy[i, 0]),
                    "query_y": float(query_xy[i, 1]),
                    "reference_x": float(reference_xy[i, 0]),
                    "reference_y": float(reference_xy[i, 1]),
                    "gt_reprojection_error_px": float(errors[i]) if np.isfinite(errors[i]) else "inf",
                    "solver_inlier": bool(inlier_mask[i]),
                    "gt_good": bool(good_mask[i]),
                }
            )


def _depth_class_color(depth_class: str) -> tuple[int, int, int]:
    if depth_class == "depth_agree":
        return (0, 170, 0)
    if depth_class == "near_occluder":
        return (220, 30, 30)
    if depth_class == "far_surface_or_hole":
        return (40, 110, 255)
    if depth_class == "missing_depth":
        return (235, 180, 0)
    return (150, 150, 150)


def _write_ray_depth_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "query_x",
        "query_y",
        "projected_x",
        "projected_y",
        "in_bounds",
        "landmark_depth_m",
        "rendered_depth_m",
        "signed_depth_error_m",
        "abs_depth_error_m",
        "depth_class",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if "error" in row:
                continue
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _draw_ray_depth_canvas(
    query_image: Image.Image,
    render_image: Image.Image,
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    max_draw: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = query_image.size
    canvas = Image.new("RGB", (width * 2, height), (255, 255, 255))
    canvas.paste(query_image, (0, 0))
    canvas.paste(render_image, (width, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    clean_rows = [row for row in rows if "error" not in row]
    for row in clean_rows[: int(max_draw)]:
        color = _depth_class_color(str(row.get("depth_class", "")))
        fill = (*color, 210)
        outline = (*color, 255)
        qx = float(row.get("query_x", 0.0))
        qy = float(row.get("query_y", 0.0))
        px = float(row.get("projected_x", 0.0)) + width
        py = float(row.get("projected_y", 0.0))
        radius = 3
        draw.ellipse((qx - radius, qy - radius, qx + radius, qy + radius), fill=fill)
        if bool(row.get("in_bounds", False)):
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=outline, width=2)
            draw.line((qx, qy, px, py), fill=(*color, 95), width=1)
    legend = [
        ("depth_agree", "agree"),
        ("near_occluder", "near"),
        ("far_surface_or_hole", "far/hole"),
        ("missing_depth", "missing"),
        ("out_of_bounds", "oob"),
    ]
    x0 = 8
    y0 = 8
    for idx, (cls, label) in enumerate(legend):
        color = _depth_class_color(cls)
        y = y0 + idx * 16
        draw.rectangle((x0, y, x0 + 10, y + 10), fill=(*color, 220))
        draw.text((x0 + 14, y - 2), label, fill=(255, 255, 255, 230))
    canvas.save(output_path)


def _preflight_short(preflight: Mapping[str, Any] | None) -> str:
    if not isinstance(preflight, Mapping):
        return "preflight: n/a"
    return (
        "vis={vis:.3f} near={near:.3f} feat={feat:.3f} cov={cov} {decision}"
    ).format(
        vis=float(preflight.get("visible_ratio", 0.0) or 0.0),
        near=float(preflight.get("near_occluder_ratio", 0.0) or 0.0),
        feat=float(preflight.get("feature_cosine_median", 0.0) or 0.0),
        cov=int(preflight.get("coverage_grid_cells", 0) or 0),
        decision=str(preflight.get("decision", "unknown")),
    )


def _add_header(image: Image.Image, lines: Sequence[str]) -> Image.Image:
    header_h = 16 + 14 * max(1, len(lines))
    panel = Image.new("RGB", (image.width, image.height + header_h), (20, 20, 20))
    panel.paste(image, (0, header_h))
    draw = ImageDraw.Draw(panel)
    for idx, line in enumerate(lines):
        draw.text((6, 5 + idx * 14), str(line), fill=(245, 245, 245))
    return panel


def _write_clean_render_visualization(
    case_dir: Path,
    *,
    query_image: Image.Image,
    dense_render: Image.Image,
    clean_visualization: Mapping[str, Any] | None,
    max_draw: int,
) -> dict[str, Any] | None:
    if not isinstance(clean_visualization, Mapping):
        return None
    selection = clean_visualization.get("selection")
    base = clean_visualization.get("base")
    selected = clean_visualization.get("selected")
    if not isinstance(selection, Mapping) or not isinstance(base, Mapping) or not isinstance(selected, Mapping):
        return None
    base_render_raw = base.get("render")
    selected_render_raw = selected.get("render")
    if base_render_raw is None or selected_render_raw is None:
        return None
    base_render = _tensor_to_image(base_render_raw, size=query_image.size)
    selected_render = _tensor_to_image(selected_render_raw, size=query_image.size)

    base_path = case_dir / "clean_render_base.jpg"
    selected_path = case_dir / "clean_render_selected.jpg"
    final_path = case_dir / "clean_render_actual_dense.jpg"
    contact_path = case_dir / "clean_render_generator.jpg"
    base_ray_path = case_dir / "clean_render_base_ray_depth.jpg"
    selected_ray_path = case_dir / "clean_render_selected_ray_depth.jpg"
    base_render.save(base_path, quality=92)
    selected_render.save(selected_path, quality=92)
    dense_render.save(final_path, quality=92)
    _draw_ray_depth_canvas(
        query_image,
        base_render,
        list(base.get("ray_depth_diagnostics") or []),
        base_ray_path,
        max_draw=max_draw,
    )
    _draw_ray_depth_canvas(
        query_image,
        selected_render,
        list(selected.get("ray_depth_diagnostics") or []),
        selected_ray_path,
        max_draw=max_draw,
    )

    decision = str(selection.get("decision", "unknown"))
    selected_label = str(selection.get("selected_label", "unknown"))
    score_gain = float(selection.get("score_gain", 0.0) or 0.0)
    base_panel = _add_header(
        base_render,
        [
            f"base render | score={float(selection.get('base_score', 0.0) or 0.0):.3f}",
            _preflight_short(base.get("preflight")),
        ],
    )
    selected_panel = _add_header(
        selected_render,
        [
            f"CleanRender selected: {selected_label} | gain={score_gain:.3f}",
            _preflight_short(selected.get("preflight")),
        ],
    )
    final_panel = _add_header(
        dense_render,
        [
            "actual dense render used by current path",
            "may differ if legacy repair selector chose another candidate",
        ],
    )
    query_panel = _add_header(query_image, [f"query | CleanRender decision={decision}", "diagnostic only"])
    panels = [query_panel, base_panel, selected_panel, final_panel]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    contact_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(contact_path, quality=92)
    return {
        "clean_render_generator": str(contact_path),
        "clean_render_base": str(base_path),
        "clean_render_selected": str(selected_path),
        "clean_render_actual_dense": str(final_path),
        "clean_render_base_ray_depth": str(base_ray_path),
        "clean_render_selected_ray_depth": str(selected_ray_path),
    }


def _summarize_ray_depth(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean_rows = [row for row in rows if "error" not in row]
    classes = [str(row.get("depth_class", "")) for row in clean_rows]
    signed = np.asarray([float(row.get("signed_depth_error_m", np.nan)) for row in clean_rows], dtype=np.float64)
    signed = signed[np.isfinite(signed)]
    return {
        "count": int(len(clean_rows)),
        "depth_agree_count": int(classes.count("depth_agree")),
        "near_occluder_count": int(classes.count("near_occluder")),
        "far_surface_or_hole_count": int(classes.count("far_surface_or_hole")),
        "missing_depth_count": int(classes.count("missing_depth")),
        "out_of_bounds_count": int(classes.count("out_of_bounds")),
        "near_occluder_ratio": float(classes.count("near_occluder") / len(clean_rows)) if clean_rows else 0.0,
        "median_signed_depth_error_m": float(np.median(signed)) if signed.size else None,
    }


def _analyze_case(
    ctx: dict[str, Any],
    row: dict[str, Any],
    output_dir: Path,
    *,
    run_label: str,
    good_px: float,
    max_draw: int,
    slcdp_repair_search: bool,
    slcdp_repair_skip_if_no_accept: bool,
    slcdp_repair_max_candidates: int,
    slcdp_repair_translation_steps_m: tuple[float, ...],
    slcdp_repair_rotation_steps_deg: tuple[float, ...],
    slcdp_repair_require_accept: bool,
    slcdp_repair_min_score_gain: float,
    slcdp_repair_translation_penalty_per_m: float,
    slcdp_allow_low_confidence_gated_base: bool,
    slcdp_low_confidence_gated_base_min_score_gain: float,
    slcdp_render_control: str,
    slcdp_guided_pose_steps_m: tuple[float, ...],
    slcdp_gating_radius_px: float,
    slcdp_gating_depth_margin_m: float,
    slcdp_gating_footprint_radius_scale: float,
    slcdp_gating_max_footprint_radius_px: float,
    slcdp_gating_min_footprint_radius_px: float,
    slcdp_gating_min_depth_m: float,
    slcdp_gating_max_depth_m: float,
    slcdp_gating_min_opacity: float,
    slcdp_gating_chunk_size: int,
    slcdp_fast_guided_max_render_candidates: int,
    slcdp_fast_guided_conflict_radius_px: float,
    slcdp_fast_guided_depth_margin_m: float,
    slcdp_transition_control: bool,
    slcdp_soft_transition_control: bool,
    slcdp_transition_max_reprojection_error_px: float,
    slcdp_transition_min_retained_ratio: float,
    slcdp_transition_max_translation_delta_m: float,
    slcdp_transition_max_rotation_delta_deg: float,
    slcdp_transition_line_search_fractions: tuple[float, ...],
    apd_dense: bool,
    apd_include_patch_candidates: bool,
    apd_dense_group_weight: float,
    apd_anchor_group_weight: float,
    apd_max_translation_delta_m: float,
    apd_max_rotation_delta_deg: float,
    apd_anchor_monotonic: bool,
    apd_anchor_monotonic_epsilon_px: float,
    apd_use_refined_pose: bool,
    apd_risk_weighted: bool,
    apd_risk_max_beta: float,
    apd_no_regression_gate: bool,
    apd_gate_min_inlier_ratio_delta: float,
    apd_gate_max_median_reproj_increase_px: float,
    apd_gate_max_p90_reproj_increase_px: float,
    apd_gate_min_inlier_count_ratio: float,
) -> dict[str, Any]:
    camera = ctx["camera_by_name"].get(row["image_name"])
    if camera is None:
        raise KeyError(f"camera not found for {ctx['scene']} {row['image_name']}")
    query_image = camera.original_image.to("cuda")
    gt_w2c = camera.world_view_transform.transpose(0, 1).detach().cpu().numpy()
    with torch.no_grad():
        sparse = _capture_sparse(ctx, query_image, camera.FoVx, camera.FoVy)
        dense = _capture_dense(
            ctx,
            sparse["coarse_map"],
            sparse["fine_map"],
            sparse["pose_w2c"],
            camera.FoVx,
            camera.FoVy,
            sparse_capture=sparse,
            slcdp_repair_search=slcdp_repair_search,
            slcdp_repair_skip_if_no_accept=slcdp_repair_skip_if_no_accept,
            slcdp_repair_max_candidates=slcdp_repair_max_candidates,
            slcdp_repair_translation_steps_m=slcdp_repair_translation_steps_m,
            slcdp_repair_rotation_steps_deg=slcdp_repair_rotation_steps_deg,
            slcdp_repair_require_accept=slcdp_repair_require_accept,
            slcdp_repair_min_score_gain=slcdp_repair_min_score_gain,
            slcdp_repair_translation_penalty_per_m=slcdp_repair_translation_penalty_per_m,
            slcdp_allow_low_confidence_gated_base=slcdp_allow_low_confidence_gated_base,
            slcdp_low_confidence_gated_base_min_score_gain=slcdp_low_confidence_gated_base_min_score_gain,
            slcdp_render_control=slcdp_render_control,
            slcdp_guided_pose_steps_m=slcdp_guided_pose_steps_m,
            slcdp_gating_radius_px=slcdp_gating_radius_px,
            slcdp_gating_depth_margin_m=slcdp_gating_depth_margin_m,
            slcdp_gating_footprint_radius_scale=slcdp_gating_footprint_radius_scale,
            slcdp_gating_max_footprint_radius_px=slcdp_gating_max_footprint_radius_px,
            slcdp_gating_min_footprint_radius_px=slcdp_gating_min_footprint_radius_px,
            slcdp_gating_min_depth_m=slcdp_gating_min_depth_m,
            slcdp_gating_max_depth_m=slcdp_gating_max_depth_m,
            slcdp_gating_min_opacity=slcdp_gating_min_opacity,
            slcdp_gating_chunk_size=slcdp_gating_chunk_size,
            slcdp_fast_guided_max_render_candidates=slcdp_fast_guided_max_render_candidates,
            slcdp_fast_guided_conflict_radius_px=slcdp_fast_guided_conflict_radius_px,
            slcdp_fast_guided_depth_margin_m=slcdp_fast_guided_depth_margin_m,
            slcdp_transition_control=slcdp_transition_control,
            slcdp_soft_transition_control=slcdp_soft_transition_control,
            slcdp_transition_max_reprojection_error_px=slcdp_transition_max_reprojection_error_px,
            slcdp_transition_min_retained_ratio=slcdp_transition_min_retained_ratio,
            slcdp_transition_max_translation_delta_m=slcdp_transition_max_translation_delta_m,
            slcdp_transition_max_rotation_delta_deg=slcdp_transition_max_rotation_delta_deg,
            slcdp_transition_line_search_fractions=slcdp_transition_line_search_fractions,
            apd_dense=apd_dense,
            apd_include_patch_candidates=apd_include_patch_candidates,
            apd_dense_group_weight=apd_dense_group_weight,
            apd_anchor_group_weight=apd_anchor_group_weight,
            apd_max_translation_delta_m=apd_max_translation_delta_m,
            apd_max_rotation_delta_deg=apd_max_rotation_delta_deg,
            apd_anchor_monotonic=apd_anchor_monotonic,
            apd_anchor_monotonic_epsilon_px=apd_anchor_monotonic_epsilon_px,
            apd_use_refined_pose=apd_use_refined_pose,
            apd_risk_weighted=apd_risk_weighted,
            apd_risk_max_beta=apd_risk_max_beta,
            apd_no_regression_gate=apd_no_regression_gate,
            apd_gate_min_inlier_ratio_delta=apd_gate_min_inlier_ratio_delta,
            apd_gate_max_median_reproj_increase_px=apd_gate_max_median_reproj_increase_px,
            apd_gate_max_p90_reproj_increase_px=apd_gate_max_p90_reproj_increase_px,
            apd_gate_min_inlier_count_ratio=apd_gate_min_inlier_count_ratio,
        )

    query_pil = _tensor_to_image(query_image, size=(sparse["width"], sparse["height"]))
    sparse_render_pkg = ctx["stdloc"].render_from_pose_gsplat(
        ctx["localizer"].gaussians,
        torch.tensor(sparse["pose_w2c"], device="cuda"),
        camera.FoVx,
        camera.FoVy,
        sparse["width"],
        sparse["height"],
        render_mode="RGB+ED",
        rgb_only=True,
        norm_feat_bf_render=ctx["config"]["dense"]["norm_before_render"],
        rasterize_mode="antialiased",
    )
    sparse_render = _tensor_to_image(sparse_render_pkg["render"], size=(sparse["width"], sparse["height"]))
    sparse_ref_xy, sparse_visible = project_points(
        sparse["p3d"],
        sparse["pose_w2c"],
        sparse["K"],
        width=sparse["width"],
        height=sparse["height"],
    )
    sparse_gt_xy, sparse_gt_valid = project_points(
        sparse["p3d"],
        gt_w2c,
        sparse["K"],
        width=sparse["width"],
        height=sparse["height"],
    )
    sparse_errors = np.linalg.norm(sparse_gt_xy - sparse["query_xy"], axis=1)
    sparse_errors[~(sparse_visible & sparse_gt_valid)] = np.inf
    sparse_inlier_mask = _inlier_mask(len(sparse_errors), sparse["inliers"])
    sparse_good = np.isfinite(sparse_errors) & (sparse_errors <= good_px)

    dense_render = _tensor_to_image(dense["render"], size=(sparse["width"], sparse["height"])) if dense.get("render") is not None else sparse_render
    dense_gt_xy, dense_gt_valid = project_points(
        dense["p3d"],
        gt_w2c,
        dense["K"],
        width=sparse["width"],
        height=sparse["height"],
    )
    dense_errors = np.linalg.norm(dense_gt_xy - dense["query_xy"], axis=1) if dense["query_xy"].shape[0] else np.empty(0)
    dense_errors[~dense_gt_valid] = np.inf
    dense_inlier_mask = _inlier_mask(len(dense_errors), dense["inliers"])
    dense_good = np.isfinite(dense_errors) & (dense_errors <= good_px)
    sparse_te_cm, sparse_re_deg = pose_error_cm_deg(sparse["pose_w2c"], gt_w2c)
    dense_te_cm, dense_re_deg = pose_error_cm_deg(dense["pose_w2c"], gt_w2c)
    raw_dense_pose = dense.get("raw_pose_w2c", dense["pose_w2c"])
    raw_dense_te_cm, raw_dense_re_deg = pose_error_cm_deg(raw_dense_pose, gt_w2c)
    slcdp_raw_step_acceptance = evaluate_dense_step_acceptance(
        sparse_query_xy=sparse["query_xy"],
        sparse_points_world=sparse["p3d"],
        sparse_pose_w2c=sparse["pose_w2c"],
        dense_pose_w2c=raw_dense_pose,
        intrinsic=sparse["K"],
        sparse_inlier_indices=sparse["inliers"],
        image_size=(int(sparse["width"]), int(sparse["height"])),
    )
    slcdp_step_acceptance = evaluate_dense_step_acceptance(
        sparse_query_xy=sparse["query_xy"],
        sparse_points_world=sparse["p3d"],
        sparse_pose_w2c=sparse["pose_w2c"],
        dense_pose_w2c=dense["pose_w2c"],
        intrinsic=sparse["K"],
        sparse_inlier_indices=sparse["inliers"],
        image_size=(int(sparse["width"]), int(sparse["height"])),
    )

    case_name = f"{ctx['scene']}_{int(row['query_index']):05d}_{run_label}"
    case_dir = output_dir / ctx["scene"] / case_name
    sparse_path = case_dir / "sparse_matches.jpg"
    dense_path = case_dir / "dense_matches.jpg"
    ray_depth_path = case_dir / "sparse_ray_depth.jpg"
    ray_depth_csv = case_dir / "sparse_ray_depth.csv"
    draw_match_canvas(
        query_pil,
        sparse_render,
        query_xy=sparse["query_xy"],
        reference_xy=sparse_ref_xy,
        gt_good_mask=sparse_good,
        solver_inlier_mask=sparse_inlier_mask,
        output_path=sparse_path,
        max_draw=max_draw,
    )
    draw_match_canvas(
        query_pil,
        dense_render,
        query_xy=dense["query_xy"],
        reference_xy=dense["rendered_xy"],
        gt_good_mask=dense_good,
        solver_inlier_mask=dense_inlier_mask,
        output_path=dense_path,
        max_draw=max_draw,
    )
    _write_matches_csv(case_dir / "sparse_matches.csv", sparse["query_xy"], sparse_ref_xy, sparse_errors, sparse_inlier_mask, sparse_good)
    _write_matches_csv(case_dir / "dense_matches.csv", dense["query_xy"], dense["rendered_xy"], dense_errors, dense_inlier_mask, dense_good)
    ray_depth_rows = list(dense.get("ray_depth_diagnostics") or [])
    _write_ray_depth_csv(ray_depth_csv, ray_depth_rows)
    _draw_ray_depth_canvas(query_pil, dense_render, ray_depth_rows, ray_depth_path, max_draw=max_draw)
    clean_render_paths = _write_clean_render_visualization(
        case_dir,
        query_image=query_pil,
        dense_render=dense_render,
        clean_visualization=dense.get("_clean_render_visualization"),
        max_draw=max_draw,
    )
    clean_render_visualization = dense.get("_clean_render_visualization")
    clean_render_summary = None
    if isinstance(clean_render_visualization, Mapping):
        base_clean = clean_render_visualization.get("base")
        selected_clean = clean_render_visualization.get("selected")
        clean_render_summary = {
            "selection": clean_render_visualization.get("selection"),
            "base": {
                "label": base_clean.get("label"),
                "preflight": base_clean.get("preflight"),
                "render_control": base_clean.get("render_control"),
                "sparse_ray_depth": _summarize_ray_depth(list(base_clean.get("ray_depth_diagnostics") or [])),
            }
            if isinstance(base_clean, Mapping)
            else None,
            "selected": {
                "label": selected_clean.get("label"),
                "preflight": selected_clean.get("preflight"),
                "render_control": selected_clean.get("render_control"),
                "sparse_ray_depth": _summarize_ray_depth(list(selected_clean.get("ray_depth_diagnostics") or [])),
            }
            if isinstance(selected_clean, Mapping)
            else None,
            "paths": clean_render_paths,
        }
    summary = {
        "scene": ctx["scene"],
        "run_label": run_label,
        "query_index": int(row["query_index"]),
        "image_name": row["image_name"],
        "categories": row.get("categories", ""),
        "report_candidate_dense_te_cm": float(row.get("candidate_dense_te_cm") or 0.0),
        "report_baseline_dense_te_cm": float(row.get("baseline_dense_te_cm") or 0.0),
        "captured_sparse_te_cm": sparse_te_cm,
        "captured_sparse_re_deg": sparse_re_deg,
        "captured_raw_dense_te_cm": raw_dense_te_cm,
        "captured_raw_dense_re_deg": raw_dense_re_deg,
        "captured_dense_te_cm": dense_te_cm,
        "captured_dense_re_deg": dense_re_deg,
        "sparse": summarize_match_quality(
            reprojection_errors_px=sparse_errors,
            solver_inlier_mask=sparse_inlier_mask,
            good_px_threshold=good_px,
        ),
        "dense": summarize_match_quality(
            reprojection_errors_px=dense_errors,
            solver_inlier_mask=dense_inlier_mask,
            good_px_threshold=good_px,
        ),
        "slcdp_preflight": dense.get("slcdp_preflight"),
        "slcdp_render_control": dense.get("slcdp_render_control"),
        "slcdp_repair_search": dense.get("slcdp_repair_search"),
        "slcdp_transition_control": dense.get("slcdp_transition_control"),
        "apd_dense": dense.get("apd_dense"),
        "slcdp_raw_step_acceptance": slcdp_raw_step_acceptance,
        "slcdp_step_acceptance": slcdp_step_acceptance,
        "sparse_ray_depth": _summarize_ray_depth(ray_depth_rows),
        "clean_render_visualization": clean_render_summary,
        "paths": {
            "sparse_matches": str(sparse_path),
            "dense_matches": str(dense_path),
            "sparse_csv": str(case_dir / "sparse_matches.csv"),
            "dense_csv": str(case_dir / "dense_matches.csv"),
            "sparse_ray_depth": str(ray_depth_path),
            "sparse_ray_depth_csv": str(ray_depth_csv),
            **(clean_render_paths or {}),
        },
    }
    write_json(case_dir / "match_summary.json", summary)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize sparse/dense STDLoc matches for Cambridge hard cases.")
    parser.add_argument("--report_dir", default="output/reports/cambridge_test_v6_guarded512_20260525")
    parser.add_argument("--candidate_root", default="output/stdloc_native/cambridge_test_v6_guarded512_20260525/selected")
    parser.add_argument("--baseline_root", default="")
    parser.add_argument("--output_dir", default="output/diagnostics/stdloc_match_visualizations/cambridge_test_v6_guarded512_20260526")
    parser.add_argument("--category", default="hard_failure")
    parser.add_argument("--eval_split", choices=["train", "test"], default="test")
    parser.add_argument("--max_cases", type=int, default=8)
    parser.add_argument("--good_px", type=float, default=5.0)
    parser.add_argument("--max_draw", type=int, default=350)
    parser.add_argument("--include_baseline", action="store_true")
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--slcdp_repair_search", action="store_true")
    parser.add_argument("--slcdp_repair_skip_if_no_accept", action="store_true")
    parser.add_argument("--slcdp_repair_max_candidates", type=int, default=37)
    parser.add_argument("--slcdp_repair_translation_steps_m", default="0.05,0.10,0.20")
    parser.add_argument("--slcdp_repair_rotation_steps_deg", default="0.25,0.50,1.00")
    parser.add_argument("--slcdp_repair_allow_non_accept", action="store_true")
    parser.add_argument("--slcdp_repair_min_score_gain", type=float, default=0.05)
    parser.add_argument("--slcdp_repair_translation_penalty_per_m", type=float, default=0.0)
    parser.add_argument("--slcdp_allow_low_confidence_gated_base", action="store_true")
    parser.add_argument("--slcdp_low_confidence_gated_base_min_score_gain", type=float, default=0.25)
    parser.add_argument(
        "--slcdp_render_control",
        choices=[
            "none",
            "gaussian_gating",
            "guided_pose",
            "gating_guided_pose",
            "fast_guided_pose",
            "fast_gating_guided_pose",
            SPARSE_CONDITIONED_RENDER_CONTROL,
        ],
        default="none",
    )
    parser.add_argument("--slcdp_guided_pose_steps_m", default="0.5,1.0,2.0,5.0")
    parser.add_argument("--slcdp_gating_radius_px", type=float, default=4.0)
    parser.add_argument("--slcdp_gating_depth_margin_m", type=float, default=1.0)
    parser.add_argument("--slcdp_gating_footprint_radius_scale", type=float, default=0.0)
    parser.add_argument("--slcdp_gating_max_footprint_radius_px", type=float, default=64.0)
    parser.add_argument("--slcdp_gating_min_footprint_radius_px", type=float, default=0.0)
    parser.add_argument("--slcdp_gating_min_depth_m", type=float, default=0.0)
    parser.add_argument("--slcdp_gating_max_depth_m", type=float, default=0.0)
    parser.add_argument("--slcdp_gating_min_opacity", type=float, default=0.0)
    parser.add_argument("--slcdp_gating_chunk_size", type=int, default=65536)
    parser.add_argument("--slcdp_fast_guided_max_render_candidates", type=int, default=4)
    parser.add_argument("--slcdp_fast_guided_conflict_radius_px", type=float, default=8.0)
    parser.add_argument("--slcdp_fast_guided_depth_margin_m", type=float, default=1.0)
    parser.add_argument("--slcdp_transition_control", action="store_true")
    parser.add_argument("--slcdp_soft_transition_control", action="store_true")
    parser.add_argument("--slcdp_transition_max_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--slcdp_transition_min_retained_ratio", type=float, default=0.90)
    parser.add_argument("--slcdp_transition_max_translation_delta_m", type=float, default=0.35)
    parser.add_argument("--slcdp_transition_max_rotation_delta_deg", type=float, default=5.0)
    parser.add_argument("--slcdp_transition_line_search_fractions", default="1.0,0.75,0.5,0.25,0.0")
    parser.add_argument("--apd_dense", action="store_true", help="Diagnostic APD-Dense refinement inside dense stage.")
    parser.add_argument("--apd_include_patch_candidates", action="store_true")
    parser.add_argument("--apd_dense_group_weight", type=float, default=1.0)
    parser.add_argument("--apd_anchor_group_weight", type=float, default=1.0)
    parser.add_argument("--apd_max_translation_delta_m", type=float, default=0.0)
    parser.add_argument("--apd_max_rotation_delta_deg", type=float, default=0.0)
    parser.add_argument("--apd_anchor_monotonic", action="store_true")
    parser.add_argument("--apd_anchor_monotonic_epsilon_px", type=float, default=1.0)
    parser.add_argument("--apd_use_refined_pose", action="store_true")
    parser.add_argument("--apd_risk_weighted", action="store_true")
    parser.add_argument("--apd_risk_max_beta", type=float, default=1.0)
    parser.add_argument("--apd_no_regression_gate", action="store_true", default=True)
    parser.add_argument("--no_apd_no_regression_gate", action="store_false", dest="apd_no_regression_gate")
    parser.add_argument("--apd_gate_min_inlier_ratio_delta", type=float, default=0.02)
    parser.add_argument("--apd_gate_max_median_reproj_increase_px", type=float, default=1.0)
    parser.add_argument("--apd_gate_max_p90_reproj_increase_px", type=float, default=3.0)
    parser.add_argument("--apd_gate_min_inlier_count_ratio", type=float, default=0.90)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    report_dir = Path(args.report_dir)
    output_dir = Path(args.output_dir)
    cases = _case_rows(report_dir / "hard_cases.csv", category=str(args.category), max_cases=int(args.max_cases))
    if args.scene:
        allowed = set(args.scene)
        cases = [row for row in cases if row.get("scene") in allowed]
    slcdp_options = _resolve_slcdp_effective_options(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    run_roots = [("candidate", Path(args.candidate_root))]
    if args.include_baseline and args.baseline_root:
        run_roots.append(("baseline", Path(args.baseline_root)))
    for row in cases:
        scene = str(row["scene"])
        for run_label, root in run_roots:
            key = (run_label, scene)
            if key not in contexts:
                contexts[key] = _build_context(root / scene, split_override=str(args.eval_split))
            summaries.append(
                _analyze_case(
                    contexts[key],
                    row,
                    output_dir,
                    run_label=run_label,
                    good_px=float(args.good_px),
                    max_draw=int(args.max_draw),
                    **slcdp_options,
                )
            )
    aggregate = {
        "schema": "loc_gs_stdloc_match_visualization_v1",
        "diagnostic_only": True,
        "paper_safe_for_tuning": False,
        "uses_gt_pose_for_visual_diagnostic": True,
        "report_dir": str(report_dir),
        "category": str(args.category),
        "case_count": len(summaries),
        "summaries": summaries,
    }
    write_json(output_dir / "match_summary.json", aggregate)
    manifest = {
        "method": "loc_gs_stdloc_match_visualization",
        "diagnostic_only": True,
        "paper_safe_for_tuning": False,
        "split_name": str(args.eval_split),
        "eval_split": str(args.eval_split),
        "report_dir": str(report_dir),
        "candidate_root": str(args.candidate_root),
        "baseline_root": str(args.baseline_root),
        "output_dir": str(output_dir),
        "case_count": len(summaries),
        "slcdp_requested_render_control": str(args.slcdp_render_control),
        **{
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in slcdp_options.items()
        },
    }
    split_audit = {
        "audit_status": "passed" if str(args.eval_split) == "train" else "failed",
        "reason": (
            "train/self-map diagnostic visualization"
            if str(args.eval_split) == "train"
            else "official test cases are visualized for diagnosis only; not valid for tuning or model selection"
        ),
        "split": str(args.eval_split),
        "diagnostic_only": True,
    }
    write_artifact_audit_bundle(
        output_dir,
        manifest=manifest,
        command=_command(),
        metrics_summary={"case_count": len(summaries), "diagnostic_only": True},
        split_audit=split_audit,
    )
    print(json.dumps({"output_dir": str(output_dir), "case_count": len(summaries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
