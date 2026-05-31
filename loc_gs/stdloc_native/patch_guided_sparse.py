from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Patch:
    patch_id: int
    x0: int
    y0: int
    x1: int
    y1: int
    image_width: int
    image_height: int

    @property
    def width(self) -> int:
        return int(self.x1 - self.x0)

    @property
    def height(self) -> int:
        return int(self.y1 - self.y0)

    @property
    def area(self) -> int:
        return int(max(0, self.width) * max(0, self.height))


@dataclass
class PatchHypothesis:
    patch_id: int
    score: float
    inlier_count: int
    camera_center: np.ndarray | None = None
    rotation: np.ndarray | None = None
    match_indices: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    components: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = float(self.score)
        self.inlier_count = int(self.inlier_count)
        if self.camera_center is None:
            self.camera_center = np.zeros((3,), dtype=np.float64)
        else:
            self.camera_center = np.asarray(self.camera_center, dtype=np.float64).reshape(3)
        if self.rotation is None:
            self.rotation = np.eye(3, dtype=np.float64)
        else:
            self.rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        self.match_indices = np.asarray(self.match_indices, dtype=np.int64).reshape(-1)


@dataclass(frozen=True)
class PoseCluster:
    patch_ids: list[int]
    hypothesis_indices: list[int]
    score: float
    inlier_count: int
    match_indices: np.ndarray


@dataclass(frozen=True)
class PatchResidualWeightPolicy:
    min_group_patches: int = 2
    local_score_scale: float = 2.0
    spatial_coverage_scale: float = 0.25
    min_patch_residual_weight: float = 0.05
    max_patch_residual_weight: float = 1.0


def _axis_starts(length: int, patch: int, overlap: int) -> list[int]:
    if length <= 0 or patch <= 0:
        raise ValueError("image and patch dimensions must be positive")
    if overlap < 0 or overlap >= patch:
        raise ValueError("overlap must satisfy 0 <= overlap < patch dimension")
    if patch >= length:
        return [0]
    stride = patch - overlap
    starts = list(range(0, max(1, length - patch + 1), stride))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def generate_patch_grid(
    *,
    image_size: tuple[int, int],
    patch_size: tuple[int, int],
    overlap: int | tuple[int, int] = 0,
) -> list[Patch]:
    """Generate deterministic overlapping image patches in row-major order."""

    width, height = (int(image_size[0]), int(image_size[1]))
    patch_w, patch_h = (int(patch_size[0]), int(patch_size[1]))
    if isinstance(overlap, tuple):
        overlap_w, overlap_h = (int(overlap[0]), int(overlap[1]))
    else:
        overlap_w = overlap_h = int(overlap)
    xs = _axis_starts(width, patch_w, overlap_w)
    ys = _axis_starts(height, patch_h, overlap_h)
    patches: list[Patch] = []
    patch_id = 0
    for y0 in ys:
        for x0 in xs:
            patches.append(
                Patch(
                    patch_id=patch_id,
                    x0=x0,
                    y0=y0,
                    x1=min(width, x0 + patch_w),
                    y1=min(height, y0 + patch_h),
                    image_width=width,
                    image_height=height,
                )
            )
            patch_id += 1
    return patches


def filter_matches_by_patch(
    matches: Mapping[str, Any],
    patch: Patch,
    *,
    xy_key: str = "xy",
) -> dict[str, Any]:
    """Return matches whose query coordinates fall inside a patch."""

    xy = np.asarray(matches[xy_key], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"{xy_key} must have shape (N, 2+) for patch filtering")
    mask = (xy[:, 0] >= patch.x0) & (xy[:, 0] < patch.x1) & (xy[:, 1] >= patch.y0) & (xy[:, 1] < patch.y1)
    indices = np.flatnonzero(mask).astype(np.int64)
    filtered: dict[str, Any] = {}
    for key, value in matches.items():
        array = np.asarray(value)
        filtered[key] = array[indices] if array.shape[:1] == (xy.shape[0],) else value
    filtered["indices"] = indices
    filtered["patch_id"] = int(patch.patch_id)
    return filtered


def _robust_unit_interval(value: float, scale: float) -> float:
    value = max(0.0, float(value))
    scale = max(1.0e-12, float(scale))
    return float(value / (value + scale))


def _logdet_proxy(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    centered = points[:, :2] - np.mean(points[:, :2], axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(1, points.shape[0] - 1)
    sign, logdet = np.linalg.slogdet(cov + 1.0e-6 * np.eye(2))
    if sign <= 0:
        return 0.0
    return _robust_unit_interval(float(logdet), 4.0)


def _rotation_delta_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = np.asarray(rotation_a, dtype=np.float64).reshape(3, 3) @ np.asarray(rotation_b, dtype=np.float64).reshape(3, 3).T
    trace = float(np.trace(relative))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _camera_center_rotation_from_w2c(pose_w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    rotation = pose[:3, :3]
    center = -rotation.T @ pose[:3, 3]
    return center.astype(np.float64), rotation.astype(np.float64)


def _project_world_points(
    *,
    points_world: np.ndarray,
    pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    intr = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    width, height = int(image_size[0]), int(image_size[1])
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=bool)

    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    camera = (pose @ homog.T).T[:, :3]
    z = camera[:, 2]
    positive_depth = z > 1.0e-9
    safe_z = np.where(positive_depth, z, 1.0)
    normalized = np.column_stack([camera[:, 0] / safe_z, camera[:, 1] / safe_z, np.ones_like(safe_z)])
    pixels_h = (intr @ normalized.T).T
    pixels = pixels_h[:, :2]
    finite = np.all(np.isfinite(pixels), axis=1)
    in_bounds = (pixels[:, 0] >= 0.0) & (pixels[:, 0] < float(width)) & (pixels[:, 1] >= 0.0) & (pixels[:, 1] < float(height))
    valid = positive_depth & finite & in_bounds
    return pixels.astype(np.float64), valid.astype(bool)


def evaluate_pose_global_consistency(
    *,
    pose_w2c: Any,
    intrinsic: Any,
    image_size: tuple[int, int],
    sparse_query_xy: Any | None = None,
    sparse_points_world: Any | None = None,
    sparse_inlier_indices: Any | None = None,
    sparse_retention_max_error_px: float = 8.0,
    sparse_retention_min_ratio: float = 0.70,
    reference_pose_w2c: Any | None = None,
    reference_max_translation_delta_m: float | None = None,
    reference_max_rotation_delta_deg: float | None = None,
) -> dict[str, Any]:
    """Check whether a patch pose is globally compatible without using GT.

    The check has two independent parts:
    - sparse inlier retention: the candidate pose must still reproject reliable
      sparse-stage 2D-3D inliers back to their query keypoints.
    - reference pose trust: dense-stage patch hypotheses can be bounded around
      the current dense pose to avoid replacing a globally reasonable pose with
      a locally self-consistent patch solution.
    """

    failed_checks = {
        "sparse_inlier_retention": False,
        "reference_translation_delta": False,
        "reference_rotation_delta": False,
    }
    result: dict[str, Any] = {
        "schema": "loc_gs_pgsh_global_consistency_v1",
        "decision": "accept_patch_pose",
        "sparse_inlier_count": 0,
        "sparse_retained_count": 0,
        "sparse_retained_ratio": None,
        "sparse_median_reprojection_error_px": None,
        "sparse_retention_max_error_px": float(sparse_retention_max_error_px),
        "sparse_retention_min_ratio": float(sparse_retention_min_ratio),
        "reference_translation_delta_m": None,
        "reference_rotation_delta_deg": None,
        "reference_max_translation_delta_m": None if reference_max_translation_delta_m is None else float(reference_max_translation_delta_m),
        "reference_max_rotation_delta_deg": None if reference_max_rotation_delta_deg is None else float(reference_max_rotation_delta_deg),
        "failed_checks": failed_checks,
        "diagnostic_only": True,
    }

    if sparse_query_xy is not None and sparse_points_world is not None and sparse_inlier_indices is not None:
        query_xy = np.asarray(sparse_query_xy, dtype=np.float64).reshape(-1, 2)
        points = np.asarray(sparse_points_world, dtype=np.float64).reshape(-1, 3)
        inlier_indices = np.asarray(sparse_inlier_indices, dtype=np.int64).reshape(-1)
        valid_indices = inlier_indices[(inlier_indices >= 0) & (inlier_indices < min(query_xy.shape[0], points.shape[0]))]
        result["sparse_inlier_count"] = int(valid_indices.shape[0])
        if valid_indices.shape[0] > 0:
            selected_xy = query_xy[valid_indices]
            selected_points = points[valid_indices]
            projected, visible = _project_world_points(
                points_world=selected_points,
                pose_w2c=np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4),
                intrinsic=np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
                image_size=image_size,
            )
            reproj_errors = np.linalg.norm(projected - selected_xy, axis=1)
            retained = visible & np.isfinite(reproj_errors) & (reproj_errors <= float(sparse_retention_max_error_px))
            retained_count = int(np.count_nonzero(retained))
            retained_ratio = float(retained_count / max(1, int(valid_indices.shape[0])))
            finite_errors = reproj_errors[np.isfinite(reproj_errors)]
            result["sparse_retained_count"] = retained_count
            result["sparse_retained_ratio"] = retained_ratio
            result["sparse_median_reprojection_error_px"] = float(np.median(finite_errors)) if finite_errors.size else None
            failed_checks["sparse_inlier_retention"] = bool(retained_ratio < float(sparse_retention_min_ratio))

    if reference_pose_w2c is not None:
        candidate_center, candidate_rotation = _camera_center_rotation_from_w2c(np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4))
        reference_center, reference_rotation = _camera_center_rotation_from_w2c(np.asarray(reference_pose_w2c, dtype=np.float64).reshape(4, 4))
        translation_delta = float(np.linalg.norm(candidate_center - reference_center))
        rotation_delta = _rotation_delta_deg(candidate_rotation, reference_rotation)
        result["reference_translation_delta_m"] = translation_delta
        result["reference_rotation_delta_deg"] = rotation_delta
        if reference_max_translation_delta_m is not None:
            failed_checks["reference_translation_delta"] = bool(translation_delta > float(reference_max_translation_delta_m))
        if reference_max_rotation_delta_deg is not None:
            failed_checks["reference_rotation_delta"] = bool(rotation_delta > float(reference_max_rotation_delta_deg))

    if any(failed_checks.values()):
        result["decision"] = "reject_patch_pose"
    return result


def filter_matches_by_reference_reprojection(
    matches: Mapping[str, Any],
    *,
    pose_w2c: Any,
    intrinsic: Any,
    image_size: tuple[int, int],
    max_reprojection_error_px: float,
    xy_key: str = "xy",
    xyz_key: str = "xyz",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep only patch matches that are geometrically plausible under a reference pose."""

    xy = np.asarray(matches[xy_key], dtype=np.float64).reshape(-1, 2)
    points = np.asarray(matches[xyz_key], dtype=np.float64).reshape(-1, 3)
    if xy.shape[0] != points.shape[0]:
        raise ValueError(f"{xy_key} and {xyz_key} must contain the same number of matches")
    projected, valid = _project_world_points(
        points_world=points,
        pose_w2c=np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4),
        intrinsic=np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
        image_size=image_size,
    )
    errors = np.linalg.norm(projected - xy, axis=1)
    keep = valid & np.isfinite(errors) & (errors <= float(max_reprojection_error_px))
    indices = np.flatnonzero(keep).astype(np.int64)
    filtered: dict[str, Any] = {}
    n_matches = int(xy.shape[0])
    for key, value in matches.items():
        array = np.asarray(value)
        filtered[key] = array[indices] if array.shape[:1] == (n_matches,) else value
    filtered["reference_reprojection_error_px"] = errors[indices].astype(np.float32)
    filtered["indices"] = indices
    finite_errors = errors[np.isfinite(errors)]
    diagnostics = {
        "schema": "loc_gs_pgsh_reference_reprojection_filter_v1",
        "enabled": True,
        "before_count": n_matches,
        "after_count": int(indices.shape[0]),
        "kept_ratio": float(indices.shape[0] / max(1, n_matches)),
        "max_reprojection_error_px": float(max_reprojection_error_px),
        "median_reprojection_error_px": float(np.median(finite_errors)) if finite_errors.size else None,
        "p90_reprojection_error_px": float(np.percentile(finite_errors, 90.0)) if finite_errors.size else None,
    }
    return filtered, diagnostics


def _subset_match_mapping(matches: Mapping[str, Any], indices: np.ndarray, *, n_matches: int) -> dict[str, Any]:
    filtered: dict[str, Any] = {}
    for key, value in matches.items():
        array = np.asarray(value)
        filtered[key] = array[indices] if array.shape[:1] == (n_matches,) else value
    filtered["indices"] = indices.astype(np.int64)
    return filtered


def filter_matches_by_quality(
    matches: Mapping[str, Any],
    *,
    min_margin: float = 0.0,
    best_per_landmark: bool = False,
    max_matches: int = 0,
    min_matches: int = 0,
    score_key: str = "score",
    margin_key: str = "margin",
    landmark_key: str = "gs_ids",
    xy_key: str = "xy",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Filter ambiguous sparse matches before patch-level PnP.

    The filter is conservative by default: if the requested filters would leave
    fewer than ``min_matches`` correspondences, it returns the original matches
    and records a fallback diagnostic.
    """

    xy = np.asarray(matches[xy_key])
    n_matches = int(xy.shape[0])
    keep = np.ones((n_matches,), dtype=bool)
    diagnostics: dict[str, Any] = {
        "schema": "loc_gs_sparse_match_quality_filter_v1",
        "enabled": bool(float(min_margin) > 0.0 or best_per_landmark or int(max_matches) > 0),
        "decision": "not_filtered",
        "input_count": n_matches,
        "min_margin": float(min_margin),
        "best_per_landmark": bool(best_per_landmark),
        "max_matches": int(max_matches),
        "min_matches": int(min_matches),
        "margin_available": bool(margin_key in matches),
        "margin_kept_count": n_matches,
        "unique_landmark_kept_count": n_matches,
        "top_score_kept_count": n_matches,
    }
    if n_matches == 0 or not diagnostics["enabled"]:
        return _subset_match_mapping(matches, np.arange(n_matches, dtype=np.int64), n_matches=n_matches), diagnostics

    if float(min_margin) > 0.0 and margin_key in matches:
        margin = np.asarray(matches[margin_key], dtype=np.float64).reshape(-1)
        if margin.shape[0] == n_matches:
            keep &= np.isfinite(margin) & (margin >= float(min_margin))
            diagnostics["margin_kept_count"] = int(np.count_nonzero(keep))

    if bool(best_per_landmark) and landmark_key in matches:
        ids = np.asarray(matches[landmark_key]).reshape(-1)
        scores = np.asarray(matches.get(score_key, np.ones((n_matches,), dtype=np.float32)), dtype=np.float64).reshape(-1)
        if ids.shape[0] == n_matches and scores.shape[0] == n_matches:
            selected = np.zeros((n_matches,), dtype=bool)
            for landmark_id in np.unique(ids[keep]):
                candidates = np.flatnonzero(keep & (ids == landmark_id))
                if candidates.size == 0:
                    continue
                best = candidates[np.lexsort((candidates, -scores[candidates]))[0]]
                selected[best] = True
            keep &= selected
            diagnostics["unique_landmark_kept_count"] = int(np.count_nonzero(keep))

    indices = np.flatnonzero(keep).astype(np.int64)
    if int(max_matches) > 0 and indices.shape[0] > int(max_matches):
        scores = np.asarray(matches.get(score_key, np.ones((n_matches,), dtype=np.float32)), dtype=np.float64).reshape(-1)
        if scores.shape[0] == n_matches:
            order = np.lexsort((indices, -scores[indices]))
            indices = indices[order[: int(max_matches)]]
            indices = np.sort(indices).astype(np.int64)
            diagnostics["top_score_kept_count"] = int(indices.shape[0])

    if int(min_matches) > 0 and indices.shape[0] < int(min_matches):
        diagnostics["decision"] = "fallback_original_too_few_matches"
        diagnostics["output_count"] = n_matches
        original = np.arange(n_matches, dtype=np.int64)
        return _subset_match_mapping(matches, original, n_matches=n_matches), diagnostics

    diagnostics["decision"] = "filtered" if indices.shape[0] != n_matches else "not_filtered"
    diagnostics["output_count"] = int(indices.shape[0])
    return _subset_match_mapping(matches, indices, n_matches=n_matches), diagnostics


def _pose_reprojection_errors(
    *,
    pose_w2c: np.ndarray,
    match_xy: np.ndarray,
    points_world: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    projected, valid = _project_world_points(
        points_world=points_world,
        pose_w2c=pose_w2c,
        intrinsic=intrinsic,
        image_size=image_size,
    )
    errors = np.linalg.norm(projected - np.asarray(match_xy, dtype=np.float64).reshape(-1, 2), axis=1)
    errors[~valid] = np.inf
    return errors.astype(np.float64)


def refine_pose_with_reference_prior(
    *,
    reference_pose_w2c: Any,
    match_xy: Any,
    points_world: Any,
    intrinsic: Any,
    image_size: tuple[int, int],
    max_iterations: int = 50,
    reprojection_loss_scale_px: float = 4.0,
    translation_prior_weight: float = 0.05,
    rotation_prior_weight: float = 0.05,
    max_translation_delta_m: float | None = None,
    max_rotation_delta_deg: float | None = None,
    match_weights: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Locally refine a pose around a reference using robust reprojection residuals.

    This is intended for PGSH diagnostics where unconstrained PnP can jump to a
    locally self-consistent but globally implausible patch pose.
    """

    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    reference = np.asarray(reference_pose_w2c, dtype=np.float64).reshape(4, 4)
    xy = np.asarray(match_xy, dtype=np.float64).reshape(-1, 2)
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    intr = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    if xy.shape[0] != points.shape[0]:
        raise ValueError("match_xy and points_world must contain the same number of matches")
    if match_weights is None:
        weights = np.ones((xy.shape[0],), dtype=np.float64)
    else:
        weights = np.asarray(match_weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != xy.shape[0]:
            raise ValueError("match_weights must contain one weight per match")
        weights = np.maximum(weights, 0.0)
    if xy.shape[0] < 4:
        return reference.astype(np.float32), {
            "schema": "loc_gs_pgsh_reference_prior_refinement_v1",
            "success": False,
            "reason": "too_few_matches",
            "match_count": int(xy.shape[0]),
            "diagnostic_only": True,
        }

    before_errors = _pose_reprojection_errors(
        pose_w2c=reference,
        match_xy=xy,
        points_world=points,
        intrinsic=intr,
        image_size=image_size,
    )
    loss_scale = max(1.0e-6, float(reprojection_loss_scale_px))
    reference_rotation = reference[:3, :3]
    reference_center = -reference_rotation.T @ reference[:3, 3]

    def pose_from_delta(delta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta, dtype=np.float64).reshape(6)
        pose = np.array(reference, copy=True)
        rotation = Rotation.from_rotvec(delta[:3]).as_matrix() @ reference_rotation
        center = reference_center + delta[3:6]
        pose[:3, :3] = rotation
        pose[:3, 3] = -rotation @ center
        return pose

    def residual(delta: np.ndarray) -> np.ndarray:
        pose = pose_from_delta(delta)
        projected, valid = _project_world_points(
            points_world=points,
            pose_w2c=pose,
            intrinsic=intr,
            image_size=image_size,
        )
        reproj = (projected - xy).reshape(-1) / loss_scale
        valid_pairs = np.repeat(valid, 2)
        reproj[~valid_pairs] = 1.0e3
        reproj = reproj * np.repeat(np.sqrt(weights), 2)
        prior = np.concatenate(
            [
                np.asarray(delta[:3], dtype=np.float64) * float(rotation_prior_weight),
                np.asarray(delta[3:6], dtype=np.float64) * float(translation_prior_weight),
            ]
        )
        return np.concatenate([reproj, prior], axis=0)

    lower = np.full((6,), -np.inf, dtype=np.float64)
    upper = np.full((6,), np.inf, dtype=np.float64)
    if max_rotation_delta_deg is not None and np.isfinite(float(max_rotation_delta_deg)):
        rot_bound = float(np.radians(max(0.0, float(max_rotation_delta_deg))))
        opt_rot_bound = max(rot_bound, 1.0e-12)
        lower[:3] = -opt_rot_bound
        upper[:3] = opt_rot_bound
    if max_translation_delta_m is not None and np.isfinite(float(max_translation_delta_m)):
        trans_bound = float(max(0.0, float(max_translation_delta_m)))
        opt_trans_bound = max(trans_bound, 1.0e-12)
        lower[3:6] = -opt_trans_bound
        upper[3:6] = opt_trans_bound

    result = least_squares(
        residual,
        np.zeros((6,), dtype=np.float64),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=max(1, int(max_iterations)),
    )
    refined_delta = np.asarray(result.x, dtype=np.float64).reshape(6).copy()
    if max_rotation_delta_deg is not None and np.isfinite(float(max_rotation_delta_deg)):
        rot_limit = float(np.radians(max(0.0, float(max_rotation_delta_deg))))
        rot_norm = float(np.linalg.norm(refined_delta[:3]))
        if rot_limit <= 0.0:
            refined_delta[:3] = 0.0
        elif rot_norm > rot_limit:
            refined_delta[:3] *= rot_limit / max(rot_norm, 1.0e-12)
    if max_translation_delta_m is not None and np.isfinite(float(max_translation_delta_m)):
        trans_limit = float(max(0.0, float(max_translation_delta_m)))
        trans_norm = float(np.linalg.norm(refined_delta[3:6]))
        if trans_limit <= 0.0:
            refined_delta[3:6] = 0.0
        elif trans_norm > trans_limit:
            refined_delta[3:6] *= trans_limit / max(trans_norm, 1.0e-12)

    refined = pose_from_delta(refined_delta)
    after_errors = _pose_reprojection_errors(
        pose_w2c=refined,
        match_xy=xy,
        points_world=points,
        intrinsic=intr,
        image_size=image_size,
    )
    before_finite = before_errors[np.isfinite(before_errors)]
    after_finite = after_errors[np.isfinite(after_errors)]
    diagnostics = {
        "schema": "loc_gs_pgsh_reference_prior_refinement_v1",
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "match_count": int(xy.shape[0]),
        "weighted_match_count": float(np.sum(weights)),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "delta_rotation_norm_deg": float(np.degrees(np.linalg.norm(refined_delta[:3]))),
        "delta_translation_norm_m": float(np.linalg.norm(refined_delta[3:6])),
        "max_translation_delta_m": None if max_translation_delta_m is None else float(max_translation_delta_m),
        "max_rotation_delta_deg": None if max_rotation_delta_deg is None else float(max_rotation_delta_deg),
        "before_median_reprojection_error_px": float(np.median(before_finite)) if before_finite.size else None,
        "after_median_reprojection_error_px": float(np.median(after_finite)) if after_finite.size else None,
        "before_p90_reprojection_error_px": float(np.percentile(before_finite, 90.0)) if before_finite.size else None,
        "after_p90_reprojection_error_px": float(np.percentile(after_finite, 90.0)) if after_finite.size else None,
        "diagnostic_only": True,
    }
    return refined.astype(np.float32), diagnostics


def score_patch_hypothesis(
    patch: Patch,
    *,
    match_xy: Any,
    inlier_mask: Any,
    reproj_errors: Any,
    depths: Any | None = None,
    bearings_2d: Any | None = None,
    camera_center: Any | None = None,
    rotation: Any | None = None,
    match_indices: Any | None = None,
    detector_scores: Any | None = None,
    detector_score_weight: float = 0.0,
    score_profile: str = "balanced",
) -> PatchHypothesis:
    """Score a sparse pose hypothesis for one patch without GT or external solvers."""

    xy = np.asarray(match_xy, dtype=np.float64).reshape(-1, 2)
    inliers = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    reproj = np.asarray(reproj_errors, dtype=np.float64).reshape(-1)
    if xy.shape[0] != inliers.shape[0] or xy.shape[0] != reproj.shape[0]:
        raise ValueError("match_xy, inlier_mask, and reproj_errors must have the same length")
    n_matches = int(xy.shape[0])
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = float(inlier_count / n_matches) if n_matches else 0.0
    inlier_count_score = _robust_unit_interval(inlier_count, 8.0)
    median_reproj = float(np.median(reproj[inliers])) if inlier_count else float(np.median(reproj)) if n_matches else np.inf
    reproj_score = 0.0 if not np.isfinite(median_reproj) else float(1.0 / (1.0 + max(0.0, median_reproj)))

    if n_matches:
        span_x = float(np.max(xy[:, 0]) - np.min(xy[:, 0]))
        span_y = float(np.max(xy[:, 1]) - np.min(xy[:, 1]))
    else:
        span_x = span_y = 0.0
    patch_fill = _robust_unit_interval((span_x * span_y) / max(1.0, float(patch.area)), 0.25)
    image_fill = _robust_unit_interval(float(patch.area) / max(1.0, float(patch.image_width * patch.image_height)), 0.10)
    spatial_extent = float(patch_fill * image_fill)

    if depths is None:
        depth_spread = 0.5 if inlier_count >= 4 else 0.0
    else:
        depth_values = np.asarray(depths, dtype=np.float64).reshape(-1)
        if depth_values.shape[0] != n_matches:
            raise ValueError("depths must have the same length as match_xy")
        active_depths = depth_values[inliers] if inlier_count else depth_values
        depth_spread = _robust_unit_interval(float(np.ptp(active_depths)) if active_depths.size else 0.0, 1.0)

    bearing_points = np.asarray(bearings_2d if bearings_2d is not None else xy, dtype=np.float64).reshape(-1, 2)
    if bearing_points.shape[0] != n_matches:
        raise ValueError("bearings_2d must have the same length as match_xy")
    logdet = _logdet_proxy(bearing_points[inliers] if inlier_count >= 2 else bearing_points)
    detector_support = 0.5
    if detector_scores is not None:
        detector_values = np.asarray(detector_scores, dtype=np.float64).reshape(-1)
        if detector_values.shape[0] != n_matches:
            raise ValueError("detector_scores must have the same length as match_xy")
        active_detector = detector_values[inliers] if inlier_count else detector_values
        if active_detector.size:
            detector_support = float(np.clip(np.median(active_detector), 0.0, 1.0))
    ambiguity_risk = float(np.clip(1.0 - 0.45 * spatial_extent - 0.35 * depth_spread - 0.20 * logdet, 0.0, 1.0))
    profile = str(score_profile or "balanced")
    if profile == "precision":
        score = (
            0.45 * inlier_ratio
            + 0.05 * inlier_count_score
            + 0.15 * logdet
            + 0.25 * reproj_score
            + 0.05 * spatial_extent
            + 0.10 * depth_spread
            + float(detector_score_weight) * (detector_support - 0.5)
            - 0.10 * ambiguity_risk
        )
    elif profile == "balanced":
        score = (
            0.25 * inlier_ratio
            + 0.20 * inlier_count_score
            + 0.15 * logdet
            + 0.15 * reproj_score
            + 0.15 * spatial_extent
            + 0.15 * depth_spread
            + float(detector_score_weight) * (detector_support - 0.5)
            - 0.20 * ambiguity_risk
        )
    else:
        raise ValueError("score_profile must be 'balanced' or 'precision'")
    components = {
        "inlier_ratio": inlier_ratio,
        "inlier_count": float(inlier_count),
        "logdet_proxy": logdet,
        "reproj_median": median_reproj,
        "reproj_score": reproj_score,
        "spatial_extent": spatial_extent,
        "depth_spread": depth_spread,
        "detector_support": detector_support,
        "ambiguity_risk": ambiguity_risk,
    }
    if match_indices is None:
        match_indices = np.arange(n_matches, dtype=np.int64)
    return PatchHypothesis(
        patch_id=patch.patch_id,
        score=score,
        inlier_count=inlier_count,
        camera_center=camera_center,
        rotation=rotation,
        match_indices=np.asarray(match_indices, dtype=np.int64),
        components=components,
    )


def _rank_hypothesis_indices(indices: Sequence[int], hypotheses: Sequence[PatchHypothesis], *, rank_by: str) -> list[int]:
    rank = str(rank_by or "inlier_count")
    if rank == "inlier_count":
        return sorted(indices, key=lambda idx: (-hypotheses[idx].inlier_count, -hypotheses[idx].score, hypotheses[idx].patch_id))
    if rank == "score":
        return sorted(indices, key=lambda idx: (-hypotheses[idx].score, -hypotheses[idx].inlier_count, hypotheses[idx].patch_id))
    raise ValueError("rank_by must be 'inlier_count' or 'score'")


def cluster_patch_hypotheses(
    hypotheses: Sequence[PatchHypothesis],
    *,
    center_thresh: float,
    rotation_thresh_deg: float,
    rank_by: str = "inlier_count",
    max_patch_count: int = 0,
) -> list[PoseCluster]:
    """Cluster patch hypotheses by camera-center distance and rotation delta."""

    clusters: list[list[int]] = []
    for hyp_idx, hyp in enumerate(hypotheses):
        placed = False
        for cluster in clusters:
            if all(
                np.linalg.norm(hyp.camera_center - hypotheses[other].camera_center) <= float(center_thresh)
                and _rotation_delta_deg(hyp.rotation, hypotheses[other].rotation) <= float(rotation_thresh_deg)
                for other in cluster
            ):
                cluster.append(hyp_idx)
                placed = True
                break
        if not placed:
            clusters.append([hyp_idx])
    capped_clusters: list[list[int]] = []
    for cluster in clusters:
        ranked = _rank_hypothesis_indices(cluster, hypotheses, rank_by=rank_by)
        if int(max_patch_count) > 0:
            ranked = ranked[: int(max_patch_count)]
        capped_clusters.append(ranked)
    pose_clusters = [_build_pose_cluster(cluster, hypotheses) for cluster in capped_clusters]
    rank = str(rank_by or "inlier_count")
    if rank == "inlier_count":
        pose_clusters.sort(key=lambda item: (-item.inlier_count, -item.score, item.patch_ids))
    elif rank == "score":
        pose_clusters.sort(key=lambda item: (-item.score, -item.inlier_count, item.patch_ids))
    else:
        raise ValueError("rank_by must be 'inlier_count' or 'score'")
    return pose_clusters


def _build_pose_cluster(indices: list[int], hypotheses: Sequence[PatchHypothesis]) -> PoseCluster:
    selected = [hypotheses[idx] for idx in indices]
    patch_ids = [int(hyp.patch_id) for hyp in selected]
    score = float(sum(hyp.score for hyp in selected))
    inlier_count = int(sum(hyp.inlier_count for hyp in selected))
    if selected:
        match_indices = np.unique(np.concatenate([hyp.match_indices for hyp in selected])).astype(np.int64)
    else:
        match_indices = np.empty((0,), dtype=np.int64)
    return PoseCluster(
        patch_ids=patch_ids,
        hypothesis_indices=list(indices),
        score=score,
        inlier_count=inlier_count,
        match_indices=match_indices,
    )


def select_pose_consistent_patch_group(
    hypotheses: Sequence[PatchHypothesis],
    *,
    center_thresh: float,
    rotation_thresh_deg: float,
    min_patch_count: int = 1,
    rank_by: str = "inlier_count",
    max_patch_count: int = 0,
) -> PoseCluster:
    clusters = cluster_patch_hypotheses(
        hypotheses,
        center_thresh=center_thresh,
        rotation_thresh_deg=rotation_thresh_deg,
        rank_by=rank_by,
        max_patch_count=int(max_patch_count),
    )
    min_count = max(1, int(min_patch_count))
    clusters = [cluster for cluster in clusters if len(cluster.patch_ids) >= min_count]
    if not clusters:
        return PoseCluster([], [], 0.0, 0, np.empty((0,), dtype=np.int64))
    return clusters[0]


def merge_group_matches(matches: Mapping[str, Any], group: PoseCluster, *, xy_key: str = "xy") -> dict[str, Any]:
    """Merge all sparse matches referenced by a pose-consistent patch group."""

    indices = np.asarray(group.match_indices, dtype=np.int64).reshape(-1)
    n_matches = int(np.asarray(matches[xy_key]).shape[0]) if xy_key in matches else None
    merged: dict[str, Any] = {}
    for key, value in matches.items():
        array = np.asarray(value)
        if n_matches is not None and array.ndim > 0 and array.shape[0] == n_matches:
            merged[key] = array[indices]
        else:
            merged[key] = value
    merged["indices"] = indices
    merged["patch_ids"] = list(group.patch_ids)
    return merged


def compute_patch_residual_group_weight(
    group: PoseCluster,
    hypotheses: Sequence[PatchHypothesis],
    *,
    total_patch_count: int | None = None,
    policy: PatchResidualWeightPolicy = PatchResidualWeightPolicy(),
) -> dict[str, Any]:
    """Compute a diagnostic residual weight for a pose-consistent patch group.

    Patch proposals are local and can be wrong in repeated structures. This
    weight keeps them as residual evidence: multi-patch, low-ambiguity,
    spatially covered groups get high weight; local-only ambiguous groups are
    downweighted before anchor-conditioned local refinement.
    """

    hyp_by_index = [hypotheses[idx] for idx in group.hypothesis_indices if 0 <= int(idx) < len(hypotheses)]
    if not hyp_by_index:
        return {
            "schema": "loc_gs_patch_residual_group_weight_v1",
            "patch_residual_weight": 0.0,
            "components": {
                "local_quality": 0.0,
                "pose_consensus": 0.0,
                "spatial_coverage": 0.0,
                "ambiguity_safety": 0.0,
                "geometry_support": 0.0,
            },
            "diagnostic_only": True,
        }
    patch_count = int(len(set(int(hyp.patch_id) for hyp in hyp_by_index)))
    local_quality = _robust_unit_interval(max(0.0, float(group.score)), float(policy.local_score_scale))
    pose_consensus = float(np.clip(patch_count / max(1, int(policy.min_group_patches)), 0.0, 1.0))
    if total_patch_count is not None and int(total_patch_count) > 0:
        coverage_raw = patch_count / float(total_patch_count)
    else:
        coverage_raw = patch_count / max(1.0, float(policy.min_group_patches))
    spatial_coverage = _robust_unit_interval(coverage_raw, float(policy.spatial_coverage_scale))
    ambiguity_values = [float(hyp.components.get("ambiguity_risk", 0.5)) for hyp in hyp_by_index]
    ambiguity_safety = float(np.clip(1.0 - float(np.mean(ambiguity_values)), 0.0, 1.0))
    geometry_terms: list[float] = []
    for hyp in hyp_by_index:
        components = hyp.components
        for key in ("spatial_extent", "depth_spread", "logdet_proxy"):
            if key in components and np.isfinite(float(components[key])):
                geometry_terms.append(float(np.clip(float(components[key]), 0.0, 1.0)))
    geometry_support = float(np.mean(geometry_terms)) if geometry_terms else 0.5
    raw = local_quality * pose_consensus * (0.5 + 0.5 * spatial_coverage) * ambiguity_safety * (0.5 + 0.5 * geometry_support)
    weight = float(
        np.clip(
            raw,
            float(policy.min_patch_residual_weight),
            float(policy.max_patch_residual_weight),
        )
    )
    return {
        "schema": "loc_gs_patch_residual_group_weight_v1",
        "patch_residual_weight": weight,
        "components": {
            "local_quality": float(local_quality),
            "pose_consensus": float(pose_consensus),
            "spatial_coverage": float(spatial_coverage),
            "ambiguity_safety": float(ambiguity_safety),
            "geometry_support": float(geometry_support),
        },
        "patch_count": patch_count,
        "total_patch_count": None if total_patch_count is None else int(total_patch_count),
        "diagnostic_only": True,
    }


def apply_patch_residual_group_weight(
    refinement_matches: Mapping[str, Any],
    *,
    patch_residual_weight: float,
    dense_match_count: int,
) -> dict[str, Any]:
    """Scale dense patch residual weights while leaving sparse anchors intact."""

    result = {key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value for key, value in refinement_matches.items()}
    xy = np.asarray(result.get("xy", np.empty((0, 2))), dtype=np.float32).reshape(-1, 2)
    weights = np.asarray(result.get("weights", np.ones((xy.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1).copy()
    dense_count = int(np.clip(int(dense_match_count), 0, weights.shape[0]))
    weight = float(np.clip(float(patch_residual_weight), 0.0, 1.0))
    if dense_count > 0:
        weights[:dense_count] *= weight
    result["weights"] = weights.astype(np.float32)
    result["patch_residual_weighting"] = {
        "schema": "loc_gs_patch_residual_weighting_applied_v1",
        "patch_residual_weight": weight,
        "dense_match_count": dense_count,
        "sparse_anchor_count": int(max(0, weights.shape[0] - dense_count)),
        "diagnostic_only": True,
    }
    return result
