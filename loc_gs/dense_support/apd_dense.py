from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from loc_gs.diagnostics.match_visualization import project_points
from loc_gs.dense_support.sparse_anchor_residual import (
    SparseAnchorResidualPolicy,
    compute_sparse_anchor_residual_group,
)
from loc_gs.dense_support.sparse_conditioned_dense_preflight import interpolate_w2c_poses
from loc_gs.stdloc_native.patch_guided_sparse import refine_pose_with_reference_prior


@dataclass(frozen=True)
class APDDensePolicy:
    """Minimal policy for Anchor-Patch Clean Dense refinement."""

    dense_group_weight: float = 1.0
    anchor_group_weight: float = 1.0
    robust_scale_px: float = 4.0
    anchor_consistency_scale_px: float = 8.0
    min_depth_spread_m: float = 0.25
    min_spatial_coverage_cells: int = 4
    grid_rows: int = 4
    grid_cols: int = 4
    min_candidate_weight: float = 1.0e-4
    max_dense_candidates: int = 4096
    max_anchor_count: int = 256
    max_refine_iterations: int = 50
    translation_prior_weight: float = 0.05
    rotation_prior_weight: float = 0.05
    max_translation_delta_m: float | None = None
    max_rotation_delta_deg: float | None = None
    anchor_monotonic: bool = False
    anchor_monotonic_epsilon_px: float = 1.0
    anchor_monotonic_alphas: tuple[float, ...] = (0.75, 0.5, 0.25, 0.1, 0.0)


def _candidate_array(candidates: Mapping[str, Any], key: str, fallback: str | None = None) -> np.ndarray:
    value = candidates.get(key)
    if value is None and fallback is not None:
        value = candidates.get(fallback)
    if value is None:
        if key in {"query_xy", "xy"}:
            return np.empty((0, 2), dtype=np.float32)
        if key in {"p3d", "points_world"}:
            return np.empty((0, 3), dtype=np.float32)
        return np.empty((0,), dtype=np.float32)
    return np.asarray(value)


def _extract_candidate_points(candidates: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    query_xy = _candidate_array(candidates, "query_xy", fallback="xy").astype(np.float64).reshape(-1, 2)
    points = _candidate_array(candidates, "p3d", fallback="points_world").astype(np.float64).reshape(-1, 3)
    count = min(query_xy.shape[0], points.shape[0])
    return query_xy[:count], points[:count]


def _as_scores(candidates: Mapping[str, Any], count: int) -> np.ndarray:
    for key in ("weights", "match_scores", "score"):
        if candidates.get(key) is not None:
            values = np.asarray(candidates.get(key), dtype=np.float64).reshape(-1)
            if values.size:
                values = values[:count]
                if values.shape[0] < count:
                    values = np.pad(values, (0, count - values.shape[0]), constant_values=float(np.median(values)))
                return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    return np.ones((count,), dtype=np.float64)


def _normalize_01(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return array
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi - lo <= 1.0e-12:
        return np.clip(np.nan_to_num(array, nan=hi), 0.0, 1.0)
    return np.clip((np.nan_to_num(array, nan=lo) - lo) / (hi - lo), 0.0, 1.0)


def _coverage_score(query_xy: np.ndarray, image_size: tuple[int, int], policy: APDDensePolicy) -> float:
    xy = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] == 0:
        return 0.0
    width, height = int(image_size[0]), int(image_size[1])
    cols = max(1, int(policy.grid_cols))
    rows = max(1, int(policy.grid_rows))
    x_ids = np.clip(np.floor(xy[:, 0] / max(width, 1) * cols).astype(np.int64), 0, cols - 1)
    y_ids = np.clip(np.floor(xy[:, 1] / max(height, 1) * rows).astype(np.int64), 0, rows - 1)
    occupied = len(set((int(x), int(y)) for x, y in zip(x_ids, y_ids)))
    return float(np.clip(occupied / max(1, int(policy.min_spatial_coverage_cells)), 0.0, 1.0))


def _camera_depths(points: np.ndarray, pose_w2c: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    homog = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (pose @ homog.T).T[:, 2]


def _depth_geometry_score(depths: np.ndarray, policy: APDDensePolicy) -> float:
    valid = np.asarray(depths, dtype=np.float64).reshape(-1)
    valid = valid[np.isfinite(valid) & (valid > 0.0)]
    if valid.shape[0] <= 1:
        return 0.0
    spread = float(np.std(valid))
    return float(np.clip(spread / max(float(policy.min_depth_spread_m), 1.0e-6), 0.0, 1.0))


def _selected_anchor_arrays(sparse_anchors: Mapping[str, Any], max_anchor_count: int) -> tuple[np.ndarray, np.ndarray]:
    query_xy = np.asarray(sparse_anchors.get("query_xy", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    points = np.asarray(sparse_anchors.get("p3d", np.empty((0, 3))), dtype=np.float64).reshape(-1, 3)
    inliers = np.asarray(
        sparse_anchors.get("inliers", np.arange(min(query_xy.shape[0], points.shape[0]))),
        dtype=np.int64,
    ).reshape(-1)
    count = min(query_xy.shape[0], points.shape[0])
    valid = inliers[(inliers >= 0) & (inliers < count)]
    if int(max_anchor_count) > 0 and valid.shape[0] > int(max_anchor_count):
        valid = valid[: int(max_anchor_count)]
    return query_xy[valid], points[valid]


def score_dense_candidates(
    candidates: Mapping[str, Any],
    *,
    sparse_pose: np.ndarray,
    sparse_anchors: Mapping[str, Any],
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    policy: APDDensePolicy = APDDensePolicy(),
) -> dict[str, Any]:
    """Score dense correspondences using only anchor, geometry, and match signals."""

    query_xy, points = _extract_candidate_points(candidates)
    count = min(query_xy.shape[0], points.shape[0])
    if count == 0:
        empty = np.empty((0,), dtype=np.float32)
        return {
            "schema": "loc_gs_apd_dense_candidate_scores_v1",
            "uses_gt": False,
            "candidate_count": 0,
            "weights": empty,
            "anchor_consistency": empty,
            "geometry": empty,
            "match_quality": empty,
            "weight_mean": 0.0,
            "anchor_consistency_median": 0.0,
            "geometry_score": 0.0,
            "match_quality_median": 0.0,
            "core_signals": ["anchor_consistency", "geometry", "match_quality"],
        }
    query_xy = query_xy[:count]
    points = points[:count]
    projected, valid = project_points(
        points,
        np.asarray(sparse_pose, dtype=np.float64).reshape(4, 4),
        np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
        width=int(image_size[0]),
        height=int(image_size[1]),
    )
    errors = np.linalg.norm(np.asarray(projected, dtype=np.float64) - query_xy, axis=1)
    errors[~valid] = np.inf
    anchor_consistency = np.exp(-np.nan_to_num(errors, nan=1.0e6, posinf=1.0e6) / max(float(policy.anchor_consistency_scale_px), 1.0e-6))
    anchor_consistency = np.clip(anchor_consistency, 0.0, 1.0)

    coverage = _coverage_score(query_xy, image_size, policy)
    depth = _depth_geometry_score(_camera_depths(points, sparse_pose), policy)
    geometry_score = float(np.sqrt(max(coverage, 0.0) * max(depth, 0.0)))
    geometry = np.full((count,), geometry_score, dtype=np.float64)

    match_quality = _normalize_01(_as_scores(candidates, count))
    if np.max(match_quality) <= 1.0e-12 and count:
        match_quality = np.ones((count,), dtype=np.float64)
    weights = np.cbrt(np.clip(anchor_consistency, 0.0, 1.0) * np.clip(geometry, 0.0, 1.0) * np.clip(match_quality, 0.0, 1.0))
    weights = np.where(weights >= float(policy.min_candidate_weight), weights, 0.0)
    return {
        "schema": "loc_gs_apd_dense_candidate_scores_v1",
        "uses_gt": False,
        "candidate_count": int(count),
        "weights": weights.astype(np.float32),
        "anchor_consistency": anchor_consistency.astype(np.float32),
        "geometry": geometry.astype(np.float32),
        "match_quality": match_quality.astype(np.float32),
        "weight_mean": float(np.mean(weights)) if weights.size else 0.0,
        "anchor_consistency_median": float(np.median(anchor_consistency)) if anchor_consistency.size else 0.0,
        "geometry_score": float(geometry_score),
        "match_quality_median": float(np.median(match_quality)) if match_quality.size else 0.0,
        "core_signals": ["anchor_consistency", "geometry", "match_quality"],
    }


def _normalize_group_weights(raw: np.ndarray, group_weight: float) -> np.ndarray:
    weights = np.asarray(raw, dtype=np.float64).reshape(-1)
    weights = np.maximum(np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    if weights.size == 0 or float(group_weight) <= 0.0:
        return np.zeros_like(weights)
    total = float(np.sum(weights))
    if total <= 1.0e-12:
        weights = np.ones_like(weights)
        total = float(weights.size)
    return weights / max(total, 1.0e-12) * float(group_weight)


def _anchor_residual_for_pose(
    pose_w2c: np.ndarray,
    *,
    sparse_anchors: Mapping[str, Any],
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    policy: APDDensePolicy,
) -> dict[str, Any]:
    return compute_sparse_anchor_residual_group(
        sparse_anchors,
        pose_w2c=np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4),
        intrinsic=np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
        image_size=(int(image_size[0]), int(image_size[1])),
        policy=SparseAnchorResidualPolicy(
            robust_scale_px=float(policy.robust_scale_px),
            max_anchor_count=int(policy.max_anchor_count),
        ),
    )


def _anchor_metric(residual: Mapping[str, Any]) -> float | None:
    value = residual.get("median_error_px")
    if value is None:
        return None
    value_f = float(value)
    if not np.isfinite(value_f):
        return None
    return value_f


def apply_sparse_anchor_monotonic_update(
    *,
    sparse_pose: np.ndarray,
    candidate_pose: np.ndarray,
    sparse_anchors: Mapping[str, Any],
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    policy: APDDensePolicy = APDDensePolicy(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Shrink a candidate update until sparse-anchor reprojection is monotonic."""

    sparse = np.asarray(sparse_pose, dtype=np.float64).reshape(4, 4)
    candidate = np.asarray(candidate_pose, dtype=np.float64).reshape(4, 4)
    sparse_residual = _anchor_residual_for_pose(
        sparse,
        sparse_anchors=sparse_anchors,
        intrinsic=intrinsic,
        image_size=image_size,
        policy=policy,
    )
    sparse_metric = _anchor_metric(sparse_residual)
    if sparse_metric is None:
        return candidate.astype(np.float32), {
            "schema": "loc_gs_apd_anchor_monotonic_v1",
            "uses_gt": False,
            "enabled": bool(policy.anchor_monotonic),
            "success": False,
            "reason": "no_valid_sparse_anchor_residual",
            "selected_alpha": 1.0,
            "sparse_anchor_median_px": None,
            "selected_anchor_median_px": None,
            "epsilon_px": float(policy.anchor_monotonic_epsilon_px),
            "evaluated": [],
        }

    tolerance = sparse_metric + max(float(policy.anchor_monotonic_epsilon_px), 0.0)
    evaluated: list[dict[str, Any]] = []
    candidate_residual = _anchor_residual_for_pose(
        candidate,
        sparse_anchors=sparse_anchors,
        intrinsic=intrinsic,
        image_size=image_size,
        policy=policy,
    )
    candidate_metric = _anchor_metric(candidate_residual)
    if candidate_metric is not None:
        evaluated.append({"alpha": 1.0, "anchor_median_px": float(candidate_metric)})
        if candidate_metric <= tolerance:
            return candidate.astype(np.float32), {
                "schema": "loc_gs_apd_anchor_monotonic_v1",
                "uses_gt": False,
                "enabled": bool(policy.anchor_monotonic),
                "success": True,
                "reason": "candidate_anchor_monotonic",
                "selected_alpha": 1.0,
                "sparse_anchor_median_px": float(sparse_metric),
                "selected_anchor_median_px": float(candidate_metric),
                "epsilon_px": float(policy.anchor_monotonic_epsilon_px),
                "evaluated": evaluated,
            }

    alphas = [float(a) for a in policy.anchor_monotonic_alphas]
    if 0.0 not in alphas:
        alphas.append(0.0)
    for alpha in alphas:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        pose_alpha = interpolate_w2c_poses(sparse, candidate, alpha)
        residual = _anchor_residual_for_pose(
            pose_alpha,
            sparse_anchors=sparse_anchors,
            intrinsic=intrinsic,
            image_size=image_size,
            policy=policy,
        )
        metric = _anchor_metric(residual)
        evaluated.append({"alpha": alpha, "anchor_median_px": None if metric is None else float(metric)})
        if metric is not None and metric <= tolerance:
            return np.asarray(pose_alpha, dtype=np.float32).reshape(4, 4), {
                "schema": "loc_gs_apd_anchor_monotonic_v1",
                "uses_gt": False,
                "enabled": bool(policy.anchor_monotonic),
                "success": True,
                "reason": "line_search_anchor_monotonic",
                "selected_alpha": float(alpha),
                "sparse_anchor_median_px": float(sparse_metric),
                "selected_anchor_median_px": float(metric),
                "epsilon_px": float(policy.anchor_monotonic_epsilon_px),
                "evaluated": evaluated,
            }

    return sparse.astype(np.float32), {
        "schema": "loc_gs_apd_anchor_monotonic_v1",
        "uses_gt": False,
        "enabled": bool(policy.anchor_monotonic),
        "success": True,
        "reason": "fallback_sparse_pose",
        "selected_alpha": 0.0,
        "sparse_anchor_median_px": float(sparse_metric),
        "selected_anchor_median_px": float(sparse_metric),
        "epsilon_px": float(policy.anchor_monotonic_epsilon_px),
        "evaluated": evaluated,
    }


def refine_pose_with_sparse_anchors_and_dense_candidates(
    *,
    sparse_pose: np.ndarray,
    reference_pose: np.ndarray | None = None,
    sparse_anchors: Mapping[str, Any],
    dense_candidates: Mapping[str, Any],
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    policy: APDDensePolicy = APDDensePolicy(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine a sparse pose with group-normalized dense and sparse-anchor residuals."""

    dense_xy, dense_points = _extract_candidate_points(dense_candidates)
    dense_count = min(dense_xy.shape[0], dense_points.shape[0])
    dense_xy = dense_xy[:dense_count]
    dense_points = dense_points[:dense_count]
    raw_dense_weights = _as_scores(dense_candidates, dense_count)
    if int(policy.max_dense_candidates) > 0 and dense_count > int(policy.max_dense_candidates):
        order = np.argsort(raw_dense_weights, kind="mergesort")[::-1][: int(policy.max_dense_candidates)]
        dense_xy = dense_xy[order]
        dense_points = dense_points[order]
        raw_dense_weights = raw_dense_weights[order]
        dense_count = int(dense_xy.shape[0])

    anchor_xy, anchor_points = _selected_anchor_arrays(sparse_anchors, int(policy.max_anchor_count))
    anchor_count = int(anchor_xy.shape[0])
    dense_weights = _normalize_group_weights(raw_dense_weights, float(policy.dense_group_weight))
    anchor_weights = _normalize_group_weights(np.ones((anchor_count,), dtype=np.float64), float(policy.anchor_group_weight))

    match_xy = np.concatenate([dense_xy, anchor_xy], axis=0).astype(np.float64)
    points = np.concatenate([dense_points, anchor_points], axis=0).astype(np.float64)
    weights = np.concatenate([dense_weights, anchor_weights], axis=0).astype(np.float64)
    keep = weights > 0.0
    match_xy = match_xy[keep]
    points = points[keep]
    weights = weights[keep]
    if match_xy.shape[0] < 4:
        return np.asarray(sparse_pose, dtype=np.float32).reshape(4, 4), {
            "schema": "loc_gs_apd_anchor_verified_refinement_v1",
            "uses_gt": False,
            "success": False,
            "reason": "too_few_weighted_correspondences",
            "dense_candidate_count": int(dense_count),
            "anchor_residual_count": int(anchor_count),
            "weighted_correspondence_count": int(match_xy.shape[0]),
        }

    reference = np.asarray(
        sparse_pose if reference_pose is None else reference_pose,
        dtype=np.float64,
    ).reshape(4, 4)
    refined, refine_diag = refine_pose_with_reference_prior(
        reference_pose_w2c=reference,
        match_xy=match_xy,
        points_world=points,
        intrinsic=np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
        image_size=(int(image_size[0]), int(image_size[1])),
        max_iterations=int(policy.max_refine_iterations),
        reprojection_loss_scale_px=float(policy.robust_scale_px),
        translation_prior_weight=float(policy.translation_prior_weight),
        rotation_prior_weight=float(policy.rotation_prior_weight),
        max_translation_delta_m=policy.max_translation_delta_m,
        max_rotation_delta_deg=policy.max_rotation_delta_deg,
        match_weights=weights,
    )
    diagnostics = {
        "schema": "loc_gs_apd_anchor_verified_refinement_v1",
        "uses_gt": False,
        "success": bool(refine_diag.get("success", False)),
        "dense_candidate_count": int(dense_count),
        "anchor_residual_count": int(anchor_count),
        "weighted_correspondence_count": int(match_xy.shape[0]),
        "dense_weight_sum": float(np.sum(dense_weights)),
        "anchor_weight_sum": float(np.sum(anchor_weights)),
        "group_normalized": True,
        "reference_pose_source": "sparse_pose" if reference_pose is None else "dense_reference_pose",
        "refinement": refine_diag,
    }
    return refined.astype(np.float32), diagnostics


def _source_count(source: str, candidates: Mapping[str, Any] | None) -> int:
    if not isinstance(candidates, Mapping):
        return 0
    xy, points = _extract_candidate_points(candidates)
    return int(min(xy.shape[0], points.shape[0]))


def _with_scored_weights(
    source: str,
    candidates: Mapping[str, Any] | None,
    *,
    sparse_pose: np.ndarray,
    dense_reference_pose: np.ndarray | None = None,
    sparse_anchors: Mapping[str, Any],
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    policy: APDDensePolicy,
) -> dict[str, Any] | None:
    if not isinstance(candidates, Mapping):
        return None
    xy, points = _extract_candidate_points(candidates)
    count = min(xy.shape[0], points.shape[0])
    if count == 0:
        return None
    base = dict(candidates)
    base["query_xy"] = xy[:count].astype(np.float32)
    base["p3d"] = points[:count].astype(np.float32)
    scores = score_dense_candidates(
        base,
        sparse_pose=sparse_pose,
        sparse_anchors=sparse_anchors,
        intrinsic=intrinsic,
        image_size=image_size,
        policy=policy,
    )
    base["weights"] = scores["weights"]
    base["source"] = source
    base["_apd_scores"] = scores
    return base


def _merge_candidate_sources(scored_sources: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    query_parts: list[np.ndarray] = []
    point_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    score_rows: list[Mapping[str, Any]] = []
    for item in scored_sources:
        if not isinstance(item, Mapping):
            continue
        xy, points = _extract_candidate_points(item)
        count = min(xy.shape[0], points.shape[0])
        if count == 0:
            continue
        weights = np.asarray(item.get("weights", np.ones((count,), dtype=np.float32)), dtype=np.float32).reshape(-1)[:count]
        source = str(item.get("source", "dense"))
        query_parts.append(xy[:count].astype(np.float32))
        point_parts.append(points[:count].astype(np.float32))
        weight_parts.append(weights.astype(np.float32))
        source_parts.append(np.asarray([source] * count, dtype=object))
        if isinstance(item.get("_apd_scores"), Mapping):
            score_rows.append(item["_apd_scores"])
    if not query_parts:
        return {
            "query_xy": np.empty((0, 2), dtype=np.float32),
            "p3d": np.empty((0, 3), dtype=np.float32),
            "weights": np.empty((0,), dtype=np.float32),
            "sources": np.empty((0,), dtype=object),
            "_score_rows": [],
        }
    return {
        "query_xy": np.concatenate(query_parts, axis=0),
        "p3d": np.concatenate(point_parts, axis=0),
        "weights": np.concatenate(weight_parts, axis=0),
        "sources": np.concatenate(source_parts, axis=0),
        "_score_rows": score_rows,
    }


def run_anchor_patch_dense_refinement(
    *,
    sparse_pose: np.ndarray,
    dense_reference_pose: np.ndarray | None = None,
    sparse_anchors: Mapping[str, Any],
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    global_dense_candidates: Mapping[str, Any] | None = None,
    clean_render_dense_candidates: Mapping[str, Any] | None = None,
    patch_dense_candidates: Mapping[str, Any] | None = None,
    policy: APDDensePolicy = APDDensePolicy(),
) -> dict[str, Any]:
    """Run APD-Dense: global/clean/patch dense candidates plus sparse anchors."""

    scored_global = _with_scored_weights(
        "global",
        global_dense_candidates,
        sparse_pose=sparse_pose,
        sparse_anchors=sparse_anchors,
        intrinsic=intrinsic,
        image_size=image_size,
        policy=policy,
    )
    scored_clean = _with_scored_weights(
        "clean_render",
        clean_render_dense_candidates,
        sparse_pose=sparse_pose,
        sparse_anchors=sparse_anchors,
        intrinsic=intrinsic,
        image_size=image_size,
        policy=policy,
    )
    scored_patch = _with_scored_weights(
        "patch",
        patch_dense_candidates,
        sparse_pose=sparse_pose,
        sparse_anchors=sparse_anchors,
        intrinsic=intrinsic,
        image_size=image_size,
        policy=policy,
    )
    pool = _merge_candidate_sources([scored_global, scored_clean, scored_patch])
    source_counts = {
        "global": _source_count("global", scored_global),
        "clean_render": _source_count("clean_render", scored_clean),
        "patch": _source_count("patch", scored_patch),
    }
    source_counts = {key: value for key, value in sorted(source_counts.items()) if value > 0}
    final_pose, refine_diag = refine_pose_with_sparse_anchors_and_dense_candidates(
        sparse_pose=sparse_pose,
        reference_pose=dense_reference_pose,
        sparse_anchors=sparse_anchors,
        dense_candidates=pool,
        intrinsic=intrinsic,
        image_size=image_size,
        policy=policy,
    )
    anchor_monotonic_diag: dict[str, Any] | None = None
    if bool(policy.anchor_monotonic):
        final_pose, anchor_monotonic_diag = apply_sparse_anchor_monotonic_update(
            sparse_pose=sparse_pose,
            candidate_pose=final_pose,
            sparse_anchors=sparse_anchors,
            intrinsic=intrinsic,
            image_size=image_size,
            policy=policy,
        )
    anchor_xy, _anchor_points = _selected_anchor_arrays(sparse_anchors, int(policy.max_anchor_count))
    diagnostics: dict[str, Any] = {
        "candidate_pool": {
            "candidate_count": int(pool["query_xy"].shape[0]),
            "source_counts": source_counts,
            "score_summaries": pool.get("_score_rows", []),
        },
        "refinement": refine_diag,
        "core_judgements": ["anchor_consistency", "candidate_geometry", "query_render_match_quality"],
    }
    if anchor_monotonic_diag is not None:
        diagnostics["anchor_monotonic"] = anchor_monotonic_diag
    return {
        "schema": "loc_gs_apd_dense_result_v1",
        "uses_gt": False,
        "diagnostic_only": False,
        "final_pose": final_pose,
        "clean_render_pose": None,
        "num_global_candidates": int(source_counts.get("global", 0)),
        "num_clean_render_candidates": int(source_counts.get("clean_render", 0)),
        "num_patch_candidates": int(source_counts.get("patch", 0)),
        "num_anchor_residuals": int(anchor_xy.shape[0]),
        "dense_refine_success": bool(refine_diag.get("success", False)),
        "diagnostics": diagnostics,
    }
