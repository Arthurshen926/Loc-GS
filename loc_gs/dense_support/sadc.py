from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from loc_gs.diagnostics.match_visualization import project_points


@dataclass(frozen=True)
class SADCCorrespondencePolicy:
    """Policy for Sparse-Anchored Dense Correspondence filtering."""

    mode: str = "topk"
    anchor_flow_scale_px: float = 8.0
    min_depth_spread_m: float = 0.25
    grid_rows: int = 4
    grid_cols: int = 4
    geometry_floor: float = 0.25
    min_candidate_weight: float = 1.0e-6
    max_candidates: int = 4096
    min_keep_count: int = 4
    native_drop_percentile: float = 95.0
    min_native_keep_ratio: float = 0.9
    min_native_conflict_score: float = 0.75
    patch_add_percentile: float = 90.0
    max_patch_fraction: float = 0.15
    min_patch_anchor_consistency: float = 0.75
    anchor_monotonic: bool = False
    anchor_monotonic_epsilon_px: float = 1.0
    anchor_monotonic_alphas: tuple[float, ...] = (0.75, 0.5, 0.25, 0.1, 0.0)
    activation_mode: str = "always"
    activation_min_risk: float = 0.5
    activation_min_sparse_confidence: float = 0.0


def _array(value: Any, *, shape_tail: tuple[int, ...] | None = None) -> np.ndarray:
    out = np.asarray(value)
    if shape_tail is not None:
        out = out.reshape(-1, *shape_tail)
    return out


def _candidate_xy(candidates: Mapping[str, Any], key: str, fallback: str | None = None) -> np.ndarray:
    value = candidates.get(key)
    if value is None and fallback is not None:
        value = candidates.get(fallback)
    if value is None:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(value, dtype=np.float64).reshape(-1, 2)


def _candidate_xyz(candidates: Mapping[str, Any]) -> np.ndarray:
    value = candidates.get("p3d")
    if value is None:
        value = candidates.get("points_world")
    if value is None:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(value, dtype=np.float64).reshape(-1, 3)


def _project_render_xy(
    points: np.ndarray,
    *,
    pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    xy, _valid = project_points(
        points,
        np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4),
        np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
        width=int(image_size[0]),
        height=int(image_size[1]),
    )
    return np.asarray(xy, dtype=np.float64).reshape(-1, 2)


def _extract_candidates(
    candidates: Mapping[str, Any],
    *,
    sparse_pose: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_xy = _candidate_xy(candidates, "query_xy", fallback="xy")
    points = _candidate_xyz(candidates)
    count = min(query_xy.shape[0], points.shape[0])
    query_xy = query_xy[:count]
    points = points[:count]
    render_xy = _candidate_xy(candidates, "render_xy")
    if render_xy.shape[0] < count:
        render_xy = _project_render_xy(
            points,
            pose_w2c=sparse_pose,
            intrinsic=intrinsic,
            image_size=image_size,
        )
    return query_xy, render_xy[:count], points


def _selected_anchor_indices(sparse_anchors: Mapping[str, Any], count: int) -> np.ndarray:
    inliers = sparse_anchors.get("inliers")
    if inliers is None:
        return np.arange(count, dtype=np.int64)
    idx = np.asarray(inliers, dtype=np.int64).reshape(-1)
    return idx[(idx >= 0) & (idx < count)]


def _anchor_arrays(
    sparse_anchors: Mapping[str, Any],
    *,
    sparse_pose: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    query_xy = _candidate_xy(sparse_anchors, "query_xy", fallback="xy")
    points = _candidate_xyz(sparse_anchors)
    count = min(query_xy.shape[0], points.shape[0])
    idx = _selected_anchor_indices(sparse_anchors, count)
    if idx.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    query_xy = query_xy[:count][idx]
    render_xy = _candidate_xy(sparse_anchors, "render_xy")
    if render_xy.shape[0] < count:
        render_xy = _project_render_xy(
            points[:count],
            pose_w2c=sparse_pose,
            intrinsic=intrinsic,
            image_size=image_size,
        )
    return query_xy, render_xy[:count][idx]


def _match_quality(candidates: Mapping[str, Any], count: int) -> np.ndarray:
    for key in ("match_scores", "score", "weights"):
        if candidates.get(key) is None:
            continue
        values = np.asarray(candidates.get(key), dtype=np.float64).reshape(-1)[:count]
        if values.shape[0] < count:
            fill = float(np.nanmedian(values)) if values.size else 0.0
            values = np.pad(values, (0, count - values.shape[0]), constant_values=fill)
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.ones((count,), dtype=np.float64)
        if float(np.min(finite)) >= 0.0 and float(np.max(finite)) <= 1.0:
            return np.clip(values, 0.0, 1.0)
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi - lo <= 1.0e-12:
            return np.ones((count,), dtype=np.float64)
        return np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return np.ones((count,), dtype=np.float64)


def _anchor_flow_consistency(
    *,
    query_xy: np.ndarray,
    render_xy: np.ndarray,
    anchor_query_xy: np.ndarray,
    anchor_render_xy: np.ndarray,
    scale_px: float,
) -> np.ndarray:
    count = int(query_xy.shape[0])
    if count == 0:
        return np.empty((0,), dtype=np.float64)
    if anchor_query_xy.shape[0] == 0 or anchor_render_xy.shape[0] == 0:
        return np.full((count,), 0.5, dtype=np.float64)
    anchor_flow = anchor_query_xy - anchor_render_xy
    dense_flow = query_xy - render_xy
    distances = np.linalg.norm(render_xy[:, None, :] - anchor_render_xy[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    flow_error = np.linalg.norm(dense_flow - anchor_flow[nearest], axis=1)
    return np.exp(-(flow_error**2) / max(float(scale_px) ** 2, 1.0e-12))


def _coverage_score(query_xy: np.ndarray, image_size: tuple[int, int], policy: SADCCorrespondencePolicy) -> float:
    if query_xy.shape[0] == 0:
        return 0.0
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    cols, rows = max(1, int(policy.grid_cols)), max(1, int(policy.grid_rows))
    x_ids = np.clip(np.floor(query_xy[:, 0] / width * cols).astype(np.int64), 0, cols - 1)
    y_ids = np.clip(np.floor(query_xy[:, 1] / height * rows).astype(np.int64), 0, rows - 1)
    occupied = len(set((int(x), int(y)) for x, y in zip(x_ids, y_ids)))
    return float(np.clip(occupied / float(rows * cols), 0.0, 1.0))


def _depth_spread_score(points: np.ndarray, sparse_pose: np.ndarray, policy: SADCCorrespondencePolicy) -> float:
    if points.shape[0] <= 1:
        return 0.0
    pose = np.asarray(sparse_pose, dtype=np.float64).reshape(4, 4)
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    depths = (pose @ homog.T).T[:, 2]
    valid = depths[np.isfinite(depths) & (depths > 0.0)]
    if valid.shape[0] <= 1:
        return 0.0
    return float(np.clip(np.std(valid) / max(float(policy.min_depth_spread_m), 1.0e-6), 0.0, 1.0))


def _geometry(query_xy: np.ndarray, points: np.ndarray, sparse_pose: np.ndarray, image_size: tuple[int, int], policy: SADCCorrespondencePolicy) -> float:
    coverage = _coverage_score(query_xy, image_size, policy)
    depth = _depth_spread_score(points, sparse_pose, policy)
    score = float(np.sqrt(max(coverage, 0.0) * max(depth, 0.0)))
    return float(np.clip(max(score, float(policy.geometry_floor)), 0.0, 1.0))


def score_sadc_correspondences(
    candidates: Mapping[str, Any],
    *,
    sparse_pose: np.ndarray,
    sparse_anchors: Mapping[str, Any],
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    policy: SADCCorrespondencePolicy = SADCCorrespondencePolicy(),
) -> dict[str, Any]:
    """Score dense correspondences by match, sparse-anchor flow, and geometry."""

    query_xy, render_xy, points = _extract_candidates(
        candidates,
        sparse_pose=sparse_pose,
        intrinsic=intrinsic,
        image_size=image_size,
    )
    count = int(min(query_xy.shape[0], render_xy.shape[0], points.shape[0]))
    query_xy = query_xy[:count]
    render_xy = render_xy[:count]
    points = points[:count]
    if count == 0:
        empty = np.empty((0,), dtype=np.float32)
        return {
            "schema": "loc_gs_sadc_correspondence_scores_v1",
            "uses_gt": False,
            "candidate_count": 0,
            "weights": empty,
            "match_quality": empty,
            "anchor_flow_consistency": empty,
            "geometry": empty,
            "geometry_score": 0.0,
            "core_signals": ["match_quality", "anchor_flow_consistency", "geometry"],
        }
    anchor_query_xy, anchor_render_xy = _anchor_arrays(
        sparse_anchors,
        sparse_pose=sparse_pose,
        intrinsic=intrinsic,
        image_size=image_size,
    )
    match = _match_quality(candidates, count)
    anchor = _anchor_flow_consistency(
        query_xy=query_xy,
        render_xy=render_xy,
        anchor_query_xy=anchor_query_xy,
        anchor_render_xy=anchor_render_xy,
        scale_px=float(policy.anchor_flow_scale_px),
    )
    geometry_score = _geometry(query_xy, points, sparse_pose, image_size, policy)
    geometry = np.full((count,), geometry_score, dtype=np.float64)
    weights = np.cbrt(
        np.clip(match, 0.0, 1.0)
        * np.clip(anchor, 0.0, 1.0)
        * np.clip(geometry, 0.0, 1.0)
    )
    weights = np.where(weights >= float(policy.min_candidate_weight), weights, 0.0)
    return {
        "schema": "loc_gs_sadc_correspondence_scores_v1",
        "uses_gt": False,
        "candidate_count": int(count),
        "weights": weights.astype(np.float32),
        "match_quality": match.astype(np.float32),
        "anchor_flow_consistency": anchor.astype(np.float32),
        "geometry": geometry.astype(np.float32),
        "geometry_score": float(geometry_score),
        "weight_mean": float(np.mean(weights)) if weights.size else 0.0,
        "weight_median": float(np.median(weights)) if weights.size else 0.0,
        "anchor_flow_median": float(np.median(anchor)) if anchor.size else 0.0,
        "match_quality_median": float(np.median(match)) if match.size else 0.0,
        "core_signals": ["match_quality", "anchor_flow_consistency", "geometry"],
    }


def _candidate_sources(candidates: Mapping[str, Any], count: int) -> np.ndarray:
    values = candidates.get("sources")
    if values is None:
        source = str(candidates.get("source", "dense"))
        return np.asarray([source] * count, dtype=object)
    out = np.asarray(values, dtype=object).reshape(-1)[:count]
    if out.shape[0] < count:
        out = np.concatenate([out, np.asarray(["dense"] * (count - out.shape[0]), dtype=object)], axis=0)
    return out


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def decide_sadc_activation(
    dense_damage_risk: Mapping[str, Any] | None,
    policy: SADCCorrespondencePolicy = SADCCorrespondencePolicy(),
) -> dict[str, Any]:
    """Decide whether SADC correspondence intervention is active.

    The decision consumes observable sparse/dense diagnostics only. It is a
    diagnostic native-default activation for SADC candidate construction, not a
    GT-based branch selector.
    """

    mode = str(policy.activation_mode or "always")
    report = dense_damage_risk if isinstance(dense_damage_risk, Mapping) else {}
    risk = _as_float(report.get("risk", 0.0), 0.0)
    components = report.get("components")
    if not isinstance(components, Mapping):
        components = {}
    sparse_confidence = _as_float(components.get("sparse_confidence", 0.0), 0.0)
    if mode == "always":
        active = True
        reason = "always_active"
    elif mode == "dense_damage_risk":
        if risk < float(policy.activation_min_risk):
            active = False
            reason = "risk_below_threshold"
        elif sparse_confidence < float(policy.activation_min_sparse_confidence):
            active = False
            reason = "sparse_confidence_below_threshold"
        else:
            active = True
            reason = "active_dense_damage_risk"
    else:
        raise ValueError(f"Unknown SADC activation mode: {mode}")
    return {
        "schema": "loc_gs_sadc_activation_v1",
        "uses_gt": False,
        "diagnostic_native_default": mode != "always",
        "active": bool(active),
        "mode": mode,
        "reason": reason,
        "risk": float(risk),
        "activation_min_risk": float(policy.activation_min_risk),
        "sparse_confidence": float(sparse_confidence),
        "activation_min_sparse_confidence": float(policy.activation_min_sparse_confidence),
        "dense_damage_risk": dict(report),
    }


def filter_sadc_correspondences(
    candidates: Mapping[str, Any],
    *,
    sparse_pose: np.ndarray,
    sparse_anchors: Mapping[str, Any],
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    policy: SADCCorrespondencePolicy = SADCCorrespondencePolicy(),
) -> dict[str, Any]:
    """Filter/re-rank dense correspondences without selecting a pose."""

    query_xy, render_xy, points = _extract_candidates(
        candidates,
        sparse_pose=sparse_pose,
        intrinsic=intrinsic,
        image_size=image_size,
    )
    count = int(min(query_xy.shape[0], render_xy.shape[0], points.shape[0]))
    query_xy = query_xy[:count]
    render_xy = render_xy[:count]
    points = points[:count]
    scores = score_sadc_correspondences(
        {
            **dict(candidates),
            "query_xy": query_xy,
            "render_xy": render_xy,
            "p3d": points,
        },
        sparse_pose=sparse_pose,
        sparse_anchors=sparse_anchors,
        intrinsic=intrinsic,
        image_size=image_size,
        policy=policy,
    )
    weights = np.asarray(scores["weights"], dtype=np.float64).reshape(-1)
    if count == 0:
        keep = np.empty((0,), dtype=np.int64)
    else:
        order = np.argsort(weights, kind="mergesort")[::-1]
        positive = order[weights[order] >= float(policy.min_candidate_weight)]
        max_candidates = int(policy.max_candidates)
        if max_candidates > 0:
            positive = positive[:max_candidates]
        min_keep = min(int(policy.min_keep_count), count)
        if max_candidates > 0:
            min_keep = min(min_keep, max_candidates)
        if positive.shape[0] < min_keep:
            needed = min_keep
            positive = order[:needed]
        keep = positive.astype(np.int64)
    match_scores = _match_quality(candidates, count).astype(np.float32)
    sources = _candidate_sources(candidates, count)
    return {
        "schema": "loc_gs_sadc_filtered_correspondences_v1",
        "uses_gt": False,
        "does_not_select_final_pose": True,
        "query_xy": query_xy[keep].astype(np.float32),
        "render_xy": render_xy[keep].astype(np.float32),
        "p3d": points[keep].astype(np.float32),
        "weights": weights[keep].astype(np.float32),
        "match_scores": match_scores[keep].astype(np.float32),
        "sources": sources[keep],
        "kept_indices": keep.astype(np.int32),
        "candidate_count": int(count),
        "kept_count": int(keep.shape[0]),
        "dropped_count": int(count - keep.shape[0]),
        "scores": scores,
        "policy": {
            "mode": str(policy.mode),
            "anchor_flow_scale_px": float(policy.anchor_flow_scale_px),
            "min_depth_spread_m": float(policy.min_depth_spread_m),
            "grid_rows": int(policy.grid_rows),
            "grid_cols": int(policy.grid_cols),
            "geometry_floor": float(policy.geometry_floor),
            "min_candidate_weight": float(policy.min_candidate_weight),
            "max_candidates": int(policy.max_candidates),
            "min_keep_count": int(policy.min_keep_count),
            "native_drop_percentile": float(policy.native_drop_percentile),
            "min_native_keep_ratio": float(policy.min_native_keep_ratio),
            "min_native_conflict_score": float(policy.min_native_conflict_score),
            "patch_add_percentile": float(policy.patch_add_percentile),
            "max_patch_fraction": float(policy.max_patch_fraction),
            "min_patch_anchor_consistency": float(policy.min_patch_anchor_consistency),
            "anchor_monotonic": bool(policy.anchor_monotonic),
            "anchor_monotonic_epsilon_px": float(policy.anchor_monotonic_epsilon_px),
            "anchor_monotonic_alphas": [float(v) for v in policy.anchor_monotonic_alphas],
            "activation_mode": str(policy.activation_mode),
            "activation_min_risk": float(policy.activation_min_risk),
            "activation_min_sparse_confidence": float(policy.activation_min_sparse_confidence),
        },
    }


def _policy_dict(policy: SADCCorrespondencePolicy) -> dict[str, Any]:
    return {
        "mode": str(policy.mode),
        "anchor_flow_scale_px": float(policy.anchor_flow_scale_px),
        "min_depth_spread_m": float(policy.min_depth_spread_m),
        "grid_rows": int(policy.grid_rows),
        "grid_cols": int(policy.grid_cols),
        "geometry_floor": float(policy.geometry_floor),
        "min_candidate_weight": float(policy.min_candidate_weight),
        "max_candidates": int(policy.max_candidates),
        "min_keep_count": int(policy.min_keep_count),
        "native_drop_percentile": float(policy.native_drop_percentile),
        "min_native_keep_ratio": float(policy.min_native_keep_ratio),
        "min_native_conflict_score": float(policy.min_native_conflict_score),
        "patch_add_percentile": float(policy.patch_add_percentile),
        "max_patch_fraction": float(policy.max_patch_fraction),
        "min_patch_anchor_consistency": float(policy.min_patch_anchor_consistency),
        "anchor_monotonic": bool(policy.anchor_monotonic),
        "anchor_monotonic_epsilon_px": float(policy.anchor_monotonic_epsilon_px),
        "anchor_monotonic_alphas": [float(v) for v in policy.anchor_monotonic_alphas],
        "activation_mode": str(policy.activation_mode),
        "activation_min_risk": float(policy.activation_min_risk),
        "activation_min_sparse_confidence": float(policy.activation_min_sparse_confidence),
    }


def _result_from_keep(
    *,
    mode: str,
    query_xy: np.ndarray,
    render_xy: np.ndarray,
    points: np.ndarray,
    match_scores: np.ndarray,
    sources: np.ndarray,
    weights: np.ndarray,
    keep: np.ndarray,
    scores: Mapping[str, Any],
    policy: SADCCorrespondencePolicy,
    conflict_scores: np.ndarray,
    native_mask: np.ndarray,
    patch_mask: np.ndarray,
) -> dict[str, Any]:
    keep = np.asarray(keep, dtype=np.int64).reshape(-1)
    keep = keep[(keep >= 0) & (keep < query_xy.shape[0])]
    kept_native = native_mask[keep] if keep.size else np.empty((0,), dtype=bool)
    kept_patch = patch_mask[keep] if keep.size else np.empty((0,), dtype=bool)
    return {
        "schema": "loc_gs_sadc_sanitized_correspondences_v1",
        "uses_gt": False,
        "does_not_select_final_pose": True,
        "mode": str(mode),
        "query_xy": query_xy[keep].astype(np.float32),
        "render_xy": render_xy[keep].astype(np.float32),
        "p3d": points[keep].astype(np.float32),
        "weights": weights[keep].astype(np.float32),
        "match_scores": match_scores[keep].astype(np.float32),
        "sources": sources[keep],
        "kept_indices": keep.astype(np.int32),
        "candidate_count": int(query_xy.shape[0]),
        "kept_count": int(keep.shape[0]),
        "dropped_count": int(query_xy.shape[0] - keep.shape[0]),
        "native_candidate_count": int(np.sum(native_mask)),
        "native_kept_count": int(np.sum(kept_native)),
        "patch_candidate_count": int(np.sum(patch_mask)),
        "patch_kept_count": int(np.sum(kept_patch)),
        "conflict_score_mean": float(np.mean(conflict_scores)) if conflict_scores.size else 0.0,
        "conflict_score_p95": float(np.percentile(conflict_scores, 95.0)) if conflict_scores.size else 0.0,
        "scores": scores,
        "policy": _policy_dict(policy),
    }


def _native_patch_masks(sources: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([str(item) for item in sources.tolist()], dtype=object)
    patch_mask = labels == "patch"
    native_mask = ~patch_mask
    return native_mask, patch_mask


def sanitize_sadc_correspondences(
    candidates: Mapping[str, Any],
    *,
    sparse_pose: np.ndarray,
    sparse_anchors: Mapping[str, Any],
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    policy: SADCCorrespondencePolicy = SADCCorrespondencePolicy(mode="conflict_filter_patch_append"),
) -> dict[str, Any]:
    """Conservative native-preserving SADC sanitizer.

    Unlike legacy top-k SADC, this mode preserves native dense correspondences by
    default, removes only extreme sparse-anchor conflicts, and admits patch
    correspondences only as high-confidence extras.
    """

    mode = str(policy.mode)
    if mode == "topk":
        out = filter_sadc_correspondences(
            candidates,
            sparse_pose=sparse_pose,
            sparse_anchors=sparse_anchors,
            intrinsic=intrinsic,
            image_size=image_size,
            policy=policy,
        )
        out["mode"] = "topk"
        return out

    query_xy, render_xy, points = _extract_candidates(
        candidates,
        sparse_pose=sparse_pose,
        intrinsic=intrinsic,
        image_size=image_size,
    )
    count = int(min(query_xy.shape[0], render_xy.shape[0], points.shape[0]))
    query_xy = query_xy[:count]
    render_xy = render_xy[:count]
    points = points[:count]
    scores = score_sadc_correspondences(
        {
            **dict(candidates),
            "query_xy": query_xy,
            "render_xy": render_xy,
            "p3d": points,
        },
        sparse_pose=sparse_pose,
        sparse_anchors=sparse_anchors,
        intrinsic=intrinsic,
        image_size=image_size,
        policy=policy,
    )
    weights = np.asarray(scores.get("weights", np.empty((0,), dtype=np.float32)), dtype=np.float64).reshape(-1)[:count]
    anchor = np.asarray(scores.get("anchor_flow_consistency", np.empty((0,), dtype=np.float32)), dtype=np.float64).reshape(-1)[:count]
    match = np.asarray(scores.get("match_quality", np.empty((0,), dtype=np.float32)), dtype=np.float64).reshape(-1)[:count]
    if anchor.shape[0] < count:
        anchor = np.pad(anchor, (0, count - anchor.shape[0]), constant_values=0.5)
    if match.shape[0] < count:
        match = np.pad(match, (0, count - match.shape[0]), constant_values=1.0)
    if weights.shape[0] < count:
        weights = np.pad(weights, (0, count - weights.shape[0]), constant_values=0.0)
    match_scores = _match_quality(candidates, count).astype(np.float32)
    sources = _candidate_sources(candidates, count)
    native_mask, patch_mask = _native_patch_masks(sources)
    conflict_scores = np.clip(0.75 * (1.0 - anchor) + 0.25 * (1.0 - match), 0.0, 1.0)
    if count == 0:
        keep = np.empty((0,), dtype=np.int64)
    elif mode in {"passthrough", "score_only"}:
        keep = np.arange(count, dtype=np.int64)
    elif mode in {"conflict_filter", "patch_append", "conflict_filter_patch_append"}:
        native_idx = np.flatnonzero(native_mask)
        patch_idx = np.flatnonzero(patch_mask)
        native_keep = native_idx.copy()
        if mode in {"conflict_filter", "conflict_filter_patch_append"} and native_idx.size:
            native_conflict = conflict_scores[native_idx]
            threshold = float(np.percentile(native_conflict, np.clip(float(policy.native_drop_percentile), 0.0, 100.0)))
            drop_mask = (native_conflict > threshold) & (native_conflict >= float(policy.min_native_conflict_score))
            native_keep = native_idx[~drop_mask]
            min_native = int(np.ceil(native_idx.size * np.clip(float(policy.min_native_keep_ratio), 0.0, 1.0)))
            min_native = min(native_idx.size, max(0, min_native))
            if native_keep.size < min_native:
                order = np.argsort(native_conflict, kind="mergesort")
                native_keep = native_idx[order[:min_native]]
                native_keep = np.sort(native_keep, kind="mergesort")
        patch_keep = np.empty((0,), dtype=np.int64)
        if mode in {"patch_append", "conflict_filter_patch_append"} and patch_idx.size:
            reference = weights[native_keep] if native_keep.size else weights[native_idx]
            if reference.size == 0:
                reference = weights
            quality_threshold = float(np.percentile(reference, np.clip(float(policy.patch_add_percentile), 0.0, 100.0))) if reference.size else 1.0
            patch_ok = (weights[patch_idx] >= quality_threshold) & (
                anchor[patch_idx] >= float(policy.min_patch_anchor_consistency)
            )
            patch_candidates = patch_idx[patch_ok]
            if patch_candidates.size:
                max_patch = int(np.floor(max(1, native_keep.size) * max(0.0, float(policy.max_patch_fraction))))
                max_patch = max(0, max_patch)
                if max_patch > 0:
                    order = np.argsort(weights[patch_candidates], kind="mergesort")[::-1]
                    patch_keep = np.sort(patch_candidates[order[:max_patch]], kind="mergesort")
        keep = np.concatenate([native_keep, patch_keep], axis=0)
        keep = np.sort(keep, kind="mergesort")
    else:
        raise ValueError(f"Unknown SADC mode: {mode}")
    return _result_from_keep(
        mode=mode,
        query_xy=query_xy,
        render_xy=render_xy,
        points=points,
        match_scores=match_scores,
        sources=sources,
        weights=weights,
        keep=keep,
        scores=scores,
        policy=policy,
        conflict_scores=conflict_scores,
        native_mask=native_mask,
        patch_mask=patch_mask,
    )


def merge_sadc_candidate_sources(
    sources: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    """Merge named dense correspondence sources into one candidate pool."""

    query_parts: list[np.ndarray] = []
    render_parts: list[np.ndarray] = []
    point_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    for name, candidates in sources.items():
        if not isinstance(candidates, Mapping):
            continue
        query_xy = _candidate_xy(candidates, "query_xy", fallback="xy")
        render_xy = _candidate_xy(candidates, "render_xy")
        points = _candidate_xyz(candidates)
        count = min(query_xy.shape[0], render_xy.shape[0], points.shape[0])
        if count == 0:
            continue
        query_parts.append(query_xy[:count].astype(np.float32))
        render_parts.append(render_xy[:count].astype(np.float32))
        point_parts.append(points[:count].astype(np.float32))
        score_parts.append(_match_quality(candidates, count).astype(np.float32))
        source_parts.append(np.asarray([str(name)] * count, dtype=object))
    if not query_parts:
        return {
            "query_xy": np.empty((0, 2), dtype=np.float32),
            "render_xy": np.empty((0, 2), dtype=np.float32),
            "p3d": np.empty((0, 3), dtype=np.float32),
            "match_scores": np.empty((0,), dtype=np.float32),
            "sources": np.empty((0,), dtype=object),
        }
    return {
        "query_xy": np.concatenate(query_parts, axis=0),
        "render_xy": np.concatenate(render_parts, axis=0),
        "p3d": np.concatenate(point_parts, axis=0),
        "match_scores": np.concatenate(score_parts, axis=0),
        "sources": np.concatenate(source_parts, axis=0),
    }


def apply_sadc_anchor_monotonic_update(
    *,
    sparse_pose: np.ndarray,
    candidate_pose: np.ndarray,
    sparse_anchors: Mapping[str, Any],
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    policy: SADCCorrespondencePolicy = SADCCorrespondencePolicy(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Shrink a SADC dense update until sparse-anchor residual is monotonic.

    This uses only sparse PnP anchors and camera geometry. It is intended as a
    deterministic trust-region step for a single dense refinement path, not as
    GT-based branch selection.
    """

    from loc_gs.dense_support.apd_dense import APDDensePolicy, apply_sparse_anchor_monotonic_update

    selected, diag = apply_sparse_anchor_monotonic_update(
        sparse_pose=sparse_pose,
        candidate_pose=candidate_pose,
        sparse_anchors=sparse_anchors,
        intrinsic=intrinsic,
        image_size=image_size,
        policy=APDDensePolicy(
            anchor_monotonic=bool(policy.anchor_monotonic),
            anchor_monotonic_epsilon_px=float(policy.anchor_monotonic_epsilon_px),
            anchor_monotonic_alphas=tuple(float(v) for v in policy.anchor_monotonic_alphas),
        ),
    )
    diag = dict(diag)
    diag["schema"] = "loc_gs_sadc_anchor_monotonic_v1"
    diag["policy"] = {
        "anchor_monotonic": bool(policy.anchor_monotonic),
        "anchor_monotonic_epsilon_px": float(policy.anchor_monotonic_epsilon_px),
        "anchor_monotonic_alphas": [float(v) for v in policy.anchor_monotonic_alphas],
    }
    return selected, diag
