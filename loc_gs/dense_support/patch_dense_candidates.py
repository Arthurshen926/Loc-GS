from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PatchDenseCandidatePolicy:
    grid_rows: int = 4
    grid_cols: int = 4
    sample_stride: int = 4
    top_patch_count: int = 8
    max_matches_per_patch: int = 256
    ambiguity_margin_target: float = 0.10
    offset_consistency_scale_px: float = 4.0
    min_depth_spread_m: float = 0.25


def _normalize_columns(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    denom = np.linalg.norm(array, axis=0, keepdims=True)
    return np.divide(array, denom, out=np.zeros_like(array), where=denom > 1.0e-12)


def _feature_map(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [C, H, W]")
    return array


def _patch_bounds(height: int, width: int, rows: int, cols: int):
    for row in range(int(rows)):
        y0 = int(np.floor(row * height / max(1, int(rows))))
        y1 = int(np.floor((row + 1) * height / max(1, int(rows))))
        for col in range(int(cols)):
            x0 = int(np.floor(col * width / max(1, int(cols))))
            x1 = int(np.floor((col + 1) * width / max(1, int(cols))))
            yield row * int(cols) + col, x0, x1, y0, y1


def _coords_for_patch(x0: int, x1: int, y0: int, y1: int, stride: int) -> np.ndarray:
    xs = np.arange(x0, x1, max(1, int(stride)), dtype=np.int64)
    ys = np.arange(y0, y1, max(1, int(stride)), dtype=np.int64)
    if xs.size == 0 or ys.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)


def _anchor_offsets(sparse_anchors: Mapping[str, Any] | None) -> np.ndarray:
    if not isinstance(sparse_anchors, Mapping):
        return np.empty((0, 2), dtype=np.float64)
    query = np.asarray(sparse_anchors.get("query_xy", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    render = np.asarray(sparse_anchors.get("render_xy", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    count = min(query.shape[0], render.shape[0])
    if count <= 0:
        return np.empty((0, 2), dtype=np.float64)
    return render[:count] - query[:count]


def _offset_consistency(offsets: np.ndarray, reference_offsets: np.ndarray, scale_px: float) -> float:
    if offsets.size == 0 or reference_offsets.size == 0:
        return 0.5
    local = np.median(np.asarray(offsets, dtype=np.float64).reshape(-1, 2), axis=0)
    reference = np.median(np.asarray(reference_offsets, dtype=np.float64).reshape(-1, 2), axis=0)
    distance = float(np.linalg.norm(local - reference))
    return float(np.exp(-distance / max(float(scale_px), 1.0e-6)))


def _cluster_consistency(offsets: np.ndarray, scale_px: float) -> float:
    if offsets.shape[0] <= 1:
        return 1.0 if offsets.shape[0] == 1 else 0.0
    spread = float(np.median(np.linalg.norm(offsets - np.median(offsets, axis=0, keepdims=True), axis=1)))
    return float(np.exp(-spread / max(float(scale_px), 1.0e-6)))


def _depth_spread_score(depth_values: np.ndarray, target: float) -> float:
    valid = np.asarray(depth_values, dtype=np.float64)
    valid = valid[np.isfinite(valid) & (valid > 0.0)]
    if valid.size <= 1:
        return 0.0
    return float(np.clip(np.std(valid) / max(float(target), 1.0e-6), 0.0, 1.0))


def _lift_render_points_to_world(
    render_xy: np.ndarray,
    depth_values: np.ndarray,
    *,
    render_pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    xy = np.asarray(render_xy, dtype=np.float64).reshape(-1, 2)
    depth = np.asarray(depth_values, dtype=np.float64).reshape(-1)
    pose = np.asarray(render_pose_w2c, dtype=np.float64).reshape(4, 4)
    intr = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    if xy.shape[0] != depth.shape[0]:
        raise ValueError("render_xy and depth_values must have matching length")
    if xy.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    fx = max(float(intr[0, 0]), 1.0e-12)
    fy = max(float(intr[1, 1]), 1.0e-12)
    cx = float(intr[0, 2])
    cy = float(intr[1, 2])
    camera = np.stack(
        [
            (xy[:, 0] - cx) * depth / fx,
            (xy[:, 1] - cy) * depth / fy,
            depth,
        ],
        axis=1,
    )
    c2w = np.linalg.inv(pose)
    homog = np.concatenate([camera, np.ones((camera.shape[0], 1), dtype=np.float64)], axis=1)
    world = (c2w @ homog.T).T[:, :3]
    return world.astype(np.float32)


def _top2(similarity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if similarity.shape[1] == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    if similarity.shape[1] == 1:
        idx = np.zeros((similarity.shape[0],), dtype=np.int64)
        score = similarity[:, 0]
        return idx, score, np.full_like(score, -1.0)
    partition = np.argpartition(similarity, kth=-2, axis=1)[:, -2:]
    two_scores = np.take_along_axis(similarity, partition, axis=1)
    order = np.argsort(two_scores, axis=1)[:, ::-1]
    sorted_idx = np.take_along_axis(partition, order, axis=1)
    sorted_scores = np.take_along_axis(two_scores, order, axis=1)
    return sorted_idx[:, 0], sorted_scores[:, 0], sorted_scores[:, 1]


def generate_patch_dense_candidates(
    query_features: np.ndarray,
    rendered_features: np.ndarray,
    rendered_depth: np.ndarray,
    sparse_anchors: Mapping[str, Any] | None = None,
    *,
    policy: PatchDenseCandidatePolicy = PatchDenseCandidatePolicy(),
    render_pose_w2c: np.ndarray | None = None,
    intrinsic: np.ndarray | None = None,
) -> dict[str, Any]:
    """Generate patch-level dense candidates without selecting a final pose."""

    query = _feature_map(query_features, "query_features")
    rendered = _feature_map(rendered_features, "rendered_features")
    if query.shape != rendered.shape:
        raise ValueError("query_features and rendered_features must have matching [C, H, W] shape")
    depth = np.asarray(rendered_depth, dtype=np.float64)
    if depth.shape != query.shape[-2:]:
        raise ValueError("rendered_depth must have shape [H, W]")

    c, height, width = query.shape
    query_flat = _normalize_columns(query.reshape(c, -1))
    render_flat = _normalize_columns(rendered.reshape(c, -1))
    anchor_offsets = _anchor_offsets(sparse_anchors)
    patch_rows: list[dict[str, Any]] = []
    patch_matches: list[dict[str, np.ndarray | float | int]] = []

    for patch_id, x0, x1, y0, y1 in _patch_bounds(
        height,
        width,
        max(1, int(policy.grid_rows)),
        max(1, int(policy.grid_cols)),
    ):
        query_coords = _coords_for_patch(x0, x1, y0, y1, int(policy.sample_stride))
        render_coords = _coords_for_patch(x0, x1, y0, y1, int(policy.sample_stride))
        if query_coords.shape[0] == 0 or render_coords.shape[0] == 0:
            continue
        query_ids = query_coords[:, 1] * width + query_coords[:, 0]
        render_ids = render_coords[:, 1] * width + render_coords[:, 0]
        similarity = query_flat[:, query_ids].T @ render_flat[:, render_ids]
        best_local, best_score, second_score = _top2(similarity)
        if best_score.size == 0:
            continue
        if best_score.shape[0] > int(policy.max_matches_per_patch) > 0:
            keep = np.argsort(best_score, kind="mergesort")[::-1][: int(policy.max_matches_per_patch)]
        else:
            keep = np.arange(best_score.shape[0], dtype=np.int64)
        selected_query = query_coords[keep].astype(np.float64)
        selected_render = render_coords[best_local[keep]].astype(np.float64)
        selected_score = best_score[keep].astype(np.float64)
        selected_second = second_score[keep].astype(np.float64)
        offsets = selected_render - selected_query
        margins = selected_score - selected_second
        local_match_quality = float(np.median(selected_score)) if selected_score.size else 0.0
        margin_median = float(np.median(margins)) if margins.size else 0.0
        low_margin_fraction = (
            float(np.mean(margins < float(policy.ambiguity_margin_target))) if margins.size else 1.0
        )
        ambiguity_score = float(np.clip(low_margin_fraction, 0.0, 1.0))
        anchor_consistency = _offset_consistency(offsets, anchor_offsets, float(policy.offset_consistency_scale_px))
        off_patch_consistency = anchor_consistency
        cluster_consistency = _cluster_consistency(offsets, float(policy.offset_consistency_scale_px))
        patch_depth = depth[selected_render[:, 1].astype(np.int64), selected_render[:, 0].astype(np.int64)]
        depth_spread = _depth_spread_score(patch_depth, float(policy.min_depth_spread_m))
        patch_weight = float(
            np.clip(
                0.35 * local_match_quality
                + 0.25 * anchor_consistency
                + 0.20 * off_patch_consistency
                + 0.15 * cluster_consistency
                + 0.05 * depth_spread
                - 0.25 * ambiguity_score,
                0.0,
                1.0,
            )
        )
        median_offset = np.median(offsets, axis=0).astype(float).tolist() if offsets.size else [0.0, 0.0]
        patch_rows.append(
            {
                "patch_id": int(patch_id),
                "bounds_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                "match_count": int(selected_query.shape[0]),
                "local_match_quality": float(local_match_quality),
                "ambiguity_margin_median": float(margin_median),
                "ambiguity_low_margin_fraction": float(low_margin_fraction),
                "ambiguity_score": float(ambiguity_score),
                "anchor_consistency": float(anchor_consistency),
                "off_patch_consistency": float(off_patch_consistency),
                "patch_pose_cluster_consistency": float(cluster_consistency),
                "depth_spread_score": float(depth_spread),
                "patch_weight": float(patch_weight),
                "median_offset_xy": [float(median_offset[0]), float(median_offset[1])],
            }
        )
        patch_matches.append(
            {
                "patch_id": int(patch_id),
                "query_xy": selected_query.astype(np.float32),
                "render_xy": selected_render.astype(np.float32),
                "depth": patch_depth.astype(np.float32),
                "score": selected_score.astype(np.float32),
                "weight": np.full((selected_query.shape[0],), patch_weight, dtype=np.float32),
            }
        )

    if patch_rows:
        order = np.argsort([row["patch_weight"] for row in patch_rows], kind="mergesort")[::-1]
        keep_patch_ids = {int(patch_rows[idx]["patch_id"]) for idx in order[: max(1, int(policy.top_patch_count))]}
        selected_patch_rows = [patch_rows[idx] for idx in order if int(patch_rows[idx]["patch_id"]) in keep_patch_ids]
    else:
        keep_patch_ids = set()
        selected_patch_rows = []

    selected_matches = [item for item in patch_matches if int(item["patch_id"]) in keep_patch_ids]
    if selected_matches:
        xy = np.concatenate([item["query_xy"] for item in selected_matches], axis=0)
        render_xy = np.concatenate([item["render_xy"] for item in selected_matches], axis=0)
        dense_depth = np.concatenate([item["depth"] for item in selected_matches], axis=0)
        weights = np.concatenate([item["weight"] for item in selected_matches], axis=0)
        match_scores = np.concatenate([item["score"] for item in selected_matches], axis=0)
        patch_ids = np.concatenate(
            [np.full((item["query_xy"].shape[0],), int(item["patch_id"]), dtype=np.int32) for item in selected_matches],
            axis=0,
        )
    else:
        xy = np.empty((0, 2), dtype=np.float32)
        render_xy = np.empty((0, 2), dtype=np.float32)
        dense_depth = np.empty((0,), dtype=np.float32)
        weights = np.empty((0,), dtype=np.float32)
        match_scores = np.empty((0,), dtype=np.float32)
        patch_ids = np.empty((0,), dtype=np.int32)
    if render_pose_w2c is not None or intrinsic is not None:
        if render_pose_w2c is None or intrinsic is None:
            raise ValueError("render_pose_w2c and intrinsic must be provided together")
        p3d = _lift_render_points_to_world(
            render_xy,
            dense_depth,
            render_pose_w2c=np.asarray(render_pose_w2c, dtype=np.float64).reshape(4, 4),
            intrinsic=np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
        )
    else:
        p3d = np.empty((0, 3), dtype=np.float32)

    return {
        "schema": "loc_gs_patch_dense_candidates_v1",
        "diagnostic_only": True,
        "does_not_select_final_pose": True,
        "uses_gt": False,
        "is_oracle_branch_selector": False,
        "xy": xy,
        "render_xy": render_xy,
        "depth": dense_depth,
        "p3d": p3d,
        "weights": weights,
        "match_scores": match_scores,
        "patch_ids": patch_ids,
        "match_count": int(xy.shape[0]),
        "patch_count": int(len(selected_patch_rows)),
        "patches": selected_patch_rows,
        "policy": {
            "grid_rows": int(policy.grid_rows),
            "grid_cols": int(policy.grid_cols),
            "sample_stride": int(policy.sample_stride),
            "top_patch_count": int(policy.top_patch_count),
            "max_matches_per_patch": int(policy.max_matches_per_patch),
            "ambiguity_margin_target": float(policy.ambiguity_margin_target),
            "offset_consistency_scale_px": float(policy.offset_consistency_scale_px),
            "min_depth_spread_m": float(policy.min_depth_spread_m),
        },
    }
