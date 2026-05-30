from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from loc_gs.diagnostics.match_visualization import project_points


@dataclass(frozen=True)
class SparseAnchorResidualPolicy:
    dense_group_weight: float = 1.0
    anchor_group_weight: float = 1.0
    robust_scale_px: float = 4.0
    max_anchor_count: int = 128


def _as_errors(values: Any) -> np.ndarray:
    errors = np.asarray(values, dtype=np.float64).reshape(-1)
    return errors[np.isfinite(errors) & (errors >= 0.0)]


def _huber_loss(errors_px: np.ndarray, scale_px: float) -> np.ndarray:
    errors = _as_errors(errors_px)
    scale = max(float(scale_px), 1.0e-6)
    quadratic = errors <= scale
    loss = np.empty_like(errors, dtype=np.float64)
    loss[quadratic] = 0.5 * errors[quadratic] ** 2
    loss[~quadratic] = scale * (errors[~quadratic] - 0.5 * scale)
    return loss


def _mean_loss(errors_px: np.ndarray, scale_px: float) -> float:
    loss = _huber_loss(errors_px, scale_px)
    return float(np.mean(loss)) if loss.size else 0.0


def _sum_loss(errors_px: np.ndarray, scale_px: float) -> float:
    loss = _huber_loss(errors_px, scale_px)
    return float(np.sum(loss)) if loss.size else 0.0


def _selected_anchor_arrays(anchors: Mapping[str, Any], max_anchor_count: int) -> tuple[np.ndarray, np.ndarray, int]:
    query_xy = np.asarray(anchors.get("query_xy", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    points = np.asarray(anchors.get("p3d", np.empty((0, 3))), dtype=np.float64).reshape(-1, 3)
    inliers = np.asarray(anchors.get("inliers", np.arange(min(query_xy.shape[0], points.shape[0]))), dtype=np.int64).reshape(-1)
    count = min(query_xy.shape[0], points.shape[0])
    valid = inliers[(inliers >= 0) & (inliers < count)]
    candidate_count = int(valid.shape[0])
    if int(max_anchor_count) > 0 and valid.shape[0] > int(max_anchor_count):
        valid = valid[: int(max_anchor_count)]
    return query_xy[valid], points[valid], candidate_count


def compute_sparse_anchor_residual_group(
    sparse_anchors: Mapping[str, Any],
    *,
    pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    policy: SparseAnchorResidualPolicy = SparseAnchorResidualPolicy(),
) -> dict[str, Any]:
    """Compute sparse-anchor reprojection residuals as a separate diagnostic group."""

    anchor_xy, anchor_xyz, candidate_count = _selected_anchor_arrays(sparse_anchors, int(policy.max_anchor_count))
    if anchor_xy.shape[0] == 0:
        errors = np.empty((0,), dtype=np.float64)
        valid = np.empty((0,), dtype=bool)
    else:
        projected, valid = project_points(
            anchor_xyz,
            np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4),
            np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
            width=int(image_size[0]),
            height=int(image_size[1]),
        )
        errors = np.linalg.norm(np.asarray(projected, dtype=np.float64) - anchor_xy, axis=1)
        errors[~valid] = np.inf
    finite = errors[np.isfinite(errors)]
    loss = _huber_loss(finite, float(policy.robust_scale_px))
    return {
        "schema": "loc_gs_sparse_anchor_residual_group_v1",
        "diagnostic_only": True,
        "does_not_select_final_pose": True,
        "uses_gt": False,
        "candidate_anchor_count": int(candidate_count),
        "evaluated_anchor_count": int(anchor_xy.shape[0]),
        "valid_anchor_count": int(finite.shape[0]),
        "median_error_px": float(np.median(finite)) if finite.size else None,
        "p90_error_px": float(np.percentile(finite, 90)) if finite.size else None,
        "group_loss": float(np.mean(loss)) if loss.size else 0.0,
        "loss_sum": float(np.sum(loss)) if loss.size else 0.0,
        "robust_scale_px": float(policy.robust_scale_px),
        "anchor_group_weight": float(policy.anchor_group_weight),
        "errors_px": finite.astype(np.float32),
    }


def summarize_sparse_anchor_residual_modes(
    dense_errors_px: np.ndarray,
    anchor_errors_px: np.ndarray,
    *,
    policy: SparseAnchorResidualPolicy = SparseAnchorResidualPolicy(),
) -> dict[str, Any]:
    """Compare no-anchor, concatenated-anchor, and separate-group objectives."""

    dense = _as_errors(dense_errors_px)
    anchors = _as_errors(anchor_errors_px)
    dense_mean = _mean_loss(dense, float(policy.robust_scale_px))
    anchor_mean = _mean_loss(anchors, float(policy.robust_scale_px))
    dense_sum = _sum_loss(dense, float(policy.robust_scale_px))
    anchor_sum = _sum_loss(anchors, float(policy.robust_scale_px))
    concat_total = dense_sum + anchor_sum
    separate_dense = float(policy.dense_group_weight) * dense_mean
    separate_anchor = float(policy.anchor_group_weight) * anchor_mean
    separate_total = separate_dense + separate_anchor
    return {
        "schema": "loc_gs_sparse_anchor_residual_modes_v1",
        "diagnostic_only": True,
        "does_not_select_final_pose": True,
        "uses_gt": False,
        "dense_count": int(dense.shape[0]),
        "anchor_count": int(anchors.shape[0]),
        "robust_scale_px": float(policy.robust_scale_px),
        "no_anchor": {
            "total_loss": float(dense_mean),
            "dense_loss": float(dense_mean),
            "anchor_loss": 0.0,
            "anchor_loss_share": 0.0,
        },
        "concat_matches": {
            "total_loss": float(concat_total / max(1, dense.shape[0] + anchors.shape[0])),
            "dense_loss_sum": float(dense_sum),
            "anchor_loss_sum": float(anchor_sum),
            "anchor_loss_share": float(anchor_sum / max(concat_total, 1.0e-12)),
        },
        "separate_residual_group": {
            "total_loss": float(separate_total),
            "dense_group_loss": float(separate_dense),
            "anchor_group_loss": float(separate_anchor),
            "anchor_loss_share": float(separate_anchor / max(separate_total, 1.0e-12)),
        },
    }
