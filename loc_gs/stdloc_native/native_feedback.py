from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from loc_gs.feedback.schema import FeedbackMatchRecord


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _at(values: Sequence[Any], index: int, default: Any = None) -> Any:
    return values[index] if index < len(values) else default


def reprojection_errors_px(
    p2d: Any,
    p3d: Any,
    intrinsic: Any,
    pose_w2c: Any,
) -> list[float]:
    points_2d = np.asarray(p2d, dtype=np.float64).reshape(-1, 2)
    points_3d = np.asarray(p3d, dtype=np.float64).reshape(-1, 3)
    if points_2d.shape[0] != points_3d.shape[0]:
        raise ValueError("p2d and p3d must contain the same number of correspondences")
    if points_2d.shape[0] == 0:
        return []
    k_mat = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    w2c = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    points_3d_h = np.concatenate(
        [points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    camera_points = (w2c[:3, :] @ points_3d_h.T).T
    projected_h = (k_mat @ camera_points.T).T
    valid = np.isfinite(projected_h).all(axis=1) & (np.abs(projected_h[:, 2]) > 1e-12)
    errors = np.full((points_2d.shape[0],), np.inf, dtype=np.float64)
    projected = projected_h[valid, :2] / projected_h[valid, 2:3]
    errors[valid] = np.linalg.norm(points_2d[valid] - projected, axis=1)
    return [float(item) for item in errors.tolist()]


def dense_transition_from_errors(
    *,
    sparse_te_cm: float | None,
    dense_te_cm: float | None,
    te_epsilon_cm: float = 0.0,
) -> tuple[str, float | None]:
    sparse = _float_or_none(sparse_te_cm)
    dense = _float_or_none(dense_te_cm)
    if sparse is None or dense is None:
        return "", None
    delta = dense - sparse
    if delta < -float(te_epsilon_cm):
        return "improved", float(delta)
    if delta > float(te_epsilon_cm):
        return "worsened", float(delta)
    return "unchanged", float(delta)


def records_from_native_sparse_capture(
    *,
    scene: str,
    image_name: str,
    keypoint_indices: Sequence[Any],
    keypoint_xy: Sequence[Sequence[Any]],
    landmark_ids: Sequence[Any],
    gaussian_ids: Sequence[Any],
    descriptor_scores: Sequence[Any],
    detector_scores: Sequence[Any],
    match_ranks: Sequence[Any],
    inlier_indices: Sequence[Any],
    reprojection_errors_px: Sequence[Any],
    visibility_scores: Sequence[Any] | None = None,
    sparse_te_cm: float | None = None,
    sparse_re_deg: float | None = None,
    dense_te_cm: float | None = None,
    dense_re_deg: float | None = None,
    dense_transition: str = "",
    dense_delta_te_cm: float | None = None,
) -> list[FeedbackMatchRecord]:
    count = len(gaussian_ids)
    if not (
        len(keypoint_indices)
        == len(keypoint_xy)
        == len(landmark_ids)
        == len(descriptor_scores)
        == len(detector_scores)
        == len(match_ranks)
        == len(reprojection_errors_px)
        == count
    ):
        raise ValueError("native sparse capture fields must have the same length")
    inliers = {int(item) for item in inlier_indices}
    transition = str(dense_transition or "")
    delta = _float_or_none(dense_delta_te_cm)
    if not transition and delta is None:
        transition, delta = dense_transition_from_errors(
            sparse_te_cm=sparse_te_cm,
            dense_te_cm=dense_te_cm,
        )
    dense_success = _float_or_none(dense_te_cm) is not None and _float_or_none(dense_re_deg) is not None
    pnp_success = len(inliers) >= 4
    visibility = list(visibility_scores or [])
    records: list[FeedbackMatchRecord] = []
    for index in range(count):
        xy = keypoint_xy[index]
        records.append(
            FeedbackMatchRecord(
                scene=str(scene),
                query_id=str(image_name),
                image_id=str(image_name),
                source_view_id="native_stdloc_sparse",
                pose_source="selfmap_native_stdloc",
                keypoint_id=f"{image_name}#{keypoint_indices[index]}",
                keypoint_xy=(
                    _float_or_none(_at(xy, 0)),
                    _float_or_none(_at(xy, 1)),
                ),
                matched_landmark_id=str(int(landmark_ids[index])),
                matched_gaussian_id=str(int(gaussian_ids[index])),
                descriptor_score=_float_or_none(descriptor_scores[index]),
                detector_score=_float_or_none(detector_scores[index]),
                match_rank=int(match_ranks[index]),
                pnp_inlier=index in inliers,
                reprojection_error_px=_float_or_none(reprojection_errors_px[index]),
                depth_consistency=None,
                visibility_score=_float_or_none(_at(visibility, index)),
                pose_error_t_cm=_float_or_none(dense_te_cm),
                pose_error_r_deg=_float_or_none(dense_re_deg),
                pnp_success=pnp_success,
                dense_refine_success=dense_success,
                dense_transition=transition,
                dense_delta_te_cm=delta,
            )
        )
    return records
