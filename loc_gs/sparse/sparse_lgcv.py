from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loc_gs.core.camera import CameraIntrinsics
from loc_gs.core.geometry import project_points_w2c


@dataclass(frozen=True)
class SparseLGCVResult:
    keep_mask: np.ndarray
    residuals_px: np.ndarray

    @property
    def keep_count(self) -> int:
        return int(np.asarray(self.keep_mask, dtype=bool).sum())


def filter_correspondences_by_reprojection(
    points3d_world: np.ndarray,
    keypoints_xy: np.ndarray,
    pose_w2c: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    threshold_px: float,
) -> SparseLGCVResult:
    points = np.asarray(points3d_world, dtype=np.float64).reshape(-1, 3)
    keypoints = np.asarray(keypoints_xy, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] != keypoints.shape[0]:
        raise ValueError("points3d_world and keypoints_xy must contain the same number of correspondences")
    projected_xy, valid = project_points_w2c(points, pose_w2c, intrinsics)
    residuals = np.linalg.norm(projected_xy - keypoints, axis=1)
    finite = np.isfinite(residuals)
    keep = np.asarray(valid, dtype=bool) & finite & (residuals <= float(threshold_px))
    residuals = np.where(finite, residuals, np.inf).astype(np.float64)
    return SparseLGCVResult(keep_mask=keep.astype(bool), residuals_px=residuals)
