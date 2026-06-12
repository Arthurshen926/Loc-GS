from __future__ import annotations

import numpy as np

from loc_gs.core.camera import CameraIntrinsics


def project_points_w2c(
    points_world: np.ndarray,
    pose_w2c: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    min_depth: float = 1.0e-9,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=bool)

    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    camera = (pose @ homog.T).T[:, :3]
    z = camera[:, 2]
    positive_depth = z > float(min_depth)
    safe_z = np.where(positive_depth, z, 1.0)
    x = float(intrinsics.fx) * camera[:, 0] / safe_z + float(intrinsics.cx)
    y = float(intrinsics.fy) * camera[:, 1] / safe_z + float(intrinsics.cy)
    xy = np.stack([x, y], axis=-1)
    finite = np.all(np.isfinite(xy), axis=1)
    in_bounds = (
        (xy[:, 0] >= 0.0)
        & (xy[:, 0] < float(intrinsics.width))
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] < float(intrinsics.height))
    )
    return xy.astype(np.float64), (positive_depth & finite & in_bounds).astype(bool)
