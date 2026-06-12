from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loc_gs.core.camera import CameraIntrinsics
from loc_gs.core.geometry import project_points_w2c


@dataclass(frozen=True)
class GaussianVisibility:
    xy: np.ndarray
    depth: np.ndarray
    visible_mask: np.ndarray

    @property
    def visible_indices(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.visible_mask, dtype=bool))

    @property
    def visible_count(self) -> int:
        return int(self.visible_indices.shape[0])


def project_gaussian_visibility(
    positions_xyz: np.ndarray,
    *,
    pose_w2c: np.ndarray,
    intrinsics: CameraIntrinsics,
    min_depth: float = 1.0e-9,
    max_depth: float | None = None,
) -> GaussianVisibility:
    points = np.asarray(positions_xyz, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    if points.shape[0] == 0:
        return GaussianVisibility(
            xy=np.empty((0, 2), dtype=np.float64),
            depth=np.empty((0,), dtype=np.float64),
            visible_mask=np.empty((0,), dtype=bool),
        )
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    camera = (pose @ homog.T).T[:, :3]
    depth = camera[:, 2].astype(np.float64)
    xy, mask = project_points_w2c(points, pose, intrinsics, min_depth=float(min_depth))
    if max_depth is not None:
        mask = mask & (depth <= float(max_depth))
    return GaussianVisibility(xy=xy, depth=depth, visible_mask=mask.astype(bool))
