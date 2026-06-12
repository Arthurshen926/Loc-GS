import numpy as np

from loc_gs.core.camera import CameraIntrinsics
from loc_gs.core.geometry import project_points_w2c
from loc_gs.sparse.sparse_lgcv import filter_correspondences_by_reprojection


def test_sparse_lgcv_filters_high_residual_correspondences():
    intr = CameraIntrinsics(width=240, height=180, fx=140.0, fy=140.0, cx=120.0, cy=90.0)
    pose = np.eye(4, dtype=np.float64)
    points = np.array(
        [
            [-0.4, -0.3, 3.0],
            [0.4, -0.2, 3.2],
            [-0.2, 0.5, 2.8],
            [0.5, 0.4, 3.4],
            [0.0, 0.0, 2.7],
        ],
        dtype=np.float64,
    )
    keypoints_xy, valid = project_points_w2c(points, pose, intr)
    assert bool(valid.all())
    keypoints_xy[-1] += np.array([35.0, 0.0], dtype=np.float64)

    result = filter_correspondences_by_reprojection(
        points,
        keypoints_xy,
        pose,
        intr,
        threshold_px=4.0,
    )

    assert result.keep_mask.tolist() == [True, True, True, True, False]
    assert result.keep_count == 4
    assert result.residuals_px[-1] > 30.0
