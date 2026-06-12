import math

import numpy as np
import pytest

from loc_gs.core.camera import CameraIntrinsics
from loc_gs.core.geometry import project_points_w2c
from loc_gs.core.metrics import pose_error_cm_deg
from loc_gs.core.pnp import OpenCvPnPConfig, solve_pnp_ransac


def test_camera_intrinsics_builds_matrix():
    intr = CameraIntrinsics(width=100, height=80, fx=50.0, fy=60.0, cx=49.5, cy=39.5)

    np.testing.assert_allclose(
        intr.matrix,
        np.array([[50.0, 0.0, 49.5], [0.0, 60.0, 39.5], [0.0, 0.0, 1.0]], dtype=np.float64),
    )


def test_project_points_w2c_uses_xy_pixel_order():
    intr = CameraIntrinsics(width=120, height=100, fx=100.0, fy=100.0, cx=50.0, cy=40.0)
    pose = np.eye(4, dtype=np.float64)
    points = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 0.5, 2.0]], dtype=np.float64)

    projected, valid = project_points_w2c(points, pose, intr)

    np.testing.assert_allclose(projected, np.array([[50.0, 40.0], [100.0, 40.0], [50.0, 65.0]]))
    assert valid.tolist() == [True, True, True]


def test_pose_error_cm_deg_uses_camera_center_translation():
    gt = np.eye(4, dtype=np.float64)
    pred = np.eye(4, dtype=np.float64)
    pred[:3, 3] = np.array([-0.1, 0.0, 0.0])
    pred[:3, :3] = np.array(
        [
            [math.cos(math.radians(10.0)), -math.sin(math.radians(10.0)), 0.0],
            [math.sin(math.radians(10.0)), math.cos(math.radians(10.0)), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    te_cm, re_deg = pose_error_cm_deg(pred, gt)

    assert te_cm == pytest.approx(10.0, abs=1e-6)
    assert re_deg == pytest.approx(10.0, abs=1e-6)


def test_solve_pnp_ransac_recovers_synthetic_pose():
    cv2 = pytest.importorskip("cv2")
    _ = cv2
    intr = CameraIntrinsics(width=200, height=160, fx=120.0, fy=120.0, cx=100.0, cy=80.0)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.array([0.05, -0.02, 0.3], dtype=np.float64)
    points = np.array(
        [
            [-0.5, -0.3, 3.0],
            [0.4, -0.2, 3.2],
            [-0.2, 0.5, 2.8],
            [0.6, 0.4, 3.5],
            [0.0, 0.0, 2.5],
            [-0.7, 0.2, 3.3],
            [0.3, -0.6, 3.1],
            [0.8, 0.1, 2.9],
        ],
        dtype=np.float64,
    )
    keypoints_xy, valid = project_points_w2c(points, pose, intr)
    assert bool(valid.all())

    result = solve_pnp_ransac(points, keypoints_xy, intr, OpenCvPnPConfig(reprojection_error_px=1.0))

    assert result.success is True
    assert result.inlier_count >= 6
    te_cm, re_deg = pose_error_cm_deg(result.pose_w2c, pose)
    assert te_cm < 0.1
    assert re_deg < 0.1


def test_solve_pnp_ransac_reports_iterative_refinement_with_inliers():
    cv2 = pytest.importorskip("cv2")
    _ = cv2
    intr = CameraIntrinsics(width=220, height=180, fx=130.0, fy=132.0, cx=109.5, cy=89.5)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.array([0.04, -0.03, 0.22], dtype=np.float64)
    points = np.array(
        [
            [-0.6, -0.3, 3.0],
            [0.5, -0.2, 3.2],
            [-0.3, 0.4, 2.8],
            [0.7, 0.5, 3.5],
            [0.0, 0.0, 2.6],
            [-0.8, 0.2, 3.4],
            [0.2, -0.6, 3.1],
            [0.8, 0.1, 2.9],
        ],
        dtype=np.float64,
    )
    keypoints_xy, valid = project_points_w2c(points, pose, intr)
    assert bool(valid.all())

    result = solve_pnp_ransac(
        points,
        keypoints_xy,
        intr,
        OpenCvPnPConfig(
            reprojection_error_px=1.0,
            method="epnp",
            refine_with_inliers=True,
        ),
    )

    assert result.success is True
    assert result.refined is True
    assert result.method == "epnp"
    assert result.refinement_method == "iterative"
    te_cm, re_deg = pose_error_cm_deg(result.pose_w2c, pose)
    assert te_cm < 0.1
    assert re_deg < 0.1
