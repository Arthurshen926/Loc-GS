import numpy as np
import pytest

from loc_gs.core.camera import CameraIntrinsics
from loc_gs.core.geometry import project_points_w2c
from loc_gs.core.metrics import pose_error_cm_deg
from loc_gs.sparse.pipeline import (
    SparseLocalizationConfig,
    SparseLocalizationInput,
    SparsePipelineCandidate,
    run_sparse_localization,
)


def _synthetic_input(*, correct_solver_score: float) -> tuple[SparseLocalizationInput, np.ndarray]:
    intr = CameraIntrinsics(width=240, height=180, fx=140.0, fy=140.0, cx=120.0, cy=90.0)
    gt_pose = np.eye(4, dtype=np.float64)
    gt_pose[:3, 3] = np.array([0.03, -0.04, 0.25], dtype=np.float64)
    correct_points = np.array(
        [
            [-0.5, -0.4, 3.0],
            [0.5, -0.4, 3.1],
            [-0.4, 0.5, 2.9],
            [0.4, 0.5, 3.3],
            [0.0, 0.0, 2.6],
            [-0.7, 0.1, 3.4],
        ],
        dtype=np.float64,
    )
    keypoints_xy, valid = project_points_w2c(correct_points, gt_pose, intr)
    assert bool(valid.all())
    candidates_by_keypoint = []
    for idx, point in enumerate(correct_points):
        wrong = point + np.array([2.0, 0.0, 0.0], dtype=np.float64)
        candidates_by_keypoint.append(
            [
                SparsePipelineCandidate(
                    keypoint_index=idx,
                    landmark_id=1000 + idx,
                    point3d=wrong,
                    native_score=0.99,
                    solver_score=0.0,
                    geometric_correct=False,
                ),
                SparsePipelineCandidate(
                    keypoint_index=idx,
                    landmark_id=idx,
                    point3d=point,
                    native_score=0.80,
                    solver_score=correct_solver_score,
                    geometric_correct=True,
                ),
            ]
        )
    return (
        SparseLocalizationInput(
            scene="GreatCourt",
            split_name="train_dev",
            query_id="q1",
            intrinsics=intr,
            keypoints_xy=keypoints_xy,
            candidates_by_keypoint=candidates_by_keypoint,
        ),
        gt_pose,
    )


def test_sparse_pipeline_uses_reranked_candidates_for_pnp():
    data, gt_pose = _synthetic_input(correct_solver_score=2.0)

    result = run_sparse_localization(
        data,
        SparseLocalizationConfig(
            rerank_prefix_fraction=0.0,
            solver_weight=1.0,
            native_weight=0.1,
            reprojection_error_px=1.0,
        ),
    )

    assert result.success is True
    assert result.inlier_count >= 5
    assert result.selected_landmark_ids == [0, 1, 2, 3, 4, 5]
    assert result.availability_summary["top1_correct"] == 0
    assert result.availability_summary["topk_available"] == 6
    te_cm, re_deg = pose_error_cm_deg(result.pose_w2c, gt_pose)
    assert te_cm < 0.2
    assert re_deg < 0.2


def test_sparse_pipeline_native_order_can_expose_oracle_gap():
    data, gt_pose = _synthetic_input(correct_solver_score=0.0)

    result = run_sparse_localization(
        data,
        SparseLocalizationConfig(
            rerank_prefix_fraction=1.0,
            solver_weight=0.0,
            native_weight=1.0,
            reprojection_error_px=1.0,
        ),
    )

    assert result.selected_landmark_ids == [1000, 1001, 1002, 1003, 1004, 1005]
    assert result.availability_summary["oracle_gap"] == 6
    assert result.success is True
    assert result.inlier_count == 6
    te_cm, _re_deg = pose_error_cm_deg(result.pose_w2c, gt_pose)
    assert te_cm > 100.0


def test_sparse_pipeline_rejects_test_split():
    data, _gt_pose = _synthetic_input(correct_solver_score=2.0)
    bad = SparseLocalizationInput(
        scene=data.scene,
        split_name="test",
        query_id=data.query_id,
        intrinsics=data.intrinsics,
        keypoints_xy=data.keypoints_xy,
        candidates_by_keypoint=data.candidates_by_keypoint,
    )

    with pytest.raises(ValueError, match="test split"):
        run_sparse_localization(bad, SparseLocalizationConfig())
