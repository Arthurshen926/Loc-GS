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
    assert result.selected_set_diagnostics["selected_geometric_correct_count"] == 6
    assert result.selected_set_diagnostics["selected_geometric_correct_ratio"] == 1.0
    assert result.selected_set_diagnostics["selected_keypoint_bbox_area_fraction"] > 0.01
    assert result.selected_set_diagnostics["selected_depth_range_m"] == pytest.approx(0.8)
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
    assert result.selected_set_diagnostics["selected_geometric_correct_count"] == 0
    assert result.selected_set_diagnostics["selected_geometric_correct_ratio"] == 0.0
    assert result.availability_summary["oracle_gap"] == 6
    assert result.success is True
    assert result.inlier_count == 6
    te_cm, _re_deg = pose_error_cm_deg(result.pose_w2c, gt_pose)
    assert te_cm > 100.0


def test_sparse_pipeline_can_run_lgcv_second_pnp():
    intr = CameraIntrinsics(width=260, height=200, fx=150.0, fy=152.0, cx=129.5, cy=99.5)
    gt_pose = np.eye(4, dtype=np.float64)
    gt_pose[:3, 3] = np.array([0.02, -0.03, 0.2], dtype=np.float64)
    correct_points = np.array(
        [
            [-0.6, -0.3, 3.0],
            [0.5, -0.2, 3.2],
            [-0.3, 0.5, 2.9],
            [0.6, 0.4, 3.4],
            [0.0, 0.0, 2.6],
            [-0.8, 0.1, 3.5],
            [0.3, -0.6, 3.1],
            [0.8, 0.2, 3.0],
        ],
        dtype=np.float64,
    )
    keypoints_xy, valid = project_points_w2c(correct_points, gt_pose, intr)
    assert bool(valid.all())
    selected_points = list(correct_points)
    selected_points.append(correct_points[0] + np.array([8.0, 8.0, 0.0], dtype=np.float64))
    selected_points.append(correct_points[1] + np.array([-8.0, 7.0, 0.0], dtype=np.float64))
    keypoints_with_outliers = np.concatenate([keypoints_xy, keypoints_xy[:2]], axis=0)
    candidates = [
        [
            SparsePipelineCandidate(
                keypoint_index=idx,
                landmark_id=idx,
                point3d=np.asarray(point, dtype=np.float64).tolist(),
                native_score=1.0,
                solver_score=0.0,
                geometric_correct=idx < len(correct_points),
            )
        ]
        for idx, point in enumerate(selected_points)
    ]
    data = SparseLocalizationInput(
        scene="GreatCourt",
        split_name="train_dev",
        query_id="q-lgcv",
        intrinsics=intr,
        keypoints_xy=keypoints_with_outliers,
        candidates_by_keypoint=candidates,
    )

    result = run_sparse_localization(
        data,
        SparseLocalizationConfig(
            pnp_method="epnp",
            second_pnp_enabled=True,
            second_pnp_method="iterative",
            lgcv_reprojection_error_px=3.0,
            refine_with_inliers=True,
            reprojection_error_px=4.0,
        ),
    )

    assert result.success is True
    assert result.pnp_stage_count == 2
    assert result.initial_inlier_count >= 8
    assert result.lgcv_keep_count == 8
    assert result.inlier_count >= 8
    te_cm, re_deg = pose_error_cm_deg(result.pose_w2c, gt_pose)
    assert te_cm < 0.5
    assert re_deg < 0.5


def test_sparse_pipeline_post_pnp_rescore_recovers_buried_candidate():
    intr = CameraIntrinsics(width=260, height=200, fx=150.0, fy=152.0, cx=129.5, cy=99.5)
    gt_pose = np.eye(4, dtype=np.float64)
    gt_pose[:3, 3] = np.array([0.02, -0.03, 0.2], dtype=np.float64)
    correct_points = np.array(
        [
            [-0.6, -0.3, 3.0],
            [0.5, -0.2, 3.2],
            [-0.3, 0.5, 2.9],
            [0.6, 0.4, 3.4],
            [0.0, 0.0, 2.6],
            [-0.8, 0.1, 3.5],
        ],
        dtype=np.float64,
    )
    keypoints_xy, valid = project_points_w2c(correct_points, gt_pose, intr)
    assert bool(valid.all())
    candidates = []
    for idx, point in enumerate(correct_points):
        if idx == len(correct_points) - 1:
            wrong = point + np.array([4.0, 0.0, 0.0], dtype=np.float64)
            candidates.append(
                [
                    SparsePipelineCandidate(
                        keypoint_index=idx,
                        landmark_id=1000 + idx,
                        point3d=wrong.tolist(),
                        native_score=1.0,
                        solver_score=0.0,
                        geometric_correct=False,
                    ),
                    SparsePipelineCandidate(
                        keypoint_index=idx,
                        landmark_id=idx,
                        point3d=point.tolist(),
                        native_score=0.1,
                        solver_score=0.0,
                        geometric_correct=True,
                    ),
                ]
            )
        else:
            candidates.append(
                [
                    SparsePipelineCandidate(
                        keypoint_index=idx,
                        landmark_id=idx,
                        point3d=point.tolist(),
                        native_score=1.0,
                        solver_score=0.0,
                        geometric_correct=True,
                    )
                ]
            )
    data = SparseLocalizationInput(
        scene="GreatCourt",
        split_name="train_dev",
        query_id="q-post-pnp",
        intrinsics=intr,
        keypoints_xy=keypoints_xy,
        candidates_by_keypoint=candidates,
    )

    result = run_sparse_localization(
        data,
        SparseLocalizationConfig(
            pnp_method="epnp",
            second_pnp_enabled=True,
            second_pnp_method="iterative",
            post_pnp_candidate_rescore=True,
            post_pnp_reprojection_weight=4.0,
            lgcv_reprojection_error_px=3.0,
            reprojection_error_px=4.0,
        ),
    )

    assert result.success is True
    assert result.post_pnp_rescore_changed_count == 1
    assert result.post_pnp_rescore_corrected_count == 1
    assert result.post_pnp_rescore_worsened_count == 0
    assert result.post_pnp_rescore_correct_delta == 1
    assert result.selected_landmark_ids[-1] == len(correct_points) - 1
    assert result.lgcv_keep_count == 6
    te_cm, re_deg = pose_error_cm_deg(result.pose_w2c, gt_pose)
    assert te_cm < 0.5
    assert re_deg < 0.5


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
