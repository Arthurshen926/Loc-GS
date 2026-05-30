import numpy as np

from loc_gs.dense_support.sparse_anchor_residual import (
    SparseAnchorResidualPolicy,
    compute_sparse_anchor_residual_group,
    summarize_sparse_anchor_residual_modes,
)


def test_sparse_anchor_residual_group_projects_anchors_and_reports_robust_loss():
    anchors = {
        "query_xy": np.array([[1.0, 1.0], [2.0, 1.0], [50.0, 50.0]], dtype=np.float32),
        "p3d": np.array([[1.0, 1.0, 10.0], [2.0, 1.0, 10.0], [1.0, 1.0, -1.0]], dtype=np.float32),
        "inliers": np.array([0, 1, 2], dtype=np.int32),
    }
    intrinsic = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    group = compute_sparse_anchor_residual_group(
        anchors,
        pose_w2c=np.eye(4, dtype=np.float32),
        intrinsic=intrinsic,
        image_size=(100, 100),
        policy=SparseAnchorResidualPolicy(robust_scale_px=4.0),
    )

    assert group["schema"] == "loc_gs_sparse_anchor_residual_group_v1"
    assert group["diagnostic_only"] is True
    assert group["candidate_anchor_count"] == 3
    assert group["valid_anchor_count"] == 2
    assert group["median_error_px"] == 0.0
    assert group["group_loss"] == 0.0


def test_sparse_anchor_residual_modes_show_concat_dilution_and_separate_group_share():
    dense_errors = np.full((100,), 2.0, dtype=np.float32)
    anchor_errors = np.array([2.0, 2.0], dtype=np.float32)

    summary = summarize_sparse_anchor_residual_modes(
        dense_errors,
        anchor_errors,
        policy=SparseAnchorResidualPolicy(anchor_group_weight=4.0, robust_scale_px=4.0),
    )

    assert summary["schema"] == "loc_gs_sparse_anchor_residual_modes_v1"
    assert summary["no_anchor"]["anchor_loss_share"] == 0.0
    assert summary["concat_matches"]["anchor_loss_share"] < 0.25
    assert summary["separate_residual_group"]["anchor_loss_share"] >= 0.80
    assert summary["separate_residual_group"]["total_loss"] > summary["no_anchor"]["total_loss"]


def test_sparse_anchor_residual_modes_are_diagnostic_not_pose_selection():
    summary = summarize_sparse_anchor_residual_modes(
        dense_errors_px=np.array([1.0, 2.0], dtype=np.float32),
        anchor_errors_px=np.array([1.0], dtype=np.float32),
    )

    assert summary["diagnostic_only"] is True
    assert summary["does_not_select_final_pose"] is True
    assert summary["uses_gt"] is False
