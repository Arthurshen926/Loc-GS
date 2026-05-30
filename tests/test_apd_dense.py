import numpy as np

from loc_gs.dense_support.apd_dense import (
    APDDensePolicy,
    refine_pose_with_sparse_anchors_and_dense_candidates,
    run_anchor_patch_dense_refinement,
    score_dense_candidates,
)
from loc_gs.diagnostics.match_visualization import project_points


def _camera():
    intrinsic = np.array(
        [[80.0, 0.0, 32.0], [0.0, 80.0, 24.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float32)
    return pose, intrinsic, (64, 48)


def _points():
    return np.array(
        [
            [-0.20, -0.10, 4.0],
            [0.20, -0.10, 4.5],
            [-0.15, 0.15, 5.0],
            [0.18, 0.12, 5.5],
            [0.00, 0.00, 6.0],
            [0.28, 0.18, 7.0],
        ],
        dtype=np.float32,
    )


def _anchors(pose, intrinsic, image_size):
    xy, _valid = project_points(_points(), pose, intrinsic, width=image_size[0], height=image_size[1])
    return {
        "query_xy": xy.astype(np.float32),
        "p3d": _points(),
        "inliers": np.arange(xy.shape[0], dtype=np.int32),
    }


def test_score_dense_candidates_uses_geometric_mean_of_three_core_signals():
    pose, intrinsic, image_size = _camera()
    points = _points()
    xy, _valid = project_points(points, pose, intrinsic, width=image_size[0], height=image_size[1])
    shifted_xy = xy + np.array([12.0, 0.0], dtype=np.float32)

    good = score_dense_candidates(
        {"query_xy": xy, "p3d": points, "match_scores": np.ones((points.shape[0],), dtype=np.float32)},
        sparse_pose=pose,
        sparse_anchors=_anchors(pose, intrinsic, image_size),
        intrinsic=intrinsic,
        image_size=image_size,
        policy=APDDensePolicy(anchor_consistency_scale_px=4.0),
    )
    bad = score_dense_candidates(
        {"query_xy": shifted_xy, "p3d": points, "match_scores": np.ones((points.shape[0],), dtype=np.float32)},
        sparse_pose=pose,
        sparse_anchors=_anchors(pose, intrinsic, image_size),
        intrinsic=intrinsic,
        image_size=image_size,
        policy=APDDensePolicy(anchor_consistency_scale_px=4.0),
    )

    assert good["schema"] == "loc_gs_apd_dense_candidate_scores_v1"
    assert good["uses_gt"] is False
    assert good["weight_mean"] > bad["weight_mean"]
    assert good["anchor_consistency_median"] > bad["anchor_consistency_median"]
    assert set(good["core_signals"]) == {"anchor_consistency", "geometry", "match_quality"}


def test_score_dense_candidates_uses_camera_depth_not_world_z_for_geometry():
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.array(
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    intrinsic = np.array(
        [[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    image_size = (64, 48)
    points = np.array(
        [
            [4.0, -0.05, 0.0],
            [4.1, -0.05, 0.0],
            [4.2, 0.10, 0.0],
            [4.3, 0.10, 0.0],
        ],
        dtype=np.float32,
    )
    xy = np.array([[30.0, 22.0], [34.0, 22.0], [31.0, 26.0], [35.0, 26.0]], dtype=np.float32)

    result = score_dense_candidates(
        {"query_xy": xy, "p3d": points, "match_scores": np.ones((points.shape[0],), dtype=np.float32)},
        sparse_pose=pose,
        sparse_anchors={"query_xy": xy, "p3d": points, "inliers": np.arange(points.shape[0], dtype=np.int32)},
        intrinsic=intrinsic,
        image_size=image_size,
        policy=APDDensePolicy(min_depth_spread_m=0.05),
    )

    assert np.std(points[:, 2]) == 0.0
    assert result["geometry_score"] > 0.0


def test_anchor_verified_refinement_group_normalizes_sparse_anchors():
    pose, intrinsic, image_size = _camera()
    points = _points()
    anchor_xy, _valid = project_points(points, pose, intrinsic, width=image_size[0], height=image_size[1])
    dense_xy = anchor_xy + np.array([10.0, 0.0], dtype=np.float32)
    anchors = _anchors(pose, intrinsic, image_size)

    dense_only_pose, dense_only_diag = refine_pose_with_sparse_anchors_and_dense_candidates(
        sparse_pose=pose,
        sparse_anchors={"query_xy": np.empty((0, 2)), "p3d": np.empty((0, 3)), "inliers": np.empty((0,), dtype=np.int32)},
        dense_candidates={"query_xy": dense_xy, "p3d": points, "weights": np.ones((points.shape[0],), dtype=np.float32)},
        intrinsic=intrinsic,
        image_size=image_size,
        policy=APDDensePolicy(anchor_group_weight=0.0, dense_group_weight=1.0),
    )
    anchored_pose, anchored_diag = refine_pose_with_sparse_anchors_and_dense_candidates(
        sparse_pose=pose,
        sparse_anchors=anchors,
        dense_candidates={"query_xy": dense_xy, "p3d": points, "weights": np.ones((points.shape[0],), dtype=np.float32)},
        intrinsic=intrinsic,
        image_size=image_size,
        policy=APDDensePolicy(anchor_group_weight=1.0, dense_group_weight=1.0),
    )

    anchor_ref_xy = anchors["query_xy"]
    dense_only_proj, _ = project_points(points, dense_only_pose, intrinsic, width=image_size[0], height=image_size[1])
    anchored_proj, _ = project_points(points, anchored_pose, intrinsic, width=image_size[0], height=image_size[1])
    dense_only_anchor_error = np.median(np.linalg.norm(dense_only_proj - anchor_ref_xy, axis=1))
    anchored_anchor_error = np.median(np.linalg.norm(anchored_proj - anchor_ref_xy, axis=1))

    assert anchored_diag["schema"] == "loc_gs_apd_anchor_verified_refinement_v1"
    assert anchored_diag["uses_gt"] is False
    assert anchored_diag["anchor_residual_count"] == points.shape[0]
    assert anchored_diag["dense_candidate_count"] == points.shape[0]
    assert dense_only_diag["anchor_residual_count"] == 0
    assert anchored_anchor_error < dense_only_anchor_error


def test_anchor_verified_refinement_can_use_dense_reference_pose_instead_of_sparse_reference():
    pose, intrinsic, image_size = _camera()
    points = _points()
    dense_xy, _valid = project_points(points, pose, intrinsic, width=image_size[0], height=image_size[1])
    sparse_pose = pose.copy()
    sparse_pose[0, 3] = 0.25
    anchors = _anchors(pose, intrinsic, image_size)

    sparse_refined, sparse_ref_diag = refine_pose_with_sparse_anchors_and_dense_candidates(
        sparse_pose=sparse_pose,
        sparse_anchors=anchors,
        dense_candidates={"query_xy": dense_xy, "p3d": points, "weights": np.ones((points.shape[0],), dtype=np.float32)},
        intrinsic=intrinsic,
        image_size=image_size,
        policy=APDDensePolicy(anchor_group_weight=0.0, dense_group_weight=1.0, max_refine_iterations=5),
    )
    dense_refined, dense_ref_diag = refine_pose_with_sparse_anchors_and_dense_candidates(
        sparse_pose=sparse_pose,
        reference_pose=pose,
        sparse_anchors=anchors,
        dense_candidates={"query_xy": dense_xy, "p3d": points, "weights": np.ones((points.shape[0],), dtype=np.float32)},
        intrinsic=intrinsic,
        image_size=image_size,
        policy=APDDensePolicy(anchor_group_weight=0.0, dense_group_weight=1.0, max_refine_iterations=5),
    )

    assert sparse_ref_diag["reference_pose_source"] == "sparse_pose"
    assert dense_ref_diag["reference_pose_source"] == "dense_reference_pose"
    assert np.linalg.norm(dense_refined[:3, 3] - pose[:3, 3]) < np.linalg.norm(sparse_refined[:3, 3] - pose[:3, 3])


def test_run_anchor_patch_dense_refinement_combines_global_clean_and_patch_sources():
    pose, intrinsic, image_size = _camera()
    points = _points()
    xy, _valid = project_points(points, pose, intrinsic, width=image_size[0], height=image_size[1])
    anchors = _anchors(pose, intrinsic, image_size)
    global_candidates = {
        "query_xy": xy[:2],
        "p3d": points[:2],
        "match_scores": np.ones((2,), dtype=np.float32),
        "source": "global",
    }
    clean_candidates = {
        "query_xy": xy[2:4],
        "p3d": points[2:4],
        "match_scores": np.ones((2,), dtype=np.float32),
        "source": "clean_render",
    }
    patch_candidates = {
        "query_xy": xy[4:],
        "p3d": points[4:],
        "match_scores": np.ones((2,), dtype=np.float32),
        "source": "patch",
    }

    result = run_anchor_patch_dense_refinement(
        sparse_pose=pose,
        sparse_anchors=anchors,
        intrinsic=intrinsic,
        image_size=image_size,
        global_dense_candidates=global_candidates,
        clean_render_dense_candidates=clean_candidates,
        patch_dense_candidates=patch_candidates,
        policy=APDDensePolicy(),
    )

    assert result["schema"] == "loc_gs_apd_dense_result_v1"
    assert result["uses_gt"] is False
    assert result["dense_refine_success"] is True
    assert result["num_global_candidates"] == 2
    assert result["num_clean_render_candidates"] == 2
    assert result["num_patch_candidates"] == 2
    assert result["num_anchor_residuals"] == points.shape[0]
    assert result["diagnostics"]["candidate_pool"]["source_counts"] == {
        "clean_render": 2,
        "global": 2,
        "patch": 2,
    }
