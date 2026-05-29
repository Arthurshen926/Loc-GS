import json
import subprocess
import sys

import numpy as np

from loc_gs.dense_support.sparse_conditioned_dense_preflight import (
    DenseTransitionPolicy,
    SLCDPThresholds,
    SLCDPRepairSearchConfig,
    apply_camera_frame_delta,
    compute_sparse_ray_gaussian_gating_mask,
    evaluate_dense_step_acceptance,
    evaluate_slcdp_from_summaries,
    generate_fast_landmark_guided_pose_candidates,
    generate_landmark_guided_pose_candidates,
    generate_pose_repair_candidates,
    select_sparse_conditioned_dense_transition,
    score_slcdp_preflight_result,
    select_repaired_pose_candidate,
    compute_sparse_ray_depth_diagnostics,
    sparse_landmark_conditioned_preflight,
    _find_sparse_ray_conflicts_projected,
)


def _identity_camera() -> tuple[np.ndarray, np.ndarray]:
    pose = np.eye(4, dtype=np.float32)
    intrinsic = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return pose, intrinsic


def test_slcdp_accepts_dense_when_render_explains_sparse_inliers():
    pose, intrinsic = _identity_camera()
    points = np.array(
        [
            [-0.2, -0.2, 2.0],
            [0.2, -0.2, 2.0],
            [-0.2, 0.2, 2.0],
            [0.2, 0.2, 2.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )
    query_xy = np.array([[4.0, 4.0], [6.0, 4.0], [4.0, 6.0], [6.0, 6.0], [5.0, 5.0]], dtype=np.float32)
    render_depth = np.full((10, 10), 2.0, dtype=np.float32)
    render_depth[5, 5] = 3.0
    query_features = np.ones((2, 10, 10), dtype=np.float32)
    render_features = np.ones((2, 10, 10), dtype=np.float32)

    result = sparse_landmark_conditioned_preflight(
        sparse_query_xy=query_xy,
        sparse_points_world=points,
        sparse_pose_w2c=pose,
        intrinsic=intrinsic,
        render_depth=render_depth,
        query_features=query_features,
        render_features=render_features,
        sparse_inlier_indices=np.arange(5),
        thresholds=SLCDPThresholds(
            min_sparse_inliers=4,
            min_visible_landmarks=4,
            min_grid_cells=4,
            min_feature_cosine=0.5,
        ),
    )

    assert result["decision"] == "accept_dense"
    assert result["visible_ratio"] == 1.0
    assert result["feature_cosine_median"] > 0.999
    assert result["coverage_grid_cells"] >= 4


def test_slcdp_skips_dense_when_sparse_is_strong_but_render_is_not_explanatory():
    pose, intrinsic = _identity_camera()
    points = np.array([[-0.2, -0.2, 2.0], [0.2, -0.2, 2.0], [-0.2, 0.2, 2.0], [0.2, 0.2, 2.0]], dtype=np.float32)
    query_xy = np.array([[4.0, 4.0], [6.0, 4.0], [4.0, 6.0], [6.0, 6.0]], dtype=np.float32)
    render_depth = np.full((10, 10), 10.0, dtype=np.float32)
    query_features = np.ones((2, 10, 10), dtype=np.float32)
    render_features = -np.ones((2, 10, 10), dtype=np.float32)

    result = sparse_landmark_conditioned_preflight(
        sparse_query_xy=query_xy,
        sparse_points_world=points,
        sparse_pose_w2c=pose,
        intrinsic=intrinsic,
        render_depth=render_depth,
        query_features=query_features,
        render_features=render_features,
        sparse_inlier_indices=np.arange(4),
        thresholds=SLCDPThresholds(min_sparse_inliers=4, min_visible_landmarks=3, min_feature_cosine=0.25),
    )

    assert result["decision"] == "skip_dense_keep_sparse"
    assert result["visible_ratio"] == 0.0
    assert result["failed_checks"]["visibility"] is True
    assert result["failed_checks"]["feature_agreement"] is True


def test_sparse_ray_depth_diagnostics_classifies_near_far_and_missing_depth():
    pose, intrinsic = _identity_camera()
    points = np.array([[-1.0, -1.0, 10.0], [1.0, -1.0, 10.0], [-1.0, 1.0, 10.0], [1.0, 1.0, 10.0]], dtype=np.float32)
    query_xy = np.array([[4.0, 4.0], [6.0, 4.0], [4.0, 6.0], [6.0, 6.0]], dtype=np.float32)
    render_depth = np.full((10, 10), 10.0, dtype=np.float32)
    render_depth[4, 4] = 2.0
    render_depth[4, 6] = 2.0
    render_depth[6, 4] = 20.0
    render_depth[6, 6] = 0.0

    diagnostics = compute_sparse_ray_depth_diagnostics(
        sparse_query_xy=query_xy,
        sparse_points_world=points,
        sparse_pose_w2c=pose,
        intrinsic=intrinsic,
        render_depth=render_depth,
        sparse_inlier_indices=np.arange(4),
        thresholds=SLCDPThresholds(max_depth_abs_error_m=0.25, max_depth_rel_error=0.15),
    )
    result = sparse_landmark_conditioned_preflight(
        sparse_query_xy=query_xy,
        sparse_points_world=points,
        sparse_pose_w2c=pose,
        intrinsic=intrinsic,
        render_depth=render_depth,
        sparse_inlier_indices=np.arange(4),
        thresholds=SLCDPThresholds(min_sparse_inliers=4, min_visible_landmarks=3),
    )

    assert [row["depth_class"] for row in diagnostics] == [
        "near_occluder",
        "near_occluder",
        "far_surface_or_hole",
        "missing_depth",
    ]
    assert result["near_occluder_count"] == 2
    assert result["far_surface_or_hole_count"] == 1
    assert result["missing_depth_count"] == 1
    assert result["near_occluder_ratio"] == 0.5


def test_sparse_ray_gaussian_gating_masks_near_conflicts_only():
    pose, intrinsic = _identity_camera()
    sparse_points = np.array([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0]], dtype=np.float32)
    sparse_query_xy = np.array([[5.0, 5.0], [6.0, 5.0]], dtype=np.float32)
    gaussian_xyz = np.array(
        [
            [0.0, 0.0, 2.0],      # same ray, too near: conflict
            [0.0, 0.0, 10.2],     # same ray, target depth: keep
            [4.0, 4.0, 2.0],      # far in screen space: keep
            [1.0, 0.0, 2.0],      # same ray as sparse point 1 but protected: keep
        ],
        dtype=np.float32,
    )

    result = compute_sparse_ray_gaussian_gating_mask(
        gaussian_xyz=gaussian_xyz,
        sparse_query_xy=sparse_query_xy,
        sparse_points_world=sparse_points,
        sparse_pose_w2c=pose,
        intrinsic=intrinsic,
        sparse_inlier_indices=np.arange(2),
        image_size=(10, 10),
        radius_px=1.5,
        depth_margin_m=1.0,
        protected_gaussian_indices=np.array([3]),
    )

    assert result["conflict_count"] == 1
    assert result["keep_mask"].tolist() == [False, True, True, True]
    assert result["conflict_indices"].tolist() == [0]
    assert result["protected_count"] == 1


def test_sparse_ray_gaussian_gating_without_protected_indices_masks_conflicts():
    pose, intrinsic = _identity_camera()
    sparse_points = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)
    sparse_query_xy = np.array([[5.0, 5.0]], dtype=np.float32)
    gaussian_xyz = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 10.2],
        ],
        dtype=np.float32,
    )

    result = compute_sparse_ray_gaussian_gating_mask(
        gaussian_xyz=gaussian_xyz,
        sparse_query_xy=sparse_query_xy,
        sparse_points_world=sparse_points,
        sparse_pose_w2c=pose,
        intrinsic=intrinsic,
        sparse_inlier_indices=np.array([0]),
        image_size=(10, 10),
        radius_px=1.5,
        depth_margin_m=1.0,
        protected_gaussian_indices=None,
    )

    assert result["protected_count"] == 0
    assert result["conflict_count"] == 1
    assert result["keep_mask"].tolist() == [False, True]


def test_sparse_ray_gaussian_gating_uses_footprint_and_opacity():
    pose, intrinsic = _identity_camera()
    sparse_points = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)
    sparse_query_xy = np.array([[5.0, 5.0]], dtype=np.float32)
    gaussian_xyz = np.array(
        [
            [0.4, 0.0, 2.0],
            [0.4, 0.0, 2.0],
            [0.4, 0.0, 2.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=np.float32,
    )
    gaussian_scale_world = np.array(
        [
            [0.8, 0.8, 0.8],
            [0.8, 0.8, 0.8],
            [0.8, 0.8, 0.8],
            [0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    gaussian_opacity = np.array([0.9, 0.001, 0.9, 0.9], dtype=np.float32)

    result = compute_sparse_ray_gaussian_gating_mask(
        gaussian_xyz=gaussian_xyz,
        gaussian_scale_world=gaussian_scale_world,
        gaussian_opacity=gaussian_opacity,
        sparse_query_xy=sparse_query_xy,
        sparse_points_world=sparse_points,
        sparse_pose_w2c=pose,
        intrinsic=intrinsic,
        sparse_inlier_indices=np.array([0]),
        image_size=(10, 10),
        radius_px=0.5,
        depth_margin_m=1.0,
        footprint_radius_scale=1.0,
        min_opacity=0.01,
        protected_gaussian_indices=np.array([2]),
    )

    assert result["conflict_indices"].tolist() == [0]
    assert result["keep_mask"].tolist() == [False, True, True, True]
    assert result["protected_count"] == 1


def test_sparse_ray_conflict_search_matches_bruteforce_with_variable_footprints():
    gaussian_xy = np.array(
        [
            [5.0, 5.0],
            [5.8, 5.1],
            [7.2, 5.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    gaussian_depth = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    gaussian_valid = np.array([True, True, True, True])
    sparse_xy = np.array([[5.0, 5.0], [7.0, 5.0]], dtype=np.float32)
    sparse_depth = np.array([10.0, 10.0], dtype=np.float32)
    footprint = np.array([0.0, 0.0, 0.5, 0.0], dtype=np.float32)

    conflict_mask, conflicted_rays = _find_sparse_ray_conflicts_projected(
        projected_xy=gaussian_xy,
        gaussian_depth=gaussian_depth,
        gaussian_valid=gaussian_valid,
        sparse_xy=sparse_xy,
        sparse_depth=sparse_depth,
        radius_px=0.75,
        footprint_radius=footprint,
        depth_margin_m=1.0,
    )

    assert conflict_mask.tolist() == [True, False, True, False]
    assert conflicted_rays.tolist() == [True, True]


def test_landmark_guided_pose_candidates_use_sparse_landmark_geometry():
    pose, _intrinsic = _identity_camera()
    sparse_points = np.array([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [-1.0, 0.0, 10.0]], dtype=np.float32)

    candidates = generate_landmark_guided_pose_candidates(
        pose,
        sparse_points_world=sparse_points,
        steps_m=(1.0, 2.0),
        max_candidates=9,
    )

    labels = {candidate["label"] for candidate in candidates}
    assert "base" in labels
    assert "ray_centroid+1.000" in labels
    assert "ray_centroid-1.000" in labels
    forward = next(candidate for candidate in candidates if candidate["label"] == "ray_centroid+1.000")
    assert np.allclose(forward["translation_cam_m"], (0.0, 0.0, 1.0), atol=1e-6)
    assert np.allclose(forward["pose_w2c"][:3, 3], [0.0, 0.0, -1.0], atol=1e-6)


def test_fast_landmark_guided_pose_candidates_preselect_geometry_without_rendering():
    pose, intrinsic = _identity_camera()
    sparse_points = np.array(
        [
            [0.0, 0.0, 10.0],
            [0.5, 0.0, 10.0],
            [-0.5, 0.0, 10.0],
        ],
        dtype=np.float32,
    )
    gaussian_xyz = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.5, 0.0, 2.0],
            [-0.5, 0.0, 2.0],
            [4.0, 4.0, 2.0],
        ],
        dtype=np.float32,
    )

    candidates = generate_fast_landmark_guided_pose_candidates(
        pose,
        sparse_points_world=sparse_points,
        gaussian_xyz=gaussian_xyz,
        intrinsic=intrinsic,
        image_size=(10, 10),
        steps_m=(1.0,),
        max_render_candidates=3,
        conflict_radius_px=1.5,
        depth_margin_m=1.0,
    )

    assert 1 < len(candidates) <= 3
    assert candidates[0]["label"] == "base"
    assert all(candidate["selection_mode"] == "fast_guided_pose" for candidate in candidates)
    base_conflicts = candidates[0]["fast_score"]["estimated_conflict_count"]
    best_conflicts = min(candidate["fast_score"]["estimated_conflict_count"] for candidate in candidates[1:])
    assert best_conflicts < base_conflicts
    assert [candidate["render_rank"] for candidate in candidates] == list(range(len(candidates)))


def test_fast_guided_pose_score_uses_deterministic_gaussian_cap(monkeypatch):
    from loc_gs.dense_support import sparse_conditioned_dense_preflight as slcdp

    observed_counts = []

    def fake_estimate(**kwargs):
        observed_counts.append(int(np.asarray(kwargs["gaussian_xyz"]).shape[0]))
        return {
            "estimated_conflict_count": 1,
            "estimated_conflicted_ray_count": 1,
            "estimated_sparse_ray_count": 2,
            "estimated_visible_sparse_count": 2,
            "estimated_coverage_grid_cells": 2,
            "conflict_radius_px": 1.0,
            "depth_margin_m": 0.0,
            "footprint_radius_scale": 0.0,
            "max_footprint_radius_px": 8.0,
        }

    monkeypatch.setattr(slcdp, "_estimate_sparse_ray_conflicts_for_pose", fake_estimate)

    candidates = slcdp.generate_fast_landmark_guided_pose_candidates(
        np.eye(4, dtype=np.float32),
        sparse_points_world=np.array([[0.0, 0.0, 4.0], [1.0, 0.0, 4.0]], dtype=np.float32),
        gaussian_xyz=np.stack(
            [np.linspace(-2.0, 2.0, 101), np.zeros(101), np.ones(101) * 2.0],
            axis=1,
        ).astype(np.float32),
        gaussian_scale_world=np.ones((101, 3), dtype=np.float32) * 0.1,
        intrinsic=np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        image_size=(64, 64),
        steps_m=(0.5, 1.0),
        pool_max_candidates=9,
        max_render_candidates=3,
        score_max_gaussians=11,
    )

    assert candidates
    assert observed_counts
    assert max(observed_counts) <= 11
    assert all(candidate["fast_pool_candidate_count"] == len(observed_counts) for candidate in candidates)


def test_fast_landmark_guided_candidate_subset_keeps_axis_diversity():
    from loc_gs.dense_support.sparse_conditioned_dense_preflight import _select_fast_guided_candidate_subset

    def item(label: str, axis: str, scalar: float) -> dict[str, object]:
        return {
            "label": label,
            "guided_axis": axis,
            "fast_score": {"scalar": scalar, "translation_norm_m": 1.0},
        }

    selected = _select_fast_guided_candidate_subset(
        [
            item("base", "base", 0.0),
            item("ray_side-5.000", "ray_side", 10.0),
            item("ray_side-2.000", "ray_side", 9.0),
            item("ray_centroid+5.000", "ray_centroid", 8.0),
            item("ray_up-5.000", "ray_up", 7.0),
        ],
        max_render_candidates=4,
    )

    assert [candidate["label"] for candidate in selected] == [
        "base",
        "ray_side-5.000",
        "ray_centroid+5.000",
        "ray_up-5.000",
    ]
    assert [candidate["render_rank"] for candidate in selected] == [0, 1, 2, 3]


def test_dense_step_acceptance_rejects_update_that_breaks_sparse_inliers():
    sparse_pose, intrinsic = _identity_camera()
    dense_pose = sparse_pose.copy()
    dense_pose[0, 3] = -5.0
    points = np.array([[-0.2, -0.2, 2.0], [0.2, -0.2, 2.0], [-0.2, 0.2, 2.0], [0.2, 0.2, 2.0]], dtype=np.float32)
    query_xy = np.array([[4.0, 4.0], [6.0, 4.0], [4.0, 6.0], [6.0, 6.0]], dtype=np.float32)

    result = evaluate_dense_step_acceptance(
        sparse_query_xy=query_xy,
        sparse_points_world=points,
        sparse_pose_w2c=sparse_pose,
        dense_pose_w2c=dense_pose,
        intrinsic=intrinsic,
        sparse_inlier_indices=np.arange(4),
        max_reprojection_error_px=5.0,
        min_retained_ratio=0.75,
        max_translation_delta_m=0.5,
    )

    assert result["decision"] == "reject_dense_update_keep_sparse"
    assert result["retained_sparse_inlier_ratio"] == 0.0
    assert result["translation_delta_m"] > 0.5


def test_dense_transition_rejects_large_update_even_when_sparse_inliers_are_retained():
    sparse_pose, intrinsic = _identity_camera()
    dense_pose = sparse_pose.copy()
    dense_pose[0, 3] = -0.6
    xs = np.linspace(-2.0, 2.0, 5)
    ys = np.linspace(-2.0, 2.0, 5)
    points = np.array([[x, y, 20.0] for y in ys for x in xs], dtype=np.float32)
    query_xy = np.array([[10.0 * x / 20.0 + 5.0, 10.0 * y / 20.0 + 5.0] for y in ys for x in xs], dtype=np.float32)

    decision = select_sparse_conditioned_dense_transition(
        sparse_query_xy=query_xy,
        sparse_points_world=points,
        sparse_pose_w2c=sparse_pose,
        dense_pose_w2c=dense_pose,
        intrinsic=intrinsic,
        sparse_inlier_indices=np.arange(points.shape[0]),
        policy=DenseTransitionPolicy(
            max_reprojection_error_px=5.0,
            min_retained_ratio=0.95,
            max_translation_delta_m=0.25,
            max_rotation_delta_deg=5.0,
            line_search_fractions=(1.0, 0.5, 0.0),
            strong_sparse_inlier_count=10,
        ),
        image_size=(10, 10),
    )

    assert decision["decision"] == "reject_dense_keep_sparse"
    assert decision["selected_fraction"] == 0.0
    assert np.allclose(decision["selected_pose_w2c"], sparse_pose)
    assert decision["candidates"][0]["acceptance"]["retained_sparse_inlier_ratio"] == 1.0


def test_dense_transition_uses_largest_sparse_consistent_line_search_step():
    sparse_pose, intrinsic = _identity_camera()
    dense_pose = sparse_pose.copy()
    dense_pose[0, 3] = -0.6
    xs = np.linspace(-2.0, 2.0, 5)
    ys = np.linspace(-2.0, 2.0, 5)
    points = np.array([[x, y, 20.0] for y in ys for x in xs], dtype=np.float32)
    query_xy = np.array([[10.0 * x / 20.0 + 5.0, 10.0 * y / 20.0 + 5.0] for y in ys for x in xs], dtype=np.float32)

    decision = select_sparse_conditioned_dense_transition(
        sparse_query_xy=query_xy,
        sparse_points_world=points,
        sparse_pose_w2c=sparse_pose,
        dense_pose_w2c=dense_pose,
        intrinsic=intrinsic,
        sparse_inlier_indices=np.arange(points.shape[0]),
        policy=DenseTransitionPolicy(
            max_reprojection_error_px=5.0,
            min_retained_ratio=0.95,
            max_translation_delta_m=0.35,
            max_rotation_delta_deg=5.0,
            line_search_fractions=(1.0, 0.5, 0.0),
            strong_sparse_inlier_count=10,
        ),
        image_size=(10, 10),
    )

    assert decision["decision"] == "accept_line_search_dense_update"
    assert decision["selected_fraction"] == 0.5
    assert np.isclose(decision["selected_acceptance"]["translation_delta_m"], 0.3, atol=1e-6)
    selected_pose = np.asarray(decision["selected_pose_w2c"], dtype=np.float64)
    selected_center = -selected_pose[:3, :3].T @ selected_pose[:3, 3]
    assert np.allclose(selected_center, [0.3, 0.0, 0.0], atol=1e-6)


def test_dense_transition_allows_large_update_when_sparse_support_is_weak():
    sparse_pose, intrinsic = _identity_camera()
    dense_pose = sparse_pose.copy()
    dense_pose[0, 3] = -0.6
    points = np.array([[-1.0, -1.0, 20.0], [1.0, -1.0, 20.0], [-1.0, 1.0, 20.0], [1.0, 1.0, 20.0]], dtype=np.float32)
    query_xy = np.array([[4.5, 4.5], [5.5, 4.5], [4.5, 5.5], [5.5, 5.5]], dtype=np.float32)

    decision = select_sparse_conditioned_dense_transition(
        sparse_query_xy=query_xy,
        sparse_points_world=points,
        sparse_pose_w2c=sparse_pose,
        dense_pose_w2c=dense_pose,
        intrinsic=intrinsic,
        sparse_inlier_indices=np.arange(points.shape[0]),
        policy=DenseTransitionPolicy(
            max_reprojection_error_px=5.0,
            min_retained_ratio=0.95,
            weak_min_retained_ratio=0.50,
            max_translation_delta_m=0.25,
            weak_max_translation_delta_m=1.0,
            line_search_fractions=(1.0, 0.0),
            strong_sparse_inlier_count=10,
        ),
        image_size=(10, 10),
    )

    assert decision["sparse_support_strength"] == "weak"
    assert decision["decision"] == "accept_dense_update"
    assert decision["selected_fraction"] == 1.0


def test_summary_preflight_flags_kings239_like_sparse_good_dense_bad_case():
    match_summary = {
        "captured_sparse_te_cm": 20.221,
        "captured_sparse_re_deg": 0.125,
        "captured_dense_te_cm": 7141.77,
        "captured_dense_re_deg": 163.38,
        "sparse": {"solver_inlier_count": 127, "solver_inlier_ratio": 0.062},
        "dense": {"match_count": 67, "solver_inlier_count": 15},
    }
    feature_summary = {
        "feature": {
            "same_pixel_cosine": {"p50": 0.0207, "mean": 0.0245},
            "query_to_render_max_cosine": {"mean": 0.219},
            "coarse_mnn_count": 44,
        }
    }

    result = evaluate_slcdp_from_summaries(match_summary, feature_summary)

    assert result["decision"] == "skip_dense_keep_sparse"
    assert result["accepted_final_te_cm"] == 20.221
    assert result["dense_te_reduction_cm"] > 7000.0
    assert "feature_agreement" in result["reasons"]


def test_slcdp_cli_writes_diagnostic_artifact(tmp_path):
    match_summary = tmp_path / "match_summary.json"
    feature_summary = tmp_path / "feature_summary.json"
    output_dir = tmp_path / "slcdp"
    match_summary.write_text(
        json.dumps(
            {
                "scene": "KingsCollege",
                "query_index": 239,
                "image_name": "seq3/frame00179.png",
                "captured_sparse_te_cm": 20.221,
                "captured_sparse_re_deg": 0.125,
                "captured_dense_te_cm": 7141.77,
                "captured_dense_re_deg": 163.38,
                "sparse": {"solver_inlier_count": 127, "solver_inlier_ratio": 0.062},
                "dense": {"match_count": 67, "solver_inlier_count": 15},
            }
        ),
        encoding="utf-8",
    )
    feature_summary.write_text(
        json.dumps(
            {
                "feature": {
                    "same_pixel_cosine": {"p50": 0.0207, "mean": 0.0245},
                    "query_to_render_max_cosine": {"mean": 0.219},
                    "coarse_mnn_count": 44,
                }
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_sparse_conditioned_dense_preflight",
            "--match_summary",
            str(match_summary),
            "--feature_summary",
            str(feature_summary),
            "--scene",
            "KingsCollege",
            "--split_name",
            "test",
            "--output_dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output_dir / "slcdp_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert summary["cases"][0]["decision"] == "skip_dense_keep_sparse"
    assert metrics["skip_dense_keep_sparse_count"] == 1
    assert metrics["mean_dense_te_reduction_cm"] > 7000.0
    assert manifest["diagnostic_only"] is True
    assert manifest["paper_safe_for_tuning"] is False


def test_apply_camera_frame_delta_moves_camera_center_in_local_axes():
    pose, _intrinsic = _identity_camera()

    shifted = apply_camera_frame_delta(pose, translation_cam_m=(0.1, -0.2, 0.3), rotation_deg=(0.0, 0.0, 0.0))

    assert np.allclose(shifted[:3, :3], np.eye(3), atol=1e-6)
    assert np.allclose(shifted[:3, 3], [-0.1, 0.2, -0.3], atol=1e-6)


def test_generate_pose_repair_candidates_includes_base_and_axis_offsets():
    pose, _intrinsic = _identity_camera()

    candidates = generate_pose_repair_candidates(
        pose,
        SLCDPRepairSearchConfig(translation_steps_m=(0.05,), rotation_steps_deg=(1.0,)),
    )

    labels = {candidate["label"] for candidate in candidates}
    assert "base" in labels
    assert "tx+0.050" in labels
    assert "tz-0.050" in labels
    assert "yaw+1.000" in labels
    assert len(candidates) == 13


def test_select_repaired_pose_candidate_prefers_visible_feature_consistent_pose():
    pose, _intrinsic = _identity_camera()
    worse_pose = apply_camera_frame_delta(pose, translation_cam_m=(0.1, 0.0, 0.0), rotation_deg=(0.0, 0.0, 0.0))
    better_pose = apply_camera_frame_delta(pose, translation_cam_m=(0.0, 0.0, -0.1), rotation_deg=(0.0, 0.0, 0.0))
    evaluations = [
        {
            "label": "base",
            "pose_w2c": pose,
            "preflight": {
                "decision": "skip_dense_keep_sparse",
                "visible_ratio": 0.0,
                "visible_count": 0,
                "coverage_grid_cells": 0,
                "feature_cosine_median": 0.0,
                "failed_checks": {"visibility": True, "feature_agreement": True},
            },
        },
        {
            "label": "tx+0.100",
            "pose_w2c": worse_pose,
            "preflight": {
                "decision": "retry_sparse_or_patch_dense",
                "visible_ratio": 0.4,
                "visible_count": 8,
                "coverage_grid_cells": 2,
                "feature_cosine_median": 0.05,
                "failed_checks": {"coverage": True},
            },
        },
        {
            "label": "tz-0.100",
            "pose_w2c": better_pose,
            "preflight": {
                "decision": "accept_dense",
                "visible_ratio": 0.8,
                "visible_count": 24,
                "coverage_grid_cells": 6,
                "feature_cosine_median": 0.25,
                "failed_checks": {},
            },
        },
    ]

    selected = select_repaired_pose_candidate(evaluations, require_accept=True)

    assert selected["decision"] == "accept_repaired_dense_pose"
    assert selected["selected_label"] == "tz-0.100"
    assert score_slcdp_preflight_result(evaluations[2]["preflight"]) > score_slcdp_preflight_result(evaluations[0]["preflight"])


def test_select_repaired_pose_candidate_can_penalize_risky_large_translation():
    evaluations = [
        {
            "label": "base",
            "translation_cam_m": (0.0, 0.0, 0.0),
            "preflight": {
                "decision": "accept_dense",
                "visible_ratio": 0.84,
                "visible_count": 27,
                "coverage_grid_cells": 6,
                "feature_cosine_median": 0.59,
                "failed_checks": {},
            },
        },
        {
            "label": "ray_up-5.000",
            "translation_cam_m": (0.0, -5.0, 0.0),
            "preflight": {
                "decision": "accept_dense",
                "visible_ratio": 1.0,
                "visible_count": 32,
                "coverage_grid_cells": 7,
                "feature_cosine_median": 0.60,
                "failed_checks": {},
            },
        },
    ]

    unpenalized = select_repaired_pose_candidate(
        evaluations,
        require_accept=True,
        translation_penalty_per_m=0.0,
    )
    penalized = select_repaired_pose_candidate(
        evaluations,
        require_accept=True,
        translation_penalty_per_m=0.12,
    )

    assert unpenalized["selected_label"] == "ray_up-5.000"
    assert penalized["decision"] == "accept_original_dense_pose"
    assert penalized["selected_label"] == "base"
    assert penalized["candidate_scores"][1]["translation_penalty"] == 0.6


def test_select_repaired_pose_candidate_returns_base_when_gain_is_too_small():
    evaluations = [
        {
            "label": "base",
            "translation_cam_m": (0.0, 0.0, 0.0),
            "preflight": {
                "decision": "accept_dense",
                "visible_ratio": 0.8,
                "visible_count": 20,
                "coverage_grid_cells": 6,
                "feature_cosine_median": 0.5,
                "failed_checks": {},
            },
        },
        {
            "label": "gated_base",
            "translation_cam_m": (0.0, 0.0, 0.0),
            "preflight": {
                "decision": "accept_dense",
                "visible_ratio": 0.85,
                "visible_count": 22,
                "coverage_grid_cells": 6,
                "feature_cosine_median": 0.5,
                "failed_checks": {},
            },
        },
    ]

    selected = select_repaired_pose_candidate(evaluations, require_accept=True, min_score_gain=2.0)

    assert selected["decision"] == "accept_original_dense_pose"
    assert selected["selected_label"] == "base"


def test_select_repaired_pose_candidate_allows_low_confidence_gated_base():
    evaluations = [
        {
            "label": "base",
            "translation_cam_m": (0.0, 0.0, 0.0),
            "preflight": {
                "decision": "accept_dense",
                "sparse_confident": False,
                "near_occluder_count": 6,
                "visible_ratio": 0.80,
                "visible_count": 26,
                "coverage_grid_cells": 5,
                "feature_cosine_median": 0.6,
                "failed_checks": {},
            },
        },
        {
            "label": "gated_base",
            "translation_cam_m": (0.0, 0.0, 0.0),
            "preflight": {
                "decision": "accept_dense",
                "sparse_confident": False,
                "near_occluder_count": 0,
                "visible_ratio": 1.0,
                "visible_count": 32,
                "coverage_grid_cells": 7,
                "feature_cosine_median": 0.52,
                "failed_checks": {},
            },
        },
    ]

    selected = select_repaired_pose_candidate(
        evaluations,
        require_accept=True,
        min_score_gain=2.0,
        allow_low_confidence_gated_base=True,
        low_confidence_gated_base_min_score_gain=0.25,
    )

    assert selected["decision"] == "accept_repaired_dense_pose"
    assert selected["selected_label"] == "gated_base"
    assert selected["low_confidence_gated_base"] is True
