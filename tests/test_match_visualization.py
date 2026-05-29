import inspect

import numpy as np
import pytest
from PIL import Image

from loc_gs.scripts import visualize_stdloc_hard_matches as hard_match_viz
from loc_gs.scripts import visualize_stdloc_feature_diagnostics as feature_diag
from loc_gs.diagnostics.match_visualization import (
    _select_draw_indices,
    cosine_similarity_map,
    draw_match_canvas,
    feature_norm_map,
    feature_pair_pca_rgb,
    offset_camera_center,
    pose_error_cm_deg,
    project_points,
    summarize_match_quality,
)


def test_project_points_uses_w2c_pose_and_intrinsics():
    points = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]], dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    intrinsic = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    projected, valid = project_points(points, pose, intrinsic, width=120, height=100)

    assert valid.tolist() == [True, True]
    np.testing.assert_allclose(projected, np.array([[50.0, 40.0], [100.0, 40.0]], dtype=np.float32))


def test_summarize_match_quality_counts_gt_good_and_solver_inliers():
    summary = summarize_match_quality(
        reprojection_errors_px=np.array([1.0, 3.0, 20.0, np.inf]),
        solver_inlier_mask=np.array([True, False, True, False]),
        good_px_threshold=5.0,
    )

    assert summary["match_count"] == 4
    assert summary["gt_good_count"] == 2
    assert summary["solver_inlier_count"] == 2
    assert summary["solver_inlier_gt_good_count"] == 1
    assert summary["gt_good_ratio"] == 0.5


def test_draw_match_canvas_writes_side_by_side_lines(tmp_path):
    query = Image.new("RGB", (16, 16), "white")
    reference = Image.new("RGB", (16, 16), "gray")
    output = tmp_path / "matches.jpg"

    draw_match_canvas(
        query,
        reference,
        query_xy=np.array([[2.0, 2.0], [10.0, 10.0]], dtype=np.float32),
        reference_xy=np.array([[3.0, 2.0], [12.0, 11.0]], dtype=np.float32),
        gt_good_mask=np.array([True, False]),
        solver_inlier_mask=np.array([True, False]),
        output_path=output,
        max_draw=10,
    )

    assert output.exists()
    assert Image.open(output).size[0] == 32


def test_select_draw_indices_is_class_balanced_not_prefix_only():
    good = np.array([False] * 80 + [True] * 20, dtype=bool)
    inliers = np.array([False] * 80 + [True] * 20, dtype=bool)
    query_xy = np.stack(
        [np.arange(100, dtype=np.float32), np.arange(100, dtype=np.float32)],
        axis=1,
    )

    selected = _select_draw_indices(good, inliers, query_xy, max_draw=20)

    assert len(selected) == 20
    assert selected.max() >= 80
    assert good[selected].any()
    assert (~good[selected]).any()


def test_pose_error_cm_deg_uses_camera_centers_and_relative_rotation():
    gt = np.eye(4, dtype=np.float32)
    pred = np.eye(4, dtype=np.float32)
    pred[:3, 3] = np.array([-0.1, 0.0, 0.0], dtype=np.float32)

    te_cm, re_deg = pose_error_cm_deg(pred, gt)

    assert te_cm == pytest.approx(10.0)
    assert re_deg == pytest.approx(0.0)


def test_dense_capture_uses_native_coarse_query_feature_map():
    assert "query_coarse" in hard_match_viz._capture_dense.__annotations__


def test_hard_match_parser_accepts_slcdp_transition_control_options():
    args = hard_match_viz.build_argparser().parse_args(
        [
            "--slcdp_render_control",
            "fast_guided_pose",
            "--slcdp_fast_guided_max_render_candidates",
            "3",
            "--slcdp_transition_control",
            "--slcdp_transition_max_translation_delta_m",
            "0.25",
            "--slcdp_transition_line_search_fractions",
            "1.0,0.5,0.0",
        ]
    )

    assert args.slcdp_transition_control is True
    assert args.slcdp_render_control == "fast_guided_pose"
    assert args.slcdp_fast_guided_max_render_candidates == 3
    assert args.slcdp_transition_max_translation_delta_m == pytest.approx(0.25)
    assert args.slcdp_transition_line_search_fractions == "1.0,0.5,0.0"


def test_sparse_conditioned_render_control_resolves_fixed_fusion_recipe():
    args = hard_match_viz.build_argparser().parse_args(
        [
            "--slcdp_render_control",
            "sparse_conditioned",
            "--slcdp_gating_radius_px",
            "999",
            "--slcdp_transition_min_retained_ratio",
            "0.1",
        ]
    )

    options = hard_match_viz._resolve_slcdp_effective_options(args)

    assert options["slcdp_render_control"] == "sparse_conditioned"
    assert options["slcdp_repair_require_accept"] is True
    assert options["slcdp_repair_skip_if_no_accept"] is True
    assert options["slcdp_transition_control"] is True
    assert options["slcdp_repair_min_score_gain"] == pytest.approx(2.0)
    assert options["slcdp_gating_radius_px"] == pytest.approx(6.0)
    assert options["slcdp_gating_depth_margin_m"] == pytest.approx(1.0)
    assert options["slcdp_gating_footprint_radius_scale"] == pytest.approx(1.5)
    assert options["slcdp_gating_min_footprint_radius_px"] == pytest.approx(12.0)
    assert options["slcdp_gating_min_depth_m"] == pytest.approx(5.0)
    assert options["slcdp_gating_max_depth_m"] == pytest.approx(60.0)
    assert options["slcdp_repair_translation_penalty_per_m"] == pytest.approx(0.75)
    assert options["slcdp_allow_low_confidence_gated_base"] is False
    assert options["slcdp_low_confidence_gated_base_min_score_gain"] == pytest.approx(0.25)
    assert options["slcdp_transition_min_retained_ratio"] == pytest.approx(0.875)


def test_sparse_conditioned_effective_options_match_analyze_case_signature():
    args = hard_match_viz.build_argparser().parse_args(["--slcdp_render_control", "sparse_conditioned"])
    options = hard_match_viz._resolve_slcdp_effective_options(args)
    signature = inspect.signature(hard_match_viz._analyze_case)

    assert set(options).issubset(set(signature.parameters))


def test_sparse_conditioned_base_candidate_does_not_apply_gating():
    assert (
        hard_match_viz._should_apply_slcdp_gating(
            render_control_mode="sparse_conditioned",
            behavior_render_control_mode="fast_gating_guided_pose",
            candidate={"label": "base"},
        )
        is False
    )
    assert (
        hard_match_viz._should_apply_slcdp_gating(
            render_control_mode="sparse_conditioned",
            behavior_render_control_mode="fast_gating_guided_pose",
            candidate={"label": "gated_base", "slcdp_apply_gating": True},
        )
        is True
    )


def test_sparse_conditioned_base_fast_path_requires_safe_preflight():
    safe = {
        "decision": "accept_dense",
        "sparse_confident": True,
        "visible_ratio": 0.9,
        "near_occluder_ratio": 0.05,
        "feature_cosine_median": 0.6,
        "coverage_grid_cells": 5,
    }
    risky = dict(safe, visible_ratio=0.0, near_occluder_ratio=1.0)

    assert hard_match_viz._should_accept_sparse_conditioned_base_fast_path(safe) is True
    assert hard_match_viz._should_accept_sparse_conditioned_base_fast_path(dict(safe, feature_cosine_median=0.10)) is True
    assert hard_match_viz._should_accept_sparse_conditioned_base_fast_path(risky) is False
    assert hard_match_viz._should_accept_sparse_conditioned_base_fast_path(dict(safe, sparse_confident=False)) is False
    assert (
        hard_match_viz._should_apply_slcdp_gating(
            render_control_mode="fast_gating_guided_pose",
            behavior_render_control_mode="fast_gating_guided_pose",
            candidate={"label": "base"},
        )
        is True
    )


def test_sparse_conditioned_reuses_high_confidence_base_dense_despite_ray_preflight_failure():
    risky_preflight = {
        "decision": "skip_dense_keep_sparse",
        "sparse_confident": True,
        "visible_ratio": 0.0,
        "near_occluder_ratio": 1.0,
        "coverage_grid_cells": 4,
    }
    high_confidence_dense = {
        "dense_pose_quality": {
            "match_count": 16000,
            "solver_inlier_count": 15000,
            "solver_inlier_ratio": 0.94,
            "median_reprojection_error_px": 1.4,
            "p90_reprojection_error_px": 3.0,
        }
    }

    decision = hard_match_viz._should_reuse_sparse_conditioned_base_dense(
        risky_preflight,
        high_confidence_dense,
    )

    assert decision["reuse_base"] is True
    assert decision["reason"] == "base_dense_high_confidence"


def test_sparse_conditioned_does_not_reuse_low_confidence_base_dense_on_bad_preflight():
    risky_preflight = {
        "decision": "skip_dense_keep_sparse",
        "sparse_confident": True,
        "visible_ratio": 0.0,
        "near_occluder_ratio": 1.0,
        "coverage_grid_cells": 4,
    }
    low_confidence_dense = {
        "dense_pose_quality": {
            "match_count": 200,
            "solver_inlier_count": 30,
            "solver_inlier_ratio": 0.15,
            "median_reprojection_error_px": 180.0,
            "p90_reprojection_error_px": 360.0,
        }
    }

    decision = hard_match_viz._should_reuse_sparse_conditioned_base_dense(
        risky_preflight,
        low_confidence_dense,
    )

    assert decision["reuse_base"] is False
    assert decision["reason"] == "base_preflight_requires_repair"


def test_sparse_conditioned_rejects_repair_when_it_degrades_dense_quality_tail():
    base_quality = {
        "match_count": 7400,
        "solver_inlier_count": 6000,
        "solver_inlier_ratio": 0.81,
        "median_reprojection_error_px": 1.0,
        "p90_reprojection_error_px": 100.0,
    }
    repair_quality = {
        "match_count": 8500,
        "solver_inlier_count": 6700,
        "solver_inlier_ratio": 0.78,
        "median_reprojection_error_px": 0.9,
        "p90_reprojection_error_px": 175.0,
    }

    decision = hard_match_viz._should_reject_sparse_conditioned_repair_for_dense_quality_regression(
        base_quality,
        repair_quality,
    )

    assert decision["reject_repair"] is True
    assert decision["reason"] == "repair_degrades_dense_quality_tail"


def test_sparse_conditioned_rejects_repair_when_it_damages_coherent_dense_core():
    base_quality = {
        "match_count": 5320,
        "solver_inlier_count": 3828,
        "solver_inlier_ratio": 0.7195,
        "median_reprojection_error_px": 0.90,
        "p90_reprojection_error_px": 249.4,
    }
    repair_quality = {
        "match_count": 5313,
        "solver_inlier_count": 1865,
        "solver_inlier_ratio": 0.3510,
        "median_reprojection_error_px": 62.9,
        "p90_reprojection_error_px": 313.4,
    }

    decision = hard_match_viz._should_reject_sparse_conditioned_repair_for_dense_quality_regression(
        base_quality,
        repair_quality,
    )

    assert decision["reject_repair"] is True
    assert decision["reason"] == "repair_degrades_dense_quality_core"


def test_sparse_conditioned_does_not_protect_bad_dense_core():
    base_quality = {
        "match_count": 3605,
        "solver_inlier_count": 689,
        "solver_inlier_ratio": 0.191,
        "median_reprojection_error_px": 192.9,
        "p90_reprojection_error_px": 435.3,
    }
    repair_quality = {
        "match_count": 4378,
        "solver_inlier_count": 706,
        "solver_inlier_ratio": 0.161,
        "median_reprojection_error_px": 172.1,
        "p90_reprojection_error_px": 422.8,
    }

    decision = hard_match_viz._should_reject_sparse_conditioned_repair_for_dense_quality_regression(
        base_quality,
        repair_quality,
    )

    assert decision["reject_repair"] is False
    assert decision["reason"] == "base_dense_quality_not_protective"


def test_sparse_conditioned_rejects_gating_only_repair_with_weak_sparse_support():
    sparse_capture = {
        "inliers": np.arange(26, dtype=np.int64),
        "p3d": np.zeros((30, 3), dtype=np.float32),
    }

    decision = hard_match_viz._should_reject_gating_only_repair_for_weak_sparse_support(
        selected_label="gated_base",
        sparse_capture=sparse_capture,
    )

    assert decision["reject_repair"] is True
    assert decision["reason"] == "weak_sparse_support_gating_only"
    assert decision["sparse_inlier_count"] == 26


def test_sparse_conditioned_allows_gating_only_repair_with_strong_sparse_support():
    sparse_capture = {
        "inliers": np.arange(96, dtype=np.int64),
        "p3d": np.zeros((96, 3), dtype=np.float32),
    }

    decision = hard_match_viz._should_reject_gating_only_repair_for_weak_sparse_support(
        selected_label="gated_base",
        sparse_capture=sparse_capture,
    )

    assert decision["reject_repair"] is False
    assert decision["reason"] == "sparse_support_sufficient_for_gating_only"


def test_sparse_conditioned_weak_sparse_guard_does_not_block_guided_pose():
    sparse_capture = {
        "inliers": np.arange(20, dtype=np.int64),
        "p3d": np.zeros((20, 3), dtype=np.float32),
    }

    decision = hard_match_viz._should_reject_gating_only_repair_for_weak_sparse_support(
        selected_label="gated_ray_centroid+5.000",
        sparse_capture=sparse_capture,
    )

    assert decision["reject_repair"] is False
    assert decision["reason"] == "not_gating_only"


def test_sparse_conditioned_keeps_repair_when_base_dense_quality_is_already_bad():
    base_quality = {
        "match_count": 5000,
        "solver_inlier_count": 2300,
        "solver_inlier_ratio": 0.46,
        "median_reprojection_error_px": 25.0,
        "p90_reprojection_error_px": 360.0,
    }
    repair_quality = {
        "match_count": 4700,
        "solver_inlier_count": 1600,
        "solver_inlier_ratio": 0.34,
        "median_reprojection_error_px": 80.0,
        "p90_reprojection_error_px": 410.0,
    }

    decision = hard_match_viz._should_reject_sparse_conditioned_repair_for_dense_quality_regression(
        base_quality,
        repair_quality,
    )

    assert decision["reject_repair"] is False
    assert decision["reason"] == "base_dense_quality_not_protective"


def test_sparse_conditioned_base_reuse_preserves_pose_and_adds_audit_metadata():
    base_dense = {
        "pose_w2c": np.eye(4, dtype=np.float32),
        "inliers": np.array([1, 2, 3], dtype=np.int32),
        "slcdp_render_control": {"mode": "none"},
    }
    base_rendered = {
        "preflight": {"decision": "accept_dense"},
        "render_control": {"mode": "sparse_conditioned", "candidate_label": "base"},
    }

    reused = hard_match_viz._reuse_sparse_conditioned_base_dense_capture(base_dense, base_rendered)

    assert reused is not base_dense
    np.testing.assert_allclose(reused["pose_w2c"], base_dense["pose_w2c"])
    assert reused["inliers"].tolist() == [1, 2, 3]
    assert reused["slcdp_render_control"]["candidate_label"] == "base"
    assert reused["slcdp_preflight"]["decision"] == "accept_dense"
    assert reused["slcdp_repair_search"]["selection"]["decision"] == "accept_original_dense_pose"
    assert reused["slcdp_repair_search"]["dense_reused_from_base"] is True


def test_transition_control_only_runs_for_non_base_repair_candidates():
    assert (
        hard_match_viz._should_apply_sparse_conditioned_transition_control(
            selected_label="base",
            repair_selection={"decision": "accept_original_dense_pose"},
        )
        is False
    )
    assert (
        hard_match_viz._should_apply_sparse_conditioned_transition_control(
            selected_label="base",
            repair_selection={"decision": "skip_dense_keep_sparse"},
        )
        is False
    )
    assert (
        hard_match_viz._should_apply_sparse_conditioned_transition_control(
            selected_label="gated_base",
            repair_selection={"decision": "accept_repaired_dense_pose"},
        )
        is True
    )
    assert (
        hard_match_viz._should_apply_sparse_conditioned_transition_control(
            selected_label="gated_ray_forward+2.000",
            repair_selection={"decision": "accept_repaired_dense_pose"},
        )
        is True
    )


def test_sparse_conditioned_rejects_low_quality_dense_repair_when_sparse_is_strong():
    sparse_capture = {
        "query_xy": np.zeros((200, 2), dtype=np.float32),
        "p3d": np.zeros((200, 3), dtype=np.float32),
        "inliers": np.arange(127, dtype=np.int32),
    }
    dense_quality = {
        "match_count": 5931,
        "solver_inlier_count": 2806,
        "solver_inlier_ratio": 0.473,
        "median_reprojection_error_px": 24.0,
        "p90_reprojection_error_px": 273.0,
    }

    decision = hard_match_viz._should_reject_sparse_conditioned_dense_for_low_quality(
        selected_label="gated_ray_up+2.000",
        repair_selection={"decision": "accept_repaired_dense_pose"},
        sparse_capture=sparse_capture,
        dense_quality=dense_quality,
    )

    assert decision["reject_dense"] is True
    assert decision["reason"] == "strong_sparse_low_quality_dense_repair"


def test_sparse_conditioned_reuses_base_dense_when_no_repair_is_accepted():
    base_dense = {
        "pose_w2c": np.eye(4, dtype=np.float32),
        "query_xy": np.array([[1.0, 2.0]], dtype=np.float32),
        "rendered_xy": np.array([[3.0, 4.0]], dtype=np.float32),
        "p3d": np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        "inliers": np.array([0], dtype=np.int32),
        "K": np.eye(3, dtype=np.float32),
    }
    base_rendered = {
        "label": "base",
        "preflight": {"decision": "skip_dense_keep_sparse"},
        "render_control": {"mode": "sparse_conditioned", "candidate_label": "base"},
    }
    repair_selection = {
        "schema": "loc_gs_slcdp_repair_selection_v1",
        "decision": "skip_dense_keep_sparse",
        "selected_label": "base",
        "best_label": "base",
        "score_gain": 0.0,
        "candidate_count": 1,
    }

    reused = hard_match_viz._reuse_base_dense_after_rejected_sparse_conditioned_repair(
        base_dense_capture=base_dense,
        base_rendered=base_rendered,
        repair_selection=repair_selection,
        rendered_candidates=[base_rendered],
    )

    np.testing.assert_allclose(reused["pose_w2c"], base_dense["pose_w2c"])
    assert reused["inliers"].tolist() == [0]
    assert reused["slcdp_repair_search"]["selection"]["decision"] == "reuse_base_dense_no_repair_accept"
    assert reused["slcdp_repair_search"]["selection"]["original_decision"] == "skip_dense_keep_sparse"
    assert reused["slcdp_repair_search"]["dense_reused_from_base"] is True
    assert reused["slcdp_repair_search"]["dense_skipped"] is False


def test_sparse_conditioned_reuses_base_dense_when_transition_rejects_repair():
    base_dense = {
        "pose_w2c": np.eye(4, dtype=np.float32),
        "query_xy": np.array([[1.0, 2.0]], dtype=np.float32),
        "rendered_xy": np.array([[3.0, 4.0]], dtype=np.float32),
        "p3d": np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        "inliers": np.array([0], dtype=np.int32),
        "K": np.eye(3, dtype=np.float32),
    }
    base_rendered = {
        "label": "base",
        "preflight": {"decision": "retry_sparse_or_patch_dense"},
        "render_control": {"mode": "sparse_conditioned", "candidate_label": "base"},
    }
    repair_selection = {
        "schema": "loc_gs_slcdp_repair_selection_v1",
        "decision": "accept_repaired_dense_pose",
        "selected_label": "gated_ray_side-5.000",
        "best_label": "gated_ray_side-5.000",
        "score_gain": 0.5,
        "candidate_count": 2,
    }
    transition_control = {
        "decision": "reject_dense_keep_sparse",
        "selected_pose_source": "sparse",
        "retained_inlier_ratio": 0.25,
    }

    reused = hard_match_viz._reuse_base_dense_after_rejected_sparse_conditioned_transition(
        base_dense_capture=base_dense,
        base_rendered=base_rendered,
        repair_selection=repair_selection,
        rendered_candidates=[
            base_rendered,
            {
                "label": "gated_ray_side-5.000",
                "preflight": {"decision": "accept_dense"},
                "render_control": {"mode": "sparse_conditioned", "candidate_label": "gated_ray_side-5.000"},
            },
        ],
        transition_control=transition_control,
    )

    np.testing.assert_allclose(reused["pose_w2c"], base_dense["pose_w2c"])
    assert reused["inliers"].tolist() == [0]
    assert reused["slcdp_repair_search"]["selection"]["decision"] == "reuse_base_dense_transition_reject"
    assert reused["slcdp_repair_search"]["selection"]["original_decision"] == "accept_repaired_dense_pose"
    assert reused["slcdp_repair_search"]["selection"]["rejected_selected_label"] == "gated_ray_side-5.000"
    assert reused["slcdp_repair_search"]["dense_reused_from_base"] is True
    assert reused["slcdp_transition_control"]["decision"] == "reject_dense_keep_sparse"


def test_offset_camera_center_moves_world_camera_center_without_rotating():
    pose = np.eye(4, dtype=np.float32)

    shifted = offset_camera_center(pose, np.array([0.5, -0.25, 0.0], dtype=np.float32))
    te_cm, re_deg = pose_error_cm_deg(shifted, pose)

    np.testing.assert_allclose(shifted[:3, :3], np.eye(3), atol=1e-6)
    assert te_cm == pytest.approx(55.901699, rel=1e-6)
    assert re_deg == pytest.approx(0.0)


def test_feature_pair_pca_rgb_uses_shared_basis_and_uint8_output():
    query = np.zeros((4, 3, 2), dtype=np.float32)
    render = np.zeros((4, 3, 2), dtype=np.float32)
    query[0] = 1.0
    render[1] = 1.0

    query_rgb, render_rgb = feature_pair_pca_rgb(query, render)

    assert query_rgb.shape == (3, 2, 3)
    assert render_rgb.shape == (3, 2, 3)
    assert query_rgb.dtype == np.uint8
    assert render_rgb.dtype == np.uint8
    assert np.isfinite(query_rgb).all()
    assert np.isfinite(render_rgb).all()


def test_feature_norm_and_cosine_maps_are_finite_for_zero_features():
    query = np.zeros((3, 2, 2), dtype=np.float32)
    render = np.zeros((3, 2, 2), dtype=np.float32)
    render[0, 0, 0] = 1.0

    norm = feature_norm_map(render)
    cosine = cosine_similarity_map(query, render)

    assert norm.shape == (2, 2)
    assert cosine.shape == (2, 2)
    assert np.isfinite(norm).all()
    assert np.isfinite(cosine).all()
    assert norm[0, 0] > norm[1, 1]


def test_feature_diagnostic_parser_marks_output_as_diagnostic():
    args = feature_diag.build_argparser().parse_args(["--max_cases", "1"])

    assert args.max_cases == 1
    assert feature_diag.DIAGNOSTIC_ONLY is True
