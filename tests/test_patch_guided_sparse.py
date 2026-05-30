import numpy as np
from loc_gs.scripts.diagnose_patch_guided_sparse import (
    _build_dense_pgsh_refinement_matches,
    _dense_capture_from_matches,
    _dense_pgsh_base_trust_acceptance,
    _dense_pgsh_report_aliases,
    _json_safe,
    _matches_from_sparse,
    _matches_from_dense,
    _resolve_diagnostic_slcdp_options,
    build_argparser,
)

from loc_gs.stdloc_native.patch_guided_sparse import (
    PatchHypothesis,
    PatchResidualWeightPolicy,
    apply_patch_residual_group_weight,
    cluster_patch_hypotheses,
    compute_patch_residual_group_weight,
    evaluate_pose_global_consistency,
    filter_matches_by_reference_reprojection,
    filter_matches_by_patch,
    generate_patch_grid,
    merge_group_matches,
    refine_pose_with_reference_prior,
    score_patch_hypothesis,
    select_pose_consistent_patch_group,
)


def test_pgsh_diagnostic_defaults_preserve_native_sparse_config():
    args = build_argparser().parse_args([])

    assert args.base_sparse_solver == "native"
    assert args.base_sparse_max_iterations == 0
    assert args.base_sparse_min_iterations == 0


def test_dense_pgsh_is_explicit_and_can_disable_sparse_pgsh():
    default_args = build_argparser().parse_args([])
    dense_args = build_argparser().parse_args(["--dense_pgsh", "--disable_sparse_pgsh"])

    assert default_args.dense_pgsh is False
    assert default_args.disable_sparse_pgsh is False
    assert default_args.pgsh_detector_score_weight == 0.0
    assert dense_args.dense_pgsh is True
    assert dense_args.disable_sparse_pgsh is True


def test_patch_diagnostic_accepts_soft_slcdp_transition_option():
    args = build_argparser().parse_args(["--slcdp_soft_transition_control"])
    resolved = _resolve_diagnostic_slcdp_options(args)

    assert resolved["slcdp_soft_transition_control"] is True
    assert resolved["slcdp_transition_control"] is False


def test_patch_diagnostic_accepts_patch_residual_weighting_option():
    args = build_argparser().parse_args(["--dense_pgsh_patch_residual_weighting", "--dense_pgsh_patch_residual_min_group_patches", "3"])

    assert args.dense_pgsh_patch_residual_weighting is True
    assert args.dense_pgsh_patch_residual_min_group_patches == 3


def test_legacy_sparse_conditioned_repair_selection_can_reproduce_basefirst_diagnostic():
    args = build_argparser().parse_args(
        [
            "--slcdp_render_control",
            "sparse_conditioned",
            "--slcdp_sparse_conditioned_legacy_repair_selection",
        ]
    )

    options = _resolve_diagnostic_slcdp_options(args)

    assert options["slcdp_repair_require_accept"] is False
    assert options["slcdp_repair_min_score_gain"] == 0.05
    assert options["slcdp_repair_translation_penalty_per_m"] == 0.12
    assert options["slcdp_repair_skip_if_no_accept"] is False


def test_dense_stage_match_adapter_uses_dense_correspondences_only():
    dense = {
        "query_xy": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "rendered_xy": np.array([[11.0, 12.0], [13.0, 14.0]], dtype=np.float32),
        "p3d": np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 3.0]], dtype=np.float32),
    }

    matches = _matches_from_dense(dense)

    assert matches["xy"].tolist() == dense["query_xy"].tolist()
    assert matches["rendered_xy"].tolist() == dense["rendered_xy"].tolist()
    assert matches["xyz"].tolist() == dense["p3d"].tolist()
    assert matches["indices"].tolist() == [0, 1]


def test_sparse_match_adapter_preserves_scene_detector_scores():
    sparse = {
        "query_xy": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "p3d": np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 3.0]], dtype=np.float32),
        "gs_ids": np.array([7, 9], dtype=np.int64),
        "scores": np.array([0.6, 0.7], dtype=np.float32),
        "detector_scores": np.array([0.2, 0.9], dtype=np.float32),
    }

    matches = _matches_from_sparse(sparse)

    assert matches["detector_score"].tolist() == sparse["detector_scores"].tolist()


def test_dense_capture_from_matches_preserves_dense_points_and_final_pose():
    base_dense = {
        "K": np.eye(3, dtype=np.float32),
        "render": "render-token",
        "ray_depth_diagnostics": [{"ray": 1}],
        "slcdp_render_control": {"mode": "sparse_conditioned"},
        "slcdp_preflight": {"decision": "accept_dense"},
        "slcdp_repair_search": {"enabled": True},
    }
    merged = {
        "xy": np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32),
        "rendered_xy": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        "xyz": np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 3.0]], dtype=np.float32),
    }
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 2.0
    inliers = np.array([1], dtype=np.int32)

    capture = _dense_capture_from_matches(
        base_dense,
        merged,
        pose,
        inliers,
        width=64,
        height=48,
        dense_pgsh_metadata={"decision": "use_dense_pgsh"},
    )

    assert capture["query_xy"].tolist() == merged["xy"].tolist()
    assert capture["rendered_xy"].tolist() == merged["rendered_xy"].tolist()
    assert capture["p3d"].tolist() == merged["xyz"].tolist()
    assert capture["inliers"].tolist() == [1]
    assert capture["pose_w2c"].tolist() == pose.tolist()
    assert capture["width"] == 64
    assert capture["height"] == 48
    assert capture["render"] == "render-token"
    assert capture["dense_pgsh"]["decision"] == "use_dense_pgsh"


def test_detector_score_weight_promotes_scene_detector_supported_patch():
    patch = generate_patch_grid(image_size=(40, 40), patch_size=(20, 20), overlap=0)[0]
    xy = np.array([[2.0, 2.0], [8.0, 2.0], [2.0, 8.0], [8.0, 8.0], [14.0, 14.0]])
    inliers = np.array([True, True, True, True, False])
    reproj = np.array([0.4, 0.5, 0.6, 0.7, 5.0])
    depths = np.array([1.0, 2.0, 4.0, 7.0, 10.0])

    low = score_patch_hypothesis(
        patch,
        match_xy=xy,
        inlier_mask=inliers,
        reproj_errors=reproj,
        depths=depths,
        bearings_2d=xy,
        detector_scores=np.array([0.1, 0.1, 0.1, 0.1, 0.9]),
        detector_score_weight=0.25,
    )
    high = score_patch_hypothesis(
        patch,
        match_xy=xy,
        inlier_mask=inliers,
        reproj_errors=reproj,
        depths=depths,
        bearings_2d=xy,
        detector_scores=np.array([0.9, 0.9, 0.9, 0.9, 0.1]),
        detector_score_weight=0.25,
    )

    assert high.score > low.score
    assert high.components["detector_support"] > low.components["detector_support"]


def test_global_consistency_retains_sparse_landmarks_for_consistent_pose():
    pose = np.eye(4, dtype=np.float32)
    intrinsic = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    points = np.array([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [0.0, 1.0, 10.0], [1.0, 1.0, 10.0]], dtype=np.float32)
    query_xy = np.array([[5.0, 5.0], [6.0, 5.0], [5.0, 6.0], [6.0, 6.0]], dtype=np.float32)

    result = evaluate_pose_global_consistency(
        pose_w2c=pose,
        intrinsic=intrinsic,
        image_size=(20, 20),
        sparse_query_xy=query_xy,
        sparse_points_world=points,
        sparse_inlier_indices=np.arange(4),
        sparse_retention_max_error_px=1.0,
        sparse_retention_min_ratio=0.75,
    )

    assert result["decision"] == "accept_patch_pose"
    assert result["sparse_retained_ratio"] == 1.0
    assert result["failed_checks"]["sparse_inlier_retention"] is False


def test_global_consistency_rejects_pose_that_does_not_retain_sparse_landmarks():
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 5.0
    intrinsic = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    points = np.array([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [0.0, 1.0, 10.0], [1.0, 1.0, 10.0]], dtype=np.float32)
    query_xy = np.array([[5.0, 5.0], [6.0, 5.0], [5.0, 6.0], [6.0, 6.0]], dtype=np.float32)

    result = evaluate_pose_global_consistency(
        pose_w2c=pose,
        intrinsic=intrinsic,
        image_size=(20, 20),
        sparse_query_xy=query_xy,
        sparse_points_world=points,
        sparse_inlier_indices=np.arange(4),
        sparse_retention_max_error_px=1.0,
        sparse_retention_min_ratio=0.75,
    )

    assert result["decision"] == "reject_patch_pose"
    assert result["sparse_retained_ratio"] == 0.0
    assert result["failed_checks"]["sparse_inlier_retention"] is True


def test_global_consistency_rejects_patch_pose_far_from_reference_pose_without_gt():
    reference = np.eye(4, dtype=np.float32)
    candidate = np.eye(4, dtype=np.float32)
    candidate[0, 3] = -2.0

    result = evaluate_pose_global_consistency(
        pose_w2c=candidate,
        intrinsic=np.eye(3, dtype=np.float32),
        image_size=(20, 20),
        reference_pose_w2c=reference,
        reference_max_translation_delta_m=0.75,
        reference_max_rotation_delta_deg=3.0,
    )

    assert result["decision"] == "reject_patch_pose"
    assert result["failed_checks"]["reference_translation_delta"] is True


def test_reference_reprojection_filter_removes_geometrically_inconsistent_patch_matches():
    matches = {
        "xy": np.array([[5.0, 5.0], [6.0, 5.0], [50.0, 50.0]], dtype=np.float32),
        "xyz": np.array([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [0.0, 1.0, 10.0]], dtype=np.float32),
        "score": np.array([0.9, 0.8, 0.7], dtype=np.float32),
    }
    pose = np.eye(4, dtype=np.float32)
    intrinsic = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    filtered, diagnostics = filter_matches_by_reference_reprojection(
        matches,
        pose_w2c=pose,
        intrinsic=intrinsic,
        image_size=(80, 80),
        max_reprojection_error_px=1.0,
    )

    assert filtered["xy"].tolist() == [[5.0, 5.0], [6.0, 5.0]]
    assert filtered["score"].tolist() == [np.float32(0.9).item(), np.float32(0.8).item()]
    assert diagnostics["before_count"] == 3
    assert diagnostics["after_count"] == 2
    assert diagnostics["kept_ratio"] == 2 / 3


def test_reference_prior_refinement_reduces_reprojection_error_from_nearby_pose():
    intrinsic = np.array([[40.0, 0.0, 20.0], [0.0, 40.0, 20.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    points = np.array(
        [
            [-1.0, -1.0, 8.0],
            [1.0, -1.0, 8.0],
            [-1.0, 1.0, 10.0],
            [1.0, 1.0, 10.0],
            [0.0, 0.0, 12.0],
            [2.0, 0.5, 12.0],
        ],
        dtype=np.float32,
    )
    true_pose = np.eye(4, dtype=np.float32)
    query_xy = np.array(
        [
            [15.0, 15.0],
            [25.0, 15.0],
            [16.0, 24.0],
            [24.0, 24.0],
            [20.0, 20.0],
            [26.666666, 21.666666],
        ],
        dtype=np.float32,
    )
    reference = np.eye(4, dtype=np.float32)
    reference[0, 3] = 0.8

    refined, diagnostics = refine_pose_with_reference_prior(
        reference_pose_w2c=reference,
        match_xy=query_xy,
        points_world=points,
        intrinsic=intrinsic,
        image_size=(40, 40),
        max_iterations=50,
        reprojection_loss_scale_px=2.0,
        translation_prior_weight=0.01,
        rotation_prior_weight=0.01,
    )

    assert diagnostics["after_median_reprojection_error_px"] < diagnostics["before_median_reprojection_error_px"]
    assert abs(float(refined[0, 3])) < abs(float(reference[0, 3]))
    assert diagnostics["success"] is True


def test_reference_prior_refinement_consumes_match_weights():
    intrinsic = np.array([[80.0, 0.0, 40.0], [0.0, 80.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    points = np.array(
        [
            [-1.0, -1.0, 8.0],
            [1.0, -1.0, 8.0],
            [-1.0, 1.0, 8.0],
            [1.0, 1.0, 8.0],
            [-0.5, -0.5, 10.0],
            [0.5, -0.5, 10.0],
            [-0.5, 0.5, 10.0],
            [0.5, 0.5, 10.0],
        ],
        dtype=np.float32,
    )
    correct_xy = np.column_stack(
        [
            intrinsic[0, 0] * points[:, 0] / points[:, 2] + intrinsic[0, 2],
            intrinsic[1, 1] * points[:, 1] / points[:, 2] + intrinsic[1, 2],
        ]
    ).astype(np.float32)
    shifted_dense_xy = correct_xy + np.array([8.0, 0.0], dtype=np.float32)
    match_xy = np.concatenate([shifted_dense_xy, correct_xy], axis=0)
    match_xyz = np.concatenate([points, points], axis=0)
    reference = np.eye(4, dtype=np.float32)

    low_weight_pose, low_weight_diagnostics = refine_pose_with_reference_prior(
        reference_pose_w2c=reference,
        match_xy=match_xy,
        points_world=match_xyz,
        intrinsic=intrinsic,
        image_size=(80, 80),
        max_iterations=80,
        reprojection_loss_scale_px=4.0,
        translation_prior_weight=0.001,
        rotation_prior_weight=0.001,
        match_weights=np.ones((match_xy.shape[0],), dtype=np.float32),
    )
    high_anchor_weights = np.concatenate(
        [np.ones((points.shape[0],), dtype=np.float32), np.full((points.shape[0],), 100.0, dtype=np.float32)],
        axis=0,
    )
    high_weight_pose, high_weight_diagnostics = refine_pose_with_reference_prior(
        reference_pose_w2c=reference,
        match_xy=match_xy,
        points_world=match_xyz,
        intrinsic=intrinsic,
        image_size=(80, 80),
        max_iterations=80,
        reprojection_loss_scale_px=4.0,
        translation_prior_weight=0.001,
        rotation_prior_weight=0.001,
        match_weights=high_anchor_weights,
    )

    assert high_weight_diagnostics["weighted_match_count"] > low_weight_diagnostics["weighted_match_count"]
    assert abs(float(high_weight_pose[0, 3])) < abs(float(low_weight_pose[0, 3]))


def test_reference_prior_refinement_respects_translation_trust_region():
    intrinsic = np.array([[40.0, 0.0, 20.0], [0.0, 40.0, 20.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    points = np.array([[-1.0, -1.0, 8.0], [1.0, -1.0, 8.0], [-1.0, 1.0, 10.0], [1.0, 1.0, 10.0]], dtype=np.float32)
    query_xy = np.array([[15.0, 15.0], [25.0, 15.0], [16.0, 24.0], [24.0, 24.0]], dtype=np.float32)
    reference = np.eye(4, dtype=np.float32)
    reference[0, 3] = 2.0

    refined, diagnostics = refine_pose_with_reference_prior(
        reference_pose_w2c=reference,
        match_xy=query_xy,
        points_world=points,
        intrinsic=intrinsic,
        image_size=(40, 40),
        max_iterations=50,
        max_translation_delta_m=0.25,
        max_rotation_delta_deg=2.0,
    )

    refined_center = -refined[:3, :3].T @ refined[:3, 3]
    reference_center = -reference[:3, :3].T @ reference[:3, 3]
    translation_delta = np.linalg.norm(refined_center - reference_center)
    assert translation_delta <= 0.2501
    assert diagnostics["delta_translation_norm_m"] <= 0.2501
    assert diagnostics["max_translation_delta_m"] == 0.25


def test_reference_prior_refinement_bounds_camera_center_motion_when_rotating():
    intrinsic = np.array([[80.0, 0.0, 32.0], [0.0, 80.0, 24.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    points = np.array(
        [
            [-1.0, -0.5, 8.0],
            [1.0, -0.4, 8.0],
            [-0.8, 0.7, 10.0],
            [1.2, 0.8, 10.0],
            [0.0, 0.1, 9.0],
        ],
        dtype=np.float32,
    )
    reference = np.eye(4, dtype=np.float32)
    reference[:3, 3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    center = -reference[:3, :3].T @ reference[:3, 3]
    angle = np.radians(3.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    target = np.eye(4, dtype=np.float32)
    target[:3, :3] = rotation
    target[:3, 3] = -rotation @ center
    camera = (target[:3, :3] @ points.T).T + target[:3, 3]
    query_xy = np.column_stack(
        [
            intrinsic[0, 0] * camera[:, 0] / camera[:, 2] + intrinsic[0, 2],
            intrinsic[1, 1] * camera[:, 1] / camera[:, 2] + intrinsic[1, 2],
        ]
    ).astype(np.float32)

    refined, diagnostics = refine_pose_with_reference_prior(
        reference_pose_w2c=reference,
        match_xy=query_xy,
        points_world=points,
        intrinsic=intrinsic,
        image_size=(64, 48),
        max_iterations=50,
        max_translation_delta_m=0.0,
        max_rotation_delta_deg=5.0,
        translation_prior_weight=0.01,
        rotation_prior_weight=0.01,
    )

    refined_center = -refined[:3, :3].T @ refined[:3, 3]
    assert np.linalg.norm(refined_center - center) <= 1.0e-5
    assert diagnostics["delta_translation_norm_m"] == 0.0
    assert diagnostics["after_median_reprojection_error_px"] < diagnostics["before_median_reprojection_error_px"]


def test_dense_pgsh_refinement_matches_add_reference_consistent_sparse_anchors():
    dense_merged = {
        "xy": np.array([[20.0, 20.0], [24.0, 20.0]], dtype=np.float32),
        "xyz": np.array([[0.0, 0.0, 10.0], [0.5, 0.0, 10.0]], dtype=np.float32),
    }
    sparse = {
        "query_xy": np.array([[1.0, 1.0], [1.6, 1.0], [90.0, 90.0]], dtype=np.float32),
        "p3d": np.array([[1.0, 1.0, 10.0], [1.6, 1.0, 10.0], [1.0, 1.0, 10.0]], dtype=np.float32),
        "inliers": np.array([0, 1, 2], dtype=np.int32),
    }
    intrinsic = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    refinement = _build_dense_pgsh_refinement_matches(
        dense_merged,
        sparse,
        reference_pose_w2c=np.eye(4, dtype=np.float32),
        intrinsic=intrinsic,
        width=100,
        height=100,
        sparse_anchor_weight=3.0,
        sparse_anchor_max_count=8,
        sparse_anchor_max_reprojection_px=2.0,
    )

    assert refinement["xy"].shape == (4, 2)
    assert refinement["xyz"].shape == (4, 3)
    assert refinement["weights"].tolist() == [1.0, 1.0, 3.0, 3.0]
    assert refinement["diagnostics"]["enabled"] is True
    assert refinement["diagnostics"]["selected_anchor_count"] == 2
    assert refinement["diagnostics"]["candidate_anchor_count"] == 3


def test_json_safe_converts_numpy_values_in_nested_diagnostics():
    payload = {
        "pose": np.eye(4, dtype=np.float32),
        "items": [{"count": np.int64(3), "fraction": np.float32(0.25)}],
    }

    safe = _json_safe(payload)

    assert safe["pose"] == np.eye(4, dtype=np.float32).tolist()
    assert safe["items"] == [{"count": 3, "fraction": np.float32(0.25).item()}]


def test_dense_pgsh_base_trust_rejects_large_pose_drift_without_gt():
    base = np.eye(4, dtype=np.float32)
    candidate = np.eye(4, dtype=np.float32)
    candidate[0, 3] = -2.0

    acceptance = _dense_pgsh_base_trust_acceptance(
        base_dense_pose_w2c=base,
        candidate_pose_w2c=candidate,
        max_translation_delta_m=0.75,
        max_rotation_delta_deg=3.0,
    )

    assert acceptance["decision"] == "reject_dense_pgsh_keep_base_dense"
    assert acceptance["failed_checks"]["translation_from_base_dense"] is True


def test_dense_pgsh_base_trust_accepts_small_pose_drift_without_gt():
    base = np.eye(4, dtype=np.float32)
    candidate = np.eye(4, dtype=np.float32)
    candidate[0, 3] = -0.1

    acceptance = _dense_pgsh_base_trust_acceptance(
        base_dense_pose_w2c=base,
        candidate_pose_w2c=candidate,
        max_translation_delta_m=0.75,
        max_rotation_delta_deg=3.0,
    )

    assert acceptance["decision"] == "accept_dense_pgsh_pose"
    assert acceptance["failed_checks"] == {
        "translation_from_base_dense": False,
        "rotation_from_base_dense": False,
    }


def test_dense_pgsh_base_trust_accepts_float_roundoff_at_translation_boundary():
    base = np.eye(4, dtype=np.float32)
    candidate = np.eye(4, dtype=np.float32)
    candidate[0, 3] = -0.7500001

    acceptance = _dense_pgsh_base_trust_acceptance(
        base_dense_pose_w2c=base,
        candidate_pose_w2c=candidate,
        max_translation_delta_m=0.75,
        max_rotation_delta_deg=3.0,
    )

    assert acceptance["decision"] == "accept_dense_pgsh_pose"
    assert acceptance["failed_checks"]["translation_from_base_dense"] is False


def test_dense_pgsh_report_aliases_are_unambiguous():
    aliases = _dense_pgsh_report_aliases(
        {
            "enabled": True,
            "decision": "fallback_base_dense_base_trust_reject",
            "dense_pgsh_te_cm": 75.0,
            "dense_pgsh_re_deg": 0.7,
            "dense_pgsh_raw_te_cm": 186.0,
            "dense_pgsh_raw_re_deg": 0.3,
        }
    )

    assert aliases == {
        "dense_pgsh_effective_te_cm": 75.0,
        "dense_pgsh_effective_re_deg": 0.7,
        "dense_pgsh_raw_candidate_te_cm": 186.0,
        "dense_pgsh_raw_candidate_re_deg": 0.3,
        "dense_pgsh_decision": "fallback_base_dense_base_trust_reject",
    }


def test_disabled_dense_pgsh_report_aliases_are_explicit():
    aliases = _dense_pgsh_report_aliases({"enabled": False})

    assert aliases["dense_pgsh_effective_te_cm"] is None
    assert aliases["dense_pgsh_raw_candidate_te_cm"] is None
    assert aliases["dense_pgsh_decision"] == "disabled"


def test_overlap_patches_are_deterministic_and_cover_image_bounds():
    patches = generate_patch_grid(image_size=(10, 8), patch_size=(4, 4), overlap=2)

    assert [(patch.x0, patch.y0, patch.x1, patch.y1) for patch in patches] == [
        (0, 0, 4, 4),
        (2, 0, 6, 4),
        (4, 0, 8, 4),
        (6, 0, 10, 4),
        (0, 2, 4, 6),
        (2, 2, 6, 6),
        (4, 2, 8, 6),
        (6, 2, 10, 6),
        (0, 4, 4, 8),
        (2, 4, 6, 8),
        (4, 4, 8, 8),
        (6, 4, 10, 8),
    ]


def test_small_patch_and_low_depth_spread_are_downweighted():
    big_patch = generate_patch_grid(image_size=(40, 40), patch_size=(20, 20), overlap=0)[0]
    small_patch = generate_patch_grid(image_size=(40, 40), patch_size=(4, 4), overlap=0)[0]
    xy = np.array([[2.0, 2.0], [8.0, 2.0], [2.0, 8.0], [8.0, 8.0], [14.0, 14.0]])
    good = score_patch_hypothesis(
        big_patch,
        match_xy=xy,
        inlier_mask=np.array([True, True, True, True, False]),
        reproj_errors=np.array([0.4, 0.5, 0.6, 0.7, 5.0]),
        depths=np.array([1.0, 2.0, 4.0, 7.0, 10.0]),
        bearings_2d=xy,
    )
    weak = score_patch_hypothesis(
        small_patch,
        match_xy=xy[:4],
        inlier_mask=np.ones(4, dtype=bool),
        reproj_errors=np.array([0.4, 0.5, 0.6, 0.7]),
        depths=np.array([3.00, 3.01, 3.02, 3.03]),
        bearings_2d=xy[:4],
    )

    assert good.components["spatial_extent"] > weak.components["spatial_extent"]
    assert good.components["depth_spread"] > weak.components["depth_spread"]
    assert weak.score < good.score


def test_pose_clustering_merges_consistent_patches_and_splits_inconsistent_pose():
    base_rotation = np.eye(3)
    hypotheses = [
        PatchHypothesis(
            patch_id=0,
            score=0.9,
            inlier_count=10,
            camera_center=np.array([0.0, 0.0, 0.0]),
            rotation=base_rotation,
            match_indices=np.array([0, 1]),
        ),
        PatchHypothesis(
            patch_id=1,
            score=0.8,
            inlier_count=9,
            camera_center=np.array([0.05, 0.0, 0.0]),
            rotation=base_rotation,
            match_indices=np.array([2, 3]),
        ),
        PatchHypothesis(
            patch_id=2,
            score=0.7,
            inlier_count=8,
            camera_center=np.array([2.0, 0.0, 0.0]),
            rotation=base_rotation,
            match_indices=np.array([4, 5]),
        ),
    ]

    clusters = cluster_patch_hypotheses(hypotheses, center_thresh=0.2, rotation_thresh_deg=2.0)

    assert [cluster.patch_ids for cluster in clusters] == [[0, 1], [2]]


def test_top_patch_group_merges_matches_from_best_pose_consistent_cluster():
    matches = {
        "xy": np.array([[1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0], [30.0, 30.0]]),
        "xyz": np.arange(15.0).reshape(5, 3),
        "score": np.array([0.9, 0.8, 0.7, 0.6, 0.1]),
    }
    patches = generate_patch_grid(image_size=(40, 40), patch_size=(16, 16), overlap=0)
    patch0_matches = filter_matches_by_patch(matches, patches[0])
    patch1_matches = filter_matches_by_patch(matches, patches[1])
    patch3_matches = filter_matches_by_patch(matches, patches[3])
    hypotheses = [
        PatchHypothesis(0, 1.0, 2, np.zeros(3), np.eye(3), patch0_matches["indices"]),
        PatchHypothesis(1, 0.9, 2, np.array([0.05, 0.0, 0.0]), np.eye(3), patch1_matches["indices"]),
        PatchHypothesis(3, 2.0, 1, np.array([5.0, 0.0, 0.0]), np.eye(3), patch3_matches["indices"]),
    ]

    group = select_pose_consistent_patch_group(hypotheses, center_thresh=0.2, rotation_thresh_deg=2.0)
    merged = merge_group_matches(matches, group)

    assert group.patch_ids == [0, 1]
    assert merged["indices"].tolist() == [0, 1, 2, 3]
    assert merged["xy"].tolist() == matches["xy"][:4].tolist()


def test_patch_group_can_require_multi_patch_consensus():
    hypotheses = [
        PatchHypothesis(
            patch_id=0,
            score=5.0,
            inlier_count=50,
            camera_center=np.array([10.0, 0.0, 0.0]),
            rotation=np.eye(3),
            match_indices=np.array([0, 1]),
        ),
        PatchHypothesis(
            patch_id=1,
            score=1.0,
            inlier_count=9,
            camera_center=np.array([0.0, 0.0, 0.0]),
            rotation=np.eye(3),
            match_indices=np.array([2, 3]),
        ),
        PatchHypothesis(
            patch_id=2,
            score=1.0,
            inlier_count=8,
            camera_center=np.array([0.05, 0.0, 0.0]),
            rotation=np.eye(3),
            match_indices=np.array([4, 5]),
        ),
    ]

    group = select_pose_consistent_patch_group(
        hypotheses,
        center_thresh=0.2,
        rotation_thresh_deg=2.0,
        min_patch_count=2,
    )

    assert group.patch_ids == [1, 2]
    assert group.match_indices.tolist() == [2, 3, 4, 5]


def test_patch_group_returns_empty_when_consensus_requirement_is_unmet():
    hypotheses = [
        PatchHypothesis(0, 1.0, 20, np.array([0.0, 0.0, 0.0]), np.eye(3), np.array([0, 1])),
        PatchHypothesis(1, 1.0, 20, np.array([5.0, 0.0, 0.0]), np.eye(3), np.array([2, 3])),
    ]

    group = select_pose_consistent_patch_group(
        hypotheses,
        center_thresh=0.2,
        rotation_thresh_deg=2.0,
        min_patch_count=2,
    )

    assert group.patch_ids == []
    assert group.match_indices.size == 0


def test_patch_residual_group_weight_downweights_local_only_ambiguous_group():
    weak_hypotheses = [
        PatchHypothesis(
            patch_id=0,
            score=2.0,
            inlier_count=30,
            match_indices=np.array([0, 1]),
            components={"ambiguity_risk": 0.9, "spatial_extent": 0.05, "depth_spread": 0.05, "logdet_proxy": 0.1},
        )
    ]
    strong_hypotheses = [
        PatchHypothesis(
            patch_id=0,
            score=1.0,
            inlier_count=20,
            match_indices=np.array([0, 1]),
            components={"ambiguity_risk": 0.1, "spatial_extent": 0.5, "depth_spread": 0.6, "logdet_proxy": 0.5},
        ),
        PatchHypothesis(
            patch_id=1,
            score=0.9,
            inlier_count=18,
            match_indices=np.array([2, 3]),
            components={"ambiguity_risk": 0.1, "spatial_extent": 0.4, "depth_spread": 0.5, "logdet_proxy": 0.5},
        ),
    ]
    weak_group = select_pose_consistent_patch_group(weak_hypotheses, center_thresh=1.0, rotation_thresh_deg=5.0)
    strong_group = select_pose_consistent_patch_group(strong_hypotheses, center_thresh=1.0, rotation_thresh_deg=5.0)

    weak = compute_patch_residual_group_weight(
        weak_group,
        weak_hypotheses,
        total_patch_count=4,
        policy=PatchResidualWeightPolicy(min_group_patches=2),
    )
    strong = compute_patch_residual_group_weight(
        strong_group,
        strong_hypotheses,
        total_patch_count=4,
        policy=PatchResidualWeightPolicy(min_group_patches=2),
    )

    assert weak["patch_residual_weight"] < strong["patch_residual_weight"]
    assert weak["components"]["pose_consensus"] < strong["components"]["pose_consensus"]
    assert weak["components"]["ambiguity_safety"] < strong["components"]["ambiguity_safety"]


def test_apply_patch_residual_group_weight_scales_dense_matches_not_sparse_anchors():
    refinement = {
        "xy": np.zeros((5, 2), dtype=np.float32),
        "xyz": np.zeros((5, 3), dtype=np.float32),
        "weights": np.ones((5,), dtype=np.float32),
    }

    weighted = apply_patch_residual_group_weight(refinement, patch_residual_weight=0.25, dense_match_count=3)

    assert weighted["weights"].tolist() == [0.25, 0.25, 0.25, 1.0, 1.0]
    assert weighted["patch_residual_weighting"]["dense_match_count"] == 3
