import numpy as np
from loc_gs.scripts.diagnose_patch_guided_sparse import (
    _build_dense_pgsh_refinement_matches,
    _dense_capture_from_matches,
    _dense_pgsh_base_trust_acceptance,
    _dense_pgsh_report_aliases,
    _json_safe,
    _match_solver_scores,
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
    filter_matches_by_quality,
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


def test_sparse_pgsh_accepts_match_quality_filter_options():
    args = build_argparser().parse_args(
        [
            "--patch_match_second_best_margin",
            "0.05",
            "--patch_match_best_per_landmark",
            "--patch_match_max_matches",
            "128",
            "--patch_match_quality_min_keep",
            "24",
        ]
    )

    assert args.patch_match_second_best_margin == 0.05
    assert args.patch_match_best_per_landmark is True
    assert args.patch_match_max_matches == 128
    assert args.patch_match_quality_min_keep == 24


def test_sparse_pgsh_accepts_semantic_patch_filter_options():
    args = build_argparser().parse_args(["--patch_semantic_mask_mode", "dynamic_sky", "--patch_require_semantic_mask"])

    assert args.patch_semantic_mask_mode == "dynamic_sky"
    assert args.patch_require_semantic_mask is True


def test_sparse_pgsh_accepts_base_reference_filter_option():
    args = build_argparser().parse_args(
        ["--pgsh_sparse_reference_filter_from_base", "--pgsh_reference_match_max_reprojection_px", "96"]
    )

    assert args.pgsh_sparse_reference_filter_from_base is True
    assert args.pgsh_reference_match_max_reprojection_px == 96.0


def test_sparse_pgsh_accepts_precision_selection_options():
    args = build_argparser().parse_args(
        ["--pgsh_patch_score_profile", "precision", "--pgsh_cluster_rank", "score", "--pgsh_group_max_patches", "2"]
    )

    assert args.pgsh_patch_score_profile == "precision"
    assert args.pgsh_cluster_rank == "score"
    assert args.pgsh_group_max_patches == 2


def test_sparse_pgsh_accepts_prosac_solver_score_options():
    args = build_argparser().parse_args(
        [
            "--patch_pnp_solver",
            "opencv_prosac_magsac",
            "--patch_solver_score_mode",
            "reference",
            "--patch_pnp_reprojection_error",
            "4.0",
        ]
    )

    assert args.patch_pnp_solver == "opencv_prosac_magsac"
    assert args.patch_solver_score_mode == "reference"
    assert args.patch_pnp_reprojection_error == 4.0


def test_sparse_pgsh_accepts_skip_visualization_option():
    args = build_argparser().parse_args(["--skip_visualization"])

    assert args.skip_visualization is True


def test_sparse_pgsh_accepts_sparse_anchor_fusion_options():
    args = build_argparser().parse_args(
        [
            "--pgsh_sparse_anchor_fusion",
            "--pgsh_sparse_anchor_max_reprojection_px",
            "12",
            "--pgsh_sparse_anchor_max_count",
            "64",
            "--pgsh_sparse_anchor_selection",
            "spatial_depth_diverse",
            "--pgsh_sparse_anchor_grid",
            "3",
        ]
    )

    assert args.pgsh_sparse_anchor_fusion is True
    assert args.pgsh_sparse_anchor_max_reprojection_px == 12.0
    assert args.pgsh_sparse_anchor_max_count == 64
    assert args.pgsh_sparse_anchor_selection == "spatial_depth_diverse"
    assert args.pgsh_sparse_anchor_grid == 3


def test_sparse_anchor_selection_can_preserve_spatial_depth_diversity():
    from loc_gs.scripts.diagnose_patch_guided_sparse import _select_sparse_anchor_indices

    candidate_indices = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    errors = np.array([0.1, 0.2, 0.3, 0.4, 1.5], dtype=np.float32)
    xy = np.array(
        [
            [10.0, 10.0],
            [12.0, 12.0],
            [14.0, 14.0],
            [16.0, 16.0],
            [90.0, 90.0],
        ],
        dtype=np.float32,
    )
    depths = np.array([4.0, 4.1, 4.2, 4.3, 12.0], dtype=np.float32)

    error_only = _select_sparse_anchor_indices(
        candidate_indices,
        errors=errors,
        query_xy=xy,
        depths=depths,
        image_size=(100, 100),
        max_anchor_count=2,
        selection_mode="error",
        grid_size=2,
    )
    diverse = _select_sparse_anchor_indices(
        candidate_indices,
        errors=errors,
        query_xy=xy,
        depths=depths,
        image_size=(100, 100),
        max_anchor_count=2,
        selection_mode="spatial_depth_diverse",
        grid_size=2,
    )

    assert error_only.tolist() == [0, 1]
    assert diverse.tolist() == [0, 4]


def test_sparse_pgsh_accepts_reference_local_refine_options():
    args = build_argparser().parse_args(
        [
            "--pgsh_sparse_reference_local_refine",
            "--pgsh_sparse_refine_max_translation_delta_m",
            "2.5",
            "--pgsh_sparse_refine_max_rotation_delta_deg",
            "1.5",
            "--pgsh_sparse_refine_rotation_prior_weight",
            "0.4",
            "--pgsh_sparse_refine_pose_mix",
            "rotation_only",
        ]
    )

    assert args.pgsh_sparse_reference_local_refine is True
    assert args.pgsh_sparse_refine_max_translation_delta_m == 2.5
    assert args.pgsh_sparse_refine_max_rotation_delta_deg == 1.5
    assert args.pgsh_sparse_refine_rotation_prior_weight == 0.4
    assert args.pgsh_sparse_refine_pose_mix == "rotation_only"


def test_resolve_camera_by_index_does_not_iterate_full_loader():
    from loc_gs.scripts.diagnose_patch_guided_sparse import _resolve_camera

    class Dataset:
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {"index": index}

    class Loader:
        dataset = Dataset()

        def __iter__(self):
            raise AssertionError("full camera iteration should not be used for query_index")

    camera = _resolve_camera({"cameras": Loader(), "camera_by_name": {}}, image_name="", query_index=2)

    assert camera == {"index": 2}


def test_lazy_camera_by_name_does_not_iterate_full_loader():
    from loc_gs.scripts.visualize_stdloc_hard_matches import _LazyCameraByName

    class Info:
        def __init__(self, image_name):
            self.image_name = image_name

    class Dataset:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return {"index": index}

    class Loader:
        dataset = Dataset()

        def __iter__(self):
            raise AssertionError("camera name mapping should not materialize the full loader")

    mapping = _LazyCameraByName(Loader(), [Info("a.png"), Info("b.png"), Info("c.png")])

    assert mapping.get("b.png") == {"index": 1}
    assert mapping.get("missing.png") is None


def test_sparse_anchor_fusion_keeps_base_inliers_explained_by_patch_pose():
    from loc_gs.scripts.diagnose_patch_guided_sparse import _fuse_sparse_anchors_with_matches

    base_sparse = {
        "query_xy": np.array([[10.0, 10.0], [20.0, 10.0], [80.0, 80.0]], dtype=np.float32),
        "p3d": np.array([[10.0, 10.0, 1.0], [20.0, 10.0, 1.0], [20.0, 20.0, 1.0]], dtype=np.float32),
        "gs_ids": np.array([1, 2, 3], dtype=np.int64),
        "scores": np.array([0.9, 0.8, 0.7], dtype=np.float32),
        "detector_scores": np.array([0.5, 0.6, 0.7], dtype=np.float32),
        "inliers": np.array([0, 1, 2], dtype=np.int32),
        "K": np.eye(3, dtype=np.float32),
        "width": 100,
        "height": 100,
    }
    patch_matches = {
        "xy": np.array([[30.0, 10.0]], dtype=np.float32),
        "xyz": np.array([[30.0, 10.0, 1.0]], dtype=np.float32),
        "gs_ids": np.array([9], dtype=np.int64),
        "score": np.array([1.0], dtype=np.float32),
    }

    fused, diagnostics = _fuse_sparse_anchors_with_matches(
        base_sparse=base_sparse,
        patch_matches=patch_matches,
        candidate_pose_w2c=np.eye(4, dtype=np.float32),
        max_reprojection_error_px=2.0,
        max_anchor_count=8,
    )

    assert diagnostics["added_anchor_count"] == 2
    assert fused["xy"].shape[0] == 3
    assert fused["gs_ids"].tolist() == [9, 1, 2]


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


def test_sparse_quality_filter_keeps_high_margin_matches():
    matches = {
        "xy": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
        "xyz": np.zeros((3, 3), dtype=np.float32),
        "score": np.array([0.9, 0.8, 0.7], dtype=np.float32),
        "margin": np.array([0.20, 0.01, 0.12], dtype=np.float32),
        "gs_ids": np.array([3, 4, 5], dtype=np.int64),
    }

    filtered, diagnostics = filter_matches_by_quality(matches, min_margin=0.10)

    assert filtered["indices"].tolist() == [0, 2]
    assert filtered["xy"].shape[0] == 2
    assert diagnostics["decision"] == "filtered"
    assert diagnostics["margin_kept_count"] == 2


def test_sparse_quality_filter_keeps_best_duplicate_landmark_match():
    matches = {
        "xy": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
        "xyz": np.zeros((3, 3), dtype=np.float32),
        "score": np.array([0.6, 0.9, 0.7], dtype=np.float32),
        "margin": np.array([0.2, 0.2, 0.2], dtype=np.float32),
        "gs_ids": np.array([8, 8, 9], dtype=np.int64),
    }

    filtered, diagnostics = filter_matches_by_quality(matches, best_per_landmark=True)

    assert filtered["indices"].tolist() == [1, 2]
    assert diagnostics["unique_landmark_kept_count"] == 2


def test_sparse_quality_filter_falls_back_when_min_matches_would_be_violated():
    matches = {
        "xy": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
        "xyz": np.zeros((3, 3), dtype=np.float32),
        "score": np.array([0.9, 0.8, 0.7], dtype=np.float32),
        "margin": np.array([0.20, 0.01, 0.02], dtype=np.float32),
        "gs_ids": np.array([3, 4, 5], dtype=np.int64),
    }

    filtered, diagnostics = filter_matches_by_quality(matches, min_margin=0.10, min_matches=2)

    assert filtered["indices"].tolist() == [0, 1, 2]
    assert diagnostics["decision"] == "fallback_original_too_few_matches"


def test_precision_patch_score_prefers_cleaner_geometry_over_more_inliers():
    patch = generate_patch_grid(image_size=(200, 200), patch_size=(100, 100), overlap=0)[0]
    many_xy = np.stack([np.linspace(0.0, 99.0, 96), np.linspace(0.0, 99.0, 96)], axis=1).astype(np.float32)
    clean_xy = many_xy[:71]
    many_inliers = np.zeros((96,), dtype=bool)
    many_inliers[:31] = True
    clean_inliers = np.zeros((71,), dtype=bool)
    clean_inliers[:25] = True
    many_errors = np.full((96,), 100.0, dtype=np.float32)
    many_errors[many_inliers] = 4.7
    clean_errors = np.full((71,), 100.0, dtype=np.float32)
    clean_errors[clean_inliers] = 2.4

    many = score_patch_hypothesis(
        patch,
        match_xy=many_xy,
        inlier_mask=many_inliers,
        reproj_errors=many_errors,
        depths=np.linspace(1.0, 5.0, 96),
        score_profile="precision",
    )
    clean = score_patch_hypothesis(
        patch,
        match_xy=clean_xy,
        inlier_mask=clean_inliers,
        reproj_errors=clean_errors,
        depths=np.linspace(1.0, 5.0, 71),
        score_profile="precision",
    )

    assert many.inlier_count > clean.inlier_count
    assert clean.components["inlier_ratio"] > many.components["inlier_ratio"]
    assert clean.components["reproj_median"] < many.components["reproj_median"]
    assert clean.score > many.score


def test_score_ranked_patch_group_can_select_lower_inlier_higher_score_cluster():
    low_quality_many = PatchHypothesis(
        patch_id=0,
        score=0.2,
        inlier_count=31,
        camera_center=np.zeros(3),
        rotation=np.eye(3),
        match_indices=np.arange(31),
    )
    high_quality_fewer = PatchHypothesis(
        patch_id=1,
        score=0.8,
        inlier_count=25,
        camera_center=np.array([10.0, 0.0, 0.0]),
        rotation=np.eye(3),
        match_indices=np.arange(25),
    )

    by_inliers = select_pose_consistent_patch_group(
        [low_quality_many, high_quality_fewer],
        center_thresh=1.0,
        rotation_thresh_deg=5.0,
        min_patch_count=1,
        rank_by="inlier_count",
    )
    by_score = select_pose_consistent_patch_group(
        [low_quality_many, high_quality_fewer],
        center_thresh=1.0,
        rotation_thresh_deg=5.0,
        min_patch_count=1,
        rank_by="score",
    )

    assert by_inliers.patch_ids == [0]
    assert by_score.patch_ids == [1]


def test_patch_group_can_cap_pose_consistent_cluster_by_score():
    hypotheses = [
        PatchHypothesis(
            patch_id=0,
            score=0.2,
            inlier_count=30,
            camera_center=np.zeros(3),
            rotation=np.eye(3),
            match_indices=np.arange(0, 30),
        ),
        PatchHypothesis(
            patch_id=1,
            score=0.8,
            inlier_count=20,
            camera_center=np.zeros(3),
            rotation=np.eye(3),
            match_indices=np.arange(30, 50),
        ),
        PatchHypothesis(
            patch_id=2,
            score=0.7,
            inlier_count=10,
            camera_center=np.zeros(3),
            rotation=np.eye(3),
            match_indices=np.arange(50, 60),
        ),
    ]

    group = select_pose_consistent_patch_group(
        hypotheses,
        center_thresh=1.0,
        rotation_thresh_deg=5.0,
        min_patch_count=1,
        rank_by="score",
        max_patch_count=2,
    )

    assert group.patch_ids == [1, 2]
    assert group.inlier_count == 30


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
    assert filtered["reference_reprojection_error_px"].shape == (2,)
    assert float(filtered["reference_reprojection_error_px"][0]) == 0.0
    assert diagnostics["before_count"] == 3
    assert diagnostics["after_count"] == 2
    assert diagnostics["kept_ratio"] == 2 / 3


def test_reference_solver_scores_prefer_low_reprojection_high_margin_matches():
    matches = {
        "xy": np.zeros((3, 2), dtype=np.float32),
        "score": np.array([0.6, 0.9, 0.7], dtype=np.float32),
        "margin": np.array([0.10, 0.01, 0.20], dtype=np.float32),
        "reference_reprojection_error_px": np.array([2.0, 90.0, 10.0], dtype=np.float32),
    }

    scores = _match_solver_scores(matches, mode="reference", reference_max_reprojection_px=96.0)

    assert scores.shape == (3,)
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]


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
