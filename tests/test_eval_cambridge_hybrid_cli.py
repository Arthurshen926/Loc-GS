import pickle
from types import SimpleNamespace

import numpy as np
import torch

from loc_gs.localization.stdloc_parity import DenseMatchResult
from loc_gs.scripts import eval_cambridge_hybrid
from loc_gs.scripts.eval_cambridge_hybrid import (
    build_argparser,
    effective_eval_config,
    fuse_projected_teacher_descriptors,
    gated_residual_descriptor_blend,
    generate_pnp_hypotheses,
    local_geometric_consistency_scores,
    local_image_pair_geometry_scores,
    prepare_query_teacher_maps,
    select_pnp_match_indices,
    select_view_landmark_indices,
    sparse_matchability_metrics,
)


def test_load_pickle_tensor_accepts_stdloc_score_dict(tmp_path):
    path = tmp_path / "sampled_scores.pkl"
    expected = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    with path.open("wb") as f:
        pickle.dump(
            {
                "sampled_scores": torch.tensor([0.2, 0.3], dtype=torch.float32),
                "score_avg": expected,
                "score_num": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            },
            f,
        )

    result = eval_cambridge_hybrid._load_pickle_tensor(path)

    assert torch.equal(result, expected)


def test_eval_parser_exposes_stdloc_parity_options():
    args = build_argparser().parse_args(
        [
            "--checkpoint",
            "output/model/latest.pth",
            "--matcher",
            "stdloc_parity",
            "--sparse_matcher",
            "lightglue",
            "--dense_matcher",
            "lightglue_rendered",
            "--dim_pipeline",
            "superpoint+lightglue",
            "--rendered_keypoint_source",
            "locability",
            "--lightglue_max_keypoints",
            "512",
            "--lightglue_filter_threshold",
            "0.05",
            "--loftr_image_scale",
            "0.5",
            "--loftr_min_confidence",
            "0.1",
            "--loftr_max_matches",
            "1024",
            "--dense_sparse_consistency_gate",
            "--dense_sparse_consistency_max_median_ratio",
            "1.2",
            "--dense_sparse_consistency_max_median_increase",
            "0.5",
            "--dense_sparse_consistency_min_inlier_ratio_factor",
            "0.8",
            "--dense_pose_delta_gate",
            "--dense_pose_delta_max_trans_cm",
            "25",
            "--dense_pose_delta_max_rot_deg",
            "2",
            "--solver",
            "poselib",
            "--poselib_refine",
            "--sparse_dual_softmax",
            "--dense_dual_softmax_temp",
            "0.2",
            "--fine_dual_softmax_temp",
            "0.3",
            "--mnn",
            "--subpixel_refine",
            "--dense_query_pixel_center_offset",
            "0.5",
            "--query_offset",
            "20",
            "--query_stride",
            "2",
            "--locability_prior_weight",
            "0.05",
            "--sparse_reprojection_error",
            "2",
            "--dense_reprojection_error",
            "12",
            "--landmark_selection",
            "per_view_spatial",
            "--landmark_source",
            "stdloc_detector",
            "--landmark_candidate_source",
            "all_gaussians",
            "--stdloc_detector_dir",
            "output/stdloc/map_cambridge_spgs/ShopFacade/detector",
            "--query_detector",
            "stdloc",
            "--query_feature_source",
            "original",
            "--eval_pose_source",
            "cameras_json",
            "--stdloc_detector_path",
            "output/stdloc/map_cambridge_spgs/ShopFacade/detector/30000_detector.pth",
            "--landmark_per_view_quota",
            "128",
            "--landmark_view_grid_size",
            "2",
            "--landmark_score_mode",
            "matchability",
            "--landmark_score_visibility_weight",
            "0.25",
            "--landmark_score_ambiguity_weight",
            "0.4",
            "--landmark_score_ambiguity_radius",
            "0.5",
            "--landmark_score_ambiguity_max_landmarks",
            "8192",
            "--landmark_score_keypoint_consensus_weight",
            "0.6",
            "--landmark_score_keypoint_consensus_radius",
            "2",
            "--landmark_score_keypoint_consensus_max_keypoints",
            "1024",
            "--landmark_score_keypoint_consensus_descriptor_weight",
            "0.35",
            "--landmark_score_legacy_keep_ratio",
            "0.75",
            "--landmark_score_spatial_grid_size",
            "4",
            "--landmark_score_prior_blend",
            "0.2",
            "--calibrated_matchability_path",
            "output/cache/calibrated/ShopFacade/matchability.pt",
            "--landmark_score_calibrated_matchability_weight",
            "0.7",
            "--match_calibrated_prior_weight",
            "0.2",
            "--match_filter_mode",
            "calibrated_coverage",
            "--match_filter_calibrated_score_weight",
            "0.3",
            "--match_filter_margin_weight",
            "0.4",
            "--match_filter_top_m",
            "1024",
            "--match_filter_image_grid_size",
            "12",
            "--match_filter_xyz_grid_size",
            "10",
            "--match_filter_max_per_image_cell",
            "4",
            "--match_filter_max_per_xyz_cell",
            "6",
            "--match_filter_min_matches",
            "64",
            "--sparse_match_filter_mode",
            "calibrated_coverage",
            "--dense_match_filter_mode",
            "none",
            "--sparse_match_filter_top_m",
            "1024",
            "--dense_match_filter_top_m",
            "0",
            "--pnp_hypotheses",
            "8",
            "--pnp_cluster_mode",
            "xyz_voxel",
            "--pnp_cluster_grid_size",
            "4",
            "--pnp_dense_verify_topk",
            "4",
            "--pnp_hypothesis_min_score_gain",
            "5.0",
            "--descriptor_source",
            "hybrid_ply_gated_residual",
            "--ply_loc_feature_weight",
            "0.75",
            "--hybrid_residual_alpha_max",
            "0.05",
            "--matchability_diagnostics",
            "--pnp_prefilter",
            "image_xyz_grid",
            "--sparse_pnp_max_matches",
            "512",
            "--dense_pnp_max_matches",
            "4096",
        ]
    )

    assert args.matcher == "stdloc_parity"
    assert args.sparse_matcher == "lightglue"
    assert args.dense_matcher == "lightglue_rendered"
    assert args.dim_pipeline == "superpoint+lightglue"
    assert args.rendered_keypoint_source == "locability"
    assert args.lightglue_max_keypoints == 512
    assert args.lightglue_filter_threshold == 0.05
    assert args.loftr_image_scale == 0.5
    assert args.loftr_min_confidence == 0.1
    assert args.loftr_max_matches == 1024
    assert args.dense_sparse_consistency_gate is True
    assert args.dense_sparse_consistency_max_median_ratio == 1.2
    assert args.dense_sparse_consistency_max_median_increase == 0.5
    assert args.dense_sparse_consistency_min_inlier_ratio_factor == 0.8
    assert args.dense_pose_delta_gate is True
    assert args.dense_pose_delta_max_trans_cm == 25
    assert args.dense_pose_delta_max_rot_deg == 2
    assert args.solver == "poselib"
    assert args.poselib_refine is True
    assert args.sparse_dual_softmax is True
    assert args.dense_dual_softmax_temp == 0.2
    assert args.fine_dual_softmax_temp == 0.3
    assert args.mnn is True
    assert args.subpixel_refine is True
    assert args.dense_query_pixel_center_offset == 0.5
    assert args.query_offset == 20
    assert args.query_stride == 2
    assert args.sparse_reprojection_error == 2
    assert args.dense_reprojection_error == 12
    assert args.landmark_source == "stdloc_detector"
    assert args.landmark_candidate_source == "all_gaussians"
    assert args.stdloc_detector_dir == "output/stdloc/map_cambridge_spgs/ShopFacade/detector"
    assert args.query_detector == "stdloc"
    assert args.query_feature_source == "original"
    assert args.eval_pose_source == "cameras_json"
    assert args.stdloc_detector_path == "output/stdloc/map_cambridge_spgs/ShopFacade/detector/30000_detector.pth"
    assert args.landmark_selection == "per_view_spatial"
    assert args.landmark_per_view_quota == 128
    assert args.landmark_view_grid_size == 2
    assert args.landmark_score_mode == "matchability"
    assert args.landmark_score_visibility_weight == 0.25
    assert args.landmark_score_ambiguity_weight == 0.4
    assert args.landmark_score_ambiguity_radius == 0.5
    assert args.landmark_score_ambiguity_max_landmarks == 8192
    assert args.landmark_score_keypoint_consensus_weight == 0.6
    assert args.landmark_score_keypoint_consensus_radius == 2
    assert args.landmark_score_keypoint_consensus_max_keypoints == 1024
    assert args.landmark_score_keypoint_consensus_descriptor_weight == 0.35
    assert args.landmark_score_legacy_keep_ratio == 0.75
    assert args.landmark_score_spatial_grid_size == 4
    assert args.landmark_score_prior_blend == 0.2
    assert args.calibrated_matchability_path == "output/cache/calibrated/ShopFacade/matchability.pt"
    assert args.landmark_score_calibrated_matchability_weight == 0.7
    assert args.match_calibrated_prior_weight == 0.2
    assert args.match_filter_mode == "calibrated_coverage"
    assert args.match_filter_calibrated_score_weight == 0.3
    assert args.match_filter_margin_weight == 0.4
    assert args.match_filter_top_m == 1024
    assert args.match_filter_image_grid_size == 12
    assert args.match_filter_xyz_grid_size == 10
    assert args.match_filter_max_per_image_cell == 4
    assert args.match_filter_max_per_xyz_cell == 6
    assert args.match_filter_min_matches == 64
    assert args.sparse_match_filter_mode == "calibrated_coverage"
    assert args.dense_match_filter_mode == "none"
    assert args.sparse_match_filter_top_m == 1024
    assert args.dense_match_filter_top_m == 0
    assert args.pnp_hypotheses == 8
    assert args.pnp_cluster_mode == "xyz_voxel"
    assert args.pnp_cluster_grid_size == 4
    assert args.pnp_dense_verify_topk == 4
    assert args.pnp_hypothesis_min_score_gain == 5.0
    assert args.descriptor_source == "hybrid_ply_gated_residual"
    assert args.ply_loc_feature_weight == 0.75
    assert args.hybrid_residual_alpha_max == 0.05
    assert args.matchability_diagnostics is True
    assert args.pnp_prefilter == "image_xyz_grid"
    assert args.sparse_pnp_max_matches == 512
    assert args.dense_pnp_max_matches == 4096

    config = effective_eval_config(args)
    assert config["sparse_reprojection_error"] == 2.0
    assert config["dense_reprojection_error"] == 12.0
    assert config["solver"] == "poselib"
    assert config["dense_full_render"] is False
    assert config["loftr_image_scale"] == 0.5
    assert config["loftr_min_confidence"] == 0.1
    assert config["loftr_max_matches"] == 1024
    assert config["dense_sparse_consistency_gate"] is True
    assert config["dense_sparse_consistency_max_median_ratio"] == 1.2
    assert config["dense_sparse_consistency_max_median_increase"] == 0.5
    assert config["dense_sparse_consistency_min_inlier_ratio_factor"] == 0.8
    assert config["dense_pose_delta_gate"] is True
    assert config["dense_pose_delta_max_trans_cm"] == 25.0
    assert config["dense_pose_delta_max_rot_deg"] == 2.0
    assert config["pnp_prefilter"] == "image_xyz_grid"
    assert config["sparse_pnp_max_matches"] == 512
    assert config["dense_pnp_max_matches"] == 4096
    assert config["match_filter_mode"] == "calibrated_coverage"
    assert config["match_calibrated_prior_weight"] == 0.2
    assert config["match_filter_top_m"] == 1024
    assert config["match_filter_margin_weight"] == 0.4
    assert config["match_filter_image_grid_size"] == 12
    assert config["match_filter_xyz_grid_size"] == 10
    assert config["match_filter_max_per_image_cell"] == 4
    assert config["match_filter_max_per_xyz_cell"] == 6
    assert config["sparse_match_filter_mode"] == "calibrated_coverage"
    assert config["dense_match_filter_mode"] == "none"
    assert config["sparse_match_filter_top_m"] == 1024
    assert config["dense_match_filter_top_m"] == 0
    assert config["pnp_hypotheses"] == 8
    assert config["pnp_cluster_mode"] == "xyz_voxel"
    assert config["pnp_dense_verify_topk"] == 4
    assert config["pnp_hypothesis_min_score_gain"] == 5.0
    assert config["hybrid_residual_alpha_max"] == 0.05
    assert config["match_filter_min_matches"] == 64


def test_eval_parser_defaults_are_baseline_preserving():
    args = build_argparser().parse_args(["--checkpoint", "output/model/latest.pth"])

    assert args.eval_split == "test"
    assert args.landmark_source == "stdloc_detector"
    assert args.descriptor_source == "ply_loc"
    assert args.query_detector == "stdloc"
    assert args.query_feature_source == "original"
    assert args.matcher == "stdloc_parity"

    config = effective_eval_config(args)
    assert config["eval_split"] == "test"


def test_detector_prior_for_all_gaussians_maps_sampled_scores_to_sampled_ids():
    scores = torch.tensor([0.2, 0.8], dtype=torch.float32)
    sampled_ids = torch.tensor([2, 4], dtype=torch.long)
    ids_all = torch.arange(6, dtype=torch.long)

    prior = eval_cambridge_hybrid._detector_prior_for_ids(
        scores,
        ids_all=ids_all,
        sampled_ids=sampled_ids,
        num_gaussians=6,
        device=torch.device("cpu"),
    )

    assert prior[4] > prior[2]
    assert prior[0] == 0.0
    assert prior[1] == 0.0
    assert prior.shape == (6,)


def test_calibrated_prior_maps_full_or_sampled_scores_to_candidate_ids():
    sampled_ids = torch.tensor([2, 4], dtype=torch.long)
    ids_all = torch.tensor([4, 2], dtype=torch.long)
    full_scores = torch.tensor([0.0, 0.0, 0.2, 0.0, 0.8, 0.0], dtype=torch.float32)

    prior = eval_cambridge_hybrid._calibrated_prior_for_ids(
        full_scores,
        ids_all=ids_all,
        sampled_ids=sampled_ids,
        num_gaussians=6,
        device=torch.device("cpu"),
    )

    assert prior[0] > prior[1]
    sampled_prior = eval_cambridge_hybrid._calibrated_prior_for_ids(
        torch.tensor([0.1, 0.9], dtype=torch.float32),
        ids_all=ids_all,
        sampled_ids=sampled_ids,
        num_gaussians=6,
        device=torch.device("cpu"),
    )
    assert sampled_prior[0] > sampled_prior[1]


def test_effective_eval_config_uses_global_reprojection_when_specific_values_absent():
    args = build_argparser().parse_args(
        [
            "--checkpoint",
            "output/model/latest.pth",
            "--matcher",
            "stdloc_parity",
            "--reprojection_error",
            "8",
        ]
    )

    config = effective_eval_config(args)

    assert config["sparse_matcher"] == "stdloc_parity"
    assert config["dense_matcher"] == "stdloc_parity"
    assert config["sparse_reprojection_error"] == 8.0
    assert config["dense_reprojection_error"] == 8.0


def test_parser_accepts_matchability_prior_mode():
    args = build_argparser().parse_args(
        [
            "--checkpoint",
            "output/model/latest.pth",
            "--landmark_score_mode",
            "matchability_prior",
        ]
    )

    assert args.landmark_score_mode == "matchability_prior"


def test_prepare_query_teacher_maps_resizes_detector_with_descriptor():
    desc = torch.randn(1, 4, 6, 8)
    det = torch.randn(1, 65, 6, 8)

    raw, desc_grid, det_grid = prepare_query_teacher_maps(desc, det, feature_height=3, feature_width=4)

    assert raw.shape == (4, 6, 8)
    assert desc_grid.shape == (4, 3, 4)
    assert det_grid.shape == (65, 3, 4)
    assert torch.allclose(desc_grid.norm(dim=0), torch.ones(3, 4), atol=1e-5)


def test_sparse_matchability_metrics_reports_gt_reprojection_inliers():
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = 100.0
    K[1, 1] = 100.0
    pose = np.eye(4, dtype=np.float64)
    xyz = np.array([[0.0, 0.0, 10.0], [0.1, 0.0, 10.0], [1.0, 0.0, 10.0]])
    query_yx = np.array([[0.0, 0.0], [0.0, 1.5], [0.0, 25.0]])

    metrics = sparse_matchability_metrics(
        query_yx,
        xyz,
        pose,
        K,
        scores=np.array([0.9, 0.8, 0.1]),
        margins=np.array([0.2, 0.1, 0.01]),
    )

    assert metrics["sparse_match_count"] == 3.0
    assert metrics["sparse_valid_match_count"] == 3.0
    assert metrics["sparse_inlier_2px"] == 2 / 3
    assert metrics["sparse_inlier_5px"] == 2 / 3
    assert metrics["sparse_inlier_8px"] == 2 / 3
    assert metrics["sparse_match_score_mean"] == np.mean([0.9, 0.8, 0.1])
    assert metrics["sparse_top2_margin_median"] == 0.1


def test_sparse_matchability_metrics_reports_pose_information_for_four_points():
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = 100.0
    K[1, 1] = 100.0
    pose = np.eye(4, dtype=np.float64)
    xyz = np.array(
        [
            [0.0, 0.0, 10.0],
            [0.2, 0.0, 10.0],
            [0.0, 0.2, 10.5],
            [0.2, 0.2, 11.0],
        ],
        dtype=np.float64,
    )
    query_yx = eval_cambridge_hybrid.project_world_points_yx_np(xyz, pose, K)

    metrics = sparse_matchability_metrics(query_yx, xyz, pose, K)

    assert np.isfinite(metrics["sparse_xyz_cov_logdet"])
    assert np.isfinite(metrics["sparse_all_pose_info_logdet"])
    assert metrics["sparse_inlier8_pose_info_min_eig"] > 0.0


def test_select_view_landmark_indices_enforces_spatial_diversity_before_topup():
    prior = torch.tensor([0.99, 0.98, 0.40, 0.39], dtype=torch.float32)
    flat_ids = torch.tensor([0, 1, 8, 9], dtype=torch.long)

    selected = select_view_landmark_indices(
        prior,
        flat_ids,
        height=4,
        width=4,
        quota=2,
        selection="per_view_spatial",
        grid_size=2,
    )

    assert selected.tolist() == [0, 2]


def test_select_pnp_match_indices_can_keep_spatially_diverse_high_scores():
    query_yx = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
            [10.0, 11.0],
        ],
        dtype=torch.float32,
    )
    points3d = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [10.0, 0.0, 1.0],
            [10.1, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.99, 0.98, 0.40, 0.39], dtype=torch.float32)

    score_only = select_pnp_match_indices(
        query_yx,
        points3d,
        scores=scores,
        max_matches=2,
        mode="score",
    )
    balanced = select_pnp_match_indices(
        query_yx,
        points3d,
        scores=scores,
        max_matches=2,
        mode="image_grid",
        image_grid_size=2,
    )

    assert score_only.tolist() == [0, 1]
    assert balanced.tolist() == [0, 2]


def test_local_geometric_consistency_prefers_neighborhood_preserving_matches():
    query_yx = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [100.0, 100.0],
        ],
        dtype=torch.float32,
    )
    points3d = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 1.0],
        ],
        dtype=torch.float32,
    )

    scores = local_geometric_consistency_scores(query_yx, points3d, k=3)

    assert scores[-1] < scores[:4].min()
    assert torch.all((scores >= 0.0) & (scores <= 1.0))


def test_select_pnp_match_indices_can_use_local_geometry_prefilter():
    query_yx = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [100.0, 100.0],
        ],
        dtype=torch.float32,
    )
    points3d = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 1.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.ones(5, dtype=torch.float32)

    keep = select_pnp_match_indices(
        query_yx,
        points3d,
        scores=scores,
        max_matches=4,
        mode="local_geometry",
    )

    assert 4 not in keep.tolist()


def test_image_pair_geometry_prefers_rendered_query_neighborhood_agreement():
    reference_yx = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    query_yx = torch.tensor(
        [
            [10.0, 10.0],
            [10.0, 12.0],
            [12.0, 10.0],
            [12.0, 12.0],
            [100.0, 100.0],
        ],
        dtype=torch.float32,
    )

    scores = local_image_pair_geometry_scores(query_yx, reference_yx, k=3)

    assert scores[-1] < scores[:4].min()
    assert torch.all((scores >= 0.0) & (scores <= 1.0))


def test_fuse_projected_teacher_descriptors_samples_visible_views():
    desc_maps = torch.zeros(1, 2, 3, 3, dtype=torch.float32)
    desc_maps[0, 0, 1, 1] = 1.0
    desc_maps[0, 1, 0, 0] = 1.0
    poses = torch.eye(4, dtype=torch.float32).view(1, 4, 4)
    K = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    points = torch.tensor([[0.0, 0.0, 1.0], [5.0, 5.0, 1.0]], dtype=torch.float32)

    fused, counts = fuse_projected_teacher_descriptors(
        points,
        poses,
        K,
        desc_maps,
        height=3,
        width=3,
        chunk_size=1,
    )

    assert torch.allclose(fused[0], torch.tensor([1.0, 0.0]))
    assert counts.tolist() == [1.0, 0.0]


def test_fuse_projected_teacher_descriptors_can_prefer_centered_observations():
    desc_maps = torch.zeros(2, 2, 3, 3, dtype=torch.float32)
    desc_maps[0, 0, 1, 1] = 1.0
    desc_maps[1, 1, 2, 2] = 1.0
    poses = torch.eye(4, dtype=torch.float32).view(1, 4, 4).repeat(2, 1, 1)
    poses[1, 0, 3] = 1.0
    poses[1, 1, 3] = 1.0
    K = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    points = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

    uniform, counts = fuse_projected_teacher_descriptors(
        points,
        poses,
        K,
        desc_maps,
        height=3,
        width=3,
        chunk_size=1,
    )
    centered, centered_counts = fuse_projected_teacher_descriptors(
        points,
        poses,
        K,
        desc_maps,
        height=3,
        width=3,
        chunk_size=1,
        centrality_power=4.0,
    )

    assert counts.tolist() == [2.0]
    assert centered_counts.tolist() == [2.0]
    assert torch.allclose(uniform[0], torch.nn.functional.normalize(torch.tensor([1.0, 1.0]), dim=0))
    assert centered[0, 0] > centered[0, 1]


def test_select_pnp_match_indices_can_use_image_pair_geometry_prefilter():
    reference_yx = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    query_yx = torch.tensor(
        [
            [10.0, 10.0],
            [10.0, 12.0],
            [12.0, 10.0],
            [12.0, 12.0],
            [100.0, 100.0],
        ],
        dtype=torch.float32,
    )
    points3d = torch.cat([reference_yx, torch.ones(5, 1)], dim=1)
    scores = torch.ones(5, dtype=torch.float32)

    keep = select_pnp_match_indices(
        query_yx,
        points3d,
        scores=scores,
        reference_yx=reference_yx,
        max_matches=4,
        mode="image_pair_geometry",
    )

    assert 4 not in keep.tolist()


def test_calibrated_coverage_filter_preserves_image_and_xyz_spread():
    query_yx = torch.tensor(
        [
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [9.1, 9.1],
            [9.2, 9.2],
            [9.3, 9.3],
        ],
        dtype=torch.float32,
    )
    points3d = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0],
            [0.2, 0.2, 0.0],
            [5.0, 5.0, 0.0],
            [5.1, 5.1, 0.0],
            [5.2, 5.2, 0.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.99, 0.98, 0.97, 0.7, 0.69, 0.68], dtype=torch.float32)

    keep = select_pnp_match_indices(
        query_yx,
        points3d,
        scores=scores,
        max_matches=4,
        mode="calibrated_coverage",
        image_grid_size=2,
        xyz_grid_size=2,
        max_per_image_cell=1,
        max_per_xyz_cell=1,
        min_matches=4,
    )

    assert keep.numel() == 4
    assert 0 in keep.tolist()
    assert any(idx in keep.tolist() for idx in (3, 4, 5))


def test_gated_residual_descriptor_blend_preserves_ply_at_zero_alpha():
    ply = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    hybrid = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    gate = torch.tensor([1.0, 0.0], dtype=torch.float32)

    zero = gated_residual_descriptor_blend(ply, hybrid, gate=gate, alpha_max=0.0)
    gated = gated_residual_descriptor_blend(ply, hybrid, gate=gate, alpha_max=1.0)

    assert torch.allclose(zero, torch.nn.functional.normalize(ply, dim=-1))
    assert gated[0, 1] > gated[0, 0]
    assert torch.allclose(gated[1], torch.nn.functional.normalize(ply, dim=-1)[1])


def test_match_filter_scores_can_use_descriptor_margin():
    raw = torch.tensor([0.90, 0.89], dtype=torch.float32)
    margin = torch.tensor([0.01, 0.30], dtype=torch.float32)

    scored = eval_cambridge_hybrid._match_filter_scores(
        raw,
        reliability=None,
        weight=0.0,
        margin=margin,
        margin_weight=1.0,
    )

    assert scored[1] > scored[0]


def test_match_filter_scores_can_use_calibrated_reliability_without_mutating_match_prior():
    scores = torch.zeros(3, dtype=torch.float32)
    detector_prior = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
    calibrated_reliability = torch.tensor([0.1, 0.9, 0.2], dtype=torch.float32)

    prior_scored = eval_cambridge_hybrid._match_filter_scores(scores, detector_prior, 1.0)
    calibrated_scored = eval_cambridge_hybrid._match_filter_scores(scores, calibrated_reliability, 1.0)

    assert torch.argmax(prior_scored).item() == 0
    assert torch.argmax(calibrated_scored).item() == 1


def test_calibrated_matchability_prior_uses_independent_logit_bias():
    corr = torch.zeros(1, 2, 3, dtype=torch.float32)
    reliability = torch.tensor([0.1, 0.8, 0.2], dtype=torch.float32)

    biased = eval_cambridge_hybrid.apply_calibrated_matchability_prior(
        corr,
        reliability,
        weight=0.5,
    )

    assert torch.argmax(biased[0, 0]).item() == 1
    assert torch.allclose(biased - biased.mean(dim=-1, keepdim=True), biased, atol=1e-6)


def test_generate_pnp_hypotheses_tries_clustered_subsets(monkeypatch):
    points = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.0, 0.1, 1.0],
            [0.1, 0.1, 1.0],
            [10.0, 10.0, 1.0],
            [10.1, 10.0, 1.0],
            [10.0, 10.1, 1.0],
            [10.1, 10.1, 1.0],
        ],
        dtype=torch.float32,
    )
    query = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.1],
            [0.1, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.0, 10.1],
            [10.1, 10.0],
            [10.1, 10.1],
        ],
        dtype=torch.float32,
    )
    calls = []

    def fake_solve(obj, img, *_args, **_kwargs):
        calls.append(obj.shape[0])
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = float(len(calls))
        return pose, int(obj.shape[0])

    monkeypatch.setattr(eval_cambridge_hybrid, "solve_pnp_ransac", fake_solve)

    hypotheses = generate_pnp_hypotheses(
        points,
        query,
        torch.eye(3),
        scores=torch.arange(8, dtype=torch.float32),
        max_hypotheses=3,
        cluster_mode="xyz_voxel",
        cluster_grid_size=2,
        reprojection_error=8.0,
        confidence=0.999,
        iterations=100,
        min_iterations=0,
        solver="opencv",
        refine_poselib=False,
    )

    assert calls[0] == 8
    assert any(count == 4 for count in calls[1:])
    assert len(hypotheses) >= 2
    assert hypotheses[0]["is_full"] in {True, False}


def test_select_verified_pnp_hypothesis_prefers_full_without_clear_gain():
    full_pose = np.eye(4, dtype=np.float32)
    cluster_pose = np.eye(4, dtype=np.float32)
    cluster_pose[0, 3] = 1.0
    hypotheses = [
        {"pose": cluster_pose, "score": 103.0, "inliers": 20, "is_full": False},
        {"pose": full_pose, "score": 100.0, "inliers": 18, "is_full": True},
    ]

    conservative = eval_cambridge_hybrid.select_verified_pnp_hypothesis(
        hypotheses,
        min_score_gain=5.0,
    )
    permissive = eval_cambridge_hybrid.select_verified_pnp_hypothesis(
        hypotheses,
        min_score_gain=1.0,
    )

    assert conservative["is_full"] is True
    assert permissive["is_full"] is False


def test_descriptor_top2_margin_uses_prior_adjusted_scores():
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    landmarks = torch.tensor([[1.0, 0.0], [0.99, 0.1]], dtype=torch.float32)
    prior = torch.tensor([0.0, 1.0], dtype=torch.float32)

    no_prior = eval_cambridge_hybrid.descriptor_top2_margin(query, landmarks)
    with_prior = eval_cambridge_hybrid.descriptor_top2_margin(
        query,
        landmarks,
        landmark_prior=prior,
        prior_weight=0.2,
    )

    assert with_prior[0] > no_prior[0]


def test_sparse_consistency_gate_rejects_pose_that_breaks_sparse_reprojection():
    sparse_xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    sparse_query_yx = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    previous = torch.eye(4).unsqueeze(0)
    candidate = torch.eye(4).unsqueeze(0)
    candidate[0, 0, 3] = 10.0

    reject, stats = eval_cambridge_hybrid.should_reject_dense_pose_by_sparse_consistency(
        previous,
        candidate,
        sparse_xyz,
        sparse_query_yx,
        torch.eye(3),
        threshold=2.0,
        max_median_ratio=1.5,
        max_median_increase=0.5,
        min_inlier_ratio_factor=0.75,
    )

    assert reject is True
    assert stats["candidate_sparse_reproj_median"] > stats["prev_sparse_reproj_median"]


def test_dense_pose_delta_gate_rejects_large_pose_jump():
    previous = torch.eye(4).unsqueeze(0)
    candidate = torch.eye(4).unsqueeze(0)
    candidate[0, 0, 3] = 2.0

    reject, stats = eval_cambridge_hybrid.should_reject_dense_pose_by_delta(
        previous,
        candidate,
        max_trans_cm=50.0,
        max_rot_deg=5.0,
    )

    assert reject is True
    assert stats["dense_pose_delta_trans_cm"] > 50.0


def test_localize_one_stdloc_dense_branch_unprojects_dense_matches(monkeypatch):
    args = SimpleNamespace(
        matcher="stdloc_parity",
        sparse_matcher="topk",
        dense_matcher="stdloc_parity",
        query_keypoints=4,
        keypoint_threshold=0.0,
        nms_radius=0,
        sparse_match_threshold=0.0,
        dense_match_threshold=0.0,
        locability_prior_weight=0.0,
        dense_iters=1,
        solver="opencv",
        sparse_dual_softmax=False,
        sparse_dual_softmax_temp=0.1,
        dense_dual_softmax_temp=0.1,
        fine_dual_softmax_temp=0.1,
        mnn=True,
        subpixel_refine=True,
        subpixel_temperature=0.1,
        topk_refine_window=1,
        render_pixel_center_offset=0.0,
        dense_query_pixel_center_offset=0.0,
        dense_full_render=False,
        match_second_best_margin=0.0,
        sparse_match_second_best_margin=None,
        dense_match_second_best_margin=None,
        reprojection_error=2.0,
        sparse_reprojection_error=None,
        dense_reprojection_error=None,
        refine_reprojection_error=0.0,
        pnp_confidence=0.9999,
        pnp_iterations=100,
    )
    keypoints = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    query_desc = torch.eye(4, dtype=torch.float32)
    landmark_xyz = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    landmark_desc = query_desc.clone()
    landmark_prior = torch.ones(4, dtype=torch.float32)
    dense_result = DenseMatchResult(
        query_yx=keypoints * 8.0,
        rendered_yx=keypoints * 8.0,
        scores=torch.ones(4, dtype=torch.float32),
        coarse_query_ids=torch.arange(4),
        coarse_rendered_ids=torch.arange(4),
    )

    class DummyTeacher:
        def __call__(self, _gray):
            return torch.ones(1, 4, 2, 2), torch.zeros(1, 65, 2, 2)

    monkeypatch.setattr(
        eval_cambridge_hybrid,
        "extract_keypoints_from_detector_logits",
        lambda *_args, **_kwargs: (keypoints, torch.ones(4)),
    )
    monkeypatch.setattr(
        eval_cambridge_hybrid,
        "sample_descriptors_bilinear",
        lambda *_args, **_kwargs: query_desc,
    )
    monkeypatch.setattr(
        eval_cambridge_hybrid,
        "render_hybrid_superpoint",
        lambda *_args, **_kwargs: {
            "descriptor": torch.ones(1, 4, 2, 2),
            "depth": torch.ones(1, 2, 2),
            "locability": None,
        },
    )
    monkeypatch.setattr(
        eval_cambridge_hybrid,
        "coarse_to_fine_dense_matches",
        lambda *_args, **_kwargs: dense_result,
    )
    calls = {"unproject": 0}

    def fake_unproject(*_args, **_kwargs):
        calls["unproject"] += 1
        return landmark_xyz

    monkeypatch.setattr(eval_cambridge_hybrid, "unproject_positions_yx", fake_unproject)
    poses = [np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)]

    def fake_solve_pnp(*_args, **_kwargs):
        return poses.pop(0), 4

    monkeypatch.setattr(eval_cambridge_hybrid, "solve_pnp_ransac", fake_solve_pnp)

    result = eval_cambridge_hybrid.localize_one(
        model=object(),
        sp_head=object(),
        renderer=SimpleNamespace(K=torch.eye(3)),
        teacher=DummyTeacher(),
        rgb=torch.zeros(1, 3, 16, 16),
        K_feature=torch.eye(3),
        K_full=torch.eye(3),
        landmark_xyz=landmark_xyz,
        landmark_desc=landmark_desc,
        landmark_prior=landmark_prior,
        args=args,
    )

    assert calls["unproject"] == 1
    assert result["dense_inliers"] == 4


def test_localize_one_lightglue_rendered_branch_uses_rendered_sparse_matches(monkeypatch):
    args = SimpleNamespace(
        matcher="topk",
        sparse_matcher="topk",
        dense_matcher="lightglue_rendered",
        dim_pipeline="superpoint+lightglue",
        rendered_keypoint_source="locability",
        lightglue_max_keypoints=4,
        lightglue_filter_threshold=0.1,
        query_keypoints=4,
        keypoint_threshold=0.0,
        nms_radius=0,
        sparse_match_threshold=0.0,
        dense_match_threshold=0.0,
        locability_prior_weight=0.0,
        dense_iters=1,
        solver="opencv",
        sparse_dual_softmax=False,
        sparse_dual_softmax_temp=0.1,
        dense_dual_softmax_temp=0.1,
        fine_dual_softmax_temp=0.1,
        mnn=True,
        subpixel_refine=False,
        subpixel_temperature=0.1,
        topk_refine_window=1,
        render_pixel_center_offset=0.0,
        dense_query_pixel_center_offset=0.0,
        dense_full_render=True,
        match_second_best_margin=0.0,
        sparse_match_second_best_margin=None,
        dense_match_second_best_margin=None,
        reprojection_error=2.0,
        sparse_reprojection_error=None,
        dense_reprojection_error=None,
        refine_reprojection_error=0.0,
        pnp_confidence=0.9999,
        pnp_iterations=100,
        pnp_min_iterations=0,
        sparse_pnp_iterations=None,
        sparse_pnp_min_iterations=None,
        dense_pnp_iterations=None,
        dense_pnp_min_iterations=None,
        query_detector="superpoint",
    )
    keypoints = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    query_desc = torch.eye(4, dtype=torch.float32)
    landmark_xyz = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=torch.float32,
    )

    class DummyTeacher:
        def __call__(self, _gray):
            return torch.ones(1, 4, 2, 2), torch.zeros(1, 65, 2, 2)

    locability = torch.zeros(1, 1, 16, 16)
    locability[0, 0, 0, 0] = 1.0
    locability[0, 0, 0, 8] = 0.9
    locability[0, 0, 8, 0] = 0.8
    locability[0, 0, 8, 8] = 0.7

    monkeypatch.setattr(
        eval_cambridge_hybrid,
        "extract_keypoints_from_detector_logits",
        lambda *_args, **_kwargs: (keypoints, torch.ones(4)),
    )
    monkeypatch.setattr(
        eval_cambridge_hybrid,
        "sample_descriptors_bilinear",
        lambda descriptor_map, sample_yx: query_desc[: sample_yx.shape[0]],
    )
    monkeypatch.setattr(
        eval_cambridge_hybrid,
        "render_hybrid_superpoint",
        lambda *_args, **_kwargs: {
            "descriptor": torch.ones(1, 4, 16, 16),
            "detector": torch.zeros(1, 65, 16, 16),
            "depth": torch.ones(1, 16, 16),
            "locability": locability,
        },
    )
    monkeypatch.setattr(
        eval_cambridge_hybrid,
        "match_lightglue_descriptors",
        lambda *_args, **_kwargs: (torch.arange(4), torch.arange(4), torch.ones(4)),
    )
    poses = [np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)]

    def fake_solve_pnp(*_args, **_kwargs):
        return poses.pop(0), 4

    monkeypatch.setattr(eval_cambridge_hybrid, "solve_pnp_ransac", fake_solve_pnp)

    result = eval_cambridge_hybrid.localize_one(
        model=object(),
        sp_head=object(),
        renderer=SimpleNamespace(K=torch.eye(3)),
        full_renderer=SimpleNamespace(K=torch.eye(3)),
        teacher=DummyTeacher(),
        rgb=torch.zeros(1, 3, 16, 16),
        K_feature=torch.eye(3),
        K_full=torch.eye(3),
        landmark_xyz=landmark_xyz,
        landmark_desc=query_desc,
        landmark_prior=torch.ones(4),
        args=args,
    )

    assert result["dense_inliers"] == 4


def test_localize_one_loftr_rendered_branch_matches_rendered_rgb_to_depth(monkeypatch):
    args = SimpleNamespace(
        matcher="topk",
        sparse_matcher="topk",
        dense_matcher="loftr_rendered",
        dim_pipeline="loftr",
        query_keypoints=4,
        keypoint_threshold=0.0,
        nms_radius=0,
        sparse_match_threshold=0.0,
        dense_match_threshold=0.0,
        locability_prior_weight=0.0,
        dense_iters=1,
        solver="opencv",
        sparse_dual_softmax=False,
        sparse_dual_softmax_temp=0.1,
        dense_dual_softmax_temp=0.1,
        fine_dual_softmax_temp=0.1,
        mnn=True,
        subpixel_refine=False,
        subpixel_temperature=0.1,
        topk_refine_window=1,
        render_pixel_center_offset=0.0,
        dense_query_pixel_center_offset=0.0,
        dense_full_render=True,
        loftr_pretrained="outdoor",
        loftr_image_scale=0.5,
        loftr_min_confidence=0.1,
        loftr_max_matches=4,
        match_second_best_margin=0.0,
        sparse_match_second_best_margin=None,
        dense_match_second_best_margin=None,
        reprojection_error=2.0,
        sparse_reprojection_error=None,
        dense_reprojection_error=None,
        refine_reprojection_error=0.0,
        pnp_confidence=0.9999,
        pnp_iterations=100,
        pnp_min_iterations=0,
        sparse_pnp_iterations=None,
        sparse_pnp_min_iterations=None,
        dense_pnp_iterations=None,
        dense_pnp_min_iterations=None,
        query_detector="superpoint",
    )
    keypoints = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    query_desc = torch.eye(4, dtype=torch.float32)
    landmark_xyz = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=torch.float32,
    )

    class DummyTeacher:
        def __call__(self, _gray):
            return torch.ones(1, 4, 2, 2), torch.zeros(1, 65, 2, 2)

    monkeypatch.setattr(
        eval_cambridge_hybrid,
        "extract_keypoints_from_detector_logits",
        lambda *_args, **_kwargs: (keypoints, torch.ones(4)),
    )
    monkeypatch.setattr(
        eval_cambridge_hybrid,
        "sample_descriptors_bilinear",
        lambda descriptor_map, sample_yx: query_desc[: sample_yx.shape[0]],
    )
    monkeypatch.setattr(
        eval_cambridge_hybrid,
        "render_hybrid_superpoint",
        lambda *_args, **_kwargs: {
            "descriptor": torch.ones(1, 4, 16, 16),
            "detector": torch.zeros(1, 65, 16, 16),
            "depth": torch.ones(1, 16, 16),
            "locability": torch.ones(1, 1, 16, 16),
            "rgb": torch.zeros(1, 3, 16, 16),
        },
    )
    loftr_calls = {"count": 0}

    def fake_match_loftr(query_rgb, rendered_rgb, **kwargs):
        loftr_calls["count"] += 1
        assert query_rgb.shape == (1, 3, 16, 16)
        assert rendered_rgb.shape == (1, 3, 16, 16)
        assert kwargs["image_scale"] == 0.5
        return keypoints * 8.0, keypoints * 8.0, torch.tensor([0.9, 0.8, 0.7, 0.6])

    monkeypatch.setattr(eval_cambridge_hybrid, "match_loftr_images", fake_match_loftr)
    monkeypatch.setattr(eval_cambridge_hybrid, "unproject_positions_yx", lambda *_args, **_kwargs: landmark_xyz)
    poses = [np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)]

    def fake_solve_pnp(*_args, **_kwargs):
        return poses.pop(0), 4

    monkeypatch.setattr(eval_cambridge_hybrid, "solve_pnp_ransac", fake_solve_pnp)

    result = eval_cambridge_hybrid.localize_one(
        model=object(),
        sp_head=object(),
        renderer=SimpleNamespace(K=torch.eye(3)),
        full_renderer=SimpleNamespace(K=torch.eye(3)),
        teacher=DummyTeacher(),
        rgb=torch.zeros(1, 3, 16, 16),
        K_feature=torch.eye(3),
        K_full=torch.eye(3),
        landmark_xyz=landmark_xyz,
        landmark_desc=query_desc,
        landmark_prior=torch.ones(4),
        args=args,
    )

    assert loftr_calls["count"] == 1
    assert result["dense_inliers"] == 4
