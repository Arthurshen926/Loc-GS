import pickle
from types import SimpleNamespace

import numpy as np
import torch

from loc_gs.localization.stdloc_parity import DenseMatchResult
from loc_gs.scripts import eval_cambridge_hybrid
from loc_gs.scripts.eval_cambridge_hybrid import (
    build_argparser,
    effective_eval_config,
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
            "--landmark_score_legacy_keep_ratio",
            "0.75",
            "--landmark_score_spatial_grid_size",
            "4",
            "--landmark_score_prior_blend",
            "0.2",
            "--descriptor_source",
            "hybrid_ply_blend",
            "--ply_loc_feature_weight",
            "0.75",
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
    assert args.landmark_score_legacy_keep_ratio == 0.75
    assert args.landmark_score_spatial_grid_size == 4
    assert args.landmark_score_prior_blend == 0.2
    assert args.descriptor_source == "hybrid_ply_blend"
    assert args.ply_loc_feature_weight == 0.75
    assert args.matchability_diagnostics is True
    assert args.pnp_prefilter == "image_xyz_grid"
    assert args.sparse_pnp_max_matches == 512
    assert args.dense_pnp_max_matches == 4096

    config = effective_eval_config(args)
    assert config["sparse_reprojection_error"] == 2.0
    assert config["dense_reprojection_error"] == 12.0
    assert config["solver"] == "poselib"
    assert config["dense_full_render"] is False
    assert config["pnp_prefilter"] == "image_xyz_grid"
    assert config["sparse_pnp_max_matches"] == 512
    assert config["dense_pnp_max_matches"] == 4096


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
