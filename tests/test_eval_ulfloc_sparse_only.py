import json

import pytest
import torch

from loc_gs.scripts.eval_ulfloc_sparse_only import (
    build_sparse_only_split_audit,
    build_argparser,
    load_landmark_prior_weights,
    resize_mask_to_image,
    serialize_sparse_match_attributions,
    summarize_sparse_match_diagnostics,
    summarize_sparse_metrics,
)


def test_summarize_sparse_metrics_reports_sparse_pose_stats():
    summary = summarize_sparse_metrics(
        translation_cm=[2.0, 10.0, 50.0],
        rotation_deg=[0.5, 1.5, 6.0],
        inliers=[100, 50, 10],
    )

    assert summary["median_te_cm"] == pytest.approx(10.0)
    assert summary["median_re_deg"] == pytest.approx(1.5)
    assert summary["recall_50cm_5d"] == pytest.approx(2 / 3)
    assert summary["recall_10cm_5d"] == pytest.approx(2 / 3)
    assert summary["recall_5cm_5d"] == pytest.approx(1 / 3)
    assert summary["mean_inliers"] == pytest.approx(160 / 3)


def test_summarize_sparse_metrics_handles_empty_input():
    summary = summarize_sparse_metrics(translation_cm=[], rotation_deg=[], inliers=[])

    assert summary["query_count"] == 0
    assert summary["median_te_cm"] is None


def test_serialize_sparse_match_attributions_exports_match_level_solver_trace():
    attributions = serialize_sparse_match_attributions(
        {
            "matched_gaussian_ids": torch.tensor([10, 11]),
            "pnp_inlier_mask": torch.tensor([True, False]),
            "descriptor_score": torch.tensor([0.9, 0.8]),
            "detector_score": torch.tensor([0.7, 0.6]),
            "reprojection_error_px": torch.tensor([1.0, 12.0]),
            "query_xy": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        }
    )

    assert attributions == [
        {
            "matched_gaussian_id": 10,
            "pnp_inlier": True,
            "descriptor_score": pytest.approx(0.9),
            "detector_score": pytest.approx(0.7),
            "reprojection_error_px": pytest.approx(1.0),
            "query_xy": [pytest.approx(5.0), pytest.approx(6.0)],
        },
        {
            "matched_gaussian_id": 11,
            "pnp_inlier": False,
            "descriptor_score": pytest.approx(0.8),
            "detector_score": pytest.approx(0.6),
            "reprojection_error_px": pytest.approx(12.0),
            "query_xy": [pytest.approx(7.0), pytest.approx(8.0)],
        },
    ]


def test_build_sparse_only_split_audit_explicitly_marks_non_test_eval():
    audit = build_sparse_only_split_audit("train_dev_seed13_20p_sparse_validation")

    assert audit["official_test_eval"] is False
    assert audit["official_test_used"] is False
    assert audit["test_split_used"] is False
    assert audit["paper_safe_for_tuning"] is True


def test_resize_mask_to_image_matches_query_resolution():
    mask = torch.ones(1, 4, 5, dtype=torch.bool)
    resized = resize_mask_to_image(mask, height=8, width=10)

    assert resized.shape == (1, 8, 10)
    assert resized.dtype == torch.bool


def test_eval_ulfloc_sparse_only_parser_accepts_scene_detector_rerank_options():
    args = build_argparser().parse_args(
        [
            "--source_path",
            "/src",
            "--model_path",
            "/model",
            "--input_log_dir",
            "/model/log",
            "--cfg",
            "/model/log/config.yaml",
            "--output_dir",
            "/out",
            "--scene_detector_checkpoint",
            "/detector.pth",
            "--scene_detector_blend_alpha",
            "0.25",
            "--scene_detector_candidate_top_k",
            "3072",
            "--scene_detector_native_keep_fraction",
            "0.5",
            "--scene_matcher_checkpoint",
            "/matcher.pt",
            "--scene_matcher_topk",
            "4",
            "--scene_matcher_score_weight",
            "0.2",
            "--scene_matcher_score_mode",
            "candidate",
            "--scene_matcher_max_descriptor_margin",
            "0.05",
            "--scene_matcher_drop_dustbin",
            "--scene_matcher_min_keep",
            "256",
        ]
    )

    assert args.scene_detector_blend_alpha == pytest.approx(0.25)
    assert args.scene_detector_candidate_top_k == 3072
    assert args.scene_detector_native_keep_fraction == pytest.approx(0.5)
    assert str(args.scene_matcher_checkpoint) == "/matcher.pt"
    assert args.scene_matcher_topk == 4
    assert args.scene_matcher_score_weight == pytest.approx(0.2)
    assert args.scene_matcher_score_mode == "candidate"
    assert args.scene_matcher_max_descriptor_margin == pytest.approx(0.05)
    assert args.scene_matcher_drop_dustbin is True
    assert args.scene_matcher_min_keep == 256


def test_eval_ulfloc_sparse_only_parser_accepts_sparse_match_filter_options():
    args = build_argparser().parse_args(
        [
            "--source_path",
            "/src",
            "--model_path",
            "/model",
            "--input_log_dir",
            "/model/log",
            "--cfg",
            "/model/log/config.yaml",
            "--output_dir",
            "/out",
            "--sparse_match_filter_mode",
            "precision",
            "--sparse_match_filter_min_query_margin",
            "0.03",
            "--sparse_match_filter_top_m",
            "1024",
            "--sparse_match_filter_min_keep",
            "128",
            "--sparse_match_filter_detector_score_weight",
            "0.1",
            "--sparse_match_filter_landmark_prior",
            "/feedback/targets.json",
            "--sparse_match_filter_landmark_prior_weight",
            "0.4",
            "--sparse_match_filter_landmark_prior_pre_topk_weight",
            "0.03",
            "--sparse_match_filter_landmark_prior_center",
            "1.0",
            "--sparse_match_filter_landmark_prior_scale",
            "0.5",
            "--sparse_match_filter_landmark_prior_clip",
            "1.5",
            "--sparse_match_filter_image_grid_size",
            "8",
            "--sparse_match_filter_max_per_image_cell",
            "24",
            "--sparse_match_filter_unique_landmark",
        ]
    )

    assert args.sparse_match_filter_mode == "precision"
    assert args.sparse_match_filter_min_query_margin == pytest.approx(0.03)
    assert args.sparse_match_filter_top_m == 1024
    assert args.sparse_match_filter_min_keep == 128
    assert args.sparse_match_filter_detector_score_weight == pytest.approx(0.1)
    assert str(args.sparse_match_filter_landmark_prior) == "/feedback/targets.json"
    assert args.sparse_match_filter_landmark_prior_weight == pytest.approx(0.4)
    assert args.sparse_match_filter_landmark_prior_pre_topk_weight == pytest.approx(0.03)
    assert args.sparse_match_filter_landmark_prior_center == pytest.approx(1.0)
    assert args.sparse_match_filter_landmark_prior_scale == pytest.approx(0.5)
    assert args.sparse_match_filter_landmark_prior_clip == pytest.approx(1.5)
    assert args.sparse_match_filter_image_grid_size == 8
    assert args.sparse_match_filter_max_per_image_cell == 24
    assert args.sparse_match_filter_unique_landmark is True


def test_load_landmark_prior_weights_reads_correspondence_targets(tmp_path):
    targets = {
        "descriptor_fusion": {
            "landmark_weights": {
                "2": 0.75,
                "5": 0.25,
            }
        }
    }
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(targets), encoding="utf-8")

    weights = load_landmark_prior_weights(path)

    assert weights.tolist() == pytest.approx([0.0, 0.0, 0.75, 0.0, 0.0, 0.25])


def test_load_landmark_prior_weights_reads_solver_feedback_pickle(tmp_path):
    path = tmp_path / "solver_feedback.pkl"
    torch.save(
        {
            "split_name": "selfmap_train",
            "landmark_weights": torch.tensor([1.0, 1.5, 0.25], dtype=torch.float32),
        },
        path,
    )

    weights = load_landmark_prior_weights(path)

    assert weights.tolist() == pytest.approx([1.0, 1.5, 0.25])


def test_summarize_sparse_match_diagnostics_reports_inlier_outlier_quality():
    diagnostics = summarize_sparse_match_diagnostics(
        {
            "matched_gaussian_ids": [4, 4, 9],
            "descriptor_score": [0.2, 0.8, 0.4],
            "detector_score": [0.1, 0.9, 0.3],
            "pnp_inlier_mask": [False, True, True],
            "reprojection_error_px": [12.0, 1.0, 3.0],
            "query_xy": [[10.0, 20.0], [90.0, 20.0], [50.0, 80.0]],
            "image_size": [100, 100],
        }
    )

    assert diagnostics["total_match_count"] == 3
    assert diagnostics["unique_landmark_count"] == 2
    assert diagnostics["pnp_inlier_count"] == 2
    assert diagnostics["descriptor_score_mean"] == pytest.approx((0.2 + 0.8 + 0.4) / 3)
    assert diagnostics["descriptor_score_inlier_mean"] == pytest.approx(0.6)
    assert diagnostics["descriptor_score_outlier_mean"] == pytest.approx(0.2)
    assert diagnostics["detector_score_inlier_mean"] == pytest.approx(0.6)
    assert diagnostics["reprojection_error_inlier_median_px"] == pytest.approx(2.0)
    assert diagnostics["query_bbox_area_norm"] == pytest.approx(0.48)
