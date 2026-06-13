import json

import pytest
import torch

from loc_gs.feedback.sparse_feedback_v4 import build_sparse_feedback_v4
from loc_gs.feedback.solver_feedback_impact import build_solver_feedback_impact
from loc_gs.scripts.eval_ulfloc_sparse_only import (
    build_sparse_feedback_query_row,
    build_sparse_feedback_trace_payload,
    build_scene_detector_eval_config,
    build_sparse_only_split_audit,
    build_argparser,
    compact_sparse_attributions_for_result_json,
    load_landmark_activation_audit,
    load_landmark_prior_weights,
    resize_mask_to_image,
    serialize_sparse_match_attributions,
    sparse_trace_payload_for_json,
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
            "matched_sampled_indices": torch.tensor([0, 1]),
            "matched_active_indices": torch.tensor([20, 21]),
            "query_keypoint_indices": torch.tensor([3, 4]),
            "pnp_inlier_mask": torch.tensor([True, False]),
            "descriptor_score": torch.tensor([0.9, 0.8]),
            "descriptor_margin": torch.tensor([0.2, 0.05]),
            "descriptor_margin_source": "descriptor_score_minus_best_nonselected",
            "detector_score": torch.tensor([0.7, 0.6]),
            "reprojection_error_px": torch.tensor([1.0, 12.0]),
            "query_xy": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            "camera_xyz": torch.tensor([[0.1, 0.2, 2.0], [0.2, 0.1, 3.0]]),
            "depth_m": torch.tensor([2.0, 3.0]),
            "bearing": torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 0.99]]),
            "query_descriptor": torch.tensor([[0.25, 0.75], [0.4, 0.6]]),
            "landmark_descriptor": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "image_size": (100, 50),
        },
        scene="ShopFacade",
        split_name="selfmap_train",
        query_id="q1",
        image_id="im1",
    )

    assert attributions == [
        {
            "matched_gaussian_id": 10,
            "gaussian_id": 10,
            "landmark_id": 10,
            "matched_sampled_index": 0,
            "sampled_row": 0,
            "matched_active_index": 20,
            "query_keypoint_index": 3,
            "match_row_index": 0,
            "pnp_inlier": True,
            "label": 1,
            "descriptor_score": pytest.approx(0.9),
            "descriptor_margin": pytest.approx(0.2),
            "descriptor_margin_source": "descriptor_score_minus_best_nonselected",
            "detector_score": pytest.approx(0.7),
            "reprojection_error_px": pytest.approx(1.0),
            "query_xy": [pytest.approx(5.0), pytest.approx(6.0)],
            "keypoint_xy": [pytest.approx(5.0), pytest.approx(6.0)],
            "query_xy_norm": [pytest.approx(0.05), pytest.approx(0.12)],
            "image_cell": 0,
            "camera_xyz": [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(2.0)],
            "depth_m": pytest.approx(2.0),
            "bearing": [pytest.approx(0.0), pytest.approx(0.0), pytest.approx(1.0)],
            "query_descriptor": [pytest.approx(0.25), pytest.approx(0.75)],
            "landmark_descriptor": [pytest.approx(1.0), pytest.approx(0.0)],
            "scene": "ShopFacade",
            "split_name": "selfmap_train",
            "query_id": "q1",
            "image_id": "im1",
            "source_role": "baseline_trace",
            "active_index_space": "active_landmark_rows",
            "sampled_row_space": "sampled_landmark_rows",
            "gaussian_id_space": "full_gaussian_ids",
        },
        {
            "matched_gaussian_id": 11,
            "gaussian_id": 11,
            "landmark_id": 11,
            "matched_sampled_index": 1,
            "sampled_row": 1,
            "matched_active_index": 21,
            "query_keypoint_index": 4,
            "match_row_index": 1,
            "pnp_inlier": False,
            "label": 0,
            "descriptor_score": pytest.approx(0.8),
            "descriptor_margin": pytest.approx(0.05),
            "descriptor_margin_source": "descriptor_score_minus_best_nonselected",
            "detector_score": pytest.approx(0.6),
            "reprojection_error_px": pytest.approx(12.0),
            "query_xy": [pytest.approx(7.0), pytest.approx(8.0)],
            "keypoint_xy": [pytest.approx(7.0), pytest.approx(8.0)],
            "query_xy_norm": [pytest.approx(0.07), pytest.approx(0.16)],
            "image_cell": 8,
            "camera_xyz": [pytest.approx(0.2), pytest.approx(0.1), pytest.approx(3.0)],
            "depth_m": pytest.approx(3.0),
            "bearing": [pytest.approx(0.1), pytest.approx(0.0), pytest.approx(0.99)],
            "query_descriptor": [pytest.approx(0.4), pytest.approx(0.6)],
            "landmark_descriptor": [pytest.approx(0.0), pytest.approx(1.0)],
            "scene": "ShopFacade",
            "split_name": "selfmap_train",
            "query_id": "q1",
            "image_id": "im1",
            "source_role": "baseline_trace",
            "active_index_space": "active_landmark_rows",
            "sampled_row_space": "sampled_landmark_rows",
            "gaussian_id_space": "full_gaussian_ids",
        },
    ]

    impact = build_solver_feedback_impact({"split_name": "train_selfmap", "records": attributions})
    assert impact["landmark_positive"][10]["support"] > 0.0
    assert impact["landmark_negative"][11]["risk"] > 0.0


def test_sparse_feedback_trace_rows_feed_v4_builder_without_side_channels():
    query_row = build_sparse_feedback_query_row(
        scene="ShopFacade",
        split_name="selfmap_train",
        query_id="q1",
        image_id="im1",
        sparse_te_cm=3.0,
        sparse_re_deg=0.1,
        matches={
            "pnp_success": True,
            "sparse_inlier_count": 1,
            "total_match_count": 1,
            "pnp_inlier_mask": torch.tensor([True]),
            "query_xy": torch.tensor([[5.0, 6.0]]),
            "image_size": (100, 50),
            "depth_m": torch.tensor([2.0]),
            "bearing": torch.tensor([[0.0, 0.0, 1.0]]),
        },
    )
    correspondence_rows = serialize_sparse_match_attributions(
        {
            "matched_gaussian_ids": torch.tensor([10]),
            "matched_sampled_indices": torch.tensor([0]),
            "query_keypoint_indices": torch.tensor([3]),
            "pnp_inlier_mask": torch.tensor([True]),
            "descriptor_score": torch.tensor([0.9]),
            "descriptor_margin": torch.tensor([0.2]),
            "detector_score": torch.tensor([0.7]),
            "reprojection_error_px": torch.tensor([1.0]),
            "query_xy": torch.tensor([[5.0, 6.0]]),
            "camera_xyz": torch.tensor([[0.1, 0.2, 2.0]]),
            "depth_m": torch.tensor([2.0]),
            "bearing": torch.tensor([[0.0, 0.0, 1.0]]),
            "query_descriptor": torch.tensor([[0.25, 0.75]]),
            "landmark_descriptor": torch.tensor([[1.0, 0.0]]),
            "image_size": (100, 50),
        },
        scene="ShopFacade",
        split_name="selfmap_train",
        query_id="q1",
        image_id="im1",
    )
    payload = build_sparse_feedback_trace_payload(
        scene="ShopFacade",
        split_name="selfmap_train",
        query_rows=[query_row],
        correspondence_rows=correspondence_rows,
        query_features={},
        query_match_descriptors={},
        split_audit={"test_split_used": False, "official_test_used": False},
    )

    built = build_sparse_feedback_v4(payload)

    assert built["metrics"]["protected_support_count"] == 1
    assert built["correspondences"][0]["query_descriptor"] == [pytest.approx(0.25), pytest.approx(0.75)]


def test_sparse_trace_payload_for_json_omits_per_correspondence_descriptors():
    compact = sparse_trace_payload_for_json(
        {
            "schema_version": "trace",
            "correspondences": [
                {
                    "query_descriptor": [0.1, 0.2, 0.3],
                    "landmark_descriptor": [0.3, 0.2, 0.1],
                    "pnp_inlier": True,
                }
            ],
            "query_match_descriptors": {"q1": torch.randn(2, 3)},
        }
    )

    assert compact["correspondences"] == [{"pnp_inlier": True}]
    assert compact["correspondence_descriptor_count"] == 1
    assert compact["query_descriptor_dim"] == 3
    assert compact["landmark_descriptor_dim"] == 3
    assert compact["correspondence_descriptors_omitted_from_json"] is True


def test_compact_sparse_attributions_for_result_json_omits_large_descriptors():
    compact = compact_sparse_attributions_for_result_json(
        [
            {
                "pnp_inlier": True,
                "query_descriptor": [0.1, 0.2, 0.3],
                "landmark_descriptor": [0.3, 0.2, 0.1],
            },
            {
                "pnp_inlier": False,
                "query_descriptor": [0.4, 0.5, 0.6],
                "landmark_descriptor": [0.6, 0.5, 0.4],
            },
        ]
    )

    assert compact == {
        "count": 2,
        "pnp_inlier_count": 1,
        "query_descriptor_dim": 3,
        "landmark_descriptor_dim": 3,
        "descriptors_omitted_from_results_json": True,
    }


def test_serialize_sparse_match_attributions_defaults_missing_descriptor_margin():
    attributions = serialize_sparse_match_attributions(
        {
            "matched_gaussian_ids": torch.tensor([10]),
            "matched_sampled_indices": torch.tensor([0]),
            "pnp_inlier_mask": torch.tensor([True]),
            "descriptor_score": torch.tensor([0.9]),
        },
        query_id="q1",
        image_id="im1",
    )

    assert attributions[0]["descriptor_margin"] == pytest.approx(0.0)
    assert attributions[0]["descriptor_margin_source"] == "missing_top2_default_zero"


def test_serialize_sparse_match_attributions_preserves_explicit_descriptor_margin_source():
    attributions = serialize_sparse_match_attributions(
        {
            "matched_gaussian_ids": torch.tensor([10]),
            "matched_sampled_indices": torch.tensor([0]),
            "pnp_inlier_mask": torch.tensor([True]),
            "descriptor_score": torch.tensor([0.9]),
            "descriptor_margin": torch.tensor([0.2]),
            "descriptor_margin_source": "descriptor_score_minus_best_nonselected",
        },
        query_id="q1",
        image_id="im1",
    )

    assert attributions[0]["descriptor_margin_source"] == "descriptor_score_minus_best_nonselected"


def test_serialize_sparse_match_attributions_treats_single_1d_vectors_as_one_match():
    attributions = serialize_sparse_match_attributions(
        {
            "matched_gaussian_ids": [10],
            "matched_sampled_indices": [0],
            "query_keypoint_indices": [3],
            "pnp_inlier_mask": [True],
            "descriptor_score": [0.9],
            "descriptor_margin": [0.2],
            "detector_score": [0.7],
            "reprojection_error_px": [1.0],
            "query_xy": [5.0, 6.0],
            "camera_xyz": [0.1, 0.2, 2.0],
            "depth_m": [2.0],
            "bearing": [0.0, 0.0, 1.0],
            "query_descriptor": [0.25, 0.75],
            "landmark_descriptor": [1.0, 0.0],
            "image_size": (100, 50),
        },
        scene="ShopFacade",
        split_name="selfmap_train",
        query_id="q1",
        image_id="im1",
    )

    assert len(attributions) == 1
    assert attributions[0]["query_descriptor"] == [pytest.approx(0.25), pytest.approx(0.75)]
    assert attributions[0]["landmark_descriptor"] == [pytest.approx(1.0), pytest.approx(0.0)]
    assert attributions[0]["camera_xyz"] == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(2.0)]
    assert attributions[0]["bearing"] == [pytest.approx(0.0), pytest.approx(0.0), pytest.approx(1.0)]


def test_serialize_sparse_match_attributions_can_omit_descriptors_for_full_scene_trace():
    attributions = serialize_sparse_match_attributions(
        {
            "matched_gaussian_ids": [10],
            "matched_sampled_indices": [0],
            "query_keypoint_indices": [3],
            "pnp_inlier_mask": [True],
            "descriptor_score": [0.9],
            "descriptor_margin": [0.2],
            "query_descriptor": [0.25, 0.75],
            "landmark_descriptor": [1.0, 0.0],
        },
        query_id="q1",
        image_id="im1",
        include_descriptors=False,
    )

    assert len(attributions) == 1
    assert attributions[0]["gaussian_id"] == 10
    assert "query_descriptor" not in attributions[0]
    assert "landmark_descriptor" not in attributions[0]


def test_serialize_sparse_match_attributions_caps_outliers_but_keeps_inliers():
    attributions = serialize_sparse_match_attributions(
        {
            "matched_gaussian_ids": [10, 11, 12, 13],
            "pnp_inlier_mask": [False, True, False, True],
            "descriptor_score": [0.7, 0.1, 0.9, 0.2],
        },
        query_id="q1",
        image_id="im1",
        include_descriptors=False,
        max_correspondences=3,
    )

    kept = {row["gaussian_id"] for row in attributions}
    assert kept == {11, 12, 13}
    assert sum(1 for row in attributions if row["pnp_inlier"]) == 2


def test_build_sparse_feedback_query_row_reports_pose_and_coverage_fields():
    row = build_sparse_feedback_query_row(
        scene="ShopFacade",
        split_name="selfmap_train",
        query_id="q1",
        image_id="im1",
        sparse_te_cm=3.0,
        sparse_re_deg=0.1,
        matches={
            "pnp_success": True,
            "sparse_inlier_count": 2,
            "total_match_count": 3,
            "pnp_inlier_mask": torch.tensor([True, True, False]),
            "query_xy": torch.tensor([[5.0, 6.0], [70.0, 40.0], [9.0, 9.0]]),
            "image_size": (100, 50),
            "depth_m": torch.tensor([2.0, 5.0, 8.0]),
            "bearing": torch.tensor([[0.0, 0.0, 1.0], [0.3, 0.0, 0.95], [0.0, 1.0, 0.0]]),
        },
    )

    assert row["scene"] == "ShopFacade"
    assert row["split_name"] == "selfmap_train"
    assert row["pose_success"] is True
    assert row["sparse_te_cm"] == pytest.approx(3.0)
    assert row["inlier_count"] == 2
    assert row["match_count"] == 3
    assert row["inlier_image_cell_count"] == 2
    assert row["inlier_depth_bin_count"] == 2
    assert row["depth_spread"] == pytest.approx(3.0)
    assert row["bearing_spread"] > 0.0


def test_build_sparse_feedback_trace_payload_uses_queries_and_correspondences_schema():
    payload = build_sparse_feedback_trace_payload(
        scene="ShopFacade",
        split_name="selfmap_train",
        query_rows=[{"query_id": "q1"}],
        correspondence_rows=[{"gaussian_id": 10}],
        query_features={"q1": torch.ones(2)},
        query_match_descriptors={"q1": torch.ones(1, 2)},
        split_audit={"test_split_used": False, "official_test_used": False},
    )

    assert payload["schema_version"] == "ulfloc_sparse_feedback_trace_v2"
    assert payload["queries"] == [{"query_id": "q1"}]
    assert payload["correspondences"] == [{"gaussian_id": 10}]
    assert "records" not in payload
    assert payload["query_match_descriptor_source"] == "matched_sparse_query_descriptors"
    assert payload["record_count"] == 1
    assert payload["query_count"] == 1


def test_build_sparse_only_split_audit_explicitly_marks_non_test_eval():
    audit = build_sparse_only_split_audit("train_dev_seed13_20p_sparse_validation")

    assert audit["official_test_eval"] is False
    assert audit["official_test_used"] is False
    assert audit["test_split_used"] is False
    assert audit["paper_safe_for_tuning"] is True


def test_build_sparse_only_split_audit_marks_unknown_split_not_paper_safe():
    audit = build_sparse_only_split_audit("unknown")

    assert audit["test_split_used"] is False
    assert audit["official_test_used"] is False
    assert audit["paper_safe_for_tuning"] is False


def test_sparse_trace_payload_for_json_strips_match_descriptor_tensors():
    payload = {
        "schema_version": "ulfloc_sparse_pnp_trace_payload_v1",
        "split_name": "train_selfmap",
        "records": [{"gaussian_id": 10}],
        "query_match_descriptors": {
            "q1": torch.ones(2, 3),
            "q2": torch.ones(1, 3),
        },
    }

    json_payload = sparse_trace_payload_for_json(payload)

    assert "query_match_descriptors" not in json_payload
    assert json_payload["query_match_descriptor_count"] == 3
    assert json_payload["query_match_descriptor_dim"] == 3
    assert json_payload["query_match_descriptor_query_count"] == 2


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
            "--scene_detector_fusion_rule",
            "residual_boost",
            "--scene_detector_candidate_top_k",
            "3072",
            "--scene_detector_nms_radius",
            "2",
            "--scene_detector_score_threshold",
            "-0.1",
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
            "--landmark_activation_checkpoint",
            "/activation.pth",
            "--landmark_activation_top_n",
            "4096",
            "--landmark_activation_disable_empty_fallback",
        ]
    )

    assert args.scene_detector_blend_alpha == pytest.approx(0.25)
    assert args.scene_detector_fusion_rule == "residual_boost"
    assert args.scene_detector_candidate_top_k == 3072
    assert args.scene_detector_nms_radius == 2
    assert args.scene_detector_score_threshold == pytest.approx(-0.1)
    assert args.scene_detector_native_keep_fraction == pytest.approx(0.5)
    assert str(args.scene_matcher_checkpoint) == "/matcher.pt"
    assert args.scene_matcher_topk == 4
    assert args.scene_matcher_score_weight == pytest.approx(0.2)
    assert args.scene_matcher_score_mode == "candidate"
    assert args.scene_matcher_max_descriptor_margin == pytest.approx(0.05)
    assert args.scene_matcher_drop_dustbin is True
    assert args.scene_matcher_min_keep == 256
    assert str(args.landmark_activation_checkpoint) == "/activation.pth"
    assert args.landmark_activation_top_n == 4096
    assert args.landmark_activation_disable_empty_fallback is True


def test_eval_ulfloc_sparse_only_parser_defaults_scene_detector_to_stdloc_fullres():
    args = build_argparser().parse_args(
        [
            "--source_path",
            "/data",
            "--model_path",
            "/model",
            "--input_log_dir",
            "/logs",
            "--cfg",
            "/cfg.yml",
            "--output_dir",
            "/out",
            "--scene_detector_checkpoint",
            "/detector.pth",
        ]
    )

    assert args.scene_detector_mode == "stdloc_fullres"


def test_scene_detector_eval_config_defaults_to_stdloc_fullres_and_marks_legacy_modes_diagnostic():
    direct = build_scene_detector_eval_config(
        checkpoint="/detector.pth",
        mode="grid",
        sparse_k=2048,
        candidate_top_k=0,
        blend_alpha=0.25,
        native_keep_fraction=0.5,
        nms_radius=4,
    )
    assert direct["enabled"] is True
    assert direct["mode"] == "grid"
    assert direct["scene_detector_mode"] == "direct_heatmap"
    assert direct["diagnostic_only"] is True
    assert direct["native_keep_fraction"] == pytest.approx(0.0)

    rerank = build_scene_detector_eval_config(
        checkpoint="/detector.pth",
        mode="rerank_superpoint",
        sparse_k=2048,
        candidate_top_k=0,
        blend_alpha=0.25,
        native_keep_fraction=0.5,
        nms_radius=4,
    )
    assert rerank["mode"] == "rerank_superpoint"
    assert rerank["scene_detector_mode"] == "rerank_superpoint"
    assert rerank["diagnostic_only"] is True
    assert rerank["native_keep_fraction"] == pytest.approx(0.5)

    score_fusion = build_scene_detector_eval_config(
        checkpoint="/detector.pth",
        mode="score_fusion",
        sparse_k=2048,
        candidate_top_k=0,
        blend_alpha=0.25,
        fusion_rule="residual_boost",
        native_keep_fraction=0.5,
        nms_radius=4,
    )
    assert score_fusion["mode"] == "score_fusion"
    assert score_fusion["scene_detector_mode"] == "score_fusion"
    assert score_fusion["diagnostic_only"] is True
    assert score_fusion["fusion_rule"] == "residual_boost"
    assert score_fusion["native_keep_fraction"] == pytest.approx(0.0)

    fullres = build_scene_detector_eval_config(
        checkpoint="/detector.pth",
        mode="stdloc_fullres",
        sparse_k=2048,
        candidate_top_k=0,
        blend_alpha=0.25,
        native_keep_fraction=0.5,
        nms_radius=4,
    )
    assert fullres["mode"] == "stdloc_fullres"
    assert fullres["scene_detector_mode"] == "stdloc_fullres"
    assert fullres["diagnostic_only"] is False
    assert fullres["native_keep_fraction"] == pytest.approx(0.0)

    implicit = build_scene_detector_eval_config(
        checkpoint="/detector.pth",
        sparse_k=2048,
    )
    assert implicit["mode"] == "stdloc_fullres"
    assert implicit["scene_detector_mode"] == "stdloc_fullres"
    assert implicit["diagnostic_only"] is False


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


def test_load_landmark_prior_weights_rejects_test_audited_artifact(tmp_path):
    path = tmp_path / "solver_feedback.pt"
    torch.save(
        {
            "split_name": "selfmap_train",
            "split_audit": {"test_split_used": True, "official_test_used": False},
            "landmark_weights": torch.tensor([1.0], dtype=torch.float32),
        },
        path,
    )

    with pytest.raises(ValueError, match="test split"):
        load_landmark_prior_weights(path)


def test_load_landmark_activation_audit_rejects_test_checkpoint(tmp_path):
    path = tmp_path / "activation.pth"
    torch.save(
        {
            "split_name": "test",
            "split_audit": {"test_split_used": True},
            "landmark_ids": torch.tensor([1, 2]),
        },
        path,
    )

    with pytest.raises(ValueError, match="test split"):
        load_landmark_activation_audit(path)


def test_load_landmark_activation_audit_reports_checkpoint_contract(tmp_path):
    path = tmp_path / "activation.pth"
    torch.save(
        {
            "split_name": "train_selfmap",
            "query_dim": 256,
            "landmark_dim": 256,
            "hidden_dim": 64,
            "top_n": 2048,
            "landmark_ids": torch.tensor([10, 20, 30]),
            "safe_core_indices": torch.tensor([0, 2]),
            "source_cache_path": "/cache.pt",
            "split_audit": {"test_split_used": False},
        },
        path,
    )

    audit = load_landmark_activation_audit(path)

    assert audit["enabled"] is True
    assert audit["checkpoint_split_name"] == "train_selfmap"
    assert audit["checkpoint_top_n"] == 2048
    assert audit["checkpoint_landmark_count"] == 3
    assert audit["safe_core_count"] == 2


def test_load_landmark_activation_audit_reports_v2_token_checkpoint_contract(tmp_path):
    path = tmp_path / "activation_v2.pth"
    torch.save(
        {
            "activation_model_type": "v2_tokens",
            "split_name": "train_selfmap",
            "query_token_dim": 258,
            "landmark_token_dim": 260,
            "hidden_dim": 64,
            "attention_top_k": 32,
            "top_n": 2048,
            "landmark_ids": torch.tensor([10, 20, 30]),
            "safe_core_indices": torch.tensor([0, 2]),
            "source_cache_path": "/cache.pt",
            "split_audit": {"test_split_used": False},
        },
        path,
    )

    audit = load_landmark_activation_audit(path)

    assert audit["activation_model_type"] == "v2_tokens"
    assert audit["query_feature_mode"] == "token_cross_attention"
    assert audit["diagnostic_only"] is False
    assert audit["query_token_dim"] == 258
    assert audit["landmark_token_dim"] == 260
    assert audit["attention_top_k"] == 32


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
