import json
import pickle
import sys
from pathlib import Path

import pytest

from loc_gs.scripts import locgsctl


def _run_cli(capsys, *args: str) -> dict:
    code = locgsctl.main(list(args))
    assert code == 0
    captured = capsys.readouterr()
    return json.loads(captured.out)


def test_status_reports_environment_and_key_paths(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    (repo / "loc_gs").mkdir(parents=True)
    (repo / "third_party" / "stdloc").mkdir(parents=True)
    (repo / "docs").mkdir()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    payload = _run_cli(capsys, "--repo-root", str(repo), "status")

    assert payload["python_executable"] == sys.executable
    assert payload["cuda_visible_devices"] == "2"
    assert payload["paths"]["repo_root"]["exists"] is True
    assert payload["paths"]["loc_gs"]["exists"] is True
    assert payload["paths"]["third_party_stdloc"]["exists"] is True
    assert payload["paths"]["docs"]["exists"] is True
    assert "git_commit" in payload


def test_summarize_compacts_summary_metrics(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "model_path": "output/stdloc/map_cambridge_spgs/ShopFacade",
                "dense": {
                    "median_te": 2.5,
                    "median_ae": 0.12,
                    "recall_10cm_5d": 0.91,
                    "recall_5cm_5d": 0.80,
                    "recall_2cm_2d": 0.30,
                },
                "sparse": {
                    "median_te_cm": 4.0,
                    "median_re_deg": 0.2,
                    "recall_5cm_5deg": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )

    payload = _run_cli(capsys, "summarize", str(run_dir))

    assert payload["source"] == str(run_dir / "summary.json")
    assert payload["model_path"] == "output/stdloc/map_cambridge_spgs/ShopFacade"
    assert payload["dense"] == {
        "median_te_cm": 2.5,
        "median_re_deg": 0.12,
        "recall_10cm_5deg": 0.91,
        "recall_5cm_5deg": 0.80,
        "recall_2cm_2deg": 0.30,
    }
    assert payload["sparse"]["median_te_cm"] == 4.0
    assert payload["sparse"]["recall_5cm_5deg"] == 0.5


def test_summarize_prefers_metrics_summary_for_audit_bundle_directories(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps({"scene": "ShopFacade", "dense": {"median_te": 99.0}}),
        encoding="utf-8",
    )
    (run_dir / "metrics_summary.json").write_text(
        json.dumps({"scene": "ShopFacade", "dense": {"median_te_cm": 2.25}}),
        encoding="utf-8",
    )

    payload = _run_cli(capsys, "summarize", str(run_dir))

    assert payload["source"] == str(run_dir / "metrics_summary.json")
    assert payload["dense"]["median_te_cm"] == 2.25


def test_summarize_compacts_candidate_mlp_cache_and_training_summaries(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "internal_candidate_mlp_feature_cache_summary_v1",
                "scene": "GreatCourt",
                "split_name": "train_dev",
                "sample_count": 40000,
                "label_count": 1073,
                "group_count": 5000,
                "native_top1_correct": 294,
                "dense_teacher_sample_count": 40000,
                "feature_materialization": "feature_cache",
                "feature_input_policy": "inference_safe",
                "paper_safe_sparse_inference": True,
            }
        ),
        encoding="utf-8",
    )

    cache_payload = _run_cli(capsys, "summarize", str(cache_dir))

    assert cache_payload["scene"] == "GreatCourt"
    assert cache_payload["split_name"] == "train_dev"
    assert cache_payload["candidate_mlp_feature_cache"] == {
        "dense_teacher_sample_count": 40000,
        "feature_input_policy": "inference_safe",
        "feature_materialization": "feature_cache",
        "group_count": 5000,
        "label_count": 1073,
        "native_top1_correct": 294,
        "paper_safe_sparse_inference": True,
        "sample_count": 40000,
    }

    scorer_dir = tmp_path / "scorer"
    scorer_dir.mkdir()
    (scorer_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "internal_candidate_mlp_scorer_training_summary_v1",
                "scene": "GreatCourt",
                "split_name": "train_dev",
                "sample_count": 40000,
                "label_count": 1073,
                "native_top1_correct": 294,
                "trained_top1_correct": 573,
                "dense_teacher_sample_count": 40000,
                "feature_materialization": "feature_cache",
                "feature_input_policy": "inference_safe",
                "paper_safe_sparse_inference": True,
            }
        ),
        encoding="utf-8",
    )

    scorer_payload = _run_cli(capsys, "summarize", str(scorer_dir))

    assert scorer_payload["candidate_mlp_scorer"]["trained_top1_correct"] == 573
    assert scorer_payload["candidate_mlp_scorer"]["feature_materialization"] == "feature_cache"


def test_summarize_compacts_landmark_selector_training_summary(tmp_path, capsys):
    selector_dir = tmp_path / "selector"
    selector_dir.mkdir()
    (selector_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "internal_landmark_selector_training_summary_v1",
                "scene": "GreatCourt",
                "split_name": "train_dev",
                "student_modules": ["landmark_selector", "conflict_graph"],
                "landmark_count": 105226,
                "observed_candidate_count": 1257472,
                "protected_support_count": 18830,
                "hard_negative_count": 1189004,
                "positive_inlier_count": 0,
                "conflict_edge_count": 63377,
                "hyperparameters": {
                    "conflict_penalty": 0.1,
                    "protected_support_gain": 2.0,
                },
            }
        ),
        encoding="utf-8",
    )

    payload = _run_cli(capsys, "summarize", str(selector_dir))

    assert payload["candidate_conflict_graph_training"] == {
        "conflict_edge_count": 63377,
        "hard_negative_count": 1189004,
        "hyperparameters": {
            "conflict_penalty": 0.1,
            "protected_support_gain": 2.0,
        },
        "landmark_count": 105226,
        "observed_candidate_count": 1257472,
        "positive_inlier_count": 0,
        "protected_support_count": 18830,
        "schema_version": "internal_landmark_selector_training_summary_v1",
        "student_modules": ["landmark_selector", "conflict_graph"],
    }


def test_summarize_compacts_online_sparse_student_training_summary(tmp_path, capsys):
    train_dir = tmp_path / "online_students"
    train_dir.mkdir()
    (train_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "internal_online_sparse_student_training_summary_v1",
                "scene": "GreatCourt",
                "split_name": "train_dev",
                "student_modules": [
                    "correspondence_scorer",
                    "candidate_mlp_scorer",
                    "landmark_selector",
                    "conflict_graph",
                    "descriptor_fusion",
                    "detector_student",
                ],
                "online_episode_count": 64,
                "missing_candidate_count": 0,
                "candidate_keypoint_count": 4096,
                "source_image_count": 53,
                "dense_helped_episode_count": 7,
                "feedback_matched_episode_count": 7,
                "render_ready_episode_count": 60,
                "missing_render_episode_count": 4,
                "top1_correct": 1066,
                "topk_available": 2088,
                "oracle_gap": 1022,
                "render_engine": "3dgs",
                "distillation": {
                    "schema_version": "internal_distillation_artifact_summary_v1",
                    "candidate_row_count": 65536,
                    "candidate_sample_count": 524288,
                    "feedback_query_count": 25,
                    "matched_feedback_row_count": 12800,
                    "dense_helped_query_count": 21,
                    "protected_support_count": 30632,
                    "hard_negative_count": 96589,
                },
                "candidate_scorer": {
                    "schema_version": "internal_sparse_candidate_scorer_training_summary_v1",
                    "sample_count": 13410,
                    "label_count": 2088,
                    "native_top1_correct": 1066,
                    "trained_top1_correct": 1181,
                    "dense_teacher_sample_count": 13410,
                },
                "candidate_mlp_scorer": {
                    "schema_version": "internal_candidate_mlp_scorer_training_summary_v1",
                    "sample_count": 13410,
                    "label_count": 2088,
                    "native_top1_correct": 1066,
                    "trained_top1_correct": 1198,
                    "feature_materialization": "feature_cache",
                    "feature_cache_enabled": True,
                },
                "landmark_selector": {
                    "schema_version": "internal_landmark_selector_training_summary_v1",
                    "student_modules": ["landmark_selector", "conflict_graph"],
                    "landmark_count": 105226,
                    "observed_candidate_count": 1257472,
                    "protected_support_count": 18830,
                    "hard_negative_count": 1189004,
                    "positive_inlier_count": 0,
                    "conflict_edge_count": 63377,
                },
                "descriptor_fusion": {
                    "schema_version": "internal_descriptor_fusion_training_summary_v1",
                    "student_modules": ["descriptor_fusion"],
                    "landmark_count": 2000,
                    "protected_support_count": 1200,
                    "positive_inlier_count": 300,
                    "hard_negative_count": 500,
                },
                "detector_student": {
                    "schema_version": "internal_detector_student_training_summary_v1",
                    "student_modules": ["detector_student"],
                    "grid_size": 8,
                    "cell_count": 17,
                    "positive_keypoint_count": 120,
                    "hard_negative_keypoint_count": 44,
                },
            }
        ),
        encoding="utf-8",
    )

    payload = _run_cli(capsys, "summarize", str(train_dir))

    evidence = payload["candidate_student_training"]
    assert evidence["online_episode_count"] == 64
    assert evidence["render_ready_episode_count"] == 60
    assert evidence["student_modules"] == [
        "correspondence_scorer",
        "candidate_mlp_scorer",
        "landmark_selector",
        "conflict_graph",
        "descriptor_fusion",
        "detector_student",
    ]
    assert evidence["distillation"]["hard_negative_count"] == 96589
    assert evidence["candidate_scorer"]["top1_gain"] == 115
    assert evidence["candidate_mlp_scorer"]["top1_gain"] == 132
    assert evidence["landmark_selector"]["conflict_edge_count"] == 63377
    assert evidence["descriptor_fusion"]["landmark_count"] == 2000
    assert evidence["detector_student"]["positive_keypoint_count"] == 120


def test_summarize_compacts_internal_sparse_failure_profile(tmp_path, capsys):
    profile_dir = tmp_path / "failure_profile"
    profile_dir.mkdir()
    (profile_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "internal_sparse_failure_profile_v1",
                "scene": "GreatCourt",
                "split_name": "train_dev",
                "query_count": 20,
                "success_count": 20,
                "median_te_cm": 55.62,
                "target_gap_cm": 45.62,
                "failure_mode_counts": {
                    "selected_set_low_precision": 20,
                    "inlier_set_wrong_dominant": 19,
                },
                "dominant_failure_modes": [
                    {"mode": "selected_set_low_precision", "count": 20},
                    {"mode": "inlier_set_wrong_dominant", "count": 19},
                ],
                "recommendation": "prioritize_set_level_selection_and_inlier_precision",
                "candidate_artifact": {
                    "top1_correct": 502,
                    "topk_available": 1453,
                    "oracle_gap": 951,
                },
                "rerank_diagnostic": {
                    "native_top1_correct": 502,
                    "reranked_top1_correct": 977,
                    "reranked_top1_gain": 475,
                    "reranked_topk_available": 1453,
                },
                "post_pnp_rescore": {
                    "enabled": False,
                    "correct_delta_sum": 0,
                    "worsened_sum": 0,
                },
                "per_query": [{"query_id": "q.png", "failure_modes": ["selected_set_low_precision"]}],
            }
        ),
        encoding="utf-8",
    )

    payload = _run_cli(capsys, "summarize", str(profile_dir))

    evidence = payload["sparse_failure_profile"]
    assert evidence["query_count"] == 20
    assert evidence["failure_mode_counts"]["selected_set_low_precision"] == 20
    assert evidence["dominant_failure_modes"][0] == {"mode": "selected_set_low_precision", "count": 20}
    assert evidence["recommendation"] == "prioritize_set_level_selection_and_inlier_precision"
    assert evidence["candidate_artifact"]["oracle_gap"] == 951
    assert evidence["rerank_diagnostic"]["reranked_top1_gain"] == 475
    assert "per_query" not in evidence


def test_summarize_compacts_internal_inlier_precision_feedback(tmp_path, capsys):
    feedback_dir = tmp_path / "inlier_feedback"
    feedback_dir.mkdir()
    (feedback_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "internal_inlier_precision_feedback_v1",
                "scene": "GreatCourt",
                "split_name": "train_dev",
                "query_count": 20,
                "hard_query_count": 15,
                "set_selection_hard_count": 11,
                "inlier_precision_hard_count": 15,
                "post_pnp_rescore_harm_count": 0,
                "mean_scorer_distill_weight": 1.42,
                "max_scorer_distill_weight": 3.1,
                "recommendation": "boost_hard_negative_and_set_level_student_weights",
                "student_consumers": ["candidate_mlp_scorer", "landmark_selector"],
                "per_query": [{"query_id": "q.png", "scorer_distill_weight": 2.0}],
            }
        ),
        encoding="utf-8",
    )

    payload = _run_cli(capsys, "summarize", str(feedback_dir))

    evidence = payload["inlier_precision_feedback"]
    assert evidence["query_count"] == 20
    assert evidence["hard_query_count"] == 15
    assert evidence["inlier_precision_hard_count"] == 15
    assert evidence["recommendation"] == "boost_hard_negative_and_set_level_student_weights"
    assert "per_query" not in evidence


def test_summarize_compacts_sparse_gate_with_scorer_training_evidence(tmp_path, capsys):
    run_dir = tmp_path / "gate"
    run_dir.mkdir()
    (run_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "internal_sparse_train_dev_gate_v1",
                "scene": "GreatCourt",
                "split_name": "train_dev",
                "sparse_gate_status": "regression",
                "baseline_median_te_cm": 72.29,
                "candidate_median_te_cm": 73.97,
                "delta_median_te_cm": 1.68,
                "target_gap_cm": 63.97,
                "candidate_scorer_training": {
                    "schema_version": "internal_candidate_mlp_scorer_training_summary_v1",
                    "feature_materialization": "feature_cache",
                    "feature_input_policy": "inference_safe",
                    "paper_safe_sparse_inference": True,
                    "native_top1_correct": 294,
                    "trained_top1_correct": 573,
                    "top1_gain": 279,
                    "relative_top1_gain": 0.9489795918367347,
                },
                "candidate_conflict_graph_training": {
                    "schema_version": "internal_landmark_selector_training_summary_v1",
                    "student_modules": ["landmark_selector", "conflict_graph"],
                    "landmark_count": 105226,
                    "observed_candidate_count": 1257472,
                    "protected_support_count": 18830,
                    "hard_negative_count": 1189004,
                    "positive_inlier_count": 0,
                    "conflict_edge_count": 63377,
                    "hyperparameters": {
                        "conflict_penalty": 0.1,
                        "protected_support_gain": 2.0,
                    },
                },
                "candidate_student_training": {
                    "schema_version": "internal_online_sparse_student_training_summary_v1",
                    "student_modules": ["correspondence_scorer", "candidate_mlp_scorer"],
                    "online_episode_count": 64,
                    "missing_candidate_count": 0,
                    "candidate_keypoint_count": 4096,
                    "source_image_count": 53,
                    "dense_helped_episode_count": 7,
                    "feedback_matched_episode_count": 7,
                    "render_ready_episode_count": 60,
                    "missing_render_episode_count": 4,
                    "top1_correct": 1066,
                    "topk_available": 2088,
                    "oracle_gap": 1022,
                    "render_engine": "3dgs",
                    "distillation": {
                        "schema_version": "internal_distillation_artifact_summary_v1",
                        "candidate_row_count": 65536,
                        "candidate_sample_count": 524288,
                        "feedback_query_count": 25,
                        "matched_feedback_row_count": 12800,
                        "dense_helped_query_count": 21,
                        "protected_support_count": 30632,
                        "hard_negative_count": 96589,
                    },
                    "candidate_mlp_scorer": {
                        "schema_version": "internal_candidate_mlp_scorer_training_summary_v1",
                        "native_top1_correct": 1066,
                        "trained_top1_correct": 1198,
                        "feature_materialization": "feature_cache",
                    },
                    "inlier_precision_feedback": {
                        "schema_version": "internal_inlier_precision_feedback_merge_summary_v1",
                        "input_solver_feedback_count": 25,
                        "input_inlier_precision_feedback_count": 20,
                        "matched_solver_feedback_count": 15,
                        "added_solver_feedback_count": 0,
                        "boosted_solver_feedback_count": 15,
                        "weight_scale": 1.0,
                        "max_distill_weight": 4.0,
                    },
                },
                "candidate_failure_profile": {
                    "schema_version": "internal_sparse_failure_profile_v1",
                    "query_count": 20,
                    "success_count": 20,
                    "median_te_cm": 55.62,
                    "target_gap_cm": 45.62,
                    "failure_mode_counts": {
                        "selected_set_low_precision": 20,
                        "inlier_set_wrong_dominant": 19,
                    },
                    "dominant_failure_modes": [
                        {"mode": "selected_set_low_precision", "count": 20},
                        {"mode": "inlier_set_wrong_dominant", "count": 19},
                    ],
                    "recommendation": "prioritize_set_level_selection_and_inlier_precision",
                    "candidate_artifact": {
                        "top1_correct": 502,
                        "topk_available": 1453,
                        "oracle_gap": 951,
                    },
                    "rerank_diagnostic": {
                        "native_top1_correct": 502,
                        "reranked_top1_correct": 977,
                        "reranked_top1_gain": 475,
                        "reranked_topk_available": 1453,
                    },
                },
                "candidate_inlier_precision_feedback": {
                    "schema_version": "internal_inlier_precision_feedback_v1",
                    "query_count": 20,
                    "hard_query_count": 15,
                    "set_selection_hard_count": 11,
                    "inlier_precision_hard_count": 15,
                    "post_pnp_rescore_harm_count": 0,
                    "mean_scorer_distill_weight": 1.42,
                    "max_scorer_distill_weight": 3.1,
                    "recommendation": "boost_hard_negative_and_set_level_student_weights",
                    "student_consumers": ["candidate_mlp_scorer", "landmark_selector"],
                    "per_query": [{"query_id": "q.png", "scorer_distill_weight": 2.0}],
                },
                "candidate_rerank_diagnostic": {
                    "rerank_diagnostic_enabled": True,
                    "rerank_diagnostic_query_count": 1536,
                    "native_top1_correct": 91,
                    "reranked_top1_correct": 225,
                    "reranked_top1_gain": 134,
                    "reranked_top1_changed_count": 1065,
                    "reranked_topk_available": 259,
                },
                "candidate_selected_set_diagnostic": {
                    "selected_geometric_correct_count_median": 32.0,
                    "selected_geometric_correct_ratio_median": 0.0625,
                    "selected_keypoint_bbox_area_fraction_median": 0.9497,
                    "selected_depth_range_m_median": 46.5,
                },
                "candidate_inlier_set_diagnostic": {
                    "inlier_geometric_correct_count_median": 21.0,
                    "inlier_geometric_correct_ratio_median": 0.18,
                    "inlier_keypoint_bbox_area_fraction_median": 0.75,
                    "inlier_depth_range_m_median": 35.0,
                },
                "candidate_set_conflict_diagnostic": {
                    "conflict_graph_enabled": True,
                    "set_conflict_penalty_enabled": True,
                    "set_conflict_penalty": 1.0,
                    "set_conflict_edge_count": 37,
                    "set_conflict_rerank_changed_count_median": 5.0,
                },
                "candidate_post_pnp_rescore_diagnostic": {
                    "post_pnp_candidate_rescore_enabled": True,
                    "post_pnp_rescore_changed_count_median": 44.0,
                    "post_pnp_rescore_corrected_count_median": 12.0,
                    "post_pnp_rescore_worsened_count_median": 3.0,
                    "post_pnp_rescore_correct_delta_median": 9.0,
                    "post_pnp_rescore_max_score_drop": 0.2,
                },
            }
        ),
        encoding="utf-8",
    )

    payload = _run_cli(capsys, "summarize", str(run_dir))

    assert payload["sparse_gate"] == {
        "baseline_median_te_cm": 72.29,
        "candidate_median_te_cm": 73.97,
        "delta_median_te_cm": 1.68,
        "sparse_gate_status": "regression",
        "target_gap_cm": 63.97,
    }
    assert payload["candidate_scorer_training"]["top1_gain"] == 279
    assert payload["candidate_scorer_training"]["feature_materialization"] == "feature_cache"
    assert payload["candidate_conflict_graph_training"]["conflict_edge_count"] == 63377
    assert payload["candidate_conflict_graph_training"]["protected_support_count"] == 18830
    assert payload["candidate_conflict_graph_training"]["student_modules"] == ["landmark_selector", "conflict_graph"]
    assert payload["candidate_student_training"]["online_episode_count"] == 64
    assert payload["candidate_student_training"]["distillation"]["protected_support_count"] == 30632
    assert payload["candidate_student_training"]["candidate_mlp_scorer"]["top1_gain"] == 132
    assert payload["candidate_student_training"]["inlier_precision_feedback"]["boosted_solver_feedback_count"] == 15
    assert payload["candidate_failure_profile"]["failure_mode_counts"]["selected_set_low_precision"] == 20
    assert payload["candidate_failure_profile"]["recommendation"] == "prioritize_set_level_selection_and_inlier_precision"
    assert payload["candidate_inlier_precision_feedback"]["hard_query_count"] == 15
    assert payload["candidate_inlier_precision_feedback"]["recommendation"] == "boost_hard_negative_and_set_level_student_weights"
    assert payload["candidate_rerank_diagnostic"]["reranked_top1_gain"] == 134
    assert payload["candidate_rerank_diagnostic"]["reranked_top1_correct"] == 225
    assert payload["candidate_selected_set_diagnostic"] == {
        "selected_depth_range_m_median": 46.5,
        "selected_geometric_correct_count_median": 32.0,
        "selected_geometric_correct_ratio_median": 0.0625,
        "selected_keypoint_bbox_area_fraction_median": 0.9497,
    }
    assert payload["candidate_inlier_set_diagnostic"] == {
        "inlier_depth_range_m_median": 35.0,
        "inlier_geometric_correct_count_median": 21.0,
        "inlier_geometric_correct_ratio_median": 0.18,
        "inlier_keypoint_bbox_area_fraction_median": 0.75,
    }
    assert payload["candidate_set_conflict_diagnostic"] == {
        "conflict_graph_enabled": True,
        "set_conflict_edge_count": 37,
        "set_conflict_penalty": 1.0,
        "set_conflict_penalty_enabled": True,
        "set_conflict_rerank_changed_count_median": 5.0,
    }
    assert payload["candidate_post_pnp_rescore_diagnostic"] == {
        "post_pnp_candidate_rescore_enabled": True,
        "post_pnp_rescore_changed_count_median": 44.0,
        "post_pnp_rescore_correct_delta_median": 9.0,
        "post_pnp_rescore_corrected_count_median": 12.0,
        "post_pnp_rescore_max_score_drop": 0.2,
        "post_pnp_rescore_worsened_count_median": 3.0,
    }


def test_summarize_compacts_sparse_cached_eval_rerank_diagnostics(tmp_path, capsys):
    run_dir = tmp_path / "eval"
    run_dir.mkdir()
    (run_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "internal_sparse_cached_eval_metrics_v1",
                "scene": "GreatCourt",
                "split_name": "train_dev",
                "median_te_cm": 76.85,
                "median_re_deg": 0.37,
                "recall_10cm_5d": 0.0,
                "candidate_scorer_enabled": True,
                "rerank_diagnostic_enabled": True,
                "rerank_diagnostic_query_count": 1536,
                "native_top1_correct": 91,
                "reranked_top1_correct": 225,
                "reranked_top1_gain": 134,
                "reranked_top1_changed_count": 1065,
                "reranked_topk_available": 259,
                "selected_geometric_correct_count_median": 32.0,
                "selected_geometric_correct_ratio_median": 0.0625,
                "selected_keypoint_bbox_area_fraction_median": 0.9497,
                "selected_depth_range_m_median": 46.5,
                "inlier_geometric_correct_count_median": 21.0,
                "inlier_geometric_correct_ratio_median": 0.18,
                "inlier_keypoint_bbox_area_fraction_median": 0.75,
                "inlier_depth_range_m_median": 35.0,
                "set_conflict_penalty_enabled": True,
                "conflict_graph_enabled": True,
                "set_conflict_penalty": 1.0,
                "set_conflict_edge_count": 37,
                "set_conflict_rerank_changed_count_median": 5.0,
                "post_pnp_candidate_rescore_enabled": True,
                "post_pnp_rescore_changed_count_median": 44.0,
                "post_pnp_rescore_corrected_count_median": 12.0,
                "post_pnp_rescore_worsened_count_median": 3.0,
                "post_pnp_rescore_correct_delta_median": 9.0,
                "post_pnp_rescore_max_score_drop": 0.2,
            }
        ),
        encoding="utf-8",
    )

    payload = _run_cli(capsys, "summarize", str(run_dir))

    assert payload["sparse"]["median_te_cm"] == 76.85
    assert payload["sparse"]["median_re_deg"] == 0.37
    assert payload["rerank_diagnostic"] == {
        "native_top1_correct": 91,
        "rerank_diagnostic_enabled": True,
        "rerank_diagnostic_query_count": 1536,
        "reranked_top1_changed_count": 1065,
        "reranked_top1_correct": 225,
        "reranked_top1_gain": 134,
        "reranked_topk_available": 259,
    }
    assert payload["selected_set_diagnostic"] == {
        "selected_depth_range_m_median": 46.5,
        "selected_geometric_correct_count_median": 32.0,
        "selected_geometric_correct_ratio_median": 0.0625,
        "selected_keypoint_bbox_area_fraction_median": 0.9497,
    }
    assert payload["inlier_set_diagnostic"] == {
        "inlier_depth_range_m_median": 35.0,
        "inlier_geometric_correct_count_median": 21.0,
        "inlier_geometric_correct_ratio_median": 0.18,
        "inlier_keypoint_bbox_area_fraction_median": 0.75,
    }
    assert payload["set_conflict_diagnostic"] == {
        "conflict_graph_enabled": True,
        "set_conflict_edge_count": 37,
        "set_conflict_penalty": 1.0,
        "set_conflict_penalty_enabled": True,
        "set_conflict_rerank_changed_count_median": 5.0,
    }
    assert payload["post_pnp_rescore_diagnostic"] == {
        "post_pnp_candidate_rescore_enabled": True,
        "post_pnp_rescore_changed_count_median": 44.0,
        "post_pnp_rescore_correct_delta_median": 9.0,
        "post_pnp_rescore_corrected_count_median": 12.0,
        "post_pnp_rescore_max_score_drop": 0.2,
        "post_pnp_rescore_worsened_count_median": 3.0,
    }


def test_compare_reports_candidate_minus_baseline_deltas(tmp_path, capsys):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "summary.json").write_text(
        json.dumps(
            {
                "dense": {
                    "median_te": 9.0,
                    "median_ae": 0.15,
                    "recall_10cm_5d": 0.50,
                    "recall_5cm_5d": 0.30,
                    "recall_2cm_2d": 0.10,
                }
            }
        ),
        encoding="utf-8",
    )
    (candidate / "summary.json").write_text(
        json.dumps(
            {
                "dense": {
                    "median_te_cm": 8.5,
                    "median_re_deg": 0.14,
                    "recall_10cm_5deg": 0.55,
                    "recall_5cm_5deg": 0.35,
                    "recall_2cm_2deg": 0.12,
                }
            }
        ),
        encoding="utf-8",
    )

    payload = _run_cli(capsys, "compare", str(baseline), str(candidate))

    assert payload["stage"] == "dense"
    assert payload["baseline"]["median_te_cm"] == 9.0
    assert payload["candidate"]["median_te_cm"] == 8.5
    assert payload["delta"]["median_te_cm"] == pytest.approx(-0.5)
    assert payload["delta"]["median_re_deg"] == pytest.approx(-0.01)
    assert payload["delta"]["recall_10cm_5deg"] == pytest.approx(0.05)
    assert payload["delta"]["recall_5cm_5deg"] == pytest.approx(0.05)
    assert payload["delta"]["recall_2cm_2deg"] == pytest.approx(0.02)


def test_list_scenes_uses_stdloc_cambridge_default_root(capsys):
    payload = _run_cli(capsys, "list-scenes")

    assert payload["scenes"][0]["data_root"].startswith("/mnt/pool/sqy/Cambridge_stdloc/")
    assert "map_cambridge_spgs_lsf_v6_guarded512_20260525" in payload["scenes"][0]["map_path"]


def test_list_scenes_reports_native_sampled_count_status(tmp_path, capsys):
    repo = tmp_path / "repo"
    detector = repo / "maps" / "ShopFacade" / "detector"
    detector.mkdir(parents=True)
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump([1, 2], handle)

    payload = _run_cli(capsys, "--repo-root", str(repo), "list-scenes", "--map-root", "maps")

    shop = next(scene for scene in payload["scenes"] if scene["scene"] == "ShopFacade")
    assert shop["sampled_count"] == 2
    assert shop["native_sampled_count_expected"] == 16384
    assert shop["native_sampled_count_status"] == "mismatch"


def test_smoke_flags_native_sampled_count_mismatch(tmp_path, capsys):
    repo = tmp_path / "repo"
    data_root = tmp_path / "Cambridge"
    checkpoint_root = tmp_path / "checkpoints"
    detector = repo / "maps" / "ShopFacade" / "detector"
    detector.mkdir(parents=True)
    (data_root / "ShopFacade").mkdir(parents=True)
    (checkpoint_root / "ShopFacade").mkdir(parents=True)
    (checkpoint_root / "ShopFacade" / "latest.pth").write_bytes(b"ckpt")
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump([1, 2], handle)

    payload = _run_cli(
        capsys,
        "--repo-root",
        str(repo),
        "smoke",
        "--scene",
        "ShopFacade",
        "--map-root",
        "maps",
        "--checkpoint-root",
        str(checkpoint_root),
        "--data-root",
        str(data_root),
        "--dry-run",
    )

    assert payload["ok"] is False
    assert payload["checks"]["native_sampled_count"]["status"] == "mismatch"
    assert payload["failed"] == ["native_sampled_count"]


def test_manifest_includes_experiment_audit_fields(tmp_path, capsys):
    output = tmp_path / "manifest.json"

    payload = _run_cli(
        capsys,
        "manifest",
        "--scene",
        "ShopFacade",
        "--split",
        "selfmap_train",
        "--checkpoint",
        "output/stdloc_hybrid/ShopFacade/latest.pth",
        "--map",
        "output/stdloc/map_cambridge_spgs/ShopFacade",
        "--data-root",
        "/mnt/pool/sqy/dataset/Cambridge/ShopFacade",
        "--hyperparameters",
        '{"rho": 0.25, "alpha": 0.0}',
        "--feedback-enabled",
        "--rho",
        "0.25",
        "--output",
        str(output),
        "--command",
        "--",
        "python",
        "-m",
        "loc_gs.scripts.eval_stdloc_native",
    )

    assert payload["scene"] == "ShopFacade"
    assert payload["checkpoint_path"] == "output/stdloc_hybrid/ShopFacade/latest.pth"
    assert payload["map_path"] == "output/stdloc/map_cambridge_spgs/ShopFacade"
    assert payload["data_roots"] == ["/mnt/pool/sqy/dataset/Cambridge/ShopFacade"]
    assert payload["hyperparameters"] == {"rho": 0.25, "alpha": 0.0}
    assert payload["feedback_enabled"] is True
    assert payload["command"] == ["python", "-m", "loc_gs.scripts.eval_stdloc_native"]
    assert json.loads(output.read_text(encoding="utf-8"))["hyperparameters"]["rho"] == 0.25
