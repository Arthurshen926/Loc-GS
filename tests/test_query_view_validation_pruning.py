import json

import pytest

from loc_gs.feedback.query_view_validation_pruning import (
    attach_trace_attributions_to_eval_rows,
    load_eval_rows,
    prune_active_fusion_plan_from_sparse_validation,
)


def test_pruning_removes_regression_outlier_view_landmark_pairs():
    active_plan = {
        "split_name": "train_dev_seed13_20p_sparse_validation",
        "landmark_fusion_plan": {
            "10": {"selected_view_ids": ["view_good", "view_bad"]},
            "11": {"selected_view_ids": ["view_good"]},
            "12": {"selected_view_ids": ["view_help"]},
        },
    }
    baseline_rows = [
        {"image_name": "q_regress", "sparse_te_cm": 3.0},
        {"image_name": "q_improve", "sparse_te_cm": 35.0},
    ]
    candidate_rows = [
        {
            "image_name": "q_regress",
            "sparse_te_cm": 30.0,
            "sparse_match_attributions": [
                {
                    "matched_gaussian_id": 10,
                    "pnp_inlier": False,
                    "descriptor_score": 0.95,
                    "reprojection_error_px": 18.0,
                },
                {
                    "matched_gaussian_id": 11,
                    "pnp_inlier": True,
                    "descriptor_score": 0.9,
                    "reprojection_error_px": 1.0,
                },
            ],
        },
        {
            "image_name": "q_improve",
            "sparse_te_cm": 6.0,
            "sparse_match_attributions": [
                {
                    "matched_gaussian_id": 12,
                    "pnp_inlier": True,
                    "descriptor_score": 0.9,
                    "reprojection_error_px": 0.5,
                }
            ],
        },
    ]

    result = prune_active_fusion_plan_from_sparse_validation(
        active_plan=active_plan,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        protected_te_cm=15.0,
        regression_margin_cm=20.0,
        improvement_margin_cm=20.0,
        min_pair_score=0.0,
        min_views_per_landmark=0,
    )

    pruned = result["active_fusion_plan"]["landmark_fusion_plan"]
    assert "10" not in pruned
    assert pruned["11"]["selected_view_ids"] == ["view_good"]
    assert pruned["12"]["selected_view_ids"] == ["view_help"]
    assert result["metrics"]["removed_pair_count"] == 2
    assert result["metrics"]["kept_pair_count"] == 2
    assert result["metrics"]["regression_query_count"] == 1
    assert result["metrics"]["improvement_query_count"] == 1
    assert result["metrics"]["attribution_status"] == "attributed"


def test_pruning_keeps_minimum_views_per_landmark():
    active_plan = {
        "split_name": "train_dev",
        "landmark_fusion_plan": {"10": {"selected_view_ids": ["view_a"]}},
    }
    baseline_rows = [{"image_name": "q0", "sparse_te_cm": 2.0}]
    candidate_rows = [
        {
            "image_name": "q0",
            "sparse_te_cm": 40.0,
            "sparse_match_attributions": [
                {"matched_gaussian_id": 10, "pnp_inlier": False, "descriptor_score": 1.0}
            ],
        }
    ]

    result = prune_active_fusion_plan_from_sparse_validation(
        active_plan=active_plan,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        min_views_per_landmark=1,
    )

    assert result["active_fusion_plan"]["landmark_fusion_plan"]["10"]["selected_view_ids"] == ["view_a"]
    assert result["metrics"]["protected_pair_count"] == 1


def test_pruning_reports_metric_only_when_match_attributions_are_missing():
    active_plan = {
        "split_name": "train_dev",
        "landmark_fusion_plan": {"10": {"selected_view_ids": ["view_a"]}},
    }
    baseline_rows = [{"image_name": "q0", "sparse_te_cm": 2.0}]
    candidate_rows = [{"image_name": "q0", "sparse_te_cm": 40.0}]

    result = prune_active_fusion_plan_from_sparse_validation(
        active_plan=active_plan,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
    )

    assert result["active_fusion_plan"]["landmark_fusion_plan"]["10"]["selected_view_ids"] == ["view_a"]
    assert result["metrics"]["attribution_status"] == "metric_only"
    assert result["metrics"]["removed_pair_count"] == 0


def test_pruning_rejects_test_split_active_plan():
    with pytest.raises(ValueError, match="test split"):
        prune_active_fusion_plan_from_sparse_validation(
            active_plan={"split_name": "test", "landmark_fusion_plan": {}},
            baseline_rows=[],
            candidate_rows=[],
        )


def test_load_eval_rows_accepts_rows_schema(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.json").write_text(
        json.dumps({"rows": [{"image_name": "q0", "sparse_te_cm": 1.0}]}),
        encoding="utf-8",
    )

    rows = load_eval_rows(run_dir)

    assert rows == [{"image_name": "q0", "sparse_te_cm": 1.0}]


def test_attach_trace_attributions_to_eval_rows_joins_payload_by_query():
    rows = [
        {"image_name": "q0", "sparse_te_cm": 2.0},
        {"image_name": "q1", "sparse_te_cm": 3.0},
    ]
    payload = {
        "split_name": "train_selfmap",
        "correspondences": [
            {"query_id": "q0", "matched_gaussian_id": 10, "pnp_inlier": False},
            {"image_id": "q0", "matched_gaussian_id": 11, "pnp_inlier": True},
            {"query_id": "q2", "matched_gaussian_id": 12, "pnp_inlier": True},
        ],
    }

    joined = attach_trace_attributions_to_eval_rows(rows, payload)

    assert len(joined[0]["sparse_match_attributions"]) == 2
    assert "sparse_match_attributions" not in joined[1]


def test_attach_trace_attributions_rejects_test_payload():
    with pytest.raises(ValueError, match="test split"):
        attach_trace_attributions_to_eval_rows(
            [{"image_name": "q0"}],
            {"split_name": "test", "correspondences": []},
        )
