import json

import pytest

from loc_gs.feedback.io import load_feedback_bank, save_feedback_bank, summarize_feedback_bank
from loc_gs.feedback.labels import (
    derive_hard_negative_labels,
    derive_inlier_labels,
    derive_landmark_reliability,
    derive_scene_reliability_baseline_relative,
)
from loc_gs.feedback.schema import FeedbackMatchRecord


def _record(**overrides):
    base = {
        "scene": "ShopFacade",
        "query_id": "render_0001",
        "source_view_id": "train_0001",
        "pose_source": "selfmap",
        "keypoint_xy": [10.0, 20.0],
        "matched_landmark_id": "lm_1",
        "matched_gaussian_id": "g_1",
        "descriptor_score": 0.9,
        "detector_score": 0.7,
        "match_rank": 1,
        "pnp_inlier": True,
        "reprojection_error_px": 1.5,
        "depth_consistency": 0.9,
        "visibility_score": 0.8,
        "pose_error_t_cm": 2.0,
        "pose_error_r_deg": 0.1,
        "pnp_success": True,
        "dense_refine_success": True,
        "jacobian_info_trace": 12.0,
        "jacobian_info_logdet_proxy": 2.0,
    }
    base.update(overrides)
    return base


def test_feedback_bank_saves_loads_and_summarizes_jsonl(tmp_path):
    path = tmp_path / "feedback_bank.jsonl"
    records = [
        FeedbackMatchRecord.from_mapping(_record()),
        _record(matched_landmark_id="lm_1", pnp_inlier=False, reprojection_error_px=18.0),
        _record(scene="OldHospital", matched_landmark_id="lm_2", descriptor_score=0.4),
    ]
    metadata = {
        "scene": "ShopFacade",
        "split_name": "selfmap_train",
        "command": "synthetic",
    }

    save_feedback_bank(path, records, metadata)
    loaded = load_feedback_bank(path)
    summary = summarize_feedback_bank(path)

    assert loaded["manifest"]["scene"] == "ShopFacade"
    assert loaded["manifest"]["split_name"] == "selfmap_train"
    assert loaded["manifest"]["git_commit"]
    assert loaded["manifest"]["timestamp_utc"]
    assert len(loaded["records"]) == 3
    assert loaded["records"][0]["keypoint_xy"] == [10.0, 20.0]
    assert summary["record_count"] == 3
    assert summary["scene_count"] == 2
    assert summary["scenes"] == ["OldHospital", "ShopFacade"]
    assert summary["pnp_inlier_rate"] == pytest.approx(2 / 3)


def test_hard_negative_and_inlier_labels_use_match_quality():
    records = [
        _record(matched_landmark_id="lm_good", pnp_inlier=True, reprojection_error_px=1.0),
        _record(matched_landmark_id="lm_bad", pnp_inlier=False, reprojection_error_px=12.0),
        _record(matched_landmark_id="lm_low_score", descriptor_score=0.2, pnp_inlier=False, reprojection_error_px=20.0),
    ]

    assert derive_inlier_labels(records) == [1, 0, 0]
    assert derive_hard_negative_labels(records, descriptor_score_min=0.5, reprojection_error_px_min=8.0) == [0, 1, 0]


def test_landmark_reliability_groups_records_and_counts_hard_negatives():
    records = [
        _record(matched_landmark_id="lm_1", pnp_inlier=True, reprojection_error_px=1.0),
        _record(matched_landmark_id="lm_1", pnp_inlier=False, reprojection_error_px=15.0),
        _record(matched_landmark_id="lm_2", pnp_inlier=True, reprojection_error_px=2.0),
    ]

    reliability = derive_landmark_reliability(records)

    assert reliability["lm_1"]["count"] == 2
    assert reliability["lm_1"]["inlier_rate"] == pytest.approx(0.5)
    assert reliability["lm_1"]["hard_negative_count"] == 1
    assert reliability["lm_2"]["reliability_score"] > reliability["lm_1"]["reliability_score"]


def test_empty_and_missing_fields_are_handled(tmp_path):
    path = tmp_path / "empty.json"
    save_feedback_bank(path, [], {"scene": "ShopFacade", "split_name": "selfmap_train"})
    summary = summarize_feedback_bank(path)

    assert summary["record_count"] == 0
    assert summary["pnp_inlier_rate"] == 0.0
    assert derive_inlier_labels([{"scene": "ShopFacade"}]) == [0]
    assert derive_hard_negative_labels([{"descriptor_score": 0.9}]) == [0]
    missing = FeedbackMatchRecord.from_mapping({"scene": "ShopFacade"})
    assert missing.scene == "ShopFacade"
    assert missing.query_id == ""
    assert json.loads(path.read_text(encoding="utf-8"))["manifest"]["split_name"] == "selfmap_train"


def test_scene_reliability_baseline_relative_reports_metric_deltas():
    result = derive_scene_reliability_baseline_relative(
        {
            "scene": "ShopFacade",
            "dense": {
                "median_te_cm": 8.0,
                "median_re_deg": 0.12,
                "recall_5cm_5deg": 0.42,
                "recall_2cm_2deg": 0.14,
            },
        },
        {
            "dense": {
                "median_te_cm": 9.0,
                "median_re_deg": 0.15,
                "recall_5cm_5deg": 0.37,
                "recall_2cm_2deg": 0.13,
            }
        },
    )

    assert result["scene"] == "ShopFacade"
    assert result["median_te_delta_cm"] == pytest.approx(-1.0)
    assert result["recall_5cm_5deg_delta"] == pytest.approx(0.05)
    assert result["paper_safe_improvement"] is True
