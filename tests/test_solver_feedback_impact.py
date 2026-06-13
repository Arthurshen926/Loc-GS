from __future__ import annotations

import pytest

from loc_gs.feedback.solver_feedback_impact import build_solver_feedback_impact


def test_rejects_test_split() -> None:
    payload = {"split_name": "test", "records": []}

    with pytest.raises(ValueError, match="test split"):
        build_solver_feedback_impact(payload)


def test_rejects_audited_test_split_even_when_top_level_split_is_train() -> None:
    payload = {
        "split_name": "train_selfmap",
        "split_audit": {"test_split_used": True, "official_test_used": False},
        "records": [],
    }

    with pytest.raises(ValueError, match="test split"):
        build_solver_feedback_impact(payload)


def test_rejects_mismatched_audit_split() -> None:
    payload = {
        "split_name": "train_selfmap",
        "split_audit": {"split_name": "other_train", "test_split_used": False, "official_test_used": False},
        "records": [],
    }

    with pytest.raises(ValueError, match="split_name mismatch"):
        build_solver_feedback_impact(payload)


def test_positive_and_negative_landmark_impact_are_separated() -> None:
    payload = {
        "split_name": "train_selfmap",
        "records": [
            {
                "query_id": "q1",
                "image_id": "im1",
                "gaussian_id": 10,
                "landmark_id": 0,
                "label": 1,
                "pnp_inlier": True,
                "pose_success": True,
                "reprojection_error_px": 1.0,
                "descriptor_margin": 0.4,
                "keypoint_xy": [20.0, 30.0],
                "query_xy_norm": [0.1, 0.2],
                "source_role": "baseline_trace",
            },
            {
                "query_id": "q1",
                "image_id": "im1",
                "gaussian_id": 11,
                "landmark_id": 1,
                "label": 0,
                "pnp_inlier": False,
                "pose_success": False,
                "reprojection_error_px": 24.0,
                "descriptor_score": 0.92,
                "descriptor_margin": 0.02,
                "query_regression_delta_cm": 35.0,
                "keypoint_xy": [80.0, 90.0],
                "query_xy_norm": [0.4, 0.5],
                "source_role": "candidate_trace",
            },
        ],
    }

    impact = build_solver_feedback_impact(payload)

    assert impact["split_name"] == "train_selfmap"
    assert impact["landmark_positive"][10]["support"] > 0.0
    assert impact["landmark_negative"][11]["risk"] > 0.0
    assert impact["view_positive"][("10", "im1")]["weight"] > 0.0
    assert impact["view_negative"][("11", "im1")]["weight"] > 0.0
    assert impact["detector_negative"]["q1"][0]["gaussian_id"] == 11


def test_accepts_feedback_bank_aliases_and_manifest_split() -> None:
    payload = {
        "manifest": {"split_name": "selfmap_train"},
        "records": [
            {
                "query_id": "q2",
                "image_id": "im2",
                "matched_gaussian_id": "12",
                "matched_landmark_id": "landmark:12",
                "label": 1,
                "pnp_inlier": True,
                "pnp_success": True,
                "reprojection_error_px": 0.5,
                "descriptor_margin": 0.3,
                "keypoint_xy": [4.0, 5.0],
            }
        ],
    }

    impact = build_solver_feedback_impact(payload)

    assert impact["split_name"] == "selfmap_train"
    assert impact["landmark_positive"][12]["count"] == 1.0
    assert impact["detector_positive"]["q2"][0]["keypoint_xy"] == [4.0, 5.0]


def test_consumes_sparse_feedback_v4_correspondences_and_preserves_query_features() -> None:
    payload = {
        "schema_version": "sparse_solver_feedback_v4",
        "split_name": "selfmap_train",
        "query_features": {"q1": [0.1, 0.2, 0.3]},
        "split_audit": {
            "split_name": "selfmap_train",
            "test_split_used": False,
            "official_test_used": False,
        },
        "correspondences": [
            {
                "query_id": "q1",
                "image_id": "im1",
                "split_name": "selfmap_train",
                "gaussian_id": 10,
                "label_role": "protected_support",
                "pnp_inlier": True,
                "pose_success": True,
                "reprojection_error_px": 1.0,
                "descriptor_margin": 0.3,
                "keypoint_xy": [2.0, 3.0],
            },
            {
                "query_id": "q1",
                "image_id": "im1",
                "split_name": "selfmap_train",
                "gaussian_id": 11,
                "label_role": "harmful_negative",
                "pnp_inlier": False,
                "reprojection_error_px": 18.0,
                "descriptor_score": 0.95,
                "descriptor_margin": 0.02,
                "query_sparse_te_cm": 40.0,
                "keypoint_xy": [5.0, 6.0],
            },
            {
                "query_id": "q1",
                "image_id": "im1",
                "split_name": "selfmap_train",
                "gaussian_id": 12,
                "label_role": "neutral_outlier",
                "pnp_inlier": False,
                "reprojection_error_px": 50.0,
                "descriptor_score": 0.99,
                "descriptor_margin": 0.0,
                "query_sparse_te_cm": 40.0,
            },
        ],
    }

    impact = build_solver_feedback_impact(payload)

    assert impact["record_count"] == 3
    assert impact["query_features"] == {"q1": [0.1, 0.2, 0.3]}
    assert impact["landmark_positive"][10]["support"] > 0.0
    assert impact["landmark_negative"][11]["risk"] > 0.0
    assert 12 not in impact["landmark_negative"]
    assert impact["metadata"]["neutral_correspondence_count"] == 1


def test_consumes_risky_competitor_negative_as_solver_risk() -> None:
    payload = {
        "schema_version": "sparse_solver_feedback_v4",
        "split_name": "selfmap_train",
        "split_audit": {
            "split_name": "selfmap_train",
            "test_split_used": False,
            "official_test_used": False,
        },
        "correspondences": [
            {
                "query_id": "q_good",
                "image_id": "im1",
                "split_name": "selfmap_train",
                "gaussian_id": 31,
                "label_role": "risky_competitor_negative",
                "pnp_inlier": False,
                "reprojection_error_px": 16.0,
                "descriptor_score": 0.94,
                "descriptor_margin": 0.01,
                "query_sparse_te_cm": 3.0,
                "keypoint_xy": [7.0, 8.0],
            },
        ],
    }

    impact = build_solver_feedback_impact(payload)

    assert impact["landmark_negative"][31]["risk"] > 0.0
    assert impact["view_negative"][("31", "im1")]["weight"] > 0.0
    assert impact["detector_negative"]["q_good"][0]["gaussian_id"] == 31


def test_rejects_official_test_aliases() -> None:
    for split_name in ("official_test", "cambridge_test", "ShopFacade_test"):
        with pytest.raises(ValueError, match="test split"):
            build_solver_feedback_impact({"split_name": split_name, "records": []})
