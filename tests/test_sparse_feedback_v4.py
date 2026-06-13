import pytest

from loc_gs.feedback.sparse_feedback_v4 import build_sparse_feedback_v4


def _query(query_id, *, te_cm, pose_success, image_id=None, extra=None):
    row = {
        "scene": "ShopFacade",
        "split_name": "selfmap_train",
        "query_id": query_id,
        "image_id": image_id or query_id,
        "pose_success": pose_success,
        "sparse_te_cm": te_cm,
        "sparse_re_deg": 0.1,
        "catastrophic_failure": te_cm > 500.0,
        "inlier_count": 80 if pose_success else 5,
        "match_count": 120,
        "inlier_image_cell_count": 12 if pose_success else 1,
        "inlier_depth_bin_count": 4 if pose_success else 1,
        "bearing_spread": 0.3,
        "depth_spread": 2.0,
    }
    if extra:
        row.update(extra)
    return row


def _corr(query_id, gaussian_id, *, inlier, score, margin, reproj, extra=None):
    row = {
        "scene": "ShopFacade",
        "split_name": "selfmap_train",
        "query_id": query_id,
        "image_id": query_id,
        "gaussian_id": gaussian_id,
        "sampled_row": gaussian_id,
        "query_keypoint_index": 0,
        "keypoint_xy": [10.0, 20.0],
        "query_xy_norm": [0.1, 0.2],
        "image_cell": 3,
        "query_descriptor": [1.0, 0.0],
        "landmark_descriptor": [1.0, 0.0],
        "descriptor_score": score,
        "descriptor_margin": margin,
        "detector_score": 0.7,
        "pnp_inlier": inlier,
        "reprojection_error_px": reproj,
        "camera_xyz": [0.1, 0.2, 3.0],
        "depth_m": 3.0,
        "bearing": [0.0, 0.0, 1.0],
        "source_role": "baseline_trace",
    }
    if extra:
        row.update(extra)
    return row


def test_feedback_v4_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        build_sparse_feedback_v4({"split_name": "test", "queries": [], "correspondences": []})


def test_feedback_v4_rejects_audit_marked_test_split():
    with pytest.raises(ValueError, match="test split"):
        build_sparse_feedback_v4(
            {
                "split_name": "selfmap_train",
                "split_audit": {"test_split_used": True},
                "queries": [],
                "correspondences": [],
            }
        )


def test_feedback_v4_rejects_top_level_official_test_usage():
    with pytest.raises(ValueError, match="test split"):
        build_sparse_feedback_v4(
            {
                "split_name": "selfmap_train",
                "official_test_used": True,
                "queries": [],
                "correspondences": [],
            }
        )


def test_feedback_v4_rejects_row_level_test_split_or_mismatch():
    with pytest.raises(ValueError, match="test split"):
        build_sparse_feedback_v4(
            {
                "split_name": "selfmap_train",
                "queries": [_query("q_good", te_cm=3.0, pose_success=True, extra={"split_name": "test"})],
                "correspondences": [_corr("q_good", 10, inlier=True, score=0.9, margin=0.3, reproj=1.0)],
            }
        )
    with pytest.raises(ValueError, match="split_name mismatch"):
        build_sparse_feedback_v4(
            {
                "split_name": "selfmap_train",
                "queries": [_query("q_good", te_cm=3.0, pose_success=True)],
                "correspondences": [
                    _corr(
                        "q_good",
                        10,
                        inlier=True,
                        score=0.9,
                        margin=0.3,
                        reproj=1.0,
                        extra={"split_name": "other_train"},
                    )
                ],
            }
        )


def test_feedback_v4_requires_strong_query_and_correspondence_fields():
    with pytest.raises(ValueError, match="query.*image_id"):
        build_sparse_feedback_v4(
            {
                "split_name": "selfmap_train",
                "queries": [{"scene": "ShopFacade", "split_name": "selfmap_train", "query_id": "q"}],
                "correspondences": [],
            }
        )
    with pytest.raises(ValueError, match="correspondence.*source_role"):
        build_sparse_feedback_v4(
            {
                "split_name": "selfmap_train",
                "queries": [_query("q_good", te_cm=3.0, pose_success=True)],
                "correspondences": [
                    _corr("q_good", 10, inlier=True, score=0.9, margin=0.3, reproj=1.0, extra={"source_role": "oracle"})
                ],
            }
        )


def test_feedback_v4_positive_inlier_requires_descriptor_margin():
    feedback = build_sparse_feedback_v4(
        {
            "split_name": "selfmap_train",
            "queries": [_query("q_ok", te_cm=8.0, pose_success=True)],
            "correspondences": [
                _corr("q_ok", 10, inlier=True, score=0.9, margin=0.0, reproj=1.0),
            ],
        },
        min_descriptor_margin=0.1,
    )

    assert feedback["correspondences"][0]["label_role"] == "neutral_outlier"


def test_feedback_v4_allows_compact_descriptorless_correspondences_for_impact_attribution():
    row = _corr("q_good", 10, inlier=True, score=0.9, margin=0.3, reproj=1.0)
    row.pop("query_descriptor")
    row.pop("landmark_descriptor")

    feedback = build_sparse_feedback_v4(
        {
            "split_name": "selfmap_train",
            "queries": [_query("q_good", te_cm=3.0, pose_success=True)],
            "correspondences": [row],
        }
    )

    assert feedback["correspondences"][0]["label_role"] == "protected_support"
    assert "query_descriptor" not in feedback["correspondences"][0]
    assert "landmark_descriptor" not in feedback["correspondences"][0]


def test_feedback_v4_assigns_positive_harmful_and_protected_labels():
    payload = {
        "split_name": "selfmap_train",
        "queries": [
            _query("q_good", te_cm=3.0, pose_success=True),
            _query("q_ok", te_cm=14.0, pose_success=True, extra={"inlier_count": 30, "match_count": 100}),
            _query(
                "q_bad",
                te_cm=80.0,
                pose_success=False,
                extra={"sparse_re_deg": 2.0, "candidate_minus_baseline_te_cm": 40.0, "match_count": 100},
            ),
        ],
        "correspondences": [
            _corr("q_good", 10, inlier=True, score=0.9, margin=0.3, reproj=1.0),
            _corr("q_ok", 12, inlier=True, score=0.85, margin=0.2, reproj=2.0),
            _corr("q_bad", 11, inlier=False, score=0.95, margin=0.02, reproj=24.0),
            _corr("q_good", 13, inlier=False, score=0.3, margin=0.1, reproj=10.0),
        ],
    }

    feedback = build_sparse_feedback_v4(
        payload,
        protected_te_cm=10.0,
        hard_te_cm=20.0,
        harmful_regression_cm=20.0,
    )

    labels = {row["gaussian_id"]: row["label_role"] for row in feedback["correspondences"]}
    assert labels[10] == "protected_support"
    assert labels[12] == "positive_inlier"
    assert labels[11] == "harmful_negative"
    assert labels[13] == "neutral_outlier"
    assert feedback["query_index"]["q_good"]["protected_good_query"] is True
    assert feedback["query_index"]["q_bad"]["hard_query"] is True
    assert feedback["query_index"]["q_good"]["inlier_ratio"] == pytest.approx(80 / 120)
    assert feedback["metrics"]["protected_support_count"] == 1
    assert feedback["metrics"]["positive_inlier_count"] == 1
    assert feedback["metrics"]["harmful_negative_count"] == 1
    assert feedback["split_audit"]["official_test_used"] is False


def test_feedback_v4_labels_protected_good_high_score_outlier_as_risky_competitor():
    payload = {
        "split_name": "selfmap_train",
        "queries": [_query("q_good", te_cm=3.0, pose_success=True)],
        "correspondences": [
            _corr("q_good", 10, inlier=True, score=0.92, margin=0.3, reproj=1.0),
            _corr("q_good", 11, inlier=False, score=0.93, margin=0.01, reproj=18.0),
            _corr("q_good", 12, inlier=False, score=0.93, margin=0.4, reproj=18.0),
        ],
    }

    feedback = build_sparse_feedback_v4(
        payload,
        high_score_threshold=0.8,
        risky_competitor_max_margin=0.05,
    )

    labels = {row["gaussian_id"]: row["label_role"] for row in feedback["correspondences"]}
    assert labels[10] == "protected_support"
    assert labels[11] == "risky_competitor_negative"
    assert labels[12] == "neutral_outlier"
    assert feedback["metrics"]["risky_competitor_negative_count"] == 1


def test_feedback_v4_preserves_query_features_for_downstream_training():
    payload = {
        "split_name": "selfmap_train",
        "query_features": {"q_good": [0.1, 0.2, 0.3]},
        "query_match_descriptors": {"q_good": [[1.0, 0.0]]},
        "queries": [_query("q_good", te_cm=3.0, pose_success=True)],
        "correspondences": [_corr("q_good", 10, inlier=True, score=0.9, margin=0.3, reproj=1.0)],
    }

    feedback = build_sparse_feedback_v4(payload)

    assert feedback["query_features"] == {"q_good": [0.1, 0.2, 0.3]}
    assert feedback["query_match_descriptor_query_count"] == 1


def test_feedback_v4_includes_solver_causal_minimal_set_and_paired_fields():
    payload = {
        "split_name": "selfmap_train",
        "queries": [_query("q_pair", te_cm=3.0, pose_success=True)],
        "correspondences": [
            _corr("q_pair", 10, inlier=True, score=0.9, margin=0.3, reproj=1.0, extra={"camera_xyz": [0.0, 0.0, 3.0]}),
            _corr("q_pair", 11, inlier=True, score=0.9, margin=0.3, reproj=1.0, extra={"camera_xyz": [0.1, 0.0, 3.1]}),
            _corr("q_pair", 12, inlier=True, score=0.9, margin=0.3, reproj=1.0, extra={"camera_xyz": [0.0, 0.1, 3.2]}),
            _corr("q_pair", 13, inlier=True, score=0.9, margin=0.3, reproj=1.0, extra={"camera_xyz": [0.1, 0.1, 3.3]}),
            _corr(
                "q_pair",
                20,
                inlier=True,
                score=0.9,
                margin=0.3,
                reproj=1.0,
                extra={"source_role": "candidate_trace", "camera_xyz": [0.0, 0.0, 3.0]},
            ),
            _corr(
                "q_pair",
                21,
                inlier=True,
                score=0.9,
                margin=0.3,
                reproj=1.0,
                extra={"source_role": "candidate_trace", "camera_xyz": [0.8, 0.0, 3.5]},
            ),
            _corr(
                "q_pair",
                22,
                inlier=True,
                score=0.9,
                margin=0.3,
                reproj=1.0,
                extra={"source_role": "candidate_trace", "camera_xyz": [0.0, 0.8, 4.0]},
            ),
            _corr(
                "q_pair",
                23,
                inlier=True,
                score=0.9,
                margin=0.3,
                reproj=1.0,
                extra={"source_role": "candidate_trace", "camera_xyz": [0.8, 0.8, 4.5]},
            ),
            _corr(
                "q_pair",
                24,
                inlier=True,
                score=0.9,
                margin=0.3,
                reproj=1.0,
                extra={"source_role": "candidate_trace", "camera_xyz": [1.2, 0.4, 5.5]},
            ),
        ],
    }

    feedback = build_sparse_feedback_v4(payload)

    candidate_rows = [row for row in feedback["correspondences"] if row["source_role"] == "candidate_trace"]
    assert feedback["metrics"]["solver_causal_attribution_enabled"] is True
    assert feedback["metrics"]["paired_query_count"] == 1
    assert feedback["query_index"]["q_pair"]["pnp_logdet_jtj"] != 0.0
    assert candidate_rows[-1]["candidate_vs_baseline_support_delta"] == 1
    assert candidate_rows[-1]["candidate_vs_baseline_logdet_delta"] != 0.0
    assert candidate_rows[-1]["minimal_set_counterfactual_support_delta"] == -1
    assert "minimal_set_logdet_drop" in candidate_rows[-1]
