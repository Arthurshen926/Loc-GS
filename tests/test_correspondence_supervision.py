from __future__ import annotations

import pytest

from loc_gs.feedback.correspondence_supervision import (
    build_correspondence_supervision,
    export_supervision_targets,
)
from loc_gs.feedback.schema import FeedbackMatchRecord


def test_correspondence_supervision_exports_five_target_views() -> None:
    records = [
        FeedbackMatchRecord(
            scene="ShopFacade",
            query_id="q0",
            matched_gaussian_id="1",
            keypoint_xy=(10.0, 12.0),
            query_xy_norm=(0.1, 0.2),
            descriptor_score=0.9,
            descriptor_margin=0.4,
            pnp_inlier=True,
            reprojection_error_px=0.5,
            detector_score=0.7,
            depth_m=4.0,
            local_geometry_score=0.8,
            image_cell=(1, 2),
            depth_bin=3,
            source_role="candidate_trace",
            pose_success=True,
            query_sparse_te_cm=8.0,
        ),
        FeedbackMatchRecord(
            scene="ShopFacade",
            query_id="q0",
            matched_gaussian_id="2",
            keypoint_xy=(14.0, 13.0),
            query_xy_norm=(0.12, 0.21),
            descriptor_score=0.88,
            descriptor_margin=0.01,
            pnp_inlier=False,
            reprojection_error_px=12.0,
            detector_score=0.5,
            depth_m=4.2,
            local_geometry_score=0.2,
        ),
    ]

    artifact = build_correspondence_supervision(records, split_name="selfmap_train")
    targets = export_supervision_targets(artifact)

    assert artifact["schema_version"] == "correspondence_supervision_v1"
    assert targets["descriptor_fusion"]["landmark_weights"][1] > targets["descriptor_fusion"]["landmark_weights"][2]
    assert targets["detector_target"]["keypoint_weights"][0] > targets["detector_target"]["keypoint_weights"][1]
    assert targets["match_scorer"]["labels"] == [1, 0]
    assert targets["match_scorer"]["query_xy_norm"][0] == [0.1, 0.2]
    assert targets["match_scorer"]["image_cell"][0] == [1, 2]
    assert targets["match_scorer"]["depth_bin"][0] == 3
    assert targets["match_scorer"]["detector_score"][0] == 0.7
    assert targets["pnp_ranking"]["pairwise_preferences"][0]["positive_gaussian_id"] == 1
    assert targets["conflict_graph"]["edges"][0]["src"] == 1
    assert targets["conflict_graph"]["edges"][0]["dst"] == 2


def test_correspondence_supervision_rejects_test_split() -> None:
    with pytest.raises(ValueError, match="test split"):
        build_correspondence_supervision([], split_name="test")


def test_correspondence_supervision_upweights_regression_query_hard_negatives() -> None:
    records = [
        FeedbackMatchRecord(
            scene="ShopFacade",
            query_id="q_regress",
            matched_gaussian_id="10",
            keypoint_xy=(10.0, 12.0),
            descriptor_score=0.91,
            descriptor_margin=0.02,
            pnp_inlier=False,
            reprojection_error_px=15.0,
            local_geometry_score=0.1,
        ),
        FeedbackMatchRecord(
            scene="ShopFacade",
            query_id="q_neutral",
            matched_gaussian_id="11",
            keypoint_xy=(11.0, 13.0),
            descriptor_score=0.91,
            descriptor_margin=0.02,
            pnp_inlier=False,
            reprojection_error_px=15.0,
            local_geometry_score=0.1,
        ),
        FeedbackMatchRecord(
            scene="ShopFacade",
            query_id="q_regress",
            matched_gaussian_id="12",
            keypoint_xy=(9.0, 9.0),
            descriptor_score=0.82,
            descriptor_margin=0.3,
            pnp_inlier=True,
            reprojection_error_px=0.8,
            local_geometry_score=0.8,
        ),
    ]

    artifact = build_correspondence_supervision(
        records,
        split_name="selfmap_train",
        query_regression_delta_cm={"q_regress": 60.0},
    )
    targets = export_supervision_targets(artifact)

    rows = artifact["records"]
    regress_negative = next(row for row in rows if row["query_id"] == "q_regress" and row["label"] == 0)
    neutral_negative = next(row for row in rows if row["query_id"] == "q_neutral")
    assert regress_negative["hard_negative"] is True
    assert regress_negative["supervision_weight"] > neutral_negative["supervision_weight"]
    assert targets["match_scorer"]["hard_negative"][0] is True
    assert targets["conflict_graph"]["edges"][0]["weight"] > 1.0
