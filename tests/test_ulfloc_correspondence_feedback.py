from __future__ import annotations

import pytest
import torch

from loc_gs.feedback.correspondence_supervision import build_correspondence_supervision
from loc_gs.feedback.schema import FeedbackMatchRecord
from loc_gs.feedback.ulfloc_correspondence_feedback import (
    build_ulfloc_correspondence_feedback,
)


def _records() -> list[FeedbackMatchRecord]:
    return [
        FeedbackMatchRecord(
            scene="ShopFacade",
            query_id="seq/frame0001.png",
            matched_gaussian_id="3",
            keypoint_xy=(10.0, 20.0),
            query_xy_norm=(0.1, 0.2),
            descriptor_score=0.9,
            descriptor_margin=0.4,
            pnp_inlier=True,
            reprojection_error_px=0.5,
            local_geometry_score=0.8,
        ),
        FeedbackMatchRecord(
            scene="ShopFacade",
            query_id="seq/frame0001.png",
            matched_gaussian_id="5",
            keypoint_xy=(11.0, 20.0),
            query_xy_norm=(0.11, 0.2),
            descriptor_score=0.88,
            descriptor_margin=0.01,
            pnp_inlier=False,
            reprojection_error_px=18.0,
            local_geometry_score=0.2,
        ),
        FeedbackMatchRecord(
            scene="ShopFacade",
            query_id="seq/frame0002.png",
            matched_gaussian_id="7",
            keypoint_xy=(30.0, 40.0),
            query_xy_norm=(0.3, 0.4),
            descriptor_score=0.72,
            descriptor_margin=0.2,
            pnp_inlier=True,
            reprojection_error_px=2.0,
            local_geometry_score=0.7,
        ),
    ]


def test_build_ulfloc_correspondence_feedback_exports_ulf_compatible_weights() -> None:
    supervision = build_correspondence_supervision(_records(), split_name="selfmap_train")

    payload = build_ulfloc_correspondence_feedback(supervision, num_landmarks=10)

    weights = torch.as_tensor(payload["landmark_weights"], dtype=torch.float32)
    assert payload["schema_version"] == "ulfloc_solver_feedback_v1"
    assert payload["correspondence_schema_version"] == "ulfloc_correspondence_feedback_v1"
    assert payload["split_name"] == "selfmap_train"
    assert weights.shape == (10,)
    assert weights[3] > 1.0
    assert weights[7] > 1.0
    assert weights[5] < 1.0
    assert weights[0].item() == pytest.approx(1.0)
    assert "seq/frame0001.png" in payload["view_landmark_weights"]
    view = payload["view_landmark_weights"]["seq/frame0001.png"]
    assert torch.as_tensor(view["landmark_ids"], dtype=torch.long).tolist() == [3, 5]
    assert torch.as_tensor(view["weights"], dtype=torch.float32)[0] > 1.0
    assert torch.as_tensor(view["weights"], dtype=torch.float32)[1] < 1.0


def test_build_ulfloc_correspondence_feedback_carries_training_targets() -> None:
    supervision = build_correspondence_supervision(_records(), split_name="selfmap_train")

    payload = build_ulfloc_correspondence_feedback(supervision, num_landmarks=10)
    targets = payload["correspondence_targets"]

    assert targets["detector_target"]["records"][0]["weight"] > targets["detector_target"]["records"][1]["weight"]
    assert targets["match_scorer"]["labels"] == [1, 0, 1]
    assert targets["pnp_ranking"]["pairwise_preferences"][0]["positive_gaussian_id"] == 3
    assert targets["conflict_graph"]["edges"][0]["src"] == 3
    assert targets["conflict_graph"]["edges"][0]["dst"] == 5


def test_build_ulfloc_correspondence_feedback_rejects_invalid_landmark_ids() -> None:
    supervision = build_correspondence_supervision(_records(), split_name="selfmap_train")

    with pytest.raises(ValueError, match="outside num_landmarks"):
        build_ulfloc_correspondence_feedback(supervision, num_landmarks=5)

