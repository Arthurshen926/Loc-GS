from __future__ import annotations

import pytest
import torch

from loc_gs.feedback.ulfloc_native_fusion_feedback import build_ulfloc_native_fusion_feedback_from_v4


def _feedback_v4(*, split_name: str = "selfmap_train") -> dict:
    def row(gaussian_id: int, image_id: str, label_role: str) -> dict:
        return {
            "scene": "ToyScene",
            "split_name": split_name,
            "query_id": image_id,
            "image_id": image_id,
            "gaussian_id": gaussian_id,
            "sampled_row": gaussian_id,
            "label_role": label_role,
            "source_role": "baseline_trace",
            "descriptor_score": 0.9,
            "descriptor_margin": 0.2,
            "reprojection_error_px": 1.0 if label_role != "harmful_negative" else 20.0,
        }

    return {
        "schema_version": "sparse_solver_feedback_v4",
        "split_name": split_name,
        "split_audit": {"split_name": split_name, "test_split_used": split_name == "test", "official_test_used": False},
        "correspondences": [
            row(1, "seq1/frame00001.png", "protected_support"),
            row(1, "seq1/frame00001.png", "positive_inlier"),
            row(2, "seq1/frame00001.png", "harmful_negative"),
            row(3, "seq1/frame00002.png", "neutral_outlier"),
        ],
    }


def test_build_ulfloc_native_fusion_feedback_from_v4_exports_full_and_view_weights() -> None:
    artifact = build_ulfloc_native_fusion_feedback_from_v4(
        _feedback_v4(),
        scene="ShopFacade",
        num_gaussians=5,
        alpha=0.5,
        risk_penalty=1.0,
        min_weight=0.25,
        max_weight=1.75,
    )

    weights = torch.as_tensor(artifact["landmark_weights"], dtype=torch.float32)
    assert artifact["schema_version"] == "ulfloc_solver_feedback_v1"
    assert artifact["split_name"] == "selfmap_train"
    assert weights.shape == (5,)
    assert weights[1] > 1.0
    assert weights[2] < 1.0
    assert weights[3].item() == pytest.approx(1.0)

    view = artifact["view_landmark_weights"]["seq1/frame00001.png"]
    by_id = dict(
        zip(
            torch.as_tensor(view["landmark_ids"], dtype=torch.long).tolist(),
            torch.as_tensor(view["weights"], dtype=torch.float32).tolist(),
        )
    )
    assert by_id[1] > 1.0
    assert by_id[2] < 1.0
    assert artifact["metadata"]["view_landmark_weight_summary"]["view_count"] == 1
    assert artifact["metadata"]["neutral_outlier_count"] == 1


def test_build_ulfloc_native_fusion_feedback_view_pruning_mode_only_drops_harmful_views() -> None:
    artifact = build_ulfloc_native_fusion_feedback_from_v4(
        _feedback_v4(),
        scene="ShopFacade",
        num_gaussians=5,
        alpha=0.5,
        risk_penalty=1.0,
        min_weight=0.0,
        max_weight=1.75,
        fusion_policy="view_pruning",
    )

    weights = torch.as_tensor(artifact["landmark_weights"], dtype=torch.float32)
    assert torch.allclose(weights, torch.ones(5, dtype=torch.float32))
    view = artifact["view_landmark_weights"]["seq1/frame00001.png"]
    by_id = dict(
        zip(
            torch.as_tensor(view["landmark_ids"], dtype=torch.long).tolist(),
            torch.as_tensor(view["weights"], dtype=torch.float32).tolist(),
        )
    )
    assert 1 not in by_id
    assert by_id[2] == pytest.approx(0.0)
    assert artifact["metadata"]["hyperparameters"]["fusion_policy"] == "view_pruning"
    assert artifact["metadata"]["weight_application"] == "absolute"


def test_build_ulfloc_native_fusion_feedback_prunes_risky_competitor_views() -> None:
    payload = _feedback_v4()
    payload["correspondences"] = [
        {
            "scene": "ToyScene",
            "split_name": "selfmap_train",
            "query_id": "q1",
            "image_id": "seq1/frame00003.png",
            "gaussian_id": 4,
            "sampled_row": 4,
            "label_role": "risky_competitor_negative",
            "source_role": "baseline_trace",
            "descriptor_score": 0.95,
            "descriptor_margin": 0.01,
            "reprojection_error_px": 20.0,
        }
    ]

    artifact = build_ulfloc_native_fusion_feedback_from_v4(
        payload,
        scene="ShopFacade",
        num_gaussians=5,
        fusion_policy="view_pruning",
        negative_roles=("harmful_negative", "risky_competitor_negative"),
    )

    view = artifact["view_landmark_weights"]["seq1/frame00003.png"]
    assert torch.as_tensor(view["landmark_ids"], dtype=torch.long).tolist() == [4]
    assert torch.as_tensor(view["weights"], dtype=torch.float32).tolist() == [0.0]
    assert artifact["metadata"]["negative_count"] == 1


def test_build_ulfloc_native_fusion_feedback_does_not_prune_risky_competitors_by_default() -> None:
    payload = _feedback_v4()
    payload["correspondences"] = [
        {
            "scene": "ToyScene",
            "split_name": "selfmap_train",
            "query_id": "q1",
            "image_id": "seq1/frame00003.png",
            "gaussian_id": 4,
            "sampled_row": 4,
            "label_role": "risky_competitor_negative",
            "source_role": "baseline_trace",
            "descriptor_score": 0.95,
            "descriptor_margin": 0.01,
            "reprojection_error_px": 20.0,
        }
    ]

    artifact = build_ulfloc_native_fusion_feedback_from_v4(
        payload,
        scene="ShopFacade",
        num_gaussians=5,
        fusion_policy="view_pruning",
    )

    assert artifact["view_landmark_weights"] == {}
    assert artifact["metadata"]["negative_count"] == 0
    assert artifact["metadata"]["ignored_negative_count"] == 1


def test_build_ulfloc_native_fusion_feedback_rejects_test_split() -> None:
    with pytest.raises(ValueError, match="test split"):
        build_ulfloc_native_fusion_feedback_from_v4(
            _feedback_v4(split_name="test"),
            scene="ShopFacade",
            num_gaussians=5,
        )
