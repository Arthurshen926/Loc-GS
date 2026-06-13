from __future__ import annotations

import pytest
import torch

from loc_gs.training.ulfloc_aggregation_samples import build_aggregation_samples, group_samples_by_landmark
from loc_gs.training.ulfloc_descriptor_aggregation import optimize_landmark_descriptor


def _feedback_v4(*, split_name: str = "selfmap_train") -> dict:
    def corr(gaussian_id: int, sampled_row: int, label_role: str, descriptor: list[float], *, source_role: str = "baseline_trace"):
        return {
            "scene": "ToyScene",
            "split_name": split_name,
            "query_id": f"q{gaussian_id}",
            "image_id": f"im{gaussian_id}",
            "gaussian_id": gaussian_id,
            "sampled_row": sampled_row,
            "query_keypoint_index": 0,
            "keypoint_xy": [5.0, 6.0],
            "query_xy_norm": [0.05, 0.12],
            "image_cell": 0,
            "query_descriptor": descriptor,
            "landmark_descriptor": [0.7, 0.7],
            "descriptor_score": 0.9,
            "descriptor_margin": 0.2,
            "detector_score": 0.7,
            "pnp_inlier": label_role in {"protected_support", "positive_inlier"},
            "reprojection_error_px": 1.0,
            "camera_xyz": [0.1, 0.2, 2.0],
            "depth_m": 2.0,
            "bearing": [0.0, 0.0, 1.0],
            "source_role": source_role,
            "label_role": label_role,
        }

    return {
        "schema_version": "sparse_solver_feedback_v4",
        "split_name": split_name,
        "split_audit": {"test_split_used": split_name == "test", "official_test_used": False},
        "correspondences": [
            corr(10, 0, "protected_support", [1.0, 0.0]),
            corr(10, 0, "positive_inlier", [0.9, 0.1]),
            corr(10, 0, "harmful_negative", [0.0, 1.0]),
            corr(10, 0, "risky_competitor_negative", [0.1, 0.9]),
            corr(10, 0, "neutral_outlier", [0.5, 0.5]),
            corr(11, 1, "positive_inlier", [0.0, 1.0], source_role="render_aug_trace"),
        ],
    }


def test_build_aggregation_samples_maps_feedback_roles_and_ignores_neutral():
    samples = build_aggregation_samples(_feedback_v4(), include_render_aug=False)

    assert [(sample.gaussian_id, sample.label_role, sample.target) for sample in samples] == [
        (10, "protected_support", 1),
        (10, "positive_inlier", 1),
        (10, "harmful_negative", 0),
        (10, "risky_competitor_negative", 0),
    ]
    assert samples[0].protected is True
    assert samples[0].view_id == "im10"
    assert samples[0].source_role == "baseline_trace"
    assert samples[0].descriptor.tolist() == pytest.approx([1.0, 0.0])

    grouped = group_samples_by_landmark(samples)
    assert sorted(grouped) == [10]
    assert len(grouped[10]) == 4


def test_build_aggregation_samples_can_include_render_aug_and_rejects_test_split():
    samples = build_aggregation_samples(_feedback_v4(), include_render_aug=True)
    assert {sample.gaussian_id for sample in samples} == {10, 11}

    with pytest.raises(ValueError, match="test split"):
        build_aggregation_samples(_feedback_v4(split_name="test"))
    payload = _feedback_v4()
    payload["split_audit"] = {"split_name": "test", "test_split_used": False, "official_test_used": False}
    with pytest.raises(ValueError, match="test split"):
        build_aggregation_samples(payload)
    payload = _feedback_v4()
    payload["split_audit"] = {"split_name": "other_train", "test_split_used": False, "official_test_used": False}
    with pytest.raises(ValueError, match="split_name mismatch"):
        build_aggregation_samples(payload)


def test_descriptor_aggregation_optimizes_view_selection_with_native_floor():
    native = torch.tensor([0.7, 0.7], dtype=torch.float32)
    positive = torch.tensor([1.0, 0.0], dtype=torch.float32)
    negative = torch.tensor([0.0, 1.0], dtype=torch.float32)
    samples = build_aggregation_samples(_feedback_v4(), include_render_aug=False)

    before = torch.nn.functional.normalize((positive + negative) / 2.0, dim=0)
    result = optimize_landmark_descriptor(
        native_descriptor=native,
        samples=samples,
        steps=80,
        lr=0.2,
        min_native_cosine=0.0,
    )
    fused = result.descriptor

    assert torch.dot(fused, torch.nn.functional.normalize(positive, dim=0)) > torch.dot(before, positive)
    assert torch.dot(fused, torch.nn.functional.normalize(negative, dim=0)) < torch.dot(before, negative)
    native_cosine = float(torch.dot(fused, torch.nn.functional.normalize(native, dim=0)).item())
    assert native_cosine >= 0.0
    assert result.metadata["positive_sample_count"] == 2
    assert result.metadata["negative_sample_count"] == 2
    assert result.metadata["negative_label_roles"] == ["harmful_negative", "risky_competitor_negative"]
