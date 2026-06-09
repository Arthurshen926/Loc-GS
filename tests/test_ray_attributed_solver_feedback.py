import pytest
import torch

from loc_gs.feedback.ray_attributed_solver_feedback import (
    accumulate_ray_feedback,
    artifact_score_from_ray,
    normalize_contributors,
    soft_point_from_contributors,
)


def test_normalize_contributors_uses_contribution_and_reliability():
    contributors = [
        {"gaussian_id": 0, "contribution": 0.7, "reliability": 1.0},
        {"gaussian_id": 1, "contribution": 0.3, "reliability": 0.5},
    ]

    normalized = normalize_contributors(contributors)

    assert normalized[0]["gaussian_id"] == 0
    assert normalized[0]["weight"] == pytest.approx(0.7 / 0.85)
    assert normalized[1]["weight"] == pytest.approx(0.15 / 0.85)
    assert sum(item["weight"] for item in normalized) == pytest.approx(1.0)


def test_soft_point_from_contributors_uses_normalized_ray_weights():
    contributors = [
        {"gaussian_id": 0, "contribution": 0.75, "xyz": [0.0, 0.0, 2.0]},
        {"gaussian_id": 1, "contribution": 0.25, "xyz": [4.0, 0.0, 6.0]},
    ]

    point = soft_point_from_contributors(contributors)

    assert point.tolist() == pytest.approx([1.0, 0.0, 3.0])


def test_artifact_score_from_ray_detects_near_occluding_gaussians():
    contributors = [
        {"gaussian_id": 0, "contribution": 0.8, "depth": 3.0},
        {"gaussian_id": 1, "contribution": 0.2, "depth": 10.0},
    ]

    score = artifact_score_from_ray(
        contributors,
        expected_depth=10.0,
        rendered_depth=3.0,
        depth_margin=2.0,
    )

    assert score > 0.7
    assert artifact_score_from_ray(contributors, expected_depth=10.0, rendered_depth=9.5) == pytest.approx(0.0)


def test_accumulate_ray_feedback_softly_distributes_solver_support():
    observations = [
        {
            "query_id": "q1",
            "source_view_id": "s1",
            "split_name": "selfmap_train",
            "pnp_inlier": True,
            "reprojection_error_px": 1.0,
            "descriptor_score": 0.8,
            "local_geometry_score": 0.9,
            "contributors": [
                {"gaussian_id": 0, "contribution": 0.7, "xyz": [0.0, 0.0, 2.0]},
                {"gaussian_id": 1, "contribution": 0.3, "xyz": [1.0, 0.0, 2.0]},
            ],
        }
    ]

    artifact = accumulate_ray_feedback(observations, num_gaussians=3, split_name="selfmap_train")

    assert artifact["observed_count"].tolist() == [1, 1, 0]
    assert artifact["positive_observed_count"].tolist() == [1, 1, 0]
    assert artifact["support_score"][0] > artifact["support_score"][1] > 0.0
    assert artifact["hard_negative_risk"][0] == pytest.approx(0.0)
    assert artifact["metadata"]["split_name"] == "selfmap_train"
    assert artifact["metadata"]["observation_count"] == 1


def test_accumulate_ray_feedback_keeps_per_query_support_for_set_coverage():
    observations = [
        {
            "query_id": "q_hard",
            "source_view_id": "s1",
            "split_name": "selfmap_train",
            "pnp_inlier": True,
            "reprojection_error_px": 1.0,
            "descriptor_score": 0.8,
            "local_geometry_score": 1.0,
            "contributors": [
                {"gaussian_id": 0, "contribution": 0.75},
                {"gaussian_id": 1, "contribution": 0.25},
            ],
        },
        {
            "query_id": "q_hard",
            "source_view_id": "s2",
            "split_name": "selfmap_train",
            "pnp_inlier": True,
            "reprojection_error_px": 3.0,
            "descriptor_score": 0.9,
            "local_geometry_score": 1.0,
            "contributors": [
                {"gaussian_id": 1, "contribution": 1.0},
            ],
        },
    ]

    artifact = accumulate_ray_feedback(observations, num_gaussians=3, split_name="selfmap_train")

    per_query = artifact["per_query_support"]
    assert set(per_query) == {"q_hard"}
    assert per_query["q_hard"][0] > 0.0
    assert per_query["q_hard"][1] > per_query["q_hard"][0]
    assert artifact["metadata"]["per_query_support_query_count"] == 1
    assert artifact["metadata"]["per_query_support_entry_count"] == 2


def test_accumulate_ray_feedback_keeps_query_observation_cells_for_geometry_coverage():
    observations = [
        {
            "query_id": "q_hard",
            "source_view_id": "s1",
            "split_name": "selfmap_train",
            "pnp_inlier": True,
            "reprojection_error_px": 1.0,
            "descriptor_score": 0.8,
            "local_geometry_score": 1.0,
            "image_cell": [3, 4],
            "depth_bin": 6,
            "contributors": [
                {"gaussian_id": 0, "contribution": 0.75},
                {"gaussian_id": 1, "contribution": 0.25},
            ],
        }
    ]

    artifact = accumulate_ray_feedback(observations, num_gaussians=3, split_name="selfmap_train")

    assert artifact["per_query_observations"]["q_hard"][0]["cell"] == [3, 4]
    assert artifact["per_query_observations"]["q_hard"][0]["depth_bin"] == 6
    assert artifact["per_query_observations"]["q_hard"][1]["cell"] == [3, 4]
    assert artifact["per_query_observations"]["q_hard"][1]["depth_bin"] == 6
    assert artifact["metadata"]["per_query_observation_query_count"] == 1
    assert artifact["metadata"]["per_query_observation_entry_count"] == 2


def test_accumulate_ray_feedback_preserves_pnp_observation_fields_for_sparse_set_optimization():
    observations = [
        {
            "query_id": "q_hard",
            "source_view_id": "s1",
            "split_name": "selfmap_train",
            "pnp_inlier": True,
            "query_xy_norm": [0.25, 0.75],
            "bearing": [0.1, -0.2, 1.0],
            "camera_xyz": [1.0, -0.5, 7.0],
            "expected_depth": 7.0,
            "rendered_depth": 3.0,
            "reprojection_error_px": 1.25,
            "descriptor_score": 0.8,
            "descriptor_margin": 0.42,
            "local_geometry_score": 0.9,
            "ray_entropy": 0.31,
            "contributors": [
                {"gaussian_id": 0, "contribution": 0.75, "depth": 3.0},
                {"gaussian_id": 1, "contribution": 0.25, "depth": 7.0},
            ],
        }
    ]

    artifact = accumulate_ray_feedback(observations, num_gaussians=2, split_name="selfmap_train")

    observation = artifact["per_query_observations"]["q_hard"][0]
    assert observation["xy_norm"] == pytest.approx([0.25, 0.75])
    assert observation["bearing"] == pytest.approx([0.1, -0.2, 1.0])
    assert observation["camera_xyz"] == pytest.approx([1.0, -0.5, 7.0])
    assert observation["depth"] == pytest.approx(7.0)
    assert observation["depth_bin"] == 7
    assert observation["reprojection_error_px"] == pytest.approx(1.25)
    assert observation["descriptor_margin"] == pytest.approx(0.42)
    assert observation["local_geometry_score"] == pytest.approx(0.9)
    assert observation["ray_entropy"] == pytest.approx(0.31)
    assert observation["ray_artifact_score"] > 0.0


def test_accumulate_ray_feedback_treats_non_inliers_as_weak_risk_not_hard_negatives():
    observations = [
        {
            "query_id": "q1",
            "source_view_id": "s1",
            "split_name": "selfmap_train",
            "pnp_inlier": False,
            "reprojection_error_px": 80.0,
            "descriptor_score": 0.95,
            "contributors": [
                {"gaussian_id": 0, "contribution": 1.0, "xyz": [0.0, 0.0, 2.0]},
            ],
        }
    ]

    artifact = accumulate_ray_feedback(
        observations,
        num_gaussians=1,
        split_name="selfmap_train",
        weak_outlier_weight=0.05,
    )

    assert artifact["support_score"][0] == pytest.approx(0.0)
    assert artifact["hard_negative_risk"][0] == pytest.approx(0.05)
    assert artifact["positive_observed_count"][0] == 0


def test_accumulate_ray_feedback_keeps_per_query_negative_support_for_view_deboost():
    observations = [
        {
            "query_id": "q1",
            "source_view_id": "s1",
            "split_name": "selfmap_train",
            "pnp_inlier": False,
            "reprojection_error_px": 80.0,
            "descriptor_score": 0.95,
            "contributors": [
                {"gaussian_id": 0, "contribution": 0.75},
                {"gaussian_id": 1, "contribution": 0.25},
            ],
        },
        {
            "query_id": "q2",
            "source_view_id": "s2",
            "split_name": "selfmap_train",
            "pnp_inlier": True,
            "reprojection_error_px": 1.0,
            "descriptor_score": 0.8,
            "expected_depth": 10.0,
            "rendered_depth": 3.0,
            "contributors": [
                {"gaussian_id": 1, "contribution": 1.0, "depth": 3.0},
            ],
        },
    ]

    artifact = accumulate_ray_feedback(
        observations,
        num_gaussians=2,
        split_name="selfmap_train",
        weak_outlier_weight=0.05,
    )

    negative = artifact["per_query_negative_support"]
    assert negative["q1"][0] > negative["q1"][1] > 0.0
    assert negative["q2"][1] > 0.0
    assert artifact["metadata"]["per_query_negative_support_query_count"] == 2
    assert artifact["metadata"]["per_query_negative_support_entry_count"] == 3


def test_accumulate_ray_feedback_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        accumulate_ray_feedback([], num_gaussians=1, split_name="test")
