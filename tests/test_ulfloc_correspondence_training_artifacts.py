import pytest
import torch

from loc_gs.feedback.ulfloc_correspondence_training_artifacts import (
    build_ulfloc_correspondence_training_artifacts,
)


def _supervision():
    return {
        "schema_version": "correspondence_supervision_v1",
        "split_name": "train_dev_seed13_20p",
        "records": [
            {
                "query_id": "q0.png",
                "gaussian_id": 10,
                "keypoint_xy": [3.0, 4.0],
                "descriptor_score": 0.92,
                "descriptor_margin": 0.35,
                "reprojection_error_px": 0.4,
                "local_geometry_score": 0.8,
                "query_xy_norm": [0.2, 0.3],
                "image_cell": [1, 2],
                "depth_bin": 3,
                "detector_score": 0.7,
                "source_role": "candidate_trace",
                "pose_success": True,
                "label": 1,
                "pnp_inlier": True,
                "supervision_weight": 1.4,
            },
            {
                "query_id": "q0.png",
                "gaussian_id": 20,
                "keypoint_xy": [13.0, 11.0],
                "descriptor_score": 0.89,
                "descriptor_margin": 0.02,
                "reprojection_error_px": 12.0,
                "local_geometry_score": 0.2,
                "label": 0,
                "pnp_inlier": False,
                "supervision_weight": 0.5,
            },
            {
                "query_id": "q1.png",
                "gaussian_id": 30,
                "keypoint_xy": [1.0, 2.0],
                "descriptor_score": 0.7,
                "descriptor_margin": 0.2,
                "reprojection_error_px": 0.6,
                "local_geometry_score": 0.6,
                "label": 1,
                "pnp_inlier": True,
                "supervision_weight": 1.0,
            },
        ],
    }


def test_build_ulfloc_training_artifacts_materializes_detector_match_ranking_and_conflict():
    artifact = build_ulfloc_correspondence_training_artifacts(
        _supervision(),
        height=12,
        width=14,
        sigma_px=0.75,
        hard_negative_min_positive_distance_px=8.0,
    )

    assert artifact["split_audit"]["audit_status"] == "passed"
    assert artifact["metadata"]["positive_detector_point_count"] == 2
    assert artifact["metadata"]["hard_negative_detector_point_count"] == 1
    assert artifact["metadata"]["match_scorer_row_count"] == 3
    assert artifact["metadata"]["ranking_preference_count"] == 1
    assert artifact["metadata"]["conflict_edge_count"] == 1
    assert artifact["metadata"]["hard_negative_record_count"] == 0
    assert artifact["metadata"]["mean_positive_solver_validity"] > 0.0
    assert sorted(artifact["detector_targets"]) == ["q0.png", "q1.png"]

    q0_heatmap = artifact["detector_targets"]["q0.png"]["heatmap"]
    assert q0_heatmap.shape == (12, 14)
    assert int(torch.argmax(q0_heatmap).item()) == 4 * 14 + 3
    q0_target = artifact["detector_targets"]["q0.png"]
    assert q0_target["negative_count"] == 1
    assert q0_target["negative_keypoint_yx"].shape == (1, 2)
    assert q0_target["negative_weights"][0].item() > 0.0
    assert artifact["match_scorer"]["labels"].tolist() == [1, 0, 1]
    assert artifact["match_scorer"]["query_xy_norm"][0].tolist() == pytest.approx([0.2, 0.3])
    assert artifact["match_scorer"]["image_cell"][0].tolist() == [1, 2]
    assert artifact["match_scorer"]["depth_bin"][0].item() == 3
    assert artifact["match_scorer"]["detector_score"][0].item() == pytest.approx(0.7)
    assert artifact["match_scorer"]["source_role"][0] == "candidate_trace"
    assert artifact["pnp_ranking"]["pairwise_preferences"][0]["positive_gaussian_id"] == 10
    assert artifact["conflict_graph"]["edges"][0]["src"] == 10
    assert artifact["conflict_graph"]["edges"][0]["dst"] == 20


def test_detector_target_weights_are_geometry_and_quality_aware():
    payload = {
        "schema_version": "correspondence_supervision_v1",
        "split_name": "train_dev_seed13_20p",
        "records": [
            {
                "query_id": "q0.png",
                "gaussian_id": 1,
                "keypoint_xy": [10.0, 10.0],
                "query_xy_norm": [0.10, 0.10],
                "depth_m": 10.0,
                "descriptor_margin": 0.4,
                "reprojection_error_px": 0.5,
                "local_geometry_score": 1.0,
                "label": 1,
                "supervision_weight": 1.0,
            },
            {
                "query_id": "q0.png",
                "gaussian_id": 2,
                "keypoint_xy": [12.0, 11.0],
                "query_xy_norm": [0.11, 0.11],
                "depth_m": 10.3,
                "descriptor_margin": 0.4,
                "reprojection_error_px": 0.5,
                "local_geometry_score": 1.0,
                "label": 1,
                "supervision_weight": 1.0,
            },
            {
                "query_id": "q0.png",
                "gaussian_id": 3,
                "keypoint_xy": [80.0, 80.0],
                "query_xy_norm": [0.80, 0.80],
                "depth_m": 30.0,
                "descriptor_margin": 0.4,
                "reprojection_error_px": 0.5,
                "local_geometry_score": 1.0,
                "label": 1,
                "supervision_weight": 1.0,
            },
            {
                "query_id": "q0.png",
                "gaussian_id": 4,
                "keypoint_xy": [50.0, 50.0],
                "query_xy_norm": [0.50, 0.50],
                "depth_m": 20.0,
                "descriptor_margin": 0.0,
                "reprojection_error_px": 12.0,
                "local_geometry_score": 0.2,
                "label": 1,
                "supervision_weight": 1.0,
            },
        ],
    }

    artifact = build_ulfloc_correspondence_training_artifacts(payload, height=100, width=100)

    weights = artifact["detector_targets"]["q0.png"]["support_weights"]
    repeated_a, repeated_b, rare, low_quality = weights.tolist()
    assert rare > repeated_a
    assert rare > repeated_b
    assert low_quality < repeated_a
    assert artifact["metadata"]["detector_weighting"] == "pnp_geometry_aware_selective_hard_negative_suppression_v3"


def test_detector_target_can_downweight_solver_invalid_positives():
    payload = {
        "schema_version": "correspondence_supervision_v1",
        "split_name": "train_dev_seed13_20p",
        "records": [
            {
                "query_id": "q0.png",
                "gaussian_id": 1,
                "keypoint_xy": [10.0, 10.0],
                "descriptor_margin": 0.4,
                "reprojection_error_px": 0.5,
                "local_geometry_score": 1.0,
                "solver_validity": 1.0,
                "label": 1,
                "supervision_weight": 1.0,
            },
            {
                "query_id": "q0.png",
                "gaussian_id": 2,
                "keypoint_xy": [80.0, 80.0],
                "descriptor_margin": 0.4,
                "reprojection_error_px": 0.5,
                "local_geometry_score": 1.0,
                "solver_validity": 0.1,
                "label": 1,
                "supervision_weight": 1.0,
            },
        ],
    }

    artifact = build_ulfloc_correspondence_training_artifacts(
        payload,
        height=100,
        width=100,
        solver_validity_power=1.0,
    )

    target = artifact["detector_targets"]["q0.png"]
    assert target["solver_validity_weights"].tolist() == pytest.approx([1.0, 0.1])
    assert target["heatmap"][10, 10] > target["heatmap"][80, 80]
    assert target["target_metadata"]["solver_validity_enabled"] is True
    assert artifact["metadata"]["detector_solver_validity_power"] == 1.0


def test_detector_residual_target_is_clamped_teacher_residual_not_direct_suppression():
    payload = {
        "schema_version": "correspondence_supervision_v1",
        "split_name": "train_dev_seed13_20p",
        "records": [
            {
                "query_id": "q0.png",
                "gaussian_id": 1,
                "keypoint_xy": [10.0, 10.0],
                "descriptor_margin": 0.4,
                "reprojection_error_px": 0.5,
                "local_geometry_score": 1.0,
                "solver_validity": 1.0,
                "label": 1,
                "supervision_weight": 1.0,
            },
            {
                "query_id": "q0.png",
                "gaussian_id": 2,
                "keypoint_xy": [80.0, 80.0],
                "descriptor_margin": 0.4,
                "reprojection_error_px": 0.5,
                "local_geometry_score": 1.0,
                "solver_validity": 0.0,
                "label": 1,
                "supervision_weight": 1.0,
            },
        ],
    }

    artifact = build_ulfloc_correspondence_training_artifacts(
        payload,
        height=100,
        width=100,
        detector_residual_alpha=0.5,
    )

    target = artifact["detector_targets"]["q0.png"]
    teacher = target["teacher_support_weights"]
    residual = target["support_weights"]
    assert artifact["metadata"]["detector_solver_validity_power"] == 0.0
    assert artifact["metadata"]["detector_residual_alpha"] == pytest.approx(0.1)
    assert target["target_metadata"]["solver_validity_enabled"] is False
    assert target["detector_residual_alpha"] == pytest.approx(0.1)
    assert residual[0].item() <= teacher[0].item() * 1.1 + 1e-6
    assert residual[1].item() >= teacher[1].item() * 0.9 - 1e-6


def test_training_artifact_metrics_report_correspondence_hard_negative_strength():
    payload = _supervision()
    payload["records"][1]["hard_negative"] = True
    payload["records"][1]["negative_supervision_multiplier"] = 3.0

    artifact = build_ulfloc_correspondence_training_artifacts(payload, height=12, width=14)

    assert artifact["metadata"]["hard_negative_record_count"] == 1
    assert artifact["metadata"]["mean_hard_negative_multiplier"] == pytest.approx(3.0)


def test_detector_hard_negatives_are_spatially_selective_and_capped_per_query():
    records = [
        {
            "query_id": "q.png",
            "gaussian_id": 1,
            "keypoint_xy": [10.0, 10.0],
            "descriptor_score": 0.9,
            "descriptor_margin": 0.4,
            "reprojection_error_px": 0.5,
            "local_geometry_score": 1.0,
            "label": 1,
            "supervision_weight": 1.0,
        },
        {
            "query_id": "q.png",
            "gaussian_id": 2,
            "keypoint_xy": [12.0, 12.0],
            "descriptor_score": 0.95,
            "descriptor_margin": 0.01,
            "reprojection_error_px": 20.0,
            "local_geometry_score": 0.0,
            "label": 0,
            "supervision_weight": 2.0,
        },
    ]
    for idx, x in enumerate([60.0, 68.0, 76.0, 84.0, 92.0]):
        records.append(
            {
                "query_id": "q.png",
                "gaussian_id": 10 + idx,
                "keypoint_xy": [x, 80.0],
                "descriptor_score": 0.9 - 0.02 * idx,
                "descriptor_margin": 0.02,
                "reprojection_error_px": 16.0,
                "local_geometry_score": 0.1,
                "label": 0,
                "supervision_weight": 1.0,
            }
        )

    artifact = build_ulfloc_correspondence_training_artifacts(
        {"split_name": "train_dev_seed13_20p", "records": records},
        height=100,
        width=120,
        hard_negative_max_per_query=2,
        hard_negative_min_positive_distance_px=16.0,
        hard_negative_grid_size=1,
        hard_negative_max_per_cell=2,
    )

    target = artifact["detector_targets"]["q.png"]
    kept_gids = set(target["negative_gaussian_ids"].tolist())
    assert target["negative_count"] == 2
    assert 2 not in kept_gids
    assert artifact["metadata"]["hard_negative_detector_candidate_count"] == 6
    assert artifact["metadata"]["hard_negative_detector_point_count"] == 2


def test_detector_hard_negatives_can_be_disabled_for_positive_only_detector_targets():
    payload = _supervision()

    artifact = build_ulfloc_correspondence_training_artifacts(
        payload,
        height=12,
        width=14,
        hard_negative_max_per_query=0,
        hard_negative_min_positive_distance_px=8.0,
    )

    assert artifact["metadata"]["hard_negative_detector_candidate_count"] == 1
    assert artifact["metadata"]["hard_negative_detector_point_count"] == 0
    assert artifact["detector_targets"]["q0.png"]["negative_count"] == 0


def test_build_ulfloc_training_artifacts_rejects_test_split():
    payload = _supervision()
    payload["split_name"] = "test"

    with pytest.raises(ValueError, match="test split"):
        build_ulfloc_correspondence_training_artifacts(payload, height=8, width=8)
