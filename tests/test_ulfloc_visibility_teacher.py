import torch

from loc_gs.training.ulfloc_visibility_teacher import build_visibility_teacher_targets_with_metrics


def test_visibility_teacher_keeps_only_visible_mask_valid_points():
    projections = {
        "im1.png": [
            {"gaussian_id": 1, "keypoint_yx": [10.0, 20.0], "visible": True, "mask_valid": True},
            {"gaussian_id": 2, "keypoint_yx": [30.0, 40.0], "visible": True, "mask_valid": False},
            {"gaussian_id": 3, "keypoint_yx": [12.0, 20.0], "visible": False, "mask_valid": True},
            {"gaussian_id": 4, "keypoint_yx": [64.0, 10.0], "visible": True, "mask_valid": True},
            {"gaussian_id": 5, "keypoint_yx": [3.0], "visible": True, "mask_valid": True},
        ],
        "im2.png": [
            {"gaussian_id": 9, "keypoint_yx": [1.0, 2.0], "visible": True},
        ],
    }

    targets, _metrics = build_visibility_teacher_targets_with_metrics(projections, height=64, width=80)

    entry = targets["im1.png"]
    assert entry["positive_count"] == 1
    assert entry["gaussian_ids"].tolist() == [1]
    assert torch.allclose(entry["keypoint_yx"], torch.tensor([[10.0, 20.0]]))
    assert torch.allclose(entry["support_weights"], torch.ones(1))
    assert entry["target_role"] == "stdloc_visibility_teacher"
    assert targets["im2.png"]["gaussian_ids"].tolist() == [9]


def test_visibility_teacher_returns_compact_empty_entries():
    targets, _metrics = build_visibility_teacher_targets_with_metrics(
        {"empty.png": [{"gaussian_id": 7, "keypoint_yx": [5.0, 6.0], "visible": False, "mask_valid": True}]},
        height=16,
        width=20,
    )

    entry = targets["empty.png"]
    assert entry["positive_count"] == 0
    assert entry["gaussian_ids"].shape == (0,)
    assert entry["keypoint_yx"].shape == (0, 2)
    assert entry["support_weights"].shape == (0,)


def test_visibility_teacher_can_cap_points_per_image_deterministically():
    targets, metrics = build_visibility_teacher_targets_with_metrics(
        {
            "im.png": [
                {"gaussian_id": 7, "keypoint_yx": [7.0, 7.0], "visible": True, "weight": 0.1},
                {"gaussian_id": 3, "keypoint_yx": [3.0, 3.0], "visible": True, "weight": 0.9},
                {"gaussian_id": 5, "keypoint_yx": [5.0, 5.0], "visible": True, "weight": 0.9},
            ]
        },
        height=16,
        width=16,
        max_points_per_image=2,
    )

    entry = targets["im.png"]
    assert entry["gaussian_ids"].tolist() == [3, 5]
    assert entry["positive_count"] == 2
    assert metrics["input_projection_count"] == 3
    assert metrics["kept_projection_count"] == 2
    assert metrics["dropped_by_image_cap_count"] == 1
    assert metrics["max_points_per_image"] == 2
