import torch

import pytest

from loc_gs.scripts.train_ulfloc_scene_detector_online import (
    build_argparser,
    _detector_state_dict_from_checkpoint,
    _resize_image_to_longest_edge,
)
from loc_gs.training.ulfloc_online_scene_detector import (
    attach_solver_residuals_to_entry,
    attach_superpoint_teacher_to_entry,
    build_online_stdloc_projection_entry,
)


def test_build_online_stdloc_projection_entry_filters_render_visible_and_mask():
    gaussian_xyz = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=torch.float32,
    )
    sampled_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    world_to_camera = torch.eye(4, dtype=torch.float32)
    intrinsic = torch.tensor([[10.0, 0.0, 5.0], [0.0, 9.0, 5.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    render_visible = torch.tensor([True, False, True])
    stable_mask = torch.ones((10, 10), dtype=torch.bool)
    stable_mask[9, 5] = False

    entry, metrics = build_online_stdloc_projection_entry(
        gaussian_xyz=gaussian_xyz,
        sampled_idx=sampled_idx,
        world_to_camera=world_to_camera,
        intrinsic=intrinsic,
        height=10,
        width=10,
        image_id="q.png",
        render_visible_mask=render_visible,
        stable_mask=stable_mask,
    )

    assert entry["gaussian_ids"].tolist() == [0]
    assert entry["keypoint_yx"].tolist() == [[5.0, 5.0]]
    assert entry["positive_count"] == 1
    assert metrics["dropped_render_invisible_count"] == 1
    assert metrics["mask_invalid_projection_count"] == 1


def test_resize_image_to_longest_edge_matches_ulfloc_eval_canvas():
    image = torch.zeros((3, 1080, 1920), dtype=torch.float32)

    resized = _resize_image_to_longest_edge(image, 1600)

    assert tuple(resized.shape) == (3, 900, 1600)


def test_detector_state_dict_from_checkpoint_accepts_wrapped_state_dict():
    state = {"cnn.0.weight": torch.ones((1,), dtype=torch.float32)}
    payload = {"state_dict": state, "metadata": {"split_name": "train_selfmap"}}

    assert _detector_state_dict_from_checkpoint(payload) == state


def test_detector_state_dict_from_checkpoint_rejects_metric_only_payload():
    with pytest.raises(KeyError):
        _detector_state_dict_from_checkpoint({"metadata": {"loss": 1.0}})


def test_attach_solver_residuals_to_entry_adds_positive_and_negative_points():
    entry = {
        "height": 20,
        "width": 30,
        "keypoint_yx": torch.tensor([[5.0, 6.0]], dtype=torch.float32),
        "gaussian_ids": torch.tensor([1], dtype=torch.long),
        "support_weights": torch.tensor([1.0], dtype=torch.float32),
    }
    impact = {
        "detector_positive": {
            "q.png": [{"gaussian_id": 2, "keypoint_xy": [10.0, 11.0], "weight": 0.7}]
        },
        "detector_negative": {
            "q.png": [
                {"gaussian_id": 3, "keypoint_xy": [12.0, 13.0], "weight": 2.0},
                {"gaussian_id": 4, "keypoint_xy": [120.0, 130.0], "weight": 5.0},
            ]
        },
    }

    updated, metrics = attach_solver_residuals_to_entry(entry, impact, image_id="q.png")

    assert updated["solver_positive_gaussian_ids"].tolist() == [2]
    assert updated["solver_positive_keypoint_yx"].tolist() == [[11.0, 10.0]]
    assert updated["negative_gaussian_ids"].tolist() == [3]
    assert updated["negative_keypoint_yx"].tolist() == [[13.0, 12.0]]
    assert metrics["solver_positive_count"] == 1
    assert metrics["negative_count"] == 1
    assert metrics["out_of_bounds_negative_count"] == 1


def test_attach_superpoint_teacher_to_entry_converts_xy_to_yx_and_filters_bounds():
    entry = {"height": 20, "width": 30, "keypoint_yx": torch.tensor([[5.0, 6.0]])}
    detection = {
        "keypoints": torch.tensor([[10.0, 11.0], [100.0, 12.0]], dtype=torch.float32),
        "keypoint_scores": torch.tensor([0.7, 0.9], dtype=torch.float32),
    }

    updated, metrics = attach_superpoint_teacher_to_entry(
        entry,
        detection,
        height=20,
        width=30,
    )

    assert updated["sp_teacher_keypoint_yx"].tolist() == [[11.0, 10.0]]
    assert updated["sp_teacher_weights"].tolist() == pytest.approx([0.7])
    assert updated["sp_teacher_count"] == 1
    assert metrics["sp_teacher_count"] == 1
    assert metrics["sp_teacher_out_of_bounds_count"] == 1


def test_online_scene_detector_disables_superpoint_teacher_by_default():
    args = build_argparser().parse_args(
        [
            "--scene",
            "GreatCourt",
            "--source_path",
            "/tmp/source",
            "--model_path",
            "/tmp/model",
            "--input_log_dir",
            "/tmp/log",
            "--cfg",
            "/tmp/cfg.yaml",
            "--output_dir",
            "/tmp/out",
            "--split_name",
            "train_selfmap",
        ]
    )

    assert args.use_superpoint_teacher is False
