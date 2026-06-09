import pytest
import torch

from loc_gs.training.ulfloc_scene_detector import (
    load_detector_target_artifact,
    make_trainable_detector_input,
    pixel_yx_to_coarse_yx,
    rasterize_compact_detector_target,
    scene_detector_loss,
)


def test_pixel_yx_to_coarse_yx_inverts_ulfloc_detector_coordinate_mapping():
    pixel_yx = torch.tensor([[19.5, 27.5], [35.5, 11.5]], dtype=torch.float32)

    coarse = pixel_yx_to_coarse_yx(pixel_yx, stride=8)

    assert torch.allclose(coarse, torch.tensor([[2.0, 3.0], [4.0, 1.0]]))


def test_rasterize_compact_detector_target_uses_coarse_grid():
    entry = {
        "keypoint_yx": torch.tensor([[19.5, 27.5]], dtype=torch.float32),
        "support_weights": torch.tensor([1.0], dtype=torch.float32),
    }

    target, weight, metadata = rasterize_compact_detector_target(
        entry,
        coarse_height=6,
        coarse_width=8,
        stride=8,
        sigma_cells=0.1,
    )

    assert target.shape == (1, 1, 6, 8)
    assert weight.shape == target.shape
    assert int(torch.argmax(target.reshape(-1)).item()) == 2 * 8 + 3
    assert metadata["point_count"] == 1


def test_rasterize_compact_detector_target_weights_solver_hard_negative_keypoints():
    entry = {
        "keypoint_yx": torch.tensor([[19.5, 27.5]], dtype=torch.float32),
        "support_weights": torch.tensor([1.0], dtype=torch.float32),
        "negative_keypoint_yx": torch.tensor([[43.5, 51.5]], dtype=torch.float32),
        "negative_weights": torch.tensor([2.0], dtype=torch.float32),
    }

    target, weight, metadata = rasterize_compact_detector_target(
        entry,
        coarse_height=8,
        coarse_width=8,
        stride=8,
        sigma_cells=0.1,
    )

    positive_index = 2 * 8 + 3
    negative_index = 5 * 8 + 6
    assert int(torch.argmax(target.reshape(-1)).item()) == positive_index
    assert target.reshape(-1)[negative_index].item() == pytest.approx(0.0)
    assert weight.reshape(-1)[negative_index].item() > 0.5
    assert metadata["point_count"] == 1
    assert metadata["negative_point_count"] == 1


def test_rasterize_compact_detector_target_uses_solver_validity_weights():
    entry = {
        "keypoint_yx": torch.tensor([[19.5, 27.5], [43.5, 51.5]], dtype=torch.float32),
        "support_weights": torch.tensor([1.0, 1.0], dtype=torch.float32),
        "solver_validity_weights": torch.tensor([1.0, 0.1], dtype=torch.float32),
    }

    target, weight, metadata = rasterize_compact_detector_target(
        entry,
        coarse_height=8,
        coarse_width=8,
        stride=8,
        sigma_cells=0.1,
        solver_validity_power=1.0,
    )

    high_valid_index = 2 * 8 + 3
    low_valid_index = 5 * 8 + 6
    assert target.reshape(-1)[high_valid_index] > target.reshape(-1)[low_valid_index]
    assert weight.reshape(-1)[high_valid_index] > weight.reshape(-1)[low_valid_index]
    assert metadata["solver_validity_enabled"] is True


def test_scene_detector_loss_weights_positive_targets():
    pred = torch.full((1, 1, 4, 4), 0.5, dtype=torch.float32)
    target = torch.zeros_like(pred)
    target[..., 2, 2] = 1.0
    weight = torch.ones_like(pred)

    base = scene_detector_loss(pred, target, weight, positive_weight=1.0)
    stronger = scene_detector_loss(pred, target, weight, positive_weight=8.0)

    assert stronger.item() > base.item()


def test_load_detector_target_artifact_rejects_test_split(tmp_path):
    path = tmp_path / "detector_targets.pt"
    torch.save(
        {
            "split_name": "test",
            "targets": {},
            "metadata": {},
            "split_audit": {"audit_status": "failed", "split_name": "test"},
        },
        path,
    )

    with pytest.raises(ValueError, match="test split"):
        load_detector_target_artifact(path)


def test_make_trainable_detector_input_converts_inference_tensor_for_backward():
    conv = torch.nn.Conv2d(1, 1, 1)
    with torch.inference_mode():
        feature = torch.ones(1, 2, 2)

    normal = make_trainable_detector_input(feature)
    loss = conv(normal.unsqueeze(0)).sum()
    loss.backward()

    assert conv.weight.grad is not None
