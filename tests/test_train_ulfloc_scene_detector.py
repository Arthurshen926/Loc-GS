import pytest
import torch

from loc_gs.training.ulfloc_scene_detector import (
    build_scene_detector_loss_terms,
    build_stdloc_fullres_detector_loss_terms,
    compose_teacher_preserving_detector_targets,
    load_detector_target_artifact,
    make_trainable_detector_input,
    pixel_yx_to_coarse_yx,
    rasterize_compact_detector_target,
    rasterize_fullres_scene_detector_components,
    rasterize_scene_detector_components,
    scene_detector_loss,
    scene_detector_loss_with_suppression,
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


def test_rasterize_compact_detector_target_can_return_separate_suppression_heatmap():
    entry = {
        "keypoint_yx": torch.tensor([[19.5, 27.5]], dtype=torch.float32),
        "support_weights": torch.tensor([1.0], dtype=torch.float32),
        "negative_keypoint_yx": torch.tensor([[43.5, 51.5]], dtype=torch.float32),
        "negative_weights": torch.tensor([2.0], dtype=torch.float32),
    }

    target, weight, suppression, metadata = rasterize_compact_detector_target(
        entry,
        coarse_height=8,
        coarse_width=8,
        stride=8,
        sigma_cells=0.1,
        return_suppression=True,
    )

    positive_index = 2 * 8 + 3
    negative_index = 5 * 8 + 6
    assert int(torch.argmax(target.reshape(-1)).item()) == positive_index
    assert weight.reshape(-1)[negative_index].item() == pytest.approx(0.0)
    assert suppression.shape == target.shape
    assert suppression.reshape(-1)[negative_index].item() > 0.5
    assert suppression.reshape(-1)[positive_index].item() == pytest.approx(0.0)
    assert metadata["negative_point_count"] == 1


def test_rasterize_scene_detector_components_separates_solver_positive_from_visibility_teacher():
    entry = {
        "keypoint_yx": torch.tensor([[19.5, 27.5]], dtype=torch.float32),
        "support_weights": torch.tensor([1.0], dtype=torch.float32),
        "solver_positive_keypoint_yx": torch.tensor([[43.5, 51.5]], dtype=torch.float32),
        "solver_positive_weights": torch.tensor([2.0], dtype=torch.float32),
        "negative_keypoint_yx": torch.tensor([[11.5, 35.5]], dtype=torch.float32),
        "negative_weights": torch.tensor([1.0], dtype=torch.float32),
    }

    teacher, visibility, solver_positive, solver_negative, metadata = rasterize_scene_detector_components(
        entry,
        coarse_height=8,
        coarse_width=8,
        stride=8,
        sigma_cells=0.1,
    )

    teacher_index = 2 * 8 + 3
    solver_positive_index = 5 * 8 + 6
    negative_index = 1 * 8 + 4
    assert int(torch.argmax(teacher.reshape(-1)).item()) == teacher_index
    assert int(torch.argmax(solver_positive.reshape(-1)).item()) == solver_positive_index
    assert solver_negative.reshape(-1)[negative_index] > 0.5
    assert visibility.shape == teacher.shape
    assert metadata["solver_positive_point_count"] == 1
    assert metadata["negative_point_count"] == 1


def test_rasterize_fullres_scene_detector_components_scales_source_points():
    entry = {
        "height": 100,
        "width": 200,
        "keypoint_yx": torch.tensor([[50.0, 100.0]], dtype=torch.float32),
        "support_weights": torch.tensor([1.0], dtype=torch.float32),
        "solver_positive_keypoint_yx": torch.tensor([[70.0, 160.0]], dtype=torch.float32),
        "solver_positive_weights": torch.tensor([0.8], dtype=torch.float32),
        "negative_keypoint_yx": torch.tensor([[20.0, 40.0]], dtype=torch.float32),
        "negative_weights": torch.tensor([0.7], dtype=torch.float32),
    }

    base, solver_positive, suppression, metadata = rasterize_fullres_scene_detector_components(
        entry,
        target_height=10,
        target_width=20,
        sigma_px=0.1,
    )

    assert base.shape == (1, 1, 10, 20)
    assert torch.argmax(base.reshape(-1)).item() == 5 * 20 + 10
    assert torch.argmax(solver_positive.reshape(-1)).item() == 7 * 20 + 16
    assert torch.argmax(suppression.reshape(-1)).item() == 2 * 20 + 4
    assert metadata["target_grid"] == "stdloc_full_resolution_feature_map"
    assert metadata["source_height"] == 100
    assert metadata["target_height"] == 10


def test_rasterize_fullres_scene_detector_components_uses_stdloc_binary_pixels():
    entry = {
        "height": 10,
        "width": 10,
        "keypoint_yx": torch.tensor([[5.4, 6.7]], dtype=torch.float32),
        "support_weights": torch.tensor([0.2], dtype=torch.float32),
    }

    base, solver_positive, suppression, metadata = rasterize_fullres_scene_detector_components(
        entry,
        target_height=10,
        target_width=10,
    )

    assert base[0, 0, 5, 6].item() == pytest.approx(1.0)
    assert base.sum().item() == pytest.approx(1.0)
    assert solver_positive.sum().item() == pytest.approx(0.0)
    assert suppression.sum().item() == pytest.approx(0.0)
    assert metadata["target_style"] == "stdloc_binary_projection"


def test_rasterize_fullres_scene_detector_components_can_smooth_projection_pixels():
    entry = {
        "height": 11,
        "width": 11,
        "keypoint_yx": torch.tensor([[5.0, 5.0]], dtype=torch.float32),
        "support_weights": torch.tensor([1.0], dtype=torch.float32),
    }

    base, solver_positive, suppression, metadata = rasterize_fullres_scene_detector_components(
        entry,
        target_height=11,
        target_width=11,
        sigma_px=1.5,
    )

    assert base[0, 0, 5, 5].item() == pytest.approx(1.0)
    assert base[0, 0, 5, 6].item() > 0.0
    assert base[0, 0, 4, 5].item() > 0.0
    assert base.sum().item() > 1.0
    assert solver_positive.sum().item() == pytest.approx(0.0)
    assert suppression.sum().item() == pytest.approx(0.0)
    assert metadata["sigma_px"] == pytest.approx(1.5)


def test_rasterize_fullres_scene_detector_components_merges_superpoint_teacher():
    entry = {
        "height": 10,
        "width": 10,
        "keypoint_yx": torch.tensor([[5.0, 6.0]], dtype=torch.float32),
        "support_weights": torch.tensor([1.0], dtype=torch.float32),
        "sp_teacher_keypoint_yx": torch.tensor([[2.0, 3.0]], dtype=torch.float32),
        "sp_teacher_weights": torch.tensor([0.9], dtype=torch.float32),
    }

    base, solver_positive, suppression, metadata = rasterize_fullres_scene_detector_components(
        entry,
        target_height=10,
        target_width=10,
    )

    assert base[0, 0, 5, 6].item() == pytest.approx(1.0)
    assert base[0, 0, 2, 3].item() == pytest.approx(1.0)
    assert base.sum().item() == pytest.approx(2.0)
    assert solver_positive.sum().item() == pytest.approx(0.0)
    assert suppression.sum().item() == pytest.approx(0.0)
    assert metadata["visibility_point_count"] == 1
    assert metadata["sp_teacher_point_count"] == 1
    assert metadata["point_count"] == 2


def test_rasterize_scene_detector_components_uses_true_superpoint_teacher_when_available():
    entry = {
        "keypoint_yx": torch.tensor([[19.5, 27.5]], dtype=torch.float32),
        "support_weights": torch.tensor([1.0], dtype=torch.float32),
        "sp_teacher_keypoint_yx": torch.tensor([[51.5, 11.5]], dtype=torch.float32),
        "sp_teacher_weights": torch.tensor([0.9], dtype=torch.float32),
    }

    teacher, visibility, solver_positive, solver_negative, metadata = rasterize_scene_detector_components(
        entry,
        coarse_height=8,
        coarse_width=8,
        stride=8,
        sigma_cells=0.1,
    )

    visibility_index = 2 * 8 + 3
    sp_teacher_index = 6 * 8 + 1
    assert int(torch.argmax(teacher.reshape(-1)).item()) == sp_teacher_index
    assert int(torch.argmax(visibility.reshape(-1)).item()) == visibility_index
    assert solver_positive.sum().item() == pytest.approx(0.0)
    assert solver_negative.sum().item() == pytest.approx(0.0)
    assert metadata["sp_teacher_point_count"] == 1
    assert metadata["visibility_point_count"] == 1
    assert metadata["sp_teacher_source"] == "superpoint_teacher"


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


def test_scene_detector_loss_with_suppression_penalizes_negative_points_separately():
    pred = torch.full((1, 1, 4, 4), 0.05, dtype=torch.float32)
    target = torch.zeros_like(pred)
    weight = torch.zeros_like(pred)
    suppression = torch.zeros_like(pred)
    suppression[..., 1, 1] = 1.0

    low_negative = scene_detector_loss_with_suppression(pred, target, weight, suppression)
    pred[..., 1, 1] = 0.95
    high_negative = scene_detector_loss_with_suppression(pred, target, weight, suppression)

    assert high_negative.item() > low_negative.item()


def test_scene_detector_loss_terms_exposes_teacher_visibility_and_solver_feedback_components():
    pred = torch.full((1, 1, 4, 4), 0.5, dtype=torch.float32)
    teacher = torch.zeros_like(pred)
    visibility = torch.zeros_like(pred)
    solver_positive = torch.zeros_like(pred)
    solver_negative = torch.zeros_like(pred)
    teacher[..., 1, 1] = 1.0
    visibility[..., 1, 1] = 1.0
    solver_positive[..., 2, 2] = 1.0
    solver_negative[..., 0, 0] = 1.0

    terms = build_scene_detector_loss_terms(
        pred,
        sp_teacher_target=teacher,
        visibility_target=visibility,
        solver_positive_target=solver_positive,
        solver_negative_target=solver_negative,
    )

    assert set(terms) == {
        "sp_teacher_loss",
        "visibility_loss",
        "solver_positive_residual_loss",
        "solver_negative_suppression_loss",
        "total_loss",
    }
    assert terms["solver_negative_suppression_loss"].item() > 0.0
    assert terms["total_loss"].item() >= terms["sp_teacher_loss"].item()


def test_stdloc_fullres_detector_loss_alpha_zero_matches_plain_bce():
    pred = torch.tensor([[[[0.2, 0.8], [0.6, 0.4]]]], dtype=torch.float32)
    base = torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]], dtype=torch.float32)
    solver_positive = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
    solver_negative = torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)

    terms = build_stdloc_fullres_detector_loss_terms(
        pred,
        base_target=base,
        solver_positive_target=solver_positive,
        solver_negative_target=solver_negative,
        residual_alpha=0.0,
        solver_negative_weight=3.0,
    )

    expected = torch.nn.functional.binary_cross_entropy(pred, base, reduction="mean")
    assert terms["stdloc_bce_loss"].item() == pytest.approx(expected.item())
    assert terms["solver_positive_residual_loss"].item() == pytest.approx(0.0)
    assert terms["solver_negative_suppression_loss"].item() == pytest.approx(0.0)
    assert terms["total_loss"].item() == pytest.approx(expected.item())


def test_stdloc_fullres_detector_loss_can_upweight_positive_projection_pixels():
    pred = torch.full((1, 1, 4, 4), 0.2, dtype=torch.float32)
    base = torch.zeros_like(pred)
    base[..., 2, 2] = 1.0
    solver_positive = torch.zeros_like(pred)
    solver_negative = torch.zeros_like(pred)

    plain = build_stdloc_fullres_detector_loss_terms(
        pred,
        base_target=base,
        solver_positive_target=solver_positive,
        solver_negative_target=solver_negative,
        residual_alpha=0.0,
    )
    weighted = build_stdloc_fullres_detector_loss_terms(
        pred,
        base_target=base,
        solver_positive_target=solver_positive,
        solver_negative_target=solver_negative,
        residual_alpha=0.0,
        background_weight=0.1,
        positive_weight=8.0,
    )

    assert weighted["stdloc_bce_loss"].item() > plain["stdloc_bce_loss"].item()
    assert weighted["solver_positive_residual_loss"].item() == pytest.approx(0.0)
    assert weighted["solver_negative_suppression_loss"].item() == pytest.approx(0.0)


def test_stdloc_fullres_detector_loss_adds_bounded_solver_residuals():
    low_pred = torch.full((1, 1, 3, 3), 0.1, dtype=torch.float32)
    high_pred = low_pred.clone()
    high_pred[..., 0, 0] = 0.9
    base = torch.zeros_like(low_pred)
    solver_positive = torch.zeros_like(low_pred)
    solver_negative = torch.zeros_like(low_pred)
    solver_positive[..., 1, 1] = 1.0
    solver_negative[..., 0, 0] = 1.0

    low_terms = build_stdloc_fullres_detector_loss_terms(
        low_pred,
        base_target=base,
        solver_positive_target=solver_positive,
        solver_negative_target=solver_negative,
        residual_alpha=0.1,
        solver_negative_weight=2.0,
    )
    high_terms = build_stdloc_fullres_detector_loss_terms(
        high_pred,
        base_target=base,
        solver_positive_target=solver_positive,
        solver_negative_target=solver_negative,
        residual_alpha=0.1,
        solver_negative_weight=2.0,
    )

    assert low_terms["solver_positive_residual_loss"].item() > 0.0
    assert low_terms["solver_negative_suppression_loss"].item() > 0.0
    assert high_terms["solver_negative_suppression_loss"].item() > low_terms["solver_negative_suppression_loss"].item()


def test_teacher_preserving_detector_target_keeps_solver_feedback_residual_small():
    teacher = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    visibility = torch.zeros_like(teacher)
    solver_positive = torch.zeros_like(teacher)
    solver_negative = torch.zeros_like(teacher)
    teacher[..., 1, 1] = 1.0
    visibility[..., 2, 2] = 1.0
    solver_positive[..., 3, 3] = 1.0
    solver_negative[..., 1, 1] = 1.0
    solver_negative[..., 0, 0] = 1.0

    positive, suppression = compose_teacher_preserving_detector_targets(
        teacher,
        visibility,
        solver_positive,
        solver_negative,
        residual_alpha=0.05,
    )

    assert positive[..., 1, 1].item() == pytest.approx(1.0)
    assert positive[..., 2, 2].item() == pytest.approx(0.05)
    assert positive[..., 3, 3].item() == pytest.approx(0.05)
    assert suppression[..., 1, 1].item() == pytest.approx(0.0)
    assert suppression[..., 0, 0].item() == pytest.approx(1.0)


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


def test_load_detector_target_artifact_rejects_official_test_alias(tmp_path):
    path = tmp_path / "detector_targets.pt"
    torch.save(
        {
            "split_name": "cambridge_test",
            "targets": {},
            "metadata": {},
            "split_audit": {"audit_status": "failed", "split_name": "cambridge_test"},
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
