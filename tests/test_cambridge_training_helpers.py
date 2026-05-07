import torch
import torch.nn.functional as F

from loc_gs.scripts.train_cambridge_hybrid import (
    build_argparser,
    make_feature_renderer_intrinsics,
    normalize_position_map,
    resize_teacher_outputs_to_feature_grid,
    superpoint_gray,
)


def test_make_feature_renderer_intrinsics_scales_full_camera_by_stride():
    K = torch.tensor([[800.0, 0.0, 320.0], [0.0, 600.0, 180.0], [0.0, 0.0, 1.0]])
    out = make_feature_renderer_intrinsics(K, stride=8)
    assert out["fx"] == 100.0
    assert out["fy"] == 75.0
    assert out["cx"] == 40.0
    assert out["cy"] == 22.5


def test_normalize_position_map_uses_scene_bounds():
    position_map = torch.tensor([[[[0.0, 1.0]], [[2.0, 3.0]], [[4.0, 5.0]]]])
    xyz = torch.tensor([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])
    normalized = normalize_position_map(position_map, xyz, margin=0.0)
    assert torch.allclose(normalized.min(), torch.tensor(0.0))
    assert torch.allclose(normalized.max(), torch.tensor(1.0))


def test_superpoint_gray_preserves_batch_and_range():
    rgb = torch.zeros(2, 3, 8, 8)
    rgb[:, 0] = 1.0
    gray = superpoint_gray(rgb)
    assert gray.shape == (2, 1, 8, 8)
    assert torch.allclose(gray, torch.full_like(gray, 0.299))


def test_resize_teacher_outputs_to_feature_grid_normalizes_descriptor():
    descriptor = torch.rand(1, 4, 6, 8)
    detector = torch.rand(1, 65, 6, 8)
    desc_small, det_small = resize_teacher_outputs_to_feature_grid(
        descriptor,
        detector,
        height=3,
        width=4,
    )

    assert desc_small.shape == (1, 4, 3, 4)
    assert det_small.shape == (1, 65, 3, 4)
    norms = desc_small.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_training_parser_exposes_same_view_geometric_match_options():
    args = build_argparser().parse_args(
        [
            "--same_view_match_weight",
            "2.0",
            "--same_view_locability_weight",
            "0.1",
            "--same_view_target_sigma_px",
            "0.5",
            "--init_checkpoint",
            "output/stdloc_hybrid/foo/latest.pth",
            "--pnp_temperature",
            "0.05",
            "--pnp_target_sigma_px",
            "1.0",
            "--teacher_feature_source",
            "original",
        ]
    )
    assert args.same_view_match_weight == 2.0
    assert args.same_view_locability_weight == 0.1
    assert args.same_view_target_sigma_px == 0.5
    assert args.init_checkpoint == "output/stdloc_hybrid/foo/latest.pth"
    assert args.pnp_temperature == 0.05
    assert args.pnp_target_sigma_px == 1.0
    assert args.teacher_feature_source == "original"


def test_training_parser_exposes_sota_extension_options():
    args = build_argparser().parse_args(
        [
            "--cross_view_weight",
            "1.5",
            "--hard_negative_weight",
            "0.25",
            "--memory_bank_size",
            "1024",
            "--memory_bank_momentum",
            "0.95",
            "--view_pair_min_overlap",
            "0.2",
            "--xview_start_epoch",
            "2",
            "--xview_positive_source",
            "model",
            "--hard_negative_start_epoch",
            "4",
            "--hard_negative_exclusion_radius",
            "2.0",
            "--landmark_budget",
            "5000",
            "--splatloc_saliency_prior_weight",
            "0.1",
            "--locability_ambiguity_weight",
            "0.3",
            "--locability_budget_weight",
            "0.2",
            "--key_gaussian_isotropy_weight",
            "0.01",
        ]
    )

    assert args.cross_view_weight == 1.5
    assert args.hard_negative_weight == 0.25
    assert args.memory_bank_size == 1024
    assert args.memory_bank_momentum == 0.95
    assert args.view_pair_min_overlap == 0.2
    assert args.xview_start_epoch == 2
    assert args.xview_positive_source == "model"
    assert args.hard_negative_start_epoch == 4
    assert args.hard_negative_exclusion_radius == 2.0
    assert args.landmark_budget == 5000
    assert args.splatloc_saliency_prior_weight == 0.1
    assert args.locability_ambiguity_weight == 0.3
    assert args.locability_budget_weight == 0.2
    assert args.key_gaussian_isotropy_weight == 0.01
