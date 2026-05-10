import torch
import torch.nn.functional as F

from loc_gs.scripts.train_cambridge_hybrid import (
    build_argparser,
    descriptor_residual_alignment_loss,
    extract_superpoint_teacher_batch,
    locability_prior_alignment_loss,
    make_feature_renderer_intrinsics,
    normalize_position_map,
    resize_teacher_outputs_to_feature_grid,
    scheduled_loss_weight,
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


def test_extract_superpoint_teacher_batch_recomputes_mixed_shape_cache_entries():
    class Entry:
        def __init__(self, desc_hw: tuple[int, int]):
            self.descriptor = torch.zeros(4, *desc_hw)
            self.detector_logits = torch.zeros(65, *desc_hw)

    class MixedShapeCache:
        def __init__(self):
            self.saved = []

        def load(self, name, map_location=None):
            return Entry((10, 10)) if name == "a.png" else Entry((5, 5))

        def save(self, name, descriptor, detector_logits):
            self.saved.append((name, tuple(descriptor.shape[-2:])))

    class DummyTeacher:
        def __call__(self, gray):
            batch, _c, height, width = gray.shape
            return (
                torch.ones(batch, 4, height // 8, width // 8),
                torch.ones(batch, 65, height // 8, width // 8),
            )

    desc, det, hits = extract_superpoint_teacher_batch(
        DummyTeacher(),
        torch.zeros(2, 3, 80, 80),
        ["a.png", "b.png"],
        cache=MixedShapeCache(),
    )

    assert desc.shape == (2, 4, 10, 10)
    assert det.shape == (2, 65, 10, 10)
    assert hits == [False, False]


def test_extract_superpoint_teacher_batch_recomputes_cache_with_unexpected_shape():
    class Entry:
        descriptor = torch.zeros(4, 5, 5)
        detector_logits = torch.zeros(65, 5, 5)

    class WrongShapeCache:
        def load(self, name, map_location=None):
            return Entry()

        def save(self, name, descriptor, detector_logits):
            pass

    class DummyTeacher:
        def __call__(self, gray):
            return torch.ones(2, 4, 10, 10), torch.ones(2, 65, 10, 10)

    desc, det, hits = extract_superpoint_teacher_batch(
        DummyTeacher(),
        torch.zeros(2, 3, 80, 80),
        ["a.png", "b.png"],
        cache=WrongShapeCache(),
        expected_hw=(10, 10),
    )

    assert desc.shape == (2, 4, 10, 10)
    assert det.shape == (2, 65, 10, 10)
    assert hits == [False, False]


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
            "--pnp_locability_target_prior_weight",
            "0.75",
            "--pnp_locability_target_prior_start_epoch",
            "2",
            "--pnp_locability_target_prior_warmup_epochs",
            "3",
            "--pnp_topk",
            "8",
            "--pnp_occlusion_depth_tolerance",
            "0.1",
            "--pnp_occlusion_depth_rel_tolerance",
            "0.03",
            "--pnp_gt_alpha_threshold",
            "0.2",
            "--pnp_start_epoch",
            "3",
            "--pnp_warmup_epochs",
            "2",
            "--locability_target_depth_weight",
            "0.4",
            "--locability_prior_target_weight",
            "0.6",
            "--locability_prior_target_start_epoch",
            "2",
            "--locability_prior_target_warmup_epochs",
            "3",
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
    assert args.pnp_locability_target_prior_weight == 0.75
    assert args.pnp_locability_target_prior_start_epoch == 2
    assert args.pnp_locability_target_prior_warmup_epochs == 3
    assert args.pnp_topk == 8
    assert args.pnp_occlusion_depth_tolerance == 0.1
    assert args.pnp_occlusion_depth_rel_tolerance == 0.03
    assert args.pnp_gt_alpha_threshold == 0.2
    assert args.pnp_start_epoch == 3
    assert args.pnp_warmup_epochs == 2
    assert args.locability_target_depth_weight == 0.4
    assert args.locability_prior_target_weight == 0.6
    assert args.locability_prior_target_start_epoch == 2
    assert args.locability_prior_target_warmup_epochs == 3
    assert args.teacher_feature_source == "original"


def test_scheduled_loss_weight_starts_late_and_warms_up_linearly():
    assert scheduled_loss_weight(epoch=1, base_weight=0.5, start_epoch=3, warmup_epochs=2) == 0.0
    assert scheduled_loss_weight(epoch=3, base_weight=0.5, start_epoch=3, warmup_epochs=2) == 0.25
    assert scheduled_loss_weight(epoch=4, base_weight=0.5, start_epoch=3, warmup_epochs=2) == 0.5
    assert scheduled_loss_weight(epoch=5, base_weight=0.5, start_epoch=3, warmup_epochs=2) == 0.5


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
            "--ply_residual_reg_weight",
            "0.05",
            "--ply_residual_reg_samples",
            "512",
            "--train_scaling",
            "--lr_scaling",
            "3e-7",
            "--external_match_supervision_weight",
            "0.4",
            "--external_match_pipeline",
            "loftr",
            "--external_match_cache_root",
            "output/cache/matches",
            "--detector_free_hard_negative_weight",
            "0.2",
            "--external_match_start_epoch",
            "3",
            "--grad_accum_steps",
            "2",
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
    assert args.ply_residual_reg_weight == 0.05
    assert args.ply_residual_reg_samples == 512
    assert args.train_scaling is True
    assert args.lr_scaling == 3e-7
    assert args.external_match_supervision_weight == 0.4
    assert args.external_match_pipeline == "loftr"
    assert args.external_match_cache_root == "output/cache/matches"
    assert args.detector_free_hard_negative_weight == 0.2
    assert args.external_match_start_epoch == 3
    assert args.grad_accum_steps == 2


def test_descriptor_residual_alignment_loss_rewards_matching_reference():
    reference = F.normalize(torch.rand(8, 16), p=2, dim=-1)
    aligned = reference.clone()
    shuffled = reference.flip(0)

    aligned_loss = descriptor_residual_alignment_loss(aligned, reference)
    shuffled_loss = descriptor_residual_alignment_loss(shuffled, reference)

    assert aligned_loss < 1e-5
    assert shuffled_loss > aligned_loss


def test_locability_prior_alignment_loss_rewards_matching_prior():
    target = torch.tensor([[[[0.9, 0.1], [0.7, 0.2]]]])
    aligned = target.clone()
    inverted = 1.0 - target

    aligned_loss = locability_prior_alignment_loss(aligned, target)
    inverted_loss = locability_prior_alignment_loss(inverted, target)

    assert aligned_loss < inverted_loss
