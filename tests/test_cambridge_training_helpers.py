import torch
import torch.nn.functional as F

from loc_gs.scripts.train_cambridge_hybrid import (
    build_argparser,
    camera_centers_from_w2c,
    descriptor_residual_alignment_loss,
    extract_superpoint_teacher_batch,
    interpolate_pose_batch,
    locability_prior_alignment_loss,
    make_feature_renderer_intrinsics,
    normalize_position_map,
    pair_candidate_indices,
    pnp_feedback_detector_loss,
    pnp_feedback_detector_target,
    perturb_pose_batch,
    pose_delta_trans_rot,
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


def test_training_parser_defaults_to_conservative_reliability_recipe():
    args = build_argparser().parse_args([])

    assert args.batch_size == 8
    assert args.num_workers == 4
    assert args.pnp_weight == 0.1
    assert args.pnp_pose_loss_weight == 0.0
    assert args.pnp_reprojection_loss_weight == 0.0
    assert args.localization_descriptor_source == "hybrid_ply_gated_residual"
    assert args.hybrid_residual_alpha_max == 0.03
    assert args.same_view_match_weight == 1.0
    assert args.rehearsal_pose_mode == "perturb"
    assert args.rehearsal_pair_probability == 0.5


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
            "--pnp_feedback_detector_weight",
            "0.07",
            "--pnp_feedback_detector_sigma_px",
            "1.5",
            "--pnp_feedback_detector_init_path",
            "output/stdloc/map/scene/detector/30000_detector.pth",
            "--pnp_feedback_detector_anchor_weight",
            "0.2",
            "--pnp_feedback_detector_full_res",
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
            "--rehearsal_pose_mode",
            "mixed",
            "--rehearsal_pair_probability",
            "0.75",
            "--rehearsal_interpolation_min",
            "0.1",
            "--rehearsal_interpolation_max",
            "0.9",
            "--rehearsal_pair_jitter_trans_m",
            "0.2",
            "--rehearsal_pair_jitter_rot_deg",
            "7.0",
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
    assert args.pnp_feedback_detector_weight == 0.07
    assert args.pnp_feedback_detector_sigma_px == 1.5
    assert args.pnp_feedback_detector_init_path == "output/stdloc/map/scene/detector/30000_detector.pth"
    assert args.pnp_feedback_detector_anchor_weight == 0.2
    assert args.pnp_feedback_detector_full_res is True
    assert args.train_scaling is True
    assert args.lr_scaling == 3e-7
    assert args.external_match_supervision_weight == 0.4
    assert args.external_match_pipeline == "loftr"
    assert args.external_match_cache_root == "output/cache/matches"
    assert args.detector_free_hard_negative_weight == 0.2
    assert args.external_match_start_epoch == 3
    assert args.grad_accum_steps == 2
    assert args.rehearsal_pose_mode == "mixed"
    assert args.rehearsal_pair_probability == 0.75
    assert args.rehearsal_interpolation_min == 0.1
    assert args.rehearsal_interpolation_max == 0.9
    assert args.rehearsal_pair_jitter_trans_m == 0.2
    assert args.rehearsal_pair_jitter_rot_deg == 7.0


def test_interpolate_pose_batch_interpolates_camera_centers():
    pose_a = torch.eye(4).unsqueeze(0)
    pose_b = torch.eye(4).unsqueeze(0)
    # w2c translation -R * C, so this camera center is at x=2.
    pose_b[:, 0, 3] = -2.0

    mid = interpolate_pose_batch(pose_a, pose_b, 0.5)
    R_mid = mid[:, :3, :3]
    t_mid = mid[:, :3, 3]
    center_mid = -(R_mid.transpose(1, 2) @ t_mid.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(center_mid, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-5)


def test_interpolate_pose_batch_allows_mild_extrapolation():
    pose_a = torch.eye(4).unsqueeze(0)
    pose_b = torch.eye(4).unsqueeze(0)
    pose_b[:, 0, 3] = -2.0

    out = interpolate_pose_batch(pose_a, pose_b, 1.25)
    center = -(out[:, :3, :3].transpose(1, 2) @ out[:, :3, 3:].contiguous()).squeeze(-1)

    assert torch.allclose(center, torch.tensor([[2.5, 0.0, 0.0]]), atol=1e-5)


def test_pair_candidate_indices_mix_local_and_global_views():
    candidates = pair_candidate_indices(5, 20, local_offsets=(1, 4), global_bins=4)

    assert candidates[:4] == [6, 4, 9, 1]
    assert 0 in candidates and 10 in candidates and 15 in candidates
    assert 5 not in candidates
    assert len(candidates) == len(set(candidates))


def test_pose_delta_trans_rot_reports_camera_motion():
    pose_a = torch.eye(4).unsqueeze(0)
    pose_b = torch.eye(4).unsqueeze(0)
    pose_b[:, 0, 3] = -2.0
    theta = torch.deg2rad(torch.tensor(90.0))
    c = float(torch.cos(theta))
    s = float(torch.sin(theta))
    pose_b[:, :3, :3] = torch.tensor(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    centers = camera_centers_from_w2c(pose_b)
    trans, rot = pose_delta_trans_rot(pose_a, pose_b)

    assert torch.allclose(centers[:, :2].norm(dim=-1), torch.tensor([2.0]), atol=1e-5)
    assert torch.allclose(trans, torch.tensor([2.0]), atol=1e-5)
    assert torch.allclose(rot, torch.tensor([90.0]), atol=1e-4)


def test_perturb_pose_batch_identity_when_noise_disabled():
    pose = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)

    out = perturb_pose_batch(pose, 0.0, 0.0)

    assert torch.allclose(out, pose)


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


def test_pnp_feedback_detector_target_marks_inliers_and_hard_negatives():
    keypoints = torch.tensor([[[1.0, 1.0], [2.0, 3.0]]])
    scores = torch.tensor([[0.9, 0.0]])
    mask = torch.tensor([[True, True]])

    target, weight = pnp_feedback_detector_target(keypoints, scores, mask, height=4, width=5, sigma_px=0.5)

    assert target.shape == (1, 1, 4, 5)
    assert weight.shape == target.shape
    assert target[0, 0, 1, 1] > 0.8
    assert target[0, 0, 2, 3] < 0.1
    assert weight[0, 0, 2, 3] > 0.2


def test_pnp_feedback_detector_loss_rewards_feedback_aligned_scores():
    keypoints = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    scores = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, True]])
    good = torch.full((1, 1, 4, 4), 0.05)
    bad = torch.full((1, 1, 4, 4), 0.05)
    good[0, 0, 1, 1] = 0.95
    bad[0, 0, 1, 1] = 0.05

    good_loss = pnp_feedback_detector_loss(good, keypoints, scores, mask, sigma_px=0.5)
    bad_loss = pnp_feedback_detector_loss(bad, keypoints, scores, mask, sigma_px=0.5)

    assert good_loss < bad_loss
