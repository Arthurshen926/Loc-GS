#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from loc_gs.data.cambridge_dataset import CambridgeHybridDataset
from loc_gs.data.external_match_cache import ExternalMatchCache
from loc_gs.data.superpoint_cache import SuperPointTeacherCache
from loc_gs.losses.cross_view import (
    DescriptorMemoryBank,
    cross_view_projective_contrastive_loss,
    memory_bank_contrastive_loss,
    projective_view_overlap,
)
from loc_gs.losses.differentiable_pnp import DifferentiablePnPMatchLoss
from loc_gs.losses.external_match import external_match_supervision_loss
from loc_gs.losses.geometric_match import geometric_keypoint_match_loss
from loc_gs.losses.landmark_selection import (
    depth_consistency_score,
    descriptor_ambiguity_loss,
    geometric_mean_score,
    key_gaussian_isotropy_loss,
    locability_budget_loss,
    splatloc_saliency_prior,
    superpoint_detector_saliency,
)
from loc_gs.losses.localization_loss import prepare_superpoint_queries
from loc_gs.models.hybrid_gaussian import HybridFeatureGaussian, SuperPointOutputHead
from loc_gs.scripts.extract_superpoint_features import SuperPointNet


def superpoint_gray(rgb: torch.Tensor) -> torch.Tensor:
    weights = rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


def load_cambridge_rgb_no_resize(
    scene_root: Path,
    image_subdir: str,
    image_name: str,
    device: torch.device,
) -> torch.Tensor:
    candidates = []
    image_subdir = image_subdir.strip("/")
    if image_subdir:
        candidates.append(scene_root / image_subdir / image_name)
    candidates.append(scene_root / image_name)
    path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device=device)


def load_cambridge_rgb_no_resize_batch(
    scene_root: Path,
    image_subdir: str,
    image_names: list[str],
    device: torch.device,
) -> torch.Tensor:
    return torch.stack(
        [
            load_cambridge_rgb_no_resize(scene_root, image_subdir, image_name, device)
            for image_name in image_names
        ],
        dim=0,
    )


def resize_teacher_outputs_to_feature_grid(
    descriptor: torch.Tensor,
    detector_logits: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if descriptor.shape[-2:] != (int(height), int(width)):
        descriptor = F.interpolate(
            descriptor.float(),
            size=(int(height), int(width)),
            mode="bilinear",
            align_corners=False,
        )
    if detector_logits.shape[-2:] != (int(height), int(width)):
        detector_logits = F.interpolate(
            detector_logits.float(),
            size=(int(height), int(width)),
            mode="bilinear",
            align_corners=False,
        )
    return F.normalize(descriptor.float(), p=2, dim=1), detector_logits.float()


def make_feature_renderer_intrinsics(K: torch.Tensor, stride: int = 8) -> dict[str, float]:
    return {
        "fx": float(K[0, 0] / stride),
        "fy": float(K[1, 1] / stride),
        "cx": float(K[0, 2] / stride),
        "cy": float(K[1, 2] / stride),
    }


def normalize_position_map(
    position_map: torch.Tensor,
    xyz: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    lo = xyz.min(dim=0).values - float(margin)
    hi = xyz.max(dim=0).values + float(margin)
    extent = (hi - lo).clamp_min(1e-6)
    return ((position_map - lo.view(1, 3, 1, 1)) / extent.view(1, 3, 1, 1)).clamp(0.0, 1.0)


def perturb_pose_batch(
    pose_w2c: torch.Tensor,
    trans_m: float,
    rot_deg: float,
) -> torch.Tensor:
    if trans_m <= 0.0 and rot_deg <= 0.0:
        return pose_w2c.detach().clone()
    B = pose_w2c.shape[0]
    dtype = pose_w2c.dtype
    device = pose_w2c.device
    R_w2c = pose_w2c[:, :3, :3]
    t_w2c = pose_w2c[:, :3, 3]
    R_c2w = R_w2c.transpose(1, 2)
    t_c2w = -(R_c2w @ t_w2c.unsqueeze(-1)).squeeze(-1)

    dt = (torch.rand(B, 3, device=device, dtype=dtype) * 2.0 - 1.0) * float(trans_m)
    axis = F.normalize(torch.randn(B, 3, device=device, dtype=dtype), p=2, dim=-1)
    theta = (torch.rand(B, device=device, dtype=dtype) * 2.0 - 1.0) * math.radians(float(rot_deg))
    zeros = torch.zeros(B, device=device, dtype=dtype)
    kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
    Kmat = torch.stack(
        [zeros, -kz, ky, kz, zeros, -kx, -ky, kx, zeros],
        dim=-1,
    ).view(B, 3, 3)
    eye = torch.eye(3, device=device, dtype=dtype).expand(B, -1, -1)
    sin_t = torch.sin(theta).view(B, 1, 1)
    cos_t = torch.cos(theta).view(B, 1, 1)
    dR = eye + sin_t * Kmat + (1.0 - cos_t) * (Kmat @ Kmat)

    R_c2w_noisy = dR @ R_c2w
    t_c2w_noisy = t_c2w + dt
    R_w2c_noisy = R_c2w_noisy.transpose(1, 2)
    t_w2c_noisy = -(R_w2c_noisy @ t_c2w_noisy.unsqueeze(-1)).squeeze(-1)
    out = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    out[:, :3, :3] = R_w2c_noisy
    out[:, :3, 3] = t_w2c_noisy
    return out


def scheduled_loss_weight(
    epoch: int,
    base_weight: float,
    start_epoch: int = 1,
    warmup_epochs: int = 1,
) -> float:
    """Linearly enable a loss after a start epoch."""
    weight = float(base_weight)
    if weight == 0.0:
        return 0.0
    start = max(1, int(start_epoch))
    if int(epoch) < start:
        return 0.0
    warmup = max(1, int(warmup_epochs))
    progress = (int(epoch) - start + 1) / float(warmup)
    return weight * min(max(progress, 0.0), 1.0)


def default_ply_path(output_root: Path, scene: str) -> Path:
    return output_root / "stdloc" / "map_cambridge_spgs" / scene / "point_cloud" / "iteration_30000" / "point_cloud.ply"


def render_hybrid_superpoint(
    model: HybridFeatureGaussian,
    sp_head: SuperPointOutputHead,
    renderer,
    pose_w2c: torch.Tensor,
    descriptor_source: str = "hybrid",
    ply_loc_feature_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    from loc_gs.models.hybrid_gaussian import unproject_depth_to_positions

    render = renderer.render_features_batch(model, pose_w2c)
    depth_map = render["depth_map"].float()
    position_map = unproject_depth_to_positions(
        depth_map,
        pose_w2c.float(),
        renderer.K.float(),
        depth_map.shape[1],
        depth_map.shape[2],
    )
    position_map = normalize_position_map(position_map, model.get_xyz())
    fused = model.decode_screen_space(render["feature_map"].float(), position_map)
    loc_logits = model.get_locability_logits()
    locability = None
    if loc_logits.numel() > 0:
        loc_render = renderer.render_feature_values_batch(model, loc_logits, pose_w2c)
        locability = torch.sigmoid(loc_render["feature_map"].float())
    sp_out = sp_head(fused)
    descriptor = sp_out["descriptor"]
    if descriptor_source != "hybrid":
        ply_loc_feature = model.get_ply_loc_feature()
        if ply_loc_feature.numel() == 0 or ply_loc_feature.shape[1] == 0:
            raise ValueError("descriptor_source requires loc_* features in the input PLY")
        loc_render = renderer.render_feature_values_batch(
            model,
            F.normalize(ply_loc_feature.float(), p=2, dim=-1),
            pose_w2c,
        )
        ply_descriptor = F.normalize(loc_render["feature_map"].float(), p=2, dim=1)
        if descriptor_source == "ply_loc":
            descriptor = ply_descriptor
        elif descriptor_source == "hybrid_ply_blend":
            weight = min(max(float(ply_loc_feature_weight), 0.0), 1.0)
            descriptor = F.normalize(
                (1.0 - weight) * descriptor.float() + weight * ply_descriptor,
                p=2,
                dim=1,
            )
        else:
            raise ValueError(f"Unsupported descriptor_source: {descriptor_source}")
    return {
        "descriptor": descriptor,
        "detector": sp_out["detector"],
        "depth": depth_map,
        "alpha": render["alpha_map"],
        "locability": locability,
        "features": fused,
    }


def locability_target_prior_from_render(
    render: dict[str, torch.Tensor],
    detector_weight: float,
    alpha_weight: float,
    depth_weight: float,
    depth_window: int = 3,
) -> torch.Tensor | None:
    weights = {
        "detector": float(detector_weight),
        "alpha": float(alpha_weight),
        "depth": float(depth_weight),
    }
    if max(weights.values()) <= 0.0:
        return None
    detector = render.get("detector")
    depth = render.get("depth")
    alpha = render.get("alpha")
    if detector is None or depth is None or alpha is None:
        return None
    chunks = []
    for idx in range(detector.shape[0]):
        valid = torch.isfinite(depth[idx]) & (depth[idx] > 0.05) & (alpha[idx].float() > 0.0)
        components = {
            "detector": superpoint_detector_saliency(detector[idx]),
            "alpha": alpha[idx].float().clamp(0.0, 1.0),
            "depth": depth_consistency_score(depth[idx].float(), valid=valid, window_size=depth_window),
        }
        chunks.append(geometric_mean_score(components, weights))
    return torch.stack(chunks, dim=0).unsqueeze(1)


@torch.no_grad()
def extract_superpoint_teacher_batch(
    teacher: SuperPointNet,
    rgb: torch.Tensor,
    image_names: list[str],
    cache: SuperPointTeacherCache | None = None,
    expected_hw: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[bool]]:
    loaded_desc = []
    loaded_det = []
    loaded_flags = []
    if cache is not None:
        for name in image_names:
            entry = cache.load(name, map_location=rgb.device)
            if entry is None:
                loaded_flags.append(False)
                loaded_desc.append(None)
                loaded_det.append(None)
            else:
                loaded_flags.append(True)
                loaded_desc.append(entry.descriptor.to(rgb.device))
                loaded_det.append(entry.detector_logits.to(rgb.device))
    if cache is not None and all(loaded_flags):
        desc_shape = tuple(loaded_desc[0].shape)
        det_shape = tuple(loaded_det[0].shape)
        shapes_match = all(
            tuple(desc.shape) == desc_shape and tuple(det.shape) == det_shape
            for desc, det in zip(loaded_desc, loaded_det)
        )
        expected_match = True
        if expected_hw is not None:
            expected = tuple(int(v) for v in expected_hw)
            expected_match = desc_shape[-2:] == expected and det_shape[-2:] == expected
        if shapes_match and expected_match:
            return torch.stack(loaded_desc, dim=0), torch.stack(loaded_det, dim=0), loaded_flags
        loaded_flags = [False for _ in image_names]

    desc, det = teacher(superpoint_gray(rgb))
    if cache is not None:
        for i, name in enumerate(image_names):
            if not loaded_flags or not loaded_flags[i]:
                cache.save(name, desc[i], det[i])
    return desc, det, [False for _ in image_names]


@torch.no_grad()
def maybe_write_superpoint_metadata(
    cache: SuperPointTeacherCache | None,
    image_names: list[str],
    teacher_desc: torch.Tensor,
    teacher_det: torch.Tensor,
    query_keypoints: torch.Tensor,
    query_descs: torch.Tensor,
) -> None:
    if cache is None:
        return
    for i, name in enumerate(image_names):
        if not cache.metadata_path(name).exists():
            cache.save_metadata(
                name,
                keypoints=query_keypoints[i],
                keypoint_descriptors=query_descs[i],
                descriptor_dim=query_descs.shape[-1],
            )


def decode_gaussian_center_descriptors(
    model: HybridFeatureGaussian,
    sp_head: SuperPointOutputHead,
    ids: torch.Tensor,
) -> torch.Tensor:
    selected_xyz = model.get_xyz()[ids]
    selected_latent = model.get_latent()[ids]
    latent_map = selected_latent.T.contiguous().view(1, -1, ids.numel(), 1)
    pos_map = selected_xyz.T.contiguous().view(1, 3, ids.numel(), 1)
    pos_map = normalize_position_map(pos_map, model.get_xyz())
    fused = model.decode_screen_space(latent_map, pos_map)
    return sp_head(fused)["descriptor"].squeeze(0).squeeze(-1).T


def select_pair_batch(
    dataset: CambridgeHybridDataset,
    frame_indices: torch.Tensor,
    model: HybridFeatureGaussian,
    renderer,
    min_overlap: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    pair_items = []
    xyz_probe = model.get_xyz().detach()
    stride = max(1, xyz_probe.shape[0] // 4096)
    xyz_probe = xyz_probe[::stride]
    for frame_idx in frame_indices.detach().cpu().tolist():
        best_idx = (int(frame_idx) + 1) % len(dataset)
        best_overlap = -1.0
        base_pose = dataset[int(frame_idx)]["pose_w2c"].to(device=device, dtype=torch.float32)
        for offset in (1, 2, 4, 8, 16):
            cand_idx = (int(frame_idx) + offset) % len(dataset)
            cand_pose = dataset[cand_idx]["pose_w2c"].to(device=device, dtype=torch.float32)
            overlap = projective_view_overlap(
                xyz_probe,
                base_pose,
                cand_pose,
                renderer.K.float(),
                renderer.image_height,
                renderer.image_width,
            )
            if overlap > best_overlap:
                best_idx = cand_idx
                best_overlap = overlap
            if overlap >= float(min_overlap):
                break
        pair_items.append(dataset[best_idx])
    return {
        "rgb": torch.stack([item["rgb"] for item in pair_items], dim=0),
        "pose_w2c": torch.stack([item["pose_w2c"] for item in pair_items], dim=0),
        "image_name": [str(item["image_name"]) for item in pair_items],
    }


def _xy_to_feature_yx(
    keypoints_xy: np.ndarray,
    image_height: int,
    image_width: int,
    feature_height: int,
    feature_width: int,
) -> torch.Tensor:
    keypoints_xy = np.asarray(keypoints_xy, dtype=np.float32)
    if keypoints_xy.size == 0:
        return torch.zeros(0, 2)
    scale_y = float(feature_height) / max(float(image_height), 1.0)
    scale_x = float(feature_width) / max(float(image_width), 1.0)
    y = keypoints_xy[:, 1] * scale_y
    x = keypoints_xy[:, 0] * scale_x
    return torch.from_numpy(np.stack([y, x], axis=-1).astype(np.float32))


def load_external_match_training_batch(
    cache: ExternalMatchCache | None,
    image_names: list[str],
    pair_image_names: list[str],
    image_height: int,
    image_width: int,
    feature_height: int,
    feature_width: int,
    max_matches: int,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if cache is None:
        return None
    positives_a: list[torch.Tensor] = []
    positives_b: list[torch.Tensor] = []
    positives_s: list[torch.Tensor] = []
    negatives_b: list[torch.Tensor] = []
    max_pos = 0
    max_neg = 0
    for image_a, image_b in zip(image_names, pair_image_names):
        entry = cache.load_matches(image_a, image_b)
        reversed_pair = False
        if entry is None:
            entry = cache.load_matches(image_b, image_a)
            reversed_pair = entry is not None
        if entry is None:
            pos_a = torch.zeros(0, 2)
            pos_b = torch.zeros(0, 2)
            pos_s = torch.zeros(0)
            neg_b = torch.zeros(0, 2)
        else:
            kpts_a = entry.kpts_b_xy if reversed_pair else entry.kpts_a_xy
            kpts_b = entry.kpts_a_xy if reversed_pair else entry.kpts_b_xy
            inliers = entry.geom_inlier_mask.astype(bool)
            pos_a = _xy_to_feature_yx(kpts_a[inliers], image_height, image_width, feature_height, feature_width)
            pos_b = _xy_to_feature_yx(kpts_b[inliers], image_height, image_width, feature_height, feature_width)
            pos_s = torch.from_numpy(entry.scores[inliers].astype(np.float32))
            neg_b = _xy_to_feature_yx(kpts_b[~inliers], image_height, image_width, feature_height, feature_width)
            if pos_a.shape[0] > max_matches:
                order = pos_s.argsort(descending=True)[:max_matches]
                pos_a = pos_a[order]
                pos_b = pos_b[order]
                pos_s = pos_s[order]
        positives_a.append(pos_a)
        positives_b.append(pos_b)
        positives_s.append(pos_s)
        negatives_b.append(neg_b)
        max_pos = max(max_pos, pos_a.shape[0])
        max_neg = max(max_neg, neg_b.shape[0])
    if max_pos == 0:
        return None
    B = len(image_names)
    kpts_a = torch.zeros(B, max_pos, 2, device=device)
    kpts_b = torch.zeros(B, max_pos, 2, device=device)
    scores = torch.zeros(B, max_pos, device=device)
    valid = torch.zeros(B, max_pos, device=device, dtype=torch.bool)
    neg = torch.zeros(B, max(1, max_neg), 2, device=device)
    for b in range(B):
        n = positives_a[b].shape[0]
        if n:
            kpts_a[b, :n] = positives_a[b].to(device)
            kpts_b[b, :n] = positives_b[b].to(device)
            scores[b, :n] = positives_s[b].to(device)
            valid[b, :n] = True
        m = min(max(1, max_neg), negatives_b[b].shape[0])
        if m:
            neg[b, :m] = negatives_b[b][:m].to(device)
    return {
        "kpts_a_yx": kpts_a,
        "kpts_b_yx": kpts_b,
        "scores": scores,
        "valid": valid,
        "negative_kpts_b_yx": neg,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Cambridge hybrid hash+latent Loc-GS with differentiable matching/PnP")
    parser.add_argument("--scene", default="ShopFacade")
    parser.add_argument("--data_root", default="/mnt/pool/sqy/Cambridge_stdloc")
    parser.add_argument("--output_root", default="output")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--ply_path", default="")
    parser.add_argument("--cameras_json", default="")
    parser.add_argument("--image_subdir", default="processed")
    parser.add_argument("--superpoint_weights", default="third_party/stdloc/encoders/sp_encoder/weights/superpoint_v1.pth")
    parser.add_argument("--teacher_feature_source", choices=["resized", "original"], default="resized")
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=360)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr_latent", type=float, default=1e-3)
    parser.add_argument("--lr_hash", type=float, default=1e-3)
    parser.add_argument("--lr_decoder", type=float, default=2e-4)
    parser.add_argument("--lr_head", type=float, default=2e-4)
    parser.add_argument("--lr_locability", type=float, default=1e-4)
    parser.add_argument("--lr_xyz", type=float, default=2e-6)
    parser.add_argument("--lr_opacity", type=float, default=1e-5)
    parser.add_argument("--lr_scaling", type=float, default=5e-7)
    parser.add_argument("--train_xyz", action="store_true")
    parser.add_argument("--train_opacity", action="store_true")
    parser.add_argument("--train_scaling", action="store_true")
    parser.add_argument("--freeze_feature_field", action="store_true")
    parser.add_argument("--freeze_superpoint_head", action="store_true")
    parser.add_argument("--geometry_unfreeze_epoch", type=int, default=3)
    parser.add_argument("--geometry_reg_weight", type=float, default=1.0)
    parser.add_argument("--sp_recon_weight", type=float, default=0.05)
    parser.add_argument("--detector_recon_weight", type=float, default=0.005)
    parser.add_argument("--pnp_weight", type=float, default=1.0)
    parser.add_argument("--pnp_start_epoch", type=int, default=1)
    parser.add_argument("--pnp_warmup_epochs", type=int, default=1)
    parser.add_argument("--pnp_temperature", type=float, default=0.07)
    parser.add_argument("--pnp_target_sigma_px", type=float, default=2.0)
    parser.add_argument("--pnp_pose_loss_weight", type=float, default=1.0)
    parser.add_argument("--pnp_match_loss_weight", type=float, default=0.5)
    parser.add_argument("--pnp_quality_loss_weight", type=float, default=0.5)
    parser.add_argument("--pnp_reprojection_loss_weight", type=float, default=0.5)
    parser.add_argument("--pnp_observability_loss_weight", type=float, default=0.02)
    parser.add_argument("--pnp_locability_loss_weight", type=float, default=0.05)
    parser.add_argument("--pnp_locability_prior_weight", type=float, default=0.1)
    parser.add_argument("--pnp_locability_target_prior_weight", type=float, default=0.0)
    parser.add_argument("--locability_target_detector_weight", type=float, default=1.0)
    parser.add_argument("--locability_target_alpha_weight", type=float, default=0.25)
    parser.add_argument("--locability_target_depth_weight", type=float, default=0.5)
    parser.add_argument("--locability_target_depth_window", type=int, default=3)
    parser.add_argument("--same_view_match_weight", type=float, default=0.0)
    parser.add_argument("--same_view_locability_weight", type=float, default=0.0)
    parser.add_argument("--same_view_temperature", type=float, default=0.07)
    parser.add_argument("--same_view_target_sigma_px", type=float, default=1.0)
    parser.add_argument("--same_view_alpha_threshold", type=float, default=0.05)
    parser.add_argument("--locability_sparse_weight", type=float, default=0.001)
    parser.add_argument("--cross_view_weight", type=float, default=0.0)
    parser.add_argument("--hard_negative_weight", type=float, default=0.0)
    parser.add_argument("--memory_bank_size", type=int, default=4096)
    parser.add_argument("--memory_bank_momentum", type=float, default=0.99)
    parser.add_argument("--view_pair_min_overlap", type=float, default=0.05)
    parser.add_argument("--xview_start_epoch", type=int, default=2)
    parser.add_argument("--xview_positive_source", choices=["teacher", "model"], default="teacher")
    parser.add_argument("--hard_negative_start_epoch", type=int, default=4)
    parser.add_argument("--hard_negative_exclusion_radius", type=float, default=1.0)
    parser.add_argument("--xview_depth_tolerance", type=float, default=0.2)
    parser.add_argument("--xview_depth_rel_tolerance", type=float, default=0.05)
    parser.add_argument("--external_match_supervision_weight", type=float, default=0.0)
    parser.add_argument("--external_match_pipeline", default="superpoint+lightglue")
    parser.add_argument("--external_match_cache_root", default="")
    parser.add_argument("--detector_free_hard_negative_weight", type=float, default=0.0)
    parser.add_argument("--external_match_start_epoch", type=int, default=2)
    parser.add_argument("--landmark_budget", type=int, default=16384)
    parser.add_argument("--splatloc_saliency_prior_weight", type=float, default=0.0)
    parser.add_argument("--locability_ambiguity_weight", type=float, default=0.0)
    parser.add_argument("--locability_budget_weight", type=float, default=0.0)
    parser.add_argument("--key_gaussian_isotropy_weight", type=float, default=0.0)
    parser.add_argument("--superpoint_cache_root", default="")
    parser.add_argument("--disable_superpoint_cache", action="store_true")
    parser.add_argument("--localization_keypoints", type=int, default=256)
    parser.add_argument("--keypoint_threshold", type=float, default=0.015)
    parser.add_argument("--nms_radius", type=int, default=2)
    parser.add_argument("--pose_noise_trans_m", type=float, default=0.20)
    parser.add_argument("--pose_noise_rot_deg", type=float, default=10.0)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hash_output_dim", type=int, default=64)
    parser.add_argument("--fine_dim", type=int, default=128)
    parser.add_argument("--coarse_dim", type=int, default=128)
    parser.add_argument("--hybrid_output_dim", type=int, default=256)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    return parser


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = build_argparser().parse_args() if args is None else args
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_root)
    scene_root = Path(args.data_root) / args.scene
    out_dir = Path(args.output_dir) if args.output_dir else output_root / "stdloc_hybrid" / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_path = Path(args.ply_path) if args.ply_path else default_ply_path(output_root, args.scene)
    cameras_json = Path(args.cameras_json) if args.cameras_json else ply_path.parents[2] / "cameras.json"
    if not ply_path.exists():
        raise FileNotFoundError(f"Initial Gaussian PLY not found: {ply_path}")
    if not cameras_json.exists():
        raise FileNotFoundError(f"Cambridge cameras.json not found: {cameras_json}")

    dataset = CambridgeHybridDataset(
        scene_root=scene_root,
        cameras_json=cameras_json,
        split="train",
        image_subdir=args.image_subdir,
        image_height=args.image_height,
        image_width=args.image_width,
        max_frames=args.max_frames,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    first = dataset[0]
    K_full = first["K"].float()
    intr = make_feature_renderer_intrinsics(K_full, stride=8)
    feature_height = args.image_height // 8
    feature_width = args.image_width // 8

    from loc_gs.rendering.feature_renderer import FeatureFieldRenderer

    renderer = FeatureFieldRenderer(
        image_height=feature_height,
        image_width=feature_width,
        fx=intr["fx"],
        fy=intr["fy"],
        cx=intr["cx"],
        cy=intr["cy"],
        max_channels_per_chunk=32,
    ).to(device)

    model = HybridFeatureGaussian(
        latent_dim=args.latent_dim,
        hash_output_dim=args.hash_output_dim,
        fine_dim=args.fine_dim,
        coarse_dim=args.coarse_dim,
        output_dim=args.hybrid_output_dim,
    )
    model.load_from_ply(str(ply_path))
    model = model.to(device)
    if args.train_xyz or args.train_opacity or args.train_scaling:
        model.enable_geometry_training(
            train_xyz=args.train_xyz,
            train_opacity=args.train_opacity,
            train_scaling=args.train_scaling,
        )
        model._xyz.requires_grad_(False) if args.train_xyz else None
        model._opacity.requires_grad_(False) if args.train_opacity else None
        model._scaling.requires_grad_(False) if args.train_scaling else None
    xyz0 = model.get_xyz().detach().clone()
    opacity0 = model.get_opacity_logits().detach().clone()
    scaling0 = model._scaling.detach().clone()

    sp_head = SuperPointOutputHead(
        fused_dim=args.hybrid_output_dim,
        descriptor_dim=256,
        detector_dim=65,
        hidden_dim=256,
        num_res_blocks=2,
        use_3x3=True,
    ).to(device)

    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(init_ckpt["model_state_dict"], strict=False)
        sp_head.load_state_dict(init_ckpt["sp_head_state_dict"], strict=False)
        print(f"[train] initialized hybrid field/head from {args.init_checkpoint}")

    if args.freeze_feature_field:
        model._latent.requires_grad_(False)
        for module in (model.hash_field, model.fine_decoder, model.coarse_decoder, model.fusion_head):
            for param in module.parameters():
                param.requires_grad_(False)
    if args.freeze_superpoint_head:
        for param in sp_head.parameters():
            param.requires_grad_(False)

    teacher = SuperPointNet().to(device)
    teacher.load_state_dict(torch.load(args.superpoint_weights, map_location=device), strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    sp_cache = None
    if not args.disable_superpoint_cache:
        cache_split = "train" if args.teacher_feature_source == "resized" else f"train_{args.teacher_feature_source}"
        sp_cache = SuperPointTeacherCache(
            args.superpoint_cache_root or output_root,
            scene=args.scene,
            split=cache_split,
        )
    external_match_cache = None
    if args.external_match_supervision_weight > 0.0:
        external_match_cache = ExternalMatchCache(
            args.external_match_cache_root or output_root,
            scene=args.scene,
            pipeline=args.external_match_pipeline,
            split="train",
        )

    pnp_loss_fn = DifferentiablePnPMatchLoss(
        temperature=args.pnp_temperature,
        target_sigma_px=args.pnp_target_sigma_px,
        pose_weight=args.pnp_pose_loss_weight,
        match_weight=args.pnp_match_loss_weight,
        quality_weight=args.pnp_quality_loss_weight,
        reprojection_weight=args.pnp_reprojection_loss_weight,
        observability_weight=args.pnp_observability_loss_weight,
        locability_weight=args.pnp_locability_loss_weight,
        locability_prior_weight=args.pnp_locability_prior_weight,
        locability_target_prior_weight=args.pnp_locability_target_prior_weight,
    ).to(device)
    def trainable_params(params):
        return [param for param in params if param.requires_grad]

    params = []
    latent_params = trainable_params([model._latent])
    locability_params = trainable_params([model._locability_logit])
    hash_params = trainable_params(model.hash_field.parameters())
    decoder_params = trainable_params(
        list(model.fine_decoder.parameters())
        + list(model.coarse_decoder.parameters())
        + list(model.fusion_head.parameters())
    )
    sp_head_params = trainable_params(sp_head.parameters())
    if latent_params:
        params.append({"params": latent_params, "lr": args.lr_latent})
    if locability_params:
        params.append({"params": locability_params, "lr": args.lr_locability})
    if hash_params:
        params.append({"params": hash_params, "lr": args.lr_hash})
    if decoder_params:
        params.append({"params": decoder_params, "lr": args.lr_decoder})
    if sp_head_params:
        params.append({"params": sp_head_params, "lr": args.lr_head})
    if args.train_xyz:
        params.append({"params": [model._xyz], "lr": args.lr_xyz})
    if args.train_opacity:
        params.append({"params": [model._opacity], "lr": args.lr_opacity})
    if args.train_scaling:
        params.append({"params": [model._scaling], "lr": args.lr_scaling})
    if not params:
        raise ValueError("No trainable parameters remain after applying freeze flags")
    optimizer = torch.optim.AdamW(params, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    memory_bank = DescriptorMemoryBank(
        num_embeddings=model.num_gaussians,
        dim=256,
        momentum=args.memory_bank_momentum,
        device=device,
    )
    saliency_prior = None
    if args.splatloc_saliency_prior_weight > 0.0:
        pose_sample_count = min(len(dataset), 256)
        pose_ids = torch.linspace(0, len(dataset) - 1, steps=pose_sample_count).long().tolist()
        train_poses = torch.stack([dataset[int(i)]["pose_w2c"] for i in pose_ids], dim=0).to(device)
        saliency_prior = splatloc_saliency_prior(
            model.get_xyz().detach(),
            train_poses,
            renderer.K.float(),
            feature_height,
            feature_width,
        )

    config_snapshot = vars(args).copy()
    config_snapshot.update(
        {
            "output_dir": str(out_dir),
            "ply_path": str(ply_path),
            "cameras_json": str(cameras_json),
            "feature_height": feature_height,
            "feature_width": feature_width,
            "feature_intrinsics": intr,
        }
    )
    (out_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        model.train()
        sp_head.train()
        if args.train_xyz and epoch >= args.geometry_unfreeze_epoch:
            model._xyz.requires_grad_(True)
        if args.train_opacity and epoch >= args.geometry_unfreeze_epoch:
            model._opacity.requires_grad_(True)
        if args.train_scaling and epoch >= args.geometry_unfreeze_epoch:
            model._scaling.requires_grad_(True)

        grad_accum_steps = max(1, int(args.grad_accum_steps))
        effective_batches = min(len(loader), args.max_train_batches or len(loader))
        accum: dict[str, float] = {}
        pbar = tqdm(
            loader,
            desc=f"Cambridge hybrid E{epoch:03d}",
            total=effective_batches,
            dynamic_ncols=True,
        )
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            if step >= effective_batches:
                break
            rgb = batch["rgb"].to(device, non_blocking=True).float()
            pose = batch["pose_w2c"].to(device, non_blocking=True).float()

            with torch.no_grad():
                image_names = [str(name) for name in batch["image_name"]]
                teacher_rgb = rgb
                if args.teacher_feature_source == "original":
                    teacher_rgb = load_cambridge_rgb_no_resize_batch(
                        scene_root,
                        args.image_subdir,
                        image_names,
                        device,
                    )
                teacher_desc, teacher_det, _cache_hits = extract_superpoint_teacher_batch(
                    teacher,
                    teacher_rgb,
                    image_names,
                    cache=sp_cache,
                    expected_hw=(teacher_rgb.shape[-2] // 8, teacher_rgb.shape[-1] // 8),
                )
                teacher_desc, teacher_det = resize_teacher_outputs_to_feature_grid(
                    teacher_desc,
                    teacher_det,
                    feature_height,
                    feature_width,
                )
                query_descs, query_keypoints, query_mask = prepare_superpoint_queries(
                    teacher_desc,
                    teacher_det,
                    max_keypoints=args.localization_keypoints,
                    confidence_threshold=args.keypoint_threshold,
                    nms_radius=args.nms_radius,
                )
                maybe_write_superpoint_metadata(
                    sp_cache,
                    image_names,
                    teacher_desc,
                    teacher_det,
                    query_keypoints,
                    query_descs,
                )

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                gt_render = render_hybrid_superpoint(model, sp_head, renderer, pose)
                pred_desc = gt_render["descriptor"]
                pred_det = gt_render["detector"]
                desc_l2 = F.mse_loss(pred_desc.float(), teacher_desc.float())
                desc_cos = 1.0 - F.cosine_similarity(
                    pred_desc.float().flatten(2),
                    teacher_desc.float().flatten(2),
                    dim=1,
                ).mean()
                sp_recon = desc_l2 + 0.5 * desc_cos
                det_recon = F.kl_div(
                    F.log_softmax(pred_det.float(), dim=1),
                    F.softmax(teacher_det.float(), dim=1),
                    reduction="batchmean",
                )
                same_valid = (
                    torch.isfinite(gt_render["depth"])
                    & (gt_render["depth"] > 0.05)
                    & (gt_render["alpha"].float() > float(args.same_view_alpha_threshold))
                )
                same_match = geometric_keypoint_match_loss(
                    query_descs=query_descs,
                    query_keypoints_yx=query_keypoints,
                    query_mask=query_mask,
                    rendered_desc=pred_desc,
                    valid_pixels=same_valid,
                    locability_map=gt_render["locability"],
                    temperature=args.same_view_temperature,
                    target_sigma_px=args.same_view_target_sigma_px,
                    locability_weight=args.same_view_locability_weight,
                )

                loc_pose = perturb_pose_batch(pose, args.pose_noise_trans_m, args.pose_noise_rot_deg).detach()
                loc_render = render_hybrid_superpoint(model, sp_head, renderer, loc_pose)
                locability_target_prior = None
                if args.pnp_locability_target_prior_weight > 0.0:
                    locability_target_prior = locability_target_prior_from_render(
                        loc_render,
                        detector_weight=args.locability_target_detector_weight,
                        alpha_weight=args.locability_target_alpha_weight,
                        depth_weight=args.locability_target_depth_weight,
                        depth_window=args.locability_target_depth_window,
                    )
                pnp_out = pnp_loss_fn(
                    query_descs=query_descs,
                    query_keypoints_yx=query_keypoints,
                    query_mask=query_mask,
                    rendered_desc=loc_render["descriptor"],
                    depth_map=loc_render["depth"],
                    render_pose_w2c=loc_pose,
                    gt_pose_w2c=pose,
                    K=renderer.K.float(),
                    locability_map=loc_render["locability"],
                    locability_target_prior_map=locability_target_prior,
                )

                locability_sparse = (
                    loc_render["locability"].mean()
                    if loc_render["locability"] is not None
                    else pred_desc.new_tensor(0.0)
                )
                xview_out = {
                    "total": pred_desc.new_tensor(0.0),
                    "info_nce": pred_desc.new_tensor(0.0),
                    "hard_negative": pred_desc.new_tensor(0.0),
                    "valid_samples": pred_desc.new_tensor(0.0),
                }
                pair_batch = None
                pair_render = None
                if args.cross_view_weight > 0.0 and epoch >= args.xview_start_epoch:
                    pair_batch = select_pair_batch(
                        dataset,
                        batch["frame_idx"],
                        model,
                        renderer,
                        args.view_pair_min_overlap,
                        device,
                    )
                    pair_pose = pair_batch["pose_w2c"].to(device, non_blocking=True).float()
                    pair_render = render_hybrid_superpoint(model, sp_head, renderer, pair_pose)
                    positive_desc_b = None
                    if args.xview_positive_source == "teacher":
                        with torch.no_grad():
                            pair_rgb = pair_batch["rgb"].to(device, non_blocking=True).float()
                            pair_image_names = [str(name) for name in pair_batch["image_name"]]
                            pair_teacher_rgb = pair_rgb
                            if args.teacher_feature_source == "original":
                                pair_teacher_rgb = load_cambridge_rgb_no_resize_batch(
                                    scene_root,
                                    args.image_subdir,
                                    pair_image_names,
                                    device,
                                )
                            pair_teacher_desc, _pair_teacher_det, _pair_cache_hits = extract_superpoint_teacher_batch(
                                teacher,
                                pair_teacher_rgb,
                                pair_image_names,
                                cache=sp_cache,
                                expected_hw=(pair_teacher_rgb.shape[-2] // 8, pair_teacher_rgb.shape[-1] // 8),
                            )
                            pair_teacher_desc, _pair_teacher_det = resize_teacher_outputs_to_feature_grid(
                                pair_teacher_desc,
                                _pair_teacher_det,
                                feature_height,
                                feature_width,
                            )
                            positive_desc_b = pair_teacher_desc.detach()
                    active_hard_weight = (
                        args.hard_negative_weight
                        if epoch >= args.hard_negative_start_epoch
                        else 0.0
                    )
                    xview_out = cross_view_projective_contrastive_loss(
                        desc_a=gt_render["descriptor"],
                        depth_a=gt_render["depth"],
                        pose_a_w2c=pose,
                        desc_b=pair_render["descriptor"],
                        pose_b_w2c=pair_pose,
                        K=renderer.K.float(),
                        positive_desc_b=positive_desc_b,
                        depth_b=pair_render["depth"],
                        alpha_b=pair_render["alpha"],
                        valid_a=same_valid,
                        locability_a=gt_render["locability"],
                        max_samples=args.localization_keypoints,
                        temperature=args.pnp_temperature,
                        hard_negative_weight=active_hard_weight,
                        hard_negative_exclusion_radius=args.hard_negative_exclusion_radius,
                        depth_tolerance=args.xview_depth_tolerance,
                        depth_rel_tolerance=args.xview_depth_rel_tolerance,
                        alpha_threshold=args.same_view_alpha_threshold,
                    )
                external_out = {
                    "total": pred_desc.new_tensor(0.0),
                    "positive": pred_desc.new_tensor(0.0),
                    "hard_negative": pred_desc.new_tensor(0.0),
                    "valid_matches": pred_desc.new_tensor(0.0),
                }
                if (
                    args.external_match_supervision_weight > 0.0
                    and epoch >= args.external_match_start_epoch
                ):
                    if pair_batch is None:
                        pair_batch = select_pair_batch(
                            dataset,
                            batch["frame_idx"],
                            model,
                            renderer,
                            args.view_pair_min_overlap,
                            device,
                        )
                    if pair_render is None:
                        pair_pose = pair_batch["pose_w2c"].to(device, non_blocking=True).float()
                        pair_render = render_hybrid_superpoint(model, sp_head, renderer, pair_pose)
                    match_batch = load_external_match_training_batch(
                        external_match_cache,
                        image_names,
                        [str(name) for name in pair_batch["image_name"]],
                        args.image_height,
                        args.image_width,
                        feature_height,
                        feature_width,
                        args.localization_keypoints,
                        device,
                    )
                    if match_batch is not None:
                        external_out = external_match_supervision_loss(
                            gt_render["descriptor"],
                            pair_render["descriptor"],
                            kpts_a_yx=match_batch["kpts_a_yx"],
                            kpts_b_yx=match_batch["kpts_b_yx"],
                            scores=match_batch["scores"],
                            valid_mask=match_batch["valid"],
                            negative_kpts_b_yx=match_batch["negative_kpts_b_yx"],
                            hard_negative_weight=args.detector_free_hard_negative_weight,
                            temperature=args.pnp_temperature,
                        )

                center_ids = None
                center_desc = None
                memory_loss = pred_desc.new_tensor(0.0)
                need_center_desc = (
                    args.hard_negative_weight > 0.0
                    or args.locability_ambiguity_weight > 0.0
                ) and epoch >= args.hard_negative_start_epoch
                if need_center_desc:
                    loc_flat = model.get_locability().detach().squeeze(-1)
                    keep = min(max(1, int(args.memory_bank_size)), loc_flat.numel())
                    center_ids = torch.topk(loc_flat, k=keep).indices
                    center_desc = decode_gaussian_center_descriptors(model, sp_head, center_ids)
                    memory_loss = memory_bank_contrastive_loss(
                        center_desc,
                        center_ids,
                        memory_bank,
                        temperature=args.pnp_temperature,
                    )

                loc_budget = (
                    locability_budget_loss(model.get_locability(), args.landmark_budget)
                    if args.locability_budget_weight > 0.0
                    else pred_desc.new_tensor(0.0)
                )
                saliency_loss = pred_desc.new_tensor(0.0)
                if saliency_prior is not None and args.splatloc_saliency_prior_weight > 0.0:
                    loc_flat = model.get_locability().squeeze(-1).clamp(1e-4, 1.0 - 1e-4)
                    with torch.cuda.amp.autocast(enabled=False):
                        saliency_loss = F.binary_cross_entropy(
                            loc_flat.float(),
                            saliency_prior.to(loc_flat.device).float(),
                        )
                ambiguity_loss = pred_desc.new_tensor(0.0)
                if (
                    args.locability_ambiguity_weight > 0.0
                    and center_desc is not None
                    and center_ids is not None
                ):
                    ambiguity_loss = descriptor_ambiguity_loss(
                        center_desc,
                        model.get_locability()[center_ids].squeeze(-1),
                    )
                isotropy_loss = (
                    key_gaussian_isotropy_loss(model.get_scaling(), model.get_locability().squeeze(-1))
                    if args.key_gaussian_isotropy_weight > 0.0
                    else pred_desc.new_tensor(0.0)
                )
                geometry_reg = pred_desc.new_tensor(0.0)
                if args.train_xyz:
                    geometry_reg = geometry_reg + F.mse_loss(model.get_xyz(), xyz0)
                if args.train_opacity:
                    geometry_reg = geometry_reg + F.mse_loss(model.get_opacity_logits(), opacity0)
                if args.train_scaling:
                    geometry_reg = geometry_reg + F.mse_loss(model._scaling, scaling0)

                active_pnp_weight = scheduled_loss_weight(
                    epoch,
                    args.pnp_weight,
                    start_epoch=args.pnp_start_epoch,
                    warmup_epochs=args.pnp_warmup_epochs,
                )
                loss = (
                    active_pnp_weight * pnp_out["total"]
                    + args.same_view_match_weight * same_match["total"]
                    + args.sp_recon_weight * sp_recon
                    + args.detector_recon_weight * det_recon
                    + args.locability_sparse_weight * locability_sparse
                    + args.geometry_reg_weight * geometry_reg
                    + args.cross_view_weight * xview_out["total"]
                    + args.external_match_supervision_weight * external_out["total"]
                    + args.hard_negative_weight * memory_loss
                    + args.locability_budget_weight * loc_budget
                    + args.splatloc_saliency_prior_weight * saliency_loss
                    + args.locability_ambiguity_weight * ambiguity_loss
                    + args.key_gaussian_isotropy_weight * isotropy_loss
                )

            scaler.scale(loss / float(grad_accum_steps)).backward()
            should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) >= effective_batches)
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad],
                    max_norm=10.0,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if center_desc is not None and center_ids is not None:
                memory_bank.update(center_ids, center_desc.detach())

            metrics = {
                "loss": float(loss.detach()),
                "pnp_weight": float(active_pnp_weight),
                "pnp": float(pnp_out["total"].detach()),
                "pose": float(pnp_out["pose"].detach()),
                "match": float(pnp_out["match"].detach()),
                "quality": float(pnp_out["quality"].detach()),
                "pnp_valid_queries": float(pnp_out["valid_queries"].detach()),
                "same_match": float(same_match["match"].detach()),
                "same_top1_1px": float(same_match["top1_1px"].detach()),
                "same_valid_queries": float(same_match["valid_queries"].detach()),
                "same_locability": float(same_match["locability"].detach()),
                "sp_recon": float(sp_recon.detach()),
                "det_recon": float(det_recon.detach()),
                "locability": float(locability_sparse.detach()),
                "geometry_reg": float(geometry_reg.detach()),
                "xview": float(xview_out["total"].detach()),
                "xview_samples": float(xview_out["valid_samples"].detach()),
                "external_match": float(external_out["total"].detach()),
                "external_match_positive": float(external_out["positive"].detach()),
                "external_match_hard_negative": float(external_out["hard_negative"].detach()),
                "external_match_samples": float(external_out["valid_matches"].detach()),
                "memory": float(memory_loss.detach()),
                "loc_budget": float(loc_budget.detach()),
                "saliency": float(saliency_loss.detach()),
                "ambiguity": float(ambiguity_loss.detach()),
                "isotropy": float(isotropy_loss.detach()),
            }
            for key, value in metrics.items():
                accum[key] = accum.get(key, 0.0) + value
            pbar.set_postfix(loss=f"{metrics['loss']:.3f}", pose=f"{metrics['pose']:.3f}")

        denom = max(1, effective_batches)
        epoch_metrics = {key: value / denom for key, value in accum.items()}
        print(f"[{time.strftime('%H:%M:%S')}] epoch {epoch}: {epoch_metrics}")
        with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, **epoch_metrics}) + "\n")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "args": config_snapshot,
                "model_state_dict": model.state_dict(),
                "sp_head_state_dict": sp_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(ckpt, out_dir / f"checkpoint_epoch_{epoch:04d}.pth")
            torch.save(ckpt, out_dir / "latest.pth")


if __name__ == "__main__":
    main()
