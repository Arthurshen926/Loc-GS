#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Subset
from tqdm import tqdm

from loc_gs.data.cambridge_dataset import CambridgeHybridDataset
from loc_gs.data.superpoint_cache import SuperPointTeacherCache
from loc_gs.localization.descriptor_blend import gated_residual_descriptor_blend
from loc_gs.localization.hybrid_localizer import (
    extract_keypoints_from_detector_logits,
    flatten_rendered_landmarks,
    match_descriptors_topk,
    pose_error_cm_deg,
    refine_rendered_positions_softargmax,
    sample_descriptors_bilinear,
    solve_pnp_ransac,
)
from loc_gs.localization.lightglue_matcher import (
    lightglue_feature_name,
    match_lightglue_descriptors,
)
from loc_gs.localization.dim_image_matcher import match_loftr_images
from loc_gs.localization.matcher_registry import (
    DENSE_MATCHERS,
    DIM_PIPELINES,
    SPARSE_MATCHERS,
    resolve_sparse_dense_matchers,
)
from loc_gs.localization.pose_metrics import pose_error_summary
from loc_gs.localization.rendered_keypoints import select_rendered_keypoints
from loc_gs.localization.stdloc_parity import (
    apply_match_prior,
    coarse_to_fine_dense_matches,
    match_correlation_matrix,
)
from loc_gs.localization.stdloc_detector import (
    StdlocKeypointDetector,
    extract_stdloc_detector_keypoints,
)
from loc_gs.losses.landmark_selection import (
    depth_consistency_score,
    descriptor_landmark_distinctiveness,
    descriptor_local_distinctiveness,
    gaussian_geometry_score,
    geometric_mean_score,
    keypoint_consensus_score,
    normalize_score01,
    spatially_balanced_topk,
    splatloc_saliency_prior,
    superpoint_detector_saliency,
)
from loc_gs.losses.localization_loss import (
    project_world_to_image_yx,
    projection_jacobian_observability,
    unproject_dense_depth_to_world,
)
from loc_gs.models.hybrid_gaussian import HybridFeatureGaussian, SuperPointOutputHead
from loc_gs.scripts.extract_superpoint_features import SuperPointNet
from loc_gs.scripts.train_cambridge_hybrid import (
    decode_gaussian_center_descriptors,
    extract_superpoint_teacher_batch,
    make_feature_renderer_intrinsics,
    maybe_write_superpoint_metadata,
    normalize_position_map,
    render_hybrid_superpoint,
    resize_teacher_outputs_to_feature_grid,
    superpoint_gray,
)


def unproject_positions_yx(
    positions_yx: torch.Tensor,
    depth_map: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
    pixel_center_offset: float = 0.0,
) -> torch.Tensor:
    if positions_yx.numel() == 0:
        return positions_yx.new_zeros((0, 3))
    H, W = depth_map.shape
    y = positions_yx[:, 0].clamp(0, H - 1)
    x = positions_yx[:, 1].clamp(0, W - 1)
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = (x0 + 1).clamp(max=W - 1)
    y1 = (y0 + 1).clamp(max=H - 1)
    wx = x - x0.float()
    wy = y - y0.float()
    d00 = depth_map[y0, x0]
    d10 = depth_map[y0, x1]
    d01 = depth_map[y1, x0]
    d11 = depth_map[y1, x1]
    depth = (1.0 - wy) * ((1.0 - wx) * d00 + wx * d10) + wy * ((1.0 - wx) * d01 + wx * d11)
    if pixel_center_offset:
        offset = float(pixel_center_offset)
        x = x + offset
        y = y + offset
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (x - cx) / fx.clamp_min(1e-8) * depth
    y_cam = (y - cy) / fy.clamp_min(1e-8) * depth
    pts_cam = torch.stack([x_cam, y_cam, depth, torch.ones_like(depth)], dim=-1)
    c2w = torch.linalg.inv(pose_w2c)
    return (c2w @ pts_cam.T).T[:, :3]


def upsample_feature_map(feature_map: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """STDLoc-style dense localization uses a full-resolution fine feature map."""
    return F.interpolate(
        feature_map.unsqueeze(0).float(),
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    )[0]


def load_cambridge_rgb_no_resize(
    scene_root: Path,
    image_subdir: str,
    image_name: str,
    device: torch.device,
) -> torch.Tensor:
    candidates = []
    if image_subdir:
        candidates.append(scene_root / image_subdir.strip("/") / image_name)
    candidates.append(scene_root / image_name)
    path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)


def prepare_query_teacher_maps(
    teacher_desc: torch.Tensor,
    teacher_det: torch.Tensor,
    feature_height: int,
    feature_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw descriptors plus descriptor/detector maps in feature-grid coordinates."""
    teacher_desc_raw = F.normalize(teacher_desc[0].float(), dim=0)
    teacher_det_raw = teacher_det[0].float()
    if teacher_desc_raw.shape[-2:] == (int(feature_height), int(feature_width)):
        return teacher_desc_raw, teacher_desc_raw, teacher_det_raw
    teacher_desc_grid, teacher_det_grid = resize_teacher_outputs_to_feature_grid(
        teacher_desc_raw.unsqueeze(0),
        teacher_det_raw.unsqueeze(0),
        int(feature_height),
        int(feature_width),
    )
    return teacher_desc_raw, teacher_desc_grid[0], teacher_det_grid[0]


def resolve_matchers(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve legacy --matcher into independently configurable sparse/dense matchers."""
    return resolve_sparse_dense_matchers(
        getattr(args, "matcher", "topk"),
        getattr(args, "sparse_matcher", ""),
        getattr(args, "dense_matcher", ""),
    )


@torch.no_grad()
def decode_landmark_descriptors(
    model: HybridFeatureGaussian,
    sp_head: SuperPointOutputHead,
    max_landmarks: int,
    chunk_size: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xyz = model.get_xyz()
    locability = model.get_locability().squeeze(-1)
    keep = min(int(max_landmarks), xyz.shape[0])
    if keep < xyz.shape[0]:
        _, ids = torch.topk(locability, k=keep)
    else:
        ids = torch.arange(xyz.shape[0], device=xyz.device)
    selected_xyz = xyz[ids]
    selected_latent = model.get_latent()[ids]
    selected_loc = locability[ids]

    desc_chunks = []
    for start in range(0, keep, chunk_size):
        end = min(start + chunk_size, keep)
        latent_map = selected_latent[start:end].T.contiguous().view(1, -1, end - start, 1)
        pos_map = selected_xyz[start:end].T.contiguous().view(1, 3, end - start, 1)
        pos_map = normalize_position_map(pos_map, model.get_xyz())
        fused = model.decode_screen_space(latent_map, pos_map)
        desc = sp_head(fused)["descriptor"].squeeze(0).squeeze(-1).T
        desc_chunks.append(F.normalize(desc.float(), dim=-1))
    return selected_xyz, torch.cat(desc_chunks, dim=0), selected_loc


def _load_pickle_tensor(path: Path) -> torch.Tensor:
    with path.open("rb") as f:
        value = pickle.load(f)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        for key in ("score_avg", "sampled_scores", "scores", "sampled_idx", "indices"):
            tensor_value = value.get(key)
            if tensor_value is None:
                continue
            if isinstance(tensor_value, torch.Tensor):
                return tensor_value.detach().cpu()
            return torch.as_tensor(tensor_value)
        raise TypeError(f"Could not find a tensor-like entry in pickle dict: {path}")
    return torch.as_tensor(value)


def _score_stats(prefix: str, score: torch.Tensor, out: dict[str, float]) -> None:
    if score.numel() == 0:
        return
    value = score.detach().float()
    finite = value[torch.isfinite(value)]
    if finite.numel() == 0:
        return
    out[f"{prefix}_mean"] = float(finite.mean().cpu())
    out[f"{prefix}_median"] = float(finite.median().cpu())
    out[f"{prefix}_min"] = float(finite.min().cpu())
    out[f"{prefix}_max"] = float(finite.max().cpu())


def _score_ref_poses(
    dataset: CambridgeHybridDataset,
    device: torch.device,
    max_views: int,
) -> torch.Tensor | None:
    ids = _score_ref_indices(len(dataset), max_views)
    if not ids:
        return None
    return torch.stack([dataset[int(i)]["pose_w2c"] for i in ids], dim=0).to(device=device, dtype=torch.float32)


def _score_ref_indices(length: int, max_views: int) -> list[int]:
    count = min(int(length), max(0, int(max_views)))
    if count <= 0:
        return []
    if count == int(length):
        return list(range(count))
    return torch.linspace(0, int(length) - 1, steps=count).long().tolist()


@torch.no_grad()
def _score_ref_keypoint_maps(
    dataset: CambridgeHybridDataset,
    teacher: SuperPointNet,
    device: torch.device,
    max_views: int,
    height: int,
    width: int,
    max_keypoints: int,
    threshold: float,
    nms_radius: int,
    include_descriptors: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    ids = _score_ref_indices(len(dataset), max_views)
    if not ids:
        return None, None
    maps = []
    desc_maps = []
    for idx in ids:
        rgb = dataset[int(idx)]["rgb"].unsqueeze(0).to(device=device, dtype=torch.float32)
        teacher_desc, teacher_det = teacher(superpoint_gray(rgb))
        teacher_desc_grid, teacher_det_grid = resize_teacher_outputs_to_feature_grid(
            teacher_desc,
            teacher_det,
            int(height),
            int(width),
        )
        keypoints, _scores = extract_keypoints_from_detector_logits(
            teacher_det_grid[0],
            max_keypoints=int(max_keypoints),
            confidence_threshold=float(threshold),
            nms_radius=int(nms_radius),
        )
        heat = torch.zeros(int(height), int(width), device=device, dtype=torch.float32)
        if keypoints.numel() > 0:
            y = keypoints[:, 0].round().long().clamp(0, int(height) - 1)
            x = keypoints[:, 1].round().long().clamp(0, int(width) - 1)
            heat[y, x] = 1.0
        maps.append(heat)
        if include_descriptors:
            desc_maps.append(F.normalize(teacher_desc_grid[0].float(), p=2, dim=0))
    return torch.stack(maps, dim=0), (torch.stack(desc_maps, dim=0) if include_descriptors else None)


@torch.no_grad()
def _score_ref_descriptor_maps(
    dataset: CambridgeHybridDataset,
    teacher: SuperPointNet,
    device: torch.device,
    max_views: int,
    height: int,
    width: int,
) -> torch.Tensor | None:
    ids = _score_ref_indices(len(dataset), max_views)
    if not ids:
        return None
    desc_maps = []
    for idx in ids:
        rgb = dataset[int(idx)]["rgb"].unsqueeze(0).to(device=device, dtype=torch.float32)
        teacher_desc, teacher_det = teacher(superpoint_gray(rgb))
        teacher_desc_grid, _teacher_det_grid = resize_teacher_outputs_to_feature_grid(
            teacher_desc,
            teacher_det,
            int(height),
            int(width),
        )
        desc_maps.append(F.normalize(teacher_desc_grid[0].float(), p=2, dim=0))
    return torch.stack(desc_maps, dim=0)


def _axis_angle_rotation_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = F.normalize(axis.float(), p=2, dim=0, eps=1e-8)
    x, y, z = axis.unbind(dim=0)
    c = torch.cos(angle.float())
    s = torch.sin(angle.float())
    one_c = 1.0 - c
    return torch.stack(
        [
            torch.stack([c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s]),
            torch.stack([y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s]),
            torch.stack([z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c]),
        ],
        dim=0,
    )


def perturb_reference_poses_camera_frame(
    poses_w2c: torch.Tensor,
    translation_m: float,
    rotation_deg: float,
) -> torch.Tensor:
    """Create deterministic camera-frame pose perturbations for rendered auxiliary views."""
    if poses_w2c.numel() == 0:
        return poses_w2c
    trans_mag = float(translation_m)
    rot_mag = math.radians(float(rotation_deg))
    out = []
    device = poses_w2c.device
    dtype = poses_w2c.dtype
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for idx, pose in enumerate(poses_w2c):
        phase = float(idx + 1) * golden
        delta = torch.eye(4, device=device, dtype=dtype)
        if rot_mag > 0.0:
            axis = torch.tensor(
                [math.sin(phase * 0.7), math.cos(phase * 1.3), 0.5 * math.sin(phase * 1.7)],
                device=device,
                dtype=torch.float32,
            )
            angle = torch.tensor(rot_mag * math.sin(phase), device=device, dtype=torch.float32)
            delta[:3, :3] = _axis_angle_rotation_matrix(axis, angle).to(device=device, dtype=dtype)
        if trans_mag > 0.0:
            delta[:3, 3] = torch.tensor(
                [
                    trans_mag * math.cos(phase),
                    trans_mag * math.sin(phase),
                    0.35 * trans_mag * math.sin(phase * 0.5),
                ],
                device=device,
                dtype=dtype,
            )
        out.append(delta @ pose)
    return torch.stack(out, dim=0)


@torch.no_grad()
def _rendered_ref_descriptor_maps(
    model: HybridFeatureGaussian,
    full_renderer,
    poses_w2c: torch.Tensor,
    teacher: SuperPointNet,
    device: torch.device,
    height: int,
    width: int,
) -> torch.Tensor | None:
    if full_renderer is None or poses_w2c is None or poses_w2c.numel() == 0:
        return None
    desc_maps = []
    for pose in poses_w2c:
        rendered = full_renderer.render_rgb(model, pose.to(device=device, dtype=torch.float32))
        rgb = rendered["rgb"].unsqueeze(0).to(device=device, dtype=torch.float32)
        teacher_desc, teacher_det = teacher(superpoint_gray(rgb))
        teacher_desc_grid, _teacher_det_grid = resize_teacher_outputs_to_feature_grid(
            teacher_desc,
            teacher_det,
            int(height),
            int(width),
        )
        desc_maps.append(F.normalize(teacher_desc_grid[0].float(), p=2, dim=0))
    return torch.stack(desc_maps, dim=0) if desc_maps else None


@torch.no_grad()
def fuse_projected_teacher_descriptors(
    points: torch.Tensor,
    poses_w2c: torch.Tensor | None,
    K: torch.Tensor,
    descriptor_maps: torch.Tensor | None,
    height: int,
    width: int,
    chunk_size: int = 2048,
    geometry_power: float = 0.0,
    centrality_power: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse multi-view teacher descriptors by projecting 3D landmarks into reference views.

    Geometry and centrality weights are opt-in so the historical uniform-fusion
    behavior remains the default.
    """
    if points.ndim != 2 or points.shape[-1] < 3:
        raise ValueError("points must have shape [N, 3+]")
    if descriptor_maps is None or poses_w2c is None or descriptor_maps.numel() == 0 or poses_w2c.numel() == 0:
        dim = 0 if descriptor_maps is None or descriptor_maps.ndim != 4 else int(descriptor_maps.shape[1])
        return points.new_zeros((points.shape[0], dim)), points.new_zeros((points.shape[0],))
    if descriptor_maps.ndim != 4:
        raise ValueError("descriptor_maps must have shape [V, C, H, W]")
    view_count = min(int(poses_w2c.shape[0]), int(descriptor_maps.shape[0]))
    if view_count <= 0:
        return points.new_zeros((points.shape[0], int(descriptor_maps.shape[1]))), points.new_zeros((points.shape[0],))

    H = int(height)
    W = int(width)
    desc_bchw = descriptor_maps[:view_count].to(device=points.device, dtype=torch.float32)
    if desc_bchw.shape[-2:] != (H, W):
        desc_bchw = F.interpolate(desc_bchw, size=(H, W), mode="bilinear", align_corners=False)
    desc_bchw = F.normalize(desc_bchw, p=2, dim=1, eps=1e-8)
    poses = poses_w2c[:view_count].to(device=points.device, dtype=torch.float32)
    K = K.to(device=points.device, dtype=torch.float32)

    fused_chunks = []
    count_chunks = []
    chunk = max(1, int(chunk_size))
    for start in range(0, points.shape[0], chunk):
        end = min(start + chunk, points.shape[0])
        pts = points[start:end, :3].to(device=points.device, dtype=torch.float32)
        pts_v = pts.unsqueeze(0).expand(view_count, -1, -1)
        proj_yx, valid_z = project_world_to_image_yx(pts_v, poses, K)
        in_frame = (
            valid_z
            & (proj_yx[..., 0] >= 0.0)
            & (proj_yx[..., 0] <= float(H - 1))
            & (proj_yx[..., 1] >= 0.0)
            & (proj_yx[..., 1] <= float(W - 1))
        )
        grid_x = proj_yx[..., 1] / float(max(W - 1, 1)) * 2.0 - 1.0 if W > 1 else torch.zeros_like(proj_yx[..., 1])
        grid_y = proj_yx[..., 0] / float(max(H - 1, 1)) * 2.0 - 1.0 if H > 1 else torch.zeros_like(proj_yx[..., 0])
        grid = torch.stack([grid_x, grid_y], dim=-1).view(view_count, -1, 1, 2)
        sampled = F.grid_sample(
            desc_bchw,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[..., 0]
        sampled = F.normalize(sampled, p=2, dim=1, eps=1e-8)
        weights = in_frame.float()
        if float(centrality_power) > 0.0:
            cy = float(max(H - 1, 1)) * 0.5
            cx = float(max(W - 1, 1)) * 0.5
            dy = (proj_yx[..., 0] - cy) / max(cy, 1.0)
            dx = (proj_yx[..., 1] - cx) / max(cx, 1.0)
            center_score = torch.exp(-0.5 * float(centrality_power) * (dy.square() + dx.square()))
            weights = weights * center_score.clamp_min(0.0)
        if float(geometry_power) > 0.0:
            observability = projection_jacobian_observability(
                pts_v,
                poses,
                K,
                min_depth=0.05,
            ).clamp_min(1e-8)
            obs_center = torch.where(in_frame, observability, torch.zeros_like(observability)).sum(dim=0)
            obs_count = in_frame.float().sum(dim=0).clamp_min(1.0)
            obs_center = (obs_center / obs_count).clamp_min(1e-8)
            geometry_score = (observability / obs_center[None, :]).clamp(0.05, 20.0)
            weights = weights * geometry_score.pow(float(geometry_power))
        desc_sum = (sampled * weights[:, None, :]).sum(dim=0).T
        counts = in_frame.float().sum(dim=0)
        fused = F.normalize(desc_sum, p=2, dim=-1, eps=1e-8)
        fused = torch.where(counts[:, None] > 0.0, fused, torch.zeros_like(fused))
        fused_chunks.append(fused)
        count_chunks.append(counts)
    return torch.cat(fused_chunks, dim=0), torch.cat(count_chunks, dim=0)


@torch.no_grad()
def _splatloc_visibility_score(
    points: torch.Tensor,
    poses_w2c: torch.Tensor | None,
    K: torch.Tensor,
    height: int,
    width: int,
    chunk_size: int = 65536,
) -> torch.Tensor:
    if points.numel() == 0:
        return points.new_empty((0,))
    if poses_w2c is None or poses_w2c.numel() == 0:
        return torch.ones(points.shape[0], device=points.device, dtype=torch.float32)
    chunks = []
    for start in range(0, points.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), points.shape[0])
        chunks.append(
            splatloc_saliency_prior(
                points[start:end].detach(),
                poses_w2c,
                K,
                height=int(height),
                width=int(width),
            )
        )
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _splatloc_observability_score(
    points: torch.Tensor,
    poses_w2c: torch.Tensor | None,
    K: torch.Tensor,
    height: int,
    width: int,
    chunk_size: int = 65536,
) -> torch.Tensor:
    if points.numel() == 0:
        return points.new_empty((0,))
    if poses_w2c is None or poses_w2c.numel() == 0:
        return torch.ones(points.shape[0], device=points.device, dtype=torch.float32)
    K = K.to(device=points.device, dtype=torch.float32)
    poses = poses_w2c.to(device=points.device, dtype=torch.float32)
    chunks = []
    for start in range(0, points.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), points.shape[0])
        pts = points[start:end].detach().float()
        obs_sum = pts.new_zeros(pts.shape[0])
        visible_count = pts.new_zeros(pts.shape[0])
        for pose in poses:
            proj, valid_z = project_world_to_image_yx(pts.unsqueeze(0), pose.unsqueeze(0), K)
            proj = proj[0]
            visible = (
                valid_z[0]
                & (proj[:, 0] >= 0)
                & (proj[:, 0] <= height - 1)
                & (proj[:, 1] >= 0)
                & (proj[:, 1] <= width - 1)
            )
            obs = projection_jacobian_observability(
                pts.unsqueeze(0),
                pose.unsqueeze(0),
                K,
                min_depth=0.05,
            )[0]
            obs_sum += torch.where(visible, obs, torch.zeros_like(obs))
            visible_count += visible.float()
        chunks.append(obs_sum / visible_count.clamp_min(1.0))
    return normalize_score01(torch.cat(chunks, dim=0))


def _score_weights(args: argparse.Namespace, rendered: bool) -> dict[str, float]:
    if getattr(args, "landmark_score_mode", "legacy") not in {"matchability", "matchability_prior"}:
        return {}
    weights = {
        "locability": float(args.landmark_score_locability_weight),
        "visibility": float(args.landmark_score_visibility_weight),
        "geometry": float(args.landmark_score_geometry_weight),
        "ambiguity": float(getattr(args, "landmark_score_ambiguity_weight", 0.0)),
        "keypoint_consensus": float(getattr(args, "landmark_score_keypoint_consensus_weight", 0.0)),
    }
    if rendered:
        weights.update(
            {
                "detector": float(args.landmark_score_detector_weight),
                "alpha": float(args.landmark_score_alpha_weight),
                "depth": float(args.landmark_score_depth_weight),
                "observability": float(args.landmark_score_observability_weight),
                "distinctiveness": float(args.landmark_score_distinctiveness_weight),
            }
        )
    else:
        weights["detector"] = float(args.landmark_score_detector_weight)
        weights["observability"] = float(args.landmark_score_observability_weight)
    return weights


def effective_eval_config(args: argparse.Namespace) -> dict[str, object]:
    """Return the effective matcher/PnP config written to evaluation summaries."""
    sparse_matcher, dense_matcher = resolve_matchers(args)
    sparse_reprojection_error = (
        args.sparse_reprojection_error
        if args.sparse_reprojection_error is not None
        else args.reprojection_error
    )
    dense_reprojection_error = (
        args.dense_reprojection_error
        if args.dense_reprojection_error is not None
        else args.reprojection_error
    )
    sparse_margin = (
        args.sparse_match_second_best_margin
        if args.sparse_match_second_best_margin is not None
        else args.match_second_best_margin
    )
    dense_margin = (
        args.dense_match_second_best_margin
        if args.dense_match_second_best_margin is not None
        else args.match_second_best_margin
    )
    return {
        "matcher": args.matcher,
        "sparse_matcher": sparse_matcher,
        "dense_matcher": dense_matcher,
        "solver": args.solver,
        "eval_split": getattr(args, "eval_split", "test"),
        "poselib_refine": bool(args.poselib_refine),
        "query_keypoints": int(args.query_keypoints),
        "keypoint_threshold": float(args.keypoint_threshold),
        "nms_radius": int(args.nms_radius),
        "sparse_match_threshold": float(args.sparse_match_threshold),
        "dense_match_threshold": float(args.dense_match_threshold),
        "match_second_best_margin": float(args.match_second_best_margin),
        "sparse_match_second_best_margin": float(sparse_margin),
        "dense_match_second_best_margin": float(dense_margin),
        "locability_prior_weight": float(args.locability_prior_weight),
        "sparse_dual_softmax": bool(args.sparse_dual_softmax),
        "sparse_dual_softmax_temp": float(args.sparse_dual_softmax_temp),
        "dense_dual_softmax_temp": float(args.dense_dual_softmax_temp),
        "fine_dual_softmax_temp": float(args.fine_dual_softmax_temp),
        "mnn": bool(args.mnn),
        "dense_iters": int(args.dense_iters),
        "dense_full_render": bool(args.dense_full_render),
        "loftr_pretrained": str(getattr(args, "loftr_pretrained", "outdoor")),
        "loftr_image_scale": float(getattr(args, "loftr_image_scale", 1.0)),
        "loftr_min_confidence": float(getattr(args, "loftr_min_confidence", 0.0)),
        "loftr_max_matches": int(getattr(args, "loftr_max_matches", 4096)),
        "dense_sparse_consistency_gate": bool(getattr(args, "dense_sparse_consistency_gate", False)),
        "dense_sparse_consistency_max_median_ratio": float(
            getattr(args, "dense_sparse_consistency_max_median_ratio", 1.5)
        ),
        "dense_sparse_consistency_max_median_increase": float(
            getattr(args, "dense_sparse_consistency_max_median_increase", 2.0)
        ),
        "dense_sparse_consistency_min_inlier_ratio_factor": float(
            getattr(args, "dense_sparse_consistency_min_inlier_ratio_factor", 0.75)
        ),
        "dense_pose_delta_gate": bool(getattr(args, "dense_pose_delta_gate", False)),
        "dense_pose_delta_max_trans_cm": float(getattr(args, "dense_pose_delta_max_trans_cm", 100.0)),
        "dense_pose_delta_max_rot_deg": float(getattr(args, "dense_pose_delta_max_rot_deg", 5.0)),
        "subpixel_refine": bool(args.subpixel_refine),
        "subpixel_temperature": float(args.subpixel_temperature),
        "topk_refine_window": int(args.topk_refine_window),
        "render_pixel_center_offset": float(args.render_pixel_center_offset),
        "dense_query_pixel_center_offset": float(args.dense_query_pixel_center_offset),
        "reprojection_error": float(args.reprojection_error),
        "sparse_reprojection_error": float(sparse_reprojection_error),
        "dense_reprojection_error": float(dense_reprojection_error),
        "refine_reprojection_error": float(args.refine_reprojection_error),
        "pnp_confidence": float(args.pnp_confidence),
        "pnp_iterations": int(args.pnp_iterations),
        "pnp_min_iterations": int(args.pnp_min_iterations),
        "sparse_pnp_iterations": None
        if args.sparse_pnp_iterations is None
        else int(args.sparse_pnp_iterations),
        "sparse_pnp_min_iterations": None
        if args.sparse_pnp_min_iterations is None
        else int(args.sparse_pnp_min_iterations),
        "dense_pnp_iterations": None
        if args.dense_pnp_iterations is None
        else int(args.dense_pnp_iterations),
        "dense_pnp_min_iterations": None
        if args.dense_pnp_min_iterations is None
        else int(args.dense_pnp_min_iterations),
        "pnp_prefilter": args.pnp_prefilter,
        "sparse_pnp_max_matches": int(args.sparse_pnp_max_matches),
        "dense_pnp_max_matches": int(args.dense_pnp_max_matches),
        "pnp_prefilter_image_grid_size": int(args.pnp_prefilter_image_grid_size),
        "pnp_prefilter_xyz_grid_size": int(args.pnp_prefilter_xyz_grid_size),
        "calibrated_matchability_path": str(getattr(args, "calibrated_matchability_path", "")),
        "landmark_score_calibrated_matchability_weight": float(
            getattr(args, "landmark_score_calibrated_matchability_weight", 0.0)
        ),
        "landmark_teacher_fusion_weight": float(getattr(args, "landmark_teacher_fusion_weight", 0.0)),
        "landmark_teacher_fusion_views": int(getattr(args, "landmark_teacher_fusion_views", 0)),
        "landmark_teacher_fusion_chunk_size": int(getattr(args, "landmark_teacher_fusion_chunk_size", 2048)),
        "landmark_teacher_fusion_geometry_power": float(
            getattr(args, "landmark_teacher_fusion_geometry_power", 0.0)
        ),
        "landmark_teacher_fusion_centrality_power": float(
            getattr(args, "landmark_teacher_fusion_centrality_power", 0.0)
        ),
        "landmark_teacher_fusion_rendered_views": int(
            getattr(args, "landmark_teacher_fusion_rendered_views", 0)
        ),
        "landmark_teacher_fusion_rendered_trans_noise_cm": float(
            getattr(args, "landmark_teacher_fusion_rendered_trans_noise_cm", 0.0)
        ),
        "landmark_teacher_fusion_rendered_rot_noise_deg": float(
            getattr(args, "landmark_teacher_fusion_rendered_rot_noise_deg", 0.0)
        ),
        "match_calibrated_prior_weight": float(getattr(args, "match_calibrated_prior_weight", 0.0)),
        "match_filter_mode": str(getattr(args, "match_filter_mode", "")),
        "match_filter_calibrated_score_weight": float(getattr(args, "match_filter_calibrated_score_weight", 0.0)),
        "match_filter_margin_weight": float(getattr(args, "match_filter_margin_weight", 0.0)),
        "match_filter_top_m": int(getattr(args, "match_filter_top_m", 0)),
        "match_filter_image_grid_size": int(getattr(args, "match_filter_image_grid_size", 8)),
        "match_filter_xyz_grid_size": int(getattr(args, "match_filter_xyz_grid_size", 8)),
        "sparse_match_filter_mode": str(getattr(args, "sparse_match_filter_mode", "")),
        "dense_match_filter_mode": str(getattr(args, "dense_match_filter_mode", "")),
        "sparse_match_filter_top_m": int(getattr(args, "sparse_match_filter_top_m", 0)),
        "dense_match_filter_top_m": int(getattr(args, "dense_match_filter_top_m", 0)),
        "match_filter_max_per_image_cell": int(getattr(args, "match_filter_max_per_image_cell", 8)),
        "match_filter_max_per_xyz_cell": int(getattr(args, "match_filter_max_per_xyz_cell", 8)),
        "match_filter_min_matches": int(getattr(args, "match_filter_min_matches", 0)),
        "pnp_hypotheses": int(getattr(args, "pnp_hypotheses", 1)),
        "pnp_cluster_mode": str(getattr(args, "pnp_cluster_mode", "none")),
        "pnp_cluster_grid_size": int(getattr(args, "pnp_cluster_grid_size", 4)),
        "pnp_dense_verify_topk": int(getattr(args, "pnp_dense_verify_topk", 1)),
        "pnp_hypothesis_min_score_gain": float(getattr(args, "pnp_hypothesis_min_score_gain", 0.0)),
        "hybrid_residual_alpha_max": float(getattr(args, "hybrid_residual_alpha_max", 0.05)),
    }


def project_world_points_yx_np(
    points_world: np.ndarray,
    pose_w2c: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64)
    pose = np.asarray(pose_w2c, dtype=np.float64)
    intr = np.asarray(K, dtype=np.float64)
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    points_cam = points @ pose[:3, :3].T + pose[:3, 3][None, :]
    z = points_cam[:, 2]
    valid_z = np.abs(z) > 1e-9
    yx = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    x = intr[0, 0] * points_cam[valid_z, 0] / z[valid_z] + intr[0, 2]
    y = intr[1, 1] * points_cam[valid_z, 1] / z[valid_z] + intr[1, 2]
    yx[valid_z, 0] = y
    yx[valid_z, 1] = x
    return yx


def _safe_cov_logdet(values: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] < 2:
        return float("-inf"), 0.0
    cov = np.cov(arr, rowvar=False)
    cov = np.atleast_2d(cov)
    eig = np.linalg.eigvalsh(cov + np.eye(cov.shape[0], dtype=np.float64) * float(eps))
    eig = np.maximum(eig, float(eps))
    return float(np.log(eig).sum()), float(eig.min())


def _pose_fisher_metrics(
    points_world: np.ndarray,
    pose_w2c: np.ndarray,
    K: np.ndarray,
    eps: float = 1e-6,
) -> dict[str, float]:
    points = np.asarray(points_world, dtype=np.float64)
    if points.shape[0] < 4:
        return {}
    pose = np.asarray(pose_w2c, dtype=np.float64)
    intr = np.asarray(K, dtype=np.float64)
    points_cam = points @ pose[:3, :3].T + pose[:3, 3][None, :]
    z = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & (np.abs(z) > 1e-9)
    points_cam = points_cam[valid]
    if points_cam.shape[0] < 4:
        return {}
    fx = float(intr[0, 0])
    fy = float(intr[1, 1])
    rows = []
    for x, y, z in points_cam:
        j_proj = np.asarray(
            [
                [fx / z, 0.0, -fx * x / (z * z)],
                [0.0, fy / z, -fy * y / (z * z)],
            ],
            dtype=np.float64,
        )
        skew = np.asarray(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ],
            dtype=np.float64,
        )
        rows.append(j_proj @ np.concatenate([-skew, np.eye(3, dtype=np.float64)], axis=1))
    jac = np.concatenate(rows, axis=0)
    fisher = jac.T @ jac
    eig = np.linalg.eigvalsh(fisher + np.eye(6, dtype=np.float64) * float(eps))
    eig = np.maximum(eig, float(eps))
    condition = float(eig.max() / eig.min())
    return {
        "pose_info_logdet": float(np.log(eig).sum()),
        "pose_info_min_eig": float(eig.min()),
        "pose_info_condition": condition,
    }


def sparse_matchability_metrics(
    query_yx: np.ndarray,
    points_world: np.ndarray,
    pose_w2c: np.ndarray,
    K: np.ndarray,
    scores: np.ndarray | None = None,
    margins: np.ndarray | None = None,
) -> dict[str, float]:
    query = np.asarray(query_yx, dtype=np.float64).reshape(-1, 2)
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    count = min(query.shape[0], points.shape[0])
    out: dict[str, float] = {"sparse_match_count": float(count)}
    if count == 0:
        return out
    query = query[:count]
    points = points[:count]
    projected = project_world_points_yx_np(points, pose_w2c, K)
    errors = np.linalg.norm(projected - query, axis=-1)
    valid = np.isfinite(errors)
    out["sparse_valid_match_count"] = float(valid.sum())
    if not valid.any():
        return out
    valid_points = points[valid]
    valid_query = query[valid]
    errors = errors[valid]
    xyz_logdet, xyz_min_eig = _safe_cov_logdet(valid_points)
    img_logdet, img_min_eig = _safe_cov_logdet(valid_query)
    out["sparse_xyz_cov_logdet"] = xyz_logdet
    out["sparse_xyz_cov_min_eig"] = xyz_min_eig
    out["sparse_image_cov_logdet"] = img_logdet
    out["sparse_image_cov_min_eig"] = img_min_eig
    out.update({f"sparse_all_{k}": v for k, v in _pose_fisher_metrics(valid_points, pose_w2c, K).items()})
    inlier8 = errors <= 8.0
    if int(inlier8.sum()) >= 4:
        inlier_points = valid_points[inlier8]
        inlier_query = valid_query[inlier8]
        xyz8_logdet, xyz8_min_eig = _safe_cov_logdet(inlier_points)
        img8_logdet, img8_min_eig = _safe_cov_logdet(inlier_query)
        out["sparse_inlier8_xyz_cov_logdet"] = xyz8_logdet
        out["sparse_inlier8_xyz_cov_min_eig"] = xyz8_min_eig
        out["sparse_inlier8_image_cov_logdet"] = img8_logdet
        out["sparse_inlier8_image_cov_min_eig"] = img8_min_eig
        out.update({f"sparse_inlier8_{k}": v for k, v in _pose_fisher_metrics(inlier_points, pose_w2c, K).items()})
    out.update(
        {
            "sparse_reproj_median_px": float(np.median(errors)),
            "sparse_reproj_mean_px": float(np.mean(errors)),
            "sparse_inlier_2px": float((errors <= 2.0).mean()),
            "sparse_inlier_5px": float((errors <= 5.0).mean()),
            "sparse_inlier_8px": float((errors <= 8.0).mean()),
        }
    )
    if scores is not None:
        score_arr = np.asarray(scores, dtype=np.float64).reshape(-1)[:count][valid]
        if score_arr.size:
            out["sparse_match_score_mean"] = float(np.mean(score_arr))
            out["sparse_match_score_median"] = float(np.median(score_arr))
    if margins is not None:
        margin_arr = np.asarray(margins, dtype=np.float64).reshape(-1)[:count][valid]
        if margin_arr.size:
            out["sparse_top2_margin_mean"] = float(np.mean(margin_arr))
            out["sparse_top2_margin_median"] = float(np.median(margin_arr))
    return out


def _balanced_topk_by_coords(
    score: torch.Tensor,
    coords: torch.Tensor,
    k: int,
    grid_size: int,
    valid: torch.Tensor,
) -> torch.Tensor:
    score = score.float().reshape(-1)
    coords = coords.to(device=score.device, dtype=torch.float32)
    valid = valid.to(device=score.device, dtype=torch.bool).reshape(-1)
    keep = min(max(int(k), 0), int(score.numel()))
    if keep <= 0:
        return torch.empty(0, dtype=torch.long, device=score.device)
    valid = valid & torch.isfinite(score) & torch.isfinite(coords).all(dim=-1)
    eligible = torch.where(valid)[0]
    if eligible.numel() <= keep:
        return eligible[torch.argsort(score[eligible], descending=True)]
    grid = max(1, int(grid_size))
    if grid <= 1:
        return eligible[torch.topk(score[eligible], k=keep).indices]

    selected_coords = coords[eligible]
    lo = selected_coords.amin(dim=0)
    hi = selected_coords.amax(dim=0)
    norm = (selected_coords - lo) / (hi - lo).clamp_min(1e-6)
    bins = torch.clamp((norm * float(grid)).long(), min=0, max=grid - 1)
    multipliers = torch.ones(bins.shape[1], device=score.device, dtype=torch.long)
    for dim in range(bins.shape[1] - 2, -1, -1):
        multipliers[dim] = multipliers[dim + 1] * grid
    cell_ids = (bins * multipliers.view(1, -1)).sum(dim=-1)

    cell_best = []
    for cell in torch.unique(cell_ids, sorted=False):
        local = torch.where(cell_ids == cell)[0]
        if local.numel() == 0:
            continue
        cell_best.append(local[torch.argmax(score[eligible[local]])])
    if not cell_best:
        return eligible[torch.topk(score[eligible], k=keep).indices]

    first_local = torch.stack(cell_best, dim=0)
    first_order = torch.argsort(score[eligible[first_local]], descending=True)
    selected = eligible[first_local[first_order[:keep]]]
    if selected.numel() < keep:
        mask = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
        mask[selected] = True
        remaining = torch.where(valid & ~mask)[0]
        topup = min(keep - int(selected.numel()), int(remaining.numel()))
        if topup > 0:
            selected = torch.cat([selected, remaining[torch.topk(score[remaining], k=topup).indices]], dim=0)
    return selected[:keep]


def _unique_keep_order(indices: list[torch.Tensor], score: torch.Tensor, k: int) -> torch.Tensor:
    chunks = [idx.reshape(-1) for idx in indices if idx.numel() > 0]
    if not chunks:
        return torch.empty(0, dtype=torch.long, device=score.device)
    merged = torch.cat(chunks, dim=0)
    seen = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
    ordered = []
    for idx in merged.tolist():
        if not bool(seen[idx]):
            ordered.append(idx)
            seen[idx] = True
        if len(ordered) >= int(k):
            break
    selected = torch.as_tensor(ordered, dtype=torch.long, device=score.device)
    if selected.numel() < int(k):
        remaining = torch.where(~seen)[0]
        topup = min(int(k) - int(selected.numel()), int(remaining.numel()))
        if topup > 0:
            selected = torch.cat([selected, remaining[torch.topk(score[remaining], k=topup).indices]], dim=0)
    return selected[: int(k)]


def _grid_cell_ids(coords: torch.Tensor, grid_size: int, valid: torch.Tensor) -> torch.Tensor:
    grid = max(1, int(grid_size))
    cells = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
    if grid <= 1 or coords.numel() == 0:
        return cells
    eligible = torch.where(valid)[0]
    if eligible.numel() == 0:
        return cells
    selected = coords[eligible].float()
    lo = selected.amin(dim=0)
    hi = selected.amax(dim=0)
    norm = (selected - lo) / (hi - lo).clamp_min(1e-6)
    bins = torch.clamp((norm * float(grid)).long(), min=0, max=grid - 1)
    multipliers = torch.ones(bins.shape[1], device=coords.device, dtype=torch.long)
    for dim in range(bins.shape[1] - 2, -1, -1):
        multipliers[dim] = multipliers[dim + 1] * grid
    cells[eligible] = (bins * multipliers.view(1, -1)).sum(dim=-1)
    return cells


def _coverage_aware_topk(
    score: torch.Tensor,
    query: torch.Tensor,
    xyz: torch.Tensor,
    keep: int,
    valid: torch.Tensor,
    image_grid_size: int,
    xyz_grid_size: int,
    max_per_image_cell: int,
    max_per_xyz_cell: int,
    min_matches: int,
) -> torch.Tensor:
    score = score.float().reshape(-1)
    keep = min(max(int(keep), 0), int(score.numel()))
    if keep <= 0:
        return torch.empty(0, dtype=torch.long, device=score.device)
    eligible = torch.where(valid)[0]
    if eligible.numel() <= keep:
        return eligible[torch.argsort(score[eligible], descending=True)]

    image_cells = _grid_cell_ids(query, image_grid_size, valid)
    xyz_cells = _grid_cell_ids(xyz[:, :3], xyz_grid_size, valid)
    image_cap = max(1, int(max_per_image_cell))
    xyz_cap = max(1, int(max_per_xyz_cell))
    image_counts: dict[int, int] = {}
    xyz_counts: dict[int, int] = {}
    selected: list[int] = []
    ordered = eligible[torch.argsort(score[eligible], descending=True)]
    for idx_t in ordered:
        idx = int(idx_t.item())
        image_cell = int(image_cells[idx].item())
        xyz_cell = int(xyz_cells[idx].item())
        if image_counts.get(image_cell, 0) >= image_cap:
            continue
        if xyz_counts.get(xyz_cell, 0) >= xyz_cap:
            continue
        selected.append(idx)
        image_counts[image_cell] = image_counts.get(image_cell, 0) + 1
        xyz_counts[xyz_cell] = xyz_counts.get(xyz_cell, 0) + 1
        if len(selected) >= keep:
            break

    min_keep = min(max(int(min_matches), 0), keep)
    if len(selected) < min_keep:
        seen = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
        if selected:
            seen[torch.as_tensor(selected, dtype=torch.long, device=score.device)] = True
        remaining = torch.where(valid & ~seen)[0]
        topup = min(min_keep - len(selected), int(remaining.numel()))
        if topup > 0:
            selected.extend(remaining[torch.topk(score[remaining], k=topup).indices].tolist())
    return torch.as_tensor(selected[:keep], dtype=torch.long, device=score.device)


def local_geometric_consistency_scores(
    query_yx: torch.Tensor,
    points3d: torch.Tensor,
    k: int = 6,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Score matches whose local 3D neighborhoods remain coherent in the query image."""
    query = query_yx.float()
    xyz = points3d.float()
    count = min(int(query.shape[0]), int(xyz.shape[0]))
    if count <= 0:
        return query.new_empty((0,), dtype=torch.float32)
    query = query[:count]
    xyz = xyz[:count, :3]
    valid = torch.isfinite(query).all(dim=-1) & torch.isfinite(xyz).all(dim=-1)
    if count < 3 or int(valid.sum()) < 3:
        return valid.float()

    d3 = torch.cdist(xyz, xyz)
    d2 = torch.cdist(query, query)
    eye = torch.eye(count, dtype=torch.bool, device=query.device)
    d3 = d3.masked_fill(eye, float("inf"))
    d2 = d2.masked_fill(eye, float("inf"))
    neighbor_count = min(max(1, int(k)), count - 1)
    nn = torch.topk(d3, k=neighbor_count, dim=-1, largest=False).indices
    d3_nn = torch.gather(d3, 1, nn).clamp_min(float(eps))
    d2_nn = torch.gather(d2, 1, nn).clamp_min(float(eps))
    finite = torch.isfinite(d3_nn) & torch.isfinite(d2_nn) & valid[:, None]

    ratios = torch.where(finite, d2_nn / d3_nn, torch.ones_like(d2_nn))
    log_ratios = torch.log(ratios.clamp_min(float(eps)))
    local_center = log_ratios.median(dim=1, keepdim=True).values
    local_mad = torch.where(finite, (log_ratios - local_center).abs(), torch.zeros_like(log_ratios))
    spread = local_mad.sum(dim=1) / finite.float().sum(dim=1).clamp_min(1.0)

    finite_d2 = torch.where(torch.isfinite(d2), d2, torch.zeros_like(d2))
    global_query_scale = finite_d2[finite_d2 > 0.0].median() if (finite_d2 > 0.0).any() else query.new_tensor(1.0)
    local_query_radius = torch.where(finite, d2_nn, torch.zeros_like(d2_nn)).sum(dim=1)
    local_query_radius = local_query_radius / finite.float().sum(dim=1).clamp_min(1.0)
    isolation = local_query_radius / global_query_scale.clamp_min(float(eps))

    raw = torch.exp(-spread) * torch.exp(-0.25 * isolation.clamp_min(0.0))
    raw = torch.where(valid, raw, torch.zeros_like(raw))
    return normalize_score01(raw, valid=valid)


def local_image_pair_geometry_scores(
    query_yx: torch.Tensor,
    reference_yx: torch.Tensor,
    k: int = 6,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Score matches whose local 2D neighborhoods agree in query and rendered images."""
    query = query_yx.float()
    reference = reference_yx.float()
    count = min(int(query.shape[0]), int(reference.shape[0]))
    if count <= 0:
        return query.new_empty((0,), dtype=torch.float32)
    query = query[:count]
    reference = reference[:count]
    valid = torch.isfinite(query).all(dim=-1) & torch.isfinite(reference).all(dim=-1)
    if count < 3 or int(valid.sum()) < 3:
        return valid.float()

    d_ref = torch.cdist(reference, reference)
    d_query = torch.cdist(query, query)
    eye = torch.eye(count, dtype=torch.bool, device=query.device)
    d_ref = d_ref.masked_fill(eye, float("inf"))
    d_query = d_query.masked_fill(eye, float("inf"))
    neighbor_count = min(max(1, int(k)), count - 1)
    nn = torch.topk(d_ref, k=neighbor_count, dim=-1, largest=False).indices

    d_ref_nn = torch.gather(d_ref, 1, nn).clamp_min(float(eps))
    d_query_nn = torch.gather(d_query, 1, nn).clamp_min(float(eps))
    neighbor_valid = torch.gather(valid[None, :].expand(count, -1), 1, nn)
    finite = torch.isfinite(d_ref_nn) & torch.isfinite(d_query_nn) & valid[:, None] & neighbor_valid

    log_scale = torch.log((d_query_nn / d_ref_nn).clamp_min(float(eps)))
    local_scale = torch.where(finite, log_scale, torch.zeros_like(log_scale)).median(dim=1, keepdim=True).values
    scale_spread = torch.where(finite, (log_scale - local_scale).abs(), torch.zeros_like(log_scale))
    scale_spread = scale_spread.sum(dim=1) / finite.float().sum(dim=1).clamp_min(1.0)
    global_scale = log_scale[finite].median() if finite.any() else query.new_tensor(0.0)
    scale_bias = (local_scale.squeeze(1) - global_scale).abs()

    q_vec = query[nn] - query[:, None, :]
    r_vec = reference[nn] - reference[:, None, :]
    q_angle = torch.atan2(q_vec[..., 0], q_vec[..., 1])
    r_angle = torch.atan2(r_vec[..., 0], r_vec[..., 1])
    angle_delta = torch.atan2(torch.sin(q_angle - r_angle), torch.cos(q_angle - r_angle))
    local_angle = torch.where(finite, angle_delta, torch.zeros_like(angle_delta)).median(dim=1, keepdim=True).values
    angle_residual = torch.atan2(torch.sin(angle_delta - local_angle), torch.cos(angle_delta - local_angle)).abs()
    angle_spread = torch.where(finite, angle_residual, torch.zeros_like(angle_residual))
    angle_spread = angle_spread.sum(dim=1) / finite.float().sum(dim=1).clamp_min(1.0)

    raw = torch.exp(-scale_spread) * torch.exp(-0.5 * scale_bias) * torch.exp(-0.25 * angle_spread)
    raw = torch.where(valid, raw, torch.zeros_like(raw))
    return normalize_score01(raw, valid=valid)


def select_pnp_match_indices(
    query_yx: torch.Tensor,
    points3d: torch.Tensor,
    scores: torch.Tensor | None = None,
    reference_yx: torch.Tensor | None = None,
    max_matches: int = 0,
    mode: str = "none",
    image_grid_size: int = 8,
    xyz_grid_size: int = 4,
    max_per_image_cell: int = 8,
    max_per_xyz_cell: int = 8,
    min_matches: int = 0,
) -> torch.Tensor:
    """Select PnP correspondences while preserving spatial/pose conditioning."""
    count = min(int(query_yx.shape[0]), int(points3d.shape[0]))
    device = query_yx.device
    if count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    keep = int(max_matches)
    if keep <= 0 or keep >= count or mode == "none":
        return torch.arange(count, dtype=torch.long, device=device)

    query = query_yx[:count].to(device=device, dtype=torch.float32)
    xyz = points3d[:count].to(device=device, dtype=torch.float32)
    if scores is None:
        score = torch.zeros(count, device=device, dtype=torch.float32)
    else:
        score = scores[:count].to(device=device, dtype=torch.float32).reshape(-1)
    valid = torch.isfinite(query).all(dim=-1) & torch.isfinite(xyz).all(dim=-1) & torch.isfinite(score)
    if int(valid.sum()) <= keep:
        eligible = torch.where(valid)[0]
        return eligible[torch.argsort(score[eligible], descending=True)]
    if mode == "score":
        eligible = torch.where(valid)[0]
        return eligible[torch.topk(score[eligible], k=keep).indices]
    if mode == "image_grid":
        return _balanced_topk_by_coords(score, query, keep, image_grid_size, valid)
    if mode == "xyz_grid":
        return _balanced_topk_by_coords(score, xyz, keep, xyz_grid_size, valid)
    if mode == "image_xyz_grid":
        image_keep = _balanced_topk_by_coords(score, query, keep, image_grid_size, valid)
        xyz_keep = _balanced_topk_by_coords(score, xyz, keep, xyz_grid_size, valid)
        return _unique_keep_order([image_keep, xyz_keep], score, keep)
    if mode == "calibrated_coverage":
        return _coverage_aware_topk(
            score,
            query,
            xyz,
            keep,
            valid,
            image_grid_size=image_grid_size,
            xyz_grid_size=xyz_grid_size,
            max_per_image_cell=max_per_image_cell,
            max_per_xyz_cell=max_per_xyz_cell,
            min_matches=min_matches,
        )
    if mode == "local_geometry":
        eligible = torch.where(valid)[0]
        candidate_limit = min(
            int(eligible.numel()),
            max(min(keep * 4, 2048), min(keep, 2048), 64),
        )
        pre_score = normalize_score01(score[eligible])
        candidate_rel = torch.topk(pre_score, k=candidate_limit).indices
        candidate_idx = eligible[candidate_rel]
        geometry = local_geometric_consistency_scores(query[candidate_idx], xyz[candidate_idx], k=6)
        combined = pre_score[candidate_rel] * geometry
        return candidate_idx[torch.topk(combined, k=min(keep, candidate_limit)).indices]
    if mode == "image_pair_geometry":
        if reference_yx is None:
            return select_pnp_match_indices(
                query_yx,
                points3d,
                scores=scores,
                max_matches=max_matches,
                mode="local_geometry",
                image_grid_size=image_grid_size,
                xyz_grid_size=xyz_grid_size,
                max_per_image_cell=max_per_image_cell,
                max_per_xyz_cell=max_per_xyz_cell,
                min_matches=min_matches,
            )
        reference = reference_yx[:count].to(device=device, dtype=torch.float32)
        valid_pair = valid & torch.isfinite(reference).all(dim=-1)
        eligible = torch.where(valid_pair)[0]
        if int(eligible.numel()) <= keep:
            return eligible[torch.argsort(score[eligible], descending=True)]
        candidate_limit = min(
            int(eligible.numel()),
            max(min(keep * 4, 4096), min(keep, 4096), 64),
        )
        pre_score = normalize_score01(score[eligible])
        candidate_rel = torch.topk(pre_score, k=candidate_limit).indices
        candidate_idx = eligible[candidate_rel]
        geometry = local_image_pair_geometry_scores(
            query[candidate_idx],
            reference[candidate_idx],
            k=6,
        )
        combined = pre_score[candidate_rel] * geometry
        return candidate_idx[torch.topk(combined, k=min(keep, candidate_limit)).indices]
    raise ValueError(f"Unsupported PnP prefilter mode: {mode}")


def _pose_verification_score(
    points3d: torch.Tensor,
    query_yx: torch.Tensor,
    K: torch.Tensor,
    pose_w2c: np.ndarray,
    reprojection_error: float,
    image_grid_size: int = 8,
    xyz_grid_size: int = 4,
) -> tuple[float, int]:
    if points3d.numel() == 0 or query_yx.numel() == 0:
        return -float("inf"), 0
    device = points3d.device
    pose = torch.as_tensor(pose_w2c, device=device, dtype=torch.float32).unsqueeze(0)
    proj, valid_z = project_world_to_image_yx(
        points3d.float().unsqueeze(0),
        pose,
        K.to(device=device, dtype=torch.float32),
    )
    err = torch.linalg.norm(proj[0] - query_yx.float(), dim=-1)
    valid = valid_z[0] & torch.isfinite(err)
    inliers = valid & (err <= float(reprojection_error))
    inlier_count = int(inliers.sum().item())
    median_err = float(err[inliers].median().item()) if inliers.any() else (
        float(err[valid].median().item()) if valid.any() else 1e6
    )
    image_cells = _grid_cell_ids(query_yx.float(), image_grid_size, inliers)
    xyz_cells = _grid_cell_ids(points3d[:, :3].float(), xyz_grid_size, inliers)
    image_coverage = int(torch.unique(image_cells[inliers]).numel()) if inliers.any() else 0
    xyz_coverage = int(torch.unique(xyz_cells[inliers]).numel()) if inliers.any() else 0
    score = float(inlier_count) + 0.05 * float(image_coverage + xyz_coverage) - 0.01 * median_err
    return score, inlier_count


def _cluster_pnp_groups(
    points3d: torch.Tensor,
    query_yx: torch.Tensor,
    scores: torch.Tensor,
    cluster_mode: str,
    grid_size: int,
    max_groups: int,
) -> list[torch.Tensor]:
    count = min(int(points3d.shape[0]), int(query_yx.shape[0]), int(scores.shape[0]))
    if count < 4 or int(max_groups) <= 0:
        return []
    valid = torch.isfinite(points3d[:count]).all(dim=-1) & torch.isfinite(query_yx[:count]).all(dim=-1)
    if int(valid.sum()) < 4:
        return []
    modes = []
    if cluster_mode in {"xyz_voxel", "image_xyz_grid"}:
        modes.append(points3d[:count, :3])
    if cluster_mode in {"image_grid", "image_xyz_grid"}:
        modes.append(query_yx[:count])
    groups: list[torch.Tensor] = []
    for coords in modes:
        cells = _grid_cell_ids(coords.float(), grid_size, valid)
        candidates = []
        for cell in torch.unique(cells[valid]).tolist():
            idx = torch.where(valid & (cells == int(cell)))[0]
            if idx.numel() >= 4:
                candidates.append((float(scores[idx].float().sum().item()), idx))
        candidates.sort(key=lambda item: item[0], reverse=True)
        groups.extend(idx for _score, idx in candidates)
    return groups[: int(max_groups)]


def generate_pnp_hypotheses(
    points3d: torch.Tensor,
    query_yx: torch.Tensor,
    K: torch.Tensor,
    *,
    scores: torch.Tensor | None = None,
    max_hypotheses: int = 1,
    cluster_mode: str = "none",
    cluster_grid_size: int = 4,
    reprojection_error: float = 12.0,
    refine_reprojection_error: float = 0.0,
    confidence: float = 0.9999,
    iterations: int = 10000,
    min_iterations: int = 0,
    solver: str = "opencv",
    refine_poselib: bool = False,
) -> list[dict[str, object]]:
    count = min(int(points3d.shape[0]), int(query_yx.shape[0]))
    if count < 4:
        return []
    pts = points3d[:count].float()
    q = query_yx[:count].float()
    score = scores[:count].float() if scores is not None else torch.zeros(count, device=pts.device)
    all_idx = torch.arange(count, device=pts.device, dtype=torch.long)
    groups = [all_idx]
    if int(max_hypotheses) > 1 and cluster_mode != "none":
        groups.extend(
            _cluster_pnp_groups(
                pts,
                q,
                score,
                cluster_mode=cluster_mode,
                grid_size=int(cluster_grid_size),
                max_groups=max(0, int(max_hypotheses) * 2),
            )
        )
    seen: set[tuple[int, ...]] = set()
    hypotheses: list[dict[str, object]] = []
    for group_index, idx in enumerate(groups):
        key = tuple(int(v) for v in idx.detach().cpu().tolist())
        if len(key) < 4 or key in seen:
            continue
        seen.add(key)
        pose, pnp_inliers = solve_pnp_ransac(
            pts[idx].detach().cpu().numpy(),
            q[idx].detach().cpu().numpy(),
            K.detach().cpu().numpy(),
            reprojection_error=reprojection_error,
            refine_reprojection_error=refine_reprojection_error,
            confidence=confidence,
            iterations=iterations,
            min_iterations=min_iterations,
            solver=solver,
            refine_poselib=refine_poselib,
            match_scores=score[idx].detach().cpu().numpy(),
        )
        if pose is None:
            continue
        verify_score, verify_inliers = _pose_verification_score(
            pts,
            q,
            K,
            pose,
            reprojection_error=reprojection_error,
        )
        hypotheses.append(
            {
                "pose": pose,
                "inliers": int(max(int(pnp_inliers), int(verify_inliers))),
                "score": float(verify_score),
                "source_matches": int(idx.numel()),
                "group_index": int(group_index),
                "is_full": bool(group_index == 0),
            }
        )
    hypotheses.sort(key=lambda item: (float(item["score"]), int(item["inliers"])), reverse=True)
    return hypotheses[: max(1, int(max_hypotheses))]


def select_verified_pnp_hypothesis(
    hypotheses: list[dict[str, object]],
    min_score_gain: float = 0.0,
) -> dict[str, object] | None:
    """Choose the best PnP hypothesis while protecting the full-match RANSAC pose."""
    if not hypotheses:
        return None
    ordered = sorted(hypotheses, key=lambda item: (float(item["score"]), int(item["inliers"])), reverse=True)
    best = ordered[0]
    full = next((item for item in ordered if bool(item.get("is_full", False))), None)
    if full is None or best is full:
        return best
    if float(best["score"]) < float(full["score"]) + max(float(min_score_gain), 0.0):
        return full
    return best


def _detector_prior_for_ids(
    scores: torch.Tensor,
    ids_all: torch.Tensor,
    sampled_ids: torch.Tensor,
    num_gaussians: int,
    device: torch.device,
) -> torch.Tensor:
    """Map STDLoc detector scores to candidate Gaussian ids without index drift."""
    score = scores.float().view(-1).to(device=device)
    ids = ids_all.long().view(-1).to(device=device)
    sampled = sampled_ids.long().view(-1).to(device=device)
    if score.numel() == int(num_gaussians):
        prior = score[ids.clamp(0, int(num_gaussians) - 1)]
    elif score.numel() == sampled.numel() and sampled.numel() > 0:
        full = torch.zeros(int(num_gaussians), device=device, dtype=torch.float32)
        valid = (sampled >= 0) & (sampled < int(num_gaussians))
        full[sampled[valid]] = score[valid]
        prior = full[ids.clamp(0, int(num_gaussians) - 1)]
    else:
        if score.numel() < ids.numel() and sampled.numel() > 0:
            full = torch.zeros(int(num_gaussians), device=device, dtype=torch.float32)
            count = min(int(score.numel()), int(sampled.numel()))
            valid = (sampled[:count] >= 0) & (sampled[:count] < int(num_gaussians))
            full[sampled[:count][valid]] = score[:count][valid]
            prior = full[ids.clamp(0, int(num_gaussians) - 1)]
        else:
            prior = score[: ids.numel()]
            if prior.numel() < ids.numel():
                prior = F.pad(prior, (0, ids.numel() - prior.numel()))
    return normalize_score01(prior)


def _calibrated_prior_for_ids(
    scores: torch.Tensor,
    ids_all: torch.Tensor,
    sampled_ids: torch.Tensor,
    num_gaussians: int,
    device: torch.device,
) -> torch.Tensor:
    """Map calibrated TP/(TP+FP) reliability to candidate Gaussian ids."""
    return _detector_prior_for_ids(
        scores,
        ids_all=ids_all,
        sampled_ids=sampled_ids,
        num_gaussians=num_gaussians,
        device=device,
    )


def _load_calibrated_matchability(path: str | Path, device: torch.device) -> torch.Tensor:
    data = torch.load(Path(path), map_location=device)
    if isinstance(data, dict):
        for key in (
            "landmark_matchability",
            "matchability",
            "calibrated_matchability",
            "landmark_tp_rate",
        ):
            if key in data:
                return torch.as_tensor(data[key], device=device, dtype=torch.float32).reshape(-1)
        if "landmark_fp_rate" in data:
            fp_rate = torch.as_tensor(data["landmark_fp_rate"], device=device, dtype=torch.float32)
            return (1.0 - fp_rate).reshape(-1)
        raise KeyError(f"Calibration file {path} does not contain a recognized matchability key")
    return torch.as_tensor(data, device=device, dtype=torch.float32).reshape(-1)


def _match_filter_mode(args: argparse.Namespace) -> str:
    mode = getattr(args, "match_filter_mode", "")
    return mode if mode else getattr(args, "pnp_prefilter", "none")


def _stage_match_filter_mode(args: argparse.Namespace, stage: str) -> str:
    mode = getattr(args, f"{stage}_match_filter_mode", "")
    return mode if mode else _match_filter_mode(args)


def _match_filter_max_matches(args: argparse.Namespace, fallback: int) -> int:
    top_m = int(getattr(args, "match_filter_top_m", 0))
    return top_m if top_m > 0 else int(fallback)


def _stage_match_filter_max_matches(args: argparse.Namespace, stage: str, fallback: int) -> int:
    top_m = int(getattr(args, f"{stage}_match_filter_top_m", 0))
    if top_m > 0:
        return top_m
    return _match_filter_max_matches(args, fallback)


def _match_filter_image_grid_size(args: argparse.Namespace) -> int:
    return int(getattr(args, "match_filter_image_grid_size", getattr(args, "pnp_prefilter_image_grid_size", 8)))


def _match_filter_xyz_grid_size(args: argparse.Namespace) -> int:
    return int(getattr(args, "match_filter_xyz_grid_size", getattr(args, "pnp_prefilter_xyz_grid_size", 4)))


def _match_filter_scores(
    scores: torch.Tensor,
    reliability: torch.Tensor | None,
    weight: float,
    margin: torch.Tensor | None = None,
    margin_weight: float = 0.0,
) -> torch.Tensor:
    out = scores.float()
    if reliability is not None and float(weight) != 0.0:
        rel = reliability.to(device=out.device, dtype=out.dtype).reshape(-1)
        if rel.numel() == out.numel():
            rel = rel.clamp(0.0, 1.0)
            rel = rel - rel.mean()
            out = out + float(weight) * rel
    if margin is not None and float(margin_weight) != 0.0:
        m = margin.to(device=out.device, dtype=out.dtype).reshape(-1)
        if m.numel() == out.numel():
            m = m.clamp_min(0.0)
            m = m - m.mean()
            out = out + float(margin_weight) * m
    return out


def sparse_reprojection_consistency_stats(
    pose_w2c: torch.Tensor,
    sparse_xyz: torch.Tensor,
    sparse_query_yx: torch.Tensor,
    K: torch.Tensor,
    threshold: float,
) -> tuple[float, float] | None:
    if sparse_xyz.shape[0] < 4 or sparse_query_yx.shape[0] < 4:
        return None
    projected, valid_z = project_world_to_image_yx(
        sparse_xyz.float().unsqueeze(0),
        pose_w2c.float(),
        K.float(),
    )
    errors = torch.linalg.norm(projected[0] - sparse_query_yx.float(), dim=-1)
    valid = valid_z[0] & torch.isfinite(errors)
    if int(valid.sum()) < 4:
        return None
    valid_errors = errors[valid]
    median_error = float(torch.median(valid_errors).detach().cpu().item())
    inlier_ratio = float((valid_errors <= float(threshold)).float().mean().detach().cpu().item())
    return median_error, inlier_ratio


def should_reject_dense_pose_by_sparse_consistency(
    previous_pose_w2c: torch.Tensor,
    candidate_pose_w2c: torch.Tensor,
    sparse_xyz: torch.Tensor,
    sparse_query_yx: torch.Tensor,
    K: torch.Tensor,
    threshold: float,
    max_median_ratio: float,
    max_median_increase: float,
    min_inlier_ratio_factor: float,
) -> tuple[bool, dict[str, float]]:
    previous = sparse_reprojection_consistency_stats(
        previous_pose_w2c,
        sparse_xyz,
        sparse_query_yx,
        K,
        threshold,
    )
    candidate = sparse_reprojection_consistency_stats(
        candidate_pose_w2c,
        sparse_xyz,
        sparse_query_yx,
        K,
        threshold,
    )
    if previous is None or candidate is None:
        return False, {}
    prev_median, prev_inlier_ratio = previous
    cand_median, cand_inlier_ratio = candidate
    median_limit = prev_median * float(max_median_ratio) + float(max_median_increase)
    inlier_limit = prev_inlier_ratio * float(min_inlier_ratio_factor)
    reject = cand_median > median_limit and cand_inlier_ratio < inlier_limit
    return reject, {
        "prev_sparse_reproj_median": prev_median,
        "candidate_sparse_reproj_median": cand_median,
        "prev_sparse_reproj_inlier_ratio": prev_inlier_ratio,
        "candidate_sparse_reproj_inlier_ratio": cand_inlier_ratio,
        "sparse_reproj_median_limit": median_limit,
        "sparse_reproj_inlier_limit": inlier_limit,
    }


def should_reject_dense_pose_by_delta(
    previous_pose_w2c: torch.Tensor,
    candidate_pose_w2c: torch.Tensor,
    max_trans_cm: float,
    max_rot_deg: float,
) -> tuple[bool, dict[str, float]]:
    prev_np = previous_pose_w2c[0].detach().cpu().numpy()
    cand_np = candidate_pose_w2c[0].detach().cpu().numpy()
    trans_cm, rot_deg = pose_error_cm_deg(cand_np, prev_np)
    reject = trans_cm > float(max_trans_cm) or rot_deg > float(max_rot_deg)
    return reject, {
        "dense_pose_delta_trans_cm": float(trans_cm),
        "dense_pose_delta_rot_deg": float(rot_deg),
        "dense_pose_delta_max_trans_cm": float(max_trans_cm),
        "dense_pose_delta_max_rot_deg": float(max_rot_deg),
    }


def apply_calibrated_matchability_prior(
    corr_matrix: torch.Tensor,
    reliability: torch.Tensor | None,
    weight: float = 0.0,
) -> torch.Tensor:
    """Add calibrated P(inlier) as a soft match logit prior.

    This stays separate from the STDLoc detector/locability prior so the
    protected landmark bank keeps its original matching prior.
    """

    if reliability is None or float(weight) == 0.0:
        return corr_matrix
    rel = reliability.to(device=corr_matrix.device, dtype=corr_matrix.dtype).reshape(-1)
    if rel.numel() != corr_matrix.shape[-1]:
        return corr_matrix
    eps = torch.finfo(corr_matrix.dtype).eps
    logit = torch.logit(rel.clamp(eps, 1.0 - eps))
    logit = logit - logit.mean()
    return corr_matrix + float(weight) * logit.reshape(*([1] * (corr_matrix.dim() - 1)), -1)


def descriptor_top2_margin(
    query_descriptors: torch.Tensor,
    landmark_descriptors: torch.Tensor,
    landmark_prior: torch.Tensor | None = None,
    prior_weight: float = 0.0,
) -> torch.Tensor:
    """Return per-query ambiguity margin after the same prior used for matching."""
    if query_descriptors.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=query_descriptors.device)
    if landmark_descriptors.shape[0] < 2:
        return torch.zeros(query_descriptors.shape[0], dtype=torch.float32, device=query_descriptors.device)
    corr = F.normalize(query_descriptors.float(), dim=-1) @ F.normalize(landmark_descriptors.float(), dim=-1).T
    if landmark_prior is not None and float(prior_weight) > 0.0:
        corr = apply_match_prior(corr.unsqueeze(0), landmark_prior, weight=float(prior_weight))[0]
    top2 = torch.topk(corr, k=2, dim=-1).values
    return (top2[:, 0] - top2[:, 1]).clamp_min(0.0)


def _mean_metric_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    out: dict[str, float] = {}
    for key in keys:
        values = [row[key] for row in rows if key in row and np.isfinite(row[key])]
        if values:
            out[key] = float(np.mean(values))
    return out


@torch.no_grad()
def build_stdloc_detector_landmark_bank(
    model: HybridFeatureGaussian,
    sp_head: SuperPointOutputHead,
    detector_dir: Path,
    max_landmarks: int,
    descriptor_source: str,
    ply_loc_feature_weight: float,
    hybrid_residual_alpha_max: float,
    device: torch.device,
    candidate_source: str = "sampled",
    score_weights: dict[str, float] | None = None,
    score_ref_poses: torch.Tensor | None = None,
    score_K: torch.Tensor | None = None,
    score_height: int = 0,
    score_width: int = 0,
    legacy_keep_ratio: float = 0.0,
    spatial_grid_size: int = 0,
    score_prior_blend: float = 1.0,
    select_by_score: bool = True,
    ambiguity_radius: float = 0.0,
    ambiguity_max_landmarks: int = 32768,
    keypoint_consensus_maps: torch.Tensor | None = None,
    keypoint_consensus_radius: int = 2,
    keypoint_consensus_descriptor_maps: torch.Tensor | None = None,
    keypoint_consensus_descriptor_weight: float = 0.0,
    teacher_fusion_poses: torch.Tensor | None = None,
    teacher_fusion_descriptor_maps: torch.Tensor | None = None,
    teacher_fusion_K: torch.Tensor | None = None,
    teacher_fusion_height: int = 0,
    teacher_fusion_width: int = 0,
    teacher_fusion_weight: float = 0.0,
    teacher_fusion_chunk_size: int = 2048,
    teacher_fusion_geometry_power: float = 0.0,
    teacher_fusion_centrality_power: float = 0.0,
    calibrated_matchability: torch.Tensor | None = None,
    calibrated_matchability_weight: float = 0.0,
    return_aux: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]] | tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], dict[str, torch.Tensor | None]
]:
    idx_path = detector_dir / "sampled_idx.pkl"
    if not idx_path.exists():
        raise FileNotFoundError(f"STDLoc detector landmark indices not found: {idx_path}")
    sampled_ids = _load_pickle_tensor(idx_path).long().view(-1).to(device=device)
    if sampled_ids.numel() == 0:
        raise RuntimeError(f"STDLoc detector landmark indices are empty: {idx_path}")
    if candidate_source == "all_gaussians":
        ids_all = torch.arange(model.num_gaussians, device=device, dtype=torch.long)
    else:
        ids_all = sampled_ids

    score_path = detector_dir / "sampled_scores.pkl"
    if score_path.exists():
        scores = _load_pickle_tensor(score_path).float().view(-1)
        detector_prior_all = _detector_prior_for_ids(
            scores,
            ids_all=ids_all,
            sampled_ids=sampled_ids,
            num_gaussians=model.num_gaussians,
            device=device,
        )
    else:
        detector_prior_all = torch.ones(ids_all.shape[0], device=device)

    stats: dict[str, float] = {}
    _score_stats("detector", detector_prior_all, stats)
    calibrated_prior_all = None
    if calibrated_matchability is not None:
        calibrated_prior_all = _calibrated_prior_for_ids(
            calibrated_matchability,
            ids_all=ids_all,
            sampled_ids=sampled_ids,
            num_gaussians=model.num_gaussians,
            device=device,
        )
        _score_stats("calibrated_matchability", calibrated_prior_all, stats)
    weights = score_weights or {}
    keep = ids_all.numel() if int(max_landmarks) <= 0 else min(int(max_landmarks), ids_all.numel())
    if weights:
        locability_all = model.get_locability().squeeze(-1)[ids_all].detach()
        geometry_all = gaussian_geometry_score(
            model.get_scaling()[ids_all].detach(),
            model.get_opacity()[ids_all].detach().squeeze(-1),
        )
        components = {
            "detector": detector_prior_all,
            "locability": locability_all,
            "geometry": geometry_all,
        }
        if weights.get("visibility", 0.0) > 0.0 and score_K is not None:
            components["visibility"] = _splatloc_visibility_score(
                model.get_xyz()[ids_all].detach(),
                score_ref_poses,
                score_K.to(device=device, dtype=torch.float32),
                height=score_height,
                width=score_width,
            )
        if weights.get("observability", 0.0) > 0.0 and score_K is not None:
            components["observability"] = _splatloc_observability_score(
                model.get_xyz()[ids_all].detach(),
                score_ref_poses,
                score_K.to(device=device, dtype=torch.float32),
                height=score_height,
                width=score_width,
            )
        if (
            weights.get("keypoint_consensus", 0.0) > 0.0
            and score_K is not None
            and score_ref_poses is not None
            and keypoint_consensus_maps is not None
        ):
            components["keypoint_consensus"] = keypoint_consensus_score(
                model.get_xyz()[ids_all].detach(),
                score_ref_poses,
                score_K.to(device=device, dtype=torch.float32),
                height=score_height,
                width=score_width,
                keypoint_maps=keypoint_consensus_maps,
                radius_px=int(keypoint_consensus_radius),
                descriptor_maps=keypoint_consensus_descriptor_maps,
                descriptor_consistency_weight=float(keypoint_consensus_descriptor_weight),
            )
        if weights.get("ambiguity", 0.0) > 0.0:
            if ids_all.numel() <= int(ambiguity_max_landmarks):
                loc_feature = model.get_ply_loc_feature()
                if loc_feature.numel() > 0 and loc_feature.shape[1] > 0:
                    ambiguity_desc = F.normalize(loc_feature.to(device=device).float()[ids_all], p=2, dim=-1)
                else:
                    ambiguity_desc = decode_gaussian_center_descriptors(model, sp_head, ids_all)
                components["ambiguity"] = descriptor_landmark_distinctiveness(
                    ambiguity_desc,
                    model.get_xyz()[ids_all].detach(),
                    exclusion_radius=float(ambiguity_radius),
                )
                stats["ambiguity_skipped"] = 0.0
            else:
                stats["ambiguity_skipped"] = 1.0
                stats["ambiguity_candidate_count"] = float(ids_all.numel())
        composite = geometric_mean_score(components, weights)
        prior_all = normalize_score01(
            (1.0 - min(max(float(score_prior_blend), 0.0), 1.0)) * detector_prior_all
            + min(max(float(score_prior_blend), 0.0), 1.0) * composite
        )
        for name, value in components.items():
            _score_stats(name, value, stats)
        _score_stats("composite", composite, stats)
        _score_stats("prior", prior_all, stats)
        if (not select_by_score) or keep >= ids_all.numel():
            order = torch.arange(keep, device=device)
            stats["score_selection"] = 0.0
        elif keep < ids_all.numel():
            legacy_keep = int(round(float(keep) * min(max(float(legacy_keep_ratio), 0.0), 1.0)))
            legacy_keep = min(legacy_keep, int(keep), int(sampled_ids.numel()))
            selected_chunks = []
            selected_mask = torch.zeros(ids_all.numel(), dtype=torch.bool, device=device)
            if legacy_keep > 0:
                if candidate_source == "all_gaussians":
                    legacy_order = sampled_ids[:legacy_keep]
                else:
                    legacy_order = torch.arange(legacy_keep, device=device)
                selected_chunks.append(legacy_order)
                selected_mask[legacy_order] = True
            fill_keep = int(keep) - legacy_keep
            if fill_keep > 0:
                if int(spatial_grid_size) > 1:
                    fill_order = spatially_balanced_topk(
                        composite,
                        model.get_xyz()[ids_all].detach(),
                        k=fill_keep,
                        grid_size=int(spatial_grid_size),
                        exclude=selected_mask,
                    )
                else:
                    remaining = torch.where(~selected_mask)[0]
                    fill_order = remaining[torch.topk(composite[remaining], k=min(fill_keep, int(remaining.numel()))).indices]
                selected_chunks.append(fill_order)
                selected_mask[fill_order] = True
            order = torch.cat(selected_chunks, dim=0) if selected_chunks else torch.empty(0, dtype=torch.long, device=device)
            if order.numel() < keep:
                remaining = torch.where(~selected_mask)[0]
                topup = min(int(keep) - int(order.numel()), int(remaining.numel()))
                if topup > 0:
                    order = torch.cat([order, remaining[torch.topk(composite[remaining], k=topup).indices]], dim=0)
            stats["legacy_keep_ratio"] = float(legacy_keep_ratio)
            stats["legacy_kept_count"] = float(legacy_keep)
            stats["matchability_kept_count"] = float(max(0, int(order.numel()) - legacy_keep))
            stats["spatial_grid_size"] = float(spatial_grid_size)
            stats["score_selection"] = 1.0
        else:
            order = torch.arange(ids_all.numel(), device=device)
            stats["score_selection"] = 0.0
        ids = ids_all[order]
        stats["prior_blend"] = float(score_prior_blend)
        stats["candidate_source_all_gaussians"] = float(candidate_source == "all_gaussians")
        prior = prior_all[order].float()
    else:
        if candidate_source == "all_gaussians":
            ids = sampled_ids[:keep]
            prior = detector_prior_all[ids].float()
        else:
            ids = ids_all[:keep]
            prior = detector_prior_all[:keep].float()

    calib_weight = min(max(float(calibrated_matchability_weight), 0.0), 1.0)
    selected_calibrated_prior = None
    if calibrated_matchability is not None:
        selected_calibrated_prior = _calibrated_prior_for_ids(
            calibrated_matchability,
            ids_all=ids,
            sampled_ids=sampled_ids,
            num_gaussians=model.num_gaussians,
            device=device,
        ).float()
    if calibrated_matchability is not None and calib_weight > 0.0:
        prior = normalize_score01((1.0 - calib_weight) * prior.float() + calib_weight * selected_calibrated_prior)
        stats["calibrated_matchability_blend_weight"] = float(calib_weight)

    xyz = model.get_xyz()[ids]

    loc_feature = model.get_ply_loc_feature()
    ply_desc = None
    if loc_feature.numel() > 0 and loc_feature.shape[1] > 0:
        ply_desc = F.normalize(loc_feature.to(device=device).float()[ids], p=2, dim=-1)

    if descriptor_source == "ply_loc":
        if ply_desc is None:
            raise ValueError("descriptor_source=ply_loc requires loc_* features in the input PLY")
        desc = ply_desc
    elif descriptor_source == "hybrid_ply_blend":
        if ply_desc is None:
            raise ValueError("descriptor_source=hybrid_ply_blend requires loc_* features in the input PLY")
        hybrid_desc = decode_gaussian_center_descriptors(model, sp_head, ids)
        weight = min(max(float(ply_loc_feature_weight), 0.0), 1.0)
        desc = F.normalize((1.0 - weight) * hybrid_desc + weight * ply_desc, p=2, dim=-1)
    elif descriptor_source == "hybrid_ply_gated_residual":
        if ply_desc is None:
            raise ValueError("descriptor_source=hybrid_ply_gated_residual requires loc_* features in the input PLY")
        hybrid_desc = decode_gaussian_center_descriptors(model, sp_head, ids)
        gate = model.get_locability().squeeze(-1)[ids].detach()
        desc = gated_residual_descriptor_blend(
            ply_desc,
            hybrid_desc,
            gate=gate,
            alpha_max=float(hybrid_residual_alpha_max),
        )
    else:
        desc = decode_gaussian_center_descriptors(model, sp_head, ids)

    fusion_weight = min(max(float(teacher_fusion_weight), 0.0), 1.0)
    if (
        fusion_weight > 0.0
        and teacher_fusion_descriptor_maps is not None
        and teacher_fusion_poses is not None
        and teacher_fusion_K is not None
    ):
        fused_desc, fusion_counts = fuse_projected_teacher_descriptors(
            xyz.detach(),
            teacher_fusion_poses,
            teacher_fusion_K,
            teacher_fusion_descriptor_maps,
            height=int(teacher_fusion_height),
            width=int(teacher_fusion_width),
            chunk_size=int(teacher_fusion_chunk_size),
            geometry_power=float(teacher_fusion_geometry_power),
            centrality_power=float(teacher_fusion_centrality_power),
        )
        usable = fusion_counts > 0.0
        if int(usable.sum()) > 0:
            desc = torch.where(
                usable[:, None],
                F.normalize((1.0 - fusion_weight) * desc.float() + fusion_weight * fused_desc.float(), p=2, dim=-1),
                desc.float(),
            )
        stats["teacher_fusion_weight"] = float(fusion_weight)
        stats["teacher_fusion_geometry_power"] = float(teacher_fusion_geometry_power)
        stats["teacher_fusion_centrality_power"] = float(teacher_fusion_centrality_power)
        stats["teacher_fusion_observed_mean"] = float(fusion_counts.float().mean().detach().cpu()) if fusion_counts.numel() else 0.0
        stats["teacher_fusion_observed_ratio"] = float(usable.float().mean().detach().cpu()) if usable.numel() else 0.0

    result = (xyz, F.normalize(desc.float(), p=2, dim=-1), prior.float(), stats)
    if return_aux:
        aux = {
            "gaussian_ids": ids.detach(),
            "calibrated_matchability": selected_calibrated_prior.detach()
            if selected_calibrated_prior is not None
            else None,
        }
        return (*result, aux)
    return result


def select_view_landmark_indices(
    prior: torch.Tensor,
    flat_ids: torch.Tensor,
    height: int,
    width: int,
    quota: int,
    selection: str = "global",
    grid_size: int = 8,
) -> torch.Tensor:
    """Select rendered landmarks from one reference view with optional spatial diversity."""
    num_points = int(prior.numel())
    device = prior.device
    if num_points == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if selection == "global" or int(quota) <= 0 or int(quota) >= num_points:
        return torch.arange(num_points, dtype=torch.long, device=device)
    keep = max(1, min(int(quota), num_points))
    if selection == "per_view":
        return torch.topk(prior.float(), k=keep).indices
    if selection != "per_view_spatial":
        raise ValueError(f"Unsupported landmark selection mode: {selection}")

    grid = max(1, int(grid_size))
    flat_ids = flat_ids.to(device=device, dtype=torch.long)
    y = flat_ids // max(int(width), 1)
    x = flat_ids % max(int(width), 1)
    cell_cols = max(1, math.ceil(float(width) / float(grid)))
    cells = (y // grid) * cell_cols + (x // grid)
    unique_cells = torch.unique(cells, sorted=True)
    if unique_cells.numel() == 0:
        return torch.topk(prior.float(), k=keep).indices

    per_cell = max(1, math.ceil(float(keep) / float(unique_cells.numel())))
    selected_chunks = []
    for cell in unique_cells:
        ids = torch.where(cells == cell)[0]
        cell_keep = min(per_cell, int(ids.numel()))
        if cell_keep <= 0:
            continue
        order = torch.topk(prior[ids].float(), k=cell_keep).indices
        selected_chunks.append(ids[order])
    if selected_chunks:
        selected = torch.cat(selected_chunks, dim=0)
    else:
        selected = torch.empty(0, dtype=torch.long, device=device)

    if selected.numel() > keep:
        order = torch.topk(prior[selected].float(), k=keep).indices
        selected = selected[order]
    elif selected.numel() < keep:
        mask = torch.zeros(num_points, dtype=torch.bool, device=device)
        mask[selected] = True
        remaining = torch.where(~mask)[0]
        topup = min(keep - int(selected.numel()), int(remaining.numel()))
        if topup > 0:
            order = torch.topk(prior[remaining].float(), k=topup).indices
            selected = torch.cat([selected, remaining[order]], dim=0)
    return selected


@torch.no_grad()
def build_rendered_landmark_bank(
    model: HybridFeatureGaussian,
    sp_head: SuperPointOutputHead,
    renderer,
    dataset: CambridgeHybridDataset,
    max_landmarks: int,
    ref_views: int,
    stride: int,
    alpha_threshold: float,
    device: torch.device,
    selection: str = "global",
    per_view_quota: int = 0,
    view_grid_size: int = 8,
    descriptor_source: str = "hybrid",
    ply_loc_feature_weight: float = 0.5,
    hybrid_residual_alpha_max: float = 0.05,
    score_weights: dict[str, float] | None = None,
    score_ref_poses: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    xyz_chunks = []
    desc_chunks = []
    prior_chunks = []
    component_chunks: dict[str, list[torch.Tensor]] = {}
    weights = score_weights or {}
    max_ref_views = len(dataset) if int(ref_views) <= 0 else min(len(dataset), int(ref_views))
    auto_quota = math.ceil(float(max_landmarks) / float(max(max_ref_views, 1)))
    raw_count = 0
    pre_global_count = 0
    for idx in tqdm(range(max_ref_views), desc="Rendering landmark bank", dynamic_ncols=True):
        item = dataset[idx]
        pose = item["pose_w2c"].to(device=device, dtype=torch.float32).unsqueeze(0)
        render = render_hybrid_superpoint(
            model,
            sp_head,
            renderer,
            pose,
            descriptor_source=descriptor_source,
            ply_loc_feature_weight=ply_loc_feature_weight,
            hybrid_residual_alpha_max=hybrid_residual_alpha_max,
        )
        world_points, _grid_yx, valid_depth = unproject_dense_depth_to_world(
            render["depth"].float(),
            pose,
            renderer.K.float(),
        )
        alpha = render["alpha"][0].float() * valid_depth[0].view_as(render["alpha"][0]).float()
        xyz, desc, ids = flatten_rendered_landmarks(
            render["descriptor"][0],
            world_points[0],
            alpha,
            stride=stride,
            alpha_threshold=alpha_threshold,
        )
        if xyz.numel() == 0:
            continue
        component_maps: dict[str, torch.Tensor] = {}
        if render["locability"] is not None:
            component_maps["locability"] = render["locability"][0, 0].float()
        if weights:
            component_maps["detector"] = superpoint_detector_saliency(render["detector"][0])
            component_maps["alpha"] = alpha.float().clamp(0.0, 1.0)
            component_maps["depth"] = depth_consistency_score(
                render["depth"][0].float(),
                valid=alpha > float(alpha_threshold),
                window_size=getattr(dataset, "landmark_score_depth_window", 3),
            )
            if weights.get("observability", 0.0) > 0.0:
                obs = projection_jacobian_observability(
                    world_points[:, :, :].float(),
                    pose.float(),
                    renderer.K.float(),
                    min_depth=0.05,
                )[0].view_as(alpha)
                component_maps["observability"] = normalize_score01(obs, valid=alpha > float(alpha_threshold))
            if weights.get("distinctiveness", 0.0) > 0.0:
                component_maps["distinctiveness"] = descriptor_local_distinctiveness(
                    render["descriptor"][0],
                    valid=alpha > float(alpha_threshold),
                )
            score_map = geometric_mean_score(component_maps, weights)
            prior = score_map.flatten()[ids]
        elif render["locability"] is not None:
            prior = render["locability"][0, 0].flatten()[ids]
        else:
            prior = torch.ones(xyz.shape[0], device=device)
        flat_components = {name: value.flatten()[ids] for name, value in component_maps.items()}
        raw_count += int(xyz.shape[0])
        if selection != "global":
            quota = int(per_view_quota) if int(per_view_quota) > 0 else auto_quota
            view_keep = select_view_landmark_indices(
                prior,
                ids,
                height=int(alpha.shape[-2]),
                width=int(alpha.shape[-1]),
                quota=quota,
                selection=selection,
                grid_size=view_grid_size,
            )
            xyz = xyz[view_keep]
            desc = desc[view_keep]
            prior = prior[view_keep]
            flat_components = {name: value[view_keep] for name, value in flat_components.items()}
        pre_global_count += int(xyz.shape[0])
        xyz_chunks.append(xyz)
        desc_chunks.append(desc)
        prior_chunks.append(prior)
        for name, value in flat_components.items():
            component_chunks.setdefault(name, []).append(value.detach())
    if not xyz_chunks:
        raise RuntimeError("Rendered landmark bank is empty; lower alpha_threshold or check the checkpoint")
    xyz_all = torch.cat(xyz_chunks, dim=0)
    desc_all = torch.cat(desc_chunks, dim=0)
    prior_all = torch.cat(prior_chunks, dim=0).clamp(0.0, 1.0)
    components_all = {
        name: torch.cat(values, dim=0).clamp(0.0, 1.0)
        for name, values in component_chunks.items()
        if values
    }
    if weights and weights.get("visibility", 0.0) > 0.0:
        visibility = _splatloc_visibility_score(
            xyz_all.detach(),
            score_ref_poses,
            renderer.K.float(),
            height=int(renderer.image_height),
            width=int(renderer.image_width),
        )
        components_all["visibility"] = visibility
        prior_all = geometric_mean_score(
            {"matchability": prior_all, "visibility": visibility},
            {"matchability": 1.0, "visibility": float(weights["visibility"])},
        )
    keep = xyz_all.shape[0] if int(max_landmarks) <= 0 else min(int(max_landmarks), xyz_all.shape[0])
    if keep < xyz_all.shape[0]:
        _, ids = torch.topk(prior_all, k=keep)
        xyz_all = xyz_all[ids]
        desc_all = desc_all[ids]
        prior_all = prior_all[ids]
        components_all = {name: value[ids] for name, value in components_all.items()}
    stats: dict[str, float] = {
        "raw_count": float(raw_count),
        "pre_global_count": float(pre_global_count),
        "kept_count": float(xyz_all.shape[0]),
    }
    _score_stats("composite", prior_all, stats)
    for name, value in components_all.items():
        _score_stats(name, value, stats)
    print(
        f"[eval] rendered landmark bank selected: raw={raw_count}, "
        f"pre_global={pre_global_count}, kept={xyz_all.shape[0]}, selection={selection}"
    )
    return xyz_all, desc_all, prior_all, stats


@torch.no_grad()
def localize_one(
    model: HybridFeatureGaussian,
    sp_head: SuperPointOutputHead,
    renderer,
    teacher: SuperPointNet,
    rgb: torch.Tensor,
    K_feature: torch.Tensor,
    K_full: torch.Tensor,
    landmark_xyz: torch.Tensor,
    landmark_desc: torch.Tensor,
    landmark_prior: torch.Tensor,
    args: argparse.Namespace,
    image_name: str = "",
    teacher_cache: SuperPointTeacherCache | None = None,
    full_renderer=None,
    stdloc_detector: StdlocKeypointDetector | None = None,
    teacher_rgb: torch.Tensor | None = None,
    landmark_filter_reliability: torch.Tensor | None = None,
) -> dict[str, object]:
    if hasattr(renderer, "K"):
        renderer.K.copy_(K_feature.to(device=renderer.K.device, dtype=renderer.K.dtype))
    if full_renderer is not None and hasattr(full_renderer, "K"):
        full_renderer.K.copy_(K_full.to(device=full_renderer.K.device, dtype=full_renderer.K.dtype))
    teacher_input = teacher_rgb if teacher_rgb is not None else rgb
    if teacher_cache is not None and image_name:
        teacher_desc, teacher_det, _cache_hits = extract_superpoint_teacher_batch(
            teacher,
            teacher_input,
            [image_name],
            cache=teacher_cache,
            expected_hw=(teacher_input.shape[-2] // 8, teacher_input.shape[-1] // 8),
        )
    else:
        teacher_desc, teacher_det = teacher(superpoint_gray(teacher_input))
    feature_height = int(getattr(renderer, "image_height", teacher_desc.shape[-2]))
    feature_width = int(getattr(renderer, "image_width", teacher_desc.shape[-1]))
    teacher_desc_raw, teacher_desc, teacher_det = prepare_query_teacher_maps(
        teacher_desc,
        teacher_det,
        feature_height,
        feature_width,
    )
    query_detector = getattr(args, "query_detector", "superpoint")
    sparse_K = K_feature
    sparse_query_offset = 0.0
    query_keypoints_are_full_res = False
    if query_detector == "stdloc":
        if stdloc_detector is None:
            raise ValueError("query_detector=stdloc requires a loaded StdlocKeypointDetector")
        full_h = int(rgb.shape[-2])
        full_w = int(rgb.shape[-1])
        query_feature_map = F.normalize(
            upsample_feature_map(teacher_desc_raw, full_h, full_w),
            p=2,
            dim=0,
        )
        keypoints, kp_scores = extract_stdloc_detector_keypoints(
            query_feature_map,
            stdloc_detector,
            max_keypoints=args.query_keypoints,
            nms_radius=args.nms_radius,
        )
        query_desc = sample_descriptors_bilinear(query_feature_map, keypoints)
        sparse_K = K_full
        sparse_query_offset = 0.5
        query_keypoints_are_full_res = True
    else:
        keypoints, kp_scores = extract_keypoints_from_detector_logits(
            teacher_det,
            max_keypoints=args.query_keypoints,
            confidence_threshold=args.keypoint_threshold,
            nms_radius=args.nms_radius,
        )
        query_desc = sample_descriptors_bilinear(teacher_desc, keypoints)
    if teacher_cache is not None and image_name:
        maybe_write_superpoint_metadata(
            teacher_cache,
            [image_name],
            teacher_desc.unsqueeze(0),
            teacher_det.unsqueeze(0),
            keypoints.unsqueeze(0),
            query_desc.unsqueeze(0),
        )
    sparse_margin = (
        args.sparse_match_second_best_margin
        if args.sparse_match_second_best_margin is not None
        else args.match_second_best_margin
    )
    dense_margin = (
        args.dense_match_second_best_margin
        if args.dense_match_second_best_margin is not None
        else args.match_second_best_margin
    )
    dense_query_offset = float(getattr(args, "dense_query_pixel_center_offset", 0.0))
    refine_poselib = bool(getattr(args, "poselib_refine", False))
    pnp_min_iterations = int(getattr(args, "pnp_min_iterations", 0))
    sparse_pnp_iterations = getattr(args, "sparse_pnp_iterations", None)
    sparse_pnp_min_iterations = getattr(args, "sparse_pnp_min_iterations", None)
    dense_pnp_iterations = getattr(args, "dense_pnp_iterations", None)
    dense_pnp_min_iterations = getattr(args, "dense_pnp_min_iterations", None)
    sparse_pnp_iterations = int(args.pnp_iterations if sparse_pnp_iterations is None else sparse_pnp_iterations)
    sparse_pnp_min_iterations = int(pnp_min_iterations if sparse_pnp_min_iterations is None else sparse_pnp_min_iterations)
    dense_pnp_iterations = int(args.pnp_iterations if dense_pnp_iterations is None else dense_pnp_iterations)
    dense_pnp_min_iterations = int(pnp_min_iterations if dense_pnp_min_iterations is None else dense_pnp_min_iterations)
    descriptor_source = getattr(args, "descriptor_source", "hybrid")
    ply_loc_feature_weight = float(getattr(args, "ply_loc_feature_weight", 0.5))
    match_filter_margin_weight = float(getattr(args, "match_filter_margin_weight", 0.0))
    match_calibrated_prior_weight = float(getattr(args, "match_calibrated_prior_weight", 0.0))
    sparse_reprojection_error = (
        args.sparse_reprojection_error
        if args.sparse_reprojection_error is not None
        else args.reprojection_error
    )
    dense_reprojection_error = (
        args.dense_reprojection_error
        if args.dense_reprojection_error is not None
        else args.reprojection_error
    )
    sparse_matcher, dense_matcher = resolve_matchers(args)
    sparse_margin_by_query = None
    sparse_diag_corr = None
    calibrated_sparse_reliability = landmark_filter_reliability
    if sparse_matcher == "stdloc_parity":
        corr = query_desc.float() @ F.normalize(landmark_desc.float(), dim=-1).T
        corr = apply_match_prior(corr.unsqueeze(0), landmark_prior, weight=args.locability_prior_weight)
        corr = apply_calibrated_matchability_prior(
            corr,
            calibrated_sparse_reliability,
            weight=match_calibrated_prior_weight,
        )
        sparse_diag_corr = corr[0]
        _b, q_ids, lm_ids, _scores = match_correlation_matrix(
            corr,
            threshold=args.sparse_match_threshold,
            dual_softmax_temp=args.sparse_dual_softmax_temp,
            use_dual_softmax=args.sparse_dual_softmax,
            use_mnn=args.mnn,
            topk=1,
            second_best_margin=sparse_margin,
        )
    elif sparse_matcher in {"lightglue", "dim"}:
        # LightGlue needs two 2D feature sets.  The global 3D landmark bank has no
        # query-view 2D layout, so sparse initialization remains top-k and the
        # learned matcher is applied in the rendered refinement stage.
        if getattr(args, "matchability_diagnostics", False) or match_filter_margin_weight != 0.0:
            sparse_diag_corr = (
                F.normalize(query_desc.float(), dim=-1)
                @ F.normalize(landmark_desc.float(), dim=-1).T
            )
            sparse_diag_corr = apply_match_prior(
                sparse_diag_corr.unsqueeze(0),
                landmark_prior,
                weight=args.locability_prior_weight,
            )[0]
            sparse_diag_corr = apply_calibrated_matchability_prior(
                sparse_diag_corr.unsqueeze(0),
                calibrated_sparse_reliability,
                weight=match_calibrated_prior_weight,
            )[0]
        corr = (
            F.normalize(query_desc.float(), dim=-1)
            @ F.normalize(landmark_desc.float(), dim=-1).T
        )
        corr = apply_match_prior(corr.unsqueeze(0), landmark_prior, weight=args.locability_prior_weight)
        corr = apply_calibrated_matchability_prior(
            corr,
            calibrated_sparse_reliability,
            weight=match_calibrated_prior_weight,
        )
        _b, q_ids, lm_ids, _scores = match_correlation_matrix(
            corr,
            threshold=args.sparse_match_threshold,
            dual_softmax_temp=args.sparse_dual_softmax_temp,
            use_dual_softmax=False,
            use_mnn=False,
            topk=1,
            second_best_margin=sparse_margin,
        )
    else:
        if getattr(args, "matchability_diagnostics", False) or match_filter_margin_weight != 0.0:
            sparse_diag_corr = (
                F.normalize(query_desc.float(), dim=-1)
                @ F.normalize(landmark_desc.float(), dim=-1).T
            )
            sparse_diag_corr = apply_match_prior(
                sparse_diag_corr.unsqueeze(0),
                landmark_prior,
                weight=args.locability_prior_weight,
            )[0]
            sparse_diag_corr = apply_calibrated_matchability_prior(
                sparse_diag_corr.unsqueeze(0),
                calibrated_sparse_reliability,
                weight=match_calibrated_prior_weight,
            )[0]
        corr = (
            F.normalize(query_desc.float(), dim=-1)
            @ F.normalize(landmark_desc.float(), dim=-1).T
        )
        corr = apply_match_prior(corr.unsqueeze(0), landmark_prior, weight=args.locability_prior_weight)
        corr = apply_calibrated_matchability_prior(
            corr,
            calibrated_sparse_reliability,
            weight=match_calibrated_prior_weight,
        )
        _b, q_ids, lm_ids, _scores = match_correlation_matrix(
            corr,
            threshold=args.sparse_match_threshold,
            dual_softmax_temp=args.sparse_dual_softmax_temp,
            use_dual_softmax=False,
            use_mnn=False,
            topk=1,
            second_best_margin=sparse_margin,
        )
    if (
        (getattr(args, "matchability_diagnostics", False) or match_filter_margin_weight != 0.0)
        and sparse_diag_corr is not None
        and sparse_diag_corr.shape[1] >= 2
        and q_ids.numel() > 0
    ):
        top2 = torch.topk(sparse_diag_corr, k=2, dim=-1).values
        sparse_margin_by_query = (top2[:, 0] - top2[:, 1])[q_ids]
    sparse_reliability = None
    if lm_ids.numel() > 0:
        reliability_source = landmark_filter_reliability if landmark_filter_reliability is not None else landmark_prior
        sparse_reliability = reliability_source[lm_ids]
    sparse_filter_scores = _match_filter_scores(
        _scores,
        sparse_reliability,
        getattr(args, "match_filter_calibrated_score_weight", 0.0),
        margin=sparse_margin_by_query,
        margin_weight=match_filter_margin_weight,
    )
    sparse_keep = select_pnp_match_indices(
        keypoints[q_ids] + sparse_query_offset,
        landmark_xyz[lm_ids],
        scores=sparse_filter_scores,
        max_matches=_stage_match_filter_max_matches(args, "sparse", getattr(args, "sparse_pnp_max_matches", 0)),
        mode=_stage_match_filter_mode(args, "sparse"),
        image_grid_size=_match_filter_image_grid_size(args),
        xyz_grid_size=_match_filter_xyz_grid_size(args),
        max_per_image_cell=getattr(args, "match_filter_max_per_image_cell", 8),
        max_per_xyz_cell=getattr(args, "match_filter_max_per_xyz_cell", 8),
        min_matches=getattr(args, "match_filter_min_matches", 0),
    )
    if sparse_keep.numel() < q_ids.numel():
        q_ids = q_ids[sparse_keep]
        lm_ids = lm_ids[sparse_keep]
        _scores = _scores[sparse_keep]
        sparse_filter_scores = sparse_filter_scores[sparse_keep]
        if sparse_margin_by_query is not None:
            sparse_margin_by_query = sparse_margin_by_query[sparse_keep]
    sparse_query_for_pnp = keypoints[q_ids] + sparse_query_offset
    if int(getattr(args, "pnp_hypotheses", 1)) > 1:
        sparse_hypotheses = generate_pnp_hypotheses(
            landmark_xyz[lm_ids],
            sparse_query_for_pnp,
            sparse_K,
            scores=sparse_filter_scores,
            max_hypotheses=int(getattr(args, "pnp_hypotheses", 1)),
            cluster_mode=str(getattr(args, "pnp_cluster_mode", "none")),
            cluster_grid_size=int(getattr(args, "pnp_cluster_grid_size", 4)),
            reprojection_error=sparse_reprojection_error,
            refine_reprojection_error=args.refine_reprojection_error,
            confidence=args.pnp_confidence,
            iterations=sparse_pnp_iterations,
            min_iterations=sparse_pnp_min_iterations,
            solver=args.solver,
            refine_poselib=refine_poselib,
        )
        verify_topk = min(
            int(getattr(args, "pnp_dense_verify_topk", 1)),
            len(sparse_hypotheses),
        )
        if verify_topk > 1:
            verify_full = query_keypoints_are_full_res and full_renderer is not None
            verify_renderer = full_renderer if verify_full else renderer
            verify_K = K_full if verify_full else K_feature
            verify_query_yx = keypoints[q_ids] if verify_full else (
                keypoints[q_ids] / 8.0 if query_keypoints_are_full_res else keypoints[q_ids]
            )
            for hypothesis in sparse_hypotheses[:verify_topk]:
                hyp_pose = torch.from_numpy(np.asarray(hypothesis["pose"], dtype=np.float32)).to(
                    device=rgb.device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                try:
                    hyp_render = render_hybrid_superpoint(
                        model,
                        sp_head,
                        verify_renderer,
                        hyp_pose,
                        descriptor_source=descriptor_source,
                        ply_loc_feature_weight=ply_loc_feature_weight,
                        hybrid_residual_alpha_max=getattr(args, "hybrid_residual_alpha_max", 0.05),
                    )
                    proj_yx, valid_proj = project_world_to_image_yx(
                        landmark_xyz[lm_ids].unsqueeze(0),
                        hyp_pose,
                        verify_K,
                    )
                    proj = proj_yx[0]
                    desc_map = hyp_render["descriptor"][0]
                    in_frame = (
                        valid_proj[0]
                        & (proj[:, 0] >= 0.0)
                        & (proj[:, 0] <= desc_map.shape[-2] - 1)
                        & (proj[:, 1] >= 0.0)
                        & (proj[:, 1] <= desc_map.shape[-1] - 1)
                    )
                    if int(in_frame.sum()) >= 4:
                        rendered_at_proj = sample_descriptors_bilinear(desc_map, proj[in_frame])
                        query_at_proj = F.normalize(query_desc[q_ids][in_frame].float(), p=2, dim=-1)
                        desc_agreement = (query_at_proj * rendered_at_proj.float()).sum(dim=-1).mean()
                        reproj_err = torch.linalg.norm(proj[in_frame] - verify_query_yx[in_frame], dim=-1)
                        geom_bonus = (reproj_err <= float(sparse_reprojection_error)).float().mean()
                        hypothesis["score"] = float(hypothesis["score"]) + float(desc_agreement.item()) + float(
                            geom_bonus.item()
                        )
                except Exception:
                    continue
            sparse_hypotheses.sort(key=lambda item: (float(item["score"]), int(item["inliers"])), reverse=True)
        selected_hypothesis = select_verified_pnp_hypothesis(
            sparse_hypotheses,
            min_score_gain=getattr(args, "pnp_hypothesis_min_score_gain", 0.0),
        )
        sparse_pose = selected_hypothesis["pose"] if selected_hypothesis is not None else None
        sparse_inliers = int(selected_hypothesis["inliers"]) if selected_hypothesis is not None else 0
    else:
        sparse_pose, sparse_inliers = solve_pnp_ransac(
            landmark_xyz[lm_ids].detach().cpu().numpy(),
            sparse_query_for_pnp.detach().cpu().numpy(),
            sparse_K.detach().cpu().numpy(),
            reprojection_error=sparse_reprojection_error,
            refine_reprojection_error=args.refine_reprojection_error,
            confidence=args.pnp_confidence,
            iterations=sparse_pnp_iterations,
            min_iterations=sparse_pnp_min_iterations,
            solver=args.solver,
            refine_poselib=refine_poselib,
            match_scores=sparse_filter_scores.detach().cpu().numpy(),
        )
    if sparse_pose is None:
        out = {"pose_w2c": None, "sparse_pose_w2c": None, "sparse_inliers": 0, "dense_inliers": 0}
        if getattr(args, "matchability_diagnostics", False):
            out["sparse_match_diagnostics"] = {
                "query_yx": (keypoints[q_ids] + sparse_query_offset).detach().cpu().numpy(),
                "xyz": landmark_xyz[lm_ids].detach().cpu().numpy(),
                "scores": _scores.detach().cpu().numpy(),
                "margins": None
                if sparse_margin_by_query is None
                else sparse_margin_by_query.detach().cpu().numpy(),
                "coord_space": "full" if query_keypoints_are_full_res else "feature",
            }
        return out

    pose = torch.from_numpy(sparse_pose).to(device=rgb.device, dtype=torch.float32).unsqueeze(0)
    dense_inliers = 0
    dense_rejections = 0
    dense_rejection_stats: dict[str, float] = {}
    for _ in range(args.dense_iters):
        previous_pose = pose
        previous_dense_inliers = dense_inliers
        render = render_hybrid_superpoint(
            model,
            sp_head,
            renderer,
            pose,
            descriptor_source=descriptor_source,
            ply_loc_feature_weight=ply_loc_feature_weight,
            hybrid_residual_alpha_max=getattr(args, "hybrid_residual_alpha_max", 0.05),
        )
        desc_map = render["descriptor"][0]
        loc_map = render["locability"]
        if dense_matcher == "stdloc_parity":
            full_h = int(rgb.shape[-2])
            full_w = int(rgb.shape[-1])
            query_fine = F.normalize(
                upsample_feature_map(teacher_desc_raw, full_h, full_w),
                p=2,
                dim=0,
            )
            if full_renderer is not None:
                full_render = render_hybrid_superpoint(
                    model,
                    sp_head,
                    full_renderer,
                    pose,
                    descriptor_source=descriptor_source,
                    ply_loc_feature_weight=ply_loc_feature_weight,
                    hybrid_residual_alpha_max=getattr(args, "hybrid_residual_alpha_max", 0.05),
                )
                rendered_fine = full_render["descriptor"][0]
                dense_depth = full_render["depth"][0].float()
                full_loc_map = full_render["locability"]
                rendered_prior = full_loc_map[0, 0] if full_loc_map is not None else None
                rendered_coarse = F.interpolate(
                    rendered_fine.unsqueeze(0).float(),
                    size=teacher_desc.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0]
            else:
                rendered_fine = upsample_feature_map(desc_map, full_h, full_w)
                dense_depth = F.interpolate(
                    render["depth"].unsqueeze(1).float(),
                    size=(full_h, full_w),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                rendered_prior = (
                    upsample_feature_map(loc_map[0, 0].unsqueeze(0), full_h, full_w).squeeze(0)
                    if loc_map is not None
                    else None
                )
                rendered_coarse = desc_map
            try:
                dense_matches = coarse_to_fine_dense_matches(
                    query_fine,
                    rendered_fine,
                    query_coarse_map=teacher_desc,
                    rendered_coarse_map=rendered_coarse,
                    rendered_prior=rendered_prior,
                    prior_weight=args.locability_prior_weight,
                    window_size=8,
                    coarse_dual_softmax_temp=args.dense_dual_softmax_temp,
                    fine_dual_softmax_temp=args.fine_dual_softmax_temp,
                    coarse_threshold=args.dense_match_threshold,
                    fine_threshold=args.dense_match_threshold,
                    use_mnn=True,
                    subpixel_refine=args.subpixel_refine,
                    subpixel_temperature=args.subpixel_temperature,
                )
            except ValueError:
                break
            if dense_matches.query_yx.shape[0] < 4:
                break
            p3d = unproject_positions_yx(
                dense_matches.rendered_yx,
                dense_depth,
                pose[0].float(),
                K_full.float(),
                pixel_center_offset=args.render_pixel_center_offset,
            )
            valid = torch.isfinite(p3d).all(dim=-1) & (p3d[:, 2].abs() < 1e8)
            if valid.sum() < 4:
                break
            valid_ids = torch.where(valid)[0]
            dense_keep = select_pnp_match_indices(
                dense_matches.query_yx[valid],
                p3d[valid],
                scores=dense_matches.scores[valid_ids],
                reference_yx=dense_matches.rendered_yx[valid],
                max_matches=_stage_match_filter_max_matches(args, "dense", getattr(args, "dense_pnp_max_matches", 0)),
                mode=_stage_match_filter_mode(args, "dense"),
                image_grid_size=_match_filter_image_grid_size(args),
                xyz_grid_size=_match_filter_xyz_grid_size(args),
                max_per_image_cell=getattr(args, "match_filter_max_per_image_cell", 8),
                max_per_xyz_cell=getattr(args, "match_filter_max_per_xyz_cell", 8),
                min_matches=getattr(args, "match_filter_min_matches", 0),
            )
            p3d_valid = p3d[valid][dense_keep]
            query_valid = dense_matches.query_yx[valid][dense_keep]
            dense_pnp_scores = dense_matches.scores[valid_ids][dense_keep]
            dense_pose, dense_inliers = solve_pnp_ransac(
                p3d_valid.detach().cpu().numpy(),
                (
                    query_valid
                    + dense_query_offset
                ).detach().cpu().numpy(),
                K_full.detach().cpu().numpy(),
                reprojection_error=dense_reprojection_error,
                refine_reprojection_error=args.refine_reprojection_error,
                confidence=args.pnp_confidence,
                iterations=dense_pnp_iterations,
                min_iterations=dense_pnp_min_iterations,
                solver=args.solver,
                refine_poselib=refine_poselib,
                match_scores=dense_pnp_scores.detach().cpu().numpy(),
            )
        elif dense_matcher == "lightglue_rendered":
            full_h = int(rgb.shape[-2])
            full_w = int(rgb.shape[-1])
            if full_renderer is not None:
                full_render = render_hybrid_superpoint(
                    model,
                    sp_head,
                    full_renderer,
                    pose,
                    descriptor_source=descriptor_source,
                    ply_loc_feature_weight=ply_loc_feature_weight,
                    hybrid_residual_alpha_max=getattr(args, "hybrid_residual_alpha_max", 0.05),
                )
                dense_desc_map = full_render["descriptor"][0]
                dense_depth = full_render["depth"][0].float()
                dense_loc_map = full_render["locability"]
                rendered_detector = full_render["detector"][0]
                dense_K = K_full
                dense_renderer_K = full_renderer.K.float()
                query_lg_yx = keypoints if query_keypoints_are_full_res else keypoints * 8.0
                query_lg_map = F.normalize(upsample_feature_map(teacher_desc_raw, full_h, full_w), p=2, dim=0)
                query_lg_desc = sample_descriptors_bilinear(query_lg_map, query_lg_yx)
            else:
                dense_desc_map = desc_map
                dense_depth = render["depth"][0].float()
                dense_loc_map = loc_map
                rendered_detector = render["detector"][0]
                dense_K = K_feature
                dense_renderer_K = renderer.K.float()
                query_lg_yx = keypoints / 8.0 if query_keypoints_are_full_res else keypoints
                query_lg_desc = query_desc
            try:
                rendered_kpts = select_rendered_keypoints(
                    dense_desc_map,
                    source=args.rendered_keypoint_source,
                    max_keypoints=args.lightglue_max_keypoints,
                    threshold=args.dense_match_threshold,
                    nms_radius=args.nms_radius,
                    detector_logits=rendered_detector,
                    locability=dense_loc_map,
                )
            except ValueError:
                break
            if rendered_kpts.keypoints_yx.shape[0] < 4 or query_lg_yx.shape[0] < 4:
                break
            q_dense, rendered_ids, _lg_scores = match_lightglue_descriptors(
                query_lg_yx,
                query_lg_desc,
                rendered_kpts.keypoints_yx,
                rendered_kpts.descriptors,
                image_hw=tuple(int(v) for v in dense_depth.shape),
                rendered_hw=tuple(int(v) for v in dense_depth.shape),
                feature_name=lightglue_feature_name(args.dim_pipeline),
                filter_threshold=args.lightglue_filter_threshold,
            )
            if rendered_ids.numel() < 4:
                break
            refined_render_yx = rendered_kpts.keypoints_yx[rendered_ids]
            dense_points3d = unproject_positions_yx(
                refined_render_yx,
                dense_depth,
                pose[0].float(),
                dense_renderer_K,
                pixel_center_offset=args.render_pixel_center_offset,
            )
            valid = torch.isfinite(dense_points3d).all(dim=-1)
            if valid.sum() < 4:
                break
            valid_ids = torch.where(valid)[0]
            dense_keep = select_pnp_match_indices(
                query_lg_yx[q_dense][valid],
                dense_points3d[valid],
                scores=_lg_scores[valid_ids],
                reference_yx=refined_render_yx[valid],
                max_matches=_stage_match_filter_max_matches(args, "dense", getattr(args, "dense_pnp_max_matches", 0)),
                mode=_stage_match_filter_mode(args, "dense"),
                image_grid_size=_match_filter_image_grid_size(args),
                xyz_grid_size=_match_filter_xyz_grid_size(args),
                max_per_image_cell=getattr(args, "match_filter_max_per_image_cell", 8),
                max_per_xyz_cell=getattr(args, "match_filter_max_per_xyz_cell", 8),
                min_matches=getattr(args, "match_filter_min_matches", 0),
            )
            dense_points3d_valid = dense_points3d[valid][dense_keep]
            query_lg_valid = query_lg_yx[q_dense][valid][dense_keep]
            dense_pnp_scores = _lg_scores[valid_ids][dense_keep]
            dense_pose, dense_inliers = solve_pnp_ransac(
                dense_points3d_valid.detach().cpu().numpy(),
                (query_lg_valid + dense_query_offset).detach().cpu().numpy(),
                dense_K.detach().cpu().numpy(),
                reprojection_error=dense_reprojection_error,
                refine_reprojection_error=args.refine_reprojection_error,
                confidence=args.pnp_confidence,
                iterations=dense_pnp_iterations,
                min_iterations=dense_pnp_min_iterations,
                solver=args.solver,
                refine_poselib=refine_poselib,
                match_scores=dense_pnp_scores.detach().cpu().numpy(),
            )
        elif dense_matcher == "loftr_rendered":
            if full_renderer is None:
                break
            full_render = render_hybrid_superpoint(
                model,
                sp_head,
                full_renderer,
                pose,
                descriptor_source=descriptor_source,
                ply_loc_feature_weight=ply_loc_feature_weight,
                hybrid_residual_alpha_max=getattr(args, "hybrid_residual_alpha_max", 0.05),
                include_rgb=True,
            )
            dense_depth = full_render["depth"][0].float()
            dense_rgb = full_render.get("rgb")
            if dense_rgb is None:
                break
            dense_rgb = dense_rgb.float().clamp(0.0, 1.0)
            query_rgb_for_match = rgb.float().clamp(0.0, 1.0)
            if query_rgb_for_match.shape[-2:] != dense_rgb.shape[-2:]:
                break
            try:
                query_yx, rendered_yx, loftr_scores = match_loftr_images(
                    query_rgb_for_match,
                    dense_rgb,
                    pretrained=getattr(args, "loftr_pretrained", "outdoor"),
                    image_scale=getattr(args, "loftr_image_scale", 1.0),
                    min_confidence=getattr(args, "loftr_min_confidence", 0.0),
                    max_matches=getattr(args, "loftr_max_matches", 4096),
                )
            except (ImportError, RuntimeError, ValueError):
                break
            if rendered_yx.shape[0] < 4:
                break
            dense_points3d = unproject_positions_yx(
                rendered_yx,
                dense_depth,
                pose[0].float(),
                full_renderer.K.float(),
                pixel_center_offset=args.render_pixel_center_offset,
            )
            valid = (
                torch.isfinite(dense_points3d).all(dim=-1)
                & torch.isfinite(query_yx).all(dim=-1)
                & torch.isfinite(loftr_scores)
                & (dense_points3d[:, 2].abs() < 1e8)
            )
            if valid.sum() < 4:
                break
            valid_ids = torch.where(valid)[0]
            dense_keep = select_pnp_match_indices(
                query_yx[valid],
                dense_points3d[valid],
                scores=loftr_scores[valid_ids],
                reference_yx=rendered_yx[valid],
                max_matches=_stage_match_filter_max_matches(args, "dense", getattr(args, "dense_pnp_max_matches", 0)),
                mode=_stage_match_filter_mode(args, "dense"),
                image_grid_size=_match_filter_image_grid_size(args),
                xyz_grid_size=_match_filter_xyz_grid_size(args),
                max_per_image_cell=getattr(args, "match_filter_max_per_image_cell", 8),
                max_per_xyz_cell=getattr(args, "match_filter_max_per_xyz_cell", 8),
                min_matches=getattr(args, "match_filter_min_matches", 0),
            )
            dense_points3d_valid = dense_points3d[valid][dense_keep]
            query_valid = query_yx[valid][dense_keep]
            dense_pnp_scores = loftr_scores[valid_ids][dense_keep]
            dense_pose, dense_inliers = solve_pnp_ransac(
                dense_points3d_valid.detach().cpu().numpy(),
                (query_valid + dense_query_offset).detach().cpu().numpy(),
                K_full.detach().cpu().numpy(),
                reprojection_error=dense_reprojection_error,
                refine_reprojection_error=args.refine_reprojection_error,
                confidence=args.pnp_confidence,
                iterations=dense_pnp_iterations,
                min_iterations=dense_pnp_min_iterations,
                solver=args.solver,
                refine_poselib=refine_poselib,
                match_scores=dense_pnp_scores.detach().cpu().numpy(),
            )
        else:
            if args.dense_full_render and full_renderer is not None:
                full_render = render_hybrid_superpoint(
                    model,
                    sp_head,
                    full_renderer,
                    pose,
                    descriptor_source=descriptor_source,
                    ply_loc_feature_weight=ply_loc_feature_weight,
                    hybrid_residual_alpha_max=getattr(args, "hybrid_residual_alpha_max", 0.05),
                )
                dense_desc_map = full_render["descriptor"][0]
                dense_depth = full_render["depth"][0].float()
                dense_loc_map = full_render["locability"]
                dense_query_yx = keypoints * 8.0
                dense_K = K_full
                dense_renderer_K = full_renderer.K.float()
            else:
                dense_desc_map = desc_map
                dense_depth = render["depth"][0].float()
                dense_loc_map = loc_map
                dense_query_yx = keypoints / 8.0 if query_keypoints_are_full_res else keypoints
                dense_K = K_feature
                dense_renderer_K = renderer.K.float()
            rendered_flat = dense_desc_map.flatten(1).T
            prior = dense_loc_map[0, 0].flatten() if dense_loc_map is not None else None
            q_dense, pix_ids, _dense_scores = match_descriptors_topk(
                query_desc,
                rendered_flat,
                topk=1,
                threshold=args.dense_match_threshold,
                landmark_prior=prior,
                prior_weight=args.locability_prior_weight,
                second_best_margin=dense_margin,
            )
            if pix_ids.numel() < 4:
                break
            depth_flat = dense_depth.reshape(-1)
            valid = torch.isfinite(depth_flat[pix_ids]) & (depth_flat[pix_ids] > 0.0)
            if valid.sum() < 4:
                break
            q_dense = q_dense[valid]
            pix_ids = pix_ids[valid]
            if args.subpixel_refine:
                refined_render_yx = refine_rendered_positions_softargmax(
                    dense_desc_map,
                    query_desc[q_dense],
                    pix_ids,
                    window_radius=args.topk_refine_window,
                    temperature=args.subpixel_temperature,
                )
            else:
                height, width = dense_depth.shape
                refined_render_yx = torch.stack(
                    [(pix_ids // width).float(), (pix_ids % width).float()],
                    dim=-1,
                )
            dense_points3d = unproject_positions_yx(
                refined_render_yx,
                dense_depth,
                pose[0].float(),
                dense_renderer_K,
                pixel_center_offset=args.render_pixel_center_offset,
            )
            dense_filter_reliability = None
            if prior is not None and pix_ids.numel() > 0:
                dense_filter_reliability = prior[pix_ids]
            dense_filter_scores = _match_filter_scores(
                _dense_scores,
                dense_filter_reliability,
                getattr(args, "match_filter_calibrated_score_weight", 0.0),
            )
            dense_keep = select_pnp_match_indices(
                dense_query_yx[q_dense],
                dense_points3d,
                scores=dense_filter_scores,
                reference_yx=refined_render_yx,
                max_matches=_stage_match_filter_max_matches(args, "dense", getattr(args, "dense_pnp_max_matches", 0)),
                mode=_stage_match_filter_mode(args, "dense"),
                image_grid_size=_match_filter_image_grid_size(args),
                xyz_grid_size=_match_filter_xyz_grid_size(args),
                max_per_image_cell=getattr(args, "match_filter_max_per_image_cell", 8),
                max_per_xyz_cell=getattr(args, "match_filter_max_per_xyz_cell", 8),
                min_matches=getattr(args, "match_filter_min_matches", 0),
            )
            dense_filter_scores = dense_filter_scores[dense_keep]
            dense_points3d = dense_points3d[dense_keep]
            dense_query_for_pnp = dense_query_yx[q_dense][dense_keep]
            dense_pose, dense_inliers = solve_pnp_ransac(
                dense_points3d.detach().cpu().numpy(),
                (
                    dense_query_for_pnp
                    + dense_query_offset
                ).detach().cpu().numpy(),
                dense_K.detach().cpu().numpy(),
                reprojection_error=dense_reprojection_error,
                refine_reprojection_error=args.refine_reprojection_error,
                confidence=args.pnp_confidence,
                iterations=dense_pnp_iterations,
                min_iterations=dense_pnp_min_iterations,
                solver=args.solver,
                refine_poselib=refine_poselib,
                match_scores=dense_filter_scores.detach().cpu().numpy(),
            )
        if dense_pose is None:
            break
        candidate_pose = torch.from_numpy(dense_pose).to(device=rgb.device, dtype=torch.float32).unsqueeze(0)
        if bool(getattr(args, "dense_pose_delta_gate", False)):
            reject_dense, delta_stats = should_reject_dense_pose_by_delta(
                previous_pose,
                candidate_pose,
                max_trans_cm=float(getattr(args, "dense_pose_delta_max_trans_cm", 100.0)),
                max_rot_deg=float(getattr(args, "dense_pose_delta_max_rot_deg", 5.0)),
            )
            dense_rejection_stats = {**dense_rejection_stats, **delta_stats}
            if reject_dense:
                dense_rejections += 1
                dense_inliers = previous_dense_inliers
                break
        if bool(getattr(args, "dense_sparse_consistency_gate", False)):
            reject_dense, rejection_stats = should_reject_dense_pose_by_sparse_consistency(
                previous_pose,
                candidate_pose,
                landmark_xyz[lm_ids],
                sparse_query_for_pnp,
                sparse_K,
                threshold=float(sparse_reprojection_error),
                max_median_ratio=float(getattr(args, "dense_sparse_consistency_max_median_ratio", 1.5)),
                max_median_increase=float(getattr(args, "dense_sparse_consistency_max_median_increase", 2.0)),
                min_inlier_ratio_factor=float(
                    getattr(args, "dense_sparse_consistency_min_inlier_ratio_factor", 0.75)
                ),
            )
            if rejection_stats:
                dense_rejection_stats = rejection_stats
            if reject_dense:
                dense_rejections += 1
                dense_inliers = previous_dense_inliers
                break
        pose = candidate_pose

    out = {
        "pose_w2c": pose[0].detach().cpu().numpy(),
        "sparse_pose_w2c": sparse_pose,
        "sparse_inliers": sparse_inliers,
        "dense_inliers": dense_inliers,
        "dense_rejections": dense_rejections,
        "dense_rejection_stats": dense_rejection_stats,
    }
    if getattr(args, "matchability_diagnostics", False):
        out["sparse_match_diagnostics"] = {
            "query_yx": (keypoints[q_ids] + sparse_query_offset).detach().cpu().numpy(),
            "xyz": landmark_xyz[lm_ids].detach().cpu().numpy(),
            "scores": _scores.detach().cpu().numpy(),
            "margins": None
            if sparse_margin_by_query is None
            else sparse_margin_by_query.detach().cpu().numpy(),
            "coord_space": "full" if query_keypoints_are_full_res else "feature",
        }
    return out


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Cambridge hybrid Loc-GS localization")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", default="")
    parser.add_argument("--data_root", default="/mnt/pool/sqy/Cambridge_stdloc")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--eval_pose_source", choices=["cambridge", "cameras_json"], default="cambridge")
    parser.add_argument("--eval_split", choices=["train", "test"], default="test")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--query_offset", type=int, default=0)
    parser.add_argument("--query_stride", type=int, default=1)
    parser.add_argument("--landmark_source", choices=["rendered", "gaussian", "stdloc_detector"], default="stdloc_detector")
    parser.add_argument("--stdloc_detector_dir", default="")
    parser.add_argument("--landmark_candidate_source", choices=["sampled", "all_gaussians"], default="sampled")
    parser.add_argument("--landmark_selection", choices=["global", "per_view", "per_view_spatial"], default="global")
    parser.add_argument("--max_landmarks", type=int, default=200000)
    parser.add_argument("--landmark_ref_views", type=int, default=20)
    parser.add_argument("--landmark_stride", type=int, default=2)
    parser.add_argument("--landmark_per_view_quota", type=int, default=0)
    parser.add_argument("--landmark_view_grid_size", type=int, default=8)
    parser.add_argument(
        "--landmark_score_mode",
        choices=["legacy", "matchability", "matchability_prior"],
        default="legacy",
    )
    parser.add_argument("--landmark_score_ref_views", type=int, default=64)
    parser.add_argument("--landmark_score_detector_weight", type=float, default=1.0)
    parser.add_argument("--landmark_score_locability_weight", type=float, default=1.0)
    parser.add_argument("--landmark_score_visibility_weight", type=float, default=0.5)
    parser.add_argument("--landmark_score_geometry_weight", type=float, default=0.5)
    parser.add_argument("--landmark_score_alpha_weight", type=float, default=0.25)
    parser.add_argument("--landmark_score_depth_weight", type=float, default=0.5)
    parser.add_argument("--landmark_score_observability_weight", type=float, default=0.25)
    parser.add_argument("--landmark_score_distinctiveness_weight", type=float, default=0.0)
    parser.add_argument("--landmark_score_ambiguity_weight", type=float, default=0.0)
    parser.add_argument("--landmark_score_ambiguity_radius", type=float, default=0.25)
    parser.add_argument("--landmark_score_ambiguity_max_landmarks", type=int, default=32768)
    parser.add_argument("--landmark_score_keypoint_consensus_weight", type=float, default=0.0)
    parser.add_argument("--landmark_score_keypoint_consensus_radius", type=int, default=2)
    parser.add_argument("--landmark_score_keypoint_consensus_max_keypoints", type=int, default=1024)
    parser.add_argument("--landmark_score_keypoint_consensus_descriptor_weight", type=float, default=0.0)
    parser.add_argument("--landmark_score_legacy_keep_ratio", type=float, default=0.9)
    parser.add_argument("--landmark_score_spatial_grid_size", type=int, default=8)
    parser.add_argument("--landmark_score_prior_blend", type=float, default=0.25)
    parser.add_argument("--calibrated_matchability_path", default="")
    parser.add_argument("--landmark_score_calibrated_matchability_weight", type=float, default=0.0)
    parser.add_argument("--landmark_teacher_fusion_weight", type=float, default=0.0)
    parser.add_argument("--landmark_teacher_fusion_views", type=int, default=0)
    parser.add_argument("--landmark_teacher_fusion_chunk_size", type=int, default=2048)
    parser.add_argument("--landmark_teacher_fusion_geometry_power", type=float, default=0.0)
    parser.add_argument("--landmark_teacher_fusion_centrality_power", type=float, default=0.0)
    parser.add_argument("--landmark_teacher_fusion_rendered_views", type=int, default=0)
    parser.add_argument("--landmark_teacher_fusion_rendered_trans_noise_cm", type=float, default=0.0)
    parser.add_argument("--landmark_teacher_fusion_rendered_rot_noise_deg", type=float, default=0.0)
    parser.add_argument("--match_calibrated_prior_weight", type=float, default=0.0)
    parser.add_argument(
        "--descriptor_source",
        choices=["hybrid", "ply_loc", "hybrid_ply_blend", "hybrid_ply_gated_residual"],
        default="ply_loc",
    )
    parser.add_argument("--ply_loc_feature_weight", type=float, default=0.5)
    parser.add_argument("--hybrid_residual_alpha_max", type=float, default=0.05)
    parser.add_argument("--alpha_threshold", type=float, default=0.05)
    parser.add_argument("--query_keypoints", type=int, default=2048)
    parser.add_argument("--query_detector", choices=["superpoint", "stdloc"], default="stdloc")
    parser.add_argument("--query_feature_source", choices=["resized", "original"], default="original")
    parser.add_argument("--stdloc_detector_path", default="")
    parser.add_argument("--keypoint_threshold", type=float, default=0.015)
    parser.add_argument("--nms_radius", type=int, default=4)
    parser.add_argument("--sparse_match_threshold", type=float, default=0.0)
    parser.add_argument("--dense_match_threshold", type=float, default=0.0)
    parser.add_argument("--locability_prior_weight", type=float, default=0.05)
    parser.add_argument("--dense_iters", type=int, default=2)
    parser.add_argument("--matcher", choices=["stdloc_parity", "topk"], default="stdloc_parity")
    parser.add_argument("--sparse_matcher", choices=["", *SPARSE_MATCHERS], default="")
    parser.add_argument("--dense_matcher", choices=["", *DENSE_MATCHERS], default="")
    parser.add_argument("--dim_pipeline", choices=DIM_PIPELINES, default="superpoint+lightglue")
    parser.add_argument("--rendered_keypoint_source", choices=["locability", "detector", "projected_gaussian"], default="locability")
    parser.add_argument("--lightglue_max_keypoints", type=int, default=2048)
    parser.add_argument("--lightglue_filter_threshold", type=float, default=0.1)
    parser.add_argument("--loftr_pretrained", default="outdoor")
    parser.add_argument("--loftr_image_scale", type=float, default=1.0)
    parser.add_argument("--loftr_min_confidence", type=float, default=0.0)
    parser.add_argument("--loftr_max_matches", type=int, default=4096)
    parser.add_argument("--dense_sparse_consistency_gate", action="store_true")
    parser.add_argument("--dense_sparse_consistency_max_median_ratio", type=float, default=1.5)
    parser.add_argument("--dense_sparse_consistency_max_median_increase", type=float, default=2.0)
    parser.add_argument("--dense_sparse_consistency_min_inlier_ratio_factor", type=float, default=0.75)
    parser.add_argument("--dense_pose_delta_gate", action="store_true")
    parser.add_argument("--dense_pose_delta_max_trans_cm", type=float, default=100.0)
    parser.add_argument("--dense_pose_delta_max_rot_deg", type=float, default=5.0)
    parser.add_argument("--solver", choices=["poselib", "opencv", "opencv_prosac", "opencv_prosac_magsac"], default="opencv")
    parser.add_argument("--poselib_refine", action="store_true")
    parser.add_argument("--sparse_dual_softmax", action="store_true")
    parser.add_argument("--sparse_dual_softmax_temp", type=float, default=0.1)
    parser.add_argument("--dense_dual_softmax_temp", type=float, default=0.1)
    parser.add_argument("--fine_dual_softmax_temp", type=float, default=0.1)
    parser.add_argument("--mnn", action="store_true")
    parser.add_argument("--subpixel_refine", action="store_true")
    parser.add_argument("--subpixel_temperature", type=float, default=0.1)
    parser.add_argument("--topk_refine_window", type=int, default=1)
    parser.add_argument("--render_pixel_center_offset", type=float, default=0.0)
    parser.add_argument("--dense_query_pixel_center_offset", type=float, default=0.0)
    parser.add_argument("--dense_full_render", action="store_true")
    parser.add_argument("--superpoint_cache_root", default="")
    parser.add_argument("--disable_superpoint_cache", action="store_true")
    parser.add_argument("--match_second_best_margin", type=float, default=0.0)
    parser.add_argument("--sparse_match_second_best_margin", type=float, default=None)
    parser.add_argument("--dense_match_second_best_margin", type=float, default=None)
    parser.add_argument("--reprojection_error", type=float, default=2.0)
    parser.add_argument("--sparse_reprojection_error", type=float, default=None)
    parser.add_argument("--dense_reprojection_error", type=float, default=None)
    parser.add_argument("--refine_reprojection_error", type=float, default=0.0)
    parser.add_argument("--pnp_confidence", type=float, default=0.9999)
    parser.add_argument("--pnp_iterations", type=int, default=10000)
    parser.add_argument("--pnp_min_iterations", type=int, default=0)
    parser.add_argument("--sparse_pnp_iterations", type=int, default=None)
    parser.add_argument("--sparse_pnp_min_iterations", type=int, default=None)
    parser.add_argument("--dense_pnp_iterations", type=int, default=None)
    parser.add_argument("--dense_pnp_min_iterations", type=int, default=None)
    parser.add_argument(
        "--pnp_prefilter",
        choices=["none", "score", "image_grid", "xyz_grid", "image_xyz_grid", "local_geometry", "image_pair_geometry"],
        default="none",
    )
    parser.add_argument("--sparse_pnp_max_matches", type=int, default=0)
    parser.add_argument("--dense_pnp_max_matches", type=int, default=0)
    parser.add_argument("--pnp_prefilter_image_grid_size", type=int, default=8)
    parser.add_argument("--pnp_prefilter_xyz_grid_size", type=int, default=4)
    parser.add_argument(
        "--match_filter_mode",
        choices=["", "none", "score", "image_grid", "xyz_grid", "image_xyz_grid", "local_geometry", "image_pair_geometry", "calibrated_coverage"],
        default="",
    )
    parser.add_argument("--match_filter_calibrated_score_weight", type=float, default=0.0)
    parser.add_argument("--match_filter_margin_weight", type=float, default=0.0)
    parser.add_argument("--match_filter_top_m", type=int, default=0)
    parser.add_argument("--match_filter_image_grid_size", type=int, default=8)
    parser.add_argument("--match_filter_xyz_grid_size", type=int, default=8)
    parser.add_argument(
        "--sparse_match_filter_mode",
        choices=["", "none", "score", "image_grid", "xyz_grid", "image_xyz_grid", "local_geometry", "image_pair_geometry", "calibrated_coverage"],
        default="",
    )
    parser.add_argument(
        "--dense_match_filter_mode",
        choices=["", "none", "score", "image_grid", "xyz_grid", "image_xyz_grid", "local_geometry", "image_pair_geometry", "calibrated_coverage"],
        default="",
    )
    parser.add_argument("--sparse_match_filter_top_m", type=int, default=0)
    parser.add_argument("--dense_match_filter_top_m", type=int, default=0)
    parser.add_argument("--match_filter_max_per_image_cell", type=int, default=8)
    parser.add_argument("--match_filter_max_per_xyz_cell", type=int, default=8)
    parser.add_argument("--match_filter_min_matches", type=int, default=0)
    parser.add_argument("--pnp_hypotheses", type=int, default=1)
    parser.add_argument(
        "--pnp_cluster_mode",
        choices=["none", "xyz_voxel", "image_grid", "image_xyz_grid"],
        default="none",
    )
    parser.add_argument("--pnp_cluster_grid_size", type=int, default=4)
    parser.add_argument("--pnp_dense_verify_topk", type=int, default=1)
    parser.add_argument("--pnp_hypothesis_min_score_gain", type=float, default=0.0)
    parser.add_argument("--matchability_diagnostics", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser


def _summary(errors_te: list[float], errors_ae: list[float], inliers: list[int]) -> dict[str, float]:
    return pose_error_summary(errors_te, errors_ae, inliers)


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = build_argparser().parse_args() if args is None else args
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt["args"]
    scene = args.scene or train_args["scene"]
    scene_root = Path(args.data_root) / scene
    output_dir = Path(args.output_dir) if args.output_dir else Path(train_args["output_dir"]) / "eval_cambridge"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_intr = train_args["feature_intrinsics"]
    full_fx = feature_intr["fx"] * 8.0
    full_fy = feature_intr["fy"] * 8.0
    full_cx = feature_intr["cx"] * 8.0
    full_cy = feature_intr["cy"] * 8.0
    if args.query_offset < 0:
        raise ValueError("--query_offset must be non-negative")
    if args.query_stride < 1:
        raise ValueError("--query_stride must be at least 1")
    need_query_subset = args.query_offset > 0 or args.query_stride > 1
    eval_split = getattr(args, "eval_split", "test")
    eval_cameras_json = train_args["cameras_json"] if args.eval_pose_source == "cameras_json" else None
    dataset = CambridgeHybridDataset(
        scene_root=scene_root,
        cameras_json=eval_cameras_json,
        split=eval_split,
        image_subdir=train_args.get("image_subdir", "processed"),
        image_height=train_args["image_height"],
        image_width=train_args["image_width"],
        fx=full_fx,
        fy=full_fy,
        cx=full_cx,
        cy=full_cy,
        max_frames=0 if need_query_subset else args.max_queries,
    )
    if need_query_subset:
        indices = list(range(args.query_offset, len(dataset), args.query_stride))
        if args.max_queries > 0:
            indices = indices[: args.max_queries]
        dataset = Subset(dataset, indices)

    from loc_gs.rendering.feature_renderer import FeatureFieldRenderer

    renderer = FeatureFieldRenderer(
        image_height=train_args["feature_height"],
        image_width=train_args["feature_width"],
        fx=feature_intr["fx"],
        fy=feature_intr["fy"],
        cx=feature_intr["cx"],
        cy=feature_intr["cy"],
    ).to(device)
    _, dense_matcher_name = resolve_matchers(args)
    full_renderer = None
    if (
        args.dense_full_render
        or dense_matcher_name in {"lightglue_rendered", "loftr_rendered"}
        or int(getattr(args, "landmark_teacher_fusion_rendered_views", 0)) > 0
    ):
        use_stdloc_render = dense_matcher_name == "stdloc_parity"
        full_renderer = FeatureFieldRenderer(
            image_height=train_args["image_height"],
            image_width=train_args["image_width"],
            fx=full_fx,
            fy=full_fy,
            cx=full_cx,
            cy=full_cy,
            max_channels_per_chunk=32,
            far_plane=10000.0 if use_stdloc_render else 100.0,
            packed=not use_stdloc_render,
            rasterize_mode="antialiased" if use_stdloc_render else "classic",
        ).to(device)
    model = HybridFeatureGaussian(
        latent_dim=train_args["latent_dim"],
        hash_output_dim=train_args["hash_output_dim"],
        fine_dim=train_args["fine_dim"],
        coarse_dim=train_args["coarse_dim"],
        output_dim=train_args["hybrid_output_dim"],
    )
    model.load_from_ply(train_args["ply_path"])
    model = model.to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    sp_head = SuperPointOutputHead(
        fused_dim=train_args["hybrid_output_dim"],
        descriptor_dim=256,
        detector_dim=65,
        hidden_dim=256,
        num_res_blocks=2,
        use_3x3=True,
    ).to(device)
    sp_head.load_state_dict(ckpt["sp_head_state_dict"])
    sp_head.eval()
    teacher = SuperPointNet().to(device)
    teacher.load_state_dict(torch.load(train_args["superpoint_weights"], map_location=device), strict=False)
    teacher.eval()
    stdloc_detector = None
    if args.query_detector == "stdloc":
        detector_path = (
            Path(args.stdloc_detector_path)
            if args.stdloc_detector_path
            else Path(train_args["ply_path"]).parents[2] / "detector" / "30000_detector.pth"
        )
        if not detector_path.exists():
            raise FileNotFoundError(f"STDLoc query detector not found: {detector_path}")
        stdloc_detector = StdlocKeypointDetector(in_dim=256).to(device)
        stdloc_detector.load_state_dict(torch.load(detector_path, map_location=device))
        stdloc_detector.eval()
    teacher_cache = None
    if not args.disable_superpoint_cache:
        cache_split = eval_split if args.query_feature_source == "resized" else f"{eval_split}_{args.query_feature_source}"
        teacher_cache = SuperPointTeacherCache(
            args.superpoint_cache_root or train_args.get("output_root", "output"),
            scene=scene,
            split=cache_split,
        )
    calibrated_matchability = None
    if args.calibrated_matchability_path:
        calibration_path = Path(args.calibrated_matchability_path)
        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibrated matchability cache not found: {calibration_path}")
        calibrated_matchability = _load_calibrated_matchability(calibration_path, device)

    landmark_score_stats: dict[str, float] = {}
    landmark_score_weights = _score_weights(args, rendered=args.landmark_source == "rendered")
    score_ref_poses = None
    score_ref_keypoint_maps = None
    score_ref_keypoint_descriptor_maps = None
    teacher_fusion_poses = None
    teacher_fusion_descriptor_maps = None
    teacher_fusion_weight = float(getattr(args, "landmark_teacher_fusion_weight", 0.0))
    teacher_fusion_views = int(getattr(args, "landmark_teacher_fusion_views", 0))
    if teacher_fusion_views <= 0:
        teacher_fusion_views = int(args.landmark_score_ref_views)
    need_score_refs = (
        (landmark_score_weights and int(args.landmark_score_ref_views) > 0)
        or (teacher_fusion_weight > 0.0 and teacher_fusion_views > 0)
    )
    if need_score_refs:
        score_ref_dataset = CambridgeHybridDataset(
            scene_root=scene_root,
            cameras_json=train_args["cameras_json"],
            split="train",
            image_subdir=train_args.get("image_subdir", "processed"),
            image_height=train_args["image_height"],
            image_width=train_args["image_width"],
            max_frames=0,
        )
        if landmark_score_weights and int(args.landmark_score_ref_views) > 0:
            score_ref_poses = _score_ref_poses(score_ref_dataset, device, args.landmark_score_ref_views)
        if landmark_score_weights.get("keypoint_consensus", 0.0) > 0.0:
            score_ref_keypoint_maps, score_ref_keypoint_descriptor_maps = _score_ref_keypoint_maps(
                score_ref_dataset,
                teacher,
                device,
                max_views=args.landmark_score_ref_views,
                height=train_args["feature_height"],
                width=train_args["feature_width"],
                max_keypoints=args.landmark_score_keypoint_consensus_max_keypoints,
                threshold=args.keypoint_threshold,
                nms_radius=args.nms_radius,
                include_descriptors=args.landmark_score_keypoint_consensus_descriptor_weight > 0.0,
            )
        if teacher_fusion_weight > 0.0 and teacher_fusion_views > 0:
            teacher_fusion_descriptor_maps = _score_ref_descriptor_maps(
                score_ref_dataset,
                teacher,
                device,
                max_views=teacher_fusion_views,
                height=train_args["feature_height"],
                width=train_args["feature_width"],
            )
            teacher_fusion_poses = _score_ref_poses(score_ref_dataset, device, teacher_fusion_views)
        rendered_fusion_views = int(getattr(args, "landmark_teacher_fusion_rendered_views", 0))
        if teacher_fusion_weight > 0.0 and rendered_fusion_views > 0:
            rendered_base_poses = _score_ref_poses(score_ref_dataset, device, rendered_fusion_views)
            rendered_poses = perturb_reference_poses_camera_frame(
                rendered_base_poses,
                translation_m=float(getattr(args, "landmark_teacher_fusion_rendered_trans_noise_cm", 0.0)) / 100.0,
                rotation_deg=float(getattr(args, "landmark_teacher_fusion_rendered_rot_noise_deg", 0.0)),
            )
            rendered_descriptor_maps = _rendered_ref_descriptor_maps(
                model,
                full_renderer,
                rendered_poses,
                teacher,
                device,
                height=train_args["feature_height"],
                width=train_args["feature_width"],
            )
            if rendered_descriptor_maps is not None:
                if teacher_fusion_descriptor_maps is None:
                    teacher_fusion_descriptor_maps = rendered_descriptor_maps
                    teacher_fusion_poses = rendered_poses
                else:
                    teacher_fusion_descriptor_maps = torch.cat(
                        [teacher_fusion_descriptor_maps, rendered_descriptor_maps],
                        dim=0,
                    )
                    teacher_fusion_poses = torch.cat([teacher_fusion_poses, rendered_poses], dim=0)

    landmark_filter_reliability = None
    if args.landmark_source == "rendered":
        train_ref_dataset = CambridgeHybridDataset(
            scene_root=scene_root,
            cameras_json=train_args["cameras_json"],
            split="train",
            image_subdir=train_args.get("image_subdir", "processed"),
            image_height=train_args["image_height"],
            image_width=train_args["image_width"],
            max_frames=args.landmark_ref_views,
        )
        print(
            f"[eval] rendering landmark bank from {len(train_ref_dataset)} reference views "
            f"(max_landmarks={args.max_landmarks})"
        )
        landmark_xyz, landmark_desc, landmark_prior, landmark_score_stats = build_rendered_landmark_bank(
            model,
            sp_head,
            renderer,
            train_ref_dataset,
            max_landmarks=args.max_landmarks,
            ref_views=args.landmark_ref_views,
            stride=args.landmark_stride,
            alpha_threshold=args.alpha_threshold,
            device=device,
            selection=args.landmark_selection,
            per_view_quota=args.landmark_per_view_quota,
            view_grid_size=args.landmark_view_grid_size,
            descriptor_source=args.descriptor_source,
            ply_loc_feature_weight=args.ply_loc_feature_weight,
            hybrid_residual_alpha_max=getattr(args, "hybrid_residual_alpha_max", 0.05),
            score_weights=landmark_score_weights,
            score_ref_poses=score_ref_poses,
        )
    elif args.landmark_source == "gaussian":
        print(f"[eval] decoding up to {args.max_landmarks} Gaussian-center landmark descriptors")
        landmark_xyz, landmark_desc, landmark_prior = decode_landmark_descriptors(
            model,
            sp_head,
            max_landmarks=args.max_landmarks,
        )
    else:
        detector_dir = (
            Path(args.stdloc_detector_dir)
            if args.stdloc_detector_dir
            else Path(train_args["ply_path"]).parents[2] / "detector"
        )
        print(f"[eval] loading STDLoc detector landmarks from {detector_dir}")
        (
            landmark_xyz,
            landmark_desc,
            landmark_prior,
            landmark_score_stats,
            landmark_aux,
        ) = build_stdloc_detector_landmark_bank(
            model,
            sp_head,
            detector_dir=detector_dir,
            max_landmarks=args.max_landmarks,
            descriptor_source=args.descriptor_source,
            ply_loc_feature_weight=args.ply_loc_feature_weight,
            hybrid_residual_alpha_max=getattr(args, "hybrid_residual_alpha_max", 0.05),
            device=device,
            candidate_source=args.landmark_candidate_source,
            score_weights=landmark_score_weights,
            score_ref_poses=score_ref_poses,
            score_K=renderer.K.float(),
            score_height=train_args["feature_height"],
            score_width=train_args["feature_width"],
            legacy_keep_ratio=args.landmark_score_legacy_keep_ratio,
            spatial_grid_size=args.landmark_score_spatial_grid_size,
            score_prior_blend=args.landmark_score_prior_blend,
            select_by_score=args.landmark_score_mode == "matchability",
            ambiguity_radius=args.landmark_score_ambiguity_radius,
            ambiguity_max_landmarks=args.landmark_score_ambiguity_max_landmarks,
            keypoint_consensus_maps=score_ref_keypoint_maps,
            keypoint_consensus_radius=args.landmark_score_keypoint_consensus_radius,
            keypoint_consensus_descriptor_maps=score_ref_keypoint_descriptor_maps,
            keypoint_consensus_descriptor_weight=args.landmark_score_keypoint_consensus_descriptor_weight,
            teacher_fusion_poses=teacher_fusion_poses,
            teacher_fusion_descriptor_maps=teacher_fusion_descriptor_maps,
            teacher_fusion_K=renderer.K.float(),
            teacher_fusion_height=train_args["feature_height"],
            teacher_fusion_width=train_args["feature_width"],
            teacher_fusion_weight=args.landmark_teacher_fusion_weight,
            teacher_fusion_chunk_size=args.landmark_teacher_fusion_chunk_size,
            teacher_fusion_geometry_power=args.landmark_teacher_fusion_geometry_power,
            teacher_fusion_centrality_power=args.landmark_teacher_fusion_centrality_power,
            calibrated_matchability=calibrated_matchability,
            calibrated_matchability_weight=args.landmark_score_calibrated_matchability_weight,
            return_aux=True,
        )
        landmark_filter_reliability = landmark_aux.get("calibrated_matchability")

    sparse_te: list[float] = []
    sparse_ae: list[float] = []
    sparse_inliers: list[int] = []
    dense_te: list[float] = []
    dense_ae: list[float] = []
    dense_inliers: list[int] = []
    matchability_rows: list[dict[str, float]] = []
    details = []

    for item in tqdm(dataset, desc="Evaluating Cambridge hybrid", dynamic_ncols=True):
        rgb = item["rgb"].unsqueeze(0).to(device)
        teacher_rgb = None
        if args.query_feature_source == "original":
            teacher_rgb = load_cambridge_rgb_no_resize(
                scene_root,
                train_args.get("image_subdir", "processed"),
                str(item["image_name"]),
                device,
            ).unsqueeze(0)
        gt_pose = item["pose_w2c"].numpy()
        K_feature = item["feature_K"].to(device)
        result = localize_one(
            model,
            sp_head,
            renderer,
            teacher,
            rgb,
            K_feature,
            item["K"].to(device),
            landmark_xyz,
            landmark_desc,
            landmark_prior,
            args,
            image_name=item["image_name"],
            teacher_cache=teacher_cache,
            full_renderer=full_renderer,
            stdloc_detector=stdloc_detector,
            teacher_rgb=teacher_rgb,
            landmark_filter_reliability=landmark_filter_reliability,
        )
        matchability_i = None
        sparse_diag = result.pop("sparse_match_diagnostics", None)
        if sparse_diag is not None:
            diag_K_tensor = item["K"] if sparse_diag["coord_space"] == "full" else K_feature
            matchability_i = sparse_matchability_metrics(
                sparse_diag["query_yx"],
                sparse_diag["xyz"],
                gt_pose,
                diag_K_tensor.detach().cpu().numpy(),
                scores=sparse_diag.get("scores"),
                margins=sparse_diag.get("margins"),
            )
            matchability_rows.append(matchability_i)
        sparse_te_i = None
        sparse_ae_i = None
        dense_te_i = None
        dense_ae_i = None
        if result["sparse_pose_w2c"] is not None:
            te, ae = pose_error_cm_deg(result["sparse_pose_w2c"], gt_pose)
            sparse_te_i = float(te)
            sparse_ae_i = float(ae)
            sparse_te.append(te)
            sparse_ae.append(ae)
            sparse_inliers.append(int(result["sparse_inliers"]))
        if result["pose_w2c"] is not None:
            te, ae = pose_error_cm_deg(result["pose_w2c"], gt_pose)
            dense_te_i = float(te)
            dense_ae_i = float(ae)
            dense_te.append(te)
            dense_ae.append(ae)
            dense_inliers.append(int(result["dense_inliers"]))
        details.append(
            {
                "image_name": item["image_name"],
                "sparse_te": sparse_te_i,
                "sparse_ae": sparse_ae_i,
                "sparse_inliers": int(result["sparse_inliers"]),
                "dense_te": dense_te_i,
                "dense_ae": dense_ae_i,
                "dense_inliers": int(result["dense_inliers"]),
                "dense_rejections": int(result.get("dense_rejections", 0)),
                "dense_rejection_stats": result.get("dense_rejection_stats", {}),
                "localized": result["pose_w2c"] is not None,
                "matchability": matchability_i,
            }
        )

    eval_config = effective_eval_config(args)
    summary = {
        "checkpoint": str(args.checkpoint),
        "scene": scene,
        "eval_pose_source": args.eval_pose_source,
        "eval_split": eval_split,
        "query_offset": int(args.query_offset),
        "query_stride": int(args.query_stride),
        "eval_config": eval_config,
        "landmark_source": args.landmark_source,
        "landmark_candidate_source": args.landmark_candidate_source,
        "landmark_selection": args.landmark_selection,
        "landmark_score_mode": args.landmark_score_mode,
        "landmark_score_weights": landmark_score_weights,
        "landmark_score_legacy_keep_ratio": float(args.landmark_score_legacy_keep_ratio),
        "landmark_score_spatial_grid_size": int(args.landmark_score_spatial_grid_size),
        "landmark_score_prior_blend": float(args.landmark_score_prior_blend),
        "landmark_score_stats": landmark_score_stats,
        "calibrated_matchability_path": str(args.calibrated_matchability_path),
        "landmark_score_calibrated_matchability_weight": float(args.landmark_score_calibrated_matchability_weight),
        "match_calibrated_prior_weight": float(args.match_calibrated_prior_weight),
        "query_detector": args.query_detector,
        "query_feature_source": args.query_feature_source,
        "descriptor_source": args.descriptor_source,
        "ply_loc_feature_weight": float(args.ply_loc_feature_weight),
        "hybrid_residual_alpha_max": float(args.hybrid_residual_alpha_max),
        "sparse_matcher": resolve_matchers(args)[0],
        "dense_matcher": resolve_matchers(args)[1],
        "dim_pipeline": args.dim_pipeline,
        "rendered_keypoint_source": args.rendered_keypoint_source,
        "match_filter_margin_weight": float(args.match_filter_margin_weight),
        "pnp_hypothesis_min_score_gain": float(args.pnp_hypothesis_min_score_gain),
        "landmarks": int(landmark_xyz.shape[0]),
        "sparse": _summary(sparse_te, sparse_ae, sparse_inliers),
        "dense": _summary(dense_te, dense_ae, dense_inliers),
        "matchability": _mean_metric_dict(matchability_rows),
        "localized": len(dense_te),
        "queries": len(dataset),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "results.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
