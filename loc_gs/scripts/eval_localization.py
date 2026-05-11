#!/usr/bin/env python3
"""
Downstream localization evaluation for SuperPoint hybrid feature field.

Pipeline for each test frame:
  1. Add controlled noise to GT pose  → perturbed pose
  2. Render SuperPoint features (desc + det + depth) from perturbed pose
  3. Extract keypoints from GT detector heatmap → GT keypoint locations
  4. Sample GT descriptors at those keypoint locations
  5. Dense-match GT descriptors against rendered descriptor map
  6. Unproject matched 2D positions using rendered depth + perturbed pose → 3D world coords
  7. PnP+RANSAC(GT 2D keypoints, 3D world points) → estimated pose
  8. Compare estimated pose to GT pose → rotation & translation error

Noise levels:
  small:  ±2cm  trans, ±1°  rot
  medium: ±5cm  trans, ±3°  rot
  large:  ±10cm trans, ±5°  rot
  xlarge: ±20cm trans, ±10° rot

Usage:
    python -m loc_gs.scripts.eval_localization \
        --config configs/superpoint_hybrid_room_0_v3.yaml \
        --checkpoint output/sp_gs/room0_hybrid_v3/checkpoints/best.pth \
        --output_dir output/sp_gs/room0_hybrid_v3/localization \
        --num_samples 100 \
        --device cuda:0
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loc_gs.config import load_config
from loc_gs.scripts.train_feature_field import LocGSTrainer


def _cv2():
    import cv2

    return cv2


# ===================================================================
# Noise levels
# ===================================================================

NOISE_LEVELS = {
    "small":  {"trans_m": 0.02, "rot_deg": 1.0},
    "medium": {"trans_m": 0.05, "rot_deg": 3.0},
    "large":  {"trans_m": 0.10, "rot_deg": 5.0},
    "xlarge": {"trans_m": 0.20, "rot_deg": 10.0},
}


# ===================================================================
# Pose perturbation
# ===================================================================

def random_rotation_matrix(angle_deg: float) -> np.ndarray:
    """Generate a random rotation matrix with rotation angle sampled
    uniformly from [-angle_deg, angle_deg] around a random axis."""
    angle_rad = np.deg2rad(angle_deg)
    # Random axis (unit vector on sphere)
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    # Random angle
    theta = np.random.uniform(-angle_rad, angle_rad)
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def perturb_pose(pose_w2c: np.ndarray, trans_m: float, rot_deg: float) -> np.ndarray:
    """Add controlled noise to a world-to-camera pose.

    Translation noise is applied in world frame.
    Rotation noise is applied as a random rotation.

    Args:
        pose_w2c: [4, 4] world-to-camera matrix
        trans_m: max translation perturbation in meters (per axis)
        rot_deg: max rotation perturbation in degrees

    Returns:
        perturbed_w2c: [4, 4] perturbed world-to-camera matrix
    """
    # Decompose w2c into c2w
    R_w2c = pose_w2c[:3, :3]
    t_w2c = pose_w2c[:3, 3]

    # c2w
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c  # camera position in world frame

    # Perturb camera position in world frame
    dt = np.random.uniform(-trans_m, trans_m, size=3)
    t_c2w_noisy = t_c2w + dt

    # Perturb rotation
    dR = random_rotation_matrix(rot_deg)
    R_c2w_noisy = dR @ R_c2w

    # Back to w2c
    R_w2c_noisy = R_c2w_noisy.T
    t_w2c_noisy = -R_w2c_noisy @ t_c2w_noisy

    perturbed = np.eye(4, dtype=pose_w2c.dtype)
    perturbed[:3, :3] = R_w2c_noisy
    perturbed[:3, 3] = t_w2c_noisy
    return perturbed


# ===================================================================
# Keypoint extraction from detector heatmap
# ===================================================================

def extract_keypoints_from_detector(
    detector_logits: torch.Tensor,
    confidence_threshold: float = 0.015,
    nms_radius: int = 2,
    max_keypoints: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract keypoints from 65-channel detector logits.

    The 65 channels represent an 8x8 sub-pixel grid + 1 dustbin channel.
    We apply softmax, pixel-shuffle to get a full-resolution probability map,
    then NMS + threshold.

    Args:
        detector_logits: [65, Hc, Wc] raw detector logits (Hc=60, Wc=80)
        confidence_threshold: min keypoint probability
        nms_radius: radius for non-maximum suppression
        max_keypoints: max keypoints to return

    Returns:
        keypoints: [K, 2] keypoint positions in feature-map coordinates (y, x)
                   at coarse resolution (Hc, Wc)
        scores: [K] confidence scores
    """
    Hc, Wc = detector_logits.shape[1], detector_logits.shape[2]

    # Softmax over 65 channels
    probs = F.softmax(detector_logits, dim=0)  # [65, Hc, Wc]

    # Remove dustbin (channel 64), keep 64 channels
    cell_probs = probs[:64]  # [64, Hc, Wc]

    # Pixel shuffle: [64, Hc, Wc] → [1, 8*Hc, 8*Wc]
    cell_probs = cell_probs.unsqueeze(0)  # [1, 64, Hc, Wc]
    heatmap = F.pixel_shuffle(cell_probs, 8)  # [1, 1, 8*Hc, 8*Wc]
    heatmap = heatmap.squeeze(0).squeeze(0)  # [H_full, W_full] = [480, 640]

    H_full, W_full = heatmap.shape

    # Simple NMS: max pool + compare
    if nms_radius > 0:
        kernel = 2 * nms_radius + 1
        heatmap_pad = heatmap.unsqueeze(0).unsqueeze(0)
        max_pool = F.max_pool2d(heatmap_pad, kernel_size=kernel, stride=1, padding=nms_radius)
        is_max = (heatmap_pad == max_pool).squeeze(0).squeeze(0)
        heatmap = heatmap * is_max.float()

    # Threshold
    mask = heatmap > confidence_threshold
    if mask.sum() == 0:
        # Fallback: take top-k
        k = min(max_keypoints, heatmap.numel())
        topk_vals, topk_idx = heatmap.flatten().topk(k)
        ys = topk_idx // W_full
        xs = topk_idx % W_full
        scores = topk_vals
    else:
        ys, xs = torch.where(mask)
        scores = heatmap[ys, xs]

    # Sort by score descending, limit to max_keypoints
    order = scores.argsort(descending=True)[:max_keypoints]
    ys = ys[order]
    xs = xs[order]
    scores = scores[order]

    # Convert full-resolution (480, 640) positions back to coarse feature-map coords (60, 80)
    # Since feature map is 1/8 of full resolution:
    ys_coarse = ys.float() / 8.0
    xs_coarse = xs.float() / 8.0

    keypoints = torch.stack([ys_coarse, xs_coarse], dim=1)  # [K, 2] in (y, x)
    return keypoints, scores


# ===================================================================
# Descriptor sampling
# ===================================================================

def sample_descriptors_bilinear(
    descriptor_map: torch.Tensor,
    keypoints: torch.Tensor,
) -> torch.Tensor:
    """Sample descriptors at sub-pixel locations using bilinear interpolation.

    Args:
        descriptor_map: [C, H, W] L2-normalized descriptor map
        keypoints: [K, 2] positions in (y, x) at feature-map resolution

    Returns:
        descriptors: [K, C] sampled and L2-normalized descriptors
    """
    C, H, W = descriptor_map.shape
    K = keypoints.shape[0]

    if K == 0:
        return torch.zeros(0, C, device=descriptor_map.device)

    # grid_sample expects coordinates in [-1, 1]
    # keypoints are in (y, x) at feature-map resolution
    y = keypoints[:, 0]
    x = keypoints[:, 1]

    # Normalize to [-1, 1]
    x_norm = 2.0 * x / (W - 1) - 1.0
    y_norm = 2.0 * y / (H - 1) - 1.0

    # grid_sample expects grid of shape [N, Hout, Wout, 2] with (x, y) order
    grid = torch.stack([x_norm, y_norm], dim=-1)  # [K, 2]
    grid = grid.unsqueeze(0).unsqueeze(2)  # [1, K, 1, 2]

    desc_map = descriptor_map.unsqueeze(0)  # [1, C, H, W]
    sampled = F.grid_sample(
        desc_map, grid, mode="bilinear", padding_mode="border", align_corners=True
    )  # [1, C, K, 1]
    sampled = sampled.squeeze(0).squeeze(-1).T  # [K, C]

    # Re-normalize
    sampled = F.normalize(sampled, p=2, dim=1)
    return sampled


# ===================================================================
# Dense matching
# ===================================================================

def match_descriptors_dense(
    query_descs: torch.Tensor,
    rendered_desc_map: torch.Tensor,
    ratio_threshold: float = 0.9,
    locability_map: Optional[torch.Tensor] = None,
    locability_weight: float = 0.0,
    subpixel_refine: bool = True,
    subpixel_window_radius: int = 2,
    subpixel_temperature: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match query descriptors against a rendered descriptor map.

    Args:
        query_descs: [K, C] L2-normalized query descriptors
        rendered_desc_map: [C, H, W] L2-normalized rendered descriptor map
        ratio_threshold: Lowe's ratio test threshold (1.0 = no filtering)

    Returns:
        matched_kp_indices: [M] indices into query keypoints that have valid matches
        matched_positions: [M, 2] matched positions in rendered map (y, x) at coarse res
        match_scores: [M] cosine similarity scores of matches
    """
    K, C = query_descs.shape
    _, H, W = rendered_desc_map.shape

    if K == 0:
        return (
            torch.zeros(0, dtype=torch.long, device=query_descs.device),
            torch.zeros(0, 2, device=query_descs.device),
            torch.zeros(0, device=query_descs.device),
        )

    # Flatten rendered map: [C, H*W]
    rendered_flat = rendered_desc_map.reshape(C, -1)  # [C, H*W]

    # Cosine similarity: [K, H*W]
    sim = query_descs @ rendered_flat  # [K, H*W]
    if locability_map is not None and locability_weight != 0.0:
        prior = locability_map.reshape(-1).to(device=sim.device, dtype=sim.dtype)
        sim = sim + locability_weight * prior.unsqueeze(0)

    # Find top-2 matches for ratio test
    topk = sim.topk(2, dim=1)
    best_scores = topk.values[:, 0]     # [K]
    second_scores = topk.values[:, 1]   # [K]
    best_indices = topk.indices[:, 0]   # [K]

    # Ratio test
    ratio = second_scores / (best_scores + 1e-8)
    valid = ratio < ratio_threshold

    matched_kp_indices = torch.where(valid)[0]
    match_scores = best_scores[valid]
    if subpixel_refine and matched_kp_indices.numel() > 0:
        matched_positions = _subpixel_refine_matches(
            sim=sim,
            query_indices=matched_kp_indices,
            best_indices=best_indices[valid],
            height=H,
            width=W,
            window_radius=subpixel_window_radius,
            temperature=subpixel_temperature,
        )
    else:
        matched_y = best_indices[valid].float() // W
        matched_x = best_indices[valid].float() % W
        matched_positions = torch.stack([matched_y, matched_x], dim=1)

    return matched_kp_indices, matched_positions, match_scores


def _subpixel_refine_matches(
    sim: torch.Tensor,
    query_indices: torch.Tensor,
    best_indices: torch.Tensor,
    height: int,
    width: int,
    window_radius: int,
    temperature: float,
) -> torch.Tensor:
    """Refine coarse dense matches with a local soft-argmax."""
    sim_map = sim.view(sim.shape[0], height, width)
    radius = max(int(window_radius), 0)
    temp = max(float(temperature), 1e-6)
    refined = []
    for q_idx, flat_idx in zip(query_indices.tolist(), best_indices.tolist()):
        yc = int(flat_idx // width)
        xc = int(flat_idx % width)
        y0 = max(0, yc - radius)
        y1 = min(height, yc + radius + 1)
        x0 = max(0, xc - radius)
        x1 = min(width, xc + radius + 1)
        patch = sim_map[q_idx, y0:y1, x0:x1]
        patch_flat = patch.reshape(-1)
        weights = F.softmax((patch_flat - patch_flat.max()) / temp, dim=0)
        yy, xx = torch.meshgrid(
            torch.arange(y0, y1, device=sim.device, dtype=sim.dtype),
            torch.arange(x0, x1, device=sim.device, dtype=sim.dtype),
            indexing="ij",
        )
        y = (weights * yy.reshape(-1)).sum()
        x = (weights * xx.reshape(-1)).sum()
        refined.append(torch.stack([y, x]))
    return torch.stack(refined, dim=0)


# ===================================================================
# Unproject to world coordinates
# ===================================================================

def unproject_to_world(
    positions_yx: torch.Tensor,
    depth_map: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """Unproject 2D positions to 3D world coordinates using depth and pose.

    Args:
        positions_yx: [M, 2] (y, x) positions at feature-map resolution
        depth_map: [H, W] z-depth map from rendering
        pose_w2c: [4, 4] world-to-camera matrix (the pose used for rendering)
        K: [3, 3] camera intrinsics (at feature-map resolution)

    Returns:
        points_3d: [M, 3] world-space 3D coordinates
    """
    M = positions_yx.shape[0]
    if M == 0:
        return torch.zeros(0, 3, device=positions_yx.device)

    device = positions_yx.device
    H, W = depth_map.shape

    y = positions_yx[:, 0]  # [M]
    x = positions_yx[:, 1]  # [M]

    # Sample depth at matched positions (bilinear interpolation)
    # grid_sample expects [-1, 1] coords
    x_norm = 2.0 * x / (W - 1) - 1.0
    y_norm = 2.0 * y / (H - 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).unsqueeze(2)  # [1, M, 1, 2]
    depth_sampled = F.grid_sample(
        depth_map.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
        grid, mode="bilinear", padding_mode="border", align_corners=True,
    ).squeeze()  # [M]

    if depth_sampled.dim() == 0:
        depth_sampled = depth_sampled.unsqueeze(0)

    # Intrinsics
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Back-project to camera space
    z = depth_sampled
    x_cam = (x - cx) / fx * z
    y_cam = (y - cy) / fy * z

    pts_cam = torch.stack([x_cam, y_cam, z, torch.ones_like(z)], dim=1)  # [M, 4]

    # Camera-to-world
    R = pose_w2c[:3, :3]  # [3, 3]
    t = pose_w2c[:3, 3]   # [3]
    R_inv = R.T
    t_inv = -R_inv @ t

    c2w = torch.eye(4, device=device, dtype=pose_w2c.dtype)
    c2w[:3, :3] = R_inv
    c2w[:3, 3] = t_inv

    pts_world = (c2w @ pts_cam.T).T[:, :3]  # [M, 3]
    return pts_world


# ===================================================================
# PnP solver
# ===================================================================

def solve_pnp(
    kp_2d: np.ndarray,
    pts_3d: np.ndarray,
    K: np.ndarray,
    ransac_threshold: float = 3.0,
    max_iters: int = 10000,
) -> Optional[np.ndarray]:
    """Solve PnP+RANSAC to estimate camera pose.

    Args:
        kp_2d: [M, 2] 2D keypoints in full image coords (x, y order for OpenCV)
        pts_3d: [M, 3] corresponding 3D world points
        K: [3, 3] camera intrinsics (at full image resolution)
        ransac_threshold: reprojection error threshold in pixels
        max_iters: max RANSAC iterations

    Returns:
        estimated_w2c: [4, 4] estimated world-to-camera matrix, or None if failed
    """
    cv2 = _cv2()
    if len(kp_2d) < 4:
        return None

    dist_coeffs = np.zeros(4)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d.astype(np.float64),
        kp_2d.astype(np.float64),
        K.astype(np.float64),
        dist_coeffs,
        iterationsCount=max_iters,
        reprojectionError=ransac_threshold,
        flags=cv2.SOLVEPNP_P3P,
    )

    if not success or inliers is None or len(inliers) < 4:
        return None

    # Refine with inliers
    rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
        pts_3d[inliers.ravel()].astype(np.float64),
        kp_2d[inliers.ravel()].astype(np.float64),
        K.astype(np.float64),
        dist_coeffs,
        rvec, tvec,
    )

    R, _ = cv2.Rodrigues(rvec_refined)
    t = tvec_refined.ravel()

    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t

    return w2c


# ===================================================================
# Pose error computation
# ===================================================================

def compute_pose_error(
    estimated_w2c: np.ndarray,
    gt_w2c: np.ndarray,
) -> Tuple[float, float]:
    """Compute rotation and translation error between two w2c poses.

    Args:
        estimated_w2c: [4, 4] estimated world-to-camera
        gt_w2c: [4, 4] ground truth world-to-camera

    Returns:
        rot_error_deg: rotation error in degrees
        trans_error_m: translation error in meters
    """
    # Relative transform: est @ gt_inv
    R_est = estimated_w2c[:3, :3]
    t_est = estimated_w2c[:3, 3]
    R_gt = gt_w2c[:3, :3]
    t_gt = gt_w2c[:3, 3]

    # Rotation error
    R_rel = R_est @ R_gt.T
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    rot_error_deg = np.degrees(np.arccos(cos_angle))

    # Translation error: compare camera positions in world frame
    # camera position in world = -R^T @ t
    pos_est = -R_est.T @ t_est
    pos_gt = -R_gt.T @ t_gt
    trans_error_m = np.linalg.norm(pos_est - pos_gt)

    return rot_error_deg, trans_error_m


# ===================================================================
# Rendering helper
# ===================================================================

@torch.no_grad()
def render_superpoint_features(
    trainer: "LocGSTrainer",
    pose_w2c: torch.Tensor,
    device: str = "cuda:0",
) -> Dict[str, torch.Tensor]:
    """Render SuperPoint features from a given pose using the trained model.

    Args:
        trainer: LocGSTrainer with loaded model
        pose_w2c: [4, 4] world-to-camera matrix

    Returns:
        dict with 'descriptor' [256, H, W], 'detector' [65, H, W], 'depth' [H, W]
    """
    pose_batch = pose_w2c.unsqueeze(0).to(device).float()  # [1, 4, 4]

    sp_out = trainer.render_superpoint_outputs_for_pose(pose_batch)
    desc = sp_out["descriptor"][0]  # [256, H, W]
    det = sp_out["detector"][0]     # [65, H, W]
    depth = sp_out["depth"][0]      # [H, W]
    locability = sp_out["locability"]
    if locability.numel() > 0:
        locability = locability[0, 0]
    else:
        locability = torch.empty(0, device=desc.device)

    return {"descriptor": desc, "detector": det, "depth": depth, "locability": locability}


# ===================================================================
# Main evaluation
# ===================================================================

@torch.no_grad()
def run_localization_eval(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    num_samples: int = 100,
    device: str = "cuda:0",
    confidence_threshold: float = 0.015,
    nms_radius: int = 2,
    max_keypoints: int = 1000,
    ratio_threshold: float = 0.9,
    ransac_threshold: float = 3.0,
    use_locability_prior: bool = False,
    locability_prior_weight: float = 0.5,
    subpixel_refine: bool = True,
    subpixel_window_radius: int = 2,
    subpixel_temperature: float = 0.05,
    seed: int = 42,
):
    """Run full localization evaluation."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    config = load_config(config_path)
    config.device = device

    trainer = LocGSTrainer(config)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    trainer.model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if "sp_output_head_state_dict" in ckpt:
        trainer.sp_output_head.load_state_dict(ckpt["sp_output_head_state_dict"])
    if "sharpener_state_dict" in ckpt:
        trainer.sharpener.load_state_dict(ckpt["sharpener_state_dict"])
    if (
        "sp_locability_adapter_state_dict" in ckpt
        and getattr(trainer, "sp_locability_adapter", None) is not None
    ):
        trainer.sp_locability_adapter.load_state_dict(
            ckpt["sp_locability_adapter_state_dict"], strict=False
        )
    print(f"Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, "
          f"best_cosine={ckpt.get('best_cosine', '?'):.4f}")

    trainer.model.eval()
    trainer.sp_output_head.eval()
    trainer.sharpener.eval()
    if getattr(trainer, "sp_locability_adapter", None) is not None:
        trainer.sp_locability_adapter.eval()

    # Camera intrinsics at feature-map resolution
    K_feat = trainer.renderer.K.cpu().numpy()  # [3, 3]

    # Camera intrinsics at full image resolution (for PnP — 2D keypoints are at full res)
    # Feature map is 1/8 of image → scale intrinsics by 8
    K_full = K_feat.copy()
    K_full[0, 0] *= 8  # fx
    K_full[1, 1] *= 8  # fy
    K_full[0, 2] = K_full[0, 2] * 8 + 3.5  # cx: (cx_feat * 8) + 3.5 to center
    K_full[1, 2] = K_full[1, 2] * 8 + 3.5  # cy

    K_feat_torch = trainer.renderer.K.float()

    # ------------------------------------------------------------------
    # Collect validation frames
    # ------------------------------------------------------------------
    val_iter = iter(trainer.val_loader)
    all_batches = []
    for batch in val_iter:
        all_batches.append(batch)
    print(f"Collected {sum(b['pose_w2c'].shape[0] for b in all_batches)} validation frames")

    # Flatten batches into individual frames
    frames = []
    for batch in all_batches:
        B = batch["pose_w2c"].shape[0]
        for b in range(B):
            frame = {
                "pose_w2c": batch["pose_w2c"][b],  # [4, 4]
                "gt_descriptor": batch["teacher_features"][b],  # [256, H, W]
            }
            if "detector_features" in batch:
                frame["gt_detector"] = batch["detector_features"][b]  # [65, H, W]
            frames.append(frame)

    num_samples = min(num_samples, len(frames))
    # Sample frames (deterministic with seed)
    indices = np.random.permutation(len(frames))[:num_samples]
    frames = [frames[i] for i in indices]
    print(f"Evaluating {num_samples} frames across {len(NOISE_LEVELS)} noise levels")

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    results = {}
    all_results_detail = {}

    for noise_name, noise_cfg in NOISE_LEVELS.items():
        trans_m = noise_cfg["trans_m"]
        rot_deg = noise_cfg["rot_deg"]
        print(f"\n{'='*60}")
        print(f"Noise level: {noise_name} (trans={trans_m*100:.0f}cm, rot={rot_deg:.0f}deg)")
        print(f"{'='*60}")

        rot_errors = []
        trans_errors = []
        num_keypoints_list = []
        num_matches_list = []
        num_inliers_list = []
        pnp_failures = 0
        match_failures = 0
        per_frame_results = []

        t0 = time.time()

        for fi, frame in enumerate(frames):
            gt_w2c = frame["pose_w2c"].numpy().astype(np.float64)

            # 1. Perturb pose
            pert_w2c = perturb_pose(gt_w2c, trans_m, rot_deg)
            pert_w2c_torch = torch.from_numpy(pert_w2c).float()

            # 2. Render features from perturbed pose
            rendered = render_superpoint_features(trainer, pert_w2c_torch, device)
            rendered_desc = rendered["descriptor"]  # [256, H, W]
            rendered_det = rendered["detector"]     # [65, H, W]
            rendered_depth = rendered["depth"]      # [H, W]
            rendered_locability = rendered.get("locability")
            if (
                not use_locability_prior
                or rendered_locability is None
                or rendered_locability.numel() == 0
            ):
                rendered_locability = None

            # 3. Extract keypoints from GT detector heatmap
            gt_det = frame.get("gt_detector")
            if gt_det is None:
                # If no GT detector, use rendered detector
                gt_det = rendered_det.cpu()
            gt_det = gt_det.to(device).float()

            keypoints, kp_scores = extract_keypoints_from_detector(
                gt_det, confidence_threshold, nms_radius, max_keypoints
            )
            num_kp = keypoints.shape[0]
            num_keypoints_list.append(num_kp)

            if num_kp < 4:
                pnp_failures += 1
                per_frame_results.append({
                    "frame_idx": fi, "status": "too_few_keypoints",
                    "num_keypoints": num_kp,
                })
                continue

            # 4. Sample GT descriptors at keypoint locations
            gt_desc = frame["gt_descriptor"].to(device).float()
            gt_desc = F.normalize(gt_desc, p=2, dim=0)
            gt_kp_descs = sample_descriptors_bilinear(gt_desc, keypoints)  # [K, 256]

            # 5. Dense match GT descriptors against rendered descriptor map
            matched_kp_idx, matched_pos, match_scores = match_descriptors_dense(
                gt_kp_descs,
                rendered_desc,
                ratio_threshold,
                locability_map=rendered_locability,
                locability_weight=locability_prior_weight,
                subpixel_refine=subpixel_refine,
                subpixel_window_radius=subpixel_window_radius,
                subpixel_temperature=subpixel_temperature,
            )
            num_matches = matched_kp_idx.shape[0]
            num_matches_list.append(num_matches)

            if num_matches < 4:
                match_failures += 1
                per_frame_results.append({
                    "frame_idx": fi, "status": "too_few_matches",
                    "num_keypoints": num_kp, "num_matches": num_matches,
                })
                continue

            # 6. Unproject matched positions to 3D using rendered depth + perturbed pose
            pts_3d = unproject_to_world(
                matched_pos, rendered_depth, pert_w2c_torch.to(device),
                K_feat_torch.to(device),
            )  # [M, 3]

            # Filter out invalid depth (too close or too far)
            valid_depth = (pts_3d[:, 2].abs() < 50.0) & (rendered_depth.flatten().abs() > 0.01)[:1].expand(pts_3d.shape[0])
            # Actually check sampled depths
            depth_vals = rendered_depth.unsqueeze(0).unsqueeze(0)
            y_pos = matched_pos[:, 0]
            x_pos = matched_pos[:, 1]
            H_d, W_d = rendered_depth.shape
            # Simple nearest depth check
            y_idx = y_pos.long().clamp(0, H_d - 1)
            x_idx = x_pos.long().clamp(0, W_d - 1)
            sampled_d = rendered_depth[y_idx, x_idx]
            valid_depth = (sampled_d > 0.05) & (sampled_d < 20.0)

            if valid_depth.sum() < 4:
                pnp_failures += 1
                per_frame_results.append({
                    "frame_idx": fi, "status": "insufficient_valid_depth",
                    "num_keypoints": num_kp, "num_matches": num_matches,
                    "valid_depth": int(valid_depth.sum()),
                })
                continue

            pts_3d_valid = pts_3d[valid_depth].cpu().numpy()
            matched_kp_idx_valid = matched_kp_idx[valid_depth.cpu()]

            # GT 2D keypoints for PnP (at full image resolution)
            gt_kp_full_res = keypoints[matched_kp_idx_valid.cpu()]  # [M', 2] in (y, x) coarse
            gt_kp_full_yx = gt_kp_full_res * 8.0  # scale to full resolution
            # OpenCV wants (x, y) order
            gt_kp_2d_xy = gt_kp_full_yx[:, [1, 0]].cpu().numpy()  # [M', 2] in (x, y)

            num_inliers_list.append(len(gt_kp_2d_xy))

            # 7. PnP + RANSAC
            estimated_w2c = solve_pnp(
                gt_kp_2d_xy, pts_3d_valid, K_full,
                ransac_threshold=ransac_threshold,
            )

            if estimated_w2c is None:
                pnp_failures += 1
                per_frame_results.append({
                    "frame_idx": fi, "status": "pnp_failed",
                    "num_keypoints": num_kp, "num_matches": num_matches,
                })
                continue

            # 8. Compute pose error
            rot_err, trans_err = compute_pose_error(estimated_w2c, gt_w2c)
            rot_errors.append(rot_err)
            trans_errors.append(trans_err)

            per_frame_results.append({
                "frame_idx": fi, "status": "success",
                "rot_error_deg": rot_err,
                "trans_error_m": trans_err,
                "num_keypoints": num_kp,
                "num_matches": num_matches,
            })

            if (fi + 1) % 20 == 0:
                elapsed = time.time() - t0
                successful = len(rot_errors)
                if successful > 0:
                    med_rot = np.median(rot_errors)
                    med_trans = np.median(trans_errors) * 100  # cm
                    print(f"  [{fi+1}/{num_samples}] "
                          f"median rot={med_rot:.2f}deg, trans={med_trans:.1f}cm, "
                          f"success={successful}/{fi+1}, "
                          f"elapsed={elapsed:.1f}s")

        elapsed = time.time() - t0

        # Aggregate results for this noise level
        n_success = len(rot_errors)
        n_total = num_samples
        success_rate = n_success / n_total if n_total > 0 else 0

        level_results = {
            "noise_level": noise_name,
            "trans_m": trans_m,
            "rot_deg": rot_deg,
            "num_total": n_total,
            "num_success": n_success,
            "success_rate": success_rate,
            "pnp_failures": pnp_failures,
            "match_failures": match_failures,
            "time_s": elapsed,
        }

        if n_success > 0:
            rot_arr = np.array(rot_errors)
            trans_arr = np.array(trans_errors)

            level_results.update({
                "rot_median_deg": float(np.median(rot_arr)),
                "rot_mean_deg": float(np.mean(rot_arr)),
                "rot_std_deg": float(np.std(rot_arr)),
                "rot_p90_deg": float(np.percentile(rot_arr, 90)),
                "trans_median_cm": float(np.median(trans_arr) * 100),
                "trans_mean_cm": float(np.mean(trans_arr) * 100),
                "trans_std_cm": float(np.std(trans_arr) * 100),
                "trans_p90_cm": float(np.percentile(trans_arr, 90) * 100),
                # Accuracy at thresholds
                "acc_1deg_1cm": float(((rot_arr < 1.0) & (trans_arr < 0.01)).mean()),
                "acc_2deg_2cm": float(((rot_arr < 2.0) & (trans_arr < 0.02)).mean()),
                "acc_5deg_5cm": float(((rot_arr < 5.0) & (trans_arr < 0.05)).mean()),
                "acc_10deg_10cm": float(((rot_arr < 10.0) & (trans_arr < 0.10)).mean()),
            })

            avg_kp = np.mean(num_keypoints_list)
            avg_matches = np.mean(num_matches_list) if num_matches_list else 0
            level_results["avg_keypoints"] = float(avg_kp)
            level_results["avg_matches"] = float(avg_matches)

        results[noise_name] = level_results
        all_results_detail[noise_name] = per_frame_results

        # Print summary for this level
        print(f"\n--- {noise_name} Summary ---")
        print(f"  Success rate: {n_success}/{n_total} ({success_rate*100:.1f}%)")
        if n_success > 0:
            print(f"  Rotation error:    median={level_results['rot_median_deg']:.3f}deg, "
                  f"mean={level_results['rot_mean_deg']:.3f}deg, "
                  f"p90={level_results['rot_p90_deg']:.3f}deg")
            print(f"  Translation error: median={level_results['trans_median_cm']:.2f}cm, "
                  f"mean={level_results['trans_mean_cm']:.2f}cm, "
                  f"p90={level_results['trans_p90_cm']:.2f}cm")
            print(f"  Accuracy <1deg/1cm:   {level_results['acc_1deg_1cm']*100:.1f}%")
            print(f"  Accuracy <2deg/2cm:   {level_results['acc_2deg_2cm']*100:.1f}%")
            print(f"  Accuracy <5deg/5cm:   {level_results['acc_5deg_5cm']*100:.1f}%")
            print(f"  Accuracy <10deg/10cm: {level_results['acc_10deg_10cm']*100:.1f}%")
        print(f"  PnP failures: {pnp_failures}, Match failures: {match_failures}")
        print(f"  Time: {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_path = os.path.join(output_dir, "localization_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    detail_path = os.path.join(output_dir, "localization_detail.json")
    with open(detail_path, "w") as f:
        json.dump(all_results_detail, f, indent=2)

    # ------------------------------------------------------------------
    # Generate summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("LOCALIZATION EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Frames: {num_samples}")
    print()

    header = f"{'Noise':>8s} | {'Success':>8s} | {'Rot Med':>8s} | {'Rot P90':>8s} | {'Tr Med':>8s} | {'Tr P90':>8s} | {'<1d/1cm':>8s} | {'<5d/5cm':>8s}"
    print(header)
    print("-" * len(header))
    for name in NOISE_LEVELS:
        r = results[name]
        if r["num_success"] > 0:
            print(f"{name:>8s} | {r['success_rate']*100:>7.1f}% | "
                  f"{r['rot_median_deg']:>7.3f}° | {r['rot_p90_deg']:>7.3f}° | "
                  f"{r['trans_median_cm']:>6.2f}cm | {r['trans_p90_cm']:>6.2f}cm | "
                  f"{r['acc_1deg_1cm']*100:>7.1f}% | {r['acc_5deg_5cm']*100:>7.1f}%")
        else:
            print(f"{name:>8s} | {r['success_rate']*100:>7.1f}% | {'N/A':>8s} | {'N/A':>8s} | "
                  f"{'N/A':>8s} | {'N/A':>8s} | {'N/A':>8s} | {'N/A':>8s}")

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    _generate_plots(results, output_dir)

    # Save text summary
    summary_path = os.path.join(output_dir, "localization_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Localization Evaluation Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Frames: {num_samples}\n")
        f.write(f"Keypoint threshold: {confidence_threshold}\n")
        f.write(f"Ratio test: {ratio_threshold}\n")
        f.write(f"RANSAC threshold: {ransac_threshold}px\n\n")
        f.write(f"Subpixel refine: {subpixel_refine}\n")
        f.write(f"Subpixel window radius: {subpixel_window_radius}\n")
        f.write(f"Subpixel temperature: {subpixel_temperature}\n\n")
        for name in NOISE_LEVELS:
            r = results[name]
            f.write(f"\n--- {name} (trans={r['trans_m']*100:.0f}cm, rot={r['rot_deg']:.0f}deg) ---\n")
            f.write(f"  Success: {r['num_success']}/{r['num_total']} ({r['success_rate']*100:.1f}%)\n")
            if r["num_success"] > 0:
                f.write(f"  Rot median: {r['rot_median_deg']:.4f} deg\n")
                f.write(f"  Rot mean:   {r['rot_mean_deg']:.4f} deg\n")
                f.write(f"  Rot p90:    {r['rot_p90_deg']:.4f} deg\n")
                f.write(f"  Trans median: {r['trans_median_cm']:.3f} cm\n")
                f.write(f"  Trans mean:   {r['trans_mean_cm']:.3f} cm\n")
                f.write(f"  Trans p90:    {r['trans_p90_cm']:.3f} cm\n")
                f.write(f"  Acc <1deg/1cm:   {r['acc_1deg_1cm']*100:.1f}%\n")
                f.write(f"  Acc <2deg/2cm:   {r['acc_2deg_2cm']*100:.1f}%\n")
                f.write(f"  Acc <5deg/5cm:   {r['acc_5deg_5cm']*100:.1f}%\n")
                f.write(f"  Acc <10deg/10cm: {r['acc_10deg_10cm']*100:.1f}%\n")

    print(f"\nSummary saved to {summary_path}")
    return results


def _generate_plots(results: Dict, output_dir: str):
    """Generate visualization plots."""
    noise_names = list(NOISE_LEVELS.keys())
    valid_names = [n for n in noise_names if results[n]["num_success"] > 0]

    if not valid_names:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Median errors by noise level
    rot_medians = [results[n]["rot_median_deg"] for n in valid_names]
    trans_medians = [results[n]["trans_median_cm"] for n in valid_names]

    ax = axes[0]
    x = np.arange(len(valid_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, rot_medians, width, label="Rotation (deg)", color="steelblue")
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, trans_medians, width, label="Translation (cm)", color="coral")
    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Rotation Error (deg)")
    ax2.set_ylabel("Translation Error (cm)")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_names)
    ax.set_title("Median Pose Error by Noise Level")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Plot 2: Success rate and accuracy
    ax = axes[1]
    success_rates = [results[n]["success_rate"] * 100 for n in valid_names]
    acc_1_1 = [results[n].get("acc_1deg_1cm", 0) * 100 for n in valid_names]
    acc_5_5 = [results[n].get("acc_5deg_5cm", 0) * 100 for n in valid_names]
    acc_10_10 = [results[n].get("acc_10deg_10cm", 0) * 100 for n in valid_names]

    ax.bar(x - 0.3, success_rates, 0.2, label="Success", color="green", alpha=0.7)
    ax.bar(x - 0.1, acc_10_10, 0.2, label="<10deg/10cm", color="steelblue", alpha=0.7)
    ax.bar(x + 0.1, acc_5_5, 0.2, label="<5deg/5cm", color="orange", alpha=0.7)
    ax.bar(x + 0.3, acc_1_1, 0.2, label="<1deg/1cm", color="red", alpha=0.7)
    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_names)
    ax.set_title("Success & Accuracy Rates")
    ax.legend()
    ax.set_ylim(0, 105)

    # Plot 3: Error reduction ratio (error relative to initial perturbation)
    ax = axes[2]
    init_rot = [NOISE_LEVELS[n]["rot_deg"] for n in valid_names]
    init_trans = [NOISE_LEVELS[n]["trans_m"] * 100 for n in valid_names]  # cm
    rot_ratio = [results[n]["rot_median_deg"] / init_rot[i] * 100 for i, n in enumerate(valid_names)]
    trans_ratio = [results[n]["trans_median_cm"] / init_trans[i] * 100 for i, n in enumerate(valid_names)]

    ax.plot(valid_names, rot_ratio, "o-", label="Rotation", color="steelblue", markersize=8)
    ax.plot(valid_names, trans_ratio, "s-", label="Translation", color="coral", markersize=8)
    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Residual Error (% of initial)")
    ax.set_title("Error Reduction (lower is better)")
    ax.legend()
    ax.set_ylim(0, max(max(rot_ratio), max(trans_ratio)) * 1.2 + 5)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="No improvement")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "localization_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {os.path.join(output_dir, 'localization_summary.png')}")


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Localization evaluation for Loc-GS")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test frames")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--confidence_threshold", type=float, default=0.015,
                        help="Keypoint detection threshold")
    parser.add_argument("--nms_radius", type=int, default=2, help="NMS radius")
    parser.add_argument("--max_keypoints", type=int, default=1000, help="Max keypoints per frame")
    parser.add_argument("--ratio_threshold", type=float, default=0.9,
                        help="Lowe's ratio test threshold")
    parser.add_argument("--ransac_threshold", type=float, default=3.0,
                        help="PnP RANSAC reprojection threshold (pixels)")
    parser.add_argument("--use_locability_prior", action="store_true",
                        help="Use rendered locability as a dense matching prior")
    parser.add_argument("--locability_prior_weight", type=float, default=0.5,
                        help="Weight for locability prior in dense matching")
    parser.add_argument("--disable_subpixel_refine", action="store_true",
                        help="Disable local soft-argmax refinement of dense matches")
    parser.add_argument("--subpixel_window_radius", type=int, default=2,
                        help="Local refinement radius in feature-map pixels")
    parser.add_argument("--subpixel_temperature", type=float, default=0.05,
                        help="Soft-argmax temperature for subpixel match refinement")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_localization_eval(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        nms_radius=args.nms_radius,
        max_keypoints=args.max_keypoints,
        ratio_threshold=args.ratio_threshold,
        ransac_threshold=args.ransac_threshold,
        use_locability_prior=args.use_locability_prior,
        locability_prior_weight=args.locability_prior_weight,
        subpixel_refine=not args.disable_subpixel_refine,
        subpixel_window_radius=args.subpixel_window_radius,
        subpixel_temperature=args.subpixel_temperature,
        seed=args.seed,
    )
