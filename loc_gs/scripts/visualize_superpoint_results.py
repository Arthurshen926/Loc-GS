#!/usr/bin/env python3
"""Generate qualitative SuperPoint reconstruction and matching visualizations.

Produces two groups of outputs:
1. Reconstruction figures:
   GT RGB | GT descriptor PCA | Rendered descriptor PCA
   Cosine map | GT detector heatmap | Rendered detector heatmap
2. Matching figures:
   GT RGB with keypoints | perturbed RGB render with matched points | match canvas
   GT detector heatmap | rendered detector heatmap | rendered depth

Usage:
    python -m loc_gs.scripts.visualize_superpoint_results \
        --config configs/superpoint_hybrid_room_0_v3.yaml \
        --checkpoint output/sp_gs/room0_hybrid_v3/checkpoints/best.pth \
        --output_dir output/sp_gs/room0_hybrid_v3/qualitative_superpoint \
        --num_reconstruction 6 \
        --num_matching 8 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from loc_gs.config import load_config
from loc_gs.data.benchmark_paths import resolve_rgb_path
from loc_gs.geometry_utils import resolve_use_2dgs
from loc_gs.rendering.feature_renderer import FeatureFieldRenderer
from loc_gs.scripts.eval_localization import (
    NOISE_LEVELS,
    compute_pose_error,
    extract_keypoints_from_detector,
    match_descriptors_dense,
    perturb_pose,
    render_superpoint_features,
    sample_descriptors_bilinear,
    unproject_to_world,
)
from loc_gs.scripts.eval_superpoint import compute_pca_visualization
from loc_gs.scripts.train_feature_field import LocGSTrainer


def _normalize_map(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def _colorize_map(values: np.ndarray, cmap: str = "magma") -> np.ndarray:
    norm = _normalize_map(values)
    rgb = plt.get_cmap(cmap)(norm)[..., :3]
    return (rgb * 255).astype(np.uint8)


def _upsample_uint8(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def _tensor_rgb_to_uint8(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)


def _detector_logits_to_heatmap(detector_logits: torch.Tensor) -> np.ndarray:
    probs = F.softmax(detector_logits.float(), dim=0)
    heatmap = F.pixel_shuffle(probs[:64].unsqueeze(0), 8).squeeze(0).squeeze(0)
    return heatmap.detach().cpu().numpy().astype(np.float32)


def _build_fullres_renderer(config, device: torch.device) -> FeatureFieldRenderer:
    renderer = FeatureFieldRenderer(
        image_height=getattr(config, "image_height", 480),
        image_width=getattr(config, "image_width", 640),
        fx=getattr(config, "fx", 320.0),
        fy=getattr(config, "fy", 320.0),
        cx=getattr(config, "cx", 319.5),
        cy=getattr(config, "cy", 239.5),
        max_channels_per_chunk=getattr(config, "max_channels_per_chunk", 32),
        use_2dgs=resolve_use_2dgs(config),
    )
    return renderer.to(device)


def _render_rgb(renderer: FeatureFieldRenderer, model: torch.nn.Module, pose_w2c: torch.Tensor) -> np.ndarray:
    rgb = renderer.render_features_and_rgb(model, pose_w2c.float().unsqueeze(0))["rgb"][0]
    return _tensor_rgb_to_uint8(rgb)


def _load_rgb_image(rgb_path: Optional[Path]) -> Optional[np.ndarray]:
    if rgb_path is None or not rgb_path.exists():
        return None
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _unwrap_dataset(dataset, index: int):
    if isinstance(dataset, Subset):
        return _unwrap_dataset(dataset.dataset, int(dataset.indices[index]))
    if isinstance(dataset, ConcatDataset):
        dataset_idx = bisect_right(dataset.cumulative_sizes, index)
        prev = 0 if dataset_idx == 0 else dataset.cumulative_sizes[dataset_idx - 1]
        local_index = index - prev
        return _unwrap_dataset(dataset.datasets[dataset_idx], local_index)
    return dataset, index


def _resolve_sample_metadata(trainer: LocGSTrainer, dataset_index: int) -> Dict[str, Any]:
    base_dataset, local_index = _unwrap_dataset(trainer.val_dataset, dataset_index)
    frame_idx = int(base_dataset.frame_indices[local_index])
    feature_path = Path(base_dataset.feature_paths[local_index])

    rgb_root = getattr(base_dataset, "rgb_dir", None)
    if rgb_root is None:
        pose_file = getattr(base_dataset, "pose_file", None)
        pose_dir = getattr(base_dataset, "pose_dir", None)
        if pose_file is not None:
            rgb_root = pose_file.parent / "rgb"
        elif pose_dir is not None:
            rgb_root = pose_dir.parent / "rgb"

    rgb_path = None
    if rgb_root is not None:
        rgb_path = resolve_rgb_path(rgb_root, frame_idx, getattr(base_dataset, "dataset_type", "replica"))

    split_name = ""
    if rgb_root is not None:
        split_name = Path(rgb_root).parent.name

    return {
        "frame_idx": frame_idx,
        "feature_path": str(feature_path),
        "rgb_path": str(rgb_path) if rgb_path is not None else "",
        "split": split_name,
        "dataset_index": dataset_index,
    }


def _align_spatial(gt_tensor: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    if gt_tensor.shape[-2:] == target_hw:
        return gt_tensor
    resized = F.interpolate(
        gt_tensor.unsqueeze(0).float(),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def _make_reconstruction_figure(
    trainer: LocGSTrainer,
    rgb_renderer: FeatureFieldRenderer,
    sample: Dict[str, torch.Tensor],
    metadata: Dict[str, Any],
    output_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    pose_w2c = sample["pose_w2c"].to(device).float()
    rendered = render_superpoint_features(trainer, pose_w2c, str(device))

    pred_desc = rendered["descriptor"]
    pred_det = rendered["detector"]

    gt_desc = sample["teacher_features"].to(device).float()
    gt_desc = _align_spatial(gt_desc, tuple(pred_desc.shape[-2:]))
    gt_desc = F.normalize(gt_desc, p=2, dim=0)

    gt_det = sample.get("detector_features")
    if gt_det is not None:
        gt_det = gt_det.to(device).float()
        gt_det = _align_spatial(gt_det, tuple(pred_det.shape[-2:]))
    else:
        gt_det = pred_det.detach()

    cosine_map = (pred_desc * gt_desc).sum(dim=0).detach().cpu().numpy()
    mean_cosine = float(cosine_map.mean())
    mse = float(F.mse_loss(pred_desc, gt_desc).item())

    gt_pca, pca_model = compute_pca_visualization(gt_desc)
    pred_pca, _ = compute_pca_visualization(pred_desc, pca_model)

    gt_heat = _detector_logits_to_heatmap(gt_det)
    pred_heat = _detector_logits_to_heatmap(pred_det)

    rgb = _load_rgb_image(Path(metadata["rgb_path"])) if metadata.get("rgb_path") else None
    if rgb is None:
        rgb = _render_rgb(rgb_renderer, trainer.model, pose_w2c)

    H_img, W_img = rgb.shape[:2]
    gt_pca_up = _upsample_uint8(gt_pca, (W_img, H_img))
    pred_pca_up = _upsample_uint8(pred_pca, (W_img, H_img))
    cosine_vis = _colorize_map(cv2.resize(cosine_map, (W_img, H_img), interpolation=cv2.INTER_LINEAR), cmap="RdYlGn")
    gt_heat_vis = _colorize_map(gt_heat, cmap="inferno")
    pred_heat_vis = _colorize_map(pred_heat, cmap="inferno")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Frame {metadata['frame_idx']} ({metadata['split']}) | cosine={mean_cosine:.4f} | mse={mse:.6f}",
        fontsize=14,
    )

    panels = [
        (rgb, "GT RGB"),
        (gt_pca_up, "GT Descriptor (PCA)"),
        (pred_pca_up, "Rendered Descriptor (PCA)"),
        (cosine_vis, f"Cosine Similarity (mean={mean_cosine:.4f})"),
        (gt_heat_vis, "GT Detector Heatmap"),
        (pred_heat_vis, "Rendered Detector Heatmap"),
    ]

    for ax, (image, title) in zip(axes.flat, panels):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "type": "reconstruction",
        "file": str(output_path),
        "frame_idx": int(metadata["frame_idx"]),
        "split": metadata["split"],
        "dataset_index": int(metadata["dataset_index"]),
        "mean_cosine": mean_cosine,
        "mse": mse,
    }


def _solve_pnp_with_inliers(
    keypoints_2d_xy: np.ndarray,
    points_3d: np.ndarray,
    K: np.ndarray,
    ransac_threshold: float,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    if len(keypoints_2d_xy) < 4:
        return None, np.zeros(len(keypoints_2d_xy), dtype=bool)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d.astype(np.float64),
        keypoints_2d_xy.astype(np.float64),
        K.astype(np.float64),
        np.zeros(4, dtype=np.float64),
        iterationsCount=10000,
        reprojectionError=ransac_threshold,
        flags=cv2.SOLVEPNP_P3P,
    )
    if not success or inliers is None or len(inliers) < 4:
        return None, np.zeros(len(keypoints_2d_xy), dtype=bool)

    rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
        points_3d[inliers.ravel()].astype(np.float64),
        keypoints_2d_xy[inliers.ravel()].astype(np.float64),
        K.astype(np.float64),
        np.zeros(4, dtype=np.float64),
        rvec,
        tvec,
    )
    R, _ = cv2.Rodrigues(rvec_refined)
    pose_w2c = np.eye(4, dtype=np.float64)
    pose_w2c[:3, :3] = R
    pose_w2c[:3, 3] = tvec_refined.reshape(-1)

    inlier_mask = np.zeros(len(keypoints_2d_xy), dtype=bool)
    inlier_mask[inliers.ravel()] = True
    return pose_w2c, inlier_mask


def _make_match_canvas(
    gt_rgb: np.ndarray,
    render_rgb: np.ndarray,
    gt_points_xy: np.ndarray,
    render_points_xy: np.ndarray,
    inlier_mask: np.ndarray,
) -> np.ndarray:
    H, W = gt_rgb.shape[:2]
    canvas = np.zeros((H, W * 2, 3), dtype=np.uint8)
    canvas[:, :W] = gt_rgb
    canvas[:, W:] = render_rgb

    if len(gt_points_xy) == 0:
        return canvas

    colors = (plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(gt_points_xy)))[:, :3] * 255).astype(np.uint8)
    for i, (pt_gt, pt_render) in enumerate(zip(gt_points_xy, render_points_xy)):
        x0, y0 = int(round(float(pt_gt[0]))), int(round(float(pt_gt[1])))
        x1, y1 = int(round(float(pt_render[0]))), int(round(float(pt_render[1])))
        color = colors[i].tolist()
        if inlier_mask.size == len(gt_points_xy) and not inlier_mask[i]:
            color = [255, 180, 40]
        cv2.circle(canvas, (x0, y0), 4, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x1 + W, y1), 4, color, -1, lineType=cv2.LINE_AA)
        cv2.line(canvas, (x0, y0), (x1 + W, y1), color, 1, lineType=cv2.LINE_AA)
    return canvas


def _overlay_points(image: np.ndarray, points_xy: np.ndarray, inlier_mask: Optional[np.ndarray] = None) -> np.ndarray:
    vis = image.copy()
    for i, point in enumerate(points_xy):
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        if inlier_mask is None:
            color = (80, 220, 255)
        else:
            color = (80, 255, 120) if inlier_mask[i] else (255, 180, 40)
        cv2.circle(vis, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (x, y), 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return vis


def _sample_depth_validity(depth_map: torch.Tensor, positions_yx: torch.Tensor) -> torch.Tensor:
    H, W = depth_map.shape
    y = positions_yx[:, 0].long().clamp(0, H - 1)
    x = positions_yx[:, 1].long().clamp(0, W - 1)
    sampled = depth_map[y, x]
    return (sampled > 0.05) & (sampled < 20.0)


def _make_matching_figure(
    trainer: LocGSTrainer,
    rgb_renderer: FeatureFieldRenderer,
    sample: Dict[str, torch.Tensor],
    metadata: Dict[str, Any],
    noise_name: str,
    output_path: Path,
    device: torch.device,
    confidence_threshold: float,
    ratio_threshold: float,
    max_keypoints: int,
    max_draw_matches: int,
    ransac_threshold: float,
) -> Dict[str, Any]:
    gt_pose = sample["pose_w2c"].float().cpu().numpy().astype(np.float64)
    gt_pose_torch = sample["pose_w2c"].to(device).float()

    noise_cfg = NOISE_LEVELS[noise_name]
    perturbed_pose = perturb_pose(gt_pose, noise_cfg["trans_m"], noise_cfg["rot_deg"])
    perturbed_pose_torch = torch.from_numpy(perturbed_pose).to(device).float()

    rendered = render_superpoint_features(trainer, perturbed_pose_torch, str(device))
    rendered_desc = rendered["descriptor"]
    rendered_det = rendered["detector"]
    rendered_depth = rendered["depth"]

    gt_desc = sample["teacher_features"].to(device).float()
    gt_desc = _align_spatial(gt_desc, tuple(rendered_desc.shape[-2:]))
    gt_desc = F.normalize(gt_desc, p=2, dim=0)

    gt_det = sample.get("detector_features")
    if gt_det is not None:
        gt_det = gt_det.to(device).float()
        gt_det = _align_spatial(gt_det, tuple(rendered_det.shape[-2:]))
    else:
        gt_det = rendered_det.detach()

    keypoints, _scores = extract_keypoints_from_detector(
        gt_det,
        confidence_threshold=confidence_threshold,
        nms_radius=2,
        max_keypoints=max_keypoints,
    )
    gt_keypoint_desc = sample_descriptors_bilinear(gt_desc, keypoints)
    matched_indices, matched_positions, match_scores = match_descriptors_dense(
        gt_keypoint_desc,
        rendered_desc,
        ratio_threshold=ratio_threshold,
    )

    valid_depth = _sample_depth_validity(rendered_depth, matched_positions)
    matched_indices = matched_indices[valid_depth]
    matched_positions = matched_positions[valid_depth]
    match_scores = match_scores[valid_depth]

    gt_points_yx = keypoints[matched_indices]
    points_3d = unproject_to_world(
        matched_positions,
        rendered_depth,
        perturbed_pose_torch,
        trainer.renderer.K.float().to(device),
    ).detach().cpu().numpy()

    gt_points_xy_full = (gt_points_yx[:, [1, 0]] * 8.0).detach().cpu().numpy()
    render_points_xy_full = (matched_positions[:, [1, 0]] * 8.0).detach().cpu().numpy()

    K_full = np.array(
        [
            [float(getattr(trainer.cfg, "fx", 320.0)), 0.0, float(getattr(trainer.cfg, "cx", 319.5))],
            [0.0, float(getattr(trainer.cfg, "fy", 320.0)), float(getattr(trainer.cfg, "cy", 239.5))],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    estimated_pose = None
    inlier_mask = np.zeros(len(gt_points_xy_full), dtype=bool)
    if len(gt_points_xy_full) >= 4:
        estimated_pose, inlier_mask = _solve_pnp_with_inliers(
            gt_points_xy_full,
            points_3d,
            K_full,
            ransac_threshold=ransac_threshold,
        )

    init_rot, init_trans = compute_pose_error(perturbed_pose, gt_pose)
    final_rot, final_trans = (None, None)
    if estimated_pose is not None:
        final_rot, final_trans = compute_pose_error(estimated_pose, gt_pose)

    order = torch.argsort(match_scores, descending=True)
    draw_count = min(max_draw_matches, len(order))
    order_np = order[:draw_count].detach().cpu().numpy()

    gt_points_xy_draw = gt_points_xy_full[order_np] if draw_count > 0 else np.zeros((0, 2), dtype=np.float32)
    render_points_xy_draw = render_points_xy_full[order_np] if draw_count > 0 else np.zeros((0, 2), dtype=np.float32)
    inlier_mask_draw = inlier_mask[order_np] if draw_count > 0 else np.zeros(0, dtype=bool)

    gt_rgb = _load_rgb_image(Path(metadata["rgb_path"])) if metadata.get("rgb_path") else None
    if gt_rgb is None:
        gt_rgb = _render_rgb(rgb_renderer, trainer.model, gt_pose_torch)
    perturbed_rgb = _render_rgb(rgb_renderer, trainer.model, perturbed_pose_torch)

    gt_rgb_overlay = _overlay_points(gt_rgb, gt_points_xy_draw)
    perturbed_rgb_overlay = _overlay_points(perturbed_rgb, render_points_xy_draw, inlier_mask_draw)
    match_canvas = _make_match_canvas(
        gt_rgb,
        perturbed_rgb,
        gt_points_xy_draw,
        render_points_xy_draw,
        inlier_mask_draw,
    )

    gt_heat = _colorize_map(_detector_logits_to_heatmap(gt_det), cmap="inferno")
    pred_heat = _colorize_map(_detector_logits_to_heatmap(rendered_det), cmap="inferno")
    depth_np = rendered_depth.detach().cpu().numpy()
    valid = depth_np > 0.0
    if valid.any():
        lo = float(np.percentile(depth_np[valid], 5.0))
        hi = float(np.percentile(depth_np[valid], 95.0))
        depth_show = np.clip((depth_np - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    else:
        depth_show = np.zeros_like(depth_np)
    depth_vis = (plt.get_cmap("viridis")(cv2.resize(depth_show, (gt_rgb.shape[1], gt_rgb.shape[0]), interpolation=cv2.INTER_LINEAR))[..., :3] * 255).astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    title = (
        f"Frame {metadata['frame_idx']} ({metadata['split']}) | noise={noise_name} | "
        f"matches={len(gt_points_xy_full)} | inliers={int(inlier_mask.sum())} | "
        f"init={init_rot:.2f}deg/{init_trans * 100:.1f}cm"
    )
    if final_rot is not None and final_trans is not None:
        title += f" | pnp={final_rot:.2f}deg/{final_trans * 100:.1f}cm"
    else:
        title += " | pnp=failed"
    fig.suptitle(title, fontsize=13)

    panels = [
        (gt_rgb_overlay, f"GT RGB + matched keypoints ({len(gt_points_xy_draw)})"),
        (perturbed_rgb_overlay, "Perturbed 3DGS RGB + matched points"),
        (match_canvas, "Dense descriptor matches"),
        (gt_heat, "GT detector heatmap"),
        (pred_heat, "Rendered detector heatmap"),
        (depth_vis, "Rendered depth"),
    ]
    for ax, (image, label) in zip(axes.flat, panels):
        ax.imshow(image)
        ax.set_title(label)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "type": "matching",
        "file": str(output_path),
        "frame_idx": int(metadata["frame_idx"]),
        "split": metadata["split"],
        "dataset_index": int(metadata["dataset_index"]),
        "noise_level": noise_name,
        "num_matches": int(len(gt_points_xy_full)),
        "num_drawn_matches": int(draw_count),
        "num_inliers": int(inlier_mask.sum()),
        "init_rot_deg": float(init_rot),
        "init_trans_cm": float(init_trans * 100.0),
        "pnp_rot_deg": None if final_rot is None else float(final_rot),
        "pnp_trans_cm": None if final_trans is None else float(final_trans * 100.0),
    }


def _save_contact_sheet(image_paths: List[Path], output_path: Path, title: str, ncols: int = 3) -> None:
    if not image_paths:
        return
    n = len(image_paths)
    cols = min(ncols, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for ax, image_path in zip(axes.flat, image_paths):
        image = plt.imread(str(image_path))
        ax.imshow(image)
        ax.set_title(image_path.stem, fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_visualization(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    num_reconstruction: int,
    num_matching: int,
    device: str,
    seed: int,
    match_noise_levels: List[str],
    confidence_threshold: float,
    ratio_threshold: float,
    max_keypoints: int,
    max_draw_matches: int,
    ransac_threshold: float,
) -> Dict[str, Any]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_root = Path(output_dir)
    recon_dir = out_root / "reconstruction"
    match_dir = out_root / "matching"
    recon_dir.mkdir(parents=True, exist_ok=True)
    match_dir.mkdir(parents=True, exist_ok=True)

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

    trainer.model.eval()
    trainer.sp_output_head.eval()
    trainer.sharpener.eval()
    if getattr(trainer, "sp_locability_adapter", None) is not None:
        trainer.sp_locability_adapter.eval()

    dev = torch.device(device)
    rgb_renderer = _build_fullres_renderer(config, dev)

    val_size = len(trainer.val_dataset)
    total_needed = min(val_size, num_reconstruction + num_matching)
    indices = np.random.default_rng(seed).permutation(val_size)[:total_needed]
    recon_indices = indices[: min(num_reconstruction, len(indices))]
    match_indices = indices[min(num_reconstruction, len(indices)) : total_needed]

    manifest: Dict[str, Any] = {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "seed": seed,
        "num_val_frames": val_size,
        "reconstruction": [],
        "matching": [],
    }

    recon_paths: List[Path] = []
    for i, dataset_index in enumerate(recon_indices.tolist()):
        sample = trainer.val_dataset[dataset_index]
        metadata = _resolve_sample_metadata(trainer, dataset_index)
        out_path = recon_dir / f"recon_{i:02d}_frame_{metadata['frame_idx']:04d}_{metadata['split'] or 'val'}.png"
        stats = _make_reconstruction_figure(
            trainer,
            rgb_renderer,
            sample,
            metadata,
            out_path,
            dev,
        )
        manifest["reconstruction"].append(stats)
        recon_paths.append(out_path)
        print(f"[recon] saved {out_path}")

    match_paths: List[Path] = []
    if not match_noise_levels:
        match_noise_levels = list(NOISE_LEVELS.keys())
    for i, dataset_index in enumerate(match_indices.tolist()):
        sample = trainer.val_dataset[dataset_index]
        metadata = _resolve_sample_metadata(trainer, dataset_index)
        noise_name = match_noise_levels[i % len(match_noise_levels)]
        out_path = match_dir / (
            f"match_{i:02d}_{noise_name}_frame_{metadata['frame_idx']:04d}_{metadata['split'] or 'val'}.png"
        )
        stats = _make_matching_figure(
            trainer,
            rgb_renderer,
            sample,
            metadata,
            noise_name,
            out_path,
            dev,
            confidence_threshold,
            ratio_threshold,
            max_keypoints,
            max_draw_matches,
            ransac_threshold,
        )
        manifest["matching"].append(stats)
        match_paths.append(out_path)
        print(f"[match] saved {out_path}")

    recon_grid = out_root / "reconstruction_grid.png"
    match_grid = out_root / "matching_grid.png"
    _save_contact_sheet(recon_paths, recon_grid, "SuperPoint Reconstruction Visualizations")
    _save_contact_sheet(match_paths, match_grid, "SuperPoint Matching Visualizations")

    manifest["reconstruction_grid"] = str(recon_grid)
    manifest["matching_grid"] = str(match_grid)

    manifest_path = out_root / "visualization_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved manifest to {manifest_path}")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SuperPoint qualitative visualizations")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", required=True, help="Directory for visualization outputs")
    parser.add_argument("--num_reconstruction", type=int, default=6, help="Number of reconstruction figures")
    parser.add_argument("--num_matching", type=int, default=8, help="Number of matching figures")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--match_noise_levels", default="small,medium,large,xlarge", help="Comma-separated noise levels")
    parser.add_argument("--confidence_threshold", type=float, default=0.015, help="Detector threshold")
    parser.add_argument("--ratio_threshold", type=float, default=0.9, help="Lowe ratio threshold")
    parser.add_argument("--max_keypoints", type=int, default=1000, help="Max detector keypoints")
    parser.add_argument("--max_draw_matches", type=int, default=80, help="Max matches to draw per figure")
    parser.add_argument("--ransac_threshold", type=float, default=3.0, help="PnP reprojection threshold")
    args = parser.parse_args()

    run_visualization(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_reconstruction=args.num_reconstruction,
        num_matching=args.num_matching,
        device=args.device,
        seed=args.seed,
        match_noise_levels=[x.strip() for x in args.match_noise_levels.split(",") if x.strip()],
        confidence_threshold=args.confidence_threshold,
        ratio_threshold=args.ratio_threshold,
        max_keypoints=args.max_keypoints,
        max_draw_matches=args.max_draw_matches,
        ransac_threshold=args.ransac_threshold,
    )
