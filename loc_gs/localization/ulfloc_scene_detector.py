from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from loc_gs.localization.stdloc_detector import simple_nms


@torch.no_grad()
def extract_ulfloc_scene_detector_keypoints(
    feature_map: torch.Tensor,
    detector: nn.Module,
    *,
    max_keypoints: int,
    nms_radius: int = 4,
    score_threshold: float = 0.0,
    descriptor_stride: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select ULF-Loc sparse keypoints from a scene-specific detector heatmap.

    ULF-Loc samples SuperPoint descriptors with pixel-space ``xy`` coordinates,
    while a scene-specific detector naturally runs on the coarse descriptor map.
    This helper converts coarse cell centers to SuperPoint pixel coordinates:
    ``x = (x_cell + 0.5) * stride - 0.5``.
    """

    fmap = torch.as_tensor(feature_map, dtype=torch.float32)
    if fmap.dim() == 3:
        detector_input = fmap.unsqueeze(0)
        height, width = int(fmap.shape[-2]), int(fmap.shape[-1])
    elif fmap.dim() == 4:
        detector_input = fmap
        height, width = int(fmap.shape[-2]), int(fmap.shape[-1])
    else:
        raise ValueError("feature_map must have shape [C,H,W] or [B,C,H,W]")
    if int(max_keypoints) <= 0:
        return fmap.new_empty((0, 2)), fmap.new_empty((0,))

    heatmap = detector(detector_input)
    heat = torch.as_tensor(heatmap, dtype=torch.float32, device=fmap.device)
    while heat.dim() > 2:
        heat = heat[0]
    if heat.shape != (height, width):
        raise ValueError(f"detector heatmap shape {tuple(heat.shape)} does not match feature map {(height, width)}")

    scores_map = simple_nms(heat, int(nms_radius)).reshape(-1)
    if float(score_threshold) > 0.0:
        valid = torch.where(scores_map > float(score_threshold))[0]
    else:
        valid = torch.where(scores_map > 0.0)[0]
    if int(valid.numel()) == 0:
        return fmap.new_empty((0, 2)), fmap.new_empty((0,))

    valid_scores = scores_map[valid]
    topk = min(int(max_keypoints), int(valid_scores.numel()))
    values, order = torch.topk(valid_scores, k=topk, sorted=True)
    ids = valid[order]
    y_cell = ids // width
    x_cell = ids % width
    stride = float(max(1, int(descriptor_stride)))
    x = (x_cell.float() + 0.5) * stride - 0.5
    y = (y_cell.float() + 0.5) * stride - 0.5
    keypoints_xy = torch.stack([x, y], dim=-1)
    return keypoints_xy.to(dtype=torch.float32), values.to(dtype=torch.float32)


@torch.no_grad()
def rerank_superpoint_keypoints_with_scene_detector(
    keypoints_xy: torch.Tensor,
    keypoint_scores: torch.Tensor,
    descriptors: torch.Tensor,
    feature_map: torch.Tensor,
    detector: nn.Module,
    *,
    max_keypoints: int,
    descriptor_stride: int = 8,
    blend_alpha: float = 0.5,
    native_keep_fraction: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use scene-specific detector heatmap to rerank native SuperPoint candidates."""

    kpts = torch.as_tensor(keypoints_xy, dtype=torch.float32)
    native_scores = torch.as_tensor(keypoint_scores, dtype=torch.float32, device=kpts.device).reshape(-1)
    desc = torch.as_tensor(descriptors, dtype=torch.float32, device=kpts.device)
    if kpts.numel() == 0:
        return kpts.reshape(0, 2), native_scores.reshape(0), desc.reshape(0, *desc.shape[1:])
    if kpts.shape[0] != native_scores.numel() or kpts.shape[0] != desc.shape[0]:
        raise ValueError("keypoints, scores, and descriptors must have matching first dimension")
    fmap = torch.as_tensor(feature_map, dtype=torch.float32, device=kpts.device)
    detector_input = fmap.unsqueeze(0) if fmap.dim() == 3 else fmap
    heatmap = detector(detector_input)
    heat = torch.as_tensor(heatmap, dtype=torch.float32, device=kpts.device)
    while heat.dim() > 2:
        heat = heat[0]
    height, width = int(heat.shape[-2]), int(heat.shape[-1])
    stride = float(max(1, int(descriptor_stride)))
    coarse_xy = ((kpts + 0.5) / stride) - 0.5
    grid_x = ((coarse_xy[:, 0] + 0.5) / max(width, 1)) * 2.0 - 1.0
    grid_y = ((coarse_xy[:, 1] + 0.5) / max(height, 1)) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2)
    scene_scores = F.grid_sample(
        heat.view(1, 1, height, width),
        grid,
        mode="bilinear",
        align_corners=False,
    ).view(-1).clamp_min(0.0)
    alpha = min(max(float(blend_alpha), 0.0), 1.0)
    combined = native_scores.clamp_min(1e-8).pow(1.0 - alpha) * scene_scores.clamp_min(1e-8).pow(alpha)
    topk = min(max(0, int(max_keypoints)), int(combined.numel()))
    if topk == 0:
        return kpts.new_empty((0, 2)), native_scores.new_empty((0,)), desc.new_empty((0, *desc.shape[1:]))
    native_keep = min(topk, max(0, int(round(topk * max(0.0, min(float(native_keep_fraction), 1.0))))))
    if native_keep > 0:
        native_values, native_order = torch.topk(native_scores, k=min(native_keep, int(native_scores.numel())), sorted=True)
        selected = [int(idx) for idx in native_order.tolist()]
        selected_set = set(selected)
        fill_count = topk - len(selected)
        if fill_count > 0:
            fill_scores = combined.clone()
            if selected:
                fill_scores[torch.tensor(selected, dtype=torch.long, device=fill_scores.device)] = -1.0
            _fill_values, fill_order = torch.topk(fill_scores, k=fill_count, sorted=True)
            selected.extend(int(idx) for idx in fill_order.tolist() if int(idx) not in selected_set)
        order = torch.tensor(selected[:topk], dtype=torch.long, device=kpts.device)
        return kpts[order], combined[order].clamp_min(native_values.min().item() if native_values.numel() else 0.0), desc[order]
    values, order = torch.topk(combined, k=topk, sorted=True)
    return kpts[order], values, desc[order]
