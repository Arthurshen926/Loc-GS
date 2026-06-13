from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from loc_gs.localization.stdloc_detector import simple_nms


@torch.no_grad()
def extract_ulfloc_fullres_scene_detector_keypoints(
    feature_map: torch.Tensor,
    detector: nn.Module,
    *,
    max_keypoints: int,
    nms_radius: int = 4,
    score_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select keypoints directly on a full-resolution detector feature map."""

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
    y = ids // width
    x = ids % width
    keypoints_xy = torch.stack([x.float(), y.float()], dim=-1)
    return keypoints_xy.to(dtype=torch.float32), values.to(dtype=torch.float32)


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
def extract_ulfloc_scene_detector_keypoints_with_superpoint_scores(
    feature_map: torch.Tensor,
    superpoint_scores: torch.Tensor,
    detector: nn.Module,
    *,
    max_keypoints: int,
    nms_radius: int = 4,
    score_threshold: float = 0.0,
    descriptor_stride: int = 8,
    blend_alpha: float = 0.5,
    fusion_rule: str = "geometric",
    remove_borders: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse a scene detector heatmap with SuperPoint's full-resolution scores.

    The coarse scene detector is not a valid direct replacement for SuperPoint:
    it has one score per descriptor cell and would quantize all keypoints to
    cell centers.  This helper keeps SuperPoint's full-resolution subcell score
    map as the coordinate source, while using the scene detector as a residual
    cell-level prior.
    """

    fmap = torch.as_tensor(feature_map, dtype=torch.float32)
    if fmap.dim() == 3:
        detector_input = fmap.unsqueeze(0)
    elif fmap.dim() == 4:
        detector_input = fmap
    else:
        raise ValueError("feature_map must have shape [C,H,W] or [B,C,H,W]")
    if int(max_keypoints) <= 0:
        return fmap.new_empty((0, 2)), fmap.new_empty((0,))

    scores = torch.as_tensor(superpoint_scores, dtype=torch.float32, device=fmap.device)
    while scores.dim() > 2:
        scores = scores[0]
    if scores.dim() != 2:
        raise ValueError("superpoint_scores must reduce to shape [H,W]")

    heatmap = detector(detector_input)
    scene_heat = torch.as_tensor(heatmap, dtype=torch.float32, device=fmap.device)
    while scene_heat.dim() > 2:
        scene_heat = scene_heat[0]
    if scene_heat.dim() != 2:
        raise ValueError("detector heatmap must reduce to shape [Hc,Wc]")

    alpha = min(max(float(blend_alpha), 0.0), 1.0)
    rule = str(fusion_rule).strip().lower()
    if alpha > 0.0:
        scene_full = F.interpolate(
            scene_heat.view(1, 1, int(scene_heat.shape[-2]), int(scene_heat.shape[-1])),
            size=(int(scores.shape[-2]), int(scores.shape[-1])),
            mode="nearest",
        )[0, 0].clamp_min(0.0)
        if rule in {"residual_boost", "boost", "positive_boost"}:
            scene_norm = scene_full
            if scene_norm.numel() > 0 and float(scene_norm.max().item()) > 0.0:
                scene_norm = scene_norm / scene_norm.max().clamp_min(1e-12)
            fused = scores.clamp_min(0.0) * (1.0 + alpha * scene_norm.clamp(0.0, 1.0))
        elif rule in {"geometric", "multiply", "multiplicative"}:
            fused = scores.clamp_min(1e-12).pow(1.0 - alpha) * scene_full.clamp_min(1e-12).pow(alpha)
        else:
            raise ValueError(f"unsupported scene detector fusion_rule: {fusion_rule}")
    else:
        fused = scores.clamp_min(0.0)

    nms_scores = simple_nms(fused, int(nms_radius))
    border = max(0, int(remove_borders))
    if border > 0 and nms_scores.numel() > 0:
        nms_scores = nms_scores.clone()
        nms_scores[:border, :] = -1.0
        nms_scores[:, :border] = -1.0
        nms_scores[-border:, :] = -1.0
        nms_scores[:, -border:] = -1.0
    flat = nms_scores.reshape(-1)
    if float(score_threshold) > 0.0:
        valid = torch.where(flat > float(score_threshold))[0]
    else:
        valid = torch.where(flat > 0.0)[0]
    if int(valid.numel()) == 0:
        return fmap.new_empty((0, 2)), fmap.new_empty((0,))

    valid_scores = flat[valid]
    topk = min(int(max_keypoints), int(valid_scores.numel()))
    values, order = torch.topk(valid_scores, k=topk, sorted=True)
    ids = valid[order]
    width = int(nms_scores.shape[-1])
    y = ids // width
    x = ids % width
    keypoints_xy = torch.stack([x.float(), y.float()], dim=-1)
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
