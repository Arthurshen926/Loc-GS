from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from loc_gs.localization.hybrid_localizer import sample_descriptors_bilinear


@dataclass
class RenderedKeypoints:
    keypoints_yx: torch.Tensor
    descriptors: torch.Tensor
    scores: torch.Tensor


def _nms_heatmap(heatmap: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return heatmap
    pooled = F.max_pool2d(
        heatmap.view(1, 1, *heatmap.shape),
        kernel_size=2 * int(radius) + 1,
        stride=1,
        padding=int(radius),
    ).view_as(heatmap)
    return heatmap * (heatmap == pooled).float()


def _detector_heatmap(detector_logits: torch.Tensor, height: int, width: int) -> torch.Tensor:
    logits = detector_logits.float()
    if logits.dim() == 4:
        logits = logits[0]
    probs = F.softmax(logits, dim=0)
    heatmap = F.pixel_shuffle(probs[:64].unsqueeze(0), 8).squeeze(0).squeeze(0)
    if heatmap.shape != (height, width):
        heatmap = F.interpolate(
            heatmap.view(1, 1, *heatmap.shape),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return heatmap


def _topk_from_heatmap(
    heatmap: torch.Tensor,
    max_keypoints: int,
    threshold: float,
    nms_radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    heatmap = _nms_heatmap(heatmap.float(), int(nms_radius))
    mask = heatmap > float(threshold)
    if mask.any():
        ys, xs = torch.where(mask)
        scores = heatmap[ys, xs]
    else:
        k = min(max(1, int(max_keypoints)), heatmap.numel())
        scores, flat = heatmap.reshape(-1).topk(k)
        ys = flat // heatmap.shape[1]
        xs = flat % heatmap.shape[1]
    order = scores.argsort(descending=True)[: max(0, int(max_keypoints))]
    keypoints = torch.stack([ys[order].float(), xs[order].float()], dim=-1)
    return keypoints, scores[order]


def select_rendered_keypoints(
    descriptor_map: torch.Tensor,
    source: str,
    max_keypoints: int = 2048,
    threshold: float = 0.0,
    nms_radius: int = 4,
    detector_logits: Optional[torch.Tensor] = None,
    locability: Optional[torch.Tensor] = None,
    projected_yx: Optional[torch.Tensor] = None,
    projected_scores: Optional[torch.Tensor] = None,
) -> RenderedKeypoints:
    """Select sparse rendered keypoints and sample descriptors at the same pixels."""
    C, H, W = descriptor_map.shape
    source = str(source)
    if source == "locability":
        if locability is None:
            raise ValueError("locability source requires locability map")
        heatmap = locability.float()
        while heatmap.dim() > 2:
            heatmap = heatmap[0]
        if heatmap.shape != (H, W):
            heatmap = F.interpolate(
                heatmap.view(1, 1, *heatmap.shape),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        keypoints, scores = _topk_from_heatmap(heatmap, max_keypoints, threshold, nms_radius)
    elif source == "detector":
        if detector_logits is None:
            raise ValueError("detector source requires detector_logits")
        heatmap = _detector_heatmap(detector_logits, H, W)
        keypoints, scores = _topk_from_heatmap(heatmap, max_keypoints, threshold, nms_radius)
    elif source == "projected_gaussian":
        if projected_yx is None:
            raise ValueError("projected_gaussian source requires projected_yx")
        scores = (
            projected_scores.to(device=projected_yx.device).float()
            if projected_scores is not None
            else torch.ones(projected_yx.shape[0], device=projected_yx.device)
        )
        valid = (
            (projected_yx[:, 0] >= 0)
            & (projected_yx[:, 0] <= H - 1)
            & (projected_yx[:, 1] >= 0)
            & (projected_yx[:, 1] <= W - 1)
            & (scores > float(threshold))
        )
        keypoints = projected_yx[valid].float()
        scores = scores[valid]
        order = scores.argsort(descending=True)[: max(0, int(max_keypoints))]
        keypoints = keypoints[order]
        scores = scores[order]
    else:
        raise ValueError(f"Unsupported rendered keypoint source: {source}")

    keypoints = keypoints.to(device=descriptor_map.device, dtype=descriptor_map.dtype)
    scores = scores.to(device=descriptor_map.device, dtype=descriptor_map.dtype)
    desc = sample_descriptors_bilinear(descriptor_map.float(), keypoints.float())
    if desc.shape[0] == 0:
        desc = descriptor_map.new_zeros((0, C))
    return RenderedKeypoints(keypoints_yx=keypoints, descriptors=desc, scores=scores)
