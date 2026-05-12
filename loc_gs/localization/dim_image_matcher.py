from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

_LOFTR_CACHE: dict[tuple[str, str], Callable] = {}


def rgb_to_grayscale(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [B, 3, H, W]")
    weights = rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (rgb.float() * weights).sum(dim=1, keepdim=True)


def _build_loftr(pretrained: str = "outdoor"):
    try:
        from kornia.feature import LoFTR
    except Exception as exc:  # pragma: no cover - dependency availability varies
        raise ImportError("Kornia LoFTR is required for LoFTR rendered matching") from exc
    return LoFTR(pretrained=str(pretrained))


def _scaled_gray(rgb: torch.Tensor, scale: float) -> tuple[torch.Tensor, float, float]:
    gray = rgb_to_grayscale(rgb.clamp(0.0, 1.0))
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError("image scale must be positive")
    if abs(scale - 1.0) < 1e-6:
        return gray, 1.0, 1.0
    h = max(8, int(round(gray.shape[-2] * scale)))
    w = max(8, int(round(gray.shape[-1] * scale)))
    resized = F.interpolate(gray, size=(h, w), mode="bilinear", align_corners=False)
    return resized, h / float(gray.shape[-2]), w / float(gray.shape[-1])


@torch.no_grad()
def match_loftr_images(
    query_rgb: torch.Tensor,
    rendered_rgb: torch.Tensor,
    *,
    matcher: Optional[Callable] = None,
    pretrained: str = "outdoor",
    image_scale: float = 1.0,
    min_confidence: float = 0.0,
    max_matches: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return detector-free LoFTR matches as query/rendered coordinates in yx order."""
    if query_rgb.ndim == 3:
        query_rgb = query_rgb.unsqueeze(0)
    if rendered_rgb.ndim == 3:
        rendered_rgb = rendered_rgb.unsqueeze(0)
    if query_rgb.shape[0] != 1 or rendered_rgb.shape[0] != 1:
        raise ValueError("LoFTR rendered matching expects a single image pair")
    device = query_rgb.device
    if matcher is None:
        cache_key = (str(pretrained), str(device))
        matcher = _LOFTR_CACHE.get(cache_key)
        if matcher is None:
            matcher = _build_loftr(pretrained=pretrained)
            if hasattr(matcher, "to"):
                matcher = matcher.to(device)
            if hasattr(matcher, "eval"):
                matcher = matcher.eval()
            _LOFTR_CACHE[cache_key] = matcher
    if hasattr(matcher, "to"):
        matcher = matcher.to(device)
    if hasattr(matcher, "eval"):
        matcher = matcher.eval()

    scale = float(image_scale)
    image0, scale0_y, scale0_x = _scaled_gray(query_rgb.to(device=device), scale)
    image1, scale1_y, scale1_x = _scaled_gray(rendered_rgb.to(device=device), scale)
    out = matcher({"image0": image0, "image1": image1})
    k0_xy = out.get("keypoints0")
    k1_xy = out.get("keypoints1")
    conf = out.get("confidence")
    if k0_xy is None or k1_xy is None or conf is None or k0_xy.numel() == 0:
        empty_long = torch.empty(0, 2, device=device, dtype=torch.float32)
        return empty_long, empty_long, torch.empty(0, device=device, dtype=query_rgb.dtype)
    k0_xy = k0_xy.to(device=device, dtype=torch.float32)
    k1_xy = k1_xy.to(device=device, dtype=torch.float32)
    k0_xy = k0_xy / k0_xy.new_tensor([scale0_x, scale0_y]).view(1, 2)
    k1_xy = k1_xy / k1_xy.new_tensor([scale1_x, scale1_y]).view(1, 2)
    scores = conf.reshape(-1).to(device=device, dtype=torch.float32)
    keep = torch.isfinite(k0_xy).all(dim=-1) & torch.isfinite(k1_xy).all(dim=-1) & torch.isfinite(scores)
    if float(min_confidence) > 0.0:
        keep = keep & (scores >= float(min_confidence))
    ids = torch.where(keep)[0]
    if ids.numel() == 0:
        empty = torch.empty(0, 2, device=device, dtype=torch.float32)
        return empty, empty, torch.empty(0, device=device, dtype=query_rgb.dtype)
    if int(max_matches) > 0 and ids.numel() > int(max_matches):
        ids = ids[torch.topk(scores[ids], k=int(max_matches)).indices]
    else:
        ids = ids[torch.argsort(scores[ids], descending=True)]
    query_yx = k0_xy[ids][:, [1, 0]]
    rendered_yx = k1_xy[ids][:, [1, 0]]
    return query_yx, rendered_yx, scores[ids].to(dtype=query_rgb.dtype)
