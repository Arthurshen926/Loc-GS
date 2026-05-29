from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _image_chw(image: torch.Tensor | Any) -> torch.Tensor:
    tensor = torch.as_tensor(image, dtype=torch.float32).cpu()
    if tensor.dim() != 3:
        raise ValueError("image must have shape [3,H,W] or [H,W,3]")
    if tensor.shape[0] == 3:
        chw = tensor
    elif tensor.shape[-1] == 3:
        chw = tensor.permute(2, 0, 1)
    else:
        raise ValueError("image must have three color channels")
    if float(chw.max().item()) > 1.5:
        chw = chw / 255.0
    return chw.clamp(0.0, 1.0)


def _texture(gray: torch.Tensor) -> torch.Tensor:
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    gx[:, 1:] = (gray[:, 1:] - gray[:, :-1]).abs()
    gy[1:, :] = (gray[1:, :] - gray[:-1, :]).abs()
    grad = (gx + gy).clamp(0.0, 1.0)
    return F.avg_pool2d(grad.reshape(1, 1, *grad.shape), kernel_size=3, stride=1, padding=1)[0, 0].clamp(0.0, 1.0)


def build_stable_region_proxy_mask(
    image: torch.Tensor | Any,
    *,
    multi_view_support: torch.Tensor | Any | None = None,
    stable_threshold: float = 0.5,
) -> dict[str, Any]:
    """Build a no-dependency stable-region proxy for v11 train/self-map filtering."""

    chw = _image_chw(image)
    red, green, blue = chw[0], chw[1], chw[2]
    gray = chw.mean(dim=0)
    texture = _texture(gray)
    glare_risk = (gray - 0.75).clamp_min(0.0) / 0.25 * (1.0 - texture).clamp(0.0, 1.0)
    vegetation_risk = ((green - torch.maximum(red, blue) - 0.10).clamp_min(0.0) / 0.50).clamp(0.0, 1.0)
    support_risk = torch.zeros_like(gray)
    if multi_view_support is not None:
        support = torch.as_tensor(multi_view_support, dtype=torch.float32).cpu()
        if support.dim() == 3:
            support = support.squeeze(0)
        if support.shape != gray.shape:
            support = F.interpolate(support.reshape(1, 1, *support.shape), size=gray.shape, mode="bilinear", align_corners=False)[0, 0]
        support_risk = (1.0 - support.clamp(0.0, 1.0)).clamp(0.0, 1.0)
    risk = torch.maximum(torch.maximum(glare_risk, vegetation_risk), support_risk).clamp(0.0, 1.0)
    stable_score = (1.0 - risk).clamp(0.0, 1.0)
    return {
        "stable_mask": stable_score >= float(stable_threshold),
        "stable_score": stable_score,
        "risk": risk,
        "risk_components": {
            "glare_sky": glare_risk,
            "vegetation": vegetation_risk,
            "low_multiview_support": support_risk,
        },
        "metadata": {
            "mask_mode": "v11_stable_region_proxy",
            "external_model": "none",
            "stable_threshold": float(stable_threshold),
        },
    }
