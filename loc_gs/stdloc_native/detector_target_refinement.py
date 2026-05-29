from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def build_lsf_detector_target(
    *,
    projected_yx: torch.Tensor | Any,
    support_weights: torch.Tensor | Any,
    height: int,
    width: int,
    sigma_px: float = 1.0,
    hard_negative_risk: torch.Tensor | Any | None = None,
    dense_worsen_risk: torch.Tensor | Any | None = None,
    native_prior: torch.Tensor | Any | None = None,
    native_prior_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Rasterize LSF-supported landmarks into detector refinement targets."""

    points = torch.as_tensor(projected_yx, dtype=torch.float32).reshape(-1, 2).cpu()
    support = torch.as_tensor(support_weights, dtype=torch.float32).reshape(-1).cpu().clamp_min(0.0)
    if support.numel() != points.shape[0]:
        raise ValueError("support_weights must match projected_yx rows")
    count = int(points.shape[0])
    h = int(height)
    w = int(width)
    if h <= 0 or w <= 0:
        raise ValueError("height and width must be positive")
    hn = torch.zeros(count, dtype=torch.float32) if hard_negative_risk is None else torch.as_tensor(hard_negative_risk, dtype=torch.float32).reshape(-1).cpu().clamp(0.0, 1.0)
    dense = torch.zeros(count, dtype=torch.float32) if dense_worsen_risk is None else torch.as_tensor(dense_worsen_risk, dtype=torch.float32).reshape(-1).cpu().clamp(0.0, 1.0)
    if hn.numel() != count or dense.numel() != count:
        raise ValueError("risk tensors must match projected_yx rows")
    risk = (0.6 * hn + 0.4 * dense).clamp(0.0, 1.0)
    effective = support * (1.0 - risk)
    if effective.numel() and float(effective.max().item()) > 0.0:
        effective = effective / effective.max().clamp_min(1e-12)

    y_grid, x_grid = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    sigma = max(float(sigma_px), 1e-4)
    heatmap = torch.zeros(1, 1, h, w, dtype=torch.float32)
    weight = torch.zeros_like(heatmap)
    for idx in range(count):
        if not torch.isfinite(points[idx]).all():
            continue
        y = points[idx, 0].clamp(0.0, float(h - 1))
        x = points[idx, 1].clamp(0.0, float(w - 1))
        kernel = torch.exp(-0.5 * ((y_grid - y).square() + (x_grid - x).square()) / (sigma * sigma))
        value = effective[idx].clamp(0.0, 1.0)
        heatmap[0, 0] = torch.maximum(heatmap[0, 0], kernel * value)
        weight[0, 0] = torch.maximum(weight[0, 0], kernel * (0.25 + 0.75 * value))

    if native_prior is not None and float(native_prior_weight) > 0.0:
        prior = torch.as_tensor(native_prior, dtype=torch.float32).cpu()
        if prior.dim() == 2:
            prior = prior.reshape(1, 1, *prior.shape)
        elif prior.dim() == 3:
            prior = prior.unsqueeze(1)
        if prior.shape[-2:] != (h, w):
            prior = F.interpolate(prior, size=(h, w), mode="bilinear", align_corners=False)
        prior = (prior[:, :1].clamp(0.0, 1.0) * float(native_prior_weight)).clamp(0.0, 1.0)
        heatmap = torch.maximum(heatmap, prior)
        weight = torch.maximum(weight, prior)
    return heatmap.clamp(0.0, 1.0), weight.clamp(0.0, 1.0), {
        "target_mode": "lsf_detector_refinement",
        "point_count": int(count),
        "risk_suppressed_count": int((risk > 0.5).sum().item()),
        "sigma_px": float(sigma_px),
        "native_prior_weight": float(native_prior_weight),
    }
