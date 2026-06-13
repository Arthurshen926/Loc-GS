from __future__ import annotations

from typing import Any, Mapping

import torch


def sample_heatmap_at_yx(heatmap: torch.Tensor | Any, points_yx: torch.Tensor | Any) -> torch.Tensor:
    heat = torch.as_tensor(heatmap, dtype=torch.float32)
    while heat.dim() > 2:
        heat = heat[0]
    points = torch.as_tensor(points_yx, dtype=torch.float32).reshape(-1, 2)
    if points.numel() == 0:
        return heat.new_empty((0,))
    h, w = int(heat.shape[-2]), int(heat.shape[-1])
    coords = points.long()
    y = coords[:, 0]
    x = coords[:, 1]
    in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if not bool(in_bounds.any()):
        return heat.new_empty((0,))
    return heat[y[in_bounds], x[in_bounds]].reshape(-1)


def summarize_detector_score_groups(
    heatmap: torch.Tensor | Any,
    groups_yx: Mapping[str, torch.Tensor | Any],
) -> dict[str, float | int | None]:
    heat = torch.as_tensor(heatmap, dtype=torch.float32)
    while heat.dim() > 2:
        heat = heat[0]
    summary: dict[str, float | int | None] = {
        "global_mean": float(heat.mean().item()) if heat.numel() else None,
        "global_max": float(heat.max().item()) if heat.numel() else None,
    }
    for name, points in groups_yx.items():
        sampled = sample_heatmap_at_yx(heat, points)
        summary[f"{name}_count"] = int(sampled.numel())
        summary[f"{name}_mean"] = float(sampled.mean().item()) if sampled.numel() else None
        summary[f"{name}_max"] = float(sampled.max().item()) if sampled.numel() else None
    return summary


def keypoint_overlap_fraction(
    native_xy: torch.Tensor | Any,
    fused_xy: torch.Tensor | Any,
) -> dict[str, float | int]:
    native = torch.as_tensor(native_xy, dtype=torch.float32).reshape(-1, 2)
    fused = torch.as_tensor(fused_xy, dtype=torch.float32).reshape(-1, 2)
    native_set = {(int(round(float(x))), int(round(float(y)))) for x, y in native.tolist()}
    fused_set = {(int(round(float(x))), int(round(float(y)))) for x, y in fused.tolist()}
    intersection = native_set.intersection(fused_set)
    native_count = len(native_set)
    fused_count = len(fused_set)
    return {
        "native_count": int(native_count),
        "fused_count": int(fused_count),
        "intersection_count": int(len(intersection)),
        "native_overlap_fraction": float(len(intersection) / native_count) if native_count else 0.0,
        "fused_overlap_fraction": float(len(intersection) / fused_count) if fused_count else 0.0,
    }

