from __future__ import annotations

from typing import Any

import torch


def _as_points(values: torch.Tensor | Any, *, dims: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1, dims).cpu()
    if tensor.numel() == 0:
        return torch.empty((0, dims), dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def _as_weights(weights: torch.Tensor | Any | None, count: int) -> torch.Tensor:
    if weights is None:
        return torch.ones((count,), dtype=torch.float32)
    tensor = torch.as_tensor(weights, dtype=torch.float32).reshape(-1).cpu()
    if tensor.shape[0] != int(count):
        raise ValueError("weights must have one value per point")
    return tensor.clamp_min(0.0)


def _weighted_covariance(points: torch.Tensor, weights: torch.Tensor, eps: float) -> torch.Tensor:
    if points.shape[0] == 0:
        return torch.zeros((points.shape[1], points.shape[1]), dtype=torch.float32)
    denom = weights.sum().clamp_min(float(eps))
    mean = (points * weights[:, None]).sum(dim=0) / denom
    centered = points - mean
    return (centered * weights[:, None]).T @ centered / denom


def _eig_summary(matrix: torch.Tensor, eps: float) -> dict[str, float]:
    if matrix.numel() == 0:
        return {
            "trace": 0.0,
            "logdet": 0.0,
            "condition": 0.0,
            "min_eigenvalue": 0.0,
        }
    eigvals = torch.linalg.eigvalsh(matrix.float()).clamp_min(0.0)
    trace = float(eigvals.sum().item())
    shifted = eigvals + float(eps)
    logdet = float(torch.log(shifted).sum().item())
    condition = float((shifted.max() / shifted.min().clamp_min(float(eps))).item()) if eigvals.numel() else 0.0
    return {
        "trace": trace,
        "logdet": logdet,
        "condition": condition,
        "min_eigenvalue": float(eigvals.min().item()) if eigvals.numel() else 0.0,
    }


def _mean_pairwise_distance(points: torch.Tensor) -> float:
    if points.shape[0] < 2:
        return 0.0
    distances = torch.pdist(points.float(), p=2)
    return float(distances.mean().item()) if distances.numel() else 0.0


def pose_information_summary(
    xy: torch.Tensor | Any,
    xyz: torch.Tensor | Any | None = None,
    *,
    weights: torch.Tensor | Any | None = None,
    eps: float = 1e-6,
) -> dict[str, float | int]:
    """Summarize a correspondence set with a stable PnP-solvability surrogate.

    This is intentionally not a differentiable PnP replacement. It measures
    whether selected correspondences have enough 2D and 3D spread to form
    well-conditioned minimal sets.
    """

    xy_tensor = _as_points(xy, dims=2, name="xy")
    count = int(xy_tensor.shape[0])
    if xyz is None:
        xyz_tensor = torch.zeros((count, 3), dtype=torch.float32)
        has_xyz = False
    else:
        xyz_tensor = _as_points(xyz, dims=3, name="xyz")
        if xyz_tensor.shape[0] != count:
            raise ValueError("xyz must have one row per xy point")
        has_xyz = True
    weight_tensor = _as_weights(weights, count)
    if count == 0 or float(weight_tensor.sum().item()) <= float(eps):
        return {
            "point_count": count,
            "trace_H": 0.0,
            "logdet_H": 0.0,
            "condition_number_H": 0.0,
            "min_eigenvalue_2d": 0.0,
            "min_eigenvalue_3d": 0.0,
            "spatial_spread_2d": 0.0,
            "spatial_spread_3d": 0.0,
            "depth_spread": 0.0,
            "has_xyz": int(has_xyz),
        }
    cov_xy = _weighted_covariance(xy_tensor, weight_tensor, eps)
    cov_xyz = _weighted_covariance(xyz_tensor, weight_tensor, eps) if has_xyz else torch.zeros((3, 3))
    xy_stats = _eig_summary(cov_xy, eps)
    xyz_stats = _eig_summary(cov_xyz, eps)
    trace_h = xy_stats["trace"] + xyz_stats["trace"]
    logdet_h = xy_stats["logdet"] + (xyz_stats["logdet"] if has_xyz else 0.0)
    condition_h = max(xy_stats["condition"], xyz_stats["condition"] if has_xyz else 0.0)
    return {
        "point_count": count,
        "trace_H": float(trace_h),
        "logdet_H": float(logdet_h),
        "condition_number_H": float(condition_h),
        "min_eigenvalue_2d": float(xy_stats["min_eigenvalue"]),
        "min_eigenvalue_3d": float(xyz_stats["min_eigenvalue"]) if has_xyz else 0.0,
        "spatial_spread_2d": _mean_pairwise_distance(xy_tensor),
        "spatial_spread_3d": _mean_pairwise_distance(xyz_tensor) if has_xyz else 0.0,
        "depth_spread": float(xyz_tensor[:, 2].std(unbiased=False).item()) if has_xyz and count else 0.0,
        "has_xyz": int(has_xyz),
    }

