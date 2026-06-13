from __future__ import annotations

from typing import Any, Mapping

import torch


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if torch.isfinite(torch.tensor(out)).item() else None


def _point_yx(row: Mapping[str, Any]) -> tuple[float, float] | None:
    point = row.get("keypoint_yx", row.get("projected_yx", row.get("yx")))
    if point is not None:
        tensor = torch.as_tensor(point, dtype=torch.float32).reshape(-1)
        if int(tensor.numel()) >= 2:
            return float(tensor[0].item()), float(tensor[1].item())
    point = row.get("keypoint_xy", row.get("projected_xy", row.get("xy")))
    if point is not None:
        tensor = torch.as_tensor(point, dtype=torch.float32).reshape(-1)
        if int(tensor.numel()) >= 2:
            return float(tensor[1].item()), float(tensor[0].item())
    return None


def _int_value(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def build_online_stdloc_projection_entry(
    *,
    gaussian_xyz: torch.Tensor,
    sampled_idx: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsic: torch.Tensor,
    height: int,
    width: int,
    image_id: str,
    render_visible_mask: torch.Tensor | None = None,
    stable_mask: torch.Tensor | None = None,
    opacity: torch.Tensor | None = None,
    max_points_per_image: int = 0,
    near_depth: float = 1e-6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a STDLoc-style detector target from the current camera online.

    The function mirrors STDLoc's detector target semantics: project the current
    sampled landmark set, keep render-visible and mask-valid projections, and
    rasterization happens later in the detector loss path.
    """

    h = int(height)
    w = int(width)
    sampled = torch.as_tensor(sampled_idx, dtype=torch.long, device=gaussian_xyz.device).reshape(-1)
    xyz = torch.as_tensor(gaussian_xyz, dtype=torch.float32, device=gaussian_xyz.device)
    if int(sampled.numel()) == 0:
        empty = torch.empty((0, 2), dtype=torch.float32)
        return (
            {
                "gaussian_ids": torch.empty((0,), dtype=torch.long),
                "keypoint_yx": empty,
                "support_weights": torch.empty((0,), dtype=torch.float32),
                "positive_count": 0,
                "height": h,
                "width": w,
                "target_role": "online_stdloc_visibility_projection",
            },
            {
                "image_id": str(image_id),
                "sampled_projection_count": 0,
                "kept_projection_count": 0,
                "dropped_render_invisible_count": 0,
                "mask_invalid_projection_count": 0,
                "dropped_behind_count": 0,
                "dropped_out_of_frame_count": 0,
            },
        )

    visible_local = torch.ones(int(sampled.numel()), dtype=torch.bool, device=sampled.device)
    if render_visible_mask is not None:
        raw_visible = torch.as_tensor(render_visible_mask, dtype=torch.bool, device=sampled.device).reshape(-1)
        if int(sampled.max().item()) >= int(raw_visible.numel()):
            raise ValueError("render_visible_mask is smaller than sampled landmark ids")
        visible_local = raw_visible.index_select(0, sampled)
    active_sampled = sampled[visible_local]
    dropped_render_invisible = int(sampled.numel() - active_sampled.numel())
    sampled_xyz = xyz.index_select(0, active_sampled) if int(active_sampled.numel()) else xyz.new_empty((0, 3))
    ones = torch.ones((int(sampled_xyz.shape[0]), 1), dtype=torch.float32, device=sampled_xyz.device)
    homo = torch.cat((sampled_xyz, ones), dim=1)
    w2c = torch.as_tensor(world_to_camera, dtype=torch.float32, device=sampled_xyz.device)
    k = torch.as_tensor(intrinsic, dtype=torch.float32, device=sampled_xyz.device)
    camera_xyz = (w2c @ homo.T).T[:, :3]
    depth = camera_xyz[:, 2]
    positive_depth = depth > float(near_depth)
    uvw = (k @ camera_xyz.T).T
    uv = uvw[:, :2] / depth.clamp_min(float(near_depth))[:, None]
    in_frame = (
        positive_depth
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(w))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(h))
    )
    projected = torch.where(in_frame)[0]
    mask_valid = torch.ones(int(projected.numel()), dtype=torch.bool, device=projected.device)
    if stable_mask is not None and int(projected.numel()) > 0:
        mask = torch.as_tensor(stable_mask, dtype=torch.bool, device=projected.device)
        xy = uv.index_select(0, projected)
        yi = xy[:, 1].long().clamp(0, h - 1)
        xi = xy[:, 0].long().clamp(0, w - 1)
        mask_valid = mask[yi, xi]
    kept_local = projected[mask_valid]
    kept_ids = active_sampled.index_select(0, kept_local).detach().cpu().long()
    yx = torch.empty((int(kept_local.numel()), 2), dtype=torch.float32, device=sampled_xyz.device)
    if int(kept_local.numel()) > 0:
        valid_uv = uv.index_select(0, kept_local)
        yx[:, 0] = valid_uv[:, 1]
        yx[:, 1] = valid_uv[:, 0]
    if opacity is None:
        weights = torch.ones(int(kept_ids.numel()), dtype=torch.float32, device=sampled_xyz.device)
    else:
        opa = torch.as_tensor(opacity, dtype=torch.float32, device=sampled_xyz.device).reshape(-1)
        weights = opa.index_select(0, kept_ids.to(device=sampled_xyz.device)).reshape(-1).clamp(0.0, 1.0)

    cap = int(max_points_per_image)
    dropped_by_cap = 0
    if cap > 0 and int(kept_ids.numel()) > cap:
        order = torch.argsort(-weights, stable=True)[:cap]
        dropped_by_cap = int(kept_ids.numel() - cap)
        kept_ids = kept_ids.index_select(0, order.cpu())
        yx = yx.index_select(0, order)
        weights = weights.index_select(0, order)

    entry = {
        "gaussian_ids": kept_ids.cpu(),
        "keypoint_yx": yx.detach().cpu(),
        "support_weights": weights.detach().cpu(),
        "positive_count": int(kept_ids.numel()),
        "height": h,
        "width": w,
        "target_role": "online_stdloc_visibility_projection",
    }
    metrics = {
        "image_id": str(image_id),
        "sampled_projection_count": int(sampled.numel()),
        "render_visible_projection_count": int(active_sampled.numel()),
        "projected_in_frame_count": int(projected.numel()),
        "kept_projection_count": int(kept_ids.numel()),
        "dropped_render_invisible_count": int(dropped_render_invisible),
        "mask_invalid_projection_count": int((~mask_valid).sum().item()),
        "dropped_behind_count": int((~positive_depth).sum().item()),
        "dropped_out_of_frame_count": int((positive_depth & ~in_frame).sum().item()),
        "dropped_by_image_cap_count": int(dropped_by_cap),
    }
    return entry, metrics


def attach_solver_residuals_to_entry(
    entry: Mapping[str, Any],
    impact: Mapping[str, Any] | None,
    *,
    image_id: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Attach detector positive/negative residual points from solver feedback."""

    out = dict(entry)
    h = int(out.get("height", 0))
    w = int(out.get("width", 0))
    impact = impact or {}
    detector_positive = impact.get("detector_positive", {}) if isinstance(impact, Mapping) else {}
    detector_negative = impact.get("detector_negative", {}) if isinstance(impact, Mapping) else {}
    if not isinstance(detector_positive, Mapping):
        detector_positive = {}
    if not isinstance(detector_negative, Mapping):
        detector_negative = {}

    def collect(rows: Any) -> tuple[list[int], list[list[float]], list[float], int]:
        gids: list[int] = []
        yx_rows: list[list[float]] = []
        weights: list[float] = []
        out_of_bounds = 0
        if not isinstance(rows, (list, tuple)):
            return gids, yx_rows, weights, out_of_bounds
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            point = _point_yx(row)
            if point is None:
                continue
            y, x = point
            if not (0.0 <= y < float(h) and 0.0 <= x < float(w)):
                out_of_bounds += 1
                continue
            weight = _finite_float(row.get("weight", row.get("risk", row.get("support", 1.0))))
            if weight is None or weight <= 0.0:
                continue
            gids.append(_int_value(row.get("gaussian_id", row.get("landmark_id", -1))))
            yx_rows.append([float(y), float(x)])
            weights.append(float(weight))
        return gids, yx_rows, weights, out_of_bounds

    pos_gids, pos_yx, pos_weights, pos_oob = collect(detector_positive.get(str(image_id), []))
    neg_gids, neg_yx, neg_weights, neg_oob = collect(detector_negative.get(str(image_id), []))
    out.update(
        {
            "solver_positive_gaussian_ids": torch.tensor(pos_gids, dtype=torch.long),
            "solver_positive_keypoint_yx": torch.tensor(pos_yx, dtype=torch.float32).reshape(-1, 2),
            "solver_positive_weights": torch.tensor(pos_weights, dtype=torch.float32),
            "solver_positive_count": int(len(pos_gids)),
            "negative_gaussian_ids": torch.tensor(neg_gids, dtype=torch.long),
            "negative_keypoint_yx": torch.tensor(neg_yx, dtype=torch.float32).reshape(-1, 2),
            "negative_weights": torch.tensor(neg_weights, dtype=torch.float32),
            "negative_count": int(len(neg_gids)),
        }
    )
    return out, {
        "solver_positive_count": int(len(pos_gids)),
        "negative_count": int(len(neg_gids)),
        "out_of_bounds_positive_count": int(pos_oob),
        "out_of_bounds_negative_count": int(neg_oob),
    }


def attach_superpoint_teacher_to_entry(
    entry: Mapping[str, Any],
    superpoint_detection: Mapping[str, Any] | None,
    *,
    height: int,
    width: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Attach SuperPoint sparse detections as a direct detector teacher.

    ULF/STDLoc landmark projections provide scene-specific visibility
    supervision, but they do not preserve the native detector's repeatability
    distribution by themselves. These teacher points are kept separate in the
    entry and are merged only by the detector target rasterizer.
    """

    out = dict(entry)
    if not isinstance(superpoint_detection, Mapping):
        out.update(
            {
                "sp_teacher_keypoint_yx": torch.empty((0, 2), dtype=torch.float32),
                "sp_teacher_weights": torch.empty((0,), dtype=torch.float32),
                "sp_teacher_count": 0,
            }
        )
        return out, {"sp_teacher_count": 0, "sp_teacher_out_of_bounds_count": 0}
    keypoints = torch.as_tensor(superpoint_detection.get("keypoints", []), dtype=torch.float32).reshape(-1, 2)
    scores = torch.as_tensor(superpoint_detection.get("keypoint_scores", []), dtype=torch.float32).reshape(-1)
    if int(scores.numel()) != int(keypoints.shape[0]):
        scores = torch.ones((int(keypoints.shape[0]),), dtype=torch.float32)
    if int(keypoints.numel()) == 0:
        yx = torch.empty((0, 2), dtype=torch.float32)
        weights = torch.empty((0,), dtype=torch.float32)
        out_of_bounds = 0
    else:
        x = keypoints[:, 0]
        y = keypoints[:, 1]
        in_bounds = (x >= 0.0) & (x < float(width)) & (y >= 0.0) & (y < float(height))
        yx = torch.stack((y[in_bounds], x[in_bounds]), dim=1).detach().cpu()
        weights = scores[in_bounds].clamp(0.0, 1.0).detach().cpu()
        out_of_bounds = int((~in_bounds).sum().item())
    out.update(
        {
            "sp_teacher_keypoint_yx": yx,
            "sp_teacher_weights": weights,
            "sp_teacher_count": int(yx.shape[0]),
        }
    )
    return out, {
        "sp_teacher_count": int(yx.shape[0]),
        "sp_teacher_out_of_bounds_count": int(out_of_bounds),
    }
