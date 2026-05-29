from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def project_points(
    points_world: np.ndarray,
    pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
    *,
    width: int,
    height: int,
    min_depth: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world-space points into an image with a world-to-camera pose."""

    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    K = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=bool)
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    camera = (pose @ homog.T).T[:, :3]
    depth = camera[:, 2]
    projected_h = (K @ camera.T).T
    xy = projected_h[:, :2] / np.maximum(projected_h[:, 2:3], min_depth)
    valid = (
        np.isfinite(xy).all(axis=1)
        & np.isfinite(depth)
        & (depth > float(min_depth))
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < float(width))
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < float(height))
    )
    return xy.astype(np.float32), valid.astype(bool)


def pose_error_cm_deg(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> tuple[float, float]:
    """Return camera-center translation error in cm and rotation error in degrees."""

    pred = np.asarray(pred_w2c, dtype=np.float64).reshape(4, 4)
    gt = np.asarray(gt_w2c, dtype=np.float64).reshape(4, 4)
    pred_r = pred[:3, :3]
    gt_r = gt[:3, :3]
    pred_c = -pred_r.T @ pred[:3, 3]
    gt_c = -gt_r.T @ gt[:3, 3]
    te_cm = float(np.linalg.norm(pred_c - gt_c) * 100.0)
    relative = pred_r @ gt_r.T
    cos_angle = float((np.trace(relative) - 1.0) * 0.5)
    re_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
    return te_cm, re_deg


def offset_camera_center(pose_w2c: np.ndarray, offset_world_m: np.ndarray) -> np.ndarray:
    """Translate a world-to-camera pose by moving its camera center in world space."""

    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4).copy()
    offset = np.asarray(offset_world_m, dtype=np.float64).reshape(3)
    rotation = pose[:3, :3]
    center = -rotation.T @ pose[:3, 3]
    shifted_center = center + offset
    pose[:3, 3] = -rotation @ shifted_center
    return pose.astype(np.float32)


def _feature_array(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("feature map must have shape [C, H, W]")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_uint8(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape, dtype=np.uint8)
    lo = float(array[finite].min())
    hi = float(array[finite].max())
    if hi - lo < 1e-8:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (np.nan_to_num(array, nan=lo) - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)


def feature_norm_map(features: np.ndarray) -> np.ndarray:
    """Return a finite per-pixel feature L2 norm map."""

    array = _feature_array(features)
    return np.linalg.norm(array, axis=0).astype(np.float32)


def cosine_similarity_map(query_features: np.ndarray, reference_features: np.ndarray) -> np.ndarray:
    """Return per-pixel cosine similarity for two aligned feature maps."""

    query = _feature_array(query_features)
    reference = _feature_array(reference_features)
    if query.shape != reference.shape:
        raise ValueError("feature maps must have the same [C, H, W] shape")
    numerator = (query * reference).sum(axis=0)
    denominator = np.linalg.norm(query, axis=0) * np.linalg.norm(reference, axis=0)
    cosine = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-8)
    return np.clip(cosine, -1.0, 1.0).astype(np.float32)


def feature_pair_pca_rgb(
    query_features: np.ndarray,
    reference_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Visualize two feature maps with a shared PCA basis as RGB uint8 images."""

    query = _feature_array(query_features)
    reference = _feature_array(reference_features)
    if query.shape[0] != reference.shape[0]:
        raise ValueError("feature maps must have the same channel count")
    q_flat = query.reshape(query.shape[0], -1).T.astype(np.float64)
    r_flat = reference.reshape(reference.shape[0], -1).T.astype(np.float64)
    combined = np.concatenate([q_flat, r_flat], axis=0)
    mean = combined.mean(axis=0, keepdims=True)
    centered = combined - mean
    if centered.shape[0] < 2 or centered.shape[1] < 1:
        projected = np.zeros((combined.shape[0], 3), dtype=np.float64)
    else:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[: min(3, vt.shape[0])].T
        projected = centered @ basis
        if projected.shape[1] < 3:
            projected = np.pad(projected, ((0, 0), (0, 3 - projected.shape[1])), mode="constant")
    rgb_channels = []
    for channel in range(3):
        rgb_channels.append(_normalize_uint8(projected[:, channel]))
    rgb = np.stack(rgb_channels, axis=1)
    q_count = q_flat.shape[0]
    q_rgb = rgb[:q_count].reshape(query.shape[1], query.shape[2], 3)
    r_rgb = rgb[q_count:].reshape(reference.shape[1], reference.shape[2], 3)
    return q_rgb.astype(np.uint8), r_rgb.astype(np.uint8)


def summarize_match_quality(
    *,
    reprojection_errors_px: np.ndarray,
    solver_inlier_mask: np.ndarray | None = None,
    good_px_threshold: float = 5.0,
) -> dict[str, Any]:
    """Summarize whether matches are geometrically correct under GT pose."""

    errors = np.asarray(reprojection_errors_px, dtype=np.float64).reshape(-1)
    finite = np.isfinite(errors)
    good = finite & (errors <= float(good_px_threshold))
    if solver_inlier_mask is None:
        inliers = np.zeros_like(good, dtype=bool)
    else:
        inliers = np.asarray(solver_inlier_mask, dtype=bool).reshape(-1)
        if inliers.shape[0] != good.shape[0]:
            raise ValueError("solver_inlier_mask must match reprojection_errors_px")
    count = int(errors.shape[0])
    good_count = int(good.sum())
    inlier_count = int(inliers.sum())
    return {
        "match_count": count,
        "finite_reprojection_count": int(finite.sum()),
        "gt_good_count": good_count,
        "gt_bad_count": int(count - good_count),
        "solver_inlier_count": inlier_count,
        "solver_inlier_gt_good_count": int((inliers & good).sum()),
        "solver_inlier_gt_bad_count": int((inliers & ~good).sum()),
        "gt_good_ratio": float(good_count / count) if count else 0.0,
        "solver_inlier_ratio": float(inlier_count / count) if count else 0.0,
        "median_reprojection_error_px": (
            float(np.median(errors[finite])) if finite.any() else None
        ),
        "p90_reprojection_error_px": (
            float(np.percentile(errors[finite], 90)) if finite.any() else None
        ),
        "good_px_threshold": float(good_px_threshold),
    }


def _as_rgb(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255.0).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _line_color(is_good: bool, is_inlier: bool) -> tuple[int, int, int]:
    if is_good and is_inlier:
        return (0, 170, 0)
    if is_good:
        return (40, 120, 255)
    if is_inlier:
        return (255, 120, 0)
    return (210, 40, 40)


def _sample_spatially(indices: np.ndarray, query_xy: np.ndarray, count: int) -> np.ndarray:
    if indices.shape[0] <= count:
        return indices.astype(np.int64)
    xy = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    order = np.lexsort((xy[indices, 0], xy[indices, 1]))
    sorted_indices = indices[order]
    take = np.linspace(0, sorted_indices.shape[0] - 1, num=int(count), dtype=np.int64)
    return sorted_indices[take].astype(np.int64)


def _select_draw_indices(
    gt_good_mask: np.ndarray,
    solver_inlier_mask: np.ndarray,
    query_xy: np.ndarray,
    *,
    max_draw: int,
) -> np.ndarray:
    good = np.asarray(gt_good_mask, dtype=bool).reshape(-1)
    inliers = np.asarray(solver_inlier_mask, dtype=bool).reshape(-1)
    count = min(good.shape[0], inliers.shape[0], np.asarray(query_xy).reshape(-1, 2).shape[0])
    if count <= int(max_draw):
        return np.arange(count, dtype=np.int64)
    good = good[:count]
    inliers = inliers[:count]
    qxy = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)[:count]
    bucket_defs = [
        ((~good) & (~inliers), 0.30),  # draw noisy context first
        ((~good) & inliers, 0.15),
        (good & (~inliers), 0.15),
        (good & inliers, 0.40),  # draw geometrically useful matches last
    ]
    buckets: list[tuple[np.ndarray, float, int]] = []
    for mask, weight in bucket_defs:
        ids = np.flatnonzero(mask)
        if ids.size:
            buckets.append((ids, float(weight), 0))
    if not buckets:
        return np.arange(min(count, int(max_draw)), dtype=np.int64)

    quotas = [min(ids.size, max(1, int(round(float(max_draw) * weight)))) for ids, weight, _ in buckets]
    while sum(quotas) > int(max_draw):
        reducible = [idx for idx, quota in enumerate(quotas) if quota > 1]
        if not reducible:
            break
        idx = max(reducible, key=lambda i: quotas[i])
        quotas[idx] -= 1
    while sum(quotas) < int(max_draw):
        expandable = [idx for idx, (ids, _weight, _unused) in enumerate(buckets) if quotas[idx] < ids.size]
        if not expandable:
            break
        idx = max(expandable, key=lambda i: buckets[i][0].size - quotas[i])
        quotas[idx] += 1

    selected = [
        _sample_spatially(ids, qxy, quota)
        for (ids, _weight, _unused), quota in zip(buckets, quotas)
        if quota > 0
    ]
    return np.concatenate(selected).astype(np.int64) if selected else np.empty(0, dtype=np.int64)


def draw_match_canvas(
    query_image: Image.Image | np.ndarray,
    reference_image: Image.Image | np.ndarray,
    *,
    query_xy: np.ndarray,
    reference_xy: np.ndarray,
    gt_good_mask: np.ndarray,
    solver_inlier_mask: np.ndarray | None,
    output_path: str | Path,
    max_draw: int = 300,
) -> Path:
    """Draw side-by-side query/reference matches.

    Color convention:
    green = GT-good and solver inlier, blue = GT-good but solver outlier,
    orange = GT-bad but solver inlier, red = GT-bad and solver outlier.
    """

    query = _as_rgb(query_image)
    reference = _as_rgb(reference_image)
    if reference.size != query.size:
        reference = reference.resize(query.size)
    qxy = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    rxy = np.asarray(reference_xy, dtype=np.float32).reshape(-1, 2)
    good = np.asarray(gt_good_mask, dtype=bool).reshape(-1)
    if solver_inlier_mask is None:
        inliers = np.zeros_like(good, dtype=bool)
    else:
        inliers = np.asarray(solver_inlier_mask, dtype=bool).reshape(-1)
    count = min(qxy.shape[0], rxy.shape[0], good.shape[0], inliers.shape[0])
    if count == 0:
        count = min(qxy.shape[0], rxy.shape[0])
        good = np.zeros(count, dtype=bool)
        inliers = np.zeros(count, dtype=bool)
    qxy = qxy[:count]
    rxy = rxy[:count]
    good = good[:count]
    inliers = inliers[:count]

    canvas = Image.new("RGB", (query.width + reference.width, query.height), "white")
    canvas.paste(query, (0, 0))
    canvas.paste(reference, (query.width, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    order = _select_draw_indices(good, inliers, qxy, max_draw=int(max_draw))
    for idx in order:
        x1, y1 = qxy[int(idx)]
        x2, y2 = rxy[int(idx)]
        color = _line_color(bool(good[int(idx)]), bool(inliers[int(idx)]))
        rgba = (*color, 150)
        draw.line((float(x1), float(y1), float(x2) + query.width, float(y2)), fill=rgba, width=1)
        draw.ellipse((float(x1) - 2, float(y1) - 2, float(x1) + 2, float(y1) + 2), fill=(*color, 220))
        draw.ellipse(
            (
                float(x2) + query.width - 2,
                float(y2) - 2,
                float(x2) + query.width + 2,
                float(y2) + 2,
            ),
            fill=(*color, 220),
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)
    return output


def write_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output
