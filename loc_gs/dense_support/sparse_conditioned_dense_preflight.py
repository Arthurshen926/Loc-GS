from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

try:  # SciPy is present in the project env; keep a small fallback for tests/tools.
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - exercised only in stripped-down envs.
    cKDTree = None


@dataclass(frozen=True)
class ImageGeometry:
    """Explicit coordinate mapping between image and feature-map frames."""

    width: int
    height: int
    feature_width: int
    feature_height: int

    def __post_init__(self) -> None:
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("ImageGeometry width and height must be positive")
        if int(self.feature_width) <= 0 or int(self.feature_height) <= 0:
            raise ValueError("ImageGeometry feature dimensions must be positive")

    @property
    def scale_x(self) -> float:
        return float(self.feature_width) / float(self.width)

    @property
    def scale_y(self) -> float:
        return float(self.feature_height) / float(self.height)

    def full_to_feature_xy(self, xy: np.ndarray) -> np.ndarray:
        coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2).copy()
        coords[:, 0] *= self.scale_x
        coords[:, 1] *= self.scale_y
        return coords


@dataclass(frozen=True)
class SLCDPThresholds:
    min_sparse_inliers: int = 40
    min_sparse_inlier_ratio: float = 0.0
    min_visible_landmarks: int = 16
    min_visibility_ratio: float = 0.35
    max_depth_rel_error: float = 0.15
    max_depth_abs_error_m: float = 0.25
    min_alpha: float = 0.05
    min_feature_cosine: float = 0.08
    min_query_to_render_max_cosine: float = 0.28
    min_coarse_mnn_count: int = 80
    min_grid_cells: int = 4
    grid_rows: int = 4
    grid_cols: int = 4
    local_patch_radius: int = 1
    min_local_feature_variance: float = 0.0


@dataclass(frozen=True)
class SLCDPRepairSearchConfig:
    translation_steps_m: tuple[float, ...] = (0.05, 0.10, 0.20)
    rotation_steps_deg: tuple[float, ...] = (0.25, 0.50, 1.00)
    max_candidates: int = 37
    min_score_gain: float = 0.05
    translation_penalty_per_m: float = 0.0


@dataclass(frozen=True)
class DenseTransitionPolicy:
    max_reprojection_error_px: float = 8.0
    min_retained_ratio: float = 0.90
    weak_min_retained_ratio: float = 0.70
    max_translation_delta_m: float = 0.35
    weak_max_translation_delta_m: float = 10.0
    max_rotation_delta_deg: float = 5.0
    line_search_fractions: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)
    strong_sparse_inlier_count: int = 80


def _as_array(value: Any, *, shape_tail: tuple[int, ...] | None = None, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape_tail is not None and array.shape[-len(shape_tail) :] != shape_tail:
        raise ValueError(f"{name} must have trailing shape {shape_tail}")
    return array


def _coerce_image_geometry(value: ImageGeometry | Mapping[str, Any] | None) -> ImageGeometry | None:
    if value is None:
        return None
    if isinstance(value, ImageGeometry):
        return value
    mapping = dict(value)
    return ImageGeometry(
        width=int(mapping["width"]),
        height=int(mapping["height"]),
        feature_width=int(mapping["feature_width"]),
        feature_height=int(mapping["feature_height"]),
    )


def _select_indices(indices: Sequence[int] | np.ndarray | None, count: int) -> np.ndarray:
    if indices is None:
        return np.arange(count, dtype=np.int64)
    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    return selected[(selected >= 0) & (selected < count)]


def _project_points_with_depth(
    points_world: np.ndarray,
    pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    K = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64), np.empty((0,), dtype=bool)
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    camera = (pose @ homog.T).T[:, :3]
    depth = camera[:, 2]
    projected_h = (K @ camera.T).T
    xy = projected_h[:, :2] / np.maximum(projected_h[:, 2:3], 1e-8)
    valid = (
        np.isfinite(xy).all(axis=1)
        & np.isfinite(depth)
        & (depth > 1e-6)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] < float(width))
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] < float(height))
    )
    return xy, depth, valid


def _gaussian_screen_radius_px(
    gaussian_scale_world: np.ndarray | None,
    gaussian_depth: np.ndarray,
    intrinsic: np.ndarray,
    *,
    start: int,
    stop: int,
    radius_scale: float,
    max_radius_px: float,
) -> np.ndarray:
    if gaussian_scale_world is None or float(radius_scale) <= 0.0:
        return np.zeros((int(stop) - int(start),), dtype=np.float64)
    scales_all = np.asarray(gaussian_scale_world, dtype=np.float64)
    if scales_all.shape[0] < int(stop):
        raise ValueError("gaussian_scale_world must have at least as many rows as gaussian_xyz")
    scales = scales_all[int(start) : int(stop)].reshape(int(stop) - int(start), -1)
    world_radius = np.nanmax(np.abs(scales), axis=1)
    focal = float(max(abs(float(intrinsic[0, 0])), abs(float(intrinsic[1, 1]))))
    radius = float(radius_scale) * focal * world_radius / np.maximum(np.asarray(gaussian_depth, dtype=np.float64), 1e-6)
    return np.clip(np.nan_to_num(radius, nan=0.0, posinf=float(max_radius_px), neginf=0.0), 0.0, float(max_radius_px))


def _find_sparse_ray_conflicts_projected(
    *,
    projected_xy: np.ndarray,
    gaussian_depth: np.ndarray,
    gaussian_valid: np.ndarray,
    sparse_xy: np.ndarray,
    sparse_depth: np.ndarray,
    radius_px: float,
    footprint_radius: np.ndarray | None = None,
    depth_margin_m: float,
    active_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find projected Gaussians that are closer than sparse landmarks on nearby rays.

    The old implementation formed a dense [num_gaussians, num_sparse_rays]
    distance matrix. Cambridge scenes can have many Gaussians and hundreds of
    sparse inliers, so the same predicate is evaluated here through local
    radius queries around sparse rays.
    """

    xy = np.asarray(projected_xy, dtype=np.float64).reshape(-1, 2)
    depth = np.asarray(gaussian_depth, dtype=np.float64).reshape(-1)
    valid = np.asarray(gaussian_valid, dtype=bool).reshape(-1)
    rays_xy = np.asarray(sparse_xy, dtype=np.float64).reshape(-1, 2)
    rays_depth = np.asarray(sparse_depth, dtype=np.float64).reshape(-1)
    if depth.shape[0] != xy.shape[0] or valid.shape[0] != xy.shape[0]:
        raise ValueError("projected_xy, gaussian_depth, and gaussian_valid must have matching lengths")
    if rays_depth.shape[0] != rays_xy.shape[0]:
        raise ValueError("sparse_xy and sparse_depth must have matching lengths")
    if footprint_radius is None:
        footprint = np.zeros((xy.shape[0],), dtype=np.float64)
    else:
        footprint = np.asarray(footprint_radius, dtype=np.float64).reshape(-1)
        if footprint.shape[0] != xy.shape[0]:
            raise ValueError("footprint_radius must match projected_xy length")
    active = np.ones((xy.shape[0],), dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool).reshape(-1)
    if active.shape[0] != xy.shape[0]:
        raise ValueError("active_mask must match projected_xy length")

    conflict_mask = np.zeros((xy.shape[0],), dtype=bool)
    conflicted_rays = np.zeros((rays_xy.shape[0],), dtype=bool)
    active_valid = valid & active & np.isfinite(xy).all(axis=1) & np.isfinite(depth)
    active_valid &= np.isfinite(footprint) & (footprint >= 0.0)
    finite_rays = np.isfinite(rays_xy).all(axis=1) & np.isfinite(rays_depth)
    if not active_valid.any() or not finite_rays.any():
        return conflict_mask, conflicted_rays

    valid_indices = np.flatnonzero(active_valid)
    valid_xy = xy[valid_indices]
    max_radius = float(radius_px) + float(np.max(footprint[valid_indices])) if valid_indices.size else float(radius_px)
    if max_radius < 0.0:
        return conflict_mask, conflicted_rays

    if cKDTree is None:
        for ray_id in np.flatnonzero(finite_rays):
            delta = valid_xy - rays_xy[ray_id]
            radius = np.maximum(float(radius_px) ** 2, (float(radius_px) + footprint[valid_indices]) ** 2)
            near = np.einsum("ij,ij->i", delta, delta) <= radius
            too_near = depth[valid_indices] < (rays_depth[ray_id] - float(depth_margin_m))
            local_conflicts = valid_indices[near & too_near]
            if local_conflicts.size:
                conflict_mask[local_conflicts] = True
                conflicted_rays[ray_id] = True
        return conflict_mask, conflicted_rays

    tree = cKDTree(rays_xy[finite_rays])
    finite_ray_ids = np.flatnonzero(finite_rays)
    radii = np.maximum(float(radius_px), float(radius_px) + footprint[valid_indices])
    neighbor_lists = tree.query_ball_point(valid_xy, r=radii)
    for local_gaussian, neighbors in enumerate(neighbor_lists):
        if not neighbors:
            continue
        gaussian_id = valid_indices[int(local_gaussian)]
        ray_ids = finite_ray_ids[np.asarray(neighbors, dtype=np.int64)]
        delta = rays_xy[ray_ids] - xy[gaussian_id]
        radius2 = max(float(radius_px) ** 2, (float(radius_px) + float(footprint[gaussian_id])) ** 2)
        near = np.einsum("ij,ij->i", delta, delta) <= radius2
        too_near = depth[gaussian_id] < (rays_depth[ray_ids] - float(depth_margin_m))
        local_ray_ids = ray_ids[near & too_near]
        if local_ray_ids.size:
            conflict_mask[gaussian_id] = True
            conflicted_rays[local_ray_ids] = True
    return conflict_mask, conflicted_rays


def _sample_hw_nearest(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    h, w = array.shape[-2], array.shape[-1]
    x = np.clip(np.rint(coords[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(coords[:, 1]).astype(np.int64), 0, h - 1)
    if array.ndim == 2:
        return array[y, x]
    if array.ndim == 3:
        return array[:, y, x].T
    raise ValueError("image must have shape [H, W] or [C, H, W]")


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    left = np.asarray(a, dtype=np.float64).reshape(a.shape[0], -1)
    right = np.asarray(b, dtype=np.float64).reshape(b.shape[0], -1)
    numerator = np.sum(left * right, axis=1)
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom > 1e-8)


def _local_feature_variance(features: np.ndarray, xy: np.ndarray, *, radius: int) -> np.ndarray:
    fmap = np.asarray(features, dtype=np.float64)
    if fmap.ndim != 3:
        raise ValueError("features must have shape [C, H, W]")
    h, w = fmap.shape[-2], fmap.shape[-1]
    values: list[float] = []
    for x_raw, y_raw in np.asarray(xy, dtype=np.float64).reshape(-1, 2):
        x = int(np.clip(round(float(x_raw)), 0, w - 1))
        y = int(np.clip(round(float(y_raw)), 0, h - 1))
        x0 = max(0, x - int(radius))
        x1 = min(w, x + int(radius) + 1)
        y0 = max(0, y - int(radius))
        y1 = min(h, y + int(radius) + 1)
        patch = fmap[:, y0:y1, x0:x1].reshape(fmap.shape[0], -1)
        values.append(float(np.mean(np.var(patch, axis=1))) if patch.shape[1] > 1 else 0.0)
    return np.asarray(values, dtype=np.float64)


def _coverage_cells(xy: np.ndarray, *, width: int, height: int, rows: int, cols: int) -> int:
    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if coords.shape[0] == 0:
        return 0
    col = np.clip(np.floor(coords[:, 0] / max(float(width), 1.0) * int(cols)).astype(np.int64), 0, int(cols) - 1)
    row = np.clip(np.floor(coords[:, 1] / max(float(height), 1.0) * int(rows)).astype(np.int64), 0, int(rows) - 1)
    return int(len({(int(r), int(c)) for r, c in zip(row, col)}))


def _depth_class(
    rendered_depth: float,
    landmark_depth: float,
    *,
    in_bounds: bool,
    max_depth_rel_error: float,
    max_depth_abs_error_m: float,
) -> str:
    if not bool(in_bounds):
        return "out_of_bounds"
    if not np.isfinite(rendered_depth) or float(rendered_depth) <= 1e-6:
        return "missing_depth"
    abs_error = abs(float(rendered_depth) - float(landmark_depth))
    rel_error = abs_error / max(abs(float(landmark_depth)), 1e-6)
    if rel_error <= float(max_depth_rel_error) or abs_error <= float(max_depth_abs_error_m):
        return "depth_agree"
    if float(rendered_depth) < float(landmark_depth):
        return "near_occluder"
    return "far_surface_or_hole"


def compute_sparse_ray_depth_diagnostics(
    *,
    sparse_query_xy: np.ndarray,
    sparse_points_world: np.ndarray,
    sparse_pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
    render_depth: np.ndarray,
    sparse_inlier_indices: Sequence[int] | np.ndarray | None = None,
    thresholds: SLCDPThresholds = SLCDPThresholds(),
) -> list[dict[str, Any]]:
    """Classify rendered depth along sparse inlier landmark rays."""

    query_xy_all = _as_array(sparse_query_xy, shape_tail=(2,), name="sparse_query_xy").reshape(-1, 2)
    points_all = _as_array(sparse_points_world, shape_tail=(3,), name="sparse_points_world").reshape(-1, 3)
    if query_xy_all.shape[0] != points_all.shape[0]:
        raise ValueError("sparse_query_xy and sparse_points_world must have matching lengths")
    selected = _select_indices(sparse_inlier_indices, query_xy_all.shape[0])
    depth = np.asarray(render_depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("render_depth must have shape [H, W]")
    height, width = depth.shape
    projected_xy, landmark_depth, in_bounds = _project_points_with_depth(
        points_all[selected],
        np.asarray(sparse_pose_w2c, dtype=np.float64),
        np.asarray(intrinsic, dtype=np.float64),
        width=width,
        height=height,
    )
    rendered_depth = _sample_hw_nearest(depth, projected_xy) if selected.shape[0] else np.empty(0)
    query_xy = query_xy_all[selected]
    rows: list[dict[str, Any]] = []
    for local_id, original_id in enumerate(selected):
        rd = float(rendered_depth[local_id])
        ld = float(landmark_depth[local_id])
        signed = rd - ld if np.isfinite(rd) and np.isfinite(ld) else float("nan")
        rows.append(
            {
                "index": int(original_id),
                "query_x": float(query_xy[local_id, 0]),
                "query_y": float(query_xy[local_id, 1]),
                "projected_x": float(projected_xy[local_id, 0]),
                "projected_y": float(projected_xy[local_id, 1]),
                "in_bounds": bool(in_bounds[local_id]),
                "landmark_depth_m": ld,
                "rendered_depth_m": rd,
                "signed_depth_error_m": float(signed),
                "abs_depth_error_m": float(abs(signed)) if np.isfinite(signed) else float("nan"),
                "depth_class": _depth_class(
                    rd,
                    ld,
                    in_bounds=bool(in_bounds[local_id]),
                    max_depth_rel_error=float(thresholds.max_depth_rel_error),
                    max_depth_abs_error_m=float(thresholds.max_depth_abs_error_m),
                ),
            }
        )
    return rows


def compute_sparse_ray_gaussian_gating_mask(
    *,
    gaussian_xyz: np.ndarray,
    gaussian_scale_world: np.ndarray | None = None,
    gaussian_opacity: np.ndarray | None = None,
    sparse_query_xy: np.ndarray,
    sparse_points_world: np.ndarray,
    sparse_pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
    sparse_inlier_indices: Sequence[int] | np.ndarray | None = None,
    image_size: tuple[int, int],
    radius_px: float = 4.0,
    depth_margin_m: float = 1.0,
    footprint_radius_scale: float = 0.0,
    max_footprint_radius_px: float = 64.0,
    min_conflict_footprint_radius_px: float = 0.0,
    min_conflict_depth_m: float = 0.0,
    max_conflict_depth_m: float | None = None,
    min_opacity: float = 0.0,
    protected_gaussian_indices: Sequence[int] | np.ndarray | None = None,
    chunk_size: int = 65536,
) -> dict[str, Any]:
    """Find near-depth Gaussian conflicts along sparse inlier landmark rays.

    A Gaussian is marked as a conflict if it projects near a sparse inlier ray
    in screen space and lies substantially closer than that sparse landmark.
    This is render-control diagnostic infrastructure; it does not use GT pose.
    """

    gaussians = _as_array(gaussian_xyz, shape_tail=(3,), name="gaussian_xyz").reshape(-1, 3)
    opacity_all = None if gaussian_opacity is None else np.asarray(gaussian_opacity, dtype=np.float64).reshape(-1)
    if opacity_all is not None and opacity_all.shape[0] != gaussians.shape[0]:
        raise ValueError("gaussian_opacity must have one value per Gaussian")
    query_xy_all = _as_array(sparse_query_xy, shape_tail=(2,), name="sparse_query_xy").reshape(-1, 2)
    points_all = _as_array(sparse_points_world, shape_tail=(3,), name="sparse_points_world").reshape(-1, 3)
    if query_xy_all.shape[0] != points_all.shape[0]:
        raise ValueError("sparse_query_xy and sparse_points_world must have matching lengths")
    width, height = int(image_size[0]), int(image_size[1])
    selected = _select_indices(sparse_inlier_indices, query_xy_all.shape[0])
    sparse_xy, sparse_depth, sparse_valid = _project_points_with_depth(
        points_all[selected],
        np.asarray(sparse_pose_w2c, dtype=np.float64),
        np.asarray(intrinsic, dtype=np.float64),
        width=width,
        height=height,
    )
    valid_sparse = sparse_valid & np.isfinite(sparse_depth)
    sparse_xy = sparse_xy[valid_sparse]
    sparse_depth = sparse_depth[valid_sparse]

    keep_mask = np.ones((gaussians.shape[0],), dtype=bool)
    conflict_mask = np.zeros((gaussians.shape[0],), dtype=bool)
    protected = (
        np.empty(0, dtype=np.int64)
        if protected_gaussian_indices is None
        else _select_indices(protected_gaussian_indices, gaussians.shape[0])
    )
    protected_mask = np.zeros((gaussians.shape[0],), dtype=bool)
    protected_mask[protected] = True
    if sparse_xy.shape[0] == 0 or gaussians.shape[0] == 0:
        return {
            "schema": "loc_gs_sparse_ray_gaussian_gating_v1",
            "keep_mask": keep_mask,
            "conflict_mask": conflict_mask,
            "conflict_indices": np.empty(0, dtype=np.int64),
            "conflict_count": 0,
            "kept_count": int(keep_mask.sum()),
            "protected_count": int(protected_mask.sum()),
            "sparse_ray_count": int(sparse_xy.shape[0]),
            "radius_px": float(radius_px),
            "depth_margin_m": float(depth_margin_m),
            "footprint_radius_scale": float(footprint_radius_scale),
            "max_footprint_radius_px": float(max_footprint_radius_px),
            "min_conflict_footprint_radius_px": float(min_conflict_footprint_radius_px),
            "min_conflict_depth_m": float(min_conflict_depth_m),
            "max_conflict_depth_m": None if max_conflict_depth_m is None else float(max_conflict_depth_m),
            "min_opacity": float(min_opacity),
            "diagnostic_only": True,
        }

    pose = np.asarray(sparse_pose_w2c, dtype=np.float64).reshape(4, 4)
    K = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    radius2 = float(radius_px) ** 2
    for start in range(0, gaussians.shape[0], max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), gaussians.shape[0])
        projected_xy, gaussian_depth, gaussian_valid = _project_points_with_depth(
            gaussians[start:stop],
            pose,
            K,
            width=width,
            height=height,
        )
        if not gaussian_valid.any():
            continue
        footprint_radius = _gaussian_screen_radius_px(
            gaussian_scale_world,
            gaussian_depth,
            K,
            start=start,
            stop=stop,
            radius_scale=float(footprint_radius_scale),
            max_radius_px=float(max_footprint_radius_px),
        )
        opacity_ok = np.ones_like(gaussian_valid, dtype=bool) if opacity_all is None else opacity_all[start:stop] >= float(min_opacity)
        artifact_like = opacity_ok & (footprint_radius >= float(min_conflict_footprint_radius_px))
        if float(min_conflict_depth_m) > 0.0:
            artifact_like &= gaussian_depth >= float(min_conflict_depth_m)
        if max_conflict_depth_m is not None and np.isfinite(float(max_conflict_depth_m)):
            artifact_like &= gaussian_depth <= float(max_conflict_depth_m)
        chunk_conflict, _chunk_conflicted_rays = _find_sparse_ray_conflicts_projected(
            projected_xy=projected_xy,
            gaussian_depth=gaussian_depth,
            gaussian_valid=gaussian_valid,
            sparse_xy=sparse_xy,
            sparse_depth=sparse_depth,
            radius_px=float(radius_px),
            footprint_radius=footprint_radius,
            depth_margin_m=float(depth_margin_m),
            active_mask=artifact_like,
        )
        conflict_mask[start:stop] = chunk_conflict
    conflict_mask &= ~protected_mask
    keep_mask &= ~conflict_mask
    conflict_indices = np.flatnonzero(conflict_mask).astype(np.int64)
    return {
        "schema": "loc_gs_sparse_ray_gaussian_gating_v1",
        "keep_mask": keep_mask,
        "conflict_mask": conflict_mask,
        "conflict_indices": conflict_indices,
        "conflict_count": int(conflict_indices.shape[0]),
        "kept_count": int(keep_mask.sum()),
        "protected_count": int(protected_mask.sum()),
        "sparse_ray_count": int(sparse_xy.shape[0]),
        "radius_px": float(radius_px),
        "depth_margin_m": float(depth_margin_m),
        "footprint_radius_scale": float(footprint_radius_scale),
        "max_footprint_radius_px": float(max_footprint_radius_px),
        "min_conflict_footprint_radius_px": float(min_conflict_footprint_radius_px),
        "min_conflict_depth_m": float(min_conflict_depth_m),
        "max_conflict_depth_m": None if max_conflict_depth_m is None else float(max_conflict_depth_m),
        "min_opacity": float(min_opacity),
        "diagnostic_only": True,
    }


def sparse_landmark_conditioned_preflight(
    *,
    sparse_query_xy: np.ndarray,
    sparse_points_world: np.ndarray,
    sparse_pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
    render_depth: np.ndarray,
    query_features: np.ndarray | None = None,
    render_features: np.ndarray | None = None,
    render_alpha: np.ndarray | None = None,
    sparse_inlier_indices: Sequence[int] | np.ndarray | None = None,
    sparse_scores: Sequence[float] | np.ndarray | None = None,
    feature_geometry: ImageGeometry | Mapping[str, Any] | None = None,
    thresholds: SLCDPThresholds = SLCDPThresholds(),
) -> dict[str, Any]:
    """Check whether the dense render at the sparse pose explains sparse PnP inliers.

    This function is diagnostic/ablation infrastructure. It uses sparse inlier
    landmarks as a pre-dense consistency contract, but it does not touch the
    vendored STDLoc evaluator.
    """

    query_xy_all = _as_array(sparse_query_xy, shape_tail=(2,), name="sparse_query_xy").reshape(-1, 2)
    points_all = _as_array(sparse_points_world, shape_tail=(3,), name="sparse_points_world").reshape(-1, 3)
    if query_xy_all.shape[0] != points_all.shape[0]:
        raise ValueError("sparse_query_xy and sparse_points_world must have matching lengths")
    selected = _select_indices(sparse_inlier_indices, query_xy_all.shape[0])
    depth = np.asarray(render_depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("render_depth must have shape [H, W]")
    height, width = depth.shape
    query_xy = query_xy_all[selected]
    points = points_all[selected]
    ray_diagnostics = compute_sparse_ray_depth_diagnostics(
        sparse_query_xy=query_xy_all,
        sparse_points_world=points_all,
        sparse_pose_w2c=sparse_pose_w2c,
        intrinsic=intrinsic,
        render_depth=render_depth,
        sparse_inlier_indices=selected,
        thresholds=thresholds,
    )
    projected_xy = np.asarray([[row["projected_x"], row["projected_y"]] for row in ray_diagnostics], dtype=np.float64)
    landmark_depth = np.asarray([row["landmark_depth_m"] for row in ray_diagnostics], dtype=np.float64)
    in_bounds = np.asarray([row["in_bounds"] for row in ray_diagnostics], dtype=bool)
    rendered_depth = _sample_hw_nearest(depth, projected_xy) if projected_xy.shape[0] else np.empty(0)
    depth_abs_error = np.abs(rendered_depth - landmark_depth)
    depth_rel_error = depth_abs_error / np.maximum(np.abs(landmark_depth), 1e-6)
    depth_ok = (
        in_bounds
        & np.isfinite(rendered_depth)
        & (rendered_depth > 1e-6)
        & (
            (depth_rel_error <= float(thresholds.max_depth_rel_error))
            | (depth_abs_error <= float(thresholds.max_depth_abs_error_m))
        )
    )
    if render_alpha is None:
        alpha_ok = np.ones_like(depth_ok, dtype=bool)
    else:
        alpha = _sample_hw_nearest(np.asarray(render_alpha, dtype=np.float64), projected_xy)
        alpha_ok = np.isfinite(alpha) & (alpha >= float(thresholds.min_alpha))
    visible = depth_ok & alpha_ok
    finite_depth_error = depth_abs_error[np.isfinite(depth_abs_error)]
    signed_depth_error = rendered_depth - landmark_depth
    finite_signed_depth_error = signed_depth_error[np.isfinite(signed_depth_error)]
    near_occluder_count = sum(1 for row in ray_diagnostics if row["depth_class"] == "near_occluder")
    far_surface_count = sum(1 for row in ray_diagnostics if row["depth_class"] == "far_surface_or_hole")
    missing_depth_count = sum(1 for row in ray_diagnostics if row["depth_class"] == "missing_depth")
    weights = np.ones((selected.shape[0],), dtype=np.float64)
    if sparse_scores is not None:
        scores = np.asarray(sparse_scores, dtype=np.float64).reshape(-1)
        if scores.shape[0] == query_xy_all.shape[0]:
            weights = np.maximum(scores[selected], 0.0)
            if float(weights.sum()) <= 1e-8:
                weights = np.ones_like(weights)
    weight_sum = float(weights.sum()) if weights.size else 0.0
    visible_ratio = float(weights[visible].sum() / weight_sum) if weight_sum > 0.0 else 0.0

    feature_cosines: np.ndarray | None = None
    feature_cosine_median: float | None = None
    local_variance_median: float | None = None
    feature_ok = True
    variance_ok = True
    geometry = _coerce_image_geometry(feature_geometry)
    feature_coordinate_frame = "native_feature_coordinates"
    if query_features is not None and render_features is not None and selected.shape[0] > 0:
        query_feature_xy = query_xy
        render_feature_xy = projected_xy
        if geometry is not None:
            query_feature_xy = geometry.full_to_feature_xy(query_xy)
            render_feature_xy = geometry.full_to_feature_xy(projected_xy)
            feature_coordinate_frame = "scaled_by_image_geometry"
        q_desc = _sample_hw_nearest(np.asarray(query_features, dtype=np.float64), query_feature_xy)
        r_desc = _sample_hw_nearest(np.asarray(render_features, dtype=np.float64), render_feature_xy)
        feature_cosines = _cosine_rows(q_desc, r_desc)
        finite_cos = feature_cosines[np.isfinite(feature_cosines)]
        feature_cosine_median = float(np.median(finite_cos)) if finite_cos.size else None
        feature_ok = feature_cosine_median is not None and feature_cosine_median >= float(thresholds.min_feature_cosine)
        local_variance = _local_feature_variance(
            np.asarray(render_features, dtype=np.float64),
            render_feature_xy,
            radius=int(thresholds.local_patch_radius),
        )
        finite_var = local_variance[np.isfinite(local_variance)]
        local_variance_median = float(np.median(finite_var)) if finite_var.size else None
        variance_ok = (
            local_variance_median is not None
            and local_variance_median >= float(thresholds.min_local_feature_variance)
        )

    coverage_grid_cells = _coverage_cells(
        projected_xy[visible],
        width=width,
        height=height,
        rows=int(thresholds.grid_rows),
        cols=int(thresholds.grid_cols),
    )
    visible_depths = landmark_depth[visible]
    depth_span_m = float(np.max(visible_depths) - np.min(visible_depths)) if visible_depths.size else 0.0
    sparse_inlier_count = int(selected.shape[0])
    sparse_confident = (
        sparse_inlier_count >= int(thresholds.min_sparse_inliers)
        and sparse_inlier_count / max(int(query_xy_all.shape[0]), 1) >= float(thresholds.min_sparse_inlier_ratio)
    )
    failed_checks = {
        "visibility": bool(
            int(visible.sum()) < int(thresholds.min_visible_landmarks)
            or visible_ratio < float(thresholds.min_visibility_ratio)
        ),
        "feature_agreement": bool(not feature_ok),
        "coverage": bool(coverage_grid_cells < int(thresholds.min_grid_cells)),
        "local_feature_variance": bool(not variance_ok),
    }
    passed = not any(failed_checks.values())
    if passed:
        decision = "accept_dense"
    elif sparse_confident:
        decision = "skip_dense_keep_sparse"
    else:
        decision = "retry_sparse_or_patch_dense"

    return {
        "schema": "loc_gs_slcdp_preflight_v1",
        "decision": decision,
        "sparse_inlier_count": sparse_inlier_count,
        "sparse_confident": bool(sparse_confident),
        "in_bounds_count": int(in_bounds.sum()),
        "depth_ok_count": int(depth_ok.sum()),
        "alpha_ok_count": int(alpha_ok.sum()),
        "visible_count": int(visible.sum()),
        "visible_ratio": visible_ratio,
        "median_depth_abs_error_m": (
            float(np.median(finite_depth_error)) if finite_depth_error.size else None
        ),
        "median_signed_depth_error_m": (
            float(np.median(finite_signed_depth_error)) if finite_signed_depth_error.size else None
        ),
        "near_occluder_count": int(near_occluder_count),
        "far_surface_or_hole_count": int(far_surface_count),
        "missing_depth_count": int(missing_depth_count),
        "near_occluder_ratio": float(near_occluder_count / sparse_inlier_count) if sparse_inlier_count else 0.0,
        "far_surface_or_hole_ratio": float(far_surface_count / sparse_inlier_count) if sparse_inlier_count else 0.0,
        "missing_depth_ratio": float(missing_depth_count / sparse_inlier_count) if sparse_inlier_count else 0.0,
        "coverage_grid_cells": int(coverage_grid_cells),
        "depth_span_m": depth_span_m,
        "feature_cosine_median": feature_cosine_median,
        "feature_coordinate_frame": feature_coordinate_frame,
        "local_feature_variance_median": local_variance_median,
        "failed_checks": failed_checks,
        "diagnostic_only": True,
    }


def _camera_center(pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    return -pose[:3, :3].T @ pose[:3, 3]


def _rotation_delta_deg(a_w2c: np.ndarray, b_w2c: np.ndarray) -> float:
    a = np.asarray(a_w2c, dtype=np.float64).reshape(4, 4)[:3, :3]
    b = np.asarray(b_w2c, dtype=np.float64).reshape(4, 4)[:3, :3]
    relative = a @ b.T
    cos_angle = float((np.trace(relative) - 1.0) * 0.5)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cos_angle = float((np.trace(R) - 1.0) * 0.5)
    angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if abs(np.pi - angle) < 1e-5:
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) * 0.5, 0.0))
        axis[0] = np.copysign(axis[0], R[2, 1] - R[1, 2])
        axis[1] = np.copysign(axis[1], R[0, 2] - R[2, 0])
        axis[2] = np.copysign(axis[2], R[1, 0] - R[0, 1])
        norm = float(np.linalg.norm(axis))
        if norm < 1e-8:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis = axis / norm
        return axis * angle
    return np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float64,
    ) * (0.5 * angle / float(np.sin(angle)))


def _so3_exp(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float64) + _skew(vector)
    axis = vector / angle
    K = _skew(axis)
    return np.eye(3, dtype=np.float64) + float(np.sin(angle)) * K + (1.0 - float(np.cos(angle))) * (K @ K)


def interpolate_w2c_poses(sparse_pose_w2c: np.ndarray, dense_pose_w2c: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate two world-to-camera poses in camera-center/SO(3) space."""

    f = float(np.clip(float(fraction), 0.0, 1.0))
    sparse = np.asarray(sparse_pose_w2c, dtype=np.float64).reshape(4, 4)
    dense = np.asarray(dense_pose_w2c, dtype=np.float64).reshape(4, 4)
    sparse_center = _camera_center(sparse)
    dense_center = _camera_center(dense)
    center = (1.0 - f) * sparse_center + f * dense_center
    sparse_rotation = sparse[:3, :3]
    dense_rotation = dense[:3, :3]
    relative = dense_rotation @ sparse_rotation.T
    rotation = _so3_exp(_so3_log(relative) * f) @ sparse_rotation
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation
    out[:3, 3] = -rotation @ center
    return out.astype(np.float32)


def evaluate_dense_step_acceptance(
    *,
    sparse_query_xy: np.ndarray,
    sparse_points_world: np.ndarray,
    sparse_pose_w2c: np.ndarray,
    dense_pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
    sparse_inlier_indices: Sequence[int] | np.ndarray | None = None,
    max_reprojection_error_px: float = 8.0,
    min_retained_ratio: float = 0.70,
    max_translation_delta_m: float = 1.0,
    max_rotation_delta_deg: float = 10.0,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    query_xy_all = _as_array(sparse_query_xy, shape_tail=(2,), name="sparse_query_xy").reshape(-1, 2)
    points_all = _as_array(sparse_points_world, shape_tail=(3,), name="sparse_points_world").reshape(-1, 3)
    if query_xy_all.shape[0] != points_all.shape[0]:
        raise ValueError("sparse_query_xy and sparse_points_world must have matching lengths")
    selected = _select_indices(sparse_inlier_indices, query_xy_all.shape[0])
    query_xy = query_xy_all[selected]
    points = points_all[selected]
    if image_size is None:
        width = int(max(np.ceil(query_xy[:, 0].max(initial=0.0) + 1.0), 1))
        height = int(max(np.ceil(query_xy[:, 1].max(initial=0.0) + 1.0), 1))
    else:
        width, height = int(image_size[0]), int(image_size[1])
    projected_xy, _depth, valid = _project_points_with_depth(
        points,
        np.asarray(dense_pose_w2c, dtype=np.float64),
        np.asarray(intrinsic, dtype=np.float64),
        width=width + 1,
        height=height + 1,
    )
    errors = np.linalg.norm(projected_xy - query_xy, axis=1) if selected.size else np.empty(0)
    retained = valid & np.isfinite(errors) & (errors <= float(max_reprojection_error_px))
    retained_ratio = float(retained.sum() / selected.size) if selected.size else 0.0
    translation_delta = float(
        np.linalg.norm(_camera_center(sparse_pose_w2c) - _camera_center(dense_pose_w2c))
    )
    rotation_delta = _rotation_delta_deg(sparse_pose_w2c, dense_pose_w2c)
    failed_checks = {
        "sparse_inlier_retention": bool(retained_ratio < float(min_retained_ratio)),
        "translation_trust_region": bool(translation_delta > float(max_translation_delta_m)),
        "rotation_trust_region": bool(rotation_delta > float(max_rotation_delta_deg)),
    }
    decision = "accept_dense_update" if not any(failed_checks.values()) else "reject_dense_update_keep_sparse"
    return {
        "schema": "loc_gs_slcdp_step_acceptance_v1",
        "decision": decision,
        "sparse_inlier_count": int(selected.size),
        "retained_sparse_inlier_count": int(retained.sum()),
        "retained_sparse_inlier_ratio": retained_ratio,
        "median_sparse_reprojection_error_px": float(np.median(errors[np.isfinite(errors)])) if np.isfinite(errors).any() else None,
        "translation_delta_m": translation_delta,
        "rotation_delta_deg": rotation_delta,
        "failed_checks": failed_checks,
        "diagnostic_only": True,
    }


def select_sparse_conditioned_dense_transition(
    *,
    sparse_query_xy: np.ndarray,
    sparse_points_world: np.ndarray,
    sparse_pose_w2c: np.ndarray,
    dense_pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
    sparse_inlier_indices: Sequence[int] | np.ndarray | None = None,
    policy: DenseTransitionPolicy = DenseTransitionPolicy(),
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Select a support-consistent sparse-to-dense update.

    The dense pose is treated as a proposed update. If the full update violates
    the sparse landmark contract, deterministic line search tries smaller
    fractions. Falling back to the sparse pose is diagnostic/ablation behavior
    and does not modify the vendored STDLoc evaluator.
    """

    fractions: list[float] = []
    for value in policy.line_search_fractions:
        fraction = float(np.clip(float(value), 0.0, 1.0))
        if not any(abs(fraction - existing) < 1e-9 for existing in fractions):
            fractions.append(fraction)
    if not any(abs(fraction) < 1e-9 for fraction in fractions):
        fractions.append(0.0)
    candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_indices = _select_indices(sparse_inlier_indices, np.asarray(sparse_query_xy).reshape(-1, 2).shape[0])
    strong_sparse = int(selected_indices.size) >= int(policy.strong_sparse_inlier_count)
    min_retained_ratio = float(policy.min_retained_ratio if strong_sparse else policy.weak_min_retained_ratio)
    max_translation_delta_m = float(
        policy.max_translation_delta_m
        if strong_sparse
        else max(float(policy.max_translation_delta_m), float(policy.weak_max_translation_delta_m))
    )
    for fraction in fractions:
        candidate_pose = interpolate_w2c_poses(sparse_pose_w2c, dense_pose_w2c, fraction)
        acceptance = evaluate_dense_step_acceptance(
            sparse_query_xy=sparse_query_xy,
            sparse_points_world=sparse_points_world,
            sparse_pose_w2c=sparse_pose_w2c,
            dense_pose_w2c=candidate_pose,
            intrinsic=intrinsic,
            sparse_inlier_indices=sparse_inlier_indices,
            max_reprojection_error_px=float(policy.max_reprojection_error_px),
            min_retained_ratio=min_retained_ratio,
            max_translation_delta_m=max_translation_delta_m,
            max_rotation_delta_deg=float(policy.max_rotation_delta_deg),
            image_size=image_size,
        )
        item = {
            "fraction": float(fraction),
            "pose_w2c": candidate_pose,
            "acceptance": acceptance,
        }
        candidates.append(item)
        if selected is None and acceptance.get("decision") == "accept_dense_update":
            selected = item
    if selected is None:
        selected = candidates[-1]
    selected_fraction = float(selected["fraction"])
    if selected_fraction >= 1.0 - 1e-9:
        decision = "accept_dense_update"
    elif selected_fraction <= 1e-9:
        decision = "reject_dense_keep_sparse"
    else:
        decision = "accept_line_search_dense_update"
    return {
        "schema": "loc_gs_slcdp_transition_control_v1",
        "decision": decision,
        "selected_fraction": selected_fraction,
        "sparse_support_strength": "strong" if strong_sparse else "weak",
        "selected_pose_w2c": np.asarray(selected["pose_w2c"], dtype=np.float32),
        "selected_acceptance": selected["acceptance"],
        "candidates": candidates,
        "policy": {
            "max_reprojection_error_px": float(policy.max_reprojection_error_px),
            "min_retained_ratio": float(policy.min_retained_ratio),
            "weak_min_retained_ratio": float(policy.weak_min_retained_ratio),
            "effective_min_retained_ratio": float(min_retained_ratio),
            "max_translation_delta_m": float(policy.max_translation_delta_m),
            "weak_max_translation_delta_m": float(policy.weak_max_translation_delta_m),
            "effective_max_translation_delta_m": float(max_translation_delta_m),
            "max_rotation_delta_deg": float(policy.max_rotation_delta_deg),
            "line_search_fractions": [float(value) for value in fractions],
            "strong_sparse_inlier_count": int(policy.strong_sparse_inlier_count),
        },
        "diagnostic_only": True,
    }


def _finite_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _soft_ratio_score(value: Any, target: float) -> float | None:
    observed = _finite_float(value, default=None)
    if observed is None:
        return None
    if float(target) <= 1.0e-12:
        return 1.0
    return float(np.clip(float(observed) / float(target), 0.0, 1.0))


def _soft_limit_score(value: Any, limit: float) -> float | None:
    observed = _finite_float(value, default=None)
    if observed is None:
        return None
    if float(limit) <= 1.0e-12:
        return 1.0 if float(observed) <= 1.0e-12 else 0.0
    return float(np.clip(float(limit) / max(float(observed), 1.0e-12), 0.0, 1.0))


def _soft_min_score(values: Sequence[float | None]) -> float:
    available = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not available:
        return 1.0
    return float(np.clip(min(available), 0.0, 1.0))


def _soft_preflight_score(preflight: Mapping[str, Any], thresholds: SLCDPThresholds) -> float:
    if not preflight:
        return 1.0
    decision = str(preflight.get("decision", "")).strip()
    if decision in {"skip_dense_keep_sparse", "retry_sparse_or_patch_dense"}:
        return 0.0
    if decision and decision != "accept_dense":
        return 0.0
    return _soft_min_score(
        [
            _soft_ratio_score(preflight.get("visible_ratio"), float(thresholds.min_visibility_ratio)),
            _soft_ratio_score(preflight.get("feature_cosine_median"), float(thresholds.min_feature_cosine)),
            _soft_ratio_score(preflight.get("local_feature_variance_median"), float(thresholds.min_local_feature_variance)),
            _soft_ratio_score(preflight.get("coverage_grid_cells"), float(thresholds.min_grid_cells)),
            _soft_ratio_score(preflight.get("query_to_render_max_cosine_mean"), float(thresholds.min_query_to_render_max_cosine)),
            _soft_ratio_score(preflight.get("coarse_mnn_count"), float(thresholds.min_coarse_mnn_count)),
        ]
    )


def select_soft_sparse_conditioned_dense_transition(
    *,
    sparse_query_xy: np.ndarray,
    sparse_points_world: np.ndarray,
    sparse_pose_w2c: np.ndarray,
    dense_pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
    sparse_inlier_indices: Sequence[int] | np.ndarray | None = None,
    preflight: Mapping[str, Any] | None = None,
    policy: DenseTransitionPolicy = DenseTransitionPolicy(),
    thresholds: SLCDPThresholds = SLCDPThresholds(),
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Select a continuous sparse-anchored dense update without GT signals.

    This is diagnostic infrastructure for the Soft-SDCG direction. It converts
    sparse-anchor retention, pose trust, and optional SLCDP preflight metrics
    into a dense reliability alpha, then interpolates between sparse and dense
    poses. It does not modify the vendored STDLoc evaluator.
    """

    selected_indices = _select_indices(sparse_inlier_indices, np.asarray(sparse_query_xy).reshape(-1, 2).shape[0])
    strong_sparse = int(selected_indices.size) >= int(policy.strong_sparse_inlier_count)
    min_retained_ratio = float(policy.min_retained_ratio if strong_sparse else policy.weak_min_retained_ratio)
    max_translation_delta_m = float(
        policy.max_translation_delta_m
        if strong_sparse
        else max(float(policy.max_translation_delta_m), float(policy.weak_max_translation_delta_m))
    )
    step_acceptance = evaluate_dense_step_acceptance(
        sparse_query_xy=sparse_query_xy,
        sparse_points_world=sparse_points_world,
        sparse_pose_w2c=sparse_pose_w2c,
        dense_pose_w2c=dense_pose_w2c,
        intrinsic=intrinsic,
        sparse_inlier_indices=sparse_inlier_indices,
        max_reprojection_error_px=float(policy.max_reprojection_error_px),
        min_retained_ratio=min_retained_ratio,
        max_translation_delta_m=max_translation_delta_m,
        max_rotation_delta_deg=float(policy.max_rotation_delta_deg),
        image_size=image_size,
    )
    step_score = _soft_min_score(
        [
            _soft_ratio_score(step_acceptance.get("retained_sparse_inlier_ratio"), min_retained_ratio),
            _soft_limit_score(step_acceptance.get("translation_delta_m"), max_translation_delta_m),
            _soft_limit_score(step_acceptance.get("rotation_delta_deg"), float(policy.max_rotation_delta_deg)),
        ]
    )
    preflight = dict(preflight or {})
    preflight_score = _soft_preflight_score(preflight, thresholds)
    alpha = float(np.clip(min(step_score, preflight_score), 0.0, 1.0))
    selected_pose = interpolate_w2c_poses(sparse_pose_w2c, dense_pose_w2c, alpha)
    if alpha <= 1.0e-9:
        decision = "soft_keep_sparse"
    elif alpha >= 1.0 - 1.0e-9:
        decision = "soft_accept_dense"
    else:
        decision = "soft_sparse_anchored_dense_update"
    return {
        "schema": "loc_gs_soft_sdcg_transition_v1",
        "decision": decision,
        "selected_fraction": alpha,
        "dense_reliability_alpha": alpha,
        "sparse_support_strength": "strong" if strong_sparse else "weak",
        "selected_pose_w2c": selected_pose,
        "step_score": float(step_score),
        "preflight_score": float(preflight_score),
        "step_acceptance": step_acceptance,
        "policy": {
            "max_reprojection_error_px": float(policy.max_reprojection_error_px),
            "min_retained_ratio": float(policy.min_retained_ratio),
            "weak_min_retained_ratio": float(policy.weak_min_retained_ratio),
            "effective_min_retained_ratio": float(min_retained_ratio),
            "max_translation_delta_m": float(policy.max_translation_delta_m),
            "weak_max_translation_delta_m": float(policy.weak_max_translation_delta_m),
            "effective_max_translation_delta_m": float(max_translation_delta_m),
            "max_rotation_delta_deg": float(policy.max_rotation_delta_deg),
            "strong_sparse_inlier_count": int(policy.strong_sparse_inlier_count),
        },
        "thresholds": {
            "min_visibility_ratio": float(thresholds.min_visibility_ratio),
            "min_feature_cosine": float(thresholds.min_feature_cosine),
            "min_local_feature_variance": float(thresholds.min_local_feature_variance),
            "min_grid_cells": int(thresholds.min_grid_cells),
            "min_query_to_render_max_cosine": float(thresholds.min_query_to_render_max_cosine),
            "min_coarse_mnn_count": int(thresholds.min_coarse_mnn_count),
        },
        "diagnostic_only": True,
    }


def _axis_angle_rotation(axis: str, degrees: float) -> np.ndarray:
    radians = np.deg2rad(float(degrees))
    c = float(np.cos(radians))
    s = float(np.sin(radians))
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"unknown axis: {axis}")


def apply_camera_frame_delta(
    pose_w2c: np.ndarray,
    *,
    translation_cam_m: Sequence[float] = (0.0, 0.0, 0.0),
    rotation_deg: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Apply a small camera-frame pose delta to a world-to-camera pose.

    Translations are expressed in the current camera coordinate frame. Rotations
    are Euler deltas around camera x/y/z axes and are left-multiplied onto the
    world-to-camera rotation.
    """

    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    delta_t_cam = np.asarray(translation_cam_m, dtype=np.float64).reshape(3)
    rx, ry, rz = np.asarray(rotation_deg, dtype=np.float64).reshape(3)
    rotation = pose[:3, :3]
    center = -rotation.T @ pose[:3, 3]
    shifted_center = center + rotation.T @ delta_t_cam
    delta_r = _axis_angle_rotation("z", rz) @ _axis_angle_rotation("y", ry) @ _axis_angle_rotation("x", rx)
    new_rotation = delta_r @ rotation
    out = pose.copy()
    out[:3, :3] = new_rotation
    out[:3, 3] = -new_rotation @ shifted_center
    return out.astype(np.float32)


def generate_pose_repair_candidates(
    pose_w2c: np.ndarray,
    config: SLCDPRepairSearchConfig = SLCDPRepairSearchConfig(),
) -> list[dict[str, Any]]:
    """Generate deterministic local pose candidates around a sparse PnP pose."""

    candidates: list[dict[str, Any]] = [
        {
            "label": "base",
            "pose_w2c": np.asarray(pose_w2c, dtype=np.float32).reshape(4, 4),
            "translation_cam_m": (0.0, 0.0, 0.0),
            "rotation_deg": (0.0, 0.0, 0.0),
        }
    ]
    axes = ("x", "y", "z")
    for step in config.translation_steps_m:
        for axis_id, axis in enumerate(axes):
            for sign, sign_label in ((1.0, "+"), (-1.0, "-")):
                delta = [0.0, 0.0, 0.0]
                delta[axis_id] = float(sign) * float(step)
                candidates.append(
                    {
                        "label": f"t{axis}{sign_label}{float(step):.3f}",
                        "pose_w2c": apply_camera_frame_delta(
                            pose_w2c,
                            translation_cam_m=delta,
                            rotation_deg=(0.0, 0.0, 0.0),
                        ),
                        "translation_cam_m": tuple(delta),
                        "rotation_deg": (0.0, 0.0, 0.0),
                    }
                )
    rot_defs = (("pitch", 0), ("yaw", 1), ("roll", 2))
    for step in config.rotation_steps_deg:
        for label, axis_id in rot_defs:
            for sign, sign_label in ((1.0, "+"), (-1.0, "-")):
                delta_r = [0.0, 0.0, 0.0]
                delta_r[axis_id] = float(sign) * float(step)
                candidates.append(
                    {
                        "label": f"{label}{sign_label}{float(step):.3f}",
                        "pose_w2c": apply_camera_frame_delta(
                            pose_w2c,
                            translation_cam_m=(0.0, 0.0, 0.0),
                            rotation_deg=delta_r,
                        ),
                        "translation_cam_m": (0.0, 0.0, 0.0),
                        "rotation_deg": tuple(delta_r),
                    }
                )
    return candidates[: max(1, int(config.max_candidates))]


def _camera_frame_points(points_world: np.ndarray, pose_w2c: np.ndarray) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    return (pose @ homog.T).T[:, :3]


def generate_landmark_guided_pose_candidates(
    pose_w2c: np.ndarray,
    *,
    sparse_points_world: np.ndarray,
    steps_m: Sequence[float] = (0.5, 1.0, 2.0, 5.0),
    max_candidates: int = 37,
) -> list[dict[str, Any]]:
    """Generate sparse-landmark-conditioned pose candidates.

    The primary axis is the median camera-frame direction to the sparse inlier
    landmarks. Two perpendicular axes are added to test parallax-based
    unocclusion without uniform pose-grid enumeration.
    """

    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    points_cam = _camera_frame_points(sparse_points_world, pose)
    valid = np.isfinite(points_cam).all(axis=1) & (points_cam[:, 2] > 1e-6) if points_cam.size else np.empty(0, dtype=bool)
    if valid.any():
        centroid = np.median(points_cam[valid], axis=0)
    else:
        centroid = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    ray_axis = centroid / max(float(np.linalg.norm(centroid)), 1e-8)
    if abs(float(np.dot(ray_axis, np.array([0.0, 1.0, 0.0])))) < 0.95:
        side_axis = np.cross(ray_axis, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    else:
        side_axis = np.cross(ray_axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    side_axis = side_axis / max(float(np.linalg.norm(side_axis)), 1e-8)
    up_axis = np.cross(side_axis, ray_axis)
    up_axis = up_axis / max(float(np.linalg.norm(up_axis)), 1e-8)
    axes = (
        ("ray_centroid", ray_axis),
        ("ray_side", side_axis),
        ("ray_up", up_axis),
    )
    candidates: list[dict[str, Any]] = [
        {
            "label": "base",
            "pose_w2c": pose.astype(np.float32),
            "translation_cam_m": (0.0, 0.0, 0.0),
            "rotation_deg": (0.0, 0.0, 0.0),
            "guided_axis": "base",
        }
    ]
    for step in steps_m:
        for axis_label, axis in axes:
            for sign, sign_label in ((1.0, "+"), (-1.0, "-")):
                translation = tuple((float(sign) * float(step) * axis).tolist())
                candidates.append(
                    {
                        "label": f"{axis_label}{sign_label}{float(step):.3f}",
                        "pose_w2c": apply_camera_frame_delta(
                            pose,
                            translation_cam_m=translation,
                            rotation_deg=(0.0, 0.0, 0.0),
                        ),
                        "translation_cam_m": translation,
                        "rotation_deg": (0.0, 0.0, 0.0),
                        "guided_axis": axis_label,
                    }
                )
    return candidates[: max(1, int(max_candidates))]


def _estimate_sparse_ray_conflicts_for_pose(
    *,
    gaussian_xyz: np.ndarray,
    gaussian_scale_world: np.ndarray | None = None,
    sparse_points_world: np.ndarray,
    pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    conflict_radius_px: float,
    depth_margin_m: float,
    footprint_radius_scale: float = 0.0,
    max_footprint_radius_px: float = 64.0,
    chunk_size: int = 65536,
) -> dict[str, Any]:
    gaussians = _as_array(gaussian_xyz, shape_tail=(3,), name="gaussian_xyz").reshape(-1, 3)
    points = _as_array(sparse_points_world, shape_tail=(3,), name="sparse_points_world").reshape(-1, 3)
    width, height = int(image_size[0]), int(image_size[1])
    sparse_xy, sparse_depth, sparse_valid = _project_points_with_depth(
        points,
        np.asarray(pose_w2c, dtype=np.float64),
        np.asarray(intrinsic, dtype=np.float64),
        width=width,
        height=height,
    )
    valid_sparse = sparse_valid & np.isfinite(sparse_depth) & (sparse_depth > 1e-6)
    valid_xy = sparse_xy[valid_sparse]
    valid_depth = sparse_depth[valid_sparse]
    conflict_count = 0
    conflicted_rays = np.zeros((valid_xy.shape[0],), dtype=bool)
    if valid_xy.shape[0] > 0 and gaussians.shape[0] > 0:
        pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
        K = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
        radius2 = float(conflict_radius_px) ** 2
        for start in range(0, gaussians.shape[0], max(1, int(chunk_size))):
            stop = min(start + max(1, int(chunk_size)), gaussians.shape[0])
            gaussian_xy, gaussian_depth, gaussian_valid = _project_points_with_depth(
                gaussians[start:stop],
                pose,
                K,
                width=width,
                height=height,
            )
            if not gaussian_valid.any():
                continue
            footprint_radius = _gaussian_screen_radius_px(
                gaussian_scale_world,
                gaussian_depth,
                K,
                start=start,
                stop=stop,
                radius_scale=float(footprint_radius_scale),
                max_radius_px=float(max_footprint_radius_px),
            )
            chunk_conflict, chunk_conflicted_rays = _find_sparse_ray_conflicts_projected(
                projected_xy=gaussian_xy,
                gaussian_depth=gaussian_depth,
                gaussian_valid=gaussian_valid,
                sparse_xy=valid_xy,
                sparse_depth=valid_depth,
                radius_px=float(conflict_radius_px),
                footprint_radius=footprint_radius,
                depth_margin_m=float(depth_margin_m),
            )
            conflict_count += int(chunk_conflict.sum())
            conflicted_rays |= chunk_conflicted_rays
    coverage = _coverage_cells(
        valid_xy,
        width=width,
        height=height,
        rows=4,
        cols=4,
    )
    return {
        "estimated_conflict_count": int(conflict_count),
        "estimated_conflicted_ray_count": int(conflicted_rays.sum()),
        "estimated_sparse_ray_count": int(valid_xy.shape[0]),
        "estimated_visible_sparse_count": int(valid_sparse.sum()),
        "estimated_coverage_grid_cells": int(coverage),
        "conflict_radius_px": float(conflict_radius_px),
        "depth_margin_m": float(depth_margin_m),
        "footprint_radius_scale": float(footprint_radius_scale),
        "max_footprint_radius_px": float(max_footprint_radius_px),
    }


def _fast_guided_sort_key(candidate: Mapping[str, Any]) -> tuple[float, float, str]:
    score = candidate.get("fast_score", {})
    scalar = float(score.get("scalar", float("-inf"))) if isinstance(score, Mapping) else float("-inf")
    translation_norm = (
        float(score.get("translation_norm_m", 0.0)) if isinstance(score, Mapping) else 0.0
    )
    return scalar, -translation_norm, str(candidate.get("label", ""))


def _deterministic_gaussian_score_subset(
    gaussian_xyz: np.ndarray,
    gaussian_scale_world: np.ndarray | None,
    *,
    max_gaussians: int,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    gaussians = _as_array(gaussian_xyz, shape_tail=(3,), name="gaussian_xyz").reshape(-1, 3)
    scales = None
    if gaussian_scale_world is not None:
        scales = _as_array(gaussian_scale_world, shape_tail=(3,), name="gaussian_scale_world").reshape(-1, 3)
        if scales.shape[0] != gaussians.shape[0]:
            raise ValueError("gaussian_scale_world must have one row per Gaussian")
    cap = int(max_gaussians)
    if cap <= 0 or gaussians.shape[0] <= cap:
        return gaussians, scales, int(gaussians.shape[0])
    indices = np.linspace(0, gaussians.shape[0] - 1, cap, dtype=np.int64)
    return gaussians[indices], (None if scales is None else scales[indices]), int(gaussians.shape[0])


def _select_fast_guided_candidate_subset(
    scored: Sequence[dict[str, Any]],
    *,
    max_render_candidates: int,
) -> list[dict[str, Any]]:
    """Select a small render set while preserving guided-axis diversity."""

    if not scored:
        return []
    keep_count = max(1, int(max_render_candidates))
    base = scored[0]
    selected: list[dict[str, Any]] = [base]
    if keep_count <= 1:
        base["render_rank"] = 0
        return selected

    non_base = sorted(list(scored[1:]), key=_fast_guided_sort_key, reverse=True)
    selected_ids = {id(base)}
    axis_best: dict[str, dict[str, Any]] = {}
    for candidate in non_base:
        axis = str(candidate.get("guided_axis", ""))
        if axis in {"ray_centroid", "ray_side", "ray_up"} and axis not in axis_best:
            axis_best[axis] = candidate
    diverse = sorted(axis_best.values(), key=_fast_guided_sort_key, reverse=True)
    for candidate in diverse:
        if len(selected) >= keep_count:
            break
        selected.append(candidate)
        selected_ids.add(id(candidate))
    for candidate in non_base:
        if len(selected) >= keep_count:
            break
        if id(candidate) in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(id(candidate))
    for rank, candidate in enumerate(selected):
        candidate["render_rank"] = int(rank)
    return selected


def generate_fast_landmark_guided_pose_candidates(
    pose_w2c: np.ndarray,
    *,
    sparse_points_world: np.ndarray,
    gaussian_xyz: np.ndarray,
    gaussian_scale_world: np.ndarray | None = None,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    steps_m: Sequence[float] = (0.5, 1.0, 2.0, 5.0),
    max_render_candidates: int = 4,
    conflict_radius_px: float = 8.0,
    depth_margin_m: float = 1.0,
    footprint_radius_scale: float = 0.0,
    max_footprint_radius_px: float = 64.0,
    pool_max_candidates: int = 37,
    chunk_size: int = 65536,
    score_max_gaussians: int = 65536,
) -> list[dict[str, Any]]:
    """Preselect sparse-landmark-guided dense render poses using geometry only.

    This does not use ground-truth pose or localization error. It scores the
    normal guided-pose pool by whether candidate camera shifts reduce near
    Gaussian conflicts along sparse inlier landmark rays, then returns only
    base plus the top geometry candidates for expensive rendering.
    """

    pool = generate_landmark_guided_pose_candidates(
        pose_w2c,
        sparse_points_world=sparse_points_world,
        steps_m=steps_m,
        max_candidates=pool_max_candidates,
    )
    score_gaussian_xyz, score_gaussian_scale_world, full_gaussian_count = _deterministic_gaussian_score_subset(
        gaussian_xyz,
        gaussian_scale_world,
        max_gaussians=int(score_max_gaussians),
    )
    scored: list[dict[str, Any]] = []
    for candidate in pool:
        fast_score = _estimate_sparse_ray_conflicts_for_pose(
            gaussian_xyz=score_gaussian_xyz,
            gaussian_scale_world=score_gaussian_scale_world,
            sparse_points_world=sparse_points_world,
            pose_w2c=np.asarray(candidate["pose_w2c"], dtype=np.float32).reshape(4, 4),
            intrinsic=intrinsic,
            image_size=image_size,
            conflict_radius_px=float(conflict_radius_px),
            depth_margin_m=float(depth_margin_m),
            footprint_radius_scale=float(footprint_radius_scale),
            max_footprint_radius_px=float(max_footprint_radius_px),
            chunk_size=int(chunk_size),
        )
        translation_norm = float(np.linalg.norm(np.asarray(candidate.get("translation_cam_m", (0.0, 0.0, 0.0)), dtype=np.float64)))
        scalar = (
            -10.0 * float(fast_score["estimated_conflicted_ray_count"])
            -0.01 * float(fast_score["estimated_conflict_count"])
            +0.10 * float(fast_score["estimated_coverage_grid_cells"])
            +0.01 * float(fast_score["estimated_visible_sparse_count"])
            -0.02 * translation_norm
        )
        enriched = dict(candidate)
        enriched["selection_mode"] = "fast_guided_pose"
        enriched["fast_score"] = {
            **fast_score,
            "scalar": float(scalar),
            "translation_norm_m": translation_norm,
            "score_gaussian_count": int(score_gaussian_xyz.shape[0]),
            "full_gaussian_count": int(full_gaussian_count),
        }
        scored.append(enriched)
    if not scored:
        return []
    keep_count = max(1, int(max_render_candidates))
    base = scored[0]
    if int(base["fast_score"]["estimated_conflict_count"]) > 0:
        selected = _select_fast_guided_candidate_subset(scored, max_render_candidates=keep_count)
    else:
        selected = [base]
    for rank, candidate in enumerate(selected):
        candidate["render_rank"] = int(rank)
        candidate["fast_pool_candidate_count"] = int(len(scored))
    return selected


def score_slcdp_preflight_result(result: Mapping[str, Any]) -> float:
    """Convert an SLCDP preflight dictionary into a scalar search score."""

    visible_ratio = float(result.get("visible_ratio", 0.0) or 0.0)
    visible_count = float(result.get("visible_count", 0.0) or 0.0)
    coverage = float(result.get("coverage_grid_cells", 0.0) or 0.0)
    feature = result.get("feature_cosine_median")
    feature_score = max(float(feature), 0.0) if feature is not None else 0.0
    failed = result.get("failed_checks", {})
    failed_count = float(sum(1 for value in failed.values() if bool(value))) if isinstance(failed, Mapping) else 0.0
    accept_bonus = 1.0 if result.get("decision") == "accept_dense" else 0.0
    return (
        accept_bonus
        + 2.0 * visible_ratio
        + 0.02 * min(visible_count, 100.0)
        + 0.08 * min(coverage, 16.0)
        + 1.5 * feature_score
        - 0.25 * failed_count
    )


def select_repaired_pose_candidate(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    require_accept: bool = True,
    min_score_gain: float = 0.05,
    translation_penalty_per_m: float = 0.0,
    allow_low_confidence_gated_base: bool = False,
    low_confidence_gated_base_min_score_gain: float = 0.25,
) -> dict[str, Any]:
    """Select a repaired dense-render pose from evaluated candidates."""

    if not evaluations:
        return {
            "schema": "loc_gs_slcdp_repair_selection_v1",
            "decision": "skip_dense_keep_sparse",
            "reason": "no_candidates",
            "candidate_count": 0,
            "diagnostic_only": True,
        }
    scored: list[tuple[float, Mapping[str, Any]]] = []
    score_rows: list[dict[str, Any]] = []
    for item in evaluations:
        preflight_score = score_slcdp_preflight_result(dict(item.get("preflight", {})))
        translation_norm = _candidate_translation_norm_m(item)
        translation_penalty = float(translation_penalty_per_m) * translation_norm
        adjusted_score = preflight_score - translation_penalty
        scored.append((adjusted_score, item))
        score_rows.append(
            {
                "label": str(item.get("label")),
                "preflight_score": float(preflight_score),
                "translation_norm_m": float(translation_norm),
                "translation_penalty": float(translation_penalty),
                "adjusted_score": float(adjusted_score),
            }
        )
    base_score = scored[0][0]
    base = scored[0][1]
    base_preflight = dict(base.get("preflight", {}))
    base_passes = base_preflight.get("decision") == "accept_dense"
    best_score, best = max(scored, key=lambda item: item[0])
    best_preflight = dict(best.get("preflight", {}))
    passes = best_preflight.get("decision") == "accept_dense"
    score_gain = float(best_score - base_score)
    low_confidence_gated_base = _is_low_confidence_gated_base_repair(
        base,
        best,
        score_gain=score_gain,
        min_score_gain=float(low_confidence_gated_base_min_score_gain),
    )
    if require_accept and not passes:
        decision = "skip_dense_keep_sparse"
    elif bool(allow_low_confidence_gated_base) and low_confidence_gated_base:
        decision = "accept_repaired_dense_pose"
    elif (not bool(base_passes)) and bool(passes) and str(best.get("label")) != "base" and score_gain > 0.0:
        decision = "accept_repaired_dense_pose"
    elif score_gain < float(min_score_gain):
        decision = "accept_original_dense_pose"
    elif str(best.get("label")) == "base":
        decision = "accept_original_dense_pose"
    else:
        decision = "accept_repaired_dense_pose"
    selected_label = str(best.get("label"))
    selected_score = float(best_score)
    if decision == "accept_original_dense_pose":
        selected_label = "base"
        selected_score = float(base_score)
    return {
        "schema": "loc_gs_slcdp_repair_selection_v1",
        "decision": decision,
        "selected_label": selected_label,
        "selected_score": selected_score,
        "best_label": str(best.get("label")),
        "best_score": float(best_score),
        "base_score": float(base_score),
        "score_gain": score_gain,
        "candidate_count": int(len(evaluations)),
        "require_accept": bool(require_accept),
        "translation_penalty_per_m": float(translation_penalty_per_m),
        "allow_low_confidence_gated_base": bool(allow_low_confidence_gated_base),
        "low_confidence_gated_base": bool(low_confidence_gated_base),
        "low_confidence_gated_base_min_score_gain": float(low_confidence_gated_base_min_score_gain),
        "candidate_scores": score_rows,
        "diagnostic_only": True,
    }


def _is_low_confidence_gated_base_repair(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    score_gain: float,
    min_score_gain: float,
) -> bool:
    if str(candidate.get("label")) != "gated_base":
        return False
    if float(score_gain) < float(min_score_gain):
        return False
    base_preflight = dict(base.get("preflight", {}))
    candidate_preflight = dict(candidate.get("preflight", {}))
    if base_preflight.get("decision") != "accept_dense" or candidate_preflight.get("decision") != "accept_dense":
        return False
    if bool(base_preflight.get("sparse_confident", False)) or bool(candidate_preflight.get("sparse_confident", False)):
        return False
    base_near = float(base_preflight.get("near_occluder_count", 0.0) or 0.0)
    candidate_near = float(candidate_preflight.get("near_occluder_count", 0.0) or 0.0)
    base_visible = float(base_preflight.get("visible_ratio", 0.0) or 0.0)
    candidate_visible = float(candidate_preflight.get("visible_ratio", 0.0) or 0.0)
    return candidate_near < base_near and candidate_visible >= base_visible


def _candidate_translation_norm_m(candidate: Mapping[str, Any]) -> float:
    fast_score = candidate.get("fast_score")
    if isinstance(fast_score, Mapping):
        try:
            return float(fast_score.get("translation_norm_m", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    translation = candidate.get("translation_cam_m")
    if translation is None:
        return 0.0
    try:
        return float(np.linalg.norm(np.asarray(translation, dtype=np.float64).reshape(-1)))
    except (TypeError, ValueError):
        return 0.0


def _float_nested(mapping: Mapping[str, Any] | None, path: Sequence[str]) -> float | None:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_slcdp_from_summaries(
    match_summary: Mapping[str, Any],
    feature_summary: Mapping[str, Any] | None = None,
    *,
    thresholds: SLCDPThresholds = SLCDPThresholds(),
) -> dict[str, Any]:
    """Classify a captured sparse/dense diagnostic case without using GT for the gate."""

    sparse = match_summary.get("sparse") if isinstance(match_summary.get("sparse"), Mapping) else {}
    sparse_inliers = int(float(sparse.get("solver_inlier_count", 0) or 0))
    sparse_ratio = float(sparse.get("solver_inlier_ratio", 0.0) or 0.0)
    sparse_confident = (
        sparse_inliers >= int(thresholds.min_sparse_inliers)
        and sparse_ratio >= float(thresholds.min_sparse_inlier_ratio)
    )
    feature = feature_summary.get("feature") if isinstance(feature_summary, Mapping) else {}
    same_pixel_p50 = _float_nested(feature, ("same_pixel_cosine", "p50"))
    same_pixel_mean = _float_nested(feature, ("same_pixel_cosine", "mean"))
    max_cosine_mean = _float_nested(feature, ("query_to_render_max_cosine", "mean"))
    coarse_mnn = _float_nested(feature, ("coarse_mnn_count",))
    reasons: list[str] = []
    feature_failed = False
    if same_pixel_p50 is not None and same_pixel_p50 < float(thresholds.min_feature_cosine):
        feature_failed = True
    if max_cosine_mean is not None and max_cosine_mean < float(thresholds.min_query_to_render_max_cosine):
        feature_failed = True
    if coarse_mnn is not None and coarse_mnn < int(thresholds.min_coarse_mnn_count):
        feature_failed = True
    if feature_failed:
        reasons.append("feature_agreement")
    if sparse_confident and feature_failed:
        decision = "skip_dense_keep_sparse"
    elif feature_failed:
        decision = "retry_sparse_or_patch_dense"
    else:
        decision = "accept_dense"

    sparse_te = _float_nested(match_summary, ("captured_sparse_te_cm",))
    sparse_re = _float_nested(match_summary, ("captured_sparse_re_deg",))
    dense_te = _float_nested(match_summary, ("captured_dense_te_cm",))
    dense_re = _float_nested(match_summary, ("captured_dense_re_deg",))
    use_sparse = decision == "skip_dense_keep_sparse"
    accepted_te = sparse_te if use_sparse else dense_te
    accepted_re = sparse_re if use_sparse else dense_re
    dense_te_reduction = dense_te - accepted_te if dense_te is not None and accepted_te is not None else None
    return {
        "schema": "loc_gs_slcdp_summary_gate_v1",
        "scene": match_summary.get("scene"),
        "query_index": match_summary.get("query_index"),
        "image_name": match_summary.get("image_name"),
        "decision": decision,
        "reasons": reasons,
        "sparse_confident": bool(sparse_confident),
        "sparse_inlier_count": int(sparse_inliers),
        "sparse_inlier_ratio": float(sparse_ratio),
        "same_pixel_cosine_p50": same_pixel_p50,
        "same_pixel_cosine_mean": same_pixel_mean,
        "query_to_render_max_cosine_mean": max_cosine_mean,
        "coarse_mnn_count": int(coarse_mnn) if coarse_mnn is not None else None,
        "sparse_te_cm": sparse_te,
        "sparse_re_deg": sparse_re,
        "dense_te_cm": dense_te,
        "dense_re_deg": dense_re,
        "accepted_final_te_cm": accepted_te,
        "accepted_final_re_deg": accepted_re,
        "dense_te_reduction_cm": dense_te_reduction,
        "diagnostic_only": True,
        "branch_selection": decision != "accept_dense",
    }
