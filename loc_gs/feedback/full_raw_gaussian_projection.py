from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from loc_gs.feedback.projected_gaussian_rays import projected_gaussians_to_sparse_source_rays, proxy_contribution
from loc_gs.feedback.ray_attributed_solver_feedback import validate_split_name


SCHEMA_VERSION = "full_raw_gaussian_projection_v1"
PROJECTION_SOURCE = "full_raw_gaussians"


def _as_matrix(value: Any, *, shape: tuple[int, int], name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if tensor.numel() != shape[0] * shape[1]:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(torch.as_tensor(value).shape)}")
    return tensor.reshape(shape)


def _as_optional_vector(value: Any, *, count: int, default: float, name: str) -> torch.Tensor:
    if value is None:
        return torch.full((count,), float(default), dtype=torch.float32)
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if tensor.numel() == 1:
        return tensor.repeat(count)
    if tensor.numel() != count:
        raise ValueError(f"{name} length {tensor.numel()} does not match gaussian count {count}")
    return tensor


def _world_to_camera_matrix(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.shape == (3, 4):
        bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        return torch.cat([tensor, bottom], dim=0)
    return _as_matrix(value, shape=(4, 4), name="world_to_camera")


def validate_full_raw_projection_bundle(
    bundle: Mapping[str, Any],
    *,
    expected_source_gaussian_count: int | None = None,
) -> dict[str, Any]:
    summary = dict(bundle.get("summary", {}))
    if summary.get("projection_source") != PROJECTION_SOURCE:
        raise ValueError("projection bundle must use projection_source=full_raw_gaussians")
    if bool(summary.get("sampled_idx_used", False)):
        raise ValueError("projection bundle must not use sampled_idx; expected full_raw_gaussians")
    count = int(summary.get("source_gaussian_count", -1))
    if count < 0:
        raise ValueError("projection bundle is missing source_gaussian_count")
    if expected_source_gaussian_count is not None and count != int(expected_source_gaussian_count):
        raise ValueError(
            f"projection source_gaussian_count {count} does not match expected {expected_source_gaussian_count}"
        )
    for idx, row in enumerate(bundle.get("projections", [])):
        if row.get("projection_source") != PROJECTION_SOURCE:
            raise ValueError(f"projection row {idx} is not marked as full_raw_gaussians")
    return summary


def project_full_raw_gaussians_to_view(
    *,
    gaussian_xyz: Any,
    world_to_camera: Any,
    intrinsic: Any,
    width: int,
    height: int,
    source_view_id: str,
    split_name: str,
    opacity: Any | None = None,
    radius_px: Any | None = None,
    near_depth: float = 1e-6,
) -> dict[str, Any]:
    split = validate_split_name(split_name)
    source = str(source_view_id).strip()
    if not source:
        raise ValueError("source_view_id is required")
    image_width = int(width)
    image_height = int(height)
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")

    xyz = torch.as_tensor(gaussian_xyz, dtype=torch.float32).reshape(-1, 3).cpu()
    count = int(xyz.shape[0])
    w2c = _world_to_camera_matrix(world_to_camera).cpu()
    K = _as_matrix(intrinsic, shape=(3, 3), name="intrinsic").cpu()
    opacities = _as_optional_vector(opacity, count=count, default=1.0, name="opacity").cpu().clamp(0.0, 1.0)
    radii = _as_optional_vector(radius_px, count=count, default=1.0, name="radius_px").cpu().clamp_min(1.0)

    if count == 0:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "projection_source": PROJECTION_SOURCE,
            "split_name": split,
            "source_view_id": source,
            "source_gaussian_count": 0,
            "projected_gaussian_count": 0,
            "dropped_behind_count": 0,
            "dropped_out_of_frame_count": 0,
            "sampled_idx_used": False,
        }
        return {"schema_version": SCHEMA_VERSION, "projections": [], "summary": summary}

    homo = torch.cat([xyz, torch.ones((count, 1), dtype=torch.float32)], dim=1)
    cam = (w2c @ homo.T).T[:, :3]
    depth = cam[:, 2]
    positive_depth = depth > float(near_depth)
    uvw = (K @ cam.T).T
    uv = uvw[:, :2] / depth.clamp_min(float(near_depth))[:, None]
    in_frame = (
        positive_depth
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(image_width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(image_height))
    )

    projections: list[dict[str, Any]] = []
    for gid in torch.where(in_frame)[0].tolist():
        projections.append(
            {
                "schema_version": SCHEMA_VERSION,
                "projection_source": PROJECTION_SOURCE,
                "source_view_id": source,
                "split_name": split,
                "gaussian_id": int(gid),
                "xy": [float(uv[gid, 0].item()), float(uv[gid, 1].item())],
                "depth": float(depth[gid].item()),
                "xyz": [float(v) for v in xyz[gid].tolist()],
                "opacity": float(opacities[gid].item()),
                "radius": float(radii[gid].item()),
            }
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "projection_source": PROJECTION_SOURCE,
        "split_name": split,
        "source_view_id": source,
        "source_gaussian_count": count,
        "projected_gaussian_count": int(len(projections)),
        "dropped_behind_count": int((~positive_depth).sum().item()),
        "dropped_out_of_frame_count": int((positive_depth & ~in_frame).sum().item()),
        "sampled_idx_used": False,
    }
    bundle = {"schema_version": SCHEMA_VERSION, "projections": projections, "summary": summary}
    validate_full_raw_projection_bundle(bundle, expected_source_gaussian_count=count)
    return bundle


def _project_full_raw_gaussian_tensors(
    *,
    gaussian_xyz: Any,
    world_to_camera: Any,
    intrinsic: Any,
    width: int,
    height: int,
    opacity: Any | None = None,
    radius_px: Any | None = None,
    near_depth: float = 1e-6,
) -> dict[str, Any]:
    image_width = int(width)
    image_height = int(height)
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")

    xyz = torch.as_tensor(gaussian_xyz, dtype=torch.float32).reshape(-1, 3).cpu()
    count = int(xyz.shape[0])
    w2c = _world_to_camera_matrix(world_to_camera).cpu()
    K = _as_matrix(intrinsic, shape=(3, 3), name="intrinsic").cpu()
    opacities = _as_optional_vector(opacity, count=count, default=1.0, name="opacity").cpu().clamp(0.0, 1.0)
    radii = _as_optional_vector(radius_px, count=count, default=1.0, name="radius_px").cpu().clamp_min(1.0)

    if count == 0:
        empty_xy = torch.empty((0, 2), dtype=torch.float32)
        empty_depth = torch.empty((0,), dtype=torch.float32)
        empty_ids = torch.empty((0,), dtype=torch.long)
        return {
            "xyz": xyz,
            "uv": empty_xy,
            "depth": empty_depth,
            "gaussian_ids": empty_ids,
            "opacity": opacities,
            "radius": radii,
            "summary": {
                "source_gaussian_count": 0,
                "projected_gaussian_count": 0,
                "dropped_behind_count": 0,
                "dropped_out_of_frame_count": 0,
            },
        }

    homo = torch.cat([xyz, torch.ones((count, 1), dtype=torch.float32)], dim=1)
    cam = (w2c @ homo.T).T[:, :3]
    depth_all = cam[:, 2]
    positive_depth = depth_all > float(near_depth)
    uvw = (K @ cam.T).T
    uv_all = uvw[:, :2] / depth_all.clamp_min(float(near_depth))[:, None]
    in_frame = (
        positive_depth
        & (uv_all[:, 0] >= 0.0)
        & (uv_all[:, 0] < float(image_width))
        & (uv_all[:, 1] >= 0.0)
        & (uv_all[:, 1] < float(image_height))
    )
    gaussian_ids = torch.where(in_frame)[0].to(torch.long)
    return {
        "xyz": xyz,
        "uv": uv_all[gaussian_ids].contiguous(),
        "depth": depth_all[gaussian_ids].contiguous(),
        "gaussian_ids": gaussian_ids,
        "opacity": opacities,
        "radius": radii,
        "summary": {
            "source_gaussian_count": count,
            "projected_gaussian_count": int(gaussian_ids.numel()),
            "dropped_behind_count": int((~positive_depth).sum().item()),
            "dropped_out_of_frame_count": int((positive_depth & ~in_frame).sum().item()),
        },
    }


def _match_source_id(record: Mapping[str, Any]) -> str:
    return str(record.get("source_view_id", record.get("view_id", ""))).strip()


def _xy_from_match(record: Mapping[str, Any]) -> list[float]:
    for key in ("source_xy", "query_xy", "xy"):
        if key not in record:
            continue
        value = record[key]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"{key} must be a coordinate sequence")
        values = list(value)
        if len(values) != 2:
            raise ValueError(f"{key} must contain two values")
        return [float(values[0]), float(values[1])]
    raise ValueError("match record is missing source_xy/query_xy/xy")


def _source_rays_from_projected_tensors(
    *,
    source_matches: Sequence[Mapping[str, Any]],
    projected: Mapping[str, Any],
    source_view_id: str,
    split_name: str,
    top_k: int,
    max_radius_px: float,
    radius_scale: float,
    min_contribution: float,
) -> dict[str, Any]:
    uv = torch.as_tensor(projected["uv"], dtype=torch.float32)
    gaussian_ids = torch.as_tensor(projected["gaussian_ids"], dtype=torch.long)
    if uv.numel() == 0:
        return {
            "rays": [],
            "dropped_no_projection": len(source_matches),
            "backend": "empty_projection",
        }

    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        projection_rows = []
        xyz = torch.as_tensor(projected["xyz"], dtype=torch.float32)
        depth = torch.as_tensor(projected["depth"], dtype=torch.float32)
        opacity = torch.as_tensor(projected["opacity"], dtype=torch.float32)
        radius = torch.as_tensor(projected["radius"], dtype=torch.float32)
        for local_idx, gid_tensor in enumerate(gaussian_ids):
            gid = int(gid_tensor.item())
            projection_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "projection_source": PROJECTION_SOURCE,
                    "source_view_id": source_view_id,
                    "split_name": split_name,
                    "gaussian_id": gid,
                    "xy": [float(v) for v in uv[local_idx].tolist()],
                    "depth": float(depth[local_idx].item()),
                    "xyz": [float(v) for v in xyz[gid].tolist()],
                    "opacity": float(opacity[gid].item()),
                    "radius": float(radius[gid].item()),
                }
            )
        bundle = projected_gaussians_to_sparse_source_rays(
            source_matches,
            projection_rows,
            split_name=split_name,
            top_k=top_k,
            max_radius_px=max_radius_px,
            radius_scale=radius_scale,
            min_contribution=min_contribution,
            require_full_raw_source=True,
        )
        return {
            "rays": bundle["rays"],
            "dropped_no_projection": int(bundle["summary"]["dropped_no_projection"]),
            "backend": "projection_rows_fallback",
        }

    uv_np = uv.numpy()
    tree = cKDTree(uv_np)
    xyz = torch.as_tensor(projected["xyz"], dtype=torch.float32)
    depth = torch.as_tensor(projected["depth"], dtype=torch.float32)
    opacity = torch.as_tensor(projected["opacity"], dtype=torch.float32)
    radius = torch.as_tensor(projected["radius"], dtype=torch.float32)
    gaussian_ids_np = gaussian_ids.numpy()
    rays: list[dict[str, Any]] = []
    dropped_no_projection = 0
    seen_pixels: set[tuple[int, int]] = set()
    for match in source_matches:
        source_xy = _xy_from_match(match)
        pixel_key = (int(round(source_xy[0])), int(round(source_xy[1])))
        if pixel_key in seen_pixels:
            continue
        seen_pixels.add(pixel_key)
        local_indices = tree.query_ball_point(np.asarray(source_xy, dtype=np.float32), r=float(max_radius_px))
        candidates: list[dict[str, Any]] = []
        for local_idx in local_indices:
            gid = int(gaussian_ids_np[int(local_idx)])
            pxy = uv_np[int(local_idx)]
            distance = float(np.hypot(float(pxy[0]) - source_xy[0], float(pxy[1]) - source_xy[1]))
            item_radius = float(radius[gid].item())
            gate = min(float(max_radius_px), max(1.0, float(radius_scale) * item_radius))
            if distance > gate:
                continue
            contribution = proxy_contribution(
                distance_px=distance,
                radius_px=item_radius,
                opacity=float(opacity[gid].item()),
            )
            if contribution < float(min_contribution):
                continue
            candidates.append(
                {
                    "gaussian_id": gid,
                    "contribution": float(contribution),
                    "depth": float(depth[int(local_idx)].item()),
                    "reliability": max(0.0, min(1.0, 1.0 - distance / max(gate, 1e-6))),
                    "xyz": [float(v) for v in xyz[gid].tolist()],
                }
            )
        if not candidates:
            dropped_no_projection += 1
            continue
        selected = sorted(candidates, key=lambda item: (-float(item["contribution"]), int(item["gaussian_id"])))[:top_k]
        weights = [float(item["contribution"]) for item in selected]
        depths = [float(item.get("depth", 0.0)) for item in selected]
        weight_sum = sum(weights)
        expected_depth = (
            sum(depth_value * weight for depth_value, weight in zip(depths, weights)) / max(weight_sum, 1e-12)
            if weight_sum > 0.0 and any(depth_value > 0.0 for depth_value in depths)
            else 0.0
        )
        positive_depths = [depth_value for depth_value in depths if depth_value > 0.0]
        rays.append(
            {
                "schema_version": "projected_gaussian_rays_v1",
                "source_view_id": source_view_id,
                "split_name": split_name,
                "pixel_xy": [float(source_xy[0]), float(source_xy[1])],
                "contributors": selected,
                "rendered_depth": float(min(positive_depths)) if positive_depths else 0.0,
                "expected_depth": float(expected_depth),
            }
        )
    return {
        "rays": rays,
        "dropped_no_projection": int(dropped_no_projection),
        "backend": "scipy_ckdtree",
    }


def build_sparse_source_rays_from_full_raw_gaussians(
    *,
    matches: Sequence[Mapping[str, Any]],
    gaussian_xyz: Any,
    views_by_source: Mapping[str, Mapping[str, Any]],
    split_name: str,
    opacity: Any | None = None,
    radius_px: Any | None = None,
    top_k: int = 8,
    max_radius_px: float = 8.0,
    radius_scale: float = 2.0,
    min_contribution: float = 1e-6,
) -> dict[str, Any]:
    """Build sparse source rays by projecting full raw Gaussians one view at a time.

    This is the scalable full-train path: it does not write or keep a global
    projection JSONL. Per view, it projects all raw Gaussians, joins only that
    view's sparse matches, then discards the temporary projections.
    """

    split = validate_split_name(split_name)
    match_groups: dict[str, list[Mapping[str, Any]]] = {}
    dropped_missing_source = 0
    for match in matches:
        source = _match_source_id(match)
        if not source:
            dropped_missing_source += 1
            continue
        match_groups.setdefault(source, []).append(match)

    all_rays: list[dict[str, Any]] = []
    dropped_no_view = 0
    dropped_no_projection = 0
    projected_gaussian_count = 0
    dropped_behind_count = 0
    dropped_out_of_frame_count = 0
    projected_views = 0
    backend_counts: dict[str, int] = {}
    source_count = int(torch.as_tensor(gaussian_xyz, dtype=torch.float32).reshape(-1, 3).shape[0])

    for source, source_matches in sorted(match_groups.items()):
        view = views_by_source.get(source)
        if view is None:
            dropped_no_view += len(source_matches)
            continue
        projected = _project_full_raw_gaussian_tensors(
            gaussian_xyz=gaussian_xyz,
            world_to_camera=view["world_to_camera"],
            intrinsic=view["intrinsic"],
            width=int(view["width"]),
            height=int(view["height"]),
            opacity=opacity,
            radius_px=radius_px,
        )
        projection_summary = dict(projected["summary"])
        if int(projection_summary["source_gaussian_count"]) != source_count:
            raise ValueError(
                f"projection source_gaussian_count {projection_summary['source_gaussian_count']} "
                f"does not match expected {source_count}"
            )
        ray_bundle = _source_rays_from_projected_tensors(
            source_matches=source_matches,
            projected=projected,
            source_view_id=source,
            split_name=split,
            top_k=int(top_k),
            max_radius_px=float(max_radius_px),
            radius_scale=float(radius_scale),
            min_contribution=float(min_contribution),
        )
        all_rays.extend(ray_bundle["rays"])
        backend = str(ray_bundle.get("backend", "unknown"))
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
        projected_views += 1
        projected_gaussian_count += int(projection_summary["projected_gaussian_count"])
        dropped_behind_count += int(projection_summary["dropped_behind_count"])
        dropped_out_of_frame_count += int(projection_summary["dropped_out_of_frame_count"])
        dropped_no_projection += int(ray_bundle["dropped_no_projection"])

    return {
        "schema_version": "full_raw_sparse_source_rays_v1",
        "rays": all_rays,
        "summary": {
            "schema_version": "full_raw_sparse_source_rays_v1",
            "projection_source": PROJECTION_SOURCE,
            "split_name": split,
            "sampled_idx_used": False,
            "materialized_projection_jsonl": False,
            "source_gaussian_count": int(source_count),
            "source_view_count_requested": int(len(match_groups)),
            "source_view_count_projected": int(projected_views),
            "match_count": int(len(matches)),
            "ray_count": int(len(all_rays)),
            "dropped_missing_source": int(dropped_missing_source),
            "dropped_no_view": int(dropped_no_view),
            "dropped_no_projection": int(dropped_no_projection),
            "projected_gaussian_count": int(projected_gaussian_count),
            "dropped_behind_count": int(dropped_behind_count),
            "dropped_out_of_frame_count": int(dropped_out_of_frame_count),
            "source_ray_backend_counts": dict(sorted(backend_counts.items())),
            "top_k": int(top_k),
            "max_radius_px": float(max_radius_px),
            "radius_scale": float(radius_scale),
        },
    }
