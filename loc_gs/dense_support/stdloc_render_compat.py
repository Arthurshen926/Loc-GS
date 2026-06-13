from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def visible_mask_from_gsplat_radii(radii: torch.Tensor) -> torch.Tensor:
    """Return a one-dimensional visibility mask for gsplat radii variants."""

    values = torch.as_tensor(radii)
    if values.dim() >= 2 and int(values.shape[0]) == 1:
        values = values.squeeze(0)
    if values.dim() == 1:
        return values > 0
    return torch.any((values > 0).reshape(int(values.shape[0]), -1), dim=1)


def _visible_locability_values(pc: Any, visible_mask: torch.Tensor) -> torch.Tensor | None:
    try:
        locability = pc.get_locability
    except AttributeError:
        return None
    if locability is None or locability.numel() == 0:
        return None
    values = locability[visible_mask]
    if values.dim() == 1:
        values = values[:, None]
    return values.clamp(0.0, 1.0)


def render_from_pose_gsplat_compat(
    pc: Any,
    pose: torch.Tensor,
    fovx: float,
    fovy: float,
    width: int,
    height: int,
    bg_color: torch.Tensor | None = None,
    render_mode: str = "RGB+ED",
    rgb_only: bool = False,
    norm_feat_bf_render: bool = True,
    near_plane: float = 0.01,
    far_plane: float = 10000,
    **rasterize_args: Any,
) -> dict[str, Any]:
    """STDLoc-compatible gsplat render with robust radii mask handling.

    Newer gsplat builds may return multichannel radii, e.g. ``[1, N, 2]``.
    Vendored STDLoc assumes ``[1, N]`` and uses the raw ``radii > 0`` mask for
    both gaussian and feature tensors. Loc-GS dense diagnostics need the same
    render semantics with a one-dimensional per-gaussian visibility mask.
    """

    from gsplat import rasterization

    means3d = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    colors = pc.get_features
    sh_degree = pc.active_sh_degree

    if int(scales.shape[1]) == 2:
        from gaussian_renderer import render_from_pose_gsplat_2dgs  # type: ignore

        return render_from_pose_gsplat_2dgs(
            pc,
            pose,
            fovx,
            fovy,
            width,
            height,
            bg_color,
            render_mode,
            rgb_only,
            norm_feat_bf_render,
            near_plane,
            far_plane,
            **rasterize_args,
        )

    tanfovx = math.tan(float(fovx) * 0.5)
    tanfovy = math.tan(float(fovy) * 0.5)
    focal_length_x = float(width) / (2.0 * tanfovx)
    focal_length_y = float(height) / (2.0 * tanfovy)
    device = means3d.device
    k_matrix = torch.tensor(
        [
            [focal_length_x, 0.0, float(width) / 2.0],
            [0.0, focal_length_y, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
    )
    if bg_color is None:
        bg_color = torch.zeros(3, device=device)

    render_colors, render_alphas, info = rasterization(
        means=means3d,
        quats=rotations,
        scales=scales,
        opacities=opacity.squeeze(-1),
        colors=colors,
        viewmats=pose[None],
        Ks=k_matrix[None],
        backgrounds=bg_color[None],
        width=int(width),
        height=int(height),
        packed=False,
        sh_degree=sh_degree,
        near_plane=float(near_plane),
        far_plane=float(far_plane),
        render_mode=render_mode,
        **rasterize_args,
    )
    rendered_image = render_colors[0].permute(2, 0, 1)
    color = rendered_image[:3]
    depth = rendered_image[3:] if int(rendered_image.shape[0]) == 4 else None
    radii = info["radii"].squeeze(0)
    visible_mask = visible_mask_from_gsplat_radii(info["radii"]).to(device=means3d.device)

    try:
        info["means2d"].retain_grad()
    except Exception:
        pass

    locability_map = None
    if not bool(rgb_only):
        loc_feature = pc.get_loc_feature[visible_mask].squeeze()
        if norm_feat_bf_render:
            loc_feature = F.normalize(loc_feature, p=2, dim=-1)

        feat_map, _alphas, _meta = rasterization(
            means3d[visible_mask],
            rotations[visible_mask],
            scales[visible_mask],
            opacity.squeeze(-1)[visible_mask],
            loc_feature,
            pose[None],
            k_matrix[None],
            int(width),
            int(height),
            packed=False,
            near_plane=float(near_plane),
            far_plane=float(far_plane),
            **rasterize_args,
        )
        feat_map = feat_map[0].permute(2, 0, 1)
        feat_map = F.normalize(feat_map, p=2, dim=0)

        locability_values = _visible_locability_values(pc, visible_mask)
        if locability_values is not None and locability_values.numel() > 0:
            locability_map, _, _ = rasterization(
                means=means3d[visible_mask],
                quats=rotations[visible_mask],
                scales=scales[visible_mask],
                opacities=opacity.squeeze(-1)[visible_mask],
                colors=locability_values,
                viewmats=pose[None],
                Ks=k_matrix[None],
                width=int(width),
                height=int(height),
                packed=False,
                near_plane=float(near_plane),
                far_plane=float(far_plane),
                **rasterize_args,
            )
            locability_map = locability_map[0].permute(2, 0, 1).clamp(0.0, 1.0)
    else:
        feat_map = None

    return {
        "render": color,
        "feature_map": feat_map,
        "viewspace_points": info["means2d"],
        "visibility_filter": visible_mask,
        "radii": radii,
        "alphas": render_alphas,
        "depth": depth,
        "locability_map": locability_map,
    }
