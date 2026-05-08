#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Subset
from tqdm import tqdm

from loc_gs.data.cambridge_dataset import CambridgeHybridDataset
from loc_gs.data.superpoint_cache import SuperPointTeacherCache
from loc_gs.localization.hybrid_localizer import (
    extract_keypoints_from_detector_logits,
    flatten_rendered_landmarks,
    match_descriptors_topk,
    pose_error_cm_deg,
    refine_rendered_positions_softargmax,
    sample_descriptors_bilinear,
    solve_pnp_ransac,
)
from loc_gs.localization.lightglue_matcher import (
    lightglue_feature_name,
    match_lightglue_descriptors,
)
from loc_gs.localization.matcher_registry import (
    DENSE_MATCHERS,
    DIM_PIPELINES,
    SPARSE_MATCHERS,
    resolve_sparse_dense_matchers,
)
from loc_gs.localization.rendered_keypoints import select_rendered_keypoints
from loc_gs.localization.stdloc_parity import (
    apply_match_prior,
    coarse_to_fine_dense_matches,
    match_correlation_matrix,
)
from loc_gs.localization.stdloc_detector import (
    StdlocKeypointDetector,
    extract_stdloc_detector_keypoints,
)
from loc_gs.losses.localization_loss import unproject_dense_depth_to_world
from loc_gs.models.hybrid_gaussian import HybridFeatureGaussian, SuperPointOutputHead
from loc_gs.scripts.extract_superpoint_features import SuperPointNet
from loc_gs.scripts.train_cambridge_hybrid import (
    decode_gaussian_center_descriptors,
    extract_superpoint_teacher_batch,
    make_feature_renderer_intrinsics,
    maybe_write_superpoint_metadata,
    normalize_position_map,
    render_hybrid_superpoint,
    superpoint_gray,
)


def unproject_positions_yx(
    positions_yx: torch.Tensor,
    depth_map: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
    pixel_center_offset: float = 0.0,
) -> torch.Tensor:
    if positions_yx.numel() == 0:
        return positions_yx.new_zeros((0, 3))
    H, W = depth_map.shape
    y = positions_yx[:, 0].clamp(0, H - 1)
    x = positions_yx[:, 1].clamp(0, W - 1)
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = (x0 + 1).clamp(max=W - 1)
    y1 = (y0 + 1).clamp(max=H - 1)
    wx = x - x0.float()
    wy = y - y0.float()
    d00 = depth_map[y0, x0]
    d10 = depth_map[y0, x1]
    d01 = depth_map[y1, x0]
    d11 = depth_map[y1, x1]
    depth = (1.0 - wy) * ((1.0 - wx) * d00 + wx * d10) + wy * ((1.0 - wx) * d01 + wx * d11)
    if pixel_center_offset:
        offset = float(pixel_center_offset)
        x = x + offset
        y = y + offset
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (x - cx) / fx.clamp_min(1e-8) * depth
    y_cam = (y - cy) / fy.clamp_min(1e-8) * depth
    pts_cam = torch.stack([x_cam, y_cam, depth, torch.ones_like(depth)], dim=-1)
    c2w = torch.linalg.inv(pose_w2c)
    return (c2w @ pts_cam.T).T[:, :3]


def upsample_feature_map(feature_map: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """STDLoc-style dense localization uses a full-resolution fine feature map."""
    return F.interpolate(
        feature_map.unsqueeze(0).float(),
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    )[0]


def load_cambridge_rgb_no_resize(
    scene_root: Path,
    image_subdir: str,
    image_name: str,
    device: torch.device,
) -> torch.Tensor:
    candidates = []
    if image_subdir:
        candidates.append(scene_root / image_subdir.strip("/") / image_name)
    candidates.append(scene_root / image_name)
    path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)


def resolve_matchers(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve legacy --matcher into independently configurable sparse/dense matchers."""
    return resolve_sparse_dense_matchers(
        getattr(args, "matcher", "topk"),
        getattr(args, "sparse_matcher", ""),
        getattr(args, "dense_matcher", ""),
    )


@torch.no_grad()
def decode_landmark_descriptors(
    model: HybridFeatureGaussian,
    sp_head: SuperPointOutputHead,
    max_landmarks: int,
    chunk_size: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xyz = model.get_xyz()
    locability = model.get_locability().squeeze(-1)
    keep = min(int(max_landmarks), xyz.shape[0])
    if keep < xyz.shape[0]:
        _, ids = torch.topk(locability, k=keep)
    else:
        ids = torch.arange(xyz.shape[0], device=xyz.device)
    selected_xyz = xyz[ids]
    selected_latent = model.get_latent()[ids]
    selected_loc = locability[ids]

    desc_chunks = []
    for start in range(0, keep, chunk_size):
        end = min(start + chunk_size, keep)
        latent_map = selected_latent[start:end].T.contiguous().view(1, -1, end - start, 1)
        pos_map = selected_xyz[start:end].T.contiguous().view(1, 3, end - start, 1)
        pos_map = normalize_position_map(pos_map, model.get_xyz())
        fused = model.decode_screen_space(latent_map, pos_map)
        desc = sp_head(fused)["descriptor"].squeeze(0).squeeze(-1).T
        desc_chunks.append(F.normalize(desc.float(), dim=-1))
    return selected_xyz, torch.cat(desc_chunks, dim=0), selected_loc


def _load_pickle_tensor(path: Path) -> torch.Tensor:
    with path.open("rb") as f:
        value = pickle.load(f)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        for key in ("score_avg", "sampled_scores", "scores", "sampled_idx", "indices"):
            tensor_value = value.get(key)
            if tensor_value is None:
                continue
            if isinstance(tensor_value, torch.Tensor):
                return tensor_value.detach().cpu()
            return torch.as_tensor(tensor_value)
        raise TypeError(f"Could not find a tensor-like entry in pickle dict: {path}")
    return torch.as_tensor(value)


@torch.no_grad()
def build_stdloc_detector_landmark_bank(
    model: HybridFeatureGaussian,
    sp_head: SuperPointOutputHead,
    detector_dir: Path,
    max_landmarks: int,
    descriptor_source: str,
    ply_loc_feature_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx_path = detector_dir / "sampled_idx.pkl"
    if not idx_path.exists():
        raise FileNotFoundError(f"STDLoc detector landmark indices not found: {idx_path}")
    ids = _load_pickle_tensor(idx_path).long().view(-1)
    if int(max_landmarks) > 0:
        ids = ids[: min(int(max_landmarks), ids.numel())]
    ids = ids.to(device=device)
    xyz = model.get_xyz()[ids]

    loc_feature = model.get_ply_loc_feature()
    ply_desc = None
    if loc_feature.numel() > 0 and loc_feature.shape[1] > 0:
        ply_desc = F.normalize(loc_feature.to(device=device).float()[ids], p=2, dim=-1)

    if descriptor_source == "ply_loc":
        if ply_desc is None:
            raise ValueError("descriptor_source=ply_loc requires loc_* features in the input PLY")
        desc = ply_desc
    elif descriptor_source == "hybrid_ply_blend":
        if ply_desc is None:
            raise ValueError("descriptor_source=hybrid_ply_blend requires loc_* features in the input PLY")
        hybrid_desc = decode_gaussian_center_descriptors(model, sp_head, ids)
        weight = min(max(float(ply_loc_feature_weight), 0.0), 1.0)
        desc = F.normalize((1.0 - weight) * hybrid_desc + weight * ply_desc, p=2, dim=-1)
    else:
        desc = decode_gaussian_center_descriptors(model, sp_head, ids)

    score_path = detector_dir / "sampled_scores.pkl"
    if score_path.exists():
        scores = _load_pickle_tensor(score_path).float().view(-1)
        if scores.numel() == model.num_gaussians:
            prior = scores.to(device=device)[ids]
        else:
            prior = scores[: ids.numel()].to(device=device)
        prior = prior - prior.min()
        prior = prior / prior.max().clamp_min(1e-6)
    else:
        prior = torch.ones(ids.shape[0], device=device)
    return xyz, F.normalize(desc.float(), p=2, dim=-1), prior.float()


def select_view_landmark_indices(
    prior: torch.Tensor,
    flat_ids: torch.Tensor,
    height: int,
    width: int,
    quota: int,
    selection: str = "global",
    grid_size: int = 8,
) -> torch.Tensor:
    """Select rendered landmarks from one reference view with optional spatial diversity."""
    num_points = int(prior.numel())
    device = prior.device
    if num_points == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if selection == "global" or int(quota) <= 0 or int(quota) >= num_points:
        return torch.arange(num_points, dtype=torch.long, device=device)
    keep = max(1, min(int(quota), num_points))
    if selection == "per_view":
        return torch.topk(prior.float(), k=keep).indices
    if selection != "per_view_spatial":
        raise ValueError(f"Unsupported landmark selection mode: {selection}")

    grid = max(1, int(grid_size))
    flat_ids = flat_ids.to(device=device, dtype=torch.long)
    y = flat_ids // max(int(width), 1)
    x = flat_ids % max(int(width), 1)
    cell_cols = max(1, math.ceil(float(width) / float(grid)))
    cells = (y // grid) * cell_cols + (x // grid)
    unique_cells = torch.unique(cells, sorted=True)
    if unique_cells.numel() == 0:
        return torch.topk(prior.float(), k=keep).indices

    per_cell = max(1, math.ceil(float(keep) / float(unique_cells.numel())))
    selected_chunks = []
    for cell in unique_cells:
        ids = torch.where(cells == cell)[0]
        cell_keep = min(per_cell, int(ids.numel()))
        if cell_keep <= 0:
            continue
        order = torch.topk(prior[ids].float(), k=cell_keep).indices
        selected_chunks.append(ids[order])
    if selected_chunks:
        selected = torch.cat(selected_chunks, dim=0)
    else:
        selected = torch.empty(0, dtype=torch.long, device=device)

    if selected.numel() > keep:
        order = torch.topk(prior[selected].float(), k=keep).indices
        selected = selected[order]
    elif selected.numel() < keep:
        mask = torch.zeros(num_points, dtype=torch.bool, device=device)
        mask[selected] = True
        remaining = torch.where(~mask)[0]
        topup = min(keep - int(selected.numel()), int(remaining.numel()))
        if topup > 0:
            order = torch.topk(prior[remaining].float(), k=topup).indices
            selected = torch.cat([selected, remaining[order]], dim=0)
    return selected


@torch.no_grad()
def build_rendered_landmark_bank(
    model: HybridFeatureGaussian,
    sp_head: SuperPointOutputHead,
    renderer,
    dataset: CambridgeHybridDataset,
    max_landmarks: int,
    ref_views: int,
    stride: int,
    alpha_threshold: float,
    device: torch.device,
    selection: str = "global",
    per_view_quota: int = 0,
    view_grid_size: int = 8,
    descriptor_source: str = "hybrid",
    ply_loc_feature_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xyz_chunks = []
    desc_chunks = []
    prior_chunks = []
    max_ref_views = len(dataset) if int(ref_views) <= 0 else min(len(dataset), int(ref_views))
    auto_quota = math.ceil(float(max_landmarks) / float(max(max_ref_views, 1)))
    raw_count = 0
    pre_global_count = 0
    for idx in tqdm(range(max_ref_views), desc="Rendering landmark bank", dynamic_ncols=True):
        item = dataset[idx]
        pose = item["pose_w2c"].to(device=device, dtype=torch.float32).unsqueeze(0)
        render = render_hybrid_superpoint(
            model,
            sp_head,
            renderer,
            pose,
            descriptor_source=descriptor_source,
            ply_loc_feature_weight=ply_loc_feature_weight,
        )
        world_points, _grid_yx, valid_depth = unproject_dense_depth_to_world(
            render["depth"].float(),
            pose,
            renderer.K.float(),
        )
        alpha = render["alpha"][0].float() * valid_depth[0].view_as(render["alpha"][0]).float()
        xyz, desc, ids = flatten_rendered_landmarks(
            render["descriptor"][0],
            world_points[0],
            alpha,
            stride=stride,
            alpha_threshold=alpha_threshold,
        )
        if xyz.numel() == 0:
            continue
        if render["locability"] is not None:
            prior = render["locability"][0, 0].flatten()[ids]
        else:
            prior = torch.ones(xyz.shape[0], device=device)
        raw_count += int(xyz.shape[0])
        if selection != "global":
            quota = int(per_view_quota) if int(per_view_quota) > 0 else auto_quota
            view_keep = select_view_landmark_indices(
                prior,
                ids,
                height=int(alpha.shape[-2]),
                width=int(alpha.shape[-1]),
                quota=quota,
                selection=selection,
                grid_size=view_grid_size,
            )
            xyz = xyz[view_keep]
            desc = desc[view_keep]
            prior = prior[view_keep]
        pre_global_count += int(xyz.shape[0])
        xyz_chunks.append(xyz)
        desc_chunks.append(desc)
        prior_chunks.append(prior)
    if not xyz_chunks:
        raise RuntimeError("Rendered landmark bank is empty; lower alpha_threshold or check the checkpoint")
    xyz_all = torch.cat(xyz_chunks, dim=0)
    desc_all = torch.cat(desc_chunks, dim=0)
    prior_all = torch.cat(prior_chunks, dim=0).clamp(0.0, 1.0)
    keep = min(int(max_landmarks), xyz_all.shape[0])
    if keep < xyz_all.shape[0]:
        _, ids = torch.topk(prior_all, k=keep)
        xyz_all = xyz_all[ids]
        desc_all = desc_all[ids]
        prior_all = prior_all[ids]
    print(
        f"[eval] rendered landmark bank selected: raw={raw_count}, "
        f"pre_global={pre_global_count}, kept={xyz_all.shape[0]}, selection={selection}"
    )
    return xyz_all, desc_all, prior_all


@torch.no_grad()
def localize_one(
    model: HybridFeatureGaussian,
    sp_head: SuperPointOutputHead,
    renderer,
    teacher: SuperPointNet,
    rgb: torch.Tensor,
    K_feature: torch.Tensor,
    K_full: torch.Tensor,
    landmark_xyz: torch.Tensor,
    landmark_desc: torch.Tensor,
    landmark_prior: torch.Tensor,
    args: argparse.Namespace,
    image_name: str = "",
    teacher_cache: SuperPointTeacherCache | None = None,
    full_renderer=None,
    stdloc_detector: StdlocKeypointDetector | None = None,
    teacher_rgb: torch.Tensor | None = None,
) -> dict[str, object]:
    if hasattr(renderer, "K"):
        renderer.K.copy_(K_feature.to(device=renderer.K.device, dtype=renderer.K.dtype))
    if full_renderer is not None and hasattr(full_renderer, "K"):
        full_renderer.K.copy_(K_full.to(device=full_renderer.K.device, dtype=full_renderer.K.dtype))
    teacher_input = teacher_rgb if teacher_rgb is not None else rgb
    if teacher_cache is not None and image_name and teacher_rgb is None:
        teacher_desc, teacher_det, _cache_hits = extract_superpoint_teacher_batch(
            teacher,
            teacher_input,
            [image_name],
            cache=teacher_cache,
        )
    else:
        teacher_desc, teacher_det = teacher(superpoint_gray(teacher_input))
    teacher_desc_raw = F.normalize(teacher_desc[0].float(), dim=0)
    teacher_desc = teacher_desc_raw
    feature_height = int(getattr(renderer, "image_height", teacher_desc.shape[-2]))
    feature_width = int(getattr(renderer, "image_width", teacher_desc.shape[-1]))
    if teacher_desc.shape[-2:] != (feature_height, feature_width):
        teacher_desc = F.normalize(
            F.interpolate(
                teacher_desc_raw.unsqueeze(0),
                size=(feature_height, feature_width),
                mode="bilinear",
                align_corners=False,
            )[0],
            p=2,
            dim=0,
        )
    teacher_det = teacher_det[0].float()
    query_detector = getattr(args, "query_detector", "superpoint")
    sparse_K = K_feature
    sparse_query_offset = 0.0
    query_keypoints_are_full_res = False
    if query_detector == "stdloc":
        if stdloc_detector is None:
            raise ValueError("query_detector=stdloc requires a loaded StdlocKeypointDetector")
        full_h = int(rgb.shape[-2])
        full_w = int(rgb.shape[-1])
        query_feature_map = F.normalize(
            upsample_feature_map(teacher_desc_raw, full_h, full_w),
            p=2,
            dim=0,
        )
        keypoints, kp_scores = extract_stdloc_detector_keypoints(
            query_feature_map,
            stdloc_detector,
            max_keypoints=args.query_keypoints,
            nms_radius=args.nms_radius,
        )
        query_desc = sample_descriptors_bilinear(query_feature_map, keypoints)
        sparse_K = K_full
        sparse_query_offset = 0.5
        query_keypoints_are_full_res = True
    else:
        keypoints, kp_scores = extract_keypoints_from_detector_logits(
            teacher_det,
            max_keypoints=args.query_keypoints,
            confidence_threshold=args.keypoint_threshold,
            nms_radius=args.nms_radius,
        )
        query_desc = sample_descriptors_bilinear(teacher_desc, keypoints)
    if teacher_cache is not None and image_name:
        maybe_write_superpoint_metadata(
            teacher_cache,
            [image_name],
            teacher_desc.unsqueeze(0),
            teacher_det.unsqueeze(0),
            keypoints.unsqueeze(0),
            query_desc.unsqueeze(0),
        )
    sparse_margin = (
        args.sparse_match_second_best_margin
        if args.sparse_match_second_best_margin is not None
        else args.match_second_best_margin
    )
    dense_margin = (
        args.dense_match_second_best_margin
        if args.dense_match_second_best_margin is not None
        else args.match_second_best_margin
    )
    dense_query_offset = float(getattr(args, "dense_query_pixel_center_offset", 0.0))
    refine_poselib = bool(getattr(args, "poselib_refine", False))
    pnp_min_iterations = int(getattr(args, "pnp_min_iterations", 0))
    sparse_pnp_iterations = getattr(args, "sparse_pnp_iterations", None)
    sparse_pnp_min_iterations = getattr(args, "sparse_pnp_min_iterations", None)
    dense_pnp_iterations = getattr(args, "dense_pnp_iterations", None)
    dense_pnp_min_iterations = getattr(args, "dense_pnp_min_iterations", None)
    sparse_pnp_iterations = int(args.pnp_iterations if sparse_pnp_iterations is None else sparse_pnp_iterations)
    sparse_pnp_min_iterations = int(pnp_min_iterations if sparse_pnp_min_iterations is None else sparse_pnp_min_iterations)
    dense_pnp_iterations = int(args.pnp_iterations if dense_pnp_iterations is None else dense_pnp_iterations)
    dense_pnp_min_iterations = int(pnp_min_iterations if dense_pnp_min_iterations is None else dense_pnp_min_iterations)
    descriptor_source = getattr(args, "descriptor_source", "hybrid")
    ply_loc_feature_weight = float(getattr(args, "ply_loc_feature_weight", 0.5))
    sparse_reprojection_error = (
        args.sparse_reprojection_error
        if args.sparse_reprojection_error is not None
        else args.reprojection_error
    )
    dense_reprojection_error = (
        args.dense_reprojection_error
        if args.dense_reprojection_error is not None
        else args.reprojection_error
    )
    sparse_matcher, dense_matcher = resolve_matchers(args)
    if sparse_matcher == "stdloc_parity":
        corr = query_desc.float() @ F.normalize(landmark_desc.float(), dim=-1).T
        corr = apply_match_prior(corr.unsqueeze(0), landmark_prior, weight=args.locability_prior_weight)
        _b, q_ids, lm_ids, _scores = match_correlation_matrix(
            corr,
            threshold=args.sparse_match_threshold,
            dual_softmax_temp=args.sparse_dual_softmax_temp,
            use_dual_softmax=args.sparse_dual_softmax,
            use_mnn=args.mnn,
            topk=1,
        )
    elif sparse_matcher in {"lightglue", "dim"}:
        # LightGlue needs two 2D feature sets.  The global 3D landmark bank has no
        # query-view 2D layout, so sparse initialization remains top-k and the
        # learned matcher is applied in the rendered refinement stage.
        q_ids, lm_ids, _scores = match_descriptors_topk(
            query_desc,
            landmark_desc,
            topk=1,
            threshold=args.sparse_match_threshold,
            landmark_prior=landmark_prior,
            prior_weight=args.locability_prior_weight,
            second_best_margin=sparse_margin,
        )
    else:
        q_ids, lm_ids, _scores = match_descriptors_topk(
            query_desc,
            landmark_desc,
            topk=1,
            threshold=args.sparse_match_threshold,
            landmark_prior=landmark_prior,
            prior_weight=args.locability_prior_weight,
            second_best_margin=sparse_margin,
        )
    sparse_pose, sparse_inliers = solve_pnp_ransac(
        landmark_xyz[lm_ids].detach().cpu().numpy(),
        (keypoints[q_ids] + sparse_query_offset).detach().cpu().numpy(),
        sparse_K.detach().cpu().numpy(),
        reprojection_error=sparse_reprojection_error,
        refine_reprojection_error=args.refine_reprojection_error,
        confidence=args.pnp_confidence,
        iterations=sparse_pnp_iterations,
        min_iterations=sparse_pnp_min_iterations,
        solver=args.solver,
        refine_poselib=refine_poselib,
    )
    if sparse_pose is None:
        return {"pose_w2c": None, "sparse_pose_w2c": None, "sparse_inliers": 0, "dense_inliers": 0}

    pose = torch.from_numpy(sparse_pose).to(device=rgb.device, dtype=torch.float32).unsqueeze(0)
    dense_inliers = 0
    for _ in range(args.dense_iters):
        render = render_hybrid_superpoint(
            model,
            sp_head,
            renderer,
            pose,
            descriptor_source=descriptor_source,
            ply_loc_feature_weight=ply_loc_feature_weight,
        )
        desc_map = render["descriptor"][0]
        loc_map = render["locability"]
        if dense_matcher == "stdloc_parity":
            full_h = int(rgb.shape[-2])
            full_w = int(rgb.shape[-1])
            query_fine = F.normalize(
                upsample_feature_map(teacher_desc_raw, full_h, full_w),
                p=2,
                dim=0,
            )
            if full_renderer is not None:
                full_render = render_hybrid_superpoint(
                    model,
                    sp_head,
                    full_renderer,
                    pose,
                    descriptor_source=descriptor_source,
                    ply_loc_feature_weight=ply_loc_feature_weight,
                )
                rendered_fine = full_render["descriptor"][0]
                dense_depth = full_render["depth"][0].float()
                full_loc_map = full_render["locability"]
                rendered_prior = full_loc_map[0, 0] if full_loc_map is not None else None
                rendered_coarse = F.interpolate(
                    rendered_fine.unsqueeze(0).float(),
                    size=teacher_desc.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0]
            else:
                rendered_fine = upsample_feature_map(desc_map, full_h, full_w)
                dense_depth = F.interpolate(
                    render["depth"].unsqueeze(1).float(),
                    size=(full_h, full_w),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                rendered_prior = (
                    upsample_feature_map(loc_map[0, 0].unsqueeze(0), full_h, full_w).squeeze(0)
                    if loc_map is not None
                    else None
                )
                rendered_coarse = desc_map
            try:
                dense_matches = coarse_to_fine_dense_matches(
                    query_fine,
                    rendered_fine,
                    query_coarse_map=teacher_desc,
                    rendered_coarse_map=rendered_coarse,
                    rendered_prior=rendered_prior,
                    prior_weight=args.locability_prior_weight,
                    window_size=8,
                    coarse_dual_softmax_temp=args.dense_dual_softmax_temp,
                    fine_dual_softmax_temp=args.fine_dual_softmax_temp,
                    coarse_threshold=args.dense_match_threshold,
                    fine_threshold=args.dense_match_threshold,
                    use_mnn=True,
                    subpixel_refine=args.subpixel_refine,
                    subpixel_temperature=args.subpixel_temperature,
                )
            except ValueError:
                break
            if dense_matches.query_yx.shape[0] < 4:
                break
            p3d = unproject_positions_yx(
                dense_matches.rendered_yx,
                dense_depth,
                pose[0].float(),
                K_full.float(),
                pixel_center_offset=args.render_pixel_center_offset,
            )
            valid = torch.isfinite(p3d).all(dim=-1) & (p3d[:, 2].abs() < 1e8)
            if valid.sum() < 4:
                break
            dense_pose, dense_inliers = solve_pnp_ransac(
                p3d[valid].detach().cpu().numpy(),
                (
                    dense_matches.query_yx[valid]
                    + dense_query_offset
                ).detach().cpu().numpy(),
                K_full.detach().cpu().numpy(),
                reprojection_error=dense_reprojection_error,
            refine_reprojection_error=args.refine_reprojection_error,
            confidence=args.pnp_confidence,
            iterations=dense_pnp_iterations,
            min_iterations=dense_pnp_min_iterations,
            solver=args.solver,
            refine_poselib=refine_poselib,
        )
        elif dense_matcher == "lightglue_rendered":
            full_h = int(rgb.shape[-2])
            full_w = int(rgb.shape[-1])
            if full_renderer is not None:
                full_render = render_hybrid_superpoint(
                    model,
                    sp_head,
                    full_renderer,
                    pose,
                    descriptor_source=descriptor_source,
                    ply_loc_feature_weight=ply_loc_feature_weight,
                )
                dense_desc_map = full_render["descriptor"][0]
                dense_depth = full_render["depth"][0].float()
                dense_loc_map = full_render["locability"]
                rendered_detector = full_render["detector"][0]
                dense_K = K_full
                dense_renderer_K = full_renderer.K.float()
                query_lg_yx = keypoints if query_keypoints_are_full_res else keypoints * 8.0
                query_lg_map = F.normalize(upsample_feature_map(teacher_desc_raw, full_h, full_w), p=2, dim=0)
                query_lg_desc = sample_descriptors_bilinear(query_lg_map, query_lg_yx)
            else:
                dense_desc_map = desc_map
                dense_depth = render["depth"][0].float()
                dense_loc_map = loc_map
                rendered_detector = render["detector"][0]
                dense_K = K_feature
                dense_renderer_K = renderer.K.float()
                query_lg_yx = keypoints / 8.0 if query_keypoints_are_full_res else keypoints
                query_lg_desc = query_desc
            try:
                rendered_kpts = select_rendered_keypoints(
                    dense_desc_map,
                    source=args.rendered_keypoint_source,
                    max_keypoints=args.lightglue_max_keypoints,
                    threshold=args.dense_match_threshold,
                    nms_radius=args.nms_radius,
                    detector_logits=rendered_detector,
                    locability=dense_loc_map,
                )
            except ValueError:
                break
            if rendered_kpts.keypoints_yx.shape[0] < 4 or query_lg_yx.shape[0] < 4:
                break
            q_dense, rendered_ids, _lg_scores = match_lightglue_descriptors(
                query_lg_yx,
                query_lg_desc,
                rendered_kpts.keypoints_yx,
                rendered_kpts.descriptors,
                image_hw=tuple(int(v) for v in dense_depth.shape),
                rendered_hw=tuple(int(v) for v in dense_depth.shape),
                feature_name=lightglue_feature_name(args.dim_pipeline),
                filter_threshold=args.lightglue_filter_threshold,
            )
            if rendered_ids.numel() < 4:
                break
            refined_render_yx = rendered_kpts.keypoints_yx[rendered_ids]
            dense_points3d = unproject_positions_yx(
                refined_render_yx,
                dense_depth,
                pose[0].float(),
                dense_renderer_K,
                pixel_center_offset=args.render_pixel_center_offset,
            )
            valid = torch.isfinite(dense_points3d).all(dim=-1)
            if valid.sum() < 4:
                break
            dense_pose, dense_inliers = solve_pnp_ransac(
                dense_points3d[valid].detach().cpu().numpy(),
                (query_lg_yx[q_dense][valid] + dense_query_offset).detach().cpu().numpy(),
                dense_K.detach().cpu().numpy(),
                reprojection_error=dense_reprojection_error,
                refine_reprojection_error=args.refine_reprojection_error,
                confidence=args.pnp_confidence,
                iterations=dense_pnp_iterations,
                min_iterations=dense_pnp_min_iterations,
                solver=args.solver,
                refine_poselib=refine_poselib,
            )
        else:
            if args.dense_full_render and full_renderer is not None:
                full_render = render_hybrid_superpoint(
                    model,
                    sp_head,
                    full_renderer,
                    pose,
                    descriptor_source=descriptor_source,
                    ply_loc_feature_weight=ply_loc_feature_weight,
                )
                dense_desc_map = full_render["descriptor"][0]
                dense_depth = full_render["depth"][0].float()
                dense_loc_map = full_render["locability"]
                dense_query_yx = keypoints * 8.0
                dense_K = K_full
                dense_renderer_K = full_renderer.K.float()
            else:
                dense_desc_map = desc_map
                dense_depth = render["depth"][0].float()
                dense_loc_map = loc_map
                dense_query_yx = keypoints / 8.0 if query_keypoints_are_full_res else keypoints
                dense_K = K_feature
                dense_renderer_K = renderer.K.float()
            rendered_flat = dense_desc_map.flatten(1).T
            prior = dense_loc_map[0, 0].flatten() if dense_loc_map is not None else None
            q_dense, pix_ids, _dense_scores = match_descriptors_topk(
                query_desc,
                rendered_flat,
                topk=1,
                threshold=args.dense_match_threshold,
                landmark_prior=prior,
                prior_weight=args.locability_prior_weight,
                second_best_margin=dense_margin,
            )
            if pix_ids.numel() < 4:
                break
            depth_flat = dense_depth.reshape(-1)
            valid = torch.isfinite(depth_flat[pix_ids]) & (depth_flat[pix_ids] > 0.0)
            if valid.sum() < 4:
                break
            q_dense = q_dense[valid]
            pix_ids = pix_ids[valid]
            if args.subpixel_refine:
                refined_render_yx = refine_rendered_positions_softargmax(
                    dense_desc_map,
                    query_desc[q_dense],
                    pix_ids,
                    window_radius=args.topk_refine_window,
                    temperature=args.subpixel_temperature,
                )
            else:
                height, width = dense_depth.shape
                refined_render_yx = torch.stack(
                    [(pix_ids // width).float(), (pix_ids % width).float()],
                    dim=-1,
                )
            dense_points3d = unproject_positions_yx(
                refined_render_yx,
                dense_depth,
                pose[0].float(),
                dense_renderer_K,
                pixel_center_offset=args.render_pixel_center_offset,
            )
            dense_pose, dense_inliers = solve_pnp_ransac(
                dense_points3d.detach().cpu().numpy(),
                (
                    dense_query_yx[q_dense]
                    + dense_query_offset
                ).detach().cpu().numpy(),
                dense_K.detach().cpu().numpy(),
                reprojection_error=dense_reprojection_error,
                refine_reprojection_error=args.refine_reprojection_error,
                confidence=args.pnp_confidence,
                iterations=dense_pnp_iterations,
                min_iterations=dense_pnp_min_iterations,
                solver=args.solver,
                refine_poselib=refine_poselib,
            )
        if dense_pose is None:
            break
        pose = torch.from_numpy(dense_pose).to(device=rgb.device, dtype=torch.float32).unsqueeze(0)

    return {
        "pose_w2c": pose[0].detach().cpu().numpy(),
        "sparse_pose_w2c": sparse_pose,
        "sparse_inliers": sparse_inliers,
        "dense_inliers": dense_inliers,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Cambridge hybrid Loc-GS localization")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", default="")
    parser.add_argument("--data_root", default="/mnt/pool/sqy/Cambridge_stdloc")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--eval_pose_source", choices=["cambridge", "cameras_json"], default="cambridge")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--query_offset", type=int, default=0)
    parser.add_argument("--query_stride", type=int, default=1)
    parser.add_argument("--landmark_source", choices=["rendered", "gaussian", "stdloc_detector"], default="rendered")
    parser.add_argument("--stdloc_detector_dir", default="")
    parser.add_argument("--landmark_selection", choices=["global", "per_view", "per_view_spatial"], default="global")
    parser.add_argument("--max_landmarks", type=int, default=200000)
    parser.add_argument("--landmark_ref_views", type=int, default=20)
    parser.add_argument("--landmark_stride", type=int, default=2)
    parser.add_argument("--landmark_per_view_quota", type=int, default=0)
    parser.add_argument("--landmark_view_grid_size", type=int, default=8)
    parser.add_argument("--descriptor_source", choices=["hybrid", "ply_loc", "hybrid_ply_blend"], default="hybrid")
    parser.add_argument("--ply_loc_feature_weight", type=float, default=0.5)
    parser.add_argument("--alpha_threshold", type=float, default=0.05)
    parser.add_argument("--query_keypoints", type=int, default=2048)
    parser.add_argument("--query_detector", choices=["superpoint", "stdloc"], default="superpoint")
    parser.add_argument("--query_feature_source", choices=["resized", "original"], default="resized")
    parser.add_argument("--stdloc_detector_path", default="")
    parser.add_argument("--keypoint_threshold", type=float, default=0.015)
    parser.add_argument("--nms_radius", type=int, default=4)
    parser.add_argument("--sparse_match_threshold", type=float, default=0.0)
    parser.add_argument("--dense_match_threshold", type=float, default=0.0)
    parser.add_argument("--locability_prior_weight", type=float, default=0.05)
    parser.add_argument("--dense_iters", type=int, default=2)
    parser.add_argument("--matcher", choices=["stdloc_parity", "topk"], default="topk")
    parser.add_argument("--sparse_matcher", choices=["", *SPARSE_MATCHERS], default="")
    parser.add_argument("--dense_matcher", choices=["", *DENSE_MATCHERS], default="")
    parser.add_argument("--dim_pipeline", choices=DIM_PIPELINES, default="superpoint+lightglue")
    parser.add_argument("--rendered_keypoint_source", choices=["locability", "detector", "projected_gaussian"], default="locability")
    parser.add_argument("--lightglue_max_keypoints", type=int, default=2048)
    parser.add_argument("--lightglue_filter_threshold", type=float, default=0.1)
    parser.add_argument("--solver", choices=["poselib", "opencv"], default="opencv")
    parser.add_argument("--poselib_refine", action="store_true")
    parser.add_argument("--sparse_dual_softmax", action="store_true")
    parser.add_argument("--sparse_dual_softmax_temp", type=float, default=0.1)
    parser.add_argument("--dense_dual_softmax_temp", type=float, default=0.1)
    parser.add_argument("--fine_dual_softmax_temp", type=float, default=0.1)
    parser.add_argument("--mnn", action="store_true")
    parser.add_argument("--subpixel_refine", action="store_true")
    parser.add_argument("--subpixel_temperature", type=float, default=0.1)
    parser.add_argument("--topk_refine_window", type=int, default=1)
    parser.add_argument("--render_pixel_center_offset", type=float, default=0.0)
    parser.add_argument("--dense_query_pixel_center_offset", type=float, default=0.0)
    parser.add_argument("--dense_full_render", action="store_true")
    parser.add_argument("--superpoint_cache_root", default="")
    parser.add_argument("--disable_superpoint_cache", action="store_true")
    parser.add_argument("--match_second_best_margin", type=float, default=0.0)
    parser.add_argument("--sparse_match_second_best_margin", type=float, default=None)
    parser.add_argument("--dense_match_second_best_margin", type=float, default=None)
    parser.add_argument("--reprojection_error", type=float, default=2.0)
    parser.add_argument("--sparse_reprojection_error", type=float, default=None)
    parser.add_argument("--dense_reprojection_error", type=float, default=None)
    parser.add_argument("--refine_reprojection_error", type=float, default=0.0)
    parser.add_argument("--pnp_confidence", type=float, default=0.9999)
    parser.add_argument("--pnp_iterations", type=int, default=10000)
    parser.add_argument("--pnp_min_iterations", type=int, default=0)
    parser.add_argument("--sparse_pnp_iterations", type=int, default=None)
    parser.add_argument("--sparse_pnp_min_iterations", type=int, default=None)
    parser.add_argument("--dense_pnp_iterations", type=int, default=None)
    parser.add_argument("--dense_pnp_min_iterations", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _summary(errors_te: list[float], errors_ae: list[float], inliers: list[int]) -> dict[str, float]:
    te = np.asarray(errors_te, dtype=np.float64)
    ae = np.asarray(errors_ae, dtype=np.float64)
    return {
        "median_te": float(np.median(te)) if len(te) else float("inf"),
        "median_ae": float(np.median(ae)) if len(ae) else float("inf"),
        "recall_5m_10d": float(((te <= 500.0) & (ae <= 10.0)).mean()) if len(te) else 0.0,
        "recall_2m_5d": float(((te <= 200.0) & (ae <= 5.0)).mean()) if len(te) else 0.0,
        "recall_5cm_5d": float(((te <= 5.0) & (ae <= 5.0)).mean()) if len(te) else 0.0,
        "recall_2cm_2d": float(((te <= 2.0) & (ae <= 2.0)).mean()) if len(te) else 0.0,
        "avg_inliers": float(np.mean(inliers)) if inliers else 0.0,
    }


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = build_argparser().parse_args() if args is None else args
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt["args"]
    scene = args.scene or train_args["scene"]
    scene_root = Path(args.data_root) / scene
    output_dir = Path(args.output_dir) if args.output_dir else Path(train_args["output_dir"]) / "eval_cambridge"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_intr = train_args["feature_intrinsics"]
    full_fx = feature_intr["fx"] * 8.0
    full_fy = feature_intr["fy"] * 8.0
    full_cx = feature_intr["cx"] * 8.0
    full_cy = feature_intr["cy"] * 8.0
    if args.query_offset < 0:
        raise ValueError("--query_offset must be non-negative")
    if args.query_stride < 1:
        raise ValueError("--query_stride must be at least 1")
    need_query_subset = args.query_offset > 0 or args.query_stride > 1
    eval_cameras_json = train_args["cameras_json"] if args.eval_pose_source == "cameras_json" else None
    dataset = CambridgeHybridDataset(
        scene_root=scene_root,
        cameras_json=eval_cameras_json,
        split="test",
        image_subdir=train_args.get("image_subdir", "processed"),
        image_height=train_args["image_height"],
        image_width=train_args["image_width"],
        fx=full_fx,
        fy=full_fy,
        cx=full_cx,
        cy=full_cy,
        max_frames=0 if need_query_subset else args.max_queries,
    )
    if need_query_subset:
        indices = list(range(args.query_offset, len(dataset), args.query_stride))
        if args.max_queries > 0:
            indices = indices[: args.max_queries]
        dataset = Subset(dataset, indices)

    from loc_gs.rendering.feature_renderer import FeatureFieldRenderer

    renderer = FeatureFieldRenderer(
        image_height=train_args["feature_height"],
        image_width=train_args["feature_width"],
        fx=feature_intr["fx"],
        fy=feature_intr["fy"],
        cx=feature_intr["cx"],
        cy=feature_intr["cy"],
    ).to(device)
    _, dense_matcher_name = resolve_matchers(args)
    full_renderer = None
    if args.dense_full_render or dense_matcher_name == "lightglue_rendered":
        use_stdloc_render = dense_matcher_name == "stdloc_parity"
        full_renderer = FeatureFieldRenderer(
            image_height=train_args["image_height"],
            image_width=train_args["image_width"],
            fx=full_fx,
            fy=full_fy,
            cx=full_cx,
            cy=full_cy,
            max_channels_per_chunk=32,
            far_plane=10000.0 if use_stdloc_render else 100.0,
            packed=not use_stdloc_render,
            rasterize_mode="antialiased" if use_stdloc_render else "classic",
        ).to(device)
    model = HybridFeatureGaussian(
        latent_dim=train_args["latent_dim"],
        hash_output_dim=train_args["hash_output_dim"],
        fine_dim=train_args["fine_dim"],
        coarse_dim=train_args["coarse_dim"],
        output_dim=train_args["hybrid_output_dim"],
    )
    model.load_from_ply(train_args["ply_path"])
    model = model.to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    sp_head = SuperPointOutputHead(
        fused_dim=train_args["hybrid_output_dim"],
        descriptor_dim=256,
        detector_dim=65,
        hidden_dim=256,
        num_res_blocks=2,
        use_3x3=True,
    ).to(device)
    sp_head.load_state_dict(ckpt["sp_head_state_dict"])
    sp_head.eval()
    teacher = SuperPointNet().to(device)
    teacher.load_state_dict(torch.load(train_args["superpoint_weights"], map_location=device), strict=False)
    teacher.eval()
    stdloc_detector = None
    if args.query_detector == "stdloc":
        detector_path = (
            Path(args.stdloc_detector_path)
            if args.stdloc_detector_path
            else Path(train_args["ply_path"]).parents[2] / "detector" / "30000_detector.pth"
        )
        if not detector_path.exists():
            raise FileNotFoundError(f"STDLoc query detector not found: {detector_path}")
        stdloc_detector = StdlocKeypointDetector(in_dim=256).to(device)
        stdloc_detector.load_state_dict(torch.load(detector_path, map_location=device))
        stdloc_detector.eval()
    teacher_cache = None
    if not args.disable_superpoint_cache:
        teacher_cache = SuperPointTeacherCache(
            args.superpoint_cache_root or train_args.get("output_root", "output"),
            scene=scene,
            split="test",
        )

    if args.landmark_source == "rendered":
        train_ref_dataset = CambridgeHybridDataset(
            scene_root=scene_root,
            cameras_json=train_args["cameras_json"],
            split="train",
            image_subdir=train_args.get("image_subdir", "processed"),
            image_height=train_args["image_height"],
            image_width=train_args["image_width"],
            max_frames=args.landmark_ref_views,
        )
        print(
            f"[eval] rendering landmark bank from {len(train_ref_dataset)} reference views "
            f"(max_landmarks={args.max_landmarks})"
        )
        landmark_xyz, landmark_desc, landmark_prior = build_rendered_landmark_bank(
            model,
            sp_head,
            renderer,
            train_ref_dataset,
            max_landmarks=args.max_landmarks,
            ref_views=args.landmark_ref_views,
            stride=args.landmark_stride,
            alpha_threshold=args.alpha_threshold,
            device=device,
            selection=args.landmark_selection,
            per_view_quota=args.landmark_per_view_quota,
            view_grid_size=args.landmark_view_grid_size,
            descriptor_source=args.descriptor_source,
            ply_loc_feature_weight=args.ply_loc_feature_weight,
        )
    elif args.landmark_source == "gaussian":
        print(f"[eval] decoding up to {args.max_landmarks} Gaussian-center landmark descriptors")
        landmark_xyz, landmark_desc, landmark_prior = decode_landmark_descriptors(
            model,
            sp_head,
            max_landmarks=args.max_landmarks,
        )
    else:
        detector_dir = (
            Path(args.stdloc_detector_dir)
            if args.stdloc_detector_dir
            else Path(train_args["ply_path"]).parents[2] / "detector"
        )
        print(f"[eval] loading STDLoc detector landmarks from {detector_dir}")
        landmark_xyz, landmark_desc, landmark_prior = build_stdloc_detector_landmark_bank(
            model,
            sp_head,
            detector_dir=detector_dir,
            max_landmarks=args.max_landmarks,
            descriptor_source=args.descriptor_source,
            ply_loc_feature_weight=args.ply_loc_feature_weight,
            device=device,
        )

    sparse_te: list[float] = []
    sparse_ae: list[float] = []
    sparse_inliers: list[int] = []
    dense_te: list[float] = []
    dense_ae: list[float] = []
    dense_inliers: list[int] = []
    details = []

    for item in tqdm(dataset, desc="Evaluating Cambridge hybrid", dynamic_ncols=True):
        rgb = item["rgb"].unsqueeze(0).to(device)
        teacher_rgb = None
        if args.query_feature_source == "original":
            teacher_rgb = load_cambridge_rgb_no_resize(
                scene_root,
                train_args.get("image_subdir", "processed"),
                str(item["image_name"]),
                device,
            ).unsqueeze(0)
        gt_pose = item["pose_w2c"].numpy()
        K_feature = item["feature_K"].to(device)
        result = localize_one(
            model,
            sp_head,
            renderer,
            teacher,
            rgb,
            K_feature,
            item["K"].to(device),
            landmark_xyz,
            landmark_desc,
            landmark_prior,
            args,
            image_name=item["image_name"],
            teacher_cache=None if teacher_rgb is not None else teacher_cache,
            full_renderer=full_renderer,
            stdloc_detector=stdloc_detector,
            teacher_rgb=teacher_rgb,
        )
        sparse_te_i = None
        sparse_ae_i = None
        dense_te_i = None
        dense_ae_i = None
        if result["sparse_pose_w2c"] is not None:
            te, ae = pose_error_cm_deg(result["sparse_pose_w2c"], gt_pose)
            sparse_te_i = float(te)
            sparse_ae_i = float(ae)
            sparse_te.append(te)
            sparse_ae.append(ae)
            sparse_inliers.append(int(result["sparse_inliers"]))
        if result["pose_w2c"] is not None:
            te, ae = pose_error_cm_deg(result["pose_w2c"], gt_pose)
            dense_te_i = float(te)
            dense_ae_i = float(ae)
            dense_te.append(te)
            dense_ae.append(ae)
            dense_inliers.append(int(result["dense_inliers"]))
        details.append(
            {
                "image_name": item["image_name"],
                "sparse_te": sparse_te_i,
                "sparse_ae": sparse_ae_i,
                "sparse_inliers": int(result["sparse_inliers"]),
                "dense_te": dense_te_i,
                "dense_ae": dense_ae_i,
                "dense_inliers": int(result["dense_inliers"]),
                "localized": result["pose_w2c"] is not None,
            }
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "scene": scene,
        "eval_pose_source": args.eval_pose_source,
        "query_offset": int(args.query_offset),
        "query_stride": int(args.query_stride),
        "landmark_source": args.landmark_source,
        "landmark_selection": args.landmark_selection,
        "query_detector": args.query_detector,
        "query_feature_source": args.query_feature_source,
        "descriptor_source": args.descriptor_source,
        "ply_loc_feature_weight": float(args.ply_loc_feature_weight),
        "sparse_matcher": resolve_matchers(args)[0],
        "dense_matcher": resolve_matchers(args)[1],
        "dim_pipeline": args.dim_pipeline,
        "rendered_keypoint_source": args.rendered_keypoint_source,
        "landmarks": int(landmark_xyz.shape[0]),
        "sparse": _summary(sparse_te, sparse_ae, sparse_inliers),
        "dense": _summary(dense_te, dense_ae, dense_inliers),
        "localized": len(dense_te),
        "queries": len(dataset),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "results.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
