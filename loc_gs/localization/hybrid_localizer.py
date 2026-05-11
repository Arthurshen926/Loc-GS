from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class _CV2Fallback:
    SOLVEPNP_EPNP = 1
    SOLVEPNP_ITERATIVE = 0

    def solvePnPRansac(self, *_args, **_kwargs):
        raise ImportError("OpenCV is required for solvePnPRansac")

    def solvePnP(self, *_args, **_kwargs):
        raise ImportError("OpenCV is required for solvePnP")

    def Rodrigues(self, src):
        arr = np.asarray(src, dtype=np.float64)
        if arr.shape == (3, 3):
            return np.zeros((3, 1), dtype=np.float64), None
        vec = arr.reshape(3)
        theta = float(np.linalg.norm(vec))
        if theta < 1e-12:
            return np.eye(3, dtype=np.float64), None
        axis = vec / theta
        kx, ky, kz = axis
        K = np.asarray(
            [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
            dtype=np.float64,
        )
        R = np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
        return R, None

    def projectPoints(self, object_points, rvec, tvec, K, _dist_coeffs):
        R, _ = self.Rodrigues(rvec)
        pts = np.asarray(object_points, dtype=np.float64) @ R.T + np.asarray(tvec, dtype=np.float64).reshape(1, 3)
        z = np.where(np.abs(pts[:, 2]) < 1e-12, 1e-12, pts[:, 2])
        intr = np.asarray(K, dtype=np.float64)
        x = intr[0, 0] * pts[:, 0] / z + intr[0, 2]
        y = intr[1, 1] * pts[:, 1] / z + intr[1, 2]
        return np.stack([x, y], axis=-1).reshape(-1, 1, 2), None


class _LazyCV2:
    def __init__(self) -> None:
        self._module = None

    def _load(self):
        if self._module is None:
            try:
                import cv2 as cv2_module
            except ImportError:
                cv2_module = _CV2Fallback()
            self._module = cv2_module
        return self._module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)


cv2 = _LazyCV2()


def _cv2():
    return cv2


def sample_descriptors_bilinear(
    descriptor_map: torch.Tensor,
    keypoints_yx: torch.Tensor,
) -> torch.Tensor:
    C, H, W = descriptor_map.shape
    if keypoints_yx.numel() == 0:
        return descriptor_map.new_zeros((0, C))
    y = keypoints_yx[:, 0]
    x = keypoints_yx[:, 1]
    x_norm = 2.0 * x / max(W - 1, 1) - 1.0
    y_norm = 2.0 * y / max(H - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        descriptor_map.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
    return F.normalize(sampled, p=2, dim=-1)


def extract_keypoints_from_detector_logits(
    detector_logits: torch.Tensor,
    max_keypoints: int = 2048,
    confidence_threshold: float = 0.015,
    nms_radius: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(detector_logits.float(), dim=0)
    heatmap = F.pixel_shuffle(probs[:64].unsqueeze(0), 8).squeeze(0).squeeze(0)
    if nms_radius > 0:
        pooled = F.max_pool2d(
            heatmap.view(1, 1, *heatmap.shape),
            kernel_size=2 * nms_radius + 1,
            stride=1,
            padding=nms_radius,
        ).view_as(heatmap)
        heatmap = heatmap * (heatmap == pooled).float()

    mask = heatmap > confidence_threshold
    if mask.any():
        ys, xs = torch.where(mask)
        scores = heatmap[ys, xs]
    else:
        scores, flat = heatmap.reshape(-1).topk(min(max_keypoints, heatmap.numel()))
        ys = flat // heatmap.shape[1]
        xs = flat % heatmap.shape[1]
    order = scores.argsort(descending=True)[:max_keypoints]
    keypoints = torch.stack([ys[order].float() / 8.0, xs[order].float() / 8.0], dim=-1)
    return keypoints, scores[order]


def match_descriptors_topk(
    query_descriptors: torch.Tensor,
    landmark_descriptors: torch.Tensor,
    topk: int = 1,
    threshold: float = 0.0,
    landmark_prior: Optional[torch.Tensor] = None,
    prior_weight: float = 0.0,
    second_best_margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if query_descriptors.numel() == 0 or landmark_descriptors.numel() == 0:
        device = query_descriptors.device
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, device=device),
        )
    corr = F.normalize(query_descriptors.float(), dim=-1) @ F.normalize(landmark_descriptors.float(), dim=-1).T
    if landmark_prior is not None and prior_weight > 0.0:
        prior = landmark_prior.to(device=corr.device, dtype=corr.dtype).reshape(1, -1)
        prior = prior.clamp(0.0, 1.0) - prior.mean()
        corr = corr + float(prior_weight) * prior
    requested_topk = min(max(1, int(topk)), corr.shape[1])
    margin = max(float(second_best_margin), 0.0)
    corr_topk = requested_topk
    if margin > 0.0 and corr.shape[1] > requested_topk:
        corr_topk = requested_topk + 1
    values_all, indices_all = torch.topk(corr, k=corr_topk, dim=-1)
    values = values_all[:, :requested_topk]
    indices = indices_all[:, :requested_topk]
    q_ids = torch.arange(corr.shape[0], device=corr.device).repeat_interleave(values.shape[1])
    lm_ids = indices.reshape(-1)
    scores = values.reshape(-1)
    keep = scores >= float(threshold)
    if margin > 0.0 and corr_topk > requested_topk:
        second_best = values_all[:, requested_topk].repeat_interleave(requested_topk)
        keep = keep & ((scores - second_best) >= margin)
    return q_ids[keep], lm_ids[keep], scores[keep]


def refine_rendered_positions_softargmax(
    descriptor_map: torch.Tensor,
    query_descriptors: torch.Tensor,
    flat_pixel_ids: torch.Tensor,
    window_radius: int = 1,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Refine coarse rendered feature-grid matches with a local descriptor soft-argmax."""
    C, H, W = descriptor_map.shape
    if flat_pixel_ids.numel() == 0:
        return descriptor_map.new_zeros((0, 2))
    radius = max(0, int(window_radius))
    if radius == 0:
        y = (flat_pixel_ids // W).float()
        x = (flat_pixel_ids % W).float()
        return torch.stack([y, x], dim=-1)

    offsets_1d = torch.arange(-radius, radius + 1, device=descriptor_map.device)
    dy, dx = torch.meshgrid(offsets_1d, offsets_1d, indexing="ij")
    offsets = torch.stack([dy.reshape(-1), dx.reshape(-1)], dim=-1)
    center_y = flat_pixel_ids.to(device=descriptor_map.device) // W
    center_x = flat_pixel_ids.to(device=descriptor_map.device) % W
    patch_y = (center_y[:, None] + offsets[None, :, 0]).clamp(0, H - 1)
    patch_x = (center_x[:, None] + offsets[None, :, 1]).clamp(0, W - 1)

    desc_hw = descriptor_map.permute(1, 2, 0).contiguous()
    patch_desc = desc_hw[patch_y, patch_x]
    query = F.normalize(query_descriptors.to(device=descriptor_map.device).float(), dim=-1)
    patch_desc = F.normalize(patch_desc.float(), dim=-1)
    scores = (patch_desc * query[:, None, :]).sum(dim=-1)
    weights = F.softmax(scores / max(float(temperature), 1e-6), dim=-1)
    refined_y = (weights * patch_y.float()).sum(dim=-1)
    refined_x = (weights * patch_x.float()).sum(dim=-1)
    return torch.stack([refined_y, refined_x], dim=-1)


def flatten_rendered_landmarks(
    descriptor_map: torch.Tensor,
    world_points: torch.Tensor,
    alpha_map: torch.Tensor,
    stride: int = 1,
    alpha_threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten a rendered descriptor/depth view into 3D descriptor landmarks."""
    C, H, W = descriptor_map.shape
    if world_points.shape[0] != H * W:
        raise ValueError("world_points must have shape [H*W, 3]")
    ids_grid = torch.arange(H * W, device=descriptor_map.device).view(H, W)
    sample_mask = torch.zeros((H, W), device=descriptor_map.device, dtype=torch.bool)
    sample_mask[:: max(1, int(stride)), :: max(1, int(stride))] = True
    valid = sample_mask & (alpha_map.to(device=descriptor_map.device) > float(alpha_threshold))
    ids = ids_grid[valid]
    desc_flat = descriptor_map.flatten(1).T
    return world_points[ids], F.normalize(desc_flat[ids].float(), dim=-1), ids


def solve_pnp_ransac(
    points3d_world: np.ndarray,
    keypoints_yx: np.ndarray,
    K: np.ndarray,
    reprojection_error: float = 12.0,
    refine_reprojection_error: Optional[float] = None,
    confidence: float = 0.9999,
    iterations: int = 10000,
    min_iterations: int = 0,
    solver: str = "opencv",
    refine_poselib: bool = False,
    match_scores: Optional[np.ndarray] = None,
) -> tuple[Optional[np.ndarray], int]:
    if points3d_world.shape[0] < 4 or keypoints_yx.shape[0] < 4:
        return None, 0
    image_points_xy = np.stack([keypoints_yx[:, 1], keypoints_yx[:, 0]], axis=-1).astype(np.float64)
    object_points = points3d_world.astype(np.float64)
    if solver == "opencv_prosac" and match_scores is not None:
        scores = np.asarray(match_scores, dtype=np.float64).reshape(-1)
        if scores.shape[0] == object_points.shape[0]:
            order = np.argsort(np.where(np.isfinite(scores), scores, -np.inf))[::-1]
            object_points = object_points[order]
            image_points_xy = image_points_xy[order]
    if solver == "poselib":
        pose, inliers = _solve_pnp_poselib(
            object_points,
            image_points_xy,
            K,
            reprojection_error=reprojection_error,
            confidence=confidence,
            iterations=iterations,
            min_iterations=min_iterations,
        )
        if pose is not None:
            if refine_poselib:
                refined_pose, refined_inliers = _refine_pose_with_reprojection_inliers(
                    object_points,
                    image_points_xy,
                    K,
                    pose,
                    reprojection_error=reprojection_error,
                    refine_reprojection_error=refine_reprojection_error,
                )
                if refined_pose is not None:
                    return refined_pose, refined_inliers
            return pose, inliers
    cv2_module = _cv2()
    if solver == "opencv_prosac":
        params = cv2_module.UsacParams()
        params.confidence = float(confidence)
        params.maxIterations = int(iterations)
        params.threshold = float(reprojection_error)
        params.sampler = cv2_module.SAMPLING_PROSAC
        params.score = cv2_module.SCORE_METHOD_MSAC
        ok, _camera_matrix, rvec, tvec, inliers = cv2_module.solvePnPRansac(
            object_points,
            image_points_xy,
            K.astype(np.float64),
            None,
            None,
            None,
            None,
            params,
        )
    else:
        ok, rvec, tvec, inliers = cv2_module.solvePnPRansac(
            object_points,
            image_points_xy,
            K.astype(np.float64),
            None,
            iterationsCount=int(iterations),
            reprojectionError=float(reprojection_error),
            confidence=float(confidence),
            flags=cv2_module.SOLVEPNP_EPNP,
        )
    if not ok:
        return None, 0
    if inliers is not None and len(inliers) >= 4:
        refine_ids = inliers[:, 0]
        if refine_reprojection_error is not None and float(refine_reprojection_error) > 0.0:
            projected, _ = cv2_module.projectPoints(
                object_points[refine_ids],
                rvec,
                tvec,
                K.astype(np.float64),
                None,
            )
            projected = projected.reshape(-1, 2)
            err = np.linalg.norm(projected - image_points_xy[refine_ids], axis=-1)
            tight = err <= float(refine_reprojection_error)
            if int(tight.sum()) >= 4:
                refine_ids = refine_ids[tight]
        inlier_obj = object_points[refine_ids]
        inlier_img = image_points_xy[refine_ids]
        ok_refine, rvec, tvec = cv2_module.solvePnP(
            inlier_obj,
            inlier_img,
            K.astype(np.float64),
            None,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2_module.SOLVEPNP_ITERATIVE,
        )
        if not ok_refine:
            return None, 0
    R, _ = cv2_module.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = tvec[:, 0]
    if not np.isfinite(pose).all():
        return None, 0
    return pose.astype(np.float32), int(0 if inliers is None else len(inliers))


def _refine_pose_with_reprojection_inliers(
    object_points: np.ndarray,
    image_points_xy: np.ndarray,
    K: np.ndarray,
    init_pose: np.ndarray,
    reprojection_error: float,
    refine_reprojection_error: Optional[float],
) -> tuple[Optional[np.ndarray], int]:
    cv2 = _cv2()
    pose = np.asarray(init_pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return None, 0
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    tvec = pose[:3, 3].reshape(3, 1)
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K.astype(np.float64),
        None,
    )
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points_xy, axis=-1)
    threshold = (
        float(refine_reprojection_error)
        if refine_reprojection_error is not None and float(refine_reprojection_error) > 0.0
        else float(reprojection_error)
    )
    refine_ids = np.flatnonzero(errors <= threshold)
    if refine_ids.shape[0] < 4:
        return None, 0

    ok_refine, rvec, tvec = cv2.solvePnP(
        object_points[refine_ids],
        image_points_xy[refine_ids],
        K.astype(np.float64),
        None,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok_refine:
        return None, 0
    R, _ = cv2.Rodrigues(rvec)
    refined_pose = np.eye(4, dtype=np.float64)
    refined_pose[:3, :3] = R
    refined_pose[:3, 3] = tvec[:, 0]
    if not np.isfinite(refined_pose).all():
        return None, 0
    return refined_pose.astype(np.float32), int(refine_ids.shape[0])


def _solve_pnp_poselib(
    object_points: np.ndarray,
    image_points_xy: np.ndarray,
    K: np.ndarray,
    reprojection_error: float,
    confidence: float,
    iterations: int,
    min_iterations: int = 0,
) -> tuple[Optional[np.ndarray], int]:
    try:
        import poselib  # type: ignore
    except Exception:
        return None, 0
    try:
        width = max(float(K[0, 2]) * 2.0, 1.0)
        height = max(float(K[1, 2]) * 2.0, 1.0)
        camera = {
            "model": "PINHOLE",
            "width": int(round(width)),
            "height": int(round(height)),
            "params": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
        }
        ransac_opt = {
            "max_reproj_error": float(reprojection_error),
            "success_prob": float(confidence),
            "max_iterations": int(iterations),
        }
        if int(min_iterations) > 0:
            ransac_opt["min_iterations"] = int(min_iterations)
        pose, info = poselib.estimate_absolute_pose(
            image_points_xy,
            object_points,
            camera,
            ransac_opt,
            {},
        )
        R = np.asarray(getattr(pose, "R"), dtype=np.float64)
        t = np.asarray(getattr(pose, "t"), dtype=np.float64).reshape(3)
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = R.astype(np.float32)
        out[:3, 3] = t.astype(np.float32)
        inliers = info.get("inliers", []) if isinstance(info, dict) else []
        return out, int(np.asarray(inliers).sum() if np.asarray(inliers).dtype == bool else len(inliers))
    except Exception:
        return None, 0


def pose_error_cm_deg(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> tuple[float, float]:
    pred_c2w = np.linalg.inv(pred_w2c)
    gt_c2w = np.linalg.inv(gt_w2c)
    te_cm = float(np.linalg.norm(pred_c2w[:3, 3] - gt_c2w[:3, 3]) * 100.0)
    rel = pred_w2c[:3, :3] @ gt_c2w[:3, :3]
    cos_angle = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    ae_deg = float(math.degrees(math.acos(cos_angle)))
    if abs(ae_deg) < 1e-10:
        ae_deg = 0.0
    if abs(te_cm) < 1e-10:
        te_cm = 0.0
    return te_cm, ae_deg
