from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loc_gs.core.camera import CameraIntrinsics


@dataclass(frozen=True)
class OpenCvPnPConfig:
    reprojection_error_px: float = 8.0
    confidence: float = 0.999
    iterations: int = 10000
    method: str = "epnp"
    refine_with_inliers: bool = False
    refinement_method: str = "iterative"


@dataclass(frozen=True)
class PnPResult:
    success: bool
    pose_w2c: np.ndarray | None
    inlier_mask: np.ndarray
    method: str = "epnp"
    refined: bool = False
    refinement_method: str | None = None

    @property
    def inlier_count(self) -> int:
        return int(np.asarray(self.inlier_mask, dtype=bool).sum())


def _cv2_method(cv2_module, method: str) -> int:
    lowered = str(method).strip().lower()
    if lowered in {"iterative", "opencv_iterative"}:
        return int(cv2_module.SOLVEPNP_ITERATIVE)
    return int(cv2_module.SOLVEPNP_EPNP)


def solve_pnp_ransac(
    points3d_world: np.ndarray,
    keypoints_xy: np.ndarray,
    intrinsics: CameraIntrinsics,
    config: OpenCvPnPConfig | None = None,
    *,
    match_scores: np.ndarray | None = None,
) -> PnPResult:
    if config is None:
        config = OpenCvPnPConfig()
    points = np.asarray(points3d_world, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(keypoints_xy, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] != pixels.shape[0]:
        raise ValueError("points3d_world and keypoints_xy must contain the same number of correspondences")
    if points.shape[0] < 4:
        return PnPResult(
            success=False,
            pose_w2c=None,
            inlier_mask=np.zeros(points.shape[0], dtype=bool),
            method=str(config.method),
        )

    order = np.arange(points.shape[0])
    if match_scores is not None:
        scores = np.asarray(match_scores, dtype=np.float64).reshape(-1)
        if scores.shape[0] != points.shape[0]:
            raise ValueError("match_scores must match correspondence count")
        order = np.argsort(np.where(np.isfinite(scores), scores, -np.inf))[::-1]
        points = points[order]
        pixels = pixels[order]

    import cv2

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points.astype(np.float64),
        pixels.astype(np.float64),
        intrinsics.matrix,
        None,
        iterationsCount=int(config.iterations),
        reprojectionError=float(config.reprojection_error_px),
        confidence=float(config.confidence),
        flags=_cv2_method(cv2, config.method),
    )
    if not ok or rvec is None or tvec is None:
        return PnPResult(
            success=False,
            pose_w2c=None,
            inlier_mask=np.zeros(order.shape[0], dtype=bool),
            method=str(config.method),
        )

    sorted_mask = np.zeros(order.shape[0], dtype=bool)
    if inliers is not None:
        sorted_mask[np.asarray(inliers, dtype=np.int64).reshape(-1)] = True
    refined = False
    refinement_method = None
    if bool(config.refine_with_inliers) and int(sorted_mask.sum()) >= 4:
        refine_points = points[sorted_mask].astype(np.float64)
        refine_pixels = pixels[sorted_mask].astype(np.float64)
        refine_ok, refine_rvec, refine_tvec = cv2.solvePnP(
            refine_points,
            refine_pixels,
            intrinsics.matrix,
            None,
            rvec=np.asarray(rvec, dtype=np.float64),
            tvec=np.asarray(tvec, dtype=np.float64),
            useExtrinsicGuess=True,
            flags=_cv2_method(cv2, config.refinement_method),
        )
        if refine_ok and refine_rvec is not None and refine_tvec is not None:
            rvec = refine_rvec
            tvec = refine_tvec
            refined = True
            refinement_method = str(config.refinement_method)

    rotation, _ = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    original_mask = np.zeros_like(sorted_mask)
    original_mask[order] = sorted_mask
    return PnPResult(
        success=True,
        pose_w2c=pose,
        inlier_mask=original_mask,
        method=str(config.method),
        refined=refined,
        refinement_method=refinement_method,
    )
