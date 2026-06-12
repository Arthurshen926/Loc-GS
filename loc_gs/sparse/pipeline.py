from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from loc_gs.core.camera import CameraIntrinsics
from loc_gs.core.pnp import OpenCvPnPConfig, solve_pnp_ransac
from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.rerank import (
    CandidateRerankConfig,
    rerank_candidate_rows,
    summarize_candidate_availability,
)
from loc_gs.sparse.sparse_lgcv import filter_correspondences_by_reprojection


@dataclass(frozen=True)
class SparsePipelineCandidate:
    keypoint_index: int
    landmark_id: int
    point3d: Sequence[float]
    native_score: float
    solver_score: float = 0.0
    geometric_correct: bool | None = None

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "keypoint_index": int(self.keypoint_index),
            "landmark_id": int(self.landmark_id),
            "point3d": [float(v) for v in self.point3d],
            "native_score": float(self.native_score),
            "solver_score": float(self.solver_score),
        }
        if self.geometric_correct is not None:
            row["geometric_correct"] = bool(self.geometric_correct)
        return row


@dataclass(frozen=True)
class SparseLocalizationInput:
    scene: str
    split_name: str
    query_id: str
    intrinsics: CameraIntrinsics
    keypoints_xy: Sequence[Sequence[float]]
    candidates_by_keypoint: Sequence[Sequence[SparsePipelineCandidate]]

    def validate(self) -> None:
        if len(self.keypoints_xy) != len(self.candidates_by_keypoint):
            raise ValueError("keypoints_xy and candidates_by_keypoint must have the same length")
        for row_idx, row in enumerate(self.candidates_by_keypoint):
            if not row:
                raise ValueError(f"candidate row {row_idx} is empty")
            for candidate in row:
                if int(candidate.keypoint_index) != row_idx:
                    raise ValueError(f"candidate row {row_idx} contains mismatched keypoint_index")


@dataclass(frozen=True)
class SparseLocalizationConfig:
    rerank_prefix_fraction: float = 0.75
    solver_weight: float = 1.0
    native_weight: float = 1.0
    reprojection_error_px: float = 8.0
    pnp_iterations: int = 10000
    min_correspondences: int = 4
    pnp_method: str = "epnp"
    refine_with_inliers: bool = False
    second_pnp_enabled: bool = False
    second_pnp_method: str = "iterative"
    lgcv_reprojection_error_px: float = 4.0


@dataclass(frozen=True)
class SparseLocalizationResult:
    success: bool
    pose_w2c: np.ndarray | None
    inlier_mask: np.ndarray
    selected_landmark_ids: list[int]
    selected_keypoint_indices: list[int]
    availability_summary: dict[str, int]
    pnp_stage_count: int = 0
    initial_inlier_count: int = 0
    lgcv_keep_count: int | None = None
    pnp_method: str = "epnp"
    second_pnp_method: str | None = None

    @property
    def inlier_count(self) -> int:
        return int(np.asarray(self.inlier_mask, dtype=bool).sum())


def _rank_candidates(
    candidates: Sequence[SparsePipelineCandidate],
    cfg: SparseLocalizationConfig,
) -> list[dict[str, object]]:
    rows = [candidate.to_row() for candidate in candidates]
    return rerank_candidate_rows(
        rows,
        CandidateRerankConfig(
            prefix_fraction=float(cfg.rerank_prefix_fraction),
            solver_weight=float(cfg.solver_weight),
            native_weight=float(cfg.native_weight),
        ),
    )


def run_sparse_localization(
    data: SparseLocalizationInput,
    cfg: SparseLocalizationConfig | None = None,
) -> SparseLocalizationResult:
    if cfg is None:
        cfg = SparseLocalizationConfig()
    reject_test_split(data.split_name, purpose="internal sparse localization")
    data.validate()

    rows_by_query = [[candidate.to_row() for candidate in row] for row in data.candidates_by_keypoint]
    availability = summarize_candidate_availability(rows_by_query)

    selected_rows: list[dict[str, object]] = []
    selected_keypoint_indices: list[int] = []
    selected_landmark_ids: list[int] = []
    for row_idx, candidates in enumerate(data.candidates_by_keypoint):
        ranked = _rank_candidates(candidates, cfg)
        selected = ranked[0]
        selected_rows.append(selected)
        selected_keypoint_indices.append(row_idx)
        selected_landmark_ids.append(int(selected["landmark_id"]))

    points = np.asarray([row["point3d"] for row in selected_rows], dtype=np.float64).reshape(-1, 3)
    keypoints_xy = np.asarray(data.keypoints_xy, dtype=np.float64).reshape(-1, 2)
    scores = np.asarray(
        [
            float(row.get("native_score", 0.0)) * float(cfg.native_weight)
            + float(row.get("solver_score", 0.0)) * float(cfg.solver_weight)
            for row in selected_rows
        ],
        dtype=np.float64,
    )
    if points.shape[0] < int(cfg.min_correspondences):
        return SparseLocalizationResult(
            success=False,
            pose_w2c=None,
            inlier_mask=np.zeros(points.shape[0], dtype=bool),
            selected_landmark_ids=selected_landmark_ids,
            selected_keypoint_indices=selected_keypoint_indices,
            availability_summary=availability,
            pnp_method=str(cfg.pnp_method),
        )

    pnp = solve_pnp_ransac(
        points,
        keypoints_xy,
        data.intrinsics,
        OpenCvPnPConfig(
            reprojection_error_px=float(cfg.reprojection_error_px),
            iterations=int(cfg.pnp_iterations),
            method=str(cfg.pnp_method),
            refine_with_inliers=bool(cfg.refine_with_inliers),
        ),
        match_scores=scores,
    )
    final_pnp = pnp
    final_mask = np.asarray(pnp.inlier_mask, dtype=bool)
    pnp_stage_count = 1 if bool(pnp.success) else 0
    lgcv_keep_count = None
    second_method = None
    if bool(pnp.success) and bool(cfg.second_pnp_enabled) and pnp.pose_w2c is not None:
        lgcv = filter_correspondences_by_reprojection(
            points,
            keypoints_xy,
            pnp.pose_w2c,
            data.intrinsics,
            threshold_px=float(cfg.lgcv_reprojection_error_px),
        )
        lgcv_keep = np.asarray(lgcv.keep_mask, dtype=bool)
        lgcv_keep_count = int(lgcv.keep_count)
        if lgcv_keep_count >= int(cfg.min_correspondences):
            second_pnp = solve_pnp_ransac(
                points[lgcv_keep],
                keypoints_xy[lgcv_keep],
                data.intrinsics,
                OpenCvPnPConfig(
                    reprojection_error_px=float(cfg.reprojection_error_px),
                    iterations=int(cfg.pnp_iterations),
                    method=str(cfg.second_pnp_method),
                    refine_with_inliers=bool(cfg.refine_with_inliers),
                ),
                match_scores=scores[lgcv_keep],
            )
            pnp_stage_count = 2
            second_method = str(cfg.second_pnp_method)
            if bool(second_pnp.success):
                mapped = np.zeros(points.shape[0], dtype=bool)
                mapped_indices = np.flatnonzero(lgcv_keep)
                mapped[mapped_indices[np.asarray(second_pnp.inlier_mask, dtype=bool)]] = True
                final_pnp = second_pnp
                final_mask = mapped
    return SparseLocalizationResult(
        success=bool(final_pnp.success),
        pose_w2c=final_pnp.pose_w2c,
        inlier_mask=final_mask,
        selected_landmark_ids=selected_landmark_ids,
        selected_keypoint_indices=selected_keypoint_indices,
        availability_summary=availability,
        pnp_stage_count=pnp_stage_count,
        initial_inlier_count=int(pnp.inlier_count),
        lgcv_keep_count=lgcv_keep_count,
        pnp_method=str(cfg.pnp_method),
        second_pnp_method=second_method,
    )
