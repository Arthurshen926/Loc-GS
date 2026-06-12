from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from loc_gs.core.camera import CameraIntrinsics
from loc_gs.sparse.correspondences import SparseCandidateBatch
from loc_gs.sparse.landmarks import CacheLandmarkResolver
from loc_gs.sparse.pipeline import SparseLocalizationInput, SparsePipelineCandidate


ScoreMode = Literal["native", "teacher_oracle"]


@dataclass(frozen=True)
class CachedSparseInputConfig:
    score_mode: ScoreMode = "native"
    max_keypoints: int | None = None
    solver_score_rows: Sequence[Sequence[float]] | None = None


def _solver_score(*, correct: bool | None, score_mode: ScoreMode) -> float:
    if score_mode == "native":
        return 0.0
    if score_mode == "teacher_oracle":
        return 1.0 if bool(correct) else 0.0
    raise ValueError(f"unsupported score_mode: {score_mode}")


def sparse_input_from_cached_batch(
    batch: SparseCandidateBatch,
    resolver: CacheLandmarkResolver,
    *,
    intrinsics: CameraIntrinsics,
    cfg: CachedSparseInputConfig | None = None,
) -> SparseLocalizationInput:
    if cfg is None:
        cfg = CachedSparseInputConfig()
    batch.validate()
    max_keypoints = batch.keypoint_count if cfg.max_keypoints is None else min(batch.keypoint_count, int(cfg.max_keypoints))
    solver_score_rows = cfg.solver_score_rows
    if solver_score_rows is not None and len(solver_score_rows) < max_keypoints:
        raise ValueError("solver_score_rows must contain at least max_keypoints rows")
    gaussian_ids = resolver.resolve_gaussian_ids(batch.candidate_landmark_ids[:max_keypoints])
    xyz = resolver.landmark_map.lookup_xyz(gaussian_ids)

    keypoints_xy: list[list[float]] = []
    candidates_by_keypoint: list[list[SparsePipelineCandidate]] = []
    valid_masks = batch.candidate_valid_mask
    correct_rows = batch.candidate_geometric_correct
    for source_row_idx in range(max_keypoints):
        candidates: list[SparsePipelineCandidate] = []
        for candidate_idx, native_score in enumerate(batch.candidate_scores[source_row_idx]):
            valid = True if valid_masks is None else bool(valid_masks[source_row_idx][candidate_idx])
            point = np.asarray(xyz[source_row_idx, candidate_idx], dtype=np.float64)
            if not valid or not np.all(np.isfinite(point)):
                continue
            correct = None if correct_rows is None else bool(correct_rows[source_row_idx][candidate_idx])
            solver_score = (
                float(solver_score_rows[source_row_idx][candidate_idx])
                if solver_score_rows is not None
                else _solver_score(correct=correct, score_mode=cfg.score_mode)
            )
            candidates.append(
                SparsePipelineCandidate(
                    keypoint_index=len(keypoints_xy),
                    landmark_id=int(gaussian_ids[source_row_idx, candidate_idx]),
                    point3d=point.tolist(),
                    native_score=float(native_score),
                    solver_score=solver_score,
                    geometric_correct=correct,
                )
            )
        if not candidates:
            continue
        keypoints_xy.append([float(v) for v in batch.keypoint_xy[source_row_idx][:2]])
        candidates_by_keypoint.append(candidates)

    return SparseLocalizationInput(
        scene=batch.scene,
        split_name=batch.split_name,
        query_id=batch.query_id,
        intrinsics=intrinsics,
        keypoints_xy=keypoints_xy,
        candidates_by_keypoint=candidates_by_keypoint,
    )
