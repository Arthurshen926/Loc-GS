from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SparseCandidateBatch:
    scene: str
    split_name: str
    query_id: str
    keypoint_xy: Sequence[Sequence[float]]
    candidate_landmark_ids: Sequence[Sequence[int]]
    candidate_scores: Sequence[Sequence[float]]
    teacher_labels: Sequence[int | None] | None = None
    candidate_valid_mask: Sequence[Sequence[bool]] | None = None
    candidate_geometric_correct: Sequence[Sequence[bool]] | None = None
    candidate_dense_consistent: Sequence[Sequence[bool]] | None = None
    candidate_sparse_inlier: Sequence[Sequence[bool]] | None = None
    candidate_reprojection_error_px: Sequence[Sequence[float]] | None = None
    candidate_solver_weight: Sequence[Sequence[float]] | None = None
    candidate_label_roles: Sequence[Sequence[str]] | None = None
    source_keypoint_ids: Sequence[str] | None = None
    source_phases: Sequence[str] | None = None
    metadata: Mapping[str, object] | None = None

    @property
    def keypoint_count(self) -> int:
        return len(self.keypoint_xy)

    def validate(self) -> None:
        if not self.scene:
            raise ValueError("scene is required")
        if not self.query_id:
            raise ValueError("query_id is required")
        if len(self.keypoint_xy) != len(self.candidate_landmark_ids):
            raise ValueError("keypoint_xy and candidate_landmark_ids must have the same keypoint count")
        if len(self.candidate_scores) != len(self.candidate_landmark_ids):
            raise ValueError("candidate_scores and candidate_landmark_ids must have the same keypoint count")
        if self.teacher_labels is not None and len(self.teacher_labels) != len(self.keypoint_xy):
            raise ValueError("teacher_labels must have the same keypoint count")
        if self.candidate_valid_mask is not None and len(self.candidate_valid_mask) != len(self.keypoint_xy):
            raise ValueError("candidate_valid_mask must have the same keypoint count")
        if self.candidate_geometric_correct is not None and len(self.candidate_geometric_correct) != len(self.keypoint_xy):
            raise ValueError("candidate_geometric_correct must have the same keypoint count")
        if self.candidate_dense_consistent is not None and len(self.candidate_dense_consistent) != len(self.keypoint_xy):
            raise ValueError("candidate_dense_consistent must have the same keypoint count")
        if self.candidate_sparse_inlier is not None and len(self.candidate_sparse_inlier) != len(self.keypoint_xy):
            raise ValueError("candidate_sparse_inlier must have the same keypoint count")
        if self.candidate_reprojection_error_px is not None and len(self.candidate_reprojection_error_px) != len(self.keypoint_xy):
            raise ValueError("candidate_reprojection_error_px must have the same keypoint count")
        if self.candidate_solver_weight is not None and len(self.candidate_solver_weight) != len(self.keypoint_xy):
            raise ValueError("candidate_solver_weight must have the same keypoint count")
        if self.candidate_label_roles is not None and len(self.candidate_label_roles) != len(self.keypoint_xy):
            raise ValueError("candidate_label_roles must have the same keypoint count")
        if self.source_keypoint_ids is not None and len(self.source_keypoint_ids) != len(self.keypoint_xy):
            raise ValueError("source_keypoint_ids must have the same keypoint count")
        if self.source_phases is not None and len(self.source_phases) != len(self.keypoint_xy):
            raise ValueError("source_phases must have the same keypoint count")
        for row_idx, xy in enumerate(self.keypoint_xy):
            if len(xy) < 2:
                raise ValueError(f"keypoint_xy row {row_idx} must contain x and y")
        for row_idx, (ids, scores) in enumerate(zip(self.candidate_landmark_ids, self.candidate_scores)):
            if len(ids) != len(scores):
                raise ValueError(f"candidate ids and scores must have the same top-k length at row {row_idx}")
        if self.candidate_valid_mask is not None:
            for row_idx, (ids, mask) in enumerate(zip(self.candidate_landmark_ids, self.candidate_valid_mask)):
                if len(ids) != len(mask):
                    raise ValueError(f"candidate ids and mask must have the same top-k length at row {row_idx}")
        if self.candidate_geometric_correct is not None:
            for row_idx, (ids, correct) in enumerate(zip(self.candidate_landmark_ids, self.candidate_geometric_correct)):
                if len(ids) != len(correct):
                    raise ValueError(
                        f"candidate ids and geometric labels must have the same top-k length at row {row_idx}"
                    )
        optional_grids = (
            ("dense consistency labels", self.candidate_dense_consistent),
            ("sparse inlier labels", self.candidate_sparse_inlier),
            ("reprojection errors", self.candidate_reprojection_error_px),
            ("solver weights", self.candidate_solver_weight),
            ("label roles", self.candidate_label_roles),
        )
        for name, grid in optional_grids:
            if grid is None:
                continue
            for row_idx, (ids, row) in enumerate(zip(self.candidate_landmark_ids, grid)):
                if len(ids) != len(row):
                    raise ValueError(f"candidate ids and {name} must have the same top-k length at row {row_idx}")

    def topk_lengths(self) -> list[int]:
        self.validate()
        return [len(row) for row in self.candidate_landmark_ids]
