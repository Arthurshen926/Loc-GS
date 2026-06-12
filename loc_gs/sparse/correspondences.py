from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SparseCandidateBatch:
    scene: str
    split_name: str
    query_id: str
    keypoint_xy: Sequence[Sequence[float]]
    candidate_landmark_ids: Sequence[Sequence[int]]
    candidate_scores: Sequence[Sequence[float]]

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
        for row_idx, xy in enumerate(self.keypoint_xy):
            if len(xy) < 2:
                raise ValueError(f"keypoint_xy row {row_idx} must contain x and y")
        for row_idx, (ids, scores) in enumerate(zip(self.candidate_landmark_ids, self.candidate_scores)):
            if len(ids) != len(scores):
                raise ValueError(f"candidate ids and scores must have the same top-k length at row {row_idx}")

    def topk_lengths(self) -> list[int]:
        self.validate()
        return [len(row) for row in self.candidate_landmark_ids]
