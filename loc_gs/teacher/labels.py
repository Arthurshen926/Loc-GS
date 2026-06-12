from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.correspondences import SparseCandidateBatch


@dataclass(frozen=True)
class TeacherLabelBatch:
    scene: str
    split_name: str
    query_id: str
    label_roles: list[list[str]]
    geometric_correct: list[list[bool]]
    dense_consistent: list[list[bool]]
    sparse_inlier: list[list[bool]]
    reprojection_error_px: list[list[float]]

    @property
    def positive_count(self) -> int:
        return sum(role in {"protected_support", "positive_inlier"} for row in self.label_roles for role in row)

    @property
    def hard_negative_count(self) -> int:
        return sum(role == "hard_negative" for row in self.label_roles for role in row)


def _bool_grid(
    values: Sequence[Sequence[bool]] | None,
    lengths: Sequence[int],
    *,
    default: bool,
    name: str,
) -> list[list[bool]]:
    if values is None:
        return [[default for _ in range(length)] for length in lengths]
    if len(values) != len(lengths):
        raise ValueError(f"{name} must have the same keypoint count as candidate rows")
    out: list[list[bool]] = []
    for row_idx, (row, length) in enumerate(zip(values, lengths)):
        if len(row) != length:
            raise ValueError(f"{name} row {row_idx} must have the same top-k length as candidate rows")
        out.append([bool(item) for item in row])
    return out


def _float_grid(
    values: Sequence[Sequence[float]] | None,
    lengths: Sequence[int],
    *,
    default: float,
    name: str,
) -> list[list[float]]:
    if values is None:
        return [[default for _ in range(length)] for length in lengths]
    if len(values) != len(lengths):
        raise ValueError(f"{name} must have the same keypoint count as candidate rows")
    out: list[list[float]] = []
    for row_idx, (row, length) in enumerate(zip(values, lengths)):
        if len(row) != length:
            raise ValueError(f"{name} row {row_idx} must have the same top-k length as candidate rows")
        out.append([float(item) for item in row])
    return out


def _role(*, geometric: bool, dense: bool, inlier: bool, reprojection_px: float) -> str:
    if geometric and dense and inlier:
        return "protected_support"
    if geometric and inlier:
        return "positive_inlier"
    if (not geometric) and reprojection_px >= 8.0:
        return "hard_negative"
    return "neutral"


def build_teacher_labels(
    batch: SparseCandidateBatch,
    *,
    geometric_correct: Sequence[Sequence[bool]],
    dense_consistent: Sequence[Sequence[bool]] | None = None,
    sparse_inlier: Sequence[Sequence[bool]] | None = None,
    reprojection_error_px: Sequence[Sequence[float]] | None = None,
) -> TeacherLabelBatch:
    split = reject_test_split(batch.split_name, purpose="teacher labels")
    batch.validate()
    lengths = batch.topk_lengths()
    geometric = _bool_grid(geometric_correct, lengths, default=False, name="geometric_correct")
    dense = _bool_grid(dense_consistent, lengths, default=False, name="dense_consistent")
    inliers = _bool_grid(sparse_inlier, lengths, default=False, name="sparse_inlier")
    reproj = _float_grid(reprojection_error_px, lengths, default=float("inf"), name="reprojection_error_px")
    roles = [
        [
            _role(
                geometric=geometric[row_idx][col_idx],
                dense=dense[row_idx][col_idx],
                inlier=inliers[row_idx][col_idx],
                reprojection_px=reproj[row_idx][col_idx],
            )
            for col_idx in range(length)
        ]
        for row_idx, length in enumerate(lengths)
    ]
    return TeacherLabelBatch(
        scene=batch.scene,
        split_name=split,
        query_id=batch.query_id,
        label_roles=roles,
        geometric_correct=geometric,
        dense_consistent=dense,
        sparse_inlier=inliers,
        reprojection_error_px=reproj,
    )
