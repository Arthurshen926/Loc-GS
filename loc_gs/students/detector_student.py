from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from loc_gs.sparse.correspondences import SparseCandidateBatch


@dataclass(frozen=True)
class DetectorStudentConfig:
    grid_size: int = 8
    protected_support_gain: float = 2.0
    positive_inlier_gain: float = 1.0
    hard_negative_penalty: float = 1.0
    score_scale: float = 1.0


@dataclass(frozen=True)
class DetectorStudentModel:
    grid_size: int
    x_extent: float
    y_extent: float
    cell_scores: dict[str, float]
    score_scale: float = 1.0

    def score_keypoint(self, xy: Sequence[float]) -> float:
        key = self.cell_key(xy)
        return float(self.cell_scores.get(key, 0.0) * float(self.score_scale))

    def cell_key(self, xy: Sequence[float]) -> str:
        if len(xy) < 2:
            raise ValueError("keypoint xy must contain x and y")
        grid = max(1, int(self.grid_size))
        x_extent = max(1.0e-9, float(self.x_extent))
        y_extent = max(1.0e-9, float(self.y_extent))
        col = min(grid - 1, max(0, int(float(xy[0]) / x_extent * grid)))
        row = min(grid - 1, max(0, int(float(xy[1]) / y_extent * grid)))
        return f"{row}:{col}"

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": "internal_detector_student_v1",
            "grid_size": int(self.grid_size),
            "x_extent": float(self.x_extent),
            "y_extent": float(self.y_extent),
            "cell_scores": {str(key): float(value) for key, value in sorted(self.cell_scores.items())},
            "score_scale": float(self.score_scale),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> "DetectorStudentModel":
        if payload.get("schema_version") != "internal_detector_student_v1":
            raise ValueError(f"unsupported detector student schema: {payload.get('schema_version')}")
        return cls(
            grid_size=int(payload.get("grid_size", 8)),
            x_extent=float(payload.get("x_extent", 1.0)),
            y_extent=float(payload.get("y_extent", 1.0)),
            cell_scores={str(key): float(value) for key, value in dict(payload.get("cell_scores", {})).items()},
            score_scale=float(payload.get("score_scale", 1.0)),
        )


def train_detector_student(
    batches: Sequence[SparseCandidateBatch],
    cfg: DetectorStudentConfig | None = None,
) -> tuple[DetectorStudentModel, dict[str, object]]:
    if cfg is None:
        cfg = DetectorStudentConfig()
    grid = max(1, int(cfg.grid_size))
    x_extent = max((float(xy[0]) for batch in batches for xy in batch.keypoint_xy), default=1.0)
    y_extent = max((float(xy[1]) for batch in batches for xy in batch.keypoint_xy), default=1.0)
    x_extent = max(1.0, x_extent)
    y_extent = max(1.0, y_extent)
    model = DetectorStudentModel(grid_size=grid, x_extent=x_extent, y_extent=y_extent, cell_scores={}, score_scale=float(cfg.score_scale))
    cell_scores: dict[str, float] = {}
    positive_keypoint_count = 0
    hard_negative_keypoint_count = 0
    for batch in batches:
        batch.validate()
        for row_idx, xy in enumerate(batch.keypoint_xy):
            roles = batch.candidate_label_roles[row_idx] if batch.candidate_label_roles is not None else []
            weights = batch.candidate_solver_weight[row_idx] if batch.candidate_solver_weight is not None else []
            row_score = 0.0
            has_positive = False
            has_hard_negative = False
            for rank, role in enumerate(roles):
                weight = max(1.0e-6, float(weights[rank]) if rank < len(weights) else 1.0)
                if str(role) == "protected_support":
                    row_score += float(cfg.protected_support_gain) * weight
                    has_positive = True
                elif str(role) == "positive_inlier":
                    row_score += float(cfg.positive_inlier_gain) * weight
                    has_positive = True
                elif str(role) == "hard_negative":
                    row_score -= float(cfg.hard_negative_penalty) * weight
                    has_hard_negative = True
            if has_positive:
                positive_keypoint_count += 1
            elif has_hard_negative:
                hard_negative_keypoint_count += 1
            key = model.cell_key(xy)
            cell_scores[key] = cell_scores.get(key, 0.0) + float(row_score)
    model = DetectorStudentModel(
        grid_size=grid,
        x_extent=x_extent,
        y_extent=y_extent,
        cell_scores=cell_scores,
        score_scale=float(cfg.score_scale),
    )
    summary = {
        "schema_version": "internal_detector_student_training_summary_v1",
        "student_modules": ["detector_student"],
        "grid_size": int(grid),
        "cell_count": int(len(cell_scores)),
        "positive_keypoint_count": int(positive_keypoint_count),
        "hard_negative_keypoint_count": int(hard_negative_keypoint_count),
        "hyperparameters": asdict(cfg),
    }
    return model, summary


def detector_score_rows(batch: SparseCandidateBatch, model: DetectorStudentModel) -> list[list[float]]:
    batch.validate()
    rows: list[list[float]] = []
    for row_idx, landmark_ids in enumerate(batch.candidate_landmark_ids):
        score = model.score_keypoint(batch.keypoint_xy[row_idx])
        rows.append([float(score) for _ in landmark_ids])
    return rows


def load_detector_student(path: str | Path) -> DetectorStudentModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"detector student JSON must contain an object: {path}")
    return DetectorStudentModel.from_json_dict(payload)
