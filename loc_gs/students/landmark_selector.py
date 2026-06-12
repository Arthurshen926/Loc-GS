from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from loc_gs.sparse.correspondences import SparseCandidateBatch


@dataclass(frozen=True)
class LandmarkSelectorConfig:
    protected_support_gain: float = 2.0
    positive_inlier_gain: float = 1.0
    dense_consistency_gain: float = 0.5
    hard_negative_penalty: float = 1.0
    conflict_penalty: float = 0.1
    score_scale: float = 1.0


@dataclass(frozen=True)
class LandmarkSelectorModel:
    landmark_scores: dict[str, float]
    conflict_edges: dict[str, float]
    conflict_degrees: dict[str, float]
    score_scale: float = 1.0
    conflict_penalty: float = 0.1

    def score_landmark(self, landmark_id: int | str) -> float:
        key = str(landmark_id)
        raw = float(self.landmark_scores.get(key, 0.0))
        conflict = float(self.conflict_degrees.get(key, 0.0))
        return float((raw - float(self.conflict_penalty) * conflict) * float(self.score_scale))

    def conflict_weight(self, first: int | str, second: int | str) -> float:
        return float(self.conflict_edges.get(_edge_key(first, second), 0.0))

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": "internal_landmark_selector_v1",
            "landmark_scores": {str(key): float(value) for key, value in sorted(self.landmark_scores.items())},
            "conflict_edges": {str(key): float(value) for key, value in sorted(self.conflict_edges.items())},
            "conflict_degrees": {str(key): float(value) for key, value in sorted(self.conflict_degrees.items())},
            "score_scale": float(self.score_scale),
            "conflict_penalty": float(self.conflict_penalty),
        }

    def to_conflict_graph_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": "internal_conflict_graph_v1",
            "conflict_edges": {str(key): float(value) for key, value in sorted(self.conflict_edges.items())},
            "conflict_degrees": {str(key): float(value) for key, value in sorted(self.conflict_degrees.items())},
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> "LandmarkSelectorModel":
        if payload.get("schema_version") != "internal_landmark_selector_v1":
            raise ValueError(f"unsupported landmark selector schema: {payload.get('schema_version')}")
        return cls(
            landmark_scores={str(key): float(value) for key, value in dict(payload.get("landmark_scores", {})).items()},
            conflict_edges={str(key): float(value) for key, value in dict(payload.get("conflict_edges", {})).items()},
            conflict_degrees={str(key): float(value) for key, value in dict(payload.get("conflict_degrees", {})).items()},
            score_scale=float(payload.get("score_scale", 1.0)),
            conflict_penalty=float(payload.get("conflict_penalty", 0.1)),
        )


def train_landmark_selector(
    batches: Sequence[SparseCandidateBatch],
    cfg: LandmarkSelectorConfig | None = None,
) -> tuple[LandmarkSelectorModel, dict[str, object]]:
    if cfg is None:
        cfg = LandmarkSelectorConfig()
    scores: dict[str, float] = {}
    observations: dict[str, int] = {}
    conflict_edges: dict[str, float] = {}
    protected_support_count = 0
    positive_inlier_count = 0
    hard_negative_count = 0
    for batch in batches:
        batch.validate()
        valid_rows = batch.candidate_valid_mask or []
        for row_idx, landmark_ids in enumerate(batch.candidate_landmark_ids):
            positive_ids: list[str] = []
            hard_ids: list[str] = []
            positive_weight = 1.0
            hard_weight = 1.0
            for rank, landmark_id in enumerate(landmark_ids):
                valid = True if not valid_rows else bool(valid_rows[row_idx][rank])
                if not valid:
                    continue
                key = str(int(landmark_id))
                observations[key] = observations.get(key, 0) + 1
                role = str(_grid_value(batch.candidate_label_roles, row_idx, rank, ""))
                solver_weight = max(1.0e-6, float(_grid_value(batch.candidate_solver_weight, row_idx, rank, 1.0)))
                dense = bool(_grid_value(batch.candidate_dense_consistent, row_idx, rank, False))
                sparse_inlier = bool(_grid_value(batch.candidate_sparse_inlier, row_idx, rank, False))
                geometric = bool(_grid_value(batch.candidate_geometric_correct, row_idx, rank, False))
                delta = 0.0
                if role == "protected_support" or (geometric and dense and sparse_inlier):
                    delta += float(cfg.protected_support_gain) * solver_weight
                    protected_support_count += 1
                    positive_ids.append(key)
                    positive_weight = max(positive_weight, solver_weight)
                elif role == "positive_inlier" or (geometric and sparse_inlier):
                    delta += float(cfg.positive_inlier_gain) * solver_weight
                    positive_inlier_count += 1
                    positive_ids.append(key)
                    positive_weight = max(positive_weight, solver_weight)
                elif role == "hard_negative":
                    delta -= float(cfg.hard_negative_penalty) * solver_weight
                    hard_negative_count += 1
                    hard_ids.append(key)
                    hard_weight = max(hard_weight, solver_weight)
                elif geometric and dense:
                    delta += float(cfg.dense_consistency_gain) * solver_weight
                scores[key] = scores.get(key, 0.0) + float(delta)
            for positive_id in positive_ids:
                for hard_id in hard_ids:
                    edge = _edge_key(positive_id, hard_id)
                    conflict_edges[edge] = conflict_edges.get(edge, 0.0) + max(positive_weight, hard_weight)
    conflict_degrees: dict[str, float] = {}
    for edge, weight in conflict_edges.items():
        first, second = edge.split("::", 1)
        conflict_degrees[first] = conflict_degrees.get(first, 0.0) + float(weight)
        conflict_degrees[second] = conflict_degrees.get(second, 0.0) + float(weight)
    model = LandmarkSelectorModel(
        landmark_scores=scores,
        conflict_edges=conflict_edges,
        conflict_degrees=conflict_degrees,
        score_scale=float(cfg.score_scale),
        conflict_penalty=float(cfg.conflict_penalty),
    )
    summary = {
        "schema_version": "internal_landmark_selector_training_summary_v1",
        "student_modules": ["landmark_selector", "conflict_graph"],
        "landmark_count": int(len(scores)),
        "conflict_edge_count": int(len(conflict_edges)),
        "protected_support_count": int(protected_support_count),
        "positive_inlier_count": int(positive_inlier_count),
        "hard_negative_count": int(hard_negative_count),
        "observed_candidate_count": int(sum(observations.values())),
        "hyperparameters": asdict(cfg),
    }
    return model, summary


def landmark_selector_score_rows(batch: SparseCandidateBatch, model: LandmarkSelectorModel) -> list[list[float]]:
    batch.validate()
    rows: list[list[float]] = []
    valid_rows = batch.candidate_valid_mask or []
    for row_idx, landmark_ids in enumerate(batch.candidate_landmark_ids):
        row: list[float] = []
        for rank, landmark_id in enumerate(landmark_ids):
            valid = True if not valid_rows else bool(valid_rows[row_idx][rank])
            row.append(model.score_landmark(int(landmark_id)) if valid else -1.0e12)
        rows.append(row)
    return rows


def load_landmark_selector(path: str | Path) -> LandmarkSelectorModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"landmark selector JSON must contain an object: {path}")
    return LandmarkSelectorModel.from_json_dict(payload)


def _grid_value(rows: Sequence[Sequence[object]] | None, row_idx: int, rank: int, default: object) -> object:
    if rows is None or row_idx >= len(rows) or rank >= len(rows[row_idx]):
        return default
    return rows[row_idx][rank]


def _edge_key(first: int | str, second: int | str) -> str:
    left = str(first)
    right = str(second)
    return "::".join(sorted((left, right), key=lambda value: int(value) if value.lstrip("-").isdigit() else value))
