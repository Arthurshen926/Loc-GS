from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from loc_gs.sparse.artifact_adapter import CachedCandidateArtifact
from loc_gs.sparse.correspondences import SparseCandidateBatch


@dataclass(frozen=True)
class CandidateScorerConfig:
    epochs: int = 100
    learning_rate: float = 0.1
    rank_feature_scale: float = 1.0


@dataclass(frozen=True)
class LinearCandidateScorer:
    weights: tuple[float, float, float]
    bias: float
    feature_names: tuple[str, str, str] = ("native_score", "negative_rank", "valid")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": "internal_sparse_candidate_scorer_v1",
            **asdict(self),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> "LinearCandidateScorer":
        weights = tuple(float(value) for value in payload.get("weights", ()))
        if len(weights) != 3:
            raise ValueError("internal sparse candidate scorer weights must contain three values")
        return cls(
            weights=weights,
            bias=float(payload.get("bias", 0.0)),
            feature_names=tuple(str(value) for value in payload.get("feature_names", ("native_score", "negative_rank", "valid"))),
        )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def _training_matrix(artifact: CachedCandidateArtifact, cfg: CandidateScorerConfig) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[float] = []
    for batch in artifact.batches:
        correct_rows = batch.candidate_geometric_correct or []
        valid_rows = batch.candidate_valid_mask or []
        for row_idx, scores in enumerate(batch.candidate_scores):
            if row_idx >= len(correct_rows):
                continue
            for rank, score in enumerate(scores):
                valid = True if not valid_rows else bool(valid_rows[row_idx][rank])
                if not valid:
                    continue
                features.append([float(score), -float(rank) * float(cfg.rank_feature_scale), 1.0])
                labels.append(1.0 if bool(correct_rows[row_idx][rank]) else 0.0)
    if not features:
        raise ValueError("candidate artifact did not contain trainable labels")
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def train_candidate_scorer(
    artifact: CachedCandidateArtifact,
    cfg: CandidateScorerConfig | None = None,
) -> tuple[LinearCandidateScorer, dict[str, object]]:
    if cfg is None:
        cfg = CandidateScorerConfig()
    x, y = _training_matrix(artifact, cfg)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    for _epoch in range(int(cfg.epochs)):
        pred = _sigmoid(x @ weights + bias)
        err = pred - y
        weights -= float(cfg.learning_rate) * (x.T @ err) / max(1, x.shape[0])
        bias -= float(cfg.learning_rate) * float(err.mean())
    model = LinearCandidateScorer(weights=tuple(float(v) for v in weights), bias=float(bias))
    trained_top1 = _count_top1_correct(artifact, model)
    native_top1 = _count_native_top1_correct(artifact)
    summary = {
        "schema_version": "internal_sparse_candidate_scorer_training_summary_v1",
        "label_count": int(y.sum()),
        "sample_count": int(y.shape[0]),
        "native_top1_correct": int(native_top1),
        "trained_top1_correct": int(trained_top1),
        "epochs": int(cfg.epochs),
        "learning_rate": float(cfg.learning_rate),
    }
    return model, summary


def score_candidate_rows(batch: SparseCandidateBatch, model: LinearCandidateScorer) -> list[list[dict[str, object]]]:
    rows: list[list[dict[str, object]]] = []
    correct_rows = batch.candidate_geometric_correct or []
    valid_rows = batch.candidate_valid_mask or []
    weights = np.asarray(model.weights, dtype=np.float64)
    for row_idx, scores in enumerate(batch.candidate_scores):
        scored: list[dict[str, object]] = []
        for rank, score in enumerate(scores):
            valid = True if not valid_rows else bool(valid_rows[row_idx][rank])
            if not valid:
                continue
            feature = np.asarray([float(score), -float(rank), 1.0], dtype=np.float64)
            solver_score = float(feature @ weights + float(model.bias))
            correct = row_idx < len(correct_rows) and bool(correct_rows[row_idx][rank])
            scored.append(
                {
                    "candidate_rank": int(rank),
                    "native_score": float(score),
                    "solver_score": solver_score,
                    "geometric_correct": correct,
                }
            )
        rows.append(sorted(scored, key=lambda row: -float(row["solver_score"])))
    return rows


def candidate_solver_score_rows(batch: SparseCandidateBatch, model: LinearCandidateScorer) -> list[list[float]]:
    weights = np.asarray(model.weights, dtype=np.float64)
    rows: list[list[float]] = []
    for scores in batch.candidate_scores:
        row: list[float] = []
        for rank, score in enumerate(scores):
            feature = np.asarray([float(score), -float(rank), 1.0], dtype=np.float64)
            row.append(float(feature @ weights + float(model.bias)))
        rows.append(row)
    return rows


def load_candidate_scorer(path: str | Path) -> LinearCandidateScorer:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"candidate scorer JSON must contain an object: {path}")
    if payload.get("schema_version") != "internal_sparse_candidate_scorer_v1":
        raise ValueError(f"unsupported candidate scorer schema: {payload.get('schema_version')}")
    return LinearCandidateScorer.from_json_dict(payload)


def _count_top1_correct(artifact: CachedCandidateArtifact, model: LinearCandidateScorer) -> int:
    total = 0
    for batch in artifact.batches:
        for row in score_candidate_rows(batch, model):
            if row and bool(row[0]["geometric_correct"]):
                total += 1
    return total


def _count_native_top1_correct(artifact: CachedCandidateArtifact) -> int:
    total = 0
    for batch in artifact.batches:
        correct_rows = batch.candidate_geometric_correct or []
        for row in correct_rows:
            if row and bool(row[0]):
                total += 1
    return total
