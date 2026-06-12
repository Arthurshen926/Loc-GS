from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from loc_gs.sparse.artifact_adapter import CachedCandidateArtifact
from loc_gs.sparse.correspondences import SparseCandidateBatch


TEACHER_ONLY_FEATURE_NAMES = (
    "dense_consistent",
    "sparse_inlier",
    "negative_reprojection_error",
    "solver_weight",
)


@dataclass(frozen=True)
class CandidateScorerConfig:
    epochs: int = 100
    learning_rate: float = 0.1
    rank_feature_scale: float = 1.0
    reprojection_error_scale_px: float = 8.0
    protected_support_weight: float = 2.0
    positive_inlier_weight: float = 1.25
    hard_negative_weight: float = 1.5
    neutral_weight: float = 1.0
    feature_names: tuple[str, ...] = (
        "native_score",
        "negative_rank",
        "valid",
    )


@dataclass(frozen=True)
class LinearCandidateScorer:
    weights: tuple[float, ...]
    bias: float
    feature_names: tuple[str, ...] = CandidateScorerConfig.feature_names

    def to_json_dict(self) -> dict[str, object]:
        feature_policy = classify_feature_input_policy(self.feature_names)
        return {
            "schema_version": "internal_sparse_candidate_scorer_v1",
            **asdict(self),
            **feature_policy,
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> "LinearCandidateScorer":
        weights = tuple(float(value) for value in payload.get("weights", ()))
        if not weights:
            raise ValueError("internal sparse candidate scorer weights must not be empty")
        feature_names = tuple(
            str(value)
            for value in payload.get("feature_names", CandidateScorerConfig.feature_names[: len(weights)])
        )
        if len(feature_names) != len(weights):
            raise ValueError("internal sparse candidate scorer feature_names must match weights")
        return cls(
            weights=weights,
            bias=float(payload.get("bias", 0.0)),
            feature_names=feature_names,
        )


def classify_feature_input_policy(feature_names: Sequence[str]) -> dict[str, object]:
    names = tuple(str(name) for name in feature_names)
    teacher_only = [name for name in TEACHER_ONLY_FEATURE_NAMES if name in names]
    inference_safe = not teacher_only
    return {
        "feature_input_policy": "inference_safe" if inference_safe else "teacher_oracle_diagnostic",
        "paper_safe_sparse_inference": bool(inference_safe),
        "teacher_only_feature_names": teacher_only,
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def _grid_value(rows: Sequence[Sequence[object]] | None, row_idx: int, rank: int, default: object) -> object:
    if rows is None or row_idx >= len(rows) or rank >= len(rows[row_idx]):
        return default
    return rows[row_idx][rank]


def _normalized_descriptor(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size == 0 or not np.all(np.isfinite(array)):
        return None
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-12:
        return None
    return array / norm


def _descriptor_pair_features(batch: SparseCandidateBatch, *, row_idx: int, rank: int) -> dict[str, float]:
    query_desc = None
    if batch.query_descriptors is not None and row_idx < len(batch.query_descriptors):
        query_desc = _normalized_descriptor(batch.query_descriptors[row_idx])
    landmark_desc = _normalized_descriptor(
        _grid_value(batch.candidate_landmark_descriptors, row_idx, rank, None)
    )
    if query_desc is None or landmark_desc is None or query_desc.shape != landmark_desc.shape:
        return {
            "descriptor_alignment": 0.0,
            "negative_descriptor_l2": 0.0,
            "negative_descriptor_abs_diff_mean": 0.0,
            "descriptor_product_mean": 0.0,
        }
    diff = query_desc - landmark_desc
    return {
        "descriptor_alignment": float(query_desc @ landmark_desc),
        "negative_descriptor_l2": -float(np.linalg.norm(diff)),
        "negative_descriptor_abs_diff_mean": -float(np.mean(np.abs(diff))),
        "descriptor_product_mean": float(np.mean(query_desc * landmark_desc)),
    }


def _candidate_feature(
    batch: SparseCandidateBatch,
    *,
    row_idx: int,
    rank: int,
    native_score: float,
    valid: bool,
    feature_names: Sequence[str],
    rank_feature_scale: float = 1.0,
    reprojection_error_scale_px: float = 8.0,
) -> list[float]:
    dense = bool(_grid_value(batch.candidate_dense_consistent, row_idx, rank, False))
    sparse_inlier = bool(_grid_value(batch.candidate_sparse_inlier, row_idx, rank, False))
    reprojection = float(_grid_value(batch.candidate_reprojection_error_px, row_idx, rank, 0.0))
    solver_weight = float(_grid_value(batch.candidate_solver_weight, row_idx, rank, 1.0))
    margin = float(_grid_value(batch.candidate_margin, row_idx, rank, 0.0))
    query_score = float(_grid_value(batch.candidate_query_score, row_idx, rank, 1.0))
    landmark_prior = float(_grid_value(batch.candidate_landmark_prior, row_idx, rank, 0.0))
    scale = max(1.0e-9, float(reprojection_error_scale_px))
    feature_values = {
        "native_score": float(native_score),
        "negative_rank": -float(rank) * float(rank_feature_scale),
        "valid": 1.0 if valid else 0.0,
        "dense_consistent": 1.0 if dense else 0.0,
        "sparse_inlier": 1.0 if sparse_inlier else 0.0,
        "negative_reprojection_error": -float(reprojection) / scale,
        "solver_weight": float(solver_weight),
        "margin": float(margin),
        "query_score": float(query_score),
        "landmark_prior": float(landmark_prior),
    }
    feature_values.update(_descriptor_pair_features(batch, row_idx=row_idx, rank=rank))
    try:
        return [float(feature_values[str(name)]) for name in feature_names]
    except KeyError as exc:
        raise ValueError(f"unsupported candidate scorer feature: {exc.args[0]}") from exc


def _candidate_target(batch: SparseCandidateBatch, *, row_idx: int, rank: int) -> float:
    role = str(_grid_value(batch.candidate_label_roles, row_idx, rank, ""))
    if role in {"protected_support", "positive_inlier"}:
        return 1.0
    if role == "hard_negative":
        return 0.0
    geometric = bool(_grid_value(batch.candidate_geometric_correct, row_idx, rank, False))
    if batch.candidate_dense_consistent is not None:
        return 1.0 if geometric and bool(_grid_value(batch.candidate_dense_consistent, row_idx, rank, False)) else 0.0
    if batch.candidate_sparse_inlier is not None:
        return 1.0 if geometric and bool(_grid_value(batch.candidate_sparse_inlier, row_idx, rank, False)) else 0.0
    return 1.0 if geometric else 0.0


def _candidate_sample_weight(batch: SparseCandidateBatch, *, row_idx: int, rank: int, target: float, cfg: CandidateScorerConfig) -> float:
    role = str(_grid_value(batch.candidate_label_roles, row_idx, rank, ""))
    dense = bool(_grid_value(batch.candidate_dense_consistent, row_idx, rank, False))
    sparse_inlier = bool(_grid_value(batch.candidate_sparse_inlier, row_idx, rank, False))
    reprojection = float(_grid_value(batch.candidate_reprojection_error_px, row_idx, rank, 0.0))
    solver_weight = max(0.0, float(_grid_value(batch.candidate_solver_weight, row_idx, rank, 1.0)))
    if role == "protected_support" or (target > 0.5 and dense and sparse_inlier):
        base = float(cfg.protected_support_weight)
    elif role == "positive_inlier" or target > 0.5:
        base = float(cfg.positive_inlier_weight)
    elif role == "hard_negative" or reprojection >= float(cfg.reprojection_error_scale_px):
        base = float(cfg.hard_negative_weight)
    else:
        base = float(cfg.neutral_weight)
    return float(base * max(1.0e-6, solver_weight))


def _training_matrix(artifact: CachedCandidateArtifact, cfg: CandidateScorerConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]:
    features: list[list[float]] = []
    labels: list[float] = []
    weights: list[float] = []
    dense_teacher_sample_count = 0
    for batch in artifact.batches:
        batch.validate()
        correct_rows = batch.candidate_geometric_correct or []
        valid_rows = batch.candidate_valid_mask or []
        for row_idx, scores in enumerate(batch.candidate_scores):
            if row_idx >= len(correct_rows):
                continue
            for rank, score in enumerate(scores):
                valid = True if not valid_rows else bool(valid_rows[row_idx][rank])
                if not valid:
                    continue
                if batch.candidate_dense_consistent is not None:
                    dense_teacher_sample_count += 1
                target = _candidate_target(batch, row_idx=row_idx, rank=rank)
                features.append(
                    _candidate_feature(
                        batch,
                        row_idx=row_idx,
                        rank=rank,
                        native_score=float(score),
                        valid=valid,
                        feature_names=cfg.feature_names,
                        rank_feature_scale=float(cfg.rank_feature_scale),
                        reprojection_error_scale_px=float(cfg.reprojection_error_scale_px),
                    )
                )
                labels.append(target)
                weights.append(_candidate_sample_weight(batch, row_idx=row_idx, rank=rank, target=target, cfg=cfg))
    if not features:
        raise ValueError("candidate artifact did not contain trainable labels")
    stats = {
        "dense_teacher_sample_count": int(dense_teacher_sample_count),
        "weighted_sample_count": float(sum(weights)),
    }
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        stats,
    )


def train_candidate_scorer(
    artifact: CachedCandidateArtifact,
    cfg: CandidateScorerConfig | None = None,
) -> tuple[LinearCandidateScorer, dict[str, object]]:
    if cfg is None:
        cfg = CandidateScorerConfig()
    x, y, sample_weights, training_stats = _training_matrix(artifact, cfg)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    normalizer = max(1.0e-9, float(sample_weights.sum()))
    for _epoch in range(int(cfg.epochs)):
        pred = _sigmoid(x @ weights + bias)
        err = (pred - y) * sample_weights
        weights -= float(cfg.learning_rate) * (x.T @ err) / normalizer
        bias -= float(cfg.learning_rate) * float(err.sum() / normalizer)
    model = LinearCandidateScorer(
        weights=tuple(float(v) for v in weights),
        bias=float(bias),
        feature_names=tuple(str(name) for name in cfg.feature_names),
    )
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
        **classify_feature_input_policy(cfg.feature_names),
        **training_stats,
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
            feature = np.asarray(
                _candidate_feature(
                    batch,
                    row_idx=row_idx,
                    rank=rank,
                    native_score=float(score),
                    valid=valid,
                    feature_names=model.feature_names,
                ),
                dtype=np.float64,
            )
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


def candidate_solver_score_rows(batch: SparseCandidateBatch, model: object) -> list[list[float]]:
    if not isinstance(model, LinearCandidateScorer):
        from loc_gs.students.candidate_mlp_scorer import CandidateMLPScorer, candidate_mlp_score_rows

        if isinstance(model, CandidateMLPScorer):
            return candidate_mlp_score_rows(batch, model)
        raise TypeError(f"unsupported candidate scorer model type: {type(model).__name__}")
    weights = np.asarray(model.weights, dtype=np.float64)
    rows: list[list[float]] = []
    valid_rows = batch.candidate_valid_mask or []
    for row_idx, scores in enumerate(batch.candidate_scores):
        row: list[float] = []
        for rank, score in enumerate(scores):
            valid = True if not valid_rows else bool(valid_rows[row_idx][rank])
            feature = np.asarray(
                _candidate_feature(
                    batch,
                    row_idx=row_idx,
                    rank=rank,
                    native_score=float(score),
                    valid=valid,
                    feature_names=model.feature_names,
                ),
                dtype=np.float64,
            )
            row.append(float(feature @ weights + float(model.bias)))
        rows.append(row)
    return rows


def load_candidate_scorer(path: str | Path) -> object:
    source = Path(path)
    if source.suffix.lower() == ".pt":
        from loc_gs.students.candidate_mlp_scorer import load_candidate_mlp_scorer

        return load_candidate_mlp_scorer(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
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
