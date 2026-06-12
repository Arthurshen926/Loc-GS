from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from loc_gs.sparse.artifact_adapter import CachedCandidateArtifact
from loc_gs.sparse.correspondences import SparseCandidateBatch
from loc_gs.training.sparse_candidate_scorer import (
    CandidateScorerConfig,
    _candidate_feature,
    _candidate_sample_weight,
    _candidate_target,
    classify_feature_input_policy,
)


@dataclass(frozen=True)
class CandidateMLPScorerConfig:
    epochs: int = 100
    learning_rate: float = 0.03
    hidden_dim: int = 32
    seed: int = 13
    listwise_loss_weight: float = 1.0
    batch_size: int = 0
    rank_feature_scale: float = 1.0
    reprojection_error_scale_px: float = 8.0
    protected_support_weight: float = 2.0
    positive_inlier_weight: float = 1.25
    hard_negative_weight: float = 1.5
    neutral_weight: float = 1.0
    scalar_feature_names: tuple[str, ...] = (
        "native_score",
        "negative_rank",
        "valid",
        "margin",
        "query_score",
        "landmark_prior",
    )

    def candidate_config(self) -> CandidateScorerConfig:
        return CandidateScorerConfig(
            epochs=int(self.epochs),
            learning_rate=float(self.learning_rate),
            rank_feature_scale=float(self.rank_feature_scale),
            reprojection_error_scale_px=float(self.reprojection_error_scale_px),
            protected_support_weight=float(self.protected_support_weight),
            positive_inlier_weight=float(self.positive_inlier_weight),
            hard_negative_weight=float(self.hard_negative_weight),
            neutral_weight=float(self.neutral_weight),
            feature_names=tuple(str(name) for name in self.scalar_feature_names),
        )


@dataclass(frozen=True)
class CandidateMLPScorer:
    scalar_feature_names: tuple[str, ...]
    descriptor_dim: int
    hidden_dim: int
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    state_dict: Mapping[str, torch.Tensor]
    logit_mean: float = 0.0
    logit_std: float = 1.0

    @property
    def input_dim(self) -> int:
        return len(self.scalar_feature_names) + 4 * int(self.descriptor_dim)

    def to_torch_dict(self) -> dict[str, object]:
        feature_policy = classify_feature_input_policy(self.scalar_feature_names)
        return {
            "schema_version": "internal_candidate_mlp_scorer_v1",
            "scalar_feature_names": list(self.scalar_feature_names),
            "descriptor_dim": int(self.descriptor_dim),
            "hidden_dim": int(self.hidden_dim),
            "input_dim": int(self.input_dim),
            "feature_mean": torch.tensor(self.feature_mean, dtype=torch.float32),
            "feature_std": torch.tensor(self.feature_std, dtype=torch.float32),
            "state_dict": {str(key): value.detach().cpu() for key, value in self.state_dict.items()},
            "score_calibration": "train_logit_zscore",
            "logit_mean": float(self.logit_mean),
            "logit_std": float(max(1.0e-6, self.logit_std)),
            **feature_policy,
        }

    @classmethod
    def from_torch_dict(cls, payload: Mapping[str, object]) -> "CandidateMLPScorer":
        if payload.get("schema_version") != "internal_candidate_mlp_scorer_v1":
            raise ValueError(f"unsupported candidate MLP scorer schema: {payload.get('schema_version')}")
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("candidate MLP scorer payload missing state_dict")
        feature_mean = _tensor_to_tuple(payload.get("feature_mean"), name="feature_mean")
        feature_std = _tensor_to_tuple(payload.get("feature_std"), name="feature_std")
        scalar_feature_names = tuple(str(name) for name in payload.get("scalar_feature_names", ()))
        descriptor_dim = int(payload.get("descriptor_dim", 0))
        hidden_dim = int(payload.get("hidden_dim", 0))
        expected_dim = len(scalar_feature_names) + 4 * descriptor_dim
        if descriptor_dim <= 0:
            raise ValueError("candidate MLP scorer descriptor_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("candidate MLP scorer hidden_dim must be positive")
        if len(feature_mean) != expected_dim or len(feature_std) != expected_dim:
            raise ValueError("candidate MLP scorer normalization vectors do not match input_dim")
        return cls(
            scalar_feature_names=scalar_feature_names,
            descriptor_dim=descriptor_dim,
            hidden_dim=hidden_dim,
            feature_mean=feature_mean,
            feature_std=feature_std,
            state_dict={str(key): _as_cpu_tensor(value) for key, value in state_dict.items()},
            logit_mean=float(payload.get("logit_mean", 0.0)),
            logit_std=max(1.0e-6, float(payload.get("logit_std", 1.0))),
        )


def train_candidate_mlp_scorer(
    artifact: CachedCandidateArtifact,
    cfg: CandidateMLPScorerConfig | None = None,
) -> tuple[CandidateMLPScorer, dict[str, object]]:
    if cfg is None:
        cfg = CandidateMLPScorerConfig()
    descriptor_dim = _infer_descriptor_dim(artifact)
    x, y, sample_weights, group_slices, training_stats = _training_matrix(artifact, cfg, descriptor_dim=descriptor_dim)
    torch.manual_seed(int(cfg.seed))
    features = torch.tensor(x, dtype=torch.float32)
    labels = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    weights = torch.tensor(sample_weights.reshape(-1, 1), dtype=torch.float32)
    feature_mean = features.mean(dim=0)
    feature_std = features.std(dim=0, unbiased=False)
    feature_std = torch.where(feature_std < 1.0e-6, torch.ones_like(feature_std), feature_std)
    features = (features - feature_mean) / feature_std
    network = _network(features.shape[1], int(cfg.hidden_dim))
    optimizer = torch.optim.AdamW(network.parameters(), lr=float(cfg.learning_rate))
    train_batches = _training_batches(
        sample_count=int(features.shape[0]),
        group_slices=group_slices,
        batch_size=int(cfg.batch_size),
    )
    optimizer_step_count = 0
    for _epoch in range(int(cfg.epochs)):
        for start, end, batch_group_slices in train_batches:
            batch_features = features[int(start) : int(end)]
            batch_labels = labels[int(start) : int(end)]
            batch_weights = weights[int(start) : int(end)]
            logits = network(batch_features)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_labels, reduction="none")
            normalizer = torch.clamp(batch_weights.sum(), min=torch.tensor(1.0e-6))
            pointwise = (loss * batch_weights).sum() / normalizer
            objective = pointwise + float(cfg.listwise_loss_weight) * _listwise_softmax_loss(
                logits,
                batch_labels,
                batch_weights,
                batch_group_slices,
            )
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
            optimizer_step_count += 1
    with torch.no_grad():
        train_logits = network(features).reshape(-1)
        logit_mean = float(train_logits.mean().item())
        logit_std = float(train_logits.std(unbiased=False).item())
        if logit_std <= 1.0e-6:
            logit_std = 1.0
    model = CandidateMLPScorer(
        scalar_feature_names=tuple(str(name) for name in cfg.scalar_feature_names),
        descriptor_dim=int(descriptor_dim),
        hidden_dim=int(cfg.hidden_dim),
        feature_mean=tuple(float(v) for v in feature_mean.tolist()),
        feature_std=tuple(float(v) for v in feature_std.tolist()),
        state_dict={str(key): value.detach().cpu() for key, value in network.state_dict().items()},
        logit_mean=float(logit_mean),
        logit_std=float(logit_std),
    )
    summary = {
        "schema_version": "internal_candidate_mlp_scorer_training_summary_v1",
        "student_modules": ["candidate_mlp_scorer"],
        "label_count": int(y.sum()),
        "sample_count": int(y.shape[0]),
        "native_top1_correct": int(_count_native_top1_correct(artifact)),
        "trained_top1_correct": int(_count_top1_correct(artifact, model)),
        "epochs": int(cfg.epochs),
        "learning_rate": float(cfg.learning_rate),
        "hidden_dim": int(cfg.hidden_dim),
        "listwise_loss_weight": float(cfg.listwise_loss_weight),
        "batch_size": int(cfg.batch_size),
        "training_batch_count": int(len(train_batches)),
        "optimizer_step_count": int(optimizer_step_count),
        "descriptor_dim": int(descriptor_dim),
        "score_calibration": "train_logit_zscore",
        "logit_mean": float(logit_mean),
        "logit_std": float(logit_std),
        "hyperparameters": asdict(cfg),
        **classify_feature_input_policy(cfg.scalar_feature_names),
        **training_stats,
    }
    return model, summary


def candidate_mlp_score_rows(batch: SparseCandidateBatch, model: CandidateMLPScorer) -> list[list[float]]:
    batch.validate()
    network = _network(model.input_dim, int(model.hidden_dim))
    network.load_state_dict(dict(model.state_dict))
    network.eval()
    mean = torch.tensor(model.feature_mean, dtype=torch.float32)
    std = torch.tensor(model.feature_std, dtype=torch.float32)
    valid_rows = batch.candidate_valid_mask or []
    rows: list[list[float]] = []
    with torch.no_grad():
        for row_idx, scores in enumerate(batch.candidate_scores):
            feature_rows: list[list[float]] = []
            valid_flags: list[bool] = []
            for rank, score in enumerate(scores):
                valid = True if not valid_rows else bool(valid_rows[row_idx][rank])
                valid_flags.append(valid)
                feature_rows.append(
                    _candidate_mlp_feature(
                        batch,
                        row_idx=row_idx,
                        rank=rank,
                        native_score=float(score),
                        valid=valid,
                        cfg=model,
                    )
                )
            if not feature_rows:
                rows.append([])
                continue
            features = (torch.tensor(feature_rows, dtype=torch.float32) - mean) / std
            logits = network(features).reshape(-1)
            calibrated = ((logits - float(model.logit_mean)) / max(1.0e-6, float(model.logit_std))).tolist()
            rows.append([float(value) if valid else -1.0e12 for value, valid in zip(calibrated, valid_flags)])
    return rows


def load_candidate_mlp_scorer(path: str | Path) -> CandidateMLPScorer:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"candidate MLP scorer torch payload must contain an object: {path}")
    return CandidateMLPScorer.from_torch_dict(payload)


def _training_matrix(
    artifact: CachedCandidateArtifact,
    cfg: CandidateMLPScorerConfig,
    *,
    descriptor_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]], dict[str, int | float]]:
    features: list[list[float]] = []
    labels: list[float] = []
    weights: list[float] = []
    group_slices: list[tuple[int, int]] = []
    dense_teacher_sample_count = 0
    linear_cfg = cfg.candidate_config()
    for batch in artifact.batches:
        batch.validate()
        valid_rows = batch.candidate_valid_mask or []
        for row_idx, scores in enumerate(batch.candidate_scores):
            row_start = len(labels)
            for rank, score in enumerate(scores):
                valid = True if not valid_rows else bool(valid_rows[row_idx][rank])
                if not valid:
                    continue
                if batch.candidate_dense_consistent is not None:
                    dense_teacher_sample_count += 1
                target = _candidate_target(batch, row_idx=row_idx, rank=rank)
                features.append(
                    _candidate_mlp_feature(
                        batch,
                        row_idx=row_idx,
                        rank=rank,
                        native_score=float(score),
                        valid=valid,
                        cfg=cfg,
                        descriptor_dim=descriptor_dim,
                    )
                )
                labels.append(target)
                weights.append(_candidate_sample_weight(batch, row_idx=row_idx, rank=rank, target=target, cfg=linear_cfg))
            if len(labels) > row_start:
                group_slices.append((int(row_start), int(len(labels))))
    if not features:
        raise ValueError("candidate artifact did not contain trainable MLP scorer samples")
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
        group_slices,
        {
            "dense_teacher_sample_count": int(dense_teacher_sample_count),
            "weighted_sample_count": float(sum(weights)),
        },
    )


def _listwise_softmax_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    group_slices: Sequence[tuple[int, int]],
) -> torch.Tensor:
    flat_logits = logits.reshape(-1)
    flat_labels = labels.reshape(-1)
    flat_weights = sample_weights.reshape(-1)
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for start, end in group_slices:
        group_labels = flat_labels[int(start) : int(end)]
        positives = group_labels > 0.5
        if not bool(positives.any()):
            continue
        target = group_labels / torch.clamp(group_labels.sum(), min=torch.tensor(1.0e-6, device=group_labels.device))
        group_logits = flat_logits[int(start) : int(end)]
        losses.append(-(target * torch.nn.functional.log_softmax(group_logits, dim=0)).sum())
        weights.append(flat_weights[int(start) : int(end)].mean())
    if not losses:
        return flat_logits.sum() * 0.0
    loss_tensor = torch.stack(losses)
    weight_tensor = torch.stack(weights)
    return (loss_tensor * weight_tensor).sum() / torch.clamp(weight_tensor.sum(), min=torch.tensor(1.0e-6, device=weight_tensor.device))


def _training_batches(
    *,
    sample_count: int,
    group_slices: Sequence[tuple[int, int]],
    batch_size: int,
) -> list[tuple[int, int, list[tuple[int, int]]]]:
    if sample_count <= 0:
        return []
    if int(batch_size) <= 0 or int(batch_size) >= int(sample_count):
        return [(0, int(sample_count), [(int(start), int(end)) for start, end in group_slices])]
    batches: list[tuple[int, int, list[tuple[int, int]]]] = []
    batch_start: int | None = None
    batch_end: int | None = None
    batch_groups: list[tuple[int, int]] = []
    for group_start, group_end in group_slices:
        group_start = int(group_start)
        group_end = int(group_end)
        if batch_start is None:
            batch_start = group_start
            batch_end = group_end
            batch_groups = [(0, group_end - group_start)]
            continue
        assert batch_end is not None
        next_count = group_end - batch_start
        if batch_groups and next_count > int(batch_size):
            batches.append((batch_start, batch_end, batch_groups))
            batch_start = group_start
            batch_end = group_end
            batch_groups = [(0, group_end - group_start)]
        else:
            batch_groups.append((group_start - batch_start, group_end - batch_start))
            batch_end = group_end
    if batch_start is not None and batch_end is not None:
        batches.append((batch_start, batch_end, batch_groups))
    return batches or [(0, int(sample_count), [(int(start), int(end)) for start, end in group_slices])]


def _candidate_mlp_feature(
    batch: SparseCandidateBatch,
    *,
    row_idx: int,
    rank: int,
    native_score: float,
    valid: bool,
    cfg: CandidateMLPScorerConfig | CandidateMLPScorer,
    descriptor_dim: int | None = None,
) -> list[float]:
    scalar_names = tuple(str(name) for name in cfg.scalar_feature_names)
    rank_feature_scale = getattr(cfg, "rank_feature_scale", 1.0)
    reprojection_error_scale_px = getattr(cfg, "reprojection_error_scale_px", 8.0)
    scalar = _candidate_feature(
        batch,
        row_idx=row_idx,
        rank=rank,
        native_score=float(native_score),
        valid=bool(valid),
        feature_names=scalar_names,
        rank_feature_scale=float(rank_feature_scale),
        reprojection_error_scale_px=float(reprojection_error_scale_px),
    )
    dim = int(descriptor_dim if descriptor_dim is not None else cfg.descriptor_dim)
    return [*scalar, *_descriptor_pair_vector(batch, row_idx=row_idx, rank=rank, descriptor_dim=dim)]


def _descriptor_pair_vector(
    batch: SparseCandidateBatch,
    *,
    row_idx: int,
    rank: int,
    descriptor_dim: int,
) -> list[float]:
    query_desc = None
    if batch.query_descriptors is not None and row_idx < len(batch.query_descriptors):
        query_desc = _normalized_vector(batch.query_descriptors[row_idx], descriptor_dim=descriptor_dim)
    landmark_desc = None
    if batch.candidate_landmark_descriptors is not None and row_idx < len(batch.candidate_landmark_descriptors):
        row = batch.candidate_landmark_descriptors[row_idx]
        if rank < len(row):
            landmark_desc = _normalized_vector(row[rank], descriptor_dim=descriptor_dim)
    if query_desc is None:
        query_desc = np.zeros((descriptor_dim,), dtype=np.float32)
    if landmark_desc is None:
        landmark_desc = np.zeros((descriptor_dim,), dtype=np.float32)
    return [
        *[float(v) for v in query_desc.tolist()],
        *[float(v) for v in landmark_desc.tolist()],
        *[float(v) for v in np.abs(query_desc - landmark_desc).tolist()],
        *[float(v) for v in (query_desc * landmark_desc).tolist()],
    ]


def _infer_descriptor_dim(artifact: CachedCandidateArtifact) -> int:
    for batch in artifact.batches:
        batch.validate()
        if batch.query_descriptors is None or batch.candidate_landmark_descriptors is None:
            continue
        for row_idx, query_desc in enumerate(batch.query_descriptors):
            q = np.asarray(query_desc, dtype=np.float32).reshape(-1)
            if q.size == 0:
                continue
            if row_idx >= len(batch.candidate_landmark_descriptors):
                continue
            for landmark_desc in batch.candidate_landmark_descriptors[row_idx]:
                l = np.asarray(landmark_desc, dtype=np.float32).reshape(-1)
                if l.size == q.size and q.size > 0:
                    return int(q.size)
    raise ValueError("candidate MLP scorer training requires query and landmark descriptors")


def _normalized_vector(value: Sequence[float], *, descriptor_dim: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size != int(descriptor_dim) or not np.all(np.isfinite(array)):
        return None
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-12:
        return None
    return array / norm


def _network(input_dim: int, hidden_dim: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(int(input_dim), int(hidden_dim)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(hidden_dim), 1),
    )


def _count_top1_correct(artifact: CachedCandidateArtifact, model: CandidateMLPScorer) -> int:
    total = 0
    for batch in artifact.batches:
        correct_rows = batch.candidate_geometric_correct or []
        score_rows = candidate_mlp_score_rows(batch, model)
        for row_idx, row in enumerate(score_rows):
            if not row or row_idx >= len(correct_rows):
                continue
            best_rank = max(range(len(row)), key=lambda idx: float(row[idx]))
            if bool(correct_rows[row_idx][best_rank]):
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


def _as_cpu_tensor(value: object) -> torch.Tensor:
    if hasattr(value, "detach"):
        return value.detach().cpu()
    return torch.tensor(value, dtype=torch.float32)


def _tensor_to_tuple(value: Any, *, name: str) -> tuple[float, ...]:
    if value is None:
        raise ValueError(f"candidate MLP scorer payload missing {name}")
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"candidate MLP scorer payload {name} must not be empty")
    return tuple(float(item) for item in array.tolist())
