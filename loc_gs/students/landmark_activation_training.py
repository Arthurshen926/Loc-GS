from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from loc_gs.sparse.artifact_adapter import CachedCandidateArtifact
from loc_gs.sparse.correspondences import SparseCandidateBatch
from loc_gs.training.query_landmark_activation_v2 import (
    DEFAULT_LANDMARK_TOKEN_COMPONENTS,
    QueryLandmarkActivationV2,
    build_landmark_activation_v2_landmark_tokens,
    query_landmark_activation_v2_loss,
)


@dataclass(frozen=True)
class LandmarkActivationTrainingConfig:
    epochs: int = 0
    learning_rate: float = 1.0e-3
    hidden_dim: int = 64
    attention_top_k: int = 32
    seed: int = 13
    top_n: int = 0


@dataclass(frozen=True)
class LandmarkActivationTrainingExample:
    query_tokens: torch.Tensor
    landmark_tokens: torch.Tensor
    landmark_ids: torch.Tensor
    geometry_visible: torch.Tensor
    solver_positive: torch.Tensor
    harmful_negative: torch.Tensor
    protected_support: torch.Tensor


def train_landmark_activation_v2_from_artifact(
    artifact: CachedCandidateArtifact,
    *,
    output_path: str | Path,
    cfg: LandmarkActivationTrainingConfig,
) -> tuple[Path, dict[str, object]]:
    examples = _examples_from_artifact(artifact)
    if not examples:
        raise ValueError("candidate artifact did not contain activation training examples")
    query_token_dim = int(examples[0].query_tokens.shape[1])
    landmark_token_dim = int(examples[0].landmark_tokens.shape[1])
    torch.manual_seed(int(cfg.seed))
    model = QueryLandmarkActivationV2(
        query_token_dim=query_token_dim,
        landmark_token_dim=landmark_token_dim,
        hidden_dim=int(cfg.hidden_dim),
        attention_top_k=int(cfg.attention_top_k),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate))
    history: list[dict[str, float | int]] = []
    for epoch in range(int(cfg.epochs)):
        total = 0.0
        for example in examples:
            logits = model(query_tokens=example.query_tokens, landmark_tokens=example.landmark_tokens)
            loss_terms = query_landmark_activation_v2_loss(
                logits,
                {
                    "geometry_visible": example.geometry_visible,
                    "solver_positive": example.solver_positive,
                    "harmful_negative": example.harmful_negative,
                    "protected_support": example.protected_support,
                },
            )
            loss = loss_terms["total_loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().item())
        history.append(
            {
                "epoch": int(epoch),
                "mean_loss": float(total / max(1, len(examples))),
                "example_count": int(len(examples)),
            }
        )
    landmark_ids = torch.unique(torch.cat([example.landmark_ids.reshape(-1) for example in examples])).cpu()
    checkpoint = {
        "activation_model_type": "v2_tokens",
        "query_feature_mode": "token_cross_attention",
        "landmark_token_components": list(DEFAULT_LANDMARK_TOKEN_COMPONENTS),
        "state_dict": model.cpu().state_dict(),
        "query_token_dim": int(query_token_dim),
        "landmark_token_dim": int(landmark_token_dim),
        "hidden_dim": int(cfg.hidden_dim),
        "attention_top_k": int(cfg.attention_top_k),
        "top_n": int(cfg.top_n),
        "landmark_ids": landmark_ids,
        "split_name": str(artifact.split_name),
        "split_audit": dict(artifact.metadata.get("split_audit", {})),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)
    summary = {
        "schema_version": "internal_landmark_activation_v2_training_summary_v1",
        "student_modules": ["landmark_activation_v2"],
        "activation_model_type": "v2_tokens",
        "query_feature_mode": "token_cross_attention",
        "split_name": str(artifact.split_name),
        "epochs": int(cfg.epochs),
        "learning_rate": float(cfg.learning_rate),
        "hidden_dim": int(cfg.hidden_dim),
        "attention_top_k": int(cfg.attention_top_k),
        "top_n": int(cfg.top_n),
        "trained_example_count": int(len(examples)),
        "landmark_count": int(landmark_ids.numel()),
        "final_loss": float(history[-1]["mean_loss"]) if history else None,
        "history": history,
        "paper_safe_sparse_inference": True,
    }
    return target, summary


def _examples_from_artifact(artifact: CachedCandidateArtifact) -> list[LandmarkActivationTrainingExample]:
    return [_example_from_batch(batch) for batch in artifact.batches if batch.keypoint_count > 0]


def _example_from_batch(batch: SparseCandidateBatch) -> LandmarkActivationTrainingExample:
    batch.validate()
    if batch.query_descriptors is None:
        raise ValueError("landmark activation training requires query_descriptors")
    if batch.candidate_landmark_descriptors is None:
        raise ValueError("landmark activation training requires candidate_landmark_descriptors")
    query_tokens = torch.tensor(batch.query_descriptors, dtype=torch.float32)
    ids: list[int] = []
    descriptors: list[Sequence[float]] = []
    index_by_landmark: dict[int, int] = {}
    geometry = []
    positive = []
    harmful = []
    protected = []
    for row_idx, row_ids in enumerate(batch.candidate_landmark_ids):
        role_row = batch.candidate_label_roles[row_idx] if batch.candidate_label_roles is not None else []
        correct_row = batch.candidate_geometric_correct[row_idx] if batch.candidate_geometric_correct is not None else []
        dense_row = batch.candidate_dense_consistent[row_idx] if batch.candidate_dense_consistent is not None else []
        sparse_row = batch.candidate_sparse_inlier[row_idx] if batch.candidate_sparse_inlier is not None else []
        for rank, landmark_id in enumerate(row_ids):
            key = int(landmark_id)
            if key not in index_by_landmark:
                index_by_landmark[key] = len(ids)
                ids.append(key)
                descriptors.append(batch.candidate_landmark_descriptors[row_idx][rank])
                geometry.append(False)
                positive.append(False)
                harmful.append(False)
                protected.append(False)
            local_idx = index_by_landmark[key]
            role = str(role_row[rank]) if rank < len(role_row) else ""
            is_correct = rank < len(correct_row) and bool(correct_row[rank])
            is_dense = rank < len(dense_row) and bool(dense_row[rank])
            is_sparse = rank < len(sparse_row) and bool(sparse_row[rank])
            geometry[local_idx] = bool(geometry[local_idx] or is_correct or is_dense or is_sparse)
            positive[local_idx] = bool(
                positive[local_idx]
                or is_dense
                or is_sparse
                or role in {"positive_inlier", "protected_support"}
            )
            harmful[local_idx] = bool(harmful[local_idx] or role == "hard_negative")
            protected[local_idx] = bool(protected[local_idx] or role == "protected_support")
    landmark_desc = torch.tensor(descriptors, dtype=torch.float32)
    landmark_tokens = build_landmark_activation_v2_landmark_tokens(landmark_desc)
    return LandmarkActivationTrainingExample(
        query_tokens=query_tokens,
        landmark_tokens=landmark_tokens,
        landmark_ids=torch.tensor(ids, dtype=torch.long),
        geometry_visible=torch.tensor(geometry, dtype=torch.float32),
        solver_positive=torch.tensor(positive, dtype=torch.float32),
        harmful_negative=torch.tensor(harmful, dtype=torch.float32),
        protected_support=torch.tensor(protected, dtype=torch.float32),
    )
