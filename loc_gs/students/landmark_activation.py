from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.correspondences import SparseCandidateBatch
from loc_gs.training.query_landmark_activation_v2 import (
    QueryLandmarkActivationV2,
    build_landmark_activation_v2_landmark_tokens,
)


@dataclass
class LandmarkActivationRuntime:
    model: QueryLandmarkActivationV2
    query_token_dim: int
    landmark_token_dim: int
    top_n: int
    split_name: str

    def score(self, query_tokens: torch.Tensor, landmark_descriptors: torch.Tensor) -> torch.Tensor:
        query = _pad_or_truncate(torch.as_tensor(query_tokens, dtype=torch.float32), int(self.query_token_dim))
        landmark_tokens = build_landmark_activation_v2_landmark_tokens(
            torch.as_tensor(landmark_descriptors, dtype=torch.float32),
            expected_dim=int(self.landmark_token_dim),
        )
        with torch.no_grad():
            logits = self.model(query_tokens=query, landmark_tokens=landmark_tokens).detach().cpu().reshape(-1)
        if int(self.top_n) > 0 and int(self.top_n) < int(logits.numel()):
            keep = torch.topk(logits, k=int(self.top_n)).indices
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[keep] = True
            floor = float(logits.min().item()) - 10.0
            logits = torch.where(mask, logits, torch.full_like(logits, floor))
        return logits


def load_landmark_activation(path: str | Path) -> LandmarkActivationRuntime:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"landmark activation checkpoint must contain a mapping: {path}")
    if payload.get("activation_model_type") != "v2_tokens":
        raise ValueError(f"unsupported landmark activation checkpoint type: {payload.get('activation_model_type')}")
    split_name = _checkpoint_split_name(payload)
    reject_test_split(split_name, purpose="landmark activation sparse inference")
    query_token_dim = int(payload.get("query_token_dim", 0))
    landmark_token_dim = int(payload.get("landmark_token_dim", 0))
    hidden_dim = int(payload.get("hidden_dim", 0))
    attention_top_k = int(payload.get("attention_top_k", 1))
    if query_token_dim <= 0 or landmark_token_dim <= 0 or hidden_dim <= 0:
        raise ValueError("landmark activation checkpoint is missing token dimensions")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("landmark activation checkpoint is missing state_dict")
    model = QueryLandmarkActivationV2(
        query_token_dim=query_token_dim,
        landmark_token_dim=landmark_token_dim,
        hidden_dim=hidden_dim,
        attention_top_k=attention_top_k,
    )
    model.load_state_dict(dict(state_dict))
    model.eval()
    return LandmarkActivationRuntime(
        model=model,
        query_token_dim=query_token_dim,
        landmark_token_dim=landmark_token_dim,
        top_n=int(payload.get("top_n", 0)),
        split_name=split_name,
    )


def landmark_activation_score_rows(
    batch: SparseCandidateBatch,
    runtime: LandmarkActivationRuntime,
) -> list[list[float]]:
    batch.validate()
    if batch.query_descriptors is None:
        raise ValueError("landmark activation requires query_descriptors in the candidate artifact")
    if batch.candidate_landmark_descriptors is None:
        raise ValueError("landmark activation requires candidate_landmark_descriptors in the candidate artifact")
    query_tokens = torch.tensor(batch.query_descriptors, dtype=torch.float32)
    unique_ids: list[int] = []
    descriptor_rows: list[Sequence[float]] = []
    index_by_landmark: dict[int, int] = {}
    for ids, descriptors in zip(batch.candidate_landmark_ids, batch.candidate_landmark_descriptors):
        for landmark_id, descriptor in zip(ids, descriptors):
            key = int(landmark_id)
            if key in index_by_landmark:
                continue
            index_by_landmark[key] = len(unique_ids)
            unique_ids.append(key)
            descriptor_rows.append(descriptor)
    if not unique_ids:
        return [[] for _row in batch.candidate_landmark_ids]
    logits = runtime.score(query_tokens, torch.tensor(descriptor_rows, dtype=torch.float32))
    valid_rows = batch.candidate_valid_mask or []
    rows: list[list[float]] = []
    for row_idx, ids in enumerate(batch.candidate_landmark_ids):
        row_scores: list[float] = []
        for rank, landmark_id in enumerate(ids):
            valid = True if not valid_rows else bool(valid_rows[row_idx][rank])
            if not valid:
                row_scores.append(-1.0e12)
                continue
            row_scores.append(float(logits[int(index_by_landmark[int(landmark_id)])].item()))
        rows.append(row_scores)
    return rows


def _checkpoint_split_name(payload: Mapping[str, object]) -> str:
    split_audit = payload.get("split_audit")
    if isinstance(split_audit, Mapping) and split_audit.get("split_name"):
        return str(split_audit["split_name"])
    return str(payload.get("split_name") or "unknown")


def _pad_or_truncate(features: torch.Tensor, expected_dim: int) -> torch.Tensor:
    if features.dim() != 2:
        raise ValueError("query tokens must have shape [num_query_tokens, dim]")
    if int(features.shape[1]) == int(expected_dim):
        return features
    if int(features.shape[1]) < int(expected_dim):
        padding = features.new_zeros((int(features.shape[0]), int(expected_dim) - int(features.shape[1])))
        return torch.cat([features, padding], dim=-1)
    return features[:, : int(expected_dim)]
