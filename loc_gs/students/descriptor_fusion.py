from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from loc_gs.sparse.correspondences import SparseCandidateBatch
from loc_gs.sparse.audit import reject_test_split


@dataclass(frozen=True)
class DescriptorFusionConfig:
    trust_region: float = 0.25
    protected_support_gain: float = 2.0
    positive_inlier_gain: float = 1.0
    hard_negative_penalty: float = 1.0
    score_scale: float = 1.0


@dataclass(frozen=True)
class DescriptorFusionModel:
    fused_descriptors: dict[str, list[float]]
    landmark_scores: dict[str, float]
    negative_scores: dict[str, float]
    score_scale: float = 1.0

    def score_landmark(self, landmark_id: int | str) -> float:
        key = str(landmark_id)
        positive = float(self.landmark_scores.get(key, 0.0))
        negative = float(self.negative_scores.get(key, 0.0))
        return float((positive - negative) * float(self.score_scale))

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": "internal_descriptor_fusion_v1",
            "fused_descriptors": {
                str(key): [float(value) for value in values]
                for key, values in sorted(self.fused_descriptors.items())
            },
            "landmark_scores": {str(key): float(value) for key, value in sorted(self.landmark_scores.items())},
            "negative_scores": {str(key): float(value) for key, value in sorted(self.negative_scores.items())},
            "score_scale": float(self.score_scale),
        }

    def to_torch_dict(self) -> dict[str, object]:
        keys = sorted(
            self.fused_descriptors,
            key=lambda value: (0, int(value)) if value.lstrip("-").isdigit() else (1, value),
        )
        descriptor_dim = len(self.fused_descriptors[keys[0]]) if keys else 0
        fused = (
            torch.tensor([self.fused_descriptors[key] for key in keys], dtype=torch.float32)
            if keys
            else torch.zeros((0, descriptor_dim), dtype=torch.float32)
        )
        return {
            "schema_version": "internal_descriptor_fusion_v1",
            "landmark_ids": keys,
            "fused_descriptors": fused,
            "landmark_scores": torch.tensor([float(self.landmark_scores.get(key, 0.0)) for key in keys], dtype=torch.float32),
            "negative_scores": torch.tensor([float(self.negative_scores.get(key, 0.0)) for key in keys], dtype=torch.float32),
            "score_scale": float(self.score_scale),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> "DescriptorFusionModel":
        if payload.get("schema_version") != "internal_descriptor_fusion_v1":
            raise ValueError(f"unsupported descriptor fusion schema: {payload.get('schema_version')}")
        return cls(
            fused_descriptors={
                str(key): [float(item) for item in value]
                for key, value in dict(payload.get("fused_descriptors", {})).items()
            },
            landmark_scores={str(key): float(value) for key, value in dict(payload.get("landmark_scores", {})).items()},
            negative_scores={str(key): float(value) for key, value in dict(payload.get("negative_scores", {})).items()},
            score_scale=float(payload.get("score_scale", 1.0)),
        )

    @classmethod
    def from_torch_dict(cls, payload: Mapping[str, object]) -> "DescriptorFusionModel":
        if payload.get("schema_version") != "internal_descriptor_fusion_v1":
            raise ValueError(f"unsupported descriptor fusion schema: {payload.get('schema_version')}")
        keys = [str(key) for key in payload.get("landmark_ids", [])]
        fused = _array(payload.get("fused_descriptors"), name="fused_descriptors")
        positive = _array(payload.get("landmark_scores"), name="landmark_scores")
        negative = _array(payload.get("negative_scores"), name="negative_scores")
        if fused.ndim != 2:
            raise ValueError("descriptor fusion fused_descriptors must have shape [N,D]")
        if len(keys) != fused.shape[0] or positive.shape[0] != fused.shape[0] or negative.shape[0] != fused.shape[0]:
            raise ValueError("descriptor fusion torch fields must share the same landmark count")
        return cls(
            fused_descriptors={key: [float(item) for item in fused[idx].tolist()] for idx, key in enumerate(keys)},
            landmark_scores={key: float(positive[idx]) for idx, key in enumerate(keys) if float(positive[idx]) != 0.0},
            negative_scores={key: float(negative[idx]) for idx, key in enumerate(keys) if float(negative[idx]) != 0.0},
            score_scale=float(payload.get("score_scale", 1.0)),
        )


def train_descriptor_fusion_from_payload(
    payload: Mapping[str, Any],
    cfg: DescriptorFusionConfig | None = None,
) -> tuple[DescriptorFusionModel, dict[str, object]]:
    if cfg is None:
        cfg = DescriptorFusionConfig()
    meta = payload.get("metadata", {})
    if isinstance(meta, Mapping):
        reject_test_split(str(meta.get("source_split_name") or meta.get("split_name") or "unknown"), purpose="descriptor fusion")
    query_desc = _array(payload.get("query_desc"), name="query_desc")
    landmark_desc = _array(payload.get("landmark_desc"), name="landmark_desc")
    landmark_ids = _array(payload.get("landmark_id"), name="landmark_id").astype(np.int64)
    labels = _array(payload.get("label"), name="label").astype(np.int64)
    masks = _array(payload.get("candidate_mask", np.ones(landmark_ids.shape, dtype=bool)), name="candidate_mask").astype(bool)
    solver_weight = _array(payload.get("solver_weight", np.ones(landmark_ids.shape, dtype=np.float32)), name="solver_weight").astype(float)
    roles = payload.get("label_roles")
    if roles is None:
        roles = [["" for _rank in range(landmark_ids.shape[1])] for _row in range(landmark_ids.shape[0])]
    if query_desc.ndim != 2 or landmark_desc.ndim != 3:
        raise ValueError("query_desc must be [N,D] and landmark_desc must be [N,K,D]")
    if landmark_ids.shape != masks.shape or landmark_ids.shape != solver_weight.shape:
        raise ValueError("landmark_id, candidate_mask, and solver_weight must have matching [N,K] shape")
    if query_desc.shape[0] != landmark_ids.shape[0] or landmark_desc.shape[:2] != landmark_ids.shape:
        raise ValueError("descriptor payload row/top-k dimensions do not match landmark_id")

    positive_vectors: dict[str, list[np.ndarray]] = {}
    base_vectors: dict[str, list[np.ndarray]] = {}
    landmark_scores: dict[str, float] = {}
    negative_scores: dict[str, float] = {}
    protected_support_count = 0
    positive_inlier_count = 0
    hard_negative_count = 0
    for row_idx in range(landmark_ids.shape[0]):
        label = int(labels[row_idx])
        for rank in range(landmark_ids.shape[1]):
            if not bool(masks[row_idx, rank]):
                continue
            key = str(int(landmark_ids[row_idx, rank]))
            weight = max(1.0e-6, float(solver_weight[row_idx, rank]))
            role = str(roles[row_idx][rank]) if row_idx < len(roles) and rank < len(roles[row_idx]) else ""
            geometric = 0 <= label < landmark_ids.shape[1] and rank == label
            base_vectors.setdefault(key, []).append(np.asarray(landmark_desc[row_idx, rank], dtype=np.float64))
            if role == "protected_support" or geometric:
                positive_vectors.setdefault(key, []).append(np.asarray(query_desc[row_idx], dtype=np.float64) * weight)
                gain = float(cfg.protected_support_gain if role == "protected_support" else cfg.positive_inlier_gain)
                landmark_scores[key] = landmark_scores.get(key, 0.0) + gain * weight
                if role == "protected_support":
                    protected_support_count += 1
                else:
                    positive_inlier_count += 1
            elif role == "hard_negative":
                negative_scores[key] = negative_scores.get(key, 0.0) + float(cfg.hard_negative_penalty) * weight
                hard_negative_count += 1

    trust = max(0.0, min(1.0, float(cfg.trust_region)))
    fused: dict[str, list[float]] = {}
    for key in sorted(base_vectors):
        base = _normalize(np.mean(np.stack(base_vectors[key], axis=0), axis=0))
        if key in positive_vectors:
            positive = _normalize(np.mean(np.stack(positive_vectors[key], axis=0), axis=0))
            value = _normalize((1.0 - trust) * base + trust * positive)
        else:
            value = base
        fused[key] = [float(item) for item in value.tolist()]
    model = DescriptorFusionModel(
        fused_descriptors=fused,
        landmark_scores=landmark_scores,
        negative_scores=negative_scores,
        score_scale=float(cfg.score_scale),
    )
    summary = {
        "schema_version": "internal_descriptor_fusion_training_summary_v1",
        "student_modules": ["descriptor_fusion"],
        "landmark_count": int(len(fused)),
        "protected_support_count": int(protected_support_count),
        "positive_inlier_count": int(positive_inlier_count),
        "hard_negative_count": int(hard_negative_count),
        "hyperparameters": asdict(cfg),
    }
    return model, summary


def descriptor_fusion_score_rows(batch: SparseCandidateBatch, model: DescriptorFusionModel) -> list[list[float]]:
    batch.validate()
    rows: list[list[float]] = []
    valid_rows = batch.candidate_valid_mask or []
    query_descriptors = batch.query_descriptors
    needed_landmark_ids = {str(int(landmark_id)) for row in batch.candidate_landmark_ids for landmark_id in row}
    fused_cache = {
        str(key): _normalize(np.asarray(value, dtype=np.float64).reshape(-1))
        for key, value in model.fused_descriptors.items()
        if str(key) in needed_landmark_ids
    }
    normalized_queries = (
        [_normalize(np.asarray(desc, dtype=np.float64).reshape(-1)) for desc in query_descriptors]
        if query_descriptors is not None
        else None
    )
    for row_idx, landmark_ids in enumerate(batch.candidate_landmark_ids):
        row: list[float] = []
        query_desc = None if normalized_queries is None else normalized_queries[row_idx]
        for rank, landmark_id in enumerate(landmark_ids):
            valid = True if not valid_rows else bool(valid_rows[row_idx][rank])
            if not valid:
                row.append(-1.0e12)
                continue
            prior_score = model.score_landmark(int(landmark_id))
            descriptor_score = _descriptor_score(query_desc, fused_cache.get(str(int(landmark_id))))
            row.append(float(prior_score + descriptor_score))
        rows.append(row)
    return rows


def load_descriptor_fusion(path: str | Path) -> DescriptorFusionModel:
    source = Path(path)
    if source.suffix.lower() == ".pt":
        payload = torch.load(source, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError(f"descriptor fusion torch payload must contain an object: {path}")
        return DescriptorFusionModel.from_torch_dict(payload)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"descriptor fusion JSON must contain an object: {path}")
    return DescriptorFusionModel.from_json_dict(payload)


def _array(value: Any, *, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"descriptor fusion payload is missing required field: {name}")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        return np.asarray(value, dtype=np.float64)
    return np.asarray(value, dtype=np.float64) / norm


def _descriptor_score(query_desc: np.ndarray | None, fused_desc: np.ndarray | None) -> float:
    if query_desc is None or fused_desc is None:
        return 0.0
    if query_desc.shape != fused_desc.shape:
        return 0.0
    return float(query_desc @ fused_desc)
