from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from loc_gs.training.ulfloc_aggregation_samples import AggregationSample


@dataclass(frozen=True)
class AggregationResult:
    descriptor: torch.Tensor
    metadata: dict[str, Any]


class LandmarkAggregationModel(nn.Module):
    """Small view-logit model for aggregation-stage view selection."""

    def __init__(self, descriptor_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.view_score = nn.Sequential(
            nn.Linear(int(descriptor_dim) + 4, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, descriptors: torch.Tensor, label_features: torch.Tensor) -> torch.Tensor:
        if descriptors.dim() != 2:
            raise ValueError("descriptors must have shape [views, descriptor_dim]")
        if label_features.dim() != 2 or label_features.shape[0] != descriptors.shape[0]:
            raise ValueError("label_features must have shape [views, 4]")
        return self.view_score(torch.cat([descriptors, label_features], dim=-1)).squeeze(-1)


def _normalized(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.as_tensor(value, dtype=torch.float32).reshape(-1), p=2, dim=0)


def _label_features(samples: list[AggregationSample]) -> torch.Tensor:
    rows: list[list[float]] = []
    for sample in samples:
        rows.append(
            [
                1.0 if sample.target > 0 else 0.0,
                1.0 if sample.target == 0 else 0.0,
                1.0 if sample.protected else 0.0,
                float(sample.reliability),
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


def _candidate_descriptor(
    native: torch.Tensor,
    samples: list[AggregationSample],
    logits: torch.Tensor,
) -> torch.Tensor:
    descriptors = torch.stack([sample.descriptor for sample in samples], dim=0).to(dtype=torch.float32)
    positives = torch.tensor([sample.target > 0 for sample in samples], dtype=torch.bool)
    if bool(positives.any()):
        pos_logits = logits[positives]
        weights = F.softmax(pos_logits, dim=0).reshape(-1, 1)
        selected = descriptors[positives]
        return F.normalize((selected * weights).sum(dim=0), p=2, dim=0)
    return native


def optimize_landmark_descriptor(
    *,
    native_descriptor: torch.Tensor,
    samples: list[AggregationSample],
    steps: int = 40,
    lr: float = 0.1,
    min_native_cosine: float = 0.95,
    hidden_dim: int = 16,
    negative_margin: float = 0.1,
    anchor_weight: float = 0.2,
) -> AggregationResult:
    """Optimize aggregation-stage view logits for one landmark descriptor."""

    native = _normalized(native_descriptor)
    if not samples:
        return AggregationResult(
            descriptor=native,
            metadata={
                "descriptor_mode": "ulfloc_solver_feedback_aggregation_v1",
                "sample_count": 0,
                "positive_sample_count": 0,
                "negative_sample_count": 0,
                "fallback_reason": "no_samples",
                "native_cosine": 1.0,
            },
        )

    positive_count = sum(1 for sample in samples if sample.target > 0)
    negative_count = sum(1 for sample in samples if sample.target == 0)
    negative_label_roles = sorted({str(sample.label_role) for sample in samples if sample.target == 0})
    if positive_count == 0:
        return AggregationResult(
            descriptor=native,
            metadata={
                "descriptor_mode": "ulfloc_solver_feedback_aggregation_v1",
                "sample_count": int(len(samples)),
                "positive_sample_count": 0,
                "negative_sample_count": int(negative_count),
                "negative_label_roles": negative_label_roles,
                "fallback_reason": "no_positive_samples",
                "native_cosine": 1.0,
            },
        )

    descriptors = torch.stack([sample.descriptor for sample in samples], dim=0).to(dtype=torch.float32)
    label_features = _label_features(samples)
    positives = torch.tensor([sample.target > 0 for sample in samples], dtype=torch.bool)
    negatives = ~positives
    model = LandmarkAggregationModel(int(native.numel()), hidden_dim=int(hidden_dim))
    optimizer = torch.optim.Adam(model.parameters(), lr=max(0.0, float(lr)))
    steps_i = max(1, int(steps))
    margin = float(negative_margin)
    anchor = max(0.0, float(anchor_weight))

    for _ in range(steps_i):
        optimizer.zero_grad()
        logits = model(descriptors, label_features)
        candidate = _candidate_descriptor(native, samples, logits)
        pos_cos = (descriptors[positives] @ candidate).mean()
        loss = -pos_cos
        if bool(negatives.any()):
            neg_cos = descriptors[negatives] @ candidate
            loss = loss + F.relu(neg_cos - margin).mean()
        loss = loss + anchor * (1.0 - torch.dot(candidate, native))
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(descriptors, label_features)
        candidate = _candidate_descriptor(native, samples, logits)
        native_cosine = float(torch.dot(candidate, native).clamp(-1.0, 1.0).item())
        floor = max(-1.0, min(1.0, float(min_native_cosine)))
        fallback_reason = ""
        if native_cosine < floor:
            candidate = native
            native_cosine = 1.0
            fallback_reason = "native_cosine_floor"

    return AggregationResult(
        descriptor=F.normalize(candidate.detach().cpu(), p=2, dim=0),
        metadata={
            "descriptor_mode": "ulfloc_solver_feedback_aggregation_v1",
            "sample_count": int(len(samples)),
            "positive_sample_count": int(positive_count),
            "negative_sample_count": int(negative_count),
            "negative_label_roles": negative_label_roles,
            "steps": int(steps_i),
            "lr": float(lr),
            "min_native_cosine": float(min_native_cosine),
            "native_cosine": float(native_cosine),
            "fallback_reason": fallback_reason,
        },
    )
