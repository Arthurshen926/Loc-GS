from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_LANDMARK_TOKEN_COMPONENTS = ("descriptor_l2", "xyz_scene_normalized", "prior_score")


def _pad_or_truncate_features(features: torch.Tensor, expected_dim: int | None) -> torch.Tensor:
    if expected_dim is None or int(expected_dim) <= 0:
        return features
    expected = int(expected_dim)
    if int(features.shape[1]) < expected:
        return torch.cat(
            [features, features.new_zeros((int(features.shape[0]), expected - int(features.shape[1])))],
            dim=-1,
        )
    if int(features.shape[1]) > expected:
        return features[:, :expected]
    return features


def _scene_normalize_xyz(xyz: torch.Tensor) -> torch.Tensor:
    if int(xyz.numel()) == 0:
        return xyz
    centered = xyz - xyz.mean(dim=0, keepdim=True)
    scale = centered.pow(2).mean().sqrt().clamp_min(1e-6)
    return centered / scale


def build_landmark_activation_v2_landmark_tokens(
    landmark_descriptors: torch.Tensor,
    *,
    landmark_xyz: torch.Tensor | None = None,
    landmark_prior: torch.Tensor | None = None,
    expected_dim: int | None = None,
) -> torch.Tensor:
    """Build the v2 activation landmark token used by both cache builders and ULF runtime.

    Token contract: l2-normalized descriptor + scene-normalized xyz + scalar prior.
    Missing xyz/prior values are explicit zeros, not silently dropped.
    """
    desc = torch.as_tensor(landmark_descriptors, dtype=torch.float32)
    if desc.dim() != 2:
        raise ValueError("landmark_descriptors must have shape [num_landmarks, descriptor_dim]")
    desc = F.normalize(desc, dim=-1)
    count = int(desc.shape[0])
    device = desc.device
    dtype = desc.dtype
    if landmark_xyz is None:
        xyz_token = desc.new_zeros((count, 3))
    else:
        xyz = torch.as_tensor(landmark_xyz, dtype=dtype, device=device)
        if xyz.dim() != 2 or int(xyz.shape[0]) != count or int(xyz.shape[1]) != 3:
            raise ValueError("landmark_xyz must have shape [num_landmarks, 3]")
        xyz_token = _scene_normalize_xyz(xyz)
    if landmark_prior is None:
        prior = desc.new_zeros((count, 1))
    else:
        prior = torch.as_tensor(landmark_prior, dtype=dtype, device=device).reshape(-1, 1)
        if int(prior.shape[0]) != count:
            raise ValueError("landmark_prior must have one value per landmark")
    return _pad_or_truncate_features(torch.cat([desc, xyz_token, prior], dim=-1), expected_dim)


class QueryLandmarkActivationV2(nn.Module):
    """Token-based query-conditioned landmark activation model."""

    def __init__(
        self,
        *,
        query_token_dim: int,
        landmark_token_dim: int,
        hidden_dim: int = 128,
        attention_top_k: int = 64,
    ) -> None:
        super().__init__()
        self.query_token_dim = int(query_token_dim)
        self.landmark_token_dim = int(landmark_token_dim)
        self.hidden_dim = int(hidden_dim)
        self.attention_top_k = int(attention_top_k)
        self.query_encoder = nn.Linear(self.query_token_dim, self.hidden_dim)
        self.landmark_encoder = nn.Linear(self.landmark_token_dim, self.hidden_dim)
        self.landmark_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, *, query_tokens: torch.Tensor, landmark_tokens: torch.Tensor) -> torch.Tensor:
        query = torch.as_tensor(query_tokens, dtype=self.query_encoder.weight.dtype, device=self.query_encoder.weight.device)
        landmarks = torch.as_tensor(
            landmark_tokens,
            dtype=self.landmark_encoder.weight.dtype,
            device=self.landmark_encoder.weight.device,
        )
        if query.dim() != 2:
            raise ValueError("query_tokens must have shape [num_query_tokens, query_token_dim]")
        if landmarks.dim() != 2:
            raise ValueError("landmark_tokens must have shape [num_landmarks, landmark_token_dim]")
        if query.shape[1] != self.query_token_dim:
            raise ValueError("query_tokens feature dimension mismatch")
        if landmarks.shape[1] != self.landmark_token_dim:
            raise ValueError("landmark_tokens feature dimension mismatch")
        if query.shape[0] == 0 or landmarks.shape[0] == 0:
            return landmarks.new_empty((int(landmarks.shape[0]),))

        q = F.normalize(self.query_encoder(query), p=2, dim=-1)
        l = F.normalize(self.landmark_encoder(landmarks), p=2, dim=-1)
        sim = l @ q.transpose(0, 1)
        top_k = min(max(1, int(self.attention_top_k)), int(q.shape[0]))
        top_values, top_indices = torch.topk(sim, k=top_k, dim=1)
        weights = F.softmax(top_values, dim=1)
        gathered = q[top_indices]
        context = (gathered * weights.unsqueeze(-1)).sum(dim=1)
        hidden = torch.cat([l, context, l * context], dim=-1)
        return self.landmark_head(hidden).reshape(-1)


def _target(labels: Mapping[str, Any], key: str, like: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(labels.get(key, torch.zeros_like(like)), dtype=torch.float32, device=like.device).reshape(-1)
    if value.shape != like.shape:
        raise ValueError(f"{key} labels must match logits shape")
    return value.clamp(0.0, 1.0)


def query_landmark_activation_v2_loss(
    logits: torch.Tensor,
    labels: Mapping[str, Any],
    *,
    geometry_weight: float = 1.0,
    solver_positive_weight: float = 1.0,
    harmful_negative_weight: float = 1.0,
    protected_weight: float = 1.0,
    coverage_weight: float = 0.01,
) -> dict[str, torch.Tensor]:
    score = torch.as_tensor(logits, dtype=torch.float32).reshape(-1)
    geometry = _target(labels, "geometry_visible", score)
    solver_positive = _target(labels, "solver_positive", score)
    harmful_negative = _target(labels, "harmful_negative", score)
    protected = _target(labels, "protected_support", score)
    geometry_visibility_loss = F.binary_cross_entropy_with_logits(score, geometry) * float(geometry_weight)
    solver_positive_loss = (
        F.binary_cross_entropy_with_logits(score, torch.ones_like(score), reduction="none") * solver_positive
    ).mean() * float(solver_positive_weight)
    harmful_negative_suppression_loss = (
        F.binary_cross_entropy_with_logits(score, torch.zeros_like(score), reduction="none") * harmful_negative
    ).mean() * float(harmful_negative_weight)
    protected_retention_loss = (
        F.binary_cross_entropy_with_logits(score, torch.ones_like(score), reduction="none") * protected
    ).mean() * float(protected_weight)
    coverage_regularizer = torch.relu(0.05 - torch.sigmoid(score).mean()) * float(coverage_weight)
    total = (
        geometry_visibility_loss
        + solver_positive_loss
        + harmful_negative_suppression_loss
        + protected_retention_loss
        + coverage_regularizer
    )
    return {
        "geometry_visibility_loss": geometry_visibility_loss,
        "solver_positive_loss": solver_positive_loss,
        "harmful_negative_suppression_loss": harmful_negative_suppression_loss,
        "protected_retention_loss": protected_retention_loss,
        "coverage_regularizer": coverage_regularizer,
        "total_loss": total,
    }
