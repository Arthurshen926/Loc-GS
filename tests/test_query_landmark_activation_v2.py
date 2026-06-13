from __future__ import annotations

import pytest
import torch

from loc_gs.training.query_landmark_activation_v2 import (
    QueryLandmarkActivationV2,
    build_landmark_activation_v2_landmark_tokens,
    query_landmark_activation_v2_loss,
)


def test_query_landmark_activation_v2_scores_landmarks_from_query_and_landmark_tokens():
    model = QueryLandmarkActivationV2(
        query_token_dim=6,
        landmark_token_dim=8,
        hidden_dim=12,
        attention_top_k=3,
    )
    query_tokens = torch.randn(5, 6)
    landmark_tokens = torch.randn(7, 8)

    logits = model(query_tokens=query_tokens, landmark_tokens=landmark_tokens)

    assert logits.shape == (7,)
    assert logits.dtype == torch.float32


def test_query_landmark_activation_v2_rejects_mean_query_descriptor_only():
    model = QueryLandmarkActivationV2(query_token_dim=6, landmark_token_dim=8, hidden_dim=12)
    mean_query_descriptor = torch.randn(6)
    landmark_tokens = torch.randn(7, 8)

    with pytest.raises(ValueError, match="query_tokens"):
        model(query_tokens=mean_query_descriptor, landmark_tokens=landmark_tokens)


def test_query_landmark_activation_v2_loss_has_solver_feedback_terms():
    logits = torch.tensor([2.0, -1.0, 0.0], dtype=torch.float32)
    labels = {
        "geometry_visible": torch.tensor([1.0, 0.0, 1.0]),
        "solver_positive": torch.tensor([1.0, 0.0, 0.0]),
        "harmful_negative": torch.tensor([0.0, 1.0, 0.0]),
        "protected_support": torch.tensor([1.0, 0.0, 0.0]),
    }

    terms = query_landmark_activation_v2_loss(logits, labels)

    assert set(terms) == {
        "geometry_visibility_loss",
        "solver_positive_loss",
        "harmful_negative_suppression_loss",
        "protected_retention_loss",
        "coverage_regularizer",
        "total_loss",
    }
    assert terms["harmful_negative_suppression_loss"].item() > 0.0
    assert terms["total_loss"].item() >= terms["solver_positive_loss"].item()


def test_landmark_activation_v2_landmark_tokens_include_xyz_and_prior():
    descriptors = torch.eye(2, 4, dtype=torch.float32)
    xyz = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=torch.float32)
    prior = torch.tensor([0.25, 0.75], dtype=torch.float32)

    tokens = build_landmark_activation_v2_landmark_tokens(
        descriptors,
        landmark_xyz=xyz,
        landmark_prior=prior,
        expected_dim=8,
    )

    assert tokens.shape == (2, 8)
    assert torch.allclose(tokens[:, :4], torch.nn.functional.normalize(descriptors, dim=-1))
    assert torch.allclose(tokens[:, -1], prior)
    assert not torch.allclose(tokens[:, 4:7], torch.zeros_like(tokens[:, 4:7]))
