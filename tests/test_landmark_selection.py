import torch

from loc_gs.losses.landmark_selection import (
    descriptor_ambiguity_loss,
    key_gaussian_isotropy_loss,
    locability_budget_loss,
    splatloc_saliency_prior,
)


def test_locability_budget_loss_penalizes_budget_mismatch():
    locability = torch.tensor([0.9, 0.8, 0.1, 0.0])

    low_budget = locability_budget_loss(locability, target_count=1)
    high_budget = locability_budget_loss(locability, target_count=3)

    assert low_budget != high_budget
    assert torch.isfinite(low_budget)


def test_descriptor_ambiguity_loss_penalizes_similar_high_weight_landmarks():
    desc = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    high = torch.tensor([1.0, 1.0, 0.0])
    low = torch.tensor([1.0, 0.0, 1.0])

    assert descriptor_ambiguity_loss(desc, high) > descriptor_ambiguity_loss(desc, low)


def test_splatloc_saliency_prior_prefers_well_observed_points():
    points = torch.tensor([[0.0, 0.0, 2.0], [10.0, 0.0, 2.0]], dtype=torch.float32)
    poses = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    poses[1, 0, 3] = -0.2
    K = torch.tensor([[4.0, 0.0, 1.0], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]])

    prior = splatloc_saliency_prior(points, poses, K, height=4, width=4)

    assert prior.shape == (2,)
    assert prior[0] > prior[1]
    assert 0.0 <= prior.min() <= prior.max() <= 1.0


def test_key_gaussian_isotropy_loss_weights_high_locability():
    scales = torch.tensor([[1.0, 1.0, 1.0], [3.0, 0.5, 0.5]])
    high_bad = torch.tensor([0.0, 1.0])
    low_bad = torch.tensor([1.0, 0.0])

    assert key_gaussian_isotropy_loss(scales, high_bad) > key_gaussian_isotropy_loss(scales, low_bad)
