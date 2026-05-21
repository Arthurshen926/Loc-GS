import torch

from loc_gs.diagnostics.alpha_composition import (
    alpha_composition_reliability,
    composition_entropy,
    composition_support_aggregation,
)


def test_alpha_composition_reliability_prefers_dominant_low_entropy_pixels():
    weights = torch.tensor(
        [
            [0.95, 0.05, 0.00],
            [0.34, 0.33, 0.33],
        ],
        dtype=torch.float32,
    )

    reliability = alpha_composition_reliability(weights)

    assert reliability["dominance"][0] > reliability["dominance"][1]
    assert reliability["entropy"][0] < reliability["entropy"][1]
    assert reliability["reliability"][0] > reliability["reliability"][1]


def test_composition_entropy_is_zero_for_single_contributor():
    entropy = composition_entropy(torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32))

    assert torch.allclose(entropy, torch.zeros(1))


def test_composition_support_aggregation_weights_gaussian_support_by_contribution():
    contributor_ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    weights = torch.tensor([[0.75, 0.25], [0.5, 0.5]], dtype=torch.float32)
    support = torch.tensor([1.0, 0.0, 0.5], dtype=torch.float32)

    agg = composition_support_aggregation(contributor_ids, weights, support)

    assert torch.allclose(agg, torch.tensor([0.75, 0.25]))

