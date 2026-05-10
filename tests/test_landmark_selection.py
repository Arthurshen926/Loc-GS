import torch

from loc_gs.losses.landmark_selection import (
    depth_consistency_score,
    descriptor_ambiguity_loss,
    descriptor_local_distinctiveness,
    gaussian_geometry_score,
    geometric_mean_score,
    key_gaussian_isotropy_loss,
    locability_budget_loss,
    normalize_score01,
    spatially_balanced_topk,
    superpoint_detector_saliency,
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


def test_geometric_mean_score_keeps_low_components_low():
    components = {
        "descriptor": torch.tensor([1.0, 1.0, 0.1]),
        "geometry": torch.tensor([1.0, 0.1, 1.0]),
    }

    score = geometric_mean_score(components, {"descriptor": 1.0, "geometry": 1.0})

    assert score[0] > score[1]
    assert score[0] > score[2]
    assert torch.all((score >= 0.0) & (score <= 1.0))


def test_normalize_score01_treats_constant_valid_component_as_neutral():
    score = torch.tensor([0.5, 0.5, 0.5])
    valid = torch.tensor([True, True, False])

    normalized = normalize_score01(score, valid=valid)

    assert torch.equal(normalized, torch.tensor([1.0, 1.0, 0.0]))


def test_geometric_mean_score_does_not_collapse_on_constant_component():
    components = {
        "constant": torch.tensor([0.2, 0.2, 0.2]),
        "ranking": torch.tensor([0.1, 0.5, 1.0]),
    }

    score = geometric_mean_score(components, {"constant": 1.0, "ranking": 1.0})

    assert score[2] > score[1] > score[0]
    assert score.max() > 0.5


def test_superpoint_detector_saliency_uses_non_dustbin_probability():
    logits = torch.zeros(65, 2, 2)
    logits[64] = 5.0
    low = superpoint_detector_saliency(logits)
    logits[0] = 6.0
    high = superpoint_detector_saliency(logits)

    assert high[0, 0] > low[0, 0]


def test_depth_consistency_prefers_locally_flat_depth():
    depth = torch.ones(5, 5)
    depth[:, 3:] = 4.0

    score = depth_consistency_score(depth, window_size=3)

    assert score[2, 0] > score[2, 3]


def test_descriptor_distinctiveness_and_geometry_scores_are_bounded():
    desc = torch.zeros(2, 3, 3)
    desc[0] = 1.0
    desc[:, 1, 1] = torch.tensor([0.0, 1.0])
    distinct = descriptor_local_distinctiveness(desc)
    geom = gaussian_geometry_score(
        torch.tensor([[1.0, 1.0, 1.0], [4.0, 0.5, 0.5]]),
        torch.tensor([1.0, 0.5]),
    )

    assert distinct[1, 1] > distinct[0, 0]
    assert torch.all((geom >= 0.0) & (geom <= 1.0))


def test_spatially_balanced_topk_spreads_before_filling_by_score():
    score = torch.tensor([1.0, 0.99, 0.7, 0.69], dtype=torch.float32)
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.1, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    selected = spatially_balanced_topk(score, xyz, k=2, grid_size=2)

    assert selected.tolist() == [0, 2]
