import torch
import torch.nn.functional as F

from loc_gs.localization.stdloc_parity import (
    apply_match_prior,
    coarse_to_fine_dense_matches,
    dual_softmax,
    match_correlation_matrix,
    mnn_match,
    soft_argmax_offsets,
)


def test_dual_softmax_suppresses_many_to_one_ambiguity():
    corr = torch.tensor([[[3.0, 1.0], [3.0, 0.5]]])
    out = dual_softmax(corr, temp=1.0)

    assert out.shape == corr.shape
    assert out[0, 0, 0] > out[0, 0, 1]
    assert out[0, 0, 0] < torch.softmax(corr, dim=-1)[0, 0, 0]


def test_mnn_match_keeps_only_mutual_best_above_threshold():
    corr = torch.tensor([[[0.9, 0.2, 0.1], [0.8, 0.7, 0.1]]])
    b_ids, q_ids, r_ids, scores = mnn_match(corr, threshold=0.5)

    assert b_ids.tolist() == [0]
    assert q_ids.tolist() == [0]
    assert r_ids.tolist() == [0]
    assert torch.allclose(scores, torch.tensor([0.9]))


def test_match_correlation_matrix_can_use_dual_softmax_mnn_and_prior():
    corr = torch.tensor([[[0.7, 0.7], [0.1, 0.8]]])
    adjusted = apply_match_prior(corr.clone(), torch.tensor([0.0, 1.0]), weight=0.2)
    b_ids, q_ids, r_ids, _scores = match_correlation_matrix(
        adjusted,
        threshold=0.0,
        dual_softmax_temp=0.1,
        use_dual_softmax=True,
        use_mnn=True,
        topk=1,
    )

    assert b_ids.tolist() == [0]
    assert q_ids.tolist() == [1]
    assert r_ids.tolist() == [1]


def test_match_correlation_matrix_filters_ambiguous_top1_by_second_best_margin():
    corr = torch.tensor([[[0.90, 0.89, 0.10], [0.80, 0.20, 0.10]]])

    _b_ids, q_ids, r_ids, scores = match_correlation_matrix(
        corr,
        threshold=0.0,
        topk=1,
        second_best_margin=0.05,
    )

    assert q_ids.tolist() == [1]
    assert r_ids.tolist() == [0]
    assert torch.allclose(scores, torch.tensor([0.80]))


def test_soft_argmax_offsets_recovers_window_expectation():
    scores = torch.full((1, 4), -4.0)
    scores[0, 3] = 4.0
    offsets = soft_argmax_offsets(scores, window_size=2, temperature=0.1)

    assert offsets.shape == (1, 2)
    assert torch.allclose(offsets[0], torch.tensor([1.0, 1.0]), atol=1e-3)


def test_coarse_to_fine_dense_matches_returns_subpixel_positions():
    query_fine = torch.zeros(2, 4, 4)
    rendered_fine = torch.zeros(2, 4, 4)
    query_fine[:, 2, 2] = torch.tensor([1.0, 0.0])
    rendered_fine[:, 2, 3] = torch.tensor([1.0, 0.0])
    query_fine[:, 0, 0] = torch.tensor([0.0, 1.0])
    rendered_fine[:, 0, 0] = torch.tensor([0.0, 1.0])
    query_fine = F.normalize(query_fine, dim=0)
    rendered_fine = F.normalize(rendered_fine, dim=0)

    result = coarse_to_fine_dense_matches(
        query_fine,
        rendered_fine,
        window_size=2,
        coarse_dual_softmax_temp=0.1,
        fine_dual_softmax_temp=0.1,
        coarse_threshold=0.0,
        fine_threshold=0.0,
        use_mnn=True,
        subpixel_refine=True,
        subpixel_temperature=0.1,
    )

    assert result.query_yx.shape[0] >= 1
    assert result.rendered_yx.shape == result.query_yx.shape
    assert torch.any(torch.linalg.norm(result.rendered_yx - torch.tensor([2.0, 3.0]), dim=-1) < 0.2)


def test_coarse_to_fine_dense_matches_crops_non_divisible_feature_maps():
    query_fine = torch.zeros(2, 5, 4)
    rendered_fine = torch.zeros(2, 5, 4)
    query_fine[:, 2, 2] = torch.tensor([1.0, 0.0])
    rendered_fine[:, 2, 3] = torch.tensor([1.0, 0.0])
    query_fine = F.normalize(query_fine, dim=0)
    rendered_fine = F.normalize(rendered_fine, dim=0)

    result = coarse_to_fine_dense_matches(
        query_fine,
        rendered_fine,
        window_size=2,
        coarse_threshold=0.0,
        fine_threshold=0.0,
    )

    assert result.query_yx.shape[0] > 0
    assert result.query_yx[:, 0].max() < 4


def test_coarse_to_fine_dense_matches_applies_fine_rendered_prior():
    query_fine = F.normalize(torch.ones(2, 2, 2), dim=0)
    rendered_fine = F.normalize(torch.ones(2, 2, 2), dim=0)
    rendered_prior = torch.tensor([[0.0, 0.0], [0.0, 1.0]])

    result = coarse_to_fine_dense_matches(
        query_fine,
        rendered_fine,
        rendered_prior=rendered_prior,
        prior_weight=10.0,
        window_size=2,
        coarse_threshold=0.0,
        fine_threshold=0.0,
        subpixel_refine=False,
    )

    assert result.query_yx.shape[0] > 0
    assert torch.all(result.rendered_yx == torch.tensor([1.0, 1.0]))


def test_coarse_to_fine_dense_matches_handles_multiple_coarse_cells_without_broadcasting():
    query_fine = F.normalize(torch.ones(2, 4, 8), dim=0)
    rendered_fine = F.normalize(torch.ones(2, 4, 8), dim=0)

    result = coarse_to_fine_dense_matches(
        query_fine,
        rendered_fine,
        window_size=2,
        coarse_threshold=0.0,
        fine_threshold=0.0,
        use_mnn=False,
        subpixel_refine=True,
        subpixel_temperature=0.1,
    )

    assert result.query_yx.shape[0] > 0
    assert result.query_yx.ndim == 2
    assert result.rendered_yx.shape == result.query_yx.shape
