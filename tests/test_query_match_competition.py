import pytest
import torch

from loc_gs.stdloc_native.query_match_competition import compute_query_match_competition


def test_query_match_competition_scores_non_support_top1_that_steals_from_support():
    query_features = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    sampled_idx = torch.tensor([10, 20, 30], dtype=torch.long)
    landmark_features = torch.tensor(
        [
            [0.96, 0.28],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    result = compute_query_match_competition(
        query_features=query_features,
        sampled_idx=sampled_idx,
        landmark_features=landmark_features,
        query_support={10: 2.0},
        top_k=2,
        score_margin=0.05,
    )

    assert result.metadata["query_feature_count"] == 1
    assert result.metadata["support_present_count"] == 1
    assert result.metadata["support_top1_count"] == 0
    assert result.metadata["support_topk_count"] == 1
    assert result.metadata["unique_support_topk_count"] == 1
    assert result.metadata["support_rank_p50"] == pytest.approx(2.0)
    assert result.stealer_scores[20].item() > 0.0
    assert result.support_match_strength[10].item() > 0.0
    assert result.support_best_rank.tolist() == [2]


def test_query_match_competition_does_not_penalize_when_support_is_top1():
    query_features = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    sampled_idx = torch.tensor([10, 20], dtype=torch.long)
    landmark_features = torch.tensor([[1.0, 0.0], [0.99, 0.1]], dtype=torch.float32)

    result = compute_query_match_competition(
        query_features=query_features,
        sampled_idx=sampled_idx,
        landmark_features=landmark_features,
        query_support={10: 1.0},
        top_k=2,
        score_margin=0.05,
    )

    assert result.metadata["support_top1_count"] == 1
    assert result.metadata["stealer_count"] == 0
    assert result.stealer_scores.sum().item() == pytest.approx(0.0)


def test_query_match_competition_reports_feature_weak_support_when_support_is_not_near_topk():
    query_features = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    sampled_idx = torch.tensor([10, 20, 30], dtype=torch.long)
    landmark_features = torch.tensor(
        [
            [0.1, 0.99],
            [1.0, 0.0],
            [0.99, 0.1],
        ],
        dtype=torch.float32,
    )

    result = compute_query_match_competition(
        query_features=query_features,
        sampled_idx=sampled_idx,
        landmark_features=landmark_features,
        query_support={10: 1.0},
        top_k=2,
        score_margin=0.05,
    )

    assert result.metadata["support_topk_count"] == 0
    assert result.metadata["feature_weak_query_feature_count"] == 1
    assert result.metadata["stealer_count"] == 0
    assert result.support_match_strength.sum().item() == pytest.approx(0.0)


def test_query_match_competition_accumulates_support_strength_when_support_is_near_topk():
    query_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    sampled_idx = torch.tensor([10, 20, 30], dtype=torch.long)
    landmark_features = torch.tensor(
        [
            [0.98, 0.2],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    result = compute_query_match_competition(
        query_features=query_features,
        sampled_idx=sampled_idx,
        landmark_features=landmark_features,
        query_support={10: 2.0, 30: 3.0},
        top_k=2,
        score_margin=0.05,
    )

    assert result.support_match_strength[10].item() > 0.0
    assert result.support_match_strength[30].item() > result.support_match_strength[10].item()
    assert result.metadata["support_match_strength_nonzero_count"] == 2
