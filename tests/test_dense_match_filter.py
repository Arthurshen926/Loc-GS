import torch

from loc_gs.dense_support.dense_match_filter import filter_dense_matches


def test_filter_dense_matches_keeps_reliable_correspondences_and_reweights_scores():
    result = filter_dense_matches(
        query_yx=torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
        reference_yx=torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
        scores=torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
        reliability=torch.tensor([0.6, 0.2, 0.9], dtype=torch.float32),
        min_reliability=0.5,
        reweight_scores=True,
    )

    assert result["query_yx"].shape[0] == 2
    assert result["metadata"]["kept_count"] == 2
    assert result["metadata"]["dropped_count"] == 1
    assert torch.allclose(result["scores"], torch.tensor([0.54, 0.63]))


def test_filter_dense_matches_can_limit_topk_after_reliability_filter():
    result = filter_dense_matches(
        query_yx=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        reference_yx=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        scores=torch.tensor([0.1, 0.9, 0.8, 0.7], dtype=torch.float32),
        reliability=torch.ones(4, dtype=torch.float32),
        min_reliability=0.0,
        topk=2,
    )

    assert result["indices"].tolist() == [1, 2]

