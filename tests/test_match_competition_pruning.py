import torch

from loc_gs.stdloc_native.match_competition_pruning import prune_match_competition


def test_prune_match_competition_removes_high_risk_nonprotected_selected_landmarks():
    sampled_idx = torch.tensor([10, 20, 30], dtype=torch.long)
    features = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    risk_scores = torch.zeros(31, dtype=torch.float32)
    risk_scores[20] = 5.0

    result = prune_match_competition(
        sampled_idx=sampled_idx,
        features=features,
        risk_scores=risk_scores,
        min_risk_score=1.0,
    )

    assert result.pruned_sampled_idx.tolist() == [10, 30]
    assert result.pruned_features.tolist() == [[0.0, 1.0], [4.0, 5.0]]
    assert result.metadata["match_competition_pruned_count"] == 1


def test_prune_match_competition_keeps_solver_protected_risky_landmark():
    sampled_idx = torch.tensor([10, 20, 30], dtype=torch.long)
    features = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    risk_scores = torch.zeros(31, dtype=torch.float32)
    risk_scores[20] = 5.0
    risk_scores[30] = 4.0
    protect_scores = torch.zeros(31, dtype=torch.float32)
    protect_scores[20] = 2.0

    result = prune_match_competition(
        sampled_idx=sampled_idx,
        features=features,
        risk_scores=risk_scores,
        protect_scores=protect_scores,
        min_protect_score=1.0,
        min_risk_score=1.0,
    )

    assert result.pruned_sampled_idx.tolist() == [10, 20]
    assert result.metadata["solver_protected_risky_count"] == 1
    assert result.metadata["match_competition_pruned_count"] == 1


def test_prune_match_competition_respects_min_keep_count():
    sampled_idx = torch.tensor([10, 20, 30], dtype=torch.long)
    features = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    risk_scores = torch.zeros(31, dtype=torch.float32)
    risk_scores[10] = 5.0
    risk_scores[20] = 4.0
    risk_scores[30] = 3.0

    result = prune_match_competition(
        sampled_idx=sampled_idx,
        features=features,
        risk_scores=risk_scores,
        min_risk_score=1.0,
        min_keep_count=2,
    )

    assert result.pruned_sampled_idx.tolist() == [20, 30]
    assert result.metadata["match_competition_pruned_count"] == 1
