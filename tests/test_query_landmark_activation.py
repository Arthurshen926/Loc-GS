import torch

from loc_gs.localization.query_landmark_activation import QueryLandmarkActivationNet, select_active_landmarks


def test_select_active_landmarks_keeps_safe_core_and_top_scores():
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8])
    safe_core = torch.tensor([0])

    selected = select_active_landmarks(scores, top_n=2, safe_core=safe_core)

    assert selected.tolist() == [0, 1, 3]


def test_select_active_landmarks_deduplicates_safe_core_overlap():
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8])
    safe_core = torch.tensor([1, 3])

    selected = select_active_landmarks(scores, top_n=2, safe_core=safe_core)

    assert selected.tolist() == [1, 3]


def test_query_landmark_activation_net_scores_each_landmark():
    model = QueryLandmarkActivationNet(query_dim=2, landmark_dim=3, hidden_dim=4)
    query = torch.tensor([0.5, -0.25], dtype=torch.float32)
    landmarks = torch.randn(5, 3)

    scores = model(query, landmarks)

    assert scores.shape == (5,)
    assert scores.dtype == torch.float32
