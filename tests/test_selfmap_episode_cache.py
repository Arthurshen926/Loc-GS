import torch

from loc_gs.localization.selfmap_episode_cache import (
    build_gaussian_advantage_labels,
    build_selfmap_episode_cache,
    label_selfmap_pairs,
)


def test_label_selfmap_pairs_requires_geometry_visibility_and_optional_pnp_inlier():
    reprojection_error = torch.tensor([[1.0, 5.0], [2.0, 1.0]], dtype=torch.float32)
    visible = torch.tensor([[True, True], [False, True]])
    pnp_inlier = torch.tensor([[True, True], [True, False]])

    labels = label_selfmap_pairs(
        reprojection_error,
        visible=visible,
        pnp_inlier=pnp_inlier,
        reprojection_threshold_px=3.0,
    )

    assert labels.tolist() == [[True, False], [False, False]]


def test_gaussian_advantage_labels_can_suppress_high_score_false_positives():
    landmark_ids = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    pair_labels = torch.tensor([True, False, False, False, True])
    cosine = torch.tensor([0.90, 0.85, 0.95, 0.80, 0.20], dtype=torch.float32)

    stats = build_gaussian_advantage_labels(
        landmark_ids,
        pair_labels,
        num_landmarks=3,
        pair_scores=cosine,
        false_positive_score_threshold=0.5,
        false_positive_weight=1.0,
        smoothing=1.0,
    )

    target = stats["target"]
    assert stats["tp_count"].tolist() == [1.0, 0.0, 1.0]
    assert stats["fp_count"].tolist() == [1.0, 2.0, 0.0]
    assert target[1] < target[0] < target[2]
    assert target[1] < 0.5


def test_build_selfmap_episode_cache_emits_pair_listwise_and_advantage_targets():
    payload = build_selfmap_episode_cache(
        query_id=torch.tensor([4, 5], dtype=torch.long),
        keypoint_yx=torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32),
        candidate_landmark_ids=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long),
        candidate_cosine=torch.tensor([[0.9, 0.8, 0.1], [0.7, 0.6, 0.49]], dtype=torch.float32),
        candidate_reprojection_error=torch.tensor([[1.0, 8.0, 9.0], [6.0, 2.0, 7.0]], dtype=torch.float32),
        candidate_visible=torch.tensor([[True, True, True], [True, True, False]]),
        candidate_pnp_inlier=torch.tensor([[True, False, False], [False, True, False]]),
        reprojection_threshold_px=3.0,
        num_landmarks=3,
        false_positive_score_threshold=0.5,
    )

    assert payload["pair_label"].tolist() == [[True, False, False], [False, True, False]]
    assert payload["listwise_label"].tolist() == [0, 1]
    assert payload["gaussian_advantage_target"].shape == (3,)
    assert payload["gaussian_advantage_target"][1] < payload["gaussian_advantage_target"][0]
    assert payload["metadata"]["format"] == "selfmap_episode_v1"
