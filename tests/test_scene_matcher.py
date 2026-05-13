import torch

from loc_gs.localization.scene_matcher import (
    SceneMatchNet,
    best_pair_per_query_indices,
    build_scene_match_pair_features,
    label_scene_match_pairs,
    load_scene_matcher,
    scene_match_feature_dim,
)


def test_scene_match_pair_features_keep_fixed_scalar_channels():
    query = torch.eye(3, dtype=torch.float32)
    landmark = torch.roll(query, shifts=1, dims=0)
    margin = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    prior = torch.tensor([0.9, 0.5, 0.1], dtype=torch.float32)

    features = build_scene_match_pair_features(
        query,
        landmark,
        margin=margin,
        landmark_prior=prior,
    )

    assert features.shape == (3, scene_match_feature_dim(descriptor_dim=3))
    assert torch.allclose(features[:, -3], margin)
    assert torch.allclose(features[:, -2], prior)
    assert torch.allclose(features[:, -1], torch.zeros(3))


def test_scene_match_pair_features_can_include_query_score_extra_scalar():
    query = torch.eye(3, dtype=torch.float32)
    landmark = query.clone()
    query_score = torch.tensor([0.2, 0.5, 0.9], dtype=torch.float32)

    features = build_scene_match_pair_features(
        query,
        landmark,
        extra_scalar_features=query_score,
    )

    assert features.shape == (3, scene_match_feature_dim(descriptor_dim=3, scalar_dim=5))
    assert torch.allclose(features[:, -1], query_score)


def test_scene_match_net_scores_pairs_and_can_reload_checkpoint(tmp_path):
    matcher = SceneMatchNet(descriptor_dim=4, hidden_dim=8, num_layers=2)
    query = torch.randn(5, 4)
    landmark = torch.randn(5, 4)

    logits = matcher(query, landmark)

    assert logits.shape == (5,)
    ckpt = tmp_path / "scene_matcher.pt"
    torch.save({"config": matcher.config, "state_dict": matcher.state_dict()}, ckpt)
    loaded = load_scene_matcher(ckpt)

    assert torch.allclose(loaded(query, landmark), logits)


def test_best_pair_per_query_indices_keeps_highest_scoring_candidate():
    q_ids = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
    scores = torch.tensor([0.2, 0.8, 0.1, 0.5, 0.4], dtype=torch.float32)

    keep = best_pair_per_query_indices(q_ids, scores, num_queries=3)

    assert keep.tolist() == [1, 3, 2]


def test_label_scene_match_pairs_uses_geometry_and_visibility():
    landmark_xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [5.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    query_yx = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    q_ids = torch.tensor([0, 0, 1], dtype=torch.long)
    lm_ids = torch.tensor([0, 2, 1], dtype=torch.long)
    depth = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
    alpha = torch.ones_like(depth)

    labels = label_scene_match_pairs(
        query_yx=query_yx,
        landmark_xyz=landmark_xyz,
        q_ids=q_ids,
        lm_ids=lm_ids,
        pose_w2c=torch.eye(4),
        K=torch.eye(3),
        reprojection_threshold_px=0.25,
        depth_map=depth,
        alpha_map=alpha,
        depth_abs_tolerance=0.1,
        depth_rel_tolerance=0.0,
        alpha_threshold=0.5,
    )

    assert labels.tolist() == [True, False, True]
