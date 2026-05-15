import torch
from torch import nn

from loc_gs.localization.scene_matcher import (
    SceneMatchListwiseNet,
    SceneMatchNet,
    best_pair_per_query_indices,
    build_scene_match_listwise_extra_features,
    build_scene_match_listwise_features,
    build_scene_match_pair_features,
    label_scene_match_topk_candidates,
    label_scene_match_pairs,
    load_scene_matcher,
    score_scene_match_candidates,
    select_scene_match_listwise_candidates,
    scene_match_listwise_feature_dim,
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


def test_scene_match_listwise_features_keep_candidate_axis_and_scalars():
    query = torch.eye(2, 3, dtype=torch.float32)
    landmark = torch.stack([query, torch.roll(query, shifts=1, dims=0)], dim=1)
    cosine = torch.tensor([[0.9, 0.1], [0.8, 0.2]], dtype=torch.float32)
    margin = torch.tensor([0.3, 0.4], dtype=torch.float32)
    prior = torch.tensor([[0.7, 0.2], [0.5, 0.1]], dtype=torch.float32)
    query_score = torch.tensor([0.6, 0.9], dtype=torch.float32)

    features = build_scene_match_listwise_features(
        query,
        landmark,
        cosine=cosine,
        margin=margin,
        landmark_prior=prior,
        extra_scalar_features=query_score,
    )

    assert features.shape == (2, 2, scene_match_listwise_feature_dim(descriptor_dim=3, scalar_dim=5))
    assert torch.allclose(features[:, :, -5], cosine)
    assert torch.allclose(features[:, :, -4], margin[:, None].expand_as(cosine))
    assert torch.allclose(features[:, :, -3], prior)
    assert torch.allclose(features[:, :, -1], query_score[:, None].expand_as(cosine))


def test_scene_match_listwise_rank_gap_extra_features_are_deterministic():
    cosine = torch.tensor(
        [
            [0.9, 0.5, 0.7],
            [0.1, 0.3, -1.0],
        ],
        dtype=torch.float32,
    )
    query_score = torch.tensor([0.6, 0.2], dtype=torch.float32)
    candidate_mask = torch.tensor(
        [
            [True, True, True],
            [True, True, False],
        ]
    )

    extra = build_scene_match_listwise_extra_features(
        "query_score_rank_gap",
        query_score=query_score,
        cosine=cosine,
        candidate_mask=candidate_mask,
    )

    assert extra.shape == (2, 3, 3)
    assert torch.allclose(extra[:, :, 0], query_score[:, None].expand_as(cosine))
    assert torch.allclose(extra[0, :, 1], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(extra[0, :, 2], torch.tensor([0.0, 0.4, 0.2]))
    assert torch.allclose(extra[1, :, 2], torch.tensor([0.2, 0.0, 0.0]))


def test_scene_match_listwise_net_can_use_rank_gap_extra_features(tmp_path):
    matcher = SceneMatchListwiseNet(
        descriptor_dim=4,
        hidden_dim=8,
        num_layers=2,
        listwise_extra_features="query_score_rank_gap",
    )
    query = torch.randn(3, 4)
    landmark = torch.randn(3, 5, 4)
    cosine = torch.randn(3, 5)
    query_score = torch.rand(3)
    candidate_mask = torch.ones(3, 5, dtype=torch.bool)

    logits = score_scene_match_candidates(
        matcher,
        query,
        landmark,
        cosine=cosine,
        query_score=query_score,
        candidate_mask=candidate_mask,
    )

    assert matcher.config["scalar_dim"] == 7
    assert logits.shape == (3, 6)

    ckpt = tmp_path / "scene_matcher_listwise_rank_gap.pt"
    torch.save({"config": matcher.config, "state_dict": matcher.state_dict()}, ckpt)
    loaded = load_scene_matcher(ckpt)
    reloaded = score_scene_match_candidates(
        loaded,
        query,
        landmark,
        cosine=cosine,
        query_score=query_score,
        candidate_mask=candidate_mask,
    )

    assert torch.allclose(reloaded, logits)


def test_scene_match_listwise_net_scores_candidates_and_dustbin(tmp_path):
    matcher = SceneMatchListwiseNet(descriptor_dim=4, hidden_dim=8, num_layers=2)
    query = torch.randn(3, 4)
    landmark = torch.randn(3, 5, 4)
    candidate_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, False, False, False],
            [False, False, False, False, False],
        ]
    )

    logits = score_scene_match_candidates(
        matcher,
        query,
        landmark,
        candidate_mask=candidate_mask,
    )

    assert logits.shape == (3, 6)
    assert torch.isneginf(logits[1, 2:5]).all()
    assert torch.isfinite(logits[:, -1]).all()

    ckpt = tmp_path / "scene_matcher_listwise.pt"
    torch.save({"config": matcher.config, "state_dict": matcher.state_dict()}, ckpt)
    loaded = load_scene_matcher(ckpt)
    reloaded = score_scene_match_candidates(loaded, query, landmark, candidate_mask=candidate_mask)

    assert torch.allclose(reloaded, logits)


def test_select_scene_match_listwise_candidates_can_drop_to_dustbin():
    class FixedListwiseMatcher(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {"model_type": "listwise"}

        def forward(self, query_desc, landmark_desc, **kwargs):
            del query_desc, landmark_desc, kwargs
            return torch.tensor(
                [
                    [0.1, 0.9, 0.0],
                    [0.2, 0.1, 0.5],
                    [0.3, float("-inf"), 0.1],
                ],
                dtype=torch.float32,
            )

    q_ids = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    lm_ids = torch.tensor([4, 5, 6, 7, 8], dtype=torch.long)

    keep, logits = select_scene_match_listwise_candidates(
        FixedListwiseMatcher(),
        torch.randn(3, 4),
        torch.randn(9, 4),
        q_ids,
        lm_ids,
        cosine=torch.rand(5),
    )

    assert keep.tolist() == [1, 4]
    assert torch.allclose(logits, torch.tensor([0.9, 0.3]))


def test_select_scene_match_listwise_candidates_can_score_instead_of_drop_dustbin():
    class FixedListwiseMatcher(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {"model_type": "listwise"}

        def forward(self, query_desc, landmark_desc, **kwargs):
            del query_desc, landmark_desc, kwargs
            return torch.tensor(
                [
                    [0.1, 0.9, 0.0],
                    [0.2, 0.1, 0.5],
                ],
                dtype=torch.float32,
            )

    keep, logits = select_scene_match_listwise_candidates(
        FixedListwiseMatcher(),
        torch.randn(2, 4),
        torch.randn(4, 4),
        torch.tensor([0, 0, 1, 1], dtype=torch.long),
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        drop_dustbin=False,
    )

    assert keep.tolist() == [1, 2]
    assert torch.allclose(logits, torch.tensor([0.9, -0.3]))


def test_best_pair_per_query_indices_keeps_highest_scoring_candidate():
    q_ids = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
    scores = torch.tensor([0.2, 0.8, 0.1, 0.5, 0.4], dtype=torch.float32)

    keep = best_pair_per_query_indices(q_ids, scores, num_queries=3)

    assert keep.tolist() == [1, 3, 2]


def test_label_scene_match_topk_candidates_uses_best_positive_or_dustbin():
    landmark_xyz = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    query_yx = torch.tensor([[0.0, 0.0], [4.0, 4.0]], dtype=torch.float32)
    lm_topk = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)

    labels, candidate_mask, errors = label_scene_match_topk_candidates(
        query_yx=query_yx,
        landmark_xyz=landmark_xyz,
        lm_topk=lm_topk,
        pose_w2c=torch.eye(4),
        K=torch.eye(3),
        reprojection_threshold_px=0.25,
    )

    assert labels.tolist() == [1, 2]
    assert candidate_mask.tolist() == [[True, True], [True, True]]
    assert errors.shape == (2, 2)


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
