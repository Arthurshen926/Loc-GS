import torch

from loc_gs.localization.ulfloc_scene_match_cache import build_ulfloc_scene_matcher_pair_cache


def test_build_ulfloc_scene_matcher_pair_cache_labels_low_reprojection_topk_candidate():
    query_desc = torch.eye(2, 3, dtype=torch.float32)
    landmark_desc = torch.eye(3, dtype=torch.float32)
    top_ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    top_values = torch.tensor([[0.9, 0.8], [0.7, 0.6]], dtype=torch.float32)
    keypoint_xy = torch.tensor([[0.0, 0.0], [4.0, 4.0]], dtype=torch.float32)
    reprojection_error = torch.tensor([[8.0, 1.0], [9.0, 7.0]], dtype=torch.float32)
    visible = torch.tensor([[True, True], [True, True]])

    cache = build_ulfloc_scene_matcher_pair_cache(
        query_desc=query_desc,
        landmark_desc=landmark_desc,
        candidate_landmark_ids=top_ids,
        candidate_cosine=top_values,
        keypoint_xy=keypoint_xy,
        candidate_reprojection_error=reprojection_error,
        candidate_visible=visible,
        query_score=torch.tensor([0.5, 0.6], dtype=torch.float32),
        reprojection_threshold_px=3.0,
    )

    assert cache["query_desc"].shape == (2, 3)
    assert torch.allclose(cache["base_landmark_desc"], landmark_desc)
    assert cache["base_gaussian_id"].tolist() == [0, 1, 2]
    assert cache["landmark_id"].tolist() == top_ids.tolist()
    assert cache["landmark_desc"].shape == (2, 2, 3)
    assert cache["label"].tolist() == [1, 2]
    assert cache["candidate_mask"].tolist() == [[True, True], [True, True]]
    assert torch.allclose(cache["margin"], torch.tensor([0.1, 0.1]))
    assert cache["metadata"]["positive_query_count"] == 1


def test_build_ulfloc_scene_matcher_pair_cache_preserves_base_gaussian_ids():
    cache = build_ulfloc_scene_matcher_pair_cache(
        query_desc=torch.eye(1, 2, dtype=torch.float32),
        landmark_desc=torch.eye(3, 2, dtype=torch.float32),
        candidate_landmark_ids=torch.tensor([[2, 1]], dtype=torch.long),
        candidate_cosine=torch.tensor([[0.8, 0.7]], dtype=torch.float32),
        keypoint_xy=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        candidate_reprojection_error=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        base_gaussian_ids=torch.tensor([10, 20, 30], dtype=torch.long),
        reprojection_threshold_px=3.0,
    )

    assert cache["base_gaussian_id"].tolist() == [10, 20, 30]
    assert cache["landmark_id"].tolist() == [[2, 1]]
