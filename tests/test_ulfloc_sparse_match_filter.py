import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from loc_gs.localization.ulfloc_sparse_match_filter import (
    SparseMatchFilterConfig,
    adjust_corr_matrix_with_landmark_prior,
    filter_ulfloc_sparse_matches,
    prepare_selected_landmark_prior_weights,
    select_scene_matcher_topk_matches,
)


def test_ulfloc_deduplicate_landmark_matches_keeps_best_query_per_landmark():
    ulf_root = Path("/root/ULF-Loc")
    if str(ulf_root) not in sys.path:
        sys.path.insert(0, str(ulf_root))

    from utils.loc_utils import deduplicate_landmark_matches  # type: ignore

    im_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    gs_ids = torch.tensor([5, 5, 6, 6], dtype=torch.long)
    scores = torch.tensor([0.1, 0.9, 0.8, 0.7], dtype=torch.float32)

    out_im, out_gs, out_scores = deduplicate_landmark_matches(im_idx, gs_ids, scores)

    assert out_im.tolist() == [1, 2]
    assert out_gs.tolist() == [5, 6]
    assert torch.allclose(out_scores, torch.tensor([0.9, 0.8]))


def test_sparse_match_filter_reranks_with_solver_landmark_prior():
    corr = torch.tensor(
        [
            [0.70, 0.69, 0.10],
            [0.68, 0.67, 0.10],
        ],
        dtype=torch.float32,
    )
    im_idx = torch.tensor([0, 1], dtype=torch.long)
    gs_ids = torch.tensor([0, 1], dtype=torch.long)
    descriptor_scores = corr[im_idx, gs_ids]
    landmark_prior = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    filtered = filter_ulfloc_sparse_matches(
        corr_matrix=corr,
        im_idx=im_idx,
        gs_ids=gs_ids,
        descriptor_scores=descriptor_scores,
        config=SparseMatchFilterConfig(
            enabled=True,
            mode="reorder",
            margin_weight=0.0,
            landmark_prior_weights=landmark_prior,
            landmark_prior_weight=1.0,
        ),
    )

    assert filtered.gs_ids.tolist() == [1, 0]
    assert filtered.metadata["landmark_prior_weight"] == 1.0
    assert filtered.combined_scores[1] > filtered.combined_scores[0]


def test_sparse_match_filter_maps_full_gaussian_prior_into_sampled_space():
    full_prior = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    sampled_idx = torch.tensor([2, 0], dtype=torch.long)

    selected = prepare_selected_landmark_prior_weights(
        full_prior,
        sampled_idx=sampled_idx,
        selected_count=2,
    )

    assert selected is not None
    assert selected.tolist() == pytest.approx([0.3, 0.1])


def test_sparse_match_filter_adjusts_corr_matrix_before_topk():
    corr = torch.tensor([[0.70, 0.69]], dtype=torch.float32)
    selected_prior = torch.tensor([1.0, 1.5], dtype=torch.float32)

    adjusted = adjust_corr_matrix_with_landmark_prior(
        corr,
        selected_prior,
        weight=0.05,
        center=1.0,
        scale=0.5,
        clip=1.0,
    )

    assert int(adjusted.argmax(dim=1).item()) == 1


def test_scene_matcher_topk_selection_can_choose_solver_preferred_candidate():
    class FixedListwiseMatcher(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {"model_type": "listwise"}

        def forward(self, query_desc, landmark_desc, **kwargs):
            del query_desc, landmark_desc, kwargs
            return torch.tensor([[0.0, 2.0, -1.0]], dtype=torch.float32)

    corr = torch.tensor([[0.90, 0.89, 0.10]], dtype=torch.float32)
    query_desc = torch.eye(1, 3, dtype=torch.float32)
    landmark_desc = torch.eye(3, dtype=torch.float32)

    result = select_scene_matcher_topk_matches(
        corr_matrix=corr,
        query_desc=query_desc,
        landmark_desc=landmark_desc,
        detector_scores=torch.tensor([0.5], dtype=torch.float32),
        matcher=FixedListwiseMatcher(),
        topk=2,
        score_weight=0.25,
        threshold=-1.0,
    )

    assert result.gs_ids.tolist() == [1]
    assert result.im_idx.tolist() == [0]
    assert result.metadata["scene_matcher_topk"] == 2
    assert result.metadata["scene_matcher_selected_count"] == 1


def test_scene_matcher_topk_selection_can_ignore_dustbin_for_candidate_rerank():
    class DustbinBiasedMatcher(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {"model_type": "listwise"}

        def forward(self, query_desc, landmark_desc, **kwargs):
            del kwargs
            batch, topk = landmark_desc.shape[:2]
            logits = torch.zeros(batch, topk + 1, dtype=query_desc.dtype, device=query_desc.device)
            logits[:, 0] = 0.0
            logits[:, 1] = 4.0
            logits[:, topk] = 10.0
            return logits

    result = select_scene_matcher_topk_matches(
        corr_matrix=torch.tensor([[0.80, 0.79]], dtype=torch.float32),
        query_desc=torch.randn(1, 4),
        landmark_desc=torch.randn(2, 4),
        detector_scores=None,
        matcher=DustbinBiasedMatcher(),
        topk=2,
        score_weight=1.0,
        scene_score_mode="candidate",
    )

    assert result.gs_ids.tolist() == [1]
    assert result.metadata["scene_matcher_score_mode"] == "candidate"


def test_scene_matcher_drop_dustbin_can_backfill_min_keep_for_pnp_solvability():
    class DustbinRejectingMatcher(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {"model_type": "listwise"}

        def forward(self, query_desc, landmark_desc, **kwargs):
            del kwargs
            batch, topk = landmark_desc.shape[:2]
            logits = torch.zeros(batch, topk + 1, dtype=query_desc.dtype, device=query_desc.device)
            logits[:, 0] = 0.0
            logits[:, 1] = 0.5
            logits[:, topk] = 10.0
            return logits

    result = select_scene_matcher_topk_matches(
        corr_matrix=torch.tensor(
            [
                [0.90, 0.89],
                [0.80, 0.79],
                [0.70, 0.69],
            ],
            dtype=torch.float32,
        ),
        query_desc=torch.randn(3, 4),
        landmark_desc=torch.randn(2, 4),
        detector_scores=None,
        matcher=DustbinRejectingMatcher(),
        topk=2,
        score_weight=0.1,
        scene_score_mode="candidate_minus_dustbin",
        drop_dustbin=True,
        min_keep=2,
    )

    assert result.im_idx.tolist() == [0, 1]
    assert result.metadata["scene_matcher_drop_dustbin"] is True
    assert result.metadata["scene_matcher_min_keep"] == 2
    assert result.metadata["scene_matcher_min_keep_backfill_count"] == 2


def test_scene_matcher_topk_selection_respects_descriptor_margin_guard():
    class FixedListwiseMatcher(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {"model_type": "listwise"}

        def forward(self, query_desc, landmark_desc, **kwargs):
            del kwargs
            batch, topk = landmark_desc.shape[:2]
            logits = torch.zeros(batch, topk + 1, dtype=query_desc.dtype, device=query_desc.device)
            logits[:, 1] = 4.0
            return logits

    result = select_scene_matcher_topk_matches(
        corr_matrix=torch.tensor([[0.95, 0.50]], dtype=torch.float32),
        query_desc=torch.randn(1, 4),
        landmark_desc=torch.randn(2, 4),
        detector_scores=None,
        matcher=FixedListwiseMatcher(),
        topk=2,
        score_weight=1.0,
        scene_score_mode="candidate",
        max_descriptor_margin=0.1,
    )

    assert result.gs_ids.tolist() == [0]
    assert result.metadata["scene_matcher_margin_guard_count"] == 1
