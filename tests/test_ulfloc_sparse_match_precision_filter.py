import pytest
import torch

from loc_gs.localization.ulfloc_sparse_match_filter import (
    SparseMatchFilterConfig,
    filter_ulfloc_sparse_matches,
)


def test_sparse_match_filter_removes_low_margin_correspondence_before_pnp():
    corr = torch.tensor(
        [
            [0.90, 0.89, 0.10],
            [0.82, 0.10, 0.05],
            [0.05, 0.80, 0.10],
            [0.10, 0.20, 0.78],
        ],
        dtype=torch.float32,
    )
    im_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    gs_ids = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    scores = corr[im_idx, gs_ids]

    result = filter_ulfloc_sparse_matches(
        corr_matrix=corr,
        im_idx=im_idx,
        gs_ids=gs_ids,
        descriptor_scores=scores,
        detector_scores=torch.ones(4),
        config=SparseMatchFilterConfig(
            enabled=True,
            min_query_margin=0.05,
            min_keep=3,
            unique_landmark=True,
        ),
    )

    assert result.im_idx.tolist() == [1, 2, 3]
    assert result.gs_ids.tolist() == [0, 1, 2]
    assert result.metadata["input_match_count"] == 4
    assert result.metadata["output_match_count"] == 3
    assert result.metadata["removed_low_margin_count"] == 1


def test_sparse_match_filter_preserves_min_keep_with_best_scored_matches():
    corr = torch.tensor(
        [
            [0.90, 0.89, 0.88],
            [0.70, 0.69, 0.68],
            [0.60, 0.59, 0.58],
        ],
        dtype=torch.float32,
    )
    im_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    gs_ids = torch.tensor([0, 0, 0], dtype=torch.long)
    scores = corr[im_idx, gs_ids]

    result = filter_ulfloc_sparse_matches(
        corr_matrix=corr,
        im_idx=im_idx,
        gs_ids=gs_ids,
        descriptor_scores=scores,
        detector_scores=torch.tensor([0.1, 1.0, 0.2]),
        config=SparseMatchFilterConfig(
            enabled=True,
            min_query_margin=0.20,
            min_keep=2,
            unique_landmark=False,
            detector_score_weight=0.1,
        ),
    )

    assert result.im_idx.numel() == 2
    assert set(result.im_idx.tolist()) == {0, 1}
    assert result.metadata["min_keep_backfill_count"] == 2


def test_sparse_match_filter_rejects_unsupported_mode():
    with pytest.raises(ValueError, match="unsupported"):
        SparseMatchFilterConfig(enabled=True, mode="oracle")


def test_sparse_match_filter_reorder_mode_preserves_all_matches_for_prosac_ordering():
    corr = torch.tensor(
        [
            [0.60, 0.59],
            [0.80, 0.10],
            [0.70, 0.69],
        ],
        dtype=torch.float32,
    )
    im_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    gs_ids = torch.tensor([0, 0, 0], dtype=torch.long)
    scores = corr[im_idx, gs_ids]

    result = filter_ulfloc_sparse_matches(
        corr_matrix=corr,
        im_idx=im_idx,
        gs_ids=gs_ids,
        descriptor_scores=scores,
        config=SparseMatchFilterConfig(
            enabled=True,
            mode="reorder",
            min_query_margin=0.50,
        ),
    )

    assert result.im_idx.tolist() == [1, 2, 0]
    assert result.metadata["output_match_count"] == 3
    assert result.metadata["reordered_for_prosac"] is True


def test_sparse_match_filter_caps_per_image_cell_to_preserve_pnp_coverage():
    corr = torch.eye(6, dtype=torch.float32)
    im_idx = torch.arange(6, dtype=torch.long)
    gs_ids = torch.arange(6, dtype=torch.long)
    scores = torch.tensor([0.99, 0.98, 0.97, 0.96, 0.70, 0.69], dtype=torch.float32)
    keypoint_xy = torch.tensor(
        [
            [5.0, 5.0],
            [8.0, 8.0],
            [10.0, 9.0],
            [12.0, 7.0],
            [80.0, 80.0],
            [85.0, 82.0],
        ],
        dtype=torch.float32,
    )

    result = filter_ulfloc_sparse_matches(
        corr_matrix=corr,
        im_idx=im_idx,
        gs_ids=gs_ids,
        descriptor_scores=scores,
        keypoint_xy=keypoint_xy,
        image_size=(100, 100),
        config=SparseMatchFilterConfig(
            enabled=True,
            min_keep=0,
            image_grid_size=2,
            max_per_image_cell=2,
        ),
    )

    assert result.im_idx.tolist() == [0, 1, 4, 5]
    assert result.metadata["image_cell_cap_removed_count"] == 2
