import pytest
import torch

from loc_gs.diagnostics.ulfloc_scene_detector_audit import (
    keypoint_overlap_fraction,
    sample_heatmap_at_yx,
    summarize_detector_score_groups,
)


def test_sample_heatmap_at_yx_uses_nearest_in_bounds_pixels():
    heatmap = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    points = torch.tensor([[1.2, 2.7], [0.0, 0.0], [5.0, 5.0]], dtype=torch.float32)

    sampled = sample_heatmap_at_yx(heatmap, points)

    assert sampled.tolist() == [6.0, 0.0]


def test_summarize_detector_score_groups_reports_group_means():
    heatmap = torch.zeros(4, 4)
    heatmap[1, 1] = 0.8
    heatmap[2, 2] = 0.2

    summary = summarize_detector_score_groups(
        heatmap,
        {
            "positive": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            "negative": torch.tensor([[2.0, 2.0]], dtype=torch.float32),
            "empty": torch.empty(0, 2),
        },
    )

    assert summary["positive_count"] == 1
    assert summary["positive_mean"] == pytest.approx(0.8)
    assert summary["negative_mean"] == pytest.approx(0.2)
    assert summary["empty_count"] == 0
    assert summary["empty_mean"] is None
    assert summary["global_mean"] == pytest.approx(float(heatmap.mean().item()))


def test_keypoint_overlap_fraction_uses_rounded_pixel_sets():
    native = torch.tensor([[1.2, 2.4], [5.0, 7.0], [8.0, 9.0]], dtype=torch.float32)
    fused = torch.tensor([[1.0, 2.0], [8.4, 9.2]], dtype=torch.float32)

    overlap = keypoint_overlap_fraction(native, fused)

    assert overlap["native_count"] == 3
    assert overlap["fused_count"] == 2
    assert overlap["intersection_count"] == 2
    assert overlap["native_overlap_fraction"] == pytest.approx(2 / 3)
