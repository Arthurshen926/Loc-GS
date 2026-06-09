import pytest
import torch

from loc_gs.diagnostics.ulfloc_detector_target_audit import (
    compare_detector_target_maps,
    sparse_result_error_map,
    summarize_detector_target_entry,
)


def test_summarize_detector_target_entry_reports_weight_and_spatial_coverage():
    entry = {
        "keypoint_yx": torch.tensor([[10.0, 10.0], [70.0, 80.0]], dtype=torch.float32),
        "support_weights": torch.tensor([1.0, 3.0], dtype=torch.float32),
        "negative_keypoint_yx": torch.tensor([[20.0, 20.0]], dtype=torch.float32),
        "negative_weights": torch.tensor([0.5], dtype=torch.float32),
    }

    summary = summarize_detector_target_entry(entry, height=100, width=100, grid_size=4)

    assert summary["positive_count"] == 2
    assert summary["negative_count"] == 1
    assert summary["weight_mean"] == pytest.approx(2.0)
    assert summary["weight_max"] == pytest.approx(3.0)
    assert summary["coverage_cell_count"] == 2
    assert summary["coverage_fraction"] == pytest.approx(2.0 / 16.0)
    assert summary["centroid_y"] == pytest.approx(40.0)
    assert summary["centroid_x"] == pytest.approx(45.0)


def test_compare_detector_target_maps_adds_target_and_error_deltas():
    baseline = {
        "q0.png": {
            "keypoint_yx": torch.tensor([[10.0, 10.0], [70.0, 80.0]], dtype=torch.float32),
            "support_weights": torch.tensor([1.0, 3.0], dtype=torch.float32),
        },
        "q1.png": {
            "keypoint_yx": torch.tensor([[20.0, 20.0]], dtype=torch.float32),
            "support_weights": torch.tensor([2.0], dtype=torch.float32),
        },
    }
    candidate = {
        "q0.png": {
            "keypoint_yx": torch.tensor([[12.0, 12.0]], dtype=torch.float32),
            "support_weights": torch.tensor([4.0], dtype=torch.float32),
            "negative_keypoint_yx": torch.tensor([[90.0, 90.0]], dtype=torch.float32),
            "negative_weights": torch.tensor([1.0], dtype=torch.float32),
        }
    }

    rows, aggregate = compare_detector_target_maps(
        baseline,
        candidate,
        height=100,
        width=100,
        baseline_errors={"q0.png": 2.0, "q1.png": 5.0},
        candidate_errors={"q0.png": 3.5},
    )

    q0 = rows[0]
    assert q0["query_id"] == "q0.png"
    assert q0["candidate_positive_count"] == 1
    assert q0["positive_count_delta"] == -1
    assert q0["candidate_negative_count"] == 1
    assert q0["sparse_te_delta_cm"] == pytest.approx(1.5)
    assert aggregate["common_query_count"] == 1
    assert aggregate["candidate_missing_query_count"] == 1
    assert aggregate["mean_sparse_te_delta_cm"] == pytest.approx(1.5)


def test_sparse_result_error_map_reads_ulfloc_sparse_result_rows():
    payload = {
        "rows": [
            {"image_name": "q0.png", "sparse_te_cm": 2.5},
            {"image_name": "q1.png", "sparse_te_cm": 7.0},
        ]
    }

    errors = sparse_result_error_map(payload)

    assert errors == {"q0.png": 2.5, "q1.png": 7.0}
