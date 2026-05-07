import torch
import pytest

from utils.selective_reconstruction import (
    summarize_locability_error,
    summarize_locability_selection,
)


def test_summarize_locability_selection_reports_score_mass_and_weight_ratio():
    scores = torch.tensor([0.9, 0.8, 0.1, 0.0])

    rows = summarize_locability_selection(
        scores,
        top_ratios=[0.25],
        min_weight=0.05,
        gamma=2.0,
    )

    row = rows[0]
    assert row["selected_points"] == 1
    assert row["selected_fraction"] == 0.25
    assert row["score_mass"] == pytest.approx(0.5)
    assert row["score_mass_gain"] == pytest.approx(2.0)
    assert row["selected_score_mean"] > row["background_score_mean"]
    assert row["selected_to_background_weight_ratio"] > 10.0


def test_summarize_locability_selection_keeps_at_least_one_point():
    rows = summarize_locability_selection(
        torch.tensor([0.2, 0.1]),
        top_ratios=[0.01],
        min_weight=0.05,
        gamma=1.0,
    )

    assert rows[0]["selected_points"] == 1


def test_summarize_locability_error_reports_selected_error():
    locability = torch.tensor([0.9, 0.8, 0.1, 0.0])
    error = torch.tensor([0.1, 0.2, 0.8, 1.0])

    rows = summarize_locability_error(error, locability, top_ratios=[0.5])

    row = rows[0]
    assert row["selected_points"] == 2
    assert row["score_mass"] == pytest.approx(1.7 / 1.8)
    assert row["selected_error_mean"] == pytest.approx(0.15)
    assert row["background_error_mean"] == pytest.approx(0.9)
    assert row["selected_to_background_error_ratio"] < 1.0


def test_summarize_locability_error_respects_mask():
    locability = torch.tensor([0.9, 0.8, 0.1, 0.0])
    error = torch.tensor([0.1, 0.2, 0.8, 1.0])
    mask = torch.tensor([False, True, True, False])

    rows = summarize_locability_error(error, locability, top_ratios=[0.5], mask=mask)

    assert rows[0]["selected_points"] == 1
    assert rows[0]["selected_error_mean"] == pytest.approx(0.2)
