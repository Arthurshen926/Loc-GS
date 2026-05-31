import math

import pytest

from loc_gs.localization.pose_metrics import pose_error_summary, pose_recall_metrics, recall_metric_key


def test_pose_recall_metrics_cover_submit_table_thresholds():
    metrics = pose_recall_metrics(
        [1.0, 7.0, 30.0, 70.0, 150.0, 600.0],
        [1.0, 4.0, 1.0, 4.0, 4.0, 11.0],
    )

    assert metrics["recall_2cm_2d"] == 1 / 6
    assert metrics["recall_5cm_5d"] == 1 / 6
    assert metrics["recall_10cm_5d"] == 2 / 6
    assert metrics["recall_25cm_2d"] == 1 / 6
    assert metrics["recall_50cm_5d"] == 3 / 6
    assert metrics["recall_1m_5d"] == 4 / 6
    assert metrics["recall_2m_5d"] == 5 / 6
    assert metrics["recall_5m_10d"] == 5 / 6


def test_pose_error_summary_handles_empty_and_aliases():
    empty = pose_error_summary([], [], [])

    assert math.isinf(empty["median_te"])
    assert empty["recall_5cm_5d"] == 0.0
    assert empty["avg_inliers"] == 0.0
    assert recall_metric_key("r5") == "recall_5cm_5d"
    assert recall_metric_key("recall_25cm_2d") == "recall_25cm_2d"


def test_pose_error_summary_reports_tail_and_failure_rates():
    metrics = pose_error_summary([1.0, 10.0, 20.0, 100.0, 600.0], [1.0] * 5, [10, 20, 30, 40, 50])

    assert metrics["p90_te_cm"] == pytest.approx(400.0)
    assert metrics["p95_te_cm"] == pytest.approx(500.0)
    assert metrics["cvar10_te_cm"] == 600.0
    assert metrics["severe_rate_1m"] == 0.2
    assert metrics["catastrophic_rate_5m"] == 0.2
