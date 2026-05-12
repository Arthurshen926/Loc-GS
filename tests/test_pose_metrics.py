import math

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
