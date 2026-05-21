from loc_gs.eval.locgs_metrics import (
    dense_transition_summary,
    lsf_pose_stage_summary,
    summarize_lsf_eval,
)


def test_lsf_pose_stage_summary_reports_sparse_and_dense_recall_thresholds():
    rows = [
        {
            "query_id": "a",
            "sparse_te_cm": 20.0,
            "sparse_re_deg": 2.0,
            "sparse_inliers": 10,
            "dense_te_cm": 4.0,
            "dense_re_deg": 1.0,
            "dense_inliers": 40,
        },
        {
            "query_id": "b",
            "sparse_te_cm": 80.0,
            "sparse_re_deg": 3.0,
            "sparse_inliers": 8,
            "dense_te_cm": 16.0,
            "dense_re_deg": 4.0,
            "dense_inliers": 35,
        },
    ]

    summary = lsf_pose_stage_summary(rows)

    assert summary["sparse"]["median_te_cm"] == 50.0
    assert summary["sparse"]["recall_50cm_5deg"] == 0.5
    assert summary["dense"]["recall_5cm_5deg"] == 0.5
    assert summary["dense"]["recall_15cm_5deg"] == 0.5
    assert summary["dense"]["avg_inliers"] == 37.5


def test_dense_transition_summary_counts_improved_and_worsened_queries():
    rows = [
        {"query_id": "better", "sparse_te_cm": 20.0, "sparse_re_deg": 1.0, "dense_te_cm": 4.0, "dense_re_deg": 1.0},
        {"query_id": "worse", "sparse_te_cm": 4.0, "sparse_re_deg": 1.0, "dense_te_cm": 20.0, "dense_re_deg": 1.0},
        {"query_id": "same", "sparse_te_cm": 8.0, "sparse_re_deg": 1.0, "dense_te_cm": 8.2, "dense_re_deg": 1.0},
    ]

    summary = dense_transition_summary(rows, tolerance_cm=1.0)

    assert summary["query_count"] == 3
    assert summary["improved_count"] == 1
    assert summary["worsened_count"] == 1
    assert summary["unchanged_count"] == 1
    assert summary["recovered_r5_count"] == 1
    assert summary["lost_r5_count"] == 1


def test_summarize_lsf_eval_combines_pose_timing_budget_and_offline_costs():
    rows = [
        {"query_id": "a", "sparse_te_cm": 20.0, "sparse_re_deg": 2.0, "dense_te_cm": 4.0, "dense_re_deg": 1.0},
        {"query_id": "b", "sparse_te_cm": 80.0, "sparse_re_deg": 3.0, "dense_te_cm": 16.0, "dense_re_deg": 4.0},
    ]
    timing = {
        "latency_ms": {
            "total": {"mean": 100.0, "median": 95.0, "p95": 150.0},
            "sparse_match": {"mean": 40.0},
        },
        "fps": {"mean_latency": 10.0},
    }

    summary = summarize_lsf_eval(
        rows,
        scene="ToyScene",
        method="lsf_sparse",
        landmark_count=12288,
        timing_profile=timing,
        offline_costs={"feedback_cache_s": 12.0, "peak_gpu_mb": 1024.0},
        map_size_mb=33.5,
    )

    assert summary["scene"] == "ToyScene"
    assert summary["method"] == "lsf_sparse"
    assert summary["landmark_budget"]["landmark_count"] == 12288
    assert summary["map_size_mb"] == 33.5
    assert summary["offline_costs"]["feedback_cache_s"] == 12.0
    assert summary["online_timing"]["latency_ms"]["total"]["mean"] == 100.0
    assert summary["dense_transition"]["improved_count"] == 2
