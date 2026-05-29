from loc_gs.stdloc_native.v13_gate import select_passing_v13_components


def test_v13_gate_only_keeps_components_that_pass_all_main_thresholds():
    result = select_passing_v13_components(
        {
            "v7": {
                "macro_dense_median_te_delta_cm": -0.6,
                "macro_dense_r10_delta": 0.006,
                "oldhospital_dense_median_te_delta_cm": -0.1,
                "stmaryschurch_dense_median_te_delta_cm": 0.0,
                "p95_te_delta_cm": -0.2,
            },
            "v9": {
                "macro_dense_median_te_delta_cm": -0.7,
                "macro_dense_r10_delta": 0.001,
                "macro_dense_r5_delta": 0.001,
                "oldhospital_dense_median_te_delta_cm": 0.2,
                "stmaryschurch_dense_median_te_delta_cm": 0.0,
                "p95_te_delta_cm": -0.1,
            },
        }
    )

    assert result["passing_components"] == ["v7"]
    assert result["rejected_components"]["v9"] == ["strict_recall_gain_too_small", "oldhospital_regression"]


def test_v13_gate_rejects_dense_verifier_when_transition_or_latency_gate_fails():
    result = select_passing_v13_components(
        {
            "v10_good": {
                "macro_dense_median_te_delta_cm": -0.6,
                "macro_dense_r5_delta": 0.006,
                "macro_dense_r2_delta": 0.0,
                "oldhospital_dense_median_te_delta_cm": -0.1,
                "stmaryschurch_dense_median_te_delta_cm": 0.0,
                "p95_te_delta_cm": -0.2,
                "dense_worsened_reduction_rate": 0.25,
                "latency_delta_fraction": 0.08,
            },
            "v10_bad": {
                "macro_dense_median_te_delta_cm": -0.6,
                "macro_dense_r5_delta": 0.001,
                "macro_dense_r2_delta": -0.001,
                "oldhospital_dense_median_te_delta_cm": -0.1,
                "stmaryschurch_dense_median_te_delta_cm": 0.0,
                "p95_te_delta_cm": -0.2,
                "dense_worsened_reduction_rate": 0.10,
                "latency_delta_fraction": 0.20,
            },
        }
    )

    assert result["passing_components"] == ["v10_good"]
    assert result["rejected_components"]["v10_bad"] == [
        "strict_recall_gain_too_small",
        "dense_r2_regression",
        "dense_worsened_reduction_too_small",
        "latency_regression",
    ]


def test_v13_gate_rejects_candidate_regression_counts_when_reported():
    result = select_passing_v13_components(
        {
            "tail_bad": {
                "macro_dense_median_te_delta_cm": -0.6,
                "macro_dense_r10_delta": 0.006,
                "oldhospital_dense_median_te_delta_cm": -0.1,
                "stmaryschurch_dense_median_te_delta_cm": -0.1,
                "p95_te_delta_cm": -0.2,
                "candidate_regression_20cm_count": 1,
                "candidate_regression_50cm_count": 0,
            }
        }
    )

    assert result["passing_components"] == []
    assert result["rejected_components"]["tail_bad"] == ["candidate_regression_20cm"]
