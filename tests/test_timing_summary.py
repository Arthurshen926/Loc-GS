from loc_gs.eval.timing import merge_offline_costs, timing_profile_digest


def test_timing_profile_digest_keeps_online_breakdown_and_fps():
    timing = {
        "latency_ms": {
            "feature": {"mean": 10.0, "median": 9.0, "p95": 15.0},
            "sparse_match": {"mean": 40.0, "median": 39.0, "p95": 55.0},
            "dense_total": {"mean": 80.0, "median": 78.0, "p95": 110.0},
            "total": {"mean": 140.0, "median": 130.0, "p95": 180.0},
        },
        "fps": {"mean_latency": 1000.0 / 140.0},
    }

    digest = timing_profile_digest(timing)

    assert digest["total_ms"]["mean"] == 140.0
    assert digest["feature_ms"]["p95"] == 15.0
    assert digest["sparse_match_ms"]["median"] == 39.0
    assert digest["dense_total_ms"]["mean"] == 80.0
    assert digest["fps"] == 1000.0 / 140.0


def test_merge_offline_costs_normalizes_time_memory_and_size():
    costs = merge_offline_costs(
        base_map_s=60.0,
        feedback_cache_s=12.5,
        selector_train_s=3.0,
        export_s=1.5,
        peak_gpu_mb=2048.0,
        map_size_mb=33.5,
    )

    assert costs["base_map_s"] == 60.0
    assert costs["locgs_extra_s"] == 17.0
    assert costs["total_offline_s"] == 77.0
    assert costs["peak_gpu_mb"] == 2048.0
    assert costs["map_size_mb"] == 33.5

