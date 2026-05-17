from loc_gs.stdloc_native.timing_profile import aggregate_timing_profile, finalize_query_timing


def test_finalize_query_timing_derives_stage_breakdowns():
    row = finalize_query_timing(
        {
            "image_name": "seq/frame.png",
            "feature_ms": 10.0,
            "sparse_total_ms": 30.0,
            "sparse_pose_ms": 7.5,
            "dense_total_ms": 90.0,
            "dense_pose_ms": 12.0,
            "total_ms": 130.0,
        }
    )

    assert row["sparse_match_ms"] == 22.5
    assert row["dense_match_render_ms"] == 78.0
    assert row["image_name"] == "seq/frame.png"


def test_aggregate_timing_profile_reports_latency_and_fps():
    rows = [
        finalize_query_timing(
            {
                "feature_ms": 10.0,
                "sparse_total_ms": 30.0,
                "sparse_pose_ms": 5.0,
                "dense_total_ms": 80.0,
                "dense_pose_ms": 10.0,
                "total_ms": 120.0,
            }
        ),
        finalize_query_timing(
            {
                "feature_ms": 20.0,
                "sparse_total_ms": 40.0,
                "sparse_pose_ms": 15.0,
                "dense_total_ms": 100.0,
                "dense_pose_ms": 20.0,
                "total_ms": 160.0,
            }
        ),
    ]

    profile = aggregate_timing_profile(
        rows,
        scene="ShopFacade",
        method="native8192",
        landmark_count=8192,
        dense_iterations=1,
    )

    assert profile["scene"] == "ShopFacade"
    assert profile["method"] == "native8192"
    assert profile["queries"] == 2
    assert profile["landmark_count"] == 8192
    assert profile["dense_iterations"] == 1
    assert profile["latency_ms"]["total"]["mean"] == 140.0
    assert profile["latency_ms"]["total"]["median"] == 140.0
    assert profile["latency_ms"]["total"]["p95"] == 158.0
    assert profile["latency_ms"]["sparse_match"]["mean"] == 25.0
    assert profile["latency_ms"]["dense_match_render"]["mean"] == 75.0
    assert profile["fps"]["mean_latency"] == 1000.0 / 140.0


def test_aggregate_timing_profile_summarizes_pose_reliability():
    rows = [
        {
            "feature_ms": 10.0,
            "sparse_total_ms": 30.0,
            "sparse_pose_ms": 5.0,
            "dense_total_ms": 80.0,
            "dense_pose_ms": 10.0,
            "total_ms": 120.0,
            "sparse_pose_reliability": {
                "match_count": 100,
                "inlier_count": 40,
                "inlier_ratio": 0.4,
                "all_reprojection_median_px": 3.0,
                "inlier_reprojection_median_px": 1.0,
            },
        },
        {
            "feature_ms": 20.0,
            "sparse_total_ms": 40.0,
            "sparse_pose_ms": 15.0,
            "dense_total_ms": 100.0,
            "dense_pose_ms": 20.0,
            "total_ms": 160.0,
            "sparse_pose_reliability": {
                "match_count": 50,
                "inlier_count": 10,
                "inlier_ratio": 0.2,
                "all_reprojection_median_px": 5.0,
                "inlier_reprojection_median_px": 2.0,
            },
        },
    ]

    profile = aggregate_timing_profile(rows, scene="ShopFacade", method="native8192")

    sparse = profile["pose_reliability"]["sparse"]
    assert sparse["count"] == 2
    assert sparse["match_count"]["mean"] == 75.0
    assert sparse["inlier_count"]["mean"] == 25.0
    assert sparse["inlier_ratio"]["mean"] == 0.30000000000000004
    assert sparse["all_reprojection_median_px"]["median"] == 4.0
    assert sparse["inlier_reprojection_median_px"]["mean"] == 1.5
