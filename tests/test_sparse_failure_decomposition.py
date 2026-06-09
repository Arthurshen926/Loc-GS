from __future__ import annotations

from loc_gs.diagnostics.sparse_failure_decomposition import decompose_sparse_failure


def test_sparse_failure_decomposition_detects_detector_starvation() -> None:
    report = decompose_sparse_failure(
        {
            "query_id": "q0",
            "keypoint_count": 80,
            "match_count": 12,
            "pnp_inlier_count": 3,
            "image_coverage": 0.08,
            "median_te_cm": 500.0,
        }
    )

    assert report["primary_cause"] == "detector_starvation"
    assert "detector_starvation" in report["active_flags"]


def test_sparse_failure_decomposition_distinguishes_ranking_from_descriptor_ambiguity() -> None:
    ranking = decompose_sparse_failure(
        {
            "query_id": "q1",
            "match_count": 400,
            "pnp_inlier_count": 40,
            "oracle_inlier_count": 90,
            "top_rank_inlier_rate": 0.04,
            "all_match_inlier_rate": 0.22,
            "descriptor_margin_median": 0.35,
            "geometry_logdet": 8.0,
        }
    )
    ambiguity = decompose_sparse_failure(
        {
            "query_id": "q2",
            "match_count": 400,
            "pnp_inlier_count": 40,
            "descriptor_margin_median": 0.02,
            "descriptor_conflict_rate": 0.55,
            "geometry_logdet": 8.0,
        }
    )

    assert ranking["primary_cause"] == "ranking_failure"
    assert ambiguity["primary_cause"] == "descriptor_ambiguity"


def test_sparse_failure_decomposition_reports_geometry_and_artifact_flags() -> None:
    report = decompose_sparse_failure(
        {
            "query_id": "q3",
            "match_count": 300,
            "pnp_inlier_count": 60,
            "descriptor_margin_median": 0.2,
            "geometry_logdet": 0.1,
            "depth_spread_m": 0.05,
            "render_valid_ratio": 0.2,
            "render_feature_cosine_median": 0.03,
        }
    )

    assert report["primary_cause"] == "map_or_render_artifact"
    assert "geometry_degeneracy" in report["active_flags"]
    assert "map_or_render_artifact" in report["active_flags"]
