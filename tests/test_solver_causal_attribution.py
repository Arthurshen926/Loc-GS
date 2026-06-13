import pytest

from loc_gs.feedback.solver_causal_attribution import (
    annotate_sparse_correspondences_with_solver_causal_fields,
    pnp_conditioning_metrics,
)


def _row(query_id, gid, xyz, *, source_role="baseline_trace", inlier=True):
    return {
        "query_id": query_id,
        "image_id": query_id,
        "gaussian_id": gid,
        "match_row_index": gid,
        "source_role": source_role,
        "pnp_inlier": inlier,
        "camera_xyz": xyz,
    }


def test_pnp_conditioning_metrics_reports_logdet_and_min_eigenvalue():
    rows = [
        _row("q1", 1, [0.0, 0.0, 3.0]),
        _row("q1", 2, [0.5, 0.0, 4.0]),
        _row("q1", 3, [0.0, 0.5, 5.0]),
        _row("q1", 4, [0.5, 0.5, 6.0]),
        _row("q1", 5, [1.0, 0.2, 7.0]),
    ]

    metrics = pnp_conditioning_metrics(rows)

    assert metrics["pnp_conditioning_support_count"] == 5
    assert metrics["pnp_logdet_jtj"] != 0.0
    assert metrics["pnp_min_eig_jtj"] >= 0.0


def test_solver_causal_annotations_include_minimal_set_and_paired_deltas():
    rows = [
        _row("q1", 1, [0.0, 0.0, 3.0], source_role="baseline_trace"),
        _row("q1", 2, [0.1, 0.0, 3.1], source_role="baseline_trace"),
        _row("q1", 3, [0.0, 0.1, 3.2], source_role="baseline_trace"),
        _row("q1", 4, [0.1, 0.1, 3.3], source_role="baseline_trace"),
        _row("q1", 1, [0.0, 0.0, 3.0], source_role="candidate_trace"),
        _row("q1", 2, [0.8, 0.0, 3.5], source_role="candidate_trace"),
        _row("q1", 3, [0.0, 0.8, 4.0], source_role="candidate_trace"),
        _row("q1", 4, [0.8, 0.8, 4.5], source_role="candidate_trace"),
        _row("q1", 5, [1.2, 0.4, 5.5], source_role="candidate_trace"),
    ]

    annotations, metrics = annotate_sparse_correspondences_with_solver_causal_fields(rows)

    assert metrics["solver_causal_attribution_enabled"] is True
    assert metrics["paired_query_count"] == 1
    candidate = annotations[-1]
    assert candidate["candidate_vs_baseline_support_delta"] == 1
    assert candidate["candidate_vs_baseline_logdet_delta"] != 0.0
    assert candidate["minimal_set_logdet_drop"] >= 0.0
    assert candidate["minimal_set_counterfactual_support_delta"] == -1


def test_solver_causal_annotations_reject_test_split_rows():
    with pytest.raises(ValueError, match="test split"):
        annotate_sparse_correspondences_with_solver_causal_fields(
            [_row("q1", 1, [0.0, 0.0, 3.0]) | {"split_name": "test"}]
        )
