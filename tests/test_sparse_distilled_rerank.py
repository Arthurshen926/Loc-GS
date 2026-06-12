from loc_gs.sparse.rerank import (
    CandidateRerankConfig,
    rerank_candidate_rows,
    summarize_candidate_availability,
)


def test_reranker_preserves_high_score_prefix_and_reranks_tail():
    rows = [
        {"candidate_id": "a", "native_score": 0.99, "solver_score": 0.1},
        {"candidate_id": "b", "native_score": 0.98, "solver_score": 0.2},
        {"candidate_id": "c", "native_score": 0.70, "solver_score": 0.3},
        {"candidate_id": "d", "native_score": 0.60, "solver_score": 0.9},
    ]

    reranked = rerank_candidate_rows(rows, CandidateRerankConfig(prefix_fraction=0.5, solver_weight=1.0))

    assert [row["candidate_id"] for row in reranked] == ["a", "b", "d", "c"]


def test_candidate_availability_reports_oracle_gap():
    summary = summarize_candidate_availability(
        [
            [{"geometric_correct": False}, {"geometric_correct": True}],
            [{"geometric_correct": False}, {"geometric_correct": False}],
        ]
    )

    assert summary["query_count"] == 2
    assert summary["top1_correct"] == 0
    assert summary["topk_available"] == 1
    assert summary["oracle_gap"] == 1
