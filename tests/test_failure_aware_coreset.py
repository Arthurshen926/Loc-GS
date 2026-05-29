from loc_gs.stdloc_native.failure_aware_coreset import select_failure_aware_replacements


def test_failure_aware_coreset_protects_native_good_queries_and_selects_hard_gain():
    result = select_failure_aware_replacements(
        source_ids=[1, 2, 3],
        candidate_ids=[10, 11],
        query_baseline_dense_te_cm={"easy": 8.0, "hard": 35.0},
        candidate_query_gain={10: {"hard": 4.0}, 11: {"hard": 9.0}},
        candidate_regression_risk={10: {"easy": 0.0}, 11: {"easy": 6.0}},
        dense_worsened_query_ids=[],
        source_loss={1: 0.1, 2: 0.2, 3: 0.3},
        max_edits=1,
    )

    assert result["selected_ids"] == [1, 2, 10]
    assert result["edits"] == [{"drop_id": 3, "add_id": 10, "score": result["edits"][0]["score"]}]
    assert result["metadata"]["rejected_protected_regression_count"] == 1


def test_failure_aware_coreset_uses_conflict_graph_to_avoid_repeated_structure_pairs():
    result = select_failure_aware_replacements(
        source_ids=[1, 2, 3],
        candidate_ids=[10, 12],
        query_baseline_dense_te_cm={"hard": 50.0},
        candidate_query_gain={10: {"hard": 3.0}, 12: {"hard": 6.0}},
        candidate_regression_risk={},
        dense_worsened_query_ids=[],
        source_loss={1: 0.1, 2: 0.1, 3: 0.1},
        negative_support_graph={
            "edge_weights": {(2, 12): 10.0},
            "unary_risk": {},
        },
        conflict_weight=1.0,
        max_edits=1,
    )

    assert result["selected_ids"] == [1, 2, 10]
    assert result["metadata"]["selected_recipe"] == "lsf_v7_failure_aware"
