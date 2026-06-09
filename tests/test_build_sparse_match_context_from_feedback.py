import json
from pathlib import Path

import pytest

from loc_gs.scripts.build_sparse_match_context_from_feedback import (
    build_sparse_match_context_from_records,
    load_profile_context_queries,
    merge_query_match_dominance,
)


def test_load_profile_context_queries_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        load_profile_context_queries({"split_name": "test", "protected_per_query_support": {"q": {"1": 1.0}}})


def test_build_sparse_match_context_keeps_weighted_inlier_and_non_inlier_context():
    records = [
        {
            "query_id": "q0",
            "matched_gaussian_id": "10",
            "descriptor_score": 0.8,
            "detector_score": 0.5,
            "pnp_inlier": True,
        },
        {
            "query_id": "q0",
            "matched_gaussian_id": "20",
            "descriptor_score": 0.9,
            "detector_score": 0.5,
            "pnp_inlier": False,
        },
        {
            "query_id": "q1",
            "matched_gaussian_id": "30",
            "descriptor_score": 1.0,
            "detector_score": 1.0,
            "pnp_inlier": True,
        },
    ]

    payload, metrics = build_sparse_match_context_from_records(
        records,
        query_ids={"q0"},
        split_name="train_dev",
        max_records_per_query=8,
        inlier_weight=1.0,
        non_inlier_weight=0.25,
    )

    strength = payload["queries"]["q0"]["support_match_strength"]
    assert set(strength) == {"10", "20"}
    assert strength["10"] > strength["20"]
    assert "q1" not in payload["queries"]
    assert payload["official_test_used"] is False
    assert metrics["query_count"] == 1
    assert metrics["context_entry_count"] == 2


def test_build_sparse_match_context_caps_per_query_entries():
    records = [
        {
            "query_id": "q0",
            "matched_gaussian_id": str(gid),
            "descriptor_score": float(gid),
            "detector_score": 1.0,
            "pnp_inlier": True,
        }
        for gid in range(1, 5)
    ]

    payload, metrics = build_sparse_match_context_from_records(
        records,
        query_ids={"q0"},
        split_name="train_dev",
        max_records_per_query=2,
    )

    strength = payload["queries"]["q0"]["support_match_strength"]
    assert list(strength) == ["4", "3"]
    assert metrics["context_entry_count"] == 2


def test_merge_query_match_dominance_preserves_base_risk_and_adds_context_strength():
    base = {
        "schema_version": "ulfloc_query_match_dominance_v1",
        "split_name": "train_dev",
        "official_test_used": False,
        "queries": {
            "q0": {
                "support_match_strength": {"10": 2.0},
                "match_competition_risk": {"20": 5.0},
            }
        },
    }
    context = {
        "schema_version": "locgs_sparse_match_context_from_feedback_v1",
        "split_name": "train_dev",
        "official_test_used": False,
        "queries": {
            "q0": {
                "support_match_strength": {"10": 1.0, "30": 3.0},
                "match_competition_risk": {},
            }
        },
    }

    merged = merge_query_match_dominance(base, context)
    query = merged["queries"]["q0"]
    assert query["support_match_strength"] == {"10": 2.0, "30": 3.0}
    assert query["match_competition_risk"] == {"20": 5.0}
    assert merged["metadata"]["base_query_count"] == 1
    assert merged["metadata"]["context_query_count"] == 1
