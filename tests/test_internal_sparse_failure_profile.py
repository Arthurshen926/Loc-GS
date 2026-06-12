import json
from pathlib import Path

import pytest

from loc_gs.scripts.build_internal_sparse_failure_profile import main
from loc_gs.sparse.failure_profile import SparseFailureProfileConfig, build_sparse_failure_profile


def _rows() -> list[dict[str, object]]:
    return [
        {
            "query_id": "q_bad_set.png",
            "success": True,
            "te_cm": 80.0,
            "selected_set_diagnostics": {
                "selected_geometric_correct_count": 25,
                "selected_geometric_correct_ratio": 0.05,
            },
            "inlier_set_diagnostics": {
                "inlier_geometric_correct_count": 12,
                "inlier_geometric_correct_ratio": 0.20,
            },
            "post_pnp_rescore_changed_count": 30,
            "post_pnp_rescore_corrected_count": 5,
            "post_pnp_rescore_worsened_count": 22,
            "post_pnp_rescore_correct_delta": -17,
            "set_conflict_rerank_changed_count": 300,
            "pnp_stage_count": 2,
            "lgcv_keep_count": 70,
        },
        {
            "query_id": "q_good.png",
            "success": True,
            "te_cm": 7.0,
            "selected_set_diagnostics": {
                "selected_geometric_correct_count": 180,
                "selected_geometric_correct_ratio": 0.35,
            },
            "inlier_set_diagnostics": {
                "inlier_geometric_correct_count": 120,
                "inlier_geometric_correct_ratio": 0.80,
            },
            "post_pnp_rescore_changed_count": 0,
            "post_pnp_rescore_corrected_count": 0,
            "post_pnp_rescore_worsened_count": 0,
            "post_pnp_rescore_correct_delta": 0,
            "set_conflict_rerank_changed_count": 0,
            "pnp_stage_count": 2,
            "lgcv_keep_count": 150,
        },
        {
            "query_id": "q_missing.png",
            "success": False,
            "reason": "missing_camera",
            "selected_set_diagnostics": {},
            "inlier_set_diagnostics": {},
        },
    ]


def _metrics() -> dict[str, object]:
    return {
        "schema_version": "internal_sparse_cached_eval_metrics_v1",
        "scene": "GreatCourt",
        "split_name": "train_dev",
        "query_count": 3,
        "success_count": 2,
        "median_te_cm": 55.6,
        "median_re_deg": 0.2,
        "rerank_diagnostic_enabled": True,
        "native_top1_correct": 10,
        "reranked_top1_correct": 25,
        "reranked_top1_gain": 15,
        "reranked_topk_available": 40,
        "candidate_artifact": {
            "top1_correct": 20,
            "topk_available": 60,
            "oracle_gap": 40,
        },
        "post_pnp_candidate_rescore_enabled": True,
    }


def test_sparse_failure_profile_classifies_internal_cached_eval_failure_modes():
    profile = build_sparse_failure_profile(
        _rows(),
        metrics=_metrics(),
        scene="GreatCourt",
        split_name="train_dev",
        cfg=SparseFailureProfileConfig(
            selected_correct_ratio_floor=0.10,
            inlier_correct_ratio_floor=0.50,
            target_median_te_cm=10.0,
        ),
    )

    assert profile["schema_version"] == "internal_sparse_failure_profile_v1"
    assert profile["scene"] == "GreatCourt"
    assert profile["split_name"] == "train_dev"
    assert profile["query_count"] == 3
    assert profile["failure_mode_counts"] == {
        "inlier_set_wrong_dominant": 1,
        "missing_or_failed_pose": 1,
        "post_pnp_rescore_harm": 1,
        "selected_set_low_precision": 1,
    }
    assert profile["dominant_failure_modes"][0] == {
        "mode": "inlier_set_wrong_dominant",
        "count": 1,
    }
    assert profile["candidate_artifact"]["oracle_gap"] == 40
    assert profile["rerank_diagnostic"]["reranked_top1_gain"] == 15
    assert profile["post_pnp_rescore"]["correct_delta_sum"] == -17
    assert profile["recommendation"] == "prioritize_set_level_selection_and_inlier_precision"
    assert profile["per_query"][0]["failure_modes"] == [
        "selected_set_low_precision",
        "inlier_set_wrong_dominant",
        "post_pnp_rescore_harm",
    ]
    assert profile["per_query"][2]["failure_modes"] == ["missing_or_failed_pose"]


def test_sparse_failure_profile_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        build_sparse_failure_profile(_rows(), metrics=_metrics(), scene="GreatCourt", split_name="test")


def test_sparse_failure_profile_cli_writes_auditable_bundle(tmp_path: Path):
    cached_eval = tmp_path / "cached_eval"
    cached_eval.mkdir()
    (cached_eval / "metrics_summary.json").write_text(json.dumps(_metrics(), sort_keys=True), encoding="utf-8")
    (cached_eval / "results.json").write_text(json.dumps(_rows(), sort_keys=True), encoding="utf-8")
    out = tmp_path / "profile"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
            "--cached_eval_dir",
            str(cached_eval),
            "--output_dir",
            str(out),
            "--target_median_te_cm",
            "10.0",
        ]
    )

    assert rc == 0
    profile = json.loads((out / "failure_profile.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))
    assert profile["failure_mode_counts"]["selected_set_low_precision"] == 1
    assert summary["schema_version"] == "internal_sparse_failure_profile_v1"
    assert manifest["schema_version"] == "internal_sparse_failure_profile_manifest_v1"
    assert manifest["dense_inference_enabled"] is False
    assert manifest["external_runtime_dependency"] == "forbidden"
    assert split_audit["audit_status"] == "passed"
    assert "loc_gs.scripts.build_internal_sparse_failure_profile" in (out / "command.txt").read_text(
        encoding="utf-8"
    )
