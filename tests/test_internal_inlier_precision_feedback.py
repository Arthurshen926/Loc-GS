import json
from pathlib import Path

import pytest

from loc_gs.scripts.build_internal_inlier_precision_feedback import main
from loc_gs.teacher.distillation_artifact import SolverFeedbackRow
from loc_gs.teacher.inlier_precision_feedback import (
    InlierPrecisionFeedbackConfig,
    apply_inlier_precision_feedback_to_solver_rows,
    build_inlier_precision_feedback,
    load_inlier_precision_feedback_rows,
)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "query_id": "hard.png",
            "success": True,
            "te_cm": 80.0,
            "selected_set_diagnostics": {
                "selected_geometric_correct_ratio": 0.05,
                "selected_geometric_correct_count": 25,
            },
            "inlier_set_diagnostics": {
                "inlier_geometric_correct_ratio": 0.20,
                "inlier_geometric_correct_count": 12,
            },
            "post_pnp_rescore_changed_count": 12,
            "post_pnp_rescore_corrected_count": 1,
            "post_pnp_rescore_worsened_count": 5,
            "post_pnp_rescore_correct_delta": -4,
            "set_conflict_rerank_changed_count": 120,
        },
        {
            "query_id": "stable.png",
            "success": True,
            "te_cm": 7.0,
            "selected_set_diagnostics": {
                "selected_geometric_correct_ratio": 0.35,
                "selected_geometric_correct_count": 180,
            },
            "inlier_set_diagnostics": {
                "inlier_geometric_correct_ratio": 0.80,
                "inlier_geometric_correct_count": 120,
            },
            "post_pnp_rescore_changed_count": 0,
            "post_pnp_rescore_corrected_count": 0,
            "post_pnp_rescore_worsened_count": 0,
            "post_pnp_rescore_correct_delta": 0,
            "set_conflict_rerank_changed_count": 0,
        },
    ]


def _metrics() -> dict[str, object]:
    return {
        "schema_version": "internal_sparse_cached_eval_metrics_v1",
        "scene": "GreatCourt",
        "split_name": "train_dev",
        "query_count": 2,
        "success_count": 2,
        "median_te_cm": 55.6,
        "median_re_deg": 0.2,
    }


def test_inlier_precision_feedback_profiles_set_and_inlier_hard_queries():
    feedback = build_inlier_precision_feedback(
        _rows(),
        metrics=_metrics(),
        scene="GreatCourt",
        split_name="train_dev",
        cfg=InlierPrecisionFeedbackConfig(
            selected_correct_ratio_floor=0.10,
            inlier_correct_ratio_floor=0.50,
            target_median_te_cm=10.0,
            max_distill_weight=4.0,
        ),
    )

    assert feedback["schema_version"] == "internal_inlier_precision_feedback_v1"
    assert feedback["hard_query_count"] == 1
    assert feedback["set_selection_hard_count"] == 1
    assert feedback["inlier_precision_hard_count"] == 1
    assert feedback["post_pnp_rescore_harm_count"] == 1
    hard = feedback["per_query"][0]
    assert hard["query_role"] == "set_and_inlier_precision_hard"
    assert hard["selected_gap"] == pytest.approx(0.5)
    assert hard["inlier_gap"] == pytest.approx(0.6)
    assert hard["scorer_distill_weight"] > 2.0
    assert hard["hard_negative_weight"] > hard["set_selection_weight"]
    assert feedback["per_query"][1]["query_role"] == "stable"


def test_inlier_precision_feedback_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        build_inlier_precision_feedback(_rows(), metrics=_metrics(), scene="GreatCourt", split_name="test")


def test_inlier_precision_feedback_rows_can_boost_solver_feedback(tmp_path: Path):
    feedback = build_inlier_precision_feedback(
        _rows(),
        metrics=_metrics(),
        scene="GreatCourt",
        split_name="train_dev",
    )
    path = tmp_path / "feedback.json"
    path.write_text(json.dumps(feedback, sort_keys=True), encoding="utf-8")

    rows = load_inlier_precision_feedback_rows(path)
    merged, summary = apply_inlier_precision_feedback_to_solver_rows(
        [
            SolverFeedbackRow(
                scene="GreatCourt",
                split_name="train_dev",
                query_id="hard.png",
                dense_helped=True,
                distill_weight=0.25,
            )
        ],
        rows,
        weight_scale=1.0,
        max_distill_weight=4.0,
    )

    by_query = {row.query_id: row for row in merged}
    assert by_query["hard.png"].dense_helped is True
    assert by_query["hard.png"].distill_weight > 2.0
    assert "stable.png" not in by_query
    assert summary["matched_solver_feedback_count"] == 1
    assert summary["added_solver_feedback_count"] == 0
    assert summary["boosted_solver_feedback_count"] == 1


def test_inlier_precision_feedback_cli_writes_auditable_bundle(tmp_path: Path):
    cached_eval = tmp_path / "cached_eval"
    cached_eval.mkdir()
    (cached_eval / "metrics_summary.json").write_text(json.dumps(_metrics(), sort_keys=True), encoding="utf-8")
    (cached_eval / "results.json").write_text(json.dumps(_rows(), sort_keys=True), encoding="utf-8")
    out = tmp_path / "feedback"

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
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    jsonl_rows = [
        json.loads(line)
        for line in (out / "inlier_precision_feedback.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metrics["schema_version"] == "internal_inlier_precision_feedback_v1"
    assert metrics["hard_query_count"] == 1
    assert manifest["schema_version"] == "internal_inlier_precision_feedback_manifest_v1"
    assert manifest["dense_inference_enabled"] is False
    assert manifest["external_runtime_dependency"] == "forbidden"
    assert jsonl_rows[0]["query_id"] == "hard.png"
    assert "loc_gs.scripts.build_internal_inlier_precision_feedback" in (
        out / "command.txt"
    ).read_text(encoding="utf-8")
