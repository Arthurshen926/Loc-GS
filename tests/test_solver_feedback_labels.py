import json
from pathlib import Path

import pytest

from loc_gs.scripts.build_internal_solver_feedback_labels import main
from loc_gs.teacher.solver_feedback import (
    SolverFeedbackConfig,
    build_solver_feedback_labels,
)


def test_solver_feedback_labels_measure_dense_improvement_and_weights():
    labels = build_solver_feedback_labels(
        scene="GreatCourt",
        split_name="train_dev",
        sparse_rows=[
            {"image_name": "q1.png", "sparse_te_cm": 20.0, "sparse_re_deg": 1.0, "sparse_inliers": 30},
            {"image_name": "q2.png", "sparse_te_cm": 8.0, "sparse_re_deg": 0.2, "sparse_inliers": 100},
        ],
        dense_rows=[
            {"image_name": "q1.png", "dense_te_cm": 9.0, "dense_re_deg": 0.2},
            {"image_name": "q2.png", "dense_te_cm": 9.0, "dense_re_deg": 0.3},
        ],
        cfg=SolverFeedbackConfig(min_dense_improvement_cm=2.0, catastrophic_sparse_cm=50.0, weight_clip_cm=20.0),
    )

    assert len(labels) == 2
    assert labels[0].query_id == "q1.png"
    assert labels[0].dense_helped is True
    assert labels[0].translation_improvement_cm == 11.0
    assert labels[0].distill_weight == pytest.approx(0.55)
    assert labels[1].dense_helped is False
    assert labels[1].distill_weight == 0.0


def test_solver_feedback_labels_accept_legacy_dense_result_field_names():
    labels = build_solver_feedback_labels(
        scene="GreatCourt",
        split_name="train",
        sparse_rows=[
            {"image_name": "q1.png", "sparse_te": 22.0, "sparse_ae": 0.9, "sparse_inliers": 30},
            {"image_name": "q2.png", "te_cm": 12.0, "re_deg": 0.4, "inlier_count": 70},
        ],
        dense_rows=[
            {"image_name": "q1.png", "sparse_conditioned_dense_te_cm": 8.0, "sparse_conditioned_dense_re_deg": 0.2},
            {"image_name": "q2.png", "base_dense_te_cm": 11.0, "base_dense_re_deg": 0.3},
        ],
        cfg=SolverFeedbackConfig(min_dense_improvement_cm=2.0, weight_clip_cm=20.0),
    )

    assert labels[0].dense_helped is True
    assert labels[0].translation_improvement_cm == 14.0
    assert labels[0].rotation_improvement_deg == 0.7
    assert labels[1].dense_helped is False


def test_solver_feedback_labels_reject_sparse_only_dense_rows():
    with pytest.raises(ValueError, match="dense result row"):
        build_solver_feedback_labels(
            scene="GreatCourt",
            split_name="train_dev",
            sparse_rows=[{"image_name": "q1.png", "sparse_te_cm": 20.0, "sparse_re_deg": 1.0}],
            dense_rows=[{"image_name": "q1.png", "sparse_te_cm": 10.0, "sparse_re_deg": 0.5}],
            cfg=SolverFeedbackConfig(min_dense_improvement_cm=2.0, weight_clip_cm=20.0),
        )


def test_solver_feedback_labels_reject_test_split():
    with pytest.raises(ValueError, match="test split"):
        build_solver_feedback_labels(
            scene="GreatCourt",
            split_name="test",
            sparse_rows=[],
            dense_rows=[],
        )


def test_solver_feedback_label_cli_writes_jsonl_manifest_and_summary(tmp_path: Path):
    sparse = tmp_path / "sparse.json"
    dense = tmp_path / "dense.json"
    sparse.write_text(
        json.dumps({"rows": [{"image_name": "q1.png", "sparse_te_cm": 20.0, "sparse_re_deg": 1.0}]}),
        encoding="utf-8",
    )
    dense.write_text(
        json.dumps({"rows": [{"image_name": "q1.png", "dense_te_cm": 10.0, "dense_re_deg": 0.5}]}),
        encoding="utf-8",
    )
    out = tmp_path / "labels"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
            "--sparse_results",
            str(sparse),
            "--dense_results",
            str(dense),
            "--output_dir",
            str(out),
            "--min_dense_improvement_cm",
            "2.0",
        ]
    )

    assert rc == 0
    labels = [json.loads(line) for line in (out / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert labels[0]["dense_helped"] is True
    assert summary["label_count"] == 1
    assert summary["dense_helped_count"] == 1
    assert manifest["dense_teacher_enabled"] is True
    assert manifest["inference_stage"] == "training_label_generation"
