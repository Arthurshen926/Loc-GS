import csv
import json

import pytest

from loc_gs.diagnostics.clean_render_phase0_benchmark import (
    Phase0Thresholds,
    classify_phase0_case,
    make_phase0_splits,
)
from loc_gs.scripts.build_clean_render_phase0_benchmark import main as build_phase0_main


def _row(scene, query_index, sparse, dense, *, image_name=None, split="train"):
    return {
        "scene": scene,
        "query_index": query_index,
        "image_name": image_name or f"seq/frame{query_index:05d}.png",
        "split": split,
        "sparse_te_cm": sparse,
        "base_dense_te_cm": dense,
        "sparse_inlier_count": 80,
    }


def test_phase0_classifier_assigns_four_sparse_dense_case_types():
    thresholds = Phase0Thresholds(
        sparse_good_te_cm=30.0,
        dense_bad_margin_cm=20.0,
        sparse_marginal_te_cm=200.0,
        dense_recovery_margin_cm=20.0,
        sparse_catastrophic_te_cm=300.0,
        normal_dense_te_cm=15.0,
    )

    assert classify_phase0_case(_row("S", 0, 20.0, 55.0), thresholds) == "A_sparse_good_dense_bad"
    assert classify_phase0_case(_row("S", 1, 120.0, 70.0), thresholds) == "B_sparse_marginal_dense_recoverable"
    assert classify_phase0_case(_row("S", 2, 450.0, 430.0), thresholds) == "C_sparse_catastrophic"
    assert classify_phase0_case(_row("S", 3, 18.0, 8.0), thresholds) == "D_normal_dense_good"


def test_phase0_splits_are_deterministic_and_hold_out_validation():
    rows = []
    for i in range(12):
        rows.append(_row("A", i, 20.0, 60.0))
        rows.append(_row("B", i, 100.0, 60.0))
        rows.append(_row("C", i, 400.0, 390.0))
        rows.append(_row("D", i, 18.0, 8.0))

    result = make_phase0_splits(rows, max_per_type=10, val_fraction=0.3)

    assert result["summary"]["total_selected"] == 40
    assert result["summary"]["type_counts"]["A_sparse_good_dense_bad"] == 10
    assert result["summary"]["split_counts"]["train"] == 28
    assert result["summary"]["split_counts"]["val"] == 12
    assert result["cases"] == make_phase0_splits(rows, max_per_type=10, val_fraction=0.3)["cases"]


def test_build_phase0_benchmark_cli_writes_csv_and_audit_bundle(tmp_path):
    results = [
        _row("S", 0, 20.0, 55.0),
        _row("S", 1, 120.0, 70.0),
        _row("S", 2, 450.0, 430.0),
        _row("S", 3, 18.0, 8.0),
    ]
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")
    output_dir = tmp_path / "phase0"

    code = build_phase0_main(
        [
            "--results",
            str(results_path),
            "--output_dir",
            str(output_dir),
            "--max_per_type",
            "4",
        ]
    )

    assert code == 0
    cases = list(csv.DictReader((output_dir / "phase0_cases.csv").open()))
    assert {row["case_type"] for row in cases} == {
        "A_sparse_good_dense_bad",
        "B_sparse_marginal_dense_recoverable",
        "C_sparse_catastrophic",
        "D_normal_dense_good",
    }
    assert (output_dir / "metrics_summary.json").exists()
    assert (output_dir / "split_audit.json").exists()


def test_build_phase0_benchmark_cli_rejects_test_split_inputs(tmp_path):
    results = [
        _row("S", 0, 20.0, 55.0, split="test"),
        _row("S", 1, 120.0, 70.0, split="train"),
    ]
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")
    output_dir = tmp_path / "phase0"

    with pytest.raises(ValueError, match="Phase0 benchmark requires train split"):
        build_phase0_main(
            [
                "--results",
                str(results_path),
                "--output_dir",
                str(output_dir),
                "--max_per_type",
                "4",
            ]
        )
