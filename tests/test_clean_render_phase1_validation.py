import csv
import json

from loc_gs.diagnostics.clean_render_phase1_validation import (
    summarize_phase0_action_safety,
    summarize_selected_render_health,
)
from loc_gs.scripts.build_clean_render_phase1_report import main as build_phase1_main


def test_phase1_action_safety_reports_non_base_actions_by_case_type():
    cases = [
        {"case_type": "D_normal_dense_good", "sparse_conditioned_label": "base", "sparse_conditioned_delta_cm": "0"},
        {"case_type": "D_normal_dense_good", "sparse_conditioned_label": "gated_base", "sparse_conditioned_delta_cm": "25"},
        {"case_type": "A_sparse_good_dense_bad", "sparse_conditioned_label": "gated_ray_up", "sparse_conditioned_delta_cm": "-30"},
    ]

    summary = summarize_phase0_action_safety(cases)

    assert summary["by_type"]["D_normal_dense_good"]["count"] == 2
    assert summary["by_type"]["D_normal_dense_good"]["non_base_action_count"] == 1
    assert summary["by_type"]["D_normal_dense_good"]["regression_20cm_count"] == 1
    assert summary["by_type"]["A_sparse_good_dense_bad"]["improvement_20cm_count"] == 1


def test_phase1_action_safety_counts_missing_delta_instead_of_treating_as_zero():
    cases = [
        {"case_type": "D_normal_dense_good", "sparse_conditioned_label": "base", "sparse_conditioned_delta_cm": ""},
        {"case_type": "D_normal_dense_good", "sparse_conditioned_label": "guided"},
    ]

    summary = summarize_phase0_action_safety(cases)

    row = summary["by_type"]["D_normal_dense_good"]
    assert row["missing_delta_count"] == 2
    assert row["median_sparse_conditioned_delta_cm"] is None
    assert row["regression_20cm_count"] == 0
    assert row["improvement_20cm_count"] == 0


def test_phase1_render_health_summarizes_case_analysis_metrics():
    rows = [
        {
            "category": "improvement",
            "selected_label": "gated_ray_up-5.000",
            "visible_ratio": "0.80",
            "near_occluder_ratio": "0.10",
            "feature_cosine_median": "0.50",
            "delta_sc_minus_none_cm": "-25",
        },
        {
            "category": "regression",
            "selected_label": "gated_base",
            "visible_ratio": "0.95",
            "near_occluder_ratio": "0.03",
            "feature_cosine_median": "0.70",
            "delta_sc_minus_none_cm": "30",
        },
    ]

    summary = summarize_selected_render_health(rows)

    assert summary["case_count"] == 2
    assert summary["by_category"]["improvement"]["median_visible_ratio"] == 0.8
    assert summary["by_category"]["regression"]["regression_20cm_count"] == 1
    assert summary["by_category"]["regression"]["non_base_selected_count"] == 1


def test_phase1_render_health_counts_missing_numeric_fields():
    rows = [
        {
            "category": "unknown",
            "selected_label": "guided",
            "visible_ratio": "",
            "near_occluder_ratio": "",
            "feature_cosine_median": "",
            "delta_sc_minus_none_cm": "",
        }
    ]

    summary = summarize_selected_render_health(rows)

    row = summary["by_category"]["unknown"]
    assert row["missing_delta_count"] == 1
    assert row["missing_visible_ratio_count"] == 1
    assert row["median_delta_sc_minus_none_cm"] is None


def test_build_phase1_report_cli_writes_phase1_metrics(tmp_path):
    phase0 = tmp_path / "phase0_cases.csv"
    with phase0.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_type", "sparse_conditioned_label", "sparse_conditioned_delta_cm"])
        writer.writeheader()
        writer.writerow(
            {
                "case_type": "D_normal_dense_good",
                "sparse_conditioned_label": "base",
                "sparse_conditioned_delta_cm": "0",
            }
        )
    case_analysis = tmp_path / "case_analysis.csv"
    with case_analysis.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "selected_label",
                "visible_ratio",
                "near_occluder_ratio",
                "feature_cosine_median",
                "delta_sc_minus_none_cm",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "category": "improvement",
                "selected_label": "gated_ray_up",
                "visible_ratio": "0.8",
                "near_occluder_ratio": "0.1",
                "feature_cosine_median": "0.5",
                "delta_sc_minus_none_cm": "-25",
            }
        )
    out = tmp_path / "phase1"

    code = build_phase1_main(
        [
            "--phase0_cases",
            str(phase0),
            "--case_analysis",
            str(case_analysis),
            "--output_dir",
            str(out),
        ]
    )

    assert code == 0
    metrics = json.loads((out / "metrics_summary.json").read_text())
    assert metrics["action_safety"]["by_type"]["D_normal_dense_good"]["count"] == 1
    assert metrics["selected_render_health"]["by_category"]["improvement"]["improvement_20cm_count"] == 1
    assert (out / "report.md").exists()


def test_build_phase1_report_cli_marks_test_phase0_source(tmp_path):
    phase0 = tmp_path / "phase0_cases.csv"
    with phase0.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_type",
                "source_split",
                "sparse_conditioned_label",
                "sparse_conditioned_delta_cm",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_type": "A_sparse_good_dense_bad",
                "source_split": "test",
                "sparse_conditioned_label": "base",
                "sparse_conditioned_delta_cm": "0",
            }
        )
    case_analysis = tmp_path / "case_analysis.csv"
    with case_analysis.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "selected_label",
                "visible_ratio",
                "near_occluder_ratio",
                "feature_cosine_median",
                "delta_sc_minus_none_cm",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "category": "improvement",
                "selected_label": "gated_ray_up",
                "visible_ratio": "0.8",
                "near_occluder_ratio": "0.1",
                "feature_cosine_median": "0.5",
                "delta_sc_minus_none_cm": "-25",
            }
        )
    out = tmp_path / "phase1"

    code = build_phase1_main(
        [
            "--phase0_cases",
            str(phase0),
            "--case_analysis",
            str(case_analysis),
            "--output_dir",
            str(out),
        ]
    )

    assert code == 0
    split_audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))
    assert split_audit["official_test_used"] is True
    assert split_audit["source_splits"] == ["test"]
