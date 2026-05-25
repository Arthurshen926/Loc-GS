import json
import subprocess
import sys

from loc_gs.eval.support_failure_mechanism import build_support_failure_mechanism_report


def _solver_report():
    return {
        "scene": "OldHospital",
        "source": {
            "summary": {
                "support_count_sum": 128,
                "viable_tuple_mass": 16.0,
                "mean_logdet_H": 4.5,
                "mean_ambiguity_risk": 0.1,
                "query_group_mode": "query_id",
                "split_audit": {"audit_status": "passed"},
            }
        },
        "candidate": {
            "summary": {
                "support_count_sum": 128,
                "viable_tuple_mass": 10.0,
                "mean_logdet_H": 2.0,
                "mean_ambiguity_risk": 0.25,
            }
        },
        "summary_delta": {
            "support_count_sum": 0,
            "viable_tuple_mass": -6.0,
            "mean_logdet_H": -2.5,
            "mean_ambiguity_risk": 0.15,
        },
    }


def test_support_failure_mechanism_flags_preserved_support_with_solver_drop():
    report = build_support_failure_mechanism_report(
        _solver_report(),
        baseline_metrics={"dense": {"median_te_cm": 9.0, "recall_5cm_5deg": 0.4, "recall_2cm_2deg": 0.12}},
        candidate_metrics={"dense": {"median_te_cm": 11.5, "recall_5cm_5deg": 0.34, "recall_2cm_2deg": 0.08}},
        baseline_transition={"summary": {"worsened_count": 3, "r5_lost_count": 1, "r2_lost_count": 0}},
        candidate_transition={"summary": {"worsened_count": 7, "r5_lost_count": 3, "r2_lost_count": 2}},
    )

    deltas = report["mechanism_table"]["deltas"]
    assert deltas["support_count_delta"] == 0
    assert deltas["viable_tuple_mass_delta"] == -6.0
    assert deltas["mean_logdet_H_delta"] == -2.5
    assert deltas["ambiguity_risk_delta"] == 0.15
    assert deltas["dense_worsened_count_delta"] == 4
    assert deltas["dense_median_te_cm_delta"] == 2.5
    assert deltas["dense_r5_delta"] == -0.06
    assert report["verdict"]["support_count_not_sufficient"] is True
    assert "support_preserved_but_solver_degraded" in report["verdict"]["flags"]
    assert "pose_degraded_with_solver_drop" in report["verdict"]["flags"]
    assert report["paper_safety"]["split_audit_status"] == "passed"
    assert report["paper_safety"]["query_group_mode"] == "query_id"


def test_support_failure_mechanism_cli_writes_json_and_markdown(tmp_path):
    solver_path = tmp_path / "solver.json"
    baseline_metrics = tmp_path / "baseline_metrics.json"
    candidate_metrics = tmp_path / "candidate_metrics.json"
    baseline_transition = tmp_path / "baseline_transition.json"
    candidate_transition = tmp_path / "candidate_transition.json"
    output_json = tmp_path / "mechanism.json"
    output_md = tmp_path / "mechanism.md"
    solver_path.write_text(json.dumps(_solver_report()), encoding="utf-8")
    baseline_metrics.write_text(json.dumps({"dense": {"median_te_cm": 9.0, "recall_5cm_5d": 0.4}}), encoding="utf-8")
    candidate_metrics.write_text(json.dumps({"dense": {"median_te_cm": 12.0, "recall_5cm_5d": 0.3}}), encoding="utf-8")
    baseline_transition.write_text(json.dumps({"summary": {"worsened_count": 2}}), encoding="utf-8")
    candidate_transition.write_text(json.dumps({"summary": {"worsened_count": 5}}), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.summarize_support_failure_mechanism",
            "--solver_report",
            str(solver_path),
            "--baseline_metrics",
            str(baseline_metrics),
            "--candidate_metrics",
            str(candidate_metrics),
            "--baseline_transition",
            str(baseline_transition),
            "--candidate_transition",
            str(candidate_transition),
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
        ],
        check=True,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    markdown = output_md.read_text(encoding="utf-8")
    assert report["verdict"]["support_count_not_sufficient"] is True
    assert "| support_count_delta |" in markdown
    assert "support-count preservation is insufficient" in markdown


def test_support_failure_mechanism_accepts_lsf_eval_report_shape():
    report = build_support_failure_mechanism_report(
        _solver_report(),
        baseline_metrics={
            "pose": {"dense": {"median_te_cm": 8.0, "recall_5cm_5deg": 0.5, "recall_2cm_2deg": 0.2}},
            "dense_transition": {"worsened_count": 2, "lost_r5_count": 1},
        },
        candidate_metrics={
            "pose": {"dense": {"median_te_cm": 9.0, "recall_5cm_5deg": 0.45, "recall_2cm_2deg": 0.18}},
            "dense_transition": {"worsened_count": 5, "lost_r5_count": 3},
        },
        baseline_transition={
            "dense_transition": {"worsened_count": 2, "lost_r5_count": 1, "lost_r2_count": 0}
        },
        candidate_transition={
            "dense_transition": {"worsened_count": 5, "lost_r5_count": 3, "lost_r2_count": 1}
        },
    )

    deltas = report["mechanism_table"]["deltas"]
    assert deltas["dense_median_te_cm_delta"] == 1.0
    assert deltas["dense_r5_delta"] == -0.05
    assert deltas["dense_worsened_count_delta"] == 3
    assert deltas["dense_r5_lost_count_delta"] == 2
    assert deltas["dense_r2_lost_count_delta"] == 1
