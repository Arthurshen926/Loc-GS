import json
import subprocess
import sys

from loc_gs.eval.query_partitions import partitioned_stage_deltas


def test_partitioned_stage_deltas_reports_hard_tail_and_easy_subset():
    baseline = [
        {"query_id": "q0", "dense_te_cm": 100.0, "dense_re_deg": 1.0},
        {"query_id": "q1", "dense_te_cm": 50.0, "dense_re_deg": 1.0},
        {"query_id": "q2", "dense_te_cm": 10.0, "dense_re_deg": 1.0},
        {"query_id": "q3", "dense_te_cm": 2.0, "dense_re_deg": 1.0},
    ]
    candidate = [
        {"query_id": "q0", "dense_te_cm": 70.0, "dense_re_deg": 1.0},
        {"query_id": "q1", "dense_te_cm": 55.0, "dense_re_deg": 1.0},
        {"query_id": "q2", "dense_te_cm": 8.0, "dense_re_deg": 1.0},
        {"query_id": "q3", "dense_te_cm": 3.0, "dense_re_deg": 1.0},
    ]

    report = partitioned_stage_deltas(baseline, candidate, stage="dense", fractions=(0.25, 0.5))

    assert report["all"]["mean_delta_te_cm"] == -6.5
    assert report["hardest_25pct"]["query_count"] == 1
    assert report["hardest_25pct"]["mean_delta_te_cm"] == -30.0
    assert report["hardest_50pct"]["mean_delta_te_cm"] == -12.5
    assert report["easiest_50pct"]["mean_delta_te_cm"] == -0.5
    assert report["all"]["recall_5cm_5deg_delta"] == 0.0


def test_summarize_query_partitions_cli_writes_json_and_markdown(tmp_path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    (baseline_dir / "results.json").write_text(
        json.dumps(
            [
                {"dense_TE": 100.0, "dense_AE": 1.0, "sparse_TE": 120.0, "sparse_AE": 1.0},
                {"dense_TE": 4.0, "dense_AE": 1.0, "sparse_TE": 6.0, "sparse_AE": 1.0},
            ]
        ),
        encoding="utf-8",
    )
    (candidate_dir / "results.json").write_text(
        json.dumps(
            [
                {"dense_TE": 80.0, "dense_AE": 1.0, "sparse_TE": 90.0, "sparse_AE": 1.0},
                {"dense_TE": 3.0, "dense_AE": 1.0, "sparse_TE": 5.0, "sparse_AE": 1.0},
            ]
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.summarize_query_partitions",
            "--baseline_run",
            str(baseline_dir),
            "--candidate_run",
            str(candidate_dir),
            "--candidate_name",
            "toy",
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
        ],
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["candidate"] == "toy"
    assert payload["partitions"]["hardest_50pct"]["mean_delta_te_cm"] == -20.0
    markdown = output_md.read_text(encoding="utf-8")
    assert "| toy | dense | all | 2 | -10.5000 |" in markdown
