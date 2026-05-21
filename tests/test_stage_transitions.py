import json
import subprocess
import sys

from loc_gs.eval.stage_transitions import dense_transition_labels, stage_transition_summary


def test_stage_transition_summary_counts_improved_worsened_and_threshold_crossings():
    rows = [
        {"query_id": "rescued", "sparse_TE": 6.0, "sparse_AE": 1.0, "dense_TE": 4.0, "dense_AE": 1.0},
        {"query_id": "lost", "sparse_TE": 4.0, "sparse_AE": 1.0, "dense_TE": 6.0, "dense_AE": 1.0},
        {"query_id": "same", "sparse_TE": 2.0, "sparse_AE": 1.0, "dense_TE": 2.0, "dense_AE": 1.0},
    ]

    report = stage_transition_summary(rows, te_epsilon_cm=0.05)

    assert report["query_count"] == 3
    assert report["improved_count"] == 1
    assert report["worsened_count"] == 1
    assert report["unchanged_count"] == 1
    assert report["mean_dense_minus_sparse_te_cm"] == 0.0
    assert report["r5_rescued_count"] == 1
    assert report["r5_lost_count"] == 1
    assert report["r2_rescued_count"] == 0
    assert report["r2_lost_count"] == 0
    assert report["worst_dense_worsened_queries"][0]["query_id"] == "lost"


def test_dense_transition_labels_mark_pose_and_recall_transitions():
    rows = [
        {"query_id": "rescued", "sparse_TE": 6.0, "sparse_AE": 1.0, "dense_TE": 4.0, "dense_AE": 1.0},
        {"query_id": "lost", "sparse_TE": 4.0, "sparse_AE": 1.0, "dense_TE": 6.0, "dense_AE": 1.0},
        {"query_id": "tail", "sparse_TE": 100.0, "sparse_AE": 1.0, "dense_TE": 25.0, "dense_AE": 1.0},
        {"query_id": "same", "sparse_TE": 2.0, "sparse_AE": 1.0, "dense_TE": 2.0, "dense_AE": 1.0},
    ]

    labels = dense_transition_labels(rows, te_epsilon_cm=0.05, hard_tail_te_cm=50.0)

    by_id = {item["query_id"]: item for item in labels}
    assert by_id["rescued"]["pose_transition"] == "improved"
    assert by_id["rescued"]["r5_transition"] == "rescued"
    assert by_id["lost"]["pose_transition"] == "worsened"
    assert by_id["lost"]["r5_transition"] == "lost"
    assert by_id["tail"]["hard_tail_before_dense"] is True
    assert by_id["tail"]["pose_transition"] == "improved"
    assert by_id["same"]["pose_transition"] == "unchanged"
    assert by_id["same"]["r2_transition"] == "stable_ok"


def test_summarize_stage_transitions_cli_writes_json_and_markdown(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.json").write_text(
        json.dumps(
            [
                {"sparse_TE": 6.0, "sparse_AE": 1.0, "dense_TE": 4.0, "dense_AE": 1.0},
                {"sparse_TE": 4.0, "sparse_AE": 1.0, "dense_TE": 6.0, "dense_AE": 1.0},
            ]
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "transition.json"
    output_md = tmp_path / "transition.md"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.summarize_stage_transitions",
            "--run_dir",
            str(run_dir),
            "--run_name",
            "toy",
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
        ],
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["run_name"] == "toy"
    assert payload["summary"]["r5_rescued_count"] == 1
    assert payload["summary"]["r5_lost_count"] == 1
    markdown = output_md.read_text(encoding="utf-8")
    assert "| toy | 2 | 1 | 1 | 1 | 1 |" in markdown


def test_summarize_stage_transitions_cli_writes_dense_transition_labels(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.json").write_text(
        json.dumps(
            [
                {"query_id": "q0", "sparse_TE": 6.0, "sparse_AE": 1.0, "dense_TE": 4.0, "dense_AE": 1.0},
                {"query_id": "q1", "sparse_TE": 4.0, "sparse_AE": 1.0, "dense_TE": 6.0, "dense_AE": 1.0},
            ]
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "transition.json"
    output_labels = tmp_path / "labels.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.summarize_stage_transitions",
            "--run_dir",
            str(run_dir),
            "--output_json",
            str(output_json),
            "--output_labels_json",
            str(output_labels),
        ],
        check=True,
    )

    labels = json.loads(output_labels.read_text(encoding="utf-8"))
    assert labels["label_type"] == "dense_transition_labels_v1"
    assert labels["labels"][0]["query_id"] == "q0"
    assert labels["labels"][0]["r5_transition"] == "rescued"
    assert labels["labels"][1]["r5_transition"] == "lost"
