import json
import subprocess
import sys

import torch


def _target_payload(targets):
    return {
        "schema_version": "ulfloc_detector_targets_from_solver_feedback_v1",
        "split_name": "train_dev",
        "targets": targets,
        "metadata": {"height": 20, "width": 30},
        "split_audit": {"audit_status": "passed", "split_name": "train_dev", "test_split_used": False},
    }


def test_audit_ulfloc_detector_targets_cli_writes_metrics_and_rows(tmp_path):
    baseline = tmp_path / "baseline.pt"
    candidate = tmp_path / "candidate.pt"
    baseline_results = tmp_path / "baseline_results.json"
    candidate_results = tmp_path / "candidate_results.json"
    output_dir = tmp_path / "audit"
    torch.save(
        _target_payload(
            {
                "q.png": {
                    "keypoint_yx": torch.tensor([[2.0, 2.0], [12.0, 20.0]]),
                    "support_weights": torch.tensor([1.0, 2.0]),
                }
            }
        ),
        baseline,
    )
    torch.save(
        _target_payload(
            {
                "q.png": {
                    "keypoint_yx": torch.tensor([[4.0, 4.0]]),
                    "support_weights": torch.tensor([3.0]),
                    "negative_keypoint_yx": torch.tensor([[10.0, 10.0]]),
                    "negative_weights": torch.tensor([1.0]),
                }
            }
        ),
        candidate,
    )
    baseline_results.write_text(json.dumps({"rows": [{"image_name": "q.png", "sparse_te_cm": 2.0}]}))
    candidate_results.write_text(json.dumps({"rows": [{"image_name": "q.png", "sparse_te_cm": 5.5}]}))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.audit_ulfloc_detector_targets",
            "--baseline_detector_targets",
            str(baseline),
            "--candidate_detector_targets",
            str(candidate),
            "--baseline_results",
            str(baseline_results),
            "--candidate_results",
            str(candidate_results),
            "--output_dir",
            str(output_dir),
            "--height",
            "20",
            "--width",
            "30",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics_summary.json").read_text())
    rows = json.loads((output_dir / "target_audit_rows.json").read_text())
    assert metrics["common_query_count"] == 1
    assert metrics["mean_sparse_te_delta_cm"] == 3.5
    assert rows[0]["candidate_negative_count"] == 1
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "command.txt").exists()


def test_audit_ulfloc_detector_targets_cli_rejects_test_targets(tmp_path):
    path = tmp_path / "test_targets.pt"
    torch.save(
        {
            "split_name": "test",
            "targets": {},
            "split_audit": {"split_name": "test", "test_split_used": True},
        },
        path,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.audit_ulfloc_detector_targets",
            "--baseline_detector_targets",
            str(path),
            "--candidate_detector_targets",
            str(path),
            "--output_dir",
            str(tmp_path / "out"),
            "--height",
            "20",
            "--width",
            "30",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split" in result.stderr
