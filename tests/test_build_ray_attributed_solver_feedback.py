import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _write_observations(path: Path, *, split_name: str = "selfmap_train") -> None:
    rows = [
        {"type": "manifest", "manifest": {"scene": "ToyScene", "split_name": split_name}},
        {
            "type": "observation",
            "observation": {
                "query_id": "q1",
                "source_view_id": "s1",
                "split_name": split_name,
                "pnp_inlier": True,
                "reprojection_error_px": 1.0,
                "descriptor_score": 0.8,
                "contributors": [
                    {"gaussian_id": 0, "contribution": 0.75, "xyz": [0.0, 0.0, 1.0]},
                    {"gaussian_id": 1, "contribution": 0.25, "xyz": [1.0, 0.0, 1.0]},
                ],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_ray_attributed_solver_feedback_cli_writes_audited_artifact(tmp_path):
    observations = tmp_path / "observations.jsonl"
    output_dir = tmp_path / "ray_feedback"
    _write_observations(observations)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ray_attributed_solver_feedback",
            "--observations",
            str(observations),
            "--output_dir",
            str(output_dir),
            "--scene",
            "ToyScene",
            "--split_name",
            "selfmap_train",
            "--num_gaussians",
            "3",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    payload = json.loads(completed.stdout)
    artifact_path = output_dir / "ray_solver_feedback.pt"
    assert payload["artifact_path"] == str(artifact_path)
    assert artifact_path.exists()
    artifact = torch.load(artifact_path, map_location="cpu")
    assert artifact["schema_version"] == "ray_attributed_solver_feedback_v1"
    assert artifact["support_score"][0] > artifact["support_score"][1] > 0.0
    assert artifact["per_query_support"]["q1"][0] > artifact["per_query_support"]["q1"][1] > 0.0

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))
    assert manifest["scene"] == "ToyScene"
    assert manifest["split_name"] == "selfmap_train"
    assert metrics["positive_observed_landmarks"] == 2
    assert metrics["per_query_support_query_count"] == 1
    assert metrics["per_query_support_entry_count"] == 2
    assert split_audit["audit_status"] == "passed"
    assert (output_dir / "command.txt").exists()
    assert (output_dir / "git_status.txt").exists()


def test_build_ray_attributed_solver_feedback_cli_rejects_test_split(tmp_path):
    observations = tmp_path / "observations.jsonl"
    _write_observations(observations, split_name="test")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ray_attributed_solver_feedback",
            "--observations",
            str(observations),
            "--output_dir",
            str(tmp_path / "ray_feedback"),
            "--scene",
            "ToyScene",
            "--split_name",
            "test",
            "--num_gaussians",
            "3",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode != 0
    assert "test split" in completed.stderr
