from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import numpy as np


def _write_feedback(path: Path, *, split_name: str = "train_selfmap") -> None:
    payload = {
        "split_name": split_name,
        "scene": "ToyScene",
        "records": [
            {
                "query_id": "q1",
                "image_id": "im1",
                "gaussian_id": 10,
                "label": 1,
                "pnp_inlier": True,
                "pose_success": True,
                "reprojection_error_px": 1.0,
                "descriptor_margin": 0.4,
                "keypoint_xy": [20.0, 30.0],
                "source_role": "baseline_trace",
            },
            {
                "query_id": "q1",
                "image_id": "im1",
                "gaussian_id": 11,
                "label": 0,
                "pnp_inlier": False,
                "pose_success": False,
                "reprojection_error_px": 24.0,
                "descriptor_score": 0.92,
                "descriptor_margin": 0.02,
                "query_regression_delta_cm": 35.0,
                "keypoint_xy": [80.0, 90.0],
                "source_role": "candidate_trace",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_solver_feedback_impact_cli_writes_outputs_and_json_safe_keys(tmp_path: Path) -> None:
    feedback = tmp_path / "feedback.json"
    output_dir = tmp_path / "impact"
    _write_feedback(feedback)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_solver_feedback_impact",
            "--feedback",
            str(feedback),
            "--output_dir",
            str(output_dir),
        ],
        check=True,
    )

    impact_json = json.loads((output_dir / "impact_attribution.json").read_text(encoding="utf-8"))
    impact_pt = torch.load(output_dir / "impact_attribution.pt", map_location="cpu")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))

    assert impact_json["landmark_positive"]["10"]["support"] > 0.0
    assert impact_json["landmark_negative"]["11"]["risk"] > 0.0
    assert impact_json["view_positive"]["10::im1"]["weight"] > 0.0
    assert impact_json["view_negative"]["11::im1"]["weight"] > 0.0
    assert impact_pt["view_positive"][("10", "im1")]["weight"] > 0.0
    assert manifest["schema_version"] == "solver_feedback_impact_manifest_v1"
    assert manifest["split_name"] == "train_selfmap"
    assert manifest["test_split_used"] is False
    assert metrics["positive_landmark_count"] == 1
    assert metrics["negative_landmark_count"] == 1
    assert split_audit["split_name"] == "train_selfmap"
    assert split_audit["test_split_used"] is False
    assert split_audit["official_test_used"] is False
    assert (output_dir / "command.txt").exists()
    assert (output_dir / "git_status.txt").exists()


def test_build_solver_feedback_impact_cli_rejects_test_split(tmp_path: Path) -> None:
    feedback = tmp_path / "feedback.json"
    _write_feedback(feedback, split_name="test")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_solver_feedback_impact",
            "--feedback",
            str(feedback),
            "--output_dir",
            str(tmp_path / "impact"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split" in result.stderr


def test_build_solver_feedback_impact_cli_rejects_audited_test_split(tmp_path: Path) -> None:
    feedback = tmp_path / "feedback.json"
    payload = {
        "split_name": "train_selfmap",
        "split_audit": {"test_split_used": True, "official_test_used": False},
        "records": [],
    }
    feedback.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_solver_feedback_impact",
            "--feedback",
            str(feedback),
            "--output_dir",
            str(tmp_path / "impact"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split" in result.stderr


def test_build_solver_feedback_impact_cli_serializes_numpy_query_features(tmp_path: Path) -> None:
    feedback = tmp_path / "feedback.pt"
    output_dir = tmp_path / "impact"
    torch.save(
        {
            "split_name": "train_selfmap",
            "query_features": {"q1": np.array([0.1, 0.2], dtype=np.float32)},
            "correspondences": [
                {
                    "query_id": "q1",
                    "image_id": "im1",
                    "gaussian_id": 10,
                    "label_role": "protected_support",
                    "pnp_inlier": True,
                    "pose_success": True,
                    "reprojection_error_px": 1.0,
                    "descriptor_margin": 0.4,
                    "source_role": "baseline_trace",
                }
            ],
            "split_audit": {"split_name": "train_selfmap", "test_split_used": False, "official_test_used": False},
        },
        feedback,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_solver_feedback_impact",
            "--feedback",
            str(feedback),
            "--output_dir",
            str(output_dir),
        ],
        check=True,
    )

    impact_json = json.loads((output_dir / "impact_attribution.json").read_text(encoding="utf-8"))
    assert impact_json["query_features"]["q1"] == [0.10000000149011612, 0.20000000298023224]
