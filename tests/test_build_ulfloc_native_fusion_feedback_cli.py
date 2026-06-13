from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import torch


def _write_feedback(path: Path, *, split_name: str = "selfmap_train") -> None:
    torch.save(
        {
            "schema_version": "sparse_solver_feedback_v4",
            "split_name": split_name,
            "split_audit": {
                "split_name": split_name,
                "test_split_used": split_name == "test",
                "official_test_used": False,
            },
            "correspondences": [
                {
                    "scene": "ToyScene",
                    "split_name": split_name,
                    "query_id": "seq/frame0001.png",
                    "image_id": "seq/frame0001.png",
                    "gaussian_id": 1,
                    "sampled_row": 1,
                    "label_role": "protected_support",
                    "descriptor_score": 0.9,
                    "descriptor_margin": 0.2,
                    "reprojection_error_px": 1.0,
                },
                {
                    "scene": "ToyScene",
                    "split_name": split_name,
                    "query_id": "seq/frame0001.png",
                    "image_id": "seq/frame0001.png",
                    "gaussian_id": 2,
                    "sampled_row": 2,
                    "label_role": "harmful_negative",
                    "descriptor_score": 0.9,
                    "descriptor_margin": 0.1,
                    "reprojection_error_px": 20.0,
                },
            ],
        },
        path,
    )


def test_build_ulfloc_native_fusion_feedback_cli_writes_ulfloc_payload_and_audits(tmp_path: Path) -> None:
    feedback = tmp_path / "feedback_v4.pt"
    output = tmp_path / "native_fusion_feedback"
    _write_feedback(feedback)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_native_fusion_feedback",
            "--feedback_v4",
            str(feedback),
            "--output_dir",
            str(output),
            "--scene",
            "ShopFacade",
            "--num_gaussians",
            "5",
            "--alpha",
            "0.5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    with (output / "solver_feedback.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics_summary.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output / "split_audit.json").read_text(encoding="utf-8"))

    weights = torch.as_tensor(payload["landmark_weights"], dtype=torch.float32)
    assert payload["schema_version"] == "ulfloc_solver_feedback_v1"
    assert weights[1] > 1.0
    assert weights[2] < 1.0
    assert "seq/frame0001.png" in payload["view_landmark_weights"]
    assert manifest["role"] == "native_train_view_feature_fusion_feedback_export"
    assert metrics["positive_count"] == 1
    assert metrics["negative_count"] == 1
    assert split_audit["test_split_used"] is False
    assert (output / "command.txt").exists()
    assert (output / "git_status.txt").exists()


def test_build_ulfloc_native_fusion_feedback_cli_can_infer_num_gaussians_from_model_path(tmp_path: Path) -> None:
    feedback = tmp_path / "feedback_v4.pt"
    output = tmp_path / "native_fusion_feedback"
    model_path = tmp_path / "model"
    ply_dir = model_path / "point_cloud" / "iteration_30000"
    ply_dir.mkdir(parents=True)
    (ply_dir / "point_cloud.ply").write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 5",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_feedback(feedback)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_native_fusion_feedback",
            "--feedback_v4",
            str(feedback),
            "--output_dir",
            str(output),
            "--scene",
            "ShopFacade",
            "--model_path",
            str(model_path),
            "--fusion_policy",
            "view_pruning",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["num_landmarks"] == 5
    assert metrics["fusion_policy"] == "view_pruning"
    assert metrics["weight_application"] == "absolute"


def test_build_ulfloc_native_fusion_feedback_cli_rejects_test_split(tmp_path: Path) -> None:
    feedback = tmp_path / "feedback_v4.pt"
    _write_feedback(feedback, split_name="test")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_native_fusion_feedback",
            "--feedback_v4",
            str(feedback),
            "--output_dir",
            str(tmp_path / "native_fusion_feedback"),
            "--scene",
            "ShopFacade",
            "--num_gaussians",
            "5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split" in result.stderr
