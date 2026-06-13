from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import numpy as np


def _trace_payload(*, split_name: str = "selfmap_train") -> dict:
    return {
        "schema_version": "ulfloc_sparse_feedback_trace_v2",
        "scene": "ToyScene",
        "split_name": split_name,
        "queries": [
            {
                "scene": "ToyScene",
                "split_name": split_name,
                "query_id": "q1",
                "image_id": "im1",
                "pose_success": True,
                "sparse_te_cm": 3.0,
                "sparse_re_deg": 0.1,
                "catastrophic_failure": False,
                "inlier_count": 1,
                "match_count": 1,
                "inlier_image_cell_count": 1,
                "inlier_depth_bin_count": 1,
                "bearing_spread": 0.0,
                "depth_spread": 0.0,
            }
        ],
        "correspondences": [
            {
                "scene": "ToyScene",
                "split_name": split_name,
                "query_id": "q1",
                "image_id": "im1",
                "gaussian_id": 10,
                "sampled_row": 0,
                "query_keypoint_index": 3,
                "keypoint_xy": [5.0, 6.0],
                "query_xy_norm": [0.05, 0.12],
                "image_cell": 0,
                "query_descriptor": [0.25, 0.75],
                "landmark_descriptor": [1.0, 0.0],
                "descriptor_score": 0.9,
                "descriptor_margin": 0.2,
                "detector_score": 0.7,
                "pnp_inlier": True,
                "reprojection_error_px": 1.0,
                "camera_xyz": [0.1, 0.2, 2.0],
                "depth_m": 2.0,
                "bearing": [0.0, 0.0, 1.0],
                "source_role": "baseline_trace",
            }
        ],
        "split_audit": {"test_split_used": split_name == "test", "official_test_used": False},
    }


def test_build_sparse_feedback_v4_cli_writes_audited_outputs(tmp_path: Path) -> None:
    trace = tmp_path / "trace.pt"
    output_dir = tmp_path / "feedback_v4"
    torch.save(_trace_payload(), trace)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_sparse_feedback_v4",
            "--trace",
            str(trace),
            "--output_dir",
            str(output_dir),
        ],
        check=True,
    )

    feedback = torch.load(output_dir / "feedback_v4.pt", map_location="cpu", weights_only=False)
    label_only = torch.load(output_dir / "feedback_v4_label_only.pt", map_location="cpu", weights_only=False)
    query_tokens = torch.load(output_dir / "query_tokens.pt", map_location="cpu", weights_only=False)
    feedback_json = json.loads((output_dir / "feedback_v4.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))

    assert feedback["schema_version"] == "sparse_solver_feedback_v4"
    assert label_only["schema_version"] == "sparse_solver_feedback_v4"
    assert label_only["correspondences"][0].get("query_descriptor") is None
    assert label_only["correspondences"][0].get("landmark_descriptor") is None
    assert query_tokens["schema_version"] == "sparse_solver_feedback_query_tokens_v1"
    assert query_tokens["split_name"] == "selfmap_train"
    assert query_tokens["query_tokens"]["q1"].shape == (1, 5)
    assert query_tokens["query_token_source_role"] == "per_keypoint_runtime_schema"
    assert feedback["metrics"]["protected_support_count"] == 1
    assert feedback_json["metrics"]["correspondence_count"] == 1
    assert feedback_json["correspondences"][0].get("query_descriptor") is None
    assert feedback_json["correspondences"][0].get("landmark_descriptor") is None
    assert feedback_json["query_descriptor_dim"] == 2
    assert feedback_json["landmark_descriptor_dim"] == 2
    assert feedback_json["correspondence_descriptors_omitted_from_json"] is True
    assert metrics["protected_support_count"] == 1
    assert manifest["schema_version"] == "sparse_feedback_v4_manifest_v1"
    assert manifest["split_name"] == "selfmap_train"
    assert manifest["trace"] == str(trace.resolve())
    assert manifest["outputs"]["feedback_v4_label_only_pt"] == str((output_dir / "feedback_v4_label_only.pt").resolve())
    assert manifest["outputs"]["query_tokens_pt"] == str((output_dir / "query_tokens.pt").resolve())
    assert manifest["hyperparameters"]["risky_competitor_max_margin"] == 0.05
    assert split_audit["test_split_used"] is False
    assert (output_dir / "command.txt").exists()
    assert (output_dir / "git_status.txt").exists()


def test_build_sparse_feedback_v4_cli_loads_json_trace(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    output_dir = tmp_path / "feedback_v4"
    trace.write_text(json.dumps(_trace_payload()), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_sparse_feedback_v4",
            "--trace",
            str(trace),
            "--output_dir",
            str(output_dir),
        ],
        check=True,
    )

    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["query_count"] == 1


def test_build_sparse_feedback_v4_cli_limits_json_correspondence_dump(tmp_path: Path) -> None:
    payload = _trace_payload()
    payload["correspondences"] = payload["correspondences"] * 3
    trace = tmp_path / "trace.pt"
    output_dir = tmp_path / "feedback_v4"
    torch.save(payload, trace)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_sparse_feedback_v4",
            "--trace",
            str(trace),
            "--output_dir",
            str(output_dir),
            "--max_json_correspondences",
            "1",
        ],
        check=True,
    )

    feedback = torch.load(output_dir / "feedback_v4.pt", map_location="cpu", weights_only=False)
    feedback_json = json.loads((output_dir / "feedback_v4.json").read_text(encoding="utf-8"))

    assert len(feedback["correspondences"]) == 3
    assert len(feedback_json["correspondences"]) == 1
    assert feedback_json["json_correspondence_count"] == 1
    assert feedback_json["json_correspondences_truncated"] is True


def test_build_sparse_feedback_v4_cli_serializes_numpy_query_features(tmp_path: Path) -> None:
    payload = _trace_payload()
    payload["query_features"] = {"q1": np.array([0.1, 0.2], dtype=np.float32)}
    trace = tmp_path / "trace.pt"
    output_dir = tmp_path / "feedback_v4"
    torch.save(payload, trace)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_sparse_feedback_v4",
            "--trace",
            str(trace),
            "--output_dir",
            str(output_dir),
        ],
        check=True,
    )

    feedback_json = json.loads((output_dir / "feedback_v4.json").read_text(encoding="utf-8"))
    assert feedback_json["query_features"]["q1"] == [0.10000000149011612, 0.20000000298023224]


def test_build_sparse_feedback_v4_cli_rejects_test_split(tmp_path: Path) -> None:
    trace = tmp_path / "trace.pt"
    torch.save(_trace_payload(split_name="test"), trace)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_sparse_feedback_v4",
            "--trace",
            str(trace),
            "--output_dir",
            str(tmp_path / "feedback_v4"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split" in result.stderr
